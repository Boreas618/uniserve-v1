//! The device time of registered streams, from CUPTI's activity records.
//!
//! [`StreamActivity::start`] enables CUPTI's records of kernels
//! (`CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL`), copies (`MEMCPY`, and
//! `MEMCPY2` for peer copies) and fills (`MEMSET`), and registers the
//! streams to account by name. A graph launch is recorded by its nodes,
//! under the stream it was launched on: CUPTI's graph-level records
//! (`GRAPH_TRACE`) make `cuGraphLaunch` of a graph instantiated before they
//! were enabled crash the process (observed with CUPTI 13.0), and the
//! worker instantiates its graphs before it serves. The device timestamps
//! each operation's start and end, and CUPTI collects the records in
//! buffers. CUPTI's own periodic flush delivers only full buffers, so the
//! recording's flush thread asks CUPTI every [`FLUSH_PERIOD`] to deliver
//! every buffer whose records completed (`cuptiActivityFlushAll`, which
//! issues no CUDA synchronization); the delivery callback merges each
//! registered stream's operations into the union of their intervals.
//! Records of other streams are ignored.
//!
//! A stream's `busy` time is the length of that union: operations that
//! overlap (concurrent nodes of one graph) count once. Its `gap` time is the
//! rest of the span from its first operation to its latest end: the time
//! the stream had nothing to run,
//! whether it waited for the host, for another stream's event or for any
//! work at all. Idle time after the latest end is counted once the next
//! operation arrives.
//!
//! CUPTI does not deliver records in device order: a buffer is delivered
//! only once every record in it completed, so a later buffer can arrive
//! first. The union therefore keeps the last [`RETAINED_NS`] of device time
//! behind the stream's latest end, and a record arriving within that much of
//! the records delivered before it merges exactly. The part of a record that
//! starts before the retained time cannot be merged; it is counted as the
//! stream's `late` time, neither busy nor withdrawn from gap. A record that
//! fills a gap after it was reported lowers the stream's gap below what was
//! reported; the reported gap never decreases, and grows again once the
//! stream's gap exceeds it.
//!
//! [`take`] reads the time accumulated since the last take from atomics the
//! delivery callback publishes, so it never waits for the callback.

use std::alloc::Layout;
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use super::{check, cupti};
use crate::{Error, Result, Stream, lock};

/// How often the flush thread has CUPTI deliver the buffers whose records
/// completed.
pub const FLUSH_PERIOD: Duration = Duration::from_millis(100);

/// How much device time behind a stream's latest end its union keeps for
/// records that arrive out of order, in nanoseconds.
pub const RETAINED_NS: u64 = 2_000_000_000;

/// Each activity buffer CUPTI fills: 1 MiB (about 4,800 kernel records),
/// aligned as CUPTI requires. While a stream runs without pause, the buffer
/// it fills always holds the record of an operation in flight, and a flush
/// delivers it only once it is full; a small buffer bounds how long its
/// operations wait to be counted.
const BUFFER: Layout = match Layout::from_size_align(1 << 20, 8) {
    Ok(layout) => layout,
    Err(_) => panic!("the activity buffer layout is invalid"),
};

/// `CUPTI_ACTIVITY_FLAG_FLUSH_FORCED`.
const FLUSH_FORCED: u32 = 1;

/// Where a record keeps its interval and its stream, in bytes from the
/// record's start (the `CUpti_Activity*` structs of the CUPTI headers the
/// loaded `libcupti.so` ships with).
#[derive(Clone, Copy)]
struct Fields {
    start: usize,
    end: usize,
    context: usize,
    stream: usize,
}

/// `CUpti_ActivityKernel10`.
const KERNEL: Fields = Fields {
    start: 16,
    end: 24,
    context: 44,
    stream: 48,
};

/// `CUpti_ActivityMemcpy6`, `CUpti_ActivityMemcpyPtoP4` and
/// `CUpti_ActivityMemset4`, which share these offsets.
const TRANSFER: Fields = Fields {
    start: 16,
    end: 24,
    context: 36,
    stream: 40,
};

/// The enabled `CUpti_ActivityKind`s and their records' fields.
const KINDS: &[(u32, Fields)] = &[
    // CONCURRENT_KERNEL
    (10, KERNEL),
    // MEMCPY
    (1, TRANSFER),
    // MEMCPY2
    (22, TRANSFER),
    // MEMSET
    (2, TRANSFER),
];

pub(super) type BufferRequested = unsafe extern "C" fn(*mut *mut u8, *mut usize, *mut usize);
pub(super) type BufferCompleted = unsafe extern "C" fn(*mut c_void, u32, *mut u8, usize, usize);

/// The time of one stream that the delivery callback publishes, in whole
/// microseconds since the stream was registered, and what [`take`] took of
/// it.
#[derive(Default)]
struct Counters {
    busy_us: AtomicU64,
    gap_us: AtomicU64,
    late_us: AtomicU64,
    taken: [AtomicU64; 3],
}

/// One registered stream's union of operation intervals, which only the
/// delivery callback changes.
struct Timeline {
    counters: Arc<Counters>,
    /// The merged intervals of the retained time, start to end in CUPTI
    /// nanoseconds: disjoint and not adjacent.
    intervals: BTreeMap<u64, u64>,
    /// Intervals that ended before this were dropped; record time before it
    /// is late.
    retained_from: u64,
    /// The span of the stream's operations so far, once it has one.
    span: Option<(u64, u64)>,
    busy_ns: u64,
    late_ns: u64,
    /// The largest gap reported, which the reported gap never falls below.
    gap_ns: u64,
}

impl Timeline {
    fn new(counters: Arc<Counters>) -> Self {
        Self {
            counters,
            intervals: BTreeMap::new(),
            retained_from: 0,
            span: None,
            busy_ns: 0,
            late_ns: 0,
            gap_ns: 0,
        }
    }

    /// Merge one operation's interval `[start, end)` into the union.
    fn add(&mut self, mut start: u64, end: u64) {
        if start < self.retained_from {
            let merged_from = end.min(self.retained_from);
            self.late_ns += merged_from - start;
            start = merged_from;
        }
        if end <= start {
            return;
        }

        self.span = Some(match self.span {
            Some((first, latest)) => (first.min(start), latest.max(end)),
            None => (start, end),
        });

        // Remove every stored interval that overlaps or touches
        // `[start, end)`, walking back from the last one starting at or
        // before `end`, and store their union with it. Stored intervals are
        // disjoint, so only those overlap the merged interval.
        let (mut merged_start, mut merged_end, mut covered) = (start, end, 0);
        while let Some((&first, &last)) = self.intervals.range(..=end).next_back() {
            if last < start {
                break;
            }
            self.intervals.remove(&first);
            covered += last.min(end).saturating_sub(first.max(start));
            merged_start = merged_start.min(first);
            merged_end = merged_end.max(last);
        }
        self.intervals.insert(merged_start, merged_end);
        self.busy_ns += (end - start).saturating_sub(covered);

        if let Some((first, latest)) = self.span {
            let gap = (latest - first).saturating_sub(self.busy_ns);
            self.gap_ns = self.gap_ns.max(gap);
        }
    }

    /// Drop the intervals that ended more than [`RETAINED_NS`] before the
    /// latest end, and publish the stream's time.
    fn publish(&mut self) {
        if let Some((_, latest)) = self.span {
            let retained_from = latest.saturating_sub(RETAINED_NS);
            if retained_from > self.retained_from {
                let mut kept = self.intervals.split_off(&retained_from);
                // The last dropped interval may reach into the retained time.
                if let Some((first, last)) = self.intervals.pop_last()
                    && last > retained_from
                {
                    kept.insert(first, last);
                }
                self.intervals = kept;
                self.retained_from = retained_from;
            }
        }
        let counters = &self.counters;
        counters
            .busy_us
            .store(self.busy_ns / 1000, Ordering::Release);
        counters.gap_us.store(self.gap_ns / 1000, Ordering::Release);
        counters
            .late_us
            .store(self.late_ns / 1000, Ordering::Release);
    }
}

/// The registered streams' timelines by CUPTI context and stream id, while
/// a [`StreamActivity`] records.
static TIMELINES: Mutex<Option<HashMap<(u32, u32), Timeline>>> = Mutex::new(None);

/// The registered streams' counters by name, which [`take`] reads; they
/// stay readable after the recording stops, until the next start.
static STREAMS: Mutex<Vec<(String, Arc<Counters>)>> = Mutex::new(Vec::new());

/// Records CUPTI dropped for lack of buffer space since the last take.
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Whether a [`StreamActivity`] records.
static RECORDING: AtomicBool = AtomicBool::new(false);

/// The time one stream accumulated since the last [`take`], in whole
/// microseconds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StreamTime {
    /// The name the stream was registered under.
    pub name: String,
    pub busy_us: u64,
    pub gap_us: u64,
    pub late_us: u64,
}

/// What [`take`] returns.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Taken {
    /// Every registered stream, in registration order.
    pub streams: Vec<StreamTime>,
    /// Activity records CUPTI dropped for lack of buffer space: their
    /// operations are missing from every stream's time.
    pub dropped_records: u64,
}

/// The time the registered streams accumulated since the last take. Reads
/// atomics only; the callback that merges records is never waited for.
pub fn take() -> Taken {
    let streams = lock(&STREAMS)
        .iter()
        .map(|(name, counters)| {
            let [busy, gap, late] = &counters.taken;
            StreamTime {
                name: name.clone(),
                busy_us: delta(&counters.busy_us, busy),
                gap_us: delta(&counters.gap_us, gap),
                late_us: delta(&counters.late_us, late),
            }
        })
        .collect();
    Taken {
        streams,
        dropped_records: DROPPED.swap(0, Ordering::AcqRel),
    }
}

/// The growth of the monotonic `total` since `taken`, which it then marks
/// as taken; concurrent takes count each part once.
fn delta(total: &AtomicU64, taken: &AtomicU64) -> u64 {
    let now = total.load(Ordering::Acquire);
    now.saturating_sub(taken.fetch_max(now, Ordering::AcqRel))
}

/// The process's recording of its registered streams' device activity;
/// stops on drop. One records at a time.
pub struct StreamActivity {
    /// The flush thread, which ends once the sender is dropped; `None`
    /// until it started.
    flusher: Option<(Sender<()>, JoinHandle<()>)>,
}

impl StreamActivity {
    /// Record the device activity of `streams`, each under its name, which
    /// [`take`] reports. Fails when `libcupti.so` is missing, CUPTI refuses
    /// a record kind, or another recording is active.
    ///
    /// CUPTI accepts one client: the caller starts the recording only when
    /// no profiler injected CUPTI into the process, as a successful
    /// [`super::BlockingWatch::start`] shows; this recording would replace
    /// the profiler's buffer callbacks.
    pub fn start<'a>(streams: impl IntoIterator<Item = (&'a str, &'a Stream)>) -> Result<Self> {
        let cupti = cupti()?;
        if RECORDING.swap(true, Ordering::AcqRel) {
            return Err(Error::ActivityRecorded);
        }
        // From here the drop disables what was enabled.
        let mut recording = Self { flusher: None };

        *lock(&TIMELINES) = Some(HashMap::new());
        // SAFETY: both callbacks match CUPTI's prototypes and never unwind.
        let status = unsafe { (cupti.register_buffers)(on_buffer_requested, on_buffer_completed) };
        check(status, "cuptiActivityRegisterCallbacks")?;
        for &(kind, _) in KINDS {
            // SAFETY: the kind is a `CUpti_ActivityKind`.
            let status = unsafe { (cupti.activity_enable)(kind) };
            check(status, "cuptiActivityEnable")?;
        }

        let mut registered = Vec::new();
        let mut timelines = HashMap::new();
        for (name, stream) in streams {
            let (mut context, mut id) = (0, 0);
            // SAFETY: the stream and its context are live handles; CUPTI
            // writes one id to each local.
            unsafe {
                check(
                    (cupti.context_id)(stream.context(), &mut context),
                    "cuptiGetContextId",
                )?;
                check(
                    (cupti.stream_id)(stream.context(), stream.handle(), 0, &mut id),
                    "cuptiGetStreamIdEx",
                )?;
            }
            let counters = Arc::new(Counters::default());
            timelines.insert((context, id), Timeline::new(Arc::clone(&counters)));
            registered.push((name.to_owned(), counters));
        }
        *lock(&TIMELINES) = Some(timelines);
        *lock(&STREAMS) = registered;

        let (stop, stopped) = mpsc::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("cupti-flush".into())
            .spawn(move || {
                // The recording's drop disconnects the channel, which ends
                // the loop.
                while let Err(RecvTimeoutError::Timeout) = stopped.recv_timeout(FLUSH_PERIOD) {
                    // SAFETY: the buffer callbacks stay registered while the
                    // recording lives.
                    let _ = unsafe { (cupti.flush_all)(0) };
                }
            })
            .map_err(|source| Error::Thread {
                name: "cupti-flush",
                source: Arc::new(source),
            })?;
        recording.flusher = Some((stop, thread));
        Ok(recording)
    }
}

impl Drop for StreamActivity {
    fn drop(&mut self) {
        if let Some((stop, thread)) = self.flusher.take() {
            drop(stop);
            let _ = thread.join();
        }
        if let Ok(cupti) = cupti() {
            // SAFETY: disabling a kind that was not enabled is harmless; the
            // forced flush delivers the remaining buffers through the
            // callbacks, which are still registered.
            unsafe {
                for &(kind, _) in KINDS {
                    let _ = (cupti.activity_disable)(kind);
                }
                let _ = (cupti.flush_all)(FLUSH_FORCED);
            }
        }
        *lock(&TIMELINES) = None;
        RECORDING.store(false, Ordering::Release);
    }
}

/// CUPTI asks for an empty buffer; a null one makes it drop records.
unsafe extern "C" fn on_buffer_requested(
    buffer: *mut *mut u8,
    size: *mut usize,
    records: *mut usize,
) {
    // SAFETY: the layout has a nonzero size.
    let allocated = unsafe { std::alloc::alloc(BUFFER) };
    // SAFETY: CUPTI passes valid out-pointers; zero records means as many
    // as fit.
    unsafe {
        *buffer = allocated;
        *size = if allocated.is_null() {
            0
        } else {
            BUFFER.size()
        };
        *records = 0;
    }
}

/// CUPTI delivers a buffer from [`on_buffer_requested`]: its records are
/// merged into the registered streams' timelines and the buffer freed.
unsafe extern "C" fn on_buffer_completed(
    context: *mut c_void,
    stream: u32,
    buffer: *mut u8,
    _size: usize,
    valid: usize,
) {
    if buffer.is_null() {
        return;
    }
    if let Ok(cupti) = cupti() {
        if let Some(timelines) = lock(&TIMELINES).as_mut() {
            let mut record = std::ptr::null_mut();
            // SAFETY: CUPTI walks the valid bytes of its own buffer and
            // points `record` at each record, until it reports none left.
            while unsafe { (cupti.next_record)(buffer, valid, &mut record) } == 0 {
                // SAFETY: the record lies in the buffer and starts with its
                // kind; its kind gives its layout.
                unsafe { merge(timelines, record) };
            }
            for timeline in timelines.values_mut() {
                timeline.publish();
            }
        }
        let mut dropped = 0;
        // SAFETY: the context and stream are those CUPTI passed; it writes
        // one count.
        if unsafe { (cupti.dropped_records)(context, stream, &mut dropped) } == 0 {
            DROPPED.fetch_add(dropped as u64, Ordering::AcqRel);
        }
    }
    // SAFETY: the buffer came from `on_buffer_requested` with this layout.
    unsafe { std::alloc::dealloc(buffer, BUFFER) };
}

/// Merge one record into the timeline of its stream, if registered.
///
/// # Safety
/// `record` points at a whole activity record.
unsafe fn merge(timelines: &mut HashMap<(u32, u32), Timeline>, record: *const u8) {
    // SAFETY: every record starts with its `CUpti_ActivityKind`.
    let kind = unsafe { read::<u32>(record, 0) };
    let Some(&(_, fields)) = KINDS.iter().find(|&&(enabled, _)| enabled == kind) else {
        return;
    };
    // SAFETY: the fields lie inside a record of this kind.
    let (start, end, context, stream) = unsafe {
        (
            read::<u64>(record, fields.start),
            read::<u64>(record, fields.end),
            read::<u32>(record, fields.context),
            read::<u32>(record, fields.stream),
        )
    };
    // A record a forced flush delivered before its operation ended has no
    // end; it is not counted.
    if start == 0 || end <= start {
        return;
    }
    if let Some(timeline) = timelines.get_mut(&(context, stream)) {
        timeline.add(start, end);
    }
}

/// # Safety
/// `offset` bytes past `record` lie `size_of::<T>()` readable bytes.
unsafe fn read<T: Copy>(record: *const u8, offset: usize) -> T {
    // SAFETY: forwarded from the caller.
    unsafe { record.add(offset).cast::<T>().read_unaligned() }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn timeline() -> Timeline {
        Timeline::new(Arc::new(Counters::default()))
    }

    #[test]
    fn overlapping_operations_count_once_and_idle_time_counts_as_gap() {
        let mut timeline = timeline();
        // Two concurrent nodes of a graph, one inside the other, then an
        // operation after an idle stream.
        timeline.add(1_000, 5_000);
        timeline.add(2_000, 3_000);
        timeline.add(8_000, 9_000);
        assert_eq!((timeline.busy_ns, timeline.gap_ns), (5_000, 3_000));
    }

    #[test]
    fn an_operation_delivered_after_later_ones_fills_their_gap() {
        let mut timeline = timeline();
        timeline.add(1_000, 2_000);
        timeline.add(6_000, 7_000);
        assert_eq!(timeline.gap_ns, 4_000);

        // The reported gap stays; it grows only beyond what was reported.
        timeline.add(3_000, 5_000);
        assert_eq!((timeline.busy_ns, timeline.gap_ns), (4_000, 4_000));
        timeline.add(10_000, 11_000);
        assert_eq!((timeline.busy_ns, timeline.gap_ns), (5_000, 5_000));
    }

    #[test]
    fn time_before_the_retained_span_counts_as_late() {
        let mut timeline = timeline();
        timeline.add(0, 1_000);
        timeline.add(RETAINED_NS + 5_000, RETAINED_NS + 6_000);
        timeline.publish();

        // The union retains the time from 6 us on.
        timeline.add(2_000, 8_000);
        assert_eq!((timeline.late_ns, timeline.busy_ns), (4_000, 4_000));
    }

    /// The time of the stream registered as `name` that [`take`] reports
    /// until `delivered` holds for its sum, or for 5 s: the flush thread
    /// has the records delivered within its period.
    pub(crate) fn taken_until(name: &str, delivered: impl Fn(&StreamTime) -> bool) -> StreamTime {
        let mut sum = StreamTime {
            name: name.to_owned(),
            ..StreamTime::default()
        };
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !delivered(&sum) && std::time::Instant::now() < deadline {
            std::thread::sleep(FLUSH_PERIOD / 4);
            for stream in take()
                .streams
                .into_iter()
                .filter(|stream| stream.name == name)
            {
                sum.busy_us += stream.busy_us;
                sum.gap_us += stream.gap_us;
                sum.late_us += stream.late_us;
            }
        }
        sum
    }

    mod device {
        use std::time::Duration;

        use libloading::Library;

        use super::super::*;
        use crate::{DeviceAllocation, DeviceId, Event, EventFlags, GraphExec, memset_d32};

        const FILL_BYTES: usize = 1 << 30;

        /// Queue `count` fills of `memory` on `stream` between two timing
        /// events, and return the events.
        fn fills(memory: &DeviceAllocation, stream: &Stream, count: usize) -> Result<[Event; 2]> {
            let flags = EventFlags {
                timing: true,
                ..EventFlags::default()
            };
            let events = [Event::new(flags), Event::new(flags)];
            events[0].record(stream)?;
            fill(memory, stream, count)?;
            events[1].record(stream)?;
            Ok(events)
        }

        fn fill(memory: &DeviceAllocation, stream: &Stream, count: usize) -> Result<()> {
            for value in 0..count {
                // SAFETY: the allocation holds `FILL_BYTES` and outlives the
                // stream's work, which the test synchronizes.
                unsafe { memset_d32(memory.address(), value as u32, FILL_BYTES / 4, stream)? };
            }
            Ok(())
        }

        fn elapsed_us([begin, end]: &[Event; 2]) -> Result<u64> {
            Ok((f64::from(begin.elapsed_ms(end)?) * 1000.0) as u64)
        }

        /// Capture `count` fills of `memory` on `stream` into a graph.
        fn captured(memory: &DeviceAllocation, stream: &Stream, count: usize) -> Result<GraphExec> {
            type Capture = unsafe extern "C" fn(*mut c_void, i32) -> i32;
            type Captured = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32;
            type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;
            // SAFETY: loading the driver runs only its initializers; each
            // type is its symbol's C prototype.
            let (begin, end, destroy) = unsafe {
                let driver = Box::leak(Box::new(Library::new("libcuda.so.1").unwrap()));
                (
                    *driver.get::<Capture>(b"cuStreamBeginCapture_v2").unwrap(),
                    *driver.get::<Captured>(b"cuStreamEndCapture").unwrap(),
                    *driver.get::<Destroy>(b"cuGraphDestroy").unwrap(),
                )
            };

            let _context = stream.enter()?;
            let mut graph = std::ptr::null_mut();
            // SAFETY: the stream is live; `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`
            // confines the capture to this thread, and the driver writes the
            // captured graph, which is destroyed once instantiated.
            unsafe {
                assert_eq!(begin(stream.handle(), 1), 0);
                fill(memory, stream, count)?;
                assert_eq!(end(stream.handle(), &mut graph), 0);
                let graph_exec = GraphExec::instantiate(stream.device(), graph as usize);
                assert_eq!(destroy(graph), 0);
                graph_exec
            }
        }

        #[test]
        #[ignore = "requires a CUDA device"]
        fn a_stream_counts_its_operations_once_and_its_idle_time_as_gap() -> Result<()> {
            let _serial = lock(&crate::cupti::tests::SERIAL);
            let device = DeviceId(0);
            let lane = Stream::new(device)?;
            let other = Stream::new(device)?;
            let memory = DeviceAllocation::linear(device, FILL_BYTES)?;
            let graph = captured(&memory, &lane, 16)?;
            let pause = Duration::from_millis(50);
            let activity = StreamActivity::start([("lane", &lane)])?;

            let first = fills(&memory, &lane, 16)?;
            lane.synchronize()?;
            // The lane stands idle for the pause while a stream that is not
            // registered works.
            fill(&memory, &other, 16)?;
            std::thread::sleep(pause);
            other.synchronize()?;
            // The graph, instantiated before the recording started, is
            // recorded by its nodes.
            let flags = EventFlags {
                timing: true,
                ..EventFlags::default()
            };
            let launched = [Event::new(flags), Event::new(flags)];
            launched[0].record(&lane)?;
            graph.launch(&lane)?;
            launched[1].record(&lane)?;
            lane.synchronize()?;

            let spanned = elapsed_us(&first)? + elapsed_us(&launched)?;
            let lane_time = super::taken_until("lane", |taken| taken.busy_us >= spanned * 9 / 10);
            let pause_us = pause.as_micros() as u64;
            // The events span the operations and the host's launch latency
            // before the first of each; the gap adds the host's return from
            // the synchronizations around the pause.
            assert!(
                (spanned * 9 / 10..=spanned + 50).contains(&lane_time.busy_us),
                "busy {} us, events {spanned} us",
                lane_time.busy_us
            );
            assert!(
                (pause_us..pause_us + 20_000).contains(&lane_time.gap_us),
                "gap {} us",
                lane_time.gap_us
            );
            assert_eq!((lane_time.late_us, take().dropped_records), (0, 0));
            drop(activity);
            Ok(())
        }
    }
}
