//! A recording of the process's CUDA timeline: every activity record of the
//! traced kinds and every runtime and driver API call.
//!
//! [`Timeline::start`] enables, on the process's subscription
//! ([`super::subscriber`]), the activity records of kernels
//! (`CONCURRENT_KERNEL`, which keeps kernels concurrent, unlike `KERNEL`),
//! copies (`MEMCPY`, `MEMCPY2`), fills (`MEMSET`), NVTX marks and their data
//! (`MARKER`, `MARKER_DATA`), memory allocations (`MEMORY2`) and the NVLink
//! and PCIe topology (`NVLINK`, `PCIE`), and the callbacks of every runtime
//! and driver API call. A CUDA graph's launch is recorded by its nodes
//! ([`GraphTrace::Nodes`]: each kernel, copy and fill of the graph carries
//! the graph's and the node's id) or as one record of the whole graph
//! ([`GraphTrace::Whole`], `GRAPH_TRACE`). Executables are instantiated in
//! their launch's tracing mode; changing that mode requires their owners
//! to instantiate the retained source graphs again before replay.
//!
//! An API call is timed from its entry callback to its exit callback in
//! CUPTI's clock on the calling thread; nested calls (a runtime call and the
//! driver calls it makes) pair by their order. The decoded [`Record`]s reach
//! the recording's sink in [`Delivered`] batches as CUPTI delivers activity
//! buffers, with the names first used in each batch; the API calls made
//! since the last batch travel with it. Dropping the timeline stops it and
//! delivers what remains.
//!
//! NVTX marks reach CUPTI only when the process loaded CUPTI as its NVTX
//! injection library at its first NVTX call: `NVTX_INJECTION64_PATH` names
//! [`super::library`] in the environment the process starts with (verified
//! with `nvidia-cuda-cupti` 13.0.85: without it no mark is recorded).

use std::cell::{Cell, RefCell};
use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use super::records::{self, Api, Names, Record};
use super::subscriber::{self, API_ENTER, DRIVER_DOMAIN, RUNTIME_DOMAIN};
use crate::{Error, Result, lock};

/// How a CUDA graph's launch is recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphTrace {
    /// One record of the whole launch.
    Whole,
    /// A record of each node.
    Nodes,
}

/// What one delivery carries.
#[derive(Debug, Default)]
pub struct Delivered {
    pub records: Vec<Record>,
    /// The names first used by these records, with their ids.
    pub names: Vec<(u32, String)>,
}

/// Where a timeline delivers.
pub type Sink = Arc<dyn Fn(Delivered) + Send + Sync>;

/// The activity kinds every timeline records.
const KINDS: [u32; 9] = [
    records::CONCURRENT_KERNEL,
    records::MEMCPY,
    records::MEMCPY2,
    records::MEMSET,
    records::MARKER,
    records::MARKER_DATA,
    records::MEMORY2,
    records::NVLINK,
    records::PCIE,
];

/// The recording in progress: its sink, its names and the kinds it decodes.
struct Recording {
    sink: Sink,
    names: Names,
    kinds: Vec<u32>,
}

static RECORDING: Mutex<Option<Recording>> = Mutex::new(None);

/// Whether API calls are recorded; read by every callback.
static APIS: AtomicBool = AtomicBool::new(false);

/// An API call that returned, whose name is interned at the next delivery.
struct Returned {
    domain: i32,
    cbid: u32,
    name: usize,
    start: u64,
    end: u64,
    thread: u32,
    correlation: u32,
    result: i32,
}

/// The API calls that returned since the last delivery.
static RETURNED: Mutex<Vec<Returned>> = Mutex::new(Vec::new());

thread_local! {
    /// The calling thread's API calls entered and not yet returned:
    /// domain, callback id, entry time and correlation id.
    static ENTERED: RefCell<Vec<(i32, u32, u64, u32)>> = const { RefCell::new(Vec::new()) };
    /// The calling thread's Linux thread id, once read.
    static THREAD: Cell<u32> = const { Cell::new(0) };
}

/// The process's timeline recording; stops on drop. One records at a time.
pub struct Timeline {
    kinds: Vec<u32>,
    domains: Vec<i32>,
}

impl Timeline {
    /// Record the process's timeline into `sink`, graphs as `graphs` says.
    ///
    /// CUDA graph executables must be instantiated in the tracing mode in
    /// which they are launched. When switching to or from whole-graph
    /// tracing, their owners reinstantiate the retained source graphs
    /// before submitting another replay. Running a whole-trace executable
    /// in node mode leaks CUPTI timestamp slots, eventually exhausting its
    /// profiling buffers.
    ///
    /// # Errors
    ///
    /// Fails when the process could not subscribe to CUPTI, CUPTI refuses a
    /// kind or a domain, or another timeline records.
    pub fn start(graphs: GraphTrace, sink: Sink) -> Result<Self> {
        subscriber::subscribe()?;
        let mut kinds = KINDS.to_vec();
        if graphs == GraphTrace::Whole {
            kinds.push(records::GRAPH_TRACE);
        }
        {
            let mut recording = lock(&RECORDING);
            if recording.is_some() {
                return Err(Error::ActivityRecorded);
            }
            *recording = Some(Recording {
                sink,
                names: Names::default(),
                kinds: kinds.clone(),
            });
        }
        // From here the drop releases what was enabled.
        let mut timeline = Self {
            kinds: Vec::new(),
            domains: Vec::new(),
        };
        for kind in kinds {
            subscriber::enable_kind(kind)?;
            timeline.kinds.push(kind);
        }
        APIS.store(true, Ordering::Release);
        for domain in [DRIVER_DOMAIN, RUNTIME_DOMAIN] {
            subscriber::enable_domain(domain)?;
            timeline.domains.push(domain);
        }
        Ok(timeline)
    }
}

impl Drop for Timeline {
    fn drop(&mut self) {
        for &domain in &self.domains {
            subscriber::disable_domain(domain);
        }
        APIS.store(false, Ordering::Release);
        for &kind in &self.kinds {
            subscriber::disable_kind(kind);
        }
        // The forced flush delivers the remaining records to the sink, and
        // the last delivery the API calls that returned after them.
        subscriber::flush(true);
        let mut recording = {
            let mut delivery = Delivery::begin();
            delivery.end();
            delivery.recording
        };
        *recording = None;
    }
}

/// The dispatch's part: times an API call from its entry to its exit.
///
/// # Safety
/// `data` is the `CUpti_CallbackData` of a runtime or driver API callback.
pub(super) unsafe fn on_api(domain: i32, cbid: u32, data: *const c_void) {
    if !APIS.load(Ordering::Acquire) {
        return;
    }
    let Ok(now) = super::timestamp() else {
        return;
    };
    let data = data.cast::<u8>();
    // SAFETY: `CUpti_CallbackData`: the site at 0, the function's name at 8,
    // a pointer to its return value at 24, the correlation id at 64.
    let (site, correlation) = unsafe {
        (
            data.cast::<i32>().read_unaligned(),
            data.add(64).cast::<u32>().read_unaligned(),
        )
    };
    if site == API_ENTER {
        ENTERED.with(|entered| entered.borrow_mut().push((domain, cbid, now, correlation)));
        return;
    }
    let started = ENTERED.with(|entered| {
        let mut entered = entered.borrow_mut();
        let position = entered
            .iter()
            .rposition(|&(d, c, _, k)| d == domain && c == cbid && k == correlation)?;
        Some(entered.remove(position).2)
    });
    let Some(start) = started else {
        // The call was entered before the recording started.
        return;
    };
    // SAFETY: as above; the return value is a `CUresult` or `cudaError_t`,
    // both 4 bytes, written before the exit callback.
    let (name, result) = unsafe {
        let name = data.add(8).cast::<*const c_char>().read_unaligned();
        let value = data.add(24).cast::<*const i32>().read_unaligned();
        (
            name,
            if value.is_null() {
                0
            } else {
                value.read_unaligned()
            },
        )
    };
    let thread = THREAD.with(|thread| {
        if thread.get() == 0 {
            // SAFETY: gettid has no preconditions.
            thread.set(unsafe { libc::gettid() } as u32);
        }
        thread.get()
    });
    lock(&RETURNED).push(Returned {
        domain,
        cbid,
        name: name as usize,
        start,
        end: now,
        thread,
        correlation,
        result,
    });
}

/// One delivered buffer's records, decoded while the delivery holds the
/// recording.
pub(super) struct Delivery {
    recording: MutexGuard<'static, Option<Recording>>,
    records: Vec<Record>,
}

impl Delivery {
    pub(super) fn begin() -> Self {
        Self {
            recording: lock(&RECORDING),
            records: Vec::new(),
        }
    }

    /// Decode one record of the buffer, if its kind is recorded.
    ///
    /// # Safety
    /// `record` points at a whole activity record.
    pub(super) unsafe fn record(&mut self, record: *const u8) {
        let Some(recording) = self.recording.as_mut() else {
            return;
        };
        // SAFETY: forwarded from the caller.
        if !recording.kinds.contains(&unsafe { records::kind(record) }) {
            return;
        }
        // SAFETY: forwarded from the caller.
        if let Some(decoded) = unsafe { records::decode(record, &mut recording.names) } {
            self.records.push(decoded);
        }
    }

    /// Deliver the decoded records with the API calls that returned since
    /// the last delivery.
    pub(super) fn end(&mut self) {
        let Some(recording) = self.recording.as_mut() else {
            return;
        };
        let returned = std::mem::take(&mut *lock(&RETURNED));
        for call in returned {
            // SAFETY: CUPTI's function names are static strings.
            let name = unsafe { recording.names.intern_c(call.name as *const c_char) };
            self.records.push(Record::Api(Api {
                domain: call.domain,
                cbid: call.cbid,
                name,
                start: call.start,
                end: call.end,
                thread: call.thread,
                correlation: call.correlation,
                result: call.result,
            }));
        }
        if self.records.is_empty() {
            return;
        }
        let delivered = Delivered {
            records: std::mem::take(&mut self.records),
            names: recording.names.take_new(),
        };
        (recording.sink)(delivered);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::{DeviceAllocation, DeviceId, Stream, memset_d32};

    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_timeline_records_a_fill_and_the_call_that_issued_it() -> Result<()> {
        let _serial = lock(&crate::cupti::tests::SERIAL);
        subscriber::subscribe()?;
        let stream = Stream::new(DeviceId(0))?;
        let memory = DeviceAllocation::linear(DeviceId(0), 1 << 20)?;
        let delivered: Arc<Mutex<Vec<Delivered>>> = Arc::default();
        let sink: Sink = {
            let delivered = Arc::clone(&delivered);
            Arc::new(move |batch| lock(&delivered).push(batch))
        };
        let timeline = Timeline::start(GraphTrace::Nodes, sink)?;
        // SAFETY: the allocation holds 1 MiB and outlives the fill, which
        // the synchronization waits for.
        unsafe { memset_d32(memory.address(), 7, 1 << 18, &stream)? };
        stream.synchronize()?;
        drop(timeline);

        let delivered = lock(&delivered);
        let records: Vec<&Record> = delivered.iter().flat_map(|batch| &batch.records).collect();
        let names: std::collections::HashMap<u32, &str> = delivered
            .iter()
            .flat_map(|batch| &batch.names)
            .map(|(id, name)| (*id, name.as_str()))
            .collect();
        let fill = records
            .iter()
            .find_map(|record| match record {
                Record::Fill(fill) if fill.bytes == 1 << 20 => Some(fill),
                _ => None,
            })
            .expect("the fill is recorded");
        assert_eq!(fill.value, 7);
        assert!(fill.end > fill.start);
        // The driver call that issued it carries its correlation id and
        // encloses its issue.
        let call = records
            .iter()
            .find_map(|record| match record {
                Record::Api(api)
                    if api.correlation == fill.correlation && api.domain == DRIVER_DOMAIN =>
                {
                    Some(api)
                }
                _ => None,
            })
            .expect("the issuing call is recorded");
        assert_eq!(names.get(&call.name), Some(&"cuMemsetD32Async"));
        assert!(call.start <= call.end && call.start <= fill.start);
        Ok(())
    }

    /// Switching tracing modes preserves every activity across more node
    /// launches than a bounded profiling-buffer pool can hold at once.
    #[test]
    #[ignore = "requires a CUDA device; run in a fresh process"]
    fn retained_graphs_record_nodes_whole_and_nodes_without_losing_timestamps() -> Result<()> {
        use crate::cupti::check;
        use crate::cupti::tests::{Spin, with_captured_graph};
        use std::sync::atomic::AtomicUsize;

        let _serial = lock(&crate::cupti::tests::SERIAL);
        subscriber::subscribe()?;
        // Limit the vendor's profiling pool so a missing buffer recycle
        // fails after hundreds of launches instead of millions.
        // SAFETY: the process's CUPTI is already loaded by the subscriber.
        let library = unsafe { libloading::Library::new(super::super::library()).unwrap() };
        type SetAttribute = unsafe extern "C" fn(i32, *mut usize, *mut c_void) -> i32;
        // SAFETY: this is the CUPTI header's cuptiActivitySetAttribute ABI.
        let set = unsafe {
            library
                .get::<SetAttribute>(b"cuptiActivitySetAttribute")
                .unwrap()
        };
        let mut size = std::mem::size_of::<usize>();
        let mut limit = 4usize;
        check(
            // SAFETY: attribute 2 takes a size_t; both pointers are live.
            unsafe { set(2, &mut size, (&raw mut limit).cast()) },
            "cuptiActivitySetAttribute",
        )?;
        let stream = Stream::new(DeviceId(0))?;
        let spin = Spin::load(&stream)?;
        let result = with_captured_graph(
            &stream,
            |stream| (0..1000).try_for_each(|_| spin.launch(stream, 1, 1)),
            |source| {
                for (mode, replays) in [
                    (GraphTrace::Nodes, 1024),
                    (GraphTrace::Whole, 1),
                    (GraphTrace::Nodes, 1024),
                ] {
                    let counts = Arc::new([
                        AtomicUsize::new(0),
                        AtomicUsize::new(0),
                        AtomicUsize::new(0),
                    ]);
                    let sink: Sink = {
                        let counts = Arc::clone(&counts);
                        Arc::new(move |batch: Delivered| {
                            for record in batch.records {
                                let (index, start, end) = match record {
                                    Record::Kernel(kernel) if kernel.graph != 0 => {
                                        (0, kernel.start, kernel.end)
                                    }
                                    Record::Graph(graph) => (1, graph.start, graph.end),
                                    _ => continue,
                                };
                                counts[index].fetch_add(1, Ordering::Relaxed);
                                if start == 0 || end <= start {
                                    counts[2].fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        })
                    };
                    let timeline = Timeline::start(mode, sink)?;
                    // The retained source is instantiated under the mode
                    // that all of this executable's launches will use.
                    // SAFETY: with_captured_graph keeps the source live,
                    // and this test is its sole owner and submitter.
                    let executable =
                        unsafe { crate::GraphExec::instantiate(stream.device(), source)? };
                    for replay in 0..replays {
                        executable.launch(&stream)?;
                        if replay % 128 == 127 {
                            stream.synchronize()?;
                            subscriber::flush(false);
                        }
                    }
                    stream.synchronize()?;
                    drop(executable);
                    drop(timeline);
                    let counts = counts.each_ref().map(|count| count.load(Ordering::Relaxed));
                    let expected = if mode == GraphTrace::Whole {
                        [0, 1, 0]
                    } else {
                        [1_024_000, 0, 0]
                    };
                    assert_eq!(counts, expected);
                }
                Ok(())
            },
        );
        limit = 250;
        check(
            // SAFETY: restore the size_t attribute through live pointers.
            unsafe { set(2, &mut size, (&raw mut limit).cast()) },
            "cuptiActivitySetAttribute",
        )?;
        result
    }
}
