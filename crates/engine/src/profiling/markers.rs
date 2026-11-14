//! Range markers: device timestamps at the entry and exit of the model's modules, always on,
//! exported as per-step module times (`forward_module_us`).
//!
//! **Where.** A model states each module's forward as a profiler range (`nn::profiler::range`):
//! its class, which names the kind of module, and its path, which names the instance. The context
//! a runner captures a graph under carries the process's [`Markers`] as its profiler
//! ([`profiler`], `ops::Context::set_profiler`), so a module whose class matches the
//! deployment's class patterns gets one marker as it is entered and one as it returns. Each is a
//! launch of [`PTX`]'s single-thread kernel on the forward's stream, so the graph captures it and
//! every replay runs it. The worker brackets each batch's execution with a step marker of the
//! same kernel on the runner's stream ([`Markers::begin_step`], [`Markers::end_step`]), which
//! stamps the batch and stores its id in the stream's step cell; the markers the replay runs read
//! the cell, so every stamp names its step.
//!
//! **What a marker does.** It reads the device switch: a word per class and a master word, in
//! device storage. A marker whose words are off returns at once, so switching markers needs no
//! capture again. A marker that is on reads `%globaltimer`, takes the next sequence number from a
//! device counter, and writes a 16-byte stamp (time, marker, step and the epoch tag below) into
//! its slot of a double buffer in pinned host storage the device writes through unified
//! addressing. Sequence number `s` writes slot `s mod N` of half `(s / N) mod 2`, `N` stamps a
//! half; the epoch `s / N` is tagged in the stamp (12 bits). The marker that completes a half
//! publishes the half's epoch plus one in the half's `filled` word.
//!
//! **Reading.** Each worker reads its device's halves after every batch
//! ([`Markers::read_device`]), without any device synchronization, and the control thread does so
//! with every metrics report ([`Markers::read`]), so an idle device's last stamps are not left
//! behind. A half is read once its `filled` word shows it complete, and at most once a
//! [`PARTIAL_READ`] so are the stamps of the half being filled, so steps are reported while the
//! marker rate is low. A stamp is taken only when its tag is that of the epoch read: a stamp not
//! yet visible is read again later, and one the device already overwrote with a later epoch (the
//! host fell two halves behind) is counted as lost.
//!
//! **Clocks.** `%globaltimer` is a nanosecond clock of the GPU. A calibration launches a kernel
//! that stores `%globaltimer` into pinned host storage and times its appearance on the host
//! ([`Markers::calibrate`]); the sample of least uncertainty aligns the GPU's clock to the host's
//! `CLOCK_REALTIME` and to CUPTI's clock.
//!
//! **Export.** At a step's end marker, its stamps are sorted by time and each module instance's
//! entry and exit are paired; the durations of a class's instances in the step are summed, and
//! the sum enters the histogram of the step's stage and the class, as does the step's own span
//! between its markers (class `step`). The histograms travel with the next batch result
//! ([`Markers::take`], [`super::forward_stats`]).
//!
//! **One process, many ranks.** The instances are the process's: every rank of a block registers
//! the same ones, and a rank of another block adds its own. Everything a device writes or a reader
//! keeps (the buffers, the step cells, the steps not yet ended, the histograms) is that device's,
//! because two ranks of one group run one batch id at once.

use std::collections::{BTreeMap, HashMap};
use std::error::Error as StdError;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crossbeam_channel::Sender;
use platform::{
    Device, DeviceAllocation, DeviceId, Function, LaunchConfig, Library, PinnedBuffer, Stream,
};
use proto::report::FenceHistogram;

use super::artifact::{ClockRow, Message};
use crate::worker::{Error, Result};
use service::lock;

/// The marker kernel and the clock kernel, in PTX the driver compiles for the device. Parameters
/// of `uniserve_range_marker`:
///
/// - `switches`: `u32[1 + classes]`, the master word then a word per class;
/// - `class`: the marker's word, 0 for a step marker (on with the master word);
/// - `marker`: the marker's id (20 bits);
/// - `cell`: the stream's step cell (`u32`);
/// - `mark`, `step`: a step marker stores `step` into the cell; a module's marker (`mark == 0`)
///   reads it;
/// - `counter`: the `u64` sequence counter; `written`: `u32[2]`, the stamps written into each
///   half's current epoch;
/// - `stamps`: `16 * 2 * N` bytes of pinned host storage; `filled`: `u64[2]` of pinned host
///   storage;
/// - `shift`: `log2 N`.
const PTX: &str = r"
.version 8.0
.target sm_90
.address_size 64

.visible .entry uniserve_range_marker(
    .param .u64 switches,
    .param .u32 class,
    .param .u32 marker,
    .param .u64 cell,
    .param .u32 mark,
    .param .u32 step,
    .param .u64 counter,
    .param .u64 written,
    .param .u64 stamps,
    .param .u64 filled,
    .param .u32 shift
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<24>;

    ld.param.u64 %rd1, [switches];
    ld.global.cg.u32 %r1, [%rd1];
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra DONE;
    ld.param.u32 %r2, [class];
    setp.eq.u32 %p1, %r2, 0;
    @%p1 bra ON;
    mul.wide.u32 %rd2, %r2, 4;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.cg.u32 %r3, [%rd3];
    setp.eq.u32 %p1, %r3, 0;
    @%p1 bra DONE;
ON:
    ld.param.u64 %rd4, [cell];
    ld.param.u32 %r4, [mark];
    ld.param.u32 %r5, [step];
    setp.ne.u32 %p2, %r4, 0;
    @%p2 st.global.u32 [%rd4], %r5;
    @!%p2 ld.global.cg.u32 %r5, [%rd4];
    mov.u64 %rd5, %globaltimer;
    ld.param.u64 %rd6, [counter];
    atom.global.add.u64 %rd7, [%rd6], 1;
    ld.param.u32 %r6, [shift];
    shr.u64 %rd9, %rd7, %r6;
    mov.u64 %rd8, 1;
    shl.b64 %rd8, %rd8, %r6;
    sub.u64 %rd10, %rd8, 1;
    and.b64 %rd10, %rd7, %rd10;
    and.b64 %rd11, %rd9, 1;
    mad.lo.u64 %rd12, %rd11, %rd8, %rd10;
    shl.b64 %rd13, %rd12, 4;
    ld.param.u64 %rd14, [stamps];
    add.u64 %rd15, %rd14, %rd13;
    ld.param.u32 %r7, [marker];
    and.b32 %r7, %r7, 1048575;
    cvt.u32.u64 %r8, %rd9;
    and.b32 %r8, %r8, 4095;
    shl.b32 %r8, %r8, 20;
    or.b32 %r9, %r7, %r8;
    mov.b64 %rd16, {%r9, %r5};
    st.global.v2.u64 [%rd15], {%rd5, %rd16};
    ld.param.u64 %rd17, [written];
    shl.b64 %rd18, %rd11, 2;
    add.u64 %rd19, %rd17, %rd18;
    atom.global.add.u32 %r10, [%rd19], 1;
    cvt.u32.u64 %r11, %rd8;
    sub.u32 %r12, %r11, 1;
    setp.ne.u32 %p3, %r10, %r12;
    @%p3 bra DONE;
    neg.s32 %r13, %r11;
    atom.global.add.u32 %r14, [%rd19], %r13;
    ld.param.u64 %rd20, [filled];
    shl.b64 %rd21, %rd11, 3;
    add.u64 %rd22, %rd20, %rd21;
    add.u64 %rd23, %rd9, 1;
    st.volatile.global.u64 [%rd22], %rd23;
DONE:
    ret;
}

.visible .entry uniserve_clock(.param .u64 out)
{
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [out];
    mov.u64 %rd2, %globaltimer;
    st.volatile.global.u64 [%rd1], %rd2;
    ret;
}
";

/// `log2` of the stamps a half holds: 4096 stamps, 64 KiB.
const SHIFT: u32 = 12;

/// Stamps a half holds.
const HALF: u64 = 1 << SHIFT;

/// The tag bits of a stamp's epoch.
const TAG_MASK: u64 = 0xfff;

/// The most classes one process's markers name: the words of the device switch. A model states a
/// handful of classes, and a capture that names more is refused rather than left unswitchable.
const CLASSES: usize = 1024;

/// The marker ids of a step's begin and end.
const STEP_BEGIN: u32 = 0xf_fffe;
const STEP_END: u32 = 0xf_ffff;

/// The step cells a device keeps: one per stream a runner computes on, and cell 0, shared once
/// every other is taken.
const CELLS: usize = 64;

/// How often the half being filled is read, at most.
const PARTIAL_READ: Duration = Duration::from_secs(1);

/// How long a complete half's stamps that are not visible yet are waited for before they are
/// counted as lost.
const VISIBILITY: Duration = Duration::from_millis(50);

/// How long a calibration spins for its timestamp on the host before it blocks until the clock
/// kernel completes: on a busy device the kernel queues behind the device's work for longer.
const CALIBRATION_SPIN: Duration = Duration::from_micros(500);

/// The steps whose end marker has not been read that a device keeps; older ones (a batch that
/// failed between its markers) are dropped.
const OPEN_STEPS: usize = 4096;

/// What one marker wrote, as a reader takes it from a device's buffer and a session's writer
/// stores it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Stamp {
    /// `%globaltimer`, nanoseconds.
    pub(crate) time: u64,
    /// `2 * instance` at a module instance's entry, `2 * instance + 1` at its exit, or a step
    /// marker's id.
    pub(crate) marker: u32,
    /// The step's batch id, low 32 bits.
    pub(crate) step: u32,
}

/// A device's marker resources: the kernels, the switch, the counter, the step cells and the
/// double buffer, and the reader's position in it.
struct DeviceMarkers {
    device: DeviceId,
    _library: Library,
    marker: Function,
    clock: Function,
    /// `u32[1 + CLASSES]`.
    switches: DeviceAllocation,
    /// The `u64` counter, then `u32[2]` written counts.
    counters: DeviceAllocation,
    /// `u32[CELLS]`.
    cells: DeviceAllocation,
    /// `[u64 time, u64 packed][2 * HALF]`.
    stamps: PinnedBuffer,
    /// `u64[2]`.
    filled: PinnedBuffer,
    /// `u64`, the clock kernel's output.
    clock_out: PinnedBuffer,
    /// The stream the switch writes and calibrations go on.
    control: Stream,
    /// The step cell of each stream that carried a marker, by the stream's handle.
    stream_cells: Mutex<HashMap<usize, usize>>,
    reader: Mutex<Reader>,
    /// The stage of each step whose begin marker was launched.
    stages: Mutex<HashMap<u32, Arc<str>>>,
    /// Histograms of completed steps since the last take, by `{stage}/{class}`.
    histograms: Mutex<BTreeMap<String, FenceHistogram>>,
    /// Where the stamps read go while a session keeps them: the class patterns it keeps (empty
    /// for every class) and its file's writer.
    tap: Mutex<Option<(Vec<String>, Sender<Message>)>>,
}

/// Where a device's reader stands.
#[derive(Default)]
struct Reader {
    /// The next sequence number to take.
    next: u64,
    /// When a complete half first showed stamps not yet visible.
    waiting_since: Option<Instant>,
    /// Stamps counted as lost since the last take.
    lost: u64,
    /// When the half being filled was last read.
    partial_read: Option<Instant>,
    /// The stamps of each step whose end marker has not been read: its begin marker's and its
    /// modules'.
    steps: BTreeMap<u32, Vec<Stamp>>,
    /// The calibration sample of least uncertainty.
    best: Option<ClockRow>,
}

/// The module instances the process's captures entered, and their classes.
#[derive(Default)]
struct Instances {
    /// Each instance's class and path, by its number.
    instances: Vec<(String, String)>,
    /// Each instance's number, by class and path.
    numbers: HashMap<(String, String), u32>,
    /// Class names, by class index from 1.
    classes: Vec<String>,
    /// Each instance's class index.
    instance_classes: Vec<u32>,
    /// Whether a class pattern of the deployment names each class seen, by name.
    timed: HashMap<String, bool>,
}

/// The range markers of this process's devices and what they measure.
pub(crate) struct Markers {
    devices: Vec<DeviceMarkers>,
    /// The deployment's class patterns (`profiling_config.fence_modules`).
    patterns: Vec<String>,
    instances: Mutex<Instances>,
    /// The switch as last stated: the master word and the patterns switched off.
    switched: Mutex<(bool, Vec<String>)>,
    /// The master word, read at every step: a step marker with the switch off stamps nothing, so
    /// it is not launched.
    recording: AtomicBool,
}

static MARKERS: OnceLock<Arc<Markers>> = OnceLock::new();

/// The profiler of a context on `stream`: the process's markers while the stream captures,
/// because a graph is what runs them at every step, and none for a forward run once or in a
/// process without markers.
///
/// A runner gives the result to `ops::Context::set_profiler` before it runs the forward it
/// captures.
pub(crate) fn profiler(stream: &Stream) -> Option<Arc<dyn ops::Profiler>> {
    let markers = MARKERS.get()?;
    let capturing = stream
        .capture_status()
        .is_ok_and(|status| status != platform::CaptureStatus::None);

    capturing.then(|| Arc::clone(markers) as Arc<dyn ops::Profiler>)
}

impl ops::Profiler for Markers {
    fn enter(
        &self,
        class: &str,
        path: &str,
        stream: &Stream,
    ) -> std::result::Result<Option<u32>, Box<dyn StdError + Send + Sync>> {
        let Some(instance) = self.instance(class, path)? else {
            return Ok(None);
        };
        self.launch(stream, 2 * instance)?;
        Ok(Some(instance))
    }

    fn exit(
        &self,
        instance: u32,
        stream: &Stream,
    ) -> std::result::Result<(), Box<dyn StdError + Send + Sync>> {
        self.launch(stream, 2 * instance + 1)?;
        Ok(())
    }
}

impl Markers {
    /// Installs the markers of the classes `patterns` name (`*` matches any run) on `devices`,
    /// every switch word on as `enabled` says but those of the classes `off` names, as the
    /// process's markers. A graph captured from here on carries them ([`profiler`]).
    ///
    /// # Errors
    ///
    /// Fails when a device cannot load the kernels or allocate the buffers, or the process
    /// installed markers already.
    pub(crate) fn install(
        devices: &[DeviceId],
        patterns: Vec<String>,
        enabled: bool,
        off: &[String],
    ) -> Result<Arc<Self>> {
        let devices = devices
            .iter()
            .map(|&device| DeviceMarkers::new(device))
            .collect::<Result<Vec<_>>>()?;
        let markers = Arc::new(Self {
            devices,
            patterns,
            instances: Mutex::default(),
            switched: Mutex::new((enabled, off.to_vec())),
            recording: AtomicBool::new(enabled),
        });
        markers.switch(enabled, off)?;

        MARKERS
            .set(Arc::clone(&markers))
            .map_err(|_| Error::State("the process installed range markers already"))?;
        Ok(markers)
    }

    /// The process's markers, once installed.
    pub(crate) fn installed() -> Option<&'static Arc<Self>> {
        MARKERS.get()
    }

    /// The module instances that carry markers, as `(class, path)` in their numbers' order.
    pub(crate) fn instances(&self) -> Vec<(String, String)> {
        lock(&self.instances).instances.clone()
    }

    /// The number of the instance at `path` of `class`, naming it on first sight, or `None` when
    /// no pattern of the deployment names the class.
    fn instance(&self, class: &str, path: &str) -> Result<Option<u32>> {
        let mut instances = lock(&self.instances);

        let timed = match instances.timed.get(class) {
            Some(&timed) => timed,
            None => {
                let timed = self.patterns.iter().any(|pattern| matches(pattern, class));
                instances.timed.insert(class.to_owned(), timed);
                timed
            }
        };
        if !timed {
            return Ok(None);
        }

        let key = (class.to_owned(), path.to_owned());
        if let Some(&number) = instances.numbers.get(&key) {
            return Ok(Some(number));
        }
        let number = u32::try_from(instances.instances.len())
            .ok()
            .filter(|number| 2 * *number < STEP_BEGIN)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "{} module instances exceed the range markers' id space",
                    instances.instances.len()
                ))
            })?;

        // A class seen for the first time takes the next switch word, written as the switch
        // stands now.
        let index = match instances.classes.iter().position(|known| known == class) {
            Some(index) => index,
            None => {
                if instances.classes.len() == CLASSES {
                    return Err(Error::Invalid(format!(
                        "{CLASSES} module classes fill the range markers' device switch"
                    )));
                }
                instances.classes.push(class.to_owned());
                let index = instances.classes.len() - 1;
                let (enabled, off) = lock(&self.switched).clone();
                let (words, _) = switch_words(&instances.classes, enabled, &off);
                self.write_switch(&words)?;
                index
            }
        };

        instances.instances.push(key.clone());
        instances.instance_classes.push(index as u32 + 1);
        instances.numbers.insert(key, number);
        Ok(Some(number))
    }

    /// Turns the markers on or off: the master word as `enabled` says, and every class's word
    /// with it but those of the classes matching a pattern of `off` (`*` matches any run), which
    /// stay off. Written on each device's control stream; markers already queued read whichever
    /// value reaches them. Returns the classes switched off.
    ///
    /// # Errors
    ///
    /// When a device refuses the write.
    pub(crate) fn switch(&self, enabled: bool, off: &[String]) -> Result<Vec<String>> {
        let instances = lock(&self.instances);
        *lock(&self.switched) = (enabled, off.to_vec());
        self.recording.store(enabled, Ordering::Release);

        let (words, switched_off) = switch_words(&instances.classes, enabled, off);
        self.write_switch(&words)?;
        Ok(switched_off)
    }

    fn write_switch(&self, words: &[u32]) -> Result<()> {
        for device in &self.devices {
            for (index, &word) in words.iter().enumerate() {
                // SAFETY: the switch holds `1 + CLASSES` words, and lives as long as the markers,
                // past the fill.
                unsafe {
                    platform::memset_d32(
                        device.switches.address() + 4 * index as u64,
                        word,
                        1,
                        &device.control,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// The step cell of the stream of handle `stream`: the one it was first given, or the next
    /// free one, or the shared cell once every cell is taken. A step marker and a module's marker
    /// on the same stream resolve the same cell whichever comes first, so a marker a capture
    /// records before the stream's first step reads the steps its step markers write.
    fn cell(device: &DeviceMarkers, stream: usize) -> usize {
        let mut cells = lock(&device.stream_cells);
        if let Some(&cell) = cells.get(&stream) {
            return cell;
        }
        let cell = next_cell(&cells);
        cells.insert(stream, cell);
        cell
    }

    fn device(&self, device: DeviceId) -> Result<&DeviceMarkers> {
        self.devices
            .iter()
            .find(|markers| markers.device == device)
            .ok_or_else(|| Error::Invalid(format!("no range markers on {device}")))
    }

    /// Launches marker `marker` (`2 * instance` at the entry, `+ 1` at the exit) on `stream`;
    /// captured when the stream captures.
    fn launch(&self, stream: &Stream, marker: u32) -> Result<()> {
        let class = *lock(&self.instances)
            .instance_classes
            .get((marker / 2) as usize)
            .ok_or_else(|| {
                Error::Invalid(format!("range marker {marker} names no module instance"))
            })?;
        let markers = self.device(stream.device())?;
        let cell = Self::cell(markers, stream.raw());
        markers.launch(stream, class, marker, cell, None)?;

        // A session's range profiler pushes the instance's range at its entry and pops it at its
        // exit; a capture pushes none, the push being a host call no replay repeats.
        if super::collectors::module_ranges_armed() {
            let capturing = stream
                .capture_status()
                .map_or(true, |status| status != platform::CaptureStatus::None);
            super::collectors::module_range(stream.device(), marker, capturing);
        }
        Ok(())
    }

    /// Launches the begin marker of step `step` (a batch id) of stage `stage` (its call kind) on
    /// `stream`, the runner's computation stream, before the step's first launch.
    ///
    /// # Errors
    ///
    /// When the stream's device has no markers or the launch fails.
    pub(crate) fn begin_step(&self, stream: &Stream, step: u64, stage: &str) -> Result<()> {
        if !self.recording.load(Ordering::Acquire) {
            return Ok(());
        }
        let markers = self.device(stream.device())?;
        let cell = Self::cell(markers, stream.raw());
        // A stamp carries the low 32 bits of the batch id.
        let step = step as u32;

        {
            let mut stages = lock(&markers.stages);
            // A step whose end marker was never read leaves its stage here.
            if stages.len() >= OPEN_STEPS {
                stages.clear();
            }
            stages.insert(step, Arc::from(stage));
        }
        markers.launch(stream, 0, STEP_BEGIN, cell, Some(step))
    }

    /// Launches the end marker of step `step` on `stream`, right behind the step's last launch:
    /// launched after the worker waited for the step, its span would hold the host's trip back
    /// as well as the device's work.
    ///
    /// # Errors
    ///
    /// When the stream's device has no markers or the launch fails.
    pub(crate) fn end_step(&self, stream: &Stream, step: u64) -> Result<()> {
        if !self.recording.load(Ordering::Acquire) {
            return Ok(());
        }
        let markers = self.device(stream.device())?;
        let cell = Self::cell(markers, stream.raw());
        markers.launch(stream, 0, STEP_END, cell, Some(step as u32))
    }

    /// Has the stamps read from `device` from now on also reach `rows`, a session's writer, until
    /// [`Markers::untap`]: those of the classes matching a pattern of `classes` (every class when
    /// empty) and the step markers'.
    pub(crate) fn tap(
        &self,
        device: DeviceId,
        classes: Vec<String>,
        rows: Sender<Message>,
    ) -> Result<()> {
        *lock(&self.device(device)?.tap) = Some((classes, rows));
        Ok(())
    }

    pub(crate) fn untap(&self, device: DeviceId) {
        if let Ok(markers) = self.device(device) {
            *lock(&markers.tap) = None;
        }
    }

    /// Reads `device`'s complete halves, and the half being filled when [`PARTIAL_READ`] passed
    /// since its last read or `partial` asks; aggregates the steps that ended. Never waits for
    /// the device.
    pub(crate) fn read_device(&self, device: DeviceId, partial: bool) {
        if let Ok(markers) = self.device(device) {
            self.read_of(markers, partial);
        }
    }

    fn read_of(&self, device: &DeviceMarkers, partial: bool) {
        let stamps = device.read(partial);
        if stamps.is_empty() {
            return;
        }
        self.aggregate(device, &stamps);

        if let Some((classes, rows)) = lock(&device.tap).as_ref() {
            let kept: Vec<Stamp> = stamps
                .into_iter()
                .filter(|stamp| self.kept(classes, stamp.marker))
                .collect();
            if !kept.is_empty() {
                super::artifact::send(
                    rows,
                    Message::Stamps {
                        device: device.device.0,
                        stamps: kept,
                    },
                );
            }
        }
    }

    /// Whether a session that keeps the classes matching `classes` keeps the stamps of `marker`.
    fn kept(&self, classes: &[String], marker: u32) -> bool {
        if classes.is_empty() || marker >= STEP_BEGIN {
            return true;
        }
        let instances = lock(&self.instances);
        let class = instances
            .instance_classes
            .get((marker / 2) as usize)
            .and_then(|&class| instances.classes.get(class as usize - 1));

        class.is_some_and(|name| classes.iter().any(|pattern| matches(pattern, name)))
    }

    /// Adds the steps whose end marker `stamps` hold to the device's histograms.
    fn aggregate(&self, device: &DeviceMarkers, stamps: &[Stamp]) {
        let mut reader = lock(&device.reader);
        let mut ended = Vec::new();
        for stamp in stamps {
            match stamp.marker {
                STEP_END => ended.push((stamp.step, stamp.time)),
                _ => reader.steps.entry(stamp.step).or_default().push(*stamp),
            }
        }
        // A step whose other stamps were all lost still ends, so that its stage is let go.
        let completed: Vec<(u32, u64, Vec<Stamp>)> = ended
            .into_iter()
            .map(|(step, end)| (step, end, reader.steps.remove(&step).unwrap_or_default()))
            .collect();
        while reader.steps.len() > OPEN_STEPS {
            reader.steps.pop_first();
        }
        drop(reader);

        let mut stages = lock(&device.stages);
        let mut histograms = lock(&device.histograms);
        let instances = lock(&self.instances);
        for (step, end, mut open) in completed {
            let stage = stages.remove(&step).unwrap_or_else(|| Arc::from("unknown"));
            let begin = open.iter().rev().find(|stamp| stamp.marker == STEP_BEGIN);
            if let Some(begin) = begin {
                histograms
                    .entry(format!("{stage}/step"))
                    .or_default()
                    .observe(end.saturating_sub(begin.time) / 1000);
            }
            for (class, ns) in class_time(&instances.instance_classes, &mut open) {
                histograms
                    .entry(format!("{stage}/{}", instances.classes[class - 1]))
                    .or_default()
                    .observe(ns / 1000);
            }
        }
    }

    /// The histograms of `device`'s steps completed since the last take, in microseconds by
    /// `{stage}/{class}`, and its stamps lost.
    pub(crate) fn take(&self, device: DeviceId) -> (BTreeMap<String, FenceHistogram>, u64) {
        let Ok(markers) = self.device(device) else {
            return Default::default();
        };
        let lost = std::mem::take(&mut lock(&markers.reader).lost);
        (std::mem::take(&mut *lock(&markers.histograms)), lost)
    }

    /// Calibrates `device`'s clock against the host's: its best sample so far, as a row of a
    /// session file's `clocks`.
    pub(crate) fn calibrate(&self, device: DeviceId) -> Option<ClockRow> {
        let markers = self.device(device).ok()?;
        match markers.calibrate() {
            Ok(sample) => Some(sample),
            Err(error) => {
                tracing::warn!("the clock calibration on {device} failed: {error}");
                lock(&markers.reader).best
            }
        }
    }
}

/// The next free step cell of a device whose streams hold `cells`, or the shared cell 0 once
/// every other is taken.
fn next_cell(cells: &HashMap<usize, usize>) -> usize {
    let next = cells.values().copied().max().unwrap_or(0) + 1;
    if next < CELLS { next } else { 0 }
}

/// Each class's summed instance durations in one step's stamps, in nanoseconds by class index:
/// the stamps are sorted by time, and each instance's entries and exits pair in that order. A
/// step marker's stamp counts for no class.
fn class_time(instance_classes: &[u32], stamps: &mut [Stamp]) -> BTreeMap<usize, u64> {
    stamps.sort_unstable_by_key(|stamp| (stamp.time, stamp.marker));
    let mut entered: HashMap<u32, Vec<u64>> = HashMap::new();
    let mut time: BTreeMap<usize, u64> = BTreeMap::new();

    for stamp in stamps.iter().filter(|stamp| stamp.marker < STEP_BEGIN) {
        let instance = stamp.marker / 2;
        if stamp.marker % 2 == 0 {
            entered.entry(instance).or_default().push(stamp.time);
        } else if let Some(start) = entered.get_mut(&instance).and_then(Vec::pop) {
            let Some(&class) = instance_classes.get(instance as usize) else {
                continue;
            };
            *time.entry(class as usize).or_default() += stamp.time.saturating_sub(start);
        }
    }
    time
}

/// Whether `name` matches `pattern`, where `*` matches any run of characters.
pub(crate) fn matches(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    let [first, middle @ .., last] = parts.as_slice() else {
        return pattern == name;
    };
    if parts.len() == 1 {
        return pattern == name;
    }

    let Some(mut rest) = name.strip_prefix(first) else {
        return false;
    };
    for part in middle {
        match rest.find(part) {
            Some(at) => rest = &rest[at + part.len()..],
            None => return false,
        }
    }
    rest.len() >= last.len() && rest.ends_with(last)
}

impl DeviceMarkers {
    fn new(device: DeviceId) -> Result<Self> {
        let handle = Device::new(device)?;
        // The driver compiles the text for the device, and it needs the terminating NUL a text
        // image is read to.
        let mut image = PTX.as_bytes().to_vec();
        image.push(0);
        let library = Library::load(&handle, &image)?;
        let marker = library.kernel("uniserve_range_marker")?;
        let clock = library.kernel("uniserve_clock")?;

        let control = Stream::with_priority(device, -5)?;
        let switches = DeviceAllocation::linear(device, 4 * (1 + CLASSES))?;
        let counters = DeviceAllocation::linear(device, 16)?;
        let cells = DeviceAllocation::linear(device, 4 * CELLS)?;
        let mut stamps = PinnedBuffer::new(device, (16 * 2 * HALF) as usize)?;
        let mut filled = PinnedBuffer::new(device, 16)?;
        let mut clock_out = PinnedBuffer::new(device, 8)?;

        // Every slot starts tagged with an epoch no write of the first two epochs carries.
        // SAFETY: no device work addresses the buffers yet.
        unsafe {
            for slot in stamps.as_mut_slice().as_chunks_mut::<16>().0 {
                slot[..8].fill(0);
                slot[8..].copy_from_slice(&(TAG_MASK << 20).to_le_bytes());
            }
            filled.as_mut_slice().fill(0);
            clock_out.as_mut_slice().fill(0);
            platform::memset_d32(switches.address(), 0, 1 + CLASSES, &control)?;
            platform::memset_d32(counters.address(), 0, 4, &control)?;
            platform::memset_d32(cells.address(), 0, CELLS, &control)?;
        }
        control.synchronize()?;

        Ok(Self {
            device,
            _library: library,
            marker,
            clock,
            switches,
            counters,
            cells,
            stamps,
            filled,
            clock_out,
            control,
            stream_cells: Mutex::new(HashMap::new()),
            reader: Mutex::new(Reader::default()),
            stages: Mutex::new(HashMap::new()),
            histograms: Mutex::new(BTreeMap::new()),
            tap: Mutex::new(None),
        })
    }

    /// Launches the marker kernel on `stream`: a module's marker of switch word `class`, or with
    /// `step` a step marker.
    fn launch(
        &self,
        stream: &Stream,
        class: u32,
        marker: u32,
        cell: usize,
        step: Option<u32>,
    ) -> Result<()> {
        let mut switches = self.switches.address();
        let mut class = class;
        let mut marker = marker;
        let mut cell = self.cells.address() + 4 * cell as u64;
        let mut mark = u32::from(step.is_some());
        let mut step = step.unwrap_or(0);
        let mut counter = self.counters.address();
        let mut written = self.counters.address() + 8;
        let mut stamps = self.stamps.address();
        let mut filled = self.filled.address();
        let mut shift = SHIFT;
        let mut parameters: [*mut c_void; 11] = [
            (&raw mut switches).cast(),
            (&raw mut class).cast(),
            (&raw mut marker).cast(),
            (&raw mut cell).cast(),
            (&raw mut mark).cast(),
            (&raw mut step).cast(),
            (&raw mut counter).cast(),
            (&raw mut written).cast(),
            (&raw mut stamps).cast(),
            (&raw mut filled).cast(),
            (&raw mut shift).cast(),
        ];

        // The parameters are the kernel's, and every buffer they address lives with the process's
        // markers, past every replay of a graph that captured the launch.
        self.marker
            .launch_ex(stream, &single_thread(), &mut parameters)?;
        Ok(())
    }

    /// The `filled` word of `half`.
    fn filled(&self, half: u64) -> u64 {
        // SAFETY: the word is pinned host storage the device writes whole; a volatile read
        // observes its latest value.
        unsafe { std::ptr::read_volatile(self.filled.as_ptr().cast::<u64>().add(half as usize)) }
    }

    /// The two words of sequence number `sequence`'s slot.
    fn slot(&self, sequence: u64) -> (u64, u64) {
        let index = ((sequence >> SHIFT) & 1) * HALF + (sequence & (HALF - 1));
        // SAFETY: the index lies in the buffer; the device writes each stamp whole in one 16-byte
        // store, read here as two volatile words.
        unsafe {
            let base = self.stamps.as_ptr().cast::<u64>().add(2 * index as usize);
            (
                std::ptr::read_volatile(base),
                std::ptr::read_volatile(base.add(1)),
            )
        }
    }

    /// Takes the stamps the device completed: the complete halves in epoch order, then, when due,
    /// the visible stamps of the half being filled.
    fn read(&self, partial: bool) -> Vec<Stamp> {
        let mut reader = lock(&self.reader);
        let mut taken = Vec::new();
        let completed = self.filled(0).max(self.filled(1));

        // The device is filling epoch `completed` (or a later one, before its predecessor shows
        // complete); a half two epochs behind it was overwritten.
        let oldest = completed.saturating_sub(1);
        if reader.next >> SHIFT < oldest {
            let resumed = oldest << SHIFT;
            reader.lost += resumed - reader.next;
            reader.next = resumed;
        }

        loop {
            let epoch = reader.next >> SHIFT;
            if self.filled(epoch & 1) != epoch + 1 {
                break;
            }
            // A complete half: every stamp past the reader's position.
            let mut pending = false;
            while reader.next >> SHIFT == epoch {
                match stamp(self.slot(reader.next), epoch) {
                    Some(stamp) => taken.push(stamp),
                    None if tag(self.slot(reader.next).1) == (epoch + 2) & TAG_MASK => {
                        reader.lost += 1;
                    }
                    None => {
                        pending = true;
                        break;
                    }
                }
                reader.next += 1;
            }
            if pending {
                let since = *reader.waiting_since.get_or_insert_with(Instant::now);
                if since.elapsed() < VISIBILITY {
                    return taken;
                }
                // Gone: the device moved on before the host saw it.
                reader.lost += 1;
                reader.next += 1;
                continue;
            }
            reader.waiting_since = None;
        }

        let due = partial
            || reader
                .partial_read
                .is_none_or(|read| read.elapsed() >= PARTIAL_READ);
        if due {
            reader.partial_read = Some(Instant::now());
            let epoch = reader.next >> SHIFT;
            while reader.next >> SHIFT == epoch {
                match stamp(self.slot(reader.next), epoch) {
                    Some(stamp) => taken.push(stamp),
                    None => break,
                }
                reader.next += 1;
            }
        }
        taken
    }

    /// One calibration sample: the clock kernel's `%globaltimer` against the host clocks around
    /// its appearance. Keeps and returns the best sample.
    ///
    /// The sample's uncertainty is half the host's window from before the launch to the value's
    /// appearance: a kernel that queued behind the device's work gives a wide sample, which a
    /// narrower one replaces.
    fn calibrate(&self) -> Result<ClockRow> {
        // SAFETY: the word is pinned host storage no device work addresses now: a calibration
        // returns only once its kernel has stored into it.
        unsafe { std::ptr::write_volatile(self.clock_out.as_ptr().cast::<u64>(), 0) };
        let mut out = self.clock_out.address();
        let mut parameters: [*mut c_void; 1] = [(&raw mut out).cast()];
        let before = (realtime(), platform::cupti::timestamp().unwrap_or(0));
        let started = Instant::now();

        // The parameter is the kernel's; the word outlives it.
        self.clock
            .launch_ex(&self.control, &single_thread(), &mut parameters)?;
        let gpu = loop {
            // SAFETY: as above; the device writes the word whole.
            let value = unsafe { std::ptr::read_volatile(self.clock_out.as_ptr().cast::<u64>()) };
            if value != 0 {
                break value;
            }
            if started.elapsed() > CALIBRATION_SPIN {
                // Blocking until the kernel completes ends the sample, and keeps the next
                // calibration, which zeroes the word, from reading this kernel's late store as
                // its own.
                self.control.synchronize()?;
                // SAFETY: as above; the kernel completed.
                let value =
                    unsafe { std::ptr::read_volatile(self.clock_out.as_ptr().cast::<u64>()) };
                if value == 0 {
                    return Err(Error::State("the clock kernel completed without its time"));
                }
                break value;
            }
            std::hint::spin_loop();
        };
        let after = (realtime(), platform::cupti::timestamp().unwrap_or(0));

        let sample = ClockRow {
            source: "globaltimer",
            device: Some(self.device.0),
            gpu: Some(gpu),
            realtime: before.0 / 2 + after.0 / 2,
            cupti: Some(before.1 / 2 + after.1 / 2),
            monotonic: None,
            uncertainty: (after.0 - before.0) / 2,
        };
        let mut reader = lock(&self.reader);
        let best = match reader.best {
            Some(best) if best.uncertainty <= sample.uncertainty => best,
            _ => sample,
        };
        reader.best = Some(best);
        Ok(best)
    }
}

/// One block of one thread.
fn single_thread() -> LaunchConfig {
    LaunchConfig::new([1, 1, 1], [1, 1, 1])
}

/// The host's `CLOCK_MONOTONIC`, nanoseconds: BPF's `ktime_get_ns`.
pub(crate) fn monotonic() -> u64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: clock_gettime writes one timespec.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
    now.tv_sec as u64 * 1_000_000_000 + now.tv_nsec as u64
}

/// The host's `CLOCK_REALTIME`, nanoseconds.
pub(crate) fn realtime() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64)
}

/// The epoch tag of a stamp's packed word.
fn tag(packed: u64) -> u64 {
    (packed >> 20) & TAG_MASK
}

/// The stamp of the slot words `(time, packed)` if it carries `epoch`'s tag.
fn stamp((time, packed): (u64, u64), epoch: u64) -> Option<Stamp> {
    (tag(packed) == epoch & TAG_MASK).then_some(Stamp {
        time,
        marker: (packed & 0xf_ffff) as u32,
        step: (packed >> 32) as u32,
    })
}

/// The switch words of `classes` (by class index from 1): the master word, on as `enabled` says,
/// then each class's, on with it unless the class matches a pattern of `off`; and the classes
/// matched.
fn switch_words(classes: &[String], enabled: bool, off: &[String]) -> (Vec<u32>, Vec<String>) {
    let mut words = vec![u32::from(enabled)];
    let mut switched_off = Vec::new();
    for class in classes {
        let excluded = off.iter().any(|pattern| matches(pattern, class));
        if excluded {
            switched_off.push(class.clone());
        }
        words.push(u32::from(enabled && !excluded));
    }
    (words, switched_off)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The master word follows the switch; a class's word follows it unless a pattern of the off
    /// list names the class, whose markers stay off while the others stamp.
    #[test]
    fn a_class_named_off_stays_off_while_the_others_stamp() {
        let classes = ["Attention".to_owned(), "Transformer".to_owned()];
        assert_eq!(
            switch_words(&classes, true, &["*Attention".to_owned()]),
            (vec![1, 0, 1], vec![classes[0].clone()])
        );
        assert_eq!(
            switch_words(&classes, true, &[]),
            (vec![1, 1, 1], Vec::new())
        );
        assert_eq!(
            switch_words(&classes, false, &["*Attention".to_owned()]),
            (vec![0, 0, 0], vec![classes[0].clone()])
        );
    }

    #[test]
    fn a_pattern_star_matches_any_run() {
        assert!(matches("*", "Linear"));
        assert!(matches("nn.linear.*", "nn.linear.QKVParallelLinear"));
        assert!(matches("*Attention", "CrossAttention"));
        assert!(matches("models.*.Decoder*", "models.qwen3.DecoderLayer"));
        assert!(!matches("nn.linear.*", "nn.norm.RMSNorm"));
        assert!(matches("RMSNorm", "RMSNorm"));
        assert!(!matches("RMS", "RMSNorm"));
    }

    #[test]
    fn a_stamp_is_taken_only_with_its_epochs_tag() {
        let packed =
            |marker: u64, epoch: u64, step: u64| marker | ((epoch & TAG_MASK) << 20) | (step << 32);
        let taken = stamp((5, packed(7, 3, 42)), 3).unwrap();
        assert_eq!((taken.time, taken.marker, taken.step), (5, 7, 42));

        // Written two epochs later, or not yet written for this one.
        assert!(stamp((5, packed(7, 5, 42)), 3).is_none());
        assert!(stamp((0, packed(0, TAG_MASK, 0)), 0).is_none());
    }

    #[test]
    fn nested_instances_each_sum_their_own_span() {
        // Instance 0 of class 1 spans 10 to 20 and holds instance 1 of class 2, 12 to 15; the
        // stamps arrive out of time order, between the step's own markers.
        let instance_classes = [1, 2];
        let mut stamps =
            [(15, 3), (9, STEP_BEGIN), (10, 0), (20, 1), (12, 2)].map(|(time, marker)| Stamp {
                time,
                marker,
                step: 4,
            });

        let time = class_time(&instance_classes, &mut stamps);

        assert_eq!(time, BTreeMap::from([(1, 10), (2, 3)]));
    }

    /// The markers a capture places around a module run at every replay: each step between its
    /// step markers counts once in the histogram of its stage and the module's class, and once in
    /// the stage's own.
    #[test]
    #[ignore = "requires a GPU"]
    fn a_captured_range_times_its_module_at_every_replay() {
        let device = DeviceId(0);
        let markers = Markers::install(&[device], vec!["Mlp".to_owned()], true, &[]).unwrap();
        let stream = Stream::new(device).unwrap();

        // A forward run once carries no marker; a capture's does.
        assert!(profiler(&stream).is_none());
        let capture = platform::Capture::begin(&stream, platform::CaptureMode::Relaxed).unwrap();
        let captured = profiler(&stream).unwrap();

        let instance = captured
            .enter("Mlp", "layers.0.mlp", &stream)
            .unwrap()
            .unwrap();
        // No pattern of the deployment names this class.
        let untimed = captured
            .enter("Attention", "layers.0.attention", &stream)
            .unwrap();
        assert_eq!(untimed, None);
        captured.exit(instance, &stream).unwrap();

        let graph = capture.end().unwrap();
        let executable = graph.instantiate().unwrap();
        for step in [7, 8] {
            markers.begin_step(&stream, step, "decode").unwrap();
            executable.launch(&stream).unwrap();
            markers.end_step(&stream, step).unwrap();
        }
        stream.synchronize().unwrap();
        markers.read_device(device, true);

        let (histograms, lost) = markers.take(device);
        let steps = |name: &str| histograms[name].buckets.iter().sum::<u64>();
        assert_eq!(lost, 0);
        assert_eq!(steps("decode/step"), 2);
        assert_eq!(steps("decode/Mlp"), 2);
        assert_eq!(
            markers.instances(),
            [("Mlp".to_owned(), "layers.0.mlp".to_owned())]
        );
    }
}
