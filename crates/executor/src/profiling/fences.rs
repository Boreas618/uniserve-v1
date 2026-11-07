//! Fence probes: device timestamps at the entry and exit of the model's components, always on,
//! exported as per-step component durations.
//!
//! **Where.** A model states each of its modules with [`nn::Forward::fenced`]: its component, which
//! names its class, and its path. Every forward a runner records into a graph carries this rank's
//! [`Probes`], so a module whose component matches `profiling_config.fence_components` gets one
//! probe as it is entered and one as it returns. Each is a launch of [`PTX`]'s single-thread kernel
//! on the forward's stream, so the graph captures it and every replay runs it. The device thread
//! brackets each batch's execution with a step marker of the same kernel on the runner's stream,
//! which records the batch and stores its id in the stream's step cell; the probes the replay runs
//! read the cell, so every record names its step.
//!
//! **What a probe does.** It reads the device switch: a word per component and a master word, in
//! device storage. A probe whose words are off returns at once, so switching probes needs no capture
//! again. A probe that is on reads `%globaltimer`, takes the next sequence number from a device
//! counter, and writes a 16-byte record (time, probe, step and the epoch tag below) into its slot of
//! a double buffer in pinned host storage the device writes through unified addressing. Sequence
//! number `s` writes slot `s mod N` of half `(s / N) mod 2`, `N` records a half; the epoch `s / N` is
//! tagged in the record (12 bits). The probe that completes a half publishes the half's epoch plus
//! one in the half's `filled` word.
//!
//! **Reading.** Each device thread reads its device's halves after every batch ([`Fences::poll`]),
//! without any device synchronization, and the control thread does so with every metrics report, so
//! an idle device's last records are not left behind. A half is read once its `filled` word shows it
//! complete, and at most once a [`PARTIAL_READ`] so are the records of the half being filled, so
//! steps are reported while the probe rate is low. A record is taken only when its tag is that of the
//! epoch read: a record not yet visible is read again later, and one the device already overwrote
//! with a later epoch (the host fell two halves behind) is counted as lost.
//!
//! **Clocks.** `%globaltimer` is a nanosecond clock of the GPU. A calibration launches a kernel that
//! stores `%globaltimer` into pinned host storage and times its appearance on the host
//! ([`Fences::calibrate`]); the sample of least uncertainty aligns the GPU's clock to the host's
//! `CLOCK_REALTIME` and to CUPTI's clock ([`Alignment`]).
//!
//! **Export.** At a step's end marker, its records are sorted by time and each component instance's
//! entry and exit are paired; the durations of a component's instances in the step are summed, and
//! the sum enters the histogram of the step's stage and the component, as does the step's own span
//! between its markers (component `step`), which travel with the next batch result
//! (`ForwardStats::fence_us`).
//!
//! **One process, many ranks.** The instances are the process's: every rank of a block registers
//! the same ones, and a rank of another block adds its own. Everything a device writes or a reader
//! keeps (the buffers, the step cells, the steps not yet ended, the histograms) is that device's,
//! because two ranks of one group run one batch id at once.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use ipc::FenceHistogram;
use platform::{
    Device, DeviceAllocation, DeviceId, Function, LaunchConfig, Library, PinnedBuffer, Stream,
};

use crate::{Error, Result, lock};

/// The probe kernel and the clock kernel, in PTX the driver compiles for the
/// device. Parameters of `uniserve_fence`:
///
/// - `switches`: `u32[1 + components]`, the master word then a word per
///   component;
/// - `component`: the probe's word, 0 for a step marker (on with the master
///   word);
/// - `probe`: the probe's id (20 bits);
/// - `cell`: the stream's step cell (`u32`);
/// - `mark`, `step`: a marker stores `step` into the cell; a probe
///   (`mark == 0`) reads it;
/// - `counter`: the `u64` sequence counter; `written`: `u32[2]`, the records
///   written into each half's current epoch;
/// - `records`: `16 * 2 * N` bytes of pinned host storage; `filled`:
///   `u64[2]` of pinned host storage;
/// - `shift`: `log2 N`.
const PTX: &str = r"
.version 8.0
.target sm_90
.address_size 64

.visible .entry uniserve_fence(
    .param .u64 switches,
    .param .u32 component,
    .param .u32 probe,
    .param .u64 cell,
    .param .u32 mark,
    .param .u32 step,
    .param .u64 counter,
    .param .u64 written,
    .param .u64 records,
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
    ld.param.u32 %r2, [component];
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
    ld.param.u64 %rd14, [records];
    add.u64 %rd15, %rd14, %rd13;
    ld.param.u32 %r7, [probe];
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

/// `log2` of the records a half holds: 4096 records, 64 KiB.
const SHIFT: u32 = 12;

/// Records a half holds.
const HALF: u64 = 1 << SHIFT;

/// The tag bits of a record's epoch.
const TAG_MASK: u64 = 0xfff;

/// The most components one process's probes name: the words of the device switch. A model states a
/// handful of classes, and a capture that names more is refused rather than left unswitchable.
const COMPONENTS: usize = 1024;

/// The next free step cell of a device whose streams hold `cells`, or the shared cell 0 once every
/// other is taken.
fn next_cell(cells: &HashMap<usize, usize>) -> usize {
    let next = cells.values().copied().max().unwrap_or(0) + 1;
    if next < CELLS { next } else { 0 }
}

/// The probe ids of the step markers.
pub(crate) const STEP_BEGIN: u32 = 0xf_fffe;
pub(crate) const STEP_END: u32 = 0xf_ffff;

/// The step cells a device keeps: one per stream a runner computes on, and cell 0, shared once every
/// other is taken.
const CELLS: usize = 64;

/// How often the half being filled is read, at most.
pub(crate) const PARTIAL_READ: Duration = Duration::from_secs(1);

/// How long a complete half's records that are not visible yet are waited for before they are
/// counted as lost.
const VISIBILITY: Duration = Duration::from_millis(50);

/// How long a calibration spins for its timestamp on the host before it blocks until the clock
/// kernel completes: on a busy device the kernel queues behind the device's work for longer.
const CALIBRATION_SPIN: Duration = Duration::from_micros(500);

/// The steps whose end marker has not been read that a device keeps; older ones (a batch that failed
/// between its markers) are dropped.
const OPEN_STEPS: usize = 4096;

/// One probe pair's module instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Instance {
    /// The module's path in the model.
    pub(crate) path: String,
    /// Its component.
    pub(crate) component: String,
}

/// One record read from a device's buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FenceRecord {
    /// `%globaltimer`, nanoseconds.
    pub(crate) time: u64,
    /// `2 * instance` at its entry, `2 * instance + 1` at its exit, or a step marker.
    pub(crate) probe: u32,
    /// The step's batch id, low 32 bits.
    pub(crate) step: u32,
}

/// The alignment of a device's `%globaltimer` to the host's clocks, from the calibration sample of
/// least uncertainty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Alignment {
    pub(crate) device: i32,
    /// `%globaltimer` at the sample.
    pub(crate) gpu: u64,
    /// The host's `CLOCK_REALTIME` and CUPTI's clock at the sample's midpoint, nanoseconds.
    pub(crate) realtime: u64,
    pub(crate) cupti: u64,
    /// Half the time the sample took on the host: the alignment's bound.
    pub(crate) uncertainty: u64,
}

/// A step's records and what the rank said of it.
#[derive(Default)]
struct OpenStep {
    records: Vec<(u64, u32)>,
    begin: Option<u64>,
}

/// A device's probe resources: the kernels, the switch, the counter, the step cells and the double
/// buffer, and the reader's position in it.
struct DeviceFences {
    device: DeviceId,
    _library: Library,
    fence: Function,
    clock: Function,
    /// `u32[1 + COMPONENTS]`.
    switches: DeviceAllocation,
    /// The `u64` counter, then `u32[2]` written counts.
    counters: DeviceAllocation,
    /// `u32[CELLS]`.
    cells: DeviceAllocation,
    /// `[u64 time, u64 packed][2 * HALF]`.
    records: PinnedBuffer,
    /// `u64[2]`.
    filled: PinnedBuffer,
    /// `u64`, the clock kernel's output.
    clock_out: PinnedBuffer,
    /// The stream the switch writes and calibrations go on.
    control: Stream,
    /// The step cell of each stream that marked or probed a step.
    stream_cells: Mutex<HashMap<usize, usize>>,
    reader: Mutex<Reader>,
    /// The stage of each step whose begin marker was launched.
    stages: Mutex<HashMap<u32, Arc<str>>>,
    /// Histograms of completed steps since the last take, by `{stage}/{component}`.
    histograms: Mutex<BTreeMap<String, FenceHistogram>>,
    /// Where the records read go while a session records them.
    tap: Mutex<Option<Tap>>,
}

/// Where a device's reader stands.
#[derive(Default)]
struct Reader {
    /// The next sequence number to take.
    next: u64,
    /// When a complete half first showed records not yet visible.
    waiting_since: Option<Instant>,
    /// Records counted as lost since the last take.
    lost: u64,
    /// When the half being filled was last read.
    partial_read: Option<Instant>,
    steps: BTreeMap<u32, OpenStep>,
    best: Option<Alignment>,
}

/// The instances the process's captures named, and their components.
#[derive(Default)]
struct Instances {
    instances: Vec<Instance>,
    /// Each instance's number, by component and path.
    numbers: HashMap<(String, String), u32>,
    /// Component names, by component index from 1.
    components: Vec<String>,
    /// Each instance's component index.
    instance_components: Vec<u32>,
    /// Whether each component a pattern names, by name.
    probed: HashMap<&'static str, bool>,
}

/// What the probes of this process's devices measure.
pub struct Fences {
    devices: Vec<DeviceFences>,
    /// `profiling_config.fence_components`.
    patterns: Vec<String>,
    instances: Mutex<Instances>,
    /// The switch as last stated: the master word and the patterns switched off.
    switched: Mutex<(bool, Vec<String>)>,
    /// The master word, read at every step: a marker with the switch off records nothing, so it is
    /// not launched.
    recording: AtomicBool,
}

/// A session's subscription to the records read: the components it keeps (empty for all) and where
/// they go.
pub(crate) struct Tap {
    pub(crate) components: Vec<String>,
    pub(crate) sink: Box<dyn Fn(i32, Vec<FenceRecord>) + Send>,
}

static FENCES: OnceLock<Arc<Fences>> = OnceLock::new();

/// A rank's half of [`nn::Fence`]: what a forward it records carries.
pub struct Probes {
    fences: Arc<Fences>,
}

/// The probes a forward on `stream` carries: those of the process's fences while the stream
/// captures, because a graph is what runs them at every step, and none for a forward run once.
pub fn capturing(stream: &Stream) -> Option<Probes> {
    let fences = FENCES.get()?;
    let recording = stream
        .capture_status()
        .is_ok_and(|status| status != platform::CaptureStatus::None);
    recording.then(|| Probes {
        fences: Arc::clone(fences),
    })
}

impl nn::Fence for Probes {
    fn enter(
        &self,
        component: &'static str,
        path: std::fmt::Arguments<'_>,
        stream: &Stream,
    ) -> nn::Result<Option<u32>> {
        let Some(instance) = self.fences.instance(component, path).map_err(fence_error)? else {
            return Ok(None);
        };
        self.fences
            .probe(stream, 2 * instance)
            .map_err(fence_error)?;
        Ok(Some(instance))
    }

    fn exit(&self, instance: u32, stream: &Stream) -> nn::Result<()> {
        self.fences
            .probe(stream, 2 * instance + 1)
            .map_err(fence_error)
    }
}

/// A probe's failure, as a forward's.
fn fence_error(error: Error) -> nn::Error {
    nn::Error::Shape {
        owner: "fence probes",
        problem: error.to_string(),
    }
}

impl Fences {
    /// Install the probes of the components `patterns` name on `devices`, every switch word on as
    /// `enabled` says but those of the components `off` names, as the process's fences. A forward
    /// recorded from here on carries them.
    ///
    /// # Errors
    ///
    /// Fails when a device cannot load the kernels or allocate the buffers, or the process installed
    /// fences already.
    pub fn install(
        devices: &[DeviceId],
        patterns: Vec<String>,
        enabled: bool,
        off: &[String],
    ) -> Result<Arc<Self>> {
        let devices = devices
            .iter()
            .map(|&device| DeviceFences::new(device))
            .collect::<Result<Vec<_>>>()?;
        let fences = Arc::new(Self {
            devices,
            patterns,
            instances: Mutex::default(),
            switched: Mutex::new((enabled, off.to_vec())),
            recording: AtomicBool::new(enabled),
        });
        fences.switch(enabled, off)?;
        FENCES
            .set(Arc::clone(&fences))
            .map_err(|_| Error::State("the process installed fence probes already"))?;
        Ok(fences)
    }

    /// The process's fences, once installed.
    pub fn installed() -> Option<&'static Arc<Self>> {
        FENCES.get()
    }

    /// The probed module instances, in their numbers' order.
    pub(crate) fn instances(&self) -> Vec<Instance> {
        lock(&self.instances).instances.clone()
    }

    /// The number of the instance at `path` of `component`, naming it on first sight, or `None`
    /// when no pattern of the deployment names the component.
    fn instance(
        &self,
        component: &'static str,
        path: std::fmt::Arguments<'_>,
    ) -> Result<Option<u32>> {
        let mut instances = lock(&self.instances);
        let patterns = &self.patterns;
        let probed = *instances
            .probed
            .entry(component)
            .or_insert_with(|| patterns.iter().any(|pattern| matches(pattern, component)));
        if !probed {
            return Ok(None);
        }
        let key = (component.to_owned(), path.to_string());
        if let Some(&number) = instances.numbers.get(&key) {
            return Ok(Some(number));
        }
        let number = u32::try_from(instances.instances.len())
            .ok()
            .filter(|number| 2 * *number < STEP_BEGIN)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "{} probed module instances exceed the probes' id space",
                    instances.instances.len()
                ))
            })?;
        let index = match instances
            .components
            .iter()
            .position(|known| known == component)
        {
            Some(index) => index,
            None => {
                if instances.components.len() == COMPONENTS {
                    return Err(Error::Invalid(format!(
                        "{COMPONENTS} probed components fill the device switch"
                    )));
                }
                instances.components.push(component.to_owned());
                let index = instances.components.len() - 1;
                let (enabled, off) = lock(&self.switched).clone();
                let (words, _) = switch_words(&instances.components, enabled, &off);
                self.write_switch(&words)?;
                index
            }
        };
        instances.instances.push(Instance {
            path: key.1.clone(),
            component: key.0.clone(),
        });
        instances.instance_components.push(index as u32 + 1);
        instances.numbers.insert(key, number);
        Ok(Some(number))
    }

    /// Turn the probes on or off: the master word as `enabled` says, and every component's word with
    /// it but those of the components matching a pattern of `off` (`*` matches any run), which stay
    /// off. Written on each device's control stream; probes already queued read whichever value
    /// reaches them. Returns the probed components switched off.
    ///
    /// # Errors
    ///
    /// When a device refuses the write.
    pub fn switch(&self, enabled: bool, off: &[String]) -> Result<Vec<String>> {
        let instances = lock(&self.instances);
        *lock(&self.switched) = (enabled, off.to_vec());
        self.recording.store(enabled, Ordering::Release);
        let (words, switched_off) = switch_words(&instances.components, enabled, off);
        self.write_switch(&words)?;
        Ok(switched_off)
    }

    fn write_switch(&self, words: &[u32]) -> Result<()> {
        for device in &self.devices {
            for (index, &word) in words.iter().enumerate() {
                // SAFETY: the switch holds `1 + COMPONENTS` words, and lives as long as the fences,
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

    /// The step cell of the stream of handle `stream`: the one it was first given, or a new one, or
    /// the shared cell once every cell is taken. A marker and a probe on the same stream resolve the
    /// same cell whichever comes first, so a probe a capture records before the stream's first step
    /// reads the steps its markers write.
    fn cell(device: &DeviceFences, stream: usize) -> usize {
        let mut cells = lock(&device.stream_cells);
        if let Some(&cell) = cells.get(&stream) {
            return cell;
        }
        let cell = next_cell(&cells);
        cells.insert(stream, cell);
        cell
    }

    fn device(&self, device: DeviceId) -> Result<&DeviceFences> {
        self.devices
            .iter()
            .find(|fences| fences.device == device)
            .ok_or_else(|| Error::Invalid(format!("no fence probes on {device}")))
    }

    /// Launch probe `probe` (`2 * instance + site`) on `stream`; captured when the stream captures.
    fn probe(&self, stream: &Stream, probe: u32) -> Result<()> {
        let component = *lock(&self.instances)
            .instance_components
            .get((probe / 2) as usize)
            .ok_or_else(|| {
                Error::Invalid(format!("probe {probe} names no probed module instance"))
            })?;
        let fences = self.device(stream.device())?;
        let cell = Self::cell(fences, stream.raw());
        fences.launch(stream, component, probe, cell, None)?;
        // A session's range profiler pushes the instance's range at its entry and pops it at its
        // exit; a capture pushes none, the push being a host call no replay repeats.
        if super::collectors::component_ranges_armed() {
            let capturing = stream
                .capture_status()
                .map_or(true, |status| status != platform::CaptureStatus::None);
            super::collectors::component_range(stream.device(), probe, capturing);
        }
        Ok(())
    }

    /// Launch the begin or end marker of step `step` (a batch id) of stage `stage` on `stream`, a
    /// runner's computation stream.
    pub(crate) fn mark(&self, stream: &Stream, step: u64, stage: &str, begin: bool) -> Result<()> {
        if !self.recording.load(Ordering::Acquire) {
            return Ok(());
        }
        let fences = self.device(stream.device())?;
        let cell = Self::cell(fences, stream.raw());
        let step = step as u32;
        if begin {
            let mut stages = lock(&fences.stages);
            // A step whose end marker was never read leaves its stage here.
            if stages.len() >= OPEN_STEPS {
                stages.clear();
            }
            stages.insert(step, Arc::from(stage));
        }
        let probe = if begin { STEP_BEGIN } else { STEP_END };
        fences.launch(stream, 0, probe, cell, Some(step))
    }

    /// Have the records read from `device` from now on also reach `tap`, until [`Fences::untap`].
    pub(crate) fn tap(&self, device: DeviceId, tap: Tap) -> Result<()> {
        *lock(&self.device(device)?.tap) = Some(tap);
        Ok(())
    }

    pub(crate) fn untap(&self, device: DeviceId) {
        if let Ok(fences) = self.device(device) {
            *lock(&fences.tap) = None;
        }
    }

    /// Read every device's halves ([`Self::poll_device`]).
    pub fn poll(&self, partial: bool) {
        for device in &self.devices {
            self.poll_of(device, partial);
        }
    }

    /// Read `device`'s complete halves, and the half being filled when [`PARTIAL_READ`] passed since
    /// its last read or `partial` asks; aggregate the steps that ended. Never waits for the device.
    pub(crate) fn poll_device(&self, device: DeviceId, partial: bool) {
        if let Ok(fences) = self.device(device) {
            self.poll_of(fences, partial);
        }
    }

    fn poll_of(&self, device: &DeviceFences, partial: bool) {
        let records = device.read(partial);
        if records.is_empty() {
            return;
        }
        self.aggregate(device, &records);
        if let Some(tap) = lock(&device.tap).as_ref() {
            let kept: Vec<FenceRecord> = records
                .into_iter()
                .filter(|record| self.kept(tap, record.probe))
                .collect();
            if !kept.is_empty() {
                (tap.sink)(device.device.0, kept);
            }
        }
    }

    /// Whether a session tap keeps records of `probe`.
    fn kept(&self, tap: &Tap, probe: u32) -> bool {
        if tap.components.is_empty() || probe >= STEP_BEGIN {
            return true;
        }
        let instances = lock(&self.instances);
        let component = instances.instance_components[(probe / 2) as usize] as usize;
        let name = &instances.components[component - 1];
        tap.components.iter().any(|pattern| matches(pattern, name))
    }

    /// Add the steps whose end marker `records` read to the device's histograms.
    fn aggregate(&self, device: &DeviceFences, records: &[FenceRecord]) {
        let mut reader = lock(&device.reader);
        let mut ended = Vec::new();
        for record in records {
            let step = reader.steps.entry(record.step).or_default();
            match record.probe {
                STEP_BEGIN => step.begin = Some(record.time),
                STEP_END => ended.push((record.step, record.time)),
                probe => step.records.push((record.time, probe)),
            }
        }
        let mut completed = Vec::new();
        for (step, end) in ended {
            if let Some(open) = reader.steps.remove(&step) {
                completed.push((step, end, open));
            }
        }
        while reader.steps.len() > OPEN_STEPS {
            reader.steps.pop_first();
        }
        drop(reader);

        let mut stages = lock(&device.stages);
        let mut histograms = lock(&device.histograms);
        let instances = lock(&self.instances);
        for (step, end, open) in completed {
            let stage = stages.remove(&step).unwrap_or_else(|| Arc::from("unknown"));
            if let Some(begin) = open.begin {
                histograms
                    .entry(format!("{stage}/step"))
                    .or_default()
                    .observe(end.saturating_sub(begin) / 1000);
            }
            for (component, ns) in component_time(&instances.instance_components, open.records) {
                histograms
                    .entry(format!("{stage}/{}", instances.components[component - 1]))
                    .or_default()
                    .observe(ns / 1000);
            }
        }
    }

    /// The histograms of `device`'s steps completed since the last take, and its records lost.
    pub(crate) fn take(&self, device: DeviceId) -> (BTreeMap<String, FenceHistogram>, u64) {
        let Ok(fences) = self.device(device) else {
            return Default::default();
        };
        let lost = std::mem::take(&mut lock(&fences.reader).lost);
        (std::mem::take(&mut *lock(&fences.histograms)), lost)
    }

    /// Calibrate `device`'s clock against the host's: answers its best alignment so far.
    pub(crate) fn calibrate(&self, device: DeviceId) -> Option<Alignment> {
        let fences = self.device(device).ok()?;
        match fences.calibrate() {
            Ok(alignment) => Some(alignment),
            Err(error) => {
                tracing::warn!("fence clock calibration on {device} failed: {error}");
                lock(&fences.reader).best
            }
        }
    }
}

/// Each component's summed instance durations in one step's records, by component index: each
/// instance's entries and exits pair in time order.
fn component_time(
    instance_components: &[u32],
    mut records: Vec<(u64, u32)>,
) -> BTreeMap<usize, u64> {
    records.sort_unstable();
    let mut entered: HashMap<u32, Vec<u64>> = HashMap::new();
    let mut time: BTreeMap<usize, u64> = BTreeMap::new();
    for (at, probe) in records {
        let instance = probe / 2;
        if probe % 2 == 0 {
            entered.entry(instance).or_default().push(at);
        } else if let Some(start) = entered.get_mut(&instance).and_then(Vec::pop) {
            let Some(&component) = instance_components.get(instance as usize) else {
                continue;
            };
            *time.entry(component as usize).or_default() += at.saturating_sub(start);
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

impl DeviceFences {
    fn new(device: DeviceId) -> Result<Self> {
        let handle = Device::new(device)?;
        // The driver compiles the text for the device, and it needs the terminating NUL a text
        // image is read to.
        let mut image = PTX.as_bytes().to_vec();
        image.push(0);
        let library = Library::load(&handle, &image)?;
        let fence = library.kernel("uniserve_fence")?;
        let clock = library.kernel("uniserve_clock")?;
        let control = Stream::with_priority(device, -5)?;
        let switches = DeviceAllocation::linear(device, 4 * (1 + COMPONENTS))?;
        let counters = DeviceAllocation::linear(device, 16)?;
        let cells = DeviceAllocation::linear(device, 4 * CELLS)?;
        let mut records = PinnedBuffer::new(device, (16 * 2 * HALF) as usize)?;
        let mut filled = PinnedBuffer::new(device, 16)?;
        let mut clock_out = PinnedBuffer::new(device, 8)?;
        // Every slot starts tagged with an epoch no write of the first two epochs carries.
        // SAFETY: no device work addresses the new buffers yet.
        unsafe {
            for slot in records.as_mut_slice().as_chunks_mut::<16>().0 {
                slot[..8].fill(0);
                slot[8..].copy_from_slice(&(TAG_MASK << 20).to_le_bytes());
            }
            filled.as_mut_slice().fill(0);
            clock_out.as_mut_slice().fill(0);
            platform::memset_d32(switches.address(), 0, 1 + COMPONENTS, &control)?;
            platform::memset_d32(counters.address(), 0, 4, &control)?;
            platform::memset_d32(cells.address(), 0, CELLS, &control)?;
        }
        control.synchronize()?;
        Ok(Self {
            device,
            _library: library,
            fence,
            clock,
            switches,
            counters,
            cells,
            records,
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

    /// Launch the fence kernel on `stream`: a probe of switch word `component`, or with `step` a
    /// marker.
    fn launch(
        &self,
        stream: &Stream,
        component: u32,
        probe: u32,
        cell: usize,
        step: Option<u32>,
    ) -> Result<()> {
        let mut switches = self.switches.address();
        let mut component = component;
        let mut probe = probe;
        let mut cell = self.cells.address() + 4 * cell as u64;
        let mut mark = u32::from(step.is_some());
        let mut step = step.unwrap_or(0);
        let mut counter = self.counters.address();
        let mut written = self.counters.address() + 8;
        let mut records = self.records.address();
        let mut filled = self.filled.address();
        let mut shift = SHIFT;
        let mut parameters: [*mut c_void; 11] = [
            (&raw mut switches).cast(),
            (&raw mut component).cast(),
            (&raw mut probe).cast(),
            (&raw mut cell).cast(),
            (&raw mut mark).cast(),
            (&raw mut step).cast(),
            (&raw mut counter).cast(),
            (&raw mut written).cast(),
            (&raw mut records).cast(),
            (&raw mut filled).cast(),
            (&raw mut shift).cast(),
        ];
        // The parameters are the kernel's, and every buffer they address lives with the process's
        // fences, past every replay of a graph that captured the launch.
        self.fence
            .launch_ex(stream, &single_thread(), &mut parameters)?;
        Ok(())
    }

    /// The `filled` word of `half`.
    fn filled(&self, half: u64) -> u64 {
        // SAFETY: the word is pinned host storage the device writes whole; a volatile read observes
        // its latest value.
        unsafe { std::ptr::read_volatile(self.filled.as_ptr().cast::<u64>().add(half as usize)) }
    }

    /// The record of sequence number `sequence`'s slot.
    fn slot(&self, sequence: u64) -> (u64, u64) {
        let index = ((sequence >> SHIFT) & 1) * HALF + (sequence & (HALF - 1));
        // SAFETY: the index lies in the buffer; the device writes each record whole in one 16-byte
        // store, read here as two volatile words.
        unsafe {
            let base = self.records.as_ptr().cast::<u64>().add(2 * index as usize);
            (
                std::ptr::read_volatile(base),
                std::ptr::read_volatile(base.add(1)),
            )
        }
    }

    /// Take the records the device completed: the complete halves in epoch order, then, when due,
    /// the visible records of the half being filled.
    fn read(&self, partial: bool) -> Vec<FenceRecord> {
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
            // A complete half: every record past the reader's position.
            let mut pending = false;
            while reader.next >> SHIFT == epoch {
                match record(self.slot(reader.next), epoch) {
                    Some(record) => taken.push(record),
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
                match record(self.slot(reader.next), epoch) {
                    Some(record) => taken.push(record),
                    None => break,
                }
                reader.next += 1;
            }
        }
        taken
    }

    /// One calibration sample: the clock kernel's `%globaltimer` against the host clocks around its
    /// appearance. Keeps and returns the best sample.
    ///
    /// The sample's uncertainty is half the host's window from before the launch to the value's
    /// appearance: a kernel that queued behind the device's work gives a wide sample, which a
    /// narrower one replaces.
    fn calibrate(&self) -> Result<Alignment> {
        // SAFETY: the word is pinned host storage no device work addresses now: a calibration returns
        // only once its kernel has stored into it.
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
                // calibration, which zeroes the word, from reading this kernel's late store as its
                // own.
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
        let sample = Alignment {
            device: self.device.0,
            gpu,
            realtime: before.0 / 2 + after.0 / 2,
            cupti: before.1 / 2 + after.1 / 2,
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

/// The epoch tag of a record's packed word.
fn tag(packed: u64) -> u64 {
    (packed >> 20) & TAG_MASK
}

/// The record `(time, packed)` if it carries `epoch`'s tag.
fn record((time, packed): (u64, u64), epoch: u64) -> Option<FenceRecord> {
    (tag(packed) == epoch & TAG_MASK).then_some(FenceRecord {
        time,
        probe: (packed & 0xf_ffff) as u32,
        step: (packed >> 32) as u32,
    })
}

/// The switch words of `components` (by component index from 1): the master word, on as `enabled`
/// says, then each component's, on with it unless the component matches a pattern of `off`; and the
/// components matched.
fn switch_words(components: &[String], enabled: bool, off: &[String]) -> (Vec<u32>, Vec<String>) {
    let mut words = vec![u32::from(enabled)];
    let mut switched_off = Vec::new();
    for component in components {
        let excluded = off.iter().any(|pattern| matches(pattern, component));
        if excluded {
            switched_off.push(component.clone());
        }
        words.push(u32::from(enabled && !excluded));
    }
    (words, switched_off)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The master word follows the switch; a component's word follows it unless a pattern of the off
    /// list names the component, whose probes stay off while the others record.
    #[test]
    fn a_component_named_off_stays_off_while_the_others_record() {
        let components = [
            "models.qwen3.transformer.Attention".to_owned(),
            "models.qwen3.transformer.Transformer".to_owned(),
        ];
        assert_eq!(
            switch_words(&components, true, &["*Attention".to_owned()]),
            (vec![1, 0, 1], vec![components[0].clone()])
        );
        assert_eq!(
            switch_words(&components, true, &[]),
            (vec![1, 1, 1], Vec::new())
        );
        assert_eq!(
            switch_words(&components, false, &["*Attention".to_owned()]),
            (vec![0, 0, 0], vec![components[0].clone()])
        );
    }

    #[test]
    fn a_pattern_star_matches_any_run() {
        assert!(matches("*", "nn.linear.Linear"));
        assert!(matches("nn.linear.*", "nn.linear.QKVParallelLinear"));
        assert!(matches("*Attention", "models.qwen3.transformer.Attention"));
        assert!(matches(
            "models.*.Decoder*",
            "models.qwen3.transformer.DecoderLayer"
        ));
        assert!(!matches("nn.linear.*", "nn.norm.RMSNorm"));
        assert!(matches("nn.norm.RMSNorm", "nn.norm.RMSNorm"));
        assert!(!matches("nn.norm.RMS", "nn.norm.RMSNorm"));
    }

    #[test]
    fn a_record_is_taken_only_with_its_epochs_tag() {
        let packed =
            |probe: u64, epoch: u64, step: u64| probe | ((epoch & TAG_MASK) << 20) | (step << 32);
        let taken = record((5, packed(7, 3, 42)), 3).unwrap();
        assert_eq!((taken.time, taken.probe, taken.step), (5, 7, 42));
        // Written two epochs later, or not yet written for this one.
        assert!(record((5, packed(7, 5, 42)), 3).is_none());
        assert!(record((0, packed(0, TAG_MASK, 0)), 0).is_none());
    }

    #[test]
    fn nested_instances_each_sum_their_own_span() {
        let instance_components = [1, 2];
        let time = component_time(
            &instance_components,
            vec![(10, 0), (12, 2), (15, 3), (20, 1)],
        );
        assert_eq!(time.get(&1), Some(&10));
        assert_eq!(time.get(&2), Some(&3));
    }
}
