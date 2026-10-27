//! Fence probes: device timestamps at the entry and exit of the model's
//! components, always on, exported as per-step component durations.
//!
//! **Where.** The numerical library attaches a probe pair to every module
//! whose component, the `<component>` of the function its class declares
//! (`uniserve.numerics`), matches `profiling_config.fence_components`
//! (`uniserve.profiling.attach_fences`): one probe as its forward is
//! entered and one as it returns. Each is a launch of [`PTX`]'s
//! single-thread kernel on the module's current stream, through the
//! library's exported `fence_probe`; a forward captured into a CUDA graph at
//! warmup captures its probes, which every replay then runs. The worker
//! brackets each batch's execution with a step marker of the same kernel on
//! the computation's stream, which records the batch and stores its id in
//! the stream's step cell; the probes the batch's work runs read the cell,
//! so every record names its step.
//!
//! **What a probe does.** It reads the device switch: a word per component
//! and a master word, in device memory. A probe whose words are off
//! returns at once, so switching probes needs no capture again. A probe
//! that is on reads `%globaltimer`, takes the next sequence number from a
//! device counter, and writes a 16-byte record (time, probe, step and the
//! epoch tag below) into its slot of a double buffer in pinned host memory
//! the device writes through unified addressing. Sequence number `s` writes
//! slot `s mod N` of half `(s / N) mod 2`, `N` records a half; the epoch
//! `s / N` is tagged in the record (12 bits). The probe that completes a
//! half publishes the half's epoch plus one in the half's `filled` word.
//!
//! **Reading.** The control thread reads a half once its `filled` word shows
//! it complete ([`Fences::poll`]), without any device synchronization, and
//! also, at most once a [`PARTIAL_READ`], the records of the half being
//! filled, so steps are reported while the probe rate is low. A record is
//! taken only when its tag is that of the epoch read: a record not yet
//! visible is read again later, and one the device already overwrote with a
//! later epoch (the host fell two halves behind) is counted as lost.
//!
//! **Clocks.** `%globaltimer` is a nanosecond clock of the GPU. A
//! calibration launches a kernel that stores `%globaltimer` into pinned host
//! memory and times its appearance on the host ([`Fences::calibrate`]); the
//! sample of least uncertainty aligns the GPU's clock to the host's
//! `CLOCK_REALTIME` and to CUPTI's clock ([`Alignment`]).
//!
//! **Export.** At a step's end marker, its records are sorted by time and
//! each component instance's entry and exit are paired; the durations of a
//! component's instances in the step are summed, and the sum enters the
//! histogram of the step's stage and the component, as does the step's own
//! span between its markers (component `step`), which travel with the next
//! batch result (`ForwardStats::fence_us`).

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use cuda::{DeviceAllocation, DeviceId, Function, Module, PinnedBuffer, Stream};
use ipc::FenceHistogram;

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
/// - `records`: `16 * 2 * N` bytes of pinned host memory; `filled`:
///   `u64[2]` of pinned host memory;
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

/// The probe ids of the step markers.
pub(crate) const STEP_BEGIN: u32 = 0xf_fffe;
pub(crate) const STEP_END: u32 = 0xf_ffff;

/// The step cells a device keeps: cell 0 for streams no computation owns,
/// and one per computation stream.
const CELLS: usize = 64;

/// How often the half being filled is read, at most.
pub(crate) const PARTIAL_READ: Duration = Duration::from_secs(1);

/// How long a complete half's records that are not visible yet are waited
/// for before they are counted as lost.
const VISIBILITY: Duration = Duration::from_millis(50);

/// How long a calibration waits for its timestamp on the host.
const CALIBRATION_WAIT: Duration = Duration::from_micros(500);

/// The steps whose end marker has not been read that a device keeps; older
/// ones (a batch that failed between its markers) are dropped.
const OPEN_STEPS: usize = 4096;

/// One probe pair's module instance.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
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
    /// `2 * instance` at its entry, `2 * instance + 1` at its exit, or a
    /// step marker.
    pub(crate) probe: u32,
    /// The step's batch id, low 32 bits.
    pub(crate) step: u32,
}

/// The alignment of a device's `%globaltimer` to the host's clocks, from the
/// calibration sample of least uncertainty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Alignment {
    pub(crate) device: i32,
    /// `%globaltimer` at the sample.
    pub(crate) gpu: u64,
    /// The host's `CLOCK_REALTIME` and CUPTI's clock at the sample's
    /// midpoint, nanoseconds.
    pub(crate) realtime: u64,
    pub(crate) cupti: u64,
    /// Half the time the sample took on the host: the alignment's bound.
    pub(crate) uncertainty: u64,
}

/// A step's records and what the worker said of it.
#[derive(Default)]
struct OpenStep {
    records: Vec<(u64, u32)>,
    begin: Option<u64>,
}

/// A device's probe resources: the kernels, the switch, the counter, the
/// step cells and the double buffer, and the reader's position in it.
struct DeviceFences {
    device: DeviceId,
    _module: Module,
    fence: Function,
    clock: Function,
    /// `u32[1 + components]`.
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
    /// Each computation stream's step cell.
    stream_cells: Mutex<HashMap<usize, usize>>,
    reader: Mutex<Reader>,
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

/// What the probes of a rank's devices measure.
pub(crate) struct Fences {
    devices: Vec<DeviceFences>,
    instances: Vec<Instance>,
    /// Component names, by component index from 1.
    components: Vec<String>,
    /// Each instance's component index.
    instance_components: Vec<u32>,
    /// The stage of each step whose begin marker was launched.
    stages: Mutex<HashMap<u32, Arc<str>>>,
    /// Histograms of completed steps since the last take, by
    /// `{stage}/{component}`.
    histograms: Mutex<BTreeMap<String, FenceHistogram>>,
    /// Where the records read go while a session records them.
    tap: Mutex<Option<Tap>>,
}

/// A session's subscription to the records read: the components it keeps
/// (empty for all) and where they go.
pub(crate) struct Tap {
    pub(crate) components: Vec<String>,
    pub(crate) sink: Box<dyn Fn(i32, Vec<FenceRecord>) + Send>,
}

static FENCES: OnceLock<Arc<Fences>> = OnceLock::new();

impl Fences {
    /// Install the probes of `instances` on `devices`, every switch word on
    /// as `enabled` says, as the process's fences. Every probe the library
    /// launches from here on reaches them.
    ///
    /// # Errors
    ///
    /// Fails when a device cannot load the kernels or allocate the
    /// buffers, or the process installed fences already.
    pub(crate) fn install(
        devices: &[DeviceId],
        instances: Vec<Instance>,
        enabled: bool,
    ) -> Result<Arc<Self>> {
        let mut components: Vec<String> = Vec::new();
        let mut instance_components = Vec::with_capacity(instances.len());
        for instance in &instances {
            let index = match components
                .iter()
                .position(|name| *name == instance.component)
            {
                Some(index) => index,
                None => {
                    components.push(instance.component.clone());
                    components.len() - 1
                }
            };
            instance_components.push(index as u32 + 1);
        }
        if instances.len() * 2 >= STEP_BEGIN as usize {
            return Err(Error::Invalid(format!(
                "{} probed module instances exceed the probes' id space",
                instances.len()
            )));
        }
        let devices = devices
            .iter()
            .map(|&device| DeviceFences::new(device, components.len()))
            .collect::<Result<Vec<_>>>()?;
        let fences = Arc::new(Self {
            devices,
            instances,
            components,
            instance_components,
            stages: Mutex::new(HashMap::new()),
            histograms: Mutex::new(BTreeMap::new()),
            tap: Mutex::new(None),
        });
        fences.switch(enabled)?;
        FENCES
            .set(Arc::clone(&fences))
            .map_err(|_| Error::State("the process installed fence probes already"))?;
        Ok(fences)
    }

    /// The process's fences, once installed.
    pub(crate) fn installed() -> Option<&'static Arc<Self>> {
        FENCES.get()
    }

    /// The probed module instances.
    pub(crate) fn instances(&self) -> &[Instance] {
        &self.instances
    }

    /// The probed components, by component index from 1.
    /// Turn every probe on or off: the master word and every component's
    /// word, written on each device's control stream. Probes already
    /// queued read whichever value reaches them.
    pub(crate) fn switch(&self, enabled: bool) -> Result<()> {
        for device in &self.devices {
            let words = 1 + self.components.len();
            // SAFETY: the switch holds `words` words, and lives as long as
            // the fences, past the fill.
            unsafe {
                cuda::memset_d32(
                    device.switches.address(),
                    u32::from(enabled),
                    words,
                    &device.control,
                )?;
            }
        }
        Ok(())
    }

    /// The step cell of `stream`, a computation's stream the worker marks
    /// steps on: one of its own from its first marker on, or the shared cell
    /// once every cell is taken.
    fn marked_cell(device: &DeviceFences, stream: &Stream) -> usize {
        let mut cells = lock(&device.stream_cells);
        let next = cells.len() + 1;
        match cells.get(&stream.raw()) {
            Some(&cell) => cell,
            None if next < CELLS => {
                cells.insert(stream.raw(), next);
                next
            }
            None => 0,
        }
    }

    fn device(&self, device: DeviceId) -> Result<&DeviceFences> {
        self.devices
            .iter()
            .find(|fences| fences.device == device)
            .ok_or_else(|| Error::Invalid(format!("no fence probes on {device}")))
    }

    /// Launch probe `probe` (`2 * instance + site`) on the stream of handle
    /// `stream` on `device`; captured when the stream captures.
    ///
    /// The caller names the device: asking the driver for a stream's device
    /// (`cuStreamGetDevice`) during a global-mode capture is unsupported and
    /// invalidates the capture, while every call made here (`cuStreamGetCtx`,
    /// a context push and pop, `cuStreamIsCapturing`, `cuLaunchKernel`) is
    /// legal while the stream captures.
    ///
    /// # Safety
    /// `stream` is a live stream on `device`.
    pub(crate) unsafe fn probe(&self, stream: usize, device: DeviceId, probe: u32) -> Result<()> {
        let instance = (probe / 2) as usize;
        let component = *self.instance_components.get(instance).ok_or_else(|| {
            Error::Invalid(format!("probe {probe} names no probed module instance"))
        })?;
        let fences = self.device(device)?;
        // SAFETY: as above; the probe's own stream keeps its context.
        let borrowed = unsafe { Stream::borrowed(device, stream) }?;
        let cell = lock(&fences.stream_cells)
            .get(&stream)
            .copied()
            .unwrap_or(0);
        fences.launch(&borrowed, component, probe, cell, None)?;
        // A session's range profiler pushes the instance's range at its
        // entry and pops it at its exit; a capture pushes none, the push
        // being a host call no replay repeats.
        if crate::profiling::session::collectors::component_ranges_armed() {
            let capturing = borrowed
                .capture_status()
                .map_or(true, |status| status != cuda::CaptureStatus::None);
            crate::profiling::session::collectors::component_range(probe, capturing);
        }
        Ok(())
    }

    /// Launch the begin or end marker of step `step` (a batch id) of stage
    /// `stage` on `stream`, a registered computation stream.
    pub(crate) fn mark(&self, stream: &Stream, step: u64, stage: &str, begin: bool) -> Result<()> {
        let fences = self.device(stream.device())?;
        let cell = Self::marked_cell(fences, stream);
        let step = step as u32;
        if begin {
            lock(&self.stages).insert(step, Arc::from(stage));
        }
        let probe = if begin { STEP_BEGIN } else { STEP_END };
        fences.launch(stream, 0, probe, cell, Some(step))
    }

    /// Have the records read from now on also reach `tap`, until
    /// [`Fences::untap`].
    pub(crate) fn tap(&self, tap: Tap) {
        *lock(&self.tap) = Some(tap);
    }

    pub(crate) fn untap(&self) {
        *lock(&self.tap) = None;
    }

    /// Read every device's complete halves, and the half being filled when
    /// [`PARTIAL_READ`] passed since its last read or `partial` asks;
    /// aggregate the steps that ended. Never waits for the device.
    pub(crate) fn poll(&self, partial: bool) {
        for device in &self.devices {
            let records = device.read(partial);
            if records.is_empty() {
                continue;
            }
            self.aggregate(device, &records);
            if let Some(tap) = lock(&self.tap).as_ref() {
                let kept: Vec<FenceRecord> = records
                    .into_iter()
                    .filter(|record| self.kept(tap, record.probe))
                    .collect();
                if !kept.is_empty() {
                    (tap.sink)(device.device.0, kept);
                }
            }
        }
    }

    /// Whether a session tap keeps records of `probe`.
    fn kept(&self, tap: &Tap, probe: u32) -> bool {
        if tap.components.is_empty() || probe >= STEP_BEGIN {
            return true;
        }
        let component = self.instance_components[(probe / 2) as usize] as usize;
        let name = &self.components[component - 1];
        tap.components.iter().any(|pattern| matches(pattern, name))
    }

    /// Add the steps whose end marker `records` read to the histograms.
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

        let mut stages = lock(&self.stages);
        let mut histograms = lock(&self.histograms);
        for (step, end, open) in completed {
            let stage = stages.remove(&step).unwrap_or_else(|| Arc::from("unknown"));
            if let Some(begin) = open.begin {
                histograms
                    .entry(format!("{stage}/step"))
                    .or_default()
                    .observe(end.saturating_sub(begin) / 1000);
            }
            for (component, ns) in self.component_time(open.records) {
                histograms
                    .entry(format!("{stage}/{}", self.components[component - 1]))
                    .or_default()
                    .observe(ns / 1000);
            }
        }
    }

    /// Each component's summed instance durations in one step's records,
    /// by component index: each instance's entries and exits pair in time
    /// order.
    fn component_time(&self, mut records: Vec<(u64, u32)>) -> BTreeMap<usize, u64> {
        records.sort_unstable();
        let mut entered: HashMap<u32, Vec<u64>> = HashMap::new();
        let mut time: BTreeMap<usize, u64> = BTreeMap::new();
        for (at, probe) in records {
            let instance = probe / 2;
            if probe % 2 == 0 {
                entered.entry(instance).or_default().push(at);
            } else if let Some(start) = entered.get_mut(&instance).and_then(Vec::pop) {
                let Some(&component) = self.instance_components.get(instance as usize) else {
                    continue;
                };
                *time.entry(component as usize).or_default() += at.saturating_sub(start);
            }
        }
        time
    }

    /// The histograms of the steps completed since the last take, and the
    /// records lost.
    pub(crate) fn take(&self) -> (BTreeMap<String, FenceHistogram>, u64) {
        let lost = self
            .devices
            .iter()
            .map(|device| std::mem::take(&mut lock(&device.reader).lost))
            .sum();
        (std::mem::take(&mut *lock(&self.histograms)), lost)
    }

    /// Calibrate every device's clock against the host's: returns each
    /// device's best alignment so far.
    pub(crate) fn calibrate(&self) -> Vec<Alignment> {
        self.devices
            .iter()
            .filter_map(|device| match device.calibrate() {
                Ok(alignment) => Some(alignment),
                Err(error) => {
                    tracing::warn!(
                        "fence clock calibration on {} failed: {error}",
                        device.device
                    );
                    lock(&device.reader).best
                }
            })
            .collect()
    }
}

/// Whether `name` matches `pattern`, where `*` matches any run of
/// characters.
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
    fn new(device: DeviceId, components: usize) -> Result<Self> {
        let module = Module::load_ptx(device, PTX)?;
        let fence = module.function("uniserve_fence")?;
        let clock = module.function("uniserve_clock")?;
        let control = Stream::with_priority(device, -5)?;
        let switches = DeviceAllocation::linear(device, 4 * (1 + components))?;
        let counters = DeviceAllocation::linear(device, 16)?;
        let cells = DeviceAllocation::linear(device, 4 * CELLS)?;
        let mut records = PinnedBuffer::new(device, (16 * 2 * HALF) as usize)?;
        let mut filled = PinnedBuffer::new(device, 16)?;
        let mut clock_out = PinnedBuffer::new(device, 8)?;
        // Every slot starts tagged with an epoch no write of the first two
        // epochs carries.
        // SAFETY: no device work addresses the new buffers yet.
        unsafe {
            for slot in records.as_mut_slice().as_chunks_mut::<16>().0 {
                slot[..8].fill(0);
                slot[8..].copy_from_slice(&(TAG_MASK << 20).to_le_bytes());
            }
            filled.as_mut_slice().fill(0);
            clock_out.as_mut_slice().fill(0);
            cuda::memset_d32(counters.address(), 0, 4, &control)?;
            cuda::memset_d32(cells.address(), 0, CELLS, &control)?;
        }
        control.synchronize()?;
        Ok(Self {
            device,
            _module: module,
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
        })
    }

    /// Launch the fence kernel on `stream`: a probe of switch word
    /// `component`, or with `step` a marker.
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
        // SAFETY: the parameters are the kernel's, and every buffer they
        // address lives with the process's fences, past every replay of a
        // graph that captured the launch.
        unsafe { self.fence.launch(stream, 1, 1, &mut parameters) }?;
        Ok(())
    }

    /// The `filled` word of `half`.
    fn filled(&self, half: u64) -> u64 {
        // SAFETY: the word is pinned host memory the device writes whole;
        // a volatile read observes its latest value.
        unsafe { std::ptr::read_volatile(self.filled.as_ptr().cast::<u64>().add(half as usize)) }
    }

    /// The record of sequence number `sequence`'s slot.
    fn slot(&self, sequence: u64) -> (u64, u64) {
        let index = ((sequence >> SHIFT) & 1) * HALF + (sequence & (HALF - 1));
        // SAFETY: the index lies in the buffer; the device writes each record
        // whole in one 16-byte store, read here as two volatile words.
        unsafe {
            let base = self.records.as_ptr().cast::<u64>().add(2 * index as usize);
            (
                std::ptr::read_volatile(base),
                std::ptr::read_volatile(base.add(1)),
            )
        }
    }

    /// Take the records the device completed: the complete halves in epoch
    /// order, then, when due, the visible records of the half being filled.
    fn read(&self, partial: bool) -> Vec<FenceRecord> {
        let mut reader = lock(&self.reader);
        let mut taken = Vec::new();
        let completed = self.filled(0).max(self.filled(1));
        // The device is filling epoch `completed` (or a later one, before
        // its predecessor shows complete); a half two epochs behind it was
        // overwritten.
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

    /// One calibration sample: the clock kernel's `%globaltimer` against the
    /// host clocks around its appearance. Keeps and returns the best sample.
    fn calibrate(&self) -> Result<Alignment> {
        // SAFETY: the word is pinned host memory no device work addresses
        // now: the last calibration's kernel stored into it before its value
        // was observed.
        unsafe { std::ptr::write_volatile(self.clock_out.as_ptr().cast::<u64>(), 0) };
        let mut out = self.clock_out.address();
        let mut parameters: [*mut c_void; 1] = [(&raw mut out).cast()];
        let before = (realtime(), cuda::cupti::timestamp().unwrap_or(0));
        let started = Instant::now();
        // SAFETY: the parameter is the kernel's; the word outlives it.
        unsafe { self.clock.launch(&self.control, 1, 1, &mut parameters)? };
        let gpu = loop {
            // SAFETY: as above; the device writes the word whole.
            let value = unsafe { std::ptr::read_volatile(self.clock_out.as_ptr().cast::<u64>()) };
            if value != 0 {
                break value;
            }
            if started.elapsed() > CALIBRATION_WAIT {
                return Err(Error::State(
                    "the clock kernel did not run within the calibration's wait",
                ));
            }
            std::hint::spin_loop();
        };
        let after = (realtime(), cuda::cupti::timestamp().unwrap_or(0));
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

/// The library's probe entry, which `uniserve.profiling.attach_fences`
/// calls at a probed module's entry and exit with the current stream's
/// handle and device ordinal: launches probe `probe` on it. Fails when the
/// process installed no fences.
fn fence_probe(stream: i64, device: i64, probe: i64) -> tvm_ffi::Result<()> {
    let fences = Fences::installed().ok_or_else(|| {
        tvm_ffi::Error::new(
            tvm_ffi::RUNTIME_ERROR,
            "the process installed no fence probes",
            "",
        )
    })?;
    // SAFETY: the library passes the handle and device of its current
    // stream, which is live during the call.
    unsafe { fences.probe(stream as usize, DeviceId(device as i32), probe as u32) }
        .map_err(|error| tvm_ffi::Error::new(tvm_ffi::RUNTIME_ERROR, &error.to_string(), ""))
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(fence_probe, fence_probe);

#[cfg(test)]
mod tests {
    use super::*;

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
}

#[cfg(test)]
mod device_tests {
    use super::*;

    /// The probes of two instances, one inside the other, between a step's
    /// markers: the step's component durations reach the histograms, the
    /// records reach a tap of the inner component alone, probes captured
    /// into a CUDA graph in global mode (as PyTorch captures a forward)
    /// leave the capture valid and record at its replay, a switched-off
    /// probe records nothing, and the clocks align.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn probes_between_markers_time_their_step_by_component() -> Result<()> {
        let device = DeviceId(0);
        let instances = vec![
            Instance {
                path: "layers.0".into(),
                component: "models.m.TransformerLayer".into(),
            },
            Instance {
                path: "layers.0.attention".into(),
                component: "models.m.Attention".into(),
            },
        ];
        let fences = Fences::install(&[device], instances, true)?;
        let tapped: Arc<Mutex<Vec<FenceRecord>>> = Arc::default();
        fences.tap(Tap {
            components: vec!["*.Attention".into()],
            sink: Box::new({
                let tapped = Arc::clone(&tapped);
                move |_, records| lock(&tapped).extend(records)
            }),
        });
        let stream = Stream::new(device)?;
        stream.synchronize()?;
        fences.mark(&stream, 7, "decode", true)?;
        let forward = |stream: &Stream| -> Result<()> {
            // SAFETY: the stream is live on `device`.
            unsafe {
                fences.probe(stream.raw(), device, 0)?;
                fences.probe(stream.raw(), device, 2)?;
                fences.probe(stream.raw(), device, 3)?;
                fences.probe(stream.raw(), device, 1)?;
            }
            Ok(())
        };
        forward(&stream)?;
        fences.mark(&stream, 7, "decode", false)?;

        let graph = capture_globally(&stream, forward)?;
        fences.mark(&stream, 9, "decode", true)?;
        graph.launch(&stream)?;
        fences.mark(&stream, 9, "decode", false)?;
        stream.synchronize()?;
        fences.poll(true);
        let (histograms, lost) = fences.take();
        assert_eq!(lost, 0);
        for key in [
            "decode/step",
            "decode/models.m.TransformerLayer",
            "decode/models.m.Attention",
        ] {
            assert_eq!(
                histograms.get(key).map(FenceHistogram::count),
                Some(2),
                "{key}: {histograms:?}"
            );
        }
        let tapped = lock(&tapped).clone();
        let probes: Vec<(u32, u32)> = tapped
            .iter()
            .map(|record| (record.step, record.probe))
            .collect();
        assert_eq!(
            probes,
            [
                (7, STEP_BEGIN),
                (7, 2),
                (7, 3),
                (7, STEP_END),
                (9, STEP_BEGIN),
                (9, 2),
                (9, 3),
                (9, STEP_END),
            ],
            "{tapped:?}"
        );
        assert!(tapped.windows(2).all(|pair| pair[0].time <= pair[1].time));

        // Switched off, a probe returns at once; a marker too.
        fences.switch(false)?;
        stream.synchronize()?;
        fences.mark(&stream, 8, "decode", true)?;
        // SAFETY: as above.
        unsafe { fences.probe(stream.raw(), device, 0)? };
        stream.synchronize()?;
        fences.poll(true);
        assert!(fences.take().0.is_empty());

        let [alignment] = fences.calibrate()[..] else {
            panic!("one device calibrates");
        };
        assert!(
            alignment.gpu > 0 && alignment.uncertainty < 1_000_000,
            "{alignment:?}"
        );
        Ok(())
    }

    /// The graph of what `issue` launches on `stream` while the stream
    /// captures in `CU_STREAM_CAPTURE_MODE_GLOBAL`, the mode of
    /// `torch.cuda.graph`; fails when the capture was invalidated.
    fn capture_globally(
        stream: &Stream,
        issue: impl FnOnce(&Stream) -> Result<()>,
    ) -> Result<cuda::GraphExec> {
        type Capture = unsafe extern "C" fn(*mut c_void, i32) -> i32;
        type Captured = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32;
        type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;
        // SAFETY: loading the driver runs only its initializers; each type is
        // its symbol's C prototype.
        let (begin, end, destroy) = unsafe {
            let driver = Box::leak(Box::new(libloading::Library::new("libcuda.so.1").unwrap()));
            (
                *driver.get::<Capture>(b"cuStreamBeginCapture_v2").unwrap(),
                *driver.get::<Captured>(b"cuStreamEndCapture").unwrap(),
                *driver.get::<Destroy>(b"cuGraphDestroy").unwrap(),
            )
        };
        let context = cuda::PrimaryContext::retain(stream.device())?;
        let _current = context.enter()?;
        let handle = stream.raw() as *mut c_void;
        let mut graph = std::ptr::null_mut();
        // SAFETY: the stream is live; the driver writes the captured graph,
        // which is destroyed once instantiated.
        unsafe {
            assert_eq!(begin(handle, 0), 0);
            issue(stream)?;
            assert_eq!(end(handle, &mut graph), 0, "the capture was invalidated");
            let graph_exec = cuda::GraphExec::instantiate(stream.device(), graph as usize);
            assert_eq!(destroy(graph), 0);
            Ok(graph_exec?)
        }
    }
}
