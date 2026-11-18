//! Range markers: device timestamps at the entry and exit of the model's modules, always on,
//! exported as per-step module times (`forward_module_us`).
//!
//! **Where.** A model states each module's forward as a profiler range (`nn::profiler::range`):
//! its class, which names the kind of module, and its path, which names the instance. The context
//! a runner captures a graph under carries the process's [`Markers`] as its profiler
//! ([`profiler`], `ops::Context::set_profiler`), so a module whose class matches the
//! deployment's class patterns ([`classes`]) gets one marker as it is entered and one as it
//! returns. Each is a launch of the stamp kernel on the forward's stream, so the graph captures
//! it and every replay runs it. The worker brackets each batch's execution with a step marker of
//! the same kernel on the runner's stream at either end ([`Markers::bracket`]), which stamps the
//! batch and stores its id in the stream's step cell; the markers the replay runs read the cell,
//! so every stamp names its step.
//!
//! **The kernels.** The device code is the package of the `range_marker` operator
//! (`crates/ops/kernels/range_marker`): a stamp kernel, and a clock kernel for calibrations. The
//! Build Server builds it for each device's target, and the start installs it through the
//! engine's module load like every operator's package ([`keys`]).
//!
//! **What a marker does.** It reads the device switch: a word per class and a master word, in
//! device storage. A marker whose words are off returns at once, so switching markers needs no
//! capture again. A marker that is on reads `%globaltimer`, takes the next sequence number from a
//! device counter, and writes a 16-byte stamp (time, marker, step and the epoch tag below) into
//! its slot of a double buffer in pinned host storage the device writes through unified
//! addressing. Sequence number `s` writes slot `s mod N` of half `(s / N) mod 2`, `N` stamps a
//! half; the epoch `s / N` is tagged in the stamp (12 bits). The marker that completes a half
//! publishes the half's epoch plus one in the half's `filled` word. A stamp's tagged word and a
//! half's `filled` word are stored with release semantics at system scope (the package states
//! the ordering), so a reader that acquires a stamp's tag reads its time, and one that acquires
//! a half's `filled` word reads every stamp of the half.
//!
//! **Reading.** Each worker reads its device's stamps after every batch
//! ([`Markers::read_device`]), without any device synchronization: the complete halves, then the
//! stamps of the half being filled up to the first the device has not written yet, which a later
//! read takes. A stamp is taken only when its tag is that of the epoch read; a slot of a complete
//! half without it was overwritten by a later epoch (the host fell two halves behind) and is
//! counted as lost.
//!
//! **Steps.** A step is measured once its begin stamp and its end stamp were read, with the
//! stamps of its modules between. A loss discards every step whose begin was read before it,
//! since the loss may have taken any of its stamps, and a module's stamp of a step whose begin
//! was not read is dropped. Steps on one stream end in their launch order, so the end of a step
//! read releases every earlier step of its stream whose end was lost: what a device keeps of its
//! steps is bounded by the steps in flight.
//!
//! **Clocks.** `%globaltimer` is a nanosecond clock of the GPU. A calibration launches the clock
//! kernel, which stores `%globaltimer` into pinned host storage, and behind it on the same stream
//! a host function that reads the host's clocks once the kernel completed
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

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::error::Error as StdError;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crossbeam_channel::Sender;
use ops::{Arguments, Kernel, KernelEntry, LaunchGrid};
use platform::{DeviceAllocation, DeviceId, DevicePtr, PinnedBuffer, Stream};
use proto::config::ProfilingConfig;
use proto::report::{FenceHistogram, FenceStep};

use super::artifact::{ClockRow, Message};
use crate::worker::{Error, Result};
use service::lock;

/// The descriptor key of the `range_marker` package's stamp kernel.
const STAMP_KERNEL: &str = "function_stamp";

/// The descriptor key of the `range_marker` package's clock kernel.
const CLOCK_KERNEL: &str = "function_clock";

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

/// The name patterns of the module classes whose instances carry markers in the graphs a
/// deployment captures: `profiling_config.fence_modules`, or the family's step stack (the class
/// of the module the family runs once per step) when it names none. Empty for a deployment that
/// places no marker.
pub(crate) fn classes(profiling: &ProfilingConfig, step_stack: &str) -> Vec<String> {
    profiling
        .fence_modules
        .clone()
        .unwrap_or_else(|| vec![step_stack.to_owned()])
}

/// The kernels a process launches for its markers, by operator, as a recorded start states the
/// kernels it launches and the module load selects their packages: the `range_marker` package's
/// stamp and clock kernels.
pub(crate) fn keys() -> BTreeMap<String, BTreeSet<String>> {
    BTreeMap::from([(
        proto::module::operator::RANGE_MARKER.to_owned(),
        BTreeSet::from([STAMP_KERNEL.to_owned(), CLOCK_KERNEL.to_owned()]),
    )])
}

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

/// What a reader took from a device's buffer, in sequence order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Read {
    /// A stamp the device wrote.
    Stamp(Stamp),
    /// Stamps the device overwrote before the reader took them.
    Lost(u64),
}

/// A step whose begin marker was launched and whose end has not been read: the stream it runs
/// on, by the stream's handle, its batch id's low 32 bits and its stage.
struct Launched {
    stream: usize,
    step: u32,
    stage: Arc<str>,
}

/// The `range_marker` package's two kernels, loaded on one device.
struct Kernels {
    stamp: Arc<Kernel>,
    clock: Arc<Kernel>,
}

impl Kernels {
    /// The stamp and clock kernels among `entries`, `device`'s installed modules of the package.
    ///
    /// # Errors
    ///
    /// When the entries lack either kernel.
    fn of(device: DeviceId, entries: &[KernelEntry]) -> Result<Self> {
        let find = |key: &str| {
            entries
                .iter()
                .find(|entry| {
                    entry.operator == proto::module::operator::RANGE_MARKER
                        && entry.descriptor == key
                })
                .map(|entry| Arc::clone(&entry.kernel))
                .ok_or_else(|| {
                    Error::Invalid(format!(
                        "the range markers of {device} have no {key} kernel of the {} package",
                        proto::module::operator::RANGE_MARKER
                    ))
                })
        };
        Ok(Self {
            stamp: find(STAMP_KERNEL)?,
            clock: find(CLOCK_KERNEL)?,
        })
    }
}

/// A device's marker resources: the kernels, the switch, the counter, the step cells and the
/// double buffer, and the reader's position in it.
struct DeviceMarkers {
    device: DeviceId,
    kernels: Kernels,
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
    /// The stream the switch writes and calibrations go on, of the device's greatest priority so
    /// that neither queues behind the device's computation.
    control: Stream,
    /// The step cell of each stream that carried a marker, by the stream's handle.
    stream_cells: Mutex<HashMap<usize, usize>>,
    reader: Mutex<Reader>,
    /// The steps whose begin marker was launched and whose end was not read, in launch order.
    launched: Mutex<VecDeque<Launched>>,
    /// Histograms of completed steps since the last take, by `{stage}/{class}`.
    histograms: Mutex<BTreeMap<FenceStep, FenceHistogram>>,
    /// Where the stamps read go while a session keeps them: the class patterns it keeps (empty
    /// for every class) and its file's writer.
    tap: Mutex<Option<(Vec<String>, Sender<Message>)>>,
}

/// Where a device's reader stands.
#[derive(Default)]
struct Reader {
    /// The next sequence number to take.
    next: u64,
    /// Stamps counted as lost since the last take.
    lost: u64,
    /// The stamps of each step whose begin stamp was read and whose end stamp was not: its begin
    /// marker's and its modules'.
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
    /// Installs the markers of the classes `patterns` name (`*` matches any run) on every device
    /// of `kernels`, each with its installed modules of the `range_marker` package ([`keys`]),
    /// every switch word on as `enabled` says but those of the classes `off` names, as the
    /// process's markers. A graph captured from here on carries them ([`profiler`]).
    ///
    /// # Errors
    ///
    /// Fails when a device's modules lack a kernel of the package or the device cannot allocate
    /// the buffers, or the process installed markers already.
    pub(crate) fn install(
        kernels: Vec<(DeviceId, Vec<KernelEntry>)>,
        patterns: Vec<String>,
        enabled: bool,
        off: &[String],
    ) -> Result<Arc<Self>> {
        let devices = kernels
            .into_iter()
            .map(|(device, entries)| DeviceMarkers::new(device, Kernels::of(device, &entries)?))
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

    /// Runs `run`, the launches of step `step` (a batch id) of stage `stage` (its call kind) on
    /// `stream`, the runner's computation stream, between the step's begin marker and its end
    /// marker. The end is launched right behind the step's last launch, whatever `run` returns,
    /// so every step whose begin marker was launched has its end launched; launched after the
    /// worker waited for the step, its span would hold the host's trip back as well as the
    /// device's work. With the markers switched off, `run` runs alone.
    ///
    /// # Errors
    ///
    /// When the stream's device has no markers or a marker's launch fails; `run`'s own outcome
    /// is the value.
    pub(crate) fn bracket<T>(
        &self,
        stream: &Stream,
        step: u64,
        stage: &str,
        run: impl FnOnce() -> T,
    ) -> Result<T> {
        if !self.recording.load(Ordering::Acquire) {
            return Ok(run());
        }
        let markers = self.device(stream.device())?;
        let cell = Self::cell(markers, stream.raw());
        // A stamp carries the low 32 bits of the batch id.
        let step = step as u32;

        markers.launch(stream, 0, STEP_BEGIN, cell, Some(step))?;
        lock(&markers.launched).push_back(Launched {
            stream: stream.raw(),
            step,
            stage: Arc::from(stage),
        });
        let ran = run();
        markers.launch(stream, 0, STEP_END, cell, Some(step))?;
        Ok(ran)
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

    /// Reads `device`'s stamps, its complete halves and the written stamps of the half being
    /// filled, and aggregates the steps that ended. Never waits for the device.
    pub(crate) fn read_device(&self, device: DeviceId) {
        if let Ok(markers) = self.device(device) {
            self.read_of(markers);
        }
    }

    fn read_of(&self, device: &DeviceMarkers) {
        let read = device.read();
        if read.is_empty() {
            return;
        }
        self.aggregate(device, &read);

        if let Some((classes, rows)) = lock(&device.tap).as_ref() {
            let kept: Vec<Stamp> = read
                .iter()
                .filter_map(|read| match read {
                    Read::Stamp(stamp) => Some(*stamp),
                    Read::Lost(_) => None,
                })
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

    /// Adds the steps whose end stamp `read` holds to the device's histograms, by the rules of the
    /// module documentation's **Steps**.
    fn aggregate(&self, device: &DeviceMarkers, read: &[Read]) {
        let ended = end_steps(&mut lock(&device.reader).steps, read);
        if ended.is_empty() {
            return;
        }

        let mut lost_ends = Vec::new();
        let mut measured = Vec::new();
        {
            let mut launched = lock(&device.launched);
            for (step, end, open) in ended {
                // A step no bracket of this process launched names no stage, and is not measured.
                let Some((stage, lost)) = release(&mut launched, step) else {
                    continue;
                };
                lost_ends.extend(lost);
                if let Some(open) = open {
                    measured.push((stage, end, open));
                }
            }
        }
        // The earlier steps of a stream whose ends were lost are not measured: what was read of
        // them goes.
        if !lost_ends.is_empty() {
            let mut reader = lock(&device.reader);
            for step in lost_ends {
                reader.steps.remove(&step);
            }
        }

        let mut histograms = lock(&device.histograms);
        let instances = lock(&self.instances);
        for (stage, end, mut open) in measured {
            let begin = open.iter().rev().find(|stamp| stamp.marker == STEP_BEGIN);
            let step_of = |module: &str| FenceStep {
                stage: stage.to_string(),
                module: module.to_owned(),
            };
            if let Some(begin) = begin {
                histograms
                    .entry(step_of("step"))
                    .or_default()
                    .observe(end.saturating_sub(begin.time) / 1000);
            }
            for (class, ns) in class_time(&instances.instance_classes, &mut open) {
                histograms
                    .entry(step_of(&instances.classes[class - 1]))
                    .or_default()
                    .observe(ns / 1000);
            }
        }
    }

    /// The histograms of `device`'s steps completed since the last take, in microseconds by stage
    /// and class, and its stamps lost.
    pub(crate) fn take(&self, device: DeviceId) -> (BTreeMap<FenceStep, FenceHistogram>, u64) {
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

/// Takes `read`, a device's stamps in sequence order, into `steps`, the stamps of each step whose
/// begin stamp was read and whose end was not, and returns each step whose end stamp it holds:
/// its batch id's low 32 bits, the end's time and the step's stamps, `None` when its begin was not
/// read. A begin stamp opens its step; a module's stamp joins its step when that is open and is
/// dropped otherwise; a loss discards every open step, since it may have taken any of their
/// stamps.
fn end_steps(
    steps: &mut BTreeMap<u32, Vec<Stamp>>,
    read: &[Read],
) -> Vec<(u32, u64, Option<Vec<Stamp>>)> {
    let mut ended = Vec::new();
    for read in read {
        match *read {
            Read::Lost(_) => steps.clear(),
            Read::Stamp(stamp) => match stamp.marker {
                STEP_BEGIN => steps.entry(stamp.step).or_default().push(stamp),
                STEP_END => ended.push((stamp.step, stamp.time, steps.remove(&stamp.step))),
                _ => {
                    if let Some(open) = steps.get_mut(&stamp.step) {
                        open.push(stamp);
                    }
                }
            },
        }
    }
    ended
}

/// Releases step `step`, whose end stamp was read, from `launched`: its entry, and every earlier
/// entry of its stream, which ended before it in stream order and whose end stamp was lost.
/// Returns the step's stage and the steps released whose ends were lost; `None` for a step no
/// entry names.
fn release(launched: &mut VecDeque<Launched>, step: u32) -> Option<(Arc<str>, Vec<u32>)> {
    let position = launched.iter().position(|entry| entry.step == step)?;
    let stream = launched[position].stream;

    let mut lost = Vec::new();
    let mut stage = None;
    let mut index = 0;
    launched.retain(|entry| {
        let at = index;
        index += 1;
        if at == position {
            stage = Some(Arc::clone(&entry.stage));
            return false;
        }
        if at < position && entry.stream == stream {
            lost.push(entry.step);
            return false;
        }
        true
    });
    stage.map(|stage| (stage, lost))
}

/// Notes `stamps` lost at the current position of a read: the reader counts them, and `taken`
/// states them in sequence order, one entry for a run of them.
fn lose(reader: &mut Reader, taken: &mut Vec<Read>, stamps: u64) {
    reader.lost += stamps;
    match taken.last_mut() {
        Some(Read::Lost(lost)) => *lost += stamps,
        _ => taken.push(Read::Lost(stamps)),
    }
}

impl DeviceMarkers {
    fn new(device: DeviceId, kernels: Kernels) -> Result<Self> {
        let control = Stream::with_greatest_priority(device)?;
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
            kernels,
            switches,
            counters,
            cells,
            stamps,
            filled,
            clock_out,
            control,
            stream_cells: Mutex::new(HashMap::new()),
            reader: Mutex::new(Reader::default()),
            launched: Mutex::new(VecDeque::new()),
            histograms: Mutex::new(BTreeMap::new()),
            tap: Mutex::new(None),
        })
    }

    /// Launches the stamp kernel on `stream`: a module's marker of switch word `class`, or with
    /// `step` a step marker.
    fn launch(
        &self,
        stream: &Stream,
        class: u32,
        marker: u32,
        cell: usize,
        step: Option<u32>,
    ) -> Result<()> {
        let address = DevicePtr::from_address;
        // Each word names the stamp kernel's parameter it fills; the launch holds them to the
        // module's signature.
        let mut arguments = Arguments::new(proto::module::operator::RANGE_MARKER);
        arguments
            .pointer("switches", address(self.switches.address()))
            .u32("klass", class)
            .u32("marker", marker)
            .pointer("cell", address(self.cells.address() + 4 * cell as u64))
            .u32("mark", u32::from(step.is_some()))
            .u32("step", step.unwrap_or(0))
            .pointer("counter", address(self.counters.address()))
            .pointer("written", address(self.counters.address() + 8))
            .pointer("stamps", address(self.stamps.address()))
            .pointer("filled", address(self.filled.address()))
            .u32("shift", SHIFT);

        // Every buffer the arguments address lives with the process's markers, past every replay
        // of a graph that captured the launch.
        self.kernels
            .stamp
            .launch(LaunchGrid::linear(1)?, &mut arguments, stream)?;
        Ok(())
    }

    /// The word at `index` of the pinned host storage `buffer`, as an atomic the device writes.
    fn word(buffer: &PinnedBuffer, index: usize) -> &AtomicU64 {
        // SAFETY: `buffer` is pinned host storage of whole, aligned `u64` words that lives as long
        // as the markers, `index` lies in it (every caller's index is within its buffer's
        // layout), and the device writes its words only with atomic stores.
        unsafe { AtomicU64::from_ptr(buffer.as_ptr().cast::<u64>().add(index)) }
    }

    /// The `filled` word of `half`, acquired: a half it shows complete has every stamp visible.
    fn filled(&self, half: u64) -> u64 {
        Self::word(&self.filled, half as usize).load(Ordering::Acquire)
    }

    /// The two words of sequence number `sequence`'s slot, `(time, packed)`: the tagged word is
    /// acquired first, so a tag of the epoch read comes with that stamp's time.
    fn slot(&self, sequence: u64) -> (u64, u64) {
        let index = (((sequence >> SHIFT) & 1) * HALF + (sequence & (HALF - 1))) as usize;
        let packed = Self::word(&self.stamps, 2 * index + 1).load(Ordering::Acquire);
        let time = Self::word(&self.stamps, 2 * index).load(Ordering::Relaxed);
        (time, packed)
    }

    /// Takes what the device wrote since the last read, in sequence order: every stamp of the
    /// complete halves, a [`Read::Lost`] where the device overwrote stamps before they were read,
    /// then the stamps of the half being filled up to the first the device has not written yet,
    /// which a later read takes.
    fn read(&self) -> Vec<Read> {
        let mut reader = lock(&self.reader);
        let mut taken = Vec::new();
        let completed = self.filled(0).max(self.filled(1));

        // The device is filling epoch `completed` (or a later one, before its predecessor shows
        // complete); a half two epochs behind it was overwritten.
        let oldest = completed.saturating_sub(1);
        if reader.next >> SHIFT < oldest {
            let resumed = oldest << SHIFT;
            let skipped = resumed - reader.next;
            lose(&mut reader, &mut taken, skipped);
            reader.next = resumed;
        }

        // A complete half: its acquired `filled` word makes every stamp of it visible, so a slot
        // without the epoch's tag was overwritten by a later epoch.
        loop {
            let epoch = reader.next >> SHIFT;
            if self.filled(epoch & 1) != epoch + 1 {
                break;
            }
            while reader.next >> SHIFT == epoch {
                match stamp(self.slot(reader.next), epoch) {
                    Some(stamp) => taken.push(Read::Stamp(stamp)),
                    None => lose(&mut reader, &mut taken, 1),
                }
                reader.next += 1;
            }
        }

        // The half being filled: its stamps up to the first the device has not written yet.
        let epoch = reader.next >> SHIFT;
        while reader.next >> SHIFT == epoch {
            match stamp(self.slot(reader.next), epoch) {
                Some(stamp) => taken.push(Read::Stamp(stamp)),
                None => break,
            }
            reader.next += 1;
        }
        taken
    }

    /// One calibration sample: the clock kernel's `%globaltimer` against the host's clocks before
    /// its launch and in a host function the stream runs once the kernel completed. Keeps and
    /// returns the best sample.
    ///
    /// The sample's uncertainty is half the host's window from before the launch to the host
    /// function: a kernel that queued behind the device's work gives a wide sample, which a
    /// narrower one replaces.
    fn calibrate(&self) -> Result<ClockRow> {
        let mut arguments = Arguments::new(proto::module::operator::RANGE_MARKER);
        arguments.pointer("out", DevicePtr::from_address(self.clock_out.address()));
        let before = (realtime(), platform::cupti::timestamp().unwrap_or(0));
        self.kernels
            .clock
            .launch(LaunchGrid::linear(1)?, &mut arguments, &self.control)?;

        let (sender, completed) = crossbeam_channel::bounded(1);
        let word = self.clock_out.as_ptr().cast::<u64>().expose_provenance();
        self.control.launch_host_fn(move || {
            let after = (realtime(), platform::cupti::timestamp().unwrap_or(0));
            // SAFETY: the word is aligned pinned host storage of the markers, which outlive the
            // calibration, and the calibration waits for this function; the kernel before it
            // stored the word.
            let gpu = unsafe { AtomicU64::from_ptr(std::ptr::with_exposed_provenance_mut(word)) }
                .load(Ordering::Acquire);
            let _ = sender.send((gpu, after));
        })?;
        // The stream completes with the host function, which holds the sample.
        self.control.synchronize()?;
        let (gpu, after) = completed
            .try_recv()
            .map_err(|_| Error::State("the calibration's host function did not run"))?;

        let sample = ClockRow {
            source: "globaltimer",
            device: Some(self.device.0),
            gpu: Some(gpu),
            realtime: before.0 / 2 + after.0 / 2,
            cupti: Some(before.1 / 2 + after.1 / 2),
            monotonic: None,
            uncertainty: after.0.saturating_sub(before.0) / 2,
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

    /// A begin stamp opens its step and its end closes it with the module stamps between; a
    /// module's stamp of a step whose begin was not read is dropped, and a loss discards the
    /// steps open before it, whose ends then close nothing.
    #[test]
    fn a_step_is_measured_only_from_its_begin_and_never_across_a_loss() {
        let at = |time, marker, step| Read::Stamp(Stamp { time, marker, step });
        let mut steps = BTreeMap::new();

        let ended = end_steps(
            &mut steps,
            &[
                at(1, STEP_BEGIN, 7),
                at(2, 0, 7),
                at(3, 1, 7),
                // Step 8's begin was not read: its module's stamp is dropped.
                at(4, 0, 8),
                at(5, STEP_END, 7),
                at(6, STEP_BEGIN, 9),
                at(7, 0, 9),
            ],
        );
        assert_eq!(ended.len(), 1);
        let (step, end, open) = &ended[0];
        assert_eq!((*step, *end), (7, 5));
        assert_eq!(open.as_ref().map(Vec::len), Some(3));
        assert!(!steps.contains_key(&8));

        // Step 9 was open when stamps were lost: its end closes nothing measured.
        let ended = end_steps(&mut steps, &[Read::Lost(3), at(9, STEP_END, 9)]);
        assert_eq!(ended, vec![(9, 9, None)]);
        assert!(steps.is_empty());
    }

    /// The end of a step releases the earlier steps of its stream, whose ends were lost, and
    /// leaves another stream's steps for their own ends.
    #[test]
    fn a_step_s_end_releases_the_earlier_steps_of_its_stream() {
        let entry = |stream, step, stage: &str| Launched {
            stream,
            step,
            stage: Arc::from(stage),
        };
        let mut launched = VecDeque::from([
            entry(1, 10, "decode"),
            entry(2, 11, "prefill"),
            entry(1, 12, "decode"),
            entry(1, 13, "verify"),
        ]);

        let (stage, lost) = release(&mut launched, 12).unwrap();
        assert_eq!((&*stage, lost), ("decode", vec![10]));
        let left: Vec<(usize, u32)> = launched
            .iter()
            .map(|entry| (entry.stream, entry.step))
            .collect();
        assert_eq!(left, [(2, 11), (1, 13)]);

        // A step no bracket launched names no stage.
        assert!(release(&mut launched, 99).is_none());
    }

    /// The `range_marker` package's modules on `device`, as the start installs them: selected and
    /// built for the device's target at the Build Server of the deployment `device-cases` on this
    /// host (`build_server::socket`), which holds the package.
    fn kernels(device: DeviceId) -> Vec<KernelEntry> {
        let socket = build_server::socket(
            &service::RuntimeDirectory::of_user().unwrap(),
            "device-cases",
        )
        .unwrap();
        let mut client = build_server::Client::connect(&socket).unwrap();
        let targets = [build_server::Target::arch(device.compute_capability().unwrap()).unwrap()];
        let (packages, routes) =
            crate::modules::load::select(&mut client, &targets, &Default::default(), &keys())
                .unwrap();
        crate::modules::load::install(&mut client, device, &packages, &routes).unwrap()
    }

    /// The markers a capture places around a module run at every replay: each step between its
    /// step markers counts once in the histogram of its stage and the module's class, and once in
    /// the stage's own; a calibration samples the device's clock.
    #[test]
    #[ignore = "requires a GPU and the host's Build Server holding the range_marker package"]
    fn a_captured_range_times_its_module_at_every_replay() {
        let device = DeviceId(0);
        let markers = Markers::install(
            vec![(device, kernels(device))],
            vec!["Mlp".to_owned()],
            true,
            &[],
        )
        .unwrap();
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
            markers
                .bracket(&stream, step, "decode", || executable.launch(&stream))
                .unwrap()
                .unwrap();
        }
        stream.synchronize().unwrap();
        markers.read_device(device);

        let (histograms, lost) = markers.take(device);
        let steps = |module: &str| {
            let step = FenceStep {
                stage: "decode".to_owned(),
                module: module.to_owned(),
            };
            histograms[&step].buckets.iter().sum::<u64>()
        };
        assert_eq!(lost, 0);
        assert_eq!(steps("step"), 2);
        assert_eq!(steps("Mlp"), 2);
        assert_eq!(
            markers.instances(),
            [("Mlp".to_owned(), "layers.0.mlp".to_owned())]
        );

        let sample = markers.calibrate(device).unwrap();
        assert!(sample.gpu.is_some_and(|gpu| gpu > 0));
    }
}
