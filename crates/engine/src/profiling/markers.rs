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
//! **Kernel probes.** A capture of a forward that enters the family's step stack also places a
//! probe before and after kernels ([`Plan`]): the stack's child ranges are its layers, two of
//! which are of one kind when their classes and the operators and descriptor keys of their
//! launch tables are the same, and each kind's instance `profiling_config.kernel_instance` names
//! carries a probe pair around each of its launches, as does every launch outside the layers.
//! Each probed launch is a site, named by its operator, its descriptor key, its module path and
//! the layers of its kind; its probes are stamps of the reserved class [`KERNEL_CLASS`], whose
//! switch word follows `profiling_config.kernel_probes`. Where the probes lie is the
//! generation's ([`Markers::place`]), so moving them captures every bucket again.
//!
//! **Spans.** At a step's end marker the reader also writes the step's span and the span of
//! every kernel probe pair of the step into the device's ring (`service::span`), each named by
//! the batch, its times converted to the host's `CLOCK_REALTIME` through the device's latest
//! calibration, which the reader takes again every [`CALIBRATION`].
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
use proto::histogram::Histogram;
use proto::report::FenceStep;

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

/// The class of the kernel probes' sites: no module's, so that no pattern of `fence_modules` or
/// `fence_probes_off` is needed to name it, and its switch word follows the kernel probes' own
/// switch.
pub(crate) const KERNEL_CLASS: &str = "(kernel)";

/// The records a device's span ring keeps: 32 MiB, about a minute of a decode's kernel spans.
const DEVICE_RECORDS: usize = 1 << 21;

/// How often a device's reader calibrates its clock against the host's for the spans it writes.
const CALIBRATION: std::time::Duration = std::time::Duration::from_secs(10);

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

/// The kernels a process launches for its markers, by operator, as the module load selects their
/// packages: the `range_marker` package's stamp and clock kernels.
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
    /// The batch's full id, which its spans name.
    batch: u64,
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
    histograms: Mutex<BTreeMap<FenceStep, Histogram>>,
    /// Where the stamps read go while a session keeps them: the class patterns it keeps (empty
    /// for every class) and its file's writer.
    tap: Mutex<Option<(Vec<String>, Sender<Message>)>>,
    /// The ring the device's step and kernel spans are written into.
    ring: &'static service::span::Ring,
    /// The calibration the spans' times are converted through, and when it was taken.
    clock: Mutex<Option<(ClockRow, std::time::Instant)>>,
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
    /// Each kernel probe's site, by its instance number.
    sites: HashMap<u32, service::span::Kernel>,
    /// Each site's instance number, by its operator, descriptor key and module path.
    site_numbers: HashMap<(&'static str, String, String), u32>,
}

/// The range markers of this process's devices and what they measure.
pub(crate) struct Markers {
    devices: Vec<DeviceMarkers>,
    /// The class patterns of the generations [`Markers::place_classes`] names none for
    /// (`profiling_config.fence_modules` as the process started).
    patterns: Vec<String>,
    /// The class patterns of each generation that names its own, which every capture of it
    /// follows.
    generation_patterns: Mutex<BTreeMap<u64, Vec<String>>>,
    instances: Mutex<Instances>,
    /// The switch as last stated: the master word, the patterns switched off and the kernel
    /// probes' word.
    switched: Mutex<Switch>,
    /// The family's step stack, whose child ranges are the layers kernel probes are placed in.
    stack: String,
    /// Which instance of each kind of layer carries kernel probes, by generation; a generation
    /// not named takes `placement`.
    placements: Mutex<BTreeMap<u64, Option<u32>>>,
    placement: Option<u32>,
    /// The master word, read at every step: a step marker with the switch off stamps nothing, so
    /// it is not launched.
    recording: AtomicBool,
}

/// The switch as last stated.
#[derive(Clone, Debug, Default)]
struct Switch {
    /// The master word.
    enabled: bool,
    /// The patterns of the probed classes switched off.
    off: Vec<String>,
    /// The kernel probes' word, on while the master word is too.
    kernels: bool,
}

static MARKERS: OnceLock<Arc<Markers>> = OnceLock::new();

/// The profiler of a context on `stream` that captures a forward of generation `generation`
/// replayed on `replay`,
/// whose meta run entered `ranges` and launched `table`: the process's markers while the stream
/// captures, because a graph is what runs them at every step, with the kernel probes the
/// generation places in the forward ([`Plan`]); none for a forward run once or in a process
/// without markers.
///
/// A runner gives the result to `ops::Context::set_profiler` before it runs the forward it
/// captures.
pub(crate) fn profiler(
    stream: &Stream,
    replay: &Stream,
    generation: u64,
    ranges: &[ops::context::Range],
    table: &[(&'static str, String, String)],
) -> Option<Arc<dyn ops::Profiler>> {
    let markers = MARKERS.get()?;
    let capturing = stream
        .capture_status()
        .is_ok_and(|status| status != platform::CaptureStatus::None);
    if !capturing {
        return None;
    }

    let plan = markers
        .placement(generation)
        .map(|instance| Plan::of(ranges, table, &markers.stack, instance));
    Some(Arc::new(Capture {
        markers: Arc::clone(markers),
        replay: replay.raw(),
        plan,
        patterns: markers.classes(generation),
    }))
}

/// Where one capture's kernel probes lie: the layers of a forward's step stack, grouped by kind,
/// and the instance of each kind that carries probes.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Plan {
    /// Each layer instance's path, with the layers of its kind when it is its kind's probed
    /// instance and `None` otherwise.
    layers: HashMap<String, Option<u32>>,
    /// Whether the forward entered the step stack: a forward that does not places no probe.
    stacked: bool,
}

impl Plan {
    /// The plan of a forward that entered `ranges` and launched `table`: the child ranges of each
    /// instance of the class `stack` names (a pattern, `*` matching any run) are its layers, each
    /// of the kind of its class and the operators and keys it launched, and of each kind the
    /// instance at position `instance` in launch order (the last of a kind with fewer) carries
    /// the probes.
    pub(crate) fn of(
        ranges: &[ops::context::Range],
        table: &[(&'static str, String, String)],
        stack: &str,
        instance: u32,
    ) -> Self {
        let mut stacked = false;
        let mut within: Option<usize> = None;
        let mut layers: Vec<(&str, &str)> = Vec::new();
        for range in ranges {
            if matches(stack, &range.class) {
                stacked = true;
                within = Some(range.depth);
                continue;
            }
            match within {
                Some(depth) if range.depth <= depth => within = None,
                Some(depth) if range.depth == depth + 1 => {
                    layers.push((&range.class, &range.path));
                }
                _ => {}
            }
        }

        // The kinds in the order their first layers were entered, each with its layers.
        let mut kinds: Vec<(Kind<'_>, Vec<&str>)> = Vec::new();
        for (class, path) in layers {
            let launches: Vec<(&str, &str)> = table
                .iter()
                .filter(|(operator, _, at)| {
                    *operator != ops::breaks::BREAK && within_path(at, path)
                })
                .map(|(operator, key, _)| (*operator, key.as_str()))
                .collect();
            let kind = (class, launches);
            match kinds.iter_mut().find(|(known, _)| *known == kind) {
                Some((_, paths)) => paths.push(path),
                None => kinds.push((kind, vec![path])),
            }
        }

        let mut plan = Self {
            layers: HashMap::new(),
            stacked,
        };
        for (_, paths) in kinds {
            let count = u32::try_from(paths.len()).unwrap_or(u32::MAX);
            let chosen = (instance as usize).min(paths.len() - 1);
            for (position, path) in paths.into_iter().enumerate() {
                plan.layers
                    .insert(path.to_owned(), (position == chosen).then_some(count));
            }
        }
        plan
    }

    /// The layers a launch inside the range at `path` stands for when it carries probes: those of
    /// its layer's kind in its kind's probed instance, one outside every layer; `None` in another
    /// instance, and in a forward that never entered the step stack.
    pub(crate) fn layers(&self, path: &str) -> Option<u32> {
        if !self.stacked {
            return None;
        }
        let mut at = path;
        loop {
            if let Some(&layers) = self.layers.get(at) {
                return layers;
            }
            match at.rfind('.') {
                Some(dot) => at = &at[..dot],
                None => return Some(1),
            }
        }
    }
}

/// A kind of layer: its class, and the operators and descriptor keys its launch table states in
/// launch order.
type Kind<'a> = (&'a str, Vec<(&'a str, &'a str)>);

/// Whether the range at `path` lies within the range at `outer`: it is that range, or one whose
/// path continues it after a dot.
fn within_path(path: &str, outer: &str) -> bool {
    path.strip_prefix(outer)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('.'))
}

/// The profiler of one capture: the process's markers, with the capture's kernel probes.
struct Capture {
    markers: Arc<Markers>,
    /// The handle of the stream the captured graph is replayed on, whose step cell its markers
    /// read: a graph captured beside the runner's, on a stream of its own, replays on the
    /// runner's.
    replay: usize,
    /// `None` when the capture's generation places no kernel probe.
    plan: Option<Plan>,
    /// The class patterns whose instances carry the capture's generation's fence probes.
    patterns: Vec<String>,
}

impl ops::Profiler for Capture {
    fn enter(
        &self,
        class: &str,
        path: &str,
        stream: &Stream,
    ) -> std::result::Result<Option<u32>, Box<dyn StdError + Send + Sync>> {
        let Some(instance) = self.markers.instance(&self.patterns, class, path)? else {
            return Ok(None);
        };
        self.markers.launch(stream, self.replay, 2 * instance)?;
        Ok(Some(instance))
    }

    fn exit(
        &self,
        instance: u32,
        stream: &Stream,
    ) -> std::result::Result<(), Box<dyn StdError + Send + Sync>> {
        self.markers.launch(stream, self.replay, 2 * instance + 1)?;
        Ok(())
    }

    fn launch(
        &self,
        operator: &'static str,
        key: &str,
        path: &str,
        stream: &Stream,
    ) -> std::result::Result<Option<u32>, Box<dyn StdError + Send + Sync>> {
        let Some(layers) = self.plan.as_ref().and_then(|plan| plan.layers(path)) else {
            return Ok(None);
        };
        let Some(site) = self.markers.site(operator, key, path, layers)? else {
            return Ok(None);
        };
        self.markers.launch(stream, self.replay, 2 * site)?;
        Ok(Some(site))
    }

    fn launched(
        &self,
        site: u32,
        stream: &Stream,
    ) -> std::result::Result<(), Box<dyn StdError + Send + Sync>> {
        self.markers.launch(stream, self.replay, 2 * site + 1)?;
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
    ///
    /// The kernel probes record while `kernels` and `enabled` both hold; `stack` is the family's
    /// step stack, whose child ranges are the layers they are placed in, and `placement` which
    /// instance of each kind of layer carries them in every generation [`Markers::place`] names
    /// no other placement for.
    pub(crate) fn install(
        kernels: Vec<(DeviceId, Vec<KernelEntry>)>,
        patterns: Vec<String>,
        (enabled, off, kernel_probes): (bool, &[String], bool),
        stack: String,
        placement: Option<u32>,
    ) -> Result<Arc<Self>> {
        let devices = kernels
            .into_iter()
            .map(|(device, entries)| DeviceMarkers::new(device, Kernels::of(device, &entries)?))
            .collect::<Result<Vec<_>>>()?;
        let markers = Arc::new(Self {
            devices,
            patterns,
            generation_patterns: Mutex::new(BTreeMap::new()),
            instances: Mutex::default(),
            switched: Mutex::new(Switch::default()),
            stack,
            placements: Mutex::new(BTreeMap::new()),
            placement,
            recording: AtomicBool::new(enabled),
        });
        markers.switch(enabled, off, kernel_probes)?;

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
    /// no pattern of `patterns`, the capture's generation's, names the class. An instance keeps
    /// its number across generations, whichever classes each probes.
    fn instance(&self, patterns: &[String], class: &str, path: &str) -> Result<Option<u32>> {
        let timed = class != KERNEL_CLASS && patterns.iter().any(|pattern| matches(pattern, class));
        if !timed {
            return Ok(None);
        }
        self.register(class, path)
    }

    /// The number of the instance at `path` of `class`, naming it on first sight; `None` once the
    /// instances fill the markers' id space.
    fn register(&self, class: &str, path: &str) -> Result<Option<u32>> {
        let mut instances = lock(&self.instances);
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
                let switch = lock(&self.switched).clone();
                let (words, _) = switch_words(&instances.classes, &switch);
                self.write_switch(&words)?;
                index
            }
        };

        instances.instances.push(key.clone());
        instances.instance_classes.push(index as u32 + 1);
        instances.numbers.insert(key, number);
        Ok(Some(number))
    }

    /// The instance number of the kernel probe site of a launch of `operator` under `key` in the
    /// range at `path`, which stands for `layers` layers, naming it on first sight; `None` once
    /// the sites fill what a span's point can name, a launch that then carries no probe.
    fn site(
        &self,
        operator: &'static str,
        key: &str,
        path: &str,
        layers: u32,
    ) -> Result<Option<u32>> {
        let named = (operator, key.to_owned(), path.to_owned());
        if let Some(&site) = lock(&self.instances).site_numbers.get(&named) {
            return Ok(Some(site));
        }
        // The site's probes are stamps of an instance of the kernel class, numbered with the
        // modules' instances.
        let site_path = format!("{path} {operator} {key}");
        let instances = lock(&self.instances);
        let number = instances.instances.len();
        drop(instances);
        let points = usize::from(service::span::POINTS - service::span::KERNEL);
        if number >= points {
            tracing::warn!(
                sites = number,
                "the kernel probes' sites fill what a span names: further launches carry none"
            );
            return Ok(None);
        }
        let Some(site) = self.register(KERNEL_CLASS, &site_path)? else {
            return Ok(None);
        };

        let mut instances = lock(&self.instances);
        instances.site_numbers.insert(named, site);
        instances.sites.insert(
            site,
            service::span::Kernel {
                op: operator.to_owned(),
                descriptor: key.to_owned(),
                module: path.to_owned(),
                layers,
            },
        );
        Ok(Some(site))
    }

    /// The kernel the probes of site `site` time, as a span names it.
    pub(crate) fn kernel(&self, site: u16) -> Option<service::span::Kernel> {
        lock(&self.instances).sites.get(&u32::from(site)).cloned()
    }

    /// Places the kernel probes of generation `generation` in each kind of layer's instance at
    /// `placement`, none when `None`: what every capture of the generation follows.
    pub(crate) fn place(&self, generation: u64, placement: Option<u32>) {
        lock(&self.placements).insert(generation, placement);
    }

    /// Where generation `generation` places the kernel probes.
    pub(crate) fn placement(&self, generation: u64) -> Option<u32> {
        lock(&self.placements)
            .get(&generation)
            .copied()
            .unwrap_or(self.placement)
    }

    /// Probes in generation `generation` the instances of the module classes `fence_modules`
    /// names, the family's step stack when `None` (`profiling_config.fence_modules`): what every
    /// capture of the generation follows.
    pub(crate) fn place_classes(&self, generation: u64, fence_modules: Option<Vec<String>>) {
        let patterns = fence_modules.unwrap_or_else(|| vec![self.stack.clone()]);
        lock(&self.generation_patterns).insert(generation, patterns);
    }

    /// The class patterns whose instances carry generation `generation`'s fence probes.
    pub(crate) fn classes(&self, generation: u64) -> Vec<String> {
        lock(&self.generation_patterns)
            .get(&generation)
            .cloned()
            .unwrap_or_else(|| self.patterns.clone())
    }

    /// Whether generation `next` places the probes otherwise than generation `serving`: its
    /// kernel probes in other layers, or its fence probes on other classes. Every bucket of such
    /// a generation is captured again.
    pub(crate) fn moved(&self, serving: u64, next: u64) -> bool {
        self.placement(serving) != self.placement(next)
            || self.classes(serving) != self.classes(next)
    }

    /// Names the place of `device`'s spans: the worker that issues the device.
    pub(crate) fn label(&self, device: DeviceId, label: service::span::Label) {
        if let Ok(markers) = self.device(device) {
            markers.ring.label(label);
        }
    }

    /// Turns the markers on or off: the master word as `enabled` says, and every class's word
    /// with it but those of the classes matching a pattern of `off` (`*` matches any run), which
    /// stay off, and the kernel probes' word, on while `enabled` and `kernels` both hold. Written
    /// on each device's control stream; markers already queued read whichever value reaches
    /// them. Returns the classes switched off.
    ///
    /// # Errors
    ///
    /// When a device refuses the write.
    pub(crate) fn switch(
        &self,
        enabled: bool,
        off: &[String],
        kernels: bool,
    ) -> Result<Vec<String>> {
        let instances = lock(&self.instances);
        let switch = Switch {
            enabled,
            off: off.to_vec(),
            kernels,
        };
        self.recording.store(enabled, Ordering::Release);

        let (words, switched_off) = switch_words(&instances.classes, &switch);
        *lock(&self.switched) = switch;
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

    /// Launches marker `marker` (`2 * instance` at the entry, `+ 1` at the exit) on `stream`,
    /// captured when the stream captures, for a graph replayed on the stream of handle `replay`:
    /// the marker reads that stream's step cell, which the replay's step markers write, whichever
    /// stream captured it.
    fn launch(&self, stream: &Stream, replay: usize, marker: u32) -> Result<()> {
        let class = *lock(&self.instances)
            .instance_classes
            .get((marker / 2) as usize)
            .ok_or_else(|| {
                Error::Invalid(format!("range marker {marker} names no module instance"))
            })?;
        let markers = self.device(stream.device())?;
        let cell = Self::cell(markers, replay);
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

        markers.launch(stream, 0, STEP_BEGIN, cell, Some(step as u32))?;
        lock(&markers.launched).push_back(Launched {
            stream: stream.raw(),
            step: step as u32,
            batch: step,
            stage: Arc::from(stage),
        });
        let ran = run();
        markers.launch(stream, 0, STEP_END, cell, Some(step as u32))?;
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
                let Some((stage, batch, lost)) = release(&mut launched, step) else {
                    continue;
                };
                lost_ends.extend(lost);
                if let Some(open) = open {
                    measured.push((stage, batch, end, open));
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

        let clock = self.clock_of(device);
        let mut histograms = lock(&device.histograms);
        let instances = lock(&self.instances);
        // The kernel probes' class, whose stamps are spans and enter no module's histogram.
        let kernel_class = instances
            .classes
            .iter()
            .position(|class| class == KERNEL_CLASS)
            .map(|index| index + 1);
        for (stage, batch, end, mut open) in measured {
            let begin = open
                .iter()
                .rev()
                .find(|stamp| stamp.marker == STEP_BEGIN)
                .map(|stamp| stamp.time);
            let step_of = |module: &str| FenceStep {
                stage: stage.to_string(),
                module: module.to_owned(),
            };
            if let Some(begin) = begin {
                histograms
                    .entry(step_of("step"))
                    .or_default()
                    .observe(seconds(end.saturating_sub(begin)));
            }
            let kernels = kernel_spans(&instances.instance_classes, kernel_class, &mut open);
            if let Some(clock) = &clock {
                write_spans(
                    device.ring,
                    clock,
                    batch,
                    begin.map(|begin| (begin, end)),
                    &kernels,
                );
            }
            for (class, ns) in class_time(&instances.instance_classes, &mut open) {
                if Some(class) == kernel_class {
                    continue;
                }
                histograms
                    .entry(step_of(&instances.classes[class - 1]))
                    .or_default()
                    .observe(seconds(ns));
            }
        }
    }

    /// The calibration `device`'s spans are converted through: its latest, taken again once it
    /// is [`CALIBRATION`] old; `None` while the device has none.
    fn clock_of(&self, device: &DeviceMarkers) -> Option<ClockRow> {
        let held = *lock(&device.clock);
        if let Some((clock, taken)) = held
            && taken.elapsed() < CALIBRATION
        {
            return Some(clock);
        }
        // The least uncertain of a few samples: one queued behind other work is wide.
        let sample = (0..3)
            .filter_map(|_| device.sample().ok())
            .min_by_key(|sample| sample.uncertainty);
        match sample {
            Some(sample) => {
                *lock(&device.clock) = Some((sample, std::time::Instant::now()));
                Some(sample)
            }
            None => held.map(|(clock, _)| clock),
        }
    }

    /// The histograms of `device`'s steps completed since the last take, in seconds by stage and
    /// class, and its stamps lost.
    pub(crate) fn take(&self, device: DeviceId) -> (BTreeMap<FenceStep, Histogram>, u64) {
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
fn release(launched: &mut VecDeque<Launched>, step: u32) -> Option<(Arc<str>, u64, Vec<u32>)> {
    let position = launched.iter().position(|entry| entry.step == step)?;
    let stream = launched[position].stream;
    let batch = launched[position].batch;

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
    stage.map(|stage| (stage, batch, lost))
}

/// The kernel probe pairs among one step's stamps, sorted by time: each site's entry paired with
/// its next exit, as `(site, start, end)` in `%globaltimer` nanoseconds, in the order they began.
/// A stamp of another class is passed over.
fn kernel_spans(
    instance_classes: &[u32],
    kernel_class: Option<usize>,
    stamps: &mut [Stamp],
) -> Vec<(u32, u64, u64)> {
    let Some(kernel_class) = kernel_class else {
        return Vec::new();
    };
    stamps.sort_unstable_by_key(|stamp| (stamp.time, stamp.marker));

    let mut entered: HashMap<u32, VecDeque<u64>> = HashMap::new();
    let mut spans = Vec::new();
    for stamp in stamps.iter().filter(|stamp| stamp.marker < STEP_BEGIN) {
        let site = stamp.marker / 2;
        let class = instance_classes.get(site as usize).copied();
        if class != Some(kernel_class as u32) {
            continue;
        }
        if stamp.marker % 2 == 0 {
            entered.entry(site).or_default().push_back(stamp.time);
        } else if let Some(start) = entered.get_mut(&site).and_then(VecDeque::pop_front) {
            spans.push((site, start, stamp.time));
        }
    }
    spans.sort_unstable_by_key(|&(_, start, _)| start);
    spans
}

/// `%globaltimer` nanoseconds `gpu` on the host's `CLOCK_REALTIME`, through `clock`, a
/// calibration sample of the device's clock against the host's.
pub(crate) fn host_time(clock: &ClockRow, gpu: u64) -> u64 {
    let Some(sampled) = clock.gpu else {
        return gpu;
    };
    let offset = i128::from(clock.realtime) - i128::from(sampled);
    u64::try_from(i128::from(gpu) + offset).unwrap_or(0)
}

/// Writes a step's spans into `ring`, named by its batch `batch`: the step's span between its
/// markers, `step` as `(begin, end)`, and each kernel probe pair `(site, start, end)`, every time
/// converted through `clock`.
fn write_spans(
    ring: &service::span::Ring,
    clock: &ClockRow,
    batch: u64,
    step: Option<(u64, u64)>,
    kernels: &[(u32, u64, u64)],
) {
    use service::span::{KERNEL, Mark, Record, Stage};

    let pair = |point: u16, start: u64, end: u64| {
        for (time, mark) in [(start, Mark::Begin), (end, Mark::End)] {
            ring.write(&Record {
                time: host_time(clock, time),
                id: batch,
                point,
                mark,
            });
        }
    };
    if let Some((begin, end)) = step {
        pair(Stage::Step.point(), begin, end);
    }
    for &(site, start, end) in kernels {
        // A site is numbered below what a point names ([`Markers::site`]).
        if let Ok(site) = u16::try_from(site) {
            pair(KERNEL + site, start, end);
        }
    }
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
            ring: service::span::Ring::new(DEVICE_RECORDS, service::span::Label::None),
            clock: Mutex::new(None),
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
        let sample = self.sample()?;
        let mut reader = lock(&self.reader);
        let best = match reader.best {
            Some(best) if best.uncertainty <= sample.uncertainty => best,
            _ => sample,
        };
        reader.best = Some(best);
        Ok(best)
    }

    /// One calibration sample, as [`DeviceMarkers::calibrate`] takes it.
    fn sample(&self) -> Result<ClockRow> {
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
        Ok(sample)
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

/// The switch words of `classes` (by class index from 1): the master word, on as `switch`'s
/// `enabled` says, then each class's, on with it unless the class matches a pattern of its
/// `off`, the kernel class's on with it while its `kernels` holds; and the classes matched.
fn switch_words(classes: &[String], switch: &Switch) -> (Vec<u32>, Vec<String>) {
    let enabled = switch.enabled;
    let mut words = vec![u32::from(enabled)];
    let mut switched_off = Vec::new();
    for class in classes {
        // The kernel probes follow their own word, whatever patterns are switched off.
        if class == KERNEL_CLASS {
            words.push(u32::from(enabled && switch.kernels));
            continue;
        }
        let excluded = switch.off.iter().any(|pattern| matches(pattern, class));
        if excluded {
            switched_off.push(class.clone());
        }
        words.push(u32::from(enabled && !excluded));
    }
    (words, switched_off)
}

/// A duration of `%globaltimer` nanoseconds in the seconds a report's histograms hold.
fn seconds(nanoseconds: u64) -> f64 {
    nanoseconds as f64 * 1e-9
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The master word follows the switch; a class's word follows it unless a pattern of the off
    /// list names the class, whose markers stay off while the others stamp.
    #[test]
    fn a_class_named_off_stays_off_while_the_others_stamp() {
        let classes = ["Attention".to_owned(), "Transformer".to_owned()];
        let switch = |enabled, off: &[&str]| Switch {
            enabled,
            off: off.iter().map(|&pattern| pattern.to_owned()).collect(),
            kernels: true,
        };
        assert_eq!(
            switch_words(&classes, &switch(true, &["*Attention"])),
            (vec![1, 0, 1], vec![classes[0].clone()])
        );
        assert_eq!(
            switch_words(&classes, &switch(true, &[])),
            (vec![1, 1, 1], Vec::new())
        );
        assert_eq!(
            switch_words(&classes, &switch(false, &["*Attention"])),
            (vec![0, 0, 0], vec![classes[0].clone()])
        );
    }

    /// The kernel probes' word follows their own switch while the master word is on, and no
    /// pattern of the classes switched off reaches it.
    #[test]
    fn the_kernel_probes_follow_their_own_switch() {
        let classes = ["Transformer".to_owned(), KERNEL_CLASS.to_owned()];
        let switch = |enabled, kernels| Switch {
            enabled,
            off: vec!["*".to_owned()],
            kernels,
        };
        assert_eq!(switch_words(&classes, &switch(true, true)).0, [1, 0, 1]);
        assert_eq!(switch_words(&classes, &switch(true, false)).0, [1, 0, 0]);
        assert_eq!(switch_words(&classes, &switch(false, true)).0, [0, 0, 0]);
    }

    /// The ranges and launch table of a forward of a step stack `Stack` at `model`: an embedding
    /// outside the stack, two layers of one kind, a layer of the same class that launches other
    /// keys, and a norm inside the stack outside every layer.
    fn forward() -> (
        Vec<ops::context::Range>,
        Vec<(&'static str, String, String)>,
    ) {
        let range = |class: &str, path: &str, depth| ops::context::Range {
            class: class.to_owned(),
            path: path.to_owned(),
            depth,
        };
        let ranges = vec![
            range("Embedding", "embed", 0),
            range("Stack", "model", 0),
            range("Layer", "model.layers.0", 1),
            range("Mlp", "model.layers.0.mlp", 2),
            range("Layer", "model.layers.1", 1),
            range("Mlp", "model.layers.1.mlp", 2),
            range("Layer", "model.layers.2", 1),
            range("Mlp", "model.layers.2.mlp", 2),
        ];
        let launch = |operator, key: &str, path: &str| (operator, key.to_owned(), path.to_owned());
        let table = vec![
            launch("embedding", "e", "embed"),
            launch("rms_norm", "n", "model.layers.0"),
            launch("gemm", "dense", "model.layers.0.mlp"),
            launch("rms_norm", "n", "model.layers.1"),
            launch(ops::breaks::BREAK, "", "model.layers.1"),
            launch("gemm", "dense", "model.layers.1.mlp"),
            launch("rms_norm", "n", "model.layers.2"),
            launch("grouped_gemm", "moe", "model.layers.2.mlp"),
            launch("rms_norm", "n", "model"),
            launch("sample", "s", ""),
        ];
        (ranges, table)
    }

    /// Each kind of layer's chosen instance carries the probes and stands for its kind's layers;
    /// a layer that launches other keys is a kind of its own; every launch outside the layers is
    /// probed once; a graph break is no launch of a layer's kind.
    #[test]
    fn each_kind_of_layer_carries_probes_in_one_instance() {
        let (ranges, table) = forward();
        let plan = Plan::of(&ranges, &table, "Stack", 0);

        assert_eq!(plan.layers("model.layers.0"), Some(2));
        assert_eq!(plan.layers("model.layers.0.mlp"), Some(2));
        assert_eq!(plan.layers("model.layers.1.mlp"), None);
        assert_eq!(plan.layers("model.layers.2.mlp"), Some(1));
        assert_eq!(plan.layers("model"), Some(1));
        assert_eq!(plan.layers("embed"), Some(1));
        assert_eq!(plan.layers(""), Some(1));

        // A position past a kind's layers takes its last.
        let later = Plan::of(&ranges, &table, "Stack", 5);
        assert_eq!(later.layers("model.layers.0.mlp"), None);
        assert_eq!(later.layers("model.layers.1.mlp"), Some(2));
        assert_eq!(later.layers("model.layers.2.mlp"), Some(1));
    }

    /// A forward that never enters the step stack places no kernel probe.
    #[test]
    fn a_forward_outside_the_step_stack_places_no_kernel_probe() {
        let (ranges, table) = forward();
        let plan = Plan::of(&ranges, &table, "Encoder", 0);
        assert_eq!(plan.layers(""), None);
        assert_eq!(plan.layers("model.layers.0"), None);
    }

    /// A kernel probe's entry pairs with its site's next exit; the stamps of other classes and
    /// the step markers make no kernel span.
    #[test]
    fn kernel_stamps_pair_by_site_in_time_order() {
        // Instance 0 is a module's (class 1), instances 1 and 2 are kernel sites (class 2).
        let instance_classes = [1, 2, 2];
        let stamp = |time, marker| Stamp {
            time,
            marker,
            step: 4,
        };
        let mut stamps = [
            stamp(9, STEP_BEGIN),
            stamp(10, 0),
            stamp(11, 2),
            stamp(13, 3),
            stamp(14, 2),
            stamp(15, 3),
            stamp(16, 4),
            stamp(18, 5),
            stamp(20, 1),
        ];

        let spans = kernel_spans(&instance_classes, Some(2), &mut stamps);
        assert_eq!(spans, [(1, 11, 13), (1, 14, 15), (2, 16, 18)]);
        assert!(kernel_spans(&instance_classes, None, &mut stamps).is_empty());
    }

    /// A device time converts to the host clock by the offset of the calibration sample.
    #[test]
    fn a_device_time_converts_through_its_calibration() {
        let clock = ClockRow {
            source: "globaltimer",
            device: Some(0),
            gpu: Some(1_000),
            realtime: 1_700_000_000_000_000_000,
            cupti: None,
            monotonic: None,
            uncertainty: 3_000,
        };
        assert_eq!(host_time(&clock, 1_250), 1_700_000_000_000_000_250);
        assert_eq!(host_time(&clock, 400), 1_699_999_999_999_999_400);
    }

    /// A step's spans go into the device's ring by its batch: its span between its markers, and
    /// each kernel probe pair as a span of its site, on the host clock.
    #[test]
    fn a_steps_spans_name_its_batch_on_the_host_clock() {
        let ring = service::span::Ring::new(64, service::span::Label::None);
        let clock = ClockRow {
            source: "globaltimer",
            device: Some(0),
            gpu: Some(100),
            realtime: 10_100,
            cupti: None,
            monotonic: None,
            uncertainty: 1,
        };
        write_spans(ring, &clock, 77, Some((100, 200)), &[(3, 120, 130)]);

        let spans = service::span::spans(&[(service::span::Label::None, ring.read())], |site| {
            (site == 3).then(|| service::span::Kernel {
                op: "gemm".to_owned(),
                descriptor: "k".to_owned(),
                module: "model.layers.0.mlp".to_owned(),
                layers: 48,
            })
        });
        let named: Vec<(u64, u64, Option<u64>, bool)> = spans
            .iter()
            .map(|span| {
                let kernel = matches!(span.name, service::span::Name::Kernel(_));
                (span.start, span.end, span.batch, kernel)
            })
            .collect();
        assert_eq!(
            named,
            [
                (10_100, 10_200, Some(77), false),
                (10_120, 10_130, Some(77), true)
            ]
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
            batch: u64::from(step),
            stage: Arc::from(stage),
        };
        let mut launched = VecDeque::from([
            entry(1, 10, "decode"),
            entry(2, 11, "prefill"),
            entry(1, 12, "decode"),
            entry(1, 13, "verify"),
        ]);

        let (stage, batch, lost) = release(&mut launched, 12).unwrap();
        assert_eq!((&*stage, batch, lost), ("decode", 12, vec![10]));
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
        installed(device, &keys())
    }

    /// The modules of `wanted` (descriptor keys by operator) on `device`, from the Build Server
    /// of the deployment `device-cases` on this host.
    fn installed(
        device: DeviceId,
        wanted: &BTreeMap<String, BTreeSet<String>>,
    ) -> Vec<KernelEntry> {
        let socket = build_server::socket(
            &service::RuntimeDirectory::of_user().unwrap(),
            "device-cases",
        )
        .unwrap();
        let mut client = build_server::Client::connect(&socket).unwrap();
        let targets = [build_server::Target::arch(device.compute_capability().unwrap()).unwrap()];
        let (packages, routes) =
            crate::modules::load::select(&mut client, &targets, &Default::default(), wanted)
                .unwrap();
        crate::modules::load::install(&mut client, device, &packages, &routes).unwrap()
    }

    /// A captured forward carries a probe pair around each launch of its layers' kind's chosen
    /// instance and of every launch outside its layers, and none around the other instance's:
    /// every replay's kernel spans name, in launch order, the operator, descriptor key and module
    /// path of those entries of the forward's launch table, with the layers each stands for,
    /// within the step's own span on the host clock, though the graph was captured on another
    /// stream than the one it replays on.
    #[test]
    #[ignore = "requires a GPU and the host's Build Server holding the range_marker and rms_norm \
                packages"]
    fn a_layers_kernel_probes_time_each_launch_its_table_states() {
        use tensor::DType;

        const ROWS: usize = 4;
        const HIDDEN: usize = 64;
        let device = DeviceId(0);
        let markers = Markers::install(
            vec![(device, kernels(device))],
            Vec::new(),
            (true, &[], true),
            "Stack".to_owned(),
            Some(0),
        )
        .unwrap();

        // Two layers of one kind, each two normalizations, and one normalization in the stack
        // outside them.
        let forward = |x: &tensor::Tensor, gain: &tensor::Tensor| -> nn::Result<()> {
            nn::profiler::range("Stack", "model", || {
                for layer in 0..2 {
                    nn::profiler::range("Layer", &format!("model.layers.{layer}"), || {
                        let normed = nn::functional::rms_norm(x, gain, 1e-6, false, DType::F32)?;
                        nn::functional::rms_norm(&normed, gain, 1e-6, false, DType::F32)?;
                        nn::Result::Ok(())
                    })?;
                }
                nn::functional::rms_norm(x, gain, 1e-6, false, DType::F32)?;
                Ok(())
            })
        };

        // The profile run: its launch table and the ranges it entered.
        let meta = ops::Context::meta(None);
        meta.record_launches();
        {
            let _guard = meta.enter();
            let weight = |address, dims: &[usize]| {
                ops::retained(address, dims, DType::F32, &ops::meta()).unwrap()
            };
            forward(
                &weight(1 << 40, &[ROWS, HIDDEN]),
                &weight(2 << 40, &[HIDDEN]),
            )
            .unwrap();
        }
        let table: Vec<(&'static str, String, String)> = meta
            .take_launches()
            .into_iter()
            .map(|(operator, key, path, _)| (operator, key, path))
            .collect();
        let ranges = meta.take_ranges();
        let mut wanted: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (operator, key, _) in &table {
            wanted
                .entry((*operator).to_owned())
                .or_default()
                .insert(key.clone());
        }
        let generation = Arc::new(ops::Generation::new(1, installed(device, &wanted)));

        // The capture, over device operands and a region for the activations, on a stream of
        // its own beside the one the graph replays on, as a capture thread's.
        let stream = Stream::new(device).unwrap();
        let beside = Stream::new(device).unwrap();
        let region = DeviceAllocation::linear(device, 1 << 20).unwrap();
        let x = DeviceAllocation::linear(device, 4 * ROWS * HIDDEN).unwrap();
        let gain = DeviceAllocation::linear(device, 4 * HIDDEN).unwrap();
        let on_device = |allocation: &DeviceAllocation, dims: &[usize]| {
            ops::retained(allocation.address(), dims, DType::F32, &ops::cuda(0)).unwrap()
        };
        let context = ops::Context::new(
            0,
            beside.clone(),
            generation,
            platform::DeviceRange::new(DevicePtr::from_address(region.address()), region.len()),
        );
        context.record_launches();
        let capture = platform::Capture::begin(&beside, platform::CaptureMode::Relaxed).unwrap();
        context.set_profiler(profiler(&beside, &stream, 1, &ranges, &table));
        {
            let _guard = context.enter();
            forward(
                &on_device(&x, &[ROWS, HIDDEN]),
                &on_device(&gain, &[HIDDEN]),
            )
            .unwrap();
        }
        context.set_profiler(None);
        let executable = capture.end().unwrap().instantiate().unwrap();
        let captured: Vec<(&'static str, String, String)> = context
            .take_launches()
            .into_iter()
            .map(|(operator, key, path, _)| (operator, key, path))
            .collect();
        assert_eq!(captured, table);

        const BATCH: u64 = 0x5_0000_0007;
        markers
            .bracket(&stream, BATCH, "decode", || executable.launch(&stream))
            .unwrap()
            .unwrap();
        stream.synchronize().unwrap();
        markers.read_device(device);

        let spans: Vec<service::span::Span> =
            service::span::spans(&service::span::records(), |site| markers.kernel(site))
                .into_iter()
                .filter(|span| span.batch == Some(BATCH))
                .collect();
        let step = spans
            .iter()
            .find(|span| span.name == service::span::Name::Stage("step".to_owned()))
            .unwrap();
        let mut timed = Vec::new();
        for span in &spans {
            if let service::span::Name::Kernel(kernel) = &span.name {
                assert!(step.start <= span.start && span.start <= span.end && span.end <= step.end);
                timed.push((
                    kernel.op.clone(),
                    kernel.descriptor.clone(),
                    kernel.module.clone(),
                    kernel.layers,
                ));
            }
        }
        // The first layer is its kind's probed instance and stands for both; the launch outside
        // the layers stands for itself; the second layer carries none.
        let expected: Vec<(String, String, String, u32)> = table
            .iter()
            .filter(|(_, _, path)| path != "model.layers.1")
            .map(|(operator, key, path)| {
                let layers = if path == "model" { 1 } else { 2 };
                ((*operator).to_owned(), key.clone(), path.clone(), layers)
            })
            .collect();
        assert_eq!(expected.len(), 3);
        assert_eq!(timed, expected);
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
            (true, &[], true),
            "Stack".to_owned(),
            None,
        )
        .unwrap();
        let stream = Stream::new(device).unwrap();

        // A forward run once carries no marker; a capture's does.
        assert!(profiler(&stream, &stream, 0, &[], &[]).is_none());
        let capture = platform::Capture::begin(&stream, platform::CaptureMode::Relaxed).unwrap();
        let captured = profiler(&stream, &stream, 0, &[], &[]).unwrap();

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
            histograms[&step].count()
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
