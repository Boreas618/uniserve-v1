//! `Graphs`: the single implementation of the profile run, the rehearsal, capture, replay and the
//! switch of generations, for every runner.
//!
//! A runner computes one forward per bucket ([`Forward::forward`]). `Graphs` runs it four ways,
//! and nothing else in the crate does:
//!
//! - **Profile run.** At load the forward runs under a meta context: it launches nothing and
//!   yields the bucket's launch table (operator, key and module path of every launch, in launch
//!   order), the peak of the allocator region, the bytes of its host-to-device copies and the
//!   most one paired call of it carves from a shadow region. The keys are what the engine
//!   resolves kernels for, and the peaks and the shadow bytes are what a device's memory is
//!   divided by.
//! - **Rehearsal.** A kernel whose first use is host work a capture cannot record (a native
//!   entry that builds its runner or allocates its workspace on first use) is first used by a
//!   rehearsal ([`Graphs::rehearse`]): the forward runs eagerly over idle inputs, under a
//!   rehearsal context ([`ops::Context::rehearsal`]), in device memory of its own, so it writes no
//!   tensor a serving forward reads and meets no other rank. Its kernels are ordered on the
//!   runner's stream, between the serving replays: a native library keeps scratch of its own per
//!   shape (cuBLASLt's workspace of a matmul) that the graphs which call the same library use as
//!   well, and a rehearsal running beside a replay on another stream writes it while the replay
//!   reads it.
//! - **Capture.** The forward is captured on a stream beside the serving one
//!   ([`Graphs::capture`]), with its launch table, its input copies and its output tensors, and
//!   instantiated. A capture launches nothing: it only records, so it runs while the runner
//!   serves.
//! - **Replay.** [`Graphs::prepare_inputs`] takes a slot of the input buffer, lets the runner
//!   write the bytes of its input tensors and, when the graph has input copies, runs the forward
//!   under a meta context over the same region with the kernels of the graph it prepares for,
//!   so the host values the model computes land in the slot at the offsets that graph's capture
//!   recorded. [`Graphs::replay`] then issues the slot's copies as one batched copy and launches
//!   the graph. Nothing is called back from the stream.
//!
//! ## Three kinds of graphs
//!
//! A runner's own graphs serve: they hold the captured graphs and replay them on the runner's
//! stream. Its worker's capture thread holds two more over the runner's second forward instance
//! (`Runner::capturer`): graphs **beside** the runner's ([`Graphs::beside`]), whose context
//! launches on a capture stream with an allocator over the runner's region, so a capture records
//! the addresses a capture of the runner would; and a **rehearsal**'s ([`Graphs::rehearsal`]),
//! whose input tensors and allocator region are a device allocation of its own, held while a
//! generation is rehearsed ([`Graphs::release_rehearsal`]).
//!
//! ## Memory
//!
//! A runner's region is one range of device memory: the block of its input tensors at fixed
//! addresses, then the region of the context's allocator, which is emptied at the start of
//! every forward. A forward of one bucket therefore receives the same addresses when it is
//! captured, beside or not, and whenever it runs again, and a graph's output tensors are the
//! ones its capture allocated. They are read after the replay and before the next forward on the
//! runner. Two generations' graphs of a bucket address the same region: only one of them
//! replays at a time, on the runner's stream.
//!
//! ## Generations
//!
//! A new generation of kernels is prepared beside the serving graphs, one bucket at a time, by
//! the capture thread: a bucket none of whose launches changed keeps its graph, and any other is
//! captured again under the new kernels ([`Graphs::examine`], [`Graphs::capture`]); no graph is
//! ever modified. Each bucket's outcome is installed here on the worker thread
//! ([`Graphs::install`]). A bucket switches to the new generation once every rank of the groups
//! that hold it installed it ([`Graphs::switch_buckets`]), and the generation serves whole once
//! every bucket switched ([`Graphs::switch`]). A replay takes the newest graph of its bucket at or
//! below what the bucket switched to and what its batch's kernel stamp allows
//! ([`Graphs::stamp`]). The replaced graphs stay until the trial is committed, and serve again
//! after a rollback.
//!
//! Before the switch, each operator whose kernel changed runs a paired forward
//! ([`Graphs::shadow`]): the new kernel beside the one it replaces on the same inputs, compared
//! under the operator's tolerance.
//!
//! ## Without a device
//!
//! [`Graphs::meta`] holds no stream. It runs profile runs, and the whole generation state
//! machine with every device step left out: a capture records the launch table and the input
//! copies of a meta forward and no executable, a rehearsal runs nothing, and a replay is refused.
//! A runner built over it makes its input tensors on the meta device.

mod capture;
mod input_buffer;
mod load;
mod shadow;

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap};

use proto::report::KernelLaunch;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ops::Generation;
use ops::allocator::ALIGNMENT;
use ops::breaks::{BREAK, HostWork};
use ops::collective::Meet;
use ops::moe::ExchangeTransport;
use ops::{Context, ContextGuard, Group};
use platform::{DeviceAllocation, DeviceRange, Event, EventFlags, Stream};
use tensor::{DType, Device, Shape, Tensor};

use self::capture::{Executable, Recording, kernels_of, take_break};
use self::input_buffer::InputBuffer;
use self::load::Counters;
use self::shadow::Shadow;
use crate::runner::Forward;
use crate::runner::buckets::Bucket;
use crate::worker::{Error, Result};

/// A launch table: `(operator, key, module path)` per launch, in launch order.
type Table = Vec<(&'static str, String, String)>;

/// A launch table as a context records it (`ops::Context::record_launches`): each launch with
/// the bytes a paired launch of it carves from a shadow region (`ops::Call::shadow_bytes`).
type Recorded = Vec<(&'static str, String, String, u64)>;

/// Input copies: `(offset in a slot's values, device address, bytes)` per copy, in the order the
/// forward makes them.
type Copies = Vec<(u64, u64, u64)>;

/// The axes of a forward's partition as [`Graphs::set_axes`] takes them: per axis its group and
/// this rank's place in it.
type Axes = Vec<(Option<Arc<Group>>, u64, u64)>;

/// What a generation needs of one bucket, as the capture thread found it ([`Graphs::examine`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Plan {
    /// Whether the bucket is captured under the generation's kernels: its launch table changed,
    /// or a kernel it launches did. A bucket that is not keeps the graph it serves from.
    pub capture: bool,
    /// Whether a capture of the bucket is preceded by a rehearsal ([`Graphs::rehearse`]): it
    /// launches a native kernel the serving generation does not hold, whose first use a capture
    /// cannot record.
    pub rehearse: bool,
}

/// What one capture cost, as the capture thread measured it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Costs {
    /// The host time of the capture: the forward run on the capturing stream.
    pub capture: Duration,
    /// The host time of the instantiation of the captured segments.
    pub instantiate: Duration,
    /// The device bytes the instantiation took, as the device's free memory fell across it;
    /// zero for graphs that hold no device.
    pub bytes: u64,
    /// The segments of the graph: one more than its graph breaks.
    pub segments: usize,
    /// The kernel nodes one replay of the graph launches.
    pub kernel_nodes: u64,
}

/// The captured graphs of one runner, the context its forwards compute under, and the
/// generations of kernels they launch.
pub struct Graphs {
    /// What issues on the device; `None` for graphs that hold no device.
    device: Option<Cuda>,
    /// The meta context of profile runs: it counts and holds no memory.
    profile: Context,
    /// The meta context of the forward of `prepare_inputs`: over the launching context's
    /// allocator region, so its copies' destinations are the capture's, and lent the values
    /// area of the slot being prepared. Graphs that hold no device count instead.
    preparing: Context,
    /// The bytes of the runner's input tensors.
    inputs: u64,
    /// Slots of the input buffer.
    slots: usize,
    /// Whether these are a rehearsal's graphs ([`Graphs::rehearsal`]).
    rehearses: bool,
    /// The device address of the runner's input tensors: the start of the runner's region, and
    /// for a rehearsal's graphs the start of the rehearsal's memory while it rehearses.
    inputs_at: Cell<u64>,
    state: RefCell<State>,
}

/// The device side of a runner's graphs.
struct Cuda {
    stream: Stream,
    /// The stream these graphs' captures are replayed on: their own, or for graphs beside a
    /// runner's, the runner's, whose step markers the captured range markers read.
    replays_on: Stream,
    /// The context that launches: a serving runner's eager and paired forwards compute under
    /// it, and graphs beside them capture under it. A rehearsal's graphs derive each rehearsal's
    /// context from it.
    launching: Context,
    /// The runner's region: its input tensors, then its allocator's.
    region: DeviceRange,
    /// Recorded on the stream behind each replay and eager forward: what a reader of the
    /// outputs waits on.
    done: Event,
    /// The device memory of a rehearsal's graphs, while a generation is rehearsed: a region as
    /// large as the runner's.
    rehearsal: RefCell<Option<DeviceAllocation>>,
    /// The runner's expert load, which the launching context lends its forwards, once it was
    /// given its routed layers ([`Graphs::set_expert_load`]).
    load: RefCell<Option<Counters>>,
}

/// What changes while a runner serves.
struct State {
    /// The generation that serves whole: eager forwards launch its kernels, and every bucket not
    /// switched ahead of it replays its graphs.
    serving: Arc<Generation>,
    /// The generation update in flight.
    update: Update,
    /// The kernel generation the step being prepared was stamped with ([`Graphs::stamp`]): the
    /// newest generation its replay may take a graph of.
    stamp: Option<u64>,
    /// The launch table of each profiled bucket.
    profiles: HashMap<Bucket, Table>,
    /// The most one paired call of each profiled bucket's forward carves from a shadow region
    /// ([`shadowed`]).
    shadows: HashMap<Bucket, u64>,
    /// The values area of a slot that holds the copies of each profiled bucket's forward at any
    /// offsets ([`input_bound`]).
    inputs: HashMap<Bucket, u64>,
    /// The graph of each captured bucket, by the generation it was captured for.
    graphs: HashMap<(Bucket, u64), Graph>,
    /// The replays of each bucket since the graphs were made: what orders a generation's
    /// captures, the most used bucket first ([`Graphs::prepare`]).
    uses: HashMap<Bucket, u64>,
    /// The input buffer: page-locked for graphs that serve or rehearse, and made at the first
    /// use over ordinary host memory for graphs that hold no device or capture beside.
    buffer: Option<InputBuffer>,
    /// The step prepared for a bucket, until the replay, the eager forward or the paired
    /// forward that issues it.
    prepared: Option<Prepared>,
    /// The step the next replay or eager forward is, when the worker stated one
    /// ([`Graphs::mark_step`]): its number and its stage.
    step: Option<(u64, &'static str)>,
    /// What paired forwards are compared with, once the device's scratch pool gave its region.
    shadow: Option<Shadow>,
    /// The pairs the paired forwards of the generation being prepared could not compare, each
    /// with the reason.
    uncompared: Vec<String>,
    /// The kernels the serving forwards launched since the graphs were made: a replay's
    /// captured kernel nodes, an eager forward's launches ([`Graphs::launches`]).
    launches: u64,
    /// The serving forwards since the graphs were made, by how they ran: graphs replayed, and
    /// forwards run eagerly ([`Graphs::forwards`]).
    replays: u64,
    eager: u64,
    /// The serving forwards not yet counted by operator and key, by bucket and the generation
    /// of the graph replayed (`None` for a forward run eagerly, which launched what the
    /// bucket's profile run did). A step only adds to its entry; [`Graphs::take_kernels`]
    /// expands the entries through their launch tables, and a graph is expanded before it is
    /// released ([`State::settle`]).
    pending: HashMap<(Bucket, Option<u64>), u64>,
    /// The operator launches counted from `pending` and not yet taken.
    kernels: BTreeMap<KernelLaunch, u64>,
}

/// The step prepared for a bucket ([`Graphs::prepare_inputs`], [`Graphs::unpaired`]).
enum Prepared {
    /// The bucket's inputs, in this slot of the input buffer, for the graph of the generation
    /// of this number; `None` for an eager forward.
    Inputs(Bucket, usize, Option<u64>),
    /// A paired forward of the bucket that its runner prepared no inputs for, with the reason:
    /// it runs nothing, and the operator it was to compare is reported uncompared.
    Unpaired(Bucket, String),
}

impl Prepared {
    /// The bucket the step is of.
    const fn bucket(&self) -> &Bucket {
        match self {
            Self::Inputs(bucket, _, _) | Self::Unpaired(bucket, _) => bucket,
        }
    }
}

/// Where a generation update stands.
enum Update {
    /// One generation serves and no other is held.
    Steady,
    /// `next` is prepared beside the serving generation. `pending` are the captured buckets
    /// whose outcome the capture thread has not installed; `switched` the installed ones that
    /// serve from `next` ahead of the others ([`Graphs::switch_buckets`]).
    Preparing {
        next: Arc<Generation>,
        pending: BTreeSet<Bucket>,
        switched: BTreeSet<Bucket>,
    },
    /// The prepared generation serves whole, on trial, and `kept` is the one it replaced.
    Trial { kept: Arc<Generation> },
}

/// One bucket's captured forward, as a capture made it.
pub struct Graph {
    /// The launches of the forward, with a [`BREAK`] entry at each graph break.
    table: Table,
    /// The input copies a replay issues before the launch.
    copies: Copies,
    /// The forward's outputs, at the addresses the capture allocated.
    outputs: Vec<Tensor>,
    /// The kernels the graph was captured with: what the forward of `prepare_inputs` runs
    /// under before its replay.
    kernels: Arc<Generation>,
    /// What a replay launches; `None` for graphs that hold no device.
    executable: Option<Executable>,
    /// What the capture cost.
    costs: Costs,
}

impl Graph {
    /// What the capture of this graph cost.
    pub fn costs(&self) -> Costs {
        self.costs
    }
}

impl Graphs {
    /// Graphs that hold no device, serving `kernels`: profile runs, and captures and generation
    /// updates without their device steps. `inputs` is the bytes of the runner's input tensors.
    pub fn meta(kernels: Arc<Generation>, inputs: u64) -> Self {
        Self {
            device: None,
            profile: Context::meta(None),
            preparing: Context::meta(None),
            inputs,
            slots: 2,
            rehearses: false,
            inputs_at: Cell::new(0),
            state: RefCell::new(State::new(kernels, None)),
        }
    }

    /// Graphs that capture and replay on `stream` with `kernels`.
    ///
    /// `region` is the runner's region of device memory, at least [`Graphs::peak`] bytes of a
    /// profile run of the same runner: its start holds the `inputs` bytes of the runner's input
    /// tensors and the rest is the allocator's. `values` is the profile run's
    /// [`Graphs::input_peak`], the bytes of host values one replay copies, and `slots` the
    /// replays whose copies may be in flight at once.
    ///
    /// The caller keeps `region` mapped at its address for as long as these graphs, and the
    /// graphs beside them, exist.
    ///
    /// # Errors
    ///
    /// When the region cannot hold the input tensors, or the page-locked input buffer cannot be
    /// allocated.
    pub fn new(
        stream: Stream,
        kernels: Arc<Generation>,
        region: DeviceRange,
        inputs: u64,
        values: u64,
        slots: usize,
    ) -> Result<Self> {
        let device = stream.device();
        let ordinal = usize::try_from(device.0)
            .map_err(|_| Error::Invalid(format!("{device} names no GPU ordinal")))?;
        let (base, allocator) = split_region(region, inputs)?;

        let buffer = InputBuffer::pinned(device, inputs, values, slots)?;
        let cuda = Cuda {
            launching: Context::new(ordinal, stream.clone(), Arc::clone(&kernels), allocator),
            replays_on: stream.clone(),
            stream,
            region,
            done: Event::new(EventFlags::default()),
            rehearsal: RefCell::new(None),
            load: RefCell::new(None),
        };

        Ok(Self {
            device: Some(cuda),
            profile: Context::meta(None),
            preparing: Context::meta(Some(allocator)),
            inputs,
            slots,
            rehearses: false,
            inputs_at: Cell::new(base),
            state: RefCell::new(State::new(kernels, Some(buffer))),
        })
    }

    /// Graphs beside these, for the capture instance of the runner's forward
    /// (`Runner::capturer`): their context launches the same kernels through groups beside
    /// these graphs' ([`ops::Context::beside`]) on a capture stream of their own, and allocates
    /// over the same region, so a forward captured under them records the addresses one
    /// captured under these graphs would, and the graph replays here. Graphs that hold no
    /// device give graphs that hold none.
    ///
    /// Make them once the runner's groups and expert exchange are set
    /// ([`Graphs::set_axes`], [`Graphs::set_exchange`]): the context beside takes them as they
    /// stand.
    ///
    /// # Errors
    ///
    /// When the capture stream or a group's side stream cannot be created.
    pub fn beside(&self) -> Result<Self> {
        let state = self.state.borrow();
        let kernels = Arc::clone(&state.serving);

        let Some(cuda) = &self.device else {
            let graphs = Self::meta(kernels, self.inputs);
            graphs.profile.set_axes(self.splits()?)?;
            graphs.preparing.set_axes(self.splits()?)?;
            return Ok(graphs);
        };

        let stream = Stream::new(cuda.stream.device())?;
        let (base, allocator) = split_region(cuda.region, self.inputs)?;
        let graphs = Self {
            device: Some(Cuda {
                launching: cuda.launching.beside(stream.clone())?,
                stream,
                replays_on: cuda.replays_on.clone(),
                region: cuda.region,
                done: Event::new(EventFlags::default()),
                rehearsal: RefCell::new(None),
                load: RefCell::new(None),
            }),
            profile: Context::meta(None),
            preparing: Context::meta(Some(allocator)),
            inputs: self.inputs,
            slots: 1,
            rehearses: false,
            inputs_at: Cell::new(base),
            state: RefCell::new(State::new(kernels, None)),
        };
        graphs.profile.set_axes(self.splits()?)?;
        graphs.preparing.set_axes(self.splits()?)?;
        Ok(graphs)
    }

    /// Graphs of a rehearsal, for the rehearsal instance of the runner's forward
    /// (`Runner::capturer`): their input tensors and allocator region are a device allocation as
    /// large as the runner's region, made at the first rehearsal and held until
    /// [`Graphs::release_rehearsal`], and each rehearsal runs under a rehearsal context
    /// ([`ops::Context::rehearsal`]) on the runner's stream, between its replays (module
    /// documentation). Graphs that hold no device give graphs that hold none, whose rehearsals
    /// run nothing.
    ///
    /// # Errors
    ///
    /// When the stream, a group's side stream or the page-locked input buffer cannot be
    /// created.
    pub fn rehearsal(&self) -> Result<Self> {
        let Some(cuda) = &self.device else {
            return self.beside();
        };

        let state = self.state.borrow();
        let kernels = Arc::clone(&state.serving);
        let values = self.input_peak();
        let stream = cuda.stream.clone();
        let buffer = InputBuffer::pinned(cuda.stream.device(), self.inputs, values, 1)?;

        let graphs = Self {
            device: Some(Cuda {
                launching: cuda.launching.beside(stream.clone())?,
                stream,
                replays_on: cuda.replays_on.clone(),
                region: cuda.region,
                done: Event::new(EventFlags::default()),
                rehearsal: RefCell::new(None),
                load: RefCell::new(None),
            }),
            profile: Context::meta(None),
            preparing: Context::meta(None),
            inputs: self.inputs,
            slots: 1,
            rehearses: true,
            // No input tensor of a rehearsal's forward lies anywhere until it rehearses.
            inputs_at: Cell::new(0),
            state: RefCell::new(State::new(kernels, Some(buffer))),
        };
        graphs.profile.set_axes(self.splits()?)?;
        graphs.preparing.set_axes(self.splits()?)?;
        Ok(graphs)
    }

    /// Graphs of another shape of the runner, in these graphs' place (`Runner::reshape`): a
    /// kernel generation that captures the runner's graphs for other buckets, candidates or
    /// drafts lays its input tensors and its allocations out otherwise. Their forwards issue on
    /// the same stream, through the same groups, expert exchange and expert load, under
    /// `kernels`, the generation's; `inputs` is the bytes of the shape's input tensors, which lie
    /// at the start of `region`, the allocator's region following them, and `values` and the
    /// input buffer are as [`Graphs::new`] takes them.
    ///
    /// `region` is these graphs' own when it is `None`: the two shapes' graphs replay on one
    /// stream, in turn, so they address one region as two generations' graphs of a bucket do.
    /// The new graphs hold no graph: each bucket of the shape is captured for them under
    /// `kernels` and installed as a warm-up installs one ([`Graphs::install`]), and they serve once
    /// the runner switched to the shape ([`Graphs::take_over`]). Graphs that hold no device give
    /// graphs that hold none.
    ///
    /// # Errors
    ///
    /// When the region cannot hold the input tensors, or the page-locked input buffer cannot be
    /// allocated.
    pub fn over(
        &self,
        kernels: Arc<Generation>,
        region: Option<DeviceRange>,
        inputs: u64,
        values: u64,
    ) -> Result<Self> {
        let Some(cuda) = &self.device else {
            return self.measuring(kernels, inputs);
        };

        let region = region.unwrap_or(cuda.region);
        let (base, allocator) = split_region(region, inputs)?;
        let buffer = InputBuffer::pinned(cuda.stream.device(), inputs, values, self.slots)?;
        let launching = cuda.launching.over(allocator)?;
        launching.set_kernels(Arc::clone(&kernels));

        let graphs = Self {
            device: Some(Cuda {
                launching,
                stream: cuda.stream.clone(),
                replays_on: cuda.replays_on.clone(),
                region,
                done: Event::new(EventFlags::default()),
                rehearsal: RefCell::new(None),
                load: RefCell::new(None),
            }),
            profile: Context::meta(None),
            preparing: Context::meta(Some(allocator)),
            inputs,
            slots: self.slots,
            rehearses: false,
            inputs_at: Cell::new(base),
            state: RefCell::new(State::new(kernels, Some(buffer))),
        };
        graphs.profile.set_axes(self.splits()?)?;
        graphs.preparing.set_axes(self.splits()?)?;
        Ok(graphs)
    }

    /// Graphs that hold no device, serving `kernels`, for a forward instance of another shape of
    /// the runner whose input tensors take `inputs` bytes: what a profile run measures that
    /// shape over ([`Graphs::profile_run`]) before anything of it is allocated. Their meta
    /// contexts take these graphs' splits.
    ///
    /// # Errors
    ///
    /// As [`Graphs::set_axes`].
    pub fn measuring(&self, kernels: Arc<Generation>, inputs: u64) -> Result<Self> {
        let graphs = Self::meta(kernels, inputs);
        graphs.profile.set_axes(self.splits()?)?;
        graphs.preparing.set_axes(self.splits()?)?;
        Ok(graphs)
    }

    /// The bytes of the runner's region: what a forward of any shape these graphs' region serves
    /// allocates in, input tensors included ([`Graphs::peak`]). Zero for graphs that hold no
    /// device.
    pub(crate) fn region_bytes(&self) -> u64 {
        self.device
            .as_ref()
            .map_or(0, |cuda| cuda.region.bytes() as u64)
    }

    /// Takes over from `from`, the graphs of the shape the runner served until now, what counts
    /// the runner's serving forwards across shapes: the expert load's counters, the shadow region
    /// of paired forwards, the replays of each bucket, and the launches and forwards counted so
    /// far. These graphs serve from here on, and a later [`Graphs::take_over`] gives it back.
    ///
    /// # Errors
    ///
    /// When the device cannot be waited for: `from`'s forwards are counted once the device
    /// passed them.
    pub(crate) fn take_over(&self, from: &Self) -> Result<()> {
        from.settle()?;

        if let (Some(cuda), Some(held)) = (&self.device, &from.device) {
            let counters = held.load.borrow_mut().take();
            *cuda.load.borrow_mut() = counters;
        }

        let mut held = from.state.borrow_mut();
        held.settle();
        let mut state = self.state.borrow_mut();
        state.shadow = held.shadow.take();
        state.uses = std::mem::take(&mut held.uses);
        state.kernels = std::mem::take(&mut held.kernels);
        state.launches = held.launches;
        state.replays = held.replays;
        state.eager = held.eager;
        Ok(())
    }

    /// Whether these are a rehearsal's graphs ([`Graphs::rehearsal`]): their input tensors lie
    /// in memory of their own, so what a forward over them writes at a fixed address a served
    /// step reads must be the forward instance's own.
    pub fn rehearses(&self) -> bool {
        self.rehearses
    }

    /// The stream these graphs issue on; `None` for graphs that hold no device.
    pub(crate) fn stream(&self) -> Option<&Stream> {
        self.device.as_ref().map(|cuda| &cuda.stream)
    }

    /// The device a runner makes its tensors on: the GPU these graphs issue on, or the meta
    /// device for graphs that hold none.
    pub(crate) fn device(&self) -> Device {
        match &self.device {
            Some(cuda) => ops::cuda(cuda.stream.device().0 as usize),
            None => ops::meta(),
        }
    }

    /// The runner's input tensor of `shape` elements of `dtype` at byte `offset` of its input
    /// tensors' block: an external tensor at a fixed address, which a graph reads on every
    /// replay and which the batched copy of [`Graphs::replay`] fills from the same offset of
    /// the slot's columns.
    ///
    /// # Errors
    ///
    /// When the tensor does not lie inside the `inputs` bytes these graphs were made for.
    pub fn input(&self, offset: u64, shape: impl Into<Shape>, dtype: DType) -> Result<Tensor> {
        let shape = shape.into();
        let element = ops::abi::bytes(dtype).ok_or_else(|| {
            Error::Invalid(format!(
                "an input tensor of {} elements has no whole-byte layout",
                ops::abi::name(dtype)
            ))
        })?;
        let bytes = shape.elem_count() as u64 * element;
        if !offset.is_multiple_of(ALIGNMENT) || offset + bytes > self.inputs {
            return Err(Error::Invalid(format!(
                "an input tensor of {bytes} bytes at offset {offset} is unaligned or lies \
                 outside the {} bytes of the runner's input tensors",
                self.inputs
            )));
        }

        let address = self
            .device
            .as_ref()
            .map_or(0, |_| self.inputs_at.get() + offset);

        // SAFETY: on a GPU the address lies in the input block at the start of the runner's
        // region, which the caller of `Graphs::new` keeps mapped for as long as these graphs and
        // the graphs beside them exist and which only the copies the runner's graphs issue
        // write; or in a rehearsal's memory, which these graphs hold while it rehearses. On the
        // meta device nothing is read.
        Ok(unsafe { ops::external(address, shape, dtype, &self.device()) }?)
    }

    /// Installs the launching context on this thread until the guard drops: what a runner reads
    /// a forward's output tensors under. The read waits for the stream.
    ///
    /// # Errors
    ///
    /// For graphs that hold no device.
    pub fn enter(&self) -> Result<ContextGuard<'_>> {
        Ok(self.cuda()?.launching.enter())
    }

    /// Waits for the device to pass everything these graphs issued: the last replay, eager
    /// forward or paired forward. Graphs that hold no device have nothing in flight.
    ///
    /// # Errors
    ///
    /// When the event cannot be waited for.
    pub fn wait(&self) -> Result<()> {
        self.settle()
    }

    /// States the step the next replay or eager forward of these graphs is: its number and its
    /// stage (the call kind of its batch). The forward is then issued between the step's two
    /// range markers, the end marker right behind its last launch, so the time between them is
    /// the device's and holds nothing of the host's read of the outputs. Without range markers
    /// in the process nothing is launched.
    pub(crate) fn mark_step(&self, step: u64, stage: &'static str) {
        self.state.borrow_mut().step = Some((step, stage));
    }

    /// States the kernel generation the batch whose step is prepared next was stamped with
    /// (`Batch::kernels`): its replay takes the newest graph of its bucket at or below it and at
    /// or below what the bucket switched to. The ranks of a group switch a bucket between the
    /// same two jobs and every rank's batch carries the same stamp, so every rank replays a
    /// batch with graphs of one generation.
    pub fn stamp(&self, kernels: u64) {
        self.state.borrow_mut().stamp = Some(kernels);
    }

    /// The operators the generation being prepared launches with another kernel than the
    /// serving one under the same key in `bucket`'s graph, in launch order, each once: what a
    /// paired forward of the bucket compares ([`Graphs::shadow`]). A key only one of the two
    /// generations holds has no pair. Empty when no generation is being prepared.
    pub(crate) fn replaced(&self, bucket: &Bucket) -> Vec<&'static str> {
        let state = self.state.borrow();
        let Update::Preparing { next, .. } = &state.update else {
            return Vec::new();
        };
        let Some((_, graph)) = state.served(bucket) else {
            return Vec::new();
        };

        let mut operators = Vec::new();
        for (operator, key, _) in &graph.table {
            let replaced = matches!(
                (state.serving.kernel(operator, key), next.kernel(operator, key)),
                (Ok(held), Ok(new)) if !Arc::ptr_eq(held, new)
            );
            if replaced && !operators.contains(operator) {
                operators.push(*operator);
            }
        }
        operators
    }

    /// The group of the runner's partitioned forwards: this rank is member `rank` of `ranks`,
    /// and collectives are issued through `group`. The meta contexts take the split alone.
    ///
    /// # Errors
    ///
    /// As `ops::Context::set_group`.
    pub(crate) fn set_group(&self, group: Option<Arc<Group>>, rank: u64, ranks: u64) -> Result<()> {
        self.set_axes(vec![(group, rank, ranks)])
    }

    /// The axes of the runner's forwards partitioned along several: per axis, in the order the
    /// model addresses them by (`nn::collective::along`), its group and this rank's place in it
    /// as [`Graphs::set_group`] takes one. The meta contexts take the splits alone.
    ///
    /// # Errors
    ///
    /// As `ops::Context::set_axes`.
    pub(crate) fn set_axes(&self, axes: Axes) -> Result<()> {
        let splits: Axes = axes
            .iter()
            .map(|(_, rank, ranks)| (None, *rank, *ranks))
            .collect();
        self.profile.set_axes(splits.clone())?;
        self.preparing.set_axes(splits)?;

        if let Some(cuda) = &self.device {
            cuda.launching.set_axes(axes)?;
        }
        Ok(())
    }

    /// The expert exchange of the runner's forwards (`ops::Context::set_exchange`).
    pub(crate) fn set_exchange(&self, transport: Option<ExchangeTransport>, meet: Option<Meet>) {
        if let Some(cuda) = &self.device {
            cuda.launching.set_exchange(transport, meet);
        }
    }

    /// Counts the tokens each expert of `layers`, the model's routed layers by module path with
    /// their expert counts, takes in the serving forwards of these graphs: zeroed counters on
    /// the device, which the launching context lends the forwards as their expert load
    /// (`ops::moe::ExpertLoad`) and [`Graphs::take_expert_tokens`] reads. Graphs that hold no
    /// device count nothing.
    ///
    /// Set it before graphs beside these are made ([`Graphs::beside`]): their context lends the
    /// load as it stands, so the graphs they capture count into the same counters. A rehearsal
    /// and a paired forward count nothing.
    ///
    /// # Errors
    ///
    /// The driver's refusal of the counters' memory, or a layer of no expert.
    pub fn set_expert_load(&self, layers: &BTreeMap<String, u32>) -> Result<()> {
        let Some(cuda) = &self.device else {
            return Ok(());
        };

        let (counters, load) = Counters::new(layers, &cuda.stream)?;
        cuda.launching.set_expert_load(Some(load));
        *cuda.load.borrow_mut() = Some(counters);
        Ok(())
    }

    /// The tokens each routed layer's experts took in the serving forwards these graphs issued
    /// up to the previous take, since the take before it, by layer module path and the model's
    /// own expert id, of the layers whose experts took any. The device is not waited for: the
    /// counts of a take's own period are read behind the stream and stated by the next one.
    /// Empty for graphs that count no expert load.
    ///
    /// # Errors
    ///
    /// The driver's refusal of the read.
    pub fn take_expert_tokens(&self) -> Result<BTreeMap<String, Vec<u64>>> {
        let Some(cuda) = &self.device else {
            return Ok(BTreeMap::new());
        };

        match cuda.load.borrow_mut().as_mut() {
            Some(counters) => counters.take(),
            None => Ok(BTreeMap::new()),
        }
    }

    /// The splits of the axes the profile context holds, with no group: what the meta contexts
    /// of graphs beside these take.
    fn splits(&self) -> Result<Axes> {
        Ok(self
            .profile
            .splits()
            .into_iter()
            .map(|(rank, ranks)| (None, rank, ranks))
            .collect())
    }

    // ---- The profile run ----

    /// Runs the forward of every bucket `runner` states, the captured ones and the eager ones
    /// ([`Forward::eager_buckets`]), under the meta context and keeps each bucket's launch
    /// table and the most one paired call of it carves from a shadow region.
    ///
    /// # Errors
    ///
    /// What a forward refuses.
    pub fn profile_run(&self, runner: &dyn Forward) -> Result<()> {
        for bucket in runner.buckets().into_iter().chain(runner.eager_buckets()) {
            self.profile(runner, &bucket)?;
        }
        Ok(())
    }

    /// Profiles `bucket`: its forward under the meta profile context, whose launch table and
    /// largest paired call ([`shadowed`]) these graphs keep.
    fn profile(&self, runner: &dyn Forward, bucket: &Bucket) -> Result<Table> {
        let (table, copies, shadow) = self.meta_forward(&self.profile, runner, bucket)?;

        let mut state = self.state.borrow_mut();
        state.profiles.insert(*bucket, table.clone());
        state.shadows.insert(*bucket, shadow);
        state.inputs.insert(*bucket, input_bound(&copies));
        Ok(table)
    }

    /// The kernels the profiled buckets launch, as `(operator, key)`: what the engine resolves
    /// for a generation these graphs serve.
    pub fn keys(&self) -> BTreeSet<(&'static str, String)> {
        let state = self.state.borrow();

        state
            .profiles
            .values()
            .flatten()
            .filter(|(operator, _, _)| *operator != BREAK)
            .map(|(operator, key, _)| (*operator, key.clone()))
            .collect()
    }

    /// The bytes of device memory a runner of the profiled buckets needs for its region: its
    /// input tensors, and the furthest byte any profiled forward allocated.
    pub fn peak(&self) -> u64 {
        // One alignment unit covers a region whose first address is not aligned.
        ALIGNMENT + self.inputs.next_multiple_of(ALIGNMENT) + self.profile.peak()
    }

    /// The bytes of the shadow region the paired forwards of the profiled buckets need: the most
    /// one launch of any profiled forward carves from it, its outputs and the operands it writes
    /// in place (`ops::Call::shadow_bytes`), and `cache_writes` beside that.
    ///
    /// A paired forward compares each paired call as soon as both kernels ran it, so the region
    /// holds one call's twins at a time ([`Graphs::shadow`]). A paired call is a call of the
    /// profiled forward over other cache pages: `cache_writes` is what a cache write of a
    /// paired forward writes in place beyond what the profiled forward's did, the scratch pages'
    /// planes of one layer (the runner states it), zero for a runner whose paired forwards write
    /// no other pages. A call that needs more compares nothing, and its batch is reported
    /// uncompared ([`Graphs::shadow`]).
    pub fn shadow_peak(&self, cache_writes: u64) -> u64 {
        let call = self
            .state
            .borrow()
            .shadows
            .values()
            .copied()
            .max()
            .unwrap_or(0);

        call + cache_writes
    }

    /// The bytes of the values area of one slot of the input buffer: what the host values of
    /// any profiled forward take there, each copy at the most its alignment adds to it
    /// ([`input_bound`]). A forward of no more copies and no more bytes than a profiled one
    /// fits whatever its sizes, which is what bounds an eager forward over shapes no profile
    /// ran: its runner profiles a forward that dominates it in both.
    pub fn input_peak(&self) -> u64 {
        let state = self.state.borrow();
        let bound = state.inputs.values().copied().max().unwrap_or(0);

        bound.max(self.profile.input_peak())
    }

    /// The launch table of `bucket`: its serving graph's when it is captured, otherwise its
    /// profile run's; `None` for a bucket neither captured nor profiled.
    pub fn launch_table(&self, bucket: &Bucket) -> Option<Vec<(&'static str, String, String)>> {
        let state = self.state.borrow();

        match state.served(bucket) {
            Some((_, graph)) => Some(graph.table.clone()),
            None => state.profiles.get(bucket).cloned(),
        }
    }

    /// The input copies of `bucket`'s serving graph, as
    /// `(offset in the slot's values, device address, bytes)`; `None` for an uncaptured bucket.
    pub fn input_copies(&self, bucket: &Bucket) -> Option<Vec<(u64, u64, u64)>> {
        let state = self.state.borrow();
        state.served(bucket).map(|(_, graph)| graph.copies.clone())
    }

    // ---- Replay ----

    /// The smallest captured bucket the serving generation serves from that `holds` a batch.
    ///
    /// `None` is the caller's error once warm-up has captured the runner's buckets: a batch no
    /// bucket holds is not served from another graph.
    pub fn select(&self, holds: impl Fn(&Bucket) -> bool) -> Option<Bucket> {
        let state = self.state.borrow();
        let serving = state.serving.number();

        state
            .graphs
            .keys()
            .filter(|(bucket, generation)| *generation <= serving && holds(bucket))
            .map(|(bucket, _)| *bucket)
            .min()
    }

    /// The captured buckets the serving generation serves from, in ascending order.
    pub(crate) fn buckets(&self) -> Vec<Bucket> {
        let state = self.state.borrow();
        let serving = state.serving.number();

        let buckets: BTreeSet<Bucket> = state
            .graphs
            .keys()
            .filter(|(_, generation)| *generation <= serving)
            .map(|(bucket, _)| *bucket)
            .collect();
        buckets.into_iter().collect()
    }

    /// Prepares the inputs of one step of `bucket`: takes the next slot of the input buffer,
    /// lets `fill` write the bytes of the runner's input tensors into it, and, when the graph
    /// the step replays has input copies, runs the bucket's forward under the meta context, with
    /// the kernels that graph was captured with, so the host values it computes land in the
    /// slot where that graph's copies read them.
    ///
    /// The graph is the bucket's newest at or below what the bucket switched to and the kernel
    /// stamp of the batch the step belongs to ([`Graphs::stamp`]). The step is
    /// issued by [`Graphs::replay`] for a captured bucket, or by [`Graphs::eager`].
    ///
    /// # Errors
    ///
    /// What `fill` or the forward refuses.
    pub fn prepare_inputs(
        &self,
        runner: &dyn Forward,
        bucket: &Bucket,
        fill: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<()> {
        let slot = self.fill_slot(fill)?;

        // An eager forward writes its host values itself, when it runs.
        let chosen = {
            let state = self.state.borrow();
            state
                .chosen(bucket, state.stamp)
                .map(|(generation, graph)| {
                    (
                        generation,
                        !graph.copies.is_empty(),
                        Arc::clone(&graph.kernels),
                    )
                })
        };
        if let Some((_, true, kernels)) = &chosen {
            self.preparing.set_kernels(Arc::clone(kernels));
            self.lend_values(&self.preparing, slot)?;
            self.meta_forward(&self.preparing, runner, bucket)?;
        }

        self.state.borrow_mut().prepared = Some(Prepared::Inputs(
            *bucket,
            slot,
            chosen.map(|(generation, _, _)| generation),
        ));
        Ok(())
    }

    /// Prepares, in place of inputs, a paired forward of `bucket` that is not run, for
    /// `reason`: its runner cannot compute the shadow batch it was given without writing what a
    /// serving request reads (its rows need more scratch pages than its set holds). The paired
    /// forward [`Graphs::shadow`] then issues for the bucket runs nothing, and
    /// [`Graphs::uncompared`] names its operator with the reason.
    pub(crate) fn unpaired(&self, bucket: &Bucket, reason: String) {
        self.state.borrow_mut().prepared = Some(Prepared::Unpaired(*bucket, reason));
    }

    /// Issues the step [`Graphs::prepare_inputs`] prepared for `bucket`: the batched copy of
    /// the slot's bytes to the input tensors and to the destinations of the graph's input
    /// copies, then the graph's launch. Returns the graph's output tensors, which hold the
    /// step's results once the stream has passed the launch ([`Graphs::wait`]) and until the
    /// next forward on these graphs.
    ///
    /// # Errors
    ///
    /// When no inputs are prepared for `bucket`, the bucket has no graph, these graphs hold no
    /// device, or the driver refuses the copies or the launch.
    pub fn replay(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let cuda = self.cuda()?;
        let (slot, generation) = self.take_prepared(bucket)?;
        let generation = generation.ok_or_else(|| {
            Error::Invariant(format!("bucket {bucket} has no captured graph to replay"))
        })?;

        let mut state = self.state.borrow_mut();
        let State {
            graphs,
            buffer,
            step,
            launches,
            replays,
            uses,
            pending,
            ..
        } = &mut *state;
        let graph = graphs.get_mut(&(*bucket, generation)).ok_or_else(|| {
            Error::Invariant(format!(
                "bucket {bucket} has no graph of generation {generation} to replay"
            ))
        })?;
        let (Some(executable), Some(buffer)) = (graph.executable.as_mut(), buffer.as_mut()) else {
            return Err(Error::State("the graph holds no executable to replay"));
        };
        // An executable the process declared stale since its instantiation (a remapping of the
        // storage it addresses, a change of how CUPTI traces graphs) is instantiated again before
        // this launch, and no other bucket's with it.
        executable.refresh()?;

        buffer.issue(
            slot,
            &cuda.stream,
            self.inputs_at.get(),
            self.inputs,
            &graph.copies,
        )?;
        buffer.release(slot, &cuda.stream)?;

        let step = step.take();
        bracket(&cuda.stream, step, || executable.launch(&cuda.stream))??;
        cuda.done.record(&cuda.stream)?;
        *launches += executable.kernel_nodes();
        *replays += 1;
        *uses.entry(*bucket).or_default() += 1;
        *pending.entry((*bucket, Some(generation))).or_default() += 1;

        Ok(graph.outputs.clone())
    }

    /// Runs the step [`Graphs::prepare_inputs`] prepared for `bucket` eagerly, for a forward
    /// the runner captures no graph of: the copy of the slot's bytes to the input tensors, then
    /// the forward under the launching context. Returns the forward's outputs, valid until the
    /// next forward on these graphs.
    ///
    /// # Errors
    ///
    /// When no inputs are prepared for `bucket`, these graphs hold no device, or the forward or
    /// the driver fails.
    pub fn eager(&self, runner: &dyn Forward, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let cuda = self.cuda()?;
        let (slot, _) = self.take_prepared(bucket)?;

        self.with_buffer(|buffer| {
            buffer.issue(slot, &cuda.stream, self.inputs_at.get(), self.inputs, &[])
        })?;
        self.lend_values(&cuda.launching, slot)?;

        let step = self.state.borrow_mut().step.take();
        cuda.launching.reset();
        let outputs = bracket(&cuda.stream, step, || {
            let _guard = cuda.launching.enter();
            runner.forward(bucket)
        })?;

        // The forward's own copies read the slot, so the slot is free only behind them.
        self.with_buffer(|buffer| buffer.release(slot, &cuda.stream))?;
        cuda.done.record(&cuda.stream)?;

        // A forward that ran launched what the bucket's profile run did.
        if outputs.is_ok() {
            let mut state = self.state.borrow_mut();
            let launched = state.profiles.get(bucket).map_or(0, |table| {
                table
                    .iter()
                    .filter(|(operator, _, _)| *operator != BREAK)
                    .count() as u64
            });
            state.launches += launched;
            state.eager += 1;
            *state.pending.entry((*bucket, None)).or_default() += 1;
        }
        outputs
    }

    /// The kernels the serving forwards of these graphs launched since they were made: every
    /// kernel node of each replayed graph and every launch of each eager forward. Paired
    /// forwards, rehearsals and captures are not serving forwards; graphs that hold no device
    /// launch none.
    pub(crate) fn launches(&self) -> u64 {
        self.state.borrow().launches
    }

    /// The operator launches of the serving forwards since the last take, by operator and
    /// descriptor key: each entry of a replayed graph's launch table, and of the profile run's
    /// table of a forward run eagerly.
    pub fn take_kernels(&self) -> BTreeMap<KernelLaunch, u64> {
        let mut state = self.state.borrow_mut();
        state.settle();
        std::mem::take(&mut state.kernels)
    }

    /// The serving forwards of these graphs since they were made, by how they ran: the graphs
    /// [`Graphs::replay`] launched, then the forwards [`Graphs::eager`] ran to the end. Paired
    /// forwards, rehearsals and captures are not serving forwards; graphs that hold no device
    /// run none.
    pub(crate) fn forwards(&self) -> (u64, u64) {
        let state = self.state.borrow();
        (state.replays, state.eager)
    }

    // ---- Capture: the capture thread's graphs ----

    /// Examines `bucket` for generation `next` beside `serving`, the generation whose graph of
    /// the bucket has the launch table `held` (`None` for a bucket with no graph yet, at
    /// warm-up, which `serving` is then `None` for): a profile run of the forward, which also
    /// states its launches for the capture's check.
    ///
    /// The bucket is captured when it has no graph, its launch table changed, a launch's kernel
    /// is another in `next`, or `next` places the kernel probes elsewhere. It is rehearsed first
    /// when it launches a native kernel whose first use at the bucket's shapes a capture cannot
    /// record: one `serving` does not hold under the same key, and any native kernel of a bucket
    /// with no graph yet or whose launch table changed, since a native entry builds its plan
    /// (a cuBLASLt matmul's, with its workspace) for each shape at that shape's first run, and a
    /// generation's new bucket, such as one a catalog adds, runs shapes no serving bucket ran.
    ///
    /// # Errors
    ///
    /// A profile run that fails, a launch `next` holds no kernel for (on graphs that hold no
    /// device, one the serving generation holds a kernel for), or a forward whose peak the
    /// runner's region cannot hold.
    pub fn examine(
        &self,
        runner: &dyn Forward,
        bucket: &Bucket,
        held: Option<&[(&'static str, String, String)]>,
        serving: Option<&Generation>,
        next: &Generation,
    ) -> Result<Plan> {
        let profiled = self.profile(runner, bucket)?;

        // A generation must hold a kernel for every launch of the bucket that it launches: on a
        // device every launch, and for graphs that hold none, which launch nothing, every launch
        // the serving generation holds a kernel for. One it cannot launch is an incomplete
        // generation, not one to switch to.
        for (operator, key, _) in &profiled {
            let launched = self.device.is_some()
                || serving.is_some_and(|serving| serving.contains(operator, key));
            if *operator != BREAK && launched && !next.contains(operator, key) {
                return Err(Error::Interface(format!(
                    "generation {} holds no kernel of `{operator}` for `{key}`, which bucket \
                     {bucket} launches",
                    next.number()
                )));
            }
        }

        // The new generation's forward allocates in the runner's region as the serving one does.
        if let Some(cuda) = &self.device
            && self.peak() > cuda.region.bytes() as u64
        {
            return Err(Error::Resource(
                "a forward of the generation needs more than the runner's region holds",
            ));
        }

        // A bucket whose kernel probes or fence probes the generation moves is captured again
        // with them.
        let probes_moved = |serving: &Generation| {
            crate::profiling::markers::Markers::installed()
                .is_some_and(|markers| markers.moved(serving.number(), next.number()))
        };
        let capture = match (held, serving) {
            (Some(held), Some(serving)) => {
                !same_launches(held, &profiled)
                    || !changed_launches(&profiled, serving, next).is_empty()
                    || probes_moved(serving)
            }
            _ => true,
        };
        // Shapes no serving graph of the bucket ran: a bucket with no graph, or one whose
        // launches changed.
        let new_shapes = held.is_none_or(|held| !same_launches(held, &profiled));
        let rehearse = capture
            && profiled.iter().any(|(operator, key, _)| {
                let Ok(kernel) = next.kernel(operator, key) else {
                    return false;
                };
                let held = serving.and_then(|serving| serving.kernel(operator, key).ok());
                !kernel.is_cubin()
                    && (new_shapes || held.is_none_or(|held| !Arc::ptr_eq(held, kernel)))
            });
        Ok(Plan { capture, rehearse })
    }

    /// Rehearses `bucket` under `kernels`: its forward runs eagerly once over idle inputs
    /// ([`Forward::idle_inputs`]), on the runner's stream between its replays and in these
    /// graphs' own device memory, under a rehearsal context that issues no collective and no
    /// expert exchange. A kernel whose first use is host work a capture cannot record does that
    /// work here. Returns once the stream ran the forward. Graphs that hold no device run
    /// nothing.
    ///
    /// The memory is a region as large as the runner's, allocated at the first rehearsal and
    /// kept for the next ones until [`Graphs::release_rehearsal`]. It is zeroed before each
    /// rehearsal, so a value a collective would have written reads as zero, which every index a
    /// kernel reads keeps within its bounds.
    ///
    /// # Errors
    ///
    /// When the memory cannot be allocated, the forward fails, or the driver refuses a copy or
    /// the wait.
    pub fn rehearse(
        &self,
        runner: &dyn Forward,
        bucket: &Bucket,
        kernels: &Arc<Generation>,
    ) -> Result<()> {
        let Some(cuda) = &self.device else {
            return Ok(());
        };

        let bytes = cuda.region.bytes();
        let memory = {
            let mut held = cuda.rehearsal.borrow_mut();
            if held.is_none() {
                *held = Some(DeviceAllocation::linear(cuda.stream.device(), bytes)?);
            }
            held.as_ref().map(DeviceAllocation::address).unwrap_or(0)
        };
        let region = DeviceRange::new(platform::DevicePtr::from_address(memory), bytes);
        let (base, allocator) = split_region(region, self.inputs)?;
        self.inputs_at.set(base);

        // SAFETY: the range is the allocation these graphs hold, which nothing else addresses.
        unsafe { platform::memset_d8(memory, 0, bytes, &cuda.stream) }?;

        let slot = self.fill_slot(|columns| runner.idle_inputs(bucket, columns))?;
        self.with_buffer(|buffer| buffer.issue(slot, &cuda.stream, base, self.inputs, &[]))?;

        let rehearsal = cuda.launching.rehearsal(cuda.stream.clone(), allocator)?;
        rehearsal.set_kernels(Arc::clone(kernels));
        self.lend_values(&rehearsal, slot)?;

        let rehearsed = {
            let _guard = rehearsal.enter();
            runner.forward(bucket)
        };

        // The forward's copies read the slot, and its kernels the memory: both are free once the
        // stream ran what the forward issued, whether it ran to its end or not.
        self.with_buffer(|buffer| buffer.release(slot, &cuda.stream))?;
        cuda.done.record(&cuda.stream)?;
        cuda.done.synchronize()?;
        rehearsed.map(drop)
    }

    /// Releases the memory of the rehearsals, once the generation they served is captured.
    pub fn release_rehearsal(&self) {
        if let Some(cuda) = &self.device {
            cuda.rehearsal.borrow_mut().take();
        }
    }

    /// Captures `bucket`'s forward under `kernels` on these graphs' stream and instantiates it,
    /// checking the capture against the profile run ([`Graphs::examine`]) and against the
    /// forward of `prepare_inputs`. Nothing runs: the graph is replayed by the runner's graphs
    /// once installed ([`Graphs::install`]).
    ///
    /// Graphs that hold no device record the launch table and the input copies of a meta
    /// forward and no executable.
    ///
    /// # Errors
    ///
    /// When the forward fails, when the capture's launches are not the profile run's, when the
    /// forward of `prepare_inputs` would not write its host values where the capture's input
    /// copies read them, or when the driver refuses the capture or the instantiation.
    pub fn capture(
        &self,
        runner: &dyn Forward,
        bucket: &Bucket,
        kernels: &Arc<Generation>,
    ) -> Result<Graph> {
        // A bucket captured without a profile run is profiled here: its keys and peaks are
        // then known, and the capture is checked against them like any other.
        if !self.state.borrow().profiles.contains_key(bucket) {
            self.profile(runner, bucket)?;
        }

        let slot = self.fill_slot(|columns| runner.idle_inputs(bucket, columns))?;
        self.lend_values(&self.preparing, slot)?;
        let (profiled, prepared, _) = self.meta_forward(&self.preparing, runner, bucket)?;
        // The module tree the forward enters, which places the generation's kernel probes.
        let ranges = self.preparing.take_ranges();

        let Some(cuda) = &self.device else {
            let (table, copies, _) = self.meta_forward(&self.preparing, runner, bucket)?;
            self.check_profile(bucket, &table)?;

            return Ok(Graph {
                table,
                copies,
                outputs: Vec::new(),
                kernels: Arc::clone(kernels),
                executable: None,
                costs: Costs::default(),
            });
        };

        cuda.launching.set_kernels(Arc::clone(kernels));
        self.lend_values(&cuda.launching, slot)?;
        cuda.launching.reset();
        cuda.launching.record_launches();
        let breaks: Arc<dyn Fn(HostWork) -> ops::Result<()> + Send + Sync> = Arc::new(take_break);
        cuda.launching.set_breaks(Some(breaks));

        let began = Instant::now();
        let captured = {
            let _guard = cuda.launching.enter();
            Recording::capture(&cuda.stream, || {
                // A captured forward carries the range markers of the modules it enters and the
                // probes of the kernels its generation places, which every replay then runs; the
                // process's markers are a profiler only while the stream captures.
                cuda.launching
                    .set_profiler(crate::profiling::markers::profiler(
                        &cuda.stream,
                        &cuda.replays_on,
                        kernels.number(),
                        &ranges,
                        &profiled,
                    ));
                runner.forward(bucket)
            })
        };
        let capture = began.elapsed();
        cuda.launching.set_profiler(None);
        cuda.launching.set_breaks(None);

        let (table, _) = shadowed(cuda.launching.take_launches());
        let copies = cuda.launching.take_input_copies();
        let (recording, outputs) = captured?;

        self.check_profile(bucket, &table)?;
        // The forward of `prepare_inputs` must write where the graph's input copies read: both
        // lists follow from the forward's shapes alone, so they are one list or the forward
        // cannot be replayed.
        if prepared != copies {
            return Err(Error::Invariant(format!(
                "the forward of bucket {bucket} makes {} host-to-device copies when it is \
                 captured and {} under the meta context, or makes them at other offsets or \
                 addresses: its host values do not follow from the bucket alone",
                copies.len(),
                prepared.len()
            )));
        }

        // The instantiation is the device memory a graph takes beside its region: what two
        // generations' graphs of a bucket hold together until the commit.
        let device = cuda.stream.device();
        let free = device.memory_info()?.free;
        let began = Instant::now();
        let executable = recording.instantiate(kernels_of(kernels, &table)?)?;
        let instantiate = began.elapsed();
        let bytes = free.saturating_sub(device.memory_info()?.free);

        let costs = Costs {
            capture,
            instantiate,
            bytes,
            segments: executable.segments(),
            kernel_nodes: executable.kernel_nodes(),
        };
        Ok(Graph {
            table,
            copies,
            outputs,
            kernels: Arc::clone(kernels),
            executable: Some(executable),
            costs,
        })
    }

    // ---- Generations ----

    /// The number of the generation that serves whole.
    pub fn generation(&self) -> u64 {
        self.state.borrow().serving.number()
    }

    /// The generation that serves whole: the kernels an eager forward launches, which the
    /// device's context loaded. The engine builds the next generation of the device from it,
    /// and the memories of the device's rank launch their own work from it.
    pub fn kernels(&self) -> Arc<Generation> {
        Arc::clone(&self.state.borrow().serving)
    }

    /// Installs a captured graph of `bucket` for the generation of `number`, the serving one at
    /// warm-up or the one being prepared, or for the one being prepared the finding that the
    /// bucket keeps its graph (`graph` is `None`). The bucket is then ready to switch.
    ///
    /// # Errors
    ///
    /// When `number` is neither, or the bucket was installed already.
    pub fn install(&self, bucket: &Bucket, number: u64, graph: Option<Graph>) -> Result<()> {
        let mut state = self.state.borrow_mut();
        let serving = state.serving.number();

        match &mut state.update {
            Update::Preparing { next, pending, .. } if next.number() == number => {
                if !pending.remove(bucket) {
                    return Err(Error::Invariant(format!(
                        "bucket {bucket} of generation {number} was installed already, or is \
                         none the generation examines"
                    )));
                }
            }
            Update::Steady if serving == number => {
                if graph.is_none() {
                    return Err(Error::Invariant(format!(
                        "bucket {bucket} is warmed up with no graph"
                    )));
                }
            }
            _ => {
                return Err(Error::Invariant(format!(
                    "generation {number} is neither being prepared nor warmed up: generation \
                     {serving} serves"
                )));
            }
        }

        if let Some(graph) = graph {
            // A replay's host values take what the graph's copies take: the input buffer is
            // sized by what these graphs know of every bucket's.
            let bound = input_bound(&graph.copies);
            let held = state.inputs.entry(*bucket).or_default();
            *held = (*held).max(bound);
            state
                .profiles
                .entry(*bucket)
                .or_insert_with(|| graph.table.clone());

            state.graphs.insert((*bucket, number), graph);
        }
        Ok(())
    }

    /// Begins preparing `next` beside the serving generation and returns the captured buckets
    /// it examines, the most replayed first and otherwise in `runner`'s capture order. The
    /// capture thread examines and, where it needs, captures each, and the worker installs the
    /// outcomes ([`Graphs::install`]). Beginning runs no forward.
    ///
    /// # Errors
    ///
    /// When an update is already in flight, or `next` is the serving generation's number.
    pub fn prepare(&self, runner: &dyn Forward, next: Arc<Generation>) -> Result<Vec<Bucket>> {
        let mut state = self.state.borrow_mut();
        if !matches!(state.update, Update::Steady) {
            return Err(Error::State(
                "a generation update is in flight: commit or roll it back first",
            ));
        }
        if state.serving.number() == next.number() {
            return Err(Error::Invariant(format!(
                "generation {} serves and cannot be prepared beside itself",
                next.number()
            )));
        }

        let mut order: Vec<Bucket> = runner
            .buckets()
            .into_iter()
            .filter(|bucket| state.served(bucket).is_some())
            .collect();
        // A stable sort keeps the capture order among buckets of as many replays.
        order.sort_by_key(|bucket| std::cmp::Reverse(state.uses.get(bucket).copied().unwrap_or(0)));

        state.uncompared.clear();
        state.update = Update::Preparing {
            next,
            pending: order.iter().copied().collect(),
            switched: BTreeSet::new(),
        };
        Ok(order)
    }

    /// Begins preparing `next` beside the serving generation for paired forwards alone, when the
    /// generation captures the runner's graphs for another shape of it (`Runner::reshape`): no
    /// bucket of these graphs is examined, since the shape's graphs are captured for graphs of
    /// their own ([`Graphs::over`]), and a paired forward of a bucket of these graphs compares
    /// each kernel the generation replaces with the serving one ([`Graphs::shadow`]). The
    /// preparation ends with [`Graphs::rollback`] of `next`, at the switch to the shape as when
    /// the generation is given back. Beginning runs no forward.
    ///
    /// # Errors
    ///
    /// As [`Graphs::prepare`].
    pub fn pair(&self, next: Arc<Generation>) -> Result<()> {
        let mut state = self.state.borrow_mut();
        if !matches!(state.update, Update::Steady) {
            return Err(Error::State(
                "a generation update is in flight: commit or roll it back first",
            ));
        }
        if state.serving.number() == next.number() {
            return Err(Error::Invariant(format!(
                "generation {} serves and cannot be prepared beside itself",
                next.number()
            )));
        }

        state.uncompared.clear();
        state.update = Update::Preparing {
            next,
            pending: BTreeSet::new(),
            switched: BTreeSet::new(),
        };
        Ok(())
    }

    /// The buckets of the generation being prepared whose outcome is not installed yet, in
    /// ascending order; empty when every one is, or no generation is being prepared.
    pub(crate) fn pending(&self) -> Vec<Bucket> {
        match &self.state.borrow().update {
            Update::Preparing { pending, .. } => pending.iter().copied().collect(),
            _ => Vec::new(),
        }
    }

    /// Switches `buckets` to the prepared generation `number` ahead of the others: from the
    /// next step on, a batch stamped with it ([`Graphs::stamp`]) replays their graphs of that
    /// generation, or the graph each kept. A bucket these graphs hold no graph of is none of
    /// theirs and is passed over.
    ///
    /// # Errors
    ///
    /// When `number` is not the generation being prepared, or a bucket's outcome is not
    /// installed yet.
    pub fn switch_buckets(&self, number: u64, buckets: &[Bucket]) -> Result<()> {
        let mut state = self.state.borrow_mut();
        let held: BTreeSet<Bucket> = state.graphs.keys().map(|(bucket, _)| *bucket).collect();

        let Update::Preparing {
            next,
            pending,
            switched,
        } = &mut state.update
        else {
            return Err(Error::Invariant(format!(
                "generation {number} is not being prepared"
            )));
        };
        if next.number() != number {
            return Err(Error::Invariant(format!(
                "generation {number} is not being prepared: generation {} is",
                next.number()
            )));
        }

        for bucket in buckets.iter().filter(|bucket| held.contains(bucket)) {
            if pending.contains(bucket) {
                return Err(Error::Invariant(format!(
                    "bucket {bucket} switches to generation {number} before its capture was \
                     installed"
                )));
            }
            switched.insert(*bucket);
        }
        Ok(())
    }

    /// Switches to the prepared generation `number` whole: from the next step on every bucket
    /// replays its graph of it, or the graph it kept, eager forwards launch its kernels, and it
    /// serves on trial, the generation it replaces kept.
    ///
    /// # Errors
    ///
    /// When `number` is not the prepared generation, or a bucket's outcome is not installed.
    pub fn switch(&self, number: u64) -> Result<()> {
        let mut state = self.state.borrow_mut();

        let next = match &state.update {
            Update::Preparing { next, pending, .. } if next.number() == number => {
                if let Some(bucket) = pending.first() {
                    return Err(Error::Invariant(format!(
                        "generation {number} is not prepared: bucket {bucket} is still to be \
                         examined"
                    )));
                }
                Arc::clone(next)
            }
            _ => {
                return Err(Error::Invariant(format!(
                    "generation {number} was not prepared: generation {} serves",
                    state.serving.number()
                )));
            }
        };

        if let Some(cuda) = &self.device {
            cuda.launching.set_kernels(Arc::clone(&next));
        }
        let kept = std::mem::replace(&mut state.serving, next);
        state.update = Update::Trial { kept };
        Ok(())
    }

    /// Gives these graphs the region the shadow outputs of paired calls are written in:
    /// device memory of the scratch pool, which no serving forward reads.
    ///
    /// # Errors
    ///
    /// For graphs that hold no device.
    pub fn set_shadow(&self, twins: ops::Twins) -> Result<()> {
        let device = self.cuda()?.stream.device();

        self.state.borrow_mut().shadow = Some(Shadow::new(device, twins));
        Ok(())
    }

    /// Runs the step [`Graphs::prepare_inputs`] prepared for `bucket` as a paired forward of
    /// `operator`: an eager forward under the generation being prepared in which every call of
    /// the operator whose kernel changed is launched with both generations' kernels over the
    /// same inputs, the serving kernel writing the outputs the forward continues on. The pairs
    /// are compared within each call's tolerance, and the elements compared are returned;
    /// `None` when the batch compared no pair, which is then no evidence for the operator.
    ///
    /// The runner prepares the inputs of a paired forward so that it writes no state a serving
    /// request reads. A paired forward it prepared none for ([`Graphs::unpaired`]) runs
    /// nothing and compares nothing: the operator is named in [`Graphs::uncompared`] with the
    /// runner's reason, and `None` is returned. So is one a call of which needs more than the
    /// shadow region ([`Graphs::shadow_peak`]). Each paired call is compared as both kernels
    /// ran it, so the region holds one call's twins at a time.
    ///
    /// # Errors
    ///
    /// When no generation is being prepared, no step was prepared for `bucket`, no shadow region
    /// was given, these graphs hold no device, the forward fails, or a pair differs beyond its
    /// tolerance.
    pub fn shadow(
        &self,
        runner: &dyn Forward,
        bucket: &Bucket,
        operator: &'static str,
    ) -> Result<Option<u64>> {
        if !matches!(self.state.borrow().update, Update::Preparing { .. }) {
            return Err(Error::State("no generation is being prepared"));
        }
        let slot = match self.take_step(bucket)? {
            Prepared::Inputs(_, slot, _) => slot,
            Prepared::Unpaired(_, reason) => {
                self.state
                    .borrow_mut()
                    .uncompared
                    .push(format!("`{operator}` in bucket {bucket}: {reason}"));
                return Ok(None);
            }
        };

        let cuda = self.cuda()?;
        let (serving, next, pairing) = {
            let state = self.state.borrow();
            let Update::Preparing { next, .. } = &state.update else {
                return Err(Error::State("no generation is being prepared"));
            };
            let shadow = state
                .shadow
                .as_ref()
                .ok_or(Error::State("the graphs were given no shadow region"))?;

            let serving = Arc::clone(&state.serving);
            let pairing = shadow.pairing(operator, Arc::clone(&serving));
            (serving, Arc::clone(next), pairing)
        };

        self.with_buffer(|buffer| {
            buffer.issue(slot, &cuda.stream, self.inputs_at.get(), self.inputs, &[])
        })?;
        self.lend_values(&cuda.launching, slot)?;

        // A paired forward runs a batch the runner served already, so it counts no expert's
        // tokens again.
        let load = cuda.launching.expert_load();
        cuda.launching.set_expert_load(None);
        cuda.launching.set_kernels(next);
        cuda.launching.set_route(Some(
            Arc::clone(&pairing) as Arc<dyn ops::Route + Send + Sync>
        ));
        cuda.launching.reset();
        let outputs = {
            let _guard = cuda.launching.enter();
            runner.forward(bucket)
        };
        cuda.launching.set_route(None);
        cuda.launching.set_kernels(serving);
        cuda.launching.set_expert_load(load);

        self.with_buffer(|buffer| buffer.release(slot, &cuda.stream))?;
        cuda.done.record(&cuda.stream)?;
        outputs?;

        Shadow::finish(&pairing, &mut self.state.borrow_mut().uncompared)
    }

    /// The pairs the paired forwards of the generation being prepared could not compare, each
    /// with the reason: what the update has no evidence for.
    pub fn uncompared(&self) -> Vec<String> {
        self.state.borrow().uncompared.clone()
    }

    /// Commits the trial of generation `number`: the kept generation is released, with every
    /// graph a newer one replaced and the kernels only they held, once the device has passed
    /// the last batch that replayed one.
    ///
    /// # Errors
    ///
    /// When `number` is not serving on trial, or the device cannot be waited for.
    pub fn commit(&self, number: u64) -> Result<()> {
        {
            let state = self.state.borrow();
            if !matches!(state.update, Update::Trial { .. }) || state.serving.number() != number {
                return Err(Error::Invariant(format!(
                    "generation {number} is not on trial: generation {} serves",
                    state.serving.number()
                )));
            }
        }
        self.settle()?;

        let mut state = self.state.borrow_mut();
        let replaced: Vec<(Bucket, u64)> = state
            .graphs
            .keys()
            .filter(|(bucket, generation)| {
                state
                    .graphs
                    .keys()
                    .any(|(other, newer)| other == bucket && newer > generation)
            })
            .copied()
            .collect();
        state.settle();
        for key in replaced {
            state.graphs.remove(&key);
        }

        // The last reference to the kept generation's kernels that no graph holds drops here.
        state.update = Update::Steady;
        Ok(())
    }

    /// Rolls generation `number` back. Being prepared, its buckets switched ahead serve their
    /// kept graphs again from the next batch on and its graphs are released; on trial, the kept
    /// generation serves again from the next batch on and the rolled-back generation's graphs
    /// are released. A graph is released once the device has passed the last batch that
    /// replayed it.
    ///
    /// # Errors
    ///
    /// When `number` is neither on trial nor being prepared, or the device cannot be waited
    /// for.
    pub fn rollback(&self, number: u64) -> Result<()> {
        {
            let state = self.state.borrow();
            let named = match &state.update {
                Update::Trial { .. } => state.serving.number() == number,
                Update::Preparing { next, .. } => next.number() == number,
                Update::Steady => false,
            };
            if !named {
                return Err(Error::Invariant(format!(
                    "generation {number} is neither on trial nor being prepared: generation {} \
                     serves",
                    state.serving.number()
                )));
            }
        }
        self.settle()?;

        let mut state = self.state.borrow_mut();
        if let Update::Trial { kept } = std::mem::replace(&mut state.update, Update::Steady) {
            if let Some(cuda) = &self.device {
                cuda.launching.set_kernels(Arc::clone(&kept));
            }
            state.serving = kept;
        }
        state.settle();
        state
            .graphs
            .retain(|(_, generation), _| *generation != number);
        Ok(())
    }

    /// The device bytes a generation that replaces `operators` takes beside the serving graphs
    /// until its commit: the instantiations of the serving graphs that launch one of them, each
    /// captured again for it, and, when any does, a rehearsal's memory, a region as large as
    /// the runner's. Each graph's figure is what its capture measured ([`Costs::bytes`]).
    pub(crate) fn update_bytes(&self, operators: &BTreeSet<String>) -> u64 {
        let state = self.state.borrow();
        let serving = state.serving.number();

        let mut graphs = 0;
        let mut any = false;
        for ((bucket, generation), graph) in &state.graphs {
            let served = served(&state.graphs, serving, bucket).map(|(newest, _)| newest);
            let launches = graph
                .table
                .iter()
                .any(|(operator, _, _)| operators.contains(*operator));
            if served == Some(*generation) && launches {
                graphs += graph.costs.bytes;
                any = true;
            }
        }

        let rehearsal = match &self.device {
            Some(cuda) if any => cuda.region.bytes() as u64,
            _ => 0,
        };
        graphs + rehearsal
    }

    /// The device bytes one graph of these graphs' instantiations takes, at the mean of the
    /// serving graphs' (`Costs::bytes`): what a generation that captures the buckets of another
    /// shape of the runner counts for each of them beside the serving graphs until it commits.
    /// Zero for graphs that hold no device or no graph.
    pub(crate) fn graph_bytes(&self) -> u64 {
        let state = self.state.borrow();
        let serving = state.serving.number();

        let (mut bytes, mut graphs) = (0_u64, 0_u64);
        for ((bucket, generation), graph) in &state.graphs {
            let served = served(&state.graphs, serving, bucket).map(|(newest, _)| newest);
            if served == Some(*generation) {
                bytes += graph.costs.bytes;
                graphs += 1;
            }
        }
        bytes.checked_div(graphs).unwrap_or(0)
    }

    // ---- What the public operations share ----

    /// The device side, or the refusal of graphs that hold none.
    fn cuda(&self) -> Result<&Cuda> {
        self.device
            .as_ref()
            .ok_or(Error::State("these graphs hold no device"))
    }

    /// Waits for the device to pass everything these graphs issued, before graphs are released.
    fn settle(&self) -> Result<()> {
        match &self.device {
            Some(cuda) => Ok(cuda.done.synchronize()?),
            None => Ok(()),
        }
    }

    /// Runs `bucket`'s forward under the meta context `context` from an empty allocator, and
    /// returns its launch table, its input copies and the most one paired call of it carves
    /// ([`shadowed`]).
    fn meta_forward(
        &self,
        context: &Context,
        runner: &dyn Forward,
        bucket: &Bucket,
    ) -> Result<(Table, Copies, u64)> {
        context.reset();
        context.record_launches();

        let forward = {
            let _guard = context.enter();
            runner.forward(bucket)
        };

        let (table, shadow) = shadowed(context.take_launches());
        let copies = context.take_input_copies();
        forward?;
        Ok((table, copies, shadow))
    }

    /// Runs `operation` on the input buffer, making a ring over ordinary host memory when the
    /// graphs hold none, or theirs has become too small for the profiled forwards: graphs that
    /// hold no device, and graphs beside a runner's, whose captures issue no copy.
    fn with_buffer<T>(&self, operation: impl FnOnce(&mut InputBuffer) -> Result<T>) -> Result<T> {
        let values = self.input_peak();
        let mut state = self.state.borrow_mut();

        let host = state
            .buffer
            .as_ref()
            .is_none_or(|buffer| buffer.is_host() && buffer.values_bytes() < values);
        if host {
            state.buffer = Some(InputBuffer::host(self.inputs, values, self.slots)?);
        }

        let buffer = state
            .buffer
            .as_mut()
            .ok_or(Error::State("the graphs hold no input buffer"))?;
        operation(buffer)
    }

    /// Takes the next slot of the input buffer and lets `fill` write the columns of the
    /// runner's input tensors into it.
    fn fill_slot(&self, fill: impl FnOnce(&mut [u8]) -> Result<()>) -> Result<usize> {
        let inputs = self.inputs as usize;

        self.with_buffer(|buffer| {
            let slot = buffer.acquire()?;
            fill(&mut buffer.columns_mut(slot)[..inputs])?;
            Ok(slot)
        })
    }

    /// Lends the values area of `slot` to `context` as its input buffer.
    fn lend_values(&self, context: &Context, slot: usize) -> Result<()> {
        let (base, bytes) = self.with_buffer(|buffer| Ok(buffer.values_ptr(slot)))?;

        // SAFETY: the area is `bytes` bytes of the ring's memory, which is replaced only by
        // `with_buffer`, before a slot is taken and lent again; nothing but the forwards of
        // this context writes it while the slot is taken, and the slot is taken again only once
        // the stream has passed the copies that read it.
        unsafe { context.set_input_buffer(base, bytes) };
        Ok(())
    }

    /// The slot holding the inputs prepared for `bucket`, which the caller now issues, and the
    /// generation of the graph they were prepared for.
    fn take_prepared(&self, bucket: &Bucket) -> Result<(usize, Option<u64>)> {
        match self.take_step(bucket)? {
            Prepared::Inputs(_, slot, generation) => Ok((slot, generation)),
            Prepared::Unpaired(..) => Err(Error::Invariant(format!(
                "bucket {bucket}'s step is a paired forward its runner prepared no inputs for"
            ))),
        }
    }

    /// The step prepared for `bucket`, which the caller now issues.
    fn take_step(&self, bucket: &Bucket) -> Result<Prepared> {
        match self.state.borrow_mut().prepared.take() {
            Some(step) if step.bucket() == bucket => Ok(step),
            Some(step) => Err(Error::Invariant(format!(
                "a step of bucket {} is prepared, and bucket {bucket} was issued",
                step.bucket()
            ))),
            None => Err(Error::State("no inputs are prepared for the step")),
        }
    }

    /// Refuses a capture whose launches are not the profile run's of the same bucket.
    fn check_profile(&self, bucket: &Bucket, table: &Table) -> Result<()> {
        let state = self.state.borrow();
        let Some(profiled) = state.profiles.get(bucket) else {
            return Ok(());
        };

        if same_launches(profiled, table) {
            return Ok(());
        }
        Err(Error::Invariant(format!(
            "the capture of bucket {bucket} launched {} kernels and its profile run stated {}, \
             or other keys: the kernels resolved from the profile run are not the capture's",
            table.len(),
            profiled.len()
        )))
    }
}

impl State {
    fn new(serving: Arc<Generation>, buffer: Option<InputBuffer>) -> Self {
        Self {
            serving,
            update: Update::Steady,
            stamp: None,
            profiles: HashMap::new(),
            shadows: HashMap::new(),
            inputs: HashMap::new(),
            graphs: HashMap::new(),
            uses: HashMap::new(),
            buffer,
            prepared: None,
            step: None,
            shadow: None,
            uncompared: Vec::new(),
            launches: 0,
            replays: 0,
            eager: 0,
            pending: HashMap::new(),
            kernels: BTreeMap::new(),
        }
    }

    /// Counts every pending forward by operator and descriptor key, through the launch table of
    /// the graph it replayed or the profile run of the bucket it ran eagerly. A forward whose
    /// table is gone counts nothing; every graph is settled before it is released.
    fn settle(&mut self) {
        for ((bucket, generation), forwards) in self.pending.drain() {
            let table = match generation {
                Some(generation) => self
                    .graphs
                    .get(&(bucket, generation))
                    .map(|graph| &graph.table),
                None => self.profiles.get(&bucket),
            };
            for (operator, key, _) in table.into_iter().flatten() {
                // A graph break is an entry of the table, not a launch.
                if *operator == BREAK {
                    continue;
                }
                let launch = KernelLaunch {
                    operator: (*operator).to_owned(),
                    key: key.clone(),
                };
                *self.kernels.entry(launch).or_default() += forwards;
            }
        }
    }

    /// The graph the serving generation serves `bucket` from, with the generation it was
    /// captured for.
    fn served(&self, bucket: &Bucket) -> Option<(u64, &Graph)> {
        served(&self.graphs, self.serving.number(), bucket)
    }

    /// The graph a step of `bucket` stamped with kernel generation `stamp` replays, with its
    /// generation: the newest at or below the generation the bucket serves from (the one being
    /// prepared when the bucket switched to it ahead of the others, otherwise the one that
    /// serves) and at or below the stamp.
    fn chosen(&self, bucket: &Bucket, stamp: Option<u64>) -> Option<(u64, &Graph)> {
        let mut top = self.serving.number();
        if let Update::Preparing { next, switched, .. } = &self.update
            && switched.contains(bucket)
        {
            top = next.number();
        }
        if let Some(stamp) = stamp {
            top = top.min(stamp);
        }
        served(&self.graphs, top, bucket)
    }
}

/// Splits `region`, a runner's region, into the address of its input tensors, at its first
/// aligned address, and the region of the allocator, which follows them.
///
/// # Errors
///
/// When the region cannot hold `inputs` bytes of input tensors.
fn split_region(region: DeviceRange, inputs: u64) -> Result<(u64, DeviceRange)> {
    let start = region.ptr().address();
    let base = start.next_multiple_of(ALIGNMENT);
    let taken = (base - start) + inputs.next_multiple_of(ALIGNMENT);
    let rest = (region.bytes() as u64).checked_sub(taken).ok_or_else(|| {
        Error::Invalid(format!(
            "a region of {} bytes cannot hold {inputs} bytes of input tensors",
            region.bytes()
        ))
    })?;

    let allocator = DeviceRange::new(
        platform::DevicePtr::from_address(start + taken),
        rest as usize,
    );
    Ok((base, allocator))
}

/// Runs `run`, a step's launches on `stream`, between the step's range markers when the worker
/// stated a step and the process holds range markers (`Markers::bracket`), and alone otherwise.
fn bracket<T>(
    stream: &Stream,
    step: Option<(u64, &'static str)>,
    run: impl FnOnce() -> T,
) -> Result<T> {
    match (step, crate::profiling::markers::Markers::installed()) {
        (Some((step, stage)), Some(markers)) => markers.bracket(stream, step, stage, run),
        _ => Ok(run()),
    }
}

/// The graph `bucket` serves from at generation `top`: the one of the newest generation at or
/// below it that holds a graph of the bucket.
///
/// A bucket none of whose kernels changed has no graph of the newer generation and keeps
/// serving from the one it has, and a generation prepared and not yet switched to serves
/// nothing.
fn served<'a>(
    graphs: &'a HashMap<(Bucket, u64), Graph>,
    top: u64,
    bucket: &Bucket,
) -> Option<(u64, &'a Graph)> {
    graphs
        .iter()
        .filter(|((held, generation), _)| held == bucket && *generation <= top)
        .max_by_key(|((_, generation), _)| *generation)
        .map(|((_, generation), graph)| (*generation, graph))
}

/// The bytes a slot's values area needs to hold `copies`, a forward's input copies, at any
/// offsets: each copy's bytes and the most its start's alignment can add to them.
///
/// A copy starts at the first aligned offset after the one before it ends (`ops::input`), so the
/// copies of another forward of no more copies and no more bytes in all end within this, however
/// their sizes fall against the alignment.
fn input_bound(copies: &Copies) -> u64 {
    copies
        .iter()
        .map(|(_, _, bytes)| bytes + (ALIGNMENT - 1))
        .sum()
}

/// The launch table of the forward `recorded` states, and the most one of its launches carves
/// from a shadow region when it is paired: its outputs and the operands it writes in place. A
/// graph break carves nothing.
fn shadowed(recorded: Recorded) -> (Table, u64) {
    let mut most = 0;
    let mut table = Vec::with_capacity(recorded.len());

    for (operator, key, path, bytes) in recorded {
        most = most.max(bytes);
        table.push((operator, key, path));
    }
    (table, most)
}

/// Whether two launch tables hold the same launches: the same operators with the same keys in
/// the same order, breaks included. Module paths are not compared: they name where a launch
/// was issued, not what it launches.
fn same_launches(left: &[(&'static str, String, String)], right: &Table) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|((operator, key, _), (other, other_key, _))| {
                operator == other && key == other_key
            })
}

/// Whether `next` launches `operator` under `key` with another kernel than `serving`.
fn kernel_changed(
    operator: &'static str,
    key: &str,
    serving: &Generation,
    next: &Generation,
) -> bool {
    match (serving.kernel(operator, key), next.kernel(operator, key)) {
        (Ok(held), Ok(new)) => !Arc::ptr_eq(held, new),
        (Err(_), Err(_)) => false,
        _ => true,
    }
}

/// The launches of `table` whose kernel is another in `next` than in `serving`, as indices
/// into the table's launches in launch order, breaks not counted.
fn changed_launches(table: &Table, serving: &Generation, next: &Generation) -> Vec<usize> {
    table
        .iter()
        .filter(|(operator, _, _)| *operator != BREAK)
        .enumerate()
        .filter(|(_, (operator, key, _))| kernel_changed(operator, key, serving, next))
        .map(|(index, _)| index)
        .collect()
}

#[cfg(test)]
pub(crate) mod tests;
