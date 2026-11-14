//! `Graphs`: the single implementation of the profile run, capture, replay and the switch of
//! generations, for every runner.
//!
//! A runner computes one forward per bucket ([`Runner::forward`]). `Graphs` runs it three ways,
//! and nothing else in the crate does:
//!
//! - **Profile run.** At load the forward runs under a meta context: it launches nothing and
//!   yields the bucket's launch table (operator, key and module path of every launch, in launch
//!   order), the peak of the allocator region, the bytes of its host-to-device copies and the
//!   most one paired call of it carves from a shadow region. The keys are what the engine
//!   resolves kernels for, and the peaks and the shadow bytes are what a device's memory is
//!   divided by.
//! - **Capture.** At warm-up the forward runs eagerly once over idle inputs and is then
//!   captured, with its launch table, its input copies and its output tensors.
//! - **Replay.** [`Graphs::prepare_inputs`] takes a slot of the input buffer, lets the runner
//!   write the bytes of its input tensors and, when the graph has input copies, runs the forward
//!   under a meta context over the same region, so the host values the model computes land in
//!   the slot at the offsets the capture recorded. [`Graphs::replay`] then issues the slot's
//!   copies as one batched copy and launches the graph. Nothing is called back from the stream.
//!
//! ## Memory
//!
//! A runner's region is one range of device memory: the block of its input tensors at fixed
//! addresses, then the region of the context's allocator, which is emptied at the start of
//! every forward. A forward of one bucket therefore receives the same addresses when it is
//! captured and whenever it runs again, and a graph's output tensors are the ones its capture
//! allocated. They are read after the replay and before the next forward on the runner.
//!
//! ## Generations
//!
//! A new generation of kernels is prepared beside the serving graphs: a bucket whose launch
//! table is unchanged is rebound (a second executable of the serving graph with the new kernels'
//! parameters on the changed launches), any other is recaptured, and a bucket none of whose
//! kernels changed keeps its graph. A rebind reads the new parameters from a capture under the
//! new kernels, and every capture, as at warm-up, follows one eager run of the forward with the
//! kernels it captures. The buckets are examined one at a time, each with a profile run of its
//! forward, and the worker serves batches between two of them. The switch takes effect between
//! two batches; the replaced generation's graphs stay until the trial is committed, and serve
//! again after a rollback.
//!
//! Before the switch, each operator whose kernel changed runs a paired forward
//! ([`Graphs::shadow`]): the new kernel beside the one it replaces on the same inputs, compared
//! under the operator's tolerance.
//!
//! ## Without a device
//!
//! [`Graphs::meta`] holds no stream. It runs profile runs, and the whole generation state
//! machine with every device step left out: a capture records the launch table and the input
//! copies of a meta forward and no executable, and a replay is refused. A runner built over it
//! makes its input tensors on the meta device.

mod capture;
mod input_buffer;
mod shadow;

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use ops::Generation;
use ops::breaks::{BREAK, HostWork};
use ops::collective::Meet;
use ops::moe::ExchangeTransport;
use ops::{Context, ContextGuard, Group};
use platform::{DeviceRange, Event, EventFlags, Stream};
use tensor::{DType, Device, Shape, Tensor};

use self::capture::{Executable, Recorder, kernels_of, take_break};
use self::input_buffer::InputBuffer;
use self::shadow::Shadow;
use crate::runner::Runner;
use crate::runner::buckets::Bucket;
use crate::worker::{Error, Result};

/// The alignment of the input tensors' block and of the allocator region behind it.
const ALIGNMENT: u64 = ops::allocator::ALIGNMENT;

/// A launch table: `(operator, key, module path)` per launch, in launch order.
type Table = Vec<(&'static str, String, String)>;

/// A launch table as a context records it (`ops::Context::record_launches`): each launch with
/// the bytes a paired launch of it carves from a shadow region (`ops::Call::shadow_bytes`).
type Recorded = Vec<(&'static str, String, String, u64)>;

/// Input copies: `(offset in a slot's values, device address, bytes)` per copy, in the order the
/// forward makes them.
type Copies = Vec<(u64, u64, u64)>;

/// What a new generation needs of one bucket's graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Preparation {
    /// No kernel the bucket launches changed: the bucket serves from the graph it has.
    Keep,
    /// The launch table is unchanged and some kernels changed: a second executable of the
    /// serving graph takes the new kernels' parameters.
    Rebind,
    /// The launch table changed: the bucket is captured again.
    Recapture,
}

/// The captured graphs of one runner, the context its forwards compute under, and the
/// generations of kernels they launch.
pub struct Graphs {
    /// What issues on the device; `None` for graphs that hold none.
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
    state: RefCell<State>,
}

/// The device side of a runner's graphs.
struct Cuda {
    stream: Stream,
    /// The context that launches: captures and eager forwards compute under it.
    launching: Context,
    /// The device address of the runner's input tensors.
    inputs: u64,
    /// Recorded on the stream behind each replay and eager forward: what a reader of the
    /// outputs waits on.
    done: Event,
}

/// What changes while a runner serves.
struct State {
    /// The generation that serves.
    serving: Arc<Generation>,
    /// The generation update in flight.
    update: Update,
    /// The launch table of each profiled bucket.
    profiles: HashMap<Bucket, Table>,
    /// The most one paired call of each profiled bucket's forward carves from a shadow region
    /// ([`shadowed`]).
    shadows: HashMap<Bucket, u64>,
    /// The values area of a slot that holds the copies of each profiled bucket's forward at any
    /// offsets ([`input_bound`]).
    inputs: HashMap<Bucket, u64>,
    /// The graph of each captured bucket, by the generation it was captured or rebound for.
    graphs: HashMap<(Bucket, u64), Graph>,
    /// The input buffer; made at the first use for graphs that hold no device.
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
}

/// The step prepared for a bucket ([`Graphs::prepare_inputs`], [`Graphs::unpaired`]).
enum Prepared {
    /// The bucket's inputs, in this slot of the input buffer.
    Inputs(Bucket, usize),
    /// A paired forward of the bucket that its runner prepared no inputs for, with the reason:
    /// it runs nothing, and the operator it was to compare is reported uncompared.
    Unpaired(Bucket, String),
}

impl Prepared {
    /// The bucket the step is of.
    const fn bucket(&self) -> &Bucket {
        match self {
            Self::Inputs(bucket, _) | Self::Unpaired(bucket, _) => bucket,
        }
    }
}

/// Where a generation update stands.
enum Update {
    /// One generation serves and no other is held.
    Steady,
    /// `next` is prepared beside the serving generation; `pending` are the captured buckets it
    /// has yet to examine and, where it needs, prepare, largest first ([`Graphs::prepare_bucket`]).
    Prepared {
        next: Arc<Generation>,
        pending: Vec<Bucket>,
    },
    /// The prepared generation serves on trial and `kept` is the one it replaced.
    Trial { kept: Arc<Generation> },
}

/// One bucket's captured forward.
struct Graph {
    /// The launches of the forward, with a [`BREAK`] entry at each graph break.
    table: Table,
    /// The input copies a replay issues before the launch.
    copies: Copies,
    /// The forward's outputs, at the addresses the capture allocated.
    outputs: Vec<Tensor>,
    /// What a replay launches; `None` for graphs that hold no device.
    executable: Option<Executable>,
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
    /// The caller keeps `region` mapped at its address for as long as these graphs exist.
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

        // The input tensors take the start of the region, aligned as the allocator's blocks
        // are; the allocator takes what follows them.
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

        let buffer = InputBuffer::pinned(device, inputs, values, slots)?;
        let cuda = Cuda {
            launching: Context::new(ordinal, stream.clone(), Arc::clone(&kernels), allocator),
            stream,
            inputs: base,
            done: Event::new(EventFlags::default()),
        };

        Ok(Self {
            device: Some(cuda),
            profile: Context::meta(None),
            preparing: Context::meta(Some(allocator)),
            inputs,
            slots,
            state: RefCell::new(State::new(kernels, Some(buffer))),
        })
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

        let address = self.device.as_ref().map_or(0, |cuda| cuda.inputs + offset);

        // SAFETY: on a GPU the address lies in the input block at the start of the region,
        // which the caller of `Graphs::new` keeps mapped for as long as these graphs exist and
        // which only the copies these graphs issue write; on the meta device nothing is read.
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
    /// forward, paired forward or capture. Graphs that hold no device have nothing in flight.
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

    /// The operators the generation being prepared launches with another kernel than the
    /// serving one under the same key in `bucket`'s graph, in launch order, each once: what a
    /// paired forward of the bucket compares ([`Graphs::shadow`]). A key only one of the two
    /// generations holds has no pair. Empty when no generation is being prepared.
    pub(crate) fn replaced(&self, bucket: &Bucket) -> Vec<&'static str> {
        let state = self.state.borrow();
        let Update::Prepared { next, .. } = &state.update else {
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
    pub(crate) fn set_axes(&self, axes: Vec<(Option<Arc<Group>>, u64, u64)>) -> Result<()> {
        let splits: Vec<(Option<Arc<Group>>, u64, u64)> = axes
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

    // ---- The profile run ----

    /// Runs the forward of every bucket `runner` states, the captured ones and the eager ones
    /// ([`Runner::eager_buckets`]), under the meta context and keeps each bucket's launch table
    /// and the most one paired call of it carves from a shadow region.
    ///
    /// # Errors
    ///
    /// What a forward refuses.
    pub fn profile_run(&self, runner: &dyn Runner) -> Result<()> {
        for bucket in runner.buckets().into_iter().chain(runner.eager_buckets()) {
            self.profile(runner, &bucket)?;
        }
        Ok(())
    }

    /// Profiles `bucket`: its forward under the meta profile context, whose launch table and
    /// largest paired call ([`shadowed`]) these graphs keep.
    fn profile(&self, runner: &dyn Runner, bucket: &Bucket) -> Result<()> {
        let (table, copies, shadow) = self.meta_forward(&self.profile, runner, bucket)?;

        let mut state = self.state.borrow_mut();
        state.profiles.insert(*bucket, table);
        state.shadows.insert(*bucket, shadow);
        state.inputs.insert(*bucket, input_bound(&copies));
        Ok(())
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

    // ---- Capture and replay ----

    /// Captures `bucket` for the serving generation: the forward runs eagerly once over the
    /// runner's idle inputs, which leaves every operator ready, and is then captured with its
    /// launch table, its input copies and its outputs.
    ///
    /// Warm-up captures a runner's buckets largest first ([`Runner::buckets`]).
    ///
    /// # Errors
    ///
    /// When the forward fails, when the capture's launches are not the profile run's, or when
    /// the forward of `prepare_inputs` would not write its host values where the capture's
    /// input copies read them.
    pub fn capture(&self, runner: &dyn Runner, bucket: &Bucket) -> Result<()> {
        let serving = Arc::clone(&self.state.borrow().serving);
        let graph = self.captured(runner, bucket, &serving)?;

        self.state
            .borrow_mut()
            .graphs
            .insert((*bucket, serving.number()), graph);
        Ok(())
    }

    /// The smallest captured bucket of the serving generation that `holds` a batch.
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
    /// lets `fill` write the bytes of the runner's input tensors into it, and, when the
    /// bucket's graph has input copies, runs the bucket's forward under the meta context so the
    /// host values it computes land in the slot where those copies read them.
    ///
    /// The step is issued by [`Graphs::replay`] for a captured bucket, or by [`Graphs::eager`].
    ///
    /// # Errors
    ///
    /// What `fill` or the forward refuses.
    pub fn prepare_inputs(
        &self,
        runner: &dyn Runner,
        bucket: &Bucket,
        fill: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<()> {
        let slot = self.fill_slot(fill)?;

        // An eager forward writes its host values itself, when it runs.
        let copies = self
            .state
            .borrow()
            .served(bucket)
            .is_some_and(|(_, graph)| !graph.copies.is_empty());
        if copies {
            self.lend_values(&self.preparing, slot)?;
            self.meta_forward(&self.preparing, runner, bucket)?;
        }

        self.state.borrow_mut().prepared = Some(Prepared::Inputs(*bucket, slot));
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
        let slot = self.take_prepared(bucket)?;

        let mut state = self.state.borrow_mut();
        let serving = state.serving.number();
        let State {
            graphs,
            buffer,
            step,
            launches,
            replays,
            ..
        } = &mut *state;
        let missing =
            || Error::Invariant(format!("bucket {bucket} has no captured graph to replay"));
        let (generation, _) = served(graphs, serving, bucket).ok_or_else(missing)?;
        let graph = graphs.get_mut(&(*bucket, generation)).ok_or_else(missing)?;
        let (Some(executable), Some(buffer)) = (graph.executable.as_mut(), buffer.as_mut()) else {
            return Err(Error::State("the graph holds no executable to replay"));
        };
        // An executable the process declared stale since its instantiation (a remapping of the
        // storage it addresses, a change of how CUPTI traces graphs) is instantiated again before
        // this launch, and no other bucket's with it.
        executable.refresh()?;

        buffer.issue(slot, &cuda.stream, cuda.inputs, self.inputs, &graph.copies)?;
        buffer.release(slot, &cuda.stream)?;

        let step = step.take();
        begin_step(&cuda.stream, step)?;
        executable.launch(&cuda.stream)?;
        end_step(&cuda.stream, step)?;
        cuda.done.record(&cuda.stream)?;
        *launches += executable.kernel_nodes();
        *replays += 1;

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
    pub fn eager(&self, runner: &dyn Runner, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let cuda = self.cuda()?;
        let slot = self.take_prepared(bucket)?;

        self.with_buffer(|buffer| buffer.issue(slot, &cuda.stream, cuda.inputs, self.inputs, &[]))?;
        self.lend_values(&cuda.launching, slot)?;

        let step = self.state.borrow_mut().step.take();
        begin_step(&cuda.stream, step)?;
        cuda.launching.reset();
        let outputs = {
            let _guard = cuda.launching.enter();
            runner.forward(bucket)
        };
        end_step(&cuda.stream, step)?;

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
        }
        outputs
    }

    /// The kernels the serving forwards of these graphs launched since they were made: every
    /// kernel node of each replayed graph and every launch of each eager forward. Paired
    /// forwards and captures are not serving forwards; graphs that hold no device launch none.
    pub(crate) fn launches(&self) -> u64 {
        self.state.borrow().launches
    }

    /// The serving forwards of these graphs since they were made, by how they ran: the graphs
    /// [`Graphs::replay`] launched, then the forwards [`Graphs::eager`] ran to the end. Paired
    /// forwards and captures are not serving forwards; graphs that hold no device run none.
    pub(crate) fn forwards(&self) -> (u64, u64) {
        let state = self.state.borrow();
        (state.replays, state.eager)
    }

    // ---- Generations ----

    /// The number of the generation that serves.
    pub fn generation(&self) -> u64 {
        self.state.borrow().serving.number()
    }

    /// The generation that serves: the kernels a replay launches, which the device's context
    /// loaded. The engine builds the next generation of the device from it, and the memories of
    /// the device's rank launch their own work from it.
    pub(crate) fn kernels(&self) -> Arc<Generation> {
        Arc::clone(&self.state.borrow().serving)
    }

    /// Begins preparing `next` beside the serving generation and returns the captured buckets it
    /// examines, largest first. Each is examined, and prepared where it needs, by
    /// [`Graphs::prepare_bucket`], one bucket per call, so that the worker serves batches
    /// between two buckets: what a bucket needs takes a profile run of its forward, and a
    /// runner's buckets are counted in thousands. Beginning runs no forward.
    ///
    /// # Errors
    ///
    /// When an update is already in flight, or `next` is the serving generation's number.
    pub fn prepare(&self, runner: &dyn Runner, next: Arc<Generation>) -> Result<Vec<Bucket>> {
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

        let pending: Vec<Bucket> = runner
            .buckets()
            .into_iter()
            .filter(|bucket| state.served(bucket).is_some())
            .collect();

        state.uncompared.clear();
        state.update = Update::Prepared {
            next,
            pending: pending.clone(),
        };
        Ok(pending)
    }

    /// The next bucket the generation being prepared has yet to examine, the largest left;
    /// `None` when it examined every bucket, or no generation is being prepared.
    pub(crate) fn pending(&self) -> Option<Bucket> {
        match &self.state.borrow().update {
            Update::Prepared { pending, .. } => pending.first().copied(),
            _ => None,
        }
    }

    /// Examines `bucket` for the generation being prepared and prepares its graph where it needs
    /// one; returns what was done.
    ///
    /// What a bucket needs follows from launch tables: the serving graph's against a profile
    /// run of the same forward now. A table that differs is a recapture; an equal table with a
    /// launch whose kernel is another in the new generation is a rebind, which is done as a
    /// recapture when the capture under the new kernels turns out to differ in structure from
    /// the serving graph; an equal table none of whose kernels changed keeps its graph
    /// ([`Preparation::Keep`]).
    ///
    /// The serving graph is untouched and keeps serving.
    ///
    /// # Errors
    ///
    /// When no generation is being prepared or `bucket` is not left to examine, a profile run
    /// fails, the new generation holds no kernel for a launch of the bucket the serving
    /// generation has one for, or as [`Graphs::capture`]. The bucket is then still left to
    /// examine, and the caller gives the generation back ([`Graphs::rollback`]).
    pub fn prepare_bucket(&self, runner: &dyn Runner, bucket: &Bucket) -> Result<Preparation> {
        let (next, serving, held) = {
            let state = self.state.borrow();
            let Update::Prepared { next, pending } = &state.update else {
                return Err(Error::State("no generation is being prepared"));
            };
            let left = || {
                Error::Invariant(format!(
                    "generation {} has no bucket {bucket} left to examine",
                    next.number()
                ))
            };
            if !pending.contains(bucket) {
                return Err(left());
            }

            // A bucket is left to examine only while its serving graph exists.
            let held = state
                .served(bucket)
                .map(|(_, graph)| graph.table.clone())
                .ok_or_else(left)?;
            (Arc::clone(next), Arc::clone(&state.serving), held)
        };

        let (profiled, _, _) = self.meta_forward(&self.profile, runner, bucket)?;
        let needed = preparation(&held, &profiled, &serving, &next)?;

        // The capture under the new kernels, a rebind's as a recapture's, follows an eager run
        // of the forward with them: a kernel of a library the process loaded for this generation
        // has its first use there, where it does the host work of its first use that a capture
        // cannot record (a native entry may build its runner on first use, and refuses a capture
        // before it).
        let prepared = match needed {
            Preparation::Keep => None,
            Preparation::Rebind => {
                let fresh = self.captured(runner, bucket, &next)?;
                Some(self.rebound(bucket, fresh, &serving, &next)?)
            }
            Preparation::Recapture => Some((
                self.captured(runner, bucket, &next)?,
                Preparation::Recapture,
            )),
        };

        let mut state = self.state.borrow_mut();
        let done = match prepared {
            Some((graph, done)) => {
                state.graphs.insert((*bucket, next.number()), graph);
                done
            }
            None => Preparation::Keep,
        };
        if let Update::Prepared { pending, .. } = &mut state.update {
            pending.retain(|held| held != bucket);
        }
        Ok(done)
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
        runner: &dyn Runner,
        bucket: &Bucket,
        operator: &'static str,
    ) -> Result<Option<u64>> {
        if !matches!(self.state.borrow().update, Update::Prepared { .. }) {
            return Err(Error::State("no generation is being prepared"));
        }
        let slot = match self.take_step(bucket)? {
            Prepared::Inputs(_, slot) => slot,
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
            let Update::Prepared { next, .. } = &state.update else {
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

        self.with_buffer(|buffer| buffer.issue(slot, &cuda.stream, cuda.inputs, self.inputs, &[]))?;
        self.lend_values(&cuda.launching, slot)?;

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

    /// Switches to the prepared generation `number`: from the next batch on its graphs and
    /// kernels serve, on trial, and the generation it replaces is kept.
    ///
    /// # Errors
    ///
    /// When `number` is not the prepared generation, or a bucket is still to be examined for it.
    pub fn switch(&self, number: u64) -> Result<()> {
        let mut state = self.state.borrow_mut();

        let next = match &state.update {
            Update::Prepared { next, pending } if next.number() == number => {
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

    /// Commits the trial of generation `number`: the kept generation is released, with every
    /// graph a newer one replaced and the kernels only they held.
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
        for key in replaced {
            state.graphs.remove(&key);
        }

        // The last reference to the kept generation's kernels that no graph holds drops here.
        state.update = Update::Steady;
        Ok(())
    }

    /// Rolls generation `number` back. On trial, the kept generation serves again from the next
    /// batch on and the rolled-back generation's graphs are released; prepared and not yet
    /// switched to, its graphs are released and the serving generation is untouched.
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
                Update::Prepared { next, .. } => next.number() == number,
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
        state
            .graphs
            .retain(|(_, generation), _| *generation != number);
        Ok(())
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
        runner: &dyn Runner,
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

    /// Runs `operation` on the input buffer, making the host ring of graphs that hold no device
    /// when it is first needed or has become too small for the profiled forwards.
    fn with_buffer<T>(&self, operation: impl FnOnce(&mut InputBuffer) -> Result<T>) -> Result<T> {
        let values = self.input_peak();
        let mut state = self.state.borrow_mut();

        if self.device.is_none() {
            let stale = state
                .buffer
                .as_ref()
                .is_none_or(|buffer| buffer.values_bytes() < values);
            if stale {
                state.buffer = Some(InputBuffer::host(self.inputs, values, self.slots)?);
            }
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

    /// The slot holding the inputs prepared for `bucket`, which the caller now issues.
    fn take_prepared(&self, bucket: &Bucket) -> Result<usize> {
        match self.take_step(bucket)? {
            Prepared::Inputs(_, slot) => Ok(slot),
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

    /// Captures `bucket`'s forward with `generation`'s kernels, after one eager run of it with
    /// them, and checks the capture against the profile run and against the forward of
    /// `prepare_inputs`.
    ///
    /// Graphs that hold no device record the launch table and the input copies of a meta
    /// forward and no executable.
    fn captured(
        &self,
        runner: &dyn Runner,
        bucket: &Bucket,
        generation: &Arc<Generation>,
    ) -> Result<Graph> {
        // A bucket captured without a profile run is profiled here: its keys and peaks are
        // then known, and the capture is checked against them like any other.
        if !self.state.borrow().profiles.contains_key(bucket) {
            self.profile(runner, bucket)?;
        }

        let slot = self.fill_slot(|columns| runner.idle_inputs(bucket, columns))?;

        let Some(cuda) = &self.device else {
            self.lend_values(&self.preparing, slot)?;
            let (table, copies, _) = self.meta_forward(&self.preparing, runner, bucket)?;
            self.check_profile(bucket, &table)?;

            return Ok(Graph {
                table,
                copies,
                outputs: Vec::new(),
                executable: None,
            });
        };

        let serving = Arc::clone(&self.state.borrow().serving);
        cuda.launching.set_kernels(Arc::clone(generation));
        self.lend_values(&cuda.launching, slot)?;

        let captured = self.capture_on(cuda, runner, bucket, slot);

        // Eager forwards between captures launch the serving generation's kernels.
        cuda.launching.set_route(None);
        cuda.launching.set_breaks(None);
        cuda.launching.set_kernels(serving);
        let mut graph = captured?;

        self.check_profile(bucket, &graph.table)?;
        let launches = graph
            .table
            .iter()
            .filter(|(name, _, _)| *name != BREAK)
            .count();
        let Some(executable) = graph.executable.as_mut() else {
            return Err(Error::Invariant(
                "a capture on a device holds no executable".into(),
            ));
        };
        if executable.launches() != launches {
            return Err(Error::Invariant(format!(
                "the capture of bucket {bucket} recorded the nodes of {} launches and its launch \
                 table holds {launches}",
                executable.launches()
            )));
        }

        // The forward of `prepare_inputs` must write where the graph's input copies read: both
        // lists follow from the forward's shapes alone, so they are one list or the forward
        // cannot be replayed.
        self.lend_values(&self.preparing, slot)?;
        let (_, prepared, _) = self.meta_forward(&self.preparing, runner, bucket)?;
        if prepared != graph.copies {
            return Err(Error::Invariant(format!(
                "the forward of bucket {bucket} makes {} host-to-device copies when it is \
                 captured and {} under the meta context, or makes them at other offsets or \
                 addresses: its host values do not follow from the bucket alone",
                graph.copies.len(),
                prepared.len()
            )));
        }

        executable.hold(kernels_of(generation, &graph.table)?);
        tracing::debug!(
            %bucket,
            generation = generation.number(),
            launches,
            kernel_nodes = executable.kernel_nodes(),
            input_copies = graph.copies.len(),
            "captured a bucket"
        );
        Ok(graph)
    }

    /// The device steps of one capture: the eager run, then the capture itself. The graph's
    /// executable holds no kernel yet.
    fn capture_on(
        &self,
        cuda: &Cuda,
        runner: &dyn Runner,
        bucket: &Bucket,
        slot: usize,
    ) -> Result<Graph> {
        // The idle inputs reach the input tensors before either forward reads them.
        self.with_buffer(|buffer| buffer.issue(slot, &cuda.stream, cuda.inputs, self.inputs, &[]))?;

        cuda.launching.reset();
        let outputs = {
            let _guard = cuda.launching.enter();
            runner.forward(bucket)
        };
        outputs?;

        // The eager run's copies read the slot's values, which the capture writes again.
        cuda.stream.synchronize()?;

        cuda.launching.reset();
        cuda.launching.record_launches();
        let breaks: Arc<dyn Fn(HostWork) -> ops::Result<()> + Send + Sync> = Arc::new(take_break);
        cuda.launching.set_route(Some(Arc::new(Recorder)));
        cuda.launching.set_breaks(Some(breaks));

        let captured = {
            let _guard = cuda.launching.enter();
            Executable::capture(&cuda.stream, || {
                // A captured forward carries the range markers of the modules it enters, which
                // every replay then runs; the process's markers are a profiler only while the
                // stream captures.
                cuda.launching
                    .set_profiler(crate::profiling::markers::profiler(&cuda.stream));
                runner.forward(bucket)
            })
        };
        cuda.launching.set_profiler(None);

        let (table, _) = shadowed(cuda.launching.take_launches());
        let copies = cuda.launching.take_input_copies();

        // The slot is free once the stream has passed the copy of the idle inputs.
        self.with_buffer(|buffer| buffer.release(slot, &cuda.stream))?;
        cuda.done.record(&cuda.stream)?;
        let (executable, outputs) = captured?;

        Ok(Graph {
            table,
            copies,
            outputs,
            executable: Some(executable),
        })
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

    /// The graph of `bucket` for `next` as a rebind of the serving graph with `fresh`'s kernels
    /// on the launches whose kernel changed, or `fresh` itself as a recapture when the two
    /// captures differ in structure.
    fn rebound(
        &self,
        bucket: &Bucket,
        fresh: Graph,
        serving: &Generation,
        next: &Generation,
    ) -> Result<(Graph, Preparation)> {
        let state = self.state.borrow();
        let (_, held) = state.served(bucket).ok_or_else(|| {
            Error::Invariant(format!("bucket {bucket} has no serving graph to rebind"))
        })?;

        // A rebind keeps the serving graph's copies and outputs, so the new capture must have
        // made the same ones.
        if held.copies != fresh.copies {
            return Ok((fresh, Preparation::Recapture));
        }

        let executable = match (&held.executable, &fresh.executable) {
            (Some(held), Some(captured)) => {
                let changed = changed_launches(&fresh.table, serving, next);
                let kernels = kernels_of(next, &fresh.table)?;

                match held.rebound(captured, &changed, kernels)? {
                    Some(executable) => Some(executable),
                    None => return Ok((fresh, Preparation::Recapture)),
                }
            }
            _ => None,
        };

        let graph = Graph {
            table: fresh.table,
            copies: fresh.copies,
            outputs: held.outputs.clone(),
            executable,
        };
        Ok((graph, Preparation::Rebind))
    }
}

impl State {
    fn new(serving: Arc<Generation>, buffer: Option<InputBuffer>) -> Self {
        Self {
            serving,
            update: Update::Steady,
            profiles: HashMap::new(),
            shadows: HashMap::new(),
            inputs: HashMap::new(),
            graphs: HashMap::new(),
            buffer,
            prepared: None,
            step: None,
            shadow: None,
            uncompared: Vec::new(),
            launches: 0,
            replays: 0,
            eager: 0,
        }
    }

    /// The graph the serving generation serves `bucket` from, with the generation it was
    /// captured or rebound for.
    fn served(&self, bucket: &Bucket) -> Option<(u64, &Graph)> {
        served(&self.graphs, self.serving.number(), bucket)
    }
}

/// Launches the begin marker of `step` on `stream`, when the worker stated a step and the
/// process holds range markers.
fn begin_step(stream: &Stream, step: Option<(u64, &'static str)>) -> Result<()> {
    match (step, crate::profiling::markers::Markers::installed()) {
        (Some((step, stage)), Some(markers)) => markers.begin_step(stream, step, stage),
        _ => Ok(()),
    }
}

/// Launches the end marker of `step` on `stream`, right behind the step's last launch.
fn end_step(stream: &Stream, step: Option<(u64, &'static str)>) -> Result<()> {
    match (step, crate::profiling::markers::Markers::installed()) {
        (Some((step, _)), Some(markers)) => markers.end_step(stream, step),
        _ => Ok(()),
    }
}

/// The graph generation `serving` serves `bucket` from: the one of the newest generation at or
/// below it that holds a graph of the bucket.
///
/// A bucket none of whose kernels changed has no graph of the newer generation and keeps
/// serving from the one it has, and a generation prepared and not yet switched to serves
/// nothing.
fn served<'a>(
    graphs: &'a HashMap<(Bucket, u64), Graph>,
    serving: u64,
    bucket: &Bucket,
) -> Option<(u64, &'a Graph)> {
    graphs
        .iter()
        .filter(|((held, generation), _)| held == bucket && *generation <= serving)
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
fn same_launches(left: &Table, right: &Table) -> bool {
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

/// What `next` needs of a bucket whose serving graph has the launch table `held` and whose
/// profile run now states `profiled`.
fn preparation(
    held: &Table,
    profiled: &Table,
    serving: &Generation,
    next: &Generation,
) -> Result<Preparation> {
    // The new generation must hold a kernel wherever the serving one does: a bucket it cannot
    // launch is an incomplete generation, not one to switch to.
    for (operator, key, _) in profiled {
        if *operator != BREAK && serving.contains(operator, key) && !next.contains(operator, key) {
            return Err(Error::Interface(format!(
                "generation {} holds no kernel of `{operator}` for `{key}`, which generation {} \
                 launches",
                next.number(),
                serving.number()
            )));
        }
    }

    if !same_launches(held, profiled) {
        return Ok(Preparation::Recapture);
    }
    if changed_launches(profiled, serving, next).is_empty() {
        return Ok(Preparation::Keep);
    }
    Ok(Preparation::Rebind)
}

#[cfg(test)]
mod tests;
