//! What one forward borrows from the worker: the stream, the loaded kernels, the activation
//! arena and the communicator its collectives are issued through.
//!
//! A layer receives [`Forward`] and nothing else about the runtime: it cannot create a stream,
//! allocate device memory, capture a graph, or look up a program (`AGENTS.md`,
//! Model Library Boundary). Every device address a layer writes is either an argument of the
//! capability call or a view [`Arena`] handed out.
//!
//! ## The arena
//!
//! The design puts the concrete activation arena with the worker — a region of the
//! graph pool plus one static activation plan per captured bucket — but the worker also owns
//! `Forward`'s borrow of it, and the crate dependency runs `worker → nn`, so the type itself
//! lives here and the worker fills it. [`Arena`] is the bump region of one forward: the worker
//! resets it at the start of a forward, every layer takes its activations from it in a fixed order,
//! and because a captured bucket always executes the same sequence of layers on the same extents,
//! the sequence of reservations is the same on capture and on every replay. Replaying a graph
//! therefore touches exactly the addresses the plan recorded, and no step allocates.
//!
//! A single-threaded worker is what makes the interior cursor sound: one worker thread issues
//! every job of its device, so two forwards never share an arena at one time.

use std::cell::Cell;

use ops::{Generation, Op, Route, Stream};

use crate::error::{Error, Result};
use crate::group::{self, Group, Transfer};
use crate::layers::DispatchViews;
use crate::tensor::{DType, DevicePtr, Shape, TensorView};

/// The alignment every arena reservation starts at.
///
/// Kernels read through vector loads, and an operator's output may become another one's input, so
/// activations start at the widest alignment the element types here need; 256 bytes covers every
/// one of them and the vector loads Triton emits.
pub const ALIGNMENT: u64 = 256;

/// The activation region of one forward.
///
/// Besides the cursor, the arena keeps the furthest byte any reservation reached since it was made
/// ([`Arena::peak`]). A forward hands per-layer scratch back with [`Arena::release`], so the cursor
/// at the end of a forward understates what the forward addressed; the peak is the extent a
/// captured graph needs mapped.
pub struct Arena {
    base: DevicePtr,
    bytes: u64,
    cursor: Cell<u64>,
    peak: Cell<u64>,
}

/// The first address a counting arena hands out: aligned, and far from zero so that no synthetic
/// view reads as a null pointer to an argument check.
const COUNTING_BASE: u64 = 1 << 40;

impl Arena {
    /// The arena of `bytes` bytes starting at `base`, aligned up to [`ALIGNMENT`].
    ///
    /// `base` is a region the worker reserved in the graph pool for this captured bucket; the
    /// worker keeps it alive for as long as any graph that addresses it may replay.
    ///
    /// The base is aligned **up** to [`ALIGNMENT`] and the same number of bytes is taken off the
    /// region's size, so that every reservation is 256-byte aligned whatever the allocator answered.
    /// An arena hands its reservations out at that granularity, so a base carrying the pool's own
    /// skew would put every buffer of a forward at the same skew — and the tensor-core programs these
    /// buffers feed are compiled for sixteen-byte tensor pointers, which turns that skew into a warp
    /// misaligned address inside a kernel that names no operand. An eight-byte-skewed region costs
    /// at most 255 bytes here and nothing anywhere else.
    pub fn new(base: DevicePtr, bytes: u64) -> Self {
        let address = base.address();
        let aligned = address.next_multiple_of(ALIGNMENT);
        let skew = aligned - address;
        Self {
            base: DevicePtr::new(aligned),
            bytes: bytes.saturating_sub(skew),
            cursor: Cell::new(0),
            peak: Cell::new(0),
        }
    }

    /// An unbounded arena at synthetic addresses, for a recording forward.
    ///
    /// Every reservation succeeds at an [`ALIGNMENT`]-aligned address that no device memory backs,
    /// so a forward walked over it reports, through [`Arena::peak`], the extent the same forward
    /// reserves on a real arena: reservations depend on shapes and element types alone, never on
    /// the base address. Nothing may read or write a view of this arena; a recording forward
    /// launches nothing.
    pub fn counting() -> Self {
        Self::new(DevicePtr::new(COUNTING_BASE), u64::MAX - COUNTING_BASE)
    }

    /// The region's first address, aligned to [`ALIGNMENT`].
    pub const fn base(&self) -> DevicePtr {
        self.base
    }

    /// The region's usable size in bytes, after the base was aligned.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Bytes handed out since the last [`Arena::reset`], net of what [`Arena::release`] gave back.
    pub fn used(&self) -> u64 {
        self.cursor.get()
    }

    /// The furthest offset any reservation reached since this arena was made.
    ///
    /// Neither [`Arena::release`] nor [`Arena::reset`] lowers it: it is the extent of the region
    /// every forward run over this arena addressed, which is what the region must map.
    pub fn peak(&self) -> u64 {
        self.peak.get()
    }

    /// Start the next forward at the region's first address.
    ///
    /// The worker calls this once at the start of every forward, before the first layer runs. The
    /// reservations that follow are the ones the static plan recorded for this bucket.
    pub fn reset(&self) {
        self.cursor.set(0);
    }

    /// The current offset, for a later [`Arena::release`].
    ///
    /// One forward reserves its layer-independent buffers once and then runs every layer; the
    /// scratch a single layer reserves is dead as soon as that layer's launches are enqueued. A
    /// captured bucket replays the same addresses, so a layer that took a fresh reservation each
    /// pass would ask the region for `layers` copies of it — which is neither what the plan states
    /// nor what a replayed graph needs.
    pub fn mark(&self) -> u64 {
        self.cursor.get()
    }

    /// Hand back everything reserved since `mark`.
    ///
    /// The memory stays in the region and the bytes are handed out again to the next layer, which
    /// is what makes a plan one layer deep. The caller owes that nothing it still uses was reserved
    /// after `mark`: the stream orders the layer's launches before the next layer's, so reusing the
    /// same addresses is what a captured graph already does.
    pub fn release(&self, mark: u64) {
        self.cursor.set(mark);
    }

    /// Reserve `bytes` bytes at [`ALIGNMENT`] and answer their first address.
    pub fn reserve(&self, bytes: u64) -> Result<DevicePtr> {
        let at = self.cursor.get().next_multiple_of(ALIGNMENT);
        let end = at.checked_add(bytes).ok_or(Error::Arena {
            needed: bytes,
            at,
            available: self.bytes,
        })?;
        if end > self.bytes {
            return Err(Error::Arena {
                needed: bytes,
                at,
                available: self.bytes,
            });
        }
        self.cursor.set(end);
        self.peak.set(self.peak.get().max(end));
        Ok(DevicePtr::new(self.base.address() + at))
    }

    /// Reserve a contiguous row-major tensor of `shape` elements of `dtype`.
    pub fn tensor(&self, dtype: DType, shape: Shape) -> Result<TensorView> {
        let bytes = dtype
            .bytes()
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "{dtype} elements are sub-byte; their storage layout names their bytes"
                ))
            })?
            .checked_mul(shape.numel())
            .ok_or_else(|| {
                Error::shape(
                    "Arena::tensor",
                    format!("{shape:?} elements overflow a byte count"),
                )
            })?;
        Ok(TensorView::new(self.reserve(bytes)?, dtype, shape))
    }
}

/// The borrowed execution resources of one forward.
pub struct Forward<'a> {
    /// The stream this forward's work is issued on, or `None` for a **recording** forward: one that
    /// states the programs a graph launches instead of launching them.
    ///
    /// A recording forward exists so that the enumeration of a model's programs can be the model's own
    /// `forward` rather than a second traversal written beside it. It needs no device: the arena is a
    /// null-backed region, `operate` formats the descriptor and hands it to the capture sink, and no
    /// kernel is resolved. That is the difference between "the walk runs the model" (which would need
    /// a GPU and the checkpoint's shards) and this: the walk runs the model's *shape arithmetic*.
    stream: Option<&'a Stream>,
    /// The generation of loaded kernels the operators resolve their programs from, or `None` for a
    /// recording forward.
    kernels: Option<&'a Generation>,
    /// The activation region of this captured bucket.
    arena: &'a Arena,
    /// The communicator of this rank's parallel group, when a launching forward's block is
    /// partitioned. Layers reach it only through this forward's collectives.
    group: Option<&'a Group>,
    /// The split a recording forward answers in place of a group's: the rank whose shapes the
    /// walk follows, and the member count.
    split: Option<Split>,
    /// The expert-parallel dispatch's received views, when the block's experts are split across ranks.
    dispatch: Option<&'a DispatchViews>,
    /// Where this forward's launches go when the runtime routes them itself ([`Route`]), or `None`
    /// to launch every call once with the current generation's program.
    route: Option<&'a dyn Route>,
    /// `[1]` i32: the token rows of this batch that are live, when the caller states them.
    ///
    /// A captured bucket is wider than the batch that fills it. A layer that partitions work by
    /// token row needs to know which rows are real, and it cannot derive that from any shape: the
    /// arena is the bucket's. The value is device-resident because it changes per replay of one
    /// captured graph, and the buffer must outlive the bucket - a layer reads it on every replay.
    live_tokens: Option<TensorView>,
    /// Times the model's components on the device, or `None`.
    ///
    /// A forward captured into a graph captures its probes, and every replay then runs them. Nothing
    /// a forward computes depends on a fence being present.
    fence: Option<&'a dyn Fence>,
}

/// How a partitioned block is split across a group: this rank's index and the member count.
///
/// A layer's shape arithmetic over a partition (a vocabulary shard, a gather buffer, a rank's
/// share of tiles) reads this from [`Forward::split`], so a recording forward with a split answers
/// the same shapes the launching forward of that rank does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Split {
    /// This rank's index in the group, below `ranks`.
    pub rank: u64,
    /// How many ranks the group has; at least one.
    pub ranks: u64,
}

/// Where one collective of a forward goes.
enum Collective<'a> {
    /// Issued through the group on the forward's stream.
    Launch(&'a Group, &'a Stream),
    /// Checked against a recording forward's split and not issued.
    Record(Split),
}

/// Times a model's components on the device (`docs/profiling.md`, fence probes).
///
/// A model states each of its modules with [`Forward::fenced`]: the module's component, which names
/// its class across the deployment (`models.<family>.<module>.<Class>`), and its path in the model
/// (`layers.3`), which names this instance. Which components carry probes is the deployment's
/// choice, so an implementation answers `None` for a component no pattern names and the module then
/// runs as if unobserved.
pub trait Fence {
    /// The module instance at `path` of `component` is entered on `stream`: launch its entry probe.
    /// Answers the instance the exit names, or `None` when this component carries no probe.
    ///
    /// # Errors
    ///
    /// When the probe could not be launched.
    fn enter(
        &self,
        component: &'static str,
        path: std::fmt::Arguments<'_>,
        stream: &Stream,
    ) -> Result<Option<u32>>;

    /// The instance [`Self::enter`] answered returns on `stream`: launch its exit probe.
    ///
    /// # Errors
    ///
    /// When the probe could not be launched.
    fn exit(&self, instance: u32, stream: &Stream) -> Result<()>;
}

impl<'a> Forward<'a> {
    /// A forward with the resources every block needs and no partitioning.
    pub fn new(stream: &'a Stream, kernels: &'a Generation, arena: &'a Arena) -> Self {
        Self {
            stream: Some(stream),
            kernels: Some(kernels),
            arena,
            group: None,
            split: None,
            dispatch: None,
            route: None,
            live_tokens: None,
            fence: None,
        }
    }

    /// A forward that records the programs a graph launches instead of launching them.
    ///
    /// The arena is the caller's: a null-backed region is enough, because no kernel reads it. Every
    /// `operate` then formats the descriptor and reports it to the installed capture sink
    /// ([`ops::capture::Sink`]), so running a model's `forward` against this collects the exact set of
    /// programs it launches — the same set a device would launch, from the same code.
    ///
    /// A recording forward is unsplit until [`Self::with_split`] states the rank whose walk it
    /// follows; its collectives then check their views and issue nothing.
    pub fn recording(arena: &'a Arena) -> Self {
        Self {
            stream: None,
            kernels: None,
            arena,
            group: None,
            split: None,
            dispatch: None,
            route: None,
            live_tokens: None,
            fence: None,
        }
    }

    /// Whether this forward launches, rather than only recording what it would launch.
    pub const fn is_recording(&self) -> bool {
        self.stream.is_none()
    }

    /// The stream this forward's launches and collectives are issued on.
    ///
    /// A recording forward has none: every `operate` returns before a kernel is resolved, and its
    /// collectives issue nothing.
    pub fn stream(&self) -> Result<&'a Stream> {
        self.stream.ok_or_else(|| {
            Error::Unsupported(
                "this forward records the programs a graph launches and has no stream".to_owned(),
            )
        })
    }

    /// Fills the first `bytes` of `view` with zero, in stream order with this forward's launches.
    ///
    /// A recording forward has no stream and issues nothing, as it issues no launch.
    pub fn zero(&self, view: TensorView, bytes: u64) -> Result<()> {
        let Some(stream) = self.stream else {
            return Ok(());
        };
        // SAFETY: `view` is a live tensor of this forward's arena and the caller's `bytes` lie in it.
        unsafe { ops::zero_bytes(view.ptr, bytes, stream) }.map_err(Error::from)
    }

    /// The same forward over a parallel group.
    ///
    /// The group's rank and member count are this forward's [`Split`], and its collectives are
    /// issued through the group on this forward's stream.
    pub fn with_group(mut self, group: &'a Group) -> Self {
        self.group = Some(group);
        self
    }

    /// The same recording forward split as rank `rank` of `ranks`.
    ///
    /// A recording forward has no communicator, so this states the split the walk follows: every
    /// layer answers the shapes rank `rank` of a group of `ranks` launches with, and every collective
    /// checks its views against that split and issues nothing.
    ///
    /// # Errors
    ///
    /// When `ranks` is zero or `rank` is not below it, and on a launching forward, whose split is
    /// its group's.
    pub fn with_split(mut self, rank: u64, ranks: u64) -> Result<Self> {
        if !self.is_recording() {
            return Err(Error::Unsupported(
                "a launching forward is split by the group it is given".to_owned(),
            ));
        }
        if rank >= ranks {
            return Err(Error::shape(
                "Forward::with_split",
                format!("rank {rank} is not a member of a group of {ranks}"),
            ));
        }
        self.split = Some(Split { rank, ranks });
        Ok(self)
    }

    /// This forward's split: its group's rank and member count, the split a recording forward was
    /// given, or `None` when the block is not partitioned.
    ///
    /// A group of one member is a split: its collectives are issued as for any group.
    pub fn split(&self) -> Option<Split> {
        match self.group {
            Some(group) => Some(Split {
                // NCCL reports a member's rank as a non-negative `int`.
                rank: u64::try_from(group.rank()).unwrap_or(0),
                ranks: group.ranks(),
            }),
            None => self.split,
        }
    }

    /// Sums every rank's `view` into every rank's `view`, in place.
    ///
    /// `view` is the same shape and element type on every rank. A launching forward issues the
    /// collective through its group on its stream; a recording forward checks `view` and issues
    /// nothing, which leaves every shape as it is.
    ///
    /// # Errors
    ///
    /// When the forward has no split, or the communicator cannot move `view`'s element type.
    pub fn all_reduce(&self, view: TensorView) -> Result<()> {
        match self.collective("all_reduce")? {
            Collective::Launch(group, stream) => group.all_reduce_sum(view, stream),
            Collective::Record(_) => group::collective(view, view, "all_reduce").map(drop),
        }
    }

    /// Gathers every rank's `send` into `recv`, in rank order.
    ///
    /// `recv` holds the split's member count times `send`'s elements. A recording forward checks the
    /// views against its split and issues nothing.
    ///
    /// # Errors
    ///
    /// When the forward has no split, or `recv` does not hold every member's `send`.
    pub fn all_gather(&self, send: TensorView, recv: TensorView) -> Result<()> {
        match self.collective("all_gather")? {
            Collective::Launch(group, stream) => group.all_gather(send, recv, stream),
            Collective::Record(split) => group::gathered(send, recv, split.ranks).map(drop),
        }
    }

    /// Exchanges equal blocks with every rank: block `j` of `send` goes to rank `j`, and block `j`
    /// of `recv` is what rank `j` sent here.
    ///
    /// Both buffers hold the split's member count of equal blocks. A recording forward checks the
    /// views against its split and issues nothing.
    ///
    /// # Errors
    ///
    /// When the forward has no split, or the buffers are not equal blocks of one element type.
    pub fn all_to_all(&self, send: TensorView, recv: TensorView) -> Result<()> {
        match self.collective("all_to_all")? {
            Collective::Launch(group, stream) => group.all_to_all(send, recv, stream),
            Collective::Record(split) => group::exchanged(send, recv, split.ranks).map(drop),
        }
    }

    /// Starts [`Self::all_to_all`] beside this forward's stream, which goes on at once.
    ///
    /// What `recv` holds is readable after [`Self::finish`] took the transfer up; the buffers
    /// belong to the transfer until then. A recording forward checks the views and answers a
    /// transfer that orders nothing.
    ///
    /// # Errors
    ///
    /// As [`Self::all_to_all`].
    pub fn start_all_to_all(&self, send: TensorView, recv: TensorView) -> Result<Transfer> {
        match self.collective("all_to_all")? {
            Collective::Launch(group, stream) => group.start_all_to_all(send, recv, stream),
            Collective::Record(split) => {
                group::exchanged(send, recv, split.ranks)?;
                Ok(Transfer::recorded())
            }
        }
    }

    /// Starts delivering rows `first..first + count` of this rank's `owner_rows` rows to every
    /// member, and theirs here, in place in `gathered`.
    ///
    /// `gathered` is the whole sequence `[members * owner_rows, width]`, member by member, and this
    /// rank's rows of the interval are already in its own block. The rows are readable after
    /// [`Self::finish`] took the transfer up. A recording forward answers a transfer that orders
    /// nothing.
    ///
    /// # Errors
    ///
    /// When the forward has no split, or the communicator cannot move `gathered`'s element type.
    pub fn start_gather_rows(
        &self,
        gathered: TensorView,
        owner_rows: u64,
        first: u64,
        count: u64,
    ) -> Result<Transfer> {
        match self.collective("gather_rows")? {
            Collective::Launch(group, stream) => {
                group.start_gather_rows(gathered, owner_rows, first, count, stream)
            }
            Collective::Record(_) => {
                group::collective(gathered, gathered, "gather_rows")?;
                Ok(Transfer::recorded())
            }
        }
    }

    /// Orders this forward's stream after `transfer`: what it received is readable from here on.
    ///
    /// # Errors
    ///
    /// When the forward has no split, or the wait could not be issued.
    pub fn finish(&self, transfer: Transfer) -> Result<()> {
        match self.collective("finish")? {
            Collective::Launch(group, stream) => group.finish(transfer, stream),
            Collective::Record(_) => Ok(()),
        }
    }

    /// Where a collective named `name` goes: the group and stream of a launching forward, or the
    /// split of a recording one.
    fn collective(&self, name: &'static str) -> Result<Collective<'a>> {
        match (self.group, self.split) {
            (Some(group), _) => Ok(Collective::Launch(group, self.stream()?)),
            (None, Some(split)) => Ok(Collective::Record(split)),
            (None, None) => Err(Error::Unsupported(format!(
                "{name}: a collective runs over a group, and this forward carries none"
            ))),
        }
    }

    /// Waits for what this forward issued so far, when it is not being captured.
    ///
    /// A launch that waits on its device for the other ranks of a group holds that device until
    /// they arrive, and the first use of a kernel takes a lock of the whole process and waits for
    /// its device to drain. The ranks of a host are threads of one process, so a rank that walked
    /// on past such a launch can hold that lock for a kernel that needs its peers, while the peers
    /// queue on the lock in front of the same launch. A captured walk runs no kernel, and a replay
    /// finds every kernel ready, so only an eager walk is held back here (`Group::settled` states
    /// the same rule for a collective).
    pub fn settled(&self) -> Result<()> {
        let Some(stream) = self.stream else {
            return Ok(());
        };
        if stream.capture_status().map_err(Error::Platform)? == ops::platform::CaptureStatus::None {
            stream.synchronize().map_err(Error::Platform)?;
        }
        Ok(())
    }

    /// The same forward over an expert-parallel dispatch.
    pub fn with_dispatch(mut self, dispatch: &'a DispatchViews) -> Self {
        self.dispatch = Some(dispatch);
        self
    }

    /// The same forward with its launches sent through `route` ([`Route`]).
    pub fn with_route(mut self, route: &'a dyn Route) -> Self {
        self.route = Some(route);
        self
    }

    /// State which token rows of this batch are live, for the layers that partition by row.
    ///
    /// The buffer is `[1]` i32 on the device and must outlive the bucket: a captured graph reads it
    /// on every replay, so a buffer reallocated per request is a dangling pointer in a graph that
    /// outlives the request.
    pub fn with_live_tokens(mut self, tokens: TensorView) -> Self {
        self.live_tokens = Some(tokens);
        self
    }

    /// Observe every layer's attention output at the launch that wrote it.
    pub fn with_fence(mut self, fence: Option<&'a dyn Fence>) -> Self {
        self.fence = fence;
        self
    }

    /// Runs `body`, the forward of the module instance at `path` of `component`, between its fence
    /// probes when this forward has a fence and the deployment probes the component.
    ///
    /// # Errors
    ///
    /// What `body` answers, or the probe's own failure.
    pub fn fenced<T, E: From<Error>>(
        &self,
        component: &'static str,
        path: std::fmt::Arguments<'_>,
        body: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        let (Some(fence), Some(stream)) = (self.fence, self.stream) else {
            return body();
        };
        let instance = fence.enter(component, path, stream)?;
        let answer = body()?;
        if let Some(instance) = instance {
            fence.exit(instance, stream)?;
        }
        Ok(answer)
    }

    /// The token rows of this batch that are live, `[1]` i32 on the device, when the caller stated
    /// them ([`Self::with_live_tokens`]).
    pub fn live_tokens(&self) -> Option<TensorView> {
        self.live_tokens
    }

    /// The expert-parallel dispatch's received views, when the caller stated them
    /// ([`Self::with_dispatch`]).
    pub fn dispatch(&self) -> Option<&'a DispatchViews> {
        self.dispatch
    }

    /// Check the call against `O`'s rules ([`Op::check`]), then resolve `O`'s kernel for this
    /// descriptor and launch it.
    ///
    /// A recording forward checks the call the same way and then only reports the descriptor key
    /// to the capture sink, so a recorded key always comes from a call the device would accept.
    ///
    /// A forward with a [`Route`] offers the checked call to it first; a call the route declines,
    /// and every call of a forward without one, is one launch of the current generation's kernel.
    pub fn operate<'t, O: Op>(&self, descriptor: &O::Descriptor, args: O::Args<'t>) -> Result<()>
    where
        O::Args<'t>: Copy,
    {
        let refused = |error| Error::launch(O::NAME, ops::descriptor::key(descriptor), error);
        O::check(descriptor, &args).map_err(refused)?;
        // A recording forward states the program and stops: no kernel to resolve, no launch to make,
        // and the descriptor key formatted only when a sink will read it.
        let (Some(stream), Some(kernels)) = (self.stream, self.kernels) else {
            if ops::capture::installed() {
                ops::capture::launched(O::NAME, &ops::descriptor::key(descriptor));
            }
            return Ok(());
        };
        if let Some(route) = self.route
            && route.launch(&ops::Checked::<O>::new(descriptor, args), kernels, stream)?
        {
            return Ok(());
        }
        let kernel = kernels.op::<O>(descriptor)?;
        O::launch(kernel, descriptor, args, stream).map_err(refused)?;
        Self::record::<O>(descriptor);
        Ok(())
    }

    /// Tell the capture sink, when one is installed, that `O`'s program just launched.
    ///
    /// A capture records the graph nodes each operator launch added, and a program update rebinds
    /// one operator's parameters at a time through that table. The key is
    /// formatted only when a sink will read it: every launch of a serving step passes through here,
    /// and a capture is the exception, not the rule.
    fn record<O: Op>(descriptor: &O::Descriptor) {
        if ops::capture::installed() {
            ops::capture::launched(O::NAME, &ops::descriptor::key(descriptor));
        }
    }

    /// The arena of this forward.
    pub fn arena(&self) -> &Arena {
        self.arena
    }
}

/// Runs `walk` with a capture sink installed and answers what it reported, as `(operator, key)`
/// pairs in launch order: the observation a layer's test makes of its forward on a recording
/// [`Forward`].
#[cfg(test)]
pub(crate) fn recorded<T>(walk: impl FnOnce() -> T) -> (T, Vec<(String, String)>) {
    struct Launches(Vec<(String, String)>);

    impl ops::capture::Sink for Launches {
        fn launched(&mut self, operator: &'static str, key: &str) {
            self.0.push((operator.to_owned(), key.to_owned()));
        }
    }

    let mut launches = Launches(Vec::new());
    ops::capture::set_sink(Some(&mut launches));
    let answer = walk();
    ops::capture::set_sink(None);
    (answer, launches.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A recording forward states the program its graph would launch and launches nothing: no stream,
    /// no kernel generation, no device. That is what lets a model's own `forward` be the enumeration of
    /// its programs — a traversal that cannot miss a launch, on a machine that has no GPU and no shards.
    #[test]
    fn a_recording_forward_reports_the_program_it_would_launch() {
        let arena = Arena::new(DevicePtr::new(0), 1 << 20);
        let forward = Forward::recording(&arena);
        assert!(forward.is_recording());

        let extent = Shape::new(&[4, 1024]);
        let x = arena.tensor(DType::BF16, extent).unwrap();
        let out = arena.tensor(DType::BF16, extent).unwrap();
        let (walked, launches) = recorded(|| crate::layers::Silu.forward(&forward, x, out));
        walked.expect("a recording forward launches nothing");

        assert_eq!(launches.len(), 1, "{launches:?}");
        let (operator, key) = &launches[0];
        assert_eq!(operator, "silu");
        assert!(key.starts_with("dtype_bf16_block_"), "{key}");
    }

    #[test]
    fn reservations_are_aligned_and_do_not_overlap() {
        let arena = Arena::new(DevicePtr::new(0x1_0000), 4096);
        let first = arena.tensor(DType::BF16, Shape::new(&[3])).unwrap();
        let second = arena.tensor(DType::F32, Shape::new(&[2])).unwrap();
        assert_eq!(first.ptr.address(), 0x1_0000);
        assert_eq!(second.ptr.address(), 0x1_0000 + ALIGNMENT);
        assert_eq!(arena.used(), ALIGNMENT + 8);
    }

    /// A pool may answer a region at eight-byte granularity, and the programs these buffers feed are
    /// compiled for sixteen-byte tensor pointers: the arena's own base is where that has to be
    /// fixed, because a reservation cannot correct a skew it inherits.
    #[test]
    fn a_skewed_region_is_aligned_up_before_anything_is_reserved() {
        let arena = Arena::new(DevicePtr::new(0x1_0008), 4096);
        assert_eq!(arena.base().address(), 0x1_0100);
        assert_eq!(arena.bytes(), 4096 - 0xF8);
        let tensor = arena.tensor(DType::BF16, Shape::new(&[8])).unwrap();
        assert_eq!(tensor.ptr.address() % ALIGNMENT, 0);
        assert_eq!(tensor.ptr.address(), 0x1_0100);
        // An already-aligned region keeps its address and its whole size.
        let held = Arena::new(DevicePtr::new(0x1_0000), 4096);
        assert_eq!(held.base().address(), 0x1_0000);
        assert_eq!(held.bytes(), 4096);
    }

    #[test]
    fn a_reset_replays_the_same_addresses() {
        let arena = Arena::new(DevicePtr::new(0x1_0000), 4096);
        let first = arena.tensor(DType::BF16, Shape::new(&[16])).unwrap();
        arena.reset();
        let again = arena.tensor(DType::BF16, Shape::new(&[16])).unwrap();
        assert_eq!(first.ptr, again.ptr);
    }

    /// One forward reserves its layer-independent buffers once and runs every layer: a layer's
    /// scratch is handed back at its end and the next layer receives the same addresses, which is
    /// what a replayed graph expects and what keeps the plan one layer deep.
    #[test]
    fn releasing_to_a_mark_hands_the_same_scratch_to_the_next_layer() {
        let arena = Arena::new(DevicePtr::new(0x1_0000), 4096);
        let held = arena.tensor(DType::BF16, Shape::new(&[16])).unwrap();
        let mark = arena.mark();
        let first = arena.tensor(DType::BF16, Shape::new(&[1024])).unwrap();
        arena.release(mark);
        let second = arena.tensor(DType::BF16, Shape::new(&[1024])).unwrap();
        assert_eq!(first.ptr, second.ptr);
        assert_eq!(second.ptr.address(), held.ptr.address() + ALIGNMENT);
        // The buffer reserved before the mark is still the arena's, and still where it was.
        assert_eq!(arena.used(), ALIGNMENT + 2048);
    }

    /// A forward that hands a layer's scratch back still addressed it: the peak keeps the furthest
    /// byte reached, which is the extent a captured graph needs mapped, while the cursor falls back.
    #[test]
    fn the_peak_keeps_what_a_release_hands_back() {
        let arena = Arena::new(DevicePtr::new(0x1_0000), 1 << 20);
        arena.tensor(DType::BF16, Shape::new(&[16])).unwrap();
        let mark = arena.mark();
        arena.tensor(DType::F32, Shape::new(&[1024])).unwrap();
        arena.release(mark);
        arena.tensor(DType::F32, Shape::new(&[8])).unwrap();
        assert_eq!(arena.used(), ALIGNMENT + 32);
        assert_eq!(arena.peak(), ALIGNMENT + 4096);
        arena.reset();
        assert_eq!(arena.peak(), ALIGNMENT + 4096);
    }

    /// A counting arena measures what a real arena of any base would reserve, without a bound.
    #[test]
    fn a_counting_arena_reserves_what_a_real_one_does() {
        let counting = Arena::counting();
        let real = Arena::new(DevicePtr::new(0x1_0000), 1 << 20);
        for arena in [&counting, &real] {
            arena.tensor(DType::BF16, Shape::new(&[3])).unwrap();
            arena.tensor(DType::F32, Shape::new(&[1000])).unwrap();
        }
        assert_eq!(counting.peak(), real.peak());
        assert_eq!(counting.base().address() % ALIGNMENT, 0);
        assert!(counting.tensor(DType::F32, Shape::new(&[1 << 40])).is_ok());
    }

    #[test]
    fn a_reservation_past_the_region_fails() {
        let arena = Arena::new(DevicePtr::new(0), 512);
        assert!(arena.tensor(DType::F32, Shape::new(&[256])).is_err());
    }
}
