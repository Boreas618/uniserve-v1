//! One rank's member of an expert union: the step agreement, the capacity classes of its
//! buckets, and the exchange its forwards run over.
//!
//! The members of an expert union run every expert layer together: an exchange launch on one
//! member waits on its device for the same launch of every other. A member with a batch
//! therefore needs every other member inside the same step. A member with a batch of its own
//! runs it in that step, and a member without one **joins**: it runs the expert layers with
//! no tokens of its own, so its experts serve the tokens routed to them.
//!
//! ## The agreement
//!
//! Before a step the members agree its capacity, the most tokens any member sends, by one
//! gather of their token counts over the union's host group ([`Member::enter`]). The gather
//! blocks each member's worker until every member has entered, which is what keeps a device
//! from running ahead of the sequence of steps; nothing spins and nothing polls.
//!
//! An idle member's worker waits for work, so a member that enters with tokens first asks
//! the others to join: it submits a join batch (`Variant::ExpertJoin`) to their workers
//! through the hook the engine gave the union. Steps are numbered, every member takes part in
//! every step, and a join batch carries the number of the step it asks for in its batch
//! number. A member that receives one for a step it already took part in, as a member with a
//! batch of its own does when two members ask each other, finds the step closed and does
//! nothing ([`Member::join`]).
//!
//! ## Capacity classes
//!
//! The expert work of a graph is shaped for the rows per member its step holds. A bucket is
//! therefore captured once per capacity class ([`classes`]), and a step runs the smallest
//! class that holds the agreed capacity. The transport reads the exact capacity from device
//! memory, where [`Member::write`] puts it ahead of the step's first launch. A speculating
//! member's step is two graphs, the target's and the drafter's behind it, both at the class
//! the agreement chose: the step's exchanges are the target's expert layers and then the
//! drafter's, once per drafter forward, and a member's join runs as many.
//!
//! ## The exchange
//!
//! The kernel package owns the transport. A member creates it once, collectively with the
//! other members ([`Member::create_transport`]), and lends a forward the regions the transport
//! laid out as the tensors of an `nn::ExpertExchange`. Before a transport exists the regions
//! are meta tensors of the extents a transport of the member's shape lays out, which is what a
//! profile run computes over. The package reaches the other members through the same host
//! group, for the gathers and barriers of its own startup.
//!
//! The exchanges differ in the module that creates the transport: `alltoall` and `deepep`
//! dispatch and combine around each member's experts ([`Mode::Dispatch`]), and `megamoe` runs
//! the whole expert layer in one launch ([`Mode::Fused`]). A member of the leading attention
//! ranks holds no experts and only sends; a member past them holds experts alone, and every
//! forward of its buckets is the join.

use std::ffi::c_void;
use std::sync::Arc;

use crate::batch::{Batch, BatchId, Columns, Shape as BatchShape, Variant};
use nn::ExpertExchange;
use ops::Generation;
use ops::moe::ExchangeTransport;
use ops::moe::exchange::{Descriptor, Mode, Transport, TransportShape};
use platform::Stream;
use tensor::Tensor;

use crate::parallel::HostGroup;
use crate::runner::buckets::{Bucket, Buckets};
use crate::worker::{Error, Result};

/// What submits a join batch to the worker of every other member of the union.
pub(crate) type JoinHook = Arc<dyn Fn(Batch) -> Result<()> + Send + Sync>;

/// The buckets of a member of an expert union: each bucket of `buckets`, with a speculating
/// member's verification and drafter buckets (`Buckets::with_drafts`), and the join, captured
/// for the step capacities it meets, of which `capacity` is the largest.
///
/// When every member decodes, a step holds at most the largest decode bucket's rows, or the
/// largest verification bucket's tokens of a speculating union, which the smallest class
/// holds. A decode or a verification beside another member's prefill of a few prompts takes
/// the middle class of `beside` rows (`moe_config.beside_rows`), and a step another member
/// fills with a large prefill takes the class of the whole capacity. One class for every step
/// would shape each decode's expert work for the largest prefill any member could send. A
/// class within a quarter of the whole capacity is not captured: the smaller shape would save
/// little. A drafter's bucket replays behind the bucket it follows, at the class the step
/// agreed for both, so it is captured at that bucket's classes.
pub(crate) fn classes(buckets: &Buckets, capacity: u32, beside: u32) -> Buckets {
    let steps = |variant: Variant| variant.is_decode() || super::verifies(variant);
    let decode = buckets
        .iter()
        .filter(|bucket| steps(bucket.variant))
        .map(|bucket| bucket.tokens)
        .max()
        .unwrap_or(1);

    // The capacities a bucket that a step of its own serves is captured at.
    let capacities = |bucket: &Bucket| {
        let small = bucket.tokens.max(decode).min(capacity);
        let mut classes = vec![capacity];

        if small.saturating_mul(4) <= capacity {
            classes.push(small);
        }

        let decodes = steps(bucket.variant) || bucket.variant == Variant::ExpertJoin;
        if decodes && small < beside && beside.saturating_mul(4) <= capacity {
            classes.push(beside);
        }
        classes
    };

    let targets = buckets
        .iter()
        .copied()
        .filter(|bucket| bucket.variant != Variant::Draft)
        .chain(std::iter::once(Bucket::expert_join()));

    Buckets::new(targets.flat_map(|target| {
        let drafter = Bucket::draft(target.rows, target.tokens);
        let followed = buckets.contains(&drafter).then_some(drafter);

        capacities(&target).into_iter().flat_map(move |class| {
            std::iter::once(target)
                .chain(followed)
                .map(move |bucket| bucket.at_capacity(class))
        })
    }))
}

/// What a member of an expert union lends its runner's forwards ([`Member::regions`]): the
/// transport's regions and the shape of an exchange over them. The runner's forward, and every
/// instance of it the worker's capture thread computes (`Runner::capturer`), shapes its exchange
/// from these alone; the member itself, which agrees each step with the others, stays with the
/// runner.
#[derive(Clone)]
pub(crate) struct Regions {
    /// What the member states of the transport: the union's ranks, its own, and the rows.
    shape: TransportShape,
    /// The regions an exchange reads, as [`Member`] holds them.
    regions: [Tensor; 7],
    /// Whether one launch runs the whole expert layer.
    fused: bool,
}

impl Regions {
    /// The exchange of a forward shaped for `step_rows` rows a member.
    ///
    /// # Errors
    ///
    /// What `nn::ExpertExchange::check` refuses: a step of no row, or of more than the
    /// transport holds.
    pub(crate) fn exchange(&self, step_rows: u32) -> Result<ExpertExchange> {
        let [states, ids, weights, combined, live, packed, packed_scales] = self.regions.clone();
        let exchange = ExpertExchange {
            ranks: self.shape.ranks,
            attention: self.shape.attention_ranks,
            rank_rows: self.shape.max_tokens,
            step_rows,
            multiprocessors: self.shape.multiprocessors,
            states,
            ids,
            weights,
            combined,
            live,
            packed,
            packed_scales,
            fused: self.fused,
            source_only: self.shape.rank < self.shape.attention_ranks,
        };

        exchange.check()?;
        Ok(exchange)
    }

    /// Whether the member holds experts alone: its forwards run the expert layers of other
    /// members' tokens and have no tokens of their own.
    pub(crate) const fn experts_only(&self) -> bool {
        self.shape.attention_ranks > 0 && self.shape.rank >= self.shape.attention_ranks
    }
}

/// One rank's member of an expert union (module documentation).
pub(crate) struct Member {
    /// The union's host group: the agreement's gathers, and the package's own.
    group: Arc<HostGroup>,
    /// What the member states of the transport: the union's ranks, its own, and the rows.
    shape: TransportShape,
    /// The module whose kernel creates the transport.
    mode: Mode,
    /// What asks the other members to join a step.
    join: JoinHook,
    /// The steps the union has agreed: the number of the next one.
    round: u64,
    /// The members' token counts of the step being agreed, four bytes a member.
    records: Vec<u8>,
    /// The regions an exchange reads, in the order `TransportShape::regions` states them:
    /// states, ids, weights, combined, live, packed and packed scales.
    regions: [Tensor; 7],
    /// Whether one launch runs the whole expert layer.
    fused: bool,
    /// The rows per rank of the union's middle capture class (`moe_config.beside_rows`,
    /// [`classes`]).
    beside: u32,
    /// The transport, once the members created it.
    transport: Option<Transport>,
    /// What the package reaches the host group through: `link.handle` is the address of
    /// `group`'s value, which the member keeps, and the transport holds `link`'s address, so
    /// the box lives as long as the member.
    link: Box<ops::moe::exchange::HostGroup>,
    /// The union's device communicator, which a model's collectives along the union run over
    /// (`nn::family::EXPERT_AXIS`); `None` for a model that collects nothing along it, and on
    /// the meta device, whose collectives issue nothing.
    communicator: Option<Arc<ops::Group>>,
}

// SAFETY: a member moves once, with its runner, to its device's worker thread and is used
// there alone; the raw addresses it holds are its own boxes and the package's transport.
unsafe impl Send for Member {}

impl Member {
    /// The member of `shape.rank` in the union whose members meet over `group`, before its
    /// transport exists.
    ///
    /// `mode` is the module that creates the transport: [`Mode::Fused`] for an exchange whose
    /// one launch runs the expert layer, [`Mode::Dispatch`] for any other. `join` submits a
    /// join batch to the worker of every other member; the batch states the phase the
    /// asking member serves and its group is the hook's to address. `beside` is the rows per
    /// rank of the union's middle capture class ([`classes`]).
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] when `group` is not the union's ranks with this member at its rank,
    /// or the shape lays out no region.
    pub(crate) fn new(
        group: Arc<HostGroup>,
        shape: TransportShape,
        mode: Mode,
        join: JoinHook,
        beside: u32,
    ) -> Result<Self> {
        if group.size() != shape.ranks as usize || group.rank() != shape.rank as usize {
            return Err(Error::Invalid(format!(
                "member {} of a union of {} ranks over a host group of {} members, entered as \
                 member {}",
                shape.rank,
                shape.ranks,
                group.size(),
                group.rank()
            )));
        }

        // A profile run computes over regions of the transport's extents and no memory.
        let meta = ops::meta();
        let mut regions = Vec::with_capacity(7);
        for (dtype, extents) in shape.regions() {
            let dims: Vec<usize> = extents.iter().map(|&extent| extent as usize).collect();
            // SAFETY: a tensor over the meta device has no memory: its address is never read.
            regions.push(unsafe { ops::external(0, dims, dtype, &meta) }?);
        }

        let link = Box::new(ops::moe::exchange::HostGroup {
            handle: Arc::as_ptr(&group).cast_mut().cast(),
            communicator: std::ptr::null_mut(),
            ranks: i64::from(shape.ranks),
            rank: i64::from(shape.rank),
            device: i64::from(shape.device),
            all_gather,
            barrier,
            error: [0; 512],
        });

        Ok(Self {
            records: vec![0; 4 * group.size()],
            group,
            shape,
            mode,
            join,
            round: 0,
            regions: regions.try_into().map_err(|_| {
                Error::Invariant("a transport lays out another number of regions".into())
            })?,
            fused: mode == Mode::Fused,
            beside,
            transport: None,
            link,
            communicator: None,
        })
    }

    /// The member whose forwards' collectives along the union run over `communicator`.
    #[must_use]
    pub(crate) fn with_communicator(self, communicator: Option<Arc<ops::Group>>) -> Self {
        Self {
            communicator,
            ..self
        }
    }

    /// The union's axis of the member's forwards (`nn::family::EXPERT_AXIS`): its communicator
    /// and the member's place among the union's members, as `ops::Context::set_axes` takes an
    /// axis.
    pub(crate) fn axis(&self) -> (Option<Arc<ops::Group>>, u64, u64) {
        (
            self.communicator.clone(),
            u64::from(self.shape.rank),
            u64::from(self.shape.ranks),
        )
    }

    /// Creates the transport with the exchange kernel of `kernels`, collectively: every member
    /// of the union calls this at the same point of its startup, each on its worker thread,
    /// and the call returns once the members have met. The regions the member lends are the
    /// transport's from then on.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] when `kernels` holds no exchange kernel for the member's shape or
    /// the package refuses; [`Error::State`] for a member that holds a transport.
    pub(crate) fn create_transport(&mut self, kernels: &Generation, stream: &Stream) -> Result<()> {
        if self.transport.is_some() {
            return Err(Error::State("the member holds its transport"));
        }

        let kernel = kernels
            .op::<ops::moe::ExpertExchange>(&self.descriptor())
            .map_err(|error| Error::Invalid(format!("the expert exchange's kernel: {error}")))?;
        let transport = Transport::create(
            Arc::clone(kernel),
            std::ptr::from_ref(self.link.as_ref()),
            self.shape,
            stream,
        )
        .map_err(|error| Error::Invalid(format!("creating the expert transport: {error}")))?;

        let views = transport
            .views()
            .map_err(|error| Error::Invalid(format!("reading the expert transport: {error}")))?;
        let device = ops::cuda(self.shape.device as usize);
        let lent = [
            views.states,
            views.ids,
            views.weights,
            views.combined,
            views.live,
            views.packed,
            views.packed_scales,
        ];
        for (region, view) in self.regions.iter_mut().zip(lent) {
            // SAFETY: the region is the transport's own memory, laid out with these extents,
            // which the package keeps for as long as the transport this member holds.
            *region = unsafe {
                ops::external(
                    view.ptr.address(),
                    view.shape().clone(),
                    view.dtype,
                    &device,
                )
            }?;
        }

        self.fused = views.fused;
        self.transport = Some(transport);
        Ok(())
    }

    /// The most rows one member sends in a step: the capacity of the largest class.
    pub(crate) const fn capacity(&self) -> u32 {
        self.shape.max_tokens
    }

    /// The rows per rank of the union's middle capture class ([`classes`]).
    pub(crate) const fn beside(&self) -> u32 {
        self.beside
    }

    /// Whether the member holds experts alone: every forward of its buckets is the join.
    pub(crate) const fn experts_only(&self) -> bool {
        self.shape.attention_ranks > 0 && self.shape.rank >= self.shape.attention_ranks
    }

    /// The transport's handle, for the context of the member's forwards; `None` before the
    /// transport exists.
    pub(crate) fn transport(&self) -> Option<ExchangeTransport> {
        self.transport.as_ref().map(Transport::handle)
    }

    /// What the member lends its runner's forwards: the transport's regions and the shape of an
    /// exchange over them, as they stand once the transport exists.
    pub(crate) fn regions(&self) -> Regions {
        Regions {
            shape: self.shape,
            regions: self.regions.clone(),
            fused: self.fused,
        }
    }

    /// Enters the union's next step with `tokens` tokens of this member's own, and returns the
    /// step's capacity once every member has entered: the most tokens any member sends.
    ///
    /// The member first asks the others to join, with a batch of the group, phase, kernel and
    /// policy generations and configuration revision of `step`, the step's own batch, numbered by
    /// the union's step. The call blocks the worker thread until the step is agreed.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for no token or more than the transport holds; what the hook or the
    /// host group fails with; [`Error::Invariant`] for a step no member sends a token in.
    pub(crate) fn enter(&mut self, tokens: u32, step: &Batch) -> Result<u32> {
        if tokens == 0 || tokens > self.shape.max_tokens {
            return Err(Error::Invalid(format!(
                "a step of {tokens} tokens in a union whose members send at most {}",
                self.shape.max_tokens
            )));
        }

        (self.join)(Batch {
            id: BatchId(self.round),
            group: step.group,
            phase: step.phase,
            kernels: step.kernels,
            policy: step.policy,
            revision: step.revision,
            copy_budget: proto::config::DEFAULT_COPY_BUDGET,
            shape: BatchShape::decode(1, Variant::ExpertJoin),
            requests: Vec::new(),
            columns: Columns::default(),
            media: Vec::new(),
            new_slots: Vec::new(),
        })?;
        self.agree(tokens)
    }

    /// Takes part in the step the join batch `batch` asks for, with no token of this member's
    /// own: the step's capacity, or `None` for a step the member already took part in.
    ///
    /// # Errors
    ///
    /// [`Error::Invariant`] for a step past the union's next one; what the host group fails
    /// with.
    pub(crate) fn join(&mut self, batch: &Batch) -> Result<Option<u32>> {
        let asked = batch.id.0;
        if asked < self.round {
            return Ok(None);
        }
        if asked > self.round {
            return Err(Error::Invariant(format!(
                "a join of expert step {asked} reached a member whose union agreed {} steps",
                self.round
            )));
        }
        self.agree(0).map(Some)
    }

    /// Writes `capacity`, the agreed rows a member of the step, on `stream`, ahead of the
    /// step's first launch.
    ///
    /// # Errors
    ///
    /// [`Error::State`] before the transport exists; the package's refusal.
    pub(crate) fn write(&mut self, capacity: u32, stream: &Stream) -> Result<()> {
        self.transport
            .as_mut()
            .ok_or(Error::State("an expert step before the transport exists"))?
            .step(capacity, stream)
            .map_err(|error| Error::Invalid(format!("writing the step's capacity: {error}")))
    }

    /// Gathers the members' token counts of the next step, and closes it.
    fn agree(&mut self, tokens: u32) -> Result<u32> {
        self.group
            .all_gather_fixed(&tokens.to_le_bytes(), &mut self.records)?;

        let capacity = self
            .records
            .as_chunks::<4>()
            .0
            .iter()
            .map(|record| u32::from_le_bytes(*record))
            .max()
            .unwrap_or(0);
        if capacity == 0 || capacity > self.shape.max_tokens {
            return Err(Error::Invariant(format!(
                "expert step {} was agreed at {capacity} rows a member, and a member sends \
                 between one and {}",
                self.round, self.shape.max_tokens
            )));
        }

        self.round += 1;
        Ok(capacity)
    }

    /// The descriptor of the kernel that creates the transport.
    const fn descriptor(&self) -> Descriptor {
        self.shape.descriptor(self.mode)
    }
}

/// The package's gather of `bytes` bytes of every member, in member order
/// (`UniserveGroup::all_gather`).
unsafe extern "C" fn all_gather(
    handle: *mut c_void,
    mine: *const c_void,
    peers: *mut c_void,
    bytes: i64,
) -> i32 {
    // SAFETY: `handle` is the host group the member holds for the transport's life.
    let group = unsafe { &*handle.cast::<HostGroup>() };
    let Ok(bytes) = usize::try_from(bytes) else {
        return 1;
    };

    // SAFETY: the package states `bytes` readable bytes at `mine`.
    let value = unsafe { std::slice::from_raw_parts(mine.cast::<u8>(), bytes) };
    let Ok(gathered) = group.all_gather_bytes(value) else {
        return 1;
    };

    for (member, value) in gathered.iter().enumerate() {
        if value.len() != bytes {
            return 1;
        }

        // SAFETY: the package states `[ranks][bytes]` writable bytes at `peers`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                value.as_ptr(),
                peers.cast::<u8>().add(member * bytes),
                bytes,
            );
        }
    }
    0
}

/// Returns once every member has called it (`UniserveGroup::barrier`).
unsafe extern "C" fn barrier(handle: *mut c_void) -> i32 {
    // SAFETY: `handle` is the host group the member holds for the transport's life.
    let group = unsafe { &*handle.cast::<HostGroup>() };

    i32::from(group.all_gather_fixed(&[], &mut []).is_err())
}
