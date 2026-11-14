//! The scheduler of one data-parallel rank (`specs/scheduler-design.md`, SD).
//!
//! One [`Scheduler`] runs on its own thread: vLLM's scheduler, with one [`Request`] row per
//! request, one `schedule` that visits the running requests and then the waiting queue, and one
//! `update` that applies a batch's output. Everything that reaches its thread arrives as a
//! [`Message`] on one channel, everything it reports leaves as an [`Output`], and the batches it
//! forms go to the [`Executor`] of its worker groups. Every memory a request holds is a block of
//! the buffer pool (`specs/buffer-pool.md`, BP), which the scheduler programs through its queue
//! pair: it writes requests into the submission queue and reads their completions, both in
//! shared memory, and never waits for one (SD §3.1, §4.1; BP §9, §11).
//!
//! This file holds the scheduler's state (SD §2.1, §2.5), the messages in and out (§2.4, §2.5),
//! the executor interface (§2.1), the constructor with the startup checks (§7.1), the step loop
//! (§3.1) and the scheduler's end of the queue pair (§3.1, §3.5).
//!
//! The loop, [`Scheduler::run`], is vLLM's `EngineCore.run_busy_loop`: each pass applies every
//! message that arrived ([`Scheduler::apply`]) and takes one [`Scheduler::step`], which reads the
//! completion queue and the capacity words once, schedules, hands the batches to the executor and
//! writes the step's requests into the submission queue. The loop's one blocking call is the
//! wait on its inbox, taken only after a step that formed no batch, until the next message, the
//! next starvation or drain deadline, the next statistics second, or, while the buffer pool owes
//! answers or a request waits for units, one polling period (`SchedulerConfig::poll_period`): the
//! queue pair runs in poll mode, and no wake-up crosses from the buffer pool.
//!
//! - `submit` takes a request for the buffer pool. A request the workers first apply to their
//!   device block tables (`Release`, `Detach`, `Commit`, `Transfer`, `Replicate`, and the
//!   `Finish` that gives a copy up) waits in `stamping` until every group whose tables may hold
//!   its entries stamped it (BP §9, obligation 2); a request that names positions such a
//!   `Release` frees waits behind it; every other request is held in order. A further `Allocate`
//!   of an owner and class whose `Allocate` is still held raises that one's count (BP §10, F5).
//! - `post` writes the held requests in order while the submission queue has room and the
//!   completion queue has room for a completion of every request outstanding. A request's number
//!   is drawn as it is written, so the numbers rise in the order the buffer pool takes the
//!   requests, which it requires (BP §7, R1).
//! - `next_completion` reads the completion queue, matches each completion to its request by
//!   number, applies what it states of the cache (`cache.rs`), drops what belongs to a forgotten
//!   owner (but a `Committed`, which outlives its owner's `Detach`) or to entries a later stamped
//!   request names (BP §9, obligation 1), and hands every completion that gives entries to the
//!   executor (`Executor::map`) before the scheduler applies it. A `Copy` it hands over is
//!   recorded in `copies` until the executor returns its `Finish`, or the step gives it up at its
//!   bound (SD §4.4).
//! - `read_capacity` reads the capacity words of every region.
//!
//! The scheduling, admission, update, drain, policy, cache and statistics rules are `impl
//! Scheduler` blocks of the sibling files (SD §6).

mod cache;
mod drain;
mod group;
mod memory;
pub mod policy;
mod request;
mod rows;
mod schedule;
mod stats;
#[cfg(test)]
pub(crate) mod tests;
mod update;
mod video;
mod view;

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::batch::{Batch, BatchId, BatchOutput, Capacity, Phase};
use crate::executor::Executor;
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use proto::config::{SchedulerConfig, WorkerGroupConfig};
use proto::report::SchedulerStats;
use proto::request::wire::WeightsInFlight;
use proto::request::{Event, FinishReason, GenerationLimits, MigrationCause, RequestId};
use scheduling::Parameters;
use storage::runtime::{self, Cause, Completion, GroupId, OwnerId, Tier};
use storage::{Key, QueuePair};

pub(crate) use cache::Cached;
pub(crate) use group::WorkerGroup;
/// The packaged policies this tree builds, for the tests of the engine that serves them.
#[cfg(test)]
pub(crate) use policy::tests::packaged;
use policy::{Policy, PolicyError, ceilings};
pub(crate) use request::{Blocks, Request, RequestStatus};
pub(crate) use view::view_of;

/// The first request id of the warm-up requests the engine adds when admission opens: an
/// ordinary request whose id is at or above it has its events dropped (SD §3.2).
pub const WARMUP_REQUEST_IDS: u64 = 1 << 63;

/// The bytes of a record's length word in a ring of a queue pair (`storage::queue`): a ring of
/// `max_record()` bytes of record holds that many more.
const LENGTH_WORD: usize = 8;

/// The bytes of an `Evict` record besides its keys (`storage::runtime`): the request's number,
/// its variant and the length of its key list.
const EVICT_RECORD: usize = 13;

/// The least time the loop sleeps for a starvation deadline: a step that could not judge a
/// starved request (every group's queue full, a paused scheduler) does not spin on it
/// (`S/run.rs:1064-1075`).
const STARVATION_WAKE: Duration = Duration::from_millis(1);

/// The phases whose calls hold a sequence of tokens in a group's KV cache, which the group's
/// longest sequence bounds.
const SEQUENCE_PHASES: [Phase; 4] = [Phase::Prefill, Phase::Decode, Phase::Verify, Phase::Refill];

/// The engine's error: what ends a scheduler, fails a batch or refuses an operation.
#[derive(Debug)]
pub enum Error {
    /// The scheduling policy faulted.
    Policy(PolicyError),
    /// The executor failed a batch or its group.
    Executor(String),
    /// The buffer pool or the queue pair broke the interface, or answered a request the
    /// scheduler chose from its own counts with a refusal.
    Storage(String),
    /// A request or an operation the scheduler refuses as stated.
    InvalidRequest(String),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Policy(error) => write!(formatter, "the scheduling policy faulted: {error}"),
            Self::Executor(message) => write!(formatter, "an executor failed: {message}"),
            Self::Storage(message) => write!(formatter, "the buffer pool failed: {message}"),
            Self::InvalidRequest(message) => write!(formatter, "invalid request: {message}"),
        }
    }
}

impl std::error::Error for Error {}

/// What reaches the scheduler's thread, on its one channel (vLLM's engine-core input queue).
///
/// The buffer pool's completions and capacity words are not messages: the thread reads them
/// from its queue pair at every pass of its loop. Every operation at a batch boundary (a policy
/// or kernel generation switch, sleep and wake, a weight update, a cache flush) is the engine
/// sending `Pause`, doing its work once the reply arrives, and sending `Resume`.
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "a batch and its output travel once per batch, by value, as the message itself"
)]
pub enum Message {
    /// A request from the daemon.
    AddRequest(proto::request::Request),
    /// End a request: `Cancelled`, `Aborted`, or `Stop` when the daemon's decoder matched a stop
    /// string.
    Cancel(RequestId, FinishReason),
    /// The output tokens of a request that the daemon's decoder has passed.
    Acknowledge(RequestId, u32),
    /// The events of a request its consumer took: credit returned.
    Consumed(RequestId, u32),
    /// A submitted batch's output, from the executor, exactly once per batch.
    Output(Batch, Result<BatchOutput, Error>),
    /// From the executor: the requests a group's workers applied to its device block tables,
    /// each with the group's stamp, and the stamped `Finish` of each copy they ran.
    Submit(Vec<runtime::Request>),
    /// From the executor: a job of a worker group outside a batch, a `map` or a `stamp`, failed.
    /// The group's device block tables lack what the scheduler counts on, so it fails at once.
    Failed(Error),
    /// Form no batch; the reply is sent once nothing is in flight.
    Pause(Sender<()>),
    Resume,
    /// Admit nothing; running requests end or migrate for the cause by the deadline; with
    /// `true`, give device memory back by `Resize`. The reply is sent once no request remains.
    Drain(MigrationCause, Instant, bool, Sender<()>),
    /// A policy generation that serves from the next step; the reply is the outcome of its
    /// instantiation, and the serving policy is untouched on failure.
    SetPolicy(Arc<wasmtime::Module>, Parameters, Sender<Result<(), Error>>),
    /// Serve the policy the last `SetPolicy` replaced.
    RollbackPolicy(Sender<Result<(), Error>>),
    /// Drop the policy kept for rollback.
    CommitPolicy,
    /// Evict every cached block the scheduler knows of, the requests it holds continuing as the
    /// [`WeightsInFlight`] says (`Keep`, `Recompute`, `Abort`). With a weight update's
    /// [`Seeds`], what keys the blocks computed from here on, as the new weights' `Capacity`
    /// would state it. The reply refuses seeds whose encoder identities differ between the
    /// scheduler's groups, changing nothing.
    Flush(Option<Seeds>, WeightsInFlight, Sender<Result<(), Error>>),
    /// `Resize { region, units }` at the buffer pool, the region by its number; the reply, sent at
    /// its `Resized`, is the units the region reached, which a shrink and a grow may stop short of.
    Resize(u8, u32, Sender<u32>),
    /// The scheduler's limits, applied in place; the reply refuses a limit above what the
    /// workers hold, which leaves the limits as they were.
    Configure(SchedulerConfig, Sender<Result<(), Error>>),
    /// End every request the scheduler holds, aborted, and stop.
    Shutdown,
}

/// What keys the blocks a scheduler computes, as a weight update's `Flush` renews it for the new
/// weights: the page seed of each cache group in class order (`Capacity::seeds`), the latent seed
/// of stored trajectories (`Capacity::latent_seed`), and per worker group whose ranks encode, the
/// identity of the weights that compute its encoder outputs (`Capacity::encoder_identity`).
pub type Seeds = (Vec<Key>, u64, Vec<(GroupId, String)>);

/// What leaves the scheduler's thread, on its one channel (vLLM's `EngineCoreOutputs`).
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "a request's event and the statistics travel by value as the message itself"
)]
pub enum Output {
    /// Requests' events, to the daemon, each request's in the order they happened: every event
    /// one batch's output produced together, once all of its calls are applied ([`Scheduler`]'s
    /// `update`), and any other event as it happens.
    Events(Vec<(RequestId, Event)>),
    /// The statistics of the last second.
    Stats(SchedulerStats),
    /// The serving policy faulted, with this fault, and the policy it replaced serves again, or
    /// with none a fresh instance of it (SD §5.2 "Faults").
    PolicyRolledBack(String),
    /// A batch computed non-finite logits for these requests.
    NumericalFault(BatchId, Vec<RequestId>),
    /// A region, by its number, and its units after a shrink of a drain.
    Shrunk(u8, u32),
    /// The scheduler stopped.
    Failed(Error),
}

/// The scheduler of one data-parallel rank and its state (SD §2.1, §2.5).
///
/// Invariants: every id in `waiting` or `running` is in `requests` with the matching status; an
/// owner is in `owners` from its `Attach` until its `Detach` is handed to the executor, and no
/// owner is drawn twice; a block is in its region's `order` exactly when it is in `cache`, on
/// that region's tier, with no pin of this scheduler's requests; the completions of `outstanding`
/// always fit the completion queue; a request is in at most one of `stamping`, `held` and
/// `outstanding`; the numbers written into the submission queue rise; all state is touched on the
/// scheduler's thread only.
///
/// A region is named in completions and requests by its number at the buffer pool, and indexed
/// here, in every per-region field, by its place in the session's regions (`Open::regions`),
/// which is how the layout of a group and a class names it too.
pub struct Scheduler {
    config: SchedulerConfig,
    /// The generation limits requests are checked against at their arrival: the model's sizing
    /// facts lowered to what the groups' workers hold (SD §7.1), with the features the groups'
    /// phases serve.
    limits: GenerationLimits,
    /// The engine epoch of the session, which names the private continuation of a token request
    /// that arrives without one (`cache.resume`, SD §3.2).
    epoch: u64,
    requests: HashMap<RequestId, Request>,
    waiting: VecDeque<RequestId>,
    running: Vec<RequestId>,
    groups: Vec<WorkerGroup>,

    /// This scheduler's end of its queue pair with the buffer pool: it writes the submission
    /// queue and reads the completion queue.
    queue: QueuePair,
    /// Requests the submission queue had no room for, in the order they are to be written.
    held: Vec<runtime::Request>,
    /// Requests handed to the executor for the groups' stamps, by the number they were handed
    /// under: each with the stamps it has gathered (in its `after`), the count of groups still to
    /// stamp it, and the requests that must follow it into the submission queue.
    stamping: HashMap<u64, (runtime::Request, u8, Vec<runtime::Request>)>,
    /// Every request written into the submission queue whose completion has not been read, by its
    /// number, with the owner it acts for and the bytes of the completion queue its answer may
    /// take, bounded when it was written (`answer_bytes`): what bounds an answer (its owner's
    /// attachment, the keys `cache` holds) may be gone before the answer is read. A request
    /// answered `Copy` stays, owing what it owed, until the answer to its `Finish`.
    outstanding: HashMap<u64, (Option<OwnerId>, runtime::Request, usize)>,
    /// The counter every number of this queue pair is drawn from: the number `post` writes a
    /// request under, and the number a request waits for its stamps under. Starts at 1, since
    /// the buffer pool takes only numbers above every one it took.
    next_submission: u64,

    /// The classes of the session; a request names a class by its index here.
    classes: Vec<runtime::Class>,
    /// The class of stored trajectories, as the session states it (`Open::trajectory`).
    trajectory: u8,
    /// Per region of the session: the number completions and requests name it by, and its tier
    /// and unit size as the layout states them.
    regions: Vec<(u8, runtime::Region)>,
    /// Where this queue pair's registers lie: per region it uses, by number, the first of its
    /// three capacity words.
    registers: runtime::Registers,
    /// Per region, its units, from the answer to `Open` and from `Resized`.
    capacity: Vec<u32>,
    /// Per region, the units an allocation of this queue pair can take: the buffer pool's word,
    /// less the units of the `Allocate`s held, being stamped or outstanding, and with the units
    /// of `order` beyond the head this scheduler installed.
    available: Vec<u32>,
    /// Per region, the units of retired blocks the buffer pool reclaims once the batches that may
    /// read them have ended.
    retiring: Vec<u32>,
    /// Every cached block the scheduler knows of, by [`Cached::id`].
    cache: HashMap<u64, Cached>,
    /// Per region, the cached blocks no request of this scheduler holds, by last use and id: the
    /// eviction order it installs.
    order: Vec<BTreeSet<(u64, u64)>>,
    /// Per region, the units of the blocks in `order`.
    ordered: Vec<u32>,
    /// Per region, the numbers of the keys of the last `Victims` this scheduler submitted there:
    /// the head of `order` it installed.
    installed: Vec<Vec<u64>>,
    /// The blocks a request of this scheduler gave up before the buffer pool answered their
    /// `Commit`, by [`Cached::id`] of the key: the last use each enters the eviction order with
    /// once its `Committed` is read, as `unpin` ranked it among its request's blocks then. An
    /// entry leaves when its `Commit` is answered.
    given_up: HashMap<u64, u64>,
    /// The request each live owner is for.
    owners: HashMap<OwnerId, RequestId>,
    /// The counter owners are drawn from: the scheduler's data-parallel rank in the top 16 bits.
    next_owner: u64,
    /// Per cache group, in class order, the seed the key chains of its blocks are rooted at (SD
    /// §4.3); a request's `CachePolicy::state_seeds` override it. Empty until stated, which roots
    /// every chain at the zero key, the seed of an empty identity. A weight update's `Flush`
    /// replaces it, so that no key reaches a block the previous weights computed.
    seeds: Vec<Key>,
    /// The identity of the weights that compute the scheduler's encoder outputs, which keys them
    /// (`encoder_output`, SD §4.3): what its encoding groups state (`Capacity::encoder_identity`),
    /// empty when none does. A weight update's `Flush` replaces it with `seeds`.
    encoder_identity: String,
    /// The seed of the keys of stored trajectories (`latent_content`, SD §4.3): what its
    /// denoising groups state (`Capacity::latent_seed`), 0 when none does. A weight update's
    /// `Flush` replaces it with `seeds`.
    latent_seed: u64,
    /// Per region, the `Store`s to the host into it that stored a block (answered `Stored` after
    /// their copy ran): a block whose own `Store` stored nothing waits for this count, among
    /// others, to move before it is offered to the host again (`Cached::refused`, SD §4.4).
    stores: Vec<u64>,
    /// The blocks, by [`Cached::id`], whose last `Store` to the host stored nothing: each step's
    /// `offload` offers each again once something that could change that answer has happened
    /// (SD §4.4 "The write policy"). A block leaves when it is offered again, has a host copy, or
    /// has no device copy.
    unstored: HashSet<u64>,
    /// Running requests whose next call waits for units of a region since their last shortage:
    /// per request, the region (by index) and the units its `available` must show before the
    /// request asks again (SD §3.6). A request listed here takes no further victims.
    short: HashMap<RequestId, (usize, u32)>,
    /// The copies the groups' workers hold, by the number of the request answered `Copy` (a
    /// `Load`, a `Clone`, a `Store` to the host, a `Save` or a `Restore`): the group whose
    /// workers run it, and the moment the scheduler gives it up (SD §4.4 "The bound of a
    /// copy"). A copy enters when its `Copy` is handed to the executor and leaves when the
    /// executor returns its `Finish` or the scheduler gives it up: a group with an entry is sent
    /// an empty batch when the step forms none for it (SD §2.1).
    copies: HashMap<u64, (GroupId, Instant)>,
    /// Completions that give entries, read before the `Attached` that states the slot row they
    /// are written under: by the owner of the attachment whose slot they wait for, each with the
    /// group it is handed to and the number, request and completion `Executor::map` takes. The
    /// buffer pool may answer an admission's requests in any order, and a worker writes a
    /// completion's entries under the slot it is handed (SD §2.1 `Executor`, §3.1). They are
    /// handed over when that `Attached` is read, and dropped with the owners they concern.
    unmapped: HashMap<OwnerId, Vec<(GroupId, runtime::RequestId, runtime::Request, Completion)>>,
    /// Per region, whether a request whose end gives units back was answered since the region's
    /// last `Resize` was submitted ([`Scheduler::gives_back`]). The buffer pool writes a
    /// completion before the capacity words its request changed, so the words read with such an
    /// answer may not show its units yet; a drain that gives memory back owes such a region one
    /// more `Resize`, whose answer is read with words that do (SD §3.8).
    released: Vec<bool>,

    policy: Policy,
    executor: Box<dyn Executor>,
    output: Sender<Output>,
    /// While `update` applies a batch's output, the events it has produced, which leave together
    /// as one `Output::Events` once it ends; `None` outside `update`, where each event leaves as
    /// it happens. Every other output leaves after the events produced before it.
    outbox: Option<Vec<(RequestId, Event)>>,
    stats: SchedulerStats,
    /// When `stats` started counting: the scheduler's start, then its last `Output::Stats`.
    published: Instant,
    /// The kernel and policy generation batches are assembled for.
    generation: Arc<AtomicU64>,
    paused: bool,
    /// Whether the scheduler stopped, on a fault (`fail`) or at `Shutdown`: it takes no message
    /// and no step from then on, and its loop ends.
    stopped: bool,
    /// The replies owed to `Pause` once nothing is in flight.
    boundary: Vec<Sender<()>>,
    /// The reply, cause, deadline and whether memory is given back, of a drain under way.
    draining: Option<(Sender<()>, MigrationCause, Instant, bool)>,
    /// The region, by number, of each `Resize` outstanding and its reply, owed at `Resized` with
    /// the units the region reached.
    resizing: Vec<(u8, Sender<u32>)>,
    /// The `List`s submitted, and those answered, which the buffer pool answers in the order it
    /// takes them: a resumed request waits for one asked after it arrived (SD §4.4, §4.5).
    lists_asked: u64,
    lists_answered: u64,
    next_batch: u64,
    step: u64,
}

impl Scheduler {
    /// A scheduler of the session the engine opened with `layout` and that `opened` answered:
    /// the `scheduler`-th of `layout.schedulers`, which is its data-parallel rank, on its end of
    /// its queue pair. `groups` are its worker groups' configurations and the capacities their
    /// workers stated, in the order `layout.schedulers` lists its groups. `limits` are the
    /// generation limits the engine resolved for the deployment; their features are the ones the
    /// groups' phases serve.
    ///
    /// The scheduler's limits are `config`'s, lowered to what the workers hold (SD §7.1): the
    /// running requests to the slots the groups hold for them, a step's tokens to the widest
    /// prefill bucket the workers captured and its calls to their largest bucket's rows. Its
    /// policy is `module`, constructed from the package's `parameters` and instantiated on the
    /// scheduler's own snapshot, as every later generation's is (SD §3.9, §5.2): the groups and
    /// the phases they serve, with no request and no cached block; its limits are checked against
    /// what the workers captured. Its first request to the buffer pool is `List`, which teaches
    /// it the cached blocks its session can reach (SD §4.5).
    ///
    /// # Errors
    ///
    /// [`Error::Storage`] when the layout and its answer do not state this scheduler: no such
    /// scheduler, other groups than `groups`, or regions and registers that do not match.
    /// [`Error::InvalidRequest`] for a deployment the startup checks refuse ([`resolved`]).
    /// [`Error::Policy`] when the policy does not instantiate, or declares limits outside the
    /// ceilings of the workers' captures.
    #[expect(
        clippy::too_many_arguments,
        reason = "the scheduler's parts come from five owners: the configuration, the session \
                  with the buffer pool, the policy package, the executor and the engine's \
                  channels"
    )]
    pub fn new(
        config: SchedulerConfig,
        limits: GenerationLimits,
        layout: &runtime::Open,
        opened: &runtime::Opened,
        scheduler: usize,
        queue: QueuePair,
        groups: Vec<(WorkerGroupConfig, Capacity)>,
        module: Arc<wasmtime::Module>,
        parameters: Parameters,
        executor: Box<dyn Executor>,
        output: Sender<Output>,
        generation: Arc<AtomicU64>,
    ) -> Result<Self, Error> {
        let stated = layout.schedulers.get(scheduler).ok_or_else(|| {
            Error::Storage(format!("the session states no scheduler {scheduler}"))
        })?;
        let registers = opened.registers.get(scheduler).cloned().ok_or_else(|| {
            Error::Storage(format!(
                "the session states no registers of scheduler {scheduler}"
            ))
        })?;
        if stated.len() != groups.len() {
            return Err(Error::Storage(format!(
                "scheduler {scheduler} has {} groups and the session states {}",
                groups.len(),
                stated.len()
            )));
        }
        if opened.regions.len() != layout.regions.len() {
            return Err(Error::Storage(format!(
                "the session states {} regions and its answer {}",
                layout.regions.len(),
                opened.regions.len()
            )));
        }

        // Each worker group with the layout the session states for it.
        let mut workers = Vec::with_capacity(groups.len());
        for ((config, capacity), id) in groups.into_iter().zip(stated) {
            let group = layout
                .groups
                .iter()
                .find(|group| group.id == *id)
                .ok_or_else(|| Error::Storage(format!("the session states no group {}", id.0)))?;
            workers.push(WorkerGroup::new(config, capacity, group.clone()));
        }
        let config = resolved(config, &workers)?;

        // A scheduler that holds no request and knows no cached block yet is told its groups.
        let bounds = ceilings(workers.iter().map(|worker| &worker.capacity));
        let snapshot = view::snapshot_of(&workers);
        let parameters = policy::budgeted(parameters, &config);
        let policy = Policy::new(module, parameters, snapshot, &bounds).map_err(Error::Policy)?;

        // The page seeds every group of the scheduler states for its cache groups, one per class
        // in class order: one key chain serves all of them, so groups that state seeds must state
        // the same. A group without a KV cache states none. Likewise the identity that keys its
        // encoder outputs and the seed that keys its stored trajectories: a request's outputs
        // and trajectory are found under one key whichever of its groups computed them. A group
        // that encodes nothing, or denoises nothing, states none.
        let stated = || workers.iter().map(|worker| &worker.capacity);
        let seeds = agreed(stated().map(|capacity| &capacity.seeds))
            .ok_or_else(|| differ("page seeds for their cache groups"))?;
        let encoder_identity = agreed(stated().map(|capacity| &capacity.encoder_identity))
            .ok_or_else(|| differ(ENCODER_IDENTITIES))?;
        let latent_seed = agreed(stated().map(|capacity| &capacity.latent_seed))
            .ok_or_else(|| differ("latent seeds for their stored trajectories"))?;

        let regions: Vec<(u8, runtime::Region)> = opened
            .regions
            .iter()
            .zip(&layout.regions)
            .map(|(&(number, _), region)| (number, region.clone()))
            .collect();
        let capacity: Vec<u32> = opened.regions.iter().map(|&(_, units)| units).collect();
        let count = regions.len();

        // Owner numbers carry the data-parallel rank in their top 16 bits, so the schedulers of
        // one session never draw the same owner; the buffer pool takes an owner number only
        // above every one it took, so neither counter starts at 0.
        let rank = u64::try_from(scheduler).unwrap_or(u64::MAX) & 0xffff;
        let limits = GenerationLimits {
            features: request::served_features(&workers),
            ..limits
        };
        let mut scheduler = Self {
            config,
            limits,
            epoch: layout.epoch,
            requests: HashMap::new(),
            waiting: VecDeque::new(),
            running: Vec::new(),
            groups: workers,
            queue,
            held: Vec::new(),
            stamping: HashMap::new(),
            outstanding: HashMap::new(),
            next_submission: 1,
            classes: layout.classes.clone(),
            trajectory: layout.trajectory,
            regions,
            registers,
            available: capacity.clone(),
            capacity,
            retiring: vec![0; count],
            cache: HashMap::new(),
            order: vec![BTreeSet::new(); count],
            ordered: vec![0; count],
            installed: vec![Vec::new(); count],
            given_up: HashMap::new(),
            owners: HashMap::new(),
            next_owner: (rank << 48) + 1,
            seeds,
            encoder_identity,
            latent_seed,
            short: HashMap::new(),
            copies: HashMap::new(),
            unmapped: HashMap::new(),
            released: vec![false; count],
            stores: vec![0; count],
            unstored: HashSet::new(),
            policy,
            executor,
            output,
            outbox: None,
            stats: stats::empty(),
            published: Instant::now(),
            generation,
            paused: false,
            stopped: false,
            boundary: Vec::new(),
            draining: None,
            resizing: Vec::new(),
            lists_asked: 0,
            lists_answered: 0,
            next_batch: 0,
            step: 0,
        };

        // Written at the first step; nothing is scheduled before its `Listing` but what the
        // scheduler commits itself, which `cache` holds exactly.
        scheduler.submit(runtime::Request::List);
        Ok(scheduler)
    }

    /// The generation limits requests are checked against at their arrival: the engine's, with
    /// the features this scheduler's groups serve.
    pub(crate) fn limits(&self) -> &GenerationLimits {
        &self.limits
    }

    /// The requests this scheduler runs at once: the deployment's `max_running_requests`, within
    /// the slots its groups hold, until a `Configure` changes it.
    pub(crate) fn max_running_requests(&self) -> usize {
        self.config.max_running_requests
    }
}

/// `config` with its limits lowered to what `groups`' workers hold, once the deployment passes
/// the startup checks (SD §7.1; `S/core.rs:185-236`, `S/submission.rs:93-105`,
/// `S/run.rs:108-162`, `S/worker/codecs.rs`).
///
/// The checks: at least one group; every group that runs calls of a token sequence states its
/// longest sequence; and, when a group encodes video, every group that decodes video has one that
/// encodes its units on the hosts that decoded them ([`group::encodes_on_host`]). The limits:
/// `max_running_requests` within `1..=` the slots ([`slots`]), which give each running request
/// the slot its state lives in, `max_num_batched_tokens` within `1..=` the widest prefill bucket's tokens, and
/// `max_batch` at most the largest bucket's rows ([`ceilings`]); a bound no group states binds
/// nothing.
///
/// # Errors
///
/// [`Error::InvalidRequest`] naming the first check the deployment fails.
fn resolved(mut config: SchedulerConfig, groups: &[WorkerGroup]) -> Result<SchedulerConfig, Error> {
    if groups.is_empty() {
        return Err(Error::InvalidRequest(
            "the workers state no worker group to schedule on".into(),
        ));
    }

    let sequences = |group: &&WorkerGroup| SEQUENCE_PHASES.iter().any(|&phase| group.serves(phase));
    if let Some(group) = groups
        .iter()
        .filter(sequences)
        .find(|group| group.capacity.max_model_len == 0)
    {
        return Err(Error::InvalidRequest(format!(
            "worker group {} runs token sequences and states no longest sequence",
            group.config.id.0
        )));
    }

    // A decoded video unit is a host product its encoder reads in place, so a decoder whose units
    // no encoder can read on their hosts could never finish a request.
    let encoders: Vec<&WorkerGroup> = groups
        .iter()
        .filter(|group| group.serves(Phase::VideoEncode))
        .collect();
    let stranded = groups
        .iter()
        .filter(|group| group.serves(Phase::VideoDecode))
        .find(|decoder| {
            !encoders.is_empty()
                && !encoders
                    .iter()
                    .any(|encoder| group::encodes_on_host(&decoder.config, &encoder.config))
        });
    if let Some(decoder) = stranded {
        return Err(Error::InvalidRequest(format!(
            "worker group {} decodes video units that no video encoding group encodes on the \
             hosts that decoded them",
            decoder.config.id.0
        )));
    }

    // Every running request holds a slot of a group of each phase it runs, whose state (a KV
    // cache's pages, a trajectory) lives in it: the groups' slots bound the running requests, and
    // a rank holds fewer slots than the deployment states when its share of its GPU holds no
    // more (`specs/engine-worker-port.md` §1, `request_slots`).
    if let Some(slots) = slots(groups) {
        let slots = usize::try_from(slots).unwrap_or(usize::MAX);
        config.max_running_requests = config.max_running_requests.min(slots).max(1);
    }

    let ceilings = ceilings(groups.iter().map(|group| &group.capacity));
    let tokens = usize::try_from(ceilings.max_num_batched_tokens).unwrap_or(usize::MAX);
    config.max_num_batched_tokens = config.max_num_batched_tokens.clamp(1, tokens.max(1));
    if let Some(rows) = NonZeroUsize::new(ceilings.max_batch) {
        config.max_batch = config.max_batch.min(rows);
    }
    Ok(config)
}

/// The one value of `stated`, the statements of a scheduler's groups, that every group stating
/// one (a value other than the empty default) states; the empty default when none does, `None`
/// when two differ.
/// What a refusal of the identities of the weights that compute the encoder outputs names.
const ENCODER_IDENTITIES: &str = "identities of the weights that compute their encoder outputs";

/// The refusal of what the scheduler's groups state differently, `what` naming it.
fn differ(what: &str) -> Error {
    Error::InvalidRequest(format!("the scheduler's groups state different {what}"))
}

fn agreed<'a, T>(stated: impl IntoIterator<Item = &'a T>) -> Option<T>
where
    T: Clone + Default + PartialEq + 'a,
{
    let empty = T::default();
    let mut stated = stated.into_iter().filter(|value| **value != empty);
    let first = stated.next().cloned().unwrap_or_default();
    stated.all(|other| *other == first).then_some(first)
}

/// The requests `groups` hold slots for at once: per phase some group serves, the slots of the
/// groups that serve it together (replicas of a stage each hold their own), and the least of
/// those over the phases, since a request holds a slot in a group of every phase it runs. `None`
/// when no group serves a phase.
fn slots(groups: &[WorkerGroup]) -> Option<u32> {
    Phase::ALL
        .into_iter()
        .filter_map(|phase| {
            groups
                .iter()
                .filter(|group| group.serves(phase))
                .map(|group| group.capacity.slots)
                .reduce(u32::saturating_add)
        })
        .min()
}

/// The step loop and the messages it applies (SD §3.1, §2.4): vLLM's `EngineCore.run_busy_loop`,
/// `_handle_client_request` and `step_with_batch_queue`.
impl Scheduler {
    /// Runs the scheduler on the calling thread until it stops: at `Shutdown`, at a fault, which
    /// it reports as `Output::Failed`, or once no sender of `inbox` is left, which is a shutdown.
    ///
    /// Each pass applies every message that arrived ([`Scheduler::apply`]) and takes one
    /// [`Scheduler::step`]. It blocks on `inbox` only after a step that formed no batch: a formed
    /// batch may have left room for another (a pipelined decode), which the next pass finds
    /// without waiting. It then waits until the next message or the next deadline of
    /// [`Scheduler::next_deadline`].
    pub(crate) fn run(mut self, inbox: Receiver<Message>) {
        let mut formed = false;
        while !self.stopped {
            let first = if formed {
                match inbox.try_recv() {
                    Ok(message) => Some(message),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => return self.shutdown(),
                }
            } else {
                match inbox.recv_deadline(self.next_deadline()) {
                    Ok(message) => Some(message),
                    Err(RecvTimeoutError::Timeout) => None,
                    Err(RecvTimeoutError::Disconnected) => return self.shutdown(),
                }
            };

            for message in first.into_iter().chain(inbox.try_iter()) {
                self.apply(message);
                if self.stopped {
                    return;
                }
            }
            formed = self.step();
        }
    }

    /// Applies one message of the inbox, as [`Scheduler::run`] does with each. A stopped scheduler
    /// takes none.
    ///
    /// A reply whose receiver has gone is dropped: its sender stopped waiting for it.
    pub fn apply(&mut self, message: Message) {
        if self.stopped {
            return;
        }

        match message {
            Message::AddRequest(input) => self.add_request(input),
            Message::Cancel(id, reason) => self.cancel(id, reason),
            Message::Acknowledge(id, tokens) => self.acknowledge(id, tokens),
            Message::Consumed(id, events) => self.consumed(id, events),
            // The requests of each completed batch advance.
            Message::Output(batch, output) => self.update(batch, output),
            // The executor's stamped requests join the step's requests, behind those held.
            Message::Submit(requests) => self.stamped(requests),
            Message::Failed(error) => self.fail(error),
            Message::Pause(reply) => {
                self.paused = true;
                self.boundary.push(reply);
            }
            Message::Resume => {
                self.paused = false;
                self.draining = None;
            }
            Message::Drain(cause, deadline, shrink, reply) => {
                self.drain(cause, deadline, shrink, reply);
            }
            Message::SetPolicy(module, parameters, reply) => {
                let _ = reply.send(self.set_policy(module, parameters));
            }
            Message::RollbackPolicy(reply) => {
                let snapshot = self.snapshot();
                let _ = reply.send(self.policy.rollback(snapshot));
            }
            Message::CommitPolicy => self.policy.commit(),
            Message::Flush(seeds, in_flight, reply) => {
                let _ = reply.send(self.flush(seeds, in_flight));
            }
            Message::Resize(region, units, reply) => {
                self.submit(runtime::Request::Resize { region, units });
                self.resizing.push((region, reply));
            }
            Message::Configure(config, reply) => {
                let _ = reply.send(self.configure(config));
            }
            Message::Shutdown => self.shutdown(),
        }
    }

    /// One step after the messages, which waits for nothing:
    ///
    /// 1. every completion the buffer pool wrote is applied as it is read, the capacity words of
    ///    every region are read (SD §4.1), and every copy past its bound is given up (§4.4);
    /// 2. unless the scheduler is paused, `schedule` assembles the batches that fit now, and each
    ///    is handed to the executor; the eviction order of each region whose installed head no
    ///    longer covers what the step's requests take is installed, ahead of them;
    /// 3. the step's requests are written into the submission queue as far as it and the
    ///    completion queue have room;
    /// 4. a drain's deadline ends what it ends, `Pause` and `Drain` are answered once due
    ///    ([`Scheduler::reply_at_boundary`]), and the statistics leave once a second has passed.
    ///
    /// True when it formed a batch. A stopped scheduler takes no step.
    pub fn step(&mut self) -> bool {
        if self.stopped {
            return false;
        }
        let _range = crate::profiling::ranges::range(c"uniserve.scheduler.step", None);

        // The running requests before the completions, of which the step's trace states the
        // admissions they end (`schedule::STEPS`).
        let running = self.running.len();
        loop {
            let (id, owner, request, completion) = match self.next_completion() {
                Ok(Some(entry)) => entry,
                Ok(None) => break,
                Err(error) => {
                    self.fail(error);
                    return false;
                }
            };
            self.answered(id, owner, request, completion);
            if self.stopped {
                return false;
            }
        }
        self.read_capacity();
        let now = Instant::now();
        self.give_up_late_copies(now);
        // A prefill whose decode groups were all full when it finished transfers once one has
        // room.
        self.start_transfers();

        let admitted = self.running.len().saturating_sub(running);
        let batches = if self.paused {
            Vec::new()
        } else {
            self.step += 1;
            self.schedule()
        };
        if self.stopped {
            return false;
        }
        self.trace_step(now, admitted, &batches);
        // The eviction order goes ahead of the step's requests: `available` counts the units of
        // the order beyond the installed head, which the step's requests may take only once the
        // buffer pool has taken a `Victims` that names them (SD §4.5).
        self.install_victims();
        let formed = !batches.is_empty();
        for batch in batches {
            if let Some(group) = self.group_index(batch.group) {
                self.groups[group].in_flight += 1;
            }
            self.count_submitted(&batch, now);
            self.executor.submit(batch);
        }

        if let Err(error) = self.post() {
            self.fail(error);
            return false;
        }
        self.drain_due(now);
        self.reply_at_boundary();
        self.publish_stats(now);
        formed
    }

    /// Until when the loop may block on its inbox after a step that formed no batch: the next
    /// statistics second; the moment the first running request passes its starvation bound (the
    /// policy's `timeout` from its last progress), at least [`STARVATION_WAKE`] away; a drain's
    /// deadline still ahead; and one polling period from now while [`Scheduler::polls`].
    fn next_deadline(&self) -> Instant {
        let now = Instant::now();
        let mut deadline = self.stats_due();

        // A request waiting for the buffer pool's transfer is exempt from the bound.
        let timeout = Duration::from_nanos(self.policy.limits.timeout.0);
        let starves = self
            .running
            .iter()
            .filter_map(|id| self.requests.get(id))
            .filter(|request| request.status == RequestStatus::Running)
            .filter_map(|request| request.last_progress.checked_add(timeout))
            .min();
        if let Some(starves) = starves {
            deadline = deadline.min(starves.max(now + STARVATION_WAKE));
        }

        // A deadline already passed was met by the step that just ran.
        if let Some((_, _, due, _)) = &self.draining
            && *due > now
        {
            deadline = deadline.min(*due);
        }

        if self.polls() {
            deadline = deadline.min(now + self.config.poll_period);
        }
        deadline
    }

    /// Whether the loop polls the queue pair (BP §9, poll mode): something changes there without
    /// a message to wake the loop. That is the case while the buffer pool owes answers to
    /// requests written, while requests wait for room in the submission queue, and, unless the
    /// scheduler is paused, while a request may wait for units of a region: one in the waiting
    /// queue, whose admission waits for `available`, or a running request with no call in flight,
    /// which may wait for the units its next call needs.
    fn polls(&self) -> bool {
        let units = !self.paused
            && (!self.waiting.is_empty()
                || self
                    .running
                    .iter()
                    .filter_map(|id| self.requests.get(id))
                    .any(Request::idle));
        !self.outstanding.is_empty() || !self.held.is_empty() || units
    }

    /// Answers `Pause` once the scheduler is paused and no group has a batch in flight, and a
    /// drain once no request remains (SD §3.8): with `shrink`, also once no request of the
    /// scheduler that gives units back ([`Scheduler::gives_back`]) is held, being stamped or
    /// outstanding, and every device region answered a `Resize` submitted after the last such
    /// request was answered, so that the capacity words read with that answer show what it gave
    /// back (the shrink submits one whenever a region has none outstanding and something to give,
    /// `drain.rs`). A migratable request's last blocks, committed as it leaves and stored to the
    /// host by the write policy, are therefore taken by the drain's shrink once their stores end,
    /// and a drain to sleep leaves its regions with no unit of theirs.
    ///
    /// A drained scheduler then forms no batch until `Resume`, as if paused: it admits nothing,
    /// as the drain stated, and has nothing else to run.
    fn reply_at_boundary(&mut self) {
        let idle = self.groups.iter().all(|group| group.in_flight == 0);
        if self.paused && idle {
            for reply in self.boundary.drain(..) {
                let _ = reply.send(());
            }
        }

        let Some((_, _, _, shrink)) = &self.draining else {
            return;
        };
        let given_back = !*shrink || (!self.giving_back() && !self.released.contains(&true));
        if self.requests.is_empty()
            && given_back
            && let Some((reply, ..)) = self.draining.take()
        {
            self.paused = true;
            let _ = reply.send(());
        }
    }

    /// Whether the end of `request` may give units of the scheduler's regions back: what its
    /// answer, or the end of its copies, releases or leaves unpinned and evictable. A `Detach`,
    /// a `Release` and an `Evict`; a `Commit`, whose block is shared and unpinned once its owner
    /// has left (a migratable request's last blocks); a `Transfer`, whose source blocks it holds
    /// until it ends; and every copy the engine runs, whose blocks the buffer pool holds until
    /// its `Finish` (a `Store` to the host reads its block's device copy). A `Store` to the
    /// remote store and a `Prefetch` read and fill the host tier alone.
    fn gives_back(request: &runtime::Request) -> bool {
        match request {
            runtime::Request::Detach { .. }
            | runtime::Request::Release { .. }
            | runtime::Request::Evict { .. }
            | runtime::Request::Commit { .. }
            | runtime::Request::Transfer { .. }
            | runtime::Request::Load { .. }
            | runtime::Request::Clone { .. }
            | runtime::Request::Save { .. }
            | runtime::Request::Restore { .. }
            | runtime::Request::Finish { .. }
            | runtime::Request::Replicate { .. } => true,
            runtime::Request::Store { tier, .. } => *tier == Tier::Host,
            runtime::Request::Attach { .. }
            | runtime::Request::Allocate { .. }
            | runtime::Request::Pin { .. }
            | runtime::Request::Victims { .. }
            | runtime::Request::Prefetch { .. }
            | runtime::Request::List
            | runtime::Request::Resize { .. } => false,
        }
    }

    /// Whether a request of this scheduler that gives units back ([`Self::gives_back`]) or a
    /// `Resize` is still to be taken or answered: held, being stamped, waiting behind a request
    /// being stamped, or outstanding.
    fn giving_back(&self) -> bool {
        let pending = |request: &runtime::Request| {
            Self::gives_back(request) || matches!(request, runtime::Request::Resize { .. })
        };
        let stamping = self
            .stamping
            .values()
            .any(|(request, _, following)| pending(request) || following.iter().any(pending));
        stamping
            || self.held.iter().any(pending)
            || self
                .outstanding
                .values()
                .any(|(_, request, _)| pending(request))
    }

    /// `Flush` (SD §2.4): the requests the scheduler holds continue as `in_flight` says, then
    /// every cached block the scheduler knows of is evicted, and with a weight update's `seeds`
    /// what keys the blocks computed from here on is replaced: the cache groups' page seeds, the
    /// latent seed, and the encoder identity, which the identities `seeds` states for this
    /// scheduler's groups agree on as their `Capacity` would (empty when none of them encodes).
    ///
    /// The requests are dealt with first, under the seeds their state was computed with:
    ///
    /// - `Keep`: every request continues with what it holds. Under a weight update its state
    ///   follows neither the old weights nor the new ones, so a token request commits nothing
    ///   from here on, and keeps the seeds its keys were rooted at unless it states its own.
    /// - `Recompute`: every request that holds blocks, admitted or being admitted, ends as
    ///   `Migratable { Drain }`, and the daemon resubmits it to be computed again; a video request
    ///   whose denoising is fully scheduled runs to its end. A request still queued waits.
    /// - `Abort`: every request ends, aborted.
    ///
    /// `Evict` names every key of `cache`, as many per record as a record holds; the buffer pool
    /// evicts those that no request pins, and `cache` keeps the keys its `Evicted` does not list.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRequest`] when `seeds` states different encoder identities for two of the
    /// scheduler's groups; nothing changes then.
    fn flush(&mut self, seeds: Option<Seeds>, in_flight: WeightsInFlight) -> Result<(), Error> {
        let seeds = match seeds {
            Some((pages, latent, encoders)) => {
                let ours = self.groups.iter().filter_map(|group| {
                    encoders
                        .iter()
                        .find(|(id, _)| *id == group.id())
                        .map(|(_, identity)| identity)
                });
                let identity = agreed(ours).ok_or_else(|| differ(ENCODER_IDENTITIES))?;
                Some((pages, latent, identity))
            }
            None => None,
        };

        let ids: Vec<RequestId> = self.requests.keys().copied().collect();
        match in_flight {
            WeightsInFlight::Keep => {
                if seeds.is_some() {
                    self.keep_lineage();
                }
            }
            WeightsInFlight::Recompute => {
                for id in ids {
                    let Some(request) = self.requests.get(&id) else {
                        continue;
                    };
                    let holds = matches!(
                        request.status,
                        RequestStatus::Running | RequestStatus::WaitingForRemoteKvs
                    );
                    if holds && !request.runs_to_end(&self.groups) {
                        self.finish(id, update::migratable(MigrationCause::Drain));
                    }
                }
            }
            WeightsInFlight::Abort => {
                for id in ids {
                    self.cancel(id, FinishReason::Aborted);
                }
            }
        }

        let fits = self.queue.max_record().saturating_sub(EVICT_RECORD) / size_of::<Key>();
        let keys: Vec<Key> = self.cache.values().map(|row| row.key).collect();
        for keys in keys.chunks(fits.max(1)) {
            self.submit(runtime::Request::Evict {
                keys: keys.to_vec(),
            });
        }

        if let Some((pages, latent, identity)) = seeds {
            self.seeds = pages;
            self.latent_seed = latent;
            self.encoder_identity = identity;
        }
        Ok(())
    }

    /// Under a weight update that keeps the requests in flight: each token request commits no
    /// block from here on, and its keys stay rooted at the seeds before the update when it states
    /// none of its own (`S/cache_storage.rs:288-326`).
    fn keep_lineage(&mut self) {
        let groups = self
            .groups
            .iter()
            .map(|group| group.capacity.cache_groups.len())
            .max()
            .unwrap_or(0);
        let seeds: Vec<[u8; 32]> = if self.seeds.is_empty() {
            vec![[0; 32]; groups]
        } else {
            self.seeds.iter().map(|seed| seed.0).collect()
        };

        for request in self.requests.values_mut() {
            if let proto::request::Request::Generation(input) = &mut request.input {
                if input.cache.state_seeds.is_empty() {
                    input.cache.state_seeds.clone_from(&seeds);
                }
                input.cache.write = false;
            }
        }
    }

    /// `Configure` (SD §2.4): `config`'s limits serve from the next step on, but the epoch's own,
    /// which the workers were started for: its drafts per verification and the graph sizes they
    /// captured. The policy's limits, which budget a step, change with the policy (`SetPolicy`).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRequest`] naming the first limit above what the workers hold: running
    /// requests beyond their slots, a step's tokens beyond the widest prefill bucket, its calls
    /// beyond the largest bucket's rows (`S/run.rs:287-347`). The limits are then unchanged.
    fn configure(&mut self, config: SchedulerConfig) -> Result<(), Error> {
        let ceilings = ceilings(self.groups.iter().map(|group| &group.capacity));
        let held = |value: Option<u32>| value.map_or(usize::MAX, |value| value as usize);
        let limits = [
            (
                "max_running_requests",
                config.max_running_requests,
                held(slots(&self.groups)),
            ),
            (
                "max_num_batched_tokens",
                config.max_num_batched_tokens,
                held(Some(ceilings.max_num_batched_tokens)),
            ),
            ("max_batch", config.max_batch.get(), ceilings.max_batch),
        ];
        if let Some((name, value, held)) = limits.into_iter().find(|&(_, value, held)| value > held)
        {
            return Err(Error::InvalidRequest(format!(
                "scheduler_config.{name} = {value} exceeds the {held} the workers were started \
                 for; an epoch change starts workers for it"
            )));
        }

        self.config = SchedulerConfig {
            max_num_batched_tokens: config.max_num_batched_tokens.max(1),
            max_running_requests: config.max_running_requests.max(1),
            draft_tokens: self.config.draft_tokens,
            graph_tiers: std::mem::take(&mut self.config.graph_tiers),
            ..config
        };
        Ok(())
    }

    /// Stops the scheduler on a fault it cannot recover from (SD §3.7). Every request whose
    /// terminal has not been sent leaves as `Migratable { Loss }`, which the daemon resubmits; a
    /// `Finished` held for the daemon's decoder is sent as it is, since that generation ended.
    /// Each request gives up what it holds as far as the workers still take it.
    /// `Output::Failed` tells the engine, and the loop ends. A failed executor cannot stamp its
    /// group's `Detach`: the buffer pool detaches the epoch's owners when its session closes
    /// (BP §12.4). Called once; later faults are dropped.
    pub(super) fn fail(&mut self, error: Error) {
        if self.stopped {
            return;
        }
        self.stopped = true;

        let ids: Vec<RequestId> = self.requests.keys().copied().collect();
        for id in ids {
            let Some(request) = self.requests.get_mut(&id) else {
                continue;
            };
            let held = request.terminal.take();
            let sent = request.status == RequestStatus::Finished && held.is_none();
            request.status = RequestStatus::Finished;

            match held {
                Some(finished @ Event::Finished { .. }) => self.send(id, finished),
                _ if !sent => self.send(id, update::migratable(MigrationCause::Loss)),
                _ => {}
            }
            self.free(id);
        }

        // An engine that no longer listens has stopped this scheduler's epoch already.
        self.emit(Output::Failed(error));
    }

    /// `Shutdown`: every request the scheduler holds ends, aborted, at once, and the scheduler
    /// stops (`S/control.rs:73-102`). A terminal still held, behind calls in flight or the
    /// daemon's decoder, is sent now: nothing more runs.
    fn shutdown(&mut self) {
        let ids: Vec<RequestId> = self.requests.keys().copied().collect();
        for id in ids {
            self.cancel(id, FinishReason::Aborted);
            if let Some(terminal) = self
                .requests
                .get_mut(&id)
                .and_then(|request| request.terminal.take())
            {
                self.send(id, terminal);
            }
        }
        self.stopped = true;
    }
}

/// The scheduler's end of its queue pair (SD §3.1, §3.5 `answered`, §4.1).
impl Scheduler {
    /// Takes a request for the buffer pool; a later [`Scheduler::post`] writes it into the
    /// submission queue. Nothing here touches the queue pair or waits.
    ///
    /// - An `Allocate` of blocks of the class's size that continues a held `Allocate` of the same
    ///   owner and class at the position after it raises that one's count instead. A sized block
    ///   (a declared result, an encoder output) is asked by an `Allocate` of its own, whose answer
    ///   goes to the groups that read it where its writer writes it (`Scheduler::mapped`).
    /// - A `Release`, a `Commit`, a `Detach`, a `Transfer` or a `Replicate` is handed to
    ///   `Executor::stamp` of every group whose device block tables may hold the entries it
    ///   names: the group of the owner's attachment, and every other group the request is
    ///   attached to that reads the class (one that states a region of the class that the
    ///   owner's group states too), and for a `Replicate` the group it copies into. The
    ///   groups are found from the request's `attachments`, so an owner's `Detach` is submitted
    ///   before its attachment is cleared, and an owner whose entries no table holds (a failed
    ///   `Transfer`) has its attachment cleared first. A `Finish { ok: false }` that gives a copy
    ///   up is handed to the copy's group. A request with no group to stamp it is held at once
    ///   with no stamp. A `Detach` takes its owner out of `owners`, and the completions waiting
    ///   for that owner's slot, or acting for it, are dropped with it.
    /// - An `Allocate`, a `Pin`, a `Load` or a `Clone` at positions a `Release` being stamped
    ///   frees waits behind that `Release`, so that the buffer pool takes it after the positions
    ///   are free.
    /// - Every other request is held, in order.
    ///
    /// True when the request is to be written as a record of its own, which one completion
    /// answers; false when it raised a held `Allocate`, whose completion answers both. The caller
    /// submits no `Allocate` of an owner and class whose `Allocate` is outstanding: that need is
    /// asked again after its completion.
    fn submit(&mut self, request: runtime::Request) -> bool {
        if self.merge(&request) {
            return false;
        }
        if request == runtime::Request::List {
            self.lists_asked += 1;
        }

        let groups = match &request {
            runtime::Request::Release { owner, class, .. }
            | runtime::Request::Commit { owner, class, .. } => self.holders(*owner, Some(*class)),
            // A crossing reads its block once no batch of a group that may write it is in flight
            // (BP §9, A2): the groups that hold the entry, and the group it copies into.
            runtime::Request::Replicate {
                owner,
                class,
                group,
                ..
            } => {
                let mut groups = self.holders(*owner, Some(*class));
                if !groups.contains(group) {
                    groups.push(*group);
                }
                groups
            }
            runtime::Request::Detach { owner, .. } | runtime::Request::Transfer { owner, .. } => {
                self.holders(*owner, None)
            }
            runtime::Request::Finish {
                copy, ok: false, ..
            } => self
                .outstanding
                .get(&copy.0)
                .and_then(|(_, copied, _)| self.copy_group(copied))
                .into_iter()
                .collect(),
            _ => {
                match self.release_ahead(&request) {
                    Some(key) => {
                        if let Some((_, _, following)) = self.stamping.get_mut(&key) {
                            following.push(request);
                        }
                    }
                    None => self.held.push(request),
                }
                return true;
            }
        };

        if let runtime::Request::Detach { owner, .. } = &request {
            self.owners.remove(owner);
            self.drop_unmapped(*owner);
        }
        self.hand(request, groups);
        true
    }

    /// Drops the completions `unmapped` holds that `owner` leaving concerns: those waiting for
    /// its slot, and those of requests acting for it, whose entries its `Detach` takes back.
    pub(super) fn drop_unmapped(&mut self, owner: OwnerId) {
        self.unmapped.remove(&owner);
        for waiting in self.unmapped.values_mut() {
            waiting.retain(|(_, _, request, _)| owner_of(request) != Some(owner));
        }
        self.unmapped.retain(|_, waiting| !waiting.is_empty());
    }

    /// Hands `request` to the executor for the stamps of the workers of `groups` and keeps it in
    /// `stamping` until each has returned it; with no group it is held at once, with no stamp.
    fn hand(&mut self, request: runtime::Request, groups: Vec<GroupId>) {
        if groups.is_empty() {
            self.held.push(request);
            return;
        }

        let key = self.next_submission;
        self.next_submission += 1;
        for &group in &groups {
            self.executor.stamp(group, vec![request.clone()]);
        }
        let count = u8::try_from(groups.len()).unwrap_or(u8::MAX);
        self.stamping.insert(key, (request, count, Vec::new()));
    }

    /// Raises the count of a held or waiting `Allocate` that `request` continues: the same
    /// owner and class, blocks of the class's size, at the position after its last. True when it
    /// did.
    fn merge(&mut self, request: &runtime::Request) -> bool {
        let runtime::Request::Allocate {
            owner,
            class,
            at,
            count,
            units,
        } = *request
        else {
            return false;
        };
        // A sized block is a declared result or an encoder output, whose answer may go to other
        // groups than its neighbour's: it keeps an `Allocate` of its own.
        if units > 0 {
            return false;
        }

        let continues = |held: &runtime::Request| {
            matches!(
                *held,
                runtime::Request::Allocate {
                    owner: o,
                    class: c,
                    at: a,
                    count: n,
                    units: u,
                } if o == owner && c == class && u == units && a.checked_add(n) == Some(at)
            )
        };
        let waiting = self
            .stamping
            .values_mut()
            .flat_map(|(_, _, following)| following.iter_mut());
        let found = self
            .held
            .iter_mut()
            .rev()
            .chain(waiting)
            .find(|held| continues(held));

        match found {
            Some(runtime::Request::Allocate { count: held, .. }) => {
                *held = held.saturating_add(count);
                true
            }
            _ => false,
        }
    }

    /// The key of the latest `Release` being stamped that frees positions `request` names, for an
    /// `Allocate`, a `Pin`, a `Load` or a `Clone`.
    fn release_ahead(&self, request: &runtime::Request) -> Option<u64> {
        if !matches!(
            request,
            runtime::Request::Allocate { .. }
                | runtime::Request::Pin { .. }
                | runtime::Request::Load { .. }
                | runtime::Request::Clone { .. }
        ) {
            return None;
        }

        self.stamping
            .iter()
            .filter(|(_, (waiting, _, _))| {
                matches!(waiting, runtime::Request::Release { .. }) && overlap(request, waiting)
            })
            .map(|(key, _)| *key)
            .max()
    }

    /// Takes the requests the executor returns ([`Message::Submit`]). A request handed for stamps
    /// gathers the stamp each returned copy carries, and once every group has returned it, it is
    /// held, followed by the requests that waited behind it. A request no executor was handed
    /// for stamps (the `Finish` of a copy a worker ran) is held as it is, and its copy leaves
    /// `copies`.
    fn stamped(&mut self, requests: Vec<runtime::Request>) {
        for mut request in requests {
            let stamps = stamps_of(&mut request)
                .map(std::mem::take)
                .unwrap_or_default();

            // The earliest handed request it is, which no group of these stamps has returned yet.
            let key = self
                .stamping
                .iter()
                .filter(|(_, (waiting, _, _))| {
                    let mut waiting = waiting.clone();
                    let gathered = stamps_of(&mut waiting)
                        .map(std::mem::take)
                        .unwrap_or_default();
                    waiting == request
                        && !gathered
                            .iter()
                            .any(|(group, _)| stamps.iter().any(|(stamp, _)| stamp == group))
                })
                .map(|(key, _)| *key)
                .min();

            // The `Finish` of a copy the executor ran: the executor holds the copy no more.
            let Some(key) = key else {
                if let runtime::Request::Finish { copy, .. } = &request {
                    self.copies.remove(&copy.0);
                }
                if let Some(after) = stamps_of(&mut request) {
                    *after = stamps;
                }
                self.held.push(request);
                continue;
            };
            let Some((waiting, remaining, _)) = self.stamping.get_mut(&key) else {
                continue;
            };
            if let Some(after) = stamps_of(waiting) {
                after.extend(stamps);
            }
            *remaining = remaining.saturating_sub(1);
            if *remaining == 0
                && let Some((waiting, _, following)) = self.stamping.remove(&key)
            {
                self.held.push(waiting);
                self.held.extend(following);
            }
        }
    }

    /// Writes the held requests into the submission queue, in order, each under the next number,
    /// while the submission queue has room and the completion queue has room for a completion of
    /// every request outstanding and of the one being written (SD §3.1). Each request's bound is
    /// taken as it is written and kept with it in `outstanding` until its answer is read. What is
    /// not written stays held for the next pass, in order; nothing waits.
    ///
    /// A request the completion queue has no room for stays held, and every request behind it
    /// with it, but a `Finish`, which is written as soon as it fits itself. The room is made by
    /// the answers outstanding, and an answer may wait for a copy to end: a request answered
    /// `Copy` owes until the answer to its `Finish`, and a transfer waits for a load that fills
    /// its target group's memory (BP §11). A `Finish` held behind a request that waits for room
    /// (a `List`, which owes the whole ring, waits for every answer) would hold back the very
    /// answers that make it.
    ///
    /// # Errors
    ///
    /// [`Error::Storage`] when a request's record exceeds a ring or the queue pair is corrupt.
    fn post(&mut self) -> Result<(), Error> {
        let ring = self.queue.max_record() + LENGTH_WORD;
        let mut owed: usize = self
            .outstanding
            .values()
            .map(|&(.., bytes)| bytes)
            .fold(0, usize::saturating_add);

        let mut requests = std::mem::take(&mut self.held).into_iter();
        let mut record = Vec::new();
        let mut waiting = false;
        let mut posted = Ok(());
        for request in requests.by_ref() {
            let bytes = self.answer_bytes(&request);
            let finish = matches!(request, runtime::Request::Finish { .. });
            if (waiting && !finish) || owed.saturating_add(bytes) > ring {
                waiting = true;
                self.held.push(request);
                continue;
            }

            record.clear();
            let number = self.next_submission;
            request.encode(runtime::RequestId(number), &mut record);
            match self.queue.submit(&record) {
                Ok(true) => {}
                // The submission queue is full: this request and every one after it wait.
                Ok(false) => {
                    self.held.push(request);
                    break;
                }
                Err(error) => {
                    self.held.push(request);
                    posted = Err(storage_error(error));
                    break;
                }
            }
            let owner = self.acting_owner(&request);
            self.next_submission += 1;
            owed = owed.saturating_add(bytes);
            self.outstanding.insert(number, (owner, request, bytes));
        }
        self.held.extend(requests);
        posted
    }

    /// Reads the next completion the scheduler applies: its request's number, the owner the
    /// request acts for, the request, and the completion. `None` when the completion queue has no
    /// more.
    ///
    /// The answer to a `Finish` is returned as the result of its copy: under the number and with
    /// the request of the copy (the `Load`, `Clone`, `Store`, `Save` or `Restore` answered
    /// `Copy`), which leaves `outstanding` with it.
    ///
    /// A completion is dropped, and the next one read, when the owner its request acts for has
    /// left `owners` (what it gave goes with that owner's `Detach`), but a `Committed`, whose
    /// block a `Detach` does not take back (BP §9, R6) and which the write policy may store; and
    /// a completion that gives entries when a `Release`, a `Commit` or a `Transfer` naming them
    /// was handed to the executor after its request was submitted (BP §9, obligation 1). A
    /// dropped `Copy` is given up at once with `Finish { ok: false }` through its group's
    /// executor; any other enters `copies` until it ends. A `Refused` with the cause `copy`
    /// answers a `Finish` for a copy another `Finish` ended, and is dropped.
    ///
    /// Before it is returned, a completion that gives entries is handed to `Executor::map`, with
    /// the request as it was written: to the group an `Attach` or a `Transfer` names, to a copy's
    /// group, and otherwise to the owner's group and every other group that reads the class, each
    /// with the slot of the request's own owner there. Where that owner's `Attached` has not been
    /// read yet, the slot is not known and the completion waits in `unmapped` for it
    /// ([`Scheduler::give`]): it is handed over as that `Attached` is, whatever order the buffer
    /// pool answers in.
    ///
    /// # Errors
    ///
    /// [`Error::Storage`] when a record is not a completion of this version, or answers no
    /// outstanding request.
    #[expect(
        clippy::type_complexity,
        reason = "a completion is applied with the number, owner and request it answers, which \
                  are the storage interface's own types"
    )]
    fn next_completion(
        &mut self,
    ) -> Result<
        Option<(
            runtime::RequestId,
            Option<OwnerId>,
            runtime::Request,
            Completion,
        )>,
        Error,
    > {
        let mut record = Vec::new();
        while self
            .queue
            .next_completion(&mut record)
            .map_err(storage_error)?
        {
            let (id, completion) = Completion::decode(&record).map_err(storage_error)?;
            let (owner, request, bytes) = self.outstanding.remove(&id.0).ok_or_else(|| {
                Error::Storage(format!(
                    "a completion answers request {}, which is not outstanding",
                    id.0
                ))
            })?;
            // What it states of the cache holds whether or not the completion is dropped.
            self.learn(&request, &completion);
            // The capacity words read with the answer of a request that gives units back may not
            // show what it gave.
            if Self::gives_back(&request) {
                for (region, _) in self.device_regions() {
                    self.released[region] = true;
                }
            }

            if let runtime::Request::Finish { copy, .. } = request {
                if let Some((owner, copied)) = self.matched_finish(id, copy, request, &completion) {
                    return Ok(Some((copy, owner, copied, completion)));
                }
                continue;
            }
            // A copy's request is applied, and kept for the answer to its `Finish`, as the keys
            // its copy covers, for which that answer speaks; the executor is handed it as it
            // was written, which the completion answers.
            let applied = cache::covered(&request, &completion);
            if self.matched(id, owner, &request, &applied, &completion, bytes) {
                return Ok(Some((id, owner, applied, completion)));
            }
        }
        Ok(None)
    }

    /// Matches the completion of a request other than a `Finish`, as
    /// [`Scheduler::next_completion`] states: false when it is dropped, true when it was handed
    /// to the executor that takes it, or waits for the slot it is handed under, and is to be
    /// applied. `request` is the request as it was written, which the executor is handed,
    /// and `applied` the request as it is applied ([`cache::covered`]); `bytes` is what it owed
    /// the completion queue.
    fn matched(
        &mut self,
        id: runtime::RequestId,
        owner: Option<OwnerId>,
        request: &runtime::Request,
        applied: &runtime::Request,
        completion: &Completion,
        bytes: usize,
    ) -> bool {
        // A request answered `Copy` waits for the answer to its `Finish`, owing what it owed.
        if matches!(completion, Completion::Copy { .. }) {
            self.outstanding
                .insert(id.0, (owner, applied.clone(), bytes));
        }

        let gives = matches!(
            completion,
            Completion::Allocated { .. }
                | Completion::Pinned { .. }
                | Completion::Transferred { .. }
                | Completion::Copy { .. }
                | Completion::Replicated { .. }
        );
        // A commit outlives its owner's `Detach` (BP §9, R6): its block is shared under its key
        // all the same, and the write policy may store it.
        let outlives = matches!(completion, Completion::Committed);
        if (self.forgotten(owner) && !outlives) || (gives && self.superseded(id.0, owner, request))
        {
            if let Completion::Copy { group, .. } = completion {
                self.give_up_copy(id, *group);
            }
            return false;
        }

        for (group, slot) in self.mapped(owner, request, completion) {
            self.give(group, slot, id, request.clone(), completion.clone());
        }

        // The slot of the owner's attachment, on the one group it is attached to, is known now:
        // the completions read before it are handed over under it, in the order they were read.
        if let (runtime::Request::Attach { owner, .. }, Completion::Attached { slot }) =
            (request, completion)
        {
            for (group, number, waited, given) in self.unmapped.remove(owner).unwrap_or_default() {
                self.give(group, Ok(*slot), number, waited, given);
            }
        }
        true
    }

    /// Hands a completion that gives entries to the executor, for `group`, under `slot` (SD §2.1
    /// `Executor::map`); a `Copy` it hands over is the workers' to run from then on, and enters
    /// `copies` with its bound (SD §4.4). With `Err(owner)`, the slot is that of `owner`'s
    /// attachment to `group`, whose `Attached` has not been read: the completion waits in
    /// `unmapped` until it is.
    fn give(
        &mut self,
        group: GroupId,
        slot: Result<u32, OwnerId>,
        id: runtime::RequestId,
        request: runtime::Request,
        completion: Completion,
    ) {
        let slot = match slot {
            Ok(slot) => slot,
            Err(owner) => {
                let waiting = self.unmapped.entry(owner).or_default();
                waiting.push((group, id, request, completion));
                return;
            }
        };
        if let Completion::Copy {
            group: copier,
            keys,
            ..
        } = &completion
        {
            let tokens = self.copy_tokens(&request, keys, *copier);
            let bound = Instant::now() + memory::prefetch_wait(tokens);
            self.copies.insert(id.0, (*copier, bound));
        }
        self.executor.map(group, slot, id, request, completion);
    }

    /// Gives up the copy that answered request `copy` with `Finish { copy, ok: false }`, stamped
    /// by its group's workers, which drop it if it has not started (SD §3.1, §4.4). The
    /// destination is released only once the copy has ended; the answer to the `Finish` is the
    /// copy's result.
    fn give_up_copy(&mut self, copy: runtime::RequestId, group: GroupId) {
        self.copies.remove(&copy.0);
        let give_up = runtime::Request::Finish {
            copy,
            ok: false,
            after: Vec::new(),
        };
        self.hand(give_up, vec![group]);
    }

    /// Gives up every copy that has not finished by its bound (SD §4.4 "The bound of a copy"):
    /// the request then computes those tokens, starts from noise, or leaves with no stored step.
    fn give_up_late_copies(&mut self, now: Instant) {
        let late: Vec<(u64, GroupId)> = self
            .copies
            .iter()
            .filter(|(_, (_, bound))| *bound <= now)
            .map(|(&copy, &(group, _))| (copy, group))
            .collect();
        for (copy, group) in late {
            self.give_up_copy(runtime::RequestId(copy), group);
        }
    }

    /// The tokens a copy of `request` that covers `keys` moves, by which its bound is set (SD
    /// §4.4): a block's tokens for each key of a cache group's class, as `group` states the class
    /// (class `c` is cache group `c`); none for a trajectory, a product or an encoder output.
    fn copy_tokens(&self, request: &runtime::Request, keys: &[Key], group: GroupId) -> u32 {
        let Some(shapes) = self.group(group).map(|group| &group.capacity.cache_groups) else {
            return 0;
        };
        let class_of = |key: &Key| match request {
            runtime::Request::Load { class, .. } | runtime::Request::Clone { class, .. } => {
                Some(*class)
            }
            runtime::Request::Store { .. } => self.cache.get(&Cached::id(key)).map(|row| row.class),
            _ => None,
        };
        keys.iter()
            .filter_map(class_of)
            .filter_map(|class| shapes.get(usize::from(class)))
            .map(|&(_, block)| block)
            .fold(0, u32::saturating_add)
    }

    /// Matches the answer to the `Finish` numbered `id` of the copy `copy`, as
    /// [`Scheduler::next_completion`] states: the owner and the request of the copy, whose result
    /// it is, or `None` when it is dropped.
    fn matched_finish(
        &mut self,
        id: runtime::RequestId,
        copy: runtime::RequestId,
        finish: runtime::Request,
        completion: &Completion,
    ) -> Option<(Option<OwnerId>, runtime::Request)> {
        // Another `Finish` ended the copy; the copy's request waits for that one's answer.
        if *completion == (Completion::Refused { cause: Cause::Copy }) {
            return None;
        }
        let (owner, copied, _) = self.outstanding.remove(&copy.0)?;
        if self.forgotten(owner) {
            return None;
        }

        // The entries of a `Load` or a `Clone` appear with the answer to its `Finish`.
        let fills = matches!(
            completion,
            Completion::Loaded | Completion::Cloned | Completion::Duplicate
        );
        if fills {
            if self.superseded(copy.0, owner, &copied) {
                return None;
            }
            if let Some(group) = self.copy_group(&copied) {
                let slot = self.slot_of(owner, group);
                self.give(group, slot, id, finish, completion.clone());
            }
        }
        Some((owner, copied))
    }

    /// Reads the `available` and `retiring` words of every region this queue pair uses, raises
    /// `available` by the units of `order` beyond the installed head, which the buffer pool may
    /// evict once a later `Victims` names them, and lowers it by the units of the scheduler's
    /// `Allocate`s, `Load`s and `Clone`s held, being stamped or outstanding, which the buffer
    /// pool's word may not show yet: a `Load` or a `Clone` answered `Copy` took its blocks then,
    /// and the word shows them.
    fn read_capacity(&mut self) {
        let words: Vec<(usize, u32, u32)> = self
            .registers
            .capacity
            .iter()
            .filter_map(|&(number, word)| {
                let region = self.region_index(number)?;
                let word = usize::try_from(word).unwrap_or(usize::MAX);
                Some((region, self.word(word), self.word(word.saturating_add(1))))
            })
            .collect();
        for (region, available, retiring) in words {
            self.available[region] = available.saturating_add(self.beyond_head(region));
            self.retiring[region] = retiring;
        }

        let copying = self.copying();
        let waiting = self
            .stamping
            .values()
            .flat_map(|(_, _, following)| following.iter());
        let unanswered = self
            .outstanding
            .iter()
            .filter(|(number, _)| !copying.contains(number))
            .map(|(_, (_, request, _))| request);
        let asked: Vec<(usize, u32)> = self
            .held
            .iter()
            .chain(waiting)
            .chain(unanswered)
            .flat_map(|request| self.taken(request))
            .collect();
        for (region, units) in asked {
            if let Some(available) = self.available.get_mut(region) {
                *available = available.saturating_sub(units);
            }
        }
    }

    /// The numbers of the requests answered `Copy` whose copy has not been answered yet: the
    /// copies the workers hold, and those whose `Finish` is held, being stamped or outstanding.
    fn copying(&self) -> HashSet<u64> {
        let stamped = self
            .stamping
            .values()
            .flat_map(|(request, _, following)| std::iter::once(request).chain(following.iter()));
        let finishes = self
            .held
            .iter()
            .chain(stamped)
            .chain(self.outstanding.values().map(|(_, request, _)| request))
            .filter_map(|request| match request {
                runtime::Request::Finish { copy, .. } => Some(copy.0),
                _ => None,
            });
        self.copies.keys().copied().chain(finishes).collect()
    }

    /// Word `at` of the capacity words of `region` (by index) in shared memory: 0 its
    /// `available` units, 1 its `retiring` units, 2 its `evictable` units (every shared block
    /// without a pin there, in any eviction order or none). 0 for a region this queue pair has no
    /// words of.
    fn capacity_word(&self, region: usize, at: usize) -> u32 {
        let Some(&(number, _)) = self.regions.get(region) else {
            return 0;
        };
        self.registers
            .capacity
            .iter()
            .find(|(stated, _)| *stated == number)
            .map_or(0, |&(_, word)| {
                let first = usize::try_from(word).unwrap_or(usize::MAX);
                self.word(first.saturating_add(at))
            })
    }

    /// The register at `index` as a count of units.
    fn word(&self, index: usize) -> u32 {
        let value = self.queue.register(index).load(Ordering::Acquire);
        u32::try_from(value).unwrap_or(u32::MAX)
    }

    /// The index of the region the buffer pool numbers `number`.
    fn region_index(&self, number: u8) -> Option<usize> {
        self.regions
            .iter()
            .position(|(stated, _)| *stated == number)
    }

    /// The index in `groups` of the group the session numbers `id`.
    fn group_index(&self, id: GroupId) -> Option<usize> {
        self.groups.iter().position(|group| group.id() == id)
    }

    fn group(&self, id: GroupId) -> Option<&WorkerGroup> {
        self.groups.iter().find(|group| group.id() == id)
    }

    /// The request `owner` is for.
    fn request_of(&self, owner: OwnerId) -> Option<&Request> {
        self.owners.get(&owner).and_then(|id| self.requests.get(id))
    }

    /// Whether the owner a request acts for has left `owners`.
    fn forgotten(&self, owner: Option<OwnerId>) -> bool {
        owner.is_some_and(|owner| !self.owners.contains_key(&owner))
    }

    /// The owner a request acts for: the one it names, or for a `Finish` the one its copy's
    /// request names.
    fn acting_owner(&self, request: &runtime::Request) -> Option<OwnerId> {
        match request {
            runtime::Request::Finish { copy, .. } => {
                self.outstanding.get(&copy.0).and_then(|(owner, ..)| *owner)
            }
            request => owner_of(request),
        }
    }

    /// The slot row of the request `owner` is for on `group`, which its attachment there states
    /// once that attachment's `Attached` is read (slot rows start at 1); before then, `Err` with
    /// the attachment's owner. 0, a row no owner holds, when the request has no attachment there
    /// (a `Store`'s copy acts for no owner).
    fn slot_of(&self, owner: Option<OwnerId>, group: GroupId) -> Result<u32, OwnerId> {
        let attached = owner
            .and_then(|owner| self.request_of(owner))
            .and_then(|request| {
                let attachments = request.attachments.iter();
                attachments
                    .into_iter()
                    .find(|attachment| attachment.0 == group)
            });
        match attached {
            Some(&(_, attached, 0, _)) => Err(attached),
            Some(&(_, _, slot, _)) => Ok(slot),
            None => Ok(0),
        }
    }

    /// The groups whose device block tables may hold entries of `owner`, of `class` or of any
    /// class: the group of the owner's attachment, and every other group its request is attached
    /// to that states a region of the class that the owner's group states too, or receives the
    /// class, holding the entries a crossing brought it.
    fn holders(&self, owner: OwnerId, class: Option<u8>) -> Vec<GroupId> {
        let Some(request) = self.request_of(owner) else {
            return Vec::new();
        };
        let Some((home, _)) = request.attachment(owner) else {
            return Vec::new();
        };
        let mut groups = vec![home];
        let Some(writer) = self.group(home) else {
            return groups;
        };

        let classes = match class {
            Some(class) => class..class.saturating_add(1),
            None => 0..u8::try_from(self.classes.len()).unwrap_or(u8::MAX),
        };
        for &(group, other, _, _) in &request.attachments {
            if other == owner || groups.contains(&group) {
                continue;
            }
            let Some(reader) = self.group(group) else {
                continue;
            };
            let reads = classes.clone().any(|class| {
                let receives = reader.layout.receives.get(usize::from(class));
                reader
                    .regions(class)
                    .iter()
                    .any(|region| writer.regions(class).contains(region))
                    || receives.is_some_and(|regions| !regions.is_empty())
            });
            if reads {
                groups.push(group);
            }
        }
        groups
    }

    /// The group whose workers run the copies of `request`: the group a `Store` names, and the
    /// owner's group of a `Load`, a `Clone`, a `Save` or a `Restore`.
    fn copy_group(&self, request: &runtime::Request) -> Option<GroupId> {
        match request {
            runtime::Request::Store { group, .. } => Some(*group),
            runtime::Request::Load { owner, .. }
            | runtime::Request::Clone { owner, .. }
            | runtime::Request::Save { owner, .. }
            | runtime::Request::Restore { owner, .. } => self
                .request_of(*owner)
                .and_then(|row| row.attachment(*owner))
                .map(|(group, _)| group),
            _ => None,
        }
    }

    /// The groups, each with its slot as [`Scheduler::slot_of`] states it, that a completion of
    /// `request` is handed to by `map`: none for a completion that gives no entries. A block's
    /// allocation or pin goes to the groups that may hold its entry and read it in a memory its
    /// writer writes it in, and a crossing's `Replicated` to the group it copied into, which
    /// reads it in the memories the crossing made it valid in (BP §9, obligation 1).
    fn mapped(
        &self,
        owner: Option<OwnerId>,
        request: &runtime::Request,
        completion: &Completion,
    ) -> Vec<(GroupId, Result<u32, OwnerId>)> {
        match (request, completion) {
            (runtime::Request::Attach { group, .. }, Completion::Attached { slot })
            | (runtime::Request::Transfer { group, .. }, Completion::Transferred { slot, .. }) => {
                vec![(*group, Ok(*slot))]
            }
            (_, Completion::Copy { group, .. }) => vec![(*group, self.slot_of(owner, *group))],
            (
                runtime::Request::Allocate {
                    owner, class, at, ..
                }
                | runtime::Request::Pin {
                    owner, class, at, ..
                },
                Completion::Allocated { .. } | Completion::Pinned { .. },
            ) => self
                .holders(*owner, Some(*class))
                .into_iter()
                .filter(|group| self.reads_where_written(*owner, (*class, *at), *group))
                .map(|group| (group, self.slot_of(Some(*owner), group)))
                .collect(),
            (runtime::Request::Replicate { owner, group, .. }, Completion::Replicated { .. }) => {
                vec![(*group, self.slot_of(Some(*owner), *group))]
            }
            _ => Vec::new(),
        }
    }

    /// Whether a `Release`, a `Commit`, a `Transfer` or a `Replicate` naming entries `request`
    /// gives was handed to the executor after `request`, numbered `number`, was submitted: it is
    /// being stamped, held, or outstanding under a later number. A request submitted after such
    /// a `Release` waited behind it, so it was written after it.
    fn superseded(&self, number: u64, owner: Option<OwnerId>, request: &runtime::Request) -> bool {
        if owner.is_none() {
            return false;
        }
        let names = |other: &runtime::Request| {
            matches!(
                other,
                runtime::Request::Release { .. }
                    | runtime::Request::Commit { .. }
                    | runtime::Request::Transfer { .. }
                    | runtime::Request::Replicate { .. }
            ) && overlap(request, other)
        };

        self.stamping.values().any(|(waiting, _, _)| names(waiting))
            || self.held.iter().any(names)
            || self
                .outstanding
                .iter()
                .any(|(later, (_, other, _))| *later > number && names(other))
    }

    /// Per region (by index), the units a request takes from the device regions of a class in
    /// its owner's group: an `Allocate` its blocks, a `Load` a block per key, which the copy
    /// from the host fills, and a `Clone` the one block it copies into; a `Replicate` the runs
    /// it takes in the receiving regions of the group it copies into; and from its class's host
    /// region, a `Prefetch` a host block per key, which it brings the key into.
    fn taken(&self, request: &runtime::Request) -> Vec<(usize, u32)> {
        if let runtime::Request::Replicate {
            owner,
            at,
            from,
            group,
            ..
        } = *request
        {
            return self
                .request_of(owner)
                .map(|row| self.receiving(row, (at, group, from)))
                .unwrap_or_default();
        }
        if let runtime::Request::Prefetch {
            class, units, keys, ..
        } = request
        {
            let count = u32::try_from(keys.len()).unwrap_or(u32::MAX);
            return self
                .host_block(*class, *units)
                .map_or_else(Vec::new, |(region, each)| {
                    vec![(region, count.saturating_mul(each))]
                });
        }
        let (owner, class, count, units) = match request {
            runtime::Request::Allocate {
                owner,
                class,
                count,
                units,
                ..
            } => (*owner, *class, *count, *units),
            runtime::Request::Load {
                owner, class, keys, ..
            } => (
                *owner,
                *class,
                u32::try_from(keys.len()).unwrap_or(u32::MAX),
                0,
            ),
            runtime::Request::Clone { owner, class, .. } => (*owner, *class, 1, 0),
            _ => return Vec::new(),
        };
        let Some(regions) = self.class_regions(owner, class) else {
            return Vec::new();
        };

        let each = self.block_units(class, units).unwrap_or(0);
        regions
            .into_iter()
            .map(|region| (region, count.saturating_mul(each)))
            .collect()
    }

    /// The regions, as indices, owner `owner`'s blocks of `class` are taken from: those of the
    /// class in the group of the owner's attachment. `None` when the owner is unknown.
    fn class_regions(&self, owner: OwnerId, class: u8) -> Option<Vec<usize>> {
        let (group, _) = self.request_of(owner)?.attachment(owner)?;
        let regions = self.group(group)?.regions(class);
        Some(regions.iter().map(|&region| usize::from(region)).collect())
    }

    /// The host region (by index) of `class`, and the units one host copy of its blocks takes
    /// there: the size the class states for its host copies, else its block size, or `units`
    /// for a sized class. `None` for a class with no host region.
    fn host_block(&self, class: u8, units: u32) -> Option<(usize, u32)> {
        let (region, _) = self.classes.get(usize::from(class))?.host?;
        let region = usize::from(region);
        let each = match self.class_units(class, region) {
            0 => units,
            each => each,
        };
        Some((region, each))
    }

    /// The units of one block of `class`: the class's block size, or `units` for a sized class.
    fn block_units(&self, class: u8, units: u32) -> Option<u32> {
        let stated = self.classes.get(usize::from(class))?.units;
        Some(if stated > 0 { stated } else { units })
    }

    /// The runs one block of `class` has in the group of `owner`, and its units.
    fn block_shape(&self, owner: OwnerId, class: u8, units: u32) -> Option<(usize, usize)> {
        let runs = self.class_regions(owner, class)?.len().max(1);
        let units = usize::try_from(self.block_units(class, units)?).ok()?;
        Some((runs, units.max(1)))
    }

    /// The most bytes the completion of `request` takes in the completion queue: the largest
    /// record any of its answers can have (BP §11), padded, with its length word, and at most a
    /// whole ring. An answer's lists are bounded by what the request names: a run per block and
    /// region, and an evicted key per unit the request takes, since every evicted block frees at
    /// least one. A `Listing`, and an answer whose lists the scheduler cannot bound, may take the
    /// whole ring.
    fn answer_bytes(&self, request: &runtime::Request) -> usize {
        // The fields of the records (`storage::runtime`): a key, a run, a list's length.
        const KEY: usize = 32;
        const RUN: usize = 9;
        const LIST: usize = 4;
        let copy = |keys: usize, from: usize, to: usize, evicted: usize| {
            1 + LIST
                + KEY.saturating_mul(keys)
                + LIST
                + RUN.saturating_mul(from)
                + LIST
                + RUN.saturating_mul(to)
                + LIST
                + KEY.saturating_mul(evicted)
        };
        let shape = |owner: OwnerId, class: u8, units: u32| self.block_shape(owner, class, units);

        let fields = match request {
            runtime::Request::Detach { .. }
            | runtime::Request::Release { .. }
            | runtime::Request::Commit { .. }
            | runtime::Request::Victims { .. } => Some(0),
            runtime::Request::Attach { .. } => Some(4),
            // Its count, and an evicted key per host unit the blocks of its keys take.
            runtime::Request::Prefetch {
                class, units, keys, ..
            } => {
                let each = self.host_block(*class, *units).map_or(0, |(_, each)| each);
                Some(4 + LIST + KEY.saturating_mul(keys.len().saturating_mul(each as usize)))
            }
            // The answers to a `Finish`: the largest is a `Stored` with its empty list.
            runtime::Request::Finish { .. } => Some(LIST),
            runtime::Request::Allocate {
                owner,
                class,
                count,
                units,
                ..
            } => shape(*owner, *class, *units).map(|(runs, units)| {
                let runs = runs.saturating_mul(*count as usize);
                LIST + RUN.saturating_mul(runs)
                    + LIST
                    + KEY.saturating_mul(runs.saturating_mul(units))
            }),
            runtime::Request::Pin {
                owner, class, keys, ..
            } => shape(*owner, *class, 0)
                .map(|(runs, _)| LIST + RUN.saturating_mul(runs.saturating_mul(keys.len())) + 4),
            runtime::Request::Clone { owner, class, .. } => shape(*owner, *class, 0)
                .map(|(runs, units)| copy(1, runs, runs, runs.saturating_mul(units))),
            // One host run per key read, and the device blocks it fills.
            runtime::Request::Load {
                owner, class, keys, ..
            } => shape(*owner, *class, 0).map(|(runs, units)| {
                let keys = keys.len();
                let blocks = runs.saturating_mul(keys);
                copy(keys, keys, blocks, blocks.saturating_mul(units))
            }),
            runtime::Request::Store {
                keys,
                tier: Tier::Remote,
                ..
            }
            | runtime::Request::Evict { keys } => Some(LIST + KEY.saturating_mul(keys.len())),
            runtime::Request::Store { group, keys, .. } => self.store_bytes(*group, keys, &copy),
            // From the slot's row to one host block of `units`.
            runtime::Request::Save { units, .. } => Some(copy(1, 1, 1, *units as usize)),
            runtime::Request::Restore { .. } => Some(copy(1, 1, 1, 0)),
            runtime::Request::Transfer { owner, group, .. } => self
                .transfer_entries(*owner, *group)
                .map(|entries| 4 + LIST + 15usize.saturating_mul(entries)),
            runtime::Request::Resize { region, units } => self
                .region_index(*region)
                .and_then(|index| self.capacity.get(index))
                .map(|capacity| 4 + LIST + KEY * capacity.saturating_sub(*units) as usize),
            // The block's runs in the group's regions and receiving regions of the class, and a
            // key evicted per unit its receiving runs take.
            runtime::Request::Replicate {
                owner,
                class,
                at,
                from,
                group,
                ..
            } => self
                .request_of(*owner)
                .zip(self.group(*group))
                .map(|(row, reader)| {
                    let receiving = self.receiving(row, (*at, *group, *from));
                    let runs = reader.regions(*class).len() + receiving.len();
                    let evicted = receiving.iter().map(|(_, units)| *units as usize).sum();
                    LIST + RUN.saturating_mul(runs) + LIST + KEY.saturating_mul(evicted)
                }),
            runtime::Request::List => None,
        };

        // The number, the variant, and at least the one byte of a `Refused` or a `Full`.
        let ring = self.queue.max_record() + LENGTH_WORD;
        fields.map_or(ring, |fields| {
            let record = 8usize.saturating_add(1).saturating_add(fields.max(1));
            record
                .checked_next_multiple_of(8)
                .map_or(ring, |padded| (padded + LENGTH_WORD).min(ring))
        })
    }

    /// The fields of the largest `Copy` answering a `Store` to the host of `keys` from `group`:
    /// per key, the runs of its device block in the group, one host run, and an evicted key per
    /// unit of its host copy. `None` when a key's class or host copy size is not known.
    fn store_bytes(
        &self,
        group: GroupId,
        keys: &[Key],
        copy: &impl Fn(usize, usize, usize, usize) -> usize,
    ) -> Option<usize> {
        let group = self.group(group)?;
        let (mut from, mut evicted) = (0usize, 0usize);
        for key in keys {
            let class = self.cache.get(&Cached::id(key))?.class;
            let (_, host) = self.classes.get(usize::from(class))?.host?;
            let units = if host > 0 {
                host
            } else {
                self.block_units(class, 0).filter(|units| *units > 0)?
            };
            from = from.saturating_add(group.regions(class).len().max(1));
            evicted = evicted.saturating_add(units as usize);
        }
        Some(copy(keys.len(), from, keys.len(), evicted))
    }

    /// The most entries a `Transferred` of `owner` to `group` states: per class, a run per
    /// region of each position asked for, in the source group or the target, whichever has more.
    fn transfer_entries(&self, owner: OwnerId, group: GroupId) -> Option<usize> {
        let request = self.request_of(owner)?;
        let source = self.group(request.attachment(owner)?.0)?;
        let target = self.group(group)?;

        let mut entries = 0usize;
        for (class, blocks) in request.blocks.iter().enumerate() {
            let class = u8::try_from(class).ok()?;
            let runs = source
                .regions(class)
                .len()
                .max(target.regions(class).len())
                .max(1);
            entries = entries.saturating_add(runs.saturating_mul(blocks.asked as usize));
        }
        Some(entries)
    }
}

/// The scheduler's one reading of the host's clocks, taken once per process, from which every
/// instant is placed: an `Instant`, the host's monotonic clock in nanoseconds at that instant,
/// and the wall clock in seconds since the Unix epoch. An instant maps to the same value however
/// the wall clock moves after.
fn clocks() -> (Instant, u64, f64) {
    static ORIGIN: OnceLock<(Instant, u64, f64)> = OnceLock::new();
    *ORIGIN.get_or_init(|| {
        let mut now = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `now` is a valid, writable `timespec` for the duration of the call, and
        // `CLOCK_MONOTONIC` is a clock every Linux host has.
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
        let seconds = u64::try_from(now.tv_sec).unwrap_or(0);
        let nanos = u64::try_from(now.tv_nsec).unwrap_or(0);
        let wall = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let monotonic = seconds.saturating_mul(1_000_000_000).saturating_add(nanos);
        (Instant::now(), monotonic, wall)
    })
}

/// Nanoseconds of the host's monotonic clock at `at`: the clock of `Cached::last_used`, of every
/// instant the policy is told (the guest's `Nanos`), and of a `Prefetch`'s deadline, which the
/// buffer pool compares with its own reading of the clock in its own process
/// (`storage::runtime::Request::Prefetch`).
pub(super) fn monotonic_nanos(at: Instant) -> u64 {
    let (origin, monotonic, _) = clocks();
    let nanos = |span: Duration| u64::try_from(span.as_nanos()).unwrap_or(u64::MAX);
    if at >= origin {
        monotonic.saturating_add(nanos(at - origin))
    } else {
        monotonic.saturating_sub(nanos(origin - at))
    }
}

/// The wall-clock time, in seconds since the Unix epoch, of the instant `at` (SD §7.2 "Timestamps
/// of `Scheduled`", `S/clock.rs:39`).
pub(super) fn unix_secs(at: Instant) -> f64 {
    let (origin, _, wall) = clocks();
    if at >= origin {
        wall + (at - origin).as_secs_f64()
    } else {
        wall - (origin - at).as_secs_f64()
    }
}

/// The owner a request names.
fn owner_of(request: &runtime::Request) -> Option<OwnerId> {
    match request {
        runtime::Request::Attach { owner, .. }
        | runtime::Request::Detach { owner, .. }
        | runtime::Request::Allocate { owner, .. }
        | runtime::Request::Pin { owner, .. }
        | runtime::Request::Release { owner, .. }
        | runtime::Request::Commit { owner, .. }
        | runtime::Request::Clone { owner, .. }
        | runtime::Request::Transfer { owner, .. }
        | runtime::Request::Load { owner, .. }
        | runtime::Request::Save { owner, .. }
        | runtime::Request::Restore { owner, .. }
        | runtime::Request::Replicate { owner, .. } => Some(*owner),
        _ => None,
    }
}

/// Whether two requests name a common entry of one owner's tables.
fn overlap(one: &runtime::Request, other: &runtime::Request) -> bool {
    let (Some(one), Some(other)) = (span(one), span(other)) else {
        return false;
    };
    one.0 == other.0
        && one
            .1
            .zip(other.1)
            .is_none_or(|(class, other)| class == other)
        && one.2.start < other.2.end
        && other.2.start < one.2.end
}

/// The entries a request names: its owner, the class (`None` for every class) and the
/// positions. A `Transfer` and a `Detach` name every entry of their owner.
fn span(request: &runtime::Request) -> Option<(OwnerId, Option<u8>, Range<u32>)> {
    let positions = |at: u32, count: usize| {
        let count = u32::try_from(count).unwrap_or(u32::MAX);
        at..at.saturating_add(count)
    };
    match request {
        runtime::Request::Allocate {
            owner,
            class,
            at,
            count,
            ..
        }
        | runtime::Request::Release {
            owner,
            class,
            at,
            count,
            ..
        } => Some((*owner, Some(*class), positions(*at, *count as usize))),
        runtime::Request::Pin {
            owner,
            class,
            at,
            keys,
        }
        | runtime::Request::Load {
            owner,
            class,
            at,
            keys,
        } => Some((*owner, Some(*class), positions(*at, keys.len()))),
        runtime::Request::Commit {
            owner, class, at, ..
        }
        | runtime::Request::Clone {
            owner, class, at, ..
        }
        | runtime::Request::Replicate {
            owner, class, at, ..
        } => Some((*owner, Some(*class), positions(*at, 1))),
        runtime::Request::Transfer { owner, .. } | runtime::Request::Detach { owner, .. } => {
            Some((*owner, None, 0..u32::MAX))
        }
        _ => None,
    }
}

/// The stamps of a request that carries them.
fn stamps_of(request: &mut runtime::Request) -> Option<&mut runtime::Stamps> {
    match request {
        runtime::Request::Detach { after, .. }
        | runtime::Request::Release { after, .. }
        | runtime::Request::Commit { after, .. }
        | runtime::Request::Transfer { after, .. }
        | runtime::Request::Finish { after, .. }
        | runtime::Request::Replicate { after, .. } => Some(after),
        _ => None,
    }
}

fn storage_error(error: storage::Error) -> Error {
    Error::Storage(error.to_string())
}
