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
//! wait taken only after a step that formed no batch, until the next message, the next
//! starvation or drain deadline, the next statistics second, or, while the buffer pool owes
//! answers, requests wait for room or a request waits for units, the buffer pool's ring of the
//! queue pair's doorbell (`storage::queue`), which it rings only while the scheduler sleeps.
//!
//! - `submit` takes a request for the buffer pool. A request the workers first apply to their
//!   device block tables (`Release`, `Detach`, `Commit`, `Lend`, `Replicate`, and the
//!   `Finish` that gives a copy up) waits in `stamping` until every group whose tables may hold
//!   its entries stamped it (BP §9, obligation 2); a request that names positions such a
//!   `Release` frees waits behind it; every other request is held in order. A further `Allocate`
//!   of an owner and class whose `Allocate` is still held raises that one's count (BP §10, F5).
//! - `post` writes the held requests in order while the submission queue has room and the
//!   completion queue has room for a completion of every request outstanding. A request's number
//!   is drawn as it is written, so the numbers rise in the order the buffer pool takes the
//!   requests, which it requires (BP §7, R1).
//! - `next_completion` reads the completion queue, matches each completion to its request by
//!   number, has the KV decisions learn what it states of the cache ([`Kv::learn`]), drops what
//!   belongs to a forgotten owner (but a `Committed`, which outlives its owner's `Detach`) or to
//!   entries a later stamped request names (BP §9, obligation 1), and hands every completion that
//!   gives entries to the executor (`Executor::map`) before the scheduler applies it. A `Copy` it
//!   hands over is recorded in `copies` until the executor returns its `Finish`, or the step
//!   gives it up at its bound (SD §4.4).
//! - `read_capacity` reads the capacity words of every region into the KV decisions.
//!
//! The KV decisions (`specs/kv-cache.md` §1.3) are a type of their own, [`Kv`] (`kv/`), which
//! owns what the scheduler knows of the buffer pool's cache and regions and decides what each
//! request asks of them; the scheduler states its requests, groups and step to it and submits
//! what it returns. The scheduling, admission, update, drain, policy and statistics rules are
//! `impl Scheduler` blocks of the sibling files (SD §6).

mod drain;
mod events;
mod group;
mod handoff;
mod kv;
pub mod policy;
mod request;
mod resize;
mod rows;
mod schedule;
mod stats;
pub(crate) use stats::STATS_PERIOD;
#[cfg(test)]
pub(crate) mod tests;
mod update;
mod video;
mod view;

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::batch::{Batch, BatchId, BatchOutput, Capacity, Generations, Phase};
use crate::executor::Executor;
use arc_swap::ArcSwap;
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use proto::config::{SchedulerConfig, Tuning, WorkerGroupConfig};
use proto::report::SchedulerStats;
use proto::request::{Event, FinishReason, GenerationLimits, MigrationCause, RequestId};
use scheduling::Parameters;
use storage::runtime::{self, Cause, Completion, GroupId, OwnerId, Tier};
use storage::{Interest, Key, QueuePair, Signals};

pub(crate) use group::WorkerGroup;
use kv::{Kv, Step};
use policy::{Ceilings, Policy, PolicyError, ceilings};
pub(crate) use request::{Blocks, Request, RequestStatus};
use schedule::Deferred;

/// The bytes of a record's length word in a ring of a queue pair (`storage::queue`): a ring of
/// `max_record()` bytes of record holds that many more.
const LENGTH_WORD: usize = 8;

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
/// or kernel generation switch, sleep and wake, an expert commit, a cache flush) is the engine
/// sending `Pause`, doing its work once the reply arrives, and sending `Resume`.
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "a batch and its output travel once per batch, by value, as the message itself"
)]
pub enum Message {
    /// A request from the daemon, with the event window its consumer stated.
    AddRequest(proto::request::Request, std::num::NonZeroU32),
    /// A warm-up request of the engine (SD §3.2), with the event window it is scheduled under:
    /// served like a request from the daemon, under an id the scheduler gives it, with its events
    /// dropped ([`Scheduler::warm_up`]).
    WarmUp(Box<proto::request::GenerationRequest>, std::num::NonZeroU32),
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
    /// How the full pages of a request that imports a lend arrive, which the KV program chose
    /// for its [`Output::Handoff`].
    Path(RequestId, scheduling::kv::Path),
    /// Form no batch; the reply is sent once nothing is in flight.
    Pause(Sender<()>),
    Resume,
    /// Admit nothing new; running requests end or migrate for the cause by the deadline, and with
    /// no deadline every request held runs to its end, the queued ones admitted in their turn;
    /// with `true`, give device memory back by `Resize`. The reply is sent once no request
    /// remains.
    Drain(MigrationCause, Option<Instant>, bool, Sender<()>),
    /// A policy generation that serves from the next step; the reply is the outcome of its
    /// instantiation, and the serving policy is untouched on failure.
    SetPolicy(Arc<wasmtime::Module>, Parameters, Sender<Result<(), Error>>),
    /// Serve the policy the last `SetPolicy` replaced.
    RollbackPolicy(Sender<Result<(), Error>>),
    /// Drop the policy kept for rollback.
    CommitPolicy,
    /// Form every batch from the next step on within what a kernel generation's graphs capture
    /// (`GraphLimits`): the generation of other graph options that switches, or the one a
    /// rollback serves again. The reply is sent once the scheduler took them.
    SetGraphs(Arc<GraphLimits>, Sender<()>),
    /// Evict every cached block the scheduler knows of; the requests it holds keep theirs. The
    /// reply is sent once the evictions are submitted.
    Flush(Sender<()>),
    /// `Resize { region, units }` at the buffer pool, the region by its number; the reply, sent at
    /// its `Resized`, is the units the region reached, which a shrink and a grow may stop short of.
    Resize(u8, u32, Sender<u32>),
    /// `Resize { region, units }` at the buffer pool, the region by its number, which bounds the
    /// region to `units` as `Resize` does; the reply is sent once a `Resized` states that the
    /// region holds no more than `units`, which it reaches as the requests holding units past
    /// the bound let them go (`resize.rs`), with the units it holds then. A drain that ends the
    /// bound first is answered with the units the region holds at that moment.
    Bound(u8, u32, Sender<u32>),
    /// Before a region (by number) gives back its units past `units`, as a generation update's
    /// room does, store to the host tier the cached blocks the shrink to `units` evicts that have
    /// no host copy, so a later prompt finds them there; the reply is sent, with the blocks
    /// demoted, once every store issued ended. A region of a class without a host tier, or a
    /// step whose values switch the tier off, demotes nothing.
    Demote(u8, u32, Sender<u32>),
    /// End every request the scheduler holds, aborted, and stop.
    Shutdown,
}

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
    /// A request imports what a prefill replica lent: how its full pages arrive is the KV
    /// program's to choose, through the engine's Storage Server, and the engine answers with
    /// [`Message::Path`].
    Handoff(RequestId, scheduling::kv::Handoff),
    /// The scheduler stopped.
    Failed(Error),
}

/// What a kernel generation's graphs bound a scheduler's steps to (`Message::SetGraphs`): the
/// configuration a scheduler resolves against its groups, of the generation's batched tokens,
/// drafts and graph tiers ([`SchedulerConfig::of`]); each group's captured shapes under the
/// generation and the phases they serve, by the group's number; and the candidates a request may
/// ask for at a position.
/// A generation of other graph options switches every scheduler to its own, and a rollback to
/// the one the kept generation captured.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphLimits {
    /// The scheduler configuration before it is lowered to the groups ([`resolved`]).
    pub config: SchedulerConfig,
    /// The shapes each group's workers captured, which replace its capacity's.
    pub buckets: Vec<(GroupId, Vec<crate::batch::Shape>)>,
    /// The phases each group's workers serve under the generation (`Capacity::phases`, bit
    /// `Phase::index` per phase), which replace its capacity's: a generation of drafts serves
    /// verifications where one of none serves decodes.
    pub phases: Vec<(GroupId, u32)>,
    /// The most likely tokens a request may ask for at a position (`max_logprobs`).
    pub max_logprobs: u32,
}

/// What one step may assemble (`specs/configuration.md` §7, step values): the step values of the
/// snapshot the step began with, lowered to what the groups' workers captured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Budget {
    /// The configuration revision the values are of, which the step's batches name.
    pub(crate) revision: u64,
    /// Tokens of the step.
    pub(crate) tokens: u32,
    /// Calls of the step.
    pub(crate) calls: usize,
    /// Prompt tokens one request prefills in the step, at most `tokens`.
    pub(crate) chunk: u32,
    /// Waiting requests before a submission is refused; unbounded when `None`.
    pub(crate) queued: Option<usize>,
    /// Whether the step reads and writes cached prefixes, where the group's cache does.
    pub(crate) prefix_caching: bool,
    /// Whether the step offloads pages to the host tier, where the group's cache does.
    pub(crate) host_tier: bool,
    /// The bytes the copies started with each of the step's batches move together over the
    /// batch's group, which the batch carries to every rank.
    pub(crate) copy_budget: u64,
}

impl Budget {
    /// The budget of a step under `tuning` within `ceilings`, for a scheduler resolved to
    /// `config`: a value the snapshot leaves unstated is the scheduler's own (the step's tokens:
    /// the widest captured prefill bucket's, else the longest sequence) or the step's tokens (the
    /// prefill chunk); the calls are at most the running requests; every value is at least one
    /// and at most what the workers captured.
    pub(crate) fn of(tuning: &Tuning, ceilings: &Ceilings, config: &SchedulerConfig) -> Self {
        let count = |value: NonZeroUsize| u32::try_from(value.get()).unwrap_or(u32::MAX);
        let resolved = u32::try_from(config.max_num_batched_tokens).unwrap_or(u32::MAX);
        let tokens = tuning
            .max_num_batched_tokens
            .map_or(resolved, count)
            .clamp(1, ceilings.tokens.max(1));
        let chunk = tuning
            .chunked_prefill_size
            .map_or(tokens, count)
            .clamp(1, tokens);
        let calls = tuning
            .max_batch
            .get()
            .min(ceilings.rows)
            .min(config.max_running_requests)
            .max(1);
        Self {
            revision: tuning.revision,
            tokens,
            calls,
            chunk,
            queued: tuning.max_queued_requests.map(NonZeroUsize::get),
            prefix_caching: tuning.prefix_caching,
            host_tier: tuning.host_tier,
            copy_budget: tuning.copy_budget,
        }
    }
}

/// The scheduler of one data-parallel rank and its state (SD §2.1, §2.5).
///
/// Invariants: every id in `waiting` or `running` is in `requests` with the matching status; an
/// owner is in `owners` from its `Attach` until its `Detach` is handed to the executor, and no
/// owner is drawn twice; the completions of `outstanding` always fit the completion queue; a request is in at most one of `stamping`, `held` and
/// `outstanding`; the numbers written into the submission queue rise; all state is touched on the
/// scheduler's thread only.
///
/// A region is named in completions and requests by its number at the buffer pool, and indexed
/// here, in every per-region field, by its place in the session's regions (`Open::regions`,
/// [`Kv::regions`]), which is how the layout of a group and a class names it too.
pub struct Scheduler {
    config: SchedulerConfig,
    /// The configuration `config` was resolved from against the groups ([`resolved`]), which a
    /// kernel generation's graphs replace ([`Scheduler::set_graphs`]).
    stated: SchedulerConfig,
    /// The engine's current step values, which every scheduler of the engine shares and the
    /// control thread replaces when a change of them arrives (`Tune`).
    tuning: Arc<ArcSwap<Tuning>>,
    /// What the step that runs may assemble: the snapshot of `tuning` it began with, lowered to
    /// the ceilings ([`Budget::of`]).
    budget: Budget,
    /// What the groups' workers captured bounds a step to, and the bounds of the policy's limits.
    ceilings: Ceilings,
    /// The generation limits requests are checked against at their arrival: the model's sizing
    /// facts lowered to what the groups' workers hold (SD §7.1), with the features the groups'
    /// phases serve.
    limits: GenerationLimits,
    /// The engine epoch of the session, which names the private continuation of a token request
    /// that arrives without one (`cache.resume`, SD §3.2).
    epoch: u64,
    /// The engine's session in its host's buffer pool, which a lend names.
    session: u32,
    requests: HashMap<RequestId, Request>,
    waiting: VecDeque<RequestId>,
    running: Vec<RequestId>,
    groups: Vec<WorkerGroup>,

    /// This scheduler's end of its queue pair with the buffer pool: it writes the submission
    /// queue and reads the completion queue.
    queue: QueuePair,
    /// What the buffer pool had published to this end when the last step began to read the
    /// pair: the loop sleeps on the pair's doorbell only while nothing arrived since
    /// ([`Scheduler::run`]).
    seen: Signals,
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

    /// Where this queue pair's registers lie: per region it uses, by number, the first of its
    /// three capacity words.
    registers: runtime::Registers,
    /// The KV decisions: what the scheduler knows of the buffer pool's cache and regions, and
    /// what each request asks of them (`kv`).
    kv: Kv,
    /// The request each live owner is for.
    owners: HashMap<OwnerId, RequestId>,
    /// The counter owners are drawn from: the scheduler's data-parallel rank in the top 16 bits.
    next_owner: u64,
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
    /// The kernel and policy generations batches are stamped with.
    generations: Arc<Generations>,
    paused: bool,
    /// Whether the scheduler stopped, on a fault (`fail`), once a `Shutdown` settled
    /// ([`Scheduler::settle_shutdown`]), or when the inbox lost its last sender: it takes no
    /// message and no step from then on, and its loop ends.
    stopped: bool,
    /// Whether `Shutdown` arrived: every request has ended, and the scheduler takes steps only
    /// until the copies its workers hold and the stores to the host it offered have ended (BP
    /// §20: host copies in flight get a grace time at shutdown). It offers no other store, and
    /// a pause does not hold the batches that run the copies.
    closing: bool,
    /// The replies owed to `Pause` once nothing is in flight.
    boundary: Vec<Sender<()>>,
    /// The reply, cause, deadline (none for a complete drain) and whether memory is given back,
    /// of a drain under way.
    draining: Option<(Sender<()>, MigrationCause, Option<Instant>, bool)>,
    /// The warm-up requests the scheduler holds, by the id it gave each, with the daemon's request
    /// of the same id that arrived meanwhile, which waits until the warm-up request is forgotten
    /// ([`Scheduler::warm_up`]).
    warm_ups: HashMap<RequestId, Option<Deferred>>,
    /// The region, by number, of each `Resize` outstanding that is not a drain's, in the order
    /// submitted, with the reply owed at its `Resized` with the units the region reached: the
    /// engine's, or none for the scheduler's own shrink toward the region's bound. The buffer
    /// pool answers the `Resize`s of one queue pair in the order it takes them.
    resizing: Vec<(u8, Option<Sender<u32>>)>,
    /// The replies owed to `Message::Bound`, each with its region by number: answered once a
    /// `Resized` states that the region holds no more than its bound, or when a drain ends the
    /// bound.
    bounded: Vec<(u8, Sender<u32>)>,
    /// The replies owed to `Message::Demote` whose `Tail` is outstanding, each with its region
    /// by number, in the order submitted: the buffer pool answers the requests of one queue pair
    /// in the order it takes them.
    tails: Vec<(u8, Sender<u32>)>,
    /// The replies owed to `Message::Demote`: the keys of the host stores each demotion issued
    /// and the blocks it demoted, answered once no store of those keys is held or outstanding.
    demoting: Vec<(std::collections::HashSet<storage::Key>, u32, Sender<u32>)>,
    next_batch: u64,
    step: u64,
    /// The scheduler's data-parallel rank, its index in the session, which its spans name and
    /// whose role the step values state.
    index: u32,
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
        generations: Arc<Generations>,
        tuning: Arc<ArcSwap<Tuning>>,
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
            workers.push(WorkerGroup::new(config, capacity, group.clone())?);
        }
        let given = config.clone();
        let config = resolved(config, &workers)?;

        // A scheduler that holds no request and knows no cached block yet is told its groups.
        let bounds = ceilings(
            workers.iter().map(|worker| &worker.capacity),
            &config.policy_budget,
        );
        let snapshot = view::snapshot_of(&workers);
        let policy = Policy::new(
            module,
            parameters,
            snapshot,
            &bounds.limits,
            &config.policy_budget,
        )
        .map_err(Error::Policy)?;
        let budget = Budget::of(&tuning.load(), &bounds, &config);

        // The groups of a media deployment serve parts of one pipeline, whose products the
        // session classes in one order: the groups that state a pipeline state the same.
        let stated = || workers.iter().map(|worker| &worker.capacity);
        agreed(stated().map(|capacity| &capacity.pipeline))
            .ok_or_else(|| differ("media pipelines"))?;

        // Owner numbers carry the data-parallel rank in their top 16 bits, so the schedulers of
        // one session never draw the same owner; the buffer pool takes an owner number only
        // above every one it took, so neither counter starts at 0.
        let rank = u64::try_from(scheduler).unwrap_or(u64::MAX) & 0xffff;
        let index = u32::try_from(rank).unwrap_or(u32::MAX);
        let kv = Kv::new(
            layout,
            opened,
            &registers,
            queue.max_record(),
            &workers,
            index,
        )?;
        let limits = GenerationLimits {
            features: request::served_features(&workers),
            ..limits
        };
        let mut scheduler = Self {
            config,
            stated: given,
            tuning,
            budget,
            ceilings: bounds,
            limits,
            epoch: layout.epoch,
            session: opened.session,
            requests: HashMap::new(),
            waiting: VecDeque::new(),
            running: Vec::new(),
            groups: workers,
            queue,
            seen: Signals::default(),
            held: Vec::new(),
            stamping: HashMap::new(),
            outstanding: HashMap::new(),
            next_submission: 1,
            registers,
            kv,
            owners: HashMap::new(),
            next_owner: (rank << 48) + 1,
            short: HashMap::new(),
            copies: HashMap::new(),
            unmapped: HashMap::new(),
            released: vec![false; layout.regions.len()],
            policy,
            executor,
            output,
            outbox: None,
            stats: stats::empty(),
            published: Instant::now(),
            generations,
            paused: false,
            stopped: false,
            closing: false,
            boundary: Vec::new(),
            draining: None,
            warm_ups: HashMap::new(),
            resizing: Vec::new(),
            bounded: Vec::new(),
            tails: Vec::new(),
            demoting: Vec::new(),
            // Batch numbers carry the data-parallel rank from bit 40, so the batches of one
            // process's schedulers, and the spans that name them, never share a number.
            next_batch: rank << 40,
            step: 0,
            index,
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
    /// the slots its groups hold.
    pub(crate) fn max_running_requests(&self) -> usize {
        self.config.max_running_requests
    }

    /// Numbers the scheduler `ordinal` across the deployment (`EngineArgs::replica` plus its
    /// index in its engine): the offset of the ports its own endpoints bind
    /// (`kv_events_config.endpoint`) and the rank its KV event batches state. Its index in its
    /// engine until numbered.
    pub(crate) fn number(&mut self, ordinal: u32) {
        self.kv.events().number(ordinal);
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
                    .any(|encoder| group::encodes_on_host(groups, &decoder.config, &encoder.config))
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

    let ceilings = ceilings(
        groups.iter().map(|group| &group.capacity),
        &config.policy_budget,
    );
    let tokens = usize::try_from(ceilings.tokens).unwrap_or(usize::MAX);
    config.max_num_batched_tokens = config.max_num_batched_tokens.clamp(1, tokens.max(1));
    if let Some(rows) = NonZeroUsize::new(ceilings.rows) {
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
    /// [`Scheduler::step`]. It blocks only after a step that formed no batch: a formed batch may
    /// have left room for another (a pipelined decode), which the next pass finds without
    /// waiting. It then declares on the queue pair what it waits for there
    /// ([`Scheduler::interest`], [`QueuePair::arm`]) and waits until the next message, the
    /// buffer pool's ring of its doorbell, or the next deadline of [`Scheduler::next_deadline`];
    /// when the buffer pool published something it waits for since the step looked, it takes the
    /// next pass at once instead.
    ///
    /// The doorbell reaches the loop as a message of its own channel: a relay thread blocks on
    /// the doorbell's descriptor, which the inbox cannot be waited on with, and forwards each
    /// ring ([`Relay`]). A relay that cannot start is a fault of the scheduler.
    pub(crate) fn run(mut self, inbox: Receiver<Message>) {
        // The spans this thread records name the scheduler.
        service::span::label(service::span::Label::Scheduler(self.index));
        let relay = match Relay::start(&self.queue) {
            Ok(relay) => relay,
            Err(error) => {
                self.fail(Error::Storage(format!(
                    "the scheduler's doorbell relay cannot start: {error}"
                )));
                return;
            }
        };

        let mut ready = false;
        while !self.stopped {
            let first = if ready {
                match inbox.try_recv() {
                    Ok(message) => Some(message),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => return self.abandon(),
                }
            } else {
                let deadline = self.next_deadline();
                let received = crossbeam_channel::select! {
                    recv(inbox) -> message => message.map(Some),
                    recv(relay.rings) -> _ => Ok(None),
                    default(deadline.saturating_duration_since(Instant::now())) => Ok(None),
                };
                // Whatever ended the wait, the buffer pool rings for nothing more until the
                // next declaration.
                self.queue.disarm();
                match received {
                    Ok(message) => message,
                    Err(_) => return self.abandon(),
                }
            };

            for message in first.into_iter().chain(inbox.try_iter()) {
                self.apply(message);
                if self.stopped {
                    return;
                }
            }
            let formed = self.step();
            // A step that formed no batch is followed by a wait, unless the buffer pool published
            // something the scheduler waits for since the step looked at the pair.
            ready = formed || (!self.stopped && !self.queue.arm(self.interest(), self.seen));
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
            Message::AddRequest(input, window) => self.add_request(input, window),
            Message::WarmUp(input, window) => self.warm_up(*input, window),
            // The daemon's messages name its own requests: never a warm-up request, whose id a
            // request of the daemon may take only once it waits for the warm-up's end.
            Message::Cancel(id, reason) => match self.warm_ups.get_mut(&id) {
                Some(Some(deferred)) => deferred.ended = Some(reason),
                Some(None) => {}
                None => self.cancel(id, reason),
            },
            Message::Acknowledge(id, tokens) if !self.warm_ups.contains_key(&id) => {
                self.acknowledge(id, tokens);
            }
            Message::Consumed(id, events) if !self.warm_ups.contains_key(&id) => {
                self.consumed(id, events);
            }
            Message::Acknowledge(..) | Message::Consumed(..) => {}
            // The requests of each completed batch advance.
            Message::Output(batch, output) => self.update(batch, output),
            // The executor's stamped requests join the step's requests, behind those held.
            Message::Submit(requests) => self.stamped(requests),
            Message::Failed(error) => self.fail(error),
            Message::Path(id, path) => self.routed(id, path),
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
            Message::SetGraphs(limits, reply) => {
                self.set_graphs(&limits);
                let _ = reply.send(());
            }
            Message::Flush(reply) => {
                self.flush();
                let _ = reply.send(());
            }
            Message::Resize(region, units, reply) => self.resize(region, units, reply),
            Message::Bound(region, units, reply) => self.bound(region, units, reply),
            Message::Demote(region, units, reply) => self.demote(region, units, reply),
            Message::Shutdown => self.shutdown(),
        }
    }

    /// Forms every batch from the next step on within `limits`, a kernel generation's graphs:
    /// each group the limits name captured their shapes, which bound its batches' rows and its
    /// call window; the scheduler's configuration is `limits`' lowered to the groups again, so
    /// the step's tokens and calls and the graph tiers are the generation's, and so are the
    /// drafts of a verification, which no request held here exceeds from now on; the ceilings
    /// follow the shapes; and a request may ask for the generation's candidates at a position.
    /// Each group serves the phases the limits state for it, so its decode steps are
    /// verifications under drafts and decodes under none; a generation of no drafts gives up
    /// every request's draft cache, which no step reads from then on. A configuration the
    /// groups refuse leaves the one that served, which the generation's preparation checked.
    fn set_graphs(&mut self, limits: &GraphLimits) {
        let speculated = self.config.draft_tokens > 0;
        for group in &mut self.groups {
            if let Some((_, phases)) = limits.phases.iter().find(|(id, _)| *id == group.layout.id) {
                group.capacity.phases = *phases;
            }
            let Some((_, buckets)) = limits.buckets.iter().find(|(id, _)| *id == group.layout.id)
            else {
                continue;
            };
            group.capacity.buckets.clone_from(buckets);
            let calls = buckets.iter().map(|shape| shape.rows).max();
            let window = crate::memory::division::call_window(
                group.config.queue_depth,
                calls.unwrap_or(1).max(1) as usize,
            );
            if let Ok(window) = window {
                group.capacity.call_window = u32::try_from(window).unwrap_or(u32::MAX);
            }
        }

        let mut stated = limits.config.clone();
        stated.report_period = self.config.report_period;
        match resolved(stated.clone(), &self.groups) {
            Ok(config) => {
                self.stated = stated;
                self.config = config;
            }
            Err(error) => {
                tracing::error!(%error, "a generation's graphs give no configuration");
            }
        }
        self.ceilings = ceilings(
            self.groups.iter().map(|group| &group.capacity),
            &self.config.policy_budget,
        );
        self.limits.max_logprobs = limits.max_logprobs;

        // A verification carries at most the generation's drafts, which its graphs hold.
        let drafts = u32::try_from(self.config.draft_tokens).unwrap_or(u32::MAX);
        for request in self.requests.values_mut() {
            request.draft_limit = request.draft_limit.min(drafts);
        }
        self.replace_steps();

        // The boundary holds no call in flight, so a draft table no step reads any more is
        // released whole; a later generation of drafts refills the cache from its first block.
        if speculated && drafts == 0 {
            let drafting: Vec<RequestId> = self
                .requests
                .values()
                .filter(|request| {
                    kv::draft_class(request, &self.groups)
                        .and_then(|(class, _)| request.blocks.get(class))
                        .is_some_and(|table| table.first < table.end)
                })
                .map(|request| request.id)
                .collect();
            for id in drafting {
                self.give_up_draft(id);
            }
        }
    }

    /// Places every request whose decode steps ran on a group that now serves them as the other
    /// phase on that phase of the same group, which holds its slot and its cache: a group whose
    /// decodes became verifications runs its requests' next steps as verifications and the
    /// refills of their draft caches, and one whose verifications became decodes runs decodes.
    fn replace_steps(&mut self) {
        let groups = &self.groups;
        let serves = |id: GroupId, phase: Phase| {
            groups
                .iter()
                .any(|group| group.id() == id && group.serves(phase))
        };

        for request in self.requests.values_mut() {
            let placement = &mut request.placement;
            for (from, to) in [
                (Phase::Decode, Phase::Verify),
                (Phase::Verify, Phase::Decode),
            ] {
                let Some(group) = placement[from.index()] else {
                    continue;
                };
                if serves(group, from) || !serves(group, to) {
                    continue;
                }
                placement[from.index()] = None;
                placement[to.index()] = Some(group);
                let refill = Phase::Refill.index();
                if to == Phase::Verify
                    && placement[refill].is_none()
                    && serves(group, Phase::Refill)
                {
                    placement[refill] = Some(group);
                }
            }
        }
    }

    /// Takes the step values of `tuning` the scheduler keeps beside the step's budget: the
    /// report period of its statistics, and each group's queue depth with the call window it
    /// bounds a request's unresolved calls to (`crate::memory::division::call_window`, of the
    /// group's widest bucket, as its ranks stated it). A group that has more batches in flight
    /// than its new depth submits none until they end.
    fn retune(&mut self, tuning: &Tuning) {
        self.config.report_period = tuning.report_period;
        self.kv.events().configure(&tuning.kv_events);
        for group in &mut self.groups {
            let Some(depth) = tuning.queue_depths.get(group.config.id.0.as_str()) else {
                continue;
            };
            if group.config.queue_depth == depth.get() {
                continue;
            }
            group.config.queue_depth = depth.get();
            let calls = group.capacity.buckets.iter().map(|shape| shape.rows).max();
            let window = crate::memory::division::call_window(
                depth.get(),
                calls.unwrap_or(1).max(1) as usize,
            );
            if let Ok(window) = window {
                group.capacity.call_window = u32::try_from(window).unwrap_or(u32::MAX);
            }
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
    /// 4. a drain's deadline ends what it ends, a region past its bound is asked to shrink again
    ///    once something there may leave ([`Scheduler::shrink_to_bounds`]), `Pause` and `Drain`
    ///    are answered once due ([`Scheduler::reply_at_boundary`]), and the statistics leave once
    ///    a second has passed.
    ///
    /// True when it formed a batch. A stopped scheduler takes no step.
    pub fn step(&mut self) -> bool {
        if self.stopped {
            return false;
        }
        let _range = crate::profiling::ranges::range(c"uniserve.scheduler.step", None);

        // The step values this step reads, whatever arrives while it runs: the snapshot the
        // control thread replaces is taken once here.
        let tuning = self.tuning.load_full();
        self.budget = Budget::of(&tuning, &self.ceilings, &self.config);
        self.retune(&tuning);
        self.answer_demotions();

        // What the buffer pool published before this look at the pair: anything it publishes
        // later keeps the loop from sleeping after the step (`run`).
        self.seen = self.queue.signals();

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
        // A request whose prompt a prefill replica computed for another replica hands its KV off
        // once nothing of it is in flight.
        self.start_handoffs();

        let admitted = self.running.len().saturating_sub(running);
        // A closing scheduler forms the batches that run its copies whether or not it is paused:
        // it holds no request, and a parked epoch's stores end only so.
        let batches = if self.paused && !self.closing {
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
        self.shrink_to_bounds();
        self.reply_at_boundary();
        self.publish_stats(now);
        // What the step changed of the cache leaves as one batch of KV events.
        self.kv.events().flush();
        self.settle_shutdown();
        formed
    }

    /// Until when the loop may block after a step that formed no batch: the next statistics
    /// second; while a step can judge starvation, the moment the first running request passes its
    /// starvation bound (the policy's `timeout` from its last progress); a drain's deadline still
    /// ahead; and the earliest bound of a copy in flight. What changes on the queue pair rings the
    /// doorbell instead ([`Scheduler::interest`]).
    ///
    /// A step judges starvation only while it schedules: unpaused, with room in some group's
    /// queue (`schedule`). Otherwise the bound waits for the message that makes the next step
    /// judge it: the output of a batch, which frees its group's queue, or `Resume`.
    fn next_deadline(&self) -> Instant {
        let now = Instant::now();
        let mut deadline = self.stats_due();

        let judges = !self.paused && self.groups.iter().any(schedule::queue_open);
        // A request waiting for the buffer pool's import or lend is exempt from the bound.
        let timeout = Duration::from_nanos(self.policy.limits.timeout.0);
        let starves = self
            .running
            .iter()
            .filter_map(|id| self.requests.get(id))
            .filter(|request| request.status == RequestStatus::Running)
            .filter_map(|request| request.last_progress.checked_add(timeout))
            .min();
        if let Some(starves) = starves.filter(|_| judges) {
            deadline = deadline.min(starves);
        }

        // A deadline already passed was met by the step that just ran.
        if let Some((_, _, Some(due), _)) = &self.draining
            && *due > now
        {
            deadline = deadline.min(*due);
        }

        // A copy handed to the executor is given up at its bound if it has not finished by then
        // (`give_up_late_copies`).
        if let Some(bound) = self.copies.values().map(|&(_, bound)| bound).min() {
            deadline = deadline.min(bound);
        }
        deadline
    }

    /// What the loop waits for on the queue pair while it sleeps (BP §9, doorbells): what
    /// changes there without a message to wake the loop. Completions while the buffer pool owes
    /// answers to requests written; room in the submission queue while requests are held for it
    /// (a request held for room in the completion queue waits for the answers outstanding); and,
    /// unless the scheduler is paused, the capacity words while a request may wait for units of
    /// a region: one in the waiting queue, whose admission waits for `available`, or a running
    /// request with no call in flight, which may wait for the units its next call needs. Nothing
    /// otherwise, and the buffer pool then rings for nothing.
    fn interest(&self) -> Interest {
        let mut interest = Interest::empty();
        if !self.outstanding.is_empty() {
            interest |= Interest::RECORDS;
        }
        if !self.held.is_empty() {
            interest |= Interest::ROOM;
        }
        let units = !self.paused
            && (!self.waiting.is_empty()
                || self
                    .running
                    .iter()
                    .filter_map(|id| self.requests.get(id))
                    .any(Request::idle));
        if units {
            interest |= Interest::REGISTERS;
        }
        interest
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
    /// has left (a migratable request's last blocks); a `Lend`, whose owner's blocks return once
    /// its lend ends; an `Import`, whose blocks return when its task fails; and every copy the
    /// engine runs, whose blocks the buffer pool holds until
    /// its `Finish` (a `Store` to the host reads its block's device copy). A `Store` to the
    /// remote store and a `Prefetch` read and fill the host tier alone.
    fn gives_back(request: &runtime::Request) -> bool {
        match request {
            runtime::Request::Detach { .. }
            | runtime::Request::Release { .. }
            | runtime::Request::Evict { .. }
            | runtime::Request::Commit { .. }
            | runtime::Request::Lend { .. }
            | runtime::Request::Import { .. }
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
            | runtime::Request::Tail { .. }
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

    /// `Flush` (SD §2.4): every cached block the scheduler knows of is evicted; the requests it
    /// holds continue with what they hold ([`Kv::flush`]).
    fn flush(&mut self) {
        for evict in self.kv.flush() {
            self.submit(evict);
        }
    }

    /// Stops the scheduler on a fault it cannot recover from (SD §3.7). A `Finished` held for the
    /// daemon's decoder is sent as it is, since that generation ended; then `Output::Failed`
    /// tells the engine; then every request whose terminal has not been sent leaves as
    /// `Migratable { Loss }`, which the daemon resubmits to the engine that serves next: the
    /// engine tells its daemons that it failed before any request leaves it, so that none is
    /// submitted to it again. Each request gives up what it holds as far as the workers still
    /// take it, and the loop ends. A failed executor cannot stamp its group's `Detach`: the
    /// buffer pool detaches the epoch's owners when its session closes (BP §12.4). Called once;
    /// later faults are dropped.
    pub(super) fn fail(&mut self, error: Error) {
        if self.stopped {
            return;
        }
        self.stopped = true;

        let ids: Vec<RequestId> = self.requests.keys().copied().collect();
        let mut lost = Vec::new();
        for id in ids {
            let Some(request) = self.requests.get_mut(&id) else {
                continue;
            };
            let held = request.terminal.take();
            let sent = request.status == RequestStatus::Finished && held.is_none();
            request.status = RequestStatus::Finished;

            match held {
                Some(finished @ Event::Finished { .. }) => self.send(id, finished),
                _ if !sent => lost.push(id),
                _ => {}
            }
            self.free(id);
        }

        // An engine that no longer listens has stopped this scheduler's epoch already.
        self.emit(Output::Failed(error));
        for id in lost {
            self.send(id, update::migratable(MigrationCause::Loss));
        }
    }

    /// `Shutdown`: every request the scheduler holds ends, aborted, at once
    /// (`S/control.rs:73-102`), and the scheduler stops once the copies its workers hold and the
    /// stores to the host it offered have ended ([`Scheduler::settle_shutdown`]): the host copies
    /// in flight get their grace time (BP §20). The steps go on meanwhile, paused or not, to read
    /// the answers and hand the copies to the workers, but offer nothing more to the host; each
    /// copy's own bound ends the wait for it (SD §4.4).
    fn shutdown(&mut self) {
        self.abort_all();
        self.closing = true;
        self.settle_shutdown();
    }

    /// No sender of the inbox is left: the engine that owns the scheduler has gone, and no step
    /// of its workers runs a copy any more. Every request ends, aborted, and the scheduler stops
    /// at once.
    fn abandon(&mut self) {
        self.abort_all();
        self.stopped = true;
    }

    /// Ends every request the scheduler holds, aborted. A terminal still held, behind calls in
    /// flight or the daemon's decoder, is sent now: the request computes nothing more.
    fn abort_all(&mut self) {
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
    }

    /// Stops a closing scheduler once nothing it waits for at shutdown is left: no copy its
    /// workers hold, and no `Store` to the host and no copy's `Finish` held, being stamped or
    /// outstanding. A `Store` is answered `Copy`, its copy runs, and the copy's `Finish` is
    /// answered with what was stored.
    fn settle_shutdown(&mut self) {
        let waits = |request: &runtime::Request| {
            matches!(
                request,
                runtime::Request::Store {
                    tier: Tier::Host,
                    ..
                } | runtime::Request::Finish { .. }
            )
        };
        let storing = !self.copies.is_empty()
            || self.held.iter().any(waits)
            || self
                .stamping
                .values()
                .any(|(request, _, following)| waits(request) || following.iter().any(waits))
            || self
                .outstanding
                .values()
                .any(|(_, request, _)| waits(request));
        if self.closing && !storing {
            self.stopped = true;
        }
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
    /// - A `Release`, a `Commit`, a `Detach`, a `Lend` or a `Replicate` is handed to
    ///   `Executor::stamp` of every group whose device block tables may hold the entries it
    ///   names: the group of the owner's attachment, and every other group the request is
    ///   attached to that reads the class (one that states a region of the class that the
    ///   owner's group states too), and for a `Replicate` the group it copies into. The
    ///   groups are found from the request's `attachments`, so an owner's `Detach` is submitted
    ///   before its attachment is cleared. A `Finish { ok: false }` that gives a copy
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
        self.kv.submitted(&request);

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
            runtime::Request::Detach { owner, .. } | runtime::Request::Lend { owner, .. } => {
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
    /// with it, but a `Finish`, which is written as soon as the submission queue takes it: it
    /// takes no room of its own, since its answer is paid for by its copy's request
    /// (`answer_bytes`). The room is made by the answers outstanding, and an answer may wait for
    /// a copy to end: a request answered `Copy` owes until the answer to its `Finish`, and an
    /// import until its task ends. A `Finish` held
    /// behind a request that waits for room (a `List`, which owes the whole ring, waits for every
    /// answer), or beside a copy whose request owes the whole ring, would hold back the very
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
    /// a completion that gives entries when a `Release` or a `Commit` naming them
    /// was handed to the executor after its request was submitted (BP §9, obligation 1). A
    /// dropped `Copy` is given up at once with `Finish { ok: false }` through its group's
    /// executor; any other enters `copies` until it ends. A `Refused` with the cause `copy`
    /// answers a `Finish` for a copy another `Finish` ended, and is dropped.
    ///
    /// Before it is returned, a completion that gives entries is handed to `Executor::map`, with
    /// the request as it was written: to the group an `Attach` names, to a copy's
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
            let applied = kv::covered(&request, &completion);
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
    /// and `applied` the request as it is applied ([`kv::covered`]); `bytes` is what it owed
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
                | Completion::Imported { .. }
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
            let bound = Instant::now() + self.config.copy_timeout(tokens);
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
            runtime::Request::Store { .. } => self.kv.class_of(key),
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

    /// Reads the capacity words of every region this queue pair uses into the KV decisions
    /// ([`Kv::set_words`]).
    fn read_words(&mut self) {
        let words: Vec<(usize, [u32; 3])> = self
            .registers
            .capacity
            .iter()
            .filter_map(|&(number, word)| {
                let region = self.kv.region_index(number)?;
                let first = usize::try_from(word).unwrap_or(usize::MAX);
                let word = |at: usize| self.word(first.saturating_add(at));
                Some((region, [word(0), word(1), word(2)]))
            })
            .collect();
        for (region, words) in words {
            self.kv.set_words(region, words);
        }
    }

    /// Reads the capacity words of every region this queue pair uses and sets what each region
    /// has available to the scheduler's allocations ([`Kv::read_capacity`]): the buffer pool's
    /// word, lowered by the units of the scheduler's `Allocate`s, `Load`s and `Clone`s held,
    /// being stamped or outstanding, which the buffer pool's word may not show yet (a `Load` or a
    /// `Clone` answered `Copy` took its blocks then, and the word shows them), and by what it
    /// counts as evictable of the blocks its `Pin`s held, being stamped or outstanding name
    /// (`Kv::evictable`), which no allocation can evict once the buffer pool has taken them.
    fn read_capacity(&mut self) {
        self.read_words();
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
        self.kv.read_capacity(&asked);
    }

    /// Tells the KV decisions what the completion of `request` states of the cache
    /// ([`Kv::learn`]), with the request of the owner it acts for, and counts the blocks it
    /// evicted. The answer to a `Finish` is the result of its copy's request, which
    /// `outstanding` keeps, with the copy's group as its stamp; one whose copy is not outstanding
    /// states nothing.
    fn learn(&mut self, request: &runtime::Request, completion: &Completion) {
        self.read_words();
        let copied;
        let (asked, copy) = match request {
            runtime::Request::Finish { copy, after, .. } => match self.outstanding.get(&copy.0) {
                Some((_, request, _)) => {
                    copied = request.clone();
                    (&copied, Some(kv::stamped_groups(after)))
                }
                None => return,
            },
            request => (request, None),
        };
        let holder = owner_of(asked)
            .and_then(|owner| self.owners.get(&owner))
            .and_then(|id| self.requests.get(id));
        let told = &mut self.policy.events;
        let evicted = self
            .kv
            .learn((asked, copy), holder, completion, &self.groups, told);
        self.count_evicted(evicted);
    }

    /// Installs the head of each region's eviction order the step's requests may take, ahead of
    /// every request not yet written, whatever step held it ([`Kv::victims`]): each `Victims`
    /// replaces one of its region not yet written, which would replace it in turn. Then asks for
    /// the `List` the KV decisions find due.
    pub(super) fn install_victims(&mut self) {
        self.read_words();
        // Per region, the units the requests not yet written take: those held and those waiting
        // behind a `Release` being stamped.
        let mut taking = vec![0u32; self.kv.regions().len()];
        let waiting = self
            .stamping
            .values()
            .flat_map(|(_, _, following)| following.iter());
        for request in self.held.iter().chain(waiting) {
            for (region, units) in self.taken(request) {
                if let Some(taken) = taking.get_mut(region) {
                    *taken = taken.saturating_add(units);
                }
            }
        }
        // The blocks the requests outstanding may have made shared and unpinned before their
        // answers are read: a `Commit`'s block, a `Store`'s and a `Prefetch`'s keys. The
        // scheduler's own requests thus never make it `List`.
        let pending = self
            .outstanding
            .values()
            .map(|(_, request, _)| match request {
                runtime::Request::Commit { .. } => 1,
                runtime::Request::Store {
                    keys,
                    tier: Tier::Host,
                    ..
                }
                | runtime::Request::Prefetch { keys, .. } => keys.len(),
                _ => 0,
            })
            .sum::<usize>();
        let pending = u32::try_from(pending).unwrap_or(u32::MAX);
        let shrinking = self.draining.as_ref().is_some_and(|(.., shrink)| *shrink);

        let (installing, list) = self.kv.victims(
            (&taking, pending),
            (shrinking, self.running.len()),
            &self.config,
            &self.groups,
        );
        let replaced = |held: &runtime::Request| {
            installing.iter().any(|victims| match (held, victims) {
                (
                    runtime::Request::Victims { region: held, .. },
                    runtime::Request::Victims { region, .. },
                ) => held == region,
                _ => false,
            })
        };
        self.held.retain(|held| !replaced(held));
        self.held.splice(0..0, installing);
        if list {
            self.submit(runtime::Request::List);
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

    /// The register at `index` as a count of units.
    fn word(&self, index: usize) -> u32 {
        let value = self.queue.register(index).load(Ordering::Acquire);
        u32::try_from(value).unwrap_or(u32::MAX)
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
            None => 0..u8::try_from(self.kv.classes().len()).unwrap_or(u8::MAX),
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
            (runtime::Request::Attach { group, .. }, Completion::Attached { slot }) => {
                vec![(*group, Ok(*slot))]
            }
            (runtime::Request::Import { owner, .. }, Completion::Imported { .. }) => self
                .holders(*owner, None)
                .into_iter()
                .map(|group| (group, self.slot_of(Some(*owner), group)))
                .collect(),
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

    /// Whether a `Release`, a `Commit` or a `Replicate` naming entries `request`
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
    /// from the host fills (of an encoder output, its size as its request computes it), and a
    /// `Clone` the one block it copies into; a `Pin` what `available` counts as evictable of the
    /// blocks it names until its `Pinned` is read (`evictable`); a `Replicate` the runs it takes
    /// in the receiving regions of the group it copies into; and from its class's host region, a
    /// `Prefetch` a host block per key, which it brings the key into.
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
        // An import takes the blocks of every table it names.
        if let runtime::Request::Import { owner, tables, .. } = request {
            let mut taken = Vec::new();
            for (class, _, count) in tables {
                let each = self.kv.block_units(*class, 0).unwrap_or(0);
                for region in self.class_regions(*owner, *class).unwrap_or_default() {
                    taken.push((region, count.saturating_mul(each)));
                }
            }
            return taken;
        }
        if let runtime::Request::Prefetch {
            class, units, keys, ..
        } = request
        {
            let count = u32::try_from(keys.len()).unwrap_or(u32::MAX);
            return self
                .kv
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
                owner,
                class,
                at,
                keys,
            } => {
                // A sized block takes its size, which its request computes (SD §4.4 "Classes").
                let size = self.request_of(*owner).and_then(|row| {
                    let (group, _) = row.attachment(*owner)?;
                    self.kv
                        .encoding_size(row, (*class, *at), group, &self.groups)
                });
                let count = u32::try_from(keys.len()).unwrap_or(u32::MAX);
                (*owner, *class, count, size.unwrap_or(0))
            }
            runtime::Request::Clone { owner, class, .. } => (*owner, *class, 1, 0),
            runtime::Request::Pin {
                owner,
                class,
                at,
                keys,
            } => {
                let Some(row) = self.request_of(*owner) else {
                    return Vec::new();
                };
                let Some((group, _)) = row.attachment(*owner) else {
                    return Vec::new();
                };
                let Some(holder) = self.group(group) else {
                    return Vec::new();
                };
                let positions = (*at..).zip(keys);
                return positions
                    .flat_map(|(at, key)| {
                        let size = self
                            .kv
                            .encoding_size(row, (*class, at), group, &self.groups);
                        self.kv.evictable(key, holder, size.unwrap_or(0))
                    })
                    .collect();
            }
            _ => return Vec::new(),
        };
        let Some(regions) = self.class_regions(owner, class) else {
            return Vec::new();
        };

        let each = self.kv.block_units(class, units).unwrap_or(0);
        regions
            .into_iter()
            .map(|region| (region, count.saturating_mul(each)))
            .collect()
    }

    /// The group of `owner`'s attachment.
    fn owner_group(&self, owner: OwnerId) -> Option<&WorkerGroup> {
        let (group, _) = self.request_of(owner)?.attachment(owner)?;
        self.group(group)
    }

    /// The regions, as indices, owner `owner`'s blocks of `class` are taken from: those of the
    /// class in the group of the owner's attachment. `None` when the owner is unknown.
    fn class_regions(&self, owner: OwnerId, class: u8) -> Option<Vec<usize>> {
        Some(Kv::class_regions(self.owner_group(owner)?, class))
    }

    /// The most bytes the completion of `request` takes in the completion queue: the largest
    /// record any of its answers can have (BP §11), padded, with its length word, and at most a
    /// whole ring. An answer's lists are bounded by what the request names: a run per block and
    /// region, and an evicted key per unit the request takes, since every evicted block frees at
    /// least one. A `Listing`, and an answer whose lists the scheduler cannot bound, may take the
    /// whole ring. A `Finish` takes nothing beyond what its copy's request owes: that request
    /// stays outstanding, owing its bound, until the answer to the `Finish` is read, and its
    /// bound is at least that answer (a `Copy` record is larger than any answer to a `Finish`).
    fn answer_bytes(&self, request: &runtime::Request) -> usize {
        // A `Finish` must fit whatever its copy's request owes, even the whole ring (a request
        // of an owner detached before it was written, whose answer has no bound): its answer is
        // what releases that debt, so reserving room beside it would hold the `Finish`, and
        // every request behind it, for good.
        if let runtime::Request::Finish { copy, .. } = request
            && self.outstanding.contains_key(&copy.0)
        {
            return 0;
        }

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
        let shape = |owner: OwnerId, class: u8, units: u32| {
            self.kv.block_shape(self.owner_group(owner)?, class, units)
        };

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
                let each = self
                    .kv
                    .host_block(*class, *units)
                    .map_or(0, |(_, each)| each);
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
            runtime::Request::Store { group, keys, .. } => self
                .group(*group)
                .and_then(|group| self.kv.store_bytes(group, keys, &copy)),
            // From the slot's row to one host block of `units`.
            runtime::Request::Save { units, .. } => Some(copy(1, 1, 1, *units as usize)),
            runtime::Request::Restore { .. } => Some(copy(1, 1, 1, 0)),
            // A run per block of its tables, and an evicted key per unit they take.
            runtime::Request::Import { owner, tables, .. } => {
                tables
                    .iter()
                    .try_fold(LIST + LIST, |bytes, (class, _, count)| {
                        let (runs, units) = shape(*owner, *class, 0)?;
                        let runs = runs.saturating_mul(*count as usize);
                        Some(
                            bytes
                                + RUN.saturating_mul(runs)
                                + KEY.saturating_mul(runs.saturating_mul(units)),
                        )
                    })
            }
            runtime::Request::Lend { .. } => Some(0),
            // A shrink evicts the blocks in the region's tail, at most one a unit there.
            runtime::Request::Resize { region, units } => self
                .kv
                .region_index(*region)
                .map(|index| self.kv.tail_units(index, *units))
                .map(|tail| 4 + LIST + KEY * tail as usize),
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
            // Every cached block in the region's tail, which other queue pairs' blocks share.
            runtime::Request::List | runtime::Request::Tail { .. } => None,
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
}

/// The scheduler's side of its KV decisions (`kv`): each call resolves the request a decision
/// is for, states the scheduler's step ([`Step`]) and the policy's event sink, and submits what
/// the decision asks of the buffer pool. The decisions themselves are [`Kv`]'s.
impl Scheduler {
    /// What the scheduler states of itself to a KV decision at this step.
    pub(super) fn kv_step(&self) -> Step<'_> {
        Step {
            groups: &self.groups,
            config: &self.config,
            budget: &self.budget,
            prefetch_min_tokens: self.policy.limits.prefetch_min_tokens,
        }
    }

    /// The first round of request `id`'s admission ([`Kv::admission_requests`]); a video
    /// request's crossings follow its placement.
    pub(super) fn admission_requests(&mut self, id: RequestId) -> Vec<runtime::Request> {
        let crossings = match self.requests.get(&id) {
            Some(request) if request.is_video() => Some(self.crossings_of(request)),
            Some(_) => None,
            None => return Vec::new(),
        };
        let step = Step {
            groups: &self.groups,
            config: &self.config,
            budget: &self.budget,
            prefetch_min_tokens: self.policy.limits.prefetch_min_tokens,
        };
        let Some(request) = self.requests.get_mut(&id) else {
            return Vec::new();
        };
        if let Some(crossings) = crossings {
            request.crossings = crossings;
        }
        self.kv.admission_requests(request, &step)
    }

    /// One more round of request `id`'s admission ([`Kv::missing_requests`]).
    pub(super) fn missing_requests(&mut self, id: RequestId) -> Vec<runtime::Request> {
        let step = Step {
            groups: &self.groups,
            config: &self.config,
            budget: &self.budget,
            prefetch_min_tokens: self.policy.limits.prefetch_min_tokens,
        };
        let Some(request) = self.requests.get_mut(&id) else {
            return Vec::new();
        };
        let told = &mut self.policy.events;
        self.kv.missing_requests(request, &step, told)
    }

    /// What the answer `completion` to `asked`, one of request `id`'s memory requests, states of
    /// its tables, and the requests it calls for ([`Kv::filled`]).
    pub(super) fn filled(
        &mut self,
        id: RequestId,
        asked: &runtime::Request,
        completion: &Completion,
    ) -> Vec<runtime::Request> {
        let step = Step {
            groups: &self.groups,
            config: &self.config,
            budget: &self.budget,
            prefetch_min_tokens: self.policy.limits.prefetch_min_tokens,
        };
        let Some(request) = self.requests.get_mut(&id) else {
            return Vec::new();
        };
        let told = &mut self.policy.events;
        self.kv.filled(request, (asked, completion), &step, told)
    }

    /// Whether the admission of request `id`, whose hit `cache` expects to be `expected` tokens,
    /// waits for a `List` ([`Kv::awaits_listing`]), which is asked here when it is due.
    pub(super) fn awaits_listing(&mut self, id: RequestId, expected: u32) -> bool {
        let Some(request) = self.requests.get(&id) else {
            return false;
        };
        let (waits, ask) = self
            .kv
            .awaits_listing(request, expected, &self.groups, &self.budget);
        if ask {
            self.submit(runtime::Request::List);
        }
        waits
    }

    /// Per region, the units the running requests still need to reach their next call and the
    /// low watermark ([`Kv::admission_reserve`]), and the runs a video request's crossings not
    /// asked yet take in their receiving regions.
    pub(super) fn admission_reserve(&self) -> Vec<u32> {
        let running = || self.running.iter().filter_map(|id| self.requests.get(id));
        let mut reserve = self.kv.admission_reserve(running(), &self.kv_step());
        for request in running() {
            for &(at, group, from, state) in &request.crossings {
                if state.is_some() || !request.reads_result(at, &self.groups) {
                    continue;
                }
                for (region, units) in self.receiving(request, (at, group, from)) {
                    if let Some(reserved) = reserve.get_mut(region) {
                        *reserved = reserved.saturating_add(units);
                    }
                }
            }
        }
        reserve
    }

    /// Per region, the units the admission of `request` with a prefix hit of `hit` tokens
    /// allocates ([`Kv::units_needed`]), and for a video request the runs its crossings take.
    pub(super) fn units_needed(&self, request: &Request, hit: u32) -> Vec<u32> {
        let units = self
            .kv
            .units_needed(request, hit, &self.groups, &self.budget);
        self.with_crossings(request, units)
    }

    /// Per region, the units `request`'s admission holds when no other request holds any
    /// ([`Kv::units_alone`]), and for a video request the runs its crossings take.
    pub(super) fn units_alone(&self, request: &Request) -> Vec<u32> {
        let units = self.kv.units_alone(request, &self.groups, &self.budget);
        self.with_crossings(request, units)
    }

    /// `units` with the runs a video request's crossings take, each in every memory its reading
    /// group reads its result in where the block has none.
    fn with_crossings(&self, request: &Request, mut units: Vec<u32>) -> Vec<u32> {
        if !request.is_video() {
            return units;
        }
        for (at, group, from, _) in self.crossings_of(request) {
            for (region, taken) in self.receiving(request, (at, group, from)) {
                if let Some(units) = units.get_mut(region) {
                    *units = units.saturating_add(taken);
                }
            }
        }
        units
    }

    /// Submits the `Commit`s of the blocks the output of request `id`'s call in `batch`'s row
    /// `row` showed full ([`Kv::commit_filled`]).
    pub(super) fn commit_filled(&mut self, id: RequestId, batch: &Batch, row: usize) {
        let Some(request) = self.requests.get_mut(&id) else {
            return;
        };
        let commits = self
            .kv
            .commit_filled(request, (batch, row), &self.groups, &self.budget);
        for commit in commits {
            self.submit(commit);
        }
    }

    /// Submits the `Commit`s of what request `id` holds and its continuation reads as it leaves
    /// as migratable ([`Kv::commit_tails`]).
    pub(super) fn commit_tails(&mut self, id: RequestId) {
        let Some(request) = self.requests.get_mut(&id) else {
            return;
        };
        let commits = self.kv.commit_tails(request, &self.groups, &self.config);
        for commit in commits {
            self.submit(commit);
        }
    }

    /// Submits the `Commit` of the encoder output the call in `batch`'s row `row` computed for
    /// request `id`, which holds it from now on ([`Kv::commit_encoding`]).
    pub(super) fn commit_encoding(&mut self, id: RequestId, batch: &Batch, row: usize) {
        let Some(request) = self.requests.get_mut(&id) else {
            return;
        };
        if let Some(commit) = self.kv.commit_encoding(request, (batch, row), &self.groups) {
            self.submit(commit);
        }
    }

    /// Submits the `Release`s of what no later call of request `id` reads after its call in
    /// `batch`'s row `row` ([`Kv::give_up_behind`]).
    pub(super) fn give_up_behind(&mut self, id: RequestId, batch: &Batch, row: usize) {
        let Some(request) = self.requests.get_mut(&id) else {
            return;
        };
        let told = &mut self.policy.events;
        let context = (&self.groups[..], &self.config);
        let releases = self.kv.give_up_behind(request, (batch, row), context, told);
        for release in releases {
            self.submit(release);
        }
    }

    /// Gives up request `id`'s whole draft table ([`Kv::give_up_draft`]) and returns, per region,
    /// the units the release frees once the buffer pool reclaims them.
    pub(super) fn give_up_draft(&mut self, id: RequestId) -> Vec<u32> {
        let none = vec![0; self.kv.regions().len()];
        let Some(request) = self.requests.get_mut(&id) else {
            return none;
        };
        let told = &mut self.policy.events;
        let Some((releases, freed, blocks)) = self.kv.give_up_draft(request, &self.groups, told)
        else {
            return none;
        };
        self.count_given_up_drafts(blocks);
        for release in releases {
            self.submit(release);
        }
        self.mark(id);
        freed
    }

    /// Request `id` gave up the read-only blocks of `keys` ([`Kv::unpin`]).
    pub(super) fn unpin(&mut self, id: RequestId, keys: &[Key]) {
        let request = self.requests.get_mut(&id);
        let told = &mut self.policy.events;
        self.kv.unpin(request, keys, &self.groups, told);
    }
}

/// The scheduler's one reading of the host's clocks, taken once per process, from which every
/// instant is placed: an `Instant`, the buffer pool's protocol clock in nanoseconds at that
/// instant (`storage::runtime::monotonic_nanos`), and the wall clock in seconds since the Unix
/// epoch. An instant maps to the same value however the wall clock moves after.
fn clocks() -> (Instant, u64, f64) {
    static ORIGIN: OnceLock<(Instant, u64, f64)> = OnceLock::new();
    *ORIGIN.get_or_init(|| {
        let monotonic = storage::runtime::monotonic_nanos();
        let wall = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
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
        | runtime::Request::Lend { owner, .. }
        | runtime::Request::Import { owner, .. }
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
/// positions. A `Lend` and a `Detach` name every entry of their owner.
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
        runtime::Request::Lend { owner, .. } | runtime::Request::Detach { owner, .. } => {
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
        | runtime::Request::Lend { after, .. }
        | runtime::Request::Finish { after, .. }
        | runtime::Request::Replicate { after, .. } => Some(after),
        _ => None,
    }
}

fn storage_error(error: storage::Error) -> Error {
    Error::Storage(error.to_string())
}

/// Forwards the rings of a scheduler's doorbell to its loop ([`Scheduler::run`]).
///
/// The loop waits on its inbox, a channel, and the doorbell is a descriptor, which a channel
/// cannot be waited on together with. The relay's thread blocks on the doorbell, takes each
/// ring and sends it on a channel of its own, which the loop waits on beside its inbox; rings
/// the loop has not taken yet coalesce into one. The thread ends when the relay is dropped:
/// it waits on its end of a socket pair beside the doorbell, whose other end the relay holds.
struct Relay {
    /// One message per ring the loop has not taken yet, at most one.
    rings: Receiver<()>,
    /// Closing it ends the thread.
    stop: std::os::unix::net::UnixStream,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Relay {
    /// Starts the relay of `queue`'s doorbell.
    ///
    /// # Errors
    ///
    /// Fails when the doorbell cannot be duplicated, the socket pair made, or the thread
    /// started.
    fn start(queue: &QueuePair) -> std::io::Result<Self> {
        use std::os::fd::AsRawFd as _;

        let doorbell = queue.doorbell().try_clone_to_owned()?;
        let (stop, stopped) = std::os::unix::net::UnixStream::pair()?;
        let (ring, rings) = crossbeam_channel::bounded(1);
        let thread = std::thread::Builder::new()
            .name("scheduler-doorbell".into())
            .spawn(move || {
                let mut count = [0_u8; 8];
                loop {
                    let mut descriptors = [
                        libc::pollfd {
                            fd: doorbell.as_raw_fd(),
                            events: libc::POLLIN,
                            revents: 0,
                        },
                        libc::pollfd {
                            fd: stopped.as_raw_fd(),
                            events: libc::POLLIN,
                            revents: 0,
                        },
                    ];
                    // SAFETY: poll borrows this live array only for the call.
                    let polled = unsafe { libc::poll(descriptors.as_mut_ptr(), 2, -1) };
                    if polled < 0 {
                        if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted
                        {
                            continue;
                        }
                        return;
                    }
                    // The relay's end closed: the scheduler stopped.
                    if descriptors[1].revents != 0 {
                        return;
                    }
                    if descriptors[0].revents & libc::POLLIN != 0 {
                        // SAFETY: an eventfd read of its eight-byte counter into a live buffer,
                        // which resets it; the descriptor is non-blocking.
                        unsafe { libc::read(doorbell.as_raw_fd(), count.as_mut_ptr().cast(), 8) };
                        // A full channel already holds a ring the loop has not taken.
                        let _ = ring.try_send(());
                    }
                }
            })?;
        Ok(Self {
            rings,
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        // Closing the relay's end of the socket pair ends the thread's poll.
        let _ = self.stop.shutdown(std::net::Shutdown::Both);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
