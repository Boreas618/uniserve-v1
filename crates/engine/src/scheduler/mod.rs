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
//! the executor interface (§2.1) and the scheduler's end of the queue pair (§3.1, §3.5):
//!
//! - `submit` takes a request for the buffer pool. A request the executors first apply to their
//!   device block tables (`Release`, `Detach`, `Commit`, `Transfer`, and the `Finish` that gives a
//!   copy up) waits in `stamping` until every group whose tables may hold its entries stamped it
//!   (BP §9, obligation 2); a request that names positions such a `Release` frees waits behind
//!   it; every other request is held in order. A further `Allocate` of an owner and class whose
//!   `Allocate` is still held raises that one's count (BP §10, F5).
//! - `post` writes the held requests in order while the submission queue has room and the
//!   completion queue has room for a completion of every request outstanding. A request's number
//!   is drawn as it is written, so the numbers rise in the order the buffer pool takes the
//!   requests, which it requires (BP §7, R1).
//! - `next_completion` reads the completion queue, matches each completion to its request by
//!   number, applies what it states of the cache (`cache.rs`), drops what belongs to a forgotten
//!   owner or to entries a later stamped request names
//!   (BP §9, obligation 1), and hands every completion that gives entries to the executors
//!   (`Executor::map`) before the scheduler applies it.
//! - `read_capacity` reads the capacity words of every region.
//!
//! The scheduling, admission, update, drain, policy, cache and statistics rules are `impl
//! Scheduler` blocks of the sibling files (SD §6).

mod cache;
mod group;
mod policy;
mod request;
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the step's assembly, `schedule`, which appends the rows and seals the batches, \
                  is not in the crate yet"
    )
)]
mod rows;
mod stats;
#[cfg(test)]
mod tests;
mod video;
mod view;

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crossbeam_channel::Sender;
use ipc::batch::{Batch, BatchId, BatchOutput};
use proto::config::{SchedulerConfig, WorkerGroupConfig};
use proto::report::SchedulerStats;
use proto::request::{Event, FinishReason, GenerationLimits, MigrationCause, RequestId};
use scheduling_guest::Parameters;
use storage::runtime::{self, Cause, Completion, GroupId, OwnerId, Tier};
use storage::{Key, QueuePair};

pub use cache::Cached;
pub use group::{Capacity, WorkerGroup};
pub use policy::{Policy, PolicyError, ceilings};
pub use request::{Blocks, Request, RequestStatus};
pub use view::view_of;

/// The first request id of the warm-up requests the engine adds when admission opens: an
/// ordinary request whose id is at or above it has its events dropped (SD §3.2).
pub const WARMUP_REQUEST_IDS: u64 = 1 << 63;

/// The bytes of a record's length word in a ring of a queue pair (`storage::queue`): a ring of
/// `max_record()` bytes of record holds that many more.
const LENGTH_WORD: usize = 8;

/// The engine's error: what ends a scheduler, fails a batch or refuses an operation.
#[derive(Debug)]
pub enum Error {
    /// The scheduling policy faulted.
    Policy(PolicyError),
    /// An executor failed a batch or its group.
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
    /// A submitted batch's output, from an executor, exactly once per batch.
    Output(Batch, Result<BatchOutput, Error>),
    /// From an executor: the requests it applied to its group's device block tables, each with
    /// its group's stamp, and the stamped `Finish` of each copy it ran.
    Submit(Vec<runtime::Request>),
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
    /// Evict every cached block the scheduler knows of; with seeds, the cache groups' seeds from
    /// here on.
    Flush(Option<Vec<Key>>, Sender<Result<(), Error>>),
    /// `Resize { region, units }` at the buffer pool, the region by its number; the reply is sent
    /// at its `Resized`.
    Resize(u8, u32, Sender<()>),
    /// The scheduler's limits, applied in place.
    Configure(SchedulerConfig),
    Shutdown,
}

/// What leaves the scheduler's thread, on its one channel (vLLM's `EngineCoreOutputs`).
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "a request's event and the statistics travel by value as the message itself"
)]
pub enum Output {
    /// A request's event, to the daemon.
    Event(RequestId, Event),
    /// The statistics of the last second.
    Stats(SchedulerStats),
    /// The serving policy faulted and the previous one serves again.
    PolicyRolledBack(String),
    /// A batch computed non-finite logits for these requests.
    NumericalFault(BatchId, Vec<RequestId>),
    /// A region, by its number, and its units after a shrink of a drain.
    Shrunk(u8, u32),
    /// The scheduler stopped.
    Failed(Error),
}

/// The executors of a scheduler's worker groups, as the scheduler drives them (BP §9 and §17,
/// A1 to A4).
///
/// - `submit`: the batch's output arrives exactly once as [`Message::Output`].
/// - `map`: a completion that gives entries (`Attached`, `Allocated`, `Pinned`, `Transferred`,
///   `Copy`, and the answers to a copy's `Finish`: `Loaded`, `Cloned`, `Duplicate`), with the
///   request it answers and that request's number, is written into `group`'s device block tables
///   at the positions the request names, under `slot` (the slot row of the request's owner on
///   that group; for a group that reads a class it does not write, the slot of the request's own
///   owner there), or its copies are queued, before the group's next batch launches. The entries
///   of a `Load` or a `Clone` appear with the answer to its `Finish`, read-only for a `Load` and
///   writable for a `Clone`. The executor writes no entry from any other source. A queued copy
///   runs with the group's next batch, and its `Finish { copy, ok, after }`, with the group's
///   stamp, returns in [`Message::Submit`].
/// - `stamp`: `Release`, `Detach`, `Commit` and `Transfer`, and the `Finish { ok: false }` that
///   gives a copy up, are applied to `group`'s device block tables (the entries a `Release`, a
///   `Detach` or a `Transfer` names are removed, the entry a `Commit` names is marked read-only; a
///   given-up copy joins no further batch and is dropped if it has not started), stamped with
///   the sequence of the last batch the group launched in their `after`, and every one of them
///   returns in [`Message::Submit`]. A given-up copy that has started is covered by that stamp
///   and gets no `Finish` of its own.
/// - A batch writes only entries that came from `Allocated` or `Cloned` and that no `Commit` has
///   marked, and a copy's destination is written only by that copy.
pub trait Executor: Send {
    /// Runs a batch on its group.
    fn submit(&self, batch: Batch);

    /// Writes the entries a completion gives into `group`'s device block tables, or queues its
    /// copies.
    fn map(
        &self,
        group: GroupId,
        slot: u32,
        id: runtime::RequestId,
        request: runtime::Request,
        completion: Completion,
    );

    /// Applies requests to `group`'s device block tables and returns them stamped.
    fn stamp(&self, group: GroupId, requests: Vec<runtime::Request>);
}

/// The scheduler of one data-parallel rank and its state (SD §2.1, §2.5).
///
/// Invariants: every id in `waiting` or `running` is in `requests` with the matching status; an
/// owner is in `owners` from its `Attach` until its `Detach` is handed to the executors, and no
/// owner is drawn twice; a block is in its region's `order` exactly when it is in `cache`, on
/// that region's tier, with no pin of this scheduler's requests; the completions of `outstanding`
/// always fit the completion queue; a request is in at most one of `stamping`, `held` and
/// `outstanding`; the numbers written into the submission queue rise; all state is touched on the
/// scheduler's thread only.
///
/// A region is named in completions and requests by its number at the buffer pool, and indexed
/// here, in every per-region field, by its place in the session's regions (`Open::regions`),
/// which is how the layout of a group and a class names it too.
#[expect(
    dead_code,
    reason = "the step loop, admission, update, drain and policy that read these fields are not \
              in the crate yet"
)]
pub struct Scheduler {
    config: SchedulerConfig,
    /// The generation limits requests are checked against at their arrival: the model's sizing
    /// facts lowered to what the groups' executors hold (SD §7.1), with the features the groups'
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
    /// Requests handed to executors for their stamps, by the number they were handed under: each
    /// with the stamps it has gathered (in its `after`), the count of groups still to stamp it,
    /// and the requests that must follow it into the submission queue.
    stamping: HashMap<u64, (runtime::Request, u8, Vec<runtime::Request>)>,
    /// Every request written into the submission queue whose completion has not been read, by its
    /// number, with the owner it acts for. A request answered `Copy` stays until the answer to
    /// its `Finish`.
    outstanding: HashMap<u64, (Option<OwnerId>, runtime::Request)>,
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
    /// The request each live owner is for.
    owners: HashMap<OwnerId, RequestId>,
    /// The counter owners are drawn from: the scheduler's data-parallel rank in the top 16 bits.
    next_owner: u64,

    policy: Policy,
    executor: Box<dyn Executor>,
    output: Sender<Output>,
    stats: SchedulerStats,
    /// When `stats` started counting: the scheduler's start, then its last `Output::Stats`.
    published: Instant,
    /// The kernel and policy generation batches are assembled for.
    generation: Arc<AtomicU64>,
    paused: bool,
    /// Whether the scheduler stopped on a fault (`fail`): its loop ends at the next pass.
    stopped: bool,
    /// The replies owed to `Pause` once nothing is in flight.
    boundary: Vec<Sender<()>>,
    /// The reply, cause, deadline and whether memory is given back, of a drain under way.
    draining: Option<(Sender<()>, MigrationCause, Instant, bool)>,
    /// The region, by number, of each `Resize` outstanding and its reply, owed at `Resized`.
    resizing: Vec<(u8, Sender<()>)>,
    next_batch: u64,
    step: u64,
}

impl Scheduler {
    /// Stops the scheduler on a fault it cannot recover from: `Output::Failed` tells the engine,
    /// which ends the epoch's requests in flight as migratable with the cause `Loss`, as for a
    /// lost process, and the loop ends at its next pass. Called once; later faults are dropped.
    pub(super) fn fail(&mut self, error: Error) {
        if self.stopped {
            return;
        }
        self.stopped = true;
        // An engine that no longer listens has stopped this scheduler's epoch already.
        let _ = self.output.send(Output::Failed(error));
    }
}

impl Scheduler {
    /// A scheduler of the session the engine opened with `layout` and that `opened` answered:
    /// the `scheduler`-th of `layout.schedulers`, which is its data-parallel rank, on its end of
    /// its queue pair. `groups` are its worker groups' configurations and the capacities their
    /// executors stated, in the order `layout.schedulers` lists its groups. `limits` are the
    /// generation limits the engine resolved for the deployment; their features are the ones the
    /// groups' phases serve.
    ///
    /// # Errors
    ///
    /// [`Error::Storage`] when the layout and its answer do not state this scheduler: no such
    /// scheduler, other groups than `groups`, or regions and registers that do not match.
    #[expect(
        clippy::too_many_arguments,
        reason = "the scheduler's parts come from four owners: the configuration, the session \
                  with the buffer pool, the executors and the engine's channels"
    )]
    pub fn new(
        config: SchedulerConfig,
        limits: GenerationLimits,
        layout: &runtime::Open,
        opened: &runtime::Opened,
        scheduler: usize,
        queue: QueuePair,
        groups: Vec<(WorkerGroupConfig, Capacity)>,
        policy: Policy,
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
        Ok(Self {
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
            owners: HashMap::new(),
            next_owner: (rank << 48) + 1,
            policy,
            executor,
            output,
            stats: stats::empty(),
            published: Instant::now(),
            generation,
            paused: false,
            stopped: false,
            boundary: Vec::new(),
            draining: None,
            resizing: Vec::new(),
            next_batch: 0,
            step: 0,
        })
    }
}

/// The scheduler's end of its queue pair (SD §3.1, §3.5 `answered`, §4.1).
#[expect(
    dead_code,
    reason = "the step loop, admission and update that call these are not in the crate yet"
)]
impl Scheduler {
    /// Takes a request for the buffer pool; a later [`Scheduler::post`] writes it into the
    /// submission queue. Nothing here touches the queue pair or waits.
    ///
    /// - An `Allocate` that continues a held `Allocate` of the same owner, class and block size
    ///   at the position after it raises that one's count instead.
    /// - A `Release`, a `Commit`, a `Detach` or a `Transfer` is handed to `Executor::stamp` of
    ///   every group whose device block tables may hold the entries it names: the group of the
    ///   owner's attachment, and every other group the request is attached to that reads the
    ///   class (one that states a region of the class that the owner's group states too). The
    ///   groups are found from the request's `attachments`, so an owner's `Detach` is submitted
    ///   before its attachment is cleared, and an owner whose entries no table holds (a failed
    ///   `Transfer`) has its attachment cleared first. A `Finish { ok: false }` that gives a copy
    ///   up is handed to the copy's group. A request with no group to stamp it is held at once
    ///   with no stamp. A `Detach` takes its owner out of `owners`.
    /// - An `Allocate`, a `Pin`, a `Load` or a `Clone` at positions a `Release` being stamped
    ///   frees waits behind that `Release`, so that the buffer pool takes it after the positions
    ///   are free.
    /// - Every other request is held, in order.
    ///
    /// The caller submits no `Allocate` of an owner and class whose `Allocate` is outstanding:
    /// that need is asked again after its completion.
    fn submit(&mut self, request: runtime::Request) {
        if self.merge(&request) {
            return;
        }

        let groups = match &request {
            runtime::Request::Release { owner, class, .. }
            | runtime::Request::Commit { owner, class, .. } => self.holders(*owner, Some(*class)),
            runtime::Request::Detach { owner, .. } | runtime::Request::Transfer { owner, .. } => {
                self.holders(*owner, None)
            }
            runtime::Request::Finish {
                copy, ok: false, ..
            } => self
                .outstanding
                .get(&copy.0)
                .and_then(|(_, copied)| self.copy_group(copied))
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
                return;
            }
        };

        if let runtime::Request::Detach { owner, .. } = &request {
            self.owners.remove(owner);
        }
        self.hand(request, groups);
    }

    /// Hands `request` to the executors of `groups` for their stamps and keeps it in `stamping`
    /// until each has returned it; with no group it is held at once, with no stamp.
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
    /// owner, class and block size, at the position after its last. True when it did.
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

    /// Takes the requests an executor returns ([`Message::Submit`]). A request handed for stamps
    /// gathers the stamp each returned copy carries, and once every group has returned it, it is
    /// held, followed by the requests that waited behind it. A request no executor was handed
    /// for stamps (the `Finish` of a copy an executor ran) is held as it is.
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

            let Some(key) = key else {
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
    /// every request outstanding and of the one being written. The first that does not fit and
    /// every one after it stay held for the next pass; nothing waits.
    ///
    /// # Errors
    ///
    /// [`Error::Storage`] when a request's record exceeds a ring or the queue pair is corrupt.
    fn post(&mut self) -> Result<(), Error> {
        let ring = self.queue.max_record() + LENGTH_WORD;
        let mut owed: usize = self
            .outstanding
            .values()
            .map(|(_, request)| self.answer_bytes(request))
            .fold(0, usize::saturating_add);

        let first = self.next_submission;
        let mut owners = Vec::new();
        let mut record = Vec::new();
        while let Some(request) = self.held.get(owners.len()) {
            let owner = self.acting_owner(request);
            let bytes = self.answer_bytes(request);
            if owed.saturating_add(bytes) > ring {
                break;
            }

            record.clear();
            let number = first + owners.len() as u64;
            request.encode(runtime::RequestId(number), &mut record);
            if !self.queue.submit(&record).map_err(storage_error)? {
                break;
            }
            owed += bytes;
            owners.push(owner);
        }

        self.next_submission = first + owners.len() as u64;
        let written: Vec<runtime::Request> = self.held.drain(..owners.len()).collect();
        for (number, (owner, request)) in (first..).zip(owners.into_iter().zip(written)) {
            self.outstanding.insert(number, (owner, request));
        }
        Ok(())
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
    /// left `owners` (what it gave goes with that owner's `Detach`), and a completion that gives
    /// entries when a `Release`, a `Commit` or a `Transfer` naming them was handed to the
    /// executors after its request was submitted (BP §9, obligation 1). A dropped `Copy` is given
    /// up at once with `Finish { ok: false }` through its group's executor. A `Refused` with the
    /// cause `copy` answers a `Finish` for a copy another `Finish` ended, and is dropped.
    ///
    /// Before it is returned, a completion that gives entries is handed to `Executor::map`: to the
    /// group an `Attach` or a `Transfer` names, to a copy's group, and otherwise to the owner's
    /// group and every other group that reads the class, each with the slot of the request's own
    /// owner there.
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
            let (owner, mut request) = self.outstanding.remove(&id.0).ok_or_else(|| {
                Error::Storage(format!(
                    "a completion answers request {}, which is not outstanding",
                    id.0
                ))
            })?;
            // What it states of the cache holds whether or not the completion is dropped.
            self.learn(&mut request, &completion);

            if let runtime::Request::Finish { copy, .. } = request {
                if let Some((owner, copied)) = self.matched_finish(id, copy, request, &completion) {
                    return Ok(Some((copy, owner, copied, completion)));
                }
            } else if self.matched(id, owner, &request, &completion) {
                return Ok(Some((id, owner, request, completion)));
            }
        }
        Ok(None)
    }

    /// Matches the completion of a request other than a `Finish`, as
    /// [`Scheduler::next_completion`] states: false when it is dropped, true when it was handed
    /// to the executors that take it and is to be applied.
    fn matched(
        &mut self,
        id: runtime::RequestId,
        owner: Option<OwnerId>,
        request: &runtime::Request,
        completion: &Completion,
    ) -> bool {
        // A request answered `Copy` waits for the answer to its `Finish`.
        if matches!(completion, Completion::Copy { .. }) {
            self.outstanding.insert(id.0, (owner, request.clone()));
        }

        let gives = matches!(
            completion,
            Completion::Allocated { .. }
                | Completion::Pinned { .. }
                | Completion::Transferred { .. }
                | Completion::Copy { .. }
        );
        if self.forgotten(owner) || (gives && self.superseded(id.0, owner, request)) {
            if let Completion::Copy { group, .. } = completion {
                let give_up = runtime::Request::Finish {
                    copy: id,
                    ok: false,
                    after: Vec::new(),
                };
                self.hand(give_up, vec![*group]);
            }
            return false;
        }

        for (group, slot) in self.mapped(owner, request, completion) {
            self.executor
                .map(group, slot, id, request.clone(), completion.clone());
        }
        true
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
        let (owner, copied) = self.outstanding.remove(&copy.0)?;
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
                self.executor
                    .map(group, slot, id, finish, completion.clone());
            }
        }
        Some((owner, copied))
    }

    /// Reads the `available` and `retiring` words of every region this queue pair uses, raises
    /// `available` by the units of `order` beyond the installed head, which the buffer pool may
    /// evict once a later `Victims` names them, and lowers it by the units of the scheduler's
    /// `Allocate`s held, being stamped or outstanding, which the buffer pool's word may not show
    /// yet.
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

        let waiting = self
            .stamping
            .values()
            .flat_map(|(_, _, following)| following.iter());
        let asked: Vec<(usize, u32)> = self
            .held
            .iter()
            .chain(waiting)
            .chain(self.outstanding.values().map(|(_, request)| request))
            .flat_map(|request| self.taken(request))
            .collect();
        for (region, units) in asked {
            if let Some(available) = self.available.get_mut(region) {
                *available = available.saturating_sub(units);
            }
        }
    }

    /// The `evictable` word of `region` (by index): the units of every shared block without a
    /// pin there, in any eviction order or none.
    fn evictable(&self, region: usize) -> u32 {
        let Some(&(number, _)) = self.regions.get(region) else {
            return 0;
        };
        self.registers
            .capacity
            .iter()
            .find(|(stated, _)| *stated == number)
            .map_or(0, |&(_, word)| {
                self.word(
                    usize::try_from(word)
                        .unwrap_or(usize::MAX)
                        .saturating_add(2),
                )
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
                self.outstanding.get(&copy.0).and_then(|(owner, _)| *owner)
            }
            request => owner_of(request),
        }
    }

    /// The slot row of the request `owner` is for on `group`; 0, a row no owner holds, when it
    /// has none there.
    fn slot_of(&self, owner: Option<OwnerId>, group: GroupId) -> u32 {
        owner
            .and_then(|owner| self.request_of(owner))
            .and_then(|request| request.slot_on(group))
            .unwrap_or(0)
    }

    /// The groups whose device block tables may hold entries of `owner`, of `class` or of any
    /// class: the group of the owner's attachment, and every other group its request is attached
    /// to that states a region of the class that the owner's group states too.
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
                reader
                    .regions(class)
                    .iter()
                    .any(|region| writer.regions(class).contains(region))
            });
            if reads {
                groups.push(group);
            }
        }
        groups
    }

    /// The group whose executor runs the copies of `request`: the group a `Store` names, and the
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

    /// The groups, each with its slot, that a completion of `request` is handed to by `map`:
    /// none for a completion that gives no entries.
    fn mapped(
        &self,
        owner: Option<OwnerId>,
        request: &runtime::Request,
        completion: &Completion,
    ) -> Vec<(GroupId, u32)> {
        match (request, completion) {
            (runtime::Request::Attach { group, .. }, Completion::Attached { slot })
            | (runtime::Request::Transfer { group, .. }, Completion::Transferred { slot, .. }) => {
                vec![(*group, *slot)]
            }
            (_, Completion::Copy { group, .. }) => vec![(*group, self.slot_of(owner, *group))],
            (
                runtime::Request::Allocate { owner, class, .. }
                | runtime::Request::Pin { owner, class, .. },
                Completion::Allocated { .. } | Completion::Pinned { .. },
            ) => self
                .holders(*owner, Some(*class))
                .into_iter()
                .map(|group| (group, self.slot_of(Some(*owner), group)))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Whether a `Release`, a `Commit` or a `Transfer` naming entries `request` gives was handed
    /// to the executors after `request`, numbered `number`, was submitted: it is being stamped,
    /// held, or outstanding under a later number. A request submitted after such a `Release`
    /// waited behind it, so it was written after it.
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
            ) && overlap(request, other)
        };

        self.stamping.values().any(|(waiting, _, _)| names(waiting))
            || self.held.iter().any(names)
            || self
                .outstanding
                .iter()
                .any(|(later, (_, other))| *later > number && names(other))
    }

    /// Per region (by index), the units an `Allocate` takes: its blocks in each region of its
    /// class in the owner's group.
    fn taken(&self, request: &runtime::Request) -> Vec<(usize, u32)> {
        let runtime::Request::Allocate {
            owner,
            class,
            count,
            units,
            ..
        } = *request
        else {
            return Vec::new();
        };
        let Some(group) = self
            .request_of(owner)
            .and_then(|row| row.attachment(owner))
            .and_then(|(group, _)| self.group(group))
        else {
            return Vec::new();
        };

        let each = self.block_units(class, units).unwrap_or(0);
        group
            .regions(class)
            .iter()
            .map(|&region| (usize::from(region), count.saturating_mul(each)))
            .collect()
    }

    /// The units of one block of `class`: the class's block size, or `units` for a sized class.
    fn block_units(&self, class: u8, units: u32) -> Option<u32> {
        let stated = self.classes.get(usize::from(class))?.units;
        Some(if stated > 0 { stated } else { units })
    }

    /// The runs one block of `class` has in the group of `owner`, and its units.
    fn block_shape(&self, owner: OwnerId, class: u8, units: u32) -> Option<(usize, usize)> {
        let group = self
            .request_of(owner)
            .and_then(|row| row.attachment(owner))
            .and_then(|(group, _)| self.group(group))?;
        let runs = group.regions(class).len().max(1);
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
            runtime::Request::Attach { .. } | runtime::Request::Prefetch { .. } => Some(4),
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
        | runtime::Request::Restore { owner, .. } => Some(*owner),
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
        | runtime::Request::Finish { after, .. } => Some(after),
        _ => None,
    }
}

fn storage_error(error: storage::Error) -> Error {
    Error::Storage(error.to_string())
}
