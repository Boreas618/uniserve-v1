//! Scheduling, KV cache management and request control of an engine.
//!
//! An engine runs one scheduler per data-parallel rank: it builds a
//! `SchedulerHandle` (`core`) over the devices it holds, which runs the
//! [`Scheduler`] on a thread of its own. The engine's control thread submits
//! and controls requests through the handle and receives each request's
//! events on its `EventRx` (`handle`).
//!
//! The scheduler's surface is five calls, vLLM's: [`Scheduler::add_request`],
//! [`Scheduler::cancel`], [`Scheduler::schedule`], [`Scheduler::update`] and
//! [`Scheduler::drain`]. [`Scheduler::run`] is the step loop over them: take
//! what arrived, `update` with each completed batch, `schedule` the batches
//! that fit, submit them to the executors (`submission`), and block until
//! something arrives. All scheduler state stays on that thread.
//!
//! A request is one type from arrival to end (`request`): waiting, running
//! or finished, with a graph of calls that is a token request's loop or a
//! video request's chain. Admission gives a request a slot and the pages its
//! next call needs after its cached prefix was looked up (`kv_cache`); a
//! request that cannot be admitted waits, a request larger than the cache is
//! rejected, and when running requests themselves cannot grow the policy
//! names a victim, which ends as migratable.
//!
//! The scheduler's modules:
//!
//! - `run`: construction, the step loop and its wait.
//! - `request`: a request, its status and the table of every request.
//! - `admission`: arrival, the waiting queue and admission.
//! - `batching`: batch assembly and the calls a batch carries.
//! - `generation`: a token request's state, call builders and result
//!   validation.
//! - `denoising`: solver progress of one latent trajectory.
//! - `graph`: the call graph of one video request.
//! - `execution`: the step (`schedule`, `update`), video scheduling and
//!   failure handling.
//! - `inflight`: batches and calls in flight, and request-local completion
//!   order.
//! - `submission`: the door to the executors and the scheduler's wake.
//! - `output`: public events, semantic result resolution, and termination.
//! - `allocation` and `dispatch`: what requests hold of the storage pools,
//!   and the worker group each call runs on.
//! - `control`: the control thread's commands.
//! - `policy`, `mechanism` and `views`: the policy module, the checks of its
//!   decisions and what it reads.
//! - `stats` and `stats_report`: statistics.

/// Scheduler-owned allocation of worker storage; only its `OutOfStorage`
/// error is public.
pub mod allocation;
mod clock;
mod core;
mod encoder_cache;
mod encoder_outputs;
mod handle;
mod host_pages;
mod kv_cache;
mod trajectories;
mod transfer;
mod warmup;
mod worker;

pub use crate::core::{SchedulerHandle, WorkerGroupConfig, WorkerGroupRole, WorkerRank};
pub use crate::handle::{
    CacheChange, Command, CommandSender, CommandWaker, Counters, EVENT_BUFFER_CAPACITY, EventRx,
    EventSendError, EventTx, HookBarrier, PolicyPackages, StageApplied, StreamCancelCause,
    Submission, SubmitError, Wake, WeightsInFlight,
};
pub use crate::worker::{
    BlockConfig, BlockDistribution, CommandResult, CommandStatus, ExecuteError, ExecutionBatch,
    PageOffloads, RankBuffer, RankEncoderPools, RequestDispatch, SequenceParallel, WorkerError,
    WorkerFailure, WorkerGroupId, WorkerInfo,
};
// The engine schedules requests in the request vocabulary its daemon
// submits them in.
pub use crate::kv_cache::{HostCacheSize, IndexedPage, prefix_seeds};
pub use crate::submission::Submissions;
pub use ipc::{MediaCall, VideoDenoiserInfo};
pub use proto::config::DEFAULT_BLOCK;
/// Serializable scheduler, cache, and worker performance snapshots.
pub use proto::report;
pub use proto::request::{
    ArtifactEvent, ArtifactMedia, CachePolicy, ConditionExtent, DiffusionRequest,
    DiffusionRequestError, EncoderInputs, Event, FinishReason, GenerationFeatures,
    GenerationLimits, GenerationRequest, GenerationRequestError, GenerationResourceError,
    ImageEncoder, ImageEncoderInput, ImageInput, MediaKind, ModelDtype, ModelDtypeParseError,
    MultimodalInputs, PositionLogprobs, RejectionKind, Request, StopReason, Trajectory,
};

/// Lock `mutex`, keeping the state a panicking holder left.
///
/// Every build profile sets `panic = "abort"`, so a holder's panic ends the
/// process before another thread observes the poisoned lock. Only test
/// builds, which ignore that setting and unwind, reach the recovery; it
/// returns the guard rather than raising a second panic in the thread that
/// locks next.
pub(crate) fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

mod completion;
mod stats;
mod terminal;
use terminal::{Terminal, Usage};
pub(crate) mod stats_report;

pub use epoch::{GroupCache, ParkedCache};
use ipc::ForwardMode;
pub use policy::{
    Arrival, BatchCapacity, Candidate, Limits, ModuleSource, PREFETCH_FLOOR_TOKENS, Parameters,
    PolicyBytes, PolicyPackage, PolicyRolledBack, PolicySlot, PolicySource, PolicyUsage, Progress,
    RouteRefusal, RouteTrial, Router, Stage, TrialState, Validators, arrival_view, ceilings_of,
};
pub use proto::config::{
    DEFAULT_CHUNKED_PREFILL_SIZE, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_RUNNING_REQUESTS, SchedulerConfig,
};
pub use scheduling::guest::view::RankView;
pub use stats::{
    DomainStats, EncoderStats, ExecutionDomainStats, GeneralStats, PrefixCacheUsage, PrefixStats,
    SchedulerStats, TimingStats, WorkerStats,
};
pub use stats_report::SchedulerStatsReporter;
pub use views::phase_id;

mod admission;
mod batching;
mod cache_storage;
mod control;
mod denoising;
mod dispatch;
pub(crate) mod encodings;
mod execution;
use execution::VideoState;
pub(crate) mod generation;
pub(crate) mod graph;
mod inflight;
mod mechanism;
pub(crate) mod output;
mod policy;
mod preemption;
mod quota;
mod request;
mod run;
mod speculation;
pub(crate) mod submission;
/// The mechanism's door to the worker crate's batch path, the in-process
/// submission path of §2.4 and §7.3's splice.
pub(crate) fn submissions_over(
    batches: Arc<dyn ipc::execution::Batches>,
) -> anyhow::Result<submission::Submissions> {
    submission::Submissions::over_batches(batches)
}
mod epoch;
pub(crate) mod views;
mod warming;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::generation::{
    BlockCursor, GenerationPhase as Phase, consumes_image_features, is_prompt_extend,
};

use crate::allocation::{BufferPool, BufferSpan, RequestPool, RequestSlot};
use crate::kv_cache::{BlockPool, BlockTable, CacheAllocation, KvCacheManager};
use crossbeam_channel::Receiver;
use ipc::RequestId;
use ipc::execution::EpochOperation;
use ipc::{
    BatchCommand, BlockTable as IpcBlockTable, BufferAllocation, CacheUnitAllocation, Call, CallId,
    CallKind, CallStatus, DecodeRange, DimBound, ForwardStats, GenerationParams, MainProcessInfo,
    NewRequest, RequestKey, SamplingState, ShapeBound, TensorRef, TimingCounters,
};
use ipc::{SharedMedia, TokenLogprob};
use proto::request::MigrationCause;
use proto::tensor::DType;

use allocation::requests::{
    MediaAllocations, MediaStorage, MediaTensorAllocation, RequestAllocations, RetiringRequest,
    UnknownMediaGroup,
};
use denoising::Denoising;
use generation::TokenState;
use graph::{VideoGraph, product};
use inflight::{InflightCall, InflightInput, PendingCompletion, PendingFinish};
use output::RequestOutput;
use request::{Graph, RequestStatus};

/// The kind of pass a call kind is batched in (see `batch_kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchKind {
    /// Every forward prefill, and text, vision, and latent encoding.
    Prefill,
    /// Calls that advance admitted requests without growing their prompt:
    /// token decode, speculative verification, and block denoising.
    Decode,
    /// Latent preparation, denoising, video and audio decoding, video and
    /// audio encoding, muxing, and transfers.
    Media,
}

/// Longest a drain (`Scheduler::drain_batches`) may go without a batch or a
/// submission leaving the pending set before it calls what it still holds
/// abandoned. It bounds silence, not the drain: every completion restarts it,
/// so a scheduler that keeps receiving results is never cut off under load.
const DRAIN_SILENCE: Duration = Duration::from_secs(300);
/// Longest a closing scheduler waits for its last pages to reach the
/// host KV cache: for the batches in flight, which carry offloads, and for
/// the batches of the remaining offloads.
const PAGE_FLUSH_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
/// Model token identifiers used for sequence boundaries.
///
/// Serving resolves these from the loaded model's configuration in the server
/// crate (`special_token_ids`).
pub struct SpecialTokenIds {
    /// Beginning-of-sequence token identifier.
    pub bos: u32,
    /// Token identifiers that terminate generation. The first entry is the
    /// primary EOS, which the output path substitutes for a completion that
    /// commits no token, so the list must not be empty.
    pub eos: Vec<u32>,
}

/// Request-held reference to a reusable encoder-cache entry.
struct EncoderCachePin {
    key: crate::encoder_cache::EncoderKey,
    product: TensorRef,
}

/// What a request's latent preparation opens its trajectory from.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Restoring {
    /// Its stored trajectory's pages are being pinned in the host latent
    /// pools; the preparation waits for the responses.
    Pinning,
    /// The pages are pinned (`Some`), or the trajectory starts from noise:
    /// the request names no stored steps, the host latent pools do not
    /// serve its denoiser, or a page is missing. The preparation takes it.
    Ready(Option<ipc::LatentRestore>),
    /// The preparation was prepared; the pins, if any, end with its result.
    Prepared,
}

impl Default for Restoring {
    fn default() -> Self {
        Self::Ready(None)
    }
}

/// A request's latent trajectory in the host latent pools
/// (`crate::trajectories`).
#[derive(Debug, Clone, Default)]
struct HostTrajectory {
    /// Whether the trajectory continues from a stored step, which its
    /// latent preparation waits for.
    restoring: Restoring,
    /// Whether the host latent pools are storing the trajectory, which a
    /// preemption ended at a step boundary (`Terminal::Migrated`); the
    /// request leaves once they stored it or failed to.
    offloading: bool,
    /// How each denoising rank's pages hold the trajectory's samples, by
    /// rank, as its latent preparation reported them
    /// (`CallCompletion::latent_layouts`): a stored trajectory's pages are
    /// keyed and recorded by them.
    layouts: BTreeMap<u32, String>,
}

/// Progress of a media request's worker admission (`NewRequest`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerRegistration {
    /// No scheduled call has carried the admission yet; no worker holds
    /// state for the request, so retirement needs no `Finish` command.
    Unsubmitted,
    /// The call carrying the admission is in flight; the request offers no
    /// further calls until it resolves.
    InFlight,
    /// A valid result of the admitting call has confirmed registration.
    Registered,
}

/// Where a scheduler is in its life. One state at a time; the storage
/// conditions (`held`, `asleep`, `resizing`), the drain guard and the epoch's
/// `trial` are separate fields because they combine with any of these.
///
/// Transitions: `Serving` → `Parking` at a park (`Scheduler::park`);
/// `Parking` → `Parked` once its requests ended (`Scheduler::poll_park`);
/// `Parked` → `Serving` at a resume (`Scheduler::resume`); any state →
/// `Failed` when the worker dies or a scheduler invariant breaks, after which
/// the control loop exits and the host converts it into engine-dead.
enum Lifecycle {
    /// Admits and runs requests.
    Serving,
    /// A park waits for the scheduler's requests to end; it responds once
    /// they did.
    Parking(epoch::Parking),
    /// Parked: it admits and executes nothing until it resumes, and keeps
    /// its storage and prefix cache index.
    Parked,
    /// Engine-fatal: the control loop stops.
    Failed,
}

/// Single-threaded owner of scheduling, storage, execution, and request state.
pub struct Scheduler {
    worker: submission::Submissions,
    inflight: inflight::Inflight,
    /// Ownership and reservation accounting of what running requests hold.
    storage: allocation::requests::Storage,
    dispatch: dispatch::Dispatch,
    /// Aggregate capacity view of the loaded workers.
    info: MainProcessInfo,
    generation_limits: crate::GenerationLimits,
    special_tokens: SpecialTokenIds,
    /// Every request the scheduler knows, waiting or running, by id.
    requests: request::Requests,
    /// The requests awaiting admission, token and video alike: the requests
    /// that resume work another engine began first, then arrival order
    /// (`Scheduler::add_request`). Which of them a step admits is the
    /// policy's decision; the mechanism tries the named ones in this order.
    waiting: VecDeque<RequestId>,
    /// What the epoch measures of speculation, which the refill policy
    /// weighs.
    speculation_meter: speculation::SpeculationMeter,
    /// Finished requests whose storage stays allocated until their workers
    /// acknowledge the `Finish` command.
    retiring_requests: HashMap<RequestId, RetiringRequest>,
    /// The step's policy: the instance that decides, installed when the
    /// scheduler starts and replaced only by the switch at a batch boundary.
    ///
    /// `None` between construction and [`Scheduler::adopt`]: a slot is
    /// installed on a state that the scheduler owns, so it is adopted once the
    /// scheduler exists and before its first step.
    policy: Option<policy::PolicySlot>,
    /// The report of a policy failure the mechanism did not take from `fail`'s
    /// response, for the control thread.
    policy_report: Option<policy::PolicyRolledBack>,
    /// §7.3 step 2's paired pass: an update asked that the following step's
    /// batches also run in shadow for an epoch generation, and the response waits
    /// for the step that shadows them.
    shadow_pass: Option<control::ShadowPass>,
    /// The candidate a precopy is preparing on a blocking thread, while it runs
    /// (§7.3 step 2): loading and instantiating a policy must not hold the
    /// serving thread.
    policy_prepare:
        Option<crossbeam_channel::Receiver<Result<policy::Candidate, scheduling::PolicyError>>>,
    /// The candidate a precopy prepared at this boundary, until the switchover
    /// puts it in.
    policy_prepared: Option<policy::Candidate>,
    /// The generation this scheduler's batches are stamped with (§7.3): the one
    /// the epoch last switched to, written by the control thread's switchover
    /// arm while it holds the generations' lock, read once per step.
    generation: Arc<std::sync::atomic::AtomicU64>,
    /// The generation the current step's batches are laid out for: read from
    /// `generation` once per step, so a switch mid-step cannot split one
    /// step across two generations.
    step_generation: u64,
    /// Tells the thread that reads this scheduler's output that there is some:
    /// its events and its responses wait for that thread, so a step that produced
    /// them says so rather than leaving them to its next periodic pass.
    progress: CommandWaker,
    /// The stage §7.3 asked for and the response to give once it is applied: a
    /// stage changes the serving policy, so it waits for a boundary where no
    /// batch and no submission is in flight.
    policy_stage: Option<(
        policy::Stage,
        Option<crate::handle::StageResponse>,
        Option<u64>,
    )>,
    /// The events and transitions a policy is told about (§4.4).
    recorder: mechanism::Recorder,
    /// The device pages that exist, kept from the epoch's snapshot and from the
    /// `Cached`/`Freed` announcements of each step: the table `Live` checks a
    /// named page against, without a step scanning the prefix cache.
    pages: std::collections::BTreeMap<scheduling::guest::PageId, scheduling::guest::PageView>,
    /// The running requests that hold a row of the request pool: what
    /// `Room::slots` subtracts from the pool's rows.
    ///
    /// Maintained where a request's allocations are installed (admission) and
    /// taken (the end of its execution), because a step must not scan the
    /// running requests to count them (design §4.4).
    rows_held: usize,
    /// Whether `pages` was built yet. It is built on the first step that
    /// announces a change, from `views::page_index`, which is the one scan.
    pages_seeded: bool,
    /// The starvation watchdog: a running request that does not advance
    /// within the policy's `timeout` is the policy's failure (§4.4).
    watchdog: mechanism::Watchdog,
    /// The moment of the last step's starvation check: a request past its
    /// bound by then was judged by that step (`Scheduler::wait`).
    starvation_judged: scheduling::guest::Nanos,
    /// The step number this scheduler is at.
    step: u64,
    /// `engine_id` of every `RequestKey` this scheduler issues.
    engine_id: u64,
    epoch: u64,
    /// Engine-wide counter for product generations, drawn by
    /// `generation::register_call` and `Scheduler::media_completion_product`;
    /// both fail once a generation would exceed `u32::MAX`.
    next_product_generation: u64,
    /// Epoch stamped on the next request state the scheduler creates, so
    /// results from an earlier lifetime of a request id never match it.
    next_request_epoch: u64,
    config: SchedulerConfig,
    /// The admitted requests, token and video alike, in admission order.
    /// Media scheduling visits them in this order; batch assembly uses it
    /// to break ties within `assembly_priority`.
    running: Vec<RequestId>,
    output: output::OutputSender,
    /// Where the scheduler is in its life: serving, parking, parked, or
    /// failed (`Lifecycle`).
    lifecycle: Lifecycle,
    /// Shared scheduler counters and latency accumulators.
    stats: Arc<SchedulerStats>,
    /// The host KV cache, when it serves a worker group
    /// (`Scheduler::enable_kv_offload`).
    kv_offload: Option<crate::kv_cache::KvOffloadManager>,
    /// The decoded tokens between two checkpoints of a request's state
    /// groups (`Scheduler::set_state_interval`).
    state_interval: usize,
    /// The host latent pools of the video denoiser, when they serve its
    /// worker group (`Scheduler::enable_trajectory_store`).
    trajectories: Option<crate::trajectories::TrajectoryStore>,
    /// The host encoder pools of the video encoders, when they serve their
    /// worker groups (`Scheduler::enable_encoder_store`).
    encoder_outputs: Option<crate::encoder_outputs::EncoderStore>,
    /// The transfers from the prefill groups to the decode groups, when the
    /// deployment has prefill groups (`Scheduler::enable_transfer`).
    transfers: Option<crate::transfer::Transfers>,
    /// Set while an operation waits for the batches in flight to complete
    /// (`Scheduler::drain_batches`, `Scheduler::flush_offloads` and the
    /// commands that drain before they act): the scheduler admits and
    /// schedules nothing new meanwhile. The operation restores what it found,
    /// so a nested drain keeps the outer one's.
    draining: bool,
    /// Until when a promoted successor holds the requests that continue
    /// from its predecessor's pages (`cache.resume`), whose index arrives
    /// once the predecessor drained (`Scheduler::inherit`); `None` once it
    /// arrived, or for a scheduler that awaits none.
    awaiting_index: Option<std::time::Instant>,
    /// The requests held for the index meanwhile, in arrival order: the
    /// others are admitted past them, and they return to the head of the
    /// queue when the index arrives or the wait ends.
    held_for_index: Vec<RequestId>,
    /// Whether forming batches is held for a change of the device's storage
    /// (`CacheChange::Hold`): nothing runs, and every request keeps its state.
    ///
    /// Orthogonal to `lifecycle`, `asleep` and `resizing`: the storage
    /// commands that set them arrive independently of a park, and the daemon
    /// may hold, sleep or resize a scheduler in any lifecycle state.
    held: bool,
    /// A shrink or sleep of the KV cache waiting for its units to be let go
    /// (`Scheduler::resize_cache`).
    resizing: Option<cache_storage::PendingResize>,
    /// The running requests the latest assembly could not grow, each with the
    /// cache units its growth needed. The next step's preemption reads them before
    /// its assembly notes them again (`Scheduler::preempt_shortage`,
    /// `Scheduler::recompute_starved`), and admission waits while any is here.
    starved: Vec<(RequestId, usize)>,
    /// The workers gave back the cache and graph storage: the scheduler
    /// admits and runs nothing until it wakes.
    asleep: bool,
    warming: warming::Warming,
    /// The scheduler's epoch is on trial until its commit
    /// (`Scheduler::commit`): any fault, non-finite logits included, fails
    /// the epoch as a whole, which the daemon rolls back rather than
    /// repairs, and the requests that resume from their generated tokens
    /// leave as migratable for the previous epoch.
    trial: bool,
    /// The first error of a fatal stop, which the scheduler's handle reports
    /// (`SchedulerHandle::failure`); set once.
    failure: Arc<std::sync::OnceLock<String>>,
}

impl Scheduler {
    /// Latches engine-fatal: the control loop stops at its next pass.
    fn latch_fatal(&mut self) {
        self.lifecycle = Lifecycle::Failed;
    }

    /// Whether an engine-fatal fault stopped the scheduler.
    fn failed(&self) -> bool {
        matches!(self.lifecycle, Lifecycle::Failed)
    }
}

impl Scheduler {
    /// Latches engine-fatal for a scheduler invariant that no longer holds.
    ///
    /// Scheduling cannot continue from inconsistent state, so the control loop
    /// stops at its next check and fails every request, as it does when a
    /// worker dies. The caller abandons the operation that found the violation.
    fn invariant_broken(&mut self, invariant: &str) {
        tracing::error!(invariant, "scheduler invariant broken; stopping the engine");
        self.fail(format!("a scheduler invariant broke: {invariant}"));
    }

    /// Latches engine-fatal for `reason`, which is kept as the stop's first
    /// error unless an earlier one is.
    fn fail(&mut self, reason: String) {
        self.latch_fatal();
        let _ = self.failure.set(reason);
    }
}

/// Adds each per-key delta of a worker forward statistic into the shared
/// cumulative map in `SchedulerStats`. A poisoned lock is recovered rather
/// than propagated.
fn add_worker_forward_map(target: &Mutex<BTreeMap<String, u64>>, delta: &BTreeMap<String, u64>) {
    if delta.is_empty() {
        return;
    }
    let mut target = target
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (key, value) in delta {
        *target.entry(key.clone()).or_default() += *value;
    }
}

/// Returns tokens that stop a device-relay successor before host semantic resolution.
///
/// The sorted, deduplicated list holds the request's stop token ids and the
/// EOS ids unless `ignore_eos` is set. It reaches the worker in
/// `GenerationParams::finish_token_ids` and bounds verified-draft prefixes in
/// `validate_generation_result`. Terminal stop/EOS tokens end the request.
fn finish_token_ids(request: &GenerationRequest, eos: &[u32]) -> Vec<u32> {
    let mut finish_token_ids = request.stop_token_ids.clone();
    if !request.sampling.ignore_eos {
        finish_token_ids.extend(eos.iter().copied());
    }
    finish_token_ids.sort_unstable();
    finish_token_ids.dedup();
    finish_token_ids
}

/// Classifies a call kind into its batch kind.
///
/// Batch kind selection (`Scheduler::select_batch_kind`), batch assembly,
/// and the prefill flag set by `Inflight::register_pending_batch` read this
/// mapping.
fn batch_kind(call_variant: CallKind) -> BatchKind {
    match call_variant {
        CallKind::Forward(ForwardMode::Prefill)
        | CallKind::Media(MediaCall::TextEncoding)
        | CallKind::Media(MediaCall::VisionEncoding)
        | CallKind::Media(MediaCall::LatentEncoding) => BatchKind::Prefill,
        CallKind::Forward(
            ForwardMode::Decode | ForwardMode::Verify | ForwardMode::TokenDenoising,
        ) => BatchKind::Decode,
        CallKind::Media(MediaCall::Denoising)
        | CallKind::Media(MediaCall::VideoDecoding)
        | CallKind::Media(MediaCall::LatentPreparation)
        | CallKind::Media(
            MediaCall::MediaReading
            | MediaCall::VideoEncoding
            | MediaCall::AudioEncoding
            | MediaCall::AudioDecoding
            | MediaCall::Muxing,
        ) => BatchKind::Media,
    }
}

/// Returns the order in which ready completions are applied; lower values go
/// first, and equal values keep arrival order.
///
/// Denoising completions precede all other kinds (`Inflight::take_ready_completions` and the resolution pass in
/// `execution`).
fn completion_priority(call_variant: CallKind) -> u8 {
    match call_variant {
        CallKind::Media(MediaCall::Denoising) => 0,
        _ => 1,
    }
}

/// Events a request's output journal may hold beyond what its bounded event
/// channel holds.
const OUTPUT_JOURNAL_CAPACITY: usize = EVENT_BUFFER_CAPACITY;
/// Output capacity, in events, that `Scheduler::output_window_ready` keeps
/// free for the request's terminal events.
const OUTPUT_TERMINAL_RESERVE: usize = 2;

/// Cache extent of one prepared generation call, in tokens.
#[derive(Debug, Clone, Copy)]
struct CacheLengths {
    /// Tokens the call's cache-writing forward row appends: the prompt slice
    /// length, the feature write's token capacity, or one for decode and
    /// verify. Zero for a block step, which only reads the request's
    /// prefix cache.
    input: u32,
    /// Scheduled visible cache length before the call.
    visible: u32,
}

/// Output events a call kind with no entry in `output_bound` can add to its
/// request's event journal.
const DEFAULT_OUTPUT_EVENTS: usize = 2;

/// The most drafts one verification carries, which bounds the tokens it
/// commits: a request's event journal (`EVENT_WINDOW` events beyond those
/// consumed) holds a verification's tokens with room to spare.
pub(crate) const MAX_VERIFY_DRAFTS: usize = 16;

/// Returns the maximum number of output events one call of `kind` can
/// add to its request's event journal.
///
/// `Scheduler::output_window_ready` charges this for every in-flight call and
/// for the call the request schedules next.
fn output_bound(kind: CallKind) -> usize {
    match kind {
        CallKind::Forward(ForwardMode::Prefill) | CallKind::Forward(ForwardMode::Decode) => 4,
        // A verification's tokens each take an event beside a decode's.
        CallKind::Forward(ForwardMode::Verify) => 4 + MAX_VERIFY_DRAFTS,
        CallKind::Media(MediaCall::Denoising) => 3,
        _ => DEFAULT_OUTPUT_EVENTS,
    }
}
