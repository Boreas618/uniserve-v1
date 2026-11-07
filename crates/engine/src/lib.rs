//! Scheduling, KV cache management and request control of a node's main
//! process (`uniserve engine`).
//!
//! A deployment runs one scheduler per DP rank: a main process builds a
//! `SchedulerHandle` (`core`) over the devices it holds and runs the
//! `Scheduler` (`scheduler`) on a thread of its own. The process's control
//! thread submits and controls requests through the handle and receives each
//! request's events on its `EventRx` (`handle`). On its thread, the
//! `Scheduler` admits requests, assigns cache pages (`kv_cache`) and other
//! storage (`allocation`), and submits batches through `Submissions`
//! (`scheduler`) to the executor crate's `Batches`.
//!
//! The `testing` feature additionally exports `SimExecutor`, `SimEngine`, and
//! `BatchEvent` (`sim`), a GPU-free executor for scheduler and frontend tests,
//! which plugs in through the `Executor` trait, and `serve_requests`
//! (`service`), which serves the request interface over simulated schedulers
//! for the daemon's tests.

/// Scheduler-owned allocation of executor storage; only its `OutOfStorage`
/// error is public.
pub mod allocation;
mod clock;
mod core;
mod encoder_outputs;
mod executor;
mod handle;
mod host_tier;
mod kv_cache;
mod scheduler;
#[cfg(feature = "testing")]
mod service;
#[cfg(feature = "testing")]
mod sim;
mod spec;
mod trajectories;
mod warmup;

pub use crate::core::{
    EngineConfig, RankFacts, SchedulerHandle, WorkerGroupConfig, WorkerGroupRole, WorkerRank,
};
#[cfg(feature = "testing")]
pub use crate::executor::{BatchResult, CallResult, Executor};
pub use crate::executor::{
    BlockConfig, BlockDistribution, CommandOutcome, CommandResult, ExecuteError, ExecutionBatch,
    ExecutorError, ExecutorFailure, ExecutorInfo, ExecutorSubmitError, PageOffloads, PoolLayout,
    RankBuffer, RankEncoderPools, RequestDispatch, SequenceParallel, TransferBackend,
    TransferConfig, TransferConfigError, TransferEdge, WorkerGroupId,
};
pub use crate::handle::{
    CacheChange, Command, CommandSender, CommandWaker, Counters, EVENT_BUFFER_CAPACITY, EventRx,
    EventSendError, EventTx, HookBarrier, PolicyPrograms, PolicySlotHandle, StageApplied,
    StreamCancelCause, Submission, SubmitError, Wake, WeightsInFlight,
};
// The engine schedules requests in the request vocabulary its daemon
// submits them in.
pub use crate::kv_cache::{HostCacheSize, IndexedPage};
pub use crate::scheduler::SpecialTokenIds;
pub use crate::scheduler::submission::Submissions;
#[cfg(feature = "testing")]
pub use crate::scheduler::testing::packaged_module;
#[cfg(feature = "testing")]
pub use crate::scheduler::testing::packaged_policy;
pub use crate::scheduler::{
    Arrival, BatchCapacity, Candidate, DEFAULT_CHUNKED_PREFILL_SIZE, DEFAULT_MAX_BATCH,
    DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_RUNNING_REQUESTS, DomainStats, EncoderStats,
    ExecutionDomainStats, ExecutorStats, GeneralStats, GroupCache, Limits, PREFETCH_FLOOR_TOKENS,
    Parameters, ParkedCache, PhaseKind, PolicyBytes, PolicyProgram, PolicyReverted, PolicySlot,
    PolicySource, PolicyUsage, PrefixCacheUsage, PrefixStats, ProgramSource, Progress, RoutePolicy,
    RouteTrial, Scheduler, SchedulerConfig, SchedulerStats, SchedulerStatsReporter, Stage,
    TimingStats, TrialState, Validators, arrival_view, ceilings_of, phase_id,
};
#[cfg(feature = "testing")]
pub use crate::service::{EpochPolicy, serve_requests};
// The simulated engine is test support: `mod sim` is built only with the
// `testing` feature, so its surface is exported only with it. A production
// build that exported these would not compile - the module they come from is
// not there.
#[cfg(feature = "testing")]
pub use crate::sim::{
    BatchEvent, ContextLog, ContextWrite, Pace, SimEngine, SimExecutor, SimSpeculation,
    sim_candidate_logprob, sim_canvas_stop_step, sim_text_token,
};
pub use ipc::{MediaCall, VideoDenoiserInfo, VideoTaskInfo};
pub use proto::config::DEFAULT_BLOCK;
/// Serializable scheduler, cache, and executor performance snapshots.
pub use proto::report;
pub use proto::request::{
    ArtifactEvent, ArtifactMedia, CachePolicy, ConditionExtent, DiffusionRequest,
    DiffusionRequestError, EncoderInputs, Event, FeedbackImage, FeedbackNextToken, FeedbackSource,
    FinishReason, GeneratedImage, GenerationConstraint, GenerationConstraintParseError,
    GenerationFeatures, GenerationLimits, GenerationRequest, GenerationRequestError,
    GenerationResourceError, ImageEncoder, ImageEncoderInput, ImageGenerationConfig, ImageInput,
    ImagePixels, ImageTrigger, MediaKind, ModelDtype, ModelDtypeParseError, MultimodalInputs,
    PixelRange, PositionLogprobs, ReadoutRow, ReadoutSlot, RejectionKind, Request, StopReason,
    Trajectory,
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
