//! Scheduling, KV cache management, worker execution and request control.
//!
//! The crate is the engine process (`uniserve engine`): it serves the request
//! interface to the leader's daemon (`service`) and drives the worker
//! processes.
//! A deployment runs one scheduler per DP rank: `SchedulerHandle` (`core`)
//! launches the worker processes, or takes a supplied executor, and runs the
//! `Scheduler` (`scheduler`) on a thread of its own. Frontends submit and
//! control requests through the handle and receive each request's events on
//! its `EventRx` (`handle`). On its thread, the `Scheduler` admits requests,
//! assigns cache pages (`kv_cache`) and other worker storage (`allocation`),
//! and submits calls through the `Executor` trait (`executor`);
//! `WorkerExecutor` (`worker`) splits each batch into per-worker submissions
//! to the worker processes.
//!
//! The `testing` feature additionally exports `SimExecutor`, `SimEngine`, and
//! `BatchEvent` (`sim`), a GPU-free executor for scheduler and frontend tests.

/// Scheduler-owned allocation of worker storage; only its `OutOfStorage`
/// error is public.
pub mod allocation;
mod clock;
mod core;
mod deployment;
mod executor;
mod handle;
mod kv_cache;
mod scheduler;
mod service;
#[cfg(feature = "testing")]
mod sim;
mod spec;
mod trajectories;
mod warmup;
mod worker;

pub use crate::core::{
    EngineConfig, SchedulerHandle, WorkerGroupConfig, WorkerGroupRole, WorkerRank,
};
pub use crate::deployment::Deployment;
pub use crate::executor::{
    BatchResult, CallResult, CommandOutcome, ComponentConfig, ComponentDistribution,
    ExecutionBatch, Executor, ExecutorError, ExecutorInfo, ExecutorSubmitError, PageOffloads,
    ParallelConfig, PoolPlacement, RankBuffer, RequestPlacement, SequenceParallel, TransferBackend,
    TransferConfig, TransferConfigError, TransferEdge, WorkerExecError, WorkerFailure,
    WorkerGroupId, WorkerResult,
};
pub use crate::handle::{
    CacheChange, Command, CommandSender, CommandWaker, EVENT_BUFFER_CAPACITY, EventRx,
    EventSendError, EventTx, HookBarrier, StreamCancelCause, Submission, SubmitError, Successor,
    Wake, WeightsInFlight,
};
// The engine schedules requests in the request vocabulary its daemon
// submits them in.
pub use crate::kv_cache::{HostCacheSize, IndexedPage};
pub use crate::scheduler::SpecialTokenIds;
pub use crate::scheduler::{
    DEFAULT_CHUNKED_PREFILL_SIZE, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_RUNNING_REQUESTS, DomainStats, EncoderStats, ExecutionDomainStats, GeneralStats,
    GroupCache, ParkedCache, PrefixCacheUsage, PrefixStats, RefillPolicy, Scheduler,
    SchedulerConfig, SchedulerStats, SchedulerStatsReporter, TimingStats, WorkerStats,
};
pub use crate::service::{run_engine, serve_requests};
#[cfg(feature = "testing")]
pub use crate::sim::{
    BatchEvent, SimEngine, SimExecutor, sim_candidate_logprob, sim_canvas_stop_step, sim_text_token,
};
pub use crate::worker::{
    BatchSubmitError, CallReaders, ExpertParallelPlacement, WorkerExecutor, WorkerGroup,
    WorkerProcessArgs,
};
pub use ipc::{DEFAULT_COMPONENT, MediaCall, VideoDenoiserInfo, VideoTaskInfo};
/// Serializable scheduler, cache, and worker performance snapshots.
pub use proto::report;
pub use proto::request::{
    ArtifactEvent, ArtifactMedia, CachePolicy, ConditionExtent, DiffusionRequest,
    DiffusionRequestError, Event, FeedbackNextToken, FeedbackSource, FinishReason,
    GenerationConstraint, GenerationConstraintParseError, GenerationFeatures, GenerationLimits,
    GenerationRequest, GenerationRequestError, GenerationResourceError, ImageEncoder,
    ImageEncoderInput, ImageGenerationConfig, ImageInput, ImageTrigger, MediaKind, ModelDtype,
    ModelDtypeParseError, MultimodalInputs, PositionLogprobs, ReadoutRow, ReadoutSlot,
    RejectionKind, Request, StopReason, Trajectory, encoder_cache_key,
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
