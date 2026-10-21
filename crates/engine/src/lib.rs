//! In-process scheduling, storage management, worker execution, and request control.
//!
//! The crate sits between the HTTP frontend (`uniserve-server`) and the
//! worker processes. `EngineCore` (`core`) launches the worker
//! processes, or takes a supplied executor, and runs the `Scheduler` on its
//! owner thread. Frontends submit and control requests through a cloneable
//! `EngineHandle` (`handle`) and receive each request's events on its
//! `EventRx`. On the owner thread, the `Scheduler` (`scheduler`) admits
//! requests, assigns cache pages (`prefix_cache`) and other worker storage (`storage`),
//! and submits calls through the `Executor` trait (`executor`);
//! `WorkerExecutor` (`worker`) splits each batch into per-worker submissions
//! to the worker processes.
//!
//! The `testing` feature additionally exports `SimExecutor`, `SimEngine`, and
//! `BatchEvent` (`sim`), a GPU-free executor for scheduler and frontend tests.

mod core;
mod deployment;
mod executor;
mod handle;
mod prefix_cache;
mod scheduler;
#[cfg(feature = "testing")]
mod sim;
/// Scheduler-owned allocation of worker storage; only its `OutOfStorage`
/// error is public.
pub mod storage;
mod warmup;
mod worker;

pub use crate::core::{EngineConfig, EngineCore, WorkerGroupConfig, WorkerGroupRole, WorkerRank};
pub use crate::deployment::Deployment;
pub use crate::executor::{
    BatchResult, CallResult, CommandOutcome, ComponentConfig, ComponentDistribution,
    ExecutionBatch, Executor, ExecutorError, ExecutorInfo, ExecutorSubmitError, PageWrites,
    ParallelConfig, PoolPlacement, RankBuffer, RequestPlacement, SequenceParallel, TransferBackend,
    TransferConfig, TransferConfigError, TransferEdge, WorkerExecError, WorkerFailure,
    WorkerGroupId, WorkerResult,
};
pub use crate::handle::{
    Command, EVENT_BUFFER_CAPACITY, EngineHandle, EventRx, EventSendError, EventTx,
    StreamCancelCause, Submission, SubmitError,
};
pub use crate::scheduler::SpecialTokenIds;
pub use crate::scheduler::{
    DEFAULT_CHUNKED_PREFILL_SIZE, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_RUNNING_REQUESTS, DomainStats, EncoderStats, ExecutionDomainStats, GeneralStats,
    PrefixCacheUsage, PrefixStats, Scheduler, SchedulerConfig, SchedulerStats,
    SchedulerStatsReporter, SchedulingPolicy, TimingStats, WorkerStats,
};
#[cfg(feature = "testing")]
pub use crate::sim::{
    BatchEvent, SimEngine, SimExecutor, sim_candidate_logprob, sim_canvas_stop_step, sim_text_token,
};
pub use crate::worker::{
    BatchSubmitError, CallReaders, ExpertExchange, ExpertParallelPlacement, FlashInferBackend,
    FlashInferBackendParseError, GraphPolicy, LaneConfig, LaneSpec, LoadFormat, WorkerExecutor,
    WorkerGroup, WorkerProcessArgs,
};
pub use uniserve_ipc::{DEFAULT_COMPONENT, MediaCall, VideoDenoiserInfo};

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
