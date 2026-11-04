//! What one call's completion carries, and what one batch's does, as the
//! mechanism reads them.
//!
//! Design §4.4's completion path is the mechanism's own: it is fed by whichever
//! executor ran the batch, and it must not be shaped by the wire protocol one of
//! them happens to speak. This module is that shape. `notes/sched/completion.md`
//! states, field by field, whether the device produces a value or the mechanism
//! already owns it; the short version is that the executor states *what
//! execution did* — the tokens it accepted, the cache extent it initialized and
//! what a successor may read, the log-probabilities a client asked for, the
//! artifact and the cache export — and the mechanism keeps *its own rules*: the
//! stop conditions are evaluated here from the sampled token
//! (`OutputSender::und_token_outcome`), and the call's budget, cache window and
//! sampling rows are the plan's.
//!
//! Two fillers produce these types, one per executor, both at the edge of the
//! mechanism and neither inventing a value:
//!
//! - [`CallCompletion::from_wire`] / [`BatchReport::from_wire`] for the
//!   worker-process executor, which speaks `ipc`. They are temporary: they are
//!   deleted with that executor (§2.4).
//! - the in-process filler, from `executor::BatchOutput`, written when the
//!   device executor reports what the completion path reads. Until it does, the
//!   in-process path has no completion to report — it does not stand one in.

use std::sync::Arc;

use ipc::{
    CacheTransfer, CallId, CallStatus, ErrorCode, ForwardStats, MediaOutput, RankLatentLayout,
    RequestKey, SharedMedia, SharedMediaError, TimingCounters, TokenLogprob,
};

use crate::executor::{BatchResult, CallResult};

/// One call's completion, as the mechanism reads it.
#[derive(Clone, Debug)]
pub(crate) struct CallCompletion {
    /// The request the call advanced.
    pub request_key: RequestKey,
    /// The request-local call identifier.
    pub call_id: CallId,
    /// Whether execution performed the call.
    pub status: CallStatus,
    /// Why it failed, when it did.
    pub error_code: Option<ErrorCode>,
    /// What the rank said about a refused input, when it refused one.
    pub error_message: Option<String>,
    /// The call's own measurements: folded into its domain's statistics.
    pub timing_counters: TimingCounters,
    /// The accepted cache prefix a successor may read.
    pub cache_visible_len: u32,
    /// The cache extent execution initialized, rejected speculative positions
    /// included.
    pub cache_computed_len: u32,
    /// A denoising trajectory's next step.
    pub step_index: u32,
    /// The tokens the sampler accepted.
    pub committed_tokens: Vec<u32>,
    /// Each committed token's log-probabilities, in `committed_tokens` order.
    pub token_logprobs: Vec<Vec<TokenLogprob>>,
    /// Ranked candidates for each scored prompt position, in input order.
    pub prompt_logprobs: Vec<Vec<TokenLogprob>>,
    /// A token-denoising readout's candidate probabilities.
    pub candidate_logprobs: Vec<f32>,
    /// The finished artifact and its external storage lifetime, when produced.
    pub media_output: Option<MediaOutput>,
    /// The physical cache export, when the call publishes one.
    pub cache_output: Option<CacheTransfer>,
    /// The drafter's proposals after the last committed token.
    pub draft_token_ids: Vec<u32>,
    /// A video latent preparation's per-rank page layout.
    pub latent_layouts: Vec<RankLatentLayout>,
    /// The immutable storage the device published, claimed by the mechanism —
    /// or its claim's error, which belongs to the request that asked for it.
    pub media: Result<Option<Arc<SharedMedia>>, SharedMediaError>,
}

impl CallCompletion {
    /// The worker-process edge: what a rank reported, in the mechanism's type.
    ///
    /// A worker's `finish_flags` are dropped rather than carried: the stop
    /// conditions are the request's own rules and the mechanism evaluates them
    /// from the sampled token, so a second, device-side copy of that rule would
    /// be a weaker one. Temporary — this filler goes with the worker-process
    /// executor.
    pub(crate) fn from_wire(result: CallResult) -> Self {
        let output = result.output;
        Self {
            request_key: output.request_key,
            call_id: output.call_id,
            status: output.status,
            error_code: output.error_code,
            error_message: output.error_message,
            timing_counters: output.timing_counters,
            cache_visible_len: output.cache_visible_len,
            cache_computed_len: output.cache_computed_len,
            step_index: output.step_index,
            committed_tokens: output.committed_tokens,
            token_logprobs: output.token_logprobs,
            prompt_logprobs: output.prompt_logprobs,
            candidate_logprobs: output.candidate_logprobs,
            media_output: output.media_output,
            cache_output: output.cache_output,
            draft_token_ids: output.draft_token_ids,
            latent_layouts: output.latent_layouts,
            media: result.media,
        }
    }
}

/// One batch's completion, as the mechanism reads it.
#[derive(Clone, Debug)]
pub(crate) struct BatchReport {
    /// The batch's number, which the mechanism assigned at assembly.
    pub batch_id: u64,
    /// One completion per call that finished, in the executor's order.
    pub calls: Vec<CallCompletion>,
    /// The receipts of the batch's lifecycle commands.
    pub command_results: Vec<crate::executor::CommandResult>,
    /// Whether this report ends the batch.
    pub done: bool,
    /// Per-worker execution durations, in microseconds.
    pub worker_exec_us: Vec<u64>,
    /// Per-worker model-forward statistics.
    pub forward_stats: Vec<ForwardStats>,
}

impl BatchReport {
    /// The worker-process edge: a logical result, in the mechanism's type.
    /// Temporary, with [`CallCompletion::from_wire`].
    pub(crate) fn from_wire(result: BatchResult) -> Self {
        Self {
            batch_id: result.batch_id,
            calls: result
                .results
                .into_iter()
                .map(CallCompletion::from_wire)
                .collect(),
            command_results: result.command_results,
            done: result.done,
            worker_exec_us: result.worker_exec_us,
            forward_stats: result.forward_stats,
        }
    }
}
