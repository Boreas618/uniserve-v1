//! Forward and kernel counters a worker reports with its batch results.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Worker-local forward/kernel counters folded into scheduler stats.
///
/// In a worker batch result (`BatchOutput::forward_stats` in the worker
/// protocol) the values cover the work reported in that result, and the engine
/// scheduler adds them into its cumulative `WorkerStats` counters. Values are
/// per-update deltas when carried in the engine's scheduler statistics
/// report.
///
/// The same counters are mirrored elsewhere, including the worker's forward
/// statistics, the engine's `WorkerStats` and `SchedulerStatsReporter`, and
/// the server's `record_scheduler_stats`. A field added here also needs an entry in
/// [`ForwardStats::merge`] and [`ForwardStats::is_empty`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ForwardStats {
    /// Forward executions grouped by runtime mode.
    pub mode_counts: BTreeMap<String, u64>,
    /// Query tokens computed by runtime mode. Modes without a token notion
    /// (encoder and decoder calls, standalone module invocations) have no
    /// entry, while `mode_counts` and `mode_us` still count them.
    pub mode_tokens: BTreeMap<String, u64>,
    /// Forward time in microseconds by runtime mode.
    pub mode_us: BTreeMap<String, u64>,
    /// Component execution time in microseconds.
    pub component_us: BTreeMap<String, u64>,
    /// Device time of the rank's primary and copy streams in microseconds,
    /// keyed `{stream}/{state}`, where `stream` is `device` (the primary
    /// stream) or `copy` (the stream of page copies through the host KV
    /// cache), from CUPTI's records of the kernels, copies and fills on
    /// each stream, graph nodes included: `busy` is the length of the union
    /// of their intervals, `gap` the rest of the span from the stream's
    /// first operation to its latest end (idle while it waited for the
    /// host, another stream or any work), and `late` the time of records
    /// that arrived too far out of order to merge, counted in neither.
    pub stream_us: BTreeMap<String, u64>,
    /// Driver calls that made a serving thread of the rank wait for the
    /// device, keyed `{thread}/{call}` (`builder`, `control`). A serving
    /// call of the executor that made one failed.
    pub blocking_calls: BTreeMap<String, u64>,
    /// Attention kernel launch count.
    pub attention_launches: u64,
    /// Cumulative attention kernel time in microseconds.
    pub attention_us: u64,
    /// Attention launches grouped by backend.
    pub attention_backend_counts: BTreeMap<String, u64>,
    /// CUDA graph capture count.
    pub cuda_graph_captures: u64,
    /// CUDA graph replay count.
    pub cuda_graph_replays: u64,
    /// CUDA graph lookup misses.
    pub cuda_graph_misses: u64,
    /// Executions that fell back from graph replay.
    pub cuda_graph_fallbacks: u64,
    /// Real query tokens of the batches CUDA graph executions computed.
    pub cuda_graph_unpadded_tokens: u64,
    /// Padding token slots CUDA graph buckets added beyond the real tokens.
    pub cuda_graph_padded_tokens: u64,
    /// CUDA graph dispatches grouped by runtime mode.
    pub cuda_graph_runtime_mode_counts: BTreeMap<String, u64>,
    /// Decode tokens served from the token relay.
    pub text_decode_token_relay_hits: u64,
    /// Decode token relay misses.
    pub text_decode_token_relay_misses: u64,
    /// Decode positions served from the position relay.
    pub text_decode_position_relay_hits: u64,
    /// Decode position relay misses.
    pub text_decode_position_relay_misses: u64,
    /// FlashInfer decode plan construction count.
    pub flashinfer_decode_plan_calls: u64,
    /// FlashInfer decode plan reuse count.
    pub flashinfer_decode_plan_reuses: u64,
    /// Rows represented by FlashInfer decode plans.
    pub flashinfer_decode_plan_rows: u64,
    /// Indices represented by FlashInfer decode plans.
    pub flashinfer_decode_plan_indices: u64,
    /// FlashInfer graph-decode plan construction count.
    pub flashinfer_decode_graph_plan_calls: u64,
    /// FlashInfer graph-decode plan reuse count.
    pub flashinfer_decode_graph_plan_reuses: u64,
    /// Rows processed by speculative verification.
    pub spec_verify_rows: u64,
    /// Draft tokens presented for speculative verification.
    pub spec_verify_draft_tokens: u64,
    /// Draft tokens accepted by speculative verification.
    pub spec_verify_accepted_tokens: u64,
    /// Draft tokens rejected by speculative verification.
    pub spec_verify_rejected_tokens: u64,
    /// Tokens committed after speculative verification.
    pub spec_verify_committed_tokens: u64,
    /// Speculative verification rows grouped by resolution path.
    pub spec_verify_path_counts: BTreeMap<String, u64>,
}

impl ForwardStats {
    /// Add another invocation's counters, retaining every keyed breakdown.
    /// Counters saturate at the wire representation's unsigned 64-bit limit.
    pub fn merge(&mut self, other: &Self) {
        for (target, values) in [
            (&mut self.mode_counts, &other.mode_counts),
            (&mut self.mode_tokens, &other.mode_tokens),
            (&mut self.mode_us, &other.mode_us),
            (&mut self.component_us, &other.component_us),
            (&mut self.stream_us, &other.stream_us),
            (&mut self.blocking_calls, &other.blocking_calls),
            (
                &mut self.attention_backend_counts,
                &other.attention_backend_counts,
            ),
            (
                &mut self.cuda_graph_runtime_mode_counts,
                &other.cuda_graph_runtime_mode_counts,
            ),
            (
                &mut self.spec_verify_path_counts,
                &other.spec_verify_path_counts,
            ),
        ] {
            for (name, &value) in values {
                let count = target.entry(name.clone()).or_default();
                *count = count.saturating_add(value);
            }
        }

        for (target, value) in [
            (&mut self.attention_launches, other.attention_launches),
            (&mut self.attention_us, other.attention_us),
            (&mut self.cuda_graph_captures, other.cuda_graph_captures),
            (&mut self.cuda_graph_replays, other.cuda_graph_replays),
            (&mut self.cuda_graph_misses, other.cuda_graph_misses),
            (&mut self.cuda_graph_fallbacks, other.cuda_graph_fallbacks),
            (
                &mut self.cuda_graph_unpadded_tokens,
                other.cuda_graph_unpadded_tokens,
            ),
            (
                &mut self.cuda_graph_padded_tokens,
                other.cuda_graph_padded_tokens,
            ),
            (
                &mut self.text_decode_token_relay_hits,
                other.text_decode_token_relay_hits,
            ),
            (
                &mut self.text_decode_token_relay_misses,
                other.text_decode_token_relay_misses,
            ),
            (
                &mut self.text_decode_position_relay_hits,
                other.text_decode_position_relay_hits,
            ),
            (
                &mut self.text_decode_position_relay_misses,
                other.text_decode_position_relay_misses,
            ),
            (
                &mut self.flashinfer_decode_plan_calls,
                other.flashinfer_decode_plan_calls,
            ),
            (
                &mut self.flashinfer_decode_plan_reuses,
                other.flashinfer_decode_plan_reuses,
            ),
            (
                &mut self.flashinfer_decode_plan_rows,
                other.flashinfer_decode_plan_rows,
            ),
            (
                &mut self.flashinfer_decode_plan_indices,
                other.flashinfer_decode_plan_indices,
            ),
            (
                &mut self.flashinfer_decode_graph_plan_calls,
                other.flashinfer_decode_graph_plan_calls,
            ),
            (
                &mut self.flashinfer_decode_graph_plan_reuses,
                other.flashinfer_decode_graph_plan_reuses,
            ),
            (&mut self.spec_verify_rows, other.spec_verify_rows),
            (
                &mut self.spec_verify_draft_tokens,
                other.spec_verify_draft_tokens,
            ),
            (
                &mut self.spec_verify_accepted_tokens,
                other.spec_verify_accepted_tokens,
            ),
            (
                &mut self.spec_verify_rejected_tokens,
                other.spec_verify_rejected_tokens,
            ),
            (
                &mut self.spec_verify_committed_tokens,
                other.spec_verify_committed_tokens,
            ),
        ] {
            *target = target.saturating_add(value);
        }
    }

    /// Returns whether every worker counter and breakdown is empty or zero.
    ///
    /// The engine reports an interval's worker counters as `None` when their
    /// delta is empty, so a field missing from this check is dropped from any
    /// interval in which only that field changed.
    pub fn is_empty(&self) -> bool {
        self.mode_counts.is_empty()
            && self.mode_tokens.is_empty()
            && self.mode_us.is_empty()
            && self.component_us.is_empty()
            && self.stream_us.is_empty()
            && self.blocking_calls.is_empty()
            && self.attention_launches == 0
            && self.attention_us == 0
            && self.attention_backend_counts.is_empty()
            && self.cuda_graph_captures == 0
            && self.cuda_graph_replays == 0
            && self.cuda_graph_misses == 0
            && self.cuda_graph_fallbacks == 0
            && self.cuda_graph_unpadded_tokens == 0
            && self.cuda_graph_padded_tokens == 0
            && self.cuda_graph_runtime_mode_counts.is_empty()
            && self.text_decode_token_relay_hits == 0
            && self.text_decode_token_relay_misses == 0
            && self.text_decode_position_relay_hits == 0
            && self.text_decode_position_relay_misses == 0
            && self.flashinfer_decode_plan_calls == 0
            && self.flashinfer_decode_plan_reuses == 0
            && self.flashinfer_decode_plan_rows == 0
            && self.flashinfer_decode_plan_indices == 0
            && self.flashinfer_decode_graph_plan_calls == 0
            && self.flashinfer_decode_graph_plan_reuses == 0
            && self.spec_verify_rows == 0
            && self.spec_verify_draft_tokens == 0
            && self.spec_verify_accepted_tokens == 0
            && self.spec_verify_rejected_tokens == 0
            && self.spec_verify_committed_tokens == 0
            && self.spec_verify_path_counts.is_empty()
    }
}
