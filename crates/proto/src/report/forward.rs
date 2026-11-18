//! Forward and kernel counters a worker reports with its batch results.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::histogram::Histogram;

/// One state of one observed stream's device time: the key of
/// [`ForwardStats::stream_us`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StreamTime {
    /// The stream: `device` (the rank's primary stream) or `copy` (the stream
    /// of page copies through the host KV cache).
    pub stream: String,
    /// `busy`, `gap` or `late` ([`ForwardStats::stream_us`]).
    pub state: String,
}

/// One driver call one serving thread made that waited for the device: the
/// key of [`ForwardStats::blocking_calls`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BlockingCall {
    /// The thread (`builder`, `control`, `device`).
    pub thread: String,
    /// The driver call.
    pub call: String,
}

/// One module class of the steps of one stage: the key of
/// [`ForwardStats::fence_seconds`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FenceStep {
    /// The steps' stage: their call kind, such as `decode`.
    pub stage: String,
    /// The module class (the name its modules give their profiler range), or
    /// `step` for the step's span between its markers.
    pub module: String,
}

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
    /// Device time of the rank's primary and copy streams in microseconds,
    /// by stream and state ([`StreamTime`]), from the profiler's records of
    /// the kernels, copies and fills on each stream, graph nodes included:
    /// `busy` is the length of the union of their intervals, `gap` the rest
    /// of the span from the stream's first operation to its latest end (idle
    /// while it waited for the host, another stream or any work), and `late`
    /// the time of records that arrived too far out of order to merge,
    /// counted in neither.
    pub stream_us: BTreeMap<StreamTime, u64>,
    /// Driver calls that made a serving thread of the rank wait for the
    /// device, by thread and call ([`BlockingCall`]). A serving call of the
    /// worker that made one failed.
    pub blocking_calls: BTreeMap<BlockingCall, u64>,
    /// Per-step durations from the fence probes in seconds, by stage and module class
    /// ([`FenceStep`]): a step's summed durations of the module class's
    /// probed instances, or the step's span between its markers for the
    /// class `step`, of the steps of that stage.
    pub fence_seconds: BTreeMap<FenceStep, Histogram>,
    /// Fence probe records the rank's reader lost: overwritten before it
    /// read them.
    pub fence_records_lost: u64,
    /// Each profiling session's progress on the rank, by session.
    pub profile: BTreeMap<u64, ProfileProgress>,
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

/// One profiling session's progress on a rank since its previous report, or summed over
/// reports and ranks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileProgress {
    /// The steps of its window the rank recorded.
    pub steps: u64,
    /// The bytes its file grew by.
    pub bytes: u64,
    /// The ranks whose window ended: 1 in the report of the rank's window's end.
    pub windows_ended: u64,
}

impl ProfileProgress {
    /// Adds `other`'s progress, saturating at the wire representation's limit.
    pub fn add(&mut self, other: &Self) {
        self.steps = self.steps.saturating_add(other.steps);
        self.bytes = self.bytes.saturating_add(other.bytes);
        self.windows_ended = self.windows_ended.saturating_add(other.windows_ended);
    }
}

impl ForwardStats {
    /// Add another invocation's counters, retaining every keyed breakdown.
    /// Counters saturate at the wire representation's unsigned 64-bit limit.
    pub fn merge(&mut self, other: &Self) {
        add_counts(&mut self.stream_us, &other.stream_us);
        add_counts(&mut self.blocking_calls, &other.blocking_calls);
        for (target, values) in [
            (&mut self.mode_counts, &other.mode_counts),
            (&mut self.mode_tokens, &other.mode_tokens),
            (&mut self.mode_us, &other.mode_us),
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
            add_counts(target, values);
        }
        for (session, progress) in &other.profile {
            self.profile.entry(*session).or_default().add(progress);
        }
        for (name, histogram) in &other.fence_seconds {
            self.fence_seconds
                .entry(name.clone())
                .or_default()
                .merge(histogram);
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
            (&mut self.fence_records_lost, other.fence_records_lost),
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
            && self.stream_us.is_empty()
            && self.blocking_calls.is_empty()
            && self.fence_seconds.is_empty()
            && self.fence_records_lost == 0
            && self.profile.is_empty()
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
            && self.spec_verify_rows == 0
            && self.spec_verify_draft_tokens == 0
            && self.spec_verify_accepted_tokens == 0
            && self.spec_verify_rejected_tokens == 0
            && self.spec_verify_committed_tokens == 0
            && self.spec_verify_path_counts.is_empty()
    }
}

/// Adds `values`' counts into `target`'s, key by key, saturating at the wire's unsigned 64-bit
/// limit.
fn add_counts<K: Ord + Clone>(target: &mut BTreeMap<K, u64>, values: &BTreeMap<K, u64>) {
    for (key, &value) in values {
        let count = target.entry(key.clone()).or_default();
        *count = count.saturating_add(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Merging two spans of worker counters adds their fence histograms step class by step
    /// class, and a span with only fence durations is not empty.
    #[test]
    fn merged_spans_add_their_fence_durations() {
        let step = FenceStep {
            stage: "decode".to_owned(),
            module: "step".to_owned(),
        };
        let span = |seconds: &[f64]| {
            let mut histogram = Histogram::new();
            for &value in seconds {
                histogram.observe(value);
            }
            ForwardStats {
                fence_seconds: [(step.clone(), histogram)].into(),
                ..ForwardStats::default()
            }
        };

        let mut stats = ForwardStats::default();
        assert!(stats.is_empty());
        stats.merge(&span(&[0.002]));
        stats.merge(&span(&[0.004, 0.006]));
        assert!(!stats.is_empty());

        let merged = &stats.fence_seconds[&step];
        assert_eq!(merged.count(), 3);
        assert!((merged.sum() - 0.012).abs() < 1e-12);
        assert_eq!(merged.max(), Some(0.006));
    }
}
