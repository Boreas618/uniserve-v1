//! Request latency, token, and completion metrics.
//!
//! Family names follow vLLM's `vllm:`-prefixed request metrics under a
//! `uniserve:` prefix. Time histograms observe seconds; token histograms
//! observe per-request token counts.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::Histogram;

use crate::metrics::{EngineLabels, HistogramFamily, U64Counter};

const TTFT_BUCKETS: [f64; 22] = [
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
    20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
];
const REQUEST_LATENCY_BUCKETS: [f64; 21] = [
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0,
    480.0, 960.0, 1920.0, 7680.0,
];

/// Builds logarithmic 1-2-5 histogram boundaries through an exact upper bound.
///
/// The explicit final boundary preserves the caller's requested maximum when it
/// does not lie on the 1-2-5 progression.
fn build_1_2_5_buckets(max_value: u32) -> Vec<f64> {
    let mut buckets = Vec::new();
    let mut exponent = 0;
    loop {
        for mantissa in [1_u32, 2, 5] {
            let value = mantissa * 10_u32.pow(exponent);
            if value <= max_value {
                buckets.push(value as f64);
            } else {
                if buckets.last().copied() != Some(max_value as f64) {
                    buckets.push(max_value as f64);
                }
                return buckets;
            }
        }
        exponent += 1;
    }
}

/// Builds the histogram used for request time to first token.
fn time_to_first_token_histogram() -> Histogram {
    Histogram::new(TTFT_BUCKETS.iter().copied())
}

/// Builds the histogram used for end-to-end and lifecycle-phase latency.
fn request_latency_histogram() -> Histogram {
    Histogram::new(REQUEST_LATENCY_BUCKETS.iter().copied())
}

/// Builds the histogram used for per-request token counts.
fn request_token_count_histogram() -> Histogram {
    // The bounds are fixed at registration rather than derived from the served
    // model's context length; counts above the last bound land only in the
    // `+Inf` bucket.
    Histogram::new(build_1_2_5_buckets(131_072))
}

/// Labels identifying a terminal request outcome.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct FinishedReasonLabels {
    /// Served model identity.
    pub model_name: String,
    /// Scheduler index.
    pub engine: u32,
    /// Stable terminal reason name.
    pub finished_reason: &'static str,
}

/// Completion counter family keyed by terminal outcome.
pub(crate) type FinishedReasonCounterFamily = Family<FinishedReasonLabels, U64Counter>;

/// Request-lifecycle Prometheus families.
///
/// The server records each request the engine accepted when it completes:
/// `request_success`, `e2e_request_latency_seconds`, and
/// `request_queue_time_seconds` for every such request, and `prompt_tokens`,
/// `generation_tokens`, `request_prompt_tokens`, `request_generation_tokens`,
/// `request_max_num_generation_tokens`, and `time_to_first_token_seconds` for
/// one whose output reported its final usage. The server's `StatsLogger`
/// reads `prompt_tokens` and `generation_tokens` to log throughput.
pub struct RequestMetrics {
    /// Prompt tokens processed by each engine.
    pub prompt_tokens: Family<EngineLabels, U64Counter>,
    /// Generated tokens processed by each engine.
    pub generation_tokens: Family<EngineLabels, U64Counter>,
    /// Successfully completed requests grouped by terminal reason.
    pub request_success: FinishedReasonCounterFamily,
    /// Prompt-token count distribution per request.
    pub request_prompt_tokens: HistogramFamily,
    /// Generated-token count distribution per request.
    pub request_generation_tokens: HistogramFamily,
    /// Requested maximum generated-token distribution.
    pub request_max_num_generation_tokens: HistogramFamily,
    /// Time-to-first-token distribution in seconds.
    pub time_to_first_token_seconds: HistogramFamily,
    /// End-to-end request latency distribution in seconds.
    pub e2e_request_latency_seconds: HistogramFamily,
    /// Time spent waiting for scheduler admission.
    pub request_queue_time_seconds: HistogramFamily,
}

impl RequestMetrics {
    /// Registers every family under its exported name and help, and
    /// returns the handles; the registry keeps a clone of each, whose state
    /// it shares.
    pub fn register(registry: &mut prometheus_client::registry::Registry) -> Self {
        let prompt_tokens: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:prompt_tokens",
            "Number of prefill tokens processed.",
            prompt_tokens.clone(),
        );
        let generation_tokens: Family<EngineLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:generation_tokens",
            "Number of generation tokens processed.",
            generation_tokens.clone(),
        );
        let request_success: FinishedReasonCounterFamily = Default::default();
        registry.register(
            "uniserve:request_success",
            "Count of successfully processed requests.",
            request_success.clone(),
        );
        let request_prompt_tokens =
            Family::new_with_constructor(request_token_count_histogram as fn() -> Histogram);
        registry.register(
            "uniserve:request_prompt_tokens",
            "Number of prefill tokens processed.",
            request_prompt_tokens.clone(),
        );
        let request_generation_tokens =
            Family::new_with_constructor(request_token_count_histogram as fn() -> Histogram);
        registry.register(
            "uniserve:request_generation_tokens",
            "Number of generation tokens processed.",
            request_generation_tokens.clone(),
        );
        let request_max_num_generation_tokens =
            Family::new_with_constructor(request_token_count_histogram as fn() -> Histogram);
        registry.register(
            "uniserve:request_max_num_generation_tokens",
            "Histogram of maximum number of requested generation tokens.",
            request_max_num_generation_tokens.clone(),
        );
        let time_to_first_token_seconds =
            Family::new_with_constructor(time_to_first_token_histogram as fn() -> Histogram);
        registry.register(
            "uniserve:time_to_first_token_seconds",
            "Histogram of time to first token in seconds.",
            time_to_first_token_seconds.clone(),
        );
        let e2e_request_latency_seconds =
            Family::new_with_constructor(request_latency_histogram as fn() -> Histogram);
        registry.register(
            "uniserve:e2e_request_latency_seconds",
            "Histogram of e2e request latency in seconds.",
            e2e_request_latency_seconds.clone(),
        );
        let request_queue_time_seconds =
            Family::new_with_constructor(request_latency_histogram as fn() -> Histogram);
        registry.register(
            "uniserve:request_queue_time_seconds",
            "Histogram of time spent in WAITING phase for request.",
            request_queue_time_seconds.clone(),
        );
        Self {
            prompt_tokens,
            generation_tokens,
            request_success,
            request_prompt_tokens,
            request_generation_tokens,
            request_max_num_generation_tokens,
            time_to_first_token_seconds,
            e2e_request_latency_seconds,
            request_queue_time_seconds,
        }
    }
}
