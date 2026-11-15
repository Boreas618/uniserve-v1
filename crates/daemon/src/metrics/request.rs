//! Request latency, token and outcome metrics.
//!
//! Time histograms observe seconds over [`latency_buckets`]; token histograms observe a request's
//! token counts over [`count_buckets`] of the served context length.

use prometheus_client::encoding::{EncodeLabelSet, EncodeLabelValue, LabelValueEncoder};
use prometheus_client::metrics::family::{Family, MetricConstructor};
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

use proto::request::FinishReason;

use crate::metrics::{U64Counter, count_buckets, latency_buckets};

/// Labels identifying where a request was served: the served model, the replica (an
/// independent engine of the epoch) and the scheduler of that replica's engine.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RequestLabels {
    /// Served model identity.
    pub model_name: String,
    /// The replica that served the request.
    pub replica: u32,
    /// The scheduler that served the request; empty for a request the engine
    /// ended at submission, or one a successor took over without naming it.
    pub scheduler: Option<u32>,
}

/// How a request ended.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Outcome {
    /// Generation ended at its end of sequence, a stop condition or normally.
    Stop,
    /// The request reached its generated-token limit.
    Length,
    /// The repetition policy ended generation.
    Repetition,
    /// The cache could not hold the request's next tokens.
    CacheCapacity,
    /// The client left, the request was cancelled or aborted, or a successor
    /// daemon took it over: neither served nor failed here.
    Abort,
    /// Execution or output processing failed.
    Error,
}

impl Outcome {
    /// The outcome of a request that finished for `reason`.
    pub fn of(reason: &FinishReason) -> Self {
        match reason {
            FinishReason::Completed | FinishReason::Eos | FinishReason::Stop => Self::Stop,
            FinishReason::MaxTokens => Self::Length,
            FinishReason::Cancelled | FinishReason::Aborted => Self::Abort,
            FinishReason::Repetition => Self::Repetition,
            FinishReason::CacheCapacity => Self::CacheCapacity,
            FinishReason::Error => Self::Error,
        }
    }

    /// The outcome's label value.
    pub fn name(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::Repetition => "repetition",
            Self::CacheCapacity => "cache_capacity",
            Self::Abort => "abort",
            Self::Error => "error",
        }
    }
}

impl EncodeLabelValue for Outcome {
    fn encode(&self, encoder: &mut LabelValueEncoder) -> Result<(), std::fmt::Error> {
        std::fmt::Write::write_str(encoder, self.name())
    }
}

/// Labels identifying a request's outcome where it was served.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct OutcomeLabels {
    /// Where the request was served.
    #[prometheus(flatten)]
    pub served: RequestLabels,
    /// How the request ended.
    pub outcome: Outcome,
}

/// The bucket bounds a histogram family's series are made with.
#[derive(Debug, Clone)]
pub struct Buckets(std::sync::Arc<[f64]>);

impl MetricConstructor<Histogram> for Buckets {
    fn new_metric(&self) -> Histogram {
        Histogram::new(self.0.iter().copied())
    }
}

/// A histogram family keyed by where a request was served.
pub(crate) type HistogramFamily = Family<RequestLabels, Histogram, Buckets>;

/// Request-lifecycle Prometheus families.
///
/// The daemon records each request a scheduler accepted when it ends: its outcome, its
/// end-to-end latency and, when the engine measured it, its queue time; and the token counts and
/// time to first token of a request whose output reported its final usage.
pub struct RequestMetrics {
    /// Prompt tokens processed by each replica.
    pub prompt_tokens: Family<RequestLabels, U64Counter>,
    /// Generated tokens processed by each replica.
    pub generation_tokens: Family<RequestLabels, U64Counter>,
    /// Ended requests by outcome.
    pub request_outcomes: Family<OutcomeLabels, U64Counter>,
    /// Prompt-token count distribution per request.
    pub request_prompt_tokens: HistogramFamily,
    /// Generated-token count distribution per request.
    pub request_generation_tokens: HistogramFamily,
    /// Time-to-first-token distribution in seconds.
    pub time_to_first_token_seconds: HistogramFamily,
    /// End-to-end request latency distribution in seconds.
    pub e2e_request_latency_seconds: HistogramFamily,
    /// Time spent waiting for scheduler admission, in seconds.
    pub request_queue_time_seconds: HistogramFamily,
}

impl RequestMetrics {
    /// Registers every family under its exported name and help, and
    /// returns the handles; the registry keeps a clone of each, whose state
    /// it shares.
    /// The token count histograms' bounds reach `max_model_len`, the served context.
    pub fn register(registry: &mut Registry, max_model_len: u32) -> Self {
        let latency = Buckets(latency_buckets().collect());
        let counts = Buckets(count_buckets(max_model_len).into());
        let histograms = |registry: &mut Registry, name, help, buckets: &Buckets| {
            let family = HistogramFamily::new_with_constructor(buckets.clone());
            registry.register(name, help, family.clone());
            family
        };
        let prompt_tokens: Family<RequestLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:prompt_tokens",
            "Prompt tokens of the requests that finished.",
            prompt_tokens.clone(),
        );
        let generation_tokens: Family<RequestLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:generation_tokens",
            "Generated tokens of the requests that finished.",
            generation_tokens.clone(),
        );
        let request_outcomes: Family<OutcomeLabels, U64Counter> = Default::default();
        registry.register(
            "uniserve:request_outcomes",
            "Requests that ended, by how they ended.",
            request_outcomes.clone(),
        );
        Self {
            prompt_tokens,
            generation_tokens,
            request_outcomes,
            request_prompt_tokens: histograms(
                registry,
                "uniserve:request_prompt_tokens",
                "Prompt tokens per finished request.",
                &counts,
            ),
            request_generation_tokens: histograms(
                registry,
                "uniserve:request_generation_tokens",
                "Generated tokens per finished request.",
                &counts,
            ),
            time_to_first_token_seconds: histograms(
                registry,
                "uniserve:time_to_first_token_seconds",
                "Time from a request's start to its first visible output, in seconds.",
                &latency,
            ),
            e2e_request_latency_seconds: histograms(
                registry,
                "uniserve:e2e_request_latency_seconds",
                "Time from a request's start to its end as the daemon observed it, in seconds.",
                &latency,
            ),
            request_queue_time_seconds: histograms(
                registry,
                "uniserve:request_queue_time_seconds",
                "Time a request waited for scheduler admission, as the engine measured it, in \
                 seconds.",
                &latency,
            ),
        }
    }
}
