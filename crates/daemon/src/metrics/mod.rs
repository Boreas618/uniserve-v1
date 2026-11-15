//! The daemon's Prometheus registry and UniServe's metric families.
//!
//! [`METRICS`] is the registry the daemon exports. Each submodule defines one group of families,
//! listed here with the code that updates it:
//!
//! - `api_server`: the HTTP metrics route layer (`track_http_metrics`), when each response body
//!   finishes or is dropped.
//! - `scheduler`: each scheduler's interval report ([`Metrics::record_scheduler`]).
//! - `request`: each request a scheduler accepted, when it ends (`stats::StatLogger`).
//! - `memory`: each replica's main process report of its allocators.
//!
//! The families are UniServe's own schema: names under `uniserve:`, one name per quantity (a
//! report's scalar is exported under its field's name), latency histograms over
//! [`latency_buckets`] and count histograms over [`count_buckets`]. The statistics log
//! (`log_stats`) is fed by the same recorders and reads no family back.
//!
//! Every metric field is a `prometheus-client` handle whose clones share their underlying
//! state, so updates through a field reach the clone held by the registry. Registration names
//! are base names; `prometheus-client` appends `_total` to counters when encoding.

use std::fmt;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;

use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::registry::Registry;

mod api_server;
pub(crate) mod log_stats;
mod memory;
mod request;
mod scheduler;
pub(crate) mod stats;

pub use api_server::{ApiServerMetrics, HttpHandlerLabels, HttpRequestLabels};
pub use memory::{DeviceLabels, MemoryMetrics, PoolLabels};
pub use request::{Outcome, OutcomeLabels, RequestLabels, RequestMetrics};
pub use scheduler::{
    BackendLabel, BlockingLabel, DomainKindLabel, DomainLabel, FenceLabel, KeyedLabels, ModeLabel,
    PathLabel, ScalarMetrics, SchedulerLabels, SchedulerMetrics, StreamLabel,
};

/// Unsigned Prometheus counter used by metric families.
pub type U64Counter = Counter<u64, AtomicU64>;
/// Unsigned Prometheus gauge used by metric families.
pub type U64Gauge = Gauge<u64, AtomicU64>;
/// Floating-point Prometheus gauge used by metric families.
pub type F64Gauge = Gauge<f64, AtomicU64>;

/// The bucket bounds, in seconds, of every latency histogram: powers of two from 1 ms to about
/// 35 minutes.
///
/// A latency spans a cached decode step to a long video generation, five orders of magnitude, so
/// the bounds double: any quantile read from the histogram is within a factor of two of the
/// observation at every scale, with 22 buckets.
pub fn latency_buckets() -> impl Iterator<Item = f64> {
    prometheus_client::metrics::histogram::exponential_buckets(0.001, 2.0, 22)
}

/// The bucket bounds of every count histogram (a request's prompt or generated tokens): powers
/// of two from 1 to 2^20.
///
/// The bounds double for the reason [`latency_buckets`] states, and reach past the longest
/// context any served model states.
pub fn count_buckets() -> impl Iterator<Item = f64> {
    prometheus_client::metrics::histogram::exponential_buckets(1.0, 2.0, 21)
}

/// A histogram whose observations arrive already counted per bucket of
/// [`proto::report::FENCE_BUCKETS_US`] (`proto::report::FenceHistogram`): the workers count each
/// step's durations and the daemon adds their counts. Values are exported
/// in seconds.
#[derive(Debug, Default, Clone)]
pub struct CountedHistogram {
    /// The sum in microseconds and the count of each bucket, the last past
    /// the largest bound.
    inner: std::sync::Arc<std::sync::Mutex<(u64, Vec<u64>)>>,
}

impl CountedHistogram {
    /// Add a span's counts.
    pub fn add(&self, histogram: &proto::report::FenceHistogram) {
        let mut inner = ::service::lock(&self.inner);
        if inner.1.is_empty() {
            inner.1 = vec![0; proto::report::FENCE_BUCKETS_US.len() + 1];
        }
        for (count, &more) in inner.1.iter_mut().zip(&histogram.buckets) {
            *count = count.saturating_add(more);
        }
        inner.0 = inner.0.saturating_add(histogram.sum_us);
    }
}

impl prometheus_client::metrics::TypedMetric for CountedHistogram {
    const TYPE: prometheus_client::metrics::MetricType =
        prometheus_client::metrics::MetricType::Histogram;
}

impl prometheus_client::encoding::EncodeMetric for CountedHistogram {
    fn encode(
        &self,
        mut encoder: prometheus_client::encoding::MetricEncoder,
    ) -> Result<(), fmt::Error> {
        let (sum_us, counts) = ::service::lock(&self.inner).clone();
        // Per-bucket counts with their upper bounds in seconds, the last
        // unbounded, as `prometheus_client`'s own histogram hands them over.
        let buckets: Vec<(f64, u64)> = proto::report::FENCE_BUCKETS_US
            .iter()
            .map(|&bound| bound as f64 / 1e6)
            .chain([f64::MAX])
            .zip(counts.iter().copied().chain(std::iter::repeat(0)))
            .collect();
        let count = counts.iter().sum();
        encoder.encode_histogram::<prometheus_client::encoding::NoLabelSet>(
            sum_us as f64 / 1e6,
            count,
            &buckets,
            None,
        )
    }

    fn metric_type(&self) -> prometheus_client::metrics::MetricType {
        <Self as prometheus_client::metrics::TypedMetric>::TYPE
    }
}

/// Prometheus registry together with handles to every UniServe metric family, and the
/// statistics log its recorders feed.
pub struct Metrics {
    registry: Registry,
    /// Scheduler, cache, and worker execution metric families.
    pub scheduler: SchedulerMetrics,
    /// Request latency, token, and outcome metric families.
    pub request: RequestMetrics,
    /// HTTP transport metric families.
    pub api_server: ApiServerMetrics,
    /// Each device's allocator budget and the bytes its pools have mapped.
    pub memory: MemoryMetrics,
    /// What the statistics log line states, fed by the same recorders.
    pub(crate) log: log_stats::StatsLog,
}

impl Metrics {
    /// Constructs a metrics registry with every UniServe family registered.
    pub fn new() -> Self {
        let mut registry = Registry::default();
        let scheduler = SchedulerMetrics::register(&mut registry);
        let request = RequestMetrics::register(&mut registry);
        let api_server = ApiServerMetrics::register(&mut registry);
        let memory = MemoryMetrics::register(&mut registry);

        Self {
            registry,
            scheduler,
            request,
            api_server,
            memory,
            log: log_stats::StatsLog::default(),
        }
    }

    /// Records one scheduler's interval report: its families, and the
    /// statistics log's interval.
    pub(crate) fn record_scheduler(
        &self,
        labels: &SchedulerLabels,
        stats: &proto::report::SchedulerStats,
    ) {
        self.scheduler.record(labels, stats);
        self.log.scheduler(labels, stats);
    }

    /// Renders the current metrics registry into OpenMetrics text format.
    ///
    /// # Errors
    ///
    /// Returns the formatter error raised while encoding the registry.
    pub fn render(&self) -> Result<String, fmt::Error> {
        let mut output = String::new();
        encode(&mut output, &self.registry)?;
        Ok(output)
    }
}

impl Default for Metrics {
    /// Constructs the fully registered default metrics collection.
    fn default() -> Self {
        Self::new()
    }
}

/// Process-global metrics registry, written and rendered by the leader's
/// daemon.
pub static METRICS: LazyLock<Metrics> = LazyLock::new(Metrics::new);
