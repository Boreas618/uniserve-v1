//! Process-wide Prometheus registry and UniServe metric families.
//!
//! [`METRICS`] is the one registry the server exports. Each submodule defines
//! one group of families, listed here with the server component that updates
//! it:
//!
//! - `api_server`: the HTTP metrics middleware (`track_http_metrics`), when
//!   each response body finishes or is dropped.
//! - `scheduler`: the server engine's scheduler-stats export tasks
//!   (`record_scheduler_stats`), from periodic `SchedulerStats` snapshots.
//! - `request`: the server engine's `RequestStats`, once for each request an
//!   scheduler accepted, when the request ends; the server's `StatsLogger`
//!   reads its prompt and generation token counters.
//!
//! Every metric field is a `prometheus-client` handle whose clones share their
//! underlying state, so updates through a field reach the clone held by the
//! registry. Registration names are base names; `prometheus-client` appends
//! `_total` to counters when encoding.

use std::fmt;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;

use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

mod api_server;
pub(crate) mod log_stats;
mod request;
mod scheduler;
pub(crate) mod stats;

pub use api_server::{ApiServerMetrics, HttpHandlerLabels, HttpRequestLabels};
pub use request::{FinishedReasonLabels, RequestMetrics};
pub use scheduler::{
    EngineBackendLabels, EngineBlockingLabels, EngineComponentLabels, EngineDomainKindLabels,
    EngineDomainLabels, EngineLabels, EngineModeLabels, EnginePathLabels, EngineStreamLabels,
    SchedulerMetrics,
};

/// Unsigned Prometheus counter used by metric families.
pub type U64Counter = Counter<u64, AtomicU64>;
/// Unsigned Prometheus gauge used by metric families.
pub type U64Gauge = Gauge<u64, AtomicU64>;
/// Floating-point Prometheus gauge used by metric families.
pub type F64Gauge = Gauge<f64, AtomicU64>;
/// Histogram family keyed by engine identity.
///
/// The `fn() -> Histogram` constructor lets each family choose its bucket
/// layout through `Family::new_with_constructor`; every series created for a
/// new label set uses that layout.
pub(crate) type HistogramFamily = Family<EngineLabels, Histogram, fn() -> Histogram>;

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
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
        let (sum_us, counts) = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
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

/// Prometheus registry together with handles to every UniServe metric family.
pub struct Metrics {
    registry: Registry,
    /// Scheduler, cache, and worker execution metric families.
    pub scheduler: SchedulerMetrics,
    /// Request latency, token, and completion metric families.
    pub request: RequestMetrics,
    /// HTTP transport metric families.
    pub api_server: ApiServerMetrics,
}

impl Metrics {
    /// Constructs a metrics registry with every UniServe family registered.
    pub fn new() -> Self {
        let mut registry = Registry::default();
        let scheduler = SchedulerMetrics::register(&mut registry);
        let request = RequestMetrics::register(&mut registry);
        let api_server = ApiServerMetrics::register(&mut registry);

        Self {
            registry,
            scheduler,
            request,
            api_server,
        }
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
