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
//!   engine core accepted, when the request ends; the server's `StatsLogger`
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
mod request;
mod scheduler;

pub use api_server::{ApiServerMetrics, HttpHandlerLabels, HttpRequestLabels};
pub use request::{FinishedReasonLabels, PromptTokenSourceLabels, RequestMetrics};
pub use scheduler::{
    EngineBackendLabels, EngineComponentLabels, EngineDomainKindLabels, EngineDomainLabels,
    EngineLabels, EngineModeLabels, EnginePathLabels, SchedulerMetrics, WaitingReasonLabels,
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

/// Process-global metrics registry, written and rendered by the server crate.
pub static METRICS: LazyLock<Metrics> = LazyLock::new(Metrics::new);
