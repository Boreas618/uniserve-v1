//! HTTP request counters and latency metrics for the API server.
//!
//! The server's `track_http_metrics` route layer records both families once per request a route
//! matched, when the response body finishes or is dropped, so streamed responses count their full
//! delivery time. The operational routes (`/metrics`, `/health`, `/version`) are outside that
//! layer and are not recorded.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

use crate::metrics::{U64Counter, latency_buckets};

/// A latency histogram over [`latency_buckets`].
fn latency_histogram() -> Histogram {
    Histogram::new(latency_buckets())
}

/// Labels identifying an HTTP request by method, status, and handler.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct HttpRequestLabels {
    /// HTTP request method.
    pub method: String,
    /// Status class of the response head: `1xx` through `5xx`, or `unknown`.
    pub status: &'static str,
    /// Matched route template.
    pub handler: String,
}

/// Labels identifying an HTTP handler latency series.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct HttpHandlerLabels {
    /// HTTP request method.
    pub method: String,
    /// Matched route template.
    pub handler: String,
}

/// Counter family keyed by HTTP request labels.
pub(crate) type HttpRequestCounterFamily = Family<HttpRequestLabels, U64Counter>;
/// Latency histogram family keyed by HTTP handler labels.
pub(crate) type HttpHandlerHistogramFamily =
    Family<HttpHandlerLabels, Histogram, fn() -> Histogram>;

/// API-server Prometheus families exported from the HTTP middleware layer.
pub struct ApiServerMetrics {
    /// HTTP request count grouped by method, status, and handler.
    pub http_requests: HttpRequestCounterFamily,
    /// HTTP request latency grouped by method and handler.
    pub http_request_duration_seconds: HttpHandlerHistogramFamily,
}

impl ApiServerMetrics {
    /// Registers every family under its exported name and help, and
    /// returns the handles; the registry keeps a clone of each, whose state
    /// it shares.
    pub fn register(registry: &mut Registry) -> Self {
        let http_requests: HttpRequestCounterFamily = Default::default();
        registry.register(
            "uniserve:http_requests",
            "HTTP requests by method, status class and matched route.",
            http_requests.clone(),
        );
        let http_request_duration_seconds =
            HttpHandlerHistogramFamily::new_with_constructor(latency_histogram);
        registry.register(
            "uniserve:http_request_duration_seconds",
            "Time from an HTTP request's arrival to the end of its response body, in seconds, by \
             method and matched route.",
            http_request_duration_seconds.clone(),
        );
        Self {
            http_requests,
            http_request_duration_seconds,
        }
    }
}
