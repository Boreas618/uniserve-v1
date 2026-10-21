//! HTTP request counters and latency metrics for the API server.
//!
//! The server's `track_http_metrics` middleware records all three families
//! once per request, when the response body finishes or is dropped, so
//! streamed responses count their full delivery time. Requests matched to the
//! operational routes in its `EXCLUDED_HANDLERS` (`/metrics`, `/health`,
//! `/version`) are not recorded. Family names, labels, and bucket layouts
//! match the request-count and latency metrics of the Python
//! prometheus-fastapi-instrumentator package's default instrumentation.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::Histogram;

use crate::U64Counter;

/// Upper bounds, in seconds, of the labeled per-handler latency histogram.
const HTTP_REQUEST_DURATION_BUCKETS: [f64; 3] = [0.1, 0.5, 1.0];
/// Upper bounds, in seconds, of the unlabeled high-resolution histogram.
const HTTP_REQUEST_DURATION_HIGHR_BUCKETS: [f64; 21] = [
    0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
    7.5, 10.0, 30.0, 60.0,
];

/// Builds one series of the per-method, per-handler latency histogram family.
fn http_request_duration_histogram() -> Histogram {
    Histogram::new(HTTP_REQUEST_DURATION_BUCKETS.iter().copied())
}

/// Builds the unlabeled high-resolution latency histogram.
fn http_request_duration_highr_histogram() -> Histogram {
    Histogram::new(HTTP_REQUEST_DURATION_HIGHR_BUCKETS.iter().copied())
}

/// Labels identifying an HTTP request by method, status, and handler.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct HttpRequestLabels {
    /// HTTP request method.
    pub method: String,
    /// Status class of the response head: `1xx` through `5xx`, or `unknown`.
    pub status: &'static str,
    /// Matched route template, or `none` when no route matched.
    pub handler: String,
}

/// Labels identifying an HTTP handler latency series.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct HttpHandlerLabels {
    /// HTTP request method.
    pub method: String,
    /// Matched route template, or `none` when no route matched.
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
    /// High-resolution HTTP request latency across all tracked handlers,
    /// without labels.
    pub http_request_duration_highr_seconds: Histogram,
}

impl ApiServerMetrics {
    /// Registers every family under its exported name and help, and
    /// returns the handles; the registry keeps a clone of each, whose state
    /// it shares.
    pub fn register(registry: &mut prometheus_client::registry::Registry) -> Self {
        let http_requests: HttpRequestCounterFamily = Default::default();
        registry.register(
            "http_requests",
            "Total number of HTTP requests by method, status, and handler.",
            http_requests.clone(),
        );
        let http_request_duration_seconds = HttpHandlerHistogramFamily::new_with_constructor(
            http_request_duration_histogram as fn() -> Histogram,
        );
        registry.register(
            "http_request_duration_seconds",
            "Duration of HTTP requests in seconds grouped by method and handler.",
            http_request_duration_seconds.clone(),
        );
        let http_request_duration_highr_seconds = http_request_duration_highr_histogram();
        registry.register(
            "http_request_duration_highr_seconds",
            "High-resolution duration of HTTP requests in seconds.",
            http_request_duration_highr_seconds.clone(),
        );
        Self {
            http_requests,
            http_request_duration_seconds,
            http_request_duration_highr_seconds,
        }
    }
}
