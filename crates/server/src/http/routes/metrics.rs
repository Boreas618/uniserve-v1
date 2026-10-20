//! OpenMetrics exposition endpoint.
//!
//! Renders the process-global `uniserve_observability::METRICS` registry, which
//! the HTTP middleware and the engine update as they run.

use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use thiserror_ext::AsReport;
use uniserve_observability::METRICS;

const OPENMETRICS_CONTENT_TYPE: &str = "application/openmetrics-text; version=1.0.0; charset=utf-8";

/// Renders the current process registry in OpenMetrics text format.
pub(super) async fn scrape() -> Response {
    match METRICS.render() {
        Ok(body) => (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static(OPENMETRICS_CONTENT_TYPE),
            )],
            body,
        )
            .into_response(),

        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to render metrics: {}", error.as_report()),
        )
            .into_response(),
    }
}
