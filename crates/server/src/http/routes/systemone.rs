//! HTTP handler for the System One decision readout (`POST /v1/systemone`).

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::AppState;

/// Answers one System One request.
///
/// The body is read as bytes so that malformed JSON is reported in the
/// endpoint's own `422` shape. A model without the readout answers `404`, as
/// for a route it does not serve.
pub(crate) async fn systemone(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let engine = state.engine();
    if !engine.serves_readout() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "detail": "Not Found" })),
        )
            .into_response();
    }
    match engine.systemone(&body).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}
