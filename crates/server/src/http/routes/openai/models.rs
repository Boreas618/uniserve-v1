//! HTTP handler for the served-model listing.

use std::sync::Arc;

use crate::openai::{ListModelsResponse, ModelObject};
use axum::Json;
use axum::extract::State;

use crate::AppState;

/// Returns the served model in OpenAI `list models` format.
///
/// A server serves exactly one model, so the list always has one entry, named
/// by `AppState::served_model_name`.
pub(crate) async fn list_models(State(state): State<Arc<AppState>>) -> Json<ListModelsResponse> {
    // OpenAI clients expect `created` to be a real Unix timestamp; the listing
    // reports the time it was produced.
    let created = uniserve_core::now_unix_secs_u64() as i64;
    Json(ListModelsResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: state.served_model_name().to_string(),
            object: "model".to_string(),
            created,
            owned_by: "uniserve".to_string(),
        }],
    })
}
