//! The OpenAI-compatible API: wire schemas, errors, and the route handlers.
//!
//! Each endpoint module builds its own router over the [`Engine`]
//! (`chat_completions`, `images`, `videos`); `crate::http::router` registers
//! the routers of the tasks the engine serves. A request is deserialized into
//! the strict schemas of `protocol` (unknown fields are rejected) and
//! validated by [`ValidatedJson`]; the engine lowers, tokenizes and submits
//! it, and its `RequestOutput` stream comes back to the endpoint module for
//! response assembly. Every failure answers in the OpenAI error envelope of
//! [`ApiError`].

pub mod chat_completions;
pub(crate) mod error;
pub mod images;
/// Token log-probability response values.
pub mod logprobs;
mod protocol;
pub(crate) mod video_jobs;
pub mod videos;

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request, State};
use axum::routing::get;
use axum::{Json, Router};
use serde::de::DeserializeOwned;
use validator::Validate;

use crate::engine::Engine;

pub use error::{ApiError, Result};
pub use protocol::*;

/// The model listing route, `GET /v1/models`.
pub(crate) fn models_router(engine: Arc<Engine>) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .with_state(engine)
}

/// Returns the served model in OpenAI `list models` format.
///
/// A server serves exactly one model, so the list always has one entry, named
/// by `Engine::served_model_name`.
async fn list_models(State(engine): State<Arc<Engine>>) -> Json<ListModelsResponse> {
    // OpenAI clients expect `created` to be a real Unix timestamp; the listing
    // reports the time it was produced.
    let created = uniserve_core::now_unix_secs_u64() as i64;
    Json(ListModelsResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: engine.served_model_name().to_string(),
            object: "model".to_string(),
            created,
            owned_by: "uniserve".to_string(),
        }],
    })
}

/// A JSON extractor that validates the request body.
///
/// After deserialization the extractor calls `Validate::validate`, whose
/// checks `T`'s `#[validate(...)]` constraints declare: ranges and
/// cross-parameter rules. Rules that depend on the served model run when the
/// engine lowers the request.
///
/// Any rejection from axum's `Json` extractor, such as a missing JSON content
/// type, malformed JSON, a schema mismatch, or a body over the size limit,
/// becomes [`ApiError::JsonParseError`]. Validation errors become
/// [`ApiError::InvalidRequest`] with the details and no `param`. Both answer
/// `400 Bad Request`.
pub(crate) struct ValidatedJson<T>(pub T);

impl<S, T> FromRequest<S> for ValidatedJson<T>
where
    T: DeserializeOwned + Validate + Send,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(request: Request, state: &S) -> Result<Self> {
        let Json(data) = Json::<T>::from_request(request, state)
            .await
            .map_err(|rejection: JsonRejection| ApiError::from(rejection))?;

        data.validate()
            .map_err(|errors| ApiError::invalid_request(errors.to_string(), None))?;
        Ok(Self(data))
    }
}

/// Rejects a request whose model name is not the served model name.
///
/// `model` is the resolved model name the caller extracted from its own
/// request shape. The comparison is exact, and a mismatch is reported as
/// [`ApiError::ModelNotFound`] (HTTP 404).
pub(crate) fn check_model_served(model: &str, served_model_name: &str) -> Result<()> {
    if model != served_model_name {
        return Err(ApiError::model_not_found(model));
    }
    Ok(())
}

/// Converts OpenAI-style `logit_bias` with string token-ID keys into the
/// internal token-ID map.
///
/// JSON object keys are always strings, so each key is parsed as a decimal
/// `u32` token ID; any key that does not parse fails the whole conversion
/// with an invalid-request error on `logit_bias`. Bias values pass through
/// unchanged, and an absent map stays `None`. Token IDs are not checked
/// against the vocabulary here.
pub(crate) fn convert_logit_bias(
    logit_bias: Option<HashMap<String, f32>>,
) -> Result<Option<HashMap<u32, f32>>> {
    logit_bias
        .map(|bias| {
            bias.into_iter()
                .map(|(key, value)| {
                    key.parse().map(|token| (token, value)).map_err(|_| {
                        ApiError::invalid_request(
                            format!(
                                "Invalid key in 'logit_bias': '{key}' is not a valid token ID. \
                                 Token IDs must be non-negative integers."
                            ),
                            Some("logit_bias"),
                        )
                    })
                })
                .collect()
        })
        .transpose()
}
