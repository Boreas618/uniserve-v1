//! The OpenAI-compatible API's route handlers, its errors, and the response
//! bodies the handlers write.
//!
//! Each endpoint module builds its own router over the served model and the
//! daemon's connections (`crate::http::served::Served`: the leader, and the
//! registry a response body is attached through); the HTTP layer registers
//! the routers of the tasks the engine serves (`chat_completions`,
//! `videos`).
//! A request is deserialized into the strict schemas of `crate::api::protocol`
//! (unknown fields are rejected) and validated by `ValidatedJson`; the server
//! lowers it through the deployment's `Processor` and submits it, and the
//! endpoint module writes the response from the outputs of the request's
//! `OutputProcessor` (`run`). A handler holds no model knowledge: what a
//! model's requests and outputs look like is input and output processing's.
//! Every failure answers in the OpenAI error envelope of [`ApiError`].

pub mod chat_completions;
pub mod error;
pub mod run;
pub mod video_jobs;
pub mod videos;

use std::sync::Arc;

use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request, State};
use axum::routing::get;
use axum::{Json, Router};
use serde::de::DeserializeOwned;
use validator::Validate;

use crate::api::protocol::{ListModelsResponse, ModelObject};
use crate::http::served::Served;

pub use error::{ApiError, Result};

/// The request's base ID, which the HTTP layer resolves and inserts into the
/// request extensions before a route handler runs.
///
/// Route handlers prefix it per API (`chatcmpl-`, `img-`, `vid-`) and name
/// the request by the result in responses and logs. As in vLLM, the ID need
/// not be unique: the engine identifies each request by its own identifier.
#[derive(Clone, Debug)]
pub struct RequestId(pub String);

/// The model listing route, `GET /v1/models`.
pub(crate) fn models_router(served: Arc<Served>) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .with_state(served)
}

/// Returns the served model, and every LoRA adapter its workers hold, in
/// OpenAI `list models` format.
///
/// A server serves exactly one model, named by `Leader::served_model_name`,
/// which a request names for the base model alone and an adapter's name
/// for the model with that adapter, as vLLM lists them.
async fn list_models(State(served): State<Arc<Served>>) -> Json<ListModelsResponse> {
    // OpenAI clients expect `created` to be a real Unix timestamp; the listing
    // reports the time it was produced.
    let created = crate::api::since_epoch().as_secs() as i64;
    let model = |id: String| ModelObject {
        id,
        object: "model".to_string(),
        created,
        owned_by: "uniserve".to_string(),
    };
    Json(ListModelsResponse {
        object: "list".to_string(),
        data: std::iter::once(served.leader.served_model_name().to_string())
            .chain(served.leader.adapter_names())
            .map(model)
            .collect(),
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
pub struct ValidatedJson<T>(pub T);

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
pub fn check_model_served(model: &str, served_model_name: &str) -> Result<()> {
    if model != served_model_name {
        return Err(ApiError::model_not_found(model));
    }
    Ok(())
}
