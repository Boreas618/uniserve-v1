//! The OpenAI-compatible API's route handlers, its errors, and the response
//! bodies the handlers write.
//!
//! Each endpoint module builds its own router over the served model, which
//! the daemon supplies as a [`Server`] (`chat_completions`, `videos`); the
//! daemon's HTTP layer registers the routers of the tasks the engine serves.
//! A request is deserialized into the strict schemas of `crate::protocol`
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

use axum::response::Response;
use futures::future::BoxFuture;

use crate::body::{Opening, Session};
use crate::events::Events;
use crate::input::video::ProcessedVideo;
use crate::input::{AdapterSelection, Processor, ServeError};
use crate::output::{OutputProcessor, RequestOutputStream};
use crate::protocol::{
    ChatCompletionRequest, ListModelsResponse, ModelObject, VideoGenerationRequest,
};

pub use error::{ApiError, Result};

/// The served model as the API's routes reach it: the daemon's request
/// facade over its input processing, its engines and its HTTP connections.
pub trait Server: Send + Sync + 'static {
    /// The single model name the API serves.
    fn served_model_name(&self) -> &str;

    /// The names of the LoRA adapters the workers hold.
    fn adapter_names(&self) -> Vec<String>;

    /// The adapter a request that names `name` as its model computes with.
    fn adapter(&self, name: &str) -> Option<AdapterSelection>;

    /// The deployment's input processing.
    fn processor(&self) -> &Processor;

    /// Lowers the chat request `request` through the deployment's
    /// `Processor`, fetching its images, and submits it; `adapter` is the
    /// LoRA adapter its `model` names, when it names one.
    ///
    /// Returns the request's engine events and the output processor that
    /// turns them into the response named `request_id`.
    ///
    /// # Errors
    ///
    /// Fails when an input cannot be fetched, input processing refuses the
    /// request, or no engine accepts it.
    fn generate(
        &self,
        request_id: String,
        request: ChatCompletionRequest,
        adapter: Option<AdapterSelection>,
    ) -> BoxFuture<'_, Result<(Box<dyn Events>, OutputProcessor)>>;

    /// Fetches, probes and processes a video request, submits it, and streams its
    /// output; also returns what the request generates.
    ///
    /// # Errors
    ///
    /// Fails when the model serves no video, the request is refused, or no
    /// engine accepts it.
    fn generate_video(
        &self,
        request_id: String,
        request: VideoGenerationRequest,
    ) -> BoxFuture<'_, std::result::Result<(ProcessedVideo, RequestOutputStream), ServeError>>;

    /// Makes `session` the body of `response`, opened as `opening`: the
    /// daemon's connection writes its frames and a hot restart can hand it to
    /// a successor. `request_id` names the response there.
    ///
    /// # Errors
    ///
    /// Fails when the body cannot be registered for a handoff.
    fn attach(
        &self,
        response: &mut Response,
        session: Box<dyn Session>,
        opening: Opening,
        request_id: String,
    ) -> std::result::Result<(), ApiError>;
}

/// The request's base ID, which the HTTP layer resolves and inserts into the
/// request extensions before a route handler runs.
///
/// Route handlers prefix it per API (`chatcmpl-`, `img-`, `vid-`) and name
/// the request by the result in responses and logs. As in vLLM, the ID need
/// not be unique: the engine identifies each request by its own identifier.
#[derive(Clone, Debug)]
pub struct RequestId(pub String);

/// The model listing route, `GET /v1/models`.
pub fn models_router(engine: Arc<dyn Server>) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .with_state(engine)
}

/// The current wall-clock time in whole seconds since the Unix epoch, as the
/// `created` field of OpenAI objects states it; times before the epoch clamp
/// to zero.
pub fn unix_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Returns the served model, and every LoRA adapter its workers hold, in
/// OpenAI `list models` format.
///
/// A server serves exactly one model, named by [`Server::served_model_name`],
/// which a request names for the base model alone and an adapter's name
/// for the model with that adapter, as vLLM lists them.
async fn list_models(State(engine): State<Arc<dyn Server>>) -> Json<ListModelsResponse> {
    // OpenAI clients expect `created` to be a real Unix timestamp; the listing
    // reports the time it was produced.
    let created = unix_time() as i64;
    let model = |id: String| ModelObject {
        id,
        object: "model".to_string(),
        created,
        owned_by: "uniserve".to_string(),
    };
    Json(ListModelsResponse {
        object: "list".to_string(),
        data: std::iter::once(engine.served_model_name().to_string())
            .chain(engine.adapter_names())
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
