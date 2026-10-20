//! `POST /v1/images/generations`: OpenAI-compatible image generation.
//!
//! The handler submits an `ImageGenerationRequest` through the engine and
//! hands the resulting `RequestOutput` stream to [`collect_image_generation`],
//! which buffers completed images until the request finishes and returns one
//! non-streaming response.

use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Extension, Json, Router};
use futures::{Stream, StreamExt as _};

use crate::engine::Engine;
use crate::http::RequestId;
use crate::openai::{
    ApiError, GeneratedImageData, ImageGenerationRequest, ImageGenerationResponse, ValidatedJson,
};
use crate::serving::{FinishReason, RequestOutput};

/// The image generation route.
pub(crate) fn router(engine: Arc<Engine>) -> Router {
    Router::new()
        .route("/v1/images/generations", post(images_generations))
        .with_state(engine)
}

/// Validates, submits, and collects one image-generation request.
async fn images_generations(
    State(engine): State<Arc<Engine>>,
    Extension(RequestId(base_id)): Extension<RequestId>,
    ValidatedJson(body): ValidatedJson<ImageGenerationRequest>,
) -> Response {
    let request_id = format!("img-{base_id}");
    let stream = match engine.generate_image(request_id.into(), body).await {
        Ok(stream) => stream,
        Err(error) => return error.into_response(),
    };
    match collect_image_generation(stream, uniserve_core::now_unix_secs_u64()).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

/// Collects image events into one OpenAI-compatible response.
///
/// Consumes `stream` until its terminal event. Each `ImageDone` event
/// contributes one entry to the response `data`, in arrival order; `created`
/// is the response timestamp in Unix seconds. Lifecycle, text, reasoning,
/// output-block, tool-call, token, and image-step events are ignored.
///
/// # Errors
///
/// Returns a stream error mapped by `From<ServeError>`, the category chosen
/// by [`ApiError::rejected`] for an admission rejection, and a server error
/// when the request fails or is aborted, emits a video event, or ends
/// without a `Finished` event or without any image.
pub async fn collect_image_generation(
    stream: impl Stream<Item = crate::serving::Result<RequestOutput>> + Send,
    created: u64,
) -> Result<ImageGenerationResponse, ApiError> {
    futures::pin_mut!(stream);

    let mut data = Vec::new();
    let mut finished = false;
    while let Some(event) = stream.next().await {
        match event? {
            RequestOutput::ImageDone(image) => data.push(GeneratedImageData {
                b64_json: image.png_base64,
                height: image.height,
                width: image.width,
                bytes: image.bytes,
            }),
            RequestOutput::Finished {
                reason: FinishReason::Abort,
                ..
            } => {
                return Err(ApiError::server_error(
                    "image generation request was aborted".to_string(),
                ));
            }
            RequestOutput::Finished { .. } => {
                finished = true;
                break;
            }
            RequestOutput::Rejected { kind, message } => {
                return Err(ApiError::rejected(kind, message));
            }
            RequestOutput::Failed { message } => return Err(ApiError::server_error(message)),
            // Artifact and media-progress events belong to video generation.
            RequestOutput::Artifact(_) | RequestOutput::MediaProgress { .. } => {
                return Err(ApiError::server_error(
                    "image request received a video event".to_owned(),
                ));
            }
            RequestOutput::Accepted(_)
            | RequestOutput::Scheduled { .. }
            | RequestOutput::TextDelta { .. }
            | RequestOutput::ReasoningDelta { .. }
            | RequestOutput::TokenDelta { .. }
            | RequestOutput::OutputBlockStart { .. }
            | RequestOutput::OutputBlockEnd { .. }
            | RequestOutput::ToolCallStart { .. }
            | RequestOutput::ToolCallArgumentsDelta { .. }
            | RequestOutput::ToolCallEnd { .. }
            | RequestOutput::ImageStep { .. } => {}
        }
    }

    if !finished || data.is_empty() {
        return Err(ApiError::server_error(
            "image generation completed without a committed image".to_string(),
        ));
    }
    Ok(ImageGenerationResponse { created, data })
}
