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
use crate::serving::RequestOutput;

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
/// Consumes `stream` until its `Finished` event. Each `ImageDone` event
/// contributes one entry to the response `data`, in arrival order; `created`
/// is the response timestamp in Unix seconds. Lifecycle, text, reasoning,
/// output-block, tool-call, image-progress, and usage events are ignored.
///
/// # Errors
///
/// Returns a stream error mapped by `From<ServeError>`, the category
/// chosen by [`ApiError::rejected`] for an admission rejection, and a server
/// error when the request fails, is cancelled or aborted, finishes with an
/// error status, emits a video event, reports an image without its
/// dimensions, byte count, or PNG payload, or ends without a `Finished` event
/// or without any image.
pub async fn collect_image_generation(
    stream: impl Stream<Item = crate::serving::Result<RequestOutput>> + Send,
    created: u64,
) -> Result<ImageGenerationResponse, ApiError> {
    futures::pin_mut!(stream);

    let mut data = Vec::new();
    let mut finished = false;
    while let Some(event) = stream.next().await {
        match event? {
            // `ImageDone` metadata is optional in the serving protocol; an
            // image response cannot be built without all four fields.
            RequestOutput::ImageDone {
                height,
                width,
                bytes,
                pixels_png_b64,
                ..
            } => data.push(GeneratedImageData {
                b64_json: required_image_field(pixels_png_b64, "PNG artifact")?,
                height: required_image_field(height, "height")?,
                width: required_image_field(width, "width")?,
                bytes: required_image_field(bytes, "byte count")?,
            }),
            RequestOutput::Finished { reason, .. } => {
                if matches!(reason, crate::serving::FinishStatus::Error) {
                    return Err(ApiError::server_error(
                        "image generation terminated with an execution error".to_string(),
                    ));
                }
                finished = true;
                break;
            }
            RequestOutput::Rejected { kind, message, .. } => {
                return Err(ApiError::rejected(kind, message));
            }
            RequestOutput::Failed { message, .. } => return Err(ApiError::server_error(message)),
            RequestOutput::Cancelled { .. } => {
                return Err(ApiError::server_error(
                    "image generation request was cancelled".to_string(),
                ));
            }
            RequestOutput::Aborted { .. } => {
                return Err(ApiError::server_error(
                    "image generation request was aborted".to_string(),
                ));
            }
            // Artifact and media-progress events belong to video generation.
            RequestOutput::Artifact(_) | RequestOutput::MediaProgress { .. } => {
                return Err(ApiError::server_error(
                    "image request received a video event".to_owned(),
                ));
            }
            RequestOutput::Accepted { .. }
            | RequestOutput::Scheduled { .. }
            | RequestOutput::TextDelta { .. }
            | RequestOutput::ReasoningDelta { .. }
            | RequestOutput::OutputBlockStart { .. }
            | RequestOutput::OutputBlockEnd { .. }
            | RequestOutput::ToolCallStart { .. }
            | RequestOutput::ToolCallArgumentsDelta { .. }
            | RequestOutput::ToolCallEnd { .. }
            | RequestOutput::ImageBegin { .. }
            | RequestOutput::ImageStep { .. }
            | RequestOutput::ImageCommit { .. }
            | RequestOutput::Usage { .. } => {}
        }
    }

    if !finished || data.is_empty() {
        return Err(ApiError::server_error(
            "image generation completed without a committed image".to_string(),
        ));
    }
    Ok(ImageGenerationResponse { created, data })
}

/// Returns a required `ImageDone` field, or a server error naming the missing
/// metadata.
fn required_image_field<T>(value: Option<T>, field: &str) -> Result<T, ApiError> {
    value.ok_or_else(|| {
        ApiError::server_error(format!(
            "image generation completed without required {field} metadata"
        ))
    })
}
