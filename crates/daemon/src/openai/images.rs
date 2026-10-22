//! `POST /v1/images/generations`: OpenAI-compatible image generation.
//!
//! The handler lowers an `ImageGenerationRequest` ([`lower_image_request`]),
//! submits it through the engine, and hands the resulting `RequestOutput`
//! stream to [`collect_image_generation`],
//! which buffers completed images until the request finishes and returns one
//! non-streaming response.

use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Extension, Json, Router};
use futures::{Stream, StreamExt as _};

use crate::leader::Leader;
use crate::openai::RequestId;
use proto::request::GenerationConstraint;

use crate::input::{GenerateRequest, ImageSamplingParams, Prompt, SamplingParams};
use crate::openai::{
    ApiError, GeneratedImageData, ImageGenerationRequest, ImageGenerationResponse, ValidatedJson,
    check_model_served,
};
use crate::output::{FinishReason, RequestOutput};

/// The image generation route.
pub(crate) fn router(engine: Arc<Leader>) -> Router {
    Router::new()
        .route("/v1/images/generations", post(images_generations))
        .with_state(engine)
}

/// Validates, submits, and collects one image-generation request.
async fn images_generations(
    State(engine): State<Arc<Leader>>,
    Extension(RequestId(base_id)): Extension<RequestId>,
    ValidatedJson(body): ValidatedJson<ImageGenerationRequest>,
) -> Response {
    let request =
        match lower_image_request(format!("img-{base_id}"), engine.served_model_name(), body) {
            Ok(request) => request,
            Err(error) => return error.into_response(),
        };
    let stream = match engine.generate(request).await {
        Ok(stream) => stream,
        Err(error) => return ApiError::from(error).into_response(),
    };
    match collect_image_generation(stream, crate::openai::unix_time()).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

/// Validates and lowers one image-generation request into the serving
/// layer's generate request: an image-only generation of exactly one image
/// with the default shared cache policy and default text sampling.
///
/// # Errors
///
/// Rejects an empty or whitespace-only prompt, a model name other than the
/// served one, and a malformed `size`.
pub fn lower_image_request(
    request_id: String,
    served_model_name: &str,
    request: ImageGenerationRequest,
) -> Result<GenerateRequest, ApiError> {
    if request.prompt.trim().is_empty() {
        return Err(ApiError::invalid_request(
            "prompt must not be empty".to_string(),
            Some("prompt"),
        ));
    }
    if let Some(model) = request.model.as_deref() {
        check_model_served(model, served_model_name)?;
    }
    // An absent size leaves both dimensions to the profile's resolution
    // policy.
    let (width, height) = request
        .size
        .as_deref()
        .map(parse_size)
        .transpose()?
        .map_or((None, None), |(width, height)| (Some(width), Some(height)));
    Ok(GenerateRequest {
        request_id,
        prompt: Prompt::Text {
            prompt: request.prompt,
            negative_prompt: request.negative_prompt,
        },
        constraint: GenerationConstraint::GenOnly,
        sampling_params: SamplingParams::default(),
        image_params: Some(ImageSamplingParams {
            width,
            height,
            steps: request.steps,
            cfg_text_scale: request.guidance_scale,
            cfg_img_scale: request.image_guidance_scale,
            cfg_interval: request.cfg_interval,
            cfg_renorm_type: request.cfg_norm,
            timestep_shift: request.timestep_shift,
            seed: request.seed,
            max_images: Some(1),
            ..ImageSamplingParams::default()
        }),
        priority: 0,
        cache_salt: None,
        adapter: None,
    })
}

/// Parses a positive `WIDTHxHEIGHT` image-size string.
fn parse_size(size: &str) -> Result<(u32, u32), ApiError> {
    let Some((width, height)) = size.split_once('x') else {
        return Err(ApiError::invalid_request(
            "size must use WIDTHxHEIGHT syntax".to_string(),
            Some("size"),
        ));
    };
    if height.contains('x') {
        return Err(ApiError::invalid_request(
            "size must use WIDTHxHEIGHT syntax".to_string(),
            Some("size"),
        ));
    }
    let width = width.parse::<u32>().ok().filter(|value| *value > 0);
    let height = height.parse::<u32>().ok().filter(|value| *value > 0);
    width.zip(height).ok_or_else(|| {
        ApiError::invalid_request(
            "size dimensions must be positive integers".to_string(),
            Some("size"),
        )
    })
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
    stream: impl Stream<Item = Result<RequestOutput, crate::output::OutputError>> + Send,
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
