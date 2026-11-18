//! The OpenAI-compatible video routes.
//!
//! Two submission routes share the request extractor `VideoBody`:
//!
//! - `POST /v1/videos/sync` (`videos_sync`) waits for generation and responds
//!   with the video bytes. It is a generation route, admitted under the
//!   server's in-flight limit like a chat completion.
//! - `POST /v1/videos` (`videos_create`) returns a job record at once and
//!   drives generation in a task of `VideoJobs`. `GET /v1/videos`,
//!   `GET`/`DELETE /v1/videos/{id}`, and `GET /v1/videos/{id}/content` list,
//!   read, delete (cancelling a running job), and download those jobs.
//!
//! `GET /v1/capabilities` (`capabilities`) reports the model's video limits
//! and the job-store settings.
//!
//! Completed video bytes arrive as `ArtifactEvent::media`, the read-only
//! mapping of the sealed memory file the engine's media host wrote
//! (`proto::MappedMedia`). Responses and retained jobs hold that mapping by
//! `Arc`, and `media_body`
//! streams from it without gathering the video into one buffer.

use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, FromRequest, Multipart, Path, Request, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use futures::StreamExt as _;
use serde_json::{Value, json};

use crate::api::openai::video_jobs::{JobStatus, StageClock, VideoFailure, VideoJob, VideoJobs};
use crate::api::openai::{ApiError, check_model_served};
use crate::api::openai::{RequestId, Trace};
use crate::api::output::video::ArtifactCollector;
use crate::api::protocol::{VideoGenerationRequest, from_form};
use crate::http::served::Served;
use String;

/// The state of the asynchronous video job routes.
#[derive(Clone)]
struct JobState {
    served: Arc<Served>,
    jobs: Arc<VideoJobs>,
}

/// The synchronous video route, `POST /v1/videos/sync`.
///
/// Its body may carry condition media inline, so its body limit is
/// `body_limit`, the configured media total.
pub(crate) fn generation_router(served: Arc<Served>, body_limit: usize) -> Router {
    Router::new()
        .route(
            "/v1/videos/sync",
            post(videos_sync).layer(DefaultBodyLimit::max(body_limit)),
        )
        .with_state(served)
}

/// The asynchronous video job routes and `GET /v1/capabilities`.
///
/// Job creation takes condition media inline like the synchronous route, so
/// `POST /v1/videos` has the same `body_limit`.
pub(crate) fn job_router(served: Arc<Served>, jobs: Arc<VideoJobs>, body_limit: usize) -> Router {
    Router::new()
        .route(
            "/v1/videos",
            get(videos_list).post(videos_create).layer(
                // Applies to the listing too, which reads no body.
                DefaultBodyLimit::max(body_limit),
            ),
        )
        .route("/v1/videos/{id}", get(videos_get).delete(videos_delete))
        .route("/v1/videos/{id}/content", get(videos_content))
        .route("/v1/capabilities", get(capabilities))
        .with_state(JobState { served, jobs })
}

/// Validates and streams one completed video artifact synchronously.
///
/// Responds with the artifact bytes, its content type, and a `Server-Timing`
/// `generation` entry in milliseconds from handler entry (after body
/// extraction) until the runtime reported completion. Submission errors,
/// stream errors, and engine rejections map through `ApiError` (`400` or
/// `503` for a rejection); any other stream that ends without an artifact is
/// a `500`.
async fn videos_sync(
    State(served): State<Arc<Served>>,
    Extension(RequestId(base_id)): Extension<RequestId>,
    Extension(Trace(trace)): Extension<Trace>,
    VideoBody(body): VideoBody,
) -> Response {
    let started_at = Instant::now();
    let request_id = format!("vid-{base_id}");

    if let Err(error) = check_model_served(&body.model, &served.leader.served_model_name()) {
        return error.into_response();
    }
    let (processed, mut stream) = match served.leader.generate_video(request_id, body, trace).await
    {
        Ok(generation) => generation,
        Err(error) => return ApiError::from(error).into_response(),
    };

    // The synchronous endpoint consumes lifecycle events until the runtime
    // confirms completion and supplies exactly one artifact descriptor.
    let mut collector = ArtifactCollector::default();
    let mut clock = StageClock::default();
    let artifact = loop {
        let event = match stream.next().await.transpose() {
            Ok(event) => event,
            Err(error) => return ApiError::from(error).into_response(),
        };
        if let Some(event) = &event {
            clock.observe(event);
        }
        if let ControlFlow::Break(result) = collector.push(event) {
            match result {
                Ok(artifact) => break artifact,
                Err(error) => return ApiError::from(error).into_response(),
            }
        }
    };

    let generation_ms = started_at.elapsed().as_secs_f64() * 1_000.0;
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, &artifact.content_type)
        .header(header::CONTENT_LENGTH, artifact.media.bytes().len());
    for (name, value) in generation_headers(&processed, generation_ms, clock.finish()) {
        response = response.header(name, value);
    }
    response
        .body(media_body(Arc::new(ArtifactBytes(artifact.media))))
        .unwrap_or_else(|error| {
            ApiError::server_error(format!("failed to construct media response: {error}"))
                .into_response()
        })
}

/// The headers a synchronous video response states its generation with, beside the video's
/// bytes: the one declaration of them.
///
/// - `server-timing`: `generation;dur=MS`, the request's time in the server, in milliseconds.
/// - `x-video-size` (`WxH`), `x-video-num-frames` and `x-video-fps`: the shape the model
///   generated, which a client holds the media to.
/// - `x-inference-time-s`: seconds from scheduling to completion, and `x-stage-durations`: the
///   seconds of each phase as one JSON object ([`GenerationTiming`]); both absent when the
///   request was never scheduled.
fn generation_headers(
    processed: &crate::api::input::video::ProcessedVideo,
    generation_ms: f64,
    timing: Option<crate::api::openai::video_jobs::GenerationTiming>,
) -> Vec<(&'static str, String)> {
    let mut headers = vec![
        (
            "server-timing",
            format!("generation;dur={generation_ms:.1}"),
        ),
        (
            "x-video-size",
            format!("{}x{}", processed.size.width, processed.size.height),
        ),
        ("x-video-num-frames", processed.num_frames.to_string()),
        ("x-video-fps", processed.fps.to_string()),
    ];
    if let Some(timing) = timing {
        headers.push((
            "x-inference-time-s",
            format!("{:.3}", timing.inference_time_s),
        ));
        headers.push((
            "x-stage-durations",
            json!(timing.stage_durations).to_string(),
        ));
    }
    headers
}

/// Video request extractor accepting `application/json` or `multipart/form-data`.
///
/// Both content types deserialize into the same `VideoGenerationRequest`, whose
/// `deny_unknown_fields` rejects unknown fields at every level. A multipart body carries every
/// field as a text part of the same name, which the field's own type reads
/// ([`from_form`]): text for a string, a number for a number, JSON for the reference list and
/// the model's options. File parts, repeated fields and fields outside the schema are rejected. A
/// JSON body that is not the schema is a `json_parse_error`, a refused multipart body an
/// `invalid_request_error` naming its field (both `400`), and an unsupported content type `415
/// Unsupported Media Type`. Semantic checks (served model, the model's own rules) happen in the
/// handlers and in `VideoService::process`.
pub struct VideoBody(pub VideoGenerationRequest);

/// Reads a multipart body's text parts, as `(field, text)` in body order.
async fn multipart_parts(mut multipart: Multipart) -> Result<Vec<(String, String)>, ApiError> {
    let invalid = |error: axum::extract::multipart::MultipartError| {
        ApiError::invalid_request(error.to_string(), None)
    };
    let mut parts = Vec::new();
    while let Some(field) = multipart.next_field().await.map_err(invalid)? {
        let name = field.name().unwrap_or("").to_owned();
        if field.file_name().is_some() {
            return Err(ApiError::invalid_request(
                format!("video field {name:?} must be text, not a file"),
                Some(name),
            ));
        }
        let text = field.text().await.map_err(invalid)?;
        parts.push((name, text));
    }
    Ok(parts)
}

impl<S: Send + Sync> FromRequest<S> for VideoBody {
    type Rejection = ApiError;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        // The request reached its route: what follows reads and parses its body.
        let trace = request.extensions().get::<Trace>().copied();
        if let Some(Trace(trace)) = trace {
            service::span::end(service::span::Stage::Receive, trace);
            service::span::begin(service::span::Stage::Parse, trace);
        }
        let parsed = Self::parse(request, state).await;
        if let Some(Trace(trace)) = trace {
            service::span::end(service::span::Stage::Parse, trace);
        }
        parsed
    }
}

impl VideoBody {
    /// The request's body, as JSON or as a multipart form.
    async fn parse<S: Send + Sync>(request: Request, state: &S) -> Result<Self, ApiError> {
        let content_type = request
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("");
        if content_type.starts_with("multipart/form-data") {
            let multipart = Multipart::from_request(request, state)
                .await
                .map_err(|error| ApiError::invalid_request(error.body_text(), None))?;
            from_form(multipart_parts(multipart).await?)
                .map(Self)
                .map_err(|error| {
                    let field = error.path().to_string();
                    ApiError::invalid_request(error.into_inner().to_string(), Some(field))
                })
        } else if content_type
            .split(';')
            .next()
            .is_some_and(|value| value.trim() == "application/json")
        {
            let Json(value) = Json::<VideoGenerationRequest>::from_request(request, state).await?;
            Ok(Self(value))
        } else {
            Err(ApiError::unsupported_media_type(
                "use application/json or multipart/form-data",
            ))
        }
    }
}

/// Creates an asynchronous video job and returns its `queued` record.
///
/// The prepared request fills the record's `seconds`, `actual_seconds`,
/// `num_frames`, `size`, and `total_steps`. Preparation and submission errors
/// map through `ApiError`. Once the record is inserted, a `VideoJobs` task
/// consumes the request's stream and publishes progress and the final result,
/// so the job continues after the HTTP response is sent or dropped.
async fn videos_create(
    State(JobState { served, jobs }): State<JobState>,
    Extension(RequestId(base_id)): Extension<RequestId>,
    Extension(Trace(trace)): Extension<Trace>,
    VideoBody(body): VideoBody,
) -> Response {
    let request_id = format!("vid-{base_id}");
    if let Err(error) = check_model_served(&body.model, &served.leader.served_model_name()) {
        return error.into_response();
    }
    let (prepared, mut stream) = match served.leader.generate_video(request_id, body, trace).await {
        Ok(generation) => generation,
        Err(error) => return ApiError::from(error).into_response(),
    };

    // Public IDs are server-generated, so caller request-ID headers cannot
    // collide with retained jobs.
    let id = format!("video_{}", uuid::Uuid::new_v4().simple());
    let record = VideoJob {
        id: id.clone(),
        object: "video".to_owned(),
        model: served.leader.served_model_name().to_string(),
        created_at: crate::api::since_epoch().as_secs(),
        completed_at: None,
        expires_at: None,
        seconds: format!("{}", prepared.seconds()),
        num_frames: prepared.num_frames,
        fps: prepared.fps.get(),
        size: format!("{}x{}", prepared.size.width, prepared.size.height),
        status: JobStatus::Queued,
        phase: None,
        completed_steps: 0,
        // The deployment's denoiser fixes every request's step count.
        total_steps: served
            .leader
            .processor()
            .video_service()
            .map_or(0, |video| video.denoise_steps()),
        inference_time_s: None,
        stage_durations: None,
        error: None,
    };

    let store = Arc::clone(&jobs);
    jobs.start(record.clone(), move |cancellation| async move {
        let mut clock = StageClock::default();
        let mut collector = ArtifactCollector::default();
        let result = loop {
            // Deletion or server shutdown ends the task; returning drops the
            // stream, which cancels the request in its scheduler.
            let event = tokio::select! {
                event = stream.next() => event,
                () = cancellation.cancelled() => return,
            };
            let event = match event.transpose() {
                Ok(event) => event,
                Err(error) => {
                    break Err(VideoFailure {
                        code: "generation_failed".to_owned(),
                        message: error.to_string(),
                    });
                }
            };
            if let Some((phase, steps)) = event.as_ref().and_then(|event| clock.observe(event)) {
                store.progress(&id, phase, steps);
            }
            if let ControlFlow::Break(result) = collector.push(event) {
                break result.map(|artifact| artifact.media).map_err(Into::into);
            }
        };

        // `finish` fails a successful result whose bytes do not fit the
        // remaining retained-byte budget, and discards the result if the
        // record was deleted.
        store.finish(&id, result, clock.finish());
    });

    Json(record).into_response()
}

/// The error of an unknown job id. Expired jobs are removed from
/// `VideoJobs`, so they answer the same way.
fn missing_video() -> ApiError {
    ApiError::not_found("video_not_found", "video does not exist or has expired")
}

/// Lists every retained job, newest first. The store's retention bounds the
/// records, so the list is returned whole and `has_more` is always `false`.
async fn videos_list(State(state): State<JobState>) -> Json<Value> {
    Json(json!({"object": "list", "data": state.jobs.list(), "has_more": false}))
}

async fn videos_get(State(state): State<JobState>, Path(id): Path<String>) -> Response {
    match state.jobs.get(&id) {
        Some(record) => Json(record).into_response(),
        None => missing_video().into_response(),
    }
}

/// Removes a job record and cancels its generation if it is still running.
///
/// The response does not wait for the cancellation to drain; the generation
/// task ends once its stream has.
async fn videos_delete(State(state): State<JobState>, Path(id): Path<String>) -> Response {
    if state.jobs.delete(&id) {
        Json(json!({"id": id, "object": "video.deleted", "deleted": true})).into_response()
    } else {
        missing_video().into_response()
    }
}

/// Streams a completed job's video as `video/mp4`.
///
/// Answers `409 Conflict` for a job without content (queued, running, or
/// failed) and `404 Not Found` for an unknown id. The response body holds the
/// `RetainedMedia`, so the mapping and its share of the retained-byte budget
/// stay reserved until the download ends, even if the job is deleted or
/// expires meanwhile.
async fn videos_content(State(state): State<JobState>, Path(id): Path<String>) -> Response {
    let Some(media) = state.jobs.content(&id) else {
        return if state.jobs.get(&id).is_some() {
            ApiError::conflict("video has no completed content").into_response()
        } else {
            missing_video().into_response()
        };
    };
    let length = media.len();
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "video/mp4")
        .header(header::CONTENT_LENGTH, length.to_string())
        .body(Body::empty())
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response());
    // The body registers before it writes a chunk, so a hot restart pauses it
    // between two chunks and the successor goes on streaming the same
    // artifact from the byte this process reached.
    let body =
        crate::api::openai::run::VideoContentRun::new(id.clone(), "video/mp4".to_owned(), media);
    let opening = crate::api::body::Opening::Streamed { keep_alive: false };
    match state
        .served
        .attach(&mut response, Box::new(body), opening, id)
    {
        Ok(()) => response,
        Err(error) => error.into_response(),
    }
}

/// A generated artifact's bytes as a response body reads them.
struct ArtifactBytes(Arc<dyn proto::request::ArtifactMedia>);

impl AsRef<[u8]> for ArtifactBytes {
    fn as_ref(&self) -> &[u8] {
        self.0.bytes()
    }
}

/// Streams `media` as a body of copied chunks of at most 64 KiB.
///
/// The stream state owns the `Arc`, so the backing mapping, and for
/// `RetainedMedia` its retained-byte permit, lives until the body completes or
/// is dropped.
fn media_body<M: AsRef<[u8]> + Send + Sync + 'static>(media: Arc<M>) -> Body {
    let chunks = futures::stream::try_unfold((media, 0_usize), |(media, offset)| async move {
        let bytes = media.as_ref().as_ref();
        if offset == bytes.len() {
            Ok::<_, std::convert::Infallible>(None)
        } else {
            let count = (bytes.len() - offset).min(64 * 1024);
            let chunk = axum::body::Bytes::copy_from_slice(&bytes[offset..offset + count]);
            Ok(Some((chunk, (media, offset + count))))
        }
    });
    Body::from_stream(chunks)
}

/// Reports the served model, its video limits
/// (`Processor::video_capabilities`), and the settings of the job store.
async fn capabilities(State(state): State<JobState>) -> Json<Value> {
    let settings = state.jobs.settings();
    Json(json!({
        "model": state.served.leader.served_model_name().to_string(),
        "video": state.served.leader.processor().video_capabilities(),
        "video_jobs": {
            "max_retained_bytes": settings.max_bytes,
            "retention_seconds": settings.retention.as_secs(),
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn video_request() -> serde_json::Value {
        serde_json::json!({
            "model": "model",
            "prompt": "A river",
            "size": "64x64",
            "seconds": 5,
        })
    }

    /// A `multipart/form-data` request of text parts.
    fn multipart(parts: &[(&str, &str)]) -> axum::extract::Request {
        let mut body = String::new();
        for (name, value) in parts {
            body.push_str(&format!(
                "--clip\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            ));
        }
        body.push_str("--clip--\r\n");
        axum::extract::Request::builder()
            .header(header::CONTENT_TYPE, "multipart/form-data; boundary=clip")
            .body(Body::from(body))
            .unwrap()
    }

    /// A JSON request carrying `body`.
    fn json_request(body: &serde_json::Value) -> axum::extract::Request {
        axum::extract::Request::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn json_and_multipart_normalize_to_the_same_request() {
        let references =
            r#"[{"type": "image", "url": "data:image/png;base64,AA==", "frame_index": 0}]"#;
        let json = json_request(&serde_json::json!({
            "model": "FastH3",
            "prompt": "A river",
            "size": "1344x768",
            "seconds": 5.5,
            "input_references": serde_json::from_str::<serde_json::Value>(references).unwrap(),
            "seed": 7,
        }));
        let multipart = multipart(&[
            ("model", "FastH3"),
            ("prompt", "A river"),
            ("size", "1344x768"),
            ("seconds", "5.5"),
            ("input_references", references),
            ("seed", "7"),
        ]);

        let left = VideoBody::from_request(json, &()).await.unwrap().0;
        let right = VideoBody::from_request(multipart, &()).await.unwrap().0;

        assert_eq!(left, right);
        assert_eq!(left.input_references[0].frame_index, Some(0));
        assert_eq!(left.seed, Some(7));
        assert_eq!(
            left.spec().unwrap().length,
            Some(processing::Length::Seconds(5.5))
        );
    }

    /// Fields outside the schema, a nested one included, and file parts are refused in both
    /// formats, and a body without a prompt names the missing `prompt`.
    #[tokio::test]
    async fn fields_outside_the_schema_are_rejected_in_both_formats() {
        let mut extra = video_request();
        extra["task"] = serde_json::json!("t2va");
        let mut nested = video_request();
        nested["input_references"] =
            serde_json::json!([{"type": "image", "url": "data:,", "role": "keyframe"}]);
        let file = axum::extract::Request::builder()
            .header(header::CONTENT_TYPE, "multipart/form-data; boundary=clip")
            .body(Body::from(
                "--clip\r\nContent-Disposition: form-data; name=\"input_references\"; filename=\"image.png\"\r\n\r\nimage\r\n--clip--\r\n",
            ))
            .unwrap();
        let requests = [
            json_request(&extra),
            json_request(&nested),
            multipart(&[("model", "FastH3"), ("input_reference", "image.png")]),
            file,
        ];
        for request in requests {
            match VideoBody::from_request(request, &()).await {
                Ok(VideoBody(body)) => panic!("{body:?} was accepted"),
                Err(error) => assert_eq!(error.status_code(), StatusCode::BAD_REQUEST),
            }
        }

        for request in [
            json_request(&serde_json::json!({"model": "FastH3"})),
            multipart(&[("model", "FastH3")]),
        ] {
            let error = match VideoBody::from_request(request, &()).await {
                Ok(VideoBody(body)) => panic!("{body:?} was accepted"),
                Err(error) => error,
            };
            assert_eq!(error.status_code(), StatusCode::BAD_REQUEST);
            let message = error.to_error_response().error.message;
            assert!(message.contains("prompt"), "{message}");
        }
    }
}
