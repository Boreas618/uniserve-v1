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
//! shared-memory mapping (`SharedMedia`) the engine claims from the worker.
//! Responses and retained jobs hold that mapping by `Arc`, and `media_body`
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

use crate::engine::Engine;
use crate::http::RequestId;
use crate::openai::video_jobs::{StageClock, VideoFailure, VideoJob, VideoJobs};
use crate::openai::{ApiError, check_model_served};
use crate::serving::video::REQUEST_FIELDS;
use crate::serving::video::VideoGenerationRequest;
use crate::serving::video::artifact::ArtifactCollector;
use String;

/// The state of the asynchronous video job routes.
#[derive(Clone)]
struct JobState {
    engine: Arc<Engine>,
    jobs: Arc<VideoJobs>,
}

/// The synchronous video route, `POST /v1/videos/sync`.
///
/// Its body may carry condition media inline, so its body limit is
/// `body_limit`, the configured media total.
pub(crate) fn generation_router(engine: Arc<Engine>, body_limit: usize) -> Router {
    Router::new()
        .route(
            "/v1/videos/sync",
            post(videos_sync).layer(DefaultBodyLimit::max(body_limit)),
        )
        .with_state(engine)
}

/// The asynchronous video job routes and `GET /v1/capabilities`.
///
/// Job creation takes condition media inline like the synchronous route, so
/// `POST /v1/videos` has the same `body_limit`.
pub(crate) fn job_router(engine: Arc<Engine>, jobs: Arc<VideoJobs>, body_limit: usize) -> Router {
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
        .with_state(JobState { engine, jobs })
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
    State(engine): State<Arc<Engine>>,
    Extension(RequestId(base_id)): Extension<RequestId>,
    VideoBody(body): VideoBody,
) -> Response {
    let started_at = Instant::now();
    let request_id = format!("vid-{base_id}");

    if let Err(error) = check_model_served(&body.model, engine.served_model_name()) {
        return error.into_response();
    }
    let (_, mut stream) = match engine.generate_video(request_id, body).await {
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
        if let ControlFlow::Break(outcome) = collector.push(event) {
            match outcome {
                Ok(artifact) => break artifact,
                Err(error) => return ApiError::from(error).into_response(),
            }
        }
    };

    let generation_ms = started_at.elapsed().as_secs_f64() * 1_000.0;
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, &artifact.content_type)
        .header(header::CONTENT_LENGTH, artifact.media.len())
        .header(
            "server-timing",
            format!("generation;dur={generation_ms:.1}"),
        );
    if let Some(timing) = clock.finish() {
        response = response
            .header(
                "x-inference-time-s",
                format!("{:.3}", timing.inference_time_s),
            )
            .header(
                "x-stage-durations",
                serde_json::to_string(&timing.stage_durations).unwrap_or_default(),
            );
    }
    response
        .body(media_body(artifact.media))
        .unwrap_or_else(|error| {
            ApiError::server_error(format!("failed to construct media response: {error}"))
                .into_response()
        })
}

/// Video request extractor accepting `application/json` or
/// `multipart/form-data`.
///
/// Both content types deserialize into the same `VideoGenerationRequest`,
/// whose `deny_unknown_fields` rejects unknown fields at every level. A
/// multipart body carries every field as a text part of the same name:
/// `conditions` and `target` as JSON text and the seed as an unsigned
/// integer. File parts, repeated fields and fields outside the schema are
/// rejected. A JSON body that is not the schema is a `json_parse_error`, a
/// refused multipart body an `invalid_request_error` (both `400`), and an
/// unsupported content type `415 Unsupported Media Type`. Semantic checks
/// (served model, task, conditions and target) happen in the handlers and in
/// `VideoService::process`.
pub(crate) struct VideoBody(pub VideoGenerationRequest);

/// How a multipart text part of the video request becomes a JSON value.
#[derive(Clone, Copy)]
enum PartKind {
    /// The text itself.
    Text,
    /// JSON text: an array or object field.
    Json,
    /// An unsigned integer.
    Unsigned,
}

/// The multipart part of each request field.
fn part_kind(name: &str) -> Option<PartKind> {
    Some(match name {
        "model" | "prompt" | "task" => PartKind::Text,
        "conditions" | "target" => PartKind::Json,
        "seed" => PartKind::Unsigned,
        _ => return None,
    })
}

/// Converts one multipart text part into the JSON value its field takes.
fn part_value(name: &'static str, kind: PartKind, text: String) -> Result<Value, ApiError> {
    let invalid =
        |message: &str| ApiError::invalid_request(format!("{name} {message}"), Some(name));
    match kind {
        PartKind::Text => Ok(Value::String(text)),
        PartKind::Json => {
            serde_json::from_str(&text).map_err(|error| invalid(&format!("must be JSON: {error}")))
        }
        PartKind::Unsigned => text
            .parse::<u64>()
            .map(Value::from)
            .map_err(|_| invalid("must be an unsigned integer")),
    }
}

/// Reads a multipart body into the JSON object its fields form.
async fn multipart_fields(mut multipart: Multipart) -> Result<Value, ApiError> {
    let invalid = |error: axum::extract::multipart::MultipartError| {
        ApiError::invalid_request(error.to_string(), None)
    };
    // Multipart values are all text, so each field is converted to the JSON
    // value its schema takes and the object is deserialized exactly like a
    // JSON body.
    let mut fields = serde_json::Map::new();
    while let Some(field) = multipart.next_field().await.map_err(invalid)? {
        let name = field.name().unwrap_or("").to_owned();
        let schema = REQUEST_FIELDS.iter().copied().find(|known| *known == name);
        let (Some(name), Some(kind)) = (schema, part_kind(&name)) else {
            return Err(ApiError::invalid_request(
                format!("unknown video field {name:?}"),
                None,
            ));
        };
        if field.file_name().is_some() {
            return Err(ApiError::invalid_request(
                format!("video field {name:?} must be text, not a file"),
                Some(name),
            ));
        }
        if fields.contains_key(name) {
            return Err(ApiError::invalid_request(
                format!("duplicate video field {name:?}"),
                Some(name),
            ));
        }
        let text = field.text().await.map_err(invalid)?;
        fields.insert(name.to_owned(), part_value(name, kind, text)?);
    }
    Ok(Value::Object(fields))
}

impl<S: Send + Sync> FromRequest<S> for VideoBody {
    type Rejection = ApiError;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        let content_type = request
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("");
        if content_type.starts_with("multipart/form-data") {
            let multipart = Multipart::from_request(request, state)
                .await
                .map_err(|error| ApiError::invalid_request(error.body_text(), None))?;
            serde_json::from_value(multipart_fields(multipart).await?)
                .map(Self)
                .map_err(|error| ApiError::invalid_request(error.to_string(), None))
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
    State(JobState { engine, jobs }): State<JobState>,
    Extension(RequestId(base_id)): Extension<RequestId>,
    VideoBody(body): VideoBody,
) -> Response {
    let request_id = format!("vid-{base_id}");
    if let Err(error) = check_model_served(&body.model, engine.served_model_name()) {
        return error.into_response();
    }
    let (prepared, mut stream) = match engine.generate_video(request_id, body).await {
        Ok(generation) => generation,
        Err(error) => return ApiError::from(error).into_response(),
    };

    // Public IDs are server-generated, so caller request-ID headers cannot
    // collide with retained jobs.
    let id = format!("video_{}", uuid::Uuid::new_v4().simple());
    let record = VideoJob {
        id: id.clone(),
        object: "video",
        model: engine.served_model_name().to_owned(),
        created_at: crate::openai::unix_time(),
        completed_at: None,
        expires_at: None,
        seconds: prepared.duration_seconds,
        actual_seconds: prepared.actual_seconds,
        num_frames: prepared.num_frames,
        size: format!("{}x{}", prepared.canvas.width, prepared.canvas.height),
        status: "queued",
        phase: "queued".to_owned(),
        completed_steps: 0,
        // The deployment's denoiser fixes every request's step count.
        total_steps: engine
            .video_denoiser()
            .map_or(0, |denoiser| denoiser.denoise_steps()),
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
            // stream, which cancels the request in its engine core.
            let event = tokio::select! {
                event = stream.next() => event,
                () = cancellation.cancelled() => return,
            };
            let event = match event.transpose() {
                Ok(event) => event,
                Err(error) => {
                    break Err(VideoFailure {
                        code: "generation_failed",
                        message: error.to_string(),
                    });
                }
            };
            if let Some((phase, steps)) = event.as_ref().and_then(|event| clock.observe(event)) {
                store.progress(&id, phase, steps);
            }
            if let ControlFlow::Break(outcome) = collector.push(event) {
                break outcome.map(|artifact| artifact.media).map_err(Into::into);
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
    (
        [
            (header::CONTENT_TYPE, "video/mp4".to_owned()),
            (header::CONTENT_LENGTH, length.to_string()),
        ],
        media_body(media),
    )
        .into_response()
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
/// (`InputProcessor::video_capabilities`), and the settings of the job store.
async fn capabilities(State(state): State<JobState>) -> Json<Value> {
    let settings = state.jobs.settings();
    Json(json!({
        "model": state.engine.served_model_name(),
        "video": state.engine.processor().video_capabilities(),
        "video_jobs": {
            "max_retained_bytes": settings.max_bytes,
            "retention_seconds": settings.retention.as_secs(),
            "restart_behavior": "jobs and retained content are removed",
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::ModelParameters;
    use crate::test_support::{SERVED_MODEL, post_json, router, send, sim_engine};

    fn video_router() -> Router {
        router(sim_engine(ModelParameters::MiniMaxH3))
    }

    fn video_request() -> serde_json::Value {
        serde_json::json!({
            "model": SERVED_MODEL,
            "prompt": "A river",
            "task": "t2va",
            "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5},
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

    /// An engine rejection reaches an asynchronous job with the error code the
    /// synchronous route answers the same rejection with.
    #[tokio::test]
    async fn a_rejected_job_reports_the_synchronous_error_code() {
        let router = video_router();

        // The simulator has no video media components, so the engine rejects
        // the request as invalid on both routes.
        let (sync_status, _, sync_body) =
            send(&router, post_json("/v1/videos/sync", &video_request())).await;
        assert_eq!(sync_status, StatusCode::BAD_REQUEST);

        let (status, _, job) = send(&router, post_json("/v1/videos", &video_request())).await;
        assert_eq!(status, StatusCode::OK);
        let uri = format!("/v1/videos/{}", job["id"].as_str().unwrap());
        let failed = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                let request = axum::extract::Request::builder()
                    .uri(&uri)
                    .body(Body::empty())
                    .unwrap();
                let (_, _, job) = send(&router, request).await;
                if job["status"] == "failed" {
                    return job;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();

        assert_eq!(failed["error"]["code"], sync_body["error"]["code"]);
    }

    /// Both submission routes refuse a model the server does not serve.
    #[tokio::test]
    async fn the_video_routes_refuse_another_model() {
        let router = video_router();
        let mut request = video_request();
        request["model"] = serde_json::json!("another-model");
        for path in ["/v1/videos/sync", "/v1/videos"] {
            let (status, _, body) = send(&router, post_json(path, &request)).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{path}");
            assert_eq!(body["error"]["code"], "model_not_found", "{path}");
        }
    }

    #[tokio::test]
    async fn json_and_multipart_normalize_to_the_same_request() {
        let conditions = r#"[{"type": "image", "uri": "data:image/png;base64,AA==", "role": "keyframe", "frame_index": 0}]"#;
        let target = r#"{"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 5.5}"#;
        let json = json_request(&serde_json::json!({
            "model": "FastH3",
            "prompt": "A river",
            "task": "fl2va",
            "conditions": serde_json::from_str::<serde_json::Value>(conditions).unwrap(),
            "target": serde_json::from_str::<serde_json::Value>(target).unwrap(),
            "seed": 7,
        }));
        let multipart = multipart(&[
            ("model", "FastH3"),
            ("prompt", "A river"),
            ("task", "fl2va"),
            ("conditions", conditions),
            ("target", target),
            ("seed", "7"),
        ]);

        let left = VideoBody::from_request(json, &()).await.unwrap().0;
        let right = VideoBody::from_request(multipart, &()).await.unwrap().0;

        assert_eq!(left, right);
        assert_eq!(left.task, models::video::VideoTask::Fl2va);
        assert_eq!(left.conditions[0].frame_index, Some(0));
        assert_eq!(left.target.duration_seconds, Some(5.5));
        assert_eq!(left.seed, 7);

        // An omitted seed is the reference's 42.
        let defaulted = VideoBody::from_request(json_request(&video_request()), &())
            .await
            .unwrap()
            .0;
        assert_eq!(defaulted.seed, crate::serving::video::DEFAULT_VIDEO_SEED);
    }

    /// Fields outside the schema, a schedule field included, and file parts
    /// are refused in both formats, and a body without a task names the
    /// missing `task`.
    #[tokio::test]
    async fn fields_outside_the_schema_are_rejected_in_both_formats() {
        let mut extra = video_request();
        extra["num_inference_steps"] = serde_json::json!(50);
        let mut nested = video_request();
        nested["target"]["fps"] = serde_json::json!(24);
        let file = axum::extract::Request::builder()
            .header(header::CONTENT_TYPE, "multipart/form-data; boundary=clip")
            .body(Body::from(
                "--clip\r\nContent-Disposition: form-data; name=\"conditions\"; filename=\"image.png\"\r\n\r\nimage\r\n--clip--\r\n",
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
            json_request(&serde_json::json!({"model": "FastH3", "prompt": "A river"})),
            multipart(&[("model", "FastH3"), ("prompt", "A river")]),
        ] {
            let error = match VideoBody::from_request(request, &()).await {
                Ok(VideoBody(body)) => panic!("{body:?} was accepted"),
                Err(error) => error,
            };
            assert_eq!(error.status_code(), StatusCode::BAD_REQUEST);
            let message = error.to_error_response().error.message;
            assert!(message.contains("task"), "{message}");
        }
    }

    /// The capabilities report what the placed denoiser serves.
    #[tokio::test]
    async fn capabilities_follow_the_denoiser_handshake() {
        let router = video_router();
        let request = axum::extract::Request::builder()
            .uri("/v1/capabilities")
            .body(Body::empty())
            .unwrap();
        let (status, _, body) = send(&router, request).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["video"]["tasks"], serde_json::json!(["t2va"]));
        assert_eq!(body["video"]["schedule"]["num_inference_steps"], 5);
        // The deployment prepared the named aspect ratios' canvases at 768p,
        // each reported with the pair that selects it.
        let canvas = |aspect_ratio: &str, width: u32, height: u32| {
            serde_json::json!({
                "short_edge": 768,
                "aspect_ratio": aspect_ratio,
                "width": width,
                "height": height,
            })
        };
        assert_eq!(
            body["video"]["canvas"]["canvases"],
            serde_json::json!([
                canvas("21:9", 1536, 672),
                canvas("16:9", 1344, 768),
                canvas("4:3", 1024, 768),
                canvas("1:1", 768, 768),
                canvas("3:4", 768, 1024),
                canvas("9:16", 768, 1344),
            ])
        );
        assert_eq!(
            body["video"]["canvas"]["aspect_ratios"],
            serde_json::json!(["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"])
        );
    }
}
