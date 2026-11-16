//! Video APIs: UniServe's `/v1/videos`, sent as the task builds it, and the
//! native forms the baselines this harness compares with take for the same work.
//!
//! The synchronous route answers with the MP4 itself; the job route answers
//! with a job, polled until it completes, whose content is then downloaded.
//! Encoded bodies stay in host memory and are inspected only after the load,
//! so no media work delays the next request; even an invalid body is kept for
//! the run's failure artifacts.
//!
//! The baselines (SGLang, vLLM-Omni, FastVideo) take a video request in their
//! own forms, which name the request's task as the MiniMax-H3 video family's
//! forms do (`t2va`, `fl2va`, `ref2va`): the forms are those systems' APIs for
//! that family, and UniServe's body is translated into them so that each is
//! asked for the same work.

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};
use serde_json::{Map, Value, json};

use crate::record::{Clock, Record, TaskRequest, VideoShape};
use crate::spec::{VideoBackend, parse_size};

/// The headers a synchronous route reports its timings in.
const TIMING_HEADERS: [&str; 3] = [
    "x-inference-time-s",
    "x-stage-durations",
    "x-peak-memory-mb",
];
/// The headers a synchronous route reports the shape it generated in.
const SHAPE_HEADERS: [&str; 3] = ["x-video-size", "x-video-num-frames", "x-video-fps"];

/// A request's body, as its backend takes it.
pub enum Native {
    Json(Map<String, Value>),
    /// A multipart form: its text fields, then its files as
    /// `(name, file name, media type, bytes)`.
    Form(
        Vec<(String, String)>,
        Vec<(String, String, String, Vec<u8>)>,
    ),
}

/// The body the request's backend takes for the request's work.
///
/// # Errors
///
/// When the request carries no video settings, or its backend cannot be sent
/// its references as the same work: in their order, or with a field it lacks.
pub fn native_request(request: &TaskRequest) -> Result<Native> {
    let video = request
        .video
        .as_ref()
        .context("a video request carries its point's video settings")?;
    let payload = request.payload.clone();
    if video.backend == VideoBackend::Uniserve {
        return Ok(Native::Json(payload));
    }
    let shape = request
        .video_shape
        .context("a baseline video request needs its video shape")?;
    let references: Vec<Map<String, Value>> = payload
        .get("input_references")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_object).cloned().collect())
        .unwrap_or_default();
    let task = family_task(&references)?;
    let field = |key: &str| payload.get(key).cloned().unwrap_or(Value::Null);
    match video.backend {
        VideoBackend::Uniserve => Ok(Native::Json(request.payload.clone())),
        VideoBackend::Sglang => {
            // Native admission names its canvas by the short edge and an aspect
            // ratio and its frames by a duration, which the point states as the
            // ones resolving to its shape; each reference is a condition whose
            // role is `keyframe` for an anchor and `reference` otherwise.
            let conditions: Vec<Value> = references
                .iter()
                .map(|reference| {
                    let mut condition = Map::new();
                    condition.insert("type".into(), reference["type"].clone());
                    condition.insert("uri".into(), reference["url"].clone());
                    let role = if reference.contains_key("frame_index") {
                        "keyframe"
                    } else {
                        "reference"
                    };
                    condition.insert("role".into(), role.into());
                    for name in ["frame_index", "start_time_seconds"] {
                        if let Some(value) = reference.get(name) {
                            condition.insert(name.into(), value.clone());
                        }
                    }
                    Value::Object(condition)
                })
                .collect();
            let mut body = Map::new();
            body.insert("model".into(), field("model"));
            body.insert("prompt".into(), field("prompt"));
            body.insert("task".into(), task.into());
            body.insert("conditions".into(), Value::Array(conditions));
            body.insert(
                "target".into(),
                json!({
                    "short_edge": shape.width.min(shape.height),
                    "aspect_ratio": video.aspect_ratio,
                    "duration_seconds": video.duration_seconds,
                }),
            );
            body.insert("seed".into(), field("seed"));
            body.extend(video.schedule());
            body.extend(video.extra_params.clone());
            Ok(Native::Json(body))
        }
        VideoBackend::VllmOmni => {
            type_order(video.backend, &references)?;
            // Omni counts schedule intervals (denoiser forwards), not points.
            // Media go up as repeated `input_references` parts, classified by
            // media type; keyframes take their indices in extra_params.
            let keyframes: Vec<Value> = references
                .iter()
                .filter_map(|reference| reference.get("frame_index").cloned())
                .collect();
            ensure!(
                keyframes.is_empty() || task == "fl2va",
                "vLLM-Omni takes keyframes only for fl2va"
            );
            let steps = video.num_inference_steps.context("vLLM-Omni's schedule")?;
            let mut extra = Map::new();
            extra.insert("task".into(), task.into());
            extra.insert(
                "duration".into(),
                video
                    .duration_seconds
                    .context("vLLM-Omni's duration")?
                    .into(),
            );
            extra.insert(
                "audio_flow_shift".into(),
                video
                    .audio_flow_shift
                    .context("vLLM-Omni's audio shift")?
                    .into(),
            );
            if !keyframes.is_empty() {
                extra.insert("frame_indices".into(), Value::Array(keyframes));
            }
            let starts: Vec<f64> = references
                .iter()
                .filter(|reference| reference.get("type").and_then(Value::as_str) == Some("video"))
                .map(|reference| {
                    reference
                        .get("start_time_seconds")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0)
                })
                .collect();
            if starts.iter().any(|start| *start != 0.0) {
                extra.insert("start_time_seconds".into(), json!(starts));
            }
            extra.extend(video.extra_params.clone());
            let text = |value: &Value| match value {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            let fields = vec![
                ("model".to_owned(), text(&field("model"))),
                ("prompt".to_owned(), text(&field("prompt"))),
                ("seed".to_owned(), text(&field("seed"))),
                ("width".to_owned(), shape.width.to_string()),
                ("height".to_owned(), shape.height.to_string()),
                (
                    "aspect_ratio".to_owned(),
                    video.aspect_ratio.clone().unwrap_or_default(),
                ),
                ("fps".to_owned(), shape.fps.to_string()),
                ("num_frames".to_owned(), shape.frames.to_string()),
                ("num_inference_steps".to_owned(), (steps - 1).to_string()),
                ("guidance_scale".to_owned(), "1.0".to_owned()),
                (
                    "flow_shift".to_owned(),
                    format!("{:?}", video.flow_shift.context("vLLM-Omni's shift")?),
                ),
                ("extra_params".to_owned(), serde_json::to_string(&extra)?),
            ];
            let files = request
                .reference_media
                .iter()
                .map(|media| {
                    let name = std::path::Path::new(&media.path)
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    (
                        "input_references".to_owned(),
                        name,
                        media.mime.clone(),
                        media.data.clone(),
                    )
                })
                .collect();
            Ok(Native::Form(fields, files))
        }
        VideoBackend::Fastvideo => {
            type_order(video.backend, &references)?;
            // FastVideo takes its shifts from the checkpoint and counts the
            // schedule in sigma points, or in fused-block forwards for a
            // parallel-decoding student; it validates an aligned frame count, so
            // the stated frames carry the work, an integral duration beside
            // them. References name local files by path in its typed fields; it
            // has no keyframe or start-offset fields.
            let points = video.num_inference_steps.context("FastVideo's schedule")?;
            let mut body = Map::new();
            body.insert("model".into(), field("model"));
            body.insert("prompt".into(), field("prompt"));
            body.insert("seed".into(), field("seed"));
            body.insert("size".into(), shape.size().into());
            body.insert("fps".into(), shape.fps.into());
            body.insert(
                "num_inference_steps".into(),
                (if video.parallel_decoding {
                    points - 1
                } else {
                    points
                })
                .into(),
            );
            body.insert("num_frames".into(), shape.frames.into());
            let seconds = video.duration_seconds.context("FastVideo's duration")?;
            if seconds.fract() == 0.0 {
                body.insert("seconds".into(), seconds.into());
            }
            if task == "t2va" {
                return Ok(Native::Json(body));
            }
            let mut typed: BTreeMap<&str, Vec<Value>> = BTreeMap::new();
            ensure!(
                references.len() == request.reference_media.len(),
                "every FastVideo reference names its media"
            );
            for (reference, media) in references.iter().zip(&request.reference_media) {
                ensure!(
                    !reference.contains_key("frame_index"),
                    "FastVideo takes references without keyframes"
                );
                ensure!(
                    reference
                        .get("start_time_seconds")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0)
                        == 0.0,
                    "FastVideo has no reference start offset"
                );
                let (field, key) = match reference.get("type").and_then(Value::as_str) {
                    Some("image") => ("image_reference", "image_url"),
                    Some("video") => ("video_reference", "video_url"),
                    _ => ("audio_reference", "audio_url"),
                };
                typed
                    .entry(field)
                    .or_default()
                    .push(json!({key: media.path}));
            }
            body.insert("task".into(), task.into());
            for (field, items) in typed {
                body.insert(field.into(), Value::Array(items));
            }
            Ok(Native::Json(body))
        }
    }
}

/// The family's task the baselines' forms name: no reference is `t2va`, image
/// anchors alone (each with a `frame_index`) are `fl2va`, any reference without
/// an anchor is `ref2va`.
///
/// # Errors
///
/// When a video or audio reference carries an anchor, which no task takes.
pub fn family_task(references: &[Map<String, Value>]) -> Result<&'static str> {
    if references.is_empty() {
        return Ok("t2va");
    }
    if references
        .iter()
        .any(|reference| !reference.contains_key("frame_index"))
    {
        return Ok("ref2va");
    }
    ensure!(
        references
            .iter()
            .all(|reference| reference.get("type").and_then(Value::as_str) == Some("image")),
        "only image references may anchor a keyframe"
    );
    Ok("fl2va")
}

/// A type-grouping backend presents images, then videos, then audio: the
/// references must already be in that order.
fn type_order(backend: VideoBackend, references: &[Map<String, Value>]) -> Result<()> {
    let rank = |reference: &Map<String, Value>| match reference.get("type").and_then(Value::as_str)
    {
        Some("image") => 0,
        Some("video") => 1,
        _ => 2,
    };
    let order: Vec<u8> = references.iter().map(rank).collect();
    ensure!(
        order.windows(2).all(|pair| pair[0] <= pair[1]),
        "{backend:?} presents images, then videos, then audio; the request's references must already be in that order"
    );
    Ok(())
}

/// Sends a video request, polls a job until it completes, and collects the
/// media's bytes.
pub async fn receive(
    client: &reqwest::Client,
    clock: &Clock,
    base_url: &str,
    request: &TaskRequest,
    record: &mut Record,
) -> Result<()> {
    let video = request
        .video
        .as_ref()
        .context("a video request carries its point's video settings")?;
    let url = format!("{}{}", base_url.trim_end_matches('/'), request.endpoint);
    let native = native_request(request)?;
    let build = |builder: reqwest::RequestBuilder| -> Result<reqwest::RequestBuilder> {
        Ok(match &native {
            Native::Json(body) => builder.json(body),
            Native::Form(fields, files) => {
                let mut form = reqwest::multipart::Form::new();
                for (name, value) in fields {
                    form = form.text(name.clone(), value.clone());
                }
                for (name, file, mime, data) in files {
                    let part = reqwest::multipart::Part::bytes(data.clone())
                        .file_name(file.clone())
                        .mime_str(mime)?;
                    form = form.part(name.clone(), part);
                }
                builder.multipart(form)
            }
        })
    };
    if request.endpoint.ends_with("/sync") {
        return media_body(build(client.post(&url))?, clock, record).await;
    }
    let response = build(client.post(&url))?.send().await?;
    record.note_http(clock, response.status().as_u16());
    if failed(response.status().as_u16(), &response, clock, record) {
        let detail = response.text().await.unwrap_or_default();
        record.error = Some(detail.chars().take(500).collect());
        return Ok(());
    }
    let mut job: Value = response.json().await?;
    let id = job
        .get("id")
        .map(text_of)
        .context("a video job without an id")?;
    let poll = format!("{url}/{}", url_escape(&id));
    loop {
        match job.get("status").and_then(Value::as_str) {
            Some("completed") => break,
            Some("failed" | "cancelled") => {
                record.close_now(clock);
                let error = job.get("error").cloned().unwrap_or_else(|| job.clone());
                record.mark_failure("video_job_failed", Some(text_of(&error)));
                return Ok(());
            }
            _ => {}
        }
        tokio::time::sleep(std::time::Duration::from_secs_f64(video.poll_interval_s)).await;
        let response = client.get(&poll).send().await?;
        if failed(response.status().as_u16(), &response, clock, record) {
            let detail = response.text().await.unwrap_or_default();
            record.error = Some(detail.chars().take(500).collect());
            return Ok(());
        }
        job = response.json().await?;
    }
    timings(
        record,
        job.get("inference_time_s"),
        job.get("stage_durations"),
        job.get("peak_storage_mb"),
    );
    record.served_shape = served_shape(
        job.get("size").map(text_of).as_deref(),
        job.get("num_frames").map(text_of).as_deref(),
        job.get("fps").map(text_of).as_deref(),
    );
    let content = match job
        .get("url")
        .and_then(Value::as_str)
        .filter(|url| !url.is_empty())
    {
        Some(relative) => join_url(base_url, relative),
        None => format!("{poll}/content"),
    };
    media_body(client.get(&content), clock, record).await
}

/// A media response: its type, its bytes (kept even when invalid), and, on a
/// synchronous route, the timings and shape its headers report (a job's content
/// download carries none, and the job's stand).
async fn media_body(
    builder: reqwest::RequestBuilder,
    clock: &Clock,
    record: &mut Record,
) -> Result<()> {
    let response = builder.send().await?;
    let status = response.status().as_u16();
    record.note_http(clock, status);
    record.video_mime = super::header(&response, "content-type");
    let headers = response.headers().clone();
    let data = response.bytes().await?.to_vec();
    record.close_now(clock);
    if status >= 400 {
        record.mark_failure(
            &format!("transport_status_{status}"),
            Some(String::from_utf8_lossy(&data).chars().take(500).collect()),
        );
        record.video_body = Some(data);
        return Ok(());
    }
    record.video_body = Some(data);
    record.mark_success();
    let value = |name: &str| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    };
    if TIMING_HEADERS
        .iter()
        .any(|name| headers.contains_key(*name))
    {
        timings(
            record,
            value("x-inference-time-s").map(Value::String).as_ref(),
            value("x-stage-durations").map(Value::String).as_ref(),
            value("x-peak-memory-mb").map(Value::String).as_ref(),
        );
    }
    if SHAPE_HEADERS.iter().any(|name| headers.contains_key(*name)) {
        record.served_shape = served_shape(
            value(SHAPE_HEADERS[0]).as_deref(),
            value(SHAPE_HEADERS[1]).as_deref(),
            value(SHAPE_HEADERS[2]).as_deref(),
        );
    }
    Ok(())
}

/// Marks a failed status; returns whether it failed.
fn failed(status: u16, _response: &reqwest::Response, clock: &Clock, record: &mut Record) -> bool {
    if status < 400 {
        return false;
    }
    record.status_code = Some(status);
    if record.final_event_time.is_none() {
        record.close_now(clock);
    }
    record.mark_failure(&format!("transport_status_{status}"), None);
    true
}

/// The shape a server reported, or `None` when its values are not one.
fn served_shape(size: Option<&str>, frames: Option<&str>, fps: Option<&str>) -> Option<VideoShape> {
    let (width, height) = parse_size(size?).ok()?;
    let frames: u32 = frames?.trim().parse().ok()?;
    let fps: u32 = fps?.trim().parse().ok()?;
    (frames >= 1 && fps >= 1).then_some(VideoShape {
        width,
        height,
        frames,
        fps,
    })
}

/// The timings a server reported that read as finite numbers; a malformed or
/// absent value stays unset.
fn timings(
    record: &mut Record,
    inference: Option<&Value>,
    stages: Option<&Value>,
    peak: Option<&Value>,
) {
    record.server_inference_s = inference.and_then(finite);
    record.server_peak_storage_mib = peak.and_then(finite);
    let stages = match stages {
        Some(Value::String(text)) => serde_json::from_str::<Value>(text).ok(),
        Some(other) => Some(other.clone()),
        None => None,
    };
    if let Some(Value::Object(stages)) = stages {
        record.server_stage_s = stages
            .iter()
            .filter_map(|(name, seconds)| Some((name.clone(), finite(seconds)?)))
            .collect();
    }
}

fn finite(value: &Value) -> Option<f64> {
    let number = match value {
        Value::Number(number) => number.as_f64()?,
        Value::String(text) => text.trim().parse().ok()?,
        _ => return None,
    };
    number.is_finite().then_some(number)
}

fn text_of(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Percent-escapes a path segment.
fn url_escape(segment: &str) -> String {
    segment
        .bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                (byte as char).to_string()
            }
            other => format!("%{other:02X}"),
        })
        .collect()
}

/// A job's content URL against the server's origin.
fn join_url(base: &str, relative: &str) -> String {
    if relative.starts_with("http://") || relative.starts_with("https://") {
        return relative.to_owned();
    }
    let base = base.trim_end_matches('/');
    if let Some(path) = relative.strip_prefix('/') {
        // An absolute path replaces the origin's path.
        let origin = base
            .find("://")
            .and_then(|scheme| base[scheme + 3..].find('/').map(|path| scheme + 3 + path));
        return format!("{}/{path}", origin.map_or(base, |end| &base[..end]));
    }
    format!("{base}/{relative}")
}
