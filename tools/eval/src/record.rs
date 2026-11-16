//! What one request observably did, the rows requests are built from, and the
//! validation a run's records are judged by.

use std::collections::BTreeMap;
use std::time::Instant;

use serde::Serialize;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::spec::Video;

/// The classifier of a request a timeline load's `cancel` change disconnected:
/// the change's doing, which a point leaves out of its metrics and validation
/// and `check-change` out of its items.
pub const CANCELLED: &str = "cancelled";

/// A dataset row, with the overrides a task applies before the point's
/// settings.
///
/// `prompt_len` and `output_len` are dataset-side token counts that stand in
/// for server-reported usage when a response carries none; the text task also
/// takes `output_len` as its output limit. `num_frames` is a video row's exact
/// frame count. `input_references` lists a video row's reference media in
/// request order: each the request's `type` and, where they apply,
/// `frame_index` (`-1` is the last frame) and `start_time_seconds`, and `media`,
/// the file's path relative to the point's `video.reference_root`. Rows of one
/// `session_id` are successive decisions: the next is sent only after the
/// preceding response.
#[derive(Debug, Clone, Default, PartialEq, Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Example {
    pub id: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_len: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_len: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_image_b64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_image_mime: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_frames: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_references: Option<Vec<Map<String, Value>>>,
    /// Always written, empty or not, as the row identity's digest covers it.
    pub metadata: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub questions: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

impl Example {
    /// The row's populated fields as JSON.
    pub fn as_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// The count and digest of the selected rows: each row's populated fields in
/// row order, as canonical JSON (sorted keys, no spaces, ASCII), so the same
/// rows in the same order give the same identity across runs.
pub fn rows_identity(rows: &[Example]) -> Value {
    let values: Vec<Value> = rows.iter().map(|row| sorted(&row.as_value())).collect();
    let encoded = canonical(&Value::Array(values));
    json!({"count": rows.len(), "sha256": hex(&Sha256::digest(encoded.as_bytes()))})
}

/// A value with every object's keys in order, as canonical JSON writes them.
fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(entries) => {
            let ordered: BTreeMap<&String, Value> = entries
                .iter()
                .map(|(key, item)| (key, sorted(item)))
                .collect();
            Value::Object(
                ordered
                    .into_iter()
                    .map(|(key, item)| (key.clone(), item))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

/// JSON with no spaces and every non-ASCII character escaped, the canonical
/// form digests are taken over.
pub fn canonical(value: &Value) -> String {
    let text = serde_json::to_string(value).unwrap_or_default();
    let mut ascii = String::with_capacity(text.len());
    for character in text.chars() {
        if character.is_ascii() {
            ascii.push(character);
        } else {
            let mut units = [0u16; 2];
            for unit in character.encode_utf16(&mut units) {
                ascii.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    ascii
}

/// Lowercase hexadecimal of bytes.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// A video's raster in pixels, its frame count and its frame rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct VideoShape {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
}

impl VideoShape {
    /// The raster as the API spells it.
    pub fn size(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }

    /// The duration the frames span at the frame rate.
    pub fn duration_s(&self) -> f64 {
        f64::from(self.frames) / f64::from(self.fps)
    }
}

/// A reference's media file, read before the request is measured.
#[derive(Debug, Clone)]
pub struct ReferenceMedia {
    /// The file's absolute path.
    pub path: String,
    /// Its media type, from its suffix.
    pub mime: String,
    /// Its bytes, for a backend that takes media as request parts.
    pub data: Vec<u8>,
}

/// An endpoint payload and its streaming mode.
///
/// A video request's `payload` is UniServe's `/v1/videos` body, which the
/// video transport sends to UniServe as is and translates into each
/// baseline's form; `video` names the backend and the fields a baseline needs
/// beyond the body, `video_shape` the stated shape the output is held to, and
/// `reference_media` the body's `file://` references' files in order, for
/// backends that take uploads or plain paths.
#[derive(Debug, Clone)]
pub struct TaskRequest {
    pub endpoint: String,
    pub payload: Map<String, Value>,
    pub stream: bool,
    pub video: Option<Video>,
    pub video_shape: Option<VideoShape>,
    pub reference_media: Vec<ReferenceMedia>,
}

impl TaskRequest {
    /// A request of a chat or image task.
    pub fn new(endpoint: &str, payload: Map<String, Value>, stream: bool) -> Self {
        Self {
            endpoint: endpoint.to_owned(),
            payload,
            stream,
            video: None,
            video_shape: None,
            reference_media: Vec::new(),
        }
    }
}

/// A validated image: its bytes and what its content states. `mime` is the
/// detected format, and `sample_filename` the digest with the format's
/// extension.
#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub data: Vec<u8>,
    pub sha256: String,
    pub mime: String,
    pub width: u32,
    pub height: u32,
    pub sample_filename: String,
}

impl DecodedImage {
    /// What a record keeps of the image: everything but its bytes.
    pub fn metadata(&self) -> Value {
        json!({
            "sha256": self.sha256,
            "byte_size": self.data.len(),
            "mime": self.mime,
            "width": self.width,
            "height": self.height,
            "sample_filename": self.sample_filename,
        })
    }
}

/// A validated MP4 and what its streams state. The frame rate is the video
/// stream's average rate as an exact rational, and `audio_samples` counts
/// samples per channel.
#[derive(Debug, Clone)]
pub struct DecodedVideo {
    pub data: Vec<u8>,
    pub sha256: String,
    pub mime: String,
    pub width: u32,
    pub height: u32,
    pub frame_count: u64,
    pub fps_numerator: u64,
    pub fps_denominator: u64,
    pub video_codec: String,
    pub audio_codec: String,
    pub audio_channels: u32,
    pub audio_sample_rate: u32,
    pub audio_samples: u64,
    pub sample_filename: String,
    pub video_variance: f64,
    pub audio_rms: f64,
}

impl DecodedVideo {
    /// The frame rate.
    pub fn fps(&self) -> f64 {
        self.fps_numerator as f64 / self.fps_denominator as f64
    }

    /// The duration the decoded frames span.
    pub fn video_duration_s(&self) -> f64 {
        self.frame_count as f64 / self.fps()
    }

    /// The duration the decoded samples span.
    pub fn audio_duration_s(&self) -> f64 {
        self.audio_samples as f64 / f64::from(self.audio_sample_rate)
    }

    /// What a record keeps of the video: everything but its bytes.
    pub fn metadata(&self) -> Value {
        json!({
            "sha256": self.sha256,
            "byte_size": self.data.len(),
            "mime": self.mime,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "fps_numerator": self.fps_numerator,
            "fps_denominator": self.fps_denominator,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "audio_channels": self.audio_channels,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_samples": self.audio_samples,
            "video_variance": self.video_variance,
            "audio_rms": self.audio_rms,
            "video_duration_s": self.video_duration_s(),
            "audio_duration_s": self.audio_duration_s(),
            "sample_filename": self.sample_filename,
        })
    }
}

/// Where a token count came from: the server's usage, or a fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    ServerUsage,
    RequestFallback,
    RequestedFallback,
}

/// One generated token: its id (or text, when the server named none), the
/// arrival time of the chunk that carried it, and its logprob candidates as
/// `(id, logprob)` pairs.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub id: Value,
    pub time: Option<f64>,
    pub candidates: Vec<(Value, Value)>,
}

/// What one request observably did.
///
/// The transport creates a record per request and calls [`Record::begin`]
/// before dispatch; the endpoint transports record the HTTP status, close the
/// record and classify it. Times are seconds on the run's monotonic clock
/// ([`Clock`]); `scheduled_time` is the arrival time the load assigned (`None`
/// for warm-up), and `start_time` is taken after any concurrency queueing, so
/// their difference is the client's dispatch wait. `latency`, `ttft` and the
/// image latencies are seconds from `start_time`, milliseconds only in
/// [`Record::to_value`].
///
/// `itl` holds the gaps between consecutive text-bearing stream events, not
/// per-token gaps (one event may carry several tokens), and leaves out a gap
/// that spans an image event; `text_times` holds every stamped text event's
/// arrival, the image-spanning ones included.
#[derive(Debug, Clone, Default)]
pub struct Record {
    pub request_id: String,
    pub task: String,
    pub success: bool,
    pub classifier: String,
    pub error: Option<String>,
    pub warnings: Vec<String>,
    pub endpoint: String,
    /// The origin the request was routed to: replicas make this per request.
    pub base_url: String,
    pub scheduled_time: Option<f64>,
    pub start_time: f64,
    pub http_response_time: Option<f64>,
    pub first_text_time: Option<f64>,
    pub first_image_done_time: Option<f64>,
    pub final_event_time: Option<f64>,
    pub latency: f64,
    pub ttft: f64,
    pub itl: Vec<f64>,
    pub text_times: Vec<f64>,
    /// The arrival of every chunk of a stream, whether it carries a token or not.
    pub chunk_times: Vec<f64>,
    pub tokens: Vec<Token>,
    pub token_timing_available: bool,
    pub prompt_len: u64,
    pub output_len: u64,
    pub requested_output_len: u64,
    pub prompt_len_source: Option<Source>,
    pub output_len_source: Option<Source>,
    pub cached_prompt_tokens: Option<u64>,
    pub generated_text: String,
    pub images: usize,
    pub image_latencies: Vec<f64>,
    pub first_image_latency: Option<f64>,
    pub image_steps: Vec<u64>,
    pub decoded_images: Vec<DecodedImage>,
    pub decoded_video: Option<DecodedVideo>,
    pub video_body: Option<Vec<u8>>,
    pub video_mime: String,
    /// The shape the request stated, which the output is held to.
    pub video_shape: Option<VideoShape>,
    /// The shape the server reported generating.
    pub served_shape: Option<VideoShape>,
    /// What a video server reported of this request: its inference time, its
    /// named stage durations, and its peak device storage in MiB.
    pub server_inference_s: Option<f64>,
    pub server_stage_s: BTreeMap<String, f64>,
    pub server_peak_storage_mib: Option<f64>,
    pub media_checks: BTreeMap<String, bool>,
    pub original_output: Option<Value>,
    pub example: Option<Value>,
    pub status_code: Option<u16>,
    pub finish_reason: Option<String>,
    pub stop_reason: Option<String>,
}

/// The run's monotonic clock: seconds since the run's start.
#[derive(Debug, Clone, Copy)]
pub struct Clock {
    origin: Instant,
}

impl Clock {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }

    /// Seconds since the clock's origin.
    pub fn now(&self) -> f64 {
        self.origin.elapsed().as_secs_f64()
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

impl Record {
    /// A record of request `request_id` of `task`.
    pub fn new(request_id: String, task: &str) -> Self {
        Self {
            request_id,
            task: task.to_owned(),
            ..Self::default()
        }
    }

    /// Starts the request's clock and states its endpoint, its scheduled
    /// arrival and the output limit it asked for.
    pub fn begin(
        &mut self,
        clock: &Clock,
        endpoint: &str,
        scheduled_time: Option<f64>,
        requested_output_len: u64,
    ) {
        self.endpoint = endpoint.to_owned();
        self.scheduled_time = scheduled_time;
        self.requested_output_len = requested_output_len;
        self.start_time = clock.now();
    }

    /// The response's headers arrived with `status`.
    pub fn note_http(&mut self, clock: &Clock, status: u16) {
        self.http_response_time = Some(clock.now());
        self.status_code = Some(status);
    }

    /// Closes the request now.
    pub fn close_now(&mut self, clock: &Clock) {
        self.close_at(clock.now());
    }

    /// Closes the request at an event's time.
    pub fn close_at(&mut self, time: f64) {
        self.final_event_time = Some(time);
        self.latency = time - self.start_time;
    }

    /// Marks the request unsuccessful; an `error` of `None` keeps the detail
    /// recorded before.
    pub fn mark_failure(&mut self, classifier: &str, error: Option<String>) {
        self.success = false;
        self.classifier = classifier.to_owned();
        if let Some(error) = error {
            self.error = Some(error);
        }
    }

    pub fn mark_success(&mut self) {
        self.success = true;
        self.classifier = "ok".to_owned();
    }

    /// Closes the request now as one its client disconnected on purpose.
    pub fn mark_cancelled(&mut self, clock: &Clock) {
        self.close_now(clock);
        self.mark_failure(CANCELLED, Some("the client disconnected the stream".into()));
    }

    /// Closes and classifies a request whose transport failed.
    pub fn mark_transport_failure(&mut self, clock: &Clock, error: &anyhow::Error) {
        self.close_now(clock);
        self.mark_failure("transport_failure", Some(format!("{error:#}")));
    }

    /// The finish and stop reasons of a completion choice.
    pub fn apply_choice(&mut self, choice: &Map<String, Value>) {
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(reason.to_owned());
        }
        if let Some(reason) = choice.get("stop_reason").and_then(Value::as_str) {
            self.stop_reason = Some(reason.to_owned());
        }
    }

    /// The server's token and image-step usage, which replaces any count
    /// before it and no fallback overwrites.
    pub fn apply_usage(&mut self, usage: &Map<String, Value>) {
        if let Some(tokens) = count(usage.get("completion_tokens")) {
            self.output_len = tokens;
            self.output_len_source = Some(Source::ServerUsage);
        }
        if let Some(tokens) = count(usage.get("prompt_tokens")) {
            self.prompt_len = tokens;
            self.prompt_len_source = Some(Source::ServerUsage);
        }
        if let Some(Value::Array(steps)) = usage.get("image_steps_per_image") {
            let parsed: Option<Vec<u64>> = steps.iter().map(|step| count(Some(step))).collect();
            if let Some(parsed) = parsed {
                self.image_steps = parsed;
            }
        }
        if let Some(cached) = usage
            .get("prompt_tokens_details")
            .and_then(Value::as_object)
            .and_then(|details| count(details.get("cached_tokens")))
        {
            self.cached_prompt_tokens = Some(cached);
        }
    }

    /// The token counts the server did not report.
    pub fn apply_token_fallbacks(&mut self, prompt_len: u64, output_len: u64) {
        if self.output_len_source != Some(Source::ServerUsage) {
            self.output_len = output_len;
            self.output_len_source = Some(Source::RequestedFallback);
        }
        if self.prompt_len_source != Some(Source::ServerUsage) {
            self.prompt_len = prompt_len;
            self.prompt_len_source = Some(Source::RequestFallback);
        }
    }

    /// Appends one stream event's text. An unstamped event (`time` `None`)
    /// updates no timing; `last_text_time` `None` makes this event the first
    /// token; `count_itl` false leaves the gap since the last text out of
    /// `itl`, as after an image event.
    pub fn add_text(
        &mut self,
        content: &str,
        time: Option<f64>,
        last_text_time: Option<f64>,
        count_itl: bool,
    ) {
        self.token_timing_available = true;
        self.generated_text.push_str(content);
        let Some(time) = time else { return };
        self.text_times.push(time);
        match last_text_time {
            None => {
                self.ttft = time - self.start_time;
                self.first_text_time = Some(time);
            }
            Some(last) if count_itl => self.itl.push(time - last),
            Some(_) => {}
        }
    }

    /// The arrival of one stream chunk.
    pub fn add_chunk(&mut self, time: f64) {
        self.chunk_times.push(time);
    }

    /// The tokens one streamed choice carries. A server names generated ids in
    /// the choice's `token_ids` and its logprob entries' tokens as
    /// `token_id:<id>`; the ids come from `token_ids`, else from the logprob
    /// entries, and a token's candidates are its entry's `top_logprobs`.
    pub fn add_tokens(&mut self, choice: &Map<String, Value>, time: Option<f64>) {
        let entries: Vec<&Map<String, Value>> = choice
            .get("logprobs")
            .and_then(Value::as_object)
            .and_then(|logprobs| logprobs.get("content"))
            .and_then(Value::as_array)
            .map(|content| content.iter().filter_map(Value::as_object).collect())
            .unwrap_or_default();
        let ids: Vec<Value> = match choice.get("token_ids") {
            Some(Value::Array(ids)) => ids.clone(),
            _ => entries
                .iter()
                .map(|entry| token_identity(entry.get("token")))
                .collect(),
        };
        for (index, id) in ids.into_iter().enumerate() {
            let candidates = entries
                .get(index)
                .and_then(|entry| entry.get("top_logprobs"))
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_object)
                        .map(|item| {
                            (
                                token_identity(item.get("token")),
                                item.get("logprob").cloned().unwrap_or(Value::Null),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
            self.tokens.push(Token {
                id,
                time,
                candidates,
            });
        }
    }

    /// `count` image parts arrived at `time`: each takes that latency.
    pub fn add_image_arrival(&mut self, count: usize, time: Option<f64>) {
        let Some(time) = time else { return };
        let latency = time - self.start_time;
        if self.first_image_latency.is_none() {
            self.first_image_latency = Some(latency);
            self.first_image_done_time = Some(time);
        }
        self.image_latencies
            .extend(std::iter::repeat_n(latency, count));
    }

    /// Attaches the validated images; with `json_latency`, each takes the whole
    /// request's latency, so the record must already be closed.
    pub fn attach_images(&mut self, decoded: Vec<DecodedImage>, json_latency: bool) {
        self.images = decoded.len();
        if json_latency && !decoded.is_empty() {
            self.image_latencies = vec![self.latency; decoded.len()];
        }
        self.decoded_images = decoded;
    }

    /// The durable record: identity and classification, the raw times, the
    /// derived durations in milliseconds, the tokens, the text with its size and
    /// digest, and the media's metadata (their bytes are artifacts of their own).
    pub fn to_value(&self) -> Value {
        let text = self.generated_text.as_bytes();
        let http_response = self
            .http_response_time
            .map(|time| (time - self.start_time) * 1000.0);
        let dispatch_wait = self
            .scheduled_time
            .map(|time| (self.start_time - time) * 1000.0);
        let ttft = (self.token_timing_available && self.ttft != 0.0).then_some(self.ttft * 1000.0);
        let tpot = (self.token_timing_available && self.output_len > 1)
            .then(|| (self.latency - self.ttft) / (self.output_len - 1) as f64 * 1000.0);
        json!({
            "request_id": self.request_id,
            "task": self.task,
            "success": self.success,
            "classifier": self.classifier,
            "error": self.error,
            "warnings": self.warnings,
            "endpoint": self.endpoint,
            "base_url": self.base_url,
            "scheduled_time": self.scheduled_time,
            "client_send_time": self.start_time,
            "http_response_time": self.http_response_time,
            "first_text_time": self.first_text_time,
            "first_image_done_time": self.first_image_done_time,
            "final_event_time": self.final_event_time,
            "client_dispatch_wait_ms": dispatch_wait,
            "http_response_ms": http_response,
            "e2e_ms": self.latency * 1000.0,
            "token_timing_available": self.token_timing_available,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "itl_count": self.itl.len(),
            "text_times": self.text_times,
            "chunk_times": self.chunk_times,
            "token_ids": self.tokens.iter().map(|token| token.id.clone()).collect::<Vec<_>>(),
            "token_times": self.tokens.iter().map(|token| token.time).collect::<Vec<_>>(),
            "top_logprobs": self.tokens.iter().map(|token| {
                token.candidates.iter().map(|(id, logprob)| json!([id, logprob])).collect::<Vec<_>>()
            }).collect::<Vec<_>>(),
            "prompt_len": self.prompt_len,
            "output_len": self.output_len,
            "requested_output_len": self.requested_output_len,
            "prompt_len_source": self.prompt_len_source.unwrap_or(Source::RequestFallback),
            "output_len_source": self.output_len_source.unwrap_or(Source::RequestedFallback),
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "cached_prompt_tokens_source": if self.cached_prompt_tokens.is_some() {
                "openai_usage_prompt_tokens_details"
            } else {
                "unavailable"
            },
            "generated_text": self.generated_text,
            "generated_text_bytes": text.len(),
            "generated_text_sha256": (!text.is_empty()).then(|| hex(&Sha256::digest(text))),
            "images": self.images,
            "image_outputs": self.decoded_images.iter().map(DecodedImage::metadata).collect::<Vec<_>>(),
            "first_image_latency_ms": self.first_image_latency.map(|latency| latency * 1000.0),
            "image_latencies_ms": self.image_latencies.iter().map(|latency| latency * 1000.0).collect::<Vec<_>>(),
            "image_steps": self.image_steps,
            "video_output": self.decoded_video.as_ref().map(DecodedVideo::metadata),
            "video_shape": self.video_shape,
            "served_shape": self.served_shape,
            "server_inference_s": self.server_inference_s,
            "server_stage_s": self.server_stage_s,
            "server_peak_storage_mib": self.server_peak_storage_mib,
            "media_checks": self.media_checks,
            "original_output": self.original_output,
            "example": self.example,
            "real_time_factor": self.decoded_video.as_ref().map(|video| self.latency / video.video_duration_s()),
            "status_code": self.status_code,
            "finish_reason": self.finish_reason,
            "stop_reason": self.stop_reason,
        })
    }
}

/// A logprob entry's token as its id when it names one: `token_id:<id>` is the
/// placeholder vLLM and UniServe render for `return_tokens_as_token_ids`; any
/// other token keeps its text.
fn token_identity(token: Option<&Value>) -> Value {
    let text = match token {
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
        None => "None".to_owned(),
    };
    match text.strip_prefix("token_id:") {
        Some(digits) if !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()) => {
            digits
                .parse::<u64>()
                .map(Value::from)
                .unwrap_or(Value::String(text.clone()))
        }
        _ => Value::String(text),
    }
}

/// A non-negative integer token count (never a boolean).
fn count(value: Option<&Value>) -> Option<u64> {
    value.and_then(Value::as_u64)
}

/// Named observable checks, statistics and warnings.
#[derive(Debug, Clone, Default)]
pub struct Validation {
    pub checks: BTreeMap<String, bool>,
    pub statistics: Map<String, Value>,
    pub warnings: Vec<String>,
}

impl Validation {
    /// A validation of `checks` alone.
    pub fn of(checks: impl IntoIterator<Item = (&'static str, bool)>) -> Self {
        Self {
            checks: checks
                .into_iter()
                .map(|(name, ok)| (name.to_owned(), ok))
                .collect(),
            ..Self::default()
        }
    }

    /// Whether at least one check exists and every check passes.
    pub fn valid(&self) -> bool {
        !self.checks.is_empty() && self.checks.values().all(|ok| *ok)
    }

    /// Combines two validations whose check names do not overlap; `other`'s
    /// statistics win on a repeated key.
    ///
    /// # Errors
    ///
    /// When the two share a check name.
    pub fn merged(mut self, other: Self) -> anyhow::Result<Self> {
        let overlap: Vec<&String> = other
            .checks
            .keys()
            .filter(|name| self.checks.contains_key(*name))
            .collect();
        anyhow::ensure!(
            overlap.is_empty(),
            "duplicate validation checks: {}",
            overlap
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        self.checks.extend(other.checks);
        self.statistics.extend(other.statistics);
        self.warnings.extend(other.warnings);
        Ok(self)
    }

    pub fn to_value(&self) -> Value {
        json!({
            "valid": self.valid(),
            "checks": self.checks,
            "statistics": self.statistics,
            "warnings": self.warnings,
        })
    }
}
