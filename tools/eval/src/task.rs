//! The task kinds the public APIs define: what a point requests for each row,
//! and how its records are judged.
//!
//! A task maps a row ([`Example`]) to an HTTP request ([`TaskRequest`]) and
//! validates the run's records against the task's observable contract. The
//! profile loader consults a task's rules (its endpoints, its streaming default,
//! whether it takes image settings, a question or an image count, what a video
//! point must state) before a point exists.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::record::{Example, Record, ReferenceMedia, TaskRequest, Validation, VideoShape};
use crate::spec::{
    CHAT_COMPLETIONS, IMAGES_GENERATIONS, Image, Point, VIDEOS_SYNC, Video, VideoBackend,
    parse_size,
};

/// What a point requests and how it validates it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskName {
    /// A streamed chat completion.
    Text,
    /// A streamed chat completion scored by its extracted final answer.
    Gsm8k,
    /// An image generated from text, by chat or the images API.
    T2i,
    /// An image edited from a source image.
    I2i,
    /// A text completion about an input image.
    I2t,
    /// A streamed answer that interleaves text and images.
    Interleave,
    /// A video with its soundtrack, from UniServe's `/v1/videos` body.
    Video,
}

impl FromStr for TaskName {
    type Err = anyhow::Error;

    fn from_str(name: &str) -> Result<Self> {
        Ok(match name {
            "text" => Self::Text,
            "gsm8k" => Self::Gsm8k,
            "t2i" => Self::T2i,
            "i2i" => Self::I2i,
            "i2t" => Self::I2t,
            "interleave" => Self::Interleave,
            "video" => Self::Video,
            _ => {
                bail!("unknown task {name:?}; known: gsm8k, i2i, i2t, interleave, t2i, text, video")
            }
        })
    }
}

/// Whether a task takes an explicit output image count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImageCount {
    Required,
    Forbidden,
    Optional,
}

/// The point's fields a video point states: the output shape and soundtrack
/// every request is held to; the frame count may come from each row instead.
const STATED_VIDEO_FIELDS: [&str; 4] = ["size", "fps", "audio_sample_rate", "audio_channels"];

impl TaskName {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Gsm8k => "gsm8k",
            Self::T2i => "t2i",
            Self::I2i => "i2i",
            Self::I2t => "i2t",
            Self::Interleave => "interleave",
            Self::Video => "video",
        }
    }

    fn endpoints(self) -> &'static [&'static str] {
        match self {
            Self::T2i => &[CHAT_COMPLETIONS, IMAGES_GENERATIONS],
            Self::Video => &[VIDEOS_SYNC, "/v1/videos"],
            _ => &[CHAT_COMPLETIONS],
        }
    }

    fn default_endpoint(self) -> &'static str {
        match self {
            Self::Video => VIDEOS_SYNC,
            _ => CHAT_COMPLETIONS,
        }
    }

    /// `sampling.stream` when a point states none.
    pub fn default_stream(self) -> bool {
        matches!(
            self,
            Self::Text | Self::Gsm8k | Self::I2t | Self::Interleave
        )
    }

    /// Whether a point may state an `image` settings table. The i2t task's
    /// input image comes from the row, and its output is text.
    fn accepts_image(self) -> bool {
        matches!(self, Self::T2i | Self::I2i | Self::Interleave)
    }

    fn accepts_question(self) -> bool {
        self == Self::I2t
    }

    fn image_count(self) -> ImageCount {
        match self {
            Self::T2i => ImageCount::Required,
            Self::I2i => ImageCount::Optional,
            _ => ImageCount::Forbidden,
        }
    }

    /// The endpoint a point posts to: the one it names, which the task must
    /// accept, else the task's default.
    pub(crate) fn check_endpoint(self, endpoint: Option<String>, context: &str) -> Result<String> {
        let chosen = endpoint.unwrap_or_else(|| self.default_endpoint().to_owned());
        ensure!(
            self.endpoints().contains(&chosen.as_str()),
            "{context}.endpoint must be one of: {}",
            self.endpoints().join(", ")
        );
        Ok(chosen)
    }

    pub(crate) fn check_question(
        self,
        question: Option<String>,
        context: &str,
    ) -> Result<Option<String>> {
        let Some(question) = question else {
            return Ok(None);
        };
        ensure!(
            self.accepts_question(),
            "{context}.question is not valid for task {}",
            self.as_str()
        );
        ensure!(
            !question.is_empty(),
            "{context}.question must be a non-empty string"
        );
        Ok(Some(question))
    }

    pub(crate) fn check_image(self, image: &Image, context: &str) -> Result<()> {
        ensure!(
            !image.is_stated() || self.accepts_image(),
            "{context}.image is not valid for this task"
        );
        match self.image_count() {
            ImageCount::Required => {
                ensure!(
                    image.image_count.is_some(),
                    "{context} requires image.image_count"
                );
            }
            ImageCount::Forbidden => ensure!(
                image.image_count.is_none(),
                "{context} does not declare a per-request image count"
            ),
            ImageCount::Optional => {}
        }
        Ok(())
    }

    /// A video point states the output shape and soundtrack it measures; other
    /// tasks build no video request and take any settings.
    pub(crate) fn check_video(self, video: &Video, context: &str) -> Result<()> {
        if self != Self::Video {
            return Ok(());
        }
        let stated = [
            video.size.is_some(),
            video.fps.is_some(),
            video.audio_sample_rate.is_some(),
            video.audio_channels.is_some(),
        ];
        let missing: Vec<&str> = STATED_VIDEO_FIELDS
            .iter()
            .zip(stated)
            .filter(|(_, stated)| !stated)
            .map(|(name, _)| *name)
            .collect();
        ensure!(
            missing.is_empty(),
            "{context}: a video point states its output shape and soundtrack and requires {missing:?}"
        );
        Ok(())
    }
}

/// The system message every measured GSM8K request carries.
pub const GSM8K_SYSTEM_PROMPT: &str = "Solve the problem step by step. End your answer with the final answer on its own line in the form `#### <number>`.";

/// Media types of the reference files the video request API accepts, by
/// suffix.
pub const MEDIA_TYPES: &[(&str, &str)] = &[
    (".jpg", "image/jpeg"),
    (".jpeg", "image/jpeg"),
    (".png", "image/png"),
    (".webp", "image/webp"),
    (".mp4", "video/mp4"),
    (".mov", "video/quicktime"),
    (".mp3", "audio/mpeg"),
    (".wav", "audio/wav"),
];

/// A task bound to the point it serves.
pub struct Task<'a> {
    pub point: &'a Point,
    reference_root: Option<PathBuf>,
}

impl<'a> Task<'a> {
    pub fn new(point: &'a Point) -> Self {
        let reference_root = point
            .video
            .reference_root
            .as_deref()
            .filter(|root| !root.is_empty())
            .map(|root| {
                Path::new(root)
                    .canonicalize()
                    .unwrap_or_else(|_| PathBuf::from(root))
            });
        Self {
            point,
            reference_root,
        }
    }

    pub fn name(&self) -> TaskName {
        self.point.task
    }

    /// The request of one row.
    ///
    /// # Errors
    ///
    /// When the row lacks what the task needs: an input image, a frame count,
    /// references that fit the video request API, or a video request its
    /// backend cannot take as the same work.
    pub fn build_request(&self, example: &Example) -> Result<TaskRequest> {
        let point = self.point;
        match point.task {
            TaskName::Text => {
                // A row's non-empty messages replace its prompt as the
                // conversation; the limit is the row's output_len, else the
                // point's max_tokens, else none.
                let limit = example.output_len.or(point.sampling.max_tokens);
                let mut payload = self.chat_payload(example, true);
                if let Some(limit) = limit {
                    payload.insert("max_completion_tokens".into(), limit.into());
                }
                self.apply_sampling(&mut payload);
                Ok(TaskRequest::new(&point.endpoint, payload, true))
            }
            TaskName::Gsm8k => {
                let limit = example.max_tokens.or(point.sampling.max_tokens);
                let mut payload = Map::new();
                payload.insert("model".into(), point.model.clone().into());
                payload.insert(
                    "messages".into(),
                    json!([
                        {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
                        {"role": "user", "content": example.prompt},
                    ]),
                );
                payload.insert("stream".into(), true.into());
                payload.insert("stream_options".into(), json!({"include_usage": true}));
                if let Some(limit) = limit {
                    payload.insert("max_completion_tokens".into(), limit.into());
                }
                self.apply_sampling(&mut payload);
                Ok(TaskRequest::new(&point.endpoint, payload, true))
            }
            TaskName::T2i => {
                // The images endpoint takes no text-sampling fields: the image
                // settings map onto its schema and extra_body is merged last.
                // It generates one image per request.
                if point.endpoint == IMAGES_GENERATIONS {
                    let mut payload = Map::new();
                    payload.insert("model".into(), point.model.clone().into());
                    payload.insert("prompt".into(), example.prompt.clone().into());
                    self.apply_image_generations(&mut payload, example);
                    payload.extend(point.sampling.extra_body.clone());
                    return Ok(TaskRequest::new(&point.endpoint, payload, false));
                }
                let mut payload = Map::new();
                payload.insert("model".into(), point.model.clone().into());
                payload.insert("modalities".into(), json!(["image"]));
                payload.insert(
                    "messages".into(),
                    json!([{"role": "user", "content": example.prompt}]),
                );
                payload.insert(
                    "image_config".into(),
                    Value::Object(self.image_fields(example, true)),
                );
                self.apply_sampling(&mut payload);
                Ok(TaskRequest::new(&point.endpoint, payload, false))
            }
            TaskName::I2i => {
                let mut payload = Map::new();
                payload.insert("model".into(), point.model.clone().into());
                payload.insert("modalities".into(), json!(["image"]));
                payload.insert("messages".into(), self.image_message(example)?);
                payload.insert(
                    "image_config".into(),
                    Value::Object(self.image_fields(example, point.image.image_count.is_some())),
                );
                self.apply_sampling(&mut payload);
                Ok(TaskRequest::new(&point.endpoint, payload, false))
            }
            TaskName::I2t => {
                // This task honours sampling.stream; a stream carries the usage
                // the validation requires only with include_usage.
                let limit = example
                    .max_tokens
                    .or(point.sampling.max_tokens)
                    .unwrap_or(512);
                let mut payload = Map::new();
                payload.insert("model".into(), point.model.clone().into());
                payload.insert("messages".into(), self.image_message(example)?);
                payload.insert("modalities".into(), json!(["text"]));
                payload.insert("max_completion_tokens".into(), limit.into());
                self.apply_sampling(&mut payload);
                let stream = point.sampling.stream;
                if stream {
                    payload.insert("stream".into(), true.into());
                    payload.insert("stream_options".into(), json!({"include_usage": true}));
                }
                Ok(TaskRequest::new(&point.endpoint, payload, stream))
            }
            TaskName::Interleave => {
                let limit = example
                    .max_tokens
                    .or(point.sampling.max_tokens)
                    .unwrap_or(512);
                let mut payload = Map::new();
                payload.insert("model".into(), point.model.clone().into());
                payload.insert("stream".into(), true.into());
                payload.insert("stream_options".into(), json!({"include_usage": true}));
                payload.insert("modalities".into(), json!(["text", "image"]));
                payload.insert(
                    "messages".into(),
                    json!([{"role": "user", "content": example.prompt}]),
                );
                payload.insert("max_completion_tokens".into(), limit.into());
                payload.insert(
                    "image_config".into(),
                    Value::Object(self.image_fields(example, false)),
                );
                self.apply_sampling(&mut payload);
                Ok(TaskRequest::new(&point.endpoint, payload, true))
            }
            TaskName::Video => self.video_request(example),
        }
    }

    /// The chat body of a text row: its messages, else its prompt as one user
    /// message.
    fn chat_payload(&self, example: &Example, stream: bool) -> Map<String, Value> {
        let messages = match &example.messages {
            Some(messages) if !messages.is_empty() => Value::Array(messages.clone()),
            _ => json!([{"role": "user", "content": example.prompt}]),
        };
        let mut payload = Map::new();
        payload.insert("model".into(), self.point.model.clone().into());
        payload.insert("messages".into(), messages);
        payload.insert("stream".into(), stream.into());
        payload.insert("stream_options".into(), json!({"include_usage": true}));
        payload
    }

    /// One user message of the row's prompt and its input image.
    fn image_message(&self, example: &Example) -> Result<Value> {
        Ok(json!([{
            "role": "user",
            "content": [
                {"type": "text", "text": example.prompt},
                {"type": "image_url", "image_url": {"url": input_image_url(example)?}},
            ],
        }]))
    }

    /// The point's sampling fields, each only when set, and `extra_body` merged
    /// last so its keys override every field set before.
    fn apply_sampling(&self, payload: &mut Map<String, Value>) {
        let sampling = &self.point.sampling;
        let fields: [(&str, Option<Value>); 8] = [
            ("temperature", sampling.temperature.map(Value::from)),
            ("top_p", sampling.top_p.map(Value::from)),
            ("ignore_eos", sampling.ignore_eos.map(Value::from)),
            ("top_k", sampling.top_k.map(Value::from)),
            ("min_p", sampling.min_p.map(Value::from)),
            (
                "repetition_penalty",
                sampling.repetition_penalty.map(Value::from),
            ),
            (
                "frequency_penalty",
                sampling.frequency_penalty.map(Value::from),
            ),
            (
                "presence_penalty",
                sampling.presence_penalty.map(Value::from),
            ),
        ];
        for (key, value) in fields {
            if let Some(value) = value {
                payload.insert(key.into(), value);
            }
        }
        if let Some(seed) = sampling.sampling_seed {
            payload.insert("seed".into(), seed.into());
        }
        payload.extend(sampling.extra_body.clone());
    }

    /// The image settings in the chat-completions `image_config` layout: the
    /// row's width, height and steps over the point's, the row's seed over the
    /// load's, width and height only as a pair, `num_images` when asked and
    /// stated.
    fn image_fields(&self, example: &Example, include_count: bool) -> Map<String, Value> {
        let image = &self.point.image;
        let width = example.width.or(image.width);
        let height = example.height.or(image.height);
        let steps = example.steps.or(image.steps);
        let seed = example.seed.unwrap_or(self.point.load.seed as i64);
        let mut payload = Map::new();
        payload.insert("seed".into(), seed.into());
        if include_count && let Some(count) = image.image_count {
            payload.insert("num_images".into(), count.into());
        }
        if let (Some(width), Some(height)) = (width, height) {
            payload.insert("width".into(), width.into());
            payload.insert("height".into(), height.into());
        }
        if let Some(steps) = steps {
            payload.insert("steps".into(), steps.into());
        }
        if let Some(scale) = image.guidance_scale {
            payload.insert("guidance_scale".into(), scale.into());
        }
        if let Some(scale) = image.image_guidance_scale {
            payload.insert("image_guidance_scale".into(), scale.into());
        }
        if let Some(norm) = &image.cfg_norm {
            payload.insert("cfg_norm".into(), norm.clone().into());
        }
        if let Some(shift) = image.timestep_shift {
            payload.insert("timestep_shift".into(), shift.into());
        }
        if let Some(interval) = image.cfg_interval {
            payload.insert("cfg_interval".into(), json!(interval));
        }
        if let Some(ratio) = &example.aspect_ratio {
            payload.insert("resolution".into(), ratio.clone().into());
        }
        payload
    }

    /// The image settings on the images API's schema: width and height as one
    /// `size`, no count and no `resolution`.
    fn apply_image_generations(&self, payload: &mut Map<String, Value>, example: &Example) {
        let rendered = self.image_fields(example, false);
        if let (Some(width), Some(height)) = (rendered.get("width"), rendered.get("height")) {
            payload.insert("size".into(), format!("{width}x{height}").into());
        }
        for key in [
            "steps",
            "seed",
            "guidance_scale",
            "image_guidance_scale",
            "cfg_norm",
            "timestep_shift",
            "cfg_interval",
        ] {
            if let Some(value) = rendered.get(key) {
                payload.insert(key.into(), value.clone());
            }
        }
    }

    /// One video request: the point's size and frame rate, the row's frame
    /// count or the point's, the row's seed or the load's, and the row's
    /// references, whose media are read here, before measurement.
    fn video_request(&self, example: &Example) -> Result<TaskRequest> {
        let point = self.point;
        let video = &point.video;
        TaskName::Video.check_video(video, &format!("benchmark {} video", point.name))?;
        let frames = example
            .num_frames
            .or(video.num_frames)
            .filter(|frames| *frames >= 1)
            .with_context(|| {
                format!(
                    "row {}: num_frames must be a positive integer, stated by the row or by the point's \
                     video.num_frames",
                    example.id
                )
            })?;
        let (width, height) = parse_size(video.size.as_deref().unwrap_or_default())?;
        let shape = VideoShape {
            width,
            height,
            frames,
            fps: video.fps.unwrap_or_default(),
        };
        let (references, media) = self.references(example)?;
        let mut payload = Map::new();
        payload.insert("model".into(), point.model.clone().into());
        payload.insert("prompt".into(), example.prompt.clone().into());
        payload.insert("size".into(), shape.size().into());
        payload.insert("num_frames".into(), shape.frames.into());
        payload.insert("fps".into(), shape.fps.into());
        payload.insert(
            "seed".into(),
            example.seed.unwrap_or(point.load.seed as i64).into(),
        );
        if !references.is_empty() {
            payload.insert("input_references".into(), Value::Array(references));
        }
        let request = TaskRequest {
            endpoint: point.endpoint.clone(),
            payload,
            stream: false,
            video: Some(video.clone()),
            video_shape: Some(shape),
            reference_media: media,
        };
        // A request its backend cannot take as the same work fails here, before
        // measurement, rather than as a measured transport failure.
        crate::transport::video::native_request(&request)?;
        Ok(request)
    }

    /// A row's references and their media in row order, each naming its file by
    /// an absolute `file://` URL. A row is checked only as far as the request
    /// API's reference fields and the backends' upload forms depend on it; the
    /// server enforces the model's own rules on which references it takes.
    fn references(&self, example: &Example) -> Result<(Vec<Value>, Vec<ReferenceMedia>)> {
        let rows = example.input_references.as_deref().unwrap_or_default();
        if rows.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let root = self.reference_root.as_ref().with_context(|| {
            format!(
                "row {}: input_references need the point's video.reference_root",
                example.id
            )
        })?;
        let mut references = Vec::new();
        let mut media = Vec::new();
        for (index, row) in rows.iter().enumerate() {
            let context = format!("row {} input_references[{index}]", example.id);
            let mut unknown: Vec<&str> = row
                .keys()
                .map(String::as_str)
                .filter(|key| !["type", "media", "frame_index", "start_time_seconds"].contains(key))
                .collect();
            unknown.sort_unstable();
            ensure!(unknown.is_empty(), "{context}: unknown keys {unknown:?}");
            let kind = row.get("type").and_then(Value::as_str).unwrap_or_default();
            let family = match kind {
                "image" => "image/",
                "video" => "video/",
                "audio" => "audio/",
                _ => bail!("{context}: unknown type {kind:?}"),
            };
            if let Some(anchor) = row.get("frame_index") {
                ensure!(anchor.is_i64(), "{context}: frame_index must be an integer");
            }
            let relative = row.get("media").and_then(Value::as_str).unwrap_or_default();
            let path = root.join(relative);
            let path = path
                .canonicalize()
                .with_context(|| format!("{context}: {}", path.display()))?;
            let suffix = path
                .extension()
                .map(|extension| format!(".{}", extension.to_string_lossy().to_lowercase()))
                .unwrap_or_default();
            let mime = MEDIA_TYPES
                .iter()
                .find(|(known, _)| *known == suffix)
                .map(|(_, mime)| *mime)
                .filter(|mime| mime.starts_with(family))
                .with_context(|| {
                    format!(
                        "{context}: {} is not {kind} media",
                        path.file_name()
                            .map(|name| name.to_string_lossy())
                            .unwrap_or_default()
                    )
                })?;
            let mut reference = Map::new();
            reference.insert("type".into(), kind.into());
            reference.insert("url".into(), format!("file://{}", path.display()).into());
            for name in ["frame_index", "start_time_seconds"] {
                if let Some(value) = row.get(name).filter(|value| !value.is_null()) {
                    reference.insert(name.into(), value.clone());
                }
            }
            references.push(Value::Object(reference));
            media.push(ReferenceMedia {
                path: path.display().to_string(),
                mime: mime.to_owned(),
                data: std::fs::read(&path)
                    .with_context(|| format!("{context}: reading {}", path.display()))?,
            });
        }
        Ok((references, media))
    }

    /// The request-level checks every task makes beside its own.
    ///
    /// # Errors
    ///
    /// When the task's own checks name `all_requests_succeeded`.
    pub fn validate(&self, records: &[&Record]) -> Result<Validation> {
        let mut common = Validation::of([(
            "all_requests_succeeded",
            !records.is_empty() && records.iter().all(|record| record.success),
        )]);
        common
            .statistics
            .insert("request_count".into(), records.len().into());
        common.statistics.insert(
            "successful_requests".into(),
            records
                .iter()
                .filter(|record| record.success)
                .count()
                .into(),
        );
        common.merged(self.validate_output(records))
    }

    /// The checks the task makes of its outputs.
    pub fn validate_output(&self, records: &[&Record]) -> Validation {
        let point = self.point;
        let any = !records.is_empty();
        match point.task {
            TaskName::Text | TaskName::I2t => {
                let mut validation = Validation::of([("server_usage", server_usage(records))]);
                if point.sampling.ignore_eos == Some(true) {
                    validation
                        .checks
                        .insert("fixed_output_length".into(), fixed_length(records));
                }
                validation
            }
            TaskName::Gsm8k => {
                let score = crate::score::score(records);
                let mut validation = Validation::of([(
                    "gold_answers",
                    any && records
                        .iter()
                        .all(|record| crate::score::reference_answer(record).is_some()),
                )]);
                for (key, value) in [
                    ("correct", score.correct),
                    ("total", score.total),
                    ("unparseable_predictions", score.unparseable),
                    ("decided_by_marker", score.by_marker),
                    ("decided_by_boxed", score.by_boxed),
                    ("without_answer_form", score.without_form),
                ] {
                    validation.statistics.insert(key.into(), value.into());
                }
                validation
            }
            TaskName::T2i => {
                let count = point.image.image_count.unwrap_or_default() as usize;
                let mut validation = Validation::of(self.image_checks(records));
                validation.checks.insert(
                    "exact_image_count".into(),
                    any && records.iter().all(|record| record.images == count),
                );
                let total: usize = records.iter().map(|record| record.images).sum();
                validation
                    .statistics
                    .insert("completed_images".into(), total.into());
                validation.statistics.insert(
                    "images_per_request".into(),
                    mean(total, records.len()).into(),
                );
                validation
            }
            TaskName::I2i => {
                let mut validation = Validation::of(self.image_checks(records));
                validation.checks.insert(
                    "image_output".into(),
                    any && records.iter().all(|record| record.images > 0),
                );
                if let Some(count) = point.image.image_count {
                    validation.checks.insert(
                        "exact_image_count".into(),
                        records.iter().all(|record| record.images == count as usize),
                    );
                }
                validation
            }
            TaskName::Interleave => {
                // How many images an answer holds is the model's choice: the run
                // is valid once interleaving was exercised, some request having
                // produced an image.
                let mut validation = Validation::of(self.image_checks(records));
                validation.checks.insert(
                    "visible_text".into(),
                    any && records
                        .iter()
                        .all(|record| !record.generated_text.is_empty()),
                );
                validation
                    .checks
                    .insert("server_usage".into(), server_usage(records));
                let total: usize = records.iter().map(|record| record.images).sum();
                validation
                    .checks
                    .insert("produced_images".into(), total > 0);
                validation
                    .statistics
                    .insert("completed_images".into(), total.into());
                validation.statistics.insert(
                    "images_per_request".into(),
                    mean(total, records.len()).into(),
                );
                validation.statistics.insert(
                    "zero_image_requests".into(),
                    records
                        .iter()
                        .filter(|record| record.images == 0)
                        .count()
                        .into(),
                );
                validation
            }
            TaskName::Video => self.video_checks(records),
        }
    }

    /// The task's own metrics for the summary, which replace generic ones of the
    /// same name: GSM8K's accuracy; none for the others.
    pub fn summary_metrics(&self, records: &[&Record]) -> Map<String, Value> {
        let mut metrics = Map::new();
        if self.point.task == TaskName::Gsm8k {
            let score = crate::score::score(records);
            metrics.insert("accuracy".into(), score.accuracy().into());
            metrics.insert("correct".into(), score.correct.into());
            metrics.insert("total".into(), score.total.into());
        }
        metrics
    }

    /// Decoded image sizes against the point's stated width and height (per-row
    /// overrides are not considered; an unset dimension is not checked).
    fn image_checks(&self, records: &[&Record]) -> [(&'static str, bool); 1] {
        let image = &self.point.image;
        let dimensions = records
            .iter()
            .flat_map(|record| &record.decoded_images)
            .all(|decoded| {
                image.width.is_none_or(|width| decoded.width == width)
                    && image.height.is_none_or(|height| decoded.height == height)
            });
        [("image_dimensions", dimensions)]
    }

    /// Decodes and classifies a video response, outside the load window.
    pub fn inspect_output(&self, record: &mut Record) {
        if !record.success {
            return;
        }
        match crate::media::inspect_video(
            record.video_body.as_deref().unwrap_or_default(),
            &record.video_mime,
        ) {
            Ok(decoded) => record.decoded_video = Some(decoded),
            Err(error) => record.mark_failure(error.classifier, Some(error.message)),
        }
        let checks = self.video_checks(&[&*record]).checks;
        let failed: Vec<&str> = checks
            .iter()
            .filter(|(_, ok)| !**ok)
            .map(|(name, _)| name.as_str())
            .collect();
        if record.success && !failed.is_empty() {
            let detail = failed.join(", ");
            record.mark_failure("invalid_video_output", Some(detail));
        }
        record.media_checks = checks;
    }

    /// Each MP4 against its own request's stated shape and the point's stated
    /// soundtrack; a UniServe output also against the shape the server reported
    /// generating. Both streams are measured against the stated length rather
    /// than each other: the video may miss one frame and the audio one frame
    /// period, so a one-frame-short video with full-length audio is valid, and a
    /// missing or truncated stream is not.
    fn video_checks(&self, records: &[&Record]) -> Validation {
        let video = &self.point.video;
        let present = !records.is_empty()
            && records
                .iter()
                .all(|record| record.decoded_video.is_some() && record.video_shape.is_some());
        let pairs: Vec<(&crate::record::DecodedVideo, VideoShape)> = records
            .iter()
            .filter_map(|record| Some((record.decoded_video.as_ref()?, record.video_shape?)))
            .collect();
        let all = |test: &dyn Fn(&crate::record::DecodedVideo, VideoShape) -> bool| {
            present && pairs.iter().all(|(decoded, shape)| test(decoded, *shape))
        };
        let mut validation = Validation::of([
            ("decoded_video", present),
            (
                "h264_video",
                all(&|decoded, _| decoded.video_codec == "h264"),
            ),
            (
                "stated_size",
                all(&|decoded, shape| {
                    (decoded.width, decoded.height) == (shape.width, shape.height)
                }),
            ),
            // The container's rate is a rational: exactly the stated one.
            (
                "stated_fps",
                all(&|decoded, shape| {
                    decoded.fps_numerator == u64::from(shape.fps) * decoded.fps_denominator
                }),
            ),
            (
                "stated_frame_count",
                all(&|decoded, shape| decoded.frame_count.abs_diff(u64::from(shape.frames)) <= 1),
            ),
            (
                "stated_aac_audio",
                all(&|decoded, _| {
                    decoded.audio_codec == "aac"
                        && Some(decoded.audio_channels) == video.audio_channels
                        && Some(decoded.audio_sample_rate) == video.audio_sample_rate
                }),
            ),
            (
                "stated_audio_duration",
                all(&|decoded, shape| {
                    (decoded.audio_duration_s() - shape.duration_s()).abs()
                        <= 1.0 / f64::from(shape.fps)
                }),
            ),
            (
                "nonzero_video_variance",
                all(&|decoded, _| {
                    decoded.video_variance.is_finite() && decoded.video_variance > 0.0
                }),
            ),
            (
                "nonzero_audio_rms",
                all(&|decoded, _| decoded.audio_rms.is_finite() && decoded.audio_rms > 0.0),
            ),
        ]);
        if video.backend == VideoBackend::Uniserve {
            validation.checks.insert(
                "served_shape".into(),
                present
                    && records.iter().all(|record| {
                        record.served_shape.is_some() && record.served_shape == record.video_shape
                    }),
            );
        }
        validation
            .statistics
            .insert("completed_videos".into(), pairs.len().into());
        validation.statistics.insert(
            "media_bytes".into(),
            pairs
                .iter()
                .map(|(decoded, _)| decoded.data.len())
                .sum::<usize>()
                .into(),
        );
        validation
    }
}

/// The embedded data URL of a row's input image; `image/png` when the row
/// states no type.
fn input_image_url(example: &Example) -> Result<String> {
    let data = example
        .input_image_b64
        .as_deref()
        .filter(|data| !data.is_empty())
        .context("input image row has no base64 payload")?;
    let mime = example.input_image_mime.as_deref().unwrap_or("image/png");
    ensure!(
        mime.starts_with("image/"),
        "input image row has an invalid MIME type"
    );
    Ok(format!("data:{mime};base64,{data}"))
}

/// Whether every request's prompt and output lengths came from the server's
/// usage; an empty run fails.
fn server_usage(records: &[&Record]) -> bool {
    use crate::record::Source;
    !records.is_empty()
        && records.iter().all(|record| {
            record.output_len_source == Some(Source::ServerUsage)
                && record.prompt_len_source == Some(Source::ServerUsage)
        })
}

/// Whether every completion reached its requested limit and ended for that
/// reason; a point without a limit fails, as does an empty run.
fn fixed_length(records: &[&Record]) -> bool {
    !records.is_empty()
        && records.iter().all(|record| {
            record.requested_output_len > 0
                && record.output_len == record.requested_output_len
                && record.finish_reason.as_deref() == Some("length")
        })
}

fn mean(total: usize, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total as f64 / count as f64
    }
}
