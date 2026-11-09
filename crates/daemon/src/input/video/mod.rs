//! Video generation: the public request vocabulary and the video service of one deployment.
//!
//! The public request is the same for every video model ([`VideoGenerationRequest`]): a prompt,
//! an output size and length, and reference media, each anchoring a generated frame or
//! conditioning the generation as a whole. What a model makes of it is the model's: its
//! processor (`processing::VideoProcessor`), which the deployment's family bound from what its
//! workers described, holds the request to the model's rules and plans it.
//!
//! [`VideoService::process`] runs one request through five stages in order:
//!
//! 1. The processor's request-only rules ([`VideoProcessor::check`]), so a malformed request is
//!    refused before anything is fetched.
//! 2. The service's [`MediaConnector`] resolves each `input_references[i].url` under the server's
//!    media policy (`data:`, `file://` under the media directory, `http(s)://` with size, time and
//!    address limits) into an owned buffer, the media of one request sharing one total.
//! 3. [`probe::MediaProber`] reads the facts a plan needs: an image's displayed size from its
//!    header and EXIF orientation; a video's display aspect, frame rate, decoded frame count and
//!    soundtrack, and an audio file's sample rate and decoded sample count, through `ffprobe`.
//! 4. The processor plans the request from the probed media with the checkpoint's tokenizer
//!    ([`VideoProcessor::plan`]): its size and frames, how each reference is decoded and encoded,
//!    the prompt as the text encoder encodes it, and the denoiser rows the references take.
//! 5. Each reference's media is published to shared memory for the worker's media reader, and the
//!    request becomes one `DiffusionRequest` describing every condition and stating the rows they
//!    take, by which the engine finds the smallest graph that holds them.
//!
//! Stages 1 to 3 wait only on I/O and run on the async runtime; stages 4 and 5 are CPU work and
//! run on the blocking pool.
//!
//! Every refusal is a `processing::InputError::Invalid` naming the request parameter at fault,
//! `input_references[i]` for a reference; the API answers it as an `invalid_request_error`.
//! Failures of the server's own resources (a missing `ffprobe`, scratch storage) are
//! `processing::InputError::Internal`.

mod encodings;
pub mod probe;

use std::sync::Arc;

use processing::{InputError, Length, Reference, ReferenceKind, VideoProcessor, VideoSpec};
use proto::MediaSource;
use proto::request::{DiffusionRequest, DiffusionSamplingParams, RequestId, Trajectory};
use proto::video::{Canvas, Conditions, MediaLocator};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::input::ServeError;
use crate::input::media::{FetchedMedia, MediaConnector};
use crate::options::MediaConfig;
use crate::tokenizer::DynTokenizer;

use probe::{MediaProber, ProbeConfig};

/// One reference medium of a request, in request order.
///
/// The order is semantic: a model presents references in it.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VideoReference {
    /// `image`, `video` or `audio`.
    #[serde(rename = "type")]
    pub kind: String,
    /// `data:`, `file://` or `http(s)://` location of the medium.
    pub url: String,
    /// The generated frame the reference anchors, a negative index counted from the last; absent
    /// for a reference that conditions the generation as a whole.
    #[serde(default)]
    pub frame_index: Option<i64>,
    /// Where a video or audio reference starts, in seconds.
    #[serde(default)]
    pub start_time_seconds: Option<f64>,
}

/// A duration, stated as a number or as a numeric string.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Seconds {
    /// A number of seconds.
    Number(f64),
    /// A numeric string of seconds.
    Text(#[serde(deserialize_with = "numeric_text")] f64),
}

/// A numeric string's value.
fn numeric_text<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    let text = String::deserialize(deserializer)?;
    text.trim()
        .parse()
        .map_err(|_| serde::de::Error::custom(format!("{text:?} is not a number of seconds")))
}

impl Seconds {
    fn value(self) -> f64 {
        match self {
            Self::Number(value) | Self::Text(value) => value,
        }
    }
}

/// A video generation request, as `/v1/videos` and `/v1/videos/sync` take it.
///
/// `deny_unknown_fields` refuses every field not listed here, at every level; a model refuses
/// what it does not take, and states its own options in `extra_params`.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VideoGenerationRequest {
    /// Served model name.
    pub model: String,
    /// The prompt.
    pub prompt: String,
    /// What the generation steers away from.
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// The output raster, `WxH` in pixels.
    #[serde(default)]
    pub size: Option<String>,
    /// The requested duration; the model aligns it to a frame count it generates.
    #[serde(default)]
    pub seconds: Option<Seconds>,
    /// The exact frame count; one the model generates.
    #[serde(default)]
    pub num_frames: Option<u32>,
    /// The output's frame rate.
    #[serde(default)]
    pub fps: Option<u32>,
    /// The noise seed.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Denoiser evaluations of the schedule.
    #[serde(default)]
    pub num_inference_steps: Option<u32>,
    /// Classifier-free guidance scale.
    #[serde(default)]
    pub guidance_scale: Option<f64>,
    /// Reference media, in request order.
    #[serde(default)]
    pub input_references: Vec<VideoReference>,
    /// Options the model defines.
    #[serde(default)]
    pub extra_params: Map<String, Value>,
}

/// The fields of the video request body, in schema order.
pub(crate) const REQUEST_FIELDS: [&str; 12] = [
    "model",
    "prompt",
    "negative_prompt",
    "size",
    "seconds",
    "num_frames",
    "fps",
    "seed",
    "num_inference_steps",
    "guidance_scale",
    "input_references",
    "extra_params",
];

impl VideoGenerationRequest {
    /// The request as a video model reads it.
    ///
    /// # Errors
    ///
    /// Refuses a size that is not `WxH` of positive pixels, a length stated twice, and a reference
    /// of an unknown kind.
    pub fn spec(&self) -> Result<VideoSpec, InputError> {
        let size = self.size.as_deref().map(parse_size).transpose()?;
        let length = match (self.seconds, self.num_frames) {
            (Some(_), Some(_)) => {
                return Err(InputError::invalid(
                    "num_frames",
                    "a request states its length once: seconds or num_frames",
                ));
            }
            (Some(seconds), None) => Some(Length::Seconds(seconds.value())),
            (None, Some(frames)) => Some(Length::Frames(frames)),
            (None, None) => None,
        };
        let references = self
            .input_references
            .iter()
            .enumerate()
            .map(|(index, reference)| {
                let kind = ReferenceKind::named(&reference.kind).ok_or_else(|| {
                    InputError::invalid(
                        format!("input_references[{index}]"),
                        format!("type {:?} is not image, video or audio", reference.kind),
                    )
                })?;
                Ok(Reference {
                    kind,
                    frame_index: reference.frame_index,
                    start_time_seconds: reference.start_time_seconds,
                })
            })
            .collect::<Result<Vec<_>, InputError>>()?;
        Ok(VideoSpec {
            prompt: self.prompt.clone(),
            negative_prompt: self.negative_prompt.clone(),
            size,
            length,
            fps: self.fps,
            seed: self.seed,
            num_inference_steps: self.num_inference_steps,
            guidance_scale: self.guidance_scale,
            references,
            extra_params: self.extra_params.clone(),
        })
    }
}

/// A `WxH` size.
fn parse_size(size: &str) -> Result<Canvas, InputError> {
    size.split_once('x')
        .and_then(|(width, height)| Some((width.parse().ok()?, height.parse().ok()?)))
        .filter(|(width, height)| *width > 0 && *height > 0)
        .map(|(width, height)| Canvas { width, height })
        .ok_or_else(|| {
            InputError::invalid(
                "size",
                format!("{size:?} is not WxH in pixels, such as 1344x768"),
            )
        })
}

/// What a processed video request generates, as its job record reports it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcessedVideo {
    /// The output raster.
    pub size: Canvas,
    /// The frames generated.
    pub num_frames: u32,
    /// The output's frame rate.
    pub fps: u32,
}

impl ProcessedVideo {
    /// The generated duration, in seconds.
    pub fn seconds(&self) -> f64 {
        f64::from(self.num_frames) / f64::from(self.fps.max(1))
    }
}

/// The reference media of one planned request, published to shared memory.
///
/// The workers read each by name while the request runs, so whoever submits the request holds
/// them until it ends; dropping them unlinks the names.
#[derive(Debug)]
pub struct PublishedMedia {
    _sources: Vec<MediaSource>,
}

/// Processes the video requests of one deployment.
///
/// One instance is shared by all requests; [`VideoService::process`] takes `&self` and runs
/// concurrently.
pub struct VideoService {
    /// The deployment's processor, which the blocking pool shares.
    processor: Arc<dyn VideoProcessor>,
    /// The media policy, which `capabilities` reports.
    media: MediaConfig,
    connector: MediaConnector,
    prober: MediaProber,
    tokenizer: DynTokenizer,
}

impl VideoService {
    /// The service of a deployment whose processor is `processor`, under the media policy
    /// `media`, tokenizing with the checkpoint's `tokenizer`.
    ///
    /// # Errors
    ///
    /// Fails when the media connector cannot be built.
    pub fn new(
        processor: Box<dyn VideoProcessor>,
        media: &MediaConfig,
        tokenizer: DynTokenizer,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            processor: Arc::from(processor),
            media: media.clone(),
            connector: MediaConnector::from_media_options(media)?,
            prober: MediaProber::new(ProbeConfig::new(media.ffprobe.clone())),
            tokenizer,
        })
    }

    /// The denoiser evaluations of every request.
    pub fn denoise_steps(&self) -> u32 {
        self.processor.denoise_steps()
    }

    /// Describes the served video contract for `GET /v1/capabilities`: the model's own part, the
    /// prompt capacity, the media sources and size caps, and the request fields.
    pub fn capabilities(&self, max_prompt_tokens: u32) -> Value {
        json!({
            "model": self.processor.capabilities(),
            "max_prompt_tokens": max_prompt_tokens,
            "media": {
                "data": true,
                "http": self.media.allow_remote,
                "file": self.media.directory.is_some(),
                "max_request_bytes": self.media.max_request_bytes,
            },
            "request_fields": REQUEST_FIELDS,
        })
    }

    /// Validates, fetches, probes and plans one video request into its engine request and what
    /// it generates.
    ///
    /// The request-only rules run before any fetch. The planned prompt must fit the prompt
    /// capacity (`--max-model-len`). Each reference's fetched media is then copied to shared
    /// memory, which the request holds until the engine retires it. `Leader::submit` stamps the
    /// engine's identifier on the request.
    ///
    /// # Errors
    ///
    /// Returns `ServeError::Video` naming the parameter at fault, or failing when processing's
    /// own resources or the shared-memory allocation fail, and
    /// `ServeError::ContextLengthExceeded` for a prompt longer than the prompt capacity.
    pub async fn process(
        &self,
        request_id: &str,
        request: &VideoGenerationRequest,
        max_prompt_tokens: u32,
    ) -> Result<(DiffusionRequest, ProcessedVideo, PublishedMedia), ServeError> {
        let spec = request.spec()?;
        self.processor.check(&spec)?;

        // Fetches and probes of different references run concurrently; the first failure ends
        // the request.
        let urls: Vec<&str> = request
            .input_references
            .iter()
            .map(|reference| reference.url.as_str())
            .collect();
        let media = self.connector.fetch_all(&urls).await.map_err(|error| {
            InputError::invalid(
                format!("input_references[{}]", error.index),
                error.source.to_string(),
            )
        })?;
        let probes =
            futures::future::try_join_all(media.iter().enumerate().map(|(index, media)| {
                self.prober
                    .probe(index, spec.references[index].kind, &media.bytes)
            }))
            .await?;

        let processor = Arc::clone(&self.processor);
        let tokenizer = Arc::clone(&self.tokenizer);
        let request_id = request_id.to_owned();
        tokio::task::spawn_blocking(move || {
            let plan = processor.plan(&spec, &probes, &*tokenizer)?;
            if plan.prompt_token_ids.len() > max_prompt_tokens as usize {
                return Err(ServeError::ContextLengthExceeded {
                    request_id,
                    prompt_tokens: plan.prompt_token_ids.len(),
                    max_tokens: max_prompt_tokens,
                });
            }
            submission(plan, &media)
        })
        .await
        .unwrap_or_else(|error| {
            Err(InputError::Internal(format!("planning the request failed: {error}")).into())
        })
    }
}

/// The engine request of `plan`, its references' media published to shared memory.
fn submission(
    plan: processing::VideoPlan,
    media: &[FetchedMedia],
) -> Result<(DiffusionRequest, ProcessedVideo, PublishedMedia), ServeError> {
    // Each reference's media reaches the worker's media reader through shared memory on this
    // host.
    let mut sources = Vec::with_capacity(media.len());
    let mut conditions = Vec::with_capacity(media.len());
    for (index, (condition, fetched)) in plan.conditions.into_iter().zip(media).enumerate() {
        let source = MediaSource::publish(&fetched.bytes).map_err(|error| {
            InputError::Internal(format!(
                "publishing input_references[{index}] for the worker failed: {error}"
            ))
        })?;
        let locator = source.locator();
        conditions.push(condition.at(MediaLocator {
            name: locator.name,
            bytes: locator.bytes,
        }));
        sources.push(source);
    }
    let conditions = Conditions {
        conditions,
        prompt_tokens: u32::try_from(plan.prompt_token_ids.len()).map_err(|_| {
            InputError::Internal("a prompt of more tokens than a u32 counts".into())
        })?,
        vision_spans: plan.vision_spans,
    };
    conditions
        .validate()
        .map_err(|reason| InputError::Internal(format!("the planned conditions: {reason}")))?;

    // What the request's encoders read, keyed by the content of each reference's media, so that
    // engines find the encoder outputs of requests that read the same.
    let contents: Vec<[u8; 32]> = media
        .iter()
        .map(|media| {
            let mut key = storage::key::Digest::new(b"media");
            key.bytes(&media.bytes);
            key.finish()
        })
        .collect();
    let encoder_inputs = encodings::encoder_inputs(&conditions, &plan.prompt_token_ids, &contents)
        .map_err(|error| {
            InputError::Internal(format!("digesting the conditions failed: {error}"))
        })?;

    let request = DiffusionRequest {
        request_id: RequestId::default(),
        prompt_token_ids: plan.prompt_token_ids,
        conditions: conditions
            .conditions
            .iter()
            .map(|condition| condition.extent())
            .collect(),
        conditioning: conditions.encode().map_err(|error| {
            InputError::Internal(format!("encoding the conditions failed: {error}"))
        })?,
        priority: 0,
        sampling: DiffusionSamplingParams {
            num_frames: plan.num_frames,
            video_units: plan.video_units,
            seed: plan.seed,
            width: plan.size.width,
            height: plan.size.height,
            condition_rows: plan.condition_rows,
        },
        // The migrator gives the request its trajectory's identity when it first submits it.
        trajectory: Trajectory::default(),
        encoder_inputs: Some(Box::new(encoder_inputs)),
    };
    let video = ProcessedVideo {
        size: plan.size,
        num_frames: plan.num_frames,
        fps: plan.fps,
    };
    Ok((request, video, PublishedMedia { _sources: sources }))
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;
    use image::ImageEncoder as _;
    use serde_json::Value;

    use super::{ServeError, VideoGenerationRequest};
    use crate::input::processor::pipeline_tokenizer;
    use crate::model::assets::PipelineCheckpoint;
    use crate::test_support::test_video_service;

    fn request(body: Value) -> VideoGenerationRequest {
        serde_json::from_value(body).unwrap()
    }

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        image::codecs::png::PngEncoder::new(&mut bytes)
            .write_image(
                &vec![128; (width * height * 3) as usize],
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        bytes
    }

    fn data_uri(bytes: &[u8]) -> String {
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        )
    }

    /// A size is `WxH` of positive pixels, a length is stated once, and a reference is an image,
    /// a video or an audio file.
    #[test]
    fn a_request_reads_as_its_spec() {
        let spec = request(serde_json::json!({
            "model": "m", "prompt": "a fox", "size": "1344x768", "seconds": "5",
            "input_references": [{"type": "image", "url": "data:,", "frame_index": -1}],
        }))
        .spec()
        .unwrap();
        assert_eq!(
            (spec.size.unwrap().width, spec.size.unwrap().height),
            (1344, 768)
        );
        assert_eq!(spec.length, Some(processing::Length::Seconds(5.0)));
        assert_eq!(spec.references[0].frame_index, Some(-1));

        let refused = |body: Value| match request(body).spec() {
            Err(processing::InputError::Invalid { param, .. }) => param,
            other => panic!("not refused: {other:?}"),
        };
        assert_eq!(
            refused(serde_json::json!({"model": "m", "prompt": "p", "size": "1344"})),
            "size"
        );
        assert_eq!(
            refused(
                serde_json::json!({"model": "m", "prompt": "p", "seconds": 5, "num_frames": 124})
            ),
            "num_frames"
        );
        assert_eq!(
            refused(serde_json::json!({
                "model": "m", "prompt": "p",
                "input_references": [{"type": "video_audio", "url": "data:,"}],
            })),
            "input_references[0]"
        );
    }

    /// The public request names no model's own vocabulary: a field it does not list is refused.
    #[test]
    fn a_field_outside_the_schema_is_refused() {
        let refused = serde_json::from_value::<VideoGenerationRequest>(serde_json::json!({
            "model": "m", "prompt": "p", "task": "t2va",
        }));
        assert!(refused.unwrap_err().to_string().contains("task"));
    }

    /// Encoder cache inputs follow the media and prompt, independently of the shared storage
    /// publication that carries them to a worker.
    #[tokio::test]
    async fn repeated_media_reuses_encoder_inputs_and_changed_media_does_not() {
        let service = test_video_service();
        let input = |prompt: &str, height: u32| {
            request(serde_json::json!({
                "model": "test",
                "prompt": prompt,
                "seconds": 5,
                "input_references": [{"type": "image", "url": data_uri(&png(160, height))}],
            }))
        };
        let mut inputs = Vec::new();
        for (prompt, height) in [("a fox", 90), ("a fox", 90), ("a bear", 90), ("a fox", 91)] {
            let (resolved, _, _media) = service
                .process("encoding", &input(prompt, height), 1 << 16)
                .await
                .unwrap();
            inputs.push(resolved.encoder_inputs.unwrap());
        }
        assert_eq!(inputs[0], inputs[1]);
        assert_ne!(inputs[0].text, inputs[2].text);
        assert_eq!(inputs[0].video_units, inputs[2].video_units);
        assert_ne!(inputs[0].video_units, inputs[3].video_units);
    }

    /// A request that breaks a request-only rule is refused before any of its media is fetched:
    /// the length is refused, not the reference whose URL could never be fetched.
    #[tokio::test]
    async fn request_rules_apply_before_fetching() {
        let body = request(serde_json::json!({
            "model": "test",
            "prompt": "a fox",
            "seconds": 0.5,
            "input_references": [{"type": "image", "url": "data:,"}],
        }));
        let Err(ServeError::Video(processing::InputError::Invalid { param, .. })) =
            test_video_service().process("rules", &body, 1 << 16).await
        else {
            panic!("the request is refused");
        };
        assert_eq!(param, "seconds");
    }

    /// The planning vectors MiniMax-H3's request rules are tested against, generated from the
    /// diffusers reference.
    fn vectors() -> Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../models/minimax-h3/tests/data/minimax_h3_plan.json"
        );
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    /// A MiniMax-H3 checkpoint's tokenizer, loaded as the server loads a pipeline's
    /// (`UNISERVE_MINIMAX_H3_MODEL` names the checkpoint root), encodes every text the vectors
    /// present to the reference's token ids and resolves the vision markers and placeholders to
    /// the reference's ids.
    #[tokio::test]
    async fn a_pipeline_tokenizer_matches_the_reference() {
        use processing::Vocabulary as _;

        let Ok(root) = std::env::var("UNISERVE_MINIMAX_H3_MODEL") else {
            eprintln!("UNISERVE_MINIMAX_H3_MODEL is not set; skipping the tokenizer parity test");
            return;
        };
        let pipeline = PipelineCheckpoint::resolve(&root).await.unwrap().unwrap();
        let tokenizer = pipeline_tokenizer(&pipeline).await.unwrap();
        let fixture = vectors();
        for (token, id) in fixture["tokens"].as_object().unwrap() {
            assert_eq!(
                tokenizer.token_id(token).map(u64::from),
                id.as_u64(),
                "{token}"
            );
        }
        for case in fixture["requests"].as_array().unwrap() {
            let Some(expected) = case.get("expected") else {
                continue;
            };
            for segment in expected["segments"].as_array().unwrap() {
                let Some(text) = segment.get("text") else {
                    continue;
                };
                let ids: Vec<u64> =
                    processing::Vocabulary::encode(&tokenizer, text.as_str().unwrap())
                        .unwrap()
                        .into_iter()
                        .map(u64::from)
                        .collect();
                assert_eq!(Value::from(ids), segment["token_ids"], "{}", case["name"]);
            }
        }
    }
}
