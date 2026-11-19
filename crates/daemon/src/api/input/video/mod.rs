//! Video generation: the video service of one deployment.
//!
//! The public request is the same for every video model
//! (`protocol::VideoGenerationRequest`): a prompt,
//! an output size and length, and reference media, each anchoring a generated frame or
//! conditioning the generation as a whole. What a model makes of it is the model's: its video
//! family in the epoch's processing program, which the deployment bound to what its workers
//! described, holds the request to the model's rules and processes it, in the request's own
//! instance of the program.
//!
//! [`VideoService::process`] runs one request through five stages in order:
//!
//! 1. The family's request-only rules (`video_check`), so a malformed request is refused before
//!    anything is fetched.
//! 2. The service's [`MediaConnector`] resolves each `input_references[i].url` under the server's
//!    media policy (`data:`, `file://` under the media directory, `http(s)://` with size, time and
//!    address limits) into an owned buffer, the media of one request sharing one total.
//! 3. [`probe::MediaProber`] reads the facts processing needs through the decoders the worker's media
//!    reader decodes them with (`media::probe`): an image's displayed size, its EXIF orientation
//!    applied; a video's display aspect, frame rate, decoded frame count and soundtrack, and an
//!    audio file's sample rate and decoded sample count.
//! 4. The family processes the request from the probed media (`video_process`), its prompt
//!    tokenized with the checkpoint's tokenizer, which the daemon answers the program with: its
//!    size and frames, how each reference is decoded and encoded, the prompt as the text encoder
//!    encodes it, and the denoiser rows the references take.
//! 5. Each reference's media is copied into a sealed anonymous memory file, which the request holds
//!    and the request interface carries beside it to the engine's media reader, and the request
//!    becomes one `DiffusionRequest` describing every condition and stating the rows they take, by
//!    which the engine finds the smallest graph that holds them.
//!
//! Stages 1 to 3 wait only on I/O and run on the async runtime; stages 4 and 5 are CPU work and
//! run on the blocking pool.
//!
//! Every refusal is an `InputError::Invalid` naming the request parameter at fault,
//! `input_references[i]` for a reference; the API answers it as an `invalid_request_error`.
//! Failures of the server's own resources (the blocking pool) are `InputError::Internal`.

mod encodings;
pub mod probe;

use std::num::NonZeroU32;
use std::sync::Arc;

use proto::MappedMedia;
use proto::request::{DiffusionRequest, DiffusionSamplingParams, RequestId, Trajectory};
use proto::video::{Canvas, Condition, Conditions};
use scheduling::processing::video::{
    ConditionInput, Facts, InputError, Length, Process, Reference, ReferenceKind, VideoInput,
    VideoSpec,
};
use serde_json::{Value, json};

use crate::api::input::media::{FetchedMedia, MediaConnector};
use crate::api::input::{ServeError, TokenizeError};
use crate::api::program::Binding;
use crate::api::protocol::{VideoGenerationRequest, request_field_names};
use proto::config::MediaConfig;

use probe::MediaProber;

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
            (Some(seconds), None) => Some(Length::Seconds(seconds)),
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
            seed: super::request_seed(self.seed),
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
                format!("{size:?} is not WxH: a width and a height in pixels, each positive"),
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
    /// The output's frame rate, in frames per second.
    pub fps: NonZeroU32,
}

impl ProcessedVideo {
    /// The generated duration, in seconds.
    pub fn seconds(&self) -> f64 {
        f64::from(self.num_frames) / f64::from(self.fps.get())
    }
}

/// Processes the video requests of one deployment.
///
/// One instance is shared by all requests; [`VideoService::process`] takes `&self` and runs
/// concurrently, each request in an instance of its own of the family's binding.
pub struct VideoService {
    /// The video family bound to the placed denoiser in the epoch's processing program.
    binding: Arc<Binding>,
    /// What the bound family serves.
    facts: Facts,
    /// The media policy, which `capabilities` reports.
    media: MediaConfig,
    connector: MediaConnector,
    prober: MediaProber,
}

impl VideoService {
    /// The service of a deployment whose video family is bound as `binding`, under the media
    /// policy `media`.
    ///
    /// # Errors
    ///
    /// Fails when the binding is no video family's or the media connector cannot be built.
    pub fn new(binding: Arc<Binding>, media: &MediaConfig) -> anyhow::Result<Self> {
        let scheduling::processing::Bound::Video(facts) = binding.facts().clone() else {
            anyhow::bail!("the binding of a video service states no video facts");
        };
        Ok(Self {
            binding,
            facts,
            media: media.clone(),
            connector: MediaConnector::new(media)?,
            prober: MediaProber::new(media)?,
        })
    }

    /// The denoiser evaluations of every request.
    pub fn denoise_steps(&self) -> u32 {
        self.facts.denoise_steps
    }

    /// Describes what the deployment serves of video for `GET /v1/capabilities`: the model's own part, the
    /// prompt capacity, the media sources and size caps, and the request fields.
    pub fn capabilities(&self, max_prompt_tokens: u32) -> Value {
        json!({
            "model": self.facts.capabilities,
            "max_prompt_tokens": max_prompt_tokens,
            "media": {
                "data": true,
                "http": self.media.allow_remote,
                "file": self.media.directory.is_some(),
                "max_request_bytes": self.media.max_request_bytes,
            },
            "request_fields": request_field_names(),
        })
    }

    /// Validates, fetches, probes and processes one video request into its engine request and what
    /// it generates.
    ///
    /// The request-only rules run before any fetch. The processed prompt must fit the prompt
    /// capacity (`--max-model-len`). Each reference's fetched media is then copied into a sealed
    /// memory file, which the request holds for as long as it lives. The daemon's submission
    /// stamps the engine's identifier on the request; `trace` is the request's trace id, which
    /// the request carries and its fetch and preprocess spans name.
    ///
    /// # Errors
    ///
    /// Returns `ServeError::Video` naming the parameter at fault, or failing when processing's
    /// own resources or the memory file fail, and `TokenizeError::PromptTooLong` for a prompt
    /// longer than the prompt capacity.
    pub async fn process(
        &self,
        request_id: &str,
        request: &VideoGenerationRequest,
        max_prompt_tokens: u32,
        trace: u64,
    ) -> Result<(DiffusionRequest, ProcessedVideo), ServeError> {
        use service::span::{Stage, enter};

        let spec = request.spec()?;
        let mut session = self.binding.checkout()?;
        session.check_video(&spec)??;

        // Fetches and probes of different references run concurrently; the first failure ends
        // the request.
        let urls: Vec<&str> = request
            .input_references
            .iter()
            .map(|reference| reference.url.as_str())
            .collect();
        let fetch = (trace != 0 && !urls.is_empty()).then(|| enter(Stage::Fetch, trace));
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
        drop(fetch);

        let request_id = request_id.to_owned();
        let priority = request.priority.unwrap_or(0);
        tokio::task::spawn_blocking(move || {
            let preprocess = (trace != 0).then(|| enter(Stage::Preprocess, trace));
            let input = session.process_video(&Process {
                spec,
                media: probes,
            })??;
            drop(session);
            drop(preprocess);
            let prompt_len = u32::try_from(input.prompt_token_ids.len()).map_err(|_| {
                InputError::Internal("a prompt of more tokens than a u32 counts".into())
            })?;
            if prompt_len > max_prompt_tokens {
                return Err(ServeError::Tokenize {
                    request_id,
                    source: TokenizeError::PromptTooLong {
                        max_model_len: max_prompt_tokens,
                        prompt_len,
                    },
                });
            }
            submission(input, &media, priority, trace)
        })
        .await
        .unwrap_or_else(|error| {
            Err(InputError::Internal(format!("processing the request failed: {error}")).into())
        })
    }
}

/// The engine request of `input` at scheduler `priority` and of trace id `trace`, holding each
/// reference's media in a sealed memory file.
fn submission(
    input: VideoInput,
    media: &[FetchedMedia],
    priority: i32,
    trace: u64,
) -> Result<(DiffusionRequest, ProcessedVideo), ServeError> {
    // Each reference's media reaches the engine's media reader beside the request.
    let mut copies = Vec::with_capacity(media.len());
    let mut conditions = Vec::with_capacity(media.len());
    for (index, (condition, fetched)) in input.conditions.into_iter().zip(media).enumerate() {
        let copy = MappedMedia::copy(&fetched.bytes).map_err(|error| {
            InputError::Internal(format!(
                "holding input_references[{index}] for the engine failed: {error}"
            ))
        })?;
        conditions.push(condition_of(condition, fetched.bytes.len() as u64));
        copies.push(Arc::new(copy));
    }
    let conditions = Conditions {
        conditions,
        prompt_tokens: u32::try_from(input.prompt_token_ids.len()).map_err(|_| {
            InputError::Internal("a prompt of more tokens than a u32 counts".into())
        })?,
        vision_spans: input.vision_spans,
    };
    conditions
        .validate()
        .map_err(|reason| InputError::Internal(format!("the processed conditions: {reason}")))?;

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
    let encoder_inputs = encodings::encoder_inputs(&conditions, &input.prompt_token_ids, &contents)
        .map_err(|error| {
            InputError::Internal(format!("digesting the conditions failed: {error}"))
        })?;

    let request = DiffusionRequest {
        trace,
        request_id: RequestId::default(),
        prompt_token_ids: input.prompt_token_ids,
        conditions: conditions
            .conditions
            .iter()
            .map(|condition| condition.extent())
            .collect(),
        conditioning: conditions.encode(),
        priority,
        sampling: DiffusionSamplingParams {
            num_frames: input.num_frames,
            video_units: input.video_units,
            seed: input.seed,
            width: input.size.width,
            height: input.size.height,
            condition_rows: input.condition_rows,
        },
        // The migrator gives the request its trajectory's identity when it first submits it.
        trajectory: Trajectory::default(),
        encoder_inputs: Some(Box::new(encoder_inputs)),
        media: copies,
    };
    let video = ProcessedVideo {
        size: input.size,
        num_frames: input.num_frames,
        fps: NonZeroU32::new(input.fps).ok_or_else(|| {
            InputError::Internal("the processor states an output of no frame rate".into())
        })?,
    };
    Ok((request, video))
}

/// The condition `input` states, whose fetched media are `bytes` long and travel beside the
/// request.
fn condition_of(input: ConditionInput, bytes: u64) -> Condition {
    Condition {
        role: input.role,
        bytes,
        media: input.media,
        vision: input.vision,
        latent_units: input.latent_units,
        audio_rows: input.audio_rows,
    }
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;
    use image::ImageEncoder as _;
    use serde_json::Value;

    use super::{ServeError, VideoGenerationRequest};
    use crate::api::test_support::test_video_service;

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
            "model": "m", "prompt": "a fox", "size": "1344x768", "seconds": 5,
            "input_references": [{"type": "image", "url": "data:,", "frame_index": -1}],
        }))
        .spec()
        .unwrap();
        assert_eq!(
            (spec.size.unwrap().width, spec.size.unwrap().height),
            (1344, 768)
        );
        assert_eq!(
            spec.length,
            Some(scheduling::processing::video::Length::Seconds(5.0))
        );
        assert_eq!(spec.references[0].frame_index, Some(-1));

        let refused = |body: Value| match request(body).spec() {
            Err(scheduling::processing::video::InputError::Invalid { param, .. }) => param,
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
            let (resolved, _) = service
                .process("encoding", &input(prompt, height), 1 << 16, 0)
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
        let Err(ServeError::Video(scheduling::processing::video::InputError::Invalid {
            param,
            ..
        })) = test_video_service()
            .process("rules", &body, 1 << 16, 0)
            .await
        else {
            panic!("the request is refused");
        };
        assert_eq!(param, "seconds");
    }
}
