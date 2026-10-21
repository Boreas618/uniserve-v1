//! MiniMax-H3 video generation: the request vocabulary and the video
//! service of one deployment.
//!
//! [`VideoService`] binds what the deployment's denoiser serves, as the
//! worker reports it in its startup handshake ([`VideoDenoiserInfo`]), to the
//! server's media policy and the checkpoint's vision processor, and prepares
//! every video request for the engine. A MiniMax-H3 request must be sized
//! before admission: the engine reserves the conditioner's token count and
//! the worker encodes each condition in the units the plan states, so both
//! are known exactly before any pixel is decoded. [`VideoService::process`]
//! runs one request through five stages in order:
//!
//! 1. [`plan::check_request`] applies every rule that needs no media
//!    (task, roles, counts, target), so a malformed request is rejected
//!    before anything is fetched.
//! 2. The service's [`MediaConnector`] resolves each `conditions[i].uri`
//!    under the server's media policy (`data:`, `file://` under the media
//!    directory, `http(s)://` with size, time and address limits) into an
//!    owned buffer, the media of one request sharing one total.
//! 3. [`probe::MediaProber`] reads the facts the plan needs: an image's
//!    displayed size from its header and EXIF orientation; a video's display
//!    aspect, frame rate, decoded frame count and soundtrack, and an audio
//!    file's sample rate and decoded sample count, through `ffprobe`.
//! 4. [`plan::plan_request`] resolves the canvas, the frame counts, how each
//!    condition is prepared, the Qwen3-VL vision grids and the encoder units;
//!    [`presentation::present`] tokenizes the conditioner's presentation with
//!    the checkpoint's tokenizer and tags every token for the denoiser's AdaLN
//!    modulation.
//! 5. Each condition's media is published to shared memory for the worker's
//!    media reader, and the request becomes one `DiffusionRequest`
//!    describing every condition. The worker that denoises the request admits
//!    its conditions into the condition capacity it provisioned, counting
//!    their rows as its denoiser packs them.
//!
//! Stages 1 to 3 wait only on I/O and run on the async runtime; stages 4
//! and 5 are CPU work, the planning, the tokenization and the copies into
//! shared memory, and run on the blocking pool ([`Presenter`]).
//!
//! The request rules are the `models` crate's ([`plan`], [`presentation`]); the
//! model package reads the canvas rule and the frame counts from them through
//! the worker library, and plans no request.
//! [`VideoService::capabilities`] describes the same contract for
//! `GET /v1/capabilities`.
//!
//! Every rejection is a [`VideoInputError::Invalid`] naming the request field
//! at fault, `conditions[i]` for a condition; the API answers it as an
//! `invalid_request_error`. Failures of the server's own resources (a missing
//! `ffprobe`, scratch storage) are [`VideoInputError::Internal`].

pub mod artifact;
pub mod probe;

use std::sync::Arc;

use models::video::plan;
use models::video::plan::{
    CANVAS_MAX_PIXELS, CANVAS_MULTIPLE, ConditionSpec, ConditionType, MAX_VIDEO_SECONDS,
    MIN_VIDEO_SECONDS, PlanLimits, ProcessorConfig, ReferenceConfig, Role, Target, VideoInputError,
};
use models::video::presentation;
use models::video::{Canvas, Conditioning, DenoiserInfo, VideoTask};
use proto::MediaSource;
use proto::request::{ConditionExtent, DiffusionRequest};
use proto::request::{DiffusionSamplingParams, RequestId};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::options::MediaOptions;
use crate::profile::tokenizer::DynTokenizer;
use crate::serving::ServeError;
use crate::serving::media::{FetchedMedia, MediaConnector};

use probe::{MediaProber, ProbeConfig};

/// The seed a request without one uses, as the official reference does.
pub const DEFAULT_VIDEO_SEED: u64 = 42;

/// One video, audio or image condition of a request, in request order.
///
/// The order is semantic: the conditioner presents references in it and
/// their ordinal labels name them.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VideoCondition {
    /// `image`, `video`, `video_audio` (a video that must have a soundtrack)
    /// or `audio`.
    #[serde(rename = "type")]
    pub condition_type: ConditionType,
    /// `data:`, `file://` or `http(s)://` location of the media.
    pub uri: String,
    /// `keyframe` (an image anchoring a generated frame) or `reference`.
    pub role: Role,
    /// The generated frame a keyframe anchors: 0 or -1.
    #[serde(default)]
    pub frame_index: Option<i64>,
    /// Offset into a video reference, seconds; its soundtrack follows it.
    #[serde(default)]
    pub start_time_seconds: Option<f64>,
}

/// The requested output; the canvas and the duration come only from here.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VideoTarget {
    /// Short edge of the canvas in pixels; the deployment serves the short
    /// edges `GET /v1/capabilities` reports.
    pub short_edge: u32,
    /// `auto` or `W:H`.
    pub aspect_ratio: String,
    /// Duration in seconds; a reference-conditioned request whose only
    /// audio-bearing reference sets its duration may omit it.
    #[serde(default)]
    pub duration_seconds: Option<f64>,
}

/// MiniMax-H3 video-and-audio generation request: the official MiniMax /
/// SGLang form, a `task`, its ordered `conditions` and the requested
/// `target`. The `/v1/videos` routes build it, and
/// `deny_unknown_fields` rejects every field not listed here, at every
/// level. The served checkpoint fixes the schedule, and every request
/// generates one video.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VideoGenerationRequest {
    /// Served model name.
    pub model: String,
    /// Prompt; the conditioner presents it after the conditions.
    pub prompt: String,
    /// `t2va`, `fl2va` or `ref2va`; one the served denoiser serves.
    pub task: VideoTask,
    /// The task's conditions, in request order.
    #[serde(default)]
    pub conditions: Vec<VideoCondition>,
    /// The requested output.
    pub target: VideoTarget,
    /// Video-generation random seed.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

const fn default_seed() -> u64 {
    DEFAULT_VIDEO_SEED
}

/// The fields of the video request body, in schema order.
pub(crate) const REQUEST_FIELDS: [&str; 6] =
    ["model", "prompt", "task", "conditions", "target", "seed"];

/// What a processed video request generates, as its job record reports it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcessedVideo {
    /// The requested duration in seconds, before frame alignment.
    pub duration_seconds: f64,
    /// The duration of the generated frames, in seconds: the frame count at
    /// the deployment's frame rate.
    pub actual_seconds: f64,
    /// The frames generated.
    pub num_frames: u32,
    /// The generated canvas.
    pub canvas: Canvas,
}

/// The condition media of one planned request, published to shared memory.
///
/// The workers read each by name while the request runs, so whoever submits
/// the request holds them until it ends; dropping them unlinks the names.
#[derive(Debug)]
pub struct PublishedMedia {
    _sources: Vec<MediaSource>,
}

/// Processes the video requests of one deployment.
///
/// One instance is shared by all requests; [`VideoService::process`] takes
/// `&self` and runs concurrently.
pub struct VideoService {
    denoiser: DenoiserInfo,
    /// The media policy, which `capabilities` reports.
    media: MediaOptions,
    connector: MediaConnector,
    prober: MediaProber,
    /// What plans and presents a request, shared with the blocking pool.
    presenter: Arc<Presenter>,
}

/// What plans and presents a fetched and probed request: the deployment's
/// limits, the vision processor, the latent encoder's round width and the
/// tokenizer. Its work is CPU-bound and runs on the blocking pool.
struct Presenter {
    /// What the deployment serves, which every request is planned against.
    limits: PlanLimits,
    vision: ProcessorConfig,
    /// Condition units one latent encoding round covers while a request's
    /// prompt is encoded, which reference images are banded to fill
    /// (`RequestPlan::image_bands`).
    latent_encoding_lane: u32,
    tokenizer: DynTokenizer,
}

/// The parts of a video request planning reads, owned so that the blocking
/// pool can take them.
struct Fetched {
    request_id: String,
    task: VideoTask,
    short_edge: u32,
    aspect_ratio: String,
    duration_seconds: Option<f64>,
    prompt: String,
    seed: u64,
    specs: Vec<ConditionSpec>,
    probes: Vec<plan::MediaFacts>,
    media: Vec<FetchedMedia>,
}

impl VideoService {
    /// Binds the served denoiser, the checkpoint's vision processor, the
    /// deployment's duration capacity, the condition units its latent
    /// encoder covers in one round, its media policy, how it sizes
    /// references, and the tokenizer.
    ///
    /// `denoiser` is the description the denoising worker reported, and
    /// `references` a sizing `ReferenceConfig::from_kwargs` validated against
    /// it.
    ///
    /// # Errors
    ///
    /// Fails when the handshake names a task this server does not know, or
    /// when the media connector cannot be built.
    pub fn new(
        denoiser: DenoiserInfo,
        vision: ProcessorConfig,
        max_video_seconds: f64,
        latent_encoding_lane: u32,
        media: &MediaOptions,
        references: ReferenceConfig,
        tokenizer: DynTokenizer,
    ) -> anyhow::Result<Self> {
        let tasks = denoiser
            .tasks
            .iter()
            .map(|name| {
                VideoTask::from_name(name)
                    .ok_or_else(|| anyhow::anyhow!("the worker serves unknown video task {name:?}"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let connector = MediaConnector::from_media_options(media)?;
        let prober = MediaProber::new(ProbeConfig::new(media.ffprobe.clone()));
        let limits = PlanLimits {
            tasks,
            max_video_seconds,
            frame_rate: denoiser.frame_rate,
            frames: denoiser.frames,
            canvases: denoiser.canvases.clone(),
            reference_limits: denoiser.references,
            references,
        };
        Ok(Self {
            denoiser,
            media: media.clone(),
            connector,
            prober,
            presenter: Arc::new(Presenter {
                limits,
                vision,
                latent_encoding_lane: latent_encoding_lane.max(1),
                tokenizer,
            }),
        })
    }

    /// The served video denoiser, as the denoising worker described it.
    pub fn denoiser(&self) -> &DenoiserInfo {
        &self.denoiser
    }

    /// The denoiser network evaluations of every request: one per schedule
    /// interval, as the handshake's video denoiser states them.
    pub fn num_inference_steps(&self) -> u32 {
        self.denoiser.denoise_steps()
    }

    /// Describes the served video contract for `GET /v1/capabilities`.
    ///
    /// Reports the tasks and their condition rules, with how the deployment
    /// sizes `ref2va` references, the canvas rule, the canvases the
    /// deployment serves with the short edge and aspect ratio that select
    /// each, in the order the deployment names them, the output frame rate,
    /// audio sample rate and frame rule, the duration range, the fixed
    /// schedule (its sigma points and shifts), the prompt capacity, and the
    /// media sources and size caps.
    pub fn capabilities(&self, max_prompt_tokens: u32) -> Value {
        // Each served canvas with the pair that selects it, and the aspect
        // ratios served at any short edge, in the order the deployment names
        // them; `auto` is 16:9 for t2va and ref2va.
        let mut aspect_ratios: Vec<&str> = Vec::new();
        let mut sizes = serde_json::Map::new();
        let mut canvases = Vec::with_capacity(self.denoiser.canvases.len());
        for served in &self.denoiser.canvases {
            if !aspect_ratios.contains(&served.aspect_ratio.as_str()) {
                aspect_ratios.push(&served.aspect_ratio);
            }
            let size = json!({"width": served.canvas.width, "height": served.canvas.height});
            let entry = sizes
                .entry(served.short_edge.to_string())
                .or_insert_with(|| json!({}));
            entry[served.aspect_ratio.as_str()] = size;
            canvases.push(json!({
                "short_edge": served.short_edge,
                "aspect_ratio": served.aspect_ratio,
                "width": served.canvas.width,
                "height": served.canvas.height,
            }));
        }
        let limits = &self.presenter.limits;
        let references = limits.references;
        let conditions: serde_json::Map<String, Value> = limits
            .tasks
            .iter()
            .map(|task| {
                let rules = match task {
                    VideoTask::T2va => json!({"conditions": []}),
                    VideoTask::Fl2va => json!({
                        "keyframe_frame_indices": [[0], [-1], [0, -1]],
                        "aspect_ratio": "auto or W:H",
                    }),
                    VideoTask::Ref2va => json!({
                        "max_images": self.denoiser.references.images,
                        "max_videos": self.denoiser.references.videos,
                        "max_audios": self.denoiser.references.audios,
                        "max_references": self.denoiser.references.total,
                        "keyframe_frame_indices": [[], [0], [-1], [0, -1]],
                        "reference_image_short_edge": references.image_short_edge,
                        "enlarge_references": references.enlarge,
                    }),
                };
                (task.as_str().to_owned(), rules)
            })
            .collect();
        json!({
            "tasks": self.denoiser.tasks,
            "task_conditions": conditions,
            "canvas": {
                "short_edges": limits.short_edges(),
                "multiple": CANVAS_MULTIPLE,
                "max_pixels": CANVAS_MAX_PIXELS,
                "aspect_ratios": aspect_ratios,
                "sizes": sizes,
                "canvases": canvases,
            },
            "fps": self.denoiser.frame_rate,
            "audio_sample_rate": self.denoiser.audio_sample_rate,
            "frames": self.denoiser.frames,
            "min_seconds": MIN_VIDEO_SECONDS,
            "max_seconds": limits.max_video_seconds,
            "model_max_seconds": MAX_VIDEO_SECONDS,
            "schedule": {
                "num_inference_steps": self.denoiser.schedule_points,
                "flow_shift": self.denoiser.video_shift,
                "audio_flow_shift": self.denoiser.audio_shift,
            },
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

    /// Validates, fetches, probes, plans and presents one video request into
    /// its engine request and what it generates.
    ///
    /// The request-only rules run before any fetch. The presentation must
    /// fit the prompt capacity (`--max-model-len`).
    /// Each condition's fetched media is then copied to shared memory, which
    /// the request holds until the engine retires it. `Engine::submit` stamps
    /// the engine's identifier on the request.
    ///
    /// # Errors
    ///
    /// Returns `ServeError::Video` naming the field at fault, or failing
    /// when processing's own resources or the shared-memory allocation fail,
    /// and `ServeError::ContextLengthExceeded` for a presentation longer than
    /// the prompt capacity.
    pub async fn process(
        &self,
        request_id: &str,
        request: &VideoGenerationRequest,
        max_prompt_tokens: u32,
    ) -> Result<(DiffusionRequest, ProcessedVideo, PublishedMedia), ServeError> {
        let target = Target {
            short_edge: request.target.short_edge,
            aspect_ratio: &request.target.aspect_ratio,
            duration_seconds: request.target.duration_seconds,
        };
        let specs: Vec<ConditionSpec> = request
            .conditions
            .iter()
            .map(|condition| ConditionSpec {
                condition_type: condition.condition_type,
                role: condition.role,
                frame_index: condition.frame_index,
                start_seconds: condition.start_time_seconds,
            })
            .collect();
        plan::check_request(request.task, &target, &specs, &self.presenter.limits)?;

        // Fetches and probes of different conditions run concurrently; the
        // first failure ends the request.
        let uris: Vec<&str> = request
            .conditions
            .iter()
            .map(|condition| condition.uri.as_str())
            .collect();
        let media =
            self.connector.fetch_all(&uris).await.map_err(|error| {
                VideoInputError::condition(error.index, error.source.to_string())
            })?;
        let probes =
            futures::future::try_join_all(media.iter().enumerate().map(|(index, media)| {
                self.prober
                    .probe(index, specs[index].condition_type, &media.bytes)
            }))
            .await?;

        let fetched = Fetched {
            request_id: request_id.to_owned(),
            task: request.task,
            short_edge: request.target.short_edge,
            aspect_ratio: request.target.aspect_ratio.clone(),
            duration_seconds: request.target.duration_seconds,
            prompt: request.prompt.clone(),
            seed: request.seed,
            specs,
            probes,
            media,
        };
        let presenter = Arc::clone(&self.presenter);
        tokio::task::spawn_blocking(move || presenter.present(fetched, max_prompt_tokens))
            .await
            .unwrap_or_else(|error| {
                Err(
                    VideoInputError::Internal(format!("planning the request failed: {error}"))
                        .into(),
                )
            })
    }
}

/// The sizes of `condition` the engine schedules its reading and encoding
/// by.
fn condition_extent(condition: &models::video::VideoCondition) -> ConditionExtent {
    ConditionExtent {
        pixel_bytes: condition.pixel_bytes(),
        audio_samples: condition.media.audio().map_or(0, |track| track.samples),
        vision_patches: condition
            .vision
            .as_ref()
            .map_or(0, |vision| vision.grid.patches()),
        vision_tokens: condition.vision.as_ref().map_or(0, |vision| vision.tokens),
        latent_units: condition.latent_units.clone(),
        audio_rows: condition.audio_rows,
    }
}

impl Presenter {
    /// Plans `fetched`, presents its prompt, which must fit
    /// `max_prompt_tokens`, and publishes each condition's media to shared
    /// memory, which the request holds until the engine retires it.
    fn present(
        &self,
        fetched: Fetched,
        max_prompt_tokens: u32,
    ) -> Result<(DiffusionRequest, ProcessedVideo, PublishedMedia), ServeError> {
        let target = Target {
            short_edge: fetched.short_edge,
            aspect_ratio: &fetched.aspect_ratio,
            duration_seconds: fetched.duration_seconds,
        };
        let plan = plan::plan_request(
            fetched.task,
            &target,
            &fetched.specs,
            &fetched.probes,
            &self.vision,
            &self.limits,
        )?;
        let presentation = presentation::present(&*self.tokenizer, &plan, &fetched.prompt)?;

        let prompt_token_ids = presentation.token_ids;
        if prompt_token_ids.len() > max_prompt_tokens as usize {
            return Err(ServeError::ContextLengthExceeded {
                request_id: fetched.request_id,
                prompt_tokens: prompt_token_ids.len(),
                max_tokens: max_prompt_tokens,
            });
        }

        // Each condition's media reaches the worker's media reader through
        // shared memory on this host.
        let image_bands = plan.image_bands(self.latent_encoding_lane);
        let mut media = Vec::with_capacity(fetched.media.len());
        let mut conditions = Vec::with_capacity(fetched.media.len());
        for ((condition, fetched), bands) in
            plan.conditions.iter().zip(&fetched.media).zip(image_bands)
        {
            let source = MediaSource::publish(&fetched.bytes).map_err(|error| {
                VideoInputError::Internal(format!(
                    "publishing conditions[{}] for the worker failed: {error}",
                    condition.index
                ))
            })?;
            let locator = source.locator();
            let locator = models::video::MediaLocator {
                name: locator.name,
                bytes: locator.bytes,
            };
            conditions.push(condition.describe(plan.canvas, locator, bands));
            media.push(source);
        }
        let conditioning = Conditioning {
            task: plan.task,
            text_tags: presentation.tags,
            conditions,
        };
        let request = DiffusionRequest {
            request_id: RequestId::default(),
            task: plan.task.as_str().to_owned(),
            prompt_token_ids,
            conditions: conditioning
                .conditions
                .iter()
                .map(condition_extent)
                .collect(),
            conditioning: conditioning.encode().map_err(|error| {
                VideoInputError::Internal(format!("encoding the conditioning failed: {error}"))
            })?,
            priority: 0,
            sampling: DiffusionSamplingParams {
                num_frames: plan.num_frames,
                // One decode unit per clip of the denoiser's frame rule.
                video_units: self.limits.frames.units(plan.num_frames),
                seed: fetched.seed,
                width: plan.canvas.width,
                height: plan.canvas.height,
            },
        };
        let video = ProcessedVideo {
            duration_seconds: plan.duration_seconds,
            actual_seconds: f64::from(plan.num_frames) / f64::from(self.limits.frame_rate),
            num_frames: plan.num_frames,
            canvas: plan.canvas,
        };
        Ok((request, video, PublishedMedia { _sources: media }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use base64::Engine as _;
    use image::ImageEncoder as _;
    use models::video::plan::{ProcessorConfig, ReferenceConfig, RequestField};
    use models::video::presentation::{TEXT_TAG, VIDEO_TAG};
    use models::video::{
        Canvas, ConditionMedia, ConditionRole, ConditionVision, ImageFit, ReferenceLimits,
        ServedCanvas, VideoTask, VisionGrid,
    };
    use serde_json::Value;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

    use super::{
        Conditioning, DenoiserInfo, DiffusionRequest, VideoGenerationRequest, VideoService,
    };
    use crate::options::MediaOptions;
    use crate::profile::assets::PipelineCheckpoint;
    use crate::profile::tokenizer::HuggingFaceTokenizer;
    use crate::serving::ServeError;
    use crate::serving::model::pipeline_tokenizer;

    /// The planning vectors the request rules are tested against.
    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../../../tests/python/fixtures/minimax_h3_plan.json"
        ))
        .unwrap()
    }

    /// The MiniMax-H3 checkpoint's vision processor geometry.
    fn vision() -> ProcessorConfig {
        serde_json::from_value(fixture()["vision"].clone()).unwrap()
    }

    /// The released checkpoint's documented references: nine images, three
    /// videos, three audio tracks and twelve in all, images at a short edge
    /// of at most 2048 pixels.
    const REFERENCE_LIMITS: ReferenceLimits = ReferenceLimits {
        images: 9,
        videos: 3,
        audios: 3,
        total: 12,
        image_short_edge: 2048,
    };

    /// The canvases of the vectors' accepted requests, as a deployment that
    /// prepares them all declares them. A `t2va` or `ref2va` request's ratio
    /// names its canvas (`auto` naming 16:9); an `fl2va` canvas, which the
    /// canvas rule resolves from a free ratio or the first keyframe, is
    /// declared under its own size.
    fn served_canvases() -> Vec<ServedCanvas> {
        let mut declared: Vec<ServedCanvas> = Vec::new();
        for case in fixture()["requests"].as_array().unwrap() {
            let Some(expected) = case.get("expected") else {
                continue;
            };
            let pair = &expected["canvas"];
            let canvas = Canvas {
                width: pair[0].as_u64().unwrap() as u32,
                height: pair[1].as_u64().unwrap() as u32,
            };
            let target = &case["target"];
            let short_edge = target["short_edge"].as_u64().unwrap() as u32;
            let aspect_ratio = match (
                case["task"].as_str().unwrap(),
                target["aspect_ratio"].as_str().unwrap(),
            ) {
                ("fl2va", _) => format!("{}:{}", canvas.width, canvas.height),
                (_, "auto") => "16:9".to_owned(),
                (_, named) => named.to_owned(),
            };
            if !declared.iter().any(|served| {
                served.short_edge == short_edge && served.aspect_ratio == aspect_ratio
            }) {
                declared.push(ServedCanvas {
                    short_edge,
                    aspect_ratio,
                    canvas,
                });
            }
        }
        declared
    }

    /// A deterministic tokenizer: one token per character, plus the vision
    /// markers and placeholders as special tokens.
    fn character_tokenizer() -> HuggingFaceTokenizer {
        let mut vocab: Vocab = (1_u32..=127)
            .map(|id| (char::from_u32(id).unwrap().to_string(), id))
            .collect();
        vocab.insert("<unk>".to_owned(), 0);
        let specials = [
            ("<|vision_start|>", 128),
            ("<|vision_end|>", 129),
            ("<|image_pad|>", 130),
            ("<|video_pad|>", 131),
        ];
        for (token, id) in specials {
            vocab.insert(token.to_owned(), id);
        }
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .unk_token("<unk>".to_owned())
            .build()
            .unwrap();
        let mut tokenizer = TokenizerBuilder::new(model);
        tokenizer.add_special_tokens(
            &specials
                .iter()
                .map(|(token, _)| AddedToken::from(*token, true))
                .collect::<Vec<_>>(),
        );
        crate::test_support::load_tokenizer(&tokenizer)
    }

    /// The conditioning a planned request carries to the workers.
    fn planned(request: &DiffusionRequest) -> Conditioning {
        Conditioning::decode(&request.conditioning).unwrap()
    }

    /// A base denoiser serving every task, on a latent encoder of one rank,
    /// in a deployment that sizes references as the released pipeline does.
    fn service() -> VideoService {
        encoding_service(1, ReferenceConfig::released(&REFERENCE_LIMITS))
    }

    /// A denoiser serving every task, whose latent encoder covers
    /// `latent_encoding_lane` condition units in one round, on a deployment
    /// that sizes references as `references` states.
    fn encoding_service(latent_encoding_lane: u32, references: ReferenceConfig) -> VideoService {
        VideoService::new(
            DenoiserInfo {
                tasks: ["t2va", "fl2va", "ref2va"].map(str::to_owned).to_vec(),
                schedule_points: 50,
                video_shift: 12.0,
                audio_shift: 3.0,
                frame_rate: 24,
                audio_sample_rate: 32_000,
                frames: models::video::FrameClips {
                    min_frames: 22,
                    clip_frames: 17,
                    clip_latent_frames: 5,
                    tail_latent_frames: 2,
                },
                canvases: served_canvases(),
                references: REFERENCE_LIMITS,
            },
            vision(),
            15.0,
            latent_encoding_lane,
            &MediaOptions::default(),
            references,
            Arc::new(character_tokenizer()),
        )
        .unwrap()
    }

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut encoded = Vec::new();
        image::codecs::png::PngEncoder::new(&mut encoded)
            .write_image(
                &vec![90; (width * height * 3) as usize],
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        encoded
    }

    fn data_uri(bytes: &[u8]) -> String {
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        )
    }

    fn request(body: serde_json::Value) -> VideoGenerationRequest {
        serde_json::from_value(body).unwrap()
    }

    /// Reads a published object's bytes by its locator, as the worker's
    /// media reader does.
    fn published(name: &str, bytes: u64) -> Vec<u8> {
        let path = std::ffi::CString::new(format!("/{name}")).unwrap();
        // SAFETY: path is a valid NUL-terminated POSIX shm name.
        let descriptor = unsafe { libc::shm_open(path.as_ptr(), libc::O_RDONLY, 0) };
        assert!(descriptor >= 0, "the media is not published");
        let mut read = vec![0_u8; bytes as usize];
        // SAFETY: descriptor is open and `read` holds `bytes` writable bytes.
        let count = unsafe { libc::read(descriptor, read.as_mut_ptr().cast(), read.len()) };
        // SAFETY: descriptor is open.
        unsafe { libc::close(descriptor) };
        assert_eq!(count, bytes as isize);
        read
    }

    /// A reference image becomes one condition the worker can read and
    /// encode: its bytes published under its locator, sized as the
    /// deployment sizes references (enlarged to its 2048 short edge here),
    /// read by the conditioner as one image block, and encoded in one band
    /// of its patch rows per unit of a latent encoding round. The
    /// presentation's tags travel with its tokens. A deployment that does
    /// not enlarge keeps the image at its own size on the 32-pixel grid.
    #[tokio::test]
    async fn a_reference_image_is_described_for_the_worker() {
        let image = png(160, 90);
        let body = serde_json::json!({
            "model": "minimax_h3",
            "prompt": "a fox",
            "task": "ref2va",
            "conditions": [{"type": "image", "uri": data_uri(&image), "role": "reference"}],
            "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
        });
        let (unenlarged, _, _unenlarged_media) =
            encoding_service(4, ReferenceConfig::new(&REFERENCE_LIMITS))
                .process("reference", &request(body.clone()), 1 << 16)
                .await
                .unwrap();
        // 90 is 2.8 patches and rounds to 96.
        let own = Canvas {
            width: 160,
            height: 96,
        };
        assert_eq!(
            planned(&unenlarged).conditions[0].media,
            ConditionMedia::Image(ImageFit {
                resized: own,
                left: 0,
                top: 0,
                size: own,
            })
        );

        let (request, video, _media) =
            encoding_service(4, ReferenceConfig::released(&REFERENCE_LIMITS))
                .process("reference", &request(body), 1 << 16)
                .await
                .unwrap();
        let planned_request = planned(&request);
        assert_eq!(request.task, "ref2va");
        assert_eq!(planned_request.task, VideoTask::Ref2va);
        assert_eq!(request.validate(), Ok(()));
        assert_eq!((video.canvas.width, video.canvas.height), (1344, 768));
        assert_eq!(video.num_frames, 124);

        // The image block's label is text; its 7296 vision tokens and their
        // two markers are video; the prompt after it is text.
        let label = "<Picture 1>: ".chars().count();
        let tags = &planned_request.text_tags;
        assert_eq!(tags.len(), request.prompt_token_ids.len());
        assert_eq!(tags.len(), label + 7296 + 2 + "a fox".len());
        assert!(tags[..label].iter().all(|&tag| tag == TEXT_TAG));
        assert!(
            tags[label..label + 7298]
                .iter()
                .all(|&tag| tag == VIDEO_TAG)
        );
        assert!(tags[label + 7298..].iter().all(|&tag| tag == TEXT_TAG));

        let [condition] = planned_request.conditions.as_slice() else {
            panic!("one condition");
        };
        let size = Canvas {
            width: 3648,
            height: 2048,
        };
        assert_eq!(condition.role, ConditionRole::Reference);
        assert_eq!(
            condition.media,
            ConditionMedia::Image(ImageFit {
                resized: size,
                left: 0,
                top: 0,
                size,
            })
        );
        assert_eq!(
            condition.vision,
            Some(ConditionVision {
                grid: VisionGrid {
                    t: 1,
                    h: 128,
                    w: 228
                },
                tokens: 7296,
                frame_indices: Vec::new(),
            })
        );
        // 64 patch rows of 114 patches, 16 rows to each of four bands.
        assert_eq!(condition.latent_units, [16 * 114; 4]);
        assert_eq!(condition.audio_rows, 0);
        assert_eq!(
            published(&condition.source.name, condition.source.bytes),
            image
        );
    }

    /// Reference images beyond one latent encoding round encode whole, a
    /// round's worth at a time, and the images of a final partial round
    /// split into bands that fill it: on a four-unit lane nine square images
    /// take 1 x 8 + 4 units and six take 1 x 4 + 2 x 2, while seven, whose
    /// final three cannot share four units evenly, stay whole. A square
    /// reference enlarged to its 2048 short edge is 64 patch rows of 64
    /// patches.
    #[tokio::test]
    async fn reference_images_fill_their_final_latent_round() {
        let image = data_uri(&png(90, 90));
        // (images, images of the final round that split, their bands)
        for (count, split, bands) in [(9, 1, 4), (6, 2, 2), (7, 0, 1)] {
            let conditions = vec![
                serde_json::json!({"type": "image", "uri": image, "role": "reference"});
                count
            ];
            let (request, _, _media) = encoding_service(
                4,
                ReferenceConfig::released(&REFERENCE_LIMITS),
            )
            .process(
                "images",
                &request(serde_json::json!({
                    "model": "minimax_h3",
                    "prompt": "a fox",
                    "task": "ref2va",
                    "conditions": conditions,
                    "target": {"short_edge": 768, "aspect_ratio": "1:1", "duration_seconds": 5.0},
                })),
                1 << 17,
            )
            .await
            .unwrap();
            for (index, condition) in planned(&request).conditions.iter().enumerate() {
                let expected = if index < count - split {
                    vec![64 * 64]
                } else {
                    vec![64 / bands * 64; bands as usize]
                };
                assert_eq!(
                    condition.latent_units, expected,
                    "{count} images, image {index}"
                );
            }
        }
    }

    /// The first keyframe is stretched onto the canvas its own aspect sets
    /// and a second one cover-cropped: each is one still image at the
    /// canvas, anchoring its end of the video, and one latent unit on any
    /// latent encoder.
    #[tokio::test]
    async fn keyframes_are_fitted_to_the_canvas() {
        let (request, processed, _media) = encoding_service(4, ReferenceConfig::new(&REFERENCE_LIMITS))
            .process(
                "keyframes",
                &request(serde_json::json!({
                    "model": "minimax_h3",
                    "prompt": "a fox",
                    "task": "fl2va",
                    "conditions": [
                        {"type": "image", "uri": data_uri(&png(160, 90)), "role": "keyframe", "frame_index": 0},
                        {"type": "image", "uri": data_uri(&png(90, 90)), "role": "keyframe", "frame_index": -1},
                    ],
                    "target": {"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 5.0},
                })),
                1 << 16,
            )
            .await
            .unwrap();
        let canvas = processed.canvas;
        let planned_request = planned(&request);
        let [first, last] = planned_request.conditions.as_slice() else {
            panic!("two keyframes");
        };
        assert_eq!(first.role, ConditionRole::FirstFrame);
        assert_eq!(
            first.media,
            ConditionMedia::Image(ImageFit {
                resized: canvas,
                left: 0,
                top: 0,
                size: canvas,
            })
        );
        assert_eq!(last.role, ConditionRole::LastFrame);
        // A square image covers the 1344x768 canvas at 1344x1344, centred.
        assert_eq!(
            last.media,
            ConditionMedia::Image(ImageFit {
                resized: Canvas {
                    width: 1344,
                    height: 1344,
                },
                left: 0,
                top: 288,
                size: canvas,
            })
        );
        assert_eq!(first.latent_units, [1008]);
        assert_eq!(last.latent_units, [1008]);
        assert!(first.vision.is_some() && last.vision.is_some());
    }

    /// The head prepares a reference request whatever rows its conditions
    /// take: the rank that denoises it counts them with the denoiser's own
    /// size rule and refuses what its provisioned capacity cannot hold.
    #[tokio::test]
    async fn the_head_leaves_condition_rows_to_the_denoising_rank() {
        let body = request(serde_json::json!({
            "model": "minimax_h3",
            "prompt": "a fox",
            "task": "ref2va",
            "conditions": [{"type": "image", "uri": data_uri(&png(160, 90)), "role": "reference"}],
            "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
        }));
        let (request, _, _media) = service().process("rows", &body, 1 << 16).await.unwrap();
        assert_eq!(planned(&request).conditions[0].latent_units, [7296]);
    }

    /// A request that breaks a request-only rule is rejected before any of
    /// its media is fetched: the duration is refused, not the condition
    /// whose URL could never be fetched.
    #[tokio::test]
    async fn request_rules_apply_before_fetching() {
        let body = request(serde_json::json!({
            "model": "minimax_h3",
            "prompt": "a fox",
            "task": "ref2va",
            "conditions": [{"type": "image", "uri": "data:,", "role": "reference"}],
            "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 3.0},
        }));
        let Err(ServeError::Video(error)) = service().process("rules", &body, 1 << 16).await else {
            panic!("the request is refused");
        };
        assert_eq!(error.field(), Some(RequestField::TargetDuration));
    }

    /// The checkpoint's own tokenizer, loaded as the server loads it
    /// (`UNISERVE_MINIMAX_H3_MODEL` names the checkpoint root), encodes every
    /// text the vectors present to the reference's token ids and resolves
    /// the vision markers and placeholders to the reference's ids.
    #[tokio::test]
    async fn checkpoint_tokenizer_matches_the_reference() {
        use models::video::presentation::Tokenizer as _;

        let Ok(root) = std::env::var("UNISERVE_MINIMAX_H3_MODEL") else {
            eprintln!("UNISERVE_MINIMAX_H3_MODEL is not set; skipping the tokenizer parity test");
            return;
        };
        let pipeline = PipelineCheckpoint::resolve(&root).await.unwrap().unwrap();
        let tokenizer = pipeline_tokenizer(&pipeline).await.unwrap();
        let fixture = fixture();
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
                let ids: Vec<u64> = models::video::presentation::Tokenizer::encode(
                    &tokenizer,
                    text.as_str().unwrap(),
                )
                .unwrap()
                .into_iter()
                .map(u64::from)
                .collect();
                assert_eq!(Value::from(ids), segment["token_ids"], "{}", case["name"]);
            }
        }
    }
}
