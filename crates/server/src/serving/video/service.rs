//! The MiniMax-H3 video service of one deployment.
//!
//! [`VideoService`] binds what the deployment's denoiser serves, as the
//! worker reports it in its startup handshake ([`VideoDenoiserInfo`]), to the
//! server's media policy and the checkpoint's vision processor. It prepares
//! every video request for the engine: the processor ([`VideoProcessor`])
//! plans and presents the request, each condition's fetched media is
//! published to
//! shared memory for the worker's media reader, and the result becomes one
//! `DiffusionRequest` describing every condition. The rank that denoises the
//! request admits its conditions into the condition capacity it provisioned,
//! counting their rows as its denoiser packs them.
//! [`VideoService::capabilities`] describes the same contract for
//! `GET /v1/capabilities`.

use std::sync::Arc;

use serde_json::{Value, json};
use uniserve_core::video::plan::{
    CANVAS_MAX_PIXELS, CANVAS_MULTIPLE, ConditionSpec, MAX_VIDEO_SECONDS, MIN_VIDEO_SECONDS,
    PlanLimits, ProcessorConfig, ReferenceConfig, Target,
};
use uniserve_core::{
    Canvas, DiffusionRequest, DiffusionSamplingParams, MediaSource, RequestId, VideoTask,
};
use uniserve_engine::VideoDenoiserInfo;

use super::probe::{MediaProber, ProbeConfig};
use super::sources::{MediaFetcher, MediaLimits, MediaPolicy, RemoteMediaPolicy};
use super::{ConditionInput, VideoInputs, VideoProcessor};
use crate::openai::{ApiError, VideoCondition, VideoGenerationRequest};
use crate::options::MediaOptions;
use crate::profile::tokenizer::DynTokenizer;
use crate::serving::{ServeError, ServeRequestId};

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

/// Processes the video requests of one deployment.
pub struct VideoService {
    denoiser: VideoDenoiserInfo,
    /// Condition units one latent encoding round covers while a request's
    /// prompt is encoded, which reference images are banded to fill
    /// (`RequestPlan::image_bands`).
    latent_encoding_lane: u32,
    media: MediaOptions,
    processor: VideoProcessor,
}

impl VideoService {
    /// Binds the served denoiser, the checkpoint's vision processor, the
    /// deployment's duration capacity, the condition units its latent
    /// encoder covers in one round, its media policy, how it sizes
    /// references, and the tokenizer.
    ///
    /// `denoiser` is the worker handshake the engine validated, and
    /// `references` a sizing `ReferenceConfig::from_kwargs` validated against
    /// it.
    ///
    /// # Errors
    ///
    /// Fails when the handshake names a task this server does not know, or
    /// when the media fetcher cannot be built.
    pub fn new(
        denoiser: VideoDenoiserInfo,
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
        let fetcher = MediaFetcher::new(MediaPolicy {
            media_directory: media.directory.clone(),
            remote: RemoteMediaPolicy {
                enabled: media.allow_remote,
                ..RemoteMediaPolicy::default()
            },
            limits: MediaLimits::new(media.max_request_bytes),
        })?;
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
            processor: VideoProcessor::new(fetcher, prober, vision, limits, tokenizer),
            denoiser,
            latent_encoding_lane: latent_encoding_lane.max(1),
            media: media.clone(),
        })
    }

    /// The served video denoiser, as the engine's handshake validated it.
    pub fn denoiser(&self) -> &VideoDenoiserInfo {
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
        let limits = MediaLimits::new(self.media.max_request_bytes);
        let references = self.processor.limits().references;
        let conditions: serde_json::Map<String, Value> = self
            .processor
            .limits()
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
                "short_edges": self.processor.limits().short_edges(),
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
            "max_seconds": self.processor.limits().max_video_seconds,
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
                "max_request_bytes": limits.total_bytes,
            },
            "request_fields": REQUEST_FIELDS,
        })
    }

    /// Validates, processes and sizes one video request into its engine
    /// request and what it generates.
    ///
    /// The presentation must fit the prompt capacity (`--max-model-len`).
    /// Each condition's fetched media is then copied to shared memory, which
    /// the request holds until the engine retires it. `Engine::submit` stamps
    /// the engine's identifier on the request.
    ///
    /// # Errors
    ///
    /// Returns `invalid_request` naming the field at fault or the prompt
    /// tokens the request needs, and a server error when processing's own
    /// resources or the shared-memory allocation fail.
    pub async fn process(
        &self,
        request_id: &ServeRequestId,
        request: &VideoGenerationRequest,
        max_prompt_tokens: u32,
    ) -> Result<(DiffusionRequest, ProcessedVideo), ApiError> {
        let conditions: Vec<ConditionInput<'_>> =
            request.conditions.iter().map(condition_input).collect();
        let processed = self
            .processor
            .process(&VideoInputs {
                task: request.task,
                prompt: &request.prompt,
                target: Target {
                    short_edge: request.target.short_edge,
                    aspect_ratio: &request.target.aspect_ratio,
                    duration_seconds: request.target.duration_seconds,
                },
                conditions,
            })
            .await?;
        let plan = &processed.plan;

        let prompt_token_ids = processed.presentation.token_ids;
        if prompt_token_ids.len() > max_prompt_tokens as usize {
            return Err(crate::openai::ApiError::from(
                ServeError::ContextLengthExceeded {
                    request_id: request_id.clone(),
                    prompt_tokens: prompt_token_ids.len(),
                    max_tokens: max_prompt_tokens,
                },
            ));
        }

        // Each condition's media reaches the worker's media reader through
        // shared memory on this host.
        let image_bands = plan.image_bands(self.latent_encoding_lane);
        let mut media = Vec::with_capacity(processed.media.len());
        let mut conditions = Vec::with_capacity(processed.media.len());
        for ((condition, fetched), bands) in plan
            .conditions
            .iter()
            .zip(&processed.media)
            .zip(image_bands)
        {
            let source = MediaSource::publish(fetched.bytes()).map_err(|error| {
                ApiError::server_error(format!(
                    "publishing conditions[{}] for the worker failed: {error}",
                    condition.index
                ))
            })?;
            conditions.push(condition.describe(plan.canvas, source.locator(), bands));
            media.push(Arc::new(source));
        }
        let request = DiffusionRequest {
            request_id: RequestId::default(),
            task: plan.task,
            prompt_token_ids,
            text_tags: processed.presentation.tags,
            conditions,
            media,
            priority: 0,
            sampling: DiffusionSamplingParams {
                num_frames: plan.num_frames,
                // One decode unit per clip of the denoiser's frame rule.
                video_units: self.denoiser.frames.units(plan.num_frames),
                seed: request.seed,
                width: plan.canvas.width,
                height: plan.canvas.height,
            },
        };
        let video = ProcessedVideo {
            duration_seconds: plan.duration_seconds,
            actual_seconds: f64::from(plan.num_frames) / f64::from(self.denoiser.frame_rate),
            num_frames: plan.num_frames,
            canvas: plan.canvas,
        };
        Ok((request, video))
    }
}

/// The processing input of one request condition.
fn condition_input(condition: &VideoCondition) -> ConditionInput<'_> {
    ConditionInput {
        spec: ConditionSpec {
            condition_type: condition.condition_type,
            role: condition.role,
            frame_index: condition.frame_index,
            start_seconds: condition.start_time_seconds,
        },
        uri: &condition.uri,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use base64::Engine as _;
    use image::ImageEncoder as _;
    use uniserve_core::ReferenceLimits;
    use uniserve_core::video::plan::ReferenceConfig;

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
    use uniserve_core::{
        Canvas, ConditionMedia, ConditionRole, ConditionVision, ImageFit, VideoTask, VisionGrid,
    };
    use uniserve_engine::VideoDenoiserInfo;

    use super::super::tests::{character_tokenizer, served_canvases, vision};
    use super::VideoService;
    use crate::openai::VideoGenerationRequest;
    use crate::options::MediaOptions;
    use crate::serving::ServeRequestId;

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
            VideoDenoiserInfo {
                tasks: ["t2va", "fl2va", "ref2va"].map(str::to_owned).to_vec(),
                schedule_points: 50,
                video_shift: 12.0,
                audio_shift: 3.0,
                frame_rate: 24,
                audio_sample_rate: 32_000,
                frames: uniserve_core::FrameClips {
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
        let (unenlarged, _) = encoding_service(4, ReferenceConfig::new(&REFERENCE_LIMITS))
            .process(
                &ServeRequestId::new("reference"),
                &request(body.clone()),
                1 << 16,
            )
            .await
            .unwrap();
        // 90 is 2.8 patches and rounds to 96.
        let own = Canvas {
            width: 160,
            height: 96,
        };
        assert_eq!(
            unenlarged.conditions[0].media,
            ConditionMedia::Image(ImageFit {
                resized: own,
                left: 0,
                top: 0,
                size: own,
            })
        );

        let (request, _) = encoding_service(4, ReferenceConfig::released(&REFERENCE_LIMITS))
            .process(&ServeRequestId::new("reference"), &request(body), 1 << 16)
            .await
            .unwrap();
        assert_eq!(request.task, VideoTask::Ref2va);
        assert_eq!(request.text_tags.len(), request.prompt_token_ids.len());
        assert_eq!(request.validate(), Ok(()));

        let [condition] = request.conditions.as_slice() else {
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
            let (request, _) = encoding_service(4, ReferenceConfig::released(&REFERENCE_LIMITS))
            .process(
                &ServeRequestId::new("images"),
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
            for (index, condition) in request.conditions.iter().enumerate() {
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
        let (request, processed) = encoding_service(4, ReferenceConfig::new(&REFERENCE_LIMITS))
            .process(
                &ServeRequestId::new("keyframes"),
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
        let [first, last] = request.conditions.as_slice() else {
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
        let (request, _) = service()
            .process(&ServeRequestId::new("rows"), &body, 1 << 16)
            .await
            .unwrap();
        assert_eq!(request.conditions[0].latent_units, [7296]);
    }
}
