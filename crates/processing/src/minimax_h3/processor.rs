//! The deployment's MiniMax-H3 request processor.
//!
//! A request states references and a size; MiniMax-H3 serves three tasks, and the references say
//! which ([`VideoTask`]): none is `t2va`, image keyframes alone (`frame_index` 0 or -1) are
//! `fl2va`, and any reference without a frame index is `ref2va`. A size selects one of the canvases
//! the deployment serves; without one, an `fl2va` request takes the canvas rule over its first
//! keyframe and every other request the deployment's first canvas's short edge at 16:9. A duration
//! aligns up to the `17n + 5` frame rule, and a frame count must already be one of it. Every
//! refusal names the public request parameter at fault.

use proto::video::Canvas;
use serde_json::{Value, json};

use super::prompt::present;
use super::rules::{REFERENCES, ReferenceSizing};
use super::{
    AUDIO_SAMPLE_RATE, CANVAS_RULE, DEFAULT_SEED, DenoiserInfo, FPS, FRAMES, MAX_SECONDS,
    MIN_SECONDS, PROCESSOR_FILES, VideoTask,
};
use crate::media::MediaMetadata;
use crate::qwen_vl::ProcessorConfig;
use crate::video::{InputError, VideoBinding, VideoInput, VideoProcessor, VideoSpec};
use crate::vocabulary::Vocabulary;

/// A deployment's MiniMax-H3 processor: what its placed denoiser serves and how it sizes
/// references.
pub struct H3Processor {
    /// What the placed denoiser serves, as its workers described it.
    pub(super) denoiser: DenoiserInfo,
    /// The denoiser's tasks, read from its description.
    pub(super) tasks: Vec<VideoTask>,
    /// The deployment's longest output, in seconds (`video_config.max_seconds`).
    pub(super) max_seconds: f64,
    /// How the deployment sizes references.
    pub(super) sizing: ReferenceSizing,
    /// The conditioner's vision processor.
    pub(super) vision: ProcessorConfig,
    /// Condition units one latent encoding round covers while a request's prompt is encoded,
    /// which reference images are banded to fill.
    pub(super) latent_encoding_units: u32,
}

impl H3Processor {
    /// Binds the processor of a deployment whose workers described its denoiser in
    /// `binding.description`.
    ///
    /// # Errors
    ///
    /// Fails when the deployment's longest output lies outside the model's bounds, the description
    /// is not a MiniMax-H3 denoiser's or names a task this model does not have, the processor
    /// options are refused, or the processor files are not the conditioner's.
    pub fn bind(binding: VideoBinding<'_>) -> Result<Self, String> {
        if !(MIN_SECONDS..=MAX_SECONDS).contains(&binding.max_seconds) {
            return Err(format!(
                "`video_config.max_seconds` must lie in [{MIN_SECONDS}, {MAX_SECONDS}] for \
                 MiniMax-H3, got {}",
                binding.max_seconds
            ));
        }
        let denoiser = DenoiserInfo::decode(binding.description).map_err(|error| {
            format!("the video worker described its denoiser unreadably: {error}")
        })?;
        denoiser
            .validate()
            .map_err(|message| format!("the video worker's denoiser: {message}"))?;
        let tasks = denoiser
            .tasks
            .iter()
            .map(|name| {
                VideoTask::from_name(name)
                    .ok_or_else(|| format!("the worker serves unknown video task {name:?}"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let sizing = ReferenceSizing::from_options(binding.processor_options, &denoiser.references)
            .map_err(|message| format!("invalid mm_processor_kwargs: {message}"))?;

        let [image, video] = binding.files else {
            return Err(format!(
                "the conditioner's processor is {} files, not {}",
                PROCESSOR_FILES.len(),
                binding.files.len()
            ));
        };
        let vision = ProcessorConfig::parse(image, video)?;

        Ok(Self {
            denoiser,
            tasks,
            max_seconds: binding.max_seconds,
            sizing,
            vision,
            latent_encoding_units: binding.latent_encoding_units.max(1),
        })
    }

    /// Refuses the request options MiniMax-H3 does not take or that this deployment's denoiser
    /// fixes: guidance, extra parameters, another frame rate or schedule length.
    fn check_options(&self, request: &VideoSpec) -> Result<(), InputError> {
        if request.negative_prompt.is_some() || request.guidance_scale.is_some() {
            let param = if request.negative_prompt.is_some() {
                "negative_prompt"
            } else {
                "guidance_scale"
            };
            return Err(InputError::invalid(
                param,
                "MiniMax-H3 runs a distilled schedule without classifier-free guidance",
            ));
        }
        if let Some(name) = request.extra_params.keys().next() {
            return Err(InputError::invalid(
                format!("extra_params.{name}"),
                "MiniMax-H3 takes no extra parameter",
            ));
        }
        if let Some(fps) = request.fps
            && fps != FPS
        {
            return Err(InputError::invalid(
                "fps",
                format!("MiniMax-H3 generates {FPS} frames per second"),
            ));
        }
        if let Some(steps) = request.num_inference_steps
            && steps != self.denoise_steps()
        {
            return Err(InputError::invalid(
                "num_inference_steps",
                format!(
                    "this deployment's schedule runs {} denoiser evaluations",
                    self.denoise_steps()
                ),
            ));
        }
        Ok(())
    }
}

impl VideoProcessor for H3Processor {
    fn capabilities(&self) -> Value {
        let limits = &self.denoiser.references;
        let task_references: serde_json::Map<String, Value> = self
            .tasks
            .iter()
            .map(|task| {
                let rules = match task {
                    VideoTask::TextToVideoAudio => json!({"input_references": []}),
                    VideoTask::FirstLastToVideoAudio => json!({
                        "keyframe_frame_indices": [[0], [-1], [0, -1]],
                    }),
                    VideoTask::ReferenceToVideoAudio => json!({
                        "max_images": limits.images,
                        "max_videos": limits.videos,
                        "max_audios": limits.audios,
                        "max_references": limits.total,
                        "keyframe_frame_indices": [[], [0], [-1], [0, -1]],
                        "reference_image_short_edge": self.sizing.image_short_edge,
                        "enlarge_references": self.sizing.enlarge,
                    }),
                };
                (task.as_str().to_owned(), rules)
            })
            .collect();
        json!({
            "tasks": self.tasks.iter().map(|task| task.as_str()).collect::<Vec<_>>(),
            "task_references": task_references,
            "sizes": self.served_sizes(),
            "canvas_rule": {
                "short_edge": CANVAS_RULE.short_edge,
                "multiple": CANVAS_RULE.multiple,
                "max_pixels": CANVAS_RULE.max_pixels,
            },
            "fps": FPS,
            "audio_sample_rate": AUDIO_SAMPLE_RATE,
            "frames": FRAMES,
            "min_seconds": MIN_SECONDS,
            "max_seconds": self.max_seconds,
            "model_max_seconds": MAX_SECONDS,
            "num_inference_steps": self.denoise_steps(),
            "schedule": {
                "flow_shift": self.denoiser.video_shift,
                "audio_flow_shift": self.denoiser.audio_shift,
            },
        })
    }

    fn denoise_steps(&self) -> u32 {
        self.denoiser.denoise_steps()
    }

    fn check(&self, request: &VideoSpec) -> Result<(), InputError> {
        self.check_options(request)?;
        self.check_request(request)
    }

    fn process(
        &self,
        request: &VideoSpec,
        media: &[MediaMetadata],
        vocabulary: &dyn Vocabulary,
    ) -> Result<VideoInput, InputError> {
        self.check_options(request)?;
        let generation = self.prepare(request, media)?;

        let packing = self.denoiser.condition_packing;
        let condition_rows = generation.condition_rows(&packing);
        if condition_rows > packing.capacity {
            return Err(InputError::invalid(
                REFERENCES,
                format!(
                    "the references take {condition_rows} denoiser rows, more than the {} this \
                     deployment serves; serve with a larger video_config.max_condition_rows",
                    packing.capacity
                ),
            ));
        }

        let prompt = present(vocabulary, &generation, &request.prompt)?;
        let image_bands = generation.image_bands(self.latent_encoding_units);
        let conditions = generation
            .conditions
            .iter()
            .zip(image_bands)
            .map(|(condition, bands)| condition.describe(generation.canvas, bands))
            .collect();

        Ok(VideoInput {
            size: Canvas {
                width: generation.canvas.width,
                height: generation.canvas.height,
            },
            num_frames: generation.num_frames,
            fps: FPS,
            seed: request.seed.unwrap_or(DEFAULT_SEED),
            prompt_token_ids: prompt.token_ids,
            vision_spans: prompt.vision_spans,
            conditions,
            // One decode unit per clip of the frame rule.
            video_units: FRAMES.units(generation.num_frames),
            condition_rows,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Map;

    use super::super::prompt::tests::CharacterTokenizer;
    use super::super::{ConditionPacking, ReferenceLimits, ServedCanvas, vectors};
    use super::*;
    use crate::video::{Length, Reference, ReferenceKind};

    /// A base denoiser serving `tasks` at 768p 16:9 and 9:16, on a 9-point schedule.
    fn processor(tasks: &[&str], max_seconds: f64) -> Result<H3Processor, String> {
        let description = DenoiserInfo {
            tasks: tasks.iter().map(|task| (*task).to_owned()).collect(),
            schedule_points: 9,
            video_shift: 12.0,
            audio_shift: 3.0,
            canvases: vec![
                ServedCanvas {
                    short_edge: 768,
                    aspect_ratio: "16:9".to_owned(),
                    canvas: Canvas {
                        width: 1344,
                        height: 768,
                    },
                },
                ServedCanvas {
                    short_edge: 768,
                    aspect_ratio: "9:16".to_owned(),
                    canvas: Canvas {
                        width: 768,
                        height: 1344,
                    },
                },
            ],
            references: ReferenceLimits::default(),
            condition_packing: ConditionPacking::default(),
        }
        .encode()
        .unwrap();
        // The conditioner's two processor files, as the checkpoint publishes them.
        let vectors = vectors();
        let vision = &vectors["vision"];
        let file = |kind: &str| {
            json!({
                "patch_size": vision["patch_size"],
                "temporal_patch_size": vision["temporal_patch_size"],
                "merge_size": vision["merge_size"],
                "size": {
                    "shortest_edge": vision[format!("{kind}_min_pixels")],
                    "longest_edge": vision[format!("{kind}_max_pixels")],
                },
            })
            .to_string()
        };
        H3Processor::bind(VideoBinding {
            description: &description,
            latent_encoding_units: 1,
            max_seconds,
            processor_options: &Map::new(),
            files: &[file("image"), file("video")],
        })
    }

    fn spec(size: Option<(u32, u32)>, length: Option<Length>) -> VideoSpec {
        VideoSpec {
            prompt: "a river".to_owned(),
            negative_prompt: None,
            size: size.map(|(width, height)| Canvas { width, height }),
            length,
            fps: None,
            seed: None,
            num_inference_steps: None,
            guidance_scale: None,
            references: Vec::new(),
            extra_params: Map::new(),
        }
    }

    fn refused(result: Result<(), InputError>) -> String {
        match result {
            Err(InputError::Invalid { param, .. }) => param,
            other => panic!("not refused: {other:?}"),
        }
    }

    /// A duration aligns up to the frame rule, a size selects its served canvas, and an omitted
    /// seed is the reference's; the prompt is the conditioner prompt's tokens.
    #[test]
    fn a_text_request_resolves_its_served_canvas_and_aligned_frames() {
        let processor = processor(&["t2va"], 15.0).unwrap();
        let request = spec(Some((768, 1344)), Some(Length::Seconds(5.0)));
        processor.check(&request).unwrap();
        let input = processor
            .process(&request, &[], &CharacterTokenizer)
            .unwrap();
        assert_eq!((input.size.width, input.size.height), (768, 1344));
        assert_eq!(input.num_frames, 124, "5 s at 24 fps aligns up to 17n + 5");
        assert_eq!(input.fps, 24);
        assert_eq!(input.seed, DEFAULT_SEED);
        assert_eq!(input.video_units, FRAMES.units(124));
        assert!(input.conditions.is_empty());
        assert!(!input.prompt_token_ids.is_empty());
        assert!(input.vision_spans.is_empty());

        let frames = spec(None, Some(Length::Frames(124)));
        assert_eq!(
            processor
                .process(&frames, &[], &CharacterTokenizer)
                .unwrap()
                .num_frames,
            124,
            "a legal frame count is the count generated"
        );
    }

    /// What the model does not serve is refused by the parameter at fault, before any media is
    /// fetched.
    #[test]
    fn unserved_requests_are_refused_by_parameter() {
        let processor = processor(&["t2va"], 15.0).unwrap();
        let seconds = Some(Length::Seconds(5.0));
        assert_eq!(
            refused(processor.check(&spec(Some((1000, 1000)), seconds))),
            "size"
        );
        assert_eq!(
            refused(processor.check(&spec(None, Some(Length::Frames(100))))),
            "num_frames"
        );
        let mut fps = spec(None, seconds);
        fps.fps = Some(30);
        assert_eq!(refused(processor.check(&fps)), "fps");
        let mut guided = spec(None, seconds);
        guided.guidance_scale = Some(5.0);
        assert_eq!(refused(processor.check(&guided)), "guidance_scale");

        // Keyframes alone are `fl2va`, which this text-to-video deployment does not serve.
        let mut keyframed = spec(None, seconds);
        keyframed.references.push(Reference {
            kind: ReferenceKind::Image,
            frame_index: Some(0),
            start_time_seconds: None,
        });
        assert_eq!(refused(processor.check(&keyframed)), "input_references");
    }

    /// A deployment's longest output must lie within the model's own bounds.
    #[test]
    fn a_deployment_beyond_the_models_duration_is_refused() {
        assert!(processor(&["t2va"], MAX_SECONDS + 1.0).is_err());
        assert!(processor(&["t2va"], MIN_SECONDS - 1.0).is_err());
        assert!(
            processor(&["t3va"], MAX_SECONDS).is_err(),
            "an unknown task"
        );
    }
}
