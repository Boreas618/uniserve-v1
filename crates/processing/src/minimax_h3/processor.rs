//! The deployment's MiniMax-H3 request processor.
//!
//! A request states references and a size; MiniMax-H3 serves three tasks, and the references say
//! which ([`VideoTask`]): none is `t2va`, image keyframes alone (`frame_index` 0 or -1) are
//! `fl2va`, and any reference without a frame index is `ref2va`. A size selects one of the canvases
//! the deployment serves; without one, an `fl2va` request takes the canvas rule over its first
//! keyframe and every other request the deployment's first canvas. A duration
//! aligns up to the `17n + 5` frame rule, and a frame count must already be one of it. Every
//! refusal names the public request parameter at fault.

use scheduling::processing::video::Canvas;
use serde_json::{Value, json};

use super::prompt::present;
use super::rules::{KEYFRAME_ANCHORS, REFERENCES, ReferenceSizing};
use super::{
    AUDIO_SAMPLE_RATE, CANVAS_RULE, DenoiserInfo, FPS, FRAMES, MAX_SECONDS, MIN_SECONDS,
    PROCESSOR_FILES, VideoTask,
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
    /// The canvas of a request that states no size: the first the deployment serves.
    pub(super) default_canvas: Canvas,
    /// The deployment's longest output, in seconds (`video_config.max_seconds`).
    pub(super) max_seconds: f64,
    /// How the deployment sizes references.
    pub(super) sizing: ReferenceSizing,
    /// The conditioner's vision processor.
    pub(super) vision: ProcessorConfig,
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
        let denoiser = DenoiserInfo::decode(binding.description)
            .map_err(|error| format!("the video worker's denoiser: {error}"))?;
        let default_canvas = denoiser
            .canvases
            .first()
            .map(|served| served.canvas)
            .ok_or_else(|| "the video worker's denoiser serves no canvas".to_owned())?;

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
            default_canvas,
            max_seconds: binding.max_seconds,
            sizing,
            vision,
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
            .denoiser
            .tasks
            .iter()
            .map(|task| {
                let rules = match task {
                    VideoTask::TextToVideoAudio => json!({"input_references": []}),
                    VideoTask::FirstLastToVideoAudio => json!({
                        "keyframe_frame_indices": KEYFRAME_ANCHORS,
                    }),
                    VideoTask::ReferenceToVideoAudio => json!({
                        "max_images": limits.images,
                        "max_videos": limits.videos,
                        "max_audios": limits.audios,
                        "max_references": limits.total,
                        "keyframe_frame_indices": std::iter::once(&[][..])
                            .chain(KEYFRAME_ANCHORS)
                            .collect::<Vec<&[i64]>>(),
                        "reference_image_short_edge": self.sizing.image_short_edge,
                        "enlarge_references": self.sizing.enlarge,
                    }),
                };
                (task.as_str().to_owned(), rules)
            })
            .collect();
        json!({
            "tasks": self.denoiser.tasks.iter().map(|task| task.as_str()).collect::<Vec<_>>(),
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
            return Err(InputError::Capacity {
                param: REFERENCES.to_owned(),
                rows: condition_rows,
                capacity: packing.capacity,
            });
        }

        let prompt = present(vocabulary, &generation, &request.prompt)?;
        let conditions = generation
            .conditions
            .iter()
            .map(|condition| condition.describe(generation.canvas))
            .collect();

        Ok(VideoInput {
            size: Canvas {
                width: generation.canvas.width,
                height: generation.canvas.height,
            },
            num_frames: generation.num_frames,
            fps: FPS,
            seed: request.seed,
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
    use super::super::{
        ConditionPacking, DescriptionError, ReferenceLimits, ServedCanvas, vectors,
    };
    use super::*;
    use crate::video::{Length, Reference, ReferenceKind};

    /// A base denoiser serving `tasks` at 768p 16:9 and 9:16, on a 9-point schedule.
    fn description(tasks: &[VideoTask]) -> DenoiserInfo {
        DenoiserInfo {
            tasks: tasks.to_vec(),
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
            condition_packing: ConditionPacking::dense(0),
        }
    }

    /// The processor of a deployment of [`description`] serving `tasks` for at most
    /// `max_seconds`.
    fn processor(tasks: &[VideoTask], max_seconds: f64) -> Result<H3Processor, String> {
        let description = description(tasks).encode();
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
            seed: 7,
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

    /// A duration aligns up to the frame rule, a size selects its served canvas, and the seed is
    /// the request's; the prompt is the conditioner prompt's tokens.
    #[test]
    fn a_text_request_resolves_its_served_canvas_and_aligned_frames() {
        let processor = processor(&[VideoTask::TextToVideoAudio], 15.0).unwrap();
        let request = spec(Some((768, 1344)), Some(Length::Seconds(5.0)));
        processor.check(&request).unwrap();
        let input = processor
            .process(&request, &[], &CharacterTokenizer)
            .unwrap();
        assert_eq!((input.size.width, input.size.height), (768, 1344));
        assert_eq!(input.num_frames, 124, "5 s at 24 fps aligns up to 17n + 5");
        assert_eq!(input.fps, 24);
        assert_eq!(input.seed, 7);
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
        let processor = processor(&[VideoTask::TextToVideoAudio], 15.0).unwrap();
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

    /// A request whose conditions take more denoiser rows than the deployment serves is refused
    /// with the rows and the capacity, for the daemon to name its option.
    #[test]
    fn conditions_beyond_the_capacity_are_refused_as_such() {
        let processor = processor(&[VideoTask::FirstLastToVideoAudio], 15.0).unwrap();
        let mut keyframed = spec(Some((1344, 768)), Some(Length::Seconds(5.0)));
        keyframed.references.push(Reference {
            kind: ReferenceKind::Image,
            frame_index: Some(0),
            start_time_seconds: None,
        });
        let keyframe = [MediaMetadata::Image(crate::ImageMetadata {
            width: 1344,
            height: 768,
        })];
        processor.check(&keyframed).unwrap();
        let refused = processor
            .process(&keyframed, &keyframe, &CharacterTokenizer)
            .unwrap_err();
        assert_eq!(
            refused,
            InputError::Capacity {
                param: "input_references".to_owned(),
                rows: 42 * 24,
                capacity: 0,
            }
        );
    }

    /// A deployment's longest output must lie within the model's own bounds.
    #[test]
    fn a_deployment_beyond_the_models_duration_is_refused() {
        assert!(processor(&[VideoTask::TextToVideoAudio], MAX_SECONDS + 1.0).is_err());
        assert!(processor(&[VideoTask::TextToVideoAudio], MIN_SECONDS - 1.0).is_err());
    }

    /// A description round-trips through its encoding; one that names a task the family does
    /// not have, or prepared no canvas, is refused.
    #[test]
    fn a_denoiser_description_round_trips_and_is_validated() {
        let described = description(&[VideoTask::TextToVideoAudio]);
        assert_eq!(
            DenoiserInfo::decode(&described.encode()),
            Ok(described.clone())
        );
        assert_eq!(described.denoise_steps(), 8);

        let mut message: crate::minimax_h3::wire::VideoDenoiser =
            prost::Message::decode(described.encode().as_slice()).unwrap();
        message.tasks = vec!["t3va".to_owned()];
        assert_eq!(
            DenoiserInfo::decode(&prost::Message::encode_to_vec(&message)),
            Err(DescriptionError::Task("t3va".to_owned()))
        );

        let bare = DenoiserInfo {
            canvases: Vec::new(),
            ..described
        };
        assert!(matches!(
            DenoiserInfo::decode(&bare.encode()),
            Err(DescriptionError::Invalid(_))
        ));
    }
}
