//! The deployment's MiniMax-H3 request processor: the model-agnostic request translated into the
//! model's own terms, held to its rules and planned.
//!
//! A request states references and a size; MiniMax-H3 serves three tasks, and the references say
//! which ([`VideoTask`]): none is `t2va`, image keyframes alone (`frame_index` 0 or -1) are
//! `fl2va`, and any reference without a frame index is `ref2va`. A size selects one of the canvases
//! the deployment serves; without one, an `fl2va` request takes the canvas rule over its first
//! keyframe and every other request the deployment's first canvas's short edge at 16:9. A
//! duration aligns up to the `17 * n + 5` frame rule, and a frame count must already be one of it.

use processing::{
    InputError, Length, MediaMetadata, ReferenceKind, VideoBinding, VideoPlan, VideoProcessor,
    VideoSpec, Vocabulary,
};
use proto::video::Canvas;
use serde_json::{Value, json};

use super::plan::{
    self, CANVAS_MAX_PIXELS, CANVAS_MULTIPLE, CANVAS_SHORT_EDGE, ConditionSpec, ConditionType,
    MAX_VIDEO_SECONDS, MIN_VIDEO_SECONDS, PlanLimits, ProcessorConfig, ReferenceConfig,
    RequestField, Role, Target, VideoInputError,
};
use super::presentation::{self, VIDEO_TAG};
use super::{ConditionPacking, DenoiserInfo};
use crate::VideoTask;

/// The seed a request without one uses, as the reference does.
pub const DEFAULT_SEED: u64 = 42;

/// The checkpoint files the processor reads: the conditioner's Qwen3-VL processor, which fixes how
/// condition media is patched.
pub const PROCESSOR_FILES: &[(&str, &str)] = &[
    ("processor", "preprocessor_config.json"),
    ("processor", "video_preprocessor_config.json"),
];

/// A deployment's MiniMax-H3 processor.
pub struct H3Processor {
    /// What the placed denoiser serves, as its workers described it.
    denoiser: DenoiserInfo,
    /// What the deployment serves, which every request is planned against.
    limits: PlanLimits,
    /// The conditioner's vision processor.
    vision: ProcessorConfig,
    /// Condition units one latent encoding round covers while a request's prompt is encoded,
    /// which reference images are banded to fill.
    latent_encoding_units: u32,
    /// How the denoiser packs a request's conditions, and how many rows they may take.
    condition_packing: ConditionPacking,
}

impl H3Processor {
    /// Binds the processor of a deployment whose workers described its denoiser in
    /// `binding.description`.
    ///
    /// # Errors
    ///
    /// Fails when the description is not a MiniMax-H3 denoiser's, names a task this model does not
    /// have, the processor options are refused, or the processor files are not the conditioner's.
    pub fn bind(binding: VideoBinding<'_>) -> Result<Self, String> {
        if !(MIN_VIDEO_SECONDS..=MAX_VIDEO_SECONDS).contains(&binding.max_seconds) {
            return Err(format!(
                "`video_config.max_seconds` must lie in [{MIN_VIDEO_SECONDS}, \
                 {MAX_VIDEO_SECONDS}] for MiniMax-H3, got {}",
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

        // The processor's options are its reference sizing, within what the denoiser declares.
        let references =
            ReferenceConfig::from_kwargs(binding.processor_options, &denoiser.references)
                .map_err(|message| format!("invalid mm_processor_kwargs: {message}"))?;

        let [image, video] = binding.files else {
            return Err(format!(
                "the conditioner's processor is {} files, not {}",
                PROCESSOR_FILES.len(),
                binding.files.len()
            ));
        };
        let vision = ProcessorConfig::parse(image, video).map_err(|error| error.to_string())?;

        let limits = PlanLimits {
            tasks,
            max_video_seconds: binding.max_seconds,
            frame_rate: denoiser.frame_rate,
            frames: denoiser.frames,
            canvases: denoiser.canvases.clone(),
            reference_limits: denoiser.references,
            references,
        };
        Ok(Self {
            condition_packing: denoiser.condition_packing,
            denoiser,
            limits,
            vision,
            latent_encoding_units: binding.latent_encoding_units.max(1),
        })
    }

    /// The request in the model's own terms: its task, its target and its conditions.
    fn translate(&self, request: &VideoSpec) -> Result<Translated, InputError> {
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
            && fps != self.denoiser.frame_rate
        {
            return Err(InputError::invalid(
                "fps",
                format!(
                    "MiniMax-H3 generates {} frames per second",
                    self.denoiser.frame_rate
                ),
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

        let specs: Vec<ConditionSpec> = request
            .references
            .iter()
            .map(|reference| ConditionSpec {
                condition_type: match reference.kind {
                    ReferenceKind::Image => ConditionType::Image,
                    ReferenceKind::Video => ConditionType::Video,
                    ReferenceKind::Audio => ConditionType::Audio,
                },
                role: if reference.frame_index.is_some() {
                    Role::Keyframe
                } else {
                    Role::Reference
                },
                frame_index: reference.frame_index,
                start_seconds: reference.start_time_seconds,
            })
            .collect();
        let task = if specs.iter().any(|spec| spec.role == Role::Reference) {
            VideoTask::ReferenceToVideoAudio
        } else if specs.is_empty() {
            VideoTask::TextToVideoAudio
        } else {
            VideoTask::FirstLastToVideoAudio
        };

        let (short_edge, aspect_ratio) = match request.size {
            Some(size) => {
                let served = self
                    .denoiser
                    .canvases
                    .iter()
                    .find(|served| served.canvas == size)
                    .ok_or_else(|| {
                        InputError::invalid(
                            "size",
                            format!(
                                "{}x{} is not served; this deployment serves {}",
                                size.width,
                                size.height,
                                self.sizes().join(", ")
                            ),
                        )
                    })?;
                (served.short_edge, served.aspect_ratio.clone())
            }
            // Without a size the canvas follows the first keyframe for `fl2va`, and is the
            // deployment's first short edge at 16:9 otherwise.
            None => (
                self.denoiser
                    .canvases
                    .first()
                    .map_or(CANVAS_SHORT_EDGE, |served| served.short_edge),
                "auto".to_owned(),
            ),
        };

        let duration_seconds = match request.length {
            None => None,
            Some(Length::Seconds(seconds)) => Some(seconds),
            Some(Length::Frames(frames)) => {
                let rule = self.denoiser.frames;
                if frames < rule.min_frames || rule.align_up(frames) != frames {
                    return Err(InputError::invalid(
                        "num_frames",
                        format!(
                            "{frames} is not a frame count MiniMax-H3 generates: {} frames and \
                             {} more per clip",
                            rule.min_frames, rule.clip_frames
                        ),
                    ));
                }
                Some(f64::from(frames) / f64::from(self.denoiser.frame_rate))
            }
        };

        Ok(Translated {
            task,
            short_edge,
            aspect_ratio,
            duration_seconds,
            specs,
        })
    }

    /// The sizes the deployment serves, as a request names them.
    fn sizes(&self) -> Vec<String> {
        self.denoiser
            .canvases
            .iter()
            .map(|served| format!("{}x{}", served.canvas.width, served.canvas.height))
            .collect()
    }
}

/// A request in MiniMax-H3's own terms.
struct Translated {
    task: VideoTask,
    short_edge: u32,
    aspect_ratio: String,
    duration_seconds: Option<f64>,
    specs: Vec<ConditionSpec>,
}

impl Translated {
    fn target(&self) -> Target<'_> {
        Target {
            short_edge: self.short_edge,
            aspect_ratio: &self.aspect_ratio,
            duration_seconds: self.duration_seconds,
        }
    }
}

/// A rule's refusal, naming the public request parameter at fault.
fn refused(error: VideoInputError) -> InputError {
    match error {
        VideoInputError::Invalid { field, message } => {
            let param = match field {
                RequestField::Prompt => "prompt".to_owned(),
                RequestField::TargetShortEdge | RequestField::TargetAspectRatio => {
                    "size".to_owned()
                }
                RequestField::TargetDuration => "seconds".to_owned(),
                RequestField::Task | RequestField::Conditions => "input_references".to_owned(),
                RequestField::Condition(index) => format!("input_references[{index}]"),
            };
            InputError::invalid(param, message)
        }
        VideoInputError::Internal(message) => InputError::Internal(message),
    }
}

/// The prompt rows each vision block occupies, as `(first row, rows)`: the runs the presentation
/// tags as vision content.
fn vision_spans(tags: &[u8]) -> Vec<(u32, u32)> {
    let mut spans = Vec::new();
    let mut first = None;
    for (row, tag) in tags.iter().copied().enumerate() {
        let row = row as u32;
        match (tag == VIDEO_TAG, first) {
            (true, None) => first = Some(row),
            (false, Some(start)) => {
                spans.push((start, row - start));
                first = None;
            }
            _ => {}
        }
    }
    if let Some(start) = first {
        spans.push((start, tags.len() as u32 - start));
    }
    spans
}

impl VideoProcessor for H3Processor {
    fn capabilities(&self) -> Value {
        let limits = &self.limits;
        let references = limits.references;
        let task_references: serde_json::Map<String, Value> = limits
            .tasks
            .iter()
            .map(|task| {
                let rules = match task {
                    VideoTask::TextToVideoAudio => json!({"input_references": []}),
                    VideoTask::FirstLastToVideoAudio => json!({
                        "keyframe_frame_indices": [[0], [-1], [0, -1]],
                    }),
                    VideoTask::ReferenceToVideoAudio => json!({
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
            "tasks": limits.tasks.iter().map(|task| task.as_str()).collect::<Vec<_>>(),
            "task_references": task_references,
            "sizes": self.sizes(),
            "canvas_rule": {
                "short_edge": CANVAS_SHORT_EDGE,
                "multiple": CANVAS_MULTIPLE,
                "max_pixels": CANVAS_MAX_PIXELS,
            },
            "fps": self.denoiser.frame_rate,
            "audio_sample_rate": self.denoiser.audio_sample_rate,
            "frames": self.denoiser.frames,
            "min_seconds": MIN_VIDEO_SECONDS,
            "max_seconds": limits.max_video_seconds,
            "model_max_seconds": MAX_VIDEO_SECONDS,
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
        let translated = self.translate(request)?;
        plan::check_request(
            translated.task,
            &translated.target(),
            &translated.specs,
            &self.limits,
        )
        .map_err(refused)
    }

    fn plan(
        &self,
        request: &VideoSpec,
        media: &[MediaMetadata],
        vocabulary: &dyn Vocabulary,
    ) -> Result<VideoPlan, InputError> {
        let translated = self.translate(request)?;
        let planned = plan::plan_request(
            translated.task,
            &translated.target(),
            &translated.specs,
            media,
            &self.vision,
            &self.limits,
        )
        .map_err(refused)?;

        let condition_rows = planned.condition_rows(&self.condition_packing);
        if condition_rows > self.condition_packing.capacity {
            return Err(InputError::invalid(
                "input_references",
                format!(
                    "the references take {condition_rows} denoiser rows, more than the {} this \
                     deployment serves; serve with a larger video_config.max_condition_rows",
                    self.condition_packing.capacity
                ),
            ));
        }

        let presented =
            presentation::present(vocabulary, &planned, &request.prompt).map_err(refused)?;
        let image_bands = planned.image_bands(self.latent_encoding_units);
        let conditions = planned
            .conditions
            .iter()
            .zip(image_bands)
            .map(|(condition, bands)| condition.describe(planned.canvas, bands))
            .collect();

        Ok(VideoPlan {
            size: Canvas {
                width: planned.canvas.width,
                height: planned.canvas.height,
            },
            num_frames: planned.num_frames,
            fps: self.denoiser.frame_rate,
            seed: request.seed.unwrap_or(DEFAULT_SEED),
            vision_spans: vision_spans(&presented.tags),
            prompt_token_ids: presented.token_ids,
            conditions,
            // One decode unit per clip of the frame rule.
            video_units: self.limits.frames.units(planned.num_frames),
            condition_rows,
        })
    }
}

#[cfg(test)]
mod tests {
    use processing::{Reference, VideoBinding};
    use serde_json::Map;

    use super::*;
    use crate::processing::plan::tests::{FRAMES, fixture};
    use crate::processing::presentation::tests::CharacterTokenizer;
    use crate::processing::{ReferenceLimits, ServedCanvas};

    /// A base denoiser serving `tasks` at 768p 16:9 and 9:16, on a 9-point schedule.
    fn processor(tasks: &[&str], max_seconds: f64) -> Result<H3Processor, String> {
        let description = DenoiserInfo {
            tasks: tasks.iter().map(|task| (*task).to_owned()).collect(),
            schedule_points: 9,
            video_shift: 12.0,
            audio_shift: 3.0,
            frame_rate: 24,
            audio_sample_rate: 32_000,
            frames: FRAMES,
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
        let vision = &fixture()["vision"];
        let file = |kind: &str| {
            serde_json::json!({
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
        let (image, video) = (file("image"), file("video"));
        H3Processor::bind(VideoBinding {
            description: &description,
            latent_encoding_units: 1,
            max_seconds,
            processor_options: &Map::new(),
            files: &[image, video],
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
    /// seed is the reference's; the prompt is the presentation's tokens.
    #[test]
    fn a_text_request_plans_its_served_canvas_and_aligned_frames() {
        let processor = processor(&["t2va"], 15.0).unwrap();
        let request = spec(Some((768, 1344)), Some(Length::Seconds(5.0)));
        processor.check(&request).unwrap();
        let plan = processor.plan(&request, &[], &CharacterTokenizer).unwrap();
        assert_eq!((plan.size.width, plan.size.height), (768, 1344));
        assert_eq!(plan.num_frames, 124, "5 s at 24 fps aligns up to 17n + 5");
        assert_eq!(plan.fps, 24);
        assert_eq!(plan.seed, DEFAULT_SEED);
        assert_eq!(plan.video_units, FRAMES.units(124));
        assert!(plan.conditions.is_empty());
        assert!(!plan.prompt_token_ids.is_empty());

        let frames = spec(None, Some(Length::Frames(124)));
        assert_eq!(
            processor
                .plan(&frames, &[], &CharacterTokenizer)
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
        assert!(processor(&["t2va"], MAX_VIDEO_SECONDS + 1.0).is_err());
        assert!(processor(&["t2va"], MIN_VIDEO_SECONDS - 1.0).is_err());
    }

    /// A vision block is a run of vision tags, wherever in the prompt it lies.
    #[test]
    fn vision_spans_are_the_runs_of_vision_tags() {
        let text = presentation::TEXT_TAG;
        assert_eq!(
            vision_spans(&[text, VIDEO_TAG, VIDEO_TAG, text, VIDEO_TAG]),
            vec![(1, 2), (4, 1)]
        );
        assert!(vision_spans(&[text, text]).is_empty());
    }
}
