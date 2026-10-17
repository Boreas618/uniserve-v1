//! The MiniMax-H3 request rules of `uniserve-core`, exported to Python.
//!
//! The Python library plans requests (`uniserve.models.minimax_h3.processing`)
//! with the same rules the server plans them with: it loads this library and
//! calls the functions below by name. Each one passes scalars, strings and
//! flat lists, and returns the same; the documentation of each states the
//! order of its lists. A request the rules reject comes back as a
//! `[field, message]` rejection, the field being the request path the
//! rejection names (`target.aspect_ratio`, `conditions[2]`); a rule that
//! rejects its arguments outside a request raises `ValueError`.
//!
//! These calls run on the caller's thread and touch no rank state: they are
//! the rules' arithmetic and nothing else.

use std::path::Path;

use tvm_ffi::{Any, AnyView, Array, Function};
use uniserve_core::video::plan::{
    self, AudioClip, AudioFacts, BUCKET_SHORT_EDGES, ConditionSpec, ConditionType, FrameRate,
    ImageFacts, MAX_VIDEO_SECONDS, MediaFacts, NAMED_ASPECT_RATIOS, PlanLimits, Prepared,
    RequestPlan, Role, Target, VideoFacts, VideoInputError, Vision, VisionConfig,
};
use uniserve_core::video::presentation::{self, Segment, Tokenizer};
use uniserve_core::{Canvas, VideoTask};

use crate::error::report;
use crate::ffi::{integer_list, integers, is_none, list};

type Result<T> = tvm_ffi::Result<T>;

fn value_error(message: &str) -> tvm_ffi::Error {
    tvm_ffi::Error::new(tvm_ffi::VALUE_ERROR, message, "")
}

fn runtime_error(message: &str) -> tvm_ffi::Error {
    tvm_ffi::Error::new(tvm_ffi::RUNTIME_ERROR, message, "")
}

fn text(value: &str) -> Any {
    Any::from(tvm_ffi::String::from(value))
}

fn canvas_values(canvas: Canvas) -> Array<Any> {
    integer_list([canvas.width, canvas.height].map(i64::from))
}

/// A value that must be a non-negative count below 2^32.
fn count(value: Any, what: &str) -> Result<u32> {
    u32::try_from(i64::try_from(value)?)
        .map_err(|_| value_error(&format!("{what} is out of range")))
}

fn optional<T>(value: Any, convert: impl FnOnce(Any) -> Result<T>) -> Result<Option<T>> {
    if is_none(&value) {
        Ok(None)
    } else {
        convert(value).map(Some)
    }
}

/// `[field, message]` of a rejected request, `None` for no rejection; a
/// failure that is not a rejection raises.
fn rejection(error: VideoInputError) -> Result<Any> {
    match error {
        VideoInputError::Invalid { field, message } => {
            Ok(Any::from(list([text(&field.to_string()), text(&message)])))
        }
        VideoInputError::Internal(message) => Err(runtime_error(&message)),
    }
}

/// `video_canvas(aspect_width, aspect_height) -> [width, height]`: the
/// canvas rule (adapt_shape_v1) applied to a display aspect.
fn video_canvas(aspect_width: f64, aspect_height: f64) -> Result<Array<Any>> {
    plan::canvas(aspect_width, aspect_height)
        .map(canvas_values)
        .map_err(|error| value_error(&error.to_string()))
}

/// `video_frame_count(seconds, max_seconds) -> frames`: the generated frame
/// count of a duration; `max_seconds` `None` is the model's 15 seconds.
fn video_frame_count(seconds: f64, max_seconds: Option<f64>) -> Result<i64> {
    let max_seconds = max_seconds.unwrap_or(MAX_VIDEO_SECONDS);
    plan::video_frame_count(seconds, max_seconds)
        .map(i64::from)
        .map_err(|error| value_error(&error.to_string()))
}

/// `video_aspect_ratios() -> [width, height, ...]`: the named aspect ratios
/// `t2va` and `ref2va` accept besides `auto`, in canonical order.
fn video_aspect_ratios() -> Result<Array<Any>> {
    Ok(integer_list(NAMED_ASPECT_RATIOS.iter().flat_map(
        |&(width, height)| [i64::from(width), i64::from(height)],
    )))
}

/// `video_buckets() -> [width, height, ...]`: the FastH3 training buckets,
/// the named aspect ratios' canvases at the 768-pixel short edge, then at
/// the 480-pixel one.
fn video_buckets() -> Result<Array<Any>> {
    let mut values = Vec::new();
    for short_edge in BUCKET_SHORT_EDGES {
        for ratio in NAMED_ASPECT_RATIOS {
            let canvas = plan::bucket(short_edge, ratio)
                .ok_or_else(|| runtime_error("a named ratio has no training bucket"))?;
            values.extend([i64::from(canvas.width), i64::from(canvas.height)]);
        }
    }
    Ok(integer_list(values))
}

fn task(name: &str) -> Result<VideoTask> {
    VideoTask::from_name(name).ok_or_else(|| value_error(&format!("unknown video task {name:?}")))
}

/// `video_named_canvas(task, short_edge, aspect_ratio) -> [rejection,
/// canvas]`: the canvas a target names without media, `[width, height]`, or
/// `None` for `fl2va` with `auto`, whose canvas follows the first keyframe.
fn video_named_canvas(
    task_name: tvm_ffi::String,
    short_edge: i64,
    aspect_ratio: tvm_ffi::String,
) -> Result<Array<Any>> {
    let target = Target {
        short_edge: u32::try_from(short_edge)
            .map_err(|_| value_error("short_edge is out of range"))?,
        aspect_ratio: aspect_ratio.as_str(),
        duration_seconds: None,
    };
    Ok(
        match plan::named_canvas(task(task_name.as_str())?, &target) {
            Ok(canvas) => list([
                Any::default(),
                canvas.map_or_else(Any::default, |canvas| Any::from(canvas_values(canvas))),
            ]),
            Err(error) => list([rejection(error)?, Any::default()]),
        },
    )
}

/// `video_vision_config(image_path, video_path) -> [patch_size,
/// temporal_patch_size, merge_size, image_min_pixels, image_max_pixels,
/// video_min_pixels, video_max_pixels]`: the Qwen3-VL processor geometry of
/// a checkpoint's image and video processor configs.
fn video_vision_config(image: tvm_ffi::String, video: tvm_ffi::String) -> Result<Array<Any>> {
    let config = VisionConfig::read(Path::new(image.as_str()), Path::new(video.as_str()))
        .map_err(|error| value_error(&report(&error)))?;
    let sizes = [
        config.image_min_pixels,
        config.image_max_pixels,
        config.video_min_pixels,
        config.video_max_pixels,
    ];
    let mut values = vec![
        i64::from(config.patch_size),
        i64::from(config.temporal_patch_size),
        i64::from(config.merge_size),
    ];
    for size in sizes {
        values
            .push(i64::try_from(size).map_err(|_| value_error("a pixel budget is out of range"))?);
    }
    Ok(integer_list(values))
}

fn vision_config(values: &[i64]) -> Result<VisionConfig> {
    let [
        patch,
        temporal,
        merge,
        image_min,
        image_max,
        video_min,
        video_max,
    ] = values
    else {
        return Err(value_error("a vision config has seven values"));
    };
    let narrow = |value: i64| {
        u32::try_from(value).map_err(|_| value_error("a vision config value is out of range"))
    };
    let budget = |value: i64| {
        u64::try_from(value).map_err(|_| value_error("a pixel budget is out of range"))
    };
    let config = VisionConfig {
        patch_size: narrow(*patch)?,
        temporal_patch_size: narrow(*temporal)?,
        merge_size: narrow(*merge)?,
        image_min_pixels: budget(*image_min)?,
        image_max_pixels: budget(*image_max)?,
        video_min_pixels: budget(*video_min)?,
        video_max_pixels: budget(*video_max)?,
    };
    if config.patch_size == 0 || config.temporal_patch_size == 0 || config.merge_size == 0 {
        return Err(value_error("a vision patch geometry must be positive"));
    }
    Ok(config)
}

/// The probed facts of one condition: `kind` is `image`, `video` or
/// `audio`; `values` are an image's `[width, height]`, a video's
/// `[display_width, display_height, rate_numerator, rate_denominator,
/// frames]` followed by its soundtrack's `[sample_rate, samples]` when it has
/// one, and an audio file's `[sample_rate, samples]`.
fn media_facts(kind: &str, values: &[i64]) -> Result<MediaFacts> {
    let narrow =
        |value: i64| u32::try_from(value).map_err(|_| value_error("a media fact is out of range"));
    let wide =
        |value: i64| u64::try_from(value).map_err(|_| value_error("a media fact is out of range"));
    let audio = |rate: i64, samples: i64| -> Result<AudioFacts> {
        Ok(AudioFacts {
            sample_rate: narrow(rate)?,
            samples: wide(samples)?,
        })
    };
    match (kind, values) {
        ("image", &[width, height]) => Ok(MediaFacts::Image(ImageFacts {
            width: narrow(width)?,
            height: narrow(height)?,
        })),
        ("audio", &[rate, samples]) => Ok(MediaFacts::Audio(audio(rate, samples)?)),
        (
            "video",
            &[
                width,
                height,
                numerator,
                denominator,
                frames,
                ref soundtrack @ ..,
            ],
        ) => {
            let soundtrack = match *soundtrack {
                [] => None,
                [rate, samples] => Some(audio(rate, samples)?),
                _ => return Err(value_error("a soundtrack is [sample_rate, samples]")),
            };
            if numerator <= 0 || denominator <= 0 {
                return Err(value_error("a video frame rate must be positive"));
            }
            Ok(MediaFacts::Video(VideoFacts {
                display_width: wide(width)?,
                display_height: wide(height)?,
                frame_rate: FrameRate {
                    numerator: narrow(numerator)?,
                    denominator: narrow(denominator)?,
                },
                frames: wide(frames)?,
                soundtrack,
            }))
        }
        _ => Err(value_error(&format!(
            "malformed {kind:?} media facts {values:?}"
        ))),
    }
}

/// One condition: `[type, role, frame_index, start_seconds, media_kind,
/// media_facts]`, the index and offset `None` when absent.
fn condition(value: Any) -> Result<(ConditionSpec, MediaFacts)> {
    let fields: Array<Any> = Array::try_from(value)?;
    if fields.len() != 6 {
        return Err(value_error("a condition has six fields"));
    }
    let name = |index: usize| -> Result<String> {
        Ok(tvm_ffi::String::try_from(fields.get(index)?)?
            .as_str()
            .to_owned())
    };
    let condition_type = name(0)?;
    let role = name(1)?;
    let spec = ConditionSpec {
        condition_type: ConditionType::from_name(&condition_type)
            .ok_or_else(|| value_error(&format!("unknown condition type {condition_type:?}")))?,
        role: Role::from_name(&role)
            .ok_or_else(|| value_error(&format!("unknown role {role:?}")))?,
        frame_index: optional(fields.get(2)?, i64::try_from)?,
        start_seconds: optional(fields.get(3)?, f64::try_from)?,
    };
    let media = media_facts(&name(4)?, &integers(fields.get(5)?)?)?;
    Ok((spec, media))
}

/// A planned soundtrack or audio reference: `[sample_rate, start_sample,
/// source_samples, samples, latents, rows]`.
fn audio_values(clip: &AudioClip) -> Vec<i64> {
    vec![
        i64::from(clip.sample_rate),
        clip.start_sample as i64,
        clip.source_samples as i64,
        clip.samples as i64,
        i64::from(clip.latents),
        i64::from(clip.rows()),
    ]
}

/// One planned condition: `[rows, prepared, vision, timestamps]`.
///
/// - `rows`: `[video_rows, audio_rows]`;
/// - `prepared`, by the condition's role and media: a keyframe's anchored
///   `[frame_index]` (0 or -1) followed by its cover crop's `[width, height,
///   left, top]` when it is cropped; an image reference's resized `[width,
///   height]`; a video's `[canvas_width, canvas_height, start_frame, frames,
///   vae_frames, latent_frames]` followed by its soundtrack's audio values;
///   an audio reference's audio values ([`audio_values`]);
/// - `vision`: empty when the conditioner does not read the condition, an
///   image's `[t, h, w, tokens]`, a video's `[t, h, w, block_tokens,
///   frame_index...]`;
/// - `timestamps`: a video's block timestamps in seconds, otherwise empty.
fn condition_values(planned: &plan::ConditionPlan) -> Array<Any> {
    let prepared: Vec<i64> = match &planned.prepared {
        Prepared::Keyframe(fit) => {
            let anchor = match fit.position {
                plan::FramePosition::First => 0,
                plan::FramePosition::Last => -1,
            };
            let mut values = vec![anchor];
            if let Some(crop) = fit.cover_crop {
                values.extend([crop.width, crop.height, crop.left, crop.top].map(i64::from));
            }
            values
        }
        Prepared::Image(size) => vec![i64::from(size.width), i64::from(size.height)],
        Prepared::Video(clip) => {
            let mut values: Vec<i64> = [
                clip.canvas.width,
                clip.canvas.height,
                clip.start_frame,
                clip.frames,
                clip.vae_frames,
                clip.latent_frames,
            ]
            .map(i64::from)
            .to_vec();
            if let Some(track) = &clip.soundtrack {
                values.extend(audio_values(track));
            }
            values
        }
        Prepared::Audio(clip) => audio_values(clip),
    };
    let grid = |grid: uniserve_core::VisionGrid| [grid.t, grid.h, grid.w].map(i64::from);
    let (vision, timestamps): (Vec<i64>, Vec<f64>) = match &planned.vision {
        None => (Vec::new(), Vec::new()),
        Some(Vision::Image(seen)) => {
            let mut values = grid(seen.grid).to_vec();
            values.push(i64::from(seen.tokens));
            (values, Vec::new())
        }
        Some(Vision::Video(seen)) => {
            let mut values = grid(seen.grid).to_vec();
            values.push(i64::from(seen.block_tokens));
            values.extend(seen.frame_indices.iter().map(|&index| i64::from(index)));
            (values, seen.block_timestamps.clone())
        }
    };
    list([
        Any::from(integer_list(
            [planned.video_rows, planned.audio_rows].map(i64::from),
        )),
        Any::from(integer_list(prepared)),
        Any::from(integer_list(vision)),
        Any::from(list(timestamps.into_iter().map(Any::from))),
    ])
}

/// The tokenizer calls a Python caller passes: `encode(text) -> [id, ...]`
/// without special tokens, and `token_id(token) -> id`.
struct Callbacks {
    encode: Function,
    token_id: Function,
}

impl Tokenizer for Callbacks {
    type Error = tvm_ffi::Error;

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let ids = integers(self.encode.call_tuple((tvm_ffi::String::from(text),))?)?;
        ids.into_iter()
            .map(|id| u32::try_from(id).map_err(|_| value_error("a token id is out of range")))
            .collect()
    }

    fn token_id(&self, token: &str) -> Option<u32> {
        let id = self
            .token_id
            .call_tuple((tvm_ffi::String::from(token),))
            .ok()?;
        if is_none(&id) {
            return None;
        }
        i64::try_from(id).ok().and_then(|id| u32::try_from(id).ok())
    }
}

/// The presentation of `plan`, as the function the plan's caller keeps:
/// `present(prompt, encode, token_id) -> [rejection, segments, token_ids,
/// tags]`. Each segment is `[text]` or a vision block's `[placeholder,
/// tokens]`. Without `encode` and `token_id` (`None`) the presentation is
/// only listed: no rejection, no tokens.
fn presentation_function(plan: RequestPlan) -> Function {
    Function::from_packed(move |arguments: &[AnyView<'_>]| -> Result<Any> {
        let [prompt, encode, token_id] = arguments else {
            return Err(value_error("present takes a prompt, encode and token_id"));
        };
        let prompt = tvm_ffi::String::try_from(*prompt)?;
        let prompt = prompt.as_str();
        let segments = list(
            presentation::segments(&plan, prompt)
                .into_iter()
                .map(|segment| {
                    Any::from(match segment {
                        Segment::Text(text_value) => list([text(&text_value)]),
                        Segment::Vision { pad, tokens } => {
                            list([text(pad.token()), Any::from(i64::from(tokens))])
                        }
                    })
                }),
        );
        let encode = Any::from(*encode);
        let token_id = Any::from(*token_id);
        if is_none(&encode) || is_none(&token_id) {
            return Ok(Any::from(list([
                Any::default(),
                Any::from(segments),
                Any::from(integer_list([])),
                Any::from(integer_list([])),
            ])));
        }
        let tokenizer = Callbacks {
            encode: Function::try_from(encode)?,
            token_id: Function::try_from(token_id)?,
        };
        Ok(Any::from(
            match presentation::present(&tokenizer, &plan, prompt) {
                Ok(presented) => list([
                    Any::default(),
                    Any::from(segments),
                    Any::from(integer_list(presented.token_ids.into_iter().map(i64::from))),
                    Any::from(integer_list(presented.tags.into_iter().map(i64::from))),
                ]),
                Err(error) => list([
                    rejection(error)?,
                    Any::from(segments),
                    Any::from(integer_list([])),
                    Any::from(integer_list([])),
                ]),
            },
        ))
    })
}

/// `video_plan(task, target, conditions, vision, limits) -> [rejection,
/// plan]`: validates a request and resolves every size it implies.
///
/// - `target`: `[short_edge, aspect_ratio, duration_seconds]`, the duration
///   `None` when the request names none;
/// - `conditions`: one list per condition in request order ([`condition`]);
/// - `vision`: the seven values of [`video_vision_config`];
/// - `limits`: `[tasks, max_seconds, canvases]`: the served task names, the
///   longest served duration (`None` for the model's 15 seconds) and the
///   only canvases served as `[width, height, ...]` (`None` for every
///   canvas the rules produce).
///
/// `plan` is `None` for a rejected request, otherwise `[sizes,
/// duration_seconds, conditions, present]`: `sizes` are `[width, height,
/// num_frames, latent_frames, audio_latents, target_video_rows,
/// target_audio_rows, condition_video_rows, condition_audio_rows]`, each
/// condition is [`condition_values`] and `present` is
/// [`presentation_function`].
fn video_plan(
    task_name: tvm_ffi::String,
    target: Array<Any>,
    conditions: Array<Any>,
    vision: Array<Any>,
    limits: Array<Any>,
) -> Result<Array<Any>> {
    if target.len() != 3 || limits.len() != 3 {
        return Err(value_error("a target and limits have three fields each"));
    }
    let aspect_ratio = tvm_ffi::String::try_from(target.get(1)?)?;
    let target = Target {
        short_edge: count(target.get(0)?, "short_edge")?,
        aspect_ratio: aspect_ratio.as_str(),
        duration_seconds: optional(target.get(2)?, f64::try_from)?,
    };
    let (specs, media): (Vec<ConditionSpec>, Vec<MediaFacts>) = conditions
        .iter()
        .map(condition)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .unzip();
    let vision = vision_config(&integers(Any::from(vision))?)?;
    let tasks: Array<tvm_ffi::String> = Array::try_from(limits.get(0)?)?;
    let limits = PlanLimits {
        tasks: tasks
            .iter()
            .map(|name| task(name.as_str()))
            .collect::<Result<_>>()?,
        max_video_seconds: optional(limits.get(1)?, f64::try_from)?.unwrap_or(MAX_VIDEO_SECONDS),
        canvases: optional(limits.get(2)?, |value| {
            let values = integers(value)?;
            let (pairs, rest) = values.as_chunks::<2>();
            if !rest.is_empty() {
                return Err(value_error("canvases are [width, height, ...]"));
            }
            pairs
                .iter()
                .map(|&[width, height]| {
                    let side = |value: i64| {
                        u32::try_from(value)
                            .map_err(|_| value_error("a canvas side is out of range"))
                    };
                    Ok(Canvas {
                        width: side(width)?,
                        height: side(height)?,
                    })
                })
                .collect()
        })?,
    };

    let planned = match plan::plan_request(
        task(task_name.as_str())?,
        &target,
        &specs,
        &media,
        &vision,
        &limits,
    ) {
        Ok(planned) => planned,
        Err(error) => return Ok(list([rejection(error)?, Any::default()])),
    };
    let sizes = integer_list(
        [
            planned.canvas.width,
            planned.canvas.height,
            planned.num_frames,
            planned.latent_frames,
            planned.audio_latents,
            planned.target_video_rows(),
            planned.target_audio_rows(),
            planned.condition_video_rows(),
            planned.condition_audio_rows(),
        ]
        .map(i64::from),
    );
    let conditions = list(
        planned
            .conditions
            .iter()
            .map(|condition| Any::from(condition_values(condition))),
    );
    let duration = planned.duration_seconds;
    let present = presentation_function(planned);
    Ok(list([
        Any::default(),
        Any::from(list([
            Any::from(sizes),
            Any::from(duration),
            Any::from(conditions),
            Any::from(present),
        ])),
    ]))
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(video_canvas, video_canvas);
tvm_ffi::tvm_ffi_dll_export_typed_func!(video_frame_count, video_frame_count);
tvm_ffi::tvm_ffi_dll_export_typed_func!(video_aspect_ratios, video_aspect_ratios);
tvm_ffi::tvm_ffi_dll_export_typed_func!(video_buckets, video_buckets);
tvm_ffi::tvm_ffi_dll_export_typed_func!(video_named_canvas, video_named_canvas);
tvm_ffi::tvm_ffi_dll_export_typed_func!(video_vision_config, video_vision_config);
tvm_ffi::tvm_ffi_dll_export_typed_func!(video_plan, video_plan);
