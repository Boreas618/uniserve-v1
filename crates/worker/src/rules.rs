//! The MiniMax-H3 request rules of the `models` crate, exported to Python.
//!
//! The server plans and presents every request with these rules. The model
//! package (`uniserve.models.minimax_h3.processing`) reads the parts of them
//! its computation and tools use, the tasks, the canvas rule, the largest
//! references a request may bring, the canvas a target names among the
//! canvases the model generates and the frame count of a duration: it loads
//! this library and calls the functions below by name. Each one passes
//! scalars and lists and returns lists; the documentation of each states
//! their order. A target the rules reject comes back as a `[field, message]`
//! rejection, the field being the request path the rejection names
//! (`target.aspect_ratio`); a rule that rejects its arguments outside a
//! request raises `ValueError`.
//!
//! These calls run on the caller's thread and touch no rank state: they are
//! the rules' arithmetic and nothing else.

use models::video::plan;
use models::video::plan::{MAX_VIDEO_SECONDS, Target, VideoInputError};
use models::video::{Canvas, FrameClips, ServedCanvas, VideoTask};
use tvm_ffi::{Any, Array};

use crate::ffi::{integer_list, integers, list};

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

/// A positive count of a rules argument.
fn positive(value: i64, what: &str) -> Result<u32> {
    u32::try_from(value)
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| value_error(&format!("{what} must be a positive count")))
}

/// A model's frame rule as the rules take it: `[min_frames, clip_frames,
/// clip_latent_frames, tail_latent_frames]` ([`FrameClips`]).
fn frame_rule(frames: Array<Any>) -> Result<FrameClips> {
    let rule = integers(Any::from(frames))?;
    let [
        min_frames,
        clip_frames,
        clip_latent_frames,
        tail_latent_frames,
    ] = rule[..]
    else {
        return Err(value_error("frames must hold four counts"));
    };
    let frames = FrameClips {
        min_frames: positive(min_frames, "frames")?,
        clip_frames: positive(clip_frames, "frames")?,
        clip_latent_frames: positive(clip_latent_frames, "frames")?,
        tail_latent_frames: positive(tail_latent_frames, "frames")?,
    };
    if !frames.is_valid() {
        return Err(value_error("frames leave no tail past the first clip"));
    }
    Ok(frames)
}

/// `video_frame_count(seconds, max_seconds, frame_rate, frames) -> frames`:
/// the generated frame count of a duration at the model's output
/// `frame_rate` under its frame rule `frames` ([`frame_rule`]);
/// `max_seconds` `None` is the model's 15 seconds.
fn video_frame_count(
    seconds: f64,
    max_seconds: Option<f64>,
    frame_rate: i64,
    frames: Array<Any>,
) -> Result<i64> {
    let max_seconds = max_seconds.unwrap_or(MAX_VIDEO_SECONDS);
    let frame_rate = positive(frame_rate, "frame_rate")?;
    let frames = frame_rule(frames)?;
    plan::video_frame_count(seconds, max_seconds, frame_rate, &frames)
        .map(i64::from)
        .map_err(|error| value_error(&error.to_string()))
}

/// `video_tasks() -> [name, ...]`: the request names of the tasks
/// ([`VideoTask`]), in canonical order.
fn video_tasks() -> Result<Array<Any>> {
    Ok(list(VideoTask::ALL.map(|task| text(task.as_str()))))
}

/// `video_reference_bounds(frames, frame_rate, frame_rule, image_short_edge)
/// -> [images, video_frames, audio_samples, video_canvases]`: the largest
/// references a request generating `frames` frames at the model's output
/// `frame_rate` under its frame rule ([`frame_rule`]) may bring, its image
/// references encoded at a short edge of at most `image_short_edge`
/// (`plan::reference_bounds`); `images`, the two largest image rasters, and
/// `video_canvases` are flat `[width, height, ...]`.
fn video_reference_bounds(
    frames: i64,
    frame_rate: i64,
    frame_rule_values: Array<Any>,
    image_short_edge: i64,
) -> Result<Array<Any>> {
    let frames = u32::try_from(frames).map_err(|_| value_error("frames is out of range"))?;
    let frame_rate = positive(frame_rate, "frame_rate")?;
    let clips = frame_rule(frame_rule_values)?;
    let image_short_edge = positive(image_short_edge, "image_short_edge")?;
    let bounds = plan::reference_bounds(frames, frame_rate, &clips, image_short_edge);
    let rasters = |canvases: &[Canvas]| {
        integer_list(
            canvases
                .iter()
                .flat_map(|canvas| [i64::from(canvas.width), i64::from(canvas.height)]),
        )
    };
    let samples = i64::try_from(bounds.audio_samples)
        .map_err(|_| value_error("audio samples are out of range"))?;
    Ok(list([
        Any::from(rasters(&bounds.images)),
        Any::from(i64::from(bounds.video_frames)),
        Any::from(samples),
        Any::from(rasters(&bounds.video_canvases)),
    ]))
}

fn task(name: &str) -> Result<VideoTask> {
    VideoTask::from_name(name).ok_or_else(|| value_error(&format!("unknown video task {name:?}")))
}

/// Served canvases from `[[short_edge, aspect_ratio, width, height], ...]`.
fn served_canvases(entries: Array<Any>) -> Result<Vec<ServedCanvas>> {
    entries
        .iter()
        .map(|entry| {
            let fields: Array<Any> = Array::try_from(entry)?;
            let fields: Vec<Any> = fields.iter().collect();
            let [short_edge, aspect_ratio, width, height] =
                <[Any; 4]>::try_from(fields).map_err(|_| {
                    value_error("a canvas is [short_edge, aspect_ratio, width, height]")
                })?;
            let side = |value: Any| positive(i64::try_from(value)?, "a canvas side");
            Ok(ServedCanvas {
                short_edge: side(short_edge)?,
                aspect_ratio: tvm_ffi::String::try_from(aspect_ratio)?.as_str().to_owned(),
                canvas: Canvas {
                    width: side(width)?,
                    height: side(height)?,
                },
            })
        })
        .collect()
}

/// `video_named_canvas(task, short_edge, aspect_ratio, canvases) ->
/// [rejection, canvas]`: the canvas a target names without media,
/// `[width, height]`, or `None` for `fl2va` with `auto`, whose canvas follows
/// the first keyframe. `canvases` are the canvases served, each
/// `[short_edge, aspect_ratio, width, height]`.
fn video_named_canvas(
    task_name: tvm_ffi::String,
    short_edge: i64,
    aspect_ratio: tvm_ffi::String,
    canvases: Array<Any>,
) -> Result<Array<Any>> {
    let target = Target {
        short_edge: u32::try_from(short_edge)
            .map_err(|_| value_error("short_edge is out of range"))?,
        aspect_ratio: aspect_ratio.as_str(),
        duration_seconds: None,
    };
    let canvases = served_canvases(canvases)?;
    Ok(
        match plan::named_canvas(task(task_name.as_str())?, &target, &canvases) {
            Ok(canvas) => list([
                Any::default(),
                canvas.map_or_else(Any::default, |canvas| Any::from(canvas_values(canvas))),
            ]),
            Err(error) => list([rejection(error)?, Any::default()]),
        },
    )
}

/// `video_image_grid(image_config, video_config, height, width) -> [t, h,
/// w]`: the patch grid the conditioner reads a `height` x `width` still
/// image on, under the checkpoint's image and video processor configs (JSON
/// texts; `plan::ProcessorConfig::image_grid`).
fn video_image_grid(
    image_config: tvm_ffi::String,
    video_config: tvm_ffi::String,
    height: i64,
    width: i64,
) -> Result<Array<Any>> {
    let processor = plan::ProcessorConfig::parse(image_config.as_str(), video_config.as_str())
        .map_err(|error| value_error(&error.to_string()))?;
    let grid = processor
        .image_grid(positive(height, "height")?, positive(width, "width")?)
        .map_err(|error| value_error(&error.to_string()))?;
    Ok(integer_list([grid.t, grid.h, grid.w].map(i64::from)))
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(video_tasks, video_tasks);
tvm_ffi::tvm_ffi_dll_export_typed_func!(video_image_grid, video_image_grid);
tvm_ffi::tvm_ffi_dll_export_typed_func!(video_canvas, video_canvas);
tvm_ffi::tvm_ffi_dll_export_typed_func!(video_frame_count, video_frame_count);
tvm_ffi::tvm_ffi_dll_export_typed_func!(video_reference_bounds, video_reference_bounds);
tvm_ffi::tvm_ffi_dll_export_typed_func!(video_named_canvas, video_named_canvas);
