//! The MiniMax-H3 request rules of `uniserve-core`, exported to Python.
//!
//! The server plans and presents every request with these rules. The model
//! package (`uniserve.models.minimax_h3.processing`) reads the parts of them
//! its computation and tools use, the canvas rule, the training buckets, the
//! largest references a request may bring, the canvas a target names, the
//! frame count of a duration and the named aspect ratios: it loads this
//! library and calls the functions below by name. Each
//! one passes scalars and strings and returns flat lists; the documentation
//! of each states their order. A target the rules reject comes back as a
//! `[field, message]` rejection, the field being the request path the
//! rejection names (`target.aspect_ratio`); a rule that rejects its
//! arguments outside a request raises `ValueError`.
//!
//! These calls run on the caller's thread and touch no rank state: they are
//! the rules' arithmetic and nothing else.

use tvm_ffi::{Any, Array};
use uniserve_core::video::plan::{
    self, BUCKET_SHORT_EDGES, MAX_VIDEO_SECONDS, NAMED_ASPECT_RATIOS, Target, VideoInputError,
};
use uniserve_core::{Canvas, VideoTask};

use crate::ffi::{integer_list, list};

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

/// `video_reference_bounds(frames) -> [images, video_frames, audio_samples,
/// video_canvases]`: the largest references a request generating `frames`
/// frames may bring (`plan::reference_bounds`); `images`, the two largest
/// image rasters, and `video_canvases` are flat `[width, height, ...]`.
fn video_reference_bounds(frames: i64) -> Result<Array<Any>> {
    let frames = u32::try_from(frames).map_err(|_| value_error("frames is out of range"))?;
    let bounds = plan::reference_bounds(frames);
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

tvm_ffi::tvm_ffi_dll_export_typed_func!(video_canvas, video_canvas);
tvm_ffi::tvm_ffi_dll_export_typed_func!(video_frame_count, video_frame_count);
tvm_ffi::tvm_ffi_dll_export_typed_func!(video_aspect_ratios, video_aspect_ratios);
tvm_ffi::tvm_ffi_dll_export_typed_func!(video_buckets, video_buckets);
tvm_ffi::tvm_ffi_dll_export_typed_func!(video_reference_bounds, video_reference_bounds);
tvm_ffi::tvm_ffi_dll_export_typed_func!(video_named_canvas, video_named_canvas);
