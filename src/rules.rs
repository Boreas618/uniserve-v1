//! The video request rules of the `models` crate, answered as JSON.
//!
//! The serving path plans every MiniMax-H3 request with these rules
//! (`crates/models/rules/src/video`). The model package's own computation and
//! its tools read the same rules, and this command is how they reach them:
//! one JSON request on standard input, one JSON answer on standard output,
//! and no rule computed anywhere but in `models`.
//!
//! The request is `{"calls": [{"call": NAME, "args": {...}}, ...]}`; one
//! process answers all of them, so a caller that knows its whole batch pays
//! one launch. Each answer states the call and the arguments it answered, so
//! a reader can check it beside its input:
//!
//! ```json
//! {"answers": [
//!   {"call": "video_canvas", "args": {"aspect_width": 16, "aspect_height": 9},
//!    "answer": [1344, 768]},
//!   {"call": "video_frame_count", "args": {...},
//!    "error": {"kind": "value", "message": "..."}}
//! ]}
//! ```
//!
//! A refusal of an argument — a ratio outside the canvas rule's range, a
//! duration outside `[4, max]`, an unknown task name — is an `error` entry
//! carrying the rule's own message, and the process exits with
//! [`REFUSED`]. An answer is never a zero-filled stand-in for a refusal.
//! The one rejection that is an answer rather than an error is
//! `video_named_canvas`'s: it reports the `[field, message]` a target is
//! rejected by beside the canvas, which is the contract its caller raises
//! from.
//!
//! The same request and arguments answer the same bytes every time.

use std::io::{Read as _, Write as _};
use std::process::ExitCode;

use anyhow::{Context, Result};
use models::video::plan;
use models::video::plan::{Target, VideoInputError};
use models::video::{Canvas, FrameClips, ServedCanvas, VideoTask};
use serde::Deserialize;
use serde_json::{Value, json};

/// The exit status of a request in which a rule refused an argument.
pub(crate) const REFUSED: u8 = 2;

/// One call: a rule's name and the arguments it takes.
#[derive(Debug, Deserialize)]
struct Call {
    call: String,
    #[serde(default)]
    args: Value,
}

/// The whole request: the calls one process answers.
#[derive(Debug, Deserialize)]
struct Request {
    calls: Vec<Call>,
}

/// Why a call did not answer.
#[derive(Debug)]
struct Refusal {
    /// `value` for an argument a rule refused, `runtime` for a failure
    /// while answering a valid one.
    kind: &'static str,
    message: String,
}

impl Refusal {
    /// The request's arguments break a rule.
    fn value(message: impl Into<String>) -> Self {
        Self {
            kind: "value",
            message: message.into(),
        }
    }

    /// A valid request could not be answered.
    fn runtime(message: impl Into<String>) -> Self {
        Self {
            kind: "runtime",
            message: message.into(),
        }
    }

    /// The refusal as the answer states it.
    fn to_json(&self) -> Value {
        json!({"kind": self.kind, "message": self.message})
    }
}

/// Runs the `uniserve rules` command.
pub(crate) fn run() -> Result<ExitCode> {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .context("reading the request from standard input")?;
    let request: Request =
        serde_json::from_str(&input).context("a rules request is a JSON object")?;
    let mut answers = Vec::with_capacity(request.calls.len());
    let mut refused = false;
    for call in request.calls {
        let mut answer = json!({"call": call.call, "args": call.args});
        match answer_call(&call.call, &call.args) {
            Ok(value) => {
                answer["answer"] = value;
            }
            Err(refusal) => {
                refused = true;
                answer["error"] = refusal.to_json();
            }
        }
        answers.push(answer);
    }
    let document = json!({ "answers": answers });
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &document).context("writing the answer")?;
    writeln!(stdout).context("writing the answer")?;
    stdout.flush().context("writing the answer")?;
    Ok(if refused {
        ExitCode::from(REFUSED)
    } else {
        ExitCode::SUCCESS
    })
}

/// Answers one call by its rule's name.
fn answer_call(name: &str, args: &Value) -> Result<Value, Refusal> {
    match name {
        "video_tasks" => Ok(json!(VideoTask::ALL.map(VideoTask::as_str))),
        "video_canvas" => video_canvas(args),
        "video_frame_count" => video_frame_count(args),
        "video_reference_bounds" => video_reference_bounds(args),
        "video_named_canvas" => video_named_canvas(args),
        "video_image_grid" => video_image_grid(args),
        other => Err(Refusal::value(format!(
            "unknown rule {other:?}: the rules are video_tasks, video_canvas, \
             video_frame_count, video_reference_bounds, video_named_canvas and \
             video_image_grid"
        ))),
    }
}

/// `video_canvas(aspect_width, aspect_height) -> [width, height]`: the canvas
/// rule applied to a display aspect (`plan::canvas`).
fn video_canvas(args: &Value) -> Result<Value, Refusal> {
    let width = number(args, "aspect_width")?;
    let height = number(args, "aspect_height")?;
    plan::canvas(width, height)
        .map(canvas_json)
        .map_err(|error| Refusal::value(error.to_string()))
}

/// `video_frame_count(seconds, max_seconds, frame_rate, frame_rule) -> frames`:
/// the generated frame count of a duration (`plan::video_frame_count`);
/// `max_seconds` null is the model's longest video.
fn video_frame_count(args: &Value) -> Result<Value, Refusal> {
    let seconds = number(args, "seconds")?;
    let longest = optional_number(args, "max_seconds")?.unwrap_or(plan::MAX_VIDEO_SECONDS);
    let frame_rate = positive(args, "frame_rate")?;
    let frames = frame_rule(args)?;
    plan::video_frame_count(seconds, longest, frame_rate, &frames)
        .map(|frames| json!(frames))
        .map_err(|error| Refusal::value(error.to_string()))
}

/// `video_reference_bounds(frames, frame_rate, frame_rule, image_short_edge)
/// -> [[width, height, ...], frames, samples, [width, height, ...]]`: the
/// largest references a request may bring (`plan::reference_bounds`).
fn video_reference_bounds(args: &Value) -> Result<Value, Refusal> {
    let frames = positive(args, "frames")?;
    let frame_rate = positive(args, "frame_rate")?;
    let clips = frame_rule(args)?;
    let short_edge = positive(args, "image_short_edge")?;
    let bounds = plan::reference_bounds(frames, frame_rate, &clips, short_edge);
    Ok(json!([
        flat_canvases(&bounds.images),
        bounds.video_frames,
        bounds.audio_samples,
        flat_canvases(&bounds.video_canvases),
    ]))
}

/// `video_named_canvas(task, short_edge, aspect_ratio, canvases)
/// -> [[field, message] | null, [width, height] | null]`: the canvas a target
/// names without media (`plan::named_canvas`), or the rejection it is
/// refused by. `canvases` are the served ones, each
/// `[short_edge, aspect_ratio, width, height]`.
fn video_named_canvas(args: &Value) -> Result<Value, Refusal> {
    let name = text(args, "task")?;
    let task = VideoTask::from_name(name).ok_or_else(|| {
        Refusal::value(format!(
            "unknown video task {name:?}: the tasks are t2va, fl2va and ref2va"
        ))
    })?;
    let short_edge = positive(args, "short_edge")?;
    let aspect_ratio = text(args, "aspect_ratio")?;
    let canvases = served_canvases(args)?;
    let target = Target {
        short_edge,
        aspect_ratio,
        duration_seconds: None,
    };
    match plan::named_canvas(task, &target, &canvases) {
        Ok(canvas) => Ok(json!([Value::Null, canvas.map(canvas_json)])),
        Err(VideoInputError::Invalid { field, message }) => {
            Ok(json!([[field.to_string(), message], Value::Null]))
        }
        Err(VideoInputError::Internal(message)) => Err(Refusal::runtime(message)),
    }
}

/// `video_image_grid(image_config, video_config, height, width) -> [t, h, w]`:
/// the patch grid the conditioner reads a still image on
/// (`plan::ProcessorConfig::image_grid`).
fn video_image_grid(args: &Value) -> Result<Value, Refusal> {
    let image = text(args, "image_config")?;
    let video = text(args, "video_config")?;
    let height = positive(args, "height")?;
    let width = positive(args, "width")?;
    let processor = plan::ProcessorConfig::parse(image, video)
        .map_err(|error| Refusal::value(error.to_string()))?;
    let grid = processor
        .image_grid(height, width)
        .map_err(|error| Refusal::value(error.to_string()))?;
    Ok(json!([grid.t, grid.h, grid.w]))
}

/// A canvas as the request vocabulary states it: `[width, height]`.
fn canvas_json(canvas: Canvas) -> Value {
    json!([canvas.width, canvas.height])
}

/// Canvases as one flat `[width, height, ...]`.
fn flat_canvases(canvases: &[Canvas]) -> Vec<u32> {
    canvases
        .iter()
        .flat_map(|canvas| [canvas.width, canvas.height])
        .collect()
}

/// A rule's frame counts from `frames`, as `[min_frames, clip_frames,
/// clip_latent_frames, tail_latent_frames]` ([`FrameClips`]).
fn frame_rule(args: &Value) -> Result<FrameClips, Refusal> {
    let mut counts = [0u32; 4];
    let values = args
        .get("frame_rule")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            Refusal::value(
                "frame_rule is [min_frames, clip_frames, clip_latent_frames, tail_latent_frames]",
            )
        })?;
    if values.len() != counts.len() {
        return Err(Refusal::value(
            "frame_rule is [min_frames, clip_frames, clip_latent_frames, tail_latent_frames]",
        ));
    }
    for (count, value) in counts.iter_mut().zip(values) {
        *count = positive_value(value, "frame_rule")?;
    }
    Ok(FrameClips {
        min_frames: counts[0],
        clip_frames: counts[1],
        clip_latent_frames: counts[2],
        tail_latent_frames: counts[3],
    })
}

/// The canvases a deployment serves from `canvases`, each
/// `[short_edge, aspect_ratio, width, height]`.
fn served_canvases(args: &Value) -> Result<Vec<ServedCanvas>, Refusal> {
    let entries = args
        .get("canvases")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            Refusal::value("canvases is [[short_edge, aspect_ratio, width, height], ...]")
        })?;
    entries
        .iter()
        .map(|entry| {
            let fields = entry.as_array().ok_or_else(|| {
                Refusal::value("a canvas is [short_edge, aspect_ratio, width, height]")
            })?;
            if fields.len() != 4 {
                return Err(Refusal::value(
                    "a canvas is [short_edge, aspect_ratio, width, height]",
                ));
            }
            let ratio = fields[1].as_str().ok_or_else(|| {
                Refusal::value("a canvas's aspect ratio is a string such as 16:9")
            })?;
            Ok(ServedCanvas {
                short_edge: positive_value(&fields[0], "short_edge")?,
                aspect_ratio: ratio.to_owned(),
                canvas: Canvas {
                    width: positive_value(&fields[2], "width")?,
                    height: positive_value(&fields[3], "height")?,
                },
            })
        })
        .collect()
}

/// The named argument `name`.
fn field<'a>(args: &'a Value, name: &str) -> Result<&'a Value, Refusal> {
    args.get(name)
        .ok_or_else(|| Refusal::value(format!("{name} is required")))
}

/// A named number.
fn number(args: &Value, name: &str) -> Result<f64, Refusal> {
    field(args, name)?
        .as_f64()
        .ok_or_else(|| Refusal::value(format!("{name} is a number")))
}

/// A named number, or `None` when it is absent or null.
fn optional_number(args: &Value, name: &str) -> Result<Option<f64>, Refusal> {
    match args.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => {
            Ok(Some(value.as_f64().ok_or_else(|| {
                Refusal::value(format!("{name} is a number"))
            })?))
        }
    }
}

/// A named string.
fn text<'a>(args: &'a Value, name: &str) -> Result<&'a str, Refusal> {
    field(args, name)?
        .as_str()
        .ok_or_else(|| Refusal::value(format!("{name} is a string")))
}

/// A named positive count.
fn positive(args: &Value, name: &str) -> Result<u32, Refusal> {
    positive_value(field(args, name)?, name)
}

/// A positive count from a JSON value.
fn positive_value(value: &Value, name: &str) -> Result<u32, Refusal> {
    value
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|&value| value > 0)
        .ok_or_else(|| Refusal::value(format!("{name} is a positive count")))
}
