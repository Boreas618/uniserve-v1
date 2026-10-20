//! Video request sizes, denoiser layouts and admission planning.
//!
//! A video request's numerical size is its output frame count, canvas,
//! prompt length, conditions (in request order) and the vision spans of its
//! presented prompt. The denoiser evaluates it in a *layout*: a size whose
//! shapes, constants and captured graphs every request it holds shares
//! (`VideoDenoiser.layout_size`/`holds`). [`MediaPlan`] fixes, at startup,
//! the frame counts, canvases and prompt capacities a rank admits and the
//! layouts it prepares; once the model sized a request, [`MediaPlan::select`]
//! admits it in the layout of the smallest prepared prompt capacity that
//! holds it.
//!
//! The denoiser is the authority on the sizes it serves and on the rows a
//! request's conditions take (its `make_size`): both are asked of the model.
//! A request whose size the denoiser or the plan does not serve is refused
//! alone, for the reason the refusal states: its calls report the refusal
//! (`RequestPool::refuse`), and the requests sized beside it are admitted as
//! usual.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_core::video::plan::requested_frames;
use uniserve_core::{Canvas, ConditionRole, ServedCanvas, VideoCondition};
use uniserve_ipc::{BufferId, NewRequest, RequestKey};

use crate::executor::lifecycle::{AdmittedSlot, LaneJobs, RequestStorage};
use crate::ffi::integer_list;
use crate::host::mux::{MuxConfig, MuxSession};
use crate::lane::{Completion, Job};
use crate::runner::denoiser::runners::DenoiserRunners;
use crate::runner::denoiser::{Evicted, VideoModel};
use crate::runner::media::MediaRunner;
use crate::{Error, Result};

/// A layout descriptor: `[frames, width, height, text_rows, condition_rows]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct VideoLayout {
    pub(crate) frames: u32,
    pub(crate) canvas: Canvas,
    /// Rows of the layout's prompt region.
    pub(crate) text_rows: u32,
    /// Rows of the layout's condition region.
    pub(crate) condition_rows: u32,
}

impl VideoLayout {
    pub(crate) fn descriptor(&self) -> Any {
        Any::from(integer_list(self.values()))
    }

    /// The layout's extents, as [`Self::from_values`] reads them.
    pub(crate) fn values(&self) -> [i64; 5] {
        [
            i64::from(self.frames),
            i64::from(self.canvas.width),
            i64::from(self.canvas.height),
            i64::from(self.text_rows),
            i64::from(self.condition_rows),
        ]
    }

    pub(crate) fn from_values(values: &[i64]) -> Result<Self> {
        let narrow = |value: i64| {
            u32::try_from(value)
                .map_err(|_| Error::Interface(format!("layout extent {value} is invalid")))
        };
        let [frames, width, height, text, rows] = values else {
            return Err(Error::Interface(
                "a layout descriptor has five extents".into(),
            ));
        };
        Ok(Self {
            frames: narrow(*frames)?,
            canvas: Canvas {
                width: narrow(*width)?,
                height: narrow(*height)?,
            },
            text_rows: narrow(*text)?,
            condition_rows: narrow(*rows)?,
        })
    }
}

/// A condition as the denoiser sizes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ConditionShape {
    pub(crate) role: ConditionRole,
    /// Encoded pixel frames and raster; `None` for audio alone.
    pub(crate) pixels: Option<(u32, Canvas)>,
    /// Model-rate audio samples; zero without a track.
    pub(crate) audio_samples: u64,
}

impl ConditionShape {
    pub(crate) fn of(condition: &VideoCondition) -> Self {
        Self {
            role: condition.role,
            pixels: condition.media.pixels(),
            audio_samples: condition.media.audio().map_or(0, |track| track.samples),
        }
    }

    /// `[role, frames, width, height, audio_samples]`.
    fn descriptor(&self) -> [i64; 5] {
        let role = match self.role {
            ConditionRole::FirstFrame => 0,
            ConditionRole::LastFrame => 1,
            ConditionRole::Reference => 2,
        };
        let (frames, canvas) = self.pixels.unwrap_or((
            0,
            Canvas {
                width: 0,
                height: 0,
            },
        ));
        [
            role,
            i64::from(frames),
            i64::from(canvas.width),
            i64::from(canvas.height),
            self.audio_samples as i64,
        ]
    }
}

/// A request's exact numerical size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VideoSize {
    pub(crate) frames: u32,
    pub(crate) canvas: Canvas,
    /// Presented prompt tokens.
    pub(crate) tokens: u32,
    pub(crate) conditions: Arc<[ConditionShape]>,
    /// `(start, stop)` token ranges of the prompt's vision blocks.
    pub(crate) spans: Arc<[(u32, u32)]>,
}

impl VideoSize {
    /// A size without conditions.
    pub(crate) fn plain(frames: u32, canvas: Canvas, tokens: u32) -> Self {
        Self {
            frames,
            canvas,
            tokens,
            conditions: Arc::from([]),
            spans: Arc::from([]),
        }
    }

    /// The size descriptor `[frames, width, height, tokens]`.
    pub(crate) fn descriptor(&self) -> Any {
        Any::from(integer_list(self.values()))
    }

    /// The flat conditions descriptor, in request order.
    pub(crate) fn conditions_descriptor(&self) -> Any {
        Any::from(integer_list(self.condition_values()))
    }

    /// The flat vision spans descriptor.
    pub(crate) fn spans_descriptor(&self) -> Any {
        Any::from(integer_list(self.span_values()))
    }

    /// The integers of [`VideoSize::descriptor`].
    pub(crate) fn values(&self) -> Vec<i64> {
        vec![
            i64::from(self.frames),
            i64::from(self.canvas.width),
            i64::from(self.canvas.height),
            i64::from(self.tokens),
        ]
    }

    /// The integers of [`VideoSize::conditions_descriptor`].
    pub(crate) fn condition_values(&self) -> Vec<i64> {
        self.conditions
            .iter()
            .flat_map(ConditionShape::descriptor)
            .collect()
    }

    /// The integers of [`VideoSize::spans_descriptor`].
    pub(crate) fn span_values(&self) -> Vec<i64> {
        self.spans
            .iter()
            .flat_map(|&(start, stop)| [i64::from(start), i64::from(stop)])
            .collect()
    }
}

/// The vision blocks of a presented prompt: maximal runs of tag 0 in its
/// token tags, as `(start, stop)` token ranges.
pub(crate) fn vision_spans(tags: &[u8]) -> Vec<(u32, u32)> {
    let mut spans = Vec::new();
    let mut start = None;
    for (index, &tag) in tags.iter().enumerate() {
        match (tag == 0, start) {
            (true, None) => start = Some(index as u32),
            (false, Some(first)) => {
                spans.push((first, index as u32));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(first) = start {
        spans.push((first, tags.len() as u32));
    }
    spans
}

/// The refusal of a request whose frame count, canvas or prompt lies outside
/// what the rank admits.
const OUTSIDE_PLAN: &str = "media input exceeds the worker frame, canvas or conditioning capacity";

/// What a rank admits and the layouts it prepares, fixed at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MediaPlan {
    /// Admitted frame counts, ascending.
    pub(crate) frame_counts: Vec<u32>,
    /// Admitted canvases in the order the deployment names them, with the
    /// names the denoiser gives them, which the handshake reports.
    pub(crate) admitted: Vec<ServedCanvas>,
    /// The admitted canvases in preparation order (see
    /// [`preparation_order`]).
    pub(crate) canvases: Vec<Canvas>,
    /// Prompt capacities of the prepared layouts, ascending.
    pub(crate) text_capacities: Vec<u32>,
    /// The longest admitted prompt.
    pub(crate) max_text_tokens: u32,
    /// Condition rows a request may occupy: the deployment's stated
    /// capacity, or without one the rows of the largest condition set the
    /// denoiser admits; zero for a denoiser that admits none.
    pub(crate) condition_rows: u32,
    /// The leading canvas's longest layout at the prompt capacity, widened
    /// to the condition capacity. The denoiser prepares it first, so it
    /// sizes the storage every layout shares.
    pub(crate) maximum: VideoLayout,
    /// Startup layouts: canvases (in preparation order) by frame counts
    /// (longest first) by prompt capacities (largest first). None for a
    /// denoiser that is not text-only: every request it admits brings
    /// conditions and evaluates in a layout of its own.
    pub(crate) layouts: Vec<VideoLayout>,
    /// Each canvas's longest layout at the prompt capacity, widened to the
    /// condition capacity; their envelope bounds every request's state.
    pub(crate) largest: Vec<VideoLayout>,
}

/// A request admitted against a plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Admission {
    pub(crate) size: VideoSize,
    /// The request's condition rows.
    pub(crate) condition_rows: u32,
    /// The layout the request evaluates in.
    pub(crate) layout: VideoLayout,
    /// The prompt capacity the layout was selected at.
    pub(crate) text_capacity: u32,
}

/// One admitted canvas as preparation orders it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CanvasExtent {
    canvas: Canvas,
    /// State and workspace elements of the canvas's largest layout.
    storage: u64,
    /// Generated video elements of the canvas's longest video.
    generated: u64,
}

/// The canvases a deployment prepares and admits, in the order it names
/// them.
///
/// `offered` are the canvases the denoiser generates, each named by the
/// `(short_edge, aspect_ratio)` a request selects it with. Every resolution
/// class of `resolutions` (`768p`) is crossed with every ratio of
/// `aspect_ratios` (`16:9`), resolution-major, and each pair admits the
/// canvas the denoiser names by it. Without resolutions, or without aspect
/// ratios, every one the denoiser offers takes their place, in its own
/// order; without either, every offered canvas is admitted.
///
/// Fails, as an invalid setup, for a resolution that is not a class or a
/// repeated name, and, as an unsupported setup, when the denoiser offers no
/// canvas or names none by a selected pair.
pub(crate) fn admitted_canvases(
    offered: &[ServedCanvas],
    resolutions: &[String],
    aspect_ratios: &[String],
) -> Result<Vec<ServedCanvas>> {
    if offered.is_empty() {
        return Err(Error::Unsupported(
            "the video denoiser prepares no canvas".into(),
        ));
    }
    if resolutions.is_empty() && aspect_ratios.is_empty() {
        return Ok(offered.to_vec());
    }

    let mut edges = Vec::with_capacity(resolutions.len());
    for name in resolutions {
        let edge = name
            .strip_suffix('p')
            .filter(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|digits| digits.parse::<u32>().ok())
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "a video resolution is a class such as 768p, got {name:?}"
                ))
            })?;
        edges.push(edge);
    }
    // Without names, every one the denoiser offers, in its own order.
    if edges.is_empty() {
        for served in offered {
            if !edges.contains(&served.short_edge) {
                edges.push(served.short_edge);
            }
        }
    }
    let mut ratios: Vec<&str> = aspect_ratios.iter().map(String::as_str).collect();
    if ratios.is_empty() {
        for served in offered {
            if !ratios.contains(&served.aspect_ratio.as_str()) {
                ratios.push(&served.aspect_ratio);
            }
        }
    }
    let distinct = |count: usize, unique: usize| count == unique;
    if !distinct(edges.len(), edges.iter().collect::<HashSet<_>>().len())
        || !distinct(ratios.len(), ratios.iter().collect::<HashSet<_>>().len())
    {
        return Err(Error::Invalid(
            "a video resolution or aspect ratio repeats".into(),
        ));
    }

    let mut admitted = Vec::with_capacity(edges.len() * ratios.len());
    for &edge in &edges {
        for &ratio in &ratios {
            let served = offered
                .iter()
                .find(|served| served.short_edge == edge && served.aspect_ratio == ratio)
                .ok_or_else(|| {
                    let generated: Vec<String> = offered
                        .iter()
                        .map(|served| format!("{}p {}", served.short_edge, served.aspect_ratio))
                        .collect();
                    Error::Unsupported(format!(
                        "the video denoiser generates only {}, not {edge}p {ratio}",
                        generated.join(", ")
                    ))
                })?;
            admitted.push(served.clone());
        }
    }
    Ok(admitted)
}

/// The first default text capacity, in prompt tokens: a short prompt pads to
/// it alone.
const TEXT_CAPACITY_FIRST: u32 = 1024;

/// The step, in prompt tokens, of the default text capacities above the
/// first. A request's layout holds at most one step of padding rows beyond
/// its prompt's tiles.
const TEXT_CAPACITY_STEP: u32 = 2048;

/// The prompt capacities, in tokens, a deployment prepares layouts for, in
/// increasing order.
///
/// The prompt capacity `max_text_tokens` (`--max-model-len`) is always the
/// largest. `listed` (`video_config.text_capacities`) names the capacities below
/// it; a listed capacity equal to it repeats the largest. Without a list the
/// capacities are [`TEXT_CAPACITY_FIRST`], then steps of
/// [`TEXT_CAPACITY_STEP`] below the prompt capacity.
///
/// Fails, as an invalid setting, for a listed capacity of zero or one above
/// the prompt capacity.
fn prompt_capacities(listed: &[u32], max_text_tokens: u32) -> Result<Vec<u32>> {
    if listed
        .iter()
        .any(|&tokens| tokens == 0 || tokens > max_text_tokens)
    {
        return Err(Error::Invalid(format!(
            "video text capacities must be positive and not exceed the prompt capacity \
             {max_text_tokens}, got {listed:?}"
        )));
    }
    let below: Vec<u32> = if listed.is_empty() {
        std::iter::once(TEXT_CAPACITY_FIRST)
            .chain((TEXT_CAPACITY_STEP..max_text_tokens).step_by(TEXT_CAPACITY_STEP as usize))
            .filter(|&tokens| tokens < max_text_tokens)
            .collect()
    } else {
        listed.to_vec()
    };
    let mut capacities: Vec<u32> = below
        .into_iter()
        .chain(std::iter::once(max_text_tokens))
        .collect();
    capacities.sort_unstable();
    capacities.dedup();
    Ok(capacities)
}

/// The order a deployment prepares its canvases in.
///
/// The leading canvas is the one whose largest layout takes the most state
/// and workspace, the first such canvas on ties: its layout is prepared
/// first and sizes the storage every later layout shares. Generated rows
/// alone do not select it: 768p 21:9 generates as many rows as 16:9 but
/// pads to more sparse-attention tiles. The other canvases follow in
/// decreasing generated elements, in their given order on ties.
fn preparation_order(extents: &[CanvasExtent]) -> Vec<Canvas> {
    let mut leading = None;
    for extent in extents {
        if leading.is_none_or(|first: &CanvasExtent| extent.storage > first.storage) {
            leading = Some(extent);
        }
    }
    let mut rest: Vec<&CanvasExtent> = extents
        .iter()
        .filter(|extent| leading.is_some_and(|first| first.canvas != extent.canvas))
        .collect();
    // A stable sort keeps the given order among equal extents.
    rest.sort_by_key(|extent| std::cmp::Reverse(extent.generated));
    leading
        .into_iter()
        .chain(rest)
        .map(|extent| extent.canvas)
        .collect()
}

impl MediaPlan {
    /// Plan a deployment's admitted sizes on the loaded denoiser.
    ///
    /// `requested` are the configured prompt capacities below the prompt
    /// capacity `max_text_tokens`, which is always the largest
    /// ([`prompt_capacities`]); each maps to the text rows of its layout,
    /// capped at the largest layout's.
    /// `stated_rows` is the deployment's condition capacity
    /// (`video_config.max_condition_rows`); without it the plan provisions the rows of
    /// the largest condition set the denoiser admits at the longest frame
    /// count, on the admitted canvas where it packs the most, so no request
    /// is refused for its conditions. A denoiser that admits no condition
    /// gets none either way. `admitted` are the canvases the deployment
    /// prepares ([`admitted_canvases`]), in the order it names them.
    pub(crate) fn new(
        model: &VideoModel,
        max_seconds: f64,
        min_seconds: Option<f64>,
        max_text_tokens: u32,
        requested: &[u32],
        stated_rows: Option<u32>,
        admitted: Vec<ServedCanvas>,
    ) -> Result<Self> {
        let facts = &model.facts;
        let longest = model.legal_frame_count(requested_frames(max_seconds, facts.frame_rate))?;
        let mut count = model.legal_frame_count(
            min_seconds
                .map_or(1, |seconds| requested_frames(seconds, facts.frame_rate))
                .max(1),
        )?;
        let mut frame_counts = vec![count];
        while count < longest {
            count = model.legal_frame_count(count + 1)?;
            frame_counts.push(count);
        }
        if count != longest {
            return Err(Error::Unsupported(
                "the video denoiser admits no frame count of the duration capacity".into(),
            ));
        }
        let mut admitted_rows = 0;
        for served in &admitted {
            admitted_rows = admitted_rows.max(model.max_condition_rows(longest, served.canvas)?);
        }
        let condition_rows = if admitted_rows == 0 {
            0
        } else {
            stated_rows.unwrap_or(admitted_rows)
        };
        let widen = |layout: VideoLayout| {
            if condition_rows == 0 {
                Ok(layout)
            } else {
                model.condition_layout(&layout, condition_rows)
            }
        };

        // Each canvas's longest layout at the prompt capacity, widened to
        // the condition capacity, bounds every layout of the canvas.
        let mut largest = Vec::with_capacity(admitted.len());
        let mut extents = Vec::with_capacity(admitted.len());
        for canvas in admitted.iter().map(|served| served.canvas) {
            let plain = model.layout_size(&VideoSize::plain(longest, canvas, max_text_tokens))?;
            let layout = widen(plain)?;
            extents.push(CanvasExtent {
                canvas,
                storage: model.layout_fields(&layout)?.storage(),
                // Global sample extents order the canvases: a rank's share
                // moves with the prompt under sequence parallelism.
                generated: model.latent_elements("video", &plain)?,
            });
            largest.push(layout);
        }
        let ordered = preparation_order(&extents);
        let reference = ordered[0];
        let leading = model.layout_size(&VideoSize::plain(longest, reference, max_text_tokens))?;
        let largest_text = leading.text_rows;

        let mut text_capacities = prompt_capacities(requested, max_text_tokens)?
            .iter()
            .map(|&tokens| {
                let layout = model.layout_size(&VideoSize::plain(longest, reference, tokens))?;
                Ok(layout.text_rows.min(largest_text))
            })
            .collect::<Result<Vec<_>>>()?;
        text_capacities.sort_unstable();
        text_capacities.dedup();

        // Only a request without conditions evaluates in a startup layout,
        // so a denoiser that is not text-only prepares none.
        let mut layouts = Vec::new();
        if facts.text_only {
            for &canvas in &ordered {
                for &frames in frame_counts.iter().rev() {
                    for &tokens in text_capacities.iter().rev() {
                        layouts.push(model.layout_size(&VideoSize::plain(frames, canvas, tokens))?);
                    }
                }
            }
        }
        let maximum = widen(leading)?;
        Ok(Self {
            frame_counts,
            admitted,
            canvases: ordered,
            text_capacities,
            max_text_tokens,
            condition_rows,
            maximum,
            layouts,
            largest,
        })
    }

    /// The longest admitted video's frames.
    pub(crate) fn max_frames(&self) -> u32 {
        self.frame_counts.last().copied().unwrap_or(0)
    }

    /// The refusal of a size outside the plan's frame counts, canvases or
    /// prompt capacity; `None` for a size the plan admits.
    fn refusal(&self, size: &VideoSize) -> Option<String> {
        (!self.frame_counts.contains(&size.frames)
            || !self.canvases.contains(&size.canvas)
            || size.tokens > self.max_text_tokens)
            .then(|| OUTSIDE_PLAN.to_owned())
    }

    /// The layout of a size without conditions at prompt capacity
    /// `capacity`: one of the startup layouts, in the order
    /// [`Self::new`] lists them.
    fn startup_layout(&self, frames: u32, canvas: Canvas, capacity: u32) -> Option<VideoLayout> {
        let position = |values: &[u32], value: u32| values.iter().position(|&each| each == value);
        let canvas = self.canvases.iter().position(|&each| each == canvas)?;
        let frames = self.frame_counts.len() - 1 - position(&self.frame_counts, frames)?;
        let capacities = self.text_capacities.len();
        let capacity = capacities - 1 - position(&self.text_capacities, capacity)?;
        self.layouts
            .get((canvas * self.frame_counts.len() + frames) * capacities + capacity)
            .copied()
    }

    /// The admission of a request the model sized: `condition_rows` are its
    /// condition rows, `layouts` the layout of each prompt capacity and
    /// `holds` whether each holds the request. The request evaluates in
    /// the layout of the smallest capacity that holds it.
    ///
    /// Returns the request's refusal when its conditions take more rows
    /// than the deployment provisioned, naming both and the option that
    /// raises the capacity, or when no capacity's layout holds it.
    fn select(
        &self,
        size: VideoSize,
        condition_rows: u32,
        layouts: &[VideoLayout],
        holds: &[bool],
    ) -> std::result::Result<Admission, String> {
        if condition_rows > self.condition_rows {
            return Err(format!(
                "conditions: the conditions take {condition_rows} denoiser rows, more than the \
                 {} this deployment serves; serve with a larger video_config.max_condition_rows",
                self.condition_rows
            ));
        }
        self.text_capacities
            .iter()
            .zip(layouts)
            .zip(holds)
            .find(|(_, held)| **held)
            .map(|((&text_capacity, &layout), _)| Admission {
                size,
                condition_rows,
                layout,
                text_capacity,
            })
            .ok_or_else(|| "media input has no admitted capacity layout".to_owned())
    }
}

/// The request-order conditions of an admission, as the denoiser sizes them.
pub(crate) fn condition_shapes(conditions: &[VideoCondition]) -> Arc<[ConditionShape]> {
    conditions.iter().map(ConditionShape::of).collect()
}

/// One admitted video request on this rank.
pub(crate) struct VideoRequest {
    pub(crate) key: RequestKey,
    pub(crate) frames: u32,
    pub(crate) canvas: Canvas,
    /// The prompt's token ids.
    pub(crate) tokens: Arc<[i64]>,
    pub(crate) conditions: Arc<[VideoCondition]>,
    /// The request's layout on a denoising rank.
    pub(crate) admission: Option<Admission>,
    /// The denoiser's prepared layout id of `admission`.
    pub(crate) layout: Option<usize>,
    /// Frames of each media unit.
    pub(crate) units: Arc<[u32]>,
    /// Samples of the generated audio track.
    pub(crate) audio_samples: u64,
    /// The seed of the request's noise.
    pub(crate) seed: u64,
    /// The request's MP4 on the muxer rank.
    pub(crate) mux: Option<Arc<MuxSession>>,
}

impl VideoRequest {
    /// The longest media unit's frames.
    pub(crate) fn longest(&self) -> u32 {
        self.units.iter().copied().max().unwrap_or(0)
    }
}

/// A started request on a denoising rank whose layout the library is asked
/// for.
struct Unsized {
    slot: usize,
    key: RequestKey,
    size: VideoSize,
    /// The rows the request's conditions take, as the denoiser packs them:
    /// zero without conditions, otherwise asked of the model once it serves
    /// the size.
    condition_rows: Option<u32>,
    /// Whether the model was asked whether it serves the size: a size
    /// without conditions is one the plan's startup layouts were sized for.
    screened: bool,
    /// The layout of each prompt capacity of the plan, in its order: a
    /// startup layout for a size without conditions, otherwise asked for.
    layouts: Vec<Option<VideoLayout>>,
    /// The request's admission once its layout is selected; it waits here
    /// while the denoiser prepares a layout it does not hold.
    admitted: Option<Admission>,
}

/// What the job a sizing submitted last was asked.
enum Asked {
    /// Whether the denoiser serves the sizes with conditions.
    Refusals,
    /// The layouts and condition rows of sizes with conditions.
    Layouts,
    /// Which of each request's layouts hold it.
    Holds,
    /// The preparation of these layouts, which displaced the held ones.
    Preparation(Vec<VideoLayout>, Vec<Evicted>),
}

/// The admission of the video requests a batch started, between the jobs
/// that ask the library to size them.
///
/// The layout of a request's size at each prompt capacity and whether a
/// layout holds it are the model's arithmetic over the request's own size,
/// so no table answers them: every request is asked for, in jobs of the
/// Python thread. A request without conditions takes one job (its layouts
/// are startup layouts); a conditioned one takes a job for its layouts
/// first. A request whose layout the denoiser does not hold (one prepared at
/// startup or for an earlier request) takes a further job, on the
/// denoiser's lane, that prepares the layout.
///
/// The requests a batch started are sized together. A request whose size
/// the denoiser or the plan does not serve leaves the sizing refused, with
/// the reason [`Sizing::take_refused`] hands the rank; a failure of the
/// sizing itself fails the batch, and with it every request of the sizing
/// that was not admitted yet.
pub(crate) struct Sizing {
    requests: Vec<Unsized>,
    asked: Asked,
    /// Requests refused since the rank last took them, with their reasons.
    refused: Vec<(RequestKey, String)>,
}

impl Sizing {
    /// The requests refused since the last call, with their reasons.
    pub(crate) fn take_refused(&mut self) -> Vec<(RequestKey, String)> {
        std::mem::take(&mut self.refused)
    }
}

/// What the model answers about a video of one admitted frame count.
struct AdmittedDuration {
    /// Frames of each media unit.
    units: Arc<[u32]>,
    /// Samples of the generated audio track.
    audio_samples: u64,
}

/// The video requests admitted on a rank, with the model and the plan that
/// admit them.
pub(crate) struct VideoRequests {
    pub(crate) model: VideoModel,
    pub(crate) plan: MediaPlan,
    /// Every admitted frame count's units and track, read from the model
    /// when the rank starts: a request is admitted without asking the
    /// library for them.
    durations: HashMap<u32, AdmittedDuration>,
    /// Admitted requests by slot.
    requests: HashMap<usize, VideoRequest>,
    /// The started requests no batch asked the library to size yet.
    waiting: Vec<Unsized>,
}

impl VideoRequests {
    /// The requests of a rank that admits `plan`. Asks `model` for the
    /// units and the track of every admitted frame count.
    pub(crate) fn new(model: VideoModel, plan: MediaPlan) -> Result<Self> {
        let durations = plan
            .frame_counts
            .iter()
            .map(|&frames| {
                let duration = AdmittedDuration {
                    units: model.unit_frames(frames)?.into(),
                    audio_samples: model.track_samples(frames)?,
                };
                Ok((frames, duration))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            model,
            plan,
            durations,
            requests: HashMap::new(),
            waiting: Vec::new(),
        })
    }

    /// Admit a started video request with its exact size. On a denoising
    /// rank the layout it evaluates in is the model's answer, which the
    /// batch that started it asks for next ([`Self::begin_sizing`]); until
    /// then the request has no layout.
    ///
    /// Returns the request's refusal when its frame count, canvas or prompt
    /// lies outside what the rank's plan admits; the caller refuses the
    /// request ([`RequestPool::refuse`](crate::request::RequestPool::refuse)).
    /// Fails, as an invalid descriptor, for a request whose task and
    /// conditions, or computation bounds, contradict each other.
    pub(crate) fn admit(
        &mut self,
        admission: &NewRequest,
        denoising: bool,
    ) -> Result<Option<String>> {
        let (Some(media), Some(video)) = (&admission.diffusion, &admission.video) else {
            return Ok(None);
        };
        let slot = admission.request_pool_idx as usize;
        let canvas = Canvas {
            width: media.width,
            height: media.height,
        };
        if (video.task == uniserve_core::VideoTask::T2va) != video.conditions.is_empty() {
            return Err(Error::Invalid(
                "a video request's task disagrees with its conditions".into(),
            ));
        }
        let Some(duration) = self
            .durations
            .get(&media.num_frames)
            .filter(|_| self.plan.canvases.contains(&canvas))
        else {
            return Ok(Some(OUTSIDE_PLAN.to_owned()));
        };
        // The request's decode units (`media.video_units`) derive from the
        // frame rule this rank declared, so they need no comparison with the
        // units of its own decoder's slicing.
        let units = Arc::clone(&duration.units);
        let conditions: Arc<[VideoCondition]> = video.conditions.clone().into();
        let size = VideoSize {
            frames: media.num_frames,
            canvas,
            tokens: admission.prompt_token_ids.len() as u32,
            conditions: condition_shapes(&conditions),
            spans: vision_spans(&video.text_tags).into(),
        };
        let request = VideoRequest {
            key: admission.request_key,
            frames: media.num_frames,
            canvas,
            tokens: admission
                .prompt_token_ids
                .iter()
                .map(|&token| i64::from(token))
                .collect(),
            conditions,
            admission: None,
            layout: None,
            audio_samples: duration.audio_samples,
            units,
            seed: media.seed,
            mux: None,
        };
        if denoising {
            if let Some(refusal) = self.plan.refusal(&size) {
                return Ok(Some(refusal));
            }
            let conditioned = !size.conditions.is_empty();
            // A request restarted in the slot replaces the one waiting.
            self.waiting.retain(|waiting| waiting.slot != slot);
            self.waiting.push(Unsized {
                slot,
                key: admission.request_key,
                layouts: self
                    .plan
                    .text_capacities
                    .iter()
                    .map(|&capacity| {
                        size.conditions
                            .is_empty()
                            .then(|| self.plan.startup_layout(size.frames, size.canvas, capacity))
                            .flatten()
                    })
                    .collect(),
                size,
                condition_rows: (!conditioned).then_some(0),
                screened: !conditioned,
                admitted: None,
            });
        }
        self.requests.insert(slot, request);
        Ok(None)
    }

    /// The requests started since the last call, for the batch that
    /// started them to have sized. `None` when there is none.
    pub(crate) fn begin_sizing(&mut self) -> Option<Sizing> {
        let requests = std::mem::take(&mut self.waiting);
        (!requests.is_empty()).then_some(Sizing {
            requests,
            refused: Vec::new(),
            asked: Asked::Holds,
        })
    }

    /// Record into `job` the next step of `sizing`: whether the denoiser
    /// serves each size with conditions; then the layouts and condition rows
    /// not yet known; once every layout is known, whether each of a
    /// request's layouts holds it; and for requests whose selected layout
    /// `denoisers` does not hold, that layout's preparation. Returns whether
    /// the job belongs on the denoiser's lane (a preparation works on its
    /// stream) rather than on the Python thread alone.
    pub(crate) fn ask(
        &self,
        sizing: &mut Sizing,
        denoisers: &mut DenoiserRunners,
        job: &Job,
    ) -> Result<bool> {
        // A size the denoiser does not serve is refused before anything else
        // about it is asked, since every other question would fail on it.
        let unscreened: Vec<&Unsized> = sizing
            .requests
            .iter()
            .filter(|request| !request.screened)
            .collect();
        if !unscreened.is_empty() {
            for request in unscreened {
                self.model.ask_refusal(&request.size, job);
            }
            sizing.asked = Asked::Refusals;
            return Ok(false);
        }

        // Selected requests wait for layouts the denoiser lacks.
        let mut lacking: Vec<VideoLayout> = Vec::new();
        for admitted in sizing
            .requests
            .iter()
            .filter_map(|request| request.admitted.as_ref())
        {
            if denoisers.lacks(&admitted.layout) && !lacking.contains(&admitted.layout) {
                lacking.push(admitted.layout);
            }
        }
        if !lacking.is_empty() {
            let evicted = lacking
                .iter()
                .map(|&layout| denoisers.ask_layout(&self.model, layout, job))
                .collect::<Result<Vec<_>>>()?;
            sizing.asked = Asked::Preparation(lacking, evicted);
            return Ok(true);
        }

        let capacities = &self.plan.text_capacities;
        let layouts = sizing
            .requests
            .iter()
            .any(|request| request.layouts.contains(&None));
        for request in &sizing.requests {
            if layouts {
                // The layout of the request's size at each prompt capacity,
                // without its vision blocks.
                for (&capacity, _) in capacities
                    .iter()
                    .zip(&request.layouts)
                    .filter(|(_, layout)| layout.is_none())
                {
                    let candidate = VideoSize {
                        tokens: capacity,
                        spans: Arc::from([]),
                        ..request.size.clone()
                    };
                    self.model.ask_layout_size(&candidate, job);
                }
                if request.condition_rows.is_none() {
                    self.model.ask_condition_rows(&request.size, job);
                }
                continue;
            }
            for layout in request.layouts.iter().flatten() {
                self.model.ask_holds(layout, &request.size, job);
            }
        }
        sizing.asked = if layouts {
            Asked::Layouts
        } else {
            Asked::Holds
        };
        Ok(false)
    }

    /// Take the answers to what [`Self::ask`] recorded from the job's
    /// `completion`. A request whose layout is selected and held in
    /// `denoisers` is admitted with it and leaves the sizing; returns
    /// whether every request has (otherwise `sizing` has more to ask).
    ///
    /// A request that ended or was restarted while it was sized is left
    /// out. A request whose size the denoiser or the plan does not serve
    /// leaves the sizing refused ([`Sizing::take_refused`]).
    pub(crate) fn answer(
        &mut self,
        sizing: &mut Sizing,
        completion: &mut Completion,
        denoisers: &mut DenoiserRunners,
    ) -> Result<bool> {
        let integer = |values: Vec<i64>| {
            values
                .first()
                .copied()
                .ok_or_else(|| Error::Interface("a sizing function returned no value".into()))
        };
        match std::mem::replace(&mut sizing.asked, Asked::Holds) {
            Asked::Refusals => {
                let mut served = Vec::with_capacity(sizing.requests.len());
                for mut request in std::mem::take(&mut sizing.requests) {
                    if request.screened {
                        served.push(request);
                        continue;
                    }
                    let reason = completion.texts()?.concat();
                    if reason.is_empty() {
                        request.screened = true;
                        served.push(request);
                    } else {
                        sizing.refused.push((request.key, reason));
                    }
                }
                sizing.requests = served;
                return Ok(sizing.requests.is_empty());
            }
            Asked::Layouts => {
                for request in &mut sizing.requests {
                    for layout in request.layouts.iter_mut().filter(|layout| layout.is_none()) {
                        *layout = Some(VideoLayout::from_values(&completion.returned()?)?);
                    }
                    if request.condition_rows.is_none() {
                        let rows = integer(completion.returned()?)?;
                        request.condition_rows = Some(u32::try_from(rows).map_err(|_| {
                            Error::Interface(format!(
                                "the denoiser counts {rows} condition rows for a request"
                            ))
                        })?);
                    }
                }
                return Ok(false);
            }
            Asked::Holds => {
                let mut selected = Vec::with_capacity(sizing.requests.len());
                for mut waiting in std::mem::take(&mut sizing.requests) {
                    let layouts: Vec<VideoLayout> =
                        waiting.layouts.iter().flatten().copied().collect();
                    let holds = layouts
                        .iter()
                        .map(|_| Ok(integer(completion.returned()?)? != 0))
                        .collect::<Result<Vec<bool>>>()?;
                    let rows = waiting.condition_rows.ok_or_else(|| {
                        Error::Invariant("a sized request's condition rows are known".into())
                    })?;
                    match self
                        .plan
                        .select(waiting.size.clone(), rows, &layouts, &holds)
                    {
                        Ok(admission) => {
                            waiting.admitted = Some(admission);
                            selected.push(waiting);
                        }
                        Err(reason) => sizing.refused.push((waiting.key, reason)),
                    }
                }
                sizing.requests = selected;
            }
            Asked::Preparation(layouts, evicted) => {
                for layout in layouts {
                    denoisers.accept_layout(&self.model, layout, completion)?;
                }
                // The job drained the stream the displaced layouts' last
                // steps ran on.
                drop(evicted);
            }
        }

        // Admit every selected request whose layout is held. A request
        // leaves the sizing once it is admitted, so one that fails is still
        // there for [`Self::abandon`], with those behind it.
        let mut waiting = Vec::new();
        for request in std::mem::take(&mut sizing.requests) {
            let held = request
                .admitted
                .as_ref()
                .is_some_and(|admitted| !denoisers.lacks(&admitted.layout));
            if !held || !waiting.is_empty() {
                waiting.push(request);
                continue;
            }
            if let Err(error) = self.admit_sized(&request, denoisers) {
                waiting.push(request);
                sizing.requests = waiting;
                return Err(error);
            }
        }
        sizing.requests = waiting;
        Ok(sizing.requests.is_empty())
    }

    /// Hold the selected layout of `sized` in `denoisers` and admit its
    /// request with it, unless the request ended or was restarted.
    fn admit_sized(&mut self, sized: &Unsized, denoisers: &mut DenoiserRunners) -> Result<()> {
        let Some(admitted) = &sized.admitted else {
            return Ok(());
        };
        let live = self
            .requests
            .get(&sized.slot)
            .is_some_and(|request| request.key == sized.key && request.admission.is_none());
        if !live {
            return Ok(());
        }
        let layout = denoisers.admit(sized.slot, admitted.layout)?;
        if let Some(request) = self.requests.get_mut(&sized.slot) {
            request.layout = layout;
            request.admission = Some(admitted.clone());
        }
        Ok(())
    }

    /// Forget the requests of a sizing that failed and were not admitted:
    /// their calls are refused as calls of requests this rank did not
    /// admit.
    pub(crate) fn abandon(&mut self, sizing: Sizing) {
        for waiting in sizing.requests {
            let unadmitted = self
                .requests
                .get(&waiting.slot)
                .is_some_and(|request| request.key == waiting.key && request.admission.is_none());
            if unadmitted {
                self.requests.remove(&waiting.slot);
            }
        }
    }

    pub(crate) fn request(&self, slot: usize, key: RequestKey) -> Result<&VideoRequest> {
        self.requests
            .get(&slot)
            .filter(|request| request.key == key)
            .ok_or_else(|| Error::Invalid("video call has no admitted media dimensions".into()))
    }

    /// The MP4 session of a request on the muxer rank, created on first use.
    pub(crate) fn mux_session(
        &mut self,
        slot: usize,
        key: RequestKey,
        media: Option<&MediaRunner>,
    ) -> Result<Arc<MuxSession>> {
        let functions = media
            .ok_or_else(|| Error::Invalid("this rank holds no muxer".into()))?
            .mux
            .clone();
        let frame_rate = self.model.facts.frame_rate;
        let audio_rate = self.model.facts.sample_rate;
        let request = self
            .requests
            .get_mut(&slot)
            .filter(|request| request.key == key)
            .ok_or_else(|| Error::Invalid("video call has no admitted media dimensions".into()))?;
        if let Some(session) = &request.mux {
            return Ok(Arc::clone(session));
        }
        let config = MuxConfig::new(
            request.canvas.width,
            request.canvas.height,
            frame_rate,
            audio_rate,
            Arc::clone(&request.units),
        )?;
        let session = Arc::new(MuxSession::new(config, functions));
        request.mux = Some(Arc::clone(&session));
        Ok(session)
    }
}

impl RequestStorage for VideoRequests {
    fn admit_slots(&mut self, _admitted: &[AdmittedSlot<'_>], _jobs: &LaneJobs) -> Result<()> {
        // A request's state is created when its admission is installed.
        Ok(())
    }

    fn release_buffers(&mut self, _buffers: &HashSet<BufferId>, _jobs: &LaneJobs) -> Result<()> {
        Ok(())
    }

    fn release_requests(
        &mut self,
        _requests: &HashSet<RequestKey>,
        _retained: &HashSet<BufferId>,
        _jobs: &LaneJobs,
    ) -> Result<()> {
        Ok(())
    }

    fn retirement_ready(
        &mut self,
        _buffers: &HashSet<BufferId>,
        _requests: &HashSet<RequestKey>,
        _retained: &HashSet<BufferId>,
    ) -> Result<bool> {
        Ok(true)
    }

    fn retire_slots(&mut self, slots: &[usize], _jobs: &LaneJobs) -> Result<()> {
        for slot in slots {
            self.requests.remove(slot);
        }
        Ok(())
    }

    fn reap(&mut self) -> Result<()> {
        Ok(())
    }

    fn close(&mut self, _aborted: bool) -> Result<()> {
        self.requests.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_spans_are_maximal_runs_of_vision_tags() {
        assert_eq!(
            vision_spans(&[1, 0, 0, 1, 1, 0, 0, 0]),
            vec![(1, 3), (5, 8)]
        );
        assert_eq!(vision_spans(&[0, 1]), vec![(0, 1)]);
        assert!(vision_spans(&[1, 1, 2]).is_empty());
        assert!(vision_spans(&[]).is_empty());
    }

    #[test]
    fn layouts_round_trip_their_descriptors() {
        let layout = VideoLayout::from_values(&[124, 1344, 768, 1024, 64]).unwrap();
        assert_eq!(
            layout.canvas,
            Canvas {
                width: 1344,
                height: 768
            }
        );
        assert_eq!(
            crate::ffi::integers(layout.descriptor()).unwrap(),
            vec![124, 1344, 768, 1024, 64]
        );
        assert!(VideoLayout::from_values(&[1, 2, 3]).is_err());
        assert!(VideoLayout::from_values(&[-1, 2, 3, 4, 5]).is_err());
    }

    fn canvas(width: u32, height: u32) -> Canvas {
        Canvas { width, height }
    }

    fn served(short_edge: u32, aspect_ratio: &str, width: u32, height: u32) -> ServedCanvas {
        ServedCanvas {
            short_edge,
            aspect_ratio: aspect_ratio.to_owned(),
            canvas: canvas(width, height),
        }
    }

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|&value| value.to_owned()).collect()
    }

    #[test]
    fn a_deployment_admits_the_offered_canvases_it_names() {
        let offered = [
            served(768, "21:9", 1536, 672),
            served(768, "16:9", 1344, 768),
            served(768, "9:16", 768, 1344),
            served(480, "16:9", 832, 480),
        ];
        // Without a selection every offered canvas is prepared.
        assert_eq!(admitted_canvases(&offered, &[], &[]).unwrap(), offered);
        // Resolutions cross aspect ratios resolution-major, in the order the
        // deployment names them.
        let admitted =
            admitted_canvases(&offered, &names(&["480p", "768p"]), &names(&["16:9"])).unwrap();
        assert_eq!(admitted, [offered[3].clone(), offered[1].clone()]);
        let admitted =
            admitted_canvases(&offered, &names(&["768p"]), &names(&["9:16", "16:9"])).unwrap();
        assert_eq!(admitted, [offered[2].clone(), offered[1].clone()]);
        // Without resolutions, every offered class takes their place.
        let admitted = admitted_canvases(&offered, &[], &names(&["16:9"])).unwrap();
        assert_eq!(admitted, [offered[1].clone(), offered[3].clone()]);
        // A pair the denoiser does not name refuses the deployment, naming
        // what it generates.
        let error = admitted_canvases(&offered, &names(&["480p"]), &names(&["21:9"])).unwrap_err();
        assert!(matches!(&error, Error::Unsupported(message)
            if message.contains("768p 21:9, 768p 16:9") && message.contains("not 480p 21:9")));
        // A name that is not a class, and a repeated name, are invalid.
        for (resolutions, ratios) in [
            (names(&["768"]), names(&[])),
            (names(&["768p", "768p"]), names(&[])),
            (names(&[]), names(&["16:9", "16:9"])),
        ] {
            assert!(matches!(
                admitted_canvases(&offered, &resolutions, &ratios),
                Err(Error::Invalid(_))
            ));
        }
        assert!(admitted_canvases(&[], &[], &[]).is_err());
    }

    #[test]
    fn the_canvas_taking_the_most_storage_is_prepared_first() {
        let extent = |canvas, storage, generated| CanvasExtent {
            canvas,
            storage,
            generated,
        };
        // 21:9 generates as many elements as 16:9 and takes more storage;
        // the 480p canvas and the portrait canvas follow by generated
        // elements, the equal ones in the order they were named.
        let wide = canvas(1536, 672);
        let landscape = canvas(1344, 768);
        let portrait = canvas(768, 1344);
        let small = canvas(832, 480);
        assert_eq!(
            preparation_order(&[
                extent(small, 40, 30),
                extent(landscape, 90, 100),
                extent(wide, 96, 100),
                extent(portrait, 90, 100),
            ]),
            vec![wide, landscape, portrait, small]
        );
        // The first canvas leads among equal storage.
        assert_eq!(
            preparation_order(&[extent(portrait, 90, 100), extent(landscape, 90, 100)]),
            vec![portrait, landscape]
        );
    }

    /// A plan of two canvases, two frame counts and two prompt capacities,
    /// whose startup layouts name what they were sized for.
    fn plan() -> MediaPlan {
        let canvases = vec![canvas(1344, 768), canvas(768, 1344)];
        let layout = |canvas, frames, text_rows| VideoLayout {
            frames,
            canvas,
            text_rows,
            condition_rows: 0,
        };
        let mut layouts = Vec::new();
        for &canvas in &canvases {
            for frames in [121, 49] {
                for capacity in [2048, 1024] {
                    layouts.push(layout(canvas, frames, capacity));
                }
            }
        }
        MediaPlan {
            frame_counts: vec![49, 121],
            admitted: vec![
                served(768, "16:9", 1344, 768),
                served(768, "9:16", 768, 1344),
            ],
            canvases: canvases.clone(),
            text_capacities: vec![1024, 2048],
            max_text_tokens: 2048,
            condition_rows: 512,
            maximum: layout(canvases[0], 121, 2048),
            layouts,
            largest: Vec::new(),
        }
    }

    /// The prompt capacity is the largest text capacity by construction: a
    /// list names the capacities below it, an entry equal to it repeats it,
    /// one above it is refused, and the default steps stop below it.
    #[test]
    fn the_prompt_capacity_is_the_largest_text_capacity() {
        assert_eq!(
            prompt_capacities(&[1024, 4096], 16384).unwrap(),
            vec![1024, 4096, 16384]
        );
        assert_eq!(
            prompt_capacities(&[1024, 10240, 16384], 16384).unwrap(),
            vec![1024, 10240, 16384]
        );
        assert_eq!(
            prompt_capacities(&[], 8192).unwrap(),
            vec![1024, 2048, 4096, 6144, 8192]
        );
        assert_eq!(prompt_capacities(&[], 1000).unwrap(), vec![1000]);
        for listed in [&[1024, 32768][..], &[0, 1024]] {
            assert!(matches!(
                prompt_capacities(listed, 16384),
                Err(Error::Invalid(_))
            ));
        }
    }

    #[test]
    fn a_size_without_conditions_takes_its_startup_layout() {
        let plan = plan();
        for &canvas in &plan.canvases {
            for &frames in &plan.frame_counts {
                for &capacity in &plan.text_capacities {
                    let layout = plan.startup_layout(frames, canvas, capacity).unwrap();
                    assert_eq!(
                        (layout.canvas, layout.frames, layout.text_rows),
                        (canvas, frames, capacity)
                    );
                }
            }
        }
        // Nothing outside the plan has a startup layout.
        assert!(plan.startup_layout(73, plan.canvases[0], 1024).is_none());
        assert!(plan.startup_layout(49, canvas(832, 480), 1024).is_none());
        assert!(plan.startup_layout(49, plan.canvases[0], 4096).is_none());
    }

    #[test]
    fn a_request_evaluates_in_the_smallest_capacity_that_holds_it() {
        let plan = plan();
        let size = VideoSize::plain(49, plan.canvases[0], 1500);
        let layouts: Vec<VideoLayout> = plan
            .text_capacities
            .iter()
            .map(|&capacity| plan.startup_layout(49, size.canvas, capacity).unwrap())
            .collect();

        let admitted = plan
            .select(size.clone(), 0, &layouts, &[false, true])
            .unwrap();
        assert_eq!(
            (admitted.text_capacity, admitted.layout),
            (2048, layouts[1])
        );
        let admitted = plan
            .select(size.clone(), 0, &layouts, &[true, true])
            .unwrap();
        assert_eq!(
            (admitted.text_capacity, admitted.layout),
            (1024, layouts[0])
        );

        // No layout holds the request, or its conditions exceed the plan's:
        // either is the request's refusal, the second naming the capacity
        // the deployment states.
        assert!(
            plan.select(size.clone(), 0, &layouts, &[false, false])
                .is_err()
        );
        let refusal = plan.select(size, 513, &layouts, &[true, true]).unwrap_err();
        assert!(refusal.contains("513") && refusal.contains("video_config.max_condition_rows"));
    }
}
