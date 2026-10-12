//! Video request sizes, denoiser layouts and admission planning.
//!
//! A video request's numerical size is its output frame count, canvas,
//! prompt length, conditions (in request order) and the vision spans of its
//! presented prompt. The denoiser evaluates it in a *layout*: a size whose
//! shapes, constants and captured graphs every request it holds shares
//! (`VideoDenoiser.layout_size`/`holds`). [`MediaPlan`] fixes, at startup,
//! the frame counts, canvases and prompt capacities a rank admits and the
//! layouts it prepares; [`MediaPlan::admit`] resolves a request's exact size
//! and the smallest prepared prompt capacity whose layout holds it.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_core::{Canvas, ConditionRole, VideoCondition};
use uniserve_ipc::{BufferId, NewRequest, RequestKey};

use crate::executor::lifecycle::{LaneJobs, RequestStorage};
use crate::ffi::integer_list;
use crate::host::mux::{MuxConfig, MuxSession};
use crate::runner::denoiser::VideoModel;
use crate::runner::denoiser::runners::DenoiserRunners;
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
        Any::from(integer_list([
            i64::from(self.frames),
            i64::from(self.canvas.width),
            i64::from(self.canvas.height),
            i64::from(self.text_rows),
            i64::from(self.condition_rows),
        ]))
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

/// What a rank admits and the layouts it prepares, fixed at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MediaPlan {
    /// Admitted frame counts, ascending.
    pub(crate) frame_counts: Vec<u32>,
    /// Admitted canvases in the order the deployment names them, which the
    /// handshake reports.
    pub(crate) admitted: Vec<Canvas>,
    /// The admitted canvases in preparation order (see
    /// [`preparation_order`]).
    pub(crate) canvases: Vec<Canvas>,
    /// Prompt capacities of the prepared layouts, ascending.
    pub(crate) text_capacities: Vec<u32>,
    /// The longest admitted prompt.
    pub(crate) max_text_tokens: u32,
    /// Condition rows a request may occupy; zero without conditioned tasks.
    pub(crate) condition_rows: u32,
    /// The leading canvas's longest layout at the prompt capacity, widened
    /// to the condition capacity. The denoiser prepares it first, so it
    /// sizes the storage every layout shares.
    pub(crate) maximum: VideoLayout,
    /// Startup layouts: canvases (in preparation order) by frame counts
    /// (longest first) by prompt capacities (largest first).
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

/// The canvases a deployment prepares and admits: `configured`, or every
/// canvas the denoiser offers without a selection.
///
/// Fails, as an unsupported setup, when the denoiser offers no canvas or
/// does not offer a configured one.
fn admitted_canvases(offered: &[Canvas], configured: &[Canvas]) -> Result<Vec<Canvas>> {
    if offered.is_empty() {
        return Err(Error::Unsupported(
            "the video denoiser prepares no canvas".into(),
        ));
    }
    if configured.is_empty() {
        return Ok(offered.to_vec());
    }
    if configured.iter().any(|canvas| !offered.contains(canvas)) {
        let names = |canvases: &[Canvas]| {
            canvases
                .iter()
                .map(|canvas| format!("{}x{}", canvas.width, canvas.height))
                .collect::<Vec<_>>()
                .join(", ")
        };
        return Err(Error::Unsupported(format!(
            "the denoiser offers only the canvases {}, not every canvas of {}",
            names(offered),
            names(configured)
        )));
    }
    Ok(configured.to_vec())
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

/// The seconds-to-frames rule the HTTP planner shares: round half to even.
fn frames_of(seconds: f64, rate: u32) -> u32 {
    (seconds * f64::from(rate)).round_ties_even() as u32
}

impl MediaPlan {
    /// Plan a deployment's admitted sizes on the loaded denoiser.
    ///
    /// `requested` are the configured prompt capacities (default
    /// `[1024, 2048, 4096, ...]` up to the prompt capacity), each mapped to
    /// the text rows of its layout, capped at the largest layout's.
    /// `canvases` are the canvases the deployment names, each one the
    /// denoiser offers; without any the deployment prepares every offered
    /// canvas.
    pub(crate) fn new(
        model: &VideoModel,
        max_seconds: f64,
        min_seconds: Option<f64>,
        max_text_tokens: u32,
        requested: &[u32],
        condition_rows: u32,
        canvases: &[Canvas],
    ) -> Result<Self> {
        let facts = &model.facts;
        let longest = model.legal_frame_count(frames_of(max_seconds, facts.frame_rate))?;
        let mut count = model.legal_frame_count(
            min_seconds
                .map_or(1, |seconds| frames_of(seconds, facts.frame_rate))
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
        let admitted = admitted_canvases(&facts.canvases, canvases)?;
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
        for &canvas in &admitted {
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

        let requested: Vec<u32> = if requested.is_empty() {
            std::iter::once(1024)
                .chain((2048..max_text_tokens).step_by(2048))
                .chain(std::iter::once(max_text_tokens))
                .collect()
        } else {
            requested.to_vec()
        };
        if requested.contains(&0) || requested.iter().copied().max().unwrap_or(0) < max_text_tokens
        {
            return Err(Error::Invalid(
                "text capacities must be positive and hold the prompt capacity".into(),
            ));
        }
        let mut text_capacities = requested
            .iter()
            .map(|&tokens| {
                let layout = model.layout_size(&VideoSize::plain(longest, reference, tokens))?;
                Ok(layout.text_rows.min(largest_text))
            })
            .collect::<Result<Vec<_>>>()?;
        text_capacities.sort_unstable();
        text_capacities.dedup();

        let mut layouts = Vec::new();
        for &canvas in &ordered {
            for &frames in frame_counts.iter().rev() {
                for &tokens in text_capacities.iter().rev() {
                    layouts.push(model.layout_size(&VideoSize::plain(frames, canvas, tokens))?);
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

    /// Resolve a request's exact size and the layout it evaluates in: the
    /// smallest prompt capacity whose layout holds it.
    ///
    /// Fails, as an invalid descriptor, when the frame count or canvas is not
    /// admitted, the prompt or conditions exceed their capacities, the model
    /// rejects the size, or no prepared capacity holds it.
    pub(crate) fn admit(&self, model: &VideoModel, size: VideoSize) -> Result<Admission> {
        if !self.frame_counts.contains(&size.frames) || !self.canvases.contains(&size.canvas) {
            return Err(Error::Invalid(
                "media input exceeds the worker frame, canvas, conditioning or condition capacity"
                    .into(),
            ));
        }
        let condition_rows = model.request_size(&size)?;
        if size.tokens > self.max_text_tokens || condition_rows > self.condition_rows {
            return Err(Error::Invalid(
                "media input exceeds the worker frame, canvas, conditioning or condition capacity"
                    .into(),
            ));
        }
        for &capacity in &self.text_capacities {
            let candidate = VideoSize {
                tokens: capacity,
                spans: Arc::from([]),
                ..size.clone()
            };
            let layout = model.layout_size(&candidate)?;
            if model.holds(&layout, &size)? {
                return Ok(Admission {
                    size,
                    condition_rows,
                    layout,
                    text_capacity: capacity,
                });
            }
        }
        Err(Error::Invalid(
            "media input has no admitted capacity layout".into(),
        ))
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
        })
    }

    /// Admit a started video request: resolve its size and, on a denoising
    /// rank, the layout `denoisers` holds for it.
    pub(crate) fn admit(
        &mut self,
        admission: &NewRequest,
        denoisers: &mut DenoiserRunners,
    ) -> Result<()> {
        let (Some(media), Some(video)) = (&admission.diffusion, &admission.video) else {
            return Ok(());
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
        let duration = self
            .durations
            .get(&media.num_frames)
            .filter(|_| self.plan.canvases.contains(&canvas))
            .ok_or_else(|| {
                Error::Invalid(
                    "media input exceeds the worker frame, canvas, conditioning or condition capacity"
                        .into(),
                )
            })?;
        let units = Arc::clone(&duration.units);
        if media.video_units as usize != units.len()
            || media.num_inference_steps != self.model.facts.num_steps
        {
            return Err(Error::Invalid(
                "video request has invalid computation bounds".into(),
            ));
        }
        let conditions: Arc<[VideoCondition]> = video.conditions.clone().into();
        let size = VideoSize {
            frames: media.num_frames,
            canvas,
            tokens: admission.prompt_token_ids.len() as u32,
            conditions: condition_shapes(&conditions),
            spans: vision_spans(&video.text_tags).into(),
        };
        let mut request = VideoRequest {
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
        if denoisers.video.is_some() {
            let admitted = self.plan.admit(&self.model, size)?;
            request.layout = denoisers.admit(slot, &self.model, admitted.layout)?;
            request.admission = Some(admitted);
        }
        self.requests.insert(slot, request);
        Ok(())
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
    fn admit_slots(&mut self, _slots: &[usize], _jobs: &LaneJobs) -> Result<()> {
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

    #[test]
    fn a_deployment_admits_the_offered_canvases_it_names() {
        let offered = [canvas(1536, 672), canvas(1344, 768), canvas(768, 1344)];
        // Without a selection every offered canvas is prepared.
        assert_eq!(admitted_canvases(&offered, &[]).unwrap(), offered);
        // A selection keeps the order the deployment names.
        let named = [canvas(768, 1344), canvas(1344, 768)];
        assert_eq!(admitted_canvases(&offered, &named).unwrap(), named);
        // A canvas the denoiser does not offer refuses the deployment,
        // naming what it offers.
        let error = admitted_canvases(&offered, &[canvas(832, 480)]).unwrap_err();
        assert!(matches!(&error, Error::Unsupported(message)
            if message.contains("1536x672, 1344x768, 768x1344") && message.contains("832x480")));
        assert!(admitted_canvases(&[], &[]).is_err());
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

    #[test]
    fn seconds_round_half_to_even_frames() {
        assert_eq!(frames_of(5.0, 24), 120);
        assert_eq!(frames_of(0.0625, 24), 2);
        assert_eq!(frames_of(0.020_833_333_333_333_33 * 2.5, 24), 1);
    }
}
