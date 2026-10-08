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

use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_core::{Canvas, ConditionRole, VideoCondition};

use crate::ffi::{integer_list, list};
use crate::model::video::VideoModel;
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

    fn descriptor(&self) -> Any {
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
        Any::from(integer_list([
            role,
            i64::from(frames),
            i64::from(canvas.width),
            i64::from(canvas.height),
            self.audio_samples as i64,
        ]))
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
        Any::from(integer_list([
            i64::from(self.frames),
            i64::from(self.canvas.width),
            i64::from(self.canvas.height),
            i64::from(self.tokens),
        ]))
    }

    /// The conditions descriptor, in request order.
    pub(crate) fn conditions_descriptor(&self) -> Any {
        Any::from(list(self.conditions.iter().map(ConditionShape::descriptor)))
    }

    /// The flat vision spans descriptor.
    pub(crate) fn spans_descriptor(&self) -> Any {
        Any::from(integer_list(
            self.spans
                .iter()
                .flat_map(|&(start, stop)| [i64::from(start), i64::from(stop)]),
        ))
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
    /// Admitted canvases, largest generated sample first.
    pub(crate) canvases: Vec<Canvas>,
    /// Prompt capacities of the prepared layouts, ascending.
    pub(crate) text_capacities: Vec<u32>,
    /// The longest admitted prompt.
    pub(crate) max_text_tokens: u32,
    /// Condition rows a request may occupy; zero without conditioned tasks.
    pub(crate) condition_rows: u32,
    /// The largest layout, widened to the condition capacity.
    pub(crate) maximum: VideoLayout,
    /// Startup layouts: canvases (largest first) by frame counts (longest
    /// first) by prompt capacities (largest first).
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
    pub(crate) fn new(
        model: &VideoModel,
        max_seconds: f64,
        min_seconds: Option<f64>,
        max_text_tokens: u32,
        requested: &[u32],
        condition_rows: u32,
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
        if facts.canvases.is_empty() {
            return Err(Error::Unsupported(
                "the video denoiser prepares no canvas".into(),
            ));
        }

        // Global sample extents bound every shard, even when a shorter prompt
        // moves more media rows onto one sequence-parallel rank.
        let mut ordered = Vec::new();
        for &canvas in &facts.canvases {
            let size = VideoSize::plain(longest, canvas, max_text_tokens);
            let layout = model.layout_size(&size)?;
            let elements = model.latent_elements("video", &layout)?;
            ordered.push((elements, canvas, layout));
        }
        // A stable sort keeps the model's order among equal extents.
        ordered.sort_by_key(|(elements, _, _)| std::cmp::Reverse(*elements));
        let reference = ordered[0].1;
        let largest_text = ordered[0].2.text_rows;

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
        for (_, canvas, _) in &ordered {
            for &frames in frame_counts.iter().rev() {
                for &tokens in text_capacities.iter().rev() {
                    layouts.push(model.layout_size(&VideoSize::plain(frames, *canvas, tokens))?);
                }
            }
        }
        let widen = |layout: VideoLayout| {
            if condition_rows == 0 {
                Ok(layout)
            } else {
                model.condition_layout(&layout, condition_rows)
            }
        };
        let largest = facts
            .canvases
            .iter()
            .map(|&canvas| {
                widen(model.layout_size(&VideoSize::plain(longest, canvas, max_text_tokens))?)
            })
            .collect::<Result<Vec<_>>>()?;
        let maximum = widen(ordered[0].2)?;
        Ok(Self {
            frame_counts,
            canvases: ordered.iter().map(|(_, canvas, _)| *canvas).collect(),
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

    #[test]
    fn seconds_round_half_to_even_frames() {
        assert_eq!(frames_of(5.0, 24), 120);
        assert_eq!(frames_of(0.0625, 24), 2);
        assert_eq!(frames_of(0.020_833_333_333_333_33 * 2.5, 24), 1);
    }
}
