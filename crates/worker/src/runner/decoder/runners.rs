//! The decoder runners of a rank.

use std::collections::HashSet;

use super::{DecodeImage, DecoderRunner, Segment, TrackUnits, UnitShape};
use crate::Result;
use crate::profiling::kernels::KernelRecord;
use crate::request::video::{MediaPlan, VideoSize};
use crate::runner::component::Component;
use crate::runner::denoiser::VideoModel;
use crate::runner::denoiser::image_model::ImageDenoiserLayout;
use crate::storage::ImageLatents;

/// The decoder runners of a rank.
#[derive(Default)]
pub(crate) struct DecoderRunners {
    /// A video model's video decoder.
    pub(crate) video: Option<Component<DecoderRunner>>,
    /// The unit shapes startup warmed, which it then captures: each with
    /// its window's latent rows and its video's longest unit.
    unit_shapes: Vec<(UnitShape, u64, u32)>,
    /// A video model's audio decoder.
    pub(crate) audio: Option<Component<DecoderRunner>>,
    /// The track units startup warmed, which it then captures, each with
    /// its track's latent rows.
    track_units: Vec<(TrackUnits, u64)>,
    /// An image model's decoder, with its component.
    pub(crate) image: Option<(String, DecoderRunner)>,
    /// Output words of an image decoding batch's pixels at most.
    pub(crate) max_pixel_words: usize,
}

impl DecoderRunners {
    /// The decoder of an image model and the most output words a decoding
    /// batch's pixels take.
    pub(crate) fn image(decoder: Option<(String, DecoderRunner)>, max_pixel_words: usize) -> Self {
        Self {
            image: decoder,
            max_pixel_words,
            ..Self::default()
        }
    }

    /// Warm the image decoder on one black `side` x `side` image whose
    /// latent rows are the head of the step workspace of `latents`.
    pub(crate) fn warm_image(
        &self,
        latents: Option<&ImageLatents>,
        layout: Option<ImageDenoiserLayout>,
        side: u32,
    ) -> Result<()> {
        let (Some((_, decoder)), Some(latents), Some(layout)) = (&self.image, latents, layout)
        else {
            return Ok(());
        };
        let units = layout.latent_rows(side, side)?;
        let workspace = latents.workspace()?;
        workspace
            .narrow(0, 0, i64::from(units))?
            .zero(decoder.stream())?;
        decoder.decode(
            &[DecodeImage {
                sample: 0,
                units,
                height: side,
                width: side,
                product: None,
            }],
            |_, _| Ok(()),
        )?;
        if let Some(stream) = decoder.stream() {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Prepare and warm the video decoder's segments and every round shape
    /// of its unit ring, then the audio decoder's durations. Runs on every
    /// rank of the group in the same order, because rounds exchange
    /// overlaps.
    pub(crate) fn prepare_startup(
        &mut self,
        model: &VideoModel,
        plan: &MediaPlan,
        rank: usize,
    ) -> Result<()> {
        if let Some(decoder) = &mut self.video {
            decoder.runner.prepare_segments(&segments(model, plan)?)?;
            let width = (decoder.config.ranks.len() * decoder.config.units_per_rank.max(1)) as u32;
            // One round of each shape warms the ring and is captured: a
            // round's shape is its canvas, whether it holds the video's
            // first and last units, and how many members hold a unit. Every
            // member derives the same rounds in the same order.
            let mut warmed = HashSet::new();
            for &canvas in &plan.canvases {
                for &frames in plan.frame_counts.iter().rev() {
                    let units = model.unit_frames(frames)?;
                    let total = units.len() as u32;
                    let longest = units.iter().copied().max().unwrap_or(0);
                    let mut cursor = 0;
                    while cursor < total {
                        let count = width.min(total - cursor);
                        let round = (canvas, cursor == 0, cursor + count == total, count);
                        let run = decoder.units(rank, cursor, count)?;
                        if warmed.insert(round) && !run.is_empty() {
                            let unit = run.start;
                            let shape = UnitShape {
                                segment: model.segment(frames, canvas, unit)?,
                                first: unit == 0,
                                last: unit + 1 == total,
                                count,
                            };
                            let rows = model.window_rows(frames, canvas, unit)?;
                            decoder.runner.warm_unit(shape, rows, longest)?;
                            self.unit_shapes.push((shape, rows, longest));
                        }
                        cursor += count;
                    }
                }
            }
        }
        if let Some(audio) = &mut self.audio {
            let total = if audio.config.distribution.is_some() {
                (audio.config.ranks.len() * audio.config.units_per_rank.max(1)) as u32
            } else {
                1
            };
            let canvas = plan.canvases[0];
            for &frames in plan.frame_counts.iter().rev() {
                let samples = model.track_samples(frames)?;
                audio
                    .runner
                    .prepare_track(model.audio_latent_frames(samples)?)?;
            }
            for &frames in plan.frame_counts.iter().rev() {
                let samples = model.track_samples(frames)?;
                let layout =
                    model.layout_size(&VideoSize::plain(frames, canvas, plan.max_text_tokens))?;
                let rows = model.latent_shape("audio", &layout)?[0];
                let run = audio.units(rank, 0, total)?;
                if !run.is_empty() {
                    let spans = model.audio_units(samples, total)?;
                    let units = TrackUnits {
                        samples,
                        latent_frames: model.audio_latent_frames(samples)?,
                        first: run.start,
                        count: run.len() as u32,
                        total,
                        span: spans[run.end as usize - 1].end - spans[run.start as usize].start,
                    };
                    audio.runner.warm_track(units, rows)?;
                    self.track_units.push((units, rows));
                }
            }
        }
        Ok(())
    }

    /// Capture the warmed unit shapes and track units, then refuse further
    /// captures.
    pub(crate) fn capture_startup(&mut self) -> Result<()> {
        if let Some(decoder) = &mut self.video {
            for (shape, rows, longest) in std::mem::take(&mut self.unit_shapes) {
                decoder.runner.capture_unit(shape, rows, longest)?;
            }
            if let Some(stream) = decoder.runner.stream() {
                stream.synchronize()?;
            }
            decoder.runner.seal();
        }
        if let Some(audio) = &mut self.audio {
            for (units, rows) in std::mem::take(&mut self.track_units) {
                audio.runner.capture_track(units, rows)?;
            }
            if let Some(stream) = audio.runner.stream() {
                stream.synchronize()?;
            }
            audio.runner.seal();
        }
        Ok(())
    }

    /// The kernels the decoders' operators selected.
    pub(crate) fn kernels(&self) -> Result<Vec<(String, Vec<KernelRecord>)>> {
        let mut kernels = Vec::new();
        if let Some(decoder) = &self.video {
            kernels.push((decoder.name.clone(), decoder.runner.kernels(&decoder.name)?));
        }
        if let Some(audio) = &self.audio {
            kernels.push((audio.name.clone(), audio.runner.kernels(&audio.name)?));
        }
        Ok(kernels)
    }

    /// Release the video decoder, then the audio decoder.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let mut result = Ok(());
        if let Some(decoder) = &mut self.video {
            result = result.and(decoder.runner.close(aborted));
        }
        if let Some(audio) = &mut self.audio {
            result = result.and(audio.runner.close(aborted));
        }
        result
    }
}

/// Every segment admitted videos decode to: canvases (in preparation order) by
/// frame counts (longest first) by units, without repeats.
pub(crate) fn segments(model: &VideoModel, plan: &MediaPlan) -> Result<Vec<Segment>> {
    let mut result = Vec::new();
    for &canvas in &plan.canvases {
        for &frames in plan.frame_counts.iter().rev() {
            for unit in 0..model.unit_frames(frames)?.len() as u32 {
                let segment = model.segment(frames, canvas, unit)?;
                if !result.contains(&segment) {
                    result.push(segment);
                }
            }
        }
    }
    Ok(result)
}
