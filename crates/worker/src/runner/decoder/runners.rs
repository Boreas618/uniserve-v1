//! The decoder runners of a rank.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use ipc::{CallKind, MediaCall};
use models::video::Canvas;

use super::{ClipOutput, DecodeImage, DecoderRunner, TrackUnits, UnitShape};
use crate::lane::{BLOCK_ON_TIMEOUT, Completion, Job, JobQueues};
use crate::memory::LatentCache;
use crate::profiling::kernels::KernelRecord;
use crate::request::video::{MediaPlan, VideoSize};
use crate::runner::component::Component;
use crate::runner::denoiser::VideoModel;
use crate::runner::denoiser::image_model::ImageDenoiserLayout;
use crate::stream::Issue;
use crate::{Error, Result};

/// The lane kinds of the video, audio and image decoders' jobs.
const VIDEO: CallKind = CallKind::Media(MediaCall::VideoDecoding);
const AUDIO: CallKind = CallKind::Media(MediaCall::AudioDecoding);
const IMAGE: CallKind = CallKind::Media(MediaCall::ImageDecoding);

/// Perform a startup `job` on the lane serving `kind` and wait for its
/// completion.
fn perform(lanes: &JobQueues, kind: CallKind, job: Job) -> Result<Completion> {
    lanes
        .block_on(kind, job, BLOCK_ON_TIMEOUT)?
        .check()
        .map_err(|failure| failure.error)
}

/// What the model answers about an admitted audio track decoded in some
/// number of units.
struct AdmittedTrack {
    latent_frames: u32,
    /// The samples of each audio unit.
    spans: Vec<Range<u64>>,
}

/// The decoder runners of a rank.
#[derive(Default)]
pub(crate) struct DecoderRunners {
    /// A video model's video decoder.
    pub(crate) video: Option<Component<DecoderRunner>>,
    /// The unit shapes startup warmed, which it then captures: each with
    /// its clip input's latent rows and its video's longest unit.
    unit_shapes: Vec<(UnitShape, u64, u32)>,
    /// The clip output each media unit of every admitted video decodes to, by
    /// `(frames, canvas, unit)`, read from the model at startup.
    clip_outputs: HashMap<(u32, Canvas, u32), ClipOutput>,
    /// A video model's audio decoder.
    pub(crate) audio: Option<Component<DecoderRunner>>,
    /// Every admitted audio track decoded in some number of units, by
    /// `(samples, units)`, read from the model at startup.
    tracks: HashMap<(u64, u32), AdmittedTrack>,
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
    /// latent rows are the head of the step workspace of `latents`: one job
    /// of the lane on `lanes` serving image decoding zeroes the rows,
    /// decodes the image as serving records the decode and waits for the
    /// stream.
    pub(crate) fn warm_image(
        &self,
        latents: Option<&LatentCache>,
        layout: Option<ImageDenoiserLayout>,
        side: u32,
        lanes: &JobQueues,
    ) -> Result<()> {
        let (Some((_, decoder)), Some(latents), Some(layout)) = (&self.image, latents, layout)
        else {
            return Ok(());
        };
        let units = layout.latent_rows(side, side)?;
        let workspace = latents.workspace()?;
        let job = Job::new();
        let issue = job.on(decoder.stream());
        // SAFETY: the latent cache keeps its workspace past this warmup,
        // whose jobs are performed before it returns; nothing else uses the
        // workspace before serving starts.
        unsafe { workspace.narrow(0, 0, i64::from(units))?.zero(issue) }?;
        let mut returns = Vec::new();
        // The staged pixels are discarded.
        decoder.record_decode(
            &[DecodeImage {
                sample: 0,
                units,
                height: side,
                width: side,
                product: None,
            }],
            |_, _| Ok(()),
            &job,
            &mut returns,
        )?;
        issue.synchronize()?;
        let mut completion = perform(lanes, IMAGE, job)?;
        decoder.check_staged(returns, &mut completion)
    }

    /// Prepare and warm the video decoder's clip outputs and every round shape
    /// of its unit ring, then the audio decoder's durations. Runs on every
    /// rank of the group in the same order, because rounds exchange
    /// overlaps. Each warmup is a job of the decoder's lane on `lanes`,
    /// which startup waits for.
    pub(crate) fn prepare_startup(
        &mut self,
        model: &VideoModel,
        plan: &MediaPlan,
        rank: usize,
        lanes: &JobQueues,
    ) -> Result<()> {
        if let Some(decoder) = &mut self.video {
            decoder
                .runner
                .prepare_clip_outputs(&clip_outputs(model, plan)?)?;
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

                    // What a decode of each unit takes from the model is
                    // read now: a serving rank asks the library nothing.
                    for unit in 0..total {
                        let clip_output = model.clip_output(frames, canvas, unit)?;
                        self.clip_outputs
                            .insert((frames, canvas, unit), clip_output);
                        decoder.runner.read_clip_rows(frames, canvas, unit)?;
                    }
                    let mut cursor = 0;
                    while cursor < total {
                        let count = width.min(total - cursor);
                        let round = (canvas, cursor == 0, cursor + count == total, count);
                        let run = decoder.units(&[], rank, cursor, count)?;
                        if warmed.insert(round) && !run.is_empty() {
                            let unit = run.start;
                            let shape = UnitShape {
                                clip_output: model.clip_output(frames, canvas, unit)?,
                                first: unit == 0,
                                last: unit + 1 == total,
                                count,
                            };
                            let rows = model.clip_rows(frames, canvas, unit)?;
                            let job = Job::new();
                            decoder.runner.warm_unit(shape, rows, longest, &job)?;
                            perform(lanes, VIDEO, job)?;
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
                let latent_frames = model.audio_latent_frames(samples)?;
                audio.runner.prepare_track(latent_frames)?;

                // A round decodes the track in up to `total` units.
                for units in 1..=total {
                    let spans = model.audio_units(samples, units)?;
                    let track = AdmittedTrack {
                        latent_frames,
                        spans,
                    };
                    self.tracks.insert((samples, units), track);
                }
            }
            for &frames in plan.frame_counts.iter().rev() {
                let samples = model.track_samples(frames)?;
                let layout =
                    model.layout_size(&VideoSize::plain(frames, canvas, plan.max_text_tokens))?;
                let rows = model.latent_shape("audio", &layout)?[0];
                let run = audio.units(&[], rank, 0, total)?;
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
                    let job = Job::new();
                    audio.runner.warm_track(units, rows, &job)?;
                    perform(lanes, AUDIO, job)?;
                    self.track_units.push((units, rows));
                }
            }
        }
        Ok(())
    }

    /// The `(frames, canvas)` clip output unit `unit` of an admitted
    /// `frames`-frame video at `canvas` decodes to.
    pub(crate) fn clip_output(&self, frames: u32, canvas: Canvas, unit: u32) -> Result<ClipOutput> {
        self.clip_outputs
            .get(&(frames, canvas, unit))
            .copied()
            .ok_or_else(|| Error::Invalid("the rank admits no video with this media unit".into()))
    }

    /// The latent frames of an admitted `samples`-sample track and the
    /// samples of each of its `units` audio units.
    pub(crate) fn track(&self, samples: u64, units: u32) -> Result<(u32, &[Range<u64>])> {
        self.tracks
            .get(&(samples, units))
            .map(|track| (track.latent_frames, track.spans.as_slice()))
            .ok_or_else(|| Error::Invalid("the rank admits no track of this length".into()))
    }

    /// Capture the warmed unit shapes and track units, then refuse further
    /// captures. Each capture is a job of the decoder's lane on `lanes`,
    /// whose completion carries the graph; a last job waits for the stream
    /// before the decoder is sealed.
    pub(crate) fn capture_startup(&mut self, lanes: &JobQueues) -> Result<()> {
        if let Some(decoder) = &mut self.video {
            for (shape, rows, longest) in std::mem::take(&mut self.unit_shapes) {
                let job = Job::new();
                if decoder.runner.capture_unit(shape, rows, longest, &job)? {
                    let mut completion = perform(lanes, VIDEO, job)?;
                    decoder.runner.complete_unit(shape, &mut completion)?;
                }
            }
            let job = Job::new();
            job.on(decoder.runner.stream()).synchronize()?;
            perform(lanes, VIDEO, job)?;
            decoder.runner.seal();
        }
        if let Some(audio) = &mut self.audio {
            for (units, rows) in std::mem::take(&mut self.track_units) {
                let job = Job::new();
                if audio.runner.capture_track(units, rows, &job)? {
                    let mut completion = perform(lanes, AUDIO, job)?;
                    audio.runner.complete_track(units, &mut completion)?;
                }
            }
            let job = Job::new();
            job.on(audio.runner.stream()).synchronize()?;
            perform(lanes, AUDIO, job)?;
            audio.runner.seal();
        }
        Ok(())
    }

    /// The kernels the decoders' operators selected.
    pub(crate) fn kernels(&self) -> Result<Vec<(String, Vec<KernelRecord>)>> {
        let mut kernels = Vec::new();
        if let Some(decoder) = &self.video {
            kernels.push((decoder.name.clone(), decoder.runner.kernels()?));
        }
        if let Some(audio) = &self.audio {
            kernels.push((audio.name.clone(), audio.runner.kernels()?));
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

/// Every clip output admitted videos decode to: canvases (in preparation order) by
/// frame counts (longest first) by units, without repeats.
pub(crate) fn clip_outputs(model: &VideoModel, plan: &MediaPlan) -> Result<Vec<ClipOutput>> {
    let mut result = Vec::new();
    for &canvas in &plan.canvases {
        for &frames in plan.frame_counts.iter().rev() {
            for unit in 0..model.unit_frames(frames)?.len() as u32 {
                let clip_output = model.clip_output(frames, canvas, unit)?;
                if !result.contains(&clip_output) {
                    result.push(clip_output);
                }
            }
        }
    }
    Ok(result)
}
