//! Decoders: completed latents to delivered pixels and samples.
//!
//! A [`DecoderRunner`] is the Rust half of the Python
//! `uniserve.runner.decoder.DecoderRunner` for one decoder of a
//! model: an image, a video or an audio decoder. The three capabilities
//! share the type and differ in the tensors and graphs of their forwards.
//!
//! **Video and audio.** A video decoding round hands each rank of the
//! decoder's unit ring one media unit. The runner launches the Python
//! half's `decode` for it:
//! the forward unpacks the unit's clip input from the request's packed latent,
//! decodes it at its clip output and cross-fades and quantizes it into the
//! unit's RGB frames, exchanging decoded overlaps with its ring neighbours.
//! It is captured into one graph per unit shape and launched natively. The
//! request's retained overlap lives in its slot's row of the overlap bank.
//! An audio decoder decodes a contiguous run of audio units into their
//! sample span. A Python runner serves one module at one size, so the
//! runner holds one per size it serves: a video decoder per clip output, an
//! audio decoder per track length.
//!
//! **Images.**
//! An image decoder of a bound component decodes images
//! whose latent rows the caller gathered into the latent pool's step
//! workspace, optionally publishes the decoded values as image products,
//! and quantizes every image to packed 8-bit HWC pixels in its staging
//! bytes. The caller copies each
//! image's staged pixels to host memory on the runner's stream and encodes
//! them there ([`crate::host::image::png_base64`]).
//!
//! Staging holds the pixels of as many of the largest images as fit its
//! bytes; a batch with more images decodes in consecutive sub-batches.
//! Images of one size decode together within a sub-batch.

mod audio;
mod image;
pub(crate) mod runners;
mod start;
mod video;
pub(crate) mod warmup;

use std::sync::Arc;

use cuda::{CopyBatch, SourceOrder, Stream};
use models::video::Canvas;
use tvm_ffi::Any;

use crate::config::Device;
use crate::cuda_graph::{GraphMode, GraphPool};
use crate::ffi::DType;
use crate::lane::{Completion, Job};
use crate::memory::allocation::{Storage, StorageView};
use crate::profiling::kernels::KernelRecord;
use crate::runner::methods::{MethodCall, kernels_call, list_kernels};
use crate::runner::{ImageSetup, RunnerSetup};
use crate::stream::Issue;
use crate::{Error, Result};

use audio::AudioDecoder;
pub(crate) use audio::{TrackUnits, decode_audio};
use image::ImageDecoder;
pub(crate) use image::image_placement;
use video::VideoDecoder;
pub(crate) use video::{ClipOutput, DecodeCapacity, UnitShape, decode_video};

const DECODER: &str = "uniserve.runner.decoder.DecoderRunner";

/// One image to decode.
#[derive(Clone, Debug)]
pub struct DecodeImage {
    /// The image's first latent row in the step workspace and its rows.
    pub sample: u32,
    pub units: u32,
    pub height: u32,
    pub width: u32,
    /// The image product receiving the decoded values, if published.
    pub product: Option<StorageView>,
}

impl DecodeImage {
    /// Bytes of the image's quantized pixels.
    pub(crate) fn pixel_bytes(&self) -> usize {
        self.height as usize * self.width as usize * 3
    }
}

/// The decoder capability a runner serves, with its state.
enum Family {
    Image(Box<ImageDecoder>),
    Video(Box<VideoDecoder>),
    Audio(Box<AudioDecoder>),
}

/// The runner of one decoder of a model.
pub struct DecoderRunner {
    family: Family,
}

impl DecoderRunner {
    /// The runner of an image model's decoder over the latent pool's step
    /// workspace `latents` and `staging_bytes` of staging.
    pub fn image(
        setup: &ImageSetup<'_>,
        range: (f32, f32),
        latents: StorageView,
        staging_bytes: usize,
    ) -> Result<Self> {
        let image = ImageDecoder::new(setup, range, latents, staging_bytes)?;
        Ok(Self {
            family: Family::Image(Box::new(image)),
        })
    }

    /// The runner of a video decoder and its RGB post-processor.
    pub(crate) fn video(
        setup: RunnerSetup<'_>,
        capacity: DecodeCapacity,
        slots: u32,
        pool: Option<Arc<GraphPool>>,
    ) -> Result<Self> {
        Ok(Self {
            family: Family::Video(Box::new(VideoDecoder::new(setup, capacity, slots, pool)?)),
        })
    }

    /// The runner of an audio decoder. `latent` is the `[rows, width]` of
    /// the longest track's complete latent and `samples` that track's
    /// samples.
    pub(crate) fn audio(
        setup: RunnerSetup<'_>,
        latent: [u64; 2],
        samples: u64,
        pool: Option<Arc<GraphPool>>,
    ) -> Result<Self> {
        Ok(Self {
            family: Family::Audio(Box::new(AudioDecoder::new(setup, latent, samples, pool)?)),
        })
    }

    fn image_decoder(&self) -> Result<&ImageDecoder> {
        match &self.family {
            Family::Image(image) => Ok(image),
            _ => Err(Error::Invariant("the decoder decodes no images".into())),
        }
    }

    fn video_decoder(&mut self) -> Result<&mut VideoDecoder> {
        match &mut self.family {
            Family::Video(video) => Ok(video),
            _ => Err(Error::Invariant("the decoder decodes no video".into())),
        }
    }

    fn audio_decoder(&mut self) -> Result<&mut AudioDecoder> {
        match &mut self.family {
            Family::Audio(audio) => Ok(audio),
            _ => Err(Error::Invariant("the decoder decodes no audio".into())),
        }
    }

    pub(crate) fn device(&self) -> Device {
        match &self.family {
            Family::Image(image) => image.device,
            Family::Video(video) => Device::Cuda(video.device),
            Family::Audio(audio) => Device::Cuda(audio.device),
        }
    }

    /// The runner's stream; `None` on the host.
    pub(crate) fn stream(&self) -> Option<&Stream> {
        match &self.family {
            Family::Image(image) => image.stream.as_ref(),
            Family::Video(video) => Some(&video.stream),
            Family::Audio(audio) => Some(&audio.stream),
        }
    }

    /// An image decoder's pixel range, `(low, high)`.
    pub(crate) fn range(&self) -> Result<(f32, f32)> {
        Ok(self.image_decoder()?.range)
    }

    /// Decode `images` at once, for a caller outside a rank, which has no
    /// lane: the decode as serving records it (`Self::record_decode`),
    /// each image's staged pixels copied out to host memory behind its
    /// sub-batch, before the next sub-batch reuses the staging bytes, all
    /// performed on the calling thread, which then waits for the stream.
    /// Returns each image's staged pixels, in input order, readable on the
    /// host.
    pub fn decode(&self, images: &[DecodeImage]) -> Result<Vec<StorageView>> {
        let job = Job::new();
        let issue = job.on(self.stream());
        let device = self.device();
        let mut staged = Vec::with_capacity(images.len());
        let mut returns = Vec::new();
        self.record_decode(
            images,
            |_, pixels| {
                let bytes = pixels.nbytes() as usize;
                let memory: Arc<dyn crate::ffi::Backing> = match device {
                    Device::Cuda(index) => Storage::pinned(index, bytes)?,
                    Device::Cpu => Storage::host(bytes),
                };
                let copy = StorageView::contiguous(memory, 0, DType::U8, pixels.shape())?;
                let mut out = CopyBatch::new(SourceOrder::Stream);
                out.push(copy.address(), pixels.address(), bytes);
                // SAFETY: the staged pixels and their fresh copy both hold
                // `bytes`; the copy follows its sub-batch's call in the job
                // and precedes the next one, and the copy's memory lives in
                // the returned view.
                unsafe { issue.copy(&out) }?;
                staged.push(copy);
                Ok(())
            },
            &job,
            &mut returns,
        )?;
        // Performing the job issues the copies; the returned host pixels
        // are complete once the stream finished them.
        issue.synchronize()?;
        let mut completion = job.perform().check().map_err(|failure| failure.error)?;
        self.check_staged(returns, &mut completion)?;
        Ok(staged)
    }

    /// Check the byte counts the decode calls of a performed job returned
    /// against what [`Self::record_decode`] expected of them, in order.
    pub(crate) fn check_staged(
        &self,
        returns: Vec<Vec<i64>>,
        completion: &mut Completion,
    ) -> Result<()> {
        let decoder = self.image_decoder()?;
        for expected in returns {
            let counts = completion.returned()?;
            if counts != expected {
                return Err(Error::Interface(format!(
                    "{} staged {counts:?} bytes of images of {expected:?} bytes",
                    decoder.decode.name()
                )));
            }
        }
        Ok(())
    }

    /// Record the decoding of `images` into `job`: each sub-batch's call
    /// and what `stage` issues for each image's staged pixels. `returns`
    /// receives the byte counts each call must return.
    pub(crate) fn record_decode(
        &self,
        images: &[DecodeImage],
        stage: impl FnMut(usize, StorageView) -> Result<()>,
        job: &Job,
        returns: &mut Vec<Vec<i64>>,
    ) -> Result<()> {
        self.image_decoder()?
            .record_decode(images, stage, job, returns)
    }

    /// Record the quantization of a resident image product holding values
    /// in `range` into `job` and what `stage` issues for its staged pixels.
    /// `returns` receives the byte count the call must return.
    pub(crate) fn quantize(
        &self,
        image: &StorageView,
        range: (f32, f32),
        stage: impl FnOnce(StorageView) -> Result<()>,
        job: &Job,
        returns: &mut Vec<Vec<i64>>,
    ) -> Result<()> {
        self.image_decoder()?
            .quantize(image, range, stage, job, returns)
    }

    /// Construct a video decoder's Python runner of every clip output, the
    /// largest first.
    pub(crate) fn prepare_clip_outputs(&mut self, clip_outputs: &[ClipOutput]) -> Result<()> {
        self.video_decoder()?.prepare(clip_outputs)
    }

    /// Read from the model the clip rows of unit `unit` of a
    /// `frames`-frame video at `canvas`, which a decode of the unit takes.
    pub(crate) fn read_clip_rows(&mut self, frames: u32, canvas: Canvas, unit: u32) -> Result<()> {
        self.video_decoder()?.read_clip_rows(frames, canvas, unit)
    }

    /// Record into `job` the decode and reconstruction of a zero latent as
    /// a media unit of `shape` whose clip input reads `rows` rows, so kernels
    /// compile and the ring's exchange of the round shape is prepared.
    /// Every ring member holding a unit of the round records it, in the
    /// same order.
    pub(crate) fn warm_unit(
        &mut self,
        shape: UnitShape,
        rows: u64,
        longest: u32,
        job: &Job,
    ) -> Result<()> {
        self.video_decoder()?.warm(shape, rows, longest, job)
    }

    /// Record into `job` the capture of the forward of a warmed unit
    /// `shape`; whether it records one. [`Self::complete_unit`] stores the
    /// graph from the job's completion.
    pub(crate) fn capture_unit(
        &mut self,
        shape: UnitShape,
        rows: u64,
        longest: u32,
        job: &Job,
    ) -> Result<bool> {
        self.video_decoder()?.capture(shape, rows, longest, job)
    }

    /// Store the graph the job of [`Self::capture_unit`] captured for
    /// `shape`.
    pub(crate) fn complete_unit(
        &mut self,
        shape: UnitShape,
        completion: &mut Completion,
    ) -> Result<()> {
        self.video_decoder()?.complete_capture(shape, completion)
    }

    /// Decode and reconstruct media unit `unit` of a `frames`-frame video
    /// into the leading `written` frames of `out` (`[longest, height,
    /// width, 3]` uint8). `latent` is the request's complete packed latent,
    /// `shape` the unit's.
    #[expect(
        clippy::too_many_arguments,
        reason = "one unit's complete reconstruction inputs"
    )]
    pub(crate) fn decode_unit(
        &mut self,
        slot: usize,
        latent: &StorageView,
        frames: u32,
        unit: u32,
        shape: UnitShape,
        longest: u32,
        written: u32,
        out: &StorageView,
        job: &Job,
    ) -> Result<GraphMode> {
        self.video_decoder()?.decode(
            slot, latent, frames, unit, shape, longest, written, out, job,
        )
    }

    /// Construct an audio decoder's Python runner of tracks of
    /// `latent_frames` latent frames.
    pub(crate) fn prepare_track(&mut self, latent_frames: u32) -> Result<()> {
        self.audio_decoder()?.prepare(latent_frames)
    }

    /// Record into `job` the decode of `units` of a zero track whose
    /// complete latent has `rows` rows, so kernels compile before any
    /// capture.
    pub(crate) fn warm_track(&mut self, units: TrackUnits, rows: u64, job: &Job) -> Result<()> {
        self.audio_decoder()?.warm(units, rows, job)
    }

    /// Record into `job` the capture of the forward of warmed track
    /// `units`; whether it records one. [`Self::complete_track`] stores the
    /// graph from the job's completion.
    pub(crate) fn capture_track(
        &mut self,
        units: TrackUnits,
        rows: u64,
        job: &Job,
    ) -> Result<bool> {
        self.audio_decoder()?.capture(units, rows, job)
    }

    /// Store the graph the job of [`Self::capture_track`] captured for
    /// `units`.
    pub(crate) fn complete_track(
        &mut self,
        units: TrackUnits,
        completion: &mut Completion,
    ) -> Result<()> {
        self.audio_decoder()?.complete_capture(units, completion)
    }

    /// Decode `units` of a track into `out` (`[span, 2]` int16). `latent`
    /// is the request's complete audio latent.
    pub(crate) fn decode_track(
        &mut self,
        latent: &StorageView,
        units: TrackUnits,
        out: &StorageView,
        job: &Job,
    ) -> Result<GraphMode> {
        self.audio_decoder()?.decode(latent, units, out, job)
    }

    /// Refuse further captures.
    pub(crate) fn seal(&mut self) {
        match &mut self.family {
            Family::Image(_) => {}
            Family::Video(video) => video.graphs.seal(),
            Family::Audio(audio) => audio.graphs.seal(),
        }
    }

    /// The calls that list the kernels the operators of the decoder's
    /// Python runners selected.
    pub(crate) fn kernel_calls(&self) -> Vec<MethodCall> {
        match &self.family {
            Family::Image(image) => vec![kernels_call(&image.kernels, &image.runner)],
            Family::Video(video) => video.kernel_calls(),
            Family::Audio(audio) => audio.kernel_calls(),
        }
    }

    /// The kernels the operators of the decoder's Python runners selected.
    pub(crate) fn kernels(&self) -> Result<Vec<KernelRecord>> {
        list_kernels(self.kernel_calls())
    }

    /// Release the graphs and Python runners of a video or audio decoder;
    /// an image decoder releases the execution context its Python half
    /// constructed.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        match &mut self.family {
            Family::Image(image) => {
                let flag = Any::from(aborted);
                image.close.invoke(&[image.runner.view(), (&flag).into()])
            }
            Family::Video(video) => video.close(aborted),
            Family::Audio(audio) => audio.close(aborted),
        }
    }
}
