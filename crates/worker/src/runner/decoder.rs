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
//! the forward unpacks the unit's window of the request's packed latent,
//! decodes it at its segment and cross-fades and quantizes it into the
//! unit's RGB frames, exchanging decoded overlaps with its ring neighbours.
//! It is captured into one graph per unit shape and launched natively. The
//! request's retained overlap lives in its slot's row of the overlap bank.
//! An audio decoder decodes a contiguous run of audio units into their
//! sample span. A Python runner serves one module at one size, so the
//! runner holds one per size it serves: a video decoder per segment, an
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

pub(crate) mod runners;
mod start;
pub(crate) mod warmup;

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_core::Canvas;
use uniserve_cuda::{CopyBatch, SourceOrder, Stream};
use uniserve_ipc::{Batch as BatchPlan, BufferId, ForwardStats};

use crate::config::Device;
use crate::cuda_graph::{GraphMode, GraphPool, GraphTable, Lookup};
use crate::executor::batch::lock;
use crate::executor::image::ImageParts;
use crate::executor::image::{Encoding, ImageBatch, allocations};
use crate::executor::products::{OutputShape, Placement, bind_outputs};
use crate::executor::video::MediaBatch;
use crate::executor::video::{Parts, decode_range, publish_bound, stream_of};
use crate::ffi::{DType, Function, Handle, Interpreter};
use crate::host::HostTask;
use crate::host::image::{ImageTask, ImageWork};
use crate::lane::{Capture, Completion, Job};
use crate::parallel::joins::GraphFunctions;
use crate::profiling::kernels::KernelRecord;
use crate::runner::methods::{
    Argument, MethodCall, Methods, RunnerClass, kernels_call, list_kernels,
};
use crate::runner::{ImageSetup, RunnerSetup, device_buffer};
use crate::storage::memory::{Memory, StorageView};
use crate::storage::{
    LatentUpdate, Metadata, OutputPlacement, ProductBuffer, RequestSlots, SlotField, ValueRange,
};
use crate::stream::{Fill, Issue};
use crate::{Error, Result};

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
    pub fn pixel_bytes(&self) -> usize {
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
        pool: Option<Arc<GraphPool<Handle>>>,
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
        pool: Option<Arc<GraphPool<Handle>>>,
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

    pub fn device(&self) -> Device {
        match &self.family {
            Family::Image(image) => image.device,
            Family::Video(video) => Device::Cuda(video.device),
            Family::Audio(audio) => Device::Cuda(audio.device),
        }
    }

    /// The runner's stream; `None` on the host.
    pub fn stream(&self) -> Option<&Stream> {
        match &self.family {
            Family::Image(image) => image.stream.as_ref(),
            Family::Video(video) => Some(&video.stream),
            Family::Audio(audio) => Some(&audio.stream),
        }
    }

    /// An image decoder's pixel range, `(low, high)`.
    pub fn range(&self) -> Result<(f32, f32)> {
        Ok(self.image_decoder()?.range)
    }

    /// Decode `images` at once, for a caller outside a rank, which has no
    /// lane: the decode as serving records it ([`Self::record_decode`]),
    /// each image's staged pixels copied out to host memory behind its
    /// sub-batch, before the next sub-batch reuses the staging bytes, all
    /// performed on the calling thread. Returns each image's staged pixels,
    /// in input order.
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
                    Device::Cuda(index) => Memory::pinned(index, bytes)?,
                    Device::Cpu => Memory::host(bytes),
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

    /// Construct a video decoder's Python runner of every segment, the
    /// largest first.
    pub(crate) fn prepare_segments(&mut self, segments: &[Segment]) -> Result<()> {
        self.video_decoder()?.prepare(segments)
    }

    /// Read from the model the window rows of unit `unit` of a
    /// `frames`-frame video at `canvas`, which a decode of the unit takes.
    pub(crate) fn read_window(&mut self, frames: u32, canvas: Canvas, unit: u32) -> Result<()> {
        self.video_decoder()?.read_window(frames, canvas, unit)
    }

    /// Record into `job` the decode and reconstruction of a zero latent as
    /// a media unit of `shape` whose window reads `rows` rows, so kernels
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

/// An image decoder of a bound component.
struct ImageDecoder {
    owner: String,
    device: Device,
    stream: Option<Stream>,
    /// The Python half and its methods, each called with it first.
    runner: Handle,
    decode: Function,
    quantize: Function,
    kernels: Function,
    close: Function,
    /// The decoder's pixel range, `(low, high)`.
    range: (f32, f32),
    staging: StorageView,
}

impl ImageDecoder {
    fn new(
        setup: &ImageSetup<'_>,
        range: (f32, f32),
        latents: StorageView,
        staging_bytes: usize,
    ) -> Result<Self> {
        let owner = setup.owner();
        let (device, stream) = (setup.device, setup.stream.clone());
        if device.is_cuda() != stream.is_some() {
            return Err(Error::Invariant(
                "an image decoder runs on a stream exactly on a CUDA device".into(),
            ));
        }
        let memory: std::sync::Arc<dyn crate::ffi::Backing> = match device {
            Device::Cuda(index) => Memory::device(index, staging_bytes.max(1))?,
            Device::Cpu => Memory::host(staging_bytes.max(1)),
        };
        let staging = StorageView::contiguous(memory, 0, DType::U8, &[staging_bytes as i64])?;
        let method = |name: &str| Interpreter::import(&format!("{DECODER}.{name}"));
        let mut arguments = setup.arguments();
        arguments.push(latents.handle()?.to_any());
        arguments.push(staging.handle()?.to_any());
        let runner = Handle::new(method("image")?.call_values(&arguments)?)?;
        Ok(Self {
            owner,
            device,
            stream,
            runner,
            decode: method("decode")?,
            quantize: method("quantize")?,
            kernels: method("kernels")?,
            close: method("close")?,
            range,
            staging,
        })
    }

    /// The numerical calls that decode `images`: each decodes the longest
    /// prefix of the remaining images that fits the staging bytes.
    ///
    /// FIXME: every sub-batch is one eager Python call. A graph per (size,
    /// count) needs fixed latent rows, a runner-owned buffer of decoded
    /// values with native copies to the product regions, and a bound on
    /// the activations each captured size retains; a decoder on a
    /// generation tower returns its pixels through a delivery stream.
    fn chunks(&self, images: &[DecodeImage]) -> Result<Vec<Chunk>> {
        let capacity = self.staging.numel() as usize;
        let mut chunks = Vec::new();
        let mut first = 0;
        while first < images.len() {
            let mut bytes = 0;
            let mut end = first;
            while end < images.len() && bytes + images[end].pixel_bytes() <= capacity {
                bytes += images[end].pixel_bytes();
                end += 1;
            }
            if end == first {
                return Err(Error::Invalid(format!(
                    "{}: a {}x{} image exceeds the decoder staging of {capacity} bytes",
                    self.owner, images[first].height, images[first].width
                )));
            }

            let mut offsets = Vec::with_capacity(end - first);
            let mut offset = 0;
            let mut rows = Vec::with_capacity((end - first) * 5);
            for image in &images[first..end] {
                offsets.push(offset);
                rows.extend([
                    i64::from(image.sample),
                    i64::from(image.units),
                    i64::from(image.height),
                    i64::from(image.width),
                    offset as i64,
                ]);
                offset += image.pixel_bytes();
            }
            // Each image's product follows the rows: a tensor, or none
            // for an image that publishes no product.
            let mut arguments = vec![
                Argument::Handle(self.runner.clone()),
                Argument::Integers(rows),
            ];
            for image in &images[first..end] {
                arguments.push(match &image.product {
                    Some(product) => Argument::Tensor(product.clone()),
                    None => Argument::None,
                });
            }
            chunks.push(Chunk {
                images: first..end,
                arguments,
                offsets,
            });
            first = end;
        }
        Ok(chunks)
    }

    /// Record the decoding of `images` into `job`: each sub-batch's call,
    /// then what `stage` issues for each image's staged pixels (in input
    /// order), which must precede the next sub-batch because it reuses the
    /// staging bytes. `returns` receives the byte counts each call must
    /// return, for the job's completion to check.
    fn record_decode(
        &self,
        images: &[DecodeImage],
        mut stage: impl FnMut(usize, StorageView) -> Result<()>,
        job: &Job,
        returns: &mut Vec<Vec<i64>>,
    ) -> Result<()> {
        for chunk in self.chunks(images)? {
            job.call_returning(MethodCall::new(&self.decode, chunk.arguments));
            let decoded = &images[chunk.images.clone()];
            returns.push(pixel_counts(decoded));
            for ((index, image), &offset) in chunk.images.zip(decoded).zip(&chunk.offsets) {
                stage(
                    index,
                    self.staging
                        .narrow(0, offset as i64, image.pixel_bytes() as i64)?,
                )?;
            }
        }
        Ok(())
    }

    /// Record the quantization of a resident image product holding values
    /// in `range` into `job`, then what `stage` issues for its staged
    /// pixels (see [`Self::record_decode`]).
    fn quantize(
        &self,
        image: &StorageView,
        range: (f32, f32),
        stage: impl FnOnce(StorageView) -> Result<()>,
        job: &Job,
        returns: &mut Vec<Vec<i64>>,
    ) -> Result<()> {
        // The quantized pixels are one byte per element of the image.
        let count = image.numel();
        if count == 0 || count > self.staging.numel() {
            return Err(Error::Invalid(format!(
                "{}: an image of {count} pixel bytes does not fit the decoder staging",
                self.owner
            )));
        }
        job.call_returning(MethodCall::new(
            &self.quantize,
            vec![
                Argument::Handle(self.runner.clone()),
                Argument::Tensor(image.clone()),
                Argument::Float(f64::from(range.0)),
                Argument::Float(f64::from(range.1)),
                Argument::Integer(0),
            ],
        ));
        returns.push(vec![count as i64]);
        stage(self.staging.narrow(0, 0, count as i64)?)
    }
}

/// One numerical call of an image decode: the images it decodes, its
/// arguments and each image's offset in the staging bytes.
struct Chunk {
    images: Range<usize>,
    arguments: Vec<Argument>,
    offsets: Vec<usize>,
}

/// The staged byte count of each of `images`.
fn pixel_counts(images: &[DecodeImage]) -> Vec<i64> {
    images
        .iter()
        .map(|image| image.pixel_bytes() as i64)
        .collect()
}

// Image decoding batches (`ImageDecoding`) and their PNG outputs: a call
// decodes its completed trajectory (gathered from the committed bank into
// the step workspace) or, without a trajectory, re-encodes a resident image
// product. The decoder optionally publishes the decoded values as the
// call's image product (feedback) and always stages the quantized HWC
// pixels, which are copied into the batch's pinned output storage on the
// decoder's stream. Once those copies complete, the host lane encodes each
// image as a base64 PNG; the result is the call's media output. Decoding
// releases the trajectory.

/// The decoder serving a batch, with its device and stream, and the
/// output words its pixels need.
pub(crate) fn image_placement(
    image: &ImageParts<'_>,
    plan: &BatchPlan,
) -> Result<(Option<i32>, Option<Stream>, usize)> {
    let runner = image
        .decoder()
        .ok_or_else(|| Error::Unsupported("image decoding is served by an image decoder".into()))?;
    let stream = runner.stream().cloned();
    // Each image's pixels, read back as raw bytes.
    let words = plan
        .calls
        .iter()
        .map(|call| {
            usize::try_from(call.bounds.max_completion_bytes.div_ceil(8)).unwrap_or(usize::MAX)
        })
        .fold(0_usize, usize::saturating_add);
    Ok((
        runner.device().cuda_index(),
        stream,
        words.min(image.decoders.max_pixel_words),
    ))
}

/// The value range of a decoder's pixel interval, as products record it.
pub(crate) fn value_range(range: (f32, f32)) -> Option<ValueRange> {
    match range {
        (low, high) if low == 0.0 && high == 1.0 => Some(ValueRange::Unit),
        (low, high) if low == -1.0 && high == 1.0 => Some(ValueRange::SignedUnit),
        _ => None,
    }
}

/// What a decoding call reads.
enum Source {
    /// A trajectory gathered at `DecodeImage::sample`.
    Latent(DecodeImage),
    /// A resident image product and its value range.
    Resident(StorageView, (f32, f32), u32, u32),
}

impl ImageBatch<'_> {
    /// Decode every active call's image and stage its pixels for PNG
    /// encoding.
    pub(crate) fn decode(&mut self) -> Result<ForwardStats> {
        // The forward records into the batch's job, taken for its length so
        // that the batch's owners stay borrowable beside it.
        let job = std::mem::take(&mut self.job);
        let stats = self.record_decode(&job);
        self.job = job;
        stats
    }

    fn record_decode(&mut self, job: &Job) -> Result<ForwardStats> {
        let plan = self.plan;
        let active = self.active();
        if active.is_empty() {
            return Ok(ForwardStats::default());
        }
        let target = self.target();
        let stream = self.batch.stream.clone();
        let issue = job.on(stream.as_ref());
        let decoder_range = self
            .parts
            .decoder()
            .ok_or_else(|| Error::Unsupported("image decoding requires a decoder".into()))?
            .range()?;

        // One encoding task per reporting call, reserved before any device
        // work so a full host lane fails the batch cleanly.
        let mut tasks: HashMap<usize, Arc<HostTask<ImageTask>>> = HashMap::new();
        if self.batch.reports {
            for &index in &active {
                let task = self.parts.host.reserve()?;
                self.encodings.push(Encoding {
                    call: index,
                    task: Arc::clone(&task),
                });
                tasks.insert(index, task);
            }
        }

        // Image products of feedback calls, in their scheduler allocations.
        let allocations = allocations(plan);
        let mut shapes: HashMap<BufferId, Vec<u64>> = HashMap::new();
        let mut bindings = Vec::new();
        for &index in &active {
            let call = &plan.calls[index];
            if let (Some(reference), Some(params)) = (&call.image_output, self.decode_size(index)?)
            {
                shapes.insert(
                    reference.buffer_id(),
                    vec![3, u64::from(params.0), u64::from(params.1)],
                );
                bindings.push((reference.clone(), target));
            }
        }
        let products: Vec<ProductBuffer> = if bindings.is_empty() {
            Vec::new()
        } else {
            self.storage.tensors.bind_outputs(
                &bindings,
                OutputPlacement {
                    allocations: Some(&allocations),
                    shapes: Some(&shapes),
                    ..OutputPlacement::default()
                },
            )?
        };
        self.batch.writes.extend(products.iter().cloned());
        let product = |reference: &uniserve_ipc::TensorRef| {
            products
                .iter()
                .find(|write| write.id() == reference.buffer_id())
                .cloned()
        };

        // Each call's source: a gathered trajectory or a resident image.
        let mut sources = Vec::with_capacity(active.len());
        let mut latent_calls = Vec::new();
        for &index in &active {
            let call = &plan.calls[index];
            if call.bounds.max_completion_bytes == 0 {
                return Err(Error::Invalid(
                    "image decoding requires a completion byte bound".into(),
                ));
            }
            if call.latent_input.is_some() {
                latent_calls.push(index);
                sources.push(None);
                continue;
            }
            let reference = call.image_input.as_ref().ok_or_else(|| {
                Error::Invalid("image decoding requires a trajectory or a resident image".into())
            })?;
            let read =
                self.storage
                    .tensors
                    .consume(reference, call.call_id, Some(target), issue)?;
            let (height, width, range) = match read.metadata() {
                Some(Metadata::Image {
                    height,
                    width,
                    value_range,
                }) => (
                    height,
                    width,
                    match value_range {
                        Some(ValueRange::Unit) => (0.0, 1.0),
                        Some(ValueRange::SignedUnit) | None => (-1.0, 1.0),
                    },
                ),
                _ => {
                    self.batch.reads.push(read);
                    return Err(Error::Invalid(
                        "image encoding requires image product metadata".into(),
                    ));
                }
            };
            let resident = read.value().clone();
            self.batch.reads.push(read);
            sources.push(Some(Source::Resident(resident, range, height, width)));
        }

        if !latent_calls.is_empty() {
            let mut trajectories = Vec::with_capacity(latent_calls.len());
            for &index in &latent_calls {
                let call = &plan.calls[index];
                let params = self
                    .plan
                    .latent_params
                    .iter()
                    .find(|params| {
                        params.request_key == call.request_key && params.call_id == call.call_id
                    })
                    .cloned()
                    .ok_or_else(|| {
                        Error::Invalid("image decoding has no latent allocation".into())
                    })?;
                let steps = self
                    .requests
                    .get(call.request_key.request_id.0)?
                    .admission()
                    .image
                    .as_ref()
                    .map(|image| u32::from(image.steps))
                    .ok_or_else(|| {
                        Error::Invalid("image decoding requires image parameters".into())
                    })?;
                if params.start_step != steps {
                    return Err(Error::Invalid(
                        "image decoding requires a completed trajectory".into(),
                    ));
                }
                trajectories.push((index, params));
            }
            let latents = self.parts.latents()?;
            latents.enter(issue)?;
            self.latents_entered = true;
            let params: Vec<_> = trajectories.iter().map(|(_, params)| params).collect();
            let scratch = latents.place(&params)?;
            for ((index, params), scratch) in trajectories.iter().zip(&scratch) {
                let call = &plan.calls[*index];
                let slot = self.batch.calls[*index].slot;
                let generation = call
                    .latent_input
                    .as_ref()
                    .map_or(0, |input| i64::from(input.generation));
                let bank = latents.pool.current_bank(
                    slot,
                    i64::from(params.start_step),
                    generation,
                    i64::from(params.latent_units),
                    i64::from(params.height),
                    i64::from(params.width),
                    &scratch.pages,
                )?;
                latents.gather(bank, scratch, issue)?;
                let product = call
                    .image_output
                    .as_ref()
                    .and_then(product)
                    .map(|write| write.view());
                let position = active
                    .iter()
                    .position(|value| value == index)
                    .ok_or_else(|| Error::Invariant("a decoding call is not active".into()))?;
                sources[position] = Some(Source::Latent(DecodeImage {
                    sample: scratch.sample,
                    units: scratch.units,
                    height: params.height,
                    width: params.width,
                    product,
                }));
                self.updates
                    .push((*index, LatentUpdate::for_call(slot, call, params)?));
            }
        }

        // Decode, then copy each image's staged pixels to the host. The
        // job's completion checks the byte counts the calls return.
        let mut returns = Vec::new();
        let mut decoded = Vec::new();
        let mut staged: Vec<(usize, StorageView, u32, u32)> = Vec::new();
        let mut stage =
            |index: usize, pixels: StorageView, height: u32, width: u32| -> Result<()> {
                let call = &plan.calls[index];
                if pixels.numel() > call.bounds.max_completion_bytes {
                    return Err(Error::Invalid(
                        "image pixels exceed the call's completion byte bound".into(),
                    ));
                }
                let mut buffer = lock(&self.batch.buffer);
                let host = buffer.capture_bytes(&pixels, issue, &self.storage.events)?;
                if self.batch.reports {
                    buffer.retain_reader()?;
                }
                staged.push((index, host, height, width));
                Ok(())
            };
        let runner = self
            .parts
            .decoders
            .image
            .as_ref()
            .map(|(_, runner)| runner)
            .ok_or_else(|| Error::Unsupported("image decoding requires a decoder".into()))?;
        for (&index, source) in active.iter().zip(&sources) {
            match source {
                Some(Source::Resident(region, range, height, width)) => {
                    runner.quantize(
                        region,
                        *range,
                        |pixels| stage(index, pixels, *height, *width),
                        job,
                        &mut returns,
                    )?;
                }
                Some(Source::Latent(image)) => decoded.push((index, image.clone())),
                None => return Err(Error::Invariant("a decoding call has no source".into())),
            }
        }
        let images: Vec<DecodeImage> = decoded.iter().map(|(_, image)| image.clone()).collect();
        runner.record_decode(
            &images,
            |position, pixels| {
                let (index, image) = &decoded[position];
                stage(*index, pixels, image.height, image.width)
            },
            job,
            &mut returns,
        )?;
        drop(stage);
        self.returns.append(&mut returns);

        // Feedback products hold the decoded values.
        for (index, image) in &decoded {
            let call = &plan.calls[*index];
            if let Some(write) = call.image_output.as_ref().and_then(product) {
                self.storage.tensors.produced(
                    &write,
                    &[3, u64::from(image.height), u64::from(image.width)],
                    issue,
                    Some(Metadata::Image {
                        height: image.height,
                        width: image.width,
                        value_range: value_range(decoder_range),
                    }),
                )?;
            }
        }

        // PNG encodings run once the batch's copies complete.
        for (index, pixels, height, width) in staged {
            let Some(task) = tasks.get(&index) else {
                continue;
            };
            task.configure(ImageTask::new(ImageWork::Encode {
                pixels,
                lease: Arc::clone(&self.batch.buffer),
                height,
                width,
                maximum: usize::try_from(plan.calls[index].bounds.max_completion_bytes)
                    .unwrap_or(usize::MAX),
            }))?;
        }
        let mut stats = ForwardStats::default();
        *stats.mode_counts.entry("eager".into()).or_default() += 1;
        Ok(stats)
    }

    /// The decoded size of call `index`: its trajectory's, or its resident
    /// image's when known only at execution.
    fn decode_size(&self, index: usize) -> Result<Option<(u32, u32)>> {
        let call = &self.plan.calls[index];
        if call.latent_input.is_none() {
            return Ok(None);
        }
        Ok(self
            .plan
            .latent_params
            .iter()
            .find(|params| params.request_key == call.request_key && params.call_id == call.call_id)
            .map(|params| (params.height, params.width)))
    }
}

/// A segment: the native frames and raster one unit decodes to.
pub(crate) type Segment = (u32, Canvas);

/// What fixes a video decoding forward's computation: the unit's segment,
/// whether the unit is its video's first and last, and how many ring
/// members hold a unit of its round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct UnitShape {
    pub(crate) segment: Segment,
    pub(crate) first: bool,
    pub(crate) last: bool,
    pub(crate) count: u32,
}

/// The largest extents of a video decoding forward's tensors.
pub(crate) struct DecodeCapacity {
    /// `[rows, width]` of the longest complete packed latent.
    pub(crate) latent: [u64; 2],
    /// `[frames, height, width]` bounding every unit's RGB frames.
    pub(crate) frames: [u64; 3],
    /// The largest canvas's overlap as `(elements, dtype)`.
    pub(crate) overlap: (u64, DType),
}

/// A video decoder and its RGB post-processor on one rank.
///
/// A forward reads and writes four tensors at fixed addresses, so one
/// captured graph per [`UnitShape`] serves every request: the request's
/// packed latent, the rows of it the unit's window reads, the request's
/// retained overlap and the unit's RGB frames. The runner copies a
/// request's values in and out of them natively around the forward.
struct VideoDecoder {
    class: RunnerClass,
    loaded: Handle,
    stream: Stream,
    device: i32,
    /// The runner of each segment, in preparation order.
    runners: Vec<(Segment, Methods)>,
    /// The retained overlap of each request slot (`[slots + 1, elements]`):
    /// the decoded tail a request's previous round left for its next.
    overlap: RequestSlots,
    /// `[rows, width]` FP32: a request's complete packed latent.
    latent: StorageView,
    /// `[rows]` int64: the latent rows a unit's window reads.
    rows: StorageView,
    /// The forward's view of the request's retained overlap.
    retained: StorageView,
    /// Flat uint8: the unit's `[frames, height, width, 3]` RGB frames.
    frames: StorageView,
    /// The rows of each `(frames, canvas, unit)` window, as the model
    /// names them.
    windows: HashMap<(u32, Canvas, u32), Vec<u8>>,
    window_rows: Function,
    graphs: GraphTable<UnitShape>,
    /// Captures a forward between its begin and end.
    capture: GraphFunctions,
    /// The graph pool captures allocate in; `None` serves forwards eagerly.
    pool: Option<Arc<GraphPool<Handle>>>,
}

impl VideoDecoder {
    pub(crate) fn new(
        setup: RunnerSetup<'_>,
        capacity: DecodeCapacity,
        slots: u32,
        pool: Option<Arc<GraphPool<Handle>>>,
    ) -> Result<Self> {
        let (elements, dtype) = capacity.overlap;
        let overlap = RequestSlots::allocate(
            &[SlotField {
                name: "video_overlap".into(),
                dtype,
                shape: vec![elements as i64],
                host: false,
            }],
            slots,
            Some(setup.device),
        )?;
        let [rows, width] = capacity.latent.map(|extent| extent as i64);
        let [frames, height, canvas_width] = capacity.frames.map(|extent| extent as i64);
        let capture = GraphFunctions::import()?;
        Ok(Self {
            device: setup.device,
            runners: Vec::new(),
            overlap,
            latent: device_buffer(setup.device, DType::F32, &[rows, width])?,
            rows: device_buffer(setup.device, DType::I64, &[rows])?,
            retained: device_buffer(setup.device, dtype, &[elements as i64])?,
            frames: device_buffer(
                setup.device,
                DType::U8,
                &[frames * height * canvas_width * 3],
            )?,
            windows: HashMap::new(),
            window_rows: Interpreter::import(&format!("{DECODER}.window_rows"))?,
            graphs: GraphTable::new(format!("{}.decoder", setup.path), capture.release.clone()),
            capture,
            pool,
            loaded: setup.loaded.clone(),
            class: RunnerClass::new(DECODER, "decode", setup.arguments())?,
            stream: setup.stream.clone(),
        })
    }

    /// Construct the runner of every segment, the largest first.
    ///
    /// Startup alone constructs (`DecoderRunners::prepare_startup`), for
    /// every segment an admitted video decodes to: a serving rank
    /// constructs no decoder runner, and a decode of a segment without one
    /// is refused.
    pub(crate) fn prepare(&mut self, segments: &[Segment]) -> Result<()> {
        for &segment in segments {
            if self
                .runners
                .iter()
                .any(|(prepared, _)| *prepared == segment)
            {
                continue;
            }
            let (frames, canvas) = segment;
            let runner = self.class.construct(vec![Argument::Integers(vec![
                i64::from(frames),
                i64::from(canvas.width),
                i64::from(canvas.height),
            ])])?;
            self.runners.push((segment, runner));
        }
        Ok(())
    }

    /// The arguments of `DecoderRunner.decode` for a unit of `shape` whose
    /// window reads `rows` latent rows and whose video's longest unit has
    /// `longest` frames.
    fn arguments(&self, shape: UnitShape, rows: u64, longest: u32) -> Result<Vec<Argument>> {
        let canvas = shape.segment.1;
        let runner = self
            .runners
            .iter()
            .find(|(prepared, _)| *prepared == shape.segment)
            .map(|(_, runner)| runner)
            .ok_or_else(|| Error::Invariant("a video segment has no decoder runner".into()))?;
        let frames = self.unit_frames(canvas, longest)?;
        Ok(vec![
            Argument::Handle(runner.runner().clone()),
            Argument::Tensor(self.latent.clone()),
            Argument::Tensor(self.rows.prefix(rows as i64)?),
            Argument::Tensor(self.retained.clone()),
            Argument::Tensor(frames),
            Argument::Bool(shape.first),
            Argument::Bool(shape.last),
            Argument::Integer(i64::from(shape.count)),
        ])
    }

    /// The forward's `[frames, height, width, 3]` view of the RGB frames.
    fn unit_frames(&self, canvas: Canvas, frames: u32) -> Result<StorageView> {
        let shape = [
            i64::from(frames),
            i64::from(canvas.height),
            i64::from(canvas.width),
            3,
        ];
        self.frames
            .prefix(shape.iter().product())?
            .reshape(DType::U8, &shape)
    }

    /// Read from the model the window rows of unit `unit` of a
    /// `frames`-frame video at `canvas`. Startup reads the window of every
    /// unit of every admitted video; a decode takes its window from that
    /// table ([`Self::window`]).
    fn read_window(&mut self, frames: u32, canvas: Canvas, unit: u32) -> Result<()> {
        let key = (frames, canvas, unit);
        if !self.windows.contains_key(&key) {
            let value = self.window_rows.call_values(&[
                self.loaded.to_any(),
                Any::from(i64::from(frames)),
                Any::from(i64::from(canvas.width)),
                Any::from(i64::from(canvas.height)),
                Any::from(i64::from(unit)),
            ])?;
            let rows = crate::ffi::integers(value)
                .map_err(|error| Error::numerical(self.window_rows.name(), &error))?;
            self.windows
                .insert(key, rows.iter().flat_map(|row| row.to_ne_bytes()).collect());
        }
        Ok(())
    }

    /// The window rows of unit `unit` of a `frames`-frame video at
    /// `canvas`, as native int64 bytes.
    fn window(&self, frames: u32, canvas: Canvas, unit: u32) -> Result<&[u8]> {
        self.windows
            .get(&(frames, canvas, unit))
            .map(Vec::as_slice)
            .ok_or_else(|| Error::Invalid("the rank admits no video with this media unit".into()))
    }

    /// Record into `job` the decode and reconstruction of a zero latent as
    /// a unit of `shape` whose window reads `rows` rows, the forward as
    /// serving records it, so kernels compile and the ring's exchange of
    /// the round shape is prepared, then the wait for the stream. Every
    /// member holding a unit of the round records it, in the same order.
    /// The forward's tensors are zero: the latent, the window rows and the
    /// retained overlap.
    pub(crate) fn warm(
        &mut self,
        shape: UnitShape,
        rows: u64,
        longest: u32,
        job: &Job,
    ) -> Result<()> {
        let stream = self.stream.clone();
        let issue = job.on(Some(&stream));
        self.latent.zero(issue)?;
        self.rows.zero(issue)?;
        self.retained.zero(issue)?;
        // No shape is captured before every shape is warmed: the forward
        // is the eager call.
        self.forward(shape, rows, longest, false, job)?;
        issue.synchronize()
    }

    /// Record into `job` the capture of the forward of a warmed `shape`;
    /// whether it records one: a decoder without a graph pool serves every
    /// shape eagerly, and a captured shape is not captured again.
    pub(crate) fn capture(
        &mut self,
        shape: UnitShape,
        rows: u64,
        longest: u32,
        job: &Job,
    ) -> Result<bool> {
        if self.pool.is_none() || self.graphs.contains(&shape, None) {
            return Ok(false);
        }
        Ok(self.forward(shape, rows, longest, true, job)? == GraphMode::Capture)
    }

    /// Store the graph the job of [`Self::capture`] captured for `shape`.
    pub(crate) fn complete_capture(
        &mut self,
        shape: UnitShape,
        completion: &mut Completion,
    ) -> Result<()> {
        self.graphs.insert(shape, None, completion.captured()?)?;
        self.graphs.record(GraphMode::Capture);
        Ok(())
    }

    /// Record in `job` the forward of a unit of `shape` whose window reads
    /// `rows` latent rows, on the decoder's stream: the launch of the
    /// shape's captured graph, or the eager call of `DecoderRunner.decode`,
    /// which serves a shape startup did not capture. With `capture`, a
    /// shape not captured yet is captured instead of called (startup's
    /// capture of a warmed shape): the job's completion carries the graph
    /// ([`Self::complete_capture`]). A capture is counted when its graph is
    /// stored.
    fn forward(
        &mut self,
        shape: UnitShape,
        rows: u64,
        longest: u32,
        capture: bool,
        job: &Job,
    ) -> Result<GraphMode> {
        let method = |decoder: &Self| -> Result<MethodCall> {
            Ok(MethodCall::new(
                &decoder.class.forward,
                decoder.arguments(shape, rows, longest)?,
            ))
        };
        let mode = match self.graphs.lookup(&shape, None) {
            Ok(Lookup::Replay(graph)) => {
                job.on(Some(&self.stream)).launch(graph)?;
                GraphMode::Replay
            }
            Ok(Lookup::Capture) if capture => {
                let pool = self.pool.as_ref().ok_or_else(|| {
                    Error::Invariant("a video decoder captures into its graph pool".into())
                })?;
                job.capture(Capture {
                    functions: self.capture.clone(),
                    pool: pool.value.clone(),
                    peers: Vec::new(),
                    stream: self.stream.raw(),
                    device: self.device,
                    method: method(self)?,
                    breaks: None,
                });
                GraphMode::Capture
            }
            // A shape startup did not capture runs on the Python thread.
            _ => {
                job.call(method(self)?);
                GraphMode::Eager
            }
        };
        if mode != GraphMode::Capture {
            self.graphs.record(mode);
        }
        Ok(mode)
    }

    /// Decode and reconstruct media unit `unit` of a `frames`-frame video
    /// into the leading `written` frames of `out` (`[longest, height,
    /// width, 3]` uint8). `latent` is the request's complete packed latent,
    /// `shape` the unit's.
    #[expect(
        clippy::too_many_arguments,
        reason = "one unit's complete reconstruction inputs"
    )]
    pub(crate) fn decode(
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
        let canvas = shape.segment.1;
        let stream = self.stream.clone();
        let issue = job.on(Some(&stream));
        let latent_bytes = latent.numel() * latent.dtype().bytes() as u64;
        if !latent.is_contiguous()
            || latent.dtype() != DType::F32
            || latent_bytes > self.latent.numel() * 4
        {
            return Err(Error::Invalid(
                "a video latent exceeds the decoder's latent rows".into(),
            ));
        }
        let window = self.window(frames, canvas, unit)?.to_vec();
        let rows = (window.len() / 8) as u64;
        let slot_overlap = self.overlap.slot("video_overlap", slot as u32)?;
        let overlap_bytes = self.retained.numel() * self.retained.dtype().bytes() as u64;
        self.rows.prefix(rows as i64)?.upload(&window, issue)?;
        // The two copies write disjoint tensors, so they need no order
        // between them.
        let mut inputs = CopyBatch::new(SourceOrder::Stream);
        inputs.push(
            self.latent.address(),
            latent.address(),
            latent_bytes as usize,
        );
        inputs.push(
            self.retained.address(),
            slot_overlap.address(),
            overlap_bytes as usize,
        );
        // SAFETY: the forward's tensors and the slot's overlap row belong to
        // this runner and the copies stay inside them, in disjoint ranges;
        // the latent is leased until the batch commits and was waited for
        // on this stream.
        unsafe { issue.copy(&inputs) }?;

        let mode = self.forward(shape, rows, longest, false, job)?;

        let frame_bytes = u64::from(canvas.height) * u64::from(canvas.width) * 3;
        let mut outputs = CopyBatch::new(SourceOrder::Stream);
        outputs.push(
            slot_overlap.address(),
            self.retained.address(),
            overlap_bytes as usize,
        );
        outputs.push(
            out.address(),
            self.frames.address(),
            (u64::from(written) * frame_bytes) as usize,
        );
        // SAFETY: as above; `out` holds the unit's frames and the copies
        // follow the forward on the stream.
        unsafe { issue.copy(&outputs) }?;
        Ok(mode)
    }

    pub(crate) fn kernel_calls(&self) -> Vec<MethodCall> {
        self.class
            .kernel_calls(self.runners.iter().map(|(_, runner)| runner))
    }

    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let graphs = self.graphs.close(aborted);
        graphs.and(
            self.class
                .close(aborted, self.runners.iter().map(|(_, runner)| runner)),
        )
    }
}

/// What fixes an audio decoding forward's computation: the track's samples
/// and which of how many units the forward decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TrackUnits {
    pub(crate) samples: u64,
    /// Latent frames of the track.
    pub(crate) latent_frames: u32,
    pub(crate) first: u32,
    pub(crate) count: u32,
    pub(crate) total: u32,
    /// Samples the units span.
    pub(crate) span: u64,
}

/// One rank's audio decoder.
///
/// A forward reads the request's audio latent and writes its samples at
/// fixed addresses, so one captured graph per [`TrackUnits`] serves every
/// request; the runner copies a request's values in and out natively.
struct AudioDecoder {
    class: RunnerClass,
    stream: Stream,
    device: i32,
    /// The runner of each track length, by its latent frames.
    runners: BTreeMap<u32, Methods>,
    /// `[rows, width]` FP32: a request's complete audio latent.
    latent: StorageView,
    /// `[samples, 2]` int16: the samples of the decoded units.
    samples: StorageView,
    graphs: GraphTable<TrackUnits>,
    /// Captures a forward between its begin and end.
    capture: GraphFunctions,
    /// The graph pool captures allocate in; `None` serves forwards eagerly.
    pool: Option<Arc<GraphPool<Handle>>>,
}

impl AudioDecoder {
    /// `latent` is the `[rows, width]` of the longest track's complete
    /// latent and `samples` that track's samples.
    pub(crate) fn new(
        setup: RunnerSetup<'_>,
        latent: [u64; 2],
        samples: u64,
        pool: Option<Arc<GraphPool<Handle>>>,
    ) -> Result<Self> {
        let capture = GraphFunctions::import()?;
        Ok(Self {
            device: setup.device,
            runners: BTreeMap::new(),
            latent: device_buffer(
                setup.device,
                DType::F32,
                &[latent[0] as i64, latent[1] as i64],
            )?,
            samples: device_buffer(setup.device, DType::I16, &[samples as i64, 2])?,
            graphs: GraphTable::new(format!("{}.decoder", setup.path), capture.release.clone()),
            capture,
            pool,
            class: RunnerClass::new(DECODER, "decode", setup.arguments())?,
            stream: setup.stream.clone(),
        })
    }

    /// Construct the runner of tracks of `latent_frames` latent frames.
    ///
    /// Startup alone constructs (`DecoderRunners::prepare_startup`), for
    /// the track of every admitted frame count: a serving rank constructs
    /// no decoder runner, and a track without one is refused.
    pub(crate) fn prepare(&mut self, latent_frames: u32) -> Result<()> {
        if !self.runners.contains_key(&latent_frames) {
            let runner = self
                .class
                .construct(vec![Argument::Integers(vec![i64::from(latent_frames)])])?;
            self.runners.insert(latent_frames, runner);
        }
        Ok(())
    }

    /// The arguments of `DecoderRunner.decode` for `units` of a track whose
    /// complete latent has `rows` rows.
    fn arguments(&self, units: TrackUnits, rows: u64) -> Result<Vec<Argument>> {
        let runner = self.runners.get(&units.latent_frames).ok_or_else(|| {
            Error::Invalid("the audio track's length has no decoder runner".into())
        })?;
        Ok(vec![
            Argument::Handle(runner.runner().clone()),
            Argument::Tensor(self.latent.prefix(rows as i64)?),
            Argument::Tensor(self.samples.prefix(units.span as i64)?),
            Argument::Integer(units.samples as i64),
            Argument::Integer(i64::from(units.first)),
            Argument::Integer(i64::from(units.count)),
            Argument::Integer(i64::from(units.total)),
        ])
    }

    /// Record into `job` the decode of `units` of a zero track whose
    /// complete latent has `rows` rows, the forward as serving records it,
    /// so kernels compile before any capture, then the wait for the stream.
    pub(crate) fn warm(&mut self, units: TrackUnits, rows: u64, job: &Job) -> Result<()> {
        let stream = self.stream.clone();
        let issue = job.on(Some(&stream));
        self.latent.zero(issue)?;
        // No units are captured before every unit is warmed: the forward is
        // the eager call.
        self.forward(units, rows, false, job)?;
        issue.synchronize()
    }

    /// Record into `job` the capture of the forward of warmed `units`;
    /// whether it records one: a decoder without a graph pool serves every
    /// track eagerly, and captured units are not captured again.
    pub(crate) fn capture(&mut self, units: TrackUnits, rows: u64, job: &Job) -> Result<bool> {
        if self.pool.is_none() || self.graphs.contains(&units, None) {
            return Ok(false);
        }
        Ok(self.forward(units, rows, true, job)? == GraphMode::Capture)
    }

    /// Store the graph the job of [`Self::capture`] captured for `units`.
    pub(crate) fn complete_capture(
        &mut self,
        units: TrackUnits,
        completion: &mut Completion,
    ) -> Result<()> {
        self.graphs.insert(units, None, completion.captured()?)?;
        self.graphs.record(GraphMode::Capture);
        Ok(())
    }

    /// Record in `job` the forward of `units` of a track whose complete
    /// latent has `rows` rows, on the decoder's stream: the launch of the
    /// units' captured graph, or the eager call of `DecoderRunner.decode`,
    /// which serves units startup did not capture. With `capture`, units
    /// not captured yet are captured instead of called (startup's capture
    /// of warmed units): the job's completion carries the graph
    /// ([`Self::complete_capture`]). A capture is counted when its graph is
    /// stored.
    fn forward(
        &mut self,
        units: TrackUnits,
        rows: u64,
        capture: bool,
        job: &Job,
    ) -> Result<GraphMode> {
        let method = |decoder: &Self| -> Result<MethodCall> {
            Ok(MethodCall::new(
                &decoder.class.forward,
                decoder.arguments(units, rows)?,
            ))
        };
        let mode = match self.graphs.lookup(&units, None) {
            Ok(Lookup::Replay(graph)) => {
                job.on(Some(&self.stream)).launch(graph)?;
                GraphMode::Replay
            }
            Ok(Lookup::Capture) if capture => {
                let pool = self.pool.as_ref().ok_or_else(|| {
                    Error::Invariant("an audio decoder captures into its graph pool".into())
                })?;
                job.capture(Capture {
                    functions: self.capture.clone(),
                    pool: pool.value.clone(),
                    peers: Vec::new(),
                    stream: self.stream.raw(),
                    device: self.device,
                    method: method(self)?,
                    breaks: None,
                });
                GraphMode::Capture
            }
            // Units startup did not capture run on the Python thread.
            _ => {
                job.call(method(self)?);
                GraphMode::Eager
            }
        };
        if mode != GraphMode::Capture {
            self.graphs.record(mode);
        }
        Ok(mode)
    }

    /// Decode `units` of a track into `out` (`[span, 2]` int16). `latent`
    /// is the request's complete audio latent.
    pub(crate) fn decode(
        &mut self,
        latent: &StorageView,
        units: TrackUnits,
        out: &StorageView,
        job: &Job,
    ) -> Result<GraphMode> {
        let stream = self.stream.clone();
        let issue = job.on(Some(&stream));
        let rows = latent.shape().first().copied().unwrap_or(0).max(0) as u64;
        let latent_bytes = latent.numel() * latent.dtype().bytes() as u64;
        if !latent.is_contiguous()
            || latent.dtype() != DType::F32
            || latent_bytes > self.latent.numel() * 4
        {
            return Err(Error::Invalid(
                "an audio latent exceeds the decoder's latent rows".into(),
            ));
        }
        let mut input = CopyBatch::new(SourceOrder::Stream);
        input.push(
            self.latent.address(),
            latent.address(),
            latent_bytes as usize,
        );
        // SAFETY: the forward's tensors belong to this runner and the copy
        // stays inside them; the latent is leased until the batch commits
        // and was waited for on this stream.
        unsafe { issue.copy(&input) }?;

        let mode = self.forward(units, rows, false, job)?;

        let mut output = CopyBatch::new(SourceOrder::Stream);
        output.push(
            out.address(),
            self.samples.address(),
            (units.span * 4) as usize,
        );
        // SAFETY: as above; `out` holds the units' samples and the copy
        // follows the forward on the stream.
        unsafe { issue.copy(&output) }?;
        Ok(mode)
    }

    pub(crate) fn kernel_calls(&self) -> Vec<MethodCall> {
        self.class.kernel_calls(self.runners.values())
    }

    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let graphs = self.graphs.close(aborted);
        graphs.and(self.class.close(aborted, self.runners.values()))
    }
}

/// Decode and reconstruct this rank's media unit of a video round into its
/// row of the round's product.
pub(crate) fn decode_video(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
    job: &Job,
) -> Result<()> {
    let call = &plan.calls[index];
    let range = decode_range(plan, call)?;
    let (frames, canvas, longest, units) = {
        let request = parts.video.request(slot, call.request_key)?;
        (
            request.frames,
            request.canvas,
            request.longest(),
            Arc::clone(&request.units),
        )
    };
    let total = units.len() as u32;
    if call.inputs.len() != 1 || call.outputs.len() != 1 {
        return Err(Error::Invalid(
            "media reconstruction requires one tensor input and output".into(),
        ));
    }
    if range.max_units == 0 || range.cursor + range.max_units > total {
        return Err(Error::Invalid(
            "media decoder assignment exceeds its media unit range".into(),
        ));
    }
    let rank = parts.rank;
    let decoder = parts
        .decoders
        .video
        .as_ref()
        .ok_or_else(|| Error::Invalid("this rank holds no video decoder".into()))?;
    let run = decoder.units(rank, range.cursor, range.max_units)?;
    if run.is_empty() {
        // This rank holds no unit of the round and joins no exchange.
        return Ok(());
    }
    if run.len() != 1 {
        return Err(Error::Unsupported(
            "video decoding reconstructs one media unit per rank and round".into(),
        ));
    }
    let unit = run.start;
    let segment: Segment = parts.decoders.segment(frames, canvas, unit)?;
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    let issue = job.on(Some(&stream));
    let latent = batch.read(owners.store, &call.inputs[0], call.call_id, issue)?;
    let row = u64::from(unit - range.cursor);
    let shape = vec![
        u64::from(range.max_units),
        u64::from(longest),
        u64::from(canvas.height),
        u64::from(canvas.width),
        3,
    ];
    let mut region: Vec<Range<u64>> = shape.iter().map(|&extent| 0..extent).collect();
    region[0] = row..row + 1;
    let write = bind_outputs(
        owners.store,
        &[OutputShape {
            product: call.outputs[0].clone(),
            shape,
            region: Some(region),
        }],
        batch.target(),
        placement,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| Error::Invariant("a decoded unit row was not bound".into()))?;
    batch.batch.writes.push(write.clone());
    let view = write.view();
    let out = view.select(0, 0)?;
    let decoder = &mut parts
        .decoders
        .video
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no video decoder".into()))?
        .runner;
    let written = units[unit as usize];
    let mode = decoder.decode_unit(
        slot,
        latent.value(),
        frames,
        unit,
        UnitShape {
            segment,
            first: unit == 0,
            last: unit + 1 == total,
            count: range.max_units,
        },
        longest,
        written,
        &out,
        job,
    )?;
    batch.record(mode);
    let frame_bytes = u64::from(canvas.height) * u64::from(canvas.width) * 3;
    let padding = u64::from(longest.saturating_sub(written)) * frame_bytes;
    if padding > 0 {
        // A short unit's row keeps zero frames after its own for its reader.
        // SAFETY: the padding lies inside the unit's bound product row,
        // written on the decoder's stream after the reconstruction.
        unsafe {
            issue.fill(Fill {
                address: out.address() + u64::from(written) * frame_bytes,
                value: 0,
                width: 1,
                count: padding as usize,
            })?;
        }
    }
    publish_bound(parts, batch, call, index, &write, true, issue)
}

/// Decode this rank's audio units into its span of the track.
pub(crate) fn decode_audio(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
    job: &Job,
) -> Result<()> {
    let call = &plan.calls[index];
    let range = decode_range(plan, call)?;
    let samples = parts.video.request(slot, call.request_key)?.audio_samples;
    if call.inputs.len() != 1 || call.outputs.len() != 1 {
        return Err(Error::Invalid(
            "media reconstruction requires one tensor input and output".into(),
        ));
    }
    let rank = parts.rank;
    let audio = parts
        .decoders
        .audio
        .as_ref()
        .ok_or_else(|| Error::Invalid("this rank holds no audio decoder".into()))?;
    let total = if audio.config.distribution.is_some() {
        (audio.config.ranks.len() * audio.config.units_per_rank.max(1)) as u32
    } else {
        1
    };
    let units = range.max_units.min(total).max(1);
    let run = audio.units(rank, 0, units)?;
    if run.is_empty() {
        return Ok(());
    }
    let (latent_frames, spans) = parts.decoders.track(samples, units)?;
    let first = spans[run.start as usize].start;
    let last = spans[run.end as usize - 1].end;
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    let issue = job.on(Some(&stream));
    let latent = batch.read(owners.store, &call.inputs[0], call.call_id, issue)?;
    let write = bind_outputs(
        owners.store,
        &[OutputShape {
            product: call.outputs[0].clone(),
            shape: vec![samples, 2],
            region: Some(vec![first..last, 0..2]),
        }],
        batch.target(),
        placement,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| Error::Invariant("an audio span was not bound".into()))?;
    batch.batch.writes.push(write.clone());
    let audio = &mut parts
        .decoders
        .audio
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no audio decoder".into()))?
        .runner;
    let mode = audio.decode_track(
        latent.value(),
        TrackUnits {
            samples,
            latent_frames,
            first: run.start,
            count: run.len() as u32,
            total: units,
            span: last - first,
        },
        &write.view(),
        job,
    )?;
    batch.record(mode);
    publish_bound(parts, batch, call, index, &write, true, issue)
}
