//! Decoders: completed latents to delivered pixels and samples.
//!
//! A [`DecoderRunner`] is the Rust half of the Python
//! `uniserve_models.runner.decoder.DecoderRunner` for one decoder of a
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
use uniserve_cuda::{Stream, memcpy_async, memset_d8};
use uniserve_ipc::{Batch as BatchPlan, BufferId, ForwardStats};

use crate::config::Device;
use crate::cuda_graph::{GraphMode, GraphPool, GraphTable, Lookup};
use crate::executor::forward::batch::MediaBatch;
use crate::executor::forward::batch::{Parts, decode_range, publish_bound, stream_of};
use crate::executor::forward::image::ImageRuntime;
use crate::executor::forward::image::batch::{Encoding, ImageBatch, allocations, lock};
use crate::executor::forward::image::host::{ImageTask, ImageWork};
use crate::executor::forward::placement::{OutputShape, Placement, bind_outputs};
use crate::ffi::{DType, Function, Handle, Record, Region, integer_list, list};
use crate::host::HostTask;
use crate::model::bind::ComponentBinding;
use crate::model::interface::Interface;
use crate::parallel::joins::GraphFunctions;
use crate::runner::{Methods, RunnerSetup, device_buffer};
use crate::storage::memory::{Memory, StorageView};
use crate::storage::{
    LatentUpdate, Metadata, OutputPlacement, ProductBuffer, RequestSlots, SlotField, ValueRange,
};
use crate::{Error, Result};

const DECODER: &str = "uniserve_models.runner.decoder.DecoderRunner";

/// One image to decode.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeImage {
    /// The image's first latent row in the step workspace and its rows.
    pub sample: u32,
    pub units: u32,
    pub height: u32,
    pub width: u32,
    /// The image product receiving the decoded values, if published.
    pub product: Option<Region>,
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
    /// Bind the image decoder at `path` of `binding` over the latent pool's
    /// step workspace `latents`, the device's product arena and
    /// `staging_bytes` of staging. `stream` borrows the binding's stream.
    #[expect(
        clippy::too_many_arguments,
        reason = "the binding's complete resources"
    )]
    pub fn bind(
        interface: &Interface,
        binding: &ComponentBinding,
        path: &str,
        range: (f32, f32),
        latents: StorageView,
        products: StorageView,
        staging_bytes: usize,
        stream: Option<Stream>,
    ) -> Result<Self> {
        let image = ImageDecoder::bind(
            interface,
            binding,
            path,
            range,
            latents,
            products,
            staging_bytes,
            stream,
        )?;
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

    fn image(&self) -> Result<&ImageDecoder> {
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
            Family::Video(video) => Some(&video.methods.stream),
            Family::Audio(audio) => Some(&audio.methods.stream),
        }
    }

    /// An image decoder's pixel range, `(low, high)`.
    pub fn range(&self) -> Result<(f32, f32)> {
        Ok(self.image()?.range)
    }

    /// Decode `images`, handing each image's staged pixels (in input
    /// order) to `stage`, which must enqueue their consumption on the
    /// runner's stream before it returns: the next sub-batch reuses the
    /// staging bytes.
    pub fn decode(
        &self,
        images: &[DecodeImage],
        stage: impl FnMut(usize, StorageView) -> Result<()>,
    ) -> Result<()> {
        self.image()?.decode(images, stage)
    }

    /// Quantize a resident image product holding values in `range` and
    /// hand its staged pixels to `stage`.
    pub fn quantize(
        &self,
        image: &Region,
        range: (f32, f32),
        stage: impl FnOnce(StorageView) -> Result<()>,
    ) -> Result<()> {
        self.image()?.quantize(image, range, stage)
    }

    /// Construct a video decoder's Python runner of every segment, the
    /// largest first.
    pub(crate) fn prepare_segments(&mut self, segments: &[Segment]) -> Result<()> {
        self.video_decoder()?.prepare(segments)
    }

    /// Decode and reconstruct a zero latent as a media unit of `shape`
    /// whose window reads `rows` rows, so kernels compile and the ring's
    /// exchange of the round shape is prepared. Every ring member holding
    /// a unit of the round calls it, in the same order.
    pub(crate) fn warm_unit(&mut self, shape: UnitShape, rows: u64, longest: u32) -> Result<()> {
        self.video_decoder()?.warm(shape, rows, longest)
    }

    /// Capture the forward of a warmed unit `shape`.
    pub(crate) fn capture_unit(&mut self, shape: UnitShape, rows: u64, longest: u32) -> Result<()> {
        self.video_decoder()?.capture(shape, rows, longest)
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
    ) -> Result<GraphMode> {
        self.video_decoder()?
            .decode(slot, latent, frames, unit, shape, longest, written, out)
    }

    /// Construct an audio decoder's Python runner of tracks of
    /// `latent_frames` latent frames.
    pub(crate) fn prepare_track(&mut self, latent_frames: u32) -> Result<()> {
        self.audio_decoder()?.prepare(latent_frames)
    }

    /// Decode `units` of a zero track whose complete latent has `rows`
    /// rows, so kernels compile before any capture.
    pub(crate) fn warm_track(&mut self, units: TrackUnits, rows: u64) -> Result<()> {
        self.audio_decoder()?.warm(units, rows)
    }

    /// Capture the forward of warmed track `units`.
    pub(crate) fn capture_track(&mut self, units: TrackUnits, rows: u64) -> Result<()> {
        self.audio_decoder()?.capture(units, rows)
    }

    /// Decode `units` of a track into `out` (`[span, 2]` int16). `latent`
    /// is the request's complete audio latent.
    pub(crate) fn decode_track(
        &mut self,
        latent: &StorageView,
        units: TrackUnits,
        out: &StorageView,
    ) -> Result<GraphMode> {
        self.audio_decoder()?.decode(latent, units, out)
    }

    /// Refuse further captures.
    pub(crate) fn seal(&mut self) {
        match &mut self.family {
            Family::Image(_) => {}
            Family::Video(video) => video.graphs.seal(),
            Family::Audio(audio) => audio.graphs.seal(),
        }
    }

    /// The kernels the operators of a video or audio decoder selected.
    pub(crate) fn kernels(&self, label: &str) -> Result<Vec<Record>> {
        match &self.family {
            Family::Image(_) => Ok(Vec::new()),
            Family::Video(video) => video.kernels(label),
            Family::Audio(audio) => audio.kernels(label),
        }
    }

    /// Release the graphs and Python runners of a video or audio decoder;
    /// an image decoder's component binding owns its resources.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        match &mut self.family {
            Family::Image(_) => Ok(()),
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
    /// The decoder's pixel range, `(low, high)`.
    range: (f32, f32),
    staging: StorageView,
}

impl ImageDecoder {
    #[expect(
        clippy::too_many_arguments,
        reason = "the binding's complete resources"
    )]
    fn bind(
        interface: &Interface,
        binding: &ComponentBinding,
        path: &str,
        range: (f32, f32),
        latents: StorageView,
        products: StorageView,
        staging_bytes: usize,
        stream: Option<Stream>,
    ) -> Result<Self> {
        let owner = if path.is_empty() {
            binding.component.clone()
        } else {
            format!("{}.{path}", binding.component)
        };
        let device = binding.device;
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
        let method = |name: &str| interface.import(&format!("{DECODER}.{name}"));
        let runner = Handle::new(method("bound")?.call_values(&[
            binding.prepared().to_any(),
            Any::from(tvm_ffi::String::from(path)),
            Any::from(latents.to_tensor()?),
            Any::from(staging.to_tensor()?),
            Any::from(products.to_tensor()?),
        ])?)?;
        Ok(Self {
            owner,
            device,
            stream,
            runner,
            decode: method("decode")?,
            quantize: method("quantize")?,
            range,
            staging,
        })
    }

    /// Decode `images`, handing each image's staged pixels (in input
    /// order) to `stage`, which must enqueue their consumption on the
    /// runner's stream before it returns: the next sub-batch reuses the
    /// staging bytes.
    ///
    /// FIXME: every sub-batch is one eager Python call. A graph per (size,
    /// count) needs fixed latent rows, a runner-owned buffer of decoded
    /// values with native copies to the product regions, and a bound on
    /// the activations each captured size retains; a decoder on a
    /// generation tower returns its pixels through a delivery stream.
    fn decode(
        &self,
        images: &[DecodeImage],
        mut stage: impl FnMut(usize, StorageView) -> Result<()>,
    ) -> Result<()> {
        let capacity = self.staging.numel() as usize;
        let mut first = 0;
        while first < images.len() {
            // The longest prefix of the remaining images that fits staging.
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
            let rows = integer_list(images[first..end].iter().flat_map(|image| {
                offsets.push(offset);
                let row = [
                    i64::from(image.sample),
                    i64::from(image.units),
                    i64::from(image.height),
                    i64::from(image.width),
                    offset as i64,
                ];
                offset += image.pixel_bytes();
                row
            }));
            let products = list(images[first..end].iter().map(|image| {
                image
                    .product
                    .as_ref()
                    .map_or_else(Any::default, Region::to_any)
            }));
            let counts = crate::ffi::integers(self.decode.call_values(&[
                self.runner.to_any(),
                Any::from(rows),
                Any::from(products),
            ])?)
            .map_err(|error| Error::numerical(self.decode.name(), &error))?;
            if counts.len() != end - first {
                return Err(Error::Interface(format!(
                    "{} returns one byte count per image",
                    self.decode.name()
                )));
            }
            for ((index, image), (&count, &offset)) in (first..end)
                .zip(&images[first..end])
                .zip(counts.iter().zip(&offsets))
            {
                if count as usize != image.pixel_bytes() {
                    return Err(Error::Interface(format!(
                        "{} staged {count} bytes of a {}x{} image",
                        self.decode.name(),
                        image.height,
                        image.width
                    )));
                }
                stage(index, self.staging.narrow(0, offset as i64, count)?)?;
            }
            first = end;
        }
        Ok(())
    }

    /// Quantize a resident image product holding values in `range` and
    /// hand its staged pixels to `stage` (see [`Self::decode`]).
    fn quantize(
        &self,
        image: &Region,
        range: (f32, f32),
        stage: impl FnOnce(StorageView) -> Result<()>,
    ) -> Result<()> {
        let count: i64 = self.quantize.call_as(&[
            self.runner.view(),
            (&image.to_any()).into(),
            (&Any::from(f64::from(range.0))).into(),
            (&Any::from(f64::from(range.1))).into(),
            (&Any::from(0_i64)).into(),
        ])?;
        if count <= 0 || count as u64 > self.staging.numel() {
            return Err(Error::Interface(format!(
                "{} staged {count} bytes",
                self.quantize.name()
            )));
        }
        stage(self.staging.narrow(0, 0, count)?)
    }
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
    runtime: &ImageRuntime,
    plan: &BatchPlan,
) -> Result<(Option<i32>, Option<Stream>, usize)> {
    let runner = runtime
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
        words.min(runtime.max_pixel_words),
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
    Resident(Region, (f32, f32), u32, u32),
}

impl ImageBatch<'_> {
    /// Decode every active call's image and stage its pixels for PNG
    /// encoding.
    pub(crate) fn decode(&mut self) -> Result<ForwardStats> {
        let plan = self.plan;
        let active = self.active();
        if active.is_empty() {
            return Ok(ForwardStats::default());
        }
        let target = self.target();
        let stream = self.stream.clone();
        let decoder_range = self
            .runtime
            .decoder()
            .ok_or_else(|| Error::Unsupported("image decoding requires a decoder".into()))?
            .range()?;

        // One encoding task per reporting call, reserved before any device
        // work so a full host lane fails the batch cleanly.
        let mut tasks: HashMap<usize, Arc<HostTask<ImageTask>>> = HashMap::new();
        if self.reports {
            for &index in &active {
                let task = self.runtime.host.reserve()?;
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
        self.writes.extend(products.iter().cloned());
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
            let read = self.storage.tensors.consume(
                reference,
                call.call_id,
                Some(target),
                stream.as_ref(),
            )?;
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
                    self.reads.push(read);
                    return Err(Error::Invalid(
                        "image encoding requires image product metadata".into(),
                    ));
                }
            };
            let arena = self.runtime.arena(target)?;
            let region = Region::within(arena, read.value());
            self.reads.push(read);
            sources.push(Some(Source::Resident(region?, range, height, width)));
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
            let latents = self.runtime.latents()?;
            latents.enter(stream.as_ref())?;
            self.latents_entered = true;
            let params: Vec<_> = trajectories.iter().map(|(_, params)| params).collect();
            let scratch = latents.place(&params)?;
            for ((index, params), scratch) in trajectories.iter().zip(&scratch) {
                let call = &plan.calls[*index];
                let slot = self.calls[*index].slot;
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
                latents.gather(bank, scratch, stream.as_ref())?;
                let product = call
                    .image_output
                    .as_ref()
                    .and_then(product)
                    .map(|write| Region::within(self.runtime.arena(target)?, &write.view()))
                    .transpose()?;
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

        // Decode, then copy each image's staged pixels to the host.
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
                let mut buffer = lock(&self.buffer);
                let host = buffer.capture_bytes(&pixels, stream.as_ref(), &self.storage.events)?;
                if self.reports {
                    buffer.retain_reader()?;
                }
                staged.push((index, host, height, width));
                Ok(())
            };
        let runner = self
            .runtime
            .decoder
            .as_ref()
            .map(|(_, runner)| runner)
            .ok_or_else(|| Error::Unsupported("image decoding requires a decoder".into()))?;
        for (&index, source) in active.iter().zip(&sources) {
            match source {
                Some(Source::Resident(region, range, height, width)) => {
                    runner.quantize(region, *range, |pixels| {
                        stage(index, pixels, *height, *width)
                    })?;
                }
                Some(Source::Latent(image)) => decoded.push((index, image.clone())),
                None => return Err(Error::Invariant("a decoding call has no source".into())),
            }
        }
        let images: Vec<DecodeImage> = decoded.iter().map(|(_, image)| image.clone()).collect();
        runner.decode(&images, |position, pixels| {
            let (index, image) = &decoded[position];
            stage(*index, pixels, image.height, image.width)
        })?;
        drop(stage);

        // Feedback products hold the decoded values.
        for (index, image) in &decoded {
            let call = &plan.calls[*index];
            if let Some(write) = call.image_output.as_ref().and_then(product) {
                self.storage.tensors.produced(
                    &write,
                    &[3, u64::from(image.height), u64::from(image.width)],
                    stream.as_ref(),
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
                lease: Arc::clone(&self.buffer),
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

impl ImageRuntime {
    /// The bound image decoder.
    pub(crate) fn decoder(&self) -> Option<&crate::runner::decoder::DecoderRunner> {
        self.decoder.as_ref().map(|(_, runner)| runner)
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
    methods: Methods,
    device: i32,
    /// The runner of each segment, in preparation order.
    runners: Vec<(Segment, Handle)>,
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
        let capture = GraphFunctions::from_interface(setup.interface)?;
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
            window_rows: setup.interface.import(&format!("{DECODER}.window_rows"))?,
            graphs: GraphTable::new(format!("{}.decoder", setup.path), capture.release.clone()),
            capture,
            pool,
            methods: Methods::new(&setup, DECODER, "decode")?,
        })
    }

    pub(crate) fn stream(&self) -> &Stream {
        &self.methods.stream
    }

    /// Construct the runner of every segment, the largest first.
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
            let runner = self.methods.runner([
                i64::from(frames),
                i64::from(canvas.width),
                i64::from(canvas.height),
            ])?;
            self.runners.push((segment, runner));
        }
        Ok(())
    }

    /// The arguments of `DecoderRunner.decode` for a unit of `shape` whose
    /// window reads `rows` latent rows and whose video's longest unit has
    /// `longest` frames.
    fn arguments(&self, shape: UnitShape, rows: u64, longest: u32) -> Result<Vec<Any>> {
        let canvas = shape.segment.1;
        let runner = self
            .runners
            .iter()
            .find(|(prepared, _)| *prepared == shape.segment)
            .map(|(_, runner)| runner)
            .ok_or_else(|| Error::Invariant("a video segment has no decoder runner".into()))?;
        let frames = self.unit_frames(canvas, longest)?;
        Ok(vec![
            runner.to_any(),
            Any::from(self.latent.to_tensor()?),
            Any::from(self.rows.prefix(rows as i64)?.to_tensor()?),
            Any::from(self.retained.to_tensor()?),
            Any::from(frames.to_tensor()?),
            Any::from(shape.first),
            Any::from(shape.last),
            Any::from(i64::from(shape.count)),
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

    /// The window rows of unit `unit` of a `frames`-frame video at
    /// `canvas`, as native int64 bytes.
    fn window(&mut self, frames: u32, canvas: Canvas, unit: u32) -> Result<&[u8]> {
        let key = (frames, canvas, unit);
        if !self.windows.contains_key(&key) {
            let value = self.window_rows.call_values(&[
                self.methods.loaded.to_any(),
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
        Ok(&self.windows[&key])
    }

    /// Decode and reconstruct a zero latent as a unit of `shape` whose
    /// window reads `rows` rows, so kernels compile and the ring's exchange
    /// of the round shape is prepared, then capture the forward when the
    /// runner captures. Every member holding a unit of the round calls it,
    /// in the same order. The padding slot's overlap serves as the retained
    /// overlap.
    pub(crate) fn warm(&mut self, shape: UnitShape, rows: u64, longest: u32) -> Result<()> {
        self.latent.zero(Some(self.stream()))?;
        self.rows.zero(Some(self.stream()))?;
        self.retained.zero(Some(self.stream()))?;
        let arguments = self.arguments(shape, rows, longest)?;
        self.methods.forward.call_values(&arguments)?;
        self.stream().synchronize()?;
        Ok(())
    }

    /// Capture the forward of a warmed `shape`.
    pub(crate) fn capture(&mut self, shape: UnitShape, rows: u64, longest: u32) -> Result<()> {
        let Some(pool) = &self.pool else {
            return Ok(());
        };
        if self.graphs.contains(&shape, None) {
            return Ok(());
        }
        let arguments = self.arguments(shape, rows, longest)?;
        let graph =
            self.capture
                .capture(&pool.value, self.methods.stream.raw(), self.device, || {
                    self.methods.forward.call_values(&arguments).map(drop)
                })?;
        self.graphs.insert(shape, None, graph)?;
        self.graphs.record(GraphMode::Capture);
        Ok(())
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
    ) -> Result<GraphMode> {
        let canvas = shape.segment.1;
        let stream = self.methods.stream.clone();
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
        self.rows
            .prefix(rows as i64)?
            .upload(&window, Some(&stream))?;
        // SAFETY: the forward's tensors and the slot's overlap row belong to
        // this runner and the copies stay inside them; the latent was read
        // on this stream.
        unsafe {
            memcpy_async(
                self.latent.address(),
                latent.address(),
                latent_bytes as usize,
                &stream,
            )?;
            memcpy_async(
                self.retained.address(),
                slot_overlap.address(),
                overlap_bytes as usize,
                &stream,
            )?;
        }

        let mode = match self.graphs.lookup(&shape, None) {
            Ok(Lookup::Replay(graph)) => {
                graph.launch(&stream)?;
                GraphMode::Replay
            }
            // A shape startup did not capture runs on the Python thread.
            _ => {
                let arguments = self.arguments(shape, rows, longest)?;
                self.methods.forward.call_values(&arguments)?;
                GraphMode::Eager
            }
        };
        self.graphs.record(mode);

        let frame_bytes = u64::from(canvas.height) * u64::from(canvas.width) * 3;
        // SAFETY: as above; `out` holds the unit's frames and the copies
        // follow the forward on the stream.
        unsafe {
            memcpy_async(
                slot_overlap.address(),
                self.retained.address(),
                overlap_bytes as usize,
                &stream,
            )?;
            memcpy_async(
                out.address(),
                self.frames.address(),
                (u64::from(written) * frame_bytes) as usize,
                &stream,
            )?;
        }
        Ok(mode)
    }

    pub(crate) fn kernels(&self, label: &str) -> Result<Vec<Record>> {
        self.methods
            .kernels(label, self.runners.iter().map(|(_, runner)| runner))
    }

    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let graphs = self.graphs.close(aborted);
        graphs.and(
            self.methods
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
    methods: Methods,
    device: i32,
    /// The runner of each track length, by its latent frames.
    runners: BTreeMap<u32, Handle>,
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
        let capture = GraphFunctions::from_interface(setup.interface)?;
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
            methods: Methods::new(&setup, DECODER, "decode")?,
        })
    }

    pub(crate) fn stream(&self) -> &Stream {
        &self.methods.stream
    }

    /// Construct the runner of tracks of `latent_frames` latent frames.
    pub(crate) fn prepare(&mut self, latent_frames: u32) -> Result<()> {
        if !self.runners.contains_key(&latent_frames) {
            let runner = self.methods.runner([i64::from(latent_frames)])?;
            self.runners.insert(latent_frames, runner);
        }
        Ok(())
    }

    /// The arguments of `DecoderRunner.decode` for `units` of a track whose
    /// complete latent has `rows` rows.
    fn arguments(&self, units: TrackUnits, rows: u64) -> Result<Vec<Any>> {
        let runner = self.runners.get(&units.latent_frames).ok_or_else(|| {
            Error::Invalid("the audio track's length has no decoder runner".into())
        })?;
        Ok(vec![
            runner.to_any(),
            Any::from(self.latent.prefix(rows as i64)?.to_tensor()?),
            Any::from(self.samples.prefix(units.span as i64)?.to_tensor()?),
            Any::from(units.samples as i64),
            Any::from(i64::from(units.first)),
            Any::from(i64::from(units.count)),
            Any::from(i64::from(units.total)),
        ])
    }

    /// Decode `units` of a zero track whose complete latent has `rows`
    /// rows, so kernels compile before any capture.
    pub(crate) fn warm(&mut self, units: TrackUnits, rows: u64) -> Result<()> {
        self.latent.zero(Some(self.stream()))?;
        let arguments = self.arguments(units, rows)?;
        self.methods.forward.call_values(&arguments)?;
        self.stream().synchronize()?;
        Ok(())
    }

    /// Capture the forward of warmed `units`.
    pub(crate) fn capture(&mut self, units: TrackUnits, rows: u64) -> Result<()> {
        let Some(pool) = &self.pool else {
            return Ok(());
        };
        if self.graphs.contains(&units, None) {
            return Ok(());
        }
        let arguments = self.arguments(units, rows)?;
        let graph =
            self.capture
                .capture(&pool.value, self.methods.stream.raw(), self.device, || {
                    self.methods.forward.call_values(&arguments).map(drop)
                })?;
        self.graphs.insert(units, None, graph)?;
        self.graphs.record(GraphMode::Capture);
        Ok(())
    }

    /// Decode `units` of a track into `out` (`[span, 2]` int16). `latent`
    /// is the request's complete audio latent.
    pub(crate) fn decode(
        &mut self,
        latent: &StorageView,
        units: TrackUnits,
        out: &StorageView,
    ) -> Result<GraphMode> {
        let stream = self.methods.stream.clone();
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
        // SAFETY: the forward's tensors belong to this runner and the copy
        // stays inside them; the latent was read on this stream.
        unsafe {
            memcpy_async(
                self.latent.address(),
                latent.address(),
                latent_bytes as usize,
                &stream,
            )?;
        }

        let mode = match self.graphs.lookup(&units, None) {
            Ok(Lookup::Replay(graph)) => {
                graph.launch(&stream)?;
                GraphMode::Replay
            }
            // Units startup did not capture run on the Python thread.
            _ => {
                let arguments = self.arguments(units, rows)?;
                self.methods.forward.call_values(&arguments)?;
                GraphMode::Eager
            }
        };
        self.graphs.record(mode);

        // SAFETY: as above; `out` holds the units' samples and the copy
        // follows the forward on the stream.
        unsafe {
            memcpy_async(
                out.address(),
                self.samples.address(),
                (units.span * 4) as usize,
                &stream,
            )?;
        }
        Ok(mode)
    }

    pub(crate) fn kernels(&self, label: &str) -> Result<Vec<Record>> {
        self.methods.kernels(label, self.runners.values())
    }

    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let graphs = self.graphs.close(aborted);
        graphs.and(self.methods.close(aborted, self.runners.values()))
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
    let segment: Segment = parts.video.model.segment(frames, canvas, unit)?;
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    let latent = batch.inputs.read(
        owners.store,
        &call.inputs[0],
        call.call_id,
        batch.device,
        Some(&stream),
    )?;
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
        batch.device,
        placement,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| Error::Invariant("a decoded unit row was not bound".into()))?;
    batch.calls[index].writes.push(write.clone());
    let view = write.view();
    let out = view.select(0, 0)?;
    let decoder = &mut parts
        .decoders
        .video
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no video decoder".into()))?
        .runner;
    let written = units[unit as usize];
    decoder.decode_unit(
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
    )?;
    let frame_bytes = u64::from(canvas.height) * u64::from(canvas.width) * 3;
    let padding = u64::from(longest.saturating_sub(written)) * frame_bytes;
    if padding > 0 {
        // A short unit's row keeps zero frames after its own for its reader.
        // SAFETY: the padding lies inside the unit's bound product row,
        // written on the decoder's stream after the reconstruction.
        unsafe {
            memset_d8(
                out.address() + u64::from(written) * frame_bytes,
                0,
                padding as usize,
                &stream,
            )?;
        }
    }
    publish_bound(parts, batch, call, index, &write, true)
}

/// Decode this rank's audio units into its span of the track.
pub(crate) fn decode_audio(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
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
    let spans = parts.video.model.audio_units(samples, units)?;
    let latent_frames = parts.video.model.audio_latent_frames(samples)?;
    let first = spans[run.start as usize].start;
    let last = spans[run.end as usize - 1].end;
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    let latent = batch.inputs.read(
        owners.store,
        &call.inputs[0],
        call.call_id,
        batch.device,
        Some(&stream),
    )?;
    let write = bind_outputs(
        owners.store,
        &[OutputShape {
            product: call.outputs[0].clone(),
            shape: vec![samples, 2],
            region: Some(vec![first..last, 0..2]),
        }],
        batch.device,
        placement,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| Error::Invariant("an audio span was not bound".into()))?;
    batch.calls[index].writes.push(write.clone());
    let audio = &mut parts
        .decoders
        .audio
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no audio decoder".into()))?
        .runner;
    audio.decode_track(
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
    )?;
    publish_bound(parts, batch, call, index, &write, true)
}
