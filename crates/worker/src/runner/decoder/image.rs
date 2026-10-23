//! Image decoding: the image decoder of a bound component and the image batches
//! it serves. Under CUDA graphs each image replays the graph captured at its
//! size: its latent rows are copied into the decoder's fixed inputs, the
//! graph decodes them into the decoder's fixed values and the head of the
//! staging bytes, and the values are copied into the image product. Without
//! graphs a batch decodes in consecutive sub-batches ([`Chunk`]) of the
//! images whose pixels fit the decoder's staging bytes, images of one size
//! together, each one eager call. Either way each decoded image is
//! optionally published as an image product and quantized into staging,
//! from which a host lane encodes it.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use cuda::{CopyBatch, DeviceId, SourceOrder, Stream};
use ipc::{Batch as BatchPlan, BufferId, ForwardStats};

use super::{DECODER, DecodeImage};
use crate::config::Device;
use crate::cuda_graph::{GraphMode, GraphPool, GraphTable, Lookup};
use crate::executor::image::ImageParts;
use crate::executor::image::{Encoding, ImageBatch, allocations};
use crate::ffi::{DType, Function, Handle, Interpreter};
use crate::host::HostTask;
use crate::host::image::{ImageTask, ImageWork};
use crate::lane::{Capture, Completion, Job};
use crate::lock;
use crate::memory::allocation::{Storage, StorageView};
use crate::memory::{
    LatentUpdate, Metadata, OutputPlacement, ProductBuffer, TrajectoryVersion, ValueRange,
};
use crate::parallel::joins::GraphFunctions;
use crate::runner::ImageSetup;
use crate::runner::methods::{Argument, MethodCall};
use crate::stream::Issue;
use crate::{Error, Result};

/// An image decoder of a bound component.
pub(super) struct ImageDecoder {
    owner: String,
    pub(super) device: Device,
    pub(super) stream: Option<Stream>,
    /// The Python half and its methods, each called with it first.
    pub(super) runner: Handle,
    pub(super) decode: Function,
    quantize: Function,
    decode_one: Function,
    quantize_one: Function,
    pub(super) kernels: Function,
    pub(super) close: Function,
    /// The decoder's pixel range, `(low, high)`.
    pub(super) range: (f32, f32),
    staging: StorageView,
    /// The latent pool's step workspace, which a batch gathers each
    /// trajectory into.
    latents: StorageView,
    /// `[units, width]` latent rows of one image, read by every captured
    /// decode.
    inputs: StorageView,
    /// Flat BF16 values of one image, as an image product holds them:
    /// written by every captured decode, read by every captured quantize.
    values: StorageView,
    pub(super) graphs: GraphTable<OneImage>,
    capture: GraphFunctions,
    /// The graph pools captures allocate in, the decoder's device's first
    /// and a generation tower's after it; empty without graphs, which
    /// decodes every image eagerly.
    pools: Vec<Arc<GraphPool>>,
}

/// The graph key of one image's decode or quantization at its size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct OneImage {
    /// A resident image product quantized; otherwise latent rows decoded.
    quantize: bool,
    height: u32,
    width: u32,
}

impl ImageDecoder {
    /// Under graphs ([`Self::set_pools`]) the decoder decodes images of at
    /// most `input_units` latent rows and `values` elements.
    pub(super) fn new(
        setup: &ImageSetup<'_>,
        range: (f32, f32),
        latents: StorageView,
        staging_bytes: usize,
        input_units: u32,
        values: usize,
    ) -> Result<Self> {
        let owner = setup.owner();
        let (device, stream) = (setup.device, setup.stream.clone());
        if device.is_cuda() != stream.is_some() {
            return Err(Error::Invariant(
                "an image decoder runs on a stream exactly on a CUDA device".into(),
            ));
        }
        let memory: std::sync::Arc<dyn crate::ffi::Backing> = match device {
            Device::Cuda(index) => Storage::device(index, staging_bytes.max(1))?,
            Device::Cpu => Storage::host(staging_bytes.max(1)),
        };
        let staging = StorageView::contiguous(memory, 0, DType::U8, &[staging_bytes as i64])?;
        let buffer = |dtype: DType, shape: &[i64]| -> Result<StorageView> {
            let bytes = shape.iter().product::<i64>().max(1) as usize * dtype.bytes();
            let memory: std::sync::Arc<dyn crate::ffi::Backing> = match device {
                Device::Cuda(index) => Storage::device(index, bytes)?,
                Device::Cpu => Storage::host(bytes),
            };
            StorageView::contiguous(memory, 0, dtype, shape)
        };
        let width = latents.shape().get(1).copied().unwrap_or(1);
        let inputs = buffer(latents.dtype(), &[i64::from(input_units.max(1)), width])?;
        let values = buffer(DType::BF16, &[values.max(1) as i64])?;
        let method = |name: &str| Interpreter::import(&format!("{DECODER}.{name}"));
        let mut arguments = setup.arguments();
        arguments.push(latents.handle()?.to_any());
        arguments.push(staging.handle()?.to_any());
        arguments.push(inputs.handle()?.to_any());
        arguments.push(values.handle()?.to_any());
        let runner = Handle::new(method("image")?.call_values(&arguments)?)?;
        let capture = GraphFunctions::import()?;
        Ok(Self {
            graphs: GraphTable::new(format!("{owner}.decoder"), capture.release.clone()),
            capture,
            pools: Vec::new(),
            owner,
            device,
            stream,
            runner,
            decode: method("decode")?,
            quantize: method("quantize")?,
            decode_one: method("decode_one")?,
            quantize_one: method("quantize_one")?,
            kernels: method("kernels")?,
            close: method("close")?,
            range,
            staging,
            latents,
            inputs,
            values,
        })
    }

    /// Whether the decoder serves under CUDA graphs: every image at a
    /// captured size, none eagerly.
    pub(super) fn graphs(&self) -> bool {
        !self.pools.is_empty()
    }

    /// Serve under CUDA graphs, capturing into `pools`: the decoder's
    /// device's, and a generation tower's.
    pub(super) fn set_pools(&mut self, pools: Vec<Arc<GraphPool>>) {
        self.pools = pools;
    }

    /// The graph pool's identity: the address of the decoder's staging
    /// bytes, which no other runner shares.
    pub(super) fn graph_key(&self) -> usize {
        self.staging.address() as usize
    }

    /// Record in `job` the call of `key`'s method on the decoder's fixed
    /// buffers: the launch of its captured graph; with `capture`, the
    /// capture of one not captured yet (startup), the job's completion
    /// carrying the graph ([`Self::complete_capture`]); without graphs and
    /// while startup warms a size, the eager call.
    fn one(&self, key: OneImage, units: u32, capture: bool, job: &Job) -> Result<GraphMode> {
        let runner = Argument::Handle(self.runner.clone());
        let method = if key.quantize {
            MethodCall::new(
                &self.quantize_one,
                vec![
                    runner,
                    Argument::Integer(i64::from(key.height)),
                    Argument::Integer(i64::from(key.width)),
                    Argument::Float(f64::from(self.range.0)),
                    Argument::Float(f64::from(self.range.1)),
                ],
            )
        } else {
            MethodCall::new(
                &self.decode_one,
                vec![
                    runner,
                    Argument::Integer(i64::from(units)),
                    Argument::Integer(i64::from(key.height)),
                    Argument::Integer(i64::from(key.width)),
                ],
            )
        };
        Ok(match self.graphs.lookup(&key, None) {
            Ok(Lookup::Replay(graph)) => {
                job.on(self.stream.as_ref()).launch(graph)?;
                GraphMode::Replay
            }
            Ok(Lookup::Capture) if capture => {
                let (Some(stream), Some(ordinal)) = (&self.stream, self.device.cuda()) else {
                    return Err(Error::Invariant(
                        "an image decoder captures on its CUDA stream".into(),
                    ));
                };
                let pool = self
                    .pools
                    .iter()
                    .find(|pool| pool.device == ordinal)
                    .ok_or_else(|| {
                        Error::Invariant("an image decoder has no graph pool on its device".into())
                    })?;
                job.capture(Capture {
                    functions: self.capture.clone(),
                    pool: pool.value.clone(),
                    peers: self
                        .pools
                        .iter()
                        .filter(|pool| pool.device != ordinal)
                        .map(|pool| (pool.device, pool.value.clone()))
                        .collect(),
                    stream: stream.raw(),
                    device: ordinal,
                    method,
                    breaks: None,
                });
                GraphMode::Capture
            }
            Err(error) if self.graphs() => return Err(error),
            _ => {
                job.call_returning(method);
                GraphMode::Eager
            }
        })
    }

    /// Record in `job` the warmup and then the capture of the decode of a
    /// `height` x `width` image of `units` latent rows and of the
    /// quantization of a resident image of that size; whether it records a
    /// capture. The warmup's staged byte counts join `returns`.
    pub(super) fn capture_size(
        &self,
        units: u32,
        height: u32,
        width: u32,
        job: &Job,
        returns: &mut Vec<Vec<i64>>,
    ) -> Result<Vec<OneImage>> {
        let issue = job.on(self.stream.as_ref());
        // SAFETY: the decoder's own buffers, which only its calls read,
        // recorded behind the fills on its stream.
        unsafe {
            self.inputs.zero(issue)?;
            self.values.zero(issue)?;
        }
        let mut captured = Vec::new();
        for quantize in [false, true] {
            let key = OneImage {
                quantize,
                height,
                width,
            };
            if self.graphs.contains(&key, None) {
                continue;
            }
            // The eager call warms the kernels the capture records.
            self.one(key, units, false, job)?;
            returns.push(vec![i64::from(height) * i64::from(width) * 3]);
            if self.one(key, units, true, job)? == GraphMode::Capture {
                captured.push(key);
            }
        }
        Ok(captured)
    }

    /// Store the graph a capture of [`Self::capture_size`] took for `key`.
    pub(super) fn complete_capture(
        &mut self,
        key: OneImage,
        completion: &mut Completion,
    ) -> Result<()> {
        self.graphs.insert(key, None, completion.captured()?)
    }

    /// Seal the captured graphs: a size startup did not capture is refused.
    pub(super) fn seal(&mut self) {
        self.graphs.seal();
    }

    /// The eager calls that decode `images` without graphs: each decodes
    /// the longest prefix of the remaining images that fits the staging
    /// bytes.
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
    pub(super) fn record_decode(
        &self,
        images: &[DecodeImage],
        mut stage: impl FnMut(usize, StorageView) -> Result<()>,
        job: &Job,
        returns: &mut Vec<Vec<i64>>,
    ) -> Result<()> {
        if self.graphs() {
            return self.record_captured(images, stage, job);
        }
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

    /// Record the decoding of `images` one at a time under graphs: each
    /// image's latent rows copied into the inputs, the replay of the graph
    /// of its size, its values copied into its product, and what `stage`
    /// issues for its staged pixels, which the next image overwrites.
    fn record_captured(
        &self,
        images: &[DecodeImage],
        mut stage: impl FnMut(usize, StorageView) -> Result<()>,
        job: &Job,
    ) -> Result<()> {
        let issue = job.on(self.stream.as_ref());
        let workspace = self.inputs.shape()[1] as usize * self.inputs.dtype().bytes();
        for (index, image) in images.iter().enumerate() {
            if i64::from(image.units) > self.inputs.shape()[0] {
                return Err(Error::Invalid(format!(
                    "{}: a {}x{} image of {} latent rows exceeds the captured sizes",
                    self.owner, image.height, image.width, image.units
                )));
            }
            let rows = self.latents_of(image)?;
            let mut input = CopyBatch::new(SourceOrder::Stream);
            input.push(
                self.inputs.address(),
                rows,
                image.units as usize * workspace,
            );
            // SAFETY: the rows lie in the latent pool's step workspace, which
            // the batch gathered them into on this stream, and the inputs
            // hold them; only this decoder's calls read the inputs, in
            // stream order.
            unsafe { issue.copy(&input) }?;
            let key = OneImage {
                quantize: false,
                height: image.height,
                width: image.width,
            };
            self.one(key, image.units, false, job)?;
            let elements = image.pixel_bytes();
            if let Some(product) = &image.product {
                if product.dtype() != DType::BF16 || product.numel() < elements as u64 {
                    return Err(Error::Invalid(
                        "an image product holds fewer BF16 values than its image".into(),
                    ));
                }
                let mut output = CopyBatch::new(SourceOrder::Stream);
                output.push(product.address(), self.values.address(), elements * 2);
                // SAFETY: the product region holds the image's values, which
                // the graph wrote into the decoder's values before this copy
                // on the same stream.
                unsafe { issue.copy(&output) }?;
            }
            stage(index, self.staging.narrow(0, 0, elements as i64)?)?;
        }
        Ok(())
    }

    /// The address of `image`'s latent rows in the step workspace.
    fn latents_of(&self, image: &DecodeImage) -> Result<u64> {
        if i64::from(image.sample) + i64::from(image.units) > self.latents.shape()[0] {
            return Err(Error::Invalid(
                "an image's latent rows exceed the step workspace".into(),
            ));
        }
        let row = self.latents.shape()[1] as u64 * self.latents.dtype().bytes() as u64;
        Ok(self.latents.address() + u64::from(image.sample) * row)
    }

    /// Record the quantization of a resident image product holding values
    /// in `range` into `job`, then what `stage` issues for its staged
    /// pixels (see [`Self::record_decode`]).
    pub(super) fn quantize(
        &self,
        image: &StorageView,
        range: (f32, f32),
        stage: impl FnOnce(StorageView) -> Result<()>,
        job: &Job,
        returns: &mut Vec<Vec<i64>>,
    ) -> Result<()> {
        if self.graphs() {
            // A resident image of this decoder's range replays the
            // quantization captured at its size from the values.
            let shape = image.shape();
            if shape.len() != 3
                || range != self.range
                || image.dtype() != DType::BF16
                || !image.is_contiguous()
            {
                return Err(Error::Invalid(format!(
                    "{}: under CUDA graphs a resident image is a [3, height, width] BF16 \
                     product of the decoder's value range",
                    self.owner
                )));
            }
            let (height, width) = (shape[1] as u32, shape[2] as u32);
            let elements = image.numel() as usize;
            if elements > self.values.numel() as usize {
                return Err(Error::Invalid(format!(
                    "{}: a {height}x{width} resident image exceeds the captured sizes",
                    self.owner
                )));
            }
            let issue = job.on(self.stream.as_ref());
            let mut input = CopyBatch::new(SourceOrder::Stream);
            input.push(self.values.address(), image.address(), elements * 2);
            // SAFETY: the product holds the image's values and the decoder's
            // values hold them; only this decoder's calls read the values,
            // in stream order.
            unsafe { issue.copy(&input) }?;
            let key = OneImage {
                quantize: true,
                height,
                width,
            };
            self.one(key, 0, false, job)?;
            return stage(self.staging.narrow(0, 0, elements as i64)?);
        }
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
) -> Result<(Option<DeviceId>, Option<Stream>, usize)> {
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
        runner.device().cuda(),
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
        let product = |reference: &ipc::TensorRef| {
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
                // The slot's committed step index, the latent pool's ledger,
                // is the end of the admitted schedule once denoising
                // completed.
                let slot = self.batch.calls[index].slot;
                if self.parts.latents()?.pool.step_index(slot) != i64::from(steps) {
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
                let step_index = latents.pool.step_index(slot);
                let trajectory = TrajectoryVersion {
                    step_index,
                    generation,
                    units: i64::from(params.latent_units),
                    height: i64::from(params.height),
                    width: i64::from(params.width),
                };
                let bank = latents
                    .pool
                    .current_bank(slot, &trajectory, &scratch.pages)?;
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
                self.updates.push((
                    *index,
                    LatentUpdate::for_call(slot, call, params, step_index)?,
                ));
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
        let graphs = runner.image_graphs()?;
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
        let mode = if graphs { "replay" } else { "eager" };
        *stats.mode_counts.entry(mode.into()).or_default() += 1;
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
