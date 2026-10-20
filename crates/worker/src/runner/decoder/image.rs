//! Image decoding: the image decoder of a bound component and the image batches
//! it serves. A batch decodes in consecutive sub-batches ([`Chunk`]) of the
//! images whose pixels fit the decoder's staging bytes, images of one size
//! together; each decoded image is optionally published as an image product and
//! quantized into staging, from which a host lane encodes it.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{Batch as BatchPlan, BufferId, ForwardStats};

use super::{DECODER, DecodeImage};
use crate::config::Device;
use crate::executor::image::ImageParts;
use crate::executor::image::{Encoding, ImageBatch, allocations};
use crate::ffi::{DType, Function, Handle, Interpreter};
use crate::host::HostTask;
use crate::host::image::{ImageTask, ImageWork};
use crate::lane::Job;
use crate::lock;
use crate::runner::ImageSetup;
use crate::runner::methods::{Argument, MethodCall};
use crate::storage::memory::{Storage, StorageView};
use crate::storage::{
    LatentUpdate, Metadata, OutputPlacement, ProductBuffer, TrajectoryVersion, ValueRange,
};
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
    pub(super) kernels: Function,
    pub(super) close: Function,
    /// The decoder's pixel range, `(low, high)`.
    pub(super) range: (f32, f32),
    staging: StorageView,
}

impl ImageDecoder {
    pub(super) fn new(
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
            Device::Cuda(index) => Storage::device(index, staging_bytes.max(1))?,
            Device::Cpu => Storage::host(staging_bytes.max(1)),
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
    /// Every sub-batch is one eager Python call, because a graph would
    /// remove too little to pay for itself. Profiled on a GB200 with the
    /// image milestone's requests, BAGEL's 512x512 decode keeps the device
    /// busy for 21.7 ms of a 23.8 ms span, and the host issues it in about
    /// 10 ms, ahead of the device: a replay could remove only the 2.0 to
    /// 2.6 ms of idle time inside the span, 1% of a two-step request
    /// (241 ms) and 0.05% of a 50-step one (4.9 s). SenseNova's decode is a
    /// call of under 1.1 ms issuing eleven small operations. A graph per
    /// (size, count) would also need fixed latent rows, a runner-owned
    /// buffer of decoded values with native copies to the product regions,
    /// and a bound on the activations each captured size retains.
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
    pub(super) fn quantize(
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
