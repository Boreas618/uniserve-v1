//! Image decoding batches (`ImageDecoding`) and their PNG outputs.
//!
//! A call decodes its completed trajectory (gathered from the committed
//! bank into the step workspace) or, without a trajectory, re-encodes a
//! resident image product. The decoder optionally publishes the decoded
//! values as the call's image product (feedback) and always stages the
//! quantized HWC pixels, which are copied into the batch's pinned output
//! storage on the decoder's stream. Once those copies complete, the host
//! lane encodes each image as a base64 PNG ([`super::host`]); the result
//! is the call's media output. Decoding releases the trajectory.

use std::collections::HashMap;
use std::sync::Arc;

use uniserve_cuda::Stream;
use uniserve_ipc::{Batch as BatchPlan, BufferId, ForwardStats};

use super::ImageRuntime;
use super::batch::{Encoding, ImageBatch, allocations, lock};
use super::host::{ImageTask, ImageWork};
use crate::ffi::Region;
use crate::host::HostTask;
use crate::runner::decoder::DecodeImage;
use crate::storage::memory::StorageView;
use crate::storage::{LatentUpdate, Metadata, OutputPlacement, ProductBuffer, ValueRange};
use crate::{Error, Result};

/// The decoder serving a batch, with its device and stream, and the
/// output words its pixels need.
pub(super) fn placement(
    runtime: &ImageRuntime,
    plan: &BatchPlan,
) -> Result<(Option<i32>, Option<Stream>, usize)> {
    let runner = runtime
        .decoder()
        .ok_or_else(|| Error::Unsupported("image decoding is served by an image decoder".into()))?;
    let stream = runner
        .stream()
        // SAFETY: the lanes own the stream for the runtime's lifetime.
        .map(|stream| unsafe { Stream::borrowed(stream.device(), stream.raw()) })
        .transpose()?;
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
pub(super) fn value_range(range: (f32, f32)) -> Option<ValueRange> {
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
    pub(super) fn decode(&mut self) -> Result<ForwardStats> {
        let plan = self.plan;
        let active = self.active();
        if active.is_empty() {
            return Ok(ForwardStats::default());
        }
        let target = self.target();
        let stream = self.borrow_stream()?;
        let decoder_range = self
            .runtime
            .decoder()
            .ok_or_else(|| Error::Unsupported("image decoding requires a decoder".into()))?
            .range();

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
