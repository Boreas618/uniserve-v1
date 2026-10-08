//! Vision and latent encoding batches (`VisionEncoding`, `LatentEncoding`).
//!
//! Each active call encodes one image into its `encoder_output` feature
//! product: an inline request image the host lane prepared before the
//! batch launched, or a resident image product (feedback of a generated
//! image), which the encoder prepares on its stream. Features are written
//! in place into the product's reserved storage and published with the
//! image's canvas dimensions (`Metadata::Feature`).

use std::collections::HashMap;

use uniserve_cuda::Stream;
use uniserve_ipc::{Batch as BatchPlan, CallKind, ForwardStats, MediaCall};

use super::ImageRuntime;
use super::batch::{ImageBatch, allocations};
use crate::ffi::Region;
use crate::runner::encoder::{PreparedImage, Tower};
use crate::storage::{Metadata, OutputPlacement, ValueRange};
use crate::{Error, Result};

/// The tower an encoding kind runs.
pub(super) fn tower(call: MediaCall) -> Result<Tower> {
    match call {
        MediaCall::VisionEncoding => Ok(Tower::Vit),
        MediaCall::LatentEncoding => Ok(Tower::Vae),
        other => Err(Error::Invalid(format!(
            "{} is not an image encoding",
            other.as_str()
        ))),
    }
}

/// The encoder serving a batch, with its device and stream.
pub(super) fn placement(
    runtime: &ImageRuntime,
    plan: &BatchPlan,
) -> Result<(Option<i32>, Option<Stream>, usize)> {
    let first = plan
        .calls
        .first()
        .ok_or_else(|| Error::Invariant("an image batch has calls".into()))?;
    let runner = &runtime.encoders[runtime.encoder_route(&first.component, first.code)?];
    let stream = runner
        .stream()
        // SAFETY: the lanes own the stream for the runtime's lifetime.
        .map(|stream| unsafe { Stream::borrowed(stream.device(), stream.raw()) })
        .transpose()?;
    Ok((runner.device().cuda_index(), stream, 0))
}

impl ImageBatch<'_> {
    /// Encode every active call's image into its feature product.
    /// `prepared` holds the host-prepared inline images by call index.
    pub(super) fn encode(
        &mut self,
        call: MediaCall,
        mut prepared: HashMap<usize, PreparedImage>,
    ) -> Result<ForwardStats> {
        let tower = tower(call)?;
        let plan = self.plan;
        let active = self.active();
        let Some(&first) = active.first() else {
            return Ok(ForwardStats::default());
        };
        let route = self
            .runtime
            .encoder_route(&plan.calls[first].component, CallKind::Media(call))?;
        let target = self.target();

        // Every image, inline or resident, prepared for the encoder.
        let mut images = Vec::with_capacity(active.len());
        for &index in &active {
            let call = &plan.calls[index];
            if call.encoder_output.is_none() {
                return Err(Error::Invalid(
                    "image encoding requires an encoder output".into(),
                ));
            }
            let image = match (call.input_image.is_some(), &call.image_input) {
                (true, None) => prepared
                    .remove(&index)
                    .ok_or_else(|| Error::Invariant("an inline image was not prepared".into()))?,
                (false, Some(reference)) => {
                    let read = self.storage.tensors.consume(
                        reference,
                        call.call_id,
                        Some(target),
                        self.stream.as_ref(),
                    )?;
                    let range = match read.metadata() {
                        Some(Metadata::Image {
                            value_range: Some(range),
                            ..
                        }) => range,
                        _ => {
                            self.reads.push(read);
                            return Err(Error::Invalid(
                                "resident image encoding requires an image value range".into(),
                            ));
                        }
                    };
                    let runner = &self.runtime.encoders[route];
                    let region = Region::within(runner.products(), read.value());
                    self.reads.push(read);
                    runner.prepare_resident(&region?, tower, range == ValueRange::SignedUnit)?
                }
                _ => {
                    return Err(Error::Invalid(
                        "image encoding requires exactly one inline or resident image".into(),
                    ));
                }
            };
            images.push(image);
        }

        // The feature products, reserved in their scheduler allocations.
        let allocations = allocations(plan);
        let bindings: Vec<_> = active
            .iter()
            .filter_map(|&index| plan.calls[index].encoder_output.clone())
            .map(|reference| (reference, target))
            .collect();
        let writes = self.storage.tensors.reserve_features(
            &bindings,
            &allocations,
            OutputPlacement::default(),
        )?;
        self.writes.extend(writes.iter().cloned());
        let runner = &mut self.runtime.encoders[route];
        let regions = writes
            .iter()
            .map(|write| Region::within(runner.products(), &write.view()))
            .collect::<Result<Vec<_>>>()?;
        let borrowed: Vec<&PreparedImage> = images.iter().collect();
        let encoded = runner.encode(&borrowed, &regions)?;

        for ((write, image), shape) in writes.iter().zip(&images).zip(&encoded.shapes) {
            let shape: Vec<u64> = shape.iter().map(|&extent| extent as u64).collect();
            self.storage.tensors.produced(
                write,
                &shape,
                self.stream.as_ref(),
                Some(Metadata::Feature {
                    height: image.height,
                    width: image.width,
                }),
            )?;
        }

        let mut stats = ForwardStats::default();
        stats.cuda_graph_replays += encoded.replays;
        if encoded.replays > 0 {
            *stats.mode_counts.entry("graph_replay".into()).or_default() += encoded.replays;
        }
        if encoded.eager > 0 {
            *stats.mode_counts.entry("eager".into()).or_default() += encoded.eager;
        }
        Ok(stats)
    }
}
