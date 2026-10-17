//! Packed vision: images encoded in fixed slots of `max_patches` patch rows
//! through one shared pixel buffer, the slot grids ([`packed_grids`]) and a
//! feature buffer ([`PackedVision`]), with one captured graph per slot count.

use uniserve_cuda::{CopyBatch, SourceOrder};

use super::{
    Encoded, EncoderGraph, EncoderInfo, EncoderRunner, ImagePixels, MAX_PACKED_IMAGES,
    PreparedImage, Tower,
};
use crate::lane::Job;
use crate::runner::methods::Argument;
use crate::storage::memory::StorageView;
use crate::stream::Issue;
use crate::{Error, Result};

/// Lay out packed images in `slots` fixed slots of `max_patches` rows.
///
/// Each slot is described by two `[rows, columns]` grids: its image's patch
/// grid (`[0, 0]` for an empty slot), then a `downsample` rows high grid
/// covering the slot's remaining rows (`[0, 0]` when the image fills it).
/// Images take the leading slots in order. This is
/// `PatchEncoder.packed_grids`, which the captured encode reads from device
/// memory.
pub(crate) fn packed_grids(
    shapes: &[[u32; 2]],
    slots: usize,
    max_patches: u32,
    downsample: u32,
) -> Result<Vec<[i64; 2]>> {
    if shapes.len() > slots {
        return Err(Error::Invalid("packed images exceed their slots".into()));
    }
    let mut grids = Vec::with_capacity(2 * slots);
    for index in 0..slots {
        let shape = shapes.get(index).copied().unwrap_or([0, 0]);
        let rows = u64::from(shape[0]) * u64::from(shape[1]);
        if index < shapes.len()
            && (shape.contains(&0)
                || shape.iter().any(|side| side % downsample != 0)
                || rows > u64::from(max_patches))
        {
            return Err(Error::Invalid(
                "packed image grids must align with spatial downsampling within the slot capacity"
                    .into(),
            ));
        }
        let padding = u64::from(max_patches) - rows;
        grids.push([i64::from(shape[0]), i64::from(shape[1])]);
        grids.push(if padding == 0 {
            [0, 0]
        } else {
            [
                i64::from(downsample),
                (padding / u64::from(downsample)) as i64,
            ]
        });
    }
    Ok(grids)
}

/// Packed vision slots of one numerical call.
pub(super) fn packed_capacity(max_images: usize) -> usize {
    max_images.clamp(1, MAX_PACKED_IMAGES)
}

/// Packed vision buffers: `pixels [capacity * max_patches, row_width]`,
/// `grids [2 * capacity, 2]` int64 and `features [capacity * slot_tokens,
/// output_size]`, shared by every captured slot count (each uses the
/// leading rows).
pub(super) struct PackedVision {
    pub(super) capacity: usize,
    pub(super) pixels: StorageView,
    pub(super) grids: StorageView,
    pub(super) features: StorageView,
}

impl EncoderRunner {
    /// The arguments of the Python half's `encode` of `slots` packed slots.
    pub(super) fn packed_arguments(&self, slots: usize) -> Result<Vec<Argument>> {
        let (EncoderInfo::Vision(vision), Some(packed)) = (&self.info, &self.packed) else {
            return Err(Error::Invariant("encoder packs no images".into()));
        };
        let slots = slots as i64;
        let patches = i64::from(vision.max_patches.unwrap_or_default());
        let tokens = i64::from(vision.slot_tokens().unwrap_or_default());
        Ok(vec![
            Argument::Handle(self.methods.runner.clone()),
            Argument::Tensor(packed.pixels.prefix(slots * patches)?),
            Argument::Tensor(packed.grids.prefix(2 * slots)?),
            Argument::Tensor(packed.features.prefix(slots * tokens)?),
        ])
    }

    pub(super) fn encode_packed(
        &mut self,
        images: &[&PreparedImage],
        out: &[StorageView],
        result: &mut Encoded,
        job: &Job,
    ) -> Result<()> {
        let (EncoderInfo::Vision(vision), Some(packed)) = (&self.info, &self.packed) else {
            return Err(Error::Invariant("encoder packs no images".into()));
        };
        let vision = *vision;
        let (capacity, pixels, grids, features) = (
            packed.capacity,
            packed.pixels.clone(),
            packed.grids.clone(),
            packed.features.clone(),
        );
        let max_patches = vision.max_patches.unwrap_or_default();
        let slot_tokens = u64::from(vision.slot_tokens().unwrap_or_default());
        let row_bytes = u64::from(vision.output_size) * vision.output_dtype.bytes() as u64;
        let shapes = images
            .iter()
            .map(|image| {
                image
                    .facts
                    .grid
                    .filter(|_| image.facts.shape.len() == 2)
                    .ok_or_else(|| {
                        Error::Invalid(
                            "packed vision images require patch rows covering their grids".into(),
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let stream = self.stream.clone();
        let issue = job.on(stream.as_ref());
        let mut offset = 0;
        for chunk in shapes.chunks(capacity) {
            let slots = chunk.len();
            let layout = packed_grids(chunk, slots, max_patches, vision.downsample)?;
            // SAFETY: the encoder's grid input, kept until it closes after
            // its stream drained; the forwards reading it run on `issue`'s
            // stream, the earlier ones before this write and this chunk's
            // after it.
            unsafe {
                grids
                    .prefix(2 * slots as i64)?
                    .upload(grid_bytes(&layout).as_slice(), issue)
            }?;

            // Each image's prepared patch rows go to the front of its slot:
            // host-prepared rows are copied there, and a resident image is
            // prepared there. Rows past an image keep earlier finite values,
            // which only padding reads. A host job reads the copies'
            // sources as it records them.
            let order = if stream.is_some() {
                SourceOrder::Stream
            } else {
                SourceOrder::DuringCall
            };
            let mut rows_in = CopyBatch::new(order);
            for (slot, image) in images[offset..offset + slots].iter().enumerate() {
                let target = pixels.narrow(
                    0,
                    slot as i64 * i64::from(max_patches),
                    image.facts.shape[0],
                )?;
                match &image.pixels {
                    ImagePixels::Host(source) => {
                        rows_in.push(target.address(), source.address(), source.nbytes() as usize)
                    }
                    ImagePixels::Resident {
                        image: value,
                        signed_unit,
                    } => self.prepare_resident(value, Tower::Vit, *signed_unit, &target, job),
                }
            }
            // SAFETY: each source is host-prepared pixels of the slot's
            // shape, which the caller keeps until the copies completed (a
            // host job copies them now), and each target lies in the
            // runner's packed pixels.
            unsafe { issue.copy(&rows_in) }?;

            let arguments = self.packed_arguments(slots)?;
            let mode = self.forward(EncoderGraph::Packed { slots }, arguments, None, job)?;

            // Slot `i` holds image `i`'s features in its leading rows; the
            // features and the product share their dtype, so the rows are
            // copied as they are.
            let mut rows_out = CopyBatch::new(SourceOrder::Stream);
            for (index, &[rows, columns]) in chunk.iter().enumerate() {
                let count =
                    u64::from(rows) * u64::from(columns) / u64::from(vision.downsample).pow(2);
                let target = out[offset + index].address();
                let source = features.address() + index as u64 * slot_tokens * row_bytes;
                rows_out.push(target, source, (count * row_bytes) as usize);
                result.shapes[offset + index] = vec![count as i64, i64::from(vision.output_size)];
            }
            // SAFETY: each image's rows lie in its slot's features and its
            // target is the product the caller reserved, both alive until
            // the batch commits; the copies follow the encode in the
            // issuer's order, on the stream or in host memory.
            unsafe { issue.copy(&rows_out) }?;
            result.record(mode);
            offset += slots;
        }
        Ok(())
    }
}

/// Native-endian int64 words of a grid layout, as the device reads them.
pub(super) fn grid_bytes(layout: &[[i64; 2]]) -> Vec<u8> {
    layout
        .iter()
        .flatten()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_grids_describe_images_and_slot_padding() -> Result<()> {
        // Two images in three slots of 64 patch rows, downsampled by 2:
        // each image's grid, then a 2-row grid over its slot's padding.
        let layout = packed_grids(&[[4, 8], [8, 8]], 3, 64, 2)?;
        assert_eq!(
            layout,
            vec![[4, 8], [2, 16], [8, 8], [0, 0], [0, 0], [2, 32]]
        );
        assert_eq!(packed_grids(&[], 1, 64, 2)?, vec![[0, 0], [2, 32]]);
        Ok(())
    }

    #[test]
    fn packed_grids_reject_unaligned_oversized_and_excess_images() {
        assert!(packed_grids(&[[3, 4]], 1, 64, 2).is_err());
        assert!(packed_grids(&[[10, 8]], 1, 64, 2).is_err());
        assert!(packed_grids(&[[2, 2], [2, 2]], 1, 64, 2).is_err());
        assert!(packed_grids(&[[0, 4]], 1, 64, 2).is_err());
    }
}
