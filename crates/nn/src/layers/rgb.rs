//! The video path's last step: a normalized raster becomes packed 8-bit RGB.
//!
//! The decoder's output is the latent's own scale — a `[3, frames, rows, columns]` raster the model
//! de-normalizes with the checkpoint's ImageNet constants — while what a caller receives is bytes in
//! `[frames, rows, columns, 3]` order. That is one device program (`ops::rgb_bytes`), the video
//! path's analogue of `ops::pcm_interleave`: the reference's arithmetic is `(x · std + mean) · 255`,
//! clamped into `[0, 255]` and rounded to the nearest byte with ties to even.
//!
//! The constants are the checkpoint's, and they never change: they are two published weights, so a
//! captured graph replays them from the arena rather than from a launch argument the worker would
//! have to upload per step. The geometry is the views' — frames, rows and columns are the request's
//! — so no descriptor field states them.

use ops::rgb::{self, CHANNELS, RgbBytes as RgbBytesOp};
use ops::{DType, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;
use crate::weights::Weight;

/// Elements one program converts.
const BLOCK: u32 = rgb::BLOCK;

/// The de-normalization and byte packing of a decoder's raster.
pub struct RgbBytes {
    dtype: DType,
    channels: u32,
    mean: Weight,
    std: Weight,
}

impl RgbBytes {
    /// The descriptor a byte packing of `channels` channels launches, stated without its weights.
    ///
    /// The de-normalization constants are weights and the raster's extents are launch values, so the
    /// key is the element type and the channel count.
    pub fn descriptor_of(dtype: DType, channels: u32) -> rgb::Descriptor {
        rgb::Descriptor {
            dtype,
            channels,
            block: BLOCK,
            pdl: PDL,
        }
    }

    /// The byte packing of a `channels`-channel raster of `dtype`, de-normalized with `mean` and
    /// `std`.
    ///
    /// The two constants are the checkpoint's `[3]` vectors, published once by
    /// [`crate::WeightLoader::constant`]; they are read as f32 whatever the raster's type, because
    /// the reference's de-normalization is f32 arithmetic.
    pub fn new(dtype: DType, channels: u32, mean: Weight, std: Weight) -> Result<Self> {
        if channels != CHANNELS {
            return Err(Error::shape(
                "RgbBytes",
                format!(
                    "the de-normalization has {CHANNELS} ImageNet constants, {channels} channels \
                     were asked for"
                ),
            ));
        }
        for (name, weight) in [("mean", &mean), ("std", &std)] {
            if weight.shape().dims() != [u64::from(CHANNELS)] {
                return Err(Error::shape(
                    "RgbBytes",
                    format!(
                        "the {name} is one value per channel, not {:?}",
                        weight.shape().dims()
                    ),
                ));
            }
            if weight.dtype() != DType::F32 {
                return Err(Error::shape(
                    "RgbBytes",
                    format!(
                        "the {name} is f32 in the reference's arithmetic, not {}",
                        weight.dtype()
                    ),
                ));
            }
        }
        Ok(Self {
            dtype,
            channels,
            mean,
            std,
        })
    }

    /// Map `input [channels, frames, rows, columns]` into `out [frames, rows, columns, channels]`.
    ///
    /// Every extent is the views': the two shapes must describe one raster, and the output's element
    /// type is `u8`, which is the whole point of the step. Nothing is rounded away on the host.
    pub fn forward(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        let channels = u64::from(self.channels);
        let shape = input.shape;
        if shape.rank() != 4 || shape.dim(0) != channels {
            return Err(Error::shape(
                "RgbBytes",
                format!(
                    "{:?} is not [{channels} channels, frames, rows, columns]",
                    shape.dims()
                ),
            ));
        }
        if input.dtype != self.dtype {
            return Err(Error::shape(
                "RgbBytes",
                format!(
                    "the raster is {}, the layer was built for {}",
                    input.dtype, self.dtype
                ),
            ));
        }
        if out.shape.dims() != [shape.dim(1), shape.dim(2), shape.dim(3), channels] {
            return Err(Error::shape(
                "RgbBytes",
                format!(
                    "{:?} is not the packed form of {:?}",
                    out.shape.dims(),
                    shape.dims()
                ),
            ));
        }
        if out.dtype != DType::U8 {
            return Err(Error::shape(
                "RgbBytes",
                format!("the packed image is u8, not {}", out.dtype),
            ));
        }
        // One construction with `RgbBytes::descriptor_of`: a walk and a launch cannot state
        // different programs for the same packing.
        let descriptor = Self::descriptor_of(self.dtype, self.channels);
        f.operate::<RgbBytesOp>(
            &descriptor,
            rgb::Args {
                input,
                mean: self.mean.view(),
                std: self.std.view(),
                out,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Backing;
    use ops::{DevicePtr, Shape};

    /// A published constant: three values of `dtype`, kept in host memory for the test's life.
    fn constants(bytes: &'static [u8], dtype: DType) -> Weight {
        let view = TensorView::new(
            DevicePtr::new(bytes.as_ptr() as u64),
            dtype,
            Shape::new(&[u64::from(CHANNELS)]),
        );
        Weight {
            backing: Backing::new(DevicePtr::new(bytes.as_ptr() as u64), bytes.len() as u64),
            view,
        }
    }

    const F32_CONSTANTS: [u8; 12] = {
        let mut bytes = [0u8; 12];
        let mut index = 0;
        while index < 3 {
            let word = 0.5f32.to_le_bytes();
            bytes[index * 4] = word[0];
            bytes[index * 4 + 1] = word[1];
            bytes[index * 4 + 2] = word[2];
            bytes[index * 4 + 3] = word[3];
            index += 1;
        }
        bytes
    };

    const BF16_CONSTANTS: [u8; 6] = {
        let word = ((0.5f32.to_bits() >> 16) as u16).to_le_bytes();
        let mut bytes = [0u8; 6];
        let mut index = 0;
        while index < 3 {
            bytes[index * 2] = word[0];
            bytes[index * 2 + 1] = word[1];
            index += 1;
        }
        bytes
    };

    #[test]
    fn the_constants_are_three_f32_values() {
        let f32_constants = || constants(&F32_CONSTANTS, DType::F32);
        assert!(RgbBytes::new(DType::BF16, CHANNELS, f32_constants(), f32_constants()).is_ok());
        assert!(
            RgbBytes::new(
                DType::BF16,
                CHANNELS,
                constants(&BF16_CONSTANTS, DType::BF16),
                f32_constants()
            )
            .is_err(),
            "the de-normalization is f32 arithmetic whatever the raster's type"
        );
        assert!(
            RgbBytes::new(DType::BF16, 4, f32_constants(), f32_constants()).is_err(),
            "the constants exist for three channels"
        );
    }

    /// The packing is the reference's arithmetic: `(x · std + mean) · 255`, clamped and rounded to
    /// the nearest byte, ties to even — which this asserts against the operator's own host oracle.
    #[test]
    fn the_packing_is_the_reference_s_rounding() {
        assert_eq!(rgb::quantize(0.0, 0.5, 0.5), 128);
        assert_eq!(rgb::quantize(1.0, 0.5, 0.5), 255);
        assert_eq!(rgb::quantize(-1.0, 0.5, 0.5), 0);
        assert_eq!(rgb::quantize(0.0, 0.0, 1.0), 0);
    }
}
