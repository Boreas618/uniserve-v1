//! The video decoder's packed tokens and the planar raster it returns.
//!
//! The video transformer hands the decoder packed tokens: one token is a `temporal × spatial ×
//! spatial` block of the latent's channels. The decoder's convolutions read a planar raster, so the
//! last step of a decode is this mapping — and it is a device program, because a captured step can
//! only replay what the device does (`ops::unpatchify_video`).
//!
//! The layer is the fold and nothing else. The reference's epilogue adds the channel bias and the
//! model's `fold_statistics` already folded the per-channel scale into the token rows, so the
//! operator's `affine` form is off here: a launch that passed a scale and bias would apply both a
//! second time.

use ops::unpatchify::{self, Unpatchify as UnpatchifyOp};
use ops::{DType, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;

/// The elements one program writes.
const BLOCK: u32 = 1024;

/// The packed-token to planar-raster mapping of a video decoder.
#[derive(Clone, Copy)]
pub struct UnpatchifyVideo {
    dtype: DType,
    channels: u32,
    temporal: u32,
    spatial: u32,
}

impl UnpatchifyVideo {
    /// The mapping of a `channels`-channel latent whose token is `temporal` frames of a
    /// `spatial × spatial` patch.
    pub fn new(dtype: DType, channels: u64, temporal: u32, spatial: u32) -> Result<Self> {
        let channels = u32::try_from(channels).map_err(|_| {
            Error::shape(
                "UnpatchifyVideo",
                format!("{channels} channels are wider than a descriptor holds"),
            )
        })?;
        if channels == 0 || temporal == 0 || spatial == 0 {
            return Err(Error::shape(
                "UnpatchifyVideo",
                "a patch has channels, frames and rows".to_owned(),
            ));
        }
        Ok(Self {
            dtype,
            channels,
            temporal,
            spatial,
        })
    }

    /// The channels of the raster.
    pub const fn channels(&self) -> u32 {
        self.channels
    }

    /// Map `tokens` into `out`.
    ///
    /// `tokens` is `[voxels, channels · temporal · spatial²]` and `out` is
    /// `[channels, frames · temporal, height · spatial, width · spatial]`. The token grid's three
    /// extents are the launch's — a clip's length and canvas are the request's — and they are read
    /// from `out`, which the caller already sized: `out.dim(1) / temporal` frames, and likewise for
    /// the rows and columns. Spare token rows past the grid are never read.
    pub fn forward(&self, f: &Forward, tokens: TensorView, out: TensorView) -> Result<()> {
        let shape = out.shape;
        if shape.rank() != 4 || shape.dim(0) != u64::from(self.channels) {
            return Err(Error::shape(
                "UnpatchifyVideo",
                format!(
                    "{:?} is not [{} channels, frames, rows, columns]",
                    shape.dims(),
                    self.channels
                ),
            ));
        }
        let (temporal, spatial) = (u64::from(self.temporal), u64::from(self.spatial));
        let patch_volume = temporal * spatial * spatial;
        if !shape.dim(1).is_multiple_of(temporal)
            || !shape.dim(2).is_multiple_of(spatial)
            || !shape.dim(3).is_multiple_of(spatial)
        {
            return Err(Error::shape(
                "UnpatchifyVideo",
                format!(
                    "{:?} is not whole {}×{}×{} patches",
                    shape.dims(),
                    self.temporal,
                    self.spatial,
                    self.spatial
                ),
            ));
        }
        let (frames, height, width) = (
            shape.dim(1) / temporal,
            shape.dim(2) / spatial,
            shape.dim(3) / spatial,
        );
        let width_per_token = u64::from(self.channels) * patch_volume;
        if tokens.shape.rank() != 2
            || tokens.shape.dim(1) != width_per_token
            || tokens.shape.dim(0) < frames * height * width
        {
            return Err(Error::shape(
                "UnpatchifyVideo",
                format!(
                    "{:?} does not cover {frames}×{height}×{width} tokens {width_per_token} wide",
                    tokens.shape.dims()
                ),
            ));
        }
        let descriptor = unpatchify::Descriptor {
            dtype: self.dtype,
            channels: self.channels,
            patch_frames: self.temporal,
            patch_height: self.spatial,
            patch_width: self.spatial,
            affine: false,
            block: BLOCK,
            pdl: PDL,
        };
        let narrow = |name: &'static str, value: u64| {
            u32::try_from(value).map_err(|_| {
                Error::shape(
                    "UnpatchifyVideo",
                    format!("{name} {value} is wider than a kernel parameter"),
                )
            })
        };
        f.operate::<UnpatchifyOp>(
            &descriptor,
            unpatchify::Args {
                tokens,
                scale: None,
                bias: None,
                out,
                frames: narrow("frames", frames)?,
                height: narrow("rows", height)?,
                width: narrow("columns", width)?,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_patch_has_channels_frames_and_rows() {
        assert!(UnpatchifyVideo::new(DType::BF16, 16, 2, 16).is_ok());
        assert!(UnpatchifyVideo::new(DType::BF16, 0, 2, 16).is_err());
        assert!(UnpatchifyVideo::new(DType::BF16, 16, 0, 16).is_err());
        assert!(UnpatchifyVideo::new(DType::BF16, 16, 2, 0).is_err());
        assert!(UnpatchifyVideo::new(DType::BF16, u64::from(u32::MAX) + 1, 2, 16).is_err());
    }
}
