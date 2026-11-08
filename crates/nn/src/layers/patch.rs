//! The diffusion transformer's packed rows and its latent's four axes.
//!
//! A transformer reads packed rows — one video row is one `1×2×2` patch of the latent's channels —
//! while the capability boundary and the video VAE speak a `[channels, frames, height, width]`
//! latent. The mapping is a device program (`ops::patch`), because a captured step can only replay
//! what the device does and the worker owns every upload.
//!
//! The layer holds the channel count and the latent's rows and columns, which a request's layout
//! fixes; the frame count is a launch value, so a short clip and a long one differ in the arguments
//! rather than in the compiled program. The move is exact.
//!
//! The layer is the transformer's *input* mapping only: a latent becomes the rows it reads. The
//! inverse — rows back to a planar raster — is [`super::UnpatchifyVideo`], the generic operator the
//! decoder calls, so there is one program for the inverse and not two.

use ops::TensorView;
use ops::patch::{self, Patch as PatchOp};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;

/// The elements one patch program walks along a row.
const BLOCK: u32 = patch::BLOCK;

/// The latent ↔ packed-rows mapping of a diffusion transformer.
#[derive(Clone, Copy)]
pub struct Patch {
    channels: u32,
    height: u32,
    width: u32,
}

impl Patch {
    /// The mapping of a `channels`-channel latent of `height` × `width` rows and columns.
    ///
    /// Both extents are even, because a patch is `1×2×2`; the frames stay a call's argument.
    pub fn new(channels: u64, height: u64, width: u64) -> Result<Self> {
        let narrow = |name: &'static str, value: u64| {
            u32::try_from(value).map_err(|_| {
                Error::shape(
                    "Patch",
                    format!("{name} {value} is wider than a descriptor holds"),
                )
            })
        };
        let (channels, height, width) = (
            narrow("channels", channels)?,
            narrow("height", height)?,
            narrow("width", width)?,
        );
        if channels == 0 || height == 0 || width == 0 {
            return Err(Error::shape(
                "Patch",
                "a latent has channels, rows and columns".to_owned(),
            ));
        }
        if !height.is_multiple_of(2) || !width.is_multiple_of(2) {
            return Err(Error::shape(
                "Patch",
                format!("a {height}×{width} latent is not whole 1×2×2 patches"),
            ));
        }
        Ok(Self {
            channels,
            height,
            width,
        })
    }

    /// The channels a packed row carries four values of.
    pub const fn channels(&self) -> u32 {
        self.channels
    }

    /// Pack the latent `[channels, frames, height, width]` into the rows
    /// `[frames · height/2 · width/2, channels · 4]` the transformer reads.
    ///
    /// `frames` is the latent's frame count; it travels as a launch value so a clip of another
    /// length replays a captured step instead of compiling again, and the latent's own extent must
    /// agree with it. The inverse of this mapping is [`super::UnpatchifyVideo::forward`].
    /// The descriptor one patchify launch takes: the element type alone.
    ///
    /// The raster's extents and the frame count travel with the call — the program reads them as
    /// words — so every canvas of an element type is one compiled module.
    pub fn descriptor_of(dtype: ops::DType) -> patch::Descriptor {
        patch::Descriptor {
            dtype,
            block: BLOCK,
            pdl: PDL,
        }
    }

    pub fn forward(
        &self,
        f: &Forward,
        latent: TensorView,
        rows: TensorView,
        frames: u64,
    ) -> Result<()> {
        let frames = self.check_frames(latent, frames)?;
        let descriptor = Self::descriptor_of(latent.dtype);
        // The layer's raster travels with the call: the program reads it as words, so every canvas
        // of an element type is one compiled module, and the layer's geometry never reaches a key.
        f.operate::<PatchOp>(
            &descriptor,
            patch::Args {
                x: latent,
                out: rows,
                frames,
                channels: self.channels,
                height: self.height,
                width: self.width,
            },
        )
    }

    /// The frames the call maps, once the latent agrees with the layout and the caller.
    fn check_frames(&self, latent: TensorView, frames: u64) -> Result<u32> {
        let shape = latent.shape;
        if shape.rank() != 4
            || shape.dim(0) != u64::from(self.channels)
            || shape.dim(2) != u64::from(self.height)
            || shape.dim(3) != u64::from(self.width)
        {
            return Err(Error::shape(
                "Patch",
                format!(
                    "{:?} is not [{}, frames, {}, {}]",
                    shape.dims(),
                    self.channels,
                    self.height,
                    self.width
                ),
            ));
        }
        if shape.dim(1) != frames {
            return Err(Error::shape(
                "Patch",
                format!(
                    "the call names {frames} frames, the latent holds {}",
                    shape.dim(1)
                ),
            ));
        }
        u32::try_from(frames).map_err(|_| {
            Error::shape(
                "Patch",
                format!("{frames} frames are more than a descriptor holds"),
            )
        })
    }
}
