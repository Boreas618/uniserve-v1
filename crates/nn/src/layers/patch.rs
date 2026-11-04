//! The diffusion transformer's packed rows and its latent's four axes.
//!
//! A transformer reads packed rows — one video row is one `1×2×2` patch of the latent's channels —
//! while the capability boundary and the video VAE speak a `[channels, frames, height, width]`
//! latent. The mapping is a device program (`ops::patch`), because a captured step can only replay
//! what the device does and the executor owns every upload.
//!
//! The layer holds the channel count and the latent's rows and columns, which a request's layout
//! fixes; the frame count is a launch value, so a short clip and a long one differ in the arguments
//! rather than in the compiled program. Both directions move elements and are exact.

use ops::TensorView;
use ops::patch::{self, Mode, Patch as PatchOp};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;

/// Which way the mapping is applied.
pub use ops::patch::Mode as PatchMode;

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

    /// The channels one packed row holds four of.
    pub const fn channels(&self) -> u32 {
        self.channels
    }

    /// Map the latent `[channels, frames, height, width]` and the packed rows
    /// `[frames · height/2 · width/2, channels · 4]` into each other.
    ///
    /// `mode` says which side is read: [`PatchMode::Patchify`] reads the latent and writes the rows —
    /// the transformer's input — and [`PatchMode::Unpatchify`] reads the rows and writes the latent,
    /// its exact inverse. `frames` is the latent's frame count; it travels as a launch value so a
    /// clip of another length replays a captured step instead of compiling again, and the latent's
    /// own extent must agree with it.
    pub fn forward(
        &self,
        f: &Forward,
        latent: TensorView,
        rows: TensorView,
        frames: u64,
        mode: Mode,
    ) -> Result<()> {
        let frames = self.check_frames(latent, frames)?;
        let (x, out) = match mode {
            Mode::Patchify => (latent, rows),
            Mode::Unpatchify => (rows, latent),
        };
        let descriptor = patch::Descriptor {
            dtype: latent.dtype,
            channels: self.channels,
            height: self.height,
            width: self.width,
            mode,
            block: BLOCK,
            pdl: PDL,
        };
        f.operate::<PatchOp>(&descriptor, patch::Args { x, out, frames })
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
