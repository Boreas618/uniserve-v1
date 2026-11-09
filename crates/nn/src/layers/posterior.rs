//! Drawing the sample a condition reads from the encoder's moments (`ops::posterior`).
//!
//! The layer is the operator without arithmetic of its own: the two moments arrive as the halves of
//! the encoder's channel axis, the draw is the caller's, and each of the four views is addressed
//! through its own two strides -- so a channel-major caller (the VAE's rasters are
//! `[channels, frames, rows, columns]`) hands the halves of one buffer through without a transpose,
//! and a channels-last caller does the same with the other stride pair. What the layer does add is the statement of the shape the four views must share — the
//! operator compares them, and this is where a caller reads that they are compared.
//!
//! The per-channel affine is deliberately not here: normalising by `latents_std` and `latents_mean` is
//! a diagonal `1×1` convolution, which is the audio decoder's own layer, and it takes the sample this
//! layer produces.

use ops::TensorView;
use ops::posterior::{self, Posterior as PosteriorOp};

use crate::error::{Error, Result};
use crate::forward::Forward;

/// The posterior sample of `[positions, channels]` moments.
#[derive(Clone, Copy, Default)]
pub struct Posterior {
    /// Whether the program was compiled for programmatic dependent launch.
    pdl: bool,
}

impl Posterior {
    /// A sample drawn with the operator's own tiling.
    pub fn new() -> Self {
        Self { pdl: false }
    }


    /// `out[p, c] = round_to_f16(mean[p, c] + exp(0.5 · clamp(log_variance[p, c])) · noise[p, c])`.
    ///
    /// All four views are `[positions, channels]` f32 and each is read through its own two strides, so
    /// `mean` and `log_variance` may be the two halves of one moments buffer cut by an offset -- in
    /// either physical layout, channels-last `(row = channels, channel = 1)` or channel-major
    /// `(row = 1, channel = positions)`. `out` may not alias any of them.
    /// The descriptor this layer's launch takes, for a walk without a device.
    pub fn descriptor(&self) -> posterior::Descriptor {
        posterior::Descriptor {
            block: posterior::BLOCK,
            pdl: self.pdl,
        }
    }

    pub fn forward(
        &self,
        f: &Forward,
        mean: TensorView,
        log_variance: TensorView,
        noise: TensorView,
        out: TensorView,
    ) -> Result<()> {
        if mean.shape != out.shape {
            return Err(Error::shape(
                "Posterior",
                format!(
                    "the sample is {:?} but the moments are {:?}: a sample has one value per moment",
                    out.shape.dims(),
                    mean.shape.dims()
                ),
            ));
        }
        f.operate::<PosteriorOp>(
            &self.descriptor(),
            posterior::Args {
                mean,
                log_variance,
                noise,
                out,
            },
        )
    }
}
