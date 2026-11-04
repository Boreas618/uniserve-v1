//! Drawing the sample a condition reads from the encoder's moments (`ops::posterior`).
//!
//! The layer is the operator without arithmetic of its own: the two moments arrive as the halves of
//! the encoder's channel axis, so their rows are twice as wide as the sample's, and the draw is the
//! caller's. What the layer does add is the statement of the shape the four views must share — the
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

    /// The same, launched with programmatic dependent launch.
    pub fn with_pdl(mut self, pdl: bool) -> Self {
        self.pdl = pdl;
        self
    }

    /// `out[p, c] = round_to_f16(mean[p, c] + exp(0.5 · clamp(log_variance[p, c])) · noise[p, c])`.
    ///
    /// All four views are `[positions, channels]` f32 and each is read through its own row stride, so
    /// `mean` and `log_variance` may be the two halves of one `[positions, 2 · channels]` moments
    /// tensor. `out` may not alias any of them.
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
            &posterior::Descriptor {
                block: posterior::BLOCK,
                pdl: self.pdl,
            },
            posterior::Args {
                mean,
                log_variance,
                noise,
                out,
            },
        )
    }
}
