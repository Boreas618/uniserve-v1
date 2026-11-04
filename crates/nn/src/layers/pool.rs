//! The audio encoder's head reduction: the mean over attention heads and the adaptive width pool,
//! in one pass.
//!
//! The encoder's attention projection ends with `attended.mean(dim=heads)` followed by
//! `F.adaptive_avg_pool1d(..., channels)`; both land on the same output element, so one operator
//! computes them together (specs/worker.md 9, "自适应池化"; `ops::adaptive_pool`). The layer holds
//! the geometry: how many heads a row has, the width each holds, and how many elements the pool
//! produces.

use ops::TensorView;
use ops::adaptive_pool::{self, AdaptivePool as AdaptivePoolOp};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;

/// The elements one pool program produces.
const BLOCK: u32 = 1024;

/// The head average and the adaptive width pool of an attention projection.
pub struct AdaptivePool {
    heads: u32,
    width: u32,
    output: u32,
}

impl AdaptivePool {
    /// Pool `heads` sets of `width` elements into `output` per row.
    pub fn new(heads: u64, width: u64, output: u64) -> Result<Self> {
        let narrow = |name: &'static str, value: u64| {
            u32::try_from(value).map_err(|_| {
                Error::shape(
                    "AdaptivePool",
                    format!("{name} {value} is wider than a descriptor holds"),
                )
            })
        };
        let (heads, width, output) = (
            narrow("heads", heads)?,
            narrow("width", width)?,
            narrow("output", output)?,
        );
        if heads == 0 || width == 0 || output == 0 {
            return Err(Error::shape(
                "AdaptivePool",
                "a pool needs heads, a width and an output width".to_owned(),
            ));
        }
        Ok(Self {
            heads,
            width,
            output,
        })
    }

    /// The pool's output width: the latent channels the encoder's head produces.
    pub const fn output(&self) -> u32 {
        self.output
    }

    /// `out[rows, output]` from `x[rows, heads, width]`.
    pub fn forward(&self, f: &Forward, x: TensorView, out: TensorView) -> Result<()> {
        let expected = [x.shape.dim(0), u64::from(self.output)];
        if x.shape.rank() != 3 || out.shape.rank() != 2 || out.shape.dims() != expected {
            return Err(Error::shape(
                "AdaptivePool",
                format!("{:?} into {:?}", x.shape.dims(), out.shape.dims()),
            ));
        }
        let descriptor = adaptive_pool::Descriptor {
            dtype: x.dtype,
            heads: self.heads,
            width: self.width,
            output: self.output,
            block: BLOCK,
            pdl: PDL,
        };
        f.operate::<AdaptivePoolOp>(&descriptor, adaptive_pool::Args { x, out })
    }
}
