//! Adaptive average pooling over the last two axes: torch's `AdaptiveAvgPool2d` to one row
//! (`ops::adaptive_pool`).
//!
//! Output `(0, i)` averages every input row and the columns `start(i) .. end(i)`, with
//! `start(i) = floor(i · width / output)` and `end(i) = ceil((i + 1) · width / output)`; a window
//! is never empty. A head average followed by a 1-d adaptive pool of the averaged width — an
//! attention projection's `mean(dim=heads)` then `adaptive_avg_pool1d` — is this pool with the
//! heads as the input rows: both reductions land on the same output element, so one pass computes
//! them, summed in FP32 and rounded once.

use ops::TensorView;
use ops::adaptive_pool::{self, AdaptivePool as AdaptivePoolOp};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;

/// The output elements one pool program produces.
const BLOCK: u32 = 1024;

/// Torch's `AdaptiveAvgPool2d((1, output))` over `[batch, rows, columns]` inputs.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveAvgPool2d {
    output: u32,
}

impl AdaptiveAvgPool2d {
    /// The pool to `size`, torch's `output_size`: one output row of `size[1]` columns.
    ///
    /// # Errors
    ///
    /// An output of more than one row, which the operator does not compute, or of no columns.
    pub fn new(size: [u32; 2]) -> Result<Self> {
        let [rows, output] = size;
        if rows != 1 || output == 0 {
            return Err(Error::shape(
                "AdaptiveAvgPool2d",
                format!("an output of {size:?}: the pool reduces every row to one row of columns"),
            ));
        }
        Ok(Self { output })
    }

    /// `[b, 1, output]` from `x[b, rows, columns]` in `x`'s element type: element `i` of a row is
    /// the mean over every row and column window `i`.
    ///
    /// # Errors
    ///
    /// A view of another rank, extents wider than a descriptor holds, an arena that cannot hold
    /// the output, or the operator's refusal.
    pub fn forward(&self, f: &Forward, x: TensorView) -> Result<TensorView> {
        const OWNER: &str = "AdaptiveAvgPool2d";
        if x.shape.rank() != 3 {
            return Err(Error::shape(
                OWNER,
                format!("{:?} is not [batch, rows, columns]", x.shape.dims()),
            ));
        }
        let pooled = f.empty(
            x.dtype,
            ops::Shape::new(&[x.shape.dim(0), 1, u64::from(self.output)]),
        )?;
        let narrow = |value: u64| {
            u32::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| Error::shape(OWNER, format!("an extent of {value}")))
        };
        // The operator reads `[rows, heads, width]` and writes `[rows, output]`: the input rows
        // are its heads and the columns its width.
        let descriptor = adaptive_pool::Descriptor {
            dtype: x.dtype,
            heads: narrow(x.shape.dim(1))?,
            width: narrow(x.shape.dim(2))?,
            output: self.output,
            block: BLOCK,
            pdl: PDL,
        };
        let out = pooled.reshape(ops::Shape::new(&[x.shape.dim(0), u64::from(self.output)]));
        f.operate::<AdaptivePoolOp>(&descriptor, adaptive_pool::Args { x, out })?;
        Ok(pooled)
    }
}
