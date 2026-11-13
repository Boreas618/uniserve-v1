//! Adaptive average pooling over the last two axes to one row: torch's `adaptive_avg_pool2d`
//! with an output of `(1, columns)`.
//!
//! Output `(0, i)` averages every input row and the columns `start(i) .. end(i)`, with
//! `start(i) = floor(i · width / output)` and `end(i) = ceil((i + 1) · width / output)`; a
//! window is never empty. A head average followed by a 1-d adaptive pool of the averaged width
//! (an attention projection's `mean(dim=heads)`, then `adaptive_avg_pool1d`) is this pool with
//! the heads as the input rows: both reductions land on the same output element, so one pass
//! computes them, summed in FP32 and rounded once.

use ops::adaptive_pool::{AdaptivePool, Args, Descriptor};
use tensor::{DType, Device, Tensor};

use super::{PDL, on_cpu};
use crate::error::{Error, Result};

/// The output elements one pool kernel produces.
const BLOCK: u32 = 1024;

/// `x`, `[batch, rows, columns]`, pooled to `[batch, 1, size[1]]` in its element type: element
/// `i` of a row is the mean over every input row and column window `i`. `size` is torch's
/// `output_size`, whose row count is one.
///
/// # Errors
///
/// When `x` is not three axes, the output is more than one row or no column, or the operator
/// refuses its arguments.
pub fn adaptive_avg_pool2d(x: &Tensor, size: [u32; 2]) -> Result<Tensor> {
    const OWNER: &str = "adaptive_avg_pool2d";

    let [1, output] = size else {
        return Err(Error::shape(
            OWNER,
            format!("an output of {size:?}: the pool reduces every row to one row of columns"),
        ));
    };
    let &[batch, rows, width] = x.dims() else {
        return Err(Error::shape(
            OWNER,
            format!("{:?} is not [batch, rows, columns]", x.dims()),
        ));
    };
    let field = |extent: usize| {
        u32::try_from(extent)
            .ok()
            .filter(|extent| *extent > 0)
            .ok_or_else(|| Error::shape(OWNER, format!("an extent of {extent}")))
    };
    let (heads, columns, outputs) = (field(rows)?, field(width)?, field(output as usize)?);

    if on_cpu(x) {
        let x32 = x.to_dtype(DType::F32)?;
        let outputs = outputs as usize;

        let mut pooled = Vec::with_capacity(outputs);
        for index in 0..outputs {
            let start = index * width / outputs;
            let end = ((index + 1) * width).div_ceil(outputs).max(start + 1);
            let window = x32.narrow(2, start, end - start)?.sum_keepdim((1, 2))?;
            let count = Tensor::new(((end - start) * rows) as f32, &Device::Cpu)?;

            pooled.push(window.broadcast_div(&count)?);
        }
        return Ok(Tensor::cat(&pooled, 2)?.to_dtype(x.dtype())?);
    }

    // The operator reads `[rows, heads, width]` contiguous and writes `[rows, output]`: the
    // input rows are its heads and the columns its width.
    let values = x.contiguous()?;
    let out = ops::empty((batch, outputs as usize), x.dtype())?;

    ops::launch::<AdaptivePool>(
        &Descriptor {
            dtype: x.dtype(),
            heads,
            width: columns,
            output: outputs,
            block: BLOCK,
            pdl: PDL,
        },
        Args {
            x: ops::operand(&values)?,
            out: ops::operand(&out)?,
        },
    )?;
    Ok(out.unsqueeze(1)?)
}
