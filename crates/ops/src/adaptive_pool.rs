//! `adaptive_pool`: the audio encoder's head average and its adaptive width pooling, in one pass.
//!
//! The audio encoder's attention projection ends with two reductions
//! (`uniserve/models/minimax_h3/audio_vae.py:431-440`):
//!
//! ```text
//! averaged  = attended.transpose(1, 2).mean(dim=2)          # [batch, frames, head_dim]: mean over heads
//! pooled    = F.adaptive_avg_pool1d(averaged, channels)     # [batch, frames, channels]: pool the width
//! ```
//!
//! Pooling to output `i` averages the input positions `start(i) .. end(i)` of a window, with
//! `start(i) = floor(i·width / output)` and `end(i) = ceil((i+1)·width / output)`; a window is never
//! empty. The checkpoint's own numbers narrow — 8 heads of 256 elements pooled to the 32 latent
//! channels, so each output averages 8 positions of all 8 heads — and the widening case (fewer
//! inputs than outputs) repeats a position, which is what adaptive pooling does.
//!
//! **The two reductions are fused**, because both land on the same element of the output:
//! `out[r, i]` is the mean over the window **and** over the heads. Mathematically the fused mean
//! equals the reference's two-step mean; the last bit of a result may differ where the two sums
//! round differently, which `tolerance` states. The fusion is worth it because the alternative is
//! two launches and an intermediate the block would have to reserve.
//!
//! One program produces one element of the output: it walks the window for each head, so a program
//! reads a strided slice of the row. Rows are `[rows, heads, width]` contiguous.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, Strides, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The envelope of one row's mean, which the absolute part of the tolerance follows.
const VALUE_ENVELOPE: f32 = 8.0;

/// One compilation of `adaptive_pool`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the input and the output.
    pub dtype: DType,
    /// Heads per row: the axis every output element averages over.
    pub heads: u32,
    /// Elements per head before the pool (`head_dim`).
    pub width: u32,
    /// Elements per row after the pool.
    pub output: u32,
    /// Output elements one thread block produces.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("heads", self.heads)
            .field("width", self.width)
            .field("output", self.output)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `adaptive_pool`.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[rows, heads, width]`, contiguous: the attended values.
    pub x: TensorView,
    /// `[rows, output]`, contiguous, in the descriptor's element type.
    pub out: TensorView,
}

/// The head average and the adaptive width pool of an attention projection.
pub struct AdaptivePool;

impl Op for AdaptivePool {
    const NAME: &'static str = "adaptive_pool";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        if descriptor.heads == 0 || descriptor.width == 0 || descriptor.output == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a pool needs heads, a width and an output width".to_owned(),
            ));
        }
        args.x.require_rank(Self::NAME, 3)?;
        let shape = args.x.shape;
        if shape.dim(1) != u64::from(descriptor.heads)
            || shape.dim(2) != u64::from(descriptor.width)
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} heads of {} elements, found {} of {}",
                    descriptor.heads,
                    descriptor.width,
                    shape.dim(1),
                    shape.dim(2)
                ),
            ));
        }
        args.x
            .require_row(Self::NAME, descriptor.dtype, shape.dim(2))?;
        if args.x.strides != Strides::row_major(shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the attended values are read contiguous; found strides {}",
                    args.x.strides
                ),
            ));
        }
        args.out.require_shape(
            Self::NAME,
            Shape::new(&[shape.dim(0), u64::from(descriptor.output)]),
        )?;
        args.out
            .require_row(Self::NAME, descriptor.dtype, u64::from(descriptor.output))?;

        let outputs = shape.dim(0) * u64::from(descriptor.output);
        let mut arguments = Arguments::new();
        arguments
            .view(args.x)
            .view(args.out)
            .u64(shape.dim(0))
            .u32(descriptor.heads)
            .u32(descriptor.width)
            .u32(descriptor.output)
            .flag(descriptor.pdl);
        kernel.launch(
            LaunchGrid::covering(outputs, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The fused mean is one division of one FP32 sum; the reference sums the heads and then the
        // window, so a result may differ in the last bit of either accumulation.
        let unit = descriptor.dtype.unit_roundoff() as f32;
        Tolerance::rounding(2, DType::F32, descriptor.dtype).with_atol(8.0 * VALUE_ENVELOPE * unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pool_geometry_is_compiled_in() {
        let descriptor = Descriptor {
            dtype: DType::F32,
            heads: 8,
            width: 256,
            output: 512,
            block: 256,
            pdl: false,
        };
        // The checkpoint's own geometry: 8 heads of 256 elements pooled to 32 latent channels.
        let encoder = Descriptor {
            dtype: DType::F32,
            heads: 8,
            width: 256,
            output: 32,
            block: 256,
            pdl: false,
        };
        assert_eq!(
            crate::descriptor::key(&encoder),
            "dtype_f32_heads_8_width_256_output_32_block_256_pdl_false"
        );
        assert_ne!(
            crate::descriptor::key(&encoder),
            crate::descriptor::key(&Descriptor {
                output: 64,
                ..encoder
            }),
            "the output width is part of the compilation"
        );
        // A widening pool repeats inputs rather than averaging more of them, and is legal.
        let _ = Descriptor {
            output: 512,
            ..descriptor
        };
    }
}
