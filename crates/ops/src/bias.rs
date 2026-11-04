//! Per-channel bias addition with an optional residual, channel-first.
//!
//! The reference adds a convolution's bias in a separate pass rather than inside the convolution
//! (`uniserve/nn/functional/_bias.py:12-56`):
//!
//! ```text
//! out = (residual + residual_bias) + (values + bias)
//! ```
//!
//! with `values` and `residual` of shape `[batch, channels, *spatial]` and the two biases one value
//! per channel. Every sum rounds once in the operands' dtype, in that grouping, which is what makes
//! the result equal the separate tensor additions bit for bit — the H3 video decoder's blocks rely
//! on that equality to keep a biased convolution identical to `conv(x, weight) + bias`.
//!
//! **Strides, and why a view is enough.** The reference accepts any storage whose spatial axes merge
//! into one strided axis: contiguous `[batch, channels, spatial]` (spatial stride 1) and
//! channels-last `[batch, spatial, channels]` (channel stride 1) are both that shape read with
//! different strides, so the operator takes a three-axis view and the strides travel as scalars. The
//! output is written **contiguous, channel-first**, so a channels-last convolution's result leaves
//! this operator in the layout its next consumer expects.
//!
//! One element of `out` per thread: the batch and channel of an element come from the linear index,
//! and the two strides are read from the argument list, so a program handles any storage.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, Strides, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// One compilation of `bias_add`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of every operand.
    pub dtype: DType,
    /// Channels: the length of the biases and the axis between the batch and the spatial axes.
    pub channels: u32,
    /// Elements one thread block adds.
    pub block: u32,
    /// Whether the call adds a residual.
    pub residual: bool,
    /// Whether the call adds a bias to the residual too.
    pub residual_bias: bool,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("channels", self.channels)
            .field("block", self.block)
            .field("residual", self.residual)
            .field("residual_bias", self.residual_bias)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `bias_add`.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[batch, channels, spatial]`, in any storage whose three axes the strides describe.
    pub values: TensorView,
    /// `[channels]`, the bias each channel receives.
    pub bias: TensorView,
    /// `[batch, channels, spatial]` in the same storage as `values`, when the descriptor adds one.
    ///
    /// An absent residual carries the values' address, so one argument list serves both variants.
    pub residual: TensorView,
    /// `[channels]`, the bias the residual receives; absent carries `bias`'s address.
    pub residual_bias: TensorView,
    /// `[batch, channels, spatial]`, contiguous: the result.
    pub out: TensorView,
}

/// A per-channel bias, and the residual it is added to.
pub struct BiasAdd;

impl Op for BiasAdd {
    const NAME: &'static str = "bias_add";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let channels = u64::from(descriptor.channels);
        args.values.require_rank(Self::NAME, 3)?;
        let shape = args.values.shape;
        if shape.dim(1) != channels {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {channels} channels on the axis after the batch, found {}",
                    shape.dim(1)
                ),
            ));
        }
        args.values
            .require_row(Self::NAME, descriptor.dtype, shape.dim(2))?;
        let parameters = Shape::new(&[channels]);
        args.bias.require_shape(Self::NAME, parameters)?;
        args.bias
            .require_row(Self::NAME, descriptor.dtype, channels)?;
        if descriptor.residual {
            args.residual.require_shape(Self::NAME, shape)?;
            args.residual
                .require_row(Self::NAME, descriptor.dtype, shape.dim(2))?;
        }
        if descriptor.residual_bias {
            if !descriptor.residual {
                return Err(Error::argument(
                    Self::NAME,
                    "a residual bias needs a residual to bias",
                ));
            }
            args.residual_bias.require_shape(Self::NAME, parameters)?;
            args.residual_bias
                .require_row(Self::NAME, descriptor.dtype, channels)?;
        }
        args.out.require_shape(Self::NAME, shape)?;
        args.out
            .require_row(Self::NAME, descriptor.dtype, shape.dim(2))?;
        if args.out.strides != Strides::row_major(shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the result is written channel-first and contiguous; found strides {}",
                    args.out.strides
                ),
            ));
        }

        let elements = shape.numel();
        let mut arguments = Arguments::new();
        arguments
            .view(args.values)
            .view(args.bias)
            .view(args.residual)
            .view(args.residual_bias)
            .view(args.out)
            .u32(descriptor.channels)
            .u64(shape.dim(0))
            .u64(shape.dim(2))
            .i64(stride(&args.values, 0)?)
            .i64(stride(&args.values, 1)?)
            .i64(stride(&args.residual, 0)?)
            .i64(stride(&args.residual, 1)?)
            .flag(descriptor.pdl);
        kernel.launch(
            LaunchGrid::covering(elements, u64::from(descriptor.block))?,
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
        // Two or three additions in the operands' dtype, in the reference's grouping; an
        // implementation that accumulates in FP32 is a different variant, not a rounding of this one.
        let _ = descriptor;
        Tolerance::exact()
    }
}

/// One axis of a view's strides as a kernel parameter.
fn stride(view: &TensorView, axis: usize) -> Result<i64> {
    i64::try_from(view.strides.get(axis)).map_err(|_| {
        Error::argument(
            BiasAdd::NAME,
            format!(
                "the stride of axis {axis} does not fit a kernel parameter: {}",
                view.strides.get(axis)
            ),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bias_length_and_the_residual_variants_are_compilations() {
        let plain = Descriptor {
            dtype: DType::BF16,
            channels: 512,
            block: 1024,
            residual: false,
            residual_bias: false,
            pdl: false,
        };
        assert_eq!(
            crate::descriptor::key(&plain),
            "dtype_bf16_channels_512_block_1024_residual_false_residual_bias_false_pdl_false"
        );
        // A call that adds a residual runs a different program than one that does not: the second
        // reads two operands and rounds twice.
        assert_ne!(
            crate::descriptor::key(&plain),
            crate::descriptor::key(&Descriptor {
                residual: true,
                ..plain
            })
        );
        // The bias length is part of the compilation, so a layer's channel count is too.
        assert_ne!(
            crate::descriptor::key(&plain),
            crate::descriptor::key(&Descriptor {
                channels: 256,
                ..plain
            })
        );
    }
}
