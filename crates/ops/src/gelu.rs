//! `gelu`: the Gaussian error linear unit, in the two forms PyTorch evaluates.
//!
//! ```text
//! exact  out = 0.5 · x · (1 + erf(x / √2))
//! tanh   out = x · sigmoid(√(8/π) · (x + 0.044715 · x³))
//! ```
//!
//! The two are the reference's two formulas: `gelu` is PyTorch's exact form and `gelu_pytorch_tanh`
//! is its tanh approximation, which is written as one logistic rather than as `0.5·x·(1+tanh(u))`
//! because the logistic needs one exponential and saturates exactly at both ends.
//!
//! Which approximation a model uses is its checkpoint's, not a preference, and one model may use
//! both: a block that gates with `approximate="tanh"`, a tower whose activation is the checkpoint's
//! choice between the two, and a projection that takes the exact form (`Gelu::exact`). A checkpoint
//! that names plain `gelu` is the exact form too. The approximation is therefore part of the
//! compilation, and the two are separate programs.
//!
//! Elementwise over `[rows, columns]`, one element per thread. The result is elementwise, so its
//! tolerance is the output's own rounding plus one envelope: both implementations evaluate the same
//! formula, but `erf` and `exp` are each accurate to about one unit in the last place of their own
//! result, and over the envelope `|x| ≤ 8` the output is bounded by `|x|`.

use std::fmt;

use uniserve_tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{Strides, Operand};
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

/// The envelope of `|x|` the tolerance's absolute part follows.
const VALUE_ENVELOPE: f32 = 8.0;

/// Which of PyTorch's two GELU formulas a compilation evaluates.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Approximation {
    /// `0.5·x·(1 + erf(x/√2))`, PyTorch's `gelu`.
    Exact,
    /// `x·sigmoid(√(8/π)·(x + 0.044715·x³))`, PyTorch's `gelu_pytorch_tanh`.
    Tanh,
}

impl fmt::Display for Approximation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Exact => "exact",
            Self::Tanh => "tanh",
        })
    }
}

/// One compilation of `gelu`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the input and the output.
    pub dtype: DType,
    /// Elements one row holds: a launch word the module reads, so it is no part of the key.
    pub columns: u32,
    /// Elements one thread block evaluates.
    pub block: u32,
    /// Which formula the module evaluates.
    pub approximate: Approximation,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("block", self.block)
            .field("approximate", self.approximate)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `gelu`.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[rows, columns]`, contiguous: the values to activate.
    pub x: Operand,
    /// `[rows, columns]`, contiguous, in the descriptor's element type.
    pub out: Operand,
}

/// The Gaussian error linear unit.
pub struct Gelu;

impl Op for Gelu {
    const NAME: &'static str = proto::program::operator::GELU;

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let columns = u64::from(descriptor.columns);
        args.x.require_rank(Self::NAME, 2)?;
        args.x.require_row(Self::NAME, descriptor.dtype, columns)?;
        if args.x.strides != Strides::row_major(args.x.shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the activation is read contiguous; found strides {}",
                    args.x.strides
                ),
            ));
        }
        args.out
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        args.out.require_shape(Self::NAME, args.x.shape)?;
        if args.out.strides != Strides::row_major(args.x.shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the result is written contiguous; found strides {}",
                    args.out.strides
                ),
            ));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let elements = args.x.shape.numel();
        let mut arguments = Arguments::new();
        arguments
            .view(args.x)
            .view(args.out)
            .u64(elements)
            .u32(descriptor.columns)
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
        // One rounding from FP32 into the output type; the `erf` or the `exp` is the transcendental
        // whose last place the envelope covers.
        let unit = tolerance::unit_roundoff(descriptor.dtype) as f32;
        Tolerance::rounding(1, DType::F32, descriptor.dtype).with_atol(8.0 * VALUE_ENVELOPE * unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_approximations_are_two_compilations() {
        let exact = Descriptor {
            dtype: DType::BF16,
            columns: 3072,
            block: 1024,
            approximate: Approximation::Exact,
            pdl: false,
        };
        let key = crate::descriptor::key(&exact);
        assert_eq!(key, "dtype_bf16_block_1024_approximate_exact_pdl_false");
        let tanh = crate::descriptor::key(&Descriptor {
            approximate: Approximation::Tanh,
            ..exact
        });
        assert!(tanh.contains("approximate_tanh"), "{tanh}");
        assert_ne!(key, tanh, "the formula is part of the compilation");
    }
}
