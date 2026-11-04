//! Row-wise normalizations and the softmax over a row.
//!
//! Each of these reduces over the last axis of its input and writes one value per element:
//!
//! ```text
//! rms_norm      out = x · rsqrt(mean(x²) + eps) · weight,  statistics in FP32
//! add_rms_norm  residual = x + residual; out = rms_norm(residual)
//! layer_norm    out = (x − mean(x)) · rsqrt(var(x) + eps) · weight + bias
//! softmax       out = exp(x − max(x)) / Σ exp(x − max(x))
//! ```
//!
//! One program handles one row and loads its whole row before storing, so the output may alias the
//! input. `add_rms_norm` adds in FP32 and rounds once on the way to the output dtype, which is the
//! fused residual normalization every model here uses (specs/worker.md 9, "RMSNorm 与 add-RMSNorm").

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// One compilation of `rms_norm`: element type, row width, and the row tile the kernel loads.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RmsNormDescriptor {
    /// Element type of the input, the weight and the output.
    pub dtype: DType,
    /// Elements per row.
    pub columns: u32,
    /// The tile one program loads at a time; at least `columns` rounds up to the next power of two.
    pub tile: u32,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for RmsNormDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("columns", self.columns)
            .field("tile", self.tile)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `rms_norm`.
#[derive(Clone, Copy)]
pub struct RmsNormArgs {
    /// `[rows, columns]`.
    pub x: TensorView,
    /// `[columns]`, the per-channel scale.
    pub weight: TensorView,
    /// `[rows, columns]`.
    pub out: TensorView,
    /// The variance floor.
    pub eps: f32,
}

/// Row-wise RMS normalization.
pub struct RmsNorm;

impl Op for RmsNorm {
    const NAME: &'static str = "rms_norm";

    type Descriptor = RmsNormDescriptor;
    type Args<'a> = RmsNormArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let columns = u64::from(descriptor.columns);
        args.x.require_row(Self::NAME, descriptor.dtype, columns)?;
        args.out
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        if args.out.rows() != args.x.rows() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} input rows but {} output rows",
                    args.x.rows(),
                    args.out.rows()
                ),
            ));
        }
        args.weight
            .require_shape(Self::NAME, Shape::new(&[columns]))?;

        let mut arguments = Arguments::new();
        arguments
            .view(args.x)
            .view(args.weight)
            .view(args.out)
            .u32(descriptor.columns)
            .f32(args.eps)
            .flag(descriptor.pdl);
        kernel.launch(LaunchGrid::linear(args.x.rows())?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        RmsNormArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The sum of squares of `columns` terms in FP32; the normalized value is that scale applied
        // to each element, rounded once to the output dtype.
        Tolerance::rounding(u64::from(descriptor.columns), DType::F32, descriptor.dtype)
    }
}

/// One compilation of `add_rms_norm`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AddRmsNormDescriptor {
    /// Element type of the input, the residual, the weight and both outputs.
    pub dtype: DType,
    /// Elements per row.
    pub columns: u32,
    /// The tile one program loads at a time.
    pub tile: u32,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for AddRmsNormDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("columns", self.columns)
            .field("tile", self.tile)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `add_rms_norm`.
#[derive(Clone, Copy)]
pub struct AddRmsNormArgs {
    /// `[rows, columns]`, this layer's output before the residual add.
    pub x: TensorView,
    /// `[rows, columns]`, the residual stream.
    pub residual: TensorView,
    /// `[columns]`, the per-channel scale.
    pub weight: TensorView,
    /// `[rows, columns]`, the normalized row.
    pub out: TensorView,
    /// `[rows, columns]`, the updated residual `x + residual`.
    pub residual_out: TensorView,
    /// The variance floor.
    pub eps: f32,
}

/// Residual add fused into a row-wise RMS normalization.
pub struct AddRmsNorm;

impl Op for AddRmsNorm {
    const NAME: &'static str = "add_rms_norm";

    type Descriptor = AddRmsNormDescriptor;
    type Args<'a> = AddRmsNormArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let columns = u64::from(descriptor.columns);
        args.x.require_row(Self::NAME, descriptor.dtype, columns)?;
        args.residual
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        args.out
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        args.residual_out
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        if args.residual.rows() != args.x.rows() || args.out.rows() != args.x.rows() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} input rows, {} residual rows, {} output rows",
                    args.x.rows(),
                    args.residual.rows(),
                    args.out.rows()
                ),
            ));
        }
        args.weight
            .require_shape(Self::NAME, Shape::new(&[columns]))?;

        let mut arguments = Arguments::new();
        arguments
            .view(args.x)
            .view(args.residual)
            .view(args.weight)
            .view(args.out)
            .view(args.residual_out)
            .u32(descriptor.columns)
            .f32(args.eps)
            .flag(descriptor.pdl);
        kernel.launch(LaunchGrid::linear(args.x.rows())?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        AddRmsNormArgs {
            out: twins.shadow_of(args.out),
            residual_out: twins.in_place(args.residual_out, stream),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The sum of squares over the row, plus one rounded add of the residual: both in FP32.
        Tolerance::rounding(
            u64::from(descriptor.columns) + 1,
            DType::F32,
            descriptor.dtype,
        )
    }
}

/// One compilation of `layer_norm`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LayerNormDescriptor {
    /// Element type of the input, the weight, the bias and the output.
    pub dtype: DType,
    /// Elements per row.
    pub columns: u32,
    /// The tile one program loads at a time.
    pub tile: u32,
    /// Whether the layer has a bias term.
    pub bias: bool,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for LayerNormDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("columns", self.columns)
            .field("tile", self.tile)
            .field("bias", self.bias)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `layer_norm`.
#[derive(Clone, Copy)]
pub struct LayerNormArgs {
    /// `[rows, columns]`.
    pub x: TensorView,
    /// `[columns]`, the per-channel scale.
    pub weight: TensorView,
    /// `[columns]`, the per-channel shift, when the descriptor declares one.
    pub bias: Option<TensorView>,
    /// `[rows, columns]`.
    pub out: TensorView,
    /// The variance floor.
    pub eps: f32,
}

/// Row-wise layer normalization: mean and variance over the whole row.
pub struct LayerNorm;

impl Op for LayerNorm {
    const NAME: &'static str = "layer_norm";

    type Descriptor = LayerNormDescriptor;
    type Args<'a> = LayerNormArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let columns = u64::from(descriptor.columns);
        args.x.require_row(Self::NAME, descriptor.dtype, columns)?;
        args.out
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        if args.out.rows() != args.x.rows() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} input rows but {} output rows",
                    args.x.rows(),
                    args.out.rows()
                ),
            ));
        }
        args.weight
            .require_shape(Self::NAME, Shape::new(&[columns]))?;
        let bias = match (descriptor.bias, args.bias) {
            (true, Some(bias)) => {
                bias.require_shape(Self::NAME, Shape::new(&[columns]))?;
                bias
            }
            (true, None) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the descriptor declares a bias and the call passed none",
                ));
            }
            (false, Some(_)) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the call passed a bias the descriptor does not declare",
                ));
            }
            (false, None) => args.weight,
        };

        let mut arguments = Arguments::new();
        arguments
            .view(args.x)
            .view(args.weight)
            .view(bias)
            .view(args.out)
            .u32(descriptor.columns)
            .f32(args.eps)
            .flag(args.bias.is_some())
            .flag(descriptor.pdl);
        kernel.launch(LaunchGrid::linear(args.x.rows())?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        LayerNormArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // Mean and variance are two sums over the row in FP32, and the normalized value is rounded
        // once to the output dtype.
        Tolerance::rounding(
            2 * u64::from(descriptor.columns),
            DType::F32,
            descriptor.dtype,
        )
    }
}

/// One compilation of `softmax`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SoftmaxDescriptor {
    /// Element type of the input and the output.
    pub dtype: DType,
    /// Elements per row.
    pub columns: u32,
    /// The tile one program loads at a time.
    pub tile: u32,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for SoftmaxDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("columns", self.columns)
            .field("tile", self.tile)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `softmax`.
#[derive(Clone, Copy)]
pub struct SoftmaxArgs {
    /// `[rows, columns]`.
    pub x: TensorView,
    /// `[rows, columns]`.
    pub out: TensorView,
}

/// Softmax over the last axis: a row that sums to one.
pub struct Softmax;

impl Op for Softmax {
    const NAME: &'static str = "softmax";

    type Descriptor = SoftmaxDescriptor;
    type Args<'a> = SoftmaxArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let columns = u64::from(descriptor.columns);
        args.x.require_row(Self::NAME, descriptor.dtype, columns)?;
        args.out
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        if args.out.rows() != args.x.rows() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} input rows but {} output rows",
                    args.x.rows(),
                    args.out.rows()
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.x)
            .view(args.out)
            .u32(descriptor.columns)
            .flag(descriptor.pdl);
        kernel.launch(LaunchGrid::linear(args.x.rows())?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        SoftmaxArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The denominator is a sum of `columns` positive terms accumulated in FP32; every output is
        // one term over that sum.
        Tolerance::rounding(u64::from(descriptor.columns), DType::F32, descriptor.dtype)
    }
}
