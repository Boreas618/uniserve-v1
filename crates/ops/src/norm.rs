//! Row-wise normalizations and the softmax over a row.
//!
//! Each of these reduces over the last axis of its input and writes one value per element:
//!
//! ```text
//! rms_norm         out = x · rsqrt(mean(x²) + eps) · weight,  statistics in FP32
//! add_rms_norm     residual = x + residual; out = rms_norm(residual)
//! gated_rms_norm   out = rms_norm(x) · silu(z)
//! layer_norm       out = (x − mean(x)) · rsqrt(var(x) + eps) · weight + bias
//! softmax          out = exp(x − max(x)) / Σ exp(x − max(x))
//! ```
//!
//! A checkpoint may store an RMS normalization's weight zero-centred, the normalized row scaled
//! by `x̂ · (1 + w)`: `rms_norm` and `add_rms_norm` read such a weight unchanged and add
//! the one in FP32 when their descriptor says [`centred`](RmsNormDescriptor::centred), so the
//! product is the reference's and no rounded `1 + w` is ever stored.
//!
//! One program handles one row and loads its whole row before storing, so the output may alias the
//! input. `add_rms_norm` adds in FP32 and rounds once on the way to the output dtype, which is the
//! fused residual normalization every model here uses.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::operand::Operand;
use crate::tolerance::{self, Tolerance};
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
    /// Whether the weight is stored zero-centred: the row is scaled by `1 + weight`.
    pub centred: bool,
    /// Rows of the gain: row `i` of the input is scaled by gain row `i mod groups`. One for a
    /// gain every row shares; the streams of a hyper-connected residual each have their own.
    pub groups: u32,
}

impl fmt::Display for RmsNormDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("columns", self.columns)
            .field("tile", self.tile)
            .field("pdl", self.pdl);
        // Named only for a zero-centred weight, so every other key stays as it was.
        let key = centred(key, self.centred);

        // Named only for a gain of several rows.
        if self.groups > 1 {
            key.field("groups", self.groups).finish(formatter)
        } else {
            key.finish(formatter)
        }
    }
}

/// `key` with the zero-centred weight named when `centred`.
fn centred(key: Key, centred: bool) -> Key {
    if centred {
        key.field("centred", true)
    } else {
        key
    }
}

/// The arguments of `rms_norm`.
#[derive(Clone)]
pub struct RmsNormArgs {
    /// `[rows, columns]`.
    pub x: Operand,
    /// `[groups · columns]`, the per-channel scale of each gain row, contiguous.
    pub weight: Operand,
    /// `[rows, columns]`.
    pub out: Operand,
    /// The variance floor.
    pub eps: f32,
}

/// Row-wise RMS normalization.
pub struct RmsNorm;

impl Op for RmsNorm {
    const NAME: &'static str = proto::module::operator::RMS_NORM;

    type Descriptor = RmsNormDescriptor;
    type Args<'a> = RmsNormArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
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
        let groups = u64::from(descriptor.groups);
        if groups == 0 || !args.x.rows().is_multiple_of(groups) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} rows under a gain of {groups} rows: the rows are whole groups",
                    args.x.rows()
                ),
            ));
        }
        args.weight.require_shape(Self::NAME, &[groups * columns])?;
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        // The gain's row count is compiled in: a program reads gain row `row mod groups`.
        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("x_ptr", &args.x)
            .view("w_ptr", &args.weight)
            .view("y_ptr", &args.out)
            .u32("columns", descriptor.columns)
            .f32("eps", args.eps)
            .flag("pdl", descriptor.pdl);
        kernel.launch(LaunchGrid::linear(args.x.rows())?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
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
    /// Whether the weight is stored zero-centred: the row is scaled by `1 + weight`.
    pub centred: bool,
}

impl fmt::Display for AddRmsNormDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("columns", self.columns)
            .field("tile", self.tile)
            .field("pdl", self.pdl);
        centred(key, self.centred).finish(formatter)
    }
}

/// The arguments of `add_rms_norm`.
#[derive(Clone)]
pub struct AddRmsNormArgs {
    /// `[rows, columns]`, this layer's output before the residual add.
    pub x: Operand,
    /// `[rows, columns]`, the residual stream.
    pub residual: Operand,
    /// `[columns]`, the per-channel scale.
    pub weight: Operand,
    /// `[rows, columns]`, the normalized row.
    pub out: Operand,
    /// `[rows, columns]`, the updated residual `x + residual`.
    pub residual_out: Operand,
    /// The variance floor.
    pub eps: f32,
}

/// Residual add fused into a row-wise RMS normalization.
pub struct AddRmsNorm;

impl Op for AddRmsNorm {
    const NAME: &'static str = proto::module::operator::ADD_RMS_NORM;

    type Descriptor = AddRmsNormDescriptor;
    type Args<'a> = AddRmsNormArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
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
        args.weight.require_shape(Self::NAME, &[columns])?;
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("x_ptr", &args.x)
            .view("r_ptr", &args.residual)
            .view("w_ptr", &args.weight)
            .view("y_ptr", &args.out)
            .view("c_ptr", &args.residual_out)
            .u32("columns", descriptor.columns)
            .f32("eps", args.eps)
            .flag("pdl", descriptor.pdl);
        kernel.launch(LaunchGrid::linear(args.x.rows())?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        AddRmsNormArgs {
            out: twins.shadow_of(args.out),
            residual_out: twins.in_place(args.residual_out),
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

/// One compilation of `gated_rms_norm`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GatedRmsNormDescriptor {
    /// Element type of the input, the gate, the weight and the output.
    pub dtype: DType,
    /// Elements per row.
    pub columns: u32,
    /// The tile one program loads at a time; at least `columns` rounds up to the next power of two.
    pub tile: u32,
    /// Whether the gate passes through a sigmoid in place of the SiLU: `x̂ · w · sigmoid(z)`
    /// rather than `x̂ · w · silu(z)`.
    pub sigmoid: bool,
}

impl fmt::Display for GatedRmsNormDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("columns", self.columns)
            .field("tile", self.tile);

        // The SiLU gate keeps the key its modules are filed under.
        if self.sigmoid {
            key.variant("gate", "sigmoid").finish(formatter)
        } else {
            key.finish(formatter)
        }
    }
}

/// The arguments of `gated_rms_norm`.
#[derive(Clone)]
pub struct GatedRmsNormArgs {
    /// `[rows, columns]`, contiguous: the rows normalized.
    pub x: Operand,
    /// `[rows, columns]`, contiguous: the gate each normalized element is multiplied by through
    /// the descriptor's activation.
    pub gate: Operand,
    /// `[columns]`, the per-channel scale.
    pub weight: Operand,
    /// `[rows, columns]`, contiguous: receives `x · rsqrt(mean(x²) + eps) · weight · act(gate)`.
    pub out: Operand,
    /// The variance floor.
    pub eps: f32,
}

/// RMS normalization gated by the SiLU of a second input, normalized before it is gated: the output
/// norm of a gated delta network (FLA's `RMSNormGated` with `norm_before_gate`).
///
/// One program normalizes one row; the statistics, the scale, the SiLU and the product are FP32 and
/// the output rounds once.
pub struct GatedRmsNorm;

impl Op for GatedRmsNorm {
    const NAME: &'static str = proto::module::operator::GATED_RMS_NORM;

    type Descriptor = GatedRmsNormDescriptor;
    type Args<'a> = GatedRmsNormArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let columns = u64::from(descriptor.columns);
        if columns == 0 || descriptor.tile < descriptor.columns {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a tile of {} elements does not cover rows of {columns}",
                    descriptor.tile
                ),
            ));
        }
        for (name, view) in [("x", &args.x), ("gate", &args.gate), ("out", &args.out)] {
            view.require_row(Self::NAME, descriptor.dtype, columns)
                .map_err(|error| Error::argument(Self::NAME, format!("{name}: {error}")))?;
            if !view.is_row_major() {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} rows are not contiguous"),
                ));
            }
            if view.rows() != args.x.rows() {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "{} input rows but {} {name} rows",
                        args.x.rows(),
                        view.rows()
                    ),
                ));
            }
        }
        args.weight.require_shape(Self::NAME, &[columns])?;
        args.weight
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("x_ptr", &args.x)
            .view("gate_ptr", &args.gate)
            .view("w_ptr", &args.weight)
            .view("y_ptr", &args.out)
            .u32("columns", descriptor.columns)
            .f32("eps", args.eps);
        kernel.launch(LaunchGrid::linear(args.x.rows())?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        GatedRmsNormArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The sum of squares over the row in FP32, then the scale, the SiLU's few ulps and the gate
        // product, rounded once into the output type.
        let rounding =
            Tolerance::rounding(u64::from(descriptor.columns), DType::F32, descriptor.dtype);
        Tolerance {
            rtol: rounding.rtol + 3.0 * tolerance::unit_roundoff(DType::F32) as f32,
            ..rounding
        }
    }
}

/// One compilation of `layer_norm`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LayerNormDescriptor {
    /// Element type of the input, the weight, the bias and the output.
    pub dtype: DType,
    /// Elements one row holds: a launch word the module reads, so it is no part of the key.
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
            .field("dtype", abi::name(self.dtype))
            .field("tile", self.tile)
            .field("bias", self.bias)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `layer_norm`.
#[derive(Clone)]
pub struct LayerNormArgs {
    /// `[rows, columns]`.
    pub x: Operand,
    /// `[columns]`, the per-channel scale.
    pub weight: Operand,
    /// `[columns]`, the per-channel shift, when the descriptor declares one.
    pub bias: Option<Operand>,
    /// `[rows, columns]`.
    pub out: Operand,
    /// The variance floor.
    pub eps: f32,
}

/// Row-wise layer normalization: mean and variance over the whole row.
pub struct LayerNorm;

impl Op for LayerNorm {
    const NAME: &'static str = proto::module::operator::LAYER_NORM;

    type Descriptor = LayerNormDescriptor;
    type Args<'a> = LayerNormArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
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
        args.weight.require_shape(Self::NAME, &[columns])?;
        match (descriptor.bias, &args.bias) {
            (true, Some(bias)) => {
                bias.require_shape(Self::NAME, &[columns])?;
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
            (false, None) => {}
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        // `check` pairs the bias with the descriptor that declares it. Without one the bias slot
        // holds the weight's address, which the program never reads under a false bias flag.
        let bias = args.bias.as_ref().unwrap_or(&args.weight);

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("x_ptr", &args.x)
            .view("weight_ptr", &args.weight)
            .view("bias_ptr", bias)
            .view("out_ptr", &args.out)
            .u32("columns", descriptor.columns)
            .f32("eps", args.eps)
            .flag("has_bias", args.bias.is_some())
            .flag("pdl", descriptor.pdl);
        kernel.launch(LaunchGrid::linear(args.x.rows())?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
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
            .field("dtype", abi::name(self.dtype))
            .field("columns", self.columns)
            .field("tile", self.tile)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `softmax`.
#[derive(Clone)]
pub struct SoftmaxArgs {
    /// `[rows, columns]`.
    pub x: Operand,
    /// `[rows, columns]`.
    pub out: Operand,
}

/// Softmax over the last axis: a row that sums to one.
pub struct Softmax;

impl Op for Softmax {
    const NAME: &'static str = proto::module::operator::SOFTMAX;

    type Descriptor = SoftmaxDescriptor;
    type Args<'a> = SoftmaxArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
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
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("x_ptr", &args.x)
            .view("out_ptr", &args.out)
            .u32("columns", descriptor.columns)
            .flag("pdl", descriptor.pdl);
        kernel.launch(LaunchGrid::linear(args.x.rows())?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        SoftmaxArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The denominator is a sum of `columns` positive terms accumulated in FP32; every output is
        // one term over that sum. One expression of the weight is a share of a maximum of one, so
        // the bound is over the whole row.
        //
        // The relative bound holds while every term stays in FP32's normal range: a row whose
        // values spread more than about 87 nats underflows its smaller terms into subnormals,
        // where the grid is absolute and far coarser than `gamma`. A vocabulary-wide row of real
        // logits does not reach that spread.
        Tolerance::rounding(u64::from(descriptor.columns), DType::F32, descriptor.dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A gain every row shares and a SiLU gate keep the keys their modules are filed under; a
    /// gain of several rows and a sigmoid gate name themselves.
    #[test]
    fn a_grouped_gain_and_a_sigmoid_gate_name_themselves() {
        let norm = RmsNormDescriptor {
            dtype: DType::BF16,
            columns: 2560,
            tile: 4096,
            pdl: false,
            centred: true,
            groups: 1,
        };
        assert_eq!(
            norm.to_string(),
            "dtype_bf16_columns_2560_tile_4096_pdl_false_centred_true"
        );
        let grouped = RmsNormDescriptor { groups: 4, ..norm };
        assert_eq!(
            grouped.to_string(),
            "dtype_bf16_columns_2560_tile_4096_pdl_false_centred_true_groups_4"
        );

        let gated = GatedRmsNormDescriptor {
            dtype: DType::BF16,
            columns: 128,
            tile: 128,
            sigmoid: false,
        };
        assert_eq!(gated.to_string(), "dtype_bf16_columns_128_tile_128");
        let sigmoid = GatedRmsNormDescriptor {
            sigmoid: true,
            ..gated
        };
        assert_eq!(
            sigmoid.to_string(),
            "dtype_bf16_columns_128_tile_128_gate_sigmoid"
        );
    }
}
