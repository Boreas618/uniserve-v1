//! The diffusion transformer's modulation pair: an indexed RMS modulation and the gated residual.
//!
//! A diffusion transformer conditions each block on a timestep embedding: it selects one row of
//! shift, scale and gate vectors per activation row, normalizes the row, applies that row's affine
//! transform, and folds a gated update into the residual stream:
//!
//! ```text
//! gated_residual       u = h + gate[i] · u            (stored into u, one rounding each)
//! modulated_rms_norm   x = v + gate[i] · u            (with an update: the residual that x receives)
//!                      n = x · rsqrt(mean(x²) + eps) · weight
//!                      out = n · (1 + scale[i]) + shift[i]
//! ```
//!
//! `i` is `row_indices[row]`, and the three modulation vectors are indexed along their leading axis,
//! whose stride the descriptor carries: one row of the parameters serves many activation rows, and
//! experts or timestep groups are laid out along that axis. Every intermediate rounds to the
//! activation dtype, as the eager expression does;
//! only the statistics of the mean square accumulate in FP32.
//!
//! One program handles one activation row in `modulated_rms_norm` and loads the whole row before
//! storing, so `out` may alias `value`. It may also write the residual it normalized back into
//! `update` and copy the normalized source into `retain`, so a block reads each activation once.
//! An E4M3 output carries one FP32 dequantization scale per row, the row's largest magnitude over
//! 448, which keeps every value inside the format.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::{Float, Key};
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::operand::Operand;
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

/// The widest row the modulation kernel reduces, in elements.
///
/// One program reduces a complete row, so the row has to fit the program's register budget.
pub const MAX_WIDTH: u32 = 32768;

/// The tile `gated_residual` walks the activation with, in elements.
pub const GATED_BLOCK: u64 = 1024;

/// One compilation of `modulated_rms_norm`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ModulatedRmsNormDescriptor {
    /// Element type of the activation, the weight and the modulation vectors.
    pub dtype: DType,
    /// Element type of the outputs: `dtype`, or E4M3 with a per-row scale.
    pub output: DType,
    /// Element type of the row indices.
    pub indices: DType,
    /// Elements per activation row.
    pub width: u32,
    /// The lanes one program loads: `width` rounded up to a power of two.
    pub block: u32,
    /// The epsilon added to the mean square before the reciprocal square root.
    pub epsilon: Float,
    /// Elements between the gate rows the row indices select.
    pub gate_stride: u64,
    /// Elements between the shift rows the row indices select.
    pub shift_stride: u64,
    /// Elements between the scale rows the row indices select.
    pub scale_stride: u64,
    /// Whether the block folds a gated update into the activation first.
    pub update: bool,
    /// Whether the normalized source is copied out for the caller.
    pub retain: bool,
}

impl fmt::Display for ModulatedRmsNormDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("output", abi::name(self.output))
            .field("indices", abi::name(self.indices))
            .field("width", self.width)
            .field("block", self.block)
            .field("epsilon", self.epsilon)
            .field("gate_stride", self.gate_stride)
            .field("shift_stride", self.shift_stride)
            .field("scale_stride", self.scale_stride)
            .field("update", self.update)
            .field("retain", self.retain)
            .finish(formatter)
    }
}

/// The arguments of `modulated_rms_norm`.
///
/// The three modulation vectors are `[parameters, width]` with `width` unit-strided and their leading
/// stride equal to the descriptor's, because the kernel addresses a parameter row from the leading
/// axis alone. `update` and `retain` are present exactly when the descriptor says so.
#[derive(Clone)]
pub struct Args {
    /// `[rows, width]`, contiguous: the activation row the mean square reduces.
    pub value: Operand,
    /// `[rows, width]`, contiguous; receives the residual `value + gate[i] · update`.
    pub update: Option<Operand>,
    /// `[parameters, width]`, unit-strided per channel: the gates of the gated update.
    pub gate: Option<Operand>,
    /// `[width]`: the learned scale of the normalized row.
    pub weight: Operand,
    /// `[parameters, width]`, unit-strided per channel: the per-row shift.
    pub shift: Operand,
    /// `[parameters, width]`, unit-strided per channel: the per-row scale offset.
    pub scale: Operand,
    /// `[rows]`: the parameter row of each activation row.
    pub row_indices: Operand,
    /// `[rows, width]`, contiguous, in the descriptor's output type.
    pub out: Operand,
    /// `[rows]`, FP32: the E4M3 dequantization scale of each row, present exactly when `out` is
    /// E4M3.
    pub output_scale: Option<Operand>,
    /// `[rows, width]`, contiguous, in the descriptor's output type; receives the normalized
    /// source row when the descriptor retains it.
    pub retain: Option<Operand>,
}

/// The indexed RMS modulation of a diffusion block.
pub struct ModulatedRmsNorm;

/// `scale` is the dequantization scale of `out`'s E4M3 rows, not of the rows in `dtype`.
fn scaled_output(output: DType) -> bool {
    output == DType::F8E4M3
}

impl Op for ModulatedRmsNorm {
    const NAME: &'static str = proto::module::operator::MODULATED_RMS_NORM;

    type Descriptor = ModulatedRmsNormDescriptor;
    type Args<'a> = Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let width = u64::from(descriptor.width);
        if !descriptor.dtype.is_float() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} is not a floating activation type",
                    abi::name(descriptor.dtype)
                ),
            ));
        }
        if descriptor.width == 0 || descriptor.width > MAX_WIDTH {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a row of {} elements is outside the kernel's 1..={MAX_WIDTH}",
                    descriptor.width
                ),
            ));
        }
        if !descriptor.block.is_power_of_two() || u64::from(descriptor.block) < width {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the {} lane block does not cover a row of {width} elements",
                    descriptor.block
                ),
            ));
        }

        args.value.require_rank(Self::NAME, 2)?;
        args.value
            .require_row(Self::NAME, descriptor.dtype, width)?;
        let rows = args.value.dim(0);

        args.weight.require_shape(Self::NAME, &[width])?;
        args.weight
            .require_row(Self::NAME, descriptor.dtype, width)?;
        args.row_indices.require_shape(Self::NAME, &[rows])?;
        if args.row_indices.dtype != descriptor.indices {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} row indices, found {}",
                    abi::name(descriptor.indices),
                    abi::name(args.row_indices.dtype)
                ),
            ));
        }
        args.row_indices
            .require_row(Self::NAME, descriptor.indices, rows)?;

        let activation = &[rows, width];
        for (name, vector, stride) in [
            ("shift", &args.shift, descriptor.shift_stride),
            ("scale", &args.scale, descriptor.scale_stride),
        ] {
            vector.require_rank(Self::NAME, 2)?;
            vector.require_row(Self::NAME, descriptor.dtype, width)?;
            if vector.stride(0) != stride {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the {name} rows are {} elements apart, the descriptor says {stride}",
                        vector.stride(0)
                    ),
                ));
            }
        }

        match (&args.update, &args.gate) {
            (Some(update), Some(gate)) => {
                update.require_shape(Self::NAME, activation)?;
                update.require_row(Self::NAME, descriptor.dtype, width)?;
                gate.require_rank(Self::NAME, 2)?;
                gate.require_row(Self::NAME, descriptor.dtype, width)?;
                if gate.stride(0) != descriptor.gate_stride {
                    return Err(Error::argument(
                        Self::NAME,
                        format!(
                            "the gate rows are {} elements apart, the descriptor says {}",
                            gate.stride(0),
                            descriptor.gate_stride
                        ),
                    ));
                }
            }
            (None, None) if !descriptor.update => {}
            (None, None) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the descriptor folds an update but no update and gate were given".to_owned(),
                ));
            }
            _ => {
                return Err(Error::argument(
                    Self::NAME,
                    "an update needs its gate and a gate needs its update".to_owned(),
                ));
            }
        }

        args.out.require_shape(Self::NAME, activation)?;
        args.out.require_row(Self::NAME, descriptor.output, width)?;
        match (&args.retain, descriptor.retain) {
            (Some(retain), true) => {
                retain.require_shape(Self::NAME, activation)?;
                retain.require_row(Self::NAME, descriptor.output, width)?;
            }
            (None, false) => {}
            (None, true) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the descriptor retains the normalized source but no buffer was given"
                        .to_owned(),
                ));
            }
            (Some(_), false) => {
                return Err(Error::argument(
                    Self::NAME,
                    "a retain buffer was given but the descriptor retains nothing".to_owned(),
                ));
            }
        }
        match (&args.output_scale, scaled_output(descriptor.output)) {
            (Some(scale), true) => {
                scale.require_shape(Self::NAME, &[rows])?;
                scale.require_row(Self::NAME, DType::F32, rows)?;
            }
            (None, false) => {}
            (None, true) => {
                return Err(Error::argument(
                    Self::NAME,
                    "an E4M3 output needs its per-row dequantization scale".to_owned(),
                ));
            }
            (Some(_), false) => {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "a dequantization scale was given for a {} output",
                        abi::name(descriptor.output)
                    ),
                ));
            }
        }

        let strides = [
            descriptor.gate_stride,
            descriptor.shift_stride,
            descriptor.scale_stride,
        ];
        for (name, stride) in ["gate", "shift", "scale"].into_iter().zip(strides) {
            if stride > i32::MAX as u64 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} row stride {stride} does not fit the kernel's int32"),
                ));
            }
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let rows = args.value.dim(0);

        // A flag is off, so its slot is never dereferenced; the kernel still takes the argument, and
        // an address already in the launch keeps the slot valid.
        let update = args.update.as_ref().unwrap_or(&args.value);
        let gate = args.gate.as_ref().unwrap_or(&args.shift);
        let output_scale = args.output_scale.as_ref().unwrap_or(&args.out);
        let retain = args.retain.as_ref().unwrap_or(&args.out);

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("hidden_ptr", &args.value)
            .view("update_ptr", update)
            .view("gate_ptr", gate)
            .view("weight_ptr", &args.weight)
            .view("shift_ptr", &args.shift)
            .view("scale_ptr", &args.scale)
            .view("row_indices_ptr", &args.row_indices)
            .view("output_ptr", &args.out)
            .view("output_scale_ptr", output_scale)
            .view("retain_ptr", retain)
            .u64("gate_row_stride", descriptor.gate_stride)
            .u64("shift_row_stride", descriptor.shift_stride)
            .u64("scale_row_stride", descriptor.scale_stride);
        kernel.launch(LaunchGrid::linear(rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        Args {
            update: args.update.map(|update| twins.in_place(update)),
            out: twins.shadow_of(args.out),
            output_scale: args.output_scale.map(|scale| twins.shadow_of(scale)),
            retain: args.retain.map(|retain| twins.shadow_of(retain)),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The mean square reduces over the row in FP32, and the normalized value rounds once to the
        // output type; the eager modulation expression then rounds its factor, its product and the
        // shifted sum, and a gated update rounds the gated term and the residual sum.
        let statistics =
            Tolerance::rounding(u64::from(descriptor.width), DType::F32, descriptor.dtype);
        let chain = if descriptor.update { 8.0 } else { 6.0 };
        Tolerance::new(
            statistics.atol,
            statistics.rtol + chain * tolerance::unit_roundoff(descriptor.dtype) as f32,
        )
    }
}

/// One compilation of `gated_residual`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GatedResidualDescriptor {
    /// Element type of the residual, the update, the gate and the written residual.
    pub dtype: DType,
    /// Element type of the row indices.
    pub indices: DType,
    /// Elements per activation row.
    ///
    /// The kernel is compiled per variant, so this is a constant it holds: the count one launch
    /// walks is `rows × width`, and `rows` arrives as the launch's last scalar because it changes
    /// with the request.
    pub width: u32,
    /// Elements between the gate rows the row indices select.
    pub gate_stride: u64,
}

impl fmt::Display for GatedResidualDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("indices", abi::name(self.indices))
            .field("width", self.width)
            .field("gate_stride", self.gate_stride)
            .finish(formatter)
    }
}

/// The arguments of `gated_residual`.
#[derive(Clone)]
pub struct GatedResidualArgs {
    /// `[rows, width]`, contiguous: the residual stream.
    pub hidden: Operand,
    /// `[rows, width]`, contiguous; receives `hidden + gate[i] · update`.
    pub update: Operand,
    /// `[parameters, width]`, unit-strided per channel: the per-row gates.
    pub gate: Operand,
    /// `[rows]`: the parameter row of each activation row.
    pub row_indices: Operand,
}

/// The gated residual of a diffusion block.
pub struct GatedResidual;

impl Op for GatedResidual {
    const NAME: &'static str = proto::module::operator::GATED_RESIDUAL;

    type Descriptor = GatedResidualDescriptor;
    type Args<'a> = GatedResidualArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let width = u64::from(descriptor.width);
        args.hidden.require_rank(Self::NAME, 2)?;
        args.hidden
            .require_row(Self::NAME, descriptor.dtype, width)?;
        let rows = args.hidden.dim(0);
        let activation = &[rows, width];

        args.update.require_shape(Self::NAME, activation)?;
        args.update
            .require_row(Self::NAME, descriptor.dtype, width)?;
        args.gate.require_rank(Self::NAME, 2)?;
        args.gate.require_row(Self::NAME, descriptor.dtype, width)?;
        if args.gate.stride(0) != descriptor.gate_stride {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the gate rows are {} elements apart, the descriptor says {}",
                    args.gate.stride(0),
                    descriptor.gate_stride
                ),
            ));
        }
        args.row_indices.require_shape(Self::NAME, &[rows])?;
        args.row_indices
            .require_row(Self::NAME, descriptor.indices, rows)?;
        if descriptor.gate_stride > i32::MAX as u64 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the gate row stride {} does not fit the kernel's int32",
                    descriptor.gate_stride
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
        let width = u64::from(descriptor.width);
        let rows = args.hidden.dim(0);

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("hidden_ptr", &args.hidden)
            .view("update_ptr", &args.update)
            .view("gate_ptr", &args.gate)
            .view("row_indices_ptr", &args.row_indices)
            .u64("gate_row_stride", descriptor.gate_stride)
            // Appended rather than inserted, so every parameter before it keeps its offset: the
            // row count is the one thing a request changes, and the kernel multiplies it by the
            // width it was compiled for.
            .u64("rows", rows);
        kernel.launch(
            LaunchGrid::covering(rows * width, GATED_BLOCK)?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        GatedResidualArgs {
            update: twins.in_place(args.update),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The gated product rounds, then the residual sum rounds: two roundings, no reduction.
        Tolerance::rounding(2, descriptor.dtype, descriptor.dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The key is what a package is compiled against, so a key must not move with the request.
    #[test]
    fn gated_residuals_key_holds_no_row_count() {
        let descriptor = GatedResidualDescriptor {
            dtype: DType::BF16,
            indices: DType::I32,
            width: 5376,
            gate_stride: 32_256,
        };
        assert_eq!(
            descriptor.to_string(),
            "dtype_bf16_indices_i32_width_5376_gate_stride_32256"
        );
    }

    #[test]
    fn the_tolerance_counts_the_product_and_the_sum_and_nothing_else() {
        let descriptor = GatedResidualDescriptor {
            dtype: DType::BF16,
            indices: DType::I32,
            width: 5376,
            gate_stride: 32_256,
        };
        assert_eq!(
            GatedResidual::tolerance(&descriptor),
            Tolerance::rounding(2, DType::BF16, DType::BF16)
        );
    }
}
