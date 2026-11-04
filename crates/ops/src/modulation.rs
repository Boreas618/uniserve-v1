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
//! activation dtype, as the eager expression does (specs/worker.md 9, "调制 RMSNorm、门控残差");
//! only the statistics of the mean square accumulate in FP32.
//!
//! One program handles one activation row in `modulated_rms_norm` and loads the whole row before
//! storing, so `out` may alias `value`. It may also write the residual it normalized back into
//! `update` and copy the normalized source into `retain`, so a block reads each activation once.
//! An E4M3 output carries one FP32 dequantization scale per row, the row's largest magnitude over
//! 448, which keeps every value inside the format.

use std::fmt;

use crate::descriptor::{Float, Key};
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
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
            .field("dtype", self.dtype)
            .field("output", self.output)
            .field("indices", self.indices)
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
#[derive(Clone, Copy)]
pub struct Args {
    /// `[rows, width]`, contiguous: the activation row the mean square reduces.
    pub value: TensorView,
    /// `[rows, width]`, contiguous; receives the residual `value + gate[i] · update`.
    pub update: Option<TensorView>,
    /// `[parameters, width]`, unit-strided per channel: the gates of the gated update.
    pub gate: Option<TensorView>,
    /// `[width]`: the learned scale of the normalized row.
    pub weight: TensorView,
    /// `[parameters, width]`, unit-strided per channel: the per-row shift.
    pub shift: TensorView,
    /// `[parameters, width]`, unit-strided per channel: the per-row scale offset.
    pub scale: TensorView,
    /// `[rows]`: the parameter row of each activation row.
    pub row_indices: TensorView,
    /// `[rows, width]`, contiguous, in the descriptor's output type.
    pub out: TensorView,
    /// `[rows]`, FP32: the E4M3 dequantization scale of each row, present exactly when `out` is
    /// E4M3.
    pub output_scale: Option<TensorView>,
    /// `[rows, width]`, contiguous, in the descriptor's output type; receives the normalized
    /// source row when the descriptor retains it.
    pub retain: Option<TensorView>,
}

/// The indexed RMS modulation of a diffusion block.
pub struct ModulatedRmsNorm;

/// `scale` is the dequantization scale of `out`'s E4M3 rows, not of the rows in `dtype`.
fn scaled_output(output: DType) -> bool {
    output == DType::F8E4M3
}

impl Op for ModulatedRmsNorm {
    const NAME: &'static str = "modulated_rms_norm";

    type Descriptor = ModulatedRmsNormDescriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let width = u64::from(descriptor.width);
        if !descriptor.dtype.is_float() {
            return Err(Error::argument(
                Self::NAME,
                format!("{} is not a floating activation type", descriptor.dtype),
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
        let rows = args.value.shape.dim(0);

        args.weight
            .require_shape(Self::NAME, Shape::new(&[width]))?;
        args.weight
            .require_row(Self::NAME, descriptor.dtype, width)?;
        args.row_indices
            .require_shape(Self::NAME, Shape::new(&[rows]))?;
        if args.row_indices.dtype != descriptor.indices {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} row indices, found {}",
                    descriptor.indices, args.row_indices.dtype
                ),
            ));
        }
        args.row_indices
            .require_row(Self::NAME, descriptor.indices, rows)?;

        let activation = Shape::new(&[rows, width]);
        for (name, vector, stride) in [
            ("shift", &args.shift, descriptor.shift_stride),
            ("scale", &args.scale, descriptor.scale_stride),
        ] {
            vector.require_rank(Self::NAME, 2)?;
            vector.require_row(Self::NAME, descriptor.dtype, width)?;
            if vector.strides.get(0) != stride {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the {name} rows are {} elements apart, the descriptor says {stride}",
                        vector.strides.get(0)
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
                if gate.strides.get(0) != descriptor.gate_stride {
                    return Err(Error::argument(
                        Self::NAME,
                        format!(
                            "the gate rows are {} elements apart, the descriptor says {}",
                            gate.strides.get(0),
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
                scale.require_shape(Self::NAME, Shape::new(&[rows]))?;
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
                        descriptor.output
                    ),
                ));
            }
        }

        // A flag is off, so its slot is never dereferenced; the kernel still takes the argument, and
        // an address already in the launch keeps the slot valid.
        let update = args.update.unwrap_or(args.value);
        let gate = args.gate.unwrap_or(args.shift);
        let output_scale = args.output_scale.unwrap_or(args.out);
        let retain = args.retain.unwrap_or(args.out);
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

        let mut arguments = Arguments::new();
        arguments
            .view(args.value)
            .view(update)
            .view(gate)
            .view(args.weight)
            .view(args.shift)
            .view(args.scale)
            .view(args.row_indices)
            .view(args.out)
            .view(output_scale)
            .view(retain)
            .i32(descriptor.gate_stride as i32)
            .i32(descriptor.shift_stride as i32)
            .i32(descriptor.scale_stride as i32);
        kernel.launch(LaunchGrid::linear(rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        Args {
            update: args.update.map(|update| twins.in_place(update, stream)),
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
            statistics.rtol + chain * descriptor.dtype.unit_roundoff() as f32,
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
    pub width: u32,
    /// The elements one launch walks; `rows × width`, baked in like the kernel's own tile mask.
    pub elements: u64,
    /// Elements between the gate rows the row indices select.
    pub gate_stride: u64,
}

impl fmt::Display for GatedResidualDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("indices", self.indices)
            .field("width", self.width)
            .field("elements", self.elements)
            .field("gate_stride", self.gate_stride)
            .finish(formatter)
    }
}

/// The arguments of `gated_residual`.
#[derive(Clone, Copy)]
pub struct GatedResidualArgs {
    /// `[rows, width]`, contiguous: the residual stream.
    pub hidden: TensorView,
    /// `[rows, width]`, contiguous; receives `hidden + gate[i] · update`.
    pub update: TensorView,
    /// `[parameters, width]`, unit-strided per channel: the per-row gates.
    pub gate: TensorView,
    /// `[rows]`: the parameter row of each activation row.
    pub row_indices: TensorView,
}

/// The gated residual of a diffusion block.
pub struct GatedResidual;

impl Op for GatedResidual {
    const NAME: &'static str = "gated_residual";

    type Descriptor = GatedResidualDescriptor;
    type Args<'a> = GatedResidualArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let width = u64::from(descriptor.width);
        args.hidden.require_rank(Self::NAME, 2)?;
        args.hidden
            .require_row(Self::NAME, descriptor.dtype, width)?;
        let rows = args.hidden.shape.dim(0);
        let activation = Shape::new(&[rows, width]);

        args.update.require_shape(Self::NAME, activation)?;
        args.update
            .require_row(Self::NAME, descriptor.dtype, width)?;
        args.gate.require_rank(Self::NAME, 2)?;
        args.gate.require_row(Self::NAME, descriptor.dtype, width)?;
        if args.gate.strides.get(0) != descriptor.gate_stride {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the gate rows are {} elements apart, the descriptor says {}",
                    args.gate.strides.get(0),
                    descriptor.gate_stride
                ),
            ));
        }
        args.row_indices
            .require_shape(Self::NAME, Shape::new(&[rows]))?;
        args.row_indices
            .require_row(Self::NAME, descriptor.indices, rows)?;
        if rows * width != descriptor.elements {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the descriptor was compiled for {} elements, the views hold {}",
                    descriptor.elements,
                    rows * width
                ),
            ));
        }
        if descriptor.gate_stride > i32::MAX as u64 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the gate row stride {} does not fit the kernel's int32",
                    descriptor.gate_stride
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.hidden)
            .view(args.update)
            .view(args.gate)
            .view(args.row_indices)
            .i32(descriptor.gate_stride as i32);
        kernel.launch(
            LaunchGrid::covering(descriptor.elements, GATED_BLOCK)?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        GatedResidualArgs {
            update: twins.in_place(args.update, stream),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The gated product rounds, then the residual sum rounds: two roundings, no reduction.
        Tolerance::rounding(2, descriptor.dtype, descriptor.dtype)
    }
}
