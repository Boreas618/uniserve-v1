//! The residual stream's normalization: a channel-scaled update folded into it, and the rows that
//! follow.
//!
//! A transformer block adds a projection's output back into the residual stream with one learned
//! scale per channel and then normalizes. The eager code expresses that as several elementwise
//! kernels and a reduction; here each composition is one operator, so a block reads the residual
//! once:
//!
//! ```text
//! scaled_add                   t = β · t + α · s
//! scaled_residual              h = h + (u + b) · scale
//! scaled_residual_rms_norm     h = h + (u + b) · scale;  out = rms(h) · weight
//! scaled_residual_layer_norm   out = layer_norm(h + (u + b) · scale) · weight + bias
//! weighted_rms_norm            out = rms(h) · weight
//! ```
//!
//! The sum and the normalization's statistics stay in FP32 until the store, which is what the eager
//! expression rounds; `scaled_residual_rms_norm` normalizes the *unrounded* sum while storing the
//! rounded one, as the reference does. A launch may also record each program's largest normalized
//! magnitude, an FP32 vector the caller reduces (specs/worker.md 9, the residual and normalization
//! rows of the MiniMax-H3 table).

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The widest row these kernels reduce, in elements.
pub const MAX_WIDTH: u32 = 32768;

/// The elements one `scaled_residual` program walks.
pub const RESIDUAL_BLOCK: u32 = 1024;

/// One compilation of `weighted_rms_norm`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WeightedRmsNormDescriptor {
    /// Element type of the rows, the weight and the output.
    pub dtype: DType,
    /// Elements per row.
    pub columns: u32,
    /// The lanes one program loads: `columns` rounded up to a power of two.
    pub tile: u32,
    /// Whether each program records its largest normalized magnitude.
    pub absmax: bool,
}

impl fmt::Display for WeightedRmsNormDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("columns", self.columns)
            .field("tile", self.tile)
            .field("absmax", self.absmax)
            .finish(formatter)
    }
}

/// The arguments of `weighted_rms_norm`.
#[derive(Clone, Copy)]
pub struct WeightedRmsNormArgs {
    /// `[rows, columns]`, contiguous.
    pub x: TensorView,
    /// `[columns]`: the learned scale.
    pub weight: TensorView,
    /// `[rows, columns]`: receives the normalized rows.
    pub out: TensorView,
    /// The epsilon added to the mean square.
    pub eps: f32,
    /// `[rows]`, FP32: each row's largest normalized magnitude, present exactly when the descriptor
    /// records it.
    pub partials: Option<TensorView>,
}

/// The weighted RMS normalization of whole rows.
pub struct WeightedRmsNorm;

impl Op for WeightedRmsNorm {
    const NAME: &'static str = "weighted_rms_norm";

    type Descriptor = WeightedRmsNormDescriptor;
    type Args<'a> = WeightedRmsNormArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        check_descriptor(Self::NAME, descriptor.columns, descriptor.tile)?;
        let columns = u64::from(descriptor.columns);
        args.x.require_rank(Self::NAME, 2)?;
        args.x.require_row(Self::NAME, descriptor.dtype, columns)?;
        let rows = args.x.shape.dim(0);
        args.weight
            .require_shape(Self::NAME, Shape::new(&[columns]))?;
        args.weight
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        args.out.require_shape(Self::NAME, args.x.shape)?;
        args.out
            .require_row(Self::NAME, descriptor.dtype, columns)?;

        let partials = matches(
            &args.partials,
            descriptor.absmax,
            Self::NAME,
            "magnitude",
            rows,
        )?;
        let mut arguments = Arguments::new();
        arguments.view(args.x).view(args.weight).view(args.out);
        if let Some(partials) = partials {
            arguments.view(partials);
        }
        arguments.f32(args.eps);
        kernel.launch(LaunchGrid::linear(rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        WeightedRmsNormArgs {
            out: twins.shadow_of(args.out),
            partials: args.partials.map(|partials| twins.shadow_of(partials)),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The mean square reduces the row in FP32; the normalized value rounds once into the output
        // type.
        Tolerance::rounding(u64::from(descriptor.columns), DType::F32, descriptor.dtype)
    }
}

/// One compilation of `scaled_residual_rms_norm`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ScaledResidualRmsNormDescriptor {
    /// Element type of the residual, the update, the scale, the weight and both outputs.
    pub dtype: DType,
    /// Elements per row.
    pub columns: u32,
    /// The lanes one program loads.
    pub tile: u32,
    /// Whether the update carries a bias.
    pub update_bias: bool,
    /// Whether each program records its largest normalized magnitude.
    pub absmax: bool,
}

impl fmt::Display for ScaledResidualRmsNormDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("columns", self.columns)
            .field("tile", self.tile)
            .field("update_bias", self.update_bias)
            .field("absmax", self.absmax)
            .finish(formatter)
    }
}

/// The arguments of `scaled_residual_rms_norm`.
#[derive(Clone, Copy)]
pub struct ScaledResidualRmsNormArgs {
    /// `[rows, columns]`, contiguous: the residual, updated in place.
    pub hidden: TensorView,
    /// `[rows, columns]`, contiguous: the block's output.
    pub update: TensorView,
    /// `[columns]`: the update's bias, present exactly when the descriptor adds one.
    pub update_bias: Option<TensorView>,
    /// `[columns]` per channel, or `[1]` for one factor over the whole stream.
    pub scale: TensorView,
    /// `[columns]`: the normalization's learned scale.
    pub weight: TensorView,
    /// `[rows, columns]`: receives the normalized unrounded sum.
    pub out: TensorView,
    /// The epsilon added to the mean square.
    pub eps: f32,
    /// `[rows]`, FP32: each row's largest normalized magnitude.
    pub partials: Option<TensorView>,
}

/// Fold a channel-scaled update into the residual and store its RMS rows.
pub struct ScaledResidualRmsNorm;

impl Op for ScaledResidualRmsNorm {
    const NAME: &'static str = "scaled_residual_rms_norm";

    type Descriptor = ScaledResidualRmsNormDescriptor;
    type Args<'a> = ScaledResidualRmsNormArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        check_descriptor(Self::NAME, descriptor.columns, descriptor.tile)?;
        let columns = u64::from(descriptor.columns);
        for (name, view) in [("residual", &args.hidden), ("update", &args.update)] {
            view.require_rank(Self::NAME, 2)?;
            view.require_row(Self::NAME, descriptor.dtype, columns)?;
            if view.shape.dim(0) != args.hidden.shape.dim(0) {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} holds {} rows", view.shape.dim(0)),
                ));
            }
        }
        let rows = args.hidden.shape.dim(0);
        let bias = match (&args.update_bias, descriptor.update_bias) {
            (Some(bias), true) => {
                bias.require_shape(Self::NAME, Shape::new(&[columns]))?;
                bias.require_row(Self::NAME, descriptor.dtype, columns)?;
                Some(*bias)
            }
            (None, false) => None,
            (None, true) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the descriptor adds an update bias but none was given".to_owned(),
                ));
            }
            (Some(_), false) => {
                return Err(Error::argument(
                    Self::NAME,
                    "an update bias was given but the descriptor adds none".to_owned(),
                ));
            }
        };
        let width = scale_width(Self::NAME, &args.scale, descriptor.dtype, columns)?;
        args.weight
            .require_shape(Self::NAME, Shape::new(&[columns]))?;
        args.weight
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        args.out.require_shape(Self::NAME, args.hidden.shape)?;
        args.out
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        let partials = matches(
            &args.partials,
            descriptor.absmax,
            Self::NAME,
            "magnitude",
            rows,
        )?;

        let mut arguments = Arguments::new();
        arguments
            .view(args.hidden)
            .view(args.update)
            .view(bias.unwrap_or(args.update))
            .view(args.scale)
            .view(args.weight)
            .view(args.out);
        if let Some(partials) = partials {
            arguments.view(partials);
        }
        arguments.f32(args.eps).u32(width);
        kernel.launch(LaunchGrid::linear(rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        ScaledResidualRmsNormArgs {
            hidden: twins.in_place(args.hidden, stream),
            out: twins.shadow_of(args.out),
            partials: args.partials.map(|partials| twins.shadow_of(partials)),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The gated update rounds twice into the residual, and the mean square reduces the row in
        // FP32 before the normalized value rounds once into the output type.
        let statistics =
            Tolerance::rounding(u64::from(descriptor.columns), DType::F32, descriptor.dtype);
        Tolerance::new(
            statistics.atol,
            statistics.rtol + 3.0 * descriptor.dtype.unit_roundoff() as f32,
        )
    }
}

/// One compilation of `scaled_residual_layer_norm`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ScaledResidualLayerNormDescriptor {
    /// Element type of the residual, the update, the scale, the affine parameters and the output.
    pub dtype: DType,
    /// Elements per row.
    pub columns: u32,
    /// The lanes one program loads.
    pub tile: u32,
    /// Whether the update carries a bias.
    pub update_bias: bool,
    /// Whether each program records its largest normalized magnitude.
    pub absmax: bool,
}

impl fmt::Display for ScaledResidualLayerNormDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("columns", self.columns)
            .field("tile", self.tile)
            .field("update_bias", self.update_bias)
            .field("absmax", self.absmax)
            .finish(formatter)
    }
}

/// The arguments of `scaled_residual_layer_norm`.
#[derive(Clone, Copy)]
pub struct ScaledResidualLayerNormArgs {
    /// `[rows, columns]`, contiguous: the residual, unchanged by the launch.
    pub hidden: TensorView,
    /// `[rows, columns]`, contiguous: the block's output.
    pub update: TensorView,
    /// `[columns]`: the update's bias, present exactly when the descriptor adds one.
    pub update_bias: Option<TensorView>,
    /// `[columns]` per channel, or `[1]` for one factor over the whole stream.
    pub scale: TensorView,
    /// `[columns]`: the normalization's learned scale.
    pub weight: TensorView,
    /// `[columns]`: the normalization's learned bias.
    pub bias: TensorView,
    /// `[rows, columns]`: receives the normalized rows.
    pub out: TensorView,
    /// The epsilon added to the variance.
    pub eps: f32,
    /// `[rows]`, FP32: each row's largest normalized magnitude.
    pub partials: Option<TensorView>,
}

/// Store the layer-normalized rows of a scaled residual without updating it.
pub struct ScaledResidualLayerNorm;

impl Op for ScaledResidualLayerNorm {
    const NAME: &'static str = "scaled_residual_layer_norm";

    type Descriptor = ScaledResidualLayerNormDescriptor;
    type Args<'a> = ScaledResidualLayerNormArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        check_descriptor(Self::NAME, descriptor.columns, descriptor.tile)?;
        let columns = u64::from(descriptor.columns);
        for (name, view) in [("residual", &args.hidden), ("update", &args.update)] {
            view.require_rank(Self::NAME, 2)?;
            view.require_row(Self::NAME, descriptor.dtype, columns)?;
            if view.shape.dim(0) != args.hidden.shape.dim(0) {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} holds {} rows", view.shape.dim(0)),
                ));
            }
        }
        let rows = args.hidden.shape.dim(0);
        let update_bias = match (&args.update_bias, descriptor.update_bias) {
            (Some(bias), true) => {
                bias.require_shape(Self::NAME, Shape::new(&[columns]))?;
                bias.require_row(Self::NAME, descriptor.dtype, columns)?;
                Some(*bias)
            }
            (None, false) => None,
            (None, true) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the descriptor adds an update bias but none was given".to_owned(),
                ));
            }
            (Some(_), false) => {
                return Err(Error::argument(
                    Self::NAME,
                    "an update bias was given but the descriptor adds none".to_owned(),
                ));
            }
        };
        let width = scale_width(Self::NAME, &args.scale, descriptor.dtype, columns)?;
        for view in [&args.weight, &args.bias] {
            view.require_shape(Self::NAME, Shape::new(&[columns]))?;
            view.require_row(Self::NAME, descriptor.dtype, columns)?;
        }
        args.out.require_shape(Self::NAME, args.hidden.shape)?;
        args.out
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        let partials = matches(
            &args.partials,
            descriptor.absmax,
            Self::NAME,
            "magnitude",
            rows,
        )?;

        let mut arguments = Arguments::new();
        arguments
            .view(args.hidden)
            .view(args.update)
            .view(update_bias.unwrap_or(args.update))
            .view(args.scale)
            .view(args.weight)
            .view(args.bias)
            .view(args.out);
        if let Some(partials) = partials {
            arguments.view(partials);
        }
        arguments.f32(args.eps).u32(width);
        kernel.launch(LaunchGrid::linear(rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        ScaledResidualLayerNormArgs {
            out: twins.shadow_of(args.out),
            partials: args.partials.map(|partials| twins.shadow_of(partials)),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // Two sums over the row in FP32 — the mean and the variance — plus the rounded update, then
        // one rounding of the normalized value into the output type.
        let statistics = Tolerance::rounding(
            2 * u64::from(descriptor.columns),
            DType::F32,
            descriptor.dtype,
        );
        Tolerance::new(
            statistics.atol,
            statistics.rtol + 3.0 * descriptor.dtype.unit_roundoff() as f32,
        )
    }
}

/// One compilation of `scaled_residual`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ScaledResidualDescriptor {
    /// Element type of the residual, the update and the scale.
    pub dtype: DType,
    /// Elements per row.
    pub columns: u32,
    /// The elements one program walks.
    pub block: u32,
    /// Whether the update carries a bias.
    pub update_bias: bool,
}

impl fmt::Display for ScaledResidualDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("columns", self.columns)
            .field("block", self.block)
            .field("update_bias", self.update_bias)
            .finish(formatter)
    }
}

/// The arguments of `scaled_residual`.
#[derive(Clone, Copy)]
pub struct ScaledResidualArgs {
    /// `[rows, columns]`, contiguous: the residual, updated in place.
    pub hidden: TensorView,
    /// `[rows, columns]`, contiguous: the block's output.
    pub update: TensorView,
    /// `[columns]`: the update's bias, present exactly when the descriptor adds one.
    pub update_bias: Option<TensorView>,
    /// `[columns]` per channel, or `[1]` for one factor over the whole stream.
    pub scale: TensorView,
}

/// Add a channel-scaled, optionally biased update into the residual.
pub struct ScaledResidual;

impl Op for ScaledResidual {
    const NAME: &'static str = "scaled_residual";

    type Descriptor = ScaledResidualDescriptor;
    type Args<'a> = ScaledResidualArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        if descriptor.columns == 0 || descriptor.block == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a row or a block of zero elements adds nothing".to_owned(),
            ));
        }
        let columns = u64::from(descriptor.columns);
        for (name, view) in [("residual", &args.hidden), ("update", &args.update)] {
            view.require_rank(Self::NAME, 2)?;
            view.require_row(Self::NAME, descriptor.dtype, columns)?;
            if view.shape != args.hidden.shape {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} does not have the residual's shape"),
                ));
            }
        }
        let bias = match (&args.update_bias, descriptor.update_bias) {
            (Some(bias), true) => {
                bias.require_shape(Self::NAME, Shape::new(&[columns]))?;
                bias.require_row(Self::NAME, descriptor.dtype, columns)?;
                Some(*bias)
            }
            (None, false) => None,
            (None, true) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the descriptor adds an update bias but none was given".to_owned(),
                ));
            }
            (Some(_), false) => {
                return Err(Error::argument(
                    Self::NAME,
                    "an update bias was given but the descriptor adds none".to_owned(),
                ));
            }
        };
        let width = scale_width(Self::NAME, &args.scale, descriptor.dtype, columns)?;
        let elements = args.hidden.shape.numel();
        if elements > i32::MAX as u64 {
            return Err(Error::argument(
                Self::NAME,
                format!("{elements} elements do not fit the kernel's int32 count"),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.hidden)
            .view(args.update)
            .view(bias.unwrap_or(args.update))
            .view(args.scale)
            .i32(elements as i32)
            .u32(width);
        kernel.launch(
            LaunchGrid::covering(elements, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        ScaledResidualArgs {
            hidden: twins.in_place(args.hidden, stream),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The biased update rounds, its product with the scale rounds, the sum rounds.
        Tolerance::new(0.0, 3.0 * descriptor.dtype.unit_roundoff() as f32)
    }
}

/// A descriptor's row width and lane tile have to cover a row the kernel reduces.
fn check_descriptor(operator: &'static str, columns: u32, tile: u32) -> Result<()> {
    if columns == 0 || columns > MAX_WIDTH {
        return Err(Error::argument(
            operator,
            format!("a row of {columns} elements is outside the kernel's 1..={MAX_WIDTH}"),
        ));
    }
    if !tile.is_power_of_two() || tile < columns {
        return Err(Error::argument(
            operator,
            format!("a tile of {tile} lanes does not cover a row of {columns} elements"),
        ));
    }
    Ok(())
}

/// A per-channel scale, or one factor for every channel: `[columns]` or `[1]`.
///
/// A diffusion solver's step rewrites its residual stream with one device scalar per
/// `(layout, step)`, and a `[columns]` view would have to be materialized per step, which is a
/// per-step upload. So a scale of one element is the broadcast form: the kernel indexes it by
/// `element % scale_width`, which is the column for `[columns]` and index 0 for `[1]`. The width
/// travels as a launch argument, so both forms are the same program and neither moves a key.
fn scale_width(
    operator: &'static str,
    scale: &TensorView,
    dtype: DType,
    columns: u64,
) -> Result<u32> {
    let width = scale.shape.numel();
    if scale.shape.rank() != 1 || (width != 1 && width != columns) {
        return Err(Error::argument(
            operator,
            format!(
                "expected a scale of {columns} channels or one broadcast factor, found {} \
                 elements in {} dimensions",
                width,
                scale.shape.rank()
            ),
        ));
    }
    scale.require_row(operator, dtype, width)?;
    Ok(width as u32)
}

/// Check an optional magnitude buffer against the descriptor's flag and the row count.
fn matches(
    partials: &Option<TensorView>,
    expected: bool,
    operator: &'static str,
    what: &str,
    rows: u64,
) -> Result<Option<TensorView>> {
    match (partials, expected) {
        (Some(partials), true) => {
            partials.require_shape(operator, Shape::new(&[rows]))?;
            partials.require_row(operator, DType::F32, rows)?;
            Ok(Some(*partials))
        }
        (None, false) => Ok(None),
        (None, true) => Err(Error::argument(
            operator,
            format!("the descriptor records {what} but no buffer was given"),
        )),
        (Some(_), false) => Err(Error::argument(
            operator,
            format!("a {what} buffer was given but the descriptor records none"),
        )),
    }
}

/// The elements one `scaled_add` program walks.
pub const SCALED_ADD_BLOCK: u32 = 1024;

/// One compilation of `scaled_add`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ScaledAddDescriptor {
    /// Element type of the target, which is read and written.
    pub target: DType,
    /// Element type of the source, which is only read.
    pub source: DType,
    /// Elements per row.
    pub columns: u32,
    /// The elements one program walks.
    pub block: u32,
}

impl fmt::Display for ScaledAddDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("target", self.target)
            .field("source", self.source)
            .field("columns", self.columns)
            .field("block", self.block)
            .finish(formatter)
    }
}

/// The arguments of `scaled_add`.
#[derive(Clone, Copy)]
pub struct ScaledAddArgs {
    /// `[rows, columns]`, contiguous: the target, updated in place.
    pub target: TensorView,
    /// `[rows, columns]`, contiguous: the source, the same shape as the target.
    pub source: TensorView,
    /// The source's coefficient, a host value: no upload, no view.
    pub alpha: f32,
    /// The target's coefficient, a host value.
    pub beta: f32,
}

/// `target ← β · target + α · source`, in place on the target.
///
/// A diffusion solver's Euler step: the latent is scaled by the schedule's coefficient and the
/// model's velocity added at its own, so the two operands may have different element types on
/// purpose (a BF16 accumulator stepped by an FP32 source). The products and the sum are computed in
/// FP32 and stored once into the target's type.
///
/// It is not [`ScaledResidual`]: that one adds a *channel-scaled, optionally biased* update into a
/// running residual stream (`h + (u + b) · scale`, one scale per channel, and no coefficient on the
/// residual itself). Here both coefficients are per-call host scalars and the target's own value is
/// scaled. Neither is the other with a flag set.
pub struct ScaledAdd;

impl Op for ScaledAdd {
    const NAME: &'static str = "scaled_add";

    type Descriptor = ScaledAddDescriptor;
    type Args<'a> = ScaledAddArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let elements = check(descriptor, args)?;
        let mut arguments = Arguments::new();
        arguments
            .view(args.target)
            .view(args.source)
            .f32(args.alpha)
            .f32(args.beta)
            .i32(elements);
        kernel.launch(
            LaunchGrid::covering(u64::from(elements as u32), u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        ScaledAddArgs {
            target: twins.in_place(args.target, stream),
            source: twins.shadow_of(args.source),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The two products and the addition each round once in the sum's type, the store once more
        // into the target's: `2u(target) + u(source)` states the target's roundings and the
        // source's own representation together. The bound is relative to the result, so a call
        // whose operands cancel (β·t ≈ −α·s, a result far below both) is compared against the
        // operands' magnitude by the caller; no per-element bound expresses a cancellation.
        //
        // The corner the bound must not hide: with β = 1 and α = 0 the call is the identity, and
        // with β = 0, α = 1 and equal element types it is a copy — both bit-exact, and both
        // tested. A stricter bound is what an element type's unit round-off buys and nothing more.
        Tolerance::new(
            0.0,
            2.0 * descriptor.target.unit_roundoff() as f32
                + descriptor.source.unit_roundoff() as f32,
        )
    }
}

/// Validate a call and answer its flattened element count.
fn check(descriptor: &ScaledAddDescriptor, args: ScaledAddArgs) -> Result<i32> {
    if descriptor.columns == 0 || descriptor.block == 0 {
        return Err(Error::argument(
            ScaledAdd::NAME,
            "a row or a block of zero elements adds nothing".to_owned(),
        ));
    }
    let columns = u64::from(descriptor.columns);
    args.target.require_rank(ScaledAdd::NAME, 2)?;
    args.target
        .require_row(ScaledAdd::NAME, descriptor.target, columns)?;
    args.source.require_rank(ScaledAdd::NAME, 2)?;
    args.source
        .require_row(ScaledAdd::NAME, descriptor.source, columns)?;
    if args.source.shape != args.target.shape {
        return Err(Error::argument(
            ScaledAdd::NAME,
            "the source does not have the target's shape".to_owned(),
        ));
    }
    let elements = args.target.shape.numel();
    if elements > i32::MAX as u64 {
        return Err(Error::argument(
            ScaledAdd::NAME,
            format!("{elements} elements do not fit the kernel's int32 count"),
        ));
    }
    Ok(elements as i32)
}

#[cfg(test)]
mod scaled_add_tests {
    use super::*;
    use crate::DevicePtr;

    fn view(dtype: DType, rows: u64, columns: u64) -> TensorView {
        TensorView::new(DevicePtr::new(0x1000), dtype, Shape::new(&[rows, columns]))
    }

    fn descriptor(target: DType, source: DType) -> ScaledAddDescriptor {
        ScaledAddDescriptor {
            target,
            source,
            columns: 64,
            block: SCALED_ADD_BLOCK,
        }
    }

    fn args(target: TensorView, source: TensorView) -> ScaledAddArgs {
        ScaledAddArgs {
            target,
            source,
            alpha: 0.5,
            beta: 1.0,
        }
    }

    #[test]
    fn the_key_names_both_element_types_because_a_call_may_mix_them() {
        assert_eq!(
            descriptor(DType::BF16, DType::F32).to_string(),
            "target_bf16_source_f32_columns_64_block_1024"
        );
    }

    #[test]
    fn the_two_operands_must_be_the_shape_the_descriptor_names() {
        let fine = args(view(DType::BF16, 2, 64), view(DType::F32, 2, 64));
        assert_eq!(
            check(&descriptor(DType::BF16, DType::F32), fine).unwrap(),
            128
        );

        let wrong_source = args(view(DType::BF16, 2, 64), view(DType::BF16, 2, 64));
        assert!(
            check(&descriptor(DType::BF16, DType::F32), wrong_source).is_err(),
            "a source of another type than the descriptor names"
        );

        let wrong_shape = args(view(DType::BF16, 2, 64), view(DType::F32, 1, 64));
        assert!(
            check(&descriptor(DType::BF16, DType::F32), wrong_shape).is_err(),
            "a source row count the target does not have"
        );
    }

    #[test]
    fn the_bound_is_the_targets_two_roundings_plus_the_sources_representation() {
        let bound = <ScaledAdd as Op>::tolerance(&descriptor(DType::BF16, DType::F32));
        let expected = 2.0 * DType::BF16.unit_roundoff() as f32 + DType::F32.unit_roundoff() as f32;
        assert_eq!(bound.rtol, expected);
        assert_eq!(bound.atol, 0.0);
    }
}

#[cfg(test)]
mod scale_tests {
    use super::*;
    use crate::DevicePtr;

    fn scale(shape: Shape) -> TensorView {
        TensorView::new(DevicePtr::new(0x1000), DType::BF16, shape)
    }

    #[test]
    fn one_factor_is_a_scale_and_so_is_a_whole_row() {
        assert_eq!(
            scale_width("scaled_residual", &scale(Shape::new(&[1])), DType::BF16, 64).unwrap(),
            1,
            "one element is the broadcast form a per-step table hands over"
        );
        assert_eq!(
            scale_width(
                "scaled_residual",
                &scale(Shape::new(&[64])),
                DType::BF16,
                64
            )
            .unwrap(),
            64,
            "a whole row is the per-channel form"
        );
    }

    #[test]
    fn a_scale_of_any_other_width_is_refused_rather_than_read_past_its_end() {
        for width in [2_u64, 63, 65] {
            assert!(
                scale_width(
                    "scaled_residual",
                    &scale(Shape::new(&[width])),
                    DType::BF16,
                    64
                )
                .is_err(),
                "a {width} element scale is neither one factor nor a row"
            );
        }
    }
}
