//! Normalizations fused with the update of the residual stream they normalize: the
//! channel-scaled update of a layer-scale block, and the per-row modulation of a diffusion block.
//!
//! ```text
//! scaled_residual_rms_norm_    h = h + (u + b) · scale;  out = rms_norm(h)
//! scaled_residual_layer_norm   out = layer_norm(h + (u + b) · scale)
//! modulated_rms_norm           out = rms_norm(x) · (1 + scale[i]) + shift[i]
//! modulated_rms_norm_          s = h + gate[i] · u;  out = rms_norm(s) · (1 + scale[i]) + shift[i]
//! ```
//!
//! A block adds a projection's output into its residual stream and normalizes the sum for the
//! next block, and each of these is that pair as one launch that reads the stream once. `i` is a
//! row's parameter row: one table of modulation vectors serves the rows of several timesteps.
//!
//! ## In-place forms
//!
//! A function whose name ends in an underscore writes one of its operands, as PyTorch's do: the
//! kernel reads and writes that tensor, so the function takes it by value and returns what it
//! holds afterwards. [`scaled_residual_rms_norm_`] updates the stream; [`modulated_rms_norm_`]
//! overwrites the update with the sum.
//!
//! ## Rounding
//!
//! The scaled-residual forms keep the sum and the statistics in FP32: the stream stores the sum
//! rounded once and the normalization reads the unrounded sum. The modulated forms round every
//! intermediate into the activation's element type, as the eager expression they replace does,
//! and only the mean square accumulates in FP32.

use ops::modulation::{self, ModulatedRmsNorm, ModulatedRmsNormDescriptor};
use ops::residual::{
    ScaledResidualLayerNorm, ScaledResidualLayerNormArgs, ScaledResidualLayerNormDescriptor,
    ScaledResidualRmsNorm, ScaledResidualRmsNormArgs, ScaledResidualRmsNormDescriptor,
};
use tensor::{D, DType, Device, Tensor};

use super::norm::rms_scaled;
use super::{Fused, as_rows, columns, compute, like_rows, vector};
use crate::error::{Error, Result};

/// Checks the operands a scaled-residual normalization shares: an `update` of the stream's
/// shape, the stream's vectors in the stream's element type, and a `scale` of one element per
/// channel or of one element for the whole stream.
fn check_scaled(
    owner: &'static str,
    hidden: &Tensor,
    update: &Tensor,
    update_bias: Option<&Tensor>,
    scale: &Tensor,
    width: u32,
) -> Result<()> {
    if update.dims() != hidden.dims() {
        return Err(Error::shape(
            owner,
            format!(
                "an update of {:?} for a residual stream of {:?}",
                update.dims(),
                hidden.dims()
            ),
        ));
    }
    if let Some(bias) = update_bias {
        vector(owner, "update's bias", bias, width)?;
    }
    if scale.dims() != [1] {
        vector(owner, "residual's scale", scale, width)?;
    }
    Ok(())
}

/// The FP32 sum `hidden + (update + update_bias) · scale` of CPU tensors.
fn scaled_sum(
    hidden: &Tensor,
    update: &Tensor,
    update_bias: Option<&Tensor>,
    scale: &Tensor,
) -> Result<Tensor> {
    let mut update = update.to_dtype(DType::F32)?;
    if let Some(bias) = update_bias {
        update = update.broadcast_add(&bias.to_dtype(DType::F32)?)?;
    }
    let scaled = update.broadcast_mul(&scale.to_dtype(DType::F32)?)?;

    Ok((hidden.to_dtype(DType::F32)? + scaled)?)
}

/// Refuses a strided view as the operand `name` a kernel writes in place: a copy of it would
/// receive the write, and the caller's tensor would not.
fn written(owner: &'static str, name: &str, tensor: &Tensor) -> Result<()> {
    if !tensor.is_contiguous() {
        return Err(Error::shape(
            owner,
            format!("the {name} is written in place, and a strided view of it is not its rows"),
        ));
    }
    Ok(())
}

/// A channel-scaled update folded into the residual stream, and the RMS normalization of the
/// sum, as one operator: `hidden = hidden + (update + update_bias) · scale` in place, and
/// `rms_norm(hidden) · weight`. Returns `(normalized, hidden)`.
///
/// `hidden` and `update` are `[..., width]`; `scale` is one layer scale per channel, or one
/// element for the whole stream; `update_bias` is the bias a projection left out. The stream,
/// its scale, the bias and the weight share one element type; the update and the normalized rows
/// share another, which may be narrower. The stream stores the sum rounded once and the
/// normalization reads the unrounded sum.
///
/// # Errors
///
/// When the update is not the stream's shape, a vector is not one element per channel, the
/// stream is a strided view, or the operator refuses its arguments.
pub fn scaled_residual_rms_norm_(
    hidden: Tensor,
    update: &Tensor,
    update_bias: Option<&Tensor>,
    scale: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = ScaledResidualRmsNormCall::OWNER;

    let width = columns(OWNER, &hidden)?;
    vector(OWNER, "weight", weight, width)?;
    check_scaled(OWNER, &hidden, update, update_bias, scale, width)?;
    written(OWNER, "residual stream", &hidden)?;

    compute(ScaledResidualRmsNormCall {
        hidden,
        update,
        update_bias,
        scale,
        weight,
        eps,
    })
}

/// A checked call of [`scaled_residual_rms_norm_`].
struct ScaledResidualRmsNormCall<'t> {
    hidden: Tensor,
    update: &'t Tensor,
    update_bias: Option<&'t Tensor>,
    scale: &'t Tensor,
    weight: &'t Tensor,
    eps: f32,
}

impl ScaledResidualRmsNormCall<'_> {
    const OWNER: &'static str = "scaled_residual_rms_norm_";
}

impl Fused for ScaledResidualRmsNormCall<'_> {
    type Output = (Tensor, Tensor);

    fn device(&self) -> &Device {
        self.hidden.device()
    }

    fn cpu(self) -> Result<(Tensor, Tensor)> {
        let Self {
            hidden,
            update,
            weight,
            ..
        } = self;
        let sum = scaled_sum(&hidden, update, self.update_bias, self.scale)?;
        let normalized =
            rms_scaled(&sum, self.eps)?.broadcast_mul(&weight.to_dtype(DType::F32)?)?;

        hidden.slice_set(&sum.to_dtype(hidden.dtype())?, 0, 0)?;
        Ok((normalized.to_dtype(update.dtype())?, hidden))
    }

    fn launch(self) -> Result<(Tensor, Tensor)> {
        let Self {
            hidden,
            update,
            update_bias,
            ..
        } = self;
        let width = columns(Self::OWNER, &hidden)?;
        let (rows, update_rows) = (
            as_rows(Self::OWNER, &hidden)?,
            as_rows(Self::OWNER, update)?,
        );
        let out = ops::empty(rows.shape(), update.dtype())?;

        ops::launch::<ScaledResidualRmsNorm>(
            &ScaledResidualRmsNormDescriptor::new(
                hidden.dtype(),
                width,
                update_bias.is_some(),
                update.dtype(),
                false,
            ),
            ScaledResidualRmsNormArgs {
                hidden: ops::operand(&rows)?,
                update: ops::operand(&update_rows)?,
                update_bias: update_bias.map(ops::operand).transpose()?,
                scale: ops::operand(self.scale)?,
                weight: ops::operand(self.weight)?,
                out: ops::operand(&out)?,
                eps: self.eps,
                partials: None,
            },
        )?;
        Ok((like_rows(out, &hidden)?, hidden))
    }
}

/// The layer normalization of a residual stream with a channel-scaled update folded in:
/// `layer_norm(hidden + (update + update_bias) · scale) · weight + bias`, in `update`'s element
/// type. The stream itself is not written: a block that reads the normalized sum alone keeps
/// its stream as it was.
///
/// The operands are those of [`scaled_residual_rms_norm_`], with the normalization's `bias`.
///
/// # Errors
///
/// When the update is not the stream's shape, a vector is not one element per channel, or the
/// operator refuses its arguments.
pub fn scaled_residual_layer_norm(
    hidden: &Tensor,
    update: &Tensor,
    update_bias: Option<&Tensor>,
    scale: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    const OWNER: &str = ScaledResidualLayerNormCall::OWNER;

    let width = columns(OWNER, hidden)?;
    vector(OWNER, "weight", weight, width)?;
    vector(OWNER, "bias", bias, width)?;
    check_scaled(OWNER, hidden, update, update_bias, scale, width)?;

    compute(ScaledResidualLayerNormCall {
        hidden,
        update,
        update_bias,
        scale,
        weight,
        bias,
        eps,
    })
}

/// A checked call of [`scaled_residual_layer_norm`].
struct ScaledResidualLayerNormCall<'t> {
    hidden: &'t Tensor,
    update: &'t Tensor,
    update_bias: Option<&'t Tensor>,
    scale: &'t Tensor,
    weight: &'t Tensor,
    bias: &'t Tensor,
    eps: f32,
}

impl ScaledResidualLayerNormCall<'_> {
    const OWNER: &'static str = "scaled_residual_layer_norm";
}

impl Fused for ScaledResidualLayerNormCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.hidden.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let sum = scaled_sum(self.hidden, self.update, self.update_bias, self.scale)?;
        let centred = sum.broadcast_sub(&sum.mean_keepdim(D::Minus1)?)?;
        let variance = centred.sqr()?.mean_keepdim(D::Minus1)?;
        let deviation = (variance + f64::from(self.eps))?.sqrt()?.recip()?;

        let normalized = centred
            .broadcast_mul(&deviation)?
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?
            .broadcast_add(&self.bias.to_dtype(DType::F32)?)?;
        Ok(normalized.to_dtype(self.update.dtype())?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self {
            hidden,
            update,
            update_bias,
            ..
        } = self;
        let width = columns(Self::OWNER, hidden)?;
        let (rows, update_rows) = (as_rows(Self::OWNER, hidden)?, as_rows(Self::OWNER, update)?);
        let out = ops::empty(rows.shape(), update.dtype())?;

        ops::launch::<ScaledResidualLayerNorm>(
            &ScaledResidualLayerNormDescriptor::new(
                hidden.dtype(),
                width,
                update_bias.is_some(),
                update.dtype(),
                false,
            ),
            ScaledResidualLayerNormArgs {
                hidden: ops::operand(&rows)?,
                update: ops::operand(&update_rows)?,
                update_bias: update_bias.map(ops::operand).transpose()?,
                scale: ops::operand(self.scale)?,
                weight: ops::operand(self.weight)?,
                bias: ops::operand(self.bias)?,
                out: ops::operand(&out)?,
                eps: self.eps,
                partials: None,
            },
        )?;
        like_rows(out, hidden)
    }
}

/// Checks a modulation's operands against the `[rows, width]` activation they modulate: tables
/// of `[parameters, width]` vectors and one `U32` parameter row per activation row.
fn check_modulation(
    owner: &'static str,
    x: &Tensor,
    tables: &[(&str, &Tensor)],
    rows: &Tensor,
    width: u32,
) -> Result<()> {
    let &[count, _] = x.dims() else {
        return Err(Error::shape(
            owner,
            format!("modulated rows are [rows, width], not {:?}", x.dims()),
        ));
    };

    for (name, table) in tables {
        if table.rank() != 2 || table.dims()[1] != width as usize {
            return Err(Error::shape(
                owner,
                format!(
                    "a {name} table is [parameters, {width}], not {:?}",
                    table.dims()
                ),
            ));
        }
    }

    if rows.dims() != [count] || rows.dtype() != DType::U32 {
        return Err(Error::shape(
            owner,
            format!(
                "the parameter rows are {} {:?}, and {count} activation rows take one u32 each",
                ops::abi::name(rows.dtype()),
                rows.dims()
            ),
        ));
    }
    Ok(())
}

/// `x` rounded into `dtype` and widened back: one rounding point of an FP32 formula.
fn rounded(x: &Tensor, dtype: DType) -> Result<Tensor> {
    Ok(x.to_dtype(dtype)?.to_dtype(DType::F32)?)
}

/// The FP32 row of the CPU `table`, `[parameters, width]`, each activation row reads:
/// `[rows, width]`. The CPU gather reads packed rows, so a column window is packed first.
fn parameter_rows(table: &Tensor, rows: &Tensor) -> Result<Tensor> {
    Ok(table
        .contiguous()?
        .index_select(rows, 0)?
        .to_dtype(DType::F32)?)
}

/// The modulated normalization of `value`, FP32 `[rows, width]` whose elements are values of
/// `activation`: `rms_norm(value) · weight`, times one plus the row's scale, plus the row's
/// shift, each step rounded into `activation`.
fn modulate(
    value: &Tensor,
    weight: &Tensor,
    shift: &Tensor,
    scale: &Tensor,
    rows: &Tensor,
    eps: f32,
    activation: DType,
) -> Result<Tensor> {
    let weighted = rms_scaled(value, eps)?.broadcast_mul(&weight.to_dtype(DType::F32)?)?;
    let normalized = rounded(&weighted, activation)?;

    let scale = parameter_rows(scale, rows)?;
    let factor = rounded(&(scale + 1.0)?, activation)?;
    let scaled = rounded(&(normalized * factor)?, activation)?;

    let shift = parameter_rows(shift, rows)?;
    rounded(&(scaled + shift)?, activation)
}

/// The per-row modulation of an RMS normalization:
/// `rms_norm(x) · weight · (1 + scale[i]) + shift[i]`, where `i = rows[row]`.
///
/// `x` is `[rows, width]`; `shift` and `scale` are `[parameters, width]` tables whose rows may
/// be strided views of one wider table (a column window of the step's gathered modulation);
/// `rows` is one `U32` parameter row per activation row. The result is `[rows, width]` in
/// `output` elements, `x`'s type or another floating type.
///
/// # Errors
///
/// When a table is not `[parameters, width]`, `rows` is not one `U32` per activation row,
/// `weight` is not one vector of the row width, or the operator refuses its arguments.
pub fn modulated_rms_norm(
    x: &Tensor,
    weight: &Tensor,
    shift: &Tensor,
    scale: &Tensor,
    rows: &Tensor,
    eps: f32,
    output: DType,
) -> Result<Tensor> {
    const OWNER: &str = ModulatedRmsNormCall::OWNER;

    let width = columns(OWNER, x)?;
    vector(OWNER, "weight", weight, width)?;
    check_modulation(OWNER, x, &[("shift", shift), ("scale", scale)], rows, width)?;

    compute(ModulatedRmsNormCall {
        x,
        weight,
        shift,
        scale,
        rows,
        eps,
        output,
    })
}

/// A checked call of [`modulated_rms_norm`].
struct ModulatedRmsNormCall<'t> {
    x: &'t Tensor,
    weight: &'t Tensor,
    shift: &'t Tensor,
    scale: &'t Tensor,
    rows: &'t Tensor,
    eps: f32,
    output: DType,
}

impl ModulatedRmsNormCall<'_> {
    const OWNER: &'static str = "modulated_rms_norm";
}

impl Fused for ModulatedRmsNormCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.x.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self { x, .. } = self;
        let value = x.to_dtype(DType::F32)?;
        let modulated = modulate(
            &value,
            self.weight,
            self.shift,
            self.scale,
            self.rows,
            self.eps,
            x.dtype(),
        )?;

        Ok(modulated.to_dtype(self.output)?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self { x, .. } = self;
        let value = x.contiguous()?;
        let out = ops::empty(x.shape(), self.output)?;
        let args = modulation::Args {
            value: ops::operand(&value)?,
            update: None,
            gate: None,
            weight: ops::operand(self.weight)?,
            shift: ops::operand(self.shift)?,
            scale: ops::operand(self.scale)?,
            row_indices: ops::operand(self.rows)?,
            out: ops::operand(&out)?,
            output_scale: None,
            retain: None,
        };
        ops::launch::<ModulatedRmsNorm>(&ModulatedRmsNormDescriptor::of(&args, self.eps)?, args)?;
        Ok(out)
    }
}

/// A gated update folded into the residual stream and the modulated normalization of the sum,
/// as one operator: `sum = hidden + gate[i] · update`, and
/// `rms_norm(sum) · weight · (1 + scale[i]) + shift[i]`. Returns `(normalized, sum)`.
///
/// `update` is the block's `[rows, width]` output and is spent: the kernel stores the sum over
/// it. The returned sum is the stream from here on, in `output` elements like the normalized
/// rows; `hidden` is not written. The gated update rounds, then the sum rounds, and the
/// normalization reads that rounded sum. The other operands are those of
/// [`modulated_rms_norm`], with `gate` a `[parameters, width]` table like the shift and scale.
///
/// # Errors
///
/// As [`modulated_rms_norm`], and when `hidden` is not `update`'s shape and element type or
/// `update` is a strided view.
#[expect(
    clippy::too_many_arguments,
    reason = "the stream, its update and gate beside the modulated normalization's operands"
)]
pub fn modulated_rms_norm_(
    update: Tensor,
    hidden: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    shift: &Tensor,
    scale: &Tensor,
    rows: &Tensor,
    eps: f32,
    output: DType,
) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = ModulatedRmsNormUpdateCall::OWNER;

    let width = columns(OWNER, &update)?;
    let activation = update.dtype();
    vector(OWNER, "weight", weight, width)?;
    check_modulation(
        OWNER,
        &update,
        &[("shift", shift), ("scale", scale), ("gate", gate)],
        rows,
        width,
    )?;
    if hidden.dims() != update.dims() || hidden.dtype() != activation {
        return Err(Error::shape(
            OWNER,
            format!(
                "a residual stream of {} {:?} for an update of {} {:?}",
                ops::abi::name(hidden.dtype()),
                hidden.dims(),
                ops::abi::name(activation),
                update.dims()
            ),
        ));
    }
    written(OWNER, "update", &update)?;

    compute(ModulatedRmsNormUpdateCall {
        update,
        hidden,
        gate,
        weight,
        shift,
        scale,
        rows,
        eps,
        output,
    })
}

/// A checked call of [`modulated_rms_norm_`].
struct ModulatedRmsNormUpdateCall<'t> {
    update: Tensor,
    hidden: &'t Tensor,
    gate: &'t Tensor,
    weight: &'t Tensor,
    shift: &'t Tensor,
    scale: &'t Tensor,
    rows: &'t Tensor,
    eps: f32,
    output: DType,
}

impl ModulatedRmsNormUpdateCall<'_> {
    const OWNER: &'static str = "modulated_rms_norm_";
}

impl Fused for ModulatedRmsNormUpdateCall<'_> {
    type Output = (Tensor, Tensor);

    fn device(&self) -> &Device {
        self.update.device()
    }

    fn cpu(self) -> Result<(Tensor, Tensor)> {
        let Self {
            update,
            hidden,
            rows,
            ..
        } = self;
        let activation = update.dtype();
        let gate = parameter_rows(self.gate, rows)?;
        let gated = rounded(&(gate * update.to_dtype(DType::F32)?)?, activation)?;
        let sum = rounded(&(hidden.to_dtype(DType::F32)? + gated)?, activation)?;
        let modulated = modulate(
            &sum,
            self.weight,
            self.shift,
            self.scale,
            rows,
            self.eps,
            activation,
        )?;

        update.slice_set(&sum.to_dtype(activation)?, 0, 0)?;
        Ok((modulated.to_dtype(self.output)?, sum.to_dtype(self.output)?))
    }

    fn launch(self) -> Result<(Tensor, Tensor)> {
        let Self { update, hidden, .. } = self;
        let value = hidden.contiguous()?;
        let out = ops::empty(update.shape(), self.output)?;
        let sum = ops::empty(update.shape(), self.output)?;
        let args = modulation::Args {
            value: ops::operand(&value)?,
            update: Some(ops::operand(&update)?),
            gate: Some(ops::operand(self.gate)?),
            weight: ops::operand(self.weight)?,
            shift: ops::operand(self.shift)?,
            scale: ops::operand(self.scale)?,
            row_indices: ops::operand(self.rows)?,
            out: ops::operand(&out)?,
            output_scale: None,
            retain: Some(ops::operand(&sum)?),
        };
        ops::launch::<ModulatedRmsNorm>(&ModulatedRmsNormDescriptor::of(&args, self.eps)?, args)?;
        Ok((out, sum))
    }
}
