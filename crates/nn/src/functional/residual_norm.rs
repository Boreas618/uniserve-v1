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

use ops::descriptor::Float;
use ops::modulation::{self, ModulatedRmsNorm, ModulatedRmsNormDescriptor};
use ops::residual::{
    ScaledResidualLayerNorm, ScaledResidualLayerNormArgs, ScaledResidualLayerNormDescriptor,
    ScaledResidualRmsNorm, ScaledResidualRmsNormArgs, ScaledResidualRmsNormDescriptor,
};
use uniserve_tensor::{D, DType, Tensor};

use super::norm::rms_scaled;
use super::{as_rows, columns, like_rows, on_cpu, row_tile, vector};
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
    const OWNER: &str = "scaled_residual_rms_norm_";

    let width = columns(OWNER, &hidden)?;
    vector(OWNER, "weight", weight, width)?;
    check_scaled(OWNER, &hidden, update, update_bias, scale, width)?;
    written(OWNER, "residual stream", &hidden)?;

    if on_cpu(&hidden) {
        let sum = scaled_sum(&hidden, update, update_bias, scale)?;
        let normalized = rms_scaled(&sum, eps)?.broadcast_mul(&weight.to_dtype(DType::F32)?)?;

        hidden.slice_set(&sum.to_dtype(hidden.dtype())?, 0, 0)?;
        return Ok((normalized.to_dtype(update.dtype())?, hidden));
    }

    let (rows, update_rows) = (as_rows(OWNER, &hidden)?, as_rows(OWNER, update)?);
    let out = backend::empty(rows.shape(), update.dtype())?;

    backend::launch::<ScaledResidualRmsNorm>(
        &ScaledResidualRmsNormDescriptor {
            dtype: hidden.dtype(),
            columns: width,
            tile: row_tile(width),
            update_bias: update_bias.is_some(),
            absmax: false,
            activation: update.dtype(),
        },
        ScaledResidualRmsNormArgs {
            hidden: backend::operand(&rows)?,
            update: backend::operand(&update_rows)?,
            update_bias: update_bias.map(backend::operand).transpose()?,
            scale: backend::operand(scale)?,
            weight: backend::operand(weight)?,
            out: backend::operand(&out)?,
            eps,
            partials: None,
        },
    )?;
    Ok((like_rows(out, &hidden)?, hidden))
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
    const OWNER: &str = "scaled_residual_layer_norm";

    let width = columns(OWNER, hidden)?;
    vector(OWNER, "weight", weight, width)?;
    vector(OWNER, "bias", bias, width)?;
    check_scaled(OWNER, hidden, update, update_bias, scale, width)?;

    if on_cpu(hidden) {
        let sum = scaled_sum(hidden, update, update_bias, scale)?;
        let centred = sum.broadcast_sub(&sum.mean_keepdim(D::Minus1)?)?;
        let variance = centred.sqr()?.mean_keepdim(D::Minus1)?;
        let deviation = (variance + f64::from(eps))?.sqrt()?.recip()?;

        let normalized = centred
            .broadcast_mul(&deviation)?
            .broadcast_mul(&weight.to_dtype(DType::F32)?)?
            .broadcast_add(&bias.to_dtype(DType::F32)?)?;
        return Ok(normalized.to_dtype(update.dtype())?);
    }

    let (rows, update_rows) = (as_rows(OWNER, hidden)?, as_rows(OWNER, update)?);
    let out = backend::empty(rows.shape(), update.dtype())?;

    backend::launch::<ScaledResidualLayerNorm>(
        &ScaledResidualLayerNormDescriptor {
            dtype: hidden.dtype(),
            columns: width,
            tile: row_tile(width),
            update_bias: update_bias.is_some(),
            absmax: false,
            activation: update.dtype(),
        },
        ScaledResidualLayerNormArgs {
            hidden: backend::operand(&rows)?,
            update: backend::operand(&update_rows)?,
            update_bias: update_bias.map(backend::operand).transpose()?,
            scale: backend::operand(scale)?,
            weight: backend::operand(weight)?,
            bias: backend::operand(bias)?,
            out: backend::operand(&out)?,
            eps,
            partials: None,
        },
    )?;
    like_rows(out, hidden)
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

/// The leading stride of `table`, a `[parameters, width]` view, in elements: where the kernel
/// finds the next parameter row.
fn row_stride(table: &ops::Operand) -> u64 {
    table.strides.get(0)
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
    const OWNER: &str = "modulated_rms_norm";

    let width = columns(OWNER, x)?;
    vector(OWNER, "weight", weight, width)?;
    check_modulation(OWNER, x, &[("shift", shift), ("scale", scale)], rows, width)?;

    if on_cpu(x) {
        let value = x.to_dtype(DType::F32)?;
        let modulated = modulate(&value, weight, shift, scale, rows, eps, x.dtype())?;

        return Ok(modulated.to_dtype(output)?);
    }

    let value = x.contiguous()?;
    let out = backend::empty(x.shape(), output)?;
    let (shift, scale) = (backend::operand(shift)?, backend::operand(scale)?);

    backend::launch::<ModulatedRmsNorm>(
        &ModulatedRmsNormDescriptor {
            dtype: x.dtype(),
            output,
            indices: rows.dtype(),
            width,
            block: row_tile(width),
            epsilon: Float::from(eps),
            gate_stride: 0,
            shift_stride: row_stride(&shift),
            scale_stride: row_stride(&scale),
            update: false,
            retain: false,
        },
        modulation::Args {
            value: backend::operand(&value)?,
            update: None,
            gate: None,
            weight: backend::operand(weight)?,
            shift,
            scale,
            row_indices: backend::operand(rows)?,
            out: backend::operand(&out)?,
            output_scale: None,
            retain: None,
        },
    )?;
    Ok(out)
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
    const OWNER: &str = "modulated_rms_norm_";

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

    if on_cpu(&update) {
        let gate = parameter_rows(gate, rows)?;
        let gated = rounded(&(gate * update.to_dtype(DType::F32)?)?, activation)?;
        let sum = rounded(&(hidden.to_dtype(DType::F32)? + gated)?, activation)?;
        let modulated = modulate(&sum, weight, shift, scale, rows, eps, activation)?;

        update.slice_set(&sum.to_dtype(activation)?, 0, 0)?;
        return Ok((modulated.to_dtype(output)?, sum.to_dtype(output)?));
    }

    let value = hidden.contiguous()?;
    let out = backend::empty(update.shape(), output)?;
    let sum = backend::empty(update.shape(), output)?;
    let (shift, scale, gate) = (
        backend::operand(shift)?,
        backend::operand(scale)?,
        backend::operand(gate)?,
    );

    backend::launch::<ModulatedRmsNorm>(
        &ModulatedRmsNormDescriptor {
            dtype: activation,
            output,
            indices: rows.dtype(),
            width,
            block: row_tile(width),
            epsilon: Float::from(eps),
            gate_stride: row_stride(&gate),
            shift_stride: row_stride(&shift),
            scale_stride: row_stride(&scale),
            update: true,
            retain: true,
        },
        modulation::Args {
            value: backend::operand(&value)?,
            update: Some(backend::operand(&update)?),
            gate: Some(gate),
            weight: backend::operand(weight)?,
            shift,
            scale,
            row_indices: backend::operand(rows)?,
            out: backend::operand(&out)?,
            output_scale: None,
            retain: Some(backend::operand(&sum)?),
        },
    )?;
    Ok((out, sum))
}
