//! Normalization over rows: RMS and layer normalization, each with the elementwise steps fused
//! around its row statistics.
//!
//! ```text
//! rms_norm         out = x · rsqrt(mean(x²) + eps) · gain
//! add_rms_norm     sum = x + residual;  out = rms_norm(sum)
//! gated_rms_norm   out = rms_norm(x) · silu(gate)
//! layer_norm       out = (x − mean(x)) · rsqrt(var(x) + eps) · weight + bias
//! ```
//!
//! The gain of an RMS normalization is its weight, or one plus its weight when the checkpoint
//! stores the scale zero-centred (Gemma, Qwen3-Next and Qwen3.5 write `x̂ · (1 + w)`): the one is
//! added in FP32, so no rounded `1 + w` is ever stored. Every statistic and every sum is formed
//! in FP32, and each output is rounded once into its element type.
//!
//! A residual stream is normalized where it is updated, so [`add_rms_norm`] is one launch that
//! reads the row once and returns both tensors: the normalized row a block computes on, and the
//! sum the stream holds from here on. The normalization reads the unrounded sum.

use ops::norm::{
    AddRmsNorm, AddRmsNormArgs, AddRmsNormDescriptor, GatedRmsNorm, GatedRmsNormArgs,
    GatedRmsNormDescriptor, LayerNorm, LayerNormArgs, LayerNormDescriptor, RmsNorm, RmsNormArgs,
    RmsNormDescriptor,
};
use ops::residual::{WeightedRmsNorm, WeightedRmsNormArgs, WeightedRmsNormDescriptor};
use uniserve_tensor::{D, DType, Tensor};

use super::{PDL, as_rows, columns, like_rows, on_cpu, row_tile, same_rows, vector};
use crate::error::{Error, Result};

/// `rows`, FP32 `[rows, columns]`, scaled by the reciprocal root of each row's mean square plus
/// `eps`.
pub(super) fn rms_scaled(rows: &Tensor, eps: f32) -> Result<Tensor> {
    let mean_square = rows.sqr()?.mean_keepdim(D::Minus1)?;
    let scale = (mean_square + f64::from(eps))?.sqrt()?.recip()?;

    Ok(rows.broadcast_mul(&scale)?)
}

/// The FP32 gain a row is multiplied by: the weight, or one plus it when `centred`.
fn gain(weight: &Tensor, centred: bool) -> Result<Tensor> {
    let weight = weight.to_dtype(DType::F32)?;

    if centred {
        Ok((weight + 1.0)?)
    } else {
        Ok(weight)
    }
}

/// RMS normalization of the rows of `x` with the per-channel scale `weight`:
/// `x · rsqrt(mean(x²) + eps) · gain`, where the gain is the weight, or one plus it when the
/// checkpoint stores it zero-centred (`centred`).
///
/// The result has `x`'s shape and `output` elements: `x`'s own type, or a narrower activation
/// type that the rows reading the result are computed in, in which case the normalization
/// narrows as it writes (`weighted_rms_norm`) and rounds once.
///
/// # Errors
///
/// When `weight` is not one vector of `x`'s row width, a zero-centred gain is asked with a
/// narrowing output (no kernel reads both), or the operator refuses its arguments.
pub fn rms_norm(
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
    centred: bool,
    output: DType,
) -> Result<Tensor> {
    const OWNER: &str = "rms_norm";

    let width = columns(OWNER, x)?;
    vector(OWNER, "weight", weight, width)?;
    if centred && output != x.dtype() {
        return Err(Error::shape(
            OWNER,
            "a zero-centred gain narrows no output type".to_owned(),
        ));
    }

    if on_cpu(x) {
        let normalized = rms_scaled(&x.to_dtype(DType::F32)?, eps)?;
        return Ok(normalized
            .broadcast_mul(&gain(weight, centred)?)?
            .to_dtype(output)?);
    }

    let rows = as_rows(OWNER, x)?;
    let out = backend::empty(rows.shape(), output)?;
    let tile = row_tile(width);

    if output == x.dtype() {
        backend::launch::<RmsNorm>(
            &RmsNormDescriptor {
                dtype: x.dtype(),
                columns: width,
                tile,
                pdl: PDL,
                centred,
            },
            RmsNormArgs {
                x: backend::operand(&rows)?,
                weight: backend::operand(weight)?,
                out: backend::operand(&out)?,
                eps,
            },
        )?;
    } else {
        backend::launch::<WeightedRmsNorm>(
            &WeightedRmsNormDescriptor {
                dtype: x.dtype(),
                columns: width,
                tile,
                absmax: false,
                activation: output,
            },
            WeightedRmsNormArgs {
                x: backend::operand(&rows)?,
                weight: backend::operand(weight)?,
                out: backend::operand(&out)?,
                eps,
                partials: None,
            },
        )?;
    }
    like_rows(out, x)
}

/// The residual add and the RMS normalization of its sum, as one operator: returns
/// `(normalized, sum)`, where `sum = x + residual` is what the residual stream holds from here
/// on and `normalized = rms_norm(sum)`.
///
/// The sum is formed in FP32; the stream stores it rounded once, and the normalization reads the
/// unrounded sum. Both results have `x`'s shape and element type.
///
/// # Errors
///
/// When `residual` is not `x`'s shape and element type, `weight` is not one vector of the row
/// width, or the operator refuses its arguments.
pub fn add_rms_norm(
    x: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
    centred: bool,
) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = "add_rms_norm";

    let width = columns(OWNER, x)?;
    vector(OWNER, "weight", weight, width)?;
    same_rows(OWNER, "residual stream", residual, x)?;

    if on_cpu(x) {
        let sum = (x.to_dtype(DType::F32)? + residual.to_dtype(DType::F32)?)?;
        let normalized = rms_scaled(&sum, eps)?.broadcast_mul(&gain(weight, centred)?)?;

        return Ok((normalized.to_dtype(x.dtype())?, sum.to_dtype(x.dtype())?));
    }

    let rows = as_rows(OWNER, x)?;
    let residual_rows = as_rows(OWNER, residual)?;
    let out = backend::empty(rows.shape(), x.dtype())?;
    let sum = backend::empty(rows.shape(), x.dtype())?;

    backend::launch::<AddRmsNorm>(
        &AddRmsNormDescriptor {
            dtype: x.dtype(),
            columns: width,
            tile: row_tile(width),
            pdl: PDL,
            centred,
        },
        AddRmsNormArgs {
            x: backend::operand(&rows)?,
            residual: backend::operand(&residual_rows)?,
            weight: backend::operand(weight)?,
            out: backend::operand(&out)?,
            residual_out: backend::operand(&sum)?,
            eps,
        },
    )?;
    Ok((like_rows(out, x)?, like_rows(sum, x)?))
}

/// RMS normalization gated by the SiLU of a second input:
/// `x · rsqrt(mean(x²) + eps) · weight · silu(gate)`, the output norm of a gated delta network.
///
/// The row is normalized before it is gated, in FP32, and rounds once into `x`'s element type.
///
/// # Errors
///
/// When `gate` is not `x`'s shape and element type, `weight` is not one vector of the row
/// width, or the operator refuses its arguments.
pub fn gated_rms_norm(x: &Tensor, gate: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    const OWNER: &str = "gated_rms_norm";

    let width = columns(OWNER, x)?;
    vector(OWNER, "weight", weight, width)?;
    same_rows(OWNER, "gate", gate, x)?;

    if on_cpu(x) {
        let normalized =
            rms_scaled(&x.to_dtype(DType::F32)?, eps)?.broadcast_mul(&gain(weight, false)?)?;
        let gated = (normalized * gate.to_dtype(DType::F32)?.silu()?)?;

        return Ok(gated.to_dtype(x.dtype())?);
    }

    let rows = as_rows(OWNER, x)?;
    let gate_rows = as_rows(OWNER, gate)?;
    let out = backend::empty(rows.shape(), x.dtype())?;

    backend::launch::<GatedRmsNorm>(
        &GatedRmsNormDescriptor {
            dtype: x.dtype(),
            columns: width,
            tile: row_tile(width),
        },
        GatedRmsNormArgs {
            x: backend::operand(&rows)?,
            gate: backend::operand(&gate_rows)?,
            weight: backend::operand(weight)?,
            out: backend::operand(&out)?,
            eps,
        },
    )?;
    like_rows(out, x)
}

/// Layer normalization of the rows of `x` with a per-channel scale and an optional bias:
/// `(x − mean(x)) · rsqrt(var(x) + eps) · weight + bias`, the variance the mean of the squared
/// deviations. The result has `x`'s shape and element type.
///
/// # Errors
///
/// When `weight` or `bias` is not one vector of `x`'s row width, or the operator refuses its
/// arguments.
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    const OWNER: &str = "layer_norm";

    let width = columns(OWNER, x)?;
    vector(OWNER, "weight", weight, width)?;
    if let Some(bias) = bias {
        vector(OWNER, "bias", bias, width)?;
    }

    if on_cpu(x) {
        let rows = x.to_dtype(DType::F32)?;
        let centred = rows.broadcast_sub(&rows.mean_keepdim(D::Minus1)?)?;
        let variance = centred.sqr()?.mean_keepdim(D::Minus1)?;
        let scale = (variance + f64::from(eps))?.sqrt()?.recip()?;

        let mut normalized = centred
            .broadcast_mul(&scale)?
            .broadcast_mul(&weight.to_dtype(DType::F32)?)?;
        if let Some(bias) = bias {
            normalized = normalized.broadcast_add(&bias.to_dtype(DType::F32)?)?;
        }
        return Ok(normalized.to_dtype(x.dtype())?);
    }

    let rows = as_rows(OWNER, x)?;
    let out = backend::empty(rows.shape(), x.dtype())?;

    backend::launch::<LayerNorm>(
        // The bias is in the descriptor, so a normalization with one and one without are two
        // kernels: the epilogue loads a second vector or it does not.
        &LayerNormDescriptor {
            dtype: x.dtype(),
            columns: width,
            tile: row_tile(width),
            bias: bias.is_some(),
            pdl: PDL,
        },
        LayerNormArgs {
            x: backend::operand(&rows)?,
            weight: backend::operand(weight)?,
            bias: bias.map(backend::operand).transpose()?,
            out: backend::operand(&out)?,
            eps,
        },
    )?;
    like_rows(out, x)
}
