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
use tensor::{D, DType, Device, Tensor};

use super::{Fused, as_rows, columns, compute, like_rows, same_rows, vector};
use crate::error::{Error, Result};

/// `rows`, FP32 `[..., columns]`, scaled by the reciprocal root of each row's mean square plus
/// `eps`.
pub(super) fn rms_scaled(rows: &Tensor, eps: f32) -> Result<Tensor> {
    let mean_square = rows.sqr()?.mean_keepdim(D::Minus1)?;
    let scale = (mean_square + f64::from(eps))?.sqrt()?.recip()?;

    Ok(rows.broadcast_mul(&scale)?)
}

/// The FP32 gain a row is multiplied by: the weight, or one plus it when `centred`.
pub(super) fn gain(weight: &Tensor, centred: bool) -> Result<Tensor> {
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
/// `weight` is one vector of the row width, or `groups` of them one after another: row `i` of
/// `x` is then scaled by gain row `i mod groups`, so the rows are whole groups (the streams of
/// a hyper-connected residual, each normalized with its own gain).
///
/// The result has `x`'s shape and `output` elements: `x`'s own type, or a narrower activation
/// type that the rows reading the result are computed in, in which case the normalization
/// narrows as it writes (`weighted_rms_norm`) and rounds once.
///
/// # Errors
///
/// When `weight` is not whole vectors of `x`'s row width whose count divides the rows, a
/// zero-centred or grouped gain is asked with a narrowing output (no kernel reads both), or the
/// operator refuses its arguments.
pub fn rms_norm(
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
    centred: bool,
    output: DType,
) -> Result<Tensor> {
    const OWNER: &str = RmsNormCall::OWNER;

    let width = columns(OWNER, x)?;
    let row_count = x.elem_count() / width as usize;
    let groups = match weight.dims() {
        &[gain] if gain > 0 && gain.is_multiple_of(width as usize) => gain / width as usize,
        _ => 0,
    };
    if groups == 0 || !row_count.is_multiple_of(groups) {
        return Err(Error::shape(
            OWNER,
            format!(
                "the weight is {:?}, and {row_count} rows of {width} elements take whole \
                 vectors of {width} whose count divides the rows",
                weight.dims()
            ),
        ));
    }
    if (centred || groups > 1) && output != x.dtype() {
        return Err(Error::shape(
            OWNER,
            "a zero-centred or grouped gain narrows no output type".to_owned(),
        ));
    }

    compute(RmsNormCall {
        x,
        weight,
        eps,
        centred,
        groups,
        output,
    })
}

/// A checked call of [`rms_norm`]: `weight` holds `groups` gain rows.
struct RmsNormCall<'t> {
    x: &'t Tensor,
    weight: &'t Tensor,
    eps: f32,
    centred: bool,
    groups: usize,
    output: DType,
}

impl RmsNormCall<'_> {
    const OWNER: &'static str = "rms_norm";
}

impl Fused for RmsNormCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.x.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self {
            x, weight, groups, ..
        } = self;
        let normalized = rms_scaled(&x.to_dtype(DType::F32)?, self.eps)?;
        let gain = gain(weight, self.centred)?;
        if groups == 1 {
            return Ok(normalized.broadcast_mul(&gain)?.to_dtype(self.output)?);
        }

        // Row `i` reads gain row `i mod groups`.
        let width = gain.elem_count() / groups;
        let scaled = normalized
            .reshape((x.elem_count() / width / groups, groups, width))?
            .broadcast_mul(&gain.reshape((groups, width))?)?;
        Ok(scaled.reshape(x.shape())?.to_dtype(self.output)?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self { x, weight, .. } = self;
        let width = columns(Self::OWNER, x)?;
        let rows = as_rows(Self::OWNER, x)?;
        let out = ops::empty(rows.shape(), self.output)?;

        if self.output == x.dtype() {
            ops::launch::<RmsNorm>(
                &RmsNormDescriptor::new(x.dtype(), width, self.centred, self.groups as u32),
                RmsNormArgs {
                    x: ops::operand(&rows)?,
                    weight: ops::operand(weight)?,
                    out: ops::operand(&out)?,
                    eps: self.eps,
                },
            )?;
        } else {
            ops::launch::<WeightedRmsNorm>(
                &WeightedRmsNormDescriptor::new(x.dtype(), width, self.output, false),
                WeightedRmsNormArgs {
                    x: ops::operand(&rows)?,
                    weight: ops::operand(weight)?,
                    out: ops::operand(&out)?,
                    eps: self.eps,
                    partials: None,
                },
            )?;
        }
        like_rows(out, x)
    }
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
    const OWNER: &str = AddRmsNormCall::OWNER;

    let width = columns(OWNER, x)?;
    vector(OWNER, "weight", weight, width)?;
    same_rows(OWNER, "residual stream", residual, x)?;

    compute(AddRmsNormCall {
        x,
        residual,
        weight,
        eps,
        centred,
    })
}

/// A checked call of [`add_rms_norm`].
struct AddRmsNormCall<'t> {
    x: &'t Tensor,
    residual: &'t Tensor,
    weight: &'t Tensor,
    eps: f32,
    centred: bool,
}

impl AddRmsNormCall<'_> {
    const OWNER: &'static str = "add_rms_norm";
}

impl Fused for AddRmsNormCall<'_> {
    type Output = (Tensor, Tensor);

    fn device(&self) -> &Device {
        self.x.device()
    }

    fn cpu(self) -> Result<(Tensor, Tensor)> {
        let Self { x, residual, .. } = self;
        let sum = (x.to_dtype(DType::F32)? + residual.to_dtype(DType::F32)?)?;
        let normalized =
            rms_scaled(&sum, self.eps)?.broadcast_mul(&gain(self.weight, self.centred)?)?;

        Ok((normalized.to_dtype(x.dtype())?, sum.to_dtype(x.dtype())?))
    }

    fn launch(self) -> Result<(Tensor, Tensor)> {
        let Self { x, residual, .. } = self;
        let width = columns(Self::OWNER, x)?;
        let rows = as_rows(Self::OWNER, x)?;
        let residual_rows = as_rows(Self::OWNER, residual)?;
        let out = ops::empty(rows.shape(), x.dtype())?;
        let sum = ops::empty(rows.shape(), x.dtype())?;

        ops::launch::<AddRmsNorm>(
            &AddRmsNormDescriptor::new(x.dtype(), width, self.centred),
            AddRmsNormArgs {
                x: ops::operand(&rows)?,
                residual: ops::operand(&residual_rows)?,
                weight: ops::operand(self.weight)?,
                out: ops::operand(&out)?,
                residual_out: ops::operand(&sum)?,
                eps: self.eps,
            },
        )?;
        Ok((like_rows(out, x)?, like_rows(sum, x)?))
    }
}

/// RMS normalization gated by an activation of a second input:
/// `x · rsqrt(mean(x²) + eps) · weight · act(gate)`, the output norm of a gated delta network.
/// The activation is the SiLU, or the sigmoid when `sigmoid`.
///
/// The row is normalized before it is gated, in FP32, and rounds once into `x`'s element type.
///
/// # Errors
///
/// When `gate` is not `x`'s shape and element type, `weight` is not one vector of the row
/// width, or the operator refuses its arguments.
pub fn gated_rms_norm(
    x: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f32,
    sigmoid: bool,
) -> Result<Tensor> {
    const OWNER: &str = GatedRmsNormCall::OWNER;

    let width = columns(OWNER, x)?;
    vector(OWNER, "weight", weight, width)?;
    same_rows(OWNER, "gate", gate, x)?;

    compute(GatedRmsNormCall {
        x,
        gate,
        weight,
        eps,
        sigmoid,
    })
}

/// A checked call of [`gated_rms_norm`].
struct GatedRmsNormCall<'t> {
    x: &'t Tensor,
    gate: &'t Tensor,
    weight: &'t Tensor,
    eps: f32,
    sigmoid: bool,
}

impl GatedRmsNormCall<'_> {
    const OWNER: &'static str = "gated_rms_norm";
}

impl Fused for GatedRmsNormCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.x.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let normalized = rms_scaled(&self.x.to_dtype(DType::F32)?, self.eps)?
            .broadcast_mul(&gain(self.weight, false)?)?;
        let gate = self.gate.to_dtype(DType::F32)?;
        let gated = (normalized
            * if self.sigmoid {
                gate.sigmoid()?
            } else {
                gate.silu()?
            })?;

        Ok(gated.to_dtype(self.x.dtype())?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self { x, gate, .. } = self;
        let width = columns(Self::OWNER, x)?;
        let rows = as_rows(Self::OWNER, x)?;
        let gate_rows = as_rows(Self::OWNER, gate)?;
        let out = ops::empty(rows.shape(), x.dtype())?;

        ops::launch::<GatedRmsNorm>(
            &GatedRmsNormDescriptor::new(x.dtype(), width, self.sigmoid),
            GatedRmsNormArgs {
                x: ops::operand(&rows)?,
                gate: ops::operand(&gate_rows)?,
                weight: ops::operand(self.weight)?,
                out: ops::operand(&out)?,
                eps: self.eps,
            },
        )?;
        like_rows(out, x)
    }
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
    const OWNER: &str = LayerNormCall::OWNER;

    let width = columns(OWNER, x)?;
    vector(OWNER, "weight", weight, width)?;
    if let Some(bias) = bias {
        vector(OWNER, "bias", bias, width)?;
    }

    compute(LayerNormCall {
        x,
        weight,
        bias,
        eps,
    })
}

/// A checked call of [`layer_norm`].
struct LayerNormCall<'t> {
    x: &'t Tensor,
    weight: &'t Tensor,
    bias: Option<&'t Tensor>,
    eps: f32,
}

impl LayerNormCall<'_> {
    const OWNER: &'static str = "layer_norm";
}

impl Fused for LayerNormCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.x.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let rows = self.x.to_dtype(DType::F32)?;
        let centred = rows.broadcast_sub(&rows.mean_keepdim(D::Minus1)?)?;
        let variance = centred.sqr()?.mean_keepdim(D::Minus1)?;
        let scale = (variance + f64::from(self.eps))?.sqrt()?.recip()?;

        let mut normalized = centred
            .broadcast_mul(&scale)?
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        if let Some(bias) = self.bias {
            normalized = normalized.broadcast_add(&bias.to_dtype(DType::F32)?)?;
        }
        Ok(normalized.to_dtype(self.x.dtype())?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self { x, bias, .. } = self;
        let width = columns(Self::OWNER, x)?;
        let rows = as_rows(Self::OWNER, x)?;
        let out = ops::empty(rows.shape(), x.dtype())?;

        ops::launch::<LayerNorm>(
            // The bias is in the descriptor, so a normalization with one and one without are
            // two kernels: the epilogue loads a second vector or it does not.
            &LayerNormDescriptor::new(x.dtype(), width, bias.is_some()),
            LayerNormArgs {
                x: ops::operand(&rows)?,
                weight: ops::operand(self.weight)?,
                bias: bias.map(ops::operand).transpose()?,
                out: ops::operand(&out)?,
                eps: self.eps,
            },
        )?;
        like_rows(out, x)
    }
}
