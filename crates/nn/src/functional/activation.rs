//! Activations: the gated activation of a transformer's MLP, and the plain GELU, SiLU and Snake.
//!
//! A gated MLP projects one hidden state into two halves and multiplies the activation of one
//! half by the other. Three operators state that product, and they differ in the FP32 evaluation
//! order and the points where they round ([`Rounding`]):
//!
//! ```text
//! act_and_mul                out = round(act(gate) · value)                    Rounding::Fused
//! act_and_mul, stepwise      out = round(round(act(gate)) · value)             Rounding::Stepwise
//! swiglu, value_first_swiglu out = round((value · gate) / (1 + exp(−gate)))    Rounding::ValueProduct
//! ```
//!
//! The three orders round differently, so a model keeps the one its reference computes: a
//! reference that evaluates the activation and the product as separate rounded steps keeps the
//! stepwise form; the SwiGLU kernels multiply value and gate before dividing by the sigmoid's
//! denominator.
//!
//! The plain activations are tensor operations ([`silu`], [`gelu`], [`gelu_tanh`]); [`snake`]
//! evaluates its whole expression in FP32 and rounds once, which a composition would not.

use ops::activation::{
    ActAndMul, ActAndMulArgs, ActAndMulDescriptor, Swiglu, SwigluArgs, SwigluDescriptor,
    ValueFirstSwiglu, ValueFirstSwigluArgs, ValueFirstSwigluDescriptor,
};
use tensor::{D, DType, Tensor};

use super::{PDL, as_rows, columns, like_rows, on_cpu, same_rows, vector};
use crate::error::{Error, Result};

pub use ops::activation::Activation;

/// The elements one kernel of a plain activation handles: a power of two keeps every vector load
/// aligned.
const BLOCK: u32 = 1024;

/// The elements one kernel of a gated activation handles: wide enough that each lane moves a
/// whole vector of a half-precision row, which is what this memory-bound step runs at.
const PACKED_BLOCK: u32 = 4096;

/// The FP32 evaluation order and rounding points of a gated product; see the module table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rounding {
    /// `act(gate) · value` in FP32, rounded once.
    Fused,
    /// `act(gate)` rounded to the gate's element type, then the product rounded.
    Stepwise,
    /// `(value · gate) / (1 + exp(−gate))` in FP32, rounded once: the SwiGLU kernels' order, SiLU
    /// only.
    ValueProduct,
}

/// The sigmoid linear unit, `x / (1 + exp(−x))`, elementwise.
///
/// # Errors
///
/// When `x` is not a floating-point tensor.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    Ok(x.silu()?)
}

/// PyTorch's `gelu`, the exact one: `0.5x(1 + erf(x/√2))`, elementwise.
///
/// # Errors
///
/// When `x` is not a floating-point tensor.
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu_erf()?)
}

/// PyTorch's `gelu_pytorch_tanh`: `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`, elementwise.
///
/// A checkpoint whose configuration says `gelu_pytorch_tanh` takes this one and a checkpoint
/// that says plain `gelu` takes [`gelu`]: they are different functions.
///
/// # Errors
///
/// When `x` is not a floating-point tensor.
pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu()?)
}

/// `activation` of the FP32 tensor `x`, elementwise.
pub(super) fn activate(x: &Tensor, activation: Activation) -> Result<Tensor> {
    Ok(match activation {
        Activation::Silu => x.silu()?,
        Activation::Gelu => x.gelu_erf()?,
        Activation::GeluTanh => x.gelu()?,
        Activation::Sigmoid => x.sigmoid()?,
    })
}

/// The row stride of `rows`, a `[rows, width]` tensor, in elements, as a kernel argument states
/// it.
fn row_stride(rows: &ops::Operand) -> u64 {
    rows.stride(0)
}

/// The gated activation `act(gate) · value`, the activation evaluated in FP32.
///
/// `gate` and `value` are `[rows, width]` of one element type, each with unit stride along its
/// rows: two tensors, or the two halves of one packed projection as strided views of its rows.
/// With `stepwise` the activation is rounded into the element type before the product, the
/// eager expression's rounding; without it the product rounds once. The result is a contiguous
/// `[rows, width]` tensor of the same element type.
///
/// # Errors
///
/// When the halves differ in shape or element type, or the operator refuses its arguments.
pub fn act_and_mul(
    gate: &Tensor,
    value: &Tensor,
    activation: Activation,
    stepwise: bool,
) -> Result<Tensor> {
    const OWNER: &str = "act_and_mul";

    let width = columns(OWNER, gate)?;
    same_rows(OWNER, "value half", value, gate)?;

    if on_cpu(gate) {
        let mut activated = activate(&gate.to_dtype(DType::F32)?, activation)?;
        if stepwise {
            activated = activated.to_dtype(gate.dtype())?.to_dtype(DType::F32)?;
        }
        let product = (activated * value.to_dtype(DType::F32)?)?;

        return Ok(product.to_dtype(gate.dtype())?);
    }

    let (gate_rows, value_rows) = (as_rows(OWNER, gate)?, as_rows(OWNER, value)?);
    let out = ops::empty(gate_rows.shape(), gate.dtype())?;
    let (gate_operand, value_operand) = (ops::operand(&gate_rows)?, ops::operand(&value_rows)?);
    let (gate_stride, value_stride) = (row_stride(&gate_operand), row_stride(&value_operand));

    // The 64-bit form follows the furthest element either input row reaches: the halves of one
    // packed projection stride over twice the output's row.
    let elements = out.elem_count() as u64;
    let rows = elements.div_ceil(u64::from(width));
    let span = elements.max(rows.saturating_mul(gate_stride.max(value_stride)));

    ops::launch::<ActAndMul>(
        &ActAndMulDescriptor {
            dtype: gate.dtype(),
            width,
            block: PACKED_BLOCK,
            activation,
            gate_stride,
            value_stride,
            wide: span >= 1 << 31,
            pdl: PDL,
            stepwise,
        },
        ActAndMulArgs {
            gate: gate_operand,
            value: value_operand,
            out: ops::operand(&out)?,
        },
    )?;
    like_rows(out, gate)
}

/// The SwiGLU product in the value-product order over two halves:
/// `((value + value_bias) · (gate + gate_bias)) / (1 + exp(−(gate + gate_bias)))`, in FP32,
/// rounded once.
///
/// `value` and `gate` are `[rows, width]` of one element type with unit stride along their rows;
/// each bias, when given, is one `[width]` vector added to its half before the activation.
///
/// # Errors
///
/// When the halves differ in shape or element type, a bias is not one vector of the row width,
/// or the operator refuses its arguments.
pub fn swiglu(
    value: &Tensor,
    gate: &Tensor,
    value_bias: Option<&Tensor>,
    gate_bias: Option<&Tensor>,
) -> Result<Tensor> {
    const OWNER: &str = "swiglu";

    let width = columns(OWNER, value)?;
    same_rows(OWNER, "gate half", gate, value)?;
    for (name, bias) in [("value bias", value_bias), ("gate bias", gate_bias)] {
        if let Some(bias) = bias {
            vector(OWNER, name, bias, width)?;
        }
    }

    if on_cpu(value) {
        let biased = |half: &Tensor, bias: Option<&Tensor>| -> Result<Tensor> {
            let half = half.to_dtype(DType::F32)?;
            match bias {
                Some(bias) => Ok(half.broadcast_add(&bias.to_dtype(DType::F32)?)?),
                None => Ok(half),
            }
        };
        let (value32, gate32) = (biased(value, value_bias)?, biased(gate, gate_bias)?);
        let denominator = (gate32.neg()?.exp()? + 1.0)?;

        return Ok(((value32 * &gate32)? / denominator)?.to_dtype(value.dtype())?);
    }

    let (value_rows, gate_rows) = (as_rows(OWNER, value)?, as_rows(OWNER, gate)?);
    let out = ops::empty(value_rows.shape(), value.dtype())?;
    let (value_operand, gate_operand) = (ops::operand(&value_rows)?, ops::operand(&gate_rows)?);

    ops::launch::<Swiglu>(
        &SwigluDescriptor {
            dtype: value.dtype(),
            width,
            block: width.next_power_of_two(),
            value_stride: row_stride(&value_operand),
            gate_stride: row_stride(&gate_operand),
            value_bias: value_bias.is_some(),
            gate_bias: gate_bias.is_some(),
            absmax: false,
        },
        SwigluArgs {
            value: value_operand,
            gate: gate_operand,
            value_bias: value_bias.map(ops::operand).transpose()?,
            gate_bias: gate_bias.map(ops::operand).transpose()?,
            out: ops::operand(&out)?,
            partials: None,
        },
    )?;
    like_rows(out, value)
}

/// The SwiGLU product in the value-product order over one packed projection: `value_gate` is
/// `[rows, 2 · width]`, each row holding the value half and then the gate half, as a checkpoint
/// that fuses `[up, gate]` value-first stores it; `bias`, when given, is the packed
/// `[2 · width]` bias of the two halves. The result is `[rows, width]`, read in one pass.
///
/// # Errors
///
/// When the packed rows are not an even width, the bias is not one vector of the packed width,
/// or the operator refuses its arguments.
pub fn value_first_swiglu(value_gate: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    const OWNER: &str = "value_first_swiglu";

    let packed = columns(OWNER, value_gate)?;
    if !packed.is_multiple_of(2) {
        return Err(Error::shape(
            OWNER,
            format!("a packed row of {packed} elements does not hold two halves"),
        ));
    }
    let width = packed / 2;
    if let Some(bias) = bias {
        vector(OWNER, "packed bias", bias, packed)?;
    }

    let rows = as_rows(OWNER, value_gate)?;
    if on_cpu(value_gate) {
        let mut halves = rows.to_dtype(DType::F32)?;
        if let Some(bias) = bias {
            halves = halves.broadcast_add(&bias.to_dtype(DType::F32)?)?;
        }
        let value = halves.narrow(D::Minus1, 0, width as usize)?;
        let gate = halves.narrow(D::Minus1, width as usize, width as usize)?;
        let denominator = (gate.neg()?.exp()? + 1.0)?;

        let product = ((value * &gate)? / denominator)?.to_dtype(value_gate.dtype())?;
        return like_rows(product, value_gate);
    }

    let out = ops::empty((rows.dims()[0], width as usize), value_gate.dtype())?;
    let projection = ops::operand(&rows)?;

    ops::launch::<ValueFirstSwiglu>(
        &ValueFirstSwigluDescriptor {
            dtype: value_gate.dtype(),
            width,
            block: PACKED_BLOCK,
            input_stride: row_stride(&projection),
            bias: bias.is_some(),
            absmax: false,
        },
        ValueFirstSwigluArgs {
            value_gate: projection,
            bias: bias.map(ops::operand).transpose()?,
            out: ops::operand(&out)?,
            partials: None,
        },
    )?;
    like_rows(out, value_gate)
}

/// The periodic activation `x + sin²(α·x) / (α + 1e-9)`, one frequency per channel, evaluated in
/// FP32 and rounded once.
///
/// `x` is `[rows, channels, width]` and `alpha` the checkpoint's raw per-channel parameter,
/// `[channels]` in `x`'s element type (not a log-space pair). The result has `x`'s shape and
/// element type.
///
/// # Errors
///
/// When `x` is not three axes, `alpha` is not one frequency per channel, or the operator
/// refuses its arguments.
pub fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    const OWNER: &str = "snake";

    let &[_, channels, _] = x.dims() else {
        return Err(Error::shape(
            OWNER,
            format!("{:?} is not [rows, channels, width]", x.dims()),
        ));
    };
    if channels == 0 || alpha.elem_count() != channels {
        return Err(Error::shape(
            OWNER,
            format!("{} frequencies for {channels} channels", alpha.elem_count()),
        ));
    }

    if on_cpu(x) {
        let x32 = x.to_dtype(DType::F32)?;
        let alpha32 = alpha.to_dtype(DType::F32)?.reshape((1, channels, 1))?;
        let periodic = x32
            .broadcast_mul(&alpha32)?
            .sin()?
            .sqr()?
            .broadcast_div(&(alpha32 + 1e-9)?)?;

        return Ok((x32 + periodic)?.to_dtype(x.dtype())?);
    }

    let out = ops::empty(x.shape(), x.dtype())?;
    ops::launch::<ops::snake::Snake>(
        &ops::snake::Descriptor {
            dtype: x.dtype(),
            channels: u32::try_from(channels).unwrap_or(u32::MAX),
            block: BLOCK,
            pdl: PDL,
        },
        ops::snake::Args {
            x: ops::operand(x)?,
            alpha: ops::operand(alpha)?,
            out: ops::operand(&out)?,
        },
    )?;
    Ok(out)
}
