//! Elementwise updates of a stream: the residual add, the two-coefficient update a solver steps
//! with, and the element-type conversion.
//!
//! ```text
//! Residual   s = residual + (update + bias) · scale       in place; scale: none, per channel or per row
//! Axpby      y = β · y + α · x                            in place; two host coefficients
//! Cast       out = x                                      a fresh tensor of the requested element type
//! ```
//!
//! Each form computes in FP32 and rounds once into the written operand's element type. The three
//! scales of [`Residual`] are three operators — `scaled_add` (no scale), `scaled_residual` (one
//! learned scale per channel), `gated_residual` (a table row per activation row) — because each
//! reads a different operand set; the layer picks the one its scale names, so a consumer states the
//! mathematics and never the kernel. An all-ones channel scale is not a fourth form: it rounds
//! exactly as the unscaled add does, so a block whose scale is one has no scale.

use ops::modulation::{self, GatedResidual as GatedResidualOp};
use ops::residual::{self, ScaledAdd, ScaledResidual};
use ops::Operand;
use uniserve_tensor::DType;

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::weights::Weight;

/// The elements one program of `scaled_add` and `scaled_residual` walks.
const BLOCK: u32 = residual::RESIDUAL_BLOCK;

/// The width of a `[columns]` parameter vector, checked once.
fn channels(owner: &'static str, weight: &Weight) -> Result<u32> {
    if weight.shape().rank() != 1 {
        return Err(Error::shape(
            owner,
            format!(
                "a per-channel parameter is one vector, not {:?}",
                weight.shape().dims()
            ),
        ));
    }
    u32::try_from(weight.shape().numel())
        .map_err(|_| Error::shape(owner, "more channels than a descriptor holds".to_owned()))
}

/// The coefficient a [`Residual`] multiplies its update by.
#[derive(Clone, Copy, Debug)]
pub enum Scale {
    /// No coefficient: `residual + update`.
    None,
    /// One learned coefficient per channel, a `[columns]` vector (layer scale).
    Channel(Weight),
    /// One coefficient row per activation row, selected from a `[parameters, columns]` table by the
    /// call's row indices (a diffusion block's gate).
    Row(Weight),
}

/// The residual add `residual + (update + bias) · scale`, in place.
///
/// The sum is formed in FP32 and rounded once into the written operand's element type. Which
/// operand receives it follows the scale, because each operator writes one of its inputs:
///
/// | scale | written |
/// | --- | --- |
/// | [`Scale::None`] | `residual` (the add is symmetric, so a caller names the operand to keep as `residual`) |
/// | [`Scale::Channel`] | `residual` |
/// | [`Scale::Row`] | `update` |
///
/// A bias is added to the update before the channel scale and exists only with one.
#[derive(Clone, Copy, Debug)]
pub struct Residual {
    scale: Scale,
    bias: Option<Weight>,
}

impl Residual {
    /// The residual add with `scale` and, for a channel scale, the update's `bias`.
    ///
    /// # Errors
    ///
    /// A bias without a channel scale, a bias of another width than the scale, or a scale of the
    /// wrong rank.
    pub fn new(scale: Scale, bias: Option<Weight>) -> Result<Self> {
        match scale {
            Scale::None if bias.is_some() => {
                return Err(Error::shape(
                    "Residual",
                    "an update bias is added under a channel scale only".to_owned(),
                ));
            }
            Scale::None => {}
            Scale::Channel(weight) => {
                let width = channels("Residual", &weight)?;
                if let Some(bias) = &bias
                    && channels("Residual", bias)? != width
                {
                    return Err(Error::shape(
                        "Residual",
                        format!("a scale of {width} channels and a bias of another count"),
                    ));
                }
            }
            Scale::Row(table) => {
                if bias.is_some() {
                    return Err(Error::shape(
                        "Residual",
                        "an update bias is added under a channel scale only".to_owned(),
                    ));
                }
                if table.shape().rank() != 2 || table.shape().last() == 0 {
                    return Err(Error::shape(
                        "Residual",
                        format!(
                            "a row scale is a [parameters, columns] table, not {:?}",
                            table.shape().dims()
                        ),
                    ));
                }
            }
        }
        Ok(Self { scale, bias })
    }

    /// `residual + (update + bias) · scale` over `[rows, columns]`, written in place into the
    /// operand the type documentation names, which the call answers.
    ///
    /// `rows` is the `[rows]` index of each activation row's scale row, present exactly under
    /// [`Scale::Row`].
    ///
    /// # Errors
    ///
    /// Views whose shapes disagree, a missing or superfluous row index, or the operator's refusal.
    pub fn forward(
        &self,
        f: &Forward,
        residual: Operand,
        update: Operand,
        rows: Option<Operand>,
    ) -> Result<Operand> {
        if rows.is_some() != matches!(self.scale, Scale::Row(_)) {
            return Err(Error::shape(
                "Residual",
                "row indices are passed exactly under a row scale".to_owned(),
            ));
        }
        match (self.scale, rows) {
            (Scale::None, _) => {
                // Both coefficients are one, so the FP32 sum is the plain one.
                Axpby.forward(f, residual, update, 1.0, 1.0)?;
                Ok(residual)
            }
            (Scale::Channel(scale), _) => {
                let descriptor = residual::ScaledResidualDescriptor {
                    dtype: residual.dtype,
                    columns: channels("Residual", &scale)?,
                    block: BLOCK,
                    update_bias: self.bias.is_some(),
                };
                let args = residual::ScaledResidualArgs {
                    hidden: residual,
                    update,
                    update_bias: self.bias.as_ref().map(Weight::view),
                    scale: scale.view(),
                };
                f.operate::<ScaledResidual>(&descriptor, args)?;
                Ok(residual)
            }
            (Scale::Row(gate), Some(rows)) => {
                let width = gate.shape().last();
                if residual.shape.rank() != 2
                    || residual.shape.last() != width
                    || update.shape != residual.shape
                {
                    return Err(Error::shape(
                        "Residual",
                        format!(
                            "{:?} and {:?} are not [rows, {width}]",
                            residual.shape.dims(),
                            update.shape.dims()
                        ),
                    ));
                }
                // The row count is a launch value, not a descriptor field: a clip's token count
                // changes with every request, and a descriptor carrying it would be a kernel per
                // clip length.
                let descriptor = modulation::GatedResidualDescriptor {
                    dtype: residual.dtype,
                    indices: rows.dtype,
                    width: u32::try_from(width).map_err(|_| {
                        Error::shape("Residual", "a row wider than a descriptor holds".to_owned())
                    })?,
                    gate_stride: gate.view().strides.get(0),
                };
                let args = modulation::GatedResidualArgs {
                    hidden: residual,
                    update,
                    gate: gate.view(),
                    row_indices: rows,
                };
                f.operate::<GatedResidualOp>(&descriptor, args)?;
                Ok(update)
            }
            (Scale::Row(_), None) => unreachable!("the row index presence is checked above"),
        }
    }
}

impl Default for Residual {
    /// The plain add `residual + update`: no scale, no bias.
    fn default() -> Self {
        Self {
            scale: Scale::None,
            bias: None,
        }
    }
}

/// The two-coefficient update `inout ← β · inout + α · x`, in place on `inout`.
///
/// A diffusion solver's arithmetic: an Euler step is `α = σ_next − σ`, `β = 1`; a rescale is
/// `α = 0`. Both coefficients are host values, so a captured graph stays valid, and the two
/// operands' element types may differ on purpose — a BF16 trajectory stepped by an FP32 velocity.
/// The products and the sum are FP32 and round once into `inout`'s element type.
pub struct Axpby;

impl Axpby {
    /// `inout ← β · inout + α · x` over two `[rows, columns]` views of one shape.
    ///
    /// # Errors
    ///
    /// Views of another rank or of different shapes, or the operator's refusal.
    pub fn forward(
        &self,
        f: &Forward,
        inout: Operand,
        x: Operand,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        if inout.shape.rank() != 2 || inout.shape != x.shape {
            return Err(Error::shape(
                "Axpby",
                format!("{:?} with {:?}", inout.shape.dims(), x.shape.dims()),
            ));
        }
        let descriptor = residual::ScaledAddDescriptor {
            target: inout.dtype,
            source: x.dtype,
            block: BLOCK,
        };
        let args = residual::ScaledAddArgs {
            target: inout,
            source: x,
            alpha,
            beta,
        };
        f.operate::<ScaledAdd>(&descriptor, args)
    }
}

/// The element-type conversion of `x` into a fresh tensor, rounded once.
///
/// It is the solver update with `α = 1`, `β = 0` — the target's old value is multiplied by zero and
/// never contributes — so one conversion is one `scaled_add` launch between any two floating
/// element types.
pub struct Cast;

impl Cast {
    /// `x`, a `[rows, columns]` view, converted into a fresh tensor of `dtype`.
    ///
    /// # Errors
    ///
    /// A view of another rank, an arena that cannot hold the output, or the operator's refusal.
    pub fn forward(&self, f: &Forward, x: Operand, dtype: DType) -> Result<Operand> {
        let out = f.empty(dtype, x.shape)?;
        Axpby.forward(f, out, x, 1.0, 0.0)?;
        Ok(out)
    }
}
