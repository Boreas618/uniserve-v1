//! Elementwise updates of a stream: the residual add, the two-coefficient update a solver steps
//! with, and the element-type conversion.
//!
//! ```text
//! Residual   out = residual + (update + bias) · scale     scale: none, per channel or per row
//! Axpby      y   = β · y + α · x                          two host coefficients
//! Cast       out = x                                      converted to out's element type
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
use ops::{DType, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::weights::Weight;

/// The elements one program of `scaled_add` and `scaled_residual` walks.
const BLOCK: u32 = residual::RESIDUAL_BLOCK;

/// Whether two views name the same elements: one address, element type, shape and strides.
fn same(a: &TensorView, b: &TensorView) -> bool {
    a.ptr == b.ptr && a.dtype == b.dtype && a.shape == b.shape && a.strides == b.strides
}

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

/// The residual add: `out = residual + (update + bias) · scale`.
///
/// The sum is formed in FP32 and rounded once into `out`'s element type. Which operand `out` may be
/// follows the scale, because each operator writes one of its inputs in place:
///
/// | scale | `out` |
/// | --- | --- |
/// | [`Scale::None`] | `residual` or `update` (the add is symmetric) |
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

    /// `out = residual + (update + bias) · scale` over `[rows, columns]`.
    ///
    /// `rows` is the `[rows]` index of each activation row's scale row, present exactly under
    /// [`Scale::Row`]. `out` aliases `residual` or `update` as the type documentation states.
    ///
    /// # Errors
    ///
    /// Views whose shapes disagree, a missing or superfluous row index, an `out` the scale's
    /// operator cannot write, or the operator's refusal.
    pub fn forward(
        &self,
        f: &Forward,
        residual: TensorView,
        update: TensorView,
        rows: Option<TensorView>,
        out: TensorView,
    ) -> Result<()> {
        if rows.is_some() != matches!(self.scale, Scale::Row(_)) {
            return Err(Error::shape(
                "Residual",
                "row indices are passed exactly under a row scale".to_owned(),
            ));
        }
        match (self.scale, rows) {
            (Scale::None, _) => {
                // The add is symmetric, so the written operand is the target and the other the
                // source; both coefficients are one, so the FP32 sum is the plain one.
                let (target, source) = if same(&out, &residual) {
                    (residual, update)
                } else if same(&out, &update) {
                    (update, residual)
                } else {
                    return Err(Error::shape(
                        "Residual",
                        "an unscaled add writes its residual or its update in place".to_owned(),
                    ));
                };
                Axpby.forward(f, target, source, 1.0, 1.0)
            }
            (Scale::Channel(scale), _) => {
                if !same(&out, &residual) {
                    return Err(Error::shape(
                        "Residual",
                        "a channel-scaled add writes its residual in place".to_owned(),
                    ));
                }
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
                f.operate::<ScaledResidual>(&descriptor, args)
            }
            (Scale::Row(gate), Some(rows)) => {
                if !same(&out, &update) {
                    return Err(Error::shape(
                        "Residual",
                        "a row-scaled add writes its update in place".to_owned(),
                    ));
                }
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
                f.operate::<GatedResidualOp>(&descriptor, args)
            }
            (Scale::Row(_), None) => unreachable!("the row index presence is checked above"),
        }
    }
}

impl Default for Residual {
    /// The plain add `out = residual + update`: no scale, no bias.
    fn default() -> Self {
        Self {
            scale: Scale::None,
            bias: None,
        }
    }
}

/// The two-coefficient update `y ← β · y + α · x`, in place on `y`.
///
/// A diffusion solver's arithmetic: an Euler step is `α = σ_next − σ`, `β = 1`; a rescale is
/// `α = 0`. Both coefficients are host values, so a captured graph stays valid, and the two
/// operands' element types may differ on purpose — a BF16 trajectory stepped by an FP32 velocity.
/// The products and the sum are FP32 and round once into `y`'s element type.
pub struct Axpby;

impl Axpby {
    /// `y ← β · y + α · x` over two `[rows, columns]` views of one shape.
    ///
    /// # Errors
    ///
    /// Views of another rank or of different shapes, or the operator's refusal.
    pub fn forward(
        &self,
        f: &Forward,
        y: TensorView,
        x: TensorView,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        if y.shape.rank() != 2 || y.shape != x.shape {
            return Err(Error::shape(
                "Axpby",
                format!("{:?} with {:?}", y.shape.dims(), x.shape.dims()),
            ));
        }
        let descriptor = residual::ScaledAddDescriptor {
            target: y.dtype,
            source: x.dtype,
            block: BLOCK,
        };
        let args = residual::ScaledAddArgs {
            target: y,
            source: x,
            alpha,
            beta,
        };
        f.operate::<ScaledAdd>(&descriptor, args)
    }
}

/// The element-type conversion `out = x`, rounded once into `out`'s element type.
///
/// It is the solver update with `α = 1`, `β = 0` — the target's old value is multiplied by zero and
/// never contributes — so one conversion is one `scaled_add` launch between any two floating
/// element types.
pub struct Cast;

impl Cast {
    /// Convert `x` into `out`, two `[rows, columns]` views of one shape.
    ///
    /// # Errors
    ///
    /// Views of another rank or of different shapes, or the operator's refusal.
    pub fn forward(&self, f: &Forward, x: TensorView, out: TensorView) -> Result<()> {
        Axpby.forward(f, out, x, 1.0, 0.0)
    }

    /// Convert `x` into a fresh `[rows, columns]` buffer of `dtype` reserved from the arena.
    ///
    /// # Errors
    ///
    /// An arena too small for the buffer, or [`Self::forward`]'s refusals.
    pub fn apply(&self, f: &Forward, x: TensorView, dtype: DType) -> Result<TensorView> {
        let out = f.arena().tensor(dtype, x.shape)?;
        self.forward(f, x, out)?;
        Ok(out)
    }
}
