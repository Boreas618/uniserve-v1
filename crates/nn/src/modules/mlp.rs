//! The gated MLP of a transformer block: `down(act(gate(x)) · up(x))`, its halves clamped first
//! when the model states a limit (`down(act(min(gate(x), L)) · clamp(up(x), ±L))`).
//!
//! The gate and up projections are column parallel and the down projection is row parallel over
//! the config's split, so a tensor-parallel block reduces once, in its down projection. A
//! checkpoint stores the two input projections either as two modules or as one module whose rows
//! hold the value half and then the gate half ([`GateUp`]); the gated product reads them as two
//! tensors or as the packed rows, at the [`Rounding`] the model's reference evaluates.

use tensor::{D, Module, Tensor};

use crate::collective::Split;
use crate::error::{Error, Result};
use crate::functional::{self, Activation, Rounding};
use crate::modules::linear::{Linear, LinearConfig, Parallel};
use crate::var_builder::VarBuilder;

/// How a checkpoint names the gated MLP's input projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateUp {
    /// Two modules, the gate and the up projection.
    Split {
        /// The module whose output the activation applies to.
        gate: &'static str,
        /// The module whose output the activation multiplies.
        up: &'static str,
    },
    /// One module whose rows hold the value (up) half and then the gate half.
    ValueFirst(&'static str),
}

/// The widths, checkpoint names and arithmetic of an [`Mlp`], and this rank's share of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlpConfig {
    /// The width the block reads and writes.
    pub hidden: u32,
    /// The gated width, before any split.
    pub intermediate: u32,
    /// The gate's activation.
    pub activation: Activation,
    /// Where the gated product rounds; a model keeps its reference's.
    pub rounding: Rounding,
    /// The clamp of the gated product's halves, when the model states one: the gate from above
    /// and the up half on both sides, before the activation.
    pub limit: Option<f32>,
    /// The input projections' module names.
    pub gate_up: GateUp,
    /// The output projection's module name.
    pub down: &'static str,
    /// Whether each projection adds its module's bias.
    pub bias: bool,
    /// The tensor-parallel split of the gated width.
    pub split: Split,
}

/// The input projections, as the checkpoint stores them.
enum Inputs {
    Split { gate: Linear, up: Linear },
    ValueFirst(Linear),
}

/// A gated MLP block.
pub struct Mlp {
    inputs: Inputs,
    activation: Activation,
    rounding: Rounding,
    limit: Option<f32>,
    down: Linear,
}

impl Mlp {
    /// The block the module at `vb` states, at `config.split`'s rank of its tensor-parallel
    /// group. A group of one holds every projection whole.
    ///
    /// # Errors
    ///
    /// A projection the loader refuses, a split that does not divide the gated width, or the
    /// value-product rounding with an activation other than SiLU or with a clamp, which have no
    /// kernel.
    pub fn load(config: &MlpConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        if config.rounding == Rounding::ValueProduct && config.limit.is_some() {
            return Err(Error::shape(
                "Mlp",
                "the value-product order has no clamp; a clamped MLP rounds fused or stepwise"
                    .to_owned(),
            ));
        }
        if config.rounding == Rounding::ValueProduct && config.activation != Activation::Silu {
            return Err(Error::shape(
                "Mlp",
                format!(
                    "the value-product order is SiLU's, not {}'s",
                    config.activation
                ),
            ));
        }

        let (column, row) = if config.split.size() == 1 {
            (Parallel::None, Parallel::None)
        } else {
            (Parallel::Column(config.split), Parallel::Row(config.split))
        };
        let input = |name: &str, outputs: u32| {
            Linear::load(
                &LinearConfig {
                    inputs: config.hidden,
                    outputs,
                    bias: config.bias,
                    parallel: column,
                },
                vb.pp(name),
            )
        };

        let inputs = match config.gate_up {
            GateUp::Split { gate, up } => Inputs::Split {
                gate: input(gate, config.intermediate)?,
                up: input(up, config.intermediate)?,
            },
            GateUp::ValueFirst(name) if config.split.size() == 1 => {
                Inputs::ValueFirst(input(name, 2 * config.intermediate)?)
            }
            // A rank's share of the packed rows is not one block of them: its value rows lie in
            // the value half and its gate rows in the gate half. The packed rows are twice the
            // split's blocks of a share each, the value half's first, so a rank reads the block
            // of its rank and the block a split's width after it, as two projections.
            GateUp::ValueFirst(name) => {
                let split = config.split;
                let half = |rank: u32| {
                    Linear::load(
                        &LinearConfig {
                            inputs: config.hidden,
                            outputs: 2 * config.intermediate,
                            bias: config.bias,
                            parallel: Parallel::Column(Split::new(2 * split.size(), rank)?),
                        },
                        vb.pp(name),
                    )
                };
                Inputs::Split {
                    up: half(split.rank())?,
                    gate: half(split.size() + split.rank())?,
                }
            }
        };
        let down = Linear::load(
            &LinearConfig {
                inputs: config.intermediate,
                outputs: config.hidden,
                bias: config.bias,
                parallel: row,
            },
            vb.pp(config.down),
        )?;

        Ok(Self {
            inputs,
            activation: config.activation,
            rounding: config.rounding,
            limit: config.limit,
            down,
        })
    }

    /// Whether the down projection is a calibrated NVFP4 matrix.
    pub const fn is_packed(&self) -> bool {
        self.down.is_packed()
    }

    /// `down(act(gate(x)) · up(x))` over `x [..., hidden]`: `[..., hidden]` in `x`'s element
    /// type.
    ///
    /// # Errors
    ///
    /// Rows the projections refuse, or an operator's refusal.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let stepwise = self.rounding == Rounding::Stepwise;

        let gated = match (&self.inputs, self.rounding) {
            (Inputs::ValueFirst(projection), Rounding::ValueProduct) => {
                functional::value_first_swiglu(&projection.forward(x)?, None)?
            }

            (Inputs::Split { gate, up }, Rounding::ValueProduct) => {
                functional::swiglu(&up.forward(x)?, &gate.forward(x)?, None, None)?
            }

            (Inputs::Split { gate, up }, _) => functional::act_and_mul(
                &gate.forward(x)?,
                &up.forward(x)?,
                self.activation,
                stepwise,
                self.limit,
            )?,

            // The halves of one packed projection, as two strided views of its rows: the value
            // half first, the gate half one half-row along.
            (Inputs::ValueFirst(projection), _) => {
                let packed = projection.forward(x)?;
                let width = packed.dim(D::Minus1)? / 2;
                let value = packed.narrow(D::Minus1, 0, width)?;
                let gate = packed.narrow(D::Minus1, width, width)?;

                functional::act_and_mul(&gate, &value, self.activation, stepwise, self.limit)?
            }
        };
        self.down.forward(&gated)
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> tensor::Result<Tensor> {
        Ok(Self::forward(self, x)?)
    }
}
