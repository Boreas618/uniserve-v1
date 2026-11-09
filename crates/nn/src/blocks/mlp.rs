//! The gated MLP: `down(act(gate(x)) · up(x))`.
//!
//! The gate and up projections are column parallel and the down projection is row parallel over the
//! spec's split, so a tensor-parallel block reduces once, in its down projection. A checkpoint
//! stores the two input projections either as two modules or as one module whose rows hold the
//! value half and then the gate half ([`GateUp`]); the gated product reads them as two views or as
//! the packed rows, at the [`Rounding`] the consumer's baseline evaluates.

use ops::TensorView;
use ops::activation::Activation;

use crate::error::Result;
use crate::family::Split;
use crate::forward::Forward;
use crate::layers::{Bias, GateBias, GatedActivation, Halves, Linear, Parallel, Rounding};
use crate::weights::Scope;

/// How a checkpoint names the gated MLP's input projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateUp<'n> {
    /// Two modules, the gate and the up projection.
    Split { gate: &'n str, up: &'n str },
    /// One module whose rows hold the value (up) half and then the gate half.
    ValueFirst(&'n str),
}

/// The geometry, names and arithmetic of a [`GatedMlp`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedMlpSpec<'n> {
    /// The width the block reads and writes.
    pub hidden: u32,
    /// The gated width, before any split.
    pub intermediate: u32,
    /// The gate's activation.
    pub activation: Activation,
    /// Where the gated product rounds; a consumer keeps its baseline's.
    pub rounding: Rounding,
    /// The input projections' module names.
    pub gate_up: GateUp<'n>,
    /// The output projection's module name.
    pub down: &'n str,
}

/// The input projections, as the checkpoint stores them.
#[allow(
    clippy::large_enum_variant,
    reason = "one value per block, held for the model's lifetime; boxing would put an indirection \
              on every forward"
)]
enum Inputs {
    Split { gate: Linear, up: Linear },
    ValueFirst(Linear),
}

/// A gated MLP block.
pub struct GatedMlp {
    inputs: Inputs,
    activation: GatedActivation,
    down: Linear,
}

impl GatedMlp {
    /// The block the module at `scope` states, at `split`'s rank of its tensor-parallel group.
    ///
    /// A group of one holds every projection whole.
    ///
    /// # Errors
    ///
    /// A projection the loader refuses, a split that does not divide the gated width, or a
    /// rounding the activation has no program for.
    pub fn load(scope: &mut Scope<'_, '_>, spec: GatedMlpSpec<'_>, split: Split) -> Result<Self> {
        let (column, row) = if split.size == 1 {
            (Parallel::None, Parallel::None)
        } else {
            (Parallel::Column(split), Parallel::Row(split))
        };
        let mut input = |name: &str, outputs: u32| {
            Linear::load(
                &mut scope.scope(name),
                spec.hidden,
                outputs,
                Bias::Absent,
                column,
            )
        };
        let inputs = match spec.gate_up {
            GateUp::Split { gate, up } => Inputs::Split {
                gate: input(gate, spec.intermediate)?,
                up: input(up, spec.intermediate)?,
            },
            GateUp::ValueFirst(name) => Inputs::ValueFirst(input(name, 2 * spec.intermediate)?),
        };
        let down = Linear::load(
            &mut scope.scope(spec.down),
            spec.intermediate,
            spec.hidden,
            Bias::Absent,
            row,
        )?;
        Ok(Self {
            inputs,
            activation: GatedActivation::new(spec.activation, spec.rounding, GateBias::None)?,
            down,
        })
    }

    /// `out = down(act(gate(x)) · up(x))` over `x [rows, hidden]` into `out [rows, hidden]`.
    ///
    /// The projections' outputs are reserved in `x`'s element type and released before return;
    /// the gated product of two split halves is written over the gate's own buffer, which the
    /// activation reads element by element before it writes. `out` may be `x`, which is read only
    /// by the input projections.
    ///
    /// # Errors
    ///
    /// A view the projections refuse, or an arena that cannot hold the intermediates.
    pub fn forward(&self, f: &Forward, x: TensorView, out: TensorView) -> Result<()> {
        let arena = f.arena();
        let mark = arena.mark();
        let gated = match &self.inputs {
            Inputs::Split { gate, up } => {
                let gate = gate.apply(f, x, x.dtype)?;
                let up = up.apply(f, x, x.dtype)?;
                let halves = Halves::Split { gate, value: up };
                self.activation.forward(f, halves, gate)?;
                gate
            }
            Inputs::ValueFirst(projection) => {
                let packed = projection.apply(f, x, x.dtype)?;
                let gated = arena.tensor(
                    x.dtype,
                    ops::Shape::new(&[x.rows(), packed.shape.last() / 2]),
                )?;
                self.activation
                    .forward(f, Halves::ValueFirst(packed), gated)?;
                gated
            }
        };
        self.down.forward(f, gated, out)?;
        arena.release(mark);
        Ok(())
    }

    /// [`Self::forward`] into an output of `x`'s shape and element type reserved from the arena.
    ///
    /// # Errors
    ///
    /// As [`Self::forward`].
    pub fn apply(&self, f: &Forward, x: TensorView) -> Result<TensorView> {
        let out = f.arena().tensor(
            x.dtype,
            ops::Shape::new(&[x.rows(), u64::from(self.down.n())]),
        )?;
        self.forward(f, x, out)?;
        Ok(out)
    }
}
