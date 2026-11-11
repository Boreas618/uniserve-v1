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
    /// Whether each projection adds its module's bias.
    pub bias: Bias,
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
                spec.bias,
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
            spec.bias,
            row,
        )?;
        Ok(Self {
            inputs,
            activation: GatedActivation::new(spec.activation, spec.rounding, GateBias::None)?,
            down,
        })
    }

    /// `down(act(gate(x)) · up(x))` over `x [rows, hidden]`: a fresh `[rows, hidden]` tensor of
    /// `x`'s element type, reserved first, then written by [`Self::forward_into`].
    ///
    /// # Errors
    ///
    /// As [`Self::forward_into`], or an arena that cannot hold the output.
    pub fn forward(&self, f: &Forward, x: TensorView) -> Result<TensorView> {
        // The down projection reserves its own output: a packed product's region holds the row
        // class it writes whole.
        let out = self.down.reserve(f, x)?;
        self.forward_into(f, x, out)?;
        Ok(out)
    }

    /// The block over `x` written into `update`, `[rows, hidden]` of `x`'s element type: the
    /// caller's buffer the output must land in, such as a stack's update carried across its
    /// layers' scopes. The projections' intermediates are handed back before return.
    ///
    /// When the down projection is packed ([`Self::is_packed`]) its product writes whole row
    /// classes ([`ops::fp4::packed_rows`]), so `update` must lie at the start of a region that
    /// holds the class of its rows.
    ///
    /// # Errors
    ///
    /// A view the projections refuse, or an arena that cannot hold the intermediates.
    pub fn forward_into(&self, f: &Forward, x: TensorView, update: TensorView) -> Result<()> {
        f.scoped(|f| {
            let halves = match &self.inputs {
                Inputs::Split { gate, up } => Halves::Split {
                    gate: gate.forward(f, x)?,
                    value: up.forward(f, x)?,
                },
                Inputs::ValueFirst(projection) => Halves::ValueFirst(projection.forward(f, x)?),
            };
            let gated = self.activation.forward(f, halves)?;
            self.down.launch(f, gated, update)
        })
    }

    /// Whether the down projection is packed, so [`Self::forward_into`]'s `update` must lie in a
    /// region of its rows' packed class.
    pub const fn is_packed(&self) -> bool {
        self.down.is_packed()
    }
}

/// The module names of a [`SharedExpert`]: its gated MLP's and its gate's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharedExpertSpec<'n> {
    /// The shared expert's gated MLP, at the module `mlp`.
    pub mlp: GatedMlpSpec<'n>,
    /// The module holding the MLP.
    pub module: &'n str,
    /// The module of the `[1, hidden]` gate projection beside it.
    pub gate: &'n str,
}

/// A shared expert beside a routed block: a gated MLP every token passes through, its output
/// scaled by the sigmoid of a per-token gate and added into the routed output (Qwen2-MoE,
/// Qwen3-Next, Qwen3.5):
///
/// ```text
/// out ← out + sigmoid(gate · x) · mlp(x)
/// ```
pub struct SharedExpert {
    mlp: GatedMlp,
    gate: Linear,
    hidden: u32,
}

impl SharedExpert {
    /// The shared expert the modules under `scope` state, at `split`'s rank. The gate is
    /// replicated: one logit per token.
    ///
    /// # Errors
    ///
    /// A module the loader refuses or a split the MLP refuses.
    pub fn load(
        scope: &mut Scope<'_, '_>,
        spec: SharedExpertSpec<'_>,
        split: Split,
    ) -> Result<Self> {
        let mlp = GatedMlp::load(&mut scope.scope(spec.module), spec.mlp, split)?;
        let gate = Linear::load(
            &mut scope.scope(spec.gate),
            spec.mlp.hidden,
            1,
            Bias::Absent,
            Parallel::None,
        )?;
        Ok(Self {
            mlp,
            gate,
            hidden: spec.mlp.hidden,
        })
    }

    /// Add the shared expert's gated output of `x [rows, hidden]` into `inout [rows, hidden]`, in
    /// place: the routed block's output it sits beside.
    ///
    /// # Errors
    ///
    /// A view a layer refuses, or an arena that cannot hold the intermediates.
    pub fn forward(&self, f: &Forward, x: TensorView, inout: TensorView) -> Result<()> {
        use ops::residual::{SigmoidGatedAdd, SigmoidGatedAddArgs, SigmoidGatedAddDescriptor};

        f.scoped(|f| {
            let update = self.mlp.forward(f, x)?;
            let gate = self.gate.forward(f, x)?;
            f.operate::<SigmoidGatedAdd>(
                &SigmoidGatedAddDescriptor {
                    dtype: inout.dtype,
                    columns: self.hidden,
                    block: crate::layers::launch::row_tile(self.hidden).min(1024),
                },
                SigmoidGatedAddArgs {
                    target: inout,
                    update,
                    gate,
                },
            )
        })
    }
}
