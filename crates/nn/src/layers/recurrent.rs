//! The recurrent layers of a gated delta network: the short causal convolution and the gated delta
//! rule, each advancing a state the batch's request slots keep (`ops::recurrent`).
//!
//! Both read a batch the way paged attention does ([`TokenBatch`]'s packed offsets and lengths) and
//! the state through a [`StateView`]'s slots and checkpoints: a row that starts at position zero
//! begins from a zero state, every other row continues from its slot's, and a row that crosses a
//! checkpoint keeps its state there for the prefix cache. The banks are the infrastructure's; a
//! layer only launches its operator over them.

use ops::recurrent::{
    CausalConv1d as CausalConv1dOp, CausalConv1dArgs, CausalConv1dDescriptor, Form,
    GatedDeltaRule as GatedDeltaRuleOp, GatedDeltaRuleArgs, GatedDeltaRuleDescriptor, RowColumns,
};
use ops::{Shape, TensorView};
use proto::program::PhaseKind;

use crate::capability::{StateView, TokenBatch};
use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::weights::{Scope, Weight};

/// Channels one `causal_conv1d` program covers.
const CONV_BLOCK: u32 = 256;

/// Tokens of one chunk of the chunked delta rule: a prefix-cache page holds a whole number of them,
/// so a checkpoint at a page boundary lands on a chunk end.
pub const DELTA_RULE_CHUNK: u32 = 64;

/// Value columns one `gated_delta_rule` program owns of a head's state; the chunked form's tensor
/// memory holds `2 · chunk + 4 · block` columns, which bounds it at 64 on sm_100.
const DELTA_RULE_BLOCK: u32 = 64;

/// The row columns the operators read, from the batch and the state view.
fn rows(batch: &TokenBatch, state: &StateView<'_>) -> RowColumns {
    RowColumns {
        query_offsets: batch.query_offsets,
        seq_lens: batch.seq_lens,
        slots: state.slots,
        checkpoints: state.checkpoints,
    }
}

/// The geometry of a [`CausalConv1d`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CausalConv1dSpec {
    /// Channels, each convolved by its own taps.
    pub channels: u32,
    /// Taps per channel.
    pub width: u32,
}

/// A depthwise causal convolution with its SiLU, continuing each row from its slot's last inputs:
/// a linear-attention layer's short convolution (`conv1d` of a gated delta network, no bias).
pub struct CausalConv1d {
    weight: Weight,
    spec: CausalConv1dSpec,
}

impl CausalConv1d {
    /// The module at `scope`: its `weight` of `[channels, 1, width]`, PyTorch's depthwise `Conv1d`.
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks the weight or holds another extent.
    pub fn load(scope: &mut Scope<'_, '_>, spec: CausalConv1dSpec) -> Result<Self> {
        let weight = scope.tensor(
            "weight",
            &[u64::from(spec.channels), 1, u64::from(spec.width)],
        )?;
        Ok(Self { weight, spec })
    }

    /// The extents one slot of its state bank holds: the last `width − 1` inputs of every channel,
    /// oldest first.
    pub fn state_dims(&self) -> Vec<u64> {
        vec![
            u64::from(self.spec.width - 1),
            u64::from(self.spec.channels),
        ]
    }

    /// Convolve `x [tokens, channels]` (rows any stride apart) into `out [tokens, channels]`,
    /// continuing from and advancing `state`, keeping `checkpoint` where the view asks.
    ///
    /// # Errors
    ///
    /// What the operator refuses.
    #[allow(
        clippy::too_many_arguments,
        reason = "the operator's inputs, its batch and its two banks"
    )]
    pub fn forward(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        view: &StateView<'_>,
        state: TensorView,
        checkpoint: TensorView,
        x: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let channels = u64::from(self.spec.channels);
        let width = u64::from(self.spec.width);
        let weight = self.weight.view().reshape(Shape::new(&[channels, width]));
        f.operate::<CausalConv1dOp>(
            &CausalConv1dDescriptor {
                dtype: x.dtype,
                channels: self.spec.channels,
                width: self.spec.width,
                block: CONV_BLOCK.min(self.spec.channels.next_power_of_two()),
            },
            CausalConv1dArgs {
                x,
                weight,
                out,
                rows: rows(batch, view),
                state,
                checkpoint,
            },
        )
    }
}

/// The geometry of a [`GatedDeltaRule`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatedDeltaRuleSpec {
    /// Query and key heads this rank holds.
    pub key_heads: u32,
    /// Value heads this rank holds, each with its own state.
    pub value_heads: u32,
    /// Elements of a query or key head.
    pub key_dim: u32,
    /// Elements of a value head.
    pub value_dim: u32,
}

/// The gated delta rule of a gated delta network, with its per-head decay (`A_log`, `dt_bias`) and
/// the batch's rows continuing from their slots' states.
///
/// A decode batch walks its one token per row in the recurrent form; every other batch walks its
/// rows in chunks ([`DELTA_RULE_CHUNK`]) solved in their WY form.
pub struct GatedDeltaRule {
    a_log: Weight,
    dt_bias: Weight,
    spec: GatedDeltaRuleSpec,
}

impl GatedDeltaRule {
    /// The parameters at `scope`: `A_log` and `dt_bias`, one per value head, this rank's window of
    /// them starting at value head `first_head`.
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks a parameter or holds another extent.
    pub fn load(
        scope: &mut Scope<'_, '_>,
        spec: GatedDeltaRuleSpec,
        first_head: u64,
    ) -> Result<Self> {
        let heads = u64::from(spec.value_heads);
        Ok(Self {
            a_log: scope.rows("A_log", first_head, &[heads])?,
            dt_bias: scope.rows("dt_bias", first_head, &[heads])?,
            spec,
        })
    }

    /// The extents one slot of its state bank holds: each value head's `key_dim × value_dim` FP32
    /// state.
    pub fn state_dims(&self) -> Vec<u64> {
        vec![
            u64::from(self.spec.value_heads),
            u64::from(self.spec.key_dim),
            u64::from(self.spec.value_dim),
        ]
    }

    /// Run the rule over `query`, `key`, `value` (`[tokens, heads, dim]`, rows one stride apart),
    /// with the decay and step inputs `a` and `b` (`[tokens, value_heads]`), into
    /// `out [tokens, value_heads, value_dim]`.
    ///
    /// # Errors
    ///
    /// What the operator refuses.
    #[allow(
        clippy::too_many_arguments,
        reason = "the rule's five inputs, its batch, its two banks and its output"
    )]
    pub fn forward(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        view: &StateView<'_>,
        (state, checkpoint): (TensorView, TensorView),
        (query, key, value): (TensorView, TensorView, TensorView),
        (a, b): (TensorView, TensorView),
        out: TensorView,
    ) -> Result<()> {
        let (form, chunk) = match batch.phase {
            PhaseKind::Decode => (Form::Recurrent, 1),
            _ => (Form::Chunked, DELTA_RULE_CHUNK),
        };
        if !batch.page_size.is_multiple_of(DELTA_RULE_CHUNK) && form == Form::Chunked {
            return Err(Error::shape(
                "GatedDeltaRule",
                format!(
                    "a page of {} tokens is no whole number of {DELTA_RULE_CHUNK}-token chunks, so \
                     its checkpoints fall inside a chunk",
                    batch.page_size
                ),
            ));
        }
        let spec = self.spec;
        f.operate::<GatedDeltaRuleOp>(
            &GatedDeltaRuleDescriptor {
                dtype: query.dtype,
                key_heads: spec.key_heads,
                value_heads: spec.value_heads,
                key_dim: spec.key_dim,
                value_dim: spec.value_dim,
                block: DELTA_RULE_BLOCK.min(spec.value_dim),
                form,
                chunk,
            },
            GatedDeltaRuleArgs {
                query,
                key,
                value,
                a,
                b,
                a_log: self.a_log.view(),
                dt_bias: self.dt_bias.view(),
                rows: rows(batch, view),
                state,
                checkpoint,
                out,
            },
        )
    }
}
