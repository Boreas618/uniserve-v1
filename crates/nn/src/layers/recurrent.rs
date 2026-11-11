//! The recurrent layers of a gated delta network: the short causal convolution and the gated delta
//! rule, each advancing a state the batch's request slots keep (`ops::recurrent`).
//!
//! Both read the batch's state rows ([`StateView`]'s packed offsets, lengths, slots and checkpoints),
//! which are its attention rows except in a verification: a row that starts at position zero
//! begins from a zero state, every other row continues from its slot's, and a row that crosses a
//! checkpoint keeps its state there for the prefix cache. A batch that keeps a trail
//! ([`crate::capability::StateLayer::trail`]) has each operator write the state after every token
//! as well. The banks are the infrastructure's; a layer only launches its operator over them.

use ops::recurrent::{
    CausalConv1d as CausalConv1dOp, CausalConv1dArgs, CausalConv1dDescriptor, Form,
    GatedDeltaRule as GatedDeltaRuleOp, GatedDeltaRuleArgs, GatedDeltaRuleDescriptor, RowColumns,
    Trail,
};
use ops::{Operand, Shape};
use proto::module::PhaseKind;

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

/// The row columns the operators read: the state view's rows.
fn rows(state: &StateView<'_>) -> RowColumns {
    RowColumns {
        query_offsets: state.query_offsets,
        seq_lens: state.seq_lens,
        slots: state.slots,
        checkpoints: state.checkpoints,
    }
}

/// One field's banks a recurrent operator reads and writes: the slots' live state, their
/// checkpoints, and the batch's trail when it keeps one.
#[derive(Clone, Copy)]
pub struct FieldBanks {
    /// `[slots, …]`: the live state.
    pub live: Operand,
    /// `[slots, …]`: the checkpoints rows keep.
    pub checkpoint: Operand,
    /// `[tokens, …]`: the state after each token, for a batch that keeps a trail.
    pub trail: Option<Operand>,
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

    /// `x [tokens, channels]` (rows any stride apart) convolved into a fresh packed
    /// `[tokens, channels]` tensor of its element type, continuing from and advancing
    /// `banks.live` in place, keeping `banks.checkpoint` where the view asks and every token's
    /// window in `banks.trail` when the batch keeps one.
    ///
    /// # Errors
    ///
    /// An arena that cannot hold the output, or what the operator refuses.
    pub fn forward(
        &self,
        f: &Forward,
        view: &StateView<'_>,
        banks: FieldBanks,
        x: Operand,
    ) -> Result<Operand> {
        let channels = u64::from(self.spec.channels);
        let out = f.empty(x.dtype, Shape::new(&[x.shape.dim(0), channels]))?;
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
                rows: rows(view),
                state: banks.live,
                checkpoint: banks.checkpoint,
                trail: banks.trail.map(|bank| Trail { bank }),
            },
        )?;
        Ok(out)
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

    /// The rule over `query`, `key`, `value` (`[tokens, heads, dim]`, rows one stride apart), with
    /// the decay and step inputs `a` and `b` (`[tokens, value_heads]`): a fresh
    /// `[tokens, value_heads, value_dim]` tensor of the query's element type, the banks advanced
    /// in place.
    ///
    /// A decode-phase batch (a decode, or a verification) walks its state rows one token at a
    /// time and keeps the trail `banks` carries; any other walks them in chunks.
    ///
    /// # Errors
    ///
    /// An arena that cannot hold the output, or what the operator refuses.
    pub fn forward(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        view: &StateView<'_>,
        banks: FieldBanks,
        (query, key, value): (Operand, Operand, Operand),
        (a, b): (Operand, Operand),
    ) -> Result<Operand> {
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
        let out = f.empty(
            query.dtype,
            Shape::new(&[
                query.shape.dim(0),
                u64::from(spec.value_heads),
                u64::from(spec.value_dim),
            ]),
        )?;
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
                rows: rows(view),
                state: banks.live,
                checkpoint: banks.checkpoint,
                trail: banks.trail.map(|bank| Trail { bank }),
                out,
            },
        )?;
        Ok(out)
    }
}
