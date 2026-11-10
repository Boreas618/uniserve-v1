//! The gated delta network: a linear-attention token mixer whose state a request slot keeps.
//!
//! The block is Qwen3-Next's and Qwen3.5's `GatedDeltaNet` (Hugging Face's
//! `Qwen3_5MoeGatedDeltaNet`, vLLM's and SGLang's `Qwen3_5GatedDeltaNet`):
//!
//! ```text
//! qkv = in_proj_qkv(x)                       [q | k | v], key heads, key heads, value heads
//! qkv = causal_conv1d(qkv)                   depthwise, SiLU, continuing from the slot's inputs
//! o   = gated_delta_rule(q, k, v, a = in_proj_a(x), b = in_proj_b(x))
//! o   = rms_norm(o) · weight · silu(in_proj_z(x))     per value head
//! out = out_proj(o)
//! ```
//!
//! Both recurrent stages read and advance the state the batch's request slots keep: field 0 of a
//! layer's banks is the convolution's last inputs, field 1 the delta rule's matrices
//! ([`GatedDeltaNet::state_fields`]). The block's intermediates are reserved from the forward's
//! arena under a mark released before it returns.

use ops::{DType, Shape, Strides, TensorView};

use crate::capability::{StateView, TokenBatch};
use crate::error::{Error, Result};
use crate::family::{Split, StateField};
use crate::forward::Forward;
use crate::layers::{
    Bias, CausalConv1d, CausalConv1dSpec, FieldBanks, GatedDeltaRule, GatedDeltaRuleSpec,
    GatedRmsNorm, Linear, Parallel,
};
use crate::weights::Scope;

/// The geometry of a [`GatedDeltaNet`], before any split.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedDeltaNetSpec {
    /// The hidden width the input projections contract and the output projection writes.
    pub hidden: u32,
    /// Query and key heads.
    pub key_heads: u32,
    /// Value heads, each with its own state; a multiple of the key heads.
    pub value_heads: u32,
    /// Elements of a query or key head.
    pub key_dim: u32,
    /// Elements of a value head.
    pub value_dim: u32,
    /// Taps of the short convolution.
    pub conv_width: u32,
    /// The output norm's variance floor.
    pub eps: f32,
}

impl GatedDeltaNetSpec {
    /// Channels the short convolution mixes: the queries, the keys and the values.
    pub fn conv_channels(&self) -> u32 {
        2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim
    }
}

/// A gated delta network block.
pub struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    conv1d: CausalConv1d,
    rule: GatedDeltaRule,
    norm: GatedRmsNorm,
    out_proj: Linear,
    spec: GatedDeltaNetSpec,
}

impl GatedDeltaNet {
    /// The block the module at `scope` states, at `split`'s rank.
    ///
    /// # Errors
    ///
    /// A module the loader refuses, value heads the key heads do not divide, or a split of more
    /// than one rank: the convolution's channels interleave three head windows, which no loaded
    /// layout shards yet.
    pub fn load(scope: &mut Scope<'_, '_>, spec: GatedDeltaNetSpec, split: Split) -> Result<Self> {
        let path = scope.path().to_owned();
        if spec.key_heads == 0 || !spec.value_heads.is_multiple_of(spec.key_heads) {
            return Err(Error::weight(
                &path,
                format!(
                    "{} value heads do not group over {} key heads",
                    spec.value_heads, spec.key_heads
                ),
            ));
        }
        if split.size != 1 {
            return Err(Error::weight(
                &path,
                format!(
                    "a gated delta network is served on one tensor rank, not {}",
                    split.size
                ),
            ));
        }
        let channels = spec.conv_channels();
        let value_width = spec.value_heads * spec.value_dim;
        let mut projection = |module: &str, inputs: u32, outputs: u32| {
            Linear::load(
                &mut scope.scope(module),
                inputs,
                outputs,
                Bias::Absent,
                Parallel::None,
            )
        };
        let in_proj_qkv = projection("in_proj_qkv", spec.hidden, channels)?;
        let in_proj_z = projection("in_proj_z", spec.hidden, value_width)?;
        let in_proj_b = projection("in_proj_b", spec.hidden, spec.value_heads)?;
        let in_proj_a = projection("in_proj_a", spec.hidden, spec.value_heads)?;
        let conv1d = CausalConv1d::load(
            &mut scope.scope("conv1d"),
            CausalConv1dSpec {
                channels,
                width: spec.conv_width,
            },
        )?;
        let rule = GatedDeltaRule::load(
            scope,
            GatedDeltaRuleSpec {
                key_heads: spec.key_heads,
                value_heads: spec.value_heads,
                key_dim: spec.key_dim,
                value_dim: spec.value_dim,
            },
            0,
        )?;
        let norm = GatedRmsNorm::load(&mut scope.scope("norm"), spec.value_dim, spec.eps)?;
        let out_proj = Linear::load(
            &mut scope.scope("out_proj"),
            value_width,
            spec.hidden,
            Bias::Absent,
            Parallel::None,
        )?;
        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv1d,
            rule,
            norm,
            out_proj,
            spec,
        })
    }

    /// The output projection, whose representation a caller may ask about (a packed product
    /// writes whole row classes).
    pub fn output(&self) -> &Linear {
        &self.out_proj
    }

    /// The state one layer keeps per request slot, in the order its banks are handed to it: the
    /// convolution's last inputs in the activations' type, then the delta rule's FP32 matrices.
    pub fn state_fields(spec: &GatedDeltaNetSpec, dtype: DType) -> Vec<StateField> {
        vec![
            StateField {
                name: "conv",
                dtype,
                dims: vec![
                    u64::from(spec.conv_width - 1),
                    u64::from(spec.conv_channels()),
                ],
            },
            StateField {
                name: "recurrent",
                dtype: DType::F32,
                dims: vec![
                    u64::from(spec.value_heads),
                    u64::from(spec.key_dim),
                    u64::from(spec.value_dim),
                ],
            },
        ]
    }

    /// Mix `hidden [tokens, hidden]` through the block into `out [tokens, hidden]`, reading and
    /// advancing recurrent layer `layer` of `state`.
    ///
    /// # Errors
    ///
    /// A state that does not hold the layer's two fields, a view an operator refuses, or an arena
    /// that cannot hold the intermediates.
    pub fn forward(
        &self,
        f: &Forward,
        hidden: TensorView,
        batch: &TokenBatch,
        state: &StateView<'_>,
        layer: usize,
        out: TensorView,
    ) -> Result<()> {
        let banks = state.layer(layer)?;
        let ([conv, recurrent], [conv_kept, recurrent_kept]) = (banks.live, banks.checkpoint)
        else {
            return Err(Error::shape(
                "GatedDeltaNet",
                format!(
                    "recurrent layer {layer} holds {} live and {} checkpoint fields, two expected",
                    banks.live.len(),
                    banks.checkpoint.len()
                ),
            ));
        };
        // A batch that keeps a trail keeps one bank per field of every layer.
        let (conv_trail, recurrent_trail) = match banks.trail {
            None => (None, None),
            Some([conv, recurrent]) => (Some(*conv), Some(*recurrent)),
            Some(trail) => {
                return Err(Error::shape(
                    "GatedDeltaNet",
                    format!(
                        "recurrent layer {layer} keeps a trail of {} fields, two expected",
                        trail.len()
                    ),
                ));
            }
        };
        let arena = f.arena();
        let mark = arena.mark();
        let dtype = hidden.dtype;
        let tokens = hidden.rows();
        let spec = self.spec;

        let qkv = self.in_proj_qkv.apply(f, hidden, dtype)?;
        let z = self.in_proj_z.apply(f, hidden, dtype)?;
        let b = self.in_proj_b.apply(f, hidden, dtype)?;
        let a = self.in_proj_a.apply(f, hidden, dtype)?;

        let channels = u64::from(spec.conv_channels());
        let mixed = arena.tensor(dtype, Shape::new(&[tokens, channels]))?;
        self.conv1d.forward(
            f,
            state,
            FieldBanks {
                live: *conv,
                checkpoint: *conv_kept,
                trail: conv_trail,
            },
            qkv,
            mixed,
        )?;

        // The convolved row is [q | k | v]: three head-major views of one row stride.
        let key_width = u64::from(spec.key_heads) * u64::from(spec.key_dim);
        let heads = |offset: u64, count: u32, dim: u32| {
            TensorView::with_strides(
                mixed.offset_elements(offset),
                dtype,
                Shape::new(&[tokens, u64::from(count), u64::from(dim)]),
                Strides::new(&[channels, u64::from(dim), 1]),
            )
        };
        let query = heads(0, spec.key_heads, spec.key_dim);
        let key = heads(key_width, spec.key_heads, spec.key_dim);
        let value = heads(2 * key_width, spec.value_heads, spec.value_dim);

        let value_heads = u64::from(spec.value_heads);
        let value_dim = u64::from(spec.value_dim);
        let core = arena.tensor(dtype, Shape::new(&[tokens, value_heads, value_dim]))?;
        self.rule.forward(
            f,
            batch,
            state,
            FieldBanks {
                live: *recurrent,
                checkpoint: *recurrent_kept,
                trail: recurrent_trail,
            },
            (query, key, value),
            (a, b),
            core,
        )?;

        // The output norm runs per value head, gated by the same head's slice of `z`.
        let per_head = Shape::new(&[tokens * value_heads, value_dim]);
        let normed = arena.tensor(dtype, per_head)?;
        self.norm
            .forward(f, core.reshape(per_head), z.reshape(per_head), normed)?;
        self.out_proj.forward(
            f,
            normed.reshape(Shape::new(&[tokens, value_heads * value_dim])),
            out,
        )?;
        arena.release(mark);
        Ok(())
    }
}
