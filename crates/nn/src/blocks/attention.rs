//! The QKV attention block: project, normalize and rotate each head, attend, project out.
//!
//! The block holds the query, key and value projections ([`QkvLinear`], column parallel over the
//! split's heads), the per-head q/k normalization in the form the consumer's baseline computes, the
//! attention, and the row-parallel output projection. The rotary embedding is the stack's
//! ([`Rotary`]), handed in with the forward's factor tables, because every layer turns by the same
//! factors.
//!
//! ## The q/k normalization
//!
//! ```text
//! QkNorm::None      no normalization; the rotation alone, when the stack rotates
//! QkNorm::Fused     per-head gains fused into the stack's rotation (`qk_norm_rope`)
//! QkNorm::Separate  a per-head RmsNorm of each projection, then the rotation alone if any
//! ```
//!
//! The fused and the separate forms round differently (the fused program normalizes and rotates in
//! FP32 and rounds once), so they are two forms and a consumer keeps its baseline's.
//!
//! ## The output gate
//!
//! A gated block (Qwen3-Next's and Qwen3.5's `attn_output_gate`, [`QkvSpec::gated`]) projects each
//! query head together with a gate of the head's width; the attended heads are multiplied by the
//! sigmoid of their gates before the output projection:
//!
//! ```text
//! [q_h | g_h] = q_proj(x)_h;   o_h = attention(q, k, v)_h · sigmoid(g_h);   out = o_proj(o)
//! ```
//!
//! The query halves are gathered into a packed tensor for the normalization and rotation, and the
//! gate halves are read where the projection wrote them.

use ops::{DType, Operand, Shape};

use crate::capability::{KvLayer, KvView, TokenBatch};
use crate::error::{Error, Result};
use crate::family::Split;
use crate::forward::Forward;
use crate::layers::Factors;
use crate::layers::{Access, Copy, GateBias, GatedActivation, Halves, Rounding};
use crate::layers::{
    Bias, DenseAttention, DenseSpec, Gain, Gains, Linear, PagedAttention, PagedSpec, Parallel,
    QkvBias, QkvLinear, QkvNames, QkvSpec, RmsNorm, Rotary, Varlen,
};
use crate::weights::{Scope, Weight};
use ops::activation::Activation;

/// The module names of a block's per-head q/k normalization, and its form.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QkNormNames<'n> {
    /// No normalization.
    None,
    /// Learned per-head gains (`<name>.weight`, `[head_dim]`), fused into the stack's rotation.
    Fused { query: &'n str, key: &'n str },
    /// Per-head RmsNorm modules of this variance floor, launched before any rotation and writing
    /// the projections' element type `output`.
    Separate {
        query: &'n str,
        key: &'n str,
        epsilon: f32,
        output: DType,
    },
}

/// Which attention the block attends with, and the geometry its descriptor pins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AttentionKind {
    /// Dense attention over packed sequences, bounded by `max_keys`.
    Dense {
        max_keys: u32,
        causal: bool,
        window: Option<u32>,
    },
    /// Paged causal attention over the prefix cache.
    /// The page size is the cache's, which the forward reads ([`crate::KvGroup::page_size`]).
    Paged {
        max_kv_len: u32,
        window: Option<u32>,
    },
}

/// The names, geometry and arithmetic of a [`QkvAttention`].
#[derive(Clone, Copy, Debug)]
pub struct QkvAttentionSpec<'n> {
    /// The heads of the whole module, before any split.
    pub qkv: QkvSpec,
    /// Where the checkpoint stores the three input projections.
    pub names: QkvNames<'n>,
    /// Their biases.
    pub bias: QkvBias<'n>,
    /// The output projection's module name.
    pub output: &'n str,
    /// Whether the output projection adds a bias.
    pub output_bias: Bias,
    /// The per-head normalization.
    pub qk_norm: QkNormNames<'n>,
    /// The attention and its geometry.
    pub kind: AttentionKind,
    /// The softmax scale applied to `q · k`.
    pub scale: f32,
}

/// The stack's rotary and the forward's factor tables, `[tokens, columns]` FP32 each.
#[derive(Clone, Copy)]
pub struct RopeFactors<'a> {
    pub rotary: &'a Rotary,
    pub factors: Factors,
}

/// What one forward's keys are: packed sequences, or the paged cache of one layer.
#[derive(Clone, Copy)]
#[allow(
    clippy::large_enum_variant,
    reason = "one value per attention call, built on the stack and consumed by that call"
)]
pub enum Keys<'a> {
    /// The forward's own keys, as one sequence (`None`) or the sequences a [`Varlen`] names.
    Dense(Option<Varlen>),
    /// The batch's keys written into the layer's cache, then the cached keys read.
    Paged {
        batch: &'a TokenBatch,
        cache: KvLayer<'a>,
    },
    /// The recurrent state of the cache's recurrent layer `layer`, which a linear-attention mixer
    /// reads and advances in place of keys ([`crate::blocks::GatedDeltaNet`]).
    Recurrent {
        batch: &'a TokenBatch,
        cache: &'a KvView<'a>,
        layer: usize,
    },
}

/// The per-head normalization a block holds.
#[allow(
    clippy::large_enum_variant,
    reason = "one value per block, held for the model's lifetime and read on every forward"
)]
enum QkNorm {
    None,
    Fused { query: Weight, key: Weight },
    Separate { query: RmsNorm, key: RmsNorm },
}

/// The attention a block holds.
enum Attention {
    Dense(DenseAttention),
    Paged(PagedAttention),
}

/// A QKV attention block.
pub struct QkvAttention {
    qkv: QkvLinear,
    qk_norm: QkNorm,
    attention: Attention,
    output: Linear,
    /// The sigmoid gate of a gated block's attended heads, `None` for an ungated one.
    gate: Option<GatedActivation>,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
}

impl QkvAttention {
    /// The block the module at `scope` states, at `split`'s rank of its tensor-parallel group.
    ///
    /// The query and key heads are this rank's share (the key heads replicated when the group is
    /// wider than they are); a group of one holds the output projection whole.
    ///
    /// # Errors
    ///
    /// A projection or gain the loader refuses, a split that does not divide the heads, or an
    /// attention its layer refuses.
    pub fn load(
        scope: &mut Scope<'_, '_>,
        spec: QkvAttentionSpec<'_>,
        split: Split,
    ) -> Result<Self> {
        let head_dim = spec.qkv.head_dim;
        let qkv = QkvLinear::load(scope, spec.qkv, spec.names, spec.bias, split)?;
        let gated = spec.qkv.gated;
        let heads = qkv.query().n() / head_dim / if gated { 2 } else { 1 };
        let kv_heads = qkv.key().n() / head_dim;
        let parallel = if split.size == 1 {
            Parallel::None
        } else {
            Parallel::Row(split)
        };
        let output = Linear::load(
            &mut scope.scope(spec.output),
            spec.qkv.heads * head_dim,
            spec.qkv.hidden,
            spec.output_bias,
            parallel,
        )?;
        let width = u64::from(head_dim);
        let qk_norm = match spec.qk_norm {
            QkNormNames::None => QkNorm::None,
            QkNormNames::Fused { query, key } => QkNorm::Fused {
                query: scope.scope(query).tensor("weight", &[width])?,
                key: scope.scope(key).tensor("weight", &[width])?,
            },
            QkNormNames::Separate {
                query,
                key,
                epsilon,
                output,
            } => QkNorm::Separate {
                query: RmsNorm::load(
                    &mut scope.scope(query),
                    head_dim,
                    epsilon,
                    Gain::Plain,
                    output,
                )?,
                key: RmsNorm::load(
                    &mut scope.scope(key),
                    head_dim,
                    epsilon,
                    Gain::Plain,
                    output,
                )?,
            },
        };
        let attention = match spec.kind {
            AttentionKind::Dense {
                max_keys,
                causal,
                window,
            } => Attention::Dense(DenseAttention::new(DenseSpec {
                head_dim,
                heads,
                kv_heads,
                max_keys,
                causal,
                window,
                scale: spec.scale,
            })?),
            AttentionKind::Paged { max_kv_len, window } => {
                Attention::Paged(PagedAttention::new(PagedSpec {
                    head_dim,
                    heads,
                    kv_heads,
                    max_kv_len,
                    window,
                    scale: spec.scale,
                })?)
            }
        };
        let gate = if gated {
            Some(GatedActivation::new(
                Activation::Sigmoid,
                Rounding::Fused,
                GateBias::None,
            )?)
        } else {
            None
        };
        Ok(Self {
            qkv,
            qk_norm,
            attention,
            output,
            gate,
            heads,
            kv_heads,
            head_dim,
        })
    }

    /// The query projection, whose representation a caller may ask about (a packed product
    /// writes whole row classes).
    pub fn query(&self) -> &Linear {
        self.qkv.query()
    }

    /// A gated projection `[rows, heads · 2 · head_dim]` as its packed queries
    /// `[rows, heads, head_dim]`, gathered from the projection, and its gates
    /// `[rows · heads, head_dim]`, read in place.
    fn split_gate(&self, f: &Forward, projected: Operand) -> Result<(Operand, Option<Operand>)> {
        let rows = projected.rows();
        let heads = u64::from(self.heads);
        let head_dim = u64::from(self.head_dim);
        let dtype = projected.dtype;
        // A head's query and gate sit side by side: the halves are two views of one head stride.
        let half = |offset: u64| {
            Operand::with_strides(
                projected.offset_elements(offset),
                dtype,
                Shape::new(&[1, rows, heads, head_dim]),
                ops::Strides::new(&[
                    rows * heads * 2 * head_dim,
                    heads * 2 * head_dim,
                    2 * head_dim,
                    1,
                ]),
            )
        };
        let query = f.empty(dtype, Shape::new(&[rows, heads, head_dim]))?;
        let packed = query.reshape(Shape::new(&[1, rows, heads, head_dim]));
        let strides = [
            (rows * heads * head_dim) as i64,
            (heads * head_dim) as i64,
            head_dim as i64,
            1,
        ];
        Copy::new(dtype, Access::Vector)?.forward(f, half(0), packed, &strides)?;
        let gate = Operand::with_strides(
            projected.offset_elements(head_dim),
            dtype,
            Shape::new(&[rows * heads, head_dim]),
            ops::Strides::new(&[2 * head_dim, 1]),
        );
        Ok((query, Some(gate)))
    }

    /// `[rows, count · head_dim]` as `[rows, count, head_dim]`.
    fn heads_of(&self, view: Operand, count: u32) -> Operand {
        view.reshape(Shape::new(&[
            view.rows(),
            u64::from(count),
            u64::from(self.head_dim),
        ]))
    }

    /// `hidden [rows, hidden]` attended and projected out: a fresh `[rows, hidden]` tensor of
    /// `hidden`'s element type.
    ///
    /// `rope` is the stack's rotary and the forward's factors, absent for a block that does not
    /// rotate; a fused q/k norm needs it. The output is reserved first and the block's
    /// intermediates after it, in a scope that hands them back once the output projection is
    /// enqueued.
    ///
    /// # Errors
    ///
    /// A fused norm without a rotation, keys of another kind than the block's attention, a view
    /// a layer refuses, or an arena that cannot hold the output and intermediates.
    pub fn forward(
        &self,
        f: &Forward,
        hidden: Operand,
        rope: Option<RopeFactors<'_>>,
        keys: Keys<'_>,
    ) -> Result<Operand> {
        let flat = |rows: u64| Shape::new(&[rows, u64::from(self.output.k())]);
        // The output projection's rows are `hidden`'s, which is all its reservation reads.
        let out = self.output.reserve(f, hidden)?;
        f.scoped(|f| {
            let attended = self.attend(f, hidden, rope, keys)?;
            // Row parallel: the projection sums each rank's partial output over the head split.
            self.output
                .launch(f, attended.reshape(flat(attended.shape.dim(0))), out)
        })?;
        Ok(out)
    }

    /// The attended heads of `hidden`, `[rows, heads, head_dim]`, gated when the block is.
    fn attend(
        &self,
        f: &Forward,
        hidden: Operand,
        rope: Option<RopeFactors<'_>>,
        keys: Keys<'_>,
    ) -> Result<Operand> {
        let (projected, key, value) = self.qkv.forward(f, hidden)?;
        let (query, gate) = match self.gate {
            None => (self.heads_of(projected, self.heads), None),
            Some(_) => self.split_gate(f, projected)?,
        };
        let key = self.heads_of(key, self.kv_heads);
        let value = self.heads_of(value, self.kv_heads);

        // Each head normalized and turned, in the block's form.
        let (query, key) = match (&self.qk_norm, rope) {
            (QkNorm::None, None) => (query, key),
            (QkNorm::Fused { .. }, None) => {
                return Err(Error::shape(
                    "QkvAttention",
                    "a q/k norm fused into the rotation needs the stack's rotation".to_owned(),
                ));
            }
            (QkNorm::None | QkNorm::Fused { .. }, Some(rope)) => {
                let gains = match &self.qk_norm {
                    QkNorm::Fused { query, key } => Some(Gains {
                        query: query.view(),
                        key: key.view(),
                    }),
                    _ => None,
                };
                rope.rotary.forward(f, query, key, gains, rope.factors)?
            }
            (
                QkNorm::Separate {
                    query: q_norm,
                    key: k_norm,
                },
                rope,
            ) => {
                // The norm is over one head's channels, so it reads each projection as one row
                // per head.
                let per_head = |view: Operand| {
                    view.reshape(Shape::new(&[
                        view.numel() / u64::from(self.head_dim),
                        u64::from(self.head_dim),
                    ]))
                };
                let normed = (
                    q_norm
                        .forward(f, per_head(query), None)?
                        .reshape(query.shape),
                    k_norm.forward(f, per_head(key), None)?.reshape(key.shape),
                );
                match rope {
                    None => normed,
                    Some(rope) => rope
                        .rotary
                        .forward(f, normed.0, normed.1, None, rope.factors)?,
                }
            }
        };

        let attended = match (&self.attention, keys) {
            (Attention::Dense(attention), Keys::Dense(varlen)) => {
                attention.forward(f, query, key, value, varlen)?
            }
            (Attention::Paged(attention), Keys::Paged { batch, cache }) => {
                attention.forward(f, batch, cache, query, key, value)?
            }
            _ => {
                return Err(Error::Unsupported(
                    "QkvAttention: the forward's keys are not of the block's attention kind"
                        .to_owned(),
                ));
            }
        };
        match (&self.gate, gate) {
            (Some(activation), Some(gate)) => {
                // Each attended head is scaled by the sigmoid of its own gate.
                let rows = attended.reshape(Shape::new(&[
                    attended.shape.dim(0) * u64::from(self.heads),
                    u64::from(self.head_dim),
                ]));
                Ok(activation
                    .forward(f, Halves::Split { gate, value: rows })?
                    .reshape(attended.shape))
            }
            _ => Ok(attended),
        }
    }
}
