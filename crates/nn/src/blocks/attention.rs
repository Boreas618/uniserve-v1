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

use ops::{Shape, TensorView};

use crate::capability::{KvView, TokenBatch};
use crate::error::{Error, Result};
use crate::family::Split;
use crate::forward::Forward;
use crate::layers::{
    Bias, DenseAttention, DenseSpec, Gains, Linear, PagedAttention, PagedSpec, Parallel, QkvBias,
    QkvLinear, QkvNames, QkvSpec, RmsNorm, Rotary, Varlen,
};
use crate::weights::{Scope, Weight};

/// The module names of a block's per-head q/k normalization, and its form.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QkNormNames<'n> {
    /// No normalization.
    None,
    /// Learned per-head gains (`<name>.weight`, `[head_dim]`), fused into the stack's rotation.
    Fused { query: &'n str, key: &'n str },
    /// Per-head RmsNorm modules of this variance floor, launched before any rotation.
    Separate {
        query: &'n str,
        key: &'n str,
        epsilon: f32,
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
    pub cosine: TensorView,
    pub sine: TensorView,
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
    /// The batch's keys written into the cache's planes of `layer`, then the cached keys read.
    Paged {
        batch: &'a TokenBatch,
        cache: &'a KvView<'a>,
        group: usize,
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
        let heads = qkv.query().n() / head_dim;
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
            } => QkNorm::Separate {
                query: RmsNorm::load(&mut scope.scope(query), head_dim, epsilon)?,
                key: RmsNorm::load(&mut scope.scope(key), head_dim, epsilon)?,
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
        Ok(Self {
            qkv,
            qk_norm,
            attention,
            output,
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

    /// `[rows, count · head_dim]` as `[rows, count, head_dim]`.
    fn heads_of(&self, view: TensorView, count: u32) -> TensorView {
        view.reshape(Shape::new(&[
            view.rows(),
            u64::from(count),
            u64::from(self.head_dim),
        ]))
    }

    /// Attend `hidden [rows, hidden]` and write the output projection into `out [rows, hidden]`.
    ///
    /// `rope` is the stack's rotary and the forward's factors, absent for a block that does not
    /// rotate; a fused q/k norm needs it. The block's intermediates are reserved from the arena
    /// and released before return.
    ///
    /// # Errors
    ///
    /// A fused norm without a rotation, keys of another kind than the block's attention, a view
    /// a layer refuses, or an arena that cannot hold the intermediates.
    pub fn forward(
        &self,
        f: &Forward,
        hidden: TensorView,
        rope: Option<RopeFactors<'_>>,
        keys: Keys<'_>,
        out: TensorView,
    ) -> Result<()> {
        let arena = f.arena();
        let mark = arena.mark();
        let dtype = hidden.dtype;
        let query = self.heads_of(self.qkv.query().apply(f, hidden, dtype)?, self.heads);
        let key = self.heads_of(self.qkv.key().apply(f, hidden, dtype)?, self.kv_heads);
        let value = self.heads_of(self.qkv.value().apply(f, hidden, dtype)?, self.kv_heads);

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
                let turned = (
                    arena.tensor(dtype, query.shape)?,
                    arena.tensor(dtype, key.shape)?,
                );
                rope.rotary.apply(
                    f,
                    query,
                    key,
                    gains,
                    rope.cosine,
                    rope.sine,
                    turned.0,
                    turned.1,
                )?;
                turned
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
                let per_head = |view: TensorView| {
                    view.reshape(Shape::new(&[
                        view.numel() / u64::from(self.head_dim),
                        u64::from(self.head_dim),
                    ]))
                };
                let normed = (
                    arena.tensor(dtype, query.shape)?,
                    arena.tensor(dtype, key.shape)?,
                );
                q_norm.forward(f, per_head(query), None, per_head(normed.0))?;
                k_norm.forward(f, per_head(key), None, per_head(normed.1))?;
                match rope {
                    None => normed,
                    Some(rope) => {
                        let turned = (
                            arena.tensor(dtype, query.shape)?,
                            arena.tensor(dtype, key.shape)?,
                        );
                        rope.rotary.apply(
                            f,
                            normed.0,
                            normed.1,
                            None,
                            rope.cosine,
                            rope.sine,
                            turned.0,
                            turned.1,
                        )?;
                        turned
                    }
                }
            }
        };

        let attended = arena.tensor(dtype, query.shape)?;
        match (&self.attention, keys) {
            (Attention::Dense(attention), Keys::Dense(varlen)) => {
                attention.forward(f, query, key, value, varlen, attended)?;
            }
            (
                Attention::Paged(attention),
                Keys::Paged {
                    batch,
                    cache,
                    group,
                    layer,
                },
            ) => attention.forward(f, batch, cache, group, layer, query, key, value, attended)?,
            _ => {
                return Err(Error::Unsupported(
                    "QkvAttention: the forward's keys are not of the block's attention kind"
                        .to_owned(),
                ));
            }
        }
        // Row parallel: the projection sums each rank's partial output over the head split.
        let flat = Shape::new(&[
            attended.shape.dim(0),
            u64::from(self.heads) * u64::from(self.head_dim),
        ]);
        self.output.forward(f, attended.reshape(flat), out)?;
        arena.release(mark);
        Ok(())
    }
}
