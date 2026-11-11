//! The diffusion block's attention: every head of this rank's rows projected, normalized and
//! partly rotated, then attended over the packed sequence.
//!
//! A layer holds the query, key and value projections, the output projection, the per-head
//! normalization gains and one attention. The trailing channels of every head pass the rotation
//! unchanged: H3 rotates 96 of each head's 128 channels over the (time, height, width) coordinates,
//! 16 frequencies per axis ([`crate::packing`]'s coordinates).
//!
//! ## The partition
//!
//! A rank holds a contiguous slice of the packed rows and every head; `nn::layers::UlyssesAttention`
//! trades the row slice for a head shard of the whole sequence around the attention and back, by
//! the deployment's projection mode. With one rank nothing is exchanged.
//!
//! ## The two attention kinds
//!
//! [`Layer::Dense`] attends every used row of the packed sequence: the queries are this rank's rows
//! and the keys are the request's used rows, both named by the step's boundary tables, so a padding
//! row is no key. [`Layer::Sparse`] is FastH3's video sparse attention over the segment packing
//! ([`crate::attention_sparse`]).
//!
//! ## Precision
//!
//! The dense attention runs in the checkpoint's own representation (BF16 activations with FP32
//! accumulation inside the fused kernels), as the reference's FlashAttention path does.

use nn::family::SequenceSplit;
use nn::layers::{
    DenseAttention, DenseSpec, Factors, Gains, Rotary, UlyssesAttention, UlyssesSpec, Varlen,
};
use nn::weights::{Weight, WeightLoader};
use nn::{Forward, Shape, Operand};

use crate::attention_sparse::Sparse;
use crate::config::{AttentionKind, TransformerConfig};
use crate::{Error, Result};

/// Which rows a query sees, by attention kind.
#[derive(Clone, Copy)]
pub enum Visibility {
    /// Every used row of the packed sequence.
    Dense(Varlen),
    /// FastH3's tiles: each tile's valid rows, its segment, and each segment's kept key tiles.
    Sparse(nn::layers::VsaTables),
}

/// One layer's attention, of whichever kind its checkpoint composes.
pub enum Layer {
    /// Full bidirectional attention over the packed sequence.
    Dense(Box<Dense>),
    /// FastH3's video sparse attention over the segment packing.
    Sparse(Box<Sparse>),
}

impl Layer {
    /// Read layer `index`'s attention of a DiT partition, of the kind the deployment composes.
    ///
    /// `bounds` is the deployment's `(packed rows, key tiles)`: the keys one call may read, and the
    /// tiles a sparse compression reduces over.
    pub fn load(
        loader: &mut WeightLoader<'_>,
        (config, kind): (&TransformerConfig, AttentionKind),
        index: u32,
        rotary: &Rotary,
        bounds: (u32, u32),
        sequence: SequenceSplit,
    ) -> Result<Self> {
        let mut scope = loader.scope(format!("transformer_blocks.{index}.attn"));
        match kind {
            AttentionKind::Dense => Ok(Self::Dense(Box::new(Dense::load(
                &mut scope, config, rotary, bounds.0, sequence,
            )?))),
            AttentionKind::Sparse { tile, .. } => Ok(Self::Sparse(Box::new(Sparse::load(
                &mut scope, config, rotary, tile, bounds, sequence,
            )?))),
        }
    }

    /// Attend `normalized`, this rank's `[rows, hidden]`, into a fresh `[rows, hidden]`, turned by
    /// the stack's `rotary` at the packed sequence's `factors` and seeing `visibility`.
    ///
    /// # Errors
    ///
    /// A visibility of the other kind, or what a layer refuses.
    pub fn forward(
        &self,
        f: &Forward,
        normalized: Operand,
        (rotary, factors): (&Rotary, Factors),
        visibility: &Visibility,
    ) -> Result<Operand> {
        match (self, visibility) {
            (Self::Dense(dense), Visibility::Dense(visible)) => {
                dense.forward(f, normalized, rotary, factors, *visible)
            }
            (Self::Sparse(sparse), Visibility::Sparse(tables)) => {
                sparse.forward(f, normalized, factors, tables)
            }
            _ => Err(Error::Unsupported(
                "a layer attends with the visibility of the kind it was built for".to_owned(),
            )),
        }
    }
}

/// The dense attention of one diffusion layer over this rank's heads of the whole sequence, inside
/// the sequence-parallel projections.
pub struct Dense {
    ulysses: UlyssesAttention<DenseAttention>,
    query_gain: Weight,
    key_gain: Weight,
    head_dim: u32,
}

impl Dense {
    /// Read the dense attention at `scope`; `keys` bounds the keys of one call.
    fn load(
        scope: &mut nn::weights::Scope<'_, '_>,
        config: &TransformerConfig,
        rotary: &Rotary,
        keys: u32,
        sequence: SequenceSplit,
    ) -> Result<Self> {
        let head = u64::from(config.head_dim);
        let query_gain = scope.scope("norm_q").tensor("weight", &[head])?;
        let key_gain = scope.scope("norm_k").tensor("weight", &[head])?;
        let local_heads = config.num_attention_heads / sequence.split.size.max(1);
        let attention = DenseAttention::new(DenseSpec {
            head_dim: config.head_dim,
            heads: local_heads,
            kv_heads: local_heads,
            max_keys: keys,
            causal: false,
            window: None,
            scale: rotary.spectrum().score_scale(config.head_dim)?,
        })?;
        let ulysses = UlyssesAttention::load(
            scope,
            UlyssesSpec {
                hidden: config.hidden_size,
                heads: config.num_attention_heads,
                head_dim: config.head_dim,
                inputs: &["to_q", "to_k", "to_v"],
                output: "to_out.0",
            },
            sequence.split,
            sequence.projection,
            |_| Ok(attention),
        )?;
        Ok(Self {
            ulysses,
            query_gain,
            key_gain,
            head_dim: config.head_dim,
        })
    }

    /// Project, normalize and rotate this rank's heads of every row, attend them and return the
    /// attended rows to their owners through the output projection.
    fn forward(
        &self,
        f: &Forward,
        normalized: Operand,
        rotary: &Rotary,
        factors: Factors,
        visible: Varlen,
    ) -> Result<Operand> {
        let ulysses = &self.ulysses;
        let rows = ulysses.rows(f, normalized)?;
        let [query, key, value] = [0, 1, 2].map(|index| ulysses.project(f, index, rows));
        let headed =
            |view: Operand| view.reshape(heads(view, ulysses.local_heads(), self.head_dim));
        let (query, key) = rotary.forward(
            f,
            headed(query?),
            headed(key?),
            Some(Gains {
                query: self.query_gain.view(),
                key: self.key_gain.view(),
            }),
            factors,
        )?;
        let attended = ulysses
            .inner()
            .forward(f, query, key, headed(value?), Some(visible))?;
        let width = u64::from(ulysses.local_heads()) * u64::from(self.head_dim);
        let attended = attended.reshape(Shape::new(&[attended.shape.numel() / width, width]));
        Ok(ulysses.return_rows(f, attended)?)
    }
}

/// A `[tokens, heads · head_dim]` view as `[tokens, heads, head_dim]`, the token axis taken from the
/// view's own element count (the exchanged and the local path differ in rows).
pub(crate) fn heads(view: Operand, count: u32, head_dim: u32) -> Shape {
    let width = u64::from(count) * u64::from(head_dim);
    Shape::new(&[
        view.shape.numel() / width.max(1),
        u64::from(count),
        u64::from(head_dim),
    ])
}

/// A channel count as a projection's width.
pub(crate) fn width(channels: u64) -> Result<u32> {
    u32::try_from(channels)
        .map_err(|_| Error::config(format!("{channels} channels exceed a projection's width")))
}
