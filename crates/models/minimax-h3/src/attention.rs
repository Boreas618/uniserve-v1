//! The diffusion block's attention: every head of this rank's rows projected, normalized and
//! partly rotated, then attended densely over the packed sequence.
//!
//! A layer holds one merged query/key/value projection, an output projection, the per-head
//! normalization gains and the rotary embedding of the head, and one attention. The trailing
//! channels of every head pass the rotation unchanged: H3 rotates 96 of each head's 128 channels
//! over the (time, height, width) coordinates, 16 frequencies per axis, interleaved
//! ([`crate::packing`]'s coordinates).
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
//! and the keys are the request's used rows, both named by device boundary tables, so a padding row
//! is no key. [`Layer::Sparse`] is FastH3's video sparse attention over the segment packing
//! ([`crate::attention_sparse`]), over `nn::layers::VsaAttention`.
//!
//! ## Precision
//!
//! The dense attention runs in the checkpoint's own representation - BF16 activations with FP32
//! accumulation inside the fused kernels - which is the FlashAttention row of the design's numerics
//! table; nothing here declares an FP32 attention and no layer casts to one. The sparse path's
//! **compression** branch is the other row: the design requires FP32 there (cuBLASLt SGEMM plus a
//! softmax program), and the reference computes its tile means, scores and weighting in FP32
//! throughout, so that branch declares
//! FP32 when it lands - in its layer's dtype and in this doc - while the fine sparse attention over
//! the prepared rows stays in the checkpoint's dtype like the dense one.

use nn::family::{Projection, Split};
use nn::layers::{DenseAttention, DenseSpec, Gains, Rotary, UlyssesAttention, UlyssesSpec, Varlen};
use nn::weights::{Weight, WeightLoader};
use nn::{Forward, Shape, TensorView};

use crate::config::TransformerConfig;
use crate::partition::Partition;
use crate::{Error, Result};

/// One layer's attention, of whichever kind its checkpoint composes.
pub enum Layer {
    /// Full bidirectional attention over the packed sequence.
    Dense(Box<Dense>),
    /// FastH3's video sparse attention over the segment packing.
    Sparse(Box<crate::attention_sparse::Sparse>),
}

impl Layer {
    /// Read layer `index`'s attention of a DiT partition, of the kind the deployment composes.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        kind: crate::config::AttentionKind,
        index: u32,
        config: &TransformerConfig,
        rows: u32,
        rotary: &Rotary,
        tile_bound: u32,
        partition: &Partition,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        match kind {
            crate::config::AttentionKind::Dense => Ok(Self::Dense(Box::new(Dense::load(
                index, config, rows, rotary, partition, loader,
            )?))),
            crate::config::AttentionKind::Sparse { tile, .. } => Ok(Self::Sparse(Box::new(
                crate::attention_sparse::Sparse::load(
                    index, config, rows, rotary, tile, tile_bound, partition, loader,
                )?,
            ))),
        }
    }

    /// Reserve the scratch this layer's kind reads, once for the whole stack.
    pub fn reserve(
        &self,
        f: &Forward,
        normalized: TensorView,
        total_rows: u32,
    ) -> Result<LayerScratch> {
        Ok(match self {
            Self::Dense(dense) => LayerScratch::Dense(Box::new(Scratch::reserve(
                dense, f, normalized, total_rows,
            )?)),
            Self::Sparse(sparse) => LayerScratch::Sparse(Box::new(
                crate::attention_sparse::Scratch::reserve(sparse, f, normalized, total_rows)?,
            )),
        })
    }

    /// Attend `normalized` `[rows, hidden]` of this rank and write `[rows, hidden]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        f: &Forward,
        normalized: TensorView,
        factors: Factors<'_>,
        visible: Option<&Varlen>,
        segments: Option<&crate::attention_sparse::Segments>,
        scratch: &LayerScratch,
        out: TensorView,
    ) -> Result<()> {
        match (self, scratch) {
            (Self::Dense(dense), LayerScratch::Dense(scratch)) => {
                let visible = visible.ok_or_else(|| {
                    Error::Unsupported("a dense layer needs the sequence's visibility".to_owned())
                })?;
                dense.forward(f, normalized, factors, visible, scratch, out)
            }
            (Self::Sparse(sparse), LayerScratch::Sparse(scratch)) => {
                let views = segments.ok_or_else(|| {
                    // The caller is the denoiser's step, and it states these views from the
                    // deployment's attention kind (`Denoiser::step`). So this refusal is what a
                    // deployment reads when the kind its step asked with and the kind this layer was
                    // built with disagree - which one value cannot do, and a stale binary can.
                    Error::Unsupported(
                        "a sparse layer was handed no segment views: the caller states \
                         `variable_block_sizes`, `tile_segments` and `segment_keep` from the \
                         deployment's attention kind, and this layer was built sparse"
                            .to_owned(),
                    )
                })?;
                sparse.forward(
                    f,
                    normalized,
                    factors,
                    views.sizes,
                    views.segments,
                    views.keep,
                    scratch,
                    out,
                )
            }
            _ => Err(Error::Unsupported(
                "a layer's scratch is of its attention kind".to_owned(),
            )),
        }
    }
}

/// The scratch of whichever kind a stack's layers are.
pub enum LayerScratch {
    Dense(Box<Scratch>),
    Sparse(Box<crate::attention_sparse::Scratch>),
}

/// The stack's rotary and its factor tables of one forward, `[tokens, rotated / 2]` FP32 each.
#[derive(Clone, Copy)]
pub struct Factors<'a> {
    /// The stack's one rotary: every layer turns by it, with its own per-head gains.
    pub rotary: &'a Rotary,
    /// The cosine table of the packed rows.
    pub cosine: TensorView,
    /// The sine table of the packed rows.
    pub sine: TensorView,
}

/// The dense attention of one diffusion layer: the stack's rotation and a dense attention over
/// this rank's heads of the whole sequence, inside the sequence-parallel projections.
pub struct Dense {
    ulysses: UlyssesAttention<DenseAttention>,
    query_gain: Weight,
    key_gain: Weight,
    head_dim: u32,
}

/// The sequence-parallel split of a DiT partition: its member count and this rank's place.
pub(crate) fn sequence_split(partition: &Partition) -> Result<Split> {
    let size = u32::try_from(partition.ulysses_degree)
        .map_err(|_| Error::config("a sequence degree wider than a u32".to_owned()))?;
    let rank = u32::try_from(partition.sequence_rank)
        .map_err(|_| Error::config("a sequence rank wider than a u32".to_owned()))?;
    Ok(Split { size, rank })
}

impl Dense {
    /// Read layer `index`'s dense attention of a DiT partition.
    ///
    /// `rows` is the layout's packed row capacity, which bounds the keys of one call; `rotary` is
    /// the stack's, whose spectrum states the softmax scale.
    pub fn load(
        index: u32,
        config: &TransformerConfig,
        rows: u32,
        rotary: &Rotary,
        partition: &Partition,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        let head = u64::from(config.head_dim);
        let mut scope = loader.scope(format!("transformer_blocks.{index}.attn"));
        let query_gain = scope.scope("norm_q").tensor("weight", &[head])?;
        let key_gain = scope.scope("norm_k").tensor("weight", &[head])?;
        // The rotation and the attention read *this rank's* heads of the whole sequence.
        let local_heads = partition.denoiser_heads(config.num_attention_heads);
        let attention = DenseAttention::new(DenseSpec {
            head_dim: config.head_dim,
            heads: local_heads,
            kv_heads: local_heads,
            max_keys: rows,
            causal: false,
            window: None,
            scale: rotary.spectrum().score_scale(config.head_dim)?,
        })?;
        let ulysses = UlyssesAttention::load(
            &mut scope,
            UlyssesSpec {
                hidden: config.hidden_size,
                heads: config.num_attention_heads,
                head_dim: config.head_dim,
                inputs: &["to_q", "to_k", "to_v"],
                output: "to_out.0",
            },
            sequence_split(partition)?,
            partition.projection,
            |_| Ok(attention),
        )?;
        Ok(Self {
            ulysses,
            query_gain,
            key_gain,
            head_dim: config.head_dim,
        })
    }

    /// Attend `normalized` `[rows, hidden]` of this rank and write `[rows, hidden]`.
    pub fn forward(
        &self,
        f: &Forward,
        normalized: TensorView,
        factors: Factors<'_>,
        visible: &Varlen,
        scratch: &Scratch,
        out: TensorView,
    ) -> Result<()> {
        let ulysses = &self.ulysses;
        let rows = ulysses.rows(f, normalized, scratch.gathered_rows)?;
        for (index, projected) in [scratch.query, scratch.key, scratch.value]
            .into_iter()
            .enumerate()
        {
            ulysses.project(f, index, rows, scratch.staging, projected)?;
        }
        // This rank's heads of every row: the normalization and the rotation are per head, and the
        // factor tables cover the whole sequence.
        let local_heads = ulysses.local_heads();
        let headed = |view: TensorView| view.reshape(heads(view, local_heads, self.head_dim));
        factors.rotary.apply(
            f,
            headed(scratch.query),
            headed(scratch.key),
            Some(Gains {
                query: self.query_gain.view(),
                key: self.key_gain.view(),
            }),
            factors.cosine,
            factors.sine,
            headed(scratch.rotated_query),
            headed(scratch.rotated_key),
        )?;
        // The operator's operands are one axis of tokens, then the heads, then a head's channels,
        // while a projection writes `[rows, inner]`. The reshaping is by `heads`, which takes the
        // token axis from the view's own element count - the rows differ between the exchanged and
        // the local path.
        ulysses.inner().forward(
            f,
            headed(scratch.rotated_query),
            headed(scratch.rotated_key),
            headed(scratch.value),
            Some(*visible),
            headed(scratch.attended),
        )?;
        ulysses
            .return_rows(f, scratch.attended, scratch.staging, scratch.merged, out)
            .map_err(Error::from)
    }
}

/// One layer's projection scratch, reserved from the forward's arena.
pub struct Scratch {
    pub query: TensorView,
    pub key: TensorView,
    pub value: TensorView,
    pub rotated_query: TensorView,
    pub rotated_key: TensorView,
    pub attended: TensorView,
    /// The exchange's staging buffer, a member's block after another, for this rank's rows.
    pub staging: TensorView,
    /// The attended rows of every head, as the output projection reads them.
    pub merged: TensorView,
    /// The whole sequence's normalized rows, in the gather mode.
    pub gathered_rows: TensorView,
}

impl Scratch {
    /// Reserve one layer's scratch for `normalized` and a key set of `total_rows` rows.
    ///
    /// The shapes do not depend on the layer, and the worker's arena is a bump allocator reset
    /// once per forward, so the stack reserves this once rather than once per layer: 50 layers of
    /// ten change-shaped buffers is what a served canvas cannot hold.
    pub fn reserve(
        layer: &Dense,
        f: &Forward,
        normalized: TensorView,
        total_rows: u32,
    ) -> Result<Self> {
        let ulysses = &layer.ulysses;
        let dtype = normalized.dtype;
        let rows = normalized.shape.dim(0);
        let head = u64::from(layer.head_dim);
        let inner = u64::from(ulysses.heads()) * head;
        let local_inner = u64::from(ulysses.local_heads()) * head;
        let total = u64::from(total_rows);
        let arena = f.arena();
        let heads = |arena: &nn::Arena| arena.tensor(dtype, Shape::new(&[total, local_inner]));
        let query = heads(arena)?;
        let key = heads(arena)?;
        let value = heads(arena)?;
        let attended = heads(arena)?;
        // A layer that does not communicate reserves nothing for it: on one rank these views are
        // buffers the layer already holds, and the branches that read them do not run.
        let (staging, merged) = if ulysses.exchanges() {
            (
                arena.tensor(dtype, Shape::new(&[rows, inner]))?,
                arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            )
        } else {
            (query, attended)
        };
        let gathered_rows = if ulysses.exchanges() && ulysses.projection() == Projection::Gather {
            arena.tensor(dtype, Shape::new(&[total, normalized.shape.last()]))?
        } else {
            normalized
        };
        Ok(Self {
            query,
            key,
            value,
            rotated_query: heads(arena)?,
            rotated_key: heads(arena)?,
            attended,
            staging,
            merged,
            gathered_rows,
        })
    }
}

/// The three-dimensional view of a projected `[rows, heads · head_dim]` block.
/// A `[tokens, inner]` view as `[tokens, heads, head_dim]`.
///
/// The token axis is the view's own element count divided by one row's width: `reshape` takes a
/// shape literally, so a `0` written there is a zero-row tensor rather than an inferred axis. It was
/// written that way, and the first launch to reach the rotation read it as `0 tokens` - three gemms
/// had already written the rows the rope then found empty.
fn heads(view: TensorView, count: u32, head_dim: u32) -> Shape {
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
