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
//! A rank holds a contiguous slice of the packed rows and every head; [`UlyssesExchange`] trades
//! the row slice for a head shard of the whole sequence around the attention and back. With one
//! rank the exchange is absent.
//!
//! [`HeadProjection`] is how a rank reaches its heads, and the deployment states which way
//! (`Projection`): the exchange mode projects this rank's rows for every member's heads, one launch
//! per member, so each member's block is contiguous and one all-to-all delivers it; the gather mode
//! gathers every member's rows once and projects this rank's heads with a head-partitioned weight.
//! After either, the attention reads `[rows of the whole sequence, this rank's heads, head width]`,
//! and [`HeadsToRows`] sends each member its rows back with every head beside each other.
//!
//! ## The two attention kinds
//!
//! [`Layer::Dense`] attends every used row of the packed sequence: the queries are this rank's rows
//! and the keys are the request's used rows, both named by device boundary tables, so a padding row
//! is no key. [`Layer::Sparse`] is FastH3's video sparse attention over the segment packing
//! ([`crate::attention_sparse`]), composed from `nn::layers::VsaTiles` and
//! `nn::layers::SegmentAttention`.
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

use nn::family::Split;
use nn::layers::{
    Bias, DenseAttention, DenseSpec, Gains, Linear, Parallel, Rotary, UlyssesExchange, Varlen,
};
use nn::weights::{Weight, WeightLoader};
use nn::{Forward, Shape, TensorView};

use crate::config::TransformerConfig;
use crate::partition::Partition;
use crate::{Error, Result};
use nn::family::Projection;

/// A query, key, value or gate projection, as one rank's attention reads it: this rank's heads of
/// every row of the sequence.
pub(crate) struct HeadProjection {
    /// One rank: the whole projection. The gather mode: this rank's head rows of it. The exchange
    /// mode: one projection per member, in member order, each holding that member's head rows.
    members: Vec<Linear>,
}

impl HeadProjection {
    /// Reads the checkpoint's `[inner, hidden]` projection module `name` for `partition`.
    ///
    /// Each member's projection is its block of the module's heads: a column-parallel split of the
    /// output channels over the sequence group's members.
    pub(crate) fn load(
        loader: &mut WeightLoader<'_>,
        name: &str,
        inner: u64,
        hidden: u64,
        partition: &Partition,
    ) -> Result<Self> {
        let (inner, hidden) = (width(inner)?, width(hidden)?);
        let degree = u32::try_from(partition.ulysses_degree)
            .map_err(|_| Error::config("a sequence degree wider than a u32".to_owned()))?;
        let mut scope = loader.scope(name);
        let mut member = |rank: u32| -> Result<Linear> {
            let parallel = if degree == 1 {
                Parallel::None
            } else {
                Parallel::Column(Split { size: degree, rank })
            };
            Ok(Linear::load(
                &mut scope,
                hidden,
                inner,
                Bias::Absent,
                parallel,
            )?)
        };
        let members = match (degree, partition.projection) {
            (1, _) => vec![member(0)?],
            (_, Projection::Gather) => {
                let rank = u32::try_from(partition.sequence_rank)
                    .map_err(|_| Error::config("a sequence rank wider than a u32".to_owned()))?;
                vec![member(rank)?]
            }
            (_, Projection::Exchange) => (0..degree).map(member).collect::<Result<Vec<_>>>()?,
        };
        Ok(Self { members })
    }

    /// The element type of the projection's weight, which is the stack's.
    pub(crate) fn dtype(&self) -> nn::DType {
        self.members[0].weight().dtype
    }

    /// Projects this rank's `rows` for every member into `staging`, one member's block after
    /// another, and starts delivering each member its heads of the whole sequence into `out`.
    pub(crate) fn start(
        &self,
        f: &Forward,
        exchange: &UlyssesExchange,
        rows: TensorView,
        staging: TensorView,
        out: TensorView,
    ) -> Result<nn::Transfer> {
        self.project(f, rows, staging, out.shape.last())?;
        Ok(exchange.start(f, staging, out)?)
    }

    /// Each member's heads of `rows`, `local` columns wide, as blocks of `staging`.
    fn project(
        &self,
        f: &Forward,
        rows: TensorView,
        staging: TensorView,
        local: u64,
    ) -> Result<()> {
        let shape = Shape::new(&[rows.rows(), local]);
        let block = rows.rows() * local;
        for (member, projection) in self.members.iter().enumerate() {
            let at = staging.offset_elements(member as u64 * block);
            projection.forward(f, rows, staging.at(at).reshape(shape))?;
        }
        Ok(())
    }

    /// Projects `rows` and writes this rank's heads of the whole sequence into `out`.
    ///
    /// `rows` is this rank's rows in the exchange mode and the gathered sequence otherwise, and
    /// `staging` is the exchange's send buffer, one member's block after another.
    pub(crate) fn forward(
        &self,
        f: &Forward,
        exchange: Option<&UlyssesExchange>,
        rows: TensorView,
        staging: TensorView,
        out: TensorView,
    ) -> Result<()> {
        match (self.members.as_slice(), exchange) {
            ([only], _) => Ok(only.forward(f, rows, out)?),
            (_, Some(exchange)) => {
                self.project(f, rows, staging, out.shape.last())?;
                Ok(exchange.exchange(f, staging, out)?)
            }
            (members, None) => Err(Error::Unsupported(format!(
                "a projection held as {} members has no exchange to deliver them",
                members.len()
            ))),
        }
    }
}

/// The rows a projection reads: this rank's, or the whole sequence's when the deployment gathers.
pub(crate) fn projected_rows(
    f: &Forward,
    exchange: Option<&UlyssesExchange>,
    projection: Projection,
    normalized: TensorView,
    gathered: TensorView,
) -> Result<TensorView> {
    match (exchange, projection) {
        (Some(exchange), Projection::Gather) => {
            exchange.gather(f, normalized, gathered)?;
            Ok(gathered)
        }
        _ => Ok(normalized),
    }
}

/// Sends each member its rows of this rank's attended heads and sets the heads side by side.
pub(crate) struct HeadsToRows;

impl HeadsToRows {
    /// `attended` is `[whole sequence, this rank's heads · width]`, whose block `m` is member `m`'s
    /// rows. `staging` receives this rank's rows of every member's heads, block by block; `out` is
    /// those rows with the members' heads side by side, `[this rank's rows, every head · width]`.
    pub(crate) fn forward(
        f: &Forward,
        exchange: &UlyssesExchange,
        attended: TensorView,
        staging: TensorView,
        out: TensorView,
    ) -> Result<()> {
        exchange.exchange(f, attended, staging)?;
        Self::merge(f, exchange.ranks(), staging, out)
    }

    /// Sets the blocks of `received`, one per member, side by side in `out`.
    pub(crate) fn merge(
        f: &Forward,
        members: u64,
        received: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let rows = out.rows();
        let local = out.shape.last() / members;
        let block = [members, rows, 1, local];
        let source = received.reshape(Shape::new(&block));
        let destination = TensorView::with_strides(
            out.ptr,
            out.dtype,
            Shape::new(&block),
            nn::Strides::new(&[local, members * local, local, 1]),
        );
        let strides = [local as i64, (members * local) as i64, local as i64, 1];
        Ok(
            nn::layers::Copy::new(out.dtype, nn::layers::Access::Vector)?.forward(
                f,
                source,
                destination,
                &strides,
            )?,
        )
    }
}

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

/// The dense attention of one diffusion layer.
pub struct Dense {
    query: HeadProjection,
    key: HeadProjection,
    value: HeadProjection,
    output: Linear,
    query_gain: Weight,
    key_gain: Weight,
    attention: DenseAttention,
    exchange: Option<UlyssesExchange>,
    projection: Projection,
    heads: u32,
    local_heads: u32,
    head_dim: u32,
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
        let prefix = format!("transformer_blocks.{index}.attn");
        let hidden = u64::from(config.hidden_size);
        let head = u64::from(config.head_dim);
        let inner = u64::from(config.num_attention_heads) * head;

        let mut attention = loader.scope(&prefix);
        let query_gain = attention.scope("norm_q").tensor("weight", &[head])?;
        let key_gain = attention.scope("norm_k").tensor("weight", &[head])?;
        let local_heads = partition.denoiser_heads(config.num_attention_heads);

        let head_projection = |loader: &mut WeightLoader<'_>, name: &str| {
            HeadProjection::load(
                loader,
                &format!("{prefix}.{name}"),
                inner,
                hidden,
                partition,
            )
        };
        Ok(Self {
            query: head_projection(loader, "to_q")?,
            key: head_projection(loader, "to_k")?,
            value: head_projection(loader, "to_v")?,
            output: Linear::load(
                &mut loader.scope(format!("{prefix}.to_out.0")),
                width(inner)?,
                width(hidden)?,
                Bias::Absent,
                Parallel::None,
            )?,
            query_gain,
            key_gain,
            attention: DenseAttention::new(DenseSpec {
                head_dim: config.head_dim,
                heads: local_heads,
                kv_heads: local_heads,
                max_keys: rows,
                causal: false,
                window: None,
                scale: rotary.spectrum().score_scale(config.head_dim)?,
            })?,
            exchange: exchange(partition),
            projection: partition.projection,
            heads: config.num_attention_heads,
            local_heads,
            head_dim: config.head_dim,
        })
    }

    /// Attend `normalized` `[rows, hidden]` of this rank and write `[rows, hidden]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        f: &Forward,
        normalized: TensorView,
        factors: Factors<'_>,
        visible: &Varlen,
        scratch: &Scratch,
        out: TensorView,
    ) -> Result<()> {
        let exchange = self.exchange.as_ref();
        let rows = projected_rows(
            f,
            exchange,
            self.projection,
            normalized,
            scratch.gathered_rows,
        )?;
        self.query
            .forward(f, exchange, rows, scratch.staging, scratch.query)?;
        self.key
            .forward(f, exchange, rows, scratch.staging, scratch.key)?;
        self.value
            .forward(f, exchange, rows, scratch.staging, scratch.value)?;
        // This rank's heads of every row: the normalization and the rotation are per head, and the
        // factor tables cover the whole sequence.
        self.rotate(f, factors, scratch)?;
        let (query, key, value) = (scratch.rotated_query, scratch.rotated_key, scratch.value);
        // The operator's operands are one axis of tokens, then the heads, then a head's channels
        // (`ops/src/attention/dense.rs` requires rank 3 of all four), while a projection writes
        // `[rows, inner]`. The reshaping is by `heads`, which takes the token axis from the view's
        // own element count - the rows differ between the exchanged and the local path, so a token
        // count computed once outside would be right on one of them and wrong on the other.
        let (local_heads, head_dim) = (self.local_heads, self.head_dim);
        self.attention.forward(
            f,
            query.reshape(heads(query, local_heads, head_dim)),
            key.reshape(heads(key, local_heads, head_dim)),
            value.reshape(heads(value, local_heads, head_dim)),
            Some(*visible),
            scratch
                .attended
                .reshape(heads(scratch.attended, local_heads, head_dim)),
        )?;
        let attended = match exchange {
            None => scratch.attended,
            Some(exchange) => {
                HeadsToRows::forward(
                    f,
                    exchange,
                    scratch.attended,
                    scratch.staging,
                    scratch.merged,
                )?;
                scratch.merged
            }
        };
        Ok(self.output.forward(f, attended, out)?)
    }

    /// Normalize and rotate every head of the query and the key.
    fn rotate(&self, f: &Forward, factors: Factors<'_>, scratch: &Scratch) -> Result<()> {
        Ok(factors.rotary.apply(
            f,
            scratch
                .query
                .reshape(heads(scratch.query, self.local_heads, self.head_dim)),
            scratch
                .key
                .reshape(heads(scratch.key, self.local_heads, self.head_dim)),
            Some(Gains {
                query: self.query_gain.view(),
                key: self.key_gain.view(),
            }),
            factors.cosine,
            factors.sine,
            scratch.rotated_query.reshape(heads(
                scratch.rotated_query,
                self.local_heads,
                self.head_dim,
            )),
            scratch.rotated_key.reshape(heads(
                scratch.rotated_key,
                self.local_heads,
                self.head_dim,
            )),
        )?)
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
        let dtype = normalized.dtype;
        let rows = normalized.shape.dim(0);
        let inner = u64::from(layer.heads) * u64::from(layer.head_dim);
        let local_inner = u64::from(layer.local_heads) * u64::from(layer.head_dim);
        let total = u64::from(total_rows);
        let arena = f.arena();
        let heads = |arena: &nn::Arena| arena.tensor(dtype, Shape::new(&[total, local_inner]));
        let query = heads(arena)?;
        let key = heads(arena)?;
        let value = heads(arena)?;
        let attended = heads(arena)?;
        // A layer that does not communicate reserves nothing for it: on one rank these views are
        // buffers the layer already holds, and the branches that read them do not run.
        let (staging, merged) = match layer.exchange {
            Some(_) => (
                arena.tensor(dtype, Shape::new(&[rows, inner]))?,
                arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            ),
            None => (query, attended),
        };
        let gathered_rows = match (&layer.exchange, layer.projection) {
            (Some(_), Projection::Gather) => {
                arena.tensor(dtype, Shape::new(&[total, normalized.shape.last()]))?
            }
            _ => normalized,
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

/// The exchange of a partition's sequence degree, absent when one rank holds the sequence.
fn exchange(partition: &Partition) -> Option<UlyssesExchange> {
    match partition.ulysses_degree {
        1 => None,
        degree => UlyssesExchange::new(degree as u64).ok(),
    }
}

/// A channel count as a projection's width.
pub(crate) fn width(channels: u64) -> Result<u32> {
    u32::try_from(channels)
        .map_err(|_| Error::config(format!("{channels} channels exceed a projection's width")))
}
