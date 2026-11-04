//! The diffusion block's attention: every head of this rank's rows projected, normalized and
//! partly rotated, then attended densely over the packed sequence.
//!
//! A layer holds one merged query/key/value projection, an output projection, the per-head
//! normalization gains and the rotary embedding of the head, and one attention. The trailing
//! channels of every head pass the rotation unchanged: H3 rotates 96 of each head's 128 channels
//! over the (time, height, width) coordinates, 16 frequencies per axis, interleaved
//! (`uniserve/models/minimax_h3/attention.py`, [`crate::packing`]'s coordinates).
//!
//! ## The partition
//!
//! A rank holds a contiguous slice of the packed rows and every head; [`UlyssesExchange`] trades
//! the row slice for a head shard of the whole sequence around the attention and back. With one
//! rank the exchange is absent.
//!
//! ## The two attention kinds
//!
//! [`Kind::Dense`] attends every used row of the packed sequence: the queries are this rank's rows
//! and the keys are the request's used rows, both named by device boundary tables, so a padding row
//! is no key. [`Kind::Sparse`] is FastH3's video sparse attention over the segment packing; it is
//! composed from `nn::layers::VsaTiles` and `nn::layers::SegmentAttention` and lands with them
//! (see `crate::transformer`'s note on the sparse preset).
//!
//! ## Precision
//!
//! The dense attention runs in the checkpoint's own representation - BF16 activations with FP32
//! accumulation inside the fused kernels - which is the FlashAttention row of the design's numerics
//! table; nothing here declares an FP32 attention and no layer casts to one. The sparse path's
//! **compression** branch is the other row: the design requires FP32 there (cuBLASLt SGEMM plus a
//! softmax program), and the reference computes its tile means, scores and weighting in FP32
//! throughout (`uniserve/nn/attention/vsa/segments.py:60-77`, `:429-468`), so that branch declares
//! FP32 when it lands - in its layer's dtype and in this doc - while the fine sparse attention over
//! the prepared rows stays in the checkpoint's dtype like the dense one.

use nn::layers::{Attention as Flash, Linear, Rotary, Rotation, Sections, UlyssesExchange};
use nn::weights::{Weight, WeightLoader, WeightRequest};
use nn::{Forward, Shape, TensorView};
use ops::gemm::WeightOrder;

use crate::config::TransformerConfig;
use crate::partition::Partition;
use crate::{Error, Result};

/// Which attention a checkpoint's block composes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Full bidirectional attention over the packed sequence: the base release and `fl2va`.
    Dense,
    /// Video sparse attention over the segment packing: FastH3's DMD students.
    Sparse,
}

/// One layer's attention, of whichever kind its checkpoint composes.
pub enum Layer {
    /// Full bidirectional attention over the packed sequence.
    Dense(Box<Dense>),
    /// FastH3's video sparse attention over the segment packing.
    Sparse(Box<crate::attention_sparse::Sparse>),
}

impl Layer {
    /// Every checkpoint tensor layer `index`'s attention reads, as complete names.
    ///
    /// The list is what [`Layer::load`] requests, in the order it requests it, and the caller
    /// intersects it with the checkpoint's index: a name here the index lacks is a weight the layer
    /// cannot load, and a weight no block names is one nothing reads. A name that is built from the
    /// layer's prefix is returned whole, because the lookup that consumes it is whole.
    pub fn keys(index: u32, kind: crate::config::AttentionKind) -> Vec<String> {
        match kind {
            crate::config::AttentionKind::Dense => Dense::keys(index),
            crate::config::AttentionKind::Sparse { .. } => {
                crate::attention_sparse::Sparse::keys(index)
            }
        }
    }

    /// Read layer `index`'s attention of a DiT partition, of the kind the deployment composes.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        kind: crate::config::AttentionKind,
        index: u32,
        config: &TransformerConfig,
        rows: u32,
        rotated_axes: &[u32],
        tile_bound: u32,
        partition: &Partition,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        match kind {
            crate::config::AttentionKind::Dense => Ok(Self::Dense(Box::new(Dense::load(
                index,
                config,
                rows,
                rotated_axes,
                partition,
                loader,
            )?))),
            crate::config::AttentionKind::Sparse { tile, .. } => Ok(Self::Sparse(Box::new(
                crate::attention_sparse::Sparse::load(
                    index,
                    config,
                    rows,
                    rotated_axes,
                    tile,
                    tile_bound,
                    partition,
                    loader,
                )?,
            ))),
        }
    }

    /// The rotary factor tables of `positions`: one table serves either kind.
    pub fn factors(
        &self,
        f: &Forward,
        positions: TensorView,
        cosine: TensorView,
        sine: TensorView,
    ) -> Result<()> {
        match self {
            Self::Dense(dense) => dense.factors(f, positions, cosine, sine),
            Self::Sparse(sparse) => sparse.factors(f, positions, cosine, sine),
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
            Self::Sparse(sparse) => {
                LayerScratch::Sparse(Box::new(crate::attention_sparse::Scratch::reserve(
                    sparse,
                    f,
                    normalized,
                    u32::try_from(normalized.shape.dim(0) / u64::from(sparse.tile_rows))
                        .map_err(|_| Error::config("more tiles than a u32".to_owned()))?,
                )?))
            }
        })
    }

    /// Attend `normalized` `[rows, hidden]` of this rank and write `[rows, hidden]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        f: &Forward,
        normalized: TensorView,
        factors: Factors,
        visible: Option<&Visible>,
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
                    Error::Unsupported(
                        "a sparse deployment's layers need the segment views".to_owned(),
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

/// The rows one attention call reads, as device boundary tables.
///
/// Both are int32 cumulative counts; a dense call names the whole packed range for its queries and
/// the request's used rows for its keys.
#[derive(Clone, Copy)]
pub struct Visible {
    /// `[tokens + 1]` int32: this rank's query rows, accumulated from zero.
    pub cu_seq_lens_q: TensorView,
    /// `[tokens + 1]` int32: the keys every query sees, accumulated from zero.
    pub cu_seq_lens_k: TensorView,
    /// The longest run of key rows any query attends: the last entry of `cu_seq_lens_k`.
    ///
    /// The attention descriptor needs it as a launch value, and it is a property of the layout's
    /// visibility rather than of the rows, so it travels beside the two tables.
    pub key_rows: u32,
}

/// The rotary factor tables of one forward, `[tokens, rotated / 2]` FP32 each.
#[derive(Clone, Copy)]
pub struct Factors {
    /// The cosine table of the packed rows.
    pub cosine: TensorView,
    /// The sine table of the packed rows.
    pub sine: TensorView,
}

/// The dense attention of one diffusion layer.
pub struct Dense {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    rotary: Rotary,
    query_gain: Weight,
    key_gain: Weight,
    attention: Flash,
    exchange: Option<UlyssesExchange>,
    heads: u32,
    local_heads: u32,
    head_dim: u32,
}

impl Dense {
    /// Every checkpoint tensor one layer's dense attention reads, as complete names.
    pub fn keys(index: u32) -> Vec<String> {
        let prefix = format!("transformer_blocks.{index}.attn");
        [
            "norm_q.weight",
            "norm_k.weight",
            "to_q.weight",
            "to_k.weight",
            "to_v.weight",
            "to_out.0.weight",
        ]
        .iter()
        .map(|suffix| format!("{prefix}.{suffix}"))
        .collect()
    }

    /// Read layer `index`'s dense attention of a DiT partition.
    ///
    /// `rows` is the layout's packed row capacity, which bounds the keys of one call; `rotated` is
    /// the rotated width of a head and `sections` its axes, in that order.
    pub fn load(
        index: u32,
        config: &TransformerConfig,
        rows: u32,
        rotated_axes: &[u32],
        partition: &Partition,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        let prefix = format!("transformer_blocks.{index}.attn");
        let hidden = u64::from(config.hidden_size);
        let head = u64::from(config.head_dim);
        let inner = u64::from(config.num_attention_heads) * head;
        let rotated: u32 = rotated_axes.iter().sum();
        if rotated == 0 || rotated > config.head_dim || !rotated.is_multiple_of(2) {
            return Err(Error::config(format!(
                "{rotated} rotated channels of a {} element head",
                config.head_dim
            )));
        }

        let query_gain = gain(loader, &format!("{prefix}.norm_q.weight"), head)?;
        let key_gain = gain(loader, &format!("{prefix}.norm_k.weight"), head)?;
        let frequencies = frequencies(loader, config, rotated)?;
        // The rotation reads *this rank's* heads: under Ulysses the sequence is split across ranks
        // and each holds `heads / degree` of them (`Partition::denoiser_heads`), so a descriptor that
        // named the global count would refuse its own query view at any degree above one.
        let local_heads = partition.denoiser_heads(config.num_attention_heads);
        let rotary = Rotary::new(
            frequencies,
            config.head_dim,
            local_heads,
            local_heads,
            config.qk_norm_eps,
            Rotation::Half,
            1.0,
        )?
        .partial(rotated)?
        .sectioned(Sections::new(rotated_axes)?.interleaved())?;

        Ok(Self {
            query: projection(loader, &format!("{prefix}.to_q.weight"), inner, hidden)?,
            key: projection(loader, &format!("{prefix}.to_k.weight"), inner, hidden)?,
            value: projection(loader, &format!("{prefix}.to_v.weight"), inner, hidden)?,
            output: projection(loader, &format!("{prefix}.to_out.0.weight"), hidden, inner)?,
            rotary,
            query_gain,
            key_gain,
            attention: Flash::dense(
                config.head_dim,
                local_heads,
                local_heads,
                rows,
                false,
                None,
                1.0 / (config.head_dim as f32).sqrt(),
            )?,
            exchange: exchange(partition),
            heads: config.num_attention_heads,
            local_heads,
            head_dim: config.head_dim,
        })
    }

    /// The rotary factor tables of `positions` `[3, rows]`, written into the two views.
    pub fn factors(
        &self,
        f: &Forward,
        positions: TensorView,
        cosine: TensorView,
        sine: TensorView,
    ) -> Result<()> {
        self.rotary
            .factors(f, positions, cosine, sine)
            .map_err(Error::from)
    }

    /// Attend `normalized` `[rows, hidden]` of this rank and write `[rows, hidden]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        f: &Forward,
        normalized: TensorView,
        factors: Factors,
        visible: &Visible,
        scratch: &Scratch,
        out: TensorView,
    ) -> Result<()> {
        self.query.forward(f, normalized, scratch.query)?;
        self.key.forward(f, normalized, scratch.key)?;
        self.value.forward(f, normalized, scratch.value)?;
        self.rotate(f, factors, scratch)?;

        let (query, key, value) = match &self.exchange {
            None => (scratch.rotated_query, scratch.rotated_key, scratch.value),
            Some(exchange) => {
                exchange.scatter(f, scratch.rotated_query, scratch.exchanged_query)?;
                exchange.scatter(f, scratch.rotated_key, scratch.exchanged_key)?;
                exchange.scatter(f, scratch.value, scratch.exchanged_value)?;
                (
                    scratch.exchanged_query,
                    scratch.exchanged_key,
                    scratch.exchanged_value,
                )
            }
        };
        // The operator's operands are one axis of tokens, then the heads, then a head's channels
        // (`ops/src/attention/dense.rs` requires rank 3 of all four), while a projection writes
        // `[rows, inner]`. The reshaping is by `heads`, which takes the token axis from the view's
        // own element count - the rows differ between the exchanged and the local path, so a token
        // count computed once outside would be right on one of them and wrong on the other.
        let (local_heads, head_dim) = (self.local_heads, self.head_dim);
        self.attention.attend_dense(
            f,
            query.reshape(heads(query, local_heads, head_dim)),
            key.reshape(heads(key, local_heads, head_dim)),
            value.reshape(heads(value, local_heads, head_dim)),
            Some(visible.cu_seq_lens_q),
            Some(visible.cu_seq_lens_k),
            // Every query attends the whole key set in one sequence: the longest key run is the
            // number of used key rows, which `Visible` states.
            visible.key_rows,
            scratch
                .attended
                .reshape(heads(scratch.attended, local_heads, head_dim)),
        )?;
        let attended = match &self.exchange {
            None => scratch.attended,
            Some(exchange) => {
                exchange.gather(f, scratch.attended, scratch.gathered)?;
                scratch.gathered
            }
        };
        Ok(self.output.forward(f, attended, out)?)
    }

    /// Normalize and rotate every head of the query and the key.
    fn rotate(&self, f: &Forward, factors: Factors, scratch: &Scratch) -> Result<()> {
        Ok(self.rotary.apply(
            f,
            scratch
                .query
                .reshape(heads(scratch.query, self.local_heads, self.head_dim)),
            scratch
                .key
                .reshape(heads(scratch.key, self.local_heads, self.head_dim)),
            Some(self.query_gain.view()),
            Some(self.key_gain.view()),
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
    pub exchanged_query: TensorView,
    pub exchanged_key: TensorView,
    pub exchanged_value: TensorView,
    pub attended: TensorView,
    pub gathered: TensorView,
}

impl Scratch {
    /// Reserve one layer's scratch for `normalized` and a key set of `total_rows` rows.
    ///
    /// The shapes do not depend on the layer, and the executor's arena is a bump allocator reset
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
        let query = arena.tensor(dtype, Shape::new(&[rows, inner]))?;
        let key = arena.tensor(dtype, Shape::new(&[rows, inner]))?;
        let value = arena.tensor(dtype, Shape::new(&[rows, inner]))?;
        let attended = arena.tensor(dtype, Shape::new(&[total, local_inner]))?;
        // A layer that does not communicate has no exchange destinations and no gathered result:
        // the two branches that read them run only with an exchange, so on one rank the views are
        // the buffers the layer already holds and nothing is reserved twice. Reserving them
        // regardless cost four `rows x inner` buffers - 2.2 GiB at a served canvas.
        let (exchanged_query, exchanged_key, exchanged_value, gathered) = match layer.exchange {
            Some(_) => (
                arena.tensor(dtype, Shape::new(&[total, local_inner]))?,
                arena.tensor(dtype, Shape::new(&[total, local_inner]))?,
                arena.tensor(dtype, Shape::new(&[total, local_inner]))?,
                arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            ),
            None => (query, key, value, attended),
        };
        Ok(Self {
            query,
            key,
            value,
            rotated_query: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            rotated_key: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            exchanged_query,
            exchanged_key,
            exchanged_value,
            attended,
            gathered,
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

/// The rotary frequencies of `rotated` channels: `theta^(-2i / rotated)`, one per pair.
///
/// The table is a constant of the checkpoint's rope parameters, published through the loader like
/// any other weight, so the operator reads it as a device weight.
fn frequencies(
    loader: &mut WeightLoader<'_>,
    config: &TransformerConfig,
    rotated: u32,
) -> Result<Weight> {
    let half = (rotated / 2) as usize;
    let mut bytes = Vec::with_capacity(half * 4);
    for index in 0..half {
        let exponent = 2.0 * index as f64 / f64::from(rotated);
        let value = (config.rope_theta as f64).powf(-exponent) as f32;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let name = format!("transformer.rotary.{rotated}.{}", config.rope_theta);
    Ok(loader.constant(&name, nn::DType::F32, Shape::new(&[half as u64]), &bytes)?)
}

/// The exchange of a partition's sequence degree, absent when one rank holds the sequence.
fn exchange(partition: &Partition) -> Option<UlyssesExchange> {
    match partition.ulysses_degree {
        1 => None,
        degree => UlyssesExchange::new(degree as u64).ok(),
    }
}

/// A projection of the checkpoint: `[out, in]`, without bias.
fn projection(
    loader: &mut WeightLoader<'_>,
    name: &str,
    rows: u64,
    columns: u64,
) -> Result<Linear> {
    let weight = loader.load(&WeightRequest::region(
        name,
        0,
        Shape::new(&[rows, columns]),
    ))?;
    Ok(Linear::new(weight, None, WeightOrder::NxK)?)
}

/// A normalization's learned gain.
fn gain(loader: &mut WeightLoader<'_>, name: &str, columns: u64) -> Result<Weight> {
    Ok(loader.load(&WeightRequest::region(name, 0, Shape::new(&[columns])))?)
}
