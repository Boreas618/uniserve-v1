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

use nn::layers::{Attention, Linear, Rotary, Rotation, Sections, UlyssesExchange};
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
    attention: Attention,
    exchange: Option<UlyssesExchange>,
    heads: u32,
    local_heads: u32,
    head_dim: u32,
}

impl Dense {
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
        let rotary = Rotary::new(
            frequencies,
            config.head_dim,
            config.num_attention_heads,
            config.num_attention_heads,
            config.qk_norm_eps,
            Rotation::Half,
            1.0,
        )?
        .partial(rotated)?
        .sectioned(Sections::new(rotated_axes)?.interleaved())?;

        let local_heads = partition.denoiser_heads(config.num_attention_heads);
        Ok(Self {
            query: projection(loader, &format!("{prefix}.to_q.weight"), inner, hidden)?,
            key: projection(loader, &format!("{prefix}.to_k.weight"), inner, hidden)?,
            value: projection(loader, &format!("{prefix}.to_v.weight"), inner, hidden)?,
            output: projection(loader, &format!("{prefix}.to_out.0.weight"), hidden, inner)?,
            rotary,
            query_gain,
            key_gain,
            attention: Attention::dense(
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
        Ok(self.rotary.factors(f, positions, cosine, sine)?)
    }

    /// Attend `normalized` `[rows, hidden]` of this rank and write `[rows, hidden]`.
    pub fn forward(
        &self,
        f: &Forward,
        normalized: TensorView,
        factors: Factors,
        visible: &Visible,
        total_rows: u32,
        out: TensorView,
    ) -> Result<()> {
        let scratch = Scratch::reserve(self, f, normalized, total_rows)?;
        self.query.forward(f, normalized, scratch.query)?;
        self.key.forward(f, normalized, scratch.key)?;
        self.value.forward(f, normalized, scratch.value)?;
        self.rotate(f, factors, &scratch)?;

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
        self.attention.attend_dense(
            f,
            query,
            key,
            value,
            Some(visible.cu_seq_lens_q),
            Some(visible.cu_seq_lens_k),
            // Every query attends the whole key set in one sequence: the longest key run is the
            // number of used key rows, which `Visible` states.
            visible.key_rows,
            scratch.attended,
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
                .reshape(heads(self.local_heads, self.head_dim)),
            scratch.key.reshape(heads(self.local_heads, self.head_dim)),
            Some(self.query_gain.view()),
            Some(self.key_gain.view()),
            factors.cosine,
            factors.sine,
            scratch
                .rotated_query
                .reshape(heads(self.local_heads, self.head_dim)),
            scratch
                .rotated_key
                .reshape(heads(self.local_heads, self.head_dim)),
        )?)
    }
}

/// One layer's projection scratch, reserved from the forward's arena.
struct Scratch {
    query: TensorView,
    key: TensorView,
    value: TensorView,
    rotated_query: TensorView,
    rotated_key: TensorView,
    exchanged_query: TensorView,
    exchanged_key: TensorView,
    exchanged_value: TensorView,
    attended: TensorView,
    gathered: TensorView,
}

impl Scratch {
    fn reserve(
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
        Ok(Self {
            query: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            key: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            value: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            rotated_query: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            rotated_key: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            exchanged_query: arena.tensor(dtype, Shape::new(&[total, local_inner]))?,
            exchanged_key: arena.tensor(dtype, Shape::new(&[total, local_inner]))?,
            exchanged_value: arena.tensor(dtype, Shape::new(&[total, local_inner]))?,
            attended: arena.tensor(dtype, Shape::new(&[total, local_inner]))?,
            gathered: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
        })
    }
}

/// The three-dimensional view of a projected `[rows, heads · head_dim]` block.
fn heads(count: u32, head_dim: u32) -> Shape {
    Shape::new(&[0, u64::from(count), u64::from(head_dim)])
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
