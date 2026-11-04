//! H3's dense text refiner: the prompt features the DiT conditions on at every step.
//!
//! The text encoder's Qwen hidden states are `text_dim` wide; the refiner projects them to the
//! transformer's width and refines each document with dense bidirectional attention and a
//! pre-normalized SwiGLU (`uniserve/models/minimax_h3/conditioning.py`). It runs once per request -
//! the executor keeps its output in the request's state and hands it to every denoising step - so
//! it is a plain block of the transformer component, not a schedule.
//!
//! ## Documents in one batch
//!
//! Every document is padded to the same row count, so the queries are the whole packed row range
//! and the keys are only the rows a document's text occupies: a padding row never becomes a key.
//! Both boundary tables are device data (`[documents + 1]` int32 cumulative counts) and the packed
//! row count is the fixed capacity, so one captured call serves every prompt length
//! ([`Documents`]).
//!
//! ## The partition
//!
//! The refiner belongs to the denoiser component, which a deployment shards over the packed
//! sequence: a rank holds a contiguous slice of the rows and every head, projects its slice with
//! the whole weight, and an all-to-all trades the row slice for a head shard of the whole sequence
//! around the attention (design 4.3). With one rank the exchange is absent and every view is the
//! rank's own.

use nn::layers::{Attention, Linear, PackedSwiGlu, RmsNorm, ScaledResidual, UlyssesExchange};
use nn::weights::{Weight, WeightLoader, WeightRequest};
use nn::{Forward, Shape, TensorView};
use ops::gemm::WeightOrder;

use crate::config::TransformerConfig;
use crate::partition::Partition;
use crate::{Error, Result};

/// The per-request device tables the refiner's attention reads.
///
/// Both are int32 cumulative counts over the packed rows, `[documents + 1]` long: the queries cover
/// every padded row and the keys stop at each document's text length.
pub struct Documents {
    /// The query boundaries: one packed row per row, so the counts are `0, rows, 2 · rows, ...`.
    pub cu_seq_lens_q: TensorView,
    /// The key boundaries: each document's text rows accumulated, so a padding row is no key.
    pub cu_seq_lens_k: TensorView,
}

/// One refined document block: pre-normalized attention and a value-first SwiGLU.
pub struct RefinerBlock {
    norm1: RmsNorm,
    norm2: RmsNorm,
    query_norm: RmsNorm,
    key_norm: RmsNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    gate_up: Linear,
    down: Linear,
    swiglu: PackedSwiGlu,
    attention: Attention,
    /// The identity add of the attention and feed-forward residuals.
    residual: ScaledResidual,
    /// `None` on one rank; the exchange otherwise.
    exchange: Option<UlyssesExchange>,
    hidden_size: u32,
    head_dim: u32,
    local_heads: u32,
    heads: u32,
    intermediate_size: u32,
    dtype: nn::DType,
}

impl RefinerBlock {
    /// Read block `index` of the checkpoint's token refiner.
    fn load(
        index: u32,
        config: &TransformerConfig,
        rows: u32,
        partition: &Partition,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        let prefix = format!("token_refiner.refiner_blocks.{index}");
        let hidden = u64::from(config.hidden_size);
        let inner = u64::from(config.num_attention_heads) * u64::from(config.head_dim);
        let ffn = u64::from(config.intermediate_size);

        let query = projection(loader, &format!("{prefix}.attn.to_q.weight"), inner, hidden)?;
        let query_dtype = query.view().dtype;
        let key = projection(loader, &format!("{prefix}.attn.to_k.weight"), inner, hidden)?;
        let value = projection(loader, &format!("{prefix}.attn.to_v.weight"), inner, hidden)?;
        let output = projection(
            loader,
            &format!("{prefix}.attn.to_out.0.weight"),
            hidden,
            inner,
        )?;
        // The checkpoint fuses the feed-forward's value half first and its gate half second.
        let gate_up = projection(
            loader,
            &format!("{prefix}.ff.net.0.proj.weight"),
            2 * ffn,
            hidden,
        )?;
        let down = projection(
            loader,
            &format!("{prefix}.ff.net.2.weight"),
            hidden,
            ffn,
        )?;

        let local_heads = partition.denoiser_heads(config.num_attention_heads);
        let attention = Attention::dense(
            config.head_dim,
            local_heads,
            local_heads,
            rows,
            false,
            None,
            scaling(config.head_dim),
        )?;

        Ok(Self {
            dtype: query_dtype,
            intermediate_size: config.intermediate_size,
            norm1: norm(loader, &format!("{prefix}.norm1.weight"), hidden, config.norm_eps)?,
            norm2: norm(loader, &format!("{prefix}.norm2.weight"), hidden, config.norm_eps)?,
            query_norm: norm(
                loader,
                &format!("{prefix}.attn.norm_q.weight"),
                u64::from(config.head_dim),
                config.qk_norm_eps,
            )?,
            key_norm: norm(
                loader,
                &format!("{prefix}.attn.norm_k.weight"),
                u64::from(config.head_dim),
                config.qk_norm_eps,
            )?,
            query: Linear::new(query, None, WeightOrder::NxK)?,
            key: Linear::new(key, None, WeightOrder::NxK)?,
            value: Linear::new(value, None, WeightOrder::NxK)?,
            output: Linear::new(output, None, WeightOrder::NxK)?,
            gate_up: Linear::new(gate_up, None, WeightOrder::NxK)?,
            down: Linear::new(down, None, WeightOrder::NxK)?,
            swiglu: PackedSwiGlu::new(None)?,
            attention,
            residual: identity(loader)?,
            exchange: exchange(partition),
            hidden_size: config.hidden_size,
            head_dim: config.head_dim,
            local_heads,
            heads: config.num_attention_heads,
        })
    }

    /// Refine one rank's row slice of `hidden`, in place.
    fn forward(
        &self,
        f: &Forward,
        hidden: TensorView,
        visible: &Documents,
        total_rows: u32,
    ) -> Result<()> {
        let local_rows = u32::try_from(hidden.shape.dim(0))
            .map_err(|_| Error::checkpoint("a refiner row count above u32::MAX".to_owned()))?;
        let rows = Rows::reserve(self, f, local_rows, total_rows)?;

        // Attention: normalize, project every head of this rank's rows, normalize each head's
        // query and key, trade the row slice for a head shard of the whole sequence, attend, and
        // trade back.
        self.norm1.forward(f, hidden, rows.normed)?;
        self.query.forward(f, rows.normed, rows.query)?;
        self.key.forward(f, rows.normed, rows.key)?;
        self.value.forward(f, rows.normed, rows.value)?;
        self.query_norm
            .forward(f, rows.query, rows.query_normed)?;
        self.key_norm.forward(f, rows.key, rows.key_normed)?;

        let (query, key, value) = self.exchange(f, &rows)?;
        self.attention.attend_dense(
            f,
            query,
            key,
            value,
            Some(visible.cu_seq_lens_q),
            Some(visible.cu_seq_lens_k),
            rows.attended,
        )?;
        let attended = self.gather(f, rows.attended, &rows)?;
        self.output.forward(f, attended, rows.update)?;
        self.residual.forward(f, hidden, rows.update)?;

        // Feed-forward: normalize, project the packed value and gate halves, gate, project back.
        self.norm2.forward(f, hidden, rows.normed)?;
        self.gate_up.forward(f, rows.normed, rows.packed)?;
        self.swiglu.forward(f, rows.packed, rows.ffn, None)?;
        self.down.forward(f, rows.ffn, rows.update)?;
        self.residual.forward(f, hidden, rows.update)?;
        Ok(())
    }

    /// The query, key and value the attention reads: `[rows, local_heads, head_dim]` over the whole
    /// sequence.
    fn exchange<'a>(
        &self,
        f: &Forward,
        rows: &'a Rows,
    ) -> Result<(TensorView, TensorView, TensorView)> {
        match &self.exchange {
            None => Ok((
                rows.query_normed,
                rows.key_normed,
                rows.value,
            )),
            Some(exchange) => {
                exchange.scatter(f, rows.query_normed, rows.exchanged_query)?;
                exchange.scatter(f, rows.key_normed, rows.exchanged_key)?;
                exchange.scatter(f, rows.value, rows.exchanged_value)?;
                Ok((
                    rows.exchanged_query,
                    rows.exchanged_key,
                    rows.exchanged_value,
                ))
            }
        }
    }

    /// Put the attended head shards back together into this rank's rows.
    fn gather<'a>(&self, f: &Forward, attended: TensorView, rows: &'a Rows) -> Result<TensorView> {
        match &self.exchange {
            None => Ok(attended),
            Some(exchange) => {
                exchange.gather(f, attended, rows.gathered)?;
                Ok(rows.gathered)
            }
        }
    }
}

/// The refiner: the projector from the text encoder's width, its blocks, and the final norm.
pub struct TokenRefiner {
    input: Linear,
    blocks: Vec<RefinerBlock>,
    norm: RmsNorm,
    hidden_size: u32,
    text_dim: u32,
    rows: u32,
}

impl TokenRefiner {
    /// Read the checkpoint's token refiner, bounded to `rows` packed rows per request.
    ///
    /// `rows` is the deployment layout's padded row count: the attention's key extent and the width
    /// every captured call is shaped for.
    pub fn load(
        config: &TransformerConfig,
        rows: u32,
        partition: &Partition,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        if config.num_refiner_layers == 0 {
            return Err(Error::config(
                "an H3 denoiser's token refiner has at least one layer".to_owned(),
            ));
        }
        let hidden = u64::from(config.hidden_size);
        let text = u64::from(config.text_dim);
        let weight = matrix(loader, "context_embedder.weight", hidden, text)?;
        let bias = vector(loader, "context_embedder.bias", hidden)?;
        let input = Linear::new(weight, Some(bias), WeightOrder::NxK)?;
        let blocks = (0..config.num_refiner_layers)
            .map(|index| RefinerBlock::load(index, config, rows, partition, loader))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input,
            blocks,
            norm: norm(
                loader,
                "token_refiner.final_norm.weight",
                hidden,
                config.norm_eps,
            )?,
            hidden_size: config.hidden_size,
            text_dim: config.text_dim,
            rows,
        })
    }

    /// How many packed rows the refiner is bounded to.
    pub const fn rows(&self) -> u32 {
        self.rows
    }

    /// Refine `features` `[rows, text_dim]` into `out` `[rows, hidden_size]`.
    ///
    /// `total_rows` is the whole packed row count of the batch, which differs from the rows this
    /// rank holds when the deployment shards the sequence.
    pub fn forward(
        &self,
        f: &Forward,
        features: TensorView,
        visible: &Documents,
        total_rows: u32,
        out: TensorView,
    ) -> Result<()> {
        let local_rows = u32::try_from(features.shape.dim(0))
            .map_err(|_| Error::checkpoint("a refiner row count above u32::MAX".to_owned()))?;
        if total_rows == 0 || local_rows == 0 || total_rows < local_rows {
            return Err(Error::checkpoint(format!(
                "{local_rows} of {total_rows} refiner rows is not a row slice"
            )));
        }
        if features.shape.last() != u64::from(self.text_dim) {
            return Err(Error::checkpoint(format!(
                "refiner features are [rows, {}], not {:?}",
                self.text_dim,
                features.shape.dims()
            )));
        }
        // The whole refined document is the chain of blocks over a working buffer; the input
        // projection is the same dtype as the features it reads.
        let working = f
            .arena()
            .tensor(features.dtype, Shape::new(&[u64::from(local_rows), u64::from(self.hidden_size)]))?;
        self.input.forward(f, features, working)?;
        for block in &self.blocks {
            block.forward(f, working, visible, total_rows)?;
        }
        Ok(self.norm.forward(f, working, out)?)
    }
}

/// The scratch one block's forward reads and writes, reserved from the forward's arena.
///
/// One block runs at a time and the arena is reused, so every buffer is written before it is read.
struct Rows {
    normed: TensorView,
    query: TensorView,
    key: TensorView,
    value: TensorView,
    query_normed: TensorView,
    key_normed: TensorView,
    exchanged_query: TensorView,
    exchanged_key: TensorView,
    exchanged_value: TensorView,
    attended: TensorView,
    gathered: TensorView,
    update: TensorView,
    packed: TensorView,
    ffn: TensorView,
}

impl Rows {
    fn reserve(block: &RefinerBlock, f: &Forward, local_rows: u32, total_rows: u32) -> Result<Self> {
        let dtype = block.dtype;
        let hidden = u64::from(block.hidden_size);
        let head = u64::from(block.head_dim);
        let inner = u64::from(block.heads) * head;
        let local_inner = u64::from(block.local_heads) * head;
        // The gate projection's rows hold the checkpoint's two fused halves of one width.
        let packed_width = 2 * u64::from(block.intermediate_size);
        let ffn = u64::from(block.intermediate_size);

        let rows = u64::from(local_rows);
        let total = u64::from(total_rows);
        let arena = f.arena();
        Ok(Self {
            normed: arena.tensor(dtype, Shape::new(&[rows, hidden]))?,
            query: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            key: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            value: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            query_normed: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            key_normed: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            exchanged_query: arena.tensor(dtype, Shape::new(&[total, local_inner]))?,
            exchanged_key: arena.tensor(dtype, Shape::new(&[total, local_inner]))?,
            exchanged_value: arena.tensor(dtype, Shape::new(&[total, local_inner]))?,
            attended: arena.tensor(dtype, Shape::new(&[total, local_inner]))?,
            gathered: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            update: arena.tensor(dtype, Shape::new(&[rows, hidden]))?,
            packed: arena.tensor(dtype, Shape::new(&[rows, packed_width]))?,
            ffn: arena.tensor(dtype, Shape::new(&[rows, ffn]))?,
        })
    }
}

/// The attention scale of one head: `1 / sqrt(head_dim)`.
fn scaling(head_dim: u32) -> f32 {
    1.0 / (head_dim as f32).sqrt()
}

/// The exchange of a partition's sequence degree, absent when one rank holds the sequence.
fn exchange(partition: &Partition) -> Option<UlyssesExchange> {
    match partition.ulysses_degree {
        1 => None,
        degree => UlyssesExchange::new(degree as u64).ok(),
    }
}

/// The identity residual: every channel's update added unchanged.
fn identity(loader: &mut WeightLoader<'_>) -> Result<ScaledResidual> {
    let one = 1.0f32.to_le_bytes();
    let scale = loader.constant("refiner.identity_scale", nn::DType::F32, Shape::new(&[1]), &one)?;
    Ok(ScaledResidual::new(scale, None)?)
}

/// A `[rows, columns]` weight of the checkpoint.
fn matrix(
    loader: &mut WeightLoader<'_>,
    name: &str,
    rows: u64,
    columns: u64,
) -> Result<Weight> {
    Ok(loader.load(&WeightRequest::region(
        name,
        0,
        Shape::new(&[rows, columns]),
    ))?)
}

/// A `[rows]` weight of the checkpoint.
fn vector(loader: &mut WeightLoader<'_>, name: &str, rows: u64) -> Result<Weight> {
    Ok(loader.load(&WeightRequest::region(name, 0, Shape::new(&[rows])))?)
}

/// A projection of the checkpoint: `[out, in]`.
fn projection(
    loader: &mut WeightLoader<'_>,
    name: &str,
    rows: u64,
    columns: u64,
) -> Result<Weight> {
    matrix(loader, name, rows, columns)
}

/// A normalization's learned gain.
fn norm(loader: &mut WeightLoader<'_>, name: &str, columns: u64, eps: f32) -> Result<RmsNorm> {
    Ok(RmsNorm::new(vector(loader, name, columns)?, eps)?)
}
