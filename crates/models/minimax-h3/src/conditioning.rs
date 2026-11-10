//! H3's dense text refiner: the prompt features the DiT conditions on at every step.
//!
//! The text encoder's Qwen hidden states are `text_dim` wide; the refiner projects them to the
//! transformer's width and refines each document with dense bidirectional attention and a
//! pre-normalized SwiGLU. It runs once per request -
//! the worker keeps its output in the request's state and hands it to every denoising step - so
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
//! The refiner belongs to the denoiser component, but it is not sharded: it reads at most the
//! prompt's rows once per request, so every rank of a Ulysses group refines the whole prompt with
//! every head and holds the result, which is what every one of its own packed rows gathers from.
//! Sharding two blocks over a thousand rows would save nothing and put two exchanges on the first
//! step of every request.

use nn::blocks::{
    AttentionKind, GateUp, GatedMlpSpec, Keys, PreNormBlock, PreNormSpec, QkNormNames,
    QkvAttention, QkvAttentionSpec, ResidualForm,
};
use nn::family::Split;
use nn::layers::{Bias, Linear, Parallel, QkvBias, QkvNames, QkvSpec, RmsNorm, Rounding, Varlen};
use nn::weights::WeightLoader;
use nn::{Forward, Shape, TensorView};

use crate::config::TransformerConfig;
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
    /// The longest key run of any document: the last entry of `cu_seq_lens_k`, as a launch value.
    pub key_rows: u32,
}

/// One refined document block: a pre-norm block over dense attention with per-head q/k norms and a
/// value-first SwiGLU.
pub struct RefinerBlock {
    /// The norms and the stepwise SwiGLU (`Rounding::Stepwise`): the refiner rounds `silu(gate)`
    /// and the product one step at a time. Each update joins the stream in its own launch.
    block: PreNormBlock,
    /// Bidirectional attention over a document's text rows, each head's query and key normalized
    /// by its own RmsNorm and not rotated.
    attention: QkvAttention,
}

impl RefinerBlock {
    /// Read block `index` of the checkpoint's token refiner.
    fn load(
        index: u32,
        config: &TransformerConfig,
        rows: u32,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        let mut scope = loader.scope(format!("token_refiner.refiner_blocks.{index}"));
        let attention = QkvAttention::load(
            &mut scope.scope("attn"),
            QkvAttentionSpec {
                qkv: QkvSpec {
                    hidden: config.hidden_size,
                    heads: config.num_attention_heads,
                    kv_heads: config.num_attention_heads,
                    head_dim: config.head_dim,
                },
                names: QkvNames::Separate(["to_q", "to_k", "to_v"]),
                bias: QkvBias::Absent,
                output: "to_out.0",
                output_bias: Bias::Absent,
                qk_norm: QkNormNames::Separate {
                    query: "norm_q",
                    key: "norm_k",
                    epsilon: config.qk_norm_eps,
                },
                kind: AttentionKind::Dense {
                    max_keys: rows,
                    causal: false,
                    window: None,
                },
                scale: scaling(config.head_dim),
            },
            Split::WHOLE,
        )?;
        let block = PreNormBlock::load(
            &mut scope,
            PreNormSpec {
                hidden: config.hidden_size,
                eps: config.norm_eps,
                norm1: "norm1",
                norm2: "norm2",
                layer_scale: None,
                norm_dtype: None,
                residual: ResidualForm::Separate,
                // The checkpoint fuses the feed-forward's value half first and its gate half second.
                mlp: GatedMlpSpec {
                    hidden: config.hidden_size,
                    intermediate: config.intermediate_size,
                    activation: ops::activation::Activation::Silu,
                    rounding: Rounding::Stepwise,
                    gate_up: GateUp::ValueFirst("ff.net.0.proj"),
                    down: "ff.net.2",
                    bias: Bias::Absent,
                },
            },
            Split::WHOLE,
        )?;
        Ok(Self { block, attention })
    }
}

/// The refiner: the projector from the text encoder's width, its blocks, and the final norm.
pub struct TokenRefiner {
    input: Linear,
    blocks: Vec<RefinerBlock>,
    /// The final norm, whose variance floor is the blocks' (`final_norm_eps` equals `norm_eps` in
    /// every shipped configuration).
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
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        if config.num_refiner_layers == 0 {
            return Err(Error::config(
                "an H3 denoiser's token refiner has at least one layer".to_owned(),
            ));
        }
        let input = Linear::load(
            &mut loader.scope("context_embedder"),
            config.text_dim,
            config.hidden_size,
            Bias::Present,
            Parallel::None,
        )?;
        let blocks = (0..config.num_refiner_layers)
            .map(|index| RefinerBlock::load(index, config, rows, loader))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input,
            blocks,
            norm: RmsNorm::load(
                &mut loader.scope("token_refiner.final_norm"),
                config.hidden_size,
                config.norm_eps,
                nn::layers::Gain::Plain,
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
    pub fn forward(
        &self,
        f: &Forward,
        features: TensorView,
        visible: &Documents,
        out: TensorView,
    ) -> Result<()> {
        let rows = features.shape.dim(0);
        if rows == 0 {
            return Err(Error::checkpoint(
                "a refiner reads at least one row".to_owned(),
            ));
        }
        if features.shape.last() != u64::from(self.text_dim) {
            return Err(Error::checkpoint(format!(
                "refiner features are [rows, {}], not {:?}",
                self.text_dim,
                features.shape.dims()
            )));
        }
        // The whole refined document is the chain of blocks over a working buffer; the input
        // projection is the same dtype as the features it reads, and the blocks' intermediates are
        // the projections' own.
        let shape = Shape::new(&[rows, u64::from(self.hidden_size)]);
        let arena = f.arena();
        let working = arena.tensor(features.dtype, shape)?;
        let dtype = self.blocks[0].attention.query().weight().dtype;
        let normalized = arena.tensor(dtype, shape)?;
        let update = arena.tensor(dtype, shape)?;
        self.input.forward(f, features, working)?;
        // One document per request over packed rows: every query sees the document's whole key set.
        let keys = Keys::Dense(Some(Varlen {
            cu_q: visible.cu_seq_lens_q,
            cu_k: visible.cu_seq_lens_k,
            max_q: None,
            max_k: visible.key_rows,
        }));
        let mut pending = None;
        for refiner in &self.blocks {
            let attend =
                |normalized, update| refiner.attention.forward(f, normalized, None, keys, update);
            pending = Some(
                refiner
                    .block
                    .forward(f, working, pending, normalized, update, attend)?,
            );
        }
        if let Some(last) = pending {
            last.join(f, working)?;
        }
        Ok(self.norm.forward(f, working, None, out)?)
    }
}

/// The attention scale of one head: `1 / sqrt(head_dim)`.
fn scaling(head_dim: u32) -> f32 {
    1.0 / (head_dim as f32).sqrt()
}
