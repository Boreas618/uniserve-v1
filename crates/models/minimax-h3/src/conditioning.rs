//! H3's dense text refiner: the prompt features the DiT conditions on at every step.
//!
//! The text encoder's Qwen hidden states are `text_dim` wide; the refiner projects them to the
//! transformer's width and refines each document with dense bidirectional attention and a
//! pre-normalized SwiGLU. It is evaluated inside each denoising step
//! ([`crate::denoiser::Denoiser`]), where the prompt's length is known; its two blocks over the
//! prompt's rows are a small fraction of a step's fifty over the packed sequence.
//!
//! ## The document
//!
//! The prompt is padded to the layout's text capacity, so the queries are every padded row and the
//! keys are only the rows the prompt occupies: a padding row never becomes a key. The two boundary
//! tables are `[2]` int32 cumulative counts the step uploads, and the padded row count is the fixed
//! capacity, so one captured call serves every prompt length.
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
use nn::{DType, Forward, Operand};

use crate::config::TransformerConfig;
use crate::{Error, Result};

/// One refined document block: a pre-norm block over dense attention with per-head q/k norms and a
/// value-first SwiGLU.
struct RefinerBlock {
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
        loader: &mut WeightLoader<'_>,
        config: &TransformerConfig,
        index: u32,
        (rows, dtype): (u32, DType),
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
                    gated: false,
                },
                names: QkvNames::Separate(["to_q", "to_k", "to_v"]),
                bias: QkvBias::Absent,
                output: "to_out.0",
                output_bias: Bias::Absent,
                qk_norm: QkNormNames::Separate {
                    query: "norm_q",
                    key: "norm_k",
                    epsilon: config.qk_norm_eps,
                    output: dtype,
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
                activation: dtype,
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
    text_dim: u32,
}

impl TokenRefiner {
    /// Read the checkpoint's token refiner in `dtype`, bounded to `rows` rows per request: the
    /// layout's text capacity, the attention's key extent and the width every call is shaped for.
    pub fn load(
        loader: &mut WeightLoader<'_>,
        config: &TransformerConfig,
        rows: u32,
        dtype: DType,
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
            .map(|index| RefinerBlock::load(loader, config, index, (rows, dtype)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input,
            blocks,
            norm: RmsNorm::load(
                &mut loader.scope("token_refiner.final_norm"),
                config.hidden_size,
                config.norm_eps,
                nn::layers::Gain::Plain,
                dtype,
            )?,
            text_dim: config.text_dim,
        })
    }

    /// The refined prompt: `features` `[rows, text_dim]` in, a fresh `[rows, hidden_size]` out.
    ///
    /// `document` names the one document: queries over every padded row, keys over the prompt's.
    pub fn forward(&self, f: &Forward, features: Operand, document: Varlen) -> Result<Operand> {
        if features.shape.rank() != 2
            || features.shape.dim(0) == 0
            || features.shape.last() != u64::from(self.text_dim)
        {
            return Err(Error::config(format!(
                "refiner features are [rows, {}], not {:?}",
                self.text_dim,
                features.shape.dims()
            )));
        }
        let stream = self.input.forward(f, features)?;
        let carried = f.empty(stream.dtype, stream.shape)?;
        let keys = Keys::Dense(Some(document));
        let mut pending = None;
        for refiner in &self.blocks {
            pending = Some(
                refiner
                    .block
                    .forward(f, stream, pending, carried, |normalized| {
                        refiner.attention.forward(f, normalized, None, keys)
                    })?,
            );
        }
        if let Some(last) = pending {
            last.join(f, stream)?;
        }
        Ok(self.norm.forward(f, stream, None)?)
    }
}

/// The attention scale of one head: `1 / sqrt(head_dim)`.
fn scaling(head_dim: u32) -> f32 {
    1.0 / (head_dim as f32).sqrt()
}
