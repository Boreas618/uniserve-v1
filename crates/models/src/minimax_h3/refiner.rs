//! The token refiner: the prompt features the transformer conditions on.
//!
//! The text encoder's hidden states are `text_dim` wide; the refiner projects them to the
//! transformer's width (`context_embedder`) and refines them with pre-norm blocks of dense
//! bidirectional attention and a SwiGLU, without modulation or rotation, then a final norm
//! (`refs/vllm-omni/…/minimax_h3/minimax_h3_transformer.py:442-522`):
//!
//! ```text
//! x = context_embedder(features)
//! x = x + attn(norm1(x));  x = x + mlp(norm2(x))        per block
//! refined = final_norm(x)
//! ```
//!
//! ## The prompt's rows
//!
//! The prompt is padded to the layout's text capacity, so the queries are every row of the
//! capacity and the keys are the rows the prompt occupies: a padding row is never a key, and
//! the rows it computes are never gathered. The two offset tables are host values of the call,
//! so one captured call serves every prompt length.
//!
//! ## Sequence parallelism
//!
//! The refiner is not split: it reads at most the prompt's rows once per request, so every
//! rank of a sequence split refines the whole prompt with every head, as the reference's
//! refiner skips the exchange (`minimax_h3_transformer.py:455-457`).

use nn::family::SequenceSplit;
use nn::modules::{
    Attention, AttentionConfig, Linear, LinearConfig, Mlp, Parallel, QkNorm, QkvBias,
    QkvLinearConfig, QkvNames, RmsNorm, RmsNormConfig,
};
use nn::{DType, Device, Split, Tensor, VarBuilder};

use crate::minimax_h3::config::TransformerConfig;
use crate::minimax_h3::transformer::mlp_config;
use crate::minimax_h3::{Error, Result};

/// One refiner block: a pre-norm block over dense attention with per-head query and key norms
/// and a value-first SwiGLU.
struct RefinerBlock {
    norm1: RmsNorm,
    norm2: RmsNorm,
    /// Bidirectional attention over the prompt's rows, each head's query and key normalized
    /// by its own RMS norm and not rotated.
    attention: Attention,
    mlp: Mlp,
}

impl RefinerBlock {
    /// The block the module at `vb` states (`token_refiner.refiner_blocks.<i>`), whose
    /// attention reads at most `rows` keys.
    fn load(config: &TransformerConfig, rows: u32, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let norm = RmsNormConfig {
            width: config.hidden_size,
            eps: config.norm_eps,
            centred: false,
            output: None,
        };
        let attention = AttentionConfig {
            qkv: QkvLinearConfig {
                hidden: config.hidden_size,
                heads: config.num_attention_heads,
                kv_heads: config.num_attention_heads,
                head_dim: config.head_dim,
                gated: false,
                names: QkvNames::Separate(["to_q", "to_k", "to_v"]),
                bias: QkvBias::Absent,
                split: Split::WHOLE,
            },
            output: "to_out.0",
            output_bias: false,
            qk_norm: QkNorm::Separate {
                modules: ["norm_q", "norm_k"],
                eps: config.qk_norm_eps,
            },
            scale: 1.0 / (config.head_dim as f32).sqrt(),
            causal: false,
            window: None,
            max_kv_len: rows,
            sequence: SequenceSplit::WHOLE,
        };

        Ok(Self {
            norm1: RmsNorm::load(&norm, vb.pp("norm1"))?,
            norm2: RmsNorm::load(&norm, vb.pp("norm2"))?,
            attention: Attention::load(&attention, vb.pp("attn"))?,
            mlp: Mlp::load(&mlp_config(config), vb.clone())?,
        })
    }

    /// The block over `x`, `[rows, hidden]`, whose keys are the rows `keys` names.
    fn forward(&self, x: &Tensor, queries: &Tensor, keys: &Tensor, rows: u32) -> Result<Tensor> {
        let attended = self.attention.forward_varlen(
            &self.norm1.forward(x)?,
            None,
            queries,
            keys,
            None,
            rows,
        )?;
        let x = (x + attended)?;

        Ok((&x + self.mlp.forward(&self.norm2.forward(&x)?)?)?)
    }
}

/// The refiner: the projection from the text encoder's width, its blocks and the final norm.
pub struct TokenRefiner {
    input: Linear,
    blocks: Vec<RefinerBlock>,
    /// The final norm, whose variance floor is the blocks'.
    norm: RmsNorm,
    text_dim: usize,
    /// The text capacity the refiner was loaded for: the rows of every call.
    rows: u32,
}

impl TokenRefiner {
    /// The refiner the checkpoint at `vb` states, for prompts padded to `rows` rows: the
    /// layout's text capacity and the key bound of its attention.
    ///
    /// # Errors
    ///
    /// A checkpoint that states no refiner block, or a weight the loader refuses.
    pub fn load(config: &TransformerConfig, rows: u32, vb: VarBuilder<'_, '_>) -> Result<Self> {
        if config.num_refiner_layers == 0 {
            return Err(Error::config(
                "an H3 denoiser's token refiner has at least one block".to_owned(),
            ));
        }
        let refiner = vb.pp("token_refiner");

        Ok(Self {
            input: Linear::load(
                &LinearConfig {
                    inputs: config.text_dim,
                    outputs: config.hidden_size,
                    bias: true,
                    parallel: Parallel::None,
                },
                vb.pp("context_embedder"),
            )?,
            blocks: (0..config.num_refiner_layers)
                .map(|index| {
                    RefinerBlock::load(config, rows, refiner.pp("refiner_blocks").pp(index))
                })
                .collect::<Result<Vec<_>>>()?,
            norm: RmsNorm::load(
                &RmsNormConfig {
                    width: config.hidden_size,
                    eps: config.norm_eps,
                    centred: false,
                    output: None,
                },
                refiner.pp("final_norm"),
            )?,
            text_dim: config.text_dim as usize,
            rows,
        })
    }

    /// The element type the refiner computes in: its projection's.
    pub fn dtype(&self) -> DType {
        self.input.weight().dtype()
    }

    /// The refined prompt: `features`, the `[rows, text_dim]` rows of the text capacity of
    /// which the first `prompt` are the prompt's, into `[rows, hidden]`.
    ///
    /// # Errors
    ///
    /// Features of another shape than the capacity's, a prompt longer than it, or what a
    /// module refuses.
    pub fn forward(&self, features: &Tensor, prompt: u32) -> Result<Tensor> {
        let rows = self.rows;
        if features.dims() != [rows as usize, self.text_dim] || prompt > rows {
            return Err(Error::request(format!(
                "the refiner reads [{rows}, {}] features of a prompt of at most {rows} rows, not \
                 {:?} of a prompt of {prompt}",
                self.text_dim,
                features.dims()
            )));
        }

        // One sequence: every row of the capacity is a query, the prompt's rows are the keys.
        let offsets = |end: u32| -> Result<Tensor> {
            Ok(Tensor::new(&[0, end], &Device::Cpu)?.to_device(features.device())?)
        };
        let (queries, keys) = (offsets(rows)?, offsets(prompt)?);

        let mut x = self.input.forward(features)?;
        for block in &self.blocks {
            x = block.forward(&x, &queries, &keys, rows)?;
        }
        Ok(self.norm.forward(&x)?)
    }
}
