//! The configuration of DeepSeek-V4.1 that its residual streams and its n-gram memories read,
//! from the checkpoint's `config.json` (`text_config`).
//!
//! A checkpoint's `config.json` is read here and nowhere else, through [`nn::config::Fields`].
//! The residual is `hc_mult` streams of `hidden_size`, each combination matrix taking
//! `hc_sinkhorn_iters` Sinkhorn iterations under `hc_eps`; the mixes' statistic and every
//! sub-layer input's normalization take the model's `rms_norm_eps`
//! (`inference/model.py:950-976`). The n-gram memories of the layers `engram_layer_ids` hash
//! the 2- to `engram_max_ngram_size`-grams of the normalized vocabulary into `engram_n_heads`
//! heads each (`inference/engram.py`).

use nn::config::Fields;
use nn::modules::{MhcConfig, NgramConfig};
use nn::{Checkpoint, Result};

use crate::deepseek_v41::ID;

/// What the n-gram memories of the model state.
#[derive(Clone, PartialEq, Debug)]
pub struct EngramConfig {
    /// The layers that add an n-gram memory into their streams before their attention.
    pub layer_ids: Vec<u32>,
    /// The table rows of each of those layers, in their order.
    pub num_embeddings: Vec<u64>,
    /// The longest n-gram, `N`.
    pub max_ngram_size: u32,
    /// The number each head's prime modulus is searched above.
    pub vocab_size: u64,
    /// Heads of each n-gram length.
    pub heads: u32,
    /// Elements of one table row.
    pub head_dim: u32,
    /// The vocabulary id a context holds before a request's first token.
    pub pad_token_id: u32,
    /// The size of the normalized vocabulary every multiplier is drawn for.
    pub compressed_vocab_size: u32,
}

/// The fields of `text_config` the residual and the n-gram memories read.
#[derive(Clone, PartialEq, Debug)]
pub struct Config {
    /// The hidden width, `d`.
    pub hidden_size: u32,
    /// The vocabulary of the embedding.
    pub vocab_size: u32,
    /// The epsilon of every RMS normalization.
    pub rms_norm_eps: f32,
    /// The residual's streams, `hc_mult`.
    pub hc_mult: u32,
    /// Sinkhorn iterations of a combination matrix.
    pub hc_sinkhorn_iters: u32,
    /// The epsilon of a pre-mix and of every Sinkhorn step.
    pub hc_eps: f32,
    /// The n-gram memories, when the model has any.
    pub engram: Option<EngramConfig>,
}

impl Config {
    /// Reads what `checkpoint`'s `config.json` states.
    ///
    /// # Errors
    ///
    /// No `config.json`, or what [`Self::parse`] refuses.
    pub fn open(checkpoint: &Checkpoint) -> Result<Self> {
        let value = checkpoint.sidecar("config.json")?.ok_or_else(|| {
            ID.config(format!(
                "{} holds no config.json",
                checkpoint.root().display()
            ))
        })?;
        Self::parse(&Fields::of("config.json", &value)?)
    }

    /// Reads `text_config` of a parsed `config.json`.
    ///
    /// # Errors
    ///
    /// A field missing or of another type, or n-gram memories whose layers and tables disagree.
    pub fn parse(root: &Fields<'_>) -> Result<Self> {
        let text = root.object("text_config")?;
        let engram = match text.optional_indices("engram_layer_ids")? {
            Some(layer_ids) if !layer_ids.is_empty() => {
                let num_embeddings = text
                    .numbers("engram_num_embeddings")?
                    .into_iter()
                    .map(|rows| rows as u64)
                    .collect::<Vec<_>>();
                if num_embeddings.len() != layer_ids.len() {
                    return Err(ID.config(format!(
                        "{} n-gram layers state {} tables",
                        layer_ids.len(),
                        num_embeddings.len()
                    )));
                }
                Some(EngramConfig {
                    layer_ids,
                    num_embeddings,
                    max_ngram_size: text.count("engram_max_ngram_size")?,
                    vocab_size: text.number("engram_vocab_size")? as u64,
                    heads: text.count("engram_n_heads")?,
                    head_dim: text.count("engram_head_dim")?,
                    pad_token_id: text.index("engram_pad_token_id")?,
                    compressed_vocab_size: text.count("engram_compressed_vocab_size")?,
                })
            }
            _ => None,
        };

        Ok(Self {
            hidden_size: text.count("hidden_size")?,
            vocab_size: text.count("vocab_size")?,
            rms_norm_eps: text.positive("rms_norm_eps")? as f32,
            hc_mult: text.count("hc_mult")?,
            hc_sinkhorn_iters: text.count("hc_sinkhorn_iters")?,
            hc_eps: text.positive("hc_eps")? as f32,
            engram,
        })
    }

    /// The residual's configuration: every connection's mixes and its input's normalization.
    pub fn mhc(&self) -> MhcConfig {
        MhcConfig {
            streams: self.hc_mult,
            hidden: self.hidden_size,
            iterations: self.hc_sinkhorn_iters,
            eps: self.hc_eps,
            rms_eps: self.rms_norm_eps,
            norm_eps: self.rms_norm_eps,
        }
    }

    /// An n-gram memory's configuration, when the model has n-gram memories.
    pub fn ngram(&self) -> Option<NgramConfig> {
        self.engram.as_ref().map(|engram| NgramConfig {
            order: engram.max_ngram_size,
            heads: engram.heads,
            row: engram.head_dim,
            streams: self.hc_mult,
            hidden: self.hidden_size,
            eps: self.rms_norm_eps,
        })
    }
}
