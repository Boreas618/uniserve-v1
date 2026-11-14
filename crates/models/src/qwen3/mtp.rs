//! The Qwen3.5 drafter: the checkpoint's multi-token-prediction module (`mtp.*`), Qwen3-Next's
//! `Qwen3NextMultiTokenPredictor` (vLLM's `Qwen3_5MultiTokenPredictor`,
//! `refs/vllm/vllm/model_executor/models/qwen3_5_mtp.py:148-193`).
//!
//! An entry at position `i` reads the model's final hidden state `h` there and the embedding
//! `e` of the token at `i + 1`:
//!
//! ```text
//! x      = fc([pre_fc_norm_embedding(e) | pre_fc_norm_hidden(h)])    [2·hidden] → [hidden]
//! x, r   = layer(x)       a full-attention decoder layer over the draft cache, routed
//! h'     = norm(x + r)                                  the drafter's hidden state at `i`
//! ```
//!
//! The layer is the model's own decoder layer, rotated by the model's text spectrum at the
//! entry's position. The embedding and the head are the model's ([`nn::Drafter`]); the
//! drafter's experts keep the checkpoint's order, because an expert redistribution names only
//! the model's layers.
//!
//! The references run one multi-token-prediction layer per forward: SGLang builds one
//! (`refs/sglang/python/sglang/srt/models/qwen3_5_mtp.py:139-140`), and vLLM runs layer
//! `step mod layers` of a speculation's steps (`qwen3_5_mtp.py:172-173`). A forward here is
//! not told its step, so a checkpoint of more than one layer is refused at load.

use nn::family::{CacheShape, SequenceSplit};
use nn::modules::{Linear, LinearConfig, Parallel, RmsNorm, Rotary};
use nn::{D, Drafter, KvCache, Split, Tensor, TokenBatch, VarBuilder, functional};

use crate::qwen3::causal_lm::text_rotary;
use crate::qwen3::config::Config;
use crate::qwen3::decoder::{self, DecoderLayer};
use crate::qwen3::{Error, Result};

/// The profiler class of the drafter's whole forward, at path `mtp`.
const DRAFTER: &str = "models.qwen3.transformer.Drafter";

/// The multi-token-prediction module of a Qwen3.5 checkpoint.
pub struct Mtp {
    pre_fc_norm_embedding: RmsNorm,
    pre_fc_norm_hidden: RmsNorm,
    fc: Linear,
    rotary_emb: Rotary,
    layer: DecoderLayer,
    norm: RmsNorm,
    /// The width of the rows it reads and writes.
    hidden: usize,
    cache: CacheShape,
    /// The module's path in the checkpoint, which names its profiler range.
    path: String,
}

impl Mtp {
    /// The drafter of `config` at `vb` (`mtp`), at the ranks `splits` state of the tensor and
    /// the expert group.
    ///
    /// # Errors
    ///
    /// A checkpoint that carries no drafter or one of several layers (module documentation),
    /// or a module the loader refuses.
    pub fn load(config: &Config, vb: VarBuilder<'_, '_>, splits: (Split, Split)) -> Result<Self> {
        match config.mtp_num_hidden_layers {
            1 => {}
            0 => {
                return Err(Error::Config(
                    "the deployment speculates and the checkpoint carries no \
                     multi-token-prediction layer"
                        .to_owned(),
                ));
            }
            layers => {
                return Err(Error::Config(format!(
                    "a drafter of {layers} layers runs one layer per speculative step, which \
                     a draft forward is not told; one layer is served"
                )));
            }
        }

        let text = &config.text;
        let norm = decoder::stream_norm(text);

        Ok(Self {
            pre_fc_norm_embedding: RmsNorm::load(&norm, vb.pp("pre_fc_norm_embedding"))?,
            pre_fc_norm_hidden: RmsNorm::load(&norm, vb.pp("pre_fc_norm_hidden"))?,
            fc: Linear::load(
                &LinearConfig {
                    inputs: 2 * text.hidden_size,
                    outputs: text.hidden_size,
                    bias: false,
                    parallel: Parallel::None,
                },
                vb.pp("fc"),
            )?,
            rotary_emb: Rotary::load(&text_rotary(text.rope), vb.pp("rotary_emb"))?,
            // The drafter's layer attends, whatever mixers the model's layers hold, and
            // routes when the model does (`qwen3_5_mtp.py:120-127`).
            layer: DecoderLayer::load(
                config,
                vb.pp("layers").pp(0),
                false,
                config.moe.is_some(),
                splits,
                SequenceSplit::WHOLE,
                None,
            )?,
            norm: RmsNorm::load(&norm, vb.pp("norm"))?,
            hidden: text.hidden_size as usize,
            cache: CacheShape {
                layers: 1,
                kv_heads: text.local_heads(splits.0)?.1,
                head_dim: text.head_dim,
                index_keys: None,
            },
            path: vb.path().to_owned(),
        })
    }
}

impl Drafter for Mtp {
    fn cache(&self) -> CacheShape {
        self.cache
    }

    fn forward(
        &self,
        batch: &TokenBatch,
        cache: &KvCache,
        embedded: &Tensor,
        hidden: &Tensor,
    ) -> nn::Result<Tensor> {
        let tokens = batch.tokens();
        for (name, rows) in [("embedded", embedded), ("hidden", hidden)] {
            if rows.dims() != [tokens, self.hidden] {
                return Err(Error::Config(format!(
                    "the drafter's {name} is {:?}, [{tokens}, {}] expected",
                    rows.dims(),
                    self.hidden
                ))
                .into());
            }
        }

        let forward = || -> Result<Tensor> {
            // The two normalized inputs side by side: the embedding's in the first half of
            // `fc`'s input and the hidden state's in the second.
            let joined = Tensor::cat(
                &[
                    self.pre_fc_norm_embedding.forward(embedded)?,
                    self.pre_fc_norm_hidden.forward(hidden)?,
                ],
                D::Minus1,
            )?;
            let stream = self.fc.forward(&joined)?;

            let (cos, sin) = self.rotary_emb.forward(&batch.positions)?;

            // The decode kernel's arrival counters are cleared once per forward, as the
            // model's own forward clears them.
            functional::zero_arrival_counters_(&cache.workspace)?;

            let (output, residual) = self.layer.forward(
                &stream,
                None,
                (&cos, &sin),
                batch,
                Some(cache),
                None,
                None,
                0,
            )?;
            let (normalized, _) = self.norm.forward_residual(&output, &residual)?;
            Ok(normalized)
        };

        Ok(nn::profiler::range(DRAFTER, &self.path, forward)?)
    }
}

impl std::fmt::Debug for Mtp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Mtp")
            .field("hidden", &self.hidden)
            .field("cache", &self.cache)
            .finish_non_exhaustive()
    }
}
