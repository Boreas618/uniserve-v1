//! The drafter: the checkpoint's multi-token-prediction module (`mtp.*`; vLLM's
//! `Qwen4ExpMultiTokenPredictor`, `refs/vllm/vllm/models/qwen4_exp/nvidia/mtp.py:181-401`).
//!
//! An entry at position `i` reads the streams `S` of position `i` (the model's, or the
//! drafter's own when it drafts further ahead) and the embedding `e` of the token at `i + 1`:
//!
//! ```text
//! e'   = fc_embedding(gemma(e; pre_fc_norm_embedding))                      [tokens, H]
//! S0_c = fc_hidden(gemma(S; pre_fc_norm_hidden)_c) + e'                     one RMS over C · H
//! S'   = layer(S0)         an attention layer with its own cache, index keys and ring
//! draft logits = lm_head(mtp.hyper_connection_mixer.mix(norm(S')))
//! ```
//!
//! `pre_fc_norm_hidden` normalizes the `C · H` values of a row together, not stream by stream
//! (`mtp.py:259-261`, `:334-338`). The embedding and the head are the model's
//! ([`nn::Drafter`]): the forward returns the streams `S'`, which the next entry reads, and
//! [`Drafter::logits_rows`] mixes the rows the model's head projects.
//!
//! The drafter's indexer runs at every step, as vLLM's does when the checkpoint states no
//! `index_share_for_mtp_iteration` (`refs/vllm/vllm/config/speculative.py:1079-1088`). Its
//! experts keep the checkpoint's order, because an expert redistribution names only the
//! model's layers; an export's block-scaled FP8 experts are decoded to dense matrices at load.
//!
//! The references run one multi-token-prediction layer per forward, and a forward here is not
//! told its step, so a checkpoint of more than one layer is refused at load.

use nn::family::{CacheShape, StateShape};
use nn::modules::{Linear, LinearConfig, Parallel, RmsNorm, RmsNormConfig, Rotary};
use nn::{DType, Drafter, KvCache, RecurrentState, Split, Tensor, TokenBatch, VarBuilder};

use crate::attention::IndexedAttention;
use crate::causal_lm::{cache_shape, coordinates, token_columns};
use crate::config::Config;
use crate::decoder::DecoderLayer;
use crate::hyper_connection::HyperConnection;
use crate::{Error, Result};

/// The profiler class of the drafter's whole forward, at path `mtp`.
const DRAFTER: &str = "models.qwen4_exp.transformer.Drafter";

/// The multi-token-prediction module of a checkpoint.
pub struct Mtp {
    pre_fc_norm_embedding: RmsNorm,
    pre_fc_norm_hidden: RmsNorm,
    fc_embedding: Linear,
    fc_hidden: Linear,
    rotary_emb: Rotary,
    layer: DecoderLayer,
    /// The read-out of the drafter's streams, without combine logits.
    hyper_connection_mixer: HyperConnection,
    /// The width of one stream and the stream count.
    hidden: usize,
    streams: usize,
    cache: CacheShape,
    /// The ring its layer keeps per request slot.
    state: StateShape,
    /// The module's path in the checkpoint, which names its profiler range.
    path: String,
}

impl Mtp {
    /// The drafter of `config` at `vb` (`mtp`), at the ranks `splits` state of the tensor and
    /// the expert group, for a deployment that drafts `draft_tokens` tokens a step. The two
    /// input projections are whole on every rank.
    ///
    /// # Errors
    ///
    /// A checkpoint that carries no drafter or one of several layers (module documentation),
    /// or a module the loader refuses.
    pub fn load(
        config: &Config,
        vb: VarBuilder<'_, '_>,
        splits: (Split, Split),
        draft_tokens: u32,
    ) -> Result<Self> {
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

        let (text, streams) = (&config.text, config.hyper_connection.streams);
        let hidden = text.hidden_size;
        let norm = |width: u32, module: &str| {
            RmsNorm::load(
                &RmsNormConfig {
                    width,
                    eps: text.rms_norm_eps,
                    centred: true,
                    output: None,
                },
                vb.pp(module),
            )
        };
        let projection = |module: &str| {
            Linear::load(
                &LinearConfig {
                    inputs: hidden,
                    outputs: hidden,
                    bias: false,
                    parallel: Parallel::None,
                },
                vb.pp(module),
            )
        };
        let fc_embedding = projection("fc_embedding")?;

        Ok(Self {
            pre_fc_norm_embedding: norm(hidden, "pre_fc_norm_embedding")?,
            pre_fc_norm_hidden: norm(hidden * streams, "pre_fc_norm_hidden")?,
            fc_hidden: projection("fc_hidden")?,
            rotary_emb: Rotary::load(&text.rope, vb.pp("rotary_emb"))?,
            // The drafter's layer attends, without a PLE layer; its experts are few enough
            // to hold decoded when the export stores them as block-scaled FP8.
            layer: DecoderLayer::load(
                config,
                vb.pp("layers").pp(0),
                false,
                false,
                splits,
                draft_tokens,
                None,
                true,
            )?,
            hyper_connection_mixer: HyperConnection::load(
                config,
                vb.pp("hyper_connection_mixer"),
                false,
            )?,
            hidden: hidden as usize,
            streams: streams as usize,
            cache: cache_shape(config, splits.0, 1),
            state: StateShape {
                layers: 1,
                fields: IndexedAttention::state_fields(
                    config,
                    fc_embedding.weight().dtype(),
                    draft_tokens,
                ),
            },
            fc_embedding,
            path: vb.path().to_owned(),
        })
    }
}

impl Drafter for Mtp {
    fn cache(&self) -> CacheShape {
        self.cache
    }

    fn state(&self) -> Vec<StateShape> {
        vec![self.state.clone()]
    }

    /// Refused: the drafter's indexer keeps a ring per request, so its forward is
    /// [`Self::forward_stateful`].
    fn forward(
        &self,
        _batch: &TokenBatch,
        _cache: &KvCache,
        _embedded: &Tensor,
        _hidden: &Tensor,
    ) -> nn::Result<Tensor> {
        Err(Error::Config(
            "the drafter's layer keeps a ring of index keys per request: it runs through \
             forward_stateful with that state"
                .to_owned(),
        )
        .into())
    }

    /// The drafter over `batch`: `embedded [tokens, H]` holds the embedding of each entry's
    /// next token and `hidden [tokens, C · H]` the streams each entry reads; returns the
    /// drafter's streams, `[tokens, C · H]`.
    fn forward_stateful(
        &self,
        batch: &TokenBatch,
        cache: &KvCache,
        state: Option<&RecurrentState>,
        embedded: &Tensor,
        hidden: &Tensor,
    ) -> nn::Result<Tensor> {
        let (tokens, width) = (batch.tokens(), self.hidden * self.streams);
        if embedded.dims() != [tokens, self.hidden] || hidden.dims() != [tokens, width] {
            return Err(Error::Config(format!(
                "the drafter's embeddings are {:?} and its streams {:?}; [{tokens}, {}] and \
                 [{tokens}, {width}] expected",
                embedded.dims(),
                hidden.dims(),
                self.hidden
            ))
            .into());
        }

        let forward = || -> Result<Tensor> {
            let state = state.ok_or_else(|| {
                Error::Config(
                    "the drafter's ring is a state, and the forward was lent none".to_owned(),
                )
            })?;
            let dtype = hidden.dtype();

            // The input fusion: the embedding's projection added to every stream's.
            let token = self
                .fc_embedding
                .forward(&self.pre_fc_norm_embedding.forward(embedded)?)?;
            let projected = self.fc_hidden.forward(
                &self
                    .pre_fc_norm_hidden
                    .forward(hidden)?
                    .reshape((tokens * self.streams, self.hidden))?,
            )?;
            let fused = projected
                .to_dtype(DType::F32)?
                .reshape((tokens, self.streams, self.hidden))?
                .broadcast_add(&token.to_dtype(DType::F32)?.unsqueeze(1)?)?
                .to_dtype(dtype)?
                .reshape((tokens, width))?;

            let coordinates = coordinates(batch)?;
            let (cos, sin) = self.rotary_emb.forward(&coordinates)?;
            let columns = token_columns(batch, state)?;

            let (streams, output, logits) = self.layer.forward(
                &fused,
                None,
                (&cos, &sin),
                &self.rotary_emb,
                &coordinates,
                &columns,
                None,
                batch,
                Some(cache),
                state,
                None,
                0,
                (0, 0),
            )?;
            HyperConnection::combine(&streams, &output, Some(&logits), self.streams)
        };

        Ok(nn::profiler::range(DRAFTER, &self.path, forward)?)
    }

    /// The rows the model's head projects of the drafter's streams: normalized and mixed by
    /// the drafter's own `hyper_connection_mixer`, `[rows, H]`.
    fn logits_rows(&self, hidden: &Tensor) -> nn::Result<Tensor> {
        let mixer = &self.hyper_connection_mixer;

        Ok(mixer.mix(&mixer.norm(hidden)?)?.0)
    }
}

impl std::fmt::Debug for Mtp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Mtp")
            .field("hidden", &self.hidden)
            .field("streams", &self.streams)
            .field("cache", &self.cache)
            .finish_non_exhaustive()
    }
}
