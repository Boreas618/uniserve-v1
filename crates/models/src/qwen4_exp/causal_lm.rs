//! The causal language model of `Qwen4ExpForConditionalGeneration` (vLLM's
//! `Qwen4ExpForCausalLM`, `vllm:vllm/models/qwen4_exp/nvidia/model.py:540-671`).
//!
//! ```text
//! embedding:  x = Σ_ranks embed_shard(ids)                          [tokens, H]
//! streams:    S = [x | x | … ]                                      [tokens, C · H]
//! layers:     S, y, r ← layer(S, (y, r))                            (crate::qwen4_exp::decoder)
//! read-out:   S ← combine(S, y, r)                                  the forward's result
//! logits:     lm_head(mix(norm(S)))                                 no final norm
//! ```
//!
//! [`CausalLm::forward`] returns the streams, which the drafter reads, and
//! [`CausalLm::compute_logits`] mixes the rows it is given through `hyper_connection_mixer`
//! and projects them: the mix is row-wise, so it runs on the rows logits are asked for.
//!
//! ## The state the layers keep
//!
//! The recurrent state's layers are, in order ([`TextModel::state_layers`]): the gated delta
//! networks (a convolution window and the delta rule's matrices each), the PLE layers (a
//! convolution window each), then the attention layers (a ring of raw index keys and its
//! rotary positions each). A deployment that drafts lends [`nn::VerifySlots`]: the windows
//! hold one row more per draft token, and a request owns one slot of the delta rule's
//! matrices per row of a step.
//!
//! ## Rotary positions
//!
//! Every batch turns by the sectioned M-RoPE spectrum over three axes: a batch of text states
//! one position per token, read on every axis, and a batch that holds image rows states three
//! ([`TokenBatch::rope_positions`]).

use nn::family::{
    CacheField, ExpertOrder, ExpertShape, FieldOwner, Format, FusedExperts, History, INDEX_KEY,
    Record, StateShape, TextModel, TextPlacement, TokenHistory,
};
use nn::modules::{Embedding, EmbeddingConfig, LmHead, LmHeadConfig, Rotary};
use nn::{
    CausalLm, Checkpoint, DType, Device, Drafter, ExpertExchange, KvCache, RecurrentState, Result,
    Split, Tensor, TokenBatch, VarBuilder, Vision,
};

use crate::qwen3::VisionTower;
use crate::qwen4_exp::ID;
use crate::qwen4_exp::attention::IndexedAttention;
use crate::qwen4_exp::config::Config;
use crate::qwen4_exp::decoder::DecoderLayer;
use crate::qwen4_exp::hyper_connection::HyperConnection;
use crate::qwen4_exp::mtp::Mtp;
use crate::qwen4_exp::ngram::{PleLayer, int};

/// The profiler class of the whole decoder stack, at path `model`.
pub(crate) const TRANSFORMER: &str = "models.qwen4_exp.transformer.Transformer";

/// The module path the language model's modules sit under.
const LANGUAGE_MODEL: &str = "model.language_model";

/// The per-token columns of a forward ([`token_columns`]), by their index.
///
/// `[tokens]` U32: the batch row each token is of, its row of the cache's page table.
pub const ROW: usize = 0;
/// `[tokens]` I32: the token's position in its request's token order.
pub const POSITION: usize = 1;
/// `[tokens]` I32: the token's request slot, negative for a token of a padding row.
pub const SLOT: usize = 2;
/// `[tokens]` I32: the tokens of the token's state row before it.
pub const LEADING: usize = 3;
/// `[tokens]` I32: the tokens of the token's state row after it.
pub const TRAILING: usize = 4;
/// `[tokens]` I32: the tokens of the token's batch row before it.
pub const INDEX: usize = 5;

/// For each of `tokens` packed tokens, the row of `offsets` (`[rows + 1]` U32 packed offsets)
/// it lies in, clamped to the last row, with the row's first token and the first token past
/// it: `(row, first, end)`, `[tokens]` each, the row U32 and the bounds I32.
fn rows_of(offsets: &Tensor, tokens: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let rows = offsets.elem_count() - 1;
    let device = offsets.device();
    let offsets = offsets.to_dtype(DType::I32)?;
    let token = Tensor::arange(0_i32, tokens as i32, &Device::Cpu)?.to_device(device)?;

    // A token lies in the row whose end is the first past it: the count of ends at or
    // before it.
    let ends = offsets.narrow(0, 1, rows)?;
    let row = token
        .unsqueeze(1)?
        .broadcast_ge(&ends.unsqueeze(0)?)?
        .to_dtype(DType::I32)?
        .sum(1)?
        .minimum((rows - 1) as f64)?
        .to_dtype(DType::U32)?;

    let first = offsets.narrow(0, 0, rows)?.index_select(&row, 0)?;
    let end = ends.index_select(&row, 0)?;
    Ok((row, first, end))
}

/// The per-token columns the layers of a forward read, as tensor expressions over the batch's
/// and the state's row columns: [`ROW`], [`POSITION`], [`SLOT`], [`LEADING`], [`TRAILING`]
/// and [`INDEX`].
///
/// A batch's rows are its attention rows and a state's rows its requests' (a verification's
/// tokens attend as one row each while one state row covers them), so a token's place is
/// stated in both.
///
/// # Errors
///
/// What a tensor operation refuses.
pub fn token_columns(batch: &TokenBatch, state: &RecurrentState) -> Result<Vec<Tensor>> {
    let tokens = batch.tokens();
    let device = batch.query_offsets.device();
    let token = Tensor::arange(0_i32, tokens as i32, &Device::Cpu)?.to_device(device)?;

    // A row has seen `seq_lens` tokens, its new ones the last of them.
    let (row, first, end) = rows_of(&batch.query_offsets, tokens)?;
    let seen = batch.seq_lens.to_dtype(DType::I32)?.index_select(&row, 0)?;
    let position = seen.sub(&end)?.add(&token)?.maximum(0.0)?;
    let index = token.sub(&first)?.maximum(0.0)?;

    // A token past the state's last row is padding: it names no slot.
    let (state_row, state_first, state_end) = rows_of(&state.query_offsets, tokens)?;
    let slot = state.slots.view(DType::I32)?.index_select(&state_row, 0)?;
    let slot = token
        .lt(&state_end)?
        .where_cond(&slot, &int(-1, &slot)?.broadcast_as(slot.shape())?)?;
    let leading = token.sub(&state_first)?.maximum(0.0)?;
    let trailing = state_end.sub(&token)?.broadcast_sub(&int(1, &token)?)?;

    Ok(vec![row, position, slot, leading, trailing, index])
}

/// The rotary coordinates of a batch's tokens, `[3, tokens]` I32: the batch's own when it
/// holds image rows, and each token's position on every axis otherwise.
pub fn coordinates(batch: &TokenBatch) -> Result<Tensor> {
    match &batch.rope_positions {
        Some(coordinates) => Ok(coordinates.clone()),
        None => {
            let position = &batch.positions;
            Ok(Tensor::stack(&[position, position, position], 0)?)
        }
    }
}

/// A Qwen3.8 Flash Next causal language model.
pub struct Qwen4ExpForCausalLm {
    config: Config,
    tensor: Split,
    /// The tokens a step of the deployment drafts.
    draft_tokens: u32,
    embed_tokens: Embedding,
    /// The sectioned rotation of every batch.
    rotary_emb: Rotary,
    layers: Vec<DecoderLayer>,
    /// The read-out of the streams: its norm and mix, without combine logits.
    hyper_connection_mixer: HyperConnection,
    lm_head: LmHead,
    mtp: Option<Mtp>,
    visual: Option<VisionTower>,
}

impl Qwen4ExpForCausalLm {
    /// The model `config` describes, the configuration of `checkpoint`, its modules read under
    /// `vb` (the checkpoint's root), at the ranks `placement` states, with the routed experts in
    /// the slots `order` states, the drafter when the deployment speculates, and the vision
    /// tower the class declares when the deployment serves it ([`VisionTower::served`]).
    ///
    /// # Errors
    ///
    /// A split the model does not divide; a pipeline or sequence split, which this model's
    /// layers do not serve; an order that names no layer; a drafter the checkpoint does not
    /// carry; a served tower the checkpoint does not hold; or a module the loader refuses.
    pub fn load(
        config: &Config,
        checkpoint: &Checkpoint,
        vb: VarBuilder<'_, '_>,
        placement: &TextPlacement,
        order: &ExpertOrder,
    ) -> Result<Self> {
        config.check_splits(placement.tensor, placement.experts)?;
        // A tensor group that holds whole experts divides the experts, not their width, and no
        // expert group divides them beside it.
        if placement.tensor_experts
            && (placement.experts.size() > 1
                || !config
                    .moe
                    .num_experts
                    .is_multiple_of(placement.tensor.size()))
        {
            return Err(ID.config(format!(
                "a tensor group of {} holding whole experts beside an expert group of {}: the \
                 tensor group alone divides the {} experts",
                placement.tensor.size(),
                placement.experts.size(),
                config.moe.num_experts
            )));
        }
        if placement.pipeline.size() > 1 || placement.sequence.split.size() > 1 {
            return Err(ID.config(format!(
                "a pipeline split over {} stages or a sequence split over {} members: this \
                 model's layers are split over tensor and expert ranks alone",
                placement.pipeline.size(),
                placement.sequence.split.size()
            )));
        }

        let text = &config.text;
        let model = vb.pp(LANGUAGE_MODEL);
        let layers = (0..text.num_hidden_layers)
            .map(|layer| {
                DecoderLayer::load(
                    config,
                    model.pp("layers").pp(layer),
                    text.linear_layers[layer as usize],
                    config.ple(layer).is_some(),
                    placement,
                    Some((layer, order)),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let mtp = if placement.drafter {
            Some(Mtp::load(config, vb.pp("mtp"), placement)?)
        } else {
            None
        };
        let visual = VisionTower::served(Some(&config.vision), checkpoint, &vb, placement)?;

        Ok(Self {
            config: config.clone(),
            tensor: placement.tensor,
            draft_tokens: placement.draft_tokens,
            embed_tokens: Embedding::load(
                &EmbeddingConfig {
                    vocabulary: text.vocab_size,
                    hidden: text.hidden_size,
                    split: placement.tensor,
                },
                model.pp("embed_tokens"),
            )?,
            rotary_emb: Rotary::load(&text.rope, model.pp("rotary_emb"))?,
            layers,
            hyper_connection_mixer: HyperConnection::load(
                config,
                model.pp("hyper_connection_mixer"),
                false,
            )?,
            lm_head: LmHead::load(
                &LmHeadConfig {
                    vocabulary: text.vocab_size,
                    hidden: text.hidden_size,
                    split: placement.tensor,
                },
                vb.pp("lm_head"),
            )?,
            mtp,
            visual,
        })
    }

    /// The checkpoint configuration this model was built for.
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// The width of the streams: `C · H`.
    fn streams_width(&self) -> usize {
        (self.config.text.hidden_size * self.config.hyper_connection.streams) as usize
    }
}

impl CausalLm for Qwen4ExpForCausalLm {
    fn embed_input_ids(&self, ids: &Tensor) -> nn::Result<Tensor> {
        self.embed_tokens.forward(ids)
    }

    /// The layers over the batch's embeddings `hidden`, `[tokens, H]`, which every stream
    /// opens as: the streams after the last layer's output is folded in, `[tokens, C · H]`.
    fn forward(
        &self,
        batch: &TokenBatch,
        cache: &[KvCache],
        state: Option<&RecurrentState>,
        exchange: Option<&ExpertExchange>,
        hidden: &Tensor,
    ) -> nn::Result<Tensor> {
        let (tokens, width) = (batch.tokens(), self.config.text.hidden_size as usize);
        if hidden.dims() != [tokens, width] {
            return Err(ID.config(format!(
                "hidden is {:?}, [{tokens}, {width}] expected",
                hidden.dims()
            )));
        }

        let forward = || -> Result<Tensor> {
            let state = state.ok_or_else(|| {
                ID.config(
                    "the model's layers keep a state, and the forward was lent none".to_owned(),
                )
            })?;
            let streams = self.config.hyper_connection.streams as usize;

            let coordinates = coordinates(batch)?;
            let (cos, sin) = self.rotary_emb.forward(&coordinates)?;
            let columns = token_columns(batch, state)?;
            let context = if self.layers.iter().any(DecoderLayer::has_ple) {
                Some(nn::modules::ngram_context(batch)?)
            } else {
                None
            };

            // The recurrent layers in the state's order: the gated delta networks, the PLE
            // layers, the attention layers' rings.
            let linear_layers = self.config.linear_layers() as usize;
            let ple_layers = self.config.ngram.layers.len();
            let group = cache.first();

            let (mut attention, mut linear, mut ple) = (0, 0, 0);
            let mut residual = Tensor::cat(&vec![hidden; streams], 1)?;
            let mut pending: Option<(Tensor, Tensor)> = None;

            for layer in &self.layers {
                let (ordinal, mixer_state) = if layer.is_linear() {
                    (linear, linear)
                } else {
                    (attention, linear_layers + ple_layers + attention)
                };

                let (folded, output, logits) = layer.forward(
                    &residual,
                    pending.as_ref().map(|(output, logits)| (output, logits)),
                    (&cos, &sin),
                    &self.rotary_emb,
                    &coordinates,
                    &columns,
                    context.as_ref(),
                    batch,
                    group,
                    state,
                    exchange,
                    ordinal,
                    (mixer_state, linear_layers + ple),
                )?;
                (residual, pending) = (folded, Some((output, logits)));

                if layer.is_linear() {
                    linear += 1;
                } else {
                    attention += 1;
                }
                if layer.has_ple() {
                    ple += 1;
                }
            }

            // The last layer's output is folded in by the read-out (`model.py:643-647`).
            match pending {
                Some((output, logits)) => {
                    HyperConnection::combine(&residual, &output, Some(&logits), streams)
                }
                None => Ok(residual),
            }
        };

        nn::profiler::range(TRANSFORMER, "model", forward)
    }

    /// The logits of `hidden`: rows of the streams, `[rows, C · H]`, which are normalized and
    /// mixed by `hyper_connection_mixer` first, or rows a drafter has mixed already,
    /// `[rows, H]` ([`Drafter::logits_rows`]).
    fn compute_logits(&self, hidden: &Tensor) -> nn::Result<Tensor> {
        let mixed = match hidden.dims() {
            &[_, width] if width == self.streams_width() => {
                let mixer = &self.hyper_connection_mixer;
                mixer.mix(&mixer.norm(hidden)?)?.0
            }
            _ => hidden.clone(),
        };

        self.lm_head.forward(&mixed)
    }

    fn join_expert_step(&self, exchange: &ExpertExchange) -> nn::Result<()> {
        nn::causal_lm::join_expert_step(self.layers.iter().map(DecoderLayer::experts), exchange)
    }
}

impl TextModel for Qwen4ExpForCausalLm {
    fn hidden(&self) -> u32 {
        self.config.text.hidden_size
    }

    fn vocabulary(&self) -> u32 {
        self.config.text.vocab_size
    }

    fn dtype(&self) -> DType {
        self.embed_tokens.weight().dtype()
    }

    fn cache(&self) -> Vec<CacheField> {
        let layers = self.config.attention_layers();
        cache_fields(&self.config, self.tensor, layers, self.dtype())
    }

    fn state_layers(&self) -> Vec<StateShape> {
        let (config, dtype, drafts) = (&self.config, self.dtype(), self.draft_tokens);
        let network = nn::modules::GatedDeltaNetConfig {
            split: self.tensor,
            draft_tokens: drafts,
            ..config.text.gated_delta_net
        };

        vec![
            StateShape {
                layers: config.linear_layers(),
                fields: network.state_fields(dtype),
            },
            StateShape {
                layers: config.ngram.layers.len() as u32,
                fields: PleLayer::state_fields(config, dtype, drafts),
            },
            StateShape {
                layers: config.attention_layers(),
                fields: IndexedAttention::state_fields(config, dtype, drafts),
            },
        ]
    }

    fn stream_width(&self) -> u32 {
        self.streams_width() as u32
    }

    fn experts(&self) -> Option<ExpertShape> {
        let moe = &self.config.moe;

        Some(ExpertShape {
            experts: moe.num_experts,
            intermediate: moe.moe_intermediate_size,
            top: moe.num_experts_per_tok,
            // Calibrated NVFP4 experts run in the exchange's one fused launch.
            fused: if self.config.quantization.is_encoded() {
                FusedExperts::NVFP4
            } else {
                FusedExperts::NONE
            },
        })
    }

    fn drafter(&self) -> Option<&dyn Drafter> {
        self.mtp.as_ref().map(|mtp| mtp as &dyn Drafter)
    }

    fn vision(&self) -> Option<&dyn Vision> {
        self.visual.as_ref().map(|tower| tower as &dyn Vision)
    }

    /// The PLE layers hash each token with the `ngram_size − 1` before it, an n-gram that
    /// starts before the request reading the checkpoint's end-of-text id there, so the
    /// forward reads token ids and their history whenever one layer holds a PLE layer.
    fn token_history(&self) -> Option<TokenHistory> {
        self.layers
            .iter()
            .any(DecoderLayer::has_ple)
            .then_some(TokenHistory {
                width: self.config.ngram.history(),
                fill: self.config.text.eos_token_id,
            })
    }
}

/// The cache fields of `layers` attention layers of `config`, as `tensor`'s rank holds them: the
/// keys and values, of which a rank of a group wider than the key/value heads holds one head,
/// and the indexer's keys, one per `compress_ratio` tokens of every layer, in `dtype`, the type
/// the layers compute them in, whatever the deployment stores keys and values in.
pub(crate) fn cache_fields(
    config: &Config,
    tensor: Split,
    layers: u32,
    dtype: DType,
) -> Vec<CacheField> {
    let (text, indexer) = (&config.text, &config.indexer);
    let kv_heads = (text.num_key_value_heads / tensor.size()).max(1);

    let mut fields = CacheField::keys_and_values(layers, kv_heads, text.head_dim);
    fields.push(CacheField {
        name: INDEX_KEY,
        owners: (0..layers)
            .map(|layer| FieldOwner {
                layer,
                ratio: indexer.compress_ratio,
            })
            .collect(),
        readers: Vec::new(),
        history: History::Full,
        record: Record::rows(Format::Elements(dtype), 1, indexer.head_dim),
    });
    fields
}

impl std::fmt::Debug for Qwen4ExpForCausalLm {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Qwen4ExpForCausalLm")
            .field("layers", &self.layers.len())
            .field("tensor", &self.tensor)
            .field("draft_tokens", &self.draft_tokens)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use nn::causal_lm::CausalLmConfig;
    use nn::config::Fields;

    use super::*;

    #[test]
    fn an_order_names_every_layer_by_its_expert_module() {
        let config = crate::qwen4_exp::fixtures::flash_next();
        let config =
            Config::parse(&Fields::of("config.json", &config).expect("an object")).expect("parsed");

        let layers = config.expert_layers();
        assert_eq!(layers.len(), 8);
        assert_eq!(layers.get("layers.7.mlp.experts"), Some(&512));
    }

    #[test]
    fn a_rank_of_four_holds_one_key_value_head_and_the_index_keys() {
        let config = crate::qwen4_exp::fixtures::flash_next();
        let config =
            Config::parse(&Fields::of("config.json", &config).expect("an object")).expect("parsed");

        // Two layers' keys and values of two heads of 256, and their index keys: one key of 128
        // BF16 elements per four tokens.
        let whole = cache_fields(&config, Split::WHOLE, 2, DType::BF16);
        let names: Vec<&str> = whole.iter().map(|field| field.name).collect();
        assert_eq!(names, [nn::family::KEY, nn::family::VALUE, INDEX_KEY]);
        for field in &whole[..2] {
            assert_eq!(field.owners.len(), 2);
            assert_eq!((field.record.heads, field.record.width), (2, 256));
            assert_eq!(field.record.format, Format::Deployment);
        }
        let keys = &whole[2];
        assert_eq!(
            keys.owners,
            [
                FieldOwner { layer: 0, ratio: 4 },
                FieldOwner { layer: 1, ratio: 4 }
            ]
        );
        assert_eq!(
            keys.record,
            Record::rows(Format::Elements(DType::BF16), 1, 128)
        );

        let rank = cache_fields(&config, Split::new(4, 3).unwrap(), 2, DType::BF16);
        assert_eq!(rank[0].record.heads, 1);
    }
}
