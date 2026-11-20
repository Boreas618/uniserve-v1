//! The Qwen3 causal language models: `Qwen3MoeForCausalLM` and the language model of
//! `Qwen3_5MoeForConditionalGeneration` (Qwen3.6-35B-A3B among them), one implementation over
//! the family's [`DecoderLayer`].
//!
//! ```text
//! embedding:  hidden = Σ_ranks embed_shard(ids)                         (vocabulary parallel)
//! layers:     attention over the paged key/value cache                  (crate::qwen3::decoder)
//!             or gated delta networks over the request slots' state     (Qwen3.5's linear layers)
//! experts:    Σ_routes weight · expert(x) [+ sigmoid(g · x) · shared(x)] (top-k, FP32 weights)
//! final:      hidden = norm(hidden + residual)
//! logits:     the vocabulary-parallel head, FP32
//! ```
//!
//! ## Rotary positions
//!
//! A Qwen3.5 checkpoint rotates the leading quarter of each attention head by interleaved
//! M-RoPE over three axes. A text batch places every token at the same position on all three
//! axes, under which interleaved M-RoPE is the default spectrum over the rotated channels, so
//! a text batch turns by that spectrum over its one position axis; a batch that holds image
//! rows states three axes ([`TokenBatch::rope_positions`]) and turns by the sectioned
//! spectrum.
//!
//! ## Experts
//!
//! The routed experts of each layer are held in the slots an [`ExpertOrder`] states
//! (`docs/admin.md`, Expert redistribution), by the module path
//! [`nn::causal_lm::expert_module`] names.
//!
//! ## Pipeline stages and sequence members
//!
//! A deployment splits the model along one axis ([`Config::check_placement`]). A **pipeline
//! stage** holds its contiguous share of the decoder layers and, whole, the embedding, the final
//! norm, the head, the drafter and the vision tower. Stage 0 starts from the embeddings; every
//! later stage starts from the layer output and the residual stream the stage before it hands
//! on, the two tensors a layer boundary of the whole stack carries, which that stage broadcasts
//! over the group; the last stage normalizes and broadcasts the rows to every stage, so every
//! stage's forward returns the whole model's rows.
//! A **sequence member** holds every weight and takes its share of the batch's rows
//! ([`ulysses::share`]); its attention reaches its heads of every row
//! ([`nn::modules::Attention`]), and the rows are gathered after the final norm. Both pass
//! their rows over the forward's group (`nn::collective`).

use std::ops::Range;

use nn::causal_lm::CausalLmConfig;
use nn::family::{
    CacheField, ExpertOrder, ExpertShape, FusedExperts, SequenceSplit, StateShape, TextModel,
    TextPlacement, TokenHistory,
};
use nn::modules::{
    Embedding, EmbeddingConfig, LmHead, LmHeadConfig, RmsNorm, Rotary, RotaryConfig, Spectrum,
    ulysses,
};
use nn::{
    CausalLm, DType, Drafter, ExpertExchange, KvCache, RecurrentState, Split, Tensor, TokenBatch,
    VarBuilder, Vision, functional,
};
use nn::{Checkpoint, Result, collective};

use crate::qwen3::ID;
use crate::qwen3::config::{Config, QWEN3_MOE};
use crate::qwen3::decoder::{self, DecoderLayer};
use crate::qwen3::mtp::Mtp;
use crate::qwen3::vision::VisionTower;

/// The profiler class of the whole decoder stack, at path `model`.
pub(crate) const TRANSFORMER: &str = "models.qwen3.transformer.Transformer";

/// A Qwen3 causal language model.
pub struct Qwen3ForCausalLm {
    config: Config,
    tensor: Split,
    /// This rank's pipeline stage.
    pipeline: Split,
    /// This rank's member of the sequence split.
    sequence: SequenceSplit,
    /// The decoder layers of the stack the stage holds, `layers` in order.
    held: Range<u32>,
    embed_tokens: Embedding,
    /// The rotation of a batch of one position axis.
    rotary_emb: Rotary,
    /// The sectioned rotation a batch with image rows turns by, when the model takes images.
    mrope: Option<Rotary>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: LmHead,
    mtp: Option<Mtp>,
    visual: Option<VisionTower>,
}

/// The module path the language model's modules sit under.
fn language_model(config: &Config) -> &'static str {
    if config.class == QWEN3_MOE {
        "model"
    } else {
        "model.language_model"
    }
}

/// The rotary embedding a batch of one position axis turns by: the default spectrum over the
/// rotated channels of an interleaved M-RoPE (module documentation), or the configuration's
/// own.
pub(crate) fn text_rotary(rope: RotaryConfig) -> RotaryConfig {
    match rope.spectrum {
        Spectrum::Sectioned { .. } => RotaryConfig {
            spectrum: Spectrum::Default,
            ..rope
        },
        _ => rope,
    }
}

impl Qwen3ForCausalLm {
    /// The model `config` describes, the configuration of `checkpoint`, its modules read under
    /// `vb` (the checkpoint's root), at the ranks `placement` states, with the routed experts in
    /// the slots `order` states, the drafter when the deployment speculates, and the vision
    /// tower the configuration declares when the deployment serves it
    /// ([`VisionTower::served`]).
    ///
    /// A pipeline stage loads its share of the decoder layers and every other module whole; a
    /// sequence member loads every module, its attention exchanging its rows over the split
    /// (module documentation).
    ///
    /// # Errors
    ///
    /// A placement the model refuses ([`Config::check_placement`]), a drafter the checkpoint
    /// does not carry, a served tower the checkpoint does not hold or the model does not serve
    /// ([`VisionTower::served`]), or a module the loader refuses.
    pub fn load(
        config: &Config,
        checkpoint: &Checkpoint,
        vb: VarBuilder<'_, '_>,
        placement: &TextPlacement,
        order: &ExpertOrder,
    ) -> Result<Self> {
        let held = config.check_placement(placement)?;

        let text = &config.text;
        let model = vb.pp(language_model(config));

        let embed_tokens = Embedding::load(
            &EmbeddingConfig {
                vocabulary: text.vocab_size,
                hidden: text.hidden_size,
                split: placement.tensor,
            },
            model.pp("embed_tokens"),
        )?;
        let lm_head = if text.tie_word_embeddings {
            LmHead::tied(&embed_tokens)?
        } else {
            LmHead::load(
                &LmHeadConfig {
                    vocabulary: text.vocab_size,
                    hidden: text.hidden_size,
                    split: placement.tensor,
                },
                vb.pp("lm_head"),
            )?
        };

        let layers = held
            .clone()
            .map(|layer| {
                DecoderLayer::load(
                    config,
                    model.pp("layers").pp(layer),
                    text.linear_layers[layer as usize],
                    config.moe.as_ref().filter(|_| config.routes(layer)),
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

        // A model that takes images turns their rows by the sectioned spectrum.
        let rotary = model.pp("rotary_emb");
        let visual = VisionTower::served(config.vision.as_ref(), checkpoint, &vb, placement)?;
        let mrope = match visual {
            Some(_) => Some(Rotary::load(&text.rope, rotary.pp("sectioned"))?),
            None => None,
        };

        Ok(Self {
            config: config.clone(),
            tensor: placement.tensor,
            pipeline: placement.pipeline,
            sequence: placement.sequence,
            held,
            embed_tokens,
            rotary_emb: Rotary::load(&text_rotary(text.rope), rotary)?,
            mrope,
            layers,
            norm: RmsNorm::load(&decoder::stream_norm(text), model.pp("norm"))?,
            lm_head,
            mtp,
            visual,
        })
    }

    /// The checkpoint configuration this model was built for.
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// The decoder layers this rank holds whose mixer is linear (`true`) or attention
    /// (`false`).
    fn held_layers(&self, linear: bool) -> u32 {
        self.held
            .clone()
            .filter(|layer| self.config.text.linear_layers[*layer as usize] == linear)
            .count() as u32
    }

    /// Refuses a forward whose group is not the split `split` of `axis`, which the model was
    /// loaded for: its rows pass over that group.
    fn require_group(axis: &str, split: Split) -> Result<()> {
        let members = collective::split().map(|(_, members)| members);
        if members != Some(u64::from(split.size())) {
            return Err(ID.config(format!(
                "a {axis} split over {} ranks runs in a forward whose group is {members:?}",
                split.size()
            )));
        }
        Ok(())
    }

    /// The handoff of stage `root` of the pipeline: the layer output and the residual stream
    /// `rows` the stage's layers left, which every stage takes from it, the two tensors a layer
    /// boundary of the whole stack carries. On `root` `rows` are its own and come back as they
    /// are; on every other stage they state the shape and element type alone.
    ///
    /// Every stage takes part in every handoff, in stage order, so the stages call one sequence
    /// of collectives over the group.
    fn handoff(&self, root: u32, (output, stream): (&Tensor, &Tensor)) -> Result<(Tensor, Tensor)> {
        Self::require_group("pipeline", self.pipeline)?;

        let output = collective::broadcast(&output.contiguous()?, root)?;
        let stream = collective::broadcast(&stream.contiguous()?, root)?;
        Ok((output, stream))
    }

    /// The rows this stage's layers start from, `(hidden, residual)`: the batch's embeddings
    /// `hidden`, which open the residual stream, on the first stage; on every later stage the
    /// rows of the stage before it, taken from the handoffs of every earlier stage in order.
    fn stage_input(&self, hidden: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let mut input = (hidden.clone(), None);
        for root in 0..self.pipeline.rank() {
            let (output, stream) = self.handoff(root, (hidden, hidden))?;
            input = (output, Some(stream));
        }
        Ok(input)
    }

    /// The model's rows after this stage's layers left `hidden` and `residual`: the stage hands
    /// them to the later stages and takes part in their handoffs; the last stage adds them and
    /// normalizes, and every stage of a pipeline ends with the last stage's rows, which it
    /// broadcasts.
    fn stage_output(&self, hidden: Tensor, residual: Tensor) -> Result<Tensor> {
        let (stage, last) = (self.pipeline.rank(), self.pipeline.size() - 1);
        if stage < last {
            for root in stage..last {
                self.handoff(root, (&hidden, &residual))?;
            }

            // The rows every stage ends with are the last stage's; this stage's output only
            // states their shape and element type.
            return collective::broadcast(&hidden.contiguous()?, last);
        }

        let (normalized, _) = self.norm.forward_residual(&hidden, &residual)?;
        if last == 0 {
            return Ok(normalized);
        }
        collective::broadcast(&normalized.contiguous()?, last)
    }
}

impl CausalLm for Qwen3ForCausalLm {
    fn embed_input_ids(&self, ids: &Tensor) -> nn::Result<Tensor> {
        self.embed_tokens.forward(ids)
    }

    /// The layers over the batch's embeddings `hidden`, which open the residual stream, and
    /// the final norm, which adds the last layer's output into the stream as it normalizes.
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
            let (rotary, positions) = match (&batch.rope_positions, &self.mrope) {
                (None, _) => (&self.rotary_emb, &batch.positions),
                (Some(coordinates), Some(mrope)) => (mrope, coordinates),
                (Some(_), None) => {
                    return Err(ID.config(
                        "a batch states image coordinates for a model without vision input"
                            .to_owned(),
                    ));
                }
            };
            // Every row's factors: a sequence member's attention turns every row of its heads.
            let (cos, sin) = rotary.forward(positions)?;

            // The model's attention layers share one cache group. Its workspace is execution's
            // and every forward reuses it, so the decode kernel's arrival counters are cleared
            // once before the first layer, for the batch's rows of this rank's query heads (the
            // load checked that the splits divide the heads).
            let group = cache.first();
            if let Some(group) = group {
                let text = &self.config.text;
                let q_heads =
                    text.num_attention_heads / (self.tensor.size() * self.sequence.split.size());
                functional::zero_arrival_counters_(&group.workspace, batch.rows(), q_heads)?;
            }

            // A sequence member's layers run over its share of the rows. The batch's live
            // count is the whole batch's, not the share's, so the member's experts route every
            // row of its share.
            let members = self.sequence.split;
            let shared;
            let (rows, batch) = if members.size() > 1 {
                Self::require_group("sequence", members)?;
                shared = TokenBatch {
                    live_tokens: None,
                    ..batch.clone()
                };
                (ulysses::share(hidden, members)?, &shared)
            } else {
                (hidden.clone(), batch)
            };

            // Each layer reads its own layer of the cache or of the state: its place among
            // the stage's layers of its mixer.
            let (mut attention, mut linear) = (0, 0);
            let (mut hidden, mut residual) = self.stage_input(&rows)?;
            for layer in &self.layers {
                let ordinal = if layer.is_linear() {
                    &mut linear
                } else {
                    &mut attention
                };

                let (output, stream) = layer.forward(
                    &hidden,
                    residual.as_ref(),
                    (&cos, &sin),
                    batch,
                    group,
                    state,
                    exchange,
                    *ordinal,
                )?;
                (hidden, residual) = (output, Some(stream));
                *ordinal += 1;
            }

            let residual =
                residual.ok_or_else(|| ID.config("a decoder stack of no layers".to_owned()))?;
            let output = self.stage_output(hidden, residual)?;
            if members.size() > 1 {
                return ulysses::gather(&output, members, tokens);
            }
            Ok(output)
        };

        nn::profiler::range(TRANSFORMER, "model", forward)
    }

    fn compute_logits(&self, hidden: &Tensor) -> nn::Result<Tensor> {
        self.lm_head.forward(hidden)
    }

    fn join_expert_step(&self, exchange: &ExpertExchange) -> nn::Result<()> {
        nn::causal_lm::join_expert_step(
            self.layers.iter().filter_map(DecoderLayer::experts),
            exchange,
        )
    }
}

impl TextModel for Qwen3ForCausalLm {
    fn hidden(&self) -> u32 {
        self.config.text.hidden_size
    }

    fn vocabulary(&self) -> u32 {
        self.config.text.vocab_size
    }

    fn dtype(&self) -> DType {
        self.embed_tokens.weight().dtype()
    }

    /// The keys and values of every attention layer the rank holds.
    fn cache(&self) -> Vec<CacheField> {
        let text = &self.config.text;

        // The load checked that the split divides the heads; at most one of the two splits is
        // above one.
        let kv_heads = text.num_key_value_heads / (self.tensor.size() * self.sequence.split.size());
        CacheField::keys_and_values(self.held_layers(false), kv_heads, text.head_dim)
    }

    /// The gated delta networks' fields, every linear layer the rank holds keeping the same;
    /// none for a stack whose every layer attends.
    fn state_layers(&self) -> Vec<StateShape> {
        self.config
            .text
            .gated_delta_net
            .map(|network| StateShape {
                layers: self.held_layers(true),
                fields: network.state_fields(self.dtype()),
            })
            .into_iter()
            .collect()
    }

    fn experts(&self) -> Option<ExpertShape> {
        let moe = self.config.moe.as_ref()?;
        let routes = self.layers.iter().any(|layer| layer.experts().is_some());

        Some(ExpertShape {
            experts: moe.num_experts,
            intermediate: moe.moe_intermediate_size,
            top: moe.num_experts_per_tok,
            // Calibrated NVFP4 experts run in the exchange's one fused launch.
            fused: if routes && self.config.quantization.is_encoded() {
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

    /// The layers read their embeddings alone.
    fn token_history(&self) -> Option<TokenHistory> {
        None
    }
}

impl std::fmt::Debug for Qwen3ForCausalLm {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Qwen3ForCausalLm")
            .field("class", &self.config.class)
            .field("layers", &self.held)
            .field("tensor", &self.tensor)
            .field("pipeline", &self.pipeline)
            .field("sequence", &self.sequence)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use nn::config::Fields;

    use super::*;

    #[test]
    fn a_text_batch_turns_by_the_default_spectrum_of_an_mrope() {
        let config = crate::qwen3::fixtures::hybrid();
        let config =
            Config::parse(&Fields::of("config.json", &config).expect("an object")).expect("parsed");

        let text = text_rotary(config.text.rope);
        assert_eq!(
            (text.spectrum, text.dim, text.theta),
            (Spectrum::Default, 64, 1e7)
        );

        // Equal positions on every axis read the same frequencies under both spectra.
        assert_eq!(
            text.inverse_frequencies().expect("frequencies"),
            config.text.rope.inverse_frequencies().expect("frequencies"),
        );
    }

    #[test]
    fn an_order_names_the_routed_layers_by_their_expert_modules() {
        let config = crate::qwen3::fixtures::moe();
        let config =
            Config::parse(&Fields::of("config.json", &config).expect("an object")).expect("parsed");

        let layers = config.expert_layers();
        assert_eq!(layers.len(), 4);
        assert_eq!(layers.get("layers.2.mlp.experts"), Some(&128));
    }
}
