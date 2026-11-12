//! The Qwen3 causal language models: `Qwen3MoeForCausalLM` and the language model of
//! `Qwen3_5MoeForConditionalGeneration` (Qwen3.6-35B-A3B among them), one implementation over
//! the family's [`DecoderLayer`].
//!
//! ```text
//! embedding:  hidden = Σ_ranks embed_shard(ids)                         (vocabulary parallel)
//! layers:     attention over the paged key/value cache                  (crate::decoder)
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
//! (`docs/admin.md`, Expert redistribution), by the module path [`expert_module`] names.

use std::collections::BTreeMap;

use nn::family::{CacheShape, ExpertOrder, ExpertShape, StateShape, TextModel, TextPlacement};
use nn::modules::{
    Embedding, EmbeddingConfig, LmHead, LmHeadConfig, RmsNorm, Rotary, RotaryConfig, Spectrum,
};
use nn::{
    CausalLm, DType, Drafter, ExpertExchange, KvCache, RecurrentState, Split, Tensor, TokenBatch,
    VarBuilder, Vision, functional,
};

use crate::config::{Config, QWEN3_MOE};
use crate::decoder::{self, DecoderLayer};
use crate::mtp::Mtp;
use crate::vision::VisionTower;
use crate::{Error, Result};

/// The profiler class of the whole decoder stack, at path `model`.
const TRANSFORMER: &str = "models.qwen3.transformer.Transformer";

/// A Qwen3 causal language model.
pub struct Qwen3ForCausalLm {
    config: Config,
    tensor: Split,
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

/// The path an [`ExpertOrder`] names the routed experts of decoder layer `layer` by.
pub fn expert_module(layer: u32) -> String {
    format!("layers.{layer}.mlp.experts")
}

/// The routed experts of every layer of `config` by module path, each with its expert count:
/// what an expert redistribution may order.
pub fn expert_layers(config: &Config) -> BTreeMap<String, u32> {
    let Some(moe) = &config.moe else {
        return BTreeMap::new();
    };

    (0..config.text.num_hidden_layers)
        .filter(|layer| config.routes(*layer))
        .map(|layer| (expert_module(layer), moe.num_experts))
        .collect()
}

impl Qwen3ForCausalLm {
    /// The model `config` describes, its modules read under `vb` (the checkpoint's root), at
    /// the ranks `placement` states, with the routed experts in the slots `order` states, the
    /// drafter when the deployment speculates, and the vision tower when the configuration
    /// states the processor its images are sized by.
    ///
    /// # Errors
    ///
    /// A split the model does not divide, an order that names no routed layer, a drafter the
    /// checkpoint does not carry, a tower with DeepStack features (a causal LM's prefill adds
    /// none), or a module the loader refuses.
    pub fn load(
        config: &Config,
        vb: VarBuilder<'_, '_>,
        placement: &TextPlacement,
        order: &ExpertOrder,
    ) -> Result<Self> {
        let splits = (placement.tensor, placement.experts);
        config.check_splits(splits.0, splits.1)?;

        let routed = expert_layers(config);
        if let Some(stray) = order.keys().find(|module| !routed.contains_key(*module)) {
            return Err(Error::Config(format!(
                "{stray} is not a routed expert layer of this model"
            )));
        }

        let text = &config.text;
        let model = vb.pp(language_model(config));

        let embed_tokens = Embedding::load(
            &EmbeddingConfig {
                vocabulary: text.vocab_size,
                hidden: text.hidden_size,
                split: splits.0,
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
                    split: splits.0,
                },
                vb.pp("lm_head"),
            )?
        };

        let layers = (0..text.num_hidden_layers)
            .map(|layer| {
                DecoderLayer::load(
                    config,
                    model.pp("layers").pp(layer),
                    text.linear_layers[layer as usize],
                    config.routes(layer),
                    splits,
                    order.get(&expert_module(layer)),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let mtp = if placement.drafter {
            Some(Mtp::load(config, vb.pp("mtp"), splits)?)
        } else {
            None
        };

        let rotary = model.pp("rotary_emb");
        let (visual, mrope) = match (config.images, &config.vision) {
            (None, _) => (None, None),
            (Some(_), None) => {
                return Err(Error::Config(
                    "the checkpoint states processor files and no tower".to_owned(),
                ));
            }
            (Some((processor, pixels)), Some(vision)) => {
                if !vision.deepstack_visual_indexes.is_empty() {
                    return Err(Error::Config(
                        "a causal language model's prefill adds no DeepStack features, but the \
                         tower states some"
                            .to_owned(),
                    ));
                }

                (
                    Some(VisionTower::load(
                        vision,
                        vb.pp("model.visual"),
                        processor,
                        pixels,
                    )?),
                    Some(Rotary::load(&text.rope, rotary.pp("sectioned"))?),
                )
            }
        };

        Ok(Self {
            config: config.clone(),
            tensor: splits.0,
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
            return Err(Error::Config(format!(
                "hidden is {:?}, [{tokens}, {width}] expected",
                hidden.dims()
            ))
            .into());
        }

        let forward = || -> Result<Tensor> {
            let (rotary, positions) = match (&batch.rope_positions, &self.mrope) {
                (None, _) => (&self.rotary_emb, &batch.positions),
                (Some(coordinates), Some(mrope)) => (mrope, coordinates),
                (Some(_), None) => {
                    return Err(Error::Config(
                        "a batch states image coordinates for a model without vision input"
                            .to_owned(),
                    ));
                }
            };
            let (cos, sin) = rotary.forward(positions)?;

            // The model's attention layers share one cache group. Its workspace is execution's
            // and every forward reuses it, so the decode kernel's arrival counters are cleared
            // once before the first layer.
            let group = cache.first();
            if let Some(group) = group {
                functional::zero_arrival_counters_(&group.workspace)?;
            }

            // Each layer reads its own layer of the cache or of the state: its place among
            // the layers of its mixer.
            let (mut attention, mut linear) = (0, 0);
            let (mut hidden, mut residual) = (hidden.clone(), None);
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
                residual.ok_or_else(|| Error::Config("a decoder stack of no layers".to_owned()))?;
            let (normalized, _) = self.norm.forward_residual(&hidden, &residual)?;
            Ok(normalized)
        };

        Ok(nn::profiler::range(TRANSFORMER, "model", forward)?)
    }

    fn compute_logits(&self, hidden: &Tensor) -> nn::Result<Tensor> {
        self.lm_head.forward(hidden)
    }

    fn join_expert_step(&self, exchange: &ExpertExchange) -> nn::Result<()> {
        for experts in self.layers.iter().filter_map(DecoderLayer::experts) {
            experts.join_expert_step(exchange)?;
        }
        Ok(())
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

    fn cache(&self) -> CacheShape {
        let text = &self.config.text;

        CacheShape {
            layers: self.config.attention_layers(),
            // The load checked that the split divides the heads.
            kv_heads: text.num_key_value_heads / self.tensor.size,
            head_dim: text.head_dim,
            index_keys: None,
        }
    }

    fn state(&self) -> Option<StateShape> {
        let network = self.config.text.gated_delta_net?;

        Some(StateShape {
            layers: self.config.linear_layers(),
            fields: network.state_fields(self.dtype()),
        })
    }

    fn experts(&self) -> Option<ExpertShape> {
        let moe = self.config.moe.as_ref()?;
        let routes = self.layers.iter().any(|layer| layer.experts().is_some());

        Some(ExpertShape {
            experts: moe.num_experts,
            intermediate: moe.moe_intermediate_size,
            top: moe.num_experts_per_tok,
            encoded: routes && self.config.quantization.is_encoded(),
        })
    }

    fn drafter(&self) -> Option<&dyn Drafter> {
        self.mtp.as_ref().map(|mtp| mtp as &dyn Drafter)
    }

    fn vision(&self) -> Option<&dyn Vision> {
        self.visual.as_ref().map(|tower| tower as &dyn Vision)
    }
}

impl std::fmt::Debug for Qwen3ForCausalLm {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Qwen3ForCausalLm")
            .field("class", &self.config.class)
            .field("layers", &self.layers.len())
            .field("tensor", &self.tensor)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use nn::config::Fields;

    use super::*;

    #[test]
    fn a_text_batch_turns_by_the_default_spectrum_of_an_mrope() {
        let config = crate::config::fixtures::hybrid();
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
        let config = crate::config::fixtures::moe();
        let config =
            Config::parse(&Fields::of("config.json", &config).expect("an object")).expect("parsed");

        let layers = expert_layers(&config);
        assert_eq!(layers.len(), 4);
        assert_eq!(layers.get("layers.2.mlp.experts"), Some(&128));
    }
}
