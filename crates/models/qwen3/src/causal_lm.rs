//! The Qwen3 causal language models: `Qwen3MoeForCausalLM` and the language model of
//! `Qwen3_5MoeForConditionalGeneration` (Qwen3.6-35B-A3B among them), one implementation over the
//! family's [`DecoderStack`].
//!
//! ```text
//! embedding:  hidden = Σ_ranks embed_shard(ids)                         (vocabulary parallel)
//! layers:     the decoder stack, paged attention over the prefix cache  (crate::decoder)
//!             or gated delta networks over the request slots' state     (Qwen3.5's linear layers)
//! experts:    combine(routes, experts(x)) [+ sigmoid(g · x) · shared(x)] (top-k, FP32 weights)
//! final:      hidden = norm(hidden)
//! logits:     the vocabulary-parallel head, FP32
//! ```
//!
//! ## Rotary positions
//!
//! A Qwen3.5 checkpoint rotates the leading quarter of each attention head by interleaved M-RoPE
//! over three axes. A text batch places every token at the same position on all three axes, under
//! which interleaved M-RoPE is the default spectrum over the rotated channels, so a text batch
//! turns by that spectrum over its one position axis; a batch carrying image rows states three
//! axes ([`TokenBatch::rope_positions`]) and turns by the sectioned spectrum.
//!
//! ## Experts
//!
//! The routed experts of each layer are held in the slots an [`ExpertOrder`] states (`docs/admin.md`,
//! Expert redistribution); one published table maps every routed layer's logical experts to its
//! slots, so a redistribution changes the table's contents and the stacks, never a capture.

use std::collections::{BTreeMap, HashMap};

use nn::blocks::{AttentionKind, GatedDeltaNet, Keys, RopeFactors, SharedExpert, SharedExpertSpec};
use nn::capability::{CausalLm, KvView, PixelNorm, TokenBatch, Vision};
use nn::family::{
    CacheShape, Drafter, ExpertOrder, ExpertShape, Split, StateShape, TextModel, TextPlacement,
};
use nn::layers::{
    ExpertNames, ExpertSlots, Moe, MoeSpec, PagedAttention, ParallelLMHead, ResidualStream,
    RmsNorm, Rotary, Spectrum, StackedExperts, VocabParallelEmbedding,
};
use nn::quantization::Declaration;
use nn::weights::{Scope, WeightLoader};
use nn::{DType, Forward, Operand, Weight};

use crate::config::{Config, Mixer, QWEN3_MOE};
use crate::decoder::{self, DecoderStack, Mlp};
use crate::mtp::Mtp;
use crate::vision::VisionTower;
use crate::{Error, Result};
use processing::qwen_vl::ProcessorConfig;

/// The fence class of the whole decoder stack, at path `model`.
const TRANSFORMER: &str = "models.qwen3.transformer.Transformer";

/// The module names a checkpoint states a routed block under: `mlp.gate` routes, and `mlp.experts`
/// holds every expert either stacked (`gate_up_proj`, `down_proj`, as released Qwen3.5 BF16
/// checkpoints store them) or one module per expert (Qwen3-MoE, and every calibrated export).
const EXPERT_NAMES: ExpertNames<'static> = ExpertNames {
    router: "gate",
    experts: "experts",
    up: "up_proj",
    gate: "gate_proj",
    down: "down_proj",
    stacked: Some(StackedExperts {
        gate_up: "gate_up_proj",
        down: "down_proj",
    }),
};

/// A Qwen3 causal language model.
pub struct Qwen3ForCausalLm {
    config: Config,
    tensor: Split,
    dtype: DType,
    embedding: VocabParallelEmbedding,
    /// The rotation of a batch of one position axis.
    rotary: Rotary,
    /// The sectioned rotation a batch with image rows turns by, when the model takes images.
    mrope: Option<Rotary>,
    stack: DecoderStack,
    norm: RmsNorm,
    head: ParallelLMHead,
    /// Whether the checkpoint's experts were read as calibrated NVFP4.
    experts_encoded: bool,
    drafter: Option<Mtp>,
    vision: Option<VisionTower>,
}

/// The module path the language model's modules sit under.
fn language_model(config: &Config) -> &'static str {
    if config.class == QWEN3_MOE {
        "model"
    } else {
        "model.language_model"
    }
}

/// The spectrum a batch of one position axis turns by: the default spectrum over the rotated
/// channels of an interleaved M-RoPE (module documentation), or the configuration's own.
pub(crate) fn text_spectrum(rope: Spectrum) -> Spectrum {
    match rope {
        Spectrum::Sectioned {
            theta,
            dim,
            precision,
            ..
        } => Spectrum::Default {
            theta,
            dim,
            precision,
        },
        other => other,
    }
}

/// The path an [`ExpertOrder`] names the routed experts of decoder layer `layer` by.
pub fn expert_module(layer: u32) -> String {
    format!("layers.{layer}.mlp.experts")
}

impl Qwen3ForCausalLm {
    /// Load the model `config` and `declaration` describe at the ranks `placement` states, with
    /// the routed experts in the slots `order` states, the drafter when the deployment speculates,
    /// and the vision tower when `images` states the processor its images are sized by.
    ///
    /// # Errors
    ///
    /// A split the model does not divide, a drafter the checkpoint does not carry, a tower with
    /// DeepStack features (a causal LM's prefill adds none), or a module the loader refuses.
    pub fn load(
        config: &Config,
        declaration: &Declaration,
        placement: &TextPlacement,
        loader: &mut WeightLoader<'_>,
        order: &ExpertOrder,
        images: Option<(ProcessorConfig, PixelNorm)>,
    ) -> Result<Self> {
        let (tensor, experts) = (placement.tensor, placement.experts);
        config.check_splits(tensor, experts)?;
        loader.declare(declaration);
        let text = &config.text;
        let prefix = language_model(config);

        let embedding = VocabParallelEmbedding::load(
            &mut loader.scope(format!("{prefix}.embed_tokens")),
            text.vocab_size,
            text.hidden_size,
            tensor,
        )?;
        let dtype = embedding.weight().dtype();
        let head = if text.tie_word_embeddings {
            ParallelLMHead::tied(&embedding)?
        } else {
            ParallelLMHead::load(
                &mut loader.scope("lm_head"),
                text.vocab_size,
                text.hidden_size,
                tensor,
            )?
        };
        let norm = RmsNorm::load(
            &mut loader.scope(format!("{prefix}.norm")),
            text.hidden_size,
            text.rms_norm_eps,
            text.rms_norm_gain,
            dtype,
        )?;
        let rotary = decoder::rotary(
            &mut loader.scope(format!("{prefix}.rotary_emb")),
            text,
            text_spectrum(text.rope),
            tensor,
        )?;

        let routing: Vec<u32> = (0..text.num_hidden_layers)
            .filter(|layer| config.routes(*layer))
            .collect();
        let slots = expert_slots(config, &routing, order, loader.scope(prefix))?;
        let attention = AttentionKind::Paged {
            max_kv_len: text.max_position_embeddings,
            window: None,
        };
        let stack = DecoderStack::load(
            &mut loader.scope(prefix),
            config,
            0..text.num_hidden_layers,
            attention,
            tensor,
            dtype,
            &mut |layer, scope| match &slots {
                Some(slots) if config.routes(layer) => {
                    routed(scope, config, slots, layer, (tensor, experts))
                }
                _ => Mlp::dense(scope, text, tensor),
            },
        )?;

        let drafter = if placement.drafter {
            Some(Mtp::load(config, (tensor, experts), dtype, loader)?)
        } else {
            None
        };
        let (vision, mrope) = match images {
            None => (None, None),
            Some((processor, norm)) => {
                let vision = config.vision.as_ref().ok_or_else(|| {
                    Error::Config("the checkpoint states processor files and no tower".to_owned())
                })?;
                if !vision.deepstack_visual_indexes.is_empty() {
                    return Err(Error::Config(
                        "a causal language model's prefill adds no DeepStack features, but the \
                         tower states some"
                            .to_owned(),
                    ));
                }
                let mrope = decoder::rotary(
                    &mut loader.scope(format!("{prefix}.rotary_emb.sectioned")),
                    text,
                    text.rope,
                    tensor,
                )?;
                (
                    Some(VisionTower::load(loader, vision, processor, norm, dtype)?),
                    Some(mrope),
                )
            }
        };
        Ok(Self {
            experts_encoded: declaration.is_encoded() && !routing.is_empty(),
            config: config.clone(),
            tensor,
            dtype,
            embedding,
            rotary,
            mrope,
            stack,
            norm,
            head,
            drafter,
            vision,
        })
    }

    /// The checkpoint configuration this model was built for.
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// The keys decoder layer `index` reads: its paged key/value layer when it attends, its
    /// recurrent state layer when it is linear.
    fn keys<'k>(
        &self,
        batch: &'k TokenBatch,
        cache: &'k KvView<'k>,
        index: usize,
    ) -> Result<Keys<'k>> {
        let ordinal = self.config.ordinal(index);
        Ok(match self.config.mixers[index] {
            Mixer::Attention => Keys::Paged {
                batch,
                cache: cache.layer(0, ordinal)?,
            },
            Mixer::Linear(_) => Keys::Recurrent {
                batch,
                cache,
                layer: ordinal,
            },
        })
    }
}

/// The routed block of layer `layer` at `scope` (its `mlp` module): the experts in the slots
/// `slots` states and, when the configuration states one, the gated shared expert.
pub(crate) fn routed(
    scope: &mut Scope<'_, '_>,
    config: &Config,
    (orders, table, rows): &Slots,
    layer: u32,
    (tensor, experts): (Split, Split),
) -> Result<Mlp> {
    let text = &config.text;
    let moe = config
        .moe
        .as_ref()
        .ok_or_else(|| Error::Config(format!("layer {layer} routes in a dense model")))?;
    let row = rows
        .iter()
        .position(|routed| *routed == layer)
        .ok_or_else(|| Error::Config(format!("layer {layer} has no row of the expert table")))?;
    let routed = Moe::load(
        scope,
        MoeSpec {
            experts: moe.num_experts,
            top: moe.num_experts_per_tok,
            hidden: text.hidden_size,
            intermediate: moe.moe_intermediate_size,
            activation: text.hidden_act,
            renormalize: moe.norm_topk_prob,
        },
        EXPERT_NAMES,
        ExpertSlots {
            order: &orders[&layer],
            map: table.row(row as u64)?,
        },
        tensor,
        experts,
    )?;
    let shared = moe
        .shared_expert_intermediate_size
        .map(|intermediate| {
            SharedExpert::load(
                scope,
                SharedExpertSpec {
                    mlp: decoder::gated_mlp(text.hidden_size, intermediate, text.hidden_act),
                    module: "shared_expert",
                    gate: "shared_expert_gate",
                },
                tensor,
            )
        })
        .transpose()?;
    Ok(Mlp::Experts { routed, shared })
}

/// Each routed layer's logical experts in slot order, the published `[routed layers, experts]`
/// slot table a route reads its slot from, and the routed layers whose rows it holds, in row
/// order.
pub(crate) type Slots = (HashMap<u32, Vec<u32>>, Weight, Vec<u32>);

/// The slots of the routed layers `layers`, from `order` (the checkpoint's order for a layer it
/// does not name), and their table, published as `expert_slots` under `scope`; `None` without a
/// routed layer.
///
/// Every layer's row is one weight's row: a retained weight takes a whole allocation granule, so
/// one table per layer would hold a granule each for a few hundred bytes.
///
/// # Errors
///
/// An order naming a module that is no routed layer, or an order that is no permutation.
pub(crate) fn expert_slots(
    config: &Config,
    layers: &[u32],
    order: &ExpertOrder,
    mut scope: Scope<'_, '_>,
) -> Result<Option<Slots>> {
    let Some(moe) = &config.moe else {
        return Ok(None);
    };
    if layers.is_empty() {
        return Ok(None);
    }
    let modules: Vec<String> = layers.iter().copied().map(expert_module).collect();
    if let Some(stray) = order.keys().find(|module| !modules.contains(module)) {
        return Err(Error::Config(format!(
            "{stray} is not a routed expert layer of this model"
        )));
    }
    let experts = moe.num_experts;
    let mut orders = HashMap::with_capacity(layers.len());
    let mut table = Vec::with_capacity(layers.len() * experts as usize);
    for (&layer, module) in layers.iter().zip(&modules) {
        let slots = order
            .get(module)
            .cloned()
            .unwrap_or_else(|| (0..experts).collect());
        if slots.len() != experts as usize {
            return Err(Error::Config(format!(
                "the order of {module} names {} experts of {experts}",
                slots.len()
            )));
        }
        let map = Moe::slot_map(&slots)
            .map_err(|error| Error::Config(format!("the order of {module}: {error}")))?;
        // A slot is below the expert count, which the router's i32 ids hold.
        table.extend(map.iter().map(|&slot| slot as i32));
        orders.insert(layer, slots);
    }
    let weight = scope.constant_i32(
        "expert_slots",
        &[layers.len() as u64, u64::from(experts)],
        &table,
    )?;
    Ok(Some((orders, weight, layers.to_vec())))
}

impl CausalLm for Qwen3ForCausalLm {
    fn embed_input_ids(&self, f: &Forward, ids: Operand) -> nn::Result<Operand> {
        self.embedding.forward(f, ids)
    }

    /// The stack over the batch's embeddings `hidden`, which become the residual stream and are
    /// updated in place, and the final norm, which adds the last layer's output into the stream.
    fn forward(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: &KvView<'_>,
        hidden: Operand,
    ) -> nn::Result<Operand> {
        let tokens = batch.tokens();
        let width = u64::from(self.config.text.hidden_size);
        if hidden.shape.dims() != [tokens, width] || hidden.dtype != self.dtype {
            return Err(Error::Config(format!(
                "hidden is {} {:?}, {} [{tokens}, {width}] expected",
                hidden.dtype,
                hidden.shape.dims(),
                self.dtype
            ))
            .into());
        }
        Ok(f.fenced(TRANSFORMER, format_args!("model"), || {
            let (rotary, positions) = match (batch.rope_positions, &self.mrope) {
                (None, _) => (&self.rotary, batch.positions),
                (Some(coordinates), Some(mrope)) => (mrope, coordinates),
                (Some(_), None) => {
                    return Err(Error::Config(
                        "a batch states image coordinates for a model without vision input"
                            .to_owned(),
                    ));
                }
            };
            let rope = RopeFactors {
                rotary,
                factors: rotary.factors(f, positions)?,
            };
            if self.config.attention_layers() > 0 {
                // The workspace is the runtime's and every bucket's forward reuses it, so the
                // decode program's arrival counters are cleared once before the first layer.
                PagedAttention::zero_counters(f, cache.workspace)?;
            }
            let (last, residual) = self.stack.forward(
                f,
                hidden,
                rope,
                |index| self.keys(batch, cache, index),
                |_, _| Ok(()),
            )?;
            Ok(self
                .norm
                .forward(f, last, Some(ResidualStream::in_place(residual)))?)
        })?)
    }

    fn compute_logits(&self, f: &Forward, hidden: Operand) -> nn::Result<Operand> {
        self.head.forward(f, hidden)
    }

    fn join_expert_step(&self, f: &Forward) -> nn::Result<()> {
        for layer in self.stack.layers() {
            if let Mlp::Experts { routed, .. } = layer.mlp() {
                f.scoped(|f| routed.join_expert_step(f))?;
            }
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
        self.dtype
    }

    fn cache(&self) -> CacheShape {
        let text = &self.config.text;
        CacheShape {
            layers: self.config.attention_layers(),
            // The load checked that the split divides the heads.
            kv_heads: text.num_key_value_heads / self.tensor.size,
            head_dim: text.head_dim,
        }
    }

    fn state(&self) -> Option<StateShape> {
        let mut linear = self.config.mixers.iter().filter_map(|mixer| match mixer {
            Mixer::Linear(spec) => Some(*spec),
            Mixer::Attention => None,
        });
        let spec = linear.next()?;
        Some(StateShape {
            layers: 1 + linear.count() as u32,
            fields: GatedDeltaNet::state_fields(&spec, self.dtype),
        })
    }

    fn experts(&self) -> Option<ExpertShape> {
        let moe = self.config.moe.as_ref()?;
        Some(ExpertShape {
            experts: moe.num_experts,
            intermediate: moe.moe_intermediate_size,
            top: moe.num_experts_per_tok,
            encoded: self.experts_encoded,
        })
    }

    fn drafter(&self) -> Option<&dyn Drafter> {
        self.drafter.as_ref().map(|drafter| drafter as &dyn Drafter)
    }

    fn vision(&self) -> Option<&dyn Vision> {
        self.vision.as_ref().map(|tower| tower as &dyn Vision)
    }
}

/// The routed experts of every layer of `config` by module path, each with its expert count: what
/// an expert redistribution may order.
pub fn expert_layers(config: &Config) -> BTreeMap<String, u32> {
    let Some(moe) = &config.moe else {
        return BTreeMap::new();
    };
    (0..config.text.num_hidden_layers)
        .filter(|layer| config.routes(*layer))
        .map(|layer| (expert_module(layer), moe.num_experts))
        .collect()
}

impl std::fmt::Debug for Qwen3ForCausalLm {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Qwen3ForCausalLm")
            .field("class", &self.config.class)
            .field("layers", &self.stack.layers().len())
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
        let Spectrum::Default { dim, theta, .. } = text_spectrum(config.text.rope) else {
            panic!("a default spectrum");
        };
        assert_eq!((dim, theta), (64, 1e7));
        // Equal positions on every axis read the same frequencies under both spectra.
        assert_eq!(
            text_spectrum(config.text.rope)
                .inverse_frequencies()
                .expect("frequencies"),
            config.text.rope.inverse_frequencies().expect("frequencies"),
        );
    }
}
