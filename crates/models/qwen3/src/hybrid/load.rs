//! The loading boundary of the Qwen3.5-MoE language model: its modules, read through the
//! checkpoint's own module paths (`model.language_model.layers.<i>.linear_attn.in_proj_qkv`).
//!
//! The family reads `config.json` and the quantization declaration once, when it opens the
//! checkpoint; this module builds the model from them. The multi-token-prediction layers (`mtp.*`)
//! and the vision tower (`model.visual.*`) are not read here: the text model serves without them.

use nn::blocks::{AttentionKind, GateUp, GatedMlpSpec, SharedExpert, SharedExpertSpec};
use nn::family::{ExpertOrder, Split};
use nn::layers::{
    Bias, ExpertNames, ExpertSlots, Moe, MoeSpec, ParallelLMHead, RmsNorm, Rounding,
    StackedExperts, VocabParallelEmbedding,
};
use nn::quantization::Declaration;
use nn::weights::WeightLoader;

use super::{COMPONENTS, Qwen35Moe, delta_net_spec, text_spectrum};
use crate::Result;
use crate::config::{HybridConfig, Mixer};
use crate::decoder::{DecoderStack, LayerForm, MixerSpec, Mlp};
use crate::moe::load::{expert_module, expert_slots, slot_map};

/// The module path the language model's modules sit under.
const LANGUAGE_MODEL: &str = "model.language_model";

/// The module names a Qwen3.5-MoE checkpoint states a routed block under: `mlp.gate` routes, and
/// `mlp.experts` holds every expert either stacked (`gate_up_proj`, `down_proj`, as the released
/// BF16 checkpoints store them) or one module per expert (as a calibrated export writes them).
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

/// The [`expert_module`] of every layer of `config`, in layer order: every layer routes.
pub fn expert_modules(config: &HybridConfig) -> Vec<String> {
    (0..config.text.num_hidden_layers)
        .map(expert_module)
        .collect()
}

/// Load the checkpoint `config` and `declaration` describe at the rank `tensor` and `experts`
/// place, with the routed experts in the slots `order` states.
///
/// # Errors
///
/// A split the model does not divide, or a module the loader refuses.
pub fn load(
    config: &HybridConfig,
    declaration: &Declaration,
    tensor: Split,
    experts: Split,
    loader: &mut WeightLoader<'_>,
    order: &ExpertOrder,
) -> Result<Qwen35Moe> {
    config.check_splits(tensor, experts)?;
    loader.declare(declaration);
    let experts_encoded = declaration.is_encoded();
    let text = &config.text;

    let embedding = VocabParallelEmbedding::load(
        &mut loader.scope(format!("{LANGUAGE_MODEL}.embed_tokens")),
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
        &mut loader.scope(format!("{LANGUAGE_MODEL}.norm")),
        text.hidden_size,
        text.rms_norm_eps,
        text.rms_norm_gain,
    )?;

    let layers: Vec<u32> = (0..text.num_hidden_layers).collect();
    let (slots, slot_table) = expert_slots(
        config.num_experts,
        &layers,
        order,
        loader.scope(LANGUAGE_MODEL),
    )?;
    let form = LayerForm {
        attention: AttentionKind::Paged {
            max_kv_len: text.max_position_embeddings,
            window: None,
        },
        // The shared expert's SiLU and product are one FP32 expression rounded once, as the
        // references' fused `SiluAndMul` evaluates them.
        rounding: Rounding::Fused,
        components: COMPONENTS,
    };
    // A text request rotates by the default spectrum over the rotated channels (`super`'s module
    // documentation); every other field is the checkpoint's.
    let mut serving = text.clone();
    serving.rope = text_spectrum(text.rope)?;
    let delta = delta_net_spec(config);
    let stack = DecoderStack::load(
        &mut loader.scope(LANGUAGE_MODEL),
        &serving,
        form,
        tensor,
        0..text.num_hidden_layers,
        |layer| match config.mixers[layer as usize] {
            Mixer::Attention => MixerSpec::Attention,
            Mixer::Linear => MixerSpec::Linear(delta),
        },
        |layer, scope| {
            let routed = Moe::load(
                scope,
                MoeSpec {
                    experts: config.num_experts,
                    top: config.num_experts_per_tok,
                    hidden: text.hidden_size,
                    intermediate: config.moe_intermediate_size,
                    activation: text.hidden_act,
                    // The router's top-k weights always renormalize to one.
                    renormalize: true,
                },
                EXPERT_NAMES,
                ExpertSlots {
                    order: &slots[&layer],
                    map: slot_map(&slot_table, layer)?,
                },
                tensor,
                experts,
            )?;
            let shared = SharedExpert::load(
                scope,
                SharedExpertSpec {
                    mlp: GatedMlpSpec {
                        hidden: text.hidden_size,
                        intermediate: config.shared_expert_intermediate_size,
                        activation: text.hidden_act,
                        rounding: form.rounding,
                        gate_up: GateUp::Split {
                            gate: "gate_proj",
                            up: "up_proj",
                        },
                        down: "down_proj",
                        bias: Bias::Absent,
                    },
                    module: "shared_expert",
                    gate: "shared_expert_gate",
                },
                tensor,
            )?;
            Ok(Some(Mlp::Experts {
                routed,
                shared: Some(shared),
            }))
        },
    )?;

    let ordinals = (0..text.num_hidden_layers)
        .map(|layer| config.ordinal(layer) as usize)
        .collect();
    Ok(Qwen35Moe {
        config: config.clone(),
        tensor,
        dtype,
        embedding,
        stack,
        norm,
        head,
        ordinals,
        experts_encoded,
    })
}
