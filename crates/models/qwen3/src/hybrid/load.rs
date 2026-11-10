//! The loading boundary of the Qwen3.5-MoE language model: its modules, read through the
//! checkpoint's own module paths (`model.language_model.layers.<i>.linear_attn.in_proj_qkv`).
//!
//! The family reads `config.json` and the quantization declaration once, when it opens the
//! checkpoint; this module builds the model from them. The multi-token-prediction layers (`mtp.*`)
//! load as the model's drafter when the deployment speculates. The vision tower (`model.visual.*`)
//! is read when the family found the checkpoint's processor files and the tower's weights
//! ([`super::vision`]).

use nn::blocks::{AttentionKind, GateUp, GatedMlpSpec, SharedExpert, SharedExpertSpec};
use nn::family::{ExpertOrder, Split};
use nn::layers::{
    Bias, ExpertNames, ExpertSlots, Linear, Moe, MoeSpec, Parallel, ParallelLMHead, QkNorm,
    RmsNorm, Rotary, RotarySpec, Rounding, StackedExperts, VocabParallelEmbedding,
};
use nn::quantization::Declaration;
use nn::weights::{Scope, WeightLoader};

use super::mtp::{DRAFTER_COMPONENTS, Mtp};
use super::vision::{HybridVision, ImageProcessor};
use super::{COMPONENTS, ImageInput, Qwen35Moe, delta_net_spec, text_spectrum};
use crate::config::{HybridConfig, Mixer, TextConfig};
use crate::decoder::{DecoderStack, LayerForm, MixerSpec, Mlp};
use crate::moe::load::{SlotTable, expert_module, expert_slots, slot_map};
use crate::{Error, Result};

/// The module path the drafter's modules sit under.
const DRAFTER: &str = "mtp";

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

/// The attention, rounding and fence components a stack of the checkpoint computes its layers
/// with, the model's or its drafter's.
fn layer_form(text: &TextConfig, components: crate::decoder::Components) -> LayerForm {
    LayerForm {
        attention: AttentionKind::Paged {
            max_kv_len: text.max_position_embeddings,
            window: None,
        },
        // The shared expert's SiLU and product are one FP32 expression rounded once, as the
        // references' fused `SiluAndMul` evaluates them.
        rounding: Rounding::Fused,
        components,
    }
}

/// The routed block of layer `layer` at `scope` (its `mlp` module): the experts in the slots
/// `slots` states, read through the slot table's row for the layer, and the gated shared expert.
fn routed_block(
    scope: &mut Scope<'_, '_>,
    config: &HybridConfig,
    form: &LayerForm,
    (tensor, experts): (Split, Split),
    (slots, slot_table): &SlotTable,
    layer: u32,
) -> Result<Option<Mlp>> {
    let text = &config.text;
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
            map: slot_map(slot_table, layer)?,
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
}

/// Load the drafter at `mtp`: the multi-token-prediction layers' norms and input projection, their
/// full-attention decoder layers (every one routed) and their output norm.
///
/// The drafter's experts keep the checkpoint's order: an expert redistribution orders the model's
/// layers ([`TextCheckpoint::expert_layers`](nn::family::TextCheckpoint::expert_layers)), and it
/// names none of the drafter's.
///
/// # Errors
///
/// A checkpoint whose drafter embeds with a table of its own, which this drafter does not read, or
/// a module the loader refuses.
fn load_drafter(
    config: &HybridConfig,
    serving: &TextConfig,
    splits: (Split, Split),
    loader: &mut WeightLoader<'_>,
) -> Result<Mtp> {
    let mtp = &config.mtp;
    if mtp.num_hidden_layers == 0 {
        return Err(Error::Config(
            "the deployment speculates and the checkpoint carries no multi-token-prediction layer              (`mtp_num_hidden_layers` is 0)"
                .to_owned(),
        ));
    }
    if mtp.use_dedicated_embeddings {
        return Err(Error::Config(
            "the checkpoint's drafter embeds with a table of its own              (`mtp_use_dedicated_embeddings`), which this drafter does not read"
                .to_owned(),
        ));
    }
    let text = &config.text;
    let (tensor, _) = splits;
    let mut scope = loader.scope(DRAFTER);
    let norm = |scope: &mut Scope<'_, '_>, module: &str| {
        RmsNorm::load(
            &mut scope.scope(module),
            text.hidden_size,
            text.rms_norm_eps,
            text.rms_norm_gain,
        )
    };
    let pre_fc_norm_embedding = norm(&mut scope, "pre_fc_norm_embedding")?;
    let pre_fc_norm_hidden = norm(&mut scope, "pre_fc_norm_hidden")?;
    // `fc` contracts the concatenation `[embedding, hidden]` back to the hidden width.
    let fc = Linear::load(
        &mut scope.scope("fc"),
        2 * text.hidden_size,
        text.hidden_size,
        Bias::Absent,
        Parallel::None,
    )?;
    let output_norm = norm(&mut scope, "norm")?;
    let layers: Vec<u32> = (0..mtp.num_hidden_layers).collect();
    let table = expert_slots(
        config.num_experts,
        &layers,
        &ExpertOrder::new(),
        loader.scope(DRAFTER),
    )?;
    let form = layer_form(text, DRAFTER_COMPONENTS);
    let mut drafter_text = serving.clone();
    // The drafter's stack is its own layers, which the stack's range check reads against.
    drafter_text.num_hidden_layers = mtp.num_hidden_layers;
    let stack = DecoderStack::load(
        &mut loader.scope(DRAFTER),
        &drafter_text,
        form,
        tensor,
        0..mtp.num_hidden_layers,
        |_| MixerSpec::Attention,
        |layer, scope| routed_block(scope, config, &form, splits, &table, layer),
    )?;
    Ok(Mtp::new(
        text.hidden_size,
        mtp.num_hidden_layers,
        text.num_key_value_heads / tensor.size,
        text.head_dim,
        pre_fc_norm_embedding,
        pre_fc_norm_hidden,
        fc,
        stack,
        output_norm,
    ))
}

/// Load the checkpoint `config` and `declaration` describe at the rank `tensor` and `experts`
/// place, with the routed experts in the slots `order` states, its drafter when `drafter` says the
/// deployment speculates with it, and its vision tower when `images` states the processor its
/// images are sized by.
///
/// # Errors
///
/// A split the model does not divide, a drafter the checkpoint does not carry, or a module the
/// loader refuses.
pub fn load(
    config: &HybridConfig,
    declaration: &Declaration,
    (tensor, experts): (Split, Split),
    drafter: bool,
    loader: &mut WeightLoader<'_>,
    order: &ExpertOrder,
    images: Option<ImageProcessor>,
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
    let table = expert_slots(
        config.num_experts,
        &layers,
        order,
        loader.scope(LANGUAGE_MODEL),
    )?;
    let form = layer_form(text, COMPONENTS);
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
        |layer, scope| routed_block(scope, config, &form, (tensor, experts), &table, layer),
    )?;
    let drafter = drafter
        .then(|| load_drafter(config, &serving, (tensor, experts), loader))
        .transpose()?;

    let vision = match images {
        None => None,
        Some(processor) => {
            // The interleaved M-RoPE of the checkpoint, published beside the text spectrum the
            // stack turns by; its factors feed the stack's rotation, whose rotated width is the
            // same.
            let (heads, kv_heads) = text.local_heads(tensor)?;
            let rotary = Rotary::load(
                &mut loader.scope(format!("{LANGUAGE_MODEL}.rotary_emb.sectioned")),
                text.rope,
                RotarySpec {
                    head_dim: text.head_dim,
                    heads,
                    kv_heads,
                    qk_norm: QkNorm::Fused {
                        epsilon: text.rms_norm_eps,
                        gain: text.rms_norm_gain,
                    },
                },
            )?;
            Some(ImageInput {
                tower: HybridVision::load(loader, &config.vision, processor, dtype)?,
                rotary,
            })
        }
    };

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
        drafter,
        vision,
    })
}
