//! The loading boundary of `Qwen3MoeForCausalLM`: the model's modules, read through the checkpoint's
//! own module paths.
//!
//! ## What the boundary owns
//!
//! The family reads `config.json` and the quantization declaration once, when it opens the
//! checkpoint; this module builds the model from them. Every module is read through a
//! [`nn::weights::Scope`] named as the checkpoint names it (`model.layers.<i>.self_attn.q_proj`),
//! and each `nn` layer reads its own tensors, shards itself by the deployment's split and chooses its
//! representation from the declaration. The rotary spectrum, which no checkpoint stores, is
//! published once for the whole stack by its [`Rotary`].
use std::collections::HashMap;

use nn::blocks::{GateUp, GatedMlp, GatedMlpSpec};
use nn::family::Split;
use nn::layers::{
    Bias, ExpertNames, ExpertSlots, Linear, Moe, MoeSpec, PagedAttention, PagedSpec, Parallel,
    ParallelLMHead, QkNorm, QkvBias, QkvLinear, QkvNames, QkvSpec, RmsNorm, Rotary, RotarySpec,
    Rounding, VocabParallelEmbedding,
};
use nn::quantization::Declaration;
use nn::weights::WeightLoader;

use crate::model::{AttentionLayer, DecoderLayer, LoadOptions, Mlp, Qwen3Moe};
use crate::{Config, Error, Partition, Result};

/// The module names a Qwen3-MoE checkpoint states a routed block under: `mlp.gate` routes, and
/// expert `e` of `mlp.experts` holds `gate_proj`, `up_proj` and `down_proj`.
const EXPERT_NAMES: ExpertNames<'static> = ExpertNames {
    router: "gate",
    experts: "experts",
    up: "up_proj",
    gate: "gate_proj",
    down: "down_proj",
};

/// Each routed layer's logical expert ids in physical slot order, by the layer's
/// [`expert_module`]. A layer this map does not name keeps the checkpoint's order.
pub type ExpertOrder = std::collections::BTreeMap<String, Vec<u32>>;

/// The path an [`ExpertOrder`] names the routed experts of decoder layer `layer` by:
/// `layers.<layer>.mlp.experts`, the model's module path without the checkpoint's `model.` prefix.
pub fn expert_module(layer: u32) -> String {
    format!("layers.{layer}.mlp.experts")
}

/// The [`expert_module`] of every routed layer of `config`, in layer order.
pub fn expert_modules(config: &Config) -> Vec<String> {
    (0..config.num_hidden_layers)
        .filter(|&layer| config.sparse(layer))
        .map(expert_module)
        .collect()
}

/// Load `checkpoint` in the representation `declaration` states about it, with the routed experts
/// in the slots `order` states (`docs/admin.md`, Expert redistribution).
///
/// `config` and `declaration` are what the family read from the checkpoint when it opened it;
/// nothing here reads a configuration file again. The loader resolves the declaration against the
/// shards once, and every layer is read by that one resolved statement: a projection the shards
/// store packed is read as NVFP4, and a packed weight reaching a layer with no encoded form is
/// refused by name.
///
/// The stacks hold the experts in slot order, and one table maps every layer's logical ids to its
/// slots. A deployment that never redistributes reads an identity table, so the graphs it captures
/// are the graphs any later order replays over: a redistribution changes what the table's address
/// and the stacks' addresses map, and no capture.
pub fn load(
    config: &Config,
    declaration: &Declaration,
    partition: Partition,
    loader: &mut WeightLoader<'_>,
    options: LoadOptions,
    order: &ExpertOrder,
) -> Result<Qwen3Moe> {
    partition.validate(config)?;
    loader.declare(declaration);
    // The model reports this option to execution as its experts' encoding, so the declaration
    // decides it once here.
    let experts_encoded = declaration.is_encoded() && config.has_sparse_layers();
    let options = LoadOptions {
        experts_encoded,
        ..options
    };
    let config = config.clone();
    let tensor = Split {
        size: partition.tensor_parallel,
        rank: partition.tensor_rank,
    };
    let experts_split = Split {
        size: partition.expert_parallel,
        rank: partition.expert_rank,
    };

    let head_dim = u64::from(config.head_dim);
    let q_heads = partition.local_attention_heads(&config);
    let kv_heads = partition.local_key_value_heads(&config);

    // The embedding and the head are vocabulary parallel over the tensor group; a tied head reads
    // the embedding's own matrix.
    let embedding = VocabParallelEmbedding::load(
        &mut loader.scope("model.embed_tokens"),
        config.vocab_size,
        config.hidden_size,
        tensor,
    )?;
    let dtype = embedding.weight().dtype();
    let head = if config.tie_word_embeddings {
        ParallelLMHead::tied(&embedding)?
    } else {
        ParallelLMHead::load(
            &mut loader.scope("lm_head"),
            config.vocab_size,
            config.hidden_size,
            tensor,
        )?
    };

    let norm = RmsNorm::load(
        &mut loader.scope("model.norm"),
        config.hidden_size,
        config.rms_norm_eps,
    )?;

    // One rotary for the stack: every layer turns its queries and keys by the same factors, and
    // the per-head norm before the rotation is fused into it at the model's epsilon.
    let rotary = Rotary::load(
        &mut loader.scope("model.rotary_emb"),
        config.rope,
        RotarySpec {
            head_dim: config.head_dim,
            heads: q_heads,
            kv_heads,
            qk_norm: QkNorm::Fused {
                epsilon: config.rms_norm_eps,
            },
        },
    )?;
    let score_scale = config.rope.score_scale(config.head_dim)?;

    let (slots, slot_table) = expert_slots(&config, order, loader)?;
    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
    for layer in 0..config.num_hidden_layers {
        let mut scope = loader.scope(format!("model.layers.{layer}"));
        let eps = config.rms_norm_eps;
        let input_norm =
            RmsNorm::load(&mut scope.scope("input_layernorm"), config.hidden_size, eps)?;
        let post_attention_norm = RmsNorm::load(
            &mut scope.scope("post_attention_layernorm"),
            config.hidden_size,
            eps,
        )?;

        let mut attention = scope.scope("self_attn");
        let bias = if config.attention_bias {
            (Bias::Present, QkvBias::Present)
        } else {
            (Bias::Absent, QkvBias::Absent)
        };
        let attention = AttentionLayer {
            qkv: QkvLinear::load(
                &mut attention,
                QkvSpec {
                    hidden: config.hidden_size,
                    heads: config.num_attention_heads,
                    kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                },
                QkvNames::Separate(["q_proj", "k_proj", "v_proj"]),
                bias.1,
                tensor,
            )?,
            o_proj: Linear::load(
                &mut attention.scope("o_proj"),
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
                bias.0,
                Parallel::Row(tensor),
            )?,
            q_norm: attention.scope("q_norm").tensor("weight", &[head_dim])?,
            k_norm: attention.scope("k_norm").tensor("weight", &[head_dim])?,
            attention: PagedAttention::new(PagedSpec {
                head_dim: config.head_dim,
                heads: q_heads,
                kv_heads,
                page_size: options.page_tokens,
                max_kv_len: config.max_position_embeddings,
                window: None,
                scale: score_scale,
            })?,
        };

        let mlp = if config.sparse(layer) {
            let map = match &slot_table {
                Some((table, layers)) => {
                    let row = layers.iter().position(|routed| *routed == layer);
                    let row = row.ok_or_else(|| {
                        Error::Config(format!("layer {layer} has no row of the expert table"))
                    })?;
                    table.row(as_u64(row, "an expert table row")?)?
                }
                None => {
                    return Err(Error::Config(format!(
                        "layer {layer} routes and no routed layer has a table"
                    )));
                }
            };
            Mlp::Experts(Moe::load(
                &mut scope.scope("mlp"),
                MoeSpec {
                    experts: config.num_experts,
                    top: config.num_experts_per_tok,
                    hidden: config.hidden_size,
                    intermediate: config.moe_intermediate_size,
                    activation: config.hidden_act,
                    renormalize: config.norm_topk_prob,
                },
                EXPERT_NAMES,
                ExpertSlots {
                    order: &slots[&layer],
                    map,
                },
                tensor,
                experts_split,
            )?)
        } else {
            // SwiGLU over the two projections' outputs, in the swiglu kernel's value-product
            // order.
            Mlp::Dense(Box::new(GatedMlp::load(
                &mut scope.scope("mlp"),
                GatedMlpSpec {
                    hidden: config.hidden_size,
                    intermediate: config.intermediate_size,
                    activation: ops::activation::Activation::Silu,
                    rounding: Rounding::ValueProduct,
                    gate_up: GateUp::Split {
                        gate: "gate_proj",
                        up: "up_proj",
                    },
                    down: "down_proj",
                },
                tensor,
            )?))
        };

        layers.push(DecoderLayer {
            input_norm,
            post_attention_norm,
            attention,
            mlp,
        });
    }

    Qwen3Moe::new(
        config, partition, dtype, embedding, rotary, layers, norm, head, options,
    )
}

/// A count as an element count.
fn as_u64(count: impl TryInto<u64>, what: &str) -> Result<u64> {
    count
        .try_into()
        .map_err(|_| Error::Config(format!("{what} does not fit an element count")))
}

/// The slot of each expert of the routed layer `module`, from `order`, the expert of each slot.
///
/// # Errors
///
/// When `order` is no permutation of `experts` experts.
fn slot_row(module: &str, order: &[u32], experts: usize) -> Result<Vec<u32>> {
    if order.len() != experts {
        return Err(Error::Config(format!(
            "the order of {module} names {} experts of {experts}",
            order.len()
        )));
    }
    Moe::slot_map(order).map_err(|error| Error::Config(format!("the order of {module}: {error}")))
}

/// Each routed layer's logical experts in slot order, keyed by layer index, and the published slot
/// table with the routed layers whose rows it holds, in row order (`None` without a routed layer).
type SlotTable = (HashMap<u32, Vec<u32>>, Option<(nn::Weight, Vec<u32>)>);

/// The experts of each routed layer in slot order, checked, with the table a route reads its slot
/// from: `[routed layers, experts]` i32, row by row in layer order.
///
/// Every layer's row is one weight's row, because a retained weight takes a whole allocation
/// granule: one table per layer would hold 2 MiB each for 512 bytes of content.
fn expert_slots(
    config: &Config,
    order: &ExpertOrder,
    loader: &mut WeightLoader<'_>,
) -> Result<SlotTable> {
    let experts = config.num_experts;
    let layers: Vec<u32> = (0..config.num_hidden_layers)
        .filter(|&layer| config.sparse(layer))
        .collect();
    let modules = expert_modules(config);
    for module in order.keys() {
        if !modules.contains(module) {
            return Err(Error::Config(format!(
                "{module} is not a routed expert layer of this model"
            )));
        }
    }
    let mut slots = HashMap::with_capacity(layers.len());
    let mut table = Vec::with_capacity(layers.len() * experts as usize);
    for &layer in &layers {
        let module = expert_module(layer);
        let order: Vec<u32> = match order.get(&module) {
            Some(order) => order.clone(),
            None => (0..experts).collect(),
        };
        // A slot is below the vocabulary, which `Config` holds in u32 and the router's I32 ids.
        table.extend(
            slot_row(&module, &order, experts as usize)?
                .iter()
                .map(|&slot| slot as i32),
        );
        slots.insert(layer, order);
    }
    if layers.is_empty() {
        return Ok((slots, None));
    }
    let weight = loader.scope("model").constant_i32(
        "expert_slots",
        &[
            as_u64(layers.len(), "the routed layers")?,
            u64::from(experts),
        ],
        &table,
    )?;
    Ok((slots, Some((weight, layers))))
}
#[cfg(test)]
mod tests {

    #[test]
    fn a_slot_row_inverts_the_order_it_is_read_from() {
        // Slot 0 holds expert 2, so a route to expert 2 reads slot 0.
        assert_eq!(
            super::slot_row("layers.0.mlp.experts", &[2, 0, 1], 3).expect("a permutation"),
            vec![1, 2, 0]
        );
    }

    #[test]
    fn a_slot_row_refuses_an_order_that_is_no_permutation() {
        for order in [&[0, 0, 1][..], &[0, 1][..], &[0, 1, 3][..]] {
            let error = super::slot_row("layers.0.mlp.experts", order, 3)
                .expect_err("no permutation of three experts");
            assert!(
                error.to_string().contains("layers.0.mlp.experts"),
                "{error}"
            );
        }
    }
}
