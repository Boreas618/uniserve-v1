//! The loading boundary of `Qwen3MoeForCausalLM`: the model's modules, read through the checkpoint's
//! own module paths, and the rotary frequency table.
//!
//! ## What the boundary owns
//!
//! The family reads `config.json` and the quantization declaration once, when it opens the
//! checkpoint; this module builds the model from them. Every module is read through a
//! [`nn::weights::Scope`] named as the checkpoint names it (`model.layers.<i>.self_attn.q_proj`),
//! and each `nn` layer reads its own tensors, shards itself by the deployment's split and chooses its
//! representation from the declaration. The rotary frequency table, which no checkpoint stores, is
//! computed here once and published to retained storage as a constant.
use std::collections::HashMap;
use std::f64::consts::PI;

use nn::family::Split;
use nn::layers::{
    Attention, Bias, ExpertNames, ExpertSlots, GateBias, GatedActivation, Linear, Moe, MoeSpec,
    Parallel, ParallelLMHead, QkvBias, QkvLinear, QkvNames, QkvSpec, RmsNorm, Rotary, Rounding,
    VocabParallelEmbedding,
};
use nn::quantization::Declaration;
use nn::weights::{Scope, WeightLoader};
use ops::rope::Rotation;

use crate::model::{AttentionLayer, DecoderLayer, DenseMlp, LoadOptions, Mlp, Qwen3Moe};
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

    let hidden = u64::from(config.hidden_size);
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

    let norm = RmsNorm::load(&mut loader.scope("model.norm"), hidden, config.rms_norm_eps)?;

    // The rotary recipe: one frequency table and the score multiplier the recipe adds.
    let (frequencies, score_scale) = rotary_recipe(&config)?;
    let table =
        loader
            .scope("model.rotary_emb")
            .constant_f32("inv_freq", &[head_dim / 2], &frequencies)?;

    let (slots, slot_table) = expert_slots(&config, order, loader)?;
    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
    for layer in 0..config.num_hidden_layers {
        let mut scope = loader.scope(format!("model.layers.{layer}"));
        let eps = config.rms_norm_eps;
        let input_norm = RmsNorm::load(&mut scope.scope("input_layernorm"), hidden, eps)?;
        let post_attention_norm =
            RmsNorm::load(&mut scope.scope("post_attention_layernorm"), hidden, eps)?;

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
            rotary: Rotary::new(
                table,
                config.head_dim,
                q_heads,
                kv_heads,
                config.rms_norm_eps,
                Rotation::Half,
                1.0,
            )?,
            attention: Attention::paged(
                config.head_dim,
                q_heads,
                kv_heads,
                options.page_tokens,
                config.max_position_embeddings,
                None,
                score_scale,
            )?,
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
            Mlp::Dense(Box::new(dense_block(
                &mut scope.scope("mlp"),
                tensor,
                &config,
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
        config, partition, dtype, embedding, layers, norm, head, options,
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
/// One dense layer's feed-forward block, the module at `scope`: the gate and up projections split
/// by output channels, the down projection by its contraction, which its forward reduces.
fn dense_block(scope: &mut Scope<'_, '_>, tensor: Split, config: &Config) -> Result<DenseMlp> {
    let (hidden, intermediate) = (config.hidden_size, config.intermediate_size);
    let mut column = |module: &str| {
        Linear::load(
            &mut scope.scope(module),
            hidden,
            intermediate,
            Bias::Absent,
            Parallel::Column(tensor),
        )
    };
    let gate_proj = column("gate_proj")?;
    let up_proj = column("up_proj")?;
    let down_proj = Linear::load(
        &mut scope.scope("down_proj"),
        intermediate,
        hidden,
        Bias::Absent,
        Parallel::Row(tensor),
    )?;
    // SwiGLU over the two projections' outputs, in the swiglu kernel's value-product order.
    let activation = GatedActivation::new(
        ops::activation::Activation::Silu,
        Rounding::ValueProduct,
        GateBias::None,
    )?;
    Ok(DenseMlp {
        gate_proj,
        up_proj,
        down_proj,
        activation,
    })
}

/// The rotary frequency table and the multiplier the recipe applies to the attention scores.
///
/// The table is `[head_dim / 2]` FP32 inverse frequencies: `1 / theta^(2i / head_dim)` for the
/// plain recipe, and transformers' YaRN interpolation for a yarn checkpoint
/// (`_compute_yarn_parameters`). The correction range `beta_fast`/`beta_slow` splits the pairs: the
/// pairs below it keep their extrapolated frequency and the ones above it are interpolated by the
/// extension factor, with a linear ramp between — the high frequencies (a pair at index 0 has
/// period 2π tokens) are the ones that must not move.
///
/// The extension factor is `max_position_embeddings / original_max_position_embeddings` when the
/// checkpoint declares the length it was fitted to, which is the reference's own rule: the declared
/// `factor` is only consulted by recipes that do not carry that field, so for a yarn checkpoint
/// that carries it the ratio is what the table is built from. The multiplier is 1 for the plain
/// recipe and the recipe's attention factor squared for yarn: the reference multiplies the cosine
/// and sine factors by it, so a product of two rotated vectors carries the
/// square.
pub fn rotary_recipe(config: &Config) -> Result<(Vec<f32>, f32)> {
    let dim = config.head_dim as usize;
    if dim < 2 || !dim.is_multiple_of(2) {
        return Err(Error::Config(format!(
            "a rotary head of {dim} elements has no rotation pair"
        )));
    }
    let half = dim / 2;
    let base = f64::from(config.rope_theta);
    if !(base.is_finite() && base > 1.0) {
        return Err(Error::Config(format!(
            "rope_theta {base} is not a valid frequency base"
        )));
    }
    let extrapolation: Vec<f64> = (0..half)
        .map(|index| 1.0 / base.powf(2.0 * index as f64 / dim as f64))
        .collect();

    // The score scale `ops` takes is the **softmax scale applied to `q · k`** (`ops::attention::
    // DenseArgs::scale`), not a multiplier on top of a default: the reference divides by
    // `sqrt(head_dim)` and a recipe multiplies that. Returning the recipe alone set the exponent
    // 11.3 times too large for this model (1.0 against 1/sqrt(128)) — a sharper distribution that
    // still reads as a plausible answer.
    let denominator = (dim as f64).sqrt();
    let Some(yarn) = &config.rope_scaling else {
        return Ok((
            extrapolation.iter().map(|&value| value as f32).collect(),
            (1.0 / denominator) as f32,
        ));
    };

    // The length the recipe was fitted to, which is both the window the correction range is
    // measured over and the denominator of the extension factor the reference derives.
    let fitted = yarn.original_max_position_embeddings as f64;
    if !(fitted.is_finite() && fitted > 0.0) {
        return Err(Error::Config(format!(
            "an original context length of {fitted} is not a window"
        )));
    }
    let factor = config.max_position_embeddings as f64 / fitted;
    if !(factor.is_finite() && factor > 0.0) {
        return Err(Error::Config(format!(
            "an extension factor of {factor} is not a positive extension"
        )));
    }
    let (low, high) = correction_range(
        f64::from(yarn.beta_fast),
        f64::from(yarn.beta_slow),
        dim as f64,
        base,
        fitted,
    );
    let interpolation: Vec<f64> = extrapolation.iter().map(|value| value / factor).collect();
    let frequencies: Vec<f32> = (0..half)
        .map(|index| {
            // The ramp is 1 above the correction range, where the pairs are interpolated, and 0
            // below it, where they keep their extrapolated frequency.
            let ramp = if low == high {
                f64::from(index as f64 >= high)
            } else {
                ((index as f64 - low) / (high - low)).clamp(0.0, 1.0)
            };
            (interpolation[index] * ramp + extrapolation[index] * (1.0 - ramp)) as f32
        })
        .collect();

    let mscale = f64::from(yarn.mscale.unwrap_or(1.0));
    let attention = match (yarn.attention_factor, yarn.mscale, yarn.mscale_all_dim) {
        (Some(declared), ..) => f64::from(declared),
        (None, Some(_), Some(all)) => {
            yarn_mscale(factor, mscale) / yarn_mscale(factor, f64::from(all))
        }
        (None, ..) => yarn_mscale(factor, mscale),
    };
    let attention = attention as f32;
    Ok((
        frequencies,
        (f64::from(attention * attention) / denominator) as f32,
    ))
}

/// transformers' `_yarn_get_mscale`.
fn yarn_mscale(scale: f64, mscale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

/// transformers' `_find_correction_range` over the rotary width.
fn correction_range(
    beta_fast: f64,
    beta_slow: f64,
    dim: f64,
    base: f64,
    max_position: f64,
) -> (f64, f64) {
    let dimension = |rotations: f64| -> f64 {
        (dim * (max_position / (rotations * 2.0 * PI)).ln()) / (2.0 * base.ln())
    };
    let low = dimension(beta_fast).floor().max(0.0);
    let high = dimension(beta_slow).ceil().min(dim - 1.0);
    (low, high)
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

    use super::*;
    use crate::{Config, YaRn};

    fn config() -> Config {
        Config {
            vocab_size: 151936,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 2,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            head_dim: 128,
            hidden_act: ops::activation::Activation::Silu,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_scaling: None,
            max_position_embeddings: 40960,
            attention_bias: false,
            tie_word_embeddings: false,
            num_experts: 128,
            num_experts_per_tok: 8,
            moe_intermediate_size: 768,
            norm_topk_prob: true,
            decoder_sparse_step: 1,
            mlp_only_layers: Vec::new(),
        }
    }

    #[test]
    fn the_plain_recipe_is_the_power_series() {
        let config = config();
        let (frequencies, scale) = rotary_recipe(&config).unwrap();
        assert_eq!(frequencies.len(), 64);
        assert_eq!(frequencies[0], 1.0);
        // 1 / theta^(2 / head_dim): the second pair of the head, in the FP64 arithmetic the
        // table is computed in before it is narrowed.
        let second = (1.0 / 1_000_000_f64.powf(2.0 / 128.0)) as f32;
        assert!((frequencies[1] - second).abs() < 1e-9);
        // `1/sqrt(head_dim)`, because that is what the attention kernel is handed: the reference's
        // `scaling` is `head_dim ** -0.5` and `ops`' field is documented as the softmax scale.
        assert!((scale - (1.0 / 128.0_f32.sqrt())).abs() < 1e-9);
        assert!(frequencies.windows(2).all(|pair| pair[0] > pair[1]));
    }

    #[test]
    fn the_yarn_recipe_interpolates_the_low_frequencies_and_keeps_the_high_ones() {
        let mut config = config();
        config.rope_scaling = Some(YaRn {
            factor: 4.0,
            original_max_position_embeddings: 4096,
            attention_factor: None,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: None,
            mscale_all_dim: None,
            truncate: false,
        });
        let (frequencies, scale) = rotary_recipe(&config).unwrap();
        let plain = rotary_recipe(&config_without_yarn()).unwrap().0;
        // The extension factor is the extended length over the length the recipe was fitted to:
        // 40960 / 4096 = 10, which is what the pairs above the correction range are divided by.
        // The fastest pair (a period of 2π tokens) must not move: interpolating it is what would
        // make the table wrong, and it is the direction a swapped ramp gets wrong.
        assert_eq!(frequencies[0], plain[0]);
        assert!((frequencies[63] - plain[63] / 10.0).abs() < 1e-9);
        // Every pair is between the extrapolated and the interpolated value, and the table falls.
        assert!(frequencies.windows(2).all(|pair| pair[0] > pair[1]));
        // attention_factor = 0.1 · ln(10) + 1, squared into the score multiplier.
        let expected = (0.1 * 10.0_f32.ln() + 1.0).powi(2) / 128.0_f32.sqrt();
        assert!((scale - expected).abs() < 1e-6);
    }

    fn config_without_yarn() -> Config {
        config()
    }
}
