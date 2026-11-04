//! The loading boundary of `Qwen3MoeForCausalLM`: checkpoint metadata, the rotary frequency table,
//! and one [`WeightRequest`] per parameter of the model.
//!
//! ## What the boundary owns
//!
//! `config.json` is read and normalized here, once, into [`Config`]; the checkpoint's tensor names
//! are resolved to model paths by [`crate::weights`]; the rotary frequency table, which no
//! checkpoint stores, is computed here and published to retained storage as a constant
//! (specs/worker.md 9: "YaRN 的 inv_freq 在加载边界由 Rust 按同样的公式算一次"). Everything below
//! this module consumes typed fields and borrowed views: no layer parses a dictionary or applies a
//! default.
//!
//! ## Partitioning
//!
//! A region of a checkpoint tensor is contiguous and row-major, so a shard along the *first* axis
//! is `start = rank × rows × width`; the vocabulary, the query/key/value projections and the
//! router are sharded that way. A shard along the *last* axis (`o_proj`, a dense `down_proj`) is
//! strided, which [`nn::weights::WeightSource`] does not yet express; a deployment that needs one
//! is refused here with the reason instead of silently loading a wrong region. The BF16 vertical
//! slice is one rank, so every shard below is the whole tensor.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::Path;

use nn::layers::{Attention, Embedding, Linear, Moe, ParallelLMHead, RmsNorm, Rotary, SwiGlu};
use nn::weights::{Weight, WeightLayout, WeightLoader, WeightRequest};
use nn::{DType, Shape};
use ops::gemm::WeightOrder;
use ops::moe::grouped::Activation as MoeActivation;
use ops::rope::Rotation;

use crate::model::{AttentionLayer, DecoderLayer, DenseMlp, LoadOptions, Mlp, Qwen3Moe};
use crate::{Activation, Config, Error, Partition, Result};

/// Load a `Qwen3MoeForCausalLM` checkpoint from `root`.
///
/// `root` holds `config.json` and the safetensors shards (an index or a directory of shards); the
/// `loader` reads the tensors and publishes the converted weights to retained storage.
pub fn load(
    root: &Path,
    partition: Partition,
    loader: &mut WeightLoader<'_>,
    options: LoadOptions,
) -> Result<Qwen3Moe> {
    let config = Config::read(root)?;
    partition.validate(&config)?;
    let sources = sources(&config);

    let hidden = as_u64(config.hidden_size, "hidden_size")?;
    let head_dim = as_u64(config.head_dim, "head_dim")?;
    let q_heads = partition.local_attention_heads(&config);
    let kv_heads = partition.local_key_value_heads(&config);
    let query = as_u64(q_heads * config.head_dim, "the query width")?;
    let key_value = as_u64(kv_heads * config.head_dim, "the key/value width")?;

    // The vocabulary shard: rows are contiguous in a `[vocabulary, hidden]` table.
    let (first, rows) = partition.vocabulary_shard(config.vocab_size);
    let padded = padded_vocabulary(rows);
    if padded != rows {
        return Err(Error::Unsupported(format!(
            "a vocabulary shard of {rows} rows needs {padded} padded rows; the checkpoint's rows \
             are contiguous, so a rank whose shard is not its padded width is not loadable yet"
        )));
    }
    let embedding_weight = region(
        loader,
        &sources,
        "embedding.weight",
        as_u64(first * config.hidden_size, "the embedding shard offset")?,
        Shape::new(&[as_u64(rows, "the vocabulary shard")?, hidden]),
    )?;
    let embedding = Embedding::new(
        embedding_weight,
        as_u32(first, "the vocabulary shard")?,
        as_u32(rows, "the vocabulary shard")?,
        as_u32(config.vocab_size, "vocab_size")?,
    )
    .map_err(Error::from)?;
    let dtype = embedding_weight.view().dtype;

    let lm_head_weight = region(
        loader,
        &sources,
        "lm_head.weight",
        as_u64(first * config.hidden_size, "the head shard offset")?,
        Shape::new(&[as_u64(rows, "the head shard")?, hidden]),
    )?;
    let head = ParallelLMHead::new(
        lm_head_weight,
        as_u32(first, "the head shard")?,
        as_u32(rows, "the head shard")?,
        as_u32(config.vocab_size, "vocab_size")?,
        as_u32(padded, "the padded shard")?,
    )
    .map_err(Error::from)?;

    let norm = RmsNorm::new(
        region(loader, &sources, "norm.weight", 0, Shape::new(&[hidden]))?,
        config.rms_norm_eps,
    )
    .map_err(Error::from)?;

    // The rotary recipe: one frequency table and the score multiplier the recipe adds.
    let (frequencies, score_scale) = rotary_recipe(&config)?;
    let table = loader
        .constant(
            "qwen3-moe.rotary.inv_freq",
            DType::F32,
            Shape::new(&[as_u64(config.head_dim / 2, "head_dim / 2")?]),
            &frequency_bytes(&frequencies),
        )
        .map_err(Error::from)?;
    let _ = head_dim;

    let experts_source: HashMap<usize, crate::weights::ExpertSource> =
        crate::weights::expert_sources(&config)
            .into_iter()
            .filter_map(|source| {
                let layer = source
                    .module
                    .strip_prefix("layers.")?
                    .split('.')
                    .next()?
                    .parse()
                    .ok()?;
                Some((layer, source))
            })
            .collect();

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer in 0..config.num_hidden_layers {
        let parameter = format!("layers.{layer}");
        let input_norm = RmsNorm::new(
            region(
                loader,
                &sources,
                &format!("{parameter}.input_norm.weight"),
                0,
                Shape::new(&[hidden]),
            )?,
            config.rms_norm_eps,
        )
        .map_err(Error::from)?;
        let post_attention_norm = RmsNorm::new(
            region(
                loader,
                &sources,
                &format!("{parameter}.post_attention_norm.weight"),
                0,
                Shape::new(&[hidden]),
            )?,
            config.rms_norm_eps,
        )
        .map_err(Error::from)?;

        let attention = AttentionLayer {
            q_proj: projection(
                loader,
                &sources,
                &format!("{parameter}.attention.q_proj"),
                &partition,
                query,
                hidden,
                config.attention_bias,
            )?,
            k_proj: projection(
                loader,
                &sources,
                &format!("{parameter}.attention.k_proj"),
                &partition,
                key_value,
                hidden,
                config.attention_bias,
            )?,
            v_proj: projection(
                loader,
                &sources,
                &format!("{parameter}.attention.v_proj"),
                &partition,
                key_value,
                hidden,
                config.attention_bias,
            )?,
            o_proj: output_projection(
                loader,
                &sources,
                &format!("{parameter}.attention.o_proj"),
                &partition,
                hidden,
                query,
                config.attention_bias,
            )?,
            q_norm: region(
                loader,
                &sources,
                &format!("{parameter}.attention.q_norm.weight"),
                0,
                Shape::new(&[head_dim]),
            )?,
            k_norm: region(
                loader,
                &sources,
                &format!("{parameter}.attention.k_norm.weight"),
                0,
                Shape::new(&[head_dim]),
            )?,
            rotary: Rotary::new(
                table,
                as_u32(config.head_dim, "head_dim")?,
                as_u32(q_heads, "the query heads")?,
                as_u32(kv_heads, "the key/value heads")?,
                config.rms_norm_eps,
                Rotation::Half,
                1.0,
            )
            .map_err(Error::from)?,
            attention: Attention::paged(
                as_u32(config.head_dim, "head_dim")?,
                as_u32(q_heads, "the query heads")?,
                as_u32(kv_heads, "the key/value heads")?,
                options.page_tokens,
                as_u32(config.max_position_embeddings, "max_position_embeddings")?,
                None,
                score_scale,
            )
            .map_err(Error::from)?,
        };

        let mlp = if config.sparse(layer) {
            Mlp::Experts(expert_block(
                loader,
                &sources,
                &experts_source[&layer],
                &partition,
                &config,
                &parameter,
                hidden,
            )?)
        } else {
            Mlp::Dense(Box::new(dense_block(
                loader, &sources, &partition, &config, &parameter, hidden,
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

/// The activation's width of a `[rows, columns]` shard of a row-major matrix.
fn as_u64(count: usize, what: &str) -> Result<u64> {
    u64::try_from(count).map_err(|_| Error::Config(format!("{what} does not fit an element count")))
}

/// A count as a `u32` extent.
fn as_u32(count: usize, what: &str) -> Result<u32> {
    u32::try_from(count).map_err(|_| Error::Config(format!("{what} exceeds u32::MAX")))
}

/// The vocabulary shard's width rounded up to the head's matmul alignment.
///
/// The head computes a whole number of tiles per rank; the slice's vocabulary (151936) is already a
/// multiple of it, so no padding rows are stored.
fn padded_vocabulary(rows: usize) -> usize {
    rows.next_multiple_of(8)
}

/// The checkpoint tensor of every model parameter, by the model's path.
fn sources(config: &Config) -> HashMap<String, String> {
    crate::weights::parameter_sources(config)
        .into_iter()
        .map(|source| (source.parameter, source.checkpoint))
        .collect()
}

/// The checkpoint tensor that fills `parameter`.
fn checkpoint_name<'a>(sources: &'a HashMap<String, String>, parameter: &str) -> Result<&'a str> {
    sources
        .get(parameter)
        .map(String::as_str)
        .ok_or_else(|| Error::Config(format!("the checkpoint mapping has no {parameter}")))
}

/// The weight of `shape` elements starting at element `start` of `parameter`'s tensor.
fn region(
    loader: &mut WeightLoader<'_>,
    sources: &HashMap<String, String>,
    parameter: &str,
    start: u64,
    shape: Shape,
) -> Result<Weight> {
    let tensor = checkpoint_name(sources, parameter)?;
    loader
        .load(&WeightRequest::region(tensor, start, shape))
        .map_err(Error::from)
}

/// One row-sharded projection: `[rows, columns]` starting at this rank's row block.
fn projection(
    loader: &mut WeightLoader<'_>,
    sources: &HashMap<String, String>,
    parameter: &str,
    partition: &Partition,
    rows: u64,
    columns: u64,
    bias: bool,
) -> Result<Linear> {
    let start = as_u64(partition.tensor_rank, "the tensor rank")? * rows * columns;
    let weight = region(
        loader,
        sources,
        &format!("{parameter}.weight"),
        start,
        Shape::new(&[rows, columns]),
    )?;
    let bias = if bias {
        Some(region(
            loader,
            sources,
            &format!("{parameter}.bias"),
            as_u64(partition.tensor_rank, "the tensor rank")? * rows,
            Shape::new(&[rows]),
        )?)
    } else {
        None
    };
    Linear::new(weight, bias, WeightOrder::NxK).map_err(Error::from)
}

/// The row-parallel output projection: `[hidden, query]`.
///
/// It is sharded along its reduction axis, whose region is strided in the checkpoint's row-major
/// tensor; until the retained weight source expresses a strided region, only one tensor rank is
/// loadable.
fn output_projection(
    loader: &mut WeightLoader<'_>,
    sources: &HashMap<String, String>,
    parameter: &str,
    partition: &Partition,
    hidden: u64,
    reduction: u64,
    bias: bool,
) -> Result<Linear> {
    if partition.tensor_parallel != 1 {
        return Err(Error::Unsupported(format!(
            "{parameter}: a row-parallel shard of the reduction axis is strided in the checkpoint; \
             tensor parallelism above one needs a strided retained region"
        )));
    }
    let weight = region(
        loader,
        sources,
        &format!("{parameter}.weight"),
        0,
        Shape::new(&[hidden, reduction]),
    )?;
    let bias = if bias {
        Some(region(
            loader,
            sources,
            &format!("{parameter}.bias"),
            0,
            Shape::new(&[hidden]),
        )?)
    } else {
        None
    };
    Linear::new(weight, bias, WeightOrder::NxK).map_err(Error::from)
}

/// One sparse layer's expert block: the replicated router and the rank's experts.
fn expert_block(
    loader: &mut WeightLoader<'_>,
    sources: &HashMap<String, String>,
    expert_source: &crate::weights::ExpertSource,
    partition: &Partition,
    config: &Config,
    parameter: &str,
    hidden: u64,
) -> Result<Moe> {
    let experts = as_u64(config.num_experts, "num_experts")?;
    let range = partition.expert_range(config);
    let local = as_u64(range.len(), "the local experts")?;
    if local == 0 {
        return Err(Error::Config(format!(
            "{} owns no expert of {}",
            parameter, config.num_experts
        )));
    }
    let intermediate = as_u64(config.moe_intermediate_size, "moe_intermediate_size")?;

    let router = Linear::new(
        region(
            loader,
            sources,
            &format!("{parameter}.mlp.router.weight"),
            0,
            Shape::new(&[experts, hidden]),
        )?,
        None,
        WeightOrder::NxK,
    )
    .map_err(Error::from)?;

    // Fused per expert, in the reading order the layout names: up before gate.
    let mut up_gate_parts = Vec::with_capacity(range.len() * 2);
    let mut down_parts = Vec::with_capacity(range.len());
    for expert in range.clone() {
        let tensors = expert_source.tensors(expert);
        up_gate_parts.push(tensors.up);
        up_gate_parts.push(tensors.gate);
        down_parts.push(tensors.down);
    }
    let up_gate = loader
        .load(&WeightRequest::parts(
            format!("{parameter}.mlp.experts.up_gate"),
            up_gate_parts,
            Shape::new(&[local, 2 * intermediate, hidden]),
            WeightLayout::UP_GATE,
        ))
        .map_err(Error::from)?;
    let down = loader
        .load(&WeightRequest::parts(
            format!("{parameter}.mlp.experts.down"),
            down_parts,
            Shape::new(&[local, hidden, intermediate]),
            WeightLayout::DOWN,
        ))
        .map_err(Error::from)?;

    let activation = activation(config.hidden_act)?;
    let local = as_u32(range.len(), "the local experts")?;
    let top = as_u32(config.num_experts_per_tok, "num_experts_per_tok")?;
    if partition.expert_parallel == 1 {
        Moe::new(
            router,
            up_gate,
            down,
            None,
            local,
            top,
            activation,
            config.norm_topk_prob,
            None,
        )
        .map_err(Error::from)
    } else {
        // One rank holds a contiguous window of the vocabulary, so the block needs no expert map:
        // `local_offset` is what tells the sort which routes are this rank's, and the router stays
        // replicated over the whole vocabulary so every rank routes a token the same way.
        let offset = as_u32(range.start, "the first local expert")?;
        Moe::expert_parallel(
            router,
            up_gate,
            down,
            None,
            as_u32(config.num_experts, "num_experts")?,
            local,
            offset,
            top,
            activation,
            config.norm_topk_prob,
            None,
        )
        .map_err(Error::from)
    }
}

/// One dense layer's feed-forward block.
fn dense_block(
    loader: &mut WeightLoader<'_>,
    sources: &HashMap<String, String>,
    partition: &Partition,
    config: &Config,
    parameter: &str,
    hidden: u64,
) -> Result<DenseMlp> {
    let intermediate = as_u64(config.intermediate_size, "intermediate_size")?;
    let gate_proj = projection(
        loader,
        sources,
        &format!("{parameter}.mlp.gate_proj"),
        partition,
        intermediate,
        hidden,
        false,
    )?;
    let up_proj = projection(
        loader,
        sources,
        &format!("{parameter}.mlp.up_proj"),
        partition,
        intermediate,
        hidden,
        false,
    )?;
    let down_proj = output_projection(
        loader,
        sources,
        &format!("{parameter}.mlp.down_proj"),
        partition,
        hidden,
        intermediate,
        false,
    )?;
    Ok(DenseMlp {
        gate_proj,
        up_proj,
        down_proj,
        activation: SwiGlu::new(as_u32(config.intermediate_size, "intermediate_size")?)
            .map_err(Error::from)?,
    })
}

/// The operator's name for the checkpoint's activation.
fn activation(activation: Activation) -> Result<MoeActivation> {
    match activation {
        Activation::Silu => Ok(MoeActivation::Silu),
        Activation::GeluTanh => Ok(MoeActivation::GeluTanh),
    }
}

/// The rotary frequency table and the multiplier the recipe applies to the attention scores.
///
/// The table is `[head_dim / 2]` FP32 inverse frequencies: `1 / theta^(2i / head_dim)` for the
/// plain recipe, and transformers' YaRN interpolation for a yarn checkpoint
/// (`_compute_yarn_parameters`: a linear ramp between the extrapolated and the interpolated
/// frequency at the correction range `beta_fast`/`beta_slow`). The multiplier is 1 for the plain
/// recipe and the recipe's attention factor squared for yarn: the reference multiplies the cosine
/// and sine factors by it (`uniserve/nn/rope.py`), so a product of two rotated vectors carries the
/// square.
pub fn rotary_recipe(config: &Config) -> Result<(Vec<f32>, f32)> {
    let dim = config.head_dim;
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

    let Some(yarn) = &config.rope_scaling else {
        return Ok((
            extrapolation.iter().map(|&value| value as f32).collect(),
            1.0,
        ));
    };

    let factor = f64::from(yarn.factor);
    if !(factor.is_finite() && factor > 0.0) {
        return Err(Error::Config(format!(
            "a yarn factor of {factor} is not a positive extension"
        )));
    }
    let (low, high) = correction_range(
        f64::from(yarn.beta_fast),
        f64::from(yarn.beta_slow),
        dim as f64,
        base,
        config.max_position_embeddings as f64,
    );
    let interpolation: Vec<f64> = extrapolation.iter().map(|value| value / factor).collect();
    let frequencies: Vec<f32> = (0..half)
        .map(|index| {
            let ramp = if low == high {
                f64::from(index as f64 >= high)
            } else {
                ((index as f64 - low) / (high - low)).clamp(0.0, 1.0)
            };
            (interpolation[index] * (1.0 - ramp) + extrapolation[index] * ramp) as f32
        })
        .collect();

    let mscale = f64::from(yarn.mscale.unwrap_or(1.0));
    let attention = match (yarn.attention_factor, yarn.mscale_all_dim) {
        (Some(factor), _) => f64::from(factor),
        (None, Some(all)) => yarn_mscale(factor, mscale) / yarn_mscale(factor, f64::from(all)),
        (None, None) => yarn_mscale(factor, mscale),
    };
    let attention = attention as f32;
    Ok((frequencies, attention * attention))
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

/// The table's little-endian… device-native FP32 bytes.
fn frequency_bytes(frequencies: &[f32]) -> Vec<u8> {
    frequencies
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
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
            hidden_act: Activation::Silu,
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
        assert_eq!(scale, 1.0);
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
        // The lowest frequencies are interpolated: a fourth of the plain table's first entry.
        assert!((frequencies[0] - plain[0] / 4.0).abs() < 1e-6);
        // The highest are kept as they were.
        assert!((frequencies[63] - plain[63]).abs() < 1e-9);
        // attention_factor = 0.1 · ln(4) + 1, squared into the score multiplier.
        let expected = (0.1 * 4.0_f32.ln() + 1.0).powi(2);
        assert!((scale - expected).abs() < 1e-6);
    }

    fn config_without_yarn() -> Config {
        config()
    }
}
