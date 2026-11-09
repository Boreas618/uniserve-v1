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
    Attention, Bias, Embedding, Linear, Moe, Parallel, ParallelLMHead, QkvBias, QkvLinear,
    QkvNames, QkvSpec, RmsNorm, Rotary, SwiGlu,
};
use nn::quantization::Declaration;
use nn::weights::{
    Nvfp4Names, PartBlock, QuantizedStack, Scope, WeightLayout, WeightLoader, WeightRequest,
};
use nn::{DType, Shape};
use ops::fp4::ScaleLayout;
use ops::gemm::WeightOrder;
use ops::moe::grouped::Activation as MoeActivation;
use ops::rope::Rotation;

use crate::model::{AttentionLayer, DecoderLayer, DenseMlp, LoadOptions, Mlp, Qwen3Moe};
use crate::{Config, Error, Partition, Result};

/// The block-scale layout the expert products read the **weights** in: the 128x4 swizzled form.
///
/// Both families carry their weight operand this way — `mSfLayoutA` is `SfLayout(3)` in every one of
/// the artifact's E2M1 kernels, and `BatchedGemmOptions.h` states the rule it follows: "mSfLayoutA
/// has to be R128cX when not batch M or not routed". The plane the two families disagree about is
/// the **tokens** operand, which is `nn::Moe`'s statement, not the loader's.
///
/// The shard stores the plain form, so the loader builds this table once at load.
pub const WEIGHT_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Swizzled128X4;

/// Each routed layer's logical expert ids in physical slot order, by the module path
/// [`crate::weights::ExpertSource::module`] states. A layer this map does not name keeps the
/// checkpoint's order.
pub type ExpertOrder = std::collections::BTreeMap<String, Vec<u32>>;

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

    let hidden = u64::from(config.hidden_size);
    let head_dim = u64::from(config.head_dim);
    let q_heads = partition.local_attention_heads(&config);
    let kv_heads = partition.local_key_value_heads(&config);

    // The vocabulary shard: rows are contiguous in a `[vocabulary, hidden]` table.
    let (first, rows) = partition.vocabulary_shard(config.vocab_size);
    let padded = partition.padded_vocabulary_shard(config.vocab_size);
    if padded != rows {
        return Err(Error::Unsupported(format!(
            "a vocabulary shard of {rows} rows needs {padded} padded rows; the checkpoint's rows \
             are contiguous, so a rank whose shard is not its padded width is not loadable yet"
        )));
    }
    let mut root = loader.scope("");
    let embedding_weight = root.scope("model.embed_tokens").rows(
        "weight",
        u64::from(first),
        &[u64::from(rows), hidden],
    )?;
    let embedding = Embedding::new(embedding_weight, first, rows, config.vocab_size)?;
    let dtype = embedding_weight.view().dtype;

    // A tied head reads the embedding's own matrix.
    let head_module = if config.tie_word_embeddings {
        "model.embed_tokens"
    } else {
        "lm_head"
    };
    let lm_head_weight =
        root.scope(head_module)
            .rows("weight", u64::from(first), &[u64::from(rows), hidden])?;
    let head = ParallelLMHead::new(lm_head_weight, first, rows, config.vocab_size, padded)?;

    let norm = RmsNorm::new(
        root.scope("model.norm").tensor("weight", &[hidden])?,
        config.rms_norm_eps,
    )?;

    // The rotary recipe: one frequency table and the score multiplier the recipe adds.
    let (frequencies, score_scale) = rotary_recipe(&config)?;
    let table = loader.constant(
        "qwen3-moe.rotary.inv_freq",
        DType::F32,
        Shape::new(&[head_dim / 2]),
        &frequency_bytes(&frequencies),
        "qwen3-moe",
    )?;

    let (slots, slot_table) = expert_slots(&config, order, loader)?;
    let experts_source: HashMap<u32, crate::weights::ExpertSource> =
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

    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
    for layer in 0..config.num_hidden_layers {
        let parameter = format!("layers.{layer}");
        let mut scope = loader.scope(format!("model.layers.{layer}"));
        let input_norm = RmsNorm::new(
            scope.scope("input_layernorm").tensor("weight", &[hidden])?,
            config.rms_norm_eps,
        )?;
        let post_attention_norm = RmsNorm::new(
            scope
                .scope("post_attention_layernorm")
                .tensor("weight", &[hidden])?,
            config.rms_norm_eps,
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
            // The router is replicated: every rank routes a token the same way.
            let router = Linear::new(
                scope
                    .scope("mlp.gate")
                    .tensor("weight", &[u64::from(config.num_experts), hidden])?,
                None,
                WeightOrder::NxK,
            )?;

            Mlp::Experts(expert_block(
                loader,
                router,
                &experts_source[&layer],
                &partition,
                &config,
                &parameter,
                hidden,
                experts_encoded,
                &slots[&layer],
                map,
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

/// A count as a `u32` extent.
fn as_u32(count: impl TryInto<u32>, what: &str) -> Result<u32> {
    count
        .try_into()
        .map_err(|_| Error::Config(format!("{what} exceeds u32::MAX")))
}

/// The slot of each logical expert, from `order`, the logical expert of each slot.
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
    let mut row = vec![u32::MAX; experts];
    for (slot, expert) in order.iter().enumerate() {
        let entry = row
            .get_mut(*expert as usize)
            .ok_or_else(|| Error::Config(format!("the order of {module} names expert {expert}")))?;
        *entry = as_u32(slot, "an expert slot")?;
    }
    if row.contains(&u32::MAX) {
        return Err(Error::Config(format!(
            "the order of {module} is not a permutation of its experts"
        )));
    }
    Ok(row)
}

/// Each routed layer's logical experts in slot order, keyed by layer index, and the published slot
/// table with the routed layers whose rows it holds, in row order (`None` without a routed layer).
type ExpertSlots = (HashMap<u32, Vec<u32>>, Option<(nn::Weight, Vec<u32>)>);

/// The experts of each routed layer in slot order, checked, with the table a route reads its slot
/// from: `[routed layers, experts]` i32, row by row in layer order.
///
/// Every layer's row is one weight's row, because a retained weight takes a whole allocation
/// granule: one table per layer would hold 2 MiB each for 512 bytes of content.
fn expert_slots(
    config: &Config,
    order: &ExpertOrder,
    loader: &mut WeightLoader<'_>,
) -> Result<ExpertSlots> {
    let experts = config.num_experts;
    let layers: Vec<u32> = (0..config.num_hidden_layers)
        .filter(|&layer| config.sparse(layer))
        .collect();
    for module in order.keys() {
        if !crate::weights::expert_sources(config)
            .iter()
            .any(|source| &source.module == module)
        {
            return Err(Error::Config(format!(
                "{module} is not a routed expert layer of this model"
            )));
        }
    }
    let mut slots = HashMap::with_capacity(layers.len());
    let mut table = Vec::with_capacity(layers.len() * experts as usize * 4);
    for &layer in &layers {
        let module = format!("layers.{layer}.mlp.experts");
        let order: Vec<u32> = match order.get(&module) {
            Some(order) => order.clone(),
            None => (0..experts).collect(),
        };
        table.extend(
            slot_row(&module, &order, experts as usize)?
                .iter()
                .flat_map(|slot| slot.to_le_bytes()),
        );
        slots.insert(layer, order);
    }
    if layers.is_empty() {
        return Ok((slots, None));
    }
    let weight = loader.constant(
        "qwen3-moe.experts.slots",
        DType::I32,
        Shape::new(&[
            as_u64(layers.len(), "the routed layers")?,
            u64::from(experts),
        ]),
        &table,
        "qwen3-moe",
    )?;
    Ok((slots, Some((weight, layers))))
}
/// One sparse layer's expert block: the replicated router and the rank's experts.
#[allow(
    clippy::too_many_arguments,
    reason = "the block is named by its parts: the loader, the checkpoint's sources, the expert \
              source, the partition, the config, the parameter and the quantization"
)]
fn expert_block(
    loader: &mut WeightLoader<'_>,
    router: Linear,
    expert_source: &crate::weights::ExpertSource,
    partition: &Partition,
    config: &Config,
    parameter: &str,
    hidden: u64,
    encoded: bool,
    slots: &[u32],
    map: nn::Weight,
) -> Result<Moe> {
    let range = partition.expert_range(config);
    let range = range.start as usize..range.end as usize;
    let local = as_u64(range.len(), "the local experts")?;
    if local == 0 {
        return Err(Error::Config(format!(
            "{} owns no expert of {}",
            parameter, config.num_experts
        )));
    }
    let intermediate = as_u64(config.moe_intermediate_size, "moe_intermediate_size")?;
    let ranks = as_u64(partition.tensor_parallel, "the tensor parallel width")?;
    let rank = as_u64(partition.tensor_rank, "the tensor rank")?;
    if !intermediate.is_multiple_of(ranks) {
        return Err(Error::Config(format!(
            "{parameter}: a tensor group of {ranks} does not divide an expert's {intermediate} \
             intermediate columns"
        )));
    }
    // The rank's block of every expert: the tensor group splits the intermediate width.
    let local_intermediate = intermediate / ranks;

    // The encoded checkpoint names the same three tensors per projection and carries three fields
    // each; the dense path names one tensor per projection. Both are fused expert-major, in the
    // reading order the layouts name: up before gate.
    if encoded {
        // An encoded block over a group runs in the exchange's one fused launch, which reads its
        // own arrangement of the same bytes: each gate block before its up block, and the block
        // scales in the swizzled table.
        if partition.expert_parallel != 1 {
            let first = as_u32(range.start, "the first local expert")?;
            let last = as_u32(range.end, "the last local expert")?;
            let vocabulary = as_u32(config.num_experts, "num_experts")?;
            let (up_gate, up_scales): (QuantizedStack, Vec<f32>) = loader
                .nvfp4_window(
                    &format!("{parameter}.mlp.experts.up_gate"),
                    &expert_source.checkpoint_prefix,
                    &["up", "gate"],
                    &slots[range.clone()],
                    vocabulary,
                    Nvfp4Names::vllm(),
                    nn::weights::ValueLayout::GateFirst,
                    ScaleLayout::Swizzled128X4,
                    true,
                )
                .map_err(Error::from)?;
            let (down, down_scales): (QuantizedStack, Vec<f32>) = loader
                .nvfp4_window(
                    &format!("{parameter}.mlp.experts.down"),
                    &expert_source.checkpoint_prefix,
                    &["down"],
                    &slots[range.clone()],
                    vocabulary,
                    Nvfp4Names::vllm(),
                    nn::weights::ValueLayout::Plain,
                    ScaleLayout::Swizzled128X4,
                    true,
                )
                .map_err(Error::from)?;
            // The kernel's three scalars per expert, as the Python provider computed them: each
            // product dequantizes with the weight's tensor scale times its rows' activation scale,
            // and the gated rows encode with the second activation scale's reciprocal. A fused
            // first projection carries one tensor scale per expert, which up and gate share.
            let tables = |bytes: Vec<f32>| -> Vec<u8> {
                bytes.iter().flat_map(|value| value.to_ne_bytes()).collect()
            };
            let local = u64::try_from(range.len()).unwrap_or(0);
            let table = |loader: &mut WeightLoader<'_>, what: &str, values: Vec<f32>| {
                loader
                    .constant(
                        &format!("{parameter}.mlp.experts.{what}"),
                        DType::F32,
                        Shape::new(&[local]),
                        &tables(values),
                        "qwen3-moe",
                    )
                    .map_err(Error::from)
            };
            let up_alpha: Vec<f32> = up_scales
                .chunks(2)
                .map(|scales| scales[0] * up_gate.activation_scale)
                .collect();
            let down_alpha: Vec<f32> = down_scales
                .iter()
                .map(|scale| scale * down.activation_scale)
                .collect();
            let gated = vec![down.activation_scale.recip(); range.len()];
            let tables = nn::layers::FusedTables {
                up_gate_alpha: table(loader, "up_gate_alpha", up_alpha)?,
                down_alpha: table(loader, "down_alpha", down_alpha)?,
                gated_scale: table(loader, "gated_scale", gated)?,
            };
            return Moe::fused_expert_parallel(
                router,
                up_gate,
                down,
                tables,
                Some(map),
                vocabulary,
                last - first,
                first,
                as_u32(config.num_experts_per_tok, "num_experts_per_tok")?,
                activation(config.hidden_act)?,
                config.norm_topk_prob,
            )
            .map_err(Error::from);
        }
        let vocabulary = as_u32(config.num_experts, "num_experts")?;
        let up_gate: QuantizedStack = loader
            .nvfp4_stack_in_slots(
                &format!("{parameter}.mlp.experts.up_gate"),
                &expert_source.checkpoint_prefix,
                &["up", "gate"],
                slots,
                Nvfp4Names::vllm(),
                nn::weights::ValueLayout::InterleavedShuffled {
                    epilogue_tile_m: 128,
                },
                WEIGHT_SCALE_LAYOUT,
                true,
            )
            .map_err(Error::from)?;
        let down: QuantizedStack = loader
            .nvfp4_stack_in_slots(
                &format!("{parameter}.mlp.experts.down"),
                &expert_source.checkpoint_prefix,
                &["down"],
                slots,
                Nvfp4Names::vllm(),
                nn::weights::ValueLayout::Shuffled {
                    epilogue_tile_m: 128,
                },
                WEIGHT_SCALE_LAYOUT,
                true,
            )
            .map_err(Error::from)?;
        let activation = activation(config.hidden_act)?;
        let top = as_u32(config.num_experts_per_tok, "num_experts_per_tok")?;
        return Moe::quantized(
            router,
            up_gate,
            down,
            Some(map),
            vocabulary,
            top,
            activation,
            config.norm_topk_prob,
            None,
        )
        .map_err(Error::from);
    }

    let mut up_gate_parts = Vec::with_capacity(range.len() * 2);
    let mut down_parts = Vec::with_capacity(range.len());
    for slot in range.clone() {
        let tensors = expert_source.tensors(slots[slot] as usize);
        up_gate_parts.push(tensors.up);
        up_gate_parts.push(tensors.gate);
        down_parts.push(tensors.down);
    }
    // A gate or up projection is `[intermediate, hidden]`, so a rank's block is its rows; a down
    // projection is `[hidden, intermediate]`, so a rank's block is its columns of every row.
    let request = |name: String, parts: Vec<String>, shape: Shape, layout, block: PartBlock| {
        if ranks == 1 {
            WeightRequest::parts(name, parts, shape, layout)
        } else {
            WeightRequest::part_blocks(name, parts, shape, layout, block)
        }
    };
    let up_gate = loader
        .load(&request(
            format!("{parameter}.mlp.experts.up_gate"),
            up_gate_parts,
            Shape::new(&[local, 2 * local_intermediate, hidden]),
            WeightLayout::UP_GATE,
            PartBlock {
                start: rank * local_intermediate * hidden,
                rows: local_intermediate,
                columns: hidden,
                row_stride: hidden,
            },
        ))
        .map_err(Error::from)?;
    let down = loader
        .load(&request(
            format!("{parameter}.mlp.experts.down"),
            down_parts,
            Shape::new(&[local, hidden, local_intermediate]),
            WeightLayout::DOWN,
            PartBlock {
                start: rank * local_intermediate,
                rows: hidden,
                columns: local_intermediate,
                row_stride: intermediate,
            },
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
            Some(map),
            local,
            top,
            activation,
            config.norm_topk_prob,
            None,
        )
        .map_err(Error::from)
    } else {
        // One rank holds a contiguous window of the slots: `local_offset` is what tells the sort
        // which routes are this rank's, and the router stays replicated over the whole vocabulary
        // so every rank routes a token the same way and the table names its slot on every rank.
        let offset = as_u32(range.start, "the first local expert")?;
        Moe::expert_parallel(
            router,
            up_gate,
            down,
            Some(map),
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
    let activation = SwiGlu::new(gate_proj.n())?;
    Ok(DenseMlp {
        gate_proj,
        up_proj,
        down_proj,
        activation,
    })
}

/// The operator's name for the checkpoint's activation.
pub(crate) fn activation(activation: ops::activation::Activation) -> Result<MoeActivation> {
    MoeActivation::try_from(activation).map_err(|error| Error::Nn(error.into()))
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

/// The table's little-endian… device-native FP32 bytes.
fn frequency_bytes(frequencies: &[f32]) -> Vec<u8> {
    frequencies
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

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

    /// The block-scale layouts each product's kernels were built for, held against the packages.
    ///
    /// A product's planes are read by kernels that were **compiled** with a layout for each operand,
    /// and nothing checks the two against each other at runtime: the layouts are not options the
    /// runner's lookup compares (`GemmOptions` carries no `SfLayout`), so a package built for one
    /// form handed a plane in another reads every block factor at the wrong offset and answers
    /// plausible numbers. This is the one place the two statements meet.
    ///
    /// `mSfLayoutA` is the **weights** and `mSfLayoutB` the **tokens** — `BatchedGemmOptions.h`
    /// states both ("mSfLayoutA has to be R128cX", "Tokens need use SF linear layout when being
    /// routed") — and the two families disagree about the second only.
    #[test]
    fn a_products_package_states_the_scale_layouts_nn_publishes() {
        let kernels = Path::new(ops::KERNELS).to_path_buf();
        let cases = [
            (
                "gather_gemm_nvfp4",
                WEIGHT_SCALE_LAYOUT,
                Moe::GATED_TOKEN_SCALE_LAYOUT,
            ),
            (
                "route_gemm_nvfp4",
                WEIGHT_SCALE_LAYOUT,
                Moe::ACTIVATION_SCALE_LAYOUT,
            ),
        ];
        let name = |layout: ScaleLayout| match layout {
            ScaleLayout::Linear => "\"linear\"",
            ScaleLayout::Swizzled128X4 => "\"swizzled_128x4\"",
        };
        for (operator, weights, tokens) in cases {
            let package = kernels.join(operator).join("trtllm/package.toml");
            let stated = std::fs::read_to_string(&package)
                .unwrap_or_else(|error| panic!("{}: {error}", package.display()));
            for (option, expected) in [("mSfLayoutA", name(weights)), ("mSfLayoutB", name(tokens))]
            {
                let line = format!("{option} = {expected}");
                assert!(
                    stated.contains(&line),
                    "{operator}'s package does not state `{line}`: the kernels it publishes would \
                     read a plane nn publishes in another form, and nothing at launch would say so"
                );
            }
        }
    }
}
