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
//! A regional shard of a checkpoint tensor is row-major, so a shard along the *first* axis is
//! `start = rank × rows × width`, and a shard along the *last* axis is a gather of rows that start
//! `full_width` elements apart — `WeightSource::Strided`, one row block per rank. The vocabulary,
//! the query/key/value projections and the router are sharded on their rows; `o_proj` and a dense
//! `down_proj` are sharded on their reduction axis, whose region is strided. Both go through one
//! loader, and a rank that owns the whole axis reads the same region either way.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::Path;

use nn::layers::{
    Attention, ENCODE_CHUNK, Embedding, Linear, Moe, Nvfp4Linear, ParallelLMHead, RmsNorm, Rotary,
    SwiGlu,
};
use nn::weights::{Nvfp4Names, QuantizedStack, Weight, WeightLayout, WeightLoader, WeightRequest};
use nn::{DType, Shape};
use ops::fp4::ScaleLayout;
use ops::gemm::WeightOrder;
use ops::moe::grouped::Activation as MoeActivation;
use ops::rope::Rotation;

use crate::model::{
    AttentionLayer, DecoderLayer, DenseMlp, LoadOptions, Mlp, Projection, Qwen3Moe,
};
use crate::representation::{self, Adopted, Representation, Role};
use crate::{Activation, Config, Error, Partition, Result};

/// The block-scale layout the expert products read the **weights** in: the 128x4 swizzled form.
///
/// Both families carry their weight operand this way — `mSfLayoutA` is `SfLayout(3)` in every one of
/// the artifact's E2M1 kernels, and `BatchedGemmOptions.h` states the rule it follows: "mSfLayoutA
/// has to be R128cX when not batch M or not routed". The plane the two families disagree about is
/// the **tokens** operand, which is `nn::Moe`'s statement, not the loader's.
///
/// The shard stores the plain form, so the loader builds this table once at load.
pub const WEIGHT_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Swizzled128X4;

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
    // The experts' encoding is a property of this entry point, not of the caller's options: `load`
    // reads dense stacks, so the plan reserves no encoded step whatever the options said.
    load_with(
        root,
        partition,
        loader,
        LoadOptions {
            experts_encoded: false,
            ..options
        },
        Quantization::Bf16,
        &Representation::Dense,
    )
}

/// Load a model whose expert stacks are stored encoded (NVFP4).
///
/// Everything but the expert block is identical to [`load`]; the expert block's weights are the
/// checkpoint's packed E2M1 field, its swizzled block scales and its per-expert tensor scales, in the
/// layout the caller names. A checkpoint that stores them any other way is refused rather than
/// reinterpreted, and the BF16 path above is untouched byte for byte.
pub fn load_encoded(
    root: &Path,
    partition: Partition,
    loader: &mut WeightLoader<'_>,
    options: LoadOptions,
    layout: ScaleLayout,
) -> Result<Qwen3Moe> {
    // Likewise for the encoded stacks: this entry point is the one that reads them, so the plan's
    // reservation follows from the call rather than from a second statement the caller could get
    // wrong.
    load_with(
        root,
        partition,
        loader,
        LoadOptions {
            experts_encoded: true,
            ..options
        },
        Quantization::Nvfp4(layout),
        &Representation::read(root)?,
    )
}

/// Load a checkpoint in the representation it declares about itself.
///
/// A calibrated export states its representation in its metadata (`quantization_config` in
/// `config.json`, or a sibling `hf_quant_config.json`); this entry point reads that statement and
/// dispatches, so the encoding is the checkpoint's fact rather than the caller's guess. Two
/// refusals happen before a tensor is opened:
///
/// - a representation this tree cannot serve at all (an algorithm, a method, or two files that
///   disagree) — [`Representation::read`] names it;
/// - a parameter the declaration quantizes whose *role* this model still builds dense. Reading such
///   a parameter as dense would read packed E2M1 as a matrix and produce a model that runs and
///   measures nothing, so it is refused by name instead.
///
/// `layout` is the deployment's: the block-scale layout its programs are compiled for, which an
/// encoded checkpoint's stored layout must equal (the expert stacks refuse a mismatch themselves).
pub fn load_checkpoint(
    root: &Path,
    partition: Partition,
    loader: &mut WeightLoader<'_>,
    options: LoadOptions,
    layout: ScaleLayout,
) -> Result<Qwen3Moe> {
    let representation = Representation::read(root)?;
    // The shards are the fact and the declaration is a claim about them, so a load resolves each
    // parameter's representation from what the checkpoint stores. A shapes walk has no shard to ask
    // and keeps the declaration's answer.
    let representation = match loader.checkpoint() {
        Some(checkpoint) => representation.resolved(checkpoint),
        None => representation,
    };
    refuse_dense_parameters(root, &representation)?;
    match representation {
        Representation::Dense => load(root, partition, loader, options),
        Representation::Nvfp4 { .. } => load_encoded(root, partition, loader, options, layout),
    }
}

/// Refuse a declaration that quantizes a parameter this model builds as a dense module.
///
/// Every parameter of the model is asked, layer by layer, because an exclusion list is matched by
/// module pattern and a pattern can exclude layer 0's projection without excluding layer 3's: one
/// representative per role would miss exactly the checkpoints worth refusing.
fn refuse_dense_parameters(root: &Path, representation: &Representation) -> Result<()> {
    if !representation.is_encoded() {
        return Ok(());
    }
    let config = Config::read(root)?;
    for (parameter, role) in representation::parameters(&config) {
        let parameter = parameter.as_str();
        // A *quantized projection* is the one shape this entry point cannot serve yet: its packed
        // product is keyed by the row count, which `keys()` has no input for. The refusal says so
        // rather than reading the parameter as dense.
        // A packed *projection* is served: the loader builds it as an encoder and a product, so only
        // a role this tree has no encoded form for is refused here.
        match (representation.adopts(parameter, role)?, role) {
            (Adopted::Source, _) | (Adopted::Quantized, Role::Projection) => {}
            (Adopted::Quantized, _) => {
                return Err(Error::Config(format!(
                    "the checkpoint's calibration quantizes `{parameter}`, and this model builds it                      as a dense projection: the encoded projection path needs its packed product                      compiled per row count, which the key walk does not enumerate yet. Load a                      checkpoint whose calibration excludes the attention and dense-MLP projections,                      or wait for that half; reading it as dense would read packed E2M1 as a matrix"
                )));
            }
        }
    }
    Ok(())
}

/// How a checkpoint stores its expert stacks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Quantization {
    /// Dense tensors, one per expert, concatenated into a stack.
    Bf16,
    /// Packed E2M1 with block scales in `ScaleLayout`.
    Nvfp4(ScaleLayout),
}

fn load_with(
    root: &Path,
    partition: Partition,
    loader: &mut WeightLoader<'_>,
    options: LoadOptions,
    quantization: Quantization,
    declaration: &Representation,
) -> Result<Qwen3Moe> {
    // The layout a packed projection's programs read, which is the deployment's choice and the same
    // one the expert stacks use. A dense deployment never builds one, so the value is unreachable
    // there and states the slice's choice rather than inventing one per checkpoint.
    let layout = match quantization {
        Quantization::Nvfp4(layout) => layout,
        // A dense deployment builds no packed projection, so the value is unreachable there; it is
        // the checkpoint's own arrangement, which is what every loader call takes.
        Quantization::Bf16 => ops::fp4::ScaleLayout::Linear,
    };
    let config = Config::read(root)?;
    partition.validate(&config)?;
    // The plan reserves the encoded step's buffers from this option, so a deployment that names the
    // wrong encoding would size an arena its own step writes past. The checkpoint decides it, and
    // the loader is where the two meet.
    let encoded = matches!(quantization, Quantization::Nvfp4(_))
        && config.num_experts > 0
        && config.has_sparse_layers();
    if options.experts_encoded != encoded {
        return Err(Error::Config(format!(
            "the options say the experts are {} and the checkpoint's experts are {}",
            if options.experts_encoded {
                "encoded"
            } else {
                "dense"
            },
            if encoded { "encoded" } else { "dense" }
        )));
    }
    let sources = sources(&config);

    let hidden = as_u64(config.hidden_size, "hidden_size")?;
    let head_dim = as_u64(config.head_dim, "head_dim")?;
    let q_heads = partition.local_attention_heads(&config);
    let kv_heads = partition.local_key_value_heads(&config);
    let query = as_u64(q_heads * config.head_dim, "the query width")?;
    let key_value = as_u64(kv_heads * config.head_dim, "the key/value width")?;

    // The vocabulary shard: rows are contiguous in a `[vocabulary, hidden]` table.
    let (first, rows) = partition.vocabulary_shard(config.vocab_size);
    let padded = partition.padded_vocabulary_shard(config.vocab_size);
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
            "qwen3-moe",
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
                declaration,
                layout,
            )?,
            k_proj: projection(
                loader,
                &sources,
                &format!("{parameter}.attention.k_proj"),
                &partition,
                key_value,
                hidden,
                config.attention_bias,
                declaration,
                layout,
            )?,
            v_proj: projection(
                loader,
                &sources,
                &format!("{parameter}.attention.v_proj"),
                &partition,
                key_value,
                hidden,
                config.attention_bias,
                declaration,
                layout,
            )?,
            o_proj: output_projection(
                loader,
                &sources,
                &format!("{parameter}.attention.o_proj"),
                &partition,
                hidden,
                query,
                config.attention_bias,
                declaration,
                layout,
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
                quantization,
            )?)
        } else {
            Mlp::Dense(Box::new(dense_block(
                loader,
                &sources,
                &partition,
                &config,
                &parameter,
                hidden,
                declaration,
                layout,
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

/// One projection, in the representation the checkpoint declares for it.
///
/// The declaration is the checkpoint's own statement, so this is the one place a representation
/// becomes a type: a projection the calibration left alone is the checkpoint's matrix, and one it
/// packed is an activation encoder and a packed product with the weight's tensor scale as the
/// product's alpha. Reading a packed tensor as dense would read E2M1 codes as a matrix, which is why
/// the choice is made here and not left to a caller.
#[allow(clippy::too_many_arguments)]
fn projection(
    loader: &mut WeightLoader<'_>,
    sources: &HashMap<String, String>,
    parameter: &str,
    partition: &Partition,
    rows: u64,
    columns: u64,
    bias: bool,
    declaration: &Representation,
    layout: ScaleLayout,
) -> Result<Projection> {
    let weight = format!("{parameter}.weight");
    match declaration.adopts(&weight, Role::Projection)? {
        Adopted::Source => Ok(Projection::Dense(dense_projection(
            loader, sources, parameter, partition, rows, columns, bias,
        )?)),
        Adopted::Quantized => encoded_projection(
            loader, sources, parameter, partition, rows, columns, bias, layout,
        ),
    }
}

/// One row-sharded encoded projection: the packed pair `[rows, columns]` at this rank's row block.
///
/// The checkpoint stores two E2M1 codes a byte along the contraction, so a row is `columns / 2` bytes
/// and the block scales are `columns / 16` of them; the row block is the same in both.
#[allow(clippy::too_many_arguments)]
fn encoded_projection(
    loader: &mut WeightLoader<'_>,
    sources: &HashMap<String, String>,
    parameter: &str,
    partition: &Partition,
    rows: u64,
    columns: u64,
    bias: bool,
    layout: ScaleLayout,
) -> Result<Projection> {
    let rank = as_u64(partition.tensor_rank, "the tensor rank")?;
    let values = WeightRequest::region(
        checkpoint_name(sources, &format!("{parameter}.weight"))?,
        rank * rows * (columns / 2),
        Shape::new(&[rows, columns / 2]),
    );
    let scales = WeightRequest::region(
        &packed_field(&values, Nvfp4Names::vllm().block_scale)?,
        rank * rows * (columns / 16),
        Shape::new(&[rows, columns / 16]),
    );
    encoded(loader, parameter, values, scales, bias, layout)
}

/// One row-parallel encoded projection: this rank's `columns` of a `[rows, columns × ranks]` matrix.
#[allow(clippy::too_many_arguments)]
fn encoded_output_projection(
    loader: &mut WeightLoader<'_>,
    sources: &HashMap<String, String>,
    parameter: &str,
    partition: &Partition,
    rows: u64,
    columns: u64,
    bias: bool,
    layout: ScaleLayout,
) -> Result<Projection> {
    let rank = as_u64(partition.tensor_rank, "the tensor rank")?;
    let ranks = as_u64(partition.tensor_parallel, "the tensor parallel width")?;
    let values = WeightRequest::strided(
        checkpoint_name(sources, &format!("{parameter}.weight"))?,
        rank * (columns / 2),
        Shape::new(&[rows, columns / 2]),
        columns * ranks / 2,
    );
    let scales = WeightRequest::strided(
        &packed_field(&values, Nvfp4Names::vllm().block_scale)?,
        rank * (columns / 16),
        Shape::new(&[rows, columns / 16]),
        columns * ranks / 16,
    );
    encoded(loader, parameter, values, scales, bias, layout)
}

/// The packed weight and the product's alpha, from the checkpoint's four tensors.
///
/// The weight's `weight_scale_2` is one positive float and its reciprocal is what the product takes,
/// so it is read once here and published as the layer's `[1]` F32 alpha; the activation's
/// `input_scale` is the encoder's, which the loader keeps as its own operand. Neither is invented: a
/// checkpoint that states no calibration cannot serve a packed projection and is refused where the
/// tensor is asked for.
#[allow(clippy::too_many_arguments)]
fn encoded(
    loader: &mut WeightLoader<'_>,
    parameter: &str,
    values: WeightRequest,
    scales: WeightRequest,
    bias: bool,
    layout: ScaleLayout,
) -> Result<Projection> {
    if bias {
        return Err(Error::Config(format!(
            "{parameter} is packed and asks for a bias, and no calibrated NVFP4 export writes one"
        )));
    }
    let tensor_scale = WeightRequest::region(
        &packed_field(&values, Nvfp4Names::vllm().tensor_scale)?,
        0,
        Shape::new(&[1]),
    );
    let global_scale = loader.tensor_scale(&tensor_scale)?;
    let input_scale = WeightRequest::region(
        &packed_field(&values, Nvfp4Names::vllm().input_scale)?,
        0,
        Shape::new(&[1]),
    );
    // A dense product dequantizes `code * block_scale` on both sides, so its alpha is the two tensor
    // scales' product: the weight's and the activation's.
    let product_alpha = global_scale * loader.tensor_scale(&input_scale)?;
    // `layout` is the **checkpoint's** arrangement — linear, one scale per sixteen elements — and
    // not the product's, which the loader is given separately and states back in
    // `QuantizedWeight::layout`, so the two cannot be confused for one another.
    // The file's arrangement is the checkpoint's; the product's is the one `nvfp4_gemm`'s
    // compilation serves, and the loader permutes between them.
    let quantized = loader.nvfp4(
        &values,
        &scales,
        Some(&input_scale),
        Nvfp4Names::vllm(),
        layout,
        ScaleLayout::Swizzled128X4,
        global_scale,
    )?;
    let alpha = loader.constant(
        &format!("{parameter}.alpha"),
        DType::F32,
        Shape::new(&[1]),
        &product_alpha.to_le_bytes(),
        "qwen3-moe",
    )?;
    let linear = Nvfp4Linear::new(
        &quantized,
        None,
        quantized.layout,
        DType::BF16,
        ENCODE_CHUNK,
    )?;
    Ok(Projection::Encoded {
        linear: Box::new(linear),
        alpha,
    })
}

/// The checkpoint field a packed projection's values name, beside them.
///
/// A calibrated export spells the encoded fields as suffixes of the parameter's own stem —
/// `…q_proj.weight`, `…q_proj.weight_scale`, `…q_proj.weight_scale_2`, `…q_proj.input_scale` — and the
/// mapping states only the first, because it is the only one a dense checkpoint has. The other three
/// are derived here from the same spelling the loader is told to read them in.
fn packed_field(values: &WeightRequest, suffix: &str) -> Result<String> {
    let stem = values.tensor.strip_suffix(Nvfp4Names::vllm().values).ok_or_else(|| {
        Error::Config(format!(
            "the packed values of {} are not spelled `….{}`, so their block scales and tensor scale \
             cannot be named beside them",
            values.tensor,
            Nvfp4Names::vllm().values
        ))
    })?;
    // The spelling's suffixes are bare words, so the stem loses the dot that joined the values to
    // their own name: `…q_proj.weight` + `weight_scale` is `…q_proj.weight_scale`, not `..`.
    let stem = stem.strip_suffix('.').unwrap_or(stem);
    Ok(format!("{stem}.{suffix}"))
}

/// One row-sharded projection: `[rows, columns]` starting at this rank's row block.
fn dense_projection(
    loader: &mut WeightLoader<'_>,
    sources: &HashMap<String, String>,
    parameter: &str,
    partition: &Partition,
    rows: u64,
    columns: u64,
    bias: bool,
) -> Result<Linear> {
    let start = as_u64(partition.tensor_rank, "the tensor rank")? * rows * columns;
    // A dense projection is a matrix that multiplies the hidden state, so an all-zero one is a
    // missing export and not a value: the packed path refuses the same defect from its codes and
    // scales (`WeightLoader::nvfp4`), and this is the dense export's way to it.
    let request = WeightRequest::region(
        checkpoint_name(sources, &format!("{parameter}.weight"))?,
        start,
        Shape::new(&[rows, columns]),
    )
    .nonzero();
    let weight = loader.load(&request).map_err(Error::from)?;
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

/// The row-parallel output projection: `[hidden, reduction]`, this rank's block of the reduction.
///
/// The checkpoint stores the whole `[hidden, reduction × ranks]` matrix row-major, so this rank's
/// block is every row's own `reduction` elements: a gather of rows `reduction × ranks` apart,
/// which the loader converts into one contiguous retained weight. The bias is not sharded.
/// One row-parallel projection, in the representation the checkpoint declares.
#[allow(clippy::too_many_arguments)]
fn output_projection(
    loader: &mut WeightLoader<'_>,
    sources: &HashMap<String, String>,
    parameter: &str,
    partition: &Partition,
    hidden: u64,
    reduction: u64,
    bias: bool,
    declaration: &Representation,
    layout: ScaleLayout,
) -> Result<Projection> {
    let weight = format!("{parameter}.weight");
    match declaration.adopts(&weight, Role::Projection)? {
        Adopted::Source => Ok(Projection::Dense(dense_output_projection(
            loader, sources, parameter, partition, hidden, reduction, bias,
        )?)),
        Adopted::Quantized => encoded_output_projection(
            loader, sources, parameter, partition, hidden, reduction, bias, layout,
        ),
    }
}

/// The dense row-parallel form, which the dispatcher above serves when the checkpoint is not packed.
fn dense_output_projection(
    loader: &mut WeightLoader<'_>,
    sources: &HashMap<String, String>,
    parameter: &str,
    partition: &Partition,
    hidden: u64,
    reduction: u64,
    bias: bool,
) -> Result<Linear> {
    // The same refusal as the row-sharded form: this matrix multiplies the hidden state, so an
    // all-zero region is a missing export.
    let request = row_parallel_region(
        partition,
        checkpoint_name(sources, &format!("{parameter}.weight"))?,
        hidden,
        reduction,
    )?
    .nonzero();
    let weight = loader.load(&request).map_err(Error::from)?;
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
#[allow(
    clippy::too_many_arguments,
    reason = "the block is named by its parts: the loader, the checkpoint's sources, the expert \
              source, the partition, the config, the parameter and the quantization"
)]
fn expert_block(
    loader: &mut WeightLoader<'_>,
    sources: &HashMap<String, String>,
    expert_source: &crate::weights::ExpertSource,
    partition: &Partition,
    config: &Config,
    parameter: &str,
    hidden: u64,
    quantization: Quantization,
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

    // The encoded checkpoint names the same three tensors per projection and carries three fields
    // each; the dense path names one tensor per projection. Both are fused expert-major, in the
    // reading order the layouts name: up before gate.
    if let Quantization::Nvfp4(layout) = quantization {
        // The encoded block is colocated by construction: its stacks hold every expert the router
        // names, so a rank holding only a window cannot be served from them.
        if partition.expert_parallel != 1 {
            return Err(Error::Unsupported(format!(
                "{parameter} is encoded, and an encoded expert block is colocated: this rank holds \
                 {} of {} experts",
                range.len(),
                config.num_experts
            )));
        }
        let vocabulary = as_u32(config.num_experts, "num_experts")?;
        // The shard's block scales are the export's own — linear, one per sixteen elements — and a
        // product that reads a swizzled table has it built once here. A checkpoint that already
        // holds a converted table is refused rather than swizzled a second time.
        if layout != ScaleLayout::Linear {
            return Err(Error::Unsupported(format!(
                "{parameter} declares {layout} block scales, and this loader builds the published \
                 table from a linear shard: a converted checkpoint needs its own reader"
            )));
        }
        let up_gate: QuantizedStack = loader
            .nvfp4_stack(
                &format!("{parameter}.mlp.experts.up_gate"),
                &expert_source.checkpoint_prefix,
                &["up", "gate"],
                vocabulary,
                Nvfp4Names::vllm(),
                nn::weights::ValueLayout::InterleavedShuffled {
                    epilogue_tile_m: 128,
                },
                WEIGHT_SCALE_LAYOUT,
                true,
            )
            .map_err(Error::from)?;
        let down: QuantizedStack = loader
            .nvfp4_stack(
                &format!("{parameter}.mlp.experts.down"),
                &expert_source.checkpoint_prefix,
                &["down"],
                vocabulary,
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
            None,
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

/// The region of a row-parallel projection's checkpoint tensor this rank owns.
///
/// The checkpoint holds `[rows, columns × tensor_parallel]` row-major, so the rank's block is its
/// own `columns` elements of every row: `Strided` with the first element at `rank × columns` and
/// rows `columns × tensor_parallel` apart. One rank lays the same region out contiguously, so the
/// two degrees share the one request form and nothing downstream can tell them apart.
fn row_parallel_region(
    partition: &Partition,
    tensor: &str,
    rows: u64,
    columns: u64,
) -> Result<WeightRequest> {
    let rank = as_u64(partition.tensor_rank, "the tensor rank")?;
    let ranks = as_u64(partition.tensor_parallel, "the tensor parallel width")?;
    let full = columns.checked_mul(ranks).ok_or_else(|| {
        Error::Config(format!(
            "{tensor}: {columns} columns over {ranks} ranks overflows"
        ))
    })?;
    let start = rank.checked_mul(columns).ok_or_else(|| {
        Error::Config(format!(
            "{tensor}: rank {rank} of {ranks} starts past the reduction axis"
        ))
    })?;
    Ok(WeightRequest::strided(
        tensor,
        start,
        Shape::new(&[rows, columns]),
        full,
    ))
}

/// One dense layer's feed-forward block.
#[allow(clippy::too_many_arguments)]
fn dense_block(
    loader: &mut WeightLoader<'_>,
    sources: &HashMap<String, String>,
    partition: &Partition,
    config: &Config,
    parameter: &str,
    hidden: u64,
    declaration: &Representation,
    layout: ScaleLayout,
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
        declaration,
        layout,
    )?;
    let up_proj = projection(
        loader,
        sources,
        &format!("{parameter}.mlp.up_proj"),
        partition,
        intermediate,
        hidden,
        false,
        declaration,
        layout,
    )?;
    let down_proj = output_projection(
        loader,
        sources,
        &format!("{parameter}.mlp.down_proj"),
        partition,
        hidden,
        intermediate,
        false,
        declaration,
        layout,
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
pub(crate) fn activation(activation: Activation) -> Result<MoeActivation> {
    match activation {
        Activation::Silu => Ok(MoeActivation::Silu),
        Activation::GeluTanh => Ok(MoeActivation::GeluTanh),
    }
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
    use super::*;
    use crate::{Config, YaRn};
    use nn::weights::WeightSource;

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
    fn a_row_parallel_projection_reads_its_own_column_block_of_every_row() {
        // Two ranks over a `[2048, 1536]` matrix: rank 1 owns columns 768..1536, which in the
        // checkpoint's row-major order is every row's second half.
        let sharded = Partition {
            tensor_parallel: 2,
            tensor_rank: 1,
            expert_parallel: 1,
            expert_rank: 0,
        };
        let request = row_parallel_region(&sharded, "o_proj.weight", 2048, 768).unwrap();
        assert_eq!(request.shape, Shape::new(&[2048, 768]));
        match &request.source {
            WeightSource::Strided {
                tensor,
                start,
                shape,
                row_stride,
            } => {
                assert_eq!(tensor, "o_proj.weight");
                assert_eq!(*start, 768);
                assert_eq!(*shape, Shape::new(&[2048, 768]));
                assert_eq!(*row_stride, 1536);
            }
            source => panic!("rank 1 of 2 read {source:?} instead of its column block"),
        }

        // Rank 0 reads the first block, and one rank reads the whole axis: the same form, the
        // region then being every row's own columns.
        let first = row_parallel_region(
            &Partition {
                tensor_rank: 0,
                ..sharded
            },
            "o_proj.weight",
            2048,
            768,
        )
        .unwrap();
        assert!(matches!(
            first.source,
            WeightSource::Strided {
                start: 0,
                row_stride: 1536,
                ..
            }
        ));
        let whole = row_parallel_region(&Partition::single(), "o_proj.weight", 2048, 768).unwrap();
        assert!(matches!(
            whole.source,
            WeightSource::Strided {
                start: 0,
                row_stride: 768,
                ..
            }
        ));
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
        let kernels = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ops/kernels");
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
