//! The modules of `nn::modules`: each one loaded from a checkpoint written here and computed on
//! the CPU device, and loaded by a meta loader and computed under a meta context for the kernels
//! it states.
//!
//! Nothing here touches a GPU.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

mod support;

use std::path::Path;

use nn::collective::Split;
use nn::functional::{self, Activation, Rounding, WeightOrder};
use nn::modules::{
    Embedding, EmbeddingConfig, GateUp, GatedRmsNorm, GatedRmsNormConfig, LayerNorm,
    LayerNormConfig, Linear, LinearConfig, LmHead, LmHeadConfig, Mlp, MlpConfig, Parallel, QkvBias,
    QkvLinear, QkvLinearConfig, QkvNames, RmsNorm, RmsNormConfig,
};
use nn::quantization::Declaration;
use nn::{Adapter, Checkpoint, CpuWeights, DType, Device, Tensor, WeightLoader};
use support::{
    assert_close, encode, f32_bytes, meta, operators, recorded, recorded_split, sequence, shard,
    shard_stating, tensor, values,
};

/// The tolerance of an FP32 result against a reference accumulated in another order.
const F32_TOLERANCE: f32 = 2e-5;

/// The checkpoint at `directory` holding the FP32 tensors `tensors` (name, extents, values).
fn checkpoint(directory: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Checkpoint {
    typed_checkpoint(directory, DType::F32, tensors)
}

/// [`checkpoint`], its tensors stored as `dtype`.
fn typed_checkpoint(
    directory: &Path,
    dtype: DType,
    tensors: &[(&str, Vec<usize>, Vec<f32>)],
) -> Checkpoint {
    let typed: Vec<(&str, DType, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, shape, values)| (*name, dtype, shape.clone(), encode(dtype, values)))
        .collect();
    shard(directory, "model.safetensors", &typed);

    Checkpoint::open(directory).unwrap()
}

/// `x [m, k] · weightᵀ` for a weight stored `[n, k]`, plus `bias`.
fn project(x: &[f32], weight: &[f32], bias: Option<&[f32]>, k: usize) -> Vec<f32> {
    let n = weight.len() / k;

    x.chunks(k)
        .flat_map(|row| {
            (0..n).map(|column| {
                let product: f32 = row
                    .iter()
                    .zip(&weight[column * k..][..k])
                    .map(|(x, weight)| x * weight)
                    .sum();
                product + bias.map_or(0.0, |bias| bias[column])
            })
        })
        .collect()
}

const WHOLE: LinearConfig = LinearConfig {
    inputs: 4,
    outputs: 6,
    bias: true,
    parallel: Parallel::None,
};

// ---- Linear -----------------------------------------------------------------------------------

#[test]
fn a_dense_projection_loads_its_module_and_projects() {
    let directory = tempfile::tempdir().unwrap();
    let (weight, bias) = (sequence(6 * 4, 1), sequence(6, 2));
    let checkpoint = checkpoint(
        directory.path(),
        &[
            ("block.proj.weight", vec![6, 4], weight.clone()),
            ("block.proj.bias", vec![6], bias.clone()),
        ],
    );
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);

    let linear = Linear::load(&WHOLE, loader.root().pp("block").pp("proj")).unwrap();
    assert_eq!((linear.k(), linear.n(), linear.is_packed()), (4, 6, false));

    let x = sequence(3 * 4, 3);
    let out = linear.forward(&tensor(&x, (3, 4), DType::F32)).unwrap();
    assert_eq!(out.dims(), [3, 6]);
    assert_close(
        &values(&out),
        &project(&x, &weight, Some(&bias), 4),
        F32_TOLERANCE,
    );

    // The module is one input, so it composes as the tensor library's `Module`.
    let applied = tensor(&x, (3, 4), DType::F32).apply(&linear).unwrap();
    assert_eq!(values(&applied), values(&out));

    // A module the checkpoint does not hold is refused by its name.
    let missing = Linear::load(&WHOLE, loader.root().pp("block").pp("absent"));
    let text = missing.err().unwrap().to_string();
    assert!(text.contains("block.absent.weight"), "{text}");
}

#[test]
fn a_parallel_projection_loads_this_rank_s_block() {
    let directory = tempfile::tempdir().unwrap();
    let (weight, bias) = (sequence(6 * 4, 4), sequence(6, 5));
    let checkpoint = checkpoint(
        directory.path(),
        &[
            ("proj.weight", vec![6, 4], weight.clone()),
            ("proj.bias", vec![6], bias.clone()),
        ],
    );
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let split = Split { size: 2, rank: 1 };

    // Column parallel: rank 1 holds output channels 3..6 and their bias.
    let column = Linear::load(
        &LinearConfig {
            parallel: Parallel::Column(split),
            ..WHOLE
        },
        loader.root().pp("proj"),
    )
    .unwrap();
    assert_eq!((column.k(), column.n()), (4, 3));
    assert_eq!(values(column.weight()), weight[12..]);

    let x = sequence(2 * 4, 6);
    let out = column.forward(&tensor(&x, (2, 4), DType::F32)).unwrap();
    assert_close(
        &values(&out),
        &project(&x, &weight[12..], Some(&bias[3..]), 4),
        F32_TOLERANCE,
    );

    // Row parallel: rank 1 holds columns 2..4 of every row, and adds no bias, which rank 0
    // adds once.
    let row = Linear::load(
        &LinearConfig {
            parallel: Parallel::Row(split),
            ..WHOLE
        },
        loader.root().pp("proj"),
    )
    .unwrap();
    assert_eq!((row.k(), row.n()), (2, 6));
    let columns: Vec<f32> = weight.chunks(4).flat_map(|row| row[2..].to_vec()).collect();
    assert_eq!(values(row.weight()), columns);

    // A partial sum is reduced over the group, and a forward with no group refuses to return
    // one rank's share.
    let refused = row
        .forward(&tensor(&x[..4], (2, 2), DType::F32))
        .unwrap_err()
        .to_string();
    assert!(refused.contains("no group"), "{refused}");

    // A split that does not divide the split axis holds no share.
    let uneven = Linear::load(
        &LinearConfig {
            parallel: Parallel::Column(Split { size: 4, rank: 0 }),
            ..WHOLE
        },
        loader.root().pp("proj"),
    );
    assert!(uneven.is_err());
}

#[test]
fn a_row_parallel_projection_reduces_under_its_group() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = typed_checkpoint(
        directory.path(),
        DType::BF16,
        &[("proj.weight", vec![8, 32], sequence(8 * 32, 7))],
    );
    let loader = WeightLoader::meta(&checkpoint, None);
    let row = Linear::load(
        &LinearConfig {
            inputs: 32,
            outputs: 8,
            bias: false,
            parallel: Parallel::Row(Split { size: 2, rank: 1 }),
        },
        loader.root().pp("proj"),
    )
    .unwrap();

    let (output, launches) = recorded_split(Some((1, 2)), || {
        Ok(row.forward(&meta((4, 16), DType::BF16))?.dims().to_vec())
    });
    assert_eq!(output.unwrap(), [4, 8]);
    assert_eq!(operators(&launches), ["gemm"], "the reduction is no kernel");

    let (alone, _) = recorded(|| row.forward(&meta((4, 16), DType::BF16)));
    assert!(alone.unwrap_err().to_string().contains("no group"));
}

#[test]
fn several_modules_are_read_as_one_merged_projection() {
    let directory = tempfile::tempdir().unwrap();
    let (first, second) = (sequence(2 * 4, 8), sequence(3 * 4, 9));
    let (first_bias, second_bias) = (sequence(2, 10), sequence(3, 11));
    let checkpoint = checkpoint(
        directory.path(),
        &[
            ("heads.a.weight", vec![2, 4], first.clone()),
            ("heads.a.bias", vec![2], first_bias.clone()),
            ("heads.b.weight", vec![3, 4], second.clone()),
            ("heads.b.bias", vec![3], second_bias.clone()),
        ],
    );
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);

    let merged =
        Linear::load_merged(loader.root().pp("heads"), &["a", "b"], 4, &[2, 3], true).unwrap();
    assert_eq!((merged.k(), merged.n()), (4, 5));

    let x = sequence(4, 12);
    let out = merged.forward(&tensor(&x, (1, 4), DType::F32)).unwrap();
    let expected = [
        project(&x, &first, Some(&first_bias), 4),
        project(&x, &second, Some(&second_bias), 4),
    ]
    .concat();
    assert_close(&values(&out), &expected, F32_TOLERANCE);

    // The merged weight is filed under both modules' tensors, in the order named.
    assert!(
        store
            .keys()
            .iter()
            .any(|key| key.to_string().contains("heads.a.weight+heads.b.weight")),
        "{:?}",
        store.keys()
    );
}

/// Writes `config.json` beside the shards of `directory`, declaring `quantization`.
fn declare(directory: &Path, quantization: &str) {
    std::fs::write(
        directory.join("config.json"),
        format!(r#"{{"quantization_config": {quantization}}}"#),
    )
    .unwrap();
}

#[test]
fn a_per_tensor_fp8_projection_is_chosen_by_the_checkpoint_s_declaration() {
    let directory = tempfile::tempdir().unwrap();
    // E4M3 values, so the stored weight is exact: two rows of one vector of sixteen, the
    // contraction an FP8 product's kernels load whole.
    let weight: Vec<f32> = [0.5_f32, -1.25, 2.0, 0.375, 1.0, 1.5, -0.75, 3.0].repeat(4);
    shard(
        directory.path(),
        "model.safetensors",
        &[
            (
                "proj.weight",
                DType::F8E4M3,
                vec![2, 16],
                weight
                    .iter()
                    .map(|value| e4m3_byte(*value))
                    .collect::<Vec<u8>>(),
            ),
            ("proj.weight_scale", DType::F32, vec![1], f32_bytes(&[0.5])),
            ("proj.input_scale", DType::F32, vec![1], f32_bytes(&[0.25])),
        ],
    );
    declare(
        directory.path(),
        r#"{"quant_algo": "MIXED_PRECISION", "quantized_layers": {"proj": {"quant_algo": "FP8"}}}"#,
    );
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let config = LinearConfig {
        inputs: 16,
        outputs: 2,
        bias: false,
        parallel: Parallel::None,
    };

    let store = CpuWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None);
    loader
        .declare(&Declaration::read(&checkpoint).unwrap())
        .unwrap();
    let linear = Linear::load(&config, loader.root().pp("proj")).unwrap();
    assert_eq!(linear.weight().dtype(), DType::F8E4M3);

    // The module computes what the function computes over the fields it loaded: the activation
    // divided by its scale, and the product scaled by both.
    let x = tensor(&sequence(3 * 16, 13), (3, 16), DType::BF16);
    let expected = functional::linear_fp8(
        &x,
        &tensor(&weight, (2, 16), DType::F8E4M3),
        None,
        &tensor(&[1.0 / 0.25], 1, DType::F32),
        &tensor(&[0.25 * 0.5], 1, DType::F32),
    )
    .unwrap();
    assert_eq!(values(&linear.forward(&x).unwrap()), values(&expected));

    // A meta load chooses the same representation, and its forward states the FP8 product.
    let mut loader = WeightLoader::meta(&checkpoint, None);
    loader
        .declare(&Declaration::read(&checkpoint).unwrap())
        .unwrap();
    let linear = Linear::load(&config, loader.root().pp("proj")).unwrap();
    let (output, launches) = recorded(|| linear.forward(&meta((3, 16), DType::BF16)));
    output.unwrap();
    assert_eq!(operators(&launches), ["fp8_quantize", "fp8_gemm"]);
}

/// The E4M3 byte of `value`, one of the format's own values: sign, four exponent bits biased by
/// seven, three mantissa bits.
fn e4m3_byte(value: f32) -> u8 {
    let sign = if value < 0.0 { 0x80 } else { 0 };
    let magnitude = value.abs();
    let exponent = magnitude.log2().floor() as i32;
    let mantissa = ((magnitude / 2f32.powi(exponent) - 1.0) * 8.0) as u8;

    sign | (((exponent + 7) as u8) << 3) | mantissa
}

/// The checkpoint of one calibrated NVFP4 projection `proj`, `[8, 32]`, and the fields it was
/// written from: the packed bytes, the linear block scales' bytes, the weight's tensor scale
/// and the activation's.
fn nvfp4_checkpoint(directory: &Path) -> (Checkpoint, Vec<u8>, Vec<u8>, f32, f32) {
    let (n, k) = (8_usize, 32_usize);
    let packed: Vec<u8> = (0..n * k / 2)
        .map(|index| (index * 37 + 11) as u8)
        .collect();
    let scale_values = [1.0_f32, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.375];
    let scales: Vec<u8> = (0..n * k / 16)
        .map(|index| e4m3_byte(scale_values[index % scale_values.len()]))
        .collect();
    let (weight_scale, input_scale) = (0.5_f32, 0.25_f32);

    shard(
        directory,
        "model.safetensors",
        &[
            ("proj.weight", DType::U8, vec![n, k / 2], packed.clone()),
            (
                "proj.weight_scale",
                DType::F8E4M3,
                vec![n, k / 16],
                scales.clone(),
            ),
            (
                "proj.weight_scale_2",
                DType::F32,
                vec![1],
                f32_bytes(&[weight_scale]),
            ),
            (
                "proj.input_scale",
                DType::F32,
                vec![1],
                f32_bytes(&[input_scale]),
            ),
        ],
    );
    declare(
        directory,
        r#"{"quant_algo": "NVFP4", "quant_method": "modelopt"}"#,
    );

    (
        Checkpoint::open(directory).unwrap(),
        packed,
        scales,
        weight_scale,
        input_scale,
    )
}

const PACKED: LinearConfig = LinearConfig {
    inputs: 32,
    outputs: 8,
    bias: false,
    parallel: Parallel::None,
};

#[test]
fn a_calibrated_nvfp4_projection_loads_its_fields_in_the_product_s_layout() {
    let directory = tempfile::tempdir().unwrap();
    let (checkpoint, packed, scales, weight_scale, input_scale) =
        nvfp4_checkpoint(directory.path());

    let store = CpuWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None);
    loader
        .declare(&Declaration::read(&checkpoint).unwrap())
        .unwrap();
    let linear = Linear::load(&PACKED, loader.root().pp("proj")).unwrap();
    assert!(linear.is_packed());
    assert_eq!(
        (linear.weight().dims(), linear.weight().dtype()),
        (&[8, 16][..], DType::U8)
    );

    // The block scales are published in the 128x4 table the product reads, the encoder's scale
    // is the reciprocal of the activation's, and the alpha the two tensor scales' product.
    let keys: Vec<String> = store.keys().iter().map(ToString::to_string).collect();
    for piece in [
        "|u8|nvfp4_values|proj.weight[0..128][8, 16]",
        "|f8e4m3|nvfp4_scales_swizzled_128x4|proj.weight_scale__nvfp4_scales[0..512][128, 4]",
        "|f32|row_major|proj.input_scale__nvfp4_input_scale[0..1][1]",
        "|f32|row_major|proj.alpha[0..1][1]|computed:nn::Linear:",
    ] {
        assert!(
            keys.iter().any(|key| key.contains(piece)),
            "no key holds `{piece}`: {keys:#?}"
        );
    }

    // The module computes what the function computes over those fields, laid out here by the
    // table's own index rule.
    let mut table = vec![0_u8; 128 * 4];
    for row in 0..8 {
        for column in 0..2 {
            let index = (column % 4) + (column / 4) * 512 + (row % 32) * 16;
            table[index] = scales[row * 2 + column];
        }
    }
    let x = tensor(&sequence(3 * 32, 14), (3, 32), DType::BF16);
    let expected = functional::linear_nvfp4(
        &x,
        &Tensor::from_slice(&packed, (8, 16), &Device::Cpu).unwrap(),
        &Tensor::from_raw_buffer(&table, DType::F8E4M3, &[128, 4], &Device::Cpu).unwrap(),
        None,
        &tensor(&[1.0 / input_scale], 1, DType::F32),
        &tensor(&[weight_scale * input_scale], 1, DType::F32),
    )
    .unwrap();
    let out = linear.forward(&x).unwrap();
    assert_eq!(out.dims(), [3, 8]);
    assert_eq!(values(&out), values(&expected));

    // A bias the module is asked for and the checkpoint does not store is refused by its name.
    let biased = Linear::load(
        &LinearConfig {
            bias: true,
            ..PACKED
        },
        loader.root().pp("proj"),
    );
    assert!(biased.err().unwrap().to_string().contains("proj.bias"));
}

#[test]
fn a_meta_load_of_a_packed_projection_states_the_encoder_and_the_product() {
    let directory = tempfile::tempdir().unwrap();
    let (checkpoint, ..) = nvfp4_checkpoint(directory.path());

    let mut loader = WeightLoader::meta(&checkpoint, None);
    loader
        .declare(&Declaration::read(&checkpoint).unwrap())
        .unwrap();
    let linear = Linear::load(&PACKED, loader.root().pp("proj")).unwrap();
    assert!(linear.is_packed());

    let (output, launches) = recorded(|| linear.forward(&meta((5, 32), DType::BF16)));
    assert_eq!(output.unwrap().dims(), [5, 8]);
    assert_eq!(
        launches,
        [
            (
                "nvfp4_quantize",
                "input_bf16_columns_32_layout_swizzled_128x4_block_32_pdl_false"
            ),
            (
                "nvfp4_gemm",
                "packed_u8_scale_f8e4m3_accumulate_f32_output_bf16_layout_swizzled_128x4_\
                 epilogue_none_m_128_n_8_k_32"
            ),
        ]
        .map(|(operator, key)| (operator, key.to_owned()))
    );

    // A module with no encoded form refuses a weight the checkpoint stores packed.
    let dense = RmsNorm::load(
        &RmsNormConfig {
            width: 16,
            eps: 1e-6,
            centred: false,
            output: None,
        },
        loader.root().pp("proj"),
    );
    assert!(dense.err().unwrap().to_string().contains("packed"));
}

#[test]
fn a_projection_an_adapter_edits_is_loaded_fused() {
    let directory = tempfile::tempdir().unwrap();
    shard(
        directory.path(),
        "model.safetensors",
        &[
            (
                "proj.weight",
                DType::BF16,
                vec![2, 3],
                encode(DType::BF16, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
            (
                "proj.bias",
                DType::BF16,
                vec![2],
                encode(DType::BF16, &[1.0, 1.0]),
            ),
        ],
    );
    let adapters = tempfile::tempdir().unwrap();
    shard_stating(
        adapters.path(),
        "adapter_model.safetensors",
        &[
            ("format", "fastvideo-lora-v2"),
            ("rank", "1"),
            ("low_rank_tensors", "2"),
            ("diff_tensors", "1"),
            ("set_weight_tensors", "0"),
        ],
        &[
            (
                "proj.lora_A.weight",
                DType::BF16,
                vec![1, 3],
                encode(DType::BF16, &[1.0, 2.0, 3.0]),
            ),
            (
                "proj.lora_B.weight",
                DType::BF16,
                vec![2, 1],
                encode(DType::BF16, &[0.5, -1.0]),
            ),
            (
                "proj.diff_b",
                DType::BF16,
                vec![2],
                encode(DType::BF16, &[0.25, 0.25]),
            ),
        ],
    );
    let adapted = Checkpoint::open(directory.path())
        .unwrap()
        .attach(None, Adapter::open(adapters.path()).unwrap())
        .unwrap();

    let store = CpuWeights::new();
    let loader = WeightLoader::new(&adapted, &store, None);
    let linear = Linear::load(
        &LinearConfig {
            inputs: 3,
            outputs: 2,
            bias: true,
            parallel: Parallel::None,
        },
        loader.root().pp("proj"),
    )
    .unwrap();

    // `W + lora_B @ lora_A`: rows of [1, 2, 3] scaled by 0.5 and −1, added to the base.
    assert_eq!(values(linear.weight()), [1.5, 3.0, 4.5, 3.0, 3.0, 3.0]);
    let out = linear
        .forward(&tensor(&[1.0, 1.0, 1.0], (1, 3), DType::BF16))
        .unwrap();
    assert_eq!(values(&out), [9.0 + 1.25, 9.0 + 1.25]);
}

#[test]
fn an_attention_s_projections_load_separately_or_as_windows_of_one_module() {
    let directory = tempfile::tempdir().unwrap();
    let hidden = 8;
    let fused = sequence(24 * hidden, 15);
    let checkpoint = checkpoint(
        directory.path(),
        &[
            (
                "attn.q_proj.weight",
                vec![16, hidden],
                sequence(16 * hidden, 16),
            ),
            (
                "attn.k_proj.weight",
                vec![4, hidden],
                sequence(4 * hidden, 17),
            ),
            (
                "attn.v_proj.weight",
                vec![4, hidden],
                sequence(4 * hidden, 18),
            ),
            ("fused.qkv.weight", vec![24, hidden], fused.clone()),
            ("fused.qkv.bias", vec![24], sequence(24, 19)),
        ],
    );
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let config = QkvLinearConfig {
        hidden: hidden as u32,
        heads: 4,
        kv_heads: 1,
        head_dim: 4,
        gated: false,
        names: QkvNames::Separate(["q_proj", "k_proj", "v_proj"]),
        bias: QkvBias::Absent,
        split: Split { size: 2, rank: 1 },
    };

    // A split wider than the key/value heads replicates the one head on both ranks, and halves
    // the query heads.
    let separate = QkvLinear::load(&config, loader.root().pp("attn")).unwrap();
    assert_eq!(
        (
            separate.query().n(),
            separate.key().n(),
            separate.value().n()
        ),
        (8, 4, 4)
    );

    // The fused module's rows are the query's, then the key's, then the value's.
    let windows = QkvLinear::load(
        &QkvLinearConfig {
            names: QkvNames::Fused("qkv"),
            bias: QkvBias::Present,
            split: Split::WHOLE,
            ..config
        },
        loader.root().pp("fused"),
    )
    .unwrap();
    assert_eq!(values(windows.query().weight()), fused[..16 * hidden]);
    assert_eq!(
        values(windows.key().weight()),
        fused[16 * hidden..20 * hidden]
    );
    assert_eq!(values(windows.value().weight()), fused[20 * hidden..]);

    let uneven = QkvLinear::load(
        &QkvLinearConfig {
            split: Split { size: 3, rank: 0 },
            ..config
        },
        loader.root().pp("attn"),
    );
    assert!(uneven.is_err());
}

// ---- Embedding and head -----------------------------------------------------------------------

#[test]
fn an_embedding_gathers_its_table_s_rows() {
    let directory = tempfile::tempdir().unwrap();
    let table = sequence(8 * 3, 20);
    let checkpoint = checkpoint(
        directory.path(),
        &[("embed_tokens.weight", vec![8, 3], table.clone())],
    );
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let config = EmbeddingConfig {
        vocabulary: 8,
        hidden: 3,
        split: Split::WHOLE,
    };

    let embedding = Embedding::load(&config, loader.root().pp("embed_tokens")).unwrap();
    assert_eq!((embedding.shard(), embedding.is_sharded()), ((0, 8), false));

    let ids = Tensor::new(&[5_u32, 0, 7], &Device::Cpu).unwrap();
    let rows = embedding.forward(&ids).unwrap();
    let expected = [&table[15..18], &table[0..3], &table[21..24]].concat();
    assert_eq!(values(&rows), expected);

    // Rank 1 of 2 holds rows 4..8, and its rows are summed over the group.
    let shard = Embedding::load(
        &EmbeddingConfig {
            split: Split { size: 2, rank: 1 },
            ..config
        },
        loader.root().pp("embed_tokens"),
    )
    .unwrap();
    assert_eq!((shard.shard(), shard.is_sharded()), ((4, 4), true));
    assert_eq!(values(shard.weight()), table[12..]);
    assert!(shard.forward(&ids).is_err(), "a shard needs its group");
}

#[test]
fn a_sharded_embedding_states_its_shard_in_its_kernel() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = typed_checkpoint(
        directory.path(),
        DType::BF16,
        &[("embed_tokens.weight", vec![32, 8], sequence(32 * 8, 21))],
    );
    let loader = WeightLoader::meta(&checkpoint, None);
    let shard = Embedding::load(
        &EmbeddingConfig {
            vocabulary: 32,
            hidden: 8,
            split: Split { size: 2, rank: 1 },
        },
        loader.root().pp("embed_tokens"),
    )
    .unwrap();

    let (output, launches) = recorded_split(Some((1, 2)), || shard.forward(&meta(5, DType::U32)));
    assert_eq!(output.unwrap().dims(), [5, 8]);
    assert_eq!(
        launches,
        [(
            "embedding",
            "dtype_bf16_ids_u32_hidden_8_first_16_rows_16_block_8_pdl_false".to_owned()
        )]
    );
}

/// A checkpoint whose `lm_head.weight` is `[vocabulary, 8]` BF16.
fn head_checkpoint(directory: &Path, vocabulary: usize) -> Checkpoint {
    typed_checkpoint(
        directory,
        DType::BF16,
        &[(
            "lm_head.weight",
            vec![vocabulary, 8],
            sequence(vocabulary * 8, 22),
        )],
    )
}

#[test]
fn a_head_projects_onto_the_vocabulary_and_widens_the_logits() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = head_checkpoint(directory.path(), 16);
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let head = LmHead::load(
        &LmHeadConfig {
            vocabulary: 16,
            hidden: 8,
            split: Split::WHOLE,
        },
        loader.root().pp("lm_head"),
    )
    .unwrap();

    let hidden = tensor(&sequence(3 * 8, 23), (3, 8), DType::BF16);
    let logits = head.forward(&hidden).unwrap();
    assert_eq!((logits.dims(), logits.dtype()), (&[3, 16][..], DType::F32));

    // The model's product, rounded into its own type and then widened.
    let weight = values(&tensor(&sequence(16 * 8, 22), (16, 8), DType::BF16));
    let product = project(&values(&hidden), &weight, None, 8);
    assert_close(
        &values(&logits),
        &values(&tensor(&product, (3, 16), DType::BF16)),
        1.0 / 128.0,
    );
}

/// A meta forward split as rank 1 of 2 takes the same gather and states the same unpad kernel
/// as that rank's launching forward.
#[test]
fn a_split_head_gathers_the_shards_of_its_group() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = head_checkpoint(directory.path(), 32);
    let loader = WeightLoader::meta(&checkpoint, None);

    // Rank 1 holds rows 16..32, a whole number of the head's eight-row tiles.
    let head = LmHead::load(
        &LmHeadConfig {
            vocabulary: 32,
            hidden: 8,
            split: Split { size: 2, rank: 1 },
        },
        loader.root().pp("lm_head"),
    )
    .unwrap();
    assert_eq!(head.padded(), 16);

    let (output, launches) = recorded_split(Some((1, 2)), || {
        let logits = head.forward(&meta((3, 8), DType::BF16))?;
        Ok((logits.dims().to_vec(), logits.dtype()))
    });
    assert_eq!(output.unwrap(), (vec![3, 32], DType::F32));
    assert_eq!(operators(&launches), ["gemm", "vocab_unpad"]);
    assert!(
        launches[1]
            .1
            .starts_with("dtype_bf16_output_f32_vocabulary_32_padded_32_"),
        "{launches:?}"
    );

    // Unsplit, the shard's 16 columns are the whole gathered width and cannot hold a vocabulary
    // of 32: the unpad refuses the call, as the device would.
    let (unsplit, _) = recorded(|| head.forward(&meta((3, 8), DType::BF16)));
    assert!(unsplit.is_err());
}

/// A whole head in a forward whose group splits something else, a pipeline's stages or a
/// sequence's members, projects onto its whole vocabulary and gathers nothing: it launches what
/// it launches outside any group.
#[test]
fn a_whole_head_gathers_nothing_in_a_group_that_splits_something_else() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = head_checkpoint(directory.path(), 32);
    let loader = WeightLoader::meta(&checkpoint, None);
    let head = LmHead::load(
        &LmHeadConfig {
            vocabulary: 32,
            hidden: 8,
            split: Split::WHOLE,
        },
        loader.root().pp("lm_head"),
    )
    .unwrap();

    let forward = || {
        let logits = head.forward(&meta((3, 8), DType::BF16))?;
        Ok((logits.dims().to_vec(), logits.dtype()))
    };
    let (grouped, in_group) = recorded_split(Some((1, 2)), forward);
    let (alone, outside) = recorded_split(None, forward);

    assert_eq!(grouped.unwrap(), (vec![3, 32], DType::F32));
    assert_eq!(alone.unwrap(), (vec![3, 32], DType::F32));
    assert_eq!(in_group, outside);
}

/// A shard that is not a whole number of tiles has no padded rows in the checkpoint to read:
/// the next rank's rows follow it, so the head is refused at load.
#[test]
fn a_shard_of_a_partial_tile_is_refused_at_load() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = head_checkpoint(directory.path(), 30);
    let loader = WeightLoader::meta(&checkpoint, None);

    let refused = LmHead::load(
        &LmHeadConfig {
            vocabulary: 30,
            hidden: 8,
            split: Split { size: 2, rank: 1 },
        },
        loader.root().pp("lm_head"),
    )
    .err()
    .expect("fifteen rows");
    assert!(refused.to_string().contains("15"), "{refused}");
}

#[test]
fn a_tied_head_projects_with_the_embedding_s_table() {
    let directory = tempfile::tempdir().unwrap();
    let table = sequence(8 * 4, 24);
    let checkpoint = checkpoint(
        directory.path(),
        &[("embed_tokens.weight", vec![8, 4], table.clone())],
    );
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let embedding = Embedding::load(
        &EmbeddingConfig {
            vocabulary: 8,
            hidden: 4,
            split: Split::WHOLE,
        },
        loader.root().pp("embed_tokens"),
    )
    .unwrap();

    let head = LmHead::tied(&embedding).unwrap();
    let hidden = sequence(2 * 4, 25);
    let logits = head.forward(&tensor(&hidden, (2, 4), DType::F32)).unwrap();
    assert_close(
        &values(&logits),
        &project(&hidden, &table, None, 4),
        F32_TOLERANCE,
    );
}

// ---- Normalization ----------------------------------------------------------------------------

#[test]
fn the_normalizations_load_their_vectors_and_normalize() {
    let directory = tempfile::tempdir().unwrap();
    let (weight, bias) = (sequence(6, 26), sequence(6, 27));
    let checkpoint = checkpoint(
        directory.path(),
        &[
            ("norm.weight", vec![6], weight.clone()),
            ("norm.bias", vec![6], bias.clone()),
        ],
    );
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let x = tensor(&sequence(2 * 6, 28), (2, 6), DType::F32);
    let weight_tensor = tensor(&weight, 6, DType::F32);

    let rms = RmsNorm::load(
        &RmsNormConfig {
            width: 6,
            eps: 1e-6,
            centred: true,
            output: None,
        },
        loader.root().pp("norm"),
    )
    .unwrap();
    assert_eq!(values(rms.weight()), weight);
    assert_eq!(
        values(&rms.forward(&x).unwrap()),
        values(&functional::rms_norm(&x, &weight_tensor, 1e-6, true, DType::F32).unwrap())
    );

    // The residual form returns the normalized sum and the sum.
    let residual = tensor(&sequence(2 * 6, 29), (2, 6), DType::F32);
    let (normalized, sum) = rms.forward_residual(&x, &residual).unwrap();
    assert_eq!(values(&sum), values(&(&x + &residual).unwrap()));
    assert_eq!(values(&normalized), values(&rms.forward(&sum).unwrap()));

    let layer = LayerNorm::load(
        &LayerNormConfig {
            width: 6,
            eps: 1e-5,
            bias: true,
        },
        loader.root().pp("norm"),
    )
    .unwrap();
    let expected = functional::layer_norm(
        &x,
        &weight_tensor,
        Some(&tensor(&bias, 6, DType::F32)),
        1e-5,
    )
    .unwrap();
    assert_eq!(values(&layer.forward(&x).unwrap()), values(&expected));

    let gated = GatedRmsNorm::load(
        &GatedRmsNormConfig {
            width: 6,
            eps: 1e-6,
            sigmoid: false,
        },
        loader.root().pp("norm"),
    )
    .unwrap();
    assert_eq!(
        values(&gated.forward(&x, &residual).unwrap()),
        values(&functional::gated_rms_norm(&x, &residual, &weight_tensor, 1e-6, false).unwrap())
    );

    // A weight of another width than the module states is the checkpoint's disagreement.
    let wider = RmsNorm::load(
        &RmsNormConfig {
            width: 8,
            eps: 1e-6,
            centred: false,
            output: None,
        },
        loader.root().pp("norm"),
    );
    assert!(wider.is_err());
}

#[test]
fn a_narrowing_normalization_states_the_narrowing_kernel_and_takes_no_residual() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        &[("norm.weight", vec![8], sequence(8, 30))],
    );
    let loader = WeightLoader::meta(&checkpoint, None);
    let norm = RmsNorm::load(
        &RmsNormConfig {
            width: 8,
            eps: 1e-6,
            centred: false,
            output: Some(DType::BF16),
        },
        loader.root().pp("norm"),
    )
    .unwrap();

    let (output, launches) = recorded(|| {
        let stream = meta((4, 8), DType::F32);
        let narrowed = norm.forward(&stream)?;
        let refused = norm.forward_residual(&stream, &stream);
        Ok((narrowed.dtype(), refused.is_err()))
    });
    assert_eq!(output.unwrap(), (DType::BF16, true));
    assert_eq!(operators(&launches), ["weighted_rms_norm"]);
}

// ---- Gated MLP --------------------------------------------------------------------------------

const HIDDEN: usize = 4;
const INTERMEDIATE: usize = 6;

fn mlp_config(rounding: Rounding, gate_up: GateUp) -> MlpConfig {
    MlpConfig {
        hidden: HIDDEN as u32,
        intermediate: INTERMEDIATE as u32,
        activation: Activation::Silu,
        rounding,
        gate_up,
        down: "down_proj",
        bias: false,
        split: Split::WHOLE,
    }
}

const SPLIT: GateUp = GateUp::Split {
    gate: "gate_proj",
    up: "up_proj",
};

#[test]
fn a_gated_mlp_projects_gates_and_projects_back() {
    let directory = tempfile::tempdir().unwrap();
    let (gate, up, down) = (
        sequence(INTERMEDIATE * HIDDEN, 31),
        sequence(INTERMEDIATE * HIDDEN, 32),
        sequence(HIDDEN * INTERMEDIATE, 33),
    );
    // The packed module's rows hold the value (up) half and then the gate half.
    let packed = [up.clone(), gate.clone()].concat();
    let checkpoint = checkpoint(
        directory.path(),
        &[
            (
                "mlp.gate_proj.weight",
                vec![INTERMEDIATE, HIDDEN],
                gate.clone(),
            ),
            ("mlp.up_proj.weight", vec![INTERMEDIATE, HIDDEN], up.clone()),
            (
                "mlp.down_proj.weight",
                vec![HIDDEN, INTERMEDIATE],
                down.clone(),
            ),
            ("ff.proj.weight", vec![2 * INTERMEDIATE, HIDDEN], packed),
            (
                "ff.down_proj.weight",
                vec![HIDDEN, INTERMEDIATE],
                down.clone(),
            ),
        ],
    );
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);

    let x = sequence(3 * HIDDEN, 34);
    let gated: Vec<f32> = project(&x, &gate, None, HIDDEN)
        .iter()
        .zip(project(&x, &up, None, HIDDEN))
        .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
        .collect();
    let expected = project(&gated, &down, None, INTERMEDIATE);

    let input = tensor(&x, (3, HIDDEN), DType::F32);
    for (rounding, gate_up, module) in [
        (Rounding::Fused, SPLIT, "mlp"),
        (Rounding::Stepwise, SPLIT, "mlp"),
        (Rounding::ValueProduct, SPLIT, "mlp"),
        (Rounding::Stepwise, GateUp::ValueFirst("proj"), "ff"),
        (Rounding::ValueProduct, GateUp::ValueFirst("proj"), "ff"),
    ] {
        let mlp = Mlp::load(&mlp_config(rounding, gate_up), loader.root().pp(module)).unwrap();
        let out = mlp.forward(&input).unwrap();

        assert_eq!(out.dims(), [3, HIDDEN], "{rounding:?} {gate_up:?}");
        assert_close(&values(&out), &expected, F32_TOLERANCE);
    }

    // The value-product order is SiLU's alone.
    let refused = Mlp::load(
        &MlpConfig {
            activation: Activation::GeluTanh,
            ..mlp_config(Rounding::ValueProduct, SPLIT)
        },
        loader.root().pp("mlp"),
    );
    assert!(refused.is_err());
}

#[test]
fn a_gated_mlp_states_the_gate_its_rounding_names() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = typed_checkpoint(
        directory.path(),
        DType::BF16,
        &[
            (
                "mlp.gate_proj.weight",
                vec![INTERMEDIATE, HIDDEN],
                sequence(INTERMEDIATE * HIDDEN, 35),
            ),
            (
                "mlp.up_proj.weight",
                vec![INTERMEDIATE, HIDDEN],
                sequence(INTERMEDIATE * HIDDEN, 36),
            ),
            (
                "mlp.down_proj.weight",
                vec![HIDDEN, INTERMEDIATE],
                sequence(HIDDEN * INTERMEDIATE, 37),
            ),
            (
                "ff.proj.weight",
                vec![2 * INTERMEDIATE, HIDDEN],
                sequence(2 * INTERMEDIATE * HIDDEN, 38),
            ),
            (
                "ff.down_proj.weight",
                vec![HIDDEN, INTERMEDIATE],
                sequence(HIDDEN * INTERMEDIATE, 39),
            ),
        ],
    );
    let loader = WeightLoader::meta(&checkpoint, None);
    let launched = |rounding, gate_up, module: &str, rows: usize| {
        let mlp = Mlp::load(&mlp_config(rounding, gate_up), loader.root().pp(module)).unwrap();
        let (output, launches) = recorded(|| {
            Ok(mlp
                .forward(&meta((rows, HIDDEN), DType::BF16))?
                .dims()
                .to_vec())
        });

        assert_eq!(output.unwrap(), [rows, HIDDEN]);
        operators(&launches)
    };

    // A split MLP projects twice, gates at its rounding and projects back.
    assert_eq!(
        launched(Rounding::ValueProduct, SPLIT, "mlp", 5),
        ["gemm", "gemm", "swiglu", "gemm"]
    );
    assert_eq!(
        launched(Rounding::Fused, SPLIT, "mlp", 5),
        ["gemm", "gemm", "act_and_mul", "gemm"]
    );

    // A value-first projection is read once and gated from its two halves.
    let value_first = GateUp::ValueFirst("proj");
    assert_eq!(
        launched(Rounding::Stepwise, value_first, "ff", 3),
        ["gemm", "act_and_mul", "gemm"]
    );
    assert_eq!(
        launched(Rounding::ValueProduct, value_first, "ff", 3),
        ["gemm", "value_first_swiglu", "gemm"]
    );
}

#[test]
fn a_dense_projection_states_one_product_whatever_its_weight_order() {
    let (output, launches) = recorded(|| {
        let x = meta((4, 16), DType::BF16);
        let stored = Linear::new(meta((8, 16), DType::BF16), None, WeightOrder::NxK)?;
        let computed = Linear::new(
            meta((16, 8), DType::BF16),
            Some(meta(8, DType::BF16)),
            WeightOrder::KxN,
        )?;

        Ok([stored.forward(&x)?, computed.forward(&x)?].map(|out| out.dims().to_vec()))
    });

    assert_eq!(output.unwrap(), [vec![4, 8], vec![4, 8]]);
    assert_eq!(
        launches,
        [
            (
                "gemm",
                "input_bf16_weight_bf16_accumulate_f32_output_bf16_order_nxk_epilogue_none"
            ),
            (
                "gemm",
                "input_bf16_weight_bf16_accumulate_f32_output_bf16_order_kxn_epilogue_bias"
            ),
        ]
        .map(|(operator, key)| (operator, key.to_owned()))
    );
}
