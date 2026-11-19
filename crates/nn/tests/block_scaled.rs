//! A block-scaled FP8 export at the loading boundary: the `fp8` producer's declaration read and
//! resolved against its shards, and its MXFP8 projections loaded and computed on the CPU device
//! against the arithmetic the scales and codes state.
//!
//! The export is written here, as DeepSeek's `fp8` producer writes one: a projection's `weight`
//! of E4M3 codes with a `scale` of one UE8M0 byte per 32 × 32 block, a table of E4M3 rows with a
//! scale per 32 elements of each row, an MXFP4 expert's packed codes with one scale per 32
//! elements of each row, and a dense BF16 router beside them. Nothing here touches a GPU.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use std::path::Path;

use nn::modules::{Linear, LinearConfig, Parallel};
use nn::quantization::{Declaration, Mxfp8Part, Scheme};
use nn::{Checkpoint, CpuWeights, DType, WeightLoader};
use nn_testing::{tensor, values, write_checkpoint};
use ops::mx::GemmDescriptor;

/// The value of a UE8M0 scale byte.
fn scale(byte: u8) -> f32 {
    2f32.powi(i32::from(byte) - 127)
}

/// The value of an E4M3 byte by the format's definition: a sign, four exponent bits biased by
/// seven and three mantissa bits, subnormal at exponent zero.
fn e4m3_value(byte: u8) -> f32 {
    let (exponent, mantissa) = (i32::from((byte >> 3) & 0x0f), f32::from(byte & 0x07));
    let magnitude = if exponent == 0 {
        mantissa / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    };
    if byte & 0x80 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// The nearest E4M3 value to `x`, ties to the even code, within ±448.
fn e4m3(x: f32) -> f32 {
    let grid: Vec<f32> = (0..=0x7e).map(e4m3_value).collect();
    let magnitude = x.abs().min(448.0);
    let index = (0..grid.len())
        .min_by(|left, right| {
            let (a, b) = (
                (grid[*left] - magnitude).abs(),
                (grid[*right] - magnitude).abs(),
            );
            a.partial_cmp(&b)
                .unwrap()
                .then((left % 2).cmp(&(right % 2)))
        })
        .unwrap();
    grid[index].copysign(x)
}

/// `x` encoded into MXFP8 and decoded again, as the encoding states it: per 32 elements, the
/// least power of two at or above `max(amax, 1e-4) / 448` scales E4M3 codes of the quotients.
fn quantized(x: &[f32]) -> Vec<f32> {
    x.chunks(32)
        .flat_map(|block| {
            let amax = block.iter().fold(1e-4_f32, |amax, x| amax.max(x.abs()));
            let unit = 2f32.powi((amax * (1.0 / 448.0)).log2().ceil() as i32);
            block.iter().map(move |x| e4m3(x / unit) * unit)
        })
        .collect()
}

/// The writer's tensors of a block-scaled export and its `config.json`.
fn export(directory: &Path, declaration: serde_json::Value) {
    let (n, k) = (64_usize, 256_usize);
    // Codes of every magnitude and sign, and block scales between 2⁻⁸ and 2⁻².
    let codes: Vec<u8> = (0..n * k)
        .map(|index| (index * 37 % 251) as u8 & 0xfe)
        .map(|byte| if byte & 0x7f == 0x7f { 0x30 } else { byte })
        .collect();
    let blocks: Vec<u8> = (0..(n / 32) * (k / 32))
        .map(|index| 119 + (index * 3 % 7) as u8)
        .collect();
    let rows: Vec<u8> = (0..n * (k / 32))
        .map(|index| 120 + (index % 5) as u8)
        .collect();
    let packed: Vec<u8> = (0..n * k / 2)
        .map(|index| (index * 29 % 256) as u8)
        .collect();
    let router = nn_testing::encode(DType::BF16, &vec![0.25; 8 * k]);

    write_checkpoint(
        directory,
        &[&[
            ("proj.weight", DType::F8E4M3, vec![n, k], codes.clone()),
            ("proj.scale", DType::U8, vec![n / 32, k / 32], blocks),
            ("table.weight", DType::F8E4M3, vec![n, k], codes),
            ("table.scale", DType::U8, vec![n, k / 32], rows.clone()),
            ("experts.0.w1.weight", DType::U8, vec![n, k / 2], packed),
            ("experts.0.w1.scale", DType::U8, vec![n, k / 32], rows),
            ("gate.weight", DType::BF16, vec![8, k], router),
        ]],
    );
    std::fs::write(
        directory.join("config.json"),
        serde_json::json!({ "quantization_config": declaration }).to_string(),
    )
    .unwrap();
}

/// DeepSeek-V4.1-Flash's declaration.
fn deepseek() -> serde_json::Value {
    serde_json::json!({
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    })
}

/// The shards decide each weight's scheme: E4M3 under 32 × 32 blocks or per-row scales is MXFP8,
/// packed bytes under per-row scales MXFP4, and a weight without a scale is dense.
#[test]
fn the_shards_decide_each_weights_mx_scheme() {
    let directory = tempfile::tempdir().unwrap();
    export(directory.path(), deepseek());
    let checkpoint = Checkpoint::open(directory.path()).unwrap();

    let representation = Declaration::read(&checkpoint)
        .unwrap()
        .resolve(&checkpoint)
        .unwrap();
    for (weight, scheme) in [
        ("proj.weight", Scheme::Mxfp8),
        ("table.weight", Scheme::Mxfp8),
        ("experts.0.w1.weight", Scheme::Mxfp4),
        ("gate.weight", Scheme::Dense),
    ] {
        assert_eq!(representation.scheme(weight), scheme, "{weight}");
    }
    assert_eq!(
        representation.block_scale("proj.weight").as_deref(),
        Some("proj.scale")
    );
    assert_eq!(representation.block_scale("gate.weight"), None);
}

/// A declaration the MX products cannot read is refused by the field that states it.
#[test]
fn a_block_scaled_declaration_no_operator_reads_is_refused_by_its_field() {
    for (field, value) in [
        ("weight_block_size", serde_json::json!([128, 128])),
        ("scale_fmt", serde_json::Value::Null),
        ("activation_scheme", serde_json::json!("static")),
        ("expert_dtype", serde_json::json!("int4")),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let mut declaration = deepseek();
        declaration[field] = value;
        export(directory.path(), declaration);
        let checkpoint = Checkpoint::open(directory.path()).unwrap();

        let error = Declaration::read(&checkpoint).unwrap_err().to_string();
        assert!(error.contains(field), "{field}: {error}");
    }
}

/// Packed bytes beside a scale are refused when the declaration states no FP4 experts: they would
/// be read as a matrix of another format.
#[test]
fn packed_bytes_without_declared_fp4_experts_are_refused() {
    let directory = tempfile::tempdir().unwrap();
    let mut declaration = deepseek();
    declaration.as_object_mut().unwrap().remove("expert_dtype");
    export(directory.path(), declaration);
    let checkpoint = Checkpoint::open(directory.path()).unwrap();

    let error = Declaration::read(&checkpoint)
        .unwrap()
        .resolve(&checkpoint)
        .unwrap_err()
        .to_string();
    assert!(error.contains("experts.0.w1.weight"), "{error}");
}

/// An MXFP8 projection on the CPU device is the product of the encoded activation and the weight
/// its codes and block scales state, each block's scale repeated over its rows: whole, and as a
/// column-parallel rank's rows and a row-parallel rank's columns.
#[test]
fn an_mxfp8_projection_is_the_product_its_codes_and_scales_state() {
    let directory = tempfile::tempdir().unwrap();
    export(directory.path(), deepseek());
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let store = CpuWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None);
    loader
        .declare(&Declaration::read(&checkpoint).unwrap())
        .unwrap();

    let (n, k) = (64_usize, 256_usize);
    let codes = checkpoint.read("proj.weight").unwrap();
    let blocks = checkpoint.read("proj.scale").unwrap();
    let weight: Vec<f32> = (0..n * k)
        .map(|index| {
            let (row, column) = (index / k, index % k);
            e4m3_value(codes[index]) * scale(blocks[(row / 32) * (k / 32) + column / 32])
        })
        .collect();
    let x: Vec<f32> = (0..3 * k)
        .map(|index| {
            ((index * 7919 % 1000) as f32 / 125.0 - 4.0) * 2f32.powi((index / 32 % 9) as i32 - 4)
        })
        .collect();
    let x = values(&tensor(&x, (3, k), DType::BF16));
    let encoded = quantized(&x);

    // Whole, and as a column-parallel rank's rows, through the projection; a row-parallel rank's
    // columns through the loader and the product, since its forward reduces over a group the CPU
    // device has none of.
    let mut cases = Vec::new();
    for (parallel, rows) in [
        (Parallel::None, 0..n),
        (Parallel::Column(nn::Split::new(2, 1).unwrap()), 32..64),
    ] {
        let linear = Linear::load(
            &LinearConfig {
                inputs: k as u32,
                outputs: n as u32,
                bias: false,
                parallel,
            },
            loader.root().pp("proj"),
        )
        .unwrap();
        assert_eq!(linear.dtype(), Some(DType::BF16));
        let out = values(&linear.forward(&tensor(&x, (3, k), DType::BF16)).unwrap());
        cases.push((format!("{parallel:?}"), rows, 0..k, out));
    }
    let share = loader
        .mxfp8(
            "proj.weight",
            &[Mxfp8Part {
                weight: "proj.weight".to_owned(),
                rows: 0..n as u64,
                columns: 128..256,
            }],
            GemmDescriptor::SCALE_LAYOUT,
        )
        .unwrap();
    let input: Vec<f32> = (0..3)
        .flat_map(|row| x[row * k + 128..row * k + 256].to_vec())
        .collect();
    let out = values(
        &nn::functional::linear_mxfp8(&tensor(&input, (3, 128), DType::BF16), &share, None)
            .unwrap(),
    );
    cases.push(("a row share".to_owned(), 0..n, 128..256, out));

    for (case, rows, columns, out) in cases {
        let width = columns.len();
        // The share's columns are whole 32-element blocks, so their encoding is the row's.
        let encoded_input: Vec<f32> = (0..3)
            .flat_map(|row| encoded[row * k + columns.start..row * k + columns.end].to_vec())
            .collect();
        for row in 0..3 {
            for (index, output) in rows.clone().enumerate() {
                let exact: f64 = (0..width)
                    .map(|column| {
                        f64::from(encoded_input[row * width + column])
                            * f64::from(weight[output * k + columns.start + column])
                    })
                    .sum();
                let got = f64::from(out[row * rows.len() + index]);
                // One rounding of the FP32 sum into BF16.
                assert!(
                    (got - exact).abs() <= exact.abs() * 2f64.powi(-8) + 1e-6,
                    "{case} [{row}, {output}]: {got} vs {exact}"
                );
            }
        }
    }
}
