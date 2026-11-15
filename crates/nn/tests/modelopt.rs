//! A packed ModelOpt export at the loading boundary: its manifest read into a declaration, and
//! the projections it packs loaded in the diffusers spelling and computed on the CPU device
//! against the arithmetic their fields and the manifest state.
//!
//! The export is written here: a shard of a few packed projections (`weight_packed`,
//! `weight_scale`, `weight_tensor_scale`, one with a `bias`) beside a dense one, and a
//! `modelopt_manifest.json` that lists each packed projection under another module tree's
//! paths. Nothing here touches a GPU.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use std::path::Path;

use nn::modules::{Linear, LinearConfig, Parallel};
use nn::quantization::{Declaration, ModeloptManifest};
use nn::{Checkpoint, CpuWeights, DType, Error, MetaWeights, WeightLoader};
use nn_testing::{
    Recording, encode, meta, operators, record, sequence, tensor, values, write_checkpoint,
};

/// The largest magnitudes an activation's calibration measured, as the manifest lists them:
/// the two halves of the fused projection (which agree), the projection back, and the biased
/// projection.
const FUSED_AMAX: f64 = 485.011_444_091_796_9;
const DOWN_AMAX: f64 = 602.315_246_582_031_2;
const OUTPUT_AMAX: f64 = 37.969_49;

/// The magnitudes an E2M1 code states; a code's fourth bit is its sign.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

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

/// The nearest value of `grid` to `magnitude`, a tie going to the entry whose index is even:
/// both formats order their codes so that an even code has an even mantissa.
fn nearest(grid: &[f32], magnitude: f32) -> f32 {
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

    grid[index]
}

/// `x`, the rows of an activation in blocks of sixteen, encoded under the tensor scale
/// `activation_scale` and decoded again, as the encoding states it:
///
/// ```text
/// scale = e4m3(amax(block) / 6 / activation_scale)      the block's scale, at most 448
/// x̂     = e2m1(x / (scale · activation_scale)) · scale · activation_scale
/// ```
fn quantized(x: &[f32], activation_scale: f32) -> Vec<f32> {
    // Every non-negative E4M3 value, in the order of its byte.
    let scales: Vec<f32> = (0..=0x7e).map(e4m3_value).collect();

    x.chunks(16)
        .flat_map(|block| {
            let amax = block.iter().fold(0.0_f32, |amax, x| amax.max(x.abs()));
            let scale = nearest(&scales, (amax / 6.0 / activation_scale).min(448.0));
            let unit = scale * activation_scale;

            block
                .iter()
                .map(|x| {
                    if unit == 0.0 {
                        0.0
                    } else {
                        nearest(&E2M1, (x / unit).abs().min(6.0)).copysign(*x) * unit
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The dense `[n, k]` matrix the packed fields state: `code · block scale · tensor scale`, two
/// codes a byte with the even element in the low nibble, one scale per sixteen elements of a
/// row.
fn decoded(packed: &[u8], scales: &[u8], tensor_scale: f32) -> Vec<f32> {
    (0..packed.len() * 2)
        .map(|element| {
            let byte = packed[element / 2];
            let code = if element % 2 == 0 {
                byte & 0x0f
            } else {
                byte >> 4
            };
            let sign = if code & 0x08 == 0 { 1.0 } else { -1.0 };

            sign * E2M1[usize::from(code & 0x07)] * e4m3_value(scales[element / 16]) * tensor_scale
        })
        .collect()
}

/// `x [m, k] · weightᵀ` for a weight stored `[n, k]`, accumulated in FP64: each element's
/// product, and the sum of its terms' magnitudes, which bounds what another order of
/// accumulation changes.
fn project(x: &[f32], weight: &[f32], k: usize) -> Vec<(f64, f64)> {
    let n = weight.len() / k;

    x.chunks(k)
        .flat_map(|row| {
            (0..n).map(|column| {
                let terms = row
                    .iter()
                    .zip(&weight[column * k..][..k])
                    .map(|(x, weight)| f64::from(*x) * f64::from(*weight));

                (terms.clone().sum(), terms.map(f64::abs).sum())
            })
        })
        .collect()
}

/// The packed bytes and the block-scale bytes of an `[n, k]` projection, different for each
/// `seed`: codes of every magnitude and sign, and scales between a quarter and twelve.
fn fields(n: usize, k: usize, seed: usize) -> (Vec<u8>, Vec<u8>) {
    let packed = (0..n * k / 2)
        .map(|index| (index * 37 + 11 + seed * 53) as u8)
        .collect();

    // Bytes 0x28..=0x53 are the E4M3 values from 0.25 to 12.
    let scales = (0..n * k / 16)
        .map(|index| 0x28 + ((index * 5 + seed) % 44) as u8)
        .collect();

    (packed, scales)
}

/// The module names the synthetic export stores its projections under.
const FUSED: &str = "blocks.0.ff.net.0.proj";
const DOWN: &str = "blocks.0.ff.net.2";
const OUTPUT: &str = "blocks.0.attn.to_out.0";
const DENSE: &str = "blocks.0.proj";

/// The tensor scales of the three packed weights and the bias of the biased one, whose
/// elements are BF16 values.
const TENSOR_SCALES: [f32; 3] = [0.012_3, 0.004_56, 0.078_9];
const BIAS: [f32; 4] = [0.75, -2.5, 16.0, 0.125];

/// The stored module of a path the synthetic manifest lists: the exporter's tree names the
/// block `net.layers.0` and its projections as a gated MLP's and an attention's.
fn stored(listed: &str) -> Option<String> {
    let module = match listed.strip_prefix("net.layers.0.")? {
        "mlp.gate_up.projections.gate" | "mlp.gate_up.projections.up" => FUSED,
        "mlp.down" => DOWN,
        "output" => OUTPUT,
        _ => return None,
    };
    Some(module.to_owned())
}

/// Writes the synthetic export into `directory`: the shard, and the manifest with `modules`
/// (listed path, calibrated magnitude) under one enabled stage beside a stage that is not
/// calibrated.
fn export(directory: &Path, modules: &[(&str, f64)]) -> Checkpoint {
    let mut tensors: Vec<(String, DType, Vec<usize>, Vec<u8>)> = Vec::new();

    for (seed, (module, n, k)) in [(FUSED, 8, 32), (DOWN, 4, 64), (OUTPUT, 4, 32)]
        .into_iter()
        .enumerate()
    {
        let (packed, scales) = fields(n, k, seed);

        tensors.push((
            format!("{module}.weight_packed"),
            DType::U8,
            vec![n, k / 2],
            packed,
        ));
        tensors.push((
            format!("{module}.weight_scale"),
            DType::F8E4M3,
            vec![n, k / 16],
            scales,
        ));
        // The export writes a tensor scale as a scalar: a tensor of no axes.
        tensors.push((
            format!("{module}.weight_tensor_scale"),
            DType::F32,
            Vec::new(),
            encode(DType::F32, &[TENSOR_SCALES[seed]]),
        ));
    }
    tensors.push((
        format!("{OUTPUT}.bias"),
        DType::BF16,
        vec![4],
        encode(DType::BF16, &BIAS),
    ));
    tensors.push((
        format!("{DENSE}.weight"),
        DType::BF16,
        vec![4, 32],
        encode(DType::BF16, &sequence(4 * 32, 40)),
    ));

    write_checkpoint(directory, &[&tensors]);

    let listed: serde_json::Map<String, serde_json::Value> = modules
        .iter()
        .map(|(path, amax)| {
            (
                (*path).to_owned(),
                serde_json::json!({"activation_amax": amax}),
            )
        })
        .collect();
    let manifest = serde_json::json!({
        "schema_version": 1,
        "numerical_format": {
            "activation": "a4",
            "block_scale": "fp8_e4m3",
            "block_size": 16,
            "output": "bf16",
            "tensor_scale": "fp32",
            "values": "e2m1",
            "weight": "w4",
        },
        "components": {
            "network": {"enabled": true, "modules": listed},
            "text_encoder": {"enabled": false, "modules": {}},
        },
    });
    std::fs::write(
        directory.join(ModeloptManifest::FILE),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    Checkpoint::open(directory).unwrap()
}

/// The paths the complete synthetic manifest lists, with their calibrated magnitudes.
const LISTED: [(&str, f64); 4] = [
    ("net.layers.0.mlp.gate_up.projections.gate", FUSED_AMAX),
    ("net.layers.0.mlp.gate_up.projections.up", FUSED_AMAX),
    ("net.layers.0.mlp.down", DOWN_AMAX),
    ("net.layers.0.output", OUTPUT_AMAX),
];

/// The declaration the manifest of `checkpoint` makes of the synthetic export's one stage.
fn declaration(checkpoint: &Checkpoint) -> Declaration {
    let manifest = checkpoint
        .sidecar(ModeloptManifest::FILE)
        .unwrap()
        .expect("the export holds a manifest");

    ModeloptManifest::parse(&manifest)
        .unwrap()
        .declaration(stored)
        .unwrap()
}

/// The config of a whole `[outputs, inputs]` projection.
fn whole(inputs: u32, outputs: u32, bias: bool) -> LinearConfig {
    LinearConfig {
        inputs,
        outputs,
        bias,
        parallel: Parallel::None,
    }
}

/// Each packed projection of the export computes, on the CPU device, the dense product of the
/// matrix its fields state with the activation encoded under the manifest's scale.
///
/// The expected products are computed here, in FP64, from the stored bytes and the manifest's
/// magnitudes alone. The tolerance is the formula's own:
///
/// - the module accumulates the `k` terms in FP32 and scales the sum once, which moves the
///   product by at most `(k + 2) · 2⁻²⁴` of the terms' magnitudes;
/// - the product rounds into BF16, which keeps eight significant bits: half a step is `2⁻⁸` of
///   the product, and a product the accumulation moved across a rounding boundary lands one
///   whole step, `2⁻⁷`, from the expected one;
/// - a bias joins the rounded product and the sum rounds again, one more step of the sum.
#[test]
fn a_packed_export_computes_the_product_its_fields_and_its_manifest_state() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = export(directory.path(), &LISTED);

    let store = CpuWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None);
    loader.declare(&declaration(&checkpoint)).unwrap();
    let root = loader.root();

    let rows = 3;
    for (seed, (module, n, k, amax, bias)) in [
        (FUSED, 8, 32, FUSED_AMAX, None),
        (DOWN, 4, 64, DOWN_AMAX, None),
        (OUTPUT, 4, 32, OUTPUT_AMAX, Some(&BIAS[..])),
    ]
    .into_iter()
    .enumerate()
    {
        let linear = Linear::load(&whole(k as u32, n as u32, bias.is_some()), root.pp(module))
            .unwrap_or_else(|error| panic!("{module}: {error}"));
        assert!(linear.is_packed(), "{module}");
        assert_eq!((linear.k(), linear.n()), (k as u32, n as u32), "{module}");

        // Activations up to the calibrated magnitude, as BF16 values.
        let amplitude = amax as f32;
        let x: Vec<f32> = values(&tensor(
            &sequence(rows * k, 60 + seed as u32)
                .iter()
                .map(|x| x * amplitude)
                .collect::<Vec<_>>(),
            (rows, k),
            DType::BF16,
        ));

        let (packed, scales) = fields(n, k, seed);
        let weight = decoded(&packed, &scales, TENSOR_SCALES[seed]);
        let activation_scale = (amax / (6.0 * 448.0)) as f32;
        let products = project(&quantized(&x, activation_scale), &weight, k);

        let out = linear.forward(&tensor(&x, (rows, k), DType::BF16)).unwrap();
        assert_eq!((out.dims(), out.dtype()), (&[rows, n][..], DType::BF16));

        let step = |value: f64| value.abs() / 128.0;
        for (index, (actual, (product, magnitude))) in values(&out).iter().zip(products).enumerate()
        {
            let accumulation = (k + 2) as f64 * 2f64.powi(-24) * magnitude;
            let (expected, bound) = match bias {
                None => (product, step(product) + accumulation),
                Some(bias) => {
                    let sum = product + f64::from(bias[index % n]);
                    (sum, step(product) + step(sum) + accumulation)
                }
            };

            assert!(
                (f64::from(*actual) - expected).abs() <= bound,
                "{module} element {index}: {actual} against {expected}, beyond {bound}"
            );
        }
    }

    // A weight the export leaves dense is read dense by the same loader.
    let dense = Linear::load(&whole(32, 4, false), root.pp(DENSE)).unwrap();
    assert!(!dense.is_packed());

    // The fields are filed as the vLLM spelling's are, under the export's tensor names; the
    // encoder's factor has no tensor behind it, so it is a constant of the projection, keyed
    // by the bytes it holds.
    let keys: Vec<String> = store.keys().iter().map(ToString::to_string).collect();
    for piece in [
        "|u8|nvfp4_values|blocks.0.ff.net.2.weight_packed[0..128][4, 32]",
        "|f8e4m3|nvfp4_scales_swizzled_128x4|blocks.0.ff.net.2.weight_scale__nvfp4_scales",
        "|f32|row_major|blocks.0.ff.net.2.input_scale__nvfp4_input_scale[0..1][1]|computed:nn::Linear:",
        "|f32|row_major|blocks.0.ff.net.2.alpha[0..1][1]|computed:nn::Linear:",
        "|bf16|row_major|blocks.0.attn.to_out.0.bias[0..4][4]",
    ] {
        assert!(
            keys.iter().any(|key| key.contains(piece)),
            "no key holds `{piece}`: {keys:#?}"
        );
    }
}

/// A meta load of the export chooses the same representations, and a forward states the
/// encoder and the packed product, with the bias as one more launch.
#[test]
fn a_meta_load_of_a_packed_export_states_the_encoder_and_the_product() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = export(directory.path(), &LISTED);

    let store = MetaWeights::new();

    let mut loader = WeightLoader::new(&checkpoint, &store, None);
    loader.declare(&declaration(&checkpoint)).unwrap();

    let fused = Linear::load(&whole(32, 8, false), loader.root().pp(FUSED)).unwrap();
    let (output, Recording { launches, .. }) =
        record(&[], || fused.forward(&meta((5, 32), DType::BF16)));
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

    // A bias under a builder that pins FP16, as a decoder's projections are pinned: the rows
    // are encoded from FP16 and the bias joins the product in it.
    let pinned = loader.root().to_dtype(DType::F16);
    let biased = Linear::load(&whole(32, 4, true), pinned.pp(OUTPUT)).unwrap();
    let (output, Recording { launches, .. }) =
        record(&[], || biased.forward(&meta((5, 32), DType::F16)));
    assert_eq!(output.unwrap().dtype(), DType::F16);
    assert_eq!(
        operators(&launches),
        ["nvfp4_quantize", "nvfp4_gemm", "elementwise_binary"]
    );
}

/// What the manifest lists and what the shards store must agree, in both directions.
#[test]
fn a_manifest_and_its_shards_that_disagree_are_refused_by_the_module() {
    // A packed projection the manifest does not calibrate has no scale to encode by.
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = export(directory.path(), &LISTED[..3]);
    let store = MetaWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None);
    loader.declare(&declaration(&checkpoint)).unwrap();

    let error = Linear::load(&whole(32, 4, true), loader.root().pp(OUTPUT))
        .err()
        .unwrap();
    assert!(
        matches!(&error, Error::Config { field, .. } if field == OUTPUT),
        "{error}"
    );

    // A calibration for a module the shards store dense is refused when the loader is given
    // the declaration, before any weight is read.
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = export(directory.path(), &LISTED);
    let manifest =
        ModeloptManifest::parse(&checkpoint.sidecar(ModeloptManifest::FILE).unwrap().unwrap())
            .unwrap();
    let misplaced = manifest
        .declaration(|listed| {
            stored(listed).map(|module| {
                if module == OUTPUT {
                    DENSE.to_owned()
                } else {
                    module
                }
            })
        })
        .unwrap();

    let store = MetaWeights::new();

    let mut loader = WeightLoader::new(&checkpoint, &store, None);
    let error = loader.declare(&misplaced).unwrap_err();
    assert!(
        matches!(&error, Error::Weight { tensor, .. } if tensor == "blocks.0.proj.weight_packed"),
        "{error}"
    );
}
