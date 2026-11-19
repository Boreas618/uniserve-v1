//! A block-scaled export's expert block on the CPU device: the sqrt-softplus router with its
//! selection bias and routed scaling, the MXFP4 experts' clamped SwiGLU over MXFP8 activations,
//! and the ungated shared expert beside them, each against the arithmetic the reference states.
//!
//! The export is written here as DeepSeek's `fp8` producer writes one, at small widths the
//! products admit (hidden 128, expert width 128): routed experts of packed E2M1 codes with one
//! UE8M0 scale per 32 elements of each row, a shared expert of E4M3 codes under 32 × 32 block
//! scales, a BF16 router and its FP32 selection bias; over an expert group, the launches a rank
//! states between the exchange's dispatch and combine. Nothing here touches a GPU.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use std::path::Path;

use nn::collective::Split;
use nn::functional::{self, Activation, Rounding, Routing, Scoring, WeightOrder};
use nn::modules::{
    ExpertLayout, GateUp, MlpConfig, MoE, MoEConfig, SharedExpert, SharedExpertConfig,
};
use nn::quantization::Declaration;
use nn::{Checkpoint, CpuWeights, DType, ExpertExchange, MetaWeights, WeightLoader};
use nn_testing::{
    Recording, encode, meta, operators, record, sequence, tensor, values, write_checkpoint,
};

const HIDDEN: usize = 128;
const INTERMEDIATE: usize = 128;
const EXPERTS: usize = 4;
const TOP: usize = 2;
const TOKENS: usize = 3;

/// DeepSeek's routed scaling factor and SwiGLU clamp.
const ROUTED_SCALE: f32 = 1.5;
const LIMIT: f32 = 10.0;

/// The magnitudes an E2M1 code states; a code's fourth bit is its sign.
const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// The value of a UE8M0 scale byte.
fn scale(byte: u8) -> f64 {
    2f64.powi(i32::from(byte) - 127)
}

/// The value of an E4M3 byte: a sign, four exponent bits biased by seven and three mantissa bits.
fn e4m3_value(byte: u8) -> f64 {
    let (exponent, mantissa) = (i32::from((byte >> 3) & 0x0f), f64::from(byte & 0x07));
    let magnitude = if exponent == 0 {
        mantissa / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + mantissa / 8.0) * 2f64.powi(exponent - 7)
    };
    if byte & 0x80 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// The nearest E4M3 value to `x`, ties to the even code, within ±448.
fn e4m3(x: f64) -> f64 {
    let grid: Vec<f64> = (0..=0x7e).map(e4m3_value).collect();
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

/// `x` encoded into MXFP8 and decoded: per 32 elements, the least power of two at or above
/// `max(amax, 1e-4) / 448` scales E4M3 codes of the quotients (the covering rule, an activation's).
fn quantized(x: &[f64]) -> Vec<f64> {
    x.chunks(32)
        .flat_map(|block| {
            let amax = block.iter().fold(1e-4_f64, |amax, x| amax.max(x.abs()));
            let unit = 2f64.powi((amax / 448.0).log2().ceil() as i32);
            block.iter().map(move |x| e4m3(x / unit) * unit)
        })
        .collect()
}

/// `x` encoded into MXFP8 by the OCP rule and decoded: per 32 elements, `2^(⌊log₂ amax⌋ − 8)`
/// scales E4M3 codes of the quotients, which saturate at ±448 (the gated rows' encoding).
fn ocp_quantized(x: &[f64]) -> Vec<f64> {
    x.chunks(32)
        .flat_map(|block| {
            // A block of zeros encodes as zeros under any scale.
            let amax = block
                .iter()
                .fold(f64::MIN_POSITIVE, |amax, x| amax.max(x.abs()));
            let unit = 2f64.powi(amax.log2().floor() as i32 - 8);
            block.iter().map(move |x| e4m3(x / unit) * unit)
        })
        .collect()
}

/// An MXFP4 matrix of `rows` by `columns`: its packed codes and its per-row scales, different
/// for each `seed`, and the values they state.
fn mxfp4(rows: usize, columns: usize, seed: usize) -> (Vec<u8>, Vec<u8>, Vec<f64>) {
    let packed: Vec<u8> = (0..rows * columns / 2)
        .map(|index| ((index * 97 + seed * 31) % 256) as u8)
        .collect();
    let scales: Vec<u8> = (0..rows * columns / 32)
        .map(|index| 122 + ((index + seed) % 4) as u8)
        .collect();
    let values = (0..rows * columns)
        .map(|element| {
            let byte = packed[element / 2];
            let code = if element % 2 == 0 {
                byte & 0x0f
            } else {
                byte >> 4
            };
            let sign = if code & 0x08 == 0 { 1.0 } else { -1.0 };
            sign * E2M1[usize::from(code & 0x07)] * scale(scales[element / 32])
        })
        .collect();
    (packed, scales, values)
}

/// An FP8 matrix of `rows` by `columns` under 32 × 32 block scales, and the values it states.
fn fp8(rows: usize, columns: usize, seed: usize) -> (Vec<u8>, Vec<u8>, Vec<f64>) {
    let codes: Vec<u8> = (0..rows * columns)
        .map(|index| ((index * 53 + seed * 7) % 251) as u8 & 0xf7)
        .collect();
    let blocks: Vec<u8> = (0..(rows / 32) * (columns / 32))
        .map(|index| 116 + ((index + seed) % 3) as u8)
        .collect();
    let values = (0..rows * columns)
        .map(|index| {
            let (row, column) = (index / columns, index % columns);
            e4m3_value(codes[index]) * scale(blocks[(row / 32) * (columns / 32) + column / 32])
        })
        .collect();
    (codes, blocks, values)
}

/// The block's dense reference parts: each expert's `(gate, up, down)` and the shared expert's.
struct Parts {
    experts: Vec<[Vec<f64>; 3]>,
    shared: [Vec<f64>; 3],
}

/// Writes the block's export under `ffn` and answers the values its weights state.
fn export(directory: &Path) -> Parts {
    let mut tensors = Vec::new();
    let mut experts = Vec::new();
    for expert in 0..EXPERTS {
        let mut parts = Vec::new();
        for (index, (name, rows, columns)) in [
            ("w1", INTERMEDIATE, HIDDEN),
            ("w3", INTERMEDIATE, HIDDEN),
            ("w2", HIDDEN, INTERMEDIATE),
        ]
        .into_iter()
        .enumerate()
        {
            let (packed, scales, values) = mxfp4(rows, columns, 3 * expert + index);
            let stem = format!("ffn.experts.{expert}.{name}");
            tensors.push((
                format!("{stem}.weight"),
                DType::U8,
                vec![rows, columns / 2],
                packed,
            ));
            tensors.push((
                format!("{stem}.scale"),
                DType::U8,
                vec![rows, columns / 32],
                scales,
            ));
            parts.push(values);
        }
        experts.push([parts[0].clone(), parts[1].clone(), parts[2].clone()]);
    }
    let mut shared = Vec::new();
    for (index, (name, rows, columns)) in [
        ("w1", INTERMEDIATE, HIDDEN),
        ("w3", INTERMEDIATE, HIDDEN),
        ("w2", HIDDEN, INTERMEDIATE),
    ]
    .into_iter()
    .enumerate()
    {
        let (codes, blocks, values) = fp8(rows, columns, 20 + index);
        let stem = format!("ffn.shared_experts.{name}");
        tensors.push((
            format!("{stem}.weight"),
            DType::F8E4M3,
            vec![rows, columns],
            codes,
        ));
        tensors.push((
            format!("{stem}.scale"),
            DType::U8,
            vec![rows / 32, columns / 32],
            blocks,
        ));
        shared.push(values);
    }
    let router = sequence(EXPERTS * HIDDEN, 41)
        .iter()
        .map(|value| value * 0.25)
        .collect::<Vec<_>>();
    tensors.push((
        "ffn.gate.weight".to_owned(),
        DType::BF16,
        vec![EXPERTS, HIDDEN],
        encode(DType::BF16, &router),
    ));
    tensors.push((
        "ffn.gate.bias".to_owned(),
        DType::F32,
        vec![EXPERTS],
        encode(DType::F32, &[0.0, 0.75, -0.5, 0.25]),
    ));

    write_checkpoint(directory, &[&tensors]);
    std::fs::write(
        directory.join("config.json"),
        serde_json::json!({ "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        }})
        .to_string(),
    )
    .unwrap();
    Parts {
        experts,
        shared: [shared[0].clone(), shared[1].clone(), shared[2].clone()],
    }
}

/// DeepSeek's routed block at the test's widths.
fn config() -> MoEConfig {
    MoEConfig {
        hidden: HIDDEN as u32,
        intermediate: INTERMEDIATE as u32,
        experts: EXPERTS as u32,
        top: TOP as u32,
        activation: Activation::Silu,
        renormalize: true,
        scoring: Scoring::SqrtSoftplus,
        selection_bias: Some("bias"),
        routed_scale: Some(ROUTED_SCALE),
        limit: Some(LIMIT),
        router: "gate",
        router_dtype: None,
        list: "experts",
        up: "w3",
        gate: "w1",
        down: "w2",
        layout: ExpertLayout::Modules,
        order: (0..EXPERTS as u32).collect(),
        split: Split::WHOLE,
        tensor_experts: false,
        exchange: Split::WHOLE,
        name: None,
    }
}

/// `matrix [rows, columns] · x`, accumulated in FP64.
fn product(matrix: &[f64], columns: usize, x: &[f64]) -> Vec<f64> {
    matrix
        .chunks(columns)
        .map(|row| row.iter().zip(x).map(|(w, x)| w * x).sum())
        .collect()
}

/// The reference's routed scores of one token's logits: the top `TOP` of `sqrt(softplus(s)) +
/// bias`, weighted by the unbiased scores renormalized and scaled.
fn route(logits: &[f32], bias: &[f32]) -> (Vec<usize>, Vec<f64>) {
    let scores: Vec<f64> = logits
        .iter()
        .map(|logit| f64::from(*logit).exp().ln_1p().sqrt())
        .collect();
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|left, right| {
        let key = |expert: &usize| scores[*expert] + f64::from(bias[*expert]);
        key(right)
            .partial_cmp(&key(left))
            .unwrap()
            .then(left.cmp(right))
    });
    order.truncate(TOP);
    let total: f64 = order.iter().map(|expert| scores[*expert]).sum();
    let weights = order
        .iter()
        .map(|expert| scores[*expert] / (total + 1e-20) * f64::from(ROUTED_SCALE))
        .collect();
    (order, weights)
}

/// The sqrt-softplus router selects by the biased scores and weighs by the unbiased ones,
/// renormalized and scaled; the bias changes which experts a token takes.
#[test]
fn a_sqrt_softplus_router_selects_by_the_bias_and_weighs_without_it() {
    let logits = [0.1_f32, 0.3, 0.2, -0.4, 1.5, 1.4, -2.0, 0.0];
    let bias = [0.0_f32, 0.75, -0.5, 0.25];
    let scale = tensor(&[ROUTED_SCALE; 4], 4, DType::F32);
    let routing = Routing {
        top: TOP as u32,
        scoring: Scoring::SqrtSoftplus,
        renormalize: true,
        bias: Some(&tensor(&bias, 4, DType::F32)),
        scale: Some(&scale),
    };

    let (ids, weights) = functional::topk_softmax(
        &tensor(&logits, (2, 4), DType::F32),
        &routing,
        None,
        None,
        None,
    )
    .unwrap();
    let ids: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
    let weights = values(&weights);
    for (token, row) in logits.chunks(4).enumerate() {
        let (want_ids, want_weights) = route(row, &bias);
        let got: Vec<usize> = ids[token * TOP..][..TOP]
            .iter()
            .map(|id| *id as usize)
            .collect();
        assert_eq!(got, want_ids, "token {token}");
        for (got, want) in weights[token * TOP..][..TOP].iter().zip(&want_weights) {
            assert!(
                (f64::from(*got) - want).abs() <= 1e-6 * want.abs(),
                "{got} vs {want}"
            );
        }
    }
    // Without the bias the first token's second pick is expert 2 (scores 0.92, 0.89, 0.86,
    // 0.72); with it expert 3's key passes both others, and its weight stays its own score's.
    assert_eq!(&ids[..TOP], &[1, 3]);
}

/// An MXFP4 block is the reference's experts: each route's expert over the encoded activation,
/// its SwiGLU clamped, its gated row encoded again, and the routes summed at their weights; the
/// ungated shared expert's MXFP8 projections add their own output.
#[test]
fn an_mxfp4_block_and_its_shared_expert_compute_the_references_experts() {
    let directory = tempfile::tempdir().unwrap();
    let parts = export(directory.path());
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let store = CpuWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None);
    loader
        .declare(&Declaration::read(&checkpoint).unwrap())
        .unwrap();
    let block = MoE::load(&config(), loader.root().pp("ffn")).unwrap();
    let shared = SharedExpert::load(
        &SharedExpertConfig {
            mlp: MlpConfig {
                hidden: HIDDEN as u32,
                intermediate: INTERMEDIATE as u32,
                activation: Activation::Silu,
                rounding: Rounding::Fused,
                limit: Some(LIMIT),
                gate_up: GateUp::Split {
                    gate: "w1",
                    up: "w3",
                },
                down: "w2",
                bias: false,
                split: Split::WHOLE,
            },
            module: "shared_experts",
            gate: None,
        },
        loader.root().pp("ffn"),
    )
    .unwrap();

    let x: Vec<f32> = sequence(TOKENS * HIDDEN, 43)
        .iter()
        .map(|value| value * 4.0)
        .collect();
    let x = values(&tensor(&x, (TOKENS, HIDDEN), DType::BF16));
    let input = tensor(&x, (TOKENS, HIDDEN), DType::BF16);
    let routed = block.forward(&input, None, None).unwrap();
    let out = values(&shared.forward(&input, routed.clone()).unwrap());
    let routed = values(&routed);

    // The router's logits as the block computes them: a BF16 product of the BF16 rows.
    let router = loader
        .root()
        .pp("ffn.gate")
        .get((EXPERTS, HIDDEN), "weight")
        .unwrap();
    let logits = values(&functional::linear(&input, &router, None, WeightOrder::NxK).unwrap());
    let bias = [0.0_f32, 0.75, -0.5, 0.25];

    let mut want_routed = vec![0.0_f64; TOKENS * HIDDEN];
    let mut want_shared = vec![0.0_f64; TOKENS * HIDDEN];
    let bf16 = |values: &[f64]| -> Vec<f64> {
        let rounded: Vec<f32> = values.iter().map(|value| *value as f32).collect();
        nn_testing::values(&tensor(&rounded, rounded.len(), DType::BF16))
            .into_iter()
            .map(f64::from)
            .collect()
    };
    for token in 0..TOKENS {
        let row: Vec<f64> = x[token * HIDDEN..][..HIDDEN]
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        let encoded = quantized(&row);
        let (experts, weights) = route(&logits[token * EXPERTS..][..EXPERTS], &bias);
        for (expert, weight) in experts.into_iter().zip(weights) {
            let [gate, up, down] = &parts.experts[expert];
            let gate = product(gate, HIDDEN, &encoded);
            let up = product(up, HIDDEN, &encoded);
            let gated: Vec<f64> = gate
                .iter()
                .zip(&up)
                .map(|(gate, up)| {
                    let gate = gate.min(f64::from(LIMIT));
                    gate / (1.0 + (-gate).exp()) * up.clamp(-f64::from(LIMIT), f64::from(LIMIT))
                })
                .collect();
            // The first product writes its gated rows as MXFP8 by the OCP rule; the second reads
            // them so.
            let y = bf16(&product(down, INTERMEDIATE, &ocp_quantized(&gated)));
            for (sum, y) in want_routed[token * HIDDEN..][..HIDDEN].iter_mut().zip(y) {
                *sum += weight * y;
            }
        }

        // The shared expert's projections each write BF16, its gated product too.
        let [gate, up, down] = &parts.shared;
        let gate = bf16(&product(gate, HIDDEN, &encoded));
        let up = bf16(&product(up, HIDDEN, &encoded));
        let gated = bf16(
            &gate
                .iter()
                .zip(&up)
                .map(|(gate, up)| {
                    let gate = gate.min(f64::from(LIMIT));
                    gate / (1.0 + (-gate).exp()) * up.clamp(-f64::from(LIMIT), f64::from(LIMIT))
                })
                .collect::<Vec<_>>(),
        );
        let y = bf16(&product(down, INTERMEDIATE, &quantized(&gated)));
        want_shared[token * HIDDEN..][..HIDDEN].copy_from_slice(&y);
    }

    // A rounding boundary of an encoding may fall between the two accumulations, moving a code
    // one step (an eighth of its value): the bound is a few such steps of the output's largest
    // magnitude, which an arrangement or a scale read at the wrong place exceeds many times over.
    let peak = want_routed
        .iter()
        .fold(0.0_f64, |peak, value| peak.max(value.abs()));
    for (index, (got, want)) in routed.iter().zip(&want_routed).enumerate() {
        assert!(
            (f64::from(*got) - want).abs() <= 2f64.powi(-5) * peak,
            "routed [{}, {}]: {got} vs {want} (peak {peak})",
            index / HIDDEN,
            index % HIDDEN
        );
    }
    let peak = want_shared
        .iter()
        .fold(peak, |peak, value| peak.max(value.abs()));
    for (index, got) in out.iter().enumerate() {
        let want = f64::from(routed[index]) + want_shared[index];
        assert!(
            (f64::from(*got) - want).abs() <= 2f64.powi(-5) * peak,
            "shared [{}, {}]: {got} vs {want} (peak {peak})",
            index / HIDDEN,
            index % HIDDEN
        );
    }
}

/// The rows of an exchange over `ranks` ranks of `rank_rows` rows each, as meta tensors.
fn exchange(ranks: u32, rank_rows: u32) -> ExpertExchange {
    let rows = (ranks * rank_rows) as usize;

    ExpertExchange {
        ranks,
        attention: 0,
        rank_rows,
        step_rows: rank_rows,
        multiprocessors: 148,
        states: meta((rows, HIDDEN), DType::BF16),
        ids: meta((rows, TOP), DType::U32),
        weights: meta((rows, TOP), DType::F32),
        combined: meta((rows, HIDDEN), DType::BF16),
        live: meta(1, DType::U32),
        packed: meta((rank_rows as usize, HIDDEN / 2), DType::U8),
        packed_scales: meta((rank_rows as usize, HIDDEN / 16), DType::F8E4M3),
        fused: false,
        source_only: false,
    }
}

/// Over an expert group a rank holds its window of whole MXFP4 experts and runs the MXFP4
/// products over the BF16 rows the exchange delivered: the dispatch, the rows' MXFP8 encoding, its
/// experts' two products and the weighted sums left for the combine, which returns its tokens' sums. A rank with no
/// tokens of its own joins the step and serves its peers' routes the same way.
#[test]
fn an_mxfp4_block_over_an_expert_group_runs_its_window_between_dispatch_and_combine() {
    let directory = tempfile::tempdir().unwrap();
    export(directory.path());
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let store = MetaWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None);
    loader
        .declare(&Declaration::read(&checkpoint).unwrap())
        .unwrap();
    let group = Split::new(2, 1).unwrap();
    let block = MoE::load(
        &MoEConfig {
            exchange: group,
            ..config()
        },
        loader.root().pp("ffn"),
    )
    .unwrap();
    assert_eq!((block.experts(), block.local_experts()), (4, 2));

    // The products of the rank's two experts over the rows every rank sent, eight of them.
    let lent = exchange(2, 4);
    let (output, Recording { launches, .. }) = record(&[group], || {
        block.forward(&meta((3, HIDDEN), DType::BF16), None, Some(&lent))
    });
    assert_eq!(output.unwrap().dims(), [3, HIDDEN]);
    let mxfp4_products = [
        "expert_exchange",
        "moe_sort",
        "mxfp8_quantize",
        "gather_gemm_mxfp4",
        "route_gemm_mxfp4",
        "moe_unpermute",
        "expert_exchange",
    ];
    let stated = operators(&launches);
    assert_eq!(
        stated[stated.len() - mxfp4_products.len()..],
        mxfp4_products
    );
    let (_, sort) = launches
        .iter()
        .find(|(operator, _)| *operator == "moe_sort")
        .unwrap();
    assert!(sort.contains("experts_4_top_2_local_2"), "{sort}");

    let (joined, Recording { launches, .. }) = record(&[group], || block.join_expert_step(&lent));
    joined.unwrap();
    assert_eq!(operators(&launches), mxfp4_products);
}
