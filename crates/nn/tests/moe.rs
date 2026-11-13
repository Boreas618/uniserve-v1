//! The mixture-of-experts operators and modules: their arithmetic on CPU tensors against each
//! formula written out in loops, and the kernels they state under a meta context.
//!
//! Nothing here touches a GPU: a CPU tensor is computed by the function's formula, and a meta
//! context launches nothing.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

mod support;

use std::ops::Range;
use std::path::Path;

use nn::collective::Split;
use nn::functional::{self, Activation, Rounding};
use nn::modules::{GateUp, MlpConfig, MoE, MoEConfig, SharedExpert, SharedExpertConfig};
use nn::quantization::{Declaration, Nvfp4Names, Nvfp4Stack, ValueLayout};
use nn::{Checkpoint, CpuWeights, DType, Device, ExpertExchange, Tensor, WeightLoader};
use ops::fp4::ScaleLayout;
use support::{
    assert_close, encode, f32_bytes, meta, operators, recorded, recorded_split, sequence, shard,
    tensor, values,
};

/// The tolerance of an FP32 result against a reference accumulated in another order.
const F32_TOLERANCE: f32 = 1e-4;

/// The tolerance of a BF16 result against an FP32 reference: half a step of eight mantissa bits,
/// with room for a sum of them.
const BF16_TOLERANCE: f32 = 1e-2;

/// The rows of a tile.
const TILE: usize = functional::TILE_ROWS as usize;

/// A CPU tensor of U32 `entries` in `shape`.
fn indices(entries: &[u32], shape: impl Into<nn::Shape>) -> Tensor {
    Tensor::from_slice(entries, shape.into(), &Device::Cpu).unwrap()
}

/// The entries of a U32 tensor, in row-major order.
fn entries(tensor: &Tensor) -> Vec<u32> {
    tensor.flatten_all().unwrap().to_vec1().unwrap()
}

/// `x / (1 + exp(−x))`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `matrix [rows, columns] · x`.
fn product(matrix: &[f32], x: &[f32]) -> Vec<f32> {
    matrix
        .chunks(x.len())
        .map(|row| row.iter().zip(x).map(|(weight, x)| weight * x).sum())
        .collect()
}

// ---- Routing ----------------------------------------------------------------------------------

/// The top `top` experts of each row of `scores` and their softmax probabilities, renormalized
/// over the kept ones: one arg-max per pick, the lowest expert among equals.
fn route_by_hand(scores: &[f32], experts: usize, top: usize) -> (Vec<u32>, Vec<f32>) {
    let (mut ids, mut weights) = (Vec::new(), Vec::new());

    for row in scores.chunks(experts) {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let shifted: Vec<f32> = row.iter().map(|score| (score - maximum).exp()).collect();
        let sum: f32 = shifted.iter().sum();
        let probabilities: Vec<f32> = shifted.iter().map(|value| value / sum).collect();

        let mut picked: Vec<usize> = Vec::new();
        for _ in 0..top {
            let mut best: Option<usize> = None;
            for expert in (0..experts).filter(|expert| !picked.contains(expert)) {
                if best.is_none_or(|best| probabilities[expert] > probabilities[best]) {
                    best = Some(expert);
                }
            }
            picked.push(best.unwrap());
        }

        let total: f32 = picked.iter().map(|expert| probabilities[*expert]).sum();
        for expert in picked {
            ids.push(expert as u32);
            weights.push(probabilities[expert] / total.max(f32::EPSILON));
        }
    }
    (ids, weights)
}

#[test]
fn topk_softmax_keeps_the_most_probable_experts_and_renormalizes_them() {
    let (tokens, experts, top) = (4, 6, 3);
    let mut scores = sequence(tokens * experts, 21);

    // Three experts of the second token tie for the most probable: the lowest ids win, in order.
    scores[experts..2 * experts].copy_from_slice(&[0.5, 1.0, 1.0, -1.0, 1.0, 0.2]);
    let (expected_ids, expected_weights) = route_by_hand(&scores, experts, top);
    assert_eq!(expected_ids[top..2 * top], [1, 2, 4]);

    let (ids, weights) = functional::topk_softmax(
        &tensor(&scores, (tokens, experts), DType::F32),
        top as u32,
        true,
        None,
        None,
    )
    .unwrap();
    assert_eq!((ids.dims(), ids.dtype()), (&[tokens, top][..], DType::U32));
    assert_eq!(entries(&ids), expected_ids);
    assert_close(&values(&weights), &expected_weights, F32_TOLERANCE);

    // Without the renormalization the weights are the softmax's own probabilities.
    let (_, raw) = functional::topk_softmax(
        &tensor(&scores, (tokens, experts), DType::F32),
        top as u32,
        false,
        None,
        None,
    )
    .unwrap();
    for (row, kept) in values(&raw).chunks(top).zip(expected_weights.chunks(top)) {
        let total: f32 = row.iter().sum();
        assert!(total < 1.0, "three of six probabilities sum to {total}");
        assert_close(
            &row.iter().map(|weight| weight / total).collect::<Vec<_>>(),
            kept,
            F32_TOLERANCE,
        );
    }
}

/// The widest served block routes each token to ten of 512 experts (Qwen3.8 Flash Next):
/// the formula keeps the ten most probable, and the launch states their counts in its key.
#[test]
fn topk_softmax_routes_ten_of_five_hundred_and_twelve_experts() {
    let (tokens, experts, top) = (3, 512, 10);
    assert!(ops::moe::routing::MAX_EXPERTS >= experts as u32);
    assert!(ops::moe::routing::MAX_TOP >= top as u32);

    let scores = sequence(tokens * experts, 77);
    let (expected_ids, expected_weights) = route_by_hand(&scores, experts, top);
    let (ids, weights) = functional::topk_softmax(
        &tensor(&scores, (tokens, experts), DType::F32),
        top as u32,
        true,
        None,
        None,
    )
    .unwrap();
    assert_eq!(entries(&ids), expected_ids);
    assert_close(&values(&weights), &expected_weights, F32_TOLERANCE);

    let (routed, launches) = recorded(|| {
        functional::topk_softmax(
            &meta((tokens, experts), DType::BF16),
            top as u32,
            true,
            None,
            None,
        )
    });
    routed.unwrap();
    assert_eq!(
        launches,
        [(
            "topk_softmax",
            "scores_bf16_experts_512_top_10_renormalize_true_scale_none".to_owned()
        )]
    );
}

#[test]
fn topk_softmax_scales_by_the_expert_and_writes_each_route_s_slot() {
    let (tokens, experts, top) = (3, 5, 2);
    let scores = sequence(tokens * experts, 22);
    let scale = [0.5_f32, 2.0, 1.5, 0.25, 3.0];
    let map = [4_u32, 2, 0, 1, 3];
    let (expected_ids, expected_weights) = route_by_hand(&scores, experts, top);

    let (ids, weights) = functional::topk_softmax(
        &tensor(&scores, (tokens, experts), DType::F32),
        top as u32,
        true,
        Some(&tensor(&scale, experts, DType::F32)),
        Some(&indices(&map, experts)),
    )
    .unwrap();

    // The selection and the scale are by the expert's own id; only the id written is the slot.
    let slots: Vec<u32> = expected_ids.iter().map(|id| map[*id as usize]).collect();
    let scaled: Vec<f32> = expected_ids
        .iter()
        .zip(&expected_weights)
        .map(|(id, weight)| weight * scale[*id as usize])
        .collect();
    assert_eq!(entries(&ids), slots);
    assert_close(&values(&weights), &scaled, F32_TOLERANCE);

    // A top beyond the experts, and a map of another length, are refused.
    let scores = tensor(&scores, (tokens, experts), DType::F32);
    assert!(functional::topk_softmax(&scores, 6, true, None, None).is_err());
    assert!(functional::topk_softmax(&scores, 2, true, None, Some(&indices(&[0, 1], 2))).is_err());
}

// ---- The sort ---------------------------------------------------------------------------------

/// The route maps of `ids`, `top` routes a token, for a rank that holds the experts `local`,
/// of a batch that fills `live_tokens` rows: each map as far as the sort writes it, in the
/// order of the maps' rows.
fn sort_by_hand(ids: &[u32], top: usize, local: Range<u32>, live_tokens: usize) -> Vec<Vec<u32>> {
    let experts = (local.end - local.start) as usize;
    let mut groups: Vec<Vec<u32>> = vec![Vec::new(); experts];
    let mut expanded_to_permuted = vec![0_u32; ids.len()];

    for (route, &id) in ids.iter().enumerate() {
        if route / top >= live_tokens {
            expanded_to_permuted[route] = u32::MAX - 1;
        } else if !local.contains(&id) {
            expanded_to_permuted[route] = u32::MAX;
        } else {
            groups[(id - local.start) as usize].push(route as u32);
        }
    }

    let (mut tile_to_expert, mut tile_to_limit) = (Vec::new(), Vec::new());
    let mut permuted_to_expanded = Vec::new();
    for (expert, group) in groups.iter().enumerate() {
        let first = permuted_to_expanded.len();
        let tiles = group.len().div_ceil(TILE);

        for (place, &route) in group.iter().enumerate() {
            expanded_to_permuted[route as usize] = (first + place) as u32;
        }
        for tile in 0..tiles {
            tile_to_expert.push(expert as u32);
            tile_to_limit.push((first + group.len()).min(first + (tile + 1) * TILE) as u32);
        }
        permuted_to_expanded.extend(group);
        permuted_to_expanded.resize(first + tiles * TILE, u32::MAX);
    }

    let permuted_to_token = permuted_to_expanded
        .iter()
        .map(|route| match *route {
            u32::MAX => u32::MAX,
            route => route / top as u32,
        })
        .collect();
    let (padded, tiles) = (permuted_to_expanded.len(), tile_to_expert.len());

    vec![
        tile_to_expert,
        tile_to_limit,
        expanded_to_permuted,
        permuted_to_expanded,
        permuted_to_token,
        vec![padded as u32],
        vec![tiles as u32],
    ]
}

/// Asserts that `maps` holds `expected`, the maps a sort writes, in the leading entries of its
/// rows.
fn assert_maps(maps: &Tensor, expected: &[Vec<u32>]) {
    for (row, (name, expected)) in [
        "tile_to_expert",
        "tile_to_limit",
        "expanded_to_permuted",
        "permuted_to_expanded",
        "permuted_to_token",
        "total_padded_tokens",
        "num_non_exiting_tiles",
    ]
    .iter()
    .zip(expected)
    .enumerate()
    {
        let written = maps.get(row).unwrap().narrow(0, 0, expected.len()).unwrap();
        assert_eq!(&entries(&written), expected, "{name}");
    }
}

#[test]
fn moe_sort_groups_the_routes_a_rank_holds_into_its_experts_tiles() {
    // 150 tokens of two routes over four experts, of which this rank holds the first three and
    // the batch fills 148 rows: expert 0 takes 130 routes (two tiles, the second of two rows),
    // expert 1 the next 18 live ones, expert 2 every odd token's second route.
    let (tokens, top, experts) = (150_usize, 2_usize, 4_u32);
    let ids: Vec<u32> = (0..tokens as u32)
        .flat_map(|token| [u32::from(token >= 130), 3 - token % 2])
        .collect();
    let live = 148;
    let expected = sort_by_hand(&ids, top, 0..3, live);
    assert_eq!(expected[functional::TILE_TO_EXPERT], [0, 0, 1, 2]);
    assert_eq!(expected[functional::TILE_TO_LIMIT], [128, 130, 274, 458]);

    let maps = functional::moe_sort(
        &indices(&ids, (tokens, top)),
        &tensor(&sequence(tokens * top, 23), (tokens, top), DType::F32),
        experts,
        0..3,
        Some(&indices(&[live as u32], 1)),
    )
    .unwrap();

    // Five tiles hold the worst routing of 300 routes over three experts.
    assert_eq!(
        (maps.dims(), maps.dtype()),
        (&[functional::SORT_MAPS, 5 * TILE][..], DType::U32)
    );
    assert_maps(&maps, &expected);

    // A route of another rank's expert has no row, and a route of a row the batch does not fill
    // says so apart.
    let rows = entries(&maps.get(functional::EXPANDED_TO_PERMUTED).unwrap());
    assert_eq!(rows[1], u32::MAX, "token 0's second route is expert 3's");
    assert_eq!(rows[2 * 148], u32::MAX - 1, "token 148 is not live");
}

#[test]
fn moe_sort_rebases_a_window_and_counts_every_row_live_without_a_count() {
    let (tokens, top) = (5, 2);
    let ids = [3_u32, 1, 2, 0, 2, 3, 1, 1, 0, 2];
    let expected = sort_by_hand(&ids, top, 2..4, tokens);

    // Tiles name the experts by their place in the window: expert 2 is 0 and expert 3 is 1.
    assert_eq!(expected[functional::TILE_TO_EXPERT], [0, 1]);

    let weights = tensor(&sequence(tokens * top, 24), (tokens, top), DType::F32);
    let maps =
        functional::moe_sort(&indices(&ids, (tokens, top)), &weights, 4, 2..4, None).unwrap();
    assert_maps(&maps, &expected);

    // A window that is not the experts', and routes that are not a matrix, are refused.
    let ids = indices(&ids, (tokens, top));
    assert!(functional::moe_sort(&ids, &weights, 4, 2..5, None).is_err());
    assert!(functional::moe_sort(&ids.flatten_all().unwrap(), &weights, 4, 2..4, None).is_err());
}

// ---- The dense products and the combine -------------------------------------------------------

/// One dense expert: its up, gate and down projections, `[intermediate, hidden]` twice and
/// `[hidden, intermediate]`, as rows of values.
fn dense_expert(hidden: usize, intermediate: usize, seed: u32) -> [Vec<f32>; 3] {
    [
        sequence(intermediate * hidden, seed),
        sequence(intermediate * hidden, seed + 1),
        sequence(hidden * intermediate, seed + 2),
    ]
}

/// `down · (silu(gate · x) · (up · x))` for one row.
fn expert_by_hand(x: &[f32], [up, gate, down]: &[Vec<f32>; 3]) -> Vec<f32> {
    let gated: Vec<f32> = product(gate, x)
        .iter()
        .zip(product(up, x))
        .map(|(gate, up)| silu(*gate) * up)
        .collect();

    product(down, &gated)
}

/// The `[experts, 2 · intermediate, hidden]` stack of `experts`, up rows before gate rows, and
/// their `[experts, hidden, intermediate]` stack.
fn stacks(experts: &[[Vec<f32>; 3]], hidden: usize, intermediate: usize) -> (Tensor, Tensor) {
    let up_gate: Vec<f32> = experts
        .iter()
        .flat_map(|[up, gate, _]| up.iter().chain(gate).copied())
        .collect();
    let down: Vec<f32> = experts
        .iter()
        .flat_map(|[_, _, down]| down.iter().copied())
        .collect();

    (
        tensor(
            &up_gate,
            (experts.len(), 2 * intermediate, hidden),
            DType::F32,
        ),
        tensor(&down, (experts.len(), hidden, intermediate), DType::F32),
    )
}

#[test]
fn the_grouped_products_and_the_combine_sum_each_token_s_experts() {
    let (tokens, top, hidden, intermediate) = (5, 2, 8, 6);
    let experts: Vec<[Vec<f32>; 3]> = (0..4)
        .map(|expert| dense_expert(hidden, intermediate, 30 + 3 * expert))
        .collect();
    let (up_gate, down) = stacks(&experts, hidden, intermediate);

    let x = sequence(tokens * hidden, 25);
    let ids = [3_u32, 1, 2, 0, 2, 3, 1, 0, 0, 2];
    let weights = sequence(tokens * top, 26);
    let (ids_tensor, weights_tensor) = (
        indices(&ids, (tokens, top)),
        tensor(&weights, (tokens, top), DType::F32),
    );
    let x_tensor = tensor(&x, (tokens, hidden), DType::F32);

    let maps = functional::moe_sort(&ids_tensor, &weights_tensor, 4, 0..4, None).unwrap();
    let gated =
        functional::gather_gemm(&x_tensor, &up_gate, &ids_tensor, &maps, Activation::Silu).unwrap();
    let products = functional::route_gemm(&gated, &down, &ids_tensor, &maps).unwrap();
    assert_eq!(gated.dims(), [4 * TILE, intermediate]);
    assert_eq!(products.dims(), [4 * TILE, hidden]);

    // Each route's product stands at its permuted row, and the rows no route takes are zero.
    let rows = sort_by_hand(&ids, top, 0..4, tokens)[functional::EXPANDED_TO_PERMUTED].clone();
    let product_rows = values(&products);
    let mut live = vec![false; 4 * TILE];
    for (route, row) in rows.iter().enumerate() {
        let expert = expert_by_hand(
            &x[route / top * hidden..][..hidden],
            &experts[ids[route] as usize],
        );

        live[*row as usize] = true;
        assert_close(
            &product_rows[*row as usize * hidden..][..hidden],
            &expert,
            F32_TOLERANCE,
        );
    }
    for (row, live) in live.iter().enumerate() {
        if !live {
            assert!(
                product_rows[row * hidden..][..hidden]
                    .iter()
                    .all(|value| *value == 0.0),
                "padding row {row} holds a value"
            );
        }
    }

    // The combine is the weighted sum of a token's routes.
    let combined = functional::moe_unpermute(&products, &maps, &weights_tensor, 4, None).unwrap();
    let expected: Vec<f32> = (0..tokens)
        .flat_map(|token| {
            let mut sum = vec![0.0_f32; hidden];
            for route in token * top..(token + 1) * top {
                let expert = expert_by_hand(
                    &x[token * hidden..][..hidden],
                    &experts[ids[route] as usize],
                );
                for (sum, value) in sum.iter_mut().zip(expert) {
                    *sum += weights[route] * value;
                }
            }
            sum
        })
        .collect();
    assert_eq!(combined.dims(), [tokens, hidden]);
    assert_close(&values(&combined), &expected, F32_TOLERANCE);

    // Lent rows receive the same sums in place: the tensor returned is the one lent.
    let lent = Tensor::zeros((tokens, hidden), DType::F32, &Device::Cpu).unwrap();
    let returned =
        functional::moe_unpermute(&products, &maps, &weights_tensor, 4, Some(&lent)).unwrap();
    assert_eq!(values(&lent), values(&combined));
    assert_eq!(values(&returned), values(&combined));
}

#[test]
fn a_rank_s_window_computes_its_own_experts_and_skips_the_others_routes() {
    let (tokens, top, hidden, intermediate) = (4, 2, 8, 6);
    let experts: Vec<[Vec<f32>; 3]> = (0..4)
        .map(|expert| dense_expert(hidden, intermediate, 50 + 3 * expert))
        .collect();

    // This rank holds experts 2 and 3, and three of the batch's four rows are live.
    let (up_gate, down) = stacks(&experts[2..], hidden, intermediate);
    let x = sequence(tokens * hidden, 27);
    let ids = [3_u32, 1, 2, 0, 2, 3, 3, 2];
    let weights = sequence(tokens * top, 28);
    let (ids_tensor, weights_tensor) = (
        indices(&ids, (tokens, top)),
        tensor(&weights, (tokens, top), DType::F32),
    );

    let maps = functional::moe_sort(
        &ids_tensor,
        &weights_tensor,
        4,
        2..4,
        Some(&indices(&[3], 1)),
    )
    .unwrap();
    let gated = functional::gather_gemm(
        &tensor(&x, (tokens, hidden), DType::F32),
        &up_gate,
        &ids_tensor,
        &maps,
        Activation::Silu,
    )
    .unwrap();
    let products = functional::route_gemm(&gated, &down, &ids_tensor, &maps).unwrap();
    let combined = functional::moe_unpermute(&products, &maps, &weights_tensor, 4, None).unwrap();

    // A token's sum holds the routes of experts 2 and 3 alone, and a row that is not live is
    // zeros.
    let expected: Vec<f32> = (0..tokens)
        .flat_map(|token| {
            let mut sum = vec![0.0_f32; hidden];
            for route in token * top..(token + 1) * top {
                if token >= 3 || ids[route] < 2 {
                    continue;
                }
                let expert = expert_by_hand(
                    &x[token * hidden..][..hidden],
                    &experts[ids[route] as usize],
                );
                for (sum, value) in sum.iter_mut().zip(expert) {
                    *sum += weights[route] * value;
                }
            }
            sum
        })
        .collect();
    assert_close(&values(&combined), &expected, F32_TOLERANCE);
    assert!(
        values(&combined)[3 * hidden..]
            .iter()
            .all(|value| *value == 0.0)
    );
}

// ---- The NVFP4 encoding and the packed products -----------------------------------------------

/// The eight magnitudes of an E2M1 code.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// The value of an E4M3 byte: a sign, four exponent bits biased by seven, three mantissa bits,
/// with a zero exponent subnormal.
fn e4m3_value(byte: u8) -> f32 {
    let (exponent, mantissa) = (i32::from(byte >> 3 & 0xf), f32::from(byte & 7));
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

/// The byte of the E4M3 value nearest to the non-negative `value`, the even byte of two equally
/// near, saturating at the largest finite one.
fn e4m3_nearest(value: f32) -> u8 {
    let mut best = 0_u8;
    for byte in 1..=0x7e_u8 {
        let (distance, kept) = (
            (e4m3_value(byte) - value).abs(),
            (e4m3_value(best) - value).abs(),
        );
        if distance < kept || (distance == kept && byte % 2 == 0) {
            best = byte;
        }
    }
    best
}

/// The E2M1 code of `value`: the nearest magnitude, the even code of two equally near, and the
/// sign bit of a negative value.
fn e2m1_nearest(value: f32) -> u8 {
    let mut best = 0_usize;
    for (code, magnitude) in E2M1.iter().enumerate().skip(1) {
        let (distance, kept) = (
            (magnitude - value.abs()).abs(),
            (E2M1[best] - value.abs()).abs(),
        );
        if distance < kept || (distance == kept && code % 2 == 0) {
            best = code;
        }
    }

    best as u8 | if value < 0.0 { 8 } else { 0 }
}

/// The value of an E2M1 code.
fn e2m1_value(code: u8) -> f32 {
    let magnitude = E2M1[usize::from(code & 7)];
    if code & 8 == 0 { magnitude } else { -magnitude }
}

/// One row encoded under `global_scale`: its codes, one per element, and its block scales'
/// bytes.
fn encode_by_hand(row: &[f32], global_scale: f32) -> (Vec<u8>, Vec<u8>) {
    let (mut codes, mut scales) = (Vec::new(), Vec::new());

    for block in row.chunks(16) {
        let amax = block
            .iter()
            .fold(0.0_f32, |amax, value| amax.max(value.abs()));
        let byte = e4m3_nearest(amax / 6.0 * global_scale);
        let relative = e4m3_value(byte) * (1.0 / global_scale);

        scales.push(byte);
        codes.extend(block.iter().map(|value| {
            if relative == 0.0 {
                e2m1_nearest(0.0) | if *value < 0.0 { 8 } else { 0 }
            } else {
                e2m1_nearest(value * (1.0 / relative))
            }
        }));
    }
    (codes, scales)
}

/// One row encoded under `global_scale` and decoded: `code · block scale / global_scale`.
fn requantized(row: &[f32], global_scale: f32) -> Vec<f32> {
    let (codes, scales) = encode_by_hand(row, global_scale);

    codes
        .iter()
        .enumerate()
        .map(|(index, code)| {
            e2m1_value(*code) * e4m3_value(scales[index / 16]) * (1.0 / global_scale)
        })
        .collect()
}

#[test]
fn nvfp4_quantize_packs_each_row_s_codes_and_block_scales() {
    let (rows, k) = (3, 32);
    let mut x = sequence(rows * k, 29);

    // A block of zeros has a zero scale and zero codes; a wide block saturates nothing.
    x[k..k + 16].fill(0.0);
    x[5] = 3.5;
    let x = tensor(&x, (rows, k), DType::BF16);
    let rounded = values(&x);
    let global_scale = 4.0_f32;

    let hand: Vec<(Vec<u8>, Vec<u8>)> = rounded
        .chunks(k)
        .map(|row| encode_by_hand(row, global_scale))
        .collect();
    let packed: Vec<u8> = hand
        .iter()
        .flat_map(|(codes, _)| codes.chunks(2).map(|pair| pair[0] | pair[1] << 4))
        .collect();
    let scale_bytes: Vec<u8> = hand.iter().flat_map(|(_, scales)| scales.clone()).collect();

    let global = tensor(&[global_scale], 1, DType::F32);
    let (values_linear, scales_linear) =
        functional::nvfp4_quantize(&x, &global, ScaleLayout::Linear).unwrap();
    assert_eq!(
        (values_linear.dims(), values_linear.dtype()),
        (&[rows, k / 2][..], DType::U8)
    );
    assert_eq!(
        values_linear
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap(),
        packed
    );
    assert_eq!(
        (scales_linear.dims(), scales_linear.dtype()),
        (&[rows, k / 16][..], DType::U8)
    );
    assert_eq!(
        scales_linear
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap(),
        scale_bytes
    );

    // The swizzled table holds the same scales at the offsets of its 128x4 layout, zero
    // elsewhere; the codes do not depend on the layout.
    let (values_swizzled, scales_swizzled) =
        functional::nvfp4_quantize(&x, &global, ScaleLayout::Swizzled128X4).unwrap();
    let mut table = vec![0_u8; 128 * 4];
    for row in 0..rows {
        for column in 0..k / 16 {
            table[column % 4 + column / 4 * 512 + row % 32 * 16] = scale_bytes[row * 2 + column];
        }
    }
    assert_eq!(scales_swizzled.dims(), [128, 4]);
    assert_eq!(
        scales_swizzled
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap(),
        table
    );
    assert_eq!(
        values_swizzled
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap(),
        packed
    );

    // Rows that are not whole blocks, and an FP32 activation, are refused.
    assert!(
        functional::nvfp4_quantize(
            &tensor(&sequence(24, 1), (1, 24), DType::BF16),
            &global,
            ScaleLayout::Linear
        )
        .is_err()
    );
    assert!(
        functional::nvfp4_quantize(
            &tensor(&sequence(32, 1), (1, 32), DType::F32),
            &global,
            ScaleLayout::Linear
        )
        .is_err()
    );
}

/// The fields one calibrated expert projection `[rows, columns]` is written from: its packed
/// bytes, its linear block scales' bytes, its tensor scale and its activation scale.
fn packed_projection(rows: usize, columns: usize, seed: usize) -> (Vec<u8>, Vec<u8>, f32, f32) {
    let scale_values = [1.0_f32, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.375];
    let packed = (0..rows * columns / 2)
        .map(|index| (index * 37 + 11 + seed * 53) as u8)
        .collect();
    let scales = (0..rows * columns / 16)
        .map(|index| e4m3_nearest(scale_values[(index + seed) % scale_values.len()]))
        .collect();

    (
        packed,
        scales,
        0.25 + 0.125 * seed as f32,
        0.125 * (1 + seed % 3) as f32,
    )
}

/// The `[rows, columns]` matrix the fields of a calibrated projection state.
fn decode_by_hand(
    (packed, scales, tensor_scale, _): &(Vec<u8>, Vec<u8>, f32, f32),
    columns: usize,
) -> Vec<f32> {
    (0..packed.len() * 2)
        .map(|index| {
            let code = packed[index / 2] >> (index % 2 * 4) & 0xf;
            let block = index / columns * (columns / 16) + index % columns / 16;

            e2m1_value(code) * e4m3_value(scales[block]) * tensor_scale
        })
        .collect()
}

/// The modules of a packed block's experts.
const PROJECTIONS: [&str; 3] = ["up_proj", "gate_proj", "down_proj"];

/// The calibrated checkpoint of a block of `experts` packed experts, hidden 32 and intermediate
/// 16, under `mlp`, and each expert's three projections' fields.
#[expect(
    clippy::type_complexity,
    reason = "a fixture's fields, read by the loops that state the expected values"
)]
fn packed_checkpoint(
    directory: &Path,
    experts: usize,
) -> (Checkpoint, Vec<[(Vec<u8>, Vec<u8>, f32, f32); 3]>) {
    let (hidden, intermediate) = (32, 16);
    let fields: Vec<[(Vec<u8>, Vec<u8>, f32, f32); 3]> = (0..experts)
        .map(|expert| {
            [
                packed_projection(intermediate, hidden, 3 * expert),
                packed_projection(intermediate, hidden, 3 * expert + 1),
                packed_projection(hidden, intermediate, 3 * expert + 2),
            ]
        })
        .collect();

    let mut names = Vec::new();
    let mut tensors = vec![(
        "mlp.gate.weight".to_owned(),
        DType::BF16,
        vec![experts, hidden],
        encode(DType::BF16, &sequence(experts * hidden, 60)),
    )];
    for (expert, projections) in fields.iter().enumerate() {
        for (module, (packed, scales, tensor_scale, input_scale)) in
            PROJECTIONS.iter().zip(projections)
        {
            let (rows, columns) = match *module {
                "down_proj" => (hidden, intermediate),
                _ => (intermediate, hidden),
            };
            let stem = format!("mlp.experts.{expert}.{module}");

            tensors.extend([
                (
                    format!("{stem}.weight"),
                    DType::U8,
                    vec![rows, columns / 2],
                    packed.clone(),
                ),
                (
                    format!("{stem}.weight_scale"),
                    DType::F8E4M3,
                    vec![rows, columns / 16],
                    scales.clone(),
                ),
                (
                    format!("{stem}.weight_scale_2"),
                    DType::F32,
                    vec![1],
                    f32_bytes(&[*tensor_scale]),
                ),
                (
                    format!("{stem}.input_scale"),
                    DType::F32,
                    vec![1],
                    f32_bytes(&[*input_scale]),
                ),
            ]);
        }
    }
    names.extend(tensors.iter().map(|(name, ..)| name.clone()));

    let borrowed: Vec<(&str, DType, Vec<usize>, Vec<u8>)> = names
        .iter()
        .zip(tensors)
        .map(|(name, (_, dtype, shape, bytes))| (name.as_str(), dtype, shape, bytes))
        .collect();
    shard(directory, "model.safetensors", &borrowed);
    std::fs::write(
        directory.join("config.json"),
        r#"{"quantization_config": {"quant_algo": "NVFP4", "quant_method": "modelopt"}}"#,
    )
    .unwrap();

    (Checkpoint::open(directory).unwrap(), fields)
}

/// The stacks of a packed block's two experts in the arrangement the routed products read.
fn routed_stacks(loader: &WeightLoader<'_>) -> (Nvfp4Stack, Nvfp4Stack) {
    let stack = |name: &str, projections: &[&str], values: ValueLayout| {
        loader
            .nvfp4_stack(
                name,
                "mlp.experts",
                projections,
                &[0, 1],
                2,
                Nvfp4Names::vllm(),
                (values, ScaleLayout::Swizzled128X4),
                true,
            )
            .unwrap()
            .0
    };

    (
        stack(
            "mlp.experts.up_gate",
            &["up_proj", "gate_proj"],
            ValueLayout::InterleavedShuffled {
                epilogue_tile_m: 128,
            },
        ),
        stack(
            "mlp.experts.down",
            &["down_proj"],
            ValueLayout::Shuffled {
                epilogue_tile_m: 128,
            },
        ),
    )
}

#[test]
fn the_packed_products_multiply_the_encoded_rows_by_the_experts_decoded_weights() {
    let directory = tempfile::tempdir().unwrap();
    let (checkpoint, fields) = packed_checkpoint(directory.path(), 2);
    let store = CpuWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None, None);
    loader
        .declare(&Declaration::read(&checkpoint).unwrap())
        .unwrap();
    let (up_gate, down) = routed_stacks(&loader);

    let (tokens, top, hidden, intermediate) = (3_usize, 2_usize, 32_usize, 16_usize);
    let x = tensor(
        &sequence(tokens * hidden, 61),
        (tokens, hidden),
        DType::BF16,
    );
    let ids = [0_u32, 1, 1, 0, 0, 1];
    let weights = tensor(&sequence(tokens * top, 62), (tokens, top), DType::F32);
    let maps =
        functional::moe_sort(&indices(&ids, (tokens, top)), &weights, 2, 0..2, None).unwrap();
    let rows = sort_by_hand(&ids, top, 0..2, tokens)[functional::EXPANDED_TO_PERMUTED].clone();

    // One encoder serves a stack, with the smallest reciprocal of its experts' activation
    // scales: the widest calibrated activation.
    let encode_scale = |projections: &[usize]| {
        fields
            .iter()
            .flat_map(|expert| {
                projections
                    .iter()
                    .map(|projection| 1.0 / expert[*projection].3)
            })
            .fold(f32::INFINITY, f32::min)
    };

    // The first product: each route's hidden row, encoded, through its expert's up and gate.
    let gated =
        functional::gather_gemm_nvfp4(&x, &up_gate, &maps, Activation::Silu, top as u32).unwrap();
    assert_eq!(
        (gated.dims(), gated.dtype()),
        (&[2 * TILE, intermediate][..], DType::BF16)
    );
    let gated_rows = values(&gated);
    let hidden_rows = values(&x);
    for (route, row) in rows.iter().enumerate() {
        let expert = &fields[ids[route] as usize];
        let encoded = requantized(
            &hidden_rows[route / top * hidden..][..hidden],
            encode_scale(&[0, 1]),
        );
        let expected: Vec<f32> = product(&decode_by_hand(&expert[1], hidden), &encoded)
            .iter()
            .zip(product(&decode_by_hand(&expert[0], hidden), &encoded))
            .map(|(gate, up)| silu(*gate) * up)
            .collect();

        assert_close(
            &gated_rows[*row as usize * intermediate..][..intermediate],
            &expected,
            BF16_TOLERANCE,
        );
    }

    // The second: each permuted row of the first, encoded, through its expert's down.
    let products = functional::route_gemm_nvfp4(&gated, &down, &maps, top as u32).unwrap();
    assert_eq!(products.dims(), [2 * TILE, hidden]);
    let product_rows = values(&products);
    for (route, row) in rows.iter().enumerate() {
        let expert = &fields[ids[route] as usize];
        let encoded = requantized(
            &gated_rows[*row as usize * intermediate..][..intermediate],
            encode_scale(&[2]),
        );
        let expected = product(&decode_by_hand(&expert[2], intermediate), &encoded);

        assert_close(
            &product_rows[*row as usize * hidden..][..hidden],
            &expected,
            BF16_TOLERANCE,
        );
    }
}

// ---- The module -------------------------------------------------------------------------------

/// A block of four experts under `mlp`, hidden 8 and intermediate 6, routed to two.
fn dense_config() -> MoEConfig {
    MoEConfig {
        hidden: 8,
        intermediate: 6,
        experts: 4,
        top: 2,
        activation: Activation::Silu,
        renormalize: true,
        router: "gate",
        list: "experts",
        up: "up_proj",
        gate: "gate_proj",
        down: "down_proj",
        stacked: None,
        order: vec![0, 1, 2, 3],
        split: Split::WHOLE,
        exchange: Split::WHOLE,
    }
}

/// The router and the experts of [`dense_config`]'s block.
fn dense_block() -> (Vec<f32>, Vec<[Vec<f32>; 3]>) {
    (
        sequence(4 * 8, 70),
        (0..4)
            .map(|expert| dense_expert(8, 6, 71 + 3 * expert))
            .collect(),
    )
}

/// The checkpoint of [`dense_block`], one module per expert, stored as `dtype`.
fn dense_checkpoint(directory: &Path, dtype: DType) -> Checkpoint {
    let (router, experts) = dense_block();
    let mut tensors = vec![("mlp.gate.weight".to_owned(), vec![4, 8], router)];
    for (expert, [up, gate, down]) in experts.into_iter().enumerate() {
        tensors.extend([
            (
                format!("mlp.experts.{expert}.up_proj.weight"),
                vec![6, 8],
                up,
            ),
            (
                format!("mlp.experts.{expert}.gate_proj.weight"),
                vec![6, 8],
                gate,
            ),
            (
                format!("mlp.experts.{expert}.down_proj.weight"),
                vec![8, 6],
                down,
            ),
        ]);
    }

    let typed: Vec<(&str, DType, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, shape, values)| (name.as_str(), dtype, shape.clone(), encode(dtype, values)))
        .collect();
    shard(directory, "model.safetensors", &typed);

    Checkpoint::open(directory).unwrap()
}

/// `Σ_routes weight · expert(x)` for every row of `x`, by the block's own router and experts.
fn block_by_hand(x: &[f32]) -> Vec<f32> {
    let (router, experts) = dense_block();
    let scores: Vec<f32> = x.chunks(8).flat_map(|row| product(&router, row)).collect();
    let (ids, weights) = route_by_hand(&scores, 4, 2);

    x.chunks(8)
        .enumerate()
        .flat_map(|(token, row)| {
            let mut sum = vec![0.0_f32; 8];
            for route in token * 2..(token + 1) * 2 {
                let expert = expert_by_hand(row, &experts[ids[route] as usize]);
                for (sum, value) in sum.iter_mut().zip(expert) {
                    *sum += weights[route] * value;
                }
            }
            sum
        })
        .collect()
}

#[test]
fn a_block_loads_its_experts_and_sums_each_token_s_routes() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = dense_checkpoint(directory.path(), DType::F32);
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);

    let block = MoE::load(&dense_config(), loader.root().pp("mlp")).unwrap();
    assert_eq!(
        (block.experts(), block.local_experts(), block.top()),
        (4, 4, 2)
    );

    let x = sequence(5 * 8, 80);
    let x_tensor = tensor(&x, (5, 8), DType::F32);
    let out = block.forward(&x_tensor, None, None).unwrap();
    assert_eq!(out.dims(), [5, 8]);
    assert_close(&values(&out), &block_by_hand(&x), F32_TOLERANCE);

    // Where the experts are stored changes no token's output: the stacks are laid out in the
    // order's slots, and each route is written as its expert's slot.
    let redistributed = MoE::load(
        &MoEConfig {
            order: vec![2, 0, 3, 1],
            ..dense_config()
        },
        loader.root().pp("mlp"),
    )
    .unwrap();
    let moved = redistributed.forward(&x_tensor, None, None).unwrap();
    assert_close(&values(&moved), &block_by_hand(&x), F32_TOLERANCE);

    let (router, _) = dense_block();
    let scores: Vec<f32> = x.chunks(8).flat_map(|row| product(&router, row)).collect();
    let (chosen, weights) = route_by_hand(&scores, 4, 2);
    let slot_of = [1_u32, 3, 0, 2];
    let (slots, routed_weights) = redistributed.route(&x_tensor).unwrap();
    assert_eq!(
        entries(&slots),
        chosen
            .iter()
            .map(|expert| slot_of[*expert as usize])
            .collect::<Vec<_>>()
    );
    assert_close(&values(&routed_weights), &weights, F32_TOLERANCE);

    // The rows a batch does not fill route nowhere.
    let live = block
        .forward(&x_tensor, Some(&indices(&[3], 1)), None)
        .unwrap();
    let live = values(&live);
    assert_close(&live[..3 * 8], &block_by_hand(&x)[..3 * 8], F32_TOLERANCE);
    assert!(live[3 * 8..].iter().all(|value| *value == 0.0));
}

#[test]
fn a_block_reads_stacked_experts_as_it_reads_one_module_each() {
    let modules = tempfile::tempdir().unwrap();
    let checkpoint = dense_checkpoint(modules.path(), DType::F32);
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);
    let config = MoEConfig {
        order: vec![3, 1, 0, 2],
        stacked: Some(["gate_up_proj", "down_proj"]),
        ..dense_config()
    };
    let block = MoE::load(&config, loader.root().pp("mlp")).unwrap();

    // The same experts as two stacked tensors: each expert's gate rows before its up rows.
    let stacked = tempfile::tempdir().unwrap();
    let (router, experts) = dense_block();
    let gate_up: Vec<f32> = experts
        .iter()
        .flat_map(|[up, gate, _]| gate.iter().chain(up).copied())
        .collect();
    let down: Vec<f32> = experts
        .iter()
        .flat_map(|[_, _, down]| down.iter().copied())
        .collect();
    shard(
        stacked.path(),
        "model.safetensors",
        &[
            (
                "mlp.gate.weight",
                DType::F32,
                vec![4, 8],
                f32_bytes(&router),
            ),
            (
                "mlp.experts.gate_up_proj",
                DType::F32,
                vec![4, 12, 8],
                f32_bytes(&gate_up),
            ),
            (
                "mlp.experts.down_proj",
                DType::F32,
                vec![4, 8, 6],
                f32_bytes(&down),
            ),
        ],
    );
    let stacked_checkpoint = Checkpoint::open(stacked.path()).unwrap();
    let stacked_store = CpuWeights::new();
    let stacked_loader = WeightLoader::new(&stacked_checkpoint, &stacked_store, None, None);
    let from_stacks = MoE::load(&config, stacked_loader.root().pp("mlp")).unwrap();

    let x = tensor(&sequence(5 * 8, 81), (5, 8), DType::F32);
    assert_eq!(
        values(&from_stacks.forward(&x, None, None).unwrap()),
        values(&block.forward(&x, None, None).unwrap())
    );
}

#[test]
fn a_block_refuses_a_routing_or_a_share_its_checkpoint_does_not_have() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = dense_checkpoint(directory.path(), DType::F32);
    let loader = WeightLoader::meta(&checkpoint, None, None);
    let refused = |config: MoEConfig| {
        MoE::load(&config, loader.root().pp("mlp"))
            .err()
            .expect("a refused block")
            .to_string()
    };

    // No route, more routes than the router's kernel holds, and more than the experts.
    for top in [0, 9, 5] {
        let message = refused(MoEConfig {
            top,
            ..dense_config()
        });
        assert!(message.contains("top"), "{message}");
    }

    // An order that is not every expert once.
    for order in [vec![0, 1, 2], vec![0, 0, 1, 2], vec![0, 1, 2, 4]] {
        let message = refused(MoEConfig {
            order,
            ..dense_config()
        });
        assert!(message.contains("permutation"), "{message}");
    }

    // Splits that do not divide the experts or their width, and a rank outside its group.
    let third = Split { size: 3, rank: 0 };
    let outside = Split { size: 2, rank: 2 };
    for config in [
        MoEConfig {
            exchange: third,
            ..dense_config()
        },
        MoEConfig {
            exchange: outside,
            ..dense_config()
        },
        MoEConfig {
            split: Split { size: 4, rank: 0 },
            ..dense_config()
        },
        MoEConfig {
            split: outside,
            ..dense_config()
        },
    ] {
        let message = refused(config);
        assert!(
            message.contains("mlp") && message.contains("group"),
            "{message}"
        );
    }

    // A router of another expert count than the block routes over.
    let message = refused(MoEConfig {
        experts: 8,
        order: (0..8).collect(),
        ..dense_config()
    });
    assert!(message.contains("mlp.gate.weight"), "{message}");
}

#[test]
fn a_block_split_over_a_tensor_group_holds_its_share_and_reduces_its_sum() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = dense_checkpoint(directory.path(), DType::BF16);
    let config = MoEConfig {
        split: Split { size: 2, rank: 1 },
        ..dense_config()
    };

    // The second rank holds rows 3..6 of every gate and up projection and columns 3..6 of every
    // down projection.
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);
    MoE::load(&config, loader.root().pp("mlp")).unwrap();
    let keys: Vec<String> = store.keys().iter().map(ToString::to_string).collect();
    for piece in [
        "|bf16|up_gate|mlp.experts.0.up_proj.weight+mlp.experts.0.gate_proj.weight+",
        "[24..216][4, 6, 8]|stride_8",
        "|bf16|down|mlp.experts.0.down_proj.weight+",
        "[3..99][4, 8, 3]|stride_6",
    ] {
        assert!(
            keys.iter().any(|key| key.contains(piece)),
            "no key holds `{piece}`: {keys:#?}"
        );
    }

    // Its output is a partial sum: a forward with no group is refused, and one with a group
    // states the same kernels as a whole block over half the width.
    let meta_loader = WeightLoader::meta(&checkpoint, None, None);
    let block = MoE::load(&config, meta_loader.root().pp("mlp")).unwrap();
    let (alone, _) = recorded(|| block.forward(&meta((4, 8), DType::BF16), None, None));
    let message = alone.err().unwrap().to_string();
    assert!(message.contains("no group"), "{message}");

    let (grouped, launches) = recorded_split(Some((1, 2)), || {
        block.forward(&meta((4, 8), DType::BF16), None, None)
    });
    assert_eq!(grouped.unwrap().dims(), [4, 8]);
    assert_eq!(
        launches[2..],
        [
            (
                "moe_sort",
                "experts_4_top_2_local_4_tile_128_scales_f32_counted_true"
            ),
            (
                "gather_gemm",
                "hidden_8_intermediate_3_experts_4_top_2_activation_silu_dtype_bf16_\
                 tile_128x128_cluster_1x1"
            ),
            (
                "route_gemm",
                "hidden_8_intermediate_3_experts_4_top_2_dtype_bf16_tile_128x128_cluster_1x1"
            ),
            (
                "moe_unpermute",
                "layout_permuted_hidden_8_top_2_experts_4_input_bf16_scales_f32_output_bf16"
            ),
        ]
        .map(|(operator, key)| (operator, key.to_owned()))
    );
}

#[test]
fn a_dense_block_states_its_routing_its_products_and_its_combine() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = dense_checkpoint(directory.path(), DType::BF16);
    let loader = WeightLoader::meta(&checkpoint, None, None);
    let block = MoE::load(&dense_config(), loader.root().pp("mlp")).unwrap();

    let (output, launches) = recorded(|| {
        block.forward(
            &meta((15, 8), DType::BF16),
            Some(&meta(1, DType::U32)),
            None,
        )
    });
    assert_eq!(output.unwrap().dims(), [15, 8]);
    assert_eq!(
        operators(&launches),
        [
            "gemm",
            "topk_softmax",
            "moe_sort",
            "gather_gemm",
            "route_gemm",
            "moe_unpermute"
        ]
    );
    assert_eq!(
        launches[1].1,
        "scores_bf16_experts_4_top_2_renormalize_true_scale_none"
    );
    assert_eq!(
        launches[2].1,
        "experts_4_top_2_local_4_tile_128_scales_f32_counted_true"
    );
}

/// A stack of `experts` packed matrices `[rows, columns]` as meta tensors: what a product's
/// kernels are named by.
fn meta_stack(experts: usize, rows: usize, columns: usize, projections: usize) -> Nvfp4Stack {
    let scales = ScaleLayout::Swizzled128X4.shape(rows as u64, columns as u64);

    Nvfp4Stack {
        values: meta((experts, rows, columns / 2), DType::U8),
        scales: meta(
            (experts, scales[0] as usize, scales[1] as usize),
            DType::F8E4M3,
        ),
        global_scale: meta((projections, experts), DType::F32),
        layout: ScaleLayout::Swizzled128X4,
        encode_scale: meta(1, DType::F32),
        input_scale: None,
        activation_scale: 1.0,
    }
}

/// The packed products are named by the checkpoint's scale type, and each product and the
/// encoding that feeds it name one activation layout: the gated product reads its tokens' block
/// scales in the linear form and the second product in the swizzled one. The keys are those of
/// a block of 128 experts of 768 columns over a hidden width of 2048, routed to eight.
#[test]
fn the_packed_products_name_the_checkpoint_s_scale_type_and_their_activation_layouts() {
    let (tokens, top) = (15, 8);
    let (up_gate, down) = (
        meta_stack(128, 1536, 2048, 2),
        meta_stack(128, 2048, 768, 1),
    );

    let (output, launches) = recorded(|| {
        let ids = meta((tokens, top), DType::U32);
        let weights = meta((tokens, top), DType::F32);
        let maps = functional::moe_sort(&ids, &weights, 128, 0..128, None)?;

        let gated = functional::gather_gemm_nvfp4(
            &meta((tokens, 2048), DType::BF16),
            &up_gate,
            &maps,
            Activation::Silu,
            top as u32,
        )?;
        let products = functional::route_gemm_nvfp4(&gated, &down, &maps, top as u32)?;
        functional::moe_unpermute(&products, &maps, &weights, 128, None)
    });

    // Fifteen tokens of eight routes over 128 experts are 120 tiles of 128 permuted rows.
    assert_eq!(output.unwrap().dims(), [tokens, 2048]);
    assert_eq!(
        launches,
        [
            (
                "moe_sort",
                "experts_128_top_8_local_128_tile_128_scales_f32_counted_true"
            ),
            (
                "nvfp4_quantize",
                "input_bf16_columns_2048_layout_linear_block_1024_pdl_false"
            ),
            (
                "gather_gemm_nvfp4",
                "hidden_2048_intermediate_768_experts_128_top_8_activation_silu_dtype_bf16_\
                 scale_f8e4m3_layout_linear_tile_128x128_cluster_1x1"
            ),
            (
                "nvfp4_quantize",
                "input_bf16_columns_768_layout_swizzled_128x4_block_256_pdl_false"
            ),
            (
                "route_gemm_nvfp4",
                "hidden_2048_intermediate_768_experts_128_top_8_dtype_bf16_scale_f8e4m3_\
                 layout_swizzled_128x4_tile_128x128_cluster_1x1"
            ),
            (
                "moe_unpermute",
                "layout_permuted_hidden_2048_top_8_experts_128_input_bf16_scales_f32_output_bf16"
            ),
        ]
        .map(|(operator, key)| (operator, key.to_owned()))
    );
}

/// A block of two packed experts under `mlp`, hidden 32 and intermediate 16, routed to both.
fn packed_config() -> MoEConfig {
    MoEConfig {
        hidden: 32,
        intermediate: 16,
        experts: 2,
        top: 2,
        order: vec![0, 1],
        ..dense_config()
    }
}

#[test]
fn a_calibrated_block_runs_the_packed_products_over_its_stacks() {
    let directory = tempfile::tempdir().unwrap();
    let (checkpoint, _) = packed_checkpoint(directory.path(), 2);
    let store = CpuWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None, None);
    loader
        .declare(&Declaration::read(&checkpoint).unwrap())
        .unwrap();
    let block = MoE::load(&packed_config(), loader.root().pp("mlp")).unwrap();

    // The block computes what the functions compute over the stacks in the routed products'
    // arrangement, which the test above holds against the checkpoint's own fields.
    let x = tensor(&sequence(3 * 32, 63), (3, 32), DType::BF16);
    let (up_gate, down) = routed_stacks(&loader);
    let (ids, weights) = block.route(&x).unwrap();
    let maps = functional::moe_sort(&ids, &weights, 2, 0..2, None).unwrap();
    let gated = functional::gather_gemm_nvfp4(&x, &up_gate, &maps, Activation::Silu, 2).unwrap();
    let products = functional::route_gemm_nvfp4(&gated, &down, &maps, 2).unwrap();
    let expected = functional::moe_unpermute(&products, &maps, &weights, 2, None).unwrap();

    let out = block.forward(&x, None, None).unwrap();
    assert_eq!((out.dims(), out.dtype()), (&[3, 32][..], DType::BF16));
    assert_eq!(values(&out), values(&expected));
    assert!(values(&out).iter().any(|value| *value != 0.0));

    // Stacks of another width than the block states, and packed experts split by a tensor
    // group, are refused at load.
    for config in [
        MoEConfig {
            intermediate: 32,
            ..packed_config()
        },
        MoEConfig {
            split: Split { size: 2, rank: 0 },
            ..packed_config()
        },
    ] {
        let message = MoE::load(&config, loader.root().pp("mlp"))
            .err()
            .expect("a refused block")
            .to_string();
        assert!(message.contains("packed"), "{message}");
    }
}

// ---- The exchange -----------------------------------------------------------------------------

/// The rows of an exchange over `ranks` ranks of `rank_rows` rows each, of `hidden` elements
/// routed to `top` experts, as meta tensors.
fn exchange(ranks: u32, rank_rows: u32, hidden: usize, top: usize) -> ExpertExchange {
    let rows = (ranks * rank_rows) as usize;

    ExpertExchange {
        ranks,
        attention: 0,
        rank_rows,
        step_rows: rank_rows,
        states: meta((rows, hidden), DType::BF16),
        ids: meta((rows, top), DType::U32),
        weights: meta((rows, top), DType::F32),
        combined: meta((rows, hidden), DType::BF16),
        live: meta(1, DType::U32),
        packed: meta((rank_rows as usize, hidden / 2), DType::U8),
        packed_scales: meta((rank_rows as usize, hidden / 16), DType::F8E4M3),
        fused: false,
        source_only: false,
    }
}

/// The key of one launch of the dense block's exchange in `mode`.
fn exchange_key(mode: &str) -> String {
    format!(
        "mode_{mode}_hidden_8_intermediate_6_top_2_experts_4_max_tokens_4_ranks_2_attention_0_\
         dtype_bf16"
    )
}

#[test]
fn a_block_over_an_expert_group_exchanges_its_routes_around_its_own_experts() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = dense_checkpoint(directory.path(), DType::BF16);
    let loader = WeightLoader::meta(&checkpoint, None, None);
    let block = MoE::load(
        &MoEConfig {
            exchange: Split { size: 2, rank: 1 },
            ..dense_config()
        },
        loader.root().pp("mlp"),
    )
    .unwrap();
    assert_eq!((block.experts(), block.local_experts()), (4, 2));

    // Without the group's exchange the block has no rows to read.
    let (alone, _) = recorded(|| block.forward(&meta((3, 8), DType::BF16), None, None));
    let message = alone.err().unwrap().to_string();
    assert!(message.contains("exchange"), "{message}");

    // With it: dispatch, the rank's own experts over the rows it received, and combine. The
    // received rows are those of every rank, so the sort is over eight rows of this rank's two
    // experts.
    let lent = exchange(2, 4, 8, 2);
    let live_tokens = meta(1, DType::U32);
    let (output, launches) = recorded_split(Some((1, 2)), || {
        block.forward(&meta((3, 8), DType::BF16), Some(&live_tokens), Some(&lent))
    });
    assert_eq!(output.unwrap().dims(), [3, 8]);
    assert_eq!(
        launches[1..],
        [
            (
                "topk_softmax",
                "scores_bf16_experts_4_top_2_renormalize_true_scale_none".to_owned()
            ),
            ("expert_exchange", exchange_key("dispatch")),
            (
                "moe_sort",
                "experts_4_top_2_local_2_tile_128_scales_f32_counted_true".to_owned()
            ),
            (
                "gather_gemm",
                "hidden_8_intermediate_6_experts_2_top_2_activation_silu_dtype_bf16_\
                 tile_128x128_cluster_1x1"
                    .to_owned()
            ),
            (
                "route_gemm",
                "hidden_8_intermediate_6_experts_2_top_2_dtype_bf16_tile_128x128_cluster_1x1"
                    .to_owned()
            ),
            (
                "moe_unpermute",
                "layout_permuted_hidden_8_top_2_experts_4_input_bf16_scales_f32_output_bf16"
                    .to_owned()
            ),
            ("expert_exchange", exchange_key("combine")),
        ]
    );

    // A rank with no tokens of its own joins the step: it sends nothing and serves its peers.
    let (joined, launches) = recorded_split(Some((1, 2)), || block.join_expert_step(&lent));
    joined.unwrap();
    assert_eq!(
        operators(&launches),
        [
            "expert_exchange",
            "moe_sort",
            "gather_gemm",
            "route_gemm",
            "moe_unpermute",
            "expert_exchange"
        ]
    );

    // A step shaped for fewer rows per rank sorts and multiplies that many of each rank's.
    let narrow = ExpertExchange {
        step_rows: 2,
        ..lent.clone()
    };
    let (stepped, _) = recorded(|| block.forward(&meta((2, 8), DType::BF16), None, Some(&narrow)));
    assert_eq!(stepped.unwrap().dims(), [2, 8]);

    // A rank that holds no experts sends its routes and takes its sums back.
    let source = ExpertExchange {
        source_only: true,
        ..lent.clone()
    };
    let (sent, launches) =
        recorded(|| block.forward(&meta((3, 8), DType::BF16), None, Some(&source)));
    sent.unwrap();
    assert_eq!(
        operators(&launches)[2..],
        ["expert_exchange", "expert_exchange"]
    );

    // More tokens than a rank's block carries, and an exchange of another group, are refused.
    let (crowded, _) = recorded(|| block.forward(&meta((5, 8), DType::BF16), None, Some(&lent)));
    let message = crowded.err().unwrap().to_string();
    assert!(message.contains("5 tokens"), "{message}");

    let wider = exchange(4, 4, 8, 2);
    let (other, _) = recorded(|| block.forward(&meta((3, 8), DType::BF16), None, Some(&wider)));
    let message = other.err().unwrap().to_string();
    assert!(message.contains("4 ranks"), "{message}");
}

#[test]
fn a_calibrated_block_over_an_expert_group_runs_in_the_exchange_s_one_launch() {
    let directory = tempfile::tempdir().unwrap();
    let (checkpoint, _) = packed_checkpoint(directory.path(), 2);
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    loader
        .declare(&Declaration::read(&checkpoint).unwrap())
        .unwrap();
    let block = MoE::load(
        &MoEConfig {
            exchange: Split { size: 2, rank: 0 },
            ..packed_config()
        },
        loader.root().pp("mlp"),
    )
    .unwrap();
    assert_eq!(block.local_experts(), 1);

    let fused = ExpertExchange {
        fused: true,
        ..exchange(2, 4, 32, 2)
    };
    let (output, launches) =
        recorded(|| block.forward(&meta((3, 32), DType::BF16), None, Some(&fused)));
    assert_eq!(output.unwrap().dims(), [3, 32]);
    assert_eq!(
        launches[2..],
        [
            (
                "nvfp4_quantize",
                "input_bf16_columns_32_layout_linear_block_32_pdl_false"
            ),
            (
                "expert_exchange",
                "mode_fused_hidden_32_intermediate_16_top_2_experts_2_max_tokens_4_ranks_2_\
                 attention_0_dtype_bf16"
            ),
        ]
        .map(|(operator, key)| (operator, key.to_owned()))
    );

    // A join encodes nothing: it has no rows of its own.
    let (joined, launches) = recorded(|| block.join_expert_step(&fused));
    joined.unwrap();
    assert_eq!(operators(&launches), ["expert_exchange"]);

    // Packed experts over a group run fused, and an exchange of two launches refuses them.
    let two = exchange(2, 4, 32, 2);
    let (refused, _) = recorded(|| block.forward(&meta((3, 32), DType::BF16), None, Some(&two)));
    assert!(refused.err().unwrap().to_string().contains("fused"));
}

#[test]
fn an_exchange_s_rows_are_checked_against_its_own_extents() {
    exchange(2, 64, 2048, 8).check().unwrap();

    // Rows of another rank count, ids of another type, a step beyond a rank's block.
    let mut combined = exchange(2, 64, 2048, 8);
    combined.combined = exchange(4, 64, 2048, 8).combined;
    assert!(
        combined
            .check()
            .unwrap_err()
            .to_string()
            .contains("combined")
    );

    let mut ids = exchange(2, 64, 2048, 8);
    ids.ids = meta((128, 8), DType::I32);
    assert!(ids.check().unwrap_err().to_string().contains("ids"));

    let step = ExpertExchange {
        step_rows: 65,
        ..exchange(2, 64, 2048, 8)
    };
    assert!(step.check().is_err());

    let mut none = exchange(2, 64, 2048, 8);
    none.ids = meta((128, 0), DType::U32);
    assert!(none.check().is_err());
}

// ---- The shared expert ------------------------------------------------------------------------

#[test]
fn sigmoid_gated_add_updates_its_target_s_storage() {
    let (rows, width) = (3, 4);
    let (target, update, gate) = (
        sequence(rows * width, 90),
        sequence(rows * width, 91),
        [0.5_f32, -1.5, 2.0],
    );
    let expected: Vec<f32> = (0..rows * width)
        .map(|index| target[index] + update[index] / (1.0 + (-gate[index / width]).exp()))
        .collect();

    // A second tensor over the target's storage sees the sum: nothing was copied.
    let storage = tensor(&target, (rows, width), DType::F32);
    let returned = functional::sigmoid_gated_add_(
        storage.clone(),
        &tensor(&update, (rows, width), DType::F32),
        &tensor(&gate, (rows, 1), DType::F32),
    )
    .unwrap();
    assert_close(&values(&returned), &expected, F32_TOLERANCE);
    assert_eq!(values(&storage), values(&returned));

    // One logit per row, of the target's type, over the update's rows.
    let rows_of = |values: &[f32], shape: (usize, usize)| tensor(values, shape, DType::F32);
    assert!(
        functional::sigmoid_gated_add_(
            rows_of(&target, (rows, width)),
            &rows_of(&update, (rows, width)),
            &rows_of(&update, (rows, width)),
        )
        .is_err()
    );
    assert!(
        functional::sigmoid_gated_add_(
            rows_of(&target, (rows, width)),
            &rows_of(&update[..8], (2, width)),
            &rows_of(&gate, (rows, 1)),
        )
        .is_err()
    );

    // On the device the target is the kernel's one in-out argument.
    let (output, launches) = recorded(|| {
        functional::sigmoid_gated_add_(
            meta((rows, 2048), DType::BF16),
            &meta((rows, 2048), DType::BF16),
            &meta((rows, 1), DType::BF16),
        )
    });
    assert_eq!(output.unwrap().dims(), [rows, 2048]);
    assert_eq!(
        launches,
        [(
            "sigmoid_gated_add",
            "dtype_bf16_columns_2048_block_1024".to_owned()
        )]
    );
}

#[test]
fn a_shared_expert_adds_its_gated_mlp_into_the_routed_output() {
    let (hidden, intermediate, rows) = (4, 6, 3);
    let (gate_proj, up_proj, down_proj, gate) = (
        sequence(intermediate * hidden, 92),
        sequence(intermediate * hidden, 93),
        sequence(hidden * intermediate, 94),
        sequence(hidden, 95),
    );
    let directory = tempfile::tempdir().unwrap();
    shard(
        directory.path(),
        "model.safetensors",
        &[
            (
                "mlp.shared_expert.gate_proj.weight",
                DType::F32,
                vec![intermediate, hidden],
                f32_bytes(&gate_proj),
            ),
            (
                "mlp.shared_expert.up_proj.weight",
                DType::F32,
                vec![intermediate, hidden],
                f32_bytes(&up_proj),
            ),
            (
                "mlp.shared_expert.down_proj.weight",
                DType::F32,
                vec![hidden, intermediate],
                f32_bytes(&down_proj),
            ),
            (
                "mlp.shared_expert_gate.weight",
                DType::F32,
                vec![1, hidden],
                f32_bytes(&gate),
            ),
        ],
    );
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let config = SharedExpertConfig {
        mlp: MlpConfig {
            hidden: hidden as u32,
            intermediate: intermediate as u32,
            activation: Activation::Silu,
            rounding: Rounding::Fused,
            gate_up: GateUp::Split {
                gate: "gate_proj",
                up: "up_proj",
            },
            down: "down_proj",
            bias: false,
            split: Split::WHOLE,
        },
        module: "shared_expert",
        gate: "shared_expert_gate",
    };

    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);
    let shared = SharedExpert::load(&config, loader.root().pp("mlp")).unwrap();

    let (x, routed) = (sequence(rows * hidden, 96), sequence(rows * hidden, 97));
    let expected: Vec<f32> = x
        .chunks(hidden)
        .zip(routed.chunks(hidden))
        .flat_map(|(row, routed)| {
            let mlp = expert_by_hand(
                row,
                &[up_proj.clone(), gate_proj.clone(), down_proj.clone()],
            );
            let logit = product(&gate, row)[0];

            routed
                .iter()
                .zip(mlp)
                .map(|(routed, mlp)| routed + mlp / (1.0 + (-logit).exp()))
                .collect::<Vec<_>>()
        })
        .collect();

    // The routed output is updated in place and returned.
    let routed = tensor(&routed, (rows, hidden), DType::F32);
    let out = shared
        .forward(&tensor(&x, (rows, hidden), DType::F32), routed.clone())
        .unwrap();
    assert_close(&values(&out), &expected, F32_TOLERANCE);
    assert_eq!(values(&routed), values(&out));

    // Under a meta context: the MLP's products and gate, the gate's logit, and the gated add.
    let meta_loader = WeightLoader::meta(&checkpoint, None, None);
    let shared = SharedExpert::load(&config, meta_loader.root().pp("mlp")).unwrap();
    let (output, launches) = recorded(|| {
        shared.forward(
            &meta((rows, hidden), DType::F32),
            meta((rows, hidden), DType::F32),
        )
    });
    assert_eq!(output.unwrap().dims(), [rows, hidden]);
    assert_eq!(
        operators(&launches),
        [
            "gemm",
            "gemm",
            "act_and_mul",
            "gemm",
            "gemm",
            "sigmoid_gated_add"
        ]
    );
    assert_eq!(launches[5].1, "dtype_f32_columns_4_block_4");
}
