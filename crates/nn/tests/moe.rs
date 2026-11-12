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

use nn::functional::{self, Activation};
use nn::quantization::{Declaration, Nvfp4Names, Nvfp4Stack, ValueLayout};
use nn::{Checkpoint, CpuWeights, DType, Device, Tensor, WeightLoader};
use ops::fp4::ScaleLayout;
use support::{assert_close, encode, f32_bytes, meta, recorded, sequence, shard, tensor, values};

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
    loader.declare(&Declaration::read(&checkpoint).unwrap());
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

/// A stack of `experts` packed matrices `[rows, columns]` as meta tensors: what a product's
/// kernels are named by.
fn meta_stack(experts: usize, rows: usize, columns: usize, projections: usize) -> Nvfp4Stack {
    let scales = ScaleLayout::Swizzled128X4.shape(rows as u64, columns as u64);

    Nvfp4Stack {
        values: meta((experts, rows, columns / 2), DType::U8),
        scales: meta(
            (experts, scales.dim(0) as usize, scales.dim(1) as usize),
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
