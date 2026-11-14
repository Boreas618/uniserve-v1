//! The functions of `nn::functional`: their arithmetic on CPU tensors against each formula written
//! out in loops, and the kernels they state under a meta context.
//!
//! Nothing here touches a GPU: a CPU tensor is computed by the function's formula, and a meta
//! context launches nothing.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

mod support;

use nn::functional::{self, Activation, WeightOrder};
use nn::{DType, Device, Tensor};
use support::{assert_close, meta, operators, recorded, sequence, tensor, values};

/// The tolerance of an FP32 result against a reference accumulated in another order.
const F32_TOLERANCE: f32 = 2e-5;

/// `rows` rows of `width` values each, as `[rows, width]` nested for a loop to read.
fn rows_of(values: &[f32], width: usize) -> Vec<&[f32]> {
    values.chunks(width).collect()
}

// ---- Normalization ----------------------------------------------------------------------------

/// One row normalized by its root mean square and scaled by `gain`.
fn rms_row(row: &[f32], gain: &[f32], eps: f32) -> Vec<f32> {
    let mean_square = row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();

    row.iter()
        .zip(gain)
        .map(|(value, gain)| value * scale * gain)
        .collect()
}

#[test]
fn rms_norm_scales_each_row_by_its_root_mean_square_and_its_gain() {
    let (x, weight) = (sequence(3 * 8, 1), sequence(8, 2));
    let eps = 1e-6;

    let plain = functional::rms_norm(
        &tensor(&x, (3, 8), DType::F32),
        &tensor(&weight, 8, DType::F32),
        eps,
        false,
        DType::F32,
    )
    .unwrap();
    let expected: Vec<f32> = rows_of(&x, 8)
        .iter()
        .flat_map(|row| rms_row(row, &weight, eps))
        .collect();
    assert_close(&values(&plain), &expected, F32_TOLERANCE);

    // A zero-centred weight scales by one plus itself.
    let centred = functional::rms_norm(
        &tensor(&x, (3, 8), DType::F32),
        &tensor(&weight, 8, DType::F32),
        eps,
        true,
        DType::F32,
    )
    .unwrap();
    let gain: Vec<f32> = weight.iter().map(|weight| 1.0 + weight).collect();
    let expected: Vec<f32> = rows_of(&x, 8)
        .iter()
        .flat_map(|row| rms_row(row, &gain, eps))
        .collect();
    assert_close(&values(&centred), &expected, F32_TOLERANCE);
}

#[test]
fn rms_norm_reads_leading_axes_as_rows_and_narrows_into_its_output_type() {
    let (x, weight) = (sequence(2 * 3 * 4, 3), sequence(4, 4));
    let narrowed = functional::rms_norm(
        &tensor(&x, (2, 3, 4), DType::F32),
        &tensor(&weight, 4, DType::F32),
        1e-5,
        false,
        DType::BF16,
    )
    .unwrap();

    assert_eq!(narrowed.dims(), [2, 3, 4]);
    assert_eq!(narrowed.dtype(), DType::BF16);

    // The FP32 formula, rounded once into BF16.
    let expected: Vec<f32> = rows_of(&x, 4)
        .iter()
        .flat_map(|row| rms_row(row, &weight, 1e-5))
        .collect();
    assert_close(
        &values(&narrowed),
        &values(&tensor(&expected, (2, 3, 4), DType::BF16)),
        1.0 / 128.0,
    );

    // No kernel adds the one of a zero-centred gain while it narrows.
    let refused = functional::rms_norm(
        &tensor(&x, (2, 3, 4), DType::F32),
        &tensor(&weight, 4, DType::F32),
        1e-5,
        true,
        DType::BF16,
    );
    assert!(refused.unwrap_err().to_string().contains("zero-centred"));
}

#[test]
fn add_rms_norm_returns_the_normalized_sum_and_the_sum() {
    let (x, residual, weight) = (sequence(2 * 8, 5), sequence(2 * 8, 6), sequence(8, 7));
    let (normalized, sum) = functional::add_rms_norm(
        &tensor(&x, (2, 8), DType::F32),
        &tensor(&residual, (2, 8), DType::F32),
        &tensor(&weight, 8, DType::F32),
        1e-6,
        false,
    )
    .unwrap();

    let summed: Vec<f32> = x.iter().zip(&residual).map(|(x, r)| x + r).collect();
    assert_close(&values(&sum), &summed, F32_TOLERANCE);

    let expected: Vec<f32> = rows_of(&summed, 8)
        .iter()
        .flat_map(|row| rms_row(row, &weight, 1e-6))
        .collect();
    assert_close(&values(&normalized), &expected, F32_TOLERANCE);
}

/// The stream stores the sum rounded once, and the normalization reads the sum before that
/// rounding: with a BF16 stream the two differ where the sum is not a BF16 value.
#[test]
fn add_rms_norm_normalizes_the_unrounded_sum() {
    // 1 + 2⁻⁹ is not a BF16 value, and neither addend alone loses it.
    let x = [1.0_f32, 0.5, 0.25, 2.0];
    let residual = [2f32.powi(-9), 0.0, 0.0, 0.0];
    let weight = [1.0_f32; 4];

    let (normalized, sum) = functional::add_rms_norm(
        &tensor(&x, (1, 4), DType::BF16),
        &tensor(&residual, (1, 4), DType::BF16),
        &tensor(&weight, 4, DType::BF16),
        0.0,
        false,
    )
    .unwrap();
    assert_eq!(values(&sum), [1.0, 0.5, 0.25, 2.0], "the stored sum rounds");

    let exact: Vec<f32> = x.iter().zip(&residual).map(|(x, r)| x + r).collect();
    let expected = tensor(&rms_row(&exact, &weight, 0.0), (1, 4), DType::BF16);
    assert_eq!(values(&normalized), values(&expected));
}

#[test]
fn gated_rms_norm_normalizes_then_gates_by_the_silu() {
    let (x, gate, weight) = (sequence(2 * 4, 8), sequence(2 * 4, 9), sequence(4, 10));
    let gated = functional::gated_rms_norm(
        &tensor(&x, (2, 4), DType::F32),
        &tensor(&gate, (2, 4), DType::F32),
        &tensor(&weight, 4, DType::F32),
        1e-6,
        false,
    )
    .unwrap();

    let normalized: Vec<f32> = rows_of(&x, 4)
        .iter()
        .flat_map(|row| rms_row(row, &weight, 1e-6))
        .collect();
    let expected: Vec<f32> = normalized
        .iter()
        .zip(&gate)
        .map(|(value, gate)| value * (gate / (1.0 + (-gate).exp())))
        .collect();
    assert_close(&values(&gated), &expected, F32_TOLERANCE);
}

#[test]
fn layer_norm_centres_scales_and_shifts_each_row() {
    let (x, weight, bias) = (sequence(3 * 6, 11), sequence(6, 12), sequence(6, 13));
    let eps = 1e-5;
    let reference = |with_bias: bool| -> Vec<f32> {
        rows_of(&x, 6)
            .iter()
            .flat_map(|row| {
                let mean = row.iter().sum::<f32>() / 6.0;
                let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 6.0;
                let scale = 1.0 / (variance + eps).sqrt();

                row.iter()
                    .enumerate()
                    .map(|(column, value)| {
                        let shift = if with_bias { bias[column] } else { 0.0 };
                        (value - mean) * scale * weight[column] + shift
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    };

    let x = tensor(&x, (3, 6), DType::F32);
    let weight_tensor = tensor(&weight, 6, DType::F32);
    let bias_tensor = tensor(&bias, 6, DType::F32);

    let shifted = functional::layer_norm(&x, &weight_tensor, Some(&bias_tensor), eps).unwrap();
    assert_close(&values(&shifted), &reference(true), F32_TOLERANCE);

    let plain = functional::layer_norm(&x, &weight_tensor, None, eps).unwrap();
    assert_close(&values(&plain), &reference(false), F32_TOLERANCE);
}

#[test]
fn a_normalization_refuses_a_weight_of_another_width() {
    let x = tensor(&sequence(8, 1), (2, 4), DType::F32);
    let weight = tensor(&sequence(3, 2), 3, DType::F32);

    for refused in [
        functional::rms_norm(&x, &weight, 1e-6, false, DType::F32),
        functional::layer_norm(&x, &weight, None, 1e-6),
        functional::gated_rms_norm(&x, &x, &weight, 1e-6, false),
    ] {
        assert!(matches!(refused, Err(nn::Error::Shape { .. })));
    }
}

// ---- Activations ------------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn act_and_mul_multiplies_the_activation_of_the_gate_by_the_value() {
    let (gate, value) = (sequence(2 * 5, 14), sequence(2 * 5, 15));
    let gate_tensor = tensor(&gate, (2, 5), DType::F32);
    let value_tensor = tensor(&value, (2, 5), DType::F32);

    let reference = |activation: fn(f32) -> f32| -> Vec<f32> {
        gate.iter()
            .zip(&value)
            .map(|(gate, value)| activation(*gate) * value)
            .collect()
    };
    for (activation, formula) in [
        (Activation::Silu, (|x| x * sigmoid(x)) as fn(f32) -> f32),
        (Activation::Sigmoid, sigmoid),
        (Activation::GeluTanh, |x| {
            let inner = (2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }),
    ] {
        let product =
            functional::act_and_mul(&gate_tensor, &value_tensor, activation, false).unwrap();
        assert_close(&values(&product), &reference(formula), F32_TOLERANCE);
    }
}

/// The stepwise form rounds the activation into the element type before the product, and the
/// fused form rounds once: a gate whose SiLU is not a BF16 value tells them apart.
#[test]
fn the_stepwise_product_rounds_the_activation_first() {
    let gate = tensor(&[1.0, 0.75, -0.5, 2.0], (1, 4), DType::BF16);
    let value = tensor(&[3.0, 1.25, 7.0, 0.375], (1, 4), DType::BF16);
    let bf16 = |values: &[f32]| values_of_bf16(values);

    let gates = values(&gate);
    let silu: Vec<f32> = gates.iter().map(|gate| gate * sigmoid(*gate)).collect();
    let operands = values(&value);

    let fused: Vec<f32> = silu.iter().zip(&operands).map(|(a, v)| a * v).collect();
    let stepwise: Vec<f32> = bf16(&silu)
        .iter()
        .zip(&operands)
        .map(|(a, v)| a * v)
        .collect();
    assert_ne!(
        bf16(&fused),
        bf16(&stepwise),
        "the fixture tells the two apart"
    );

    let product = |stepwise| {
        values(&functional::act_and_mul(&gate, &value, Activation::Silu, stepwise).unwrap())
    };
    assert_eq!(product(false), bf16(&fused));
    assert_eq!(product(true), bf16(&stepwise));
}

/// `values` rounded into BF16 and widened back.
fn values_of_bf16(values: &[f32]) -> Vec<f32> {
    self::values(&tensor(values, values.len(), DType::BF16))
}

#[test]
fn swiglu_multiplies_value_and_gate_before_the_sigmoid_s_denominator() {
    let (value, gate) = (sequence(2 * 4, 16), sequence(2 * 4, 17));
    let (value_bias, gate_bias) = (sequence(4, 18), sequence(4, 19));
    let reference = |biased: bool| -> Vec<f32> {
        value
            .iter()
            .zip(&gate)
            .enumerate()
            .map(|(index, (value, gate))| {
                let (value, gate) = if biased {
                    (value + value_bias[index % 4], gate + gate_bias[index % 4])
                } else {
                    (*value, *gate)
                };
                (value * gate) / (1.0 + (-gate).exp())
            })
            .collect()
    };

    let value_tensor = tensor(&value, (2, 4), DType::F32);
    let gate_tensor = tensor(&gate, (2, 4), DType::F32);
    let plain = functional::swiglu(&value_tensor, &gate_tensor, None, None).unwrap();
    assert_close(&values(&plain), &reference(false), F32_TOLERANCE);

    let biased = functional::swiglu(
        &value_tensor,
        &gate_tensor,
        Some(&tensor(&value_bias, 4, DType::F32)),
        Some(&tensor(&gate_bias, 4, DType::F32)),
    )
    .unwrap();
    assert_close(&values(&biased), &reference(true), F32_TOLERANCE);

    // The packed form reads the value half and then the gate half of each row, with one packed
    // bias over both.
    let packed: Vec<f32> = rows_of(&value, 4)
        .iter()
        .zip(rows_of(&gate, 4))
        .flat_map(|(value, gate)| [*value, gate].concat())
        .collect();
    let packed_bias = [value_bias.clone(), gate_bias.clone()].concat();
    let from_packed = functional::value_first_swiglu(
        &tensor(&packed, (2, 8), DType::F32),
        Some(&tensor(&packed_bias, 8, DType::F32)),
    )
    .unwrap();
    assert_eq!(from_packed.dims(), [2, 4]);
    assert_close(&values(&from_packed), &reference(true), F32_TOLERANCE);
}

#[test]
fn snake_adds_the_squared_sine_over_each_channel_s_frequency() {
    let (x, alpha) = (sequence(2 * 3 * 4, 20), [0.5_f32, 1.25, 2.0]);
    let activated = functional::snake(
        &tensor(&x, (2, 3, 4), DType::F32),
        &tensor(&alpha, 3, DType::F32),
    )
    .unwrap();

    let expected: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(index, x)| {
            let alpha = alpha[(index / 4) % 3];
            x + (alpha * x).sin().powi(2) / (alpha + 1e-9)
        })
        .collect();
    assert_close(&values(&activated), &expected, F32_TOLERANCE);
}

#[test]
fn the_plain_activations_are_the_functions_their_names_state() {
    let x = sequence(6, 21);
    let input = tensor(&x, 6, DType::F32);

    let silu: Vec<f32> = x.iter().map(|x| x * sigmoid(*x)).collect();
    assert_close(
        &values(&functional::silu(&input).unwrap()),
        &silu,
        F32_TOLERANCE,
    );

    // The exact GELU through the error function of the standard library's complement.
    let tanh: Vec<f32> = x
        .iter()
        .map(|x| {
            let inner = (2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        })
        .collect();
    assert_close(
        &values(&functional::gelu_tanh(&input).unwrap()),
        &tanh,
        F32_TOLERANCE,
    );

    // The two GELUs are different functions, close to each other.
    let exact = values(&functional::gelu(&input).unwrap());
    assert_ne!(exact, values(&functional::gelu_tanh(&input).unwrap()));
    assert_close(&exact, &tanh, 1e-3);
}

// ---- Projections ------------------------------------------------------------------------------

/// `x [m, k] · weightᵀ` for a weight stored `[n, k]`, plus `bias`.
fn project(x: &[f32], weight: &[f32], bias: Option<&[f32]>, k: usize) -> Vec<f32> {
    let n = weight.len() / k;

    rows_of(x, k)
        .iter()
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

#[test]
fn linear_multiplies_the_rows_by_the_weight_in_either_order() {
    let (x, weight, bias) = (sequence(3 * 4, 22), sequence(5 * 4, 23), sequence(5, 24));
    let input = tensor(&x, (3, 4), DType::F32);

    let stored = tensor(&weight, (5, 4), DType::F32);
    let out = functional::linear(
        &input,
        &stored,
        Some(&tensor(&bias, 5, DType::F32)),
        WeightOrder::NxK,
    )
    .unwrap();
    assert_eq!(out.dims(), [3, 5]);
    assert_close(
        &values(&out),
        &project(&x, &weight, Some(&bias), 4),
        F32_TOLERANCE,
    );

    // The same matrix with the contraction axis first.
    let transposed = stored.t().unwrap().contiguous().unwrap();
    let out = functional::linear(&input, &transposed, None, WeightOrder::KxN).unwrap();
    assert_close(&values(&out), &project(&x, &weight, None, 4), F32_TOLERANCE);

    // Leading axes count rows.
    let batched = functional::linear(
        &tensor(&x, (1, 3, 4), DType::F32),
        &stored,
        None,
        WeightOrder::NxK,
    )
    .unwrap();
    assert_eq!(batched.dims(), [1, 3, 5]);

    let refused = functional::linear(&input, &stored, None, WeightOrder::KxN);
    assert!(matches!(refused, Err(nn::Error::Shape { .. })));
}

/// The nearest E4M3 value of `value`, ties to the even mantissa, saturating at ±448: every
/// E4M3 value is enumerated from the format's definition and the nearest is taken.
fn e4m3(value: f32) -> f32 {
    let mut grid: Vec<(f32, u32)> = Vec::new();
    for exponent in 0..16_i32 {
        for mantissa in 0..8_u32 {
            if exponent == 15 && mantissa == 7 {
                continue;
            }
            let magnitude = if exponent == 0 {
                mantissa as f32 / 8.0 * 2f32.powi(-6)
            } else {
                (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exponent - 7)
            };
            grid.push((magnitude, mantissa));
        }
    }

    let magnitude = value.abs().min(448.0);
    let nearest = grid
        .iter()
        .copied()
        .min_by(|left, right| {
            let (a, b) = ((left.0 - magnitude).abs(), (right.0 - magnitude).abs());
            // A tie goes to the even mantissa.
            a.partial_cmp(&b)
                .unwrap()
                .then((left.1 % 2).cmp(&(right.1 % 2)))
        })
        .unwrap();
    nearest.0.copysign(value)
}

#[test]
fn linear_fp8_encodes_the_activation_by_its_static_scale() {
    let (x, bias) = (sequence(2 * 4, 25), sequence(3, 26));
    // E4M3 values, so the weight is exact in its own type.
    let weight = [
        0.5_f32, -1.25, 2.0, 0.375, 1.0, 1.5, -0.75, 3.0, -2.5, 0.25, 0.875, -1.0,
    ];
    let (input_scale, weight_scale) = (0.0625_f32, 0.5_f32);
    let alpha = input_scale * weight_scale;

    let out = functional::linear_fp8(
        &tensor(&x, (2, 4), DType::BF16),
        &tensor(&weight, (3, 4), DType::F8E4M3),
        Some(&tensor(&bias, 3, DType::BF16)),
        &tensor(&[1.0 / input_scale], 1, DType::F32),
        &tensor(&[alpha], 1, DType::F32),
    )
    .unwrap();
    assert_eq!((out.dims(), out.dtype()), (&[2, 3][..], DType::BF16));

    let encoded: Vec<f32> = values_of_bf16(&x)
        .iter()
        .map(|x| e4m3((x / input_scale).clamp(-448.0, 448.0)))
        .collect();
    let scaled: Vec<f32> = project(&encoded, &weight, None, 4)
        .iter()
        .zip(values_of_bf16(&bias).iter().cycle())
        .map(|(product, bias)| product * alpha + bias)
        .collect();
    assert_eq!(values(&out), values_of_bf16(&scaled));

    // An activation beyond the format's range saturates at 448 times its scale.
    let saturated = functional::linear_fp8(
        &tensor(&[1000.0, 0.0, 0.0, 0.0], (1, 4), DType::BF16),
        &tensor(&[1.0, 0.0, 0.0, 0.0], (1, 4), DType::F8E4M3),
        None,
        &tensor(&[1.0], 1, DType::F32),
        &tensor(&[1.0], 1, DType::F32),
    )
    .unwrap();
    assert_eq!(values(&saturated), [448.0]);
}

/// The magnitudes an E2M1 code states.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// The nearest E2M1 value of `value`, ties to the even code, saturating at ±6.
fn e2m1(value: f32) -> f32 {
    let magnitude = value.abs().min(6.0);
    let code = (0..8)
        .min_by(|left, right| {
            let (a, b) = (
                (E2M1[*left] - magnitude).abs(),
                (E2M1[*right] - magnitude).abs(),
            );
            a.partial_cmp(&b)
                .unwrap()
                .then((left % 2).cmp(&(right % 2)))
        })
        .unwrap();
    E2M1[code].copysign(value)
}

/// The position of the block scale of row `row`, scale column `column` in the swizzled 128x4
/// table of `padded_columns` scale columns.
fn swizzled(row: usize, column: usize, padded_columns: usize) -> usize {
    (column % 4)
        + (column / 4) * 512
        + (row % 32) * 16
        + ((row % 128) / 32) * 4
        + (row / 128) * (128 * padded_columns)
}

#[test]
fn linear_nvfp4_multiplies_the_encoded_activation_by_the_decoded_weight() {
    let (m, n, k) = (2_usize, 3_usize, 32_usize);
    let blocks = k / 16;

    // The weight: codes of every value and sign, and E4M3 block scales, read as the checkpoint
    // packs them (the even element in the low nibble) and as the loader lays the scales out.
    let codes: Vec<u8> = (0..n * k)
        .map(|index| ((index * 7 + 3) % 16) as u8)
        .collect();
    let packed: Vec<u8> = codes.chunks(2).map(|pair| pair[0] | pair[1] << 4).collect();
    let block_scales = [1.0_f32, 0.5, 2.0, 0.25, 1.5, 0.75];
    let mut table = vec![0.0_f32; 128 * 4];
    for row in 0..n {
        for column in 0..blocks {
            table[swizzled(row, column, 4)] = block_scales[row * blocks + column];
        }
    }
    let weight: Vec<f32> = codes
        .iter()
        .enumerate()
        .map(|(index, code)| {
            let sign = if code & 8 != 0 { -1.0 } else { 1.0 };
            sign * E2M1[usize::from(code & 7)] * block_scales[index / 16]
        })
        .collect();

    // The activation, encoded as the kernel states it: each block's scale is its largest
    // magnitude over six under the global scale, in E4M3, and each element the nearest E2M1
    // code of the element over that scale.
    let x: Vec<f32> = values_of_bf16(
        &sequence(m * k, 27)
            .iter()
            .map(|x| x * 3.0)
            .collect::<Vec<_>>(),
    );
    let (input_scale, alpha) = (0.25_f32, 0.125_f32);
    let encoded: Vec<f32> = x
        .chunks(16)
        .flat_map(|block| {
            let amax = block.iter().fold(0.0_f32, |amax, x| amax.max(x.abs()));
            let scale = e4m3(amax / 6.0 * input_scale);
            let relative = scale / input_scale;

            block
                .iter()
                .map(|x| {
                    if relative == 0.0 {
                        0.0
                    } else {
                        e2m1(x / relative) * scale
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let expected: Vec<f32> = project(&encoded, &weight, None, k)
        .iter()
        .map(|product| product * alpha)
        .collect();

    let project = |bias: Option<&Tensor>| {
        functional::linear_nvfp4(
            &tensor(&x, (m, k), DType::BF16),
            &Tensor::from_slice(&packed, (n, k / 2), &Device::Cpu).unwrap(),
            &tensor(&table, (128, 4), DType::F8E4M3),
            bias,
            &tensor(&[input_scale], 1, DType::F32),
            &tensor(&[alpha], 1, DType::F32),
        )
        .unwrap()
    };
    let out = project(None);
    assert_eq!((out.dims(), out.dtype()), (&[m, n][..], DType::BF16));
    assert_close(&values(&out), &values_of_bf16(&expected), 1.0 / 128.0);

    // A bias joins the product after it has rounded, and the sum rounds again: exactly the
    // BF16 sum of the unbiased result and the bias.
    let bias = [0.75_f32, -2.5, 16.0];
    let biased = project(Some(&tensor(&bias, n, DType::BF16)));
    let sums: Vec<f32> = values(&out)
        .chunks(n)
        .flat_map(|row| row.iter().zip(&bias).map(|(product, bias)| product + bias))
        .collect();
    assert_eq!(values(&biased), values_of_bf16(&sums));
}

// ---- Embedding and logits ---------------------------------------------------------------------

#[test]
fn embedding_gathers_the_shard_s_rows_and_zeros_the_tokens_outside_it() {
    // Rows 10..14 of a vocabulary: row `r` holds `r` and `-r`.
    let table: Vec<f32> = (10..14)
        .flat_map(|row| [row as f32, -(row as f32)])
        .collect();
    let ids = Tensor::new(&[12_u32, 3, 13, 14, 10, 4_000_000_000], &Device::Cpu).unwrap();

    let rows = functional::embedding(&ids, &tensor(&table, (4, 2), DType::F32), 10).unwrap();
    assert_eq!(rows.dims(), [6, 2]);
    assert_eq!(
        values(&rows),
        [
            12.0, -12.0, 0.0, 0.0, 13.0, -13.0, 0.0, 0.0, 10.0, -10.0, 0.0, 0.0
        ]
    );

    // A rank past the vocabulary's end holds no row.
    let empty = Tensor::zeros((0, 2), DType::F32, &Device::Cpu).unwrap();
    let rows = functional::embedding(&ids, &empty, 16).unwrap();
    assert_eq!(values(&rows), [0.0; 12]);
}

#[test]
fn vocab_unpad_reads_each_column_from_the_shard_that_holds_it() {
    // Two ranks of a vocabulary of five: each shard holds three ids in four columns, and the
    // gather lays rank 0's `[rows, 4]` block before rank 1's.
    let (rows, width, ranks) = (2_usize, 4_usize, 2_usize);
    let gathered: Vec<f32> = (0..ranks * rows * width)
        .map(|index| {
            let (rank, row, column) = (
                index / (rows * width),
                (index / width) % rows,
                index % width,
            );
            (100 * rank + 10 * row + column) as f32
        })
        .collect();

    let logits =
        functional::vocab_unpad(&tensor(&gathered, (rows, width * ranks), DType::BF16), 5, 2)
            .unwrap();
    assert_eq!((logits.dims(), logits.dtype()), (&[2, 5][..], DType::F32));
    assert_eq!(
        values(&logits),
        [0.0, 1.0, 2.0, 100.0, 101.0, 10.0, 11.0, 12.0, 110.0, 111.0]
    );

    // Two shards of two columns cannot hold a vocabulary of five.
    let narrow = tensor(&gathered[..8], (2, 4), DType::BF16);
    assert!(functional::vocab_unpad(&narrow, 5, 2).is_err());
}

// ---- Kernels stated under a meta context ------------------------------------------------------

#[test]
fn the_normalizations_state_their_kernels() {
    let (output, launches) = recorded(|| {
        let x = meta((5, 8), DType::BF16);
        let weight = meta(8, DType::BF16);

        let plain = functional::rms_norm(&x, &weight, 1e-6, false, DType::BF16)?;
        let centred = functional::rms_norm(&x, &weight, 1e-6, true, DType::BF16)?;
        let stream = meta((5, 8), DType::F32);
        let narrowed =
            functional::rms_norm(&stream, &meta(8, DType::F32), 1e-6, false, DType::BF16)?;
        let (normalized, sum) = functional::add_rms_norm(&x, &plain, &weight, 1e-6, false)?;
        let gated = functional::gated_rms_norm(&x, &centred, &weight, 1e-6, false)?;
        let layer = functional::layer_norm(&x, &weight, Some(&weight), 1e-5)?;

        Ok([plain, centred, narrowed, normalized, sum, gated, layer]
            .map(|tensor| (tensor.dims().to_vec(), tensor.dtype())))
    });

    assert_eq!(output.unwrap(), [(); 7].map(|()| (vec![5, 8], DType::BF16)));
    assert_eq!(
        launches,
        [
            ("rms_norm", "dtype_bf16_columns_8_tile_8_pdl_false"),
            (
                "rms_norm",
                "dtype_bf16_columns_8_tile_8_pdl_false_centred_true"
            ),
            (
                "weighted_rms_norm",
                "dtype_f32_columns_8_tile_8_absmax_false_activation_bf16"
            ),
            ("add_rms_norm", "dtype_bf16_columns_8_tile_8_pdl_false"),
            ("gated_rms_norm", "dtype_bf16_columns_8_tile_8"),
            // The row width is not in a layer normalization's key: its tile is.
            ("layer_norm", "dtype_bf16_tile_8_bias_true_pdl_false"),
        ]
        .map(|(operator, key)| (operator, key.to_owned()))
    );
}

#[test]
fn the_gated_activations_state_their_kernels_and_their_row_strides() {
    let (output, launches) = recorded(|| {
        let gate = meta((3, 16), DType::BF16);
        let value = meta((3, 16), DType::BF16);
        let packed = meta((3, 32), DType::BF16);

        let fused = functional::act_and_mul(&gate, &value, Activation::Silu, false)?;
        // The halves of one packed projection stride over the packed row.
        let halves = functional::act_and_mul(
            &packed.narrow(1, 16, 16)?,
            &packed.narrow(1, 0, 16)?,
            Activation::GeluTanh,
            true,
        )?;
        let swiglu = functional::swiglu(&value, &gate, None, None)?;
        let value_first = functional::value_first_swiglu(&packed, None)?;
        let snake = functional::snake(&meta((2, 4, 6), DType::F32), &meta(4, DType::F32))?;

        Ok([fused, halves, swiglu, value_first]
            .map(|tensor| tensor.dims().to_vec())
            .into_iter()
            .chain([snake.dims().to_vec()])
            .collect::<Vec<_>>())
    });

    assert_eq!(
        output.unwrap(),
        [
            vec![3, 16],
            vec![3, 16],
            vec![3, 16],
            vec![3, 16],
            vec![2, 4, 6]
        ]
    );
    assert_eq!(
        launches,
        [
            (
                "act_and_mul",
                "dtype_bf16_width_16_block_4096_activation_silu_gate_stride_16_value_stride_16_\
                 wide_false_pdl_false_stepwise_false"
            ),
            (
                "act_and_mul",
                "dtype_bf16_width_16_block_4096_activation_gelu_tanh_gate_stride_32_\
                 value_stride_32_wide_false_pdl_false_stepwise_true"
            ),
            (
                "swiglu",
                "dtype_bf16_width_16_block_16_value_stride_16_gate_stride_16_value_bias_false_\
                 gate_bias_false_absmax_false"
            ),
            (
                "value_first_swiglu",
                "dtype_bf16_width_16_block_4096_input_stride_32_bias_false_absmax_false"
            ),
            ("snake", "dtype_f32_channels_4_block_1024_pdl_false"),
        ]
        .map(|(operator, key)| (operator, key.to_owned()))
    );
}

/// The 64-bit form follows the packed input's span, which runs twice as far as the output's:
/// with 14336 channels the last row of 74899 is past `2**31` elements of input while the output
/// is at half of that.
#[test]
fn a_packed_projection_goes_wide_on_its_input_span() {
    let width = 14_336_usize;
    let wide = |rows: usize| {
        let (output, launches) = recorded(|| {
            let packed = meta((rows, 2 * width), DType::BF16);
            functional::act_and_mul(
                &packed.narrow(1, width, width)?,
                &packed.narrow(1, 0, width)?,
                Activation::Silu,
                true,
            )
        });
        output.unwrap();
        launches[0].1.contains("_wide_true_")
    };

    assert!(!wide(74_898), "74898 rows stay inside 32 bits");
    assert!(wide(74_899), "74899 rows do not");
}

#[test]
fn the_plain_activations_are_elementwise_primitives() {
    let (output, launches) = recorded(|| {
        let x = meta((3, 8), DType::BF16);
        functional::silu(&x)?;
        functional::gelu(&x)?;
        functional::gelu_tanh(&x)
    });

    output.unwrap();
    assert_eq!(operators(&launches), ["elementwise_unary"; 3]);
    for (launch, function) in launches.iter().zip(["silu", "gelu_erf", "gelu"]) {
        assert!(
            launch
                .1
                .starts_with(&format!("function_{function}_dtype_bf16")),
            "{launch:?}"
        );
    }
}

#[test]
fn the_dense_and_the_fp8_projections_state_their_kernels() {
    let (output, launches) = recorded(|| {
        let x = meta((7, 64), DType::BF16);
        let dense = functional::linear(
            &x,
            &meta((8, 64), DType::BF16),
            Some(&meta(8, DType::BF16)),
            WeightOrder::NxK,
        )?;
        let fp8 = functional::linear_fp8(
            &x,
            &meta((8, 64), DType::F8E4M3),
            None,
            &meta(1, DType::F32),
            &meta(1, DType::F32),
        )?;

        Ok([dense, fp8].map(|tensor| tensor.dims().to_vec()))
    });

    assert_eq!(output.unwrap(), [vec![7, 8], vec![7, 8]]);
    assert_eq!(
        launches,
        [
            (
                "gemm",
                "input_bf16_weight_bf16_accumulate_f32_output_bf16_order_nxk_epilogue_bias"
            ),
            ("fp8_quantize", "dtype_bf16_block_1024"),
            (
                "fp8_gemm",
                "dtype_bf16_block_m_128_block_n_128_block_k_128_bias_false"
            ),
        ]
        .map(|(operator, key)| (operator, key.to_owned()))
    );
}

/// The launch table of one NVFP4 projection of `rows` activation rows of `input` over an
/// `[8, 64]` weight whose block scales are `scale` elements.
fn nvfp4_launches(
    rows: usize,
    input: DType,
    scale: DType,
) -> (nn::Result<Vec<usize>>, Vec<(&'static str, String)>) {
    recorded(|| {
        let out = functional::linear_nvfp4(
            &meta((rows, 64), input),
            &meta((8, 32), DType::U8),
            &meta((128, 4), scale),
            None,
            &meta(1, DType::F32),
            &meta(1, DType::F32),
        )?;
        Ok(out.dims().to_vec())
    })
}

/// The keys a launch table holds for `operator`.
fn keys_of<'a>(launches: &'a [(&'static str, String)], operator: &str) -> Vec<&'a str> {
    launches
        .iter()
        .filter(|(name, _)| *name == operator)
        .map(|(_, key)| key.as_str())
        .collect()
}

#[test]
fn a_packed_product_is_keyed_per_row_class_and_a_dense_one_is_not() {
    let launched = |rows: usize| {
        let (output, launches) = nvfp4_launches(rows, DType::BF16, DType::F8E4M3);
        assert_eq!(
            output.unwrap(),
            [rows, 8],
            "the result is the request's rows"
        );
        launches
    };
    let (one, eight) = (launched(1), launched(8));
    assert_eq!(operators(&one), ["nvfp4_quantize", "nvfp4_gemm"]);

    // One and eight rows are the 128-row class: the same kernel, not two.
    assert_eq!(keys_of(&one, "nvfp4_gemm"), keys_of(&eight, "nvfp4_gemm"));
    assert_ne!(
        keys_of(&launched(128), "nvfp4_gemm"),
        keys_of(&launched(129), "nvfp4_gemm")
    );
    // The activation encoding is one kernel whatever the row count.
    assert_eq!(
        keys_of(&one, "nvfp4_quantize"),
        keys_of(&launched(129), "nvfp4_quantize")
    );
    assert_eq!(
        keys_of(&one, "nvfp4_quantize"),
        ["input_bf16_columns_64_layout_swizzled_128x4_block_64_pdl_false"]
    );

    // Rows above the largest class are computed block by block: whole blocks of the largest
    // class, and the rest at the smallest class that holds them.
    let blocks = launched(2 * 16384 + 200);
    assert_eq!(
        operators(&blocks),
        ["nvfp4_quantize", "nvfp4_gemm"].repeat(3)
    );
    let classes: Vec<&str> = keys_of(&blocks, "nvfp4_gemm")
        .iter()
        .map(|key| key.split("_m_").nth(1).unwrap())
        .collect();
    assert_eq!(
        classes,
        ["16384_n_8_k_64", "16384_n_8_k_64", "256_n_8_k_64"]
    );

    // A dense projection's key takes no row count at all.
    let dense = |rows: usize| {
        recorded(|| {
            functional::linear(
                &meta((rows, 64), DType::BF16),
                &meta((8, 64), DType::BF16),
                None,
                WeightOrder::NxK,
            )
        })
        .1
    };
    assert_eq!(dense(1), dense(129));
}

/// The encoder is keyed by the activation's own type, a type NVFP4 does not encode is refused
/// at the call, and block scales declared as bytes are refused where the key is built.
#[test]
fn a_packed_projection_encodes_the_activation_it_is_given() {
    let (output, launches) = nvfp4_launches(4, DType::F16, DType::F8E4M3);
    output.unwrap();
    assert!(
        keys_of(&launches, "nvfp4_quantize")
            .iter()
            .all(|key| key.contains("f16") && !key.contains("bf16")),
        "{launches:?}"
    );

    let (refused, _) = nvfp4_launches(4, DType::F32, DType::F8E4M3);
    assert!(refused.unwrap_err().to_string().contains("f32"));

    let (refused, _) = nvfp4_launches(16, DType::BF16, DType::U8);
    let text = refused.unwrap_err().to_string();
    assert!(text.contains("f8e4m3 block scale"), "{text}");
}

#[test]
fn the_embedding_and_the_unpad_state_their_kernels() {
    let (output, launches) = recorded(|| {
        let rows = functional::embedding(&meta(5, DType::U32), &meta((16, 8), DType::BF16), 32)?;
        let logits = functional::vocab_unpad(&meta((3, 32), DType::BF16), 32, 2)?;

        Ok((rows.dims().to_vec(), logits.dims().to_vec(), logits.dtype()))
    });

    assert_eq!(output.unwrap(), (vec![5, 8], vec![3, 32], DType::F32));
    assert_eq!(
        launches,
        [
            (
                "embedding",
                "dtype_bf16_ids_u32_hidden_8_first_32_rows_16_block_8_pdl_false"
            ),
            (
                "vocab_unpad",
                "dtype_bf16_output_f32_vocabulary_32_padded_32_block_1024_pdl_false"
            ),
        ]
        .map(|(operator, key)| (operator, key.to_owned()))
    );
}
