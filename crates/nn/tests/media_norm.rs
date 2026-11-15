//! The normalizations fused with a residual update or a modulation, the per-frame group
//! normalization, and the blocks built from them: their arithmetic on CPU tensors against each
//! formula written out in loops, and the kernels they state under a meta context.
//!
//! Nothing here touches a GPU.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

mod support;

use nn::collective::Split;
use nn::functional::{self, Activation, Padding, Rounding};
use nn::modules::{
    AdaLnBlock, AdaLnBlockConfig, AdaLnOut, AdaLnOutConfig, FrameNorm, FrameNormConfig, GateUp,
    MlpConfig, ResnetBlock3d, ResnetBlock3dConfig,
};
use nn::{Checkpoint, CpuWeights, DType, Device, MetaWeights, Tensor, WeightLoader};
use support::{
    assert_close, f32_bytes, meta, operators, recorded, sequence, shard, tensor, values,
};

/// The tolerance of an FP32 result against a reference accumulated in another order.
const F32_TOLERANCE: f32 = 2e-5;

/// The checkpoint at `directory` holding the FP32 tensors `tensors` (name, extents, values).
fn checkpoint(directory: &std::path::Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Checkpoint {
    let typed: Vec<(&str, DType, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, shape, values)| (*name, DType::F32, shape.clone(), f32_bytes(values)))
        .collect();
    shard(directory, "model.safetensors", &typed);

    Checkpoint::open(directory).unwrap()
}

/// `value` rounded into BF16, to nearest even, as the library's conversion rounds it.
fn bf16(value: f32) -> f32 {
    values(&tensor(&[value], 1, DType::BF16))[0]
}

/// A column of `U32` parameter rows.
fn parameter_rows(rows: &[u32]) -> Tensor {
    Tensor::from_slice(rows, rows.len(), &Device::Cpu).unwrap()
}

// ---- Scaled residual --------------------------------------------------------------------------

/// The FP32 sum `hidden + (update + bias) · scale`, row by row.
fn scaled_sum(hidden: &[f32], update: &[f32], bias: &[f32], scale: &[f32]) -> Vec<f32> {
    let width = bias.len();

    hidden
        .iter()
        .zip(update)
        .enumerate()
        .map(|(index, (hidden, update))| {
            let column = index % width;
            hidden + (update + bias[column]) * scale[column % scale.len()]
        })
        .collect()
}

#[test]
fn a_scaled_residual_rms_norm_updates_the_stream_and_normalizes_its_sum() {
    let (hidden, update) = (sequence(3 * 8, 1), sequence(3 * 8, 2));
    let (bias, scale, weight) = (sequence(8, 3), sequence(8, 4), sequence(8, 5));
    let eps = 1e-6;

    let stream = tensor(&hidden, (3, 8), DType::F32);
    let (normalized, updated) = functional::scaled_residual_rms_norm_(
        stream.clone(),
        &tensor(&update, (3, 8), DType::F32),
        Some(&tensor(&bias, 8, DType::F32)),
        &tensor(&scale, 8, DType::F32),
        &tensor(&weight, 8, DType::F32),
        eps,
    )
    .unwrap();

    let sum = scaled_sum(&hidden, &update, &bias, &scale);
    let expected: Vec<f32> = sum
        .chunks(8)
        .flat_map(|row| {
            let mean_square = row.iter().map(|value| value * value).sum::<f32>() / 8.0;
            let scale = 1.0 / (mean_square + eps).sqrt();
            row.iter()
                .zip(&weight)
                .map(move |(value, weight)| value * scale * weight)
        })
        .collect();
    assert_close(&values(&normalized), &expected, F32_TOLERANCE);

    // The stream is written in place: the caller's tensor holds the sum.
    assert_close(&values(&updated), &sum, F32_TOLERANCE);
    assert_eq!(values(&stream), values(&updated));

    // One scale for the whole stream broadcasts, and a strided stream has no rows to write.
    let (_, updated) = functional::scaled_residual_rms_norm_(
        tensor(&hidden, (3, 8), DType::F32),
        &tensor(&update, (3, 8), DType::F32),
        None,
        &tensor(&[0.5], 1, DType::F32),
        &tensor(&weight, 8, DType::F32),
        eps,
    )
    .unwrap();
    assert_close(
        &values(&updated),
        &scaled_sum(&hidden, &update, &[0.0; 8], &[0.5]),
        F32_TOLERANCE,
    );

    let strided = tensor(&hidden, (8, 3), DType::F32).t().unwrap();
    let refused = functional::scaled_residual_rms_norm_(
        strided,
        &tensor(&update, (3, 8), DType::F32),
        None,
        &tensor(&scale, 8, DType::F32),
        &tensor(&weight, 8, DType::F32),
        eps,
    );
    assert!(refused.unwrap_err().to_string().contains("in place"));
}

/// The stream stores the sum rounded once, and the normalization reads the sum before that
/// rounding and writes the update's narrower type.
#[test]
fn a_scaled_residual_rms_norm_keeps_the_stream_s_type_and_writes_the_update_s() {
    let (hidden, update) = (sequence(2 * 4, 6), sequence(2 * 4, 7));
    let (scale, weight) = (sequence(4, 8), sequence(4, 9));

    let narrow_update = tensor(&update, (2, 4), DType::BF16);
    let (normalized, stream) = functional::scaled_residual_rms_norm_(
        tensor(&hidden, (2, 4), DType::F32),
        &narrow_update,
        None,
        &tensor(&scale, 4, DType::F32),
        &tensor(&weight, 4, DType::F32),
        1e-6,
    )
    .unwrap();
    assert_eq!(
        (stream.dtype(), normalized.dtype()),
        (DType::F32, DType::BF16)
    );

    let sum = scaled_sum(&hidden, &values(&narrow_update), &[0.0; 4], &scale);
    assert_close(&values(&stream), &sum, F32_TOLERANCE);
}

#[test]
fn a_scaled_residual_layer_norm_leaves_the_stream_as_it_was() {
    let (hidden, update) = (sequence(3 * 8, 10), sequence(3 * 8, 11));
    let (scale, weight, bias) = (sequence(8, 12), sequence(8, 13), sequence(8, 14));
    let eps = 1e-5;

    let stream = tensor(&hidden, (3, 8), DType::F32);
    let normalized = functional::scaled_residual_layer_norm(
        &stream,
        &tensor(&update, (3, 8), DType::F32),
        None,
        &tensor(&scale, 8, DType::F32),
        &tensor(&weight, 8, DType::F32),
        &tensor(&bias, 8, DType::F32),
        eps,
    )
    .unwrap();

    let expected: Vec<f32> = scaled_sum(&hidden, &update, &[0.0; 8], &scale)
        .chunks(8)
        .flat_map(|row| {
            let mean = row.iter().sum::<f32>() / 8.0;
            let variance = row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / 8.0;
            let scale = 1.0 / (variance + eps).sqrt();
            row.iter()
                .zip(weight.iter().zip(&bias))
                .map(move |(value, (weight, bias))| (value - mean) * scale * weight + bias)
                .collect::<Vec<f32>>()
        })
        .collect();
    assert_close(&values(&normalized), &expected, F32_TOLERANCE);
    assert_eq!(values(&stream), hidden);
}

// ---- Modulation -------------------------------------------------------------------------------

/// The modulated normalization of one BF16 row by its parameter row's scale and shift, every
/// step rounded into BF16 where the kernel rounds.
fn modulated_row(row: &[f32], weight: &[f32], scale: &[f32], shift: &[f32], eps: f32) -> Vec<f32> {
    let mean_square = row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
    let rsqrt = 1.0 / (mean_square + eps).sqrt();

    (0..row.len())
        .map(|column| {
            let normalized = bf16(row[column] * rsqrt * weight[column]);
            let factor = bf16(1.0 + scale[column]);
            bf16(bf16(normalized * factor) + shift[column])
        })
        .collect()
}

#[test]
fn a_modulated_rms_norm_rounds_each_step_into_the_activation_type() {
    const WIDTH: usize = 4;
    let narrow = |values: &[f32], shape: (usize, usize)| tensor(values, shape, DType::BF16);

    // A table of three parameter rows of two vectors each: shift, then scale.
    let table = narrow(&sequence(3 * 2 * WIDTH, 15), (3, 2 * WIDTH));
    let (shift, scale) = (
        table.narrow(1, 0, WIDTH).unwrap(),
        table.narrow(1, WIDTH, WIDTH).unwrap(),
    );
    let x = narrow(&sequence(4 * WIDTH, 16), (4, WIDTH));
    let weight = tensor(&sequence(WIDTH, 17), WIDTH, DType::BF16);
    let rows = [2_u32, 0, 2, 1];

    let out = functional::modulated_rms_norm(
        &x,
        &weight,
        &shift,
        &scale,
        &parameter_rows(&rows),
        1e-6,
        DType::BF16,
    )
    .unwrap();

    let (x, weight, table) = (values(&x), values(&weight), values(&table));
    let expected: Vec<f32> = rows
        .iter()
        .enumerate()
        .flat_map(|(row, parameter)| {
            let vectors = &table[*parameter as usize * 2 * WIDTH..][..2 * WIDTH];
            modulated_row(
                &x[row * WIDTH..][..WIDTH],
                &weight,
                &vectors[WIDTH..],
                &vectors[..WIDTH],
                1e-6,
            )
        })
        .collect();
    assert_close(&values(&out), &expected, 1.0 / 128.0);

    // The parameter rows are 32-bit unsigned indices, one per activation row.
    let signed = Tensor::from_slice(&[0_i64; 4], 4, &Device::Cpu).unwrap();
    let narrow_x = narrow(&x, (4, WIDTH));
    let narrow_weight = tensor(&weight, WIDTH, DType::BF16);
    for wrong in [signed, parameter_rows(&[0, 1])] {
        assert!(
            functional::modulated_rms_norm(
                &narrow_x,
                &narrow_weight,
                &shift,
                &scale,
                &wrong,
                1e-6,
                DType::BF16
            )
            .is_err()
        );
    }
}

#[test]
fn a_gated_modulated_rms_norm_stores_the_sum_over_its_update() {
    const WIDTH: usize = 4;
    let narrow = |values: &[f32], shape: (usize, usize)| tensor(values, shape, DType::BF16);

    let table = narrow(&sequence(2 * 3 * WIDTH, 18), (2, 3 * WIDTH));
    let vector = |index: usize| table.narrow(1, index * WIDTH, WIDTH).unwrap();
    let (hidden, update) = (
        narrow(&sequence(3 * WIDTH, 19), (3, WIDTH)),
        narrow(&sequence(3 * WIDTH, 20), (3, WIDTH)),
    );
    let weight = tensor(&sequence(WIDTH, 21), WIDTH, DType::BF16);
    let rows = [1_u32, 0, 1];
    let update_values = values(&update);

    let (normalized, sum) = functional::modulated_rms_norm_(
        update.clone(),
        &hidden,
        &vector(2),
        &weight,
        &vector(0),
        &vector(1),
        &parameter_rows(&rows),
        1e-6,
        DType::BF16,
    )
    .unwrap();

    // The gated update rounds, then the sum rounds, and the normalization reads that sum.
    let (hidden, weight, table) = (values(&hidden), values(&weight), values(&table));
    let mut expected_sum = Vec::new();
    let mut expected = Vec::new();
    for (row, parameter) in rows.iter().enumerate() {
        let vectors = &table[*parameter as usize * 3 * WIDTH..][..3 * WIDTH];
        let residual: Vec<f32> = (0..WIDTH)
            .map(|column| {
                let gated = bf16(vectors[2 * WIDTH + column] * update_values[row * WIDTH + column]);
                bf16(hidden[row * WIDTH + column] + gated)
            })
            .collect();

        expected.extend(modulated_row(
            &residual,
            &weight,
            &vectors[WIDTH..2 * WIDTH],
            &vectors[..WIDTH],
            1e-6,
        ));
        expected_sum.extend(residual);
    }
    assert_close(&values(&sum), &expected_sum, 1.0 / 128.0);
    assert_close(&values(&normalized), &expected, 1.0 / 128.0);

    // The update is spent: the kernel stores the sum over it.
    assert_eq!(values(&update), values(&sum));
}

#[test]
fn the_fused_normalizations_state_their_kernels() {
    let (stream, update) = (meta((3, 16), DType::BF16), meta((3, 16), DType::BF16));
    let vector = || meta(16, DType::BF16);
    let (weight, scale, bias) = (vector(), vector(), vector());

    let (output, launches) = recorded(|| {
        let (normalized, stream) =
            functional::scaled_residual_rms_norm_(stream, &update, None, &scale, &weight, 1e-6)?;
        functional::scaled_residual_layer_norm(
            &stream,
            &normalized,
            Some(&bias),
            &scale,
            &weight,
            &bias,
            1e-6,
        )
    });
    assert_eq!(output.unwrap().dims(), [3, 16]);
    assert_eq!(
        launches,
        [
            (
                "scaled_residual_rms_norm",
                "dtype_bf16_columns_16_tile_16_update_bias_false_absmax_false"
            ),
            (
                "scaled_residual_layer_norm",
                "dtype_bf16_columns_16_tile_16_update_bias_true_absmax_false"
            ),
        ]
        .map(|(operator, key)| (operator, key.to_owned()))
    );

    // The vectors of a modulation are column windows of one table, so the kernel is told the
    // table's row stride; the gated form also stores the sum it normalized.
    let table = meta((4, 3 * 16), DType::BF16);
    let rows = meta(3, DType::U32);
    let (output, launches) = recorded(|| {
        let window = |index: usize| table.narrow(1, index * 16, 16);
        let hidden = meta((3, 16), DType::BF16);

        let normalized = functional::modulated_rms_norm(
            &hidden,
            &weight,
            &window(0)?,
            &window(1)?,
            &rows,
            1e-6,
            DType::BF16,
        )?;
        functional::modulated_rms_norm_(
            normalized,
            &hidden,
            &window(2)?,
            &weight,
            &window(0)?,
            &window(1)?,
            &rows,
            1e-6,
            DType::BF16,
        )
    });
    let (normalized, sum) = output.unwrap();
    assert_eq!(
        (normalized.dims(), sum.dims()),
        (&[3, 16][..], &[3, 16][..])
    );

    let key = |gate_stride: u32, fused: bool| {
        format!(
            "dtype_bf16_output_bf16_indices_u32_width_16_block_16_epsilon_0x358637bd_\
             gate_stride_{gate_stride}_shift_stride_48_scale_stride_48_update_{fused}_\
             retain_{fused}"
        )
    };
    assert_eq!(
        launches,
        [
            ("modulated_rms_norm", key(0, false)),
            ("modulated_rms_norm", key(48, true)),
        ]
    );
}

// ---- Frames -----------------------------------------------------------------------------------

/// `silu(norm(x) · weight + bias)` of a `[c, d, h, w]` clip, each frame's groups normalized by
/// that frame's statistics, in loops.
fn frame_norm_reference(
    x: &[f32],
    [c, d, h, w]: [usize; 4],
    affine: Option<(&[f32], &[f32])>,
    groups: usize,
    eps: f32,
) -> Vec<f32> {
    let (group_channels, pixels) = (c / groups, h * w);
    let mut out = vec![0.0; x.len()];

    for frame in 0..d {
        for group in 0..groups {
            let channels = group * group_channels..(group + 1) * group_channels;
            let elements: Vec<usize> = channels
                .flat_map(|channel| {
                    (0..pixels).map(move |pixel| (channel * d + frame) * pixels + pixel)
                })
                .collect();

            let count = elements.len() as f32;
            let mean = elements.iter().map(|index| x[*index]).sum::<f32>() / count;
            let variance = elements
                .iter()
                .map(|index| (x[*index] - mean).powi(2))
                .sum::<f32>()
                / count;
            let rstd = 1.0 / (variance + eps).sqrt();

            for index in elements {
                let channel = index / (d * pixels);
                let (weight, bias) = affine.map_or((1.0, 0.0), |(weight, bias)| {
                    (weight[channel], bias[channel])
                });
                let normalized = (x[index] - mean) * rstd * weight + bias;
                out[index] = normalized / (1.0 + (-normalized).exp());
            }
        }
    }
    out
}

#[test]
fn a_frame_norm_normalizes_each_frame_s_groups_activates_and_pads() {
    let dims = [8, 3, 4, 5];
    let x = sequence(dims.iter().product(), 22);
    let (weight, bias) = (sequence(8, 23), sequence(8, 24));
    let clip = tensor(&x, (1, 8, 3, 4, 5), DType::F32);
    let (weight_tensor, bias_tensor) =
        (tensor(&weight, 8, DType::F32), tensor(&bias, 8, DType::F32));

    for affine in [true, false] {
        let parameters = affine.then_some((&weight_tensor, &bias_tensor));
        let padded = functional::frame_norm_pad(
            &clip,
            parameters,
            4,
            1e-6,
            [1, 1, 1, 0],
            2,
            Padding::Reflect,
        )
        .unwrap();
        assert_eq!(padded.dims(), [1, 8, 5, 6, 6]);

        // The padded frame is the causal pad of the normalized and activated clip.
        let reference = frame_norm_reference(
            &x,
            dims,
            affine.then_some((&weight[..], &bias[..])),
            4,
            1e-6,
        );
        let expected = functional::causal_pad(
            &tensor(&reference, (1, 8, 3, 4, 5), DType::F32),
            [1, 1, 1, 0],
            2,
            Padding::Reflect,
        )
        .unwrap();
        assert_close(&values(&padded), &values(&expected), 1e-4);
    }

    // A frame of one pixel normalizes over its group's channels alone.
    let dots = sequence(8 * 3, 25);
    let padded = functional::frame_norm_pad(
        &tensor(&dots, (1, 8, 3, 1, 1), DType::F32),
        Some((&weight_tensor, &bias_tensor)),
        2,
        1e-6,
        [0; 4],
        0,
        Padding::Replicate,
    )
    .unwrap();
    assert_close(
        &values(&padded),
        &frame_norm_reference(&dots, [8, 3, 1, 1], Some((&weight, &bias)), 2, 1e-6),
        1e-4,
    );

    // The kernels normalize FP32 frames in groups that divide the channels.
    let refuse = |x: &Tensor, groups| {
        functional::frame_norm_pad(x, None, groups, 1e-6, [0; 4], 0, Padding::Reflect).is_err()
    };
    assert!(refuse(&clip, 3));
    assert!(refuse(&clip, 0));
    assert!(refuse(&tensor(&x, (1, 8, 3, 4, 5), DType::BF16), 4));
    assert!(refuse(&tensor(&x, (8, 3, 4, 5), DType::F32), 4));
}

/// The tensors of a residual block of `from` to 8 channels under `name`.
fn resnet_tensors(name: &str, from: usize, seed: u32) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut tensors = vec![
        (format!("{name}.norm1.weight"), vec![from]),
        (format!("{name}.norm1.bias"), vec![from]),
        (format!("{name}.conv1.weight"), vec![8, from, 3, 3, 3]),
        (format!("{name}.conv1.bias"), vec![8]),
        (format!("{name}.norm2.weight"), vec![8]),
        (format!("{name}.norm2.bias"), vec![8]),
        (format!("{name}.conv2.weight"), vec![8, 8, 3, 3, 3]),
        (format!("{name}.conv2.bias"), vec![8]),
    ];
    if from != 8 {
        tensors.push((
            format!("{name}.conv_shortcut.weight"),
            vec![8, from, 1, 1, 1],
        ));
        tensors.push((format!("{name}.conv_shortcut.bias"), vec![8]));
    }

    tensors
        .into_iter()
        .enumerate()
        .map(|(index, (name, shape))| {
            let values = sequence(shape.iter().product(), seed + index as u32);
            (name, shape, values)
        })
        .collect()
}

const RESNET: ResnetBlock3dConfig = ResnetBlock3dConfig {
    in_channels: 4,
    out_channels: 8,
    groups: 2,
    eps: 1e-6,
    spatial: [1; 4],
    front: 2,
    padding: Padding::Reflect,
    kernel: 3,
    tf32: true,
};

#[test]
fn a_resnet_block_adds_its_input_or_its_projection() {
    let directory = tempfile::tempdir().unwrap();
    let mut tensors = resnet_tensors("wide", 4, 30);
    tensors.extend(resnet_tensors("same", 8, 50));
    let named: Vec<(&str, Vec<usize>, Vec<f32>)> = tensors
        .iter()
        .map(|(name, shape, values)| (name.as_str(), shape.clone(), values.clone()))
        .collect();
    let checkpoint = checkpoint(directory.path(), &named);
    let same_config = ResnetBlock3dConfig {
        in_channels: 8,
        ..RESNET
    };

    // On the CPU the block is the composition its documentation states.
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let wide = ResnetBlock3d::load(&RESNET, loader.root().pp("wide")).unwrap();
    let x = tensor(&sequence(4 * 3 * 4 * 4, 70), (1, 4, 3, 4, 4), DType::F32);
    let out = wide.forward(&x).unwrap();
    assert_eq!(out.dims(), [1, 8, 3, 4, 4]);

    let weight = |name: &str| {
        let (_, shape, values) = tensors.iter().find(|(tensor, ..)| tensor == name).unwrap();
        tensor(values, shape.as_slice(), DType::F32)
    };
    let stage = |x: &Tensor, norm: &str, conv: &str| {
        let padded = functional::frame_norm_pad(
            x,
            Some((
                &weight(&format!("{norm}.weight")),
                &weight(&format!("{norm}.bias")),
            )),
            2,
            1e-6,
            [1; 4],
            2,
            Padding::Reflect,
        )
        .unwrap();
        functional::conv3d(
            &padded,
            &weight(&format!("{conv}.weight")),
            Some(&weight(&format!("{conv}.bias"))),
            [1; 3],
            [0; 3],
            [1; 3],
            1,
            true,
        )
        .unwrap()
    };
    let hidden = stage(&x, "wide.norm1", "wide.conv1");
    let update = stage(&hidden, "wide.norm2", "wide.conv2");
    let shortcut = functional::conv3d(
        &x,
        &weight("wide.conv_shortcut.weight"),
        Some(&weight("wide.conv_shortcut.bias")),
        [1; 3],
        [0; 3],
        [1; 3],
        1,
        true,
    )
    .unwrap();
    assert_eq!(values(&out), values(&(update + shortcut).unwrap()));

    // On a device each normalization is two launches and each convolution one; a block that
    // keeps its width adds its input and launches no projection.
    let store = MetaWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let wide = ResnetBlock3d::load(&RESNET, loader.root().pp("wide")).unwrap();
    let same = ResnetBlock3d::load(&same_config, loader.root().pp("same")).unwrap();

    let (out, launches) = recorded(|| wide.forward(&meta((1, 4, 3, 6, 6), DType::F32)));
    assert_eq!(out.unwrap().dims(), [1, 8, 3, 6, 6]);
    assert_eq!(
        operators(&launches),
        [
            "frame_moments",
            "frame_norm_pad",
            "conv",
            "frame_moments",
            "frame_norm_pad",
            "conv",
            "conv",
            "elementwise_binary",
        ]
    );
    assert_eq!(launches[0].1, "dtype_f32");
    assert_eq!(launches[1].1, "dtype_f32");

    let (out, launches) = recorded(|| same.forward(&meta((1, 8, 3, 6, 6), DType::F32)));
    assert!(out.is_ok());
    assert_eq!(launches.len(), 7, "no projection: {launches:?}");
    let (refused, _) = recorded(|| same.forward(&meta((1, 4, 3, 6, 6), DType::F32)));
    assert!(refused.is_err());

    // A normalization without parameters loads none, and one of no groups is refused.
    let bare = FrameNormConfig {
        channels: 8,
        groups: 4,
        eps: 1e-6,
        affine: false,
        spatial: [1, 1, 1, 0],
        front: 2,
        padding: Padding::Replicate,
    };
    let norm = FrameNorm::load(&bare, loader.root().pp("absent")).unwrap();
    let (padded, launches) = recorded(|| norm.forward(&meta((1, 8, 3, 6, 6), DType::F32)));
    assert_eq!(padded.unwrap().dims(), [1, 8, 5, 8, 7]);
    assert_eq!(operators(&launches), ["frame_moments", "frame_norm_pad"]);
    assert!(FrameNorm::load(&FrameNormConfig { groups: 0, ..bare }, loader.root()).is_err());
}

// ---- AdaLN ------------------------------------------------------------------------------------

const HIDDEN: usize = 8;
const INTERMEDIATE: usize = 12;

const ADALN: AdaLnBlockConfig = AdaLnBlockConfig {
    hidden: HIDDEN as u32,
    eps: 1e-6,
    norm1: "norm1",
    norm2: "norm2",
    mlp: MlpConfig {
        hidden: HIDDEN as u32,
        intermediate: INTERMEDIATE as u32,
        activation: Activation::GeluTanh,
        rounding: Rounding::Stepwise,
        gate_up: GateUp::ValueFirst("ff.proj"),
        down: "ff.down_proj",
        bias: false,
        split: Split::WHOLE,
    },
};

fn adaln_checkpoint(directory: &std::path::Path) -> Checkpoint {
    checkpoint(
        directory,
        &[
            ("block.norm1.weight", vec![HIDDEN], sequence(HIDDEN, 80)),
            ("block.norm2.weight", vec![HIDDEN], sequence(HIDDEN, 81)),
            (
                "block.ff.proj.weight",
                vec![2 * INTERMEDIATE, HIDDEN],
                sequence(2 * INTERMEDIATE * HIDDEN, 82),
            ),
            (
                "block.ff.down_proj.weight",
                vec![HIDDEN, INTERMEDIATE],
                sequence(HIDDEN * INTERMEDIATE, 83),
            ),
            ("norm_out.weight", vec![HIDDEN], sequence(HIDDEN, 84)),
        ],
    )
}

/// The block normalizes the attention's input under its modulation, runs the caller's attention
/// on it, folds the gated update into one modulated norm for the MLP, and joins the MLP's
/// output through its gate.
#[test]
fn an_adaln_block_modulates_both_inputs_and_gates_both_updates() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = adaln_checkpoint(directory.path());
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let block = AdaLnBlock::load(&ADALN, loader.root().pp("block")).unwrap();
    assert_eq!(block.dtype(), DType::F32);

    let tokens = 5;
    let table = tensor(&sequence(3 * 6 * HIDDEN, 85), (3, 6 * HIDDEN), DType::F32);
    let rows = parameter_rows(&[0, 2, 1, 2, 0]);
    let hidden = tensor(&sequence(tokens * HIDDEN, 86), (tokens, HIDDEN), DType::F32);

    // The attention here doubles its input, so the update is twice the modulated norm.
    let out = block
        .forward(&table, &rows, &hidden, |normalized| Ok((normalized * 2.0)?))
        .unwrap();

    let window = |index: usize| table.narrow(1, index * HIDDEN, HIDDEN).unwrap();
    let norm = |name: &str| {
        tensor(
            &sequence(HIDDEN, if name == "norm1" { 80 } else { 81 }),
            HIDDEN,
            DType::F32,
        )
    };
    let attention_input = functional::modulated_rms_norm(
        &hidden,
        &norm("norm1"),
        &window(0),
        &window(1),
        &rows,
        1e-6,
        DType::F32,
    )
    .unwrap();
    assert_eq!(
        values(&block.attention_input(&table, &rows, &hidden).unwrap()),
        values(&attention_input)
    );

    let (normalized, residual) = functional::modulated_rms_norm_(
        (&attention_input * 2.0).unwrap(),
        &hidden,
        &window(2),
        &norm("norm2"),
        &window(3),
        &window(4),
        &rows,
        1e-6,
        DType::F32,
    )
    .unwrap();

    // The MLP of the block: its packed projection, the stepwise gate and the down projection.
    let packed = functional::linear(
        &normalized,
        &tensor(
            &sequence(2 * INTERMEDIATE * HIDDEN, 82),
            (2 * INTERMEDIATE, HIDDEN),
            DType::F32,
        ),
        None,
        functional::WeightOrder::NxK,
    )
    .unwrap();
    let gated = functional::act_and_mul(
        &packed.narrow(1, INTERMEDIATE, INTERMEDIATE).unwrap(),
        &packed.narrow(1, 0, INTERMEDIATE).unwrap(),
        Activation::GeluTanh,
        true,
    )
    .unwrap();
    let projected = functional::linear(
        &gated,
        &tensor(
            &sequence(HIDDEN * INTERMEDIATE, 83),
            (HIDDEN, INTERMEDIATE),
            DType::F32,
        ),
        None,
        functional::WeightOrder::NxK,
    )
    .unwrap();

    // `residual + gate_m[row] · mlp(normalized)`, element by element.
    let (residual, projected, table_values) =
        (values(&residual), values(&projected), values(&table));
    let expected: Vec<f32> = (0..tokens * HIDDEN)
        .map(|index| {
            let (row, column) = (index / HIDDEN, index % HIDDEN);
            let parameter = [0, 2, 1, 2, 0][row];
            let gate = table_values[parameter * 6 * HIDDEN + 5 * HIDDEN + column];
            residual[index] + gate * projected[index]
        })
        .collect();
    assert_close(&values(&out), &expected, F32_TOLERANCE);

    // A final norm's table carries no gated stage.
    let output_table = tensor(&sequence(3 * 2 * HIDDEN, 87), (3, 2 * HIDDEN), DType::F32);
    assert!(
        block
            .attention_input(&output_table, &rows, &hidden)
            .is_err()
    );
}

#[test]
fn an_adaln_block_states_two_modulated_norms_around_the_attention() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = adaln_checkpoint(directory.path());
    let store = MetaWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let block = AdaLnBlock::load(&ADALN, loader.root().pp("block")).unwrap();
    let norm = AdaLnOut::load(
        &AdaLnOutConfig {
            hidden: HIDDEN as u32,
            eps: 1e-6,
        },
        loader.root().pp("norm_out"),
    )
    .unwrap();

    let (table, rows) = (meta((6, 6 * HIDDEN), DType::F32), meta(5, DType::U32));
    let hidden = meta((5, HIDDEN), DType::F32);

    let mut attended = None;
    let (out, launches) = recorded(|| {
        block.forward(&table, &rows, &hidden, |input| {
            attended = Some((input.dims().to_vec(), input.dtype()));
            Ok(input.clone())
        })
    });
    assert_eq!(out.unwrap().dims(), [5, HIDDEN]);
    assert_eq!(attended, Some((vec![5, HIDDEN], DType::F32)));

    // The MLP's gated update joins the stream as a row gather of the packed gate window and
    // broadcast arithmetic.
    assert_eq!(
        operators(&launches),
        [
            "modulated_rms_norm",
            "modulated_rms_norm",
            "gemm",
            "act_and_mul",
            "gemm",
            "copy_strided",
            "index",
            "elementwise_binary",
            "elementwise_binary",
        ]
    );

    // The final norm is one modulated launch of its input's shape, and reads its own table.
    let output_table = meta((2, 2 * HIDDEN), DType::F32);
    let (out, launches) = recorded(|| norm.forward(&output_table, &rows, &hidden));
    assert_eq!(out.unwrap().dims(), [5, HIDDEN]);
    assert_eq!(operators(&launches), ["modulated_rms_norm"]);

    let (refused, _) = recorded(|| norm.forward(&table, &rows, &hidden));
    assert!(refused.is_err());
}

#[test]
fn the_final_adaln_norm_modulates_by_shift_and_scale() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = adaln_checkpoint(directory.path());
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let norm = AdaLnOut::load(
        &AdaLnOutConfig {
            hidden: HIDDEN as u32,
            eps: 1e-6,
        },
        loader.root().pp("norm_out"),
    )
    .unwrap();

    let table = tensor(&sequence(2 * 2 * HIDDEN, 88), (2, 2 * HIDDEN), DType::F32);
    let rows = parameter_rows(&[1, 0, 1]);
    let hidden = sequence(3 * HIDDEN, 89);
    let out = norm
        .forward(&table, &rows, &tensor(&hidden, (3, HIDDEN), DType::F32))
        .unwrap();

    // `rms_norm(x) · weight · (1 + scale[row]) + shift[row]`, in FP32.
    let (weight, table) = (sequence(HIDDEN, 84), values(&table));
    let expected: Vec<f32> = hidden
        .chunks(HIDDEN)
        .zip([1_usize, 0, 1])
        .flat_map(|(row, parameter)| {
            let vectors = &table[parameter * 2 * HIDDEN..][..2 * HIDDEN];
            let mean_square = row.iter().map(|value| value * value).sum::<f32>() / HIDDEN as f32;
            let rsqrt = 1.0 / (mean_square + 1e-6).sqrt();

            (0..HIDDEN)
                .map(|column| {
                    row[column] * rsqrt * weight[column] * (1.0 + vectors[HIDDEN + column])
                        + vectors[column]
                })
                .collect::<Vec<f32>>()
        })
        .collect();
    assert_close(&values(&out), &expected, F32_TOLERANCE);
}
