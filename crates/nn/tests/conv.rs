//! The convolutions and the causal pad: their arithmetic on CPU tensors against each definition
//! written out in loops, the modules loaded from a checkpoint written here, and the kernels they
//! state under a meta context.
//!
//! Nothing here touches a GPU.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use nn::functional::{self, Padding};
use nn::modules::{Conv1d, Conv3d, ConvConfig, ConvTranspose1d};
use nn::{CpuWeights, DType, MetaWeights, WeightLoader};
use nn_testing::{
    Recording, assert_close, checkpoint, meta, operators, record, sequence, tensor, values,
};

/// The tolerance of an FP32 result against a reference accumulated in another order.
const F32_TOLERANCE: f32 = 2e-5;

/// The key of a convolution over FP32 operands: the element types and nothing of the window.
const F32_KEY: &str = "input_f32_weight_f32_output_f32";

/// The arguments of a one-dimensional convolution, as torch names them.
#[derive(Clone, Copy)]
struct Window {
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
}

/// torch's `conv1d` of `x [n, c, l]` with `weight [k, c/groups, t]` by its definition: each
/// output reads its group's channels at `position · stride − padding + tap · dilation`.
fn conv1d_reference(
    x: &[f32],
    [n, c, l]: [usize; 3],
    weight: &[f32],
    [k, cg, t]: [usize; 3],
    bias: &[f32],
    window: Window,
) -> Vec<f32> {
    let out = (l + 2 * window.padding - window.dilation * (t - 1) - 1) / window.stride + 1;
    let mut values = Vec::new();

    for batch in 0..n {
        for channel in 0..k {
            let first = channel / (k / window.groups) * cg;
            for position in 0..out {
                let mut sum = bias[channel];
                for tap_c in 0..cg {
                    for tap in 0..t {
                        let source = position * window.stride + tap * window.dilation;
                        if source < window.padding || source - window.padding >= l {
                            continue;
                        }
                        sum += x[(batch * c + first + tap_c) * l + source - window.padding]
                            * weight[(channel * cg + tap_c) * t + tap];
                    }
                }
                values.push(sum);
            }
        }
    }
    values
}

/// torch's `conv_transpose1d` of `x [n, c, l]` with `weight [c, k/groups, t]` by its definition:
/// each input element adds its filter into the outputs at
/// `position · stride − padding + tap · dilation`.
fn conv_transpose1d_reference(
    x: &[f32],
    [n, c, l]: [usize; 3],
    weight: &[f32],
    [_, kg, t]: [usize; 3],
    bias: &[f32],
    window: Window,
) -> Vec<f32> {
    let k = kg * window.groups;
    let out = (l - 1) * window.stride + window.dilation * (t - 1) + 1 - 2 * window.padding;
    let mut values: Vec<f32> = (0..n * k * out)
        .map(|index| bias[index / out % k])
        .collect();

    for batch in 0..n {
        for channel in 0..c {
            let group = channel / (c / window.groups);
            for position in 0..l {
                for tap_k in 0..kg {
                    for tap in 0..t {
                        let target = position * window.stride + tap * window.dilation;
                        if target < window.padding || target - window.padding >= out {
                            continue;
                        }
                        values[(batch * k + group * kg + tap_k) * out + target - window.padding] +=
                            x[(batch * c + channel) * l + position]
                                * weight[(channel * kg + tap_k) * t + tap];
                    }
                }
            }
        }
    }
    values
}

#[test]
fn conv1d_sums_its_group_s_channels_over_the_filter_s_taps() {
    let window = Window {
        stride: 2,
        padding: 3,
        dilation: 2,
        groups: 2,
    };
    let (x, weight, bias) = (
        sequence(2 * 4 * 11, 1),
        sequence(6 * 2 * 3, 2),
        sequence(6, 3),
    );

    let out = functional::conv1d(
        &tensor(&x, (2, 4, 11), DType::F32),
        &tensor(&weight, (6, 2, 3), DType::F32),
        Some(&tensor(&bias, 6, DType::F32)),
        2,
        3,
        2,
        2,
        false,
    )
    .unwrap();

    // (11 + 6 − 2·2 − 1) / 2 + 1 positions.
    assert_eq!(out.dims(), [2, 6, 7]);
    assert_close(
        &values(&out),
        &conv1d_reference(&x, [2, 4, 11], &weight, [6, 2, 3], &bias, window),
        F32_TOLERANCE,
    );
}

#[test]
fn conv1d_rounds_its_fp32_sum_once_into_the_input_s_type() {
    let (x, weight) = (sequence(3 * 9, 4), sequence(5 * 3 * 4, 5));
    let window = Window {
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
    };

    let narrow = |values: &[f32], shape: (usize, usize, usize)| tensor(values, shape, DType::BF16);
    let out = functional::conv1d(
        &narrow(&x, (1, 3, 9)),
        &narrow(&weight, (5, 3, 4)),
        None,
        1,
        0,
        1,
        1,
        false,
    )
    .unwrap();
    assert_eq!(out.dtype(), DType::BF16);

    // The reference reads the operands as the BF16 values they are and rounds its sum once.
    let expected = conv1d_reference(
        &values(&narrow(&x, (1, 3, 9))),
        [1, 3, 9],
        &values(&narrow(&weight, (5, 3, 4))),
        [5, 3, 4],
        &[0.0; 5],
        window,
    );
    assert_close(
        &values(&out),
        &values(&tensor(&expected, (1, 5, 6), DType::BF16)),
        1.0 / 128.0,
    );
}

#[test]
fn conv_transpose1d_adds_each_input_s_filter_into_its_outputs() {
    let window = Window {
        stride: 3,
        padding: 2,
        dilation: 1,
        groups: 2,
    };
    let (x, weight, bias) = (
        sequence(2 * 4 * 5, 6),
        sequence(4 * 3 * 7, 7),
        sequence(6, 8),
    );

    let out = functional::conv_transpose1d(
        &tensor(&x, (2, 4, 5), DType::F32),
        &tensor(&weight, (4, 3, 7), DType::F32),
        Some(&tensor(&bias, 6, DType::F32)),
        3,
        2,
        1,
        2,
        false,
    )
    .unwrap();

    // (5 − 1)·3 − 2·2 + (7 − 1) + 1 positions: torch's transposed length.
    assert_eq!(out.dims(), [2, 6, 15]);
    assert_close(
        &values(&out),
        &conv_transpose1d_reference(&x, [2, 4, 5], &weight, [4, 3, 7], &bias, window),
        F32_TOLERANCE,
    );
}

/// torch's `conv3d` in the scatter order: each input element adds its products into the outputs
/// whose window holds it.
#[expect(
    clippy::too_many_arguments,
    reason = "the operands and the window of one reference convolution"
)]
fn conv3d_reference(
    x: &[f32],
    [n, c, d, h, w]: [usize; 5],
    weight: &[f32],
    [k, cg, kd, kh, kw]: [usize; 5],
    bias: &[f32],
    stride: [usize; 3],
    padding: [usize; 3],
    groups: usize,
) -> (Vec<f32>, [usize; 5]) {
    let extent = |axis: usize, length: usize, taps: usize| {
        (length + 2 * padding[axis] - taps) / stride[axis] + 1
    };
    let (od, oh, ow) = (extent(0, d, kd), extent(1, h, kh), extent(2, w, kw));
    let mut values: Vec<f32> = (0..n * k * od * oh * ow)
        .map(|index| bias[index / (od * oh * ow) % k])
        .collect();

    // The output position whose tap `tap` reads input position `source`, when there is one.
    let target = |axis: usize, source: usize, tap: usize, out: usize| {
        (source + padding[axis])
            .checked_sub(tap)
            .filter(|reach| reach % stride[axis] == 0)
            .map(|reach| reach / stride[axis])
            .filter(|position| *position < out)
    };

    for batch in 0..n {
        for channel in 0..c {
            let group = channel / cg;
            for (z, y, x_) in
                (0..d).flat_map(|z| (0..h).flat_map(move |y| (0..w).map(move |x| (z, y, x))))
            {
                let element = x[(((batch * c + channel) * d + z) * h + y) * w + x_];

                for out_channel in group * (k / groups)..(group + 1) * (k / groups) {
                    for (tz, ty, tx) in (0..kd)
                        .flat_map(|z| (0..kh).flat_map(move |y| (0..kw).map(move |x| (z, y, x))))
                    {
                        let (Some(oz), Some(oy), Some(ox)) = (
                            target(0, z, tz, od),
                            target(1, y, ty, oh),
                            target(2, x_, tx, ow),
                        ) else {
                            continue;
                        };
                        let tap =
                            (((out_channel * cg + channel % cg) * kd + tz) * kh + ty) * kw + tx;
                        values[(((batch * k + out_channel) * od + oz) * oh + oy) * ow + ox] +=
                            element * weight[tap];
                    }
                }
            }
        }
    }
    (values, [n, k, od, oh, ow])
}

#[test]
fn conv3d_sums_each_window_of_its_group_s_channels() {
    let (x, weight, bias) = (
        sequence(2 * 4 * 5 * 6 * 7, 9),
        sequence(6 * 2 * 3 * 3 * 2, 10),
        sequence(6, 11),
    );

    let out = functional::conv3d(
        &tensor(&x, (2, 4, 5, 6, 7), DType::F32),
        &tensor(&weight, (6, 2, 3, 3, 2), DType::F32),
        Some(&tensor(&bias, 6, DType::F32)),
        [1, 2, 2],
        [1, 0, 1],
        [1, 1, 1],
        2,
        true,
    )
    .unwrap();

    let (expected, dims) = conv3d_reference(
        &x,
        [2, 4, 5, 6, 7],
        &weight,
        [6, 2, 3, 3, 2],
        &bias,
        [1, 2, 2],
        [1, 0, 1],
        2,
    );
    assert_eq!(out.dims(), dims);
    assert_close(&values(&out), &expected, F32_TOLERANCE);
}

#[test]
fn a_convolution_refuses_operands_that_are_not_its_window() {
    let x = tensor(&sequence(4 * 8, 12), (1, 4, 8), DType::F32);
    let weight = tensor(&sequence(6 * 4 * 3, 13), (6, 4, 3), DType::F32);
    let conv = |x: &nn::Tensor, weight: &nn::Tensor, bias: Option<&nn::Tensor>, stride, groups| {
        functional::conv1d(x, weight, bias, stride, 0, 1, groups, false)
    };

    // Channels the filter does not read, no groups, no stride, a bias of another width.
    assert!(conv(&x, &weight, None, 1, 2).is_err());
    assert!(conv(&x, &weight, None, 1, 0).is_err());
    assert!(conv(&x, &weight, None, 0, 1).is_err());
    let bias = tensor(&[0.0; 4], 4, DType::F32);
    assert!(conv(&x, &weight, Some(&bias), 1, 1).is_err());

    // An input shorter than the filter's reach leaves no output.
    let short = tensor(&sequence(4 * 2, 14), (1, 4, 2), DType::F32);
    let refused = conv(&short, &weight, None, 1, 1).unwrap_err().to_string();
    assert!(refused.contains("leaves no output"), "{refused}");
}

#[test]
fn the_convolution_modules_load_the_filters_their_torch_arguments_state() {
    let directory = tempfile::tempdir().unwrap();
    let (direction, gain) = (
        sequence(8 * 4 * 3, 15),
        vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    );
    let (bias, up, up_bias) = (sequence(8, 16), sequence(8 * 4 * 4, 17), sequence(4, 18));
    let (frame, frame_bias) = (sequence(6 * 4 * 2 * 2 * 2, 19), sequence(6, 20));
    let checkpoint = checkpoint(
        directory.path(),
        DType::F32,
        &[
            ("conv.weight_v", vec![8, 4, 3], direction.clone()),
            ("conv.weight_g", vec![8, 1, 1], gain.clone()),
            ("conv.bias", vec![8], bias.clone()),
            ("up.weight", vec![8, 4, 4], up.clone()),
            ("up.bias", vec![4], up_bias.clone()),
            ("frame.weight", vec![6, 4, 2, 2, 2], frame.clone()),
            ("frame.bias", vec![6], frame_bias.clone()),
        ],
    );
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);

    let config = ConvConfig {
        padding: [1],
        weight_norm: true,
        ..ConvConfig::new(4, 8, [3])
    };
    let conv = Conv1d::load(&config, loader.root().pp("conv")).unwrap();

    // The weight-norm pair is folded once: `v · g / ‖v‖`, the norm over each output channel.
    let folded: Vec<f32> = direction
        .chunks(4 * 3)
        .zip(&gain)
        .flat_map(|(channel, gain)| {
            let norm = channel
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            channel.iter().map(move |value| value * gain / norm)
        })
        .collect();
    let x = sequence(2 * 4 * 16, 21);
    let out = conv.forward(&tensor(&x, (2, 4, 16), DType::F32)).unwrap();
    let window = Window {
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
    };
    assert_eq!(out.dims(), [2, 8, 16]);
    assert_close(
        &values(&out),
        &conv1d_reference(&x, [2, 4, 16], &folded, [8, 4, 3], &bias, window),
        F32_TOLERANCE,
    );

    // A filter of another width than the config states, and a config that is no convolution.
    let wider = ConvConfig {
        out_channels: 16,
        ..config
    };
    assert!(Conv1d::load(&wider, loader.root().pp("conv")).is_err());
    let ungrouped = ConvConfig {
        groups: 0,
        ..config
    };
    assert!(Conv1d::load(&ungrouped, loader.root().pp("conv")).is_err());

    // The transposed module reads torch's `[in, out, k]` filter.
    let up_config = ConvConfig {
        stride: [2],
        padding: [1],
        ..ConvConfig::new(8, 4, [4])
    };
    let upsample = ConvTranspose1d::load(&up_config, loader.root().pp("up")).unwrap();
    let doubled = upsample.forward(&out).unwrap();
    let window = Window {
        stride: 2,
        padding: 1,
        dilation: 1,
        groups: 1,
    };
    assert_eq!(doubled.dims(), [2, 4, 32], "torch's transposed length");
    assert_close(
        &values(&doubled),
        &conv_transpose1d_reference(&values(&out), [2, 8, 16], &up, [8, 4, 4], &up_bias, window),
        F32_TOLERANCE,
    );

    let frame_config = ConvConfig::new(4, 6, [2; 3]);
    let conv3d = Conv3d::load(&frame_config, loader.root().pp("frame")).unwrap();
    let pixels = sequence(4 * 3 * 3 * 3, 22);
    let out = conv3d
        .forward(&tensor(&pixels, (1, 4, 3, 3, 3), DType::F32))
        .unwrap();
    let (expected, dims) = conv3d_reference(
        &pixels,
        [1, 4, 3, 3, 3],
        &frame,
        [6, 4, 2, 2, 2],
        &frame_bias,
        [1; 3],
        [0; 3],
        1,
    );
    assert_eq!(out.dims(), dims);
    assert_close(&values(&out), &expected, F32_TOLERANCE);

    // A module over weights a model computed checks them against its config.
    let computed = tensor(&frame, (6, 4, 2, 2, 2), DType::F32);
    let unbiased = ConvConfig {
        bias: false,
        ..frame_config
    };
    assert!(Conv3d::new(computed.clone(), None, &unbiased).is_ok());
    assert!(Conv3d::new(computed, None, &frame_config).is_err());
}

#[test]
fn each_convolution_is_one_launch_keyed_by_its_element_types() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        DType::F32,
        &[
            ("conv.weight", vec![8, 2, 3], sequence(8 * 2 * 3, 23)),
            ("conv.bias", vec![8], sequence(8, 24)),
            ("up.weight", vec![8, 4, 4], sequence(8 * 4 * 4, 25)),
            ("frame.weight", vec![8, 4, 2, 2, 2], sequence(8 * 4 * 8, 26)),
        ],
    );
    let store = MetaWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);

    // A grouped, biased convolution is still one launch: the groups and the bias travel in it.
    let grouped = ConvConfig {
        groups: 2,
        padding: [1],
        ..ConvConfig::new(4, 8, [3])
    };
    let conv = Conv1d::load(&grouped, loader.root().pp("conv")).unwrap();
    let up_config = ConvConfig {
        stride: [2],
        padding: [1],
        bias: false,
        ..ConvConfig::new(8, 4, [4])
    };
    let up = ConvTranspose1d::load(&up_config, loader.root().pp("up")).unwrap();

    let (output, Recording { launches, .. }) = record(&[], || {
        let out = conv.forward(&meta((2, 4, 16), DType::F32))?;
        Ok((out.dims().to_vec(), up.forward(&out)?.dims().to_vec()))
    });
    assert_eq!(output.unwrap(), (vec![2, 8, 16], vec![2, 4, 32]));
    assert_eq!(
        launches,
        [("conv", F32_KEY), ("conv", F32_KEY)].map(|(operator, key)| (operator, key.to_owned()))
    );

    // A frame stored channels-last launches the channel-first kernel, and its output is stored
    // the way its input is.
    let unbiased = ConvConfig {
        bias: false,
        ..ConvConfig::new(4, 8, [2; 3])
    };
    let frame = Conv3d::load(&unbiased, loader.root().pp("frame")).unwrap();
    let strides = |channels_last: bool| {
        record(&[], || {
            let x = if channels_last {
                meta((1, 3, 3, 3, 4), DType::F32).permute((0, 4, 1, 2, 3))?
            } else {
                meta((1, 4, 3, 3, 3), DType::F32)
            };
            let out = frame.forward(&x)?;
            Ok((out.dims().to_vec(), out.layout().stride().to_vec()))
        })
    };

    let (
        first,
        Recording {
            launches: first_launches,
            ..
        },
    ) = strides(false);
    assert_eq!(first.unwrap(), (vec![1, 8, 2, 2, 2], vec![64, 8, 4, 2, 1]));
    assert_eq!(first_launches, [("conv", F32_KEY.to_owned())]);

    let (
        last,
        Recording {
            launches: last_launches,
            ..
        },
    ) = strides(true);
    assert_eq!(last.unwrap(), (vec![1, 8, 2, 2, 2], vec![64, 1, 32, 16, 8]));
    assert_eq!(last_launches, first_launches, "the order keys no kernel");
}

/// The frame `x [c, d, h, w]` padded by the definition: each output element reads the source
/// element its mirrored or clamped row and column name, and a frame in front reads zero.
fn causal_pad_reference(
    x: &[f32],
    [c, d, h, w]: [usize; 4],
    [top, bottom, left, right]: [usize; 4],
    front: usize,
    padding: Padding,
) -> Vec<f32> {
    // The source index of padded index `index` along an axis of `length` after `before` pads.
    let source = |index: usize, before: usize, length: usize| {
        let offset = index as isize - before as isize;
        let last = length as isize - 1;
        let inside = match padding {
            Padding::Replicate => offset.clamp(0, last),
            Padding::Reflect if offset < 0 => -offset,
            Padding::Reflect if offset > last => 2 * last - offset,
            Padding::Reflect => offset,
        };
        inside as usize
    };

    let mut values = Vec::new();
    for channel in 0..c {
        for frame in 0..d + front {
            for row in 0..h + top + bottom {
                for column in 0..w + left + right {
                    values.push(if frame < front {
                        0.0
                    } else {
                        x[((channel * d + frame - front) * h + source(row, top, h)) * w
                            + source(column, left, w)]
                    });
                }
            }
        }
    }
    values
}

#[test]
fn a_causal_pad_fills_the_border_by_its_mode_and_the_front_with_zero_frames() {
    let x = sequence(2 * 3 * 4 * 5, 27);
    let frame = tensor(&x, (1, 2, 3, 4, 5), DType::F32);

    for padding in [Padding::Reflect, Padding::Replicate] {
        let padded = functional::causal_pad(&frame, [2, 1, 1, 3], 2, padding).unwrap();
        assert_eq!(padded.dims(), [1, 2, 5, 7, 9]);
        assert_eq!(
            values(&padded),
            causal_pad_reference(&x, [2, 3, 4, 5], [2, 1, 1, 3], 2, padding),
            "{padding:?}"
        );
    }

    // A mirror reads inside the edge, so it cannot be as wide as the frame.
    assert!(functional::causal_pad(&frame, [4, 0, 0, 0], 0, Padding::Reflect).is_err());
    assert!(functional::causal_pad(&frame, [4, 0, 0, 0], 0, Padding::Replicate).is_ok());
    assert!(
        functional::causal_pad(
            &tensor(&x, (2, 3, 4, 5), DType::F32),
            [1; 4],
            2,
            Padding::Reflect
        )
        .is_err()
    );
}

#[test]
fn a_causal_pad_on_a_device_is_a_concatenation_of_generic_primitives() {
    let (padded, Recording { launches, .. }) = record(&[], || {
        functional::causal_pad(
            &meta((1, 3, 5, 16, 16), DType::F32),
            [1; 4],
            2,
            Padding::Reflect,
        )
    });

    assert_eq!(padded.unwrap().dims(), [1, 3, 7, 18, 18]);
    assert!(!launches.is_empty());
    assert!(
        operators(&launches)
            .iter()
            .all(|operator| ["copy_strided", "fill"].contains(operator)),
        "{launches:?}"
    );
}
