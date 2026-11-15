//! The tiled VAE functions, the alias-free activation, the vocoder and the audio encoder: their
//! arithmetic on CPU tensors against each definition written out in loops, and the kernels they
//! state under a meta context.
//!
//! Nothing here touches a GPU.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

mod support;

use std::ops::Range;

use nn::modules::{
    Activation1d, Activation1dConfig, BigVgan, BigVganConfig, DacEncoder, DacEncoderConfig,
};
use nn::{Checkpoint, CpuWeights, DType, MetaWeights, Tensor, WeightLoader, functional, vae};
use support::{
    assert_close, f32_bytes, meta, operators, recorded, recorded_split, sequence, shard, tensor,
    values,
};

/// `value` rounded into BF16, to nearest even, as the library's conversion rounds it.
fn bf16(value: f32) -> f32 {
    values(&tensor(&[value], 1, DType::BF16))[0]
}

/// The one tile of an axis that a single tile covers.
fn whole(length: usize) -> Vec<Range<usize>> {
    std::iter::once(0..length).collect()
}

// ---- Blend ------------------------------------------------------------------------------------

/// The reference's fade of one position: the ramp, its complement, both products and their sum
/// each rounded by `round`.
fn fade(previous: f32, current: f32, position: usize, extent: usize, round: fn(f32) -> f32) -> f32 {
    let ratio = round(round(position as f32) / round(extent as f32));
    let complement = round(1.0 - ratio);
    round(round(previous * complement) + round(current * ratio))
}

#[test]
fn a_blend_fades_the_leading_positions_in_the_element_type() {
    let dims = [2, 3, 4, 5];
    let (previous, current) = (sequence(120, 1), sequence(120, 2));
    let index = |[c, f, h, w]: [usize; 4]| ((c * 3 + f) * 4 + h) * 5 + w;

    for (dtype, round) in [(DType::F32, (|x| x) as fn(f32) -> f32), (DType::BF16, bf16)] {
        let (a, b) = (
            tensor(&previous, &dims, dtype),
            tensor(&current, &dims, dtype),
        );
        let (a_values, b_values) = (values(&a), values(&b));

        // Along the rows, the columns and the frames of a `[channels, frames, rows, columns]`
        // block: the previous block's last positions against the current block's first.
        for (axis, extent) in [(2, 3), (3, 2), (1, 2)] {
            let blended = vae::blend(&a, &b, extent, axis).unwrap();
            assert_eq!(blended.dims(), dims);

            let mut expected = b_values.clone();
            for c in 0..2 {
                for f in 0..3 {
                    for h in 0..4 {
                        for w in 0..5 {
                            let mut at = [c, f, h, w];
                            let position = at[axis];
                            if position >= extent {
                                continue;
                            }
                            at[axis] = dims[axis] - extent + position;

                            expected[index([c, f, h, w])] = fade(
                                a_values[index(at)],
                                b_values[index([c, f, h, w])],
                                position,
                                extent,
                                round,
                            );
                        }
                    }
                }
            }
            assert_eq!(values(&blended), expected, "{dtype:?} along axis {axis}");
        }
    }

    // A clip fades its first frames against its predecessor's tail of exactly those frames.
    let tail = tensor(&previous[..40], (2, 1, 4, 5), DType::F32);
    let clip = tensor(&current, &dims, DType::F32);
    let faded = vae::blend(&tail, &clip, 1, 1).unwrap();
    assert_eq!(faded.dims(), dims);
    assert_eq!(
        values(&faded)[..20],
        previous[..20],
        "position zero is the previous frame"
    );

    assert!(
        vae::blend(&tail, &clip, 0, 1).is_err(),
        "no position to fade"
    );
    assert!(
        vae::blend(&tail, &clip, 2, 1).is_err(),
        "a tail of one frame"
    );
    let narrow = tensor(&current, &dims, DType::BF16);
    assert!(
        vae::blend(&clip, &narrow, 1, 2).is_err(),
        "two element types"
    );
}

// ---- Tiles ------------------------------------------------------------------------------------

/// The tiles of `raster`, `[channels, height, width]` values, cut at `rows` by `columns`.
fn cut(
    raster: &[f32],
    [channels, height, width]: [usize; 3],
    rows: &[Range<usize>],
    columns: &[Range<usize>],
) -> Vec<Tensor> {
    let whole = tensor(raster, (channels, height, width), DType::F32);

    rows.iter()
        .flat_map(|tile_rows| {
            let strip = whole.narrow(1, tile_rows.start, tile_rows.len()).unwrap();
            columns
                .iter()
                .map(move |tile_columns| {
                    strip
                        .narrow(2, tile_columns.start, tile_columns.len())
                        .unwrap()
                        .contiguous()
                        .unwrap()
                })
                .collect::<Vec<Tensor>>()
        })
        .collect()
}

#[test]
fn a_stitch_fades_each_tile_over_its_neighbours_as_they_were_written() {
    // A 10 by 12 raster in 2 by 3 tiles of 6 by 6 cells: the rows share two cells, the columns
    // three.
    let rows = vae::split_tiles(10, 6, 2, 1).unwrap();
    let columns = vae::split_tiles(12, 6, 3, 1).unwrap();
    assert_eq!(
        (rows.clone(), columns.clone()),
        (vec![0..6, 4..10], vec![0..6, 3..9, 6..12])
    );

    // Tiles cut from one raster agree where they overlap, so the stitch is the raster.
    let raster = sequence(2 * 10 * 12, 3);
    let stitched =
        vae::stitch_tiles(&cut(&raster, [2, 10, 12], &rows, &columns), &rows, &columns).unwrap();
    assert_eq!(stitched.dims(), [2, 10, 12]);
    assert_close(&values(&stitched), &raster, 1e-6);

    // Tiles that disagree: each pixel is its own tile's value, faded against the tile above
    // where it lies in their seam and then against the tile to the left, both as written.
    let tiles: Vec<Vec<f32>> = (0..6).map(|tile| sequence(2 * 6 * 6, 10 + tile)).collect();
    let blocks: Vec<Tensor> = tiles
        .iter()
        .map(|tile| tensor(tile, (2, 6, 6), DType::F32))
        .collect();
    let stitched = vae::stitch_tiles(&blocks, &rows, &columns).unwrap();

    // The tile that keeps a position along an axis, the position inside it, and its seam.
    let keeper = |ranges: &[Range<usize>], position: usize| {
        let tile = ranges
            .iter()
            .rposition(|range| range.start <= position)
            .unwrap();
        let seam = if tile == 0 {
            0
        } else {
            ranges[tile - 1].end - ranges[tile].start
        };
        (tile, position - ranges[tile].start, seam)
    };
    let mut expected = Vec::new();
    for channel in 0..2 {
        for y in 0..10 {
            for x in 0..12 {
                let (tile_row, row, row_seam) = keeper(&rows, y);
                let (tile_column, column, column_seam) = keeper(&columns, x);
                let tile = tile_row * 3 + tile_column;
                let at = |tile: usize, row: usize, column: usize| {
                    tiles[tile][(channel * 6 + row) * 6 + column]
                };

                let mut value = at(tile, row, column);
                if row < row_seam {
                    let above = at(tile - 3, 6 - row_seam + row, column);
                    value = fade(above, value, row, row_seam, |x| x);
                }
                if column < column_seam {
                    let left = at(tile - 1, row, 6 - column_seam + column);
                    value = fade(left, value, column, column_seam, |x| x);
                }
                expected.push(value);
            }
        }
    }
    assert_eq!(values(&stitched), expected);

    // A grid takes one block per tile, each of its tile's extents.
    assert!(vae::stitch_tiles(&blocks[..5], &rows, &columns).is_err());
    assert!(vae::stitch_tiles(&blocks, &[0..5, 3..8], &columns).is_err());
    assert!(vae::stitch_tiles(&blocks, &[4..10, 0..6], &columns).is_err());
}

#[test]
fn a_tiled_encode_runs_each_tile_and_stitches_the_blocks() {
    // A 2 by 3 latent raster in one tile row of two tiles that share one column.
    let rows = vae::split_tiles(32, 32, 16, 16).unwrap();
    let columns = vae::split_tiles(48, 32, 16, 16).unwrap();
    let cells = |ranges: &[Range<usize>]| -> Vec<Range<usize>> {
        ranges
            .iter()
            .map(|range| range.start / 16..range.end / 16)
            .collect()
    };
    let (rows, columns) = (cells(&rows), cells(&columns));
    assert_eq!(
        (rows.clone(), columns.clone()),
        (whole(2), vec![0..2, 1..3])
    );

    // The encoder of a tile is the tile's cells of one latent, so the stitch is the latent.
    let latent = sequence(4 * 3 * 2 * 3, 20);
    let whole = tensor(&latent, (4, 3, 2, 3), DType::F32);
    let mut encoded = Vec::new();
    let raster = vae::tiled_encode(&rows, &columns, &[4, 3], DType::F32, |index| {
        encoded.push(index);
        Ok(whole.narrow(3, columns[index].start, 2)?.contiguous()?)
    })
    .unwrap();
    assert_eq!(encoded, [0, 1]);
    assert_eq!(raster.dims(), [4, 3, 2, 3]);
    assert_close(&values(&raster), &latent, 1e-6);

    // A block of another extent or element type than its tile is refused.
    let wide = |_| Ok(whole.clone());
    assert!(vae::tiled_encode(&rows, &columns, &[4, 3], DType::F32, wide).is_err());
    let half = |_| Ok(tensor(&latent[..48], (4, 3, 2, 2), DType::F16));
    assert!(vae::tiled_encode(&rows, &columns, &[4, 3], DType::F32, half).is_err());

    // In a group of two, each rank encodes its share and gathers the other's before the
    // stitch; the blend and the assembly are generic primitives.
    let mut encoded = Vec::new();
    let (raster, launches) = recorded_split(Some((1, 2)), || {
        vae::tiled_encode(&rows, &columns, &[4, 3], DType::F32, |index| {
            encoded.push(index);
            Ok(meta((4, 3, 2, 2), DType::F32))
        })
    });
    assert_eq!(raster.unwrap().dims(), [4, 3, 2, 3]);
    assert_eq!(encoded, [1], "rank 1 of 2 encodes the second tile");
    assert!(!launches.is_empty());
    assert!(
        operators(&launches)
            .iter()
            .all(|operator| ["copy_strided", "elementwise_binary", "cast"].contains(operator)),
        "{launches:?}"
    );
}

#[test]
fn the_voxels_of_a_tile_are_its_own_cells_frame_by_frame() {
    let (rows, columns) = (vec![0..2, 1..3], whole(2));
    let voxels = vae::tile_voxels(&rows, &columns, 2, 3, 4).unwrap();

    let mut expected = Vec::new();
    for tile_rows in &rows {
        for tile_columns in &columns {
            for frame in 0..2 {
                for row in tile_rows.clone() {
                    for column in tile_columns.clone() {
                        expected.push(((frame * 3 + row) * 4 + column) as u32);
                    }
                }
            }
        }
    }
    assert_eq!(voxels.dtype(), DType::U32);
    assert_eq!(voxels.to_vec1::<u32>().unwrap(), expected);

    // A voxel of the overlap has one entry per tile that holds it.
    assert_eq!(expected.iter().filter(|voxel| **voxel == 4).count(), 2);
    assert!(
        vae::tile_voxels(&rows, &columns, 2, 2, 4).is_err(),
        "a tile below the raster"
    );
}

// ---- The alias-free activation ----------------------------------------------------------------

/// The alias-free SnakeBeta of one channel, stage by stage.
fn activation_reference(
    x: &[f32],
    alpha: f32,
    beta: f32,
    up: &[f32],
    down: &[f32],
    ratio: usize,
) -> Vec<f32> {
    let kernel = up.len();
    let replicated = |values: &[f32], left: usize, right: usize| -> Vec<f32> {
        let mut padded = vec![values[0]; left];
        padded.extend_from_slice(values);
        padded.extend(std::iter::repeat_n(values[values.len() - 1], right));
        padded
    };

    // The transposed convolution: every padded sample adds its filter at stride `ratio`.
    let pad = kernel / ratio - 1;
    let padded = replicated(x, pad, pad);
    let mut upsampled = vec![0.0_f32; (padded.len() - 1) * ratio + kernel];
    for (index, sample) in padded.iter().enumerate() {
        for (tap, weight) in up.iter().enumerate() {
            upsampled[index * ratio + tap] += sample * weight;
        }
    }
    let crop = 3 * (kernel - ratio) / 2;
    let activated: Vec<f32> = upsampled[crop..crop + x.len() * ratio]
        .iter()
        .map(|sample| {
            let sample = sample * ratio as f32;
            let angle = (alpha.exp() * sample).sin();
            sample + angle * angle / (beta.exp() + 1e-9)
        })
        .collect();

    // The low-pass: a strided correlation of the replicate-padded signal.
    let padded = replicated(&activated, kernel / 2 - 1, kernel / 2);
    (0..x.len())
        .map(|index| {
            down.iter()
                .enumerate()
                .map(|(tap, weight)| padded[index * ratio + tap] * weight)
                .sum()
        })
        .collect()
}

#[test]
fn the_alias_free_activation_upsamples_activates_and_filters() {
    const CHANNELS: usize = 3;
    const LENGTH: usize = 9;
    let x = sequence(2 * CHANNELS * LENGTH, 30);
    let (alpha, beta) = (sequence(CHANNELS, 31), sequence(CHANNELS, 32));
    let (up, down) = (sequence(12, 33), sequence(12, 34));

    let out = functional::activation1d(
        &tensor(&x, (2, CHANNELS, LENGTH), DType::F32),
        &tensor(&alpha, CHANNELS, DType::F32),
        &tensor(&beta, CHANNELS, DType::F32),
        &tensor(&up, (1, 1, 12), DType::F32),
        &tensor(&down, (1, 1, 12), DType::F32),
        2,
    )
    .unwrap();
    assert_eq!(
        out.dims(),
        [2, CHANNELS, LENGTH],
        "the chain keeps the length"
    );

    let expected: Vec<f32> = x
        .chunks(LENGTH)
        .enumerate()
        .flat_map(|(row, signal)| {
            let channel = row % CHANNELS;
            activation_reference(signal, alpha[channel], beta[channel], &up, &down, 2)
        })
        .collect();
    assert_close(&values(&out), &expected, 1e-4);

    // A signal of other channels than the parameters, and a filter shorter than the ratio.
    let refuse = |x: &Tensor, filter: &Tensor, ratio| {
        let parameter = tensor(&alpha, CHANNELS, DType::F32);
        functional::activation1d(x, &parameter, &parameter, filter, filter, ratio).is_err()
    };
    let filter = tensor(&up, (1, 1, 12), DType::F32);
    assert!(refuse(
        &tensor(&x[..36], (2, 2, LENGTH), DType::F32),
        &filter,
        2
    ));
    assert!(refuse(
        &tensor(&x, (2, CHANNELS, LENGTH), DType::F32),
        &filter,
        13
    ));
}

// ---- Vocoder and audio encoder ----------------------------------------------------------------

/// The checkpoint at `directory` holding FP32 tensors of the stated extents and fixed values.
fn checkpoint(directory: &std::path::Path, tensors: &[(String, Vec<usize>)]) -> Checkpoint {
    let typed: Vec<(&str, DType, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .enumerate()
        .map(|(index, (name, shape))| {
            let values = sequence(shape.iter().product(), 40 + index as u32);
            (name.as_str(), DType::F32, shape.clone(), f32_bytes(&values))
        })
        .collect();
    shard(directory, "model.safetensors", &typed);

    Checkpoint::open(directory).unwrap()
}

/// The weight-normalized module `name` of a 1-d convolution: its `weight_v`, `weight_g` and,
/// with `bias`, its bias of the filter's output channels.
fn normalized_conv(
    name: &str,
    filter: [usize; 3],
    bias: Option<usize>,
) -> Vec<(String, Vec<usize>)> {
    let mut tensors = vec![
        (format!("{name}.weight_v"), filter.to_vec()),
        (format!("{name}.weight_g"), vec![filter[0], 1, 1]),
    ];
    if let Some(channels) = bias {
        tensors.push((format!("{name}.bias"), vec![channels]));
    }
    tensors
}

/// An alias-free activation module of `channels` and its 12-tap filters.
fn activation(name: &str, channels: usize) -> Vec<(String, Vec<usize>)> {
    vec![
        (format!("{name}.act.alpha"), vec![channels]),
        (format!("{name}.act.beta"), vec![channels]),
        (format!("{name}.upsample.filter"), vec![1, 1, 12]),
        (format!("{name}.downsample.lowpass.filter"), vec![1, 1, 12]),
    ]
}

/// A one-stage BigVGAN of one residual kernel at two dilations, as its checkpoint names it.
fn bigvgan_tensors() -> Vec<(String, Vec<usize>)> {
    let mut tensors = normalized_conv("decoder.conv_pre", [8, 4, 7], Some(8));
    tensors.extend(normalized_conv("decoder.ups.0.0", [8, 4, 4], Some(4)));
    for step in 0..2 {
        let block = "decoder.resblocks.0";
        tensors.extend(normalized_conv(
            &format!("{block}.convs1.{step}"),
            [4, 4, 3],
            Some(4),
        ));
        tensors.extend(normalized_conv(
            &format!("{block}.convs2.{step}"),
            [4, 4, 3],
            Some(4),
        ));
        tensors.extend(activation(&format!("{block}.activations.{}", 2 * step), 4));
        tensors.extend(activation(
            &format!("{block}.activations.{}", 2 * step + 1),
            4,
        ));
    }
    tensors.extend(activation("decoder.activation_post", 4));
    tensors.extend(normalized_conv("decoder.conv_post", [1, 4, 7], None));
    tensors
}

fn bigvgan_config() -> BigVganConfig {
    BigVganConfig {
        in_channels: 4,
        channels: 8,
        upsample_rates: vec![2],
        upsample_kernels: vec![4],
        resblock_kernels: vec![3],
        resblock_dilations: vec![vec![1, 3]],
        activation_ratio: 2,
        final_bias: false,
    }
}

/// A one-stage BigVGAN loads its modules by the generator's names and runs pre-convolution,
/// upsample, the block's dilation steps, the post activation and the post-convolution; the
/// output is the input length times the stage rates.
#[test]
fn a_bigvgan_generator_upsamples_through_its_stages() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path(), &bigvgan_tensors());
    let config = bigvgan_config();

    let store = MetaWeights::new();

    let loader = WeightLoader::new(&checkpoint, &store, None);
    let generator = BigVgan::load(&config, loader.root().pp("decoder")).unwrap();
    assert_eq!(generator.samples(16), 32);
    let odd = BigVganConfig {
        upsample_kernels: vec![4, 4],
        ..config.clone()
    };
    assert!(BigVgan::load(&odd, loader.root().pp("decoder")).is_err());

    let (waveform, launches) = recorded(|| generator.forward(&meta((2, 4, 16), DType::F32)));
    assert_eq!(waveform.unwrap().dims(), [2, 1, 32]);
    let count = |name: &str| {
        launches
            .iter()
            .filter(|(operator, _)| *operator == name)
            .count()
    };

    // Two dilation steps of two activations each, then the post activation, each once per
    // batch row; the block's four convolutions between the pre- and post-convolutions.
    assert_eq!(count("activation1d"), 2 * (2 * 2 + 1));
    // The pre-convolution, the block's four, the post-convolution and the upsampling one.
    assert_eq!(count("conv"), 1 + 4 + 1 + 1);
    assert_eq!(
        launches.first().map(|(operator, _)| *operator),
        Some("conv")
    );
    assert_eq!(
        launches
            .iter()
            .find(|(operator, _)| *operator == "activation1d")
            .unwrap()
            .1,
        "dtype_f32_channels_4_kernel_12_ratio_2_block_256_pdl_false"
    );
    let (refused, _) = recorded(|| generator.forward(&meta((2, 16), DType::F32)));
    assert!(refused.is_err());

    // On the CPU the generator is its modules in order: here one block, so the stage average
    // is the block's output.
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let generator = BigVgan::load(&config, loader.root().pp("decoder")).unwrap();
    let post = Activation1d::load(
        &Activation1dConfig {
            channels: 4,
            ratio: 2,
        },
        loader.root().pp("decoder").pp("activation_post"),
    )
    .unwrap();

    let latent = tensor(&sequence(4 * 5, 90), (1, 4, 5), DType::F32);
    let waveform = generator.forward(&latent).unwrap();
    assert_eq!(waveform.dims(), [1, 1, 10]);
    assert!(values(&waveform).iter().all(|sample| sample.is_finite()));

    // The post activation keeps the length of the signal it reads.
    let signal = tensor(&sequence(4 * 10, 91), (1, 4, 10), DType::F32);
    assert_eq!(post.forward(&signal).unwrap().dims(), [1, 4, 10]);
}

/// A DAC encoder loads `encoder.block`'s numbered modules, runs three Snake residual units and
/// a widening convolution per stage, and refuses a track that is not a whole number of hops.
#[test]
fn a_dac_encoder_widens_and_strides_through_its_stages() {
    let mut tensors = normalized_conv("enc.0", [2, 1, 7], Some(2));
    for unit in 0..3 {
        let base = format!("enc.1.block.{unit}.block");
        tensors.extend(normalized_conv(&format!("{base}.1"), [2, 2, 7], Some(2)));
        tensors.extend(normalized_conv(&format!("{base}.3"), [2, 2, 1], Some(2)));
        tensors.push((format!("{base}.0.alpha"), vec![1, 2, 1]));
        tensors.push((format!("{base}.2.alpha"), vec![1, 2, 1]));
    }
    tensors.push(("enc.1.block.3.alpha".to_owned(), vec![1, 2, 1]));
    tensors.extend(normalized_conv("enc.1.block.4", [4, 2, 4], Some(4)));
    tensors.push(("enc.2.alpha".to_owned(), vec![1, 4, 1]));
    tensors.extend(normalized_conv("enc.3", [8, 4, 3], Some(8)));

    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path(), &tensors);
    let config = DacEncoderConfig {
        channels: 2,
        strides: vec![2],
        latent_channels: 8,
    };

    let store = MetaWeights::new();

    let loader = WeightLoader::new(&checkpoint, &store, None);
    let encoder = DacEncoder::load(&config, loader.root().pp("enc")).unwrap();
    assert_eq!(encoder.hop(), 2);
    let no_stage = DacEncoderConfig {
        strides: Vec::new(),
        ..config.clone()
    };
    assert!(DacEncoder::load(&no_stage, loader.root().pp("enc")).is_err());

    let (latent, launches) = recorded(|| encoder.forward(&meta((2, 1, 16), DType::F32)));
    assert_eq!(latent.unwrap().dims(), [2, 8, 8]);
    let count = |name: &str| {
        launches
            .iter()
            .filter(|(operator, _)| *operator == name)
            .count()
    };
    assert_eq!(count("snake"), 3 * 2 + 1 + 1);
    assert_eq!(count("conv"), 1 + 3 * 2 + 1 + 1);
    assert_eq!(count("elementwise_binary"), 3, "one residual add per unit");
    let (refused, _) = recorded(|| encoder.forward(&meta((2, 1, 15), DType::F32)));
    assert!(refused.is_err());

    // On the CPU a track of whole hops becomes one frame per hop.
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let encoder = DacEncoder::load(&config, loader.root().pp("enc")).unwrap();
    let track = tensor(&sequence(12, 92), (1, 1, 12), DType::F32);
    let latent = encoder.forward(&track).unwrap();
    assert_eq!(latent.dims(), [1, 8, 6]);
    assert!(values(&latent).iter().all(|value| value.is_finite()));
}
