//! The solver formulas, the timestep modules, the noise draw and the adaptive pool: their
//! arithmetic on CPU tensors against each formula written out in loops, and the kernels they
//! state under a meta context.
//!
//! Nothing here touches a GPU.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

mod support;

use nn::modules::{TimestepEmbedding, TimestepEmbeddingConfig, TimestepTable, TimestepTableConfig};
use nn::{
    Checkpoint, CpuWeights, DType, Device, MetaWeights, Step, Tensor, WeightLoader, diffusion,
    functional,
};
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

// ---- Solver formulas --------------------------------------------------------------------------

const STEP: Step = Step {
    index: 1,
    count: 4,
    sigma: 0.75,
    sigma_next: 0.4375,
};

#[test]
fn the_solver_formulas_form_their_sums_in_fp32_and_round_once() {
    let (sample, other) = (sequence(2 * 3 * 4, 1), sequence(2 * 3 * 4, 2));
    let latents = |values: &[f32], dtype| tensor(values, (2, 3, 4), dtype);

    // `x + v · (sigma − sigma_next)`.
    let advanced = diffusion::euler_step(
        &latents(&sample, DType::F32),
        &latents(&other, DType::F32),
        STEP,
    )
    .unwrap();
    let expected: Vec<f32> = sample
        .iter()
        .zip(&other)
        .map(|(x, v)| x + v * (0.75 - 0.4375))
        .collect();
    assert_eq!(advanced.dims(), [2, 3, 4]);
    assert_eq!(values(&advanced), expected);

    // In a narrower type the operands are the values they hold and the sum rounds once.
    let (narrow_sample, narrow_velocity) =
        (latents(&sample, DType::BF16), latents(&other, DType::BF16));
    let advanced = diffusion::euler_step(&narrow_sample, &narrow_velocity, STEP).unwrap();
    let expected: Vec<f32> = values(&narrow_sample)
        .iter()
        .zip(values(&narrow_velocity))
        .map(|(x, v)| bf16(x + v * (0.75 - 0.4375)))
        .collect();
    assert_eq!(advanced.dtype(), DType::BF16);
    assert_eq!(values(&advanced), expected);

    // `x + x · (sigma − 1)`, which is not `x · sigma` in the last bit.
    let scaled = diffusion::scale_noise(&latents(&sample, DType::F32), 0.999).unwrap();
    let expected: Vec<f32> = sample.iter().map(|x| x + x * (0.999 - 1.0)).collect();
    assert_eq!(values(&scaled), expected);

    // `t · x + (1 − t) · noise`.
    let mixed = diffusion::add_noise(
        &latents(&sample, DType::F32),
        &latents(&other, DType::F32),
        0.875,
    )
    .unwrap();
    let expected: Vec<f32> = sample
        .iter()
        .zip(&other)
        .map(|(x, noise)| 0.875 * x + (1.0 - 0.875) * noise)
        .collect();
    assert_eq!(values(&mixed), expected);

    // A velocity or a noise of another type or extent covers no sample.
    let sample = latents(&sample, DType::F32);
    let short = tensor(&other[..12], (1, 3, 4), DType::F32);
    assert!(diffusion::euler_step(&sample, &narrow_velocity, STEP).is_err());
    assert!(diffusion::euler_step(&sample, &short, STEP).is_err());
    assert!(diffusion::add_noise(&sample, &short, 0.5).is_err());
}

/// A step's coefficient reaches the device as a one-element tensor, so the formula is tensor
/// arithmetic and no step-specific kernel.
#[test]
fn a_solver_step_on_a_device_is_generic_arithmetic_over_a_host_value() {
    let (advanced, launches) = recorded(|| {
        let latents = || meta((16, 3, 4, 4), DType::F32);
        diffusion::euler_step(&latents(), &latents(), STEP)
    });
    assert_eq!(advanced.unwrap().dims(), [16, 3, 4, 4]);
    assert_eq!(
        operators(&launches),
        ["elementwise_binary", "elementwise_binary"]
    );
}

// ---- Timestep conditioning --------------------------------------------------------------------

/// The row is every cosine, then every sine of the same angles: a permuted row is a different
/// conditioning that every downstream check accepts.
#[test]
fn the_embedding_is_every_cosine_then_every_sine() {
    let (dim, times) = (8, [0.25_f32, 0.999]);
    let embedding = diffusion::sinusoidal(&tensor(&times, 2, DType::F32), dim).unwrap();
    assert_eq!(embedding.dims(), [2, 8]);

    let half = dim as usize / 2;
    let rows = values(&embedding);
    for (row, time) in times.iter().enumerate() {
        for index in 0..half {
            // The references' FP32 frequencies: `exp(−ln(10000) · i / half)`, then `t · f`.
            let frequency = (-10_000_f32.ln() * index as f32 / half as f32).exp();
            let angle = time * frequency;
            assert_eq!(rows[row * 8 + index], angle.cos(), "element {index}");
            assert_eq!(
                rows[row * 8 + half + index],
                angle.sin(),
                "element {}",
                half + index
            );
        }
    }

    let none = Tensor::zeros(0, DType::F32, &Device::Cpu).unwrap();
    assert!(diffusion::sinusoidal(&none, dim).is_err());
    assert!(diffusion::sinusoidal(&tensor(&times, 2, DType::F32), 7).is_err());
}

#[test]
fn a_timestep_embedding_projects_activates_and_projects() {
    let directory = tempfile::tempdir().unwrap();
    let (first, first_bias) = (sequence(6 * 16, 3), sequence(6, 4));
    let (second, second_bias) = (sequence(5 * 6, 5), sequence(5, 6));
    let checkpoint = checkpoint(
        directory.path(),
        &[
            ("time.linear_1.weight", vec![6, 16], first.clone()),
            ("time.linear_1.bias", vec![6], first_bias.clone()),
            ("time.linear_2.weight", vec![5, 6], second.clone()),
            ("time.linear_2.bias", vec![5], second_bias.clone()),
        ],
    );
    let config = TimestepEmbeddingConfig {
        frequency_dim: 16,
        hidden: 6,
        out: 5,
    };

    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let embedding = TimestepEmbedding::load(&config, loader.root().pp("time")).unwrap();
    assert_eq!(embedding.dtype(), DType::F32);

    let x = sequence(2 * 16, 7);
    let out = embedding.forward(&tensor(&x, (2, 16), DType::F32)).unwrap();

    // `linear_2(silu(linear_1(x)))`, in loops.
    let project = |x: &[f32], weight: &[f32], bias: &[f32], k: usize| -> Vec<f32> {
        x.chunks(k)
            .flat_map(|row| {
                weight.chunks(k).zip(bias).map(|(column, bias)| {
                    row.iter().zip(column).map(|(x, w)| x * w).sum::<f32>() + bias
                })
            })
            .collect()
    };
    let hidden: Vec<f32> = project(&x, &first, &first_bias, 16)
        .into_iter()
        .map(|value| value / (1.0 + (-value).exp()))
        .collect();
    assert_close(
        &values(&out),
        &project(&hidden, &second, &second_bias, 6),
        F32_TOLERANCE,
    );

    let store = MetaWeights::new();

    let loader = WeightLoader::new(&checkpoint, &store, None);
    let embedding = TimestepEmbedding::load(&config, loader.root().pp("time")).unwrap();
    let (out, launches) = recorded(|| embedding.forward(&meta((2, 16), DType::F32)));
    assert_eq!(out.unwrap().dims(), [2, 5]);
    assert_eq!(operators(&launches), ["gemm", "elementwise_unary", "gemm"]);
}

const TABLE: TimestepTableConfig = TimestepTableConfig {
    frequency_dim: 16,
    layers: 2,
    layer_columns: 12,
    output_columns: 4,
    dtype: DType::F32,
};

/// Three steps of two groups; the last step repeats the first's levels.
const LEVELS: [f32; 6] = [1.0, 0.75, 0.5, 0.25, 1.0, 0.75];

#[test]
fn a_timestep_table_holds_each_distinct_level_once() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path(), &[("placeholder", vec![1], vec![0.0])]);
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let levels = tensor(&LEVELS, (3, 2), DType::F32);

    let table = TimestepTable::load(&TABLE, &levels, loader.root().pp("adaln")).unwrap();
    assert_eq!((table.groups(), table.steps()), (2, 3));
    assert_eq!(table.layer_table(1).unwrap().dims(), [4, 12]);
    assert_eq!(table.output_table().dims(), [4, 4]);
    assert!(table.layer_table(2).is_err(), "two blocks");

    // One embedding row per distinct level, in the order the schedule first reaches it.
    let distinct = tensor(&[1.0, 0.75, 0.5, 0.25], 4, DType::F32);
    assert_eq!(
        values(table.frequencies()),
        values(&diffusion::sinusoidal(&distinct, 16).unwrap())
    );

    // The model writes the tables once, and a step gathers the rows of its groups' entries.
    let rows = sequence(4 * 12, 8);
    table
        .layer_table(1)
        .unwrap()
        .slice_set(&tensor(&rows, (4, 12), DType::F32), 0, 0)
        .unwrap();
    let outputs = sequence(4 * 4, 9);
    table
        .output_table()
        .slice_set(&tensor(&outputs, (4, 4), DType::F32), 0, 0)
        .unwrap();

    for (step, entries) in [(0, [0_usize, 1]), (1, [2, 3]), (2, [0, 1])] {
        let gathered = table.entries_of(step).unwrap();
        assert_eq!(gathered.dims(), [2]);

        let expected: Vec<f32> = entries
            .iter()
            .flat_map(|entry| rows[entry * 12..][..12].to_vec())
            .collect();
        assert_eq!(values(&table.layer(1, &gathered).unwrap()), expected);

        let expected: Vec<f32> = entries
            .iter()
            .flat_map(|entry| outputs[entry * 4..][..4].to_vec())
            .collect();
        assert_eq!(values(&table.output(&gathered).unwrap()), expected);
    }
    assert!(table.entries_of(3).is_err(), "three steps");

    // Levels that are not steps of groups hold no table.
    let flat = tensor(&LEVELS, 6, DType::F32);
    assert!(TimestepTable::load(&TABLE, &flat, loader.root().pp("flat")).is_err());
}

#[test]
fn a_step_gathers_its_entries_and_its_rows_on_the_device() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path(), &[("placeholder", vec![1], vec![0.0])]);
    let store = MetaWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let levels = tensor(&LEVELS, (3, 2), DType::F32);
    let table = TimestepTable::load(&TABLE, &levels, loader.root().pp("adaln")).unwrap();

    let (rows, launches) = recorded(|| {
        let entries = table.entries_of(2)?;
        Ok((
            table.layer(1, &entries)?.dims().to_vec(),
            table.output(&entries)?.dims().to_vec(),
        ))
    });
    assert_eq!(rows.unwrap(), (vec![2, 12], vec![2, 4]));
    assert_eq!(operators(&launches), ["index", "index", "index"]);
}

// ---- Noise ------------------------------------------------------------------------------------

/// Philox4x32-10 with every word in a 64-bit register masked after each step, as the kernel
/// holds them.
fn philox(counter: [u64; 4], key: [u64; 2]) -> [u64; 4] {
    const MASK: u64 = 0xFFFF_FFFF;
    let [mut c0, mut c1, mut c2, mut c3] = counter;
    let [mut k0, mut k1] = key;

    for round in 0..10 {
        if round > 0 {
            k0 = (k0 + 0x9E37_79B9) & MASK;
            k1 = (k1 + 0xBB67_AE85) & MASK;
        }
        let (first, second) = (0xD251_1F53 * c0, 0xCD9E_8D57 * c2);
        (c0, c1, c2, c3) = (
            ((second >> 32) ^ c1 ^ k0) & MASK,
            second & MASK,
            ((first >> 32) ^ c3 ^ k1) & MASK,
            first & MASK,
        );
    }
    [c0, c1, c2, c3]
}

/// Element `element` of the normal stream of `key`: one half of a Philox word pair.
fn normal(key: u64, stage: u32, draw: u32, element: u64) -> f32 {
    let pair = element / 2;
    let words = philox(
        [
            pair & 0xFFFF_FFFF,
            pair >> 32,
            u64::from(stage),
            u64::from(draw),
        ],
        [key & 0xFFFF_FFFF, key >> 32],
    );

    let uniform = |word: u64| (word >> 8) as f32 / 16_777_216.0;
    let radius = (-2.0 * (1.0 - uniform(words[0])).ln()).sqrt();
    let angle = std::f32::consts::TAU * uniform(words[1]);
    if element.is_multiple_of(2) {
        radius * angle.cos()
    } else {
        radius * angle.sin()
    }
}

#[test]
fn the_philox_bijection_matches_the_reference_known_vectors() {
    // Random123's known-response vectors of Philox4x32-10.
    assert_eq!(
        philox([0; 4], [0; 2]),
        [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
    );
    assert_eq!(
        philox([0xffff_ffff; 4], [0xffff_ffff; 2]),
        [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
    );
}

#[test]
fn a_draw_is_a_function_of_the_seed_and_the_element_alone() {
    const SEED: u64 = 0x1234_5678_9abc_def0;
    let zeros = |shape: &[usize]| Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap();

    // A raster of any rank is drawn flattened.
    let raster = zeros(&[2, 3, 5]);
    functional::philox_normal(&raster, SEED, 3, 1, 0).unwrap();
    let expected: Vec<f32> = (0..30).map(|element| normal(SEED, 3, 1, element)).collect();
    assert_eq!(values(&raster), expected);

    // Two draws at adjacent offsets are one longer draw, at an odd offset too.
    let (head, tail) = (zeros(&[7]), zeros(&[23]));
    functional::philox_normal(&head, SEED, 3, 1, 0).unwrap();
    functional::philox_normal(&tail, SEED, 3, 1, 7).unwrap();
    assert_eq!([values(&head), values(&tail)].concat(), expected);

    // Another stage or draw is another stream.
    let other = zeros(&[30]);
    functional::philox_normal(&other, SEED, 3, 2, 0).unwrap();
    assert_ne!(values(&other), expected);

    // A draw returned as a new tensor is the same stream, in the shape and type of its model.
    let like = Tensor::zeros((2, 3, 5), DType::BF16, &Device::Cpu).unwrap();
    let drawn = functional::philox_normal_like(&like, SEED, 3, 1, 0).unwrap();
    assert_eq!((drawn.dims(), drawn.dtype()), (like.dims(), DType::BF16));
    assert_eq!(
        values(&drawn),
        expected.iter().copied().map(bf16).collect::<Vec<f32>>()
    );

    // One key per row draws one stream per row, each from its own first element.
    let keys = [SEED as i64, -5];
    let rows = zeros(&[2, 9]);
    functional::philox_normal_rows(
        &rows,
        &Tensor::from_slice(&keys, 2, &Device::Cpu).unwrap(),
        3,
        1,
        4,
    )
    .unwrap();
    let expected: Vec<f32> = keys
        .iter()
        .flat_map(|key| (4..13).map(|element| normal(*key as u64, 3, 1, element)))
        .collect();
    assert_eq!(values(&rows), expected);

    // A narrower buffer holds the FP32 draw rounded once.
    let narrow = Tensor::zeros(8, DType::BF16, &Device::Cpu).unwrap();
    functional::philox_normal(&narrow, SEED, 0, 0, 0).unwrap();
    let expected: Vec<f32> = (0..8)
        .map(|element| bf16(normal(SEED, 0, 0, element)))
        .collect();
    assert_eq!(values(&narrow), expected);
}

#[test]
fn a_draw_states_its_distribution_and_its_key_source() {
    let (drawn, launches) = recorded(|| {
        // A latent arrives as a four-axis raster and is drawn through its folded rows.
        functional::philox_normal(&meta((24, 37, 48, 84), DType::F32), 7, 0, 0, 0)?;
        functional::philox_normal_like(&meta((24, 2, 3, 4), DType::F32), 7, 0, 0, 0)?;
        functional::philox_normal_rows(&meta((2, 207), DType::BF16), &meta(2, DType::I64), 0, 0, 0)
    });
    drawn.unwrap();
    assert_eq!(
        launches,
        [
            (
                "philox_noise",
                "dtype_f32_block_1024_kind_normal_source_scalar"
            ),
            (
                "philox_noise",
                "dtype_f32_block_1024_kind_normal_source_scalar"
            ),
            ("philox_noise", "dtype_bf16_block_1024_kind_normal"),
        ]
        .map(|(operator, key)| (operator, key.to_owned()))
    );

    let (refused, _) = recorded(|| {
        functional::philox_normal_rows(&meta((2, 207), DType::BF16), &meta(3, DType::I64), 0, 0, 0)
    });
    assert!(refused.is_err(), "three keys for two rows");
}

// ---- Adaptive pool ----------------------------------------------------------------------------

#[test]
fn an_adaptive_pool_averages_every_row_over_each_column_window() {
    // 24 columns pooled to 7 give windows of four and of five; 4 pooled to 6 repeat a column.
    for (rows, width, output) in [(3, 24, 7), (2, 4, 6), (8, 16, 4)] {
        let x = sequence(2 * rows * width, 10);
        let pooled = functional::adaptive_avg_pool2d(
            &tensor(&x, (2, rows, width), DType::F32),
            [1, output as u32],
        )
        .unwrap();
        assert_eq!(pooled.dims(), [2, 1, output]);

        let mut expected = Vec::new();
        for batch in 0..2 {
            for index in 0..output {
                let start = index * width / output;
                let end = ((index + 1) * width).div_ceil(output).max(start + 1);
                let sum: f32 = (0..rows)
                    .flat_map(|row| (start..end).map(move |column| (row, column)))
                    .map(|(row, column)| x[(batch * rows + row) * width + column])
                    .sum();
                expected.push(sum / ((end - start) * rows) as f32);
            }
        }
        assert_close(&values(&pooled), &expected, F32_TOLERANCE);
    }

    // The pool reduces every row to one row of columns, of a three-axis input.
    let x = tensor(&sequence(5 * 8 * 16, 11), (5, 8, 16), DType::F32);
    assert!(functional::adaptive_avg_pool2d(&x, [2, 32]).is_err());
    assert!(functional::adaptive_avg_pool2d(&x, [1, 0]).is_err());
    assert!(functional::adaptive_avg_pool2d(&x.flatten(0, 1).unwrap(), [1, 4]).is_err());

    let (pooled, launches) =
        recorded(|| functional::adaptive_avg_pool2d(&meta((5, 8, 256), DType::F32), [1, 32]));
    assert_eq!(pooled.unwrap().dims(), [5, 1, 32]);
    assert_eq!(
        launches,
        [(
            "adaptive_pool",
            "dtype_f32_heads_8_width_256_output_32_block_1024_pdl_false".to_owned()
        )]
    );
}
