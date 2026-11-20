//! Rotary position embedding: the spectra a configuration states, the functions' arithmetic on
//! CPU tensors against the rotation written out in loops, and the kernels the functions and the
//! `Rotary` module state under a meta context.
//!
//! Nothing here touches a GPU.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use std::path::Path;

use nn::config::Fields;
use nn::functional;
use nn::modules::{Precision, Rotary, RotaryConfig, Spectrum, Yarn};
use nn::{Checkpoint, CpuWeights, DType, Device, MetaWeights, Tensor, WeightLoader};
use nn_testing::{
    Recording, assert_close, encode, meta, record, sequence, tensor, values, write_checkpoint,
};

/// The tolerance of an FP32 result against a reference accumulated in another order.
const F32_TOLERANCE: f32 = 2e-5;

/// A configuration of `spectrum` over `dim` rotated channels of base `theta`.
fn config(theta: f32, dim: u32, spectrum: Spectrum, precision: Precision) -> RotaryConfig {
    RotaryConfig {
        theta,
        dim,
        spectrum,
        precision,
    }
}

/// The plain f64 spectrum over `dim` channels of base one million.
fn plain(dim: u32) -> RotaryConfig {
    config(1_000_000.0, dim, Spectrum::Default, Precision::F64)
}

/// A checkpoint of one placeholder tensor: a rotary reads no tensor, it publishes its table.
fn checkpoint(directory: &Path) -> Checkpoint {
    write_checkpoint(
        directory,
        &[&[(
            "placeholder",
            DType::F32,
            vec![1],
            encode(DType::F32, &[0.0]),
        )]],
    );
    Checkpoint::open(directory).unwrap()
}

// ---- Spectra ----------------------------------------------------------------------------------

/// A plain spectrum is the f64 power series, its first pair unity and the rest falling.
#[test]
fn a_default_spectrum_is_the_power_series_of_its_base() {
    let table = plain(128).inverse_frequencies().unwrap();

    assert_eq!(table.len(), 64);
    assert_eq!(table[0], 1.0);
    assert_eq!(table[1], (1.0 / 1_000_000_f64.powf(2.0 / 128.0)) as f32);
    assert!(table.windows(2).all(|pair| pair[0] > pair[1]));

    // A width with no whole pair and a base outside its domain have no table.
    assert!(plain(127).inverse_frequencies().is_err());
    assert!(plain(0).inverse_frequencies().is_err());
    let negative = config(-1.0, 128, Spectrum::Default, Precision::F64);
    assert!(negative.inverse_frequencies().is_err());
}

/// The f32 rule is evaluated in f32: its table is the f32 power, which differs from the f64
/// rule's in the last bits, and the model chooses which one by name.
#[test]
fn the_f32_rule_is_evaluated_in_f32() {
    let spectrum = |precision| config(5_000_000.0, 128, Spectrum::Default, precision);

    let narrow = spectrum(Precision::F32).inverse_frequencies().unwrap();
    for (index, value) in narrow.iter().enumerate() {
        assert_eq!(
            *value,
            1.0 / 5_000_000_f32.powf((2 * index) as f32 / 128.0),
            "pair {index}"
        );
    }
    assert_ne!(
        narrow,
        spectrum(Precision::F64).inverse_frequencies().unwrap()
    );
}

/// A repeated spectrum gives every axis the same frequencies, each axis in its own block.
#[test]
fn a_repeated_spectrum_gives_every_axis_one_block_of_the_same_frequencies() {
    let repeated = config(10_000.0, 96, Spectrum::Repeated { axes: 3 }, Precision::F64);
    let table = repeated.inverse_frequencies().unwrap();

    assert_eq!(table.len(), 48);
    for axis in 1..3 {
        assert_eq!(table[axis * 16..(axis + 1) * 16], table[..16]);
    }

    // The per-axis series is over 32 channels, not the 96 the axes rotate together.
    assert_eq!(table[1], (1.0 / 10_000_f64.powf(2.0 / 32.0)) as f32);

    // The axes take equal shares of the rotated channels.
    let uneven = config(
        10_000.0,
        100,
        Spectrum::Repeated { axes: 3 },
        Precision::F64,
    );
    assert!(uneven.inverse_frequencies().is_err());
}

/// M-RoPE sections count frequencies and cover the spectrum exactly.
#[test]
fn sections_cover_the_spectrum_exactly() {
    let sectioned = |widths: &[u32]| {
        config(
            5_000_000.0,
            128,
            Spectrum::sectioned(widths).unwrap(),
            Precision::F32,
        )
    };

    let table = sectioned(&[24, 20, 20]).inverse_frequencies().unwrap();
    assert_eq!(table.len(), 64);
    assert!(sectioned(&[24, 20, 19]).inverse_frequencies().is_err());

    assert!(Spectrum::sectioned(&[]).is_err());
    assert!(Spectrum::sectioned(&[0, 32, 32]).is_err());
    assert!(Spectrum::sectioned(&[8, 8, 8, 8, 32]).is_err());
}

/// YaRN keeps the fastest pair, divides the slowest by its factor, and squares its attention
/// factor into the softmax scale.
#[test]
fn yarn_interpolates_the_slow_pairs_and_scales_the_scores() {
    let yarn = Yarn {
        factor: 10.0,
        original_max_position_embeddings: 4096,
        attention_factor: None,
        beta_fast: 32.0,
        beta_slow: 1.0,
        mscale: None,
        mscale_all_dim: None,
    };
    let scaled = config(1_000_000.0, 128, Spectrum::Yarn(yarn), Precision::F64);
    let (plain_table, table) = (
        plain(128).inverse_frequencies().unwrap(),
        scaled.inverse_frequencies().unwrap(),
    );

    assert_eq!(table[0], plain_table[0]);
    assert!((table[63] - plain_table[63] / 10.0).abs() < 1e-9);
    assert!(table.windows(2).all(|pair| pair[0] > pair[1]));

    let expected = (0.1 * 10.0_f32.ln() + 1.0).powi(2) / 128.0_f32.sqrt();
    assert!((scaled.score_scale(128).unwrap() - expected).abs() < 1e-6);

    // The recipe needs a base above one for its correction range.
    let flat = config(1.0, 128, Spectrum::Yarn(yarn), Precision::F64);
    assert!(flat.inverse_frequencies().is_err());
}

/// Under the f32 rule a YaRN table is the one DeepSeek's reference computes in f32
/// (`precompute_freqs_cis`): its bits, from the reference evaluated by PyTorch, for a base of
/// 160000 over 64 channels extended sixteen times from 65536 positions, β 32 and 1.
#[test]
fn a_yarn_table_under_the_f32_rule_is_the_references() {
    const REFERENCE: [u32; 32] = [
        0x3f80_0000,
        0x3f30_0a3a,
        0x3ef2_1c1f,
        0x3ea6_7d01,
        0x3e64_f92e,
        0x3e1d_7475,
        0x3dd8_8cb4,
        0x3d94_e963,
        0x3d4c_cccd,
        0x3d0c_d4fb,
        0x3cc1_b019,
        0x3c85_30ce,
        0x3c37_2dbf,
        0x3bfb_ed88,
        0x3bad_3d5e,
        0x3b6e_4237,
        0x3b14_7ae1,
        0x3ab7_14e0,
        0x3a5e_bdb6,
        0x3a05_30ce,
        0x399b_b3af,
        0x3930_5978,
        0x38be_904d,
        0x383e_9b5f,
        0x37a3_d70c,
        0x36b4_43d0,
        0x3677_eba6,
        0x362a_7be8,
        0x35ea_77ff,
        0x35a1_3bdc,
        0x355d_bf30,
        0x3518_7c4d,
    ];
    let yarn = Yarn {
        factor: 16.0,
        original_max_position_embeddings: 65_536,
        attention_factor: None,
        beta_fast: 32.0,
        beta_slow: 1.0,
        mscale: None,
        mscale_all_dim: None,
    };

    let table = config(160_000.0, 64, Spectrum::Yarn(yarn), Precision::F32)
        .inverse_frequencies()
        .unwrap();
    let bits: Vec<u32> = table.iter().map(|frequency| frequency.to_bits()).collect();
    assert_eq!(bits, REFERENCE);
}

/// The softmax scale follows the configuration's precision: the two rules differ at 72
/// channels.
#[test]
fn the_score_scale_is_computed_in_the_configurations_precision() {
    let spectrum = |precision| config(10_000.0, 36, Spectrum::Default, precision);
    let (wide, narrow) = (
        spectrum(Precision::F64).score_scale(72).unwrap(),
        spectrum(Precision::F32).score_scale(72).unwrap(),
    );

    assert_eq!(wide, (1.0 / 72_f64.sqrt()) as f32);
    assert_eq!(narrow, 1.0 / 72_f32.sqrt());
    assert_ne!(wide, narrow);
    assert!(spectrum(Precision::F64).score_scale(0).is_err());
}

/// A Hugging Face configuration states its recipe in one of transformers' two layouts, read
/// as one; a configuration that mixes them, omits its base, or names a recipe that is not served
/// is refused.
#[test]
fn a_hugging_face_recipe_is_read_in_its_layout() {
    let read = |value: serde_json::Value| {
        RotaryConfig::from_hf(
            &Fields::of("config.json", &value).unwrap(),
            128,
            40_960,
            Some(1.0),
            Precision::F64,
        )
    };

    // The earlier layout: the base beside the fields, the recipe in `rope_scaling`.
    assert_eq!(
        read(serde_json::json!({"rope_theta": 1_000_000.0, "rope_scaling": null})).unwrap(),
        plain(128)
    );
    assert_eq!(
        read(serde_json::json!({"rope_theta": 1_000_000.0, "partial_rotary_factor": 0.5})).unwrap(),
        plain(64)
    );

    // The standardized layout: one object, whose YaRN factor extends the fitted length to the
    // configuration's.
    let yarn = read(serde_json::json!({
        "rope_parameters": {"rope_type": "yarn", "rope_theta": 1_000_000.0, "factor": 10.0,
                            "original_max_position_embeddings": 4096},
    }))
    .unwrap();
    assert!(matches!(
        yarn.spectrum,
        Spectrum::Yarn(yarn) if yarn.factor == 10.0
            && yarn.original_max_position_embeddings == 4096
    ));
    assert_eq!(yarn.precision, Precision::F64);

    let sectioned = RotaryConfig::from_hf(
        &Fields::of(
            "config.json",
            &serde_json::json!({
                "rope_parameters": {"rope_type": "default", "rope_theta": 10_000_000.0,
                                    "mrope_section": [11, 11, 10], "mrope_interleaved": true,
                                    "partial_rotary_factor": 0.25},
                "partial_rotary_factor": 0.25,
            }),
        )
        .unwrap(),
        256,
        262_144,
        None,
        Precision::F32,
    )
    .unwrap();
    assert_eq!(sectioned.dim, 64);
    assert_eq!(
        sectioned.spectrum,
        Spectrum::sectioned(&[11, 11, 10]).unwrap()
    );

    for refused in [
        // A YaRN factor that does not extend the fitted length to the configuration's.
        serde_json::json!({"rope_parameters": {"rope_type": "yarn", "rope_theta": 1e6,
                                               "factor": 4.0,
                                               "original_max_position_embeddings": 4096}}),
        serde_json::json!({"rope_theta": 1e6, "rope_scaling": {"rope_type": "linear",
                                                               "factor": 2.0}}),
        // Both layouts at once.
        serde_json::json!({"rope_theta": 1e6, "rope_parameters": {"rope_type": "default",
                                                                  "rope_theta": 1e6}}),
        serde_json::json!({"rope_scaling": {"rope_type": "default"},
                           "rope_parameters": {"rope_type": "default", "rope_theta": 1e6}}),
        // Two rotated fractions, and no base.
        serde_json::json!({"partial_rotary_factor": 0.5,
                           "rope_parameters": {"rope_type": "default", "rope_theta": 1e6,
                                               "partial_rotary_factor": 0.25}}),
        serde_json::json!({}),
    ] {
        assert!(read(refused.clone()).is_err(), "{refused}");
    }
}

// ---- Factors ----------------------------------------------------------------------------------

/// The cosine and the sine of `positions[axis_of[j]][t] · frequencies[j]`, `[tokens][columns]`
/// flat each.
fn factors(positions: &[Vec<f32>], frequencies: &[f32], axis_of: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let tokens = positions[0].len();
    let angle =
        |token: usize, column: usize| positions[axis_of[column]][token] * frequencies[column];
    let table = |factor: fn(f32) -> f32| -> Vec<f32> {
        (0..tokens)
            .flat_map(|token| {
                (0..frequencies.len()).map(move |column| factor(angle(token, column)))
            })
            .collect()
    };

    (table(f32::cos), table(f32::sin))
}

#[test]
fn rotary_factors_are_the_cosine_and_sine_of_position_times_frequency() {
    let frequencies = plain(16).inverse_frequencies().unwrap();
    let inv_freq = tensor(&frequencies, 8, DType::F32);
    let positions = Tensor::new(&[0_i32, 1, 7, 300, 40_959], &Device::Cpu).unwrap();

    let (cos, sin) = functional::rotary_factors(&positions, &inv_freq, &[], false).unwrap();
    assert_eq!((cos.dims(), sin.dims()), (&[5, 8][..], &[5, 8][..]));
    assert_eq!((cos.dtype(), sin.dtype()), (DType::F32, DType::F32));

    let (expected_cos, expected_sin) = factors(
        &[vec![0.0, 1.0, 7.0, 300.0, 40_959.0]],
        &frequencies,
        &[0; 8],
    );
    assert_close(&values(&cos), &expected_cos, 1e-6);
    assert_close(&values(&sin), &expected_sin, 1e-6);
}

/// Each frequency reads the coordinate of the axis that owns it: contiguous blocks, or the
/// axes cycled over the spectrum with the frequencies a narrow axis cannot claim falling to
/// the first.
#[test]
fn sectioned_factors_read_each_frequencys_own_axis() {
    let frequencies = sequence(8, 1);
    let inv_freq = tensor(&frequencies, 8, DType::F32);
    let coordinates = vec![
        vec![3.0, 4.0, 5.0, 6.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 1.0, 0.0, 1.0],
    ];
    let flat: Vec<i32> = coordinates.iter().flatten().map(|x| *x as i32).collect();
    let positions = Tensor::new(flat.as_slice(), &Device::Cpu)
        .unwrap()
        .reshape((3, 4))
        .unwrap();

    // Four, two and two frequencies, as eight, four and four rotated channels.
    let sections = [8, 4, 4];
    let cases: [(bool, [usize; 8]); 2] = [
        (false, [0, 0, 0, 0, 1, 1, 2, 2]),
        // Cycled: 0 1 2 0 1 2 0 1, where the second axis's third claim (frequency 7) is past
        // its two frequencies' share of the cycle and falls to the first axis.
        (true, [0, 1, 2, 0, 1, 2, 0, 0]),
    ];
    for (interleaved, axis_of) in cases {
        let (cos, sin) =
            functional::rotary_factors(&positions, &inv_freq, &sections, interleaved).unwrap();
        let (expected_cos, expected_sin) = factors(&coordinates, &frequencies, &axis_of);

        assert_close(&values(&cos), &expected_cos, 1e-6);
        assert_close(&values(&sin), &expected_sin, 1e-6);
    }

    // Sections that do not cover the frequencies, and positions that are not axis-major.
    assert!(functional::rotary_factors(&positions, &inv_freq, &[8, 4, 2], false).is_err());
    let flat = positions.flatten_all().unwrap();
    let refused = functional::rotary_factors(&flat, &inv_freq, &sections, false);
    assert!(refused.unwrap_err().to_string().contains("per axis"));
}

#[test]
fn the_factor_function_states_its_axes_in_its_key() {
    let inv_freq = meta(48, DType::F32);

    // One axis: every position is one token's.
    let (single, Recording { launches, .. }) = record(&[], || {
        functional::rotary_factors(&meta(10, DType::I32), &meta(64, DType::F32), &[], false)
    });
    let (cos, sin) = single.unwrap();
    assert_eq!((cos.dims(), sin.dims()), (&[10, 64][..], &[10, 64][..]));
    assert_eq!(
        launches,
        [(
            "rotary_factors",
            "dtype_f32_positions_i32_columns_64_block_256_scale_0x3f800000_pdl_false".to_owned()
        )]
    );

    // Three repeated axes of sixteen pairs: one column per rotated pair, one coordinate per
    // axis and token.
    let (repeated, Recording { launches, .. }) = record(&[], || {
        functional::rotary_factors(&meta((3, 10), DType::I32), &inv_freq, &[32, 32, 32], false)
    });
    assert_eq!(repeated.unwrap().0.dims(), [10, 48]);
    assert_eq!(
        launches[0].1,
        "dtype_f32_positions_i32_columns_48_block_256_scale_0x3f800000_pdl_false_axes_3_sections_32-32-32"
    );

    // The interleaved cut is another kernel.
    let (cycled, Recording { launches, .. }) = record(&[], || {
        functional::rotary_factors(&meta((3, 10), DType::I32), &inv_freq, &[48, 24, 24], true)
    });
    cycled.unwrap();
    assert!(
        launches[0]
            .1
            .ends_with("_axes_3_sections_48-24-24_interleaved_true")
    );
}

// ---- The rotation -----------------------------------------------------------------------------

/// One head of `dim` channels turned by `pairs` pairs of factors, the half-split pairing; the
/// channels past the pairs pass through.
fn rotate(head: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
    let pairs = cos.len();
    let mut turned = head.to_vec();

    for pair in 0..pairs {
        turned[pair] = head[pair] * cos[pair] - head[pair + pairs] * sin[pair];
        turned[pair + pairs] = head[pair + pairs] * cos[pair] + head[pair] * sin[pair];
    }
    turned
}

/// One head scaled by the reciprocal root of its mean square plus `eps`, and by `gain`.
fn normalize(head: &[f32], gain: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mean_square = head.iter().map(|x| x * x).sum::<f32>() / head.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();

    head.iter()
        .enumerate()
        .map(|(channel, x)| x * scale * gain.map_or(1.0, |gain| gain[channel]))
        .collect()
}

/// Every head of `x` (`[tokens][heads][dim]`, flat) through `turn`, with its token's factor
/// rows of `pairs` columns.
fn per_head(
    x: &[f32],
    (heads, dim, pairs): (usize, usize, usize),
    (cos, sin): (&[f32], &[f32]),
    turn: impl Fn(&[f32], &[f32], &[f32]) -> Vec<f32>,
) -> Vec<f32> {
    x.chunks(dim)
        .enumerate()
        .flat_map(|(index, head)| {
            let token = index / heads;
            turn(
                head,
                &cos[token * pairs..][..pairs],
                &sin[token * pairs..][..pairs],
            )
        })
        .collect()
}

#[test]
fn rotary_turns_the_leading_pairs_and_passes_the_rest_through() {
    // Heads of eight channels, of which three pairs rotate.
    let (tokens, heads, dim, pairs) = (3, 2, 8, 3);
    let x = sequence(tokens * heads * dim, 1);
    let (cos, sin) = (sequence(tokens * pairs, 2), sequence(tokens * pairs, 3));

    let turned = functional::rotary(
        &tensor(&x, (tokens, heads, dim), DType::F32),
        &tensor(&cos, (tokens, pairs), DType::F32),
        &tensor(&sin, (tokens, pairs), DType::F32),
    )
    .unwrap();
    assert_eq!(turned.dims(), [tokens, heads, dim]);

    let expected = per_head(&x, (heads, dim, pairs), (&cos, &sin), rotate);
    assert_close(&values(&turned), &expected, F32_TOLERANCE);

    // The two trailing channels of every head are its input's.
    for (turned, x) in values(&turned).chunks(dim).zip(x.chunks(dim)) {
        assert_eq!(turned[6..], x[6..]);
    }

    // A rotation wider than the head, and tables that are not one row per token.
    let wide = tensor(&sequence(tokens * 5, 4), (tokens, 5), DType::F32);
    let input = tensor(&x, (tokens, heads, dim), DType::F32);
    assert!(functional::rotary(&input, &wide, &wide).is_err());

    let short = tensor(&cos[..2 * pairs], (2, pairs), DType::F32);
    assert!(functional::rotary(&input, &short, &short).is_err());
}

#[test]
fn qk_norm_rope_normalizes_each_head_then_turns_it() {
    let (tokens, heads, kv_heads, dim) = (2, 4, 2, 8);
    let pairs = dim / 2;
    let (query, key) = (
        sequence(tokens * heads * dim, 1),
        sequence(tokens * kv_heads * dim, 2),
    );
    let (query_gain, key_gain) = (sequence(dim, 3), sequence(dim, 4));
    let (cos, sin) = (sequence(tokens * pairs, 5), sequence(tokens * pairs, 6));
    let eps = 1e-6;

    let tables = (
        tensor(&cos, (tokens, pairs), DType::F32),
        tensor(&sin, (tokens, pairs), DType::F32),
    );
    let inputs = (
        tensor(&query, (tokens, heads, dim), DType::F32),
        tensor(&key, (tokens, kv_heads, dim), DType::F32),
    );
    let gains = (
        tensor(&query_gain, dim, DType::F32),
        tensor(&key_gain, dim, DType::F32),
    );
    let fused = |weights: Option<(&Tensor, &Tensor)>, centred| {
        functional::qk_norm_rope(
            &inputs.0, &inputs.1, weights, &tables.0, &tables.1, eps, centred,
        )
        .unwrap()
    };
    let reference = |x: &[f32], heads: usize, gain: Option<&[f32]>| {
        per_head(x, (heads, dim, pairs), (&cos, &sin), |head, cos, sin| {
            rotate(&normalize(head, gain, eps), cos, sin)
        })
    };

    // Learned gains scale the normalized head before it turns.
    let (turned_query, turned_key) = fused(Some((&gains.0, &gains.1)), false);
    assert_eq!(turned_query.dims(), [tokens, heads, dim]);
    assert_eq!(turned_key.dims(), [tokens, kv_heads, dim]);
    assert_close(
        &values(&turned_query),
        &reference(&query, heads, Some(&query_gain)),
        F32_TOLERANCE,
    );
    assert_close(
        &values(&turned_key),
        &reference(&key, kv_heads, Some(&key_gain)),
        F32_TOLERANCE,
    );

    // A zero-centred gain scales by one plus itself.
    let centred: Vec<f32> = query_gain.iter().map(|gain| 1.0 + gain).collect();
    let (turned_query, _) = fused(Some((&gains.0, &gains.1)), true);
    assert_close(
        &values(&turned_query),
        &reference(&query, heads, Some(&centred)),
        F32_TOLERANCE,
    );

    // Without gains the head is normalized and turned as it is.
    let (turned_query, turned_key) = fused(None, false);
    assert_close(
        &values(&turned_query),
        &reference(&query, heads, None),
        F32_TOLERANCE,
    );
    assert_close(
        &values(&turned_key),
        &reference(&key, kv_heads, None),
        F32_TOLERANCE,
    );

    // Gains of another width are refused.
    let narrow = tensor(&query_gain[..4], 4, DType::F32);
    let refused = functional::qk_norm_rope(
        &inputs.0,
        &inputs.1,
        Some((&narrow, &narrow)),
        &tables.0,
        &tables.1,
        eps,
        false,
    );
    assert!(refused.unwrap_err().to_string().contains("query gain"));
}

/// The in-place form turns the heads of two column ranges of a merged projection where they
/// lie, by a table the tokens repeat, and leaves every other column as it was.
#[test]
fn the_in_place_form_turns_two_ranges_of_a_merged_projection() {
    // Rows of [query | key | value], two heads of four channels each; four tokens repeat a
    // table of two rows.
    let (tokens, heads, dim, pairs) = (4, 2, 4, 2);
    let row = 3 * heads * dim;
    let merged = sequence(tokens * row, 1);
    let (cos, sin) = (sequence(2 * pairs, 2), sequence(2 * pairs, 3));
    let eps = 1e-5;

    let projected = tensor(&merged, (tokens, row), DType::F32);
    let turned = functional::qk_norm_rope_(
        projected.clone(),
        0..heads,
        heads..2 * heads,
        dim as u32,
        &tensor(&cos, (2, pairs), DType::F32),
        &tensor(&sin, (2, pairs), DType::F32),
        eps,
    )
    .unwrap();

    // The tensor returned is the projection itself, turned.
    assert_eq!(values(&turned), values(&projected));

    let repeated = |table: &[f32]| [table, table].concat();
    let (cos, sin) = (repeated(&cos), repeated(&sin));
    for (token, (turned, merged)) in values(&turned)
        .chunks(row)
        .zip(merged.chunks(row))
        .enumerate()
    {
        let factors = (
            &cos[token * pairs..][..pairs],
            &sin[token * pairs..][..pairs],
        );
        let expected: Vec<f32> = merged[..2 * heads * dim]
            .chunks(dim)
            .flat_map(|head| rotate(&normalize(head, None, eps), factors.0, factors.1))
            .collect();

        assert_close(&turned[..2 * heads * dim], &expected, F32_TOLERANCE);
        assert_eq!(turned[2 * heads * dim..], merged[2 * heads * dim..]);
    }

    // Ranges of two lengths, and a table the tokens do not repeat.
    let table = tensor(&cos[..3 * pairs], (3, pairs), DType::F32);
    let again = || tensor(&merged, (tokens, row), DType::F32);
    assert!(
        functional::qk_norm_rope_(again(), 0..2, 2..3, dim as u32, &table, &table, eps).is_err()
    );
    assert!(
        functional::qk_norm_rope_(again(), 0..2, 2..4, dim as u32, &table, &table, eps).is_err()
    );
}

#[test]
fn the_rotations_state_their_kernels() {
    let query = meta((5, 8, 128), DType::BF16);
    let key = meta((5, 2, 128), DType::BF16);
    let gain = meta(128, DType::BF16);
    let (cos, sin) = (meta((5, 64), DType::F32), meta((5, 64), DType::F32));

    // The fused form is one launch for the queries and the keys, and a normalization with
    // learned gains is another kernel than one without.
    let fused = |weights: Option<(&Tensor, &Tensor)>, centred| {
        let (turned, Recording { launches, .. }) = record(&[], || {
            functional::qk_norm_rope(&query, &key, weights, &cos, &sin, 1e-6, centred)
        });
        let (turned_query, turned_key) = turned.unwrap();
        assert_eq!(turned_query.dims(), query.dims());
        assert_eq!(turned_key.dims(), key.dims());
        launches
    };
    let base = "dtype_bf16_output_bf16_head_dim_128_q_heads_8_k_heads_2_rotation_half\
                _epsilon_0x358637bd_rows_16_block_128_pdl_false";

    assert_eq!(
        fused(Some((&gain, &gain)), false),
        [("qk_norm_rope", base.to_owned())]
    );
    assert_eq!(
        fused(Some((&gain, &gain)), true),
        [("qk_norm_rope", format!("{base}_centred_true"))]
    );
    assert_eq!(
        fused(None, true),
        [("qk_norm_rope", format!("{base}_affine_false"))]
    );

    // A partial rotation names the channels it turns.
    let narrow = meta((5, 48), DType::F32);
    let (partial, Recording { launches, .. }) = record(&[], || {
        functional::qk_norm_rope(&query, &key, None, &narrow, &narrow, 1e-6, false)
    });
    partial.unwrap();
    assert_eq!(launches[0].1, format!("{base}_rotated_96_affine_false"));

    // The rotation alone is one launch per tensor.
    let (turned, Recording { launches, .. }) = record(&[], || functional::rotary(&key, &cos, &sin));
    assert_eq!(turned.unwrap().dims(), key.dims());
    assert_eq!(
        launches,
        [(
            "rope_apply",
            "dtype_bf16_output_bf16_head_dim_128_heads_2_rotation_half_rows_8_block_128_pdl_false"
                .to_owned()
        )]
    );

    // The in-place form turns two ranges of one projection by a table of the caller's type.
    let table = meta((5, 64), DType::BF16);
    let (turned, Recording { launches, .. }) = record(&[], || {
        functional::qk_norm_rope_(
            meta((10, 24 * 128), DType::BF16),
            0..8,
            8..16,
            128,
            &table,
            &table,
            1e-6,
        )
    });
    assert_eq!(turned.unwrap().dims(), [10, 24 * 128]);
    assert_eq!(
        launches,
        [(
            "qk_norm_rope",
            "dtype_bf16_output_bf16_head_dim_128_q_heads_8_k_heads_8_rotation_half\
             _epsilon_0x358637bd_rows_8_block_128_pdl_false_affine_false_factors_bf16_strided_true"
                .to_owned()
        )]
    );
}

// ---- Grid positions ---------------------------------------------------------------------------

/// A grid's coordinates are each cell's centre along its axis, scaled, axis-major and in
/// raster order.
#[test]
fn grid_positions_are_scaled_cell_centres_in_raster_order() {
    let grid = functional::grid_positions(&[2, 3], 1.0).unwrap();
    assert_eq!(grid.dims(), [2, 6]);
    assert_eq!(grid.dtype(), DType::F32);

    // Two rows of three columns: the rows' coordinates, then the columns'.
    let table = values(&grid);
    assert_eq!(&table[..6], &[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]);

    let third = (0.5_f32 / 3.0) * 2.0 - 1.0;
    assert_eq!(table[6], third);
    assert_eq!(table[7], 0.0);
    assert_eq!(table[8], -third);
    assert_eq!(&table[9..], &table[6..9]);

    // The scale multiplies the FP32 coordinate after the centre is formed.
    let turned = functional::grid_positions(&[3], std::f32::consts::TAU).unwrap();
    assert_eq!(values(&turned)[0], std::f32::consts::TAU * third);

    assert!(functional::grid_positions(&[2, 0], 1.0).is_err());
    assert!(functional::grid_positions(&[], 1.0).is_err());
}

// ---- The module -------------------------------------------------------------------------------

#[test]
fn a_rotary_publishes_its_spectrum_and_computes_a_batchs_factors() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path());
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);

    // Interleaved M-RoPE over sixteen channels: four, two and two frequencies.
    let sectioned = config(
        10_000.0,
        16,
        Spectrum::sectioned(&[4, 2, 2]).unwrap(),
        Precision::F32,
    );
    let rotary = Rotary::load(&sectioned, loader.root().pp("rotary_emb")).unwrap();
    assert_eq!((rotary.columns(), rotary.axes()), (8, 3));
    assert_eq!(rotary.config(), &sectioned);

    let coordinates = vec![vec![5.0, 6.0], vec![0.0, 1.0], vec![2.0, 0.0]];
    let positions = Tensor::new(&[[5_i32, 6], [0, 1], [2, 0]], &Device::Cpu).unwrap();
    let (cos, sin) = rotary.forward(&positions).unwrap();

    let (expected_cos, expected_sin) = factors(
        &coordinates,
        &sectioned.inverse_frequencies().unwrap(),
        &[0, 1, 2, 0, 1, 2, 0, 0],
    );
    assert_close(&values(&cos), &expected_cos, 1e-6);
    assert_close(&values(&sin), &expected_sin, 1e-6);

    // A spectrum with no whole pair publishes nothing.
    assert!(Rotary::load(&plain(127), loader.root().pp("rotary_emb")).is_err());
}

/// A repeated spectrum reads one coordinate per axis, axis-major: a flat position list is
/// refused instead of read transposed.
#[test]
fn a_rotary_reads_one_coordinate_per_axis_into_one_column_per_pair() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path());
    let store = MetaWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);

    // Sixteen pairs repeated over three axes of a 128-channel head.
    let repeated = config(10_000.0, 96, Spectrum::Repeated { axes: 3 }, Precision::F64);
    let rotary = Rotary::load(&repeated, loader.root().pp("rotary_emb")).unwrap();
    assert_eq!((rotary.columns(), rotary.axes()), (48, 3));

    let (computed, Recording { launches, .. }) =
        record(&[], || rotary.forward(&meta((3, 10), DType::I32)));
    let (cos, sin) = computed.unwrap();
    assert_eq!(cos.dims(), [10, 48], "one column per rotated pair");
    assert_eq!((sin.dims(), sin.dtype()), (cos.dims(), DType::F32));
    assert_eq!(
        launches,
        [(
            "rotary_factors",
            "dtype_f32_positions_i32_columns_48_block_256_scale_0x3f800000_pdl_false_axes_3\
             _sections_32-32-32"
                .to_owned()
        )]
    );

    let (flat, Recording { launches, .. }) = record(&[], || rotary.forward(&meta(30, DType::I32)));
    assert!(
        flat.is_err(),
        "a sectioned rotation reads axis-major positions"
    );
    assert!(launches.is_empty());
}
