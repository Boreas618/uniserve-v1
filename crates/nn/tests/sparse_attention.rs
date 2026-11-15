//! Block-sparse attention, the VSA operators and the modules over them: their arithmetic on CPU
//! tensors against each definition written out in loops, and the kernels they state under a
//! meta context.
//!
//! Nothing here touches a GPU.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use nn::collective::Split;
use nn::family::Projection;
use nn::functional;
use nn::modules::{UlyssesAttention, UlyssesAttentionConfig, VsaAttention, VsaAttentionConfig};
use nn::{Checkpoint, CpuWeights, DType, Device, MetaWeights, Tensor, WeightLoader};
use nn_testing::{
    Recording, assert_close, checkpoint, meta, operators, record, sequence, tensor, values,
};

/// The tolerance of an FP32 result against a reference accumulated in another order.
const F32_TOLERANCE: f32 = 2e-5;

/// `value` rounded into BF16, to nearest even, as the library's conversion rounds it.
fn bf16(value: f32) -> f32 {
    values(&tensor(&[value], 1, DType::BF16))[0]
}

/// A CPU tensor of unsigned entries.
fn unsigned(entries: &[u32], shape: impl Into<nn::Shape>) -> Tensor {
    Tensor::from_slice(entries, shape.into(), &Device::Cpu).unwrap()
}

/// A CPU tensor of signed entries.
fn signed(entries: &[i32]) -> Tensor {
    Tensor::from_slice(entries, entries.len(), &Device::Cpu).unwrap()
}

/// The entries of an unsigned tensor, in row-major order.
fn entries(tensor: &Tensor) -> Vec<u32> {
    tensor.flatten_all().unwrap().to_vec1().unwrap()
}

// ---- Sparse attention -------------------------------------------------------------------------

#[test]
fn sparse_attention_attends_the_visible_rows_of_the_selected_blocks() {
    const HEADS: usize = 2;
    const WIDTH: usize = 4;
    const TILE: usize = 2;
    let (query, key, value) = (
        sequence(4 * HEADS * WIDTH, 1),
        sequence(6 * HEADS * WIDTH, 2),
        sequence(6 * HEADS * WIDTH, 3),
    );

    // Three key blocks of two, one and no visible rows.
    let valid = [2_u32, 1, 0];
    let indices = [[[0_u32, 1, 2], [2, 0, 1]], [[1, 0, 2], [0, 1, 2]]];
    let counts = [[2_u32, 1], [3, 0]];

    let rows = |values: &[f32], count: usize| tensor(values, (count, HEADS, WIDTH), DType::F32);
    let out = functional::sparse_attention(
        (&rows(&query, 4), &rows(&key, 6), &rows(&value, 6)),
        (
            &unsigned(indices.as_flattened().as_flattened(), (2, 2, 3)),
            &unsigned(counts.as_flattened(), (2, 2)),
        ),
        &unsigned(&valid, 3),
        TILE as u32,
        6,
        DType::F32,
    )
    .unwrap();
    assert_eq!(out.dims(), [4, HEADS, WIDTH]);

    let scale = 1.0 / (WIDTH as f32).sqrt();
    let at = |values: &[f32], row: usize, head: usize, column: usize| {
        values[(row * HEADS + head) * WIDTH + column]
    };
    let mut expected = vec![0.0_f32; 4 * HEADS * WIDTH];
    for row in 0..4 {
        for head in 0..HEADS {
            let tile = row / TILE;
            let visible: Vec<usize> = indices[head][tile][..counts[head][tile] as usize]
                .iter()
                .flat_map(|block| {
                    let first = *block as usize * TILE;
                    first..first + valid[*block as usize] as usize
                })
                .collect();

            let scores: Vec<f32> = visible
                .iter()
                .map(|key_row| {
                    (0..WIDTH)
                        .map(|column| {
                            at(&query, row, head, column) * at(&key, *key_row, head, column)
                        })
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|score| (score - peak).exp()).collect();
            let total: f32 = weights.iter().sum();

            // A tile that reads no visible key writes zeros.
            for column in 0..WIDTH {
                let attended: f32 = visible
                    .iter()
                    .zip(&weights)
                    .map(|(key_row, weight)| weight * at(&value, *key_row, head, column))
                    .sum();
                expected[(row * HEADS + head) * WIDTH + column] = if visible.is_empty() {
                    0.0
                } else {
                    attended / total
                };
            }
        }
    }
    assert_close(&values(&out), &expected, F32_TOLERANCE);

    // The selection's tables are unsigned, and the rows whole tiles.
    let wrong = Tensor::from_slice(&[2_i64, 1, 0], 3, &Device::Cpu).unwrap();
    let call = |valid: &Tensor, query_rows: usize| {
        functional::sparse_attention(
            (
                &rows(&query[..query_rows * HEADS * WIDTH], query_rows),
                &rows(&key, 6),
                &rows(&value, 6),
            ),
            (
                &unsigned(indices.as_flattened().as_flattened(), (2, 2, 3)),
                &unsigned(counts.as_flattened(), (2, 2)),
            ),
            valid,
            TILE as u32,
            6,
            DType::F32,
        )
    };
    assert!(call(&wrong, 4).is_err());
    assert!(call(&unsigned(&valid, 3), 3).is_err());
}

// ---- Preparation ------------------------------------------------------------------------------

#[test]
fn a_preparation_normalizes_rotates_zeroes_and_pools_its_tiles() {
    const TILES: usize = 3;
    const TILE: usize = 4;
    const HEADS: usize = 2;
    const WIDTH: usize = 8;
    const ROTATED: usize = 4;
    const ROWS: usize = TILES * TILE;
    let eps = 1e-6;
    let narrow = |seed: u32| {
        tensor(
            &sequence(ROWS * HEADS * WIDTH, seed),
            (ROWS, HEADS, WIDTH),
            DType::BF16,
        )
    };

    let (query, key, value) = (narrow(4), narrow(5), narrow(6));
    let before = [values(&query), values(&key), values(&value)];
    let gains =
        [sequence(WIDTH, 7), sequence(WIDTH, 8)].map(|gain| tensor(&gain, WIDTH, DType::BF16));
    let (cosine, sine) = (
        sequence(ROWS * ROTATED / 2, 9),
        sequence(ROWS * ROTATED / 2, 10),
    );
    let sizes = [4_u32, 2, 0];

    let ([query, key, value], pooled) = functional::vsa_prepare_tiles_(
        [query, key, value],
        Some((&gains[0], &gains[1])),
        Some((
            &tensor(&cosine, (ROWS, ROTATED / 2), DType::F32),
            &tensor(&sine, (ROWS, ROTATED / 2), DType::F32),
        )),
        &unsigned(&sizes, TILES),
        TILE as u32,
        ROTATED as u32,
        eps,
    )
    .unwrap();
    assert_eq!(pooled.dims(), [3, TILES, HEADS, WIDTH]);

    // One head of one valid row, by the kernel's stepwise rounding: the weighted
    // normalization, the factors, each product and the rotated sum round into BF16.
    let half = ROTATED / 2;
    let prepared = |head: &[f32], gain: &[f32], row: usize| -> Vec<f32> {
        let mean_square = head.iter().map(|value| value * value).sum::<f32>() / WIDTH as f32;
        let inverse = 1.0 / (mean_square + eps).sqrt();
        let normalized: Vec<f32> = head
            .iter()
            .zip(gain)
            .map(|(value, gain)| bf16(value * inverse * gain))
            .collect();

        (0..WIDTH)
            .map(|column| {
                if column >= ROTATED {
                    return normalized[column];
                }
                let pair = column % half;
                let (cos, sin) = (
                    bf16(cosine[row * half + pair]),
                    bf16(sine[row * half + pair]),
                );
                let direct = bf16(normalized[column] * cos);

                if column < half {
                    bf16(direct - bf16(normalized[column + half] * sin))
                } else {
                    bf16(direct + bf16(normalized[column - half] * sin))
                }
            })
            .collect()
    };

    let mut expected = before.clone();
    for (index, rows) in expected.iter_mut().enumerate() {
        for row in 0..ROWS {
            for head in 0..HEADS {
                let stored = &mut rows[(row * HEADS + head) * WIDTH..][..WIDTH];

                if row % TILE >= sizes[row / TILE] as usize {
                    stored.fill(0.0);
                } else if index < 2 {
                    let turned = prepared(stored, &values(&gains[index]), row);
                    stored.copy_from_slice(&turned);
                }
            }
        }
    }
    for (stored, expected) in [&query, &key, &value].into_iter().zip(&expected) {
        assert_close(&values(stored), expected, 1.0 / 128.0);
    }

    // A tile's mean is the sum of its stored rows over its valid size, or over one.
    let mut means = Vec::new();
    for rows in [&query, &key, &value].map(values) {
        for tile in 0..TILES {
            for head in 0..HEADS {
                for column in 0..WIDTH {
                    let sum: f32 = (0..TILE)
                        .map(|row| rows[((tile * TILE + row) * HEADS + head) * WIDTH + column])
                        .sum();
                    means.push(sum / sizes[tile].max(1) as f32);
                }
            }
        }
    }
    assert_close(&values(&pooled), &means, F32_TOLERANCE);

    // A rotation reads the gains and both factor tables.
    let refused = functional::vsa_prepare_tiles_(
        [narrow(4), narrow(5), narrow(6)],
        Some((&gains[0], &gains[1])),
        None,
        &unsigned(&sizes, TILES),
        TILE as u32,
        ROTATED as u32,
        eps,
    );
    assert!(refused.is_err());
}

// ---- Scores, selection and compression --------------------------------------------------------

#[test]
fn the_scores_are_one_scaled_dot_product_per_head_and_tile_pair() {
    const HEADS: usize = 2;
    const WIDTH: usize = 4;
    let (queries, keys) = (
        sequence(3 * HEADS * WIDTH, 11),
        sequence(5 * HEADS * WIDTH, 12),
    );

    let scores = functional::vsa_scores(
        &tensor(&queries, (3, HEADS, WIDTH), DType::F32),
        &tensor(&keys, (5, HEADS, WIDTH), DType::F32),
    )
    .unwrap();
    assert_eq!(scores.dims(), [HEADS, 3, 5]);

    let mut expected = Vec::new();
    for head in 0..HEADS {
        for query in 0..3 {
            for key in 0..5 {
                let dot: f32 = (0..WIDTH)
                    .map(|column| {
                        queries[(query * HEADS + head) * WIDTH + column]
                            * keys[(key * HEADS + head) * WIDTH + column]
                    })
                    .sum();
                expected.push(dot / (WIDTH as f32).sqrt());
            }
        }
    }
    assert_close(&values(&scores), &expected, F32_TOLERANCE);
}

#[test]
fn a_selection_keeps_the_dense_tiles_and_each_segment_s_highest_scores() {
    // Tile 0 is outside the segments, tiles 1 to 3 are segment 0 (tile 3 empty) and tiles 4
    // and 5 segment 1; a query keeps two tiles of segment 0 and one of segment 1.
    let segments = signed(&[-1, 0, 0, 0, 1, 1]);
    let sizes = unsigned(&[4, 4, 4, 0, 4, 4], 6);
    let keep = unsigned(&[2, 1, 0, 0, 0, 0], 6);

    // Head 0 ranks tiles 3 and 2 of segment 0, and breaks segment 1's tie by the lower tile.
    // Head 1 ranks a NaN above everything and reads −0 as +0.
    let rows = [
        [9.0_f32, 0.5, 0.7, 0.9, 0.3, 0.3],
        [0.0, f32::NAN, 0.7, 0.9, -0.0, 0.0],
    ];
    let scores: Vec<f32> = rows
        .iter()
        .flat_map(|row| std::iter::repeat_n(row, 6).flatten().copied())
        .collect();

    let (indices, counts) = functional::vsa_select_tiles(
        &tensor(&scores, (2, 6, 6), DType::F32),
        &sizes,
        &segments,
        &keep,
        0,
    )
    .unwrap();

    // A live query outside the segments keeps every live tile, and an empty query none; the
    // kept tiles come first, each part in ascending order.
    let dense = [0, 1, 2, 4, 5, 3];
    let empty = [0, 1, 2, 3, 4, 5];
    let selected = [[0, 2, 3, 4, 1, 5], [0, 1, 3, 4, 2, 5]];
    let expected: Vec<u32> = selected
        .iter()
        .flat_map(|kept| [dense, *kept, *kept, empty, *kept, *kept])
        .flatten()
        .collect();
    assert_eq!(entries(&indices), expected);
    assert_eq!(entries(&counts), [5, 4, 4, 0, 4, 4, 5, 4, 4, 0, 4, 4]);

    // A rank of an all-gather group ranks its own interval of query tiles, here tiles 2 to 4,
    // over every key tile: its selection is the whole selection's rows of those tiles.
    let interval: Vec<f32> = rows
        .iter()
        .flat_map(|row| std::iter::repeat_n(row, 3).flatten().copied())
        .collect();
    let (indices, counts) = functional::vsa_select_tiles(
        &tensor(&interval, (2, 3, 6), DType::F32),
        &sizes,
        &segments,
        &keep,
        2,
    )
    .unwrap();
    let expected: Vec<u32> = selected
        .iter()
        .flat_map(|kept| [*kept, empty, *kept])
        .flatten()
        .collect();
    assert_eq!(entries(&indices), expected);
    assert_eq!(entries(&counts), [4, 0, 4, 4, 0, 4]);

    // An interval past the tiles, a signed size or an unsigned segment table is refused.
    let past = functional::vsa_select_tiles(
        &tensor(&interval, (2, 3, 6), DType::F32),
        &sizes,
        &segments,
        &keep,
        4,
    );
    assert!(past.is_err());
    let refused = functional::vsa_select_tiles(
        &tensor(&scores, (2, 6, 6), DType::F32),
        &sizes,
        &keep,
        &keep,
        0,
    );
    assert!(refused.is_err());
}

#[test]
fn a_compression_weighs_the_live_tiles_pooled_values_by_their_softmax() {
    const HEADS: usize = 2;
    const WIDTH: usize = 4;
    const TILES: usize = 4;
    let (scores, pooled) = (
        sequence(HEADS * 3 * TILES, 13),
        sequence(TILES * HEADS * WIDTH, 14),
    );
    let sizes = [4_u32, 0, 2, 4];

    let compress = |sizes: &[u32]| {
        functional::vsa_compress_pooled(
            &tensor(&scores, (HEADS, 3, TILES), DType::F32),
            &tensor(&pooled, (TILES, HEADS, WIDTH), DType::F32),
            &unsigned(sizes, TILES),
            64,
            DType::F32,
        )
        .unwrap()
    };
    let compressed = compress(&sizes);
    assert_eq!(compressed.dims(), [HEADS, 3, WIDTH]);

    let mut expected = Vec::new();
    for head in 0..HEADS {
        for query in 0..3 {
            let row = &scores[(head * 3 + query) * TILES..][..TILES];
            let weights: Vec<f32> = row
                .iter()
                .zip(&sizes)
                .map(|(score, size)| if *size > 0 { score.exp() } else { 0.0 })
                .collect();
            let total: f32 = weights.iter().sum();

            for column in 0..WIDTH {
                let sum: f32 = (0..TILES)
                    .map(|tile| weights[tile] * pooled[(tile * HEADS + head) * WIDTH + column])
                    .sum();
                expected.push(sum / total);
            }
        }
    }
    assert_close(&values(&compressed), &expected, F32_TOLERANCE);

    // With no live tile there is nothing to weigh.
    assert!(
        values(&compress(&[0; TILES]))
            .iter()
            .all(|value| *value == 0.0)
    );
}

// ---- Padding and the gated fold ---------------------------------------------------------------

#[test]
fn the_gated_fold_adds_each_tile_s_compression_times_the_gate() {
    const TILES: usize = 4;
    const TILE: usize = 2;
    const HEADS: usize = 2;
    const WIDTH: usize = 3;
    const ROWS: usize = TILES * TILE;
    let narrow = |values: &[f32], shape: (usize, usize, usize)| tensor(values, shape, DType::BF16);

    let fine = narrow(&sequence(ROWS * HEADS * WIDTH, 15), (ROWS, HEADS, WIDTH));
    let gate = narrow(&sequence(ROWS * HEADS * WIDTH, 16), (ROWS, HEADS, WIDTH));
    let compressed = narrow(&sequence(HEADS * TILES * WIDTH, 17), (HEADS, TILES, WIDTH));

    // The product rounds, then the sum rounds.
    let (fine_values, gate_values, tile_values) =
        (values(&fine), values(&gate), values(&compressed));
    let folded: Vec<f32> = (0..ROWS * HEADS * WIDTH)
        .map(|index| {
            let (row, head, column) = (
                index / (HEADS * WIDTH),
                index / WIDTH % HEADS,
                index % WIDTH,
            );
            let tile = tile_values[(head * TILES + row / TILE) * WIDTH + column];
            bf16(fine_values[index] + bf16(tile * gate_values[index]))
        })
        .collect();

    // Two members of four rows each return rows 2..4 of their own, member 1's block first.
    let payload =
        functional::vsa_compose_gated_rows(&fine, &compressed, &gate, TILE as u32, 2..4, &[1, 0])
            .unwrap();
    assert_eq!(payload.dims(), [2, 2, HEADS, WIDTH]);
    let block = |first: usize| folded[first * HEADS * WIDTH..][..2 * HEADS * WIDTH].to_vec();
    assert_eq!(values(&payload), [block(6), block(2)].concat());

    let refuse = |rows, slots: &[u32]| {
        functional::vsa_compose_gated_rows(&fine, &compressed, &gate, TILE as u32, rows, slots)
            .is_err()
    };
    assert!(refuse(2..6, &[0, 1]), "an interval past a member's rows");
    assert!(refuse(0..2, &[0, 0]), "two members in one slot");
    assert!(refuse(0..2, &[]), "no member");

    // The fold itself writes the fine rows in place.
    let handle = fine.clone();
    let returned = functional::vsa_add_gated_tiles_(fine, &compressed, &gate, TILE as u32).unwrap();
    assert_eq!(values(&returned), folded);
    assert_eq!(values(&handle), folded);
}

#[test]
fn the_padding_rows_of_each_tile_become_zeros_in_place() {
    let rows = sequence(3 * 4 * 2 * 3, 18);
    let stored = tensor(&rows, (12, 2, 3), DType::F32);
    let handle = stored.clone();

    let zeroed = functional::vsa_zero_padding_(stored, &unsigned(&[4, 1, 0], 3), 4).unwrap();
    let expected: Vec<f32> = rows
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let row = index / 6;
            if row % 4 < [4, 1, 0][row / 4] {
                *value
            } else {
                0.0
            }
        })
        .collect();
    assert_eq!(values(&zeroed), expected);
    assert_eq!(values(&handle), expected);
}

// ---- The module -------------------------------------------------------------------------------

/// A layer of `heads` heads `width` wide over `tile_rows`-row tiles with a tile bound of 4096.
fn vsa_config(
    tile_rows: u32,
    heads: u32,
    width: u32,
    norm: bool,
    rotated: u32,
) -> VsaAttentionConfig {
    VsaAttentionConfig {
        tile_rows,
        heads,
        head_dim: width,
        qk_norm: norm.then_some(("norm_q", "norm_k")),
        eps: 1e-6,
        rotated,
        max_tiles: 4_096,
        max_keys: 64 * tile_rows,
        output: DType::BF16,
    }
}

#[test]
fn a_sparse_layer_states_its_tiles_bounds_and_rotation() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        DType::BF16,
        &[
            ("attn.norm_q.weight", vec![256], sequence(256, 19)),
            ("attn.norm_k.weight", vec![256], sequence(256, 20)),
        ],
    );
    let store = MetaWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let load = |config: VsaAttentionConfig| VsaAttention::load(&config, loader.root().pp("attn"));

    // A tile is whole sixteen-row kernel blocks: 64, 128 and 96 rows serve, 100 do not.
    for tile_rows in [64, 128, 96] {
        assert!(load(vsa_config(tile_rows, 8, 256, false, 0)).is_ok());
    }
    let text = load(vsa_config(100, 8, 256, false, 0))
        .err()
        .unwrap()
        .to_string();
    assert!(text.contains("100"), "{text}");
    assert!(load(vsa_config(0, 8, 256, false, 0)).is_err());
    assert!(load(vsa_config(64, 0, 256, false, 0)).is_err());
    assert!(load(vsa_config(64, 8, 0, false, 0)).is_err());

    // The tile bound and the key bound are stated, the latter in whole tiles.
    let base = vsa_config(64, 8, 256, false, 0);
    for (max_tiles, max_keys) in [
        (0, base.max_keys),
        (base.max_tiles, 0),
        (base.max_tiles, 100),
    ] {
        assert!(
            load(VsaAttentionConfig {
                max_tiles,
                max_keys,
                ..base
            })
            .is_err()
        );
    }

    // A rotation reads pairs of a head's features, with the normalization.
    assert!(load(vsa_config(64, 8, 256, true, 96)).is_ok());
    assert!(
        load(vsa_config(64, 8, 256, false, 96)).is_err(),
        "a rotation without the norm"
    );
    assert!(
        load(vsa_config(64, 8, 256, true, 97)).is_err(),
        "an odd rotation"
    );
    assert!(
        load(vsa_config(64, 8, 256, true, 512)).is_err(),
        "more features than a head has"
    );
}

#[test]
fn an_attend_prepares_scores_selects_attends_and_compresses() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        DType::BF16,
        &[
            ("attn.norm_q.weight", vec![128], sequence(128, 21)),
            ("attn.norm_k.weight", vec![128], sequence(128, 22)),
        ],
    );
    let store = MetaWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let layer =
        VsaAttention::load(&vsa_config(64, 4, 128, true, 96), loader.root().pp("attn")).unwrap();

    let rows = 4 * 64;
    let attend = |rows: usize| {
        let projection = || meta((rows, 4 * 128), DType::BF16);
        let factors = (meta((rows, 48), DType::F32), meta((rows, 48), DType::F32));
        let table = |dtype| meta(4, dtype);

        layer.attend(
            (projection(), projection(), projection()),
            Some((&factors.0, &factors.1)),
            (&table(DType::U32), &table(DType::I32), &table(DType::U32)),
        )
    };

    let (branches, Recording { launches, .. }) = record(&[], || {
        let (fine, compressed) = attend(rows)?;
        let gate = meta((rows, 4 * 128), DType::BF16);

        // The gated fold on one rank, and one interval of a two-member return.
        let payload = layer.compose(&fine, &compressed, &gate, 0..64, &[0, 1])?;
        let folded = layer.add_gated_(fine, &compressed, &gate)?;
        Ok((
            folded.dims().to_vec(),
            compressed.dims().to_vec(),
            payload.dims().to_vec(),
        ))
    });
    assert_eq!(
        branches.unwrap(),
        (vec![rows, 4 * 128], vec![4, 4, 128], vec![2, 64, 4, 128])
    );
    assert_eq!(
        operators(&launches),
        [
            "vsa_prepare_tiles",
            "vsa_scores",
            "vsa_select_tiles",
            "sparse_attention",
            "vsa_zero_padding",
            "vsa_compress_pooled",
            "vsa_compose_gated_rows",
            "vsa_add_gated_tiles",
        ]
    );
    assert_eq!(
        launches[3].1,
        "block_64_dtype_bf16_output_bf16_head_dim_128_lse_false"
    );

    // Rows that are not whole tiles are refused before any launch.
    let (ragged, Recording { launches, .. }) = record(&[], || attend(rows + 1));
    assert!(ragged.is_err());
    assert!(launches.is_empty());
}

/// The selection's kernel ranks the layout's own tiles rounded up to a power of two: its tile
/// count is a launch value and the kernel masks the tiles past it, so one compilation serves
/// every layout of a class.
#[test]
fn the_select_block_is_the_layout_s_tiles_rounded_up() {
    let block = |tiles: usize| {
        let (selection, Recording { launches, .. }) = record(&[], || {
            functional::vsa_select_tiles(
                &meta((4, 1, tiles), DType::F32),
                &meta(tiles, DType::U32),
                &meta(tiles, DType::I32),
                &meta(tiles, DType::U32),
                0,
            )
        });
        let (indices, counts) = selection.unwrap();
        assert_eq!((indices.dtype(), counts.dtype()), (DType::U32, DType::U32));
        assert_eq!(launches.len(), 1, "{launches:?}");
        launches[0].1.clone()
    };

    // 5 400 rounds up to 8 192, and a layout below the floor is compiled at the floor.
    assert_eq!(block(5_400), "block_8192");
    assert_eq!(block(1), block(256));
}

#[test]
fn a_sparse_layer_on_the_cpu_is_its_operators_in_order() {
    const TILE: usize = 16;
    const HEADS: usize = 2;
    const WIDTH: usize = 4;
    const ROWS: usize = 2 * TILE;
    let directory = tempfile::tempdir().unwrap();
    let (query_gain, key_gain) = (sequence(WIDTH, 23), sequence(WIDTH, 24));
    let checkpoint = checkpoint(
        directory.path(),
        DType::BF16,
        &[
            ("attn.norm_q.weight", vec![WIDTH], query_gain.clone()),
            ("attn.norm_k.weight", vec![WIDTH], key_gain.clone()),
        ],
    );
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let config = vsa_config(TILE as u32, HEADS as u32, WIDTH as u32, true, 2);
    let layer = VsaAttention::load(&config, loader.root().pp("attn")).unwrap();

    let projection = |seed: u32| {
        tensor(
            &sequence(ROWS * HEADS * WIDTH, seed),
            (ROWS, HEADS * WIDTH),
            DType::BF16,
        )
    };
    let factors = (
        tensor(&sequence(ROWS, 28), (ROWS, 1), DType::F32),
        tensor(&sequence(ROWS, 29), (ROWS, 1), DType::F32),
    );
    let (sizes, segments, keep) = (
        unsigned(&[16, 9], 2),
        signed(&[-1, 0]),
        unsigned(&[1, 0], 2),
    );

    let (fine, compressed) = layer
        .attend(
            (projection(25), projection(26), projection(27)),
            Some((&factors.0, &factors.1)),
            (&sizes, &segments, &keep),
        )
        .unwrap();
    assert_eq!(fine.dims(), [ROWS, HEADS * WIDTH]);
    assert_eq!(compressed.dims(), [HEADS, 2, WIDTH]);

    // The same operators, called one by one on the same projections.
    let headed = |seed: u32| projection(seed).reshape((ROWS, HEADS, WIDTH)).unwrap();
    let gain = |values: &[f32]| tensor(values, WIDTH, DType::BF16);
    let ([query, key, value], pooled) = functional::vsa_prepare_tiles_(
        [headed(25), headed(26), headed(27)],
        Some((&gain(&query_gain), &gain(&key_gain))),
        Some((&factors.0, &factors.1)),
        &sizes,
        TILE as u32,
        2,
        1e-6,
    )
    .unwrap();
    let scores = functional::vsa_scores(&pooled.get(0).unwrap(), &pooled.get(1).unwrap()).unwrap();
    let (indices, counts) =
        functional::vsa_select_tiles(&scores, &sizes, &segments, &keep, 0).unwrap();
    let attended = functional::sparse_attention(
        (&query, &key, &value),
        (&indices, &counts),
        &sizes,
        TILE as u32,
        config.max_keys,
        DType::BF16,
    )
    .unwrap();
    let expected = functional::vsa_zero_padding_(attended, &sizes, TILE as u32).unwrap();
    assert_eq!(values(&fine), values(&expected));

    // The padding rows of the partial tile are zeros, and its valid rows are not.
    let rows = values(&fine);
    let row = |index: usize| &rows[index * HEADS * WIDTH..][..HEADS * WIDTH];
    assert!(row(TILE + 9).iter().all(|value| *value == 0.0));
    assert!(row(TILE + 8).iter().any(|value| *value != 0.0));

    let pooled_values = pooled.get(2).unwrap();
    let expected =
        functional::vsa_compress_pooled(&scores, &pooled_values, &sizes, 4_096, DType::BF16)
            .unwrap();
    assert_eq!(values(&compressed), values(&expected));

    // A layout past the deployment's tile bound is refused.
    let bounded = VsaAttention::load(
        &VsaAttentionConfig {
            max_tiles: 1,
            ..config
        },
        loader.root().pp("attn"),
    )
    .unwrap();
    let refused = bounded.attend(
        (projection(25), projection(26), projection(27)),
        Some((&factors.0, &factors.1)),
        (&sizes, &segments, &keep),
    );
    assert!(refused.err().unwrap().to_string().contains("bound of 1"));
}

// ---- Sequence parallelism ---------------------------------------------------------------------

/// A member of an all-gather group attends its own interval of query tiles over the whole
/// sequence's keys, which the group gathers: it prepares its interval, gathers the keys, the
/// values and their means, and its outputs are its interval's. An interval past the sequence is
/// refused.
#[test]
fn a_member_of_an_all_gather_group_attends_its_interval_over_every_tile() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        DType::BF16,
        &[
            ("attn.norm_q.weight", vec![128], sequence(128, 21)),
            ("attn.norm_k.weight", vec![128], sequence(128, 22)),
        ],
    );
    let store = MetaWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let layer =
        VsaAttention::load(&vsa_config(64, 4, 128, true, 96), loader.root().pp("attn")).unwrap();

    // Member 1 of 2 holds tiles 2 and 3 of a sequence of four.
    let attend = |first: usize| {
        let projection = || meta((2 * 64, 4 * 128), DType::BF16);
        let factors = (
            meta((2 * 64, 48), DType::F32),
            meta((2 * 64, 48), DType::F32),
        );
        let table = |dtype| meta(4, dtype);

        layer.attend_gathered(
            (projection(), projection(), projection()),
            Some((&factors.0, &factors.1)),
            (&table(DType::U32), &table(DType::I32), &table(DType::U32)),
            first,
        )
    };
    let (branches, Recording { launches, .. }) = record(&[Split::new(2, 1).unwrap()], || attend(2));
    let (fine, compressed) = branches.unwrap();
    assert_eq!(fine.dims(), [2 * 64, 4 * 128]);
    assert_eq!(
        compressed.dims(),
        [4, 2, 128],
        "one row per head and own query tile"
    );
    let operators = operators(&launches);
    for operator in ["vsa_prepare_tiles", "vsa_select_tiles", "sparse_attention"] {
        assert!(operators.contains(&operator), "{operators:?}");
    }

    let (past, _) = record(&[Split::new(2, 1).unwrap()], || attend(3));
    assert!(past.is_err());
}

/// Member 1 of 2 of a four-head, sixteen-wide attention reaching its heads by `projection`.
fn ulysses(checkpoint: &Checkpoint, projection: Projection) -> UlyssesAttention {
    let store = MetaWeights::new();
    let loader = WeightLoader::new(checkpoint, &store, None);

    UlyssesAttention::load(
        &UlyssesAttentionConfig {
            hidden: 32,
            heads: 4,
            head_dim: 16,
            inputs: &["to_q"],
            output: "to_out",
            split: Split::new(2, 1).unwrap(),
            projection,
            tensor: Split::WHOLE,
            axis: 0,
        },
        loader.root().pp("attn"),
    )
    .unwrap()
}

/// In the exchange mode a member projects its rows for each member's heads and exchanges the
/// blocks; in the gather mode it gathers every member's rows and projects its own heads.
/// Either way the return sets the members' blocks side by side before the output projection,
/// and a forward split over another member count is refused before any collective.
#[test]
fn a_sequence_split_projects_heads_by_its_mode_and_returns_rows() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        DType::BF16,
        &[
            ("attn.to_q.weight", vec![64, 32], sequence(64 * 32, 30)),
            ("attn.to_out.weight", vec![32, 64], sequence(32 * 64, 31)),
        ],
    );
    let split = [Split::new(2, 1).unwrap()];

    // Four own rows of a sequence of eight; this member's heads are two of four, 32 channels.
    let exchange = ulysses(&checkpoint, Projection::Exchange);
    assert_eq!((exchange.local_heads(), exchange.dtype()), (2, DType::BF16));
    let (projected, Recording { launches, .. }) = record(&split, || {
        let rows = exchange.rows(&meta((4, 32), DType::BF16))?;
        exchange.project(0, &rows)
    });
    assert_eq!(
        projected.unwrap().dims(),
        [8, 32],
        "every row of this member's heads"
    );
    assert_eq!(
        operators(&launches),
        ["gemm", "gemm", "copy_strided", "copy_strided"],
        "one product per member, set one after another"
    );

    let gather = ulysses(&checkpoint, Projection::Gather);
    let (projected, Recording { launches, .. }) = record(&split, || {
        let rows = gather.rows(&meta((4, 32), DType::BF16))?;
        gather.project(0, &rows)
    });
    assert_eq!(projected.unwrap().dims(), [8, 32]);
    assert_eq!(
        operators(&launches),
        ["gemm"],
        "one product of this member's heads"
    );

    // The return: one exchange, the merge of the members' blocks and the output projection.
    let (returned, Recording { launches, .. }) =
        record(&split, || exchange.return_rows(&meta((8, 32), DType::BF16)));
    assert_eq!(
        returned.unwrap().dims(),
        [4, 32],
        "this member's rows of every head"
    );
    assert_eq!(operators(&launches), ["copy_strided", "gemm"]);

    // The return by intervals: a started exchange, taken up into the interval's rows.
    let (returned, Recording { launches, .. }) = record(&split, || {
        let payload = meta((2, 2, 2, 16), DType::BF16);
        let (received, started) = exchange.start_exchange(&payload)?;
        exchange.receive(started, &received)
    });
    assert_eq!(returned.unwrap().dims(), [2, 32]);
    assert_eq!(operators(&launches), ["copy_strided", "gemm"]);

    let (gathered, _) = record(&split, || {
        let sequence = meta((8, 32), DType::BF16);
        let started = gather.start_gather(&sequence, 4, 0..2)?;
        gather.finish(started)
    });
    gathered.unwrap();

    // A group of another size, and a forward with none, are refused before any collective.
    let (refused, _) = record(&[Split::new(4, 1).unwrap()], || {
        exchange.project(0, &meta((4, 32), DType::BF16))
    });
    let text = refused.err().unwrap().to_string();
    assert!(text.contains("loaded for 2 members"), "{text}");
    let (refused, _) = record(&[], || exchange.project(0, &meta((4, 32), DType::BF16)));
    assert!(refused.is_err());
}

/// Inside a tensor split a member holds its share of its tensor rank's heads, exchanges along
/// the sequence axis its config names, and its output projection sums its share of the
/// contraction along the axis the caller's forward addresses: the tensor group's.
#[test]
fn a_sequence_split_inside_a_tensor_split_holds_its_share_of_the_tensor_rank_s_heads() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        DType::BF16,
        &[
            ("attn.to_q.weight", vec![64, 32], sequence(64 * 32, 30)),
            ("attn.to_out.weight", vec![32, 64], sequence(32 * 64, 31)),
        ],
    );
    let store = MetaWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let member = UlyssesAttention::load(
        &UlyssesAttentionConfig {
            hidden: 32,
            heads: 4,
            head_dim: 16,
            inputs: &["to_q"],
            output: "to_out",
            split: Split::new(2, 1).unwrap(),
            projection: Projection::Exchange,
            tensor: Split::new(2, 1).unwrap(),
            axis: 1,
        },
        loader.root().pp("attn"),
    )
    .unwrap();
    assert_eq!(member.local_heads(), 1, "one head of the tensor rank's two");

    // The tensor group is the first axis, the sequence group the second.
    let context = ops::Context::meta(None);
    context.set_axes(vec![(None, 0, 2), (None, 1, 2)]).unwrap();
    context.record_launches();
    let _guard = context.enter();

    // Four own rows of a sequence of eight, one head of sixteen channels from each member.
    let projected = member.project(0, &meta((4, 32), DType::BF16)).unwrap();
    assert_eq!(projected.dims(), [8, 16]);

    // The return: the exchange along the sequence axis, then the projection of the tensor
    // rank's 32 channels, reduced along the tensor axis.
    let returned = member.return_rows(&meta((8, 16), DType::BF16)).unwrap();
    assert_eq!(returned.dims(), [4, 32]);

    // Heads a tensor split and a sequence split do not divide are refused.
    let refused = UlyssesAttention::load(
        &UlyssesAttentionConfig {
            hidden: 32,
            heads: 4,
            head_dim: 16,
            inputs: &["to_q"],
            output: "to_out",
            split: Split::new(4, 0).unwrap(),
            projection: Projection::Exchange,
            tensor: Split::new(2, 0).unwrap(),
            axis: 1,
        },
        loader.root().pp("attn"),
    );
    assert!(refused.is_err());
}
