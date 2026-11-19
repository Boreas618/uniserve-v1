//! Sparse latent attention's functions on CPU tensors against their definitions written out in
//! loops: the record write of a paged field, the compression of a request's tokens, the
//! attention over a window and a selection with a sink, the weighted index scores, the
//! selection of the best of them, and the rotation of a head's trailing channels; and the
//! kernels the same functions state under a meta context at the model's shapes.
//!
//! Nothing here touches a GPU.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use nn::functional::{self, Format, Layout, Plane, RecordRow, Ring, Scoring, Selected, Window};
use nn::{DType, Device, Tensor};
use nn_testing::{Recording, assert_close, meta, record, sequence, tensor, values};
use ops::record as codec;

/// The tolerance of an FP32 result against a reference accumulated in another order.
const F32_TOLERANCE: f32 = 2e-5;

/// A CPU tensor of `entries`.
fn int<T: tensor::WithDType>(entries: Vec<T>, shape: impl Into<nn::Shape>) -> Tensor {
    Tensor::from_vec(entries, shape.into(), &Device::Cpu).unwrap()
}

/// `value` rounded into BF16 and back.
fn bf16(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}

/// The decoded record at `slot` of `page` of a packed plane of records of `width` values.
fn decoded(
    plane: &Tensor,
    (format, layout): (Format, Layout),
    width: usize,
    (page, slot): (usize, usize),
) -> Vec<f32> {
    let entries = plane.dims()[1] as u64;
    let bytes: Vec<u8> = plane
        .get(page)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let (at, scales) = layout.offsets(format, entries, width as u64, slot as u64);
    let (at, scales) = (at as usize, scales as usize);
    let (value_bytes, scale_bytes) = (
        format.value_bytes(width as u64) as usize,
        format.scale_bytes(width as u64) as usize,
    );
    codec::decode(
        format,
        width,
        &bytes[at..at + value_bytes],
        &bytes[scales..scales + scale_bytes],
    )
}

/// Holds `actual` scores to `expected`: the same columns infinite, the finite ones close.
fn assert_scores(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "the score counts differ");
    for (column, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_infinite() {
            assert_eq!(actual, expected, "column {column}");
        } else {
            assert_close(&[*actual], &[*expected], F32_TOLERANCE);
        }
    }
}

#[test]
fn a_record_lands_on_its_page_and_slot_in_either_layout() {
    const WIDTH: usize = 64;
    const ENTRIES: usize = 4;
    for (format, layout) in [
        (Format::Mxfp4, Layout::Page),
        (Format::Mxfp8, Layout::Entry),
        (Format::Nvfp4, Layout::Entry),
    ] {
        let record_bytes = format.record_bytes(WIDTH as u64) as usize;
        let plane = Tensor::zeros((5, ENTRIES, record_bytes), DType::U8, &Device::Cpu).unwrap();
        // Two rows of three pages each; row 1's table is read modulo its width.
        let table = int(vec![4_u32, 0, 2, 1, 3, 0], (2, 3));
        let rows = sequence(4 * WIDTH, 1);
        let x = tensor(&rows, (4, WIDTH), DType::BF16);
        let entries = int(vec![5_i32, -1, 0, 13], 4);
        let row_of = int(vec![0_u32, 0, 1, 1], 4);

        functional::record_write_(
            &x,
            RecordRow {
                norm: None,
                inv_freq: None,
                ratio: 1,
            },
            Plane {
                records: &plane,
                table: &table,
                format,
                layout,
            },
            &entries,
            &row_of,
        )
        .unwrap();

        // Token 0, entry 5 of row 0: page table[0][1] = 0, slot 1. Token 2, entry 0 of row 1:
        // page 1, slot 0. Token 3, entry 13 of row 1: column 3 mod 3 = 0, page 1, slot 1.
        let stored = values(&x);
        for (token, page, slot) in [(0, 0, 1), (2, 1, 0), (3, 1, 1)] {
            let row = &stored[token * WIDTH..(token + 1) * WIDTH];
            let (codes, scales) = codec::encode(format, row);
            let expected = codec::decode(format, WIDTH, &codes, &scales);
            assert_eq!(
                decoded(&plane, (format, layout), WIDTH, (page, slot)),
                expected,
                "{format}: token {token}"
            );
        }
        // The token of no entry writes nothing: every other page stays zero.
        for page in [2, 3, 4] {
            let bytes: Vec<u8> = plane
                .get(page)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert!(bytes.iter().all(|byte| *byte == 0), "{format}: page {page}");
        }
    }
}

#[test]
fn a_normalized_row_rotates_its_trailing_pairs_at_its_groups_first_position() {
    const WIDTH: usize = 16;
    const PAIRS: usize = 4;
    let plane = Tensor::zeros((2, 2, WIDTH), DType::BF16, &Device::Cpu).unwrap();
    let table = int(vec![1_u32, 0], (1, 2));
    let rows = sequence(2 * WIDTH, 2);
    let gain = sequence(WIDTH, 3);
    let inv_freq = [1.0_f32, 0.5, 0.25, 0.125];
    let eps = 1e-6;

    functional::record_write_(
        &tensor(&rows, (2, WIDTH), DType::BF16),
        RecordRow {
            norm: Some((&tensor(&gain, WIDTH, DType::BF16), eps)),
            inv_freq: Some(&tensor(&inv_freq, PAIRS, DType::F32)),
            ratio: 2,
        },
        Plane {
            records: &plane,
            table: &table,
            format: Format::Bf16,
            layout: Layout::Entry,
        },
        &int(vec![3_i32, 0], 2),
        &int(vec![0_u32, 0], 2),
    )
    .unwrap();

    let stored = values(&plane);
    for (token, entry) in [(0_usize, 3_usize), (1, 0)] {
        let row: Vec<f32> = rows[token * WIDTH..(token + 1) * WIDTH]
            .iter()
            .map(|value| bf16(*value))
            .collect();
        let gain: Vec<f32> = gain.iter().map(|value| bf16(*value)).collect();
        let mean = row.iter().map(|value| value * value).sum::<f32>() / WIDTH as f32;
        let inverse = 1.0 / (mean + eps).sqrt();
        let mut normed: Vec<f32> = row
            .iter()
            .zip(&gain)
            .map(|(value, gain)| bf16(gain * (value * inverse)))
            .collect();
        // Entry `e` stands for positions `2e, 2e + 1`, and turns by the angles of `2e`.
        let position = (2 * entry) as f32;
        for (pair, frequency) in inv_freq.iter().enumerate() {
            let at = WIDTH - 2 * PAIRS + 2 * pair;
            let (cos, sin) = ((position * frequency).cos(), (position * frequency).sin());
            let (a, b) = (normed[at], normed[at + 1]);
            normed[at] = bf16(a * cos - b * sin);
            normed[at + 1] = bf16(b * cos + a * sin);
        }
        // Entry 3 is page table[0][1] = 0, slot 1; entry 0 is page 1, slot 0.
        let at = if entry == 3 { WIDTH } else { 2 * WIDTH };
        assert_close(&stored[at..at + WIDTH], &normed, 1e-6);
    }
}

#[test]
fn a_completing_token_pools_its_group_from_the_batch_and_the_ring() {
    const WIDTH: usize = 4;
    const RING: usize = 8;
    // One request's state row at slot 1 continuing at position 5 for four tokens (5..=8), one
    // padding row, and the ring holding position 4's projection in row 4.
    let projected = sequence(4 * 2 * WIDTH, 4);
    let mut ring = vec![0.0_f32; 2 * RING * 2 * WIDTH];
    let held = sequence(2 * WIDTH, 5);
    let at = (RING + 4) * 2 * WIDTH;
    ring[at..at + 2 * WIDTH].copy_from_slice(&held);
    let ring = tensor(&ring, (2, RING, 2 * WIDTH), DType::F32);
    let gain = vec![1.0_f32; WIDTH];

    let compressed = functional::latent_compress(
        &tensor(&projected, (4, 2 * WIDTH), DType::F32),
        (&tensor(&gain, WIDTH, DType::F32), 1e-6),
        Ring {
            rows: &ring,
            slots: &int(vec![1_u32, u32::MAX], 2),
            query_offsets: &int(vec![0_u32, 4, 4], 3),
            seq_lens: &int(vec![9_u32, 0], 2),
        },
        &int(vec![5_i32, 6, 7, 8], 4),
        2,
    )
    .unwrap();

    // Positions 5 and 7 complete groups 2 (members 4, 5) and 3 (members 6, 7).
    assert_eq!(compressed.entries.to_vec1::<i32>().unwrap(), [2, -1, 3, -1]);
    // The state row's last eight tokens are all four: ring rows of slot 1.
    assert_eq!(
        compressed.ring_rows.to_vec1::<i32>().unwrap(),
        [13, 14, 15, 8]
    );
    let member = |token: usize| &projected[token * 2 * WIDTH..(token + 1) * 2 * WIDTH];
    let latent = values(&compressed.latent);
    for (token, first, second) in [(0, &held[..], member(0)), (2, member(1), member(2))] {
        let pooled: Vec<f32> = (0..WIDTH)
            .map(|channel| {
                let (a, b) = (first[WIDTH + channel], second[WIDTH + channel]);
                let peak = a.max(b);
                let (wa, wb) = ((a - peak).exp(), (b - peak).exp());
                bf16((first[channel] * wa + second[channel] * wb) / (wa + wb))
            })
            .collect();
        let mean = pooled.iter().map(|value| value * value).sum::<f32>() / WIDTH as f32;
        let normed: Vec<f32> = pooled
            .iter()
            .map(|value| bf16(value / (mean + 1e-6).sqrt()))
            .collect();
        assert_close(
            &latent[token * WIDTH..(token + 1) * WIDTH],
            &normed,
            1.0 / 128.0,
        );
    }
}

#[test]
fn a_query_attends_its_window_and_its_selection_with_a_sink() {
    const HEADS: usize = 2;
    const WIDTH: usize = 16;
    const WINDOW: u32 = 3;
    // Window pages of 4 entries, compressed pages of 2; one row.
    let window_latents = sequence(2 * 4 * WIDTH, 6);
    let compressed_latents = sequence(3 * 2 * WIDTH, 7);
    let window = tensor(&window_latents, (2, 4, WIDTH), DType::BF16);
    let compressed = tensor(&compressed_latents, (3, 2, WIDTH), DType::BF16);
    let (window_table, compressed_table) =
        (int(vec![1_u32, 0], (1, 2)), int(vec![2_u32, 0, 1], (1, 3)));
    let query = sequence(2 * HEADS * WIDTH, 8);
    let sink = [0.5_f32, -1.0];
    let selection = int(vec![4_i32, -1, 1, 0, -1, -1], (2, 3));
    let scale = 0.25;

    let out = functional::latent_attention(
        &tensor(&query, (2, HEADS, WIDTH), DType::BF16),
        Window {
            plane: Plane {
                records: &window,
                table: &window_table,
                format: Format::Bf16,
                layout: Layout::Entry,
            },
            size: WINDOW,
        },
        Some(Selected {
            plane: Plane {
                records: &compressed,
                table: &compressed_table,
                format: Format::Bf16,
                layout: Layout::Entry,
            },
            selection: &selection,
        }),
        (&int(vec![0_u32, 0], 2), &int(vec![5_i32, 1], 2)),
        &tensor(&sink, HEADS, DType::F32),
        scale,
    )
    .unwrap();

    let latent = |plane: &[f32], table: &[u32], entries: usize, entry: usize| -> Vec<f32> {
        let slot = table[entry / entries] as usize * entries + entry % entries;
        plane[slot * WIDTH..(slot + 1) * WIDTH]
            .iter()
            .map(|v| bf16(*v))
            .collect()
    };
    let got = values(&out);
    for (token, position, selected) in [(0_usize, 5_usize, vec![4_usize, 1]), (1, 1, vec![0])] {
        // The window positions max(0, p - 2) ..= p, then the selected compressed entries.
        let mut read: Vec<Vec<f32>> = (position.saturating_sub(2)..=position)
            .map(|entry| latent(&window_latents, &[1, 0], 4, entry))
            .collect();
        read.extend(
            selected
                .iter()
                .map(|entry| latent(&compressed_latents, &[2, 0, 1], 2, *entry)),
        );
        for head in 0..HEADS {
            let q: Vec<f32> = query[(token * HEADS + head) * WIDTH..][..WIDTH]
                .iter()
                .map(|v| bf16(*v))
                .collect();
            let scores: Vec<f32> = read
                .iter()
                .map(|c| scale * q.iter().zip(c).map(|(a, b)| a * b).sum::<f32>())
                .collect();
            let peak = scores.iter().copied().fold(-1e30_f32, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - peak).exp()).collect();
            let total: f32 = weights.iter().sum::<f32>() + (sink[head] - peak).exp();
            let expected: Vec<f32> = (0..WIDTH)
                .map(|channel| {
                    read.iter()
                        .zip(&weights)
                        .map(|(c, w)| bf16(*w) * c[channel])
                        .sum::<f32>()
                        / total
                })
                .collect();
            assert_close(
                &got[(token * HEADS + head) * WIDTH..][..WIDTH],
                &expected,
                1.0 / 128.0,
            );
        }
    }

    // A padding token (a negative position and no selection) reads no entry: zeros.
    let padding = functional::latent_attention(
        &tensor(&query[..HEADS * WIDTH], (1, HEADS, WIDTH), DType::BF16),
        Window {
            plane: Plane {
                records: &window,
                table: &window_table,
                format: Format::Bf16,
                layout: Layout::Entry,
            },
            size: WINDOW,
        },
        Some(Selected {
            plane: Plane {
                records: &compressed,
                table: &compressed_table,
                format: Format::Bf16,
                layout: Layout::Entry,
            },
            selection: &int(vec![-1_i32, -1, -1], (1, 3)),
        }),
        (&int(vec![0_u32], 1), &int(vec![-1_i32], 1)),
        &tensor(&sink, HEADS, DType::F32),
        scale,
    )
    .unwrap();
    assert!(values(&padding).iter().all(|value| *value == 0.0));
}

#[test]
fn the_index_scores_weight_their_heads_and_keep_to_the_candidates() {
    const HEADS: usize = 2;
    const WIDTH: usize = 32;
    const GROUPS: usize = 4;
    let keys_values = sequence(3 * GROUPS * WIDTH, 9);
    let keys = tensor(&keys_values, (3, GROUPS, WIDTH), DType::BF16);
    let table = int(vec![2_u32, 0, 1], (1, 3));
    let query = sequence(HEADS * WIDTH, 10);
    let weights = [0.75_f32, -0.5];
    let plane = Plane {
        records: &keys,
        table: &table,
        format: Format::Bf16,
        layout: Layout::Page,
    };
    let score = |group: usize| -> f32 {
        let slot = [2, 0, 1][group / GROUPS] * GROUPS + group % GROUPS;
        let key = &keys_values[slot * WIDTH..(slot + 1) * WIDTH];
        (0..HEADS)
            .map(|head| {
                let q = &query[head * WIDTH..(head + 1) * WIDTH];
                let dot: f32 = q.iter().zip(key).map(|(a, b)| bf16(*a) * bf16(*b)).sum();
                dot.max(0.0) * weights[head]
            })
            .sum()
    };
    let call = |candidates: Option<(&Tensor, u32)>| {
        values(
            &functional::index_scores(
                &tensor(&query, (1, HEADS, WIDTH), DType::BF16),
                plane,
                &int(vec![0_u32], 1),
                &int(vec![9_u32], 1),
                Scoring {
                    weights: Some(&tensor(&weights, (1, HEADS), DType::F32)),
                    query: Format::Bf16,
                    candidates,
                },
            )
            .unwrap(),
        )
    };

    // Dense: the twelve groups of the table, the nine visible ones scored.
    let dense = call(None);
    let expected: Vec<f32> = (0..12)
        .map(|group| {
            if group < 9 {
                score(group)
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    assert_scores(&dense, &expected);

    // Restricted to blocks of two groups: blocks 3 and 1 (groups 6, 7, 2, 3) and none.
    let candidates = int(vec![3_i32, 1, -1], (1, 3));
    let restricted = call(Some((&candidates, 2)));
    let expected = [
        score(6),
        score(7),
        score(2),
        score(3),
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    assert_scores(&restricted, &expected);
}

#[test]
fn the_selection_keeps_the_best_lower_columns_first_and_pins_the_newest_block() {
    let inf = f32::NEG_INFINITY;
    let scores = tensor(
        &[
            1.0, 3.0, 2.0, 3.0, 0.5, inf, inf, inf, //
            0.1, 0.2, inf, inf, inf, inf, inf, inf,
        ],
        (2, 8),
        DType::F32,
    );
    let visible = int(vec![5_u32, 2], 2);

    // Three kept: the scores strictly above the third by column, then the ties at it by column;
    // a row of two finite scores keeps a `-1` for the third.
    let kept = functional::index_select(&scores, &visible, None, 1, 3).unwrap();
    assert_eq!(
        kept.to_vec2::<i32>().unwrap(),
        [vec![1, 3, 2], vec![0, 1, -1]]
    );

    // Blocks of two: each row's newest visible block (row 0's group 4, block 2; row 1's group
    // 1, block 0) ranks first; row 0's tie between blocks 0 and 1 keeps block 0.
    let blocks = functional::index_select(&scores, &visible, None, 2, 2).unwrap();
    assert_eq!(blocks.to_vec2::<i32>().unwrap(), [vec![2, 0], vec![0, -1]]);

    // Mapped through candidate blocks of four groups: columns 1 and 3 of block 7.
    let candidates = int(vec![7_i32, 2, 9, 0], (2, 2));
    let mapped = functional::index_select(&scores, &visible, Some((&candidates, 4)), 1, 2).unwrap();
    assert_eq!(
        mapped.to_vec2::<i32>().unwrap(),
        [vec![29, 31], vec![37, 36]]
    );
}

#[test]
fn a_trailing_rotation_turns_only_the_trailing_pairs_and_its_inverse_turns_them_back() {
    const WIDTH: usize = 8;
    let x = sequence(2 * 3 * WIDTH, 11);
    let rotated = tensor(&x, (2, 3, WIDTH), DType::F32);
    let (cos, sin) = (
        tensor(&[0.6, 0.8, 1.0, 0.0], (2, 2), DType::F32),
        tensor(&[0.8, -0.6, 0.0, 1.0], (2, 2), DType::F32),
    );

    functional::rotary_trailing_(&rotated, &cos, &sin, false).unwrap();
    let turned = values(&rotated);
    for token in 0..2 {
        for head in 0..3 {
            let at = (token * 3 + head) * WIDTH;
            assert_eq!(
                turned[at..at + 4],
                x[at..at + 4],
                "the leading channels stay"
            );
            for pair in 0..2 {
                let (c, s) = (
                    [[0.6, 0.8], [1.0, 0.0]][token][pair],
                    [[0.8, -0.6], [0.0, 1.0]][token][pair],
                );
                let (a, b) = (x[at + 4 + 2 * pair], x[at + 5 + 2 * pair]);
                assert_close(
                    &turned[at + 4 + 2 * pair..at + 6 + 2 * pair],
                    &[a * c - b * s, b * c + a * s],
                    F32_TOLERANCE,
                );
            }
        }
    }

    functional::rotary_trailing_(&rotated, &cos, &sin, true).unwrap();
    assert_close(&values(&rotated), &x, F32_TOLERANCE);
}

/// One layer's launches at the shapes of a 16-head rank of a 64-head latent attention (512
/// channels, 64 rotated, a window of 128 in pages of 64, NVFP4 compressed entries two tokens
/// apiece, MXFP4 index-key pages, 32 index heads of 128), over a decode step of 32 rows of
/// one token: every device call passes its operator's check and states the key its package
/// declares.
#[test]
fn a_layer_states_its_kernels_at_the_models_shapes() {
    const TOKENS: usize = 32;
    let (window_pages, compressed_pages) = (64, 32);
    let column = |dtype| meta(TOKENS, dtype);
    let (positions, rows) = (column(DType::I32), column(DType::U32));
    let window = meta((TOKENS * window_pages, 64, 528), DType::U8);
    let window_table = meta((TOKENS, window_pages), DType::U32);
    let compressed = meta((TOKENS * compressed_pages, 32, 288), DType::U8);
    let compressed_table = meta((TOKENS, compressed_pages), DType::U32);
    let keys = meta((TOKENS * compressed_pages, 32, 68), DType::U8);
    let inv_freq = meta(32, DType::F32);
    let (cos, sin) = (
        meta((TOKENS, 32), DType::F32),
        meta((TOKENS, 32), DType::F32),
    );
    let plane = |records, table, format, layout| Plane {
        records,
        table,
        format,
        layout,
    };

    let (stated, Recording { launches, .. }) = record(&[], || {
        // The window write, the compression of the state rows and the index keys' write.
        functional::record_write_(
            &meta((TOKENS, 512), DType::BF16),
            RecordRow {
                norm: Some((&meta(512, DType::BF16), 1e-20)),
                inv_freq: Some(&inv_freq),
                ratio: 1,
            },
            plane(&window, &window_table, Format::Mxfp8, Layout::Entry),
            &positions,
            &rows,
        )?;
        let ring = meta((TOKENS, 8, 1024), DType::F32);
        let (slots, seen) = (column(DType::U32), column(DType::U32));
        let offsets = meta(TOKENS + 1, DType::U32);
        let compression = functional::latent_compress(
            &meta((TOKENS, 1024), DType::F32),
            (&meta(512, DType::BF16), 1e-20),
            Ring {
                rows: &ring,
                slots: &slots,
                query_offsets: &offsets,
                seq_lens: &seen,
            },
            &positions,
            2,
        )?;
        functional::record_write_(
            &meta((TOKENS, 128), DType::BF16),
            RecordRow {
                norm: Some((&meta(128, DType::BF16), 1e-20)),
                inv_freq: Some(&inv_freq),
                ratio: 2,
            },
            plane(&keys, &compressed_table, Format::Mxfp4, Layout::Page),
            &compression.entries,
            &rows,
        )?;

        // The indexer's queries, scores and selection, then the attention's.
        let index_query = meta((TOKENS, 32, 128), DType::BF16);
        functional::rotary_trailing_(&index_query, &cos, &sin, false)?;
        let scores = functional::index_scores(
            &index_query,
            plane(&keys, &compressed_table, Format::Mxfp4, Layout::Page),
            &rows,
            &column(DType::U32),
            Scoring {
                weights: Some(&meta((TOKENS, 32), DType::F32)),
                query: Format::Mxfp4,
                candidates: None,
            },
        )?;
        let selection = functional::index_select(&scores, &column(DType::U32), None, 1, 512)?;
        let candidates = functional::index_select(&scores, &column(DType::U32), None, 8, 64)?;
        let query = meta((TOKENS, 16, 512), DType::BF16);
        functional::rotary_trailing_(&query, &cos, &sin, false)?;
        let attended = functional::latent_attention(
            &query,
            Window {
                plane: plane(&window, &window_table, Format::Mxfp8, Layout::Entry),
                size: 128,
            },
            Some(Selected {
                plane: plane(&compressed, &compressed_table, Format::Nvfp4, Layout::Entry),
                selection: &selection,
            }),
            (&rows, &positions),
            &meta(16, DType::F32),
            512_f32.powf(-0.5),
        )?;
        functional::rotary_trailing_(&attended, &cos, &sin, true)?;
        Ok((attended.dims().to_vec(), candidates.dims().to_vec()))
    });
    assert_eq!(stated.unwrap(), (vec![TOKENS, 16, 512], vec![TOKENS, 64]));

    let expected = [
        (
            "record_write",
            "dtype_bf16_width_512_norm_true_rotated_64_ratio_1_format_mxfp8_layout_entry_entries_64",
        ),
        ("latent_compress", "ratio_2_width_512_ring_8_gain_bf16"),
        (
            "record_write",
            "dtype_bf16_width_128_norm_true_rotated_64_ratio_2_format_mxfp4_layout_page_entries_32",
        ),
        (
            "rope_apply",
            "dtype_bf16_output_bf16_head_dim_128_heads_32_rotation_interleaved_rows_8_block_128_\
             pdl_false_rotated_64_first_64_inplace_true",
        ),
        (
            "index_scores",
            "dtype_bf16_keys_mxfp4_heads_32_width_128_groups_32_block_256_weighted_true_query_mxfp4",
        ),
        ("index_select", "pool_1_candidates_0_tile_4096"),
        ("index_select", "pool_8_candidates_0_tile_4096"),
        (
            "rope_apply",
            "dtype_bf16_output_bf16_head_dim_512_heads_16_rotation_interleaved_rows_8_block_128_\
             pdl_false_rotated_64_first_448_inplace_true",
        ),
        (
            "latent_attention",
            "dtype_bf16_heads_16_head_dim_512_window_format_mxfp8_window_128_page_64_compressed_\
             nvfp4_entries_32_columns_512_block_16",
        ),
        (
            "rope_apply",
            "dtype_bf16_output_bf16_head_dim_512_heads_16_rotation_interleaved_rows_8_block_128_\
             pdl_false_rotated_64_first_448_inverse_true_inplace_true",
        ),
    ];
    let launched: Vec<(&str, &str)> = launches
        .iter()
        .map(|(operator, key)| (*operator, key.as_str()))
        .collect();
    assert_eq!(launched, expected);
}
