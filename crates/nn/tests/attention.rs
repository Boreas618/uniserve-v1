//! Attention and the key/value cache: the functions' arithmetic on CPU tensors against the
//! softmax formula written out in loops, the cache write read back by the attention that follows
//! it, and the kernels each function and the `Attention` module state under a meta context.
//!
//! Nothing here touches a GPU.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

mod support;

use std::path::Path;

use nn::collective::Split;
use nn::family::SequenceSplit;
use nn::functional::{self, WeightOrder};
use nn::modules::{Attention, AttentionConfig, QkNorm, QkvBias, QkvLinearConfig, QkvNames};
use nn::{
    Checkpoint, CpuWeights, DType, Device, KvCache, KvPlanes, NO_SLOT, PhaseKind, Tensor,
    TokenBatch, WeightLoader,
};
use support::{
    assert_close, encode, meta, operators, recorded, recorded_split, sequence, shard, tensor,
    values,
};

/// The tolerance of an FP32 result against a reference accumulated in another order.
const F32_TOLERANCE: f32 = 2e-5;

/// The softmax attention of `query` (`[queries][heads][dim]`, flat) over `key` and `value`
/// (`[keys][kv_heads][dim]`, flat): each query head reads the key head of its group, and query
/// `i` weighs the keys `visible(i, j)` names. A query that sees no key has a row of zeros.
fn attention(
    (query, key, value): (&[f32], &[f32], &[f32]),
    (heads, kv_heads, dim): (usize, usize, usize),
    scale: f32,
    visible: impl Fn(usize, usize) -> bool,
) -> Vec<f32> {
    let queries = query.len() / (heads * dim);
    let keys = key.len() / (kv_heads * dim);
    let group = heads / kv_heads;
    let mut out = vec![0.0; query.len()];

    for i in 0..queries {
        for head in 0..heads {
            let kv_head = head / group;
            let q = &query[(i * heads + head) * dim..][..dim];
            let seen: Vec<usize> = (0..keys).filter(|j| visible(i, *j)).collect();

            let scores: Vec<f32> = seen
                .iter()
                .map(|j| {
                    let k = &key[(j * kv_heads + kv_head) * dim..][..dim];
                    scale * q.iter().zip(k).map(|(q, k)| q * k).sum::<f32>()
                })
                .collect();
            let peak = scores.iter().copied().fold(f32::MIN, f32::max);
            let weights: Vec<f32> = scores.iter().map(|score| (score - peak).exp()).collect();
            let total: f32 = weights.iter().sum();

            for (weight, j) in weights.iter().zip(&seen) {
                let v = &value[(j * kv_heads + kv_head) * dim..][..dim];
                for (out, v) in out[(i * heads + head) * dim..][..dim].iter_mut().zip(v) {
                    *out += weight * v / total;
                }
            }
        }
    }
    out
}

/// A CPU column of U32 `values`.
fn column(values: &[u32]) -> Tensor {
    Tensor::new(values, &Device::Cpu).unwrap()
}

// ---- Dense attention --------------------------------------------------------------------------

#[test]
fn dense_attention_weighs_the_visible_keys_of_one_sequence() {
    let (heads, kv_heads, dim) = (4, 2, 8);
    let (query, key, value) = (
        sequence(3 * heads * dim, 1),
        sequence(5 * kv_heads * dim, 2),
        sequence(5 * kv_heads * dim, 3),
    );
    let tensors = (
        tensor(&query, (3, heads, dim), DType::F32),
        tensor(&key, (5, kv_heads, dim), DType::F32),
        tensor(&value, (5, kv_heads, dim), DType::F32),
    );
    let attend = |causal, window| {
        functional::dense_attention(&tensors.0, &tensors.1, &tensors.2, 0.5, causal, window)
    };
    let reference = |visible: &dyn Fn(usize, usize) -> bool| {
        attention((&query, &key, &value), (heads, kv_heads, dim), 0.5, visible)
    };

    // Without a mask every query sees every key: a cross-attention of three queries to five
    // keys.
    let dense = attend(false, None).unwrap();
    assert_eq!(dense.dims(), [3, heads, dim]);
    assert_close(&values(&dense), &reference(&|_, _| true), F32_TOLERANCE);

    // Under the causal mask a query sees the keys at its own row and before it, and under a
    // window at most that many keys before its own.
    let causal = attend(true, None).unwrap();
    assert_close(&values(&causal), &reference(&|i, j| j <= i), F32_TOLERANCE);

    let windowed = attend(true, Some(1)).unwrap();
    assert_close(
        &values(&windowed),
        &reference(&|i, j| j <= i && j + 1 >= i),
        F32_TOLERANCE,
    );

    // A window reads the causal mask only, and a scale of zero attends nothing.
    let refused = attend(false, Some(1)).unwrap_err().to_string();
    assert!(refused.contains("causal mask"), "{refused}");
    assert!(
        functional::dense_attention(&tensors.0, &tensors.1, &tensors.2, 0.0, false, None).is_err()
    );
}

#[test]
fn dense_attention_rounds_once_into_the_querys_element_type() {
    let (heads, dim) = (2, 8);
    let (query, key) = (sequence(2 * heads * dim, 4), sequence(3 * heads * dim, 5));
    let attended = functional::dense_attention(
        &tensor(&query, (2, heads, dim), DType::BF16),
        &tensor(&key, (3, heads, dim), DType::BF16),
        &tensor(&key, (3, heads, dim), DType::BF16),
        0.25,
        false,
        None,
    )
    .unwrap();
    assert_eq!(attended.dtype(), DType::BF16);

    // The formula over the rounded inputs, in FP32, rounded once.
    let rounded = |x: &[f32], rows| values(&tensor(x, (rows, heads, dim), DType::BF16));
    let (query, key) = (rounded(&query, 2), rounded(&key, 3));
    let expected = attention((&query, &key, &key), (heads, heads, dim), 0.25, |_, _| true);
    assert_close(
        &values(&attended),
        &values(&tensor(&expected, (2, heads, dim), DType::BF16)),
        1.0 / 128.0,
    );
}

#[test]
fn dense_attention_varlen_keeps_each_sequence_to_its_own_keys() {
    let (heads, kv_heads, dim) = (2, 1, 8);
    // Two sequences: queries 0..2 and 2..5 against keys 0..3 and 3..5; the sixth query row is
    // past the last offset and belongs to no sequence.
    let (query, key, value) = (
        sequence(6 * heads * dim, 6),
        sequence(5 * kv_heads * dim, 7),
        sequence(5 * kv_heads * dim, 8),
    );
    let tensors = (
        tensor(&query, (6, heads, dim), DType::F32),
        tensor(&key, (5, kv_heads, dim), DType::F32),
        tensor(&value, (5, kv_heads, dim), DType::F32),
    );
    let (cu_q, cu_k) = (column(&[0, 2, 5]), column(&[0, 3, 5]));
    let attend = |causal, window| {
        functional::dense_attention_varlen(
            &tensors.0,
            &tensors.1,
            &tensors.2,
            &cu_q,
            &cu_k,
            Some(3),
            5,
            1.0,
            causal,
            window,
        )
    };

    // The sequence and the place within it of each query and key row.
    let query_place = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0)];
    let key_place = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)];
    let reference = |visible: &dyn Fn(usize, usize) -> bool| {
        attention(
            (&query, &key, &value),
            (heads, kv_heads, dim),
            1.0,
            |i, j| query_place[i].0 == key_place[j].0 && visible(query_place[i].1, key_place[j].1),
        )
    };

    let dense = attend(false, None).unwrap();
    assert_close(&values(&dense), &reference(&|_, _| true), F32_TOLERANCE);

    // The row of no sequence saw nothing.
    assert!(values(&dense)[5 * heads * dim..].iter().all(|x| *x == 0.0));

    // The causal mask counts both places from their sequence's start.
    let causal = attend(true, Some(1)).unwrap();
    assert_close(
        &values(&causal),
        &reference(&|i, j| j <= i && j + 1 >= i),
        F32_TOLERANCE,
    );

    // The key bound covers every key passed, and the tables are two u32 vectors of one length.
    let short = functional::dense_attention_varlen(
        &tensors.0, &tensors.1, &tensors.2, &cu_q, &cu_k, None, 4, 1.0, false, None,
    );
    assert!(short.unwrap_err().to_string().contains("4-key bound"));

    let signed = Tensor::new(&[0_i64, 2, 5], &Device::Cpu).unwrap();
    let refused = functional::dense_attention_varlen(
        &tensors.0, &tensors.1, &tensors.2, &signed, &cu_k, None, 5, 1.0, false, None,
    );
    assert!(refused.unwrap_err().to_string().contains("cu_seqlens_q"));
}

#[test]
fn dense_attention_states_one_kernel_per_sequence_form() {
    let query = meta((16, 32, 128), DType::BF16);
    let key = meta((16, 4, 128), DType::BF16);
    let offsets = meta(3, DType::U32);

    let (single, one) =
        recorded(|| functional::dense_attention(&query, &key, &key, 0.088, false, None));
    assert_eq!(single.unwrap().dims(), [16, 32, 128]);
    assert_eq!(
        one,
        [(
            "dense_attention",
            "dtype_bf16_kv_bf16_out_bf16_head_dim_128_q_heads_32_kv_heads_4_mask_dense_window_false\
             _cu_seq_lens_false"
                .to_owned()
        )]
    );

    // Sequence tables select another kernel, whose key the scheduled batch path always had.
    let (packed, two) = recorded(|| {
        functional::dense_attention_varlen(
            &query,
            &key,
            &key,
            &offsets,
            &offsets,
            Some(8),
            16,
            0.088,
            true,
            Some(1_024),
        )
    });
    packed.unwrap();
    assert_eq!(
        two,
        [(
            "dense_attention",
            "dtype_bf16_kv_bf16_out_bf16_head_dim_128_q_heads_32_kv_heads_4_mask_causal_window_true"
                .to_owned()
        )]
    );

    // Query heads that do not group the key heads are refused before a launch.
    let odd = meta((16, 5, 128), DType::BF16);
    let (refused, none) =
        recorded(|| functional::dense_attention(&query, &odd, &odd, 0.088, false, None));
    assert!(refused.is_err());
    assert!(none.is_empty());
}

// ---- The cache write and paged attention --------------------------------------------------------

/// The heads of the cache tests: four query heads over two key heads of eight elements.
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const DIM: usize = 8;

/// The tokens of one page and the pages of the cache.
const PAGE: u32 = 4;
const PAGES: usize = 3;

/// A CPU cache of [`PAGES`] zeroed pages of `dtype` in one layer, with the page table `table`
/// of `rows` rows; E4M3 planes have their page scales.
fn cache(dtype: DType, table: &[u32], rows: usize) -> KvCache {
    let zeros = |shape: (usize, usize, usize, usize), dtype| {
        Tensor::zeros(shape, dtype, &Device::Cpu).unwrap()
    };
    let plane = || zeros((PAGES, PAGE as usize, KV_HEADS, DIM), dtype);
    let per_page = |dtype| zeros((PAGES, 1, 1, 1), dtype);
    let scale = || (dtype == DType::F8E4M3).then(|| per_page(DType::F32));

    KvCache {
        block_tables: column(table).reshape((rows, table.len() / rows)).unwrap(),
        page_size: PAGE,
        layers: vec![KvPlanes {
            key: plane(),
            value: plane(),
            key_initialized: per_page(DType::U8),
            value_initialized: per_page(DType::U8),
            key_scale: scale(),
            value_scale: scale(),
            index_keys: None,
        }],
        workspace: Tensor::zeros(64, DType::U8, &Device::Cpu).unwrap(),
    }
}

/// A CPU batch of `phase` whose tokens take `slots`, in rows of `seq_lens` keys packed at
/// `offsets`.
fn batch(phase: PhaseKind, slots: &[u32], seq_lens: &[u32], offsets: &[u32]) -> TokenBatch {
    TokenBatch {
        positions: Tensor::zeros(slots.len(), DType::I32, &Device::Cpu).unwrap(),
        slot_mapping: column(slots),
        seq_lens: column(seq_lens),
        query_offsets: column(offsets),
        logits_rows: column(&vec![0; seq_lens.len()]),
        phase,
        page_size: PAGE,
        num_tokens: slots.len() as u32,
        max_kv_len: 64,
        draft_lens: None,
        rope_positions: None,
        live_tokens: None,
        token_ids: None,
        ngram_context: None,
    }
}

/// `tokens` tokens of `heads` heads of [`DIM`] values, different for each `seed`.
fn heads(tokens: usize, heads: usize, seed: u32) -> Vec<f32> {
    sequence(tokens * heads * DIM, seed)
}

/// Rows `rows` (tokens) of a flat `[tokens][heads][DIM]` projection.
fn token_rows(x: &[f32], heads: usize, rows: std::ops::Range<usize>) -> Vec<f32> {
    x[rows.start * heads * DIM..rows.end * heads * DIM].to_vec()
}

/// What was written is what is attended: a prefill writes two rows' keys and values into pages
/// in another order than the rows', and each query attends its own row's keys up to its own
/// position; a decode then appends one token to each row and attends the whole of it.
#[test]
fn paged_attention_reads_the_rows_kv_write_stored() {
    // Row 0 holds six keys in pages 2 then 0, row 1 three keys in page 1. The tenth token is
    // the bucket's padding: it has no slot and belongs to no row.
    let cache = cache(DType::F32, &[2, 0, 1, 1], 2);
    let prefill = batch(
        PhaseKind::Prefill,
        &[8, 9, 10, 11, 0, 1, 4, 5, 6, NO_SLOT],
        &[6, 3],
        &[0, 6, 9],
    );
    let (query, key, value) = (
        heads(10, HEADS, 1),
        heads(10, KV_HEADS, 2),
        heads(10, KV_HEADS, 3),
    );
    let stored = |x: &[f32]| tensor(x, (10, KV_HEADS, DIM), DType::F32);

    functional::kv_write_(&cache, 0, &prefill, &stored(&key), &stored(&value)).unwrap();
    let attended = functional::paged_attention(
        &tensor(&query, (10, HEADS, DIM), DType::F32),
        &prefill,
        &cache,
        0,
        0.5,
        None,
        64,
    )
    .unwrap();
    assert_eq!(attended.dims(), [10, HEADS, DIM]);

    // Each row is a causal attention over its own tokens' keys.
    let shape = (HEADS, KV_HEADS, DIM);
    let row = |tokens: std::ops::Range<usize>| {
        attention(
            (
                &token_rows(&query, HEADS, tokens.clone()),
                &token_rows(&key, KV_HEADS, tokens.clone()),
                &token_rows(&value, KV_HEADS, tokens),
            ),
            shape,
            0.5,
            |i, j| j <= i,
        )
    };
    let mut expected = row(0..6);
    expected.extend(row(6..9));
    expected.extend(vec![0.0; HEADS * DIM]);
    assert_close(&values(&attended), &expected, F32_TOLERANCE);

    // Every page that received a row is flagged, in both planes, and the padding token wrote
    // nothing: the last slot of page 1 is as the cache's owner left it.
    let planes = &cache.layers[0];
    assert_eq!(values(&planes.key_initialized), [1.0, 1.0, 1.0]);
    assert_eq!(values(&planes.value_initialized), [1.0, 1.0, 1.0]);
    let last_slot = values(&planes.key)[7 * KV_HEADS * DIM..][..KV_HEADS * DIM].to_vec();
    assert!(last_slot.iter().all(|x| *x == 0.0));

    // A decode appends one token to each row and its query sees the whole row; under a window
    // of two it sees its own key and the two before it.
    let decode = batch(PhaseKind::Decode, &[2, 7], &[7, 4], &[0, 1, 2]);
    let (next_query, next_key, next_value) = (
        heads(2, HEADS, 4),
        heads(2, KV_HEADS, 5),
        heads(2, KV_HEADS, 6),
    );
    let appended = |x: &[f32]| tensor(x, (2, KV_HEADS, DIM), DType::F32);
    functional::kv_write_(
        &cache,
        0,
        &decode,
        &appended(&next_key),
        &appended(&next_value),
    )
    .unwrap();

    let step = |window| {
        functional::paged_attention(
            &tensor(&next_query, (2, HEADS, DIM), DType::F32),
            &decode,
            &cache,
            0,
            0.5,
            window,
            64,
        )
        .unwrap()
    };
    let row = |index: usize, tokens: std::ops::Range<usize>, window: usize| {
        let with = |old: &[f32], new: &[f32]| {
            let mut all = token_rows(old, KV_HEADS, tokens.clone());
            all.extend(token_rows(new, KV_HEADS, index..index + 1));
            all
        };
        let keys = tokens.len() + 1;
        attention(
            (
                &token_rows(&next_query, HEADS, index..index + 1),
                &with(&key, &next_key),
                &with(&value, &next_value),
            ),
            shape,
            0.5,
            |_, j| j + window + 1 >= keys,
        )
    };

    let mut expected = row(0, 0..6, usize::MAX / 2);
    expected.extend(row(1, 6..9, usize::MAX / 2));
    assert_close(&values(&step(None)), &expected, F32_TOLERANCE);

    let mut expected = row(0, 0..6, 2);
    expected.extend(row(1, 6..9, 2));
    assert_close(&values(&step(Some(2))), &expected, F32_TOLERANCE);
}

/// An E4M3 cache holds each page divided by its own scale: the write derives the scale from
/// the rows it brings, grows it when later rows are larger and re-encodes what the page held,
/// and the attention multiplies the scale back.
#[test]
fn an_fp8_cache_scales_each_page_by_the_rows_it_holds() {
    let cache = cache(DType::F8E4M3, &[1, 0], 1);
    let planes = &cache.layers[0];
    let row = KV_HEADS * DIM;

    // The largest magnitude a page's rows may be off by after one E4M3 rounding at `scale`:
    // three mantissa bits, and a subnormal step of 2⁻⁹ in the page's units.
    let bound = |x: f32, scale: f32| x.abs() / 16.0 + scale / 1024.0;
    let decoded = |plane: &Tensor, scale: &Tensor| {
        let scales = values(scale);
        values(plane)
            .chunks(PAGE as usize * row)
            .zip(&scales)
            .flat_map(|(page, scale)| page.iter().map(|x| x * scale).collect::<Vec<f32>>())
            .collect::<Vec<f32>>()
    };

    // Two tokens into page 1, and one without a slot whose magnitude must not reach any scale.
    let mut first = heads(3, KV_HEADS, 1);
    first[2 * row] = 1.0e4;
    let stored = tensor(&first, (3, KV_HEADS, DIM), DType::F32);
    let prefill = batch(PhaseKind::Prefill, &[4, 5, NO_SLOT], &[2], &[0, 2]);
    functional::kv_write_(&cache, 0, &prefill, &stored, &stored).unwrap();

    let largest = |x: &[f32]| x.iter().fold(0.0_f32, |largest, x| largest.max(x.abs()));
    let first_scale = largest(&first[..2 * row]) / 448.0;
    assert_eq!(
        values(planes.key_scale.as_ref().unwrap()),
        [0.0, first_scale, 0.0]
    );
    assert_eq!(values(&planes.key_initialized), [0.0, 1.0, 0.0]);

    let held = decoded(&planes.key, planes.key_scale.as_ref().unwrap());
    for (held, written) in held[4 * row..6 * row].iter().zip(&first) {
        assert!((held - written).abs() <= bound(*written, first_scale));
    }

    // A third token of eight times the magnitude grows the page's scale; the two rows it
    // already held keep their decoded values, within a second rounding.
    let second: Vec<f32> = heads(1, KV_HEADS, 2).iter().map(|x| x * 8.0).collect();
    let appended = tensor(&second, (1, KV_HEADS, DIM), DType::F32);
    let decode = batch(PhaseKind::Decode, &[6], &[3], &[0, 1]);
    functional::kv_write_(&cache, 0, &decode, &appended, &appended).unwrap();

    let grown = largest(&second) / 448.0;
    assert!(grown > first_scale);
    assert_eq!(
        values(planes.key_scale.as_ref().unwrap()),
        [0.0, grown, 0.0]
    );

    let held = decoded(&planes.key, planes.key_scale.as_ref().unwrap());
    for (held, written) in held[4 * row..6 * row].iter().zip(&first) {
        assert!((held - written).abs() <= bound(*written, first_scale) + bound(*written, grown));
    }
    for (held, written) in held[6 * row..7 * row].iter().zip(&second) {
        assert!((held - written).abs() <= bound(*written, grown));
    }

    // The attention reads the page through its scale: the formula over the decoded rows.
    let query = heads(1, HEADS, 3);
    let attended = functional::paged_attention(
        &tensor(&query, (1, HEADS, DIM), DType::F32),
        &decode,
        &cache,
        0,
        0.5,
        None,
        64,
    )
    .unwrap();
    let keys = held[4 * row..7 * row].to_vec();
    let expected = attention(
        (&query, &keys, &keys),
        (HEADS, KV_HEADS, DIM),
        0.5,
        |_, _| true,
    );
    assert_close(&values(&attended), &expected, F32_TOLERANCE);
}

/// A dense cache of another type than the projections holds each element rounded once into its
/// type, to nearest even, with ties to the even neighbour: a BF16 model's F16 cache, and an F32
/// model's BF16 one. Every source value below is exact in the source type, and its expected
/// value is the nearest one of the cache's type, worked out by hand.
#[test]
fn a_dense_cache_of_another_type_holds_each_element_at_its_nearest_value() {
    let row = KV_HEADS * DIM;
    // F16's smallest subnormal, and BF16's unit in the last place at one.
    let tiny = 2f32.powi(-24);
    let unit = 2f32.powi(-7);

    // Values F16 holds exactly stay as they are; below F16's normal range its subnormals are
    // coarser than BF16, and a source between two of them takes the nearer.
    let bf16_into_f16 = [
        (1.5, 1.5),
        (-2.25, -2.25),
        (0.75 * tiny, tiny),
        (0.25 * tiny, 0.0),
        (0.5 * tiny, 0.0),
        (1.5 * tiny, 2.0 * tiny),
        (-1.5 * tiny, -2.0 * tiny),
        (2.5 * tiny, 2.0 * tiny),
    ];
    // BF16 keeps seven fraction bits: a tie at one goes to one, a tie at one unit up goes to two
    // units up, and anything off the middle goes to the nearer side.
    let f32_into_bf16 = [
        (1.0 + unit / 2.0, 1.0),
        (1.0 + 1.5 * unit, 1.0 + 2.0 * unit),
        (1.0 + unit / 4.0, 1.0),
        (1.0 + 0.75 * unit, 1.0 + unit),
        (-(1.0 + unit / 2.0), -1.0),
        (0.5, 0.5),
    ];

    for (source, plane, pairs) in [
        (DType::BF16, DType::F16, &bf16_into_f16[..]),
        (DType::F32, DType::BF16, &f32_into_bf16[..]),
    ] {
        let cache = cache(plane, &[0, 1], 1);
        let planes = &cache.layers[0];

        // Two tokens, into page 1 then page 0; the values are the keys negated.
        let keys: Vec<f32> = pairs
            .iter()
            .map(|pair| pair.0)
            .cycle()
            .take(2 * row)
            .collect();
        let expected: Vec<f32> = pairs
            .iter()
            .map(|pair| pair.1)
            .cycle()
            .take(2 * row)
            .collect();
        let negated = |x: &[f32]| x.iter().map(|x| -x).collect::<Vec<f32>>();
        let prefill = batch(PhaseKind::Prefill, &[5, 2], &[2], &[0, 2]);
        functional::kv_write_(
            &cache,
            0,
            &prefill,
            &tensor(&keys, (2, KV_HEADS, DIM), source),
            &tensor(&negated(&keys), (2, KV_HEADS, DIM), source),
        )
        .unwrap();

        // The planes keep their type, each token's row is at its slot, and every other slot is
        // as the cache's owner left it.
        assert_eq!((planes.key.dtype(), planes.value.dtype()), (plane, plane));
        let mut stored = vec![0.0; PAGES * PAGE as usize * row];
        stored[5 * row..6 * row].copy_from_slice(&expected[..row]);
        stored[2 * row..3 * row].copy_from_slice(&expected[row..]);
        assert_eq!(values(&planes.key), stored, "{source:?} into {plane:?}");
        assert_eq!(values(&planes.value), negated(&stored));
        assert_eq!(values(&planes.key_initialized), [1.0, 1.0, 0.0]);
        assert_eq!(values(&planes.value_initialized), [1.0, 1.0, 0.0]);
    }
}

#[test]
fn the_cache_functions_refuse_what_no_kernel_reads() {
    let prefill = batch(PhaseKind::Prefill, &[0, 1], &[2], &[0, 2]);
    let query = tensor(&heads(2, HEADS, 1), (2, HEADS, DIM), DType::F32);
    let key = tensor(&heads(2, KV_HEADS, 2), (2, KV_HEADS, DIM), DType::F32);
    let attend = |cache: &KvCache, batch: &TokenBatch| {
        functional::paged_attention(&query, batch, cache, 0, 0.5, None, 64)
            .unwrap_err()
            .to_string()
    };

    // An E4M3 plane is read through one scale per page: none, one scalar, a table short of the
    // plane's pages and a table of another type are all refused, and so is a scale on a plane
    // of the projections' own type.
    let fp8 = || cache(DType::F8E4M3, &[0], 1);
    let scale = |pages: usize, dtype| Tensor::zeros((pages, 1, 1, 1), dtype, &Device::Cpu).unwrap();

    let mut unscaled = fp8();
    unscaled.layers[0].key_scale = None;
    assert!(attend(&unscaled, &prefill).contains("per-page scale"));

    for wrong in [
        scale(1, DType::F32),
        scale(PAGES - 1, DType::F32),
        scale(PAGES, DType::BF16),
    ] {
        let mut cache = fp8();
        cache.layers[0].value_scale = Some(wrong);
        assert!(attend(&cache, &prefill).contains("f32 page scales"));
    }

    let mut scaled = cache(DType::F32, &[0], 1);
    scaled.layers[0].key_scale = Some(scale(PAGES, DType::F32));
    assert!(attend(&scaled, &prefill).contains("has no scale"));

    // The page table is one u32 row per row of the batch.
    let mut flat = cache(DType::F32, &[0], 1);
    flat.block_tables = column(&[0]);
    assert!(attend(&flat, &prefill).contains("one row per row"));

    let mut signed = cache(DType::F32, &[0], 1);
    signed.block_tables = Tensor::zeros((1, 1), DType::I32, &Device::Cpu).unwrap();
    assert!(attend(&signed, &prefill).contains("u32 [1, pages]"));

    // The batch and the cache state one page size, and the key bound covers a page.
    let plain = cache(DType::F32, &[0], 1);
    let mut other = batch(PhaseKind::Prefill, &[0, 1], &[2], &[0, 2]);
    other.page_size = 8;
    assert!(attend(&plain, &other).contains("tokens per page"));

    let short = functional::paged_attention(&query, &prefill, &plain, 0, 0.5, None, 2);
    assert!(short.unwrap_err().to_string().contains("shorter than"));

    // A layer the group does not hold, a strided projection, keys and values of two types, and
    // planes of two types or of a type no write fills have no write.
    assert!(functional::kv_write_(&plain, 1, &prefill, &key, &key).is_err());

    let wide = tensor(
        &heads(2, 2 * KV_HEADS, 3),
        (2, 2 * KV_HEADS, DIM),
        DType::F32,
    );
    let strided = wide.narrow(1, 0, KV_HEADS).unwrap();
    let refused = functional::kv_write_(&plain, 0, &prefill, &strided, &key);
    assert!(refused.unwrap_err().to_string().contains("strided view"));

    let half = key.to_dtype(DType::BF16).unwrap();
    let refused = functional::kv_write_(&plain, 0, &prefill, &key, &half);
    assert!(
        refused
            .unwrap_err()
            .to_string()
            .contains("one element type")
    );

    let mut mixed = cache(DType::BF16, &[0], 1);
    mixed.layers[0].value = Tensor::zeros(
        (PAGES, PAGE as usize, KV_HEADS, DIM),
        DType::F16,
        &Device::Cpu,
    )
    .unwrap();
    let mut integer = cache(DType::F32, &[0], 1);
    integer.layers[0].key = Tensor::zeros(
        (PAGES, PAGE as usize, KV_HEADS, DIM),
        DType::I32,
        &Device::Cpu,
    )
    .unwrap();
    integer.layers[0].value = integer.layers[0].key.zeros_like().unwrap();
    for cache in [mixed, integer] {
        let refused = functional::kv_write_(&cache, 0, &prefill, &key, &key);
        assert!(refused.unwrap_err().to_string().contains("in both planes"));
    }
}

/// A meta batch of `tokens` tokens in one row over 64-token pages.
fn meta_batch(phase: PhaseKind, tokens: usize) -> TokenBatch {
    TokenBatch {
        positions: meta(tokens, DType::I32),
        slot_mapping: meta(tokens, DType::U32),
        seq_lens: meta(1, DType::U32),
        query_offsets: meta(2, DType::U32),
        logits_rows: meta(1, DType::U32),
        phase,
        page_size: 64,
        num_tokens: tokens as u32,
        max_kv_len: 64,
        draft_lens: None,
        rope_positions: None,
        live_tokens: None,
        token_ids: None,
        ngram_context: None,
    }
}

/// A meta cache of one page of `kv_heads` heads of `head_dim` elements of `dtype`, in one
/// layer.
fn meta_cache(dtype: DType, kv_heads: usize, head_dim: usize) -> KvCache {
    let plane = || meta((1, 64, kv_heads, head_dim), dtype);
    let scale = || (dtype == DType::F8E4M3).then(|| meta((1, 1, 1, 1), DType::F32));

    KvCache {
        block_tables: meta((1, 1), DType::U32),
        page_size: 64,
        layers: vec![KvPlanes {
            key: plane(),
            value: plane(),
            key_initialized: meta((1, 1, 1, 1), DType::U8),
            value_initialized: meta((1, 1, 1, 1), DType::U8),
            key_scale: scale(),
            value_scale: scale(),
            index_keys: None,
        }],
        workspace: meta(1 << 21, DType::U8),
    }
}

/// The kernels a write and the attention after it state for `tokens` tokens of 32 query heads
/// over four key heads of 128, over a cache of `dtype`.
fn paged_launches(phase: PhaseKind, tokens: usize, dtype: DType) -> Vec<(&'static str, String)> {
    let (batch, cache) = (meta_batch(phase, tokens), meta_cache(dtype, 4, 128));
    let query = meta((tokens, 32, 128), DType::BF16);
    let key = meta((tokens, 4, 128), DType::BF16);

    let (attended, launches) = recorded(|| {
        functional::kv_write_(&cache, 0, &batch, &key, &key)?;
        functional::paged_attention(&query, &batch, &cache, 0, 0.088, None, 40_960)
    });
    assert_eq!(attended.unwrap().dims(), [tokens, 32, 128]);
    launches
}

#[test]
fn the_cache_writes_key_the_plane_type_and_the_reads_key_the_phase() {
    // A cache of the projections' own type is one scatter of both planes.
    let plain = paged_launches(PhaseKind::Prefill, 8, DType::BF16);
    assert_eq!(
        plain,
        [
            (
                "kv_write",
                "dtype_bf16_kv_bf16_page_64_heads_4_head_dim_128_tile_128_pdl_false".to_owned()
            ),
            (
                "paged_attention",
                "phase_prefill_dtype_bf16_kv_bf16_out_bf16_head_dim_128_page_64_q_heads_32\
                 _kv_heads_4_mask_causal_window_false_len_40960"
                    .to_owned()
            ),
        ]
    );

    // A dense cache of another type is the same one scatter, compiled for the pair of types:
    // the rounding into the cache is the kernel's store, not a launch of its own.
    let converted = paged_launches(PhaseKind::Prefill, 8, DType::F16);
    assert_eq!(
        converted[0],
        (
            "kv_write",
            "dtype_bf16_kv_f16_page_64_heads_4_head_dim_128_tile_128_pdl_false".to_owned()
        )
    );
    assert_eq!(converted.len(), 2);
    assert!(converted[1].1.contains("_dtype_bf16_kv_f16_out_bf16_"));

    // An E4M3 cache is the quantizing write, one launch per plane, and another attention
    // kernel.
    let quantized = paged_launches(PhaseKind::Prefill, 8, DType::F8E4M3);
    let scaled = "dtype_bf16_scale_f32_page_64_heads_4_head_dim_128_block_128_pdl_false";
    assert_eq!(
        quantized,
        [
            ("kv_write_scaled", scaled.to_owned()),
            ("kv_write_scaled", scaled.to_owned()),
            (
                "paged_attention",
                "phase_prefill_dtype_bf16_kv_f8e4m3_out_bf16_head_dim_128_page_64_q_heads_32\
                 _kv_heads_4_mask_causal_window_false_len_40960"
                    .to_owned()
            ),
        ]
    );

    // A prefill reads a whole sequence and a decode one token: one attention kernel each.
    let decode = paged_launches(PhaseKind::Decode, 1, DType::BF16);
    assert_eq!(
        decode[1].1,
        "phase_decode_dtype_bf16_kv_bf16_out_bf16_head_dim_128_page_64_q_heads_32_kv_heads_4\
         _mask_causal_window_false_len_40960"
    );
    assert_eq!(decode[0], plain[0]);
}

#[test]
fn the_arrival_counters_are_cleared_at_the_head_of_the_workspace() {
    // On the CPU the head of the workspace is zeroed and the rest is as it was.
    let workspace = Tensor::ones(8, DType::U8, &Device::Cpu).unwrap();
    functional::zero_arrival_counters_(&workspace).unwrap();
    assert_eq!(values(&workspace), [0.0; 8]);

    // Under a meta context the fill is one launch over the counters' megabyte.
    let workspace = meta(1 << 21, DType::U8);
    let (cleared, launches) = recorded(|| functional::zero_arrival_counters_(&workspace));
    cleared.unwrap();
    assert_eq!(operators(&launches), ["fill"]);

    assert!(functional::zero_arrival_counters_(&meta((2, 4), DType::U8)).is_err());
}

// ---- The module -------------------------------------------------------------------------------

/// The width of the rows the module tests attend.
const HIDDEN: usize = 16;

/// The variance floor of the module tests' q/k normalization.
const EPS: f32 = 1e-6;

/// The weights of an attention module `attn` of [`HEADS`] query heads over [`KV_HEADS`] key
/// heads of [`DIM`]: separate projections, per-head norm gains and an output projection. A
/// gated module's query projection is twice as wide.
fn attention_weights(gated: bool) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    let query = HEADS * DIM * if gated { 2 } else { 1 };
    let matrix =
        |rows: usize, columns: usize, seed| (vec![rows, columns], sequence(rows * columns, seed));
    let named = |name, (shape, values): (Vec<usize>, Vec<f32>)| (name, shape, values);

    vec![
        named("attn.q_proj.weight", matrix(query, HIDDEN, 1)),
        named("attn.k_proj.weight", matrix(KV_HEADS * DIM, HIDDEN, 2)),
        named("attn.v_proj.weight", matrix(KV_HEADS * DIM, HIDDEN, 3)),
        named("attn.o_proj.weight", matrix(HIDDEN, HEADS * DIM, 4)),
        ("attn.q_norm.weight", vec![DIM], sequence(DIM, 5)),
        ("attn.k_norm.weight", vec![DIM], sequence(DIM, 6)),
    ]
}

/// The checkpoint at `directory` holding `weights` as `dtype`.
fn checkpoint(
    directory: &Path,
    dtype: DType,
    weights: &[(&'static str, Vec<usize>, Vec<f32>)],
) -> Checkpoint {
    let typed: Vec<(&str, DType, Vec<usize>, Vec<u8>)> = weights
        .iter()
        .map(|(name, shape, values)| (*name, dtype, shape.clone(), encode(dtype, values)))
        .collect();
    shard(directory, "model.safetensors", &typed);

    Checkpoint::open(directory).unwrap()
}

/// The weight `name` of `weights`, as an FP32 CPU tensor.
fn weight(weights: &[(&'static str, Vec<usize>, Vec<f32>)], name: &str) -> Tensor {
    let (_, shape, values) = weights.iter().find(|(stored, ..)| *stored == name).unwrap();
    tensor(values, shape.as_slice(), DType::F32)
}

/// The fused q/k normalization with the module's learned gains.
const FUSED: QkNorm = QkNorm::Fused {
    gains: Some(["q_norm", "k_norm"]),
    eps: EPS,
    centred: false,
};

/// The configuration of the module `attn` with `qk_norm`, under the causal mask or not.
fn attention_config(qk_norm: QkNorm, causal: bool, gated: bool) -> AttentionConfig {
    AttentionConfig {
        qkv: QkvLinearConfig {
            hidden: HIDDEN as u32,
            heads: HEADS as u32,
            kv_heads: KV_HEADS as u32,
            head_dim: DIM as u32,
            gated,
            names: QkvNames::Separate(["q_proj", "k_proj", "v_proj"]),
            bias: QkvBias::Absent,
            split: Split::WHOLE,
        },
        output: "o_proj",
        output_bias: false,
        qk_norm,
        scale: 0.125,
        causal,
        window: None,
        max_kv_len: 1024,
        sequence: SequenceSplit::WHOLE,
    }
}

/// FP32 factor tables of `tokens` rows for heads of [`DIM`] channels.
fn tables(tokens: usize) -> (Tensor, Tensor) {
    let table = |seed| {
        tensor(
            &sequence(tokens * DIM / 2, seed),
            (tokens, DIM / 2),
            DType::F32,
        )
    };
    (table(7), table(8))
}

/// `x · weightᵀ` for a checkpoint weight of `weights`.
fn project(x: &Tensor, weights: &[(&'static str, Vec<usize>, Vec<f32>)], name: &str) -> Tensor {
    functional::linear(x, &weight(weights, name), None, WeightOrder::NxK).unwrap()
}

/// The module is its functions in order: the three projections as heads, the fused
/// normalization and rotation, the attention, and the output projection.
#[test]
fn an_attention_projects_turns_attends_and_projects_out() {
    let directory = tempfile::tempdir().unwrap();
    let weights = attention_weights(false);
    let checkpoint = checkpoint(directory.path(), DType::F32, &weights);
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);

    let attention = Attention::load(
        &attention_config(FUSED, false, false),
        loader.root().pp("attn"),
    )
    .unwrap();
    assert_eq!(attention.query().n(), (HEADS * DIM) as u32);

    let tokens = 5;
    let hidden = tensor(&sequence(tokens * HIDDEN, 9), (tokens, HIDDEN), DType::F32);
    let (cos, sin) = tables(tokens);
    let out = attention.forward(&hidden, Some((&cos, &sin))).unwrap();
    assert_eq!(out.dims(), [tokens, HIDDEN]);

    let heads = |name, count| {
        project(&hidden, &weights, name)
            .reshape((tokens, count, DIM))
            .unwrap()
    };
    let (query, key) = functional::qk_norm_rope(
        &heads("attn.q_proj.weight", HEADS),
        &heads("attn.k_proj.weight", KV_HEADS),
        Some((
            &weight(&weights, "attn.q_norm.weight"),
            &weight(&weights, "attn.k_norm.weight"),
        )),
        &cos,
        &sin,
        EPS,
        false,
    )
    .unwrap();
    let attended = functional::dense_attention(
        &query,
        &key,
        &heads("attn.v_proj.weight", KV_HEADS),
        0.125,
        false,
        None,
    )
    .unwrap();
    let expected = project(
        &attended.reshape((tokens, HEADS * DIM)).unwrap(),
        &weights,
        "attn.o_proj.weight",
    );
    assert_eq!(values(&out), values(&expected));

    // A norm fused into the rotation needs the rotation, and one sequence stays under the key
    // bound the module was loaded for.
    let refused = attention.forward(&hidden, None).unwrap_err().to_string();
    assert!(
        refused.contains("needs the forward's rotation"),
        "{refused}"
    );

    let bounded = AttentionConfig {
        max_kv_len: 4,
        ..attention_config(FUSED, false, false)
    };
    let short = Attention::load(&bounded, loader.root().pp("attn")).unwrap();
    let refused = short
        .forward(&hidden, Some((&cos, &sin)))
        .unwrap_err()
        .to_string();
    assert!(refused.contains("5-key bound"), "{refused}");
}

/// A row attended over the cache is the row attended as one causal sequence: the paged forward
/// writes the turned keys and the values into the layer's pages and reads them back.
#[test]
fn a_paged_forward_attends_what_it_wrote_into_the_cache() {
    let directory = tempfile::tempdir().unwrap();
    let weights = attention_weights(false);
    let checkpoint = checkpoint(directory.path(), DType::F32, &weights);
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);

    // Six tokens of one row, in pages 1 then 0.
    let tokens = 6;
    let hidden = tensor(&sequence(tokens * HIDDEN, 10), (tokens, HIDDEN), DType::F32);
    let (cos, sin) = tables(tokens);
    let prefill = batch(PhaseKind::Prefill, &[4, 5, 6, 7, 0, 1], &[6], &[0, 6]);

    let separate = QkNorm::Separate {
        modules: ["q_norm", "k_norm"],
        eps: EPS,
    };
    for (qk_norm, rope) in [
        (FUSED, Some((&cos, &sin))),
        (separate, Some((&cos, &sin))),
        (separate, None),
        (QkNorm::None, Some((&cos, &sin))),
        (QkNorm::None, None),
    ] {
        let attention = Attention::load(
            &attention_config(qk_norm, true, false),
            loader.root().pp("attn"),
        )
        .unwrap();
        let cache = cache(DType::F32, &[1, 0], 1);

        let paged = attention
            .forward_paged(&hidden, rope, &prefill, &cache, 0)
            .unwrap();
        let dense = attention.forward(&hidden, rope).unwrap();
        assert_close(&values(&paged), &values(&dense), F32_TOLERANCE);

        // The values are stored as projected, token by token at their slots.
        let stored = values(&cache.layers[0].value);
        let projected = values(&project(&hidden, &weights, "attn.v_proj.weight"));
        let row = KV_HEADS * DIM;
        assert_eq!(stored[4 * row..8 * row], projected[..4 * row]);
        assert_eq!(stored[..2 * row], projected[4 * row..]);
    }

    // Attention over the cache is causal, and a page fits under the module's key bound.
    let cache = cache(DType::F32, &[1, 0], 1);
    let dense = Attention::load(
        &attention_config(FUSED, false, false),
        loader.root().pp("attn"),
    )
    .unwrap();
    let refused = dense.forward_paged(&hidden, Some((&cos, &sin)), &prefill, &cache, 0);
    assert!(refused.unwrap_err().to_string().contains("causal"));

    let bounded = AttentionConfig {
        max_kv_len: 2,
        ..attention_config(FUSED, true, false)
    };
    let short = Attention::load(&bounded, loader.root().pp("attn")).unwrap();
    let refused = short.forward_paged(&hidden, Some((&cos, &sin)), &prefill, &cache, 0);
    assert!(refused.unwrap_err().to_string().contains("shorter than"));
    assert!(
        values(&cache.layers[0].key_initialized)
            .iter()
            .all(|flag| *flag == 0.0)
    );
}

/// A gated attention projects each query head with a gate beside it, and scales the attended
/// head by the sigmoid of its gate.
#[test]
fn a_gated_attention_scales_each_head_by_the_sigmoid_of_its_gate() {
    let directory = tempfile::tempdir().unwrap();
    let weights = attention_weights(true);
    let checkpoint = checkpoint(directory.path(), DType::F32, &weights);
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);

    let attention = Attention::load(
        &attention_config(QkNorm::None, true, true),
        loader.root().pp("attn"),
    )
    .unwrap();
    let tokens = 3;
    let hidden = tensor(&sequence(tokens * HIDDEN, 11), (tokens, HIDDEN), DType::F32);
    let out = attention.forward(&hidden, None).unwrap();

    // Each head's row of the query projection is its query, then its gate.
    let projected = values(&project(&hidden, &weights, "attn.q_proj.weight"));
    let half = |offset: usize| -> Vec<f32> {
        projected
            .chunks(2 * DIM)
            .flat_map(|head| head[offset..offset + DIM].to_vec())
            .collect()
    };
    let (query, gate) = (half(0), half(DIM));
    let heads = |name| {
        project(&hidden, &weights, name)
            .reshape((tokens, KV_HEADS, DIM))
            .unwrap()
    };
    let attended = functional::dense_attention(
        &tensor(&query, (tokens, HEADS, DIM), DType::F32),
        &heads("attn.k_proj.weight"),
        &heads("attn.v_proj.weight"),
        0.125,
        true,
        None,
    )
    .unwrap();

    let gated: Vec<f32> = values(&attended)
        .iter()
        .zip(&gate)
        .map(|(attended, gate)| attended / (1.0 + (-gate).exp()))
        .collect();
    let expected = project(
        &tensor(&gated, (tokens, HEADS * DIM), DType::F32),
        &weights,
        "attn.o_proj.weight",
    );
    assert_close(&values(&out), &values(&expected), F32_TOLERANCE);
}

/// Each q/k-norm form launches its own kernels between the projections and the attention: the
/// gains fused into the rotation, a per-head RmsNorm of each projection, or the rotation alone.
#[test]
fn each_qk_norm_form_launches_its_own_kernels() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path(), DType::BF16, &attention_weights(false));
    let loader = WeightLoader::meta(&checkpoint, None, None);
    let hidden = meta((6, HIDDEN), DType::BF16);
    let (cos, sin) = (
        meta((6, DIM / 2), DType::F32),
        meta((6, DIM / 2), DType::F32),
    );
    let rope = Some((&cos, &sin));

    let separate = QkNorm::Separate {
        modules: ["q_norm", "k_norm"],
        eps: EPS,
    };
    for (qk_norm, rope, between) in [
        (FUSED, rope, &["qk_norm_rope"][..]),
        (separate, None, &["rms_norm", "rms_norm"][..]),
        (QkNorm::None, rope, &["rope_apply", "rope_apply"][..]),
    ] {
        let attention = Attention::load(
            &attention_config(qk_norm, false, false),
            loader.root().pp("attn"),
        )
        .unwrap();
        let (out, launches) = recorded(|| attention.forward(&hidden, rope));
        assert_eq!(out.unwrap().dims(), [6, HIDDEN]);

        let mut expected = vec!["gemm", "gemm", "gemm"];
        expected.extend(between);
        expected.extend(["dense_attention", "gemm"]);
        assert_eq!(operators(&launches), expected, "{qk_norm:?}");
    }

    // Packed sequences name their tables and their bounds, which stay under the module's.
    let attention = Attention::load(
        &attention_config(FUSED, false, false),
        loader.root().pp("attn"),
    )
    .unwrap();
    let offsets = meta(3, DType::U32);
    let varlen = |max_kv_len| {
        recorded(|| {
            attention.forward_varlen(&hidden, rope, &offsets, &offsets, Some(3), max_kv_len)
        })
    };
    let (packed, launches) = varlen(6);
    packed.unwrap();
    assert_eq!(
        launches[4].1,
        "dtype_bf16_kv_bf16_out_bf16_head_dim_8_q_heads_4_kv_heads_2_mask_dense_window_false"
    );
    assert!(varlen(2_048).0.is_err(), "a bound above the module's");
    assert!(varlen(4).0.is_err(), "a bound below the keys passed");

    // A gated module packs its queries out of the projection and gates the attended heads.
    let gated_checkpoint = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint_of_gated(gated_checkpoint.path());
    let loader = WeightLoader::meta(&checkpoint, None, None);
    let attention = Attention::load(
        &attention_config(FUSED, true, true),
        loader.root().pp("attn"),
    )
    .unwrap();
    let (out, launches) = recorded(|| attention.forward(&hidden, rope));
    assert_eq!(out.unwrap().dims(), [6, HIDDEN]);
    assert_eq!(
        operators(&launches),
        [
            "gemm",
            "gemm",
            "gemm",
            "copy_strided",
            "qk_norm_rope",
            "dense_attention",
            "act_and_mul",
            "gemm"
        ]
    );
}

/// A BF16 checkpoint of the gated module.
fn checkpoint_of_gated(directory: &Path) -> Checkpoint {
    checkpoint(directory, DType::BF16, &attention_weights(true))
}

/// A paged forward writes its keys and values into its layer's planes before it reads them,
/// and a rank of a tensor-parallel group attends its share of the heads and reduces in its
/// output projection.
#[test]
fn a_paged_forward_writes_its_layers_planes_then_reads_them() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path(), DType::BF16, &attention_weights(false));
    let loader = WeightLoader::meta(&checkpoint, None, None);

    let tokens = 4;
    let hidden = meta((tokens, HIDDEN), DType::BF16);
    let (cos, sin) = (
        meta((tokens, DIM / 2), DType::F32),
        meta((tokens, DIM / 2), DType::F32),
    );
    let batch = meta_batch(PhaseKind::Prefill, tokens);

    let attention = Attention::load(
        &attention_config(FUSED, true, false),
        loader.root().pp("attn"),
    )
    .unwrap();
    let cache = meta_cache(DType::BF16, KV_HEADS, DIM);
    let (out, launches) =
        recorded(|| attention.forward_paged(&hidden, Some((&cos, &sin)), &batch, &cache, 0));
    assert_eq!(out.unwrap().dims(), [tokens, HIDDEN]);
    assert_eq!(
        operators(&launches),
        [
            "gemm",
            "gemm",
            "gemm",
            "qk_norm_rope",
            "kv_write",
            "paged_attention",
            "gemm"
        ]
    );
    assert_eq!(
        launches[5].1,
        "phase_prefill_dtype_bf16_kv_bf16_out_bf16_head_dim_8_page_64_q_heads_4_kv_heads_2\
         _mask_causal_window_false_len_1024"
    );

    // Rank 1 of 2 holds two query heads over one key head, and its output projection is a
    // partial sum the group reduces.
    let mut split = attention_config(FUSED, true, false);
    split.qkv.split = Split { size: 2, rank: 1 };
    let attention = Attention::load(&split, loader.root().pp("attn")).unwrap();
    let cache = meta_cache(DType::BF16, 1, DIM);

    let forward = || attention.forward_paged(&hidden, Some((&cos, &sin)), &batch, &cache, 0);
    let (out, launches) = recorded_split(Some((1, 2)), forward);
    assert_eq!(out.unwrap().dims(), [tokens, HIDDEN]);
    assert_eq!(
        launches[5].1,
        "phase_prefill_dtype_bf16_kv_bf16_out_bf16_head_dim_8_page_64_q_heads_2_kv_heads_1\
         _mask_causal_window_false_len_1024"
    );

    let (alone, _) = recorded(forward);
    assert!(alone.unwrap_err().to_string().contains("no group"));
}

/// A member of a two-member sequence split, under either projection form, attends its share of
/// the heads over every row of the batch: given its share of a five-token batch, three rows
/// whose last pads the sequence, it writes and attends its two query heads over one key head of
/// all five rows and returns its three rows of every head projected whole. Under the gather form
/// it holds its heads' rows of the input projections; under the exchange form it holds them
/// whole. A split attention runs in a forward whose group is its split, and over the paged
/// forward alone; it is refused beside a tensor split.
#[test]
fn a_sequence_split_member_attends_its_heads_of_every_row_and_returns_its_own_rows() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path(), DType::BF16, &attention_weights(false));
    let loader = WeightLoader::meta(&checkpoint, None, None);

    let (tokens, share) = (5, 3);
    let hidden = meta((share, HIDDEN), DType::BF16);
    let (cos, sin) = (
        meta((tokens, DIM / 2), DType::F32),
        meta((tokens, DIM / 2), DType::F32),
    );
    let batch = meta_batch(PhaseKind::Prefill, tokens);
    let cache = meta_cache(DType::BF16, KV_HEADS / 2, DIM);

    for (projection, held) in [
        (nn::family::Projection::Gather, HEADS / 2),
        (nn::family::Projection::Exchange, HEADS),
    ] {
        let config = AttentionConfig {
            sequence: SequenceSplit {
                split: Split { size: 2, rank: 1 },
                projection,
            },
            ..attention_config(FUSED, true, false)
        };
        let attention = Attention::load(&config, loader.root().pp("attn")).unwrap();
        assert_eq!(attention.query().n(), (held * DIM) as u32, "{projection:?}");

        let forward = || attention.forward_paged(&hidden, Some((&cos, &sin)), &batch, &cache, 0);
        let (out, launches) = recorded_split(Some((1, 2)), forward);
        assert_eq!(out.unwrap().dims(), [share, HIDDEN], "{projection:?}");

        let written = launches
            .iter()
            .find(|(operator, _)| *operator == "kv_write")
            .unwrap();
        assert!(written.1.contains("_heads_1_"), "{written:?}");
        let attended = launches
            .iter()
            .find(|(operator, _)| *operator == "paged_attention")
            .unwrap();
        assert!(
            attended.1.contains("_q_heads_2_kv_heads_1_"),
            "{attended:?}"
        );

        // The member's rows are its share of the batch's, and its group is its split.
        let (alone, _) = recorded(forward);
        assert!(alone.is_err(), "{projection:?}");
        let whole = meta((tokens, HIDDEN), DType::BF16);
        let (unshared, _) = recorded_split(Some((1, 2)), || {
            attention.forward_paged(&whole, Some((&cos, &sin)), &batch, &cache, 0)
        });
        assert!(unshared.is_err(), "{projection:?}");
        let (dense, _) = recorded_split(Some((1, 2)), || {
            attention.forward(&hidden, Some((&cos, &sin)))
        });
        assert!(dense.is_err(), "{projection:?}");
    }

    let mut both = attention_config(FUSED, true, false);
    both.qkv.split = Split { size: 2, rank: 0 };
    both.sequence.split = Split { size: 2, rank: 0 };
    assert!(Attention::load(&both, loader.root().pp("attn")).is_err());
}
