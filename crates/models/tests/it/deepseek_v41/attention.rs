//! The sparse latent attention modules on the CPU device, loaded with the real parameters of
//! DeepSeek-V4.1-Flash's layers 2 (ratio 2) and 20 (ratio 1, the candidate source), against a
//! direct port of the checkpoint's reference formulas (`generate_reference.py`) over one fixed
//! prefill of a few thousand tokens.
//!
//! The modules read the inputs the model's projections compute (the query latent `qr`, the
//! unrotated queries and the latent projection), which the reference vectors hold, and write
//! the cache fields in their production records: the window as MXFP8, the compressed latents as
//! NVFP4, the index keys as MXFP4 pages. Decoding those records is the reference's fake
//! quantization exactly, so what differs is the order of FP32 reductions and the BF16 roundings
//! between them.
//!
//! | Compared | Bound |
//! | --- | --- |
//! | compressed latents | `4u` of the tensor's largest magnitude, `u = 2⁻⁸` BF16's unit roundoff: each latent is rounded into BF16 after its pooling and after its normalization, and each rounding may land one BF16 spacing (`2u` of the value) from the reference's |
//! | selection | every kept entry is among the reference's best by its own scores, within the BF16 rounding of a sum of the index heads' products (`heads · u` of the row's largest score) |
//! | attention, with the reference's selection | the output's relative L2 within `4u`, no row beyond five times it |
//!
//! The selection is not compared entry by entry: the reference ranks BF16 scores, so near ties
//! decide its choice, and a selection is defined only up to them. The attention over the
//! modules' own selection is reported beside the bounded one.
//!
//! The vectors are generated from the checkpoint (they hold its parameters) into
//! `artifacts/deepseek_v41/attention-reference`, and the tests are ignored without them:
//!
//! ```text
//! python crates/models/tests/it/deepseek_v41/generate_reference.py \
//!     --checkpoint <DeepSeek-V4.1-Flash-1017> --out artifacts/deepseek_v41/attention-reference
//! cargo test -p models --test it deepseek_v41 -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use nn::collective::Split;
use nn::functional::{self, Format, Layout, Plane, Ring, Selected, WeightOrder};
use nn::modules::{
    Candidates, Compressor, CompressorConfig, Indexer, IndexerConfig, LatentAttention,
    LatentAttentionConfig, LatentBatch, Linear, Precision, RotaryConfig, Spectrum, Yarn,
};
use nn::{Checkpoint, CpuWeights, DType, Device, Tensor, WeightLoader};

/// The checkpoint's attention dimensions (`config.json`, `text_config`).
const HIDDEN: u32 = 5120;
const HEADS: u32 = 64;
const HEAD_DIM: u32 = 512;
const ROTATED: u32 = 64;
const WINDOW: u32 = 128;
const EPS: f32 = 1e-20;
const INDEX_HEADS: u32 = 32;
const INDEX_DIM: u32 = 128;
const INDEX_TOP: u32 = 512;
const CANDIDATE_BLOCKS: u32 = 2048;
const CANDIDATE_BLOCK: u32 = 8;

/// Tokens of a cache page.
const PAGE: usize = 64;

/// Rows of a request's compressor ring.
const RING: usize = 8;

/// BF16's unit roundoff `u`: half the spacing of BF16 values relative to their magnitude.
const BF16_UNIT: f32 = 1.0 / 256.0;

/// The directory the reference vectors are generated into.
fn reference() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/deepseek_v41/attention-reference")
}

/// Every tensor of the reference file `name`, by name.
fn tensors(name: &str) -> HashMap<String, Tensor> {
    let path = reference().join(name);
    let file = Checkpoint::open_file(&path).unwrap_or_else(|error| {
        panic!(
            "{}: {error}; generate it with generate_reference.py",
            path.display()
        )
    });
    file.names()
        .map(|tensor| {
            let location = file.locate(tensor).unwrap();
            let bytes = file.read(tensor).unwrap();
            let read = Tensor::from_raw_buffer(
                &bytes,
                location.dtype,
                location.shape.dims(),
                &Device::Cpu,
            )
            .unwrap();
            (tensor.to_owned(), read)
        })
        .collect()
}

/// A CPU tensor of `values`.
fn column<T: tensor::WithDType>(values: Vec<T>, shape: impl Into<nn::Shape>) -> Tensor {
    Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
}

/// The values of `tensor` as FP32, row-major.
fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// A zeroed plane of `pages` pages of `entries` records of `width` values in `format`, and its
/// table of one row naming every page in order.
fn plane(format: Format, pages: usize, entries: usize, width: u64) -> (Tensor, Tensor) {
    let (dtype, last) = format.plane(width);
    (
        Tensor::zeros((pages, entries, last as usize), dtype, &Device::Cpu).unwrap(),
        column((0..pages as u32).collect(), (1, pages)),
    )
}

/// The relative L2 difference of `got` from `expected` over rows of `width`, and the largest
/// relative difference of one row, a row's norm floored at a tenth of the mean row norm.
fn relative_l2(got: &[f32], expected: &[f32], width: usize) -> (f32, f32) {
    let squared = |values: &[f32]| values.iter().map(|value| value * value).sum::<f32>();
    let difference: Vec<f32> = got.iter().zip(expected).map(|(a, b)| a - b).collect();
    let whole = (squared(&difference) / squared(expected)).sqrt();

    let rows = expected.len() / width;
    let norms: Vec<f32> = expected
        .chunks(width)
        .map(|row| squared(row).sqrt())
        .collect();
    let floor = 0.1 * norms.iter().sum::<f32>() / rows as f32;
    let worst = difference
        .chunks(width)
        .zip(&norms)
        .map(|(row, norm)| squared(row).sqrt() / norm.max(floor))
        .fold(0.0, f32::max);
    (whole, worst)
}

/// The reference's own state of one layer: its vectors under `layers.<layer>.attn.`.
struct Vectors {
    all: HashMap<String, Tensor>,
    prefix: String,
}

impl Vectors {
    fn of(layer: usize) -> Self {
        Self {
            all: tensors("vectors.safetensors"),
            prefix: format!("layers.{layer}.attn"),
        }
    }

    fn get(&self, name: &str) -> Tensor {
        let key = format!("{}.{name}", self.prefix);
        self.all
            .get(&key)
            .or_else(|| self.all.get(name))
            .unwrap_or_else(|| panic!("the reference vectors hold no `{key}`"))
            .clone()
    }
}

/// Runs layer `layer`'s compressor, indexer and attention over the reference's prefill and
/// holds each result to the reference's (module documentation).
fn check_layer(layer: usize, ratio: u32, candidates: Candidates) {
    let parameters = Checkpoint::open_file(&reference().join("parameters.safetensors"))
        .expect("the reference parameters; generate them with generate_reference.py");
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&parameters, &store, None);
    let vb = loader.root().pp("layers").pp(layer).pp("attn");
    let vectors = Vectors::of(layer);

    let x = vectors.get("x");
    let tokens = x.dims()[0];
    let groups = tokens / ratio as usize;

    // The modules over the layer's real parameters; the index query projection is the
    // dequantized block-scaled weight, exactly representable in BF16.
    let compressor = Compressor::load(
        &CompressorConfig {
            hidden: HIDDEN,
            width: HEAD_DIM,
            ratio,
            rotated: ROTATED,
            eps: EPS,
        },
        vb.pp("compressor"),
    )
    .unwrap();
    let wq_b = parameters
        .read(&format!("layers.{layer}.attn.indexer.wq_b.weight"))
        .unwrap();
    let wq_b = Tensor::from_raw_buffer(
        &wq_b,
        DType::BF16,
        &[(INDEX_HEADS * INDEX_DIM) as usize, 1280],
        &Device::Cpu,
    )
    .unwrap();
    let indexer = Indexer::load(
        &IndexerConfig {
            hidden: HIDDEN,
            heads: INDEX_HEADS,
            width: INDEX_DIM,
            latent: HEAD_DIM,
            rotated: ROTATED,
            top: INDEX_TOP,
            ratio,
            eps: EPS,
            owns_keys: true,
            candidates,
            split: Split::WHOLE,
        },
        vb.pp("indexer"),
        Linear::new(wq_b, None, WeightOrder::NxK).unwrap(),
    )
    .unwrap();
    let attention = LatentAttention::load(
        &LatentAttentionConfig {
            heads: HEADS,
            head_dim: HEAD_DIM,
            rotated: ROTATED,
            window: WINDOW,
            eps: EPS,
            scale: (HEAD_DIM as f32).powf(-0.5),
            split: Split::WHOLE,
        },
        vb.clone(),
    )
    .unwrap();

    // The layer's rotary embedding: YaRN over 160000, in the reference's f32 rule.
    let inv_freq = RotaryConfig {
        theta: 160_000.0,
        dim: ROTATED,
        spectrum: Spectrum::Yarn(Yarn {
            factor: 16.0,
            original_max_position_embeddings: 65_536,
            attention_factor: None,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: None,
            mscale_all_dim: None,
        }),
        precision: Precision::F32,
    }
    .inverse_frequencies()
    .unwrap();
    let inv_freq = column(inv_freq, ROTATED as usize / 2);
    let positions = column((0..tokens as i32).collect(), tokens);
    let (cos, sin) = functional::rotary_factors(&positions, &inv_freq, &[], false).unwrap();
    let rows = column(vec![0_u32; tokens], tokens);
    let starts = column(vec![0_u32], 1);
    let batch = LatentBatch {
        positions: &positions,
        written: &positions,
        rows: &rows,
        window_starts: &starts,
        cos: &cos,
        sin: &sin,
        inv_freq: &inv_freq,
    };

    // One request's cache fields in their production records, every page on its row.
    let entries = PAGE / ratio as usize;
    let (window, window_table) = plane(Format::Mxfp8, tokens.div_ceil(PAGE), PAGE, HEAD_DIM.into());
    let compressed_pages = groups.div_ceil(entries);
    let (compressed, compressed_table) =
        plane(Format::Nvfp4, compressed_pages, entries, HEAD_DIM.into());
    let (keys, keys_table) = plane(Format::Mxfp4, compressed_pages, entries, INDEX_DIM.into());
    let window = Plane {
        records: &window,
        table: &window_table,
        format: Format::Mxfp8,
        layout: Layout::Entry,
    };
    let compressed = Plane {
        records: &compressed,
        table: &compressed_table,
        format: Format::Nvfp4,
        layout: Layout::Entry,
    };
    let keys = Plane {
        records: &keys,
        table: &keys_table,
        format: Format::Mxfp4,
        layout: Layout::Page,
    };
    let ring = Tensor::zeros((1, RING, 2 * HEAD_DIM as usize), DType::F32, &Device::Cpu).unwrap();
    let (slots, offsets, seen) = (
        column(vec![0_u32], 1),
        column(vec![0_u32, tokens as u32], 2),
        column(vec![tokens as u32], 1),
    );
    let ring = (ratio > 1).then_some(Ring {
        rows: &ring,
        slots: &slots,
        query_offsets: &offsets,
        seq_lens: &seen,
    });

    // The layer's forward: compress, index, store, attend.
    let completed = compressor.forward(&x, &batch, ring).unwrap();
    let selection = indexer
        .forward(
            (&x, &vectors.get("qr")),
            Some(&completed),
            keys,
            &batch,
            None,
        )
        .unwrap();
    compressor.store_(&completed, compressed, &batch).unwrap();
    let attend = |chosen: &Tensor| {
        let query = vectors.get("q").copy().unwrap();
        attention
            .forward(
                (&query, &vectors.get("kv")),
                window,
                Some(Selected {
                    plane: compressed,
                    selection: chosen,
                }),
                &batch,
            )
            .unwrap()
    };

    // The compressed latents, at the tokens that complete their groups.
    let completing: Vec<i32> = completed.entries.to_vec1().unwrap();
    let latent = values(&completed.latent);
    let width = HEAD_DIM as usize;
    let mut got = Vec::with_capacity(groups * width);
    for (token, entry) in completing.iter().enumerate() {
        if *entry >= 0 {
            got.extend_from_slice(&latent[token * width..(token + 1) * width]);
        }
    }
    let expected = values(&vectors.get("latent"));
    assert_eq!(
        got.len(),
        expected.len(),
        "layer {layer}: the completed groups"
    );
    let peak = expected
        .iter()
        .fold(0.0_f32, |peak, value| peak.max(value.abs()));
    let latent_error = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("layer {layer}: compressed latents differ by {latent_error} of a peak {peak}");
    assert!(
        latent_error <= 4.0 * BF16_UNIT * peak,
        "layer {layer}: compressed latents differ by {latent_error}, bound {}",
        4.0 * BF16_UNIT * peak
    );

    // The selection: each kept entry among the reference's best by its own scores.
    let scores = vectors.get("index_scores").to_vec2::<f32>().unwrap();
    let kept = selection.entries.to_vec2::<i32>().unwrap();
    let ours_reference = vectors.get("selection").to_vec2::<i32>().unwrap();
    let (mut shared, mut named, mut worst) = (0_usize, 0_usize, 0.0_f32);
    for token in 0..tokens {
        let row = &scores[token];
        let mut finite: Vec<f32> = row
            .iter()
            .copied()
            .filter(|score| score.is_finite())
            .collect();
        finite.sort_by(|a, b| b.total_cmp(a));
        let reference: Vec<i32> = ours_reference[token]
            .iter()
            .copied()
            .filter(|e| *e >= 0)
            .collect();
        let mine: Vec<i32> = kept[token].iter().copied().filter(|e| *e >= 0).collect();
        assert_eq!(
            mine.len(),
            reference.len(),
            "layer {layer}, token {token}: kept entries"
        );
        if reference.is_empty() {
            continue;
        }
        let threshold = finite[reference.len() - 1];
        let scale = finite
            .iter()
            .fold(0.0_f32, |peak, score| peak.max(score.abs()));
        for entry in &mine {
            worst = worst.max((threshold - row[*entry as usize]) / scale);
        }
        shared += mine
            .iter()
            .filter(|entry| reference.contains(entry))
            .count();
        named += reference.len();
    }
    let slack = INDEX_HEADS as f32 * BF16_UNIT;
    println!(
        "layer {layer}: {:.2}% of the reference's selected entries kept; the worst kept entry \
         is {worst:.3e} of a row's peak below the reference's threshold (bound {slack:.3e})",
        100.0 * shared as f32 / named as f32
    );
    assert!(
        worst <= slack,
        "layer {layer}: a kept entry lies {worst} below the reference's threshold"
    );

    // The candidate blocks of the source: every block of a row this short is a candidate.
    if let Some(candidates) = &selection.candidates {
        let blocks = candidates.to_vec2::<i32>().unwrap();
        let mask = vectors.get("candidates").to_vec2::<u8>().unwrap();
        for token in 0..tokens {
            let mut ours = vec![0_u8; mask[token].len()];
            for block in blocks[token].iter().filter(|block| **block >= 0) {
                let start = *block as usize * CANDIDATE_BLOCK as usize;
                for flag in ours.iter_mut().skip(start).take(CANDIDATE_BLOCK as usize) {
                    *flag = 1;
                }
            }
            assert_eq!(
                ours, mask[token],
                "layer {layer}, token {token}: candidate blocks"
            );
        }
        println!("layer {layer}: candidate blocks equal the reference's");
    }

    // The attention over the reference's selection, bounded; over the modules' own, reported.
    let expected = values(&vectors.get("attention"));
    let teacher = values(&attend(&vectors.get("selection")));
    let (whole, row) = relative_l2(&teacher, &expected, width);
    let own = values(&attend(&selection.entries));
    let (own_whole, own_row) = relative_l2(&own, &expected, width);
    println!(
        "layer {layer}: attention over the reference's selection: relative L2 {whole:.3e}, \
         worst row {row:.3e}; over the modules' own selection: {own_whole:.3e}, worst row \
         {own_row:.3e}"
    );
    let bound = 4.0 * BF16_UNIT;
    assert!(
        whole <= bound && row <= 5.0 * bound,
        "layer {layer}: attention differs by {whole} (worst row {row}), bound {bound}"
    );
}

#[test]
#[ignore = "reads the reference vectors generate_reference.py writes from the checkpoint"]
fn layer_2_compresses_indexes_and_attends_as_the_reference() {
    check_layer(2, 2, Candidates::None);
}

#[test]
#[ignore = "reads the reference vectors generate_reference.py writes from the checkpoint"]
fn layer_20_compresses_indexes_and_attends_as_the_reference() {
    check_layer(
        20,
        1,
        Candidates::Source {
            blocks: CANDIDATE_BLOCKS,
            block: CANDIDATE_BLOCK,
        },
    );
}
