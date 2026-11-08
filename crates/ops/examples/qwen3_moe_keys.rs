//! The descriptor keys the Qwen3-30B-A3B BF16 vertical slice compiles.
//!
//! One line per key, `operator<TAB>key`, so a kernel program can enumerate exactly the variants it
//! must package for that slice. Run it with
//!
//! ```text
//! cargo run -p ops --example qwen3_moe_keys
//! ```
//!
//! The geometry is the checkpoint's (128 experts of top-8, hidden 2048, expert intermediate 768
//! with the fused FC1, 32 query and 4 KV heads of 128 features, a 151936-token vocabulary, 64-token
//! pages, BF16 weights and cache) at one tensor-parallel and one expert-parallel rank.
//!
//! The constants a layer's descriptor carries belong to the program package; the layer crate
//! declares them once (`crates/nn/src/layers/mod.rs`, module `launch`, and the MMA tile in
//! `layers/moe.rs`) and this generator mirrors those values, because `ops` sits below `nn` and
//! cannot call it. If a value here and the layer crate's disagree, the layer crate is right and
//! this file follows it. Candidate widths and the sampler's extras are the slice's own request
//! state (`crates/nn/src/layers/sampler.rs`, `Sampler::new`).
//!
//! **Which side launches a row.** A row exists because a call site launches that key, never because
//! a kernel exists. Two families make up this file:
//!
//! - **the model's**, which `Qwen3Moe::keys()` reproduces from the layers' own descriptor
//!   accessors (`crates/models/qwen3-moe/src/keys.rs`); if this generator and that walk disagree,
//!   the walk is right;
//! - **the executor's**, whose geometry is request and scheduling state the model does not hold:
//!   the sampler's chain and the logits gathers. Those are keyed where the executor compiles its
//!   plans, and the rows here are its list, not this file's invention.
//!
//! Gaps this file does not fill, named rather than invented, so that the 85 declared operators are
//! accounted for between the two fixtures and these lines:
//!
//! - **The quantized export's family**: `nvfp4_gemm`, `nvfp4_quantize`, `gather_gemm_nvfp4`,
//!   `route_gemm_nvfp4`, `act_and_mul_fp8`, `value_first_swiglu_fp8` and `expert_exchange`. Their descriptors carry the export's own choices — the block-scale dtype,
//!   the expert packing, the exchange's ranks — and no NVFP4 shard exists in this workspace, so a
//!   key here would be a guess about an artifact nobody has opened.
//! - **The executor's path, not the model's**: `slot_commit` (its views are the sampler's state),
//!   `attention_gather_rows`, `attention_gather_decode_inputs`, `page_conversion`,
//!   `cache_block_snapshot`, `cache_block_scatter` and `cache_copy_region` (the page geometry and
//!   the block shapes are the executor's scheduling), and `paged_attention`, whose decode attention
//!   is trtllm-gen. These are keyed where the executor compiles its own plans.
//! - `logprob_gather`, `logits_gather`, `vocab_unpad` and the sampler's chain are above: they are the
//!   slice's, and the widths are its request state.

use ops::attention::{Descriptor as AttentionDescriptor, Mask, Phase};
use ops::moe;
use ops::rope::{FactorsDescriptor, MAX_AXES, QkNormRopeDescriptor, Rotation};
use ops::{DType, Float};

/// Whether the layers' descriptors are the programmatic-dependent-launch variants.
const PDL: bool = false;

/// The lanes one `top_k` ordering pass reads, matching `nn::layers::sampler`'s own chunk.
const TOP_TILE: u32 = 2048;

/// Tokens one `embedding` program handles (`launch::EMBEDDING_TOKENS`).
const EMBEDDING_TOKENS: u32 = 8;

/// Elements one `rotary_factors` program handles (`launch::ROTARY_BLOCK`).
const ROTARY_BLOCK: u32 = 256;

/// Head rows one `qk_norm_rope` program handles (`launch::QK_NORM_ROWS`).
const QK_NORM_ROWS: u32 = 16;

/// The row width a row-wise kernel loads at a time (`launch::row_tile`).
fn row_tile(columns: u32) -> u32 {
    columns.max(1).next_power_of_two()
}

/// The element count of one cache row a `kv_write` program covers (`launch::kv_tile`).
fn kv_tile(head_dim: u32) -> u32 {
    head_dim.clamp(1, 128).next_power_of_two().min(128)
}

/// The MMA tile of the grouped GEMMs (`layers/moe.rs`, `MMA_TILE`).
const MMA_TILE: (u32, u32) = (128, 128);

/// The cluster shape of the grouped GEMMs (`layers/moe.rs`, `MMA_CLUSTER`).
const MMA_CLUSTER: (u32, u32) = (1, 1);

/// Permuted rows per tile of the expert sort (`layers/moe.rs`, `Moe::new`'s tile).
const SORT_TILE: u32 = 128;

/// Lanes one candidate program walks at a time: a chunk of the row, not the row.
const CANDIDATE_TILE: u32 = 8192;

/// The candidate width the slice's sampler keeps per row (`Sampler::new`'s `width`).
///
/// It is the **vocabulary**, not a top-k bound: a request may ask for any top-k it likes, so the
/// runner builds the sampler with a candidate buffer as wide as the vocabulary — a narrower one
/// would silently sample a different set (`src/model_runner.rs`, the one place a
/// runner's geometry is stated). A deployment that narrowed it would change these keys, which is
/// why the width is a descriptor field at all.
const CANDIDATE_WIDTH: u32 = VOCABULARY;

/// Extra counted tokens a sampling row carries in the slice's request state.
const EXTRAS: u32 = 0;

// The checkpoint's geometry.
const HIDDEN: u32 = 2048;
const HEADS: u32 = 32;
const KV_HEADS: u32 = 4;
const HEAD_DIM: u32 = 128;
const EXPERTS: u32 = 128;
const TOP: u32 = 8;
/// One expert's intermediate width, as the checkpoint's `moe_intermediate_size` states it: the
/// descriptor field of both expert products means this number, not the gated product's width. The
/// first product contracts `hidden` into `2 · intermediate` and fuses the gate, so its *output* is
/// `intermediate` wide; the launch multiplies by two internally and the key never names the
/// doubled width. `a7b390915` corrected the fixture row that had it wrong; this is the generator
/// that would have put it back.
const INTERMEDIATE: u32 = 768;
/// FC1's fused output width: the gate and up projections side by side.
const VOCABULARY: u32 = 151936;
const PAGE: u32 = 64;
const MAX_KV_LEN: u32 = 40960;

/// The row width of a projection whose contraction is `k`; the keys only carry the contraction.
const PROJECTION_K: [u32; 3] = [HIDDEN, HEADS * HEAD_DIM, INTERMEDIATE];

fn main() {
    let dtype = DType::BF16;
    let mut lines: Vec<(String, String)> = Vec::new();

    // A model walk's print, when one is given, is merged in rather than restated: the rows a layer
    // launches are the walk's statement (`Qwen3Moe::keys`), and this file may only add the executor's
    // own — so a merged print is the file, and a difference between the two is visible as a row.
    let mut arguments = std::env::args().skip(1);
    let mut model_keys: Option<std::path::PathBuf> = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--model-keys" => match arguments.next() {
                Some(path) => model_keys = Some(std::path::PathBuf::from(path)),
                None => {
                    eprintln!("--model-keys takes the path of a `qwen3-moe --example keys` print");
                    return;
                }
            },
            other => {
                eprintln!("unknown argument {other}");
                eprintln!("usage: qwen3_moe_keys [--model-keys PATH]");
                return;
            }
        }
    }

    // Attention, both phases of the slice.
    for phase in [Phase::Prefill, Phase::Decode] {
        let descriptor = AttentionDescriptor {
            phase,
            query: dtype,
            key_value: dtype,
            output: dtype,
            head_dim: HEAD_DIM,
            page_size: PAGE,
            q_heads: HEADS,
            kv_heads: KV_HEADS,
            mask: Mask::Causal,
            windowed: false,
            max_kv_len: MAX_KV_LEN,
        };
        lines.push(("paged_attention".to_owned(), key(&descriptor)));
    }

    // Routing and the mixture of experts.
    lines.push((
        "topk_softmax".to_owned(),
        key(&ops::moe::routing::TopkDescriptor {
            scores: dtype,
            experts: EXPERTS,
            top: TOP,
            renormalize: true,
            scale: None,
        }),
    ));
    lines.push((
        "moe_sort".to_owned(),
        key(&moe::sort::Descriptor {
            experts: EXPERTS,
            top: TOP,
            local_experts: EXPERTS,
            tile_tokens: SORT_TILE,
            scales: DType::F32,
            counted: true,
        }),
    ));
    lines.push((
        "gather_gemm".to_owned(),
        key(&moe::grouped::GatherGemmDescriptor {
            hidden: HIDDEN,
            intermediate: INTERMEDIATE,
            experts: EXPERTS,
            top: TOP,
            activation: moe::grouped::Activation::Silu,
            dtype,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        }),
    ));
    lines.push((
        "route_gemm".to_owned(),
        key(&moe::grouped::RouteGemmDescriptor {
            hidden: HIDDEN,
            intermediate: INTERMEDIATE,
            experts: EXPERTS,
            top: TOP,
            dtype,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        }),
    ));
    lines.push((
        "moe_unpermute".to_owned(),
        key(&moe::combine::Descriptor {
            layout: moe::combine::Layout::Permuted,
            hidden: HIDDEN,
            top: TOP,
            experts: EXPERTS,
            input: dtype,
            scales: DType::F32,
            output: dtype,
        }),
    ));

    // The KV cache.
    lines.push((
        "kv_write".to_owned(),
        key(&ops::kv::Descriptor {
            dtype,
            kv: dtype,
            page: PAGE,
            heads: KV_HEADS,
            head_dim: HEAD_DIM,
            tile: kv_tile(HEAD_DIM),
            pdl: PDL,
        }),
    ));

    // A deployment that stores its pages as E4M3 reads them through the same attention program with
    // one scale per page, and fills them with the write that derives that scale.
    for phase in [
        ops::attention::Phase::Prefill,
        ops::attention::Phase::Decode,
    ] {
        lines.push((
            "paged_attention".to_owned(),
            key(&ops::attention::Descriptor {
                phase,
                query: dtype,
                key_value: DType::F8E4M3,
                output: dtype,
                head_dim: HEAD_DIM,
                page_size: PAGE,
                q_heads: HEADS,
                kv_heads: KV_HEADS,
                mask: ops::attention::Mask::Causal,
                windowed: false,
                max_kv_len: MAX_KV_LEN,
            }),
        ));
    }
    lines.push((
        "kv_write_scaled".to_owned(),
        key(&ops::kv::ScaledWriteDescriptor {
            dtype,
            scale: DType::F32,
            page: PAGE,
            heads: KV_HEADS,
            head_dim: HEAD_DIM,
            block: kv_tile(HEAD_DIM),
            pdl: PDL,
        }),
    ));

    // Normalization, rotation and the embedding. One `rms_norm` width: the model's three
    // normalizations are all `[hidden]` — the output norm and every layer's two pre-norms. A head's
    // 128 features are normalized inside `qk_norm_rope`, which is a program of its own, so a
    // 128-wide `rms_norm` is a module nothing launches.
    lines.push((
        "rms_norm".to_owned(),
        key(&ops::norm::RmsNormDescriptor {
            dtype,
            columns: HIDDEN,
            tile: row_tile(HIDDEN),
            pdl: PDL,
        }),
    ));
    lines.push((
        "add_rms_norm".to_owned(),
        key(&ops::norm::AddRmsNormDescriptor {
            dtype,
            columns: HIDDEN,
            tile: row_tile(HIDDEN),
            pdl: PDL,
        }),
    ));
    lines.push((
        "rotary_factors".to_owned(),
        key(&FactorsDescriptor {
            dtype: DType::F32,
            positions: DType::I32,
            columns: HEAD_DIM / 2,
            block: ROTARY_BLOCK,
            // The layer's scale, not the rope base: `Rotary` multiplies every angle by 1.0 and the
            // frequencies table carries Qwen3's 1e6 base, so a key that named the base would compile
            // the same program under a name no layer asks for.
            scale: Float::from(1.0_f32),
            pdl: PDL,
            // One-dimensional positions: Qwen3 is text only, so no section is cut.
            axes: 1,
            sections: [0; MAX_AXES],
            interleaved: false,
        }),
    ));
    lines.push((
        "qk_norm_rope".to_owned(),
        key(&QkNormRopeDescriptor {
            factors: ops::tensor::DType::F32,
            strided: false,
            dtype,
            output: dtype,
            head_dim: HEAD_DIM,
            q_heads: HEADS,
            k_heads: KV_HEADS,
            // Qwen3 rotates a feature against the opposite half of its head (`rotate_half`).
            rotation: Rotation::Half,
            // The checkpoint's rms_norm_eps, by bit pattern.
            epsilon: Float::from(1e-6_f32),
            rotated: HEAD_DIM,
            rows: QK_NORM_ROWS,
            block: row_tile(HEAD_DIM).min(1024),
            affine: true,
            pdl: PDL,
        }),
    ));
    lines.push((
        "embedding".to_owned(),
        key(&ops::embedding::Descriptor {
            dtype,
            ids: DType::I32,
            hidden: HIDDEN,
            first: 0,
            rows: VOCABULARY,
            block: EMBEDDING_TOKENS,
            pdl: PDL,
        }),
    ));

    // Every contraction the slice launches: q, k, v, o, the router, the head and both expert
    // projections. The descriptor carries the contraction length; the output width is the launch's.
    for k in PROJECTION_K {
        lines.push((
            "gemm".to_owned(),
            key(&ops::gemm::Descriptor {
                input: dtype,
                weight: dtype,
                accumulate: DType::F32,
                output: dtype,
                // The checkpoint stores a projection as [out, contraction].
                order: ops::gemm::WeightOrder::NxK,
                epilogue: ops::gemm::Epilogue::None,
                k,
            }),
        ));
    }

    // The head and the sampler.
    lines.push((
        "logits_gather".to_owned(),
        key(&ops::logits::GatherDescriptor {
            dtype,
            columns: HIDDEN,
            block: 1024,
            pdl: PDL,
        }),
    ));
    lines.push((
        "vocab_unpad".to_owned(),
        key(&ops::logits::UnpadDescriptor {
            dtype,
            // The sampler's candidate buffer is FP32 (the row below says the same for
            // `logits_shape`), so the head's product is widened here.
            output: DType::F32,
            vocabulary: VOCABULARY,
            padded: VOCABULARY,
            block: 1024,
            pdl: PDL,
        }),
    ));
    lines.push((
        "logits_shape".to_owned(),
        // The logits are the runner's buffer, and it allocates them in FP32 whatever the model's
        // activations are (`causal_lm.rs`: the plan's `logits_dtype`), so this row is the one place
        // the sampler chain's element type is not the model's.
        key(&ops::sampler::ShapeDescriptor {
            dtype: DType::F32,
            vocabulary: VOCABULARY,
            block: row_tile(VOCABULARY),
            extras: EXTRAS,
            divide: true,
            pdl: PDL,
        }),
    ));
    lines.push((
        "top_k".to_owned(),
        key(&ops::sampler::TopKDescriptor {
            dtype: DType::F32,
            vocabulary: VOCABULARY,
            width: CANDIDATE_WIDTH,
            // The chunk one ordering pass reads: the only size `top_k`'s key names now that the
            // kept count is a launch value and the ordering walks the row instead of sorting it.
            tile: TOP_TILE,
            ids: DType::I32,
            pdl: PDL,
        }),
    ));
    lines.push((
        "filter_candidates".to_owned(),
        key(&ops::sampler::FilterDescriptor {
            width: CANDIDATE_WIDTH,
            tile: CANDIDATE_TILE,
            ids: DType::I32,
            pdl: PDL,
        }),
    ));
    // The permutation between the filtering and the draw: it takes the candidate pair and writes it
    // in token order, so the draw does not depend on the order the top-k program produced. Its
    // candidate count is a launch value, so the key names only the two element types.
    lines.push((
        "candidate_order".to_owned(),
        key(&ops::sampler::OrderDescriptor {
            values: DType::F32,
            ids: DType::I32,
            pdl: PDL,
        }),
    ));
    lines.push((
        "inverse_cdf".to_owned(),
        key(&ops::sampler::InverseCdfDescriptor {
            width: CANDIDATE_WIDTH,
            tile: CANDIDATE_TILE,
            ids: DType::I32,
            tokens: DType::I32,
            pdl: PDL,
        }),
    ));
    lines.push((
        "argmax".to_owned(),
        key(&ops::sampler::ArgMaxDescriptor {
            dtype: DType::F32,
            // Greedy reads the row the head shaped, so the width is the vocabulary and not the
            // candidate width the top-k chain keeps (`nn::Sampler::greedy` passes `self.vocabulary`).
            width: VOCABULARY,
            ids: DType::I32,
            pdl: PDL,
        }),
    ));
    lines.push((
        // The candidates' probabilities: the general selection's step between the top-k program and
        // the filters, over the candidate width rather than the vocabulary.
        "softmax".to_owned(),
        // The tile is a chunk of the row, not the row: the program walks the width in tiles and
        // keeps a running maximum and denominator, which is what lets the candidate width be the
        // vocabulary. Its sibling `log_softmax` takes the row-sized step because its kernel derives
        // its own tile from the same number.
        key(&ops::norm::SoftmaxDescriptor {
            dtype: DType::F32,
            columns: CANDIDATE_WIDTH,
            tile: CANDIDATE_TILE,
            pdl: PDL,
        }),
    ));
    lines.push((
        "log_softmax".to_owned(),
        key(&ops::sampler::LogSoftmaxDescriptor {
            dtype: DType::F32,
            columns: VOCABULARY,
            tile: row_tile(VOCABULARY),
            pdl: PDL,
        }),
    ));
    lines.push((
        "logprob_gather".to_owned(),
        key(&ops::sampler::LogprobDescriptor {
            columns: VOCABULARY,
            ids: DType::I32,
            pdl: PDL,
        }),
    ));

    if let Some(path) = model_keys {
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let mut merged = 0;
                for line in text.lines() {
                    if line.starts_with('#') || line.trim().is_empty() {
                        continue;
                    }
                    if let Some((operator, key)) = line.split_once('\t') {
                        lines.push((operator.trim().to_owned(), key.trim().to_owned()));
                        merged += 1;
                    }
                }
                eprintln!("# {merged} rows merged from {}", path.display());
            }
            Err(error) => {
                eprintln!("{}: {error}", path.display());
                return;
            }
        }
    }

    // One key per line, the same key once: a variant two layers launch is packaged once.
    lines.sort();
    lines.dedup();
    for (operator, descriptor) in lines {
        println!("{operator}\t{descriptor}");
    }
}

/// A descriptor's key.
fn key(descriptor: &impl std::fmt::Display) -> String {
    descriptor.to_string()
}
