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

use ops::attention::{Descriptor as AttentionDescriptor, Mask, Phase};
use ops::moe;
use ops::rope::{FactorsDescriptor, MAX_AXES, QkNormRopeDescriptor, Rotation};
use ops::{DType, Float};

/// Whether the layers' descriptors are the programmatic-dependent-launch variants.
const PDL: bool = false;

/// Tokens one `embedding` program handles (`launch::EMBEDDING_TOKENS`).
const EMBEDDING_TOKENS: u32 = 8;

/// Elements one `rotary_factors` program handles (`launch::ROTARY_BLOCK`).
const ROTARY_BLOCK: u32 = 256;

/// Tokens one `qk_norm_rope` program handles (`launch::QK_NORM_TOKENS`).
const QK_NORM_TOKENS: u32 = 1;

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

/// The candidate width the slice's sampler keeps per row (`Sampler::new`'s `width`).
const CANDIDATE_WIDTH: u32 = 2048;

/// Extra counted tokens a sampling row carries in the slice's request state.
const EXTRAS: u32 = 0;

// The checkpoint's geometry.
const HIDDEN: u32 = 2048;
const HEADS: u32 = 32;
const KV_HEADS: u32 = 4;
const HEAD_DIM: u32 = 128;
const EXPERTS: u32 = 128;
const TOP: u32 = 8;
const INTERMEDIATE: u32 = 768;
/// FC1's fused output width: the gate and up projections side by side.
const FUSED: u32 = 2 * INTERMEDIATE;
const VOCABULARY: u32 = 151936;
const PAGE: u32 = 64;
const MAX_KV_LEN: u32 = 40960;

/// The row width of a projection whose contraction is `k`; the keys only carry the contraction.
const PROJECTION_K: [u32; 3] = [HIDDEN, HEADS * HEAD_DIM, INTERMEDIATE];

fn main() {
    let dtype = DType::BF16;
    let mut lines: Vec<(String, String)> = Vec::new();

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
        "expert_gather".to_owned(),
        key(&ops::moe::routing::GatherDescriptor {
            experts: EXPERTS,
            top: TOP,
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
        }),
    ));
    lines.push((
        "gather_gemm".to_owned(),
        key(&moe::grouped::GatherGemmDescriptor {
            hidden: HIDDEN,
            intermediate: FUSED,
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

    // Normalization, rotation and the embedding.
    for columns in [HIDDEN, HEAD_DIM] {
        lines.push((
            "rms_norm".to_owned(),
            key(&ops::norm::RmsNormDescriptor {
                dtype,
                columns,
                tile: row_tile(columns),
                pdl: PDL,
            }),
        ));
    }
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
            // Qwen3's rope theta is 1e6, from the checkpoint's configuration.
            scale: Float::from(1_000_000.0_f32),
            pdl: PDL,
            // One-dimensional positions: Qwen3 is text only, so no section is cut.
            axes: 1,
            sections: [0; MAX_AXES],
        }),
    ));
    lines.push((
        "qk_norm_rope".to_owned(),
        key(&QkNormRopeDescriptor {
            dtype,
            output: dtype,
            head_dim: HEAD_DIM,
            q_heads: HEADS,
            k_heads: KV_HEADS,
            // Qwen3 rotates a feature against the opposite half of its head (`rotate_half`).
            rotation: Rotation::Half,
            // The checkpoint's rms_norm_eps, by bit pattern.
            epsilon: Float::from(1e-6_f32),
            rows: QK_NORM_TOKENS,
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
            vocabulary: VOCABULARY,
            padded: VOCABULARY,
            block: 1024,
            pdl: PDL,
        }),
    ));
    lines.push((
        "logits_shape".to_owned(),
        key(&ops::sampler::ShapeDescriptor {
            dtype,
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
            ids: DType::I32,
            pdl: PDL,
        }),
    ));
    lines.push((
        "filter_candidates".to_owned(),
        key(&ops::sampler::FilterDescriptor {
            width: CANDIDATE_WIDTH,
            ids: DType::I32,
            pdl: PDL,
        }),
    ));
    lines.push((
        "inverse_cdf".to_owned(),
        key(&ops::sampler::InverseCdfDescriptor {
            width: CANDIDATE_WIDTH,
            ids: DType::I32,
            tokens: DType::I32,
            pdl: PDL,
        }),
    ));
    lines.push((
        "argmax".to_owned(),
        key(&ops::sampler::ArgMaxDescriptor {
            dtype: DType::F32,
            width: CANDIDATE_WIDTH,
            ids: DType::I32,
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
