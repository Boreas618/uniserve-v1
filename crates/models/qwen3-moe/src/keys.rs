//! The programs a Qwen3-MoE deployment compiles, enumerated from its configuration.
//!
//! A fixture that names programs by hand drifts from the code that launches them: a layer gains a
//! descriptor field and the note keeps the old key, so the store builds a module nothing launches
//! and the launch finds no module. This walk removes the possibility for the model's own programs —
//! every key comes from the same constructor the layer passes to [`nn::Forward`] — and it does so
//! **without weights and without a device**: every descriptor reads geometry and element types, so a
//! configuration and a partition state the whole set. A deployment can therefore answer "what must
//! be compiled?" before it opens a device, and a machine that holds only a checkpoint's metadata can
//! answer it too.
//!
//! # The deployment's values
//!
//! Four values are the runtime's and not the checkpoint's, and the model must be told them:
//!
//! - `page_tokens`, the prefix cache's page;
//! - `cache_dtype`, the planes the cache holds;
//! - `capacity`, the exchange's rows per rank on a deployment that shards the experts, which the
//!   exchange path refuses as zero because that is the size of its buffers;
//! - `buckets`, the row counts the executor captures, because **the dense packed product is compiled
//!   per row count**: its grid is `m` rows, so a module serves one `m` and the deployment states the
//!   finite set it pads to (`specs/replaceable-processes.md`: "a forward no bucket holds is an error,
//!   never an eager fallback"). An empty list is refused when the projections are packed, because a
//!   store built from it would hold no product for any row count the executor runs.
//!
//! The asymmetry is worth reading twice: the **expert** products' grid is over *routes*
//! (tokens × top, computed at launch) and their keys carry no row count, so `buckets` does not
//! multiply them, and the encoder's grid covers its rows in one launch so it is keyed by `columns`,
//! not by `m`. Only `nvfp4_gemm` — the dense encoded product — is per bucket.
//!
//! # What is not here
//!
//! Two families of programs are not the model's, and this walk does not invent them: the sampler's,
//! which the executor builds from its request's sampling settings, and the batch's, which the
//! executor's own columns decide.

use nn::DType;
use nn::layers::{
    Attention, Embedding, Linear, Moe, MoeGeometry, Nvfp4Linear, ParallelLMHead, RmsNorm, Rotary,
    TILE_ROWS,
};
use ops::Op;
use ops::attention::Phase;
use ops::gemm::WeightOrder;
use ops::moe::exchange::Mode;
use ops::moe::grouped::{GatherGemmNvfp4, RouteGemmNvfp4};
use ops::fp4::ScaleLayout;

use crate::load::activation;
use crate::{Config, Error, Partition, Qwen3Moe, Result};

/// The cache-write row a deployment compiles for `cache_dtype`: the operator's name and its key.
///
/// `nn` dispatches the write on the plane's type — an E4M3 plane is written through the scaled write,
/// which divides by the page's own scale on the way in, and every other plane through the plain
/// scatter — so the row must follow the dispatch. A row that named one program for both planes
/// builds a module nothing launches and leaves the launched one absent, and the failure then reads
/// as a missing program rather than as this function's.
pub fn write_row(
    attention: &Attention,
    projection: DType,
    cache: DType,
) -> (&'static str, String) {
    if cache == DType::F8E4M3 {
        (
            <ops::kv::KvWriteScaled as Op>::NAME,
            attention.scaled_write_descriptor(projection).to_string(),
        )
    } else {
        (
            <ops::kv::KvWrite as Op>::NAME,
            attention.write_descriptor(projection, cache).to_string(),
        )
    }
}

impl Qwen3Moe {
    /// Every program this loaded model launches, as `(operator, key)` pairs, sorted and unique.
    ///
    /// The loaded form states what the loader read: the model's own dtype, its page geometry, and
    /// whether its experts are encoded. Its projections are dense — a checkpoint whose projections
    /// are packed is refused by [`crate::load::load_checkpoint`] until the encoded projection path
    /// is wired — so `projections_packed` is `false` here, and a deployment that wants the row a
    /// packed projection will need asks the static [`keys`] directly.
    pub fn keys(
        &self,
        cache_dtype: DType,
        capacity: u32,
        buckets: &[u32],
    ) -> Result<Vec<(String, String)>> {
        keys(
            self.config(),
            self.partition(),
            self.dtype(),
            self.cache_page_tokens(),
            cache_dtype,
            capacity,
            buckets,
            self.experts_encoded(),
            false,
        )
    }
}

/// Every program a deployment of `config` on `partition` compiles, as `(operator, key)` pairs.
///
/// `experts_encoded` states that the checkpoint's expert stacks are calibrated NVFP4, which makes
/// every routed layer's step encode its activations and read the packed products; `projections_packed`
/// states the same for the attention and dense-MLP projections, whose product is
/// [`ops::fp4::Nvfp4Gemm`] and is compiled **once per entry of `buckets`**.
#[allow(clippy::too_many_arguments)]
pub fn keys(
    config: &Config,
    partition: &Partition,
    dtype: DType,
    page_tokens: u32,
    cache_dtype: DType,
    capacity: u32,
    buckets: &[u32],
    experts_encoded: bool,
    projections_packed: bool,
) -> Result<Vec<(String, String)>> {
    let width = |value: usize, what: &str| -> Result<u32> {
        u32::try_from(value).map_err(|_| Error::Config(format!("this checkpoint's {what} is beyond u32")))
    };
    let mut keys: Vec<(String, String)> = Vec::new();
    let mut push = |operator: &str, key: String| keys.push((operator.to_owned(), key));

    if projections_packed && buckets.is_empty() {
        return Err(Error::Config(
            "this deployment's projections are packed NVFP4, whose product is compiled per row \
             count, and the bucket list given is empty: a store built from it would hold no product \
             for any row count the executor captures, and a forward no bucket holds is an error \
             rather than an eager fallback"
                .to_owned(),
        ));
    }

    let hidden = width(config.hidden_size, "hidden width")?;
    let head_dim = width(config.head_dim, "head width")?;
    let q_heads = width(partition.local_attention_heads(config), "query heads")?;
    let kv_heads = width(partition.local_key_value_heads(config), "key/value heads")?;
    let query = q_heads * head_dim;
    let key_value = kv_heads * head_dim;
    let dense_width = width(config.intermediate_size, "dense intermediate width")?;
    let moe = MoeGeometry {
        hidden: u64::from(hidden),
        intermediate: u64::from(width(config.moe_intermediate_size, "expert intermediate width")?),
        top: u64::from(width(config.num_experts_per_tok, "routes per token")?),
        experts: u64::from(width(config.num_experts, "expert count")?),
        tile: u64::from(TILE_ROWS),
    };

    // The vocabulary: this rank's shard of the embedding, and the head that projects it. The shard is
    // the deployment's rank count, the vocabulary is the checkpoint's.
    let (first, _) = partition.vocabulary_shard(config.vocab_size);
    let shard = width(partition.padded_vocabulary_shard(config.vocab_size), "vocabulary shard")?;
    let vocabulary = width(config.vocab_size, "vocabulary")?;
    push(
        "embedding",
        Embedding::descriptor_of(
            shard,
            hidden,
            width(first, "vocabulary shard start")?,
            dtype,
            DType::I32,
        )
        .to_string(),
    );

    let attention = Attention::paged(
        head_dim,
        q_heads,
        kv_heads,
        page_tokens,
        width(config.max_position_embeddings, "max_position_embeddings")?,
        None,
        1.0 / (head_dim as f32).sqrt(),
    )
    .map_err(Error::from)?;

    for layer in 0..config.num_hidden_layers {
        // The fused residual-add and normalization that opens each sublayer.
        push("add_rms_norm", RmsNorm::add_descriptor_of(hidden, dtype).to_string());
        push("add_rms_norm", RmsNorm::add_descriptor_of(hidden, dtype).to_string());

        // The rotation's own table, and the fused per-head normalization that reads it: the affine
        // form, because the checkpoint has learned Q/K norm weights.
        push(
            "rotary_factors",
            Rotary::factors_descriptor_of(head_dim, 1.0, None, DType::I32).to_string(),
        );
        push(
            "qk_norm_rope",
            Rotary::apply_descriptor_of(
                head_dim,
                q_heads,
                kv_heads,
                nn::layers::Rotation::Half,
                config.rms_norm_eps,
                head_dim,
                dtype,
                dtype,
                true,
            )
            .to_string(),
        );

        // The four attention projections: one dense product each, and one packed product each per
        // bucket when the checkpoint's projections are calibrated NVFP4.
        let projections = [
            (query, hidden),
            (key_value, hidden),
            (key_value, hidden),
            (hidden, query),
        ];
        for (n, k) in projections {
            push(
                <ops::gemm::Gemm as Op>::NAME,
                Linear::descriptor_of(k, dtype, WeightOrder::NxK, false, dtype, dtype).to_string(),
            );
            if projections_packed {
                for rows in buckets {
                    push(
                        <ops::fp4::Nvfp4Gemm as Op>::NAME,
                        Nvfp4Linear::descriptor_of(*rows, n, k, ScaleLayout::Swizzled128X4, false, dtype)
                            .to_string(),
                    );
                }
            }
        }
        if projections_packed {
            for k in [hidden, query, hidden] {
                push(
                    <ops::fp4::Nvfp4Quantize as Op>::NAME,
                    Moe::quantize_descriptor_of(dtype, k, ScaleLayout::Swizzled128X4).to_string(),
                );
            }
        }

        // The cache write and the two attention phases.
        let (operator, key) = write_row(&attention, dtype, cache_dtype);
        push(operator, key);
        for phase in [Phase::Prefill, Phase::Decode] {
            push(
                "paged_attention",
                attention.descriptor(phase, dtype, cache_dtype, dtype).to_string(),
            );
        }

        if config.sparse(layer) {
            push("topk_softmax", moe.topk_descriptor(dtype, config.norm_topk_prob, None).to_string());
            push("expert_gather", moe.map_descriptor().to_string());
            // The sort covers the experts this rank owns: every expert of a dense deployment, its
            // own shard under expert parallelism.
            let local = width(partition.expert_range(config).len(), "local expert count")?;
            push("moe_sort", moe.sort_descriptor(local).to_string());
            if experts_encoded {
                for k in [hidden, width(config.moe_intermediate_size, "expert intermediate width")?] {
                    push(
                        <ops::fp4::Nvfp4Quantize as Op>::NAME,
                        Moe::quantize_descriptor_of(dtype, k, ScaleLayout::Swizzled128X4).to_string(),
                    );
                }
                push(
                    <GatherGemmNvfp4 as Op>::NAME,
                    moe.gather_nvfp4_descriptor(dtype, activation(config.hidden_act)?, DType::F8E4M3, ScaleLayout::Swizzled128X4)
                        .to_string(),
                );
                push(
                    <RouteGemmNvfp4 as Op>::NAME,
                    moe.route_nvfp4_descriptor(dtype, DType::F8E4M3, ScaleLayout::Swizzled128X4).to_string(),
                );
            } else {
                push(
                    "gather_gemm",
                    moe.gather_descriptor(dtype, activation(config.hidden_act)?).to_string(),
                );
                push("route_gemm", moe.route_descriptor(dtype).to_string());
            }
            // The combine reads the *products'* dtype, which is the model's, and rounds once into
            // `dtype`: the walk that named `F32` here named a program the layer never launches.
            push("moe_unpermute", moe.combine_descriptor(dtype, dtype).to_string());
        } else {
            for (n, k) in [(dense_width, hidden), (dense_width, hidden), (hidden, dense_width)] {
                push(
                    <ops::gemm::Gemm as Op>::NAME,
                    Linear::descriptor_of(k, dtype, WeightOrder::NxK, false, dtype, dtype).to_string(),
                );
                if projections_packed {
                    for rows in buckets {
                        push(
                            <ops::fp4::Nvfp4Gemm as Op>::NAME,
                            Nvfp4Linear::descriptor_of(*rows, n, k, ScaleLayout::Swizzled128X4, false, dtype)
                                .to_string(),
                        );
                    }
                }
            }
        }
    }

    // The output norm, and the vocabulary head's two programs.
    push("rms_norm", RmsNorm::descriptor_of(hidden, dtype).to_string());
    push(
        <ops::gemm::Gemm as Op>::NAME,
        Linear::descriptor_of(hidden, dtype, WeightOrder::NxK, false, dtype, dtype).to_string(),
    );
    let gathered = shard * width(partition.tensor_parallel, "tensor parallel width")?;
    push(
        "vocab_unpad",
        ParallelLMHead::unpad_descriptor_of(vocabulary, dtype, DType::F32, gathered).to_string(),
    );

    // The expert exchange, on a deployment that splits the experts.
    let ranks = partition.expert_parallel;
    if ranks > 1 {
        let ranks = width(ranks, "expert parallel width")?;
        if capacity == 0 {
            return Err(Error::Config(
                "this deployment splits the experts and its exchange carries rows per rank, and the \
                 capacity given is 0"
                    .to_owned(),
            ));
        }
        let dispatch = ops::moe::exchange::Descriptor {
            mode: Mode::Dispatch,
            hidden,
            top: width(config.num_experts_per_tok, "routes per token")?,
            experts: width(config.num_experts, "expert count")?,
            max_tokens: capacity,
            ranks,
            dtype,
        };
        for mode in [Mode::Dispatch, Mode::Combine] {
            push(
                <ops::moe::exchange::ExpertExchange as Op>::NAME,
                ops::moe::exchange::Descriptor { mode, ..dispatch }.to_string(),
            );
        }
    }

    keys.sort();
    keys.dedup();
    Ok(keys)
}
