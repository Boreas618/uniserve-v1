//! The programs a Qwen3-MoE deployment compiles, enumerated from the loaded model.
//!
//! A fixture that names programs by hand drifts from the code that launches them: a layer gains a
//! descriptor field and the note keeps the old key, so the store builds a module nothing launches
//! and the launch finds no module. This walk removes the possibility for the model's own programs —
//! every key here comes from the same `descriptor` accessor the layer passes to [`nn::Forward`], so
//! the two cannot disagree.
//!
//! It is device-free by construction: the accessors read the layer's geometry and element types,
//! which the checkpoint's metadata already fixed, and none of them touches an arena, a batch or a
//! stream. A deployment can therefore answer "what must be compiled?" before it opens a device.
//!
//! # What is not here
//!
//! Two families of programs are not the model's, and this walk does not invent them:
//!
//! - **the sampler's**, which the executor builds from its request's sampling settings
//!   (`nn::layers::Sampler`), so the sampling keys are enumerated where the sampler is built;
//! - **the cache's and the batch's**, `kv_write` excepted because the attention layer owns its page
//!   geometry, whose page size and context bound are the deployment's.
//!
//! The returned list is sorted and deduplicated: a program several layers launch is compiled once,
//! and a key that differs only by a layer index is one key.

use nn::DType;
use ops::Op;
use ops::attention::Phase;

use crate::model::Mlp;
use crate::{Error, Qwen3Moe, Result};

impl Qwen3Moe {
    /// Every program this deployment compiles, as `(operator, key)` pairs, sorted and unique.
    ///
    /// `cache_dtype` is the prefix cache's plane type, which the deployment chooses and the model
    /// sees only in the attention layer's write and its read; every other element type is the
    /// model's own (`self.dtype`), the query's and the output's.
    ///
    /// The result is what a store must hold for this model to run: `build_store` consumes the same
    /// pairs a fixture prints, so `keys()` and the note cannot disagree.
    pub fn keys(&self, cache_dtype: DType) -> Result<Vec<(String, String)>> {
        let dtype = self.dtype();
        let mut keys: Vec<(String, String)> = Vec::new();
        let mut push = |operator: &str, key: String| keys.push((operator.to_owned(), key));

        // The vocabulary embedding: the checkpoint's matrix, its shard, and the ids' type.
        push(
            "embedding",
            self.embedding().descriptor(dtype, DType::I32).to_string(),
        );

        for layer in self.layers() {
            // The fused residual-add and normalization that opens each sublayer.
            push(
                "add_rms_norm",
                layer.input_norm.add_descriptor(dtype).to_string(),
            );
            push(
                "add_rms_norm",
                layer.post_attention_norm.add_descriptor(dtype).to_string(),
            );

            let attention = &layer.attention;
            // The rotation's own table, and the fused per-head normalization that reads it.
            push(
                "rotary_factors",
                attention.rotary.factors_descriptor(DType::I32).to_string(),
            );
            // The Q/K norms are the checkpoint's own weights, so the affine form is the one a loaded
            // model launches; `false` here would name a program this model never runs.
            push(
                "qk_norm_rope",
                attention
                    .rotary
                    .apply_descriptor(dtype, dtype, true)
                    .to_string(),
            );

            // The four projections are dense products; the type pair names the program, and the row
            // count is the launch's, so one key covers every sequence length.
            for projection in [
                &attention.q_proj,
                &attention.k_proj,
                &attention.v_proj,
                &attention.o_proj,
            ] {
                push(
                    <ops::gemm::Gemm as Op>::NAME,
                    projection.descriptor(dtype, dtype).to_string(),
                );
            }

            // The cache write and the two attention phases: the same layer, three programs, because
            // a prefill reads a whole sequence and a decode reads one token per row.
            push(
                "kv_write",
                attention
                    .attention
                    .write_descriptor(dtype, cache_dtype)
                    .to_string(),
            );
            for phase in [Phase::Prefill, Phase::Decode] {
                push(
                    "paged_attention",
                    attention
                        .attention
                        .descriptor(phase, dtype, cache_dtype, dtype)
                        .to_string(),
                );
            }

            match &layer.mlp {
                Mlp::Experts(moe) => {
                    push("topk_softmax", moe.topk_descriptor(dtype).to_string());
                    push("expert_gather", moe.map_descriptor().to_string());
                    // The sort covers the experts this rank owns: every expert of a dense
                    // deployment, its own shard under expert parallelism.
                    let local = self.partition().expert_range(self.config()).len() as u32;
                    push("moe_sort", moe.sort_descriptor(local).to_string());
                    push("gather_gemm", moe.gather_descriptor(dtype).to_string());
                    push("route_gemm", moe.route_descriptor(dtype).to_string());
                    push(
                        "moe_unpermute",
                        moe.combine_descriptor(DType::F32, dtype).to_string(),
                    );
                }
                Mlp::Dense(dense) => {
                    // The dense block's two products share the routed path's program: the expert
                    // count is one and the activation is the same program's, so the key differs only
                    // in its extents — which are launch values.
                    for projection in [&dense.gate_proj, &dense.up_proj, &dense.down_proj] {
                        push(
                            <ops::gemm::Gemm as Op>::NAME,
                            projection.descriptor(dtype, dtype).to_string(),
                        );
                    }
                }
            }
        }

        // The output norm, and the vocabulary head's two programs: the shard's projection and the
        // unpad that gathers the ranks' shards back to the checkpoint's vocabulary.
        push("rms_norm", self.norm().descriptor(dtype).to_string());
        push(
            <ops::gemm::Gemm as Op>::NAME,
            self.head()
                .projection_descriptor(dtype, dtype)
                .map_err(Error::from)?
                .to_string(),
        );
        // The gathered width is every rank's shard side by side: the head's padded width times the
        // tensor-parallel degree. The model states both, so the key follows the deployment's rank
        // count and never a guess.
        let gathered = self.head().padded() * self.partition().tensor_parallel as u32;
        push(
            "vocab_unpad",
            self.head().unpad_descriptor(dtype, gathered).to_string(),
        );

        keys.sort();
        keys.dedup();
        Ok(keys)
    }
}
