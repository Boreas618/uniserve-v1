//! The declarative weight mapping of `Qwen3MoeForCausalLM`.
//!
//! Every parameter of the loaded model is named here once, together with the checkpoint tensor it
//! comes from (`uniserve/models/qwen3/weights.py`, `parameter_sources`/`expert_sources`). The
//! mapping is data: it names no device, no layout and no partition — the [`WeightLoader`] applies
//! the deployment's partition and the layout the reading operator declares, and converts the
//! element type once (specs/worker.md 4.3, 8).
//!
//! [`WeightLoader`]: nn::WeightLoader
//!
//! ## Parameter names
//!
//! The names below are the model's own paths, composed exactly as the model is:
//! `embedding.weight`, `norm.weight`, `lm_head.weight`, `layers.<i>.input_norm.weight`,
//! `layers.<i>.attention.{q,k,v,o}_proj.weight`, `layers.<i>.attention.{q,k}_norm.weight`,
//! `layers.<i>.post_attention_norm.weight`, `layers.<i>.mlp.router.weight`, and for a dense layer
//! `layers.<i>.mlp.{gate,up,down}_proj.weight`. Sparse layers' expert tensors are named by
//! [`expert_sources`] because the checkpoint stores one tensor per expert.
//!
//! ## Checkpoint prefix
//!
//! Released checkpoints store the stack under `model.layers.<i>…`, and every name in this table is
//! that canonical form. [`WeightLoader`] matches a checkpoint tensor's name **exactly** - the load
//! path has no `model.`-prefix fallback - so a checkpoint serialized without the prefix is refused
//! by the name it lacks rather than silently bound to a different tensor. The audit
//! (`examples/audit_checkpoint.rs`) strips the prefix itself, only to report how many names the
//! prefix accounts for, and `declared_not_consumed` does the same to recognise the two calibrated
//! cache scales under either form; neither is the load.

use crate::Config;

/// One model parameter and the checkpoint tensor that fills it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ParameterSource {
    /// The model's path for the parameter, e.g. `layers.3.attention.q_proj.weight`.
    pub parameter: String,
    /// The checkpoint tensor's name, e.g. `model.layers.3.self_attn.q_proj.weight`.
    pub checkpoint: String,
}

impl ParameterSource {
    fn new(parameter: impl Into<String>, checkpoint: impl Into<String>) -> Self {
        Self {
            parameter: parameter.into(),
            checkpoint: checkpoint.into(),
        }
    }
}

/// The checkpoint prefix of one sparse layer's per-expert tensors.
///
/// The checkpoint stores expert `e` of prefix `model.layers.<i>.mlp.experts` as
/// `model.layers.<i>.mlp.experts.<e>.{gate,up,down}_proj.weight`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExpertSource {
    /// The model's path for the stacked expert module, e.g. `layers.3.mlp.experts`.
    pub module: String,
    /// The checkpoint prefix holding one tensor per expert.
    pub checkpoint_prefix: String,
}

/// One expert's three checkpoint tensor names.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExpertTensors {
    /// The causal (down) projection, `[hidden, intermediate]`.
    pub down: String,
    /// The up projection, `[intermediate, hidden]`.
    pub up: String,
    /// The gate projection, `[intermediate, hidden]`.
    pub gate: String,
}

impl ExpertSource {
    /// The checkpoint tensor names of expert `expert` under this source.
    pub fn tensors(&self, expert: usize) -> ExpertTensors {
        let at = |projection: &str| {
            format!(
                "{}.{expert}.{projection}_proj.weight",
                self.checkpoint_prefix
            )
        };
        ExpertTensors {
            down: at("down"),
            up: at("up"),
            gate: at("gate"),
        }
    }
}

/// One expert's three projections in an encoded (NVFP4) checkpoint.
///
/// A quantized checkpoint stores each projection three times over: the packed E2M1 nibbles in
/// `…{projection}_proj.weight` (`[rows, columns/2]` U8), the block scales in
/// `…{projection}_proj.weight_scale` (one E4M3 scale per sixteen elements in the package's swizzled
/// layout), and the projection's own FP32 scale in `…{projection}_proj.weight_scale_2`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EncodedProjection {
    /// The packed weights, two E2M1 codes per byte along the contraction.
    pub values: String,
    /// The block scales.
    pub scales: String,
    /// The projection's FP32 tensor scale, whose reciprocal the loader carries.
    pub global_scale: String,
}

/// One expert's three encoded projections.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EncodedTensors {
    /// The causal (down) projection.
    pub down: EncodedProjection,
    /// The up projection.
    pub up: EncodedProjection,
    /// The gate projection.
    pub gate: EncodedProjection,
}

impl ExpertSource {
    /// The encoded checkpoint tensor names of expert `expert` under this source.
    ///
    /// The suffix set is the one ModelOpt writes and `WeightLoader::nvfp4` consumes; a checkpoint
    /// that names them differently is a layout the loader does not know and must refuse rather than
    /// reinterpret bytes.
    pub fn encoded(&self, expert: usize) -> EncodedTensors {
        let at = |projection: &str| {
            let stem = format!("{}.{expert}.{projection}_proj", self.checkpoint_prefix);
            EncodedProjection {
                values: format!("{stem}.weight"),
                scales: format!("{stem}.weight_scale"),
                global_scale: format!("{stem}.weight_scale_2"),
            }
        };
        EncodedTensors {
            down: at("down"),
            up: at("up"),
            gate: at("gate"),
        }
    }
}

/// Why a tensor a ModelOpt checkpoint declares is declared here and not consumed.
pub const CALIBRATED_CACHE_SCALE: &str = "a calibrated per-tensor FP8 cache scalar, declared by the \
checkpoint's `kv_cache_quant_algo FP8`; the cache takes one scale per page and derives it from the \
values it writes, and the reference's quantizer refuses a calibrated scalar for FP8";

/// The tensors a checkpoint declares that this model knows about and deliberately does not consume.
///
/// A ModelOpt checkpoint quantized to NVFP4 with `kv_cache_quant_algo: FP8` carries, in every layer,
/// `self_attn.k_proj.k_scale` and `self_attn.v_proj.v_scale` as F32 scalars. They calibrate a
/// *per-tensor* FP8 K/V cache. This model's cache stores **one scale per page** and derives it from
/// the values it writes (`uniserve/prefix_cache/mha.py:282-299`), and a calibrated scalar is
/// inexpressible for an FP8 cache by the reference quantizer's own rule — `calibrated_scale` is
/// refused unless the format is `nvfp4` (`uniserve/quantization/quantizer.py:63-70`). A per-tensor
/// scalar for the whole cache would be a different layout with a different reader, and seeding one
/// into a page's buffer would change nothing: a page whose initialized flag is clear takes the
/// derived scale.
///
/// Returns the reason for such a name, and `None` for every other tensor — so a loader or an audit
/// can say *declared, not consumed* rather than silently dropping two F32 tensors per layer, which is
/// indistinguishable from never having seen them.
pub fn declared_not_consumed(name: &str) -> Option<&'static str> {
    let stripped = name.strip_prefix("model.").unwrap_or(name);
    for suffix in [".self_attn.k_proj.k_scale", ".self_attn.v_proj.v_scale"] {
        if let Some(prefix) = stripped.strip_suffix(suffix) {
            // `layers.<index>`: the declaration is per layer, and a name outside the stack is not one.
            let per_layer = prefix.strip_prefix("layers.").is_some_and(|index| {
                !index.is_empty() && index.chars().all(|character| character.is_ascii_digit())
            });
            if per_layer {
                return Some(CALIBRATED_CACHE_SCALE);
            }
        }
    }
    None
}

/// Every parameter of a Qwen3-MoE model, keyed by the model's path.
///
/// Sparse layers contribute their router here and their experts through [`expert_sources`]. A tied
/// head maps to the embedding tensor: the checkpoint stores one matrix and the loader gives both
/// parameters the same retained storage.
pub fn parameter_sources(config: &Config) -> Vec<ParameterSource> {
    let mut sources = vec![
        ParameterSource::new("embedding.weight", "model.embed_tokens.weight"),
        ParameterSource::new("norm.weight", "model.norm.weight"),
        ParameterSource::new(
            "lm_head.weight",
            if config.tie_word_embeddings {
                "model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            },
        ),
    ];

    for layer in 0..config.num_hidden_layers {
        let parameter = format!("layers.{layer}");
        let checkpoint = format!("model.layers.{layer}");

        sources.push(ParameterSource::new(
            format!("{parameter}.input_norm.weight"),
            format!("{checkpoint}.input_layernorm.weight"),
        ));
        sources.push(ParameterSource::new(
            format!("{parameter}.post_attention_norm.weight"),
            format!("{checkpoint}.post_attention_layernorm.weight"),
        ));

        let projections = if config.attention_bias {
            &["weight", "bias"][..]
        } else {
            &["weight"][..]
        };
        for field in projections {
            for branch in ["q", "k", "v"] {
                sources.push(ParameterSource::new(
                    format!("{parameter}.attention.{branch}_proj.{field}"),
                    format!("{checkpoint}.self_attn.{branch}_proj.{field}"),
                ));
            }
            sources.push(ParameterSource::new(
                format!("{parameter}.attention.o_proj.{field}"),
                format!("{checkpoint}.self_attn.o_proj.{field}"),
            ));
        }

        for branch in ["q", "k"] {
            sources.push(ParameterSource::new(
                format!("{parameter}.attention.{branch}_norm.weight"),
                format!("{checkpoint}.self_attn.{branch}_norm.weight"),
            ));
        }

        if config.sparse(layer) {
            sources.push(ParameterSource::new(
                format!("{parameter}.mlp.router.weight"),
                format!("{checkpoint}.mlp.gate.weight"),
            ));
        } else {
            for branch in ["gate", "up"] {
                sources.push(ParameterSource::new(
                    format!("{parameter}.mlp.{branch}_proj.weight"),
                    format!("{checkpoint}.mlp.{branch}_proj.weight"),
                ));
            }
            sources.push(ParameterSource::new(
                format!("{parameter}.mlp.down_proj.weight"),
                format!("{checkpoint}.mlp.down_proj.weight"),
            ));
        }
    }

    sources
}

/// The sparse layers' per-expert checkpoint prefixes.
pub fn expert_sources(config: &Config) -> Vec<ExpertSource> {
    (0..config.num_hidden_layers)
        .filter(|&layer| config.sparse(layer))
        .map(|layer| ExpertSource {
            module: format!("layers.{layer}.mlp.experts"),
            checkpoint_prefix: format!("model.layers.{layer}.mlp.experts"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The declaration is per layer and in both directions: the model knows these names, must not
    /// claim them, and must not treat any other tensor as one of them.
    #[test]
    fn the_calibrated_cache_scalars_are_declared_and_not_consumed() {
        for layer in [0, 1, 47] {
            for projection in ["k_proj.k_scale", "v_proj.v_scale"] {
                for name in [
                    format!("model.layers.{layer}.self_attn.{projection}"),
                    // Without the prefix the stack is serialized under, the loader's fallback form.
                    format!("layers.{layer}.self_attn.{projection}"),
                ] {
                    assert_eq!(
                        declared_not_consumed(&name),
                        Some(CALIBRATED_CACHE_SCALE),
                        "{name}"
                    );
                }
            }
        }
        assert!(CALIBRATED_CACHE_SCALE.contains("derives it"));
        for name in [
            "model.layers.3.self_attn.k_proj.weight",
            "model.layers.3.self_attn.k_proj.input_scale",
            "model.layers.3.self_attn.q_proj.k_scale",
            "model.layers.3.self_attn.v_proj.weight_scale",
            "model.layers.3.mlp.gate.weight",
            "lm_head.weight",
            // A name that carries the suffix outside the stack is not the declaration.
            "self_attn.k_proj.k_scale",
            "model.layers.x.self_attn.k_proj.k_scale",
        ] {
            assert_eq!(declared_not_consumed(name), None, "{name}");
        }
    }
    use crate::{Activation, Config};
    use nn::Shape;

    #[test]
    fn an_encoded_expert_names_the_packed_scales_and_tensor_scale_the_loader_reads() {
        let source = ExpertSource {
            module: "layers.3.mlp.experts".to_owned(),
            checkpoint_prefix: "model.layers.3.mlp.experts".to_owned(),
        };
        let expert = source.encoded(17);
        // Three names per projection, in the suffix set `WeightLoader::nvfp4` consumes; a checkpoint
        // that spells them otherwise is a layout the loader refuses rather than reinterprets.
        assert_eq!(
            expert.up.values,
            "model.layers.3.mlp.experts.17.up_proj.weight"
        );
        assert_eq!(
            expert.up.scales,
            "model.layers.3.mlp.experts.17.up_proj.weight_scale"
        );
        assert_eq!(
            expert.up.global_scale,
            "model.layers.3.mlp.experts.17.up_proj.weight_scale_2"
        );
        assert!(expert.down.values.ends_with("17.down_proj.weight"));
        assert!(expert.gate.scales.ends_with("17.gate_proj.weight_scale"));
        // The dense path's names are untouched by the encoded one.
        assert_eq!(
            source.tensors(17).up,
            "model.layers.3.mlp.experts.17.up_proj.weight"
        );
    }

    fn config(sparse_layers: &[usize], bias: bool, tied: bool) -> Config {
        Config {
            vocab_size: 151936,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 4,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            head_dim: 128,
            hidden_act: Activation::Silu,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_scaling: None,
            max_position_embeddings: 40960,
            attention_bias: bias,
            tie_word_embeddings: tied,
            num_experts: 128,
            num_experts_per_tok: 8,
            moe_intermediate_size: 768,
            norm_topk_prob: true,
            decoder_sparse_step: 1,
            mlp_only_layers: (0..4)
                .filter(|layer| !sparse_layers.contains(layer))
                .collect(),
        }
    }

    #[test]
    fn the_slice_checkpoint_maps_every_parameter() {
        // The local 30B-A3B metadata, when the machine has it: 48 sparse layers, untied head.
        // The checkpoint's metadata, when the Hub cache holds it.
        let Ok(root) = nn::hub::model_dir(crate::REPO_ID) else {
            return;
        };
        let path = root.join("config.json");
        let path = path.as_path();
        if !path.is_file() {
            return;
        }
        let config = crate::Config::read(path.parent().expect("a parent directory"))
            .expect("the local Qwen3-30B-A3B metadata");
        let sources = parameter_sources(&config);
        // embedding, norm and head; then per layer two norms, four projections (q, k, v, o), two
        // per-head norms and the router.
        assert_eq!(sources.len(), 3 + 48 * 9);
        let names: Vec<&str> = sources
            .iter()
            .map(|source| source.checkpoint.as_str())
            .collect();
        assert!(names.contains(&"model.embed_tokens.weight"));
        assert!(names.contains(&"lm_head.weight"));
        assert!(names.contains(&"model.layers.47.self_attn.q_norm.weight"));
        assert!(names.contains(&"model.layers.47.mlp.gate.weight"));
        assert!(!names.iter().any(|name| name.contains("mlp.experts")));
        assert_eq!(expert_sources(&config).len(), 48);
        let experts = expert_sources(&config);
        assert_eq!(
            experts[7].tensors(3).up,
            "model.layers.7.mlp.experts.3.up_proj.weight"
        );
    }

    #[test]
    fn every_parameter_of_the_local_checkpoint_has_the_shape_the_loader_expects() {
        // The name mapping alone cannot catch a transposed or mis-shaped request; the checkpoint's
        // own headers can, before anything reaches a device. Shapes are derived from `config.json`
        // here rather than from the loader, so the two agree only if both read the design right.
        // The checkpoint's metadata, when the Hub cache holds it.
        let Ok(root) = nn::hub::model_dir(crate::REPO_ID) else {
            return;
        };
        let path = root.join("config.json");
        let path = path.as_path();
        if !path.is_file() {
            return;
        }
        let root = path.parent().expect("a parent directory");
        // The metadata is here on every machine; the shards are not. Without at least one shard
        // there are no headers to read, so the shape check runs where the weights live (a GPU host)
        // and is skipped where they do not.
        let shards = std::fs::read_dir(root)
            .expect("the checkpoint's directory")
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".safetensors")
            });
        if !shards {
            return;
        }
        let config = crate::Config::read(root).expect("the local Qwen3-30B-A3B metadata");
        let checkpoint = nn::Checkpoint::open(root).expect("the local checkpoint's headers");

        let hidden = config.hidden_size as u64;
        let query = (config.num_attention_heads * config.head_dim) as u64;
        let key_value = (config.num_key_value_heads * config.head_dim) as u64;
        let head_dim = config.head_dim as u64;
        let vocabulary = config.vocab_size as u64;
        let intermediate = config.intermediate_size as u64;

        let expected = |parameter: &str| -> Shape {
            if parameter.ends_with("attention.q_proj.weight") {
                Shape::new(&[query, hidden])
            } else if parameter.ends_with("attention.k_proj.weight")
                || parameter.ends_with("attention.v_proj.weight")
            {
                Shape::new(&[key_value, hidden])
            } else if parameter.ends_with("attention.o_proj.weight") {
                Shape::new(&[hidden, query])
            } else if parameter.ends_with("attention.q_norm.weight")
                || parameter.ends_with("attention.k_norm.weight")
            {
                Shape::new(&[head_dim])
            } else if parameter.ends_with("mlp.router.weight") {
                Shape::new(&[config.num_experts as u64, hidden])
            } else if parameter.ends_with("mlp.gate_proj.weight")
                || parameter.ends_with("mlp.up_proj.weight")
            {
                Shape::new(&[intermediate, hidden])
            } else if parameter.ends_with("mlp.down_proj.weight") {
                Shape::new(&[hidden, intermediate])
            } else if parameter == "embedding.weight" || parameter == "lm_head.weight" {
                Shape::new(&[vocabulary, hidden])
            } else {
                // The norms: one value per hidden element.
                Shape::new(&[hidden])
            }
        };

        for source in parameter_sources(&config) {
            let tensor = checkpoint.tensor(&source.checkpoint).unwrap_or_else(|| {
                panic!(
                    "the checkpoint has no {} ({})",
                    source.checkpoint, source.parameter
                )
            });
            assert_eq!(
                tensor.shape,
                expected(&source.parameter),
                "{} has the wrong shape",
                source.checkpoint
            );
        }

        // The expert stacks are per expert and per projection, both read as `[out, in]`.
        let experts = config.moe_intermediate_size as u64;
        for source in expert_sources(&config) {
            for expert in 0..config.num_experts {
                let tensors = source.tensors(expert);
                for (name, shape) in [
                    (tensors.up, Shape::new(&[experts, hidden])),
                    (tensors.gate, Shape::new(&[experts, hidden])),
                    (tensors.down, Shape::new(&[hidden, experts])),
                ] {
                    let tensor = checkpoint
                        .tensor(&name)
                        .unwrap_or_else(|| panic!("the checkpoint has no {name}"));
                    assert_eq!(
                        tensor.shape,
                        shape,
                        "{name} has the wrong shape",
                        name = name
                    );
                }
            }
        }
    }

    #[test]
    fn every_layer_of_the_local_checkpoint_maps() {
        let config = config(&[0, 1, 2, 3], false, false);
        let sources = parameter_sources(&config);
        let names: Vec<&str> = sources
            .iter()
            .map(|source| source.parameter.as_str())
            .collect();
        assert!(names.contains(&"layers.0.attention.q_proj.weight"));
        assert!(names.contains(&"layers.3.mlp.router.weight"));
        assert!(!names.contains(&"layers.3.mlp.gate_proj.weight"));
        assert_eq!(sources.len(), 3 + 4 * (2 + 4 + 2 + 1));
        assert_eq!(expert_sources(&config).len(), 4);
    }

    #[test]
    fn a_dense_layer_maps_its_mlp_and_no_router() {
        let config = config(&[0], false, false);
        let sources = parameter_sources(&config);
        assert!(
            sources
                .iter()
                .any(|source| source.parameter == "layers.1.mlp.gate_proj.weight")
        );
        assert!(
            !sources
                .iter()
                .any(|source| source.parameter == "layers.1.mlp.router.weight")
        );
        // Layers 1..3 are dense; only layer 0 routes, so one expert source.
        assert_eq!(expert_sources(&config).len(), 1);
    }

    #[test]
    fn attention_bias_adds_the_projection_biases() {
        let config = config(&[0], true, false);
        let sources = parameter_sources(&config);
        assert!(
            sources
                .iter()
                .any(|source| source.parameter == "layers.0.attention.q_proj.bias")
        );
        assert!(
            sources
                .iter()
                .any(|source| source.parameter == "layers.0.attention.o_proj.bias")
        );
    }

    #[test]
    fn a_tied_head_reads_the_embedding_tensor() {
        let config = config(&[0], false, true);
        let sources = parameter_sources(&config);
        let head = sources
            .iter()
            .find(|source| source.parameter == "lm_head.weight")
            .expect("the head");
        assert_eq!(head.checkpoint, "model.embed_tokens.weight");
    }

    #[test]
    fn expert_tensors_are_named_per_expert() {
        let config = config(&[2], false, false);
        let source = &expert_sources(&config)[0];
        assert_eq!(source.module, "layers.2.mlp.experts");
        let tensors = source.tensors(7);
        assert_eq!(tensors.up, "model.layers.2.mlp.experts.7.up_proj.weight");
        assert_eq!(
            tensors.gate,
            "model.layers.2.mlp.experts.7.gate_proj.weight"
        );
        assert_eq!(
            tensors.down,
            "model.layers.2.mlp.experts.7.down_proj.weight"
        );
    }
}
