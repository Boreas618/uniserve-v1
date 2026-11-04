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
//! Released checkpoints store the stack under `model.layers.<i>…`; some serialize it without the
//! `model.` prefix. The loader resolves a name once with that fallback
//! (`uniserve/models/qwen3/weights.py`, `checkpoint_mappings`), so this table always gives the
//! canonical `model.`-prefixed name.

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
    use crate::{Activation, Config};

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
        assert_eq!(sources.len(), 3 + 4 * (2 + 5 + 2 + 1));
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
        assert_eq!(expert_sources(&config).len(), 3);
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
