//! The checkpoint configuration of `Qwen3MoeForCausalLM`, normalized once at the loading boundary.
//!
//! A checkpoint's `config.json` is read here and nowhere else: absent keys take the defaults the
//! Qwen3-MoE family defines, aliases are resolved, and every field is validated before a [`Config`]
//! exists. The model, the layers and the weight layouts then consume typed fields and never parse a
//! dictionary again (`AGENTS.md`, Model Library Boundary).
//!
//! The rules are those of `uniserve/models/qwen3/config.py` (`read_config`, `Config`) restricted to
//! the mixture-of-experts architecture this crate serves, and `uniserve/rotary.py` for the rotary
//! recipe.

use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::{Error, Result};

/// The gating function of the expert block, as the expert kernels implement it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Activation {
    /// SiLU, the checkpoint's `silu` or `swish`.
    Silu,
    /// Tanh-approximated GELU, the checkpoint's `gelu_pytorch_tanh`.
    GeluTanh,
}

impl Activation {
    /// The name the operator descriptors carry.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Silu => "silu",
            Self::GeluTanh => "gelu_pytorch_tanh",
        }
    }
}

/// A YaRN rotary recipe, the only scaled recipe a Qwen3 checkpoint may use.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct YaRn {
    /// The context extension factor.
    pub factor: f32,
    /// The context length the recipe was fitted to.
    pub original_max_position_embeddings: usize,
    /// The attention scaling, when the checkpoint declares one.
    pub attention_factor: Option<f32>,
    /// The high-frequency correction boundary.
    pub beta_fast: f32,
    /// The low-frequency correction boundary.
    pub beta_slow: f32,
    /// The YaRN magnitude scale, when declared.
    pub mscale: Option<f32>,
    /// The all-dimension magnitude scale, when declared.
    pub mscale_all_dim: Option<f32>,
    /// Whether the recipe truncates the correction at the boundaries.
    pub truncate: bool,
}

/// The rotary recipe of the checkpoint: plain positions when `None`.
pub type RopeScaling = Option<YaRn>;

/// Immutable numerical parameters of one Qwen3-MoE checkpoint.
///
/// A zero [`Config::num_experts`] is not representable: this crate serves
/// `Qwen3MoeForCausalLM`, whose layers route tokens to experts. `sparse(layer)` reports whether a
/// layer routes or keeps the dense MLP of [`Config::intermediate_size`].
#[derive(Clone, PartialEq, Debug)]
pub struct Config {
    /// Vocabulary size of the checkpoint, before any tensor-parallel sharding.
    pub vocab_size: usize,
    /// Hidden width of the residual stream.
    pub hidden_size: usize,
    /// Dense MLP width; only `mlp_only_layers` use it.
    pub intermediate_size: usize,
    /// Depth of the decoder stack.
    pub num_hidden_layers: usize,
    /// Query heads of one attention layer.
    pub num_attention_heads: usize,
    /// Key/value heads of one attention layer.
    pub num_key_value_heads: usize,
    /// Width of one attention head; may differ from `hidden_size / num_attention_heads`.
    pub head_dim: usize,
    /// The expert gate function.
    pub hidden_act: Activation,
    /// RMSNorm epsilon of both pre-norms, the Q/K norms and the final norm.
    pub rms_norm_eps: f32,
    /// Rotary base frequency.
    pub rope_theta: f32,
    /// The scaled rotary recipe, `None` for plain rotary positions.
    pub rope_scaling: RopeScaling,
    /// The context length the checkpoint was trained for.
    pub max_position_embeddings: usize,
    /// Whether the Q/K/V and output projections carry a bias.
    pub attention_bias: bool,
    /// Whether the vocabulary head shares the embedding matrix.
    pub tie_word_embeddings: bool,
    /// Experts per sparse layer.
    pub num_experts: usize,
    /// Experts each token is routed to.
    pub num_experts_per_tok: usize,
    /// Hidden width of one expert's intermediate projection.
    pub moe_intermediate_size: usize,
    /// Whether the selected experts' weights renormalize to one.
    pub norm_topk_prob: bool,
    /// Every `decoder_sparse_step`-th layer is sparse when it is not in `mlp_only_layers`.
    pub decoder_sparse_step: usize,
    /// Layers that keep a dense MLP although the checkpoint is a mixture of experts.
    pub mlp_only_layers: Vec<usize>,
}

impl Config {
    /// Read and validate `<dir>/config.json`.
    ///
    /// This is the model package's loading boundary: the file format, the family's defaults and
    /// every alias are handled here. A checkpoint of another architecture is
    /// [`Error::Config`].
    pub fn read(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = fs::read_to_string(&path).map_err(|source| Error::Io {
            path: path.display().to_string(),
            source,
        })?;
        let raw: RawConfig = serde_json::from_str(&text).map_err(|source| Error::Json {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_raw(&raw)
    }

    /// Validate an already-parsed `config.json`.
    fn from_raw(raw: &RawConfig) -> Result<Self> {
        let architectures = raw
            .architectures
            .as_ref()
            .ok_or_else(|| ConfigError::new("checkpoint declares no architectures"))?;
        if architectures.len() != 1 || architectures[0] != "Qwen3MoeForCausalLM" {
            return Err(ConfigError::new(format!(
                "expected exactly one architecture Qwen3MoeForCausalLM, found {architectures:?}"
            ))
            .into());
        }

        let hidden_size = raw.required_positive("hidden_size")?;
        let num_attention_heads = raw.required_positive("num_attention_heads")?;
        let num_hidden_layers = raw.required_positive("num_hidden_layers")?;

        // The mixture-of-experts family derives the head width from the attention width when the
        // checkpoint omits it (Transformers' Qwen3MoeConfig).
        if hidden_size % num_attention_heads != 0 && raw.head_dim.is_none() {
            return Err(ConfigError::new(
                "hidden_size must be divisible by num_attention_heads when head_dim is absent",
            )
            .into());
        }
        let head_dim = raw.optional_positive("head_dim", hidden_size / num_attention_heads)?;

        // Sliding-window attention changes the visible keys and is not served.
        if raw.use_sliding_window.unwrap_or(false) {
            return Err(ConfigError::new("sliding-window attention is not supported").into());
        }
        if let Some(kinds) = &raw.layer_types
            && kinds.iter().any(|kind| kind != "full_attention")
        {
            return Err(ConfigError::new(
                "layer_types other than full_attention are not supported",
            )
            .into());
        }

        let num_key_value_heads = raw.required_positive("num_key_value_heads")?;
        if num_attention_heads % num_key_value_heads != 0 {
            return Err(ConfigError::new(
                "num_attention_heads must be divisible by num_key_value_heads",
            )
            .into());
        }
        if head_dim % 2 != 0 {
            return Err(ConfigError::new("head_dim must be even for rotary positions").into());
        }

        let max_position_embeddings = raw.optional_positive("max_position_embeddings", 32768)?;
        let rotary = raw.rotary(max_position_embeddings)?;

        // Transformers serializes the expert count as `num_local_experts` in newer releases; a
        // checkpoint carrying both must agree.
        let experts = match (raw.num_experts, raw.num_local_experts) {
            (Some(a), Some(b)) if a != b => {
                return Err(ConfigError::new(
                    "conflicting expert-count aliases num_experts/num_local_experts",
                )
                .into());
            }
            (Some(count), _) | (None, Some(count)) => count,
            (None, None) => {
                return Err(ConfigError::new(
                    "Qwen3MoeForCausalLM checkpoint declares no expert count",
                )
                .into());
            }
        };
        if experts == 0 {
            return Err(
                ConfigError::new("Qwen3MoeForCausalLM checkpoint declares no experts").into(),
            );
        }
        let num_experts_per_tok = raw.optional_positive("num_experts_per_tok", 8)?;
        if num_experts_per_tok > experts {
            return Err(ConfigError::new("num_experts_per_tok must not exceed num_experts").into());
        }
        let moe_intermediate_size = raw.optional_positive("moe_intermediate_size", 768)?;

        let hidden_act = match raw.hidden_act.as_deref().unwrap_or("silu") {
            "silu" | "swish" => Activation::Silu,
            "gelu_pytorch_tanh" => Activation::GeluTanh,
            other => {
                return Err(
                    ConfigError::new(format!("expert gate {other:?} is not implemented")).into(),
                );
            }
        };

        let decoder_sparse_step = raw.optional_positive("decoder_sparse_step", 1)?;
        let mlp_only_layers = match &raw.mlp_only_layers {
            None => Vec::new(),
            Some(layers) => {
                if layers.iter().any(|&layer| layer >= num_hidden_layers) {
                    return Err(ConfigError::new(
                        "mlp_only_layers must name layers below num_hidden_layers",
                    )
                    .into());
                }
                layers.clone()
            }
        };

        Ok(Self {
            vocab_size: raw.required_positive("vocab_size")?,
            hidden_size,
            intermediate_size: raw.required_positive("intermediate_size")?,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_act,
            rms_norm_eps: raw.optional_positive_f32("rms_norm_eps", 1e-6)?,
            rope_theta: rotary.theta,
            rope_scaling: rotary.scaling,
            max_position_embeddings,
            attention_bias: raw.attention_bias.unwrap_or(false),
            tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(false),
            num_experts: experts,
            num_experts_per_tok,
            moe_intermediate_size,
            norm_topk_prob: raw.norm_topk_prob.unwrap_or(false),
            decoder_sparse_step,
            mlp_only_layers,
        })
    }

    /// Whether any layer keeps a dense feed-forward block instead of routing to experts.
    ///
    /// The activation plan carries a dense buffer pair exactly when this is true, so a checkpoint
    /// whose layers all route plans no dense feed-forward memory.
    pub fn has_dense_layers(&self) -> bool {
        (0..self.num_hidden_layers).any(|layer| !self.sparse(layer))
    }

    /// Whether `layer` routes tokens to experts.
    pub fn sparse(&self, layer: usize) -> bool {
        self.num_experts > 0
            && !self.mlp_only_layers.contains(&layer)
            && (layer + 1).is_multiple_of(self.decoder_sparse_step)
    }

    /// The width of one KV head's row in the cache.
    pub const fn kv_head_dim(&self) -> usize {
        self.head_dim
    }

    /// Query heads per key/value head.
    pub const fn group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// The rotary recipe resolved from the checkpoint's metadata.
struct Rotary {
    theta: f32,
    scaling: RopeScaling,
}

impl RawConfig {
    /// Resolve `rope_theta`, `rope_scaling`/`rope_parameters` and the partial factor, rejecting the
    /// recipes this model does not serve (`uniserve/rotary.py:read_rotary`).
    fn rotary(&self, max_position_embeddings: usize) -> Result<Rotary> {
        let scaling = self.rope_scaling.as_ref();
        let parameters = self.rope_parameters.as_ref();
        let kind = parameters
            .and_then(|map| map.get("rope_type"))
            .or_else(|| scaling.and_then(|map| map.get("type")))
            .and_then(Value::as_str)
            .unwrap_or("default");

        if let Some(other) = scaling
            .and_then(|map| map.get("type"))
            .and_then(Value::as_str)
            && other != kind
        {
            return Err(ConfigError::new(
                "conflicting rotary type aliases rope_scaling/rope_parameters",
            )
            .into());
        }

        let theta = self
            .rope_theta
            .or_else(|| map_f64(parameters, "rope_theta"))
            .or_else(|| map_f64(scaling, "rope_theta"))
            .unwrap_or(10_000.0);
        if !(theta.is_finite() && theta > 0.0) {
            return Err(ConfigError::new("rope_theta must be finite and positive").into());
        }
        let partial = self
            .partial_rotary_factor
            .or_else(|| map_f64(parameters, "partial_rotary_factor"))
            .or_else(|| map_f64(scaling, "partial_rotary_factor"))
            .unwrap_or(1.0);
        if partial != 1.0 {
            return Err(ConfigError::new("partial rotary positions are not supported").into());
        }

        let scaling = match kind {
            "default" => None,
            "yarn" => Some(yarn(parameters, scaling, max_position_embeddings)?),
            other => {
                return Err(ConfigError::new(format!(
                    "unsupported rotary recipe {other:?}; Qwen3-MoE serves default and yarn"
                ))
                .into());
            }
        };
        Ok(Rotary {
            theta: theta as f32,
            scaling,
        })
    }
}

/// Resolve the YaRN recipe's fields, with `rope_parameters` taking precedence over `rope_scaling`.
fn yarn(
    parameters: Option<&serde_json::Map<String, Value>>,
    scaling: Option<&serde_json::Map<String, Value>>,
    default_original: usize,
) -> Result<YaRn> {
    let factor = map_f32(parameters, "factor")
        .or_else(|| map_f32(scaling, "factor"))
        .ok_or_else(|| ConfigError::new("a yarn rotary recipe requires factor"))?;
    let original = map_usize(parameters, "original_max_position_embeddings")
        .or_else(|| map_usize(scaling, "original_max_position_embeddings"))
        .unwrap_or(default_original);
    let attention_factor =
        map_f32(parameters, "attention_factor").or_else(|| map_f32(scaling, "attention_factor"));
    let beta_fast = map_f32(parameters, "beta_fast")
        .or_else(|| map_f32(scaling, "beta_fast"))
        .unwrap_or(32.0);
    let beta_slow = map_f32(parameters, "beta_slow")
        .or_else(|| map_f32(scaling, "beta_slow"))
        .unwrap_or(1.0);
    let mscale = map_f32(parameters, "mscale").or_else(|| map_f32(scaling, "mscale"));
    let mscale_all_dim =
        map_f32(parameters, "mscale_all_dim").or_else(|| map_f32(scaling, "mscale_all_dim"));
    let truncate = map_bool(parameters, "truncate")
        .or_else(|| map_bool(scaling, "truncate"))
        .unwrap_or(true);

    for (name, value) in [
        ("factor", factor),
        ("beta_fast", beta_fast),
        ("beta_slow", beta_slow),
    ] {
        if !(value.is_finite() && value > 0.0) {
            return Err(
                ConfigError::new(format!("yarn {name} must be finite and positive")).into(),
            );
        }
    }
    if original == 0 {
        return Err(
            ConfigError::new("yarn original_max_position_embeddings must be positive").into(),
        );
    }
    Ok(YaRn {
        factor,
        original_max_position_embeddings: original,
        attention_factor,
        beta_fast,
        beta_slow,
        mscale,
        mscale_all_dim,
        truncate,
    })
}

fn map_f64(map: Option<&serde_json::Map<String, Value>>, key: &str) -> Option<f64> {
    map.and_then(|map| map.get(key)).and_then(Value::as_f64)
}

fn map_f32(map: Option<&serde_json::Map<String, Value>>, key: &str) -> Option<f32> {
    map_f64(map, key).map(|value| value as f32)
}

fn map_usize(map: Option<&serde_json::Map<String, Value>>, key: &str) -> Option<usize> {
    map.and_then(|map| map.get(key))
        .and_then(Value::as_u64)
        .map(|value| value as usize)
}

fn map_bool(map: Option<&serde_json::Map<String, Value>>, key: &str) -> Option<bool> {
    map.and_then(|map| map.get(key)).and_then(Value::as_bool)
}

/// A configuration error with the model's name in the message.
struct ConfigError(String);

impl ConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl From<ConfigError> for Error {
    fn from(error: ConfigError) -> Self {
        Error::Config(error.0)
    }
}

/// `config.json` as written, with every optional key kept optional so the defaults are resolved
/// explicitly in [`Config::from_raw`] rather than by the deserializer.
#[derive(Deserialize)]
struct RawConfig {
    architectures: Option<Vec<String>>,
    vocab_size: Option<u64>,
    hidden_size: Option<u64>,
    intermediate_size: Option<u64>,
    num_hidden_layers: Option<u64>,
    num_attention_heads: Option<u64>,
    num_key_value_heads: Option<u64>,
    head_dim: Option<u64>,
    hidden_act: Option<String>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    rope_scaling: Option<serde_json::Map<String, Value>>,
    rope_parameters: Option<serde_json::Map<String, Value>>,
    partial_rotary_factor: Option<f64>,
    max_position_embeddings: Option<u64>,
    attention_bias: Option<bool>,
    tie_word_embeddings: Option<bool>,
    num_experts: Option<usize>,
    num_local_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    moe_intermediate_size: Option<usize>,
    norm_topk_prob: Option<bool>,
    decoder_sparse_step: Option<usize>,
    mlp_only_layers: Option<Vec<usize>>,
    use_sliding_window: Option<bool>,
    layer_types: Option<Vec<String>>,
}

impl RawConfig {
    fn required_positive(&self, name: &str) -> Result<usize> {
        let value = match name {
            "vocab_size" => self.vocab_size,
            "hidden_size" => self.hidden_size,
            "intermediate_size" => self.intermediate_size,
            "num_hidden_layers" => self.num_hidden_layers,
            "num_attention_heads" => self.num_attention_heads,
            "num_key_value_heads" => self.num_key_value_heads,
            other => {
                return Err(ConfigError::new(format!("unknown required field {other}")).into());
            }
        };
        match value {
            Some(value) if value > 0 => Ok(value as usize),
            _ => Err(ConfigError::new(format!("{name} must be a positive integer")).into()),
        }
    }

    fn optional_positive(&self, name: &str, default: usize) -> Result<usize> {
        let value = match name {
            "head_dim" => self.head_dim,
            "max_position_embeddings" => self.max_position_embeddings,
            "num_experts_per_tok" => self.num_experts_per_tok.map(|value| value as u64),
            "moe_intermediate_size" => self.moe_intermediate_size.map(|value| value as u64),
            "decoder_sparse_step" => self.decoder_sparse_step.map(|value| value as u64),
            other => {
                return Err(ConfigError::new(format!("unknown optional field {other}")).into());
            }
        };
        match value {
            Some(value) if value > 0 => Ok(value as usize),
            Some(_) => Err(ConfigError::new(format!("{name} must be positive")).into()),
            None => Ok(default),
        }
    }

    fn optional_positive_f32(&self, name: &str, default: f32) -> Result<f32> {
        let value = match name {
            "rms_norm_eps" => self.rms_norm_eps,
            other => {
                return Err(ConfigError::new(format!("unknown optional field {other}")).into());
            }
        };
        match value {
            Some(value) if value.is_finite() && value > 0.0 => Ok(value as f32),
            Some(_) => Err(ConfigError::new(format!("{name} must be finite and positive")).into()),
            None => Ok(default),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The metadata of the deployment this crate was written for, when the machine has it.
    ///
    /// The path is the one `specs/worker.md` names for the local checkpoint; a machine without the
    /// asset skips the test rather than failing, and a machine with it checks the parse against the
    /// values the design's slice depends on.
    #[test]
    fn the_local_checkpoint_metadata_parses() {
        let path = std::path::Path::new("/opt/uv/models/Qwen3-30B-A3B/config.json");
        if !path.is_file() {
            return;
        }
        let config = Config::read(path.parent().expect("a parent directory"))
            .expect("the local Qwen3-30B-A3B metadata");
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.intermediate_size, 6144);
        assert_eq!(config.num_hidden_layers, 48);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_experts, 128);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.moe_intermediate_size, 768);
        assert!(config.norm_topk_prob);
        assert!(!config.attention_bias);
        assert!(!config.tie_word_embeddings);
        // Every layer of this checkpoint routes to experts.
        assert!((0..config.num_hidden_layers).all(|layer| config.sparse(layer)));
        assert!(!config.has_dense_layers());
    }

    fn raw(value: serde_json::Value) -> RawConfig {
        serde_json::from_value(value).expect("a config object")
    }

    fn complete() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Qwen3MoeForCausalLM"],
            "vocab_size": 151936,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 40960,
            "attention_bias": false,
            "tie_word_embeddings": false,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "norm_topk_prob": true,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "use_sliding_window": false,
            "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention"]
        })
    }

    #[test]
    fn the_local_checkpoint_shape_normalizes() {
        let config = Config::from_raw(&raw(complete())).expect("a supported checkpoint");
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.group_size(), 8);
        assert_eq!(config.num_experts, 128);
        assert!(config.sparse(0) && config.sparse(3));
        assert_eq!(config.hidden_act, Activation::Silu);
        assert!(config.rope_scaling.is_none());
    }

    #[test]
    fn head_dim_defaults_to_the_attention_width() {
        let mut value = complete();
        value.as_object_mut().expect("object").remove("head_dim");
        let config = Config::from_raw(&raw(value)).expect("a supported checkpoint");
        assert_eq!(config.head_dim, 64);
    }

    #[test]
    fn a_non_moe_architecture_is_refused() {
        let mut value = complete();
        value["architectures"] = serde_json::json!(["Qwen3ForCausalLM"]);
        assert!(matches!(
            Config::from_raw(&raw(value)),
            Err(Error::Config(message)) if message.contains("Qwen3MoeForCausalLM")
        ));
    }

    #[test]
    fn sliding_window_is_refused() {
        let mut value = complete();
        value["use_sliding_window"] = serde_json::json!(true);
        assert!(Config::from_raw(&raw(value)).is_err());
    }

    #[test]
    fn a_yarn_recipe_is_read_typed() {
        let mut value = complete();
        value["rope_scaling"] = serde_json::json!({"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096});
        let config = Config::from_raw(&raw(value)).expect("a supported checkpoint");
        let yarn = config.rope_scaling.expect("a yarn recipe");
        assert_eq!(yarn.factor, 4.0);
        assert_eq!(yarn.original_max_position_embeddings, 4096);
        assert!(yarn.truncate);
    }

    #[test]
    fn a_partial_rotary_factor_is_refused() {
        let mut value = complete();
        value["partial_rotary_factor"] = serde_json::json!(0.5);
        assert!(Config::from_raw(&raw(value)).is_err());
    }

    #[test]
    fn conflicting_expert_aliases_are_refused() {
        let mut value = complete();
        value["num_local_experts"] = serde_json::json!(64);
        assert!(Config::from_raw(&raw(value)).is_err());
    }
}
