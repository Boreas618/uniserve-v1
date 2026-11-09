//! The checkpoint configuration of `Qwen3MoeForCausalLM`, normalized once at the loading boundary.
//!
//! A checkpoint's `config.json` is read here and nowhere else, through [`nn::config::Fields`]:
//! absent keys take the defaults the Qwen3-MoE family defines, aliases are resolved, and every
//! field is validated before a [`Config`] exists. The model, the layers and the weight layouts then
//! consume typed fields and never parse a dictionary again (`AGENTS.md`, Model Library Boundary).
//!
//! The rules are those of the Hugging Face `Qwen3MoeConfig`, restricted to the mixture-of-experts
//! architecture this crate serves, together with its rotary recipe (plain or YaRN).

use nn::config::Fields;
use ops::activation::Activation;

use crate::{Error, Result};

/// A YaRN rotary recipe, the only scaled recipe a Qwen3 checkpoint may use.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct YaRn {
    /// The context extension factor the checkpoint declares.
    ///
    /// Recorded, not applied: the table is built from `max_position_embeddings /
    /// original_max_position_embeddings`, which is transformers' rule; vLLM uses this factor. The
    /// two agree for a checkpoint that states consistent numbers, and the choice is a recorded
    /// divergence from one reference (`specs/model-library.md`, D10).
    pub factor: f32,
    /// The context length the recipe was fitted to.
    pub original_max_position_embeddings: u32,
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
    /// Whether the recipe truncates the correction at the boundaries; read and, like `factor`,
    /// recorded rather than applied (D10).
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
    pub vocab_size: u32,
    /// Hidden width of the residual stream.
    pub hidden_size: u32,
    /// Dense MLP width; only `mlp_only_layers` use it.
    pub intermediate_size: u32,
    /// Depth of the decoder stack.
    pub num_hidden_layers: u32,
    /// Query heads of one attention layer.
    pub num_attention_heads: u32,
    /// Key/value heads of one attention layer.
    pub num_key_value_heads: u32,
    /// Width of one attention head; may differ from `hidden_size / num_attention_heads`.
    pub head_dim: u32,
    /// The expert gate function: SiLU or the tanh-approximated GELU, the two the expert kernels
    /// implement.
    pub hidden_act: Activation,
    /// RMSNorm epsilon of both pre-norms, the Q/K norms and the final norm.
    pub rms_norm_eps: f32,
    /// Rotary base frequency.
    pub rope_theta: f32,
    /// The scaled rotary recipe, `None` for plain rotary positions.
    pub rope_scaling: RopeScaling,
    /// The context length the checkpoint was trained for.
    pub max_position_embeddings: u32,
    /// Whether the Q/K/V and output projections carry a bias.
    pub attention_bias: bool,
    /// Whether the vocabulary head shares the embedding matrix.
    pub tie_word_embeddings: bool,
    /// Experts per sparse layer.
    pub num_experts: u32,
    /// Experts each token is routed to.
    pub num_experts_per_tok: u32,
    /// Hidden width of one expert's intermediate projection.
    pub moe_intermediate_size: u32,
    /// Whether the selected experts' weights renormalize to one.
    pub norm_topk_prob: bool,
    /// Every `decoder_sparse_step`-th layer is sparse when it is not in `mlp_only_layers`.
    pub decoder_sparse_step: u32,
    /// Layers that keep a dense MLP although the checkpoint is a mixture of experts.
    pub mlp_only_layers: Vec<u32>,
}

impl Config {
    /// Read and validate the checkpoint's `config.json`.
    ///
    /// This is the model package's loading boundary: the file format, the family's defaults and
    /// every alias are handled here. A checkpoint of another architecture is [`Error::Config`].
    pub fn open(checkpoint: &nn::Checkpoint) -> Result<Self> {
        let value = checkpoint.sidecar("config.json")?.ok_or_else(|| {
            Error::Config(format!(
                "{} holds no config.json",
                checkpoint.root().display()
            ))
        })?;
        Self::parse(&Fields::of("config.json", &value)?)
    }

    /// Validate an already-parsed `config.json`.
    pub fn parse(fields: &Fields<'_>) -> Result<Self> {
        let architectures = fields
            .optional_texts("architectures")?
            .ok_or_else(|| Error::Config("checkpoint declares no architectures".to_owned()))?;
        if architectures != ["Qwen3MoeForCausalLM"] {
            return Err(Error::Config(format!(
                "expected exactly one architecture Qwen3MoeForCausalLM, found {architectures:?}"
            )));
        }

        let hidden_size = fields.count("hidden_size")?;
        let num_attention_heads = fields.count("num_attention_heads")?;
        let num_hidden_layers = fields.count("num_hidden_layers")?;

        // The mixture-of-experts family derives the head width from the attention width when the
        // checkpoint omits it (Transformers' Qwen3MoeConfig).
        let head_dim = match fields.optional_count("head_dim")? {
            Some(head_dim) => head_dim,
            None if hidden_size.is_multiple_of(num_attention_heads) => {
                hidden_size / num_attention_heads
            }
            None => {
                return Err(Error::Config(
                    "hidden_size must be divisible by num_attention_heads when head_dim is absent"
                        .to_owned(),
                ));
            }
        };

        // Sliding-window attention changes the visible keys and is not served.
        if fields.optional_flag("use_sliding_window")?.unwrap_or(false) {
            return Err(Error::Config(
                "sliding-window attention is not supported".to_owned(),
            ));
        }
        if let Some(kinds) = fields.optional_texts("layer_types")?
            && kinds.iter().any(|kind| *kind != "full_attention")
        {
            return Err(Error::Config(
                "layer_types other than full_attention are not supported".to_owned(),
            ));
        }

        let num_key_value_heads = fields.count("num_key_value_heads")?;
        if !num_attention_heads.is_multiple_of(num_key_value_heads) {
            return Err(Error::Config(
                "num_attention_heads must be divisible by num_key_value_heads".to_owned(),
            ));
        }
        if !head_dim.is_multiple_of(2) {
            return Err(Error::Config(
                "head_dim must be even for rotary positions".to_owned(),
            ));
        }

        let max_position_embeddings = fields
            .optional_count("max_position_embeddings")?
            .unwrap_or(32768);
        let rotary = Rotary::read(fields, max_position_embeddings)?;

        // Transformers serializes the expert count as `num_local_experts` in newer releases; a
        // checkpoint carrying both must agree.
        let experts = fields
            .alias(&["num_experts", "num_local_experts"])?
            .ok_or_else(|| {
                Error::Config("Qwen3MoeForCausalLM checkpoint declares no expert count".to_owned())
            })?;
        let experts = fields.index(experts)?;
        if experts == 0 {
            return Err(Error::Config(
                "Qwen3MoeForCausalLM checkpoint declares no experts".to_owned(),
            ));
        }
        let num_experts_per_tok = fields.optional_count("num_experts_per_tok")?.unwrap_or(8);
        if num_experts_per_tok > experts {
            return Err(Error::Config(
                "num_experts_per_tok must not exceed num_experts".to_owned(),
            ));
        }
        let moe_intermediate_size = fields
            .optional_count("moe_intermediate_size")?
            .unwrap_or(768);

        let hidden_act = fields
            .optional_activation("hidden_act")?
            .unwrap_or(Activation::Silu);
        if hidden_act == Activation::Gelu {
            return Err(Error::Config(
                "expert gate \"gelu\" is not implemented: the expert kernels gate with silu or \
                 gelu_pytorch_tanh"
                    .to_owned(),
            ));
        }

        let decoder_sparse_step = fields.optional_count("decoder_sparse_step")?.unwrap_or(1);
        let mlp_only_layers = fields
            .optional_indices("mlp_only_layers")?
            .unwrap_or_default();
        if mlp_only_layers
            .iter()
            .any(|&layer| layer >= num_hidden_layers)
        {
            return Err(Error::Config(
                "mlp_only_layers must name layers below num_hidden_layers".to_owned(),
            ));
        }

        Ok(Self {
            vocab_size: fields.count("vocab_size")?,
            hidden_size,
            intermediate_size: fields.count("intermediate_size")?,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_act,
            rms_norm_eps: fields
                .optional_positive("rms_norm_eps")?
                .map_or(1e-6, |value| value as f32),
            rope_theta: rotary.theta,
            rope_scaling: rotary.scaling,
            max_position_embeddings,
            attention_bias: fields.optional_flag("attention_bias")?.unwrap_or(false),
            tie_word_embeddings: fields
                .optional_flag("tie_word_embeddings")?
                .unwrap_or(false),
            num_experts: experts,
            num_experts_per_tok,
            moe_intermediate_size,
            norm_topk_prob: fields.optional_flag("norm_topk_prob")?.unwrap_or(false),
            decoder_sparse_step,
            mlp_only_layers,
        })
    }

    /// Whether any layer keeps a dense feed-forward block instead of routing to experts.
    ///
    /// The forward reserves a dense buffer pair exactly when this is true, so a checkpoint whose
    /// layers all route reserves no dense feed-forward memory.
    pub fn has_dense_layers(&self) -> bool {
        (0..self.num_hidden_layers).any(|layer| !self.sparse(layer))
    }

    /// Whether any layer routes tokens to experts, which is what gives a calibrated checkpoint an
    /// encoded expert step.
    pub fn has_sparse_layers(&self) -> bool {
        (0..self.num_hidden_layers).any(|layer| self.sparse(layer))
    }

    /// Whether `layer` routes tokens to experts.
    pub fn sparse(&self, layer: u32) -> bool {
        self.num_experts > 0
            && !self.mlp_only_layers.contains(&layer)
            && (layer + 1).is_multiple_of(self.decoder_sparse_step)
    }
}

/// The rotary recipe resolved from the checkpoint's metadata.
struct Rotary {
    theta: f32,
    scaling: RopeScaling,
}

impl Rotary {
    /// Resolve `rope_theta`, `rope_scaling`/`rope_parameters` and the partial factor, rejecting the
    /// recipes this model does not serve.
    ///
    /// `rope_parameters` is the newer transformers spelling of the recipe and takes precedence over
    /// `rope_scaling`; a value either states is read through both in that order.
    fn read(fields: &Fields<'_>, max_position_embeddings: u32) -> Result<Self> {
        let scaling = fields.optional_object("rope_scaling")?;
        let parameters = fields.optional_object("rope_parameters")?;
        let maps = [parameters.as_ref(), scaling.as_ref()];

        let stated_type = match &scaling {
            Some(scaling) => scaling.optional_text("type")?,
            None => None,
        };
        let kind = match &parameters {
            Some(parameters) => parameters.optional_text("rope_type")?,
            None => None,
        }
        .or(stated_type)
        .unwrap_or("default");
        if let Some(other) = stated_type
            && other != kind
        {
            return Err(Error::Config(
                "conflicting rotary type aliases rope_scaling/rope_parameters".to_owned(),
            ));
        }

        let theta = match fields.optional_number("rope_theta")? {
            Some(theta) => Some(theta),
            None => first(&maps, |map| map.optional_number("rope_theta"))?,
        }
        .unwrap_or(10_000.0);
        if !(theta.is_finite() && theta > 0.0) {
            return Err(Error::Config(
                "rope_theta must be finite and positive".to_owned(),
            ));
        }
        let partial = match fields.optional_number("partial_rotary_factor")? {
            Some(partial) => Some(partial),
            None => first(&maps, |map| map.optional_number("partial_rotary_factor"))?,
        }
        .unwrap_or(1.0);
        if partial != 1.0 {
            return Err(Error::Config(
                "partial rotary positions are not supported".to_owned(),
            ));
        }

        let scaling = match kind {
            "default" => None,
            "yarn" => Some(yarn(&maps, max_position_embeddings)?),
            other => {
                return Err(Error::Config(format!(
                    "unsupported rotary recipe {other:?}; Qwen3-MoE serves default and yarn"
                )));
            }
        };
        Ok(Self {
            theta: theta as f32,
            scaling,
        })
    }
}

/// The first of `maps` that states a value through `read`.
fn first<T>(
    maps: &[Option<&Fields<'_>>],
    read: impl Fn(&Fields<'_>) -> nn::Result<Option<T>>,
) -> Result<Option<T>> {
    for map in maps.iter().flatten() {
        if let Some(value) = read(map)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

/// Resolve the YaRN recipe's fields from `maps`, the recipe's objects in precedence order.
fn yarn(maps: &[Option<&Fields<'_>>], default_original: u32) -> Result<YaRn> {
    let number = |key: &str| first(maps, |map| map.optional_number(key));
    let factor = number("factor")?
        .ok_or_else(|| Error::Config("a yarn rotary recipe requires factor".to_owned()))?
        as f32;
    let original = first(maps, |map| {
        map.optional_index("original_max_position_embeddings")
    })?
    .unwrap_or(default_original);
    let attention_factor = number("attention_factor")?.map(|value| value as f32);
    let beta_fast = number("beta_fast")?.map_or(32.0, |value| value as f32);
    let beta_slow = number("beta_slow")?.map_or(1.0, |value| value as f32);
    let mscale = number("mscale")?.map(|value| value as f32);
    let mscale_all_dim = number("mscale_all_dim")?.map(|value| value as f32);
    let truncate = first(maps, |map| map.optional_flag("truncate"))?.unwrap_or(true);

    for (name, value) in [
        ("factor", factor),
        ("beta_fast", beta_fast),
        ("beta_slow", beta_slow),
    ] {
        if !(value.is_finite() && value > 0.0) {
            return Err(Error::Config(format!(
                "yarn {name} must be finite and positive"
            )));
        }
    }
    if original == 0 {
        return Err(Error::Config(
            "yarn original_max_position_embeddings must be positive".to_owned(),
        ));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The metadata of a Qwen3-30B-A3B checkpoint parses to the published architecture.
    #[test]
    #[ignore = "requires UNISERVE_QWEN3_MODEL naming a Qwen3-MoE checkpoint"]
    fn the_local_checkpoint_metadata_parses() {
        let root = std::env::var_os("UNISERVE_QWEN3_MODEL")
            .expect("UNISERVE_QWEN3_MODEL names a Qwen3-MoE checkpoint");
        let checkpoint = nn::Checkpoint::open(std::path::Path::new(&root)).expect("the shards");
        let config = Config::open(&checkpoint).expect("the local Qwen3-30B-A3B metadata");
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

    fn parse(value: &serde_json::Value) -> Result<Config> {
        Config::parse(&Fields::of("config.json", value)?)
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
        let config = parse(&complete()).expect("a supported checkpoint");
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_experts, 128);
        assert!(config.sparse(0) && config.sparse(3));
        assert_eq!(config.hidden_act, Activation::Silu);
        assert!(config.rope_scaling.is_none());
    }

    #[test]
    fn head_dim_defaults_to_the_attention_width() {
        let mut value = complete();
        value.as_object_mut().expect("object").remove("head_dim");
        let config = parse(&value).expect("a supported checkpoint");
        assert_eq!(config.head_dim, 64);
    }

    #[test]
    fn a_non_moe_architecture_is_refused() {
        let mut value = complete();
        value["architectures"] = serde_json::json!(["Qwen3ForCausalLM"]);
        assert!(matches!(
            parse(&value),
            Err(Error::Config(message)) if message.contains("Qwen3MoeForCausalLM")
        ));
    }

    #[test]
    fn sliding_window_is_refused() {
        let mut value = complete();
        value["use_sliding_window"] = serde_json::json!(true);
        assert!(parse(&value).is_err());
    }

    #[test]
    fn a_yarn_recipe_is_read_typed() {
        let mut value = complete();
        value["rope_scaling"] = serde_json::json!({"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096});
        let config = parse(&value).expect("a supported checkpoint");
        let yarn = config.rope_scaling.expect("a yarn recipe");
        assert_eq!(yarn.factor, 4.0);
        assert_eq!(yarn.original_max_position_embeddings, 4096);
        assert!(yarn.truncate);
    }

    #[test]
    fn a_partial_rotary_factor_is_refused() {
        let mut value = complete();
        value["partial_rotary_factor"] = serde_json::json!(0.5);
        assert!(parse(&value).is_err());
    }

    #[test]
    fn conflicting_expert_aliases_are_refused() {
        let mut value = complete();
        value["num_local_experts"] = serde_json::json!(64);
        assert!(parse(&value).is_err());
        // The same count under both spellings is one statement.
        value["num_local_experts"] = serde_json::json!(128);
        assert_eq!(parse(&value).expect("agreeing aliases").num_experts, 128);
    }

    #[test]
    fn a_field_of_the_wrong_type_is_refused_by_name() {
        let mut value = complete();
        value["attention_bias"] = serde_json::json!("no");
        let error = parse(&value)
            .expect_err("a string is not a flag")
            .to_string();
        assert!(error.contains("config.json.attention_bias"), "{error}");
    }
}
