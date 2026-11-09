//! The Qwen3 family's checkpoint configurations, normalized once at the loading boundary.
//!
//! A checkpoint's `config.json` is read here and nowhere else, through [`nn::config::Fields`]:
//! absent keys take the defaults the Hugging Face Qwen3 configurations define, aliases are
//! resolved, and every field is validated before a configuration exists. The model, the layers and
//! the weight layouts then consume typed fields and never parse a dictionary again (`AGENTS.md`,
//! Model Library Boundary).
//!
//! [`TextConfig`] is the decoder stack every Qwen3 model shares; [`MoeConfig`] adds the routed
//! experts of `Qwen3MoeForCausalLM`.

use nn::config::Fields;
use nn::family::Split;
use nn::layers::{Precision, Spectrum};
use ops::activation::Activation;

use crate::{Error, Result};

/// Immutable numerical parameters of a Qwen3 decoder stack: Hugging Face's `Qwen3Config` fields
/// that the decoder layers read.
#[derive(Clone, PartialEq, Debug)]
pub struct TextConfig {
    /// Vocabulary size of the checkpoint, before any tensor-parallel sharding.
    pub vocab_size: u32,
    /// Hidden width of the residual stream.
    pub hidden_size: u32,
    /// Width of a dense layer's gated MLP.
    pub intermediate_size: u32,
    /// Depth of the decoder stack the checkpoint holds.
    pub num_hidden_layers: u32,
    /// Query heads of one attention layer.
    pub num_attention_heads: u32,
    /// Key/value heads of one attention layer.
    pub num_key_value_heads: u32,
    /// Width of one attention head; may differ from `hidden_size / num_attention_heads`.
    pub head_dim: u32,
    /// The gate of every gated MLP of the stack, dense or routed.
    pub hidden_act: Activation,
    /// RMSNorm epsilon of both pre-norms, the Q/K norms and the final norm.
    pub rms_norm_eps: f32,
    /// The rotary spectrum over every channel of a head.
    pub rope: Spectrum,
    /// The context length the checkpoint was trained for.
    pub max_position_embeddings: u32,
    /// Whether the Q/K/V and output projections carry a bias.
    pub attention_bias: bool,
    /// Whether the vocabulary head shares the embedding matrix.
    pub tie_word_embeddings: bool,
}

impl TextConfig {
    /// Read a Qwen3 text configuration: a `config.json` root, or a multimodal checkpoint's
    /// `text_config` object.
    ///
    /// `precision` is the arithmetic the consumer's baseline computes a default or sectioned
    /// rotary spectrum in ([`Precision`]); a YaRN spectrum is always f64. A `rope_scaling` that
    /// states `mrope_section` is Qwen3-VL's interleaved M-RoPE ([`Spectrum::Sectioned`]).
    ///
    /// # Errors
    ///
    /// A field of the wrong type or outside its domain, sliding-window attention, a head count
    /// the key/value heads do not divide, or a rotary recipe that is not served.
    pub fn parse(fields: &Fields<'_>, precision: Precision) -> Result<Self> {
        let hidden_size = fields.count("hidden_size")?;
        let num_attention_heads = fields.count("num_attention_heads")?;

        // Transformers derives the head width from the attention width when the checkpoint omits
        // it.
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
        let rope = spectrum(fields, head_dim, max_position_embeddings, precision)?;

        Ok(Self {
            vocab_size: fields.count("vocab_size")?,
            hidden_size,
            intermediate_size: fields.count("intermediate_size")?,
            num_hidden_layers: fields.count("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_act: fields
                .optional_activation("hidden_act")?
                .unwrap_or(Activation::Silu),
            rms_norm_eps: fields
                .optional_positive("rms_norm_eps")?
                .map_or(1e-6, |value| value as f32),
            rope,
            max_position_embeddings,
            attention_bias: fields.optional_flag("attention_bias")?.unwrap_or(false),
            tie_word_embeddings: fields
                .optional_flag("tie_word_embeddings")?
                .unwrap_or(false),
        })
    }

    /// The query and key/value heads one rank of `split` holds.
    ///
    /// # Errors
    ///
    /// A split that does not divide both head counts, or a rank outside it.
    pub fn local_heads(&self, split: Split) -> Result<(u32, u32)> {
        if split.size == 0 || split.rank >= split.size {
            return Err(Error::Config(format!(
                "tensor-parallel rank {} of a group of {}",
                split.rank, split.size
            )));
        }
        for (name, heads) in [
            ("attention", self.num_attention_heads),
            ("key/value", self.num_key_value_heads),
        ] {
            if !heads.is_multiple_of(split.size) {
                return Err(Error::Config(format!(
                    "tensor parallel degree {} does not divide {heads} {name} heads",
                    split.size
                )));
            }
        }
        Ok((
            self.num_attention_heads / split.size,
            self.num_key_value_heads / split.size,
        ))
    }
}

/// The rotary spectrum of a text configuration over `head_dim` channels, in `precision`.
///
/// A `rope_scaling` with `mrope_section` is Qwen3-VL's: interleaved M-RoPE over the default
/// spectrum, its section widths counting frequencies. Every other recipe is
/// [`Spectrum::from_hf`]'s, a default spectrum taking the consumer's precision.
fn spectrum(
    fields: &Fields<'_>,
    head_dim: u32,
    max_position_embeddings: u32,
    precision: Precision,
) -> Result<Spectrum> {
    if let Some(scaling) = fields.optional_object("rope_scaling")?
        && scaling.contains("mrope_section")
    {
        if scaling.text("rope_type")? != "default" || !scaling.flag("mrope_interleaved")? {
            return Err(Error::Config(
                "an M-RoPE rope_scaling must be the default recipe with interleaved sections"
                    .to_owned(),
            ));
        }
        let sections = scaling.indices("mrope_section")?;
        if sections.iter().sum::<u32>() != head_dim / 2 {
            return Err(Error::Config(format!(
                "M-RoPE sections {sections:?} must split half of the {head_dim} channel head"
            )));
        }
        return Ok(Spectrum::Sectioned {
            theta: fields.positive("rope_theta")? as f32,
            dim: head_dim,
            sections: nn::layers::Sections::new(&sections)?,
            precision,
        });
    }
    Ok(
        match Spectrum::from_hf(fields, head_dim, max_position_embeddings)? {
            Spectrum::Default { theta, dim, .. } => Spectrum::Default {
                theta,
                dim,
                precision,
            },
            other => other,
        },
    )
}

/// Immutable numerical parameters of one `Qwen3MoeForCausalLM` checkpoint: its decoder stack and
/// its routed experts.
///
/// A zero [`MoeConfig::num_experts`] is not representable: the layers of this class route tokens
/// to experts. [`MoeConfig::sparse`] reports whether a layer routes or keeps the dense MLP of
/// [`TextConfig::intermediate_size`].
#[derive(Clone, PartialEq, Debug)]
pub struct MoeConfig {
    /// The decoder stack.
    pub text: TextConfig,
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

impl MoeConfig {
    /// The precision the Qwen3-MoE baseline computes its rotary spectrum in: the f64 rule.
    const PRECISION: Precision = Precision::F64;

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
        let text = TextConfig::parse(fields, Self::PRECISION)?;
        if text.hidden_act == Activation::Gelu {
            return Err(Error::Config(
                "expert gate \"gelu\" is not implemented: the expert kernels gate with silu or \
                 gelu_pytorch_tanh"
                    .to_owned(),
            ));
        }

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

        let decoder_sparse_step = fields.optional_count("decoder_sparse_step")?.unwrap_or(1);
        let mlp_only_layers = fields
            .optional_indices("mlp_only_layers")?
            .unwrap_or_default();
        if mlp_only_layers
            .iter()
            .any(|&layer| layer >= text.num_hidden_layers)
        {
            return Err(Error::Config(
                "mlp_only_layers must name layers below num_hidden_layers".to_owned(),
            ));
        }

        Ok(Self {
            text,
            num_experts: experts,
            num_experts_per_tok,
            moe_intermediate_size,
            norm_topk_prob: fields.optional_flag("norm_topk_prob")?.unwrap_or(false),
            decoder_sparse_step,
            mlp_only_layers,
        })
    }

    /// Check a deployment's tensor and expert splits against the model: the tensor split must
    /// divide the heads ([`TextConfig::local_heads`]) and the expert split the experts.
    ///
    /// # Errors
    ///
    /// A split that does not divide what it shards, or a rank outside its split.
    pub fn check_splits(&self, tensor: Split, experts: Split) -> Result<()> {
        self.text.local_heads(tensor)?;
        if experts.size == 0 || experts.rank >= experts.size {
            return Err(Error::Config(format!(
                "expert rank {} of a group of {}",
                experts.rank, experts.size
            )));
        }
        if !self.num_experts.is_multiple_of(experts.size) {
            return Err(Error::Config(format!(
                "expert parallel degree {} does not divide {} experts",
                experts.size, self.num_experts
            )));
        }
        Ok(())
    }

    /// Whether any layer routes tokens to experts, which is what gives a calibrated checkpoint an
    /// encoded expert step.
    pub fn has_sparse_layers(&self) -> bool {
        (0..self.text.num_hidden_layers).any(|layer| self.sparse(layer))
    }

    /// Whether `layer` routes tokens to experts.
    pub fn sparse(&self, layer: u32) -> bool {
        self.num_experts > 0
            && !self.mlp_only_layers.contains(&layer)
            && (layer + 1).is_multiple_of(self.decoder_sparse_step)
    }
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
        let config = MoeConfig::open(&checkpoint).expect("the local Qwen3-30B-A3B metadata");
        let text = &config.text;
        assert_eq!(text.vocab_size, 151936);
        assert_eq!(text.hidden_size, 2048);
        assert_eq!(text.intermediate_size, 6144);
        assert_eq!(text.num_hidden_layers, 48);
        assert_eq!(text.num_attention_heads, 32);
        assert_eq!(text.num_key_value_heads, 4);
        assert_eq!(text.head_dim, 128);
        assert_eq!(config.num_experts, 128);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.moe_intermediate_size, 768);
        assert!(config.norm_topk_prob);
        assert!(!text.attention_bias);
        assert!(!text.tie_word_embeddings);
        // Every layer of this checkpoint routes to experts.
        assert!((0..text.num_hidden_layers).all(|layer| config.sparse(layer)));
    }

    fn parse(value: &serde_json::Value) -> Result<MoeConfig> {
        MoeConfig::parse(&Fields::of("config.json", value)?)
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
        assert_eq!(config.text.hidden_size, 2048);
        assert_eq!(config.text.head_dim, 128);
        assert_eq!(config.num_experts, 128);
        assert!(config.sparse(0) && config.sparse(3));
        assert_eq!(config.text.hidden_act, Activation::Silu);
        assert!(matches!(
            config.text.rope,
            Spectrum::Default { theta, dim: 128, precision: Precision::F64 } if theta == 1e6
        ));
    }

    #[test]
    fn head_dim_defaults_to_the_attention_width() {
        let mut value = complete();
        value.as_object_mut().expect("object").remove("head_dim");
        let config = parse(&value).expect("a supported checkpoint");
        assert_eq!(config.text.head_dim, 64);
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
        let Spectrum::Yarn { yarn, .. } = config.text.rope else {
            panic!("a yarn recipe, not {:?}", config.text.rope);
        };
        assert_eq!(yarn.factor, 4.0);
        assert_eq!(yarn.original_max_position_embeddings, 4096);
        assert_eq!(yarn.max_position_embeddings, 40960);
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

    #[test]
    fn splits_must_divide_the_heads_and_the_experts() {
        let config = parse(&complete()).expect("a supported checkpoint");
        let split = |size, rank| Split { size, rank };
        assert_eq!(config.text.local_heads(split(4, 2)).expect("four"), (8, 1));
        assert!(config.check_splits(split(4, 3), split(4, 2)).is_ok());
        // Three ranks divide neither head count; a rank outside its group is no rank.
        assert!(config.text.local_heads(split(3, 0)).is_err());
        assert!(config.text.local_heads(split(4, 4)).is_err());
        // 128 experts do not split over three ranks.
        assert!(config.check_splits(split(1, 0), split(3, 0)).is_err());
    }
}
