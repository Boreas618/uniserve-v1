//! The Qwen3 family's checkpoint configurations, normalized once at the loading boundary.
//!
//! A checkpoint's `config.json` is read here and nowhere else, through [`nn::config::Fields`]:
//! absent keys take the defaults the Hugging Face Qwen3 configurations define, aliases are
//! resolved, and every field is validated before a configuration exists. The model, the layers and
//! the weight layouts then consume typed fields and never parse a dictionary again (`AGENTS.md`,
//! Model Library Boundary).
//!
//! [`TextConfig`] is the decoder stack every Qwen3 model shares; [`MoeConfig`] adds the routed
//! experts of `Qwen3MoeForCausalLM`, and [`VlConfig`] the vision tower and the placeholder tokens
//! of a Qwen3-VL checkpoint.

use nn::config::Fields;
use nn::family::Split;
use nn::layers::{Gain, Precision, Spectrum};
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
    /// Width of a dense layer's gated MLP; zero for a stack whose every layer routes and that states
    /// none (Qwen3.5-MoE).
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
    /// How those norms' stored weights scale: Qwen3 and Qwen3-VL store the scale, Qwen3.5 stores it
    /// zero-centred (`x̂ · (1 + w)`).
    pub rms_norm_gain: Gain,
    /// The rotary spectrum over a head's leading channels: every channel for Qwen3 and Qwen3-VL,
    /// the `partial_rotary_factor` of them for Qwen3.5.
    pub rope: Spectrum,
    /// The context length the checkpoint was trained for.
    pub max_position_embeddings: u32,
    /// Whether the Q/K/V and output projections carry a bias.
    pub attention_bias: bool,
    /// Whether each query head projects a gate of its width beside it, whose sigmoid scales the
    /// attended head (Qwen3.5's `attn_output_gate`).
    pub attn_output_gate: bool,
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
        if let Some(kinds) = fields.optional_texts("layer_types")?
            && kinds.iter().any(|kind| *kind != "full_attention")
        {
            return Err(Error::Config(
                "layer_types other than full_attention are not supported".to_owned(),
            ));
        }
        Self::read(
            fields,
            fields.count("intermediate_size")?,
            |head_dim, max_position_embeddings| {
                spectrum(fields, head_dim, max_position_embeddings, precision)
            },
        )
    }

    /// The fields every Qwen3 decoder stack states, with the dense MLP's width `intermediate_size`
    /// and the spectrum `rope` reads for a head width and a context length.
    fn read(
        fields: &Fields<'_>,
        intermediate_size: u32,
        rope: impl FnOnce(u32, u32) -> Result<Spectrum>,
    ) -> Result<Self> {
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
        let rope = rope(head_dim, max_position_embeddings)?;

        Ok(Self {
            vocab_size: fields.count("vocab_size")?,
            hidden_size,
            intermediate_size,
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
            rms_norm_gain: Gain::Plain,
            rope,
            max_position_embeddings,
            attention_bias: fields.optional_flag("attention_bias")?.unwrap_or(false),
            attn_output_gate: false,
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

        let (experts, num_experts_per_tok) = routed_experts(fields, "Qwen3MoeForCausalLM")?;
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

/// The routed experts a mixture-of-experts text configuration states: the expert count and the
/// experts each token is routed to.
///
/// Transformers serializes the expert count as `num_local_experts` in newer releases; a checkpoint
/// carrying both must agree.
fn routed_experts(fields: &Fields<'_>, class: &str) -> Result<(u32, u32)> {
    let experts = fields
        .alias(&["num_experts", "num_local_experts"])?
        .ok_or_else(|| Error::Config(format!("{class} checkpoint declares no expert count")))?;
    let experts = fields.index(experts)?;
    if experts == 0 {
        return Err(Error::Config(format!(
            "{class} checkpoint declares no experts"
        )));
    }
    let num_experts_per_tok = fields.optional_count("num_experts_per_tok")?.unwrap_or(8);
    if num_experts_per_tok > experts {
        return Err(Error::Config(
            "num_experts_per_tok must not exceed num_experts".to_owned(),
        ));
    }
    Ok((experts, num_experts_per_tok))
}

/// The dimensions of the Qwen3-VL vision encoder and its token mergers.
///
/// Pixels arrive as tubelets of `in_channels x temporal_patch_size x patch_size x patch_size`
/// values. `depth` blocks of width `hidden_size` with `num_heads` heads encode them; every
/// `spatial_merge_size x spatial_merge_size` patch block merges into one `out_hidden_size` token,
/// which is the language model's width.
#[derive(Clone, PartialEq, Debug)]
pub struct VisionConfig {
    /// Transformer blocks of the tower.
    pub depth: u32,
    /// The tower's width.
    pub hidden_size: u32,
    /// The tower's feed-forward width.
    pub intermediate_size: u32,
    /// Attention heads.
    pub num_heads: u32,
    /// The blocks' MLP activation: PyTorch's tanh-approximated GELU, the one the blocks are
    /// served with.
    pub hidden_act: Activation,
    /// Pixel channels of one tubelet voxel.
    pub in_channels: u32,
    /// Side of one square spatial patch, in pixels.
    pub patch_size: u32,
    /// Frames of one temporal patch.
    pub temporal_patch_size: u32,
    /// Side, in patches, of one block that merges into a language token.
    pub spatial_merge_size: u32,
    /// Width of a merged token: the language model's.
    pub out_hidden_size: u32,
    /// Entries of the learned position table, a square grid.
    pub num_position_embeddings: u32,
    /// The index of each block whose output feeds a DeepStack merger, in increasing order.
    pub deepstack_visual_indexes: Vec<u32>,
}

impl VisionConfig {
    /// Read a Qwen3-VL checkpoint's `vision_config`.
    ///
    /// # Errors
    ///
    /// A field of the wrong type, a block activation other than `gelu_pytorch_tanh`, DeepStack
    /// blocks out of order or outside the tower, heads that do not split the width into two even
    /// rotary halves, or a position table that is not a square grid.
    pub fn parse(values: &Fields<'_>) -> Result<Self> {
        let owner = values.owner();
        let hidden_act = values.activation("hidden_act")?;
        if hidden_act != Activation::GeluTanh {
            return Err(Error::Config(format!(
                "{owner}.hidden_act must be gelu_pytorch_tanh, got {hidden_act}"
            )));
        }
        let depth = values.count("depth")?;
        // The DeepStack mergers join the language model after the blocks their indexes name, so an
        // index outside the tower, or two out of order, is refused here.
        // Qwen3.5's tower states an empty list: no block feeds a DeepStack merger.
        let deepstack_visual_indexes = values
            .optional_indices("deepstack_visual_indexes")?
            .ok_or_else(|| {
                Error::Config(format!("{owner}.deepstack_visual_indexes is required"))
            })?;
        if deepstack_visual_indexes
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(Error::Config(format!(
                "{owner}.deepstack_visual_indexes must increase strictly"
            )));
        }
        if let Some(outside) = deepstack_visual_indexes
            .iter()
            .find(|index| **index >= depth - 1)
        {
            return Err(Error::Config(format!(
                "{owner}.deepstack_visual_indexes names block {outside}, outside the {depth}-block \
                 tower"
            )));
        }
        let config = Self {
            depth,
            hidden_size: values.count("hidden_size")?,
            intermediate_size: values.count("intermediate_size")?,
            num_heads: values.count("num_heads")?,
            hidden_act,
            in_channels: values.count("in_channels")?,
            patch_size: values.count("patch_size")?,
            temporal_patch_size: values.count("temporal_patch_size")?,
            spatial_merge_size: values.count("spatial_merge_size")?,
            out_hidden_size: values.count("out_hidden_size")?,
            num_position_embeddings: values.count("num_position_embeddings")?,
            deepstack_visual_indexes,
        };
        if !config.hidden_size.is_multiple_of(config.num_heads) {
            return Err(Error::Config(format!(
                "Qwen3-VL vision heads {} do not divide the hidden size {}",
                config.num_heads, config.hidden_size
            )));
        }
        // The 2D rotary embedding rotates half of each head by the patch row and half by its
        // column, each as split-half pairs.
        if !(config.hidden_size / config.num_heads).is_multiple_of(4) {
            return Err(Error::Config(format!(
                "Qwen3-VL vision head width {} must split into two even rotary halves",
                config.hidden_size / config.num_heads
            )));
        }
        let side = f64::from(config.num_position_embeddings).sqrt().round();
        if (side * side) as u32 != config.num_position_embeddings {
            return Err(Error::Config(
                "Qwen3-VL vision position table must be a square grid".to_owned(),
            ));
        }
        Ok(config)
    }
}

/// A Qwen3-VL checkpoint's configuration: the language model, the vision tower, and the
/// placeholder tokens a prompt presents its media by.
#[derive(Clone, PartialEq, Debug)]
pub struct VlConfig {
    /// The language model: a Qwen3 decoder stack rotating by interleaved M-RoPE.
    pub text: TextConfig,
    /// The vision tower.
    pub vision: VisionConfig,
    /// The token that stands for one merged image patch.
    pub image_token_id: u32,
    /// The token that stands for one merged video patch.
    pub video_token_id: u32,
}

impl VlConfig {
    /// The precision the Qwen3-VL reference computes the language model's spectrum in: FP32.
    const PRECISION: Precision = Precision::F32;

    /// The position axes of the language model's M-RoPE: time, height and width.
    pub const POSITION_AXES: u32 = 3;

    /// Read a Qwen3-VL `config.json`: its `text_config`, `vision_config` and placeholder ids.
    ///
    /// # Errors
    ///
    /// What [`TextConfig::parse`] and [`VisionConfig::parse`] refuse, a language model that does
    /// not rotate by three-axis M-RoPE, a placeholder outside the vocabulary, or merged tokens of
    /// another width than the language model's.
    pub fn parse(values: &Fields<'_>) -> Result<Self> {
        let text = TextConfig::parse(&values.object("text_config")?, Self::PRECISION)?;
        let axes = match text.rope {
            Spectrum::Sectioned { sections, .. } => sections.axes(),
            _ => 0,
        };
        if axes != Self::POSITION_AXES {
            return Err(Error::Config(format!(
                "a Qwen3-VL language model rotates by interleaved M-RoPE over {} axes, not {:?}",
                Self::POSITION_AXES,
                text.rope
            )));
        }
        let config = Self {
            text,
            vision: VisionConfig::parse(&values.object("vision_config")?)?,
            image_token_id: values.index("image_token_id")?,
            video_token_id: values.index("video_token_id")?,
        };
        for token in [config.image_token_id, config.video_token_id] {
            if token >= config.text.vocab_size {
                return Err(Error::Config(format!(
                    "vision placeholder {token} is outside the {} entry vocabulary",
                    config.text.vocab_size
                )));
            }
        }
        // The tower's merged tokens replace placeholder rows of the language model.
        if config.vision.out_hidden_size != config.text.hidden_size {
            return Err(Error::Config(format!(
                "vision tokens are {} wide and the language model {}",
                config.vision.out_hidden_size, config.text.hidden_size
            )));
        }
        Ok(config)
    }
}

/// The token mixer of one decoder layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mixer {
    /// Softmax attention over the key/value cache (`full_attention`).
    Attention,
    /// A gated delta network whose state a request slot keeps (`linear_attention`).
    Linear,
}

/// The gated delta networks of a hybrid stack: Hugging Face's `linear_*` fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearAttentionConfig {
    /// Query and key heads.
    pub num_key_heads: u32,
    /// Value heads, each with its own state; a multiple of the key heads.
    pub num_value_heads: u32,
    /// Elements of a query or key head.
    pub key_head_dim: u32,
    /// Elements of a value head.
    pub value_head_dim: u32,
    /// Taps of the short convolution.
    pub conv_kernel_dim: u32,
}

/// The multi-token-prediction layers a checkpoint carries for speculative drafting (`mtp.*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpConfig {
    /// Decoder layers of the drafter.
    pub num_hidden_layers: u32,
    /// Whether the drafter embeds with a table of its own rather than the model's.
    pub use_dedicated_embeddings: bool,
}

/// A `Qwen3_5MoeForConditionalGeneration` checkpoint's configuration (Qwen3.5-MoE and Qwen3.6, for
/// example Qwen3.6-35B-A3B): a hybrid decoder stack of gated delta networks and gated full
/// attention, every layer routed with a gated shared expert, its multi-token-prediction layers, its
/// vision tower and the placeholder tokens a prompt presents media by.
///
/// The text model normalizes with zero-centred RMS norms (`x̂ · (1 + w)`), rotates the leading
/// `partial_rotary_factor` of each attention head by interleaved M-RoPE, and gates each attended
/// head by the sigmoid of a gate projected beside its query.
#[derive(Clone, PartialEq, Debug)]
pub struct HybridConfig {
    /// The decoder stack's shared fields; its attention heads are the full-attention layers'.
    pub text: TextConfig,
    /// Each decoder layer's token mixer, in stack order (`layer_types`).
    pub mixers: Vec<Mixer>,
    /// The gated delta networks.
    pub linear: LinearAttentionConfig,
    /// Routed experts of every layer.
    pub num_experts: u32,
    /// Experts each token is routed to; their weights always renormalize to one.
    pub num_experts_per_tok: u32,
    /// Hidden width of one routed expert's intermediate projection.
    pub moe_intermediate_size: u32,
    /// Hidden width of the shared expert's intermediate projection.
    pub shared_expert_intermediate_size: u32,
    /// The drafter's layers.
    pub mtp: MtpConfig,
    /// The vision tower.
    pub vision: VisionConfig,
    /// The token that stands for one merged image patch.
    pub image_token_id: u32,
    /// The token that stands for one merged video patch.
    pub video_token_id: u32,
    /// The token that opens a vision span.
    pub vision_start_token_id: u32,
    /// The token that closes a vision span.
    pub vision_end_token_id: u32,
}

impl HybridConfig {
    /// The checkpoint class this configuration describes.
    pub const CLASS: &'static str = "Qwen3_5MoeForConditionalGeneration";

    /// The precision the references compute the rotary spectrum in: FP32, as Hugging Face's
    /// `compute_default_rope_parameters` and vLLM's rotary embedding do.
    const PRECISION: Precision = Precision::F32;

    /// The position axes of the language model's M-RoPE: time, height and width.
    pub const POSITION_AXES: u32 = 3;

    /// Read and validate the checkpoint's `config.json`.
    ///
    /// # Errors
    ///
    /// A missing file, or what [`Self::parse`] refuses.
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
    ///
    /// # Errors
    ///
    /// Another architecture, a field of the wrong type or outside its domain, a layer type other
    /// than `linear_attention` or `full_attention`, a stack without both kinds' counts consistent,
    /// a rotary recipe other than interleaved three-axis M-RoPE over the rotated channels, or a
    /// placeholder outside the vocabulary.
    pub fn parse(fields: &Fields<'_>) -> Result<Self> {
        let architectures = fields
            .optional_texts("architectures")?
            .ok_or_else(|| Error::Config("checkpoint declares no architectures".to_owned()))?;
        if architectures != [Self::CLASS] {
            return Err(Error::Config(format!(
                "expected exactly one architecture {}, found {architectures:?}",
                Self::CLASS
            )));
        }
        let values = fields.object("text_config")?;
        if values.optional_flag("use_sliding_window")?.unwrap_or(false) {
            return Err(Error::Config(
                "sliding-window attention is not supported".to_owned(),
            ));
        }
        // Every layer routes: the stack states no dense MLP width.
        let mut text =
            TextConfig::read(&values, 0, |head_dim, _| hybrid_spectrum(&values, head_dim))?;
        text.rms_norm_gain = Gain::Centred;
        text.attn_output_gate = values.optional_flag("attn_output_gate")?.unwrap_or(true);
        if text.hidden_act != Activation::Silu {
            return Err(Error::Config(format!(
                "the experts of a {} gate with silu, not {}",
                Self::CLASS,
                text.hidden_act
            )));
        }

        let mixers = mixers(&values, text.num_hidden_layers)?;
        let linear = LinearAttentionConfig {
            num_key_heads: values.count("linear_num_key_heads")?,
            num_value_heads: values.count("linear_num_value_heads")?,
            key_head_dim: values.count("linear_key_head_dim")?,
            value_head_dim: values.count("linear_value_head_dim")?,
            conv_kernel_dim: values.count("linear_conv_kernel_dim")?,
        };
        if !linear.num_value_heads.is_multiple_of(linear.num_key_heads) {
            return Err(Error::Config(format!(
                "{} linear value heads do not group over {} key heads",
                linear.num_value_heads, linear.num_key_heads
            )));
        }
        if linear.conv_kernel_dim < 2 {
            return Err(Error::Config(
                "a short convolution of fewer than two taps keeps no state".to_owned(),
            ));
        }
        // The state is FP32 in every reference; a checkpoint that asks for another is refused
        // rather than silently computed in FP32.
        if let Some(dtype) = values.optional_text("mamba_ssm_dtype")?
            && dtype != "float32"
        {
            return Err(Error::Config(format!(
                "a recurrent state of {dtype} is not served; the state is float32"
            )));
        }

        let (num_experts, num_experts_per_tok) = routed_experts(&values, Self::CLASS)?;
        let mtp = MtpConfig {
            num_hidden_layers: values.optional_index("mtp_num_hidden_layers")?.unwrap_or(0),
            use_dedicated_embeddings: values
                .optional_flag("mtp_use_dedicated_embeddings")?
                .unwrap_or(false),
        };
        let config = Self {
            mixers,
            linear,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size: values.count("moe_intermediate_size")?,
            shared_expert_intermediate_size: values.count("shared_expert_intermediate_size")?,
            mtp,
            vision: VisionConfig::parse(&fields.object("vision_config")?)?,
            image_token_id: fields.index("image_token_id")?,
            video_token_id: fields.index("video_token_id")?,
            vision_start_token_id: fields.index("vision_start_token_id")?,
            vision_end_token_id: fields.index("vision_end_token_id")?,
            text,
        };
        for token in [
            config.image_token_id,
            config.video_token_id,
            config.vision_start_token_id,
            config.vision_end_token_id,
        ] {
            if token >= config.text.vocab_size {
                return Err(Error::Config(format!(
                    "vision token {token} is outside the {} entry vocabulary",
                    config.text.vocab_size
                )));
            }
        }
        if config.vision.out_hidden_size != config.text.hidden_size {
            return Err(Error::Config(format!(
                "vision tokens are {} wide and the language model {}",
                config.vision.out_hidden_size, config.text.hidden_size
            )));
        }
        Ok(config)
    }

    /// The decoder layers whose mixer is `mixer`, in stack order.
    pub fn layers_of(&self, mixer: Mixer) -> impl Iterator<Item = u32> + '_ {
        (0..self.text.num_hidden_layers).filter(move |layer| self.mixers[*layer as usize] == mixer)
    }

    /// The ordinal of `layer` among the layers of its own mixer: its index into the key/value cache
    /// for an attention layer, into the recurrent state for a linear one.
    pub fn ordinal(&self, layer: u32) -> u32 {
        let mixer = self.mixers[layer as usize];
        self.mixers[..layer as usize]
            .iter()
            .filter(|kind| **kind == mixer)
            .count() as u32
    }

    /// Check a deployment's tensor and expert splits against the model.
    ///
    /// # Errors
    ///
    /// A split that does not divide the attention heads or the experts, or a rank outside its
    /// split.
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
}

/// Each decoder layer's mixer, from `layer_types`, or from `full_attention_interval` (every
/// interval-th layer attends) when the list is absent; a checkpoint stating both must agree.
fn mixers(values: &Fields<'_>, layers: u32) -> Result<Vec<Mixer>> {
    let interval = values.optional_count("full_attention_interval")?;
    let from_interval = interval.map(|interval| {
        (0..layers)
            .map(|layer| {
                if (layer + 1).is_multiple_of(interval) {
                    Mixer::Attention
                } else {
                    Mixer::Linear
                }
            })
            .collect::<Vec<_>>()
    });
    let stated = match values.optional_texts("layer_types")? {
        Some(kinds) => Some(
            kinds
                .iter()
                .map(|kind| match *kind {
                    "full_attention" => Ok(Mixer::Attention),
                    "linear_attention" => Ok(Mixer::Linear),
                    other => Err(Error::Config(format!(
                        "layer type {other} is neither full_attention nor linear_attention"
                    ))),
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        None => None,
    };
    let mixers = match (stated, from_interval) {
        (Some(stated), Some(derived)) if stated != derived => {
            return Err(Error::Config(
                "layer_types disagrees with full_attention_interval".to_owned(),
            ));
        }
        (Some(stated), _) => stated,
        (None, Some(derived)) => derived,
        (None, None) => {
            return Err(Error::Config(
                "a hybrid stack states neither layer_types nor full_attention_interval".to_owned(),
            ));
        }
    };
    if mixers.len() != layers as usize {
        return Err(Error::Config(format!(
            "{} layer types for {layers} layers",
            mixers.len()
        )));
    }
    Ok(mixers)
}

/// The spectrum of a hybrid stack's attention heads: interleaved M-RoPE over the leading
/// `partial_rotary_factor` of a `head_dim`-wide head, from `rope_parameters` (or its older
/// spelling `rope_scaling` with a top-level `rope_theta` and `partial_rotary_factor`).
fn hybrid_spectrum(values: &Fields<'_>, head_dim: u32) -> Result<Spectrum> {
    let parameters = match values.optional_object("rope_parameters")? {
        Some(parameters) => parameters,
        None => values.object("rope_scaling")?,
    };
    let read = |key: &str| -> Result<f64> {
        match parameters.optional_number(key)? {
            Some(value) => Ok(value),
            None => Ok(values.number(key)?),
        }
    };
    if parameters.optional_text("rope_type")?.unwrap_or("default") != "default"
        || !parameters.flag("mrope_interleaved")?
    {
        return Err(Error::Config(
            "a hybrid stack rotates by the default recipe with interleaved M-RoPE sections"
                .to_owned(),
        ));
    }
    let factor = read("partial_rotary_factor")?;
    let rotated = f64::from(head_dim) * factor;
    if !(factor > 0.0 && factor <= 1.0)
        || rotated.fract() != 0.0
        || !(rotated as u32).is_multiple_of(2)
    {
        return Err(Error::Config(format!(
            "a partial_rotary_factor of {factor} rotates no whole number of pairs of a \
             {head_dim}-channel head"
        )));
    }
    let dim = rotated as u32;
    let sections = parameters.indices("mrope_section")?;
    if sections.len() != HybridConfig::POSITION_AXES as usize
        || sections.iter().sum::<u32>() != dim / 2
    {
        return Err(Error::Config(format!(
            "M-RoPE sections {sections:?} must split the {dim} rotated channels' {} frequencies \
             over three axes",
            dim / 2
        )));
    }
    Ok(Spectrum::Sectioned {
        theta: read("rope_theta")? as f32,
        dim,
        sections: nn::layers::Sections::new(&sections)?,
        precision: HybridConfig::PRECISION,
    })
}

/// A released Qwen3-VL checkpoint's configuration, as this crate's tests read it.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;

    /// [`vl`], read.
    pub(crate) fn released() -> VlConfig {
        VlConfig::parse(&Fields::of("config.json", &vl()).expect("an object"))
            .expect("the released configuration")
    }

    /// The configuration, trimmed to the fields the family reads.
    pub(crate) fn vl() -> serde_json::Value {
        serde_json::json!({
            "image_token_id": 151655,
            "video_token_id": 151656,
            "tie_word_embeddings": false,
            "text_config": {
                "attention_bias": false,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 5120,
                "intermediate_size": 25600,
                "max_position_embeddings": 262144,
                "num_attention_heads": 64,
                "num_hidden_layers": 64,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_scaling": {
                    "mrope_interleaved": true,
                    "mrope_section": [24, 20, 20],
                    "rope_type": "default"
                },
                "rope_theta": 5000000,
                "vocab_size": 151936
            },
            "vision_config": {
                "deepstack_visual_indexes": [8, 16, 24],
                "depth": 27,
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1152,
                "in_channels": 3,
                "intermediate_size": 4304,
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": 5120,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::vl;
    use super::*;

    fn parse_vl(value: &serde_json::Value) -> Result<VlConfig> {
        VlConfig::parse(&Fields::of("config.json", value)?)
    }

    #[test]
    fn a_vl_checkpoint_reads_its_dimensions_and_an_fp32_mrope_spectrum() {
        let config = parse_vl(&vl()).expect("parsed");
        assert_eq!(config.text.hidden_size, 5120);
        assert_eq!(config.text.num_hidden_layers, 64);
        assert_eq!(config.text.num_key_value_heads * config.text.head_dim, 1024);
        let Spectrum::Sectioned {
            sections,
            precision,
            ..
        } = config.text.rope
        else {
            panic!("an M-RoPE spectrum, not {:?}", config.text.rope);
        };
        assert_eq!(sections.widths(), [24, 20, 20]);
        assert_eq!(precision, Precision::F32);
        assert_eq!(config.vision.depth, 27);
        assert_eq!(config.vision.deepstack_visual_indexes, vec![8, 16, 24]);
        // The checkpoint's widths count frequencies and the operator's count rotated elements;
        // the device's assignment names one axis per compact frequency.
        let widths: Vec<u32> = sections.widths().iter().map(|width| width * 2).collect();
        assert_eq!(
            ops::rope::interleaved_axes(&widths).len(),
            (config.text.head_dim / 2) as usize
        );
        let frequencies = config.text.rope.inverse_frequencies().expect("the recipe");
        assert_eq!(frequencies.len(), config.text.head_dim as usize / 2);
        assert_eq!(frequencies[0], 1.0);
        assert!(frequencies[63] < frequencies[0]);
    }

    #[test]
    fn a_vl_checkpoint_refuses_mrope_sections_that_do_not_split_a_head() {
        let mut values = vl();
        values["text_config"]["rope_scaling"]["mrope_section"] = serde_json::json!([24, 20, 19]);
        let error = parse_vl(&values).expect_err("sections");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("M-RoPE")),
            "{error}"
        );
    }

    #[test]
    fn a_vl_checkpoint_refuses_a_rope_that_is_not_interleaved_mrope() {
        let mut plain = vl();
        plain["text_config"]["rope_scaling"] = serde_json::json!({"rope_type": "default"});
        assert!(parse_vl(&plain).is_err());
        let mut contiguous = vl();
        contiguous["text_config"]["rope_scaling"]["mrope_interleaved"] = serde_json::json!(false);
        assert!(parse_vl(&contiguous).is_err());
    }

    #[test]
    fn a_vision_tower_refuses_a_position_table_that_is_not_square_and_blocks_outside_it() {
        let mut table = vl();
        table["vision_config"]["num_position_embeddings"] = serde_json::json!(2305);
        assert!(parse_vl(&table).is_err());

        let mut blocks = vl();
        blocks["vision_config"]["deepstack_visual_indexes"] = serde_json::json!([8, 16, 27]);
        assert!(parse_vl(&blocks).is_err());

        let mut repeated = vl();
        repeated["vision_config"]["deepstack_visual_indexes"] = serde_json::json!([8, 8]);
        assert!(parse_vl(&repeated).is_err());
    }

    #[test]
    fn merged_tokens_must_be_the_language_models_width() {
        let mut values = vl();
        values["vision_config"]["out_hidden_size"] = serde_json::json!(4096);
        assert!(parse_vl(&values).is_err());
    }

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
