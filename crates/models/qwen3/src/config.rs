//! The Qwen3 family's checkpoint configuration, read once at the loading boundary.
//!
//! A checkpoint's `config.json` is read here and nowhere else, through [`nn::config::Fields`]:
//! absent keys take the defaults the Hugging Face configurations define, aliases are resolved, and
//! every field is validated before a configuration exists. The models consume typed fields and
//! never parse a dictionary again.
//!
//! One [`Config`] describes every Qwen3 model this crate builds, composed along the model's
//! numerical parts:
//!
//! | Part | Type | Present for |
//! | --- | --- | --- |
//! | decoder stack | [`TextConfig`] | every model |
//! | token mixer per layer | [`Mixer`] | every model; linear layers in Qwen3.5 |
//! | routed experts | [`MoeConfig`] | Qwen3-MoE, Qwen3.5-MoE |
//! | vision tower and placeholders | [`VisionConfig`] | Qwen3-VL, Qwen3.5 checkpoints with a tower |
//! | multi-token-prediction layers | [`Config::mtp_layers`] | Qwen3.5 |
//!
//! Every rotary spectrum is computed in FP32, as the references compute `inv_freq`
//! (`torch.arange(..., dtype=torch.float)` in vLLM's and transformers' rotary embeddings).

use nn::blocks::GatedDeltaNetSpec;
use nn::config::Fields;
use nn::family::Split;
use nn::layers::{Gain, Precision, Sections, Spectrum};
use ops::activation::Activation;

use crate::{Error, Result};

/// The precision every Qwen3 rotary spectrum is evaluated in.
const PRECISION: Precision = Precision::F32;

/// The M-RoPE position axes of a Qwen3-VL or Qwen3.5 language model: time, height and width.
pub const POSITION_AXES: u32 = 3;

/// The decoder stack: Hugging Face's `Qwen3Config` fields the decoder layers read.
#[derive(Clone, PartialEq, Debug)]
pub struct TextConfig {
    /// Vocabulary size, before any tensor-parallel sharding.
    pub vocab_size: u32,
    /// Width of the residual stream.
    pub hidden_size: u32,
    /// Width of a dense layer's gated MLP; zero for a stack whose every layer routes and that
    /// states none (Qwen3.5-MoE).
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
    /// Variance floor of both pre-norms, the q/k norms and the final norm.
    pub rms_norm_eps: f32,
    /// How those norms' stored weights scale: Qwen3 and Qwen3-VL store the scale, Qwen3.5 stores
    /// it zero-centred (`x̂ · (1 + w)`).
    pub rms_norm_gain: Gain,
    /// The rotary spectrum over a head's leading channels: every channel for Qwen3 and Qwen3-VL,
    /// the `partial_rotary_factor` of them for Qwen3.5. Interleaved M-RoPE is
    /// [`Spectrum::Sectioned`].
    pub rope: Spectrum,
    /// The context length the checkpoint was trained for.
    pub max_position_embeddings: u32,
    /// Whether the q/k/v and output projections carry a bias.
    pub attention_bias: bool,
    /// Whether each query head projects a gate of its width beside it, whose sigmoid scales the
    /// attended head (Qwen3.5's `attn_output_gate`).
    pub attn_output_gate: bool,
    /// Whether the vocabulary head shares the embedding matrix.
    pub tie_word_embeddings: bool,
}

/// The token mixer of one decoder layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mixer {
    /// Softmax attention of the stack's heads (`self_attn`).
    Attention,
    /// A gated delta network of this geometry (`linear_attn`).
    Linear(GatedDeltaNetSpec),
}

/// The routed experts of a mixture-of-experts stack (Qwen3-MoE's `Qwen3MoeSparseMoeBlock`,
/// Qwen3.5-MoE's `Qwen3_5MoeSparseMoeBlock`).
#[derive(Clone, PartialEq, Debug)]
pub struct MoeConfig {
    /// Experts per routed layer.
    pub num_experts: u32,
    /// Experts each token is routed to.
    pub num_experts_per_tok: u32,
    /// Width of one expert's intermediate projection.
    pub moe_intermediate_size: u32,
    /// Whether the selected experts' weights renormalize to one.
    pub norm_topk_prob: bool,
    /// Every `decoder_sparse_step`-th layer routes when `mlp_only_layers` does not name it.
    pub decoder_sparse_step: u32,
    /// Layers that keep the dense MLP.
    pub mlp_only_layers: Vec<u32>,
    /// Width of the gated shared expert every token also passes through, when the block has one.
    pub shared_expert_intermediate_size: Option<u32>,
}

impl MoeConfig {
    /// Whether `layer` routes its tokens to experts.
    pub fn routes(&self, layer: u32) -> bool {
        !self.mlp_only_layers.contains(&layer)
            && (layer + 1).is_multiple_of(self.decoder_sparse_step)
    }
}

/// The Qwen3-VL vision tower and the placeholder tokens its rows replace.
///
/// Pixels arrive as tubelets of `in_channels × temporal_patch_size × patch_size²` values; `depth`
/// blocks of width `hidden_size` encode them, and every `spatial_merge_size²` patch block merges
/// into one `out_hidden_size` token, the language model's width.
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
    /// The blocks' MLP activation: PyTorch's tanh-approximated GELU.
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
    /// The blocks whose output feeds a DeepStack merger, in increasing order; the language model
    /// adds feature `j` to its stream after its own layer `j`.
    pub deepstack_visual_indexes: Vec<u32>,
    /// The placeholder token an image's merged rows replace.
    pub image_token_id: u32,
    /// The placeholder token a video's merged rows replace.
    pub video_token_id: u32,
}

impl VisionConfig {
    /// Elements of one head: the tower states no head width, so the width over the heads.
    pub fn head_dim(&self) -> u32 {
        self.hidden_size / self.num_heads
    }

    /// The tubelet width the patch embedding reads: `in_channels · temporal_patch_size · patch²`.
    pub fn tubelet(&self) -> u32 {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// The width of one merged row before the merger projects it: `merge² · hidden`.
    pub fn merged_width(&self) -> u32 {
        self.hidden_size * self.spatial_merge_size * self.spatial_merge_size
    }

    /// The side of the square learned position table.
    pub fn position_side(&self) -> u32 {
        f64::from(self.num_position_embeddings).sqrt().round() as u32
    }
}

/// One Qwen3 checkpoint's model configuration.
#[derive(Clone, PartialEq, Debug)]
pub struct Config {
    /// The checkpoint's model class (`architectures[0]`).
    pub class: String,
    /// The decoder stack.
    pub text: TextConfig,
    /// The token mixer of each layer, in stack order.
    pub mixers: Vec<Mixer>,
    /// The routed experts, for a mixture-of-experts stack.
    pub moe: Option<MoeConfig>,
    /// The vision tower, for a checkpoint that states one.
    pub vision: Option<VisionConfig>,
    /// The multi-token-prediction layers the checkpoint carries (Qwen3.5's `mtp.*`), zero for
    /// none. They embed with the model's own table.
    pub mtp_layers: u32,
}

/// The model classes [`Config::parse`] reads.
pub const CLASSES: [&str; 3] = [QWEN3_MOE, QWEN3_5_MOE, "Qwen3VLForConditionalGeneration"];

/// `Qwen3MoeForCausalLM`, as request processing claims it.
pub const QWEN3_MOE: &str = processing::qwen3::MOE_CLASS;

/// `Qwen3_5MoeForConditionalGeneration`, as request processing claims it.
pub const QWEN3_5_MOE: &str = processing::qwen3::HYBRID_CLASS;

impl Config {
    /// Read and validate the checkpoint's `config.json`.
    ///
    /// # Errors
    ///
    /// No `config.json`, or what [`Self::parse`] refuses.
    pub fn open(checkpoint: &nn::Checkpoint) -> Result<Self> {
        let value = checkpoint.sidecar("config.json")?.ok_or_else(|| {
            Error::Config(format!(
                "{} holds no config.json",
                checkpoint.root().display()
            ))
        })?;
        Self::parse(&Fields::of("config.json", &value)?)
    }

    /// Validate a parsed `config.json` of one of [`CLASSES`].
    ///
    /// # Errors
    ///
    /// Another class, a field of the wrong type or outside its domain, or an architecture
    /// feature that is not served (sliding windows, a recurrent state other than FP32, a drafter
    /// with its own embedding).
    pub fn parse(fields: &Fields<'_>) -> Result<Self> {
        let architectures = fields
            .optional_texts("architectures")?
            .ok_or_else(|| Error::Config("checkpoint declares no architectures".to_owned()))?;
        let [class] = architectures.as_slice() else {
            return Err(Error::Config(format!(
                "expected one architecture, found {architectures:?}"
            )));
        };
        let config = match *class {
            QWEN3_MOE => Self::qwen3_moe(fields)?,
            QWEN3_5_MOE => Self::qwen3_5_moe(fields)?,
            "Qwen3VLForConditionalGeneration" => Self::qwen3_vl(fields)?,
            other => {
                return Err(Error::Config(format!(
                    "{other} is not a Qwen3 class this crate builds ({CLASSES:?})"
                )));
            }
        };
        config.check()?;
        Ok(config)
    }

    /// `Qwen3MoeForCausalLM`: a text root, every layer attending, the experts at the root.
    fn qwen3_moe(fields: &Fields<'_>) -> Result<Self> {
        if let Some(kinds) = fields.optional_texts("layer_types")?
            && kinds.iter().any(|kind| *kind != "full_attention")
        {
            return Err(Error::Config(
                "layer_types other than full_attention are not served".to_owned(),
            ));
        }
        let text = TextConfig::read(
            fields,
            fields.count("intermediate_size")?,
            |head_dim, max| rope(fields, head_dim, max),
        )?;
        let (num_experts, num_experts_per_tok) = routed_experts(fields)?;
        let mlp_only_layers = fields
            .optional_indices("mlp_only_layers")?
            .unwrap_or_default();
        let moe = MoeConfig {
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size: fields
                .optional_count("moe_intermediate_size")?
                .unwrap_or(768),
            norm_topk_prob: fields.optional_flag("norm_topk_prob")?.unwrap_or(false),
            decoder_sparse_step: fields.optional_count("decoder_sparse_step")?.unwrap_or(1),
            mlp_only_layers,
            shared_expert_intermediate_size: None,
        };
        Ok(Self {
            class: QWEN3_MOE.to_owned(),
            mixers: vec![Mixer::Attention; text.num_hidden_layers as usize],
            text,
            moe: Some(moe),
            vision: None,
            mtp_layers: 0,
        })
    }

    /// `Qwen3VLForConditionalGeneration`: a dense text model under `text_config`, rotating by
    /// interleaved M-RoPE, and its vision tower.
    fn qwen3_vl(fields: &Fields<'_>) -> Result<Self> {
        let values = fields.object("text_config")?;
        let text = TextConfig::read(
            &values,
            values.count("intermediate_size")?,
            |head_dim, max| rope(&values, head_dim, max),
        )?;
        Ok(Self {
            class: "Qwen3VLForConditionalGeneration".to_owned(),
            mixers: vec![Mixer::Attention; text.num_hidden_layers as usize],
            text,
            moe: None,
            vision: Some(VisionConfig::read(fields)?),
            mtp_layers: 0,
        })
    }

    /// `Qwen3_5MoeForConditionalGeneration`: a hybrid stack of gated delta networks and gated full
    /// attention under `text_config`, every layer routed with a gated shared expert, zero-centred
    /// norms, partial interleaved M-RoPE, and the multi-token-prediction layers.
    fn qwen3_5_moe(fields: &Fields<'_>) -> Result<Self> {
        let values = fields.object("text_config")?;
        let mut text = TextConfig::read(&values, 0, |head_dim, _| partial_rope(&values, head_dim))?;
        text.rms_norm_gain = Gain::Centred;
        text.attn_output_gate = values.optional_flag("attn_output_gate")?.unwrap_or(true);
        // The state is FP32 in every reference; a checkpoint that asks for another is refused
        // rather than computed in FP32.
        if let Some(dtype) = values.optional_text("mamba_ssm_dtype")?
            && dtype != "float32"
        {
            return Err(Error::Config(format!(
                "a recurrent state of {dtype} is not served; the state is float32"
            )));
        }
        let linear = GatedDeltaNetSpec {
            hidden: text.hidden_size,
            key_heads: values.count("linear_num_key_heads")?,
            value_heads: values.count("linear_num_value_heads")?,
            key_dim: values.count("linear_key_head_dim")?,
            value_dim: values.count("linear_value_head_dim")?,
            conv_width: values.count("linear_conv_kernel_dim")?,
            eps: text.rms_norm_eps,
        };
        if !linear.value_heads.is_multiple_of(linear.key_heads) {
            return Err(Error::Config(format!(
                "{} linear value heads do not group over {} key heads",
                linear.value_heads, linear.key_heads
            )));
        }
        if linear.conv_width < 2 {
            return Err(Error::Config(
                "a short convolution of fewer than two taps keeps no state".to_owned(),
            ));
        }
        let mixers = mixers(&values, text.num_hidden_layers, linear)?;
        if values
            .optional_flag("mtp_use_dedicated_embeddings")?
            .unwrap_or(false)
        {
            return Err(Error::Config(
                "the drafter embeds with a table of its own (mtp_use_dedicated_embeddings), which \
                 is not served"
                    .to_owned(),
            ));
        }
        let (num_experts, num_experts_per_tok) = routed_experts(&values)?;
        let moe = MoeConfig {
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size: values.count("moe_intermediate_size")?,
            // The router's top-k weights renormalize to one (`Qwen3_5MoeTopKRouter`).
            norm_topk_prob: true,
            decoder_sparse_step: 1,
            mlp_only_layers: Vec::new(),
            shared_expert_intermediate_size: Some(values.count("shared_expert_intermediate_size")?),
        };
        Ok(Self {
            class: QWEN3_5_MOE.to_owned(),
            mtp_layers: values.optional_index("mtp_num_hidden_layers")?.unwrap_or(0),
            text,
            mixers,
            moe: Some(moe),
            vision: Some(VisionConfig::read(fields)?),
        })
    }

    /// The cross-part checks: experts gate with an activation the expert kernels serve, the vision
    /// tower merges into the language width, and its placeholders are tokens of the vocabulary.
    fn check(&self) -> Result<()> {
        if self.moe.is_some() && self.text.hidden_act == Activation::Gelu {
            return Err(Error::Config(
                "experts gate with silu or gelu_pytorch_tanh, not exact gelu".to_owned(),
            ));
        }
        if let Some(moe) = &self.moe {
            if moe.num_experts_per_tok > moe.num_experts {
                return Err(Error::Config(
                    "num_experts_per_tok exceeds num_experts".to_owned(),
                ));
            }
            if moe
                .mlp_only_layers
                .iter()
                .any(|layer| *layer >= self.text.num_hidden_layers)
            {
                return Err(Error::Config(
                    "mlp_only_layers names a layer outside the stack".to_owned(),
                ));
            }
        }
        if let Some(vision) = &self.vision {
            if vision.out_hidden_size != self.text.hidden_size {
                return Err(Error::Config(format!(
                    "vision tokens are {} wide and the language model {}",
                    vision.out_hidden_size, self.text.hidden_size
                )));
            }
            for token in [vision.image_token_id, vision.video_token_id] {
                if token >= self.text.vocab_size {
                    return Err(Error::Config(format!(
                        "vision placeholder {token} is outside the {}-entry vocabulary",
                        self.text.vocab_size
                    )));
                }
            }
            let axes = match self.text.rope {
                Spectrum::Sectioned { sections, .. } => sections.axes(),
                _ => 0,
            };
            if axes != POSITION_AXES {
                return Err(Error::Config(format!(
                    "a language model with vision rotates by interleaved M-RoPE over \
                     {POSITION_AXES} axes, not {:?}",
                    self.text.rope
                )));
            }
        }
        Ok(())
    }

    /// Whether `layer` routes its tokens to experts.
    pub fn routes(&self, layer: u32) -> bool {
        self.moe.as_ref().is_some_and(|moe| moe.routes(layer))
    }

    /// The layers whose mixer is attention, in stack order: the layers the key/value cache holds.
    pub fn attention_layers(&self) -> u32 {
        self.mixers
            .iter()
            .filter(|mixer| matches!(mixer, Mixer::Attention))
            .count() as u32
    }

    /// The index of `layer` among the layers of its own mixer kind: its key/value cache layer when
    /// it attends, its recurrent state layer when it is linear.
    pub fn ordinal(&self, layer: usize) -> usize {
        let linear = matches!(self.mixers[layer], Mixer::Linear(_));
        self.mixers[..layer]
            .iter()
            .filter(|mixer| matches!(mixer, Mixer::Linear(_)) == linear)
            .count()
    }

    /// Check a deployment's tensor and expert splits: the tensor split divides the heads and the
    /// expert split the experts.
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
        if let Some(moe) = &self.moe
            && !moe.num_experts.is_multiple_of(experts.size)
        {
            return Err(Error::Config(format!(
                "expert parallel degree {} does not divide {} experts",
                experts.size, moe.num_experts
            )));
        }
        Ok(())
    }
}

impl TextConfig {
    /// The fields every Qwen3 decoder stack states, with the dense MLP's width `intermediate_size`
    /// and the spectrum `rope` answers for a head width and a context length.
    fn read(
        fields: &Fields<'_>,
        intermediate_size: u32,
        rope: impl FnOnce(u32, u32) -> Result<Spectrum>,
    ) -> Result<Self> {
        if fields.optional_flag("use_sliding_window")?.unwrap_or(false) {
            return Err(Error::Config(
                "sliding-window attention is not served".to_owned(),
            ));
        }
        let hidden_size = fields.count("hidden_size")?;
        let num_attention_heads = fields.count("num_attention_heads")?;
        // Transformers derives the head width from the attention width when it is absent.
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
        let num_key_value_heads = fields.count("num_key_value_heads")?;
        if !num_attention_heads.is_multiple_of(num_key_value_heads) {
            return Err(Error::Config(
                "num_attention_heads must be divisible by num_key_value_heads".to_owned(),
            ));
        }
        if !head_dim.is_multiple_of(2) {
            return Err(Error::Config("head_dim must be even".to_owned()));
        }
        let max_position_embeddings = fields
            .optional_count("max_position_embeddings")?
            .unwrap_or(32768);
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
            rope: rope(head_dim, max_position_embeddings)?,
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

impl VisionConfig {
    /// The tower under `vision_config` and the placeholder tokens at the root.
    ///
    /// # Errors
    ///
    /// A field of the wrong type, a block activation other than `gelu_pytorch_tanh`, DeepStack
    /// blocks out of order or outside the tower, heads that do not split the width into two even
    /// rotary halves, or a position table that is not a square grid.
    fn read(root: &Fields<'_>) -> Result<Self> {
        let values = root.object("vision_config")?;
        let owner = values.owner().to_owned();
        let hidden_act = values.activation("hidden_act")?;
        if hidden_act != Activation::GeluTanh {
            return Err(Error::Config(format!(
                "{owner}.hidden_act must be gelu_pytorch_tanh, got {hidden_act}"
            )));
        }
        let depth = values.count("depth")?;
        let deepstack_visual_indexes = values
            .optional_indices("deepstack_visual_indexes")?
            .unwrap_or_default();
        if deepstack_visual_indexes
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
            || deepstack_visual_indexes.iter().any(|index| *index >= depth)
        {
            return Err(Error::Config(format!(
                "{owner}.deepstack_visual_indexes {deepstack_visual_indexes:?} must increase \
                 strictly inside the {depth}-block tower"
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
            image_token_id: root.index("image_token_id")?,
            video_token_id: root.index("video_token_id")?,
        };
        // The 2-D rotary turns half of each head by the patch row and half by its column.
        if !config.hidden_size.is_multiple_of(config.num_heads)
            || !config.head_dim().is_multiple_of(4)
        {
            return Err(Error::Config(format!(
                "{} vision heads must split the {} wide tower into heads of a multiple of four",
                config.num_heads, config.hidden_size
            )));
        }
        let side = config.position_side();
        if side * side != config.num_position_embeddings {
            return Err(Error::Config(
                "the vision position table must be a square grid".to_owned(),
            ));
        }
        Ok(config)
    }
}

/// The routed experts a configuration states: the expert count and the experts per token.
///
/// Transformers serializes the expert count as `num_local_experts` in newer releases; a
/// configuration carrying both must agree.
fn routed_experts(fields: &Fields<'_>) -> Result<(u32, u32)> {
    let experts = fields
        .alias(&["num_experts", "num_local_experts"])?
        .ok_or_else(|| Error::Config("a mixture of experts declares no expert count".to_owned()))?;
    let experts = fields.index(experts)?;
    if experts == 0 {
        return Err(Error::Config(
            "a mixture of experts declares no experts".to_owned(),
        ));
    }
    Ok((
        experts,
        fields.optional_count("num_experts_per_tok")?.unwrap_or(8),
    ))
}

/// The rotary spectrum of a Qwen3 or Qwen3-VL text configuration over `head_dim` channels.
///
/// A `rope_scaling` with `mrope_section` is Qwen3-VL's interleaved M-RoPE over the default
/// spectrum, its section widths counting frequencies; every other recipe is
/// [`Spectrum::from_hf`]'s.
fn rope(fields: &Fields<'_>, head_dim: u32, max_position_embeddings: u32) -> Result<Spectrum> {
    if let Some(scaling) = fields.optional_object("rope_scaling")?
        && scaling.contains("mrope_section")
    {
        if scaling.text("rope_type")? != "default" || !scaling.flag("mrope_interleaved")? {
            return Err(Error::Config(
                "an M-RoPE rope_scaling must be the default recipe with interleaved sections"
                    .to_owned(),
            ));
        }
        return sectioned(
            fields.positive("rope_theta")? as f32,
            head_dim,
            &scaling.indices("mrope_section")?,
        );
    }
    Ok(
        match Spectrum::from_hf(fields, head_dim, max_position_embeddings)? {
            Spectrum::Default { theta, dim, .. } => Spectrum::Default {
                theta,
                dim,
                precision: PRECISION,
            },
            other => other,
        },
    )
}

/// Qwen3.5's spectrum: interleaved M-RoPE over the `partial_rotary_factor` of a head's channels,
/// stated in `rope_parameters` (or `rope_scaling`), a value the recipe does not state read from
/// the text configuration itself.
fn partial_rope(values: &Fields<'_>, head_dim: u32) -> Result<Spectrum> {
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
            "a Qwen3.5 stack rotates by the default recipe with interleaved M-RoPE sections"
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
    sectioned(
        read("rope_theta")? as f32,
        rotated as u32,
        &parameters.indices("mrope_section")?,
    )
}

/// Interleaved M-RoPE over `dim` rotated channels, `sections` counting frequencies per axis.
fn sectioned(theta: f32, dim: u32, sections: &[u32]) -> Result<Spectrum> {
    if sections.len() != POSITION_AXES as usize || sections.iter().sum::<u32>() != dim / 2 {
        return Err(Error::Config(format!(
            "M-RoPE sections {sections:?} must split the {} frequencies of {dim} rotated channels \
             over {POSITION_AXES} axes",
            dim / 2
        )));
    }
    Ok(Spectrum::Sectioned {
        theta,
        dim,
        sections: Sections::new(sections)?,
        precision: PRECISION,
    })
}

/// Each layer's mixer of a hybrid stack, from `layer_types` or `full_attention_interval`, which
/// must agree when both are stated.
fn mixers(values: &Fields<'_>, layers: u32, linear: GatedDeltaNetSpec) -> Result<Vec<Mixer>> {
    let derived = values
        .optional_count("full_attention_interval")?
        .map(|interval| {
            (0..layers)
                .map(|layer| {
                    if (layer + 1).is_multiple_of(interval) {
                        Mixer::Attention
                    } else {
                        Mixer::Linear(linear)
                    }
                })
                .collect::<Vec<_>>()
        });
    let stated = values
        .optional_texts("layer_types")?
        .map(|kinds| {
            kinds
                .iter()
                .map(|kind| match *kind {
                    "full_attention" => Ok(Mixer::Attention),
                    "linear_attention" => Ok(Mixer::Linear(linear)),
                    other => Err(Error::Config(format!(
                        "layer type {other} is neither full_attention nor linear_attention"
                    ))),
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let mixers = match (stated, derived) {
        (Some(stated), Some(derived)) if stated != derived => {
            return Err(Error::Config(
                "layer_types disagrees with full_attention_interval".to_owned(),
            ));
        }
        (Some(mixers), _) | (None, Some(mixers)) => mixers,
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

/// Released configurations, as this crate's tests read them.
#[cfg(test)]
pub(crate) mod fixtures {
    /// A Qwen3-VL text encoder's configuration (the MiniMax-H3 text encoder's), trimmed to the
    /// fields the family reads.
    pub(crate) fn vl() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Qwen3VLForConditionalGeneration"],
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

    /// A Qwen3-30B-A3B configuration, four layers deep.
    pub(crate) fn moe() -> serde_json::Value {
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
            "use_sliding_window": false
        })
    }

    /// A Qwen3.6-35B-A3B configuration, four layers deep (three linear, one attention).
    pub(crate) fn hybrid() -> serde_json::Value {
        let mut root = vl();
        root["architectures"] = serde_json::json!(["Qwen3_5MoeForConditionalGeneration"]);
        root["vision_config"]["deepstack_visual_indexes"] = serde_json::json!([]);
        root["vision_config"]["out_hidden_size"] = serde_json::json!(2048);
        root["text_config"] = serde_json::json!({
            "attn_output_gate": true,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "moe_intermediate_size": 512,
            "mtp_num_hidden_layers": 1,
            "num_attention_heads": 16,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default"
            },
            "shared_expert_intermediate_size": 512,
            "vocab_size": 248320
        });
        root
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{hybrid, moe, vl};
    use super::*;

    fn parse(value: &serde_json::Value) -> Result<Config> {
        Config::parse(&Fields::of("config.json", value)?)
    }

    #[test]
    fn a_moe_checkpoint_routes_every_layer_and_rotates_in_fp32() {
        let config = parse(&moe()).expect("a supported checkpoint");
        assert_eq!(config.text.hidden_size, 2048);
        assert!((0..4).all(|layer| config.routes(layer)));
        assert_eq!(config.attention_layers(), 4);
        assert!(config.vision.is_none());
        assert!(matches!(
            config.text.rope,
            Spectrum::Default { theta, dim: 128, precision: Precision::F32 } if theta == 1e6
        ));
    }

    #[test]
    fn mlp_only_layers_and_the_sparse_step_keep_dense_layers() {
        let mut value = moe();
        value["decoder_sparse_step"] = serde_json::json!(2);
        value["mlp_only_layers"] = serde_json::json!([3]);
        let config = parse(&value).expect("parsed");
        let routed: Vec<u32> = (0..4).filter(|layer| config.routes(*layer)).collect();
        assert_eq!(routed, [1]);
    }

    #[test]
    fn head_dim_defaults_to_the_attention_width() {
        let mut value = moe();
        value.as_object_mut().expect("object").remove("head_dim");
        assert_eq!(parse(&value).expect("parsed").text.head_dim, 64);
    }

    #[test]
    fn another_class_and_sliding_windows_are_refused() {
        let mut class = moe();
        class["architectures"] = serde_json::json!(["Qwen3ForCausalLM"]);
        assert!(parse(&class).is_err());
        let mut window = moe();
        window["use_sliding_window"] = serde_json::json!(true);
        assert!(parse(&window).is_err());
    }

    #[test]
    fn conflicting_expert_aliases_are_refused() {
        let mut value = moe();
        value["num_local_experts"] = serde_json::json!(64);
        assert!(parse(&value).is_err());
        value["num_local_experts"] = serde_json::json!(128);
        assert_eq!(
            parse(&value)
                .expect("agreeing aliases")
                .moe
                .expect("experts")
                .num_experts,
            128
        );
    }

    #[test]
    fn a_field_of_the_wrong_type_is_refused_by_name() {
        let mut value = moe();
        value["attention_bias"] = serde_json::json!("no");
        let error = parse(&value)
            .expect_err("a string is not a flag")
            .to_string();
        assert!(error.contains("config.json.attention_bias"), "{error}");
    }

    #[test]
    fn splits_must_divide_the_heads_and_the_experts() {
        let config = parse(&moe()).expect("parsed");
        let split = |size, rank| Split { size, rank };
        assert_eq!(config.text.local_heads(split(4, 2)).expect("four"), (8, 1));
        assert!(config.check_splits(split(4, 3), split(4, 2)).is_ok());
        assert!(config.text.local_heads(split(3, 0)).is_err());
        assert!(config.text.local_heads(split(4, 4)).is_err());
        assert!(config.check_splits(split(1, 0), split(3, 0)).is_err());
    }

    #[test]
    fn a_vl_checkpoint_reads_its_tower_and_an_interleaved_mrope_spectrum() {
        let config = parse(&vl()).expect("parsed");
        let vision = config.vision.as_ref().expect("a tower");
        assert_eq!(vision.depth, 27);
        assert_eq!(vision.deepstack_visual_indexes, [8, 16, 24]);
        assert_eq!(vision.head_dim(), 72);
        assert_eq!(vision.position_side(), 48);
        let Spectrum::Sectioned { sections, .. } = config.text.rope else {
            panic!("an M-RoPE spectrum, not {:?}", config.text.rope);
        };
        assert_eq!(sections.widths(), [24, 20, 20]);
        assert!(config.moe.is_none());
    }

    #[test]
    fn a_vl_checkpoint_refuses_sections_that_do_not_split_a_head_and_a_plain_rope() {
        let mut sections = vl();
        sections["text_config"]["rope_scaling"]["mrope_section"] = serde_json::json!([24, 20, 19]);
        assert!(parse(&sections).is_err());
        let mut plain = vl();
        plain["text_config"]["rope_scaling"] = serde_json::json!({"rope_type": "default"});
        assert!(parse(&plain).is_err());
    }

    #[test]
    fn a_vision_tower_refuses_a_table_that_is_not_square_and_blocks_outside_it() {
        let mut table = vl();
        table["vision_config"]["num_position_embeddings"] = serde_json::json!(2305);
        assert!(parse(&table).is_err());
        let mut blocks = vl();
        blocks["vision_config"]["deepstack_visual_indexes"] = serde_json::json!([8, 16, 27]);
        assert!(parse(&blocks).is_err());
        let mut width = vl();
        width["vision_config"]["out_hidden_size"] = serde_json::json!(4096);
        assert!(parse(&width).is_err());
    }

    #[test]
    fn a_hybrid_checkpoint_interleaves_its_mixers_and_shares_an_expert() {
        let config = parse(&hybrid()).expect("parsed");
        assert!(matches!(config.mixers[3], Mixer::Attention));
        assert!(matches!(config.mixers[0], Mixer::Linear(spec) if spec.value_heads == 32));
        assert_eq!(config.attention_layers(), 1);
        assert_eq!((config.ordinal(2), config.ordinal(3)), (2, 0));
        assert_eq!(config.text.rms_norm_gain, Gain::Centred);
        assert!(config.text.attn_output_gate);
        assert_eq!(config.mtp_layers, 1);
        let moe = config.moe.expect("experts");
        assert_eq!(moe.shared_expert_intermediate_size, Some(512));
        assert!(moe.norm_topk_prob);
        assert!(matches!(
            config.text.rope,
            Spectrum::Sectioned { dim: 64, .. }
        ));
    }

    #[test]
    fn disagreeing_layer_types_and_a_dedicated_drafter_embedding_are_refused() {
        let mut types = hybrid();
        types["text_config"]["layer_types"] = serde_json::json!([
            "full_attention",
            "linear_attention",
            "linear_attention",
            "full_attention"
        ]);
        assert!(parse(&types).is_err());
        let mut embedding = hybrid();
        embedding["text_config"]["mtp_use_dedicated_embeddings"] = serde_json::json!(true);
        assert!(parse(&embedding).is_err());
    }
}
