//! The Qwen3 family's checkpoint configuration, read once at the loading boundary.
//!
//! A checkpoint's `config.json` is read here and nowhere else, through [`nn::config::Fields`]:
//! absent keys take the defaults the Hugging Face configurations define, aliases are resolved,
//! and every field is validated before a configuration exists. The modules consume typed fields
//! and never parse a dictionary again.
//!
//! One [`Config`] describes every Qwen3 model this family builds, composed along the model's
//! numerical submodules:
//!
//! | Submodule | Configuration | Present for |
//! | --- | --- | --- |
//! | decoder stack | [`TextConfig`] | every model |
//! | gated delta networks | [`TextConfig::gated_delta_net`] | Qwen3.5's linear layers |
//! | routed experts | [`MoeConfig`] | Qwen3-MoE, Qwen3.5-MoE |
//! | vision tower and placeholders | [`VisionConfig`] | Qwen3-VL, Qwen3.5 |
//! | multi-token-prediction layers | [`Config::mtp_num_hidden_layers`] | Qwen3.5 |
//!
//! [`Config::open`] also reads what the checkpoint's other sidecars state of the same modules:
//! the quantization its export declares and the processor rules its images are sized by.
//!
//! Every rotary spectrum is computed in FP32, as the references compute `inv_freq`
//! (`torch.arange(..., dtype=torch.float)` in vLLM's and transformers' rotary embeddings).

use std::ops::Range;

use nn::config::Fields;
use nn::family::TextPlacement;
use nn::functional::Activation;
use nn::modules::{GatedDeltaNetConfig, Precision, RotaryConfig, Spectrum};
use nn::quantization::Declaration;
use nn::{PixelNorm, Split};
use processing::qwen_vl::ProcessorConfig;

use crate::qwen3::{Error, Result};

/// The precision every Qwen3 rotary spectrum is evaluated in.
const PRECISION: Precision = Precision::F32;

/// The M-RoPE position axes of a Qwen3-VL or Qwen3.5 language model: time, height and width.
pub const POSITION_AXES: u32 = 3;

/// A tensor every Qwen3-VL and Qwen3.5 vision tower holds: the patch embedding.
const TOWER_WEIGHT: &str = "model.visual.patch_embed.proj.weight";

/// The decoder stack: the fields of Hugging Face's `Qwen3Config` its layers read.
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
    /// Whether those norms' weights are stored zero-centred, so a row is scaled by one plus the
    /// weight (Qwen3.5); Qwen3 and Qwen3-VL store the scale itself.
    pub rms_norm_centred: bool,
    /// The rotary embedding over a head's leading channels: every channel for Qwen3 and
    /// Qwen3-VL, the `partial_rotary_factor` of them for Qwen3.5. Interleaved M-RoPE is
    /// [`Spectrum::Sectioned`].
    pub rope: RotaryConfig,
    /// The context length the checkpoint was trained for.
    pub max_position_embeddings: u32,
    /// Whether the q/k/v and output projections add a bias.
    pub attention_bias: bool,
    /// Whether each query head projects a gate of its width beside it, whose sigmoid scales
    /// the attended head (Qwen3.5's `attn_output_gate`).
    pub attn_output_gate: bool,
    /// Whether the vocabulary head shares the embedding matrix.
    pub tie_word_embeddings: bool,
    /// Whether each layer mixes its tokens with a gated delta network (`linear_attention` in
    /// `layer_types`) in place of softmax attention (`full_attention`), in stack order.
    pub linear_layers: Vec<bool>,
    /// The gated delta network of the linear layers, before any split; `None` for a stack
    /// whose every layer attends.
    pub gated_delta_net: Option<GatedDeltaNetConfig>,
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
    /// Width of the gated shared expert every token also passes through, when the block has
    /// one.
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
/// Pixels arrive as tubelets of `in_channels × temporal_patch_size × patch_size²` values;
/// `depth` blocks of width `hidden_size` encode them, and every `spatial_merge_size²` patch
/// block merges into one `out_hidden_size` token, the language model's width.
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
    /// The blocks whose output feeds a DeepStack merger, in increasing order; the language
    /// model adds feature `j` to its stream after its own layer `j`.
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

    /// The tubelet width the patch embedding reads: `in_channels · temporal_patch_size ·
    /// patch²`.
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

/// One Qwen3 checkpoint's configuration.
#[derive(Clone, PartialEq, Debug)]
pub struct Config {
    /// The checkpoint's model class (`architectures[0]`).
    pub class: String,
    /// The decoder stack.
    pub text: TextConfig,
    /// The routed experts, for a mixture-of-experts stack.
    pub moe: Option<MoeConfig>,
    /// The vision tower, for a checkpoint that states one.
    pub vision: Option<VisionConfig>,
    /// The multi-token-prediction layers the checkpoint carries (Qwen3.5's `mtp.*`), zero for
    /// none. They embed with the model's own table.
    pub mtp_num_hidden_layers: u32,
    /// What the checkpoint's export declares about its packed weights; dense for a
    /// configuration read from `config.json` alone ([`Config::parse`]).
    pub quantization: Declaration,
    /// The processor rules the tower's images are sized by and their pixel normalization,
    /// when the checkpoint holds both the tower's weights and its processor files; `None` for
    /// a checkpoint that serves text alone.
    pub images: Option<(ProcessorConfig, PixelNorm)>,
}

/// The model classes [`Config::parse`] reads.
pub const CLASSES: [&str; 3] = [QWEN3_MOE, QWEN3_5_MOE, QWEN3_VL];

/// `Qwen3MoeForCausalLM`, as request processing claims it.
pub const QWEN3_MOE: &str = processing::qwen::QWEN3_MOE_CLASS;

/// `Qwen3_5MoeForConditionalGeneration`, as request processing claims it.
pub const QWEN3_5_MOE: &str = processing::qwen::QWEN3_5_MOE_CLASS;

/// `Qwen3VLForConditionalGeneration`: the class of a pipeline's text encoder.
pub const QWEN3_VL: &str = "Qwen3VLForConditionalGeneration";

impl Config {
    /// Reads and validates what `checkpoint` states: its `config.json`, the quantization its
    /// export declares, and, when it holds a vision tower, its processor files.
    ///
    /// # Errors
    ///
    /// No `config.json`, what [`Self::parse`] refuses, a declaration the loader does not
    /// serve, or processor files that are not a Qwen-VL processor's.
    pub fn open(checkpoint: &nn::Checkpoint) -> Result<Self> {
        let value = checkpoint.sidecar("config.json")?.ok_or_else(|| {
            Error::Config(format!(
                "{} holds no config.json",
                checkpoint.root().display()
            ))
        })?;
        let mut config = Self::parse(&Fields::of("config.json", &value)?)?;
        config.quantization = Declaration::read(checkpoint)?;

        // The tower serves when the checkpoint holds both its weights and its processor
        // files; otherwise the checkpoint serves text alone.
        if checkpoint.tensor(TOWER_WEIGHT).is_some() {
            config.images = processor(checkpoint)?;
        }
        Ok(config)
    }

    /// Validates a parsed `config.json` of one of [`CLASSES`].
    ///
    /// # Errors
    ///
    /// Another class, a field of the wrong type or outside its domain, or an architecture
    /// feature that is not served (sliding windows, layer scales, a recurrent state other than
    /// FP32, a drafter with its own embedding).
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
            QWEN3_VL => Self::qwen3_vl(fields)?,
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

        let text = TextConfig::read(fields, fields.count("intermediate_size")?, Some(1.0))?;
        let (num_experts, num_experts_per_tok) = routed_experts(fields)?;
        let moe = MoeConfig {
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size: fields
                .optional_count("moe_intermediate_size")?
                .unwrap_or(768),
            norm_topk_prob: fields.optional_flag("norm_topk_prob")?.unwrap_or(false),
            decoder_sparse_step: fields.optional_count("decoder_sparse_step")?.unwrap_or(1),
            mlp_only_layers: fields
                .optional_indices("mlp_only_layers")?
                .unwrap_or_default(),
            shared_expert_intermediate_size: None,
        };

        Ok(Self::of(QWEN3_MOE, text, Some(moe), None, 0))
    }

    /// `Qwen3VLForConditionalGeneration`: a dense text model under `text_config`, rotating by
    /// interleaved M-RoPE, and its vision tower.
    fn qwen3_vl(fields: &Fields<'_>) -> Result<Self> {
        let values = fields.object("text_config")?;
        let text = TextConfig::read(&values, values.count("intermediate_size")?, Some(1.0))?;

        Ok(Self::of(
            QWEN3_VL,
            text,
            None,
            Some(VisionConfig::read(fields)?),
            0,
        ))
    }

    /// `Qwen3_5MoeForConditionalGeneration`: a hybrid stack of gated delta networks and gated
    /// full attention under `text_config`, every layer routed with a gated shared expert,
    /// zero-centred norms, partial interleaved M-RoPE, and the multi-token-prediction layers.
    fn qwen3_5_moe(fields: &Fields<'_>) -> Result<Self> {
        let values = fields.object("text_config")?;
        let mut text = TextConfig::read(&values, 0, None)?;
        text.rms_norm_centred = true;
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

        let linear = GatedDeltaNetConfig {
            hidden: text.hidden_size,
            key_heads: values.count("linear_num_key_heads")?,
            value_heads: values.count("linear_num_value_heads")?,
            key_dim: values.count("linear_key_head_dim")?,
            value_dim: values.count("linear_value_head_dim")?,
            conv_width: values.count("linear_conv_kernel_dim")?,
            eps: text.rms_norm_eps,
            // Qwen3.5 gates the output norm with the SiLU and keeps one slot per request,
            // whose state rows are the keys'.
            sigmoid_gate: false,
            axes: nn::functional::StateAxes::KeyValue,
            draft_tokens: 0,
            split: Split::WHOLE,
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
        text.linear_layers = linear_layers(&values, text.num_hidden_layers)?;
        text.gated_delta_net = Some(linear);

        if values
            .optional_flag("mtp_use_dedicated_embeddings")?
            .unwrap_or(false)
        {
            return Err(Error::Config(
                "the drafter embeds with a table of its own (mtp_use_dedicated_embeddings), \
                 which is not served"
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

        Ok(Self::of(
            QWEN3_5_MOE,
            text,
            Some(moe),
            Some(VisionConfig::read(fields)?),
            values.optional_index("mtp_num_hidden_layers")?.unwrap_or(0),
        ))
    }

    /// The configuration of `class` over its submodules, as `config.json` alone states it.
    fn of(
        class: &str,
        text: TextConfig,
        moe: Option<MoeConfig>,
        vision: Option<VisionConfig>,
        mtp_num_hidden_layers: u32,
    ) -> Self {
        Self {
            class: class.to_owned(),
            text,
            moe,
            vision,
            mtp_num_hidden_layers,
            quantization: Declaration::dense(),
            images: None,
        }
    }

    /// The checks across submodules: experts gate with an activation the expert products
    /// serve, the vision tower merges into the language width, and its placeholders are tokens
    /// of the vocabulary.
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

            let axes = match self.text.rope.spectrum {
                Spectrum::Sectioned { axes, .. } => axes,
                _ => 0,
            };
            if axes != POSITION_AXES {
                return Err(Error::Config(format!(
                    "a language model with vision rotates by interleaved M-RoPE over \
                     {POSITION_AXES} axes, not {:?}",
                    self.text.rope.spectrum
                )));
            }
        }
        Ok(())
    }

    /// Whether `layer` routes its tokens to experts.
    pub fn routes(&self, layer: u32) -> bool {
        self.moe.as_ref().is_some_and(|moe| moe.routes(layer))
    }

    /// The layers whose mixer is attention: the layers the key/value cache holds.
    pub fn attention_layers(&self) -> u32 {
        self.text
            .linear_layers
            .iter()
            .filter(|linear| !**linear)
            .count() as u32
    }

    /// The layers whose mixer is a gated delta network: the layers that keep a recurrent
    /// state.
    pub fn linear_layers(&self) -> u32 {
        self.text.num_hidden_layers - self.attention_layers()
    }

    /// The index of `layer` among the layers of its own mixer: its key/value cache layer when
    /// it attends, its recurrent state layer when it is linear.
    pub fn ordinal(&self, layer: usize) -> usize {
        let linear = self.text.linear_layers[layer];

        self.text.linear_layers[..layer]
            .iter()
            .filter(|other| **other == linear)
            .count()
    }

    /// Checks a deployment's tensor and expert splits: the tensor split divides the heads and
    /// the expert split the experts.
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

    /// Checks a deployment's placement of the model and returns the decoder layers this rank's
    /// pipeline stage holds: the tensor and expert splits as [`Self::check_splits`] checks them,
    /// a tensor group that holds whole experts dividing them alone, at most one of the tensor,
    /// pipeline and sequence splits above one rank (a forward runs over one group), and the
    /// pipeline's and the sequence's own conditions.
    ///
    /// Stage `s` of `P` holds layers `s·L/P .. (s + 1)·L/P` of the stack's `L`. `P` divides `L`
    /// and every stage holds as many attention layers as every other: every rank of a group
    /// lays its cache unit out alike. A sequence split's members share the attention and the
    /// key/value heads evenly; it serves neither a recurrent state, whose scan reads every row in
    /// order, nor a drafter, whose attention is not split.
    ///
    /// # Errors
    ///
    /// A split the model does not divide, a rank outside its split, two splits above one rank,
    /// or a sequence split of a model that keeps a recurrent state or of a placement that loads
    /// the drafter.
    pub fn check_placement(&self, placement: &TextPlacement) -> Result<Range<u32>> {
        let (tensor, pipeline, sequence) = (
            placement.tensor,
            placement.pipeline,
            placement.sequence.split,
        );
        self.check_splits(tensor, placement.experts)?;
        // A tensor group that holds whole experts divides the experts, not their width, and no
        // expert group divides them beside it.
        if placement.tensor_experts
            && let Some(moe) = &self.moe
            && (placement.experts.size > 1 || !moe.num_experts.is_multiple_of(tensor.size))
        {
            return Err(Error::Config(format!(
                "a tensor group of {} holding whole experts beside an expert group of {}: the \
                 tensor group alone divides the {} experts",
                tensor.size, placement.experts.size, moe.num_experts
            )));
        }

        let split = [tensor.size, pipeline.size, sequence.size];
        if split.iter().filter(|size| **size > 1).count() > 1 {
            return Err(Error::Config(format!(
                "tensor, pipeline and sequence degrees {split:?}: a forward runs over one group, \
                 so one of them splits the model"
            )));
        }
        for (name, split) in [("pipeline", pipeline), ("sequence", sequence)] {
            if split.size == 0 || split.rank >= split.size {
                return Err(Error::Config(format!(
                    "{name} rank {} of a group of {}",
                    split.rank, split.size
                )));
            }
        }

        if sequence.size > 1 {
            let text = &self.text;
            let heads = [text.num_attention_heads, text.num_key_value_heads];
            if heads
                .iter()
                .any(|heads| !heads.is_multiple_of(sequence.size))
            {
                return Err(Error::Config(format!(
                    "a sequence split over {} members of {heads:?} attention and key/value \
                     heads: the members share the heads evenly",
                    sequence.size
                )));
            }
            if self.linear_layers() > 0 || placement.drafter {
                return Err(Error::Config(format!(
                    "a sequence split over {} members of a model whose {} layers keep a \
                     recurrent state or that loads its drafter: a recurrent scan reads every row \
                     in order, and the drafter's attention is not split",
                    sequence.size,
                    self.linear_layers()
                )));
            }
        }

        let layers = self.text.num_hidden_layers;
        let stages = pipeline.size;
        let attention_in = |range: Range<u32>| -> usize {
            range
                .filter(|layer| !self.text.linear_layers[*layer as usize])
                .count()
        };
        let per_stage = layers / stages;
        let even = layers.is_multiple_of(stages)
            && per_stage > 0
            && (1..stages).all(|stage| {
                attention_in(stage * per_stage..(stage + 1) * per_stage)
                    == attention_in(0..per_stage)
            });
        if !even {
            return Err(Error::Config(format!(
                "a pipeline of {stages} stages over {layers} decoder layers: every stage holds \
                 as many layers, and as many attention layers, as every other"
            )));
        }
        let first = pipeline.rank * per_stage;
        Ok(first..first + per_stage)
    }
}

impl TextConfig {
    /// The fields every Qwen3 decoder stack states, every layer attending, with the dense
    /// MLP's width `intermediate_size`. `partial_rotary_factor` is the share of a head an
    /// M-RoPE recipe rotates when the configuration states none, or `None` for a family whose
    /// recipe must state it.
    fn read(
        fields: &Fields<'_>,
        intermediate_size: u32,
        partial_rotary_factor: Option<f64>,
    ) -> Result<Self> {
        if fields.optional_flag("use_sliding_window")?.unwrap_or(false) {
            return Err(Error::Config(
                "sliding-window attention is not served".to_owned(),
            ));
        }
        // The references multiply each block's output by a learned scale under this flag
        // (`refs/vllm/vllm/model_executor/models/qwen3_next.py:612`); no served checkpoint
        // sets it, and a stack that ignored it would compute another model.
        if fields.optional_flag("layer_scale")?.unwrap_or(false) {
            return Err(Error::Config(
                "per-layer output scales (layer_scale) are not served".to_owned(),
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
        let num_hidden_layers = fields.count("num_hidden_layers")?;

        Ok(Self {
            vocab_size: fields.count("vocab_size")?,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_act: fields
                .optional_activation("hidden_act")?
                .unwrap_or(Activation::Silu),
            rms_norm_eps: fields
                .optional_positive("rms_norm_eps")?
                .map_or(1e-6, |value| value as f32),
            rms_norm_centred: false,
            rope: rope(
                fields,
                head_dim,
                max_position_embeddings,
                partial_rotary_factor,
            )?,
            max_position_embeddings,
            attention_bias: fields.optional_flag("attention_bias")?.unwrap_or(false),
            attn_output_gate: false,
            tie_word_embeddings: fields
                .optional_flag("tie_word_embeddings")?
                .unwrap_or(false),
            linear_layers: vec![false; num_hidden_layers as usize],
            gated_delta_net: None,
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
    /// A field of the wrong type, a block activation other than `gelu_pytorch_tanh`,
    /// DeepStack blocks out of order or outside the tower, heads that do not split the width
    /// into two even rotary halves, or a position table that is not a square grid.
    pub fn read(root: &Fields<'_>) -> Result<Self> {
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

/// The processor rules of `checkpoint` and its images' pixel normalization, or `None` when it
/// states no processor files.
pub fn processor(checkpoint: &nn::Checkpoint) -> Result<Option<(ProcessorConfig, PixelNorm)>> {
    let [image_file, video_file] = ProcessorConfig::FILES;
    let (Some(image), Some(video)) = (
        checkpoint.sidecar(image_file)?,
        checkpoint.sidecar(video_file)?,
    ) else {
        return Ok(None);
    };

    let norm = pixel_norm(&Fields::of(image_file, &image)?)?;
    let processor = ProcessorConfig::from_values(&image, &video).map_err(Error::Config)?;
    Ok(Some((processor, norm)))
}

/// The pixel normalization a processor's `image_mean` and `image_std` state: each RGB channel
/// maps `x` in `[0, 255]` to `(x / 255 - mean) / std` ([`PixelNorm`]).
///
/// # Errors
///
/// Fields of the wrong type or length, or a deviation that is not positive.
pub fn pixel_norm(values: &Fields<'_>) -> Result<PixelNorm> {
    let owner = values.owner();
    let mean = values.fixed_numbers("image_mean", 3)?;
    let std = values.fixed_numbers("image_std", 3)?;

    let mean: [f32; 3] = [mean[0] as f32, mean[1] as f32, mean[2] as f32];
    let std: [f32; 3] = [std[0] as f32, std[1] as f32, std[2] as f32];
    if std.iter().any(|value| *value <= 0.0) {
        return Err(Error::Config(format!(
            "{owner}.image_std must be positive, got {std:?}"
        )));
    }
    Ok(PixelNorm { mean, std })
}

/// The routed experts a configuration states: the expert count and the experts per token.
///
/// Transformers serializes the expert count as `num_local_experts` in newer releases; a
/// configuration that states both must agree.
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

/// The rotary embedding of a text configuration over the heads of `head_dim` channels.
///
/// A recipe (`rope_parameters`, or `rope_scaling`) that states `mrope_section` is interleaved
/// M-RoPE over the default spectrum: its sections count frequencies per axis, over the
/// `partial_rotary_factor` of a head's channels, and a value the recipe does not state is read
/// from the text configuration itself. `partial_rotary_factor` is the factor of a
/// configuration that states none, or `None` when it must state one. Every other recipe is
/// [`RotaryConfig::from_hf`]'s.
pub fn rope(
    values: &Fields<'_>,
    head_dim: u32,
    max_position_embeddings: u32,
    partial_rotary_factor: Option<f64>,
) -> Result<RotaryConfig> {
    let recipe = match values.optional_object("rope_parameters")? {
        Some(parameters) => Some(parameters),
        None => values.optional_object("rope_scaling")?,
    };
    let Some(recipe) = recipe.filter(|recipe| recipe.contains("mrope_section")) else {
        let config = RotaryConfig::from_hf(values, head_dim, max_position_embeddings)?;

        return Ok(match config.spectrum {
            Spectrum::Default => RotaryConfig {
                precision: PRECISION,
                ..config
            },
            _ => config,
        });
    };

    let read = |key: &str| -> Result<Option<f64>> {
        match recipe.optional_number(key)? {
            Some(value) => Ok(Some(value)),
            None => Ok(values.optional_number(key)?),
        }
    };
    let stated = |key: &str, value: Option<f64>| {
        value.ok_or_else(|| Error::Config(format!("an M-RoPE recipe states no {key}")))
    };

    if recipe.optional_text("rope_type")?.unwrap_or("default") != "default"
        || !recipe.flag("mrope_interleaved")?
    {
        return Err(Error::Config(
            "an M-RoPE recipe must be the default recipe with interleaved sections".to_owned(),
        ));
    }

    let factor = stated(
        "partial_rotary_factor",
        read("partial_rotary_factor")?.or(partial_rotary_factor),
    )?;
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

    let theta = stated("rope_theta", read("rope_theta")?)?;
    if !(theta.is_finite() && theta > 0.0) {
        return Err(Error::Config(format!("a rope_theta of {theta}")));
    }
    sectioned(
        theta as f32,
        rotated as u32,
        &recipe.indices("mrope_section")?,
    )
}

/// Interleaved M-RoPE over `dim` rotated channels, `sections` counting frequencies per axis.
fn sectioned(theta: f32, dim: u32, sections: &[u32]) -> Result<RotaryConfig> {
    if sections.len() != POSITION_AXES as usize || sections.iter().sum::<u32>() != dim / 2 {
        return Err(Error::Config(format!(
            "M-RoPE sections {sections:?} must split the {} frequencies of {dim} rotated \
             channels over {POSITION_AXES} axes",
            dim / 2
        )));
    }

    Ok(RotaryConfig {
        theta,
        dim,
        spectrum: Spectrum::sectioned(sections)?,
        precision: PRECISION,
    })
}

/// Which layers of a hybrid stack of `layers` layers are linear, from `layer_types` or
/// `full_attention_interval`, which must agree when both are stated.
pub fn linear_layers(values: &Fields<'_>, layers: u32) -> Result<Vec<bool>> {
    let derived = values
        .optional_count("full_attention_interval")?
        .map(|interval| {
            (0..layers)
                .map(|layer| !(layer + 1).is_multiple_of(interval))
                .collect::<Vec<_>>()
        });
    let stated = values
        .optional_texts("layer_types")?
        .map(|kinds| {
            kinds
                .iter()
                .map(|kind| match *kind {
                    "full_attention" => Ok(false),
                    "linear_attention" => Ok(true),
                    other => Err(Error::Config(format!(
                        "layer type {other} is neither full_attention nor linear_attention"
                    ))),
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;

    let linear = match (stated, derived) {
        (Some(stated), Some(derived)) if stated != derived => {
            return Err(Error::Config(
                "layer_types disagrees with full_attention_interval".to_owned(),
            ));
        }
        (Some(linear), _) | (None, Some(linear)) => linear,
        (None, None) => {
            return Err(Error::Config(
                "a hybrid stack states neither layer_types nor full_attention_interval".to_owned(),
            ));
        }
    };
    if linear.len() != layers as usize {
        return Err(Error::Config(format!(
            "{} layer types for {layers} layers",
            linear.len()
        )));
    }
    Ok(linear)
}

/// Released configurations, as this module's unit tests read them.
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
        assert_eq!(
            config.text.rope,
            RotaryConfig {
                theta: 1e6,
                dim: 128,
                spectrum: Spectrum::Default,
                precision: Precision::F32,
            }
        );
    }

    #[test]
    fn a_recipe_under_rope_parameters_states_the_base() {
        // Transformers 5 writes the base inside `rope_parameters`.
        let mut value = moe();
        value.as_object_mut().expect("object").remove("rope_theta");
        value["rope_parameters"] = serde_json::json!({"rope_theta": 5e5, "rope_type": "default"});

        let rope = parse(&value).expect("parsed").text.rope;
        assert_eq!((rope.theta, rope.precision), (5e5, Precision::F32));
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
    fn another_class_sliding_windows_and_layer_scales_are_refused() {
        let mut class = moe();
        class["architectures"] = serde_json::json!(["Qwen3ForCausalLM"]);
        assert!(parse(&class).is_err());

        let mut window = moe();
        window["use_sliding_window"] = serde_json::json!(true);
        assert!(parse(&window).is_err());

        let mut scale = hybrid();
        scale["text_config"]["layer_scale"] = serde_json::json!(true);
        assert!(parse(&scale).is_err());
    }

    #[test]
    fn conflicting_expert_aliases_are_refused() {
        let mut value = moe();
        value["num_local_experts"] = serde_json::json!(64);
        assert!(parse(&value).is_err());

        value["num_local_experts"] = serde_json::json!(128);
        let experts = parse(&value).expect("agreeing aliases").moe;
        assert_eq!(experts.expect("experts").num_experts, 128);
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

    /// A pipeline stage holds its contiguous share of the decoder layers; a pipeline that does
    /// not split the layers evenly, a sequence split that does not share the heads evenly, and
    /// two splits of one placement are refused.
    #[test]
    fn a_placement_holds_one_split_that_divides_what_it_splits() {
        let config = parse(&moe()).expect("parsed");
        let layers = config.text.num_hidden_layers;
        let split = |size, rank| Split { size, rank };
        let placed = |tensor, pipeline, sequence| TextPlacement {
            tensor,
            pipeline,
            sequence: nn::family::SequenceSplit {
                split: sequence,
                projection: nn::family::Projection::Gather,
            },
            experts: Split::WHOLE,
            tensor_experts: false,
            drafter: false,
            draft_tokens: 0,
        };
        let whole = Split::WHOLE;

        assert_eq!(
            config
                .check_placement(&placed(whole, whole, whole))
                .expect("whole"),
            0..layers
        );
        assert_eq!(
            config
                .check_placement(&placed(whole, split(2, 1), whole))
                .expect("the second stage"),
            layers / 2..layers
        );
        assert_eq!(
            config
                .check_placement(&placed(whole, whole, split(4, 3)))
                .expect("a sequence member"),
            0..layers
        );

        assert!(
            config
                .check_placement(&placed(whole, split(layers + 1, 0), whole))
                .is_err()
        );
        assert!(
            config
                .check_placement(&placed(whole, whole, split(3, 0)))
                .is_err()
        );
        assert!(
            config
                .check_placement(&placed(split(2, 0), split(2, 0), whole))
                .is_err()
        );
        assert!(
            config
                .check_placement(&placed(whole, whole, split(2, 2)))
                .is_err()
        );

        // A tensor group of four holding whole experts holds every layer; an expert group beside
        // it is refused.
        let holding = TextPlacement {
            tensor_experts: true,
            ..placed(split(4, 1), whole, whole)
        };
        assert_eq!(
            config
                .check_placement(&holding)
                .expect("a tensor group holding whole experts"),
            0..layers
        );
        let error = config
            .check_placement(&TextPlacement {
                experts: split(2, 0),
                ..holding
            })
            .expect_err("two groups dividing the experts")
            .to_string();
        assert!(error.contains("whole experts"), "{error}");

        // A hybrid stack of three linear layers and one attention layer: two stages would hold
        // zero and one attention layers, and its recurrent scan is not split over members.
        let hybrid = parse(&hybrid()).expect("parsed");
        assert!(
            hybrid
                .check_placement(&placed(whole, split(2, 0), whole))
                .is_err()
        );
        assert!(
            hybrid
                .check_placement(&placed(whole, whole, split(2, 0)))
                .is_err()
        );
    }

    #[test]
    fn a_vl_checkpoint_reads_its_tower_and_an_interleaved_mrope_spectrum() {
        let config = parse(&vl()).expect("parsed");
        let vision = config.vision.as_ref().expect("a tower");

        assert_eq!(vision.depth, 27);
        assert_eq!(vision.deepstack_visual_indexes, [8, 16, 24]);
        assert_eq!(vision.head_dim(), 72);
        assert_eq!(vision.position_side(), 48);
        assert_eq!(
            config.text.rope.spectrum,
            Spectrum::sectioned(&[24, 20, 20]).expect("three axes")
        );
        assert!(config.moe.is_none());
        assert_eq!(config.attention_layers(), 64);
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

        assert_eq!(config.text.linear_layers, [true, true, true, false]);
        let linear = config.text.gated_delta_net.expect("a gated delta network");
        assert_eq!(linear.value_heads, 32);
        assert_eq!((config.attention_layers(), config.linear_layers()), (1, 3));
        assert_eq!((config.ordinal(2), config.ordinal(3)), (2, 0));
        assert!(config.text.rms_norm_centred);
        assert!(config.text.attn_output_gate);
        assert_eq!(config.mtp_num_hidden_layers, 1);

        let moe = config.moe.expect("experts");
        assert_eq!(moe.shared_expert_intermediate_size, Some(512));
        assert!(moe.norm_topk_prob);

        // A quarter of the 256 channels of a head rotate, over three axes.
        assert_eq!(config.text.rope.dim, 64);
        assert_eq!(
            config.text.rope.spectrum,
            Spectrum::sectioned(&[11, 11, 10]).expect("three axes")
        );
    }

    #[test]
    fn a_hybrid_recipe_must_state_its_partial_rotation() {
        let mut value = hybrid();
        let recipe = value["text_config"]["rope_parameters"]
            .as_object_mut()
            .expect("object");
        recipe.remove("partial_rotary_factor");

        assert!(parse(&value).is_err());
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

    #[test]
    fn a_processor_states_three_positive_deviations() {
        let fields =
            serde_json::json!({"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.0, 0.5]});
        let error =
            pixel_norm(&Fields::of("preprocessor_config.json", &fields).expect("an object"))
                .expect_err("a zero deviation is refused");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("image_std")),
            "{error}"
        );

        let fields =
            serde_json::json!({"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]});
        let norm = pixel_norm(&Fields::of("preprocessor_config.json", &fields).expect("an object"))
            .expect("three positive deviations");
        assert_eq!((norm.mean, norm.std), ([0.5; 3], [0.5; 3]));
    }
}
