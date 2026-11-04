//! The Qwen3-VL text encoder's configuration: the language layers H3 conditions on, the vision
//! tower that encodes condition media, and the pixel normalization the processor applies.
//!
//! H3 conditions on the hidden state after the first [`RETAINED_LAYERS`] decoder layers of a
//! deeper checkpoint; the depth is not in the checkpoint's configuration, so it is stated here.
//! The language layers rotate by interleaved M-RoPE, whose three section widths the checkpoint
//! states.

use serde_json::{Map, Value};

use super::field;
use crate::{Error, Result};

/// The decoder layers whose hidden state H3 conditions on.
///
/// The text encoder checkpoint is a 64-layer Qwen3-VL; H3 reads the state after layer 50. The
/// checkpoint's configuration does not state how many layers a consumer keeps.
pub const RETAINED_LAYERS: u32 = 50;

/// The activation of the language layers. H3's checkpoint states SiLU.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Activation {
    /// `x * sigmoid(x)`.
    Silu,
}

/// How the checkpoint's processor normalizes a pixel before the vision tower patches it.
///
/// Each RGB channel maps `x` in `[0, 255]` to `(x / 255 - mean) / std`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PixelConfig {
    /// Per-channel means.
    pub mean: [f32; 3],
    /// Per-channel standard deviations.
    pub std: [f32; 3],
}

impl PixelConfig {
    /// Read a processor's `image_mean` and `image_std`.
    pub fn parse(values: &Map<String, Value>, owner: &str) -> Result<Self> {
        let mean = field::fixed_numbers(values, "image_mean", owner, 3)?;
        let std = field::fixed_numbers(values, "image_std", owner, 3)?;
        let mean: [f32; 3] = [mean[0] as f32, mean[1] as f32, mean[2] as f32];
        let std: [f32; 3] = [std[0] as f32, std[1] as f32, std[2] as f32];
        if std.iter().any(|value| *value <= 0.0) {
            return Err(Error::config(format!(
                "{owner}.image_std must be positive, got {std:?}"
            )));
        }
        Ok(Self { mean, std })
    }
}

/// The activation of the vision tower. H3's checkpoint states PyTorch's tanh-approximated GELU.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VisionActivation {
    /// `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`.
    GeluTanh,
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
    /// The tower's activation.
    pub hidden_act: VisionActivation,
    /// Pixel channels of one tubelet voxel.
    pub in_channels: u32,
    /// Side of one square spatial patch, in pixels.
    pub patch_size: u32,
    /// Frames of one temporal patch.
    pub temporal_patch_size: u32,
    /// Side, in patches, of one merges into a language token.
    pub spatial_merge_size: u32,
    /// Width of a merged token: the language model's.
    pub out_hidden_size: u32,
    /// Entries of the learned position table, a square grid.
    pub num_position_embeddings: u32,
    /// The index of each block whose output feeds a DeepStack merger, in increasing order.
    pub deepstack_visual_indexes: Vec<u32>,
}

impl VisionConfig {
    /// Read a text encoder checkpoint's `vision_config`.
    pub fn parse(values: &Map<String, Value>, owner: &str) -> Result<Self> {
        let hidden_act = match field::text(values, "hidden_act", owner)? {
            "gelu_pytorch_tanh" => VisionActivation::GeluTanh,
            other => {
                return Err(Error::config(format!(
                    "{owner}.hidden_act must be gelu_pytorch_tanh, got {other:?}"
                )));
            }
        };
        let depth = field::count(values, "depth", owner)?;
        // The DeepStack mergers join the language model after the blocks their indexes name, so an
        // index outside the tower, or two out of order, is refused here.
        let deepstack_visual_indexes = field::indices(values, "deepstack_visual_indexes", owner)?;
        if deepstack_visual_indexes
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(Error::config(format!(
                "{owner}.deepstack_visual_indexes must increase strictly"
            )));
        }
        if let Some(outside) = deepstack_visual_indexes
            .iter()
            .find(|index| **index >= depth - 1)
        {
            return Err(Error::config(format!(
                "{owner}.deepstack_visual_indexes names block {outside}, outside the {depth}-block \
                 tower"
            )));
        }
        let config = Self {
            depth,
            hidden_size: field::count(values, "hidden_size", owner)?,
            intermediate_size: field::count(values, "intermediate_size", owner)?,
            num_heads: field::count(values, "num_heads", owner)?,
            hidden_act,
            in_channels: field::count(values, "in_channels", owner)?,
            patch_size: field::count(values, "patch_size", owner)?,
            temporal_patch_size: field::count(values, "temporal_patch_size", owner)?,
            spatial_merge_size: field::count(values, "spatial_merge_size", owner)?,
            out_hidden_size: field::count(values, "out_hidden_size", owner)?,
            num_position_embeddings: field::count(values, "num_position_embeddings", owner)?,
            deepstack_visual_indexes,
        };
        if !config.hidden_size.is_multiple_of(config.num_heads) {
            return Err(Error::config(format!(
                "Qwen3-VL vision heads {} do not divide the hidden size {}",
                config.num_heads, config.hidden_size
            )));
        }
        // The 2D rotary embedding rotates half of each head by the patch row and half by its
        // column, each as split-half pairs.
        if !(config.hidden_size / config.num_heads).is_multiple_of(4) {
            return Err(Error::config(format!(
                "Qwen3-VL vision head width {} must split into two even rotary halves",
                config.hidden_size / config.num_heads
            )));
        }
        let side = f64::from(config.num_position_embeddings).sqrt().round();
        if (side * side) as u32 != config.num_position_embeddings {
            return Err(Error::config(
                "Qwen3-VL vision position table must be a square grid".to_owned(),
            ));
        }
        Ok(config)
    }
}

/// The Qwen3-VL text encoder's configuration.
#[derive(Clone, PartialEq, Debug)]
pub struct TextEncoderConfig {
    /// Vocabulary entries.
    pub vocab_size: u32,
    /// The language model's width.
    pub hidden_size: u32,
    /// The language model's feed-forward width.
    pub intermediate_size: u32,
    /// Decoder layers the checkpoint holds.
    pub num_checkpoint_layers: u32,
    /// Decoder layers whose hidden state H3 conditions on.
    pub num_retained_layers: u32,
    /// Query heads.
    pub num_attention_heads: u32,
    /// Key/value heads, which group-query attention shares.
    pub num_key_value_heads: u32,
    /// Width of one head.
    pub head_dim: u32,
    /// The rotary base.
    pub rope_theta: f32,
    /// The normalization epsilon.
    pub rms_norm_eps: f32,
    /// Positions the checkpoint's rotary embedding is built for.
    pub max_position_embeddings: u32,
    /// The M-RoPE section widths of the temporal, height and width axes.
    pub mrope_sections: [u32; 3],
    /// The token that stands for one merged image patch.
    pub image_token_id: u32,
    /// The token that stands for one merged video patch.
    pub video_token_id: u32,
    /// The vision tower.
    pub vision: VisionConfig,
    /// The pixel normalization the processor applies.
    pub pixels: PixelConfig,
}

impl TextEncoderConfig {
    /// Read `text_encoder/config.json` and the processor's pixel normalization.
    ///
    /// `metadata` is the whole text encoder configuration: its `text_config` holds the language
    /// layers, its `vision_config` the tower, and its image and video token ids the placeholders a
    /// prompt's media are presented by.
    pub fn parse(metadata: &Value, processor: &Value) -> Result<Self> {
        let values = field::object(metadata, "MiniMax-H3 text_encoder/config.json")?;
        let text = field::nested(values, "text_config", "MiniMax-H3 text encoder")?;
        // The language layers are the Qwen3-VL dense text model H3 was trained against: SiLU, no
        // attention bias, an independent vocabulary head, and interleaved M-RoPE.
        if field::text(text, "hidden_act", "MiniMax-H3 text encoder")? != "silu" {
            return Err(Error::config(
                "unsupported MiniMax-H3 text encoder hidden_act".to_owned(),
            ));
        }
        if field::flag(text, "attention_bias", "MiniMax-H3 text encoder")? {
            return Err(Error::config(
                "unsupported MiniMax-H3 text encoder: it states an attention bias".to_owned(),
            ));
        }
        if field::flag(values, "tie_word_embeddings", "MiniMax-H3 text encoder")? {
            return Err(Error::config(
                "MiniMax-H3 checkpoint requires an independent vocabulary head".to_owned(),
            ));
        }
        let rope = field::nested(text, "rope_scaling", "MiniMax-H3 text encoder")?;
        if field::text(rope, "rope_type", "MiniMax-H3 text encoder rope_scaling")? != "default"
            || !field::flag(
                rope,
                "mrope_interleaved",
                "MiniMax-H3 text encoder rope_scaling",
            )?
        {
            return Err(Error::config(
                "unsupported MiniMax-H3 text encoder rope_scaling: the language layers rotate by \
                 interleaved M-RoPE"
                    .to_owned(),
            ));
        }
        let sections = field::indices(
            rope,
            "mrope_section",
            "MiniMax-H3 text encoder rope_scaling",
        )?;
        if sections.len() != 3 {
            return Err(Error::config(format!(
                "MiniMax-H3 text encoder M-RoPE sections must split half the head over three \
                 axes, got {} sections",
                sections.len()
            )));
        }
        let vision = field::nested(values, "vision_config", "MiniMax-H3 text encoder")?;
        let vision = VisionConfig::parse(vision, "MiniMax-H3 vision_config")?;
        let config = Self {
            vocab_size: field::count(text, "vocab_size", "MiniMax-H3 text encoder")?,
            hidden_size: field::count(text, "hidden_size", "MiniMax-H3 text encoder")?,
            intermediate_size: field::count(text, "intermediate_size", "MiniMax-H3 text encoder")?,
            num_checkpoint_layers: field::count(
                text,
                "num_hidden_layers",
                "MiniMax-H3 text encoder",
            )?,
            num_retained_layers: RETAINED_LAYERS,
            num_attention_heads: field::count(
                text,
                "num_attention_heads",
                "MiniMax-H3 text encoder",
            )?,
            num_key_value_heads: field::count(
                text,
                "num_key_value_heads",
                "MiniMax-H3 text encoder",
            )?,
            head_dim: field::count(text, "head_dim", "MiniMax-H3 text encoder")?,
            rope_theta: field::positive(text, "rope_theta", "MiniMax-H3 text encoder")? as f32,
            rms_norm_eps: field::positive(text, "rms_norm_eps", "MiniMax-H3 text encoder")? as f32,
            max_position_embeddings: field::count(
                text,
                "max_position_embeddings",
                "MiniMax-H3 text encoder",
            )?,
            mrope_sections: [sections[0], sections[1], sections[2]],
            image_token_id: field::index(values, "image_token_id", "MiniMax-H3 text_encoder")?,
            video_token_id: field::index(values, "video_token_id", "MiniMax-H3 text_encoder")?,
            vision,
            pixels: PixelConfig::parse(
                field::object(processor, "MiniMax-H3 processor/preprocessor_config.json")?,
                "MiniMax-H3 processor",
            )?,
        };
        config.validate()?;
        Ok(config)
    }

    /// Check the language layers' and the tower's dimensions against each other.
    fn validate(&self) -> Result<()> {
        if self.num_retained_layers == 0 || self.num_retained_layers > self.num_checkpoint_layers {
            return Err(Error::config(format!(
                "H3 keeps {} of the text encoder's {} layers",
                self.num_retained_layers, self.num_checkpoint_layers
            )));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
            || !self.head_dim.is_multiple_of(2)
        {
            return Err(Error::config(format!(
                "H3 text encoder needs divisible group-query attention and even rotary \
                 dimensions: {} heads, {} key/value heads, head width {}",
                self.num_attention_heads, self.num_key_value_heads, self.head_dim
            )));
        }
        // M-RoPE splits half of each head over the three axes; the other half stays unrotated.
        if self.mrope_sections.iter().sum::<u32>() != self.head_dim / 2 {
            return Err(Error::config(format!(
                "H3 text encoder M-RoPE sections {:?} must split half of the {} channel head",
                self.mrope_sections, self.head_dim
            )));
        }
        for token in [self.image_token_id, self.video_token_id] {
            if token >= self.vocab_size {
                return Err(Error::config(format!(
                    "H3 vision placeholder {token} is outside the {} entry vocabulary",
                    self.vocab_size
                )));
            }
        }
        // The tower's merged tokens must be the language model's width.
        if self.vision.out_hidden_size != self.hidden_size {
            return Err(Error::config(format!(
                "H3 vision tokens are {} wide and the language model {}",
                self.vision.out_hidden_size, self.hidden_size
            )));
        }
        Ok(())
    }

    /// The projection width of one query head times the head count.
    pub const fn query_width(&self) -> u32 {
        self.num_attention_heads * self.head_dim
    }

    /// The width of the key and value projection of one rank.
    pub const fn key_value_width(&self) -> u32 {
        self.num_key_value_heads * self.head_dim
    }

    /// The width of one M-RoPE axis's rotation: `2 * sections[axis]` channels of each head.
    pub fn mrope_axis_width(&self, axis: usize) -> u32 {
        self.mrope_sections[axis] * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released text encoder checkpoint's configuration, trimmed to the fields H3 reads.
    fn metadata() -> Value {
        serde_json::from_str(
            r#"{
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
            }"#,
        )
        .expect("the fixture is JSON")
    }

    fn processor() -> Value {
        serde_json::from_str(r#"{"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}"#)
            .expect("the fixture is JSON")
    }

    #[test]
    fn the_released_text_encoder_reads_its_dimensions() {
        let config = TextEncoderConfig::parse(&metadata(), &processor()).expect("parsed");
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.num_checkpoint_layers, 64);
        assert_eq!(config.num_retained_layers, RETAINED_LAYERS);
        assert_eq!(config.mrope_sections, [24, 20, 20]);
        assert_eq!(config.mrope_axis_width(0), 48);
        assert_eq!(config.query_width(), 8192);
        assert_eq!(config.key_value_width(), 1024);
        assert_eq!(config.vision.depth, 27);
        assert_eq!(config.vision.deepstack_visual_indexes, vec![8, 16, 24]);
        assert_eq!(config.pixels.mean, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn a_text_encoder_refuses_mrope_sections_that_do_not_split_a_head() {
        let mut values = metadata();
        values["text_config"]["rope_scaling"]["mrope_section"] = serde_json::json!([24, 20, 19]);
        let error = TextEncoderConfig::parse(&values, &processor()).expect_err("sections");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("M-RoPE")),
            "{error}"
        );
    }

    #[test]
    fn a_text_encoder_refuses_a_tied_head_and_a_rope_that_is_not_interleaved_mrope() {
        let mut tied = metadata();
        tied["tie_word_embeddings"] = serde_json::json!(true);
        assert!(TextEncoderConfig::parse(&tied, &processor()).is_err());

        let mut plain = metadata();
        plain["text_config"]["rope_scaling"] = serde_json::json!({"rope_type": "default"});
        assert!(TextEncoderConfig::parse(&plain, &processor()).is_err());
    }

    #[test]
    fn a_vision_tower_refuses_a_position_table_that_is_not_square_and_blocks_outside_it() {
        let mut table = metadata();
        table["vision_config"]["num_position_embeddings"] = serde_json::json!(2305);
        assert!(TextEncoderConfig::parse(&table, &processor()).is_err());

        let mut blocks = metadata();
        blocks["vision_config"]["deepstack_visual_indexes"] = serde_json::json!([8, 16, 27]);
        assert!(TextEncoderConfig::parse(&blocks, &processor()).is_err());
    }

    #[test]
    fn a_processor_states_three_positive_deviations() {
        let mut negative = processor();
        negative["image_std"] = serde_json::json!([0.5, 0.0, 0.5]);
        let error = PixelConfig::parse(
            negative.as_object().expect("an object"),
            "MiniMax-H3 processor",
        )
        .expect_err("a zero deviation is refused");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("image_std")),
            "{error}"
        );
    }
}
