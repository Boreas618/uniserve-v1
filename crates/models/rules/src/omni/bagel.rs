//! Bagel model parameters, prompt layout, and image configuration.
//!
//! Input images feed both the VAE and the ViT encoder, in that order;
//! generated images feed back through the VAE only. The resize policies here
//! fix the values the worker's Bagel `image_processor`
//! (`uniserve/models/bagel/processing.py`) uses with the default vision
//! `image_size` of 980 and `patch_size` of 14.

use proto::request::{
    FeedbackNextToken, FeedbackSource, GenerationConstraint, GenerationLimits, ImageEncoderInput,
    ImageGenerationConfig, ImageTrigger, ModelDtype,
};
use serde::{Deserialize, Serialize};

use super::resolution::{ResolutionBucket, ResolutionName, ResolutionPolicy};
use super::{
    GenerationControls, ImageGenerationDefaults, OmniProfile, OutputFilterPolicy, encode,
    model_dtype_bytes, required_token, required_token_id, stride_resize_tokens, stride_resized,
};
use crate::{FactsError, Vocabulary, VocabularyError};

/// System instruction used under `GenerationConstraint::Default` when the
/// request supplies none.
const DEFAULT_SYSTEM_PROMPT: &str = "You should first think about the planning process in the mind and then generate the image. \n     The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Loaded Bagel profile and tokenizer assets.
pub struct BagelProfile {
    /// Special tokens that delimit multimodal generation.
    pub controls: GenerationControls,
    /// Default image-generation parameters.
    pub image_defaults: ImageGenerationDefaults,
    /// Supported image dimensions and default resolution.
    pub resolution_policy: ResolutionPolicy,
    /// Ordered encoders and their loaded cache requirements for input images.
    pub image_encoders: Vec<ImageEncoderInput>,
    /// Logical positions contributed by one input image, independently of cache tokens.
    pub image_num_positions: u32,
    /// Image trigger and feedback encoder requirements.
    pub image_generation: ImageGenerationConfig,
}

impl BagelProfile {
    /// Stable profile identifier.
    pub const ID: &'static str = "bagel";
    /// Pixel-to-latent downsampling factor.
    pub const LATENT_DOWNSAMPLE: u32 = 16;
    /// Elements of one latent unit: 16 VAE latent channels over a 2x2 latent
    /// patch.
    const LATENT_UNIT_ELEMENTS: u64 = 16 * 2 * 2;
    /// The resize every input or fed-back image goes through before the VAE,
    /// as `(max_side, min_side, stride, max_pixels)`; the stride is the
    /// VAE's `LATENT_DOWNSAMPLE`.
    const VAE_RESIZE: (u32, u32, u32, u64) = (1024, 512, 16, 1_806_336);
    /// The resize the ViT applies to the VAE canvas, as `VAE_RESIZE` is
    /// stated; its stride is the ViT patch size.
    const VIT_RESIZE: (u32, u32, u32, u64) = (980, 224, 14, 1_806_336);
    /// Start- and end-of-image tokens that frame an image's grid.
    const MARKER_TOKENS: u32 = 2;
    /// Primary checkpoint files in the order the worker's loader tries them
    /// (`uniserve/models/bagel/weights.py`); the first present one holds the
    /// latent position table.
    pub const PRIMARY_CHECKPOINT: [&'static str; 2] = ["ema.safetensors", "model.safetensors"];
    /// Learned latent position table, `[side * side, hidden]` for a square
    /// grid of `side` latent patches per axis; the side of that grid is the
    /// largest latent canvas side the profile serves ([`BagelProfile::resolve`]).
    /// `config.json`'s `max_latent_size` is not used: the published
    /// BAGEL-7B-MoT config declares 32 for its 64x64 table.
    pub const LATENT_POSITION_TABLE: &'static str = "latent_pos_embed.pos_embed";

    /// Returns the profile's runtime sizing for the selected dtype.
    ///
    /// The grid-token bounds are those of the largest image the resize
    /// policies produce: an image at the VAE's maximum side, and the ViT grid
    /// of that canvas. The served features, the latent capacity, the vision
    /// feature bytes and the encoder cache are not stated here: the
    /// checkpoint's latent grid bounds a canvas's sides
    /// (`ResolutionPolicy::max_side`), and the engine scheduler's
    /// `resolve_generation_limits` takes the features from the loaded
    /// workers' call kinds and the capacities from what they report.
    pub fn runtime_limits(model_dtype: ModelDtype) -> GenerationLimits {
        let dtype_bytes = model_dtype_bytes(model_dtype);
        let vae_tokens = Self::max_grid_tokens(Self::VAE_RESIZE);
        let vit_tokens = Self::max_grid_tokens(Self::VIT_RESIZE);
        let latent_bytes_per_unit = Self::LATENT_UNIT_ELEMENTS * dtype_bytes;
        GenerationLimits {
            max_latent_units: u64::MAX,
            latent_downsample: Self::LATENT_DOWNSAMPLE,
            max_vae_grid_tokens: vae_tokens,
            max_vit_grid_tokens: vit_tokens,
            latent_bytes_per_unit,
            max_latent_feature_bytes: u64::from(vae_tokens) * latent_bytes_per_unit,
            max_vision_feature_bytes: u64::MAX,
            commit_marker_tokens: Self::MARKER_TOKENS,
            max_cfg_branches: 3,
            encoder_cache_entries: u32::MAX,
            ..GenerationLimits::default()
        }
    }

    /// The most tokens, framing markers included, an image can have after a
    /// `(max_side, min_side, stride, max_pixels)` resize: the grid of a
    /// canvas at the longest side on both axes, held to the pixel bound.
    fn max_grid_tokens((max_side, _, stride, max_pixels): (u32, u32, u32, u64)) -> u32 {
        let cells_per_side = u64::from(max_side / stride);
        let grid = (cells_per_side * cells_per_side).min(max_pixels / u64::from(stride).pow(2));
        u32::try_from(grid)
            .unwrap_or(u32::MAX)
            .saturating_add(Self::MARKER_TOKENS)
    }

    /// Resolves required control tokens from the loaded tokenizer.
    ///
    /// Encoder token counts stay unset here; the `*_for_dimensions` methods
    /// return per-image copies with the counts set. `max_latent_size` is the
    /// checkpoint's latent position grid side (see
    /// [`Self::read_max_latent_size`]) and bounds each side of a generated
    /// canvas. Fails when the tokenizer lacks any of the ChatML or vision
    /// delimiter tokens.
    pub fn resolve(vocabulary: &dyn Vocabulary, max_latent_size: u32) -> crate::Result<Self> {
        let (start_of_image, start_of_image_text) =
            required_token(vocabulary, "<|vision_start|>", "Bagel start-of-image")?;
        let (end_of_image, end_of_image_text) =
            required_token(vocabulary, "<|vision_end|>", "Bagel end-of-image")?;
        let controls = GenerationControls {
            bos: required_token_id(vocabulary, "<|im_start|>", "Bagel beginning-of-sequence")?,
            eos: required_token_id(vocabulary, "<|im_end|>", "Bagel end-of-sequence")?,
            start_of_image,
            end_of_image,
            start_of_image_text,
            end_of_image_text,
        };
        let image_encoders = vec![
            ImageEncoderInput {
                encoder: proto::request::ImageEncoder::Latent,
                num_cache_tokens: None,
                max_cache_tokens: None,
                height: None,
                width: None,
            },
            ImageEncoderInput {
                encoder: proto::request::ImageEncoder::Vision,
                num_cache_tokens: None,
                max_cache_tokens: None,
                height: None,
                width: None,
            },
        ];
        let image_num_positions = 1;
        let image_generation = ImageGenerationConfig {
            trigger: ImageTrigger::Token {
                token_id: controls.start_of_image,
            },
            requires_text_for_image: false,
            feedback_source: Some(FeedbackSource::DeviceProduct),
            feedback_next_token: FeedbackNextToken::Token {
                token_id: controls.end_of_image,
            },
            num_feedback_positions: 2,
            feedback_encoders: vec![ImageEncoderInput {
                encoder: proto::request::ImageEncoder::Latent,
                num_cache_tokens: None,
                max_cache_tokens: None,
                height: None,
                width: None,
            }],
            sample_feedback_continuation: false,
        };
        let default_resolution = ResolutionBucket {
            name: ResolutionName::Square,
            width: 512,
            height: 512,
        };
        Ok(Self {
            controls,
            image_defaults: ImageGenerationDefaults {
                steps: 50,
                cfg_text_scale: 4.0,
                cfg_img_scale: 1.0,
                cfg_renorm_type: proto::request::CfgRenorm::Global,
                cfg_renorm_min: 0.0,
                cfg_interval: (0.0, 1.0),
                timestep_shift: 1.0,
                seed: None,
                max_images: 1,
            },
            // The denoiser reads a latent patch's position embedding from
            // table row `row * max_latent_size + col`, so a canvas wider than
            // the grid reads another row's embedding and a taller one indexes
            // past the table.
            resolution_policy: ResolutionPolicy {
                default: default_resolution.clone(),
                buckets: vec![default_resolution],
                allow_custom: true,
                max_side: Some(max_latent_size.saturating_mul(Self::LATENT_DOWNSAMPLE)),
            },
            image_encoders,
            image_num_positions,
            image_generation,
        })
    }

    /// Wraps plain context text in the profile's required delimiters.
    ///
    /// The result is `bos`, the encoded text, then `eos`, with no role line or
    /// newline.
    pub fn wrap_context_text(
        &self,
        vocabulary: &dyn Vocabulary,
        text: &str,
    ) -> std::result::Result<Vec<u32>, VocabularyError> {
        let mut ids = vec![self.controls.bos];
        ids.extend(encode(vocabulary, text)?);
        ids.push(self.controls.eos);
        Ok(ids)
    }
}

impl OmniProfile for BagelProfile {
    fn controls(&self) -> &GenerationControls {
        &self.controls
    }

    fn image_defaults(&self) -> &ImageGenerationDefaults {
        &self.image_defaults
    }

    fn resolution_policy(&self) -> &ResolutionPolicy {
        &self.resolution_policy
    }

    fn image_num_positions(&self) -> u32 {
        self.image_num_positions
    }

    /// Only `Default` has a system instruction.
    fn default_system_prompt(&self, constraint: GenerationConstraint) -> Option<&'static str> {
        matches!(constraint, GenerationConstraint::Default).then_some(DEFAULT_SYSTEM_PROMPT)
    }

    /// Every constraint opens a new assistant turn.
    fn assistant_prefix(&self, _constraint: GenerationConstraint) -> &'static str {
        ""
    }

    /// An image slot is removed, and the image enters after the text before
    /// it.
    fn image_slot(&self) -> (String, Option<u32>) {
        (String::new(), None)
    }

    /// - `Default` places the system instruction directly after
    ///   `<|im_start|>`, with no role line, followed by a user turn and an open
    ///   assistant turn.
    /// - `UndOnly` renders a plain user turn and an open assistant turn.
    /// - `GenOnly` wraps the prompt with [`BagelProfile::wrap_context_text`].
    fn render_prompt_ids(
        &self,
        vocabulary: &dyn Vocabulary,
        constraint: GenerationConstraint,
        prompt: &str,
    ) -> std::result::Result<Vec<u32>, VocabularyError> {
        match constraint {
            GenerationConstraint::Default => encode(
                vocabulary,
                &format!(
                    "<|im_start|>{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                ),
            ),
            GenerationConstraint::UndOnly => encode(
                vocabulary,
                &format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"),
            ),
            GenerationConstraint::GenOnly => self.wrap_context_text(vocabulary, prompt),
        }
    }

    /// An empty prompt yields no tokens rather than an empty framed turn.
    fn render_negative_prompt_ids(
        &self,
        vocabulary: &dyn Vocabulary,
        prompt: &str,
    ) -> std::result::Result<Vec<u32>, VocabularyError> {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }
        self.wrap_context_text(vocabulary, prompt)
    }

    /// The VAE grid comes from the VAE resize, whose size the latent block
    /// carries; the ViT grid comes from the ViT resize applied to that VAE
    /// canvas. Each adds two marker tokens. The image count does not affect
    /// the result.
    fn image_encoders_for_dimensions(
        &self,
        width: u32,
        height: u32,
        _image_count: usize,
    ) -> crate::Result<Vec<ImageEncoderInput>> {
        let vae_tokens = stride_resize_tokens(
            width,
            height,
            &[Self::VAE_RESIZE],
            Self::VAE_RESIZE.2,
            Self::MARKER_TOKENS,
        )?;
        let (vae_width, vae_height) = stride_resized(width, height, &[Self::VAE_RESIZE])?;
        let vit_tokens = stride_resize_tokens(
            width,
            height,
            &[Self::VAE_RESIZE, Self::VIT_RESIZE],
            Self::VIT_RESIZE.2,
            Self::MARKER_TOKENS,
        )?;
        Ok(vec![
            ImageEncoderInput {
                encoder: proto::request::ImageEncoder::Latent,
                num_cache_tokens: Some(vae_tokens),
                max_cache_tokens: None,
                height: Some(vae_height),
                width: Some(vae_width),
            },
            ImageEncoderInput {
                encoder: proto::request::ImageEncoder::Vision,
                num_cache_tokens: Some(vit_tokens),
                max_cache_tokens: None,
                height: None,
                width: None,
            },
        ])
    }

    fn input_image_cache_salt(&self, _image_count: usize) -> Option<u64> {
        None
    }

    /// Fails when the canvas cannot be resized under the VAE policy, or when
    /// the profile's feedback configuration has no feedback source or is not
    /// a single VAE encoder.
    fn image_generation_for_dimensions(
        &self,
        width: u32,
        height: u32,
    ) -> crate::Result<ImageGenerationConfig> {
        let tokens = stride_resize_tokens(
            width,
            height,
            &[Self::VAE_RESIZE],
            Self::VAE_RESIZE.2,
            Self::MARKER_TOKENS,
        )?;
        let mut policy = self.image_generation.clone();
        if policy.feedback_source.is_none()
            || policy.feedback_encoders.len() != 1
            || policy.feedback_encoders[0].encoder != proto::request::ImageEncoder::Latent
        {
            return Err(FactsError::invalid(
                "image feedback encoders do not match the model processor",
            ));
        }
        policy.feedback_encoders[0].num_cache_tokens = Some(tokens);
        policy.feedback_encoders[0].max_cache_tokens = None;
        Ok(policy)
    }

    fn output_filter(&self) -> Option<&OutputFilterPolicy> {
        None
    }
}
