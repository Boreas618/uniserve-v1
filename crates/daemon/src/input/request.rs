//! The serving layer's request vocabulary.
//!
//! A [`GenerateRequest`] is one generation request before tokenization, as
//! vLLM's text and chat requests are: its [`Prompt`], what it generates, its
//! northbound [`SamplingParams`], and its scheduling and cache controls. The
//! API routes lower their wire requests into it; `InputProcessor::preprocess`
//! splits it into the engine's `GenerationRequest`, which moves into the
//! engine, and its `OutputParams` (`crate::output`), which the daemon keeps
//! to decode and parse the output.

use std::collections::HashMap;

use proto::request::GenerationConstraint;

use crate::input::chat::ChatRequest;
use crate::output::text::TextDecodeOptions;

/// The prompt of a generation request.
#[derive(Debug, Clone, PartialEq)]
pub enum Prompt {
    /// The text prompt of an image-generation request.
    Text {
        /// The prompt.
        prompt: String,
        /// The prompt classifier-free guidance steers the image away from.
        negative_prompt: Option<String>,
    },
    /// A chat conversation; its input images are `image_url` parts.
    Chat(ChatRequest),
}

/// The northbound sampling parameters of a generation request, as vLLM's
/// `SamplingParams`. `None` leaves a value to the model's defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// Sampling temperature; zero samples greedily.
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability mass.
    pub top_p: Option<f32>,
    /// Highest-probability candidates considered per step.
    pub top_k: Option<u32>,
    /// Minimum token probability relative to the most likely candidate.
    pub min_p: Option<f32>,
    /// Deterministic sampler seed; its bit pattern seeds the engine.
    pub seed: Option<i64>,
    /// Maximum generated tokens.
    pub max_tokens: Option<u32>,
    /// Generated tokens before a stop condition may end generation.
    pub min_tokens: Option<u32>,
    /// Penalty proportional to a token's prior frequency.
    pub frequency_penalty: Option<f32>,
    /// Penalty applied once to tokens that already appear.
    pub presence_penalty: Option<f32>,
    /// Multiplicative repetition penalty.
    pub repetition_penalty: Option<f32>,
    /// Whether end-of-sequence tokens are ignored as stop conditions.
    pub ignore_eos: bool,
    /// Decoded strings that end generation.
    pub stop: Vec<String>,
    /// Token IDs that end generation, besides the model's EOS.
    pub stop_token_ids: Vec<u32>,
    /// Words the sampler may not generate.
    pub bad_words: Vec<String>,
    /// The only token IDs the sampler may generate.
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Additive logit adjustments keyed by token ID.
    pub logit_bias: Option<HashMap<u32, f32>>,
    /// Candidates reported per generated token; `-1` reports every token.
    pub logprobs: Option<i32>,
    /// Candidates reported per prompt token; `-1` reports every token.
    pub prompt_logprobs: Option<i32>,
    /// Whether special tokens are omitted from decoded text.
    pub skip_special_tokens: bool,
    /// Whether a matched stop string remains in the output.
    pub include_stop_str_in_output: bool,
}

impl Default for SamplingParams {
    /// Leaves every value to the model's defaults, skips special tokens, and
    /// excludes a matched stop string from the output.
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            seed: None,
            max_tokens: None,
            min_tokens: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            ignore_eos: false,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            bad_words: Vec::new(),
            allowed_token_ids: None,
            logit_bias: None,
            logprobs: None,
            prompt_logprobs: None,
            skip_special_tokens: true,
            include_stop_str_in_output: false,
        }
    }
}

impl SamplingParams {
    /// The decoder's options for these parameters.
    pub fn decode_options(&self) -> TextDecodeOptions {
        TextDecodeOptions {
            skip_special_tokens: self.skip_special_tokens,
            include_stop_str_in_output: self.include_stop_str_in_output,
            stop_strings: (!self.stop.is_empty()).then(|| self.stop.clone()),
            min_tokens: self.min_tokens.unwrap_or(0),
        }
    }
}

/// One generation request before tokenization.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    /// Names the request in its errors and logs.
    pub request_id: String,
    /// The prompt.
    pub prompt: Prompt,
    /// What the request generates: text (`UndOnly`), an image (`GenOnly`),
    /// or both (`Default`).
    pub constraint: GenerationConstraint,
    /// How text is sampled and decoded.
    pub sampling_params: SamplingParams,
    /// How images are generated, for a request that generates them.
    pub image_params: Option<ImageSamplingParams>,
    /// Scheduling priority; lower is earlier.
    pub priority: i32,
    /// Salt that isolates the request's prefix-cache and encoder-cache
    /// entries from other salts.
    pub cache_salt: Option<String>,
    /// The LoRA adapter the request computes with; `None` for the base
    /// model alone.
    pub adapter: Option<AdapterSelection>,
}

/// The LoRA adapter a request computes with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterSelection {
    /// The slot of the workers' adapter pool it lies in.
    pub slot: u32,
    /// What keeps its cache entries apart from every other adapter's, one
    /// loaded into the same slot before included.
    pub identity: String,
}

/// Image-generation controls: dimensions, denoise steps, guidance controls,
/// seed, and image-count bounds.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ImageSamplingParams {
    /// Named output-resolution preset.
    pub resolution: Option<models::omni::resolution::ResolutionName>,
    /// Explicit output width in pixels.
    pub width: Option<u32>,
    /// Explicit output height in pixels.
    pub height: Option<u32>,
    /// Diffusion step count.
    pub steps: Option<u16>,
    /// Classifier-free guidance scale for text conditioning.
    pub cfg_text_scale: Option<f32>,
    /// Classifier-free guidance scale for image conditioning.
    pub cfg_img_scale: Option<f32>,
    /// Fractional diffusion interval over which guidance applies.
    pub cfg_interval: Option<[f32; 2]>,
    /// Guidance renormalization algorithm.
    pub cfg_renorm_type: Option<proto::request::CfgRenorm>,
    /// Minimum scale at which guidance renormalization applies.
    pub cfg_renorm_min: Option<f32>,
    /// Diffusion timestep shift.
    pub timestep_shift: Option<f32>,
    /// Optional deterministic seed. For SenseNova and Bagel requests it
    /// replaces `SamplingParams::seed` as the sampler seed; Qwen3
    /// preprocessing resolves its seed from `SamplingParams::seed` alone.
    pub seed: Option<u64>,
    /// Maximum number of generated images.
    pub max_images: Option<u16>,
}
