//! The serving layer's request vocabulary.
//!
//! A [`GenerateRequest`] is one generation request before tokenization: its
//! chat conversation, what it generates, the engine's sampling parameters (lowered
//! once from the wire request, the model's defaults applied), its length and
//! stop controls, and its scheduling and cache controls. The API routes lower
//! their wire requests into it; `InputProcessor::preprocess` splits it into
//! the engine's `GenerationRequest`, which moves into the engine, and its
//! `OutputParams` (`crate::output`), which the daemon keeps to decode and
//! parse the output.

use proto::request::SamplingParams;

use crate::input::chat::ChatRequest;
use crate::output::text::TextDecodeOptions;

/// One generation request before tokenization.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    /// Names the request in its errors and logs.
    pub request_id: String,
    /// The chat conversation; its input images are `image_url` parts.
    pub prompt: ChatRequest,
    /// Whether the request asks for image output, which no served model
    /// generates: feature validation refuses it.
    pub image_output: bool,
    /// How the engine samples text: the request's controls over the model's
    /// defaults, validated.
    pub sampling: SamplingParams,
    /// The most tokens the request generates; the model's default, then the
    /// context left after the prompt, bound it when `None`.
    pub max_tokens: Option<u32>,
    /// Token IDs that end generation, besides the model's EOS.
    pub stop_token_ids: Vec<u32>,
    /// How generated text is decoded: stop strings, special tokens and the
    /// tokens before a stop condition may end generation.
    pub decode: TextDecodeOptions,
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
