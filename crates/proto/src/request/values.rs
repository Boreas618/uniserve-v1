//! Identities and value types of the request vocabulary.

use serde::{Deserialize, Serialize};

/// Request identity: the daemon names each request it submits, and the
/// engine schedules it under that name.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
pub struct RequestId(pub u64);

/// One ranked vocabulary candidate at a generated or prompt token position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    /// Vocabulary token identity.
    pub token_id: u32,
    /// Natural-log probability assigned to the token.
    pub logprob: f32,
    /// One-based competition rank at the position; tied log probabilities
    /// share a rank.
    pub rank: u32,
}

/// Effective diffusion controls, resolved once by model preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionSamplingParams {
    /// Number of output frames after model-specific alignment.
    pub num_frames: u32,
    /// Video media units the decoder reconstructs, independent of GPU rank count.
    pub video_units: u32,
    /// Deterministic request-level noise seed.
    pub seed: u64,
    /// Output raster width in pixels: the request's canvas.
    pub width: u32,
    /// Output raster height in pixels: the request's canvas.
    pub height: u32,
}

/// The encoder that adds an image to a request's context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageEncoder {
    /// The vision encoder, whose features enter the context as attention
    /// blocks of a context prefill.
    Vision,
    /// The latent encoder, whose image latents a request ingests by calls of
    /// their own.
    Latent,
}

impl ImageEncoder {
    /// The name of the encoding call the encoder runs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Vision => "vision_encoding",
            Self::Latent => "latent_encoding",
        }
    }
}
