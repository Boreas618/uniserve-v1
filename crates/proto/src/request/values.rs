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

/// Block-diffusion sampling of a deployment's canvases.
///
/// Text is generated in blocks of `canvas_length` tokens. A block starts as
/// uniformly random tokens and is denoised for at most `max_steps` steps: a
/// step samples every position from the model's logits at a temperature that
/// falls linearly from `t_max` to `t_min`, accepts the lowest-entropy samples
/// within `entropy_bound` and renoises the others. A block stops early once
/// its argmax canvas has held for `stability_threshold` steps and its mean
/// entropy is below `confidence_threshold`. The block's argmax canvas is then
/// committed to the request's context as causal prompt, and the next block
/// follows it. Every draw follows the request's `SamplingParams::seed`.
///
/// The deployment's launch configuration owns the sampling: every request a
/// block-diffusion runtime generates, rather than reads out, runs under it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CanvasSampling {
    /// Tokens of one block.
    pub canvas_length: u32,
    /// Denoising steps a block runs at most.
    pub max_steps: u32,
    /// Entropy, in nats, the accepted samples of one step may carry beyond
    /// the largest one.
    pub entropy_bound: f32,
    /// Sampling temperature of a block's last step.
    pub t_min: f32,
    /// Sampling temperature of a block's first step.
    pub t_max: f32,
    /// Mean canvas entropy, in nats, below which a stable block stops; zero
    /// never stops a block early.
    pub confidence_threshold: f32,
    /// Steps the argmax canvas must hold unchanged before a block stops;
    /// zero stops a block on confidence alone.
    pub stability_threshold: u32,
}

impl CanvasSampling {
    /// Returns the first parameter outside the domain the block-diffusion
    /// sampler takes, if any: a positive canvas length and step limit, a
    /// positive entropy bound, temperatures with `0 <= t_min < t_max`, and a
    /// non-negative confidence threshold, all finite. A zero stability
    /// threshold stops a block on confidence alone, and a zero confidence
    /// threshold runs every block for all its steps.
    pub fn invalid_parameter(&self) -> Option<&'static str> {
        let positive = |value: f32| value.is_finite() && value > 0.0;
        if self.canvas_length == 0 {
            Some("canvas_length")
        } else if self.max_steps == 0 {
            Some("max_steps")
        } else if !positive(self.entropy_bound) {
            Some("entropy_bound")
        } else if !(self.t_min.is_finite() && self.t_min >= 0.0) {
            Some("t_min")
        } else if !(self.t_max.is_finite() && self.t_max > self.t_min) {
            Some("t_max")
        } else if !(self.confidence_threshold.is_finite() && self.confidence_threshold >= 0.0) {
            Some("confidence_threshold")
        } else {
            None
        }
    }
}
