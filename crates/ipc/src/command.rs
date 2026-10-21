//! Request lifecycle commands a batch carries: a request's start with the
//! state it is admitted with ([`NewRequest`]), the end of a request epoch,
//! and the release of a persistent product.

use super::*;

/// Ordered request and buffer commands carried with logical work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchCommand {
    /// Establish one request before its first call on a pool.
    Start {
        /// Static request state installed by the worker.
        request: Box<NewRequest>,
    },
    /// Close an exact request epoch after admitted work and physical readers finish.
    Finish {
        /// Exact request epoch to close.
        request_key: RequestKey,
        /// Products that remain readable under independent ownership until Free.
        retained_buffers: Vec<BufferId>,
    },
    /// Free one exact persistent product after its final consumer.
    Free {
        /// Persistent buffer identity to release.
        buffer: BufferId,
    },
}

impl BatchCommand {
    /// Returns the request targeted by this command.
    pub fn request_key(&self) -> RequestKey {
        match self {
            Self::Start { request } => request.request_key,
            Self::Finish { request_key, .. } => *request_key,
            Self::Free { buffer } => buffer.owner,
        }
    }

    /// Returns a discriminant that distinguishes command variants.
    ///
    /// [`Batch::validate`] uses it in the identity that detects repeated
    /// commands.
    pub const fn variant_index(&self) -> u8 {
        match self {
            Self::Start { .. } => 0,
            Self::Finish { .. } => 1,
            Self::Free { .. } => 2,
        }
    }

    /// Validates command identity and retained resource constraints.
    pub fn validate(&self) -> ValidationResult<()> {
        match self {
            Self::Start { request } => request.validate()?,
            Self::Finish { .. } => {}
            Self::Free { buffer } => buffer.validate()?,
        }
        if let Self::Finish {
            request_key,
            retained_buffers,
            ..
        } = self
        {
            let mut retained = std::collections::HashSet::new();
            for buffer in retained_buffers {
                buffer.validate()?;
                ensure_valid!(
                    buffer.owner == *request_key,
                    "retained buffer belongs to another request"
                );
                ensure_valid!(
                    retained.insert(*buffer),
                    "request retirement repeats a retained buffer"
                );
            }
        }
        Ok(())
    }
}

/// A request's token-generation parameters, fixed for its lifetime on a rank.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationParams {
    /// Sampling policy shared by the request's token calls.
    pub sampling: SamplingParams,
    /// Token identifiers used for negative-conditioning input.
    pub negative_token_ids: Vec<u32>,
    /// Canonical token identifiers that terminate generation.
    pub finish_token_ids: Vec<u32>,
    /// Logical position assigned to the first request token.
    pub initial_position: u32,
    /// Tokens the request generated on a predecessor engine before it
    /// resumed on this one (a control-plane handover), in order. They end
    /// its prompt, and its penalties count them as generated tokens.
    pub generated_token_ids: Vec<u32>,
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

/// What a video request admits besides its sampling controls.
///
/// The engine forms it from the `DiffusionRequest` it admits; the conditions'
/// media stay published by the engine until the request retires, so a
/// worker reads them by locator while the request lives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoAdmission {
    /// The task the request runs.
    pub task: models::video::VideoTask,
    /// The denoiser's AdaLN tag of each prompt token: 0 for a vision token
    /// or vision marker, 1 for text.
    pub text_tags: Vec<u8>,
    /// The conditions in request order.
    pub conditions: Vec<models::video::VideoCondition>,
}

/// Request-start framing. Carries the per-domain parameters a request
/// needs before its calls run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewRequest {
    /// Globally unique request identity.
    pub request_key: RequestKey,
    /// Scheduler-assigned stable request-state row. Index zero is reserved for
    /// inactive graph padding and never identifies a live request.
    pub request_pool_idx: u32,
    /// Token-generation parameters, when the request generates tokens.
    pub generation: Option<GenerationParams>,
    /// Image request parameters, when applicable.
    pub image: Option<ImageParams>,
    /// Diffusion request parameters, when applicable.
    pub diffusion: Option<DiffusionSamplingParams>,
    /// A video request's task, presentation tags and conditions; present
    /// exactly with `diffusion`.
    pub video: Option<VideoAdmission>,
    /// Tokenized conditioning supplied at diffusion admission.
    pub prompt_token_ids: Vec<u32>,
    /// Number of input images the request carries. Model image preprocessing
    /// that divides a pixel budget among a request's input images reads it
    /// when it encodes one of them.
    pub input_images: u32,
}

impl NewRequest {
    /// Constructs static token-generation or multimodal admission state.
    ///
    /// Checks only the request-pool index and that `generation` or `image` is present;
    /// it does not run [`Self::validate`] on the family parameters.
    pub fn new(
        request_key: RequestKey,
        request_pool_idx: u32,
        generation: Option<GenerationParams>,
        image: Option<ImageParams>,
        input_images: u32,
    ) -> ValidationResult<Self> {
        ensure_valid!(request_pool_idx > 0, "request-pool index must be positive");
        ensure_valid!(
            generation.is_some() || image.is_some(),
            "request start must declare token-generation or image parameters"
        );
        let admission = Self {
            request_key,
            request_pool_idx,
            generation,
            image,
            diffusion: None,
            video: None,
            prompt_token_ids: Vec::new(),
            input_images,
        };
        Ok(admission)
    }

    /// Constructs static video admission state and validates it.
    pub fn new_media(
        request_key: RequestKey,
        request_pool_idx: u32,
        prompt_token_ids: Vec<u32>,
        diffusion: DiffusionSamplingParams,
        video: VideoAdmission,
    ) -> ValidationResult<Self> {
        ensure_valid!(request_pool_idx > 0, "request-pool index must be positive");
        let admission = Self {
            request_key,
            request_pool_idx,
            generation: None,
            image: None,
            diffusion: Some(diffusion),
            video: Some(video),
            prompt_token_ids,
            input_images: 0,
        };
        admission.validate()?;
        Ok(admission)
    }

    /// Validates request identity and the selected family parameters.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.request_pool_idx > 0,
            "request-pool index must be positive"
        );
        ensure_valid!(
            self.generation.is_some() || self.image.is_some() || self.diffusion.is_some(),
            "request start must declare one runtime-family parameter set"
        );
        if let Some(generation) = &self.generation {
            generation.sampling.validate()?;
            ensure_valid!(
                generation
                    .finish_token_ids
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
                "token-generation finish token ids are not canonical"
            );
        }
        if let Some(branch) = &self.image {
            branch.validate()?;
        }
        if let Some(diffusion) = &self.diffusion {
            ensure_valid!(
                !self.prompt_token_ids.is_empty(),
                "diffusion prompt tokens must not be empty"
            );
            ensure_valid!(
                diffusion.num_frames > 0
                    && diffusion.video_units > 0
                    && self.prompt_token_ids.len() <= u32::MAX as usize,
                "diffusion parameters are invalid"
            );
        }
        ensure_valid!(
            self.video.is_some() == self.diffusion.is_some(),
            "a video admission carries both its sampling and its inputs"
        );
        if let Some(video) = &self.video {
            ensure_valid!(
                video.text_tags.len() == self.prompt_token_ids.len(),
                "every video prompt token requires one tag"
            );
            ensure_valid!(
                (video.task == models::video::VideoTask::T2va) == video.conditions.is_empty(),
                "only a conditioned video task carries conditions"
            );
            for condition in &video.conditions {
                if let Err(message) = condition.validate() {
                    bail_invalid!("{message}");
                }
            }
        }
        Ok(())
    }
}
