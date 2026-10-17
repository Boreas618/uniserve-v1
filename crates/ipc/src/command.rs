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

/// Autoregressive request parameters fixed for the worker request lifetime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArRequestParams {
    /// Sampling policy shared by autoregressive calls.
    pub sampling: SamplingParams,
    /// Token identifiers used for negative-conditioning input.
    pub negative_token_ids: Vec<u32>,
    /// Canonical token identifiers that terminate generation.
    pub finish_token_ids: Vec<u32>,
    /// Logical position assigned to the first request token.
    pub initial_position: u32,
    /// Block-diffusion sampling of a request that generates its text in
    /// canvases; its seed is `sampling.seed`.
    pub canvas: Option<CanvasSampling>,
}

pub use uniserve_core::{CanvasSampling, DiffusionSamplingParams};

/// What a video request admits besides its sampling controls.
///
/// The engine forms it from the `DiffusionRequest` it admits; the conditions'
/// media stay published by the engine until the request retires, so a
/// worker reads them by locator while the request lives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoAdmission {
    /// The task the request runs.
    pub task: uniserve_core::VideoTask,
    /// The denoiser's AdaLN tag of each prompt token: 0 for a vision token
    /// or vision marker, 1 for text.
    pub text_tags: Vec<u8>,
    /// The conditions in request order.
    pub conditions: Vec<uniserve_core::VideoCondition>,
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
    /// Autoregressive request parameters, when applicable.
    pub ar: Option<ArRequestParams>,
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
    /// Constructs static autoregressive or multimodal admission state.
    ///
    /// Checks only the request-pool index and that `ar` or `image` is present;
    /// it does not run [`Self::validate`] on the family parameters.
    pub fn new(
        request_key: RequestKey,
        request_pool_idx: u32,
        ar: Option<ArRequestParams>,
        image: Option<ImageParams>,
        input_images: u32,
    ) -> ValidationResult<Self> {
        ensure_valid!(request_pool_idx > 0, "request-pool index must be positive");
        ensure_valid!(
            ar.is_some() || image.is_some(),
            "request start must declare autoregressive or image parameters"
        );
        let admission = Self {
            request_key,
            request_pool_idx,
            ar,
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
            ar: None,
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
            self.ar.is_some() || self.image.is_some() || self.diffusion.is_some(),
            "request start must declare one runtime-family parameter set"
        );
        if let Some(ar) = &self.ar {
            ar.sampling.validate()?;
            if let Some(canvas) = &ar.canvas
                && let Some(parameter) = canvas.invalid_parameter()
            {
                bail_invalid!("invalid canvas sampling {parameter}");
            }

            ensure_valid!(
                ar.finish_token_ids.windows(2).all(|pair| pair[0] < pair[1]),
                "autoregressive finish token ids are not canonical"
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
                    && self.prompt_token_ids.len() <= u32::MAX as usize
                    && diffusion.num_inference_steps > 0,
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
                (video.task == uniserve_core::VideoTask::T2va) == video.conditions.is_empty(),
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
