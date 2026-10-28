//! Requests and their lifecycle events.
//!
//! The daemon submits each request as a [`Request`], a token generation or a
//! video diffusion, and receives that request's [`Event`]s in order.
//! `Finished`, `Rejected`, and `Error` are terminal.

use serde::{Deserialize, Serialize};

use std::sync::Arc;

use super::{DiffusionSamplingParams, RequestId, TokenLogprob};

/// Terminal cause for one generation request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Generation completed normally.
    Completed,
    /// An end-of-sequence token ended text generation.
    Eos,
    /// The request reached its generated-token limit.
    MaxTokens,
    /// A configured stop token or string matched.
    Stop,
    /// The image-only generation objective completed.
    ImageDone,
    /// The client cancelled the request at a public output boundary.
    Cancelled,
    /// The server aborted the request immediately.
    Aborted,
    /// Repetition policy ended generation.
    Repetition,
    /// Execution or output processing failed.
    Error,
}

/// Typed payload identifying the exact stop condition that ended generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum StopReason {
    /// Vocabulary token that matched a stop condition.
    Token(u32),
    /// Text sequence that matched a stop condition.
    String(String),
}

/// Ranked candidates for one scored token position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionLogprobs {
    /// Ranked token candidates for this position.
    pub entries: Vec<TokenLogprob>,
}

/// Typed event stream the engine emits for each request, of either kind.
#[derive(Debug, Clone)]
pub enum Event {
    /// Reports scheduler admission timing.
    Scheduled {
        /// Time the request entered the queue, in seconds since the Unix epoch.
        queued_at: f64,
        /// Time the request was admitted, in seconds since the Unix epoch.
        scheduled_at: f64,
    },
    /// Publishes one generated text token.
    TextToken {
        /// Vocabulary token identity.
        id: u32,
        /// Token log probability when requested.
        logprob: Option<f32>,
    },
    /// Publishes generated text tokens committed together, in order, such as
    /// a stopped block-diffusion canvas; they carry no log probabilities.
    /// Consumers treat them as consecutive `TextToken`s.
    TextTokens {
        /// Vocabulary token identities.
        ids: Vec<u32>,
    },
    /// Publishes ranked candidates associated with a text token.
    TokenLogprobs {
        /// Generated token identity this payload scores.
        id: u32,
        /// Ranked vocabulary candidates.
        candidates: Vec<TokenLogprob>,
    },
    /// Publishes scores for prompt positions processed during prefill.
    PromptLogprobs {
        /// Scored prompt positions in prompt order.
        positions: Vec<PositionLogprobs>,
    },
    /// Publishes a readout request's answer, once, before its `Finished`.
    Readout {
        /// Natural-log probability of every candidate of every slot, in
        /// row, slot, and candidate order
        /// (`GenerationRequest::readout_candidates` values).
        candidate_logprobs: Vec<f32>,
    },
    /// Opens the lifecycle of one generated image.
    ///
    /// The engine emits it together with the image's first committed
    /// denoising step, before that step's `ImageStep`.
    ImageBegin {
        /// Request-local image identity.
        image_id: u32,
        /// Output height in pixels.
        height: u32,
        /// Output width in pixels.
        width: u32,
        /// Configured denoising step count.
        steps: u16,
    },
    /// Reports one committed diffusion step. Each committed step is reported
    /// once, in increasing order.
    ImageStep {
        /// Request-local image identity.
        image_id: u32,
        /// One-based committed step number.
        step: u16,
    },
    /// Marks the transition from denoising to image materialization.
    ///
    /// The engine emits it when the image-decoding call returns, before the
    /// image's `ImageDone`.
    ImageCommit {
        /// Request-local image identity.
        image_id: u32,
    },
    /// Publishes a materialized PNG image.
    ///
    /// The engine reads `height`, `width`, and `bytes` from the PNG header
    /// and the payload length; a payload without a PNG header finishes the
    /// request with [`FinishReason::Error`] instead.
    ImageDone {
        /// Request-local image identity.
        image_id: u32,
        /// Decoded image height in pixels.
        height: u32,
        /// Decoded image width in pixels.
        width: u32,
        /// PNG payload size in bytes.
        bytes: u64,
        /// Base64-encoded PNG payload.
        pixels_png_b64: String,
        /// The decoded image product the request's feedback encoders read,
        /// when it feeds the image back from the device
        /// (`FeedbackSource::DeviceProduct`): the image's exact feedback
        /// input, which the daemon keeps to submit a migrated request again
        /// with it (`GeneratedImage`).
        feedback: Option<Arc<super::ImagePixels>>,
    },
    /// A generated token the request's text does not include: one that opens
    /// an image or closes a round, or one of a request that publishes no
    /// text. It counts as generated. The daemon keeps it, in order with the
    /// text tokens, to submit a migrated request again with every token it
    /// generated (`GenerationRequest::generated_token_ids`); no client reads
    /// it.
    ControlToken {
        /// Vocabulary token identity.
        id: u32,
    },
    /// Reports actual media computation progress committed by the engine.
    ///
    /// The engine emits it for diffusion media requests after each valid media
    /// call result, until the request fails. While delivery is backed up, a
    /// newer progress event replaces an undelivered one that is still the most
    /// recently queued event.
    MediaProgress {
        /// Current numerical or output phase: `preparing`, `denoising`,
        /// `decoding`, or `finalizing`.
        phase: String,
        /// Number of completed denoising steps.
        completed_steps: u32,
    },
    /// Publishes a transport-backed media artifact.
    Artifact(ArtifactEvent),
    /// Generated media could not be acquired from its published storage.
    ///
    /// Not terminal by itself: the engine also fails the call whose output
    /// could not be acquired.
    ArtifactUnavailable {
        /// Storage error presented to the artifact consumer.
        message: String,
    },
    /// Terminates a successfully accepted request.
    Finished {
        /// Terminal generation cause.
        reason: FinishReason,
        /// Exact matched stop condition, when applicable.
        stop_reason: Option<StopReason>,
        /// Number of prompt tokens charged to the request.
        prompt_tokens: usize,
        /// Number of generated tokens charged to the request.
        completion_tokens: usize,
        /// Number of generated images.
        images: usize,
    },
    /// Rejects a request before execution begins.
    Rejected {
        /// Whether the request itself is unservable or the engine is full.
        kind: RejectionKind,
        /// Human-readable rejection reason.
        message: String,
    },
    /// Terminates a request after an execution failure.
    Error {
        /// Human-readable failure reason.
        message: String,
    },
    /// Ends a request on this engine without finishing it: every token it
    /// generated here was sent before, and the daemon may submit it again
    /// with those tokens (`GenerationRequest::generated_token_ids`), or a
    /// video request with the trajectory steps the engine stored
    /// (`Trajectory::denoised_steps`), to the engine that serves next.
    Migratable {
        /// Why the request leaves its engine.
        cause: MigrationCause,
        /// For a video request, the denoising steps of its trajectory the
        /// engine stored in its host latent pools at the step boundary it
        /// was preempted at; zero when it stored none.
        denoised_steps: u32,
        /// With stored steps, where the engine put the trajectory's pages
        /// and how each stored rank laid its part out
        /// (`Trajectory::stored_layout`); empty otherwise.
        stored_layout: Vec<u8>,
    },
}

/// Why a request leaves the engine that served it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationCause {
    /// The engine parks for a world change or a rollback.
    Drain,
    /// The workers that held the request were lost.
    Loss,
}

impl Event {
    /// Whether the event ends its request's stream.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Finished { .. }
                | Self::Rejected { .. }
                | Self::Error { .. }
                | Self::Migratable { .. }
        )
    }
}

/// Why admission refused a request.
///
/// Callers map the kind to their own vocabulary: an invalid request fails the
/// same way on every retry, while an overloaded engine may accept the same
/// request once its waiting queue drains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectionKind {
    /// The deployment cannot serve the request as specified.
    Invalid,
    /// The waiting queue is at its bound.
    Overloaded,
}

/// Kind of media referenced by an artifact event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaKind {
    /// Still-image artifact.
    Image,
    /// Video artifact.
    Video,
    /// Audio artifact.
    Audio,
}

/// The bytes of one generated artifact, immutable once published.
///
/// The engine holds the worker's export and passes it on by descriptor
/// (`descriptor`); the daemon holds the mapping of the descriptor it
/// received. Either keeps the bytes readable for as long as it lives.
pub trait ArtifactMedia: std::fmt::Debug + Send + Sync {
    /// The artifact's bytes.
    fn bytes(&self) -> &[u8];

    /// The shared-memory object holding the bytes, which another process
    /// maps; `None` for bytes held in this process alone.
    fn descriptor(&self) -> Option<std::os::fd::BorrowedFd<'_>>;
}

/// One caller-visible artifact retaining its immutable shared-storage mapping.
#[derive(Debug, Clone)]
pub struct ArtifactEvent {
    /// Semantic media type of the artifact.
    pub media_kind: MediaKind,
    /// MIME content type of the artifact payload.
    pub content_type: String,
    /// Mapped payload retained through delivery, job retention, and active downloads.
    pub media: Arc<dyn ArtifactMedia>,
}

/// A video request. Final media bytes are returned through shared storage.
///
/// What the request asks of the model family (its task's semantics, its
/// presentation tags and its conditions) travels opaque in `conditioning`
/// to the workers; the engine schedules it by the task's name and by the
/// sizes of its conditions (`conditions`) alone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionRequest {
    /// Engine request identity.
    pub request_id: RequestId,
    /// The task the request runs, by its request name; the deployment's
    /// denoiser must serve it.
    pub task: String,
    /// The conditioner's presentation (condition labels, vision placeholders
    /// and the prompt), tokenized.
    pub prompt_token_ids: Vec<u32>,
    /// The request's task, presentation tags and conditions as the model
    /// family plans them, which the workers decode.
    pub conditioning: Vec<u8>,
    /// The sizes of each condition, in request order; empty for a task
    /// without conditions. The submitter keeps each condition's media
    /// published, under the locator the conditioning names, until the
    /// request ends.
    pub conditions: Vec<ConditionExtent>,
    /// Scheduler priority.
    pub priority: i32,
    /// Effective diffusion controls and model-preprocessed media unit count.
    pub sampling: DiffusionSamplingParams,
    /// The request's latent trajectory across the engines that serve it.
    pub trajectory: Trajectory,
    /// The digests of what each of its encoder computations reads, which
    /// key their outputs in the host tier; `None` for a request whose
    /// encoder outputs are neither stored nor found. Boxed, as the digests
    /// are larger than the rest of the request.
    #[serde(default)]
    pub encoder_inputs: Option<Box<EncoderInputs>>,
}

/// The SHA-256 digests of what each encoder computation of a video request
/// reads: the text encoding's (the presentation and the vision features'
/// inputs), the vision encoding's, each visual condition unit's latent
/// encoding's and the audio encoding's.
///
/// The daemon computes them from the request's facts (its presentation and
/// the content and plan of each condition's media), so that requests whose
/// encoders read the same inputs, the same request on another engine among
/// them, find each other's outputs. An engine keys an encoder's output by
/// these digests and the identity of what the encoder computes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncoderInputs {
    /// The text encoding's inputs.
    pub text: [u8; 32],
    /// The vision encoding's inputs; `None` without a vision block.
    pub vision: Option<[u8; 32]>,
    /// Each visual condition unit's inputs, in encoding order: the units of
    /// [`ConditionExtent::latent_units`], condition by condition.
    pub video_units: Vec<[u8; 32]>,
    /// The audio encoding's inputs; `None` without an audio track.
    pub audio: Option<[u8; 32]>,
}

/// A video request's latent trajectory across the engines that serve it.
///
/// An engine that preempts the request at a denoising-step boundary stores
/// the trajectory in its host latent pools, each page keyed by `id`, the
/// steps it took, the latent seed of what its denoising computes and the
/// part of the samples the page holds; the daemon submits the request again
/// with those steps and the engine's record of the stored layout
/// (`Event::Migratable`), and the engine that serves it next continues from
/// them when its pools hold the trajectory under its own latent seed (or
/// `state_seed`), laying the pages out in its own ranks' pages whatever
/// parallel configuration stored them, and from noise otherwise. Either way
/// it computes the same trajectory, since a trajectory's state at a step
/// follows from the request alone.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Trajectory {
    /// The identity that keys the stored trajectory: the daemon gives a
    /// request one when it first submits it and keeps it across its
    /// migrations.
    pub id: u64,
    /// The denoising steps of the trajectory an engine stored when it
    /// preempted the request; zero for a request that starts from noise.
    pub denoised_steps: u32,
    /// The latent seed of the world whose stored trajectory the request
    /// continues although the serving world's denoising computes other
    /// numerics (a world change's `in_flight = keep`); `None` reads a
    /// trajectory of the serving world's own latent seed only.
    pub state_seed: Option<u64>,
    /// Where the engine that stored the trajectory put its pages and how
    /// each stored rank laid its part of the samples out, as that engine
    /// recorded it; empty when `denoised_steps` is zero. Opaque to the
    /// daemon.
    pub stored_layout: Vec<u8>,
}

/// The sizes of one condition of a video request, which size the products of
/// the calls that read and encode it.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ConditionExtent {
    /// Bytes of the RGB24 pixels the video encoder encodes; zero for a
    /// condition without pixels.
    pub pixel_bytes: u64,
    /// Samples of the condition's audio track; zero without one.
    pub audio_samples: u64,
    /// Patches the conditioner reads; zero for a condition it does not read.
    pub vision_patches: u64,
    /// Vision tokens the condition occupies in the presentation.
    pub vision_tokens: u32,
    /// Denoiser rows each temporal unit of the condition's pixels encodes
    /// to, in unit order; empty without pixels.
    pub latent_units: Vec<u32>,
    /// Denoiser audio rows of the condition's audio track; zero without one.
    pub audio_rows: u32,
}

impl ConditionExtent {
    /// Denoiser video rows of the condition.
    pub fn video_rows(&self) -> u64 {
        self.latent_units.iter().map(|rows| u64::from(*rows)).sum()
    }
}

impl DiffusionRequest {
    /// Checks the request's own consistency.
    ///
    /// The prompt is nonempty and its token count fits in `u32`; the frame
    /// and media-unit counts and the canvas are positive; and the task is
    /// named and the conditioning present.
    ///
    /// The engine's media admission rejects a failing request with
    /// `RejectionKind::Invalid`; it also refuses a task the deployment's
    /// denoiser does not serve, or conditions the task does not take.
    pub fn validate(&self) -> Result<(), DiffusionRequestError> {
        if self.prompt_token_ids.is_empty() {
            return Err(DiffusionRequestError::EmptyPromptTokens);
        }
        if self.sampling.num_frames == 0
            || self.sampling.video_units == 0
            || self.sampling.width == 0
            || self.sampling.height == 0
            || self.prompt_token_ids.len() > u32::MAX as usize
        {
            return Err(DiffusionRequestError::InvalidSampling);
        }
        if self.task.is_empty() || self.conditioning.is_empty() {
            return Err(DiffusionRequestError::Conditions(
                "the request names no task or carries no conditioning",
            ));
        }
        // The encoder inputs name what the request's encoders read: a unit
        // digest for each visual condition unit, a vision digest when a
        // condition occupies vision tokens, an audio digest when one has
        // an audio track.
        if let Some(inputs) = &self.encoder_inputs {
            let units: usize = self
                .conditions
                .iter()
                .map(|condition| condition.latent_units.len())
                .sum();
            let vision = self
                .conditions
                .iter()
                .any(|condition| condition.vision_tokens > 0);
            let audio = self
                .conditions
                .iter()
                .any(|condition| condition.audio_rows > 0);
            if inputs.video_units.len() != units
                || inputs.vision.is_some() != vision
                || inputs.audio.is_some() != audio
            {
                return Err(DiffusionRequestError::Conditions(
                    "the encoder inputs do not name the request's encoders",
                ));
            }
        }
        Ok(())
    }
}

/// Validation failures for a media generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DiffusionRequestError {
    /// The tokenized prompt contains no tokens.
    #[error("media prompt tokens must not be empty")]
    EmptyPromptTokens,
    /// A frame, media-unit, or step count or a canvas side is zero, or the
    /// prompt token count exceeds `u32::MAX`.
    #[error("diffusion parameters are invalid")]
    InvalidSampling,
    /// The task or the conditioning is missing.
    #[error("video conditions are invalid: {0}")]
    Conditions(&'static str),
}

/// Immutable request payload selected before it enters the engine.
///
/// The engine serves each kind on the workers whose capabilities cover it and
/// refuses one they cannot serve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Request {
    /// Token generation: autoregressive text, interleaved text and images, or
    /// block-diffusion canvases and readouts. Boxed, as it is several times
    /// the size of a diffusion request.
    Generation(Box<super::GenerationRequest>),
    /// Video diffusion.
    Diffusion(DiffusionRequest),
}

impl From<super::GenerationRequest> for Request {
    /// Wraps a token generation request in the shared request enum.
    fn from(request: super::GenerationRequest) -> Self {
        Self::Generation(Box::new(request))
    }
}

impl Request {
    /// Returns the request identifier.
    pub fn request_id(&self) -> RequestId {
        match self {
            Self::Generation(request) => request.request_id,
            Self::Diffusion(request) => request.request_id,
        }
    }

    /// Returns the scheduling priority.
    pub fn priority(&self) -> i32 {
        match self {
            Self::Generation(request) => request.priority,
            Self::Diffusion(request) => request.priority,
        }
    }
}
