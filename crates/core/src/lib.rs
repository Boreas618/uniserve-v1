//! Foundational request values, identifiers, clocks, and sampling primitives.
//!
//! The crate contains transport-independent contracts shared by the server,
//! scheduler, worker protocol, and simulation runtime. This root module holds
//! request and call identities, wall-clock and monotonic clocks, dtype wire
//! names, [`SamplingParams`] and [`ImageParams`] with their request
//! validation, prefix-cache descriptors, and the [`CommandWaker`] that interrupts
//! an engine waiting for commands.

mod allocation;
pub use allocation::ByteAllocator;
mod parallel;
pub use parallel::{
    ComponentConfig, ComponentDistribution, ComponentWeights, HeadProjection, ParallelConfig,
    ParallelConfigError, SequenceParallel,
};

use std::sync::{Arc, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Serializable statistics and observer-facing wire values.
pub mod codec;
mod events;
mod media;
mod shared_memory;
pub mod video;
pub use media::{MediaSource, SharedMedia, SharedMediaError};
pub use shared_memory::SharedMemory;
pub use video::{
    AudioClip, ConditionMedia, ConditionRole, ConditionTiles, ConditionVision, ImageFit,
    MediaLocator, VideoClip, VideoCondition, VisionGrid,
};
/// Multimodal generation descriptors, resource bounds, and validation.
pub mod generation;
pub mod philox;
pub use codec::stats::ForwardStats;
pub use events::{
    ArtifactEvent, Canvas, DiffusionRequest, DiffusionRequestError, DiffusionSamplingParams,
    EngineCoreOutput, FinishReason, MediaKind, PositionLogprobs, RejectionKind, Request,
    StopReason, TokenLogprob, VideoTask,
};
pub use generation::{
    CachePolicy, CanvasSampling, FeedbackNextToken, FeedbackSource, GenerationConstraint,
    GenerationConstraintParseError, GenerationFeatures, GenerationLimits, GenerationRequest,
    GenerationRequestError, GenerationResourceError, ImageEncoderInput, ImageGenerationConfig,
    ImageInput, ImageTrigger, MultimodalInputs, ReadoutRow, ReadoutSlot, encoder_cache_key,
};

/// Thread-safe notification used to wake an engine after command enqueue.
///
/// Callers run `wake` on their own thread: `EngineHandle` right after it
/// enqueues a command, and `EventRx` after it consumes an output event and
/// frees channel capacity. The engine's implementations (the `WakeSignal`
/// socket write and the simulator's progress channel send) do not block.
pub trait Wake: Send + Sync {
    /// Signals the waiting engine owner.
    fn wake(&self);
}

impl<F> Wake for F
where
    F: Fn() + Send + Sync,
{
    /// Invokes the closure as a wake notification.
    fn wake(&self) {
        self()
    }
}

/// Cloneable command wake backed by a live notifier or a no-op value.
#[derive(Clone, Default)]
pub enum CommandWaker {
    /// Performs no notification.
    #[default]
    Noop,
    /// Delegates notification to a shared [`Wake`] implementation.
    Live(Arc<dyn Wake>),
}

impl CommandWaker {
    /// Returns a waker that performs no notification.
    pub fn noop() -> Self {
        Self::Noop
    }

    /// Wraps a thread-safe wake closure.
    pub fn new(wake: impl Fn() + Send + Sync + 'static) -> Self {
        Self::Live(Arc::new(wake))
    }

    /// Signals the live notifier, or returns immediately for [`Self::Noop`].
    pub fn wake(&self) {
        if let Self::Live(waker) = self {
            waker.wake();
        }
    }

    /// Returns whether this waker performs no notification.
    pub fn is_noop(&self) -> bool {
        matches!(self, Self::Noop)
    }
}

impl std::fmt::Debug for CommandWaker {
    /// Formats the waker by variant without exposing the live closure.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CommandWaker")
            .field(&if self.is_noop() { "Noop" } else { "Live" })
            .finish()
    }
}

/// Returns wall-clock time in fractional seconds since the Unix epoch.
///
/// Times before the epoch clamp to zero.
pub fn now_unix_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Returns wall-clock time in whole seconds since the Unix epoch.
///
/// Times before the epoch clamp to zero.
pub fn now_unix_secs_u64() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Returns process-local monotonic time in fractional seconds from a stable epoch.
///
/// The epoch is fixed by the first call to this function or
/// [`now_monotonic_us`] in the process, so values are not comparable across
/// processes. Differences between values remain valid across wall-clock
/// adjustments.
pub fn now_monotonic_secs() -> f64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// Returns process-local monotonic time in whole microseconds from the epoch of
/// [`now_monotonic_secs`]. Used for lifecycle phase stamps, whose differences
/// stay valid across wall-clock adjustments.
pub fn now_monotonic_us() -> u64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_micros() as u64
}

/// Process-wide origin shared by the monotonic clocks, set on first use.
static EPOCH: OnceLock<Instant> = OnceLock::new();

/// Index of a physical allocation unit in the paged cache pool.
///
/// The engine's block pool allocates units, and a block table maps each
/// logical page of a cache group onto that group's number of units; worker
/// calls name them as unit ids. The worker treats unit zero as its padding
/// sentinel, valid memory in every plane, so the pool never allocates it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct UnitId(pub u32);

/// Request identity within one engine.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
pub struct RequestId(pub u64);

/// Logical call identity, preserved when a scheduler batch is split across workers.
/// The request index is the ordinal assigned at selection, before physical row packing.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct CallId {
    /// Scheduler batch the call was selected into.
    pub batch_id: u64,
    /// Ordinal of the call within that batch at selection time.
    pub request_index: u32,
}

impl CallId {
    pub const fn new(batch_id: u64, request_index: u32) -> Self {
        Self {
            batch_id,
            request_index,
        }
    }
}

/// A concrete media-input, encoder, diffusion, decoder, or media-output
/// computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum MediaCall {
    /// Decodes a video request's condition media on a host rank into the
    /// inputs of the vision and latent encoders.
    MediaReading,
    VisionEncoding,
    LatentEncoding,
    TextEncoding,
    LatentPreparation,
    Denoising,
    ImageDecoding,
    VideoDecoding,
    AudioDecoding,
    VideoEncoding,
    AudioEncoding,
    Muxing,
}

impl MediaCall {
    /// Every media call in declaration order.
    pub const ALL: [Self; 12] = [
        Self::MediaReading,
        Self::VisionEncoding,
        Self::LatentEncoding,
        Self::TextEncoding,
        Self::LatentPreparation,
        Self::Denoising,
        Self::ImageDecoding,
        Self::VideoDecoding,
        Self::AudioDecoding,
        Self::VideoEncoding,
        Self::AudioEncoding,
        Self::Muxing,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MediaReading => "media_reading",
            Self::VisionEncoding => "vision_encoding",
            Self::LatentEncoding => "latent_encoding",
            Self::TextEncoding => "text_encoding",
            Self::LatentPreparation => "latent_preparation",
            Self::Denoising => "denoising",
            Self::ImageDecoding => "image_decoding",
            Self::VideoDecoding => "video_decoding",
            Self::AudioDecoding => "audio_decoding",
            Self::VideoEncoding => "video_encoding",
            Self::AudioEncoding => "audio_encoding",
            Self::Muxing => "muxing",
        }
    }

    /// The calls that read this call's products in a video request's graph,
    /// in the order the graph consumes them.
    ///
    /// The engine plans a request's consumption from this table, and both
    /// the engine and the worker derive from it which ranks back a
    /// component's persistent products: the ranks that execute the
    /// component and those that execute a reader of its products. A reader
    /// of resident state ([`Self::reads_resident_state`]) reads the
    /// denoiser's latent pages instead of a persistent product.
    pub const fn readers(self) -> &'static [Self] {
        match self {
            Self::MediaReading => &[Self::VisionEncoding, Self::LatentEncoding],
            Self::VisionEncoding => &[Self::TextEncoding],
            Self::TextEncoding | Self::LatentEncoding => &[Self::LatentPreparation],
            Self::LatentPreparation => &[Self::Denoising],
            Self::Denoising => &[Self::Denoising, Self::VideoDecoding, Self::AudioDecoding],
            Self::VideoDecoding => &[Self::VideoEncoding],
            Self::VideoEncoding => &[Self::Muxing],
            Self::AudioDecoding => &[Self::AudioEncoding],
            Self::ImageDecoding | Self::AudioEncoding | Self::Muxing => &[],
        }
    }

    /// Whether this call reads its predecessor's trajectory from the
    /// denoiser's resident latent pages instead of a persistent product.
    pub const fn reads_resident_state(self) -> bool {
        matches!(self, Self::Denoising)
    }
}

/// Understanding and generation branches of a multimodal model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    /// Understanding or text branch.
    Und,
    /// Image-generation or latent branch.
    Gen,
}

/// Effective model dtype reported after runtime configuration resolves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelDtype {
    /// IEEE 754 half precision.
    #[serde(rename = "float16")]
    Float16,
    /// Brain floating-point half precision.
    #[serde(rename = "bfloat16")]
    BFloat16,
    /// IEEE 754 single precision.
    #[serde(rename = "float32")]
    Float32,
}

impl ModelDtype {
    /// Returns the stable wire name for this dtype.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Float32 => "float32",
        }
    }

    /// Parses a supported wire name without allocating an error.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "float16" => Some(Self::Float16),
            "bfloat16" => Some(Self::BFloat16),
            "float32" => Some(Self::Float32),
            _ => None,
        }
    }
}

impl std::str::FromStr for ModelDtype {
    type Err = ModelDtypeParseError;

    /// Parses a stable model-dtype wire name.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value).ok_or_else(|| ModelDtypeParseError(value.to_owned()))
    }
}

impl std::fmt::Display for ModelDtype {
    /// Writes the stable model-dtype wire name.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Error returned for an unsupported model dtype string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported model dtype {0:?}")]
pub struct ModelDtypeParseError(String);

/// Storage dtype used by paged key/value cache tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrefixCacheDtype {
    /// IEEE 754 half precision.
    #[serde(rename = "float16")]
    Float16,
    /// Brain floating-point half precision.
    #[serde(rename = "bfloat16")]
    BFloat16,
    /// IEEE 754 single precision.
    #[serde(rename = "float32")]
    Float32,
    /// Finite-only E4M3 8-bit floating point.
    #[serde(rename = "float8_e4m3fn")]
    Float8E4m3Fn,
}

impl PrefixCacheDtype {
    /// Returns the stable wire name for this dtype.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Float32 => "float32",
            Self::Float8E4m3Fn => "float8_e4m3fn",
        }
    }
}

impl std::str::FromStr for PrefixCacheDtype {
    type Err = PrefixCacheDtypeParseError;

    /// Parses a stable prefix-cache dtype wire name.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "float16" => Ok(Self::Float16),
            "bfloat16" => Ok(Self::BFloat16),
            "float32" => Ok(Self::Float32),
            "float8_e4m3fn" => Ok(Self::Float8E4m3Fn),
            _ => Err(PrefixCacheDtypeParseError(value.to_owned())),
        }
    }
}

impl std::fmt::Display for PrefixCacheDtype {
    /// Writes the stable prefix-cache dtype wire name.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Error returned for an unsupported prefix-cache dtype string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported KV-cache dtype {0:?}")]
pub struct PrefixCacheDtypeParseError(String);

/// Text sampling parameters.
///
/// Worker-side math (temperature, top_k, top_p, min_p, typical_p, penalties,
/// logit_bias, allowed and forced tokens) operates on the logits tensor inside
/// the worker. The engine scheduler enforces the
/// control-flow fields on the host: `min_tokens` and `bad_words_ids` become
/// per-step suppression masks, and `ignore_eos` decides whether a sampled EOS
/// finishes the request. Fields are scalars or small id/weight lists — never
/// tensors — so they can cross the worker boundary without carrying device
/// state.
///
/// The worker receives these fields in a request's admission message and
/// stages them for the Python samplers in `uniserve.sampling`. All consumers share this FP32 representation
/// and its numerical validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Softmax temperature; zero selects greedy decoding.
    pub temperature: f32,
    /// Maximum number of highest-logit candidates; zero disables the cutoff.
    pub top_k: u32,
    /// Cumulative probability mass retained by nucleus sampling.
    pub top_p: f32,
    /// Whether a sampled EOS token leaves the request running instead of
    /// finishing it. EOS stays eligible for sampling either way once the
    /// minimum-token floor is met.
    pub ignore_eos: bool,
    /// Optional deterministic random seed.
    pub seed: Option<u64>,
    /// Minimum-probability cutoff relative to the top token.
    pub min_p: f32,
    /// Multiplicative repetition penalty (1.0 == no-op).
    pub repetition_penalty: f32,
    /// Additive frequency penalty over the running output (0.0 == no-op).
    pub frequency_penalty: f32,
    /// Additive presence penalty over the running output (0.0 == no-op).
    pub presence_penalty: f32,
    /// Per-token logit bias, `(token_id, bias)` pairs (empty == no-op).
    pub logit_bias: Vec<(u32, f32)>,
    /// Floor on generated tokens before EOS/stop may fire.
    pub min_tokens: usize,
    /// Whether to return the sampled token's log probability.
    pub return_logprobs: bool,
    /// How many highest-probability alternatives to return per generated step.
    pub n_logprobs: u32,
    /// Whether to score prompt positions during prefill.
    pub return_prompt_logprobs: bool,
    /// How many highest-probability alternatives to return per scored prompt position.
    pub n_prompt_logprobs: u32,
    /// Bad-word token sequences: generation may not complete any of these.
    /// Enforced host-side by masking the completing token.
    pub bad_words_ids: Vec<Vec<u32>>,
    /// If set, only these token ids may be sampled (whitelist mask).
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Typical-sampling mass cutoff over the locally-typical set (1.0 == no-op).
    pub typical_p: f32,
    /// Forced-decoding schedule: point `i` of a call's span is forced to
    /// `forced_token_ids[i]` when present, overriding stochastic selection.
    pub forced_token_ids: Vec<u32>,
}

/// Validation failure returned by [`SamplingParams::validate`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SamplingParamsError {
    /// A floating-point parameter is NaN or infinite.
    #[error("{field} must be finite, got {got}")]
    NonFinite {
        /// Name of the invalid parameter.
        field: &'static str,
        /// Supplied non-finite value.
        got: f32,
    },
    /// Temperature is below zero.
    #[error("temperature must be non-negative, got {got}")]
    NegativeTemperature {
        /// Supplied temperature.
        got: f32,
    },
    /// Nucleus probability lies outside `(0, 1]`.
    #[error("top_p must be in (0, 1], got {got}")]
    TopP {
        /// Supplied probability.
        got: f32,
    },
    /// Typical-sampling probability lies outside `(0, 1]`.
    #[error("typical_p must be in (0, 1], got {got}")]
    TypicalP {
        /// Supplied probability.
        got: f32,
    },
    /// Minimum relative probability lies outside `[0, 1]`.
    #[error("min_p must be in [0, 1], got {got}")]
    MinP {
        /// Supplied probability.
        got: f32,
    },
    /// Repetition penalty is not positive.
    #[error("repetition_penalty must be positive, got {got}")]
    RepetitionPenalty {
        /// Supplied penalty.
        got: f32,
    },
    /// An explicit allowed-token set is empty.
    #[error("allowed_token_ids must contain at least one token when present")]
    EmptyAllowedTokenIds,
    /// A bad-word sequence is empty.
    #[error("bad_words_ids[{index}] must contain at least one token")]
    EmptyBadWord {
        /// Index of the empty sequence.
        index: usize,
    },
}

impl SamplingParams {
    /// Whether token-count penalties affect selection.
    pub fn uses_penalties(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
    }

    /// Parameters that permit unpenalized device argmax without score readout.
    /// Per-call token masks and draft chains are checked by the sampler.
    pub fn device_greedy(&self) -> bool {
        self.temperature <= 0.0
            && self.top_k == 0
            && self.top_p == 1.0
            && self.typical_p == 1.0
            && !self.generated_logprobs_requested()
            && self.logit_bias.is_empty()
            && !self.uses_penalties()
    }

    /// Returns whether generated-token logprobs are enabled.
    pub fn generated_logprobs_requested(&self) -> bool {
        self.return_logprobs || self.n_logprobs > 0
    }

    /// Returns whether prompt-token logprobs are enabled.
    pub fn prompt_logprobs_requested(&self) -> bool {
        self.return_prompt_logprobs || self.n_prompt_logprobs > 0
    }

    /// Validates sampling math inputs before they reach a worker or simulator.
    ///
    /// Callers include the server's sampling lowering,
    /// `GenerationRequest::validate`, and the worker IPC
    /// `NewRequest::validate`. It checks finiteness, the numeric domain of
    /// each transform, and that token constraints are non-empty. It does not
    /// check token ids against a vocabulary or bound `top_k`, `min_tokens`, or
    /// the logprob counts.
    pub fn validate(&self) -> Result<(), SamplingParamsError> {
        // Reject non-finite scalar math inputs before applying their individual
        // range constraints.
        for (field, value) in [
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("typical_p", self.typical_p),
            ("min_p", self.min_p),
            ("repetition_penalty", self.repetition_penalty),
            ("frequency_penalty", self.frequency_penalty),
            ("presence_penalty", self.presence_penalty),
        ] {
            if !value.is_finite() {
                return Err(SamplingParamsError::NonFinite { field, got: value });
            }
        }

        // Bias values participate directly in logit arithmetic and follow the
        // same finite-value contract as scalar processors.
        for &(_, bias) in &self.logit_bias {
            if !bias.is_finite() {
                return Err(SamplingParamsError::NonFinite {
                    field: "logit_bias",
                    got: bias,
                });
            }
        }

        // Enforce the mathematical domains of each sampling transform.
        if self.temperature < 0.0 {
            return Err(SamplingParamsError::NegativeTemperature {
                got: self.temperature,
            });
        }
        if !(0.0 < self.top_p && self.top_p <= 1.0) {
            return Err(SamplingParamsError::TopP { got: self.top_p });
        }
        if !(0.0 < self.typical_p && self.typical_p <= 1.0) {
            return Err(SamplingParamsError::TypicalP {
                got: self.typical_p,
            });
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(SamplingParamsError::MinP { got: self.min_p });
        }
        if self.repetition_penalty <= 0.0 {
            return Err(SamplingParamsError::RepetitionPenalty {
                got: self.repetition_penalty,
            });
        }

        // Empty vocabulary constraints cannot express a valid sampling set.
        if self.allowed_token_ids.as_ref().is_some_and(Vec::is_empty) {
            return Err(SamplingParamsError::EmptyAllowedTokenIds);
        }
        if let Some(index) = self.bad_words_ids.iter().position(Vec::is_empty) {
            return Err(SamplingParamsError::EmptyBadWord { index });
        }
        Ok(())
    }
}

impl Default for SamplingParams {
    /// Defaults direct construction to greedy sampling.
    ///
    /// Server request lowering does not inherit this temperature: text
    /// requests start from the checkpoint's sampling defaults (temperature
    /// `1.0` when it declares none), omni requests start from fixed profile
    /// defaults, and both then apply the caller's explicit values.
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            ignore_eos: false,
            seed: None,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: Vec::new(),
            min_tokens: 0,
            return_logprobs: false,
            n_logprobs: 0,
            return_prompt_logprobs: false,
            n_prompt_logprobs: 0,
            bad_words_ids: Vec::new(),
            allowed_token_ids: None,
            typical_p: 1.0,
            forced_token_ids: Vec::new(),
        }
    }
}

/// Image-generation parameters bounded for scheduler admission.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageParams {
    /// Number of denoising steps.
    pub steps: u16,
    /// Classifier-free guidance scale for text conditioning.
    pub cfg_text_scale: f32,
    /// Classifier-free guidance scale for image conditioning.
    pub cfg_img_scale: f32,
    /// Guidance renormalization strategy.
    pub cfg_renorm_type: CfgRenorm,
    /// Lower bound applied by guidance renormalization.
    pub cfg_renorm_min: f32,
    /// Inclusive normalized step interval where guidance is active.
    pub cfg_interval: (f32, f32),
    /// Shift applied to the diffusion timestep schedule.
    pub timestep_shift: f32,
    /// Output height in pixels.
    pub height: u32,
    /// Output width in pixels.
    pub width: u32,
    /// Optional deterministic diffusion seed.
    pub seed: Option<u64>,
}

impl ImageParams {
    /// Returns the classifier-free-guidance branch count implied by both axes.
    ///
    /// A scale of `1.0` (within a relative tolerance) disables its axis. The
    /// count is one conditional branch plus one per active axis, so it lies in
    /// `1..=3`.
    pub fn cfg_branch_count(&self) -> u8 {
        1 + u8::from(self.guides_text()) + u8::from(self.guides_image())
    }

    /// Whether text guidance is active: its text-unconditional branch is
    /// evaluated beside the conditioned one.
    pub fn guides_text(&self) -> bool {
        !scale_approx(self.cfg_text_scale, 1.0)
    }

    /// Whether image guidance is active: its image-unconditional branch is
    /// evaluated beside the conditioned one.
    pub fn guides_image(&self) -> bool {
        !scale_approx(self.cfg_img_scale, 1.0)
    }

    /// Text and image guidance scales whose [`Self::cfg_branch_count`] is
    /// `branches`: no guidance for one branch, text guidance for two, text
    /// and image guidance for three. `None` outside `1..=3`.
    pub fn guidance_scales(branches: u8) -> Option<(f32, f32)> {
        match branches {
            1 => Some((1.0, 1.0)),
            2 => Some((4.0, 1.0)),
            3 => Some((4.0, 2.0)),
            _ => None,
        }
    }
}

/// Returns whether two guidance scales are equal within relative tolerance.
fn scale_approx(left: f32, right: f32) -> bool {
    (left - right).abs() <= 1.0e-6_f32 * left.abs().max(right.abs()).max(1.0)
}

/// Validation failure returned by [`ImageParams::validate`].
///
/// Each variant names the offending field and the bound it violated so the
/// caller can report a precise input error before scheduling diffusion work.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ImageParamsError {
    /// Denoising step count is zero.
    #[error("steps must be positive")]
    Steps,
    /// Image height is zero or not a multiple of the grid.
    #[error("height must be a non-zero multiple of {multiple}, got {got}")]
    Height {
        /// Supplied height.
        got: u32,
        /// Required height alignment.
        multiple: u32,
    },
    /// Image width is zero or not a multiple of the grid.
    #[error("width must be a non-zero multiple of {multiple}, got {got}")]
    Width {
        /// Supplied width.
        got: u32,
        /// Required width alignment.
        multiple: u32,
    },
    /// A guidance scale is non-finite or negative.
    #[error("{field} must be finite and non-negative, got {got}")]
    CfgScale {
        /// Name of the invalid scale.
        field: &'static str,
        /// Supplied scale.
        got: f32,
    },
    /// A floating-point parameter is NaN or infinite.
    #[error("{field} must be finite, got {got}")]
    NonFinite {
        /// Name of the invalid parameter.
        field: &'static str,
        /// Supplied non-finite value.
        got: f32,
    },
    /// Guidance interval endpoints are descending.
    #[error("cfg_interval must be an ordered pair, got ({lo}, {hi})")]
    CfgIntervalOrder {
        /// Supplied lower endpoint.
        lo: f32,
        /// Supplied upper endpoint.
        hi: f32,
    },
}

/// Classifier-free guidance renormalization strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CfgRenorm {
    /// Leaves classifier-free guidance unnormalized.
    None,
    /// Renormalizes the combined guidance prediction globally.
    #[default]
    Global,
    /// Renormalizes against the text-conditioned prediction.
    TextChannel,
}

impl CfgRenorm {
    /// Returns the stable wire name for this strategy.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Global => "global",
            Self::TextChannel => "text_channel",
        }
    }
}

impl std::str::FromStr for CfgRenorm {
    type Err = CfgRenormParseError;

    /// Parses a stable guidance-renormalization wire name.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "global" => Ok(Self::Global),
            "text_channel" => Ok(Self::TextChannel),
            _ => Err(CfgRenormParseError(value.to_owned())),
        }
    }
}

impl std::fmt::Display for CfgRenorm {
    /// Writes the stable guidance-renormalization wire name.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Error returned for an unsupported guidance renormalization string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported CFG renormalization mode {0:?}")]
pub struct CfgRenormParseError(String);

impl ImageParams {
    /// Required divisibility for each image dimension.
    pub const DIM_MULTIPLE: u32 = 16;

    /// Validates diffusion parameters before they reach the worker.
    ///
    /// Callers include `GenerationRequest::validate` and the worker IPC
    /// `NewRequest::validate`. `height` and `width` must each be non-zero
    /// and a multiple of `DIM_MULTIPLE`. The model's own limits bound the
    /// image's extent and the schedule's length at admission. The first
    /// violation found is returned.
    pub fn validate(&self) -> Result<(), ImageParamsError> {
        if self.steps == 0 {
            return Err(ImageParamsError::Steps);
        }

        // Both dimensions obey the same positive, aligned grid contract.
        let check_dim = |got: u32| -> bool { got != 0 && got.is_multiple_of(Self::DIM_MULTIPLE) };
        if !check_dim(self.height) {
            return Err(ImageParamsError::Height {
                got: self.height,
                multiple: Self::DIM_MULTIPLE,
            });
        }
        if !check_dim(self.width) {
            return Err(ImageParamsError::Width {
                got: self.width,
                multiple: Self::DIM_MULTIPLE,
            });
        }

        // Each guidance axis accepts finite non-negative scaling.
        for (field, scale) in [
            ("cfg_text_scale", self.cfg_text_scale),
            ("cfg_img_scale", self.cfg_img_scale),
        ] {
            if !scale.is_finite() || scale < 0.0 {
                return Err(ImageParamsError::CfgScale { field, got: scale });
            }
        }

        // Remaining continuous parameters must be finite before interval order
        // is evaluated.
        for (field, value) in [
            ("cfg_renorm_min", self.cfg_renorm_min),
            ("cfg_interval.0", self.cfg_interval.0),
            ("cfg_interval.1", self.cfg_interval.1),
            ("timestep_shift", self.timestep_shift),
        ] {
            if !value.is_finite() {
                return Err(ImageParamsError::NonFinite { field, got: value });
            }
        }

        if self.cfg_interval.0 > self.cfg_interval.1 {
            return Err(ImageParamsError::CfgIntervalOrder {
                lo: self.cfg_interval.0,
                hi: self.cfg_interval.1,
            });
        }
        Ok(())
    }
}

impl Default for ImageParams {
    /// Returns a bounded single-image diffusion configuration.
    fn default() -> Self {
        Self {
            steps: 50,
            cfg_text_scale: 4.0,
            cfg_img_scale: 1.0,
            cfg_renorm_type: CfgRenorm::Global,
            cfg_renorm_min: 0.0,
            cfg_interval: (0.0, 1.0),
            timestep_shift: 1.0,
            height: 512,
            width: 512,
            seed: None,
        }
    }
}

/// History retention policy of a cache group.
///
/// A model whose layers read different histories (for example full and
/// sliding-window attention) reports one group per policy and page shape.
/// Every group draws its pages from the same unit pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CacheGroupKind {
    /// Full attention: every page is retained for the request's lifetime.
    #[default]
    Full,
    /// Sliding-window attention. A reader starting at absolute position `q`
    /// needs the history `[q - window, q)`, so pages that lie entirely before
    /// the oldest position any remaining reader needs are released while the
    /// request continues. `window` is the longest history any reader of the
    /// group needs, not the checkpoint's nominal window.
    SlidingWindow {
        /// History tokens any reader of the group needs. A table keeps one
        /// contiguous run of pages, so no prefix is retained outside the
        /// window.
        window: u32,
    },
}

impl CacheGroupKind {
    /// Returns the history bound in tokens, or `None` for full attention.
    pub const fn window(self) -> Option<u32> {
        match self {
            Self::Full => None,
            Self::SlidingWindow { window } => Some(window),
        }
    }
}

/// One cache group of a worker's unit pool, reported at handshake.
///
/// A logical page of the group holds `page_tokens` tokens of every layer in
/// the group and occupies `units_per_page` physical units. Unit and column
/// geometry belong to the worker; the scheduler only needs the page shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheGroup {
    /// History retention policy of the group.
    pub kind: CacheGroupKind,
    /// Tokens stored per logical page; a power of two.
    pub page_tokens: u32,
    /// Physical units one logical page occupies.
    pub units_per_page: u32,
    /// Global cache-layer ids of this rank's layers in the group, in the
    /// column order the worker stores them.
    pub layer_ids: Vec<u32>,
    /// KV heads this rank stores per layer.
    pub num_kv_heads: u32,
    /// Logical KV heads of each layer across all tensor-parallel members.
    pub total_kv_heads: u32,
    /// First logical KV head stored by this rank.
    pub kv_head_offset: u32,
    /// Elements stored per KV head.
    pub head_dim: u32,
}

impl CacheGroup {
    /// Whether two groups share retention policy, page shape and logical head
    /// geometry, whatever layers and heads each rank stores.
    pub fn same_page_shape(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.page_tokens == other.page_tokens
            && self.units_per_page == other.units_per_page
            && self.total_kv_heads == other.total_kv_heads
            && self.head_dim == other.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct construction stays greedy; request lowering, not this default,
    /// supplies the API temperature.
    #[test]
    fn sampling_default_is_greedy() {
        let sp = SamplingParams::default();
        assert_eq!(
            sp.temperature, 0.0,
            "core default must stay greedy (0.0); unset->1.0 is resolved in lowering, not here"
        );
        assert_eq!(sp.top_p, 1.0);
        assert_eq!(sp.repetition_penalty, 1.0);
        assert_eq!(sp.top_k, 0);
    }

    #[test]
    fn image_params_default_is_valid() {
        assert!(ImageParams::default().validate().is_ok());
    }

    /// The scales of a branch count evaluate that many guidance branches.
    #[test]
    fn guidance_scales_realize_their_branch_count() {
        for branches in 1..=3 {
            let (cfg_text_scale, cfg_img_scale) = ImageParams::guidance_scales(branches).unwrap();
            let image = ImageParams {
                cfg_text_scale,
                cfg_img_scale,
                ..ImageParams::default()
            };
            assert_eq!(image.cfg_branch_count(), branches);
            assert!(image.validate().is_ok());
        }
        assert_eq!(ImageParams::guidance_scales(0), None);
        assert_eq!(ImageParams::guidance_scales(4), None);
    }

    #[test]
    fn image_params_validate_rejects_out_of_bounds() {
        let bad_steps = ImageParams {
            steps: 0,
            ..Default::default()
        };
        assert!(matches!(bad_steps.validate(), Err(ImageParamsError::Steps)));

        let zero_height = ImageParams {
            height: 0,
            ..Default::default()
        };
        assert!(matches!(
            zero_height.validate(),
            Err(ImageParamsError::Height { .. })
        ));

        let unaligned_width = ImageParams {
            width: 513, // not a multiple of DIM_MULTIPLE
            ..Default::default()
        };
        assert!(matches!(
            unaligned_width.validate(),
            Err(ImageParamsError::Width { .. })
        ));

        let nan_scale = ImageParams {
            cfg_text_scale: f32::NAN,
            ..Default::default()
        };
        assert!(matches!(
            nan_scale.validate(),
            Err(ImageParamsError::CfgScale { .. })
        ));

        let neg_scale = ImageParams {
            cfg_img_scale: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            neg_scale.validate(),
            Err(ImageParamsError::CfgScale { .. })
        ));
    }

    #[test]
    fn model_dtype_uses_canonical_strings() {
        assert_eq!(
            serde_json::to_value(ModelDtype::Float16).unwrap(),
            serde_json::json!("float16")
        );
        assert_eq!(ModelDtype::parse("bfloat16"), Some(ModelDtype::BFloat16));
        assert_eq!(ModelDtype::parse("float32"), Some(ModelDtype::Float32));
    }
}
