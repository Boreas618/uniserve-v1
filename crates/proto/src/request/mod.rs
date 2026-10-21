//! The request vocabulary: what a daemon submits to an engine process and
//! the events it receives back.
//!
//! These are the values both ends work with: the daemon builds and checks
//! requests with them before submitting, the engine schedules them, and the
//! engine's interface with its workers (`ipc`) carries the sampling values on
//! to the ranks. Between a daemon and an engine process they travel as the
//! request interface's Protobuf messages.

mod event;
mod generation;
mod image;
mod sampling;
mod values;

pub use event::{
    ArtifactEvent, ArtifactMedia, ConditionExtent, DiffusionRequest, DiffusionRequestError, Event,
    FinishReason, MediaKind, PositionLogprobs, RejectionKind, Request, StopReason,
};
pub use generation::{
    CachePolicy, FeedbackNextToken, FeedbackSource, GenerationConstraint,
    GenerationConstraintParseError, GenerationFeatures, GenerationLimits, GenerationRequest,
    GenerationRequestError, GenerationResourceError, ImageEncoderInput, ImageGenerationConfig,
    ImageInput, ImageTrigger, MultimodalInputs, ReadoutRow, ReadoutSlot, encoder_cache_key,
};
pub use image::{CfgRenorm, CfgRenormParseError, ImageParams, ImageParamsError};
pub use sampling::{SamplingParams, SamplingParamsError};
pub use values::{DiffusionSamplingParams, ImageEncoder, RequestId, TokenLogprob};
