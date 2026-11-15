//! The request vocabulary: what a daemon submits to an engine process and
//! the events it receives back.
//!
//! These are the values both ends work with: the daemon builds and checks
//! requests with them before submitting, the engine schedules them, and the
//! engine's interface with its workers (`ipc`) carries the sampling values on
//! to the ranks. Between a daemon and an engine process they travel as the
//! request interface's Protobuf messages.

mod convert;
mod dtype;
mod event;
mod generation;
mod sampling;
mod spec;
mod values;

pub use convert::WireError;
pub use dtype::{ModelDtype, ModelDtypeParseError};
pub use event::{
    ArtifactEvent, ArtifactMedia, ConditionExtent, DiffusionRequest, DiffusionRequestError,
    EncoderInputs, Event, FinishReason, MediaKind, MigrationCause, PositionLogprobs, RejectionKind,
    Request, StopReason, Trajectory,
};
pub use generation::{
    CachePolicy, CacheResume, GenerationFeatures, GenerationLimits, GenerationRequest,
    GenerationRequestError, GenerationResourceError, ImageEncoderInput, ImageInput,
    MultimodalInputs,
};
pub use sampling::{SamplingParams, SamplingParamsError};
pub use spec::{
    ExpertExchange, FlashInferBackend, FlashInferBackendParseError, PrefixCacheDtype,
    PrefixCacheDtypeParseError, SchedulingPolicy, StandbyStorage,
};
pub use values::{DiffusionSamplingParams, ImageEncoder, RequestId, TokenLogprob};

/// The most events the engine sends a request beyond those its consumer took
/// (`Consumed`).
pub const EVENT_WINDOW: u64 = 64;

/// The messages of the request interface (`request.proto`).
#[allow(clippy::all, reason = "code generated from the schema")]
pub mod wire {
    include!(concat!(env!("OUT_DIR"), "/uniserve.request.rs"));
}

/// A worker group's prefix cache index, as the process interface carries it too
/// (`process::Drained`, `process::Inherit`).
pub use wire::GroupIndex;
