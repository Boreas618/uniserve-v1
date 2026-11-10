//! Request processing's model contract.
//!
//! The daemon turns public API requests into engine requests. What it does for every model is
//! shared: it fetches and probes media, tokenizes, publishes condition media, and schedules. What
//! depends on the model is the model family's, and it reaches the daemon through this contract:
//!
//! - a family states how its checkpoints declare it and what it serves ([`Family`], [`Serving`]);
//!   the numerical family its model crate exports (`nn::family::Family`) holds that value beside
//!   what execution loads;
//! - a chat family names the output formats its checkpoints write ([`ChatProfile`]);
//! - a video family binds a [`VideoProcessor`] once its workers described the placed denoiser
//!   ([`VideoFamily::bind`]); the processor validates a model-agnostic request ([`VideoSpec`])
//!   and plans it ([`VideoPlan`]).
//!
//! The executable registers the families it serves once; request processing reads them as one
//! [`Catalogue`], and nothing in the daemon or the frontend names a model.
//!
//! The family-neutral rules a video request is planned with are stated here once, each
//! parameterized by values the family states: the frame rule ([`FrameClips`]), the canvas rule
//! ([`CanvasRule`]), the cover crop ([`cover_crop`]) and the resampled length of a track
//! ([`resampled_len`]).

mod canvas;
mod family;
mod frames;
mod media;
mod serving;
mod video;
mod vocabulary;

pub use canvas::{CanvasError, CanvasRule, CoverCrop, cover_crop};
pub use family::{Catalogue, ClaimError, Family};
pub use frames::{ClipSpan, FrameClips, requested_frames};
pub use media::{
    AudioMetadata, FrameRate, ImageMetadata, MediaMetadata, VideoMetadata, resampled_len,
};
pub use serving::{ChatProfile, ReasoningFormat, Serving, ToolCallFormat};
pub use video::{
    ConditionPlan, InputError, Length, Reference, ReferenceKind, VideoBinding, VideoFamily,
    VideoPlan, VideoProcessor, VideoSpec,
};
pub use vocabulary::{Vocabulary, VocabularyError};
