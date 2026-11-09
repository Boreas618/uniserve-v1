//! Request processing's model contract.
//!
//! The daemon turns public API requests into engine requests. What it does for every model is
//! shared: it fetches and probes media, tokenizes, publishes condition media, and schedules. What
//! depends on the model is the model family's, and it reaches the daemon through this contract:
//!
//! - a [`Family`] claims the classes its checkpoints declare and states what it serves
//!   ([`Serving`]);
//! - a chat family names the output formats its checkpoints write ([`ChatProfile`]);
//! - a video family binds a [`VideoProcessor`] once its workers described the placed denoiser
//!   ([`VideoFamily::bind`]); the processor validates a model-agnostic request ([`VideoSpec`])
//!   and plans it ([`VideoPlan`]).
//!
//! The executable registers the families it serves ([`Catalogue`]); nothing in the daemon names a
//! model.

mod catalogue;
mod media;
mod video;
mod vocabulary;

pub use catalogue::{Catalogue, ChatProfile, Family, ReasoningFormat, Serving, ToolCallFormat};
pub use media::{AudioMetadata, FrameRate, ImageMetadata, MediaMetadata, VideoMetadata};
pub use video::{
    ConditionPlan, InputError, Length, Reference, ReferenceKind, VideoBinding, VideoFamily,
    VideoPlan, VideoProcessor, VideoSpec,
};
pub use vocabulary::{Vocabulary, VocabularyError};
