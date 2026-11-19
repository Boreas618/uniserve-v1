//! Request processing: what a model family supplies to turn public API requests into engine
//! requests.
//!
//! The daemon turns public API requests into engine requests. What it does for every model is
//! shared: it fetches and probes media, tokenizes, publishes condition media, and schedules. What
//! depends on the model is the model family's, and it reaches the daemon through these interfaces:
//!
//! - a family states how its checkpoints declare it and what it serves ([`Family`], [`Serving`]);
//!   the numerical family its model crate exports (`nn::family::Family`) holds that value beside
//!   what execution loads;
//! - a chat family's profile names the output formats its checkpoints write, each named for the
//!   format and not for a model, and, when it takes images, the processor that sizes their
//!   blocks ([`ChatProfile`], [`ReasoningFormat`], [`ToolCallFormat`], [`ImageFamily`]);
//! - a video family binds a [`VideoProcessor`] once its workers described the placed denoiser
//!   ([`VideoFamily::bind`]); the processor validates a model-agnostic request ([`VideoSpec`])
//!   and processes it into the engine's input ([`VideoInput`]).
//!
//! The crate is also the machinery of the **processing program**, which serves [`FAMILIES`]
//! through the processing interface (`scheduling::processing`) as a module of its own crate
//! (`crates/processing/program`, built for `wasm32-unknown-unknown`): the leader daemon loads the
//! module its epoch's module set fixes, so an epoch's request side is built from the same tree as
//! its engine and nothing in the daemon names a model ([`program`]). The request side of chat
//! lives here with the families: the Jinja chat template a request renders with, or the encoder of
//! a family whose checkpoints state their prompt format as a reference encoder instead
//! ([`chat`]), and the parsers of a response's reasoning and tool calls ([`output`]). The engine
//! links the crate natively for the numerical facts its models read, and registers the numerical
//! side of the same families.
//!
//! The family-neutral rules a video request is processed with are stated here once, each
//! parameterized by values the family states: the frame rule ([`FrameClips`]), the canvas rule
//! ([`CanvasRule`]), the cover crop ([`cover_crop`]) and the resampled length of a track
//! ([`resampled_len`]).
//!
//! As in the reference serving systems, a family's own request preprocessing is a stage beside its
//! model's numerics rather than part of them, and lives here with the request facts it states once:
//! [`qwen`] (the Qwen families' chat profiles), [`qwen_vl`] (the Qwen-VL image and video processor
//! rules), [`deepseek_v41`] (DeepSeek-V4.1's chat profile and prompt encoder) and [`minimax_h3`]
//! (MiniMax-H3's request rules, conditioner prompt and processor). The model crates read those
//! facts from here.

mod canvas;
pub mod chat;
pub mod deepseek_v41;
#[cfg(target_arch = "wasm32")]
mod entropy;
mod family;
mod frames;
mod image;
mod media;
pub mod minimax_h3;
pub mod output;
pub mod program;
mod python_json;
pub mod qwen;
pub mod qwen_vl;
mod serving;
#[cfg(test)]
mod testing;
mod video;
mod vocabulary;

pub use canvas::{CanvasError, CanvasRule, CoverCrop, cover_crop};
pub use family::{FAMILIES, Family, Registry, RegistryError, ResolveError, RootFile};
pub use frames::{ClipSpan, FrameClips, requested_frames};
pub use image::{ImageBlock, ImageFamily, ImageProcessor};
pub use media::{
    AudioMetadata, FrameRate, ImageMetadata, MediaMetadata, VideoMetadata, resampled_len,
};
pub use serving::{
    ChatCheckpoint, ChatProfile, ChatTemplate, Opening, ReasoningFormat, Serving, ToolCallFormat,
};
pub use video::{
    BasePin, ConditionInput, InputError, Length, Reference, ReferenceKind, VideoBinding,
    VideoFamily, VideoInput, VideoProcessor, VideoSpec,
};
pub use vocabulary::{Vocabulary, VocabularyError};
