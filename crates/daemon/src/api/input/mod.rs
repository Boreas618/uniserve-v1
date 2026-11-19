//! Input processing: what turns an API request into the request the engine
//! serves.
//!
//! [`Processor`] renders and tokenizes a request against the loaded model: its prompt rendered by
//! the epoch's processing program from the request as `chat` states it, sampling controls
//! (`sampling`), images (`media`), and the lowering of each model family's requests (`video`).
//! Each request leaves as its engine request, the lowering its output processing is built from
//! (`proto::handoff::Lowering`) and its instance of the program, which parses its response. A
//! request's cache partition is keyed by `storage::key::isolation`.

pub mod chat;
mod error;
pub mod media;
pub mod processor;
mod sampling;
pub mod video;

pub use error::{ServeError, TokenizeError};
pub use media::ImageInput;
pub use processor::{AdapterSelection, ModelResolutionError, Processor};
pub use processor::{LoadedBinding, LoadedModel};

/// Result type returned by serving-runtime calls.
pub type Result<T> = std::result::Result<T, ServeError>;

/// The seed a request is served with: the one it states, or one drawn at random when it states
/// none, for every kind of request alike. A request carries its seed from entry on, so its draws
/// repeat wherever it runs, on every engine a migration moves it to.
pub(crate) fn request_seed(stated: Option<u64>) -> u64 {
    stated.unwrap_or_else(|| uuid::Uuid::new_v4().as_u64_pair().0)
}
