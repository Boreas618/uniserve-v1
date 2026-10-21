//! Input processing: what turns an API request into the request the engine
//! serves.
//!
//! [`InputProcessor`] renders and tokenizes a request against the loaded
//! model: chat templates (`chat`), sampling controls (`sampling`), images
//! (`media`), and the lowering of each model family's requests (`omni`,
//! `diffusion_gemma`, `video`, the System One readout `readout`). Each
//! request leaves with its engine request and the parameters its output
//! processing takes (`OutputParams`, `crate::output`).

pub mod chat;
pub(crate) mod diffusion_gemma;
mod error;
pub mod media;
mod omni;
mod processor;
pub mod readout;
mod request;
mod sampling;
pub mod video;

pub use error::{ServeError, TokenizeError};
pub use media::ImageInput;
pub use processor::{InputProcessor, ModelResolutionError, RequestRenderer};
pub(crate) use processor::{LoadedModel, LoadedRenderer};
pub use request::{GenerateRequest, ImageSamplingParams, Prompt, SamplingParams};

/// Result type returned by serving-runtime calls.
pub type Result<T> = std::result::Result<T, ServeError>;

/// The 64-bit FNV-1a hash of `bytes`, stable across processes, builds and
/// platforms. It keys cache isolation and request-derived seeds; it is not
/// collision resistant.
pub(crate) fn stable_hash(bytes: &[u8]) -> u64 {
    use std::hash::Hasher as _;

    let mut hasher = fnv::FnvHasher::default();
    hasher.write(bytes);
    hasher.finish()
}

/// Folds a request's cache salt into a stable isolation key.
///
/// Returns `None` when the salt is absent, which leaves the request in the shared cache
/// partition. Otherwise the key is the 64-bit FNV-1a hash of the salt, which is
/// deterministic across processes.
pub(crate) fn cache_isolation_key(salt: Option<&str>) -> Option<u64> {
    salt.map(|salt| stable_hash(salt.as_bytes()))
}
