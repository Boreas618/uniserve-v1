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
pub use request::{AdapterSelection, GenerateRequest, ImageSamplingParams, Prompt, SamplingParams};

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

/// Folds a request's cache salt and the identity of the LoRA adapter it
/// computes with into a stable isolation key.
///
/// Returns `None` when both are absent, which leaves the request in the
/// shared cache partition. A salt alone keys by its 64-bit FNV-1a hash, which
/// is deterministic across processes; an adapter keys apart from every salt
/// and every other adapter, since its cache entries differ from the base
/// model's.
pub(crate) fn cache_isolation_key(salt: Option<&str>, adapter: Option<&str>) -> Option<u64> {
    match (salt, adapter) {
        (None, None) => None,
        (Some(salt), None) => Some(stable_hash(salt.as_bytes())),
        (salt, Some(adapter)) => {
            let mut bytes = b"uniserve-adapter\0".to_vec();
            bytes.extend_from_slice(adapter.as_bytes());
            bytes.push(0);
            bytes.extend_from_slice(salt.unwrap_or_default().as_bytes());
            Some(stable_hash(&bytes))
        }
    }
}
