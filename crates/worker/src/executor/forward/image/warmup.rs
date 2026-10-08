//! Image warmup: encoder and decoder warm runs, packed vision capture.
//!
//! Unpacked encoders and the image decoder each run once on a black image
//! (two latent patches on a side), in binding order, so their kernels are
//! selected and compiled before serving; a latent encoder sampling its
//! posterior draws from the device's default generator there as serving
//! startup always has. Every packed vision encoder then captures its slot
//! counts `capacity..=1` into the graph pool reserved for it at startup
//! (largest first, so smaller captures reuse pool blocks). Serving replays
//! them; the image denoiser runs eagerly.

use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::warmup::WarmupHooks;

/// The image runtime's warmup work.
pub(crate) struct ImageWarmup;

impl WarmupHooks for ImageWarmup {
    fn prepare(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        let Some(runtime) = backend.image_mut() else {
            return Ok(());
        };
        runtime.warm().map_err(|error| {
            WorkerError::classify(error, Some("image warmup")).with_phase("warmup")
        })
    }

    fn capture(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        let Some(runtime) = backend.image_mut() else {
            return Ok(());
        };
        runtime.capture().map_err(|error| {
            WorkerError::classify(error, Some("image graph capture")).with_phase("warmup")
        })
    }
}
