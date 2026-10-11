//! Warmup of the rank's denoiser runners.

use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::warmup::WarmupHooks;

/// The warmup work of the rank's denoiser runners.
pub(crate) struct DenoiserWarmup;

impl WarmupHooks for DenoiserWarmup {
    fn prepare(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        if backend.serves_video() {
            let parts = backend.media_parts()?;
            parts
                .denoisers
                .prepare_startup(&parts.video.model, &parts.video.plan)
                .map_err(|error| WorkerError::classify(error, Some("warmup")))?;
        }
        Ok(())
    }

    fn capture(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        if backend.serves_video() {
            let parts = backend.media_parts()?;
            parts
                .denoisers
                .capture_startup(&parts.video.model, &parts.video.plan)
                .map_err(|error| WorkerError::classify(error, Some("capture")))?;
        }
        Ok(())
    }
}
