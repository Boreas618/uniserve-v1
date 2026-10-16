//! Warmup of the rank's denoiser runners.

use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::warmup::WarmupHooks;

/// The warmup work of the rank's denoiser runners.
pub(crate) struct DenoiserWarmup;

impl WarmupHooks for DenoiserWarmup {
    fn prepare(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        if backend.serves_video() {
            let lanes = backend.job_queues().clone();
            let parts = backend.media_parts()?;
            parts
                .denoisers
                .prepare_startup(&parts.video.model, &parts.video.plan, &lanes)
                .map_err(|error| WorkerError::classify(error, Some("warmup")))?;
        }
        Ok(())
    }

    fn capture(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        if backend.serves_video() {
            let lanes = backend.job_queues().clone();
            let parts = backend.media_parts()?;
            parts
                .denoisers
                .capture_startup(&parts.video.model, &parts.video.plan, &lanes)
                .map_err(|error| WorkerError::classify(error, Some("capture")))?;
        } else {
            let lanes = backend.job_queues().clone();
            let parts = backend.parts();
            if let Some(image) = parts.image {
                let prefix_cache = parts
                    .storage
                    .and_then(|storage| storage.prefix_cache.as_mut());
                image
                    .denoisers
                    .capture_flow(image.latents.as_deref(), prefix_cache, &lanes)
                    .map_err(|error| {
                        WorkerError::classify(error, Some("image graph capture"))
                            .with_phase("warmup")
                    })?;
            }
        }
        Ok(())
    }
}
