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
        } else {
            let parts = backend.parts();
            if let Some(image) = parts.image {
                let kv = parts.storage.and_then(|storage| storage.kv.as_mut());
                image
                    .denoisers
                    .capture_flow(image.latents.as_deref(), kv)
                    .map_err(|error| {
                        WorkerError::classify(error, Some("image graph capture"))
                            .with_phase("warmup")
                    })?;
            }
        }
        Ok(())
    }
}
