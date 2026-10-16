//! Warmup of the rank's decoder runners.

use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::warmup::WarmupHooks;

/// The warmup work of the rank's decoder runners.
pub(crate) struct DecoderWarmup;

impl WarmupHooks for DecoderWarmup {
    fn prepare(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        let lanes = backend.job_queues().clone();
        if backend.serves_video() {
            let parts = backend.media_parts()?;
            parts
                .decoders
                .prepare_startup(&parts.video.model, &parts.video.plan, parts.rank, &lanes)
                .map_err(|error| WorkerError::classify(error, Some("warmup")))?;
        } else if let Some(image) = backend.parts().image {
            let layout = image.model.denoiser;
            let side = layout.map_or(64, |layout| 2 * layout.downsample);
            image
                .decoders
                .warm_image(image.latents.as_deref(), layout, side, &lanes)
                .map_err(|error| {
                    WorkerError::classify(error, Some("image warmup")).with_phase("warmup")
                })?;
        }
        Ok(())
    }

    fn capture(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        if backend.serves_video() {
            let lanes = backend.job_queues().clone();
            let parts = backend.media_parts()?;
            parts
                .decoders
                .capture_startup(&lanes)
                .map_err(|error| WorkerError::classify(error, Some("capture")))?;
        }
        Ok(())
    }
}
