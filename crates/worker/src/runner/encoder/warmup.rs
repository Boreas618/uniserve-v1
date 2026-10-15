//! Warmup of the rank's encoder runners.

use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::warmup::WarmupHooks;

/// The warmup work of the rank's encoder runners.
pub(crate) struct EncoderWarmup;

impl WarmupHooks for EncoderWarmup {
    fn prepare(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        if let Some(image) = backend.parts().image {
            let side = image
                .model
                .denoiser
                .map_or(64, |layout| 2 * layout.downsample);
            image.encoders.warm_image(side).map_err(|error| {
                WorkerError::classify(error, Some("image warmup")).with_phase("warmup")
            })?;
        }
        Ok(())
    }

    fn capture(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        if backend.serves_video() {
            let parts = backend.media_parts()?;
            parts
                .encoders
                .capture_text()
                .map_err(|error| WorkerError::classify(error, Some("capture")))?;
        } else if let Some(image) = backend.parts().image {
            image.encoders.capture_packed().map_err(|error| {
                WorkerError::classify(error, Some("image graph capture")).with_phase("warmup")
            })?;
        }
        Ok(())
    }
}
