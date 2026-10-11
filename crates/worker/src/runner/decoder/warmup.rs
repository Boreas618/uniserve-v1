//! Warmup of the rank's decoder runners.

use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::warmup::WarmupHooks;

/// The warmup work of the rank's decoder runners.
pub(crate) struct DecoderWarmup;

impl WarmupHooks for DecoderWarmup {
    fn prepare(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        if backend.serves_video() {
            let parts = backend.media_parts()?;
            parts
                .decoders
                .prepare_startup(&parts.video.model, &parts.video.plan, parts.rank)
                .map_err(|error| WorkerError::classify(error, Some("warmup")))?;
        }
        Ok(())
    }

    fn capture(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        if backend.serves_video() {
            let parts = backend.media_parts()?;
            parts
                .decoders
                .capture_startup()
                .map_err(|error| WorkerError::classify(error, Some("capture")))?;
        }
        Ok(())
    }
}
