//! Warmup of the rank's encoder runners.

use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::warmup::WarmupHooks;

/// The warmup work of the rank's encoder runners.
pub(crate) struct EncoderWarmup;

impl WarmupHooks for EncoderWarmup {
    fn capture(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        if backend.serves_video() {
            let parts = backend.media_parts()?;
            parts
                .encoders
                .capture_text()
                .map_err(|error| WorkerError::classify(error, Some("capture")))?;
        }
        Ok(())
    }
}
