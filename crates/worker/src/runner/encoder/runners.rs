//! The encoder runners of a rank.

use std::sync::Arc;

use super::EncoderRunner;
use crate::Result;
use crate::cuda_graph::GraphPool;
use crate::ffi::{Handle, Record};
use crate::runner::component::Component;

/// The encoder runners of a rank.
#[derive(Default)]
pub(crate) struct EncoderRunners {
    /// A video model's text encoder.
    pub(crate) text: Option<Component<EncoderRunner>>,
    /// The vision tower bound with the text encoder, for conditioned tasks.
    pub(crate) vision: Option<EncoderRunner>,
    /// The text encoder's graph pool, when graphs are captured.
    pub(crate) text_pool: Option<Arc<GraphPool<Handle>>>,
    /// A video model's condition latent encoders.
    pub(crate) latent: Option<Component<EncoderRunner>>,
}

impl EncoderRunners {
    /// Capture the text encoder's prompt capacities, then refuse further
    /// captures.
    pub(crate) fn capture_text(&mut self) -> Result<()> {
        if let Some(text) = &mut self.text {
            // Prompt capacities are warmed and captured largest first, the
            // same order on every rank of a tensor-parallel encoder.
            text.runner
                .capture(self.text_pool.as_ref(), &mut || Ok(()))?;
            if let Some(stream) = text.runner.stream() {
                stream.synchronize()?;
            }
            text.runner.seal();
            if let Some(vision) = &mut self.vision {
                vision.seal();
            }
        }
        Ok(())
    }

    /// The kernels the text encoder's operators selected.
    pub(crate) fn text_kernels(&self) -> Result<Vec<(String, Vec<Record>)>> {
        let mut kernels = Vec::new();
        if let Some(text) = &self.text {
            kernels.push((text.name.clone(), text.runner.kernels(&text.name)?));
        }
        Ok(kernels)
    }

    /// The kernels the condition latent encoders' operators selected.
    pub(crate) fn latent_kernels(&self) -> Result<Vec<(String, Vec<Record>)>> {
        let mut kernels = Vec::new();
        if let Some(latent) = &self.latent {
            kernels.push((latent.name.clone(), latent.runner.kernels(&latent.name)?));
        }
        Ok(kernels)
    }

    /// Release the text encoder and its vision tower. The tower computes
    /// with the text encoder's binding, which the text encoder releases.
    pub(crate) fn close_text(&mut self, aborted: bool) -> Result<()> {
        let mut result = Ok(());
        if let Some(vision) = &mut self.vision {
            result = result.and(vision.close(aborted));
        }
        if let Some(text) = &mut self.text {
            result = result.and(text.runner.close(aborted));
        }
        result
    }

    /// Release the condition latent encoders.
    pub(crate) fn close_latent(&mut self, aborted: bool) -> Result<()> {
        match &mut self.latent {
            Some(latent) => latent.runner.close(aborted),
            None => Ok(()),
        }
    }
}
