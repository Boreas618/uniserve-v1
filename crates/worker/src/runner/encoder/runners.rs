//! The encoder runners of a rank.

use std::collections::HashMap;
use std::sync::Arc;

use uniserve_ipc::{CallKind, MediaCall};

use super::EncoderRunner;
use crate::cuda_graph::GraphPool;
use crate::lane::{BLOCK_ON_TIMEOUT, Completion, Job, JobQueues};
use crate::profiling::kernels::KernelRecord;
use crate::runner::component::Component;
use crate::stream::Issue;
use crate::{Error, Result};

/// Perform a startup `job` on the lane serving `kind` and wait for its
/// completion.
fn perform(lanes: &JobQueues, kind: CallKind, job: Job) -> Result<Completion> {
    lanes
        .block_on(kind, job, BLOCK_ON_TIMEOUT)?
        .check()
        .map_err(|failure| failure.error)
}

/// The call kind image encoder `index` is routed for: the lane serving it
/// performs the encoder's startup jobs.
fn routed(routes: &HashMap<(String, CallKind), usize>, index: usize) -> Result<CallKind> {
    routes
        .iter()
        .find(|(_, routed)| **routed == index)
        .map(|((_, kind), _)| *kind)
        .ok_or_else(|| Error::Invariant("an image encoder serves no call kind".into()))
}

/// The encoder runners of a rank.
#[derive(Default)]
pub(crate) struct EncoderRunners {
    /// A video model's text encoder.
    pub(crate) text: Option<Component<EncoderRunner>>,
    /// The vision tower bound with the text encoder, for conditioned tasks.
    pub(crate) vision: Option<EncoderRunner>,
    /// The text encoder's graph pool, when graphs are captured.
    pub(crate) text_pool: Option<Arc<GraphPool>>,
    /// A video model's condition latent encoders.
    pub(crate) latent: Option<Component<EncoderRunner>>,
    /// An image model's encoders, in construction order.
    pub(crate) image: Vec<EncoderRunner>,
    /// The image encoder serving each component's call kind.
    routes: HashMap<(String, CallKind), usize>,
    /// Graph pools of packed vision encoders, by encoder.
    pools: HashMap<usize, Arc<GraphPool>>,
}

impl EncoderRunners {
    /// The encoders of an image model, routed by component and call kind,
    /// with the graph pools of the packed ones.
    pub(crate) fn image(
        encoders: Vec<EncoderRunner>,
        routes: HashMap<(String, CallKind), usize>,
        pools: HashMap<usize, Arc<GraphPool>>,
    ) -> Self {
        Self {
            image: encoders,
            routes,
            pools,
            ..Self::default()
        }
    }

    /// The image encoder serving `kind` calls of `component`.
    pub(crate) fn route(&self, component: &str, kind: CallKind) -> Result<usize> {
        self.routes
            .get(&(component.to_owned(), kind))
            .copied()
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "component {component:?} has no {} encoder",
                    kind.as_str()
                ))
            })
    }

    /// Warm every unpacked image encoder on one black `side` x `side`
    /// image, in construction order, as serving prepares its images: one
    /// job of the lane on `lanes` serving the call kind the encoder is
    /// routed for, which also waits for the encoder's stream.
    pub(crate) fn warm_image(&self, side: u32, lanes: &JobQueues) -> Result<()> {
        for (index, runner) in self.image.iter().enumerate() {
            let kind = routed(&self.routes, index)?;
            let job = Job::new();
            runner.warm(side, &job);
            job.on(runner.stream()).synchronize()?;
            perform(lanes, kind, job)?;
        }
        Ok(())
    }

    /// Capture every packed vision encoder's graphs into its pool, each
    /// forward a job of the lane on `lanes` serving the call kind the
    /// encoder is routed for.
    pub(crate) fn capture_packed(&mut self, lanes: &JobQueues) -> Result<()> {
        for (index, runner) in self.image.iter_mut().enumerate() {
            let Some(pool) = self.pools.get(&index) else {
                continue;
            };
            let kind = routed(&self.routes, index)?;
            runner.capture_with(Some(pool), &mut || Ok(()), &mut |job| {
                perform(lanes, kind, job)
            })?;
        }
        Ok(())
    }

    /// Seal every image encoder's graphs.
    pub(crate) fn seal_image(&mut self) {
        for runner in &mut self.image {
            runner.seal();
        }
    }

    /// Capture the text encoder's prompt capacities, each forward a job of
    /// the text encoder's lane on `lanes`, then refuse further captures
    /// once a last job waited for the stream.
    pub(crate) fn capture_text(&mut self, lanes: &JobQueues) -> Result<()> {
        if let Some(text) = &mut self.text {
            let kind = CallKind::Media(MediaCall::TextEncoding);
            // Prompt capacities are warmed and captured largest first, the
            // same order on every rank of a tensor-parallel encoder.
            text.runner
                .capture_with(self.text_pool.as_ref(), &mut || Ok(()), &mut |job| {
                    perform(lanes, kind, job)
                })?;
            let job = Job::new();
            job.on(text.runner.stream()).synchronize()?;
            perform(lanes, kind, job)?;
            text.runner.seal();
            if let Some(vision) = &mut self.vision {
                vision.seal();
            }
        }
        Ok(())
    }

    /// The kernels the text encoder's operators selected.
    pub(crate) fn text_kernels(&self) -> Result<Vec<(String, Vec<KernelRecord>)>> {
        let mut kernels = Vec::new();
        if let Some(text) = &self.text {
            kernels.push((text.name.clone(), text.runner.kernels()?));
        }
        Ok(kernels)
    }

    /// The kernels the condition latent encoders' operators selected.
    pub(crate) fn latent_kernels(&self) -> Result<Vec<(String, Vec<KernelRecord>)>> {
        let mut kernels = Vec::new();
        if let Some(latent) = &self.latent {
            kernels.push((latent.name.clone(), latent.runner.kernels()?));
        }
        Ok(kernels)
    }

    /// Release the text encoder and its vision tower. The tower computes
    /// with the text encoder's execution context, which the text encoder
    /// releases.
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
