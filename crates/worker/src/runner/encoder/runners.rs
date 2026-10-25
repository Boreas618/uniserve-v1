//! The encoder runners of a rank.

use std::collections::HashMap;
use std::sync::Arc;

use ipc::CallKind;

use super::EncoderRunner;
use crate::cuda_graph::GraphPool;
use crate::lane::{BLOCK_ON_TIMEOUT, Completion, Job, JobQueues};
use crate::stream::Issue;
use crate::{Error, Result};

/// Perform a startup `job` on the lane and wait for its completion.
fn perform(lanes: &JobQueues, job: Job) -> Result<Completion> {
    lanes
        .block_on(job, BLOCK_ON_TIMEOUT)?
        .check()
        .map_err(|failure| failure.error)
}

/// An image model's encoder runners on a rank.
#[derive(Default)]
pub(crate) struct EncoderRunners {
    /// An image model's encoders, in construction order.
    pub(crate) image: Vec<EncoderRunner>,
    /// The image encoder serving each component's call kind.
    routes: HashMap<(String, CallKind), usize>,
    /// Graph pools of the image encoders that capture, by encoder.
    pools: HashMap<usize, Arc<GraphPool>>,
}

impl EncoderRunners {
    /// The encoders of an image model, routed by component and call kind,
    /// with the graph pools of those that capture.
    pub(crate) fn image(
        encoders: Vec<EncoderRunner>,
        routes: HashMap<(String, CallKind), usize>,
        pools: HashMap<usize, Arc<GraphPool>>,
    ) -> Self {
        Self {
            image: encoders,
            routes,
            pools,
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
    /// job of the lane on `lanes`, which also waits for the encoder's
    /// stream.
    pub(crate) fn warm_image(&self, side: u32, lanes: &JobQueues) -> Result<()> {
        for runner in &self.image {
            let job = Job::new();
            runner.warm(side, &job);
            job.on(runner.stream()).synchronize()?;
            perform(lanes, job)?;
        }
        Ok(())
    }

    /// Capture every image encoder's graphs into its pool (packed vision
    /// slot counts, latent image sizes), each forward a job of the lane on
    /// `lanes`.
    pub(crate) fn capture_image(&mut self, lanes: &JobQueues) -> Result<()> {
        for (index, runner) in self.image.iter_mut().enumerate() {
            let Some(pool) = self.pools.get(&index) else {
                continue;
            };
            runner.capture_with(Some(pool), &mut || Ok(()), &mut |job| perform(lanes, job))?;
        }
        Ok(())
    }

    /// The image sizes the latent encoder captured
    /// (`WorkerInfo::captured_latents`), which alone it encodes; `None` when
    /// it encodes eagerly or the rank holds none.
    pub(crate) fn captured_latents(&self) -> Option<Vec<[u32; 2]>> {
        self.image
            .iter()
            .find_map(|runner| runner.latent_sizes().map(<[_]>::to_vec))
    }

    /// Seal every image encoder's graphs.
    pub(crate) fn seal_image(&mut self) {
        for runner in &mut self.image {
            runner.seal();
        }
    }
}
