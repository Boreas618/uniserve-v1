//! The denoiser runners of a rank.

use std::sync::Arc;

use super::DenoiserRunner;
use super::image::{ConditionRow, ConditionShape, DenoiseRow};
use crate::Result;
use crate::cuda_graph::GraphPool;
use crate::lane::{BLOCK_ON_TIMEOUT, Completion, Job, JobQueues};
use crate::memory::{LatentCache, PrefixCache, UnitColumn};
use crate::stream::Issue;

/// Perform a startup `job` on the lane and wait for its completion.
fn perform(lanes: &JobQueues, job: Job) -> Result<Completion> {
    lanes
        .block_on(job, BLOCK_ON_TIMEOUT)?
        .check()
        .map_err(|failure| failure.error)
}

/// Prefix tokens of each scratch slot a captured denoising step reads; host
/// prefix lengths are rebound when the step replays.
pub(crate) const CAPTURE_PREFIX: usize = 16;

/// A denoising step captured at startup: `images` images of one size, each
/// evaluating `branches` guidance branches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FlowShape {
    pub(crate) images: usize,
    pub(crate) branches: usize,
    pub(crate) height: u32,
    pub(crate) width: u32,
}

/// An image model's denoiser runner on a rank.
#[derive(Default)]
pub(crate) struct DenoiserRunners {
    /// An image model's denoiser, with its component.
    pub(crate) image: Option<(String, DenoiserRunner)>,
    /// The denoising steps captured at startup, largest first, and their
    /// graph pools: one on the denoiser's device and one on each other
    /// device its modules run on (a generation tower). Empty when flow
    /// graphs are off.
    flow: Vec<FlowShape>,
    flow_pools: Vec<Arc<GraphPool>>,
    /// The `[height, width]` latent image sizes whose conditioning startup
    /// captures into the same pools: those the latent encoder captured.
    conditions: Vec<[u32; 2]>,
    /// Whether the image denoiser serves under CUDA graphs (graph policy
    /// not `off`, on a CUDA device): it then denoises only the captured
    /// steps.
    image_graphs: bool,
    /// Resets newly assigned cache units on the image denoiser's stream.
    pub(crate) units: Option<UnitColumn>,
}

impl DenoiserRunners {
    /// The denoiser of an image model with the steps and the conditioning
    /// sizes startup captures, their graph pools and the unit reset column
    /// of its stream. Under `graphs` the denoiser evaluates only what
    /// startup captured once it is sealed ([`Self::seal_image`]).
    pub(crate) fn image(
        denoiser: Option<(String, DenoiserRunner)>,
        flow: Vec<FlowShape>,
        flow_pools: Vec<Arc<GraphPool>>,
        conditions: Vec<[u32; 2]>,
        graphs: bool,
        units: Option<UnitColumn>,
    ) -> Self {
        let image_graphs = graphs && denoiser.is_some();
        Self {
            image: denoiser,
            flow,
            flow_pools,
            conditions,
            image_graphs,
            units,
        }
    }

    /// Whether the image denoiser serves under CUDA graphs, only the steps
    /// it captured.
    pub(crate) fn image_graphs(&self) -> bool {
        self.image_graphs
    }

    /// The image denoiser's captured steps, largest first; empty without an
    /// image denoiser on this rank or with flow graphs off.
    pub(crate) fn flow_shapes(&self) -> &[FlowShape] {
        if self.image.is_some() {
            &self.flow
        } else {
            &[]
        }
    }

    /// Capture the image denoiser's configured steps and conditionings over
    /// scratch prefix cache of `prefix_cache`, with their samples and
    /// features in the step workspace of `latents`, as jobs of the lane on
    /// `lanes` serving denoising; a last job waits for the stream.
    pub(crate) fn capture_flow(
        &mut self,
        latents: Option<&LatentCache>,
        prefix_cache: Option<&mut PrefixCache>,
        lanes: &JobQueues,
    ) -> Result<()> {
        let (Some(prefix_cache), Some(latents)) = (
            prefix_cache.filter(|_| !self.flow_pools.is_empty()),
            latents,
        ) else {
            return Ok(());
        };
        let pools = self.flow_pools.clone();
        let shapes = self.flow.clone();
        for shape in &shapes {
            self.capture_step(shape, &pools, latents, prefix_cache, lanes)?;
        }
        for size in self.conditions.clone() {
            self.capture_condition(size, &pools, latents, prefix_cache, lanes)?;
        }
        if let Some((_, denoiser)) = &self.image {
            let job = Job::new();
            job.on(denoiser.stream()).synchronize()?;
            perform(lanes, job)?;
            tracing::info!(
                conditionings = self.conditions.len(),
                "captured {} image denoising graphs",
                denoiser.captured()
            );
        }
        Ok(())
    }

    /// Capture the conditioning of one latent image of `height` x `width`
    /// after a scratch slot's prefix, its features zero rows of the step
    /// workspace of `latents`. Three jobs of the lane, as
    /// [`Self::capture_step`] runs them: one zeroes the features and
    /// installs the scratch table, one warms and captures the conditioning,
    /// whose completion carries the graph, and one clears the table and
    /// resets the scratch units, whether or not the capture succeeded.
    fn capture_condition(
        &mut self,
        [height, width]: [u32; 2],
        pools: &[Arc<GraphPool>],
        latents: &LatentCache,
        prefix_cache: &mut PrefixCache,
        lanes: &JobQueues,
    ) -> Result<()> {
        let Some((_, denoiser)) = &mut self.image else {
            return Ok(());
        };
        let layout = *denoiser.layout()?;
        let units = layout.latent_rows(height, width)?;
        let prefix = CAPTURE_PREFIX as u32;
        let length = prefix + layout.sequence_length(height, width)?;
        let features = latents.workspace()?.narrow(0, 0, i64::from(units))?;
        let stream = denoiser.stream().cloned();

        let before = Job::new();
        // SAFETY: the latent cache keeps its workspace past the capture,
        // whose jobs are performed in order before it returns; nothing else
        // uses the workspace before serving starts.
        unsafe { features.zero(before.on(stream.as_ref())) }?;
        let scratch =
            prefix_cache.install_scratch(&[length as usize], before.on(stream.as_ref()))?;
        perform(lanes, before)?;

        let job = Job::new();
        let row = ConditionRow {
            features,
            height,
            width,
            slot: 1,
            prefix,
            position: u64::from(prefix),
        };
        let captured = denoiser
            .condition_latents(&[row], prefix_cache, Some(pools), &job)
            .and_then(|()| perform(lanes, job))
            .and_then(|mut completion| {
                denoiser.complete_condition(ConditionShape { height, width }, &mut completion)
            });

        let after = Job::new();
        let released = prefix_cache.release_scratch(scratch, 1, after.on(stream.as_ref()));
        let performed = perform(lanes, after).map(drop);
        captured.and(released).and(performed)
    }

    /// Run the image denoiser's eager calls once over scratch prefix cache
    /// of `prefix_cache`, with samples in the step workspace of `latents`,
    /// as jobs of the lane: one denoising step of the
    /// smallest image (two latent patches a side) over a slot's prefix,
    /// and, when the model `conditions` on latent images, the write of a
    /// zero latent image's K/V after that prefix.
    ///
    /// Without CUDA graphs serving evaluates every step and every latent
    /// conditioning eagerly. Running each call here selects and compiles
    /// the kernels of its attention inputs (a segmented read of the prefix,
    /// and non-causal paged rows) before the first image request; kernel
    /// choice depends on the input class, not on lengths. Four jobs: one
    /// zeroes the samples and installs the scratch table, one evaluates
    /// the step, one writes the conditioning, and one clears the table and
    /// resets its units whether or not the calls succeeded.
    pub(crate) fn warm_image(
        &mut self,
        latents: Option<&LatentCache>,
        prefix_cache: Option<&mut PrefixCache>,
        conditions: bool,
        lanes: &JobQueues,
    ) -> Result<()> {
        let (Some((_, denoiser)), Some(latents), Some(prefix_cache)) =
            (&mut self.image, latents, prefix_cache)
        else {
            return Ok(());
        };
        let layout = *denoiser.layout()?;
        let side = 2 * layout.downsample;
        let units = layout.latent_rows(side, side)?;
        let prefix = CAPTURE_PREFIX as u32;
        // The step reads the prefix; the conditioning writes its sequence
        // after it.
        let length = prefix
            + if conditions {
                layout.sequence_length(side, side)?
            } else {
                0
            };
        let samples = latents.workspace()?.narrow(0, 0, i64::from(units))?;
        let stream = denoiser.stream().cloned();

        let before = Job::new();
        // SAFETY: the latent cache keeps its workspace past this warmup,
        // whose jobs are performed in order before it returns; nothing else
        // uses the workspace before serving starts.
        unsafe { samples.zero(before.on(stream.as_ref())) }?;
        let scratch =
            prefix_cache.install_scratch(&[length as usize], before.on(stream.as_ref()))?;
        perform(lanes, before)?;

        let step = Job::new();
        let row = DenoiseRow {
            sample: 0,
            units,
            height: side,
            width: side,
            slot: 1,
            prefix,
            temporal: u64::from(prefix),
            timestep: 0.5,
        };
        let mut warmed = denoiser
            .predict(&[row], prefix_cache, None, &step)
            .and_then(|_| perform(lanes, step).map(drop));
        if conditions && warmed.is_ok() {
            let job = Job::new();
            let condition = ConditionRow {
                features: samples,
                height: side,
                width: side,
                slot: 1,
                prefix,
                position: u64::from(prefix),
            };
            warmed = denoiser
                .condition_latents(&[condition], prefix_cache, None, &job)
                .and_then(|()| perform(lanes, job).map(drop));
        }

        let after = Job::new();
        let released = prefix_cache.release_scratch(scratch, 1, after.on(stream.as_ref()));
        let performed = perform(lanes, after).map(drop);
        warmed.and(released).and(performed)
    }

    /// Capture one denoising step: `shape.images` images placed in the
    /// step workspace as serving places them (consecutive whole pages from
    /// its start), each evaluating `shape.branches` guidance branches over
    /// one scratch slot's prefix. Three jobs of the lane:
    /// one zeroes the samples and installs the scratch tables, one warms
    /// and captures the step, whose completion carries the graph, and one
    /// clears the tables and resets the scratch units, whether or not the
    /// step succeeded.
    fn capture_step(
        &mut self,
        shape: &FlowShape,
        pools: &[Arc<GraphPool>],
        latents: &LatentCache,
        prefix_cache: &mut PrefixCache,
        lanes: &JobQueues,
    ) -> Result<()> {
        let page_units = latents.pool.page_units() as u32;
        let workspace = latents.workspace()?.clone();
        let Some((_, denoiser)) = &mut self.image else {
            return Ok(());
        };
        let units = denoiser.layout()?.latent_rows(shape.height, shape.width)?;
        let padded = units.div_ceil(page_units) * page_units;
        let stream = denoiser.stream().cloned();
        let before = Job::new();
        // Captured samples start from zeros, as fresh scratch pages do.
        // SAFETY: the latent cache keeps its workspace past the capture,
        // whose jobs are performed in order before it returns; nothing else
        // uses the workspace before serving starts.
        unsafe {
            workspace
                .narrow(0, 0, i64::from(padded) * shape.images as i64)?
                .zero(before.on(stream.as_ref()))
        }?;
        let prefix = CAPTURE_PREFIX;
        let mut rows = Vec::with_capacity(shape.images * shape.branches);
        for image in 0..shape.images {
            for _ in 0..shape.branches {
                rows.push(DenoiseRow {
                    sample: image as u32 * padded,
                    units,
                    height: shape.height,
                    width: shape.width,
                    slot: image as u32 + 1,
                    prefix: prefix as u32,
                    temporal: prefix as u64,
                    timestep: 0.5,
                });
            }
        }
        let lengths = vec![prefix; shape.images];
        let scratch = prefix_cache.install_scratch(&lengths, before.on(stream.as_ref()))?;
        perform(lanes, before)?;

        let job = Job::new();
        let stepped = match denoiser.capture_step(&rows, prefix_cache, pools, &job) {
            Ok(Some(captured)) => perform(lanes, job).and_then(|mut completion| {
                denoiser.complete_step(captured, &mut completion, &mut || Ok(()))
            }),
            Ok(None) => Ok(()),
            Err(error) => Err(error),
        };
        let after = Job::new();
        let released =
            prefix_cache.release_scratch(scratch, lengths.len(), after.on(stream.as_ref()));
        let performed = perform(lanes, after).map(drop);
        stepped.and(released).and(performed)
    }

    /// Seal the image denoiser's graphs. Under CUDA graphs a step no
    /// captured graph holds is refused from then on; the warmup before
    /// evaluated its steps eagerly.
    pub(crate) fn seal_image(&mut self) -> Result<()> {
        if let Some((_, runner)) = &mut self.image {
            runner.seal();
            if self.image_graphs {
                runner.require_graphs()?;
            }
        }
        Ok(())
    }
}
