//! The denoiser runners of a rank.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ipc::{BufferId, CallKind, MediaCall, RequestKey};

use super::image::{ConditionRow, DenoiseRow};
use super::{DenoiserRunner, Evicted, VideoModel};
use crate::cuda_graph::GraphPool;
use crate::executor::lifecycle::{AdmittedSlot, LaneJobs, RequestStorage};
use crate::lane::{BLOCK_ON_TIMEOUT, Completion, Job, JobQueues};
use crate::memory::{LatentCache, PrefixCache, UnitColumn};
use crate::profiling::kernels::KernelRecord;
use crate::request::video::{MediaPlan, VideoLayout, VideoSize};
use crate::runner::component::Component;
use crate::stream::Issue;
use crate::{Error, Result};

/// The lane kind of a denoiser's jobs.
const DENOISING: CallKind = CallKind::Media(MediaCall::Denoising);

/// Perform a startup `job` on the lane serving denoising and wait for its
/// completion.
fn perform(lanes: &JobQueues, job: Job) -> Result<Completion> {
    lanes
        .block_on(DENOISING, job, BLOCK_ON_TIMEOUT)?
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

/// The denoiser runners of a rank.
#[derive(Default)]
pub(crate) struct DenoiserRunners {
    /// A video model's denoiser.
    pub(crate) video: Option<Component<DenoiserRunner>>,
    /// The layout each admitted slot holds on the video denoiser.
    layouts: HashMap<usize, usize>,
    /// An image model's denoiser, with its component.
    pub(crate) image: Option<(String, DenoiserRunner)>,
    /// The denoising steps captured at startup, largest first, and their
    /// graph pools: one on the denoiser's device and one on each other
    /// device its modules run on (a generation tower). Empty when flow
    /// graphs are off.
    flow: Vec<FlowShape>,
    flow_pools: Vec<Arc<GraphPool>>,
    /// Resets newly assigned cache units on the image denoiser's stream.
    pub(crate) units: Option<UnitColumn>,
}

impl DenoiserRunners {
    /// The denoiser of an image model with the steps startup captures,
    /// their graph pools and the unit reset column of its stream.
    pub(crate) fn image(
        denoiser: Option<(String, DenoiserRunner)>,
        flow: Vec<FlowShape>,
        flow_pools: Vec<Arc<GraphPool>>,
        units: Option<UnitColumn>,
    ) -> Self {
        Self {
            image: denoiser,
            flow,
            flow_pools,
            units,
            ..Self::default()
        }
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

    /// Capture the image denoiser's configured steps over scratch prefix
    /// cache of `prefix_cache`, with their samples in the step workspace of
    /// `latents`, as jobs of the lane on `lanes` serving denoising; a last
    /// job waits for the stream.
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
        if let Some((_, denoiser)) = &self.image {
            let job = Job::new();
            job.on(denoiser.stream()).synchronize()?;
            perform(lanes, job)?;
            tracing::info!("captured {} image denoising graphs", denoiser.captured());
        }
        Ok(())
    }

    /// Run the image denoiser's eager calls once over scratch prefix cache
    /// of `prefix_cache`, with samples in the step workspace of `latents`,
    /// as jobs of the lane serving denoising: one denoising step of the
    /// smallest image (two latent patches a side) over a slot's prefix,
    /// and, when the model `conditions` on latent images, the write of a
    /// zero latent image's K/V after that prefix.
    ///
    /// Serving evaluates every step no graph holds, and every latent
    /// conditioning, eagerly. Running each call here selects and compiles
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
                .condition_latents(&[condition], prefix_cache, &job)
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
    /// one scratch slot's prefix. Three jobs of the lane serving denoising:
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

    /// Seal the image denoiser's graphs.
    pub(crate) fn seal_image(&mut self) {
        if let Some((_, runner)) = &mut self.image {
            runner.seal();
        }
    }

    /// Whether `layout` needs preparing before a request is admitted with
    /// it: the rank's video denoiser does not hold it yet.
    pub(crate) fn lacks(&self, layout: &VideoLayout) -> bool {
        self.video
            .as_ref()
            .is_some_and(|denoiser| !denoiser.runner.prepared(layout))
    }

    /// Record into `job`, a job of the video denoiser's lane, the
    /// preparation of transient layout `layout`
    /// ([`super::DenoiserRunner::ask_layout`]).
    pub(crate) fn ask_layout(
        &mut self,
        model: &VideoModel,
        layout: VideoLayout,
        job: &Job,
    ) -> Result<Evicted> {
        self.video
            .as_mut()
            .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?
            .runner
            .ask_layout(model, layout, job)
    }

    /// Keep transient layout `layout` as the job of [`Self::ask_layout`]
    /// prepared it.
    pub(crate) fn accept_layout(
        &mut self,
        model: &VideoModel,
        layout: VideoLayout,
        completion: &mut Completion,
    ) -> Result<()> {
        self.video
            .as_mut()
            .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?
            .runner
            .accept_layout(model, layout, completion)
    }

    /// Hold prepared `layout` for the request admitted in `slot` and
    /// return its id, releasing the layout of the request the slot held
    /// before. `None` on a rank without a video denoiser.
    pub(crate) fn admit(&mut self, slot: usize, layout: VideoLayout) -> Result<Option<usize>> {
        let Some(denoiser) = &mut self.video else {
            return Ok(None);
        };
        let id = denoiser.runner.acquire(layout)?.ok_or_else(|| {
            Error::Invariant("a request was admitted with a layout that is not prepared".into())
        })?;
        if let Some(previous) = self.layouts.insert(slot, id) {
            denoiser.runner.release(previous)?;
        }
        Ok(Some(id))
    }

    /// Prepare and warm every startup layout (the largest first, so it
    /// sizes the shared workspace) and the conditioner capacities. Each
    /// warmup is a job of the denoiser's lane on `lanes`, which startup
    /// waits for.
    pub(crate) fn prepare_startup(
        &mut self,
        model: &VideoModel,
        plan: &MediaPlan,
        lanes: &JobQueues,
    ) -> Result<()> {
        if let Some(denoiser) = &mut self.video {
            let runner = &mut denoiser.runner;
            let mut ids = Vec::new();
            for layout in std::iter::once(&plan.maximum)
                .chain(&plan.layouts)
                .chain(&plan.largest)
            {
                let id = runner.prepare_startup(model, *layout)?;
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
            // A placeholder request in slot one warms every layout before
            // any capture: kernels, plans and scratch are created by a
            // layout's first step.
            for &id in &ids {
                let key = runner.key(id)?;
                let size = VideoSize::plain(
                    key.frames,
                    key.canvas,
                    key.text_rows.min(plan.max_text_tokens),
                );
                // The placeholder's host sources are filled, then its
                // layout's step warms from them, in one job.
                let job = Job::new();
                runner.prepare_latents(id, 1, &size, 0, &job)?;
                runner.warm(id, 1, &job)?;
                perform(lanes, job)?;
            }
            let job = Job::new();
            runner.warm_conditioner(false, &job)?;
            perform(lanes, job)?;
        }
        Ok(())
    }

    /// Capture the startup layouts and conditioner capacities, then refuse
    /// further captures. Each layout's capture is a job of the denoiser's
    /// lane on `lanes`, whose completion carries the graph; a last job
    /// warms and captures the conditioner capacities and waits for the
    /// stream before the runner is sealed.
    pub(crate) fn capture_startup(
        &mut self,
        model: &VideoModel,
        plan: &MediaPlan,
        lanes: &JobQueues,
    ) -> Result<()> {
        if let Some(denoiser) = &mut self.video {
            let runner = &mut denoiser.runner;
            for layout in &plan.layouts {
                let id = runner.prepare_startup(model, *layout)?;
                let job = Job::new();
                if runner.capture(id, &job)? {
                    let mut completion = perform(lanes, job)?;
                    runner.complete_capture(id, &mut completion)?;
                }
            }
            let job = Job::new();
            let capacities = runner.warm_conditioner(true, &job)?;
            job.on(runner.stream()).synchronize()?;
            let mut completion = perform(lanes, job)?;
            runner.complete_conditioner(capacities, &mut completion)?;
            runner.seal();
        }
        Ok(())
    }

    /// The kernels the video denoiser's operators selected.
    pub(crate) fn kernels(&self) -> Result<Vec<(String, Vec<KernelRecord>)>> {
        let mut kernels = Vec::new();
        if let Some(denoiser) = &self.video {
            kernels.push((denoiser.name.clone(), denoiser.runner.kernels()?));
        }
        Ok(kernels)
    }
}

impl RequestStorage for DenoiserRunners {
    fn admit_slots(&mut self, _admitted: &[AdmittedSlot<'_>], _jobs: &LaneJobs) -> Result<()> {
        // A slot's layout is held when its request's admission is installed.
        Ok(())
    }

    fn release_buffers(&mut self, _buffers: &HashSet<BufferId>, _jobs: &LaneJobs) -> Result<()> {
        Ok(())
    }

    fn release_requests(
        &mut self,
        _requests: &HashSet<RequestKey>,
        _retained: &HashSet<BufferId>,
        _jobs: &LaneJobs,
    ) -> Result<()> {
        Ok(())
    }

    fn retirement_ready(
        &mut self,
        _buffers: &HashSet<BufferId>,
        _requests: &HashSet<RequestKey>,
        _retained: &HashSet<BufferId>,
    ) -> Result<bool> {
        Ok(true)
    }

    fn retire_slots(&mut self, slots: &[usize], _jobs: &LaneJobs) -> Result<()> {
        let Some(denoiser) = &mut self.video else {
            return Ok(());
        };
        let mut released = Vec::new();
        for slot in slots {
            if let Some(id) = self.layouts.remove(slot) {
                released.push(*slot);
                denoiser.runner.release(id)?;
            }
        }
        if !released.is_empty() {
            denoiser.runner.release_slots(&released)?;
        }
        Ok(())
    }

    fn reap(&mut self) -> Result<()> {
        Ok(())
    }

    fn close(&mut self, aborted: bool) -> Result<()> {
        self.layouts.clear();
        match &mut self.video {
            Some(denoiser) => denoiser.runner.close(aborted),
            None => Ok(()),
        }
    }
}
