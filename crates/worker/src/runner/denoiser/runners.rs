//! The denoiser runners of a rank.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use uniserve_ipc::{BufferId, RequestKey};

use super::image::{DenoiseRow, ImageDiffusion};
use super::{DenoiserRunner, VideoModel};
use crate::Result;
use crate::cuda_graph::GraphPool;
use crate::executor::lifecycle::{LaneJobs, RequestStorage};
use crate::ffi::Handle;
use crate::lane::Job;
use crate::profiling::kernels::KernelRecord;
use crate::request::video::{MediaPlan, VideoLayout, VideoSize};
use crate::runner::causal_lm::warmup::with_scratch;
use crate::runner::component::Component;
use crate::storage::{ImageLatents, KVCache, UnitColumn};

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
    pub(crate) image: Option<(String, ImageDiffusion)>,
    /// The denoising steps captured at startup, largest first, and their
    /// graph pools: one on the denoiser's device and one on each other
    /// device its modules run on (a generation tower). Empty when flow
    /// graphs are off.
    flow: Vec<FlowShape>,
    flow_pools: Vec<Arc<GraphPool<Handle>>>,
    /// Resets newly assigned KV units on the image denoiser's stream.
    pub(crate) units: Option<UnitColumn>,
}

impl DenoiserRunners {
    /// The denoiser of an image model with the steps startup captures,
    /// their graph pools and the unit reset column of its stream.
    pub(crate) fn image(
        denoiser: Option<(String, ImageDiffusion)>,
        flow: Vec<FlowShape>,
        flow_pools: Vec<Arc<GraphPool<Handle>>>,
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

    /// Capture the image denoiser's configured steps over scratch KV of
    /// `kv`, with their samples in the step workspace of `latents`.
    pub(crate) fn capture_flow(
        &mut self,
        latents: Option<&ImageLatents>,
        kv: Option<&mut KVCache>,
    ) -> Result<()> {
        let (Some(kv), Some(latents)) = (kv.filter(|_| !self.flow_pools.is_empty()), latents)
        else {
            return Ok(());
        };
        let pools = self.flow_pools.clone();
        let shapes = self.flow.clone();
        for shape in &shapes {
            self.capture_step(shape, &pools, latents, kv)?;
        }
        if let Some((_, diffusion)) = &self.image {
            if let Some(stream) = diffusion.stream() {
                stream.synchronize()?;
            }
            tracing::info!(
                target: "uniserve_worker.warmup",
                "captured {} image denoising graphs",
                diffusion.captured()
            );
        }
        Ok(())
    }

    /// Capture one denoising step: `shape.images` images placed in the
    /// step workspace as serving places them (consecutive whole pages from
    /// its start), each evaluating `shape.branches` guidance branches over
    /// one scratch slot's prefix.
    fn capture_step(
        &mut self,
        shape: &FlowShape,
        pools: &[Arc<GraphPool<Handle>>],
        latents: &ImageLatents,
        kv: &mut KVCache,
    ) -> Result<()> {
        let page_units = latents.pool.page_units() as u32;
        let workspace = latents.workspace()?.clone();
        let Some((_, diffusion)) = &mut self.image else {
            return Ok(());
        };
        let units = diffusion.layout().latent_rows(shape.height, shape.width)?;
        let padded = units.div_ceil(page_units) * page_units;
        let stream = diffusion.stream().cloned();
        // Captured samples start from zeros, as fresh scratch pages do.
        workspace
            .narrow(0, 0, i64::from(padded) * shape.images as i64)?
            .zero(stream.as_ref())?;
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
        with_scratch(kv, &lengths, stream.as_ref(), |kv| {
            diffusion.capture(&rows, kv.tables(), pools, &mut || Ok(()))
        })
    }

    /// Seal the image denoiser's graphs.
    pub(crate) fn seal_image(&mut self) {
        if let Some((_, runner)) = &mut self.image {
            runner.seal();
        }
    }

    /// Hold `layout` for the request admitted in `slot` and return its id,
    /// releasing the layout of the request the slot held before. `None` on
    /// a rank without a video denoiser.
    pub(crate) fn admit(
        &mut self,
        slot: usize,
        model: &VideoModel,
        layout: VideoLayout,
    ) -> Result<Option<usize>> {
        let Some(denoiser) = &mut self.video else {
            return Ok(None);
        };
        let id = denoiser.runner.acquire(model, layout)?;
        if let Some(previous) = self.layouts.insert(slot, id) {
            denoiser.runner.release(previous)?;
        }
        Ok(Some(id))
    }

    /// Prepare and warm every startup layout (the largest first, so it
    /// sizes the shared workspace) and the conditioner capacities.
    pub(crate) fn prepare_startup(&mut self, model: &VideoModel, plan: &MediaPlan) -> Result<()> {
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
                // The placeholder's host sources are filled, at once, before
                // its layout warms.
                let job = Job::new();
                runner.prepare_latents(id, 1, &size, 0, &job)?;
                job.perform().check().map_err(|failure| failure.error)?;
                runner.warm(id, 1)?;
            }
            runner.warm_conditioner(false)?;
        }
        Ok(())
    }

    /// Capture the startup layouts and conditioner capacities, then refuse
    /// further captures.
    pub(crate) fn capture_startup(&mut self, model: &VideoModel, plan: &MediaPlan) -> Result<()> {
        if let Some(denoiser) = &mut self.video {
            let runner = &mut denoiser.runner;
            for layout in &plan.layouts {
                let id = runner.prepare_startup(model, *layout)?;
                runner.capture(id)?;
            }
            runner.warm_conditioner(true)?;
            runner.stream().synchronize()?;
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
    fn admit_slots(&mut self, _slots: &[usize], _jobs: &LaneJobs) -> Result<()> {
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
