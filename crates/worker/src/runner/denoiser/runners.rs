//! The denoiser runners of a rank.

use std::collections::{HashMap, HashSet};

use uniserve_ipc::{BufferId, RequestKey};

use super::{DenoiserRunner, VideoModel};
use crate::Result;
use crate::executor::lifecycle::RequestStorage;
use crate::ffi::Record;
use crate::request::video::{MediaPlan, VideoLayout, VideoSize};
use crate::runner::component::Component;

/// The denoiser runners of a rank.
#[derive(Default)]
pub(crate) struct DenoiserRunners {
    /// A video model's denoiser.
    pub(crate) video: Option<Component<DenoiserRunner>>,
    /// The layout each admitted slot holds on the video denoiser.
    layouts: HashMap<usize, usize>,
}

impl DenoiserRunners {
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
                runner.prepare_latents(id, 1, &size, 0)?;
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
    pub(crate) fn kernels(&self) -> Result<Vec<(String, Vec<Record>)>> {
        let mut kernels = Vec::new();
        if let Some(denoiser) = &self.video {
            kernels.push((
                denoiser.name.clone(),
                denoiser.runner.kernels(&denoiser.name)?,
            ));
        }
        Ok(kernels)
    }
}

impl RequestStorage for DenoiserRunners {
    fn admit_slots(&mut self, _slots: &[usize]) -> Result<()> {
        // A slot's layout is held when its request's admission is installed.
        Ok(())
    }

    fn release_buffers(&mut self, _buffers: &HashSet<BufferId>) -> Result<()> {
        Ok(())
    }

    fn release_requests(
        &mut self,
        _requests: &HashSet<RequestKey>,
        _retained: &HashSet<BufferId>,
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

    fn retire_slots(&mut self, slots: &[usize]) -> Result<()> {
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
