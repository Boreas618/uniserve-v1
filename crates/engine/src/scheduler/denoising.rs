//! Solver progress of one request's latent trajectory.
//!
//! Image generation inside a text request and the video pipeline keep their
//! own lifecycles, but both advance a trajectory the same way: latent
//! preparation opens it at step zero, each denoising call covers the next
//! step of its fixed schedule, and the worker reports the steps it has
//! accepted. [`Denoising`] owns that progress for both.

use ipc::{CallId, LatentParams, MediaCall, RequestKey, TensorRef};

/// Where an image request's latent lives on its worker and the raster it
/// describes.
///
/// The scheduler's latent allocator assigns the pages from the worker's pool,
/// so the worker learns them from the calls that name them; the raster is the
/// image the samples decode to. A standalone video denoiser's request slot
/// fixes its own pages, which the worker derives from its pool geometry, so
/// a video request has no placement.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct LatentPlacement {
    /// Pages the request owns in its worker's latent pool.
    pub(crate) page_table: Vec<u32>,
    /// Latent units the trajectory occupies in those pages, in the
    /// model-defined unit the worker sizes pages by (`latent_page_units`).
    pub(crate) latent_units: u32,
    /// Raster height in pixels: the requested image height.
    pub(crate) height: u32,
    /// Raster width in pixels: the requested image width.
    pub(crate) width: u32,
}

/// Solver progress of one latent trajectory.
///
/// Only accepted progress is stored. The steps already submitted are derived
/// from the request's pending calls, so a call that fails or is invalidated
/// needs no counter repair.
#[derive(Debug, Clone)]
pub(crate) struct Denoising {
    /// Steps in the trajectory's fixed schedule.
    steps: u32,
    /// Steps the worker has accepted.
    completed: u32,
    /// The latest accepted latent product, when the worker publishes one per
    /// call.
    latent: Option<TensorRef>,
}

impl Denoising {
    /// Returns the progress of a trajectory of `steps` solver steps.
    pub(crate) const fn new(steps: u32) -> Self {
        Self {
            steps,
            completed: 0,
            latent: None,
        }
    }

    /// Returns the length of the trajectory's schedule.
    pub(crate) const fn steps(&self) -> u32 {
        self.steps
    }

    /// Returns the steps the worker has accepted.
    pub(crate) const fn completed(&self) -> u32 {
        self.completed
    }

    /// Returns whether every step has been accepted.
    pub(crate) const fn is_complete(&self) -> bool {
        self.completed >= self.steps
    }

    /// Returns the latest accepted latent product.
    pub(crate) const fn latent(&self) -> Option<&TensorRef> {
        self.latent.as_ref()
    }

    /// Returns the steps covered once the request's submitted calls complete.
    ///
    /// `pending` lists the request's submitted media calls in submission
    /// order, each with the step a latent preparation opens its trajectory
    /// at: zero, or the stored step it restores. Image decoding closes a
    /// trajectory, leaving it at step zero, and each denoising call covers
    /// the next step of the schedule.
    pub(crate) fn scheduled(&self, pending: impl IntoIterator<Item = (MediaCall, u32)>) -> u32 {
        pending
            .into_iter()
            .fold(self.completed, |steps, (call, opened)| match call {
                MediaCall::LatentPreparation => opened,
                MediaCall::ImageDecoding => 0,
                MediaCall::Denoising => steps.saturating_add(1),
                _ => steps,
            })
    }

    /// Returns whether the denoising call covering `step` ends the trajectory.
    pub(crate) const fn ends(&self, step: u32) -> bool {
        step.saturating_add(1) == self.steps
    }

    /// Returns the latent pages and raster one call on an image trajectory
    /// addresses. The worker's latent pool keeps the trajectory's step
    /// ledger, so the call states no step.
    pub(crate) fn params(
        request_key: RequestKey,
        call_id: CallId,
        placement: &LatentPlacement,
    ) -> LatentParams {
        LatentParams {
            request_key,
            call_id,
            page_table: placement.page_table.clone(),
            latent_units: placement.latent_units,
            height: placement.height,
            width: placement.width,
        }
    }

    /// Opens a trajectory at step `step`, with its prepared latent: zero, or
    /// the stored step a video request's latent preparation restored.
    /// Returns `false`, leaving progress unchanged, for a restored step that
    /// leaves no denoising step of the schedule, which a preemption never
    /// stores.
    pub(crate) fn open(&mut self, step: u32, latent: Option<TensorRef>) -> bool {
        if step > 0 && step >= self.steps {
            return false;
        }
        self.completed = step;
        self.latent = latent;
        true
    }

    /// Accepts a completed denoising call and its successor latent.
    ///
    /// The worker owns the trajectory's step ledger and reports the steps it
    /// has completed; the report becomes the request's progress. Returns
    /// `false`, leaving progress unchanged, when the report does not advance
    /// it, since a denoising call that makes no progress would be scheduled
    /// again without end. A `None` latent keeps the previously accepted
    /// product.
    pub(crate) fn accept(&mut self, completed: u32, latent: Option<TensorRef>) -> bool {
        if completed <= self.completed {
            return false;
        }
        self.completed = completed;
        if latent.is_some() {
            self.latent = latent;
        }
        true
    }

    /// Closes the trajectory once its latent is released; the next one starts
    /// at step zero.
    pub(crate) fn close(&mut self) {
        self.completed = 0;
        self.latent = None;
    }
}
