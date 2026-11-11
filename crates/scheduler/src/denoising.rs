//! Solver progress of one request's latent trajectory.
//!
//! Latent preparation opens a trajectory at step zero, or at the stored step
//! a resumed request restores, each denoising call covers the next step of
//! its fixed schedule, and the worker reports the steps it has accepted.
//! [`Denoising`] owns that progress.

use ipc::MediaCall;

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
}

impl Denoising {
    /// Returns the progress of a trajectory of `steps` solver steps.
    pub(crate) const fn new(steps: u32) -> Self {
        Self {
            steps,
            completed: 0,
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

    /// Returns the steps covered once the request's submitted calls complete.
    ///
    /// `pending` lists the request's submitted media calls in submission
    /// order, each with the step a latent preparation opens its trajectory
    /// at: zero, or the stored step it restores. Each denoising call covers
    /// the next step of the schedule.
    pub(crate) fn scheduled(&self, pending: impl IntoIterator<Item = (MediaCall, u32)>) -> u32 {
        pending
            .into_iter()
            .fold(self.completed, |steps, (call, opened)| match call {
                MediaCall::LatentPreparation => opened,
                MediaCall::Denoising => steps.saturating_add(1),
                _ => steps,
            })
    }

    /// Returns whether the denoising call covering `step` ends the trajectory.
    pub(crate) const fn ends(&self, step: u32) -> bool {
        step.saturating_add(1) == self.steps
    }

    /// Opens a trajectory at step `step`: zero, or the stored step a resumed
    /// request's latent preparation restored; a video request is not
    /// preempted once its denoising is scheduled whole. Returns `false`,
    /// leaving progress unchanged, for a restored step past the schedule,
    /// which a preemption never stores.
    pub(crate) fn open(&mut self, step: u32) -> bool {
        if step > self.steps {
            return false;
        }
        self.completed = step;
        true
    }

    /// Accepts a completed denoising call.
    ///
    /// The worker owns the trajectory's step usage and reports the steps it
    /// has completed; the report becomes the request's progress. Returns
    /// `false`, leaving progress unchanged, when the report does not advance
    /// it, since a denoising call that makes no progress would be scheduled
    /// again without end.
    pub(crate) fn accept(&mut self, completed: u32) -> bool {
        if completed <= self.completed {
            return false;
        }
        self.completed = completed;
        true
    }
}
