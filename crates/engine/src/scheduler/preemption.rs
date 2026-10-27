//! Preempting and resuming video requests at denoising-step boundaries.
//!
//! A video request's trajectory is derived state: the samples of its
//! committed step, which the request's facts and the step count determine.
//! Whatever preempts a request (a world's park, a sleep, an in-place weight
//! update, the loss of its workers) preempts it the same way, and a resumed
//! request is the same whether it resumes on this world or on another:
//!
//! 1. **Preempt.** The request offers no further call
//!    (`TerminalIntent::Migrate`); its calls in flight complete, which ends
//!    it at a step boundary. A request whose denoising is scheduled whole
//!    is not preempted: it finishes, as its remaining work is bounded.
//! 2. **Store.** At that boundary, a trajectory past its first step and
//!    short of its last, on a denoiser whose ranks map host latent pools,
//!    is written to them (`crate::trajectories`); a lost worker's is not.
//! 3. **Leave.** The request ends as migratable with the steps stored
//!    (zero for none), and the daemon submits it again with them, or with
//!    the stored steps it named before when it stored none: a trajectory's
//!    state at a step is the same whenever it is read.
//! 4. **Resume.** An admitted request that names stored steps has its
//!    pages pinned before its latent preparation, which restores them and
//!    opens the trajectory at that step when every page is pinned, and
//!    starts from noise otherwise. Every other call of the request (the
//!    reading, encoding and refining of its conditions) runs as for a new
//!    request: they are recomputed, and compute what they computed before.

use super::*;
use crate::trajectories::{Offloaded, Restored};

impl Scheduler {
    /// Preempts, for `cause`, every running video request that `select`
    /// picks and that has a denoising step left to schedule, at its next
    /// step boundary; queued video requests leave at once with the steps
    /// they name.
    pub(super) fn preempt_media(
        &mut self,
        cause: MigrationCause,
        select: impl Fn(RequestId) -> bool,
    ) {
        let (queued, staying): (VecDeque<PendingMedia>, VecDeque<PendingMedia>) =
            std::mem::take(&mut self.waiting_media)
                .into_iter()
                .partition(|pending| select(pending.request.request_id));
        self.waiting_media = staying;
        for pending in queued {
            // A queued request stored nothing here: the daemon submits it
            // again naming the trajectory it named.
            let _ = pending.event_tx.send(Event::Migratable {
                cause,
                denoised_steps: 0,
            });
        }
        let running: Vec<RequestId> = self
            .running_media
            .iter()
            .filter(|(id, state)| {
                select(**id)
                    && !state.terminal_intent.is_terminal()
                    && self.scheduled_steps(**id, &state.denoising) < state.denoising.steps()
            })
            .map(|(id, _)| *id)
            .collect();
        for id in running {
            if let Some(state) = self.media_state_mut(id) {
                tracing::info!(
                    request = id.0,
                    completed = state.denoising.completed(),
                    ?cause,
                    "preempting a video request at its next denoising step boundary"
                );
                state.terminal_intent = TerminalIntent::Migrate(cause);
            }
        }
    }

    /// Ends preempted video request `id`, whose calls completed: its
    /// trajectory is stored when it reached a step worth storing, and it
    /// leaves once the store answered. Returns whether it left.
    pub(super) fn retire_preempted(&mut self, id: RequestId, cause: MigrationCause) -> bool {
        let Some(state) = self.media_state(id) else {
            return true;
        };
        if state.offloading {
            let outcome = self
                .trajectories
                .as_mut()
                .and_then(|store| store.offloaded(id));
            return match outcome {
                Some(Offloaded::Stored(steps)) => {
                    tracing::info!(request = id.0, steps, "stored a preempted video trajectory");
                    self.finish_media(id, DiffusionTerminal::Migrated(cause, steps));
                    true
                }
                Some(Offloaded::Failed) => {
                    self.finish_media(id, DiffusionTerminal::Migrated(cause, 0));
                    true
                }
                None => false,
            };
        }
        // A lost worker's trajectory is gone with it; a trajectory at step
        // zero is the request's noise, which any engine draws again.
        let steps = state.denoising.completed();
        let stores = cause == MigrationCause::Drain
            && steps > 0
            && steps < state.denoising.steps()
            && state.admission_state == WorkerRegistration::Registered;
        let request_key = state.admission.request_key;
        let trajectory = state.request.trajectory.id;
        let denoiser = self
            .info
            .media_components
            .get(&MediaCall::Denoising)
            .and_then(|component| {
                self.placement
                    .affinity
                    .get(&(request_key, component.clone()))
                    .cloned()
            });
        let offloading = match (stores, denoiser, self.trajectories.as_mut()) {
            (true, Some(worker), Some(store)) => {
                store.offload(id, request_key, &worker, trajectory, steps)
            }
            _ => false,
        };
        if offloading {
            if let Some(state) = self.media_state_mut(id) {
                state.offloading = true;
            }
            return false;
        }
        self.finish_media(id, DiffusionTerminal::Migrated(cause, 0));
        true
    }

    /// Takes the host latent pools' answers: a resumed request's restore
    /// resolves, and the copies of a preempted request's trajectory leave
    /// in a batch of latent offloads alone to its denoiser's worker group.
    /// Returns whether anything moved on.
    pub(super) fn poll_trajectories(&mut self) -> bool {
        let Some(store) = self.trajectories.as_mut() else {
            return false;
        };
        let moved = store.poll();
        let pinning: Vec<RequestId> = self
            .running_media
            .iter()
            .filter(|(_, state)| state.restoring == Restoring::Pinning)
            .map(|(id, _)| *id)
            .collect();
        for id in pinning {
            let Some(restored) = self
                .trajectories
                .as_ref()
                .and_then(|store| store.restored(id))
            else {
                continue;
            };
            let restore = match restored {
                Restored::Hit(restore) => {
                    tracing::info!(
                        request = id.0,
                        steps = restore.steps,
                        "a resumed video request continues from its stored trajectory"
                    );
                    Some(restore)
                }
                Restored::Miss => {
                    tracing::info!(
                        request = id.0,
                        "a resumed video request's stored trajectory is missing: it starts from noise"
                    );
                    None
                }
            };
            if let Some(state) = self.media_state_mut(id) {
                state.restoring = Restoring::Ready(restore);
            }
        }
        // The copies wait for nothing on the device: every call of the
        // request completed, and the worker orders them behind the batches
        // that used the latent banks.
        let queued = self
            .trajectories
            .as_ref()
            .map(crate::trajectories::TrajectoryStore::queued)
            .unwrap_or_default();
        let mut submitted = false;
        for id in queued {
            let mut batch = self.finish_generation_batch(
                ExecutionBatch::new(0, Vec::new(), Vec::new(), Vec::new()),
                Instant::now(),
            );
            let Some((worker, offload)) = self
                .trajectories
                .as_mut()
                .and_then(|store| store.take_offload(id, batch.id))
            else {
                continue;
            };
            batch.page_offloads = Some(crate::executor::PageOffloads {
                worker,
                pages: Vec::new(),
                latents: vec![offload],
            });
            self.inflight.pending_submissions.push_back(batch);
            submitted = true;
        }
        moved || submitted
    }

    /// Starts restoring admitted video request `id`'s stored trajectory, which
    /// its latent preparation then waits for; a request that names no
    /// stored steps, or whose denoiser the host latent pools do not serve,
    /// starts from noise.
    pub(super) fn start_restore(&mut self, id: RequestId) -> Restoring {
        let Some(state) = self.media_state(id) else {
            return Restoring::Ready(None);
        };
        let trajectory = state.request.trajectory;
        if trajectory.denoised_steps == 0 || trajectory.denoised_steps >= state.denoising.steps() {
            return Restoring::Ready(None);
        }
        let request_key = state.admission.request_key;
        let Some(worker) = self
            .info
            .media_components
            .get(&MediaCall::Denoising)
            .and_then(|component| {
                self.placement
                    .affinity
                    .get(&(request_key, component.clone()))
                    .cloned()
            })
        else {
            return Restoring::Ready(None);
        };
        let pinning = self.trajectories.as_mut().is_some_and(|store| {
            store.restore(
                id,
                &worker,
                trajectory.id,
                trajectory.denoised_steps,
                trajectory.state_seed,
            )
        });
        if pinning {
            Restoring::Pinning
        } else {
            Restoring::Ready(None)
        }
    }

    /// Ends the restore of video request `id`, whose latent preparation
    /// resolved or which left: the pages it pinned are released.
    pub(super) fn end_restore(&mut self, id: RequestId) {
        if let Some(store) = self.trajectories.as_mut() {
            store.release(id);
        }
    }

    /// The batch `batch` carrying latent offloads completed: they are
    /// committed when it `succeeded`, and aborted otherwise.
    pub(super) fn trajectory_batch_completed(&mut self, batch: u64, succeeded: bool) {
        if let Some(store) = self.trajectories.as_mut() {
            store.completed(batch, succeeded);
        }
    }
}
