//! Preempting and resuming requests at denoising-step boundaries: video
//! requests, and the image a token request is generating (BAGEL,
//! SenseNova).
//!
//! A request's trajectory is derived state: the samples of its committed
//! step, which the request's facts and the step count determine. Whatever
//! preempts a request (a world's park, a sleep, an in-place weight update,
//! a shrink of the KV cache, the loss of its workers) preempts it the same
//! way, and a resumed request is the same whether it resumes on this world
//! or on another:
//!
//! 1. **Preempt.** The request offers no further call
//!    (`TerminalIntent::Migrate`); its calls in flight complete, which ends
//!    it at a step boundary. A video request whose denoising is scheduled
//!    whole is not preempted: it finishes, as its remaining work is
//!    bounded. A token request leaves after every token and image it
//!    generated; the denoising steps its calls in flight complete advance
//!    the trajectory of the image it was generating, as its worker commits
//!    them, and its client reads their progress.
//! 2. **Store.** At that boundary, a trajectory past its first step and
//!    not past its last, on a denoiser whose ranks map host latent pools,
//!    is written to them (`crate::trajectories`); a lost worker's is not.
//! 3. **Leave.** The request ends as migratable with the steps stored
//!    (zero for none) and the record of where their pages lie and how each
//!    stored rank laid its part out, and the daemon submits it again with
//!    them, or with the stored steps it named before when it stored none: a
//!    trajectory's state at a step is the same whenever it is read. The
//!    daemon forgets a token request's stored steps once the image they
//!    belong to completes.
//! 4. **Resume.** A resumed request that names stored steps has the pages
//!    its record names pinned before its latent preparation, which lays
//!    them out in its own ranks' pages, whatever parallel configuration
//!    stored them, and opens the trajectory at that step when every page is
//!    pinned, and starts from noise otherwise. A video request pins them as
//!    it is admitted; a token request once it rebuilt its context and the
//!    preparation of the image it had begun is due, for the replica the
//!    preparation is placed on. Every other call of the request runs as for
//!    a new request: a video request's encoders read the outputs the host
//!    encoder pools hold (`crate::encoder_outputs`) and compute the others,
//!    and the reading and refining of its conditions are recomputed; a
//!    token request rebuilds the context its image is conditioned on from
//!    its facts (`generation::continuation`). Either computes what it
//!    computed before.

use super::*;
use crate::WorkerGroupId;
use crate::trajectories::{Offloaded, Restored, image_trajectory};

/// What [`Scheduler::store_preempted`] reads of a preempted request.
struct Preempted {
    request_key: RequestKey,
    /// Whether the request's workers hold it.
    registered: bool,
    /// The identity that keys its trajectory; zero for a request no engine
    /// stores a trajectory of.
    trajectory: u64,
    /// The steps its trajectory completed.
    completed: u32,
    /// The steps of its trajectory's schedule.
    steps: u32,
}

/// Where the store of a preempted request's trajectory stands.
enum Retirement {
    /// The request leaves with the steps of its trajectory the host latent
    /// pools stored (zero for none) and the record of where they lie.
    Leave(u32, Vec<u8>),
    /// The host latent pools started storing the trajectory: the request
    /// waits for them.
    Storing,
    /// The host latent pools are storing the trajectory.
    Waiting,
}

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
                stored_layout: Vec::new(),
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

    /// The worker group that holds `request_key`'s trajectory: the replica
    /// its denoising calls are placed on, as the component that serves them
    /// binds there.
    fn denoiser_worker(&self, request_key: RequestKey) -> Option<WorkerGroupId> {
        let kind = CallKind::Media(MediaCall::Denoising);
        let component = self
            .info
            .media_components
            .get(&MediaCall::Denoising)
            .map_or(DEFAULT_COMPONENT, String::as_str);
        let (_, bound, _) = self
            .placement
            .component_candidates(self.executor.as_ref(), kind, component)
            .next()?;
        self.placement
            .affinity
            .get(&(request_key, bound.to_owned()))
            .cloned()
    }

    /// Stores, for `cause`, the trajectory of request `id`, preempted with
    /// its calls completed (`preempted`), when it reached a step worth
    /// storing: past its first step and not past its last, of a request the
    /// daemon gave a trajectory identity, preempted by a drain while its
    /// workers hold it. `offloading` states whether the host latent pools
    /// are storing it already, whose answer then decides how it leaves.
    fn store_preempted(
        &mut self,
        id: RequestId,
        cause: MigrationCause,
        preempted: Preempted,
        offloading: bool,
        layouts: &BTreeMap<u32, String>,
    ) -> Retirement {
        if offloading {
            let outcome = self
                .trajectories
                .as_mut()
                .and_then(|store| store.offloaded(id));
            return match outcome {
                Some(Offloaded::Stored(steps, record)) => Retirement::Leave(steps, record),
                Some(Offloaded::Failed) => Retirement::Leave(0, Vec::new()),
                None => Retirement::Waiting,
            };
        }
        // A lost worker's trajectory is gone with it; a trajectory at step
        // zero is the request's noise, which any engine draws again.
        let stores = cause == MigrationCause::Drain
            && preempted.trajectory != 0
            && preempted.completed > 0
            && preempted.completed <= preempted.steps
            && preempted.registered;
        let worker = stores
            .then(|| self.denoiser_worker(preempted.request_key))
            .flatten();
        let started = match (worker, self.trajectories.as_mut()) {
            (Some(worker), Some(store)) => store.offload(
                id,
                preempted.request_key,
                &worker,
                preempted.trajectory,
                preempted.completed,
                layouts,
            ),
            _ => false,
        };
        if started {
            Retirement::Storing
        } else {
            Retirement::Leave(0, Vec::new())
        }
    }

    /// Ends preempted video request `id`, whose calls completed: its
    /// trajectory is stored when it reached a step worth storing, and it
    /// leaves once the store answered. Returns whether it left.
    pub(super) fn retire_preempted(&mut self, id: RequestId, cause: MigrationCause) -> bool {
        let Some(state) = self.media_state(id) else {
            return true;
        };
        let preempted = Preempted {
            request_key: state.admission.request_key,
            registered: state.admission_state == WorkerRegistration::Registered,
            trajectory: state.request.trajectory.id,
            completed: state.denoising.completed(),
            steps: state.denoising.steps(),
        };
        let offloading = state.host_trajectory.offloading;
        let layouts = state.host_trajectory.layouts.clone();
        match self.store_preempted(id, cause, preempted, offloading, &layouts) {
            Retirement::Leave(steps, record) => {
                if steps > 0 {
                    tracing::info!(request = id.0, steps, "stored a preempted video trajectory");
                }
                self.finish_media(id, DiffusionTerminal::Migrated(cause, steps, record));
                true
            }
            Retirement::Storing => {
                if let Some(state) = self.media_state_mut(id) {
                    state.host_trajectory.offloading = true;
                }
                false
            }
            Retirement::Waiting => false,
        }
    }

    /// Ends preempted token request `id`, whose calls completed: the
    /// trajectory of the image it was generating is stored when it reached
    /// a step worth storing, keyed by the request's trajectory identity and
    /// the image's index ([`image_trajectory`]), and the request leaves as
    /// migratable once the store answered, naming the steps stored.
    pub(super) fn retire_preempted_token(&mut self, id: RequestId, cause: MigrationCause) {
        let Some(state) = self.running.get(&id) else {
            return;
        };
        // An image's trajectory opens at its latent preparation and closes
        // at its decoding, which leave its progress at step zero between
        // images.
        let trajectory = match state.req.trajectory.id {
            0 => 0,
            id => image_trajectory(id, state.image_id),
        };
        let image = state.image_id;
        let preempted = Preempted {
            request_key: RequestKey::new(self.engine_id, id, state.request_epoch),
            registered: state.worker_registered,
            trajectory,
            completed: state.denoising.completed(),
            steps: state.denoising.steps(),
        };
        let offloading = state.host_trajectory.offloading;
        let layouts = state.host_trajectory.layouts.clone();
        match self.store_preempted(id, cause, preempted, offloading, &layouts) {
            Retirement::Leave(steps, record) => {
                if steps > 0 {
                    tracing::info!(
                        request = id.0,
                        image,
                        steps,
                        "stored a preempted image trajectory"
                    );
                }
                self.migrate_stored(id, cause, steps, record);
            }
            Retirement::Storing => {
                if let Some(state) = self.running.get_mut(&id) {
                    state.host_trajectory.offloading = true;
                }
            }
            Retirement::Waiting => {}
        }
    }

    /// Applies the trajectory progress of `call`, an image trajectory call
    /// of preempted token request `id` (`TerminalIntent::Migrate`) that
    /// completed with `record`, which takes no other effect: the worker
    /// committed it, so the trajectory a preemption stores continues after
    /// it. A latent preparation opens the trajectory at the step it
    /// restored (or zero) and states how the denoising ranks hold it; a
    /// denoising step advances it, and the client reads the steps it
    /// completed.
    pub(super) fn accept_preempted_trajectory(
        &mut self,
        id: RequestId,
        call: &Call,
        record: &ipc::CallOutput,
    ) {
        if record.status != CallStatus::Ok || call.latent_output.is_none() {
            return;
        }
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        match call.kind {
            CallKind::Media(MediaCall::LatentPreparation) => {
                let opened = call
                    .latent_restore
                    .as_ref()
                    .map_or(0, |restore| restore.steps);
                if record.step_index == opened
                    && state.denoising.open(opened, call.latent_output.clone())
                {
                    state.host_trajectory.layouts = record
                        .latent_layouts
                        .iter()
                        .map(|layout| (layout.rank, layout.layout.clone()))
                        .collect();
                }
            }
            CallKind::Media(MediaCall::Denoising) => {
                let previous = state.denoising.completed();
                if state
                    .denoising
                    .accept(record.step_index, call.latent_output.clone())
                {
                    self.emit_image_steps(id, previous);
                }
            }
            _ => {}
        }
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
            .filter(|(_, state)| state.host_trajectory.restoring == Restoring::Pinning)
            .map(|(id, _)| *id)
            .chain(
                self.running
                    .iter()
                    .filter(|(_, state)| state.host_trajectory.restoring == Restoring::Pinning)
                    .map(|(id, _)| *id),
            )
            .collect();
        for id in pinning {
            let Some(restored) = self
                .trajectories
                .as_ref()
                .and_then(|store| store.restored(id))
            else {
                continue;
            };
            let image = self.running.get(&id).map(|state| state.image_id);
            let restore = match (restored, image) {
                (Restored::Hit(restore), None) => {
                    tracing::info!(
                        request = id.0,
                        steps = restore.steps,
                        "a resumed video request continues from its stored trajectory"
                    );
                    Some(restore)
                }
                (Restored::Hit(restore), Some(image)) => {
                    tracing::info!(
                        request = id.0,
                        image,
                        steps = restore.steps,
                        "a resumed image continues from its stored trajectory"
                    );
                    Some(restore)
                }
                (Restored::Miss, None) => {
                    tracing::info!(
                        request = id.0,
                        "a resumed video request's stored trajectory is missing: it starts from noise"
                    );
                    None
                }
                (Restored::Miss, Some(image)) => {
                    tracing::info!(
                        request = id.0,
                        image,
                        "a resumed image's stored trajectory is missing: it starts from noise"
                    );
                    None
                }
            };
            if let Some(state) = self.media_state_mut(id) {
                state.host_trajectory.restoring = Restoring::Ready(restore);
            } else if let Some(state) = self.running.get_mut(&id) {
                state.host_trajectory.restoring = Restoring::Ready(restore);
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
        let trajectory = state.request.trajectory.clone();
        if trajectory.denoised_steps == 0 || trajectory.denoised_steps >= state.denoising.steps() {
            return Restoring::Ready(None);
        }
        let request_key = state.admission.request_key;
        let Some(worker) = self.denoiser_worker(request_key) else {
            return Restoring::Ready(None);
        };
        let pinning = self.trajectories.as_mut().is_some_and(|store| {
            store.restore(
                id,
                &worker,
                trajectory.id,
                trajectory.denoised_steps,
                trajectory.state_seed,
                &trajectory.stored_layout,
            )
        });
        if pinning {
            Restoring::Pinning
        } else {
            Restoring::Ready(None)
        }
    }

    /// The stored trajectory the latent preparation of token request
    /// `id`'s current image opens at: `Some(Some(restore))` for the stored
    /// step of the image it resumes, `Some(None)` from noise, and `None`
    /// while the stored pages are pinned. A request that resumes the image
    /// it had begun (`Restoring::Due`) pins them when the preparation is
    /// first due, for the replica it places the preparation on: the stored
    /// steps of its trajectory (`GenerationRequest::trajectory`), keyed by
    /// the image's index ([`image_trajectory`]). A replica that has no room
    /// for the call yet is chosen at a later step.
    pub(super) fn image_restore(&mut self, id: RequestId) -> Option<Option<ipc::LatentRestore>> {
        let state = self.running.get(&id)?;
        match &state.host_trajectory.restoring {
            Restoring::Due => {}
            Restoring::Pinning => return None,
            Restoring::Ready(restore) => return Some(restore.clone()),
            Restoring::Planned => return Some(None),
        }
        let request_key = RequestKey::new(self.engine_id, id, state.request_epoch);
        let stored = state.req.trajectory.clone();
        let trajectory = image_trajectory(stored.id, state.image_id);
        let worker = self.placement.place(
            self.executor.as_ref(),
            &self.info,
            request_key,
            CallKind::Media(MediaCall::LatentPreparation),
        )?;
        let pinning = self.trajectories.as_mut().is_some_and(|store| {
            store.restore(
                id,
                &worker,
                trajectory,
                stored.denoised_steps,
                stored.state_seed,
                &stored.stored_layout,
            )
        });
        let state = self.running.get_mut(&id)?;
        if pinning {
            state.host_trajectory.restoring = Restoring::Pinning;
            None
        } else {
            state.host_trajectory.restoring = Restoring::Ready(None);
            Some(None)
        }
    }

    /// Ends the restore of request `id`, whose latent preparation resolved
    /// or which left: the pages it pinned are released.
    pub(super) fn end_restore(&mut self, id: RequestId) {
        if let Some(store) = self.trajectories.as_mut() {
            store.release(id);
        }
    }

    /// Ends the restore of token request `id`'s image, whose latent
    /// preparation completed, as `end_restore` does: the request's later
    /// images start from noise.
    pub(super) fn end_image_restore(&mut self, id: RequestId) {
        self.end_restore(id);
        if let Some(state) = self.running.get_mut(&id) {
            state.host_trajectory.restoring = Restoring::Planned;
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
