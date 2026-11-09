//! Cancellation, terminal cleanup, and lifecycle handling.
//!
//! The scheduler thread applies each frontend `Command` (see `crate::handle`)
//! here. Cancelling a queued request finishes it at once. A running request
//! records a terminal intent and closes only after every submitted call has
//! resolved: a media request with nothing in flight finishes immediately, and
//! `reap_cancellations` finishes the rest once their calls drain.
//! Acknowledgements resolve the stop-string decisions a request awaits
//! (`RequestOutput::decoder_boundaries`), and `abort_all_requests` ends every
//! request when the control loop stops.

use super::*;
use crate::handle;

/// The name a stage's refusal uses.
fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::Switchover => "switchover",
        Stage::Commit => "commit",
        Stage::Revert => "revert",
        Stage::Purge => "purge",
    }
}

/// What an engine command this scheduler does not apply itself is called, for
/// the answer that names its owner.
fn engine_command_name(kind: &proto::process::command::Kind) -> &'static str {
    use proto::process::command::Kind;
    match kind {
        Kind::Park(_) => "park",
        Kind::Promote(_) => "promote",
        Kind::Precopy(_) => "precopy",
        Kind::Switchover(_) => "switchover",
        Kind::Commit(_) => "commit",
        Kind::Revert(_) => "revert",
        Kind::Purge(_) => "purge",
        Kind::Drain(_) => "drain",
        Kind::Serve(_) => "serve",
        Kind::Pause(_) => "pause",
    }
}

/// An update's ask that the following step's batches also run in shadow for an
/// epoch generation (§7.3 step 2), with the answer that waits for the step which
/// shadows them.
///
/// The ask is kept until it is answered: a refused shadow submission answers it
/// at once with the worker's reason, and a submitted one answers it when its
/// step has ended, so an answer never claims a pass that has not run.
pub(super) struct ShadowPass {
    /// The epoch generation whose operators the shadow pass runs.
    pub(super) generation: u64,
    /// One of the step's batches was submitted as a shadow: the step that owns
    /// the answer has begun, and ends before the answer is sent.
    pub(super) shadowed: bool,
    /// The shapes this pass has compared. A shape is compared once (§7.3 step
    /// 2): a pass that shadowed every batch of a shape would keep a slow
    /// comparison in flight for as long as traffic lasts, and it would neither
    /// settle nor leave the device to the requests it serves.
    pub(super) shapes: std::collections::HashSet<::worker::plan::Shape>,
    /// Where the ask's answer goes, in the shape every stage's answer takes.
    pub(super) answer: crossbeam_channel::Sender<Result<(), String>>,
}

impl Scheduler {
    /// Removes a queued media request while preserving FIFO order for its peers.
    fn remove_waiting_media(&mut self, request_id: RequestId) -> Option<PendingMedia> {
        let index = self
            .waiting_media
            .iter()
            .position(|submission| submission.request.request_id == request_id)?;
        self.waiting_media.remove(index)
    }

    /// Removes a queued token request while preserving admission order for its peers.
    pub(super) fn remove_waiting(&mut self, request_id: RequestId) -> Option<RequestState> {
        let index = self
            .waiting
            .iter()
            .position(|queued| queued.req.request_id == request_id)?;
        self.waiting.remove(index)
    }

    /// Aborts every queued/gated/running request with a terminal event.
    ///
    /// `Scheduler::run` calls this on shutdown and after engine-fatal, and
    /// `Scheduler::step` on shutdown, each right before closing the worker.
    /// Shutdown covers both a `Shutdown` command and a disconnected command
    /// channel. Running requests finish immediately rather than waiting for
    /// their in-flight calls, and batches assembled but not yet submitted are
    /// dropped. Events still journaled behind a full channel, including these
    /// terminal events, are handed to their receivers, since no later flush
    /// will deliver them.
    pub(super) fn abort_all_requests(&mut self) {
        self.inflight.pending_submissions.clear();

        while let Some(submission) = self.waiting_media.pop_front() {
            let _ = submission.event_tx.send(Event::Finished {
                reason: FinishReason::Aborted,
                stop_reason: None,
                prompt_tokens: submission.request.prompt_token_ids.len(),
                completion_tokens: 0,
                images: 0,
            });
        }
        let media = self.media_ids();
        for id in media {
            self.finish_media(id, DiffusionTerminal::Finished(FinishReason::Aborted));
        }

        let queued: Vec<RequestId> = {
            let mut ids = Vec::new();
            while let Some(st) = self.waiting.pop_front() {
                ids.push(st.req.request_id);
                let _ = st.output.events.event_tx.send(Event::Finished {
                    reason: FinishReason::Aborted,
                    stop_reason: None,
                    prompt_tokens: st.charged_prompt_tokens(),
                    completion_tokens: 0,
                    images: 0,
                });
            }
            ids
        };
        let _ = queued;
        let running: Vec<RequestId> = self.running.keys().copied().collect();
        for id in running {
            self.finish(id, FinishReason::Aborted);
        }

        // Every running request has now retired its journal, so the retained
        // journals hold all output the control loop can no longer deliver.
        self.output.hand_off_retired();
    }

    /// Applies one command; returns true on shutdown.
    ///
    /// Every submission is validated by `enqueue` or `enqueue_media`, which
    /// refuse what the loaded workers cannot serve and report their own
    /// rejections.
    pub(super) fn handle_command(&mut self, cmd: Command) -> bool {
        // Work that finished on another thread before this command arrived is
        // taken first, so every answer describes the state the command found:
        // the slot holds the candidate a precopy prepared by the time its
        // answer was sent.
        self.take_prepared_policy();
        self.apply_policy_stage();

        match cmd {
            Command::Submit(submissions) => {
                for Submission { request, event_tx } in submissions {
                    match *request {
                        Request::Generation(request) => self.enqueue(*request, event_tx),
                        Request::Diffusion(request) => self.enqueue_media(request, event_tx, now()),
                    }
                }
            }
            Command::Cancel(request_id) => self.mark_cancelled(request_id, false),
            Command::Acknowledge {
                request_id,
                output_token_count,
            } => self.acknowledge_output(request_id, output_token_count),
            Command::Stop(request_id) => self.mark_stopped(request_id),
            Command::Abort(id) => self.mark_cancelled(id, true),
            Command::Park { lend, answer } => self.park(answer, lend),
            Command::Resume { reclaim, answer } => {
                let _ = answer.send(self.resume(reclaim));
            }
            Command::Promote {
                cache,
                recompute,
                grows,
                answer,
            } => {
                let _ = answer.send(self.promote(&cache, recompute, grows));
            }
            Command::Commit(answer) => {
                let _ = answer.send(self.commit());
            }
            Command::PublishEpoch(answer) => {
                let published = self
                    .worker
                    .publish_epoch()
                    .map_err(|error| format!("{error:#}"));
                let _ = answer.send(published);
            }
            Command::Retire { epochs, answer } => {
                let retired = self
                    .worker
                    .retire(&epochs)
                    .map_err(|error| format!("{error:#}"));
                let _ = answer.send(retired);
            }
            Command::KeepWeights {
                epochs,
                to_host,
                cache_keys,
                answer,
            } => {
                let kept = self
                    .worker
                    .keep_weights(&epochs, to_host, &cache_keys)
                    .map_err(|error| format!("{error:#}"));
                let _ = answer.send(kept);
            }
            Command::Ledgers(answer) => {
                let read = self.worker.ledgers().map_err(|error| format!("{error:#}"));
                let _ = answer.send(read);
            }
            Command::Throttle { halve, answer } => {
                let throttled = self
                    .worker
                    .throttle(halve)
                    .map_err(|error| format!("{error:#}"));
                let _ = answer.send(throttled);
            }
            Command::DescribeCache(answer) => {
                let _ = answer.send(self.describe_cache());
            }
            Command::DescribeRequests(answer) => {
                let _ = answer.send(self.migrating_cache().1);
            }
            Command::TakePolicyReport(answer) => {
                let _ = answer.send(self.take_policy_report());
            }
            Command::Progress { answer } => {
                let progress = policy::Progress {
                    step: self.step,
                    usage: self.policy.as_ref().map(policy::PolicySlot::usage),
                };
                let _ = answer.send(progress);
            }
            Command::PublishExpertLoad { load } => {
                self.set_expert_load(load);
            }
            Command::TakeExpertOrder(answer) => {
                let _ = answer.send(self.take_expert_order());
            }
            Command::DescribePolicyState { ranks, answer } => {
                // The state is the mechanism's, not the serving policy's: an
                // epoch's first precopy instantiates the epoch's policy on the
                // state the mechanism holds before any policy serves (§7.3
                // step 2).
                let state =
                    views::trial_state_with_ranks(self, self.step, self.recorder.recorded(), ranks);
                let _ = answer.send(Some(state));
            }
            Command::CacheWarm {
                operation,
                together,
                answer,
            } => {
                let _ = answer.send(self.warm_cache(operation, &together));
            }
            Command::FlushCache => {
                let pages = self.storage.flush_prefix_cache();
                tracing::info!(pages, "flushed the prefix cache");
            }
            Command::Hook {
                name,
                payload,
                together,
                answer,
            } => {
                let _ = answer.send(self.run_hook_together(&name, &payload, &together));
            }
            Command::Configure { limits, answer } => {
                let _ = answer.send(self.reconfigure(*limits));
            }
            Command::ResizeCache { change, answer } => self.resize_cache(change, answer),
            Command::UpdateWeights {
                model_path,
                in_flight,
                answer,
            } => {
                self.update_weights(model_path, in_flight, answer);
            }
            Command::CommitExperts {
                order,
                together,
                answer,
            } => {
                let committed = self.run_together(&together, |scheduler| {
                    scheduler
                        .worker
                        .commit_experts(&order)
                        .map_err(|error| format!("{error:#}"))
                });
                if let Err(error) = &committed {
                    // Recovery uses the committed distribution. No scheduler
                    // may issue a batch through a partially changed union.
                    self.fail(format!("expert commit failed: {error}"));
                }
                let _ = answer.send(committed);
            }
            Command::PrecopyPolicy {
                program,
                source,
                answer,
            } => self.precopy_policy(program, source, answer),
            Command::StagePolicy { stage, answer } => {
                self.request_stage(stage, Some(answer), None);
            }
            Command::EngineCommand { command, answer } => {
                self.engine_command(command, answer);
            }
            Command::ProgramCommand { command, answer } => {
                self.program_command(command, answer);
            }
            Command::Shadow { generation, answer } => {
                self.ask_shadow(generation, answer);
            }
            Command::Counters { answer } => {
                let _ = answer.send(self.counters());
            }
            Command::PolicyPrograms { answer } => {
                let _ = answer.send(handle::PolicyPrograms {
                    running: self.policy.as_ref().map(|policy| policy.program().clone()),
                    prepared: self
                        .policy_prepared
                        .as_ref()
                        .map(|candidate| candidate.program().clone()),
                });
            }
            Command::DiscardPolicy => {
                self.policy_prepared = None;
                self.policy_stage = None;
                if let Some(policy) = self.policy.as_mut() {
                    policy.discard();
                }
            }
            Command::Shutdown => return true,
        }
        // A precopy that finished on its blocking thread is taken here, and a
        // stage that waited for a boundary is applied when there is one.
        self.take_prepared_policy();
        self.apply_policy_stage();
        false
    }

    /// §7.3 step 2: prepares the next policy on this boundary's state.
    ///
    /// The state is read here, at a boundary, and the load, the instantiation
    /// and the trial of each decision run on a blocking thread, so the serving
    /// thread is never held by them. The answer arrives when that thread is
    /// done; [`Self::take_prepared_policy`] takes the candidate.
    fn precopy_policy(
        &mut self,
        program: PolicyProgram,
        source: Arc<dyn PolicySource + Send + Sync>,
        answer: crossbeam_channel::Sender<Result<(), String>>,
    ) {
        let answer_moved = answer.clone();
        let step = self.step;
        // The scheduler's own entry for §7.3 step 2, on the serving thread at a boundary: a log that
        // has the control thread's marker and not this line is a command that never left the queue,
        // and one that has this line and no resolve is a wait inside the preparation itself.
        tracing::info!(
            backend = %program.backend,
            revision = program.revision,
            step,
            "the scheduler took this precopy at a boundary and is preparing it"
        );
        let state = views::trial_state(self, step, self.recorder.recorded());
        let live = views::live(self, step, self.recorder.recorded());
        let ceilings = ceilings_of(BatchCapacity::of(&self.info));
        let engine = policy::process_engine().cloned();
        let (prepared, candidate) = crossbeam_channel::bounded(1);
        let spawned = std::thread::Builder::new()
            .name("policy-precopy".to_owned())
            .spawn(move || {
                let room = state.room.clone();
                let built = engine.and_then(|engine| {
                    let checked = mechanism::Checked::new(&live, &room);
                    PolicySlot::prepare(
                        &engine,
                        source.as_ref(),
                        program,
                        state,
                        &ceilings,
                        &checked,
                    )
                });
                let failed = built.as_ref().err().map(|error| error.to_string());
                let _ = prepared.send(built);
                let _ = match failed {
                    Some(error) => answer_moved.send(Err(error)),
                    None => answer_moved.send(Ok(())),
                };
            });
        match spawned {
            Ok(_) => self.policy_precopy = Some(candidate),
            Err(error) => {
                let _ = self.policy_precopy.take();
                let _ = answer.send(Err(format!("the precopy thread did not start: {error}")));
            }
        }
    }

    /// Installs `program` as this scheduler's policy without a control thread.
    ///
    /// §7.3's two stages run here on the calling thread: the precopy
    /// instantiates the program on this boundary's state and trials every
    /// decision of §4.4's table against it, and the switchover puts it in. An
    /// epoch with no control thread — the integration tests, and a simulated
    /// engine's — has no other way to have a policy, and a scheduler with no
    /// policy decides nothing, so without this such an epoch schedules no work
    /// at all.
    ///
    /// Production installs a policy through the `PrecopyPolicy` and
    /// `StagePolicy` commands, with the load, the instantiation and the trial on
    /// a blocking thread so the serving thread is never held; this is that pair
    /// without the thread and without the blocking hop, and it is for tests
    /// only: the caller pays the whole cost of the precopy.
    ///
    /// The stage is applied where a stage is always applied — at a boundary with
    /// nothing in flight — so the first call happens before the epoch submits
    /// anything.
    pub fn install_policy(
        &mut self,
        program: PolicyProgram,
        source: &dyn PolicySource,
    ) -> Result<(), String> {
        let state = views::trial_state(self, self.step, self.recorder.recorded());
        let live = views::live(self, self.step, self.recorder.recorded());
        let ceilings = ceilings_of(BatchCapacity::of(&self.info));
        let engine = policy::process_engine().map_err(|error| error.to_string())?;
        let room = state.room.clone();
        let checked = mechanism::Checked::new(&live, &room);
        let candidate = PolicySlot::prepare(engine, source, program, state, &ceilings, &checked)
            .map_err(|error| error.to_string())?;
        self.policy_prepared = Some(candidate);
        self.request_stage(Stage::Switchover, None, None);
        self.apply_policy_stage();
        if self.policy.is_none() {
            return Err("the policy did not become this scheduler's".to_owned());
        }
        Ok(())
    }

    /// Takes the candidate a precopy prepared, once its blocking thread is done.
    ///
    /// The scheduler that is about to switch reads it; a candidate that failed
    /// is dropped, and its failure is the answer the precopy already gave.
    pub(super) fn take_prepared_policy(&mut self) {
        let Some(precopy) = self.policy_precopy.as_ref() else {
            return;
        };
        let Ok(built) = precopy.try_recv() else {
            return;
        };
        self.policy_precopy = None;
        match built {
            Ok(candidate) => self.policy_prepared = Some(candidate),
            Err(error) => tracing::warn!(%error, "a policy precopy failed"),
        }
    }

    /// §7.3 step 2's shadow pass: the following step's batches also run in
    /// shadow for `generation`, the worker pairing that epoch's operators
    /// against the serving one's resident data without writing resident memory.
    ///
    /// The ask is kept until a step has shadowed its batches, which is when it
    /// is answered; a shadow submission the worker refuses answers it at once
    /// with the worker's reason. Only the in-process arm shadows: a supplied
    /// worker runs no shadow pass, so the ask is refused there rather than
    /// left waiting for a step that cannot run it.
    fn ask_shadow(&mut self, generation: u64, answer: crossbeam_channel::Sender<Result<(), String>>) {
        if !self.worker.serves_in_process() {
            let _ = answer.send(Err(format!(
                "the shadow pass of generation {generation} starts when a step submits through \
                 ::worker::Batches; this scheduler submits its batches to a supplied worker"
            )));
            return;
        }
        if let Some(pending) = self.shadow_pass.as_ref() {
            let _ = answer.send(Err(format!(
                "the shadow pass of generation {} is waiting for the step that runs it",
                pending.generation
            )));
            return;
        }
        self.shadow_pass = Some(ShadowPass {
            generation,
            shadowed: false,
            shapes: std::collections::HashSet::new(),
            answer,
        });
    }

    /// The epoch generation the following step's batches also run in shadow
    /// for, if an update asked (§7.3 step 2). The step reads it before it
    /// submits, since it names the kernels the shadow pass compares against.
    pub(super) fn shadow_generation(&self, batch: &ExecutionBatch) -> Option<u64> {
        let pass = self.shadow_pass.as_ref()?;
        let shape = batch
            .columns
            .as_ref()
            .map(crate::batching::columns::batch_shape);
        // A media batch states no token shape: its computation is compared once.
        let compared = match &shape {
            Some(shape) => pass.shapes.contains(shape),
            None => pass.shadowed,
        };
        (!compared).then_some(pass.generation)
    }

    /// Records that a batch of this step was submitted as a shadow pass: the
    /// ask is answered when the step ends, so an answer never claims a pass
    /// that has not run.
    pub(super) fn note_shadowed(&mut self, shape: Option<::worker::plan::Shape>) {
        if let Some(pass) = self.shadow_pass.as_mut() {
            pass.shadowed = true;
            pass.shapes.extend(shape);
        }
    }

    /// A shadow submission the worker refused: a shadow pass's failure is the
    /// update's (§7.3 step 2), so the ask is answered with it and the epoch
    /// serves on.
    pub(super) fn shadow_failed(&mut self, reason: impl Into<String>) {
        if let Some(pass) = self.shadow_pass.take() {
            let _ = pass.answer.send(Err(reason.into()));
        }
    }

    /// Answers an ask whose step has shadowed its batches, once every shadow
    /// batch of that pass has finished (7.3 step 2).
    ///
    /// The answer is the comparison's: a paired forward that left its
    /// operator's tolerance, or failed on the device, is the update's failure.
    /// Answering when the step ended rather than when the batches finished let
    /// a wrong kernel report as compared and reach the switchover.
    pub(super) fn settle_shadow(&mut self) {
        let shadowed = self.shadow_pass.as_ref().is_some_and(|pass| pass.shadowed);
        if !shadowed {
            return;
        }
        match self.worker.shadow_status() {
            // A device still working: the next pass of the loop asks again, and
            // the completion that ends it wakes this thread.
            crate::submission::ShadowStatus::Pending => {}
            crate::submission::ShadowStatus::Settled => {
                if let Some(pass) = self.shadow_pass.take() {
                    let _ = pass.answer.send(Ok(()));
                }
            }
            crate::submission::ShadowStatus::Failed(reason) => {
                self.shadow_failed(reason);
            }
        }
    }

    /// What this scheduler has done and what its policy has cost (§12).
    fn counters(&self) -> handle::Counters {
        let usage = self
            .policy
            .as_ref()
            .map(policy::PolicySlot::usage)
            .unwrap_or_default();
        handle::Counters {
            steps: self.step,
            batches: self
                .stats
                .general
                .steps
                .load(std::sync::atomic::Ordering::Relaxed),
            tokens: crate::lock(&self.stats.worker.forward_mode_tokens)
                .values()
                .sum(),
            policy_fuel: usage.fuel,
            policy_calls: usage.calls,
            kernel_launches: 0,
            device_ns: 0,
        }
    }

    /// Applies one command of the program-layer update (§2.3, §7.3) to this
    /// scheduler.
    ///
    /// The stages are a scheduler's: it cuts over at the boundary the command
    /// names, ends probation, reverts, and drops the generation before the
    /// serving one. A `Precopy` arrives with the operator programs, which are
    /// the program layer's, and the scheduling policy, whose module comes from
    /// the Program Server: the process that holds that source resolves it once
    /// and precopies each scheduler (`SchedulerHandle::precopy_policy`), so the
    /// command itself is answered with that rather than half applied here.
    fn program_command(
        &mut self,
        command: ipc::ProgramCommand,
        answer: crossbeam_channel::Sender<Result<(), String>>,
    ) {
        let (stage, at) = match command {
            ipc::ProgramCommand::Switchover { step, .. } => (Stage::Switchover, step),
            ipc::ProgramCommand::Commit { .. } => (Stage::Commit, None),
            ipc::ProgramCommand::Revert { .. } => (Stage::Revert, None),
            ipc::ProgramCommand::Purge { .. } => (Stage::Purge, None),
            ipc::ProgramCommand::Precopy { .. } => {
                let _ = answer.send(Err(
                    "a precopy's policy is resolved through the Program Server by the process \
                     that holds its client, which then precopies each scheduler's policy"
                        .to_owned(),
                ));
                return;
            }
        };
        self.request_stage(stage, Some(handle::StageAnswer::Applied(answer)), at);
    }

    /// Applies an engine command that concerns this scheduler (§7.3).
    ///
    /// The steps of an update are one scheduler's to apply — a switchover, a
    /// commit, a revert, a purge — so they are answered here. The rest belong to
    /// the process that owns what they name: a precopy is resolved by the
    /// process holding the Program Server (it instantiates one candidate per
    /// scheduler), and a park, promote, drain, serve or pause is the engine
    /// process's own lifecycle. Each answer names that rather than reporting
    /// success for work that did not happen.
    fn engine_command(
        &mut self,
        command: proto::process::Command,
        answer: crossbeam_channel::Sender<Result<(), String>>,
    ) {
        use proto::process::command::Kind;

        let stage = match command.kind {
            Some(Kind::Switchover(_)) => Stage::Switchover,
            Some(Kind::Commit(_)) => Stage::Commit,
            Some(Kind::Revert(_)) => Stage::Revert,
            Some(Kind::Purge(_)) => Stage::Purge,
            Some(Kind::Precopy(_)) => {
                let _ = answer.send(Err(
                    "a precopy is resolved once per epoch by the process that holds the Program \
                     Server, which then precopies each scheduler's policy"
                        .to_owned(),
                ));
                return;
            }
            Some(other) => {
                let _ = answer.send(Err(format!(
                    "{} is the main process's own command, not one scheduler's",
                    engine_command_name(&other)
                )));
                return;
            }
            None => {
                let _ = answer.send(Err("an engine command names no kind".to_owned()));
                return;
            }
        };
        self.request_stage(stage, Some(handle::StageAnswer::Applied(answer)), None);
    }

    /// Asks for a stage of §7.3, answered when it is applied.
    ///
    /// A stage changes the serving policy, so it is applied where no batch and
    /// no submission is in flight — never in the middle of one (§7.3: "在一个
    /// 批次边界（更新点）上"). A switchover with nothing prepared is refused at
    /// once: the daemon's precopy failed and the update is discarded.
    ///
    /// `at` is the step an expert group agreed to cut over at, when a peer main
    /// process named it: the stage then waits for that step and refuses to apply
    /// at any other.
    fn request_stage(
        &mut self,
        stage: Stage,
        answer: Option<handle::StageAnswer>,
        at: Option<u64>,
    ) {
        // One stage owns the next boundary. A second one is refused with the
        // first's name rather than replacing it: replacing it would drop the
        // first caller's answer, and applying both would change the serving
        // policy twice on one boundary.
        if let Some((waiting, _, _)) = &self.policy_stage {
            let waiting = *waiting;
            if let Some(answer) = answer {
                answer.send(Err(format!(
                    "a {} is already waiting for the boundary that applies it",
                    stage_name(waiting)
                )));
            }
            return;
        }
        if stage == Stage::Switchover {
            self.take_prepared_policy();
            if self.policy_prepared.is_none() {
                if let Some(answer) = answer {
                    answer.send(Err(
                        "a switchover arrived without a policy prepared by a precopy".to_owned(),
                    ));
                }
                return;
            }
        }
        self.policy_stage = Some((stage, answer, at));
        self.apply_policy_stage();
    }

    /// Applies the stage asked for, when the scheduler is at a boundary: no
    /// batch and no submission is in flight, so the policy that decided them is
    /// no longer needed by anything (§7.3 steps 3 to 6).
    /// Whether a policy stage waits for this boundary: one that names a later step is not due yet.
    pub(super) fn policy_stage_due(&self) -> bool {
        matches!(&self.policy_stage, Some((_, _, at)) if at.is_none_or(|at| self.step >= at))
    }

    pub(super) fn apply_policy_stage(&mut self) {
        let ready = self.inflight.pending_batches.is_empty()
            && self.inflight.pending_submissions.is_empty();
        if !ready {
            if self.policy_stage.is_some() {
                tracing::debug!(
                    batches = self.inflight.pending_batches.len(),
                    submissions = self.inflight.pending_submissions.len(),
                    step = self.step,
                    "a policy stage waits for a boundary with nothing in flight"
                );
            }
            return;
        }
        // A command that names the step its expert group agreed applies at that
        // step alone: before it, the stage waits; after it, the boundary the
        // other ranks took never comes here, so it is refused rather than
        // applied at a boundary they did not take.
        let agreed = match &self.policy_stage {
            Some((_, _, at)) => *at,
            None => return,
        };
        if let Some(at) = agreed {
            if self.step < at {
                return;
            }
            if self.step > at {
                let Some((stage, answer, _)) = self.policy_stage.take() else {
                    return;
                };
                if let Some(answer) = answer {
                    answer.send(Err(format!(
                        "the {} was to apply at step {at}, which this scheduler passed at step {}",
                        stage_name(stage),
                        self.step
                    )));
                }
                return;
            }
        }
        let Some((stage, answer, _at)) = self.policy_stage.take() else {
            return;
        };
        tracing::info!(
            stage = stage_name(stage),
            step = self.step,
            "a policy stage applies"
        );
        let state = views::trial_state(self, self.step, self.recorder.recorded());
        // The step the stage applied at: the boundary this scheduler reached,
        // where the policy it replaced had decided its last batch. The ranks of
        // an expert group take their steps on the union's batches, so their
        // boundaries carry the same number.
        let applied_at = handle::StageApplied { step: self.step };
        let applied = match self.stage_policy(stage, &state) {
            Ok(report) => {
                if let Some(report) = report {
                    self.policy_report = Some(report);
                }
                Ok(applied_at)
            }
            Err(reason) => {
                let program = self.policy.as_ref().map(|policy| policy.program().clone());
                if let Some(program) = program {
                    self.policy_report = Some(policy::PolicyReverted {
                        scheduler: None,
                        program,
                        reason: reason.clone(),
                        fatal: true,
                    });
                }
                Err(reason)
            }
        };
        tracing::info!(ok = applied.is_ok(), "a policy stage answered");
        if let Some(answer) = answer {
            answer.send(applied);
        }
    }

    /// Applies `stage` to the serving slot, installing an epoch's first policy
    /// from the candidate its precopy prepared.
    fn stage_policy(
        &mut self,
        stage: Stage,
        state: &TrialState,
    ) -> Result<Option<policy::PolicyReverted>, String> {
        let Some(policy) = self.policy.as_mut() else {
            // An epoch's first switchover: no policy served before, so the
            // candidate's precopy prepared the only instance this policy has.
            let Some(candidate) = self.policy_prepared.take() else {
                return Ok(None);
            };
            let engine = policy::process_engine().map_err(|error| error.to_string())?;
            let ceilings = ceilings_of(BatchCapacity::of(&self.info));
            self.adopt(PolicySlot::from_candidate(engine, ceilings, candidate));
            // From here the mechanism decides: a scheduler with no policy
            // forms no batch, and the serving path's warm-up waits for this.
            self.note_policy_serves();
            return Ok(None);
        };
        let mut serves = false;
        if stage == Stage::Switchover {
            let Some(candidate) = self.policy_prepared.take() else {
                return Err("a switchover arrived without a prepared policy".to_owned());
            };
            policy.adopt(candidate);
            serves = true;
        }
        policy.request(stage);
        let applied = policy.boundary(state).map_err(|error| error.to_string());
        if applied.is_ok() && matches!(stage, Stage::Switchover | Stage::Revert) {
            self.seed_from_policy();
        }
        if serves {
            // A policy instance serves from this boundary on (§7.3 step 3).
            self.note_policy_serves();
        }
        applied
    }

    /// Selects commands with available destinations without bypassing an older
    /// command for the same request. Other requests remain independently eligible.
    ///
    /// A command is deferred when `include` rejects it or
    /// `Worker::command_has_capacity` reports a full destination; deferred
    /// commands keep their relative order in `pending_commands`.
    pub(super) fn take_commands(
        &mut self,
        include: impl Fn(&BatchCommand) -> bool,
    ) -> Vec<BatchCommand> {
        let mut blocked = HashSet::new();
        let mut selected = Vec::new();
        for command in std::mem::take(&mut self.inflight.pending_commands) {
            let request = command.request_key();
            // Once one command of a request is deferred, every later command
            // of that request is deferred with it.
            if blocked.contains(&request)
                || !include(&command)
                || !self.worker.command_has_capacity(&command)
            {
                blocked.insert(request);
                self.inflight.pending_commands.push_back(command);
            } else {
                selected.push(command);
            }
        }
        selected
    }

    /// Tells the policy that a request ended, when the mechanism ended it
    /// itself.
    ///
    /// A policy's state is append-only: it learns a request from `Arrived` and
    /// nothing else about it unless the mechanism says so. A queued request the
    /// mechanism drops without an `End` stays in the policy's own queue, and
    /// the next `dispatch` names it in `admit`, where the mechanism refuses it
    /// as "not queued" — a policy failure, on a name the policy could not have
    /// known was stale. So every pop of a queue the policy did not fill is
    /// announced here, with the end that pop means.
    pub(super) fn announce_end(&mut self, id: RequestId, end: scheduling::guest::event::End) {
        self.recorder.record(views::ended(id, end));
    }

    /// Records cancellation or abortion across queued, media, and running request states.
    ///
    /// A queued request finishes here with `Aborted` or `Cancelled`. A running
    /// media request records the reason, keeping an earlier terminal intent,
    /// and finishes here only when none of its calls are in flight. A running
    /// token request drops its pending decoder decisions, records the reason
    /// as its terminal intent and is finished by `reap_cancellations` once its
    /// calls drain. Unknown requests are ignored.
    pub(super) fn mark_cancelled(&mut self, id: RequestId, abort: bool) {
        if let Some(submission) = self.remove_waiting_media(id) {
            let reason = if abort {
                FinishReason::Aborted
            } else {
                FinishReason::Cancelled
            };
            // Both reasons are the client's will: it cancelled the request, or
            // it asked for it to be dropped, and the epoch goes on either way.
            self.announce_end(id, scheduling::guest::event::End::Cancelled);
            let _ = submission.event_tx.send(Event::Finished {
                reason,
                stop_reason: None,
                prompt_tokens: submission.request.prompt_token_ids.len(),
                completion_tokens: 0,
                images: 0,
            });
            return;
        }
        if let Some(state) = self.media_state_mut(id) {
            state.terminal_intent.finish(if abort {
                FinishReason::Aborted
            } else {
                FinishReason::Cancelled
            });
            let drained = !self.inflight.has_pending_calls(id);
            if drained {
                self.finish_media(
                    id,
                    DiffusionTerminal::Finished(if abort {
                        FinishReason::Aborted
                    } else {
                        FinishReason::Cancelled
                    }),
                );
            }
            return;
        }
        if let Some(st) = self.running.get_mut(&id) {
            st.output.decoder_boundaries.clear();
            st.terminal_intent = if abort {
                TerminalIntent::Finish(FinishReason::Aborted)
            } else {
                TerminalIntent::Finish(FinishReason::Cancelled)
            };
        }
        // A request not yet admitted leaves the waiting queue with its reason.
        if let Some(st) = self.remove_waiting(id) {
            self.announce_end(id, scheduling::guest::event::End::Cancelled);
            let reason = if abort {
                FinishReason::Aborted
            } else {
                FinishReason::Cancelled
            };
            let _ = st.output.events.event_tx.send(Event::Finished {
                reason,
                stop_reason: None,
                prompt_tokens: st.charged_prompt_tokens(),
                completion_tokens: 0,
                images: 0,
            });
        }
    }

    /// Resolves the decoder decisions a prefix accepted by the frontend
    /// decoder covers.
    ///
    /// Every decoder boundary within the first `output_token_count` public
    /// tokens is resolved, which releases scheduling held on those decisions
    /// and may apply a finish deferred on them. Requests that are not running
    /// are ignored.
    pub(super) fn acknowledge_output(&mut self, id: RequestId, output_token_count: usize) {
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        while state
            .output
            .decoder_boundaries
            .front()
            .is_some_and(|count| *count <= output_token_count)
        {
            state.output.decoder_boundaries.pop_front();
        }
        self.finish_pending_if_idle(id);
    }

    /// Ends every token request that resumes from its generated tokens as
    /// migratable, for `cause`.
    ///
    /// A request resumes from the tokens and images it generated when it is
    /// no readout (`GenerationRequest::is_migratable`), on a deployment that
    /// does not generate in canvases. A queued one ends at once. A running
    /// one stops scheduling, drops the decoder decisions it awaits, and ends
    /// once its calls in flight drain, after every token and image it
    /// generated, whatever it computes; tokens those calls produce are
    /// discarded, and the engine that serves next generates them again.
    /// A completed image is published before migration because its decoding
    /// releases the latent trajectory. The denoising steps of an image it had
    /// begun advance its trajectory, which is stored in the host latent
    /// pools before the request leaves (`Scheduler::retire_preempted_token`)
    /// and which the engine that serves next continues. A request already
    /// finishing finishes. Video requests and readouts go on.
    pub(super) fn migrate_requests(&mut self, cause: MigrationCause) {
        if self.config.canvas.is_some() {
            return;
        }
        let queued: Vec<RequestId> = self
            .waiting
            .iter()
            .filter(|st| st.req.is_migratable())
            .map(|st| st.req.request_id)
            .collect();
        for id in queued {
            if let Some(st) = self.remove_waiting(id) {
                self.announce_end(id, scheduling::guest::event::End::Migratable);
                let _ = st.output.events.event_tx.send(Event::Migratable {
                    cause,
                    denoised_steps: 0,
                    stored_layout: Vec::new(),
                });
            }
        }
        self.migrate_running(cause, |_| true);
    }

    /// Ends as migratable, for `cause`, every running token request that
    /// `select` picks and that resumes from its generated tokens, as
    /// `migrate_requests` ends a running one.
    pub(super) fn migrate_running(
        &mut self,
        cause: MigrationCause,
        select: impl Fn(RequestId) -> bool,
    ) {
        if self.config.canvas.is_some() {
            return;
        }
        for (id, st) in &mut self.running {
            if select(*id)
                && st.req.is_migratable()
                && !st.terminal_intent.is_terminal()
                && !self.inflight.pending_finishes.contains_key(id)
            {
                st.output.decoder_boundaries.clear();
                st.terminal_intent = TerminalIntent::Migrate(cause);
            }
        }
    }

    /// Stops a running token request whose frontend decoder matched a stop
    /// string.
    ///
    /// Drops pending decoder decisions and sets a `Stop` terminal intent that
    /// `reap_cancellations` applies once the request's calls drain. Requests
    /// that are not running are ignored.
    pub(super) fn mark_stopped(&mut self, id: RequestId) {
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        state.output.decoder_boundaries.clear();
        state.terminal_intent = TerminalIntent::Finish(FinishReason::Stop);
    }
}
