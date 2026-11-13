//! Cancellation, terminal cleanup, and lifecycle handling.
//!
//! The scheduler thread applies each frontend `Command` (see `crate::handle`)
//! here. Cancelling a waiting request finishes it at once. A running request
//! has its end decided and closes only after every submitted call has
//! resolved: a media request with nothing in flight finishes immediately, and
//! `reap_cancellations` finishes the rest once their calls drain.
//! Acknowledgements resolve the stop-string decisions a request awaits
//! (`RequestOutput::decoder_boundaries`), and `abort_all_requests` ends every
//! request when the control loop stops.

use super::*;
use crate::handle;
use crate::request::Request;

/// The name a stage's refusal uses.
fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::Switch => "switch",
        Stage::Commit => "commit",
        Stage::Rollback => "rollback",
        Stage::Release => "release",
    }
}

/// What an engine command this scheduler does not apply itself is called, for
/// the response that names its owner.
fn engine_command_name(kind: &proto::process::command::Kind) -> &'static str {
    use proto::process::command::Kind;
    match kind {
        Kind::Prepare(_) => "prepare",
        Kind::Switch(_) => "switch",
        Kind::Commit(_) => "commit",
        Kind::Rollback(_) => "rollback",
        Kind::Drain(_) => "drain",
        Kind::Promote(_) => "promote",
        Kind::Pause(_) => "pause",
        Kind::Serve(_) => "serve",
    }
}

/// An update's ask that the following step's batches also run in shadow for an
/// epoch generation (§7.3 step 2), with the response that waits for the step which
/// shadows them.
///
/// The ask is kept until it is responded: a refused shadow submission responds to it
/// at once with the worker's reason, and a submitted one responds to it when its
/// step has ended, so a response never states a pass that has not run.
pub(super) struct ShadowPass {
    /// The epoch generation whose operators the shadow pass runs.
    pub(super) generation: u64,
    /// One of the step's batches was submitted as a shadow: the step that owns
    /// the response has begun, and ends before the response is sent.
    pub(super) shadowed: bool,
    /// The shapes this pass has compared. A shape is compared once (§7.3 step
    /// 2): a pass that shadowed every batch of a shape would keep a slow
    /// comparison in flight for as long as traffic lasts, and it would neither
    /// settle nor leave the device to the requests it serves.
    pub(super) shapes: std::collections::HashSet<ipc::execution::Shape>,
    /// Where the ask's response goes, in the shape every stage's response takes.
    pub(super) response: crossbeam_channel::Sender<Result<(), String>>,
}

impl Scheduler {
    /// Takes a waiting request out of the queue and the request table,
    /// preserving the order of its peers; `None` for a request that is not
    /// waiting.
    pub(super) fn remove_waiting(&mut self, id: RequestId) -> Option<Request> {
        if let Some(index) = self.held_for_index.iter().position(|held| *held == id) {
            self.held_for_index.remove(index);
        } else {
            let index = self.waiting.iter().position(|queued| *queued == id)?;
            self.waiting.remove(index);
        }
        self.requests.remove(&id)
    }

    /// Aborts every queued/gated/running request with a terminal event.
    ///
    /// `Scheduler::run` calls this on shutdown and after an engine-fatal fault.
    /// Shutdown covers both a `Shutdown` command and a disconnected command
    /// channel. Running requests finish immediately rather than waiting for
    /// their in-flight calls, and batches assembled but not yet submitted are
    /// dropped. Events still journaled behind a full channel, including these
    /// terminal events, are given to their receivers, since no later flush
    /// will deliver them.
    pub(super) fn abort_all_requests(&mut self) {
        self.release_held_for_index();
        self.inflight.pending_submissions.clear();

        while let Some(id) = self.waiting.pop_front() {
            if let Some(request) = self.requests.remove(&id) {
                let _ = request.event_tx().send(Event::Finished {
                    reason: FinishReason::Aborted,
                    stop_reason: None,
                    prompt_tokens: request.prompt_tokens(),
                    completion_tokens: 0,
                });
            }
        }
        for id in self.running.clone() {
            self.retire(id, Terminal::finish(FinishReason::Aborted));
        }

        // Every running request has now retired its journal, so the retained
        // journals hold all output the control loop can no longer deliver.
        self.output.transfer_retired();
    }

    /// Applies one command; returns true on shutdown.
    ///
    /// Every submission is validated by `add_request`, which refuses what
    /// the loaded workers cannot serve and reports its own rejections.
    pub(super) fn handle_command(&mut self, cmd: Command) -> bool {
        // Work that finished on another thread before this command arrived is
        // taken first, so every response describes the state the command found:
        // the slot holds the candidate a precopy prepared by the time its
        // response was sent.
        self.take_prepared_policy();
        self.apply_policy_stage();

        match cmd {
            Command::Submit(submissions) => {
                for Submission { request, event_tx } in submissions {
                    self.add_request(*request, event_tx);
                }
            }
            Command::Cancel(request_id) => self.cancel(request_id),
            Command::Acknowledge {
                request_id,
                output_token_count,
            } => self.acknowledge_output(request_id, output_token_count),
            Command::Stop(request_id) => self.mark_stopped(request_id),
            Command::Abort(id) => self.mark_cancelled(id, true),
            Command::Park { lend, response } => self.park(response, lend),
            Command::Resume {
                take_back,
                response,
            } => {
                let _ = response.send(self.resume(take_back));
            }
            Command::Promote {
                cache,
                recompute,
                grows,
                response,
            } => {
                let _ = response.send(self.promote(&cache, recompute, grows));
            }
            Command::AwaitIndex(bound) => {
                self.awaiting_index = Some(std::time::Instant::now() + bound);
            }
            Command::Inherit { caches, response } => {
                let _ = response.send(self.inherit(&caches));
            }
            Command::Commit(response) => {
                let _ = response.send(self.commit());
            }
            // The four storage commands below concern storage the ranks of this
            // process do not hold through their scheduler: the engine's storage path
            // publishes and retires epochs, keeps weights and reads the quota, so a
            // scheduler has nothing to do and nothing to report.
            Command::PublishEpoch(response) | Command::Retire { response, .. } => {
                let _ = response.send(Ok(()));
            }
            Command::KeepWeights { response, .. } | Command::Usage(response) => {
                let _ = response.send(Ok(Vec::new()));
            }
            Command::DescribeCache(response) => {
                let _ = response.send(self.describe_cache());
            }
            Command::DescribeRequests(response) => {
                let _ = response.send(self.migrating_cache().1);
            }
            Command::TakePolicyReport(response) => {
                let _ = response.send(self.take_policy_report());
            }
            Command::Progress { response } => {
                let progress = policy::Progress {
                    step: self.step,
                    usage: self.policy.as_ref().map(policy::PolicySlot::usage),
                };
                let _ = response.send(progress);
            }
            Command::CacheWarm {
                operation,
                together,
                response,
            } => {
                let _ = response.send(self.warm_cache(operation, &together));
            }
            Command::FlushCache => {
                let pages = self.storage.flush_prefix_cache();
                tracing::info!(pages, "flushed the prefix cache");
            }
            Command::Configure { limits, response } => {
                let _ = response.send(self.reconfigure(*limits));
            }
            Command::ResizeCache { change, response } => self.resize_cache(change, response),
            Command::UpdateWeights {
                model_path,
                in_flight,
                response,
            } => {
                self.update_weights(model_path, in_flight, response);
            }
            Command::CommitExperts {
                order,
                together,
                response,
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
                let _ = response.send(committed);
            }
            Command::PreparePolicy {
                package,
                source,
                response,
            } => self.prepare_policy(package, source, response),
            Command::StagePolicy { stage, response } => {
                self.request_stage(stage, Some(response), None);
            }
            Command::EngineCommand { command, response } => {
                self.engine_command(command, response);
            }
            Command::GenerationCommand { command, response } => {
                self.generation_command(command, response);
            }
            Command::Shadow {
                generation,
                response,
            } => {
                self.ask_shadow(generation, response);
            }
            Command::Counters { response } => {
                let _ = response.send(self.counters());
            }
            Command::PolicyPackages { response } => {
                let _ = response.send(handle::PolicyPackages {
                    running: self.policy.as_ref().map(|policy| policy.package().clone()),
                    prepared: self
                        .policy_prepared
                        .as_ref()
                        .map(|candidate| candidate.package().clone()),
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
    /// thread is never held by them. The response arrives when that thread is
    /// done; [`Self::take_prepared_policy`] takes the candidate.
    fn prepare_policy(
        &mut self,
        package: PolicyPackage,
        source: Arc<dyn PolicySource + Send + Sync>,
        response: crossbeam_channel::Sender<Result<(), String>>,
    ) {
        let response_moved = response.clone();
        let step = self.step;
        // The scheduler's own entry for §7.3 step 2, on the serving thread at a boundary: a log that
        // has the control thread's marker and not this line is a command that never left the queue,
        // and one that has this line and no resolve is a wait inside the preparation itself.
        tracing::info!(
            backend = %package.backend,
            revision = package.revision,
            step,
            "the scheduler took this precopy at a boundary and is preparing it"
        );
        let state = views::trial_state(self, step, self.recorder.recorded());
        let live = views::live(self, step, self.recorder.recorded());
        let ceilings = ceilings_of(BatchCapacity::of(&self.info));
        let engine = policy::process_engine().cloned();
        let (prepared, candidate) = crossbeam_channel::bounded(1);
        // The response is sent from the blocking thread, so that thread wakes the control
        // thread that waits for it.
        let progress = self.progress.clone();
        let spawned = std::thread::Builder::new()
            .name("policy-prepare".to_owned())
            .spawn(move || {
                let room = state.room.clone();
                let built = engine.and_then(|engine| {
                    let checked = mechanism::Checked::new(&live, &room);
                    PolicySlot::prepare(
                        &engine,
                        source.as_ref(),
                        package,
                        state,
                        &ceilings,
                        &checked,
                    )
                });
                let failed = built.as_ref().err().map(|error| error.to_string());
                let _ = prepared.send(built);
                let _ = match failed {
                    Some(error) => response_moved.send(Err(error)),
                    None => response_moved.send(Ok(())),
                };
                progress.wake();
            });
        match spawned {
            Ok(_) => self.policy_prepare = Some(candidate),
            Err(error) => {
                let _ = self.policy_prepare.take();
                let _ = response.send(Err(format!("the precopy thread did not start: {error}")));
            }
        }
    }

    /// Takes the candidate a precopy prepared, once its blocking thread is done.
    ///
    /// The scheduler that is about to switch reads it; a candidate that failed
    /// is dropped, and its failure is the response the precopy already gave.
    pub(super) fn take_prepared_policy(&mut self) {
        let Some(precopy) = self.policy_prepare.as_ref() else {
            return;
        };
        let Ok(built) = precopy.try_recv() else {
            return;
        };
        self.policy_prepare = None;
        match built {
            Ok(candidate) => self.policy_prepared = Some(candidate),
            Err(error) => tracing::warn!(%error, "a policy precopy failed"),
        }
    }

    /// §7.3 step 2's shadow pass: the following step's batches also run in
    /// shadow for `generation`, the worker pairing that epoch's operators
    /// against the serving one's data without writing its memory.
    ///
    /// The ask is kept until a step has shadowed its batches, which is when it
    /// is responded; a shadow submission the worker refuses responds to it at once
    /// with the worker's reason. Only the in-process arm shadows: a supplied
    /// worker runs no shadow pass, so the ask is refused there rather than
    /// left waiting for a step that cannot run it.
    fn ask_shadow(
        &mut self,
        generation: u64,
        response: crossbeam_channel::Sender<Result<(), String>>,
    ) {
        if let Some(pending) = self.shadow_pass.as_ref() {
            let _ = response.send(Err(format!(
                "the shadow pass of generation {} is waiting for the step that runs it",
                pending.generation
            )));
            return;
        }
        self.shadow_pass = Some(ShadowPass {
            generation,
            shadowed: false,
            shapes: std::collections::HashSet::new(),
            response,
        });
    }

    /// The epoch generation the following step's batches also run in shadow
    /// for, if an update asked (§7.3 step 2). The step reads it before it
    /// submits, since it names the kernels the shadow pass compares against.
    pub(super) fn shadow_generation(&self, batch: &ExecutionBatch) -> Option<u64> {
        let pass = self.shadow_pass.as_ref()?;
        let shape = batch.token_shape();
        // A media batch states no token shape: its computation is compared once.
        let compared = match &shape {
            Some(shape) => pass.shapes.contains(shape),
            None => pass.shadowed,
        };
        (!compared).then_some(pass.generation)
    }

    /// Records that a batch of this step was submitted as a shadow pass: the
    /// ask gets its response when the step ends, so a response never states a pass
    /// that has not run.
    pub(super) fn note_shadowed(&mut self, shape: Option<ipc::execution::Shape>) {
        if let Some(pass) = self.shadow_pass.as_mut() {
            pass.shadowed = true;
            pass.shapes.extend(shape);
        }
    }

    /// A shadow submission the worker refused: a shadow pass's failure is the
    /// update's (§7.3 step 2), so the ask gets its response with it and the epoch
    /// serves on.
    pub(super) fn shadow_failed(&mut self, reason: impl Into<String>) {
        if let Some(pass) = self.shadow_pass.take() {
            let _ = pass.response.send(Err(reason.into()));
        }
    }

    /// Responds to an ask whose step has shadowed its batches, once every shadow
    /// batch of that pass has finished (7.3 step 2).
    ///
    /// The response is the comparison's: a paired forward that left its
    /// operator's tolerance, or failed on the device, is the update's failure.
    /// Responding when the step ended rather than when the batches finished let
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
                    let _ = pass.response.send(Ok(()));
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

    /// Applies one command of the generation update (§2.3, §7.3) to this
    /// scheduler.
    ///
    /// The stages are a scheduler's: it cuts over at the boundary the command
    /// names, ends probation, reverts, and drops the generation before the
    /// serving one. A `Precopy` arrives with the operator packages, which are
    /// the generation layer's, and the scheduling policy, whose module comes from
    /// the Build Server: the process that holds that source resolves it once
    /// and precopies each scheduler (`SchedulerHandle::prepare_policy`), so the
    /// command itself gets its response with that rather than half applied here.
    fn generation_command(
        &mut self,
        command: ipc::GenerationCommand,
        response: crossbeam_channel::Sender<Result<(), String>>,
    ) {
        let (stage, at) = match command {
            ipc::GenerationCommand::Switch { step, .. } => (Stage::Switch, step),
            ipc::GenerationCommand::Commit { .. } => (Stage::Commit, None),
            ipc::GenerationCommand::Rollback { .. } => (Stage::Rollback, None),
            ipc::GenerationCommand::Prepare { .. } => {
                let _ = response.send(Err(
                    "a prepare's policy is resolved through the Build Server by the process \
                     that holds its client, which then prepares each scheduler's policy"
                        .to_owned(),
                ));
                return;
            }
        };
        self.request_stage(stage, Some(handle::StageResponse::Applied(response)), at);
    }

    /// Applies an engine command that concerns this scheduler (§7.3).
    ///
    /// The steps of an update are one scheduler's to apply — a switchover, a
    /// commit, a revert, a purge — so they get their responses here. The rest belong to
    /// the process that owns what they name: a precopy is resolved by the
    /// process holding the Build Server (it instantiates one candidate per
    /// scheduler), and a park, promote, drain, serve or pause is the engine
    /// process's own lifecycle. Each response names that rather than reporting
    /// success for work that did not happen.
    fn engine_command(
        &mut self,
        command: proto::process::Command,
        response: crossbeam_channel::Sender<Result<(), String>>,
    ) {
        use proto::process::command::Kind;

        let stage = match command.kind {
            Some(Kind::Switch(_)) => Stage::Switch,
            Some(Kind::Commit(_)) => Stage::Commit,
            Some(Kind::Rollback(_)) => Stage::Rollback,
            Some(Kind::Prepare(_)) => {
                let _ = response.send(Err(
                    "a prepare is resolved once per epoch by the process that holds the Build \
                     Server, which then prepares each scheduler's policy"
                        .to_owned(),
                ));
                return;
            }
            Some(other) => {
                let _ = response.send(Err(format!(
                    "{} is the main process's own command, not one scheduler's",
                    engine_command_name(&other)
                )));
                return;
            }
            None => {
                let _ = response.send(Err("an engine command names no kind".to_owned()));
                return;
            }
        };
        self.request_stage(stage, Some(handle::StageResponse::Applied(response)), None);
    }

    /// Asks for a stage of §7.3, responded when it is applied.
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
        response: Option<handle::StageResponse>,
        at: Option<u64>,
    ) {
        // One stage owns the next boundary. A second one is refused with the
        // first's name rather than replacing it: replacing it would drop the
        // first caller's response, and applying both would change the serving
        // policy twice on one boundary.
        if let Some((waiting, _, _)) = &self.policy_stage {
            let waiting = *waiting;
            if let Some(response) = response {
                response.send(Err(format!(
                    "a {} is already waiting for the boundary that applies it",
                    stage_name(waiting)
                )));
            }
            return;
        }
        if stage == Stage::Switch {
            self.take_prepared_policy();
            if self.policy_prepared.is_none() {
                if let Some(response) = response {
                    response.send(Err(
                        "a switchover arrived without a policy prepared by a precopy".to_owned(),
                    ));
                }
                return;
            }
        }
        self.policy_stage = Some((stage, response, at));
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
                let Some((stage, response, _)) = self.policy_stage.take() else {
                    return;
                };
                if let Some(response) = response {
                    response.send(Err(format!(
                        "the {} was to apply at step {at}, which this scheduler passed at step {}",
                        stage_name(stage),
                        self.step
                    )));
                }
                return;
            }
        }
        let Some((stage, response, _at)) = self.policy_stage.take() else {
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
                let package = self.policy.as_ref().map(|policy| policy.package().clone());
                if let Some(package) = package {
                    self.policy_report = Some(policy::PolicyRolledBack {
                        scheduler: None,
                        package,
                        reason: reason.clone(),
                        fatal: true,
                    });
                }
                Err(reason)
            }
        };
        tracing::info!(ok = applied.is_ok(), "a policy stage responded");
        if let Some(response) = response {
            response.send(applied);
        }
    }

    /// Applies `stage` to the serving slot, installing an epoch's first policy
    /// from the candidate its precopy prepared.
    fn stage_policy(
        &mut self,
        stage: Stage,
        state: &TrialState,
    ) -> Result<Option<policy::PolicyRolledBack>, String> {
        let Some(policy) = self.policy.as_mut() else {
            // An epoch's first switchover: no policy served before, so the
            // candidate's precopy prepared the only instance this policy has.
            // A commit or a purge with no policy changes nothing; a switchover
            // or a revert that would leave the scheduler with no policy fails,
            // since a scheduler with no policy forms no batch.
            if matches!(stage, Stage::Commit | Stage::Release) {
                return Ok(None);
            }
            let Some(candidate) = self.policy_prepared.take() else {
                return Err(format!(
                    "a {} found no serving policy and no policy prepared by a precopy",
                    stage_name(stage)
                ));
            };
            if stage == Stage::Rollback {
                // A revert before any policy served has nothing to go back to.
                return Err("a revert found no serving policy to revert".to_owned());
            }
            let engine = policy::process_engine().map_err(|error| error.to_string())?;
            let ceilings = ceilings_of(BatchCapacity::of(&self.info));
            // From here the mechanism decides: a scheduler with no policy
            // forms no batch.
            self.adopt(PolicySlot::from_candidate(engine, ceilings, candidate));
            return Ok(None);
        };
        if stage == Stage::Switch {
            let Some(candidate) = self.policy_prepared.take() else {
                return Err("a switchover arrived without a prepared policy".to_owned());
            };
            policy.adopt(candidate);
        }
        policy.request(stage);
        let applied = policy.boundary(state).map_err(|error| error.to_string());
        if applied.is_ok() && matches!(stage, Stage::Switch | Stage::Rollback) {
            self.seed_from_policy();
        }
        applied
    }

    /// Selects commands with available destinations without bypassing an older
    /// command for the same request. Other requests remain independently eligible.
    ///
    /// A command is deferred when `include` rejects it; deferred commands
    /// keep their relative order in `pending_commands`.
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
            if blocked.contains(&request) || !include(&command) {
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
    /// nothing else about it unless the mechanism says so. A waiting request the
    /// mechanism drops without an `End` stays in the policy's own queue, and
    /// the next `dispatch` names it in `admit`, where the mechanism refuses it
    /// as "not queued" — a policy failure, on a name the policy could not have
    /// known was stale. So every pop of a queue the policy did not fill is
    /// announced here, with the end that pop means.
    pub(super) fn announce_end(&mut self, id: RequestId, end: scheduling::guest::event::End) {
        self.recorder.record(views::ended(id, end));
    }

    /// Cancels the request `id` for its client: a waiting request finishes
    /// at once with `Cancelled`, a running one once its calls in flight
    /// drain. An unknown request is ignored.
    pub fn cancel(&mut self, id: RequestId) {
        self.mark_cancelled(id, false);
    }

    /// Ends every request as migratable for `cause`, to be submitted again
    /// with what it generated: the engine is being drained, or lost.
    ///
    /// A waiting request leaves at once. A running token request ends once
    /// its calls in flight drain, with the tokens it generated; a running
    /// video request finishes its calls in flight, stores its trajectory at
    /// its next denoising step boundary and ends with its stored step.
    pub fn drain(&mut self, cause: MigrationCause) {
        // The requests held for a predecessor's index leave with the others.
        self.release_held_for_index();
        self.migrate_requests(cause);
        self.preempt_media(cause, |_| true);
    }

    /// Cancels the request `id` for its client, or drops it when `abort`.
    ///
    /// A waiting request finishes here with `Aborted` or `Cancelled`. A
    /// running video request records the reason, keeping an earlier decided
    /// end, and finishes here only when none of its calls are in flight. A
    /// running token request drops its pending decoder decisions, records
    /// the reason as its end and is retired by `reap_cancellations` once its
    /// calls drain. Unknown requests are ignored.
    pub(super) fn mark_cancelled(&mut self, id: RequestId, abort: bool) {
        let reason = if abort {
            FinishReason::Aborted
        } else {
            FinishReason::Cancelled
        };
        let Some(request) = self.requests.get_mut(&id) else {
            return;
        };
        if !request.admitted() {
            // Both reasons are the client's will: it cancelled the request, or
            // it asked for it to be dropped, and the epoch goes on either way.
            let prompt_tokens = request.prompt_tokens();
            self.end_waiting(
                id,
                scheduling::guest::event::End::Cancelled,
                Event::Finished {
                    reason,
                    stop_reason: None,
                    prompt_tokens,
                    completion_tokens: 0,
                },
            );
            return;
        }
        match &mut request.graph {
            Graph::Video(_) => {
                request.status.finish(Terminal::finish(reason.clone()));
                if !self.inflight.has_pending_calls(id) {
                    self.retire(id, Terminal::finish(reason));
                }
            }
            Graph::Tokens(state) => {
                state.output.decoder_boundaries.clear();
                request.status = RequestStatus::Finished(Terminal::finish(reason));
            }
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
        let Some(state) = self.requests.tokens_mut(&id) else {
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
    /// A request resumes from the tokens and images it generated on a
    /// deployment that does not generate in blocks. A waiting one ends at
    /// once. A running one stops scheduling, drops the decoder decisions it
    /// awaits, and ends
    /// once its calls in flight drain, after every token and image it
    /// generated, whatever it computes; tokens those calls produce are
    /// discarded, and the engine that serves next generates them again.
    /// A completed image is published before migration because its decoding
    /// releases the latent trajectory. The denoising steps of an image it had
    /// begun advance its trajectory, which is stored in the host latent
    /// pools before the request leaves (`Scheduler::retire_preempted_token`)
    /// and which the engine that serves next continues. A request already
    /// finishing finishes. Video requests go on.
    pub(super) fn migrate_requests(&mut self, cause: MigrationCause) {
        if crate::BLOCK_SAMPLING.is_some() {
            return;
        }
        let waiting: Vec<RequestId> = self
            .waiting
            .iter()
            .chain(&self.held_for_index)
            .copied()
            .filter(|id| {
                self.requests
                    .get(id)
                    .is_some_and(|request| request.tokens().is_some())
            })
            .collect();
        for id in waiting {
            self.end_waiting(
                id,
                scheduling::guest::event::End::Migratable,
                Event::Migratable {
                    cause,
                    denoised_steps: 0,
                    stored_layout: Vec::new(),
                },
            );
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
        if crate::BLOCK_SAMPLING.is_some() {
            return;
        }
        for id in self.running.clone() {
            if !select(id) || self.inflight.pending_finishes.contains_key(&id) {
                continue;
            }
            let Some(request) = self.requests.get_mut(&id) else {
                continue;
            };
            if request.status.is_finished() {
                continue;
            }
            if let Graph::Tokens(state) = &mut request.graph {
                state.output.decoder_boundaries.clear();
                request.status = RequestStatus::Finished(Terminal::migrate(cause));
            }
        }
    }

    /// Stops a running token request whose frontend decoder matched a stop
    /// string.
    ///
    /// Drops pending decoder decisions and decides a `Stop` end that
    /// `reap_cancellations` applies once the request's calls drain. Requests
    /// that are not running are ignored.
    pub(super) fn mark_stopped(&mut self, id: RequestId) {
        let Some(request) = self
            .requests
            .get_mut(&id)
            .filter(|request| request.admitted())
        else {
            return;
        };
        if let Graph::Tokens(state) = &mut request.graph {
            state.output.decoder_boundaries.clear();
            request.status = RequestStatus::Finished(Terminal::finish(FinishReason::Stop));
        }
    }
}
