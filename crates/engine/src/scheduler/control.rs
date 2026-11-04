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
    /// `Scheduler::step` on shutdown, each right before closing the executor.
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
            Command::PublishWorld(answer) => {
                let published = self
                    .executor
                    .publish_world()
                    .map_err(|error| format!("{error:#}"));
                let _ = answer.send(published);
            }
            Command::Retire { worlds, answer } => {
                let retired = self
                    .executor
                    .retire(&worlds)
                    .map_err(|error| format!("{error:#}"));
                let _ = answer.send(retired);
            }
            Command::KeepWeights {
                worlds,
                to_host,
                cache_keys,
                answer,
            } => {
                let kept = self
                    .executor
                    .keep_weights(&worlds, to_host, &cache_keys)
                    .map_err(|error| format!("{error:#}"));
                let _ = answer.send(kept);
            }
            Command::Ledgers(answer) => {
                let read = self
                    .executor
                    .ledgers()
                    .map_err(|error| format!("{error:#}"));
                let _ = answer.send(read);
            }
            Command::Throttle { halve, answer } => {
                let throttled = self
                    .executor
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
            Command::DescribePolicyState { ranks, answer } => {
                // No policy, no state a policy is instantiated with: the epoch
                // waits for its first switchover (lead's ruling).
                let state = self.policy.as_ref().map(|_| {
                    views::trial_state_with_ranks(self, self.step, self.recorder.recorded(), ranks)
                });
                let _ = answer.send(state);
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
                        .executor
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
            Command::SwitchoverPolicy { answer } => {
                self.request_stage(Stage::Switchover, Some(answer));
            }
            Command::CommitPolicy => self.request_stage(Stage::Commit, None),
            Command::RevertPolicy { answer } => self.request_stage(Stage::Revert, Some(answer)),
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

    /// Asks for a stage of §7.3, answered when it is applied.
    ///
    /// A stage changes the serving policy, so it is applied where no batch and
    /// no submission is in flight — never in the middle of one (§7.3: "在一个
    /// 批次边界（更新点）上"). A switchover with nothing prepared is refused at
    /// once: the daemon's precopy failed and the update is discarded.
    fn request_stage(
        &mut self,
        stage: Stage,
        answer: Option<crossbeam_channel::Sender<Result<(), String>>>,
    ) {
        if stage == Stage::Switchover {
            self.take_prepared_policy();
            if self.policy_prepared.is_none() {
                if let Some(answer) = answer {
                    let _ = answer.send(Err(
                        "a switchover arrived without a policy prepared by a precopy".to_owned(),
                    ));
                }
                return;
            }
        }
        self.policy_stage = Some((stage, answer));
        self.apply_policy_stage();
    }

    /// Applies the stage asked for, when the scheduler is at a boundary: no
    /// batch and no submission is in flight, so the policy that decided them is
    /// no longer needed by anything (§7.3 steps 3 to 6).
    pub(super) fn apply_policy_stage(&mut self) {
        let ready = self.inflight.pending_batches.is_empty()
            && self.inflight.pending_submissions.is_empty();
        if !ready {
            return;
        }
        let Some((stage, answer)) = self.policy_stage.take() else {
            return;
        };
        let state = views::trial_state(self, self.step, self.recorder.recorded());
        let applied = match self.stage_policy(stage, &state) {
            Ok(report) => {
                if let Some(report) = report {
                    self.policy_report = Some(report);
                }
                Ok(())
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
        if let Some(answer) = answer {
            let _ = answer.send(applied);
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
            return Ok(None);
        };
        if stage == Stage::Switchover {
            let Some(candidate) = self.policy_prepared.take() else {
                return Err("a switchover arrived without a prepared policy".to_owned());
            };
            policy.adopt(candidate);
        }
        policy.request(stage);
        policy.boundary(state).map_err(|error| error.to_string())
    }

    /// Selects commands with available destinations without bypassing an older
    /// command for the same request. Other requests remain independently eligible.
    ///
    /// A command is deferred when `include` rejects it or
    /// `Executor::command_has_capacity` reports a full destination; deferred
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
                || !self.executor.command_has_capacity(&command)
            {
                blocked.insert(request);
                self.inflight.pending_commands.push_back(command);
            } else {
                selected.push(command);
            }
        }
        selected
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
