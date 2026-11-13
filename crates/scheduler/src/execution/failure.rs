//! Failure reconciliation: `on_worker_error` retires only the work a
//! `WorkerFailure` invalidates. Any other worker error, and a worker failure
//! classified as a control-plane invariant violation, is engine-fatal.

use super::*;

impl Scheduler {
    /// Preserves unrelated requests after a reconciled Worker failure. Unclassified
    /// worker errors and control-plane invariant violations remain fatal.
    ///
    /// A `WorkerFailure` is reconciled by `fail_after_worker_failure`; the
    /// failure class `InvariantViolation` also latches engine-fatal. Losing
    /// every execution-capable worker also stops the engine, so later
    /// admissions cannot wait for capacity that will never return. Any other error latches engine-fatal and fails every in-flight
    /// call.
    pub(crate) fn on_worker_error(&mut self, error: anyhow::Error) {
        if let Some(failure) = error.downcast_ref::<WorkerFailure>() {
            let execution = failure.execution.as_ref();
            let code = execution.and_then(|value| value.code);
            // Requests refused earlier keep their place while a call of
            // theirs is in flight.
            let pending = &self.inflight.pending_calls;
            self.inflight
                .refused
                .retain(|request| pending.contains_key(&request.request_id));
            // A failure is a refusal when every request it names was refused,
            // by this report or by an earlier one whose consequence it is.
            let refused = !failure.requests.is_empty()
                && failure.requests.iter().all(|request| {
                    failure.refusals.contains_key(request)
                        || self.inflight.refused.contains(request)
                });
            // The batches the failure struck, before their retirement below
            // forgets them: the batch it names, or else every batch in flight
            // that holds a call of the requests it names.
            let mut struck: Vec<u64> = match execution.and_then(|value| value.batch_id) {
                Some(batch) => vec![batch],
                None => self
                    .inflight
                    .pending_batches
                    .iter()
                    .filter(|(_, batch)| {
                        batch
                            .calls
                            .iter()
                            .any(|(request, _)| failure.requests.contains(request))
                    })
                    .map(|(id, _)| *id)
                    .collect(),
            };
            struck.sort_unstable();
            if code == Some(ipc::ErrorCode::InvariantViolation) {
                self.fail(format!(
                    "a worker's control-plane invariant failed: {error:#}"
                ));
                tracing::error!(worker = %failure.worker_id, ?code, "control-plane invariant failed: {error}");
            } else if code == Some(ipc::ErrorCode::InputError) {
                tracing::info!(worker = %failure.worker_id, ?code, "Worker rejected request: {error}");
            } else if refused {
                tracing::info!(
                    worker = %failure.worker_id,
                    refusals = ?failure.refusals,
                    "Worker refused requests as invalid"
                );
            } else {
                tracing::warn!(worker = %failure.worker_id, ?code, "Worker failed affected work: {error}");
            }
            self.fail_after_worker_failure(failure);
            if refused {
                let pending = &self.inflight.pending_calls;
                let in_flight = failure
                    .refusals
                    .keys()
                    .filter(|request| pending.contains_key(&request.request_id))
                    .copied()
                    .collect::<Vec<_>>();
                self.inflight.refused.extend(in_flight);
            }
            // An epoch on trial is not repaired: a worker fault, rather than a
            // request its workers rejected, fails it as a whole, and the
            // daemon rolls back to the previous epoch.
            // The failure names the batch it struck and its requests, which
            // the rollback keeps with its reason for diagnosis.
            if self.trial && !refused && code != Some(ipc::ErrorCode::InputError) {
                let requests: Vec<u64> = failure
                    .requests
                    .iter()
                    .map(|request| request.request_id.0)
                    .collect();
                self.fail(match (code, execution.and_then(|value| value.batch_id)) {
                    (Some(ipc::ErrorCode::NumericalFault), Some(batch)) => {
                        format!(
                            "batch {batch} computed non-finite logits for requests {requests:?}"
                        )
                    }
                    _ => format!(
                        "a worker failed during the epoch's trial in batches {struck:?} of \
                         requests {requests:?}: {error:#}"
                    ),
                });
                return;
            }
            // A lost group may be replaced, and its requests wait for it
            // meanwhile. Keep independent groups usable, but an expert-only
            // group cannot serve requests once every model group has closed.
            if !self
                .worker
                .info()
                .workers
                .iter()
                .any(|(_, info)| !info.supported_calls.is_empty())
            {
                self.fail(format!("every model worker group closed: {error:#}"));
            }
            return;
        }
        self.fail(format!("the worker failed: {error:#}"));
        tracing::error!("fatal worker error: {error}");
        self.fail_all_inflight(&error.to_string());
    }
}

impl Scheduler {
    /// Fails every submitted call while preserving requests that can be rescheduled.
    ///
    /// Every request owning an in-flight call fails with `msg`, but a token
    /// request that resumes from its generated tokens leaves as migratable
    /// (`MigrationCause::Loss`) for the daemon to submit again, and one whose
    /// migration began leaves for that migration's cause; requests with no
    /// call in flight stay waiting or running. The lifecycle commands of
    /// the dropped batches are acknowledged as if they had completed.
    pub(crate) fn fail_all_inflight(&mut self, msg: &str) {
        self.inflight.pending_submissions.clear();
        self.fail_inflight_domain_credits();
        // Submitted batches cannot return after this boundary, so their timing
        // and command ownership must be retired together.
        let (ids, commands) = self.inflight.clear_failed_calls();
        self.acknowledge_commands(&commands);
        for id in &ids {
            self.requests.mark(*id);
        }
        for id in ids {
            // A request whose migration began ends as it began.
            let decided = self.requests.status(&id).and_then(RequestStatus::terminal);
            let migrating = decided
                .filter(|_| self.requests.tokens(&id).is_some())
                .and_then(Terminal::migration);
            if self.media_state(id).is_some() {
                // A video request leaves as migratable, for the cause its
                // preemption states or as lost, its trajectory not stored.
                let cause = match decided {
                    Some(Terminal::Migrated { cause, .. }) => Some(*cause),
                    None if crate::BLOCK_SAMPLING.is_none() => Some(MigrationCause::Loss),
                    _ => None,
                };
                let terminal = match cause {
                    Some(cause) => Terminal::migrate(cause),
                    None => Terminal::Failed(msg.to_string()),
                };
                self.retire(id, terminal);
            } else if let Some(cause) = migrating {
                self.retire(id, Terminal::migrate(cause));
            } else if self.migrates_on_loss(id) {
                self.retire(id, Terminal::migrate(MigrationCause::Loss));
            } else if self.requests.tokens(&id).is_some() {
                self.emit(
                    id,
                    Event::Error {
                        message: msg.to_string(),
                    },
                );
                self.finish(id, FinishReason::Error);
            }
        }
    }

    /// Hands every request that resumes from its generated tokens back to
    /// the daemon after a fatal stop of an epoch on trial: the calls in
    /// flight are dropped (`fail_all_inflight`), and each such request, queued
    /// or running, ends as migratable now (`MigrationCause::Loss`), after
    /// every token it generated, or for the cause a migration it already
    /// began states. The daemon submits each again to the previous epoch,
    /// which generates the dropped tokens again. Every video request leaves
    /// the same way, naming the trajectory it named before. The others are
    /// left to `abort_all_requests`.
    pub(crate) fn hand_back_requests(&mut self) {
        if crate::BLOCK_SAMPLING.is_some() {
            return;
        }
        let failure = self
            .failure
            .get()
            .cloned()
            .unwrap_or_else(|| "the epoch on trial failed".to_owned());
        self.fail_all_inflight(&failure);

        // Every waiting request leaves as migratable, a video request too:
        // the epoch that held their trajectories failed, and the previous
        // epoch computes them again from what they name.
        let waiting: Vec<RequestId> = self
            .waiting
            .iter()
            .chain(&self.held_for_index)
            .copied()
            .collect();
        for id in waiting {
            self.end_waiting(
                id,
                scheduling::guest::event::End::Migratable,
                Event::Migratable {
                    cause: MigrationCause::Loss,
                    denoised_steps: 0,
                    stored_layout: Vec::new(),
                },
            );
        }
        // So does every running video request, whatever it reached.
        for id in self.media_ids() {
            let cause = match self.requests.status(&id).map(RequestStatus::terminal) {
                Some(Some(Terminal::Migrated { cause, .. })) => *cause,
                Some(None) => MigrationCause::Loss,
                _ => continue,
            };
            self.retire(id, Terminal::migrate(cause));
        }

        let running: Vec<(RequestId, MigrationCause)> = self
            .requests
            .running_tokens()
            .filter_map(|(id, status, _)| match status.terminal() {
                Some(Terminal::Migrated { cause, .. }) => Some((id, *cause)),
                _ if self.migrates_on_loss(id) => Some((id, MigrationCause::Loss)),
                _ => None,
            })
            .collect();
        for (id, cause) in running {
            self.retire(id, Terminal::migrate(cause));
        }
    }

    /// Whether running request `id` leaves as migratable when the workers
    /// that hold it are lost: a token request that resumes from its
    /// generated tokens, not already finishing, on a deployment that does
    /// not generate in blocks.
    fn migrates_on_loss(&self, id: RequestId) -> bool {
        crate::BLOCK_SAMPLING.is_none()
            && !self.inflight.pending_finishes.contains_key(&id)
            && self.requests.tokens(&id).is_some()
            && !self.requests.finished(&id)
    }

    /// Reconciles failed work without resetting allocations owned by live consumers.
    ///
    /// Only the request epochs `loss` names fail, together with requests whose
    /// queued work targets an invalidated worker incarnation or reads an
    /// invalidated buffer; every other request continues. A media request
    /// `loss` names as refused is rejected with the reason the worker stated
    /// rather than failed. A retired call that has no pending record latches
    /// engine-fatal and stops reconciliation.
    fn fail_after_worker_failure(&mut self, loss: &WorkerFailure) {
        // Blocks published from an invalidated worker incarnation leave
        // prefix-cache lookup; pages still referenced keep their owners.
        if let Some(cache) = &self.storage.cache {
            for endpoint in &loss.endpoints {
                cache.pool().invalidate_source(endpoint);
            }
        }
        // The failed worker's page offloads are aborted and its admission pins
        // released.
        if let Some(kv_offload) = self.kv_offload.as_mut() {
            kv_offload.worker_failed(&loss.worker_id);
        }
        // So are the trajectory offloads its batches carried, and the
        // encoder outputs its calls read or wrote.
        if let Some(store) = self.trajectories.as_mut() {
            store.worker_failed(&loss.worker_id);
        }
        if let Some(store) = self.encoder_outputs.as_mut() {
            store.worker_failed(&loss.worker_id);
        }

        // Close the affected set over work not yet submitted: a queued call
        // is affected when it reads an invalidated buffer, or is placed on
        // the failed worker and the failure invalidated incarnations, and the
        // outputs of an affected request's queued calls are invalidated in
        // turn. Repeat until neither set grows.
        let mut requests = loss.requests.iter().copied().collect::<HashSet<_>>();
        let mut buffers = loss.buffers.iter().cloned().collect::<HashSet<_>>();
        loop {
            let before = (requests.len(), buffers.len());
            for batch in &self.inflight.pending_submissions {
                for (call, dispatch) in &batch.requests {
                    if (!loss.endpoints.is_empty() && dispatch.worker == loss.worker_id)
                        || call.input_buffers().any(|buffer| buffers.contains(&buffer))
                    {
                        requests.insert(call.request_key);
                    }
                    if requests.contains(&call.request_key) {
                        buffers.extend(call.output_buffers());
                    }
                }
            }
            if before == (requests.len(), buffers.len()) {
                break;
            }
        }

        // Encoder-cache entries backed by invalidated buffers leave lookup;
        // those no consumer still pins are freed.
        let releasable = self.storage.encoder_cache.invalidate_buffers(&buffers);
        self.free_buffers(releasable.into_iter().map(|product| product.buffer_id()));

        // Strip affected requests' calls and admissions from queued batches
        // and retire them together with the calls the failure named. A batch
        // left empty is forgotten; the rest stay waiting with their command
        // receipts kept in step with the commands they still carry.
        let mut retired = loss.retired.clone();
        let mut pending = VecDeque::new();
        for mut batch in std::mem::take(&mut self.inflight.pending_submissions) {
            retired.extend(
                batch
                    .retire_requests(&requests)
                    .into_iter()
                    .map(|(call, _)| (batch.id, call.request_key, call.call_id)),
            );
            if let Some(pending) = self.inflight.pending_batches.get_mut(&batch.id) {
                pending.commands = batch
                    .commands
                    .iter()
                    .filter(|command| !matches!(command, BatchCommand::Start { .. }))
                    .cloned()
                    .collect();
            }
            pending.push_back(batch);
        }
        for (batch_id, request, call_id) in retired {
            self.requests.mark(request.request_id);
            let Some(inflight) = self.inflight.retire_call(batch_id, request, call_id) else {
                self.latch_fatal();
                tracing::error!(
                    batch_id,
                    ?request,
                    ?call_id,
                    "Worker failure named an unknown pending call"
                );
                return;
            };
            self.return_domain_credit(inflight.call.kind, true);
        }
        for batch in pending {
            if batch.requests.is_empty() && batch.commands.is_empty() {
                self.inflight.pending_batches.remove(&batch.id);
            } else {
                self.inflight.pending_submissions.push_back(batch);
            }
        }

        // An affected request's queued `Finish` and `Free` commands still
        // retire its worker state; only its admissions are dropped.
        self.inflight.pending_commands.retain_mut(|command| {
            if !requests.contains(&command.request_key()) {
                return true;
            }
            !matches!(command, BatchCommand::Start { .. })
        });

        // Fail each affected request that is still on the named epoch: a media
        // request records a failure intent, or a rejection with the worker's
        // reason when the worker refused it. A token request that resumes from
        // its generated tokens leaves as migratable once its in-flight calls
        // drain, after every token it generated, unless the worker refused it
        // or it already finishes; the daemon submits it again, and it waits
        // for the group's replacement. Any other token request emits an error
        // unless an error finish is already pending, and defers an error
        // finish until its in-flight calls drain.
        let engine_id = self.engine_id;
        let migrates = crate::BLOCK_SAMPLING.is_none()
            && loss
                .execution
                .as_ref()
                .and_then(|execution| execution.code)
                .is_none_or(|code| code != ipc::ErrorCode::InputError);
        for request in requests {
            let id = request.request_id;
            let refusal = loss.refusals.get(&request);
            let mut migrating = false;
            let error_emitted = self
                .inflight
                .pending_finishes
                .get(&id)
                .is_some_and(|finish| finish.reason == FinishReason::Error);
            let Some(Request { status, graph, .. }) = self
                .requests
                .get_mut(&id)
                .filter(|request| request.admitted())
            else {
                continue;
            };
            if let Graph::Video(state) = graph {
                // A video request whose workers were lost leaves as
                // migratable once its calls drain, its trajectory lost with
                // them, unless the worker refused it or it already fails;
                // the daemon submits it again.
                if state.request_key == request {
                    *status = RequestStatus::Finished(match refusal {
                        Some(reason) => Terminal::Rejected(reason.clone()),
                        None if migrates
                            && !status.terminal().is_some_and(Terminal::is_failure) =>
                        {
                            migrating = true;
                            Terminal::migrate(MigrationCause::Loss)
                        }
                        None => Terminal::Failed(loss.message.clone()),
                    });
                }
            } else if let Graph::Tokens(state) = graph {
                if request != RequestKey::new(engine_id, id, state.request_epoch) {
                    continue;
                }
                state.output.decoder_boundaries.clear();
                if migrates
                    && refusal.is_none()
                    && !status.is_finished()
                    && !self.inflight.pending_finishes.contains_key(&id)
                {
                    *status = RequestStatus::Finished(Terminal::migrate(MigrationCause::Loss));
                    migrating = true;
                } else {
                    if !error_emitted {
                        self.emit(
                            id,
                            Event::Error {
                                message: loss.message.clone(),
                            },
                        );
                    }
                    let terminal_published = self
                        .inflight
                        .pending_finishes
                        .get(&id)
                        .is_some_and(|finish| finish.terminal_published);
                    self.inflight.pending_finishes.insert(
                        id,
                        PendingFinish {
                            reason: FinishReason::Error,
                            stop_reason: None,
                            terminal_published,
                        },
                    );
                }
            }
            // A later completion may already be queued behind a call the
            // failure retired. With that call gone it reaches the front of the
            // queue, so it drains here: its call is removed and its products
            // are freed without applying the result.
            while let Some(call_id) = self
                .inflight
                .pending_calls
                .get(&id)
                .and_then(|queue| queue.front())
                .filter(|inflight| inflight.call.request_key == request)
                .map(|inflight| inflight.call.call_id)
            {
                let Some(completion) = self
                    .inflight
                    .pending_completions
                    .get_mut(&id)
                    .and_then(|values| values.remove(&call_id))
                else {
                    break;
                };
                let Some((call, _)) = self.pop_inflight(request, call_id) else {
                    unreachable!("ready completion owns its call");
                };
                drop(completion);
                self.free_buffers(call.output_buffers());
            }
            if self
                .inflight
                .pending_completions
                .get(&id)
                .is_some_and(BTreeMap::is_empty)
            {
                self.inflight.pending_completions.remove(&id);
            }
            if self
                .media_state(id)
                .is_some_and(|state| state.request_key == request)
            {
                if !self.inflight.has_pending_calls(id) {
                    let terminal = match refusal {
                        Some(reason) => Terminal::Rejected(reason.clone()),
                        None if migrating => Terminal::migrate(MigrationCause::Loss),
                        None => Terminal::Failed(loss.message.clone()),
                    };
                    self.retire(id, terminal);
                }
            } else if self.requests.tokens(&id).is_some() && !migrating {
                self.finish_after_inflight(id, FinishReason::Error, None);
            }
        }
    }

    /// Fails every queued and running request and releases scheduler-owned resources.
    ///
    /// Covers queued media submissions and every running media and token
    /// request; a token request that resumes from its generated tokens
    /// leaves as migratable (`MigrationCause::Loss`) instead. Waiting token
    /// requests in `waiting` are not touched: every caller latches
    /// engine-fatal first, and `run` then stops and calls
    /// `abort_all_requests`. The dropped batches' lifecycle commands are
    /// discarded without acknowledgement.
    pub(crate) fn fail_all_running(&mut self, message: &str) {
        self.inflight.pending_submissions.clear();
        self.fail_inflight_domain_credits();
        let _ = self.inflight.clear_failed_calls();
        // A waiting video request fails with the engine; a waiting token
        // request is left to `abort_all_requests`.
        let waiting: Vec<RequestId> = self
            .waiting
            .iter()
            .copied()
            .filter(|id| {
                self.requests
                    .get(id)
                    .is_some_and(|request| request.video().is_some())
            })
            .collect();
        for id in waiting {
            self.end_waiting(
                id,
                scheduling::guest::event::End::Failed,
                Event::Error {
                    message: message.to_string(),
                },
            );
        }
        let media = self.media_ids();
        for id in media {
            self.retire(id, Terminal::Failed(message.to_string()));
        }
        let ids = self.running.clone();
        for id in ids {
            if self.migrates_on_loss(id) {
                self.retire(id, Terminal::migrate(MigrationCause::Loss));
                continue;
            }
            self.emit(
                id,
                Event::Error {
                    message: message.to_string(),
                },
            );
            self.finish(id, FinishReason::Error);
        }
    }
}
