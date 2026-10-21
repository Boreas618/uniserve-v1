//! Failure reconciliation: `on_executor_error` retires only the work a
//! `WorkerFailure` invalidates. Any other executor error, and a worker failure
//! classified as a control-plane invariant violation, is engine-fatal.

use super::*;

impl Scheduler {
    /// Preserves unrelated requests after a reconciled Worker failure. Unclassified
    /// executor errors and control-plane invariant violations remain fatal.
    ///
    /// A `WorkerFailure` is reconciled by `fail_after_worker_failure`; the
    /// failure class `InvariantViolation` also latches engine-fatal. Losing
    /// every execution-capable worker also stops the engine, so later
    /// admissions cannot wait for capacity that will never return. Any other error latches engine-fatal and fails every in-flight
    /// call.
    pub(in crate::scheduler) fn on_executor_error(&mut self, error: anyhow::Error) {
        if let Some(failure) = error.downcast_ref::<WorkerFailure>() {
            let execution = failure.execution.as_ref();
            let code = execution.and_then(|value| value.code);
            if code == Some(ipc::ErrorCode::InvariantViolation) {
                self.fatal = true;
                tracing::error!(worker = %failure.worker_id, ?code, "control-plane invariant failed: {error}");
            } else if code == Some(ipc::ErrorCode::InputError) {
                tracing::info!(worker = %failure.worker_id, ?code, "Worker rejected request: {error}");
            } else if !failure.refusals.is_empty()
                && failure
                    .requests
                    .iter()
                    .all(|request| failure.refusals.contains_key(request))
            {
                tracing::info!(
                    worker = %failure.worker_id,
                    refusals = ?failure.refusals,
                    "Worker refused requests as invalid"
                );
            } else {
                tracing::warn!(worker = %failure.worker_id, ?code, "Worker failed affected work: {error}");
            }
            self.fail_after_worker_failure(failure);
            // A lost group is replaced before its loss is reported, by its
            // standby, or later by its failure domain's relaunch, and its
            // requests wait for it meanwhile. Keep independent workers
            // usable, but an expert-only group cannot serve requests once
            // every model worker has closed.
            if !self
                .executor
                .info()
                .workers
                .iter()
                .any(|(id, info)| !info.supported_calls.is_empty() && !self.executor.is_closed(id))
            {
                self.fatal = true;
            }
            return;
        }
        self.fatal = true;
        tracing::error!("fatal executor error: {error}");
        self.fail_all_inflight(&error.to_string());
    }
}

impl Scheduler {
    /// Fails every submitted call while preserving requests that can be rescheduled.
    ///
    /// Every request owning an in-flight call fails with `msg`; requests with
    /// no call in flight stay queued or running. The lifecycle commands of
    /// the dropped batches are acknowledged as if they had completed.
    pub(in crate::scheduler) fn fail_all_inflight(&mut self, msg: &str) {
        self.inflight.pending_submissions.clear();
        self.fail_inflight_domain_credits();
        // Submitted batches cannot return after this boundary, so their timing
        // and command ownership must be retired together.
        let (ids, commands) = self.inflight.clear_failed_calls();
        self.acknowledge_commands(&commands);
        for id in ids {
            if self.media_state(id).is_some() {
                self.finish_media(id, DiffusionTerminal::Failed(msg.to_string()));
            } else if self.running.contains_key(&id) {
                self.emit(
                    id,
                    EngineCoreOutput::Error {
                        message: msg.to_string(),
                    },
                );
                self.finish(id, FinishReason::Error);
            }
        }
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
                cache.block_pool.invalidate_source(endpoint);
            }
        }
        // The failed worker's page writes are aborted and its admission pins
        // released.
        if let Some(tiers) = self.tiers.as_mut() {
            tiers.worker_failed(&loss.worker_id);
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
                for (call, placement) in &batch.requests {
                    if (!loss.endpoints.is_empty() && placement.worker == loss.worker_id)
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
        let reclaimable = self.storage.encoder_cache.invalidate_buffers(&buffers);
        self.free_buffers(reclaimable.into_iter().map(|product| product.buffer_id()));

        // Strip affected requests' calls and admissions from queued batches
        // and retire them together with the calls the failure named. A batch
        // left empty is forgotten; the rest stay queued with their command
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
            let Some(inflight) = self.inflight.retire_call(batch_id, request, call_id) else {
                self.fatal = true;
                tracing::error!(
                    batch_id,
                    ?request,
                    ?call_id,
                    "Worker failure named an unknown pending call"
                );
                return;
            };
            self.reclaim_domain_credit(inflight.call.kind, true);
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
        // reason when the worker refused it; a token request emits an error
        // unless an error finish is already pending, and defers an error
        // finish until its in-flight calls drain.
        let engine_id = self.engine_id;
        for request in requests {
            let id = request.request_id;
            let refusal = loss.refusals.get(&request);
            let error_emitted = self
                .inflight
                .pending_finishes
                .get(&id)
                .is_some_and(|finish| finish.reason == FinishReason::Error);
            if let Some(state) = self.media_state_mut(id) {
                if state.admission.request_key == request {
                    state.terminal_intent = match refusal {
                        Some(reason) => TerminalIntent::Rejected(reason.clone()),
                        None => TerminalIntent::Failure(loss.message.clone()),
                    };
                }
            } else if let Some(state) = self.running.get_mut(&id) {
                if request != RequestKey::new(engine_id, id, state.request_epoch) {
                    continue;
                }
                state.output.decoder_boundaries.clear();
                if !error_emitted {
                    self.emit(
                        id,
                        EngineCoreOutput::Error {
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
                .is_some_and(|state| state.admission.request_key == request)
            {
                if !self.inflight.has_pending_calls(id) {
                    let terminal = match refusal {
                        Some(reason) => DiffusionTerminal::Rejected(reason.clone()),
                        None => DiffusionTerminal::Failed(loss.message.clone()),
                    };
                    self.finish_media(id, terminal);
                }
            } else if self.running.contains_key(&id) {
                self.finish_after_inflight(id, FinishReason::Error, None);
            }
        }
    }

    /// Fails every queued and running request and releases scheduler-owned resources.
    ///
    /// Covers queued media submissions and every running media and token
    /// request. Queued token requests in `waiting` are not touched: every
    /// caller latches engine-fatal first, and `run` then stops and calls
    /// `abort_all_requests`. The dropped batches' lifecycle commands are
    /// discarded without acknowledgement.
    pub(in crate::scheduler) fn fail_all_running(&mut self, message: &str) {
        self.inflight.pending_submissions.clear();
        self.fail_inflight_domain_credits();
        let _ = self.inflight.clear_failed_calls();
        while let Some(submission) = self.waiting_media.pop_front() {
            let _ = submission.event_tx.send(EngineCoreOutput::Error {
                message: message.to_string(),
            });
        }
        let media = self.media_ids();
        for id in media {
            self.finish_media(id, DiffusionTerminal::Failed(message.to_string()));
        }
        let ids = self.running.keys().copied().collect::<Vec<_>>();
        for id in ids {
            self.emit(
                id,
                EngineCoreOutput::Error {
                    message: message.to_string(),
                },
            );
            self.finish(id, FinishReason::Error);
        }
    }
}
