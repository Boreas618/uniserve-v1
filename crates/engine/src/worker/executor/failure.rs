//! Reconciliation of a [`WorkerFailure`]: a lost group's resident requests,
//! buffers and exports, or the named calls, requests or batch of an intact one,
//! and every call, request and buffer that depends on them.

use std::collections::{HashSet, VecDeque};

use anyhow::Context;

use super::WorkerExecutor;
use crate::executor::WorkerResult;
use crate::executor::{CommandOutcome, WorkerFailure};

impl WorkerExecutor {
    /// Reconciles abandoned work and its dependents while retaining live physical owners.
    ///
    /// A failure is one of two kinds. With endpoints it is a loss: the
    /// `WorkerGroup` terminated its ranks and holds none of its resident
    /// requests, buffers, or exports. Without endpoints the group is
    /// intact and only the named calls, requests or batch failed; the calls
    /// of a named request still queued are retired here.
    ///
    /// Returns the failure extended to everything it invalidates: `requests`
    /// and `buffers` include the dependents found here, and `retired` lists
    /// every pending call that will now never complete, each removed from its
    /// pending batch. Batches left with no pending worker publish an empty
    /// terminal result. An error that is not a `WorkerFailure` is returned
    /// unchanged.
    pub(super) fn reconcile_worker_failure(
        &mut self,
        error: anyhow::Error,
    ) -> anyhow::Result<WorkerFailure> {
        let mut loss = error.downcast::<WorkerFailure>()?;
        let index = *self
            .routing
            .get(&loss.worker_id)
            .context("failed WorkerGroup is not bound")?;
        let lost = !loss.endpoints.is_empty();
        if lost {
            self.refresh_worker(index);
        }

        // Without a loss, an execution error names the batch that failed on
        // this worker; only a batch already submitted here can have failed.
        let failed_run = (!lost)
            .then(|| loss.execution.as_ref().and_then(|error| error.batch_id))
            .flatten()
            .filter(|batch_id| {
                self.pending.get(batch_id).is_some_and(|pending_batch| {
                    pending_batch
                        .workers
                        .get(&index)
                        .is_some_and(|worker| worker.submitted)
                })
            });
        let mut retired = loss.retired.iter().copied().collect::<HashSet<_>>();
        let failed_calls = retired
            .iter()
            .map(|(_, request, call)| (*request, *call))
            .collect::<HashSet<_>>();
        let mut requests = loss.requests.iter().copied().collect::<HashSet<_>>();
        let mut buffers = loss.buffers.iter().cloned().collect::<HashSet<_>>();

        // A lost incarnation's locations are gone. An export that no longer
        // covers its whole value is invalid; one that other ranks or workers
        // still cover in full stays usable.
        if lost {
            for (buffer, payload) in &mut self.transfer_products {
                let handle = &mut payload.value;
                let tensors = handle.tensors_mut();
                for tensor in tensors.iter_mut() {
                    tensor
                        .locations
                        .retain(|location| !loss.endpoints.contains(&location.source));
                }
                if tensors.iter().any(|tensor| !tensor.has_complete_coverage()) {
                    buffers.insert(*buffer);
                }
            }
        }
        if lost {
            for (buffer, export) in &mut self.cache_transfers {
                for tensor in export.tensors_mut() {
                    tensor
                        .locations
                        .retain(|location| !loss.endpoints.contains(&location.source));
                }
                if export
                    .tensors()
                    .any(|tensor| !tensor.has_complete_coverage())
                {
                    buffers.insert(*buffer);
                }
            }
        }

        // Buffers produced on the lost worker that were never published are
        // gone with it, and a failed call's outputs will never be produced.
        for (buffer, route) in &self.buffer_routes {
            if (lost
                && route.worker_index == index
                && !self.transfer_products.contains_key(buffer)
                && !self.cache_transfers.contains_key(buffer))
                || failed_calls.contains(&(buffer.owner, buffer.producer_call_id))
            {
                buffers.insert(*buffer);
            }
        }
        // On a loss, every request with a call still pending on the worker is
        // affected.
        if lost {
            for pending in self.pending.values() {
                if let Some(worker) = pending.workers.get(&index) {
                    requests.extend(worker.calls.keys().map(|(request, _)| *request));
                }
            }
        }

        // Propagate to a fixed point. A call reading an invalid buffer taints
        // its request. Abandoning an unstarted producer also invalidates its
        // future buffers. Already running work on another instance keeps its
        // real completion path, so an in-flight call only taints its request.
        loop {
            let before = (requests.len(), buffers.len());
            for (_, worker) in &self.workers {
                for call in worker.inflight_calls() {
                    if call.input_buffers().any(|buffer| buffers.contains(&buffer)) {
                        requests.insert(call.request_key);
                    }
                }
            }
            for submission in self.worker_submissions.iter().flatten() {
                for (call, _) in &submission.batch.requests {
                    if call.input_buffers().any(|buffer| buffers.contains(&buffer)) {
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

        // Remove the tainted requests' calls, and the dependencies no remaining
        // call reads, from every queued submission. A lost worker's queued
        // commands are dropped too, since the state they act on is gone. A
        // submission left empty is discarded.
        for (worker_index, queue) in self.worker_submissions.iter_mut().enumerate() {
            let mut active = VecDeque::new();
            for mut submission in queue.drain(..) {
                let batch_id = submission.batch.id;
                for (call, _) in submission.batch.retire_requests(&requests) {
                    retired.insert((batch_id, call.request_key, call.call_id));
                }
                let inputs = submission
                    .batch
                    .requests
                    .iter()
                    .flat_map(|(call, _)| call.input_buffers())
                    .collect::<HashSet<_>>();
                submission
                    .dependencies
                    .retain(|product| inputs.contains(product));
                if lost && worker_index == index {
                    submission.batch.commands.clear();
                }
                if !submission.batch.requests.is_empty() || !submission.batch.commands.is_empty() {
                    active.push_back(submission);
                }
            }
            *queue = active;
        }

        // Retire this worker's outstanding calls in every batch on a loss, or in
        // the failed batch only. The group sends no report for such a batch, so
        // the entry stops awaiting one. Commands of tainted requests are
        // marked `Retired`, and in the failed batch the reported requests'
        // commands are marked `Failed`.
        for (batch_id, pending_batch) in &mut self.pending {
            if let Some(worker) = pending_batch.workers.get_mut(&index)
                && (lost || (worker.submitted && failed_run == Some(*batch_id)))
            {
                retired.extend(
                    worker
                        .calls
                        .keys()
                        .map(|identity| (*batch_id, identity.0, identity.1)),
                );
                worker.submitted = false;
            }
            pending_batch.set_command_outcome(&requests, CommandOutcome::Retired);
        }
        if let Some(batch_id) = failed_run {
            let failed_requests = loss.requests.iter().copied().collect();
            if let Some(pending) = self.pending.get_mut(&batch_id) {
                pending.set_command_outcome(&failed_requests, CommandOutcome::Failed);
            }
        }

        // Remove every retired call from its pending batch. Each must still be
        // pending on some worker, or the bookkeeping has diverged.
        loss.retired.clear();
        for (batch_id, request, call) in retired {
            let pending_batch = self
                .pending
                .get_mut(&batch_id)
                .context("retired call has no pending batch")?;
            let removed = pending_batch
                .workers
                .values_mut()
                .any(|worker| worker.calls.remove(&(request, call)).is_some());
            anyhow::ensure!(removed, "abandoned call is not pending");
            loss.retired.push((batch_id, request, call));
        }

        // A lost worker holds no admissions or allocations: a later call on it
        // sends its request's `Start` again, and no `Free` targets it.
        if lost {
            self.admitted_workers.retain(|(worker, _)| *worker != index);
            self.buffer_allocations
                .retain(|(worker, _), _| *worker != index);
            for workers in self.buffer_workers.values_mut() {
                workers.remove(&index);
            }
        }

        // Drop workers with nothing left to report, then complete the batches
        // that no worker still owes a result.
        for (batch_id, pending) in &mut self.pending {
            pending.workers.retain(|worker_index, worker| {
                worker.submitted
                    || !worker.calls.is_empty()
                    || self.worker_submissions[*worker_index]
                        .iter()
                        .any(|submission| submission.batch.id == *batch_id)
            });
        }
        let batches = self.pending.keys().copied().collect::<Vec<_>>();
        for batch_id in batches {
            if self
                .pending
                .get(&batch_id)
                .is_some_and(|pending| pending.workers.is_empty())
            {
                self.publish_result(WorkerResult {
                    batch_id,
                    done: true,
                    results: Vec::new(),
                    products: Vec::new(),
                    worker_exec_us: None,
                    forward_stats: None,
                })?;
            }
        }
        loss.requests = requests.into_iter().collect();
        loss.buffers = buffers.into_iter().collect();
        Ok(loss)
    }
}
