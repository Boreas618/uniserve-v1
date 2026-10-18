//! Routing of worker results: call completions published as they arrive,
//! command receipts once every worker finished the batch, the executor's
//! retirement bookkeeping for each applied command, and the routing a finished
//! request leaves behind.

use std::collections::HashSet;

use anyhow::Context;
use uniserve_ipc::{
    BatchCommand, BufferId, CallKind, CallStatus, ForwardMode, RequestKey, TransferMode,
};

use super::WorkerExecutor;
use crate::executor::WorkerResult;
use crate::executor::{CommandOutcome, WorkerFailure, logical_result};

impl WorkerExecutor {
    /// Routes a worker result into its worker aggregate and publishes transferable products.
    ///
    /// Validates the report against the worker's pending calls, stores the
    /// product and cache exports it returns, and publishes its completions.
    /// When some calls failed, the report, failed completions included, is
    /// still published and the method then returns a `WorkerFailure` naming
    /// the failed calls' requests and the buffers those calls would have
    /// produced.
    pub(super) fn route_result(
        &mut self,
        worker_index: usize,
        report: WorkerResult,
    ) -> anyhow::Result<()> {
        let mut report = report;
        let failed_calls = report
            .results
            .iter()
            .filter(|completion| completion.output.status == uniserve_ipc::CallStatus::Error)
            .map(|completion| (completion.output.request_key, completion.output.call_id))
            .collect::<HashSet<_>>();
        if !failed_calls.is_empty() {
            // The worker names the code; the request's caller sees only that
            // its dependent work could not complete, so the code is recorded
            // here, where the failing call and worker are known.
            let failed = report
                .results
                .iter()
                .filter(|completion| completion.output.status == uniserve_ipc::CallStatus::Error)
                .map(|completion| {
                    (
                        completion.output.request_key,
                        completion.output.call_id,
                        completion.output.error_code,
                    )
                })
                .collect::<Vec<_>>();
            tracing::warn!(
                worker = %self.workers[worker_index].0,
                batch_id = report.batch_id,
                ?failed,
                "worker returned failed calls"
            );
        }
        let failed_buffers = self
            .buffer_routes
            .keys()
            .filter(|product| failed_calls.contains(&(product.owner, product.producer_call_id)))
            .cloned()
            .collect();

        // Every completion must answer a distinct call still pending on this
        // worker, and a terminal report must answer all of them.
        let batch_id = report.batch_id;
        {
            let pending_batch = self
                .pending
                .get_mut(&batch_id)
                .ok_or_else(|| anyhow::anyhow!("worker returned unknown batch {batch_id}"))?;
            let worker = pending_batch
                .workers
                .get(&worker_index)
                .context("worker returned an unexpected batch")?;
            anyhow::ensure!(
                worker.submitted,
                "worker returned a batch before submission"
            );
            let mut seen = HashSet::new();
            for completion in &report.results {
                let identity = (completion.output.request_key, completion.output.call_id);
                let kind = worker
                    .calls
                    .get(&identity)
                    .context("worker returned an unknown call")?;
                anyhow::ensure!(seen.insert(identity), "worker returned a duplicate call");
                // A completion's payload must fit the kind of the call it
                // answers: only a successful cache-export call exports cache
                // state, and only a successful token-denoising call reports
                // candidate log-probabilities.
                let output = &completion.output;
                let successful = output.status == CallStatus::Ok;
                anyhow::ensure!(
                    output.cache_output.is_none()
                        || (successful && *kind == CallKind::Transfer(TransferMode::CacheExport)),
                    "worker returned a cache export for a call that does not export one"
                );
                anyhow::ensure!(
                    output.candidate_logprobs.is_empty()
                        || (successful && *kind == CallKind::Forward(ForwardMode::TokenDenoising)),
                    "worker returned candidate log-probabilities for a call that reads none"
                );
            }
            anyhow::ensure!(
                !report.done || seen.len() == worker.calls.len(),
                "worker {worker_index} ended batch {batch_id} before every call completed"
            );
        }

        // Store returned exports. Each must come from its buffer's
        // producing worker; a later report for the same buffer adds locations.
        for product in report.products.drain(..) {
            let route = self
                .buffer_routes
                .get(&product.product.buffer_id())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "worker {worker_index} returned an unplanned cross-worker product {:?}",
                        product.product
                    )
                })?;
            anyhow::ensure!(
                route.worker_index == worker_index,
                "worker {worker_index} returned a cross-worker product owned by worker {}",
                route.worker_index
            );
            if let Some(existing) = self.transfer_products.get_mut(&product.product.buffer_id()) {
                existing.merge_locations(&product)?;
            } else {
                self.transfer_products
                    .insert(product.product.buffer_id(), product.clone());
            }
        }
        for export in report
            .results
            .iter()
            .filter_map(|completion| completion.output.cache_output.as_ref())
        {
            let route = self
                .buffer_routes
                .get(&export.source)
                .context("worker returned an unplanned cache export")?;
            anyhow::ensure!(
                route.worker_index == worker_index,
                "cache export was returned by another owner"
            );
            if let Some(stored) = self.cache_transfers.get_mut(&export.source) {
                stored.merge_locations(export)?;
            } else {
                self.cache_transfers.insert(export.source, export.clone());
            }
        }

        // Keep outstanding identities available to recovery until all returned
        // exports have passed their owner and layout checks.
        let pending = self
            .pending
            .get_mut(&batch_id)
            .context("result has no pending batch")?;
        if report.done {
            pending.workers.remove(&worker_index);
        } else {
            // Computation can finish before command-only ranks acknowledge.
            // Keep that worker's ownership until its terminal report arrives.
            let worker = pending
                .workers
                .get_mut(&worker_index)
                .context("result has no pending worker")?;
            for completion in &report.results {
                worker
                    .calls
                    .remove(&(completion.output.request_key, completion.output.call_id));
            }
        }
        self.publish_result(report)?;
        if !failed_calls.is_empty() {
            // Keep the actual failed completion and all accepted independent work.
            // Future consumers cannot become ready when their producer has failed.
            return Err(WorkerFailure {
                worker_id: self.workers[worker_index].0.clone(),
                endpoints: Vec::new(),
                requests: failed_calls.iter().map(|(request, _)| *request).collect(),
                retired: Vec::new(),
                buffers: failed_buffers,
                execution: None,
                message: "call failed before its dependent work could complete".into(),
            }
            .into());
        }
        self.dispatch_ready()
    }

    /// Publishes ready calls immediately; only lifecycle receipts wait for all workers.
    ///
    /// The batch is terminal once no worker remains in its pending record;
    /// callers remove workers from it as they finish. A terminal result
    /// carries a receipt for every lifecycle command other than `Start`, with
    /// its recorded outcome or `Applied`, and applies the executor's own
    /// retirement bookkeeping for each command that did not fail. A result is
    /// queued for `poll` when it carries completions or is terminal.
    pub(super) fn publish_result(&mut self, report: WorkerResult) -> anyhow::Result<()> {
        let batch_id = report.batch_id;
        let std::collections::btree_map::Entry::Occupied(entry) = self.pending.entry(batch_id)
        else {
            anyhow::bail!("result has no pending batch");
        };
        let pending = entry.get();
        let done = pending.workers.is_empty();
        let mut result = logical_result(report, done, &pending.commands);
        for receipt in &mut result.command_results {
            receipt.outcome = pending
                .command_outcomes
                .get(&receipt.command_index)
                .copied()
                .unwrap_or(CommandOutcome::Applied);
        }
        if done {
            let pending = entry.remove();
            for (index, command) in pending.commands.into_iter().enumerate() {
                // A failed command keeps physical ownership, so what it would
                // release stays indexed until a later release succeeds.
                if pending.command_outcomes.get(&(index as u32)) == Some(&CommandOutcome::Failed) {
                    continue;
                }
                match command {
                    BatchCommand::Finish {
                        request_key,
                        retained_buffers,
                    } => {
                        self.forget_request(request_key, &retained_buffers.into_iter().collect());
                    }
                    BatchCommand::Free { buffer } => {
                        self.buffer_workers.remove(&buffer);
                        self.buffer_allocations
                            .retain(|(_, _, identity), _| *identity != buffer);
                        self.buffer_routes.retain(|identity, _| *identity != buffer);
                        self.cache_transfers.remove(&buffer);
                        self.transfer_products
                            .retain(|identity, _| *identity != buffer);
                    }
                    BatchCommand::Start { .. } => {
                        unreachable!("admissions are tracked by worker ownership")
                    }
                }
            }
        }
        if !result.results.is_empty() || done {
            self.ready.push_back(result);
        }
        Ok(())
    }

    /// Retires request routing while preserving independently owned product locations.
    fn forget_request(&mut self, request: RequestKey, retained: &HashSet<BufferId>) {
        self.admissions
            .retain(|request_key, _| *request_key != request);
        self.admitted_workers
            .retain(|(_, request_key)| *request_key != request);
        self.buffer_routes
            .retain(|buffer, _| buffer.owner != request || retained.contains(buffer));
        self.transfer_products
            .retain(|buffer, _| buffer.owner != request || retained.contains(buffer));
        self.cache_transfers
            .retain(|buffer, _| buffer.owner != request || retained.contains(buffer));
        self.buffer_allocations
            .retain(|(_, _, buffer), _| buffer.owner != request || retained.contains(buffer));
        self.buffer_workers
            .retain(|buffer, _| buffer.owner != request || retained.contains(buffer));
    }
}
