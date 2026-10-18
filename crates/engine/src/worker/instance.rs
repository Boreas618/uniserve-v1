//! Physical worker fan-out, completion agreement, and process recovery.
//!
//! A `WorkerGroup` presents one worker's cooperating ranks to `WorkerExecutor`
//! as a single submission queue of `depth` physical runs. `submit_batch` uses
//! `rank_projection` to cut each `Batch` into per-rank projections under the
//! same batch identifier, each holding the calls that rank's components
//! execute and every command of the batch, and sends each projection through
//! that rank's `RankProcess`. Returned rank results wait in per-rank buffers
//! until every rank planned for a set of calls has reported them; `try_join`
//! then merges those reports into one `WorkerResult` published by the calls'
//! output owner, the first rank of their component. An execution error that
//! every failing rank reports identically, while no succeeding rank holds an
//! unjoined call, retires only its batch unless it is fatal, is coded as a
//! scheduler bug or invariant violation, or failed a batch carrying a `Free`
//! command (`join_rank_errors`); rank loss, a lane a rank reports stalled,
//! protocol violations, and every other error replace the whole rank group
//! (`recover_workers`) and invalidate every request resident on it.
//!
//! The module also launches groups: `spawn_all` starts every group's ranks,
//! locally or through the host launchers in `launcher`, before any group
//! waits for its ranks' registrations and startup capabilities.

mod reports;
pub(crate) mod routing;
mod spawn;

use crate::executor::WorkerResult;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use super::registry::Registry;
use super::{BatchSubmitError, IncarnationRegistry, RankProcess};
use crate::executor::{RankBuffer, WorkerExecError, WorkerFailure};
use uniserve_ipc::{Batch, BatchCommand, RequestKey, WorkerInfo};

use crate::worker::WorkerProcessArgs;

use reports::{
    merge_rank_report, report_call_ids, take_rank_calls, validate_and_order_rank_report,
};
use routing::rank_projection;

/// Cooperative worker processes with one IPC channel per rank.
pub struct WorkerGroup {
    /// Rank processes in rank order; empty after `close` or a failed recovery.
    workers: Vec<RankProcess>,
    /// Results each rank returned, indexed by rank. `take_rank_calls`
    /// consumes their calls as they are joined, and `finish_batch` drops what
    /// remains of a retired batch.
    buffers: Vec<VecDeque<WorkerResult>>,
    /// The group's agreed capabilities: rank 0's report with the group-wide
    /// product storage and cache byte accounting `from_ranks` derives.
    info: WorkerInfo,
    /// Physical runs admitted concurrently, from the ranks' queue depth.
    depth: usize,
    /// The last accepted batch identifier; `submit_batch` refuses a batch
    /// whose identifier does not exceed it.
    last_batch_id: Option<u64>,
    /// Time of construction or of the last submission, joined result, or
    /// group replacement, from which `WorkerProcessArgs::watchdog_timeout`
    /// is measured.
    last_progress: Instant,
    /// Admitted batches not yet retired, by batch identifier.
    pending_batches: BTreeMap<u64, PendingBatch>,
    /// Launch arguments. Recovery relaunches the group from them, and
    /// projection and joining read its placement and components.
    process_args: WorkerProcessArgs,
    /// Which component serves each media call across every worker of the
    /// deployment; the executor states it once all workers have reported.
    media_routing: BTreeMap<uniserve_ipc::MediaCall, String>,
    /// The worker and rank each live rank incarnation of the deployment
    /// serves. The group registers its own ranks and retires them when it
    /// loses them; the executor makes it the deployment's one table.
    incarnations: IncarnationRegistry,
    /// Requests that may hold state on the ranks: added by a submitted call
    /// or `Start` command, removed when a batch carrying their `Finish`
    /// command is joined as done. Recovery reports them as lost.
    resident_requests: HashSet<RequestKey>,
    /// Set by recovery and close; `take_readiness_change` reports and clears it.
    readiness_changed: bool,
    /// Set by `close` and by a failed recovery. A closed group accepts no
    /// batch and returns nothing from `poll_batch`.
    closed: bool,
    /// Rank loss is reported after all already-agreed results have been delivered.
    failure: Option<anyhow::Error>,
    /// Groups sharing an expert communicator cannot recover independently.
    /// The first terminal loss wakes every group's ordinary progress loop.
    failure_domain: Option<Arc<FailureDomain>>,
    failure_wake: Option<crate::handle::WakeSignal>,
    /// The deployment's launchers of the hosts this instance does not run
    /// on, shared with the other groups. A launcher terminates the ranks it
    /// started when the registry's last owner closes the connection, so the
    /// group holds it for as long as it holds those ranks.
    launchers: Option<super::launcher::Launchers>,
    /// The deployment's listener, which a relaunch's ranks connect to.
    registry: Registry,
}

/// One terminal failure shared by groups participating in the same collectives.
///
/// Each owner keeps its own pollable wake, so an idle replica observes failure
/// even when another replica consumes its notification first. Normal request
/// errors and graceful retirement do not fail the domain.
struct FailureDomain {
    cause: OnceLock<String>,
    wakers: Vec<uniserve_core::CommandWaker>,
}

impl FailureDomain {
    fn fail(&self, worker: &str, cause: &anyhow::Error) {
        if self
            .cause
            .set(format!("worker {worker}: {cause:#}"))
            .is_ok()
        {
            for waker in &self.wakers {
                waker.wake();
            }
        }
    }
}

/// A call's identity across request epochs: engine identifier, request
/// identifier, request epoch, and call identifier, in that order.
type CallIdentity = (u64, u64, u64, uniserve_ipc::CallId);

/// The worker routed to each component reading a call's products, keyed by
/// the call's request and call identifiers (`RequestPlacement::readers`).
pub type CallReaders =
    HashMap<(RequestKey, uniserve_ipc::CallId), BTreeMap<String, crate::WorkerId>>;

/// A physical batch and its participating ranks share one retirement lifetime.
struct PendingBatch {
    /// The batch as submitted, before rank projection.
    batch: Batch,
    /// Calls not yet joined into a published result.
    remaining: BTreeSet<CallIdentity>,
    /// Every rank that received a projection, by rank.
    ranks: BTreeMap<usize, RankResult>,
}

/// A participating rank's received prefix, independent of transport fragmentation.
struct RankResult {
    /// The calls planned on this rank and whether the rank returned each.
    calls: BTreeMap<CallIdentity, bool>,
    /// Whether the rank returned its result for the batch.
    complete: bool,
    /// The execution error the rank reported for the batch, if any.
    error: Option<WorkerExecError>,
}

/// Returns the request and call identifiers carried by a worker call.
fn call_identity(request: uniserve_ipc::RequestKey, call_id: uniserve_ipc::CallId) -> CallIdentity {
    (
        request.engine_id,
        request.request_id.0,
        request.request_epoch,
        call_id,
    )
}

impl WorkerGroup {
    /// Drains immediately available rank results into per-rank agreement buffers.
    ///
    /// A result that does not match the rank's plan for its batch fails the
    /// drain. A rank error that `record_rank_error` accepts is recorded for
    /// `join_rank_errors`; any other is kept as the group's first `failure`
    /// and ends that rank's drain, so results already buffered can still be
    /// joined before `poll_batch` reports the failure.
    fn pump_once(&mut self) -> anyhow::Result<()> {
        for rank in 0..self.workers.len() {
            loop {
                let result = self.workers[rank].poll_batch(Duration::ZERO);
                match result {
                    Ok(Some(result)) => {
                        let batch_id = result.batch_id;
                        let progress = self
                            .pending_batches
                            .get_mut(&batch_id)
                            .and_then(|batch| batch.ranks.get_mut(&rank))
                            .ok_or_else(|| {
                                anyhow::anyhow!("rank {rank} returned unsubmitted batch {batch_id}")
                            })?;
                        anyhow::ensure!(
                            !progress.complete,
                            "rank {rank} returned after retirement"
                        );
                        let identities = report_call_ids(&result);
                        anyhow::ensure!(
                            result.products.iter().all(|product| identities.contains(
                                &call_identity(
                                    product.product.request_key,
                                    product.product.producer_call_id,
                                )
                            )),
                            "rank {rank} published a product without its call completion"
                        );
                        for identity in identities {
                            let completed = progress.calls.get_mut(&identity).ok_or_else(|| {
                                anyhow::anyhow!(
                                    "rank {rank} returned an unplanned call for batch {batch_id}"
                                )
                            })?;
                            anyhow::ensure!(
                                !*completed,
                                "rank {rank} returned a call more than once for batch {batch_id}"
                            );
                            *completed = true;
                        }
                        // A batch returns one result, so its every call
                        // completes with it.
                        anyhow::ensure!(
                            progress.calls.values().all(|completed| *completed),
                            "rank {rank} ended batch {batch_id} without all completions"
                        );
                        progress.complete = true;
                        self.buffers[rank].push_back(result);
                    }
                    Ok(None) => break,
                    Err(error) => {
                        if let Err(error) = self.record_rank_error(rank, &error) {
                            self.failure.get_or_insert(error);
                            break;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Records a rank's execution error against its pending batch.
    ///
    /// Fails, without recording, when the error is not a `WorkerExecError`,
    /// carries no batch identifier, names a batch or rank with no pending
    /// entry, or differs from an error the rank already reported for that
    /// batch.
    fn record_rank_error(&mut self, rank: usize, error: &anyhow::Error) -> anyhow::Result<()> {
        let execution = error
            .downcast_ref::<WorkerExecError>()
            .ok_or_else(|| anyhow::anyhow!("rank {rank} failed: {error:#}"))?;
        let batch_id = execution.batch_id.ok_or_else(|| {
            anyhow::anyhow!("rank {rank} returned an execution error without a batch identity")
        })?;
        let progress = self
            .pending_batches
            .get_mut(&batch_id)
            .and_then(|batch| batch.ranks.get_mut(&rank))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "rank {rank} returned an execution error for unknown batch {batch_id}"
                )
            })?;
        if let Some(existing) = &progress.error {
            anyhow::ensure!(
                *existing == *execution,
                "rank {rank} returned conflicting errors for batch {batch_id}"
            );
        } else {
            progress.error = Some(execution.clone());
        }
        Ok(())
    }

    /// Replaces the complete rank group after worker loss and invalidates affected requests.
    ///
    /// Returns the [`WorkerFailure`] that reports the lost allocations. Recovery
    /// always invalidates resident work, so the result is an error whether or
    /// not the replacement group started; a failed replacement also closes
    /// this group.
    fn recover_workers(&mut self, cause: &anyhow::Error) -> anyhow::Error {
        // The returned failure reaches its reader only once the replacement
        // group has started, so the cause is logged before the relaunch.
        tracing::warn!(
            worker = %self.process_args.worker_id,
            "replacing the rank group: {cause:#}"
        );

        if let Some(domain) = &self.failure_domain {
            domain.fail(&self.process_args.worker_id, cause);
        }
        // The lost endpoints, resident requests, and reported capabilities
        // are captured before the ranks and bookkeeping are cleared.
        let endpoints = self
            .workers
            .iter()
            .map(|worker| worker.info().endpoint.clone())
            .collect();
        let requests = self.resident_requests.iter().copied().collect();
        let expected = self
            .workers
            .iter()
            .map(|worker| worker.info().clone())
            .collect::<Vec<_>>();
        for worker in &mut self.workers {
            worker.terminate();
        }
        self.workers.clear();
        // The lost incarnations stop resolving at once, so a publication they
        // made is never bound as an input while the group is relaunched.
        self.incarnations
            .assign(&crate::WorkerId(self.process_args.worker_id.clone()), []);
        self.clear_execution();

        let recovery = (|| -> anyhow::Result<()> {
            // The group's ranks on other hosts are stopped by their launchers
            // before the group is relaunched through the same registry, which
            // stays connected for the deployment's other groups.
            if let Some(registry) = &self.launchers {
                super::launcher::lock(registry)?.stop_worker(&self.process_args.worker_id)?;
            }
            let (mut workers, launchers) =
                self.process_args
                    .launch(None, &self.registry, self.launchers.clone())?;
            self.launchers = launchers;
            super::process::finish_startup(&mut workers.iter_mut().collect::<Vec<_>>())?;
            anyhow::ensure!(
                workers.len() == expected.len(),
                "replacement rank count changed"
            );
            for (rank, (before, after)) in expected.iter().zip(&workers).enumerate() {
                validate_replacement_info(before, after.info(), rank)?;
            }
            for worker in &mut workers {
                worker.check_worker("WorkerGroup readiness")?;
                worker.set_startup_cancel(None);
            }
            self.install_replacement(workers);
            Ok(())
        })();

        self.readiness_changed = true;
        let recovery_status = match recovery {
            Ok(()) => "rank group recovered".to_owned(),
            Err(error) => {
                self.closed = true;
                format!("rank recovery failed: {error:#}")
            }
        };
        WorkerFailure {
            worker_id: crate::WorkerId(self.process_args.worker_id.clone()),
            endpoints,
            requests,
            retired: Vec::new(),
            buffers: Vec::new(),
            execution: cause.downcast_ref::<WorkerExecError>().cloned(),
            message: format!(
                "WorkerGroup {} lost its resident allocations; {recovery_status}: {cause:#}",
                self.process_args.worker_id
            ),
        }
        .into()
    }

    /// Installs a capability-compatible replacement rank group and resets rank-local state.
    fn install_replacement(&mut self, workers: Vec<RankProcess>) {
        // `validate_replacement_info` has required every replacement rank to
        // match its predecessor apart from the endpoint, so the group's agreed
        // info keeps everything but rank 0's new endpoint.
        self.info.endpoint = workers[0].info().endpoint.clone();
        self.incarnations.assign(
            &crate::WorkerId(self.process_args.worker_id.clone()),
            workers.iter().map(|worker| &worker.info().endpoint),
        );
        self.workers = workers;
        self.last_progress = Instant::now();
        let ranks = self.process_args.ranks.len();
        self.buffers = (0..ranks).map(|_| VecDeque::new()).collect();
    }

    /// Clears execution bookkeeping after physical ownership has retired.
    fn clear_execution(&mut self) {
        self.failure = None;
        self.pending_batches.clear();
        self.buffers.iter_mut().for_each(VecDeque::clear);
        self.resident_requests.clear();
    }

    /// Joins mutually agreeing rank reports into one logical physical result.
    ///
    /// Takes the call set `joinable_report_key` selects from every rank
    /// planned for it, validates each report against the batch, and merges
    /// the participants into the output owner's report, which alone carries
    /// public output and forward statistics. The result is `done` once every
    /// call is joined and every rank of the batch has completed; the batch is
    /// then retired. When nothing is joinable, `join_rank_errors` may resolve
    /// a batch whose ranks reported execution errors, returning its error;
    /// otherwise the result is `None`.
    fn try_join(&mut self) -> anyhow::Result<Option<WorkerResult>> {
        let Some((batch_id, output_rank, call_ids)) = self.joinable_report_key() else {
            // Preserve successful partial results already agreed by every rank
            // before retiring the unresolved remainder of a failed batch.
            self.join_rank_errors()?;
            return Ok(None);
        };

        let pending = self
            .pending_batches
            .get_mut(&batch_id)
            .ok_or_else(|| anyhow::anyhow!("joined batch {batch_id} has no pending batch"))?;
        // The ranks planned for every selected call take part; the final join,
        // with an empty call set, takes every rank of the batch.
        let participants = pending
            .ranks
            .iter()
            .filter_map(|(&rank, result)| {
                (call_ids.is_empty() || call_ids.iter().all(|id| result.calls.contains_key(id)))
                    .then_some(rank)
            })
            .collect::<Vec<_>>();
        let batch = &pending.batch;
        let mut out = take_rank_calls(&mut self.buffers[output_rank], batch, &call_ids, true)?;
        validate_and_order_rank_report(batch, &mut out, output_rank)?;
        for rank in participants.into_iter().filter(|rank| *rank != output_rank) {
            let mut report = take_rank_calls(&mut self.buffers[rank], batch, &call_ids, false)?;
            validate_and_order_rank_report(batch, &mut report, rank)?;
            merge_rank_report(batch, &mut out, &report, rank)?;
        }
        let remaining = &mut pending.remaining;
        for identity in &call_ids {
            anyhow::ensure!(
                remaining.remove(identity),
                "joined batch {batch_id} repeated a call"
            );
        }
        let done = remaining.is_empty() && pending.ranks.values().all(|result| result.complete);
        out.done = done;
        if done {
            // Report each rank's accumulated execution time once, after every
            // rank reports. Ranks execute concurrently, so the batch's duration
            // is the maximum of these sums rather than their total.
            out.worker_exec_us = self
                .buffers
                .iter()
                .filter_map(|buffer| {
                    buffer
                        .iter()
                        .filter(|report| report.batch_id == batch_id)
                        .filter_map(|report| report.worker_exec_us)
                        .reduce(u64::saturating_add)
                })
                .max();
            for command in &pending.batch.commands {
                if let BatchCommand::Finish { request_key, .. } = command {
                    self.resident_requests.remove(request_key);
                }
            }
            self.finish_batch(batch_id);
        }
        Ok(Some(out))
    }

    /// Returns the file descriptors that signal worker progress.
    pub(crate) fn progress_fds(&self) -> Vec<libc::pollfd> {
        self.workers
            .iter()
            .flat_map(RankProcess::progress_fds)
            .chain(
                self.failure_wake
                    .iter()
                    .filter(|_| !self.closed)
                    .map(|wake| libc::pollfd {
                        fd: wake.descriptor(),
                        events: libc::POLLIN,
                        revents: 0,
                    }),
            )
            .collect()
    }

    fn peer_failure(&self) -> Option<anyhow::Error> {
        self.failure_domain
            .as_ref()
            .and_then(|domain| domain.cause.get())
            .map(|cause| anyhow::anyhow!("shared expert communicator failed: {cause}"))
    }

    /// Resolves a batch once every rank reports either success or a compatible error.
    ///
    /// Only batches with a rank error whose other ranks have all completed or
    /// failed are resolved, at most one per invocation. When a rank that
    /// succeeded still holds unjoined calls, the ranks disagree: the batch is
    /// retired and a plain error returned. Otherwise every error must be
    /// identical, or a plain error is returned. A fatal error, an
    /// `InvariantViolation` code, or a batch carrying a `Free` command returns the `WorkerExecError` itself; `poll_batch`
    /// answers these and the plain errors with recovery. Any other error
    /// retires the batch and returns a `WorkerFailure` naming its unjoined
    /// calls and affected requests, which `poll_batch` returns without
    /// recovery.
    fn join_rank_errors(&mut self) -> anyhow::Result<()> {
        let failing = self
            .pending_batches
            .iter()
            .filter_map(|(&batch_id, pending)| {
                pending
                    .ranks
                    .values()
                    .any(|rank| rank.error.is_some())
                    .then_some(batch_id)
            })
            .collect::<Vec<_>>();
        for batch_id in failing {
            let pending = &self.pending_batches[&batch_id];
            if pending
                .ranks
                .values()
                .any(|rank| rank.error.is_none() && !rank.complete)
            {
                continue;
            }
            let errors = pending
                .ranks
                .iter()
                .filter_map(|(&rank, result)| result.error.as_ref().map(|error| (rank, error)))
                .collect::<Vec<_>>();
            if pending.ranks.values().any(|result| {
                result.error.is_none()
                    && result.calls.keys().any(|id| pending.remaining.contains(id))
            }) {
                let details = errors
                    .iter()
                    .map(|(rank, error)| format!("rank {rank}: {error}"))
                    .collect::<Vec<_>>()
                    .join("; ");
                self.finish_batch(batch_id);
                anyhow::bail!(
                    "worker ranks disagreed between success and failure for batch {batch_id}: {details}"
                );
            }
            let canonical = errors[0].1.clone();
            for (rank, error) in errors.iter().copied().skip(1) {
                anyhow::ensure!(
                    *error == canonical,
                    "rank {rank} returned a different execution error for batch {batch_id}"
                );
            }
            let batch = &self.pending_batches[&batch_id].batch;
            if canonical.fatal
                || canonical.code == Some(uniserve_ipc::ErrorCode::InvariantViolation)
                || batch
                    .commands
                    .iter()
                    .any(|command| matches!(command, BatchCommand::Free { .. }))
            {
                // A failed physical release cannot return an allocation safely.
                // Replacing this instance proves retirement without retrying
                // the same unsuccessful release indefinitely.
                return Err(canonical.into());
            }
            let pending = &self.pending_batches[&batch_id].remaining;
            let retired = batch
                .calls
                .iter()
                .filter(|call| pending.contains(&call_identity(call.request_key, call.call_id)))
                .map(|call| (batch.batch_id, call.request_key, call.call_id))
                .collect::<Vec<_>>();
            let requests = retired
                .iter()
                .map(|(_, request, _)| *request)
                .chain(batch.commands.iter().map(BatchCommand::request_key))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            let failure = WorkerFailure {
                worker_id: crate::WorkerId(self.process_args.worker_id.clone()),
                endpoints: Vec::new(),
                requests,
                retired,
                buffers: Vec::new(),
                message: canonical.to_string(),
                execution: Some(canonical),
            };
            self.finish_batch(batch_id);
            return Err(failure.into());
        }
        Ok(())
    }

    /// Retires all rank-local tracking for one agreed terminal batch.
    fn finish_batch(&mut self, batch_id: u64) {
        self.pending_batches.remove(&batch_id);
        for buffer in &mut self.buffers {
            buffer.retain(|report| report.batch_id != batch_id);
        }
    }

    /// Join one component's completed work independently of rank and transport framing.
    ///
    /// Batches are considered in identifier order. A batch whose calls are
    /// all joined yields an empty call set once every participating rank has
    /// completed, owned by its lowest participating rank; that final join
    /// takes no call and marks the batch done once ranks without a pending
    /// call, such as ranks that received only commands, have reported.
    /// Otherwise a pending call's output owner is the first rank of its
    /// component, and one of the owner's reports yields every call it carries
    /// that is still pending, belongs to the same component, is planned on
    /// the same ranks, and has been returned by each of them.
    /// Returns the batch identifier, the output owner, and the call set.
    fn joinable_report_key(&self) -> Option<(u64, usize, Vec<CallIdentity>)> {
        for (&batch_id, pending_batch) in &self.pending_batches {
            let pending = &pending_batch.remaining;
            if pending.is_empty() {
                if pending_batch.ranks.values().all(|result| result.complete) {
                    return Some((
                        batch_id,
                        *pending_batch.ranks.first_key_value()?.0,
                        Vec::new(),
                    ));
                }
                continue;
            }
            let batch = &self.pending_batches[&batch_id].batch;
            for call in &batch.calls {
                let identity = call_identity(call.request_key, call.call_id);
                if !pending.contains(&identity) {
                    continue;
                }
                let owner = self.process_args.components[&call.component].ranks[0];
                let members = self.call_members(batch_id, identity);
                for report in self.buffers[owner]
                    .iter()
                    .filter(|report| report.batch_id == batch_id)
                {
                    let shared = report_call_ids(report)
                        .into_iter()
                        .filter(|id| {
                            pending.contains(id)
                                && self.call_members(batch_id, *id) == members
                                && batch.calls.iter().any(|candidate| {
                                    call_identity(candidate.request_key, candidate.call_id) == *id
                                        && candidate.component == call.component
                                })
                                && members.iter().all(|rank| {
                                    self.buffers[*rank].iter().any(|result| {
                                        result.batch_id == batch_id
                                            && result.results.iter().any(|completion| {
                                                call_identity(
                                                    completion.output.request_key,
                                                    completion.output.call_id,
                                                ) == *id
                                            })
                                    })
                                })
                        })
                        .collect::<Vec<_>>();
                    if !shared.is_empty() {
                        return Some((batch_id, owner, shared));
                    }
                }
            }
        }
        None
    }

    /// Returns the ranks, in rank order, whose projection of the batch plans
    /// the call.
    fn call_members(&self, batch_id: u64, identity: CallIdentity) -> Vec<usize> {
        self.pending_batches[&batch_id]
            .ranks
            .iter()
            .filter_map(|(&rank, result)| result.calls.contains_key(&identity).then_some(rank))
            .collect()
    }
}

/// Validates that a replacement rank preserves the established worker contract.
fn validate_replacement_info(
    expected: &WorkerInfo,
    actual: &WorkerInfo,
    rank: usize,
) -> anyhow::Result<()> {
    let mut normalized_expected = expected.clone();
    anyhow::ensure!(
        actual.endpoint.incarnation != expected.endpoint.incarnation,
        "replacement rank {rank} did not establish a new endpoint incarnation"
    );
    normalized_expected.endpoint = actual.endpoint.clone();
    anyhow::ensure!(
        normalized_expected == *actual,
        "replacement rank {rank} worker info changed"
    );
    Ok(())
}

impl WorkerGroup {
    /// Reports whether this instance is open and holds its rank group.
    ///
    /// `close` and a failed recovery make it permanently unready.
    pub fn is_ready(&self) -> bool {
        !self.closed && !self.workers.is_empty()
    }

    /// Physical submission slots not yet owned by accepted batches.
    pub(crate) fn available_slots(&self) -> usize {
        if self.is_ready() {
            self.depth.saturating_sub(self.pending_batches.len())
        } else {
            0
        }
    }

    /// Reports and clears whether recovery or close ran since the last call.
    pub(crate) fn take_readiness_change(&mut self) -> bool {
        std::mem::take(&mut self.readiness_changed)
    }

    /// Returns the group's agreed worker metadata.
    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Return the same bindings used to initialize this instance's transports.
    pub(crate) fn transfer_config(&self) -> &crate::executor::TransferConfig {
        &self.process_args.transfer
    }

    /// Exposes accepted calls not yet joined into a published result.
    pub(crate) fn inflight_calls(&self) -> impl Iterator<Item = &uniserve_ipc::Call> {
        self.pending_batches.values().flat_map(|pending| {
            pending.batch.calls.iter().filter(|call| {
                pending
                    .remaining
                    .contains(&call_identity(call.request_key, call.call_id))
            })
        })
    }

    /// Returns the loaded physical endpoints used for transfer binding.
    pub(crate) fn rank_info(&self, rank: usize) -> Option<&WorkerInfo> {
        self.workers.get(rank).map(RankProcess::info)
    }

    /// Submit each call to its component members and lifetime commands to the rank group.
    ///
    /// Returns `WouldBlock` with the batch when the group holds no ranks or
    /// all `depth` slots are held. A closed group, a batch identifier that does
    /// not exceed the previous one, an invalid batch, or a failed projection
    /// or input binding returns `Failed` with no state change; a failed input
    /// binding carries the first refusing rank's `UnroutableInputs`. A rank
    /// submission that fails after the batch is recorded replaces the rank
    /// group and returns `Failed` with the recovery's `WorkerFailure`.
    ///
    /// `readers` names, for the batch's routed media calls, the worker each
    /// consuming component was routed to; each call's products then name
    /// only those workers' reading ranks (see `media_consumer_slots`).
    ///
    /// `buffers` places the batch's persistent buffers in each rank's own
    /// pool; a rank receives the spans of the buffers its calls read and
    /// write. The batch itself carries no spans (`Batch::validate_unplaced`).
    pub fn submit_batch(
        &mut self,
        batch: Batch,
        readers: &CallReaders,
        buffers: &[RankBuffer],
    ) -> Result<(), BatchSubmitError> {
        if self.closed {
            return Err(BatchSubmitError::Failed(anyhow::anyhow!(
                "WorkerGroup is closed"
            )));
        }
        if let Some(error) = self.peer_failure() {
            return Err(BatchSubmitError::Failed(self.recover_workers(&error)));
        }
        if self
            .last_batch_id
            .is_some_and(|last| batch.batch_id <= last)
        {
            return Err(BatchSubmitError::Failed(anyhow::anyhow!(
                "batch IDs must increase in submission order"
            )));
        }
        if !self.is_ready() || self.pending_batches.len() >= self.depth {
            return Err(BatchSubmitError::WouldBlock(Box::new(batch)));
        }
        // The batch was validated where it was lowered; each rank's
        // projection is validated where it is built (`rank_projection`).
        let rank_batches = rank_projection(
            &batch,
            &self.process_args.components,
            self.workers.len(),
            buffers,
            |call, members, rank| {
                let routed = readers.get(&(call.request_key, call.call_id));
                self.consumer_slots(call, routed, members, rank)
            },
        )
        .and_then(|batches| {
            batches
                .into_iter()
                .map(|(rank, mut batch)| {
                    self.process_args.transfer.bind_inputs(
                        &self.incarnations,
                        &mut batch.input_products,
                        &mut batch.cache_inputs,
                        &self.workers[rank].info().endpoint,
                    )?;
                    Ok((rank, batch))
                })
                .collect::<anyhow::Result<Vec<_>>>()
        })
        .map_err(BatchSubmitError::Failed)?;

        // The batch's requests are recorded as resident before any rank
        // receives it, so the recovery that follows a failed rank submission
        // reports them as lost too.
        let batch_id = batch.batch_id;
        let requests = batch
            .calls()
            .map(|call| call.request_key)
            .collect::<Vec<_>>();
        self.last_batch_id = Some(batch_id);
        self.resident_requests.extend(requests);
        self.resident_requests
            .extend(batch.commands.iter().filter_map(|command| match command {
                BatchCommand::Start { request } => Some(request.request_key),
                _ => None,
            }));
        let ranks = rank_batches
            .iter()
            .map(|(rank, batch)| {
                (
                    *rank,
                    RankResult {
                        calls: batch
                            .calls
                            .iter()
                            .map(|call| (call_identity(call.request_key, call.call_id), false))
                            .collect(),
                        complete: false,
                        error: None,
                    },
                )
            })
            .collect();
        self.pending_batches.insert(
            batch_id,
            PendingBatch {
                remaining: batch
                    .calls
                    .iter()
                    .map(|call| call_identity(call.request_key, call.call_id))
                    .collect(),
                batch,
                ranks,
            },
        );
        self.last_progress = Instant::now();

        for (rank, batch) in rank_batches {
            if let Err(error) = self.workers[rank].submit_batch(batch) {
                // The group admitted the batch against the queue depth every
                // rank was launched with, so a rank that hands it back, for a
                // full queue or a full socket send queue, fails the group
                // rather than applying backpressure.
                let error = match error {
                    BatchSubmitError::WouldBlock(_) => {
                        anyhow::anyhow!(
                            "physical rank {rank} rejected an admitted cooperative batch as full"
                        )
                    }
                    BatchSubmitError::Failed(error) => {
                        error.context(format!("submit to rank {rank} failed"))
                    }
                };
                return Err(BatchSubmitError::Failed(self.recover_workers(&error)));
            }
        }
        Ok(())
    }

    /// Waits for rank progress and returns the next fully agreed physical result.
    ///
    /// Returns `None` when `timeout` passes without a joinable result, and at
    /// once for a closed group. The `WorkerFailure` of an agreed execution
    /// error that `join_rank_errors` retires with its batch is returned as is
    /// and the group stays usable. Any other error, including rank loss, a
    /// protocol violation, an agreed error `join_rank_errors` does not retire,
    /// a lane a rank reports stalled, and an expired
    /// `WorkerProcessArgs::watchdog_timeout` (measured from the group's last
    /// progress and enforced only while a batch is pending), replaces the
    /// rank group and returns the recovery's `WorkerFailure`.
    pub fn poll_batch(&mut self, timeout: Duration) -> anyhow::Result<Option<WorkerResult>> {
        if self.closed {
            return Ok(None);
        }
        match self.poll_progress(timeout) {
            Ok(report) => Ok(report),
            Err(error) if error.is::<WorkerFailure>() => Err(error),
            Err(error) => Err(self.recover_workers(&error)),
        }
    }

    fn poll_progress(&mut self, timeout: Duration) -> anyhow::Result<Option<WorkerResult>> {
        let deadline = Instant::now() + timeout;
        loop {
            // A drain can observe both final replies and rank death. Join every
            // complete reply before recovery invalidates outstanding ownership.
            if let Some(result) = self.try_join()? {
                self.last_progress = Instant::now();
                return Ok(Some(result));
            }
            if let Some(error) = self.failure.take() {
                return Err(error);
            }
            if let Some(error) = self.peer_failure() {
                return Err(error);
            }
            self.pump_once()?;
            if let Some(result) = self.try_join()? {
                self.last_progress = Instant::now();
                return Ok(Some(result));
            }
            if let Some(error) = self.failure.take() {
                return Err(error);
            }
            self.check_progress_deadline()?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            crate::worker::park_descriptors(&self.progress_fds(), remaining)?;
        }
    }

    /// Fails when a batch is pending and no rank has made progress within
    /// the watchdog timeout, measured from `last_progress`. Expiry is an
    /// error that `poll_batch` answers by replacing the rank group.
    fn check_progress_deadline(&self) -> anyhow::Result<()> {
        let timeout = self.process_args.watchdog_timeout;
        anyhow::ensure!(
            self.pending_batches.is_empty() || self.last_progress.elapsed() < timeout,
            "cooperative workers produced no progress within {timeout:?}",
        );
        Ok(())
    }

    fn release_closed_resources(&mut self) {
        // A closed group never triggers recovery: `poll_batch` returns early
        // and `submit_batch` refuses.
        self.closed = true;
        self.workers.clear();
        self.clear_execution();
        self.readiness_changed = true;
    }

    /// Closes every rank in the physical worker group.
    ///
    /// The ranks are closed together (`close_ranks`): each is asked to shut
    /// down before any is waited for, since a rank's teardown retires the
    /// communicators it shares with the others and completes only once all
    /// of them take part. Ranks on other hosts are followed until their
    /// launchers report their exits. Failures to reach or stop a rank are
    /// absorbed, so this returns `Ok`.
    pub fn close(&mut self) -> anyhow::Result<()> {
        self.closed = true;
        for worker in &self.workers {
            worker.cancel_startup();
        }
        if self.peer_failure().is_some() {
            // A failed collective cannot drain. Terminate the local readers
            // and ask remote launchers to do the same before dropping storage.
            for worker in &mut self.workers {
                worker.terminate();
            }
            if let Some(registry) = &self.launchers {
                let _ = super::launcher::lock(registry)
                    .and_then(|mut hosts| hosts.stop_worker(&self.process_args.worker_id));
            }
            self.release_closed_resources();
            return Ok(());
        }
        super::process::close_ranks(
            &mut self.workers,
            self.launchers
                .as_ref()
                .map(|launchers| (launchers, self.process_args.worker_id.as_str())),
        );
        self.release_closed_resources();
        Ok(())
    }
}

impl Drop for WorkerGroup {
    /// Closes the group's ranks together, as `close` does; a closed group
    /// has no ranks left.
    fn drop(&mut self) {
        let _ = self.close();
    }
}
