//! The mechanism's one door to execution.
//!
//! The step submits a batch, waits for results, and asks what the pool can do;
//! it does not know whether the batch goes to worker processes over the ipc
//! channel or to the in-process device executor (design §2.4). This module is
//! that seam: the scheduler holds [`Submissions`], and swapping what sits
//! behind it is a change to this file, not to the step loop, the completion
//! path, or the admission of results.
//!
//! Today one implementation sits behind it, the worker-process
//! [`Executor`](crate::executor::Executor) the mechanism has always used.
//! `executor::Batches` — `submit`/`shadow`, `inventory()`, and a
//! `Submission::wait()` answering `Outcome<BatchOutput>` — is the seam the
//! in-process path plugs into; it is not wired here yet, and the plan for
//! finishing it is `notes/sched/splice.md`.

use std::collections::BTreeMap;
use std::time::Duration;

use crate::WorkerGroupId;
use crate::executor::ExecutionBatch;
use crate::executor::{Executor, ExecutorError, ExecutorInfo, ExecutorSubmitError};
use crate::scheduler::completion::BatchReport;
use ipc::BatchCommand;

/// What the mechanism executes through.
pub(super) struct Submissions {
    executor: Box<dyn Executor>,
}

impl Submissions {
    pub(super) fn new(executor: Box<dyn Executor>) -> Self {
        Self { executor }
    }

    /// The pool's capabilities: its workers, components, pools, and limits.
    pub(super) fn info(&self) -> &ExecutorInfo {
        self.executor.info()
    }

    /// Whether every worker of `worker` is ready to accept execution.
    pub(super) fn is_ready(&self, worker: &WorkerGroupId) -> bool {
        self.executor.is_ready(worker)
    }

    /// Whether `worker` executes nothing further.
    pub(super) fn is_closed(&self, worker: &WorkerGroupId) -> bool {
        self.executor.is_closed(worker)
    }

    /// Whether `worker` has an unclaimed destination submission slot.
    pub(super) fn has_capacity(&self, worker: &WorkerGroupId) -> bool {
        self.executor.has_capacity(worker)
    }

    /// Whether every physical owner of a lifecycle command can accept it.
    pub(super) fn command_has_capacity(&self, command: &BatchCommand) -> bool {
        self.executor.command_has_capacity(command)
    }

    /// Submits one logical batch without blocking for capacity.
    pub(super) fn submit(&mut self, batch: ExecutionBatch) -> Result<(), ExecutorSubmitError> {
        self.executor.submit(batch)
    }

    /// Waits up to `timeout` for one partial or terminal batch result.
    /// Waits up to `timeout` for one partial or terminal batch result, in the
    /// mechanism's own completion type.
    pub(super) fn poll(&mut self, timeout: Duration) -> Result<Option<BatchReport>, ExecutorError> {
        Ok(self.executor.poll(timeout)?.map(BatchReport::from_wire))
    }

    /// Closes the pool and its physical workers.
    pub(super) fn close(&mut self) -> Result<(), ExecutorError> {
        self.executor.close()
    }

    /// The nodes that hold the deployment's storage.
    pub(super) fn storage_nodes(&self) -> BTreeMap<String, String> {
        self.executor.storage_nodes()
    }

    /// Maps the host KV cache's page pools on every worker's ranks, each
    /// holding `tokens` tokens of every cache group.
    pub(super) fn map_host_cache(
        &mut self,
        tokens: u64,
    ) -> Result<Vec<(WorkerGroupId, Vec<String>)>, ExecutorError> {
        self.executor.map_host_cache(tokens)
    }

    /// Commits the pool's world: every worker takes the predecessor world's
    /// storage it mapped read-only.
    pub(super) fn commit(&mut self) -> Result<Vec<String>, ExecutorError> {
        self.executor.commit()
    }

    /// Publishes this world's immutable pages after its commit decision.
    pub(super) fn publish_world(&mut self) -> Result<(), ExecutorError> {
        self.executor.publish_world()
    }

    /// Ends the storage of every world of `worlds` on the pool's nodes.
    pub(super) fn retire(&mut self, worlds: &[u64]) -> Result<(), ExecutorError> {
        self.executor.retire(worlds)
    }

    /// Keeps the weights alone of every world of `worlds`.
    pub(super) fn keep_weights(
        &mut self,
        worlds: &[u64],
        to_host: bool,
        cache_keys: &[Vec<u8>],
    ) -> Result<Vec<String>, ExecutorError> {
        self.executor.keep_weights(worlds, to_host, cache_keys)
    }

    /// Halves, or restores when not `halve`, the deployment's link budgets.
    pub(super) fn throttle(&mut self, halve: bool) -> Result<Vec<String>, ExecutorError> {
        self.executor.throttle(halve)
    }

    /// The storage ledgers of the pool's nodes.
    pub(super) fn ledgers(&mut self) -> Result<Vec<String>, ExecutorError> {
        self.executor.ledgers()
    }

    /// Runs the worker hook `name` on every worker while none runs a batch.
    pub(super) fn hook(&mut self, name: &str, payload: &str) -> Result<Vec<String>, ExecutorError> {
        self.executor.hook(name, payload)
    }

    /// Runs the worker hook `name`, answering each worker group with its
    /// ranks' results.
    pub(super) fn hook_groups(
        &mut self,
        name: &str,
        payload: &str,
    ) -> Result<Vec<(WorkerGroupId, Vec<String>)>, ExecutorError> {
        self.executor.hook_groups(name, payload)
    }

    /// Commits prepared expert allocations and retains their order.
    pub(super) fn commit_experts(
        &mut self,
        order: &BTreeMap<String, Vec<u32>>,
    ) -> Result<Vec<String>, ExecutorError> {
        self.executor.commit_experts(order)
    }

    /// The executor itself, for the placement and view helpers that read a
    /// pool's workers and components without submitting anything.
    pub(super) fn inner(&self) -> &dyn Executor {
        &*self.executor
    }
}
