//! Startup warmup: graph budgets, capability preparation and capture,
//! representative requests, then sealing.
//!
//! Phase order (identical on every rank, because capture and scenarios run
//! collectives):
//!
//! 1. Graph budgets: each CUDA device's graph pools may hold what they hold
//!    now plus the grant left after product storage not yet resident.
//! 2. Capability preparation ([`WarmupHooks::prepare`]): media layouts,
//!    encoders and decoders, each runtime's placeholder steps.
//! 3. Capture ([`WarmupHooks::capture`]): graphs in each runtime's kind and
//!    shape order, every capture followed by a graph-storage check.
//! 4. Scenarios: representative requests through the serving executor
//!    ([`Scenarios`]), the lifecycle scenario first, then each runtime's.
//! 5. Completion: kernel table, graph storage seal.
//!
//! Capability runtimes contribute phases 2 to 4 through [`WarmupHooks`].

use std::sync::Arc;
use std::time::Duration;

use uniserve_core::{RequestId, SamplingParams};
use uniserve_ipc::{ArRequestParams, Batch as BatchPlan, BatchCommand, NewRequest, RequestKey};

use crate::error::WorkerError;
use crate::executor::backend::{WorkerBackend, WorkerBatch};
use crate::executor::{Batch, BatchResult, Executor};
use crate::{Error, Result};

/// Engine identity of warmup requests; no engine uses it.
const WARMUP_ENGINE: u64 = u64::MAX;

/// Warmup work of one capability runtime. Every method runs on every rank
/// of the group in the same order.
pub trait WarmupHooks {
    /// Prepare modules that serve several sizes and run their placeholder
    /// steps.
    fn prepare(&mut self, _backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        Ok(())
    }

    /// Capture the runtime's graphs.
    fn capture(&mut self, _backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        Ok(())
    }

    /// Run representative requests through the executor.
    fn scenarios(
        &mut self,
        _scenarios: &mut Scenarios<'_>,
    ) -> std::result::Result<(), WorkerError> {
        Ok(())
    }
}

/// Runs synthetic batches through the serving executor before it serves.
///
/// Batch ids count from one with collective sequences `16 * id + 1`, so a
/// runtime's warmup collectives order identically on every rank; request
/// slots are taken downward from the highest, so prefix and request rows of
/// one scenario never collide.
pub struct Scenarios<'a> {
    executor: &'a mut Executor<WorkerBackend>,
    next_batch: u64,
    next_slot: u32,
    next_request: u64,
}

impl<'a> Scenarios<'a> {
    pub fn new(executor: &'a mut Executor<WorkerBackend>) -> Self {
        let slots = executor.backend().info().request_slots;
        Self {
            executor,
            next_batch: 1,
            next_slot: slots,
            next_request: 1,
        }
    }

    pub fn backend(&mut self) -> &mut WorkerBackend {
        self.executor.backend_mut()
    }

    /// The next batch id and its collective sequence.
    pub fn batch_id(&mut self) -> (u64, u64) {
        let id = self.next_batch;
        self.next_batch += 1;
        (id, 16 * id + 1)
    }

    /// A fresh request slot, counting down from the highest.
    pub fn slot(&mut self) -> Result<u32> {
        if self.next_slot == 0 {
            return Err(Error::Resource(
                "warmup scenarios exhausted the request slots",
            ));
        }
        let slot = self.next_slot;
        self.next_slot -= 1;
        Ok(slot)
    }

    /// A fresh warmup request identity.
    pub fn request_key(&mut self) -> RequestKey {
        let key = RequestKey::new(WARMUP_ENGINE, RequestId(self.next_request), 0);
        self.next_request += 1;
        key
    }

    /// Submit one batch and drive it to its result.
    pub fn run(&mut self, plan: BatchPlan) -> std::result::Result<BatchResult, WorkerError> {
        let plan = Arc::new(plan);
        let batch = Batch::from_plan(&plan, WorkerBatch::new(Arc::clone(&plan)));
        let submission = self.executor.submit(batch, true)?;
        loop {
            self.executor.advance()?;
            if let Some(result) = self.executor.poll(&submission)? {
                return Ok(result);
            }
            std::thread::sleep(Duration::from_micros(50));
        }
    }

    /// Admit one request and finish it: the executor's admission,
    /// retirement and result paths, with no numerical work.
    pub fn lifecycle(&mut self) -> std::result::Result<(), WorkerError> {
        let slot = self.slot()?;
        let key = self.request_key();
        let admission = NewRequest::new(
            key,
            slot,
            Some(ArRequestParams {
                sampling: SamplingParams::default(),
                negative_token_ids: Vec::new(),
                finish_token_ids: Vec::new(),
                initial_position: 0,
                canvas: None,
            }),
            None,
            0,
        )
        .map_err(|error| Error::Invalid(error.to_string()))?;
        let (id, sequence) = self.batch_id();
        let mut start = BatchPlan::new(id, vec![admission], Vec::new());
        start.collective_seq = sequence;
        self.run(start)?;

        let (id, sequence) = self.batch_id();
        let mut finish =
            BatchPlan::new(id, Vec::new(), Vec::new()).with_commands(vec![BatchCommand::Finish {
                request_key: key,
                retained_buffers: Vec::new(),
            }]);
        finish.collective_seq = sequence;
        self.run(finish)?;
        self.release(key);
        Ok(())
    }

    /// Forget a retired warmup request, returning its slot. Serving starts
    /// with no resident request.
    pub fn release(&mut self, key: RequestKey) {
        let requests = self.executor.backend_mut().requests_mut();
        if requests.peek(key.request_id.0).is_some() {
            requests.remove(key.request_id.0);
            self.next_slot += 1;
        }
    }
}
