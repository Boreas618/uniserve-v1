//! A pool that serves every batch in process, and the inventory it states.
//!
//! This is the `Batches` seam the handle drives when a deployment's ranks are
//! absent: `SchedulerHandle::over_batches` builds a full handle over it — pools,
//! rows and data parallelism are checked as they are for a launch — so a case
//! written against it states the mechanism's behaviour rather than a process's.

use std::sync::Mutex;

use executor::batch::{
    Batch, BatchOutput, Batches, CallOutput, Inventory, Placement, RowState, Submission,
    SubmissionId,
};
use executor::completion::Outcome;
use ipc::{ComponentConfig, TimingCounters, WorkerEndpoint, WorkerInfo};

/// The token this pool samples for the `index`th call of every batch.
pub(crate) const SAMPLED: u32 = 1000;

/// A pool that serves every batch with the values the mechanism reads back.
pub(crate) struct Serving {
    inventory: Vec<Inventory>,
    /// Every batch the mechanism handed over, in submission order: what the
    /// mechanism asked for is half of what a case observes.
    submitted: Mutex<Vec<Batch>>,
    next: Mutex<u64>,
}

impl Serving {
    pub(crate) fn new(inventory: Vec<Inventory>) -> Self {
        Self {
            inventory,
            submitted: Mutex::new(Vec::new()),
            next: Mutex::new(1),
        }
    }

    pub(crate) fn submitted(&self) -> Vec<Batch> {
        self.submitted.lock().expect("the pool's batches").clone()
    }

    /// The batch's calls, each answered as a completed call that sampled
    /// [`SAMPLED`] and leaves the cache extent its row was stated with.
    ///
    /// The extents are the columns', not invented: `seq_lens` is a row's
    /// attention length, cached prefix plus its own query, which is exactly the
    /// cache the call leaves behind — so a prefill of eight prompt tokens
    /// leaves eight, and a decode of one token on top leaves nine. The
    /// mechanism's next call reads them to know what a successor may read.
    fn serve(&self, batch: Batch) -> Submission {
        let calls = batch
            .calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                let seen = batch
                    .columns
                    .rows
                    .seq_lens
                    .get(index)
                    .copied()
                    .unwrap_or(0)
                    .max(0) as u32;
                CallOutput::completed(
                    *call,
                    vec![SAMPLED + index as u32],
                    RowState {
                        // Text carries no denoising trajectory: the step index
                        // is a media row's, and this pool serves forwards.
                        step_index: 0,
                        cache_computed_len: seen,
                        cache_visible_len: seen,
                    },
                )
            })
            .collect::<Vec<_>>();
        let mut next = self.next.lock().expect("the pool's submissions");
        let submission = Submission::completed(
            SubmissionId(*next),
            Outcome::Success(BatchOutput {
                calls,
                timing: TimingCounters::default(),
            }),
        );
        *next += 1;
        submission
    }
}

impl Batches for Serving {
    fn submit(&self, batch: Batch) -> Result<Submission, executor::Error> {
        self.submitted
            .lock()
            .expect("the pool's batches")
            .push(batch.clone());
        Ok(self.serve(batch))
    }

    fn shadow(&self, batch: Batch) -> Result<Submission, executor::Error> {
        Ok(self.serve(batch))
    }

    fn inventory(&self) -> &[Inventory] {
        &self.inventory
    }
}

/// One rank of a text pool: prefill and decode, eight request rows, a prefix
/// cache, and the batch bounds a launch states.
pub(crate) fn rank(host: &str) -> WorkerInfo {
    WorkerInfo {
        endpoint: WorkerEndpoint {
            host: host.to_owned(),
            address_space: format!("{host}-space"),
            incarnation: format!("{host}-rank"),
        },
        supported_calls: vec![
            ipc::CallKind::Forward(ipc::ForwardMode::Prefill),
            ipc::CallKind::Forward(ipc::ForwardMode::Decode),
        ],
        request_slots: 8,
        max_batch_calls: 16,
        max_batch_tokens: 512,
        prefix_cache: None,
        buffer_pool_bytes: 1 << 20,
        ..WorkerInfo::default()
    }
}

/// A text pool of one rank in group `group`.
pub(crate) fn pool(group: &str) -> Inventory {
    Inventory {
        group: group.to_owned(),
        ranks: vec![rank("gpu-a")],
        placement: Placement {
            queue_depth: 1,
            ranks: 1,
            components: std::collections::BTreeMap::from([(
                ipc::DEFAULT_COMPONENT.to_owned(),
                ComponentConfig {
                    ranks: vec![0],
                    parallel_config: Default::default(),
                    units_per_rank: 1,
                    weights: Default::default(),
                    distribution: None,
                },
            )]),
        },
        max_model_len: 4096,
    }
}
