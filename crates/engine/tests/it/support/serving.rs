//! A pool that serves every batch in process, and the inventory it states.
//!
//! This is the `Batches` seam the handle drives when a deployment's ranks are
//! absent: `SchedulerHandle::over_batches` builds a full handle over it — pools,
//! rows and data parallelism are checked as they are for a launch — so a case
//! written against it states the mechanism's behaviour rather than a process's.

use std::sync::Mutex;

use engine::SchedulerConfig;
use proto::config::EngineArgs;

use ipc::{MainProcessEndpoint, MainProcessInfo, TimingCounters};
use proto::config::BlockConfig;
use worker::batch::{
    Batch, BatchOutput, Batches, CallOutput, Inventory, PoolLayout, RowState, Submission,
    SubmissionId,
};
use worker::completion::Outcome;

/// The token this pool samples for the `index`th call of every batch.
pub(crate) const SAMPLED: u32 = 1000;

/// A pool that serves every batch with the values the mechanism reads back.
pub(crate) struct Serving {
    inventory: Vec<Inventory>,
    /// Every batch the mechanism handed over, in submission order: what the
    /// mechanism asked for is half of what a case observes.
    submitted: Mutex<Vec<Batch>>,
    next: Mutex<u64>,
    /// Whether a call's token follows the cache extent its row states rather
    /// than its position in the batch.
    ///
    /// A positional token makes every batch look alike; a token that follows
    /// the extent is a function of the request's history as the mechanism
    /// states it, which is what a case about continuing a request needs: a
    /// continuation from three generated tokens samples what the uninterrupted
    /// request sampled fourth.
    by_extent: bool,
}

impl Serving {
    pub(crate) fn new(inventory: Vec<Inventory>) -> Self {
        Self {
            inventory,
            submitted: Mutex::new(Vec::new()),
            next: Mutex::new(1),
            by_extent: false,
        }
    }

    /// A pool whose sampled token follows each call's cache extent: the same
    /// request continued from a successor samples what it would have sampled
    /// uninterrupted.
    pub(crate) fn by_extent(inventory: Vec<Inventory>) -> Self {
        Self {
            by_extent: true,
            ..Self::new(inventory)
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
                    vec![if self.by_extent {
                        // A function of the extent, so a continuation samples
                        // what the uninterrupted request sampled next. The
                        // modulo is a vocabulary bound, not a matching device:
                        // ids stay inside the 5 000-wide window the simulated
                        // model's own tokens occupy (`sim_text_token`), which
                        // keeps the id a legal token. It cannot hide a
                        // successor that prefilled the prompt again: that
                        // difference is read from the batch's shape and row
                        // extent, never from the token.
                        SAMPLED + (seen % 5_000)
                    } else {
                        SAMPLED + index as u32
                    }],
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
                forward_stats: None,
            }),
        );
        *next += 1;
        submission
    }
}

impl Batches for Serving {
    fn submit(&self, batch: Batch) -> Result<Submission, worker::Error> {
        self.submitted
            .lock()
            .expect("the pool's batches")
            .push(batch.clone());
        Ok(self.serve(batch))
    }

    fn shadow(&self, batch: Batch) -> Result<Submission, worker::Error> {
        Ok(self.serve(batch))
    }

    fn inventory(&self) -> &[Inventory] {
        &self.inventory
    }
}

/// One rank of a text pool: prefill and decode, eight request rows, a prefix
/// cache, and the batch bounds a launch states.
pub(crate) fn rank(host: &str) -> MainProcessInfo {
    MainProcessInfo {
        endpoint: MainProcessEndpoint {
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
        // No `prefix_cache` here on purpose: `MainProcessInfo::default` states the
        // paged KV pool every token-serving rank advertises, and setting it to
        // `None` makes the engine refuse every generation request with
        // "generation request requires worker KV resources"
        // (`scheduler/admission.rs`). A media-only rank is the one that states
        // none, and that is not this pool.
        buffer_pool_bytes: 1 << 20,
        ..MainProcessInfo::default()
    }
}

/// A text pool of one rank in group `group`.
pub(crate) fn pool(group: &str) -> Inventory {
    Inventory {
        group: group.to_owned(),
        ranks: vec![rank("gpu-a")],
        layout: PoolLayout {
            queue_depth: 1,
            ranks: 1,
            components: std::collections::BTreeMap::from([(
                proto::config::DEFAULT_BLOCK.to_owned(),
                BlockConfig {
                    ranks: vec![0],
                    tensor_parallel_size: 1,
                    pipeline_parallel_size: 1,
                    sequence_parallel: Default::default(),
                    units_per_rank: 1,
                    weights: Default::default(),
                    distribution: None,
                },
            )]),
        },
        max_model_len: 4096,
    }
}

/// A configuration for a handle over `pool`: the simulation's own facts, with
/// the running-request count the pool can actually hold.
///
/// `SchedulerHandle::over_batches` refuses a config that asks for more running
/// requests than the ranks hold rows, plus any guidance prefix rows a denoising
/// rank needs. A test pool of one rank holds eight, and the sim configuration's
/// default is 128, so a case that leaves the default is refused at
/// construction. With the KV resources absent the same check is skipped
/// (`uses_cache` is false), which is why these two defects hidden each other:
/// the pool stripped its cache, and the row count it should have refused
/// silently passed. The row count is read from the pool, so there is one
/// authority for the number rather than a constant each case remembers.
pub(crate) fn config(pool: &Serving, name: &str) -> (EngineArgs, SchedulerConfig) {
    (args(name), limits(pool))
}

/// The simulation's arguments for a deployment named `name`.
pub(crate) fn args(name: &str) -> EngineArgs {
    engine::sim_args(name)
}

/// The mechanism's limits for `pool`: as many running requests as its ranks hold request rows.
pub(crate) fn limits(pool: &Serving) -> SchedulerConfig {
    let rows: usize = pool
        .inventory()
        .iter()
        .map(|inventory| {
            inventory
                .ranks
                .iter()
                .map(|rank| rank.request_slots as usize)
                .sum::<usize>()
        })
        .sum();
    SchedulerConfig {
        max_running_requests: rows.max(1),
        ..SchedulerConfig::default()
    }
}
