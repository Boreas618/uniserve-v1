//! The runner a block's phase is served by, and the registry the device executor holds.
//!
//! A runner serves one block's phases (specs/worker.md 4.5): it owns the captured graphs of those
//! phases, the arena their activations live in, and the slot state a token batch advances. It is
//! called only on the device executor's thread and only one job at a time, which is why the design
//! gives it `&mut self` and no synchronization.
//!
//! [`Runner`] is the design's trait verbatim. Its `precopy`, `switchover` and `purge` are the
//! runner's half of the program-layer update (7.3): precopy prepares the next generation's graphs
//! from the current one, switchover takes effect at a batch boundary, and purge releases the graphs
//! that still reference a replaced generation after its last batch.

use std::sync::Arc;

use ipc::PhaseId;
use ops::Generation;

use crate::Result;
use crate::batch::{Batch, Jobs};

/// The executor's name for one runner on a device.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct RunnerId(pub u32);

/// What the design calls a runner (4.5).
///
/// A runner serves the phases it names. `submit` turns one batch into the device work that serves
/// it, after staging the batch's host columns; `precopy` prepares the graphs the next generation
/// needs and returns the jobs that finish the preparation on the device; `switchover` and `purge`
/// are executed at batch boundaries by the device executor.
pub trait Runner: Send {
    /// The phases this runner serves, as the scheduler addresses them (4.5).
    fn phases(&self) -> &[PhaseId];

    /// Serves one batch: stage its inputs, choose its bucket, and return the device work.
    fn submit(&mut self, batch: Batch) -> Result<Jobs>;

    /// Captures every bucket of the serving generation that has no graph yet, largest first
    /// (`specs/replaceable-processes.md` 9.4 step 3: the executor captures before it serves).
    ///
    /// The returned jobs are the captures, each performed by this runner when the device executor
    /// reaches it, so a warmup interleaves with anything else the device is doing. A runner whose
    /// buckets are all captured returns no job.
    fn warmup(&mut self) -> Result<Jobs>;

    /// Prepares the next generation (7.3): rebind the buckets whose program change leaves the
    /// launch structure intact, recapture those whose structure changes, and return the jobs the
    /// device must run between serving batches to finish it.
    ///
    /// Precopy leaves the current generation serving. The runner holds both generations until
    /// `purge` releases the old one.
    fn precopy(&mut self, next: &Arc<Generation>) -> Result<Jobs>;

    /// Switches to `generation` at a batch boundary: from the next batch on, `submit` serves with
    /// its kernels and graphs. The runner keeps the previous generation for a revert (7.3).
    fn switchover(&mut self, generation: u64);

    /// Reverts a prepared generation: it never takes over, the serving generation stays, and its
    /// graphs are released (7.3 step 4).
    ///
    /// This is `switchover`'s reverse for an update that did not commit. Design 4.5's method list
    /// does not name it and a failed precopy cannot do without it, so it is a filled silence beside
    /// `serve` and `warmup`.
    fn revert(&mut self, generation: u64);

    /// Releases everything that references a generation older than `generation` (7.3): the graphs
    /// and the kernels they held. It runs after the last batch of the old generation completed.
    fn purge(&mut self, generation: u64);

    /// What this runner's graph pools hold and what they may hold (4.1, item 5's invariant).
    ///
    /// A runner without graph pools answers zeroes. The figures feed
    /// `FromEngine.metrics` through the executor's counters, so the delivery test can assert
    /// `mapped <= budget` while both generations' graphs exist.
    fn pool_metrics(&self) -> crate::arena::PoolMetrics {
        crate::arena::PoolMetrics::default()
    }

    /// Gives the runner the device's graph pools, before its first capture.
    ///
    /// One pool set exists per device and every runner issuing on it books into the same one
    /// (4.1): two runners each reserving their own range would double-book the device's graph
    /// storage. The device executor calls this when it starts.
    fn attach_pools(&mut self, pools: std::sync::Arc<std::sync::Mutex<crate::arena::Pools>>) {
        let _ = pools;
    }

    /// Performs one job this runner returned from `submit` or `precopy`, on the device executor's
    /// thread, when the executor reaches it in submission order.
    ///
    /// The design's trait has no such method and needs one: 4.5 has a recapture run "as a job on
    /// the executor, alternating with serving batches", and only the runner owns the model and the
    /// arena that a capture, a replay or a rebind needs. The executor cannot perform those jobs
    /// itself, so the runner that returned them performs them here.
    fn serve(&mut self, job: &crate::batch::Job) -> Result<crate::batch::BatchOutput>;
}

/// One runner the device executor owns: its name, the phases it serves, and the runner itself.
///
/// The device executor keeps these in [`RunnerId`] order and serves a batch through the runner
/// whose phase it names.
pub struct NamedRunner {
    /// The executor's name for this runner.
    pub id: RunnerId,
    /// The phases it serves, in the order its `phases` returns them.
    pub phases: Vec<PhaseId>,
    /// The runner itself.
    pub runner: Box<dyn Runner>,
}

impl NamedRunner {
    /// A named runner, taking the phase list from the runner itself.
    pub fn new(id: RunnerId, runner: Box<dyn Runner>) -> Self {
        let phases = runner.phases().to_vec();
        Self { id, phases, runner }
    }

    /// Whether this runner serves `phase`.
    pub fn serves(&self, phase: PhaseId) -> bool {
        self.phases.contains(&phase)
    }
}

impl std::fmt::Debug for NamedRunner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NamedRunner")
            .field("id", &self.id)
            .field("phases", &self.phases)
            .finish_non_exhaustive()
    }
}
