//! Schedulers over the one fake worker, `testkit::FakeBatches`, through the
//! production submission door (`Submissions::over_batches`).
//!
//! A case that drives a [`Scheduler`] directly builds it here; a case that
//! drives a launched epoch builds a `SchedulerHandle::over_batches` over the
//! same fake. Either way every completion travels the serving completion path
//! (`BatchReport::from_output`), and the fake is the only thing replaced.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use ipc::execution::{Batches, Inventory};
use scheduler::{BatchCapacity, Parameters, Scheduler, SchedulerConfig, Submissions, WorkerInfo};
use testkit::FakeBatches;

/// A text pool named `group` of one rank holding `rows` request rows, with
/// the batch bounds the sharing cases need: every row of a pass fits one
/// batch of calls and tokens.
pub(crate) fn text_pool(group: &str, rows: u32) -> Inventory {
    let mut pool = testkit::pool(group);
    for rank in &mut pool.ranks {
        rank.request_slots = rows;
        rank.max_batch_calls = rows.max(16);
        rank.max_batch_tokens = 8192;
    }
    pool
}

/// The prefix cache of every rank of `pool`, for a case that shapes it.
pub(crate) fn cache_of(pool: &mut Inventory) -> impl Iterator<Item = &mut ipc::PrefixCacheInfo> {
    pool.ranks.iter_mut().map(|rank| {
        rank.prefix_cache
            .as_mut()
            .expect("a text rank states a prefix cache")
    })
}

/// The rank `batches` states, as the scheduler reads it: what the policy's
/// ceilings are derived from.
pub(crate) fn rank_of(batches: &FakeBatches) -> ipc::MainProcessInfo {
    WorkerInfo::from_inventory(batches.inventory())
        .and_then(|info| info.runtime_info())
        .expect("the fake states a pool")
}

/// A scheduler over `batches` through the production door, with the model
/// limits its pools' call kinds imply (`super::generation_limits`), the
/// launches' control tokens (`testkit::special_tokens`), and `fcfs` installed
/// under the ceilings its pool advertises with `stated` beside them.
pub(crate) fn scheduler_over(
    batches: Arc<dyn Batches>,
    config: SchedulerConfig,
    stated: Parameters,
) -> anyhow::Result<Scheduler> {
    let info = WorkerInfo::from_inventory(batches.inventory())?.runtime_info()?;
    let limits = super::generation_limits(&info.supported_calls);
    let submissions = Submissions::over_batches(batches)?;
    let mut scheduler = Scheduler::new(submissions, testkit::special_tokens(), config, limits)?;
    super::install_policy_stating(&mut scheduler, BatchCapacity::of(&info), stated);
    Ok(scheduler)
}

/// The scheduler of [`scheduler_over`] running on a thread of its own, and
/// the command sender a case submits through.
pub(crate) fn spawn(
    scheduler: Scheduler,
) -> (scheduler::CommandSender, std::thread::JoinHandle<bool>) {
    let (tx, rx) = crossbeam_channel::unbounded();
    let thread = std::thread::spawn(move || scheduler.run(rx));
    (scheduler::CommandSender::new(tx), thread)
}

/// The token the fake samples for unseeded request `id` at logical
/// `position`, as a scheduler built here keys it: engine 1, the request's
/// first admission (request epoch 1). The token after a context of `n`
/// tokens is sampled at position `n`.
pub(crate) fn token_at(id: u64, position: u64) -> u32 {
    testkit::draw_token(ipc::philox::DrawAddress {
        key: ipc::philox::request_sampling_key(None, 1, id, 1),
        position,
    })
}

/// A device slower than its scheduler: every batch the fake accepts stays on
/// the device until this pacer completes it, one batch per `interval`, oldest
/// first. Batches the scheduler submits in one step are then in flight
/// together, which is what an occupancy rule bounds.
pub(crate) struct Pacer {
    fake: Arc<FakeBatches>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Pacer {
    /// Holds `fake`'s batches and completes one every `interval`.
    pub(crate) fn start(fake: &Arc<FakeBatches>, interval: Duration) -> Self {
        fake.hold();
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let fake = Arc::clone(fake);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(interval);
                    fake.release(1);
                }
            })
        };
        Self {
            fake: Arc::clone(fake),
            stop,
            thread: Some(thread),
        }
    }
}

impl Drop for Pacer {
    /// Completes every held batch and lets later ones complete at once.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        self.fake.resume();
    }
}
