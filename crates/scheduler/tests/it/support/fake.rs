//! Schedulers over the one fake worker, `testkit::FakeBatches`, through the
//! production submission door (`Submissions::over_batches`).
//!
//! A case that drives a [`Scheduler`] directly builds it here; a case that
//! drives a launched epoch builds a `SchedulerHandle::over_batches` over the
//! same fake. Either way every completion travels the serving completion path
//! (`BatchReport::from_output`), and the fake is the only thing replaced.

use std::sync::Arc;

use ipc::execution::{Batches, Inventory};
use scheduler::{
    GenerationFeatures, GenerationLimits, Parameters, Scheduler, SchedulerConfig, SpecialTokenIds,
    Submissions,
};
use testkit::FakeBatches;

/// The end of text the test launches state (`testkit::args`): the fake
/// samples in `SAMPLED..SAMPLED + 5000` and never reaches it, so a request
/// ends at its token budget unless a case answers it.
pub(crate) const EOS: u32 = 151_645;

/// The control tokens of the test launches: the BOS a prompt opens with and
/// the end of text a request stops at.
pub(crate) fn special_tokens() -> SpecialTokenIds {
    SpecialTokenIds {
        bos: 151_644,
        eos: vec![EOS, 151_643],
    }
}

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
    scheduler::WorkerInfo::from_inventory(batches.inventory())
        .and_then(|info| info.runtime_info())
        .expect("the fake states a pool")
}

/// A scheduler over `batches` that serves text with `config`, with `fcfs`
/// installed under the ceilings its pool advertises and `stated` beside them.
///
/// # Panics
///
/// Panics when the pool cannot be scheduled against or the policy does not
/// install: both are the case's setup, not the behaviour under test.
pub(crate) fn scheduler_over(
    batches: Arc<FakeBatches>,
    config: SchedulerConfig,
    stated: Parameters,
) -> Scheduler {
    let info = rank_of(&batches);
    let submissions = Submissions::over_batches(batches as Arc<dyn Batches>)
        .expect("the production door serves the fake");
    let limits = GenerationLimits {
        features: GenerationFeatures::UNDERSTANDING,
        ..Default::default()
    };
    let mut scheduler = Scheduler::new(submissions, special_tokens(), config, limits)
        .expect("the scheduler is built over the fake");
    super::install_policy_stating(&mut scheduler, scheduler::BatchCapacity::of(&info), stated);
    scheduler
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
