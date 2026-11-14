//! Preemption (SD §8.1): what a running request that cannot grow does, and what it takes from
//! others (SD §3.6, §7.2 "Preemption order", "Starvation bound", "Draft release").
//!
//! A running request is short when its next call writes a block it does not hold and the
//! region's `available` does not cover the `Allocate` for it, or the buffer pool answered that
//! `Allocate` `Full`. Then, in order: a request that can never fit ends with `CacheCapacity`; a
//! shortage that the region's retiring units cover waits for its `available` word and takes
//! nothing; otherwise draft tables are given up first, by idle requests with the fewest tokens
//! left, then admissions under way return to the waiting queue, then the policy names its
//! victims: unpinned cached blocks, evicted by key, and running requests, which end as
//! `Migratable { Drain }` with their blocks committed, tails included, and resume from them when
//! the daemon resubmits them. A running request that makes no progress within the policy's
//! timeout fails alone while other requests are served, and with nothing else to serve the
//! scheduler fails. The reports count what a shortage took (SD §7.5): the requests preempted,
//! the cached blocks evicted and the draft blocks given up.
//!
//! The capacity words a case sets (`InProcessPool::set_capacity`) state what the buffer pool
//! would state of a region under a load the case does not build: no unit available, or units
//! retiring.

use std::sync::Arc;
use std::time::Duration;

use engine::batch::Phase;
use engine::batch::cache::CacheGroupKind;
use engine::scheduler::{Message, policy};
use proto::report::SchedulerStats;
use proto::request::{Event, FinishReason, MigrationCause, RequestId};
use scheduling::{Nanos, Parameters};
use storage::runtime::Request;

use crate::harness::{A, BLOCK, Deployment, Harness, prompt, token};

/// Whether request `id`'s life ended as `Migratable { Drain }`, a preemption's end.
fn preempted(harness: &Harness, id: RequestId) -> bool {
    matches!(
        harness.terminal(id),
        Some(Event::Migratable {
            cause: MigrationCause::Drain,
            ..
        })
    )
}

/// Whether the buffer pool took an `Evict`.
fn evicted(harness: &Harness) -> bool {
    harness
        .taken()
        .iter()
        .any(|request| matches!(request, Request::Evict { .. }))
}

/// What the scheduler's reports counted of `field` over every report it sent, once its next
/// report arrived (it reports once a second; SD §7.5).
fn reported(harness: &mut Harness, field: fn(&SchedulerStats) -> u64) -> u64 {
    harness.next_report();
    harness.reports().into_iter().map(field).sum()
}

/// Steps until no step forms a batch for five steps in a row: every running request is left out
/// for want of blocks. At most `steps` steps.
///
/// # Panics
///
/// When batches are still formed after `steps` steps.
fn run_dry(harness: &mut Harness, steps: u64) {
    let mut idle = 0;
    for _ in 0..steps {
        idle = if harness.step() { 0 } else { idle + 1 };
        if idle >= 5 {
            return;
        }
    }
    panic!("batches are still formed: {}", harness.summary());
}

/// A request whose next call needs more blocks than the whole region holds ends with
/// `CacheCapacity` rather than taking victims or starving; the request queued behind it is then
/// admitted and served (SD §3.6, §7.2 "Preemption order": a request that can never fit).
#[test]
fn a_request_longer_than_the_cache_ends_with_cache_capacity_and_the_queue_goes_on() {
    let mut harness = Deployment::tokens().kv_units(4).slots(1).build();
    let long = harness.submit(token(1, 6, 100));
    let next = harness.submit(token(2, 6, 3));
    harness.settle(300);

    assert!(
        harness.finished_with(long, FinishReason::CacheCapacity),
        "{:?}",
        harness.events(long)
    );
    assert!(harness.finished_with(next, FinishReason::MaxTokens));
    assert_eq!(harness.tokens(next).len(), 3);
    assert!(!evicted(&harness), "no victim was taken");
}

/// A shortage that the region's retiring units cover takes no victim: the running requests run
/// out of the blocks they hold and wait for the region's `available` word, no request ends and
/// nothing is evicted; once the word shows the units, both ask again and run to their end (SD
/// §3.6; `specs/buffer-pool.md` §12.5).
#[test]
fn a_shortage_that_retiring_units_cover_waits_for_available_and_takes_no_victim() {
    // The region holds both requests to their ends.
    let mut harness = Deployment::tokens().kv_units(32).build();
    let kv = harness.region("kv-0");
    let ids = [
        harness.submit(token(1, 6, 24)),
        harness.submit(token(2, 6, 24)),
    ];
    harness.run_until(30, |harness| {
        ids.iter().all(|id| harness.tokens(*id).len() >= 2)
    });

    // Nothing available, and more units retiring than any shortage needs.
    harness.pool.set_capacity(kv, Some((0, 64, 0)));
    run_dry(&mut harness, 40);
    for id in ids {
        assert!(harness.terminal(id).is_none(), "{:?}", harness.events(id));
        assert!(harness.tokens(id).len() < 24, "it waits for blocks");
    }
    assert!(!evicted(&harness));
    let released = harness
        .taken()
        .iter()
        .any(|request| matches!(request, Request::Release { .. } | Request::Detach { .. }));
    assert!(!released, "no request gave anything up");

    harness.pool.set_capacity(kv, None);
    harness.settle(200);
    for id in ids {
        assert!(
            harness.finished_with(id, FinishReason::MaxTokens),
            "{:?}",
            harness.events(id)
        );
        assert_eq!(harness.tokens(id).len(), 24);
    }
}

/// Under a shortage, draft blocks are given up before any victim is taken: the idle running
/// request with the fewest tokens left gives up its whole draft table by one `Release` through
/// the executor, before anything is evicted, any request returns to the queue or ends, and
/// before the idle request with more tokens left gives up its own (SD §3.6, §4.4 "Drafts
/// first", §7.2 "Draft release"). Two requests are idle while the daemon's decoder has not
/// passed their outputs (stop strings); the third grows until the region is short.
#[test]
fn draft_blocks_are_given_up_first_by_the_idle_request_with_the_fewest_tokens_left() {
    let mut harness = Deployment::tokens().speculative(2).kv_units(32).build();
    let idle = |id: u64, max: usize| {
        let mut request = token(id, 8, max);
        request.stop_strings = vec!["end".to_owned()];
        request
    };
    let fewest = harness.submit(idle(1, 10));
    let more = harness.submit(idle(2, 30));
    let growing = harness.submit(token(3, 8, 36));
    harness.run_until(40, |harness| {
        harness.tokens(fewest).len() >= 2 && harness.tokens(more).len() >= 2
    });
    let owner = |harness: &Harness, id| harness.owner(id, A).unwrap();
    let (fewest_owner, more_owner) = (owner(&harness, fewest), owner(&harness, more));

    // The draft class is the model's last cache group: class 1.
    let first_relief = |harness: &Harness| {
        harness.taken().into_iter().find(|request| match request {
            Request::Release { class: 1, .. } | Request::Evict { .. } => true,
            Request::Detach { owner, .. } => *owner != fewest_owner && *owner != more_owner,
            _ => false,
        })
    };
    harness.run_until(400, |harness| first_relief(harness).is_some());
    let released = match first_relief(&harness).unwrap() {
        Request::Release {
            owner,
            class: 1,
            count,
            ..
        } if owner == fewest_owner => count,
        relief => panic!("{relief:?}"),
    };
    for id in [fewest, more, growing] {
        assert!(harness.terminal(id).is_none(), "{:?}", harness.events(id));
    }

    // The daemon passes their outputs: every request ends.
    harness.run_until(600, |harness| {
        for id in [fewest, more] {
            let published = harness.tokens(id).len() as u32;
            harness.apply(Message::Acknowledge(id, published));
        }
        [fewest, more, growing]
            .iter()
            .all(|id| harness.terminal(*id).is_some())
    });
    assert_eq!(harness.tokens(fewest).len(), 10);
    assert_eq!(harness.tokens(more).len(), 30);

    // The draft table given up is counted in the reports.
    let given_up = reported(&mut harness, |report| report.given_up_draft_blocks);
    assert!(
        released > 0 && given_up >= u64::from(released),
        "{given_up} of {released}"
    );
}

/// Under a shortage that no draft relieves, an admission under way returns to the waiting queue
/// before the policy is asked for victims: its owner is detached (its `Detach` handed to the
/// executor), nothing is evicted and no running request ends; the admission starts again later
/// with a new owner, and every request runs to its end (SD §3.6 `return_admissions`, §3.5
/// `Full`). The admission reads the prompt blocks of the request that grows, and its `Pinned` is
/// withheld, so that it holds the blocks of its `Allocated` while it waits; the withheld answer
/// is released once the admission has returned.
#[test]
fn an_admission_under_way_returns_to_the_queue_before_any_victim() {
    // The growing request alone needs fourteen of the region's sixteen units by its end; with
    // the admission's four, the region is short.
    let mut harness = Deployment::tokens().kv_units(16).build();
    let growing = harness.submit(token(2, 13, 40));
    harness.run_until(20, |harness| harness.tokens(growing).len() >= 2);

    // Request 3 reads request 2's three prompt blocks and allocates four of its own: its
    // `Pinned` is withheld, its `Allocated` not.
    harness
        .pool
        .withhold(|request, _| matches!(request, Request::Pin { .. }));
    let mut reading = token(3, 22, 3);
    reading.prompt_token_ids = prompt(2, 12).into_iter().chain(prompt(3, 10)).collect();
    let admitting = harness.submit(reading);
    harness.run_until(10, |harness| harness.pool.withheld() == 1);
    harness.pool.withhold(|_, _| false);
    let first_owner = harness
        .taken()
        .into_iter()
        .rev()
        .find_map(|request| match request {
            Request::Attach { owner, .. } => Some(owner),
            _ => None,
        })
        .unwrap();

    let returned = |harness: &Harness| {
        let stamps = harness.fake.stamps();
        stamps.into_iter().any(|(_, request)| match request {
            Request::Detach { owner, .. } => owner == first_owner,
            _ => false,
        })
    };
    harness.run_until(200, |harness| returned(harness));
    assert!(harness.terminal(growing).is_none());
    assert!(!evicted(&harness), "the admission went before any victim");
    assert!(harness.events(admitting).is_empty(), "it waits again");

    harness.pool.release();
    harness.settle(400);
    assert_eq!(harness.tokens(admitting).len(), 3);
    assert_eq!(harness.tokens(growing).len(), 40);
    assert!(!preempted(&harness, growing));
    assert!(!evicted(&harness));
    let detached = harness
        .taken()
        .iter()
        .any(|request| matches!(request, Request::Detach { owner, .. } if *owner == first_owner));
    assert!(detached);
    let attached = harness
        .taken()
        .iter()
        .filter(|request| matches!(request, Request::Attach { .. }))
        .count();
    assert!(
        attached >= 3,
        "request 3 was admitted again, with a new owner"
    );
}

/// When nothing else relieves a shortage, the policy's victims are taken: the `fcfs` policy
/// names the least recently used cached blocks no request holds, which the scheduler evicts by
/// their keys, the deepest block of the request that freed them first; no request ends (SD
/// §3.6; the `fcfs` policy's `preempt`; §4.4 "The eviction order").
#[test]
fn the_policys_victim_blocks_are_evicted_by_key() {
    let mut harness = Deployment::tokens().build();
    let kv = harness.region("kv-0");
    // Its prompt's three blocks are committed while it decodes, and freed when it ends.
    let cached = harness.submit(token(1, 13, 3));
    harness.settle(100);
    let keys = harness.committed(harness.owner(cached, A).unwrap());
    assert_eq!(keys.len(), 3);
    let growing = harness.submit(token(2, 5, 30));
    harness.run_until(20, |harness| harness.tokens(growing).len() >= 2);

    // The capacity words state no unit available: the cached blocks are all that can go.
    harness.pool.set_capacity(kv, Some((0, 0, 0)));
    harness.run_until(40, |harness| evicted(harness));
    let victims = harness
        .taken()
        .into_iter()
        .find_map(|request| match request {
            Request::Evict { keys } => Some(keys),
            _ => None,
        })
        .unwrap();
    assert!(!victims.is_empty());
    assert!(victims.iter().all(|key| keys.contains(key)), "{victims:?}");
    assert_eq!(
        victims[0], keys[2],
        "the deepest block first: {victims:?} of {keys:?}"
    );
    assert!(harness.terminal(growing).is_none());

    harness.pool.set_capacity(kv, None);
    harness.settle(200);
    assert!(harness.finished_with(growing, FinishReason::MaxTokens));
    assert_eq!(harness.tokens(growing).len(), 30);

    // Every victim the buffer pool evicted is counted, and no request was preempted.
    let evicted = reported(&mut harness, |report| report.evicted_blocks);
    assert!(evicted >= victims.len() as u64, "{evicted} of {victims:?}");
    assert_eq!(
        reported(&mut harness, |report| report.preempted_requests),
        0
    );
}

/// When no cached block is free, the policy's victim is a running request: the `fcfs` policy
/// names the one admitted last, which ends as `Migratable { Drain }` with its blocks committed,
/// its partial last block under its tail key; resubmitted by the daemon with what it generated,
/// it is admitted ahead of new requests, reads its full blocks and its tail, computes from where
/// it stopped and generates what it had left (SD §3.6, §3.7 `free`, §4.3 tails, §4.4 "Pins,
/// loads, clones and prefetches").
#[test]
fn the_policys_victim_request_ends_as_migratable_and_resumes_from_its_committed_blocks() {
    let mut harness = Deployment::tokens().build();
    let kv = harness.region("kv-0");
    let kept = harness.submit(token(1, 6, 30));
    let victim = harness.submit(token(2, 7, 30));
    harness.run_until(30, |harness| {
        harness.tokens(kept).len() >= 3 && harness.tokens(victim).len() >= 3
    });

    // No unit available, and no cached block free: the policy names a request.
    harness.pool.set_capacity(kv, Some((0, 0, 0)));
    harness.run_until(40, |harness| harness.terminal(victim).is_some());
    assert!(preempted(&harness, victim), "{:?}", harness.events(victim));
    assert!(harness.terminal(kept).is_none());
    let generated = harness.tokens(victim);
    let owner = harness.owner(victim, A).unwrap();
    // Its commits reach the buffer pool ahead of its owner's `Detach`.
    harness.run_until(10, |harness| {
        harness
            .taken()
            .iter()
            .any(|request| matches!(request, Request::Detach { owner: of, .. } if *of == owner))
    });
    // What it reached: its prompt and every token it generated but the last, whose call is the
    // resumed request's first.
    let computed = 7 + generated.len() as u32 - 1;
    let tails: Vec<u16> = harness
        .taken()
        .into_iter()
        .filter_map(|request| match request {
            Request::Commit {
                owner: of, tokens, ..
            } if of == owner => Some(tokens),
            _ => None,
        })
        .collect();
    assert_eq!(
        tails.last().copied(),
        Some((computed % BLOCK) as u16),
        "its partial last block is committed under its tail"
    );
    harness.pool.set_capacity(kv, None);

    // The daemon resubmits it with what it generated; a new request arrives after it.
    let mut resumed = token(2, 7, 30);
    resumed.generated_token_ids = generated.clone();
    let before = harness.batches().len();
    harness.submit(resumed);
    let newer = harness.submit(token(3, 6, 2));
    harness.settle(300);

    let prefill = harness
        .batches()
        .split_off(before)
        .into_iter()
        .find(|batch| batch.phase == Phase::Prefill && batch.requests.contains(&victim));
    let first = prefill.unwrap().columns.rows.cache_positions[0];
    assert_eq!(first as u32, computed, "it computes from where it stopped");
    let cloned = harness
        .taken()
        .iter()
        .any(|request| matches!(request, Request::Clone { .. }));
    assert!(cloned, "its tail is cloned");
    assert_eq!(
        harness.tokens(victim).len(),
        30,
        "both lives generated its budget"
    );
    assert!(harness.finished_with(victim, FinishReason::MaxTokens));
    assert_eq!(harness.tokens(newer).len(), 2);
    assert_eq!(harness.tokens(kept).len(), 30);

    // The one preemption is counted.
    assert_eq!(
        reported(&mut harness, |report| report.preempted_requests),
        1
    );
}

/// When every running request is short at once, the one of the greatest priority value is
/// preempted, and among equal priorities the one admitted last: a request of priority 1 goes
/// before an equal later one and before an earlier one of priority 0 (SD §3.6, §7.2 "Preemption
/// order").
#[test]
fn a_mutual_shortage_preempts_the_last_admitted_of_the_lowest_priority() {
    let mut harness = Deployment::tokens().build();
    let kv = harness.region("kv-0");
    let mut low = token(1, 6, 30);
    low.priority = 1;
    let low = harness.submit(low);
    let high = harness.submit(token(2, 6, 30));
    harness.run_until(30, |harness| {
        harness.tokens(low).len() >= 3 && harness.tokens(high).len() >= 3
    });

    harness.pool.set_capacity(kv, Some((0, 0, 0)));
    harness.run_until(40, |harness| harness.terminal(low).is_some());
    assert!(preempted(&harness, low), "{:?}", harness.events(low));
    assert!(harness.terminal(high).is_none());

    harness.pool.set_capacity(kv, None);
    harness.settle(200);
    assert!(harness.finished_with(high, FinishReason::MaxTokens));
    assert_eq!(harness.tokens(high).len(), 30);
    assert_eq!(
        reported(&mut harness, |report| report.preempted_requests),
        1
    );
}

/// Serves the `fcfs` policy again with a starvation bound of `timeout`, the deployment's other
/// limits as `Deployment::tokens` states them (chunks and steps of 64 tokens, 8 calls a step).
fn starve_after(harness: &mut Harness, timeout: Duration) {
    let module = Arc::new(policy::load(&testkit::packaged_module()).unwrap());
    let parameters = Parameters {
        chunked_prefill_size: Some(64),
        max_batch: Some(8),
        max_num_batched_tokens: Some(64),
        timeout: Some(Nanos(u64::try_from(timeout.as_nanos()).unwrap())),
        ..Parameters::default()
    };
    let (reply, replied) = crossbeam_channel::bounded(1);
    harness.apply(Message::SetPolicy(module, parameters, reply));
    assert!(matches!(replied.try_recv(), Ok(Ok(()))));
    harness.apply(Message::CommitPolicy);
}

/// Withholds the answers to every later `Allocate` of request `id`'s owner, once it ran a batch:
/// it runs out of blocks within its low watermark and is left out from then on.
fn withhold_allocations(harness: &mut Harness, id: RequestId) {
    harness.run_until(20, |harness| harness.tokens(id).len() >= 2);
    let owner = harness.owner(id, A).unwrap();
    harness.pool.withhold(
        move |request, _| matches!(request, Request::Allocate { owner: of, .. } if *of == owner),
    );
}

/// A running request that makes no progress within the policy's timeout ends with `Error` while
/// another request is served, which runs on to its end (SD §3.6 `fail_starved`, §7.2 "Starvation
/// bound").
#[test]
fn a_starved_request_fails_alone_while_another_is_served() {
    let mut harness = Deployment::tokens().build();
    starve_after(&mut harness, Duration::from_millis(300));
    let starved = harness.submit(token(1, 5, 40));
    let served = harness.submit(token(2, 5, 50));
    withhold_allocations(&mut harness, starved);

    // Steps 20 ms apart: the served request decodes past the bound.
    harness.run_until(40, |harness| {
        std::thread::sleep(Duration::from_millis(20));
        harness.terminal(starved).is_some()
    });
    assert!(harness.finished_with(starved, FinishReason::Error));
    assert!(harness.terminal(served).is_none(), "the other is served on");
    assert_eq!(harness.failure(), None);

    harness.pool.release();
    harness.settle(400);
    assert_eq!(harness.tokens(served).len(), 50);
}

/// A running request that makes no progress within the policy's timeout, with no other request
/// to serve, means the scheduler forms nothing for anything it runs: it fails, and the request
/// leaves as `Migratable { Loss }` (SD §3.6 `fail_starved`, §7.2 "Starvation bound").
#[test]
fn a_starved_request_with_nothing_else_to_serve_fails_the_scheduler() {
    let mut harness = Deployment::tokens().build();
    starve_after(&mut harness, Duration::from_millis(300));
    let starved = harness.submit(token(1, 5, 40));
    withhold_allocations(&mut harness, starved);

    harness.run_until(200, |harness| {
        std::thread::sleep(Duration::from_millis(10));
        harness.failure().is_some()
    });
    let failure = harness.failure().unwrap();
    assert!(
        failure.contains("no batch formed for 1 running request(s) at step "),
        "{failure}"
    );
    assert!(matches!(
        harness.terminal(starved),
        Some(Event::Migratable {
            cause: MigrationCause::Loss,
            ..
        })
    ));
}

/// The daemon's part for request `id` between two steps: its consumer takes every event it was
/// sent (`taken` counts those taken before), and a request that left as migratable since
/// (`lives` counts its lives that did) is resubmitted with what it generated, as `request`.
/// Returns the reason its life finished with, once it ended with `Finished`.
fn serve(
    harness: &mut Harness,
    id: RequestId,
    request: impl Fn() -> proto::request::GenerationRequest,
    taken: &mut usize,
    lives: &mut usize,
) -> Option<FinishReason> {
    let events = harness.events(id);
    harness.apply(Message::Consumed(id, (events.len() - *taken) as u32));
    *taken = events.len();
    let left = events
        .iter()
        .filter(|event| matches!(event, Event::Migratable { .. }))
        .count();
    if left > *lives {
        *lives = left;
        let mut resumed = request();
        resumed.generated_token_ids = harness.tokens(id);
        harness.submit(resumed);
    }
    match events.last() {
        Some(Event::Finished { reason, .. }) => Some(reason.clone()),
        _ => None,
    }
}

/// Requests that outgrow the cache while they decode give units up by preemption and get them
/// back when the daemon resubmits them, and once every one has ended none of the region's units
/// stays held: a request alone then writes every position of the region's blocks and ends with
/// `CacheCapacity` only when its next call needs a block beyond them (SD §3.6, §7.2 "Preemption
/// order"; `FinishReason::CacheCapacity`).
#[test]
fn after_requests_that_outgrew_the_cache_a_request_alone_fills_every_unit() {
    const UNITS: u32 = 32;
    let mut harness = Deployment::tokens().kv_units(UNITS).build();

    // Eight requests of twelve blocks each by their ends, three times what the region holds.
    let ids: Vec<RequestId> = (1..=8).map(|id| harness.submit(token(id, 6, 40))).collect();
    let mut taken = vec![0; ids.len()];
    let mut lives = vec![0; ids.len()];
    harness.run_until(4_000, |harness| {
        for (index, &id) in ids.iter().enumerate() {
            let request = || token(id.0, 6, 40);
            serve(harness, id, request, &mut taken[index], &mut lives[index]);
        }
        ids.iter()
            .all(|id| matches!(harness.terminal(*id), Some(Event::Finished { .. })))
    });
    for &id in &ids {
        assert!(harness.finished_with(id, FinishReason::MaxTokens));
        assert_eq!(harness.tokens(id).len(), 40, "every life together");
    }
    assert!(
        lives.iter().any(|lived| *lived > 0),
        "the region was outgrown"
    );

    let alone = harness.submit(token(100, 6, 1_000));
    let (mut taken, mut lives) = (0, 0);
    harness.run_until(1_000, |harness| {
        let request = || token(100, 6, 1_000);
        serve(harness, alone, request, &mut taken, &mut lives);
        harness.terminal(alone).is_some()
    });
    let written = harness
        .batches_of(alone)
        .iter()
        .flat_map(|batch| batch.columns.rows.cache_positions.clone())
        .max();
    assert_eq!(written, Some((UNITS * BLOCK) as i32 - 1));
    assert!(harness.finished_with(alone, FinishReason::CacheCapacity));
}

/// Serves request ids `1..=count`, each built by `request` and generating `tokens` tokens, as the
/// daemon serves them (every event taken, every request that left as migratable resubmitted with
/// what it generated, [`serve`]), until each has ended at its length; the load preempts.
///
/// # Panics
///
/// When the scheduler fails, when a step forms no batch for two thousand steps in a row while
/// requests remain (every batch the load needs forms within a few thousand steps), or when a
/// request ends otherwise.
fn serve_to_their_length(
    harness: &mut Harness,
    count: u64,
    request: impl Fn(u64) -> proto::request::GenerationRequest,
    tokens: usize,
) {
    let ids: Vec<RequestId> = (1..=count).map(|id| harness.submit(request(id))).collect();
    let mut taken = vec![0; ids.len()];
    let mut lives = vec![0; ids.len()];
    let mut ended = vec![None; ids.len()];

    let mut idle = 0;
    while ended.iter().any(Option::is_none) {
        idle = if harness.step() { 0 } else { idle + 1 };
        for (index, &id) in ids.iter().enumerate() {
            // A request that finished takes no further event.
            if ended[index].is_none() {
                let request = || request(id.0);
                ended[index] = serve(harness, id, request, &mut taken[index], &mut lives[index]);
            }
        }
        assert_eq!(harness.failure(), None, "the scheduler failed");
        assert!(idle < 2_000, "no batch formed: {}", harness.summary());
    }
    assert!(
        lives.iter().sum::<usize>() > 0,
        "the load preempted no request"
    );
    for (&id, reason) in ids.iter().zip(ended) {
        assert_eq!(reason, Some(FinishReason::MaxTokens), "request {}", id.0);
        assert_eq!(harness.tokens(id).len(), tokens, "request {}", id.0);
    }
}

/// Many more requests than the cache holds all end at their length: 64 requests over 24 prompts,
/// each of 24 blocks by its end, beside a device region of 64 units and a host region of 128, 32
/// slots. Preemptions return admissions under way to the queue while their reads are held, so
/// their requests are written for an owner already detached, whose answers the scheduler cannot
/// bound; a `Load` among them answered `Copy` keeps owing the whole completion ring until the
/// answer to its copy's `Finish`, and that `Finish`, which takes no room of its own, is still
/// written (SD §3.1, §3.6).
#[test]
fn requests_that_outgrow_the_cache_many_times_over_all_end_at_their_length() {
    let mut harness = Deployment::tokens()
        .kv_units(64)
        .host(128)
        .slots(32)
        .build();
    let request = |id: u64| crate::harness::sharing(id, 1000 + id % 24, 32, 64);
    serve_to_their_length(&mut harness, 64, request, 64);
}

/// Requests resumed with many blocks cached all end at their length beside the requests that
/// run short: 96 requests over 48 prompts, each of 40 blocks by its end, beside a device region of
/// 96 units and a host region of 192, 64 slots. A preempted request comes back with a hit as large
/// as what the region has left besides the running requests' blocks. Those blocks are held by no
/// request, so the region's `available` counts them as evictable, but its admission's `Pin` takes
/// them from every allocation, its own `Allocate` included: an admission is tried only once
/// `available` covers its hit's unheld blocks as well as what it allocates, and a `Pin` not yet
/// answered keeps them out of `available`. A running request short of a unit therefore gets one
/// once `available` shows it, and is never answered `Full` step after step until its starvation
/// bound ends it (SD §3.5 `fits`, §3.6, §4.5).
#[test]
fn requests_resumed_with_their_blocks_cached_all_end_at_their_length() {
    let mut harness = Deployment::tokens()
        .kv_units(96)
        .host(192)
        .slots(64)
        .build();
    let request = |id: u64| crate::harness::sharing(id, 1000 + id % 48, 32, 128);
    serve_to_their_length(&mut harness, 96, request, 128);
}

/// Requests of a hybrid model, whose recurrent-state checkpoint takes many units where a page of
/// attention takes one, all end at their length under memory pressure: 64 requests over 48
/// prompts beside a device region of 256 pages and 32 checkpoints, a host region of 192 units and
/// 32 slots. Admissions that share a prompt load its blocks from the host, and a `Load` whose
/// leading keys another admission brought to the device meanwhile is answered `Loaded` with
/// nothing copied and no entry given (BP §11): its positions are missing, like those of a short
/// pin, so the request neither restores a checkpoint nor releases a block it does not hold
/// (SD §3.5 `answered`, §4.4 "The hit").
#[test]
fn hybrid_requests_whose_checkpoints_take_many_units_all_end_at_their_length() {
    let mut harness = Deployment::tokens()
        .cache(&[
            (CacheGroupKind::Full, BLOCK),
            (CacheGroupKind::State, BLOCK),
        ])
        .block_units(&[1, 99])
        .kv_units(256 + 32 * 99)
        .host(192)
        .slots(32)
        .build();
    let request = |id: u64| crate::harness::sharing(id, 1000 + id % 48, 32, 64);
    serve_to_their_length(&mut harness, 64, request, 64);
}
