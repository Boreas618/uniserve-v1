//! Drain (SD §8.1): leaving the engine (SD §3.8). A drain admits nothing; the requests that wait
//! leave at once as migratable, and those that run end by themselves or leave as migratable at
//! the deadline, a video request at a step boundary; the daemon resubmits a migrated request with
//! what it generated, and it continues its budget and its usage from the blocks it committed.
//! With `shrink` (a drain to an epoch of another layout), the scheduler gives its device memory
//! back by `Resize` as requests end, installing the whole of its eviction order, reports each
//! `Resized` as `Output::Shrunk`, and answers the drain once no request remains and every device
//! region answered a `Resize` taken after the last `Detach`.
//!
//! A drain's reply is the engine's signal that the epoch is drained; after it the scheduler forms
//! no batch until `Resume`. Every step of every case checks the scheduler's side of A1 to A4.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use engine::batch::cache::CacheGroupKind;
use engine::batch::{Batch, Phase};
use engine::scheduler::{Message, Output};
use proto::request::{Event, FinishReason, GenerationRequest, MigrationCause, RequestId};
use storage::Key;
use storage::runtime::{Completion, Request};

use crate::harness::{BLOCK, Deployment, Harness, VIDEO_PHASES, token, video};

/// Whether `event` is the `Migratable` of a drain with no stored step.
fn drained(event: Option<&Event>) -> bool {
    matches!(
        event,
        Some(Event::Migratable {
            cause: MigrationCause::Drain,
            denoised_steps: 0,
            stored_layout,
        }) if stored_layout.is_empty()
    )
}

/// Steps until `replied` holds the drain's reply, at most `steps` steps; returns whether it came.
fn replied_within(
    harness: &mut Harness,
    replied: &crossbeam_channel::Receiver<()>,
    steps: u64,
) -> bool {
    for _ in 0..steps {
        if replied.try_recv().is_ok() {
            return true;
        }
        harness.step();
    }
    replied.try_recv().is_ok()
}

/// A drain ends the waiting requests at once as migratable, and a request that arrives while it
/// drains at the next pass, none of them admitted; the running request keeps stepping until the
/// deadline and leaves then, once its call in flight returned. The drain is answered once no
/// request remains (SD §3.8 `drain`, `drain_due`).
#[test]
fn waiting_requests_leave_at_once_and_running_ones_at_the_deadline() {
    let mut harness = Deployment::tokens().slots(1).build();
    let running = harness.submit(token(1, 6, 1_000));
    let waiting = harness.submit(token(2, 6, 4));
    harness.run_until(20, |harness| harness.tokens(running).len() >= 2);
    assert!(harness.events(waiting).is_empty(), "one slot: it waits");

    let deadline = Instant::now() + Duration::from_millis(300);
    let replied = harness.drain(deadline, false);
    let events = harness.events(waiting);
    assert!(events.len() == 1 && drained(events.first()), "{events:?}");

    // Admission is closed: a request that arrives now leaves at the next pass.
    let late = harness.submit(token(3, 6, 4));
    harness.step();
    let events = harness.events(late);
    assert!(events.len() == 1 && drained(events.first()), "{events:?}");
    assert!(harness.batches_of(late).is_empty());

    // The running request runs on until the deadline.
    let generated = harness.tokens(running).len();
    while harness.terminal(running).is_none() {
        assert!(
            Instant::now() < deadline + Duration::from_secs(5),
            "the running request did not leave at its deadline: {}",
            harness.summary()
        );
        assert!(
            replied.try_recv().is_err(),
            "the drain was answered while a request ran"
        );
        harness.step();
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(Instant::now() >= deadline, "it left before the deadline");
    assert!(
        harness.tokens(running).len() > generated,
        "it stepped while the drain waited for its deadline"
    );
    assert!(drained(harness.events(running).last()));
    assert!(replied_within(&mut harness, &replied, 10));
    assert_eq!(harness.batches_of(waiting).len(), 0);
}

/// Running requests that end by themselves before the deadline end on this engine, as they would
/// have without the drain, and the drain is answered once the last of them ended.
#[test]
fn requests_that_end_before_the_deadline_end_on_this_engine() {
    let mut harness = Deployment::tokens().build();
    let ids = [
        harness.submit(token(1, 6, 8)),
        harness.submit(token(2, 10, 5)),
    ];
    harness.run_until(20, |harness| {
        ids.iter().all(|id| !harness.tokens(*id).is_empty())
    });

    let replied = harness.drain(Instant::now() + Duration::from_secs(60), false);
    for _ in 0..100 {
        if ids.iter().all(|id| harness.terminal(*id).is_some()) {
            break;
        }
        assert!(
            replied.try_recv().is_err(),
            "the drain was answered while a request ran"
        );
        harness.step();
    }
    for (id, max) in ids.into_iter().zip([8, 5]) {
        let Some(Event::Finished {
            reason,
            completion_tokens,
            ..
        }) = harness.terminal(id)
        else {
            panic!("request {} did not finish: {:?}", id.0, harness.events(id));
        };
        assert_eq!(reason, FinishReason::MaxTokens);
        assert_eq!(completion_tokens, max);
        assert_eq!(harness.tokens(id).len(), max);
    }
    assert!(replied_within(&mut harness, &replied, 10));
}

/// A video request whose denoising steps are on the device at the deadline is not cut inside a
/// step: no further step is formed, and it leaves as migratable once the steps in flight
/// returned. With no trajectory identity it stores nothing (SD §3.8, §7.3 "Preemption").
#[test]
fn a_video_request_at_the_deadline_leaves_at_its_step_boundary() {
    let mut harness = Deployment::video(&[&VIDEO_PHASES]).build();
    let id = harness.submit_video(video(1, 1));
    harness.denoise_and_hold(id, 1);
    assert!(harness.fake.held() > 0, "a denoising step is on the device");

    let steps = harness.media_rows(id, Phase::Denoise).len();
    let replied = harness.drain(Instant::now(), false);
    harness.run(5);
    assert!(
        harness.terminal(id).is_none(),
        "its steps in flight have not returned"
    );

    harness.fake.resume();
    assert!(replied_within(&mut harness, &replied, 20));
    assert!(
        drained(harness.events(id).last()),
        "{:?}",
        harness.events(id)
    );
    assert_eq!(
        harness.media_rows(id, Phase::Denoise).len(),
        steps,
        "no step was formed after the deadline"
    );
    let saved = harness
        .taken()
        .into_iter()
        .any(|request| matches!(request, Request::Save { .. }));
    assert!(!saved, "a request with no trajectory identity stores none");
}

/// The tokens a migrated request may generate in all.
const BUDGET: usize = 12;

/// Request 1 of a prompt of `prompt` tokens, under stop strings, which the daemon's decoder has
/// passed nothing of: it runs until its group's queue depth of outputs (two) waits for the
/// decoder, with no call in flight; a drain then ends it at once as migratable, its blocks holding
/// the prompt and its first generated token. Returns the request and the tokens it generated,
/// with admission open again (`Resume`).
fn migrated(harness: &mut Harness, prompt: u32) -> (GenerationRequest, Vec<u32>) {
    let mut request = token(1, prompt, BUDGET);
    request.stop_strings = vec!["end".to_owned()];
    let id = harness.submit(request.clone());
    harness.run(20);
    let generated = harness.tokens(id);
    assert_eq!(
        generated.len(),
        2,
        "the queue depth of outputs ahead of the daemon"
    );

    let replied = harness.drain(Instant::now(), false);
    assert!(replied_within(harness, &replied, 10));
    assert!(drained(harness.events(id).last()));

    // Its tails' `Commit`s and its `Detach` reach the buffer pool once the workers stamped them.
    harness.run_until(10, |harness| {
        let taken = harness.taken();
        taken
            .iter()
            .any(|request| matches!(request, Request::Detach { .. }))
    });
    harness.apply(Message::Resume);
    (request, generated)
}

/// The daemon resubmits `request` with the tokens it generated, and its decoder passes every
/// token as it is published, until the request finishes. Returns the batches formed meanwhile.
fn resubmitted(
    harness: &mut Harness,
    mut request: GenerationRequest,
    generated: Vec<u32>,
) -> Vec<Batch> {
    let id = request.request_id;
    let before = harness.batches().len();
    request.generated_token_ids = generated;
    harness.submit(request);
    harness.run_until(200, |harness| {
        let published = harness.tokens(id).len() as u32;
        harness.apply(Message::Acknowledge(id, published));
        matches!(harness.terminal(id), Some(Event::Finished { .. }))
    });
    harness.batches().split_off(before)
}

/// The resumed request `id` was not charged its generated tokens again, and its usage counts its
/// original prompt and every token of both lives; its first prefill wrote `positions`.
fn continued(harness: &Harness, id: RequestId, prompt: u32, batches: &[Batch], positions: &[i32]) {
    let Some(Event::Finished {
        reason,
        prompt_tokens,
        completion_tokens,
        ..
    }) = harness.terminal(id)
    else {
        panic!("{:?}", harness.events(id));
    };
    assert_eq!(reason, FinishReason::MaxTokens);
    assert_eq!(
        prompt_tokens, prompt as usize,
        "usage charges the original prompt"
    );
    assert_eq!(completion_tokens, BUDGET, "usage counts both lives");
    assert_eq!(
        harness.tokens(id).len(),
        BUDGET,
        "it generates what its budget had left"
    );

    let prefill = batches
        .iter()
        .find(|batch| batch.phase == Phase::Prefill && batch.requests.contains(&id))
        .unwrap();
    assert_eq!(prefill.columns.rows.cache_positions, positions);
}

/// A request that leaves as migratable is resubmitted by the daemon with the tokens it generated:
/// it is not charged them again, its usage counts the original prompt and every token of both
/// lives, and it reuses the blocks it committed, those of its generated tokens included, so that
/// its prefill computes only the last token of its context (SD §3.6, §4.4 "The hit", §7.2 "A
/// resumed token request"). Its blocks end at a block boundary: it pins them all.
#[test]
fn a_migrated_request_continues_its_budget_and_usage_from_its_committed_blocks() {
    // A prompt of seven tokens and one generated token written: two full blocks. The resubmitted
    // context is nine tokens, whose last its prefill computes.
    let mut harness = Deployment::tokens().build();
    let (request, generated) = migrated(&mut harness, 7);
    let id = request.request_id;
    let batches = resubmitted(&mut harness, request, generated);
    let pinned = harness
        .answers()
        .into_iter()
        .find_map(|answer| match answer {
            (Request::Pin { keys, .. }, Completion::Pinned { blocks, .. }) => {
                Some((keys.len(), blocks.len()))
            }
            _ => None,
        });
    assert_eq!(pinned, Some((2, 2)), "its admission pins its two blocks");
    continued(&harness, id, 7, &batches, &[8]);
}

/// A migrated request whose blocks end inside a block commits that partial block under its tail
/// key with the tokens it holds, and the resubmitted request clones it, so that its prefill
/// computes only the last token of its context (SD §3.7 `free`, §4.3 "tails", §4.4 "The hit").
#[test]
fn a_migrated_request_clones_its_committed_tail_and_continues_from_it() {
    // A prompt of six tokens and one generated token written: one full block and three tokens.
    // The resubmitted context is eight tokens, whose last its prefill computes.
    let mut harness = Deployment::tokens().build();
    let (request, generated) = migrated(&mut harness, 6);
    let id = request.request_id;
    let tails: Vec<(u32, u16)> = harness
        .taken()
        .into_iter()
        .filter_map(|request| match request {
            Request::Commit { at, tokens, .. } if u32::from(tokens) < BLOCK => Some((at, tokens)),
            _ => None,
        })
        .collect();
    assert_eq!(
        tails,
        [(1, 3)],
        "its partial last block is committed under its tail key"
    );

    let batches = resubmitted(&mut harness, request, generated);
    let cloned = harness.answers().into_iter().any(|answer| {
        matches!(
            answer,
            (Request::Finish { ok: true, .. }, Completion::Cloned)
        )
    });
    assert!(cloned, "its admission clones its tail");
    continued(&harness, id, 6, &batches, &[7]);
}

/// The committed keys whose `Commit` the buffer pool answered `Committed`.
fn committed(harness: &Harness) -> HashSet<Key> {
    let answers = harness.answers().into_iter();
    answers
        .filter_map(|answer| match answer {
            (Request::Commit { key, .. }, Completion::Committed) => Some(key),
            _ => None,
        })
        .collect()
}

/// A deployment where three requests committed ten full blocks each and ended: thirty blocks no
/// request holds, more than the head of the eviction order a step installs (the blocks one step's
/// tokens can take), and a fourth request running. Returns the committed keys and the running
/// request.
fn cached_and_running(harness: &mut Harness) -> (HashSet<Key>, RequestId) {
    for id in 1..=3 {
        harness.submit(token(id, 40, 2));
    }
    harness.settle(100);
    let cached = committed(harness);
    assert_eq!(cached.len(), 30);

    let running = harness.submit(token(4, 6, 30));
    harness.run_until(20, |harness| !harness.tokens(running).is_empty());
    (cached, running)
}

/// The keys of each `Victims` of region `region` among `requests`, in order.
fn installed(requests: &[Request], region: u8) -> Vec<Vec<Key>> {
    let victims = requests.iter().filter_map(|request| match request {
        Request::Victims { region: of, keys } if *of == region => Some(keys.clone()),
        _ => None,
    });
    victims.collect()
}

/// Steps until the drain is answered; returns how many completions the buffer pool had written
/// by then.
///
/// # Panics
///
/// When the drain is not answered within 200 steps.
fn answered_after(harness: &mut Harness, replied: &crossbeam_channel::Receiver<()>) -> usize {
    for _ in 0..200 {
        harness.step();
        if replied.try_recv().is_ok() {
            return harness.pool.completions().len();
        }
    }
    panic!("the drain was not answered: {}", harness.summary());
}

/// A drain to an epoch of another layout gives the device memory back as requests end (SD §3.8
/// "A drain to an epoch of another layout"): the whole eviction order is installed and the shrink
/// evicts it; each device region is asked to keep only what live requests hold, and again after
/// the running request's `Detach`; every `Resized` is reported as `Output::Shrunk` with the units
/// the region reached; and the drain is answered once no request remains and every device region
/// answered a `Resize` taken after the last `Detach`, by which time each region holds nothing.
#[test]
fn a_shrinking_drain_gives_the_device_memory_back_as_requests_end() {
    let mut harness = Deployment::tokens().build();
    let regions = ["kv-0", "slots-0", "buffers"].map(|name| harness.region(name));
    let kv = regions[0];
    let (cached, running) = cached_and_running(&mut harness);
    let heads = installed(&harness.taken(), kv);
    assert!(
        heads.iter().all(|keys| keys.len() < cached.len()),
        "a step installs the head of the order"
    );

    // At its deadline the running request leaves once its call in flight returned.
    let from = harness.taken().len();
    let replied = harness.drain(Instant::now(), true);
    let answered_at = answered_after(&mut harness, &replied);
    assert!(drained(harness.events(running).last()));

    // The whole order was installed, and the shrink evicted it.
    let whole = installed(&harness.taken()[from..], kv)
        .into_iter()
        .any(|keys| cached.iter().all(|key| keys.contains(key)));
    assert!(
        whole,
        "the whole order is installed while the drain shrinks"
    );
    let evicted: HashSet<Key> = harness
        .answers()
        .into_iter()
        .filter_map(|answer| match answer {
            (Request::Resize { .. }, Completion::Resized { evicted, .. }) => Some(evicted),
            _ => None,
        })
        .flatten()
        .collect();
    assert!(cached.is_subset(&evicted), "the shrink evicts the order");

    // Every `Resized` is reported, with the units the region reached.
    let resized: Vec<(u8, u32)> = harness
        .answers()
        .into_iter()
        .filter_map(|answer| match answer {
            (Request::Resize { region, .. }, Completion::Resized { units, .. }) => {
                Some((region, units))
            }
            _ => None,
        })
        .collect();
    let shrunk: Vec<(u8, u32)> = harness
        .outputs()
        .iter()
        .filter_map(|output| match output {
            Output::Shrunk(region, units) => Some((*region, *units)),
            _ => None,
        })
        .collect();
    assert_eq!(shrunk, resized);
    for region in regions {
        let last = shrunk.iter().rev().find(|(of, _)| *of == region);
        assert_eq!(
            last,
            Some(&(region, 0)),
            "region {region} gives back everything once no request remains"
        );
    }

    // The reply waited for a `Resize` of every device region taken after the last `Detach`, and
    // for its answer.
    let numbered = harness.pool.requests().to_vec();
    let last_detach = numbered
        .iter()
        .filter(|(_, _, request)| matches!(request, Request::Detach { .. }))
        .map(|(_, number, _)| number.0)
        .max()
        .unwrap();
    let answered: HashSet<u64> = harness.pool.completions()[..answered_at]
        .iter()
        .map(|(_, number, _)| number.0)
        .collect();
    for region in regions {
        let after = numbered.iter().any(|(_, number, request)| {
            matches!(request, Request::Resize { region: of, .. } if *of == region)
                && number.0 > last_detach
                && answered.contains(&number.0)
        });
        assert!(
            after,
            "region {region} answered no `Resize` taken after the last `Detach` before the reply"
        );
    }
}

/// A request running when a drain to an epoch of another layout begins steps on to its end, as
/// the shrink gives back only what no live request holds (SD §3.8: the serving scheduler keeps
/// stepping its running requests until they end or the deadline; §10 E9, E10 and E12: no request
/// fails for the shrink, and one that needs units waits for the capacity words). The drain is
/// answered once it ended.
#[test]
fn a_request_running_through_a_shrinking_drain_steps_on_to_its_end() {
    let mut harness = Deployment::tokens().build();
    let (_, running) = cached_and_running(&mut harness);
    let replied = harness.drain(Instant::now() + Duration::from_secs(60), true);
    answered_after(&mut harness, &replied);

    let Some(Event::Finished {
        reason,
        completion_tokens,
        ..
    }) = harness.terminal(running)
    else {
        panic!("{:?}", harness.events(running));
    };
    assert_eq!((reason, completion_tokens), (FinishReason::MaxTokens, 30));
}

/// A request running through a shrinking drain steps on to its end when the capacity words read
/// with a `Resized` predate it (SD §3.8): the Storage Server writes a turn's completions before
/// its capacity words, so a shrink's answer can arrive with words that still show free the units
/// it just took. What the shrink gives back stays within what the region holds beyond the blocks
/// the running request asked for and those it is still to take.
#[test]
fn a_shrinking_drain_keeps_what_a_running_request_takes_when_the_words_predate_its_resized() {
    // Blocks of sixteen tokens: the running request fills none, and so gives nothing back, while
    // the first shrink is answered.
    const WIDE_BLOCK: u32 = 16;
    const UNITS: u32 = 256;
    let mut harness = Deployment::tokens()
        .cache(&[(CacheGroupKind::Full, WIDE_BLOCK)])
        .kv_units(UNITS)
        .build();
    let kv = harness.region("kv-0");

    // One block of prompt and thirty tokens to decode: three blocks at its end and the low
    // watermark's, of which its admission asks for two.
    let running = harness.submit(token(1, 6, 30));
    harness.run_until(20, |harness| !harness.tokens(running).is_empty());

    // The drain asks for the request's four units; the next step writes the `Resize`, and the one
    // after it takes it, with words that still show free every unit but the request's two.
    let replied = harness.drain(Instant::now() + Duration::from_secs(60), true);
    harness.step();
    harness.pool.set_capacity(kv, Some((UNITS - 2, 0, 0)));
    harness.step();
    harness.pool.set_capacity(kv, None);
    let shrunk = harness.outputs().iter().any(
        |output| matches!(output, Output::Shrunk(region, units) if *region == kv && *units < UNITS),
    );
    assert!(shrunk, "the first shrink was answered");
    let generated = harness.tokens(running).len() as u32;
    assert!(6 + generated < WIDE_BLOCK, "it filled a block: {generated}");

    answered_after(&mut harness, &replied);
    let Some(Event::Finished {
        reason,
        completion_tokens,
        ..
    }) = harness.terminal(running)
    else {
        panic!("{:?}", harness.events(running));
    };
    assert_eq!((reason, completion_tokens), (FinishReason::MaxTokens, 30));
}
