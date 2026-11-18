//! Limits (SD §8.1): what bounds a request's admission and a step's work (SD §3.2, §3.4, §3.5,
//! §7.1). The waiting queue's bound; the running requests the scheduler admits, and the slots a
//! group holds for them; the calls and the tokens one step charges; the blocks a call must hold
//! before it is scheduled; the batches a group holds in flight; the consumer's window of events,
//! which stops a request whose consumer is slow and still lets its terminal through; and the
//! limits `Configure` changes in place.
//!
//! The window's numbers are SD §7.1's: 64 events per request, a reserve of 2 for the terminal,
//! and a bound of 4 events per prefill or decode in flight and per call scheduled. Every event a
//! request is sent takes one credit, its `Scheduled` included, so a request whose consumer takes
//! nothing is scheduled a decode while `63 - t >= 2 + 4` after `t` tokens: it stops at 58.

use std::num::NonZeroUsize;

use engine::batch::{Batch, Phase};
use engine::scheduler::Message;
use proto::request::{Event, FinishReason, RejectionKind, RequestId};
use storage::runtime::Request;

use crate::harness::{A, BLOCK, Deployment, Harness, row_positions, token};

/// The tokens a request whose consumer takes no event is sent before its window stops it (module
/// documentation).
const UNCONSUMED: usize = 58;

/// The batches one step formed.
fn stepped(harness: &mut Harness) -> Vec<Batch> {
    let before = harness.batches().len();
    harness.step();
    harness.batches().split_off(before)
}

/// Whether request `id` left the waiting queue: its `Scheduled` was sent.
fn scheduled(harness: &Harness, id: RequestId) -> bool {
    let events = harness.events(id);
    events
        .iter()
        .any(|event| matches!(event, Event::Scheduled { .. }))
}

/// The requests of `ids` that left the waiting queue and have not ended.
fn live(harness: &Harness, ids: &[RequestId]) -> usize {
    let running = ids
        .iter()
        .filter(|id| scheduled(harness, **id) && harness.terminal(**id).is_none());
    running.count()
}

/// The rejection request `id` was sent as its only event, if it was.
fn rejection(harness: &Harness, id: RequestId) -> Option<(RejectionKind, String)> {
    match &harness.events(id)[..] {
        [Event::Rejected { kind, message }] => Some((*kind, message.clone())),
        _ => None,
    }
}

/// The `Finished` of a token request of `prompt_tokens` that generated `completion_tokens`.
fn finished(reason: FinishReason, prompt_tokens: usize, completion_tokens: usize) -> String {
    let event = Event::Finished {
        reason,
        stop_reason: None,
        prompt_tokens,
        completion_tokens,
    };
    format!("{event:?}")
}

/// The cache positions request `id`'s rows wrote, over its batches in order.
fn written(harness: &Harness, id: RequestId) -> Vec<i32> {
    let batches = harness.batches_of(id);
    batches
        .iter()
        .flat_map(|batch| row_positions(batch, id))
        .collect()
}

/// The waiting queue's bound counts the requests that wait, not those that run, and a request
/// that leaves the queue makes room for one more (SD §3.2 `queue_full`: `waiting.len() >=
/// max_queued_requests`; §7.2 "Rejection messages").
#[test]
fn the_queue_bound_counts_the_waiting_requests_and_frees_as_they_are_admitted() {
    let mut harness = Deployment::tokens().slots(1).max_queued(2).build();
    let running = harness.submit(token(1, 6, 4));
    harness.run_until(10, |harness| !harness.tokens(running).is_empty());

    // One request runs and two wait: the queue is at its bound.
    let first = harness.submit(token(2, 6, 4));
    let second = harness.submit(token(3, 6, 4));
    let refused = harness.submit(token(4, 6, 4));
    let overloaded = Some((
        RejectionKind::Overloaded,
        "scheduler waiting queue is full".to_owned(),
    ));
    assert_eq!(rejection(&harness, refused), overloaded);
    assert!(harness.events(first).is_empty());
    assert!(harness.events(second).is_empty());

    // Once the first waiting request is admitted, the queue takes one more, and only one.
    harness.run_until(100, |harness| scheduled(harness, first));
    assert!(!scheduled(&harness, second));
    let taken = harness.submit(token(5, 6, 4));
    let refused = harness.submit(token(6, 6, 4));
    assert!(harness.events(taken).is_empty(), "it waits");
    assert_eq!(rejection(&harness, refused), overloaded);

    harness.settle(300);
    for id in [running, first, second, taken] {
        assert_eq!(harness.tokens(id).len(), 4, "{}", harness.summary());
    }
}

/// No more requests run at once than the groups hold slots for: `max_running_requests` is the
/// slots of the deployment (SD §7.1 "Limits resolved at construction"), and an admission waits
/// while that many run (SD §3.5 `admit`).
#[test]
fn no_more_requests_run_at_once_than_the_groups_hold_slots_for() {
    let mut harness = Deployment::tokens().slots(2).build();
    let ids: Vec<RequestId> = (1..=5).map(|id| harness.submit(token(id, 6, 5))).collect();

    let mut most = 0;
    for _ in 0..300 {
        if ids.iter().all(|id| harness.terminal(*id).is_some()) {
            break;
        }
        harness.step();
        let running = live(&harness, &ids);
        assert!(running <= 2, "{running} requests run at once");
        most = most.max(running);
    }
    assert_eq!(most, 2, "the bound is reached");
    harness.settle(100);
    for id in &ids {
        assert_eq!(harness.tokens(*id).len(), 5);
    }
}

/// A step charges at most `max_batch` calls over all its batches, and a batch holds at most the
/// rows of the largest bucket (SD §3.4 `charge`: the step's budget of calls; §7.1 "Rows of a
/// batch").
#[test]
fn a_step_charges_at_most_max_batch_calls() {
    let mut harness = Deployment::tokens().batch(2, 64).build();
    let ids: Vec<RequestId> = (1..=5).map(|id| harness.submit(token(id, 6, 6))).collect();

    let mut fullest = 0;
    for _ in 0..300 {
        if ids.iter().all(|id| harness.terminal(*id).is_some()) {
            break;
        }
        let formed = stepped(&mut harness);
        let calls: usize = formed.iter().map(|batch| batch.requests.len()).sum();
        assert!(calls <= 2, "a step charged {calls} calls");
        for batch in &formed {
            assert!(batch.shape.rows <= 2, "{:?}", batch.shape);
        }
        fullest = fullest.max(calls);
    }
    assert_eq!(fullest, 2, "the bound is reached");
    harness.settle(100);
    for id in &ids {
        assert_eq!(harness.tokens(*id).len(), 6);
    }
}

/// A step charges at most `max_num_batched_tokens` tokens over all its batches: a prefill chunk
/// is cut to the tokens the step has left, and its prompt's next chunk continues it (SD §3.4
/// `charge`; §7.1 "A prefill chunk": `min(prompt left, chunk limit, budget)`).
#[test]
fn a_step_charges_at_most_max_num_batched_tokens() {
    let mut harness = Deployment::tokens().batch(8, 16).chunk(16).build();
    let first = harness.submit(token(1, 12, 3));
    let second = harness.submit(token(2, 12, 3));

    let mut steps = Vec::new();
    for _ in 0..100 {
        if [first, second]
            .iter()
            .all(|id| harness.terminal(*id).is_some())
        {
            break;
        }
        steps.push(stepped(&mut harness));
    }
    for formed in &steps {
        let tokens: usize = formed.iter().map(|batch| batch.columns.tokens()).sum();
        assert!(tokens <= 16, "a step charged {tokens} tokens");
    }

    // The first prefill step takes request 1's prompt whole and the four tokens it leaves of
    // request 2's; request 2's next chunk takes the rest.
    let chunks = |id: RequestId| -> Vec<Vec<i32>> {
        let prefills = harness.batches_of_phase(id, Phase::Prefill);
        prefills
            .iter()
            .map(|batch| row_positions(batch, id))
            .collect()
    };
    assert_eq!(chunks(first), [(0..12).collect::<Vec<i32>>()]);
    assert_eq!(
        chunks(second),
        [(0..4).collect::<Vec<i32>>(), (4..12).collect()]
    );
    let together = steps.iter().any(|formed| {
        formed
            .iter()
            .any(|batch| batch.phase == Phase::Prefill && batch.requests == [first, second])
    });
    assert!(together, "the first chunks share one step");
    harness.settle(100);
    assert_eq!(harness.tokens(first).len(), 3);
    assert_eq!(harness.tokens(second).len(), 3);
}

/// A policy whose package states no step token budget is given the scheduler's (SD §5.2, §7.1
/// "Limits resolved at construction"): with buckets of 16384 prefill tokens, three prompts of
/// 3000 tokens admitted together are prefilled whole in one step of 9000 tokens, above the 8192
/// tokens a policy declares by default.
#[test]
fn a_policy_that_states_no_token_budget_is_budgeted_by_the_scheduler() {
    const PROMPT: u32 = 3000;
    let mut harness = Deployment::tokens()
        .batch(8, 16384)
        .chunk(8192)
        .unstated_budget()
        // Room for the three prompts' blocks of 4 tokens, and for their admissions' answers.
        .kv_units(4096)
        .ring_bytes(1 << 20)
        .build();
    let ids: Vec<RequestId> = (1..=3)
        .map(|id| harness.submit(token(id, PROMPT, 2)))
        .collect();

    let mut prefills = Vec::new();
    for _ in 0..20 {
        let formed = stepped(&mut harness);
        prefills.extend(
            formed
                .into_iter()
                .filter(|batch| batch.phase == Phase::Prefill),
        );
        if !prefills.is_empty() {
            break;
        }
    }
    let [prefill] = &prefills[..] else {
        panic!("one step forms the first prefill: {}", harness.summary());
    };
    assert_eq!(prefill.requests, ids, "every prompt joins the step");
    assert_eq!(prefill.columns.tokens(), 3 * PROMPT as usize, "each whole");

    harness.settle(100);
    for id in ids {
        assert_eq!(harness.tokens(id).len(), 2);
    }
}

/// A request is admitted only onto a group with a free slot: with one slot on each of two
/// groups, a third request waits until a request ends, and runs on the group whose slot that
/// request gave back, while the other still runs (SD §3.5 `place`: the candidates of a phase are
/// the groups with room; §7.1 "Free slots told to the policy").
#[test]
fn a_request_waits_for_a_free_slot_and_runs_where_one_was_given_back() {
    let mut harness = Deployment::tokens().copies(2).slots(1).build();
    let short = harness.submit(token(1, 6, 2));
    let long = harness.submit(token(2, 6, 30));
    let waiting = harness.submit(token(3, 6, 4));
    harness.run_until(10, |harness| {
        !harness.tokens(short).is_empty() && !harness.tokens(long).is_empty()
    });
    assert!(harness.events(waiting).is_empty(), "no slot is free");

    let group = |harness: &Harness, id: RequestId| harness.batches_of(id)[0].group;
    assert_ne!(group(&harness, short), group(&harness, long));
    harness.run_until(100, |harness| scheduled(harness, waiting));
    assert!(harness.terminal(short).is_some());
    assert!(
        harness.terminal(long).is_none(),
        "it took the first slot free"
    );

    harness.settle(200);
    let freed = group(&harness, short);
    let groups = harness.batches_of(waiting).into_iter();
    assert!(groups.map(|batch| batch.group).all(|of| of == freed));
    assert_eq!(harness.tokens(waiting).len(), 4);
    assert_eq!(harness.tokens(long).len(), 30);
}

/// A call is scheduled only within the blocks its request holds: a request whose `Allocate`s
/// are not answered writes every position of the blocks it holds and none beyond, is left out of
/// the steps, which count each call left out, and continues once its blocks arrive. A request
/// whose `Allocate`s are answered in time is never left out: its low watermark asks for the
/// blocks of its next call a step ahead (SD §3.4 `schedule_call`; §4.4 "The low watermark"; §7.6
/// "No count of calls left out of a step for want of blocks"; §10, E1).
#[test]
fn a_call_beyond_its_allocated_blocks_waits_for_them_and_is_counted() {
    let mut harness = Deployment::tokens().build();
    let timely = harness.submit(token(1, 5, 12));
    harness.settle(100);
    assert_eq!(harness.tokens(timely).len(), 12);
    assert_eq!(harness.calls_without_blocks(), 0);

    let starved = harness.submit(token(2, 5, 40));
    harness.run_until(20, |harness| harness.tokens(starved).len() >= 2);
    let owner = harness.owner(starved, A).unwrap();
    harness.pool.withhold(
        move |request, _| matches!(request, Request::Allocate { owner: of, .. } if *of == owner),
    );
    let mut left_out = 0;
    for _ in 0..60 {
        let formed = stepped(&mut harness);
        if formed.iter().any(|batch| batch.requests.contains(&starved)) {
            left_out = 0;
        } else {
            left_out += 1;
        }
        if left_out == 5 {
            break;
        }
    }
    assert_eq!(left_out, 5, "it is left out");

    // Every position of the blocks it holds is written, and none beyond them.
    let entries = harness.fake.entries(A);
    let held = entries
        .iter()
        .filter(|((_, class, _), entry)| *class == 0 && entry.owner == owner)
        .count();
    let last = written(&harness, starved).into_iter().max().unwrap();
    assert_eq!(last, (held as u32 * BLOCK) as i32 - 1);
    assert!(harness.calls_without_blocks() > 0);

    harness.pool.release();
    harness.settle(200);
    assert_eq!(harness.tokens(starved).len(), 40);
    let positions = written(&harness, starved);
    assert_eq!(positions, (0..44).collect::<Vec<i32>>(), "it continues");
}

/// A group holds at most its queue depth of batches on the device: while none returns, the
/// steps form no more for it (SD §3.4: a group's queue; §7.1 `queue_depth`).
#[test]
fn a_group_holds_at_most_its_queue_depth_of_batches_on_the_device() {
    for depth in [1, 3] {
        let mut harness = Deployment::tokens().queue_depth(depth).build();
        harness.fake.hold();
        let mut most = 0;
        for id in 1..=10 {
            harness.submit(token(id, 6, 4));
            harness.step();
            let held = harness.fake.held();
            assert!(held <= depth, "{held} batches on the device, depth {depth}");
            most = most.max(held);
        }
        assert_eq!(most, depth, "the depth is reached");

        harness.fake.resume();
        harness.settle(300);
        for id in 1..=10 {
            assert_eq!(harness.tokens(RequestId(id)).len(), 4);
        }
    }
}

/// A request whose consumer takes no event runs while its window holds the events of its next
/// call and its terminal, and then has no call in flight and keeps its slot, while other
/// requests run to their end; the credit its consumer returns lets it run to its own (SD §3.4
/// "The consumer's window"; §7.1).
#[test]
fn a_slow_consumer_stops_its_request_at_its_window_while_others_run() {
    let mut harness = Deployment::tokens().build();
    let slow = harness.submit(token(1, 6, 200));
    harness.run_until(200, |harness| harness.tokens(slow).len() >= UNCONSUMED);
    let owner = harness.owner(slow, A).unwrap();
    let carried = harness.batches_of(slow).len();

    let other = harness.submit(token(2, 6, 10));
    harness.run_until(100, |harness| harness.terminal(other).is_some());
    harness.run(10);
    assert_eq!(harness.tokens(other).len(), 10);
    assert_eq!(harness.tokens(slow).len(), UNCONSUMED);
    assert_eq!(harness.events(slow).len(), UNCONSUMED + 1);
    assert_eq!(harness.batches_of(slow).len(), carried, "no call of it ran");
    let detached = harness
        .taken()
        .into_iter()
        .any(|request| matches!(request, Request::Detach { owner: of, .. } if of == owner));
    assert!(!detached, "it keeps its slot");

    // The consumer takes every event it was sent, as it is sent.
    let mut taken = 0;
    harness.run_until(400, |harness| {
        let sent = harness.events(slow).len();
        harness.apply(Message::Consumed(slow, (sent - taken) as u32));
        taken = sent;
        harness.terminal(slow).is_some()
    });
    assert_eq!(harness.tokens(slow).len(), 200);
    assert_eq!(
        format!("{:?}", harness.terminal(slow).unwrap()),
        finished(FinishReason::MaxTokens, 6, 200)
    );
}

/// The window keeps room for a request's terminal: a request whose last call leaves the
/// window its reserve ends at its limit with no credit returned, and a request stopped at its
/// window is sent its cancel's terminal at once (SD §3.4, §7.1: a reserve of 2 for the terminal).
#[test]
fn a_terminal_is_sent_behind_a_full_window() {
    let mut harness = Deployment::tokens().build();
    let ending = harness.submit(token(1, 6, UNCONSUMED));
    let stopped = harness.submit(token(2, 6, 200));
    harness.run_until(300, |harness| {
        harness.terminal(ending).is_some() && harness.tokens(stopped).len() >= UNCONSUMED
    });
    harness.run(5);

    let events = harness.events(ending);
    assert_eq!(events.len(), UNCONSUMED + 2);
    assert_eq!(
        format!("{:?}", events.last().unwrap()),
        finished(FinishReason::MaxTokens, 6, UNCONSUMED)
    );

    assert_eq!(harness.tokens(stopped).len(), UNCONSUMED);
    harness.apply(Message::Cancel(stopped, FinishReason::Cancelled));
    assert_eq!(
        format!("{:?}", harness.terminal(stopped).unwrap()),
        finished(FinishReason::Cancelled, 6, UNCONSUMED)
    );
    assert_eq!(harness.events(stopped).len(), UNCONSUMED + 2);
    harness.settle(100);
}

/// A change of the step values serves from the next step on, without a new policy instance:
/// a queue bound that arrives refuses the submissions beyond it, and one that leaves admits them
/// again (`specs/configuration.md` §7, step values).
#[test]
fn step_values_take_effect_at_the_next_step() {
    let mut harness = Deployment::tokens().slots(1).build();
    let running = harness.submit(token(1, 6, 4));
    harness.run_until(10, |harness| !harness.tokens(running).is_empty());
    let overloaded = Some((
        RejectionKind::Overloaded,
        "scheduler waiting queue is full".to_owned(),
    ));

    // No bound: two requests wait beside the running one.
    let first = harness.submit(token(2, 6, 4));
    let second = harness.submit(token(3, 6, 4));
    assert!(harness.events(first).is_empty() && harness.events(second).is_empty());

    // A bound of two arrives: the queue is at it from the next step on.
    harness.tune(|tuning| tuning.max_queued_requests = NonZeroUsize::new(2));
    harness.step();
    let refused = harness.submit(token(4, 6, 4));
    assert_eq!(rejection(&harness, refused), overloaded);

    // The bound leaves again: the queue takes one more.
    harness.tune(|tuning| tuning.max_queued_requests = None);
    harness.step();
    let taken = harness.submit(token(5, 6, 4));
    assert_ne!(rejection(&harness, taken), overloaded);

    harness.settle(400);
    for id in [running, first, second, taken] {
        assert_eq!(harness.tokens(id).len(), 4, "{}", harness.summary());
    }
}
