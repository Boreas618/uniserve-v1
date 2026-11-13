//! L6 (SD §10; `specs/buffer-pool.md` §19): the scheduler over a buffer pool whose completions
//! are delayed, reordered and withheld. No step waits for a completion; a request whose blocks
//! do not cover its next call is left out of the step, and counted, while the others run; every
//! request ends.
//!
//! Each case runs a mixed load under one pace of the buffer pool: token requests of several
//! lengths and a chat image request on a deployment that encodes images, and the token requests
//! on a deployment that speculates. Every step of every case also checks the scheduler's side of
//! A1 to A4 (the harness's step).

use std::collections::HashSet;

use engine::batch::Phase;
use proto::request::{Event, FinishReason, RequestId};
use storage::runtime::{Completion, Request};

use crate::harness::{Deployment, Harness, chat_image, token};

/// The deployment the load with its chat image runs on: two replicas of the group that
/// prefills, decodes and encodes images, with prefill chunks of 8 tokens and steps of 32, so that
/// the longer prompts are prefilled in chunks beside the decodes.
fn images() -> Deployment {
    Deployment::tokens()
        .images()
        .replicas(2)
        .batch(8, 32)
        .chunk(8)
}

/// How the buffer pool paces its completions.
#[derive(Clone, Copy, Debug)]
enum Pace {
    /// Every completion is written this many steps after the buffer pool produced it.
    Delayed(u64),
    /// Completions are held over windows of this many steps, then written in reverse.
    Reordered(u64),
    /// The answers to one running request's `Allocate`s are withheld until released.
    Withheld,
}

/// The load: token requests of several prompt and output lengths, and a chat image request, each
/// with the tokens it may generate.
const TOKENS: [(u64, u32, usize); 5] =
    [(1, 5, 3), (2, 17, 12), (3, 40, 24), (4, 9, 30), (5, 3, 16)];
const CHAT: (u64, u32, u32, usize) = (6, 12, 6, 8);

/// The request whose `Allocate`s are withheld: a long one.
const STARVED: RequestId = RequestId(4);

/// The steps a case may take: far more than the load needs at any pace.
const STEPS: u64 = 2_000;

/// What a run observed: the steps that formed a batch while the buffer pool owed completions.
struct Run {
    harness: Harness,
    ids: Vec<RequestId>,
    formed_while_owed: u64,
}

/// Submits the load (the chat image request with `images`) and steps under `pace` until every
/// request ended and the buffer pool holds nothing of any of them.
fn run(deployment: Deployment, images: bool, pace: Pace) -> Run {
    let mut harness = deployment.build();
    match pace {
        Pace::Delayed(steps) => harness.pool.delay(steps),
        Pace::Reordered(steps) => harness.reorder(Some(steps)),
        Pace::Withheld => {}
    }

    let mut ids: Vec<RequestId> = TOKENS
        .iter()
        .map(|&(id, prompt, max)| harness.submit(token(id, prompt, max)))
        .collect();
    if images {
        let (id, prompt, image, max) = CHAT;
        ids.push(harness.submit(chat_image(id, prompt, image, max)));
    }

    let mut formed_while_owed = 0;
    let mut step = |harness: &mut Harness| {
        if harness.step() && harness.outstanding() > 0 {
            formed_while_owed += 1;
        }
    };
    if let Pace::Withheld = pace {
        withhold_allocations(&mut harness, &mut step);
    }
    for _ in 0..STEPS {
        let ended = ids.iter().all(|id| harness.terminal(*id).is_some());
        if ended && harness.held() == 0 {
            break;
        }
        step(&mut harness);
    }
    harness.settle(1);
    Run {
        harness,
        ids,
        formed_while_owed,
    }
}

/// Withholds the answers to [`STARVED`]'s `Allocate`s once it decodes, until it has been left out
/// of the steps for want of blocks while the other requests ran; then releases them.
fn withhold_allocations(harness: &mut Harness, step: &mut impl FnMut(&mut Harness)) {
    let decodes = |harness: &Harness| {
        let batches = harness.batches_of(STARVED);
        batches
            .iter()
            .any(|batch| matches!(batch.phase, Phase::Decode | Phase::Verify))
    };
    while !decodes(harness) {
        step(harness);
    }
    let group = harness.batches_of(STARVED).last().unwrap().group;
    let owner = harness.owner(STARVED, group).unwrap();
    harness.pool.withhold(
        move |request, _| matches!(request, Request::Allocate { owner: of, .. } if *of == owner),
    );

    // Its blocks run out within its low watermark's tokens; from then on the steps leave it out
    // and run the others.
    let mut left_out = 0;
    let mut others = 0;
    for _ in 0..60 {
        let before = harness.batches().len();
        step(harness);
        let formed = harness.batches().split_off(before);
        let carries = |id: RequestId| formed.iter().any(|batch| batch.requests.contains(&id));
        if !carries(STARVED) && harness.pool.withheld() > 0 {
            left_out += 1;
            others += u64::from(formed.iter().any(|batch| !batch.requests.is_empty()));
        } else {
            left_out = 0;
        }
        if left_out >= 10 {
            break;
        }
    }
    assert!(left_out >= 10, "request {} was not left out", STARVED.0);
    assert!(others > 0, "no other request ran while it was left out");
    assert!(harness.terminal(STARVED).is_none());
    harness.pool.release();
}

/// What every pace must show: steps formed batches while completions were owed; calls were left
/// out for want of blocks and counted; every request ended by its own end, with every token it
/// may generate sent once, and gave back everything it held; the scheduler did not fail.
fn ended(mut run: Run) {
    let harness = &mut run.harness;
    assert_eq!(harness.failure(), None);
    assert!(
        run.formed_while_owed > 0,
        "no step formed a batch while completions were owed"
    );
    assert!(
        harness.calls_without_blocks() > 0,
        "no call was left out for want of blocks"
    );

    let limits: Vec<usize> = TOKENS
        .iter()
        .map(|&(_, _, max)| max)
        .chain([CHAT.3])
        .collect();
    for (id, max) in run.ids.iter().zip(limits) {
        let events = harness.events(*id);
        let Some(Event::Finished {
            reason,
            completion_tokens,
            ..
        }) = events.last()
        else {
            panic!("request {} did not finish: {events:?}", id.0);
        };
        assert_eq!(*reason, FinishReason::MaxTokens, "{events:?}");
        assert_eq!(*completion_tokens, max);
        assert_eq!(harness.tokens(*id).len(), max);
        assert!(matches!(events.first(), Some(Event::Scheduled { .. })));
    }
    assert_eq!(harness.held(), 0);

    // Every completion the buffer pool wrote answered a request it took, once.
    let answered: Vec<u64> = harness
        .pool
        .completions()
        .iter()
        .map(|(_, id, _)| id.0)
        .collect();
    let unique: HashSet<u64> = answered.iter().copied().collect();
    assert_eq!(unique.len(), answered.len());
    let refused = harness
        .answers()
        .into_iter()
        .filter(|(_, completion)| matches!(completion, Completion::Refused { .. }))
        .count();
    assert_eq!(refused, 0);
}

#[test]
fn delayed_completions_leave_requests_out_and_every_request_ends() {
    ended(run(images(), true, Pace::Delayed(6)));
}

#[test]
fn reordered_completions_leave_requests_out_and_every_request_ends() {
    ended(run(images(), true, Pace::Reordered(5)));
}

#[test]
fn withheld_allocations_leave_their_request_out_while_the_others_run_and_every_request_ends() {
    ended(run(images(), true, Pace::Withheld));
}

#[test]
fn delayed_completions_under_speculation_leave_requests_out_and_every_request_ends() {
    ended(run(
        Deployment::tokens().speculative(2),
        false,
        Pace::Delayed(6),
    ));
}

#[test]
fn reordered_completions_under_speculation_leave_requests_out_and_every_request_ends() {
    ended(run(
        Deployment::tokens().speculative(2),
        false,
        Pace::Reordered(5),
    ));
}

#[test]
fn withheld_allocations_under_speculation_leave_their_request_out_and_every_request_ends() {
    ended(run(
        Deployment::tokens().speculative(2),
        false,
        Pace::Withheld,
    ));
}

/// The mixed load with its chat image request, on a deployment that speculates.
#[test]
fn withheld_allocations_under_speculation_with_a_chat_image_and_every_request_ends() {
    ended(run(
        Deployment::tokens().images().speculative(2),
        true,
        Pace::Withheld,
    ));
}
