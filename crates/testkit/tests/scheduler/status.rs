//! Status (SD §8.1): a request goes from waiting to running to finished, for every reason the
//! scheduler ends one (SD §2.1 `RequestStatus`, §3.7, §7.2 "Stop conditions", "Stop strings", "A
//! call's failure"); it is rejected at its arrival or at its admission (§3.2, §3.5, §7.2
//! "Rejection messages"); and a cancel, an abort or a stop string the daemon matched ends it
//! (§3.8, §7.6).
//!
//! What a request's status is reaches the daemon as its events: `Scheduled` when it leaves the
//! waiting queue, its tokens while it runs, and one terminal event when it ends, after which
//! nothing more is sent for it.

use engine::scheduler::Message;
use ipc::batch::{ErrorCode, Phase};
use proto::request::{Event, FinishReason, RejectionKind, RequestId, StopReason};

use crate::harness::{Deployment, EOS, Harness, VIDEO_PHASES, chat_image, token, video};

/// Tokens a case scripts: below the fake's sampled window, above every prompt's ids.
const FIRST: u32 = 950;
const SECOND: u32 = 951;
const STOP: u32 = 960;

/// Request `id`'s events as its life: `Scheduled` first (queued no later than scheduled), then
/// only its tokens, then its terminal event last. Returns its tokens and its terminal event.
fn lifecycle(harness: &Harness, id: RequestId) -> (Vec<u32>, Event) {
    let events = harness.events(id);
    let Some((
        Event::Scheduled {
            queued_at,
            scheduled_at,
        },
        rest,
    )) = events.split_first()
    else {
        panic!("request {} was not scheduled first: {events:?}", id.0);
    };
    assert!(queued_at <= scheduled_at, "{events:?}");
    let Some((terminal, running)) = rest.split_last() else {
        panic!("request {} has not ended: {events:?}", id.0);
    };
    let tokens = running.iter().map(|event| match event {
        Event::TextToken { id, .. } => *id,
        other => panic!("request {} was sent {other:?} while it ran", id.0),
    });
    (tokens.collect(), terminal.clone())
}

/// The `Finished` of a token request of `prompt_tokens` that generated `completion_tokens`.
fn finished(
    reason: FinishReason,
    stop_reason: Option<StopReason>,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> String {
    let event = Event::Finished {
        reason,
        stop_reason,
        prompt_tokens,
        completion_tokens,
    };
    format!("{event:?}")
}

#[test]
fn a_request_runs_from_waiting_to_finished_at_its_token_limit() {
    let mut harness = Deployment::tokens().build();
    let id = harness.submit(token(1, 10, 5));
    harness.settle(100);

    let (tokens, terminal) = lifecycle(&harness, id);
    assert_eq!(tokens.len(), 5);
    assert_eq!(
        format!("{terminal:?}"),
        finished(FinishReason::MaxTokens, None, 10, 5)
    );
}

/// An end-of-sequence token ends the request and is never published; usage counts it.
#[test]
fn an_end_of_sequence_token_finishes_the_request_without_being_sent() {
    let mut harness = Deployment::tokens().build();
    let id = harness.submit(token(1, 10, 8));
    harness.sample(id, &[FIRST, SECOND, EOS]);
    harness.settle(100);

    let (tokens, terminal) = lifecycle(&harness, id);
    assert_eq!(tokens, [FIRST, SECOND]);
    assert_eq!(
        format!("{terminal:?}"),
        finished(FinishReason::Eos, None, 10, 3)
    );
}

/// A stop token ends the request with the token it matched, and is published only to a request
/// that includes it.
#[test]
fn a_stop_token_finishes_the_request_and_is_sent_only_when_included() {
    let mut harness = Deployment::tokens().build();
    let mut excluded = token(1, 10, 8);
    excluded.stop_token_ids = vec![STOP];
    let mut included = token(2, 10, 8);
    included.stop_token_ids = vec![STOP];
    included.include_stop_token = true;
    let excluded = harness.submit(excluded);
    let included = harness.submit(included);
    harness.sample(excluded, &[FIRST, STOP, SECOND]);
    harness.sample(included, &[FIRST, STOP, SECOND]);
    harness.settle(100);

    let stopped = finished(FinishReason::Stop, Some(StopReason::Token(STOP)), 10, 2);
    let (tokens, terminal) = lifecycle(&harness, excluded);
    assert_eq!(tokens, [FIRST]);
    assert_eq!(format!("{terminal:?}"), stopped);
    let (tokens, terminal) = lifecycle(&harness, included);
    assert_eq!(tokens, [FIRST, STOP]);
    assert_eq!(format!("{terminal:?}"), stopped);
}

/// A stop string the daemon's decoder matched ends a running request with `Stop` and no stop
/// token. Under stop strings a request runs no further ahead of the daemon's acknowledgement than
/// its group's queue depth of outputs, those of its calls in flight included (SD §3.3 `Decode`,
/// §7.2 "Pipelined decodes").
#[test]
fn a_stop_string_the_daemon_matched_finishes_the_running_request_with_stop() {
    let mut harness = Deployment::tokens().queue_depth(2).build();
    let mut request = token(1, 10, 20);
    request.stop_strings = vec!["end".to_owned()];
    let id = harness.submit(request);

    // The daemon passes nothing: the prefill's token and one decode are two outputs.
    harness.run(30);
    assert_eq!(harness.tokens(id).len(), 2);
    harness.apply(Message::Acknowledge(id, 2));
    harness.run_until(30, |harness| harness.tokens(id).len() == 4);
    harness.run(10);
    assert_eq!(harness.tokens(id).len(), 4);

    harness.apply(Message::Cancel(id, FinishReason::Stop));
    harness.settle(100);
    let (tokens, terminal) = lifecycle(&harness, id);
    assert_eq!(tokens.len(), 4, "no token after the stop");
    assert_eq!(
        format!("{terminal:?}"),
        finished(FinishReason::Stop, None, 10, 4)
    );
}

/// Steps until request `id` has published `tokens` tokens, acknowledging each one as the daemon's
/// decoder passes it but the last `unpassed`.
fn acknowledging(harness: &mut Harness, id: RequestId, tokens: usize, unpassed: usize) {
    harness.run_until(100, |harness| {
        let published = harness.tokens(id).len();
        let passed = published.min(tokens - unpassed);
        harness.apply(Message::Acknowledge(id, passed as u32));
        published >= tokens
    });
}

/// A request with stop strings that finished by itself holds its `Finished` until the daemon's
/// decoder passed every token it published, which may still match a stop string.
#[test]
fn a_finish_under_stop_strings_waits_for_the_daemon_to_pass_its_tokens() {
    let mut harness = Deployment::tokens().build();
    let mut request = token(1, 10, 3);
    request.stop_strings = vec!["end".to_owned()];
    let id = harness.submit(request);
    acknowledging(&mut harness, id, 3, 1);
    harness.run(10);
    assert!(harness.terminal(id).is_none(), "{:?}", harness.events(id));

    harness.apply(Message::Acknowledge(id, 3));
    let (tokens, terminal) = lifecycle(&harness, id);
    assert_eq!(tokens.len(), 3);
    assert_eq!(
        format!("{terminal:?}"),
        finished(FinishReason::MaxTokens, None, 10, 3)
    );
    harness.settle(50);
}

/// A stop string the decoder matches after the request finished by itself turns its held
/// `Finished` into a `Stop` with no stop token. Its final end-of-sequence token was never
/// published, so the decoder holds two tokens of its three.
#[test]
fn a_stop_string_matched_after_the_finish_turns_it_into_a_stop() {
    let mut harness = Deployment::tokens().build();
    let mut request = token(1, 10, 8);
    request.stop_strings = vec!["end".to_owned()];
    let id = harness.submit(request);
    harness.sample(id, &[FIRST, SECOND, EOS]);
    acknowledging(&mut harness, id, 2, 1);
    harness.run(10);
    assert!(harness.terminal(id).is_none(), "{:?}", harness.events(id));

    harness.apply(Message::Cancel(id, FinishReason::Stop));
    let (tokens, terminal) = lifecycle(&harness, id);
    assert_eq!(tokens, [FIRST, SECOND]);
    assert_eq!(
        format!("{terminal:?}"),
        finished(FinishReason::Stop, None, 10, 3)
    );
    harness.settle(50);
}

/// A cancel ends a waiting request at once, with nothing generated, and it never runs; the
/// request ahead of it runs on.
#[test]
fn a_cancel_ends_a_waiting_request_at_once_with_no_tokens() {
    let mut harness = Deployment::tokens().slots(1).build();
    let running = harness.submit(token(1, 10, 6));
    let waiting = harness.submit(token(2, 10, 6));
    harness.run(3);
    assert!(harness.events(waiting).is_empty(), "one slot: it waits");

    harness.apply(Message::Cancel(waiting, FinishReason::Cancelled));
    assert_eq!(
        harness
            .events(waiting)
            .iter()
            .map(|event| format!("{event:?}"))
            .collect::<Vec<_>>(),
        [finished(FinishReason::Cancelled, None, 10, 0)]
    );
    harness.settle(100);
    assert!(harness.batches_of(waiting).is_empty());
    let (tokens, _) = lifecycle(&harness, running);
    assert_eq!(tokens.len(), 6);
}

/// A cancel ends a running request at once, while its call is still on the device: the call's
/// output is dropped when it returns, and the request's owner is given up only then (SD §7.6).
#[test]
fn a_cancel_ends_a_running_request_at_once_and_drops_its_call_in_flight() {
    let mut harness = Deployment::tokens().build();
    let id = harness.submit(token(1, 10, 20));
    harness.run_until(30, |harness| harness.tokens(id).len() >= 2);
    harness.fake.hold();
    harness.run_until(10, |harness| harness.fake.held() > 0);
    let generated = harness.tokens(id).len();

    harness.apply(Message::Cancel(id, FinishReason::Cancelled));
    let (tokens, terminal) = lifecycle(&harness, id);
    assert_eq!(tokens.len(), generated);
    assert_eq!(
        format!("{terminal:?}"),
        finished(FinishReason::Cancelled, None, 10, generated)
    );
    harness.run(5);
    let detached = |harness: &Harness| {
        let taken = harness.taken();
        taken
            .iter()
            .any(|request| matches!(request, storage::runtime::Request::Detach { .. }))
    };
    assert!(!detached(&harness), "its call is still in flight");

    harness.fake.resume();
    harness.settle(50);
    assert!(detached(&harness));
    assert_eq!(
        harness.events(id).len(),
        generated + 2,
        "nothing after its end"
    );
}

/// An abort ends a running request with `Aborted`.
#[test]
fn an_abort_ends_a_running_request_with_aborted() {
    let mut harness = Deployment::tokens().build();
    let id = harness.submit(token(1, 10, 20));
    harness.run_until(30, |harness| harness.tokens(id).len() >= 2);

    harness.apply(Message::Cancel(id, FinishReason::Aborted));
    let generated = harness.tokens(id).len();
    harness.settle(50);
    let (_, terminal) = lifecycle(&harness, id);
    assert_eq!(
        format!("{terminal:?}"),
        finished(FinishReason::Aborted, None, 10, generated)
    );
}

/// A call that failed for a reason other than its input, a numerical fault or its rank's loss
/// fails its request: an `Error` naming the batch, the row and the phase, then `Finished` with
/// `Error` (SD §7.2 "A call's failure").
#[test]
fn a_failed_call_fails_its_request_with_an_error_naming_the_call() {
    let mut harness = Deployment::tokens().build();
    let id = harness.submit(token(1, 10, 20));
    harness.run_until(30, |harness| harness.tokens(id).len() >= 2);
    harness.fail_calls(id, Some((ErrorCode::ResourceError, None)));
    harness.run_until(10, |harness| harness.terminal(id).is_some());
    harness.settle(50);

    let events = harness.events(id);
    let [.., Event::Error { message }, terminal] = &events[..] else {
        panic!("{events:?}");
    };
    let failed = harness.batches_of(id).pop().unwrap();
    let row = failed.requests.iter().position(|of| *of == id).unwrap();
    assert!(
        message.ends_with(&format!(
            "(call {}.{row}, {:?})",
            failed.id.0,
            Phase::Decode
        )),
        "{message}"
    );
    let generated = events.len() - 3;
    assert_eq!(
        format!("{terminal:?}"),
        finished(FinishReason::Error, None, 10, generated)
    );
}

/// A request whose next call needs more blocks than the whole cache holds ends with
/// `CacheCapacity` rather than starving (SD §3.6).
#[test]
fn a_request_the_whole_cache_cannot_hold_finishes_with_cache_capacity() {
    let mut harness = Deployment::tokens().kv_units(4).build();
    let id = harness.submit(token(1, 6, 100));
    harness.settle(200);

    let (tokens, terminal) = lifecycle(&harness, id);
    assert_eq!(
        format!("{terminal:?}"),
        finished(FinishReason::CacheCapacity, None, 6, tokens.len())
    );
}

/// A request that outgrows the cache ends at the length the whole cache holds: its calls write
/// every position of the cache's blocks, and it ends when its next call needs a block beyond them
/// (SD §3.6: a running request is short when its next call writes a block it does not hold;
/// `FinishReason::CacheCapacity`; `S/preemption/tokens.rs:127-164`).
#[test]
fn a_request_that_outgrows_the_cache_ends_at_the_length_the_cache_holds() {
    const UNITS: u32 = 4;
    let mut harness = Deployment::tokens().kv_units(UNITS).build();
    let id = harness.submit(token(1, 6, 100));
    harness.settle(200);

    let written = harness
        .batches_of(id)
        .iter()
        .flat_map(|batch| batch.columns.rows.cache_positions.clone())
        .max();
    let cached = (UNITS * crate::harness::BLOCK) as i32;
    assert_eq!(written, Some(cached - 1));
    let (tokens, terminal) = lifecycle(&harness, id);
    assert_eq!(
        format!("{terminal:?}"),
        finished(FinishReason::CacheCapacity, None, 6, tokens.len())
    );
}

/// A video request runs from waiting to `Completed`: its progress while it runs, then its
/// artifact, then its end (SD §7.3 "Progress and artifact").
#[test]
fn a_video_request_finishes_completed_after_its_artifact() {
    let mut harness = Deployment::video(&[&VIDEO_PHASES]).build();
    let id = harness.submit_video(video(1, 1));
    harness.settle(200);
    video_completed(&harness, id);
}

/// A video request whose media units take several decode rounds completes as one of one round.
#[test]
fn a_video_request_of_several_units_finishes_completed() {
    let mut harness = Deployment::video(&[&VIDEO_PHASES]).build();
    let id = harness.submit_video(video(1, 2));
    harness.settle(300);
    video_completed(&harness, id);
}

/// Request `id`'s events are those of a video request that completed: `Scheduled`, its progress,
/// its artifact, then `Finished` with `Completed` and its prompt as its usage.
fn video_completed(harness: &Harness, id: RequestId) {
    let events = harness.events(id);
    assert!(
        matches!(events.first(), Some(Event::Scheduled { .. })),
        "{events:?}"
    );
    let [.., Event::Artifact(artifact), terminal] = &events[..] else {
        panic!("{events:?}");
    };
    assert_eq!(artifact.content_type, "video/mp4");
    assert_eq!(
        format!("{terminal:?}"),
        finished(FinishReason::Completed, None, 40, 0)
    );
    let running = &events[1..events.len() - 2];
    assert!(
        running
            .iter()
            .all(|event| matches!(event, Event::MediaProgress { .. })),
        "{events:?}"
    );
}

/// A request that fails its validation is rejected as invalid at its arrival, and the scheduler
/// keeps nothing of it.
#[test]
fn a_request_failing_validation_is_rejected_at_arrival() {
    let mut harness = Deployment::tokens().build();
    let id = harness.submit(token(1, 10, 0));
    let events = harness.events(id);
    let [Event::Rejected { kind, message }] = &events[..] else {
        panic!("{events:?}");
    };
    assert_eq!(*kind, RejectionKind::Invalid);
    assert!(
        message.starts_with("invalid generation request: "),
        "{message}"
    );

    harness.run(5);
    assert_eq!(harness.events(id).len(), 1);
    assert!(harness.batches_of(id).is_empty());
}

/// A request that arrives while the waiting queue is at the policy's bound is rejected as
/// overloaded at its arrival, and kept nowhere (SD §3.2, §7.2 "Rejection messages").
#[test]
fn a_request_beyond_the_queue_bound_is_rejected_at_arrival_as_overloaded() {
    let mut harness = Deployment::tokens().slots(1).max_queued(1).build();
    let running = harness.submit(token(1, 6, 8));
    harness.run_until(10, |harness| !harness.tokens(running).is_empty());
    let waiting = harness.submit(token(2, 6, 2));
    let refused = harness.submit(token(3, 6, 2));
    let events = harness.events(refused);
    let [Event::Rejected { kind, message }] = &events[..] else {
        panic!("{events:?}");
    };
    assert_eq!(*kind, RejectionKind::Overloaded);
    assert_eq!(message, "scheduler waiting queue is full");
    harness.settle(100);
    assert_eq!(harness.tokens(waiting).len(), 2);
    assert!(harness.batches_of(refused).is_empty());
}

/// A token request to a deployment that holds no KV cache is rejected as invalid at its arrival.
#[test]
fn a_token_request_without_a_kv_cache_to_run_on_is_rejected_at_arrival() {
    let mut harness = Deployment::video(&[&VIDEO_PHASES]).build();
    let id = harness.submit(token(1, 10, 4));
    let events = harness.events(id);
    let [Event::Rejected { kind, message }] = &events[..] else {
        panic!("{events:?}");
    };
    assert_eq!(*kind, RejectionKind::Invalid);
    assert_eq!(message, "generation request requires worker KV resources");
}

/// A request whose prompt needs more blocks than the whole cache holds is rejected at its
/// admission and never scheduled, and the queue behind it is admitted.
#[test]
fn a_request_longer_than_the_whole_cache_is_rejected_at_admission() {
    let mut harness = Deployment::tokens().kv_units(4).build();
    let long = harness.submit(token(1, 40, 4));
    let short = harness.submit(token(2, 6, 4));
    harness.settle(100);

    let events = harness.events(long);
    let [Event::Rejected { kind, message }] = &events[..] else {
        panic!("{events:?}");
    };
    assert_eq!(*kind, RejectionKind::Invalid);
    assert_eq!(message, "request exceeds total KV capacity");
    let (tokens, _) = lifecycle(&harness, short);
    assert_eq!(tokens.len(), 4);
}

/// Requests cancelled at every point of their life (waiting, admitting, prefilling, decoding,
/// with calls in flight) each end once with `Cancelled`, are sent nothing after it, and give
/// back everything they held.
#[test]
fn requests_cancelled_at_every_point_of_their_life_each_end_once() {
    let mut harness = Deployment::tokens().images().slots(3).build();
    let ids: Vec<RequestId> = (1..=8)
        .map(|id| harness.submit(token(id, 12 + id as u32, 30)))
        .collect();
    let chat = harness.submit(chat_image(9, 12, 6, 30));

    for (step, id) in ids.iter().chain([&chat]).enumerate() {
        if step % 2 == 0 {
            harness.fake.hold();
        }
        harness.step();
        harness.apply(Message::Cancel(*id, FinishReason::Cancelled));
        harness.fake.resume();
    }
    harness.settle(200);

    for id in ids.iter().chain([&chat]) {
        let events = harness.events(*id);
        let ends: Vec<&Event> = events
            .iter()
            .filter(|event| matches!(event, Event::Finished { .. }))
            .collect();
        assert_eq!(ends.len(), 1, "{events:?}");
        let Some(Event::Finished { reason, .. }) = events.last() else {
            panic!("request {} ended with {events:?}", id.0);
        };
        assert_eq!(*reason, FinishReason::Cancelled);
    }
}
