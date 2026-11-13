//! Failures (SD §8.1): what a failed batch and a failed call do to the requests they carried and
//! to the scheduler (SD §3.7 `update`, §7.2 "A call's failure", §7.6).
//!
//! - An executor that failed a batch, or answered it with other calls than it was given, stops
//!   the scheduler: every request it holds, running or waiting, leaves as `Migratable { Loss }`,
//!   which the daemon resubmits, and the engine is told the first error, which names the batch.
//! - A call that failed fails its request alone: an input the rank refused rejects it as invalid,
//!   non-finite logits fail it and are reported with the batch (`Output::NumericalFault`), and a
//!   rank that can no longer serve stops the scheduler.

use engine::scheduler::{Message, Output};
use ipc::batch::{BatchOutput, ErrorCode, Phase};
use proto::request::{Event, FinishReason, MigrationCause, RejectionKind, RequestId};

use crate::harness::{Deployment, Harness, token};

/// Whether request `id`'s life ended as `Migratable { Loss }`.
fn lost(harness: &Harness, id: RequestId) -> bool {
    matches!(
        harness.terminal(id),
        Some(Event::Migratable {
            cause: MigrationCause::Loss,
            ..
        })
    )
}

/// The scheduler's failures (`Output::Failed`), as they read, in the order sent.
fn failures(harness: &Harness) -> Vec<String> {
    let failed = harness.outputs().iter().filter_map(|output| match output {
        Output::Failed(error) => Some(error.to_string()),
        _ => None,
    });
    failed.collect()
}

/// Two requests in one deployment of one slot: the first decoding with one batch of it held on
/// the device, the second waiting behind it. Returns the deployment and both ids.
fn decoding_and_waiting() -> (Harness, RequestId, RequestId) {
    let mut harness = Deployment::tokens().slots(1).build();
    let running = harness.submit(token(1, 10, 20));
    harness.run_until(30, |harness| harness.tokens(running).len() >= 2);
    harness.fake.hold();
    harness.run_until(10, |harness| harness.fake.held() > 0);
    let waiting = harness.submit(token(2, 10, 4));
    harness.step();
    assert!(harness.events(waiting).is_empty(), "one slot: it waits");
    (harness, running, waiting)
}

/// A batch the device failed after it launched (its executor answers it with an error) stops the
/// scheduler: the running request and the waiting one both leave as `Migratable { Loss }`, and
/// the engine is told once, by the first error, which names the failed batch and what the device
/// said (SD §3.7, §7.6; `in_process_stalls.rs`).
#[test]
fn a_failed_batch_ends_its_requests_as_migratable_and_the_first_error_names_it() {
    let (mut harness, running, waiting) = decoding_and_waiting();
    // The oldest batch held fails first: its error is the first.
    let batches = harness.batches();
    let failed = batches[batches.len() - harness.fake.held()].clone();
    assert!(failed.requests.contains(&running));

    assert!(harness.fake.fail_held("the forward stalled for 4.0 s") >= 1);
    harness.step();
    assert!(lost(&harness, running), "{:?}", harness.events(running));
    assert!(lost(&harness, waiting), "{:?}", harness.events(waiting));

    let failures = failures(&harness);
    let [failure] = &failures[..] else {
        panic!("one failure is reported: {failures:?}");
    };
    for named in [
        format!("batch {}", failed.id.0),
        "the forward stalled for 4.0 s".to_owned(),
    ] {
        assert!(failure.contains(&named), "{failure} does not name {named}");
    }

    // A stopped scheduler forms nothing more and sends nothing more.
    let batches = harness.batches().len();
    let outputs = harness.outputs().len();
    harness.run(5);
    assert_eq!(harness.batches().len(), batches);
    assert_eq!(harness.outputs().len(), outputs);
}

/// A batch whose executor answers it with other calls than it was given stops the scheduler: its
/// requests and the waiting one leave as `Migratable { Loss }`, and the failure names the batch
/// (SD §3.7 `update`).
#[test]
fn an_output_naming_other_calls_than_its_batch_fails_the_scheduler() {
    let (mut harness, running, waiting) = decoding_and_waiting();
    let held = harness.batches().pop().unwrap();

    // The executor answers the held batch with no call at all.
    let output = BatchOutput {
        calls: Vec::new(),
        ..BatchOutput::default()
    };
    harness.apply(Message::Output(held.clone(), Ok(output)));
    assert!(lost(&harness, running), "{:?}", harness.events(running));
    assert!(lost(&harness, waiting), "{:?}", harness.events(waiting));
    assert_eq!(
        failures(&harness),
        [format!(
            "an executor failed: batch {} returned other calls than it was given",
            held.id.0
        )]
    );
}

/// A call the rank refused for its input rejects its request as invalid with the rank's reason;
/// the other request of the batch runs to its end, and the scheduler serves on (SD §7.2 "A
/// call's failure": `InputError`).
#[test]
fn an_input_error_rejects_its_request_alone() {
    let mut harness = Deployment::tokens().build();
    let refused = harness.submit(token(1, 10, 20));
    let other = harness.submit(token(2, 10, 6));
    harness.run_until(30, |harness| harness.tokens(refused).len() >= 2);
    let reason = "the image cannot be decoded".to_owned();
    harness.fail_calls(refused, Some((ErrorCode::InputError, Some(reason.clone()))));
    harness.settle(100);

    let events = harness.events(refused);
    let Some(Event::Rejected { kind, message }) = events.last() else {
        panic!("{events:?}");
    };
    assert_eq!((*kind, message), (RejectionKind::Invalid, &reason));
    assert_eq!(harness.tokens(other).len(), 6);
    assert_eq!(harness.failure(), None);
}

/// A call that computed non-finite logits fails its request alone (an `Error` naming the call,
/// then `Finished { Error }`) and is reported once, with the batch that carried it and that
/// request alone; the other request of the batch runs to its end (SD §3.7, §7.2 "A call's
/// failure": `NumericalFault`; §2.5 `Output::NumericalFault`).
#[test]
fn a_numerical_fault_fails_its_request_alone_and_is_reported_with_its_batch() {
    let mut harness = Deployment::tokens().build();
    let faulted = harness.submit(token(1, 10, 20));
    let other = harness.submit(token(2, 10, 8));
    harness.run_until(30, |harness| harness.tokens(faulted).len() >= 2);
    harness.fail_calls(faulted, Some((ErrorCode::NumericalFault, None)));
    harness.settle(100);

    let reports: Vec<_> = harness
        .outputs()
        .iter()
        .filter_map(|output| match output {
            Output::NumericalFault(batch, requests) => Some((*batch, requests.clone())),
            _ => None,
        })
        .collect();
    let [(batch, requests)] = &reports[..] else {
        panic!("one fault is reported: {reports:?}");
    };
    assert_eq!(requests, &[faulted]);
    let carried = harness.batches().into_iter().find(|of| of.id == *batch);
    let carried = carried.unwrap();
    assert!(carried.requests.contains(&faulted));

    let events = harness.events(faulted);
    let [.., Event::Error { message }, Event::Finished { reason, .. }] = &events[..] else {
        panic!("{events:?}");
    };
    assert_eq!(*reason, FinishReason::Error);
    let row = carried
        .requests
        .iter()
        .position(|id| *id == faulted)
        .unwrap();
    let call = format!("(call {}.{row}, {:?})", batch.0, Phase::Decode);
    assert!(message.ends_with(&call), "{message}");
    assert_eq!(harness.tokens(other).len(), 8);
    assert_eq!(harness.failure(), None);
}

/// A call whose rank broke an invariant it relies on, or can no longer serve, stops the
/// scheduler, whose failure names the batch and the call; every request leaves as
/// `Migratable { Loss }` (SD §7.2 "A call's failure": `InvariantViolation`, `FatalWorkerFailure`).
#[test]
fn a_call_whose_rank_cannot_serve_fails_the_scheduler() {
    for code in [ErrorCode::InvariantViolation, ErrorCode::FatalWorkerFailure] {
        let mut harness = Deployment::tokens().build();
        let failing = harness.submit(token(1, 10, 20));
        let other = harness.submit(token(2, 10, 20));
        harness.run_until(30, |harness| harness.tokens(failing).len() >= 2);
        harness.fail_calls(
            failing,
            Some((code, Some("the rank's tables broke".to_owned()))),
        );
        harness.run_until(10, |harness| harness.failure().is_some());

        let failure = harness.failure().unwrap();
        let named = harness.batches_of(failing).iter().any(|batch| {
            let row = batch.requests.iter().position(|id| *id == failing);
            row.is_some_and(|row| failure.contains(&format!("batch {}, call {row}", batch.id.0)))
        });
        assert!(named, "{failure}");
        assert!(failure.contains(code.as_str()), "{failure}");
        assert!(lost(&harness, failing) && lost(&harness, other), "{code:?}");
    }
}
