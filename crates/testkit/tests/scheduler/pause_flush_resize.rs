//! Pause, flush and resize (SD §8.1): the operations the engine runs at a batch boundary (SD
//! §2.4, §3.1). `Pause` forms no batch and is answered once nothing is in flight; `Flush` evicts
//! every cached block the scheduler knows of, with a weight update's seeds from then on, while
//! the requests it holds keep their state, compute it again or end as the update says; `Resize`
//! is answered at its `Resized` with the units the region reached, which are its capacity from
//! then on.

use crossbeam_channel::{Receiver, bounded};
use engine::scheduler::{Message, Seeds};
use proto::request::wire::WeightsInFlight;
use proto::request::{Event, FinishReason, MigrationCause, RejectionKind};
use storage::Key;
use storage::runtime::{Completion, Request};

use crate::harness::{A, Deployment, Harness, sharing, token};

/// Sends `Flush` with the seeds and the treatment of the requests in flight of a weight update
/// (`None`: a flush alone), and returns the reply, which the scheduler sends at once.
pub(crate) fn flush(
    harness: &mut Harness,
    seeds: Option<Seeds>,
    in_flight: WeightsInFlight,
) -> Result<(), engine::Error> {
    let (reply, replied) = bounded(1);
    harness.apply(Message::Flush(seeds, in_flight, reply));
    replied.try_recv().unwrap()
}

/// The page seeds of a weight update: one per cache group of the harness's token deployment,
/// whose groups encode nothing.
fn update_seeds() -> Option<Seeds> {
    Some((vec![Key([9; 32])], 7, Vec::new()))
}

/// `Pause` is answered once no batch is in flight, and from it on the scheduler forms no batch;
/// it forms them again at `Resume`, and the request runs to its end (SD §2.4, §3.1
/// `reply_at_boundary`).
#[test]
fn pause_is_answered_once_nothing_is_in_flight_and_forms_nothing_until_resume() {
    let mut harness = Deployment::tokens().build();
    let id = harness.submit(token(1, 10, 20));
    harness.run_until(30, |harness| harness.tokens(id).len() >= 2);
    harness.fake.hold();
    harness.run_until(10, |harness| harness.fake.held() > 0);

    let (reply, replied): (_, Receiver<()>) = bounded(1);
    harness.apply(Message::Pause(reply));
    let formed = harness.batches().len();
    harness.run(5);
    assert!(replied.try_recv().is_err(), "a batch is in flight");
    assert_eq!(
        harness.batches().len(),
        formed,
        "a paused scheduler forms nothing"
    );

    // The batches in flight return: the pause is answered at the step that applies them.
    harness.fake.resume();
    harness.step();
    assert!(replied.try_recv().is_ok());
    let generated = harness.tokens(id).len();
    harness.run(5);
    assert_eq!(harness.batches().len(), formed, "still paused");
    assert_eq!(harness.tokens(id).len(), generated);

    harness.apply(Message::Resume);
    harness.settle(100);
    assert_eq!(harness.tokens(id).len(), 20);
}

/// A weight update that keeps the requests in flight lets a running request continue to its end
/// with the state it reached, committing no block from then on, since its state follows neither
/// the old weights nor the new ones; a later request of the same prompt reaches none of the
/// blocks the old weights computed, and commits its own under the update's seeds (SD §2.4
/// `Flush`, §4.3 "What computes a block").
#[test]
fn a_weight_update_keeping_requests_lets_them_run_on_and_commit_nothing_more() {
    let mut harness = Deployment::tokens().build();
    let kept = harness.submit(token(1, 9, 16));
    harness.run_until(30, |harness| harness.tokens(kept).len() >= 2);
    let owner = harness.owner(kept, A).unwrap();
    let before = harness.committed(owner);
    assert!(!before.is_empty(), "its prompt's blocks were committed");

    assert!(flush(&mut harness, update_seeds(), WeightsInFlight::Keep).is_ok());
    harness.settle(100);
    assert_eq!(harness.tokens(kept).len(), 16);
    assert!(harness.finished_with(kept, FinishReason::MaxTokens));
    assert_eq!(
        harness.committed(owner),
        before,
        "nothing committed after the update"
    );

    // The same prompt under the new weights computes from its first token.
    let later = harness.submit(sharing(2, 1, 9, 2));
    harness.settle(100);
    assert_eq!(harness.first_written(later, 0), Some(0));
    let pinned = harness.taken().into_iter().any(|request| match request {
        Request::Pin { keys, .. } => keys.iter().any(|key| before.contains(key)),
        _ => false,
    });
    assert!(!pinned, "no key of the old weights is read");
    // Its prompt's two blocks, under keys the old weights' blocks do not have.
    let renewed = harness.committed(harness.owner(later, A).unwrap());
    assert_eq!(renewed.len(), 2);
    assert!(renewed.iter().all(|key| !before.contains(key)));
}

/// A weight update that recomputes the requests in flight ends every request holding blocks as
/// `Migratable { Drain }`, which the daemon resubmits to be computed again; a request still
/// queued waits and runs under the new weights (SD §2.4 `Flush`).
#[test]
fn a_weight_update_recomputing_requests_ends_those_holding_blocks_as_migratable() {
    let mut harness = Deployment::tokens().slots(1).build();
    let running = harness.submit(token(1, 10, 20));
    harness.run_until(30, |harness| harness.tokens(running).len() >= 2);
    let queued = harness.submit(token(2, 10, 4));
    harness.step();
    assert!(harness.events(queued).is_empty(), "one slot: it waits");

    assert!(flush(&mut harness, update_seeds(), WeightsInFlight::Recompute).is_ok());
    harness.settle(100);
    assert!(
        matches!(
            harness.terminal(running),
            Some(Event::Migratable {
                cause: MigrationCause::Drain,
                ..
            })
        ),
        "{:?}",
        harness.events(running)
    );
    assert!(harness.finished_with(queued, FinishReason::MaxTokens));
    assert_eq!(harness.tokens(queued).len(), 4);
}

/// A weight update that aborts the requests in flight ends every request, running or queued,
/// with `Aborted` at once (SD §2.4 `Flush`).
#[test]
fn a_weight_update_aborting_requests_ends_every_one_aborted() {
    let mut harness = Deployment::tokens().slots(1).build();
    let running = harness.submit(token(1, 10, 20));
    harness.run_until(30, |harness| harness.tokens(running).len() >= 2);
    let queued = harness.submit(token(2, 10, 4));
    harness.step();

    assert!(flush(&mut harness, update_seeds(), WeightsInFlight::Abort).is_ok());
    for id in [running, queued] {
        assert!(
            harness.finished_with(id, FinishReason::Aborted),
            "{:?}",
            harness.events(id)
        );
    }
    harness.settle(100);
    assert!(harness.batches_of(queued).is_empty());
}

/// A flush asks the buffer pool to evict every cached block the scheduler knows of; the blocks a
/// running request still pins stay, and `cache` keeps the keys `Evicted` does not list: a later
/// request reads the blocks that stayed, and computes those evicted (SD §2.4 `Flush`, §3.5
/// `Evicted`).
#[test]
fn a_flush_evicts_every_known_key_and_cache_keeps_the_keys_evicted_does_not_list() {
    let mut harness = Deployment::tokens().build();
    let ended = harness.submit(token(1, 13, 1));
    harness.settle(100);
    let ended_keys = harness.committed(harness.owner(ended, A).unwrap());
    assert_eq!(ended_keys.len(), 3);
    let running = harness.submit(token(2, 9, 40));
    harness.run_until(30, |harness| harness.tokens(running).len() >= 2);
    let running_keys = harness.committed(harness.owner(running, A).unwrap());

    assert!(flush(&mut harness, None, WeightsInFlight::Keep).is_ok());
    harness.run_until(10, |harness| {
        harness
            .answers()
            .iter()
            .any(|(request, _)| matches!(request, Request::Evict { .. }))
    });
    let (asked, evicted) = harness
        .answers()
        .into_iter()
        .find_map(|answer| match answer {
            (Request::Evict { keys }, Completion::Evicted { keys: evicted }) => {
                Some((keys, evicted))
            }
            _ => None,
        })
        .unwrap();
    for key in ended_keys.iter().chain(&running_keys) {
        assert!(asked.contains(key), "every known key is named");
    }
    let mut gone = ended_keys.clone();
    gone.sort_unstable_by_key(|key| key.0);
    let mut listed = evicted.clone();
    listed.sort_unstable_by_key(|key| key.0);
    assert_eq!(listed, gone, "the blocks a request pins stay");

    // The pinned prompt is read; the evicted one is computed again.
    let reads_pinned = harness.submit(sharing(3, 2, 9, 2));
    let reads_evicted = harness.submit(sharing(4, 1, 13, 2));
    harness.settle(200);
    assert_eq!(harness.first_written(reads_pinned, 0), Some(8));
    assert_eq!(harness.first_written(reads_evicted, 0), Some(0));
    let pinned_evicted = harness.taken().into_iter().any(|request| match request {
        Request::Pin { keys, .. } => keys.iter().any(|key| ended_keys.contains(key)),
        _ => false,
    });
    assert!(!pinned_evicted, "`cache` forgot the evicted keys");
}

/// `Resize` is answered at its `Resized`, with the units the region reached, and those units are
/// the region's capacity from then on: a prompt the smaller region can never hold is rejected
/// at its admission, and the same prompt runs once the region is grown back (SD §2.4 `Resize`,
/// §3.5 `Resized`, `never_fits`).
#[test]
fn a_resize_is_replied_at_resized_and_the_regions_capacity_changes() {
    let mut harness = Deployment::tokens().build();
    let kv = harness.region("kv-0");
    let (reply, replied) = bounded(1);
    harness.apply(Message::Resize(kv, 8, reply));
    assert!(replied.try_recv().is_err(), "replied only at `Resized`");
    harness.run_until(10, |_| !replied.is_empty());
    assert_eq!(replied.try_recv(), Ok(8));
    let resized = harness.answers().into_iter().any(|answer| match answer {
        (Request::Resize { region, units: 8 }, Completion::Resized { units: 8, .. }) => {
            region == kv
        }
        _ => false,
    });
    assert!(resized);

    // Ten blocks of prompt, more than the eight units the region holds.
    let refused = harness.submit(token(1, 40, 2));
    harness.settle(50);
    let events = harness.events(refused);
    let [Event::Rejected { kind, message }] = &events[..] else {
        panic!("{events:?}");
    };
    assert_eq!(*kind, RejectionKind::Invalid);
    assert_eq!(message, "request exceeds total KV capacity");

    assert_eq!(harness.resize("kv-0", 256), 256);
    let admitted = harness.submit(token(2, 40, 2));
    harness.settle(100);
    assert_eq!(harness.tokens(admitted).len(), 2);
}
