//! Pause, flush and resize (SD §8.1): the operations the engine runs at a batch boundary (SD
//! §2.4, §3.1). `Pause` forms no batch and is answered once nothing is in flight; `Flush` evicts
//! every cached block the scheduler knows of while the requests it holds keep their state;
//! `Resize` is answered at its `Resized` with the units the region reached, which are its
//! capacity from then on, and the units it asked for bound the region: a shrink that stops short
//! goes on as the units held past them come free.

use crossbeam_channel::{Receiver, bounded};
use engine::scheduler::{Message, Output};
use proto::request::{Event, RejectionKind};
use storage::runtime::{Completion, Request};

use crate::harness::{A, Deployment, Harness, sharing, token};

/// Sends `Flush` and waits for its reply, which the scheduler sends at once.
pub(crate) fn flush(harness: &mut Harness) {
    let (reply, replied) = bounded(1);
    harness.apply(Message::Flush(reply));
    replied.try_recv().unwrap();
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

    flush(&mut harness);
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

/// A grow that stops short leaves the region declared to hold the units the engine resized it to
/// (SD §2.4 `Resize`, §3.5 `never_fits`): a request those units hold waits at the head of the
/// queue rather than being refused against the units the region reached, and runs once the
/// memory the grow waits for is free and the engine resizes the region again; a request beyond
/// the declared units is refused at once all the same. This is a rollback's promotion: the epoch
/// it returns to gave its memory to a target of another layout, grows its regions while the
/// target still holds that memory, and grows them again as the target's drain gives it back.
#[test]
fn a_grow_stopped_short_holds_requests_until_the_region_reaches_its_declared_units() {
    // Eight units of KV region, in chunks of eight tokens (two blocks).
    let mut harness = Deployment::tokens().kv_units(8).chunk(8).build();

    // The region gives its memory back to a target of another layout, and the engine grows it
    // again while the target holds that memory.
    assert_eq!(harness.resize("kv-0", 0), 0);
    let target = harness.beside("kv-0", 8);
    assert_eq!(harness.resize("kv-0", 8), 0, "the grow stops short");

    // Ten blocks of prompt, more than the eight declared units; three blocks, which they hold.
    let refused = harness.submit(token(1, 40, 2));
    let held = harness.submit(token(2, 12, 2));
    harness.run(50);
    let events = harness.events(refused);
    let [Event::Rejected { kind, message }] = &events[..] else {
        panic!("{events:?}");
    };
    assert_eq!(*kind, RejectionKind::Invalid);
    assert_eq!(message, "request exceeds total KV capacity");
    let events = harness.events(held);
    assert!(
        events.is_empty(),
        "the request waits for the region: {events:?}"
    );

    // The target's drain gives the memory back, and the engine resizes the region again.
    harness.elsewhere(&[Request::Resize {
        region: target,
        units: 0,
    }]);
    assert_eq!(harness.resize("kv-0", 8), 8);
    harness.settle(100);
    assert_eq!(harness.tokens(held).len(), 2);
}

/// The units a `Resize` asks for bound the region from then on, whatever its shrink reaches (SD
/// §2.4 `Resize`): a shrink that a running request's blocks stop short of goes on by itself as
/// they come free, the scheduler asking the buffer pool again until the region reaches those
/// units, with no further `Message::Resize` and no `Shrunk`; meanwhile no allocation takes the
/// units held past the bound, and a request that needs more than the bound is refused at its
/// admission, since the bound is what the region is declared to hold.
#[test]
fn a_shrink_stopped_short_goes_on_as_the_units_held_past_its_bound_come_free() {
    let mut harness = Deployment::tokens().build();
    let kv = harness.region("kv-0");

    // Ten blocks of prompt and room to decode: the request holds more than the four units the
    // region is resized to while it runs, so the shrink stops short of its blocks.
    let running = harness.submit(token(1, 40, 30));
    harness.run_until(20, |harness| !harness.tokens(running).is_empty());
    let reached = harness.resize("kv-0", 4);
    assert!(
        reached > 4,
        "the shrink stops short of the held blocks: {reached}"
    );

    // Six blocks of prompt: fewer than the region holds after the first shrink, more than its
    // bound.
    let refused = harness.submit(token(2, 24, 2));
    harness.settle(200);

    assert!(
        harness.terminal(running).is_some(),
        "{:?}",
        harness.events(running)
    );
    let events = harness.events(refused);
    let [Event::Rejected { kind, message }] = &events[..] else {
        panic!("{events:?}");
    };
    assert_eq!(*kind, RejectionKind::Invalid);
    assert_eq!(message, "request exceeds total KV capacity");

    // The scheduler asks again by itself as the blocks come free, until an answer reaches the
    // bound.
    let resized = |harness: &Harness| -> Vec<u32> {
        harness
            .answers()
            .into_iter()
            .filter_map(|answer| match answer {
                (Request::Resize { region, units: 4 }, Completion::Resized { units, .. })
                    if region == kv =>
                {
                    Some(units)
                }
                _ => None,
            })
            .collect()
    };
    harness.run_until(50, |harness| resized(harness).last() == Some(&4));
    let resized = resized(&harness);
    assert!(resized.len() > 1, "{resized:?}");
    assert_eq!(resized.last(), Some(&4), "{resized:?}");
    assert!(
        !harness
            .outputs()
            .iter()
            .any(|output| matches!(output, Output::Shrunk(..)))
    );
}
