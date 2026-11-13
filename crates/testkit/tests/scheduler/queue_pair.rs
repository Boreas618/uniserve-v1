//! Queue pair (SD §8.1): the scheduler's end of its queue pair with the buffer pool (SD §3.1,
//! §3.5, §4.1; `specs/buffer-pool.md` §9, §11). Admissions that wait for answers let later ones
//! pass; a request whose blocks have not arrived is left out; requests are held and merged when
//! the queues have no room; each completion is matched to its request by its number, dropped when
//! its owner is gone or its entries were given up, and handed to the executors before a batch
//! reads it; requests that name entries reach the buffer pool only with every stamp; copies end
//! with one `Finish`; refusals and `Full` answers are applied by kind.
//!
//! The case of completions delayed, reordered and withheld with every request ending is L6's
//! (`l6.rs`). Every step of every case checks the scheduler's side of A1 to A4.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use engine::scheduler::Message;
use ipc::CacheGroupKind;
use ipc::batch::Phase;
use proto::request::{
    Event, FinishReason, GenerationRequest, MigrationCause, RequestId, Trajectory,
};
use storage::Key;
use storage::runtime::{Cause, Completion, OwnerId, Request, Tier};

use crate::harness::{
    A, B, BLOCK, Deployment, Harness, STEPS, VIDEO_PHASES, chat_image, prompt, token, video,
};

/// The requests the buffer pool took, each with its number, in the order taken.
fn numbered(harness: &Harness) -> Vec<(u64, Request)> {
    let taken = harness.pool.requests().iter();
    taken
        .map(|(_, id, request)| (id.0, request.clone()))
        .collect()
}

/// The completion the buffer pool wrote for the request it took under number `id`.
fn answer_to(harness: &Harness, id: u64) -> Option<Completion> {
    let mut written = harness.pool.completions().iter();
    written
        .find(|(_, answered, _)| answered.0 == id)
        .map(|(_, _, completion)| completion.clone())
}

/// Token request `id` that may generate `max` tokens, whose prompt is request `of`'s prompt of
/// `tokens` tokens: it reads the blocks that request committed.
fn sharing(id: u64, of: u64, tokens: u32, max: usize) -> GenerationRequest {
    GenerationRequest {
        prompt_token_ids: prompt(of, tokens),
        ..token(id, tokens, max)
    }
}

/// The first cache position request `id`'s first prefill wrote: where its computation started.
fn first_written(harness: &Harness, id: RequestId) -> Option<i32> {
    let prefill = harness
        .batches_of(id)
        .into_iter()
        .find(|batch| batch.phase == Phase::Prefill)?;
    let row = prefill.requests.iter().position(|of| *of == id)?;
    let rows = &prefill.columns.rows;
    let call = rows.calls.get(row)?;
    let first = *rows.query_offsets.get(call.rows.start as usize)?;
    rows.cache_positions
        .get(usize::try_from(first).ok()?)
        .copied()
}

/// The cache positions the rows of request `id` wrote, over its batches in order.
fn row_positions(harness: &Harness, id: RequestId) -> Vec<i32> {
    let mut positions = Vec::new();
    for batch in harness.batches_of(id) {
        let Some(row) = batch.requests.iter().position(|of| *of == id) else {
            continue;
        };
        let rows = &batch.columns.rows;
        let Some(call) = rows.calls.get(row) else {
            continue;
        };
        let start = rows.query_offsets[call.rows.start as usize] as usize;
        let end = rows.query_offsets[call.rows.end as usize] as usize;
        positions.extend_from_slice(&rows.cache_positions[start..end]);
    }
    positions
}

/// Sends `Message::Resize` of the session's region `name` to `units` and steps until its reply.
fn resize(harness: &mut Harness, name: &str, units: u32) {
    let region = harness.region(name);
    let (reply, replied) = crossbeam_channel::bounded(1);
    harness.apply(Message::Resize(region, units, reply));
    harness.run_until(20, |_| replied.try_recv().is_ok());
}

/// A deployment with a host tier where request 1, of a prompt of 5 tokens, committed its one
/// full block and stored it on the host, whose device copy a shrink of the KV region then
/// evicted: a request of the same prompt finds that block on the host alone.
fn host_hit() -> Harness {
    let mut harness = Deployment::tokens().host(64).build();
    harness.submit(token(1, 5, 1));
    harness.settle(100);
    let stored = |harness: &Harness| {
        harness.answers().iter().any(|answer| {
            matches!(
                answer,
                (Request::Finish { ok: true, .. }, Completion::Stored { .. })
            )
        })
    };
    harness.run_until(20, |harness| stored(harness));
    resize(&mut harness, "kv-0", 0);
    resize(&mut harness, "kv-0", 256);
    harness
}

/// The number the buffer pool took request `id`'s first `Load` under.
fn load_of(harness: &Harness, owner: Option<OwnerId>) -> Option<u64> {
    numbered(harness)
        .into_iter()
        .find_map(|(number, request)| match request {
            Request::Load { owner: of, .. } if owner.is_none_or(|owner| owner == of) => {
                Some(number)
            }
            _ => None,
        })
}

#[test]
fn a_later_request_is_admitted_past_one_waiting_for_its_pinned() {
    let mut harness = Deployment::tokens().build();
    harness.submit(token(1, 12, 2));
    harness.settle(100);

    // Request 2 reads request 1's two blocks within its reuse limit: its round pins them.
    harness
        .pool
        .withhold(|request, _| matches!(request, Request::Pin { .. }));
    let waiting = harness.submit(sharing(2, 1, 12, 3));
    harness.run_until(10, |harness| harness.pool.withheld() == 1);
    let passing = harness.submit(token(3, 6, 3));
    harness.run_until(50, |harness| harness.terminal(passing).is_some());
    assert!(
        harness.events(waiting).is_empty(),
        "its `Pinned` is withheld"
    );

    harness.pool.release();
    harness.settle(100);
    assert_eq!(harness.tokens(waiting).len(), 3);
    assert_eq!(
        first_written(&harness, waiting),
        Some(8),
        "it runs from its hit"
    );
}

/// A `Full` to an admission returns the request to the waiting queue at its place, placed
/// nowhere and detached: its owner's `Detach` reaches the buffer pool, its next admission draws a
/// new owner, and it is admitted before a request that arrived after it (SD §3.5 `Full`).
#[test]
fn a_full_to_an_admission_returns_the_request_to_its_place_detached() {
    // A queue three batches deep keeps room for steps beside the batches held on the device: a
    // step whose groups' queues are all full asks the policy nothing (SD §3.4).
    let mut harness = Deployment::tokens().kv_units(10).queue_depth(3).build();
    let kv = harness.region("kv-0");

    // Request 1 holds seven of the ten units while its prefill is on the device.
    harness.fake.hold();
    let first = harness.submit(token(1, 20, 2));
    harness.run_until(10, |harness| harness.fake.held() >= 1);

    // The capacity words say the region has room; the buffer pool has not.
    harness.pool.set_capacity(kv, Some((100, 0, 0)));
    harness.step();
    let returned = harness.submit(token(2, 12, 2));
    harness.step();
    harness.pool.set_capacity(kv, None);
    let full = |harness: &Harness| {
        harness
            .answers()
            .into_iter()
            .find_map(|answer| match answer {
                (Request::Allocate { owner, .. }, Completion::Full { .. }) => Some(owner),
                _ => None,
            })
    };
    harness.run_until(10, |harness| full(harness).is_some());
    let owner = full(&harness).unwrap();
    // Request 3 needs as many units as request 2, which the region holds beside it.
    let later = harness.submit(token(3, 12, 2));
    harness.run(5);
    assert!(harness.events(returned).is_empty());

    harness.fake.resume();
    harness.settle(200);
    let taken = numbered(&harness);
    let detached = taken
        .iter()
        .any(|(_, request)| matches!(request, Request::Detach { owner: of, .. } if *of == owner));
    assert!(detached, "the returned admission's owner is detached");
    let owners: Vec<OwnerId> = taken
        .iter()
        .filter_map(|(_, request)| match request {
            Request::Attach { owner, .. } => Some(*owner),
            _ => None,
        })
        .collect();
    let unique: HashSet<OwnerId> = owners.iter().copied().collect();
    assert_eq!(
        unique.len(),
        owners.len(),
        "every admission draws a new owner"
    );
    assert!(owners.len() >= 4, "request 2 was admitted again");

    let scheduled = |id: RequestId| {
        let mut outputs = harness.outputs().iter();
        outputs.position(|output| {
            matches!(output, engine::scheduler::Output::Event(of, Event::Scheduled { .. }) if *of == id)
        })
    };
    assert!(scheduled(returned) < scheduled(later), "it keeps its place");
    assert!(harness.terminal(first).is_some());
    for id in [returned, later] {
        assert_eq!(harness.tokens(id).len(), 2, "{}", harness.summary());
    }
}

/// A running request whose `Allocate` has not been answered is left out of the steps once its
/// blocks no longer cover its next call, while another request runs; it is scheduled again once
/// its `Allocated` arrives, from the position it stopped at.
#[test]
fn a_request_whose_blocks_have_not_arrived_is_left_out_until_its_allocated_arrives() {
    let mut harness = Deployment::tokens().build();
    let starved = harness.submit(token(1, 5, 30));
    let other = harness.submit(token(2, 5, 30));
    harness.run_until(20, |harness| harness.tokens(starved).len() >= 2);
    let owner = harness.owner(starved, A).unwrap();
    harness.pool.withhold(
        move |request, _| matches!(request, Request::Allocate { owner: of, .. } if *of == owner),
    );

    let mut left_out = 0;
    for _ in 0..40 {
        let before = harness.batches().len();
        harness.step();
        let formed = harness.batches().split_off(before);
        let carries = |id| formed.iter().any(|batch| batch.requests.contains(&id));
        if carries(starved) {
            left_out = 0;
        } else if carries(other) {
            left_out += 1;
        }
        if left_out == 5 {
            break;
        }
    }
    assert_eq!(left_out, 5, "left out while the other request ran");
    let stopped = row_positions(&harness, starved).last().copied().unwrap();

    let before = row_positions(&harness, starved).len();
    harness.pool.release();
    harness.run_until(10, |harness| row_positions(harness, starved).len() > before);
    let resumed = row_positions(&harness, starved).split_off(before);
    assert_eq!(resumed[0], stopped + 1, "it continues where it stopped");
    harness.settle(200);
    assert_eq!(harness.tokens(starved).len(), 30);
    assert_eq!(harness.tokens(other).len(), 30);
}

/// Each class's blocks asked for cover a running request's next call and one block beyond it,
/// and no more than that (SD §4.4 "The low watermark", §3.4 `allocate_ahead`): after every step,
/// the `Allocate`s of each class the buffer pool took cover the next call of the request's last
/// decode and one block, and asked for one more block only when they did not.
#[test]
fn allocates_keep_each_class_one_block_ahead_of_the_next_call() {
    let window = CacheGroupKind::SlidingWindow { window: 8 };
    let mut harness = Deployment::tokens()
        .cache(&[(CacheGroupKind::Full, BLOCK), (window, BLOCK)])
        .queue_depth(1)
        .build();
    let id = harness.submit(token(1, 6, 24));
    harness.run_until(10, |harness| !harness.batches_of(id).is_empty());
    let owner = harness.owner(id, A).unwrap();

    // Per class, the end of the positions asked for, from the `Allocate`s the buffer pool took.
    let asked = |harness: &Harness, class: u8| {
        let taken = numbered(harness).into_iter();
        taken
            .filter_map(|(_, request)| match request {
                Request::Allocate {
                    owner: of,
                    class: asked,
                    at,
                    count,
                    ..
                } if of == owner && asked == class => Some(at + count),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    };
    while harness.terminal(id).is_none() {
        let written = harness.batches_of(id).into_iter().last().unwrap();
        let last = *written.columns.rows.cache_positions.last().unwrap() as u32;
        let decoding = written.phase == Phase::Decode && harness.tokens(id).len() < 23;
        // The step that formed the batch asked; the next turn takes what it asked.
        harness.step();
        harness.step();
        if decoding {
            // The next call writes `last + 1`: its blocks, and one block beyond them.
            let cover = (last + 2).div_ceil(BLOCK) + 1;
            for class in [0, 1] {
                assert_eq!(
                    asked(&harness, class),
                    cover,
                    "class {class} after position {last}"
                );
            }
        }
    }
    harness.settle(100);
    assert_eq!(harness.tokens(id).len(), 24);
}

/// While the buffer pool takes nothing, its submission queue fills: the scheduler keeps
/// stepping its requests within the blocks they hold and holds its requests, merging a further
/// need of an owner and class into its held `Allocate`; once the buffer pool serves again, at
/// most one `Allocate` of an owner and class is ever unanswered (SD §3.1; BP §10, F5).
#[test]
fn a_full_submission_queue_holds_requests_with_one_allocate_per_owner_and_class() {
    let mut harness = Deployment::tokens().ring_bytes(256).build();
    let ids: Vec<RequestId> = (1..=4).map(|id| harness.submit(token(id, 5, 40))).collect();
    harness.run_until(30, |harness| {
        ids.iter().all(|id| harness.tokens(*id).len() >= 2)
    });

    harness.pool.stop(true);
    let taken = harness.pool.requests().len();
    let batches = harness.batches().len();
    harness.run(30);
    assert_eq!(
        harness.pool.requests().len(),
        taken,
        "a stopped pool takes nothing"
    );
    assert!(harness.batches().len() > batches, "the steps went on");

    harness.pool.stop(false);
    let restarted = harness.pool.requests().len();
    let unanswered = |harness: &Harness| {
        let written: HashSet<u64> = harness
            .pool
            .completions()
            .iter()
            .map(|(_, id, _)| id.0)
            .collect();
        let mut open: HashMap<(OwnerId, u8), u32> = HashMap::new();
        for (number, request) in numbered(harness) {
            if let Request::Allocate { owner, class, .. } = request
                && !written.contains(&number)
            {
                *open.entry((owner, class)).or_default() += 1;
            }
        }
        open
    };
    for _ in 0..300 {
        if ids.iter().all(|id| harness.terminal(*id).is_some()) {
            break;
        }
        harness.step();
        let open = unanswered(&harness);
        assert!(open.values().all(|count| *count <= 1), "{open:?}");
    }
    harness.settle(100);
    let merged = numbered(&harness)
        .into_iter()
        .skip(restarted)
        .any(|(_, request)| matches!(request, Request::Allocate { count, .. } if count > 1));
    assert!(merged, "a further need raised a held `Allocate`");
    for id in &ids {
        assert_eq!(harness.tokens(*id).len(), 40);
    }
}

/// The scheduler writes a request only while the completion queue has room for the answers of
/// every request outstanding: while the buffer pool withholds every answer, it stops writing;
/// once released, every withheld answer fits the completion queue at once (SD §3.1; BP §9).
#[test]
fn nothing_is_written_while_the_completion_queue_lacks_room_for_every_answer() {
    const RING: usize = 512;
    let mut harness = Deployment::tokens().ring_bytes(RING).build();
    let before = harness.pool.requests().len();
    harness.pool.withhold(|_, _| true);
    for id in 1..=8 {
        harness.submit(token(id, 5, 4));
    }
    harness.run(10);
    let outstanding: Vec<(u64, Request)> = numbered(&harness).split_off(before);
    assert!(!outstanding.is_empty());
    assert!(
        outstanding.len() < 16,
        "eight admissions of two requests each were not all written"
    );

    // The answers withheld fall due together at the next turn.
    harness.pool.release();
    harness.pool.turn().unwrap();
    let bytes: usize = outstanding
        .iter()
        .map(|(number, _)| {
            let completion = answer_to(&harness, *number).unwrap();
            let mut record = Vec::new();
            completion.encode(storage::runtime::RequestId(*number), &mut record);
            record.len() + 8
        })
        .sum();
    assert!(
        bytes <= RING,
        "{bytes} bytes of answers in a ring of {RING}"
    );
}

/// A completion for an owner the scheduler gave up is dropped: a request cancelled while its
/// admission's answers were withheld has its owner detached, and its answers reach no executor
/// (SD §3.5 `answered`).
#[test]
fn a_completion_for_a_forgotten_owner_is_dropped() {
    let mut harness = Deployment::tokens().build();
    harness.pool.withhold(|_, _| true);
    let cancelled = harness.submit(token(1, 5, 4));
    let attached = |harness: &Harness| {
        harness
            .taken()
            .into_iter()
            .find_map(|request| match request {
                Request::Attach { owner, .. } => Some(owner),
                _ => None,
            })
    };
    harness.run_until(10, |harness| attached(harness).is_some());
    let owner = attached(&harness).unwrap();
    harness.apply(Message::Cancel(cancelled, FinishReason::Cancelled));
    harness.pool.release();
    harness.run(10);

    let mapped = harness
        .fake
        .mapped()
        .into_iter()
        .any(|(.., request, _)| match request {
            Request::Attach { owner: of, .. } | Request::Allocate { owner: of, .. } => of == owner,
            _ => false,
        });
    assert!(
        !mapped,
        "an answer of the forgotten owner reached an executor"
    );
    let detached = harness
        .taken()
        .into_iter()
        .any(|request| matches!(request, Request::Detach { owner: of, .. } if of == owner));
    assert!(detached);

    let next = harness.submit(token(2, 5, 4));
    harness.settle(100);
    assert_eq!(harness.tokens(next).len(), 4);
    assert_eq!(harness.failure(), None);
}

/// A request placed on two groups is attached to each by an owner of its own, and each group's
/// batches name the request by the slot its `Attached` gave there.
#[test]
fn each_placed_group_gets_an_owner_and_the_slot_its_attached_gave() {
    let mut harness = Deployment::tokens().encoder().build();
    let id = harness.submit(chat_image(1, 12, 6, 3));
    harness.settle(100);

    let slots: HashMap<storage::runtime::GroupId, (OwnerId, u32)> = harness
        .fake
        .mapped()
        .into_iter()
        .filter_map(
            |(group, _, _, request, completion)| match (request, completion) {
                (Request::Attach { owner, group: of }, Completion::Attached { slot })
                    if of == group =>
                {
                    Some((group, (owner, slot)))
                }
                _ => None,
            },
        )
        .collect();
    assert_eq!(slots.len(), 2, "{slots:?}");
    assert_ne!(slots[&A].0, slots[&B].0, "one owner per group");

    for batch in harness.batches_of(id) {
        let row = batch.requests.iter().position(|of| *of == id).unwrap();
        let slot = match batch.phase {
            Phase::VisionEncode => batch.media[row].slot,
            _ => batch.columns.rows.calls[row].slot,
        };
        assert_eq!(
            slot, slots[&batch.group].1,
            "{:?} on group {}",
            batch.phase, batch.group.0
        );
    }
    assert_eq!(harness.tokens(id).len(), 3);
}

/// Every completion is matched to its request by its number, whatever the order the buffer pool
/// writes them in: with every answer held over windows of steps and written in reverse, each
/// answer reaches the executors with the request it answers, and every request ends.
#[test]
fn completions_in_any_order_are_matched_to_their_requests_by_number() {
    let mut harness = Deployment::tokens().build();
    harness.reorder(Some(4));
    let ids: Vec<RequestId> = (1..=4).map(|id| harness.submit(token(id, 9, 8))).collect();
    harness.settle(500);
    reversed_and_ended(&harness, &ids);
}

/// The completions of requests the buffer pool defers to their stamps (`Commit`, `Release`,
/// `Detach`) arrive in another order than the requests (BP §9 "Order"): withheld over several
/// steps, they are written after the answers of requests taken after them; each is matched to
/// its request by its number, and every request ends.
#[test]
fn completions_of_stamped_requests_in_another_order_are_matched_by_number() {
    let mut harness = Deployment::tokens().build();
    harness.pool.withhold(|request, _| {
        matches!(
            request,
            Request::Commit { .. } | Request::Release { .. } | Request::Detach { .. }
        )
    });
    let ids: Vec<RequestId> = (1..=4).map(|id| harness.submit(token(id, 9, 8))).collect();
    harness.run_until(30, |harness| harness.pool.withheld() >= 4);
    harness.run(5);
    harness.pool.release();
    harness.settle(200);
    reversed_and_ended(&harness, &ids);
}

/// The buffer pool wrote some completion before that of a request it took earlier; every
/// completion the executors were handed is the buffer pool's answer to the request handed with
/// it (the fake's record, checked at every step); each of `ids` generated all its tokens.
fn reversed_and_ended(harness: &Harness, ids: &[RequestId]) {
    let order: HashMap<u64, usize> = numbered(harness)
        .iter()
        .enumerate()
        .map(|(at, (number, _))| (*number, at))
        .collect();
    let written: Vec<usize> = harness
        .pool
        .completions()
        .iter()
        .filter_map(|(_, id, _)| order.get(&id.0).copied())
        .collect();
    assert!(
        written.windows(2).any(|pair| pair[0] > pair[1]),
        "every completion was written in the order of its request"
    );
    for id in ids {
        assert_eq!(harness.tokens(*id).len(), 8);
    }
    assert_eq!(harness.failure(), None);
}

/// Every completion that gives entries is handed to the executors with the request it answers,
/// and before any batch reads the entries: every block a batch's rows read, the cached prefix a
/// request pinned included, is in the executors' tables under the row's slot when the batch is
/// formed (SD §3.5 `answered`; BP §9, obligation 1).
#[test]
fn every_completion_with_entries_is_mapped_before_a_batch_reads_them() {
    let mut harness = Deployment::tokens().build();
    harness.submit(token(1, 13, 3));
    harness.settle(100);
    let ids = [
        harness.submit(sharing(2, 1, 13, 6)),
        harness.submit(token(3, 10, 6)),
    ];

    for _ in 0..100 {
        let before = harness.batches().len();
        harness.step();
        for batch in harness.batches().split_off(before) {
            let entries = harness.fake.entries(batch.group);
            let rows = &batch.columns.rows;
            for call in &rows.calls {
                let tokens = rows.query_offsets[call.rows.start as usize] as usize
                    ..rows.query_offsets[call.rows.end as usize] as usize;
                let end = rows.cache_positions[tokens]
                    .iter()
                    .max()
                    .copied()
                    .unwrap_or(0);
                for block in 0..=(end as u32 / BLOCK) {
                    assert!(
                        entries.contains_key(&(call.slot, 0, block)),
                        "batch {} reads block {block} of slot {}, which no answer gave",
                        batch.id.0,
                        call.slot
                    );
                }
            }
        }
        if ids.iter().all(|id| harness.terminal(*id).is_some()) {
            break;
        }
    }
    harness.settle(100);

    // Each answer with entries was handed over with its request and number.
    let mapped: HashSet<(u64, String)> = harness
        .fake
        .mapped()
        .into_iter()
        .map(|(_, _, id, request, completion)| (id.0, format!("{request:?} {completion:?}")))
        .collect();
    for (number, request) in numbered(&harness) {
        let completion = answer_to(&harness, number).unwrap();
        if matches!(
            completion,
            Completion::Attached { .. } | Completion::Allocated { .. } | Completion::Pinned { .. }
        ) {
            assert!(
                mapped.contains(&(number, format!("{request:?} {completion:?}"))),
                "{request:?} was answered {completion:?} and not mapped"
            );
        }
    }
}

/// `Release`, `Detach`, `Commit` and `Finish` reach the buffer pool with the stamp of every group
/// whose tables hold their entries: a chat request's encoder output, written by the encoding group
/// and read by the token group, is committed and given up with a stamp of each; its token blocks
/// with the token group's alone (SD §3.1, §4.2; BP §9, obligation 2).
#[test]
fn stamped_requests_carry_the_stamp_of_every_group_holding_their_entries() {
    let mut harness = Deployment::tokens().encoder().build();
    let id = harness.submit(chat_image(1, 12, 6, 3));
    harness.settle(100);
    assert_eq!(harness.tokens(id).len(), 3);

    let owners: HashMap<storage::runtime::GroupId, OwnerId> = harness
        .taken()
        .into_iter()
        .filter_map(|request| match request {
            Request::Attach { owner, group } => Some((group, owner)),
            _ => None,
        })
        .collect();
    let groups = |after: &storage::runtime::Stamps| {
        let mut groups: Vec<u8> = after.iter().map(|(group, _)| group.0).collect();
        groups.sort_unstable();
        groups
    };
    // The token group holds entries of both owners: its own, and the encoder output it reads;
    // the encoding group holds the encoder output's.
    let mut encoder_output = 0;
    let mut detached = HashSet::new();
    for request in harness.taken() {
        let (owner, after) = match &request {
            Request::Detach { owner, after } | Request::Commit { owner, after, .. } => {
                (*owner, after)
            }
            Request::Release { owner, after, .. } => (*owner, after),
            _ => continue,
        };
        let stamps = groups(after);
        assert!(stamps.contains(&A.0), "{request:?}");
        if owner == owners[&B] {
            assert_eq!(stamps, [A.0, B.0], "{request:?}");
            encoder_output += 1;
        }
        if let Request::Detach { owner, .. } = request {
            detached.insert(owner);
        }
    }
    assert_eq!(detached, owners.values().copied().collect::<HashSet<_>>());
    assert_eq!(
        encoder_output, 2,
        "the encoder output's `Commit` and its owner's `Detach`"
    );
}

/// An admission whose hit is shallower than `cache` expected gives up, by `Release`, the
/// read-only entries beyond the hit its answers support, and asks for those positions again
/// with an `Allocate` that the buffer pool takes after that `Release` (SD §3.1, §4.4). Here the
/// recurrent state's checkpoint at the prompt's boundary is gone, so the hit is empty.
#[test]
fn an_allocate_at_positions_a_release_frees_is_taken_after_that_release() {
    let mut harness = Deployment::tokens()
        .cache(&[
            (CacheGroupKind::Full, BLOCK),
            (CacheGroupKind::State, BLOCK),
        ])
        .build();
    harness.submit(token(1, 9, 1));
    harness.settle(100);
    let checkpoint = harness
        .taken()
        .into_iter()
        .find_map(|request| match request {
            Request::Commit { class: 1, key, .. } => Some(key),
            _ => None,
        });
    let evicted = harness.elsewhere(&[Request::Evict {
        keys: vec![checkpoint.unwrap()],
    }]);
    assert!(matches!(&evicted[..], [Completion::Evicted { keys }] if keys.len() == 1));

    let id = harness.submit(sharing(2, 1, 9, 3));
    harness.settle(100);
    let taken = numbered(&harness);
    let owner = harness.owner(id, A).unwrap();
    let released = taken.iter().position(|(_, request)| {
        matches!(request, Request::Release { owner: of, class: 0, at: 0, .. } if *of == owner)
    });
    let allocated = taken.iter().position(|(_, request)| {
        matches!(request, Request::Allocate { owner: of, class: 0, at: 0, .. } if *of == owner)
    });
    assert!(released.is_some() && released < allocated, "{taken:#?}");
    assert_eq!(first_written(&harness, id), Some(0));
    assert_eq!(harness.tokens(id).len(), 3);
}

/// A `Commit` the buffer pool takes while it waits for its stamps, followed by its request's
/// `Detach`, still commits the block (BP §9, R6): a later request of the same prompt finds it.
#[test]
fn a_commit_followed_by_its_requests_detach_still_commits() {
    // A queue three batches deep keeps room for steps beside the batches held on the device.
    let mut harness = Deployment::tokens().queue_depth(3).build();
    harness.fake.hold();
    // Request 1's prompt fills one block; it ends with the token its prefill samples.
    let first = harness.submit(token(1, 4, 1));
    harness.run_until(10, |harness| harness.fake.held() >= 1);
    // A later batch is still on the device when request 1's commit and detach are stamped.
    harness.submit(token(2, 6, 1));
    harness.run_until(10, |harness| harness.fake.held() >= 2);
    harness.fake.release(1);
    harness.run_until(10, |harness| harness.terminal(first).is_some());

    let commit = |harness: &Harness| {
        numbered(harness)
            .into_iter()
            .find_map(|(number, request)| match request {
                Request::Commit { key, .. } => Some((number, key)),
                _ => None,
            })
    };
    let detach = |harness: &Harness| {
        numbered(harness).into_iter().find_map(|(number, request)| {
            matches!(request, Request::Detach { .. }).then_some(number)
        })
    };
    harness.run_until(10, |harness| {
        commit(harness).is_some() && detach(harness).is_some()
    });
    let (number, key) = commit(&harness).unwrap();
    assert!(
        detach(&harness) > Some(number),
        "the detach is taken after the commit"
    );
    assert_eq!(
        answer_to(&harness, number),
        None,
        "the commit waits for its stamp"
    );

    harness.fake.resume();
    harness.settle(100);
    assert_eq!(answer_to(&harness, number), Some(Completion::Committed));
    // The reader's prompt continues request 1's: its reuse limit reaches the full block.
    let mut reader = token(3, 9, 2);
    reader.prompt_token_ids = prompt(1, 4).into_iter().chain(prompt(3, 5)).collect();
    let reader = harness.submit(reader);
    harness.settle(100);
    let pinned = harness.answers().into_iter().any(|answer| match answer {
        (Request::Pin { keys, .. }, Completion::Pinned { runs, .. }) => {
            keys.first() == Some(&key) && runs.len() == 1
        }
        _ => false,
    });
    assert!(pinned, "the committed block is found");
    assert_eq!(harness.tokens(reader).len(), 2);
}

/// Every `Copy` is answered with a `Finish`: the stores the write policy asks for, run by the
/// group's executor with its next batch, end with the executor's `Finish { ok: true }` (SD §3.5
/// `answered`; BP §11).
#[test]
fn every_copy_ends_with_the_executors_finish() {
    let harness = host_hit();
    let copies: Vec<u64> = harness
        .pool
        .completions()
        .iter()
        .filter(|(_, _, completion)| matches!(completion, Completion::Copy { .. }))
        .map(|(_, id, _)| id.0)
        .collect();
    assert!(!copies.is_empty(), "the stores were answered `Copy`");
    for copy in copies {
        assert!(
            finished(&harness, copy, true),
            "copy {copy} was not finished"
        );
    }
}

/// Whether the buffer pool took a `Finish` of copy `copy` with `ok`.
fn finished(harness: &Harness, copy: u64, ok: bool) -> bool {
    harness.taken().into_iter().any(|request| {
        matches!(request, Request::Finish { copy: of, ok: was, .. } if of.0 == copy && was == ok)
    })
}

/// The `Copy` of a request cancelled while it was withheld is given up with `Finish { ok: false }`
/// through its group's executor, and the request's owner is detached (SD §3.5 `answered`; BP §11
/// "Deadlines").
#[test]
fn a_dropped_copy_is_given_up_with_a_finish_that_fails_it() {
    let mut harness = host_hit();
    // Request 2's `Load` of the block on the host is answered `Copy` after it was cancelled.
    harness
        .pool
        .withhold(|request, _| matches!(request, Request::Load { .. }));
    let cancelled = harness.submit(sharing(2, 1, 5, 2));
    harness.run_until(10, |harness| harness.pool.withheld() == 1);
    harness.apply(Message::Cancel(cancelled, FinishReason::Cancelled));
    harness.pool.release();
    harness.settle(100);

    let load = load_of(&harness, None).unwrap();
    assert!(matches!(
        answer_to(&harness, load),
        Some(Completion::Copy { .. })
    ));
    // The give-up is stamped by the copy's executor first (SD §3.1), so it reaches the buffer
    // pool a pass or two after the request ended.
    harness.run_until(5, |harness| finished(harness, load, false));
    let stamped = harness.fake.stamps().into_iter().any(|(group, request)| {
        group == A && matches!(request, Request::Finish { copy, ok: false, .. } if copy.0 == load)
    });
    assert!(stamped, "the give-up passed through the copy's group");
    let next = harness.submit(token(3, 6, 2));
    harness.settle(100);
    assert_eq!(harness.tokens(next).len(), 2);
}

/// A resumed video request whose trajectory the buffer pool does not hold (`Refused` with the
/// cause `key`) starts from noise: it denoises every step and completes (SD §3.5 `Refused`,
/// §3.8 "Resume").
#[test]
fn a_refused_key_of_a_restore_starts_the_video_from_noise() {
    let mut harness = Deployment::video(&[&VIDEO_PHASES]).build();
    let mut resumed = video(1, 1);
    resumed.trajectory = Trajectory {
        id: 7,
        denoised_steps: 2,
        ..Trajectory::default()
    };
    let id = harness.submit_video(resumed);
    harness.settle(200);

    let refused = harness.answers().into_iter().any(|answer| {
        matches!(
            answer,
            (
                Request::Restore { .. },
                Completion::Refused { cause: Cause::Key }
            )
        )
    });
    assert!(refused);
    let steps: Vec<u32> = harness
        .batches_of(id)
        .into_iter()
        .filter(|batch| batch.phase == Phase::Denoise)
        .map(|batch| batch.media[0].step)
        .collect();
    assert_eq!(steps, (0..STEPS).collect::<Vec<_>>());
    assert!(matches!(
        harness.terminal(id),
        Some(Event::Finished {
            reason: FinishReason::Completed,
            ..
        })
    ));
}

/// A `Load` refused with the cause `class` (the host copy was written by a layout whose encoding
/// describes other tensors) is a miss: the request computes the block (SD §3.5 `Refused`).
#[test]
fn a_refused_class_of_a_load_is_taken_as_a_miss() {
    let mut harness = host_hit();
    harness
        .pool
        .refuse(|request| matches!(request, Request::Load { .. }).then_some(Cause::Class));
    let id = harness.submit(sharing(2, 1, 5, 3));
    harness.settle(100);
    assert!(harness.answers().into_iter().any(|answer| {
        matches!(
            answer,
            (
                Request::Load { .. },
                Completion::Refused {
                    cause: Cause::Class
                }
            )
        )
    }));
    assert_eq!(
        first_written(&harness, id),
        Some(0),
        "it computes the block"
    );
    assert_eq!(harness.tokens(id).len(), 3);
    assert_eq!(harness.failure(), None);
}

/// A `Clone` refused with the cause `key` (the partial block `cache` believed held is gone) is a
/// miss: a resumed request computes from its full blocks' end (SD §3.5 `Refused`, §4.4). The
/// request it continues left as migratable with no call in flight (its stop strings held its
/// decodes back), so its committed tail is the one the resumed request's hit can reach.
#[test]
fn a_refused_key_of_a_clone_is_taken_as_a_miss() {
    let mut harness = Deployment::tokens().build();
    let mut request = token(1, 6, 20);
    request.stop_strings = vec!["end".to_owned()];
    let id = harness.submit(request.clone());
    harness.run(20);
    let generated = harness.tokens(id);
    assert_eq!(
        generated.len(),
        2,
        "the queue depth of outputs ahead of the daemon"
    );

    // The drain ends the request as migratable: its partial last block is committed under its
    // tail key.
    let (reply, _replied) = crossbeam_channel::bounded(1);
    harness.apply(Message::Drain(
        MigrationCause::Drain,
        std::time::Instant::now(),
        false,
        reply,
    ));
    harness.settle(100);
    assert!(matches!(
        harness.terminal(id),
        Some(Event::Migratable { .. })
    ));
    harness.apply(Message::Resume);

    harness
        .pool
        .refuse(|request| matches!(request, Request::Clone { .. }).then_some(Cause::Key));
    request.generated_token_ids = generated.clone();
    let full = (6 + generated.len() as u32 - 1) / BLOCK * BLOCK;
    let before = harness.batches().len();
    harness.submit(request);
    harness.run_until(200, |harness| {
        let published = harness.tokens(id).len();
        harness.apply(Message::Acknowledge(id, published as u32));
        matches!(harness.terminal(id), Some(Event::Finished { .. }))
    });
    assert!(harness.answers().into_iter().any(|answer| {
        matches!(
            answer,
            (
                Request::Clone { .. },
                Completion::Refused { cause: Cause::Key }
            )
        )
    }));
    let prefill = harness
        .batches()
        .split_off(before)
        .into_iter()
        .find(|batch| batch.phase == Phase::Prefill && batch.requests.contains(&id));
    let first = prefill.unwrap().columns.rows.cache_positions[0];
    assert_eq!(first as u32, full, "it computes from its full blocks' end");
    let resumed = harness.tokens(id).len() - generated.len();
    assert_eq!(
        resumed,
        20 - generated.len(),
        "it generates what it had left"
    );
}

/// A refusal that names nothing `cache` can be wrong about (here an `Allocate`'s position, which
/// the scheduler chose from its own counts) is the scheduler's defect: it fails with the buffer
/// pool's error naming the cause (SD §3.5 `Refused`, §3.7).
#[test]
fn any_other_refusal_fails_the_scheduler() {
    let mut harness = Deployment::tokens().build();
    harness
        .pool
        .refuse(|request| matches!(request, Request::Allocate { .. }).then_some(Cause::Position));
    let id = harness.submit(token(1, 5, 4));
    harness.run_until(10, |harness| harness.failure().is_some());
    let failure = harness.failure().unwrap();
    assert!(failure.starts_with("the buffer pool failed: "), "{failure}");
    assert!(failure.contains("Position"), "{failure}");
    assert!(matches!(
        harness.terminal(id),
        Some(Event::Migratable {
            cause: MigrationCause::Loss,
            ..
        })
    ));
}

/// The scheduler gives a copy up at its bound while the executor's own `Finish` of it is on its
/// way: the buffer pool answers the first `Finish` it takes and refuses the second with the cause
/// `copy`, which the scheduler drops; the copy ends once (SD §3.5 `Refused`; BP §11 "Deadlines").
#[test]
fn a_give_up_crossing_the_executors_finish_ends_the_copy_once() {
    let mut harness = host_hit();
    // The empty batch that runs the load's copy stays on the device past the copy's bound.
    harness.fake.hold();
    let id = harness.submit(sharing(2, 1, 5, 3));
    harness.run_until(10, |harness| harness.fake.held() == 1);
    std::thread::sleep(Duration::from_millis(1_100));

    // The batch ends and the executor's `Finish` is on its way when the step gives the copy up.
    harness.fake.release(1);
    harness.scheduler.step();
    harness.fake.resume();
    harness.settle(200);

    let load = load_of(&harness, None).unwrap();
    let finishes: Vec<(bool, Completion)> = numbered(&harness)
        .into_iter()
        .filter_map(|(number, request)| match request {
            Request::Finish { copy, ok, .. } if copy.0 == load => {
                Some((ok, answer_to(&harness, number).unwrap()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        finishes,
        [
            (true, Completion::Loaded),
            (false, Completion::Refused { cause: Cause::Copy })
        ]
    );
    assert_eq!(
        first_written(&harness, id),
        Some(4),
        "it runs from the loaded block"
    );
    assert_eq!(harness.tokens(id).len(), 3);
    assert_eq!(harness.failure(), None);
}

/// A request waiting for the transfer of its blocks to its decode group is running, not
/// waiting: no step takes it for an admission while its `Transferred` is withheld, and it decodes
/// on the decode group once it arrives (SD §3.5 "Prefill to decode transfer").
#[test]
fn a_request_waiting_on_its_transfer_is_never_taken_for_an_admission() {
    let mut harness = Deployment::tokens().disaggregated().build();
    harness
        .pool
        .withhold(|request, _| matches!(request, Request::Transfer { .. }));
    let id = harness.submit(token(1, 9, 5));
    harness.run_until(20, |harness| harness.pool.withheld() == 1);
    let attaches = |harness: &Harness| {
        let taken = harness.taken().into_iter();
        taken
            .filter(|request| matches!(request, Request::Attach { .. }))
            .count()
    };
    let attached = attaches(&harness);
    harness.run(10);
    assert_eq!(
        attaches(&harness),
        attached,
        "no admission round while it transfers"
    );

    harness.pool.release();
    harness.settle(100);
    assert_eq!(harness.tokens(id).len(), 5);
    let decodes = harness
        .batches_of(id)
        .into_iter()
        .filter(|batch| batch.phase == Phase::Decode);
    assert!(decodes.clone().count() > 0);
    assert!(decodes.into_iter().all(|batch| batch.group == B));
}

/// A `Store` to the host answered `Full` leaves its blocks where they are: the request ends as
/// any other, and a later request of the same prompt finds the blocks on the device (SD §3.5
/// `Full`, §4.4 "The write policy").
#[test]
fn a_full_to_a_store_leaves_its_blocks_on_the_device() {
    let mut harness = Deployment::tokens().host(1).build();
    let id = harness.submit(token(1, 13, 2));
    harness.settle(100);
    assert_eq!(harness.tokens(id).len(), 2);
    let full = harness.answers().into_iter().any(|answer| {
        matches!(
            answer,
            (
                Request::Store {
                    tier: Tier::Host,
                    ..
                },
                Completion::Full { .. }
            )
        )
    });
    assert!(full, "the host region holds one block of three");

    let reader = harness.submit(sharing(2, 1, 13, 2));
    harness.settle(100);
    let pinned = harness
        .answers()
        .into_iter()
        .find_map(|answer| match answer {
            (Request::Pin { keys, .. }, Completion::Pinned { runs, .. }) => {
                Some((keys.len(), runs.len()))
            }
            _ => None,
        });
    assert_eq!(
        pinned,
        Some((3, 3)),
        "every block within the reuse limit is on the device"
    );
    assert_eq!(first_written(&harness, reader), Some(12));
}

/// A `Load` answered `Full` returns the admission to the waiting queue, detached; it is admitted
/// again once the region has room (SD §3.5 `Full`).
#[test]
fn a_full_to_a_load_returns_the_admission_detached() {
    let mut harness = host_hit();
    let kv = harness.region("kv-0");
    // Request 3 holds the five units the region keeps while its prefill is on the device: its
    // prompt's three blocks, and two more ahead of it.
    resize(&mut harness, "kv-0", 5);
    // It caches nothing, so its blocks are free once it ends.
    harness.fake.hold();
    let mut holding = token(3, 12, 1);
    holding.cache.write = false;
    let holding = harness.submit(holding);
    harness.run_until(10, |harness| harness.fake.held() >= 1);
    // The capacity words say the region has room; the buffer pool has none.
    harness.pool.set_capacity(kv, Some((100, 0, 0)));
    let id = harness.submit(sharing(2, 1, 5, 3));
    harness.run_until(10, |harness| {
        harness
            .answers()
            .into_iter()
            .any(|answer| matches!(answer, (Request::Load { .. }, Completion::Full { .. })))
    });
    harness.pool.set_capacity(kv, None);
    harness.fake.resume();
    harness.settle(200);
    assert_eq!(harness.tokens(holding).len(), 1, "{}", harness.summary());
    assert_eq!(harness.tokens(id).len(), 3);
    let owners: Vec<OwnerId> = harness
        .taken()
        .into_iter()
        .filter_map(|request| match request {
            Request::Attach { owner, .. } => Some(owner),
            _ => None,
        })
        .collect();
    assert!(owners.len() >= 4, "request 2 was admitted again");
}

/// A preempted video request whose `Save` is answered `Full` leaves with no stored step (SD §3.5
/// `Full`, §3.8).
#[test]
fn a_full_to_a_save_ends_the_video_with_no_stored_step() {
    // The latent pool holds one unit of 4096 bytes; a trajectory takes 30 000.
    let mut harness = Deployment::video(&[&VIDEO_PHASES]).latents(1).build();
    let mut request = video(1, 1);
    request.trajectory.id = 7;
    let id = harness.submit_video(request);
    let denoised = |harness: &Harness| {
        harness.events(id).into_iter().any(|event| {
            matches!(event, Event::MediaProgress { completed_steps, .. } if completed_steps >= 1)
        })
    };
    harness.fake.hold();
    harness.run_until(30, |harness| {
        harness.fake.release(1);
        denoised(harness)
    });
    let (reply, _replied) = crossbeam_channel::bounded(1);
    harness.apply(Message::Drain(
        MigrationCause::Drain,
        std::time::Instant::now(),
        false,
        reply,
    ));
    harness.fake.resume();
    harness.settle(100);

    let full = harness
        .answers()
        .into_iter()
        .any(|answer| matches!(answer, (Request::Save { .. }, Completion::Full { .. })));
    assert!(full, "{:#?}", harness.answers());
    assert!(matches!(
        harness.terminal(id),
        Some(Event::Migratable {
            cause: MigrationCause::Drain,
            denoised_steps: 0,
            ..
        })
    ));
}

/// When the answers to an admission show a shallower hit than `cache` expected (another epoch
/// evicted a block it held), the admission takes one more round for the positions the hit lost
/// and computes from the hit (SD §3.5, §4.4; BP §10 "Admission").
#[test]
fn a_shallower_hit_than_cache_expected_costs_one_more_round() {
    let mut harness = Deployment::tokens().build();
    harness.submit(token(1, 13, 1));
    harness.settle(100);
    let second = harness
        .taken()
        .into_iter()
        .find_map(|request| match request {
            Request::Commit { at: 1, key, .. } => Some(key),
            _ => None,
        });
    let evicted = harness.elsewhere(&[Request::Evict {
        keys: vec![second.unwrap()],
    }]);
    assert!(matches!(&evicted[..], [Completion::Evicted { keys }] if keys.len() == 1));

    let id = harness.submit(sharing(2, 1, 13, 3));
    harness.settle(100);
    let pinned = harness
        .answers()
        .into_iter()
        .find_map(|answer| match answer {
            (Request::Pin { keys, .. }, Completion::Pinned { runs, .. }) => {
                Some((keys.len(), runs.len()))
            }
            _ => None,
        });
    assert_eq!(
        pinned,
        Some((3, 1)),
        "`cache` expected three blocks; one is held"
    );
    assert_eq!(
        first_written(&harness, id),
        Some(4),
        "it computes from the hit"
    );
    assert_eq!(harness.tokens(id).len(), 3);
}

/// A copy that has not finished by its bound is given up with `Finish { ok: false }`; the
/// positions it was to fill are missing, and the request computes those tokens again (SD §4.4
/// "The bound of a copy").
#[test]
fn a_copy_given_up_at_its_bound_computes_its_tokens_again() {
    let mut harness = host_hit();
    // The empty batch that runs the load's copy stays on the device past the copy's bound.
    harness.fake.hold();
    let id = harness.submit(sharing(2, 1, 5, 3));
    harness.run_until(10, |harness| harness.fake.held() == 1);
    std::thread::sleep(Duration::from_millis(1_100));
    let load = load_of(&harness, None).unwrap();
    harness.run_until(10, |harness| {
        harness.taken().into_iter().any(
            |request| matches!(request, Request::Finish { copy, ok: false, .. } if copy.0 == load),
        )
    });
    harness.fake.resume();
    harness.settle(200);

    let loaded = harness
        .answers()
        .into_iter()
        .any(|answer| matches!(answer, (_, Completion::Loaded)));
    assert!(!loaded, "the copy was given up");
    assert_eq!(
        first_written(&harness, id),
        Some(0),
        "it computes the block again"
    );
    assert_eq!(harness.tokens(id).len(), 3);
}

/// When a region's `evictable` word counts more unpinned units than the scheduler's view
/// accounts for (another epoch committed blocks this scheduler does not know of), the scheduler
/// submits one `List`, and the next `Victims` of the region installs what it learned (SD §4.5).
#[test]
fn evictable_beyond_the_schedulers_order_leads_to_one_list() {
    let mut harness = Deployment::tokens().build();
    let lists = |harness: &Harness| {
        let taken = harness.taken().into_iter();
        taken
            .filter(|request| matches!(request, Request::List))
            .count()
    };
    assert_eq!(lists(&harness), 1, "the startup `List`");

    let owner = OwnerId(1);
    let key = Key([7; 32]);
    let answers = harness.elsewhere(&[
        Request::Attach { owner, group: A },
        Request::Allocate {
            owner,
            class: 0,
            at: 0,
            count: 1,
            units: 0,
        },
        Request::Commit {
            owner,
            class: 0,
            at: 0,
            key,
            tokens: BLOCK as u16,
            after: vec![(A, 0)],
        },
        Request::Detach {
            owner,
            after: vec![(A, 0)],
        },
    ]);
    assert!(answers.contains(&Completion::Committed), "{answers:?}");

    harness.run(10);
    assert_eq!(lists(&harness), 2, "one more `List`");
    let installed = harness
        .taken()
        .into_iter()
        .any(|request| matches!(request, Request::Victims { keys, .. } if keys.contains(&key)));
    assert!(
        installed,
        "the next `Victims` installs the block it learned"
    );
}
