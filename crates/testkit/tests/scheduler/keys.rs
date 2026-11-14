//! Keys (SD §8.1): the key of every block the scheduler names (SD §4.3) and what an admission
//! and a request's outputs ask of the buffer pool by key (SD §4.1, §4.4). The chain of a prompt
//! meets its fixed vectors, with and without an isolation key, and an isolated chain reaches only
//! its own requests; an admission pins the device part of its hit within the reuse limit, loads
//! the host part after it and clones a partial last block, and its prefill starts from the hit;
//! a request commits its full blocks as they fill, and its partial last blocks under their tail
//! keys when it leaves as migratable, per class; the read and write switches stop the one or the
//! other; a resumed request reads the chain over its prompt and generated tokens whatever its read
//! switch says.
//!
//! Every deployment here roots its chains at the zero seed: its groups state no page seeds.

use engine::batch::Phase;
use engine::batch::cache::CacheGroupKind;
use engine::scheduler::Message;
use proto::request::wire::WeightsInFlight;
use proto::request::{Event, GenerationRequest, MigrationCause, RequestId};
use storage::Key;
use storage::key::{page_key, page_root};
use storage::runtime::{Completion, OwnerId, Request};

use crate::harness::{A, BLOCK, Deployment, Harness, prompt, row_positions, token};

/// The keys of the blocks of the prompt `1..=8` in blocks of four tokens under the zero seed, in
/// cache group 0: without an isolation key, and with isolation key 7 (SD §4.3).
const PLAIN: [&str; 2] = [
    "d82404dd8bfea249aec17a3856952089f9a004f8c56e533b9f9cbae86f47b421",
    "ba86d1709ff8ac538e9aabba75a7f8838f16d3c622ce9b286bd63301cb520c37",
];
const ISOLATED: [&str; 2] = [
    "e96a1c1f75c37f1ee0012bb59a32daabf95e1086c1e0585b2df75f575abd99d9",
    "2fa8b480f6a0e7db59c2d7d139b2eb3ed0e146ea90f851648a545756422560d5",
];

fn hex(key: &Key) -> String {
    key.0.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The chain of `class` over the complete blocks of `tokens` under the zero seed, without an
/// isolation key (SD §4.3).
fn chain(class: u32, tokens: &[u32]) -> Vec<Key> {
    let mut parent = page_root(&Key([0; 32]), None);
    let (blocks, _) = tokens.as_chunks::<{ BLOCK as usize }>();
    blocks
        .iter()
        .map(|block| {
            parent = page_key(&parent, class, 0, block);
            parent
        })
        .collect()
}

/// The `Commit`s the buffer pool took for `owner`, in order: class, position, key, tokens.
fn commits(harness: &Harness, owner: OwnerId) -> Vec<(u8, u32, Key, u16)> {
    let taken = harness.taken().into_iter();
    taken
        .filter_map(|request| match request {
            Request::Commit {
                owner: of,
                class,
                at,
                key,
                tokens,
                ..
            } if of == owner => Some((class, at, key, tokens)),
            _ => None,
        })
        .collect()
}

/// Whether the buffer pool took a request that reads cached blocks (`Pin`, `Load`, `Clone`) for
/// `owner`.
fn reads(harness: &Harness, owner: OwnerId) -> bool {
    harness.taken().into_iter().any(|request| match request {
        Request::Pin { owner: of, .. }
        | Request::Load { owner: of, .. }
        | Request::Clone { owner: of, .. } => of == owner,
        _ => false,
    })
}

/// The first cache position request `id`'s first prefill after the `skip` first batches the
/// executor was handed wrote: where its computation started.
fn first_written(harness: &Harness, id: RequestId, skip: usize) -> i32 {
    let batches = harness.batches().into_iter().skip(skip);
    let mut prefills =
        batches.filter(|batch| batch.phase == Phase::Prefill && batch.requests.contains(&id));
    let prefill = prefills.next().expect("a prefill of the request");
    row_positions(&prefill, id)[0]
}

/// The requests that read cached blocks (`Pin`, `Load`, `Clone`) the buffer pool took after the
/// first `skip` it took, in order.
fn read_requests(harness: &Harness, skip: usize) -> Vec<Request> {
    let taken = harness.taken().into_iter().skip(skip);
    taken
        .filter(|request| {
            matches!(
                request,
                Request::Pin { .. } | Request::Load { .. } | Request::Clone { .. }
            )
        })
        .collect()
}

/// Request `of`'s prompt of `tokens` tokens, followed by `more` tokens of request `id`'s: a
/// request that continues another's prompt.
fn continuing(id: u64, of: u64, tokens: u32, more: u32, max: usize) -> GenerationRequest {
    let mut request = token(id, tokens + more, max);
    request.prompt_token_ids = prompt(of, tokens)
        .into_iter()
        .chain(prompt(id, more))
        .collect();
    request
}

/// Runs token request `id` until it was sent at least `tokens` tokens, then ends it as
/// migratable with nothing of it in flight: `Pause`, one step that applies its call in flight,
/// and a drain whose deadline has passed (SD §3.8). Steps until the drain is answered, and
/// resumes. Returns its owner on group [`A`], the tokens it was sent and the cache positions its
/// calls wrote, `0..written`.
fn migrate(harness: &mut Harness, id: RequestId, tokens: usize) -> (OwnerId, Vec<u32>, u32) {
    harness.run_until(100, |harness| harness.tokens(id).len() >= tokens);
    let owner = harness.owner(id, A).unwrap();
    let (reply, paused) = crossbeam_channel::bounded(1);
    harness.apply(Message::Pause(reply));
    harness.run_until(10, |_| paused.try_recv().is_ok());

    let (reply, drained) = crossbeam_channel::bounded(1);
    let deadline = std::time::Instant::now();
    harness.apply(Message::Drain(
        MigrationCause::Drain,
        deadline,
        false,
        reply,
    ));
    harness.run_until(50, |_| drained.try_recv().is_ok());
    harness.apply(Message::Resume);
    let Some(Event::Migratable { .. }) = harness.terminal(id) else {
        panic!("{:?}", harness.events(id));
    };

    let written = harness
        .batches_of(id)
        .iter()
        .flat_map(|batch| batch.columns.rows.cache_positions.clone())
        .max()
        .unwrap();
    (owner, harness.tokens(id), written as u32 + 1)
}

/// The commits of the prompt `1..=8` carry the fixed vectors of the key chain: a block's key
/// chains its parent's, from the root of the zero seed, without an isolation key and with one,
/// at its position, with the block's tokens (SD §4.3; §4.4 "Commit").
#[test]
fn the_commits_of_a_prompt_carry_the_fixed_vectors_of_its_key_chain() {
    let mut harness = Deployment::tokens().build();
    let mut plain = token(1, 8, 1);
    plain.prompt_token_ids = (1..=8).collect();
    let mut isolated = plain.clone();
    isolated.request_id = RequestId(2);
    isolated.cache.isolation_key = Some(7);
    let plain = harness.submit(plain);
    let isolated = harness.submit(isolated);
    harness.run_until(10, |harness| {
        harness.owner(plain, A).is_some() && harness.owner(isolated, A).is_some()
    });
    let owners = [harness.owner(plain, A), harness.owner(isolated, A)];
    harness.settle(100);

    for (owner, keys) in owners.into_iter().zip([PLAIN, ISOLATED]) {
        let commits: Vec<(u8, u32, String, u16)> = commits(&harness, owner.unwrap())
            .iter()
            .map(|(class, at, key, tokens)| (*class, *at, hex(key), *tokens))
            .collect();
        let expected: Vec<(u8, u32, String, u16)> = (0..)
            .zip(keys)
            .map(|(at, key)| (0, at, key.to_owned(), 4))
            .collect();
        assert_eq!(commits, expected);
    }
}

/// An isolation key roots a chain of its own: a request of the same prompt without it finds
/// nothing and computes the prompt, and one with it pins the blocks the first committed (SD §4.3:
/// `page_root(seed, isolation_key)`).
#[test]
fn an_isolation_key_roots_a_chain_only_its_own_requests_reach() {
    let mut harness = Deployment::tokens().build();
    let mut first = token(1, 13, 1);
    first.cache.isolation_key = Some(7);
    let first = harness.submit(first);
    harness.run_until(10, |harness| harness.owner(first, A).is_some());
    let owner = harness.owner(first, A).unwrap();
    harness.settle(100);
    let committed: Vec<Key> = commits(&harness, owner)
        .iter()
        .map(|commit| commit.2)
        .collect();
    assert_eq!(committed.len(), 3);

    let mut open = token(2, 13, 1);
    open.prompt_token_ids = prompt(1, 13);
    let skip = (harness.batches().len(), harness.taken().len());
    let open = harness.submit(open);
    harness.settle(100);
    assert_eq!(first_written(&harness, open, skip.0), 0);
    assert!(read_requests(&harness, skip.1).is_empty());

    let mut same = token(3, 13, 1);
    same.prompt_token_ids = prompt(1, 13);
    same.cache.isolation_key = Some(7);
    let skip = (harness.batches().len(), harness.taken().len());
    let same = harness.submit(same);
    harness.settle(100);
    assert_eq!(first_written(&harness, same, skip.0), 12);
    let reads = read_requests(&harness, skip.1);
    let [
        Request::Pin {
            class, at, keys, ..
        },
    ] = &reads[..]
    else {
        panic!("{reads:?}");
    };
    assert_eq!((*class, *at, keys), (0, 0, &committed));
}

/// An admission pins the blocks of its hit from position 0, at most the reuse limit
/// `(prompt_len - 1) / step * step`, so that its prefill computes at least its last prompt
/// token, and its prefill starts at the hit, which its scheduling event reports as its cached
/// tokens (SD §4.3 "Reading"; §4.4 "The hit"; §7.4 "Reuse limit and step").
#[test]
fn an_admission_pins_its_hit_within_the_reuse_limit_and_its_prefill_starts_there() {
    let mut harness = Deployment::tokens().build();
    let first = harness.submit(token(1, 12, 1));
    harness.run_until(10, |harness| harness.owner(first, A).is_some());
    let owner = harness.owner(first, A).unwrap();
    harness.settle(100);
    let keys = chain(0, &prompt(1, 12));
    let committed: Vec<Key> = commits(&harness, owner)
        .iter()
        .map(|commit| commit.2)
        .collect();
    assert_eq!(committed, keys);

    // The same prompt: its reuse limit is 8, two blocks of the three cached.
    let mut same = token(2, 12, 1);
    same.prompt_token_ids = prompt(1, 12);
    let skip = (harness.batches().len(), harness.taken().len());
    let same = harness.submit(same);
    harness.settle(100);
    let reads = read_requests(&harness, skip.1);
    let [
        Request::Pin {
            class,
            at,
            keys: pinned,
            ..
        },
    ] = &reads[..]
    else {
        panic!("{reads:?}");
    };
    assert_eq!((*class, *at, &pinned[..]), (0, 0, &keys[..2]));
    assert_eq!(first_written(&harness, same, skip.0), 8);
    assert_eq!((cached(&harness, first), cached(&harness, same)), (0, 8));

    // A prompt that continues it: all three blocks.
    let skip = (harness.batches().len(), harness.taken().len());
    let longer = harness.submit(continuing(3, 1, 12, 5, 1));
    harness.settle(100);
    let reads = read_requests(&harness, skip.1);
    let [
        Request::Pin {
            class,
            at,
            keys: pinned,
            ..
        },
    ] = &reads[..]
    else {
        panic!("{reads:?}");
    };
    assert_eq!((*class, *at, pinned), (0, 0, &keys));
    assert_eq!(first_written(&harness, longer, skip.0), 12);
    assert_eq!(harness.tokens(longer).len(), 1);
    assert_eq!(cached(&harness, longer), 12);
}

/// The cached tokens the scheduling event of request `id` reports: the prompt tokens its
/// admission found cached.
fn cached(harness: &Harness, id: RequestId) -> usize {
    let events = harness.events(id);
    let Some(Event::Scheduled { cached_tokens, .. }) = events.first() else {
        panic!("request {} was not scheduled first: {events:?}", id.0);
    };
    *cached_tokens
}

/// An admission whose hit lies partly on the device and partly on the host alone pins the
/// device part from position 0 and loads the host part at the positions after it, in one round
/// (SD §4.4 "Pins, loads, clones and prefetches"; §4.1, `admit`'s first round).
#[test]
fn an_admission_pins_the_device_part_of_its_hit_and_loads_the_host_part_after_it() {
    let mut harness = Deployment::tokens().host(64).build();
    let first = harness.submit(token(1, 13, 1));
    harness.settle(100);
    let keys = chain(0, &prompt(1, 12));
    let stored = |harness: &Harness| {
        let answers = harness.answers().into_iter();
        let stored = answers.filter(|answer| {
            matches!(
                answer,
                (Request::Finish { ok: true, .. }, Completion::Stored { .. })
            )
        });
        stored.count()
    };
    // The policy's `offload` may store a block again while its first store is under way.
    harness.run_until(30, |harness| stored(harness) >= 3);
    assert_eq!(harness.tokens(first).len(), 1);

    // Every cached block leaves the device; the host keeps its copies.
    let (reply, flushed) = crossbeam_channel::bounded(1);
    harness.apply(Message::Flush(None, WeightsInFlight::Keep, reply));
    assert!(flushed.try_recv().unwrap().is_ok());
    let evicted = |harness: &Harness| {
        let mut answers = harness.answers().into_iter();
        answers.any(|answer| matches!(answer, (Request::Evict { .. }, Completion::Evicted { .. })))
    };
    harness.run_until(10, |harness| evicted(harness));

    // A request that reads nothing computes the first block again: it alone is on the device.
    let mut recompute = token(2, 5, 1);
    recompute.prompt_token_ids = prompt(1, 5);
    recompute.cache.read = false;
    harness.submit(recompute);
    harness.settle(100);

    let skip = harness.taken().len();
    harness.submit(continuing(3, 1, 13, 3, 1));
    harness.run_until(20, |harness| read_requests(harness, skip).len() >= 2);
    let reads = read_requests(&harness, skip);
    let [
        Request::Pin {
            class: 0,
            at: 0,
            keys: pinned,
            ..
        },
        Request::Load {
            class: 0,
            at: 1,
            keys: loaded,
            ..
        },
    ] = &reads[..]
    else {
        panic!("{reads:?}");
    };
    assert_eq!(pinned[..], keys[..1]);
    assert_eq!(loaded[..], keys[1..]);
}

/// A request that leaves as migratable commits, in each full-attention and sliding-window class,
/// its partial last block under its tail key with its token count, after the full blocks it
/// committed as they filled, each under the class's chain (SD §4.3 "A tail"; §4.4 "Commit"; §3.7
/// `free`).
#[test]
fn a_migrating_request_commits_its_partial_last_block_of_each_class_under_its_tail_key() {
    let window = CacheGroupKind::SlidingWindow { window: 8 };
    let mut harness = Deployment::tokens()
        .cache(&[(CacheGroupKind::Full, BLOCK), (window, BLOCK)])
        .build();
    let id = harness.submit(token(1, 6, 20));
    let (owner, generated, written) = migrate(&mut harness, id, 5);
    let tokens: Vec<u32> = prompt(1, 6).into_iter().chain(generated).collect();
    let (full, rest) = (written / BLOCK, written % BLOCK);
    assert!(rest > 0, "{written} tokens end in a partial block");
    harness.run(5);

    let commits = commits(&harness, owner);
    for class in [0, 1] {
        let keys = chain(u32::from(class), &tokens[..(full * BLOCK) as usize]);
        let tail = page_key(
            &keys[full as usize - 1],
            u32::from(class),
            0,
            &tokens[(full * BLOCK) as usize..written as usize],
        );
        let mut expected: Vec<(u8, u32, Key, u16)> = (0..)
            .zip(keys)
            .map(|(at, key)| (class, at, key, BLOCK as u16))
            .collect();
        expected.push((class, full, tail, rest as u16));
        let of_class: Vec<(u8, u32, Key, u16)> = commits
            .iter()
            .filter(|commit| commit.0 == class)
            .copied()
            .collect();
        assert_eq!(of_class, expected, "class {class}");
    }
}

/// A request whose cache policy reads nothing pins nothing and computes its prompt, though its
/// blocks are cached; one that writes nothing commits nothing over its life (SD §4.3 "Reading",
/// "Writing": `cache.read`, `cache.write`).
#[test]
fn a_request_that_reads_nothing_pins_nothing_and_one_that_writes_nothing_commits_nothing() {
    let mut harness = Deployment::tokens().build();
    harness.submit(token(1, 13, 1));
    harness.settle(100);

    let mut unread = token(2, 13, 4);
    unread.prompt_token_ids = prompt(1, 13);
    unread.cache.read = false;
    let mut unwritten = token(3, 13, 12);
    unwritten.cache.write = false;
    let skip = harness.batches().len();
    let unread = harness.submit(unread);
    let unwritten = harness.submit(unwritten);
    harness.run_until(10, |harness| {
        harness.owner(unread, A).is_some() && harness.owner(unwritten, A).is_some()
    });
    let owners = (
        harness.owner(unread, A).unwrap(),
        harness.owner(unwritten, A).unwrap(),
    );
    harness.settle(100);

    assert_eq!(first_written(&harness, unread, skip), 0);
    assert!(!reads(&harness, owners.0));
    assert!(commits(&harness, owners.1).is_empty());
    assert_eq!(harness.tokens(unwritten).len(), 12);
}

/// A resumed request's prompt holds the tokens it generated before, and its chain over them is
/// the chain of the request it continues: its admission pins the blocks that request committed,
/// whatever its read switch says, and it continues from them (SD §4.3: "the chain of a resumed
/// request's prompt ... equals the chain over prompt and generated tokens of the request it
/// continues"; "A resumed request's own blocks").
#[test]
fn a_resumed_request_pins_the_chain_over_its_generated_tokens_whatever_its_read_switch() {
    let mut harness = Deployment::tokens().build();
    let id = harness.submit(token(1, 6, 20));
    let (owner, generated, written) = migrate(&mut harness, id, 6);
    assert_eq!(written % BLOCK, 0, "{written} tokens fill whole blocks");
    harness.run(5);
    let committed: Vec<Key> = commits(&harness, owner)
        .iter()
        .map(|commit| commit.2)
        .collect();
    let tokens: Vec<u32> = prompt(1, 6).into_iter().chain(generated.clone()).collect();
    assert_eq!(committed, chain(0, &tokens[..written as usize]));

    let mut resumed = token(1, 6, 20);
    resumed.generated_token_ids = generated.clone();
    resumed.cache.read = false;
    let skip = (harness.batches().len(), harness.taken().len());
    harness.submit(resumed);
    harness.run_until(300, |harness| {
        matches!(harness.events(id).last(), Some(Event::Finished { .. }))
    });

    let reads = read_requests(&harness, skip.1);
    let [
        Request::Pin {
            class, at, keys, ..
        },
    ] = &reads[..]
    else {
        panic!("{reads:?}");
    };
    assert_eq!((*class, *at, keys), (0, 0, &committed));
    assert_eq!(first_written(&harness, id, skip.0), written as i32);
    let all = harness.tokens(id);
    assert_eq!(all.len(), 20, "it continues its budget");
}

/// A resumed request whose hit ends in the partial last block the request it continues
/// committed as it left clones that block under its tail key, at the position after the blocks
/// it pins (SD §4.4 "The hit": the hit extends to a partial last block every class holds under
/// its tail key, which the request clones and writes; §8.1 "Keys").
#[test]
fn a_resumed_request_clones_the_partial_last_block_its_predecessor_left() {
    let mut harness = Deployment::tokens().build();
    let id = harness.submit(token(1, 6, 20));
    let (owner, generated, written) = migrate(&mut harness, id, 5);
    let (full, rest) = (written / BLOCK, written % BLOCK);
    assert!(rest > 0, "{written} tokens end in a partial block");
    harness.run(5);
    let commits = commits(&harness, owner);
    let tail = commits
        .iter()
        .find(|commit| commit.1 == full && commit.3 == rest as u16)
        .map(|commit| commit.2)
        .expect("the tail's commit");

    let mut resumed = token(1, 6, 20);
    resumed.generated_token_ids = generated;
    resumed.cache.read = false;
    let skip = harness.taken().len();
    harness.submit(resumed);
    harness.run_until(20, |harness| read_requests(harness, skip).len() >= 2);
    let reads = read_requests(&harness, skip);
    let pinned: Vec<Key> = commits
        .iter()
        .filter(|commit| commit.1 < full)
        .map(|commit| commit.2)
        .collect();
    let [
        Request::Pin {
            class: 0,
            at: 0,
            keys,
            ..
        },
        Request::Clone {
            class: 0, at, key, ..
        },
    ] = &reads[..]
    else {
        panic!("{reads:?}");
    };
    assert_eq!(keys, &pinned);
    assert_eq!((*at, *key), (full, tail));
}
