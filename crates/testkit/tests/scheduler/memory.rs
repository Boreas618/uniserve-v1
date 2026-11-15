//! Memory decisions (SD §8.1): the rules of SD §4.4 by which the scheduler decides what each
//! request asks of the buffer pool, as the requests `schedule` and `update` submit them.
//!
//! - The blocks of each class a call needs: every block before its end for full attention, the
//!   blocks of its window for a sliding window, the checkpoint of each kept boundary for a
//!   recurrent state; the first chunk and the low watermark after a prefix hit; the fewest blocks
//!   that must coexist to compute a prompt in chunks.
//! - What a table gives up behind: a sliding window the blocks no later call reads, a recurrent
//!   state all but its latest checkpoint.
//! - The hit: the deepest boundary every class accepts, read by pins of the device copies, loads
//!   of the host copies, a clone of a partial last block, and prefetches from the remote store or
//!   the peers when they deepen it enough; and its statistics per tier and per source.
//! - The write policy: committed full-attention blocks stored to the host, sliding windows only
//!   before aligned boundaries, each block once, and the host copies stored on to the remote
//!   store; a block whose store stored nothing offered again only once its answer could differ.
//! - What the scheduler knows of the cache (SD §4.5): the eviction order of each region, least
//!   recently used first, whose head it installs; what `Pinned`, `Refused` and the keys a
//!   completion evicted correct.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use engine::batch::cache::CacheGroupKind;
use engine::batch::{Batch, Phase};
use engine::scheduler::Message;
use proto::request::wire::WeightsInFlight;
use proto::request::{Event, FinishReason, MigrationCause, RejectionKind, RequestId};
use storage::Key;
use storage::key::{page_key, page_root};
use storage::runtime::{Cause, Completion, GroupId, OwnerId, Request, Source, Tier};

use crate::harness::{A, BLOCK, Deployment, Harness, prompt, sharing, token};

/// A sliding window of eight tokens, two blocks.
const WINDOW: CacheGroupKind = CacheGroupKind::SlidingWindow { window: 8 };

/// The model of a full-attention cache group (class 0) and a sliding window (class 1).
pub(crate) const HYBRID: [(CacheGroupKind, u32); 2] =
    [(CacheGroupKind::Full, BLOCK), (WINDOW, BLOCK)];

/// The commits of `owner` the buffer pool took, per class, each class's keys by position.
pub(crate) fn commits(harness: &Harness, owner: OwnerId) -> HashMap<u8, Vec<(u32, Key)>> {
    let mut commits: HashMap<u8, Vec<(u32, Key)>> = HashMap::new();
    for request in harness.taken() {
        if let Request::Commit {
            owner: of,
            class,
            at,
            key,
            ..
        } = request
            && of == owner
        {
            commits.entry(class).or_default().push((at, key));
        }
    }
    for keys in commits.values_mut() {
        keys.sort_unstable_by_key(|(at, _)| *at);
    }
    commits
}

/// The keys of `class` among `commits`, by position.
pub(crate) fn keys_of(commits: &HashMap<u8, Vec<(u32, Key)>>, class: u8) -> Vec<Key> {
    let keys = commits.get(&class).into_iter().flatten();
    keys.map(|(_, key)| *key).collect()
}

/// The requests of `owner` the buffer pool took, in order: its `Pin`s, `Load`s, `Clone`s and
/// `Allocate`s.
fn asked_by(harness: &Harness, owner: OwnerId) -> Vec<Request> {
    let asked = harness.taken().into_iter().filter(|request| match request {
        Request::Pin { owner: of, .. }
        | Request::Load { owner: of, .. }
        | Request::Clone { owner: of, .. }
        | Request::Allocate { owner: of, .. } => *of == owner,
        _ => false,
    });
    asked.collect()
}

/// The keys of every block of `tokens` under the zero seed and no isolation key, for cache group
/// 0 of blocks of [`BLOCK`] tokens (SD §4.3): the keys the scheduler computes for a prompt.
fn prompt_keys(tokens: &[u32]) -> Vec<Key> {
    let mut parent = page_root(&Key([0; 32]), None);
    let (blocks, _) = tokens.as_chunks::<{ BLOCK as usize }>();
    blocks
        .iter()
        .map(|block| {
            parent = page_key(&parent, 0, 0, block);
            parent
        })
        .collect()
}

/// The host's monotonic clock in nanoseconds, which a prefetch's deadline is stated in.
fn monotonic_nanos() -> u64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `clock_gettime` writes the clock into the `timespec` it is given, which lives here.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
    let seconds = u64::try_from(now.tv_sec).unwrap_or(0);
    seconds * 1_000_000_000 + u64::try_from(now.tv_nsec).unwrap_or(0)
}

/// How long a prefetch of `tokens` tokens may take (SD §4.4): one second, and a quarter second
/// per 1024 tokens.
fn prefetch_wait(tokens: u32) -> Duration {
    Duration::from_secs(1) + Duration::from_millis(250 * u64::from(tokens) / 1024)
}

/// The hit counters of every report up to the next one: the tokens of the hits, those loaded
/// from the host, and those of them a prefetch from the peers brought.
fn hits(harness: &mut Harness) -> (u64, u64, u64) {
    harness.next_report();
    let reports = harness.reports();
    let stats = reports.iter().map(|report| &report.prefix_cache_stats);
    stats.fold((0, 0, 0), |(hits, host, peers), stats| {
        (
            hits + stats.base.hits,
            host + stats.host_hits,
            peers + stats.peer_hits,
        )
    })
}

/// Steps until every `Store` to `tier` the buffer pool took has been answered `Stored`.
fn stored(harness: &mut Harness, tier: Tier) {
    harness.run_until(100, |harness| {
        let stores = harness
            .taken()
            .into_iter()
            .filter(|request| matches!(request, Request::Store { tier: of, .. } if *of == tier))
            .count();
        let answered = harness.answers().into_iter().filter(|answer| match answer {
            (Request::Store { tier: of, .. }, Completion::Stored { .. }) => *of == tier,
            (Request::Finish { ok: true, .. }, Completion::Stored { .. }) => tier == Tier::Host,
            _ => false,
        });
        stores > 0 && answered.count() == stores
    });
}

/// A deployment with a host tier of 64 units where request 1, of a prompt of 13 tokens, committed
/// its three prompt blocks while it decoded and stored them on the host, whose device copies a
/// shrink of the KV region then evicted: a request of the same prompt finds them on the host
/// alone. Returns the harness and the blocks' keys.
fn stored_on_the_host() -> (Harness, Vec<Key>) {
    let mut harness = Deployment::tokens().host(64).build();
    let first = harness.submit(token(1, 13, 3));
    harness.settle(100);
    stored(&mut harness, Tier::Host);
    let keys = harness.committed(harness.owner(first, A).unwrap());
    assert_eq!(harness.resize("kv-0", 0), 0);
    assert_eq!(harness.resize("kv-0", 256), 256);
    (harness, keys)
}

/// The blocks each class of a hybrid model asks for after a prefix hit: the full-attention class
/// reads every block before the hit, the sliding window only the blocks of its window, its table
/// starting there; each class then allocates the blocks of the first chunk and the low watermark
/// from the hit on, and the request computes from the hit (SD §4.2, §4.4 "The blocks a call
/// needs", "The low watermark", "The hit").
#[test]
fn after_a_prefix_each_class_reads_what_its_kind_needs_and_allocates_its_first_chunk() {
    let mut harness = Deployment::tokens().cache(&HYBRID).build();
    let first = harness.submit(token(1, 17, 3));
    harness.settle(100);
    let cached = commits(&harness, harness.owner(first, A).unwrap());
    let reader = harness.submit(sharing(2, 1, 17, 2));
    harness.settle(100);
    let owner = harness.owner(reader, A).unwrap();

    // The hit is the reuse limit, 16 tokens: four blocks of full attention, the window's last two.
    let asked = asked_by(&harness, owner);
    let pins: Vec<(u8, u32, Vec<Key>)> = asked
        .iter()
        .filter_map(|request| match request {
            Request::Pin {
                class, at, keys, ..
            } => Some((*class, *at, keys.clone())),
            _ => None,
        })
        .collect();
    let full = keys_of(&cached, 0);
    let window = keys_of(&cached, 1);
    assert_eq!(
        pins,
        [(0, 0, full[..4].to_vec()), (1, 2, window[2..4].to_vec())]
    );
    // The first chunk is the prompt's last token: blocks to 5, and one more.
    let allocates: Vec<(u8, u32, u32)> = asked
        .iter()
        .filter_map(|request| match request {
            Request::Allocate {
                class, at, count, ..
            } => Some((*class, *at, *count)),
            _ => None,
        })
        .take(2)
        .collect();
    assert_eq!(allocates, [(0, 4, 2), (1, 4, 2)]);
    assert_eq!(harness.first_written(reader, 0), Some(16));
    assert_eq!(harness.tokens(reader).len(), 2);
}

/// A prompt whose blocks outnumber the region's units is rejected when every block must be held
/// at once (full attention), and admitted when a sliding window lets it be computed in chunks
/// with the fewest blocks that must coexist, `min(blocks, ceil((window + chunk) / block) + 1)`,
/// giving up the blocks behind its window as it goes (SD §4.4 "The blocks a call needs", §7.2
/// "Admission to the cache").
#[test]
fn a_prompt_longer_than_the_region_is_computed_in_chunks_when_a_window_bounds_its_blocks() {
    // Ten blocks of prompt in a region of eight units, in chunks of eight tokens.
    let mut full = Deployment::tokens().kv_units(8).chunk(8).build();
    let refused = full.submit(token(1, 40, 2));
    full.settle(50);
    let events = full.events(refused);
    let [Event::Rejected { kind, message }] = &events[..] else {
        panic!("{events:?}");
    };
    assert_eq!(*kind, RejectionKind::Invalid);
    assert_eq!(message, "request exceeds total KV capacity");

    let mut windowed = Deployment::tokens()
        .cache(&[(WINDOW, BLOCK)])
        .kv_units(8)
        .chunk(8)
        .build();
    let id = windowed.submit(token(1, 40, 2));
    windowed.settle(200);
    assert_eq!(windowed.tokens(id).len(), 2);
    let prefills = windowed
        .batches_of(id)
        .iter()
        .filter(|batch| batch.phase == Phase::Prefill)
        .count();
    assert_eq!(prefills, 5, "five chunks of eight tokens");
    let released = windowed
        .taken()
        .into_iter()
        .any(|request| matches!(request, Request::Release { class: 0, .. }));
    assert!(released, "the blocks behind the window were given up");
}

/// The batches `harness`'s workers launched on group `group`, in launch order: a stamp names
/// the last of them by its position from 1.
fn launched(harness: &Harness, group: GroupId) -> Vec<Batch> {
    let batches = harness.batches().into_iter();
    batches.filter(|batch| batch.group == group).collect()
}

/// A sliding window gives up its blocks behind the window as the request runs, and never one a
/// later call reads: every `Release` of the window's class names blocks that lie entirely before
/// the window of every row launched after the workers stamped it (`[position + 1 - window,
/// position]`), the releases follow each other from the table's first block, and the
/// full-attention class gives up nothing (SD §4.4 "Giving up behind", §7.4).
#[test]
fn a_sliding_window_gives_up_its_blocks_behind_the_window_and_never_one_a_later_call_reads() {
    let mut harness = Deployment::tokens().cache(&HYBRID).build();
    let id = harness.submit(token(1, 6, 30));
    harness.settle(200);
    assert_eq!(harness.tokens(id).len(), 30);
    let owner = harness.owner(id, A).unwrap();

    let releases: Vec<(u8, u32, u32, u64)> = harness
        .taken()
        .into_iter()
        .filter_map(|request| match request {
            Request::Release {
                owner: of,
                class,
                at,
                count,
                after,
            } if of == owner => Some((class, at, count, after.first()?.1)),
            _ => None,
        })
        .collect();
    assert!(!releases.is_empty());
    let batches = launched(&harness, A);
    let mut next = 0;
    for (class, at, count, stamp) in releases {
        assert_eq!(class, 1, "full attention keeps every block");
        assert_eq!(at, next, "each release continues the last");
        next = at + count;
        let later = batches.iter().skip(usize::try_from(stamp).unwrap());
        for batch in later {
            for position in &batch.columns.rows.cache_positions {
                let window_start = (*position + 1 - 8).max(0) as u32;
                assert!(
                    next * BLOCK <= window_start,
                    "blocks to {next} given up before batch {} read position {position}",
                    batch.id.0
                );
            }
        }
    }
}

/// A recurrent-state table keeps a checkpoint only at a kept boundary (the block boundary each
/// prompt chunk ends at; a decode only on a multiple of the state interval, which a short
/// request never reaches), one checkpoint block asked for at a time at position
/// `boundary / block - 1`, and gives up the older checkpoint once the next is written, so that it
/// holds only the latest (SD §3.4 "Recurrent-state checkpoints", §4.4 "Giving up behind", §7.4).
#[test]
fn checkpoints_are_kept_only_at_kept_boundaries_and_the_table_keeps_the_latest() {
    let mut harness = Deployment::tokens()
        .cache(&[
            (CacheGroupKind::Full, BLOCK),
            (CacheGroupKind::State, BLOCK),
        ])
        .chunk(8)
        .build();
    let id = harness.submit(token(1, 20, 3));
    harness.settle(100);
    assert_eq!(harness.tokens(id).len(), 3);
    let owner = harness.owner(id, A).unwrap();

    let kept: Vec<(Phase, i32)> = harness
        .batches_of(id)
        .iter()
        .map(|batch| {
            let row = batch.requests.iter().position(|of| *of == id).unwrap();
            (
                batch.phase,
                batch.columns.checkpoints.get(row).copied().unwrap_or(-1),
            )
        })
        .collect();
    let prefills: Vec<i32> = kept
        .iter()
        .filter(|(phase, _)| *phase == Phase::Prefill)
        .map(|(_, boundary)| *boundary)
        .collect();
    // A chunk ends at a boundary whose checkpoint it holds, or short of the boundary.
    let boundaries: Vec<i32> = prefills.iter().copied().filter(|kept| *kept >= 0).collect();
    assert_eq!(boundaries, [8, 16, 20], "{prefills:?}");
    assert!(
        kept.iter()
            .filter(|(phase, _)| *phase == Phase::Decode)
            .all(|(_, boundary)| *boundary < 0),
        "{kept:?}"
    );

    let state = |harness: &Harness, release: bool| -> Vec<u32> {
        let asked = harness
            .taken()
            .into_iter()
            .filter_map(|request| match request {
                Request::Allocate {
                    owner: of,
                    class: 1,
                    at,
                    ..
                } if of == owner && !release => Some(at),
                Request::Release {
                    owner: of,
                    class: 1,
                    at,
                    count: 1,
                    ..
                } if of == owner && release => Some(at),
                _ => None,
            });
        asked.collect()
    };
    assert_eq!(state(&harness, false), [1, 3, 4]);
    assert_eq!(state(&harness, true), [1, 3], "the older checkpoint goes");
}

/// A hybrid model's prompt cached but for its window's block 1, which another epoch evicted and
/// which a reader of its first nine tokens found gone (its `Pinned` holds one window block of
/// two); the reader is cancelled once it learned it. Returns the harness and the cached prompt's
/// keys of the full-attention class and of the window, by position.
fn window_with_a_hole() -> (Harness, Vec<Key>, Vec<Key>) {
    let mut harness = Deployment::tokens().cache(&HYBRID).build();
    let first = harness.submit(token(1, 17, 3));
    harness.settle(100);
    let cached = commits(&harness, harness.owner(first, A).unwrap());
    let (full, window) = (keys_of(&cached, 0), keys_of(&cached, 1));
    let evicted = harness.elsewhere(&[Request::Evict {
        keys: vec![window[1]],
    }]);
    assert_eq!(
        evicted,
        [Completion::Evicted {
            keys: vec![window[1]]
        }]
    );

    let teacher = harness.submit(sharing(2, 1, 9, 2));
    harness.run_until(10, |harness| {
        harness.answers().into_iter().any(|answer| match answer {
            (Request::Pin { class: 1, .. }, Completion::Pinned { blocks, .. }) => blocks.len() == 1,
            _ => false,
        })
    });
    harness.apply(Message::Cancel(teacher, FinishReason::Cancelled));
    harness.settle(50);
    (harness, full, window)
}

/// The hit is the deepest boundary every class accepts, and a boundary a class rejects does not
/// end the search: with the window's block 1 gone, boundaries 8 and 12 are rejected by the
/// window, yet 16, whose window is blocks 2 and 3, is accepted, and a reader of the whole prompt
/// reads four blocks of full attention and the window's last two (SD §4.4 "The hit", §7.4).
#[test]
fn the_hit_is_the_deepest_boundary_every_class_accepts_past_rejected_ones() {
    let (mut harness, full, window) = window_with_a_hole();
    let reader = harness.submit(sharing(3, 1, 17, 2));
    harness.settle(100);

    let owner = harness.owner(reader, A).unwrap();
    let pins: Vec<(u8, u32, Vec<Key>)> = asked_by(&harness, owner)
        .into_iter()
        .filter_map(|request| match request {
            Request::Pin {
                class, at, keys, ..
            } => Some((class, at, keys)),
            _ => None,
        })
        .collect();
    assert_eq!(
        pins,
        [(0, 0, full[..4].to_vec()), (1, 2, window[2..4].to_vec())]
    );
    assert_eq!(harness.first_written(reader, 0), Some(16));
}

/// `Pinned` corrects what `cache` expects: once an answer showed the window's block 1 gone, a
/// reader of the first nine tokens expects the hit at 4, reads exactly the blocks that hold it,
/// and is admitted in one round (SD §3.5 `Pinned`, §4.5 "What fills it").
#[test]
fn pinned_corrects_what_cache_expects() {
    let (mut harness, full, window) = window_with_a_hole();
    let reader = harness.submit(sharing(4, 1, 9, 2));
    harness.settle(100);

    let owner = harness.owner(reader, A).unwrap();
    let pins: Vec<(u8, u32, Vec<Key>)> = asked_by(&harness, owner)
        .into_iter()
        .filter_map(|request| match request {
            Request::Pin {
                class, at, keys, ..
            } => Some((class, at, keys)),
            _ => None,
        })
        .collect();
    assert_eq!(pins, [(0, 0, vec![full[0]]), (1, 0, vec![window[0]])]);
    assert_eq!(harness.first_written(reader, 0), Some(4));
    assert_eq!(harness.tokens(reader).len(), 2);
}

/// A request that left as migratable with its partial last block committed under its tail key,
/// held by no request. Its prompt is six tokens, it generated two (its stop strings held its
/// decodes back, so no call was in flight when it left): seven tokens computed, block 0 and a
/// tail of three in block 1. Returns the harness, the request as the daemon resubmits it, and the
/// tail's key.
fn left_with_a_tail() -> (Harness, proto::request::GenerationRequest, Key) {
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
    let owner = harness.owner(id, A).unwrap();
    let tail = harness
        .taken()
        .into_iter()
        .find_map(|request| match request {
            Request::Commit {
                owner: of,
                at: 1,
                key,
                tokens: 3,
                ..
            } if of == owner => Some(key),
            _ => None,
        });
    harness.apply(Message::Resume);
    request.generated_token_ids = generated;
    (harness, request, tail.unwrap())
}

/// A `Clone` refused with the cause `key` (the tail `cache` believed held was evicted through
/// another epoch's queue pair) takes the key out of `cache`: the resumed request computes from
/// its full blocks' end, and a later request of the same tokens expects no tail, reads the full
/// block alone and clones nothing (SD §3.5 `Refused`, §4.5 "What fills it").
#[test]
fn a_refused_clone_corrects_what_cache_expects() {
    let (mut harness, resumed, tail) = left_with_a_tail();
    let evicted = harness.elsewhere(&[Request::Evict { keys: vec![tail] }]);
    assert_eq!(evicted, [Completion::Evicted { keys: vec![tail] }]);

    let id = resumed.request_id;
    let lived = harness.batches_of(id).len();
    let continued: Vec<u32> = resumed
        .prompt_token_ids
        .iter()
        .chain(&resumed.generated_token_ids)
        .copied()
        .collect();
    harness.submit(resumed);
    harness.run_until(200, |harness| {
        let published = harness.tokens(id).len() as u32;
        harness.apply(Message::Acknowledge(id, published));
        harness.finished_with(id, FinishReason::MaxTokens)
    });
    let refused = harness.answers().into_iter().any(|answer| match answer {
        (Request::Clone { key, .. }, Completion::Refused { cause }) => {
            key == tail && cause == Cause::Key
        }
        _ => false,
    });
    assert!(refused);
    assert_eq!(harness.first_written(id, lived), Some(4));

    // The same tokens again: the full block is read, and no tail is cloned.
    let clones = |harness: &Harness| {
        harness
            .taken()
            .iter()
            .filter(|request| matches!(request, Request::Clone { .. }))
            .count()
    };
    let before = clones(&harness);
    let mut again = token(2, 8, 2);
    again.prompt_token_ids = continued;
    let again = harness.submit(again);
    harness.settle(100);
    assert_eq!(clones(&harness), before, "no tail is expected");
    assert_eq!(harness.first_written(again, 0), Some(4));
}

/// The keys a completion lists as evicted leave `cache`: an allocation that evicted the deepest
/// cached block of a prompt leaves the reader of that prompt expecting the blocks before it, which
/// it reads, computing from there in one round (SD §3.5, §4.5 "What fills it"; BP §11
/// "Evictions are reported").
#[test]
fn the_keys_a_completion_evicted_leave_cache() {
    let mut harness = Deployment::tokens().kv_units(8).build();
    let cached = harness.submit(token(1, 13, 3));
    harness.settle(100);
    let keys = harness.committed(harness.owner(cached, A).unwrap());
    assert_eq!(keys.len(), 3);

    // Six blocks of a request that caches nothing: cached blocks are evicted for them.
    let mut taking = token(2, 20, 1);
    taking.cache.write = false;
    harness.submit(taking);
    harness.settle(100);
    let evicted: Vec<Key> = harness
        .answers()
        .into_iter()
        .flat_map(|answer| match answer {
            (Request::Allocate { .. }, Completion::Allocated { evicted, .. }) => evicted,
            _ => Vec::new(),
        })
        .collect();
    // Deepest first: the blocks evicted are the prompt's last ones.
    let kept = keys.len() - evicted.len();
    assert!(kept > 0 && kept < keys.len(), "{evicted:?}");
    let deepest: Vec<Key> = keys[kept..].iter().rev().copied().collect();
    assert_eq!(evicted, deepest, "the deepest block first");

    let reader = harness.submit(sharing(3, 1, 13, 2));
    harness.settle(100);
    let owner = harness.owner(reader, A).unwrap();
    let pinned = asked_by(&harness, owner)
        .into_iter()
        .find_map(|request| match request {
            Request::Pin { keys, .. } => Some(keys),
            _ => None,
        });
    assert_eq!(pinned, Some(keys[..kept].to_vec()));
    assert_eq!(
        harness.first_written(reader, 0),
        Some(kept as i32 * BLOCK as i32)
    );
    assert_eq!(harness.tokens(reader).len(), 2);
}

/// A prefix the device no longer holds and the host does is loaded, read-only, at its
/// positions, and the request computes from it; its hit is counted as a hit and a host hit (SD
/// §4.4 "Pins, loads, clones and prefetches", "Hit statistics"; §7.5).
#[test]
fn a_host_hit_is_loaded_and_counted_as_a_host_hit() {
    let (mut harness, keys) = stored_on_the_host();
    let reader = harness.submit(sharing(2, 1, 13, 2));
    harness.settle(100);

    let owner = harness.owner(reader, A).unwrap();
    let loads: Vec<(u32, Vec<Key>)> = asked_by(&harness, owner)
        .into_iter()
        .filter_map(|request| match request {
            Request::Load { at, keys, .. } => Some((at, keys)),
            _ => None,
        })
        .collect();
    assert_eq!(loads, [(0, keys.clone())]);
    assert_eq!(harness.first_written(reader, 0), Some(12));
    assert_eq!(harness.tokens(reader).len(), 2);
    assert_eq!(hits(&mut harness), (12, 12, 0));
}

/// A deployment with a host tier of 80 units and a remote store, whose step holds 512 tokens,
/// where request 1's prompt of 264 tokens is cached in the remote store alone: its 66 blocks
/// were stored to the host and on to the remote store, then the KV region and the host region
/// were shrunk to nothing, each evicting the whole order the scheduler installed there. The
/// remote store holds the keys (`InProcessPool::hold_in`). Returns the harness and the keys.
fn in_the_remote_store_alone() -> (Harness, Vec<Key>) {
    let mut harness = Deployment::tokens().host(80).remote().batch(8, 512).build();
    let first = harness.submit(token(1, 264, 1));
    harness.settle(200);
    stored(&mut harness, Tier::Host);
    stored(&mut harness, Tier::Remote);
    let keys = harness.committed(harness.owner(first, A).unwrap());
    assert_eq!(keys.len(), 66);
    for key in &keys {
        harness.pool.hold_in(Source::Remote, *key, BLOCK as u16);
    }
    for (region, units) in [("kv-0", 256), ("host", 80)] {
        assert_eq!(harness.resize(region, 0), 0);
        assert_eq!(harness.resize(region, units), units);
    }
    (harness, keys)
}

/// When the remote store holds, by `cache`, keys that deepen the hit of the device and the host
/// by at least `prefetch_min_tokens` (256), the admission prefetches them from it, with the
/// deadline `prefetch_wait` of the deepening from now; its next round loads what was found, and
/// the request computes from the deepened hit (SD §4.4 "Pins, loads, clones and prefetches",
/// §7.4 "Loads and prefetches", "The prefetch wait").
#[test]
fn a_deep_enough_remainder_is_prefetched_from_the_remote_store_and_loaded() {
    let (mut harness, keys) = in_the_remote_store_alone();
    let before = monotonic_nanos();
    let reader = harness.submit(sharing(2, 1, 264, 2));
    let prefetch = |harness: &Harness| {
        harness
            .taken()
            .into_iter()
            .find_map(|request| match request {
                Request::Prefetch {
                    class,
                    keys,
                    source,
                    deadline,
                    ..
                } => Some((class, keys, source, deadline)),
                _ => None,
            })
    };
    harness.run_until(10, |harness| prefetch(harness).is_some());
    let after = monotonic_nanos();

    // The reuse limit is 260 tokens: 65 blocks, all deepening the hit.
    let (class, asked, source, deadline) = prefetch(&harness).unwrap();
    assert_eq!((class, source), (0, Source::Remote));
    assert_eq!(asked, keys[..65]);
    let wait = u64::try_from(prefetch_wait(260).as_nanos()).unwrap();
    assert!(
        (before + wait..=after + wait).contains(&deadline),
        "{deadline} not in {before} + {wait} ..= {after} + {wait}"
    );

    harness.settle(200);
    let owner = harness.owner(reader, A).unwrap();
    let loaded = asked_by(&harness, owner)
        .into_iter()
        .any(|request| matches!(request, Request::Load { at: 0, keys, .. } if keys == asked));
    assert!(loaded, "what the prefetch found is loaded");
    assert_eq!(harness.first_written(reader, 0), Some(260));
}

/// When the remote store does not deepen the hit and the deployment has peers, a prompt whose
/// reuse limit lies at least `prefetch_min_tokens` beyond the hit prefetches from the peers every
/// key beyond it; the leading keys they hold are loaded, the request computes from there, and
/// its hit is counted as a hit, a host hit and a peer hit (SD §4.4 "Pins, loads, clones and
/// prefetches", "Hit statistics"; §7.5 "Statistics rules").
#[test]
fn a_deep_enough_remainder_is_prefetched_from_the_peers_and_counted_as_a_peer_hit() {
    let mut harness = Deployment::tokens().host(80).peers().batch(8, 512).build();
    let tokens = prompt(1, 264);
    let keys = prompt_keys(&tokens);
    // The peers hold the first forty blocks.
    for key in &keys[..40] {
        harness.pool.hold_in(Source::Peers, *key, BLOCK as u16);
    }
    let id = harness.submit(token(1, 264, 2));
    harness.settle(200);

    let owner = harness.owner(id, A).unwrap();
    let prefetches: Vec<(Source, Vec<Key>)> = harness
        .taken()
        .into_iter()
        .filter_map(|request| match request {
            Request::Prefetch { keys, source, .. } => Some((source, keys)),
            _ => None,
        })
        .collect();
    assert_eq!(prefetches, [(Source::Peers, keys[..65].to_vec())]);
    let loaded = asked_by(&harness, owner)
        .into_iter()
        .any(|request| match request {
            Request::Load {
                at: 0,
                keys: loaded,
                ..
            } => loaded == keys[..40],
            _ => false,
        });
    assert!(loaded, "what the peers held is loaded");
    assert_eq!(harness.first_written(id, 0), Some(160));
    assert_eq!(hits(&mut harness), (160, 160, 160));
}

/// A remainder shallower than `prefetch_min_tokens` beyond the hit is computed, not prefetched:
/// on a deployment with peers, a prompt whose reuse limit is 196 tokens asks no `Prefetch` and
/// computes from its first token (SD §4.4 "Pins, loads, clones and prefetches").
#[test]
fn a_shallow_remainder_is_computed_and_not_prefetched() {
    let mut harness = Deployment::tokens().host(80).peers().batch(8, 512).build();
    let id = harness.submit(token(1, 200, 2));
    harness.settle(200);
    let prefetched = harness
        .taken()
        .iter()
        .any(|request| matches!(request, Request::Prefetch { .. }));
    assert!(!prefetched);
    assert_eq!(harness.first_written(id, 0), Some(0));
    assert_eq!(harness.tokens(id).len(), 2);
}

/// A request that left as migratable committed its partial last block under its tail key; the
/// daemon resubmits it with what it generated, and its admission reads its full blocks, clones
/// the tail into a block of its own, and writes the rest of that block: its first call computes
/// the last token it generated, inside the cloned block (SD §4.3 tails, §4.4 "The hit", "Pins,
/// loads, clones and prefetches"). The request held no call in flight when it left (its stop
/// strings held its decodes back), so its tail is what the resumed prompt continues.
#[test]
fn a_resumed_requests_partial_last_block_is_cloned_and_written() {
    let (mut harness, resumed, tail) = left_with_a_tail();
    let id = resumed.request_id;
    let lived = harness.batches_of(id).len();
    harness.submit(resumed);
    harness.run_until(200, |harness| {
        let published = harness.tokens(id).len() as u32;
        harness.apply(Message::Acknowledge(id, published));
        harness.finished_with(id, FinishReason::MaxTokens)
    });

    let owner = harness.owner(id, A).unwrap();
    let cloned = asked_by(&harness, owner).into_iter().any(
        |request| matches!(request, Request::Clone { class: 0, at: 1, key, .. } if key == tail),
    );
    assert!(cloned, "the tail is cloned at its position");
    assert_eq!(harness.first_written(id, lived), Some(7));
    assert_eq!(harness.tokens(id).len(), 20);
}

/// The keys of every `Store` to the host the buffer pool took, in order, a key once per store.
pub(crate) fn host_stores(harness: &Harness) -> Vec<Key> {
    let stores = harness
        .taken()
        .into_iter()
        .flat_map(|request| match request {
            Request::Store {
                keys,
                tier: Tier::Host,
                ..
            } => keys,
            _ => Vec::new(),
        });
    stores.collect()
}

/// With a host tier, every committed full-attention block is stored to the host, and a sliding
/// window's block only when it lies in the window before an aligned boundary: here the last
/// boundary a reissue of the prompt can reuse and the last of the prompt, both 20 tokens, so the
/// window's blocks 3 and 4, and not block 5, which the request still holds in its window; no
/// block is offered twice (SD §4.4 "The write policy", §7.4). The request is held, idle, with 26
/// tokens computed while the stores are observed (stop strings hold its decodes back): the
/// policy's `offload`, which adds the blocks no request holds, cannot add block 5 then.
#[test]
fn full_attention_blocks_are_stored_and_windows_only_before_aligned_boundaries_each_once() {
    let mut harness = Deployment::tokens().cache(&HYBRID).host(64).build();
    let mut request = token(1, 21, 20);
    request.stop_strings = vec!["end".to_owned()];
    let id = harness.submit(request);
    harness.run_until(40, |harness| {
        let published = harness.tokens(id).len() as u32;
        harness.apply(Message::Acknowledge(id, published.min(4)));
        published >= 6
    });
    // The stores the write policy asks for follow the commits' answers.
    harness.run(10);
    let cached = commits(&harness, harness.owner(id, A).unwrap());

    let stored_keys = host_stores(&harness);
    let full = keys_of(&cached, 0);
    assert_eq!(full.len(), 6, "24 of the 26 tokens fill blocks");
    assert!(full.iter().all(|key| stored_keys.contains(key)));
    let window = keys_of(&cached, 1);
    assert_eq!(window.len(), 6);
    assert!(stored_keys.contains(&window[3]) && stored_keys.contains(&window[4]));
    assert!(
        !stored_keys.contains(&window[5]),
        "past the aligned boundaries"
    );

    harness.run_until(100, |harness| {
        let published = harness.tokens(id).len() as u32;
        harness.apply(Message::Acknowledge(id, published));
        harness.terminal(id).is_some()
    });
    harness.settle(100);
    let stored_keys = host_stores(&harness);
    let mut unique = stored_keys.clone();
    unique.sort_unstable_by_key(|key| key.0);
    unique.dedup();
    assert_eq!(
        unique.len(),
        stored_keys.len(),
        "each block is offered once"
    );
}

/// With a remote store, every block stored on the host is then stored to the remote store, once
/// its host copy's `Stored` arrived (SD §4.4 "The write policy").
#[test]
fn host_copies_are_stored_on_to_the_remote_store() {
    let mut harness = Deployment::tokens().host(64).remote().build();
    let id = harness.submit(token(1, 13, 3));
    harness.settle(100);
    stored(&mut harness, Tier::Host);
    stored(&mut harness, Tier::Remote);
    let keys = harness.committed(harness.owner(id, A).unwrap());
    assert_eq!(keys.len(), 3);

    // Per key, the host store's copy finished before the remote store was taken.
    let taken: Vec<(u64, Request)> = harness
        .pool
        .requests()
        .iter()
        .map(|(_, number, request)| (number.0, request.clone()))
        .collect();
    for key in &keys {
        let host = taken.iter().find_map(|(number, request)| match request {
            Request::Store {
                keys,
                tier: Tier::Host,
                ..
            } if keys.contains(key) => Some(*number),
            _ => None,
        });
        let finished = taken.iter().position(|(_, request)| {
            matches!(request, Request::Finish { copy, ok: true, .. } if Some(copy.0) == host)
        });
        let remote = taken.iter().position(|(_, request)| {
            matches!(request, Request::Store { keys, tier: Tier::Remote, .. } if keys.contains(key))
        });
        assert!(finished.is_some() && remote > finished, "{key:?}");
    }
    let remote_stored: Vec<Key> = harness
        .answers()
        .into_iter()
        .flat_map(|answer| match answer {
            (
                Request::Store {
                    tier: Tier::Remote, ..
                },
                Completion::Stored { keys },
            ) => keys,
            _ => Vec::new(),
        })
        .collect();
    for key in &keys {
        assert!(remote_stored.contains(key));
    }
}

/// The keys the buffer pool holds copies of on the host: those each `Copy` answering a `Store`
/// to the host covered, once the `Finish` of that copy was answered `Stored`.
fn on_the_host(harness: &Harness) -> HashSet<Key> {
    let completions = harness.pool.completions();
    let covered: HashMap<u64, &Vec<Key>> = completions
        .iter()
        .filter_map(|(_, number, completion)| match completion {
            Completion::Copy { keys, .. } => Some((number.0, keys)),
            _ => None,
        })
        .collect();
    let finishes: HashMap<u64, u64> = harness
        .pool
        .requests()
        .iter()
        .filter_map(|(_, number, request)| match request {
            Request::Finish { copy, ok: true, .. } => Some((number.0, copy.0)),
            _ => None,
        })
        .collect();
    let stored = completions
        .iter()
        .filter(|(_, _, completion)| matches!(completion, Completion::Stored { .. }))
        .filter_map(|(_, number, _)| covered.get(finishes.get(&number.0)?));
    stored.flat_map(|keys| keys.iter().copied()).collect()
}

/// Checks that every key of `keys` was named by exactly one `Store` to the host: the write
/// policy's at its commit, the policy's `offload` naming it in vain since.
fn offered_once(harness: &Harness, keys: &[Key]) {
    let offered = host_stores(harness);
    for key in keys {
        let times = offered.iter().filter(|of| *of == key).count();
        assert_eq!(times, 1, "{key:?} was offered to the host {times} times");
    }
}

/// A block whose `Store` to the host found no room is not offered to the host again until the
/// host region has room: while the region holds no unit, neither the scheduler nor the policy's
/// `offload`, which names a block once no request holds it, stores any of them again, and once
/// the region grows the scheduler offers them again and they are stored (SD §4.4 "The write
/// policy": a store that stored nothing backs off until something that could change its answer
/// happens).
#[test]
fn a_block_the_host_had_no_room_for_is_offered_again_once_the_host_has_room() {
    let mut harness = Deployment::tokens().host(64).build();
    assert_eq!(harness.resize("host", 0), 0);
    let id = harness.submit(token(1, 13, 2));
    harness.settle(100);
    let keys = harness.committed(harness.owner(id, A).unwrap());
    assert_eq!(keys.len(), 3);

    harness.run(30);
    offered_once(&harness, &keys);
    assert!(on_the_host(&harness).is_empty());

    assert_eq!(harness.resize("host", 64), 64);
    harness.run_until(30, |harness| {
        let stored = on_the_host(harness);
        keys.iter().all(|key| stored.contains(key))
    });
}

/// A block whose copy to the host failed is not offered to the host again until a `Store` into
/// the host region stores a block, which shows that the tier copies again: then the scheduler
/// offers it again and it is stored (SD §4.4 "The write policy").
#[test]
fn a_block_whose_copy_to_the_host_failed_is_offered_again_once_the_host_stores_again() {
    let mut harness = Deployment::tokens().host(64).build();
    harness.fake.fail_copies(true);
    let id = harness.submit(token(1, 13, 2));
    harness.settle(100);
    let keys = harness.committed(harness.owner(id, A).unwrap());
    assert_eq!(keys.len(), 3);

    harness.run(30);
    offered_once(&harness, &keys);
    assert!(on_the_host(&harness).is_empty());

    // Another request's blocks are stored by the write policy, and the refused ones follow.
    harness.fake.fail_copies(false);
    harness.submit(token(2, 9, 2));
    harness.settle(100);
    harness.run_until(30, |harness| {
        let stored = on_the_host(harness);
        keys.iter().all(|key| stored.contains(key))
    });
}

/// Runs token request `id` of a prompt of `tokens` tokens that may generate `max`, with stop
/// strings, so that it holds its blocks idle once two of its outputs wait for the daemon's
/// decoder: once the buffer pool answered every `Commit` of it, its outputs are passed and it
/// runs to its end, giving up blocks `cache` holds. Returns its id and its committed keys.
fn cached_while_running(
    harness: &mut Harness,
    id: u64,
    tokens: u32,
    max: usize,
) -> (RequestId, Vec<Key>) {
    let mut request = token(id, tokens, max);
    request.stop_strings = vec!["end".to_owned()];
    let id = harness.submit(request);
    harness.run_until(20, |harness| harness.tokens(id).len() >= 2);
    let owner = harness.owner(id, A).unwrap();
    harness.run_until(20, |harness| {
        let committed = harness.answers().into_iter().filter(|answer| match answer {
            (Request::Commit { owner: of, .. }, Completion::Committed) => *of == owner,
            _ => false,
        });
        let count = committed.count();
        count > 0 && count == harness.committed(owner).len()
    });
    harness.run_until(100, |harness| {
        let published = harness.tokens(id).len() as u32;
        harness.apply(Message::Acknowledge(id, published));
        harness.terminal(id).is_some() && harness.held() == 0
    });
    (id, harness.committed(owner))
}

/// The `Victims` of the KV region the buffer pool took last, by then.
fn installed(harness: &mut Harness) -> Vec<Key> {
    let kv = harness.region("kv-0");
    let mut taken = harness.taken().into_iter().rev();
    let last = taken.find_map(|request| match request {
        Request::Victims { region, keys } if region == kv => Some(keys),
        _ => None,
    });
    last.unwrap_or_default()
}

/// The eviction order of a region is least recently used first, a request's blocks entering it
/// when it gives them up, its deepest block first so that the prefix before it stays cached
/// longest; its head is installed by `Victims` and replaced as it moves: a pin takes blocks out,
/// and they return as the most recently used when given up. The policy's victim under a
/// shortage is the head's first block, which is evicted by its key (SD §4.4 "The eviction
/// order", §4.5 "Installing the order", §3.6).
#[test]
fn the_eviction_order_is_least_recently_used_first_and_its_head_is_installed_as_it_moves() {
    let mut harness = Deployment::tokens().build();
    let (_, first_keys) = cached_while_running(&mut harness, 1, 13, 3);
    let (k0, k1, k2) = (first_keys[0], first_keys[1], first_keys[2]);
    assert_eq!(installed(&mut harness), [k2, k1, k0]);

    let (_, second_keys) = cached_while_running(&mut harness, 2, 9, 3);
    let (r0, r1) = (second_keys[0], second_keys[1]);
    assert_eq!(installed(&mut harness), [k2, k1, k0, r1, r0]);

    // A reader of the first prompt pins its blocks: the head moves past them; once it ends they
    // are the most recently used.
    let reader = harness.submit(sharing(3, 1, 13, 2));
    harness.run_until(20, |harness| !harness.tokens(reader).is_empty());
    assert_eq!(installed(&mut harness), [r1, r0]);
    harness.settle(100);
    assert_eq!(installed(&mut harness), [r1, r0, k2, k1, k0]);

    // A shortage: the policy's victim is the head's first block, evicted by its key.
    let kv = harness.region("kv-0");
    let growing = harness.submit(token(4, 5, 30));
    harness.run_until(20, |harness| harness.tokens(growing).len() >= 2);
    harness.pool.set_capacity(kv, Some((0, 0, 0)));
    let evict = |harness: &Harness| {
        harness
            .taken()
            .into_iter()
            .find_map(|request| match request {
                Request::Evict { keys } => Some(keys),
                _ => None,
            })
    };
    harness.run_until(40, |harness| evict(harness).is_some());
    assert_eq!(evict(&harness).unwrap().first(), Some(&r1));
    harness.pool.set_capacity(kv, None);
    harness.settle(200);
    assert_eq!(harness.tokens(growing).len(), 30);
}

/// A request that ends before the buffer pool answered the commits of its blocks (a prompt that
/// ends at the token its prefill samples) gives them up as any other: they enter the eviction
/// order deepest block first, so that its prompt's prefix stays cached longest (SD §4.4 "The
/// eviction order"; `vllm/v1/core/block_pool.py` `free_blocks`, which a request's blocks enter
/// tail first).
#[test]
fn a_request_that_ends_before_its_commits_are_answered_enters_the_order_deepest_first() {
    let mut harness = Deployment::tokens().build();
    let first = harness.submit(token(1, 13, 1));
    harness.settle(100);
    let keys = harness.committed(harness.owner(first, A).unwrap());
    assert_eq!(keys.len(), 3);
    harness.run(3);
    assert_eq!(installed(&mut harness), [keys[2], keys[1], keys[0]]);
}

/// Requests whose host hits take more units than the device has free all end, one after another,
/// each reading its prefix from the host: an admission's `Load` allocates the device blocks of
/// its host part, which `fits` counts beside the blocks it computes, and the eviction order the
/// scheduler installs ahead of the step's requests covers what they take, the blocks a `Load`
/// copies to included, so that no admission the scheduler counted as fitting is answered `Full`
/// (SD §3.5 `fits`, §4.4 "Pins, loads, clones and prefetches", §4.5 "Installing the order").
///
/// Four prompts of 31 full blocks each leave their blocks on the host and none on the device,
/// then each is asked again with a tail of three more blocks, all four at once: the device's 64
/// units hold one such request and not two, and once one has run its blocks wait in the eviction
/// order, beyond the head a step's token budget covers.
#[test]
fn host_hits_beyond_what_the_device_has_free_are_admitted_in_turn_and_all_end() {
    let mut harness = Deployment::tokens()
        .kv_units(64)
        .host(256)
        .batch(8, 32)
        .chunk(32)
        .build();
    let end = [7];
    let tail = prompt(9, 12);
    let asked = |turn: u64, tail: &[u32]| -> Vec<u32> {
        let words = prompt(turn, 126);
        words
            .into_iter()
            .chain(tail.iter().copied())
            .chain(end)
            .collect()
    };
    for turn in 1..=4 {
        let mut request = token(turn, 1, 1);
        request.prompt_token_ids = asked(turn, &[]);
        let id = harness.submit(request);
        harness.run_until(400, |harness| harness.terminal(id).is_some());
    }
    stored(&mut harness, Tier::Host);
    crate::pause_flush_resize::flush(&mut harness, None, WeightsInFlight::Keep).unwrap();
    harness.run(5);

    let ids: Vec<RequestId> = (1..=4)
        .map(|turn| {
            let mut request = token(10 + turn, 1, 1);
            request.prompt_token_ids = asked(turn, &tail);
            harness.submit(request)
        })
        .collect();
    harness.settle(1_000);

    for id in &ids {
        assert!(
            harness.finished_with(*id, FinishReason::MaxTokens),
            "{:?}",
            harness.events(*id)
        );
        assert_eq!(
            harness.first_written(*id, 0),
            Some(124),
            "it read 31 blocks"
        );
    }
    // Each read its 31 blocks from the host.
    assert_eq!(hits(&mut harness), (4 * 124, 4 * 124, 0));
    let refused = harness.answers().into_iter().any(|(request, answer)| {
        matches!(request, Request::Load { .. }) && matches!(answer, Completion::Full { .. })
    });
    assert!(!refused, "no admission counted as fitting found no room");
}

/// A hit is read only as deep as the admission holds it alone in the region: a hit near the end
/// of a chunked prompt lets the first call reach the prompt's end, and with the low watermark
/// beyond it the admission would hold every block of the prompt and one more, though computed in
/// chunks from position 0 the prompt takes only its own blocks. In a region of exactly those
/// blocks, the request reads the deepest boundary with which its blocks of the hit, its first
/// call and the low watermark fit the region's units, and neither it nor the request behind it
/// waits for good at the head of the queue (SD §3.5 `never_fits`, `fits`; §4.4 "The hit", "The
/// low watermark").
#[test]
fn a_hit_is_read_only_as_deep_as_the_admission_holds_it_alone_in_the_region() {
    // A prompt of eight blocks in a region of eight units, computed in chunks of four blocks.
    let mut harness = Deployment::tokens().kv_units(8).chunk(16).build();
    let first = harness.submit(token(1, 32, 1));
    harness.settle(100);
    assert!(harness.finished_with(first, FinishReason::MaxTokens));

    let reader = harness.submit(sharing(2, 1, 32, 1));
    let behind = harness.submit(token(3, 8, 1));
    harness.settle(200);
    // At the reuse limit, 28 tokens, and at every boundary from 16 on, the hit's blocks and the
    // first call's with the low watermark are nine; at 12 tokens they are the hit's three, the
    // chunk's four up to token 28 and one more.
    assert_eq!(harness.first_written(reader, 0), Some(12));
    for id in [reader, behind] {
        assert!(harness.finished_with(id, FinishReason::MaxTokens), "{id:?}");
    }
}

/// A request resumed on the engine of a later epoch of the same layout reads every block its
/// previous life committed there, its partial last block by its tail key, though those commits
/// came through the earlier epoch's queue pair after this scheduler listed the cache: its
/// admission waits for a `List` asked after it arrived, then pins the full blocks, clones the
/// tail, and computes only the last token its previous life emitted (SD §3.8 "A drain to an
/// epoch of the same layout", §4.3 "Writing", §4.4 "The hit", §4.5).
#[test]
fn a_request_resumed_from_another_epoch_reads_every_block_that_epoch_committed() {
    let mut harness = Deployment::tokens().build();

    // Its previous life, on epoch 2's queue pair: a prompt of 7 tokens and 12 generated, of
    // which it computed all but the last: four full blocks and a tail of two tokens.
    let generated: Vec<u32> = (500..512).collect();
    let tokens: Vec<u32> = prompt(1, 7).into_iter().chain(generated.clone()).collect();
    let keys = prompt_keys(&tokens[..16]);
    let tail = page_key(&keys[3], 0, 0, &tokens[16..18]);
    let owner = OwnerId(1);
    let after = vec![(A, 0)];
    let mut asks = vec![
        Request::Attach { owner, group: A },
        Request::Allocate {
            owner,
            class: 0,
            at: 0,
            count: 5,
            units: 0,
        },
    ];
    for (at, &key) in (0u32..).zip(&keys) {
        asks.push(Request::Commit {
            owner,
            class: 0,
            at,
            key,
            tokens: BLOCK as u16,
            after: after.clone(),
        });
    }
    asks.push(Request::Commit {
        owner,
        class: 0,
        at: 4,
        key: tail,
        tokens: 2,
        after: after.clone(),
    });
    asks.push(Request::Detach { owner, after });
    let answers = harness.elsewhere(&asks);
    let committed = answers
        .iter()
        .filter(|answer| **answer == Completion::Committed)
        .count();
    assert_eq!(committed, 5, "{answers:?}");

    // The daemon resubmits it at once, with what it generated.
    let mut resumed = token(1, 7, 20);
    resumed.generated_token_ids = generated;
    let id = harness.submit(resumed);
    harness.run_until(200, |harness| {
        let published = harness.tokens(id).len() as u32;
        harness.apply(Message::Acknowledge(id, published));
        harness.finished_with(id, FinishReason::MaxTokens)
    });

    let owner = harness.owner(id, A).unwrap();
    let cloned = asked_by(&harness, owner).into_iter().any(
        |request| matches!(request, Request::Clone { class: 0, at: 4, key, .. } if key == tail),
    );
    assert!(cloned, "the tail is cloned at its position");
    assert_eq!(harness.first_written(id, 0), Some(18));
    assert_eq!(harness.tokens(id).len(), 8);
    assert_eq!(hits(&mut harness), (18, 0, 0));
}

/// A prompt's recurrent-state checkpoint cached without the attention blocks before it: request
/// 1 commits both, then another epoch evicts its attention blocks, which this scheduler learns
/// only from its own answers. Request 2 of the same prompt expects its 32 whole-block tokens as
/// its hit, pins the checkpoint and finds no attention block, so its hit is 0 and its first chunk
/// computes the prompt from its start, keeping the state at boundary 32, where its table holds
/// the pinned checkpoint read-only. The chunk keeps into a private block allocated at that
/// position once the read-only checkpoint is given up, never into the pinned block, which other
/// requests read (SD §3.4 "Recurrent-state checkpoints", §4.4 "The hit"; BP §9 A4).
#[test]
fn a_chunk_from_the_start_keeps_its_checkpoint_in_a_private_block_where_a_pinned_one_lies() {
    let mut harness = Deployment::tokens()
        .cache(&[
            (CacheGroupKind::Full, BLOCK),
            (CacheGroupKind::State, BLOCK),
        ])
        .build();
    let first = harness.submit(token(1, 34, 2));
    harness.settle(200);
    assert!(harness.finished_with(first, FinishReason::MaxTokens));

    let attention: Vec<Key> = harness
        .taken()
        .into_iter()
        .filter_map(|request| match request {
            Request::Commit { class: 0, key, .. } => Some(key),
            _ => None,
        })
        .collect();
    assert!(
        !attention.is_empty(),
        "request 1 committed its attention blocks"
    );
    harness.elsewhere(&[Request::Evict { keys: attention }]);

    let second = harness.submit(sharing(2, 1, 34, 2));
    harness.settle(200);
    assert_eq!(harness.failure(), None, "the scheduler failed");
    assert!(harness.finished_with(second, FinishReason::MaxTokens));

    // Its first chunk computed the prompt from its start and kept the state at boundary 32,
    // the end of the pinned checkpoint's block.
    let prefill = harness.batches_of_phase(second, Phase::Prefill).remove(0);
    let row = prefill
        .requests
        .iter()
        .position(|id| *id == second)
        .unwrap();
    let first_token = prefill.columns.rows.query_offsets[row] as usize;
    assert_eq!(prefill.columns.rows.cache_positions[first_token], 0);
    assert_eq!(prefill.columns.checkpoints.get(row), Some(&32));
}
