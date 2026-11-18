//! A worker group's workers against recorded completions, batches and stamped requests, without
//! a device, in every interleaving the recording allows: the worker's half of L5 of the buffer
//! pool's design (`specs/buffer-pool.md` §19).
//!
//! ## The group
//!
//! Two ranks: rank 0 is this node's worker, and rank 1 is the worker of a peer engine process,
//! which the node reaches over the peer link as it reaches a group's ranks on another host. Each
//! runs a [`Reader`] over [`Graphs::meta`]: a runner that states what every batch reads (the block
//! tables, the device table it wrote from their changes, the units it was asked to reset and the
//! copies that started with it) and holds the batch until the test ends it on that rank, which
//! stands for the batch's device work ending there.
//!
//! A reader serves a batch in one of two ways ([`Turns`]), and every recording is replayed under
//! both: in the one turn of `Runner::execute`, which holds the worker until the batch ends, or as
//! a device runner whose step is in flight across two turns, launched by `Runner::start` and
//! landed by `Runner::execute` once the batch ended, the worker taking the jobs behind it, the
//! next batch's launch included, meanwhile.
//!
//! ## The recordings
//!
//! A recording is what a scheduler hands the group's [`Route`]: completions with the requests
//! they answer (`map`), requests to apply and stamp (`stamp`), batches (`submit`), and the buffer
//! pool's answers to the `Finish` of the copies the batches ran. Each event waits for what the
//! buffer pool's order and the scheduler's knowledge make it wait for: the completions of
//! unstamped requests arrive in the order the requests were submitted; a request names only
//! entries its owner holds, so a `Commit` follows the completion that gave its block, and in some
//! recordings a batch that wrote it; the answer to a copy's `Finish` arrives once `completed`
//! reached the batch that ran the copy. A completion that gives an owner entries is dropped when
//! the owner's `Detach` or `Transfer` was handed before it (§9, obligation 1); a copy's completion
//! is still handed, so the copy runs and ends.
//!
//! ## The interleavings
//!
//! Every order of the scheduler's hand-offs, and every order in which the two ranks end their
//! batches, with each other and with the hand-offs: the peer lags when its rank ends a batch after
//! this node's, and its report reaches the node only then. Each rank takes the group's jobs in the
//! order they were handed, so two interleavings that differ only in when a rank ended a batch
//! relative to a hand-off that does not wait for `completed` show every rank the same tables at
//! every batch and the same `completed` at every end. The search keeps one interleaving of each
//! such class (a depth-first search with sleep sets) and replays each against a new group; each
//! test states how many its recording has.
//!
//! ## What is asserted after every step
//!
//! The four obligations of the control plane (§9; the assumptions A1 to A4 of §17), with what every
//! rank must hold derived from the requests and their completions by §11:
//!
//! 1. At every batch on every rank, the tables and the device table the runner wrote from their
//!    changes hold exactly the runs the completions handed before the batch gave, at the positions
//!    their requests named, less the entries the `Release`, `Detach` and `Transfer` handed before
//!    it took away.
//! 2. Every stamped request returns with the number of batches handed before it: the group's
//!    launched sequence when every rank applied it. By 1, no batch numbered above the stamp reads
//!    an entry the request took away, so the entries had left every rank's table by then. A copy's
//!    `Finish` returns with the batch that ran the copy.
//! 3. When a rank's batch ends, before its worker counts the end, the group's `completed` and the
//!    word the buffer pool reads have not reached the batch; after every step, both equal the last
//!    batch every rank has ended.
//! 4. An entry is writable exactly when it came from `Allocated`, `Cloned` or a private block of
//!    `Transferred` and no `Commit` named it since; a batch resets only units `Allocated` gave since
//!    the rank's previous batch whose entry is still writable, since a reset writes them; it starts
//!    every copy handed and neither started nor given up, and no other; the destination of a copy
//!    is no entry until the copy's answer gives it.

// The helpers below unwrap, as tests do (`clippy.toml`).
#![cfg(test)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use engine::batch::{Batch, BatchId, BatchOutput, Columns, Phase, Shape, Variant};
use engine::executor::Route;
use engine::executor::peers::{Liveness, Remote, Upstream, accept, register};
use engine::memory::tables::{BlockCopy, BlockTables, Change};
use engine::runner::buckets::Bucket;
use engine::runner::graphs::Graphs;
use engine::runner::{Forward, HostForward, Runner};
use engine::worker::completion::{Outcome, Submission};
use engine::worker::{Error, Result};
use engine::worker::{Sequences, Worker};
use ops::Generation;
use service::lock;
use storage::runtime::{Completion, GroupId, OwnerId, Request, RequestId, Run, Stamps, Tier};
use storage::{Interest, Key, QueuePair};
use tensor::Tensor;
use testkit::LONG;

/// The worker group, and the regions each of its ranks maps, by the numbers `Open` gave them: the
/// KV units on the device and the host pool of their copies.
const GROUP: GroupId = GroupId(3);
const KV: u8 = 0;
const HOST: u8 = 1;

/// The bytes of a unit of each region, summed over the group's ranks. The recordings' copies
/// move a few units together, far within the workers' copy budget, so every batch starts every
/// copy handed before it.
const UNIT_BYTES: [(u8, u64); 2] = [(KV, 256), (HOST, 256)];

/// The group an owner moves to or comes from.
const ELSEWHERE: GroupId = GroupId(4);

/// The one class of the recordings, and the slots their owners are kept under.
const CLASS: u8 = 0;
const SLOTS: [u32; 2] = [1, 2];

/// The group's ranks: this node's worker, then the peer's.
const RANKS: usize = 2;
const PEER: usize = 1;

/// The engine number the peer registers under.
const ENGINE: u64 = 7;

/// The request numbers of the `Finish` requests whose answers are mapped: this plus the copy's.
const ANSWERS: u64 = 1_000;

/// A position of a rank's tables: slot, class and position.
type Position = (u32, u8, u32);

/// The copies of one request: its number, and each copy from a run to a run.
type Copies = (u64, Vec<(Run, Run)>);

/// An entry as a completion gives it: class, position, run, and whether a batch may write it.
type Entry = (u8, u32, Run, bool);

// ---- The recordings ----

/// What the scheduler hands the group's workers.
#[derive(Clone, Debug)]
enum Hand {
    /// The completion of request `id`, with the request, under the slot that keeps its entries.
    Map {
        slot: u32,
        id: u64,
        request: Request,
        completion: Completion,
    },
    /// A request the workers apply and stamp.
    Stamp(Request),
    /// The group's next batch.
    Batch,
    /// The buffer pool's answer to the `Finish` the workers returned for the copies of request
    /// `copy`, handed under `slot`.
    Answer {
        slot: u32,
        copy: u64,
        answer: Completion,
    },
}

/// What an event waits for, besides an answer's wait for `completed`.
#[derive(Clone, Copy, Debug)]
enum Need {
    /// The named event was handed, or dropped.
    Handed(&'static str),
    /// A batch was handed after the named event: the block it gave was written.
    Written(&'static str),
}

/// One event of a recording.
struct Event {
    name: &'static str,
    hand: Hand,
    needs: Vec<Need>,
}

fn event(name: &'static str, hand: Hand, needs: &[Need]) -> Event {
    Event {
        name,
        hand,
        needs: needs.to_vec(),
    }
}

fn map(slot: u32, id: u64, request: Request, completion: Completion) -> Hand {
    Hand::Map {
        slot,
        id,
        request,
        completion,
    }
}

fn key(byte: u8) -> Key {
    Key([byte; 32])
}

/// Blocks of one unit each, at `units` of the KV region.
fn blocks(units: &[u32]) -> Vec<Run> {
    units.iter().map(|unit| (KV, *unit, 1)).collect()
}

fn attach(id: u64, owner: u64, slot: u32) -> Hand {
    let request = Request::Attach {
        owner: OwnerId(owner),
        group: GROUP,
    };
    map(slot, id, request, Completion::Attached { slot })
}

/// An `Allocate` of one block per unit of `units` at `at`, with its `Allocated`.
fn allocate(id: u64, owner: u64, slot: u32, at: u32, units: &[u32]) -> Hand {
    let request = Request::Allocate {
        owner: OwnerId(owner),
        class: CLASS,
        at,
        count: units.len() as u32,
        units: 0,
    };
    let allocated = Completion::Allocated {
        runs: blocks(units),
        evicted: Vec::new(),
    };
    map(slot, id, request, allocated)
}

/// A `Pin` of `keys` keys at `at`, the leading ones pinned at `units`, the others on the host.
fn pin(id: u64, owner: u64, slot: u32, at: u32, keys: u8, units: &[u32]) -> Hand {
    let request = Request::Pin {
        owner: OwnerId(owner),
        class: CLASS,
        at,
        keys: (0..keys).map(key).collect(),
    };
    let pinned = Completion::Pinned {
        blocks: blocks(units).into_iter().map(|run| vec![run]).collect(),
        host: u32::from(keys) - units.len() as u32,
    };
    map(slot, id, request, pinned)
}

/// The `Copy` that answers request `id` with one copy of its key from `from` to `to`.
fn copied(id: u64, from: Run, to: Run) -> Completion {
    Completion::Copy {
        group: GROUP,
        keys: vec![key(id as u8)],
        from: vec![from],
        to: vec![to],
        evicted: Vec::new(),
        encodings: None,
        parts: Vec::new(),
    }
}

/// A `Load` of one key into position `at`, copied from the host's `from` to the device's `to`.
fn load(id: u64, owner: u64, slot: u32, at: u32, from: Run, to: Run) -> Hand {
    let request = Request::Load {
        owner: OwnerId(owner),
        class: CLASS,
        at,
        keys: vec![key(id as u8)],
    };
    map(slot, id, request, copied(id, from, to))
}

/// A `Clone` of the cached block at `from` into position `at`, whose new block is `to`.
fn clone_block(id: u64, owner: u64, slot: u32, at: u32, from: Run, to: Run) -> Hand {
    let request = Request::Clone {
        owner: OwnerId(owner),
        class: CLASS,
        at,
        key: key(id as u8),
    };
    map(slot, id, request, copied(id, from, to))
}

/// A `Store` of the cached block at `from` to the host's `to`. It names no owner, so no slot.
fn store(id: u64, from: Run, to: Run) -> Hand {
    let request = Request::Store {
        group: GROUP,
        keys: vec![key(id as u8)],
        tier: Tier::Host,
    };
    map(0, id, request, copied(id, from, to))
}

/// The owner arriving from another group into `slot` with `entries`: per position, its unit and
/// whether its block is private.
fn moved_in(id: u64, owner: u64, slot: u32, entries: &[(u32, u32, bool)]) -> Hand {
    let request = Request::Transfer {
        owner: OwnerId(owner),
        group: GROUP,
        after: vec![(ELSEWHERE, 1)],
    };
    let transferred = Completion::Transferred {
        slot,
        runs: entries
            .iter()
            .map(|&(position, unit, private)| (CLASS, position, (KV, unit, 1), private))
            .collect(),
    };
    map(slot, id, request, transferred)
}

fn release(owner: u64, at: u32, count: u32) -> Hand {
    Hand::Stamp(Request::Release {
        owner: OwnerId(owner),
        class: CLASS,
        at,
        count,
        after: Vec::new(),
    })
}

fn commit(owner: u64, at: u32) -> Hand {
    Hand::Stamp(Request::Commit {
        owner: OwnerId(owner),
        class: CLASS,
        at,
        key: key(at as u8 + 100),
        tokens: 64,
        after: Vec::new(),
    })
}

fn detach(owner: u64) -> Hand {
    Hand::Stamp(Request::Detach {
        owner: OwnerId(owner),
        after: Vec::new(),
    })
}

/// The owner leaving for another group.
fn moved_out(owner: u64) -> Hand {
    Hand::Stamp(Request::Transfer {
        owner: OwnerId(owner),
        group: ELSEWHERE,
        after: Vec::new(),
    })
}

fn give_up(copy: u64) -> Hand {
    Hand::Stamp(Request::Finish {
        copy: RequestId(copy),
        ok: false,
        after: Vec::new(),
    })
}

fn answer(copy: u64, slot: u32, answer: Completion) -> Hand {
    Hand::Answer { slot, copy, answer }
}

/// Owner 1 in slot 1 is given two blocks by `Allocated` and one of two keys by `Pinned`. A
/// `Commit` names the first block, before or after a batch reads it: a batch that follows the
/// `Commit` resets none of its units, since the entry is read-only from then on. A `Release` takes
/// the second away and another `Allocated` fills its position again; the `Detach` takes the rest,
/// and drops the completions handed after it.
fn entries() -> Vec<Event> {
    use Need::Handed;

    vec![
        event("attach", attach(1, 1, 1), &[]),
        event(
            "allocate",
            allocate(2, 1, 1, 0, &[10, 11]),
            &[Handed("attach")],
        ),
        event("pin", pin(3, 1, 1, 2, 2, &[20]), &[Handed("allocate")]),
        event("b1", Hand::Batch, &[]),
        event("b2", Hand::Batch, &[Handed("b1")]),
        event("commit", commit(1, 0), &[Handed("allocate")]),
        event("release", release(1, 1, 1), &[Handed("allocate")]),
        event(
            "refill",
            allocate(4, 1, 1, 1, &[12]),
            &[Handed("pin"), Handed("release")],
        ),
        event("detach", detach(1), &[Handed("commit"), Handed("release")]),
    ]
}

/// Owner 1 in slot 1 loads a key into position 0 and clones a cached block into position 1; each
/// copy runs with the first batch after its completion, and its answer (`Loaded`, `Cloned`)
/// gives the position its destination. A batch writes the clone, a `Commit` names it, and a
/// later batch may read it.
fn copies() -> Vec<Event> {
    use Need::{Handed, Written};

    vec![
        event("attach", attach(1, 1, 1), &[]),
        event(
            "load",
            load(2, 1, 1, 0, (HOST, 0, 1), (KV, 30, 1)),
            &[Handed("attach")],
        ),
        event(
            "clone",
            clone_block(3, 1, 1, 1, (KV, 20, 1), (KV, 31, 1)),
            &[Handed("load")],
        ),
        event("b1", Hand::Batch, &[]),
        event("b2", Hand::Batch, &[Handed("b1")]),
        event("b3", Hand::Batch, &[Handed("b2")]),
        event("loaded", answer(2, 1, Completion::Loaded), &[]),
        event("cloned", answer(3, 1, Completion::Cloned), &[]),
        event("commit", commit(1, 1), &[Written("cloned")]),
    ]
}

/// A store of a cached block to the host is given up (`Finish { ok: false }`), before or after a
/// batch started it, while owner 1 in slot 1 loads a key whose block stays its own (`Duplicate`).
/// The owner may be detached at any point: the answer that would give it the loaded block is
/// then dropped, and the load still runs.
fn give_up_store() -> Vec<Event> {
    use Need::Handed;

    vec![
        event("attach", attach(1, 1, 1), &[]),
        event(
            "load",
            load(2, 1, 1, 0, (HOST, 0, 1), (KV, 30, 1)),
            &[Handed("attach")],
        ),
        event("store", store(3, (KV, 10, 1), (HOST, 5, 1)), &[]),
        event("give-up", give_up(3), &[Handed("store")]),
        event("duplicate", answer(2, 1, Completion::Duplicate), &[]),
        event("b1", Hand::Batch, &[]),
        event("b2", Hand::Batch, &[Handed("b1")]),
        event("detach", detach(1), &[Handed("attach")]),
    ]
}

/// Owner 1 in slot 1 holds an allocated block and moves to another group (`Transfer`); owner 2
/// arrives in slot 2 from another group with a shared and a private block (`Transferred`), and a
/// `Release` takes the shared one away.
fn transfer() -> Vec<Event> {
    use Need::Handed;

    vec![
        event("attach", attach(1, 1, 1), &[]),
        event("allocate", allocate(2, 1, 1, 0, &[10]), &[Handed("attach")]),
        event(
            "moved-in",
            moved_in(3, 2, 2, &[(0, 40, false), (1, 41, true)]),
            &[],
        ),
        event("release", release(2, 0, 1), &[Handed("moved-in")]),
        event("moved-out", moved_out(1), &[Handed("allocate")]),
        event("b1", Hand::Batch, &[]),
        event("b2", Hand::Batch, &[Handed("b1")]),
    ]
}

/// The owner a request names, among those whose completions give entries.
fn named(request: &Request) -> Option<u64> {
    match request {
        Request::Attach { owner, .. }
        | Request::Allocate { owner, .. }
        | Request::Pin { owner, .. }
        | Request::Transfer { owner, .. }
        | Request::Load { owner, .. }
        | Request::Clone { owner, .. } => Some(owner.0),
        _ => None,
    }
}

/// The owner whose entries event `index` gives, when it gives any. A copy's completion gives none:
/// the copy runs whatever became of its owner.
fn gives(recording: &[Event], index: usize) -> Option<u64> {
    match &recording[index].hand {
        Hand::Map {
            request,
            completion,
            ..
        } if !matches!(completion, Completion::Copy { .. }) => named(request),
        Hand::Answer { copy, .. } => recording.iter().find_map(|event| match &event.hand {
            Hand::Map { id, request, .. } if id == copy => named(request),
            _ => None,
        }),
        _ => None,
    }
}

/// The owner a stamped request takes every entry of: a `Detach`, or a `Transfer` away.
fn leaves(hand: &Hand) -> Option<u64> {
    match hand {
        Hand::Stamp(Request::Detach { owner, .. } | Request::Transfer { owner, .. }) => {
            Some(owner.0)
        }
        _ => None,
    }
}

// ---- The interleavings ----

/// One step of an interleaving.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Step {
    /// The scheduler hands the event of this index.
    Hand(usize),
    /// The rank of this index ends the oldest batch it has not ended.
    End(usize),
}

/// What decides the steps an interleaving may take next.
#[derive(Clone)]
struct Progress {
    /// Per event, once it was handed, the number of batches handed before it.
    handed: Vec<Option<u64>>,
    /// Per event, whether it was dropped.
    dropped: Vec<bool>,
    batches: u64,
    /// Per rank, the last batch it ended.
    ended: [u64; RANKS],
    /// The copies handed that no batch started and no give-up dropped.
    queued: Vec<u64>,
    /// The batch that started each copy.
    started: BTreeMap<u64, u64>,
}

impl Progress {
    fn new(recording: &[Event]) -> Self {
        Self {
            handed: vec![None; recording.len()],
            dropped: vec![false; recording.len()],
            batches: 0,
            ended: [0; RANKS],
            queued: Vec::new(),
            started: BTreeMap::new(),
        }
    }

    fn index(recording: &[Event], name: &str) -> usize {
        recording
            .iter()
            .position(|event| event.name == name)
            .unwrap()
    }

    fn met(&self, recording: &[Event], need: Need) -> bool {
        match need {
            Need::Handed(name) => {
                let index = Self::index(recording, name);
                self.handed[index].is_some() || self.dropped[index]
            }
            Need::Written(name) => self.handed[Self::index(recording, name)]
                .is_some_and(|before| self.batches > before),
        }
    }

    /// The steps that may come next: events in the recording's order, then ends.
    fn enabled(&self, recording: &[Event]) -> Vec<Step> {
        let mut enabled = Vec::new();

        for (index, event) in recording.iter().enumerate() {
            if self.handed[index].is_some() || self.dropped[index] {
                continue;
            }
            // A copy's `Finish` is answered once `completed` reached the batch that ran it.
            let answerable = match &event.hand {
                Hand::Answer { copy, .. } => self
                    .started
                    .get(copy)
                    .is_some_and(|batch| self.ended.iter().all(|ended| ended >= batch)),
                _ => true,
            };
            if answerable && event.needs.iter().all(|need| self.met(recording, *need)) {
                enabled.push(Step::Hand(index));
            }
        }

        for (rank, ended) in self.ended.iter().enumerate() {
            if *ended < self.batches {
                enabled.push(Step::End(rank));
            }
        }
        enabled
    }

    fn take(&mut self, recording: &[Event], step: Step) {
        let index = match step {
            Step::End(rank) => {
                self.ended[rank] += 1;
                return;
            }
            Step::Hand(index) => index,
        };
        self.handed[index] = Some(self.batches);

        let hand = &recording[index].hand;
        match hand {
            Hand::Batch => {
                self.batches += 1;
                for copy in self.queued.drain(..) {
                    self.started.insert(copy, self.batches);
                }
            }
            Hand::Map {
                id,
                completion: Completion::Copy { .. },
                ..
            } => self.queued.push(*id),
            Hand::Stamp(Request::Finish { copy, .. }) => {
                self.queued.retain(|queued| *queued != copy.0);
            }
            _ => {}
        }

        // Obligation 1: what gives the leaving owner entries is dropped once its leave is handed.
        if let Some(owner) = leaves(hand) {
            for other in 0..recording.len() {
                if self.handed[other].is_none() && gives(recording, other) == Some(owner) {
                    self.dropped[other] = true;
                }
            }
        }
    }
}

/// Whether `a` and `b` lead from a state to the same state and the same observations in either
/// order: a hand-off and the end of a rank's batch, unless the hand-off waits for `completed`.
fn independent(recording: &[Event], a: Step, b: Step) -> bool {
    match (a, b) {
        (Step::Hand(event), Step::End(_)) | (Step::End(_), Step::Hand(event)) => {
            !matches!(recording[event].hand, Hand::Answer { .. })
        }
        _ => false,
    }
}

/// Every interleaving of `recording` up to the order of independent steps: a depth-first search
/// that puts a step to sleep once its subtree is explored, and keeps it asleep below the steps
/// independent of it.
fn interleavings(recording: &[Event]) -> Vec<Vec<Step>> {
    fn explore(
        recording: &[Event],
        progress: &Progress,
        path: &mut Vec<Step>,
        mut asleep: BTreeSet<Step>,
        paths: &mut Vec<Vec<Step>>,
    ) {
        let enabled = progress.enabled(recording);
        if enabled.is_empty() {
            paths.push(path.clone());
            return;
        }

        for step in enabled {
            if asleep.contains(&step) {
                continue;
            }
            let mut next = progress.clone();
            next.take(recording, step);
            let still = asleep
                .iter()
                .copied()
                .filter(|other| independent(recording, step, *other))
                .collect();

            path.push(step);
            explore(recording, &next, path, still, paths);
            path.pop();
            asleep.insert(step);
        }
    }

    let mut paths = Vec::new();
    explore(
        recording,
        &Progress::new(recording),
        &mut Vec::new(),
        BTreeSet::new(),
        &mut paths,
    );
    paths
}

// ---- What every rank must hold ----

/// What every rank's tables must hold after the jobs handed so far, by §11: each completion's
/// runs at the positions its request named, writable when the block is the owner's private one,
/// until a stamped request takes them away or a `Commit` seals them.
#[derive(Default)]
struct Expected {
    /// The slot each owner's entries are kept under.
    slots: BTreeMap<u64, u32>,
    /// The run at each position, whether a batch may write it, and its owner.
    entries: BTreeMap<Position, (Run, bool, u64)>,
    /// The positions `Allocated` gave since the last batch. A reset writes the units, so the
    /// next batch may reset those that are still writable, and no other.
    fresh: BTreeSet<Position>,
    /// The copies no batch started, in the order their completions were handed.
    queued: Vec<Copies>,
    /// Per copy of a load or a clone, its owner and the entries its answer gives.
    awaited: BTreeMap<u64, (u64, Vec<Entry>)>,
    /// The destinations of the copies handed that no answer made entries: their copy alone
    /// writes them.
    destinations: BTreeSet<Run>,
}

/// What a batch must see on every rank.
struct View {
    tables: BTreeMap<Position, (Run, bool)>,
    /// The positions whose units the batch may reset.
    fresh: BTreeSet<Position>,
    copies: Vec<Copies>,
    destinations: BTreeSet<Run>,
}

impl Expected {
    fn give(&mut self, owner: u64, slot: u32, entry: Entry) {
        let (class, position, run, writable) = entry;
        self.entries
            .insert((slot, class, position), (run, writable, owner));
    }

    /// Queues the copies of request `id`, whose answer gives `fills` (owner, class, first
    /// position, whether a batch may write them) their destinations.
    fn queue(&mut self, id: u64, from: &[Run], to: &[Run], fills: Option<(u64, u8, u32, bool)>) {
        let runs = from.iter().copied().zip(to.iter().copied()).collect();
        self.queued.push((id, runs));
        self.destinations.extend(to.iter().copied());

        if let Some((owner, class, at, writable)) = fills {
            let entries = (0..)
                .zip(to)
                .map(|(index, run)| (class, at + index, *run, writable))
                .collect();
            self.awaited.insert(id, (owner, entries));
        }
    }

    fn map(&mut self, slot: u32, id: u64, request: &Request, completion: &Completion) {
        match (request, completion) {
            (Request::Attach { owner, .. }, Completion::Attached { slot }) => {
                self.slots.insert(owner.0, *slot);
            }
            (
                Request::Allocate {
                    owner, class, at, ..
                },
                Completion::Allocated { runs, .. },
            ) => {
                for (index, run) in (0..).zip(runs) {
                    self.give(owner.0, slot, (*class, at + index, *run, true));
                    self.fresh.insert((slot, *class, at + index));
                }
            }
            (
                Request::Pin {
                    owner, class, at, ..
                },
                Completion::Pinned { blocks, .. },
            ) => {
                for (index, block) in (0..).zip(blocks) {
                    // The model's ranks hold every class in one memory: a block of one run.
                    let [run] = block.as_slice() else {
                        panic!("a pinned block of the model states {} runs", block.len());
                    };
                    self.give(owner.0, slot, (*class, at + index, *run, false));
                }
            }
            (Request::Transfer { owner, .. }, Completion::Transferred { slot, runs }) => {
                self.slots.insert(owner.0, *slot);
                for entry in runs {
                    self.give(owner.0, *slot, *entry);
                }
            }
            (
                Request::Load {
                    owner, class, at, ..
                },
                Completion::Copy { from, to, .. },
            ) => {
                self.queue(id, from, to, Some((owner.0, *class, *at, false)));
            }
            (
                Request::Clone {
                    owner, class, at, ..
                },
                Completion::Copy { from, to, .. },
            ) => {
                self.queue(id, from, to, Some((owner.0, *class, *at, true)));
            }
            (_, Completion::Copy { from, to, .. }) => self.queue(id, from, to, None),
            (
                Request::Finish { copy, .. },
                Completion::Loaded | Completion::Duplicate | Completion::Cloned,
            ) => {
                if let Some((owner, entries)) = self.awaited.remove(&copy.0) {
                    for entry in entries {
                        self.destinations.remove(&entry.2);
                        self.give(owner, slot, entry);
                    }
                }
            }
            _ => {}
        }
    }

    fn stamp(&mut self, request: &Request) {
        match request {
            Request::Release {
                owner,
                class,
                at,
                count,
                ..
            } => {
                let released = *at..at + count;
                self.entries.retain(|&(_, held, position), (.., holder)| {
                    *holder != owner.0 || held != *class || !released.contains(&position)
                });
            }
            Request::Detach { owner, .. } | Request::Transfer { owner, .. } => {
                self.slots.remove(&owner.0);
                self.entries.retain(|_, (.., holder)| *holder != owner.0);
                self.awaited.retain(|_, (holder, _)| *holder != owner.0);
            }
            Request::Commit {
                owner, class, at, ..
            } => {
                let slot = self.slots.get(&owner.0).copied().unwrap_or_default();
                if let Some((_, writable, holder)) = self.entries.get_mut(&(slot, *class, *at))
                    && *holder == owner.0
                {
                    *writable = false;
                }
            }
            // A give-up drops a copy no batch started; a copy a batch ran ended with it.
            Request::Finish { copy, .. } => {
                if let Some(index) = self.queued.iter().position(|(id, _)| *id == copy.0) {
                    let (_, runs) = self.queued.remove(index);
                    self.awaited.remove(&copy.0);
                    for (_, to) in runs {
                        self.destinations.remove(&to);
                    }
                }
            }
            _ => {}
        }

        let entries = &self.entries;
        self.fresh.retain(|position| entries.contains_key(position));
    }

    /// The next batch: what every rank must see when it runs it, with the copies it starts.
    fn batch(&mut self) -> View {
        View {
            tables: self
                .entries
                .iter()
                .map(|(&position, &(run, writable, _))| (position, (run, writable)))
                .collect(),
            fresh: std::mem::take(&mut self.fresh)
                .into_iter()
                .filter(|position| self.entries.get(position).is_some_and(|entry| entry.1))
                .collect(),
            copies: std::mem::take(&mut self.queued),
            destinations: self.destinations.clone(),
        }
    }
}

impl View {
    /// Whether `seen`, what a rank's runner stated of the batch, is what the batch must see.
    fn check(&self, seen: &Seen) -> std::result::Result<(), String> {
        let at = format!("batch {} on rank {}", seen.batch, seen.rank);
        let runs = |tables: &BTreeMap<Position, (Run, bool)>| -> BTreeMap<Position, Run> {
            tables
                .iter()
                .map(|(&position, &(run, _))| (position, run))
                .collect()
        };
        let given = runs(&self.tables);

        if runs(&seen.tables) != given {
            return Err(format!(
                "obligation 1: {at} reads the entries {:?}; the completions handed before it give \
                 {given:?}, less what the requests stamped before it took away",
                runs(&seen.tables)
            ));
        }
        if seen.device != given {
            return Err(format!(
                "obligation 1: {at} reads the device table {:?}, written from the tables' \
                 changes, and the tables hold {given:?}",
                seen.device
            ));
        }
        if seen.tables != self.tables {
            return Err(format!(
                "obligation 4: {at} may write {:?}; only {:?} came from `Allocated`, `Cloned` or \
                 a private block of `Transferred` with no `Commit` since",
                writable(&seen.tables),
                writable(&self.tables)
            ));
        }
        if let Some(position) = seen.resets.difference(&self.fresh).next() {
            return Err(format!(
                "obligation 4: {at} resets the units at {position:?}, which no `Allocated` gave \
                 since the rank's previous batch, or which a `Commit` sealed"
            ));
        }
        if seen.copies != self.copies {
            return Err(format!(
                "obligation 4: {at} starts the copies {:?}; those handed before it and neither \
                 started nor given up are {:?}",
                seen.copies, self.copies
            ));
        }
        if let Some((position, (run, _))) = seen
            .tables
            .iter()
            .find(|(_, (run, _))| self.destinations.contains(run))
        {
            return Err(format!(
                "obligation 4: {at} addresses {run:?} at {position:?}, the destination of a copy \
                 whose answer has not given it"
            ));
        }
        Ok(())
    }
}

/// The positions of `tables` a batch may write.
fn writable(tables: &BTreeMap<Position, (Run, bool)>) -> Vec<Position> {
    tables
        .iter()
        .filter(|(_, (_, writable))| *writable)
        .map(|(position, _)| *position)
        .collect()
}

// ---- The ranks ----

/// What a rank's runner stated of one batch when it began to run it.
#[derive(Debug)]
struct Seen {
    rank: usize,
    batch: u64,
    /// Every entry of the tables, with whether a batch may write it.
    tables: BTreeMap<Position, (Run, bool)>,
    /// The device table the runner wrote from the tables' changes.
    device: BTreeMap<Position, Run>,
    /// The positions whose units the changes since the rank's previous batch asked to reset.
    resets: BTreeSet<Position>,
    /// The copies that started with the batch.
    copies: Vec<Copies>,
}

/// What the ranks of one group share with the test: the group's `completed` as the scheduler
/// node knows it, the word the buffer pool reads, and what a runner found wrong.
struct Watch {
    sequences: Arc<Sequences>,
    /// The buffer pool's end of the queue pair whose register 0 holds the group's word.
    pool: Mutex<QueuePair>,
    /// A second handle of the buffer pool's end, whose doorbell the test sleeps on until the
    /// group raises its word: the ranks' threads read the word through `pool` meanwhile.
    waiter: Mutex<QueuePair>,
    violations: Mutex<Vec<String>>,
}

impl Watch {
    fn word(&self) -> u64 {
        lock(&self.pool).register(0).load(Ordering::Acquire)
    }

    /// Notes a breach of obligation 3 when the group's `completed`, or the word the buffer pool
    /// reads, has reached batch `batch`, whose end `rank` has not reported.
    fn below(&self, rank: usize, batch: u64) {
        let completed = self.sequences.completed();
        let word = self.word();

        if completed >= batch || word >= batch {
            lock(&self.violations).push(format!(
                "obligation 3: the group's completed is {completed} and the buffer pool reads \
                 {word} while rank {rank} has not ended batch {batch}"
            ));
        }
    }
}

/// How a [`Reader`] serves a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Turns {
    /// In `Runner::execute`, which holds the worker until the test releases the batch.
    One,
    /// As a step in flight: `Runner::start` states what the batch reads and returns, and the
    /// test's release of the batch wakes the worker for `Runner::execute`, which lands it.
    Two,
}

/// What `Runner::start` is given to wake the worker with.
type Resume = Arc<dyn Fn() + Send + Sync>;

/// The steps a two-turn reader launched and the test has not released, oldest first; `closed`
/// once the test dropped its end, after which every step lands at once.
#[derive(Default)]
struct Flights {
    steps: VecDeque<Resume>,
    closed: bool,
}

/// A rank's runner over [`Graphs::meta`]: it states what each batch reads, and holds the batch
/// until the test releases it, which is when the batch's device work ends.
struct Reader {
    rank: usize,
    phases: [Phase; 1],
    graphs: Graphs,
    turns: Turns,
    /// The device block table, as the runner writes it from the changes it is handed.
    device: BTreeMap<Position, Run>,
    resets: BTreeSet<Position>,
    /// The copies issued for the batch that runs next.
    copies: Vec<Copies>,
    entered: Sender<Seen>,
    /// The test's releases, which a one-turn reader takes itself; a two-turn reader's releases
    /// are taken by a thread of its own, which wakes the worker for the oldest step in `flights`.
    release: Option<Receiver<()>>,
    flights: Arc<Mutex<Flights>>,
    /// The batches launched by `start` whose landing has not run.
    launched: BTreeSet<u64>,
    /// Where the runner reports a batch ended, once it checked `completed` at its end.
    left: Sender<u64>,
    watch: Arc<Watch>,
}

impl Reader {
    /// What `batch` reads of `tables` and of what the runner was handed since its last batch.
    fn seen(&mut self, batch: &Batch, tables: &BlockTables) -> Seen {
        let mut entries = BTreeMap::new();
        for slot in SLOTS {
            for (position, run) in tables.entries(slot, CLASS) {
                let writable = tables.writable(slot, CLASS, position);
                entries.insert((slot, CLASS, position), (run, writable));
            }
        }
        Seen {
            rank: self.rank,
            batch: batch.id.0,
            tables: entries,
            device: self.device.clone(),
            resets: std::mem::take(&mut self.resets),
            copies: std::mem::take(&mut self.copies),
        }
    }

    /// Batch `number` ended: the last moment before the worker learns it, `completed` only rises,
    /// so it has not reached the batch at any earlier moment either.
    fn left(&self, number: u64) {
        self.watch.below(self.rank, number);
        let _ = self.left.send(number);
    }
}

/// Takes the test's releases of a two-turn reader's steps: each wakes the worker for the oldest
/// step in flight, and once the test drops its end every step left lands, and so does every
/// later one.
fn release_flights(released: Receiver<()>, flights: Arc<Mutex<Flights>>) {
    std::thread::spawn(move || {
        while released.recv().is_ok() {
            let resume = lock(&flights).steps.pop_front();
            if let Some(resume) = resume {
                resume();
            }
        }

        let left = {
            let mut flights = lock(&flights);
            flights.closed = true;
            std::mem::take(&mut flights.steps)
        };
        for resume in left {
            resume();
        }
    });
}

impl Forward for Reader {
    fn buckets(&self) -> Vec<Bucket> {
        Vec::new()
    }

    fn forward(&self, _bucket: &Bucket) -> Result<Vec<Tensor>> {
        Ok(Vec::new())
    }

    fn idle_inputs(&self, _bucket: &Bucket, _columns: &mut [u8]) -> Result<()> {
        Ok(())
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }
}

impl Runner for Reader {
    fn phases(&self) -> &[Phase] {
        &self.phases
    }

    fn capturer(&self, graphs: Graphs) -> Result<Box<dyn Forward>> {
        Ok(Box::new(HostForward::new(graphs)))
    }

    fn prepare_inputs(&mut self, _batch: &Batch, _tables: &BlockTables) -> Result<Bucket> {
        Err(Error::Unsupported("the reader runs no forward".to_owned()))
    }

    /// A two-turn reader launches the batch: it states what the batch reads now, as a device
    /// step reads its tables when it is issued, and returns with the step in flight.
    fn start(&mut self, batch: &Batch, tables: &BlockTables, resume: Resume) -> Result<bool> {
        if self.turns == Turns::One {
            return Ok(false);
        }

        // The step is queued for its release before the test can see it and release it.
        {
            let mut flights = lock(&self.flights);
            if flights.closed {
                return Ok(false);
            }
            flights.steps.push_back(resume);
        }
        self.launched.insert(batch.id.0);
        let seen = self.seen(batch, tables);
        let _ = self.entered.send(seen);
        Ok(true)
    }

    fn execute(&mut self, batch: &Batch, tables: &BlockTables) -> Result<BatchOutput> {
        let number = batch.id.0;

        // The landing of a step `start` launched, which the test's release woke: it reads no
        // table.
        if self.launched.remove(&number) {
            self.left(number);
            return Ok(BatchOutput::default());
        }

        // The batch's device work ends when the test releases it, or drops its end at a close.
        let seen = self.seen(batch, tables);
        let _ = self.entered.send(seen);
        if let Some(release) = &self.release {
            let _ = release.recv();
        }
        self.left(number);
        Ok(BatchOutput::default())
    }

    fn write_tables(&mut self, changes: &[Change]) -> Result<()> {
        for change in changes {
            let position = (change.slot, change.class, change.position);
            match change.run {
                Some(run) => self.device.insert(position, run),
                None => self.device.remove(&position),
            };
            if change.reset {
                self.resets.insert(position);
            }
        }
        Ok(())
    }

    /// The rank's KV units and their host pool, which its tables map.
    fn maps(&mut self, region: u8) -> bool {
        [KV, HOST].contains(&region)
    }

    fn copy(&mut self, copies: &[BlockCopy]) -> Result<()> {
        self.copies.extend(
            copies
                .iter()
                .map(|copy| (copy.request.0, copy.runs.clone())),
        );
        Ok(())
    }
}

/// The test's ends of a rank: the batches it runs, their release, and the report of their end.
struct Ends {
    seen: Receiver<Seen>,
    release: Sender<()>,
    left: Receiver<u64>,
}

/// The worker with no device of rank `rank` (the test's number), reporting to `sequences` as
/// rank `place`, its reader serving batches in `turns`, with the test's ends of it.
fn worker(
    rank: usize,
    place: usize,
    sequences: &Arc<Sequences>,
    watch: &Arc<Watch>,
    turns: Turns,
) -> (Arc<Worker>, Ends) {
    let (entered, seen) = crossbeam_channel::unbounded();
    let (release, released) = crossbeam_channel::unbounded();
    let (left, ended) = crossbeam_channel::unbounded();
    let kernels = Arc::new(Generation::new(1, []));

    let flights = Arc::new(Mutex::new(Flights::default()));
    let release_taken = match turns {
        Turns::One => Some(released),
        Turns::Two => {
            release_flights(released, Arc::clone(&flights));
            None
        }
    };
    let runner = Reader {
        rank,
        phases: [Phase::Decode],
        graphs: Graphs::meta(Arc::clone(&kernels), 0),
        turns,
        device: BTreeMap::new(),
        resets: BTreeSet::new(),
        copies: Vec::new(),
        entered,
        release: release_taken,
        flights,
        launched: BTreeSet::new(),
        left,
        watch: Arc::clone(watch),
    };
    let worker = Worker::spawn(
        None,
        vec![Box::new(runner)],
        kernels,
        BlockTables::new(GROUP, &[KV, HOST], &UNIT_BYTES),
        Arc::clone(sequences),
        place,
    )
    .unwrap();

    let ends = Ends {
        seen,
        release,
        left: ended,
    };
    (Arc::new(worker), ends)
}

/// What the replays of one recording share: the node's peer listener, and the peer's liveness,
/// which notes the close of each replay's link as a loss.
struct Peers {
    listener: service::Listener,
    liveness: Arc<Liveness>,
}

impl Peers {
    fn new() -> Self {
        Self {
            listener: service::Listener::bind(&"127.0.0.1:0".parse().unwrap()).unwrap(),
            liveness: Liveness::new(Arc::new(|| {})),
        }
    }
}

/// One group under test: the scheduler node's route over its rank and the link to the peer's.
struct Group {
    route: Route,
    ends: [Ends; RANKS],
    watch: Arc<Watch>,
    remote: Arc<Remote>,
    upstream: Upstream,
}

impl Group {
    fn start(peers: &Peers, turns: Turns) -> Self {
        let pool = QueuePair::create(4096, 1).unwrap();
        let engine = QueuePair::open(pool.duplicate().unwrap(), true).unwrap();
        let waiter = QueuePair::open(pool.duplicate().unwrap(), false).unwrap();
        let sequences = Sequences::new(GROUP, RANKS, vec![(engine, 0)]);
        let watch = Arc::new(Watch {
            sequences: Arc::clone(&sequences),
            pool: Mutex::new(pool),
            waiter: Mutex::new(waiter),
            violations: Mutex::default(),
        });

        // The peer process: its one rank of the group, serving what the node hands it.
        let address = peers.listener.endpoint().to_string();
        let dialling =
            std::thread::spawn(move || register(&address, PEER as u32, ENGINE, LONG).unwrap());
        let (first, channel) = accept(&peers.listener, 1, ENGINE, LONG)
            .unwrap()
            .pop()
            .unwrap();
        let held = Sequences::new(GROUP, 1, Vec::new());
        let (far, far_ends) = worker(PEER, 0, &held, &watch, turns);
        let ranks = Route::new(vec![held], vec![far], Vec::new()).unwrap();
        // The scheduler node runs no operation of the data plane here.
        let operate: engine::executor::peers::Operate =
            Box::new(|_| Err("this peer runs no operation".to_owned()));
        let upstream = Upstream::serve(
            dialling.join().unwrap(),
            Arc::new(ranks),
            Arc::clone(&peers.liveness),
            operate,
        )
        .unwrap();

        // The scheduler node: its rank 0, and the link that holds rank 1.
        let (near, near_ends) = worker(0, 0, &sequences, &watch, turns);
        let remote = Remote::spawn(first, channel, Liveness::new(Arc::new(|| {}))).unwrap();
        let route = Route::new(
            vec![sequences],
            vec![near],
            vec![(vec![GROUP], Arc::clone(&remote))],
        )
        .unwrap();

        Self {
            route,
            ends: [near_ends, far_ends],
            watch,
            remote,
            upstream,
        }
    }

    /// Ends the group: every batch still held ends, the node closes its link first, so it never
    /// reads the peer's close as a loss, and every rank's thread is joined.
    fn close(self) {
        let Self {
            route,
            ends,
            remote,
            upstream,
            ..
        } = self;

        drop(ends);
        remote.close();
        upstream.close();
        drop(route);
    }
}

// ---- A replay ----

/// A job handed to the group, with what its submission must return.
struct Job {
    event: usize,
    submission: Submission,
    returns: Vec<Request>,
    /// The last batch every rank must have ended for the job to have completed: those handed
    /// before it, or a batch's own number.
    due: u64,
    /// What it returned, once that was checked.
    returned: Option<Vec<Request>>,
}

/// One interleaving replayed against a new group, with what the group must do at every step.
struct Replay<'a> {
    recording: &'a [Event],
    group: Group,
    expected: Expected,
    /// Per batch handed, in order, what every rank must see when it runs it.
    views: Vec<View>,
    /// Per rank, the last batch it ended.
    ended: [u64; RANKS],
    jobs: Vec<Job>,
    /// The job of each batch, by number.
    batches: BTreeMap<u64, usize>,
    /// The batch that started each copy.
    started: BTreeMap<u64, u64>,
}

/// `request`, a request the workers stamp, with the stamps `stamps`.
fn stamped(request: &Request, stamps: Stamps) -> Request {
    let mut request = request.clone();
    match &mut request {
        Request::Detach { after, .. }
        | Request::Release { after, .. }
        | Request::Commit { after, .. }
        | Request::Transfer { after, .. }
        | Request::Finish { after, .. } => *after = stamps,
        _ => {}
    }
    request
}

/// The group's batch numbered `number`: one row of one request, which no forward reads. A batch
/// with a request is one a runner may serve in two turns (`Runner::start`).
fn batch(number: u64) -> Batch {
    Batch {
        id: BatchId(number),
        group: GROUP,
        phase: Phase::Decode,
        kernels: 1,
        policy: 1,
        revision: 0,
        copy_budget: proto::config::DEFAULT_COPY_BUDGET,
        shape: Shape::decode(1, Variant::DecodeGreedy),
        requests: vec![proto::request::RequestId(number)],
        columns: Columns::default(),
        media: Vec::new(),
        new_slots: Vec::new(),
    }
}

impl<'a> Replay<'a> {
    fn new(recording: &'a [Event], peers: &Peers, turns: Turns) -> Self {
        Self {
            recording,
            group: Group::start(peers, turns),
            expected: Expected::default(),
            views: Vec::new(),
            ended: [0; RANKS],
            jobs: Vec::new(),
            batches: BTreeMap::new(),
            started: BTreeMap::new(),
        }
    }

    /// Takes `step`, then checks what it must have brought about.
    fn step(&mut self, step: Step) -> std::result::Result<(), String> {
        match step {
            Step::Hand(index) => self.hand(index)?,
            Step::End(rank) => self.end(rank)?,
        }

        if let Some(violation) = lock(&self.group.watch.violations).first() {
            return Err(violation.clone());
        }
        let reached = self.ended.iter().copied().min().unwrap_or_default();
        for index in 0..self.jobs.len() {
            if self.jobs[index].returned.is_none() && self.jobs[index].due <= reached {
                self.check(index)?;
            }
        }
        self.converge()
    }

    fn push(&mut self, event: usize, submitted: Result<Submission>, returns: Vec<Request>) {
        // A job is pushed once its batch, if it is one, has its view: either way every rank has
        // reached it once it ended the batches handed so far.
        self.jobs.push(Job {
            event,
            submission: submitted.unwrap(),
            returns,
            due: self.views.len() as u64,
            returned: None,
        });
    }

    fn hand(&mut self, index: usize) -> std::result::Result<(), String> {
        match self.recording[index].hand.clone() {
            Hand::Map {
                slot,
                id,
                request,
                completion,
            } => {
                self.expected.map(slot, id, &request, &completion);
                let submitted =
                    self.group
                        .route
                        .map(GROUP, slot, RequestId(id), request, completion);
                self.push(index, submitted, Vec::new());
            }
            Hand::Stamp(request) => {
                // Every rank applies the request after the batches handed before it and before
                // those handed after it.
                self.expected.stamp(&request);
                let returns = vec![stamped(&request, vec![(GROUP, self.views.len() as u64)])];
                let submitted = self.group.route.stamp(GROUP, vec![request]);
                self.push(index, submitted, returns);
            }
            Hand::Batch => {
                let number = self.views.len() as u64 + 1;
                let view = self.expected.batch();
                let returns = view
                    .copies
                    .iter()
                    .map(|(copy, _)| Request::Finish {
                        copy: RequestId(*copy),
                        ok: true,
                        after: vec![(GROUP, number)],
                    })
                    .collect();
                for (copy, _) in &view.copies {
                    self.started.insert(*copy, number);
                }
                self.views.push(view);

                let submitted = self.group.route.submit(batch(number));
                self.batches.insert(number, self.jobs.len());
                self.push(index, submitted, returns);
            }
            Hand::Answer { slot, copy, answer } => {
                // The scheduler hands the answer with the `Finish` it submitted: the one the
                // batch that ran the copy returned, which every rank has ended.
                let job = self.batches[&self.started[&copy]];
                let finish = self.jobs[job]
                    .returned
                    .iter()
                    .flatten()
                    .find(|request| {
                        matches!(request, Request::Finish { copy: ran, .. } if ran.0 == copy)
                    })
                    .cloned()
                    .ok_or_else(|| {
                        format!(
                            "obligation 2: the batch that ran copy {copy} returned no `Finish` for \
                             it"
                        )
                    })?;

                self.expected.map(slot, ANSWERS + copy, &finish, &answer);
                let submitted =
                    self.group
                        .route
                        .map(GROUP, slot, RequestId(ANSWERS + copy), finish, answer);
                self.push(index, submitted, Vec::new());
            }
        }
        Ok(())
    }

    fn end(&mut self, rank: usize) -> std::result::Result<(), String> {
        let batch = self.ended[rank] + 1;
        let ends = &self.group.ends[rank];
        let seen = ends
            .seen
            .recv_timeout(LONG)
            .map_err(|_| format!("rank {rank} did not run batch {batch}"))?;
        if seen.batch != batch {
            return Err(format!(
                "rank {rank} ran batch {} where batch {batch} was next",
                seen.batch
            ));
        }
        self.views[batch as usize - 1].check(&seen)?;

        ends.release.send(()).unwrap();
        match ends.left.recv_timeout(LONG) {
            Ok(left) if left == batch => {}
            _ => return Err(format!("rank {rank} did not end batch {batch}")),
        }
        self.ended[rank] = batch;
        Ok(())
    }

    /// Whether job `index`, which every rank has reached, returned what it must.
    fn check(&mut self, index: usize) -> std::result::Result<(), String> {
        let job = &mut self.jobs[index];
        let name = self.recording[job.event].name;

        match job.submission.wait_timeout(LONG) {
            Some(Outcome::Success((_, returned))) if returned == job.returns => {
                job.returned = Some(returned);
                Ok(())
            }
            Some(Outcome::Success((_, returned))) => Err(format!(
                "obligation 2: {name} returned {returned:?}; the group's launched sequence makes \
                 it {:?}",
                job.returns
            )),
            Some(Outcome::Failed(error)) => Err(format!("{name} failed: {error}")),
            Some(Outcome::Cancelled) => Err(format!("{name} was cancelled")),
            None => Err(format!(
                "{name} did not complete once every rank reached it"
            )),
        }
    }

    /// Waits until the group's `completed` and the buffer pool's word reach the last batch every
    /// rank ended, and fails when either passes it or stays below it.
    ///
    /// The group raises its `completed` before it writes the word and rings the buffer pool's
    /// doorbell, so the test sleeps on that doorbell between looks.
    fn converge(&self) -> std::result::Result<(), String> {
        let ended = self.ended.iter().copied().min().unwrap_or_default();
        let deadline = Instant::now() + LONG;
        let waiter = lock(&self.group.watch.waiter);

        loop {
            // The counts before the look: a word raised after it rings the doorbell.
            let seen = waiter.signals();
            let completed = self.group.route.completed(GROUP).unwrap();
            let word = self.group.watch.word();
            if completed > ended || word > ended {
                return Err(format!(
                    "obligation 3: the group's completed is {completed} and the buffer pool reads \
                     {word}, past batch {ended}, the last every rank ended"
                ));
            }
            if completed == ended && word == ended {
                return Ok(());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if !waiter
                .wait(Interest::REGISTERS, seen, Some(remaining))
                .unwrap()
            {
                return Err(format!(
                    "obligation 3: the group's completed stays at {completed} and the buffer pool \
                     reads {word} once every rank ended batch {ended}"
                ));
            }
        }
    }
}

/// `path` in words, for a failure.
fn describe(recording: &[Event], path: &[Step]) -> String {
    let words: Vec<String> = path
        .iter()
        .map(|step| match step {
            Step::Hand(index) => recording[*index].name.to_owned(),
            Step::End(rank) => format!("rank {rank} ends a batch"),
        })
        .collect();
    words.join(", ")
}

/// Replays every interleaving of `recording`, of which there are `count`, with readers serving
/// batches in `turns`, and fails at the first step where the group breaks an obligation, with the
/// steps that led to it.
fn replay_all(recording: &[Event], count: usize, turns: Turns) {
    let paths = interleavings(recording);
    assert_eq!(paths.len(), count, "the interleavings of the recording");

    let peers = Peers::new();
    for path in &paths {
        let mut replay = Replay::new(recording, &peers, turns);
        let failed = path
            .iter()
            .enumerate()
            .find_map(|(at, step)| replay.step(*step).err().map(|finding| (at + 1, finding)));

        replay.group.close();
        if let Some((steps, finding)) = failed {
            panic!(
                "{finding}\nwith readers in {turns:?} turns, after: {}",
                describe(recording, &path[..steps])
            );
        }
    }
}

/// 2,988 interleavings, with a batch held in its one turn.
#[test]
fn allocated_and_pinned_entries_are_read_until_their_release_and_detach() {
    replay_all(&entries(), 2_988, Turns::One);
}

/// The same 2,988 interleavings, with each step in flight across two turns.
#[test]
fn allocated_and_pinned_entries_are_read_until_their_release_and_detach_with_steps_in_flight() {
    replay_all(&entries(), 2_988, Turns::Two);
}

/// 4,004 interleavings, with a batch held in its one turn.
#[test]
fn a_loaded_or_cloned_block_is_an_entry_only_once_its_copy_is_answered() {
    replay_all(&copies(), 4_004, Turns::One);
}

/// The same 4,004 interleavings, with each step in flight across two turns.
#[test]
fn a_loaded_or_cloned_block_is_an_entry_only_once_its_copy_is_answered_with_steps_in_flight() {
    replay_all(&copies(), 4_004, Turns::Two);
}

/// 3,528 interleavings, with a batch held in its one turn.
#[test]
fn a_copy_given_up_before_a_batch_starts_it_never_runs() {
    replay_all(&give_up_store(), 3_528, Turns::One);
}

/// The same 3,528 interleavings, with each step in flight across two turns.
#[test]
fn a_copy_given_up_before_a_batch_starts_it_never_runs_with_steps_in_flight() {
    replay_all(&give_up_store(), 3_528, Turns::Two);
}

/// 1,260 interleavings, with a batch held in its one turn.
#[test]
fn an_owner_moving_in_brings_its_entries_and_one_moving_out_takes_them() {
    replay_all(&transfer(), 1_260, Turns::One);
}

/// The same 1,260 interleavings, with each step in flight across two turns.
#[test]
fn an_owner_moving_in_brings_its_entries_and_one_moving_out_takes_them_with_steps_in_flight() {
    replay_all(&transfer(), 1_260, Turns::Two);
}
