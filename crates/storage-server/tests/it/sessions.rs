//! Sessions of engine epochs with the buffer pool: an engine's main process
//! opens its epoch's session on the server's socket, and its schedulers'
//! requests travel on their queue pairs in shared memory, which the buffer
//! pool's thread serves.
//!
//! The test process plays the engine: its main process (a head of the
//! epoch), its schedulers (the engine's ends of the queue pairs, whose
//! `completed` registers they write as the workers would) and its ranks
//! (workers). No GPU is needed: the regions' memory is host storage.

use std::os::fd::{AsFd as _, OwnedFd};
use std::os::unix::fs::FileExt as _;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use ::storage::protocol::{
    BlockState, ErrorKind, Listing, PageLayout, UnitLayout, UnitRun, Waiting, planes,
};
use ::storage::runtime::{
    Cause, Class, Completion, Group, GroupId, Lender, Open, Opened, OwnerId, Region, Registers,
    Request, RequestId, Tier, VERSIONS,
};
use ::storage::{
    Access, Claim, Claimant, Client, Interest, Key, Location, Mode, QueueDescriptors, QueuePair,
    Role, Segment,
};

use crate::server::{
    Running, engine_head, head, host, key, memfd, scope, server_error, tensor, worker,
};
use crate::transfer::{Gpus, read, serving};

pub(crate) const PREFILL: GroupId = GroupId(0);
pub(crate) const DECODE: GroupId = GroupId(1);

/// The class of KV blocks: one unit each, host copies of one unit.
pub(crate) const KV: u8 = 0;

/// The indices of the layout's regions.
pub(crate) const UNITS: usize = 0;
pub(crate) const HOST: usize = 3;

/// A region of a layout, its units of `unit_bytes` in parts of `part_bytes`.
fn region(
    name: &str,
    tier: Tier,
    (unit_bytes, part_bytes): (u64, u64),
    reserved: u32,
    units: u32,
) -> Region {
    Region {
        name: name.into(),
        tier,
        space: name.into(),
        unit_bytes,
        part_bytes,
        reserved,
        units,
        limit: reserved + units,
        segments: Vec::new(),
        memory: String::new(),
    }
}

/// The layout of an engine epoch: a prefill group and a decode group on
/// memories of their own, each the only group of its scheduler, over one
/// KV region of eight units, their slot rows (two each, the epoch's own)
/// and a host region of four units of 32 bytes, two parts of 16.
pub(crate) fn layout(epoch: u64) -> Open {
    Open {
        oldest: VERSIONS.oldest(),
        newest: Some(VERSIONS.newest()),
        epoch,
        trial: false,
        regions: vec![
            region("units", Tier::Device, (64, 32), 1, 8),
            region(&format!("prefill rows {epoch}"), Tier::Device, (8, 8), 1, 2),
            region(&format!("decode rows {epoch}"), Tier::Device, (8, 8), 1, 2),
            region("host", Tier::Host, (32, 16), 0, 4),
        ],
        classes: vec![Class {
            units: 1,
            host: Some((HOST as u8, 1)),
            state_in_slot: false,
            tensors: [0; 32],
            encoding: [0; 32],
        }],
        trajectory: KV,
        groups: vec![
            Group {
                id: PREFILL,
                memory: "gpu0".into(),
                slots: 1,
                regions: vec![vec![UNITS as u8]],
                receives: Vec::new(),
                parts: Vec::new(),
            },
            Group {
                id: DECODE,
                memory: "gpu1".into(),
                slots: 2,
                regions: vec![vec![UNITS as u8]],
                receives: Vec::new(),
                parts: Vec::new(),
            },
        ],
        schedulers: vec![vec![PREFILL], vec![DECODE]],
    }
}

/// A scheduler's end of its queue pair and its registers.
pub(crate) struct Scheduler {
    pair: QueuePair,
    registers: Registers,
    number: u64,
}

impl Scheduler {
    /// Each scheduler of an answer to `Open`, in order.
    pub(crate) fn all(opened: &Opened, descriptors: Vec<QueueDescriptors>) -> Vec<Self> {
        opened
            .registers
            .iter()
            .zip(descriptors)
            .map(|(registers, descriptor)| Self {
                pair: QueuePair::open(descriptor, true).unwrap(),
                registers: registers.clone(),
                number: 0,
            })
            .collect()
    }

    /// Submits `request` under the next number.
    pub(crate) fn submit(&mut self, request: Request) -> RequestId {
        self.number += 1;
        let id = RequestId(self.number);
        let mut record = Vec::new();
        request.encode(id, &mut record);
        assert!(
            self.pair.submit(&record).unwrap(),
            "the submission queue is full"
        );
        id
    }

    /// The next completion, within the tests' patience.
    pub(crate) fn next(&mut self) -> (RequestId, Completion) {
        let deadline = Instant::now() + crate::PATIENCE;
        let mut record = Vec::new();
        loop {
            let seen = self.pair.signals();
            if self.pair.next_completion(&mut record).unwrap() {
                return Completion::decode(&record).unwrap();
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            let woken = self.pair.wait(Interest::RECORDS, seen, Some(remaining));
            assert!(woken.unwrap(), "no completion arrived");
        }
    }

    /// Submits `request` and returns its completion, the next to arrive.
    pub(crate) fn ask(&mut self, request: Request) -> Completion {
        let id = self.submit(request);
        let (answered, completion) = self.next();
        assert_eq!(answered, id, "{completion:?} answers another request");
        completion
    }

    /// Writes the `completed` word of the scheduler's group `index`, as its
    /// worker does.
    pub(crate) fn complete_batches(&self, index: usize, completed: u64) {
        let register = self.registers.completed[index] as usize;
        self.pair
            .register(register)
            .store(completed, Ordering::Release);
        self.pair.notify_registers();
    }

    /// Waits until region `region`'s capacity words read `words`.
    pub(crate) fn await_capacity(&self, region: u8, words: (u64, u64, u64)) {
        let (_, first) = *self
            .registers
            .capacity
            .iter()
            .find(|(number, _)| *number == region)
            .expect("the scheduler has the region's capacity words");
        let read = |offset: u32| {
            let register = self.pair.register((first + offset) as usize);
            register.load(Ordering::Acquire)
        };
        let deadline = Instant::now() + crate::PATIENCE;
        loop {
            let seen = self.pair.signals();
            if (read(0), read(1), read(2)) == words {
                return;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            let woken = self.pair.wait(Interest::REGISTERS, seen, Some(remaining));
            assert!(
                woken.unwrap(),
                "the capacity words read {:?}, not {words:?}",
                (read(0), read(1), read(2))
            );
        }
    }
}

/// The buffer pool of the deployment as an operator lists it.
pub(crate) fn listed(socket: &std::path::Path) -> ::storage::protocol::BufferPoolListing {
    let mut operator = Client::connect(socket, Role::Operator).unwrap();
    let listing = operator.list().unwrap();
    listing
        .buffer_pools
        .into_iter()
        .find(|pool| pool.deployment == "model")
        .expect("the deployment's buffer pool is listed")
}

#[test]
fn an_engine_opens_a_session_and_its_schedulers_are_served_on_their_queue_pairs() {
    serve_a_session(&Running::with_children(None));
}

/// An engine opens a session on `server`, its two schedulers are served on
/// their queue pairs, and it closes the session.
pub(crate) fn serve_a_session(server: &Running) {
    let mut engine = engine_head(&server.socket, 0);
    let (opened, descriptors) = engine.open_pool(layout(0)).unwrap();
    let units: Vec<u32> = opened.regions.iter().map(|(_, units)| *units).collect();
    assert_eq!(units, [8, 2, 2, 4]);
    let region = opened.regions[UNITS].0;
    let mut schedulers = Scheduler::all(&opened, descriptors);
    let [prefill, decode] = &mut schedulers[..] else {
        panic!("two schedulers were answered");
    };

    // An owner on the prefill group, two private blocks, a commit stamped
    // with the prefill group's batch 5.
    let attach = Request::Attach {
        owner: OwnerId(1),
        group: PREFILL,
    };
    assert_eq!(prefill.ask(attach), Completion::Attached { slot: 1 });
    let allocate = Request::Allocate {
        owner: OwnerId(1),
        class: KV,
        at: 0,
        count: 2,
        units: 0,
    };
    let allocated = Completion::Allocated {
        runs: vec![(region, 1, 1), (region, 2, 1)],
        evicted: Vec::new(),
    };
    assert_eq!(prefill.ask(allocate), allocated);
    let commit = prefill.submit(Request::Commit {
        owner: OwnerId(1),
        class: KV,
        at: 0,
        key: key(7),
        tokens: 64,
        after: vec![(PREFILL, 5)],
    });
    prefill.await_capacity(region, (6, 0, 0));

    // The commit takes effect when the group's `completed` reaches its stamp.
    prefill.complete_batches(0, 4);
    let release = Request::Release {
        owner: OwnerId(1),
        class: KV,
        at: 1,
        count: 1,
        after: Vec::new(),
    };
    assert_eq!(prefill.ask(release), Completion::Done);
    prefill.complete_batches(0, 5);
    assert_eq!(prefill.next(), (commit, Completion::Committed));

    // The owner's pin is a holder: the shared block is evictable once it leaves.
    prefill.await_capacity(region, (7, 0, 0));
    let detach = Request::Detach {
        owner: OwnerId(1),
        after: Vec::new(),
    };
    assert_eq!(prefill.ask(detach), Completion::Done);
    prefill.await_capacity(region, (7, 0, 1));

    // The decode scheduler's queue pair is its own: its group's memory does
    // not hold the block, so a pin finds nothing there.
    let attach = Request::Attach {
        owner: OwnerId(2),
        group: DECODE,
    };
    assert_eq!(decode.ask(attach), Completion::Attached { slot: 1 });
    let pin = Request::Pin {
        owner: OwnerId(2),
        class: KV,
        at: 0,
        keys: vec![key(7)],
    };
    let none = Completion::Pinned {
        blocks: Vec::new(),
        host: 0,
    };
    assert_eq!(decode.ask(pin), none);

    // A record that is no request is refused under its number.
    let mut record = 50_u64.to_le_bytes().to_vec();
    record.push(200);
    assert!(decode.pair.submit(&record).unwrap());
    let refused = Completion::Refused {
        cause: Cause::Request,
    };
    assert_eq!(decode.next(), (RequestId(50), refused));

    // The operator sees the shared block, the decode owner's slot row and
    // what each group last completed.
    let pool = listed(&server.socket);
    let shared = pool.blocks.iter().find(|count| count.region == "units");
    let shared = shared.map(|count| (count.class, count.state, count.blocks));
    assert_eq!(shared, Some((Some(KV), BlockState::Shared, 1)));
    let groups: Vec<(u8, u64)> = pool
        .groups
        .iter()
        .map(|group| (group.group, group.completed))
        .collect();
    assert_eq!(groups, [(0, 5), (1, 0)]);

    // Closing the session detaches its owners; what it committed stays.
    engine.close_pool().unwrap();
    let pool = listed(&server.socket);
    assert!(pool.groups.is_empty());
    assert_eq!(
        pool.blocks
            .iter()
            .map(|count| count.region.as_str())
            .collect::<Vec<_>>(),
        Vec::<&str>::new(),
        "the KV region closed with its last session, and its copies with it"
    );
}

/// The operator's `list` prints each deployment's buffer pool as `specs/buffer-pool.md` §8 states
/// it, a line each: the blocks of each region and class in each observed state, each worker group's `completed` word with the stamps of what waits on it,
/// and each request whose copies are in flight.
#[test]
fn the_operator_s_list_prints_the_buffer_pool_s_blocks_groups_and_copies() {
    let server = Running::with_children(None);
    let mut engine = engine_head(&server.socket, 0);
    let (opened, descriptors) = engine.open_pool(layout(0)).unwrap();
    let mut schedulers = Scheduler::all(&opened, descriptors);
    let prefill = &mut schedulers[0];

    // An owner on the prefill group holds a slot row and three private blocks.
    let attach = Request::Attach {
        owner: OwnerId(1),
        group: PREFILL,
    };
    assert_eq!(prefill.ask(attach), Completion::Attached { slot: 1 });
    let allocate = Request::Allocate {
        owner: OwnerId(1),
        class: KV,
        at: 0,
        count: 3,
        units: 0,
    };
    assert!(matches!(
        prefill.ask(allocate),
        Completion::Allocated { .. }
    ));

    // The first block is committed at once: shared, the owner's pin on it. The second's commit
    // waits for the prefill group's batch 9, and the second stays private. The third leaves the
    // owner's table at once, but its release waits for batch 9: only the waiting release holds
    // it, and it is retired.
    let commit = |at, byte, after| Request::Commit {
        owner: OwnerId(1),
        class: KV,
        at,
        key: key(byte),
        tokens: 64,
        after,
    };
    assert_eq!(prefill.ask(commit(0, 7, Vec::new())), Completion::Committed);
    prefill.submit(commit(1, 8, vec![(PREFILL, 9)]));
    let release = Request::Release {
        owner: OwnerId(1),
        class: KV,
        at: 2,
        count: 1,
        after: vec![(PREFILL, 9)],
    };
    assert_eq!(prefill.ask(release), Completion::Done);

    // A store of the shared block to the host: its copy, which the prefill group's workers
    // runs, is in flight until its `Finish`, and the host block it writes is private until then.
    let store = prefill.submit(Request::Store {
        group: PREFILL,
        keys: vec![key(7)],
        tier: Tier::Host,
    });
    let (answered, answer) = prefill.next();
    assert!(
        answered == store && matches!(answer, Completion::Copy { .. }),
        "{answered:?} answered {answer:?}"
    );

    // The operator's command prints the listing as JSON for tools.
    let mut printed = Vec::new();
    storage_server::list(
        &server.socket,
        storage_server::Rendering::Json,
        &mut printed,
    )
    .unwrap();
    let listing: Listing = serde_json::from_slice(&printed).unwrap();
    let [pool] = &listing.buffer_pools[..] else {
        panic!("one buffer pool is listed: {listing:?}");
    };
    assert_eq!(pool.deployment, "model");
    let blocks = |region: &str, class: Option<u8>, state: BlockState| {
        let found = pool
            .blocks
            .iter()
            .find(|count| count.region == region && count.class == class && count.state == state);
        found.map(|count| count.blocks)
    };
    assert_eq!(blocks("units", Some(0), BlockState::Shared), Some(1));
    assert_eq!(blocks("units", Some(0), BlockState::Private), Some(1));
    assert_eq!(blocks("units", Some(0), BlockState::Retired), Some(1));
    assert_eq!(blocks("host", Some(0), BlockState::Private), Some(1));
    assert_eq!(blocks("prefill rows 0", None, BlockState::Private), Some(1));
    assert_eq!(pool.blocks.len(), 5, "{pool:?}");

    let groups: Vec<(u8, u64)> = pool
        .groups
        .iter()
        .map(|group| (group.group, group.completed))
        .collect();
    assert_eq!(groups, [(0, 0), (1, 0)]);
    let waiting: Vec<&[(u64, Waiting)]> =
        pool.groups.iter().map(|group| &group.waiting[..]).collect();
    assert_eq!(
        waiting,
        [&[(9, Waiting::Commit), (9, Waiting::Release)][..], &[][..]]
    );
    let copies: Vec<(Option<u64>, u16, u64, u32)> = pool
        .copies
        .iter()
        .map(|copy| (copy.epoch, copy.scheduler, copy.request, copy.copies))
        .collect();
    assert_eq!(copies, [(Some(0), 0, store.0, 1)]);
}

#[test]
fn a_session_is_refused_what_the_server_does_not_serve() {
    let server = Running::with_children(None);
    let mut engine = engine_head(&server.socket, 0);

    // A window of the interface's versions that shares none with the server's, both named, and
    // another epoch than the head's.
    for version in [VERSIONS.oldest() - 1, VERSIONS.newest() + 1] {
        let mut other = layout(0);
        other.oldest = version;
        other.newest = None;
        let refused = engine.open_pool(other).unwrap_err();
        let ::storage::Error::Server { kind, message } = refused else {
            panic!("{refused}");
        };
        assert_eq!(kind, ErrorKind::Invalid);
        assert!(
            message.contains(&format!("speaks versions {version} "))
                && message.contains(&format!("buffer pool {VERSIONS}")),
            "{message}"
        );
    }
    assert_eq!(
        server_error(engine.open_pool(layout(1))),
        ErrorKind::Invalid
    );

    // One session per engine of an epoch.
    let (first, _) = engine.open_pool(layout(0)).unwrap();
    assert_eq!(server_error(engine.open_pool(layout(0))), ErrorKind::Held);

    // A region stated again under its name is that region, of its layout.
    let mut next = engine_head(&server.socket, 1);
    let mut restated = layout(1);
    restated.regions[UNITS].unit_bytes = 128;
    assert_eq!(server_error(next.open_pool(restated)), ErrorKind::Invalid);

    // At most 64 memories and host parts over the deployment's sessions.
    let mut wide = layout(1);
    for index in 2..66 {
        wide.groups.push(Group {
            id: GroupId(index),
            memory: format!("gpu{index}"),
            slots: 1,
            regions: vec![vec![UNITS as u8]],
            receives: Vec::new(),
            parts: Vec::new(),
        });
        wide.schedulers.push(vec![GroupId(index)]);
    }
    assert_eq!(server_error(next.open_pool(wide)), ErrorKind::Invalid);

    // Two epochs of one layout name the same KV region; their slot rows are
    // each epoch's own.
    let (second, _) = next.open_pool(layout(1)).unwrap();
    assert_eq!(second.regions[UNITS], first.regions[UNITS]);
    assert_ne!(second.regions[1].0, first.regions[1].0);
}

/// The standby of an epoch's serving engine, its own engine of the epoch (its own launch
/// generation), opens a session of its own beside the serving engine's: it names the epoch's KV
/// region and finds its segment table, and keeps slot rows of its own. Each engine's session
/// closes on its own, and the other stays open.
#[test]
fn an_epochs_standby_opens_a_session_of_its_own_beside_its_serving_engines() {
    let server = Running::with_children(None);
    let mut serving = engine_head(&server.socket, 0);
    let (opened, _) = serving.open_pool(layout(0)).unwrap();

    let mut standby = Client::connect(
        &server.socket,
        Role::Head {
            deployment: "model".into(),
            epoch: 0,
            generation: 2,
        },
    )
    .unwrap();
    let mut own = layout(0);
    for (index, name) in [
        (1, "prefill rows of the standby"),
        (2, "decode rows of the standby"),
    ] {
        own.regions[index].name = name.to_owned();
        own.regions[index].space = name.to_owned();
    }
    let (beside, _) = standby.open_pool(own.clone()).unwrap();
    assert_eq!(beside.regions[UNITS], opened.regions[UNITS]);
    assert_eq!(beside.segments[UNITS], opened.segments[UNITS]);
    assert_ne!(beside.regions[1].0, opened.regions[1].0);
    assert_eq!(server_error(standby.open_pool(own)), ErrorKind::Held);

    standby.close_pool().unwrap();
    serving.close_pool().unwrap();
}

/// A worker of `group`'s rank `rank`, of epoch 0.
pub(crate) fn rank(socket: &Path, group: &str, rank: u32, claim: Claim) -> Client {
    worker(socket, scope(group, rank), 1, claim)
}

/// The heads of each layer of the KV region of the handoff cases, the bytes of a head of a
/// token, the tokens of a page and the units a rank holds: a page's plane is two tokens of two
/// heads of four bytes on a rank that holds both heads.
pub(crate) const HEADS: u32 = 2;
pub(crate) const HEAD_BYTES: u32 = 4;
pub(crate) const PAGE: u32 = 2;
const CELLS: u32 = 9;

/// How a rank holding `heads` heads from `offset` lays out its part of the KV region's units:
/// one layer in one column, a page a unit.
pub(crate) fn kv_layout(offset: u32, heads: u32) -> UnitLayout {
    UnitLayout {
        columns: 1,
        classes: vec![PageLayout {
            class: KV,
            block_size: PAGE,
            layers: vec![0],
            kv_heads: heads,
            kv_head_offset: offset,
            total_kv_heads: HEADS,
            head_bytes: HEAD_BYTES,
            quantized: false,
            index_bytes: 0,
        }],
    }
}

/// The bytes of a rank's plane of a page when it holds `heads` heads.
pub(crate) fn plane(heads: u32) -> u64 {
    u64::from(PAGE * heads * HEAD_BYTES)
}

/// Where a rank holding `heads` heads keeps every unit's fields in the tensor `key` at
/// `location`: the keys' planes, the values' planes, then the two planes' written flags, nine
/// units each.
pub(crate) fn kv_runs(key: Key, heads: u32, location: Location) -> Vec<UnitRun> {
    let plane = plane(heads);
    let fields = [
        (planes::VALUES[0], 0, plane),
        (planes::VALUES[1], u64::from(CELLS) * plane, plane),
        (planes::FLAGS[0], 2 * u64::from(CELLS) * plane, 1),
        (
            planes::FLAGS[1],
            2 * u64::from(CELLS) * plane + u64::from(CELLS),
            1,
        ),
    ];
    fields
        .into_iter()
        .map(|(name, offset, bytes)| UnitRun {
            field: planes::field(name, 0),
            key,
            location,
            first: 0,
            count: CELLS,
            offset,
            bytes,
        })
        .collect()
}

/// The bytes of a rank's tensor of [`kv_runs`].
pub(crate) fn kv_bytes(heads: u32) -> u64 {
    2 * u64::from(CELLS) * plane(heads) + 2 * u64::from(CELLS)
}

/// The byte of plane `plane` (keys or values) of page `page`, token `token`, head `head`.
pub(crate) fn kv_byte(page: u32, plane: u32, token: u32, head: u32, byte: u32) -> u8 {
    (page * 37 + plane * 19 + token * 7 + head * 5 + byte + 1) as u8
}

/// Writes pages, page `index` into unit `units[index]`, as a rank holding `heads` heads from
/// `offset` holds them in the tensor `file` laid out by [`kv_runs`].
pub(crate) fn write_pages(file: &std::fs::File, units: &[u32], (offset, heads): (u32, u32)) {
    let plane_bytes = plane(heads);
    for (page, unit) in units.iter().enumerate() {
        for kind in 0..2 {
            let mut bytes = Vec::new();
            for token in 0..PAGE {
                for head in offset..offset + heads {
                    for byte in 0..HEAD_BYTES {
                        bytes.push(kv_byte(page as u32, kind, token, head, byte));
                    }
                }
            }
            let at = u64::from(kind * CELLS + unit) * plane_bytes;
            file.write_all_at(&bytes, at).unwrap();
            let flag = 2 * u64::from(CELLS) * plane_bytes + u64::from(kind * CELLS + unit);
            file.write_all_at(&[1], flag).unwrap();
        }
    }
}

/// The ranks of the handoff cases' two replicas, on one server: the prefill group's rank 0,
/// which holds both heads, and the decode group's ranks 0 and 1, one head each; each with the
/// tensor it holds its units in, placed in its group's memory.
pub(crate) fn handoff_ranks(socket: &Path, claim: Claim) -> [(Client, OwnedFd); 3] {
    let placed =
        |group: &str, number: u32, memory: &str, part: u32, (offset, heads): (u32, u32)| {
            let mut rank = rank(socket, group, number, claim);
            let storage = memfd(&vec![0; kv_bytes(heads) as usize]);
            let name = 20 + number as u8 + if group == "decode" { 10 } else { 0 };
            let values = tensor(key(name), "kv", 0, 0, Access::Exclusive);
            rank.put(vec![values], &[(host(kv_bytes(heads)), storage.as_fd())])
                .unwrap();
            let layout = Some(kv_layout(offset, heads));
            rank.place(
                "units".into(),
                memory.into(),
                part,
                kv_runs(key(name), heads, Location::Host { numa_node: None }),
                layout,
            )
            .unwrap();
            (rank, OwnedFd::from(storage))
        };
    [
        placed("prefill", 0, "gpu0", 0, (0, HEADS)),
        placed("decode", 1, "gpu1", 0, (0, 1)),
        placed("decode", 2, "gpu1", 1, (1, 1)),
    ]
}

/// Lends owner 1 of the prefill scheduler, whose rank wrote two pages into its units (`write`,
/// given the units), and attaches owner 2 of the decode scheduler. Returns the lender.
pub(crate) fn lend_two_pages(
    schedulers: &mut [Scheduler],
    opened: &Opened,
    write: impl FnOnce(&[u32]),
) -> Lender {
    let lender = lend_pages(&mut schedulers[0], opened, write);
    attach_decode(&mut schedulers[1]);
    lender
}

/// Attaches owner `owner` to `group` through `scheduler`.
fn attach(scheduler: &mut Scheduler, owner: u64, group: GroupId) {
    let attach = Request::Attach {
        owner: OwnerId(owner),
        group,
    };
    assert!(matches!(scheduler.ask(attach), Completion::Attached { .. }));
}

/// The decode scheduler attaches owner 2, which imports a lend.
pub(crate) fn attach_decode(decode: &mut Scheduler) {
    attach(decode, 2, DECODE);
}

/// The prefill scheduler's half of [`lend_two_pages`]: owner 1 allocates two blocks, `write`
/// fills their units, and the owner lends them.
pub(crate) fn lend_pages(
    prefill: &mut Scheduler,
    opened: &Opened,
    write: impl FnOnce(&[u32]),
) -> Lender {
    attach(prefill, 1, PREFILL);
    let allocate = Request::Allocate {
        owner: OwnerId(1),
        class: KV,
        at: 0,
        count: 2,
        units: 0,
    };
    let Completion::Allocated { runs, .. } = prefill.ask(allocate) else {
        panic!("no units");
    };
    let lent: Vec<u32> = runs.into_iter().map(|(_, first, _)| first).collect();
    write(&lent);
    let lend = Request::Lend {
        owner: OwnerId(1),
        lease: 60_000_000_000,
        after: Vec::new(),
    };
    assert_eq!(prefill.ask(lend), Completion::Done);
    Lender {
        endpoint: None,
        session: opened.session,
        owner: OwnerId(1),
    }
}

/// The units of an import's answer.
pub(crate) fn imported_units(completion: Completion) -> Vec<u32> {
    let Completion::Imported { runs, .. } = completion else {
        panic!("the import answered {completion:?}");
    };
    runs.into_iter().map(|(_, first, _)| first).collect()
}

/// Whether the decode ranks' tensors hold the two pages in `units`, each rank its own head,
/// with the planes' flags set.
pub(crate) fn assert_imported(decode: &[&OwnedFd], units: &[u32]) {
    for (head, storage) in decode.iter().enumerate() {
        let expected = memfd(&vec![0; kv_bytes(1) as usize]);
        write_pages(&expected, units, (head as u32, 1));
        let file = std::fs::File::from(storage.try_clone().unwrap());
        let bytes = kv_bytes(1) as usize;
        assert_eq!(
            read(&file, 0, bytes),
            read(&expected, 0, bytes),
            "decode rank {head} holds its head of both pages"
        );
    }
}

#[test]
fn an_import_reshards_a_lent_owner_s_pages_into_its_owner_s_ranks_through_the_transfer_server() {
    let server = serving(Gpus::Hidden);
    let claim = head(&server.socket)
        .claim(
            Claimant::Head,
            0,
            vec![scope("prefill", 0), scope("decode", 1), scope("decode", 2)],
            Vec::new(),
        )
        .unwrap();
    let [(_prefill, lent), (_first, first), (_second, second)] =
        handoff_ranks(&server.socket, claim);

    // Two replicas of one engine: the prefill scheduler's owner lends what its rank wrote, and
    // the decode scheduler's owner imports it, each of its two ranks taking its head.
    let mut engine = engine_head(&server.socket, 0);
    let (opened, descriptors) = engine.open_pool(layout(0)).unwrap();
    let mut schedulers = Scheduler::all(&opened, descriptors);
    let lender = lend_two_pages(&mut schedulers, &opened, |units| {
        write_pages(
            &std::fs::File::from(lent.try_clone().unwrap()),
            units,
            (0, HEADS),
        );
    });
    let import = Request::Import {
        owner: OwnerId(2),
        tables: vec![(KV, 0, 2)],
        lender: lender.clone(),
        push: false,
        transport: scheduling::kv::Transport::Nixl,
        trace: 0,
    };
    let units = imported_units(schedulers[1].ask(import));
    assert_imported(&[&first, &second], &units);

    // The import ended the lend: another of it fails at once.
    let again = Request::Import {
        owner: OwnerId(2),
        tables: vec![(KV, 2, 2)],
        lender,
        push: true,
        transport: scheduling::kv::Transport::Nixl,
        trace: 0,
    };
    assert_eq!(schedulers[1].ask(again), Completion::Failed);
}

#[test]
fn a_region_s_storage_is_mapped_writable_by_another_epoch_without_taking_its_writer() {
    let server = Running::with_children(None);
    let mut daemon = head(&server.socket);
    let claim = daemon
        .claim(Claimant::Head, 0, vec![scope("prefill", 0)], Vec::new())
        .unwrap();
    daemon.publish(vec![0]).unwrap();
    let mut serving = rank(&server.socket, "prefill", 0, claim);
    let storage = memfd(&[1; 16]);
    let values_tensor = tensor(key(10), "values", 0, 0, Access::Exclusive);
    serving
        .put(vec![values_tensor], &[(host(16), storage.as_fd())])
        .unwrap();

    // A rank of the next epoch, of the same scope and a later launch, maps
    // it: no writer passes on, so the serving rank is not shot down.
    daemon
        .claim(Claimant::Head, 1, vec![scope("prefill", 0)], Vec::new())
        .unwrap();
    let mut target = Client::connect(
        &server.socket,
        Role::Worker {
            scope: scope("prefill", 0),
            generation: 2,
            claim,
            epoch: 1,
        },
    )
    .unwrap();
    let (got, descriptors) = target.get(vec![key(10)], Mode::Region, Vec::new()).unwrap();
    assert!(got.missing.is_empty());
    assert_eq!(descriptors.len(), 1);
    let listing = target.list().unwrap();
    let writer = listing.tensors.iter().find(|tensor| tensor.key == key(10));
    assert_eq!(
        writer.and_then(|tensor| tensor.writer),
        Some(std::process::id() as i32)
    );
    assert!(serving.list().is_ok(), "the serving rank is connected");
}

/// An engine of epoch 0 with its schedulers, whose two prefill ranks took
/// their parts of the host region, and whose prefill scheduler cached key
/// 7 on the device and on the host, in unit 0 of the host region; with the
/// parts' memfds, by rank.
pub(crate) fn host_copy(server: &Running) -> (Client, Vec<Scheduler>, Opened, Vec<OwnedFd>) {
    let claim = head(&server.socket)
        .claim(
            Claimant::Head,
            0,
            vec![scope("prefill", 0), scope("prefill", 1)],
            Vec::new(),
        )
        .unwrap();
    let mut engine = engine_head(&server.socket, 0);
    let (opened, descriptors) = engine.open_pool(layout(0)).unwrap();
    let parts = host_parts(&server.socket, claim);

    let mut schedulers = Scheduler::all(&opened, descriptors);
    store_on_host(&mut schedulers[0]);
    (engine, schedulers, opened, parts)
}

/// The prefill scheduler caches key 7 on the device and stores it on the
/// host, in unit 0 of the host region, the test running the store's copy.
pub(crate) fn store_on_host(prefill: &mut Scheduler) {
    let attach = Request::Attach {
        owner: OwnerId(1),
        group: PREFILL,
    };
    prefill.ask(attach);
    let allocate = Request::Allocate {
        owner: OwnerId(1),
        class: KV,
        at: 0,
        count: 1,
        units: 0,
    };
    prefill.ask(allocate);
    let commit = Request::Commit {
        owner: OwnerId(1),
        class: KV,
        at: 0,
        key: key(7),
        tokens: 64,
        after: Vec::new(),
    };
    assert_eq!(prefill.ask(commit), Completion::Committed);
    let store = prefill.submit(Request::Store {
        group: PREFILL,
        keys: vec![key(7)],
        tier: Tier::Host,
    });
    assert!(matches!(prefill.next(), (id, Completion::Copy { .. }) if id == store));
    let finish = Request::Finish {
        copy: store,
        ok: true,
        after: Vec::new(),
    };
    let stored = Completion::Stored { keys: Vec::new() };
    assert_eq!(prefill.ask(finish), stored);
}

/// The memfds of the host region's parts, of 16 bytes a unit, that the two
/// prefill ranks of the deployment take.
pub(crate) fn host_parts(socket: &Path, claim: Claim) -> Vec<OwnedFd> {
    (0..2)
        .map(|index| {
            let mut part = rank(socket, "prefill", index, claim);
            let (units, descriptor) = part.host_part("host".into(), index, 16, None).unwrap();
            assert_eq!(units, 4);
            descriptor
        })
        .collect()
}

/// A host region's parts are named by the ranks' index in their worker
/// groups: the prefill and the decode group's ranks of one index hold the
/// same share of every unit and share one part, so a deployment of two
/// groups fills each unit once, whatever the deployment numbers its ranks.
#[test]
fn the_ranks_of_one_index_in_two_groups_share_a_part_of_a_host_region() {
    let server = Running::with_children(None);
    let scopes = vec![
        scope("prefill", 0),
        scope("prefill", 1),
        scope("decode", 2),
        scope("decode", 3),
    ];
    let claim = head(&server.socket)
        .claim(Claimant::Head, 0, scopes, Vec::new())
        .unwrap();
    let mut engine = engine_head(&server.socket, 0);
    engine.open_pool(layout(0)).unwrap();

    // The prefill group's two ranks take the region's two parts of 16 bytes.
    let parts = host_parts(&server.socket, claim);

    // The decode group's ranks, the deployment's ranks 2 and 3, are its
    // ranks 0 and 1: each takes the part of its index, and what one group
    // writes there the other reads.
    for (index, written) in parts.iter().enumerate() {
        let index = u32::try_from(index).unwrap();
        let mut decode = rank(&server.socket, "decode", 2 + index, claim);
        let (units, descriptor) = decode.host_part("host".into(), index, 16, None).unwrap();
        assert_eq!(units, 4);

        let written = std::fs::File::from(written.try_clone().unwrap());
        written.write_all_at(&[index as u8 + 1; 16], 16).unwrap();
        let mut read = [0; 16];
        std::fs::File::from(descriptor)
            .read_exact_at(&mut read, 16)
            .unwrap();
        assert_eq!(read, [index as u8 + 1; 16]);
    }

    // A third part would hold more of a unit than the region's unit size.
    let mut third = rank(&server.socket, "decode", 3, claim);
    assert_eq!(
        server_error(third.host_part("host".into(), 2, 16, None)),
        ErrorKind::Invalid
    );
}

/// A held host region keeps the numbering its parts were created with: an
/// epoch that states it at another size opens over it, and its scheduler
/// resizes it to the units it asks for within that numbering, as a device
/// region is resized.
#[test]
fn an_epoch_that_states_a_held_host_region_at_another_size_resizes_it() {
    let server = Running::with_children(None);
    let mut serving = engine_head(&server.socket, 0);
    serving.open_pool(layout(0)).unwrap();

    // The next epoch's host tier is half the size: two units of the four.
    let mut smaller = layout(1);
    smaller.regions[HOST].units = 2;
    smaller.regions[HOST].limit = 2;
    let mut target = engine_head(&server.socket, 1);
    let (opened, descriptors) = target.open_pool(smaller).unwrap();
    assert_eq!(opened.limits[HOST], 4);
    let host = opened.regions[HOST].0;

    let mut schedulers = Scheduler::all(&opened, descriptors);
    let resize = |units| Request::Resize {
        region: host,
        units,
    };
    let resized = |units| Completion::Resized {
        units,
        evicted: Vec::new(),
    };
    assert_eq!(schedulers[0].ask(resize(2)), resized(2));
    // A grow reaches the numbering and no further: the parts hold no more.
    assert_eq!(schedulers[0].ask(resize(8)), resized(4));
}

/// A host region whose ranks lie on several hosts holds on each host only
/// that host's parts: a session's `Open` admits nothing of it, and each part
/// is admitted, whole, as a rank of this host takes it.
#[test]
fn a_host_region_is_admitted_part_by_part_as_its_ranks_take_them() {
    for refused in [true, false] {
        // A server that leaves the host's whole storage to other processes
        // admits no new host storage.
        let server = Running::reserving(if refused { u64::MAX } else { 0 });
        let claim = head(&server.socket)
            .claim(Claimant::Head, 0, vec![scope("prefill", 0)], Vec::new())
            .unwrap();
        let mut engine = engine_head(&server.socket, 0);
        engine.open_pool(layout(0)).unwrap();

        let mut part = rank(&server.socket, "prefill", 0, claim);
        let taken = part.host_part("host".into(), 0, 16, None);
        if refused {
            assert_eq!(server_error(taken), ErrorKind::Resource);
        } else {
            let (units, _descriptor) = taken.unwrap();
            assert_eq!(units, 4);
        }
    }
}

#[test]
fn an_engine_s_session_closes_once_its_processes_are_out() {
    let server = Running::with_children(None);
    // An engine process opens the session of epoch 0, attaches an owner, closes its connection
    // once the server acted on its requests, and stays until its standard input ends.
    let open = ::storage::protocol::encode(&layout(0)).unwrap();
    let arguments = [
        "--socket".to_owned(),
        server.socket.display().to_string(),
        "--epoch".to_owned(),
        "0".to_owned(),
        "--open".to_owned(),
        crate::server::hex(&open),
    ];
    let mut child = crate::server::start_client("engine", &arguments, |_| {});

    // The connection closed while the engine's process lives: its owner
    // stays attached.
    let attached = |socket: &Path| {
        let pool = listed(socket);
        let rows = pool
            .blocks
            .iter()
            .find(|count| count.region == "prefill rows 0");
        (pool.groups.len(), rows.map(|count| count.blocks))
    };
    assert_eq!(attached(&server.socket), (2, Some(1)));
    // Once it has exited, the session closes and the owner is detached.
    drop(child.stdin.take());
    assert!(child.wait().unwrap().success());
    let deadline = Instant::now() + crate::PATIENCE;
    while attached(&server.socket) != (0, None) {
        assert!(Instant::now() < deadline, "the session did not close");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn what_an_epoch_on_trial_commits_is_its_own_until_its_head_publishes_it() {
    let server = Running::with_children(None);
    let mut daemon = head(&server.socket);
    daemon
        .claim(Claimant::Head, 0, Vec::new(), Vec::new())
        .unwrap();
    daemon.publish(vec![0]).unwrap();
    daemon
        .claim(Claimant::Head, 1, Vec::new(), Vec::new())
        .unwrap();

    // The serving epoch and the target on trial, of one layout.
    let mut serving = engine_head(&server.socket, 0);
    let (opened, descriptors) = serving.open_pool(layout(0)).unwrap();
    let mut serving_schedulers = Scheduler::all(&opened, descriptors);
    let mut target = engine_head(&server.socket, 1);
    let mut trial = layout(1);
    trial.trial = true;
    let (opened, descriptors) = target.open_pool(trial).unwrap();
    let mut target_schedulers = Scheduler::all(&opened, descriptors);

    let attach = Request::Attach {
        owner: OwnerId(1),
        group: PREFILL,
    };
    let allocate = Request::Allocate {
        owner: OwnerId(1),
        class: KV,
        at: 0,
        count: 1,
        units: 0,
    };
    let commit = Request::Commit {
        owner: OwnerId(1),
        class: KV,
        at: 0,
        key: key(7),
        tokens: 64,
        after: Vec::new(),
    };
    let on_trial = &mut target_schedulers[0];
    on_trial.ask(attach.clone());
    on_trial.ask(allocate);
    assert_eq!(on_trial.ask(commit), Completion::Committed);

    // R8: the serving epoch does not see it until the update is committed.
    let pin = Request::Pin {
        owner: OwnerId(1),
        class: KV,
        at: 0,
        keys: vec![key(7)],
    };
    let prefill = &mut serving_schedulers[0];
    prefill.ask(attach);
    let none = Completion::Pinned {
        blocks: Vec::new(),
        host: 0,
    };
    assert_eq!(prefill.ask(pin.clone()), none);
    daemon.publish(vec![1]).unwrap();
    assert!(matches!(prefill.ask(pin), Completion::Pinned { blocks, .. } if blocks.len() == 1));
}

/// A segment of the KV region's memory space.
fn segment(epoch: u64, ordinal: u32) -> Segment {
    Segment { epoch, ordinal }
}

/// The layout of an engine epoch whose ranks created the first segment of
/// the KV region's memory, units 0 to 4, of which it asks for the three
/// past the reserved unit.
fn segmented(epoch: u64) -> Open {
    let mut open = layout(epoch);
    open.regions[UNITS].units = 3;
    open.regions[UNITS].segments = vec![(segment(epoch, 0), 0, 4)];
    open
}

#[test]
fn a_region_grows_into_the_segments_its_ranks_created_and_releases_them_after_a_shrink() {
    let server = Running::with_children(None);
    let mut engine = engine_head(&server.socket, 0);
    let (opened, descriptors) = engine.open_pool(segmented(0)).unwrap();
    assert_eq!(opened.regions[UNITS].1, 3);
    assert_eq!(opened.segments[UNITS], [(segment(0, 0), 0, 4)]);
    assert!(opened.segments[HOST].is_empty());
    let region = opened.regions[UNITS].0;
    let mut schedulers = Scheduler::all(&opened, descriptors);
    let prefill = &mut schedulers[0];
    let resize = |units| Request::Resize { region, units };
    let resized = |units| Completion::Resized {
        units,
        evicted: Vec::new(),
    };

    // The region grows only into units whose segments exist: a grant of
    // the next segment lets it.
    assert_eq!(prefill.ask(resize(7)), resized(3));
    engine.grant("units".into(), segment(0, 1), 4, 4).unwrap();
    assert_eq!(prefill.ask(resize(7)), resized(7));

    // A segment is granted once, and none shares a unit with another.
    let again = engine.grant("units".into(), segment(0, 1), 4, 4);
    assert_eq!(server_error(again), ErrorKind::Invalid);
    let overlapping = engine.grant("units".into(), segment(0, 2), 6, 3);
    assert_eq!(server_error(overlapping), ErrorKind::Invalid);

    // Its units are in the region: it is not withdrawn until a shrink has
    // taken them out; then the region grows no more into them.
    let withdrawn = engine.withdraw("units".into(), segment(0, 1));
    assert_eq!(server_error(withdrawn), ErrorKind::Invalid);
    assert_eq!(prefill.ask(resize(3)), resized(3));
    engine.withdraw("units".into(), segment(0, 1)).unwrap();
    assert_eq!(prefill.ask(resize(7)), resized(3));

    // The next epoch of the layout maps the segments the serving one created.
    let mut next = engine_head(&server.socket, 1);
    let (second, _) = next.open_pool(layout(1)).unwrap();
    assert_eq!(second.regions[UNITS].0, region);
    assert_eq!(second.segments[UNITS], [(segment(0, 0), 0, 4)]);

    // Sleep: the region shrunk to nothing, its last segment withdrawn.
    assert_eq!(prefill.ask(resize(0)), resized(0));
    engine.withdraw("units".into(), segment(0, 0)).unwrap();
    let (asleep, _) = engine_head(&server.socket, 2).open_pool(layout(2)).unwrap();
    assert!(asleep.segments[UNITS].is_empty());

    // Wake: the ranks of epoch 1 create the segments again, its head
    // grants them, and the region grows.
    next.grant("units".into(), segment(1, 0), 0, 4).unwrap();
    next.grant("units".into(), segment(1, 1), 4, 4).unwrap();
    assert_eq!(prefill.ask(resize(7)), resized(7));
    let (awake, _) = engine_head(&server.socket, 3).open_pool(layout(3)).unwrap();
    assert_eq!(
        awake.segments[UNITS],
        [(segment(1, 0), 0, 4), (segment(1, 1), 4, 4)]
    );
}
