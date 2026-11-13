//! Sessions of engine epochs with the buffer pool: an engine's main process
//! opens its epoch's session on the server's socket, and its schedulers'
//! requests travel on their queue pairs in shared memory, which the buffer
//! pool's thread serves.
//!
//! The test process plays the engine: its main process (a head of the
//! epoch), its schedulers (the engine's ends of the queue pairs, whose
//! `completed` registers they write as the executors would) and its ranks
//! (workers). No GPU is needed: the regions' memory is host storage.

use std::io::BufRead as _;
use std::os::fd::{AsFd as _, OwnedFd};
use std::os::unix::fs::FileExt as _;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use ::storage::protocol::{BlockState, ErrorKind, Event, HostCopy, RegionPart, UnitRun};
use ::storage::runtime::{
    Cause, Class, Completion, Group, GroupId, Open, Opened, OwnerId, Region, Registers, Request,
    RequestId, Source, Tier, VERSION,
};
use ::storage::{Access, Claim, Client, Key, Location, Mode, QueuePair, Role, Segment};

use crate::server::{
    FORKING, Running, engine_head, head, host, key, memfd, scope, server_error, tensor, worker,
};
use crate::transfer::{Gpus, pattern, read, serving};

pub(crate) const PREFILL: GroupId = GroupId(0);
pub(crate) const DECODE: GroupId = GroupId(1);

/// The class of KV blocks: one unit each, host copies of one unit.
pub(crate) const KV: u8 = 0;

/// The indices of the layout's regions.
pub(crate) const UNITS: usize = 0;
pub(crate) const HOST: usize = 3;

/// A region of a layout.
fn region(name: &str, tier: Tier, unit_bytes: u64, reserved: u32, units: u32) -> Region {
    Region {
        name: name.into(),
        tier,
        space: name.into(),
        unit_bytes,
        reserved,
        units,
        limit: reserved + units,
        segments: Vec::new(),
    }
}

/// The layout of an engine epoch: a prefill group and a decode group on
/// memories of their own, each the only group of its scheduler, over one
/// KV region of eight units, their slot rows (two each, the epoch's own)
/// and a host region of four units of 32 bytes, two parts of 16.
pub(crate) fn layout(epoch: u64) -> Open {
    Open {
        version: VERSION,
        epoch,
        trial: false,
        regions: vec![
            region("units", Tier::Device, 64, 1, 8),
            region(&format!("prefill rows {epoch}"), Tier::Device, 8, 1, 2),
            region(&format!("decode rows {epoch}"), Tier::Device, 8, 1, 2),
            region("host", Tier::Host, 32, 0, 4),
        ],
        classes: vec![Class {
            units: 1,
            host: Some((HOST as u8, 1)),
            state_in_slot: false,
        }],
        trajectory: KV,
        groups: vec![
            Group {
                id: PREFILL,
                memory: "gpu0".into(),
                slots: 1,
                regions: vec![vec![UNITS as u8]],
            },
            Group {
                id: DECODE,
                memory: "gpu1".into(),
                slots: 2,
                regions: vec![vec![UNITS as u8]],
            },
        ],
        schedulers: vec![vec![PREFILL], vec![DECODE]],
        peers: Vec::new(),
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
    pub(crate) fn all(opened: &Opened, descriptors: Vec<OwnedFd>) -> Vec<Self> {
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

    /// The next completion, within ten seconds.
    pub(crate) fn next(&mut self) -> (RequestId, Completion) {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut record = Vec::new();
        loop {
            if self.pair.next_completion(&mut record).unwrap() {
                return Completion::decode(&record).unwrap();
            }
            assert!(Instant::now() < deadline, "no completion arrived");
            std::thread::sleep(Duration::from_micros(100));
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
    /// executor does.
    pub(crate) fn complete_batches(&self, index: usize, completed: u64) {
        let register = self.registers.completed[index] as usize;
        self.pair
            .register(register)
            .store(completed, Ordering::Release);
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
        let deadline = Instant::now() + Duration::from_secs(10);
        while (read(0), read(1), read(2)) != words {
            assert!(
                Instant::now() < deadline,
                "the capacity words read {:?}, not {words:?}",
                (read(0), read(1), read(2))
            );
            std::thread::sleep(Duration::from_micros(100));
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
    let server = Running::with_transfer(|_| None);
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

    // The commit takes effect when the executor's `completed` reaches its stamp.
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
        runs: Vec::new(),
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

#[test]
fn a_session_is_refused_what_the_server_does_not_serve() {
    let server = Running::with_transfer(|_| None);
    let mut engine = engine_head(&server.socket, 0);

    // Another interface version, and another epoch than the head's.
    let mut other = layout(0);
    other.version = VERSION + 1;
    assert_eq!(server_error(engine.open_pool(other)), ErrorKind::Invalid);
    assert_eq!(
        server_error(engine.open_pool(layout(1))),
        ErrorKind::Invalid
    );

    // One session per epoch.
    let (first, _) = engine.open_pool(layout(0)).unwrap();
    assert_eq!(server_error(engine.open_pool(layout(0))), ErrorKind::Held);

    // A region stated again under its name is that region, of its layout.
    let mut next = engine_head(&server.socket, 1);
    let mut restated = layout(1);
    restated.regions[UNITS].unit_bytes = 128;
    assert_eq!(server_error(next.open_pool(restated)), ErrorKind::Invalid);

    // At most eight device memories over the deployment's sessions.
    let mut wide = layout(1);
    for index in 2..9 {
        wide.groups.push(Group {
            id: GroupId(index),
            memory: format!("gpu{index}"),
            slots: 1,
            regions: vec![vec![UNITS as u8]],
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

/// A worker of `group`'s rank `rank`, of epoch 0.
pub(crate) fn rank(socket: &Path, group: &str, rank: u32, claim: Claim) -> Client {
    worker(socket, scope(group, rank), 1, claim)
}

/// Where a rank's values of all nine units of the KV region lie: 16 bytes
/// a unit in the host tensor `key`.
pub(crate) fn values(key: Key) -> Vec<UnitRun> {
    vec![UnitRun {
        field: "values".into(),
        key,
        location: Location::Host { numa_node: None },
        first: 0,
        count: 9,
        offset: 0,
        bytes: 16,
    }]
}

#[test]
fn a_transfer_copies_an_owner_s_units_between_its_groups_ranks_through_the_transfer_server() {
    let server = serving(Gpus::Hidden);
    let claim = head(&server.socket)
        .claim(0, vec![scope("prefill", 0), scope("decode", 0)], Vec::new())
        .unwrap();

    // Each group's rank holds the KV region's values in storage of its own,
    // and states where the units lie in its group's memory.
    let planes = [memfd(&[0; 144]), memfd(&[0; 144])];
    let mut ranks = [
        rank(&server.socket, "prefill", 0, claim),
        rank(&server.socket, "decode", 0, claim),
    ];
    for ((rank, planes), (name, memory)) in ranks
        .iter_mut()
        .zip(&planes)
        .zip([(10, "gpu0"), (11, "gpu1")])
    {
        let values_tensor = tensor(key(name), "values", 0, 0, Access::Exclusive);
        rank.put(vec![values_tensor], &[(host(144), planes.as_fd())])
            .unwrap();
        rank.place("units".into(), memory.into(), values(key(name)))
            .unwrap();
    }

    let mut engine = engine_head(&server.socket, 0);
    let (opened, descriptors) = engine.open_pool(layout(0)).unwrap();
    let region = opened.regions[UNITS].0;
    let mut schedulers = Scheduler::all(&opened, descriptors);
    let prefill = &mut schedulers[0];
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
    assert!(matches!(
        prefill.ask(allocate),
        Completion::Allocated { .. }
    ));

    // The prefill group's batches wrote units 1 and 2; the transfer copies
    // them into the decode rank's planes, and the owner moves.
    let written = pattern(32, 5);
    planes[0].write_all_at(&written, 16).unwrap();
    let transfer = prefill.submit(Request::Transfer {
        owner: OwnerId(1),
        group: DECODE,
        after: Vec::new(),
    });
    let runs = vec![(KV, 0, (region, 1, 1), true), (KV, 1, (region, 2, 1), true)];
    assert_eq!(
        prefill.next(),
        (transfer, Completion::Transferred { slot: 1, runs })
    );
    assert_eq!(read(&planes[1], 16, 32), written);
    assert_eq!(
        read(&planes[1], 48, 16),
        [0; 16],
        "only the owner's units move"
    );
}

#[test]
fn a_region_s_storage_is_mapped_writable_by_another_epoch_without_taking_its_writer() {
    let server = Running::with_transfer(|_| None);
    let mut daemon = head(&server.socket);
    let claim = daemon
        .claim(0, vec![scope("prefill", 0)], Vec::new())
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
        .claim(1, vec![scope("prefill", 0)], Vec::new())
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

/// The connector's next event, within ten seconds.
fn connector_event(connector: &mut Client) -> Event {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        connector.pump().unwrap();
        if let Some((event, _)) = connector.next_event() {
            return event;
        }
        assert!(Instant::now() < deadline, "the connector got no event");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// An engine of epoch 0 with its schedulers, whose two prefill ranks took
/// their parts of the host region, and whose prefill scheduler cached key
/// 7 on the device and on the host, in unit 0 of the host region; with the
/// parts' memfds, by rank.
pub(crate) fn host_copy(server: &Running) -> (Client, Vec<Scheduler>, Opened, Vec<OwnedFd>) {
    let claim = head(&server.socket)
        .claim(
            0,
            vec![scope("prefill", 0), scope("prefill", 1)],
            Vec::new(),
        )
        .unwrap();
    let mut engine = engine_head(&server.socket, 0);
    let (opened, descriptors) = engine.open_pool(layout(0)).unwrap();
    let parts = host_parts(&server.socket, claim);

    let mut schedulers = Scheduler::all(&opened, descriptors);
    let prefill = &mut schedulers[0];
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
    (engine, schedulers, opened, parts)
}

/// The memfds of the host region's parts, of 16 bytes a unit, that the two
/// prefill ranks of the deployment take.
pub(crate) fn host_parts(socket: &Path, claim: Claim) -> Vec<OwnedFd> {
    (0..2)
        .map(|index| {
            let mut part = rank(socket, "prefill", index, claim);
            let (units, descriptor) = part.host_part("host".into(), 16, None).unwrap();
            assert_eq!(units, 4);
            descriptor
        })
        .collect()
}

#[test]
fn a_put_and_a_prefetch_move_host_copies_through_the_connector_part_by_part() {
    let server = Running::with_transfer(|_| None);
    // The test plays the server's connector.
    let mut connector = Client::connect(&server.socket, Role::Connector).unwrap();
    let (_engine, mut schedulers, opened, _parts) = host_copy(&server);
    let host = opened.regions[HOST].0;
    let mut parts = Vec::new();
    for _ in 0..2 {
        match connector_event(&mut connector) {
            Event::Pool { pool, shape, slots } => {
                assert_eq!((shape.page_bytes, slots), (16, 4));
                parts.push(pool);
            }
            other => panic!("the connector got {other:?}"),
        }
    }
    let prefill = &mut schedulers[0];

    // A put: one upload of the block's page per part.
    let put = prefill.submit(Request::Store {
        group: PREFILL,
        keys: vec![key(7)],
        tier: Tier::Remote,
    });
    let mut uploaded = Vec::new();
    for _ in 0..2 {
        let Event::Upload { task, pool, pages } = connector_event(&mut connector) else {
            panic!("the connector got no upload");
        };
        assert_eq!(pages, [(key(7), 0)]);
        uploaded.push(pool);
        let finished = connector.call(
            &::storage::protocol::Request::Finished {
                task,
                results: vec![true],
            },
            &[],
        );
        assert!(finished.is_ok());
    }
    uploaded.sort_unstable();
    parts.sort_unstable();
    assert_eq!(uploaded, parts);
    let stored = Completion::Stored { keys: vec![key(7)] };
    assert_eq!(prefill.next(), (put, stored));

    // A prefetch of two keys: the second's page fails on one part, so the
    // first alone is found.
    let prefetch = prefill.submit(Request::Prefetch {
        class: KV,
        units: 0,
        keys: vec![key(8), key(9)],
        source: Source::Remote,
        deadline: u64::MAX / 2,
    });
    for answer in [vec![true, true], vec![true, false]] {
        let Event::Download { task, pages, .. } = connector_event(&mut connector) else {
            panic!("the connector got no download");
        };
        assert_eq!(pages, [(key(8), 1), (key(9), 2)]);
        let finished = ::storage::protocol::Request::Finished {
            task,
            results: answer,
        };
        assert!(connector.call(&finished, &[]).is_ok());
    }
    assert_eq!(
        prefill.next(),
        (prefetch, Completion::Prefetched { found: 1 })
    );
    let pool = listed(&server.socket);
    let host_blocks: u32 = pool
        .blocks
        .iter()
        .filter(|count| count.region == "host" && count.state == BlockState::Shared)
        .map(|count| count.blocks)
        .sum();
    assert_eq!(host_blocks, 2, "keys 7 and 8 have host copies");
    prefill.await_capacity(host, (2, 0, 2));

    // The session names no peer: a prefetch from the peers finds nothing.
    let peers = Request::Prefetch {
        class: KV,
        units: 0,
        keys: vec![key(10)],
        source: Source::Peers,
        deadline: u64::MAX / 2,
    };
    assert_eq!(prefill.ask(peers), Completion::Prefetched { found: 0 });
}

#[test]
fn a_peer_holds_the_host_copies_it_reads_until_it_releases_them() {
    let server = Running::with_transfer(|_| None);
    let (_engine, schedulers, opened, _parts) = host_copy(&server);
    let host = opened.regions[HOST].0;
    let prefill = &schedulers[0];
    prefill.await_capacity(host, (3, 0, 1));

    // Another host's Storage Server holds what it reads; a held copy is no
    // longer evictable.
    let mut peer = Client::connect_peer(&server.tcp, Duration::from_secs(30)).unwrap();
    let (copies, mut parts) = peer
        .peer_hold("model".into(), vec![key(7), key(8)])
        .unwrap();
    let held = HostCopy {
        key: key(7),
        tokens: 64,
        region: "host".into(),
        first: 0,
        count: 1,
    };
    assert_eq!(copies, [held]);
    parts.sort_by_key(|part| part.rank);
    let part = |rank| RegionPart {
        region: "host".into(),
        rank,
        bytes: 16,
    };
    assert_eq!(parts, [part(0), part(1)]);
    prefill.await_capacity(host, (3, 0, 0));

    // Its release ends the holds, which stood: what it read is valid.
    assert!(peer.peer_release("model".into()).unwrap());
    prefill.await_capacity(host, (3, 0, 1));
    assert!(!peer.peer_release("model".into()).unwrap());

    // The holds end with the peer's connection too.
    peer.peer_hold("model".into(), vec![key(7)]).unwrap();
    prefill.await_capacity(host, (3, 0, 0));
    drop(peer);
    prefill.await_capacity(host, (3, 0, 1));
}

/// The engine `an_engine_s_session_closes_once_its_processes_are_out` starts
/// as a child process: it opens the session of epoch 0, attaches an owner,
/// closes its connection and stays until its standard input closes.
#[test]
#[ignore = "a stand-in engine this file's tests start as a child process"]
fn stand_in_engine() {
    let Ok(socket) = std::env::var("STORAGE_TEST_SOCKET") else {
        return;
    };
    let mut engine = engine_head(Path::new(&socket), 0);
    let (opened, descriptors) = engine.open_pool(layout(0)).unwrap();
    let mut schedulers = Scheduler::all(&opened, descriptors);
    let attach = Request::Attach {
        owner: OwnerId(1),
        group: PREFILL,
    };
    assert_eq!(schedulers[0].ask(attach), Completion::Attached { slot: 1 });
    drop(engine);
    println!("left");
    let mut line = String::new();
    let _ = std::io::stdin().read_line(&mut line);
}

#[test]
fn an_engine_s_session_closes_once_its_processes_are_out() {
    let server = Running::with_transfer(|_| None);
    let forking = FORKING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "sessions::stand_in_engine",
            "--ignored",
            "--nocapture",
        ])
        .env("STORAGE_TEST_SOCKET", &server.socket)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    drop(forking);
    let stdout = child.stdout.take().unwrap();
    let mut lines = std::io::BufReader::new(stdout).lines();
    while lines.next().unwrap().unwrap() != "left" {}

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
    for _ in 0..20 {
        assert_eq!(attached(&server.socket), (2, Some(1)));
        std::thread::sleep(Duration::from_millis(10));
    }

    // Once it has exited, the session closes and the owner is detached.
    drop(child.stdin.take());
    assert!(child.wait().unwrap().success());
    let deadline = Instant::now() + Duration::from_secs(10);
    while attached(&server.socket) != (0, None) {
        assert!(Instant::now() < deadline, "the session did not close");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn what_an_epoch_on_trial_commits_is_its_own_until_its_head_publishes_it() {
    let server = Running::with_transfer(|_| None);
    let mut daemon = head(&server.socket);
    daemon.claim(0, Vec::new(), Vec::new()).unwrap();
    daemon.publish(vec![0]).unwrap();
    daemon.claim(1, Vec::new(), Vec::new()).unwrap();

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
        runs: Vec::new(),
        host: 0,
    };
    assert_eq!(prefill.ask(pin.clone()), none);
    daemon.publish(vec![1]).unwrap();
    assert!(matches!(prefill.ask(pin), Completion::Pinned { runs, .. } if runs.len() == 1));
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
    let server = Running::with_transfer(|_| None);
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
