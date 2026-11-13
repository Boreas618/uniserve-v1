//! Sessions of engine epochs with the buffer pool: an engine's main process
//! opens its epoch's session on the server's socket, and its schedulers'
//! requests travel on their queue pairs in shared memory, which the buffer
//! pool's thread serves.
//!
//! The test process plays the engine: its main process (a head of the
//! epoch), its schedulers (the engine's ends of the queue pairs, whose
//! `completed` registers they write as the workers would) and its ranks
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
        version: VERSION,
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
    /// worker does.
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
    serve_a_session(&Running::with_transfer(|_| None));
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

/// The operator's `list` prints each deployment's buffer pool as `specs/buffer-pool.md` §8 states
/// it, a line each: the inconsistencies the checks found, the blocks of each region and class in
/// each observed state, each worker group's `completed` word with the stamps of what waits on it,
/// and each request whose copies are in flight.
#[test]
fn the_operator_s_list_prints_the_buffer_pool_s_blocks_groups_copies_and_check_failures() {
    let server = Running::configured(|server| server.watch_pools(None, true));
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

    let mut printed = Vec::new();
    storage_server::list(&server.socket, &mut printed).unwrap();
    let printed = String::from_utf8(printed).unwrap();
    let mut pool: Vec<&str> = printed
        .lines()
        .filter(|line| line.starts_with("buffer pool of model: "))
        .collect();
    pool.sort_unstable();
    let copies = format!(
        "buffer pool of model: 1 copies in flight of request {} of scheduler 0 of epoch 0",
        store.0
    );
    let mut expected = vec![
        "buffer pool of model: 0 check failures",
        "buffer pool of model: 1 shared blocks of class 0 in region units",
        "buffer pool of model: 1 private blocks of class 0 in region units",
        "buffer pool of model: 1 retired blocks of class 0 in region units",
        "buffer pool of model: 1 private blocks of class 0 in region host",
        "buffer pool of model: 1 private slot rows in region prefill rows 0",
        "buffer pool of model: group 0 of epoch 0 completed 0, waiting commit at 9, release at 9",
        "buffer pool of model: group 1 of epoch 0 completed 0, nothing waits",
        &copies,
    ];
    expected.sort_unstable();
    assert_eq!(pool, expected, "{printed}");
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
    let server = Running::with_transfer(|_| None);
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
        .claim(0, vec![scope("prefill", 0), scope("decode", 1)], Vec::new())
        .unwrap();

    // Each group's rank holds the KV region's values in storage of its own,
    // and states where the units lie in its group's memory. The deployment
    // numbers the decode group's rank 1, after the prefill group's; in its
    // group it is the first, which holds the share of a unit the prefill
    // group's first rank holds: the two are matched by that index, their
    // part.
    let planes = [memfd(&[0; 144]), memfd(&[0; 144])];
    let mut ranks = [
        rank(&server.socket, "prefill", 0, claim),
        rank(&server.socket, "decode", 1, claim),
    ];
    for ((rank, planes), (name, memory)) in ranks
        .iter_mut()
        .zip(&planes)
        .zip([(10, "gpu0"), (11, "gpu1")])
    {
        let values_tensor = tensor(key(name), "values", 0, 0, Access::Exclusive);
        rank.put(vec![values_tensor], &[(host(144), planes.as_fd())])
            .unwrap();
        rank.place("units".into(), memory.into(), 0, values(key(name)))
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
    store_on_host(&mut schedulers[0]);
    (engine, schedulers, opened, parts)
}

/// An engine of epoch 0 whose prefill group has a rank on each of two
/// hosts, with its schedulers. Its head opens the session on `here`, naming
/// `peers`, where rank 0 takes part 0 of the host region; rank 1 takes part
/// 1 on `there`, at whose server the engine's peer opened the session's
/// host regions alone, and the head states where that rank's units lie.
/// Returns the head, the peer's session, the schedulers, the answer, and
/// the memfds of the parts here and there.
pub(crate) fn split_host_region(
    here: &Running,
    there: &Running,
    peers: Vec<String>,
) -> (Client, Client, Vec<Scheduler>, Opened, [OwnedFd; 2]) {
    let mut open = layout(0);
    open.peers = peers;
    let claim = head(&here.socket)
        .claim(0, vec![scope("prefill", 0)], Vec::new())
        .unwrap();
    let mut engine = engine_head(&here.socket, 0);
    let (opened, descriptors) = engine.open_pool(open.clone()).unwrap();
    let mut first = rank(&here.socket, "prefill", 0, claim);
    let (_, here_part) = first.host_part("host".into(), 0, 16, None).unwrap();

    // The peer's session states the layout's host regions alone, with no
    // group and no scheduler.
    let mut hosts = open;
    hosts.regions.retain(|region| region.tier == Tier::Host);
    hosts.classes.clear();
    hosts.groups.clear();
    hosts.schedulers.clear();
    hosts.peers.clear();
    let there_claim = head(&there.socket)
        .claim(0, vec![scope("prefill", 1)], Vec::new())
        .unwrap();
    let mut peer = engine_head(&there.socket, 0);
    peer.open_pool(hosts).unwrap();
    let mut second = rank(&there.socket, "prefill", 1, there_claim);
    let (_, there_part) = second.host_part("host".into(), 1, 16, None).unwrap();
    engine
        .place_remote(
            "units".into(),
            "gpu0".into(),
            1,
            there.tcp.clone(),
            values(key(11)),
        )
        .unwrap();

    let schedulers = Scheduler::all(&opened, descriptors);
    (engine, peer, schedulers, opened, [here_part, there_part])
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
    let server = Running::with_transfer(|_| None);
    let scopes = vec![
        scope("prefill", 0),
        scope("prefill", 1),
        scope("decode", 2),
        scope("decode", 3),
    ];
    let claim = head(&server.socket).claim(0, scopes, Vec::new()).unwrap();
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
    let server = Running::with_transfer(|_| None);
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
            .claim(0, vec![scope("prefill", 0)], Vec::new())
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
        group: PREFILL,
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
        group: PREFILL,
        class: KV,
        units: 0,
        keys: vec![key(10)],
        source: Source::Peers,
        deadline: u64::MAX / 2,
    };
    assert_eq!(prefill.ask(peers), Completion::Prefetched { found: 0 });
}

/// Answers the connector's next task, an upload when `upload` and a
/// download otherwise, which must move `pages`, with `results`.
fn run_pages(connector: &mut Client, upload: bool, pages: &[(Key, u64)], results: Vec<bool>) {
    let (task, moved) = match connector_event(connector) {
        Event::Upload { task, pages, .. } if upload => (task, pages),
        Event::Download { task, pages, .. } if !upload => (task, pages),
        other => panic!("the connector got {other:?}"),
    };
    assert_eq!(moved, pages);
    let finished = ::storage::protocol::Request::Finished { task, results };
    assert!(connector.call(&finished, &[]).is_ok());
}

/// A host region whose parts lie on two hosts: a put and a prefetch of the
/// scheduler's host move each part's pages through the connector of the
/// host it lies on, the other host's asked by this one's server, and a key
/// counts only once every part of it moved.
#[test]
fn a_put_and_a_prefetch_move_each_part_through_the_connector_of_its_host() {
    let here = Running::with_transfer(|_| None);
    let there = Running::with_transfer(|_| None);
    // The test plays each server's connector, which is given its host's
    // part.
    let mut connectors =
        [&here, &there].map(|server| Client::connect(&server.socket, Role::Connector).unwrap());
    let (_engine, _peer, mut schedulers, _opened, _parts) =
        split_host_region(&here, &there, Vec::new());
    store_on_host(&mut schedulers[0]);
    for connector in &mut connectors {
        assert!(matches!(
            connector_event(connector),
            Event::Pool { slots: 4, .. }
        ));
    }
    let prefill = &mut schedulers[0];

    // A put: the block's page of each part, written by its host's
    // connector.
    let put = prefill.submit(Request::Store {
        group: PREFILL,
        keys: vec![key(7)],
        tier: Tier::Remote,
    });
    for connector in &mut connectors {
        run_pages(connector, true, &[(key(7), 0)], vec![true]);
    }
    let stored = Completion::Stored { keys: vec![key(7)] };
    assert_eq!(prefill.next(), (put, stored));

    // A prefetch of two keys: the second's page fails on the other host's
    // part, so the first alone is found.
    let prefetch = prefill.submit(Request::Prefetch {
        group: PREFILL,
        class: KV,
        units: 0,
        keys: vec![key(8), key(9)],
        source: Source::Remote,
        deadline: u64::MAX / 2,
    });
    let pages = [(key(8), 1), (key(9), 2)];
    run_pages(&mut connectors[0], false, &pages, vec![true, true]);
    run_pages(&mut connectors[1], false, &pages, vec![true, false]);
    assert_eq!(
        prefill.next(),
        (prefetch, Completion::Prefetched { found: 1 })
    );

    // A put the other host's connector fails stores nothing.
    let put = prefill.submit(Request::Store {
        group: PREFILL,
        keys: vec![key(8)],
        tier: Tier::Remote,
    });
    run_pages(&mut connectors[0], true, &[(key(8), 1)], vec![true]);
    run_pages(&mut connectors[1], true, &[(key(8), 1)], vec![false]);
    let stored = Completion::Stored { keys: Vec::new() };
    assert_eq!(prefill.next(), (put, stored));
}

/// Two groups whose ranks of one index lie on two hosts, each copying with
/// the parts of its own host: a prefetch fills the host parts of the group
/// it is for and a put reads those that hold the copy, each through the
/// connector of their host, and a peer reads a copy only from parts that
/// hold it.
#[test]
fn host_parts_of_one_index_on_two_hosts_are_kept_apart() {
    let here = Running::with_transfer(|_| None);
    let there = Running::with_transfer(|_| None);
    // The test plays each server's connector.
    let mut connectors =
        [&here, &there].map(|server| Client::connect(&server.socket, Role::Connector).unwrap());

    // The prefill group's ranks copy with the parts of host a, the
    // session's, and the decode group's with those of host b, where the
    // engine's peer opened the host regions alone. Each rank takes its part
    // at its host's server and states its host there; the head states where
    // the decode ranks' parts lie.
    let mut open = layout(0);
    open.groups[0].parts = vec![vec![("a".into(), 0), ("a".into(), 1)]];
    open.groups[1].parts = vec![vec![("b".into(), 0), ("b".into(), 1)]];
    let scopes = |group| vec![scope(group, 0), scope(group, 1)];
    let claim = head(&here.socket)
        .claim(0, scopes("prefill"), Vec::new())
        .unwrap();
    let mut engine = engine_head(&here.socket, 0);
    let (opened, descriptors) = engine.open_pool(open.clone()).unwrap();
    let mut hosts = open;
    hosts.regions.retain(|region| region.tier == Tier::Host);
    hosts.classes.clear();
    hosts.groups.clear();
    hosts.schedulers.clear();
    let there_claim = head(&there.socket)
        .claim(0, scopes("decode"), Vec::new())
        .unwrap();
    let mut peer = engine_head(&there.socket, 0);
    peer.open_pool(hosts).unwrap();
    let mut ranks = Vec::new();
    for index in 0..2 {
        for (server, group, claimed, host) in [
            (&here, "prefill", claim, "a"),
            (&there, "decode", there_claim, "b"),
        ] {
            let mut part = rank(&server.socket, group, index, claimed);
            part.host_part("host".into(), index, 16, None).unwrap();
            part.place("host".into(), host.into(), index, Vec::new())
                .unwrap();
            ranks.push(part);
        }
        engine
            .place_remote(
                "host".into(),
                "b".into(),
                index,
                there.tcp.clone(),
                Vec::new(),
            )
            .unwrap();
    }
    for connector in &mut connectors {
        for _ in 0..2 {
            assert!(matches!(
                connector_event(connector),
                Event::Pool { slots: 4, .. }
            ));
        }
    }
    let mut schedulers = Scheduler::all(&opened, descriptors);

    // Each group's prefetch fills the parts of its own host, through that
    // host's connector, part by part.
    for (scheduler, group, byte, unit) in [(1, DECODE, 8, 0), (0, PREFILL, 9, 1)] {
        let prefetch = schedulers[scheduler].submit(Request::Prefetch {
            group,
            class: KV,
            units: 0,
            keys: vec![key(byte)],
            source: Source::Remote,
            deadline: u64::MAX / 2,
        });
        for _ in 0..2 {
            run_pages(
                &mut connectors[scheduler],
                false,
                &[(key(byte), unit)],
                vec![true],
            );
        }
        let found = Completion::Prefetched { found: 1 };
        assert_eq!(schedulers[scheduler].next(), (prefetch, found));
    }
    let listed = Completion::Listing {
        keys: vec![
            (key(8), KV, 64, 0b010, 0, 0b10),
            (key(9), KV, 64, 0b010, 0, 0b01),
        ],
    };
    assert_eq!(schedulers[0].ask(Request::List), listed);

    // A put reads each copy from the parts that hold it, on their host.
    for (connector, byte, unit) in [(1, 8, 0), (0, 9, 1)] {
        let put = schedulers[0].submit(Request::Store {
            group: PREFILL,
            keys: vec![key(byte)],
            tier: Tier::Remote,
        });
        for _ in 0..2 {
            run_pages(
                &mut connectors[connector],
                true,
                &[(key(byte), unit)],
                vec![true],
            );
        }
        let stored = Completion::Stored {
            keys: vec![key(byte)],
        };
        assert_eq!(schedulers[0].next(), (put, stored));
    }

    // A peer reads a region's copies from the parts its first copy lies in;
    // the second, which other parts hold, is left out.
    let mut reader = Client::connect_peer(&here.tcp, Duration::from_secs(30)).unwrap();
    let (copies, mut parts) = reader
        .peer_hold("model".into(), vec![key(8), key(9)])
        .unwrap();
    let held: Vec<Key> = copies.iter().map(|copy| copy.key).collect();
    assert_eq!(held, [key(8)]);
    parts.sort_by_key(|part| part.rank);
    let part = |rank| RegionPart {
        region: "host".into(),
        rank,
        bytes: 16,
        endpoint: Some(there.tcp.clone()),
    };
    assert_eq!(parts, [part(0), part(1)]);
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
        endpoint: None,
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
