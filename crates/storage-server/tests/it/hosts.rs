//! Buffer pools that reach other hosts: a prefetch that reads the host
//! copies another host's Storage Server holds (a peer), a worker group with
//! a rank on another host, which a transfer reaches through that host's
//! Transfer Server, and a product that crosses to a main process of another
//! host.
//!
//! Two Storage Servers of this host stand for two hosts, each on a thread
//! of the test process with a socket and a TCP endpoint of its own; they
//! reach each other only by their endpoints. Each runs the real Transfer
//! Server, copying on no GPU, which reads another host's storage through
//! NIXL, so these cases need NIXL; run them with `cargo test -p
//! storage-server --test it hosts -- --ignored`.

use std::fs::File;
use std::os::fd::AsFd;
use std::os::unix::fs::FileExt;

use ::storage::protocol::UnitRun;
use ::storage::runtime::{
    Class, Completion, Group, GroupId, Open, OwnerId, Request, Source, Tier, VERSION,
};
use ::storage::{Access, Claimant, Key, Location};

use crate::server::{Running, engine_head, head, host, key, memfd, scope, tensor, transfer_child};
use crate::sessions::{
    DECODE, HOST, KV, PREFILL, Scheduler, UNITS, host_copy, host_parts, layout, rank,
    split_host_region, store_on_host, values,
};
use crate::transfer::{pattern, read, transfer_pid};

/// A server whose Transfer Server copies on no GPU and reads other hosts'
/// regions through NIXL.
fn network_server() -> Running {
    let server = Running::with_children(Some(transfer_child(false)));
    transfer_pid(&server.socket, None);
    server
}

/// The parts of a host region whose storage the test holds, as files.
fn files(parts: Vec<std::os::fd::OwnedFd>) -> Vec<File> {
    parts.into_iter().map(File::from).collect()
}

/// A prefetch of keys 7 and 9 from the peers.
fn from_peers() -> Request {
    Request::Prefetch {
        group: PREFILL,
        class: KV,
        units: 0,
        keys: vec![key(7), key(9)],
        source: Source::Peers,
        deadline: u64::MAX / 2,
    }
}

#[test]
#[ignore = "requires NIXL"]
fn a_prefetch_from_the_peers_reads_the_host_copies_a_peer_held_throughout() {
    // The peer cached key 7 on its host, in unit 0 of each part of its host
    // region; the test holds the parts' storage.
    let peer = crate::server::Running::with_children(None);
    let (_peer_engine, peer_schedulers, peer_opened, peer_parts) = host_copy(&peer);
    let peer_parts = files(peer_parts);
    let written = [pattern(16, 3), pattern(16, 4)];
    for (part, bytes) in peer_parts.iter().zip(&written) {
        part.write_all_at(bytes, 0).unwrap();
    }

    // The reading host names a peer nothing listens on, and the peer.
    let reading = network_server();
    let claim = head(&reading.socket)
        .claim(
            Claimant::Head,
            0,
            vec![scope("prefill", 0), scope("prefill", 1)],
            Vec::new(),
        )
        .unwrap();
    let nobody = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .to_string();
    let mut open = layout(0);
    open.peers = vec![nobody, peer.tcp.clone()];
    let mut engine = engine_head(&reading.socket, 0);
    let (opened, descriptors) = engine.open_pool(open).unwrap();
    let parts = files(host_parts(&reading.socket, claim));
    let mut schedulers = Scheduler::all(&opened, descriptors);
    let prefill = &mut schedulers[0];

    // Key 7 is read into a new host block, unit 0 of each part here; key 9
    // is held by no peer.
    assert_eq!(
        prefill.ask(from_peers()),
        Completion::Prefetched {
            found: 1,
            evicted: Vec::new()
        }
    );
    for (part, bytes) in parts.iter().zip(&written) {
        assert_eq!(read(part, 0, 16), *bytes);
    }
    let listed = Completion::Listing {
        keys: vec![(key(7), KV, 64, 0b010, 0, 0b11)],
    };
    assert_eq!(prefill.ask(Request::List), listed);

    // The peer's holds ended with the read: its copy is evictable again.
    peer_schedulers[0].await_capacity(peer_opened.regions[HOST].0, (3, 0, 1));
}

/// A reading engine whose KV class keeps its own copies in another encoding than the peer's,
/// in a host region of 8 bytes a part, and converts from the peer's encoding: the class names a
/// converted-load source region of the peer's 16 bytes a part as its source. A prefetch from the
/// peers reads the peer's copy into the source region, never into the class's host region, and
/// a load of the key names that copy with the two encodings the worker converts between.
#[test]
#[ignore = "requires NIXL"]
fn a_prefetch_from_the_peers_reads_a_copy_of_the_source_encoding_into_the_source_region() {
    // The peer cached key 7 on its host in the layout's encoding, in unit 0 of each part of its
    // host region.
    let peer = Running::with_children(None);
    let (_peer_engine, _peer_schedulers, _, peer_parts) = host_copy(&peer);
    let peer_parts = files(peer_parts);
    let written = [pattern(16, 3), pattern(16, 4)];
    for (part, bytes) in peer_parts.iter().zip(&written) {
        part.write_all_at(bytes, 0).unwrap();
    }

    // The reading engine's class is written in encoding 1, in a host region of half the bytes,
    // and converts from the peer's encoding 0 through a source region of two units.
    let reading = network_server();
    let claim = head(&reading.socket)
        .claim(
            Claimant::Head,
            0,
            vec![scope("prefill", 0), scope("prefill", 1)],
            Vec::new(),
        )
        .unwrap();
    let mut open = layout(0);
    open.peers = vec![peer.tcp.clone()];
    let source = open.regions.len();
    let mut region = open.regions[HOST].clone();
    (region.unit_bytes, region.part_bytes) = (16, 8);
    open.regions[HOST] = region.clone();
    (region.name, region.space) = ("source".into(), "source".into());
    (
        region.unit_bytes,
        region.part_bytes,
        region.units,
        region.limit,
    ) = (32, 16, 2, 2);
    open.regions.push(region);
    let class = &mut open.classes[usize::from(KV)];
    class.encoding = [1; 32];
    class.source = Some((source as u8, [0; 32]));
    let mut engine = engine_head(&reading.socket, 0);
    let (opened, descriptors) = engine.open_pool(open).unwrap();

    // Each prefill rank takes its part of both host regions.
    let parts: Vec<(File, File)> = (0..2)
        .map(|index| {
            let mut taking = rank(&reading.socket, "prefill", index, claim);
            let (_, own) = taking.host_part("host".into(), index, 8, None).unwrap();
            let (units, staged) = taking.host_part("source".into(), index, 16, None).unwrap();
            assert_eq!(units, 2);
            (File::from(own), File::from(staged))
        })
        .collect();
    let mut schedulers = Scheduler::all(&opened, descriptors);
    let prefill = &mut schedulers[0];

    // Key 7 is read into unit 0 of each part of the source region and nothing into the host
    // region; key 9 is held by no peer.
    assert_eq!(
        prefill.ask(from_peers()),
        Completion::Prefetched {
            found: 1,
            evicted: Vec::new()
        }
    );
    for ((own, staged), bytes) in parts.iter().zip(&written) {
        assert_eq!(read(staged, 0, 16), *bytes);
        assert_eq!(read(own, 0, 8), [0; 8]);
    }

    // A load of key 7 reads the staged copy and names the encodings to convert between.
    prefill.ask(Request::Attach {
        owner: OwnerId(1),
        group: PREFILL,
    });
    let load = Request::Load {
        owner: OwnerId(1),
        class: KV,
        at: 0,
        keys: vec![key(7)],
    };
    let Completion::Copy {
        keys,
        from,
        encodings,
        ..
    } = prefill.ask(load)
    else {
        panic!("the load copies nothing")
    };
    assert_eq!(keys, [key(7)]);
    assert_eq!(from, [(opened.regions[source].0, 0, 1)]);
    assert_eq!(encodings, Some(([0; 32], [1; 32])));
}

/// Both the reading engine and the peer's have a rank on each of two hosts:
/// each part of the read copy is read from the host the peer's part of its
/// index lies on, by the Transfer Server of the host the reader's part lies
/// on.
#[test]
#[ignore = "requires NIXL"]
fn a_prefetch_from_the_peers_reads_each_part_on_the_host_it_lies_on() {
    // The peer's engine cached key 7 on the host, in unit 0 of its part
    // on its scheduler's host and of its part on its other host.
    let peer = Running::with_children(None);
    let peer_other = Running::with_children(None);
    let (_peer_engine, _peer_peer, mut peer_schedulers, _, peer_parts) =
        split_host_region(&peer, &peer_other, Vec::new());
    store_on_host(&mut peer_schedulers[0]);
    let peer_parts = files(peer_parts.into());
    let written = [pattern(16, 3), pattern(16, 4)];
    for (part, bytes) in peer_parts.iter().zip(&written) {
        part.write_all_at(bytes, 0).unwrap();
    }

    // The reading engine names the peer.
    let reading = network_server();
    let reading_other = network_server();
    let (_engine, _peer, mut schedulers, _, parts) =
        split_host_region(&reading, &reading_other, vec![peer.tcp.clone()]);
    let parts = files(parts.into());
    let prefill = &mut schedulers[0];

    // Key 7 is read into unit 0 of each part, the part on the other host
    // by that host's Transfer Server; key 9 is held by no peer.
    assert_eq!(
        prefill.ask(from_peers()),
        Completion::Prefetched {
            found: 1,
            evicted: Vec::new()
        }
    );
    for (part, bytes) in parts.iter().zip(&written) {
        assert_eq!(read(part, 0, 16), *bytes);
    }
    let listed = Completion::Listing {
        keys: vec![(key(7), KV, 64, 0b010, 0, 0b11)],
    };
    assert_eq!(prefill.ask(Request::List), listed);
}

#[test]
#[ignore = "requires NIXL"]
fn a_transfer_reaches_a_rank_of_another_host_through_that_host_s_transfer_server() {
    let here = network_server();
    let there = network_server();

    // The prefill rank is on the scheduler's host, the decode rank on the
    // other one, each holding the KV region's values in storage of its
    // own, retained by its host's Storage Server.
    let prefill_planes = memfd(&[0; 144]);
    let claim = head(&here.socket)
        .claim(Claimant::Head, 0, vec![scope("prefill", 0)], Vec::new())
        .unwrap();
    let mut prefill_rank = rank(&here.socket, "prefill", 0, claim);
    let prefill_tensor = tensor(key(10), "values", 0, 0, Access::Exclusive);
    prefill_rank
        .put(vec![prefill_tensor], &[(host(144), prefill_planes.as_fd())])
        .unwrap();
    prefill_rank
        .place("units".into(), "gpu0".into(), 0, values(key(10)))
        .unwrap();
    let decode_planes = memfd(&[0; 144]);
    let there_claim = head(&there.socket)
        .claim(Claimant::Head, 0, vec![scope("decode", 0)], Vec::new())
        .unwrap();
    let mut decode_rank = rank(&there.socket, "decode", 0, there_claim);
    let decode_tensor = tensor(key(11), "values", 0, 0, Access::Exclusive);
    decode_rank
        .put(vec![decode_tensor], &[(host(144), decode_planes.as_fd())])
        .unwrap();

    // The engine's head states where the decode rank's units lie, as that
    // rank stated it over the engine's link.
    let mut engine = engine_head(&here.socket, 0);
    let (opened, descriptors) = engine.open_pool(layout(0)).unwrap();
    engine
        .place_remote(
            "units".into(),
            "gpu1".into(),
            0,
            there.tcp.clone(),
            values(key(11)),
        )
        .unwrap();
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

    // Into the rank of the other host: its host's Transfer Server reads
    // units 1 and 2 from this host.
    let written = pattern(32, 5);
    prefill_planes.write_all_at(&written, 16).unwrap();
    let transfer = |group| Request::Transfer {
        owner: OwnerId(1),
        group,
        after: Vec::new(),
    };
    let runs = vec![(KV, 0, (region, 1, 1), true), (KV, 1, (region, 2, 1), true)];
    let moved = prefill.submit(transfer(DECODE));
    assert_eq!(
        prefill.next(),
        (moved, Completion::Transferred { slot: 1, runs })
    );
    assert_eq!(read(&decode_planes, 16, 32), written);

    // Back from it: this host's Transfer Server reads them from there.
    let rewritten = pattern(32, 9);
    decode_planes.write_all_at(&rewritten, 16).unwrap();
    let moved = prefill.submit(transfer(PREFILL));
    assert!(matches!(
        prefill.next(),
        (id, Completion::Transferred { .. }) if id == moved
    ));
    assert_eq!(read(&prefill_planes, 16, 32), rewritten);
}

/// The group that writes products in the main process of the session's
/// host, and the group that reads them in the main process of another host.
const WRITER: GroupId = GroupId(0);
const READER: GroupId = GroupId(1);

/// The class of products, sized, in units of 256 bytes.
const PRODUCT: u8 = 0;

/// The indices of [`processes`]' product buffers.
const WRITTEN: usize = 2;
const RECEIVING: usize = 3;

/// The layout of an epoch whose main processes lie on two hosts: the
/// writer's rank backs products in the product buffer of process `a`, and
/// the reader's ranks, in process `b`, read them there, receiving them in
/// the product buffer of the reader's rank there. Each group is a scheduler
/// of its own.
fn processes() -> Open {
    let region =
        |name: &str, unit_bytes, reserved, units, memory: &str| ::storage::runtime::Region {
            name: name.into(),
            tier: Tier::Device,
            space: name.into(),
            unit_bytes,
            part_bytes: unit_bytes,
            reserved,
            units,
            limit: reserved + units,
            segments: Vec::new(),
            memory: memory.into(),
        };
    let group = |id, memory: &str, slots, regions, receives| Group {
        id,
        memory: memory.into(),
        slots,
        regions,
        receives,
        parts: Vec::new(),
    };
    Open {
        version: VERSION,
        epoch: 0,
        trial: false,
        regions: vec![
            region("rows a", 8, 1, 2, ""),
            region("rows b", 8, 1, 2, ""),
            region("products a", 256, 0, 4, "process a"),
            region("products b", 256, 0, 4, "process b"),
        ],
        classes: vec![Class {
            units: 0,
            host: None,
            state_in_slot: false,
            tensors: [1; 32],
            encoding: [0; 32],
            source: None,
        }],
        trajectory: PRODUCT,
        groups: vec![
            group(
                WRITER,
                "process a",
                0,
                vec![vec![WRITTEN as u8]],
                Vec::new(),
            ),
            group(
                READER,
                "process b",
                1,
                vec![Vec::new()],
                vec![vec![RECEIVING as u8]],
            ),
        ],
        schedulers: vec![vec![WRITER], vec![READER]],
        peers: Vec::new(),
    }
}

/// Where a rank's product buffer of four units of 256 bytes lies: in the
/// host tensor `key`.
fn products(key: Key) -> Vec<UnitRun> {
    vec![UnitRun {
        field: "products".into(),
        key,
        location: Location::Host { numa_node: None },
        first: 0,
        count: 4,
        offset: 0,
        bytes: 256,
    }]
}

#[test]
#[ignore = "requires NIXL"]
fn a_crossing_copies_a_product_into_a_main_process_of_another_host_through_that_host_s_transfer_server()
 {
    let here = network_server();
    let there = network_server();

    // The writer's rank, on the session's host, and the reader's, on the
    // other, each hold a product buffer in storage of its own, retained by
    // its host's Storage Server; the head states where the reader's lies.
    let buffers = [memfd(&[0; 1024]), memfd(&[0; 1024])];
    let mut ranks = Vec::new();
    for ((server, group, name), buffer) in [(&here, "writer", 10), (&there, "reader", 11)]
        .into_iter()
        .zip(&buffers)
    {
        let claim = head(&server.socket)
            .claim(Claimant::Head, 0, vec![scope(group, 0)], Vec::new())
            .unwrap();
        let mut worker = rank(&server.socket, group, 0, claim);
        let buffer_tensor = tensor(key(name), "products", 0, 0, Access::Exclusive);
        worker
            .put(vec![buffer_tensor], &[(host(1024), buffer.as_fd())])
            .unwrap();
        ranks.push(worker);
    }
    ranks[0]
        .place(
            "products a".into(),
            "process a".into(),
            0,
            products(key(10)),
        )
        .unwrap();
    let mut engine = engine_head(&here.socket, 0);
    let (opened, descriptors) = engine.open_pool(processes()).unwrap();
    engine
        .place_remote(
            "products b".into(),
            "process b".into(),
            0,
            there.tcp.clone(),
            products(key(11)),
        )
        .unwrap();
    let (written, receiving) = (opened.regions[WRITTEN].0, opened.regions[RECEIVING].0);
    let mut schedulers = Scheduler::all(&opened, descriptors);
    let writer = &mut schedulers[0];

    // The writer's batch wrote a product of two units in its rank's buffer.
    writer.ask(Request::Attach {
        owner: OwnerId(1),
        group: WRITER,
    });
    let allocate = Request::Allocate {
        owner: OwnerId(1),
        class: PRODUCT,
        at: 0,
        count: 1,
        units: 2,
    };
    let allocated = Completion::Allocated {
        runs: vec![(written, 0, 2)],
        evicted: Vec::new(),
    };
    assert_eq!(writer.ask(allocate), allocated);
    let product = pattern(512, 7);
    buffers[0].write_all_at(&product, 0).unwrap();

    // The crossing takes a run in the reader's buffer, which the other
    // host's Transfer Server fills from this host's.
    let crossing = Request::Replicate {
        owner: OwnerId(1),
        class: PRODUCT,
        at: 0,
        from: written,
        group: READER,
        after: Vec::new(),
    };
    let replicated = Completion::Replicated {
        runs: vec![(receiving, 0, 2)],
        evicted: Vec::new(),
    };
    assert_eq!(writer.ask(crossing), replicated);
    assert_eq!(read(&buffers[1], 0, 512), product);
    assert_eq!(read(&buffers[1], 512, 512), [0; 512], "only the run moves");
}
