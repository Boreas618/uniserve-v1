//! Buffer pools that reach other hosts: a prefetch that reads the host
//! copies another host's Storage Server holds (a peer), a worker group with
//! a rank on another host, which a transfer reaches through that host's
//! Transfer Server, and the deadline of a transfer.
//!
//! Two Storage Servers of this host stand for two hosts, each on a thread
//! of the test process with a socket and a TCP endpoint of its own; they
//! reach each other only by their endpoints. Reading another host's storage
//! crosses the network through NIXL, which these tests do without: each
//! server runs a stand-in Transfer Server
//! (`stand_in_network_transfer_server`), which copies the storage its
//! server gives it as the real one does, and reads a region of another host
//! from the file of the test's network directory that stands for it: a
//! link to the descriptor the test holds of that region's storage.

use std::collections::HashMap;
use std::fs::File;
use std::os::fd::{AsFd, AsRawFd};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ::storage::protocol::{
    Allocation, Event, Link, LinkBandwidth, LinkEnd, Move, Region, Request as Message, TaskStatus,
};
use ::storage::runtime::{Completion, OwnerId, Request, Source};
use ::storage::{Access, Client, Location, Role};
use storage_server::Launch;

use crate::server::{FORKING, Running, engine_head, head, host, key, memfd, scope, tensor};
use crate::sessions::{
    DECODE, HOST, KV, PREFILL, Scheduler, UNITS, host_copy, host_parts, layout, rank,
    split_host_region, store_on_host, values,
};
use crate::transfer::{pattern, read, transfer_pid};

/// The file of a network directory that stands for `region` of the host
/// whose Storage Server listens at `endpoint`.
fn network_file(network: &Path, endpoint: &str, region: &Region) -> PathBuf {
    let name = match region {
        Region::Part { region, rank } => format!("part-{region}-{rank}"),
        Region::Tensor { key, .. } => format!("tensor-{}", key.hex()),
        other => panic!("no host region or tensor: {other:?}"),
    };
    network.join(endpoint.replace(':', "_")).join(name)
}

/// Makes `region` of the host at `endpoint` the storage the test holds as
/// `file`, for the stand-in Transfer Servers that read it.
fn link(network: &Path, endpoint: &str, region: &Region, file: &impl AsRawFd) {
    let path = network_file(network, endpoint, region);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let target = format!("/proc/{}/fd/{}", std::process::id(), file.as_raw_fd());
    std::os::unix::fs::symlink(target, path).unwrap();
}

/// The Transfer Server these tests' servers start: it copies between the
/// storages its server gives it and from the files of the network directory
/// that stand for other hosts' regions. With a gate, it runs no task until
/// the gate's file exists, and a task cancelled meanwhile ends cancelled.
#[test]
#[ignore = "the Transfer Server this file's tests start as the server's child"]
fn stand_in_network_transfer_server() {
    let (Ok(socket), Some(network)) = (
        std::env::var("STORAGE_TEST_SOCKET"),
        std::env::var_os("STORAGE_TEST_NETWORK"),
    ) else {
        return;
    };
    let network = PathBuf::from(network);
    let gate = std::env::var_os("STORAGE_TEST_GATE").map(PathBuf::from);
    let mut server = Client::connect(Path::new(&socket), Role::Transfer).unwrap();
    let link = Link {
        source: LinkEnd::Host,
        destination: LinkEnd::Host,
    };
    let measured = Message::Measured {
        links: vec![LinkBandwidth {
            link,
            bytes_per_second: 1 << 30,
        }],
        contexts: Vec::new(),
    };
    server.call(&measured, &[]).unwrap();

    let mut storages: HashMap<Allocation, File> = HashMap::new();
    let mut tasks: Vec<(u64, Vec<Move>)> = Vec::new();
    let mut ended = Vec::new();
    loop {
        // The server closed the connection: it stopped.
        if server.pump().is_err() {
            return;
        }
        while let Some((event, descriptors)) = server.next_event() {
            match event {
                Event::Allocation { allocation, .. } => {
                    if let Some(descriptor) = descriptors.into_iter().next() {
                        storages.insert(allocation, File::from(descriptor));
                    }
                }
                Event::AllocationReleased { allocation } => {
                    storages.remove(&allocation);
                }
                Event::Task { task, moves, .. } => tasks.push((task, moves)),
                Event::Cancel { task } => {
                    if let Some(at) = tasks.iter().position(|(waiting, _)| *waiting == task) {
                        tasks.remove(at);
                        ended.push((task, 0, TaskStatus::Cancelled));
                    }
                }
                _ => {}
            }
        }
        if gate.as_ref().is_none_or(|gate| gate.exists()) {
            for (task, moves) in std::mem::take(&mut tasks) {
                let (bytes, status) = copy(&storages, &network, &moves);
                ended.push((task, bytes, status));
            }
        }
        for (task, bytes, status) in std::mem::take(&mut ended) {
            let copied = Message::Copied {
                task,
                bytes,
                status: Some(status),
            };
            if server.call(&copied, &[]).is_err() {
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Copies a task's moves: the bytes copied, and how the task ended.
fn copy(storages: &HashMap<Allocation, File>, network: &Path, moves: &[Move]) -> (u64, TaskStatus) {
    let mut copied = 0;
    for step in moves {
        let source = match &step.source {
            Allocation::Remote {
                endpoint, region, ..
            } => File::open(network_file(network, endpoint, region)).ok(),
            allocation => storages
                .get(allocation)
                .and_then(|file| file.try_clone().ok()),
        };
        let (Some(source), Some(destination)) = (source, storages.get(&step.destination)) else {
            let message = format!("a move's storage is unknown: {step:?}");
            return (copied, TaskStatus::Failed { message });
        };
        let mut bytes = vec![0; step.bytes as usize];
        for row in 0..step.rows {
            let from = step.source_offset + row * step.source_pitch;
            let to = step.destination_offset + row * step.destination_pitch;
            let moved = source
                .read_exact_at(&mut bytes, from)
                .and_then(|()| destination.write_all_at(&bytes, to));
            if let Err(error) = moved {
                let message = error.to_string();
                return (copied, TaskStatus::Failed { message });
            }
            copied += step.bytes;
        }
    }
    (copied, TaskStatus::Complete)
}

/// A server whose Transfer Server is the stand-in, which reads other hosts'
/// regions from `network`, held by `gate` when given; with the buffer
/// pools' deadlines when given.
fn network_server(
    network: &Path,
    gate: Option<&Path>,
    deadlines: Option<(Duration, Duration)>,
) -> Running {
    let launch = |socket: &Path| {
        let mut arguments: Vec<std::ffi::OsString> = vec![
            "CUDA_VISIBLE_DEVICES=".into(),
            format!("STORAGE_TEST_SOCKET={}", socket.display()).into(),
            format!("STORAGE_TEST_NETWORK={}", network.display()).into(),
        ];
        if let Some(gate) = gate {
            arguments.push(format!("STORAGE_TEST_GATE={}", gate.display()).into());
        }
        arguments.push(std::env::current_exe().unwrap().into());
        arguments.extend(
            [
                "--exact",
                "hosts::stand_in_network_transfer_server",
                "--ignored",
                "--nocapture",
                "--quiet",
            ]
            .map(Into::into),
        );
        Some(Launch {
            program: "env".into(),
            arguments,
        })
    };
    // The server forks the Transfer Server; no test's dropped connection
    // must stay open in the child meanwhile.
    let _forking = FORKING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let server = match deadlines {
        Some(deadlines) => Running::with_deadlines(deadlines, launch),
        None => Running::with_transfer(launch),
    };
    transfer_pid(&server.socket, None);
    server
}

/// Waits until the server at `socket` runs a transfer task.
fn await_transfer(socket: &Path) {
    let mut operator = Client::connect(socket, Role::Operator).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while operator
        .list()
        .unwrap()
        .transfer
        .is_none_or(|transfer| transfer.tasks.is_empty())
    {
        assert!(Instant::now() < deadline, "no transfer task started");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The parts of a host region whose storage the test holds, as files.
fn files(parts: Vec<std::os::fd::OwnedFd>) -> Vec<File> {
    parts.into_iter().map(File::from).collect()
}

/// A prefetch of keys 7 and 9 from the peers.
fn from_peers() -> Request {
    Request::Prefetch {
        class: KV,
        units: 0,
        keys: vec![key(7), key(9)],
        source: Source::Peers,
        deadline: u64::MAX / 2,
    }
}

#[test]
fn a_prefetch_from_the_peers_reads_the_host_copies_a_peer_held_throughout() {
    let network = tempfile::tempdir().unwrap();

    // The peer cached key 7 on its host, in unit 0 of each part of its host
    // region; the test holds the parts' storage.
    let peer = crate::server::Running::with_transfer(|_| None);
    let (_peer_engine, peer_schedulers, peer_opened, peer_parts) = host_copy(&peer);
    let peer_parts = files(peer_parts);
    let written = [pattern(16, 3), pattern(16, 4)];
    for (rank, (part, bytes)) in peer_parts.iter().zip(&written).enumerate() {
        part.write_all_at(bytes, 0).unwrap();
        let region = Region::Part {
            region: "host".into(),
            rank: rank as u32,
        };
        link(network.path(), &peer.tcp, &region, part);
    }

    // The reading host names a peer nothing listens on, and the peer.
    let reading = network_server(network.path(), None, None);
    let claim = head(&reading.socket)
        .claim(
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
        Completion::Prefetched { found: 1 }
    );
    for (part, bytes) in parts.iter().zip(&written) {
        assert_eq!(read(part, 0, 16), *bytes);
    }
    let listed = Completion::Listing {
        keys: vec![(key(7), KV, 64, 0b010, 0)],
    };
    assert_eq!(prefill.ask(Request::List), listed);

    // The peer's holds ended with the read: its copy is evictable again.
    peer_schedulers[0].await_capacity(peer_opened.regions[HOST].0, (3, 0, 1));
}

/// Both the reading engine and the peer's have a rank on each of two hosts:
/// each part of the read copy is read from the host the peer's part of its
/// index lies on, by the Transfer Server of the host the reader's part lies
/// on.
#[test]
fn a_prefetch_from_the_peers_reads_each_part_on_the_host_it_lies_on() {
    let network = tempfile::tempdir().unwrap();

    // The peer's engine cached key 7 on the host, in unit 0 of its part
    // on its scheduler's host and of its part on its other host.
    let peer = Running::with_transfer(|_| None);
    let peer_other = Running::with_transfer(|_| None);
    let (_peer_engine, _peer_peer, mut peer_schedulers, _, peer_parts) =
        split_host_region(&peer, &peer_other, Vec::new());
    store_on_host(&mut peer_schedulers[0]);
    let peer_parts = files(peer_parts.into());
    let written = [pattern(16, 3), pattern(16, 4)];
    for (rank, (part, endpoint)) in peer_parts
        .iter()
        .zip([&peer.tcp, &peer_other.tcp])
        .enumerate()
    {
        part.write_all_at(&written[rank], 0).unwrap();
        let region = Region::Part {
            region: "host".into(),
            rank: rank as u32,
        };
        link(network.path(), endpoint, &region, part);
    }

    // The reading engine names the peer.
    let reading = network_server(network.path(), None, None);
    let reading_other = network_server(network.path(), None, None);
    let (_engine, _peer, mut schedulers, _, parts) =
        split_host_region(&reading, &reading_other, vec![peer.tcp.clone()]);
    let parts = files(parts.into());
    let prefill = &mut schedulers[0];

    // Key 7 is read into unit 0 of each part, the part on the other host
    // by that host's Transfer Server; key 9 is held by no peer.
    assert_eq!(
        prefill.ask(from_peers()),
        Completion::Prefetched { found: 1 }
    );
    for (part, bytes) in parts.iter().zip(&written) {
        assert_eq!(read(part, 0, 16), *bytes);
    }
    let listed = Completion::Listing {
        keys: vec![(key(7), KV, 64, 0b010, 0)],
    };
    assert_eq!(prefill.ask(Request::List), listed);
}

#[test]
fn a_prefetch_from_a_peer_lost_during_the_read_caches_nothing() {
    let network = tempfile::tempdir().unwrap();
    let gate = network.path().join("gate");
    let peer = crate::server::Running::with_transfer(|_| None);
    let (peer_engine, peer_schedulers, _, peer_parts) = host_copy(&peer);
    let peer_parts = files(peer_parts);
    for (rank, part) in peer_parts.iter().enumerate() {
        let region = Region::Part {
            region: "host".into(),
            rank: rank as u32,
        };
        link(network.path(), &peer.tcp, &region, part);
    }

    let reading = network_server(network.path(), Some(&gate), None);
    let claim = head(&reading.socket)
        .claim(
            0,
            vec![scope("prefill", 0), scope("prefill", 1)],
            Vec::new(),
        )
        .unwrap();
    let mut open = layout(0);
    open.peers = vec![peer.tcp.clone()];
    let mut engine = engine_head(&reading.socket, 0);
    let (opened, descriptors) = engine.open_pool(open).unwrap();
    let _parts = host_parts(&reading.socket, claim);
    let host_region = opened.regions[HOST].0;
    let mut schedulers = Scheduler::all(&opened, descriptors);
    let prefill = &mut schedulers[0];

    // The read runs while the peer holds the copy; the peer is lost before
    // it ends, so its holds did not stand throughout.
    let prefetch = prefill.submit(from_peers());
    await_transfer(&reading.socket);
    drop((peer_schedulers, peer_engine));
    drop(peer);
    std::fs::write(&gate, b"").unwrap();

    assert_eq!(
        prefill.next(),
        (prefetch, Completion::Prefetched { found: 0 })
    );
    let listed = Completion::Listing { keys: Vec::new() };
    assert_eq!(prefill.ask(Request::List), listed);
    prefill.await_capacity(host_region, (4, 0, 0));
}

#[test]
fn a_transfer_reaches_a_rank_of_another_host_through_that_host_s_transfer_server() {
    let network = tempfile::tempdir().unwrap();
    let here = network_server(network.path(), None, None);
    let there = network_server(network.path(), None, None);
    let location = Some(Location::Host { numa_node: None });

    // The prefill rank is on the scheduler's host, the decode rank on the
    // other one, each holding the KV region's values in storage of its
    // own, retained by its host's Storage Server.
    let prefill_planes = memfd(&[0; 144]);
    let claim = head(&here.socket)
        .claim(0, vec![scope("prefill", 0)], Vec::new())
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
        .claim(0, vec![scope("decode", 0)], Vec::new())
        .unwrap();
    let mut decode_rank = rank(&there.socket, "decode", 0, there_claim);
    let decode_tensor = tensor(key(11), "values", 0, 0, Access::Exclusive);
    decode_rank
        .put(vec![decode_tensor], &[(host(144), decode_planes.as_fd())])
        .unwrap();
    let tensor_region = |byte| Region::Tensor {
        key: key(byte),
        at: location,
    };
    link(
        network.path(),
        &here.tcp,
        &tensor_region(10),
        &prefill_planes,
    );
    link(
        network.path(),
        &there.tcp,
        &tensor_region(11),
        &decode_planes,
    );

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

#[test]
fn a_transfer_past_the_server_s_deadline_fails_and_leaves_the_owner_where_it_was() {
    let network = tempfile::tempdir().unwrap();
    let gate = network.path().join("gate");
    let deadlines = (Duration::from_millis(200), Duration::from_secs(60));
    let server = network_server(network.path(), Some(&gate), Some(deadlines));
    let claim = head(&server.socket)
        .claim(0, vec![scope("prefill", 0), scope("decode", 0)], Vec::new())
        .unwrap();
    let planes = [memfd(&[0; 144]), memfd(&[0; 144])];
    for (name, memory, group, planes) in [
        (10, "gpu0", "prefill", &planes[0]),
        (11, "gpu1", "decode", &planes[1]),
    ] {
        let mut worker = rank(&server.socket, group, 0, claim);
        let values_tensor = tensor(key(name), "values", 0, 0, Access::Exclusive);
        worker
            .put(vec![values_tensor], &[(host(144), planes.as_fd())])
            .unwrap();
        worker
            .place("units".into(), memory.into(), 0, values(key(name)))
            .unwrap();
    }

    let mut engine = engine_head(&server.socket, 0);
    let (opened, descriptors) = engine.open_pool(layout(0)).unwrap();
    let mut schedulers = Scheduler::all(&opened, descriptors);
    let prefill = &mut schedulers[0];
    prefill.ask(Request::Attach {
        owner: OwnerId(1),
        group: PREFILL,
    });
    prefill.ask(Request::Allocate {
        owner: OwnerId(1),
        class: KV,
        at: 0,
        count: 2,
        units: 0,
    });

    // The Transfer Server holds the task past the deadline: it is
    // cancelled, and the transfer fails.
    let started = Instant::now();
    let transfer = prefill.submit(Request::Transfer {
        owner: OwnerId(1),
        group: DECODE,
        after: Vec::new(),
    });
    assert_eq!(prefill.next(), (transfer, Completion::Failed));
    assert!(started.elapsed() >= deadlines.0);

    // The owner is in its group with its blocks, and takes requests again.
    let release = Request::Release {
        owner: OwnerId(1),
        class: KV,
        at: 0,
        count: 2,
        after: Vec::new(),
    };
    assert_eq!(prefill.ask(release), Completion::Done);
}
