//! Buffer pools that reach other hosts: a handoff's import of a lend of
//! another host, a worker group with a rank on another host, which a transfer
//! reaches through that host's Transfer Server, and a product that crosses to
//! a main process of another host.
//!
//! Two Storage Servers of this host stand for two hosts, each on a thread
//! of the test process with a socket and a TCP endpoint of its own; they
//! reach each other only by their endpoints. Each runs the real Transfer
//! Server, copying on no GPU, which reads another host's storage through
//! NIXL, so these cases need NIXL; run them with `cargo test -p
//! storage-server --test it hosts -- --ignored`.

use std::fs::File;
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::fs::FileExt;

use ::storage::protocol::UnitRun;
use ::storage::runtime::{
    Class, Completion, Group, GroupId, Lender, Open, OwnerId, Request, Tier, VERSIONS,
};
use ::storage::{Access, Claimant, Key, Location};

use crate::server::{Running, engine_head, head, host, key, memfd, scope, tensor, transfer_child};
use crate::sessions::{
    HEADS, KV, Scheduler, assert_imported, handoff_ranks, imported_units, layout, lend_two_pages,
    rank, write_pages,
};
use crate::transfer::{Paths, calibration, pattern, read, transfer_pid};

/// A server whose Transfer Server copies on no GPU and reads other hosts'
/// regions through NIXL, at the rate its calibration states.
fn network_server() -> Running {
    let paths = Paths {
        reads: true,
        ..Paths::HOST
    };
    let server = Running::with_children(Some(transfer_child(calibration(paths))));
    transfer_pid(&server.socket, None);
    server
}

/// The two replicas of the handoff cases on two hosts: the prefill replica's engine and its
/// rank on `lender`, the decode replica's engine and its two ranks on `importer`, each engine
/// with its scheduler; the prefill owner lent two pages. Returns the importer's scheduler, the
/// lender (named by its host's endpoint) and the decode ranks' tensors.
fn two_hosts(
    lender: &Running,
    importer: &Running,
) -> (Scheduler, Lender, [OwnedFd; 2], Vec<Box<dyn std::any::Any>>) {
    let lender_claim = head(&lender.socket)
        .claim(Claimant::Head, 0, vec![scope("prefill", 0)], Vec::new())
        .unwrap();
    let importer_claim = head(&importer.socket)
        .claim(
            Claimant::Head,
            0,
            vec![scope("decode", 1), scope("decode", 2)],
            Vec::new(),
        )
        .unwrap();
    let [(prefill, lent), ..] = handoff_ranks(&lender.socket, lender_claim);
    let [_, (first, first_storage), (second, second_storage)] =
        handoff_ranks(&importer.socket, importer_claim);

    let mut prefill_engine = engine_head(&lender.socket, 0);
    let (prefill_opened, descriptors) = prefill_engine.open_pool(layout(0)).unwrap();
    let mut prefill_schedulers = Scheduler::all(&prefill_opened, descriptors);
    let mut decode_engine = engine_head(&importer.socket, 0);
    let (decode_opened, descriptors) = decode_engine.open_pool(layout(0)).unwrap();
    let mut decode_schedulers = Scheduler::all(&decode_opened, descriptors);

    let mut both = [prefill_schedulers.remove(0), decode_schedulers.remove(1)];
    let mut lender_owner = lend_two_pages(&mut both, &prefill_opened, |units| {
        write_pages(&File::from(lent.try_clone().unwrap()), units, (0, HEADS));
    });
    lender_owner.endpoint = Some(lender.tcp.clone());
    let [_, decode] = both;
    // The engines, ranks and schedulers serve on while the test holds them.
    let held: Vec<Box<dyn std::any::Any>> = vec![
        Box::new(prefill),
        Box::new(first),
        Box::new(second),
        Box::new(prefill_engine),
        Box::new(decode_engine),
        Box::new(prefill_schedulers),
        Box::new(decode_schedulers),
    ];
    (decode, lender_owner, [first_storage, second_storage], held)
}

#[test]
#[ignore = "requires NIXL"]
fn an_import_reads_a_lend_of_another_host_through_this_host_s_transfer_server() {
    let lender = network_server();
    let importer = network_server();
    let (mut decode, lent, [first, second], _held) = two_hosts(&lender, &importer);

    // This host's Transfer Server READs each head from the lender's host into its rank.
    let import = Request::Import {
        owner: OwnerId(2),
        tables: vec![(KV, 0, 2)],
        lender: lent.clone(),
        push: false,
        transport: scheduling::kv::Transport::Nixl,
        trace: 0,
    };
    let units = imported_units(decode.ask(import));
    assert_imported(&[&first, &second], &units);

    // The import returned the lend there: another import of it fails.
    let again = Request::Import {
        owner: OwnerId(2),
        tables: vec![(KV, 2, 2)],
        lender: lent,
        push: false,
        transport: scheduling::kv::Transport::Nixl,
        trace: 0,
    };
    assert_eq!(decode.ask(again), Completion::Failed);
}

#[test]
#[ignore = "requires NIXL"]
fn an_import_pushed_from_another_host_is_written_by_that_host_s_transfer_server() {
    let lender = network_server();
    let importer = network_server();
    let (mut decode, lent, [first, second], _held) = two_hosts(&lender, &importer);

    // The lender's host's Transfer Server WRITEs each head into this host's ranks.
    let import = Request::Import {
        owner: OwnerId(2),
        tables: vec![(KV, 0, 2)],
        lender: lent,
        push: true,
        transport: scheduling::kv::Transport::Nixl,
        trace: 0,
    };
    let units = imported_units(decode.ask(import));
    assert_imported(&[&first, &second], &units);
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
        oldest: VERSIONS.oldest(),
        newest: Some(VERSIONS.newest()),
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
            None,
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
            None,
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
