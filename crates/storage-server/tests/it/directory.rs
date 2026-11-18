//! The cluster directory on the leader node's Storage Server: what every node reports of its
//! pages, and what Mooncake's master states of the store's, as a head reads the prefix hits of
//! chains of page keys (`Request::Hits`).
//!
//! Another node is a second server on the test's host, which reports to the leader's over its
//! TCP endpoint, or a peer connection the test makes itself; the master is a ZMQ publisher of
//! the test's that sends batches as Mooncake's master packs them.

use std::time::{Duration, Instant};

use ::storage::protocol::{Hits, Holding, NodeHits};
use ::storage::runtime::Tier;
use ::storage::{Client, Key};

use crate::PATIENCE;
use crate::server::{Running, head, key};
use crate::sessions::host_copy;

/// Settings that report every 20 ms to `directory`, the leader's server, or hold the directory
/// when `None`, subscribing to the master's publisher at `events`.
fn settings(directory: Option<&str>, events: Option<String>) -> ::storage::Settings {
    ::storage::Settings {
        server: ::storage::ServerSettings {
            directory: directory.map(|endpoint| endpoint.parse().unwrap()),
            directory_period_ms: 20,
            ..::storage::ServerSettings::default()
        },
        connector: events.map(|events| ::storage::connector::Settings {
            master: "127.0.0.1:1".parse().unwrap(),
            metadata: ::storage::connector::Metadata::P2pHandshake,
            transport: ::storage::connector::Transport::Tcp,
            device: String::new(),
            segment_gib: 0,
            events: Some(events.parse().unwrap()),
        }),
    }
}

/// Asks `head` for the hits of `chain` until `done` holds of them.
fn hits_until(head: &mut Client, chain: &[Key], done: impl Fn(&Hits) -> bool) -> Hits {
    let started = Instant::now();
    loop {
        let hits = head.hits(vec![chain.to_vec()]).unwrap().remove(0);
        if done(&hits) {
            return hits;
        }
        assert!(started.elapsed() < PATIENCE, "the hits were {hits:?}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// A node's pages reach the leader's directory with its next report: a node whose buffer pool
/// caches key 7 on its device and holds its host copy states a hit of one page on both tiers,
/// none past the first page it lacks. Once its server ends, the connection it reported on
/// closes, and the leader forgets the node.
#[test]
fn a_node_s_pages_reach_the_leader_s_directory_until_its_connection_closes() {
    let leader = Running::with_settings(settings(None, None));
    let member = Running::with_settings(settings(Some(&leader.tcp), None));
    let (_engine, _schedulers, _opened, _parts) = host_copy(&member);

    let mut head = head(&leader.socket);
    let chain = [key(7), key(8), key(9)];
    // The host copy is committed once its store's copy ran, which a report may precede.
    let hits = hits_until(&mut head, &chain, |hits| {
        hits.nodes
            .iter()
            .any(|node| (node.device, node.host) == (1, 1))
    });
    assert_eq!(hits.nodes.len(), 1, "{hits:?}");
    assert_eq!(hits.store, 0);
    // A chain whose first page no node holds has no hit, whatever it holds after it.
    let later = head.hits(vec![vec![key(8), key(7)]]).unwrap();
    assert!(
        later[0]
            .nodes
            .iter()
            .all(|node| node.device == 0 && node.host == 0),
        "{later:?}"
    );

    drop(member);
    hits_until(&mut head, &chain, |hits| hits.nodes.is_empty());
}

/// A node's report replaces what it reported before, so the directory states what the node
/// last reported, however long ago: an entry a node no longer reports is gone at its next
/// report, and one reported once stays until then.
#[test]
fn a_report_replaces_what_its_node_reported_before() {
    let leader = Running::with_settings(settings(None, None));
    let mut peer = Client::connect_peer(&leader.tcp).unwrap();
    let report = |peer: &mut Client, device: &[u8], host: &[u8]| {
        let holding = |tier, keys: &[u8]| Holding {
            tier,
            keys: keys.iter().copied().map(key).collect(),
            capacity: 8,
        };
        peer.report_holdings(
            "model".into(),
            "other".into(),
            vec![holding(Tier::Device, device), holding(Tier::Host, host)],
        )
        .unwrap();
    };
    let mut head = head(&leader.socket);
    let chain = vec![key(1), key(2), key(3)];
    let other = |hits: Vec<Hits>| {
        hits[0]
            .nodes
            .iter()
            .find(|node| node.node == "other")
            .cloned()
    };

    report(&mut peer, &[1], &[1, 2, 3]);
    assert_eq!(
        other(head.hits(vec![chain.clone()]).unwrap()),
        Some(NodeHits {
            node: "other".into(),
            device: 1,
            host: 3,
        })
    );
    report(&mut peer, &[1, 2], &[]);
    assert_eq!(
        other(head.hits(vec![chain]).unwrap()),
        Some(NodeHits {
            node: "other".into(),
            device: 2,
            host: 0,
        })
    );
}

/// The object key the Mooncake connector writes page `byte`'s part of heads `heads` under.
fn object(byte: u8, heads: &str) -> String {
    format!("uniserve_{}_{}_{heads}", key(byte).hex(), key(0xee).hex())
}

/// A batch of `events` (`stored` or `removed`, object key) as Mooncake's master packs it
/// (`mooncake-store/src/kv_event/kv_event_publisher.cpp`, with its legacy fields and the object
/// key): `[timestamp_ms, [event…], dp_rank]`.
fn batch(events: &[(&str, String)]) -> Vec<u8> {
    use rmp::encode::{write_array_len, write_map_len, write_nil, write_str, write_uint};

    let mut out = Vec::new();
    write_array_len(&mut out, 3).unwrap();
    write_uint(&mut out, 1_795_012_729_927).unwrap();
    write_array_len(&mut out, events.len() as u32).unwrap();
    for (index, (kind, object)) in events.iter().enumerate() {
        let legacy = if *kind == "stored" {
            "BlockStored"
        } else {
            "BlockRemoved"
        };
        write_map_len(&mut out, 8).unwrap();
        write_str(&mut out, "event_id").unwrap();
        write_uint(&mut out, index as u64 + 1).unwrap();
        write_str(&mut out, "event_type").unwrap();
        write_str(&mut out, kind).unwrap();
        write_str(&mut out, "type").unwrap();
        write_str(&mut out, legacy).unwrap();
        write_str(&mut out, "tenant_id").unwrap();
        write_str(&mut out, "default").unwrap();
        write_str(&mut out, "backend_id").unwrap();
        write_str(&mut out, "mooncake-test").unwrap();
        write_str(&mut out, "medium").unwrap();
        write_str(&mut out, "cpu").unwrap();
        write_str(&mut out, "object_key").unwrap();
        write_str(&mut out, object).unwrap();
        write_str(&mut out, "parent_hash").unwrap();
        write_nil(&mut out).unwrap();
    }
    write_uint(&mut out, 0).unwrap();
    out
}

/// The directory subscribes to the master's object events: the pages whose objects the master
/// stored count as the store's hit, past them none, and a page whose every object the master
/// removed counts no more. A batch numbered below the last one is a master that restarted, whose
/// store holds nothing of what it held before.
#[test]
fn the_directory_follows_the_store_s_pages_by_the_master_s_object_events() {
    use zeromq::{Socket as _, SocketSend as _};

    // The socket's tasks (accepting the subscriber, reading its subscription) run on a worker
    // of the test's runtime while the test waits.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let _entered = runtime.enter();
    let mut publisher = zeromq::PubSocket::new();
    let bound = runtime
        .block_on(publisher.bind("tcp://127.0.0.1:0"))
        .unwrap();
    let endpoint = bound.to_string();
    let endpoint = endpoint.trim_start_matches("tcp://").to_owned();

    let leader = Running::with_settings(settings(None, Some(endpoint)));
    let mut head = head(&leader.socket);
    let chain = [key(1), key(2), key(3)];
    let mut send = |sequence: u64, events: &[(&str, String)]| {
        let frames = [Vec::new(), sequence.to_be_bytes().to_vec(), batch(events)];
        let mut message = zeromq::ZmqMessage::from(frames[0].clone());
        message.push_back(frames[1].clone().into());
        message.push_back(frames[2].clone().into());
        runtime.block_on(publisher.send(message)).unwrap();
    };

    // The subscriber connects at its own pace: the master's events are published until the
    // directory has taken them, each batch a full statement of the two pages' objects.
    let stored = [
        ("stored", object(1, "0-2")),
        ("stored", object(1, "2-2")),
        ("stored", object(2, "0-4")),
    ];
    let started = Instant::now();
    let mut sequence = 0_u64;
    while head.hits(vec![chain.to_vec()]).unwrap()[0].store != 2 {
        assert!(started.elapsed() < PATIENCE, "the directory took no event");
        sequence += 1;
        send(sequence, &stored);
        std::thread::sleep(Duration::from_millis(50));
    }

    // One of page 1's two objects removed leaves it held; both removed, page 1 is gone and so
    // is the chain's hit, though page 2 stays.
    send(sequence + 1, &[("removed", object(1, "0-2"))]);
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(head.hits(vec![chain.to_vec()]).unwrap()[0].store, 2);
    send(sequence + 2, &[("removed", object(1, "2-2"))]);
    hits_until(&mut head, &chain, |hits| hits.store == 0);
    assert_eq!(head.hits(vec![vec![key(2)]]).unwrap()[0].store, 1);

    // A restarted master numbers its batches from 1 again.
    send(1, &[("stored", object(1, "0-4"))]);
    hits_until(&mut head, &[key(1), key(2)], |hits| hits.store == 1);
}
