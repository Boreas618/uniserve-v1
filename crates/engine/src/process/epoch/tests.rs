//! The epoch's jobs over the parts of a serving engine on the meta device
//! (`operations::tests::Node`).

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use proto::config::{ParallelConfig, WorkerGroupConfig, WorkerGroupId, WorkerRank};

use super::{by_host, drain, promote, release};
use crate::process::operations::ask;
use crate::process::operations::tests::{KV_UNITS, LONG, Node, SEGMENT_UNITS, UNIT_BYTES, segment};
use crate::scheduler::Message;

/// The epoch whose sessions read what the buffer pool holds, beside the engines of each case.
const READER: u64 = 9;

#[test]
fn a_drain_with_no_request_replies_at_once() {
    let node = Node::start();

    let replies = drain(
        &node.schedulers,
        Instant::now() + Duration::from_secs(3600),
        false,
    )
    .unwrap();
    assert_eq!(replies.len(), 1);
    for reply in replies {
        // Long before the deadline: a scheduler that holds no request has nothing to end.
        assert_eq!(reply.recv_timeout(LONG), Ok(()));
    }
}

#[test]
fn a_release_after_a_shrink_withdraws_only_the_segments_past_the_units() {
    let node = Node::start();
    let kv = node.kv();

    // The region keeps 31 units past its reserved one, units 1 to 31: segments 0 and 1 hold some
    // of them, and segments 2 and 3, from unit 32 on, none.
    let reached = ask(&node.schedulers, |reply| Message::Resize(kv, 31, reply)).unwrap();
    assert_eq!(reached, [31]);

    let (released, bytes) = release(&node.head, &node.binding, kv, 31).unwrap();
    assert!(
        released.iter().all(|(_, devices)| devices.is_empty()),
        "a rank of the meta device has no GPU"
    );
    assert_eq!(bytes, 2 * u64::from(SEGMENT_UNITS) * UNIT_BYTES);
    assert_eq!(node.segments_seen(2), [segment(0), segment(1)]);

    // A region of no unit covers none of segment 0's units past the reserved one either.
    let reached = ask(&node.schedulers, |reply| Message::Resize(kv, 0, reply)).unwrap();
    assert_eq!(reached, [0]);
    let (_, bytes) = release(&node.head, &node.binding, kv, 0).unwrap();
    assert_eq!(bytes, 2 * u64::from(SEGMENT_UNITS) * UNIT_BYTES);
    assert!(node.segments_seen(3).is_empty());
}

/// The device cache's size is no layout (`kv_cache_config.max_tokens`): an epoch of the node's
/// layout whose capacity numbers half the node's KV planes names the node's region of them, which
/// keeps the node's numbering, and its promotion resizes the region to its own units; the node's
/// promotion, as at a rollback, resizes the region to the node's units again; and once another
/// epoch of half the cache was promoted, the promotion of one of the whole cache grows the region
/// back from the memory the shrink left in its space. Each size is what a session opened then
/// finds in the region: what its blocks can take.
#[test]
fn an_epoch_of_the_same_layout_resizes_the_shared_region_to_its_own_capacity() {
    let node = Node::start();
    let kv = node.kv();
    // Every unit past the padding unit, of the whole cache and of half of it.
    let (whole, half) = (KV_UNITS - 1, KV_UNITS / 2 - 1);
    assert_eq!(node.units_seen(READER), whole);

    let target = node.successor(2, KV_UNITS / 2);
    assert_eq!(
        target.kv(),
        kv,
        "an epoch of the same layout names the node's region"
    );
    promote(&target.schedulers, Some(&target.head), &target.binding).unwrap();
    assert_eq!(node.units_seen(READER), half);

    // The rollback: the node serves again, at its own size, and the target goes.
    promote(&node.schedulers, Some(&node.head), &node.binding).unwrap();
    assert_eq!(node.units_seen(READER), whole);
    drop(target);

    let shrunk = node.successor(3, KV_UNITS / 2);
    promote(&shrunk.schedulers, Some(&shrunk.head), &shrunk.binding).unwrap();
    assert_eq!(node.units_seen(READER), half);
    let grown = node.successor(4, KV_UNITS);
    promote(&grown.schedulers, Some(&grown.head), &grown.binding).unwrap();
    assert_eq!(node.units_seen(READER), whole);
}

/// The daemon is told of a release once per host whose devices gave memory back: this
/// process's devices on its own host, a peer's on the host its first rank is placed on, and a
/// peer whose ranks hold no device not at all.
#[test]
fn a_release_is_reported_once_per_host_its_ranks_gave_memory_back_on() {
    // The deployment numbers its ranks over its groups in order: ranks 0 and 1 on host a, ranks
    // 2 and 3 on host b, rank 4 on host c.
    let group = |id: &str, hosts: &[&str]| WorkerGroupConfig {
        id: WorkerGroupId(id.to_owned()),
        role: Default::default(),
        replicas: 1,
        ranks: hosts
            .iter()
            .map(|host| WorkerRank {
                host: (*host).to_owned(),
                device: "cuda:0".to_owned(),
            })
            .collect(),
        blocks: BTreeMap::new(),
        queue_depth: 2,
        mem_fraction_static: None,
    };
    let parallel = ParallelConfig {
        tensor_parallel_size: NonZeroUsize::MIN,
        data_parallel_size: NonZeroUsize::MIN,
        engine_host: "a".to_owned(),
        worker_groups: vec![
            group("prefill", &["a", "a", "b"]),
            group("decode", &["b", "c"]),
        ],
    };

    let released = vec![
        (None, vec![0, 1]),
        (Some(2), vec![0, 1]),
        (Some(4), Vec::new()),
    ];
    assert_eq!(
        by_host(&parallel, "a", released),
        [("a".to_owned(), vec![0, 1]), ("b".to_owned(), vec![0, 1])]
    );
}
