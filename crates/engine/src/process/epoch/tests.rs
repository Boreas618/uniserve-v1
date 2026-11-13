//! The epoch's jobs over the parts of a serving engine on the meta device
//! (`operations::tests::Node`).

use std::time::{Duration, Instant};

use super::{drain, release};
use crate::process::operations::ask;
use crate::process::operations::tests::{LONG, Node, SEGMENT_UNITS, UNIT_BYTES, segment};
use crate::scheduler::Message;

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

    let (devices, bytes) = release(&node.head, &node.binding, &node.workers, kv, 31).unwrap();
    assert!(devices.is_empty(), "a rank of the meta device has no GPU");
    assert_eq!(bytes, 2 * u64::from(SEGMENT_UNITS) * UNIT_BYTES);
    assert_eq!(node.segments_seen(2), [segment(0), segment(1)]);

    // A region of no unit covers none of segment 0's units past the reserved one either.
    let reached = ask(&node.schedulers, |reply| Message::Resize(kv, 0, reply)).unwrap();
    assert_eq!(reached, [0]);
    let (_, bytes) = release(&node.head, &node.binding, &node.workers, kv, 0).unwrap();
    assert_eq!(bytes, 2 * u64::from(SEGMENT_UNITS) * UNIT_BYTES);
    assert!(node.segments_seen(3).is_empty());
}
