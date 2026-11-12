//! Another host's scheduler reading this host's pages: pins by key over
//! the server's TCP endpoint, which see only published epochs and end with
//! the peer's unpin or connection, and the reading host's reservations that
//! its own transfer tasks write.
//!
//! Each server runs on a thread of the test process and is reached by its
//! TCP endpoint as another host reaches it. No GPU and no NIXL is needed:
//! the pins are accounting of the source host's server, and the transfer
//! that writes the reading host's reservation never reaches a READ.

use std::io::Read;
use std::time::{Duration, Instant};

use ::storage::protocol::{
    ErrorKind, Event, PageSlot, PoolListing, Priority, Region, Request, Response, TaskStatus,
};
use ::storage::{Client, Key, Role};

use crate::server::{
    FORKING, Running, engine_head, epoch_worker, head, key, publish, scope, trial_pool,
};
use crate::transfer::{
    Gpus, Slow, pattern, read, remote_region, serving, slow, span, tensor_region, transfer_event,
};

/// How long a peer's blocking calls wait for an response.
const PEER_TIMEOUT: Duration = Duration::from_secs(30);

/// The layout of the pages `trial_pool` holds.
fn layout() -> Key {
    key(3)
}

/// Pool `pool` as the server on `socket` lists it.
fn listed(socket: &std::path::Path, pool: u64) -> PoolListing {
    let mut operator = Client::connect(socket, Role::Operator).unwrap();
    operator
        .list()
        .unwrap()
        .page_pools
        .into_iter()
        .find(|listed| listed.pool == pool)
        .expect("the pool is listed")
}

/// Wait until `done` holds of the listing of the server on `socket`, for a
/// minute at most.
fn await_listing(
    socket: &std::path::Path,
    what: &str,
    done: impl Fn(&::storage::protocol::Listing) -> bool,
) {
    let mut operator = Client::connect(socket, Role::Operator).unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    while !done(&operator.list().unwrap()) {
        assert!(Instant::now() < deadline, "{what} did not happen");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// A peer pins only the committed pages of published epochs, in the pools
/// of the layouts it names, and a pinned page is not evicted; the page of an
/// epoch on trial becomes visible to peers once the epoch is published.
#[test]
fn a_peer_pins_only_the_pages_of_published_epochs() {
    let server = Running::with_transfer(|_| None);
    let mut daemon = head(&server.socket);
    let claim = daemon.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    daemon.publish(vec![1]).unwrap();
    let mut worker = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    let pool = trial_pool(&mut worker);

    // Epoch 1 serves and publishes a page; epoch 2 is on trial and keeps
    // its page to itself.
    let mut serving = engine_head(&server.socket, 1);
    publish(&mut serving, pool, &[key(1)]);
    daemon.claim(2, vec![scope("text", 0)], Vec::new()).unwrap();
    let mut trial = engine_head(&server.socket, 2);
    publish(&mut trial, pool, &[key(2)]);

    let mut peer = Client::connect_peer(&server.tcp, PEER_TIMEOUT).unwrap();
    let pools = peer
        .peer_pin("model".into(), vec![layout()], vec![key(1), key(2), key(9)])
        .unwrap();
    assert_eq!(pools.len(), 1);
    assert_eq!(pools[0].pool, pool);
    assert_eq!(pools[0].shape.layout, layout());
    assert_eq!(pools[0].shape.page_bytes, 16);
    assert!(
        matches!(pools[0].slots[..], [Some(_), None, None]),
        "{:?}",
        pools[0].slots
    );

    // Pools of another layout, and another deployment's, hold nothing the
    // peer reads.
    assert!(
        peer.peer_pin("model".into(), vec![key(4)], vec![key(1)])
            .unwrap()
            .is_empty()
    );
    assert!(
        peer.peer_pin("other".into(), vec![layout()], vec![key(1)])
            .unwrap()
            .is_empty()
    );

    // The pinned page stays while the serving epoch fills the pool of four
    // slots: the page of epoch 2 gives way, the pinned one does not.
    let Response::Reserved { slots } = serving
        .call(
            &Request::Reserve {
                pool,
                writer: Some(scope("text", 0)),
                keys: vec![key(5), key(6), key(7), key(8)],
                window: None,
            },
            &[],
        )
        .unwrap()
        .0
    else {
        panic!("the reservation was refused");
    };
    assert_eq!(slots[3], PageSlot::Full);
    assert_eq!(listed(&server.socket, pool).pinned, 1);
    serving
        .call(
            &Request::Abort {
                pool,
                keys: vec![key(5), key(6), key(7), key(8)],
            },
            &[],
        )
        .unwrap();

    // An unpin ends the pin; the page stays committed.
    peer.peer_unpin(pool, vec![key(1)]).unwrap();
    let after = listed(&server.socket, pool);
    assert_eq!((after.pinned, after.ready), (0, 1));

    // Once published, epoch 2's pages are read by peers too.
    publish(&mut trial, pool, &[key(2)]);
    let before = peer
        .peer_pin("model".into(), vec![layout()], vec![key(2)])
        .unwrap();
    assert!(before.is_empty(), "{before:?}");
    daemon.publish(vec![2]).unwrap();
    let published = peer
        .peer_pin("model".into(), vec![layout()], vec![key(2)])
        .unwrap();
    assert!(
        matches!(published[..], [ref pool] if pool.slots[0].is_some()),
        "{published:?}"
    );

    // Pins are a peer's requests; a head's are refused.
    let refused = serving
        .call(
            &Request::PeerPin {
                deployment: "model".into(),
                layouts: vec![layout()],
                keys: vec![key(1)],
            },
            &[],
        )
        .unwrap()
        .0;
    assert!(
        matches!(
            refused,
            Response::Error {
                kind: ErrorKind::Invalid,
                ..
            }
        ),
        "{refused:?}"
    );
}

/// A peer's pins end with its connection: the page becomes evictable again.
#[test]
fn a_closed_peer_connection_ends_its_pins() {
    // The server must see the dropped connection close: no child is forked
    // from this process meanwhile.
    let _forking = FORKING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let server = Running::with_transfer(|_| None);
    let mut daemon = head(&server.socket);
    let claim = daemon.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    daemon.publish(vec![1]).unwrap();
    let mut worker = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    let pool = trial_pool(&mut worker);
    let mut serving = engine_head(&server.socket, 1);
    publish(&mut serving, pool, &[key(1), key(2)]);

    let mut peer = Client::connect_peer(&server.tcp, PEER_TIMEOUT).unwrap();
    let pools = peer
        .peer_pin("model".into(), vec![layout()], vec![key(1), key(2)])
        .unwrap();
    assert!(matches!(pools[0].slots[..], [Some(_), Some(_)]));
    assert_eq!(listed(&server.socket, pool).pinned, 2);

    drop(peer);
    await_listing(&server.socket, "the peer's pins ending", |listing| {
        listing
            .page_pools
            .iter()
            .any(|listed| listed.pool == pool && listed.pinned == 0 && listed.ready == 2)
    });
}

/// A head reserves slots its own transfer task writes (no writer scope).
/// When the head's connection closes while the task runs, the task is
/// cancelled, but its slots stay reserved until the task ended, since the
/// Transfer Server may write them until then; then they are aborted.
///
/// The task's source lies on a host that accepts the Transfer Server's
/// connection and never responds to, which holds the task until the test
/// closes that connection.
#[test]
fn a_disconnected_heads_task_reservations_end_after_its_task() {
    let reading = serving(Gpus::Hidden);
    let _forking = FORKING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut daemon = head(&reading.socket);
    let claim = daemon.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    daemon.publish(vec![1]).unwrap();
    let mut worker = epoch_worker(&reading.socket, scope("text", 0), 1, claim);
    let pool = trial_pool(&mut worker);
    let mut reader = engine_head(&reading.socket, 1);

    let Response::Reserved { slots } = reader
        .call(
            &Request::Reserve {
                pool,
                writer: None,
                keys: vec![key(1)],
                window: None,
            },
            &[],
        )
        .unwrap()
        .0
    else {
        panic!("the reservation was refused");
    };
    let [PageSlot::Reserved { slot }] = slots[..] else {
        panic!("no slot was reserved: {slots:?}");
    };

    // A source host that never responds to the Transfer Server's hello.
    let silent = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let source_host = silent.local_addr().unwrap().to_string();
    let task = reader
        .copy(
            vec![span(
                remote_region(&source_host, Region::Pool { pool: 1 }),
                0,
                Region::Pool { pool },
                slot * 16,
                16,
            )],
            Priority::Request,
            1,
        )
        .unwrap();
    let (mut held, _) = silent.accept().unwrap();

    drop(reader);
    await_listing(
        &reading.socket,
        "the head's task being cancelled",
        |listing| {
            listing.transfer.as_ref().is_some_and(|transfer| {
                transfer.tasks.iter().any(|t| t.task == task && t.cancelled)
            })
        },
    );
    assert_eq!(listed(&reading.socket, pool).reserved, 1);

    // The source host's connection closes: the task fails, and with it the
    // last task of the gone head ended.
    let mut hello = [0; 1];
    let _ = held.read(&mut hello);
    drop(held);
    await_listing(&reading.socket, "the head's task ending", |listing| {
        listing
            .transfer
            .as_ref()
            .is_some_and(|transfer| transfer.tasks.is_empty())
    });
    let after = listed(&reading.socket, pool);
    assert_eq!((after.reserved, after.ready), (0, 0));
}

/// A request's load overtakes a promotion's copies on their link.
#[test]
fn a_request_transfer_overtakes_a_promotion_on_its_link() {
    let server = serving(Gpus::Hidden);
    let Slow {
        mut worker,
        destination,
        bytes,
        ..
    } = slow(&server, 256 << 20);

    let promotion = worker
        .copy(
            vec![span(tensor_region(1), 0, tensor_region(2), 0, bytes - 4096)],
            Priority::Promotion,
            1,
        )
        .unwrap();
    let request = worker
        .copy(
            vec![span(
                tensor_region(1),
                0,
                tensor_region(2),
                bytes - 4096,
                4096,
            )],
            Priority::Request,
            1,
        )
        .unwrap();

    let mut ended = Vec::new();
    while ended.len() < 2 {
        if let Event::TaskEnded {
            task, status: end, ..
        } = transfer_event(&mut worker)
        {
            assert_eq!(end, TaskStatus::Complete);
            ended.push(task);
        }
    }
    assert_eq!(ended, [request, promotion]);
    assert_eq!(read(&destination, bytes - 4096, 4096), pattern(4096, 3));
}
