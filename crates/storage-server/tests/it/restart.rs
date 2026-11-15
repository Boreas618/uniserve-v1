//! Restarts in place across builds: a successor takes over the state of a
//! predecessor of the build before, whose image is of the format before its
//! own and whose sessions speak the version of the interface before its
//! own, and a server whose successor does not take it over goes on.

use std::os::fd::{AsFd as _, BorrowedFd, IntoRawFd as _, OwnedFd};
use std::os::unix::net::UnixStream;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use ::storage::protocol::{ClientMessage, Request as Call, Response, ServerMessage};
use ::storage::runtime::{Completion, OwnerId, Request, RequestId};
use ::storage::{Client, QueuePair, Role};
use storage_server::Server;

use crate::server::{FORKING, Running, key};
use crate::sessions::{DECODE, PREFILL, Scheduler, layout};

/// The takeover a server of the build before images stated their format
/// (image format 1, version 2 of the runtime interface) handed its
/// successor: one session of epoch 0 of `model`, whose head is connected,
/// with one scheduler of group 0 over the KV region `units` (number 0) and
/// the slot rows `rows` (number 1). Owner 1 holds the blocks at positions 0
/// and 1: the first committed and cached as key 7, the commit of the second
/// as key 8 (request 4) waiting for batch 2 of group 0, which has completed
/// batch 1. It names five descriptors: the listening socket, its lock, the
/// head's pidfd, the head's connection and the queue pair's memfd.
const TAKEOVER: &[u8] = include_bytes!("../fixtures/takeover-format-1.msgpack");

/// The registers of the takeover's queue pair: group 0's `completed` word,
/// then three capacity words for each of its two regions.
const REGISTERS: usize = 7;

/// A frame of the handoff socket, as its bytes.
struct Frame(Vec<u8>);

impl service::Message for Frame {
    fn encode(&self) -> std::io::Result<Vec<u8>> {
        Ok(self.0.clone())
    }

    fn decode(payload: &[u8]) -> std::io::Result<Self> {
        Ok(Self(payload.to_vec()))
    }
}

/// The next completion of `pair`, a record of `version`, within ten seconds.
fn completion(pair: &mut QueuePair, version: u32) -> (RequestId, Completion) {
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut record = Vec::new();
    loop {
        if pair.next_completion(&mut record).unwrap() {
            return Completion::decode_as(&record, version).unwrap();
        }
        assert!(Instant::now() < deadline, "no completion arrived");
        std::thread::sleep(Duration::from_micros(100));
    }
}

/// Submits `request` under `number` as a record of `version`.
fn submit(pair: &mut QueuePair, version: u32, number: u64, request: &Request) {
    let mut record = Vec::new();
    request.encode_as(version, RequestId(number), &mut record);
    assert!(pair.submit(&record).unwrap());
}

/// A successor of this build takes over a server of the build before:
/// it reads the image of format 1, goes on with the session in version 2,
/// whose waiting commit takes effect once its stamp is reached, and serves
/// the head's connection it was handed.
#[test]
fn a_successor_goes_on_from_an_image_of_the_format_before_its_own() {
    let directory = tempfile::tempdir().unwrap();
    let socket = directory.path().join("storage.sock");
    let (listener, lock) = service::bind(&socket).unwrap().unwrap();
    let pidfd = rustix::process::pidfd_open(
        rustix::process::getpid(),
        rustix::process::PidfdFlags::empty(),
    )
    .unwrap();
    let (head, handed) = UnixStream::pair().unwrap();
    let made = QueuePair::create(4 << 20, REGISTERS).unwrap();
    let memfd = made.descriptor().try_clone_to_owned().unwrap();
    let mut pair = QueuePair::open(made.descriptor().try_clone_to_owned().unwrap(), true).unwrap();
    drop(made);

    // The test plays the predecessor on the handoff socket.
    let (ours, theirs) = UnixStream::pair().unwrap();
    let descriptor = theirs.into_raw_fd();
    let successor = std::thread::spawn(move || Server::resume(descriptor, None, None).unwrap());
    let descriptors: Vec<OwnedFd> = vec![
        OwnedFd::from(listener),
        lock.hand_over(),
        pidfd,
        OwnedFd::from(handed),
        memfd,
    ];
    let borrowed: Vec<BorrowedFd<'_>> = descriptors.iter().map(|fd| fd.as_fd()).collect();
    let mut predecessor = service::Connection::new(ours).unwrap();
    predecessor
        .send_with(&Frame(TAKEOVER.to_vec()), &borrowed)
        .unwrap();
    predecessor.flush().unwrap();
    drop(descriptors);
    let server = Running::resumed(directory, socket, successor.join().unwrap());

    // Batch 2 ends: the commit taken before the restart takes effect, and
    // is answered in version 2.
    pair.register(0).store(2, Ordering::Release);
    assert_eq!(
        completion(&mut pair, 2),
        (RequestId(4), Completion::Committed)
    );

    // The session's records stay of version 2: a listing without host groups.
    submit(&mut pair, 2, 5, &Request::List);
    let (answered, listed) = completion(&mut pair, 2);
    assert_eq!(answered, RequestId(5));
    let Completion::Listing { mut keys } = listed else {
        panic!("{listed:?}");
    };
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            (key(7), 0, 64, 0b001, 0b01, 0),
            (key(8), 0, 64, 0b001, 0b01, 0)
        ]
    );
    let detach = Request::Detach {
        owner: OwnerId(1),
        after: Vec::new(),
    };
    submit(&mut pair, 2, 6, &detach);
    assert_eq!(completion(&mut pair, 2), (RequestId(6), Completion::Done));

    let mut operator = Client::connect(&server.socket, Role::Operator).unwrap();
    let pools = operator.list().unwrap().buffer_pools;
    let groups = pools.iter().flat_map(|pool| &pool.groups);
    assert_eq!(groups.map(|group| group.completed).collect::<Vec<_>>(), [2]);

    // The head's connection goes on: it closes its session.
    let mut head = service::Connection::new(head).unwrap();
    let close = ClientMessage::Request {
        id: 9,
        request: Call::Close,
    };
    head.send(&close).unwrap();
    let response = head
        .recv::<ServerMessage>(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert!(
        matches!(
            response,
            ServerMessage::Response {
                id: 9,
                response: Response::Closed
            }
        ),
        "{response:?}"
    );
    assert!(operator.list().unwrap().buffer_pools[0].groups.is_empty());
}

/// A restart in place whose successor ends without taking the server over,
/// as one of a build that states no versions does, is given up: the buffer
/// pools take requests again and the server goes on serving.
#[test]
fn a_server_whose_successor_ends_without_taking_it_over_goes_on() {
    let _forking = FORKING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let server = Running::with_transfer(|_| None);
    let mut engine = crate::server::engine_head(&server.socket, 0);
    let (opened, descriptors) = engine.open_pool(layout(0)).unwrap();
    let mut schedulers = Scheduler::all(&opened, descriptors);

    let mut operator = Client::connect(&server.socket, Role::Operator).unwrap();
    let successor = operator.restart(Some("/bin/false".into())).unwrap();
    assert!(successor > 0);

    // The pools serve again once the restart is given up.
    let attach = Request::Attach {
        owner: OwnerId(1),
        group: PREFILL,
    };
    assert_eq!(schedulers[0].ask(attach), Completion::Attached { slot: 1 });
    assert_eq!(operator.list().unwrap().buffer_pools.len(), 1);

    // The restart is no longer under way: another may begin, and is given
    // up alike.
    operator.restart(Some("/bin/false".into())).unwrap();
    let attach = Request::Attach {
        owner: OwnerId(2),
        group: DECODE,
    };
    assert_eq!(schedulers[1].ask(attach), Completion::Attached { slot: 1 });
}
