//! A Storage Server serving clients over its socket: publishing and mapping
//! tensors, write access, claims and shoot-down.
//!
//! Each test runs a server on a thread of the test process with a socket of
//! its own, and host storages (memfds), so no GPU is needed. A process shot
//! down is a child this test binary starts as a stand-in worker
//! (`stand_in_writer`), never the test process itself.

use std::io::{Read, Seek, Write};
use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ::storage::protocol::{ErrorKind, Item};
use ::storage::{
    Access, Claim, Client, Error, Key, Location, Mode, Role, Scope, StorageRecord, TensorRecord,
};
use storage_server::{Launch, Server};

/// Held by a test while it starts a child process, and by a test whose
/// server must see its dropped connections close: a child forked from this
/// process holds every connection open until it executes its program.
pub(crate) static FORKING: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A server on its own socket, stopped when dropped.
pub(crate) struct Running {
    pub(crate) socket: PathBuf,
    /// The loopback address other hosts' heads connect to.
    pub(crate) tcp: String,
    stop: Arc<AtomicBool>,
    wake: OwnedFd,
    thread: Option<std::thread::JoinHandle<()>>,
    _directory: tempfile::TempDir,
}

impl Running {
    fn start() -> Self {
        Self::with_transfer(|_| None)
    }

    /// A server that runs the Transfer Server `transfer` names for its
    /// socket.
    pub(crate) fn with_transfer(transfer: impl FnOnce(&Path) -> Option<Launch>) -> Self {
        Self::serving(0, transfer)
    }

    /// A server that leaves `host_reserve` bytes of the host's available
    /// storage to other processes.
    fn reserving(host_reserve: u64) -> Self {
        Self::serving(host_reserve, |_| None)
    }

    fn serving(host_reserve: u64, transfer: impl FnOnce(&Path) -> Option<Launch>) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("storage.sock");
        let transfer = transfer(&socket);
        let (listener, lock) = service::bind(&socket).unwrap().unwrap();
        let mut ends = [0; 2];
        // SAFETY: pipe2 fills the two descriptors it creates.
        let created = unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_NONBLOCK) };
        assert_eq!(created, 0);
        // SAFETY: two new descriptors this test owns.
        let (reader, writer) =
            unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) };
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let tcp = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = tcp.local_addr().unwrap().to_string();
        let thread = std::thread::spawn(move || {
            let _lock = lock;
            // Pools of four 16-byte pages.
            let mut server = Server::new(listener, 64, host_reserve, None, transfer).unwrap();
            server.listen_tcp(tcp).unwrap();
            server
                .run(reader.as_fd(), &|| stopped.load(Ordering::Acquire))
                .unwrap();
        });
        Self {
            socket,
            tcp: address,
            stop,
            wake: writer,
            thread: Some(thread),
            _directory: directory,
        }
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        // SAFETY: a write to a pipe this test owns.
        unsafe { libc::write(self.wake.as_raw_fd(), b"x".as_ptr().cast(), 1) };
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

pub(crate) fn scope(group: &str, rank: u32) -> Scope {
    Scope {
        deployment: "model".into(),
        group: group.into(),
        rank,
    }
}

pub(crate) fn key(byte: u8) -> Key {
    Key([byte; 32])
}

pub(crate) fn head(socket: &Path) -> Client {
    Client::connect(
        socket,
        Role::Head {
            deployment: "model".into(),
            epoch: 0,
        },
    )
    .unwrap()
}

pub(crate) fn worker(socket: &Path, scope: Scope, generation: u64, claim: Claim) -> Client {
    Client::connect(
        socket,
        Role::Worker {
            scope,
            generation,
            claim,
            epoch: 0,
        },
    )
    .unwrap()
}

/// A memfd holding `bytes`.
pub(crate) fn memfd(bytes: &[u8]) -> std::fs::File {
    // SAFETY: memfd_create with a static name.
    let fd = unsafe { libc::memfd_create(c"storage-test".as_ptr(), libc::MFD_CLOEXEC) };
    assert!(fd >= 0);
    // SAFETY: a new descriptor this test owns.
    let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
    file.write_all(bytes).unwrap();
    file
}

/// The bytes a received storage descriptor holds.
pub(crate) fn contents(descriptor: OwnedFd) -> Vec<u8> {
    let mut file = std::fs::File::from(descriptor);
    file.rewind().unwrap();
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).unwrap();
    bytes
}

pub(crate) fn tensor(
    key: Key,
    name: &str,
    storage: u32,
    offset: u64,
    access: Access,
) -> TensorRecord {
    TensorRecord {
        key,
        name: name.into(),
        storage,
        offset,
        dtype: "uint8".into(),
        shape: vec![4],
        strides: vec![1],
        access,
    }
}

pub(crate) fn host(bytes: u64) -> StorageRecord {
    StorageRecord {
        location: Location::Host { numa_node: None },
        bytes,
        segment: None,
        fabric: None,
    }
}

pub(crate) fn server_error(result: Result<impl std::fmt::Debug, Error>) -> ErrorKind {
    match result {
        Err(Error::Server { kind, .. }) => kind,
        other => panic!("expected a server error, got {other:?}"),
    }
}

#[test]
fn a_published_read_only_tensor_is_shared_with_other_workers_and_retained_by_scope() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head
        .claim(0, vec![scope("text", 0), scope("text", 1)], Vec::new())
        .unwrap();

    // Two tensors packed in one storage, published by rank 0.
    let mut first = worker(&server.socket, scope("text", 0), 1, claim);
    let storage = memfd(b"abcdefgh");
    first
        .put(
            vec![
                tensor(key(1), "a", 0, 0, Access::ReadOnly),
                tensor(key(2), "b", 0, 4, Access::ReadOnly),
            ],
            &[(host(8), storage.as_fd())],
        )
        .unwrap();
    drop(storage);

    // Rank 1 maps the same bytes; one storage travels for both tensors.
    let mut second = worker(&server.socket, scope("text", 1), 1, claim);
    let (got, descriptors) = second
        .get(vec![key(1), key(2), key(3)], Mode::Read, Vec::new())
        .unwrap();
    assert_eq!(got.missing, vec![key(3)]);
    assert_eq!(got.storages.len(), 1);
    assert_eq!(
        got.tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor.offset))
            .collect::<Vec<_>>(),
        vec![("a", 0), ("b", 4)]
    );
    let descriptor = descriptors.into_iter().next().unwrap();
    assert_eq!(contents(descriptor), b"abcdefgh");

    // The tensors live while some scope retains them.
    assert_eq!(first.retain(Vec::new(), Vec::new()).unwrap(), 0);
    assert_eq!(second.retain(vec![key(2)], Vec::new()).unwrap(), 1);
    let listed: Vec<String> = head
        .list()
        .unwrap()
        .tensors
        .into_iter()
        .map(|tensor| tensor.name)
        .collect();
    assert_eq!(listed, vec!["b".to_owned()]);
}

#[test]
fn putting_a_retained_read_only_key_again_keeps_the_retained_bytes_for_both_scopes() {
    let server = Running::start();
    let claim = head(&server.socket)
        .claim(0, vec![scope("text", 0), scope("text", 1)], Vec::new())
        .unwrap();
    let mut first = worker(&server.socket, scope("text", 0), 1, claim);
    let storage = memfd(b"abcd");
    first
        .put(
            vec![tensor(key(4), "weight", 0, 0, Access::ReadOnly)],
            &[(host(4), storage.as_fd())],
        )
        .unwrap();

    // A key names its content: a second publisher of the same key adds its
    // scope, and the server keeps the bytes it retains.
    let mut second = worker(&server.socket, scope("text", 1), 1, claim);
    let copy = memfd(b"abcd");
    second
        .put(
            vec![tensor(key(4), "weight", 0, 0, Access::ReadOnly)],
            &[(host(4), copy.as_fd())],
        )
        .unwrap();
    let listing = second.list().unwrap();
    assert_eq!(listing.tensors.len(), 1);
    assert_eq!(
        listing.tensors[0].scopes,
        vec![scope("text", 0), scope("text", 1)]
    );
    assert_eq!(
        listing
            .host_bytes
            .iter()
            .map(|entry| entry.bytes)
            .sum::<u64>(),
        8
    );

    // Exclusive tensors are created once.
    second
        .put(
            vec![tensor(key(5), "planes", 0, 0, Access::Exclusive)],
            &[(host(4), copy.as_fd())],
        )
        .unwrap();
    assert_eq!(
        server_error(first.put(
            vec![tensor(key(5), "planes", 0, 0, Access::Exclusive)],
            &[(host(4), storage.as_fd())],
        )),
        ErrorKind::Invalid
    );
}

#[test]
fn rekeying_moves_read_only_tensors_only_the_callers_deployment_holds() {
    let server = Running::start();
    // The head stays connected, so another deployment's claim keeps this
    // one.
    let mut model_head = head(&server.socket);
    let claim = model_head
        .claim(0, vec![scope("text", 0), scope("text", 1)], Vec::new())
        .unwrap();
    model_head.publish(vec![0]).unwrap();
    let mut first = worker(&server.socket, scope("text", 0), 1, claim);
    let storage = memfd(b"abcdefgh");
    first
        .put(
            vec![
                tensor(key(1), "a", 0, 0, Access::ReadOnly),
                tensor(key(2), "b", 0, 4, Access::ReadOnly),
                tensor(key(3), "planes", 0, 0, Access::Exclusive),
            ],
            &[(host(8), storage.as_fd())],
        )
        .unwrap();
    // Another rank of the deployment holds `a` too.
    let mut second = worker(&server.socket, scope("text", 1), 1, claim);
    second.get(vec![key(1)], Mode::Read, Vec::new()).unwrap();

    // Refused, moving none: an exclusive tensor, a retained new key, or one
    // new key twice.
    assert_eq!(
        server_error(first.rekey(vec![(key(1), key(11)), (key(3), key(13))], Vec::new())),
        ErrorKind::Invalid
    );
    assert_eq!(
        server_error(first.rekey(vec![(key(1), key(11)), (key(2), key(1))], Vec::new())),
        ErrorKind::Invalid
    );
    assert_eq!(
        server_error(first.rekey(vec![(key(1), key(11)), (key(2), key(11))], Vec::new())),
        ErrorKind::Invalid
    );
    assert_eq!(
        first
            .get(vec![key(1), key(2)], Mode::Read, Vec::new())
            .unwrap()
            .0
            .missing,
        vec![]
    );

    // The deployment's tensors move with their scopes; the old keys name
    // nothing more and the new ones the same bytes.
    first
        .rekey(vec![(key(1), key(11)), (key(2), key(12))], Vec::new())
        .unwrap();
    let (got, descriptors) = second
        .get(vec![key(1), key(11), key(12)], Mode::Read, Vec::new())
        .unwrap();
    assert_eq!(got.missing, vec![key(1)]);
    let descriptor = descriptors.into_iter().next().unwrap();
    assert_eq!(contents(descriptor), b"abcdefgh");
    let listing = first.list().unwrap();
    let a = listing
        .tensors
        .iter()
        .find(|tensor| tensor.name == "a")
        .unwrap();
    assert_eq!(a.scopes, vec![scope("text", 0), scope("text", 1)]);

    // A tensor another deployment holds as well is not the caller's to move.
    let other = scope_of("other");
    let other_claim = head_of(&server.socket, "other")
        .claim(0, vec![other.clone()], Vec::new())
        .unwrap();
    let mut stranger = worker(&server.socket, other, 1, other_claim);
    stranger.get(vec![key(12)], Mode::Read, Vec::new()).unwrap();
    assert_eq!(
        server_error(first.rekey(vec![(key(12), key(22))], Vec::new())),
        ErrorKind::Held
    );
}

#[test]
fn write_access_to_an_exclusive_tensor_passes_on_by_release() {
    let server = Running::start();
    let claim = head(&server.socket)
        .claim(0, vec![scope("flow", 0)], Vec::new())
        .unwrap();
    let mut writer = worker(&server.socket, scope("flow", 0), 1, claim);
    let storage = memfd(&[0; 4]);
    writer
        .put(
            vec![tensor(key(9), "planes", 0, 0, Access::Exclusive)],
            &[(host(4), storage.as_fd())],
        )
        .unwrap();

    // Exclusive tensors are not mapped for reading, and a second current
    // worker of the same launch does not take a live writer's access.
    let mut other = worker(&server.socket, scope("flow", 0), 1, claim);
    assert_eq!(
        server_error(other.get(vec![key(9)], Mode::Read, Vec::new())),
        ErrorKind::Invalid
    );
    assert_eq!(
        server_error(other.get(vec![key(9)], Mode::Write, Vec::new())),
        ErrorKind::Held
    );

    writer.release(vec![key(9)]).unwrap();
    let (got, _) = other.get(vec![key(9)], Mode::Write, Vec::new()).unwrap();
    assert_eq!(got.tensors.len(), 1);
    let pid = std::process::id() as i32;
    let listing = other.list().unwrap();
    assert_eq!(listing.tensors[0].writer, Some(pid));
}

#[test]
fn a_new_head_is_refused_while_the_previous_one_is_connected_and_drops_unlisted_scopes() {
    // The new claim shoots down every worker still connected, which the
    // workers below, of this process, must not be.
    let _forking = FORKING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let server = Running::start();
    let mut first = head(&server.socket);
    let claim = first
        .claim(0, vec![scope("text", 0), scope("old", 0)], Vec::new())
        .unwrap();
    for group in ["text", "old"] {
        let mut client = worker(&server.socket, scope(group, 0), 1, claim);
        let storage = memfd(&[1; 4]);
        let key = key(group.len() as u8);
        client
            .put(
                vec![tensor(key, group, 0, 0, Access::ReadOnly)],
                &[(host(4), storage.as_fd())],
            )
            .unwrap();
    }

    let mut second = head(&server.socket);
    assert_eq!(
        server_error(second.claim(0, Vec::new(), Vec::new())),
        ErrorKind::Claimed
    );
    drop(first);
    // The server notices the disconnect on its next turn.
    let claim = retry(|| second.claim(0, vec![scope("text", 0)], Vec::new()));
    let names: Vec<String> = second
        .list()
        .unwrap()
        .tensors
        .into_iter()
        .map(|tensor| tensor.name)
        .collect();
    assert_eq!(names, vec!["text".to_owned()]);

    // A worker of the previous head publishes nothing.
    let stale = Claim([0; 16]);
    assert_ne!(stale, claim);
    let mut client = worker(&server.socket, scope("text", 0), 1, stale);
    let storage = memfd(&[2; 4]);
    let result = client.put(
        vec![tensor(key(7), "late", 0, 0, Access::ReadOnly)],
        &[(host(4), storage.as_fd())],
    );
    assert_eq!(server_error(result), ErrorKind::Invalid);
}

/// The child process that holds write access, started by
/// `a_relaunched_worker_takes_write_access_by_shooting_down_its_predecessor`.
#[test]
#[ignore = "a stand-in worker this file's tests start as a child process"]
fn stand_in_writer() {
    let Ok(socket) = std::env::var("STORAGE_TEST_SOCKET") else {
        return;
    };
    let claim = Claim(
        std::env::var("STORAGE_TEST_CLAIM")
            .unwrap()
            .split(',')
            .map(|byte| byte.parse().unwrap())
            .collect::<Vec<u8>>()
            .try_into()
            .unwrap(),
    );
    let mut client = worker(Path::new(&socket), scope("flow", 0), 1, claim);
    let storage = memfd(&[3; 4]);
    client
        .put(
            vec![tensor(key(5), "latents", 0, 0, Access::Exclusive)],
            &[(host(4), storage.as_fd())],
        )
        .unwrap();
    println!("holding");
    std::thread::sleep(Duration::from_secs(600));
}

#[test]
fn a_relaunched_worker_takes_write_access_by_shooting_down_its_predecessor() {
    let server = Running::start();
    let claim = head(&server.socket)
        .claim(0, vec![scope("flow", 0)], Vec::new())
        .unwrap();
    let encoded: Vec<String> = claim.0.iter().map(u8::to_string).collect();
    let forking = FORKING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "server::stand_in_writer",
            "--ignored",
            "--nocapture",
        ])
        .env("STORAGE_TEST_SOCKET", &server.socket)
        .env("STORAGE_TEST_CLAIM", encoded.join(","))
        .stdout(std::process::Stdio::piped())
        // A worker leads its own process group, which a shoot-down kills.
        .process_group(0)
        .spawn()
        .unwrap();
    drop(forking);
    let mut output = std::io::BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    while !line.contains("holding") {
        line.clear();
        assert!(std::io::BufRead::read_line(&mut output, &mut line).unwrap() > 0);
    }

    // The next launch of the same scope takes the tensor: the server kills
    // the child, waits for it to exit and hands write access on.
    let mut relaunched = worker(&server.socket, scope("flow", 0), 2, claim);
    let (got, descriptors) = relaunched
        .get(vec![key(5)], Mode::Write, Vec::new())
        .unwrap();
    assert_eq!(got.tensors.len(), 1);
    let status = child.wait().unwrap();
    assert_eq!(
        std::os::unix::process::ExitStatusExt::signal(&status),
        Some(libc::SIGKILL)
    );
    let descriptor = descriptors.into_iter().next().unwrap();
    assert_eq!(contents(descriptor), [3; 4]);
}

/// The value `attempt` returns once it succeeds, retried briefly while the
/// server catches up with a disconnect.
fn retry<T>(mut attempt: impl FnMut() -> Result<T, Error>) -> T {
    for _ in 0..200 {
        match attempt() {
            Ok(value) => return value,
            Err(Error::Server {
                kind: ErrorKind::Claimed,
                ..
            }) => std::thread::sleep(Duration::from_millis(10)),
            Err(error) => panic!("{error}"),
        }
    }
    panic!("the server did not notice the disconnect");
}

#[test]
fn a_head_on_another_host_moves_pages_through_a_workers_pool() {
    use ::storage::protocol::{PageLevel, PageShape, PageSlot, Request, Response};

    let server = Running::start();
    let claim = head(&server.socket)
        .claim(0, vec![scope("text", 0)], Vec::new())
        .unwrap();

    // The worker maps its pool on the host.
    let mut worker = worker(&server.socket, scope("text", 0), 1, claim);
    let shape = PageShape {
        layout: key(3),
        kv_head_offset: 0,
        kv_head_count: 2,
        page_bytes: 16,
    };
    let (response, descriptors) = worker
        .call(
            &Request::Pool {
                shape,
                numa_node: None,
                bytes: None,
            },
            &[],
        )
        .unwrap();
    let Response::Pool { pool, slots, .. } = response else {
        panic!("a page pool request answered {response:?}");
    };
    assert_eq!((slots, descriptors.len()), (4, 1));

    // The head reaches the pool from another host.
    let mut remote = Client::connect_tcp(
        &server.tcp,
        Role::Head {
            deployment: "model".into(),
            epoch: 0,
        },
    )
    .unwrap();
    let mut ask = |request: Request| remote.call(&request, &[]).unwrap().0;
    let Response::Reserved { slots } = ask(Request::Reserve {
        pool,
        writer: Some(scope("text", 0)),
        keys: vec![key(1), key(2)],
        window: None,
    }) else {
        panic!("a reservation was refused");
    };
    assert!(matches!(
        slots[..],
        [PageSlot::Reserved { .. }, PageSlot::Reserved { .. }]
    ));
    ask(Request::Commit {
        pool,
        keys: vec![key(1)],
    });
    assert_eq!(
        ask(Request::Lookup {
            pool,
            keys: vec![key(1), key(2)],
        }),
        Response::Lookup {
            levels: vec![PageLevel::Host, PageLevel::Absent]
        }
    );
    // A connection from another host claims nothing.
    assert!(matches!(
        ask(Request::Claim {
            scopes: Vec::new(),
            epoch: 0,
            retire: Vec::new(),
        }),
        Response::Error {
            kind: ErrorKind::Invalid,
            ..
        }
    ));

    // A worker connects on its host only.
    assert!(matches!(
        Client::connect_tcp(
            &server.tcp,
            Role::Worker {
                scope: scope("text", 0),
                generation: 1,
                claim,
                epoch: 0,
            },
        ),
        Err(Error::Refused(_))
    ));
}

/// The next event `client` receives, waiting up to ten seconds.
pub(crate) fn next_event(client: &mut Client) -> (::storage::protocol::Event, Vec<OwnedFd>) {
    for _ in 0..1000 {
        client.pump().unwrap();
        if let Some(event) = client.next_event() {
            return event;
        }
        let mut descriptor = libc::pollfd {
            fd: client.as_fd().as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: polls one descriptor this test holds.
        unsafe { libc::poll(&mut descriptor, 1, 10) };
    }
    panic!("no event arrived");
}

#[test]
fn the_connector_extends_lookups_fills_prefetches_and_uploads_committed_pages() {
    use ::storage::protocol::{Event, PageLevel, PageShape, PageSlot, Request, Response};

    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head.claim(0, vec![scope("text", 0)], Vec::new()).unwrap();
    head.publish(vec![0]).unwrap();
    let mut worker = worker(&server.socket, scope("text", 0), 1, claim);
    let shape = PageShape {
        layout: key(3),
        kv_head_offset: 0,
        kv_head_count: 2,
        page_bytes: 16,
    };
    let Response::Pool { pool, .. } = worker
        .call(
            &Request::Pool {
                shape,
                numa_node: None,
                bytes: None,
            },
            &[],
        )
        .unwrap()
        .0
    else {
        panic!("the page pool was refused");
    };

    // The connector is given the pool with its storage.
    let mut connector = Client::connect(&server.socket, Role::Connector).unwrap();
    let (event, descriptors) = next_event(&mut connector);
    assert_eq!(
        event,
        Event::Pool {
            pool,
            shape,
            slots: 4
        }
    );
    assert_eq!(descriptors.len(), 1);
    // A second connector is refused.
    assert!(matches!(
        Client::connect(&server.socket, Role::Connector),
        Err(Error::Refused(_))
    ));

    // A committed page is uploaded from its slot.
    let Response::Reserved { slots } = head
        .call(
            &Request::Reserve {
                pool,
                writer: Some(scope("text", 0)),
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
        panic!("the reservation answered {slots:?}");
    };
    head.call(
        &Request::Commit {
            pool,
            keys: vec![key(1)],
        },
        &[],
    )
    .unwrap();
    let (
        Event::Upload {
            task,
            pool: uploaded,
            pages,
        },
        _,
    ) = next_event(&mut connector)
    else {
        panic!("the committed page was not uploaded");
    };
    assert_eq!((uploaded, pages), (pool, vec![(key(1), slot)]));
    let finished = Request::Finished {
        task,
        results: vec![true],
    };
    assert_eq!(
        connector.call(&finished, &[]).unwrap().0,
        Response::Finished
    );

    // A lookup asks the connector for the pages the pool does not hold.
    let lookup = head
        .submit(
            &Request::Lookup {
                pool,
                keys: vec![key(1), key(2), key(3)],
            },
            &[],
        )
        .unwrap();
    let (Event::Exists { task, keys, .. }, _) = next_event(&mut connector) else {
        panic!("the lookup did not reach the connector");
    };
    assert_eq!(keys, vec![key(2), key(3)]);
    connector
        .call(
            &Request::Finished {
                task,
                results: vec![true, false],
            },
            &[],
        )
        .unwrap();
    assert_eq!(
        head.wait(lookup).unwrap().0,
        Response::Lookup {
            levels: vec![PageLevel::Host, PageLevel::Storage, PageLevel::Absent]
        }
    );

    // A prefetch reads the pages the pool lacks into reserved slots, and its
    // event reports which pages the pool holds after.
    let prefetch = Request::Prefetch {
        pool,
        keys: vec![key(1), key(2), key(3)],
        tag: 7,
    };
    assert_eq!(head.call(&prefetch, &[]).unwrap().0, Response::Prefetching);
    let (Event::Download { task, pages, .. }, _) = next_event(&mut connector) else {
        panic!("the prefetch did not reach the connector");
    };
    assert_eq!(
        pages.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
        vec![key(2), key(3)]
    );
    connector
        .call(
            &Request::Finished {
                task,
                results: vec![true, false],
            },
            &[],
        )
        .unwrap();
    assert_eq!(
        next_event(&mut head).0,
        Event::Prefetched {
            tag: 7,
            results: vec![true, true, false]
        }
    );

    // A connector that leaves fails its lookups: L3 holds nothing beyond L2.
    let lookup = head
        .submit(
            &Request::Lookup {
                pool,
                keys: vec![key(1), key(2), key(3)],
            },
            &[],
        )
        .unwrap();
    next_event(&mut connector);
    drop(connector);
    assert_eq!(
        head.wait(lookup).unwrap().0,
        Response::Lookup {
            levels: vec![PageLevel::Host, PageLevel::Host, PageLevel::Absent]
        }
    );
}

/// A head of `deployment`.
fn head_of(socket: &Path, deployment: &str) -> Client {
    Client::connect(
        socket,
        Role::Head {
            deployment: deployment.into(),
            epoch: 0,
        },
    )
    .unwrap()
}

/// The first worker of group `text` of `deployment`.
fn scope_of(deployment: &str) -> Scope {
    Scope {
        deployment: deployment.into(),
        group: "text".into(),
        rank: 0,
    }
}

/// The page pool `worker` maps for pages of 16 bytes, asking for `bytes`:
/// its id and slots.
fn page_pool(worker: &mut Client, bytes: Option<u64>) -> Result<(u64, u64), Error> {
    use ::storage::protocol::{PageShape, Request, Response};

    let shape = PageShape {
        layout: key(3),
        kv_head_offset: 0,
        kv_head_count: 2,
        page_bytes: 16,
    };
    let (response, _) = worker.call(
        &Request::Pool {
            shape,
            numa_node: None,
            bytes,
        },
        &[],
    )?;
    match response {
        Response::Pool { pool, slots, .. } => Ok((pool, slots)),
        reply => panic!("a page pool request answered {reply:?}"),
    }
}

#[test]
fn page_pools_belong_to_their_deployment_and_leave_with_it() {
    use ::storage::protocol::{Request, Response};

    let server = Running::start();
    // The head stays connected, so another deployment's claim keeps this
    // one.
    let mut model_head = head_of(&server.socket, "model");
    let claim = model_head
        .claim(0, vec![scope_of("model")], Vec::new())
        .unwrap();
    let mut first = worker(&server.socket, scope_of("model"), 1, claim);
    // A worker sizes its pool: two pages of 16 bytes.
    let (pool, slots) = page_pool(&mut first, Some(32)).unwrap();
    assert_eq!(slots, 2);
    // The deployment's other workers of the same pages share it.
    assert_eq!(page_pool(&mut first, None).unwrap(), (pool, 2));

    // Another deployment's workers of the same pages have a pool of their
    // own, which its head alone reaches.
    let mut other_head = head_of(&server.socket, "other");
    let other_claim = other_head
        .claim(0, vec![scope_of("other")], Vec::new())
        .unwrap();
    let mut second = worker(&server.socket, scope_of("other"), 1, other_claim);
    let (other_pool, other_slots) = page_pool(&mut second, None).unwrap();
    assert_ne!(other_pool, pool);
    assert_eq!(other_slots, 4);
    let (lookup, _) = other_head
        .call(
            &Request::Lookup {
                pool,
                keys: vec![key(1)],
            },
            &[],
        )
        .unwrap();
    assert!(matches!(
        lookup,
        Response::Error {
            kind: ErrorKind::Invalid,
            ..
        }
    ));
    let owners: Vec<(u64, String)> = other_head
        .list()
        .unwrap()
        .page_pools
        .into_iter()
        .map(|listed| (listed.pool, listed.deployment))
        .collect();
    assert_eq!(
        owners,
        vec![(pool, "model".to_owned()), (other_pool, "other".to_owned())]
    );

    // Retiring the other deployment releases its pool.
    drop(second);
    drop(other_head);
    let mut operator = Client::connect(&server.socket, Role::Operator).unwrap();
    retry(|| operator.retire("other".into()));
    let pools: Vec<u64> = operator
        .list()
        .unwrap()
        .page_pools
        .into_iter()
        .map(|listed| listed.pool)
        .collect();
    assert_eq!(pools, vec![pool]);
}

#[test]
fn a_claim_retires_every_deployment_whose_head_is_gone() {
    let server = Running::start();
    // A deployment whose head leaves keeps a tensor and a page pool.
    let mut gone = head_of(&server.socket, "gone");
    let claim = gone.claim(0, vec![scope_of("gone")], Vec::new()).unwrap();
    let mut gone_worker = worker(&server.socket, scope_of("gone"), 1, claim);
    let storage = memfd(&[1; 4]);
    gone_worker
        .put(
            vec![tensor(key(1), "gone", 0, 0, Access::ReadOnly)],
            &[(host(4), storage.as_fd())],
        )
        .unwrap();
    page_pool(&mut gone_worker, None).unwrap();
    drop(gone_worker);
    drop(gone);
    // A deployment whose head stays keeps its pool.
    let mut kept = head_of(&server.socket, "kept");
    let kept_claim = kept.claim(0, vec![scope_of("kept")], Vec::new()).unwrap();
    let mut kept_worker = worker(&server.socket, scope_of("kept"), 1, kept_claim);
    let (kept_pool, _) = page_pool(&mut kept_worker, None).unwrap();

    // A new deployment's claim retires the one whose head left, once the
    // server has noticed it leave.
    let mut operator = Client::connect(&server.socket, Role::Operator).unwrap();
    for _ in 0..200 {
        let listing = operator.list().unwrap();
        if listing
            .deployments
            .iter()
            .any(|listed| listed.deployment == "gone" && !listed.head_connected)
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    head_of(&server.socket, "new")
        .claim(0, vec![scope_of("new")], Vec::new())
        .unwrap();
    let listing = operator.list().unwrap();
    assert!(listing.tensors.is_empty());
    let deployments: Vec<String> = listing
        .deployments
        .into_iter()
        .map(|listed| listed.deployment)
        .collect();
    assert!(!deployments.contains(&"gone".to_owned()));
    assert!(deployments.contains(&"kept".to_owned()));
    let pools: Vec<u64> = listing
        .page_pools
        .into_iter()
        .map(|listed| listed.pool)
        .collect();
    assert_eq!(pools, vec![kept_pool]);
}

#[test]
fn a_page_pool_the_hosts_storage_cannot_hold_is_refused() {
    use ::storage::protocol::{PageShape, Request};

    let server = Running::start();
    let claim = head(&server.socket)
        .claim(0, vec![scope("text", 0)], Vec::new())
        .unwrap();
    let mut worker = worker(&server.socket, scope("text", 0), 1, claim);
    // A million pages of a GiB: a pebibyte, beyond any host's storage.
    let shape = PageShape {
        layout: key(3),
        kv_head_offset: 0,
        kv_head_count: 2,
        page_bytes: 1 << 30,
    };
    let (refused, _) = worker
        .call(
            &Request::Pool {
                shape,
                numa_node: None,
                bytes: Some(1 << 50),
            },
            &[],
        )
        .unwrap();
    assert!(matches!(
        refused,
        ::storage::protocol::Response::Error {
            kind: ErrorKind::Resource,
            ..
        }
    ));
    let listing = worker.list().unwrap();
    assert!(listing.page_pools.is_empty());
}

/// A worker of `scope` in `epoch`.
pub(crate) fn epoch_worker(socket: &Path, scope: Scope, epoch: u64, claim: Claim) -> Client {
    Client::connect(
        socket,
        Role::Worker {
            scope,
            generation: 1,
            claim,
            epoch,
        },
    )
    .unwrap()
}

/// The names of the tensors the server retains, sorted.
fn retained_names(client: &mut Client) -> Vec<String> {
    let mut names: Vec<String> = client
        .list()
        .unwrap()
        .tensors
        .into_iter()
        .map(|tensor| tensor.name)
        .collect();
    names.sort();
    names
}

/// Two epochs of one deployment share a scope without touching each other's
/// storage: the head keeps one claim token for both, each epoch's `retain`
/// releases its own tensors alone, and a claim that retires an epoch releases
/// the tensors no other epoch retains.
#[test]
fn epochs_retain_their_own_storage_and_a_claim_retires_one() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let first_claim = head.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    head.publish(vec![1]).unwrap();

    // Epoch 1's worker publishes shared weights, its exclusive planes and a
    // buffer of its own.
    let mut first = epoch_worker(&server.socket, scope("text", 0), 1, first_claim);
    let storages = [memfd(&[1; 4]), memfd(&[2; 4]), memfd(&[3; 4])];
    first
        .put(
            vec![
                tensor(key(1), "weights", 0, 0, Access::ReadOnly),
                tensor(key(2), "planes", 1, 0, Access::Exclusive),
                tensor(key(3), "buffer", 2, 0, Access::Exclusive),
            ],
            &[
                (host(4), storages[0].as_fd()),
                (host(4), storages[1].as_fd()),
                (host(4), storages[2].as_fd()),
            ],
        )
        .unwrap();

    // Epoch 2 claims the same scope under the same token: epoch 1's worker
    // stays current.
    let second_claim = head.claim(2, vec![scope("text", 0)], Vec::new()).unwrap();
    assert_eq!(first_claim, second_claim);
    let mut second = epoch_worker(&server.socket, scope("text", 0), 2, second_claim);
    let (got, _) = second.get(vec![key(1)], Mode::Read, Vec::new()).unwrap();
    assert_eq!(got.tensors.len(), 1);
    let (got, _) = second.get(vec![key(2)], Mode::Standby, Vec::new()).unwrap();
    assert_eq!(got.tensors.len(), 1);
    let buffer = memfd(&[4; 4]);
    second
        .put(
            vec![tensor(key(4), "buffer", 0, 0, Access::Exclusive)],
            &[(host(4), buffer.as_fd())],
        )
        .unwrap();

    // Epoch 2's worker retains what it uses; epoch 1's buffer, which it
    // never took, stays epoch 1's.
    assert_eq!(
        second
            .retain(vec![key(1), key(2), key(4)], Vec::new())
            .unwrap(),
        0
    );
    assert_eq!(
        retained_names(&mut head),
        vec!["buffer", "buffer", "planes", "weights"]
    );
    let listing = head.list().unwrap();
    let planes = listing
        .tensors
        .iter()
        .find(|tensor| tensor.name == "planes")
        .unwrap();
    let epochs: Vec<u64> = planes
        .retainers
        .iter()
        .map(|retainer| retainer.epoch)
        .collect();
    assert_eq!(epochs, vec![1, 2]);

    // Epoch 1's worker keeps writing what it holds.
    first
        .retain(vec![key(1), key(2), key(3)], Vec::new())
        .unwrap();

    // The commit of epoch 2 retires epoch 1: its buffer goes, what epoch 2
    // retains stays.
    head.claim(2, vec![scope("text", 0)], vec![1]).unwrap();
    assert_eq!(
        retained_names(&mut head),
        vec!["buffer", "planes", "weights"]
    );
    let listing = head.list().unwrap();
    assert!(
        listing
            .tensors
            .iter()
            .all(|tensor| tensor.retainers.iter().all(|retainer| retainer.epoch == 2)),
        "{listing:?}"
    );
}

/// The engine of `epoch` of the deployment, which sends its page requests.
pub(crate) fn engine_head(socket: &Path, epoch: u64) -> Client {
    Client::connect(
        socket,
        Role::Head {
            deployment: "model".into(),
            epoch,
        },
    )
    .unwrap()
}

/// A page pool of four 16-byte pages that `worker` maps.
pub(crate) fn trial_pool(worker: &mut Client) -> u64 {
    use ::storage::protocol::{PageShape, Request, Response};

    let shape = PageShape {
        layout: key(3),
        kv_head_offset: 0,
        kv_head_count: 2,
        page_bytes: 16,
    };
    let (response, _) = worker
        .call(
            &Request::Pool {
                shape,
                numa_node: None,
                bytes: None,
            },
            &[],
        )
        .unwrap();
    let Response::Pool { pool, .. } = response else {
        panic!("the page pool was refused: {response:?}");
    };
    pool
}

/// `head` writes and commits the pages of `keys` in `pool`.
pub(crate) fn publish(head: &mut Client, pool: u64, keys: &[Key]) {
    use ::storage::protocol::{PageSlot, Request, Response};

    let Response::Reserved { slots } = head
        .call(
            &Request::Reserve {
                pool,
                writer: Some(scope("text", 0)),
                keys: keys.to_vec(),
                window: None,
            },
            &[],
        )
        .unwrap()
        .0
    else {
        panic!("the reservation was refused");
    };
    assert!(
        slots
            .iter()
            .all(|slot| matches!(slot, PageSlot::Reserved { .. })),
        "{slots:?}"
    );
    head.call(
        &Request::Commit {
            pool,
            keys: keys.to_vec(),
        },
        &[],
    )
    .unwrap();
}

/// The levels `head` finds the pages of `keys` at in `pool`, without a
/// connector.
fn levels(head: &mut Client, pool: u64, keys: &[Key]) -> Vec<::storage::protocol::PageLevel> {
    use ::storage::protocol::{Request, Response};

    let Response::Lookup { levels } = head
        .call(
            &Request::Lookup {
                pool,
                keys: keys.to_vec(),
            },
            &[],
        )
        .unwrap()
        .0
    else {
        panic!("the lookup was refused");
    };
    levels
}

/// An epoch on trial beside the epoch it succeeds keeps the pages it commits
/// to itself: the serving epoch neither reads them nor finds them in the KV
/// storage, which takes them only once the epoch is explicitly committed. The epoch on trial reads the serving epoch's pages.
#[test]
fn a_epoch_on_trial_publishes_its_pages_to_itself_until_its_commit() {
    use ::storage::protocol::{Event, PageLevel, Request};

    let server = Running::start();
    let mut daemon = head(&server.socket);
    let claim = daemon.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    daemon.publish(vec![1]).unwrap();
    let mut worker = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    let pool = trial_pool(&mut worker);
    let mut connector = Client::connect(&server.socket, Role::Connector).unwrap();
    assert!(matches!(next_event(&mut connector).0, Event::Pool { .. }));
    let finish = |connector: &mut Client, task: u64, results: Vec<bool>| {
        connector
            .call(&Request::Finished { task, results }, &[])
            .unwrap();
    };

    // The serving epoch's pages are every epoch's, and the KV storage
    // takes them at once.
    let mut serving = engine_head(&server.socket, 1);
    publish(&mut serving, pool, &[key(1)]);
    let (Event::Upload { task, pages, .. }, _) = next_event(&mut connector) else {
        panic!("the serving epoch's page was not uploaded");
    };
    assert_eq!(pages.len(), 1);
    assert_eq!(pages[0].0, key(1));
    finish(&mut connector, task, vec![true]);

    // Epoch 2 starts on trial and commits a page of its own.
    daemon.claim(2, vec![scope("text", 0)], Vec::new()).unwrap();
    let mut trial = engine_head(&server.socket, 2);
    publish(&mut trial, pool, &[key(2)]);
    assert_eq!(
        levels(&mut trial, pool, &[key(1), key(2)]),
        vec![PageLevel::Host, PageLevel::Host]
    );

    // The serving epoch asks the KV storage for the page it does not read:
    // the connector's next task is that lookup, so no upload preceded it.
    let lookup = serving
        .submit(
            &Request::Lookup {
                pool,
                keys: vec![key(1), key(2)],
            },
            &[],
        )
        .unwrap();
    let (Event::Exists { task, keys, .. }, _) = next_event(&mut connector) else {
        panic!("a page of the epoch on trial was uploaded before its commit");
    };
    assert_eq!(keys, vec![key(2)]);
    finish(&mut connector, task, vec![false]);
    assert_eq!(
        serving.wait(lookup).unwrap().0,
        ::storage::protocol::Response::Lookup {
            levels: vec![PageLevel::Host, PageLevel::Absent]
        }
    );
    // Nor does it pin it.
    let ::storage::protocol::Response::Pinned { slots } = serving
        .call(
            &Request::Pin {
                pool,
                reader: scope("text", 0),
                keys: vec![key(2)],
            },
            &[],
        )
        .unwrap()
        .0
    else {
        panic!("the pin was refused");
    };
    assert_eq!(slots, vec![None]);

    // The commit publishes epoch 2 before retiring epoch 1, and
    // the KV storage takes them.
    daemon.publish(vec![2]).unwrap();
    daemon.claim(2, vec![scope("text", 0)], vec![1]).unwrap();
    let (Event::Upload { task, pages, .. }, _) = next_event(&mut connector) else {
        panic!("the committed epoch's pages were not uploaded");
    };
    assert_eq!(
        pages.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
        vec![key(2)]
    );
    finish(&mut connector, task, vec![true]);
    let mut next = engine_head(&server.socket, 3);
    assert_eq!(
        levels(&mut next, pool, &[key(1), key(2)]),
        vec![PageLevel::Host, PageLevel::Host]
    );
}

/// Independent replicas share committed pages while each candidate stays
/// private. A group commit publishes all successors together, regardless
/// of the numerical order of the epoch identifiers.
#[test]
fn independent_replicas_publish_successors_as_a_group() {
    use ::storage::protocol::PageLevel::{Absent, Host};

    let server = Running::start();
    let mut daemon = head(&server.socket);
    let claim = daemon.claim(9, vec![scope("text", 0)], Vec::new()).unwrap();
    daemon.claim(3, vec![scope("text", 0)], Vec::new()).unwrap();
    daemon.publish(vec![9, 3]).unwrap();
    let mut worker = epoch_worker(&server.socket, scope("text", 0), 9, claim);
    let pool = trial_pool(&mut worker);
    let mut first = engine_head(&server.socket, 9);
    let mut second = engine_head(&server.socket, 3);
    publish(&mut first, pool, &[key(1)]);
    publish(&mut second, pool, &[key(2)]);
    assert_eq!(
        levels(&mut first, pool, &[key(1), key(2)]),
        vec![Host, Host]
    );
    assert_eq!(
        levels(&mut second, pool, &[key(1), key(2)]),
        vec![Host, Host]
    );

    daemon.claim(4, vec![scope("text", 0)], Vec::new()).unwrap();
    daemon.claim(8, vec![scope("text", 0)], Vec::new()).unwrap();
    let mut candidate = engine_head(&server.socket, 4);
    let mut peer = engine_head(&server.socket, 8);
    publish(&mut candidate, pool, &[key(3)]);
    publish(&mut peer, pool, &[key(4)]);
    assert_eq!(server_error(candidate.publish(vec![4])), ErrorKind::Invalid);
    assert_eq!(
        levels(&mut first, pool, &[key(3), key(4)]),
        vec![Absent, Absent]
    );
    assert_eq!(
        levels(&mut candidate, pool, &[key(3), key(4)]),
        vec![Host, Absent]
    );

    // A rejected set publishes none of its otherwise valid members.
    assert_eq!(
        server_error(daemon.publish(vec![4, 99])),
        ErrorKind::Invalid
    );
    assert_eq!(
        levels(&mut first, pool, &[key(3), key(4)]),
        vec![Absent, Absent]
    );
    daemon.publish(vec![4, 8]).unwrap();
    assert_eq!(
        levels(&mut first, pool, &[key(3), key(4)]),
        vec![Host, Host]
    );
    daemon.claim(4, vec![scope("text", 0)], vec![9, 3]).unwrap();
    assert_eq!(
        levels(&mut candidate, pool, &[key(1), key(2), key(3), key(4)]),
        vec![Host; 4]
    );
}

/// A rollback discards what the epoch on trial published: its pages leave
/// the pool, and the read-only tensors it alone put leave the server. Until
/// then the serving epoch reads none of them, and a tensor of the same key
/// that the serving epoch puts carries the serving epoch's bytes.
#[test]
fn a_rollback_discards_what_the_epoch_on_trial_published() {
    use ::storage::protocol::PageLevel;

    let server = Running::start();
    let mut daemon = head(&server.socket);
    let claim = daemon.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    daemon.publish(vec![1]).unwrap();
    let mut serving_worker = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    let pool = trial_pool(&mut serving_worker);
    daemon.claim(2, vec![scope("text", 0)], Vec::new()).unwrap();
    let mut trial_worker = epoch_worker(&server.socket, scope("text", 0), 2, claim);

    // Epoch 2 publishes a page and two read-only tensors.
    let mut trial = engine_head(&server.socket, 2);
    publish(&mut trial, pool, &[key(5)]);
    let trial_bytes = [memfd(b"bad!"), memfd(b"new!")];
    trial_worker
        .put(
            vec![
                tensor(key(6), "weights", 0, 0, Access::ReadOnly),
                tensor(key(7), "other", 1, 0, Access::ReadOnly),
            ],
            &[
                (host(4), trial_bytes[0].as_fd()),
                (host(4), trial_bytes[1].as_fd()),
            ],
        )
        .unwrap();

    // The serving epoch reads none of them and retains none.
    let mut serving = engine_head(&server.socket, 1);
    assert_eq!(
        levels(&mut serving, pool, &[key(5)]),
        vec![PageLevel::Absent]
    );
    let (got, _) = serving_worker
        .get(vec![key(6), key(7)], Mode::Read, Vec::new())
        .unwrap();
    assert!(got.tensors.is_empty());
    assert_eq!(got.missing, vec![key(6), key(7)]);
    serving_worker.retain(vec![key(6)], Vec::new()).unwrap();
    // A key it puts itself holds its own bytes, for every epoch.
    let serving_bytes = memfd(b"good");
    serving_worker
        .put(
            vec![tensor(key(6), "weights", 0, 0, Access::ReadOnly)],
            &[(host(4), serving_bytes.as_fd())],
        )
        .unwrap();
    let (_, mut descriptors) = trial_worker
        .get(vec![key(6)], Mode::Read, Vec::new())
        .unwrap();
    assert_eq!(contents(descriptors.remove(0)), b"good");

    // Epoch 2 reserves a page it commits only after the rollback.
    let ::storage::protocol::Response::Reserved { .. } = trial
        .call(
            &::storage::protocol::Request::Reserve {
                pool,
                writer: Some(scope("text", 0)),
                keys: vec![key(8)],
                window: None,
            },
            &[],
        )
        .unwrap()
        .0
    else {
        panic!("the reservation was refused");
    };

    // The rollback retires epoch 2. Neither its committed page nor the one
    // it commits as it leaves is read by any epoch, nor what it reserves
    // after.
    daemon.claim(1, vec![scope("text", 0)], vec![2]).unwrap();
    trial
        .call(
            &::storage::protocol::Request::Commit {
                pool,
                keys: vec![key(8)],
            },
            &[],
        )
        .unwrap();
    publish(&mut trial, pool, &[key(9)]);
    assert_eq!(
        levels(&mut serving, pool, &[key(5), key(8), key(9)]),
        vec![PageLevel::Absent; 3]
    );
    let mut next = engine_head(&server.socket, 3);
    assert_eq!(
        levels(&mut next, pool, &[key(5), key(8), key(9)]),
        vec![PageLevel::Absent; 3]
    );
    // The page committed before the rollback left the pool.
    let listing = daemon.list().unwrap();
    assert_eq!(listing.page_pools[0].ready, 2, "{listing:?}");
    assert_eq!(retained_names(&mut daemon), vec!["weights"]);
    let (_, mut descriptors) = serving_worker
        .get(vec![key(6)], Mode::Read, Vec::new())
        .unwrap();
    assert_eq!(contents(descriptors.remove(0)), b"good");
}

/// What the serving epoch published stays the serving epoch's to read when
/// only the epoch on trial still retains it: a previous epoch restarted on
/// its weights maps the cache segments it published again.
#[test]
fn a_epoch_reads_what_it_published_that_only_a_epoch_on_trial_retains() {
    let server = Running::start();
    let mut daemon = head(&server.socket);
    let claim = daemon.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    daemon.publish(vec![1]).unwrap();
    let mut serving = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    let planes = memfd(&[1; 4]);
    serving
        .put(
            vec![tensor(key(10), "planes", 0, 0, Access::Exclusive)],
            &[(host(4), planes.as_fd())],
        )
        .unwrap();
    daemon.claim(2, vec![scope("text", 0)], Vec::new()).unwrap();
    let mut trial = epoch_worker(&server.socket, scope("text", 0), 2, claim);
    trial.get(vec![key(10)], Mode::Standby, Vec::new()).unwrap();
    serving.retain(Vec::new(), Vec::new()).unwrap();
    drop(serving);

    let mut restarted = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    let (got, _) = restarted
        .get(vec![key(10)], Mode::Standby, Vec::new())
        .unwrap();
    assert_eq!(got.tensors.len(), 1, "{:?}", got.missing);
}

/// An epoch's ranks on a host hold one token count in its host KV cache:
/// what the epoch asks for, unless the host's budget over the bytes a token
/// takes on all of them holds fewer.
#[test]
fn a_host_sizes_the_host_kv_cache_of_a_epoch_in_tokens() {
    use ::storage::protocol::{Request, Response};

    let server = Running::start();
    let mut daemon = head(&server.socket);
    daemon.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    daemon.publish(vec![1]).unwrap();
    let mut engine = engine_head(&server.socket, 1);
    let mut ask = |tokens: u64, host_bytes: Option<u64>| {
        let (response, _) = engine
            .call(
                &Request::HostCache {
                    tokens,
                    token_bytes: vec![1024, 3072],
                    host_bytes,
                },
                &[],
            )
            .unwrap();
        let Response::HostCache { tokens } = response else {
            panic!("the host KV cache was not sized: {response:?}");
        };
        tokens
    };

    // A stated budget of 64 MiB holds 16384 tokens of 4 KiB over both ranks.
    assert_eq!(ask(1 << 20, Some(64 << 20)), 16384);
    assert_eq!(ask(4096, Some(64 << 20)), 4096);
    // The host's available storage holds a few thousand tokens.
    assert_eq!(ask(4096, None), 4096);
}

#[test]
fn host_storage_is_admitted_before_allocation_and_retires_with_its_owner() {
    for refused in [true, false] {
        let server = Running::reserving(if refused { u64::MAX } else { 0 });
        let mut head = head(&server.socket);
        let claim = head.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
        let mut worker = epoch_worker(&server.socket, scope("text", 0), 1, claim);
        let result = worker.allocate_host(key(42), "staged cache".into(), 4096);
        if refused {
            assert_eq!(server_error(result), ErrorKind::Resource);
            assert!(worker.list().unwrap().tensors.is_empty());
            continue;
        }
        let (got, descriptors) = result.unwrap();
        assert_eq!(got.tensors[0].shape, [4096]);
        let descriptor = descriptors.into_iter().next().unwrap();
        let mut file = std::fs::File::from(descriptor);
        file.write_all(b"cache bytes").unwrap();
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut bytes = [0; 11];
        file.read_exact(&mut bytes).unwrap();
        assert_eq!(&bytes, b"cache bytes");
        assert_eq!(worker.list().unwrap().tensors.len(), 1);
        drop(worker);
        assert_eq!(head.keep_weights(vec![1], vec![key(42)]).unwrap(), 0);
        let mut resumed = epoch_worker(&server.socket, scope("text", 0), 1, claim);
        let (_, descriptors) = resumed
            .get(vec![key(42)], Mode::Standby, Vec::new())
            .unwrap();
        let descriptor = descriptors.into_iter().next().unwrap();
        let mut restored = std::fs::File::from(descriptor);
        restored.seek(std::io::SeekFrom::Start(0)).unwrap();
        restored.read_exact(&mut bytes).unwrap();
        assert_eq!(&bytes, b"cache bytes");
        drop(resumed);
        assert_eq!(head.keep_weights(vec![1], Vec::new()).unwrap(), 1);
        assert!(head.list().unwrap().tensors.is_empty());
    }
}

/// A host that admits no page pool, its whole storage reserved, holds no
/// token of an epoch's host KV cache and refuses its pools. Once a KV
/// storage keeps what the pools commit (a connector connected), it holds
/// what a staging pool of `STAGING_POOL_BYTES` holds on each rank, and
/// admits such a pool from its reserve, though no larger one: pages pass
/// through it into the KV storage.
#[test]
fn a_host_that_admits_no_pool_stages_pages_for_the_kv_storage() {
    use ::storage::protocol::{PageShape, Request, Response};
    use storage_server::STAGING_POOL_BYTES;

    let server = Running::reserving(u64::MAX);
    let mut daemon = head(&server.socket);
    let claim = daemon.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    daemon.publish(vec![1]).unwrap();
    let mut engine = engine_head(&server.socket, 1);
    let mut ask = |tokens: u64, host_bytes: Option<u64>| {
        let (response, _) = engine
            .call(
                &Request::HostCache {
                    tokens,
                    token_bytes: vec![1024, 3072],
                    host_bytes,
                },
                &[],
            )
            .unwrap();
        let Response::HostCache { tokens } = response else {
            panic!("the host KV cache was not sized: {response:?}");
        };
        tokens
    };
    let mut worker = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    // The slots of a pool of pages of 16 bytes the worker asks for of
    // `bytes`, or `None` when the host refuses it.
    let mut pool = |bytes: u64| {
        let shape = PageShape {
            layout: key(3),
            kv_head_offset: 0,
            kv_head_count: 2,
            page_bytes: 16,
        };
        let request = Request::Pool {
            shape,
            numa_node: None,
            bytes: Some(bytes),
        };
        match worker.call(&request, &[]).unwrap().0 {
            Response::Pool { slots, .. } => Some(slots),
            Response::Error { .. } => None,
            reply => panic!("a page pool request answered {reply:?}"),
        }
    };

    assert_eq!(ask(1 << 20, None), 0);
    assert_eq!(pool(32), None);

    let _connector = Client::connect(&server.socket, Role::Connector).unwrap();
    // The larger rank takes 3072 bytes a token; a stated budget the host
    // cannot admit stages as well.
    let staging = STAGING_POOL_BYTES / 3072;
    assert_eq!(ask(1 << 20, None), staging);
    assert_eq!(ask(1 << 20, Some(64 << 30)), staging);
    assert_eq!(ask(64, None), 64);
    assert_eq!(pool(STAGING_POOL_BYTES + 16), None);
    assert_eq!(pool(STAGING_POOL_BYTES), Some(STAGING_POOL_BYTES / 16));
}

/// A server serves clients of its own protocol only and refuses any other version.
#[test]
fn a_server_serves_its_own_protocol_only() {
    use ::storage::protocol::VERSION;

    let server = Running::start();
    Client::connect(&server.socket, Role::Operator).unwrap();

    let welcome = |version: u32| {
        let mut connection = ::storage::transport::Connection::new(
            std::os::unix::net::UnixStream::connect(&server.socket).unwrap(),
        );
        connection
            .send(
                &::storage::protocol::ClientMessage::Hello(::storage::protocol::Hello {
                    version,
                    role: Role::Operator,
                }),
                &[],
            )
            .unwrap();
        connection
            .recv::<::storage::protocol::ServerMessage>()
            .unwrap()
            .0
    };
    assert!(matches!(
        welcome(VERSION),
        ::storage::protocol::ServerMessage::Welcome(::storage::protocol::Welcome::Accepted { .. })
    ));
    for other in [VERSION - 1, VERSION + 1] {
        assert!(matches!(
            welcome(other),
            ::storage::protocol::ServerMessage::Welcome(
                ::storage::protocol::Welcome::Refused { .. }
            )
        ));
    }
}

#[test]
fn one_key_keeps_a_copy_at_each_location() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head
        .claim(0, vec![scope("text", 0), scope("text", 1)], Vec::new())
        .unwrap();
    let near = Location::Host { numa_node: Some(0) };
    let far = Location::Host { numa_node: Some(1) };
    let on = |location| StorageRecord {
        location,
        bytes: 4,
        segment: None,
        fabric: None,
    };

    // Two ranks publish what one key names, each at its own location.
    let mut first = worker(&server.socket, scope("text", 0), 1, claim);
    let mut second = worker(&server.socket, scope("text", 1), 1, claim);
    let storage = memfd(b"aaaa");
    first
        .put(
            vec![tensor(key(1), "near", 0, 0, Access::ReadOnly)],
            &[(on(near), storage.as_fd())],
        )
        .unwrap();
    let storage = memfd(b"bbbb");
    second
        .put(
            vec![tensor(key(1), "far", 0, 0, Access::ReadOnly)],
            &[(on(far), storage.as_fd())],
        )
        .unwrap();

    // A get selects the copy where the caller needs it, or every copy.
    let (got, descriptors) = second.get(vec![key(1)], Mode::Read, vec![far]).unwrap();
    assert_eq!(got.storages.len(), 1);
    let (record, descriptor) = got.storages.into_iter().zip(descriptors).next().unwrap();
    assert_eq!(record.location, far);
    assert_eq!(contents(descriptor), b"bbbb");
    let (every, _) = second.get(vec![key(1)], Mode::Read, Vec::new()).unwrap();
    assert_eq!(every.storages.len(), 2);
    let elsewhere = Location::Host { numa_node: Some(2) };
    assert_eq!(
        second
            .get(vec![key(1)], Mode::Read, vec![elsewhere])
            .unwrap()
            .0
            .missing,
        vec![key(1)]
    );

    // A put of the same key at a location with a copy shares that copy.
    let storage = memfd(b"cccc");
    first
        .put(
            vec![tensor(key(1), "again", 0, 0, Access::ReadOnly)],
            &[(on(far), storage.as_fd())],
        )
        .unwrap();
    let (_, descriptors) = first.get(vec![key(1)], Mode::Read, vec![far]).unwrap();
    let descriptor = descriptors.into_iter().next().unwrap();
    assert_eq!(contents(descriptor), b"bbbb");

    // Rekeying moves the selected copy only.
    first.rekey(vec![(key(1), key(2))], vec![near]).unwrap();
    let (got, _) = first
        .get(vec![key(1), key(2)], Mode::Read, Vec::new())
        .unwrap();
    let mut held: Vec<(Key, Location)> = got
        .tensors
        .iter()
        .map(|tensor| (tensor.key, got.storages[tensor.storage as usize].location))
        .collect();
    held.sort_by_key(|(key, _)| *key);
    assert_eq!(held, vec![(key(1), far), (key(2), near)]);

    // Each copy lives while some scope retains it at its location.
    second.retain(Vec::new(), Vec::new()).unwrap();
    first.retain(vec![key(2)], vec![near]).unwrap();
    let listed: Vec<(Key, Location)> = head
        .list()
        .unwrap()
        .tensors
        .into_iter()
        .map(|tensor| (tensor.key, tensor.location))
        .collect();
    assert_eq!(listed, vec![(key(2), near)]);
}

/// The GPU the quota tests reserve on, which no process touches: the server
/// never interprets a storage, so a memfd stands for device storage.
pub(crate) const GPU: [u8; 16] = [0x5a; 16];

/// A storage of `bytes` on [`GPU`], of KV cache `segment` when given.
pub(crate) fn device(bytes: u64, segment: Option<::storage::Segment>) -> StorageRecord {
    StorageRecord {
        location: Location::Device { gpu: GPU },
        bytes,
        segment,
        fabric: None,
    }
}

/// Segment `ordinal` of `epoch`.
fn segment(epoch: u64, ordinal: u32) -> Option<::storage::Segment> {
    Some(::storage::Segment { epoch, ordinal })
}

/// The deployment's quota of [`GPU`] as `client` reads it.
pub(crate) fn quota_of(client: &mut Client) -> ::storage::Quota {
    let quotas = client.usage(vec![GPU]).unwrap().quotas;
    assert_eq!(quotas.len(), 1, "{quotas:?}");
    quotas.into_iter().next().unwrap()
}

/// The bytes of `quota`'s reservations of `item` held by exactly `epochs`.
pub(crate) fn reserved(quota: &::storage::Quota, item: Item, epochs: &[u64]) -> u64 {
    quota.bytes(|reservation| reservation.item == item && reservation.epochs == epochs)
}

/// Waits until `client` reads a usage of `usage` bytes: a
/// connection's end reaches the server a moment after the client drops it.
fn await_usage(client: &mut Client, usage: u64) {
    let mut reserved = 0;
    for _ in 0..200 {
        reserved = quota_of(client).usage();
        if reserved == usage {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("the usage is {reserved} bytes, not {usage}");
}

/// Two epochs of a deployment share a GPU within one quota: each storage is
/// reserved once, however many epochs map it, an epoch's reserved items end with
/// its processes, nothing takes a GPU past its quota, and a commit
/// hands the predecessor's segments the successor maps to the successor on
/// the quota, as a transfer rather than a release and an allocation.
#[test]
fn two_epochs_share_a_gpus_quota_and_a_commit_transfers_the_segments() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    head.publish(vec![1]).unwrap();

    // Epoch 1 states its quota of 700 bytes of the GPU's 1000 and reserves its
    // process and graph pool, then publishes its weights, a KV cache
    // segment and its buffer.
    let mut first = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    first
        .reserve(
            GPU,
            1000,
            Some(700),
            vec![(Item::Process, 100), (Item::GraphPool, 100)],
        )
        .unwrap();
    let storages = [memfd(&[1; 4]), memfd(&[2; 4]), memfd(&[3; 4])];
    first
        .put(
            vec![
                tensor(key(1), "weights", 0, 0, Access::ReadOnly),
                tensor(key(2), "values#0/0", 1, 0, Access::Exclusive),
                tensor(key(3), "buffer", 2, 0, Access::Exclusive),
            ],
            &[
                (device(200, None), storages[0].as_fd()),
                (device(50, segment(1, 0)), storages[1].as_fd()),
                (device(20, None), storages[2].as_fd()),
            ],
        )
        .unwrap();
    let quota = quota_of(&mut head);
    assert_eq!(
        (quota.limit, quota.total, quota.usage()),
        (Some(700), 1000, 470)
    );
    assert_eq!(reserved(&quota, Item::CacheSegment, &[1]), 50);
    assert_eq!(reserved(&quota, Item::Exclusive, &[1]), 20);

    // Epoch 2 maps the weights and the segment: each is reserved once, held
    // by both epochs, beside the process and graph pool epoch 2 reserves.
    head.claim(2, vec![scope("text", 0)], Vec::new()).unwrap();
    let mut second = epoch_worker(&server.socket, scope("text", 0), 2, claim);
    second.get(vec![key(1)], Mode::Read, Vec::new()).unwrap();
    second.get(vec![key(2)], Mode::Standby, Vec::new()).unwrap();
    let quota = second
        .reserve(
            GPU,
            1000,
            Some(700),
            vec![(Item::Process, 100), (Item::GraphPool, 100)],
        )
        .unwrap()
        .remove(0);
    assert_eq!(quota.usage(), 670);
    assert_eq!(reserved(&quota, Item::Weights, &[1, 2]), 200);
    assert_eq!(reserved(&quota, Item::CacheSegment, &[1, 2]), 50);

    // The successor's own segment takes the quota to its quota; one byte
    // more is refused, naming every entry of the quota.
    let own = memfd(&[4; 4]);
    second
        .put(
            vec![tensor(key(4), "values#0/0", 0, 0, Access::Exclusive)],
            &[(device(30, segment(2, 0)), own.as_fd())],
        )
        .unwrap();
    let more = memfd(&[5; 4]);
    let refused = second.put(
        vec![tensor(key(5), "values#0/1", 0, 0, Access::Exclusive)],
        &[(device(1, segment(2, 1)), more.as_fd())],
    );
    match refused {
        Err(Error::Server {
            kind: ErrorKind::Quota,
            message,
        }) => {
            assert!(message.contains("past its quota of 700 bytes"), "{message}");
            assert!(
                message.contains("graph_pool of model/text/0 of epoch 1: 100 bytes"),
                "{message}"
            );
            assert!(
                message.contains("cache_segment 1/0 of model/text/0 of epoch 1 and 2: 50 bytes"),
                "{message}"
            );
        }
        other => panic!("expected a refusal past the quota, got {other:?}"),
    }
    assert_eq!(quota_of(&mut head).usage(), 700);

    // The commit: epoch 1's processes end in order, withdrawing what they
    // reserved, and the claim retires epoch 1, which releases its buffer. The
    // segment epoch 2 maps is epoch 2's alone, its bytes unchanged.
    first
        .reserve(
            GPU,
            1000,
            Some(700),
            vec![(Item::Process, 0), (Item::GraphPool, 0)],
        )
        .unwrap();
    drop(first);
    await_usage(&mut head, 500);
    head.claim(2, vec![scope("text", 0)], vec![1]).unwrap();
    let quota = quota_of(&mut head);
    assert_eq!(quota.usage(), 480);
    assert_eq!(reserved(&quota, Item::CacheSegment, &[2]), 80);
    assert_eq!(reserved(&quota, Item::Weights, &[2]), 200);
    assert_eq!(reserved(&quota, Item::Exclusive, &[1]), 0);
    assert_eq!(quota.limit, Some(700));
}

/// A previous epoch kept as its weights keeps nothing else once its
/// processes exited: what they reserved ends with them, and the head ends its
/// retention of every exclusive storage, so its buffer is released and the
/// segment its successor maps is the successor's alone, while the weights
/// both map stay reserved once for both. The successor's room on the quota
/// grows by all of it.
#[test]
fn a_epoch_kept_as_its_weights_leaves_its_successor_the_rest() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    head.publish(vec![1]).unwrap();
    let mut first = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    first
        .reserve(GPU, 1000, Some(700), vec![(Item::Process, 100)])
        .unwrap();
    let storages = [memfd(&[1; 4]), memfd(&[2; 4]), memfd(&[3; 4])];
    first
        .put(
            vec![
                tensor(key(1), "weights", 0, 0, Access::ReadOnly),
                tensor(key(2), "values#0/0", 1, 0, Access::Exclusive),
                tensor(key(3), "buffer", 2, 0, Access::Exclusive),
            ],
            &[
                (device(200, None), storages[0].as_fd()),
                (device(50, segment(1, 0)), storages[1].as_fd()),
                (device(20, None), storages[2].as_fd()),
            ],
        )
        .unwrap();
    head.claim(2, vec![scope("text", 0)], Vec::new()).unwrap();
    let mut second = epoch_worker(&server.socket, scope("text", 0), 2, claim);
    second.get(vec![key(1)], Mode::Read, Vec::new()).unwrap();
    second.get(vec![key(2)], Mode::Standby, Vec::new()).unwrap();
    assert_eq!(quota_of(&mut head).usage(), 370);

    // The previous epoch's processes exit; the head keeps its weights.
    // The kept epoch's processes end in order, withdrawing what they reserved.
    first
        .reserve(GPU, 1000, Some(700), vec![(Item::Process, 0)])
        .unwrap();
    drop(first);
    await_usage(&mut head, 270);
    assert_eq!(head.keep_weights(vec![1], Vec::new()).unwrap(), 1);
    let quota = quota_of(&mut head);
    assert_eq!(quota.usage(), 250);
    assert_eq!(reserved(&quota, Item::Weights, &[1, 2]), 200);
    assert_eq!(reserved(&quota, Item::CacheSegment, &[2]), 50);
    assert_eq!(reserved(&quota, Item::Exclusive, &[1]), 0);
    // A worker is no head.
    let (refused, _) = second
        .call(
            &::storage::protocol::Request::KeepWeights {
                epochs: vec![1],
                cache_keys: Vec::new(),
            },
            &[],
        )
        .unwrap();
    assert!(
        matches!(
            refused,
            ::storage::protocol::Response::Error {
                kind: ErrorKind::Invalid,
                ..
            }
        ),
        "{:?}",
        refused
    );
}

/// A head moves an epoch's weights between its GPUs and host storage only
/// through the Transfer Server: without one, an offload of weights is
/// refused and moves nothing, while an epoch with no weights to move (none
/// that it alone retains on a GPU, none in host storage) is answered at once.
/// Only the head moves weights.
#[test]
fn a_weight_move_without_a_transfer_server_is_refused_and_moves_nothing() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    head.publish(vec![1]).unwrap();
    let mut first = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    let storages = [memfd(&[1; 4]), memfd(&[2; 4])];
    first
        .put(
            vec![
                tensor(key(1), "weights", 0, 0, Access::ReadOnly),
                tensor(key(2), "values#0/0", 1, 0, Access::Exclusive),
            ],
            &[
                (device(200, None), storages[0].as_fd()),
                (device(50, segment(1, 0)), storages[1].as_fd()),
            ],
        )
        .unwrap();

    assert_eq!(server_error(head.offload(1)), ErrorKind::Unavailable);
    assert_eq!(reserved(&quota_of(&mut head), Item::Weights, &[1]), 200);
    // Epoch 2 retains nothing; epoch 1 has nothing in host storage.
    assert_eq!(head.offload(2).unwrap(), (0, 0));
    assert_eq!(head.restore(1).unwrap(), (0, 0));
    // A worker is no head.
    let (refused, _) = first
        .call(&::storage::protocol::Request::Offload { epoch: 1 }, &[])
        .unwrap();
    assert!(
        matches!(
            refused,
            ::storage::protocol::Response::Error {
                kind: ErrorKind::Invalid,
                ..
            }
        ),
        "{:?}",
        refused
    );
}

/// The processes of one scope and epoch serve in turn: a standby or a
/// relaunch that reserves beside the rank it replaces counts once, as the
/// larger reservation.
#[test]
fn the_processes_of_one_scope_count_as_their_largest_booking() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    head.publish(vec![1]).unwrap();
    let mut rank = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    rank.reserve(GPU, 1000, Some(700), vec![(Item::Process, 300)])
        .unwrap();
    let mut standby = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    standby
        .reserve(GPU, 1000, Some(700), vec![(Item::Process, 350)])
        .unwrap();
    assert_eq!(quota_of(&mut head).usage(), 350);
    drop(standby);
    await_usage(&mut head, 300);
}

/// A lost rank's reservation stays reserved for the process that replaces it:
/// another scope cannot reserve into it, and the relaunch takes each item over
/// as it reserves it.
#[test]
fn a_lost_rank_s_booking_stays_reserved_for_its_relaunch() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head
        .claim(1, vec![scope("text", 0), scope("text", 1)], Vec::new())
        .unwrap();
    head.publish(vec![1]).unwrap();
    let mut rank = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    rank.reserve(
        GPU,
        1000,
        Some(700),
        vec![(Item::Process, 300), (Item::GraphPool, 200)],
    )
    .unwrap();
    drop(rank);

    // The lost rank's 500 bytes leave 200 of the quota to another scope.
    let mut other = epoch_worker(&server.socket, scope("text", 1), 1, claim);
    assert_eq!(
        server_error(other.reserve(GPU, 1000, Some(700), vec![(Item::Process, 300)])),
        ErrorKind::Quota
    );
    assert_eq!(quota_of(&mut head).usage(), 500);

    // The relaunch reserves a smaller process and takes that item over; the
    // graph pool it has not reserved yet stays reserved.
    let mut relaunch = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    relaunch
        .reserve(GPU, 1000, Some(700), vec![(Item::Process, 250)])
        .unwrap();
    assert_eq!(quota_of(&mut head).usage(), 450);
    relaunch
        .reserve(GPU, 1000, Some(700), vec![(Item::GraphPool, 150)])
        .unwrap();
    assert_eq!(quota_of(&mut head).usage(), 400);
}

/// A rank that withdraws its reservations before its connection ends, as a
/// worker that ends in order does, leaves nothing reserved, and a lost
/// rank's reservation ends with its epoch's retirement.
#[test]
fn reservations_end_with_a_withdrawal_or_the_epoch_s_retirement() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head
        .claim(1, vec![scope("text", 0), scope("text", 1)], Vec::new())
        .unwrap();
    head.publish(vec![1]).unwrap();
    let mut closed = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    closed
        .reserve(GPU, 1000, Some(700), vec![(Item::Process, 300)])
        .unwrap();
    closed
        .reserve(GPU, 1000, Some(700), vec![(Item::Process, 0)])
        .unwrap();
    drop(closed);
    await_usage(&mut head, 0);

    let mut lost = epoch_worker(&server.socket, scope("text", 1), 1, claim);
    lost.reserve(GPU, 1000, Some(700), vec![(Item::Process, 300)])
        .unwrap();
    drop(lost);
    assert_eq!(quota_of(&mut head).usage(), 300);
    head.claim(2, vec![scope("text", 0)], Vec::new()).unwrap();
    head.publish(vec![2]).unwrap();
    head.claim(2, vec![scope("text", 0)], vec![1]).unwrap();
    // Nothing is reserved on the GPU any more: it has no quota, or an empty
    // one.
    let quotas = head.usage(vec![GPU]).unwrap().quotas;
    assert_eq!(quotas.iter().map(::storage::Quota::usage).sum::<u64>(), 0);
}

/// A worker reserves only its own items, and a reservation that leaves a quota
/// past its quota (as a smaller fraction an epoch states since may) shrinks
/// it but never grows it.
#[test]
fn a_ledger_past_its_quota_shrinks_but_never_grows() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    head.publish(vec![1]).unwrap();
    let mut worker = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    worker
        .reserve(GPU, 1000, Some(700), vec![(Item::Process, 600)])
        .unwrap();
    for item in [Item::TransferContext, Item::Weights, Item::CacheSegment] {
        assert_eq!(
            server_error(worker.reserve(GPU, 1000, None, vec![(item, 1)])),
            ErrorKind::Invalid
        );
    }

    worker
        .reserve(GPU, 1000, Some(500), vec![(Item::Process, 550)])
        .unwrap();
    assert_eq!(
        server_error(worker.reserve(GPU, 1000, None, vec![(Item::Process, 560)])),
        ErrorKind::Quota
    );
    let quota = quota_of(&mut worker);
    assert_eq!((quota.limit, quota.usage()), (Some(500), 550));
}
