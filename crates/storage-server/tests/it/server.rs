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

use storage::protocol::ErrorKind;
use storage::{
    Access, Claim, Client, Error, Key, Mode, Placement, Role, Scope, StorageRecord, TensorRecord,
    socket,
};
use storage_server::Server;

/// Held by a test while it starts a child process, and by a test whose
/// server must see its dropped connections close: a child forked from this
/// process holds every connection open until it executes its program.
static FORKING: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A server on its own socket, stopped when dropped.
struct Running {
    socket: PathBuf,
    /// The loopback address other hosts' heads connect to.
    tcp: String,
    stop: Arc<AtomicBool>,
    wake: OwnedFd,
    thread: Option<std::thread::JoinHandle<()>>,
    _directory: tempfile::TempDir,
}

impl Running {
    fn start() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("storage.sock");
        let (listener, lock) = socket::bind(&socket).unwrap().unwrap();
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
            let mut server = Server::new(listener, 64, 0, None).unwrap();
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

fn scope(group: &str, rank: u32) -> Scope {
    Scope {
        deployment: "model".into(),
        group: group.into(),
        rank,
    }
}

fn key(byte: u8) -> Key {
    Key([byte; 32])
}

fn head(socket: &Path) -> Client {
    Client::connect(
        socket,
        Role::Head {
            deployment: "model".into(),
        },
    )
    .unwrap()
}

fn worker(socket: &Path, scope: Scope, generation: u64, claim: Claim) -> Client {
    Client::connect(
        socket,
        Role::Worker {
            scope,
            generation,
            claim,
        },
    )
    .unwrap()
}

/// A memfd holding `bytes`.
fn memfd(bytes: &[u8]) -> std::fs::File {
    // SAFETY: memfd_create with a static name.
    let fd = unsafe { libc::memfd_create(c"storage-test".as_ptr(), libc::MFD_CLOEXEC) };
    assert!(fd >= 0);
    // SAFETY: a new descriptor this test owns.
    let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
    file.write_all(bytes).unwrap();
    file
}

/// The bytes a received storage descriptor holds.
fn contents(descriptor: OwnedFd) -> Vec<u8> {
    let mut file = std::fs::File::from(descriptor);
    file.rewind().unwrap();
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).unwrap();
    bytes
}

fn tensor(key: Key, name: &str, storage: u32, offset: u64, access: Access) -> TensorRecord {
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

fn host(bytes: u64) -> StorageRecord {
    StorageRecord {
        placement: Placement::Host { numa_node: None },
        bytes,
    }
}

fn server_error(result: Result<impl std::fmt::Debug, Error>) -> ErrorKind {
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
        .claim(vec![scope("text", 0), scope("text", 1)])
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
    let got = second
        .get(vec![key(1), key(2), key(3)], Mode::Read)
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
    let (_, descriptor) = got.storages.into_iter().next().unwrap();
    assert_eq!(contents(descriptor), b"abcdefgh");

    // The tensors live while some scope retains them.
    assert_eq!(first.retain(Vec::new()).unwrap(), 0);
    assert_eq!(second.retain(vec![key(2)]).unwrap(), 1);
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
        .claim(vec![scope("text", 0), scope("text", 1)])
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
        .claim(vec![scope("text", 0), scope("text", 1)])
        .unwrap();
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
    second.get(vec![key(1)], Mode::Read).unwrap();

    // Refused, moving none: an exclusive tensor, a retained new key, or one
    // new key twice.
    assert_eq!(
        server_error(first.rekey(vec![(key(1), key(11)), (key(3), key(13))])),
        ErrorKind::Invalid
    );
    assert_eq!(
        server_error(first.rekey(vec![(key(1), key(11)), (key(2), key(1))])),
        ErrorKind::Invalid
    );
    assert_eq!(
        server_error(first.rekey(vec![(key(1), key(11)), (key(2), key(11))])),
        ErrorKind::Invalid
    );
    assert_eq!(
        first.get(vec![key(1), key(2)], Mode::Read).unwrap().missing,
        vec![]
    );

    // The deployment's tensors move with their scopes; the old keys name
    // nothing more and the new ones the same bytes.
    first
        .rekey(vec![(key(1), key(11)), (key(2), key(12))])
        .unwrap();
    let got = second
        .get(vec![key(1), key(11), key(12)], Mode::Read)
        .unwrap();
    assert_eq!(got.missing, vec![key(1)]);
    let (_, descriptor) = got.storages.into_iter().next().unwrap();
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
        .claim(vec![other.clone()])
        .unwrap();
    let mut stranger = worker(&server.socket, other, 1, other_claim);
    stranger.get(vec![key(12)], Mode::Read).unwrap();
    assert_eq!(
        server_error(first.rekey(vec![(key(12), key(22))])),
        ErrorKind::Held
    );
}

#[test]
fn write_access_to_an_exclusive_tensor_passes_on_by_release() {
    let server = Running::start();
    let claim = head(&server.socket).claim(vec![scope("flow", 0)]).unwrap();
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
        server_error(other.get(vec![key(9)], Mode::Read)),
        ErrorKind::Invalid
    );
    assert_eq!(
        server_error(other.get(vec![key(9)], Mode::Write)),
        ErrorKind::Held
    );

    writer.release(vec![key(9)]).unwrap();
    let got = other.get(vec![key(9)], Mode::Write).unwrap();
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
        .claim(vec![scope("text", 0), scope("old", 0)])
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
    assert_eq!(server_error(second.claim(Vec::new())), ErrorKind::Claimed);
    drop(first);
    // The server notices the disconnect on its next turn.
    let claim = retry(|| second.claim(vec![scope("text", 0)]));
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
    let claim = head(&server.socket).claim(vec![scope("flow", 0)]).unwrap();
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
    let got = relaunched.get(vec![key(5)], Mode::Write).unwrap();
    assert_eq!(got.tensors.len(), 1);
    let status = child.wait().unwrap();
    assert_eq!(
        std::os::unix::process::ExitStatusExt::signal(&status),
        Some(libc::SIGKILL)
    );
    let (_, descriptor) = got.storages.into_iter().next().unwrap();
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
    use storage::protocol::{PageLevel, PageShape, PageSlot, Reply, Request};

    let server = Running::start();
    let claim = head(&server.socket).claim(vec![scope("text", 0)]).unwrap();

    // The worker maps its pool on the host.
    let mut worker = worker(&server.socket, scope("text", 0), 1, claim);
    let shape = PageShape {
        layout: key(3),
        kv_head_offset: 0,
        kv_head_count: 2,
        page_bytes: 16,
    };
    let answer = worker
        .call(
            &Request::PagePool {
                shape,
                numa_node: None,
                bytes: None,
            },
            &[],
        )
        .unwrap();
    let Reply::PagePool { pool, slots, .. } = answer.reply else {
        panic!("a page pool request answered {:?}", answer.reply);
    };
    assert_eq!((slots, answer.descriptors.len()), (4, 1));

    // The head reaches the pool from another host.
    let mut remote = Client::connect_tcp(
        &server.tcp,
        Role::Head {
            deployment: "model".into(),
        },
    )
    .unwrap();
    let mut ask = |request: Request| remote.call(&request, &[]).unwrap().reply;
    let Reply::Reserved { slots } = ask(Request::Reserve {
        pool,
        writer: scope("text", 0),
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
        Reply::Lookup {
            levels: vec![PageLevel::Host, PageLevel::Absent]
        }
    );
    // A connection from another host claims nothing.
    assert!(matches!(
        ask(Request::Claim { scopes: Vec::new() }),
        Reply::Error {
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
            },
        ),
        Err(Error::Refused(_))
    ));
}

/// The next event `client` receives, waiting up to ten seconds.
fn next_event(client: &mut Client) -> (storage::protocol::Event, Vec<OwnedFd>) {
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
    use storage::protocol::{Event, PageLevel, PageShape, PageSlot, Reply, Request};

    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head.claim(vec![scope("text", 0)]).unwrap();
    let mut worker = worker(&server.socket, scope("text", 0), 1, claim);
    let shape = PageShape {
        layout: key(3),
        kv_head_offset: 0,
        kv_head_count: 2,
        page_bytes: 16,
    };
    let Reply::PagePool { pool, .. } = worker
        .call(
            &Request::PagePool {
                shape,
                numa_node: None,
                bytes: None,
            },
            &[],
        )
        .unwrap()
        .reply
    else {
        panic!("the page pool was refused");
    };

    // The connector is given the pool with its memory.
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
    let Reply::Reserved { slots } = head
        .call(
            &Request::Reserve {
                pool,
                writer: scope("text", 0),
                keys: vec![key(1)],
                window: None,
            },
            &[],
        )
        .unwrap()
        .reply
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
        connector.call(&finished, &[]).unwrap().reply,
        Reply::Finished
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
        head.wait(lookup).unwrap().reply,
        Reply::Lookup {
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
    assert_eq!(head.call(&prefetch, &[]).unwrap().reply, Reply::Prefetching);
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
        head.wait(lookup).unwrap().reply,
        Reply::Lookup {
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
    use storage::protocol::{PageShape, Reply, Request};

    let shape = PageShape {
        layout: key(3),
        kv_head_offset: 0,
        kv_head_count: 2,
        page_bytes: 16,
    };
    let answer = worker.call(
        &Request::PagePool {
            shape,
            numa_node: None,
            bytes,
        },
        &[],
    )?;
    match answer.reply {
        Reply::PagePool { pool, slots, .. } => Ok((pool, slots)),
        reply => panic!("a page pool request answered {reply:?}"),
    }
}

#[test]
fn page_pools_belong_to_their_deployment_and_leave_with_it() {
    use storage::protocol::{Reply, Request};

    let server = Running::start();
    // The head stays connected, so another deployment's claim keeps this
    // one.
    let mut model_head = head_of(&server.socket, "model");
    let claim = model_head.claim(vec![scope_of("model")]).unwrap();
    let mut first = worker(&server.socket, scope_of("model"), 1, claim);
    // A worker sizes its pool: two pages of 16 bytes.
    let (pool, slots) = page_pool(&mut first, Some(32)).unwrap();
    assert_eq!(slots, 2);
    // The deployment's other workers of the same pages share it.
    assert_eq!(page_pool(&mut first, None).unwrap(), (pool, 2));

    // Another deployment's workers of the same pages have a pool of their
    // own, which its head alone reaches.
    let mut other_head = head_of(&server.socket, "other");
    let other_claim = other_head.claim(vec![scope_of("other")]).unwrap();
    let mut second = worker(&server.socket, scope_of("other"), 1, other_claim);
    let (other_pool, other_slots) = page_pool(&mut second, None).unwrap();
    assert_ne!(other_pool, pool);
    assert_eq!(other_slots, 4);
    let lookup = other_head
        .call(
            &Request::Lookup {
                pool,
                keys: vec![key(1)],
            },
            &[],
        )
        .unwrap();
    assert!(matches!(
        lookup.reply,
        Reply::Error {
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
    let claim = gone.claim(vec![scope_of("gone")]).unwrap();
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
    let kept_claim = kept.claim(vec![scope_of("kept")]).unwrap();
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
        .claim(vec![scope_of("new")])
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
fn a_page_pool_the_hosts_memory_cannot_hold_is_refused() {
    use storage::protocol::{PageShape, Request};

    let server = Running::start();
    let claim = head(&server.socket).claim(vec![scope("text", 0)]).unwrap();
    let mut worker = worker(&server.socket, scope("text", 0), 1, claim);
    // A million pages of a GiB: a pebibyte, beyond any host's memory.
    let shape = PageShape {
        layout: key(3),
        kv_head_offset: 0,
        kv_head_count: 2,
        page_bytes: 1 << 30,
    };
    let refused = worker
        .call(
            &Request::PagePool {
                shape,
                numa_node: None,
                bytes: Some(1 << 50),
            },
            &[],
        )
        .unwrap();
    assert!(matches!(
        refused.reply,
        storage::protocol::Reply::Error {
            kind: ErrorKind::Resource,
            ..
        }
    ));
    let listing = worker.list().unwrap();
    assert!(listing.page_pools.is_empty());
}
