//! A Storage Server serving clients over its socket: publishing and mapping
//! tensors, write access, claims and shoot-down.
//!
//! Each test runs a server on a thread of the test process with a socket of
//! its own, and host storages (memfds), so no GPU is needed. A process shot
//! down is a child this test binary starts as a stand-in worker
//! (`stand_in_writer`), never the test process itself.

#![cfg(test)]

use std::io::{Read, Seek, Write};
use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use uniserve_storage::protocol::ErrorKind;
use uniserve_storage::{
    Access, Claim, Client, Error, Key, Mode, Placement, Role, Scope, Server, StorageRecord,
    TensorRecord, socket,
};

/// A server on its own socket, stopped when dropped.
struct Running {
    socket: PathBuf,
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
        let thread = std::thread::spawn(move || {
            let _lock = lock;
            let mut server = Server::new(listener).unwrap();
            server
                .run(reader.as_fd(), &|| stopped.load(Ordering::Acquire))
                .unwrap();
        });
        Self {
            socket,
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
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "stand_in_writer", "--ignored", "--nocapture"])
        .env("STORAGE_TEST_SOCKET", &server.socket)
        .env("STORAGE_TEST_CLAIM", encoded.join(","))
        .stdout(std::process::Stdio::piped())
        // A worker leads its own process group, which a shoot-down kills.
        .process_group(0)
        .spawn()
        .unwrap();
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
