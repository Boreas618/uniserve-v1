//! Probe (a) of `specs/worker.md` §12: can one main process hold several rank
//! writer identities at the Storage Server?
//!
//! §2.5 records this as unverified: the Storage Server identifies a writer by
//! pidfd, scope and generation (`crates/storage-server/src/lib.rs`, the writer
//! rules), and the design merges the per-GPU worker processes into one main
//! process per node. The question the design cannot answer from its text is
//! whether that process can be the writer of every rank of its node: either the
//! server's scopes keep one process's ranks apart, or a process is one writer and
//! its ranks share an identity.
//!
//! What this program measures, on one Storage Server with no GPU:
//!
//! 1. one head claims one world with two scopes (rank 0 and rank 1) of one group;
//! 2. two clients of this one process connect as the workers of those two
//!    scopes, and each publishes an exclusive tensor of its own;
//! 3. rank 1 asks for write access to rank 0's tensor — a second *rank* of the
//!    same process — and the program reports whether the server admits it or
//!    refuses it as held;
//! 4. the listing names the writer of each tensor, which shows whether the
//!    server tells the two ranks apart.
//!
//! A pass is: both clients connected and published, the step-3 answer is
//! recorded as either admitted or `Held` (the run prints which), the listing
//! shows each tensor's writer, and the server stays usable afterwards (a
//! subsequent read succeeds). The answer to the design question is what the run
//! prints; the assertion states the outcome the current server code implies, so
//! that a change in that code changes the test.
//!
//! Run on one GPU-less host; no GPU is used.
//!
//! ```text
//! cargo test -p storage-server --test writer_identities one_process_holds_two_rank_writer_identities -- --nocapture
//! ```

use std::io::Write as _;
use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use storage::{
    Access, Claim, Client, Error, Key, Mode, Role, Scope, StorageRecord, TensorRecord, socket,
};
use storage_server::Server;

/// A Storage Server on its own socket, stopped when dropped.
struct Running {
    socket: PathBuf,
    stop: Arc<AtomicBool>,
    wake: OwnedFd,
    thread: Option<std::thread::JoinHandle<()>>,
    _directory: tempfile::TempDir,
}

impl Running {
    fn start() -> Self {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let socket = directory.path().join("storage.sock");
        let (listener, lock) = socket::bind(&socket)
            .expect("a bound socket")
            .expect("no server");
        let mut ends = [0; 2];
        // SAFETY: pipe2 fills the two descriptors it creates.
        let created = unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_NONBLOCK) };
        assert_eq!(created, 0);
        // SAFETY: two new descriptors this program owns.
        let (reader, writer) =
            unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) };
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let thread = std::thread::spawn(move || {
            let _lock = lock;
            // Pools of 64 four-byte pages, nothing reserved for other processes.
            let mut server = Server::new(listener, 64, 0, None, None).expect("a server");
            server
                .run(reader.as_fd(), &|| stopped.load(Ordering::Acquire))
                .expect("the server runs");
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
        // SAFETY: a write to a pipe this program owns.
        unsafe { libc::write(self.wake.as_raw_fd(), b"x".as_ptr().cast(), 1) };
        if let Some(thread) = self.thread.take() {
            thread.join().expect("the server thread ended");
        }
    }
}

fn scope(rank: u32) -> Scope {
    Scope {
        deployment: "probe".into(),
        group: "text".into(),
        rank,
    }
}

fn key(number: u8) -> Key {
    let mut bytes = [0; 32];
    bytes[0] = number;
    Key(bytes)
}

fn memfd(bytes: &[u8]) -> std::fs::File {
    // SAFETY: memfd_create with a static name.
    let fd = unsafe { libc::memfd_create(c"writers-probe".as_ptr(), libc::MFD_CLOEXEC) };
    assert!(fd >= 0);
    // SAFETY: a new descriptor this program owns.
    let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
    file.write_all(bytes).expect("the memfd takes bytes");
    file
}

fn host(bytes: u64) -> StorageRecord {
    StorageRecord {
        placement: storage::Placement::Host { numa_node: None },
        bytes,
        segment: None,
        fabric: None,
    }
}

fn tensor(key: Key, name: &str, storage: u32, access: Access) -> TensorRecord {
    TensorRecord {
        key,
        name: name.into(),
        storage,
        offset: 0,
        dtype: "uint8".into(),
        shape: vec![4],
        strides: vec![1],
        access,
    }
}

/// The worker client of one rank scope in the claimed world.
fn worker(socket: &Path, scope: Scope, claim: Claim, world: u64) -> Client {
    Client::connect(
        socket,
        Role::Worker {
            scope,
            generation: 1,
            claim,
            world,
        },
    )
    .expect("a worker connection")
}

#[test]
fn one_process_holds_two_rank_writer_identities() {
    let server = Running::start();
    let world = 1;
    let mut head = Client::connect(
        &server.socket,
        Role::Head {
            deployment: "probe".into(),
            world,
        },
    )
    .expect("a head connection");
    let claim = head
        .claim(world, vec![scope(0), scope(1)], Vec::new())
        .expect("the head claims the world with both ranks");

    // Each rank publishes an exclusive tensor of its own from this one process.
    let mut rank0 = worker(&server.socket, scope(0), claim, world);
    let mut rank1 = worker(&server.socket, scope(1), claim, world);
    for (rank, number) in [(0u32, 1u8), (1, 2)] {
        let storage = memfd(&[number; 4]);
        let client = if rank == 0 { &mut rank0 } else { &mut rank1 };
        client
            .put(
                vec![tensor(
                    key(number),
                    &format!("rank{rank}"),
                    0,
                    Access::Exclusive,
                )],
                &[(host(4), storage.as_fd())],
            )
            .expect("a rank publishes its tensor");
    }

    // The question: does rank 1, in this same process, take rank 0's write
    // access? `Held` means the server keeps the ranks apart; an admitted answer
    // means it identifies the writer by process, so this process's ranks share
    // one writer identity.
    let cross = rank1.get(vec![key(1)], Mode::Write, Vec::new());
    let admitted = cross.is_ok();
    match &cross {
        Ok(got) => println!(
            "probe(writers): rank 1 took rank 0's write access in one process: admitted ({} tensor(s), {} storage(s))",
            got.tensors.len(),
            got.storages.len()
        ),
        Err(Error::Server { kind, message }) => {
            println!(
                "probe(writers): rank 1 taking rank 0's write access: refused ({kind:?}: {message})"
            )
        }
        Err(other) => panic!("unexpected transport answer: {other:?}"),
    }

    // What the listing says of each tensor's writer: one pid for both ranks means
    // the server exposes one writer identity for this process.
    let listing = head.list().expect("the head lists the storage");
    for record in &listing.tensors {
        println!(
            "probe(writers): tensor {} published by scope? writer={:?}",
            record.name, record.writer
        );
    }
    let pid = std::process::id() as i32;
    let writers: Vec<Option<i32>> = listing.tensors.iter().map(|record| record.writer).collect();
    assert!(
        writers.iter().all(|writer| *writer == Some(pid)),
        "every tensor of this process names this process as its writer, got {writers:?}"
    );

    // The server is still usable: rank 0 reads its own tensor back.
    let got = rank0
        .get(vec![key(1)], Mode::Read, Vec::new())
        .expect("rank 0 reads its own tensor");
    assert_eq!(got.tensors.len(), 1);

    if admitted {
        println!(
            "probe(writers): ANSWER §2.5 - one process can hold both rank writer identities; scopes of one process are not kept apart"
        );
    } else {
        println!(
            "probe(writers): ANSWER §2.5 - one process cannot hold both rank writer identities; the server keeps the scopes apart"
        );
    }
}
