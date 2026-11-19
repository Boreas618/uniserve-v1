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

use ::storage::protocol::{ClientMessage, ErrorKind, Hello, Item, ServerMessage, Welcome};
use ::storage::{
    Access, Claim, Claimant, Client, Error, Key, Location, Mode, Role, Scope, StorageRecord,
    TensorRecord,
};
use storage_server::{Children, Program, Server};

/// The helper executable a test's server runs its children from, logging warnings and errors
/// alone beside the test's output.
pub(crate) fn helper() -> Program {
    let path = Path::new(env!("CARGO_BIN_EXE_storage-server-helper"));
    Program::open(path, vec!["--log-level".into(), "warn".into()]).unwrap()
}

/// The Transfer Server a test's server runs from the helper, with the default settings' options
/// and `calibration`, which states the GPUs it copies on.
pub(crate) fn transfer_child(calibration: storage_server::transfer::Calibration) -> Children {
    let settings = ::storage::ServerSettings::default();
    Children {
        program: helper(),
        connector: None,
        transfer: Some(storage_server::transfer::Options::of(
            &settings,
            calibration,
        )),
    }
}

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
        Self::with_children(None)
    }

    /// A server that runs `children`.
    pub(crate) fn with_children(children: Option<Children>) -> Self {
        Self::serving(0, |_| {}, children)
    }

    /// A server that runs `children` and accepts other hosts on every address of this host,
    /// which they reach at this host's name ([`Running::tcp`]).
    pub(crate) fn reachable(children: Option<Children>) -> Self {
        let settings = ::storage::Settings::default();
        Self::listening(settings, |_| {}, children, "0.0.0.0:0")
    }

    /// A server that `configure` sets up before it serves, with no children.
    pub(crate) fn configured(configure: impl FnOnce(&mut Server) + Send + 'static) -> Self {
        Self::serving(0, configure, None)
    }

    /// A server that leaves `host_reserve_gib` GiB of the host's available
    /// storage to other processes.
    pub(crate) fn reserving(host_reserve_gib: u64) -> Self {
        Self::serving(host_reserve_gib, |_| {}, None)
    }

    /// A server that serves with `settings`, with no children.
    pub(crate) fn with_settings(settings: ::storage::Settings) -> Self {
        Self::serving_with(settings, |_| {}, None)
    }

    fn serving(
        host_reserve_gib: u64,
        configure: impl FnOnce(&mut Server) + Send + 'static,
        children: Option<Children>,
    ) -> Self {
        let settings = ::storage::Settings {
            server: ::storage::ServerSettings {
                host_reserve_gib,
                ..::storage::ServerSettings::default()
            },
            connector: None,
        };
        Self::serving_with(settings, configure, children)
    }

    fn serving_with(
        settings: ::storage::Settings,
        configure: impl FnOnce(&mut Server) + Send + 'static,
        children: Option<Children>,
    ) -> Self {
        Self::listening(settings, configure, children, "127.0.0.1:0")
    }

    /// A server that accepts other hosts on `bind`: on the loopback address, or on every address
    /// of this host, which other hosts reach at its name.
    fn listening(
        settings: ::storage::Settings,
        configure: impl FnOnce(&mut Server) + Send + 'static,
        children: Option<Children>,
        bind: &str,
    ) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("storage.sock");
        let (listener, starting) = service::host::bind::<::storage::Settings>(&socket)
            .unwrap()
            .owned(&socket)
            .unwrap();
        let tcp = std::net::TcpListener::bind(bind).unwrap();
        let bound = tcp.local_addr().unwrap();
        let address = if bound.ip().is_unspecified() {
            let host = rustix::system::uname()
                .nodename()
                .to_string_lossy()
                .into_owned();
            format!("{host}:{}", bound.port())
        } else {
            bound.to_string()
        };
        let record = readiness(&socket, Some(&address));
        let lock = starting.finish(&record).unwrap();
        Self::on_thread(directory, socket, address, move || {
            let mut server = Server::new(listener, settings, children).unwrap();
            configure(&mut server);
            server.listen_tcp(tcp).unwrap();
            // A server states what `serve` publishes for it.
            server.set_readiness(record);
            (server, lock)
        })
    }
    /// Runs the server `make` makes, with what holds its socket's lock, on
    /// a thread of its own until dropped.
    fn on_thread<L: Send + 'static>(
        directory: tempfile::TempDir,
        socket: PathBuf,
        tcp: String,
        make: impl FnOnce() -> (Server, L) + Send + 'static,
    ) -> Self {
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
            let (mut server, _lock) = make();
            server
                .run(reader.as_fd(), &|| stopped.load(Ordering::Acquire))
                .unwrap();
        });
        Self {
            socket,
            tcp,
            stop,
            wake: writer,
            thread: Some(thread),
            _directory: directory,
        }
    }
}

/// The readiness record of a test's server on `socket`, accepting other hosts at `tcp`.
fn readiness(socket: &Path, tcp: Option<&str>) -> ::storage::Readiness {
    service::Readiness::of_process(
        service::State::Serving,
        None,
        service::Endpoints {
            storage: Some(service::Endpoint::unix(socket)),
            peer: tcp.map(|tcp| tcp.parse().unwrap()),
            ..service::Endpoints::default()
        },
        ::storage::Settings::default(),
    )
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
            generation: 0,
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
        .claim(
            Claimant::Head,
            0,
            vec![scope("text", 0), scope("text", 1)],
            Vec::new(),
        )
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
        .claim(
            Claimant::Head,
            0,
            vec![scope("text", 0), scope("text", 1)],
            Vec::new(),
        )
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
    let claim = head(&server.socket)
        .claim(Claimant::Head, 0, vec![scope("flow", 0)], Vec::new())
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
    // workers below, of this process, must not be: each closes, acknowledged, first.
    let server = Running::start();
    let mut first = head(&server.socket);
    let claim = first
        .claim(
            Claimant::Head,
            0,
            vec![scope("text", 0), scope("old", 0)],
            Vec::new(),
        )
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
        client.close().unwrap();
    }

    let mut second = head(&server.socket);
    assert_eq!(
        server_error(second.claim(Claimant::Head, 0, Vec::new(), Vec::new())),
        ErrorKind::Claimed
    );
    first.close().unwrap();
    let claim = second
        .claim(Claimant::Head, 0, vec![scope("text", 0)], Vec::new())
        .unwrap();
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

/// Hexadecimal of `bytes`, as the helper's arguments state them.
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Starts the helper's `role` with `arguments`, its standard input a pipe the test holds, and
/// returns once it said it is ready: one byte on its standard output.
pub(crate) fn start_client(
    role: &str,
    arguments: &[String],
    configure: impl FnOnce(&mut std::process::Command),
) -> std::process::Child {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_storage-server-helper"));
    command
        .arg(role)
        .args(arguments)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped());
    configure(&mut command);
    let mut child = command.spawn().unwrap();
    let mut ready = [0_u8; 1];
    child
        .stdout
        .take()
        .unwrap()
        .read_exact(&mut ready)
        .expect("the helper ended before it was ready");
    child
}

#[test]
fn a_relaunched_worker_takes_write_access_by_shooting_down_its_predecessor() {
    let server = Running::start();
    let claim = head(&server.socket)
        .claim(Claimant::Head, 0, vec![scope("flow", 0)], Vec::new())
        .unwrap();
    // A worker of the scope holds write access to an exclusive tensor; a worker leads its own
    // process group, which a shoot-down kills.
    let arguments = [
        "--socket".to_owned(),
        server.socket.display().to_string(),
        "--group".to_owned(),
        "flow".to_owned(),
        "--rank".to_owned(),
        "0".to_owned(),
        "--claim".to_owned(),
        hex(&claim.0),
        "--key".to_owned(),
        hex(&key(5).0),
        "--bytes".to_owned(),
        "4".to_owned(),
        "--fill".to_owned(),
        "3".to_owned(),
    ];
    let mut child = start_client("writer", &arguments, |command| {
        command.process_group(0);
    });
    // The next launch of the same scope takes the tensor: the server kills
    // the child, waits for it to exit and passes write access on.
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

/// The engines of one epoch on a host each claim, through the node's daemon,
/// the scopes they place, each as its own claimant (the independent replicas
/// of the epoch's configuration, and a process that replaces one of them):
/// one engine's claim leaves the storage another engine of the epoch retains
/// under its own scope, and a claim that retires the epoch ends it.
#[test]
fn an_engine_s_claim_keeps_another_engine_of_its_epoch_retained() {
    let server = Running::start();
    let mut claim = ::storage::ClaimClient::connect(&server.socket, "model").unwrap();
    let mut observer = head(&server.socket);
    let token = claim
        .claim(
            Claimant::Engine(1),
            1,
            vec![scope("model-0", 0)],
            Vec::new(),
        )
        .unwrap();

    // The first replica's worker publishes its exclusive planes.
    let mut first = epoch_worker(&server.socket, scope("model-0", 0), 1, token);
    let planes = memfd(&[1; 4]);
    first
        .put(
            vec![tensor(key(1), "planes", 0, 0, Access::Exclusive)],
            &[(host(4), planes.as_fd())],
        )
        .unwrap();
    assert_eq!(retained_names(&mut observer), vec!["planes"]);

    // The second replica of the same epoch claims its own scope, and the
    // first replica's relaunch, another engine, claims the first one's again.
    let second = claim
        .claim(
            Claimant::Engine(2),
            1,
            vec![scope("model-1", 0)],
            Vec::new(),
        )
        .unwrap();
    assert_eq!(second, token);
    assert_eq!(retained_names(&mut observer), vec!["planes"]);
    let relaunch = claim
        .claim(
            Claimant::Engine(3),
            1,
            vec![scope("model-0", 0)],
            Vec::new(),
        )
        .unwrap();
    assert_eq!(relaunch, token);
    assert_eq!(retained_names(&mut observer), vec!["planes"]);

    // The next epoch's claim that retires epoch 1 ends its storage.
    claim
        .claim(Claimant::Head, 2, vec![scope("model-0", 0)], vec![1])
        .unwrap();
    assert!(retained_names(&mut observer).is_empty());
}

/// The server keeps an epoch's storage of a scope while any claimant of the
/// epoch names it: another claimant's claim that does not name the scope
/// ends nothing, and the storage ends once its own claimant claims again
/// without it.
#[test]
fn a_claimant_s_claim_never_ends_another_claimant_s_retention() {
    let server = Running::start();
    let mut daemon = head(&server.socket);
    let token = daemon
        .claim(
            Claimant::Engine(1),
            1,
            vec![scope("model-0", 0)],
            Vec::new(),
        )
        .unwrap();
    let mut first = epoch_worker(&server.socket, scope("model-0", 0), 1, token);
    let planes = memfd(&[1; 4]);
    first
        .put(
            vec![tensor(key(1), "planes", 0, 0, Access::Exclusive)],
            &[(host(4), planes.as_fd())],
        )
        .unwrap();

    // Another claimant of the epoch names other scopes alone, and the head on
    // the epoch's behalf names none: the first claimant's storage stays.
    daemon
        .claim(
            Claimant::Engine(2),
            1,
            vec![scope("model-1", 0)],
            Vec::new(),
        )
        .unwrap();
    daemon
        .claim(Claimant::Head, 1, Vec::new(), Vec::new())
        .unwrap();
    assert_eq!(retained_names(&mut daemon), vec!["planes"]);

    // Its own claimant claims again without the scope: the storage ends.
    daemon
        .claim(Claimant::Engine(1), 1, Vec::new(), Vec::new())
        .unwrap();
    assert!(retained_names(&mut daemon).is_empty());
}

/// An epoch that serves on after an update moved one of its replicas ends that replica's storage
/// alone: `Unclaim` ends the epoch's retention of the moved scope whichever claimant named it,
/// and the other replica's storage, and a later claim of the same claimant, keep the epoch's
/// other scope.
#[test]
fn an_unclaim_ends_an_epochs_scope_and_keeps_its_others() {
    let server = Running::start();
    let mut claim = ::storage::ClaimClient::connect(&server.socket, "model").unwrap();
    let mut observer = head(&server.socket);
    let token = claim
        .claim(
            Claimant::Engine(1),
            1,
            vec![scope("model-0", 0)],
            Vec::new(),
        )
        .unwrap();
    claim
        .claim(
            Claimant::Engine(2),
            1,
            vec![scope("model-1", 0)],
            Vec::new(),
        )
        .unwrap();
    for (index, name, group) in [(1, "moved", "model-0"), (2, "staying", "model-1")] {
        let mut worker = epoch_worker(&server.socket, scope(group, 0), 1, token);
        let planes = memfd(&[1; 4]);
        worker
            .put(
                vec![tensor(key(index), name, 0, 0, Access::Exclusive)],
                &[(host(4), planes.as_fd())],
            )
            .unwrap();
    }
    assert_eq!(retained_names(&mut observer), vec!["moved", "staying"]);

    assert_eq!(claim.unclaim(1, vec![scope("model-0", 0)]).unwrap(), 1);
    assert_eq!(retained_names(&mut observer), vec!["staying"]);
    claim
        .claim(
            Claimant::Engine(2),
            1,
            vec![scope("model-1", 0)],
            Vec::new(),
        )
        .unwrap();
    assert_eq!(retained_names(&mut observer), vec!["staying"]);
}

/// Two epochs of one deployment share a scope without touching each other's
/// storage: the head keeps one claim token for both, each epoch's `retain`
/// releases its own tensors alone, and a claim that retires an epoch releases
/// the tensors no other epoch retains.
#[test]
fn epochs_retain_their_own_storage_and_a_claim_retires_one() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let first_claim = head
        .claim(Claimant::Head, 1, vec![scope("text", 0)], Vec::new())
        .unwrap();
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
    let second_claim = head
        .claim(Claimant::Head, 2, vec![scope("text", 0)], Vec::new())
        .unwrap();
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
    head.claim(Claimant::Head, 2, vec![scope("text", 0)], vec![1])
        .unwrap();
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

/// The engine of `epoch` of the deployment: a head of that epoch.
pub(crate) fn engine_head(socket: &Path, epoch: u64) -> Client {
    Client::connect(
        socket,
        Role::Head {
            deployment: "model".into(),
            epoch,
            generation: 0,
        },
    )
    .unwrap()
}

/// The read-only tensors `worker` puts, each of four bytes in host storage of its own, under
/// `keys`.
fn put_read_only(worker: &mut Client, keys: &[Key]) {
    let storages: Vec<std::fs::File> = keys.iter().map(|key| memfd(&[key.0[0]; 4])).collect();
    let tensors = keys
        .iter()
        .enumerate()
        .map(|(index, key)| {
            let name = format!("tensor-{}", key.0[0]);
            tensor(*key, &name, index as u32, 0, Access::ReadOnly)
        })
        .collect();
    let storages: Vec<_> = storages
        .iter()
        .map(|storage| (host(4), storage.as_fd()))
        .collect();
    worker.put(tensors, &storages).unwrap();
}

/// The keys of `keys` whose read-only tensors `worker` reads, in key order.
fn readable(worker: &mut Client, keys: &[Key]) -> Vec<Key> {
    let (found, _) = worker.get(keys.to_vec(), Mode::Read, Vec::new()).unwrap();
    let mut readable: Vec<Key> = found.tensors.iter().map(|tensor| tensor.key).collect();
    readable.sort();
    readable
}

/// Independent replicas read each other's committed tensors while each candidate's stay
/// private. A group commit publishes all successors together, regardless of the numerical order
/// of the epoch identifiers; only the claiming head publishes, and a set naming an epoch it did
/// not claim publishes none.
#[test]
fn independent_replicas_publish_successors_as_a_group() {
    let server = Running::start();
    let mut daemon = head(&server.socket);
    let claim = daemon
        .claim(Claimant::Head, 9, vec![scope("text", 0)], Vec::new())
        .unwrap();
    daemon
        .claim(Claimant::Head, 3, vec![scope("text", 0)], Vec::new())
        .unwrap();
    daemon.publish(vec![9, 3]).unwrap();
    let mut first = epoch_worker(&server.socket, scope("text", 0), 9, claim);
    let mut second = epoch_worker(&server.socket, scope("text", 0), 3, claim);
    put_read_only(&mut first, &[key(1)]);
    put_read_only(&mut second, &[key(2)]);
    assert_eq!(readable(&mut first, &[key(1), key(2)]), [key(1), key(2)]);
    assert_eq!(readable(&mut second, &[key(1), key(2)]), [key(1), key(2)]);

    daemon
        .claim(Claimant::Head, 4, vec![scope("text", 0)], Vec::new())
        .unwrap();
    daemon
        .claim(Claimant::Head, 8, vec![scope("text", 0)], Vec::new())
        .unwrap();
    let mut candidate = epoch_worker(&server.socket, scope("text", 0), 4, claim);
    let mut peer = epoch_worker(&server.socket, scope("text", 0), 8, claim);
    put_read_only(&mut candidate, &[key(3)]);
    put_read_only(&mut peer, &[key(4)]);
    let mut candidate_head = engine_head(&server.socket, 4);
    assert_eq!(
        server_error(candidate_head.publish(vec![4])),
        ErrorKind::Invalid
    );
    assert!(readable(&mut first, &[key(3), key(4)]).is_empty());
    assert_eq!(readable(&mut candidate, &[key(3), key(4)]), [key(3)]);

    // A rejected set publishes none of its otherwise valid members.
    assert_eq!(
        server_error(daemon.publish(vec![4, 99])),
        ErrorKind::Invalid
    );
    assert!(readable(&mut first, &[key(3), key(4)]).is_empty());
    daemon.publish(vec![4, 8]).unwrap();
    assert_eq!(readable(&mut first, &[key(3), key(4)]), [key(3), key(4)]);
}

/// A rollback discards what the epoch on trial published: the read-only tensors it alone put
/// leave the server. Until then the serving epoch reads none of them, and a tensor of the same
/// key that the serving epoch puts carries the serving epoch's bytes.
#[test]
fn a_rollback_discards_what_the_epoch_on_trial_published() {
    let server = Running::start();
    let mut daemon = head(&server.socket);
    let claim = daemon
        .claim(Claimant::Head, 1, vec![scope("text", 0)], Vec::new())
        .unwrap();
    daemon.publish(vec![1]).unwrap();
    let mut serving_worker = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    daemon
        .claim(Claimant::Head, 2, vec![scope("text", 0)], Vec::new())
        .unwrap();
    let mut trial_worker = epoch_worker(&server.socket, scope("text", 0), 2, claim);

    // Epoch 2 publishes two read-only tensors.
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

    // The rollback retires epoch 2: what it alone put leaves the server.
    daemon
        .claim(Claimant::Head, 1, vec![scope("text", 0)], vec![2])
        .unwrap();
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
    let claim = daemon
        .claim(Claimant::Head, 1, vec![scope("text", 0)], Vec::new())
        .unwrap();
    daemon.publish(vec![1]).unwrap();
    let mut serving = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    let planes = memfd(&[1; 4]);
    serving
        .put(
            vec![tensor(key(10), "planes", 0, 0, Access::Exclusive)],
            &[(host(4), planes.as_fd())],
        )
        .unwrap();
    daemon
        .claim(Claimant::Head, 2, vec![scope("text", 0)], Vec::new())
        .unwrap();
    let mut trial = epoch_worker(&server.socket, scope("text", 0), 2, claim);
    trial.get(vec![key(10)], Mode::Standby, Vec::new()).unwrap();
    serving.retain(Vec::new(), Vec::new()).unwrap();
    serving.close().unwrap();

    let mut restarted = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    let (got, _) = restarted
        .get(vec![key(10)], Mode::Standby, Vec::new())
        .unwrap();
    assert_eq!(got.tensors.len(), 1, "{:?}", got.missing);
}

#[test]
fn host_storage_is_admitted_before_allocation_and_retires_with_its_owner() {
    for refused in [true, false] {
        let server = Running::reserving(if refused { u64::MAX } else { 0 });
        let mut head = head(&server.socket);
        let claim = head
            .claim(Claimant::Head, 1, vec![scope("text", 0)], Vec::new())
            .unwrap();
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
        worker.close().unwrap();
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
        resumed.close().unwrap();
        assert_eq!(head.keep_weights(vec![1], Vec::new()).unwrap(), 1);
        assert!(head.list().unwrap().tensors.is_empty());
    }
}

/// A running server takes the settings it reads when it acts: new host storage its reserve
/// refused is admitted once the operator lowers the reserve, and the server states the settings
/// it serves with from then on. A setting fixed when its processes start is refused, and the
/// server serves on with the settings it had.
#[test]
fn a_running_server_takes_the_settings_it_reads_when_it_acts() {
    let server = Running::reserving(u64::MAX);
    let mut head = head(&server.socket);
    let claim = head
        .claim(Claimant::Head, 1, vec![scope("text", 0)], Vec::new())
        .unwrap();
    let mut worker = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    let refused = worker.allocate_host(key(42), "staged cache".into(), 4096);
    assert_eq!(server_error(refused), ErrorKind::Resource);

    let mut operator = Client::connect(&server.socket, Role::Operator).unwrap();
    let lowered = ::storage::ServerSettings {
        host_reserve_gib: 0,
        transfer_deadline_s: 5,
        shoot_down_s: 5,
        ..::storage::ServerSettings::default()
    };
    operator.configure(lowered.clone()).unwrap();
    let (admitted, _) = worker
        .allocate_host(key(42), "staged cache".into(), 4096)
        .unwrap();
    assert_eq!(admitted.tensors[0].shape, [4096]);
    assert_eq!(operator.status().unwrap().settings.server, lowered);

    let listening = ::storage::ServerSettings {
        tcp: Some("127.0.0.1:0".parse().unwrap()),
        ..lowered.clone()
    };
    assert_eq!(
        server_error(operator.configure(listening)),
        ErrorKind::Invalid
    );
    assert_eq!(operator.status().unwrap().settings.server, lowered);
}

/// A server serves a client in the newest protocol version both windows hold: a client of the
/// server's window, or of a later one that holds its newest, is welcomed and served, and a client
/// whose window shares no version is refused at its hello with the typed refusal naming both
/// windows.
#[test]
fn a_server_serves_the_newest_protocol_version_both_speak() {
    use ::storage::protocol::{Refusal, Request, Response, VERSIONS, VersionRefusal};

    let server = Running::start();
    let timeout = crate::PATIENCE;
    let connect = |oldest: u32, newest: u32| {
        let stream = std::os::unix::net::UnixStream::connect(&server.socket).unwrap();
        let mut connection = service::Connection::new(stream).unwrap();
        let hello = Hello {
            oldest,
            newest: Some(newest),
            role: Role::Operator,
        };
        connection.send(&ClientMessage::Hello(hello)).unwrap();
        let welcome = connection.recv::<ServerMessage>(timeout).unwrap().unwrap();
        (connection, welcome)
    };

    let newest = VERSIONS.newest();
    let (_, welcome) = connect(newest, newest + 1);
    assert_eq!(welcome, ServerMessage::Welcome(Welcome::Accepted));
    let (_, welcome) = connect(VERSIONS.oldest() - 1, newest + 1);
    assert_eq!(
        welcome,
        ServerMessage::Welcome(Welcome::Agreed { version: newest })
    );

    let (mut connection, welcome) = connect(VERSIONS.oldest(), newest);
    assert_eq!(
        welcome,
        ServerMessage::Welcome(Welcome::of_version(
            &Hello::new(VERSIONS, Role::Operator),
            newest
        ))
    );
    let listed = ClientMessage::Request {
        id: 7,
        request: Request::List,
    };
    connection.send(&listed).unwrap();
    let response = connection.recv::<ServerMessage>(timeout).unwrap().unwrap();
    assert!(
        matches!(
            response,
            ServerMessage::Response {
                id: 7,
                response: Response::Listing(_)
            }
        ),
        "{response:?}"
    );

    for offered in [VERSIONS.oldest() - 1, newest + 1] {
        let (_, welcome) = connect(offered, offered);
        assert_eq!(
            welcome,
            ServerMessage::Welcome(Welcome::Refused {
                refusal: Refusal::Version(VersionRefusal {
                    offered,
                    spoken: newest,
                    offered_newest: Some(offered),
                    spoken_oldest: Some(VERSIONS.oldest()),
                }),
            }),
            "a client of protocol {offered}"
        );
    }
}

#[test]
fn one_key_keeps_a_copy_at_each_location() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head
        .claim(
            Claimant::Head,
            0,
            vec![scope("text", 0), scope("text", 1)],
            Vec::new(),
        )
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

    // Each copy lives while some scope retains it at its location.
    second.retain(Vec::new(), Vec::new()).unwrap();
    first.retain(vec![key(1)], vec![near]).unwrap();
    let listed: Vec<(Key, Location)> = head
        .list()
        .unwrap()
        .tensors
        .into_iter()
        .map(|tensor| (tensor.key, tensor.location))
        .collect();
    assert_eq!(listed, vec![(key(1), near)]);
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

/// Two epochs of a deployment share a GPU within one quota: each storage is
/// reserved once, however many epochs map it, an epoch's reserved items end with
/// its processes, nothing takes a GPU past its quota, and a commit
/// passes the predecessor's segments the successor maps to the successor on
/// the quota, as a transfer rather than a release and an allocation.
#[test]
fn two_epochs_share_a_gpus_quota_and_a_commit_transfers_the_segments() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head
        .claim(Claimant::Head, 1, vec![scope("text", 0)], Vec::new())
        .unwrap();
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
    head.claim(Claimant::Head, 2, vec![scope("text", 0)], Vec::new())
        .unwrap();
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
    first.close().unwrap();
    assert_eq!(quota_of(&mut head).usage(), 500);
    head.claim(Claimant::Head, 2, vec![scope("text", 0)], vec![1])
        .unwrap();
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
    let claim = head
        .claim(Claimant::Head, 1, vec![scope("text", 0)], Vec::new())
        .unwrap();
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
    head.claim(Claimant::Head, 2, vec![scope("text", 0)], Vec::new())
        .unwrap();
    let mut second = epoch_worker(&server.socket, scope("text", 0), 2, claim);
    second.get(vec![key(1)], Mode::Read, Vec::new()).unwrap();
    second.get(vec![key(2)], Mode::Standby, Vec::new()).unwrap();
    assert_eq!(quota_of(&mut head).usage(), 370);

    // The previous epoch's processes exit; the head keeps its weights.
    // The kept epoch's processes end in order, withdrawing what they reserved.
    first
        .reserve(GPU, 1000, Some(700), vec![(Item::Process, 0)])
        .unwrap();
    first.close().unwrap();
    assert_eq!(quota_of(&mut head).usage(), 270);
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

/// The bytes reserved on `GPU` over every quota: none when nothing is
/// reserved there and the GPU has no quota.
fn usage_on_gpu(client: &mut Client) -> u64 {
    let quotas = client.usage(vec![GPU]).unwrap().quotas;
    quotas.iter().map(::storage::Quota::usage).sum()
}

/// A previous epoch's processes the head stopped end without withdrawing what
/// they reserved, as lost processes do. Once the epoch keeps its weights
/// alone, nothing of theirs stays reserved, whether a process ended before the
/// request or after it; a rollback's relaunch reserves again, and its
/// reservation is kept for its own relaunch.
#[test]
fn a_epoch_kept_as_its_weights_keeps_nothing_its_stopped_processes_reserved() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let scopes = vec![scope("text", 0), scope("text", 1)];
    let claim = head
        .claim(Claimant::Head, 1, scopes.clone(), Vec::new())
        .unwrap();
    head.publish(vec![1]).unwrap();
    let mut stopped = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    stopped
        .reserve(
            GPU,
            1000,
            Some(700),
            vec![(Item::Process, 300), (Item::GraphPool, 200)],
        )
        .unwrap();
    let mut later = epoch_worker(&server.socket, scope("text", 1), 1, claim);
    later
        .reserve(GPU, 1000, Some(700), vec![(Item::Process, 100)])
        .unwrap();
    head.claim(Claimant::Head, 2, scopes, Vec::new()).unwrap();

    // One process is stopped before the request: its 500 bytes are kept as a
    // lost process's until the epoch keeps its weights alone.
    stopped.close().unwrap();
    head.keep_weights(vec![1], Vec::new()).unwrap();
    assert_eq!(quota_of(&mut head).usage(), 100);

    // The other is stopped after it.
    later.close().unwrap();
    assert_eq!(usage_on_gpu(&mut head), 0);

    // A rollback starts the epoch again: its process reserves, and a loss of
    // it keeps the reservation for its relaunch, once the server has seen the
    // connection end.
    let mut relaunch = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    relaunch
        .reserve(GPU, 1000, Some(700), vec![(Item::Process, 250)])
        .unwrap();
    relaunch.close().unwrap();
    assert_eq!(usage_on_gpu(&mut head), 250);
}

/// A process of an epoch that a commit retired while the process was still
/// connected, as a head's stop and its claim race, keeps nothing reserved once
/// its connection ends: no relaunch of a retired epoch takes it over.
#[test]
fn a_retired_epoch_s_process_keeps_nothing_reserved_when_it_ends() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head
        .claim(Claimant::Head, 1, vec![scope("text", 0)], Vec::new())
        .unwrap();
    head.publish(vec![1]).unwrap();
    let mut stopped = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    stopped
        .reserve(
            GPU,
            1000,
            Some(700),
            vec![(Item::Process, 300), (Item::GraphPool, 200)],
        )
        .unwrap();
    head.claim(Claimant::Head, 2, vec![scope("text", 0)], Vec::new())
        .unwrap();
    head.publish(vec![2]).unwrap();
    head.claim(Claimant::Head, 2, vec![scope("text", 0)], vec![1])
        .unwrap();
    assert_eq!(usage_on_gpu(&mut head), 500);

    stopped.close().unwrap();
    assert_eq!(usage_on_gpu(&mut head), 0);
}

/// A head moves an epoch's weights between its GPUs and host storage only
/// through the Transfer Server: without one, an offload of weights is
/// refused and moves nothing, while an epoch with no weights to move (none
/// that it alone retains on a GPU, none in host storage) gets its response at once.
/// Only the head moves weights.
#[test]
fn a_weight_move_without_a_transfer_server_is_refused_and_moves_nothing() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head
        .claim(Claimant::Head, 1, vec![scope("text", 0)], Vec::new())
        .unwrap();
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
fn the_processes_of_one_scope_count_as_their_largest_reservation() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head
        .claim(Claimant::Head, 1, vec![scope("text", 0)], Vec::new())
        .unwrap();
    head.publish(vec![1]).unwrap();
    let mut rank = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    rank.reserve(GPU, 1000, Some(700), vec![(Item::Process, 300)])
        .unwrap();
    let mut standby = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    standby
        .reserve(GPU, 1000, Some(700), vec![(Item::Process, 350)])
        .unwrap();
    assert_eq!(quota_of(&mut head).usage(), 350);
    standby.close().unwrap();
    assert_eq!(quota_of(&mut head).usage(), 300);
}

/// A lost rank's reservation stays reserved for the process that replaces it:
/// another scope cannot reserve into it, and the relaunch takes each item over
/// as it reserves it.
#[test]
fn a_lost_rank_s_reservation_is_kept_for_its_relaunch() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head
        .claim(
            Claimant::Head,
            1,
            vec![scope("text", 0), scope("text", 1)],
            Vec::new(),
        )
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
    rank.close().unwrap();

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

/// A new head's claim stands once every process of the previous claim is out,
/// so nothing those processes reserved is kept: a deployment started again,
/// whose new head numbers its epoch as the previous one did, reserves its
/// processes against the whole quota.
#[test]
fn a_new_head_keeps_nothing_the_previous_claim_s_processes_reserved() {
    let server = Running::start();
    let mut previous = head(&server.socket);
    let claim = previous
        .claim(Claimant::Head, 1, vec![scope("text", 0)], Vec::new())
        .unwrap();
    previous.publish(vec![1]).unwrap();
    let mut rank = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    rank.reserve(
        GPU,
        1000,
        Some(700),
        vec![(Item::Process, 300), (Item::GraphPool, 200)],
    )
    .unwrap();
    rank.close().unwrap();
    assert_eq!(quota_of(&mut previous).usage(), 500);
    previous.close().unwrap();

    let mut started = head(&server.socket);
    let claim = started
        .claim(Claimant::Head, 1, vec![scope("text", 0)], Vec::new())
        .unwrap();
    started.publish(vec![1]).unwrap();
    assert_eq!(quota_of(&mut started).usage(), 0);

    let mut rank = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    rank.reserve(GPU, 1000, Some(700), vec![(Item::Process, 600)])
        .unwrap();
    assert_eq!(quota_of(&mut started).usage(), 600);
}

/// A rank that withdraws its reservations before its connection ends, as a
/// worker that ends in order does, leaves nothing reserved, and a lost
/// rank's reservation ends with its epoch's retirement.
#[test]
fn reservations_end_with_a_withdrawal_or_the_epoch_s_retirement() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head
        .claim(
            Claimant::Head,
            1,
            vec![scope("text", 0), scope("text", 1)],
            Vec::new(),
        )
        .unwrap();
    head.publish(vec![1]).unwrap();
    let mut closed = epoch_worker(&server.socket, scope("text", 0), 1, claim);
    closed
        .reserve(GPU, 1000, Some(700), vec![(Item::Process, 300)])
        .unwrap();
    closed
        .reserve(GPU, 1000, Some(700), vec![(Item::Process, 0)])
        .unwrap();
    closed.close().unwrap();
    assert_eq!(quota_of(&mut head).usage(), 0);

    let mut lost = epoch_worker(&server.socket, scope("text", 1), 1, claim);
    lost.reserve(GPU, 1000, Some(700), vec![(Item::Process, 300)])
        .unwrap();
    lost.close().unwrap();
    assert_eq!(quota_of(&mut head).usage(), 300);
    head.claim(Claimant::Head, 2, vec![scope("text", 0)], Vec::new())
        .unwrap();
    head.publish(vec![2]).unwrap();
    head.claim(Claimant::Head, 2, vec![scope("text", 0)], vec![1])
        .unwrap();
    // Nothing is reserved on the GPU any more: it has no quota, or an empty
    // one.
    let quotas = head.usage(vec![GPU]).unwrap().quotas;
    assert_eq!(quotas.iter().map(::storage::Quota::usage).sum::<u64>(), 0);
}

/// A worker reserves only its own items, and a reservation that leaves a quota
/// past its quota (as a smaller fraction an epoch states since may) shrinks
/// it but never grows it.
#[test]
fn a_usage_past_its_quota_shrinks_but_never_grows() {
    let server = Running::start();
    let mut head = head(&server.socket);
    let claim = head
        .claim(Claimant::Head, 1, vec![scope("text", 0)], Vec::new())
        .unwrap();
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

/// A standby of a scope maps the scope's exclusive storage without writing
/// it, and adopts it by promotion once its writer gave it up; a worker of
/// another scope cannot take it from the promoted writer.
#[test]
fn a_standby_of_the_scope_adopts_its_storage_by_promotion() {
    let server = Running::start();
    let claim = head(&server.socket)
        .claim(
            Claimant::Head,
            0,
            vec![scope("flow", 0), scope("flow", 1)],
            Vec::new(),
        )
        .unwrap();

    let mut writer = worker(&server.socket, scope("flow", 0), 1, claim);
    let storage = memfd(&[7; 4]);
    writer
        .put(
            vec![tensor(key(9), "planes", 0, 0, Access::Exclusive)],
            &[(host(4), storage.as_fd())],
        )
        .unwrap();

    // The standby maps the very storage, and a key it does not hold in
    // standby is not promoted.
    let mut standby = worker(&server.socket, scope("flow", 0), 2, claim);
    let (mapped, descriptors) = standby
        .get(vec![key(9)], Mode::Standby, Vec::new())
        .unwrap();
    assert_eq!(mapped.tensors.len(), 1);
    assert_eq!(
        contents(descriptors.into_iter().next().unwrap()),
        vec![7; 4]
    );
    assert_eq!(
        server_error(standby.promote(vec![key(8)])),
        ErrorKind::Invalid
    );

    writer.release(vec![key(9)]).unwrap();
    standby.promote(vec![key(9)]).unwrap();

    // The storage has one writer: another scope's worker is refused it.
    let mut other = worker(&server.socket, scope("flow", 1), 1, claim);
    assert_eq!(
        server_error(other.get(vec![key(9)], Mode::Write, Vec::new())),
        ErrorKind::Held
    );
}
