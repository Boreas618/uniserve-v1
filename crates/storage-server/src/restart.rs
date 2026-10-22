//! Hot restart: a successor server, of the same build or another, goes on
//! with everything this server holds, and no client sees a connection
//! close.
//!
//! An operator's [`Request::Restart`] starts the successor (`uniserve
//! storage serve ... --takeover FD`) with one end of a socket pair at that
//! descriptor and answers with its process id. The server serves on until
//! nothing is in flight: no request waits, no shoot-down runs, the connector
//! runs no task, and every reply was written. It then stops its connector,
//! and hands the successor its state with every descriptor it holds: the
//! listening sockets and their lock, each client's connection and pidfd,
//! each storage's memory and each page pool's memfd ([`Server::hand_over`]).
//! The successor goes on from there ([`Server::resume`]), starts its own
//! connector, says it serves, and this server exits. A client's request
//! sent meanwhile waits in its connection until the successor reads it.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ffi::OsString;
use std::io;
use std::os::fd::{AsFd as _, AsRawFd as _, BorrowedFd, FromRawFd as _, OwnedFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt as _;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use storage::protocol::{Access, Claim, Key, Role, Scope, StorageRecord};
use storage::transport::{Connection, Socket};

use crate::kv_storage::KvStorage;
use crate::pages::{HandedPools, Pools};
use crate::{ClientId, ConnectorLaunch, Deployment, Holder, Server, StorageEntry, TensorEntry};

/// The descriptor a successor finds the handoff socket at.
pub const TAKEOVER_DESCRIPTOR: i32 = 3;

/// How long the successor takes to answer.
const STEP_TIMEOUT: Duration = Duration::from_secs(60);

/// A restart under way: the successor and this server's end of the
/// handoff socket.
pub struct Pending {
    successor: std::process::Child,
    channel: UnixStream,
}

/// The handoff socket's messages.
#[derive(Debug, Serialize, Deserialize)]
enum Message {
    /// What the server holds; the descriptors travel in the order
    /// [`HandedServer`] names them.
    Takeover(Box<HandedServer>),
    /// The successor serves.
    Serving,
    /// The successor gives up, and why.
    Failed { reason: String },
}

/// A server's state, its descriptors travelling apart: first the listener
/// and its lock, then the TCP listener when there is one, then every pidfd
/// of [`HandedServer::pidfds`], every client's connection, every storage's
/// memory and every page pool's memfd.
#[derive(Debug, Serialize, Deserialize)]
struct HandedServer {
    tcp: Option<String>,
    /// How many pidfds travel; holders and clients name theirs by index.
    pidfds: usize,
    clients: Vec<HandedClient>,
    next_client: ClientId,
    storages: Vec<(u64, StorageRecord, usize)>,
    next_storage: u64,
    tensors: Vec<(Key, HandedTensor)>,
    deployments: Vec<(String, Claim, Option<ClientId>)>,
    /// The pidfds of the processes shot down that left every GPU, and of
    /// those that did not leave one in time.
    shot_finished: Vec<usize>,
    shot_failed: Vec<usize>,
    pages: HandedPools,
    host_reserve: u64,
    relaunch: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct HandedClient {
    id: ClientId,
    unix: bool,
    pid: i32,
    pidfd: Option<usize>,
    role: Option<Role>,
    /// Bytes received and not yet taken as frames.
    received: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
struct HandedTensor {
    name: String,
    storage: u64,
    offset: u64,
    dtype: String,
    shape: Vec<u64>,
    strides: Vec<u64>,
    access: Access,
    scopes: BTreeSet<Scope>,
    writer: Option<HandedHolder>,
    standby: Vec<HandedHolder>,
}

#[derive(Debug, Serialize, Deserialize)]
struct HandedHolder {
    client: Option<ClientId>,
    pid: i32,
    pidfd: usize,
    scope: Scope,
    generation: u64,
    claim: Claim,
}

/// The pidfds a handoff names, each once.
#[derive(Default)]
struct Pidfds {
    descriptors: Vec<Arc<OwnedFd>>,
}

impl Pidfds {
    fn index(&mut self, pidfd: &Arc<OwnedFd>) -> usize {
        if let Some(index) = self
            .descriptors
            .iter()
            .position(|known| Arc::ptr_eq(known, pidfd))
        {
            return index;
        }
        self.descriptors.push(Arc::clone(pidfd));
        self.descriptors.len() - 1
    }

    fn holder(&mut self, holder: &Holder) -> HandedHolder {
        HandedHolder {
            client: holder.client,
            pid: holder.pid,
            pidfd: self.index(&holder.pidfd),
            scope: holder.scope.clone(),
            generation: holder.generation,
            claim: holder.claim,
        }
    }
}

impl Server {
    /// Starts the successor of a hot restart as `executable` (this
    /// server's own when `None`), with the arguments this server was
    /// started with ([`Server::set_relaunch`]); the handoff follows once
    /// nothing is in flight.
    pub(crate) fn begin_restart(&mut self, executable: Option<&str>) -> io::Result<u32> {
        if self.restart.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "a restart is under way",
            ));
        }
        let executable = match executable {
            Some(executable) => std::path::PathBuf::from(executable),
            None => std::env::current_exe()?,
        };
        let (ours, theirs) = UnixStream::pair()?;
        let descriptor = theirs.as_raw_fd();
        let mut command = std::process::Command::new(&executable);
        command
            .args(&self.relaunch)
            .args(["--takeover", &TAKEOVER_DESCRIPTOR.to_string()]);
        // SAFETY: the closure runs in the forked child before exec and calls
        // only `dup2`, which is async-signal-safe; `theirs` stays open in
        // this process until the spawn returned.
        unsafe {
            command.pre_exec(move || {
                if libc::dup2(descriptor, TAKEOVER_DESCRIPTOR) == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let successor = command.spawn()?;
        drop(theirs);
        let pid = successor.id();
        tracing::info!(pid, executable = %executable.display(), "started the successor server");
        self.restart = Some(Pending {
            successor,
            channel: ours,
        });
        Ok(pid)
    }

    /// Whether nothing is in flight: no request waits or is held, no
    /// shoot-down runs, the connector runs no task, and every reply was
    /// written.
    pub(crate) fn quiescent(&self) -> bool {
        self.shootdowns.pending_pidfds().is_empty()
            && self.kv_storage.tasks.is_empty()
            && self.events.is_empty()
            && self.clients.values().all(|client| {
                !client.held && client.queue.is_empty() && !client.connection.wants_write()
            })
    }

    /// The arguments a successor server starts with, before
    /// `--takeover`: those of the `uniserve storage serve` that started
    /// this one, without `--daemon`.
    pub fn set_relaunch(&mut self, arguments: Vec<String>) {
        self.relaunch = arguments;
    }

    /// Hands everything to the successor of `pending`, and returns once it
    /// serves; this server holds nothing afterwards. `lock` is the
    /// listening socket's lock, which the successor holds from then on.
    ///
    /// # Errors
    ///
    /// Fails when the handoff socket fails or the successor gives up; the
    /// server's state is gone either way.
    pub fn hand_over(mut self, pending: Pending, lock: OwnedFd) -> io::Result<u32> {
        // The successor starts a connector of its own.
        self.kv_storage.stop();
        let mut pidfds = Pidfds::default();
        let mut clients = Vec::with_capacity(self.clients.len());
        let mut client_descriptors = Vec::with_capacity(self.clients.len());
        for (id, client) in std::mem::take(&mut self.clients) {
            let pidfd = client.pidfd.as_ref().map(|pidfd| pidfds.index(pidfd));
            let (socket, received) = client.connection.hand_over()?;
            let (descriptor, unix) = match socket {
                Socket::Unix(stream) => (OwnedFd::from(stream), true),
                Socket::Tcp(stream) => (OwnedFd::from(stream), false),
            };
            client_descriptors.push(descriptor);
            clients.push(HandedClient {
                id,
                unix,
                pid: client.pid,
                pidfd,
                role: client.role,
                received,
            });
        }
        let mut tensors = Vec::with_capacity(self.tensors.len());
        for (key, tensor) in std::mem::take(&mut self.tensors) {
            tensors.push((
                key,
                HandedTensor {
                    name: tensor.name,
                    storage: tensor.storage,
                    offset: tensor.offset,
                    dtype: tensor.dtype,
                    shape: tensor.shape,
                    strides: tensor.strides,
                    access: tensor.access,
                    scopes: tensor.scopes,
                    writer: tensor.writer.as_ref().map(|holder| pidfds.holder(holder)),
                    standby: tensor
                        .standby
                        .iter()
                        .map(|holder| pidfds.holder(holder))
                        .collect(),
                },
            ));
        }
        let shot_finished = self
            .shootdowns
            .finished_pidfds()
            .iter()
            .map(|pidfd| pidfds.index(pidfd))
            .collect();
        let shot_failed = self
            .shootdowns
            .failed_pidfds()
            .iter()
            .map(|pidfd| pidfds.index(pidfd))
            .collect();
        let mut storages = Vec::with_capacity(self.storages.len());
        let mut storage_descriptors = Vec::with_capacity(self.storages.len());
        for (id, storage) in std::mem::take(&mut self.storages) {
            storages.push((id, storage.record, storage.tensors));
            storage_descriptors.push(storage._descriptor);
        }
        let (pages, pool_descriptors) = self.pages.hand_over();

        let tcp = self.tcp.take();
        let mut descriptors: Vec<OwnedFd> = vec![OwnedFd::from(self.listener.try_clone()?), lock];
        if let Some((listener, _)) = &tcp {
            descriptors.push(OwnedFd::from(listener.try_clone()?));
        }
        for pidfd in &pidfds.descriptors {
            descriptors.push(pidfd.as_fd().try_clone_to_owned()?);
        }
        descriptors.extend(client_descriptors);
        descriptors.extend(storage_descriptors);
        descriptors.extend(pool_descriptors);
        let handed = HandedServer {
            tcp: tcp.map(|(_, address)| address),
            pidfds: pidfds.descriptors.len(),
            clients,
            next_client: self.next_client,
            storages,
            next_storage: self.next_storage,
            tensors,
            deployments: self
                .deployments
                .drain()
                .map(|(name, deployment)| (name, deployment.claim, deployment.head))
                .collect(),
            shot_finished,
            shot_failed,
            pages,
            host_reserve: self.host_reserve,
            relaunch: std::mem::take(&mut self.relaunch),
        };

        let Pending {
            mut successor,
            channel,
        } = pending;
        let pid = successor.id();
        let mut connection = Connection::new(channel);
        let borrowed: Vec<BorrowedFd<'_>> = descriptors.iter().map(|fd| fd.as_fd()).collect();
        connection.send(&Message::Takeover(Box::new(handed)), &borrowed)?;
        while !connection.flush()? {
            std::thread::sleep(Duration::from_millis(5));
        }
        drop(descriptors);
        let deadline = Instant::now() + STEP_TIMEOUT;
        loop {
            if let Some((message, _)) = connection.next_frame::<Message>()? {
                return match message {
                    Message::Serving => {
                        tracing::info!(pid, "the successor server serves");
                        Ok(pid)
                    }
                    Message::Failed { reason } => Err(io::Error::other(format!(
                        "the successor server gave up: {reason}"
                    ))),
                    Message::Takeover(_) => Err(io::Error::other(
                        "the successor server answered out of protocol",
                    )),
                };
            }
            if let Some(status) = successor.try_wait()? {
                return Err(io::Error::other(format!(
                    "the successor server exited: {status}"
                )));
            }
            if Instant::now() >= deadline || connection.is_closed() {
                return Err(io::Error::other("the successor server did not answer"));
            }
            std::thread::sleep(Duration::from_millis(5));
            connection.receive()?;
        }
    }

    /// Goes on with the server a predecessor hands over on the handoff
    /// socket at `descriptor`, running the connector `connector` as the
    /// predecessor did; returns the server and the listening socket's lock.
    ///
    /// # Errors
    ///
    /// Fails when the handoff fails; the predecessor then exits all the
    /// same, and its clients see their connections close.
    pub fn resume(
        descriptor: i32,
        connector: Option<ConnectorLaunch>,
    ) -> io::Result<(Self, OwnedFd)> {
        // SAFETY: the predecessor placed the handoff socket at this
        // descriptor, which nothing else in this process owns.
        let channel = UnixStream::from(unsafe { OwnedFd::from_raw_fd(descriptor) });
        let mut connection = Connection::new(channel);
        let (message, descriptors) = loop {
            if let Some(frame) = connection.next_frame::<Message>()? {
                break frame;
            }
            if connection.is_closed() {
                return Err(io::Error::other(
                    "the predecessor closed the handoff socket",
                ));
            }
            connection.receive()?;
            std::thread::sleep(Duration::from_millis(5));
        };
        let Message::Takeover(handed) = message else {
            return Err(io::Error::other("the predecessor sent no takeover"));
        };
        match Self::rebuild(*handed, descriptors, connector) {
            Ok(resumed) => {
                connection.send(&Message::Serving, &[])?;
                while !connection.flush()? {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Ok(resumed)
            }
            Err(error) => {
                let _ = connection.send(
                    &Message::Failed {
                        reason: error.to_string(),
                    },
                    &[],
                );
                let _ = connection.flush();
                Err(error)
            }
        }
    }

    fn rebuild(
        handed: HandedServer,
        descriptors: Vec<OwnedFd>,
        connector: Option<ConnectorLaunch>,
    ) -> io::Result<(Self, OwnedFd)> {
        let mut descriptors = descriptors.into_iter();
        let mut next = || {
            descriptors.next().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "the takeover lacks a descriptor",
                )
            })
        };
        let listener = UnixListener::from(next()?);
        listener.set_nonblocking(true)?;
        let lock = next()?;
        let tcp = match handed.tcp {
            Some(address) => {
                let listener = std::net::TcpListener::from(next()?);
                listener.set_nonblocking(true)?;
                Some((listener, address))
            }
            None => None,
        };
        let pidfds: Vec<Arc<OwnedFd>> = (0..handed.pidfds)
            .map(|_| next().map(Arc::new))
            .collect::<io::Result<_>>()?;
        let pidfd = |index: usize| {
            pidfds.get(index).cloned().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "a pidfd index is out of range")
            })
        };
        let mut clients = BTreeMap::new();
        for client in handed.clients {
            let descriptor = next()?;
            let socket = if client.unix {
                let stream = UnixStream::from(descriptor);
                stream.set_nonblocking(true)?;
                Socket::Unix(stream)
            } else {
                let stream = std::net::TcpStream::from(descriptor);
                stream.set_nonblocking(true)?;
                Socket::Tcp(stream)
            };
            clients.insert(
                client.id,
                crate::Client {
                    connection: Connection::resume(socket, client.received),
                    pid: client.pid,
                    pidfd: client.pidfd.map(pidfd).transpose()?,
                    role: client.role,
                    queue: VecDeque::new(),
                    held: false,
                },
            );
        }
        let holder = |holder: HandedHolder| -> io::Result<Holder> {
            Ok(Holder {
                client: holder.client,
                pid: holder.pid,
                pidfd: pidfd(holder.pidfd)?,
                scope: holder.scope,
                generation: holder.generation,
                claim: holder.claim,
            })
        };
        let mut tensors = HashMap::with_capacity(handed.tensors.len());
        for (key, tensor) in handed.tensors {
            tensors.insert(
                key,
                TensorEntry {
                    name: tensor.name,
                    storage: tensor.storage,
                    offset: tensor.offset,
                    dtype: tensor.dtype,
                    shape: tensor.shape,
                    strides: tensor.strides,
                    access: tensor.access,
                    scopes: tensor.scopes,
                    writer: tensor.writer.map(holder).transpose()?,
                    standby: tensor
                        .standby
                        .into_iter()
                        .map(holder)
                        .collect::<io::Result<_>>()?,
                },
            );
        }
        let mut storages = HashMap::with_capacity(handed.storages.len());
        for (id, record, count) in handed.storages {
            storages.insert(
                id,
                StorageEntry {
                    record,
                    _descriptor: next()?,
                    tensors: count,
                },
            );
        }
        let pool_descriptors: Vec<OwnedFd> = descriptors.collect();
        let pages = Pools::resume(handed.pages, pool_descriptors)?;
        let mut shootdowns = crate::shoot::ShootDowns::default();
        for index in handed.shot_finished {
            shootdowns.note_finished(pidfd(index)?);
        }
        for index in handed.shot_failed {
            shootdowns.note_failed(pidfd(index)?);
        }
        let server = Self {
            listener,
            tcp,
            clients,
            next_client: handed.next_client,
            storages,
            next_storage: handed.next_storage,
            tensors,
            deployments: handed
                .deployments
                .into_iter()
                .map(|(name, claim, head)| (name, Deployment { claim, head }))
                .collect(),
            shootdowns,
            pages,
            host_reserve: handed.host_reserve,
            kv_storage: KvStorage::new(connector),
            events: Vec::new(),
            relaunch: handed.relaunch,
            restart: None,
        };
        tracing::info!(
            clients = server.clients.len(),
            tensors = server.tensors.len(),
            pools = server.pages.listing().len(),
            "the server goes on from its predecessor"
        );
        Ok((server, lock))
    }
}

/// The arguments of a `uniserve storage serve` that a successor starts with:
/// `arguments` (this process's, the program's name first) without the
/// program's name, `--daemon` and an earlier `--takeover`.
pub fn relaunch_arguments(arguments: Vec<OsString>) -> Vec<String> {
    let mut kept = Vec::new();
    let mut arguments = arguments
        .into_iter()
        .skip(1)
        .map(|argument| argument.into_string().unwrap_or_default());
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--daemon" => {}
            "--takeover" => {
                arguments.next();
            }
            _ if argument.starts_with("--takeover=") => {}
            _ => kept.push(argument),
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;
    use std::os::fd::IntoRawFd as _;
    use std::sync::atomic::{AtomicBool, Ordering};

    use storage::protocol::{Mode, Placement, TensorRecord};
    use storage::{Client, socket};

    use super::*;

    /// A server running on a thread of its own until `stop`, which then
    /// hands it back.
    fn running(
        mut server: Server,
        stop: Arc<AtomicBool>,
    ) -> (std::thread::JoinHandle<Server>, UnixStream) {
        let (wake, waker) = UnixStream::pair().unwrap();
        wake.set_nonblocking(true).unwrap();
        let thread = std::thread::spawn(move || {
            let pending = server
                .run(wake.as_fd(), &|| stop.load(Ordering::Acquire))
                .unwrap();
            assert!(pending.is_none());
            server
        });
        (thread, waker)
    }

    /// A server handed over goes on serving its clients' connections, the
    /// tensors they put and its listening socket from the successor.
    #[test]
    fn a_successor_serves_what_the_server_held() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("storage.sock");
        let (listener, lock) = socket::bind(&path).unwrap().unwrap();
        let server = Server::new(listener, 64, 0, None).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let (thread, waker) = running(server, Arc::clone(&stop));

        let scope = Scope {
            deployment: "model".into(),
            group: "text".into(),
            rank: 0,
        };
        let mut head = Client::connect(
            &path,
            Role::Head {
                deployment: "model".into(),
            },
        )
        .unwrap();
        let claim = head.claim(vec![scope.clone()]).unwrap();
        let mut worker = Client::connect(
            &path,
            Role::Worker {
                scope: scope.clone(),
                generation: 1,
                claim,
            },
        )
        .unwrap();
        let mut memory = tempfile::tempfile().unwrap();
        memory.write_all(b"abcd").unwrap();
        worker
            .put(
                vec![TensorRecord {
                    key: Key([1; 32]),
                    name: "weight".into(),
                    storage: 0,
                    offset: 0,
                    dtype: "uint8".into(),
                    shape: vec![4],
                    strides: vec![1],
                    access: Access::ReadOnly,
                }],
                &[(
                    StorageRecord {
                        placement: Placement::Host { numa_node: None },
                        bytes: 4,
                    },
                    memory.as_fd(),
                )],
            )
            .unwrap();

        // Stop the server between two rounds and hand it to a successor on
        // another thread; a sleeping process stands for the successor's.
        stop.store(true, Ordering::Release);
        (&waker).write_all(b"x").unwrap();
        let server = thread.join().unwrap();
        let (ours, theirs) = UnixStream::pair().unwrap();
        let descriptor = theirs.into_raw_fd();
        let successor = std::thread::spawn(move || Server::resume(descriptor, None).unwrap());
        let placeholder = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .unwrap();
        let pending = Pending {
            successor: placeholder,
            channel: ours,
        };
        let pid = server.hand_over(pending, lock.hand_over()).unwrap();
        let (resumed, lock) = successor.join().unwrap();
        let _ = std::process::Command::new("kill")
            .arg(pid.to_string())
            .status();
        let _lock = socket::SocketLock::resume(lock, path.clone());
        let stop = Arc::new(AtomicBool::new(false));
        let (thread, waker) = running(resumed, Arc::clone(&stop));

        // The same connections go on: the worker finds its tensor, the head
        // still holds the claim, and a new client reaches the socket.
        let got = worker.get(vec![Key([1; 32])], Mode::Read).unwrap();
        assert!(got.missing.is_empty());
        assert_eq!(head.list().unwrap().deployments.len(), 1);
        let mut operator = Client::connect(&path, Role::Operator).unwrap();
        let listing = operator.list().unwrap();
        assert_eq!(listing.tensors.len(), 1);
        assert_eq!(listing.deployments.len(), 1);

        stop.store(true, Ordering::Release);
        (&waker).write_all(b"x").unwrap();
        drop(thread.join().unwrap());
    }

    #[test]
    fn a_successor_starts_with_the_serve_arguments_but_detaching() {
        let arguments = [
            "/bin/uniserve",
            "storage",
            "serve",
            "--socket",
            "/tmp/s.sock",
            "--daemon",
            "--takeover",
            "3",
            "--page-pool-gib",
            "8",
        ]
        .map(OsString::from)
        .to_vec();
        assert_eq!(
            relaunch_arguments(arguments),
            [
                "storage",
                "serve",
                "--socket",
                "/tmp/s.sock",
                "--page-pool-gib",
                "8"
            ]
        );
    }
}
