//! Hot restart: a successor server, of the same build or another, goes on
//! with everything this server holds, and no client sees a connection
//! close.
//!
//! An operator's [`Request::Restart`] starts the successor (`uniserve
//! storage-server serve ... --predecessor FD`) with one end of a socket pair at that
//! descriptor and responds with its process id. The server serves on until
//! nothing is in flight: no request waits, no shoot-down runs, neither the
//! connector nor the Transfer Server runs a task, the buffer pools have no
//! task outstanding, and every reply was written. From the restart's start
//! the buffer pools' thread takes no request and reads no `completed` word,
//! so no work starts there while the tasks end; an engine's requests wait
//! in its queue pairs meanwhile. The server then stops both children, and
//! passes the successor its state with every descriptor it holds: the
//! listening sockets and their lock, each client's connection and pidfd,
//! each storage's storage, each queue pair's memfd and each page pool's
//! memfd ([`Server::handoff`]). The buffer pools travel as images, with
//! the sessions, the parts of the host regions and where the units lie,
//! and the queue pairs keep their indices.
//! The successor goes on from there ([`Server::resume`]), starts its own
//! connector and Transfer Server, says it serves, and this server exits. A client's request
//! sent meanwhile waits in its connection until the successor reads it.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ffi::OsString;
use std::io;
use std::os::fd::{AsFd as _, AsRawFd as _, BorrowedFd, FromRawFd as _, OwnedFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt as _;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ::storage::protocol::{Access, Claim, Key, Location, Role, Scope, StorageRecord};
use ::storage::transport::{Connection, Socket};
use serde::{Deserialize, Serialize};

use crate::kv_storage::KvStorage;
use crate::pages::{Pools, PoolsHandoff};
use crate::quota::Quotas;
use crate::sessions::{Sessions, SessionsHandoff};
use crate::transfers::Transfers;
use crate::{ClientId, Deployment, Holder, Launch, Server, StorageEntry, TensorEntry};
use ::storage::protocol::Retainer;

/// The descriptor a successor finds the handoff socket at.
pub const PREDECESSOR_DESCRIPTOR: i32 = 3;

/// How long the successor takes to respond.
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
    /// [`Handoff`] names them.
    Takeover(Box<Handoff>),
    /// The successor serves.
    Serving,
    /// The successor gives up, and why.
    Failed { reason: String },
}

/// A server's state, its descriptors travelling apart: first the listener
/// and its lock, then the TCP listener when there is one, then every pidfd
/// of [`Handoff::pidfds`], every client's connection, every storage's
/// storage, every queue pair's memfd and every page pool's memfd.
#[derive(Debug, Serialize, Deserialize)]
struct Handoff {
    tcp: Option<String>,
    /// How many pidfds travel; holders and clients name theirs by index.
    pidfds: usize,
    clients: Vec<ClientHandoff>,
    next_client: ClientId,
    /// Each storage, its tensors and whether it is a host region's part.
    storages: Vec<(u64, StorageRecord, usize, bool)>,
    /// The host copies of epochs' weights, with the GPU each left
    /// ([`crate::moves`]).
    moved: Vec<(u64, [u8; 16])>,
    next_storage: u64,
    tensors: Vec<(Key, TensorHandoff)>,
    /// Each deployment's claim, its head and its epochs.
    deployments: Vec<(String, Deployment)>,
    /// The pidfds of the processes shot down that left every GPU, and of
    /// those that did not leave one in time.
    shot_finished: Vec<usize>,
    shot_failed: Vec<usize>,
    pages: PoolsHandoff,
    host_reserve: u64,
    relaunch: Vec<String>,
    /// The GPUs' totals, the epochs' limits and the reservations of the
    /// clients the handoff carries.
    quotas: Quotas,
    /// What each deployment's tasks copied, by priority.
    task_totals: Vec<(String, Vec<::storage::protocol::TaskTotal>)>,
    /// The sessions with the buffer pools and the pools' images.
    sessions: SessionsHandoff,
    /// The processes followed for a session whose engine is gone: each
    /// process's id, its pidfd's index and the GPUs it must leave.
    followed: Vec<(i32, usize, Vec<[u8; 16]>)>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ClientHandoff {
    id: ClientId,
    unix: bool,
    pid: i32,
    pidfd: Option<usize>,
    role: Option<Role>,
    /// Bytes received and not yet taken as frames.
    received: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TensorHandoff {
    name: String,
    storage: u64,
    offset: u64,
    dtype: String,
    shape: Vec<u64>,
    strides: Vec<u64>,
    access: Access,
    retainers: BTreeSet<Retainer>,
    /// The deployment and epoch that published the tensor.
    publisher: (String, u64),
    writer: Option<HolderHandoff>,
    standby: Vec<HolderHandoff>,
}

#[derive(Debug, Serialize, Deserialize)]
struct HolderHandoff {
    client: Option<ClientId>,
    pid: i32,
    pidfd: usize,
    scope: Scope,
    generation: u64,
    claim: Claim,
    epoch: u64,
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

    fn holder(&mut self, holder: &Holder) -> HolderHandoff {
        HolderHandoff {
            client: holder.client,
            pid: holder.pid,
            pidfd: self.index(&holder.pidfd),
            scope: holder.scope.clone(),
            generation: holder.generation,
            claim: holder.claim,
            epoch: holder.epoch,
        }
    }
}

impl Server {
    /// Starts the successor of a restart in place as `executable` (this
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
            .args(["--predecessor", &PREDECESSOR_DESCRIPTOR.to_string()]);
        // SAFETY: the closure runs in the forked child before exec and calls
        // only `dup2`, which is async-signal-safe; `theirs` stays open in
        // this process until the spawn returned.
        unsafe {
            command.pre_exec(move || {
                if libc::dup2(descriptor, PREDECESSOR_DESCRIPTOR) == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let successor = command.spawn()?;
        drop(theirs);
        let pid = successor.id();
        tracing::info!(pid, executable = %executable.display(), "started the successor server");
        // The buffer pools start no work from now on, and their tasks end.
        self.sessions.thread.send(|pools| pools.freeze());
        self.restart = Some(Pending {
            successor,
            channel: ours,
        });
        Ok(pid)
    }

    /// Whether nothing is in flight: no request waits or is held, no
    /// shoot-down runs, neither the connector nor the Transfer Server runs a
    /// task, the buffer pools have no task outstanding, and every reply was
    /// written. The processes followed for a session pass over.
    pub(crate) fn quiescent(&self) -> bool {
        !self.shootdowns.killing()
            && self.kv_storage.tasks.is_empty()
            && self.transfers.tasks.is_empty()
            && self.events.is_empty()
            && self.clients.values().all(|client| {
                !client.held && client.queue.is_empty() && !client.connection.wants_write()
            })
            && self.sessions.quiet()
    }

    /// The arguments a successor server starts with, before
    /// `--predecessor`: those of the `uniserve storage-server serve` that started
    /// this one, without `--daemon`.
    pub fn set_relaunch(&mut self, arguments: Vec<String>) {
        self.relaunch = arguments;
    }

    /// Passes everything to the successor of `pending`, and returns once it
    /// serves; this server holds nothing afterwards. `lock` is the
    /// listening socket's lock, which the successor holds from then on.
    ///
    /// # Errors
    ///
    /// Fails when the handoff socket fails or the successor gives up; the
    /// server's state is gone either way.
    pub fn handoff(mut self, pending: Pending, lock: OwnedFd) -> io::Result<u32> {
        // The successor starts a connector and a Transfer Server of its own.
        self.kv_storage.process.stop();
        self.transfers.process.stop();
        let mut pidfds = Pidfds::default();
        let mut clients = Vec::with_capacity(self.clients.len());
        let mut client_descriptors = Vec::with_capacity(self.clients.len());
        for (id, client) in std::mem::take(&mut self.clients) {
            let pidfd = client.pidfd.as_ref().map(|pidfd| pidfds.index(pidfd));
            let (socket, received) = client.connection.handoff()?;
            let (descriptor, unix) = match socket {
                Socket::Unix(stream) => (OwnedFd::from(stream), true),
                Socket::Tcp(stream) => (OwnedFd::from(stream), false),
            };
            client_descriptors.push(descriptor);
            clients.push(ClientHandoff {
                id,
                unix,
                pid: client.pid,
                pidfd,
                role: client.role,
                received,
            });
        }
        let mut tensors = Vec::with_capacity(self.tensors.len());
        self.copies.clear();
        for ((key, _), tensor) in std::mem::take(&mut self.tensors) {
            tensors.push((
                key,
                TensorHandoff {
                    name: tensor.name,
                    storage: tensor.storage,
                    offset: tensor.offset,
                    dtype: tensor.dtype,
                    shape: tensor.shape,
                    strides: tensor.strides,
                    access: tensor.access,
                    retainers: tensor.retainers,
                    publisher: tensor.publisher,
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
        let mut moved = Vec::new();
        for (id, storage) in std::mem::take(&mut self.storages) {
            storages.push((id, storage.record, storage.tensors, storage.part));
            moved.extend(storage.moved_from.map(|gpu| (id, gpu)));
            storage_descriptors.push(storage._descriptor);
        }
        let (pages, pool_descriptors) = self.pages.handoff();
        let followed = self
            .shootdowns
            .followed()
            .into_iter()
            .map(|(pid, pidfd, gpus)| (pid, pidfds.index(&pidfd), gpus))
            .collect();
        let (sessions, queue_descriptors) = self.sessions.hand_off(|pidfd| pidfds.index(pidfd))?;

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
        descriptors.extend(queue_descriptors);
        descriptors.extend(pool_descriptors);
        let handoff = Handoff {
            tcp: tcp.map(|(_, address)| address),
            pidfds: pidfds.descriptors.len(),
            clients,
            next_client: self.next_client,
            storages,
            moved,
            next_storage: self.next_storage,
            tensors,
            deployments: self.deployments.drain().collect(),
            shot_finished,
            shot_failed,
            pages,
            host_reserve: self.host_reserve,
            relaunch: std::mem::take(&mut self.relaunch),
            quotas: std::mem::take(&mut self.quotas),
            task_totals: self.transfers.task_totals(),
            sessions,
            followed,
        };

        let Pending {
            mut successor,
            channel,
        } = pending;
        let pid = successor.id();
        let mut connection = Connection::new(channel);
        let borrowed: Vec<BorrowedFd<'_>> = descriptors.iter().map(|fd| fd.as_fd()).collect();
        connection.send(&Message::Takeover(Box::new(handoff)), &borrowed)?;
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
                        "the successor server responded out of protocol",
                    )),
                };
            }
            if let Some(status) = successor.try_wait()? {
                return Err(io::Error::other(format!(
                    "the successor server exited: {status}"
                )));
            }
            if Instant::now() >= deadline || connection.is_closed() {
                return Err(io::Error::other("the successor server did not respond"));
            }
            std::thread::sleep(Duration::from_millis(5));
            connection.receive()?;
        }
    }

    /// Goes on with the server a predecessor passed on over the handoff
    /// socket at `descriptor`, running the connector `connector` and the
    /// Transfer Server `transfer` as the predecessor did; returns the server
    /// and the listening socket's lock.
    ///
    /// # Errors
    ///
    /// Fails when the handoff fails; the predecessor then exits all the
    /// same, and its clients see their connections close.
    pub fn resume(
        descriptor: i32,
        connector: Option<Launch>,
        transfer: Option<Launch>,
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
        let Message::Takeover(handoff) = message else {
            return Err(io::Error::other("the predecessor sent no takeover"));
        };
        match Self::rebuild(*handoff, descriptors, connector, transfer) {
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
        handoff: Handoff,
        descriptors: Vec<OwnedFd>,
        connector: Option<Launch>,
        transfer: Option<Launch>,
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
        let tcp = match handoff.tcp {
            Some(address) => {
                let listener = std::net::TcpListener::from(next()?);
                listener.set_nonblocking(true)?;
                Some((listener, address))
            }
            None => None,
        };
        let pidfds: Vec<Arc<OwnedFd>> = (0..handoff.pidfds)
            .map(|_| next().map(Arc::new))
            .collect::<io::Result<_>>()?;
        let pidfd = |index: usize| {
            pidfds.get(index).cloned().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "a pidfd index is out of range")
            })
        };
        let mut clients = BTreeMap::new();
        for client in handoff.clients {
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
        let holder = |holder: HolderHandoff| -> io::Result<Holder> {
            Ok(Holder {
                client: holder.client,
                pid: holder.pid,
                pidfd: pidfd(holder.pidfd)?,
                scope: holder.scope,
                generation: holder.generation,
                claim: holder.claim,
                epoch: holder.epoch,
            })
        };
        // A tensor lies where its storage lies.
        let locations: HashMap<u64, Location> = handoff
            .storages
            .iter()
            .map(|(id, record, _, _)| (*id, record.location))
            .collect();
        let mut tensors = HashMap::with_capacity(handoff.tensors.len());
        let mut copies: HashMap<Key, Vec<Location>> = HashMap::new();
        for (key, tensor) in handoff.tensors {
            let location = *locations.get(&tensor.storage).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("the handoff's tensor {} names no storage", tensor.name),
                )
            })?;
            copies.entry(key).or_default().push(location);
            tensors.insert(
                (key, location),
                TensorEntry {
                    name: tensor.name,
                    storage: tensor.storage,
                    offset: tensor.offset,
                    dtype: tensor.dtype,
                    shape: tensor.shape,
                    strides: tensor.strides,
                    access: tensor.access,
                    retainers: tensor.retainers,
                    publisher: tensor.publisher,
                    writer: tensor.writer.map(holder).transpose()?,
                    standby: tensor
                        .standby
                        .into_iter()
                        .map(holder)
                        .collect::<io::Result<_>>()?,
                },
            );
        }
        let moved: HashMap<u64, [u8; 16]> = handoff.moved.into_iter().collect();
        let mut storages = HashMap::with_capacity(handoff.storages.len());
        for (id, record, count, part) in handoff.storages {
            storages.insert(
                id,
                StorageEntry {
                    record,
                    _descriptor: next()?,
                    tensors: count,
                    moved_from: moved.get(&id).copied(),
                    moving: false,
                    part,
                },
            );
        }
        let queues = (0..handoff.sessions.queues)
            .map(|_| next())
            .collect::<io::Result<Vec<_>>>()?;
        let sessions = Sessions::resume(handoff.sessions, queues, pidfd)?;
        let pool_descriptors: Vec<OwnedFd> = descriptors.collect();
        let pages = Pools::resume(handoff.pages, pool_descriptors)?;
        let mut shootdowns = crate::shoot::ShootDowns::default();
        for index in handoff.shot_finished {
            shootdowns.note_finished(pidfd(index)?);
        }
        for index in handoff.shot_failed {
            shootdowns.note_failed(pidfd(index)?);
        }
        for (pid, index, gpus) in handoff.followed {
            shootdowns.follow(pid, pidfd(index)?, gpus);
        }
        let server = Self {
            listener,
            tcp,
            clients,
            next_client: handoff.next_client,
            storages,
            next_storage: handoff.next_storage,
            tensors,
            copies,
            deployments: handoff.deployments.into_iter().collect(),
            shootdowns,
            pages,
            host_reserve: handoff.host_reserve,
            quotas: handoff.quotas,
            kv_storage: KvStorage::new(connector),
            transfers: {
                let mut transfers = Transfers::new(transfer);
                transfers.resume_totals(handoff.task_totals);
                transfers
            },
            moves: crate::moves::Moves::default(),
            events: Vec::new(),
            relaunch: handoff.relaunch,
            restart: None,
            sessions,
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

/// The arguments of a `uniserve storage-server serve` that a successor starts with:
/// `arguments` (this process's, the program's name first) without the
/// program's name, `--daemon` and an earlier `--predecessor`.
pub fn relaunch_arguments(arguments: Vec<OsString>) -> Vec<String> {
    let mut kept = Vec::new();
    let mut arguments = arguments
        .into_iter()
        .skip(1)
        .map(|argument| argument.into_string().unwrap_or_default());
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--daemon" => {}
            "--predecessor" => {
                arguments.next();
            }
            _ if argument.starts_with("--predecessor=") => {}
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

    use ::storage::Client;
    use ::storage::protocol::{Item, Location, Mode, TensorRecord};

    use super::*;

    /// A server running on a thread of its own until `stop`, which then
    /// passes it back.
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

    /// A server passed on goes on serving its clients' connections, the
    /// tensors they put and its listening socket from the successor.
    #[test]
    fn a_successor_serves_what_the_server_held() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("storage.sock");
        let (listener, lock) = service::bind(&path).unwrap().unwrap();
        let server = Server::new(listener, 64, 0, None, None).unwrap();
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
                epoch: 0,
            },
        )
        .unwrap();
        let claim = head.claim(0, vec![scope.clone()], Vec::new()).unwrap();
        head.publish(vec![0]).unwrap();
        let mut worker = Client::connect(
            &path,
            Role::Worker {
                scope: scope.clone(),
                generation: 1,
                claim,
                epoch: 0,
            },
        )
        .unwrap();
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"abcd").unwrap();
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
                        location: Location::Host { numa_node: None },
                        bytes: 4,
                        segment: None,
                        fabric: None,
                    },
                    file.as_fd(),
                )],
            )
            .unwrap();
        // The worker reserves its process storage on a GPU under a quota.
        let gpu = [9; 16];
        worker
            .reserve(gpu, 1000, Some(700), vec![(Item::Process, 300)])
            .unwrap();

        // Committed replicas share their outputs; a candidate's output is
        // still private when both kinds cross the restart.
        for epoch in [2, 3] {
            head.claim(epoch, vec![scope.clone()], Vec::new()).unwrap();
            let mut replica = Client::connect(
                &path,
                Role::Worker {
                    scope: scope.clone(),
                    generation: 1,
                    claim,
                    epoch,
                },
            )
            .unwrap();
            replica
                .put(
                    vec![TensorRecord {
                        key: Key([epoch as u8; 32]),
                        name: "replica_weight".into(),
                        storage: 0,
                        offset: 0,
                        dtype: "uint8".into(),
                        shape: vec![4],
                        strides: vec![1],
                        access: Access::ReadOnly,
                    }],
                    &[(
                        StorageRecord {
                            location: Location::Host { numa_node: None },
                            bytes: 4,
                            segment: None,
                            fabric: None,
                        },
                        file.as_fd(),
                    )],
                )
                .unwrap();
        }
        head.publish(vec![2]).unwrap();

        // Stop the server between two rounds and hand it to a successor on
        // another thread; a sleeping process stands for the successor's.
        stop.store(true, Ordering::Release);
        (&waker).write_all(b"x").unwrap();
        let server = thread.join().unwrap();
        let (ours, theirs) = UnixStream::pair().unwrap();
        let descriptor = theirs.into_raw_fd();
        let successor = std::thread::spawn(move || Server::resume(descriptor, None, None).unwrap());
        let placeholder = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .unwrap();
        let pending = Pending {
            successor: placeholder,
            channel: ours,
        };
        let pid = server.handoff(pending, lock.hand_over()).unwrap();
        let (resumed, lock) = successor.join().unwrap();
        let _ = std::process::Command::new("kill")
            .arg(pid.to_string())
            .status();
        let _lock = service::SocketLock::resume(lock, path.clone());
        let stop = Arc::new(AtomicBool::new(false));
        let (thread, waker) = running(resumed, Arc::clone(&stop));

        // The same connections go on: the worker finds its tensor, the head
        // still holds the claim, and a new client reaches the socket.
        let (got, _) = worker
            .get(vec![Key([1; 32])], Mode::Read, Vec::new())
            .unwrap();
        assert!(got.missing.is_empty());
        assert_eq!(head.list().unwrap().deployments.len(), 1);
        let mut operator = Client::connect(&path, Role::Operator).unwrap();
        let listing = operator.list().unwrap();
        assert_eq!(listing.tensors.len(), 3);
        assert_eq!(listing.deployments.len(), 1);
        let (got, _) = worker
            .get(vec![Key([2; 32]), Key([3; 32])], Mode::Read, Vec::new())
            .unwrap();
        assert_eq!(got.missing, vec![Key([3; 32])]);
        head.publish(vec![3]).unwrap();
        assert!(
            worker
                .get(vec![Key([3; 32])], Mode::Read, Vec::new())
                .unwrap()
                .0
                .missing
                .is_empty()
        );
        // The quota goes on too: its quota and the worker's reservation, which
        // the successor holds against a reservation past the quota.
        let quotas = worker.usage(vec![gpu]).unwrap().quotas;
        assert_eq!(quotas.len(), 1);
        assert_eq!((quotas[0].limit, quotas[0].usage()), (Some(700), 300));
        let refused = worker.reserve(gpu, 1000, None, vec![(Item::Staging, 401)]);
        assert!(
            matches!(
                refused,
                Err(::storage::Error::Server {
                    kind: ::storage::protocol::ErrorKind::Quota,
                    ..
                })
            ),
            "{refused:?}"
        );

        stop.store(true, Ordering::Release);
        (&waker).write_all(b"x").unwrap();
        drop(thread.join().unwrap());
    }

    /// The next completion on the engine's end of a queue pair, within ten
    /// seconds.
    fn completion(
        pair: &mut ::storage::QueuePair,
    ) -> (
        ::storage::runtime::RequestId,
        ::storage::runtime::Completion,
    ) {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut record = Vec::new();
        loop {
            if pair.next_completion(&mut record).unwrap() {
                return ::storage::runtime::Completion::decode(&record).unwrap();
            }
            assert!(Instant::now() < deadline, "no completion arrived");
            std::thread::sleep(Duration::from_micros(100));
        }
    }

    /// An engine's session goes on across a restart in place: its queue
    /// pair keeps its indices, and a commit taken before the restart takes
    /// effect in the successor when its stamp is reached.
    #[test]
    fn a_session_survives_a_restart_with_a_request_deferred_across_it() {
        use ::storage::runtime::{
            Class, Completion, Group, GroupId, Open, OwnerId, Region, Request, RequestId, Tier,
            VERSION,
        };

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("storage.sock");
        let (listener, lock) = service::bind(&path).unwrap().unwrap();
        let server = Server::new(listener, 64, 0, None, None).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let (thread, waker) = running(server, Arc::clone(&stop));

        let region = |name: &str, units| Region {
            name: name.into(),
            tier: Tier::Device,
            space: name.into(),
            unit_bytes: 64,
            reserved: 1,
            units,
            limit: units + 1,
        };
        let open = Open {
            version: VERSION,
            epoch: 0,
            trial: false,
            regions: vec![region("units", 4), region("rows", 2)],
            classes: vec![Class {
                units: 1,
                host: None,
                state_in_slot: false,
            }],
            trajectory: 0,
            groups: vec![Group {
                id: GroupId(0),
                memory: "gpu0".into(),
                slots: 1,
                regions: vec![vec![0]],
            }],
            schedulers: vec![vec![GroupId(0)]],
        };
        let mut engine = Client::connect(
            &path,
            Role::Head {
                deployment: "model".into(),
                epoch: 0,
            },
        )
        .unwrap();
        let (opened, descriptors) = engine.open_pool(open).unwrap();
        let mut pair =
            ::storage::QueuePair::open(descriptors.into_iter().next().unwrap(), true).unwrap();
        let completed = opened.registers[0].completed[0] as usize;
        let submit = |pair: &mut ::storage::QueuePair, number, request: Request| {
            let mut record = Vec::new();
            request.encode(RequestId(number), &mut record);
            assert!(pair.submit(&record).unwrap());
        };
        submit(
            &mut pair,
            1,
            Request::Attach {
                owner: OwnerId(1),
                group: GroupId(0),
            },
        );
        submit(
            &mut pair,
            2,
            Request::Allocate {
                owner: OwnerId(1),
                class: 0,
                at: 0,
                count: 1,
                units: 0,
            },
        );
        submit(
            &mut pair,
            3,
            Request::Commit {
                owner: OwnerId(1),
                class: 0,
                at: 0,
                key: Key([7; 32]),
                tokens: 64,
                after: vec![(GroupId(0), 2)],
            },
        );
        assert_eq!(
            completion(&mut pair),
            (RequestId(1), Completion::Attached { slot: 1 })
        );
        assert!(matches!(
            completion(&mut pair),
            (RequestId(2), Completion::Allocated { .. })
        ));

        // The restart: the commit waits for batch 2 in the pool handed over.
        stop.store(true, Ordering::Release);
        (&waker).write_all(b"x").unwrap();
        let server = thread.join().unwrap();
        let (ours, theirs) = UnixStream::pair().unwrap();
        let descriptor = theirs.into_raw_fd();
        let successor = std::thread::spawn(move || Server::resume(descriptor, None, None).unwrap());
        let placeholder = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .unwrap();
        let pending = Pending {
            successor: placeholder,
            channel: ours,
        };
        let pid = server.handoff(pending, lock.hand_over()).unwrap();
        let (resumed, lock) = successor.join().unwrap();
        let _ = std::process::Command::new("kill")
            .arg(pid.to_string())
            .status();
        let _lock = service::SocketLock::resume(lock, path.clone());
        let stop = Arc::new(AtomicBool::new(false));
        let (thread, waker) = running(resumed, Arc::clone(&stop));

        // The executor completes batch 2 after the restart, and submits more.
        pair.register(completed).store(2, Ordering::Release);
        assert_eq!(completion(&mut pair), (RequestId(3), Completion::Committed));
        submit(
            &mut pair,
            4,
            Request::Detach {
                owner: OwnerId(1),
                after: Vec::new(),
            },
        );
        assert_eq!(completion(&mut pair), (RequestId(4), Completion::Done));
        let listing = engine.list().unwrap();
        let groups = listing.buffer_pools.iter().flat_map(|pool| &pool.groups);
        assert_eq!(groups.map(|group| group.completed).collect::<Vec<_>>(), [2]);
        engine.close_pool().unwrap();

        stop.store(true, Ordering::Release);
        (&waker).write_all(b"x").unwrap();
        drop(thread.join().unwrap());
    }

    #[test]
    fn a_successor_starts_with_the_serve_arguments_but_detaching() {
        let arguments = [
            "/bin/uniserve",
            "storage-server",
            "serve",
            "--socket",
            "/tmp/s.sock",
            "--daemon",
            "--predecessor",
            "3",
            "--page-pool-gib",
            "8",
        ]
        .map(OsString::from)
        .to_vec();
        assert_eq!(
            relaunch_arguments(arguments),
            [
                "storage-server",
                "serve",
                "--socket",
                "/tmp/s.sock",
                "--page-pool-gib",
                "8"
            ]
        );
    }
}
