//! Hot restart: a successor server goes on with everything this server holds, and no client
//! sees a connection close.
//!
//! An operator's [`Request::Restart`] passes the executable the successor runs (the build of the
//! operator's command) as a descriptor, with the arguments that select its `storage-server`
//! command. The server starts the successor from that image (`storage-server serve --predecessor
//! FD`) with one end of a socket pair at that descriptor, responds with its process id, and
//! states at once the versions it speaks: its protocol version, which every connection speaks,
//! its version of the buffer pool's runtime interface, which every session speaks, and the
//! format of the buffer pools' image. A successor of a build that speaks another of them says so
//! and ends: the protocol has one version per release, so a successor of another release could
//! serve none of the connections. The restart is then given up, as it is when the successor ends
//! before it answers, and the server goes on with everything it holds. The server waits for the
//! answer on the handoff socket and the successor's pidfd in its loop, serving meanwhile.
//!
//! From the restart's start the buffer pools' thread takes no request and reads no `completed`
//! word, so no work starts there while the tasks end; an engine's requests wait in its queue
//! pairs meanwhile. Once the successor speaks the versions and nothing is in flight (no request
//! waits, no shoot-down runs, neither the connector nor the Transfer Server runs a task, the
//! buffer pools have no task outstanding, and every reply was written), the server stops both
//! children and passes the successor its state with every descriptor it holds: the listening
//! sockets and their lock, the socket's path and the settings it serves with, each client's
//! connection and pidfd, each storage's storage and each queue pair's memfd and doorbells
//! ([`Server::handoff`]). The buffer pools travel as images, with the sessions, the parts of the
//! host regions and where the units lie, and the queue pairs keep their indices. The successor
//! goes on from there ([`Server::resume`]), serving every connection and session, starts its own
//! connector and Transfer Server, says it serves, and this server exits. A client's request sent
//! meanwhile waits in its connection until the successor reads it.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io;
use std::os::fd::{AsFd as _, AsRawFd as _, BorrowedFd, OwnedFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use ::storage::protocol::{Access, Claim, Key, Location, Role, Scope, StorageRecord, VERSION};
use ::storage::runtime;
use rustix::event::{PollFd, PollFlags, poll};
use serde::{Deserialize, Serialize};
use service::{Connection, Handed};

use crate::buffer_pool::FORMAT;
use crate::child::Program;
use crate::kv_storage::KvStorage;
use crate::quota::Quotas;
use crate::sessions::{Sessions, SessionsHandoff};
use crate::transfers::Transfers;
use crate::{ClientId, Deployment, Holder, Server, StorageEntry, TensorEntry};
use ::storage::protocol::Retainer;

/// A restart under way: the successor and this server's end of the handoff socket.
pub struct Pending {
    successor: Successor,
    channel: Connection,
    /// The settings the successor serves with: the operator's, or this server's.
    settings: ::storage::Settings,
    /// Whether the successor said it speaks this server's versions: the server passes itself
    /// on once nothing is in flight from then on.
    agreed: bool,
}

/// A restart's successor process, by its pidfd: readable once the process ended, and the handle
/// that ends and reaps it when the restart is given up.
struct Successor {
    pid: u32,
    pidfd: OwnedFd,
}

impl Successor {
    /// Whether the process ended.
    fn ended(&self) -> bool {
        let mut fds = [PollFd::new(&self.pidfd, PollFlags::IN)];
        let zero = rustix::time::Timespec::default();
        matches!(poll(&mut fds, Some(&zero)), Ok(ready) if ready > 0)
    }

    /// Ends the process and reaps it.
    fn end(self) {
        let _ = rustix::process::pidfd_send_signal(&self.pidfd, rustix::process::Signal::KILL);
        let _ = rustix::process::waitid(
            rustix::process::WaitId::PidFd(self.pidfd.as_fd()),
            rustix::process::WaitIdOptions::EXITED,
        );
    }
}

/// What a restart came to in the server's loop.
pub(crate) enum Answer {
    /// The successor has not answered yet.
    Waiting,
    /// It speaks the server's versions.
    Agreed,
    /// It does not, ended, or answered out of protocol: the restart is given up.
    Refused(String),
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
    /// Before the takeover: the protocol version the connections speak, the
    /// runtime version the sessions speak, and the format of the buffer
    /// pools' image.
    Versions {
        protocol: u32,
        runtime: u32,
        image: u32,
    },
    /// The successor speaks the versions and reads the image: the takeover
    /// follows.
    Speaks,
}

/// The handoff socket's messages are the Storage Server protocol's: MessagePack with named fields
/// in the workspace's frames.
impl service::Message for Message {
    fn encode(&self) -> io::Result<Vec<u8>> {
        ::storage::protocol::encode(self)
    }

    fn decode(payload: &[u8]) -> io::Result<Self> {
        ::storage::protocol::decode(payload)
    }
}

/// A server's state, its descriptors travelling apart: first the listener
/// and its lock, then the TCP listener when there is one, then every pidfd
/// of [`Handoff::pidfds`], every client's connection, every storage's
/// storage and every queue pair's memfd and doorbells.
#[derive(Debug, Serialize, Deserialize)]
struct Handoff {
    /// The path the listening socket is bound to.
    socket: PathBuf,
    /// What the server serves with.
    settings: ::storage::Settings,
    /// The address the TCP listener for other hosts is bound to, whose descriptor travels.
    tcp: Option<std::net::SocketAddr>,
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
    /// Each deployment's KV program as its head set it, whose state the
    /// server saved before it handed over: the successor restores it.
    programs: Vec<(String, ::storage::protocol::KvProgram)>,
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

/// Whether this build speaks the predecessor's protocol and runtime versions
/// and reads its image's format; the reason it does not otherwise.
fn speaks(protocol: u32, sessions: u32, image: u32) -> Result<(), String> {
    if protocol != VERSION {
        return Err(format!(
            "connections speak protocol {protocol}; the successor speaks {VERSION}"
        ));
    }
    if sessions != runtime::VERSION {
        return Err(format!(
            "sessions speak version {sessions} of the buffer pool's interface; the successor \
             speaks {}",
            runtime::VERSION
        ));
    }
    if image != FORMAT {
        return Err(format!(
            "the buffer pools' image is of format {image}; the successor reads {FORMAT}"
        ));
    }
    Ok(())
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

/// A server a restart's successor goes on with, with what it serves on and with.
pub struct Resumed {
    pub server: Server,
    /// The listening socket's lock, which the successor holds from now on.
    pub lock: OwnedFd,
    /// The path the listening socket is bound to.
    pub socket: PathBuf,
    /// What the server serves with.
    pub settings: ::storage::Settings,
}

impl Server {
    /// Starts the successor of a restart in place from `image`, the executable an operator
    /// passed, whose `storage-server` command `arguments` select, and states this server's
    /// versions to it; the handoff follows once it speaks them and nothing is in flight.
    pub(crate) fn begin_restart(
        &mut self,
        image: Option<OwnedFd>,
        arguments: &[String],
        settings: Option<::storage::Settings>,
    ) -> io::Result<u32> {
        if self.restart.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "a restart is under way",
            ));
        }
        let image = image.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "a restart passes the executable its successor runs",
            )
        })?;
        let program = Program::of_image(image, arguments.to_vec());
        let (ours, theirs) = UnixStream::pair()?;
        let (process, pidfd) = program.spawn(
            &crate::command::successor_entry(theirs.as_raw_fd()),
            &[theirs.as_fd()],
            Stdio::null(),
            &[],
        )?;
        drop(theirs);
        let pid = process.id();
        let pidfd = Arc::try_unwrap(pidfd)
            .map_err(|_| io::Error::other("the successor's pidfd is shared"))?;
        tracing::info!(pid, "started the successor server");
        let settings = settings.unwrap_or_else(|| self.settings.clone());
        let pending = Pending::new(Successor { pid, pidfd }, Connection::new(ours)?, settings);
        // The buffer pools start no work from now on, and their tasks end.
        self.sessions.thread.send(|pools| pools.freeze());
        self.restart = Some(pending);
        Ok(pid)
    }

    /// Reads the restart's handoff socket after it or the successor's pidfd turned readable:
    /// a successor that speaks this server's versions is agreed, and one that refuses them or
    /// ended is given up, the buffer pools taking requests again and the server going on.
    pub(crate) fn read_restart(&mut self) {
        let Some(pending) = self.restart.as_mut() else {
            return;
        };
        match pending.answer() {
            Answer::Waiting => {}
            Answer::Agreed => pending.agreed = true,
            Answer::Refused(reason) => {
                tracing::error!(
                    pid = pending.successor.pid,
                    %reason,
                    "the restart in place is given up: the successor does not take the server \
                     over, which goes on"
                );
                if let Some(pending) = self.restart.take() {
                    pending.successor.end();
                }
                self.sessions.thread.send(|pools| pools.thaw());
            }
        }
    }

    /// Whether the restart under way may pass the server on now: its successor speaks the
    /// server's versions, and nothing is in flight.
    ///
    /// # Errors
    ///
    /// The buffer pools' thread ended ([`Self::quiescent`]).
    pub(crate) fn ready_to_hand_over(&self) -> io::Result<bool> {
        let agreed = self.restart.as_ref().is_some_and(|pending| pending.agreed);
        Ok(agreed && self.quiescent()?)
    }

    /// The descriptors the loop waits on for the restart under way: its handoff socket and its
    /// successor's pidfd.
    pub(crate) fn restart_descriptors(&self) -> Option<[BorrowedFd<'_>; 2]> {
        self.restart
            .as_ref()
            .map(|pending| [pending.channel.as_fd(), pending.successor.pidfd.as_fd()])
    }

    /// Whether nothing is in flight: no request waits or is held, no
    /// shoot-down runs, neither the connector nor the Transfer Server runs a
    /// task (nor a lost connector may still run one), the buffer pools have
    /// no task outstanding, and every reply was
    /// written. The processes followed for a session pass over.
    ///
    /// # Errors
    ///
    /// The buffer pools' thread ended, which leaves nothing to hand over.
    pub(crate) fn quiescent(&self) -> io::Result<bool> {
        Ok(!self.shootdowns.killing()
            && self.kv_storage.tasks.is_empty()
            && self.kv_storage.lost.is_empty()
            && self.transfers.tasks.is_empty()
            && self.events.is_empty()
            && self.clients.values().all(|client| {
                !client.held && client.queue.is_empty() && !client.connection.wants_write()
            })
            && self.sessions.quiet()?)
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
        // The successor starts a connector and a Transfer Server of its own; the channels of
        // these end with them.
        self.kv_storage.process.stop();
        self.transfers.process.stop();
        self.clients.retain(|_, client| client.child.is_none());
        let socket = self
            .listener
            .local_addr()?
            .as_pathname()
            .map(std::path::Path::to_path_buf)
            .ok_or_else(|| io::Error::other("the listening socket is bound to no path"))?;
        let mut pidfds = Pidfds::default();
        let mut clients = Vec::with_capacity(self.clients.len());
        let mut client_descriptors = Vec::with_capacity(self.clients.len());
        for (id, client) in std::mem::take(&mut self.clients) {
            let pidfd = client.pidfd.as_ref().map(|pidfd| pidfds.index(pidfd));
            let Handed {
                descriptor,
                unix,
                received,
            } = client.connection.hand_over()?;
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
        let followed = self
            .shootdowns
            .followed()
            .into_iter()
            .map(|(pid, pidfd, gpus)| (pid, pidfds.index(&pidfd), gpus))
            .collect();
        let (sessions, queue_descriptors) = self.sessions.hand_off(|pidfd| pidfds.index(pidfd))?;

        // A listener for other hosts the successor's settings state otherwise is the successor's
        // to bind; this one closes with this server.
        let tcp = self
            .tcp
            .take()
            .filter(|_| pending.settings.server.tcp == self.settings.server.tcp);
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
        let handoff = Handoff {
            socket,
            settings: pending.settings.clone(),
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
            quotas: std::mem::take(&mut self.quotas),
            task_totals: self.transfers.task_totals(),
            sessions,
            followed,
            programs: {
                self.programs.save_all();
                self.programs.setups()
            },
        };

        let Pending {
            successor,
            channel: mut connection,
            ..
        } = pending;
        let borrowed: Vec<BorrowedFd<'_>> = descriptors.iter().map(|fd| fd.as_fd()).collect();
        connection.send_with(&Message::Takeover(Box::new(handoff)), &borrowed)?;
        connection.flush()?;
        drop(descriptors);
        // The successor answers once it rebuilt the server, or ends: the wait is on its answer
        // and on its pidfd.
        loop {
            match connection.try_recv::<Message>() {
                Ok(Some(Message::Serving)) => {
                    tracing::info!(pid = successor.pid, "the successor server serves");
                    return Ok(successor.pid);
                }
                Ok(Some(Message::Failed { reason })) => {
                    return Err(io::Error::other(format!(
                        "the successor server gave up: {reason}"
                    )));
                }
                Ok(Some(_)) => {
                    return Err(io::Error::other(
                        "the successor server responded out of protocol",
                    ));
                }
                Ok(None) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                    return Err(io::Error::other(
                        "the successor server closed the handoff socket",
                    ));
                }
                Err(error) => return Err(error),
            }
            if successor.ended() {
                return Err(io::Error::other(
                    "the successor server exited before it served",
                ));
            }
            let mut fds = [
                PollFd::new(&connection, PollFlags::IN),
                PollFd::new(&successor.pidfd, PollFlags::IN),
            ];
            match poll(&mut fds, None) {
                Ok(_) | Err(rustix::io::Errno::INTR) => {}
                Err(error) => return Err(error.into()),
            }
        }
    }

    /// Goes on with the server a predecessor passes on over the handoff socket `handoff`,
    /// running its children from `program` with the settings the predecessor served with.
    ///
    /// # Errors
    ///
    /// Fails when the handoff fails; the predecessor then gives the restart up, or, once it
    /// handed its state over, exits all the same, and its clients see their connections close.
    pub fn resume(handoff: OwnedFd, program: Option<Program>) -> io::Result<Resumed> {
        let mut connection = Connection::new(UnixStream::from(handoff))?;
        // The predecessor states the versions it speaks first.
        let Message::Versions {
            protocol,
            runtime: sessions,
            image,
        } = receive(&mut connection)?.0
        else {
            return Err(io::Error::other(
                "the predecessor stated no versions before its takeover",
            ));
        };
        if let Err(reason) = speaks(protocol, sessions, image) {
            let _ = connection.send(&Message::Failed {
                reason: reason.clone(),
            });
            let _ = connection.flush();
            return Err(io::Error::other(reason));
        }
        connection.send(&Message::Speaks)?;
        connection.flush()?;
        let (message, descriptors) = receive(&mut connection)?;
        let Message::Takeover(handoff) = message else {
            return Err(io::Error::other("the predecessor sent no takeover"));
        };
        match Self::rebuild(*handoff, descriptors, program) {
            Ok(resumed) => {
                connection.send(&Message::Serving)?;
                connection.flush()?;
                Ok(resumed)
            }
            Err(error) => {
                let _ = connection.send(&Message::Failed {
                    reason: error.to_string(),
                });
                let _ = connection.flush();
                Err(error)
            }
        }
    }

    fn rebuild(
        handoff: Handoff,
        descriptors: Vec<OwnedFd>,
        program: Option<Program>,
    ) -> io::Result<Resumed> {
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
        let tcp = match (handoff.tcp, &handoff.settings.server.tcp) {
            (Some(address), _) => {
                let listener = std::net::TcpListener::from(next()?);
                listener.set_nonblocking(true)?;
                Some((listener, address))
            }
            // The predecessor accepted other hosts elsewhere or not at all: this server binds
            // the endpoint its settings state.
            (None, Some(endpoint)) => {
                let listener = crate::serve::bind_tcp(endpoint)?;
                listener.set_nonblocking(true)?;
                let address = listener.local_addr()?;
                tracing::info!(%address, "accepting other hosts' connections");
                Some((listener, address))
            }
            (None, None) => None,
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
            let connection = Connection::resume(Handed {
                descriptor: next()?,
                unix: client.unix,
                received: client.received,
            })?;
            clients.insert(
                client.id,
                crate::Client {
                    connection,
                    pid: client.pid,
                    pidfd: client.pidfd.map(pidfd).transpose()?,
                    role: client.role,
                    child: None,
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
        let sessions = Sessions::resume(handoff.sessions, queues, pidfd, &handoff.settings.server)?;
        let mut shootdowns = crate::shoot::ShootDowns::new(handoff.settings.server.shoot_down())?;
        for index in handoff.shot_finished {
            shootdowns.note_finished(pidfd(index)?);
        }
        for index in handoff.shot_failed {
            shootdowns.note_failed(pidfd(index)?);
        }
        for (pid, index, gpus) in handoff.followed {
            shootdowns.follow(pid, pidfd(index)?, gpus);
        }
        let children = program
            .map(|program| crate::Children::of(program, &handoff.settings))
            .transpose()
            .map_err(io::Error::other)?;
        let (program, connector, transfer) =
            crate::child::supervisors(children, &handoff.settings.server)?;
        let mut server = Self {
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
            quotas: handoff.quotas,
            kv_storage: KvStorage::new(connector),
            transfers: {
                let mut transfers = Transfers::new(transfer);
                transfers.resume_totals(handoff.task_totals);
                transfers
            },
            program,
            moves: crate::moves::Moves::default(),
            events: Vec::new(),
            restart: None,
            sessions,
            readiness: None,
            stopping: None,
            programs: crate::programs::Programs::resume(handoff.programs),
            // The directory is not handed over: every node, this one among them, states its
            // pages again with its next report, on the connections the successor goes on with.
            directory: crate::directory::Directory::new(crate::host_name()?, &handoff.settings)?,
            settings: handoff.settings.clone(),
        };
        server.watch_pools(handoff.settings.server.buffer_pool_trace.clone());
        tracing::info!(
            clients = server.clients.len(),
            tensors = server.tensors.len(),
            "the server goes on from its predecessor"
        );
        Ok(Resumed {
            server,
            lock,
            socket: handoff.socket,
            settings: handoff.settings,
        })
    }
}

impl Pending {
    /// A restart of `successor`, to which this server states the versions it speaks on its end
    /// of the handoff socket, `channel`.
    fn new(successor: Successor, mut channel: Connection, settings: ::storage::Settings) -> Self {
        let versions = Message::Versions {
            protocol: VERSION,
            runtime: runtime::VERSION,
            image: FORMAT,
        };
        // A successor that ended already (one that refuses its command line ends at once) fails
        // the send; the restart then stands until the loop reads that end as its answer.
        if let Err(error) = channel.send(&versions) {
            tracing::info!(%error, "the successor ended before it was told the versions");
        }
        Self {
            successor,
            channel,
            settings,
            agreed: false,
        }
    }

    /// What the successor answered to the versions this server stated, as far as it arrived.
    fn answer(&mut self) -> Answer {
        let failed =
            |error: io::Error| Answer::Refused(format!("the handoff socket failed: {error}"));
        if let Err(error) = self.channel.flush() {
            return failed(error);
        }
        match self.channel.try_recv::<Message>() {
            Ok(Some(Message::Speaks)) => Answer::Agreed,
            Ok(Some(Message::Failed { reason })) => Answer::Refused(reason),
            Ok(Some(_)) => Answer::Refused("the successor answered out of protocol".into()),
            Ok(None) if self.successor.ended() => {
                Answer::Refused("the successor ended without answering".into())
            }
            Ok(None) => Answer::Waiting,
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                Answer::Refused("the successor closed the handoff socket without answering".into())
            }
            Err(error) => failed(error),
        }
    }
}

/// The next message on the handoff socket, with its descriptors.
///
/// # Errors
///
/// Fails when the predecessor closed the socket or the message is none.
fn receive(connection: &mut Connection) -> io::Result<(Message, Vec<OwnedFd>)> {
    let closed = || io::Error::other("the predecessor closed the handoff socket");
    match connection.recv_with::<Message>(Duration::MAX) {
        Ok(Some(frame)) => Ok(frame),
        Ok(None) => Err(closed()),
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Err(closed()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Instant;

    use ::storage::protocol::{Item, Location, Mode, TensorRecord};
    use ::storage::{Claimant, Client};

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

    /// A server of no children on `listener` that leaves the host's storage to no other process.
    fn server_on(listener: UnixListener) -> Server {
        let settings = ::storage::Settings {
            server: ::storage::ServerSettings {
                host_reserve_gib: 0,
                ..::storage::ServerSettings::default()
            },
            connector: None,
        };
        Server::new(listener, settings, None).unwrap()
    }

    /// A restart of `server` whose successor is [`Server::resume`] on a thread of this process,
    /// which stands for the successor's process (it does not end while the test runs); returns
    /// once the successor speaks the server's versions.
    fn restart_in_process(server: &mut Server) -> std::thread::JoinHandle<Resumed> {
        let (ours, theirs) = UnixStream::pair().unwrap();
        let successor =
            std::thread::spawn(move || Server::resume(OwnedFd::from(theirs), None).unwrap());
        let pid = rustix::process::getpid();
        let this_process = Successor {
            pid: pid.as_raw_nonzero().get().unsigned_abs(),
            pidfd: rustix::process::pidfd_open(pid, rustix::process::PidfdFlags::empty()).unwrap(),
        };
        server.restart = Some(Pending::new(
            this_process,
            Connection::new(ours).unwrap(),
            server.settings.clone(),
        ));
        while server
            .restart
            .as_ref()
            .is_some_and(|pending| !pending.agreed)
        {
            let pending = server.restart.as_mut().unwrap();
            pending.channel.wait(Duration::MAX).unwrap();
            server.read_restart();
        }
        assert!(
            server.restart.is_some(),
            "the successor refused the versions"
        );
        successor
    }

    /// A server passed on goes on serving its clients' connections, the
    /// tensors they put and its listening socket from the successor.
    #[test]
    fn a_successor_serves_what_the_server_held() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("storage.sock");
        let (listener, starting) = service::host::bind::<::storage::Settings>(&path)
            .unwrap()
            .owned(&path)
            .unwrap();
        let lock = starting
            .finish(&crate::serve::readiness(
                &path,
                None,
                &::storage::Settings::default(),
            ))
            .unwrap();
        let server = server_on(listener);
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
                generation: 0,
            },
        )
        .unwrap();
        let claim = head
            .claim(Claimant::Head, 0, vec![scope.clone()], Vec::new())
            .unwrap();
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
            head.claim(Claimant::Head, epoch, vec![scope.clone()], Vec::new())
                .unwrap();
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
        // another thread.
        stop.store(true, Ordering::Release);
        (&waker).write_all(b"x").unwrap();
        let mut server = thread.join().unwrap();
        let successor = restart_in_process(&mut server);
        let pending = server.restart.take().unwrap();
        server.handoff(pending, lock.hand_over()).unwrap();
        let resumed = successor.join().unwrap();
        assert_eq!(resumed.socket, path);
        let _lock = service::host::SocketLock::resume(resumed.lock, path.clone());
        let stop = Arc::new(AtomicBool::new(false));
        let (thread, waker) = running(resumed.server, Arc::clone(&stop));

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
            let seen = pair.signals();
            if pair.next_completion(&mut record).unwrap() {
                return ::storage::runtime::Completion::decode(&record).unwrap();
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            let woken = pair.wait(::storage::Interest::RECORDS, seen, Some(remaining));
            assert!(woken.unwrap(), "no completion arrived");
        }
    }

    /// An engine's session goes on across a restart in place: the successor
    /// speaks the server's versions, its queue pair keeps its indices, and a
    /// commit taken before the restart takes effect in the successor when
    /// its stamp is reached.
    #[test]
    fn a_session_survives_a_restart_with_a_request_deferred_across_it() {
        use ::storage::runtime::{
            Class, Completion, Group, GroupId, Open, OwnerId, Region, Request, RequestId, Tier,
            VERSION,
        };

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("storage.sock");
        let (listener, starting) = service::host::bind::<::storage::Settings>(&path)
            .unwrap()
            .owned(&path)
            .unwrap();
        let lock = starting
            .finish(&crate::serve::readiness(
                &path,
                None,
                &::storage::Settings::default(),
            ))
            .unwrap();
        let server = server_on(listener);
        let stop = Arc::new(AtomicBool::new(false));
        let (thread, waker) = running(server, Arc::clone(&stop));

        let region = |name: &str, units| Region {
            name: name.into(),
            tier: Tier::Device,
            space: name.into(),
            unit_bytes: 64,
            part_bytes: 64,
            reserved: 1,
            units,
            limit: units + 1,
            segments: Vec::new(),
            memory: String::new(),
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
                tensors: [0; 32],
                encoding: [0; 32],
            }],
            trajectory: 0,
            groups: vec![Group {
                id: GroupId(0),
                memory: "gpu0".into(),
                slots: 1,
                regions: vec![vec![0]],
                receives: Vec::new(),
                parts: Vec::new(),
            }],
            schedulers: vec![vec![GroupId(0)]],
        };
        let mut engine = Client::connect(
            &path,
            Role::Head {
                deployment: "model".into(),
                epoch: 0,
                generation: 0,
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

        // The restart: the successor speaks the server's versions, and the
        // commit waits for batch 2 in the pool handed over.
        stop.store(true, Ordering::Release);
        (&waker).write_all(b"x").unwrap();
        let mut server = thread.join().unwrap();
        let successor = restart_in_process(&mut server);
        let pending = server.restart.take().unwrap();
        server.handoff(pending, lock.hand_over()).unwrap();
        let resumed = successor.join().unwrap();
        assert_eq!(resumed.socket, path);
        let _lock = service::host::SocketLock::resume(resumed.lock, path.clone());
        let stop = Arc::new(AtomicBool::new(false));
        let (thread, waker) = running(resumed.server, Arc::clone(&stop));

        // The workers complete batch 2 after the restart, and submit more.
        pair.register(completed).store(2, Ordering::Release);
        pair.notify_registers();
        assert_eq!(completion(&mut pair), (RequestId(3), Completion::Committed));
        submit(&mut pair, 4, Request::List);
        let listing = Completion::Listing {
            keys: vec![(Key([7; 32]), 0, 64, 0b001, 0b01, 0)],
        };
        assert_eq!(completion(&mut pair), (RequestId(4), listing));
        submit(
            &mut pair,
            5,
            Request::Detach {
                owner: OwnerId(1),
                after: Vec::new(),
            },
        );
        assert_eq!(completion(&mut pair), (RequestId(5), Completion::Done));
        let listing = engine.list().unwrap();
        let groups = listing.buffer_pools.iter().flat_map(|pool| &pool.groups);
        assert_eq!(groups.map(|group| group.completed).collect::<Vec<_>>(), [2]);
        engine.close_pool().unwrap();

        stop.store(true, Ordering::Release);
        (&waker).write_all(b"x").unwrap();
        drop(thread.join().unwrap());
    }

    /// A successor of a build that speaks another version of the protocol
    /// refuses the takeover before anything is handed over: it says why and
    /// ends, and the predecessor's restart is given up.
    #[test]
    fn a_successor_refuses_a_predecessor_of_another_protocol_version() {
        let (ours, theirs) = UnixStream::pair().unwrap();
        let successor =
            std::thread::spawn(move || Server::resume(OwnedFd::from(theirs), None).map(drop));
        let mut predecessor = Connection::new(ours).unwrap();
        predecessor
            .send(&Message::Versions {
                protocol: VERSION + 1,
                runtime: runtime::VERSION,
                image: FORMAT,
            })
            .unwrap();
        predecessor.flush().unwrap();

        let Message::Failed { reason } = receive(&mut predecessor).unwrap().0 else {
            panic!("the successor did not refuse the versions");
        };
        assert!(
            reason.contains(&format!("protocol {}", VERSION + 1))
                && reason.contains(&format!("speaks {VERSION}")),
            "{reason}"
        );
        assert!(successor.join().unwrap().is_err());
    }
}
