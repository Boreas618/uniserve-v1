//! The processes a daemon starts on its node and supervises.
//!
//! A node's processes are engines: a spawn (`Spawn`) carries an engine's launch, and the node runs
//! the engine package its module set names, resolved at the node's own Build Server, with the
//! launch inherited and its lifeline. Each becomes a child of the daemon's service thread and
//! leads a process group of its own, so the Storage Server's shoot-down of a previous writer ends
//! the processes it started and never the daemon. The daemon reaps each child and reports its exit
//! to the asker of its launch. A process writes its own log, to the standard error it inherits
//! from the daemon; how it ended travels as its exit status, and a failure it observed as its
//! report on the process interface.
//!
//! A process ends with its daemon through its lifeline: a pipe whose write
//! end the daemon holds and whose read end every process inherits, at the
//! descriptor the last option of its command line names
//! (`proto::LIFELINE_OPTION`). The engine exits once it reads the pipe's end,
//! which the kernel delivers when the last write end closed, however the
//! daemon ended. A hot restart hands the processes,
//! and the write end with them, to the successor daemon
//! (`Supervisor::hand_over`), which supervises them from then on through
//! their pidfds (`Supervisor::adopt`): the pipe stays open across it, so no
//! process ends.
//!
//! An engine starts from its launch, which the spawn carries (`Spawn::launch`): the daemon of the
//! node it runs on states its own node's Build Server in it and passes it as a sealed memfd
//! (`crate::engine::launch`), so the leader starts an engine on any node of the deployment the
//! same way.
//!
//! A worker group's first rank serves the group's collective store. The
//! daemon of that rank's node binds the store's socket when the engine
//! reserves it (`Reserve`) and holds it until it spawns the rank, which
//! inherits it, so no other process takes the port in between.
//!
//! Every process is keyed by the engine connection that started it and by
//! its group, rank and launch generation: two engines of one deployment (the
//! serving epoch's and its successor's) name the same groups, and a
//! group's standby ranks run beside its serving ones under a later
//! generation. A `Stop` ends one group of one engine, every generation of
//! it; an engine whose connection closes takes its remaining processes with
//! it.

use proto::process::{Exit, Reserve, Reserved, Spawn};
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, ExitStatus};

/// Whom a request of the process interface is served for (`proto::process::Owner`): the engine
/// of a number, which owns the processes it starts and claims as itself, or the deployment's head
/// (the updater and the daemon's own clients of the socket), which owns the engines it launches
/// and claims on the epoch's behalf.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub(crate) enum Owner {
    Engine(u64),
    Head,
}

impl Owner {
    /// The claimant a claim served for this owner is made as.
    pub(crate) fn claimant(self) -> storage::Claimant {
        match self {
            Self::Engine(engine) => storage::Claimant::Engine(engine),
            Self::Head => storage::Claimant::Head,
        }
    }
}

impl From<Owner> for proto::process::Owner {
    fn from(owner: Owner) -> Self {
        use proto::process::owner::Owner as Wire;
        Self {
            owner: Some(match owner {
                Owner::Engine(engine) => Wire::Engine(engine),
                Owner::Head => Wire::Head(proto::process::Head {}),
            }),
        }
    }
}

impl TryFrom<Option<proto::process::Owner>> for Owner {
    type Error = String;

    fn try_from(owner: Option<proto::process::Owner>) -> Result<Self, String> {
        use proto::process::owner::Owner as Wire;
        match owner.and_then(|owner| owner.owner) {
            Some(Wire::Engine(engine)) => Ok(Self::Engine(engine)),
            Some(Wire::Head(_)) => Ok(Self::Head),
            None => Err("a forwarded request states no owner".to_owned()),
        }
    }
}

/// Who asked for a launch or any other request: a connection of the leader's process socket, by
/// the identity the leader's service gives it, or the leader's updater, which asks over no
/// connection. A reply goes back to it, and so does the exit report of a launch it asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub(crate) enum Asker {
    Connection(u64),
    Updater,
}

impl Asker {
    /// The asker of a forwarded request's `connection` field.
    pub(crate) fn of_wire(connection: Option<u64>) -> Self {
        connection.map_or(Self::Updater, Self::Connection)
    }

    /// This asker as a forwarded request's `connection` field.
    pub(crate) fn wire(self) -> Option<u64> {
        match self {
            Self::Connection(connection) => Some(connection),
            Self::Updater => None,
        }
    }
}

/// A supervised process: the owner it was started for, who asked for it, its worker group, its
/// rank and the generation of the group's launch. The asker is the launch's own: every client of
/// the socket shares the deployment's head as owner, and two of them may name one group and rank,
/// as two engines do during an epoch change. Its exit report goes to that asker, and it ends with
/// that asker's connection.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Key {
    owner: Owner,
    asker: Asker,
    group: String,
    rank: u32,
    generation: u64,
}

impl Key {
    /// The exit of the process of this key on `host`, which ended as `status` states, with the
    /// asker of its launch.
    fn exit(self, host: &str, status: String) -> (Asker, Exit) {
        let Self {
            asker,
            group,
            rank,
            generation,
            ..
        } = self;
        (
            asker,
            Exit {
                host: host.to_owned(),
                group,
                rank,
                generation,
                status,
            },
        )
    }
}

/// A listening socket held for a group until a process inherits it.
struct Reservation {
    owner: Owner,
    group: String,
    listener: TcpListener,
}

/// The processes of one node.
pub(crate) struct Supervisor {
    /// This node's host identity, which every exit names.
    host: String,
    /// The pipe every process inherits the read end of.
    lifeline: Lifeline,
    children: HashMap<Key, Child>,
    reservations: HashMap<u64, Reservation>,
    next_reservation: u64,
    /// The exits of the processes a stop ended, which [`Supervisor::exits`] reports with the
    /// next exits it reaps: a stopped engine ends as every engine does, for its asker and the
    /// updater, which keeps a previous epoch's weights on that exit.
    stopped: Vec<(Asker, Exit)>,
    /// This node's Build Server, which an engine started here resolves its modules from:
    /// it is started on the first such spawn when it does not serve yet, and its socket is the one
    /// the launch file names. `None` keeps the socket the launch states.
    build_server: Option<build_server::command::Settings>,
}

/// The executable an engine started from `launch` runs as: the module of the engine
/// package the launch's module set names, at the Build Server the launch names.
///
/// Every host of the epoch prepared the module set before any of its processes was spawned
/// (`Request::PrepareModules`), so this is a read of the node's store, within the reply time of a spawn;
/// a store that lost the module builds it again first. A module of another version of the
/// engine interface than this daemon's ([`proto::ENGINE_INTERFACE_VERSION`]) is refused here,
/// before a process starts.
pub(crate) fn engine_executable(launch: &proto::config::EngineArgs) -> Result<PathBuf, String> {
    let engine = &launch.modules.engine;
    let named = format!(
        "the engine package {} revision {}",
        engine.backend, engine.revision
    );

    let mut client = build_server::Client::connect(std::path::Path::new(&launch.build_server))
        .map_err(|error| {
            format!(
                "connecting to the Build Server at {} for {named}: {error}",
                launch.build_server
            )
        })?;
    let module = client
        .resolve(
            proto::module::ENGINE,
            engine,
            &build_server::Target::host(),
            proto::module::SINGLE_KEY,
            &mut |progress| {
                tracing::info!(
                    path = %progress.path,
                    state = ?progress.state,
                    "the Build Server is building the engine's module"
                );
            },
        )
        .map_err(|error| format!("resolving {named}: {error}"))?;
    if module.interface_version != proto::ENGINE_INTERFACE_VERSION {
        return Err(format!(
            "{named} implements version {} of the engine interface and this daemon speaks \
             version {}",
            module.interface_version,
            proto::ENGINE_INTERFACE_VERSION
        ));
    }

    // The module holds the executable its manifest names, with any tuning data its package
    // carries beside it.
    let entry = module
        .executable
        .ok_or_else(|| format!("the module of {named} names no executable"))?;
    module
        .files
        .into_iter()
        .find(|file| file.name == entry)
        .map(|file| file.path)
        .ok_or_else(|| format!("the module of {named} holds no file {entry}, its executable"))
}

/// A daemon's lifeline: both ends of the pipe, close-on-exec in the
/// daemon; a spawn makes the read end its process's.
struct Lifeline {
    read: OwnedFd,
    write: OwnedFd,
}

impl Lifeline {
    fn new() -> std::io::Result<Self> {
        let mut ends = [0; 2];
        // SAFETY: pipe2 fills the two descriptors it creates.
        if unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: pipe2 returned two new descriptors this daemon owns.
        Ok(unsafe {
            Self {
                read: OwnedFd::from_raw_fd(ends[0]),
                write: OwnedFd::from_raw_fd(ends[1]),
            }
        })
    }
}

impl Supervisor {
    /// The supervisor of host `host`'s processes.
    ///
    /// # Errors
    ///
    /// Fails when the lifeline's pipe cannot be created.
    pub(crate) fn new(host: String) -> std::io::Result<Self> {
        Ok(Self {
            host,
            lifeline: Lifeline::new()?,
            children: HashMap::new(),
            reservations: HashMap::new(),
            next_reservation: 1,
            stopped: Vec::new(),
            build_server: None,
        })
    }

    /// States this node's Build Server, which the engines started here resolve their
    /// modules from.
    #[must_use]
    pub(crate) fn with_build_server(mut self, settings: build_server::command::Settings) -> Self {
        self.build_server = Some(settings);
        self
    }

    /// This node's Build Server, which the engines started here resolve their modules from and
    /// an epoch's module set is prepared at (`Request::PrepareModules`); `None` when the node states
    /// none.
    pub(crate) fn build_server(&self) -> Option<&build_server::command::Settings> {
        self.build_server.as_ref()
    }

    /// The executable of the engine `spawn` starts and the launch it reads
    /// ([`crate::engine::launch::memfd`]), with this node's facts stated in it.
    ///
    /// The executable is the engine package the launch's module set names, resolved at this
    /// node's Build Server: every node runs the revision the epoch fixed, built for its own
    /// platform.
    fn main_process(&self, spawn: &Spawn) -> Result<(PathBuf, OwnedFd), String> {
        let mut launch: proto::config::EngineArgs =
            serde_json::from_slice(&spawn.launch).map_err(|error| {
                format!("reading the launch of engine {}: {error}", spawn.generation)
            })?;
        if let Some(settings) = &self.build_server {
            let socket = crate::engine::control::ensure_build_server(settings)
                .map_err(|error| format!("starting this node's Build Server: {error:#}"))?;
            launch.build_server = socket.display().to_string();
        }
        let executable = engine_executable(&launch)?;
        let launch = crate::engine::launch::memfd(&launch).map_err(|error| {
            format!("writing the launch of engine {}: {error}", spawn.generation)
        })?;
        Ok((executable, launch))
    }

    /// Starts the process `spawn` describes for `owner`, as `asker` asked, as a child of this
    /// thread, and returns its process id.
    ///
    /// # Errors
    ///
    /// Fails when a reservation to inherit is not this owner's, or the
    /// process cannot be started.
    pub(crate) fn spawn(
        &mut self,
        owner: Owner,
        asker: Asker,
        spawn: &Spawn,
    ) -> Result<u32, String> {
        let key = Key {
            owner,
            asker,
            group: spawn.group.clone(),
            rank: spawn.rank,
            generation: spawn.generation,
        };

        // Every reservation is checked before any is taken, so a refused
        // spawn leaves the group's reservations held.
        if let Some(reservation) = spawn.inherit.iter().find(|reservation| {
            !self
                .reservations
                .get(reservation)
                .is_some_and(|held| held.owner == owner && held.group == spawn.group)
        }) {
            // The refusal names which of the three cases this is: the reservation is gone, it
            // names another group, or it is another owner's.
            return Err(match self.reservations.get(reservation) {
                None => format!("this node holds no reservation {reservation}"),
                Some(held) if held.owner != owner => format!(
                    "reservation {reservation} is {:?}'s, not {owner:?}'s",
                    held.owner
                ),
                Some(held) => format!(
                    "reservation {reservation} is worker group {}'s, not {}'s",
                    held.group, spawn.group
                ),
            });
        }

        // The process runs as the engine package its launch names, with the launch inherited, at
        // this daemon's log level, and with its lifeline last on its command line.
        let (executable, launch) = self.main_process(spawn)?;
        let mut command = Command::new(&executable);
        crate::engine::launch::inherit(&mut command, launch);
        command.args(::service::logging::arguments());
        inherit_lifeline(&mut command, &self.lifeline.read);
        // A worker leads a process group of its own, with every process it
        // starts: the Storage Server shoots down a previous writer by its
        // group, which ends the children that may have inherited its device
        // descriptors, and never the daemon.
        command.process_group(0);
        for reservation in &spawn.inherit {
            if let Some(held) = self.reservations.remove(reservation) {
                inherit_listener(&mut command, held.listener);
            }
        }

        let child = Child::spawn(&mut command)
            .map_err(|error| format!("spawning {}: {error}", executable.display()))?;
        let pid = child.id();
        tracing::info!(group = %spawn.group, rank = spawn.rank, generation = spawn.generation, pid, "started a process");
        if let Some(mut previous) = self.children.insert(key, child) {
            // A spawn repeated for one launch replaces its process.
            let _ = previous.kill();
            let _ = previous.wait();
        }
        Ok(pid)
    }

    /// Binds a listening socket for `owner`'s group and holds it until a
    /// spawn of the group inherits it. `address` is this node's address
    /// other nodes reach, which a routable reservation names.
    ///
    /// # Errors
    ///
    /// Fails when no port can be bound.
    pub(crate) fn reserve(
        &mut self,
        owner: Owner,
        reserve: &Reserve,
        address: Option<IpAddr>,
    ) -> Result<Reserved, String> {
        // A loopback address reaches only this node, so a socket other nodes
        // connect to binds every interface and is named by this node's
        // routable address rather than by the wildcard.
        let (bind, advertise) = match (reserve.routable, address) {
            (true, Some(address)) => (Ipv4Addr::UNSPECIFIED, address),
            _ => (Ipv4Addr::LOCALHOST, IpAddr::V4(Ipv4Addr::LOCALHOST)),
        };
        let listener = TcpListener::bind(SocketAddr::from((bind, 0)))
            .map_err(|error| format!("reserving a port: {error}"))?;
        let port = listener
            .local_addr()
            .map_err(|error| format!("reading a reserved port: {error}"))?
            .port();
        let reservation = self.next_reservation;
        self.next_reservation += 1;
        // The process that inherits the socket finds it at this number.
        let descriptor = listener.as_raw_fd();
        tracing::debug!(
            reservation,
            ?owner,
            group = ?reserve.group,
            "a node reserved a socket"
        );
        self.reservations.insert(
            reservation,
            Reservation {
                owner,
                group: reserve.group.clone(),
                listener,
            },
        );
        Ok(Reserved {
            reservation,
            address: SocketAddr::new(advertise, port).to_string(),
            descriptor,
        })
    }

    /// Stops every process of `owner`'s `group`, of launch generation `generation` when one is
    /// named, and reports each exit with the next [`Supervisor::exits`]; a stop of the whole
    /// group also closes its reservations, ahead of its next launch.
    pub(crate) fn stop(&mut self, owner: Owner, group: &str, generation: Option<u64>) {
        let stopped: Vec<Key> = self
            .children
            .keys()
            .filter(|key| {
                key.owner == owner
                    && key.group == group
                    && generation.is_none_or(|generation| key.generation == generation)
            })
            .cloned()
            .collect();
        let ended = self.terminate(stopped);
        let host = self.host.clone();
        self.stopped.extend(
            ended
                .into_iter()
                .map(|(key, status)| key.exit(&host, status)),
        );
        // A group's reservations serve every launch of it; they end with the
        // group.
        if generation.is_none() {
            self.reservations
                .retain(|_, held| held.owner != owner || held.group != group);
        }
    }

    /// Ends what a connection that closed leaves: every process it asked for, and, when it was
    /// `engine`'s process link, every process and reservation that engine owns.
    pub(crate) fn release(&mut self, connection: u64, engine: Option<u64>) {
        let owner = engine.map(Owner::Engine);
        let ended: Vec<Key> = self
            .children
            .keys()
            .filter(|key| key.asker == Asker::Connection(connection) || Some(key.owner) == owner)
            .cloned()
            .collect();
        // Their asker or their owner is gone, and with it whoever would read their exits.
        let _ = self.terminate(ended);
        if let Some(owner) = owner {
            self.reservations.retain(|_, held| held.owner != owner);
        }
    }

    /// Ends every process this daemon supervises.
    pub(crate) fn terminate_all(&mut self) {
        let all: Vec<Key> = self.children.keys().cloned().collect();
        // The daemon ends with them: no one reads their exits.
        let _ = self.terminate(all);
        self.reservations.clear();
    }

    /// Returns, with the asker of its launch, every child that exited since the last call, which
    /// is supervised no longer. A child whose status cannot be read is reported as exited.
    pub(crate) fn exits(&mut self) -> Vec<(Asker, Exit)> {
        let mut exited = Vec::new();
        for (key, child) in &mut self.children {
            let (status, success) = match child.try_wait() {
                Ok(Some(Exited::Status(status))) => (status.to_string(), status.success()),
                // An adopted process's parent was the daemon this one
                // replaced: it exited while it should have run.
                Ok(Some(Exited::Unread)) => (
                    "exited (adopted from the previous daemon, whose child it was; its \
                     status is not readable here)"
                        .to_owned(),
                    false,
                ),
                Ok(None) => continue,
                Err(error) => (format!("exit status unreadable: {error}"), false),
            };
            exited.push((key.clone(), status, success));
        }
        let mut reported = std::mem::take(&mut self.stopped);
        for (key, status, success) in exited {
            self.children.remove(&key);
            // The report names the owner the process was spawned for, its rank and how it
            // ended; what the process explained of its end is in its own log.
            let Key {
                owner,
                group,
                rank,
                generation,
                ..
            } = &key;
            if success {
                tracing::info!(?owner, %group, rank, generation, "a process exited");
            } else {
                tracing::warn!(?owner, %group, rank, generation, %status, "a process exited");
            }
            reported.push(key.exit(&self.host, status));
        }
        reported
    }

    /// The descriptors that turn readable when a supervised process exits, one per process: the
    /// service waits on them beside its sockets, so an exit ends its wait.
    pub(crate) fn exit_descriptors(&self) -> impl Iterator<Item = std::os::fd::RawFd> + '_ {
        self.children.values().map(Child::exit_descriptor)
    }

    /// Kills the processes of `keys`, then reaps each one; returns each key with how its process
    /// ended.
    fn terminate(&mut self, keys: Vec<Key>) -> Vec<(Key, String)> {
        let mut ending = Vec::with_capacity(keys.len());
        for key in keys {
            if let Some(mut child) = self.children.remove(&key) {
                if let Err(error) = child.kill() {
                    tracing::warn!(group = %key.group, rank = key.rank, %error, "a process could not be stopped");
                }
                ending.push((key, child));
            }
        }
        ending
            .into_iter()
            .map(|(key, mut child)| {
                let status = match child.wait() {
                    Ok(Exited::Status(status)) => status.to_string(),
                    Ok(Exited::Unread) => {
                        "stopped (adopted from the previous daemon, whose child it was; its \
                         status is not readable here)"
                            .to_owned()
                    }
                    Err(error) => format!("exit status unreadable: {error}"),
                };
                (key, status)
            })
            .collect()
    }
}

impl Supervisor {
    /// The processes this node supervises now, named as `hand_over` names
    /// them and without giving any of them up: what the daemon's process
    /// table reports to the admin API.
    pub(crate) fn live(&self) -> Vec<ProcessHandoff> {
        self.children
            .iter()
            .map(|(key, child)| ProcessHandoff::of(key, child.id()))
            .collect()
    }

    /// Hands every process and reservation to this daemon's successor in a
    /// hot restart, which goes on with them ([`Supervisor::adopt`]): their
    /// keys, and a pidfd for each process followed by each reservation's
    /// socket, then the lifeline's read and write ends, in `descriptors`.
    /// This daemon supervises nothing afterwards.
    ///
    /// # Errors
    ///
    /// Fails when a process's pidfd cannot be opened; the supervisor keeps
    /// every process then.
    pub(crate) fn hand_over(&mut self) -> std::io::Result<(ProcessesHandoff, Vec<OwnedFd>)> {
        let mut processes = Vec::with_capacity(self.children.len());
        let mut descriptors = Vec::with_capacity(self.children.len() + self.reservations.len());
        for (key, child) in &self.children {
            descriptors.push(child.pidfd()?);
            processes.push(ProcessHandoff::of(key, child.id()));
        }
        let mut reservations = Vec::with_capacity(self.reservations.len());
        for (id, reservation) in self.reservations.drain() {
            reservations.push(ReservationHandoff {
                id,
                owner: reservation.owner,
                group: reservation.group,
            });
            descriptors.push(OwnedFd::from(reservation.listener));
        }
        descriptors.push(self.lifeline.read.try_clone()?);
        descriptors.push(self.lifeline.write.try_clone()?);
        // The successor supervises them now; dropping a spawned child's
        // handle neither kills nor waits for it.
        self.children.clear();
        Ok((
            ProcessesHandoff {
                processes,
                reservations,
                next_reservation: self.next_reservation,
            },
            descriptors,
        ))
    }

    /// Goes on with the processes and reservations a predecessor handed over
    /// ([`Supervisor::hand_over`]), with their descriptors in its order.
    ///
    /// # Errors
    ///
    /// Fails when a descriptor is missing.
    pub(crate) fn adopt(
        host: String,
        handoff: ProcessesHandoff,
        descriptors: Vec<OwnedFd>,
    ) -> std::io::Result<Self> {
        let mut descriptors = descriptors.into_iter();
        let mut next = || {
            descriptors.next().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "a process or reservation of the handoff lacks its descriptor",
                )
            })
        };
        let mut supervisor = Self::new(host)?;
        for process in handoff.processes {
            let child = Child::adopt(process.pid, next()?);
            tracing::info!(group = %process.group, rank = process.rank, generation = process.generation, pid = process.pid, "adopted a process");
            supervisor.children.insert(
                Key {
                    owner: process.owner,
                    asker: process.asker,
                    group: process.group,
                    rank: process.rank,
                    generation: process.generation,
                },
                child,
            );
        }
        for reservation in handoff.reservations {
            supervisor.reservations.insert(
                reservation.id,
                Reservation {
                    owner: reservation.owner,
                    group: reservation.group,
                    listener: TcpListener::from(next()?),
                },
            );
        }
        supervisor.lifeline = Lifeline {
            read: next()?,
            write: next()?,
        };
        // The lifeline is the processes': a supervised process ends when every write
        // end of the pipe closes, so a successor that did not take these would let the
        // engine die the moment the predecessor exits.
        tracing::info!(
            processes = supervisor.children.len(),
            "a supervisor adopted the processes the previous daemon supervised, with their \
             lifeline"
        );
        supervisor.next_reservation = handoff.next_reservation;
        Ok(supervisor)
    }
}

impl Drop for Supervisor {
    /// A daemon's processes never outlive it, unless it handed them over.
    fn drop(&mut self) {
        self.terminate_all();
    }
}

/// The processes a daemon hands its successor (`Supervisor::hand_over`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ProcessesHandoff {
    pub(crate) processes: Vec<ProcessHandoff>,
    pub(crate) reservations: Vec<ReservationHandoff>,
    pub(crate) next_reservation: u64,
}

/// One process of the handoff: its key and process id.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ProcessHandoff {
    pub(crate) owner: Owner,
    /// Who asked for its launch, which its exit report goes to.
    pub(crate) asker: Asker,
    pub(crate) group: String,
    pub(crate) rank: u32,
    pub(crate) generation: u64,
    pub(crate) pid: u32,
}

impl ProcessHandoff {
    fn of(key: &Key, pid: u32) -> Self {
        Self {
            owner: key.owner,
            asker: key.asker,
            group: key.group.clone(),
            rank: key.rank,
            generation: key.generation,
            pid,
        }
    }
}

/// One reservation of the handoff.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ReservationHandoff {
    pub(crate) id: u64,
    pub(crate) owner: Owner,
    pub(crate) group: String,
}

/// Makes the process `command` starts inherit the lifeline's read end at its
/// descriptor number, which the option appended to its command line names
/// (`proto::LIFELINE_OPTION`).
fn inherit_lifeline(command: &mut Command, read: &OwnedFd) {
    let descriptor = read.as_raw_fd();
    command
        .arg(proto::LIFELINE_OPTION)
        .arg(descriptor.to_string());
    // SAFETY: the closure runs in the forked child between fork and exec and
    // only calls `fcntl`, which is async-signal-safe; the daemon holds the
    // descriptor open for as long as it supervises.
    unsafe {
        command.pre_exec(move || {
            if libc::fcntl(descriptor, libc::F_SETFD, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

/// Makes the process `command` starts inherit `listener` at its descriptor
/// number.
///
/// The command owns `listener` from here on and closes it when the command is
/// dropped, so the socket stays open until the spawn has duplicated it and the
/// daemon holds no listening copy afterwards. Close-on-exec is cleared only
/// in the forked child: the descriptor keeps the flag in the daemon, so no
/// other process it starts meanwhile inherits the socket. A failure to clear
/// the flag fails the spawn.
fn inherit_listener(command: &mut Command, listener: TcpListener) {
    // SAFETY: the closure runs in the forked child between fork and exec, and
    // only calls `fcntl`, which is async-signal-safe. `listener` is moved into
    // the closure, so the descriptor is open in the parent at fork time.
    unsafe {
        command.pre_exec(move || {
            // FD_CLOEXEC is the only descriptor flag, so clearing every flag
            // leaves the socket open across exec at the same number.
            if libc::fcntl(listener.as_raw_fd(), libc::F_SETFD, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

/// A process this daemon supervises: one it spawned and reaps as its
/// parent, or one a daemon it replaced spawned (adopted), whose parent is
/// the host's init and whose exit this daemon observes through its pidfd.
///
/// What a process holds of the host is its own to release: a rank that ends
/// in order destroys its NCCL communicators and its CUDA contexts, which
/// removes the shared memory the libraries keep for them. The daemon reaps
/// and reports; it releases nothing on a rank's behalf.
pub(crate) struct Child {
    process: Process,
    /// How the process ended, once it did; a reaped child's id may name
    /// another process from then on, so it is never waited on again.
    exited: Option<Exited>,
}

enum Process {
    /// Spawned by this daemon, which reaps it; its pidfd turns readable when it exits, which is
    /// what the service's wait observes.
    Spawned {
        child: std::process::Child,
        pidfd: OwnedFd,
    },
    /// Spawned by a daemon this one replaced, observed through its pidfd.
    Adopted { pid: u32, pidfd: OwnedFd },
}

/// How a supervised process ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exited {
    /// Its status, which this daemon read when it reaped it.
    Status(ExitStatus),
    /// An adopted process, whose status only its parent reads.
    Unread,
}

impl Child {
    /// Spawns the process `command` describes.
    pub(crate) fn spawn(command: &mut Command) -> std::io::Result<Self> {
        let mut child = command.spawn()?;
        // An unreaped child's id names it alone, so the pidfd is the child's.
        let pidfd = match pidfd_open(child.id()) {
            Ok(pidfd) => pidfd,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        };
        Ok(Self {
            process: Process::Spawned { child, pidfd },
            exited: None,
        })
    }

    /// The descriptor that turns readable once the process exited.
    fn exit_descriptor(&self) -> std::os::fd::RawFd {
        match &self.process {
            Process::Spawned { pidfd, .. } | Process::Adopted { pidfd, .. } => pidfd.as_raw_fd(),
        }
    }

    /// Supervises process `pid`, which a daemon this one replaced spawned,
    /// through its pidfd.
    fn adopt(pid: u32, pidfd: OwnedFd) -> Self {
        Self {
            process: Process::Adopted { pid, pidfd },
            exited: None,
        }
    }

    /// The rank's process id.
    pub(crate) fn id(&self) -> u32 {
        match &self.process {
            Process::Spawned { child, .. } => child.id(),
            Process::Adopted { pid, .. } => *pid,
        }
    }

    /// A pidfd of the process, which names it whatever process later takes
    /// its id, for a successor daemon to adopt it with.
    fn pidfd(&self) -> std::io::Result<OwnedFd> {
        match &self.process {
            Process::Spawned { pidfd, .. } | Process::Adopted { pidfd, .. } => pidfd.try_clone(),
        }
    }

    /// Sends the rank SIGKILL, unless it already ended.
    pub(crate) fn kill(&mut self) -> std::io::Result<()> {
        if self.exited.is_some() {
            return Ok(());
        }
        match &mut self.process {
            Process::Spawned { child, .. } => child.kill(),
            Process::Adopted { pidfd, .. } => {
                // SAFETY: the call takes a live pidfd and no pointers but a
                // null `siginfo`.
                let sent = unsafe {
                    libc::syscall(
                        libc::SYS_pidfd_send_signal,
                        pidfd.as_raw_fd(),
                        libc::SIGKILL,
                        std::ptr::null::<libc::siginfo_t>(),
                        0,
                    )
                };
                if sent == -1 {
                    let error = std::io::Error::last_os_error();
                    // A process that already exited cannot be signaled.
                    if error.raw_os_error() != Some(libc::ESRCH) {
                        return Err(error);
                    }
                }
                Ok(())
            }
        }
    }

    /// Returns how the rank ended if it has (reaping it, when this daemon is
    /// its parent), and `None` while it runs.
    pub(crate) fn try_wait(&mut self) -> std::io::Result<Option<Exited>> {
        if self.exited.is_none() {
            self.exited = match &mut self.process {
                Process::Spawned { child, .. } => child.try_wait()?.map(Exited::Status),
                Process::Adopted { pidfd, .. } => {
                    pidfd_exited(pidfd, false)?.then_some(Exited::Unread)
                }
            };
        }
        Ok(self.exited)
    }

    /// Waits for the rank to exit, reaps it when this daemon is its parent,
    /// and returns how it ended.
    pub(crate) fn wait(&mut self) -> std::io::Result<Exited> {
        if let Some(exited) = self.exited {
            return Ok(exited);
        }
        let exited = match &mut self.process {
            Process::Spawned { child, .. } => Exited::Status(child.wait()?),
            Process::Adopted { pidfd, .. } => {
                pidfd_exited(pidfd, true)?;
                Exited::Unread
            }
        };
        self.exited = Some(exited);
        Ok(exited)
    }
}

/// Whether the process `pidfd` names has exited: a pidfd turns readable once
/// its process exited, whichever process's child it is. Blocks until it exits
/// when `block`.
pub(crate) fn pidfd_exited(pidfd: &OwnedFd, block: bool) -> std::io::Result<bool> {
    let mut polled = libc::pollfd {
        fd: pidfd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    loop {
        // SAFETY: one valid pollfd for the call.
        let ready = unsafe { libc::poll(&mut polled, 1, if block { -1 } else { 0 }) };
        if ready == -1 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        return Ok(ready > 0);
    }
}

/// A pidfd of process `pid`, which names that process whatever process later
/// takes its id.
pub(crate) fn pidfd_open(pid: u32) -> std::io::Result<OwnedFd> {
    // SAFETY: the call takes integers only.
    let descriptor = unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::pid_t, 0) };
    if descriptor == -1 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: the call returned a new descriptor this daemon owns.
    Ok(unsafe { OwnedFd::from_raw_fd(descriptor as libc::c_int) })
}
