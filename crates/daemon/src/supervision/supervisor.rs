//! The processes a daemon starts on its node and supervises.
//!
//! An engine asks the daemon of each node to start its workers there
//! (`Spawn`); each becomes a child of the daemon's service thread and leads
//! a process group of its own, so the Storage Server's shoot-down of a
//! previous writer ends the processes it started and never the daemon. The
//! daemon reaps each child, removes the driver and NCCL shared memory a
//! child that did not end in order left (`Child`), and reports its exit to
//! the engine that started it.
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
//! An engine starts from a launch file, which the spawn carries
//! (`Spawn::launch`): the daemon of the node it runs on states its own node's
//! Build Server in it and writes it to its own runtime directory, so the
//! leader starts an engine on any node of the deployment the same way.
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
use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};

/// A supervised process: the owner it was started for, the connection that
/// asked for it, its worker group, its rank and the generation of the group's
/// launch. The connection is the launch's own: every client of the socket
/// shares the deployment's owner, and two of them may name one group and rank,
/// as two engines do during an epoch change. Its exit report goes down that
/// connection, and it ends with that connection.
type Key = (u64, u64, String, u32, u64);

/// One supervised process.
struct Supervised {
    child: Child,
    /// The last lines the process wrote to its standard error, which its exit
    /// report carries: a worker that dies before it registers explains itself
    /// there and nowhere else the daemon can see.
    tail: StderrTail,
}

/// The last lines a spawned process wrote to its standard error.
///
/// The stream is drained on its own thread for as long as the process runs, so
/// a full pipe never blocks it, and only a bounded number of lines is kept.
/// This is a report: nothing in the supervisor decides anything from it.
#[derive(Default)]
struct StderrTail(Arc<Mutex<VecDeque<String>>>);

impl StderrTail {
    /// How many of the process's last stderr lines the exit report keeps.
    const LINES: usize = 8;
    /// The longest line prefix kept of any one line.
    const LINE: usize = 400;

    /// Reads `reader` to its end on its own thread, keeping the last lines.
    fn read<R: Read + Send + 'static>(mut reader: R) -> Self {
        let tail = Self::default();
        let kept = Arc::clone(&tail.0);
        std::thread::spawn(move || {
            let mut pending = String::new();
            let mut chunk = [0u8; 4096];
            loop {
                match reader.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(read) => {
                        pending.push_str(&String::from_utf8_lossy(&chunk[..read]));
                        while let Some(end) = pending.find('\n') {
                            let line = pending[..end].trim_end().to_owned();
                            pending.drain(..end + 1);
                            Self::push(&kept, line);
                        }
                    }
                }
            }
            if !pending.trim().is_empty() {
                Self::push(&kept, pending.trim_end().to_owned());
            }
        });
        tail
    }

    /// Keeps `line` in the ring, and echoes it where the process's standard
    /// error went before it was piped: the run's own log.
    ///
    /// The echo is not decoration. A worker writes its explanation as it dies,
    /// and the ring only carries the last line — the ones before it are what a
    /// reader follows a failure through, so they go out unchanged.
    fn push(kept: &Mutex<VecDeque<String>>, line: String) {
        eprintln!("{line}");
        let kept_line: String = line.chars().take(Self::LINE).collect();
        if let Ok(mut kept) = kept.lock() {
            kept.push_back(kept_line);
            while kept.len() > Self::LINES {
                kept.pop_front();
            }
        }
    }

    /// The last line the process wrote, when it wrote one.
    fn last(&self) -> Option<String> {
        self.0.lock().ok()?.back().cloned()
    }
}

/// A listening socket held for a group until a process inherits it.
struct Reservation {
    engine: u64,
    group: String,
    listener: TcpListener,
}

/// The processes of one node.
pub(crate) struct Supervisor {
    /// This node's host identity, which every exit names.
    host: String,
    /// The pipe every process inherits the read end of.
    lifeline: Lifeline,
    children: HashMap<Key, Supervised>,
    reservations: HashMap<u64, Reservation>,
    next_reservation: u64,
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

    // The module is the executable, with any tuning data its package carries beside it.
    use std::os::unix::fs::PermissionsExt as _;
    module
        .files
        .into_iter()
        .find(|file| {
            file.file
                .metadata()
                .is_ok_and(|metadata| metadata.permissions().mode() & 0o111 != 0)
        })
        .map(|file| file.path)
        .ok_or_else(|| format!("the module of {named} holds no executable file"))
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

    /// The executable of the engine `spawn` starts and the launch file written for it,
    /// with this node's facts stated in it.
    ///
    /// The executable is the engine package the launch's module set names, resolved at this
    /// node's Build Server: every node runs the revision the epoch fixed, built for its own
    /// platform.
    fn main_process(&self, spawn: &Spawn) -> Result<(PathBuf, PathBuf), String> {
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
        let directory = service::RuntimeDirectory::of_user()
            .map_err(|error| format!("finding the user's runtime directory: {error}"))?;
        let file = crate::engine::launch::write(directory.path(), spawn.generation, &launch)
            .map_err(|error| {
                format!(
                    "writing the launch file of engine {}: {error}",
                    spawn.generation
                )
            })?;
        Ok((executable, file))
    }

    /// Starts the process `spawn` describes for `engine`, as a child of this
    /// thread, and returns its process id.
    ///
    /// # Errors
    ///
    /// Fails when a reservation to inherit is not this engine's, or the
    /// process cannot be started.
    pub(crate) fn spawn(
        &mut self,
        engine: u64,
        reporter: u64,
        spawn: &Spawn,
    ) -> Result<u32, String> {
        let key = (
            engine,
            reporter,
            spawn.group.clone(),
            spawn.rank,
            spawn.generation,
        );

        // An engine runs as the engine package its launch names; any
        // other process as the executable the request states. Every one's
        // command line ends with its lifeline.
        let mut command = if spawn.launch.is_empty() {
            Command::new(&spawn.executable)
        } else {
            let (executable, launch) = self.main_process(spawn)?;
            let mut command = Command::new(executable);
            command.arg("--launch").arg(launch);
            // An engine process logs at this daemon's level.
            command.args(::service::logging::arguments());
            command
        };
        command.args(&spawn.arguments);
        inherit_lifeline(&mut command, &self.lifeline.read);
        // A worker leads a process group of its own, with every process it
        // starts: the Storage Server shoots down a previous writer by its
        // group, which ends the children that may have inherited its device
        // descriptors, and never the daemon.
        command.process_group(0);
        // Every reservation is checked before any is taken, so a refused
        // spawn leaves the group's reservations held.
        if let Some(reservation) = spawn.inherit.iter().find(|reservation| {
            !self
                .reservations
                .get(reservation)
                .is_some_and(|held| held.engine == engine && held.group == spawn.group)
        }) {
            // Which of the three cases this is decides three different fixes:
            // the reservation is gone (a race between two launches), it names
            // another group, or it names another engine. The table is named
            // with the absent case, so an entry filed under another peer or
            // another id is visible rather than missing.
            match self.reservations.get(reservation) {
                None => tracing::warn!(
                    reservation,
                    engine,
                    spawn_group = ?spawn.group,
                    held = ?self
                        .reservations
                        .iter()
                        .map(|(id, held)| (*id, held.engine, held.group.clone()))
                        .collect::<Vec<_>>(),
                    "a spawn inherits a reservation the node does not hold"
                ),
                Some(held) => tracing::warn!(
                    reservation,
                    case = if held.engine != engine { "engine" } else { "group" },
                    engine,
                    spawn_group = ?spawn.group,
                    held_engine = held.engine,
                    held_group = ?held.group,
                    "a spawn inherits a reservation filed under another identity"
                ),
            }
            return Err(format!(
                "reservation {reservation} is not one of worker group {}'s",
                spawn.group
            ));
        }
        for reservation in &spawn.inherit {
            if let Some(held) = self.reservations.remove(reservation) {
                inherit_listener(&mut command, held.listener);
            }
        }

        // A process's own words are where a launch failure explains itself: a
        // worker that dies before it registers is otherwise only a status. The
        // stream is read for as long as the process runs, so the pipe never
        // fills and the process never blocks on it.
        command.stderr(Stdio::piped());
        let mut child = Child::spawn(&mut command)
            .map_err(|error| format!("spawning {}: {error}", spawn.executable))?;
        let tail = child
            .take_stderr()
            .map_or_else(StderrTail::default, StderrTail::read);
        let pid = child.id();
        tracing::info!(group = %spawn.group, rank = spawn.rank, generation = spawn.generation, pid, "started a process");
        if let Some(mut previous) = self.children.insert(key, Supervised { child, tail }) {
            // A spawn repeated for one launch replaces its process.
            let _ = previous.child.kill();
            let _ = previous.child.wait();
        }
        Ok(pid)
    }

    /// Binds a listening socket for `engine`'s group and holds it until a
    /// spawn of the group inherits it. `address` is this node's address
    /// other nodes reach, which a routable reservation names.
    ///
    /// # Errors
    ///
    /// Fails when no port can be bound.
    pub(crate) fn reserve(
        &mut self,
        engine: u64,
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
            engine,
            group = ?reserve.group,
            "a node reserved a socket"
        );
        self.reservations.insert(
            reservation,
            Reservation {
                engine,
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

    /// Stops every process of `engine`'s `group` without reporting its exit,
    /// and closes the group's reservations, ahead of the group's next launch.
    pub(crate) fn stop(&mut self, engine: u64, group: &str, generation: Option<u64>) {
        let stopped: Vec<Key> = self
            .children
            .keys()
            .filter(|(owner, _, name, _, launched)| {
                *owner == engine
                    && name == group
                    && generation.is_none_or(|generation| *launched == generation)
            })
            .cloned()
            .collect();
        self.terminate(stopped);
        // A group's reservations serve every launch of it; they end with the
        // group.
        if generation.is_none() {
            self.reservations
                .retain(|_, held| held.engine != engine || held.group != group);
        }
    }

    /// Ends every process and reservation of a connection that closed: what
    /// the engine it is owns, and what it asked for as a client.
    pub(crate) fn release(&mut self, engine: u64) {
        let owned: Vec<Key> = self
            .children
            .keys()
            .filter(|(owner, reporter, ..)| *owner == engine || *reporter == engine)
            .cloned()
            .collect();
        self.terminate(owned);
        self.reservations.retain(|_, held| held.engine != engine);
    }

    /// Ends every process this daemon supervises.
    pub(crate) fn terminate_all(&mut self) {
        let all: Vec<Key> = self.children.keys().cloned().collect();
        self.terminate(all);
        self.reservations.clear();
    }

    /// Returns, with the connection that asked for its launch, every child
    /// that exited since the last call, which is supervised no longer. A
    /// child whose status cannot be read is reported as exited.
    pub(crate) fn exits(&mut self) -> Vec<(u64, Exit)> {
        let mut exited = Vec::new();
        for (key, supervised) in &mut self.children {
            let (status, success) = match supervised.child.try_wait() {
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
            exited.push((key.clone(), status, success, supervised.tail.last()));
        }
        exited
            .into_iter()
            .map(|((engine, reporter, group, rank, generation), status, success, last_stderr)| {
                self.children
                    .remove(&(engine, reporter, group.clone(), rank, generation));
                // The report names the engine the process was spawned for, its
                // rank, how it ended, and the last line it wrote itself: a
                // launch that failed for a reason outside the engine says so
                // here rather than arriving as an unexplained status.
                if success {
                    tracing::info!(engine, %group, rank, generation, last_stderr = ?last_stderr, "a process exited");
                } else {
                    tracing::warn!(engine, %group, rank, generation, %status, last_stderr = ?last_stderr, "a process exited");
                }
                (
                    reporter,
                    Exit {
                        host: self.host.clone(),
                        group,
                        rank,
                        generation,
                        status,
                        last_line: last_stderr.unwrap_or_default(),
                    },
                )
            })
            .collect()
    }

    /// Kills the processes of `keys`, then reaps each one.
    fn terminate(&mut self, keys: Vec<Key>) {
        let mut ending = Vec::with_capacity(keys.len());
        for key in keys {
            if let Some(mut supervised) = self.children.remove(&key) {
                if let Err(error) = supervised.child.kill() {
                    tracing::warn!(group = %key.2, rank = key.3, %error, "a process could not be stopped");
                }
                ending.push(supervised);
            }
        }
        for mut supervised in ending {
            let _ = supervised.child.wait();
        }
    }
}

impl Supervisor {
    /// The processes this node supervises now, named as `hand_over` names
    /// them and without giving any of them up: what the daemon's process
    /// table reports to the admin API.
    pub(crate) fn live(&self) -> Vec<ProcessHandoff> {
        self.children
            .iter()
            .map(
                |((engine, reporter, group, rank, generation), supervised)| ProcessHandoff {
                    engine: *engine,
                    reporter: *reporter,
                    group: group.clone(),
                    rank: *rank,
                    generation: *generation,
                    pid: supervised.child.id(),
                },
            )
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
        for ((engine, reporter, group, rank, generation), supervised) in &self.children {
            descriptors.push(supervised.child.pidfd()?);
            processes.push(ProcessHandoff {
                engine: *engine,
                reporter: *reporter,
                group: group.clone(),
                rank: *rank,
                generation: *generation,
                pid: supervised.child.id(),
            });
        }
        let mut reservations = Vec::with_capacity(self.reservations.len());
        for (id, reservation) in self.reservations.drain() {
            reservations.push(ReservationHandoff {
                id,
                engine: reservation.engine,
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
                (
                    process.engine,
                    process.reporter,
                    process.group,
                    process.rank,
                    process.generation,
                ),
                Supervised {
                    child,
                    // An adopted process's standard error was this daemon's
                    // predecessor's pipe: its last line is not readable here.
                    tail: StderrTail::default(),
                },
            );
        }
        for reservation in handoff.reservations {
            supervisor.reservations.insert(
                reservation.id,
                Reservation {
                    engine: reservation.engine,
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
        // engine die the moment the predecessor exits — silently, on stderr no one
        // reads any more.
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
    pub(crate) engine: u64,
    /// The connection its exit report goes down.
    pub(crate) reporter: u64,
    pub(crate) group: String,
    pub(crate) rank: u32,
    pub(crate) generation: u64,
    pub(crate) pid: u32,
}

/// One reservation of the handoff.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ReservationHandoff {
    pub(crate) id: u64,
    pub(crate) engine: u64,
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
    /// Spawned by this daemon, which reaps it.
    Spawned(std::process::Child),
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
        Ok(Self {
            process: Process::Spawned(command.spawn()?),
            exited: None,
        })
    }

    /// Supervises process `pid`, which a daemon this one replaced spawned,
    /// through its pidfd.
    fn adopt(pid: u32, pidfd: OwnedFd) -> Self {
        Self {
            process: Process::Adopted { pid, pidfd },
            exited: None,
        }
    }

    /// The process's standard error, when this daemon spawned it and it was
    /// spawned with a pipe.
    fn take_stderr(&mut self) -> Option<std::process::ChildStderr> {
        match &mut self.process {
            Process::Spawned(child) => child.stderr.take(),
            Process::Adopted { .. } => None,
        }
    }

    /// The rank's process id.
    pub(crate) fn id(&self) -> u32 {
        match &self.process {
            Process::Spawned(child) => child.id(),
            Process::Adopted { pid, .. } => *pid,
        }
    }

    /// A pidfd of the process, which names it whatever process later takes
    /// its id, for a successor daemon to adopt it with.
    fn pidfd(&self) -> std::io::Result<OwnedFd> {
        match &self.process {
            // An unreaped child's id names it alone.
            Process::Spawned(child) => pidfd_open(child.id()),
            Process::Adopted { pidfd, .. } => pidfd.try_clone(),
        }
    }

    /// Sends the rank SIGKILL, unless it already ended.
    pub(crate) fn kill(&mut self) -> std::io::Result<()> {
        if self.exited.is_some() {
            return Ok(());
        }
        match &mut self.process {
            Process::Spawned(child) => child.kill(),
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
                Process::Spawned(child) => child.try_wait()?.map(Exited::Status),
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
            Process::Spawned(child) => Exited::Status(child.wait()?),
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

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    /// A worker that dies before it registers says why on its own standard
    /// error; the report keeps its last line and no more than a bounded number
    /// of them, whatever it wrote.
    #[test]
    fn a_process_s_last_stderr_line_reaches_its_exit_report() {
        let written = (0..StderrTail::LINES * 3)
            .map(|line| format!("line {line}\n"))
            .collect::<String>();
        let tail = StderrTail::read(std::io::Cursor::new(written));

        // The reader runs on its own thread, so the line appears when it has
        // been read, not when `read` returns.
        let deadline = Instant::now() + Duration::from_secs(5);
        let last = loop {
            if let Some(line) = tail.last().filter(|line| line == "line 23") {
                break line;
            }
            assert!(Instant::now() < deadline, "the process's last line is kept");
            std::thread::sleep(Duration::from_millis(10));
        };
        assert_eq!(last, "line 23");
    }

    /// A successor's supervisor adopts the processes and reservations a
    /// supervisor hands over: it reports none of them exited while they
    /// run, ends them as its own, and observes their exits.
    #[test]
    fn the_successor_supervises_the_processes_of_a_handoff() {
        let mut first = Supervisor::new("host".to_owned()).unwrap();
        let spawn = Spawn {
            host: "host".to_owned(),
            group: "model".to_owned(),
            rank: 0,
            generation: 1,
            executable: "sh".to_owned(),
            arguments: vec!["-c".to_owned(), "exec sleep 600".to_owned()],
            ..Spawn::default()
        };
        let pid = first.spawn(7, 7, &spawn).unwrap();
        let reserved = first
            .reserve(
                7,
                &Reserve {
                    host: "host".to_owned(),
                    group: "model".to_owned(),
                    routable: false,
                    engine: 7,
                },
                None,
            )
            .unwrap();
        let (handoff, descriptors) = first.hand_over().unwrap();
        // The first supervisor holds nothing more: dropping it ends nothing.
        drop(first);
        assert!(std::path::Path::new(&format!("/proc/{pid}")).exists());

        let mut successor = Supervisor::adopt("host".to_owned(), handoff, descriptors).unwrap();
        assert!(successor.exits().is_empty());
        assert!(successor.reservations.contains_key(&reserved.reservation));
        successor.stop(7, "model", None);
        assert!(successor.children.is_empty());
        // The killed process is gone once the host's init reaps it, which
        // in a test is this process: its zombie is reaped here.
        let mut status = 0;
        // SAFETY: waits for this test's own child, which the first
        // supervisor spawned.
        unsafe { libc::waitpid(pid as libc::pid_t, &mut status, 0) };
        assert!(libc::WIFSIGNALED(status));
    }
}
