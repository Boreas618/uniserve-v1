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
//! descriptor [`LIFELINE_ENV`] names. The engine and the workers exit once
//! they read the pipe's end, which the kernel delivers when the last write
//! end closed, however the daemon ended. A hot restart hands the processes,
//! and the write end with them, to the successor daemon
//! (`Supervisor::hand_over`), which supervises them from then on through
//! their pidfds (`Supervisor::adopt`): the pipe stays open across it, so no
//! process ends.
//!
//! A worker group's first rank serves the group's collective store. The
//! daemon of that rank's node binds the store's socket when the engine
//! reserves it (`Reserve`) and holds it until it spawns the rank, which
//! inherits it, so no other process takes the port in between.
//!
//! Every process is keyed by the engine connection that started it and by
//! its group, rank and launch generation: two engines of one deployment (the
//! serving world's and its successor's) name the same groups, and a
//! group's standby ranks run beside its serving ones under a later
//! generation. A `Stop` ends one group of one engine, every generation of
//! it; an engine whose connection closes takes its remaining processes with
//! it.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::process::{Command, ExitStatus};
use std::time::{Duration, Instant};

use proto::process::{Exit, Reserve, Reserved, Spawn};

/// How often a running child's NCCL names are noted
/// (`Child::note_nccl_segments`). A worker reports that it is ready to its
/// engine, not to its daemon, so the daemon notes them while the worker
/// lives; its communicators exist from its startup on.
const SEGMENT_NOTE_INTERVAL: Duration = Duration::from_secs(5);

/// The environment variable that names a supervised process's lifeline:
/// the descriptor number of the read end of a pipe whose write end its
/// daemon holds.
pub const LIFELINE_ENV: &str = "UNISERVE_LIFELINE";

/// A supervised process: the engine connection that started it, its worker
/// group, its rank and the generation of the group's launch.
type Key = (u64, String, u32, u64);

/// One supervised process.
struct Supervised {
    child: Child,
    /// When the process's NCCL names were last noted.
    noted: Instant,
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
        })
    }

    /// Starts the process `spawn` describes for `engine`, as a child of this
    /// thread, and returns its process id.
    ///
    /// # Errors
    ///
    /// Fails when a reservation to inherit is not this engine's, or the
    /// process cannot be started.
    pub(crate) fn spawn(&mut self, engine: u64, spawn: &Spawn) -> Result<u32, String> {
        let key = (engine, spawn.group.clone(), spawn.rank, spawn.generation);

        let mut command = Command::new(&spawn.executable);
        command.args(&spawn.arguments);
        for (name, value) in &spawn.environment {
            command.env(name, value);
        }
        // A worker leads a process group of its own, with every process it
        // starts: the Storage Server shoots down a previous writer by its
        // group, which ends the children that may have inherited its device
        // descriptors, and never the daemon.
        command.process_group(0);
        inherit_lifeline(&mut command, &self.lifeline.read);
        // Every reservation is checked before any is taken, so a refused
        // spawn leaves the group's reservations held.
        if let Some(reservation) = spawn.inherit.iter().find(|reservation| {
            !self
                .reservations
                .get(reservation)
                .is_some_and(|held| held.engine == engine && held.group == spawn.group)
        }) {
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

        let child = Child::spawn(&mut command)
            .map_err(|error| format!("spawning {}: {error}", spawn.executable))?;
        let pid = child.id();
        tracing::info!(group = %spawn.group, rank = spawn.rank, generation = spawn.generation, pid, "started a process");
        if let Some(mut previous) = self.children.insert(
            key,
            Supervised {
                child,
                noted: Instant::now(),
            },
        ) {
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
            .filter(|(owner, name, _, launched)| {
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

    /// Ends every process and reservation of an engine whose connection
    /// closed.
    pub(crate) fn release(&mut self, engine: u64) {
        let owned: Vec<Key> = self
            .children
            .keys()
            .filter(|(owner, _, _, _)| *owner == engine)
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

    /// Notes each running child's NCCL names every
    /// [`SEGMENT_NOTE_INTERVAL`], and returns, with its engine, every child
    /// that exited since the last call, which is supervised no longer. A
    /// child whose status cannot be read is reported as exited.
    pub(crate) fn exits(&mut self) -> Vec<(u64, Exit)> {
        let mut exited = Vec::new();
        for (key, supervised) in &mut self.children {
            if supervised.noted.elapsed() >= SEGMENT_NOTE_INTERVAL {
                supervised.noted = Instant::now();
                let _ = supervised.child.note_nccl_segments();
            }
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
            exited.push((key.clone(), status, success));
        }
        exited
            .into_iter()
            .map(|((engine, group, rank, generation), status, success)| {
                self.children
                    .remove(&(engine, group.clone(), rank, generation));
                if success {
                    tracing::info!(%group, rank, generation, "a process exited");
                } else {
                    tracing::warn!(%group, rank, generation, %status, "a process exited");
                }
                (
                    engine,
                    Exit {
                        host: self.host.clone(),
                        group,
                        rank,
                        generation,
                        status,
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
                    tracing::warn!(group = %key.1, rank = key.2, %error, "a process could not be stopped");
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
    pub(crate) fn hand_over(&mut self) -> std::io::Result<(HandedProcesses, Vec<OwnedFd>)> {
        let mut processes = Vec::with_capacity(self.children.len());
        let mut descriptors = Vec::with_capacity(self.children.len() + self.reservations.len());
        for ((engine, group, rank, generation), supervised) in &self.children {
            descriptors.push(supervised.child.pidfd()?);
            processes.push(HandedProcess {
                engine: *engine,
                group: group.clone(),
                rank: *rank,
                generation: *generation,
                pid: supervised.child.id(),
            });
        }
        let mut reservations = Vec::with_capacity(self.reservations.len());
        for (id, reservation) in self.reservations.drain() {
            reservations.push(HandedReservation {
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
            HandedProcesses {
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
        handed: HandedProcesses,
        descriptors: Vec<OwnedFd>,
    ) -> std::io::Result<Self> {
        let mut descriptors = descriptors.into_iter();
        let mut next = || {
            descriptors.next().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "a handed-over process or reservation lacks its descriptor",
                )
            })
        };
        let mut supervisor = Self::new(host)?;
        for process in handed.processes {
            let mut child = Child::adopt(process.pid, next()?);
            let _ = child.note_nccl_segments();
            tracing::info!(group = %process.group, rank = process.rank, generation = process.generation, pid = process.pid, "adopted a process");
            supervisor.children.insert(
                (
                    process.engine,
                    process.group,
                    process.rank,
                    process.generation,
                ),
                Supervised {
                    child,
                    noted: Instant::now(),
                },
            );
        }
        for reservation in handed.reservations {
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
        supervisor.next_reservation = handed.next_reservation;
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
pub(crate) struct HandedProcesses {
    pub(crate) processes: Vec<HandedProcess>,
    pub(crate) reservations: Vec<HandedReservation>,
    pub(crate) next_reservation: u64,
}

/// One handed-over process: its key and process id.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct HandedProcess {
    pub(crate) engine: u64,
    pub(crate) group: String,
    pub(crate) rank: u32,
    pub(crate) generation: u64,
    pub(crate) pid: u32,
}

/// One handed-over reservation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct HandedReservation {
    pub(crate) id: u64,
    pub(crate) engine: u64,
    pub(crate) group: String,
}

/// Makes the process `command` starts inherit the lifeline's read end at
/// its descriptor number, which [`LIFELINE_ENV`] names to it.
fn inherit_lifeline(command: &mut Command, read: &OwnedFd) {
    let descriptor = read.as_raw_fd();
    command.env(LIFELINE_ENV, descriptor.to_string());
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
/// parent, or one a daemon it replaced spawned (adopted).
///
/// A rank that ends in order destroys its CUDA contexts, which removes the
/// shared memory the driver keeps for each context's interprocess events. A
/// rank that is killed or crashes leaves that storage in `/dev/shm`, named
/// for the rank's process id (`cuda::shm::remove_event_storage`), and the names of
/// its NCCL communicators' proxy storage (`cuda::shm::NcclSegments`), which the
/// daemon notes while the rank lives. Every wait here therefore observes
/// the exit first without reaping it, removes the storage named for the rank
/// while its zombie still holds the process id, so that no live process can
/// own those names, and the noted NCCL names, and then reaps it.
///
/// An adopted process's parent is the host's init, which reaps it: this
/// daemon observes its exit through its pidfd and removes what it left at
/// once. Its id may be taken again only after the kernel handed out every
/// other id up to `pid_max`, so a process that took it would have to do so,
/// and create interprocess events, between the exit and the removal.
pub(crate) struct Child {
    process: Process,
    /// How the process ended, once it did; a reaped child's id may name
    /// another process from then on, so it is never waited on again.
    exited: Option<Exited>,
    /// The NCCL shared memory the rank was last seen to map under names
    /// still linked ([`Self::note_nccl_segments`]).
    segments: cuda::shm::NcclSegments,
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

impl Exited {
    /// The exit status, when this daemon read it.
    #[cfg(test)]
    pub(crate) fn status(self) -> Option<ExitStatus> {
        match self {
            Self::Status(status) => Some(status),
            Self::Unread => None,
        }
    }
}

impl Child {
    /// Spawns the process `command` describes.
    pub(crate) fn spawn(command: &mut Command) -> std::io::Result<Self> {
        Ok(Self {
            process: Process::Spawned(command.spawn()?),
            exited: None,
            segments: cuda::shm::NcclSegments::default(),
        })
    }

    /// Supervises process `pid`, which a daemon this one replaced spawned,
    /// through its pidfd.
    fn adopt(pid: u32, pidfd: OwnedFd) -> Self {
        Self {
            process: Process::Adopted { pid, pidfd },
            exited: None,
            segments: cuda::shm::NcclSegments::default(),
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

    /// Notes the NCCL shared memory the running rank maps under names still
    /// linked, which the reaping of a rank that died without destroying its
    /// communicators removes. A rank's communicators exist once it reported
    /// that it is ready: warmup and graph capture have connected them all.
    pub(crate) fn note_nccl_segments(&mut self) -> std::io::Result<()> {
        if self.exited.is_none() {
            self.segments
                .extend(cuda::shm::NcclSegments::mapped_by(Some(self.id()))?);
        }
        Ok(())
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

    /// Returns how the rank ended if it has, after releasing what it left
    /// (and reaping it, when this daemon is its parent), and `None` while it
    /// runs.
    pub(crate) fn try_wait(&mut self) -> std::io::Result<Option<Exited>> {
        if self.exited.is_none() && self.ended(false)? {
            self.release()?;
        }
        Ok(self.exited)
    }

    /// Waits for the rank to exit, releases what it left, reaps it when
    /// this daemon is its parent, and returns how it ended.
    pub(crate) fn wait(&mut self) -> std::io::Result<Exited> {
        if let Some(exited) = self.exited {
            return Ok(exited);
        }
        self.ended(true)?;
        self.release()
    }

    /// Whether the rank has exited, leaving a child unreaped
    /// (`waitid(WNOWAIT)`) and an adopted process observed through its
    /// pidfd. Blocks until it exits when `block`.
    fn ended(&self, block: bool) -> std::io::Result<bool> {
        match &self.process {
            Process::Spawned(child) => exited(child.id(), if block { 0 } else { libc::WNOHANG }),
            Process::Adopted { pidfd, .. } => {
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
                    // A pidfd turns readable once its process exited.
                    return Ok(ready > 0);
                }
            }
        }
    }

    /// Releases what the exited rank left and, as its parent, reaps it.
    fn release(&mut self) -> std::io::Result<Exited> {
        // The rank inherited this process's user. The removal is best
        // effort: the rank is reaped whatever it finds.
        // SAFETY: getuid has no preconditions.
        let uid = unsafe { libc::getuid() };
        let _ = cuda::shm::remove_event_storage(uid, self.id());
        let _ = self.segments.remove();
        let exited = match &mut self.process {
            Process::Spawned(child) => Exited::Status(child.wait()?),
            Process::Adopted { .. } => Exited::Unread,
        };
        self.exited = Some(exited);
        Ok(exited)
    }
}

/// Whether child `pid` has exited, observed with `waitid(WNOWAIT)`, which
/// leaves it unreaped. Blocks until it exits unless `flags` holds
/// `WNOHANG`.
fn exited(pid: u32, flags: libc::c_int) -> std::io::Result<bool> {
    loop {
        // SAFETY: an all-zero `siginfo_t` is a valid value for the call to
        // fill.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is a valid out-pointer and `pid` is this process's
        // unreaped child.
        let waited = unsafe {
            libc::waitid(
                libc::P_PID,
                pid,
                &mut info,
                libc::WEXITED | libc::WNOWAIT | flags,
            )
        };
        if waited == -1 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        // With WNOHANG, a child that has not exited leaves `si_pid` zero.
        // SAFETY: the call succeeded, so `info` describes the child.
        return Ok(unsafe { info.si_pid() } != 0);
    }
}

/// A pidfd of process `pid`.
fn pidfd_open(pid: u32) -> std::io::Result<OwnedFd> {
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
    use std::os::unix::process::ExitStatusExt;
    use std::path::{Path, PathBuf};

    use super::*;

    /// A stand-in rank: creates an interprocess event on device 0, which
    /// makes the driver create its shared memory named for this process, and
    /// waits to be killed.
    const EVENT_HOLDER: &str = "\
import ctypes, time
cuda = ctypes.CDLL('libcuda.so.1')
device, context, event = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_void_p()
assert cuda.cuInit(0) == 0
assert cuda.cuDeviceGet(ctypes.byref(device), 0) == 0
assert cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device) == 0
assert cuda.cuCtxPushCurrent_v2(context) == 0
# CU_EVENT_DISABLE_TIMING | CU_EVENT_INTERPROCESS
assert cuda.cuEventCreate(ctypes.byref(event), 2 | 4) == 0
time.sleep(600)
";

    /// A stand-in rank that maps shared memory under NCCL's prefix, as a
    /// communicator's proxy maps its pool, writes the object's path to the
    /// file its argument names and waits to be killed.
    const NCCL_HOLDER: &str = "\
import mmap, os, sys, tempfile, time
fd, path = tempfile.mkstemp(prefix='nccl-', dir='/dev/shm')
os.ftruncate(fd, 4096)
pool = mmap.mmap(fd, 4096)
with open(sys.argv[1] + '.tmp', 'w') as file:
    file.write(path)
os.rename(sys.argv[1] + '.tmp', sys.argv[1])
time.sleep(600)
";

    /// Kills and reaps a holder when a test ends, including when it panics.
    struct Holder(Child);

    impl Drop for Holder {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    /// The interprocess-event storage the driver created for process `pid`,
    /// by its name in `/dev/shm` (`cuda.shm.<uid in hex>.<pid in hex>.<n>`).
    fn event_storage(pid: u32) -> Vec<PathBuf> {
        // SAFETY: getuid has no preconditions.
        let prefix = format!("cuda.shm.{:x}.{pid:x}.", unsafe { libc::getuid() });
        std::fs::read_dir("/dev/shm")
            .unwrap()
            .map(|entry| entry.unwrap())
            .filter(|entry| entry.file_name().to_string_lossy().starts_with(&prefix))
            .map(|entry| entry.path())
            .collect()
    }

    /// Starts an [`EVENT_HOLDER`] and waits until its event storage exists.
    fn hold_event() -> (Holder, Vec<PathBuf>) {
        let mut command = Command::new("python3");
        command.args(["-c", EVENT_HOLDER]);
        let holder = Holder(Child::spawn(&mut command).unwrap());
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let storage = event_storage(holder.0.id());
            if !storage.is_empty() {
                return (holder, storage);
            }
            assert!(
                Instant::now() < deadline,
                "the holder created no event storage"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Starts an [`NCCL_HOLDER`] and returns it with its object's path, once
    /// the object is mapped.
    fn hold_nccl(directory: &Path, name: &str) -> (Holder, PathBuf) {
        let path_file = directory.join(name);
        let mut command = Command::new("python3");
        command.args(["-c", NCCL_HOLDER]).arg(&path_file);
        let holder = Holder(Child::spawn(&mut command).unwrap());
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Ok(path) = std::fs::read_to_string(&path_file) {
                return (holder, PathBuf::from(path));
            }
            assert!(Instant::now() < deadline, "the holder mapped nothing");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Sends SIGKILL to `holder` from outside, as a crash or an operator
    /// would, and polls until it is reaped.
    fn kill_and_poll(holder: &mut Holder) -> ExitStatus {
        assert_eq!(
            // SAFETY: kill takes integers only; the pid is this test's own
            // unreaped child, so it names no other process.
            unsafe { libc::kill(holder.0.id() as libc::pid_t, libc::SIGKILL) },
            0
        );
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(exited) = holder.0.try_wait().unwrap() {
                return exited.status().unwrap();
            }
            assert!(Instant::now() < deadline, "the killed rank was not reaped");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// A rank killed by someone else is reaped by polling, as the daemon
    /// polls its children, and one killed by its parent is reaped by
    /// waiting: either way no storage named for the dead rank remains, while
    /// a live rank's stays.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_killed_rank_leaves_no_event_storage_once_reaped() {
        let (mut killed, killed_storage) = hold_event();
        let (mut live, live_storage) = hold_event();

        assert_eq!(kill_and_poll(&mut killed).signal(), Some(libc::SIGKILL));
        assert!(
            killed_storage.iter().all(|path| !path.exists()),
            "{killed_storage:?} outlived its rank"
        );
        assert!(
            live_storage.iter().all(|path| path.exists()),
            "a live rank lost {live_storage:?}"
        );

        live.0.kill().unwrap();
        assert_eq!(
            live.0.wait().unwrap().status().unwrap().signal(),
            Some(libc::SIGKILL)
        );
        assert!(
            live_storage.iter().all(|path| !path.exists()),
            "{live_storage:?} outlived its rank"
        );
    }

    /// A rank killed by someone else leaves its NCCL names, which carry no
    /// process id; its parent noted the names it mapped while it ran and
    /// removes them when it reaps it, while a live rank's stay.
    #[test]
    fn a_killed_rank_leaves_none_of_its_nccl_names_once_reaped() {
        let directory = tempfile::tempdir().unwrap();
        let (mut killed, killed_path) = hold_nccl(directory.path(), "killed");
        let (mut live, live_path) = hold_nccl(directory.path(), "live");
        killed.0.note_nccl_segments().unwrap();
        live.0.note_nccl_segments().unwrap();

        kill_and_poll(&mut killed);
        assert!(!killed_path.exists(), "{killed_path:?} outlived its rank");
        assert!(live_path.exists(), "a live rank lost {live_path:?}");

        live.0.kill().unwrap();
        live.0.wait().unwrap();
        assert!(!live_path.exists(), "{live_path:?} outlived its rank");
    }

    /// A successor's supervisor adopts the processes and reservations a
    /// supervisor hands over: it reports none of them exited while they
    /// run, ends them as its own, and observes their exits.
    #[test]
    fn a_handed_over_process_is_supervised_by_the_successor() {
        let mut first = Supervisor::new("host".to_owned()).unwrap();
        let spawn = Spawn {
            host: "host".to_owned(),
            group: "model".to_owned(),
            rank: 0,
            generation: 1,
            executable: "sleep".to_owned(),
            arguments: vec!["600".to_owned()],
            ..Spawn::default()
        };
        let pid = first.spawn(7, &spawn).unwrap();
        let reserved = first
            .reserve(
                7,
                &Reserve {
                    host: "host".to_owned(),
                    group: "model".to_owned(),
                    routable: false,
                },
                None,
            )
            .unwrap();
        let (handed, descriptors) = first.hand_over().unwrap();
        // The first supervisor holds nothing more: dropping it ends nothing.
        drop(first);
        assert!(std::path::Path::new(&format!("/proc/{pid}")).exists());

        let mut successor = Supervisor::adopt("host".to_owned(), handed, descriptors).unwrap();
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
