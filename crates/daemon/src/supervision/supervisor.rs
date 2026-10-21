//! The processes a daemon starts on its node and supervises.
//!
//! An engine asks the daemon of each node to start its workers there
//! (`Spawn`); each becomes a child of the daemon's service thread, dies with
//! it (`PR_SET_PDEATHSIG`), and leads a process group of its own, so the
//! Storage Server's shoot-down of a previous writer ends the processes it
//! started and never the daemon. The daemon reaps each child, removes the
//! driver and NCCL shared memory a child that did not end in order left
//! (`Child`), and reports its exit to the engine that started it.
//!
//! A worker group's first rank serves the group's collective store. The
//! daemon of that rank's node binds the store's socket when the engine
//! reserves it (`Reserve`) and holds it until it spawns the rank, which
//! inherits it, so no other process takes the port in between.
//!
//! Every process is keyed by the engine connection that started it and by
//! its group, rank and launch generation: two engines of one deployment (the
//! running one and a standby of a hot update) name the same groups, and a
//! group's standby ranks run beside its serving ones under a later
//! generation. A `Stop` ends one group of one engine, every generation of
//! it; an engine whose connection closes takes its remaining processes with
//! it.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::os::fd::AsRawFd;
use std::os::unix::process::CommandExt;
use std::process::{Command, ExitStatus};
use std::time::{Duration, Instant};

use proto::process::{Exit, Reserve, Reserved, Spawn};

/// How often a running child's NCCL names are noted
/// (`Child::note_nccl_segments`). A worker reports that it is ready to its
/// engine, not to its daemon, so the daemon notes them while the worker
/// lives; its communicators exist from its startup on.
const SEGMENT_NOTE_INTERVAL: Duration = Duration::from_secs(5);

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
    children: HashMap<Key, Supervised>,
    reservations: HashMap<u64, Reservation>,
    next_reservation: u64,
}

impl Supervisor {
    pub(crate) fn new(host: String) -> Self {
        Self {
            host,
            children: HashMap::new(),
            reservations: HashMap::new(),
            next_reservation: 1,
        }
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
        die_with_daemon(&mut command);
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
                Ok(Some(status)) => (status.to_string(), status.success()),
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

impl Drop for Supervisor {
    /// A daemon's processes never outlive it.
    fn drop(&mut self) {
        self.terminate_all();
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

/// Makes the process `command` starts die with this daemon.
///
/// The kernel sends the process SIGKILL when the thread that spawned it
/// exits, which covers a daemon killed by a signal or aborted before its
/// supervisor ends its processes. Every process is spawned from the
/// daemon's service thread, which lives as long as the daemon. A process
/// whose daemon exited between fork and the request is already orphaned, so
/// it fails to start instead.
fn die_with_daemon(command: &mut Command) {
    let daemon = std::process::id() as libc::pid_t;
    // SAFETY: the closure runs in the forked child between fork and exec and
    // calls only `prctl` and `getppid`, which are async-signal-safe, and
    // builds its errors without allocating.
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() != daemon {
                return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
            }
            Ok(())
        });
    }
}

/// A process this daemon spawned, and is the parent that reaps.
///
/// A rank that ends in order destroys its CUDA contexts, which removes the
/// shared memory the driver keeps for each context's interprocess events. A
/// rank that is killed or crashes leaves that memory in `/dev/shm`, named
/// for the rank's process id (`cuda::shm::remove_event_memory`), and the names of
/// its NCCL communicators' proxy memory (`cuda::shm::NcclSegments`), which the
/// parent notes while the rank lives. Every wait here therefore observes
/// the exit first without reaping it, removes the memory named for the rank
/// while its zombie still holds the process id, so that no live process can
/// own those names, and the noted NCCL names, and then reaps it.
pub(crate) struct Child {
    child: std::process::Child,
    /// The exit status once the process was reaped; its id may name another
    /// process from then on, so it is never waited on again.
    status: Option<ExitStatus>,
    /// The NCCL shared memory the rank was last seen to map under names
    /// still linked ([`Self::note_nccl_segments`]).
    segments: cuda::shm::NcclSegments,
}

impl Child {
    /// Spawns the process `command` describes.
    pub(crate) fn spawn(command: &mut Command) -> std::io::Result<Self> {
        Ok(Self {
            child: command.spawn()?,
            status: None,
            segments: cuda::shm::NcclSegments::default(),
        })
    }

    /// The rank's process id.
    pub(crate) fn id(&self) -> u32 {
        self.child.id()
    }

    /// Notes the NCCL shared memory the running rank maps under names still
    /// linked, which the reaping of a rank that died without destroying its
    /// communicators removes. A rank's communicators exist once it reported
    /// that it is ready: warmup and graph capture have connected them all.
    pub(crate) fn note_nccl_segments(&mut self) -> std::io::Result<()> {
        if self.status.is_none() {
            self.segments
                .extend(cuda::shm::NcclSegments::mapped_by(Some(self.child.id()))?);
        }
        Ok(())
    }

    /// Sends the rank SIGKILL, unless it was already reaped.
    pub(crate) fn kill(&mut self) -> std::io::Result<()> {
        if self.status.is_some() {
            return Ok(());
        }
        self.child.kill()
    }

    /// Returns the rank's exit status if it has exited, after releasing what
    /// it left and reaping it, and `None` while it runs.
    pub(crate) fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
        if self.status.is_none() && self.exited(libc::WNOHANG)? {
            self.reap()?;
        }
        Ok(self.status)
    }

    /// Waits for the rank to exit, releases what it left, reaps it and
    /// returns its exit status.
    pub(crate) fn wait(&mut self) -> std::io::Result<ExitStatus> {
        if let Some(status) = self.status {
            return Ok(status);
        }
        self.exited(0)?;
        self.reap()
    }

    /// Whether the rank has exited, observed with `waitid(WNOWAIT)`, which
    /// leaves it unreaped. Blocks until it exits unless `flags` holds
    /// `WNOHANG`.
    fn exited(&self, flags: libc::c_int) -> std::io::Result<bool> {
        let pid = self.child.id();
        loop {
            // SAFETY: an all-zero `siginfo_t` is a valid value for the call
            // to fill.
            let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
            // SAFETY: `info` is a valid out-pointer and `pid` is this
            // process's unreaped child.
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

    /// Releases what the exited, unreaped rank left and reaps it.
    fn reap(&mut self) -> std::io::Result<ExitStatus> {
        // The rank inherited this process's user. The removal is best
        // effort: the rank is reaped whatever it finds.
        // SAFETY: getuid has no preconditions.
        let uid = unsafe { libc::getuid() };
        let _ = cuda::shm::remove_event_memory(uid, self.child.id());
        let _ = self.segments.remove();
        let status = self.child.wait()?;
        self.status = Some(status);
        Ok(status)
    }
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

    /// The interprocess-event memory the driver created for process `pid`,
    /// by its name in `/dev/shm` (`cuda.shm.<uid in hex>.<pid in hex>.<n>`).
    fn event_memory(pid: u32) -> Vec<PathBuf> {
        // SAFETY: getuid has no preconditions.
        let prefix = format!("cuda.shm.{:x}.{pid:x}.", unsafe { libc::getuid() });
        std::fs::read_dir("/dev/shm")
            .unwrap()
            .map(|entry| entry.unwrap())
            .filter(|entry| entry.file_name().to_string_lossy().starts_with(&prefix))
            .map(|entry| entry.path())
            .collect()
    }

    /// Starts an [`EVENT_HOLDER`] and waits until its event memory exists.
    fn hold_event() -> (Holder, Vec<PathBuf>) {
        let mut command = Command::new("python3");
        command.args(["-c", EVENT_HOLDER]);
        let holder = Holder(Child::spawn(&mut command).unwrap());
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let memory = event_memory(holder.0.id());
            if !memory.is_empty() {
                return (holder, memory);
            }
            assert!(
                Instant::now() < deadline,
                "the holder created no event memory"
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
            if let Some(status) = holder.0.try_wait().unwrap() {
                return status;
            }
            assert!(Instant::now() < deadline, "the killed rank was not reaped");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// A rank killed by someone else is reaped by polling, as the daemon
    /// polls its children, and one killed by its parent is reaped by
    /// waiting: either way no memory named for the dead rank remains, while
    /// a live rank's stays.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_killed_rank_leaves_no_event_memory_once_reaped() {
        let (mut killed, killed_memory) = hold_event();
        let (mut live, live_memory) = hold_event();

        assert_eq!(kill_and_poll(&mut killed).signal(), Some(libc::SIGKILL));
        assert!(
            killed_memory.iter().all(|path| !path.exists()),
            "{killed_memory:?} outlived its rank"
        );
        assert!(
            live_memory.iter().all(|path| path.exists()),
            "a live rank lost {live_memory:?}"
        );

        live.0.kill().unwrap();
        assert_eq!(live.0.wait().unwrap().signal(), Some(libc::SIGKILL));
        assert!(
            live_memory.iter().all(|path| !path.exists()),
            "{live_memory:?} outlived its rank"
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
}
