//! The server's child processes, its Mooncake connector and its Transfer Server, and the
//! executable the server runs them and its successor from.
//!
//! Each child runs in a process of its own so that a fault in it (Mooncake's client, a CUDA
//! context) ends that process and never the server. The server makes the child's connection
//! itself: one end of a socket pair stays the server's, admitted as that child's client, and the
//! child inherits the other at its own descriptor number, which its command line names
//! (`storage-server child --channel FD`, [`CHANNEL_OPTION`]). The first message on the channel is
//! the child's [`Launch`]: which child it is, with its typed settings. A child's role is
//! therefore never claimed over the server's public socket, and no setting round-trips through
//! the command line.
//!
//! The server watches each child's pidfd, reaps it when it ends and starts it again after the
//! wait its settings state (`child_restart_s`), which doubles with every process in a row that
//! ended before it connected, up to `child_restart_max_s` ([`Backoff`]). What the server does when a child's
//! connection closes (failing the tasks it held) is the owner's business.
//!
//! A child and a restart's successor run the server's [`Program`]: the executable image the
//! server opened when it started, so a binary replaced on disk meanwhile never gives the server a
//! child of another build.

use std::ffi::{CString, OsStr, OsString};
use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt as _;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use crate::timer::Timer;

use ::storage::protocol::{Child, ChildListing};
use serde::{Deserialize, Serialize};
use service::Connection;

/// The option of a child's command line that names its channel's descriptor.
pub(crate) const CHANNEL_OPTION: &str = "channel";

/// What a child is told when it starts: the first message on its channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum Launch {
    /// The Mooncake connector, with its settings.
    Connector(::storage::connector::Settings),
    /// The Transfer Server, with its options.
    Transfer(crate::transfer::Options),
}

impl Launch {
    /// Which child the launch starts.
    pub(crate) fn child(&self) -> Child {
        match self {
            Self::Connector(_) => Child::Connector,
            Self::Transfer(_) => Child::TransferServer,
        }
    }
}

/// A launch crosses the channel in the storage protocol's encoding and the workspace's frames.
impl service::Message for Launch {
    fn encode(&self) -> io::Result<Vec<u8>> {
        ::storage::protocol::encode(self)
    }

    fn decode(payload: &[u8]) -> io::Result<Self> {
        ::storage::protocol::decode(payload)
    }
}

/// An executable the server starts processes of, opened once: its children, and the successor
/// of a restart in place.
#[derive(Clone)]
pub struct Program {
    /// The executable's image, opened for execution.
    image: Arc<OwnedFd>,
    /// The process name a started process sees as its first argument.
    name: OsString,
    /// The arguments before an entry's own, which select the executable's `storage-server`
    /// command: the executable's global options (its log levels) and the command's path.
    prefix: Vec<String>,
}

impl std::fmt::Debug for Program {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Program")
            .field("name", &self.name)
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl Program {
    /// This process's own executable, as the kernel names the image the process runs
    /// (`/proc/self/exe`, proc(5)), whose `storage-server` command `prefix` selects.
    ///
    /// # Errors
    ///
    /// Fails when the image cannot be opened.
    pub fn running(prefix: Vec<String>) -> io::Result<Self> {
        let name = std::env::args_os()
            .next()
            .unwrap_or_else(|| OsString::from("storage-server"));
        Ok(Self {
            image: Arc::new(open_image(Path::new("/proc/self/exe"))?),
            name,
            prefix,
        })
    }

    /// The executable at `path`, whose `storage-server` command `prefix` selects.
    ///
    /// # Errors
    ///
    /// Fails when the file cannot be opened.
    pub fn open(path: &Path, prefix: Vec<String>) -> io::Result<Self> {
        Ok(Self {
            image: Arc::new(open_image(path)?),
            name: path.as_os_str().to_owned(),
            prefix,
        })
    }

    /// The executable's image, as a client passes it to a server it restarts.
    pub fn image(&self) -> BorrowedFd<'_> {
        self.image.as_fd()
    }

    /// The arguments that select the executable's `storage-server` command.
    pub fn prefix(&self) -> &[String] {
        &self.prefix
    }

    /// The executable a client passed as `image`, whose `storage-server` command `prefix`
    /// selects.
    pub(crate) fn of_image(image: OwnedFd, prefix: Vec<String>) -> Self {
        let name = std::env::args_os()
            .next()
            .unwrap_or_else(|| OsString::from("storage-server"));
        Self {
            image: Arc::new(image),
            name,
            prefix,
        }
    }

    /// Starts a process of the image with `entry` after the prefix, which inherits each of
    /// `inherited` at its own descriptor number and `stdin` as its standard input; every other
    /// descriptor of this process is close-on-exec, and stays so.
    ///
    /// # Errors
    ///
    /// Fails when the process cannot be created or the image cannot be executed.
    pub(crate) fn spawn(
        &self,
        entry: &[OsString],
        inherited: &[BorrowedFd<'_>],
        stdin: Stdio,
    ) -> io::Result<(std::process::Child, Arc<OwnedFd>)> {
        let arguments = std::iter::once(self.name.as_os_str())
            .chain(self.prefix.iter().map(OsStr::new))
            .chain(entry.iter().map(OsString::as_os_str))
            .map(c_string)
            .collect::<io::Result<Vec<_>>>()?;
        let environment = std::env::vars_os()
            .map(|(name, value)| {
                let mut pair = name;
                pair.push("=");
                pair.push(value);
                c_string(&pair)
            })
            .collect::<io::Result<Vec<_>>>()?;
        let exec = Exec::new(arguments, environment);
        let image = self.image.as_raw_fd();
        let inherited: Vec<RawFd> = inherited.iter().map(AsRawFd::as_raw_fd).collect();

        // The command's own program is never executed: the closure replaces the forked child
        // with the image, and returns only when that fails.
        let mut command = Command::new(&self.name);
        command.stdin(stdin);
        // SAFETY: the closure runs in the forked child before exec and calls only `fcntl` and
        // `fexecve`, which are async-signal-safe; the argument and environment vectors it passes
        // were built before the fork and live in the closure.
        unsafe {
            command.pre_exec(move || {
                for &descriptor in &inherited {
                    if libc::fcntl(descriptor, libc::F_SETFD, 0) == -1 {
                        return Err(io::Error::last_os_error());
                    }
                }
                libc::fexecve(image, exec.arguments(), exec.environment());
                Err(io::Error::last_os_error())
            });
        }
        let child = command.spawn()?;
        let pid = rustix::process::Pid::from_child(&child);
        let pidfd = rustix::process::pidfd_open(pid, rustix::process::PidfdFlags::empty())?;
        Ok((child, Arc::new(pidfd)))
    }
}

/// `path` opened as an executable image, close-on-exec.
fn open_image(path: &Path) -> io::Result<OwnedFd> {
    use rustix::fs::{Mode, OFlags};

    Ok(rustix::fs::open(
        path,
        OFlags::RDONLY | OFlags::CLOEXEC,
        Mode::empty(),
    )?)
}

/// `text` as a C string; an argument or an environment entry holds no NUL.
fn c_string(text: &OsStr) -> io::Result<CString> {
    CString::new(text.as_bytes()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} holds a NUL byte", text.to_string_lossy()),
        )
    })
}

/// The argument and environment vectors of an exec, built before the fork: NUL-terminated
/// arrays of pointers into the strings they own.
struct Exec {
    _strings: Vec<CString>,
    arguments: Vec<*const libc::c_char>,
    environment: Vec<*const libc::c_char>,
}

// SAFETY: the pointers point into the strings `Exec` owns, which nothing mutates; a forked child
// reads them once, before it executes the image.
unsafe impl Send for Exec {}
// SAFETY: as above; the vectors are only read.
unsafe impl Sync for Exec {}

impl Exec {
    fn new(arguments: Vec<CString>, environment: Vec<CString>) -> Self {
        let pointers = |strings: &[CString]| {
            strings
                .iter()
                .map(|string| string.as_ptr())
                .chain(std::iter::once(std::ptr::null()))
                .collect::<Vec<_>>()
        };
        let argument_pointers = pointers(&arguments);
        let environment_pointers = pointers(&environment);
        let mut strings = arguments;
        strings.extend(environment);
        Self {
            _strings: strings,
            arguments: argument_pointers,
            environment: environment_pointers,
        }
    }

    fn arguments(&self) -> *const *const libc::c_char {
        self.arguments.as_ptr()
    }

    fn environment(&self) -> *const *const libc::c_char {
        self.environment.as_ptr()
    }
}

/// The children a server runs, and the executable it runs them from.
#[derive(Debug, Clone)]
pub struct Children {
    pub program: Program,
    /// The Mooncake connector's settings; no connector when `None`.
    pub connector: Option<::storage::connector::Settings>,
    /// The Transfer Server's options; no Transfer Server when `None`, so that nothing the
    /// server holds can be copied.
    pub transfer: Option<crate::transfer::Options>,
}

impl Children {
    /// The children of a server serving with `settings`, run from `program`: the connector the
    /// settings state, and the Transfer Server, by the host's calibration the settings name
    /// (none without one, so transfers are refused as unavailable).
    ///
    /// # Errors
    ///
    /// Fails when the calibration the settings name cannot be read or does not hold together.
    pub fn of(program: Program, settings: &::storage::Settings) -> Result<Self, crate::Error> {
        let transfer = match &settings.server.transfer_calibration {
            Some(path) => {
                let calibration = crate::transfer::Calibration::read(path)?;
                Some(crate::transfer::Options::of(&settings.server, calibration))
            }
            None => {
                tracing::warn!(
                    "no transfer calibration is set (`transfer_calibration`): the server runs no \
                     Transfer Server, and refuses transfers as unavailable"
                );
                None
            }
        };
        Ok(Self {
            program,
            connector: settings.connector.clone(),
            transfer,
        })
    }
}

/// The supervisors of `children` with the waits `settings` state (the connector's, then the
/// Transfer Server's), and the program they run; supervisors that run nothing without children.
///
/// # Errors
///
/// Fails when a supervisor's restart timer cannot be created.
pub(super) fn supervisors(
    children: Option<Children>,
    settings: &::storage::ServerSettings,
) -> io::Result<(Option<Program>, Process, Process)> {
    let backoff = Backoff::of(settings);
    let Some(children) = children else {
        return Ok((
            None,
            Process::new(None, backoff)?,
            Process::new(None, backoff)?,
        ));
    };
    Ok((
        Some(children.program),
        Process::new(children.connector.map(Launch::Connector), backoff)?,
        Process::new(children.transfer.map(Launch::Transfer), backoff)?,
    ))
}

/// The entry arguments of a child of `channel` (`storage-server child --channel FD`), as the
/// `storage-server` command reads them ([`crate::command::Command::Child`]).
pub(crate) fn child_entry(channel: RawFd) -> Vec<OsString> {
    vec![
        "child".into(),
        format!("--{CHANNEL_OPTION}").into(),
        channel.to_string().into(),
    ]
}

/// A child process started, as its supervisor hands it to the server: the server's end of its
/// channel, with the launch already queued on it, and the process.
pub(super) struct Started {
    pub(super) connection: Connection,
    pub(super) pid: i32,
    pub(super) pidfd: Arc<OwnedFd>,
}

/// One child process the server keeps running.
pub(super) struct Process {
    /// What the child is told when it starts; `None` once the server stopped it.
    launch: Option<Launch>,
    /// The waits before a start again.
    restart_after: Backoff,
    /// The running process, with its pidfd.
    running: Option<(std::process::Child, Arc<OwnedFd>)>,
    /// Fires when a process is to start again, while none runs; armed only then.
    restart: Timer,
    /// How many processes in a row ended before they connected.
    failures: u32,
    /// Whether the running process connected.
    connected: bool,
    /// How the last process ended.
    ended: Option<String>,
}

/// How long a supervisor waits before it starts a child again.
#[derive(Debug, Clone, Copy)]
pub(super) struct Backoff {
    first: Duration,
    longest: Duration,
}

impl Backoff {
    /// The waits `settings` state.
    pub(super) fn of(settings: &::storage::ServerSettings) -> Self {
        Self {
            first: Duration::from_secs(settings.child_restart_s),
            longest: Duration::from_secs(settings.child_restart_max_s),
        }
    }

    /// The wait after `failures` processes in a row ended before they connected: the first
    /// wait doubled per failure, at most the longest.
    fn after(self, failures: u32) -> Duration {
        self.first
            .checked_mul(1_u32.checked_shl(failures).unwrap_or(u32::MAX))
            .map_or(self.longest, |wait| wait.min(self.longest))
    }
}

impl Process {
    /// A child of `launch`, due at once; none without a launch.
    ///
    /// # Errors
    ///
    /// Fails when its restart timer cannot be created.
    pub(super) fn new(launch: Option<Launch>, restart_after: Backoff) -> io::Result<Self> {
        let restart = Timer::new()?;
        if launch.is_some() {
            restart.arm(Duration::ZERO)?;
        }
        Ok(Self {
            launch,
            restart_after,
            running: None,
            restart,
            failures: 0,
            connected: false,
            ended: None,
        })
    }

    /// Waits as `settings` state before the child starts again from now on, and starts a
    /// Transfer Server again with the chunk time and progress period they state, which the
    /// running one is told apart ([`::storage::protocol::Event::Pace`]).
    pub(super) fn configure(&mut self, settings: &::storage::ServerSettings) {
        self.restart_after = Backoff::of(settings);
        if let Some(Launch::Transfer(options)) = &mut self.launch {
            options.chunk = settings.transfer_chunk();
            options.progress = settings.transfer_progress();
        }
    }

    /// The child the supervisor runs, while it runs one.
    pub(super) fn child(&self) -> Option<Child> {
        self.launch.as_ref().map(Launch::child)
    }

    /// The running process's id.
    pub(super) fn pid(&self) -> Option<u32> {
        self.running.as_ref().map(|(child, _)| child.id())
    }

    /// The running process's pidfd, readable once it ended.
    pub(super) fn pidfd(&self) -> Option<&OwnedFd> {
        self.running.as_ref().map(|(_, pidfd)| pidfd.as_ref())
    }

    /// The running process said hello on its channel: a later end restarts it after the first
    /// wait.
    pub(super) fn connected(&mut self) {
        self.connected = true;
        self.failures = 0;
    }

    /// Starts a process of `program` when a start is due, with a new channel; returns the
    /// server's end of the channel, the launch queued on it, and the process.
    pub(super) fn start_due(&mut self, program: &Program) -> Option<Started> {
        let launch = self.launch.as_ref()?;
        if self.running.is_some() || !self.restart.fired() {
            return None;
        }
        let child = launch.child();
        let started = UnixStream::pair().and_then(|(ours, theirs)| {
            let (process, pidfd) = program.spawn(
                &child_entry(theirs.as_raw_fd()),
                &[theirs.as_fd()],
                Stdio::null(),
            )?;
            drop(theirs);
            let mut connection = Connection::new(ours)?;
            connection.send(launch)?;
            Ok((process, pidfd, connection))
        });
        match started {
            Ok((process, pidfd, connection)) => {
                let pid = rustix::process::Pid::from_child(&process)
                    .as_raw_nonzero()
                    .get();
                tracing::info!(pid, "started the {child}");
                self.running = Some((process, Arc::clone(&pidfd)));
                self.connected = false;
                Some(Started {
                    connection,
                    pid,
                    pidfd,
                })
            }
            Err(error) => {
                tracing::warn!(%error, "the {child} could not be started");
                self.note_end(error.to_string());
                None
            }
        }
    }

    /// Reaps the process when it ended, and schedules the next.
    pub(super) fn reap(&mut self) {
        let Some((process, _)) = &mut self.running else {
            return;
        };
        if let Ok(Some(status)) = process.try_wait() {
            tracing::warn!(%status, child = ?self.child(), "a child process ended");
            self.running = None;
            self.note_end(status.to_string());
        }
    }

    /// A process ended as `how`, or could not be started: the next starts after the wait its
    /// failures in a row call for.
    fn note_end(&mut self, how: String) {
        if !self.connected {
            self.failures = self.failures.saturating_add(1);
        }
        self.connected = false;
        let wait = self.restart_after.after(self.failures.saturating_sub(1));
        if let Err(error) = self.restart.arm(wait) {
            tracing::error!(%error, "the next start of a child cannot be scheduled");
        }
        self.ended = Some(how);
    }

    /// Ends the child and waits for it, as a restart in place does before its successor starts
    /// one of its own; none starts here again.
    pub(super) fn stop(&mut self) {
        self.launch = None;
        self.restart.disarm();
        if let Some((mut process, _)) = self.running.take() {
            let _ = process.kill();
            let _ = process.wait();
        }
    }

    /// The timer the loop waits on while a start of the child is due later: readable once it
    /// is due.
    pub(super) fn restart_timer(&self) -> Option<BorrowedFd<'_>> {
        (self.launch.is_some() && self.running.is_none()).then(|| self.restart.as_fd())
    }

    /// The child in a listing, while the server runs one.
    pub(super) fn listed(&self) -> Option<ChildListing> {
        let launch = self.launch.as_ref()?;
        Some(ChildListing {
            child: launch.child(),
            pid: self.pid(),
            failures: self.failures,
            ended: self.ended.clone(),
            restart_in_ms: self
                .restart_timer()
                .and(self.restart.remaining())
                .map(|wait| u64::try_from(wait.as_millis()).unwrap_or(u64::MAX)),
        })
    }
}

/// Runs the child a server started on the channel at `channel` until the server closes it: reads
/// the launch, which says which child this process is, and runs that child over the channel.
///
/// # Errors
///
/// Fails when the channel yields no launch, or the child fails.
pub(crate) fn run(channel: OwnedFd, levels: &service::logging::Levels) -> anyhow::Result<()> {
    use anyhow::Context as _;

    let mut connection =
        Connection::new(UnixStream::from(channel)).context("taking the channel the server made")?;
    let launch = connection
        .recv::<Launch>(Duration::MAX)
        .context("reading the launch")?
        .context("the server closed the channel before it sent the launch")?;
    let (name, ran) = match launch {
        Launch::Connector(settings) => {
            service::logging::init(
                "MooncakeConnector",
                levels,
                service::logging::Output::Stderr,
            );
            let ran = crate::mooncake::run(connection, &settings).map_err(anyhow::Error::from);
            ("the Mooncake connector", ran)
        }
        Launch::Transfer(options) => {
            service::logging::init("TransferServer", levels, service::logging::Output::Stderr);
            let ran = crate::transfer::run(connection, &options).map_err(anyhow::Error::from);
            ("the Transfer Server", ran)
        }
    };
    if let Err(error) = &ran {
        tracing::error!("{name} failed: {error:#}");
    }
    ran
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The waits double from the first with each failure in a row, up to the longest.
    #[test]
    fn a_child_that_keeps_failing_waits_longer_up_to_the_longest() {
        let backoff = Backoff {
            first: Duration::from_secs(5),
            longest: Duration::from_secs(300),
        };
        let waits: Vec<u64> = (0..8)
            .map(|failures| backoff.after(failures).as_secs())
            .collect();
        assert_eq!(waits, [5, 10, 20, 40, 80, 160, 300, 300]);
        assert_eq!(backoff.after(u32::MAX), Duration::from_secs(300));
    }
}
