//! Host services: the servers a host runs once per serving user.
//!
//! The Storage Server and the Program Server are host services. Each listens
//! on a Unix socket in a directory only its user can enter
//! ([`runtime_directory`]), so only that user's processes connect, and
//! outlives the processes that use it. Every file of a service sits beside
//! its socket and is named after it: `NAME.sock` is the socket, `NAME.lock`
//! the lock that makes one server the socket's owner ([`bind`]), and
//! `NAME.log` the log of a server started detached ([`detach`]).
//!
//! A process that needs a service its host does not run starts one with
//! [`ensure`]: the executable's `NAME serve --socket PATH --daemon` command,
//! which leaves its starter's session and process tree, so the starter's end
//! never ends the service. A server logs to standard error at the level of
//! `UNISERVE_<NAME>_LOG` ([`init_logging`]) and stops on SIGTERM or SIGINT
//! ([`stop_signals`]).

use std::fs::{File, OpenOptions};
use std::io::{self, Read as _, Write as _};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// The directory of the calling user's host services:
/// `$XDG_RUNTIME_DIR/uniserve`, or `/tmp/uniserve-<uid>` without a runtime
/// directory.
pub fn runtime_directory() -> PathBuf {
    match std::env::var_os("XDG_RUNTIME_DIR") {
        Some(runtime) if !runtime.is_empty() => PathBuf::from(runtime).join("uniserve"),
        // SAFETY: getuid has no preconditions.
        _ => PathBuf::from(format!("/tmp/uniserve-{}", unsafe { libc::getuid() })),
    }
}

/// The lock that makes a server the owner of its socket, held for the
/// server's life.
pub struct SocketLock {
    file: File,
    socket: PathBuf,
}

impl SocketLock {
    /// The lock's file, for a successor that goes on with the server to hold
    /// ([`SocketLock::resume`]); this process leaves the socket in place.
    pub fn hand_over(self) -> OwnedFd {
        let lock = std::mem::ManuallyDrop::new(self);
        // SAFETY: `lock` is never dropped, so its file is read out exactly
        // once; its path is leaked with it.
        let file = unsafe { std::ptr::read(&lock.file) };
        OwnedFd::from(file)
    }

    /// Holds the lock a predecessor handed over for the socket at `socket`,
    /// which this process removes when it stops.
    pub fn resume(file: OwnedFd, socket: PathBuf) -> Self {
        Self {
            file: File::from(file),
            socket,
        }
    }
}

impl Drop for SocketLock {
    /// The owner removes its socket when it stops, so a later server binds a
    /// fresh one.
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket);
    }
}

/// Binds `socket` for a server, creating its directory with mode 0700 and the
/// socket with mode 0600.
///
/// Returns `None` when another live server owns the socket: the lock beside
/// it is taken, so a second server started at the same time leaves the first
/// serving. A socket a dead server left behind is replaced.
///
/// # Errors
///
/// Fails when the directory, the lock or the socket cannot be created.
pub fn bind(socket: &Path) -> io::Result<Option<(UnixListener, SocketLock)>> {
    let directory = socket.parent().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "the socket has no directory")
    })?;
    std::fs::create_dir_all(directory)?;
    std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))?;

    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .mode(0o600)
        .open(socket.with_extension("lock"))?;
    // SAFETY: flock on a descriptor this function owns.
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::EWOULDBLOCK) {
            return Ok(None);
        }
        return Err(error);
    }

    // The lock is ours, so a socket file present now belongs to no live
    // server.
    match std::fs::remove_file(socket) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    let listener = UnixListener::bind(socket)?;
    std::fs::set_permissions(socket, std::fs::Permissions::from_mode(0o600))?;
    Ok(Some((
        listener,
        SocketLock {
            file: lock,
            socket: socket.to_owned(),
        },
    )))
}

/// Whether a server accepts connections on `socket`.
pub fn serving(socket: &Path) -> bool {
    UnixStream::connect(socket).is_ok()
}

/// A detached server's end of its starter's readiness pipe.
///
/// [`Ready::signal`] lets the starter return. Dropping it unsignalled, as a
/// server that fails before it listens does, makes the starter exit with a
/// failure.
pub struct Ready(OwnedFd);

impl Ready {
    /// Tells the starter that the server listens, or that it found another
    /// server owning the socket.
    pub fn signal(self) {
        let _ = File::from(self.0).write_all(b"1");
    }
}

/// Leaves the starter: forks, lets the parent wait for the child's readiness
/// and exit, and makes the child the leader of a session of its own with its
/// standard input on `/dev/null` and its standard output and error appended
/// to `NAME.log` beside `socket`.
///
/// Returns, in the child, its end of the readiness pipe; the parent never
/// returns. Must be called before the process starts a thread: only the
/// calling thread survives a fork.
///
/// # Errors
///
/// Fails when the log or the readiness pipe cannot be created, or the
/// process cannot fork.
pub fn detach(socket: &Path) -> io::Result<Ready> {
    let directory = socket.parent().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "the socket has no directory")
    })?;
    std::fs::create_dir_all(directory)?;
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(socket.with_extension("log"))?;

    let mut ends = [0; 2];
    // SAFETY: pipe2 fills the two descriptors it creates.
    if unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: pipe2 returned two new descriptors this function owns.
    let (reader, writer) =
        unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) };

    // SAFETY: the caller guarantees the process has one thread, so the child
    // may run anything.
    match unsafe { libc::fork() } {
        -1 => Err(io::Error::last_os_error()),
        0 => {
            drop(reader);
            // SAFETY: setsid, open, dup2 and close on descriptors this
            // process owns.
            unsafe {
                libc::setsid();
                let null = libc::open(c"/dev/null".as_ptr(), libc::O_RDONLY);
                if null >= 0 {
                    libc::dup2(null, 0);
                    libc::close(null);
                }
                libc::dup2(log.as_raw_fd(), 1);
                libc::dup2(log.as_raw_fd(), 2);
            }
            Ok(Ready(writer))
        }
        _ => {
            drop(writer);
            // The child writes once it listens; a child that dies first
            // closes the pipe, which reads as end of file.
            let mut byte = [0_u8; 1];
            let started = File::from(reader).read(&mut byte).unwrap_or(0) == 1;
            // SAFETY: `_exit` ends the starter without running handlers that
            // belong to the child.
            unsafe { libc::_exit(if started { 0 } else { 1 }) }
        }
    }
}

/// Logs the host service `name` to standard error, at the level of
/// `UNISERVE_<NAME>_LOG` (`name` upper-cased, `-` as `_`; such as
/// `UNISERVE_STORAGE_SERVER_LOG`), `info` when unset.
///
/// A server started detached has its standard error on its log beside its
/// socket ([`detach`]), so it calls this after [`detach`].
pub fn init_logging(name: &str) {
    let variable = format!("UNISERVE_{}_LOG", name.to_uppercase().replace('-', "_"));
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env(&variable)
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();
}

/// A descriptor that becomes readable, and a flag that is set, when SIGTERM
/// or SIGINT arrives: a server that polls waits on the descriptor, and one
/// that loops reads the flag.
///
/// # Errors
///
/// Fails when the pipe or a handler cannot be made.
pub fn stop_signals() -> io::Result<(OwnedFd, Arc<AtomicBool>)> {
    let mut ends = [0; 2];
    // SAFETY: pipe2 fills the two descriptors it creates.
    if unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_CLOEXEC | libc::O_NONBLOCK) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: pipe2 returned two new descriptors this function owns.
    let (reader, writer) =
        unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) };

    // The write end lives for the process: a handler may run until exit.
    let writer = Box::leak(Box::new(writer)).as_raw_fd();
    let stop = Arc::new(AtomicBool::new(false));
    for signal in [libc::SIGTERM, libc::SIGINT] {
        let flag = Arc::clone(&stop);
        // SAFETY: the action stores to an atomic and writes to a descriptor
        // that is never closed, both async-signal-safe.
        unsafe {
            signal_hook_registry::register(signal, move || {
                flag.store(true, Ordering::Release);
                let _ = libc::write(writer, b"x".as_ptr().cast(), 1);
            })
        }?;
    }
    Ok((reader, stop))
}

/// Makes sure the host service `name` serves on `socket`: when none does,
/// starts `executable name serve --socket SOCKET --daemon ARGUMENTS`.
///
/// The service runs in a session of its own, not as a child of this process,
/// so this process's end never ends it. A service that serves already keeps
/// the arguments it was started with.
///
/// The `--daemon` start returns once the service listens or found another one
/// listening; the wait for a connection is bounded by `timeout` all the same.
///
/// # Errors
///
/// Fails when the service cannot be started or does not serve in time.
pub fn ensure(
    executable: &Path,
    name: &str,
    socket: &Path,
    arguments: &[String],
    timeout: Duration,
) -> io::Result<()> {
    if serving(socket) {
        return Ok(());
    }

    let status = Command::new(executable)
        .args([name, "serve", "--socket"])
        .arg(socket)
        .arg("--daemon")
        .args(arguments)
        .status()
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("starting {}: {error}", executable.display()),
            )
        })?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "{} could not start a {name} on {}: {status}; its log is {}",
            executable.display(),
            socket.display(),
            socket.with_extension("log").display()
        )));
    }

    let deadline = Instant::now() + timeout;
    while !serving(socket) {
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("no {name} serves on {}", socket.display()),
            ));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A socket has one owner: a second bind while the first server holds
    /// the lock is refused, and the socket binds again once the owner stops.
    #[test]
    fn one_server_owns_a_socket() {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("service.sock");

        let (_listener, lock) = bind(&socket).unwrap().unwrap();
        assert!(serving(&socket));
        assert!(bind(&socket).unwrap().is_none());

        drop(lock);
        assert!(!socket.exists());
        assert!(bind(&socket).unwrap().is_some());
    }
}
