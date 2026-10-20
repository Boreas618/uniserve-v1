//! Where a host's Storage Server listens, and starting it when it is absent.
//!
//! A host runs one Storage Server per serving user. It listens on a Unix
//! socket in a directory only that user can enter, so only that user's
//! processes connect. A lock file beside the socket makes the server that
//! holds it the socket's one owner: a second server started at once finds
//! the lock taken and leaves the first serving.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

/// The socket of this host's Storage Server for the calling user:
/// `$XDG_RUNTIME_DIR/uniserve/storage.sock`, or
/// `/tmp/uniserve-<uid>/storage.sock` without a runtime directory.
pub fn default_socket() -> PathBuf {
    let directory = match std::env::var_os("XDG_RUNTIME_DIR") {
        Some(runtime) if !runtime.is_empty() => PathBuf::from(runtime).join("uniserve"),
        // SAFETY: getuid has no preconditions.
        _ => PathBuf::from(format!("/tmp/uniserve-{}", unsafe { libc::getuid() })),
    };
    directory.join("storage.sock")
}

/// The lock that makes a server the owner of its socket, held for the
/// server's life.
pub struct SocketLock {
    _file: File,
    socket: PathBuf,
}

impl Drop for SocketLock {
    /// The owner removes its socket when it stops, so a later server binds
    /// a fresh one.
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket);
    }
}

/// Bind `socket` for a server, creating its directory with mode 0700 and
/// the socket with mode 0600. Returns `None` when another live server owns
/// the socket. A socket a dead server left behind is replaced.
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
            _file: lock,
            socket: socket.to_owned(),
        },
    )))
}

/// Whether a server accepts connections on `socket`.
pub fn serving(socket: &Path) -> bool {
    UnixStream::connect(socket).is_ok()
}

/// Make sure a Storage Server serves on `socket`: when none does, start
/// `executable` (`uniserve-storage-server`) in a session of its own, not as
/// a child of this process, so that this process's end never ends it.
///
/// A server this starts is given `arguments` after its socket, such as
/// `--tcp` to accept other hosts' heads or its Mooncake connector's
/// settings (`connector::Settings::arguments`). A server that serves already
/// keeps the arguments it was started with.
///
/// The server's `--daemon` start returns once the server listens or found
/// another server listening; the wait for a connection is bounded by
/// `timeout` all the same. Fails when the server cannot be started or does
/// not serve in time.
pub fn ensure(
    executable: &Path,
    socket: &Path,
    timeout: Duration,
    arguments: &[String],
) -> io::Result<()> {
    if serving(socket) {
        return Ok(());
    }
    let mut command = Command::new(executable);
    command
        .arg("serve")
        .arg("--socket")
        .arg(socket)
        .arg("--daemon")
        .args(arguments);
    let status = command.status().map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("starting {}: {error}", executable.display()),
        )
    })?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "{} could not start a storage server on {}: {status}",
            executable.display(),
            socket.display()
        )));
    }
    let deadline = Instant::now() + timeout;
    while !serving(socket) {
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("no storage server serves on {}", socket.display()),
            ));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    Ok(())
}
