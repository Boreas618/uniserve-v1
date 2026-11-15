//! Host services: the servers a host runs once per serving user, and their lifetime.
//!
//! The Storage Server and the Build Server are host services. Each listens on a Unix socket in the
//! user's runtime directory ([`crate::RuntimeDirectory`]), so only that user's processes connect,
//! and outlives the processes that use it. Every file of a service sits beside its socket and is
//! named after it: `NAME.sock` is the socket, `NAME.lock` the lock that makes one server the
//! socket's owner for its life, `NAME.start` the lock its owner holds while it starts,
//! `NAME.ready` the readiness record the owner publishes once it serves (the service's entry in
//! the host's registry of services), and `NAME.log` the log of a server started detached.
//!
//! A server binds with [`bind`]: it becomes the socket's owner, or learns that another server
//! owns it and reads that server's record. Starts are serialized by the start lock, which a
//! starting owner holds until it published its record ([`Starting::finish`]), so a second
//! starter blocks on that lock until the first serves or dies, and never waits by probing.
//!
//! A process that needs a service starts it with [`start`]: when no server owns the socket
//! ([`running`]), it runs the executable's `NAME serve --socket PATH --daemon` command, which
//! leaves its starter's session and process tree ([`detach`]) and reports its readiness record
//! on the starter's standard output; [`start`] returns that record, or the running server's.
//! The service's settings travel in its record, so a caller compares them with the settings it
//! states. A server stops on its `stop` request or SIGTERM ([`crate::stop_signals`]).

use std::fs::{File, OpenOptions};
use std::io::{self, Read as _, Write as _};
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::readiness::{Readiness, ReadyFd, ReadyReader, State};

/// The mode of a service's socket and lock files: its owner alone.
const PRIVATE_FILE: u32 = 0o600;

/// A file of the service whose socket is `socket`: `socket` with the extension `extension`.
fn beside(socket: &Path, extension: &str) -> PathBuf {
    socket.with_extension(extension)
}

/// Opens (creating it, mode 0600) the lock file `path`.
fn open_lock(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .mode(PRIVATE_FILE)
        .open(path)
}

/// Takes `operation` (`LOCK_EX`, `LOCK_SH`, with `LOCK_NB` or without) on `file`; whether it was
/// taken (`false` only for a non-blocking lock another holder has).
fn flock(file: &File, operation: libc::c_int) -> io::Result<bool> {
    loop {
        // SAFETY: flock on a descriptor `file` owns.
        if unsafe { libc::flock(file.as_raw_fd(), operation) } == 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::EWOULDBLOCK) => return Ok(false),
            Some(libc::EINTR) => {}
            _ => return Err(error),
        }
    }
}

/// The lock that makes a server the owner of its socket, held for the server's life; dropping it
/// removes the socket and the published record.
pub struct SocketLock {
    file: File,
    socket: PathBuf,
}

impl SocketLock {
    /// Publishes `record` as the service's entry (`NAME.ready`), replacing any earlier one at
    /// once: the record is written beside and renamed into place, so a reader never sees part of
    /// one.
    ///
    /// # Errors
    ///
    /// What serializing, writing or renaming fails with.
    pub fn publish<S: Serialize>(&self, record: &Readiness<S>) -> io::Result<()> {
        let path = beside(&self.socket, "ready");
        let staged = beside(&self.socket, &format!("ready.{}", std::process::id()));
        let written = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .mode(PRIVATE_FILE)
            .open(&staged)
            .and_then(|mut file| file.write_all(&record.line()?))
            .and_then(|()| std::fs::rename(&staged, &path));
        if written.is_err() {
            let _ = std::fs::remove_file(&staged);
        }
        written
    }

    /// The lock's file, for a successor that goes on with the server to hold
    /// ([`SocketLock::resume`]); this process leaves the socket and the record in place.
    pub fn hand_over(self) -> OwnedFd {
        let lock = std::mem::ManuallyDrop::new(self);
        // SAFETY: `lock` is never dropped, so its file is read out exactly once; its path is
        // leaked with it.
        let file = unsafe { std::ptr::read(&lock.file) };
        OwnedFd::from(file)
    }

    /// Holds the lock a predecessor handed over for the socket at `socket`, which this process
    /// removes when it stops.
    pub fn resume(file: OwnedFd, socket: PathBuf) -> Self {
        Self {
            file: File::from(file),
            socket,
        }
    }
}

impl Drop for SocketLock {
    /// The owner removes its socket and its record when it stops, so a later server binds a
    /// fresh socket and no reader finds the record of a server that is gone.
    fn drop(&mut self) {
        let _ = std::fs::remove_file(beside(&self.socket, "ready"));
        let _ = std::fs::remove_file(&self.socket);
    }
}

/// An owner that has bound its socket and not yet published its record: it holds the start lock,
/// so every other starter waits until [`Starting::finish`] or until this owner ends.
pub struct Starting {
    lock: SocketLock,
    /// The start lock, held exclusively until the record is published.
    _start: File,
}

impl Starting {
    /// Publishes `record` and ends the start: other starters now find this server serving.
    /// Returns the ownership lock the server holds for its life.
    ///
    /// # Errors
    ///
    /// What [`SocketLock::publish`] fails with.
    pub fn finish<S: Serialize>(self, record: &Readiness<S>) -> io::Result<SocketLock> {
        self.lock.publish(record)?;
        Ok(self.lock)
    }
}

impl<S> Bound<S> {
    /// The listener and the start of an owner; `AddrInUse` naming `socket` when another server
    /// owns it, for a caller that serves only a socket it owns (a build that holds a store, a
    /// test's server).
    ///
    /// # Errors
    ///
    /// `AddrInUse` when another server owns the socket.
    pub fn owned(self, socket: &Path) -> io::Result<(UnixListener, Starting)> {
        match self {
            Self::Owner { listener, starting } => Ok((listener, starting)),
            Self::Serving(record) => Err(io::Error::new(
                io::ErrorKind::AddrInUse,
                format!(
                    "another server (process {}) serves on {}",
                    record.pid,
                    socket.display()
                ),
            )),
        }
    }
}

/// What [`bind`] found.
pub enum Bound<S> {
    /// This process owns the socket: the bound listener, and the start it finishes once it
    /// serves.
    Owner {
        listener: UnixListener,
        starting: Starting,
    },
    /// Another server owns the socket and serves: its published record.
    Serving(Box<Readiness<S>>),
}

/// Binds `socket` for a server, creating its directory with mode 0700 and the socket with mode
/// 0600, or finds the server that owns it.
///
/// The start lock serializes starts: a server that is starting holds it until it published its
/// record, so this call blocks until that server serves or ends. The ownership lock then tells
/// whether a server owns the socket; one that owns it published its record, which is returned
/// ([`Bound::Serving`]). Otherwise this process owns the socket: a socket a dead server left is
/// replaced, and the start lock stays held until the caller finishes its start.
///
/// # Errors
///
/// Fails when the directory, a lock or the socket cannot be created, or a serving owner's record
/// cannot be read.
pub fn bind<S: DeserializeOwned>(socket: &Path) -> io::Result<Bound<S>> {
    let directory = socket.parent().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "the socket has no directory")
    })?;
    std::fs::create_dir_all(directory)?;
    std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))?;

    let start = open_lock(&beside(socket, "start"))?;
    flock(&start, libc::LOCK_EX)?;
    let lock = open_lock(&beside(socket, "lock"))?;
    if !flock(&lock, libc::LOCK_EX | libc::LOCK_NB)? {
        // The owner finished its start before this process took the start lock, so its record
        // is published.
        return read_published(socket).map(|record| Bound::Serving(Box::new(record)));
    }

    // The lock is ours, so a socket file present now belongs to no live server.
    match std::fs::remove_file(socket) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    let listener = UnixListener::bind(socket)?;
    std::fs::set_permissions(socket, std::fs::Permissions::from_mode(PRIVATE_FILE))?;
    Ok(Bound::Owner {
        listener,
        starting: Starting {
            lock: SocketLock {
                file: lock,
                socket: socket.to_owned(),
            },
            _start: start,
        },
    })
}

/// The record of the server that owns `socket`, once it serves; `None` when no server owns it.
/// A server that is starting is waited for (its start lock), not probed.
///
/// # Errors
///
/// Fails when a lock cannot be taken or a serving owner's record cannot be read.
pub fn running<S: DeserializeOwned>(socket: &Path) -> io::Result<Option<Readiness<S>>> {
    let start = match OpenOptions::new().write(true).open(beside(socket, "start")) {
        Ok(start) => start,
        // No server ever started here.
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    flock(&start, libc::LOCK_SH)?;
    let lock = match OpenOptions::new().write(true).open(beside(socket, "lock")) {
        Ok(lock) => lock,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    if flock(&lock, libc::LOCK_SH | libc::LOCK_NB)? {
        return Ok(None);
    }
    read_published(socket).map(Some)
}

/// The record the owner of `socket` published.
fn read_published<S: DeserializeOwned>(socket: &Path) -> io::Result<Readiness<S>> {
    let path = beside(socket, "ready");
    let line = std::fs::read(&path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "the server that owns {} published no record at {}: {error}",
                socket.display(),
                path.display()
            ),
        )
    })?;
    Readiness::parse(line.strip_suffix(b"\n").unwrap_or(&line))
}

/// Leaves the starter: forks, lets the parent relay the child's readiness records and exit, and
/// makes the child the leader of a session of its own with its standard input on `/dev/null` and
/// its standard output and error appended to `NAME.log` beside `socket`.
///
/// Returns, in the child, the descriptor its readiness records go to; the parent never returns.
/// The parent writes every record the child reports to `relay` (its standard output when
/// `None`) and exits once the child closes the descriptor: with success when the child reported
/// a record, with failure when it ended first. Must be called before the process starts a
/// thread: only the calling thread survives a fork.
///
/// # Errors
///
/// Fails when the log or the readiness pipe cannot be created, or the process cannot fork.
pub fn detach(socket: &Path, relay: Option<ReadyFd>) -> io::Result<ReadyFd> {
    let directory = socket.parent().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "the socket has no directory")
    })?;
    std::fs::create_dir_all(directory)?;
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(PRIVATE_FILE)
        .open(beside(socket, "log"))?;

    let mut ends = [0; 2];
    // SAFETY: pipe2 fills the two descriptors it creates.
    if unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: pipe2 returned two new descriptors this function owns.
    let (reader, writer) =
        unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) };

    // SAFETY: the caller guarantees the process has one thread, so the child may run anything.
    match unsafe { libc::fork() } {
        -1 => Err(io::Error::last_os_error()),
        0 => {
            drop(reader);
            drop(relay);
            // SAFETY: setsid, open, dup2 and close on descriptors this process owns.
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
            Ok(ReadyFd::from_owned(writer))
        }
        _ => {
            drop(writer);
            // The child writes its records and closes the pipe once it serves; a child that
            // dies first closes it without one.
            let mut records = Vec::new();
            let read = File::from(reader).read_to_end(&mut records);
            let reported = read.is_ok() && records.contains(&b'\n');
            let relayed = match relay {
                Some(relay) => relay.write_bytes(&records),
                None => io::stdout().lock().write_all(&records),
            };
            let started = reported && relayed.is_ok();
            // SAFETY: `_exit` ends the starter without running handlers that belong to the
            // child.
            unsafe { libc::_exit(if started { 0 } else { 1 }) }
        }
    }
}

/// Makes sure the host service `name` serves on `socket` and returns its readiness record: the
/// record of the server that owns the socket ([`running`]), or, when none does, of the server
/// started as `executable name serve --socket SOCKET --daemon ARGUMENTS` with `input` on its
/// standard input.
///
/// The service runs in a session of its own, not as a child of this process, so this process's
/// end never ends it. The starter reports the server's records on its standard output once the
/// server serves ([`detach`]) and exits once the server closes its readiness descriptor, so the
/// wait for them ends with the server serving or with its end, which is reported with its log.
///
/// # Errors
///
/// Fails when the service cannot be started, or ends before it serves.
pub fn start<S: DeserializeOwned>(
    executable: &Path,
    name: &str,
    socket: &Path,
    arguments: &[String],
    input: Option<&[u8]>,
) -> io::Result<Readiness<S>> {
    if let Some(record) = running(socket)? {
        return Ok(record);
    }

    let log = beside(socket, "log");
    let mut starter = Command::new(executable)
        .args([name, "serve", "--socket"])
        .arg(socket)
        .arg("--daemon")
        .args(arguments)
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("starting {}: {error}", executable.display()),
            )
        })?;
    if let (Some(input), Some(mut stdin)) = (input, starter.stdin.take()) {
        stdin.write_all(input)?;
    }
    let output = starter
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("the starter's standard output was not piped"))?;
    let record = ReadyReader::new(OwnedFd::from(output)).wait(State::Serving, Duration::MAX);
    let status = starter.wait()?;
    match record {
        Ok(record) if status.success() => Ok(record),
        Ok(_) => Err(io::Error::other(format!(
            "{} reported a {name} on {} but exited with {status}; its log is {}",
            executable.display(),
            socket.display(),
            log.display()
        ))),
        Err(error) => Err(io::Error::new(
            error.kind(),
            format!(
                "{} did not start a {name} on {} ({status}): {error}; its log is {}",
                executable.display(),
                socket.display(),
                log.display()
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Endpoint;
    use crate::readiness::Endpoints;

    fn record(socket: &Path) -> Readiness<u32> {
        Readiness::of_process(
            State::Serving,
            None,
            Endpoints {
                storage: Some(Endpoint::unix(socket)),
                ..Endpoints::default()
            },
            7,
        )
    }

    /// A socket has one owner: a second bind while the first server owns it finds that server
    /// serving with the record it published, and the socket binds again once the owner stops,
    /// leaving no record behind.
    #[test]
    fn one_server_owns_a_socket() {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("service.sock");
        assert!(running::<u32>(&socket).unwrap().is_none());

        let Bound::Owner { listener, starting } = bind::<u32>(&socket).unwrap() else {
            panic!("the first server does not own the socket");
        };
        let published = record(&socket);
        let lock = starting.finish(&published).unwrap();
        match bind::<u32>(&socket).unwrap() {
            Bound::Serving(found) => assert_eq!(*found, published),
            Bound::Owner { .. } => panic!("a second server owns the socket"),
        }
        assert_eq!(running::<u32>(&socket).unwrap(), Some(published));

        drop(listener);
        drop(lock);
        assert!(!socket.exists());
        assert!(running::<u32>(&socket).unwrap().is_none());
        assert!(matches!(bind::<u32>(&socket).unwrap(), Bound::Owner { .. }));
    }

    /// A starter that finds a server starting waits until it published its record, and then
    /// finds it serving rather than taking the socket over.
    #[test]
    fn a_second_starter_waits_for_the_first_to_serve() {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("service.sock");
        let Bound::Owner {
            listener: _listener,
            starting,
        } = bind::<u32>(&socket).unwrap()
        else {
            panic!("the first server does not own the socket");
        };

        let waiting = {
            let socket = socket.clone();
            std::thread::spawn(move || match bind::<u32>(&socket).unwrap() {
                Bound::Serving(found) => *found,
                Bound::Owner { .. } => panic!("the second starter took the socket over"),
            })
        };
        // The second starter cannot pass the start lock before the first finishes its start.
        assert!(!waiting.is_finished());
        let published = record(&socket);
        let _lock = starting.finish(&published).unwrap();
        assert_eq!(waiting.join().unwrap(), published);
    }
}
