//! The readiness record: what a server reports once it is bound and once it serves.
//!
//! Every server binds all of its listeners when it starts, before any slow work, and accepts
//! port 0; it then reports the endpoints it bound in one typed record ([`Readiness`]). A caller
//! reads the record instead of choosing ports or deriving socket paths, and waits for the record
//! that states [`State::Serving`] instead of probing. The record travels on a descriptor the
//! caller passes (`--ready-fd`, the `sd_notify` pattern, [`ReadyFd`]), one JSON object per line:
//! a server writes one record when its listeners are bound ([`State::Listening`]) and another
//! once it serves, then closes the descriptor. A server that ends before it serves closes the
//! descriptor without the second record, which the reader sees as end of file. The same record
//! is served by the server's status route or request.

use std::fs::File;
use std::io::{self, BufRead as _, Write as _};
use std::os::fd::{FromRawFd as _, OwnedFd, RawFd};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::Endpoint;

/// How far a server's start has come.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    /// Every listener is bound and its endpoint stated; connections wait in the kernel's
    /// backlog until the server serves. A deployment's members join the leader from here on.
    Listening,
    /// The server answers on every endpoint.
    Serving,
}

/// The endpoints a server bound, each `None` when the server has no such listener.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoints {
    /// The client API (a daemon's HTTP server).
    pub client: Option<Endpoint>,
    /// The operator's admin API (a daemon's Unix socket).
    pub admin: Option<Endpoint>,
    /// The harness's subset of the admin API (a daemon's Unix socket).
    pub harness: Option<Endpoint>,
    /// Where a deployment's other nodes join its leader (TCP).
    pub join: Option<Endpoint>,
    /// A Storage Server's socket on its host, or the one a daemon's deployment uses.
    pub storage: Option<Endpoint>,
    /// A Storage Server's TCP endpoint for other hosts' heads and peers.
    pub peer: Option<Endpoint>,
    /// A Build Server's socket.
    pub build: Option<Endpoint>,
}

/// What a server reports of itself: how far its start has come, its process, its instance
/// directory, the endpoints it bound and the settings it serves with (`S`, the server's own
/// settings type; `()` for a server whose settings no caller compares).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Readiness<S = ()> {
    pub state: State,
    /// The server's process.
    pub pid: u32,
    /// The server's instance directory ([`crate::Instance`]); `None` for a deployment's server,
    /// whose files lie beside its socket.
    pub instance: Option<PathBuf>,
    pub endpoints: Endpoints,
    pub settings: S,
}

impl<S> Readiness<S> {
    /// A record of this process in `state`.
    pub fn of_process(
        state: State,
        instance: Option<PathBuf>,
        endpoints: Endpoints,
        settings: S,
    ) -> Self {
        Self {
            state,
            pid: std::process::id(),
            instance,
            endpoints,
            settings,
        }
    }
}

impl<S: Serialize> Readiness<S> {
    /// The record as one line of JSON, newline included.
    ///
    /// # Errors
    ///
    /// Fails when the settings do not serialize.
    pub fn line(&self) -> io::Result<Vec<u8>> {
        let mut line = serde_json::to_vec(self).map_err(io::Error::other)?;
        line.push(b'\n');
        Ok(line)
    }
}

impl<S: DeserializeOwned> Readiness<S> {
    /// The record one line of JSON states.
    ///
    /// # Errors
    ///
    /// `InvalidData` when the line is no record.
    pub fn parse(line: &[u8]) -> io::Result<Self> {
        serde_json::from_slice(line).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("the readiness record does not parse: {error}"),
            )
        })
    }
}

/// The write end of the descriptor a server's caller passed for its readiness records
/// (`--ready-fd`).
#[derive(Debug)]
pub struct ReadyFd(File);

impl ReadyFd {
    /// Takes the inherited descriptor `fd`, which the caller opened for writing and passed to
    /// this process; it turns close-on-exec, so no process this one starts holds it open.
    ///
    /// # Safety
    ///
    /// `fd` must be an open descriptor this process inherited for this purpose and that nothing
    /// else in the process owns.
    pub unsafe fn inherit(fd: RawFd) -> io::Result<Self> {
        // SAFETY: the caller guarantees the descriptor is open and owned by nothing else.
        let owned = unsafe { OwnedFd::from_raw_fd(fd) };
        // SAFETY: fcntl takes the descriptor this function now owns and integers.
        if unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(File::from(owned)))
    }

    /// Wraps the write end of a pipe this process made itself.
    pub fn from_owned(fd: OwnedFd) -> Self {
        Self(File::from(fd))
    }

    /// Writes `record` as one line. A reader that is gone loses nothing the server needs: the
    /// failure is returned for the caller to log.
    ///
    /// # Errors
    ///
    /// What serializing or writing fails with.
    pub fn report<S: Serialize>(&mut self, record: &Readiness<S>) -> io::Result<()> {
        self.0.write_all(&record.line()?)?;
        self.0.flush()
    }

    /// Writes records another process reported, as they arrived: a detached server's starter
    /// relays its child's records so ([`crate::host::detach`]).
    pub(crate) fn write_bytes(mut self, records: &[u8]) -> io::Result<()> {
        self.0.write_all(records)?;
        self.0.flush()
    }
}

/// Reads readiness records from the read end of a caller's readiness descriptor.
pub struct ReadyReader(io::BufReader<File>);

impl ReadyReader {
    /// Reads from `fd`.
    pub fn new(fd: OwnedFd) -> Self {
        Self(io::BufReader::new(File::from(fd)))
    }

    /// Waits up to `timeout` for the next record; `None` when the server closed the descriptor
    /// without writing another, as a server that ends does.
    ///
    /// # Errors
    ///
    /// `TimedOut` when no record arrived in time, `InvalidData` when a line is no record, and
    /// what reading fails with.
    pub fn next<S: DeserializeOwned>(
        &mut self,
        timeout: Duration,
    ) -> io::Result<Option<Readiness<S>>> {
        use std::os::fd::AsRawFd as _;

        let deadline = Instant::now().checked_add(timeout);
        let mut line = Vec::new();
        loop {
            // Bytes the reader buffered already need no wait.
            if self.0.buffer().is_empty() {
                let remaining = deadline
                    .map(|deadline| deadline.saturating_duration_since(Instant::now()))
                    .unwrap_or(Duration::MAX);
                let mut descriptor = [libc::pollfd {
                    fd: self.0.get_ref().as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                }];
                crate::connection::wait_ready(&mut descriptor, Some(remaining))?;
                if descriptor[0].revents == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        format!("no readiness record arrived within {timeout:?}"),
                    ));
                }
            }
            let available = self.0.fill_buf()?;
            if available.is_empty() {
                // End of file: a partial line is no record.
                return if line.is_empty() {
                    Ok(None)
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "the readiness descriptor closed inside a record",
                    ))
                };
            }
            match available.iter().position(|byte| *byte == b'\n') {
                Some(end) => {
                    line.extend_from_slice(&available[..end]);
                    self.0.consume(end + 1);
                    return Readiness::parse(&line).map(Some);
                }
                None => {
                    let length = available.len();
                    line.extend_from_slice(available);
                    self.0.consume(length);
                }
            }
        }
    }

    /// Waits up to `timeout` for the record that states `state`, skipping earlier ones.
    ///
    /// # Errors
    ///
    /// `UnexpectedEof` when the server closed the descriptor first, and what
    /// [`ReadyReader::next`] fails with.
    pub fn wait<S: DeserializeOwned>(
        &mut self,
        state: State,
        timeout: Duration,
    ) -> io::Result<Readiness<S>> {
        let deadline = Instant::now().checked_add(timeout);
        loop {
            let remaining = deadline
                .map(|deadline| deadline.saturating_duration_since(Instant::now()))
                .unwrap_or(Duration::MAX);
            match self.next(remaining)? {
                Some(record) if record.state == state => return Ok(record),
                Some(_) => {}
                None => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!("the server ended before it was {state:?}"),
                    ));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pipe() -> (OwnedFd, OwnedFd) {
        let mut ends = [0; 2];
        // SAFETY: pipe2 fills the two descriptors it creates.
        let created = unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_CLOEXEC) };
        assert_eq!(created, 0);
        // SAFETY: pipe2 returned two new descriptors this test owns.
        unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) }
    }

    /// A reader waiting for the serving record skips the listening one and reads the endpoints
    /// the server bound; a server that closes the descriptor first ends the wait.
    #[test]
    fn a_reader_waits_for_the_serving_record() {
        let (reader, writer) = pipe();
        let endpoints = Endpoints {
            client: Some(Endpoint::tcp("127.0.0.1:41000".parse().unwrap())),
            ..Endpoints::default()
        };
        let mut ready = ReadyFd::from_owned(writer);
        ready
            .report(&Readiness::of_process(
                State::Listening,
                None,
                endpoints.clone(),
                (),
            ))
            .unwrap();
        ready
            .report(&Readiness::of_process(
                State::Serving,
                None,
                endpoints.clone(),
                (),
            ))
            .unwrap();
        drop(ready);

        let mut reader = ReadyReader::new(reader);
        let serving: Readiness = reader.wait(State::Serving, Duration::from_secs(5)).unwrap();
        assert_eq!(serving.endpoints, endpoints);
        assert_eq!(serving.pid, std::process::id());
        let ended = reader
            .wait::<()>(State::Serving, Duration::from_secs(5))
            .unwrap_err();
        assert_eq!(ended.kind(), io::ErrorKind::UnexpectedEof);
    }

    /// A server that writes nothing within the timeout ends the wait with `TimedOut`.
    #[test]
    fn a_silent_server_times_out() {
        let (reader, _writer) = pipe();
        let mut reader = ReadyReader::new(reader);
        let silent = reader.next::<()>(Duration::from_millis(20)).unwrap_err();
        assert_eq!(silent.kind(), io::ErrorKind::TimedOut);
    }
}
