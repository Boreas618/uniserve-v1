//! Listener abstraction for TCP and Unix-domain sockets.
//!
//! [`Listener`] presents a single accept interface to Axum over either a TCP or
//! a Unix-domain listener, bound as `server_config` states. `crate::http`
//! binds it and logs its inherent `local_addr` string.
//!
//! Every accepted TCP connection takes `TCP_NODELAY`, as vLLM's Rust listener
//! sets it on accept: a streamed response writes each SSE event as its own
//! small frame, which Nagle's algorithm would hold back until the previous
//! frame is acknowledged. Python servers get the option from asyncio's TCP
//! transports.
//!
//! A Unix-domain socket bound at a path leaves its file behind when the socket
//! closes. The listener therefore owns the file of a socket it binds: binding
//! first removes a stale socket file nobody listens on, and dropping the
//! listener (at shutdown) removes the file again.
//!
//! A hot restart hands a duplicate of the listening socket to the successor
//! daemon ([`Listener::hand_over`]), which accepts on it beside this one and
//! owns a Unix socket's file from then on ([`Listener::resume`]).

use std::io::{ErrorKind, Result};
use std::os::fd::{AsFd as _, OwnedFd};
use std::os::unix::fs::{FileTypeExt as _, MetadataExt as _};
use std::path::{Path, PathBuf};

use tokio::net::{TcpListener, TcpStream, UnixListener, UnixStream};
use tokio_util::either::Either;
use tracing::trace;

use crate::options::ServerOptions;

/// Runtime listener type used by the OpenAI-compatible HTTP server, which is
/// either a TCP listener or a Unix-domain listener.
#[derive(Debug)]
pub(crate) enum Listener {
    Tcp(TcpListener),
    Unix {
        listener: UnixListener,
        /// The socket's file, removed when the listener is dropped; `None`
        /// once a successor owns it.
        socket_file: Option<SocketFile>,
    },
}

/// What a successor daemon needs of a handed-over listener besides its
/// socket: a Unix socket's file, which the successor owns from then on.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct HandedListener {
    unix: bool,
    /// The socket file's path, device and inode.
    socket_file: Option<(PathBuf, u64, u64)>,
}

/// The filesystem entry of a Unix-domain socket this process bound, removed
/// when dropped.
///
/// The entry is identified by device and inode, so removal is skipped when the
/// path no longer names the bound socket, for example after the file was
/// deleted and another server bound the same path.
#[derive(Debug)]
pub(crate) struct SocketFile {
    path: PathBuf,
    device: u64,
    inode: u64,
}

impl SocketFile {
    /// The file's identity, for a successor to own; this process no longer
    /// removes it.
    fn disown(self) -> (PathBuf, u64, u64) {
        let file = std::mem::ManuallyDrop::new(self);
        (file.path.clone(), file.device, file.inode)
    }
}

impl Drop for SocketFile {
    fn drop(&mut self) {
        let bound = std::fs::symlink_metadata(&self.path)
            .is_ok_and(|metadata| metadata.dev() == self.device && metadata.ino() == self.inode);
        if bound {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

impl Listener {
    /// Binds the listener `server` states: a Unix-domain socket at `uds`,
    /// else TCP on `host` and `port`.
    ///
    /// Binding a Unix-domain socket first removes a stale socket file at its
    /// path (see `remove_stale_socket`); it fails with `EADDRINUSE` when the
    /// path holds a live server's socket or a file that is not a socket. The
    /// bound socket's file is removed when the listener is dropped.
    pub(crate) async fn bind(server: &ServerOptions) -> Result<Self> {
        match &server.uds {
            Some(path) => Self::bind_unix(Path::new(path)).await,
            None => Ok(Self::Tcp(
                TcpListener::bind((server.host.as_str(), server.port)).await?,
            )),
        }
    }

    /// Binds a Unix-domain listener at `path` and takes ownership of its file.
    async fn bind_unix(path: &Path) -> Result<Self> {
        remove_stale_socket(path).await?;
        let listener = UnixListener::bind(path)?;
        let metadata = std::fs::symlink_metadata(path)?;
        Ok(Self::Unix {
            listener,
            socket_file: Some(SocketFile {
                path: path.to_owned(),
                device: metadata.dev(),
                inode: metadata.ino(),
            }),
        })
    }

    /// A duplicate of the listening socket and what the successor of a hot
    /// restart needs beside it ([`Listener::resume`]); the successor owns a
    /// Unix socket's file from then on.
    pub(crate) fn hand_over(&mut self) -> Result<(HandedListener, OwnedFd)> {
        match self {
            Self::Tcp(listener) => Ok((
                HandedListener {
                    unix: false,
                    socket_file: None,
                },
                listener.as_fd().try_clone_to_owned()?,
            )),
            Self::Unix {
                listener,
                socket_file,
            } => Ok((
                HandedListener {
                    unix: true,
                    socket_file: socket_file.take().map(SocketFile::disown),
                },
                listener.as_fd().try_clone_to_owned()?,
            )),
        }
    }

    /// Accepts on a listening socket a daemon this one replaced handed over.
    pub(crate) fn resume(handed: HandedListener, descriptor: OwnedFd) -> Result<Self> {
        if handed.unix {
            let listener = std::os::unix::net::UnixListener::from(descriptor);
            listener.set_nonblocking(true)?;
            Ok(Self::Unix {
                listener: UnixListener::from_std(listener)?,
                socket_file: handed.socket_file.map(|(path, device, inode)| SocketFile {
                    path,
                    device,
                    inode,
                }),
            })
        } else {
            let listener = std::net::TcpListener::from(descriptor);
            listener.set_nonblocking(true)?;
            Ok(Self::Tcp(TcpListener::from_std(listener)?))
        }
    }

    /// Returns a log-friendly local address string for either TCP or Unix
    /// sockets.
    pub(crate) fn local_addr(&self) -> Result<String> {
        match self {
            Self::Tcp(listener) => Ok(listener.local_addr()?.to_string()),
            Self::Unix { listener, .. } => Ok(match listener.local_addr()?.as_pathname() {
                Some(path) => format!("unix:{}", path.display()),
                None => "unix:<unnamed>".to_string(),
            }),
        }
    }
}

/// Removes the socket file at `path` when no server listens on it.
///
/// A socket whose connection attempt is refused was left behind by a server
/// that exited without removing it, such as a crashed or killed one. Anything
/// else at `path` (a live server's socket, a file that is not a socket, or a
/// socket that cannot be probed) is left in place for `bind` to refuse.
async fn remove_stale_socket(path: &Path) -> Result<()> {
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return Ok(());
    };
    if !metadata.file_type().is_socket() {
        return Ok(());
    }
    let refused = UnixStream::connect(path)
        .await
        .is_err_and(|error| error.kind() == ErrorKind::ConnectionRefused);
    if refused {
        match std::fs::remove_file(path) {
            Err(error) if error.kind() != ErrorKind::NotFound => return Err(error),
            _ => {}
        }
    }
    Ok(())
}

/// Allows the unified listener to plug directly into `axum::serve(...)`.
impl axum::serve::Listener for Listener {
    type Addr = Either<std::net::SocketAddr, tokio::net::unix::SocketAddr>;
    type Io = Either<TcpStream, UnixStream>;

    /// Accepts the next incoming connection.
    ///
    /// Each arm resolves to Axum's `Listener::accept` for the Tokio listener
    /// rather than Tokio's fallible inherent `accept`. Axum retries accept
    /// errors internally (logging and pausing on errors other than
    /// per-connection ones), so this future resolves only with a connection.
    /// A TCP connection that refuses `TCP_NODELAY` is traced and served
    /// without it.
    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        match self {
            Self::Tcp(listener) => {
                let (io, addr) = axum::serve::Listener::accept(listener).await;
                if let Err(error) = io.set_nodelay(true) {
                    trace!(%error, "failed to enable TCP_NODELAY on an accepted connection");
                }
                (Either::Left(io), Either::Left(addr))
            }
            Self::Unix { listener, .. } => {
                let (io, addr) = axum::serve::Listener::accept(listener).await;
                (Either::Right(io), Either::Right(addr))
            }
        }
    }

    /// Returns the listener socket address.
    fn local_addr(&self) -> Result<Self::Addr> {
        match self {
            Self::Tcp(listener) => listener.local_addr().map(Either::Left),
            Self::Unix { listener, .. } => listener.local_addr().map(Either::Right),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Listener;
    use crate::options::ServerOptions;

    /// The server options of a Unix-domain socket at `path`.
    fn uds(path: &std::path::Path) -> ServerOptions {
        ServerOptions {
            uds: Some(path.to_str().unwrap().to_owned()),
            ..ServerOptions::default()
        }
    }

    /// A socket file left by a server that exited without unlinking it (a
    /// crash, or a kill) does not prevent a restart on the same path.
    #[tokio::test(flavor = "current_thread")]
    async fn a_stale_socket_file_is_replaced() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("uniserve.sock");
        drop(std::os::unix::net::UnixListener::bind(&path).unwrap());
        assert!(path.exists());

        let listener = Listener::bind(&uds(&path)).await.unwrap();

        assert!(matches!(listener, Listener::Unix { .. }));
        std::os::unix::net::UnixStream::connect(&path).unwrap();
    }

    /// Dropping a bound Unix listener, as a clean shutdown does, removes its
    /// socket file.
    #[tokio::test(flavor = "current_thread")]
    async fn dropping_a_unix_listener_removes_its_socket_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("uniserve.sock");
        let mode = uds(&path);

        let listener = Listener::bind(&mode).await.unwrap();
        assert!(path.exists());
        drop(listener);

        assert!(!path.exists());
    }

    /// Only a socket nobody listens on is replaced: a live server's socket and
    /// a file that is not a socket are left in place and the bind fails.
    #[tokio::test(flavor = "current_thread")]
    async fn a_live_socket_or_another_file_is_not_replaced() {
        let directory = tempfile::tempdir().unwrap();

        let live = directory.path().join("live.sock");
        let server = std::os::unix::net::UnixListener::bind(&live).unwrap();
        let error = Listener::bind(&uds(&live))
            .await
            .expect_err("a live server's socket must not be taken over");
        assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse);
        std::os::unix::net::UnixStream::connect(&live).unwrap();
        drop(server);

        let file = directory.path().join("data.sock");
        std::fs::write(&file, b"not a socket").unwrap();
        let error = Listener::bind(&uds(&file))
            .await
            .expect_err("a regular file must not be replaced");
        assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse);
        assert_eq!(std::fs::read(&file).unwrap(), b"not a socket");
    }
}
