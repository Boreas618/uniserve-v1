//! Listener abstraction for TCP and Unix-domain sockets.
//!
//! [`Listener`] presents a single accept interface to Axum over either a TCP or
//! a Unix-domain listener, bound as `admin_config` states.
//!
//! Every accepted TCP connection takes `TCP_NODELAY`: a streamed response
//! writes each SSE event as its own small frame, which Nagle's algorithm would
//! hold back until the previous frame is acknowledged.
//!
//! A Unix-domain socket bound at a path leaves its file behind when the socket
//! closes, so the listener owns the file of a socket it binds. The client API's
//! socket is owned through its lock (`NAME.lock` beside it,
//! [`service::host::own`]): binding takes the lock, replaces a socket file an
//! owner that is gone left, and refuses a path another daemon owns; dropping
//! the listener (at shutdown) removes the file. The admin sockets are taken
//! over by rename ([`Listener::bind_over`]), which a hot restart's successor
//! relies on while its predecessor still serves its own admin requests.
//!
//! A hot restart hands the listening socket to the successor daemon
//! ([`Listener::hand_over`]), with the lock of a Unix socket, which the
//! successor accepts on and owns from then on ([`Listener::resume`]).

use std::io::{ErrorKind, Result};
use std::os::fd::{AsFd as _, OwnedFd};
use std::os::unix::fs::MetadataExt as _;
use std::path::{Path, PathBuf};

use tokio::net::{TcpListener, TcpStream, UnixListener, UnixStream};
use tokio_util::either::Either;
use tracing::trace;

use crate::options::AdminConfig;

/// Runtime listener type used by the OpenAI-compatible HTTP server, which is
/// either a TCP listener or a Unix-domain listener.
#[derive(Debug)]
pub(crate) enum Listener {
    Tcp(TcpListener),
    Unix {
        listener: UnixListener,
        /// What makes this process the owner of the socket's file; `None`
        /// once a successor owns it.
        owner: Option<Owner>,
    },
}

/// How a listener owns its Unix socket's file.
#[derive(Debug)]
pub(crate) enum Owner {
    /// The socket's lock, held for the listener's life; dropping it removes
    /// the file.
    Lock(LockedSocket),
    /// The file this process renamed onto the path, held for its drop, which
    /// removes it while the path still names it.
    Renamed { _file: SocketFile },
}

/// A socket's ownership lock ([`service::host::SocketLock`]), which has no
/// `Debug` of its own.
pub(crate) struct LockedSocket(service::host::SocketLock);

impl std::fmt::Debug for LockedSocket {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("LockedSocket")
    }
}

/// What a successor daemon needs of a listener in a handoff besides its
/// descriptors: whether it is a Unix socket, and that socket's path, which the
/// successor owns from then on through the lock that travels after the
/// socket.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct ListenerHandoff {
    /// The Unix socket's path; `None` for a TCP listener.
    unix: Option<PathBuf>,
}

/// The filesystem entry of a Unix-domain socket this process renamed onto its
/// path, removed when dropped.
///
/// The entry is identified by device and inode, so removal is skipped when the
/// path no longer names the bound socket: a hot restart's successor renamed its
/// own socket onto it.
#[derive(Debug)]
pub(crate) struct SocketFile {
    path: PathBuf,
    device: u64,
    inode: u64,
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

/// The address a listener binds: the TCP host and port, or a Unix socket in
/// place of them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BindAddress<'a> {
    pub(crate) host: &'a str,
    pub(crate) port: u16,
    pub(crate) uds: Option<&'a str>,
}

impl<'a> From<&'a AdminConfig> for BindAddress<'a> {
    fn from(admin: &'a AdminConfig) -> Self {
        Self {
            host: &admin.host,
            port: admin.port,
            uds: admin.uds.as_deref(),
        }
    }
}

impl Listener {
    /// Binds the listener `address` states: a Unix-domain socket at `uds`,
    /// owned through its lock, else TCP on `host` and `port`.
    ///
    /// # Errors
    ///
    /// Fails with `AddrInUse` when another process owns the Unix socket's path
    /// or the path holds a file that is not a socket, and as the bind fails.
    pub(crate) async fn bind<'a, A: Into<BindAddress<'a>>>(address: A) -> Result<Self> {
        let address = address.into();
        match address.uds {
            Some(path) => Self::bind_unix(Path::new(path)),
            None => Ok(Self::Tcp(
                TcpListener::bind((address.host, address.port)).await?,
            )),
        }
    }

    /// Binds a Unix-domain listener at `path` as the owner of its lock.
    fn bind_unix(path: &Path) -> Result<Self> {
        let Some((listener, lock)) = service::host::own(path)? else {
            return Err(std::io::Error::new(
                ErrorKind::AddrInUse,
                format!("another process owns the socket {}", path.display()),
            ));
        };
        listener.set_nonblocking(true)?;
        Ok(Self::Unix {
            listener: UnixListener::from_std(listener)?,
            owner: Some(Owner::Lock(LockedSocket(lock))),
        })
    }

    /// Binds a Unix-domain listener at `path` with file mode `mode`, taking
    /// the path over from whoever holds it: the socket is bound beside the
    /// path and renamed onto it, so the path always names a listening socket.
    ///
    /// The admin listeners bind so: their access is the socket file's
    /// permissions, which are set before the path names the socket, and a hot
    /// restart's successor takes them over while its predecessor finishes the
    /// requests it accepted on its own, now unnamed, socket. The predecessor's
    /// drop leaves the file alone, because the path no longer names the
    /// socket it bound. The path's directory exists: the daemon's instance
    /// directory, or the one the operator stated.
    pub(crate) fn bind_over(path: &Path, mode: u32) -> Result<Self> {
        use std::os::unix::fs::PermissionsExt as _;

        let beside = path.with_extension(format!("{}.binding", std::process::id()));
        let _ = std::fs::remove_file(&beside);
        let listener = std::os::unix::net::UnixListener::bind(&beside)?;
        listener.set_nonblocking(true)?;
        let placed = std::fs::set_permissions(&beside, std::fs::Permissions::from_mode(mode))
            .and_then(|()| std::fs::rename(&beside, path));
        if let Err(error) = placed {
            let _ = std::fs::remove_file(&beside);
            return Err(error);
        }
        let metadata = std::fs::symlink_metadata(path)?;
        Ok(Self::Unix {
            listener: UnixListener::from_std(listener)?,
            owner: Some(Owner::Renamed {
                _file: SocketFile {
                    path: path.to_owned(),
                    device: metadata.dev(),
                    inode: metadata.ino(),
                },
            }),
        })
    }

    /// The listening socket and what the successor of a hot restart needs
    /// beside it ([`Listener::resume`]): the socket's descriptor, then a Unix
    /// socket's lock. The successor owns the socket's file from then on.
    ///
    /// # Errors
    ///
    /// Fails when the socket cannot be duplicated, and for a socket taken over
    /// by rename ([`Listener::bind_over`]), which a successor binds anew.
    pub(crate) fn hand_over(&mut self) -> Result<(ListenerHandoff, Vec<OwnedFd>)> {
        match self {
            Self::Tcp(listener) => Ok((
                ListenerHandoff { unix: None },
                vec![listener.as_fd().try_clone_to_owned()?],
            )),
            Self::Unix { listener, owner } => {
                let path = listener
                    .local_addr()?
                    .as_pathname()
                    .map(Path::to_path_buf)
                    .ok_or_else(|| {
                        std::io::Error::new(ErrorKind::InvalidInput, "an unnamed Unix socket")
                    })?;
                if !matches!(owner, Some(Owner::Lock(_))) {
                    return Err(std::io::Error::new(
                        ErrorKind::InvalidInput,
                        format!("{} is not owned through its lock", path.display()),
                    ));
                }
                let mut descriptors = vec![listener.as_fd().try_clone_to_owned()?];
                if let Some(Owner::Lock(LockedSocket(lock))) = owner.take() {
                    descriptors.push(lock.hand_over());
                }
                Ok((ListenerHandoff { unix: Some(path) }, descriptors))
            }
        }
    }

    /// Accepts on a listening socket a daemon this one replaced handed over,
    /// taking its descriptors from `next` in [`Listener::hand_over`]'s order.
    pub(crate) fn resume(
        handoff: ListenerHandoff,
        next: &mut impl FnMut() -> anyhow::Result<OwnedFd>,
    ) -> anyhow::Result<Self> {
        let socket = next()?;
        match handoff.unix {
            Some(path) => {
                let listener = std::os::unix::net::UnixListener::from(socket);
                listener.set_nonblocking(true)?;
                let lock = service::host::SocketLock::resume(next()?, path);
                Ok(Self::Unix {
                    listener: UnixListener::from_std(listener)?,
                    owner: Some(Owner::Lock(LockedSocket(lock))),
                })
            }
            None => {
                let listener = std::net::TcpListener::from(socket);
                listener.set_nonblocking(true)?;
                Ok(Self::Tcp(TcpListener::from_std(listener)?))
            }
        }
    }

    /// The endpoint the listener is bound at, with the port a TCP listener picked.
    ///
    /// # Errors
    ///
    /// Fails when the socket's address cannot be read, or a Unix socket has no path.
    pub(crate) fn endpoint(&self) -> Result<service::Endpoint> {
        match self {
            Self::Tcp(listener) => Ok(service::Endpoint::tcp(listener.local_addr()?)),
            Self::Unix { listener, .. } => listener
                .local_addr()?
                .as_pathname()
                .map(service::Endpoint::unix)
                .ok_or_else(|| {
                    std::io::Error::new(ErrorKind::InvalidInput, "an unnamed Unix socket")
                }),
        }
    }
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
    use crate::options::AdminConfig;

    /// The listener options of a Unix-domain socket at `path`.
    fn uds(path: &std::path::Path) -> AdminConfig {
        AdminConfig {
            uds: Some(path.to_str().unwrap().to_owned()),
            ..AdminConfig::default()
        }
    }

    /// A socket file left by an owner that is gone (a crash, or a kill) does
    /// not prevent a restart on the same path: its lock went with it.
    #[tokio::test(flavor = "current_thread")]
    async fn a_socket_file_its_owner_left_is_replaced() {
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

    /// A path another daemon owns, or a file that is not a socket, is not
    /// taken over: the bind fails and the path keeps what it holds.
    #[tokio::test(flavor = "current_thread")]
    async fn an_owned_socket_or_another_file_is_not_replaced() {
        let directory = tempfile::tempdir().unwrap();

        let owned = directory.path().join("owned.sock");
        let owner = Listener::bind(&uds(&owned)).await.unwrap();
        let error = Listener::bind(&uds(&owned))
            .await
            .expect_err("an owned socket must not be taken over");
        assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse);
        std::os::unix::net::UnixStream::connect(&owned).unwrap();
        drop(owner);

        let file = directory.path().join("data.sock");
        std::fs::write(&file, b"not a socket").unwrap();
        let error = Listener::bind(&uds(&file))
            .await
            .expect_err("a regular file must not be replaced");
        assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse);
        assert_eq!(std::fs::read(&file).unwrap(), b"not a socket");
    }

    /// A handed-over Unix listener keeps its ownership in the successor: the
    /// path stays refused to others, and the successor's drop removes it.
    #[tokio::test(flavor = "current_thread")]
    async fn a_handed_over_socket_stays_owned_by_the_successor() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("uniserve.sock");
        let mut predecessor = Listener::bind(&uds(&path)).await.unwrap();

        let (handoff, descriptors) = predecessor.hand_over().unwrap();
        drop(predecessor);
        assert!(
            path.exists(),
            "the predecessor leaves the file to its successor"
        );
        let mut descriptors = descriptors.into_iter();
        let successor = Listener::resume(handoff, &mut || {
            descriptors
                .next()
                .ok_or_else(|| anyhow::anyhow!("a descriptor"))
        })
        .unwrap();
        assert!(
            Listener::bind(&uds(&path)).await.is_err(),
            "the successor owns the path"
        );
        std::os::unix::net::UnixStream::connect(&path).unwrap();

        drop(successor);
        assert!(!path.exists(), "the successor removes the file it owns");
    }
}
