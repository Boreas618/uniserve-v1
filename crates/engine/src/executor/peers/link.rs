//! The peer link between the engine processes of an epoch: the codec its messages take on the
//! workspace's one connection (`service::Connection`), and the scheduler node's listener.
//!
//! Each [`MainProcessRequest`], [`MainProcessResponse`], [`PeerRegistration`] and [`PeerAccepted`]
//! is one length-prefixed frame (`service::frame`) whose payload is the message in MessagePack
//! (`rmp-serde`). The message types are the only definition of the protocol: the encoding is the
//! one their `serde` derives describe, and both ends are built from this workspace
//! ([`super::registration::VERSION`]), so no schema, generated code or version field accompanies
//! it, and no bound beyond the length field's limits a message.
//!
//! The encoding is positional: a struct is the array of its fields in declaration order, without
//! their names. Every message type must therefore decode from a sequence, which rules out three
//! `serde` forms in anything a message reaches: a flattened member, a struct variant of an
//! adjacently tagged or untagged enum, and a field omitted when absent (`skip_serializing_if`)
//! anywhere but last in its struct. A message type also gives none of its fields a default: a
//! shorter array, as another build would send, must fail to decode rather than be completed. The
//! round-trip tests of this crate carry every message family through its encoding.

use std::io;
use std::net::TcpListener;
use std::os::fd::AsRawFd as _;

use service::Connection;

use super::messages::{MainProcessRequest, MainProcessResponse};
use super::registration::{PeerAccepted, PeerRegistration};

/// Implements the workspace's connection codec for each message of the peer link.
macro_rules! peer_messages {
    ($($message:ty),* $(,)?) => {
        $(
            impl service::Message for $message {
                const FRAMING: service::Framing = service::Framing::Length;

                fn encode(&self) -> io::Result<Vec<u8>> {
                    rmp_serde::to_vec(self).map_err(|error| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("encoding a peer message: {error}"),
                        )
                    })
                }

                fn decode(payload: &[u8]) -> io::Result<Self> {
                    rmp_serde::from_slice(payload).map_err(|error| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("decoding a peer message: {error}"),
                        )
                    })
                }
            }
        )*
    };
}

peer_messages!(
    MainProcessRequest,
    MainProcessResponse,
    PeerRegistration,
    PeerAccepted
);

/// Accepts the peer link's connections on one or more addresses.
pub struct Listener {
    listeners: Vec<service::Listener>,
    /// The bound addresses, in the order they were given.
    addresses: Vec<String>,
}

impl Listener {
    /// Binds every address in `addresses`: `unix:<path>` or `<host>:<port>`, where port zero
    /// picks a free port that [`Self::addresses`] then reports.
    ///
    /// # Errors
    ///
    /// Fails when an address cannot be bound, for example a Unix socket path that exists.
    pub fn bind(addresses: &[&str]) -> io::Result<Self> {
        let listeners = addresses
            .iter()
            .map(|address| service::Listener::bind(address))
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Self::over(listeners))
    }

    /// Accepts on a TCP socket that is already bound and listening: one the process inherited
    /// from the daemon that reserved it, so the address its peers were given is live before the
    /// process starts.
    ///
    /// # Errors
    ///
    /// Fails when the socket's address cannot be read or it cannot be made non-blocking.
    pub(crate) fn from_tcp(listener: TcpListener) -> io::Result<Self> {
        Ok(Self::over(vec![service::Listener::from_tcp(listener)?]))
    }

    /// The bound addresses, in the order they were given to [`Self::bind`].
    pub fn addresses(&self) -> &[String] {
        &self.addresses
    }

    /// Takes one waiting connection, if any, without blocking.
    ///
    /// # Errors
    ///
    /// Fails when accepting fails.
    pub(crate) fn try_accept(&self) -> io::Result<Option<Connection>> {
        for listener in &self.listeners {
            if let Some(connection) = listener.try_accept()? {
                return Ok(Some(connection));
            }
        }
        Ok(None)
    }

    /// The descriptors that become readable when a connection waits.
    pub(crate) fn poll_fds(&self) -> Vec<libc::pollfd> {
        self.listeners
            .iter()
            .map(|listener| libc::pollfd {
                fd: listener.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            })
            .collect()
    }

    fn over(listeners: Vec<service::Listener>) -> Self {
        let addresses = listeners
            .iter()
            .map(|listener| listener.address().to_owned())
            .collect();
        Self {
            listeners,
            addresses,
        }
    }
}

#[cfg(test)]
mod tests;
