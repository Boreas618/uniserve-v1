//! The typed address of every socket a process of the workspace listens on or connects to.
//!
//! An [`Endpoint`] is a Unix socket of this host, by path, or a TCP endpoint, by host and port.
//! Its text form is the one configuration files, command lines and readiness records state:
//! `unix:<path>` for a Unix socket and `<host>:<port>` for TCP, an IPv6 address in brackets
//! (`[::1]:8000`). A listener binds an endpoint and reports the endpoint it bound, with the port
//! the kernel picked when the endpoint stated port 0 ([`crate::Listener::endpoint`]).

use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// The text that starts a Unix socket's endpoint; the rest is the socket's path.
const UNIX_SCHEME: &str = "unix:";

/// Where a socket listens: a Unix socket of this host or a TCP endpoint.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Endpoint {
    /// A Unix socket of this host, which also carries descriptors.
    Unix(PathBuf),
    /// A TCP endpoint: a host name or IP address, and a port. Port 0 asks a listener to pick
    /// one; a bound listener's endpoint states the port it took.
    Tcp {
        /// The host name or IP address, without brackets.
        host: String,
        /// The port.
        port: u16,
    },
}

impl Endpoint {
    /// The Unix socket at `path`.
    pub fn unix(path: impl Into<PathBuf>) -> Self {
        Self::Unix(path.into())
    }

    /// The TCP endpoint of `address`.
    pub fn tcp(address: SocketAddr) -> Self {
        Self::Tcp {
            host: address.ip().to_string(),
            port: address.port(),
        }
    }

    /// The Unix socket's path; `None` for TCP.
    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::Unix(path) => Some(path),
            Self::Tcp { .. } => None,
        }
    }

    /// The TCP port; `None` for a Unix socket.
    pub fn port(&self) -> Option<u16> {
        match self {
            Self::Unix(_) => None,
            Self::Tcp { port, .. } => Some(*port),
        }
    }

    /// This TCP endpoint with its host replaced by `host`: how a listener bound on every
    /// interface (`0.0.0.0`) states the address its peers reach it at. A Unix socket is
    /// returned unchanged.
    pub fn with_host(&self, host: IpAddr) -> Self {
        match self {
            Self::Unix(_) => self.clone(),
            Self::Tcp { port, .. } => Self::tcp(SocketAddr::new(host, *port)),
        }
    }
}

impl fmt::Display for Endpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unix(path) => write!(formatter, "{UNIX_SCHEME}{}", path.display()),
            // An IPv6 address holds colons, so its text form brackets it.
            Self::Tcp { host, port } if host.contains(':') => {
                write!(formatter, "[{host}]:{port}")
            }
            Self::Tcp { host, port } => write!(formatter, "{host}:{port}"),
        }
    }
}

/// Why a text is no endpoint.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{text:?} is no endpoint ({reason}); write `unix:<path>` or `<host>:<port>`")]
pub struct EndpointError {
    text: String,
    reason: &'static str,
}

impl From<EndpointError> for std::io::Error {
    /// A text that is no endpoint, as the input error of the call that read it.
    fn from(error: EndpointError) -> Self {
        Self::new(std::io::ErrorKind::InvalidInput, error)
    }
}

impl FromStr for Endpoint {
    type Err = EndpointError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let refused = |reason| EndpointError {
            text: text.to_owned(),
            reason,
        };
        if let Some(path) = text.strip_prefix(UNIX_SCHEME) {
            if path.is_empty() {
                return Err(refused("the socket's path is empty"));
            }
            return Ok(Self::Unix(PathBuf::from(path)));
        }

        let (host, port) = text
            .rsplit_once(':')
            .ok_or_else(|| refused("it names no port"))?;
        let port = port
            .parse::<u16>()
            .map_err(|_| refused("its port is no number from 0 to 65535"))?;
        let host = match host.strip_prefix('[') {
            Some(bracketed) => bracketed
                .strip_suffix(']')
                .ok_or_else(|| refused("an opening bracket is not closed"))?,
            None if host.contains(':') => {
                return Err(refused("an IPv6 address is written in brackets"));
            }
            None => host,
        };
        if host.is_empty() {
            return Err(refused("its host is empty"));
        }
        Ok(Self::Tcp {
            host: host.to_owned(),
            port,
        })
    }
}

impl serde::Serialize for Endpoint {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> serde::Deserialize<'de> for Endpoint {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every endpoint reads back from the text it writes, an IPv6 host in brackets.
    #[test]
    fn an_endpoint_reads_back_from_its_text() {
        for endpoint in [
            Endpoint::unix("/run/user/1000/uniserve/daemon.x/admin.sock"),
            Endpoint::tcp("127.0.0.1:8000".parse().unwrap()),
            Endpoint::tcp("[::1]:0".parse().unwrap()),
            Endpoint::Tcp {
                host: "rank-0".to_owned(),
                port: 41000,
            },
        ] {
            let text = endpoint.to_string();
            assert_eq!(text.parse::<Endpoint>().unwrap(), endpoint, "{text}");
        }
        assert_eq!(
            Endpoint::tcp("[::1]:9".parse().unwrap()).to_string(),
            "[::1]:9"
        );
    }

    /// A text that states no port, an empty path or an unbracketed IPv6 address is refused.
    #[test]
    fn a_text_without_a_port_or_path_is_refused() {
        for text in [
            "localhost",
            "unix:",
            ":80",
            "::1:80",
            "[::1:80",
            "host:99999",
        ] {
            assert!(text.parse::<Endpoint>().is_err(), "{text} parsed");
        }
    }
}
