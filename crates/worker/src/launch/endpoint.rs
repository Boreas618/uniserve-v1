//! The rank's channel endpoint and its registration with the head.
//!
//! A rank binds its endpoint and reports it before loading weights, so the
//! head can connect while the model loads and observe startup failures as
//! process exits. The report is one JSON line ([`RankReport`]) written to the
//! head's registration address. The rank takes the head's connection once it
//! is ready to serve.

use std::io::Write;
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

use uniserve_ipc::{Channel, Listener, RankReport, TCP_CHANNEL, Wake};

use crate::config::IpcConfig;
use crate::{Error, Result};

/// Bounds connecting to the registration address and writing the report; it
/// does not bound model startup.
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(60);

/// Bounds the wait for the head's connection once the rank is ready to
/// serve. The head connects as soon as it has read the report, which is
/// before the rank finishes loading, so only a head that is gone exceeds it.
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(60);

/// The rank's bound channel address, before the head has connected.
pub struct Endpoint {
    listener: Listener,
}

impl Endpoint {
    /// Bind the channel the placement calls for: a TCP port on every
    /// interface, or a Unix socket at a unique path.
    pub fn bind(ipc: &IpcConfig, rank: u32) -> Result<Self> {
        let address = if ipc.channel_transport == TCP_CHANNEL {
            "0.0.0.0:0".to_owned()
        } else {
            let name = format!(
                "uniserve-{}-{rank}-{}.sock",
                std::process::id(),
                uuid::Uuid::new_v4().simple()
            );
            format!("unix:{}", std::env::temp_dir().join(name).display())
        };
        let listener = Listener::bind(&[&address], ipc.max_payload_bytes, ipc.queue_depth)
            .map_err(|error| Error::Transport(format!("failed to bind the channel: {error:#}")))?;
        Ok(Self { listener })
    }

    /// Report the endpoint to the head.
    ///
    /// A TCP endpoint reports the interface address used to reach the head,
    /// since a wildcard bind address cannot be dialed from another host.
    pub fn register(&self, ipc: &IpcConfig, worker_id: &str, rank: u32) -> Result<()> {
        let io = |error: std::io::Error| {
            Error::Transport(format!(
                "registering with {}: {error}",
                ipc.registration_address
            ))
        };
        let mut failure = std::io::Error::new(
            std::io::ErrorKind::AddrNotAvailable,
            "registration address resolves to no endpoints",
        );
        let mut connection = None;
        for address in ipc.registration_address.to_socket_addrs().map_err(io)? {
            match TcpStream::connect_timeout(&address, REGISTRATION_TIMEOUT) {
                Ok(stream) => {
                    connection = Some(stream);
                    break;
                }
                Err(error) => failure = error,
            }
        }
        let mut connection = connection.ok_or_else(|| io(failure))?;
        connection
            .set_write_timeout(Some(REGISTRATION_TIMEOUT))
            .map_err(io)?;

        let mut endpoint = self.listener.addresses()[0].clone();
        if ipc.channel_transport == TCP_CHANNEL {
            let (_, port) = endpoint
                .rsplit_once(':')
                .ok_or_else(|| Error::Invalid("TCP endpoint must name a port".into()))?;
            let port = port
                .parse()
                .map_err(|error| Error::Invalid(format!("invalid endpoint port: {error}")))?;
            endpoint = SocketAddr::new(connection.local_addr().map_err(io)?.ip(), port).to_string();
        }
        let report = RankReport {
            worker_id: worker_id.to_owned(),
            rank,
            transport: ipc.channel_transport.clone(),
            endpoint,
        };
        let mut line = serde_json::to_vec(&report)
            .map_err(|error| Error::Invalid(format!("encoding the rank report: {error}")))?;
        line.push(b'\n');
        connection.write_all(&line).map_err(io)
    }

    /// Take the head's connection, which ends the endpoint's listening.
    pub fn accept(self) -> Result<Connection> {
        let channel = self
            .listener
            .accept(CONNECTION_TIMEOUT)
            .map_err(|error| Error::Transport(format!("accepting the head: {error}")))?
            .ok_or_else(|| Error::Transport("the head did not connect to the channel".into()))?;
        Ok(Connection { channel })
    }
}

/// The rank's connection to the head.
pub struct Connection {
    channel: Channel,
}

impl Connection {
    pub fn channel(&mut self) -> &mut Channel {
        &mut self.channel
    }

    /// The wake completion callbacks fire to interrupt the service's waits.
    pub fn completion_wake(&self) -> Wake {
        self.channel.wake()
    }
}
