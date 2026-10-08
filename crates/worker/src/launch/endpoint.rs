//! The rank's channel endpoint and its registration with the head.
//!
//! A rank binds its endpoint and reports it before loading weights, so the
//! head can connect while the model loads and observe startup failures as
//! process exits. The report is one JSON line ([`RankReport`]) written to the
//! head's registration address.

use std::io::Write;
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

use uniserve_ipc::{RankReport, RankServer, SOCKET_CHANNEL, Wake, service_name};

use crate::config::IpcConfig;
use crate::{Error, Result};

/// Bounds connecting to the registration address and writing the report; it
/// does not bound model startup.
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(60);

/// The rank's bound channel endpoint.
pub struct Endpoint {
    // Field order is drop order: on a shared-storage channel the completion
    // wake holds a port of the endpoint's node and must be released before
    // the server releases the node.
    wake: Wake,
    server: RankServer,
    /// The endpoint the rank reports.
    name: String,
}

impl Endpoint {
    /// Bind the channel the placement calls for. A socket endpoint binds
    /// every interface; a shared-storage endpoint takes a unique service
    /// name.
    pub fn bind(ipc: &IpcConfig, rank: u32) -> Result<Self> {
        let service = if ipc.channel_transport == SOCKET_CHANNEL {
            "0.0.0.0".to_owned()
        } else {
            service_name(&format!(
                "{}_{rank}_{}",
                std::process::id(),
                uuid::Uuid::new_v4().simple()
            ))
        };
        let server = RankServer::bind(
            &ipc.channel_transport,
            &service,
            ipc.max_payload_bytes,
            ipc.queue_depth,
        )
        .map_err(|error| Error::Transport(format!("failed to bind IPC service: {error:#}")))?;
        Ok(Self {
            wake: server.completion_wake(),
            name: server.endpoint(&service),
            server,
        })
    }

    /// Report the endpoint to the head.
    ///
    /// A socket endpoint reports the interface address used to reach the
    /// head, since a wildcard bind address cannot be dialed from another
    /// host.
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

        let mut endpoint = self.name.clone();
        if ipc.channel_transport == SOCKET_CHANNEL {
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

    pub fn server(&mut self) -> &mut RankServer {
        &mut self.server
    }

    /// The wake completion callbacks fire to interrupt the service's waits.
    pub fn completion_wake(&self) -> Wake {
        self.wake.clone()
    }
}
