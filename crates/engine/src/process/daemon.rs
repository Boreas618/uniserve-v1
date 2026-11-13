//! The daemon's two interfaces, as the main process holds them.
//!
//! The daemon has two interfaces and they are two sockets with two vocabularies: the **request
//! interface** (`crates/proto/src/request.proto`) carries the epoch's requests, their events, the
//! engine's state and its metrics, one connection per daemon front (the daemon that launched the
//! process, and each daemon that attached since); the **process interface**
//! (`crates/proto/src/process.proto`) carries the Storage Server claim, every command the daemon
//! sends this engine and the engine's reports.
//!
//! Each connection is read by a reader thread of its own ([`read_requests`], [`read_commands`]),
//! which blocks on the socket and hands each frame to the control thread's inbox; the control
//! thread keeps a write half of each and writes frames to it ([`send`], [`report`]). The handshakes
//! run before the readers start: [`connect`] on the request interface, [`claim`] on the process
//! interface, and [`attach`] for a further front.

use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use crossbeam_channel::Sender;
use proto::config::EngineArgs;
use proto::process::{downstream, request, upstream};
use proto::request::ArtifactMedia;
use proto::request::wire;
use service::Connection;

use super::Input;

/// How long the daemon may take to welcome this engine on either interface, which it does at
/// once: the bound fails a wrong address instead of hanging the process.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(60);

/// How long the daemon may take to reply to the storage claim: starting the node's Storage
/// Server is the slow part, and it is the daemon's to do.
const CLAIM_TIMEOUT: Duration = Duration::from_secs(60);

/// How long an attaching daemon has to welcome this engine: the daemon's own connect bound, so a
/// hot restart that hangs is reported on the daemon's side.
const ATTACH_TIMEOUT: Duration = Duration::from_secs(10);

/// How long a reader thread waits on its socket before it looks again: a read wakes as soon as a
/// frame arrives or the connection closes, so this bounds nothing but a lost wake.
const READ_WAIT: Duration = Duration::from_secs(60);

/// The most artifact bytes one frame carries to a front on another host.
const ARTIFACT_PART: usize = 1 << 20;

/// The request id of the storage claim, the one request this process sends on the process
/// interface.
const CLAIM: u64 = 1;

/// Says hello on the request interface over `connection` as the engine `launch.engine`, reads
/// the daemon's welcome, and states `Starting` before anything loads.
///
/// # Errors
///
/// When the daemon closes, refuses this engine, or does not welcome it within a minute.
pub fn connect(connection: &mut Connection, launch: &EngineArgs) -> Result<()> {
    hello(connection, launch.engine, HANDSHAKE_TIMEOUT)?;

    // `Starting` goes out before the rank load, which takes minutes, so the daemon reads a
    // process that is alive rather than one that vanished after its hello.
    send(
        connection,
        wire::from_engine::Kind::State(wire::State {
            state: wire::EngineState::Starting as i32,
            info: None,
            error: String::new(),
        }),
        None,
    )
    .context("stating that this engine starts")
}

/// Says hello on the process interface over `process`, then asks this node's daemon for the
/// node's Storage Server claim, before any storage is mapped.
///
/// Returns the Storage Server launch this process's ranks open their storage with, and the
/// commands the daemon sent while the claim was outstanding, in order.
///
/// # Errors
///
/// When the launch states engine 0, the daemon refuses the process or its claim, lists no node,
/// or does not answer in time.
pub(crate) fn claim(
    process: &mut Connection,
    launch: &EngineArgs,
) -> Result<(ipc::StorageLaunch, Vec<proto::process::Command>)> {
    // The daemon keys this connection by the engine number, so a process that states none would
    // be filed in the place of a process it can never command.
    if launch.engine == 0 {
        bail!(
            "the launch file states engine 0, which names no engine process: the daemon \
             addresses its commands by this number"
        );
    }
    process.deliver(&proto::process::Hello {
        major: proto::ENGINE_INTERFACE_VERSION,
        build: proto::BUILD.to_owned(),
        peer: Some(proto::process::hello::Peer::Engine(
            proto::process::Engine {
                deployment: launch.engine_config.deployment.clone(),
                engine: launch.engine,
                interface: proto::process::Interface::Process as i32,
            },
        )),
    })?;
    let welcome: proto::process::Welcome = process
        .recv(HANDSHAKE_TIMEOUT)?
        .context("the daemon did not welcome this engine's process interface")?;
    if !welcome.refusal.is_empty() {
        bail!(
            "the daemon refused this engine's process interface: {}",
            welcome.refusal
        );
    }
    // The nodes are what the claim and every later request are answered by: a welcome with none
    // admits the process to a deployment the daemon does not know it in.
    if welcome.nodes.is_empty() {
        bail!("the daemon admitted this engine but listed no node of the deployment");
    }

    // The scopes are this process's ranks of its group: consecutive from its first, one per
    // device.
    let ranks = (0..launch.devices.len()).map(|index| launch.rank + index as u32);
    process.deliver(&proto::process::Upstream {
        kind: Some(upstream::Kind::Request(proto::process::Request {
            id: CLAIM,
            kind: Some(request::Kind::Claim(proto::process::Claim {
                host: launch.host.clone(),
                deployment: launch.engine_config.deployment.clone(),
                scopes: ranks
                    .map(|rank| proto::process::Scope {
                        group: launch.group.clone(),
                        rank,
                    })
                    .collect(),
                // A server the daemon starts for this claim takes the deployment's connector
                // settings; the engine states none of its own.
                server_arguments: Vec::new(),
                epoch: launch.epoch,
                retire: Vec::new(),
            })),
        })),
    })?;

    let mut filed = Vec::new();
    let claimed = loop {
        let message: proto::process::Downstream =
            process.recv(CLAIM_TIMEOUT)?.with_context(|| {
                format!("the daemon did not answer the claim within {CLAIM_TIMEOUT:?}")
            })?;
        match message.kind {
            Some(downstream::Kind::Reply(reply)) if reply.id == CLAIM => {
                if !reply.error.is_empty() {
                    bail!("the daemon refused this engine's claim: {}", reply.error);
                }
                let Some(proto::process::reply::Kind::Claimed(claimed)) = reply.kind else {
                    bail!("the daemon answered the claim with another reply");
                };
                break claimed;
            }
            Some(downstream::Kind::Reply(reply)) => {
                tracing::warn!(
                    id = reply.id,
                    "a reply to a request this engine did not send"
                );
            }
            // A command is the control thread's; it is served in order once the thread runs.
            Some(downstream::Kind::Event(proto::process::Event {
                kind: Some(proto::process::event::Kind::Command(command)),
            })) => filed.push(command),
            Some(downstream::Kind::Event(_)) | None => {}
        }
    };

    let token: [u8; 16] = claimed.claim.as_slice().try_into().map_err(|_| {
        anyhow::anyhow!(
            "a claim token is 16 bytes, and the daemon sent {}",
            claimed.claim.len()
        )
    })?;
    let storage = ipc::StorageLaunch {
        socket: claimed.socket,
        deployment: launch.engine_config.deployment.clone(),
        claim: token,
        tcp: (!claimed.tcp.is_empty()).then_some(claimed.tcp),
        // The reply names this node's server alone: another node's servers are not stated to
        // this process.
        peers: Vec::new(),
        epoch: launch.epoch,
        // A process launched beside its predecessor stands by until it is promoted.
        standby: launch.parked(),
        standby_storage: launch.standby_storage,
    };
    Ok((storage, filed))
}

/// Connects to another daemon's request socket at `address`, as a hot restart's successor asks
/// (`Attach`), says hello there as the engine `engine` and reads its welcome. The caller states
/// the engine's state on the connection before anything else.
///
/// # Errors
///
/// When the connection cannot be made, or the daemon refuses this engine or does not welcome it
/// in time.
pub(crate) fn attach(address: &str, engine: u64) -> Result<Connection> {
    let mut connection = Connection::connect(address)
        .with_context(|| format!("connecting to the attaching daemon at {address}"))?;
    hello(&mut connection, engine, ATTACH_TIMEOUT)?;
    Ok(connection)
}

/// The hello of the request interface and the daemon's welcome.
fn hello(connection: &mut Connection, engine: u64, timeout: Duration) -> Result<()> {
    connection.deliver(&wire::Hello {
        major: proto::ENGINE_INTERFACE_VERSION,
        build: proto::BUILD.to_owned(),
        engine,
        // The request link: the process link states `PROCESS` on its own socket, and both name
        // the same engine number.
        interface: proto::process::Interface::Requests as i32,
    })?;
    let welcome: wire::Welcome = connection
        .recv(timeout)?
        .context("the daemon closed the request interface before it welcomed this engine")?;
    if !welcome.refusal.is_empty() {
        bail!("the daemon refused this engine: {}", welcome.refusal);
    }
    // The daemon compares the major versions too and refuses on a difference; a welcome that
    // states another is a daemon of another interface.
    if welcome.major != proto::ENGINE_INTERFACE_VERSION {
        bail!(
            "the daemon's interface is version {} and this engine's {}",
            welcome.major,
            proto::ENGINE_INTERFACE_VERSION
        );
    }
    Ok(())
}

/// Starts the reader thread of front `front`, which hands every frame of `connection` to the
/// control thread as [`Input::Frame`] and ends with [`Input::FrontClosed`] once the daemon closed
/// it or a frame does not decode.
///
/// # Errors
///
/// When the thread cannot start.
pub(crate) fn read_requests(
    front: usize,
    mut connection: Connection,
    inbox: Sender<Input>,
) -> Result<JoinHandle<()>> {
    std::thread::Builder::new()
        .name(format!("front-{front}"))
        .spawn(move || {
            loop {
                match connection.try_recv::<wire::ToEngine>() {
                    Ok(Some(frame)) => {
                        if inbox.send(Input::Frame(front, frame)).is_err() {
                            return;
                        }
                    }
                    Ok(None) => match connection.wait(READ_WAIT) {
                        Ok(()) => {}
                        // The daemon closed the front.
                        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                        Err(error) => {
                            tracing::warn!(%error, front, "waiting on a daemon front failed");
                            break;
                        }
                    },
                    Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(error) => {
                        tracing::warn!(%error, front, "reading a daemon front failed");
                        break;
                    }
                }
            }
            let _ = inbox.send(Input::FrontClosed(front));
        })
        .context("starting a daemon front's reader")
}

/// Starts the reader thread of the process interface, which hands every command of `connection`
/// to the control thread as [`Input::Command`] and ends with `Command(None)` once the daemon
/// closed it.
///
/// # Errors
///
/// When the thread cannot start.
pub(crate) fn read_commands(
    mut connection: Connection,
    inbox: Sender<Input>,
) -> Result<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("commands".to_owned())
        .spawn(move || {
            loop {
                match connection.try_recv::<proto::process::Downstream>() {
                    Ok(Some(message)) => match message.kind {
                        Some(downstream::Kind::Event(proto::process::Event {
                            kind: Some(proto::process::event::Kind::Command(command)),
                        })) => {
                            if inbox.send(Input::Command(Some(command))).is_err() {
                                return;
                            }
                        }
                        // An exit names a group this process does not supervise, and this
                        // process sends no request after its claim.
                        Some(other) => {
                            tracing::debug!(message = ?other, "a frame no engine takes");
                        }
                        None => {}
                    },
                    Ok(None) => match connection.wait(READ_WAIT) {
                        Ok(()) => {}
                        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                        Err(error) => {
                            tracing::warn!(%error, "waiting on the process interface failed");
                            break;
                        }
                    },
                    Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(error) => {
                        tracing::warn!(%error, "reading the process interface failed");
                        break;
                    }
                }
            }
            let _ = inbox.send(Input::Command(None));
        })
        .context("starting the process interface's reader")
}

/// Sends one frame of the request interface on a front, with the artifact bytes it travels
/// with.
///
/// A front on this host maps the artifact's shared memory, whose descriptor rides the frame
/// itself. A front on another host, or bytes no descriptor names, takes bounded parts ahead of
/// the frame, each stating the artifact's whole length, which the daemon writes into the
/// artifact's storage.
///
/// # Errors
///
/// When the front closed or its socket failed.
pub(crate) fn send(
    connection: &mut Connection,
    kind: wire::from_engine::Kind,
    media: Option<&Arc<dyn ArtifactMedia>>,
) -> std::io::Result<()> {
    let frame = wire::FromEngine { kind: Some(kind) };
    let Some(media) = media else {
        return connection.deliver(&frame);
    };

    let descriptor = media
        .descriptor()
        .filter(|_| connection.peer_addr().is_none());
    if let Some(descriptor) = descriptor {
        connection.send_with(&frame, &[descriptor])?;
        return connection.flush();
    }
    let bytes = media.bytes();
    for data in bytes.chunks(ARTIFACT_PART) {
        connection.deliver(&wire::FromEngine {
            kind: Some(wire::from_engine::Kind::ArtifactPart(wire::ArtifactPart {
                size: bytes.len() as u64,
                data: data.to_vec(),
            })),
        })?;
    }
    connection.deliver(&frame)
}

/// Sends one report on the process interface, which the daemon answers with nothing.
///
/// # Errors
///
/// When the interface closed or its socket failed.
pub(crate) fn report(
    process: &mut Connection,
    report: proto::process::Report,
) -> std::io::Result<()> {
    process.deliver(&proto::process::Upstream {
        kind: Some(upstream::Kind::Report(report)),
    })
}

#[cfg(test)]
mod tests {
    /// The engine interface carries the Storage Server protocol's version, which the interfaces
    /// state without linking the storage crate: the two are one fact, and an engine whose
    /// interface version did not move with the storage protocol would be admitted by a daemon
    /// whose Storage Server it cannot speak to.
    #[test]
    fn the_engine_interface_carries_the_storage_protocol_version() {
        assert_eq!(proto::STORAGE_PROTOCOL, storage::protocol::VERSION);
    }
}
