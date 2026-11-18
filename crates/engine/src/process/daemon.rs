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
//!
//! No wait on a daemon has a bound of its own: a daemon answers or closes the connection, and the
//! daemon that started this process bounds its own waits on it and ends it, as the owner of its
//! lifetime, when one passes.

use std::num::NonZeroU32;
use std::os::fd::OwnedFd;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use crossbeam_channel::Sender;
use proto::config::EngineArgs;
use proto::process::{downstream, request, upstream};
use proto::request::ArtifactMedia;
use proto::request::wire;
use proto::{MappedMedia, MediaReceiver};
use service::{Connection, Message};

use super::Input;

/// The request id of the storage claim, the one request this process sends on the process
/// interface.
const CLAIM: u64 = 1;

/// Says hello on the request interface over `connection` as the engine `launch.engine`, reads
/// the daemon's welcome, and states `Starting` before anything loads.
///
/// # Errors
///
/// When the daemon closes or refuses this engine.
pub fn connect(connection: &mut Connection, launch: &EngineArgs) -> Result<NonZeroU32> {
    let window = hello(connection, launch.engine)?;

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
    .context("stating that this engine starts")?;
    Ok(window)
}

/// Says hello on the process interface over `process`, then asks this node's daemon for the
/// node's Storage Server claim, before any storage is mapped.
///
/// Returns the Storage Server launch this process's ranks open their storage with, and the
/// commands the daemon sent while the claim was outstanding, in order.
///
/// # Errors
///
/// When the launch states engine 0, the daemon closes the interface or refuses the process or its
/// claim, or lists no node.
pub(crate) fn claim(
    process: &mut Connection,
    launch: &EngineArgs,
) -> Result<(
    crate::memory::client::StorageLaunch,
    Vec<proto::process::Command>,
)> {
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
                deployment: launch.deployment.clone(),
                engine: launch.engine,
                interface: proto::process::Interface::Process as i32,
            },
        )),
    })?;
    let welcome: proto::process::Welcome =
        next(process).context("the daemon did not welcome this engine's process interface")?;
    if let Some(refusal) = welcome.refusal {
        return Err(anyhow::Error::new(refusal)
            .context("the daemon refused this engine's process interface"));
    }
    // The nodes are what the claim and every later request are answered by: a welcome with none
    // admits the process to a deployment the daemon does not know it in.
    if welcome.nodes.is_empty() {
        bail!("the daemon admitted this engine but listed no node of the deployment");
    }

    // The scopes are this process's ranks, each of the worker group that places it under its
    // number in the deployment, as each rank connects to the Storage Server.
    let groups = &launch.engine_config.parallel.worker_groups;
    let ranks = crate::process::ranks::rank::placed_ranks(launch)?
        .into_iter()
        .map(|(rank, group, ..)| (groups[group].id.0.clone(), rank));
    process.deliver(&proto::process::Upstream {
        kind: Some(upstream::Kind::Request(proto::process::Request {
            id: CLAIM,
            kind: Some(request::Kind::Claim(proto::process::Claim {
                host: launch.host.clone(),
                deployment: launch.deployment.clone(),
                scopes: ranks
                    .map(|(group, rank)| proto::process::Scope { group, rank })
                    .collect(),
                // A server the daemon starts for this claim takes the deployment's settings; the
                // engine states none of its own.
                storage: None,
                epoch: launch.epoch,
                retire: Vec::new(),
            })),
        })),
    })?;

    let mut filed = Vec::new();
    let claimed = loop {
        // Starting the node's Storage Server for the claim is the daemon's, with its own bound.
        let message: proto::process::Downstream =
            next(process).context("the daemon did not answer the claim")?;
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
    let storage = crate::memory::client::StorageLaunch {
        socket: claimed.socket,
        deployment: launch.deployment.clone(),
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
/// When the connection cannot be made, or the daemon closes it or refuses this engine.
pub(crate) fn attach(address: &str, engine: u64) -> Result<(Connection, NonZeroU32)> {
    let mut connection = address
        .parse()
        .map_err(std::io::Error::from)
        .and_then(|endpoint| Connection::connect(&endpoint))
        .with_context(|| format!("connecting to the attaching daemon at {address}"))?;
    let window = hello(&mut connection, engine)?;
    Ok((connection, window))
}

/// The hello of the request interface and the daemon's welcome.
fn hello(connection: &mut Connection, engine: u64) -> Result<NonZeroU32> {
    connection.deliver(&wire::Hello {
        major: proto::ENGINE_INTERFACE_VERSION,
        build: proto::BUILD.to_owned(),
        engine,
        // The request link: the process link states `PROCESS` on its own socket, and both name
        // the same engine number.
        interface: proto::process::Interface::Requests as i32,
    })?;
    let welcome: wire::Welcome = next(connection)
        .context("the daemon closed the request interface before it welcomed this engine")?;
    if let Some(refusal) = welcome.refusal {
        return Err(anyhow::Error::new(refusal).context("the daemon refused this engine"));
    }
    // The daemon states how far ahead of its reports it takes a request's events.
    NonZeroU32::new(welcome.event_window)
        .context("the daemon's welcome states no event window, which is positive")
}

/// The next message of `connection`, for as long as the daemon keeps it open.
///
/// # Errors
///
/// The connection's failure or the daemon's close, and a frame that does not decode.
fn next<M: Message>(connection: &mut Connection) -> std::io::Result<M> {
    next_with(connection).map(|(message, _)| message)
}

/// [`next`], with the descriptors the message's frame carried.
///
/// # Errors
///
/// As [`next`].
fn next_with<M: Message>(connection: &mut Connection) -> std::io::Result<(M, Vec<OwnedFd>)> {
    loop {
        if let Some(frame) = connection.recv_with(Duration::MAX)? {
            return Ok(frame);
        }
    }
}

/// Starts the reader thread `name` of `connection`, which blocks on the connection and hands each
/// message, with the descriptors its frame carried, to `deliver` until `deliver` says to stop
/// (the control thread is gone, or the message cannot be taken), the daemon closes the
/// connection, or a frame does not decode; `closed` then tells the control thread the connection
/// ended.
///
/// # Errors
///
/// When the thread cannot start.
fn read<M: Message>(
    name: String,
    mut connection: Connection,
    mut deliver: impl FnMut(M, Vec<OwnedFd>) -> bool + Send + 'static,
    closed: impl FnOnce() + Send + 'static,
) -> Result<JoinHandle<()>> {
    std::thread::Builder::new()
        .name(name.clone())
        .spawn(move || {
            loop {
                match next_with::<M>(&mut connection) {
                    Ok((message, descriptors)) => {
                        if !deliver(message, descriptors) {
                            break;
                        }
                    }
                    // The daemon closed the connection.
                    Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(error) => {
                        tracing::warn!(%error, reader = name, "reading a daemon connection failed");
                        break;
                    }
                }
            }
            closed();
        })
        .context("starting a daemon connection's reader")
}

/// Starts the reader thread of front `front`, which hands every frame of `connection` to the
/// control thread as [`Input::Frame`], with the media that came with it, and ends with
/// [`Input::FrontClosed`] once the daemon closed it or a frame or a medium does not decode.
///
/// A `Submit` from a front on this host carries its requests' condition media as descriptors; a
/// front on another host sends them as parts ahead of it (`condition_part`), which the reader
/// writes into memory files of this process's own. Either way the frame reaches the control
/// thread with its media in the order the requests' conditions state them.
///
/// # Errors
///
/// When the thread cannot start.
pub(crate) fn read_requests(
    front: usize,
    connection: Connection,
    inbox: Sender<Input>,
) -> Result<JoinHandle<()>> {
    let closing = inbox.clone();
    // The medium arriving in parts, and those that arrived whole, for the next frame.
    let mut receiving: Option<MediaReceiver> = None;
    let mut arrived: Vec<Arc<MappedMedia>> = Vec::new();
    read(
        format!("front-{front}"),
        connection,
        move |frame: wire::ToEngine, descriptors| {
            let received = match &frame.kind {
                Some(wire::to_engine::Kind::ConditionPart(part)) => {
                    receive_part(&mut receiving, &mut arrived, part).map(|()| None)
                }
                _ => descriptors
                    .into_iter()
                    .map(|descriptor| MappedMedia::whole(descriptor).map(Arc::new))
                    .collect::<std::io::Result<Vec<_>>>()
                    .map(|passed| {
                        let mut media = std::mem::take(&mut arrived);
                        media.extend(passed);
                        Some(media)
                    }),
            };
            match received {
                Ok(Some(media)) => inbox
                    .send(Input::Frame(front, Box::new(frame), media))
                    .is_ok(),
                Ok(None) => true,
                Err(error) => {
                    tracing::warn!(%error, front, "a medium from a daemon front does not map");
                    false
                }
            }
        },
        move || {
            let _ = closing.send(Input::FrontClosed(front));
        },
    )
}

/// Writes `part` of a condition's medium into the medium arriving (`receiving`), starting one when
/// none is, and moves the medium to `arrived` once its last part came.
///
/// # Errors
///
/// A part that changes or exceeds its medium's length, or a memory file the kernel refuses.
fn receive_part(
    receiving: &mut Option<MediaReceiver>,
    arrived: &mut Vec<Arc<MappedMedia>>,
    part: &wire::MediaPart,
) -> std::io::Result<()> {
    let medium = match receiving {
        Some(medium) => medium,
        None => receiving.insert(MediaReceiver::new(part.size)?),
    };
    medium.append(part.size, &part.data)?;
    if medium.complete()
        && let Some(medium) = receiving.take()
    {
        arrived.push(Arc::new(medium.finish(part.size)?));
    }
    Ok(())
}

/// Starts the reader thread of the process interface, which hands every command of `connection`
/// to the control thread as [`Input::Command`] and ends with `Command(None)` once the daemon
/// closed it.
///
/// # Errors
///
/// When the thread cannot start.
pub(crate) fn read_commands(
    connection: Connection,
    inbox: Sender<Input>,
) -> Result<JoinHandle<()>> {
    let closing = inbox.clone();
    read(
        "commands".to_owned(),
        connection,
        // The process interface passes no descriptor to the engine.
        move |message: proto::process::Downstream, _| match message.kind {
            Some(downstream::Kind::Event(proto::process::Event {
                kind: Some(proto::process::event::Kind::Command(command)),
            })) => inbox.send(Input::Command(Some(command))).is_ok(),
            // An exit names a group this process does not supervise, and this process sends no
            // request after its claim.
            Some(other) => {
                tracing::debug!(message = ?other, "a frame no engine takes");
                true
            }
            None => true,
        },
        move || {
            let _ = closing.send(Input::Command(None));
        },
    )
}

/// Sends one frame of the request interface on a front, with the artifact bytes it travels
/// with, and waits until the socket took it ([`queue`], then the connection's flush).
///
/// # Errors
///
/// When the front closed or its socket failed.
pub(crate) fn send(
    connection: &mut Connection,
    kind: wire::from_engine::Kind,
    media: Option<&Arc<dyn ArtifactMedia>>,
) -> std::io::Result<()> {
    queue(connection, kind, media)?;
    connection.flush()
}

/// Queues one frame of the request interface on a front, with the artifact bytes it travels
/// with, and writes what the socket takes now; the caller flushes the connection once it queued
/// the frames that go together.
///
/// A front on this host maps the artifact's shared memory, whose descriptor rides the frame
/// itself. A front on another host, or bytes no descriptor names, takes bounded parts ahead of
/// the frame, each stating the artifact's whole length, which the daemon writes into the
/// artifact's storage; each part is written whole before the next is encoded, so an artifact is
/// never held twice.
///
/// # Errors
///
/// When the front closed or its socket failed.
pub(crate) fn queue(
    connection: &mut Connection,
    kind: wire::from_engine::Kind,
    media: Option<&Arc<dyn ArtifactMedia>>,
) -> std::io::Result<()> {
    let frame = wire::FromEngine { kind: Some(kind) };
    let Some(media) = media else {
        return connection.send(&frame);
    };

    let descriptor = media
        .descriptor()
        .filter(|_| connection.peer_addr().is_none());
    if let Some(descriptor) = descriptor {
        return connection.send_with(&frame, &[descriptor]);
    }
    let bytes = media.bytes();
    for data in bytes.chunks(proto::media::PART_BYTES) {
        connection.deliver(&wire::FromEngine {
            kind: Some(wire::from_engine::Kind::ArtifactPart(wire::MediaPart {
                size: bytes.len() as u64,
                data: data.to_vec(),
            })),
        })?;
    }
    connection.send(&frame)
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
    use std::os::unix::net::UnixStream;

    use service::Listener;

    use super::*;

    /// A `Submit` whose request states `conditions` conditions.
    fn submit(conditions: usize) -> wire::ToEngine {
        let request = wire::Request {
            trace: 0,
            id: 7,
            priority: 0,
            kind: Some(wire::request::Kind::Diffusion(wire::DiffusionRequest {
                conditions: vec![wire::ConditionExtent::default(); conditions],
                ..wire::DiffusionRequest::default()
            })),
            scheduler: None,
        };
        wire::ToEngine {
            kind: Some(wire::to_engine::Kind::Submit(wire::Submit {
                requests: vec![request],
            })),
        }
    }

    /// The media of the first frame front 0's reader hands the control thread, a submission.
    fn media_of(inbox: &crossbeam_channel::Receiver<Input>) -> Vec<Arc<MappedMedia>> {
        match inbox.recv_timeout(Duration::from_secs(10)) {
            Ok(Input::Frame(0, frame, media)) => {
                assert!(matches!(frame.kind, Some(wire::to_engine::Kind::Submit(_))));
                media
            }
            _ => panic!("the reader hands the control thread the submission"),
        }
    }

    /// A front on this host passes each condition's sealed memory file beside the `Submit`, and
    /// the control thread receives the submission with a mapping of each, in order.
    #[test]
    fn a_local_front_s_condition_media_come_with_their_submission() {
        let (daemon, engine) = UnixStream::pair().unwrap();
        let mut daemon = Connection::new(daemon).unwrap();
        let (sender, inbox) = crossbeam_channel::unbounded();
        read_requests(0, Connection::new(engine).unwrap(), sender).unwrap();

        let first = MappedMedia::copy(b"first condition").unwrap();
        let second = MappedMedia::copy(b"second").unwrap();
        let descriptors = [first.descriptor().unwrap(), second.descriptor().unwrap()];
        daemon.send_with(&submit(2), &descriptors).unwrap();
        daemon.flush().unwrap();

        let media = media_of(&inbox);
        assert_eq!(media.len(), 2);
        assert_eq!(media[0].bytes(), b"first condition");
        assert_eq!(media[1].bytes(), b"second");
    }

    /// A front on another host sends each condition's medium as parts ahead of the `Submit`, and
    /// the control thread receives the submission with each medium whole, in order.
    #[test]
    fn a_remote_front_s_condition_media_arrive_in_parts_before_their_submission() {
        let listener = Listener::bind(&"127.0.0.1:0".parse().unwrap()).unwrap();
        let mut daemon = Connection::connect(listener.endpoint()).unwrap();
        let engine = listener.try_accept().unwrap().unwrap();
        let (sender, inbox) = crossbeam_channel::unbounded();
        read_requests(0, engine, sender).unwrap();

        for medium in [b"0123456789".as_slice(), b"abc".as_slice()] {
            for data in medium.chunks(4) {
                daemon
                    .deliver(&wire::ToEngine {
                        kind: Some(wire::to_engine::Kind::ConditionPart(wire::MediaPart {
                            size: medium.len() as u64,
                            data: data.to_vec(),
                        })),
                    })
                    .unwrap();
            }
        }
        daemon.deliver(&submit(2)).unwrap();

        let media = media_of(&inbox);
        assert_eq!(media.len(), 2);
        assert_eq!(media[0].bytes(), b"0123456789");
        assert_eq!(media[1].bytes(), b"abc");
    }
}
