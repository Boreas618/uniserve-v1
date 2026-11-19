//! UniServe's interfaces across builds.
//!
//! A deployment's daemons outlive its engines: an epoch change replaces the
//! engine with one of another build while the daemons serve on. The
//! interfaces between them are therefore Protobuf schemas that change only by
//! additions (`process.proto`, `request.proto`), compiled into this crate:
//!
//! - [`process`]: how daemons and engine processes start, stop and supervise
//!   the processes of a deployment.
//! - [`request`]: what the leader daemon submits to an engine process and the
//!   events it receives back, as the Rust values both ends work with.
//! - [`report`]: the schedulers' statistics an engine reports to its daemon.
//! - [`profile`]: the profiling sessions the leader daemon starts on the
//!   workers, JSON in worker hooks.
//! - [`module`]: the package revisions an epoch fills its operators with, the phases
//!   and slots they fill, and the operators a build has.
//! - [`admin`]: the fields of the observation rows a node's admin API answers, JSON, which
//!   the daemon writes and its clients (`uniserve-manager`, a harness) read.
//!
//! Each connection opens with a hello exchange in which the peer states the
//! window of versions of the engine interface it speaks, the daemon agrees on
//! the newest version both speak ([`ENGINE_INTERFACE`], [`agree`]) and states
//! it in its welcome, and a peer whose window shares no version with the
//! daemon's is refused with a typed refusal that names both windows. The
//! workspace's one connection (`service::Connection`) carries the messages,
//! one frame each, over a Unix socket on one node or TCP
//! between nodes: every envelope of the interfaces implements
//! `service::Message` here.

// The reload derive names its trait by the crate's path (`::proto::reload`), which this crate's own
// configuration structures reach through this alias.
extern crate self as proto;

pub mod admin;
pub mod catalog;
pub mod config;
pub mod histogram;
pub mod media;
pub mod module;
pub mod profile;
pub mod reload;
pub mod report;
pub mod request;
pub mod shared_memory;
mod storage_settings;
pub mod tensor;
pub mod video;

pub use media::{MappedMedia, MediaReceiver};

/// The messages of the process interface (`process.proto`).
#[allow(clippy::all, reason = "code generated from the schema")]
pub mod process {
    include!(concat!(env!("OUT_DIR"), "/uniserve.process.rs"));
}

/// The state a hot restart hands its successor for the responses in flight
/// (`handoff.proto`).
#[allow(clippy::all, reason = "code generated from the schema")]
pub mod handoff {
    include!(concat!(env!("OUT_DIR"), "/uniserve.handoff.rs"));
}

/// The versions of the engine interface this build speaks: the process, request and handoff
/// messages an engine exchanges with its daemon, and a daemon with its members and with the
/// successor of its hot restart (this crate's schemas).
///
/// A peer states the window in its hello and the daemon serves it in the newest version both
/// speak ([`agree`]), refusing a peer whose window shares none with a typed refusal that names
/// both windows (`process::refusal::Reason::Version`, `request::wire::refusal::Reason::Version`).
/// An engine process speaks the version its first hello agreed on every link for its whole life
/// ([`Agreed`]), so a hot restart's successor that speaks it serves on with the same processes.
/// The engine package's executable states the window (`build_server::interface_section!`), and a
/// daemon starts only an engine whose window shares a version with its own.
///
/// The window's versions are one schema: every version changes the messages only by additions,
/// as Protobuf's rules for compatible changes state them (a new field number, oneof member or
/// enum value; never a renumbered, retyped or reused one). The newest version is raised by each
/// change of the messages a receiver must act on; a sender uses what a version added only on a
/// connection whose agreed version has it. The window holds at least the previous release's
/// version and the current one: raising the newest keeps the oldest, and a later change raises
/// the oldest past a version once no release that serves needs it, removing every handling of
/// the versions it passes (a version check `agreed >= V` whose `V` is the new oldest, a handler
/// of a message the window no longer holds).
///
/// The engine's other interfaces have windows of their own, each agreed at that interface's
/// hello: the Storage Server's protocol (`storage::protocol::VERSIONS`) at the Storage Server,
/// the buffer pool's runtime interface (`storage::runtime::VERSIONS`) when a session opens.
pub const ENGINE_INTERFACE: service::Versions = service::Versions::one(14);

/// The version of the engine interface a peer of the window `spoken` agrees on with a peer
/// whose hello states `oldest` and `newest` ([`process::Hello`], [`request::wire::Hello`]): the
/// newest both speak.
///
/// # Errors
///
/// [`Disagreement`] when the windows share no version (a hello that states no version at all
/// shares none).
pub fn agree(oldest: u32, newest: u32, spoken: service::Versions) -> Result<u32, Disagreement> {
    service::Versions::stated(oldest, newest)
        .and_then(|offered| offered.agree(spoken))
        .ok_or(Disagreement {
            oldest,
            newest,
            spoken,
        })
}

/// A hello whose window shares no version of the engine interface with the window a daemon
/// speaks: the hello's versions as it stated them, and the daemon's window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Disagreement {
    oldest: u32,
    newest: u32,
    spoken: service::Versions,
}

/// The version of the engine interface a peer that offered `offered` in its hello speaks with
/// the daemon whose welcome states `version` ([`process::Welcome::version`],
/// [`request::wire::Welcome::version`]): that version, or the offer's oldest when the welcome
/// states none, as a daemon of one version accepts it.
///
/// # Errors
///
/// When the welcome states a version the offer does not hold: the daemon broke the agreement.
pub fn agreed(offered: service::Versions, version: u32) -> Result<u32, String> {
    match version {
        0 => Ok(offered.oldest()),
        version if offered.speaks(version) => Ok(version),
        version => Err(format!(
            "the daemon welcomed versions {offered} of the engine interface in version {version}"
        )),
    }
}

/// The one version of the engine interface an engine process speaks once its first hello agreed
/// on it: every later hello of the process (its process link, an attachment to a hot restart's
/// successor) offers it alone, so the process speaks one version on all its links, whichever
/// daemon serves them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Agreed(pub u32);

impl Agreed {
    /// The window a later hello of the process offers: its agreed version alone.
    pub fn offer(self) -> service::Versions {
        service::Versions::one(self.0)
    }
}

/// The windows a version refusal states, as text: the peer's (`offered` to `offered_newest`)
/// and the refuser's (`spoken_oldest` to `spoken`), each a single version where its second field
/// is unset (a refuser or a peer of one version).
fn refused_windows(
    offered: u32,
    offered_newest: u32,
    spoken: u32,
    spoken_oldest: u32,
) -> (String, String) {
    let window = |oldest: u32, newest: u32| match service::Versions::stated(oldest, newest) {
        Some(window) => window.to_string(),
        None => format!("{oldest} to {newest}"),
    };
    let spoken_oldest = if spoken_oldest == 0 {
        spoken
    } else {
        spoken_oldest
    };
    (
        window(offered, offered_newest),
        window(spoken_oldest, spoken),
    )
}

impl process::Refusal {
    /// The refusal of a peer whose hello's window shares no version with the daemon's.
    pub fn of_versions(disagreement: Disagreement) -> Self {
        Self {
            reason: Some(process::refusal::Reason::Version(process::VersionRefusal {
                offered: disagreement.oldest,
                spoken: disagreement.spoken.newest(),
                offered_newest: disagreement.newest,
                spoken_oldest: disagreement.spoken.oldest(),
            })),
        }
    }

    /// A refusal for `reason`.
    pub fn because(reason: process::refusal::Reason) -> Self {
        Self {
            reason: Some(reason),
        }
    }
}

/// The longest hello or welcome a peer may send: one states versions, a build and the identity
/// of the peer, a few hundred bytes, and a longer one is a corrupt stream.
const HANDSHAKE_BYTES: usize = 64 << 10;

/// The longest other message a peer of the interfaces may send. The largest are a submission,
/// which carries a request as the daemon's HTTP server admitted it (its body limit is 64 MiB),
/// and a launch, which carries a deployment's configuration; media cross by descriptor or in
/// parts. A longer length is a corrupt stream.
const MESSAGE_BYTES: usize = 64 << 20;

/// The bytes of the spans one reply of a read of spans carries (`service::span::page`): an
/// engine's `Spans` and a node's `NodeSpans`, half of what a message takes, which leaves the rest
/// to the envelope. A read takes as many pages as its spans fill.
pub const SPAN_PAGE_BYTES: usize = MESSAGE_BYTES / 2;

impl std::fmt::Display for process::Refusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use process::refusal::Reason;

        match &self.reason {
            Some(Reason::Version(version)) => {
                let (offered, spoken) = refused_windows(
                    version.offered,
                    version.offered_newest,
                    version.spoken,
                    version.spoken_oldest,
                );
                write!(
                    formatter,
                    "the peer speaks versions {offered} of the engine interface and the daemon \
                     {spoken}, which share none"
                )
            }
            Some(Reason::Unserved(_)) => {
                formatter.write_str("the peer is not one this socket serves")
            }
            Some(Reason::RankTaken(rank)) => write!(formatter, "node rank {rank} is taken"),
            Some(Reason::Unnumbered(_)) => formatter.write_str(
                "a process link states no engine number: no command could be addressed to it",
            ),
            Some(Reason::Linked(engine)) => write!(
                formatter,
                "engine {engine} already has the link this connection states"
            ),
            None => formatter.write_str("the daemon refused the peer and stated no reason"),
        }
    }
}

impl std::error::Error for process::Refusal {}

impl request::wire::Refusal {
    /// The refusal of an engine whose hello's window shares no version with the daemon's.
    pub fn of_versions(disagreement: Disagreement) -> Self {
        Self {
            reason: Some(request::wire::refusal::Reason::Version(
                request::wire::VersionRefusal {
                    offered: disagreement.oldest,
                    spoken: disagreement.spoken.newest(),
                    offered_newest: disagreement.newest,
                    spoken_oldest: disagreement.spoken.oldest(),
                },
            )),
        }
    }
}

impl std::fmt::Display for request::wire::Refusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use request::wire::refusal::Reason;

        match &self.reason {
            Some(Reason::Version(version)) => {
                let (offered, spoken) = refused_windows(
                    version.offered,
                    version.offered_newest,
                    version.spoken,
                    version.spoken_oldest,
                );
                write!(
                    formatter,
                    "the engine speaks versions {offered} of the engine interface and the daemon \
                     {spoken}, which share none"
                )
            }
            Some(Reason::Engine(engine)) => write!(
                formatter,
                "engine {} connected on the link of engine {}",
                engine.stated, engine.expected
            ),
            None => formatter.write_str("the daemon refused the engine and stated no reason"),
        }
    }
}

impl std::error::Error for request::wire::Refusal {}

/// Implements the workspace's connection codec for each envelope of the
/// interfaces: its Protobuf encoding as one frame's payload, of at most
/// `$bytes` bytes.
macro_rules! envelopes {
    ($bytes:expr => $($message:ty),* $(,)?) => {
        $(
            impl service::Message for $message {
                const MAX_BYTES: usize = $bytes;

                fn encode(&self) -> std::io::Result<Vec<u8>> {
                    Ok(prost::Message::encode_to_vec(self))
                }

                fn decode(payload: &[u8]) -> std::io::Result<Self> {
                    <Self as prost::Message>::decode(payload).map_err(|error| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("undecodable message: {error}"),
                        )
                    })
                }
            }
        )*
    };
}

envelopes!(
    HANDSHAKE_BYTES =>
    process::Hello,
    process::Welcome,
    request::wire::Hello,
    request::wire::Welcome,
);

envelopes!(
    MESSAGE_BYTES =>
    process::Upstream,
    process::Downstream,
    process::ToMember,
    process::FromMember,
    request::wire::ToEngine,
    request::wire::FromEngine,
    request::wire::Artifact,
    request::wire::MediaPart,
);

/// The command-line option that names a supervised process's lifeline: the
/// descriptor number, inherited from the daemon that supervises the process, of
/// the read end of a pipe whose write end that daemon holds. The daemon appends
/// it to every process's command line; an engine ends once the pipe does.
pub const LIFELINE_OPTION: &str = "--lifeline";

/// The command-line option that names an engine process's launch: the descriptor number,
/// inherited from the daemon that starts the process, of a sealed anonymous file that holds the
/// process's [`config::EngineArgs`] (JSON). The process reads it whole as it starts.
pub const LAUNCH_OPTION: &str = "--launch";

/// The build this crate is compiled in: the package version, then `+` and a
/// digest of the workspace sources every artifact of the build is compiled
/// from, as the engine interface's registration states it.
pub const BUILD: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "+",
    env!("UNISERVE_SOURCE_DIGEST")
);

/// The workspace this crate is compiled in, as Cargo locates it: the root the
/// build's digest covers. Tests that read files another package of the
/// workspace owns, such as the deployment files under `configs/`, join their
/// paths to it.
pub const WORKSPACE: &str = env!("UNISERVE_WORKSPACE");
