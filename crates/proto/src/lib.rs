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
//! Each connection opens with a hello exchange in which the daemon refuses a
//! peer of any other version of the engine interface than its own
//! ([`ENGINE_INTERFACE_VERSION`]) with a typed refusal. The workspace's one
//! connection (`service::Connection`) carries the messages,
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

/// The version of the engine interface: the process, request and handoff
/// messages an engine exchanges with its daemon (this crate's schemas). The
/// interface has one version per release, raised by every change of its
/// messages. The engine package's manifest states it, a peer sends it in its
/// hello, and a daemon serves a peer of this version alone, refusing any other
/// in its welcome with a typed refusal (`process::refusal::Reason::Version`,
/// `request::wire::refusal::Reason::Version`).
///
/// The engine's other interfaces have versions of their own, each checked at
/// that interface's hello: the Storage Server's protocol
/// (`storage::protocol::VERSION`) at the Storage Server, the buffer pool's
/// runtime interface (`storage::runtime::VERSION`) when a session opens.
pub const ENGINE_INTERFACE_VERSION: u32 = 13;

impl process::Refusal {
    /// The refusal of a peer whose hello states version `offered` of the engine
    /// interface, when that is not this build's; `None` for this build's.
    pub fn of_version(offered: u32) -> Option<Self> {
        (offered != ENGINE_INTERFACE_VERSION).then_some(Self {
            reason: Some(process::refusal::Reason::Version(process::VersionRefusal {
                offered,
                spoken: ENGINE_INTERFACE_VERSION,
            })),
        })
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

impl std::fmt::Display for process::Refusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use process::refusal::Reason;

        match &self.reason {
            Some(Reason::Version(version)) => write!(
                formatter,
                "the peer speaks version {} of the engine interface and the daemon {}",
                version.offered, version.spoken
            ),
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
    /// The refusal of an engine whose hello states version `offered` of the
    /// engine interface, when that is not this build's; `None` for this
    /// build's.
    pub fn of_version(offered: u32) -> Option<Self> {
        (offered != ENGINE_INTERFACE_VERSION).then_some(Self {
            reason: Some(request::wire::refusal::Reason::Version(
                request::wire::VersionRefusal {
                    offered,
                    spoken: ENGINE_INTERFACE_VERSION,
                },
            )),
        })
    }
}

impl std::fmt::Display for request::wire::Refusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use request::wire::refusal::Reason;

        match &self.reason {
            Some(Reason::Version(version)) => write!(
                formatter,
                "the engine speaks version {} of the engine interface and the daemon {}",
                version.offered, version.spoken
            ),
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
