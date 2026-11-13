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
//!
//! Each connection opens with a hello exchange that agrees the version of the
//! engine interface it speaks ([`ENGINE_INTERFACE_VERSION`], [`speaks`]). The
//! workspace's one connection (`service::Connection`) carries the messages,
//! one length-prefixed frame each, over a Unix socket on one node or TCP
//! between nodes: every envelope of the interfaces implements
//! `service::Message` here.

pub mod catalog;
pub mod config;
mod media;
pub mod module;
pub mod profile;
pub mod report;
pub mod request;
mod shared_memory;
pub mod tensor;
pub mod video;

pub use media::{MappedMedia, MediaLocator, MediaReceiver, MediaSource};
pub use shared_memory::SharedMemory;

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

/// The revision of the messages this crate states (the process, request and
/// handoff interfaces). A change that only adds messages or fields leaves it;
/// any other change raises it.
const MESSAGES_REVISION: u32 = 3;

/// The storage protocol's version at version 1 of the engine interface.
const STORAGE_PROTOCOL_BASE: u32 = 21;

/// The version of the Storage Server's protocol the engine interface carries:
/// `storage::protocol::VERSION`, stated here because the interfaces depend on
/// no storage crate. The two are one fact, and a test of the engine, which
/// links both, fails if they drift; `UNISERVE_STORAGE_PROTOCOL_RAISE=1` at
/// build time raises both by one.
pub const STORAGE_PROTOCOL: u32 = 26
    + match option_env!("UNISERVE_STORAGE_PROTOCOL_RAISE") {
        None => 0,
        Some(raise) if raise.len() == 1 && raise.as_bytes()[0] == b'1' => 1,
        Some(_) => panic!("UNISERVE_STORAGE_PROTOCOL_RAISE is 1 when it is set"),
    };

/// The storage protocol's version at which the messages took their revision
/// ([`MESSAGES_REVISION`]): an engine built for an earlier storage protocol
/// exchanges other messages with the daemon. A change that raises the
/// revision sets this to the storage protocol's version it is made at.
const MESSAGES_STORAGE_PROTOCOL: u32 = 24;

/// The version of the engine interface: every message an engine exchanges with
/// the daemon (the process and request interfaces), the Storage Server and the
/// Build Server. The interface has this one version. The engine package's
/// manifest states it, an engine sends it in its hello, and a daemon serves an
/// engine of a version its build speaks ([`speaks`]) and refuses any other.
///
/// The version carries the storage protocol's ([`STORAGE_PROTOCOL`]): each
/// version of that protocol past the base raises it by one, as each change of
/// this crate's messages that is not an addition does. Neither term ever
/// falls, so an engine built for a storage protocol a build does not speak
/// states an interface version it does not speak either, and is refused at its
/// hello, before any of its ranks reaches the Storage Server.
pub const ENGINE_INTERFACE_VERSION: u32 =
    MESSAGES_REVISION + (STORAGE_PROTOCOL - STORAGE_PROTOCOL_BASE);

/// The earliest version of the engine interface this build speaks: that of the
/// storage protocol before its own at the same revision of the messages. A
/// build speaks its storage protocol and the one before
/// (`storage::protocol::OLDEST`), and an engine built for the one before
/// exchanges this build's messages with the daemon, so a daemon restarted in
/// place into the next build goes on serving the engine it adopts until the
/// next epoch replaces it. A build whose messages took their revision at its
/// own storage protocol speaks its own version alone.
pub const ENGINE_INTERFACE_OLDEST: u32 = MESSAGES_REVISION
    + (if STORAGE_PROTOCOL > MESSAGES_STORAGE_PROTOCOL {
        STORAGE_PROTOCOL - 1
    } else {
        STORAGE_PROTOCOL
    })
    - STORAGE_PROTOCOL_BASE;

/// Whether this build speaks version `version` of the engine interface: one
/// from [`ENGINE_INTERFACE_OLDEST`] to [`ENGINE_INTERFACE_VERSION`]. A daemon
/// welcomes a peer whose hello states such a version in the peer's own, which
/// the connection speaks for its life, and refuses a peer of any other.
pub const fn speaks(version: u32) -> bool {
    ENGINE_INTERFACE_OLDEST <= version && version <= ENGINE_INTERFACE_VERSION
}

/// The longest frame a peer of the interfaces may send: their messages are
/// small, and a longer length is a corrupt stream.
const MAX_FRAME: usize = 64 << 20;

/// Implements the workspace's connection codec for each envelope of the
/// interfaces: its Protobuf encoding in one length-prefixed frame.
macro_rules! envelopes {
    ($($message:ty),* $(,)?) => {
        $(
            impl service::Message for $message {
                const FRAMING: service::Framing = service::Framing::Length;
                const MAX_BYTES: usize = MAX_FRAME;

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
    process::Hello,
    process::Welcome,
    process::Upstream,
    process::Downstream,
    process::ToMember,
    process::FromMember,
    request::wire::Hello,
    request::wire::Welcome,
    request::wire::ToEngine,
    request::wire::FromEngine,
    request::wire::Artifact,
    request::wire::ArtifactPart,
);

/// The environment variable that names a supervised process's lifeline: the
/// descriptor number of the read end of a pipe whose write end its daemon
/// holds. The process ends once the pipe does.
pub const LIFELINE_ENV: &str = "UNISERVE_LIFELINE";

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
