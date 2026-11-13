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
//! Each connection opens with a hello exchange that compares
//! [`ENGINE_INTERFACE_VERSION`]. The workspace's one connection
//! (`service::Connection`) carries the messages, one length-prefixed frame
//! each, over a Unix socket on one node or TCP between nodes: every envelope
//! of the interfaces implements `service::Message` here.

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
/// links both, fails if they drift.
pub const STORAGE_PROTOCOL: u32 = 25;

/// The version of the engine interface: every message an engine exchanges with
/// the daemon (the process and request interfaces), the Storage Server and the
/// Build Server. The interface has this one version. The engine package's
/// manifest states it, an engine sends it in its hello, and a daemon refuses
/// an engine built for another version.
///
/// The version carries the storage protocol's ([`STORAGE_PROTOCOL`]): each
/// version of that protocol past the base raises it by one, as each change of
/// this crate's messages that is not an addition does. Neither term ever
/// falls, so an engine built for another storage protocol states another
/// interface version and is refused at its hello, before any of its ranks
/// reaches the Storage Server.
pub const ENGINE_INTERFACE_VERSION: u32 =
    MESSAGES_REVISION + (STORAGE_PROTOCOL - STORAGE_PROTOCOL_BASE);

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
