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
//! [`ENGINE_INTERFACE_VERSION`]. [`Connection`] carries the messages,
//! one frame each, over a Unix socket on one node or TCP between nodes.

pub mod catalog;
pub mod config;
mod connection;
mod media;
pub mod module;
pub mod profile;
pub mod report;
pub mod request;
mod shared_memory;
pub mod tensor;
pub mod video;

pub use connection::{Connection, Error, Handed, Listener, Result};
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

/// The version of the engine interface: every message an engine exchanges with
/// the daemon (the process and request interfaces), the Storage Server and the
/// Build Server. The interface has this one version. The engine package's
/// manifest states it, an engine sends it in its hello, and a daemon refuses
/// an engine built for another version. A change that only adds messages or
/// fields leaves the version; any other change raises it.
pub const ENGINE_INTERFACE_VERSION: u32 = 1;

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
