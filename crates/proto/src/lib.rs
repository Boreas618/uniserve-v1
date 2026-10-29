//! UniServe's interfaces across builds.
//!
//! A deployment's daemons outlive its engines: a world change replaces the
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
//!
//! Each connection opens with a hello exchange that compares
//! [`PROTOCOL_MAJOR`]; a peer never sends a field the other's
//! [`PROTOCOL_MINOR`] does not know. [`Connection`] carries the messages,
//! one frame each, over a Unix socket on one node or TCP between nodes.

mod connection;
mod media;
pub mod profile;
pub mod report;
pub mod request;
mod shared_memory;

pub use connection::{Connection, Error, Handed, Listener, Result};
pub use media::{MappedMedia, MediaLocator, MediaReceiver, MediaSource};
pub use shared_memory::SharedMemory;

/// The messages of the process interface (`process.proto`).
#[allow(clippy::all, reason = "code generated from the schema")]
pub mod process {
    include!(concat!(env!("OUT_DIR"), "/uniserve.process.rs"));
}

/// The interfaces' major version: peers of different majors refuse each
/// other. It changes only when a field is removed or changes meaning, which
/// takes a planned restart of every daemon.
pub const PROTOCOL_MAJOR: u32 = 1;

/// The interfaces' minor version: the additions this build knows. Minor 1
/// adds the leader's answer of the nodes it knows now (`NodesQuery`). Minor 2
/// adds explicit publication of committed worlds' immutable pages. Minor 3
/// carries generated artifacts over TCP in bounded parts.
pub const PROTOCOL_MINOR: u32 = 3;

/// The build this crate is compiled in: the package version, then `+` and a
/// digest of the workspace sources every artifact of the build is compiled
/// from, as the engine interface's registration states it.
pub const BUILD: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "+",
    env!("UNISERVE_SOURCE_DIGEST")
);
