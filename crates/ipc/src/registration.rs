//! The first frame on a worker's connection to its engine.
//!
//! A worker is started with the engine's address, its own identity and the
//! launch it belongs to. It connects, and the first frame it sends is its
//! [`Registration`]. The engine answers with the worker's
//! [`LaunchDescriptor`](crate::LaunchDescriptor), which carries every other
//! launch value, and the connection then carries requests and responses
//! until it closes.

use serde::{Deserialize, Serialize};

/// The version of this build, which a registration states: the package
/// version, then `+` and a digest of the workspace sources the build is
/// compiled from (computed by this crate's build script), for example
/// `2.0.0+1f3c9a0d5e7b2c48`.
///
/// The message types are the only definition of the protocol and their
/// encoding is positional, so the two ends of a connection must be the same
/// build: a message type reordered or retyped between two builds would fail
/// to decode or, worse, decode into other fields. The engine and a worker
/// library may be installed from different trees, so the engine refuses a
/// registration that states another version. The digest tells apart two
/// builds of one package version.
pub const VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "+",
    env!("UNISERVE_SOURCE_DIGEST")
);

/// The first frame a worker sends: who it is and what it was built as.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Registration {
    /// One worker of a worker group.
    Worker {
        /// Worker group the worker was started for.
        worker_id: String,
        /// The worker's rank within that group.
        rank: u32,
        /// The launch of the worker group the worker was started for. The
        /// engine numbers each group's launches from 1, and a relaunch after
        /// the group lost a worker takes the next number, so a registration
        /// of another launch than the group's current one is a worker of an
        /// earlier launch, which the engine refuses.
        generation: u64,
        /// [`VERSION`] of the build the worker runs.
        version: String,
    },
}
