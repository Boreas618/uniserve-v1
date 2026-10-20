//! The first frame on a connection to the head.
//!
//! A worker is started with the head's address, its own identity and the
//! launch it belongs to, and a host's launcher with the head's address and
//! its host identity. Each connects, and the first frame it sends is its
//! [`Registration`]. The head answers a worker with its
//! [`LaunchDescriptor`](crate::LaunchDescriptor), which carries every other
//! launch value, and the connection then carries requests and responses
//! until it closes. A launcher's connection carries the messages of
//! [`launcher`](crate::launcher) from then on.

use serde::{Deserialize, Serialize};

/// The version of this build, which a registration states: the package
/// version, then `+` and a digest of the workspace sources the build is
/// compiled from (computed by this crate's build script), for example
/// `2.0.0+1f3c9a0d5e7b2c48`.
///
/// The message types are the only definition of the protocol and their
/// encoding is positional, so the two ends of a connection must be the same
/// build: a message type reordered or retyped between two builds would fail
/// to decode or, worse, decode into other fields. The head, a launcher and a
/// worker are installed as separate artifacts, so the head refuses a
/// registration that states another version. The digest tells apart two
/// builds of one package version, such as a head and a worker library
/// compiled from different trees.
pub const VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "+",
    env!("UNISERVE_SOURCE_DIGEST")
);

/// The first frame a worker or a launcher sends: who it is and what it was
/// built as.
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
        /// head numbers each group's launches from 1, and a relaunch after
        /// the group lost a worker takes the next number, so a registration
        /// of another launch than the group's current one is a worker of an
        /// earlier launch, which the head refuses.
        generation: u64,
        /// [`VERSION`] of the build the worker runs.
        version: String,
    },
    /// The launcher of one host, which starts the workers placed there.
    Launcher {
        /// The host identity the launcher owns, as the placement names it.
        host: String,
        /// [`VERSION`] of the build the launcher runs.
        version: String,
    },
}

impl Registration {
    /// Returns the build version the registration states.
    pub fn version(&self) -> &str {
        match self {
            Self::Worker { version, .. } | Self::Launcher { version, .. } => version,
        }
    }
}
