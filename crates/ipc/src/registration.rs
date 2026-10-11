//! The first frame on a connection to the head.
//!
//! A rank is started with the head's address and its own identity, and a
//! host's launcher with the head's address and its host identity. Each
//! connects, and the first frame it sends is its [`Registration`]. The head
//! answers a rank with its [`LaunchDescriptor`](crate::LaunchDescriptor),
//! which carries every other launch value, and the connection then carries
//! requests and responses until it closes. A launcher's connection carries
//! the messages of [`launcher`](crate::launcher) from then on.

use serde::{Deserialize, Serialize};

use crate::frame::Message;
use crate::{ValidationError, ValidationResult};

/// The version of this build, which a registration states.
///
/// The message types are the only definition of the protocol, so the two
/// ends of a connection must be the same build. The head refuses a
/// registration that states another version. The version is the package's,
/// so two builds of one version are not told apart; a difference between
/// them then surfaces as a decode error.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The first frame a rank or a launcher sends: who it is and what it was
/// built as.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Registration {
    /// One rank of a worker group.
    Rank {
        /// Worker group the rank was started for.
        worker_id: String,
        /// Rank within that group's process world.
        rank: u32,
        /// The launch of the group that started this rank. A group
        /// relaunched after a failure carries the next generation, so a rank
        /// of the launch it replaces cannot register in its place.
        generation: u64,
        /// [`VERSION`] of the build the rank runs.
        version: String,
    },
    /// The launcher of one host, which starts the ranks placed there.
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
            Self::Rank { version, .. } | Self::Launcher { version, .. } => version,
        }
    }
}

impl Message for Registration {
    /// A registration names its worker group or its host.
    fn validate(&self) -> ValidationResult<()> {
        match self {
            Self::Rank { worker_id, .. } => {
                ensure_valid!(!worker_id.is_empty(), "a registration names no worker");
            }
            Self::Launcher { host, .. } => {
                ensure_valid!(!host.is_empty(), "a registration names no host");
            }
        }
        Ok(())
    }
}
