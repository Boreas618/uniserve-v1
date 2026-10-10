//! A rank's registration with the head.
//!
//! A rank is started with the head's address and its own identity. It
//! connects, and the first frame it sends is its [`Registration`]. The head
//! answers with the rank's [`LaunchDescriptor`](crate::LaunchDescriptor),
//! which carries every other launch value, and the connection then carries
//! requests and responses until it closes.

use serde::{Deserialize, Serialize};

use crate::frame::Message;
use crate::{ValidationError, ValidationResult};

/// The version of this build, which a registration states.
///
/// The message types are the only definition of the protocol, so the two
/// ends of a connection must be the same build. The head refuses a
/// registration that states another version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The first frame a rank sends: who it is and what it was built as.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Registration {
    /// Worker group the rank was started for.
    pub worker_id: String,
    /// Rank within that group's process world.
    pub rank: u32,
    /// The launch of the group that started this rank. A group relaunched
    /// after a failure carries the next generation, so a rank of the launch
    /// it replaces cannot register in its place.
    pub generation: u64,
    /// [`VERSION`] of the build the rank runs.
    pub version: String,
}

impl Message for Registration {
    /// A registration names its worker group.
    fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(!self.worker_id.is_empty(), "a registration names no worker");
        Ok(())
    }
}
