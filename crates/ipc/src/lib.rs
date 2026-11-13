//! The batch a scheduler hands its executor, and the peer link between the engine processes of
//! an epoch.
//!
//! [`batch::Batch`] is the batch the scheduler hands to its executor and
//! [`batch::BatchOutput`] what the batch produced: plain data that names a
//! request's memory by positions in its tables, never by an address.
//! [`Capacity`] is what a worker group's executors serve, stated once they
//! are ready.
//!
//! [`MainProcessRequest`] and [`MainProcessResponse`] are the envelopes exchanged per
//! rank, each the message of one frame on the workspace's one connection
//! (`service::Connection`; the codec is [`link`]): a length followed by the
//! message in the binary encoding its `serde` derive describes. These message
//! types are the only definition of the protocol, so the main processes of an
//! epoch must be built from the same sources. The connection is a Unix socket
//! to a rank on the scheduler's node, TCP to a rank on another node. A rank's
//! main process connects to the scheduler's [`Listener`] and sends its
//! [`PeerRegistration`].

use serde::{Deserialize, Serialize};

/// Result type for semantic worker-message validation.
pub type ValidationResult<T> = std::result::Result<T, ValidationError>;

/// Describes a worker message that violates the protocol.
#[derive(Debug, thiserror::Error)]
#[error("invalid worker message: {0}")]
pub struct ValidationError(String);

impl ValidationError {
    /// Constructs a validation error from contextual text.
    pub(crate) fn message(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// What the message violates, without the protocol's prefix.
    pub fn reason(&self) -> &str {
        &self.0
    }
}

// The validation macros below are textually scoped: they are visible only in
// modules declared after them, which is how the message modules use them.
// Keep them above those `mod` declarations.
macro_rules! invalid_message {
    ($($arg:tt)*) => {
        ValidationError::message(format!($($arg)*))
    };
}

macro_rules! bail_invalid {
    ($($arg:tt)*) => {
        return Err(invalid_message!($($arg)*))
    };
}

macro_rules! ensure_valid {
    ($condition:expr, $($arg:tt)*) => {
        if !$condition {
            bail_invalid!($($arg)*);
        }
    };
}

/// The batch the scheduler hands to its executor, and what it produces.
pub mod batch;
/// What a worker group's executors serve, as they state it once they are ready.
pub mod capacity;
pub mod launch;
pub mod link;
pub mod memory;

/// Counter-based random draws of the rank's token sampler.
pub mod philox;

pub use capacity::Capacity;
pub use launch::StorageLaunch;
pub use link::Listener;
/// The peer link's connection and its local wake, which the worker's link threads name through
/// this crate.
pub use service::{Connection, Wake};

// Message types are split by concern and re-exported flat at the crate root;
// each module takes the shared imports through `use super::*` and the
// validation macros through textual scope.
mod cache;
mod info;
mod media;
/// The first frame between main processes of one epoch (§2.3).
pub mod peer;
pub mod product;
mod request;

// The parts of a call's completion the batch's output states.
pub use batch::{ArtifactHandle, DraftWork, ErrorCode, MediaOutput, TimingCounters};
pub use cache::*;
pub use info::*;
pub use media::{SharedMedia, SharedMediaError};
pub use peer::{PeerAccepted, PeerRegistration, VERSION};
pub use proto::report::{FenceHistogram, ForwardStats};
pub use proto::request::{PrefixCacheDtype, RequestId, StandbyStorage, TokenLogprob};
pub use proto::{MediaSource, SharedMemory};
pub use request::*;
