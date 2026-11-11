//! Scheduler-to-main-process protocol and its rank transports.
//!
//! [`NewRequest`] admits static state, [`execution::Batch`] is the batch the
//! scheduler submits and the worker serves, and [`execution::BatchOutput`]
//! returns its completions. [`MainProcessInfo`]
//! describes a peer main process's ranks before execution begins. Numeric
//! request, call, point, and generation identities remain stable across
//! serialization.
//!
//! [`MainProcessRequest`] and [`MainProcessResponse`] are the envelopes exchanged per
//! rank, each as the one message of a [`Frame`]: a length followed by the
//! message in the binary encoding its `serde` derive describes. These message
//! types are the only definition of the protocol, so the main processes of an
//! epoch must be built from the same sources. A [`Channel`] carries the
//! frames over a stream connection: a Unix socket to a rank on the scheduler's
//! node, TCP to a rank on another node. A rank's main process connects to the
//! scheduler's and sends its [`PeerRegistration`].

use std::collections::HashSet;

use proto::tensor::DType;
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

impl From<SamplingParamsError> for ValidationError {
    /// Converts sampling validation failures into worker-protocol failures.
    fn from(error: SamplingParamsError) -> Self {
        Self::message(error.to_string())
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

pub mod channel;
/// Failures at the channel boundary.
pub mod error;
/// The batch the scheduler submits and the worker serves, and what it produces (§2.4, §4.4).
pub mod execution;
/// The unit a channel carries: a length and one encoded message.
pub mod frame;
pub mod launch;
pub mod memory;

/// Counter-based random draws of the rank's token sampler.
pub mod philox;

pub use channel::{Channel, Listener, Wake, Waker};
pub use error::{EncodingError, IpcError, IpcResult, TransportError};
pub use frame::Frame;
pub use launch::StorageLaunch;

// Message types are split by concern and re-exported flat at the crate root;
// each module takes the shared imports through `use super::*` and the
// validation macros through textual scope.
mod batch;
mod cache;
mod call;
mod command;
mod identity;
mod info;
mod media;
mod output;
/// The first frame between main processes of one epoch (§2.3).
pub mod peer;
/// The scheduler's name for one phase (§4.5).
pub mod phase;
pub mod product;
pub mod program;
mod request;
mod tensor;
/// The vocabulary of a program-layer update (§7.3).
pub mod update;

pub use batch::*;
pub use cache::*;
pub use call::*;
pub use command::*;
pub use identity::*;
pub use info::*;
pub use media::{SharedMedia, SharedMediaError};
pub use proto::{MediaLocator, MediaSource, SharedMemory};
// The request values the engine carries on to its ranks are the request
// vocabulary's own.
pub use output::*;
pub use peer::{PeerAccepted, PeerRegistration, VERSION};
pub use phase::PhaseId;
pub use program::ProgramCommand;
pub use proto::report::{FENCE_BUCKETS_US, FenceHistogram, ForwardStats};
pub use proto::request::{
    BlockSampling, DiffusionSamplingParams, PrefixCacheDtype, PrefixCacheDtypeParseError,
    RequestId, SamplingParams, SamplingParamsError, StandbyStorage, TokenLogprob,
};
pub use request::*;
pub use tensor::*;
pub use update::{ProgramStatus, UpdateFailure};
