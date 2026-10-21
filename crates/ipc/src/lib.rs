//! Scheduler-to-worker protocol and its rank transports.
//!
//! [`NewRequest`] admits static state, [`Batch`] submits planned calls, and
//! [`BatchOutput`] returns completions and products. [`WorkerInfo`] describes a
//! loaded worker before execution begins. Numeric request, call, point,
//! and generation identities remain stable across serialization.
//!
//! [`WorkerRequest`] and [`WorkerResponse`] are the envelopes exchanged per
//! rank, each as the one message of a [`Frame`]: a length followed by the
//! message in the binary encoding its `serde` derive describes. These message
//! types are the only definition of the protocol, so the engine and the worker
//! library must be built from the same sources. A [`Channel`] carries the
//! frames over a stream connection: a Unix socket to a rank on the engine's
//! node, TCP to a rank on another node. A rank connects to the engine, sends
//! its [`Registration`] and receives its [`LaunchDescriptor`]; the worker
//! runtime (`uniserve worker`) then serves its end until the connection
//! closes.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// Result type for semantic worker-message validation.
pub type ValidationResult<T> = std::result::Result<T, ValidationError>;

/// Describes a worker message that violates the protocol contract.
#[derive(Debug, thiserror::Error)]
#[error("invalid worker message: {0}")]
pub struct ValidationError(String);

impl ValidationError {
    /// Constructs a validation error from contextual text.
    pub(crate) fn message(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl From<SamplingParamsError> for ValidationError {
    /// Converts sampling validation failures into worker-protocol failures.
    fn from(error: SamplingParamsError) -> Self {
        Self::message(error.to_string())
    }
}

impl From<ImageParamsError> for ValidationError {
    /// Converts image validation failures into worker-protocol failures.
    fn from(error: ImageParamsError) -> Self {
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
/// The unit a channel carries: a length and one encoded message.
pub mod frame;
pub mod launch;
pub mod registration;

/// Counter-based random draws of the rank's token sampler.
pub mod philox;

pub use channel::{Channel, Listener, Wake};
pub use error::{EncodingError, IpcError, IpcResult, TransportError};
pub use frame::Frame;
pub use launch::{ExpertParallelDescriptor, LaunchDescriptor, StorageLaunch};
pub use registration::{Registration, VERSION};

// Message types are split by concern and re-exported flat at the crate root;
// each module takes the shared imports through `use super::*` and the
// validation macros through textual scope.
mod batch;
mod cache;
mod call;
mod command;
mod config;
mod identity;
mod info;
mod media;
mod output;
mod parallel;
mod request;
mod tensor;
mod transfer;

pub use batch::*;
pub use cache::*;
pub use call::*;
pub use command::*;
pub use config::{LaneConfig, LaneConfigError};
pub use identity::*;
pub use info::*;
pub use media::{SharedMedia, SharedMediaError};
pub use proto::{MediaLocator, MediaSource, SharedMemory};
// The request values the engine carries on to its ranks are the request
// vocabulary's own.
pub use output::*;
pub use parallel::{
    ComponentConfig, ComponentDistribution, ComponentWeights, HeadProjection, ParallelConfig,
    ParallelConfigError, SequenceParallel,
};
pub use proto::report::ForwardStats;
pub use proto::request::{
    CanvasSampling, CfgRenorm, CfgRenormParseError, DiffusionSamplingParams, ImageParams,
    ImageParamsError, PrefixCacheDtype, PrefixCacheDtypeParseError, RequestId, SamplingParams,
    SamplingParamsError, StandbyMemory, TokenLogprob,
};
pub use request::*;
pub use tensor::*;
pub use transfer::*;
