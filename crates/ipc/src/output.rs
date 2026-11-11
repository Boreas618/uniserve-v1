//! What a call's completion states besides its tokens: its status and error
//! class, its timing, the media artifacts it materializes and the latent page
//! layouts it reports. The completion itself is
//! [`crate::execution::CallOutput`].

use super::*;

/// Terminal status of one call's completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum CallStatus {
    /// Execution completed successfully.
    Ok = 0,
    /// The call was skipped because its predicate was false.
    Predicated = 1,
    /// Execution failed with a deterministic error code.
    Error = 2,
}

/// The class of a rank's failure: the one vocabulary of a failed call's
/// completion and of an error response.
///
/// The rank classifies every failure it reports with these classes, and the
/// engine decides by them: an invariant violation is a control-plane defect
/// that stops the engine, an input error is a rejected request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// The rank does not serve the call's kind.
    UnsupportedCall,
    /// A message, a call or a launch value breaks the protocol.
    InvalidDescriptor,
    /// The deployment asks for a rank layout, a device or a kernel the rank
    /// cannot provide.
    UnsupportedSetup,
    /// A request's inputs are invalid for its computation.
    InputError,
    /// Device or numerical computation failed.
    ComputeError,
    /// The computation produced non-finite values (NaN or infinity) where
    /// its numerics admit none, as among a forward's logits: a fault of the
    /// computation, never of the request.
    NumericalFault,
    /// A reserved or physical resource could not satisfy the work.
    ResourceError,
    /// An invariant the rank relies on is broken.
    InvariantViolation,
    /// The rank failed in a way that leaves it unable to serve.
    FatalWorkerFailure,
}

impl ErrorCode {
    /// The class's name as the protocol spells it.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnsupportedCall => "unsupported_call",
            Self::InvalidDescriptor => "invalid_descriptor",
            Self::UnsupportedSetup => "unsupported_setup",
            Self::InputError => "input_error",
            Self::ComputeError => "compute_error",
            Self::NumericalFault => "numerical_fault",
            Self::ResourceError => "resource_error",
            Self::InvariantViolation => "invariant_violation",
            Self::FatalWorkerFailure => "fatal_worker_failure",
        }
    }

    /// Whether a failure of this class leaves the rank unsafe to serve
    /// further work.
    pub const fn fatal(self) -> bool {
        matches!(self, Self::InvariantViolation | Self::FatalWorkerFailure)
    }
}

impl std::fmt::Display for ErrorCode {
    /// Formats the class as its protocol name.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Per-call timing counters. Accounting only; never state identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct TimingCounters {
    /// Time spent waiting for worker execution, in microseconds.
    pub queued_us: u64,
    /// Time spent executing device work, in microseconds.
    pub device_us: u64,
    /// Time spent transferring completion data, in microseconds.
    pub copy_us: u64,
    /// Time spent processing the completion on the host, in microseconds.
    pub host_us: u64,
}

/// Transport handle for a completed media artifact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactHandle {
    /// Artifact stored in a POSIX shared-memory object.
    PosixShm {
        /// Shared-memory object name without a path separator.
        name: String,
    },
}

impl ArtifactHandle {
    /// Validates the transport-specific artifact identifier.
    pub fn validate(&self) -> ValidationResult<()> {
        match self {
            Self::PosixShm { name } => ensure_valid!(
                !name.is_empty() && !name.contains('/'),
                "POSIX shared-memory artifact name is invalid"
            ),
        }
        Ok(())
    }
}

/// Metadata describing one materialized media artifact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MediaOutput {
    /// Transport descriptor for the materialized artifact.
    pub handle: ArtifactHandle,
    /// Artifact length in bytes.
    pub bytes: u64,
}

/// How one denoising rank's request slot pages hold a video request's
/// latent samples.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankLatentLayout {
    /// The rank's index in its worker group.
    pub rank: u32,
    /// The slot's pages as one storage (`storage::manifest::StorageLayout`,
    /// JSON): the logical sample tensors, the part of each the rank holds
    /// and where its values lie in the pages.
    pub layout: String,
}
