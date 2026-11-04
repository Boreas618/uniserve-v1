//! What loading, launching and shadowing an operator can fail with.
//!
//! An operator never falls back and never ignores a mismatch: a kernel whose module does not match
//! the descriptor, an argument whose shape contradicts it, a twin region that is too small and a
//! native entry that reports an error are all failures of the launch, and the caller (the runner)
//! turns that into a failed precopy or a failed batch (specs/worker.md 7.3).

use crate::platform::PlatformError;

/// A failed operator launch or generation lookup.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A call into the CUDA boundary failed.
    #[error("platform: {0}")]
    Platform(#[from] PlatformError),

    /// The generation holds no kernel for this operator and descriptor.
    #[error(
        "generation {generation} has no kernel for operator `{operator}` descriptor `{descriptor}`"
    )]
    MissingKernel {
        operator: &'static str,
        descriptor: String,
        generation: u64,
    },

    /// The kernel is of the other form than the operator launches it as.
    #[error("operator `{operator}` launches `{kernel}` as a {expected}, but it is a {actual}")]
    KernelForm {
        operator: &'static str,
        kernel: String,
        expected: &'static str,
        actual: &'static str,
    },

    /// A native entry returned a nonzero status.
    #[error("native entry `{entry}` failed with status {status}{}", match message { Some(message) => format!(": {message}"), None => String::new() })]
    Native {
        entry: String,
        status: i32,
        message: Option<String>,
    },

    /// The module's metadata cannot be read, or lacks a field a launch needs.
    #[error("kernel metadata: {0}")]
    Metadata(String),

    /// A launch grid does not fit the driver's limits.
    #[error("grid: {0}")]
    Grid(String),

    /// An argument's shape, dtype, stride or rank does not match the operator.
    #[error("operator `{operator}`: {problem}")]
    Argument {
        operator: &'static str,
        problem: String,
    },

    /// A shape has more axes than the ABI holds, or a dtype has no whole-byte size.
    #[error("shape: {0}")]
    Shape(String),

    /// The twin region cannot hold the shadow buffers this operator needs.
    #[error("twin region has {available} bytes left, {needed} needed for operator `{operator}`")]
    Twins {
        operator: &'static str,
        needed: u64,
        available: u64,
    },

    /// This operator cannot express this call, e.g. a collective with a non-uniform count.
    #[error("operator `{operator}`: {problem}")]
    Unsupported {
        operator: &'static str,
        problem: String,
    },
}

impl Error {
    /// An argument mismatch of `operator`.
    pub(crate) fn argument(operator: &'static str, problem: impl Into<String>) -> Self {
        Self::Argument {
            operator,
            problem: problem.into(),
        }
    }
}

/// The result of an operator call.
pub type Result<T, E = Error> = std::result::Result<T, E>;
