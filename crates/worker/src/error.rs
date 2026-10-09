//! Worker failures: internal errors and their classification for the engine.
//!
//! Owners report [`Error`]. The executor classifies a failure once, into a
//! [`WorkerError`] with a stable [`ErrorCode`], fatality, phase and the calls
//! it affects, and the service sends its [`WorkerResponseError`] form. The
//! codes are a wire contract: the engine matches on them (for example it
//! escalates `SchedulerBug` and `InvariantViolation` like fatal errors).

use uniserve_ipc::{ErrorCallIdentity, WorkerResponseError};

/// Invalid submissions and operations on unavailable worker state.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Transport(String),
    /// A CUDA driver or NCCL call failed.
    #[error(transparent)]
    Cuda(#[from] uniserve_cuda::Error),
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    State(&'static str),
    #[error("{0}")]
    Resource(&'static str),
    #[error("{0}")]
    Unsupported(String),
    #[error("asynchronous transfer ticket capacity is exhausted")]
    ReadBackpressure { returns: u64 },
    #[error("{0}")]
    Invariant(String),
    /// A Python numerical function raised. `kind` is the translated error
    /// kind (`uniserve_models.interface.errors`) and `traceback` the Python
    /// traceback carried by TVM-FFI.
    #[error("{function}: {kind}: {message}")]
    Numerical {
        function: String,
        kind: String,
        message: String,
        traceback: String,
    },
    /// The interface table or a binding does not satisfy its contract.
    #[error("{0}")]
    Interface(String),
    /// The host's node name cannot identify this rank's node to transfer
    /// peers: it is empty or not UTF-8.
    #[error("host name {0:?} cannot identify the rank's node in transfers")]
    HostName(std::ffi::CString),
}

impl Error {
    /// Wrap an error raised while calling `function` through TVM-FFI.
    pub fn numerical(function: &str, error: &tvm_ffi::Error) -> Self {
        Self::Numerical {
            function: function.to_owned(),
            kind: error.kind().as_str().to_owned(),
            message: error.message().to_owned(),
            traceback: error.backtrace().to_owned(),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Stable worker error identifiers carried as the response's `code`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    UnsupportedCall,
    UnsupportedControl,
    InvalidDescriptor,
    UnsupportedSetup,
    ResourceLeaseViolation,
    InputError,
    ComputeError,
    ResourceError,
    InvariantViolation,
    FatalWorkerFailure,
    SchedulerBug,
}

impl ErrorCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnsupportedCall => "UnsupportedCall",
            Self::UnsupportedControl => "UnsupportedControl",
            Self::InvalidDescriptor => "InvalidDescriptor",
            Self::UnsupportedSetup => "UnsupportedSetup",
            Self::ResourceLeaseViolation => "ResourceLeaseViolation",
            Self::InputError => "InputError",
            Self::ComputeError => "ComputeError",
            Self::ResourceError => "ResourceError",
            Self::InvariantViolation => "InvariantViolation",
            Self::FatalWorkerFailure => "FatalWorkerFailure",
            Self::SchedulerBug => "SchedulerBug",
        }
    }

    /// A failure of this kind leaves the rank unsafe to serve further work.
    pub const fn fatal(self) -> bool {
        matches!(self, Self::InvariantViolation | Self::FatalWorkerFailure)
    }

    /// Failures worth a full traceback in the rank's log.
    pub const fn traces(self) -> bool {
        matches!(
            self,
            Self::ComputeError
                | Self::ResourceError
                | Self::InvariantViolation
                | Self::FatalWorkerFailure
        )
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Context-corrupting CUDA failures require teardown, even if allocation
/// also failed.
const FATAL_CUDA_MESSAGES: &[&str] = &[
    "illegal memory access",
    "cudaerrorillegaladdress",
    "device-side assert",
    "cudaerrorlaunchfailure",
    "unspecified launch failure",
    "unrecoverable",
    "uncorrectable ecc",
    "misaligned address",
];

const OUT_OF_MEMORY_MESSAGES: &[&str] =
    &["out of memory", "cuda oom", "cublas_status_alloc_failed"];

/// A classified failure, as reported to the engine.
///
/// The code and fatality are inline; the rest is boxed, so a `Result` that
/// carries a failure stays small on the success path.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerError {
    pub code: ErrorCode,
    /// The rank must stop serving after reporting it.
    pub fatal: bool,
    detail: Box<ErrorDetail>,
}

#[derive(Debug, Clone, PartialEq)]
struct ErrorDetail {
    message: String,
    /// Worker phase that failed, such as `batch export`.
    phase: Option<String>,
    /// Call kind of the failed batch; `None` for failures outside a batch
    /// with calls (an undecodable request, admission, a lifecycle batch).
    route: Option<String>,
    calls: Vec<ErrorCallIdentity>,
    /// The Python traceback of a numerical failure.
    traceback: Option<String>,
    /// Secondary failures observed while cleaning up after this one.
    notes: Vec<String>,
}

impl WorkerError {
    /// A failure with an explicit code and its default fatality.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            fatal: code.fatal(),
            detail: Box::new(ErrorDetail {
                message: message.into(),
                phase: None,
                route: None,
                calls: Vec::new(),
                traceback: None,
                notes: Vec::new(),
            }),
        }
    }

    /// Classify an internal error. `context` prefixes the message, naming
    /// the operation that failed.
    ///
    /// Order: fatal CUDA messages, then out-of-memory messages or kinds,
    /// then the error's own kind.
    pub fn classify(error: Error, context: Option<&str>) -> Self {
        let mut message = error_message(&error);
        if let Some(context) = context.filter(|context| !context.is_empty()) {
            message = format!("{context}: {message}");
        }
        let lowered = message.to_lowercase();
        let code = if FATAL_CUDA_MESSAGES
            .iter()
            .any(|text| lowered.contains(text))
        {
            ErrorCode::FatalWorkerFailure
        } else if OUT_OF_MEMORY_MESSAGES
            .iter()
            .any(|text| lowered.contains(text))
            || matches!(&error, Error::Numerical { kind, .. } if kind.to_lowercase().contains("outofmemory"))
        {
            ErrorCode::ResourceError
        } else {
            kind_code(&error)
        };
        let mut classified = Self::new(code, message);
        classified.detail.traceback = traceback(&error);
        classified
    }

    pub fn message(&self) -> &str {
        &self.detail.message
    }

    pub fn phase(&self) -> Option<&str> {
        self.detail.phase.as_deref()
    }

    pub fn route(&self) -> Option<&str> {
        self.detail.route.as_deref()
    }

    pub fn calls(&self) -> &[ErrorCallIdentity] {
        &self.detail.calls
    }

    pub fn traceback(&self) -> Option<&str> {
        self.detail.traceback.as_deref()
    }

    /// Name the phase that failed, unless one is already named.
    pub fn with_phase(mut self, phase: impl Into<String>) -> Self {
        self.detail.phase.get_or_insert_with(|| phase.into());
        self
    }

    /// Name the batch's call kind, unless one is already named.
    pub fn with_route(mut self, route: impl Into<String>) -> Self {
        self.detail.route.get_or_insert_with(|| route.into());
        self
    }

    /// Name the affected calls, unless some are already named.
    pub fn with_calls(mut self, calls: Vec<ErrorCallIdentity>) -> Self {
        if self.detail.calls.is_empty() {
            self.detail.calls = calls;
        }
        self
    }

    /// Record a failure observed while cleaning up after this one.
    pub fn note(&mut self, cleanup: &WorkerError) {
        self.detail
            .notes
            .push(format!("cleanup failed: {}", cleanup.detail.message));
    }

    /// The wire form sent to the engine.
    pub fn response(&self) -> WorkerResponseError {
        WorkerResponseError {
            message: self.detail.message.clone(),
            code: Some(self.code.as_str().to_owned()),
            fatal: self.fatal,
            phase: self.detail.phase.clone(),
            route: self.detail.route.clone(),
            calls: self.detail.calls.clone(),
        }
    }
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail.message)?;
        for note in &self.detail.notes {
            write!(formatter, "; {note}")?;
        }
        Ok(())
    }
}

impl std::error::Error for WorkerError {}

impl From<Error> for WorkerError {
    fn from(error: Error) -> Self {
        Self::classify(error, None)
    }
}

/// The message without the function prefix of a numerical error; the kind
/// is kept so a reader sees which exception the library raised.
fn error_message(error: &Error) -> String {
    match error {
        Error::Numerical {
            function,
            kind,
            message,
            ..
        } => {
            let message = if message.is_empty() {
                kind.as_str()
            } else {
                message.as_str()
            };
            format!("{function}: {message}")
        }
        other => other.to_string(),
    }
}

fn traceback(error: &Error) -> Option<String> {
    match error {
        Error::Numerical { traceback, .. } if !traceback.is_empty() => Some(traceback.clone()),
        _ => None,
    }
}

/// The code an error's own kind maps to.
fn kind_code(error: &Error) -> ErrorCode {
    match error {
        Error::Invalid(_) => ErrorCode::InvalidDescriptor,
        Error::Resource(_) | Error::ReadBackpressure { .. } => ErrorCode::ResourceError,
        Error::Unsupported(_) | Error::Interface(_) | Error::HostName(_) => {
            ErrorCode::UnsupportedSetup
        }
        Error::Invariant(_) => ErrorCode::InvariantViolation,
        Error::Transport(_) | Error::Cuda(_) | Error::State(_) => ErrorCode::ComputeError,
        Error::Numerical { kind, .. } => numerical_code(kind),
    }
}

/// Map a translated Python error kind (or a raw exception class name, for a
/// function the interface did not wrap) to a code.
pub fn numerical_code(kind: &str) -> ErrorCode {
    match kind {
        "UnsupportedSetup" => ErrorCode::UnsupportedSetup,
        "OutOfMemory" => ErrorCode::ResourceError,
        "UnsupportedCall" | "NotImplementedError" => ErrorCode::UnsupportedCall,
        "InputError" | "KeyError" | "IndexError" | "TypeError" | "ValueError" => {
            ErrorCode::InputError
        }
        "InvariantViolation" | "AssertionError" => ErrorCode::InvariantViolation,
        _ => ErrorCode::ComputeError,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numerical(kind: &str, message: &str) -> Error {
        Error::Numerical {
            function: "text.forward".into(),
            kind: kind.into(),
            message: message.into(),
            traceback: "Traceback".into(),
        }
    }

    #[test]
    fn fatal_cuda_messages_outrank_allocation_failures() {
        let error = WorkerError::classify(
            numerical(
                "OutOfMemory",
                "CUDA out of memory after an illegal memory access",
            ),
            Some("decode"),
        );
        assert_eq!(error.code, ErrorCode::FatalWorkerFailure);
        assert!(error.fatal);
        assert_eq!(
            error.message(),
            "decode: text.forward: CUDA out of memory after an illegal memory access"
        );
        assert_eq!(error.traceback(), Some("Traceback"));
    }

    #[test]
    fn allocation_failures_are_resource_errors() {
        for error in [
            numerical("ComputeError", "CUBLAS_STATUS_ALLOC_FAILED"),
            numerical("OutOfMemory", "tried to allocate"),
            Error::Resource("worker admission queue is full"),
        ] {
            let error = WorkerError::classify(error, None);
            assert_eq!(error.code, ErrorCode::ResourceError);
            assert!(!error.fatal);
        }
    }

    #[test]
    fn library_kinds_keep_their_classification() {
        let cases = [
            ("NotImplementedError", ErrorCode::UnsupportedCall),
            ("UnsupportedCall", ErrorCode::UnsupportedCall),
            ("ValueError", ErrorCode::InputError),
            ("InputError", ErrorCode::InputError),
            ("AssertionError", ErrorCode::InvariantViolation),
            ("UnsupportedSetup", ErrorCode::UnsupportedSetup),
            ("RuntimeError", ErrorCode::ComputeError),
        ];
        for (kind, code) in cases {
            let error = WorkerError::classify(numerical(kind, "failed"), None);
            assert_eq!(error.code, code, "{kind}");
            assert_eq!(error.fatal, code == ErrorCode::InvariantViolation);
        }
    }

    #[test]
    fn native_errors_map_to_their_codes() {
        assert_eq!(
            WorkerError::classify(Error::Invalid("bad".into()), None).code,
            ErrorCode::InvalidDescriptor
        );
        let invariant = WorkerError::classify(Error::Invariant("broken".into()), None);
        assert_eq!(invariant.code, ErrorCode::InvariantViolation);
        assert!(invariant.fatal);
        let response = invariant
            .with_phase("batch export")
            .with_route("prefill")
            .response();
        assert_eq!(response.code.as_deref(), Some("InvariantViolation"));
        assert_eq!(response.phase.as_deref(), Some("batch export"));
        assert_eq!(response.route.as_deref(), Some("prefill"));
    }
}
