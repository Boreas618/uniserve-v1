//! Worker failures: internal errors and their classification for the engine.
//!
//! Owners report [`Error`]. The executor classifies a failure once, into a
//! [`WorkerError`] with a stable [`ErrorCode`], fatality, phase and the calls
//! it affects, and the service sends its [`WorkerResponseError`] form. The
//! codes are a wire contract: the engine matches on them (for example it
//! escalates `InvariantViolation` like a fatal error).

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
    /// kind (`uniserve.errors`) and `traceback` the Python
    /// traceback carried by TVM-FFI.
    #[error("{function}: {kind}: {message}")]
    Numerical {
        function: String,
        kind: String,
        message: String,
        traceback: String,
    },
    /// A library function or a value it returned does not satisfy its
    /// contract.
    #[error("{0}")]
    Interface(String),
    /// A failure every party of one operation observes: the result of a
    /// read, the state of a pool or an export that several owners consult.
    /// It reports and classifies as the failure it shares.
    #[error(transparent)]
    Shared(std::sync::Arc<Error>),
    /// Python returned a TVM-FFI tensor where the worker expects an opaque
    /// object. Handles cross threads; a TVM-FFI tensor, a DLPack import of
    /// memory its producer owns, must not. A torch tensor returned through
    /// the import path crosses as an opaque handle instead.
    #[error("a TVM-FFI tensor cannot be held as an opaque handle")]
    TensorHandle,
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
    InvalidDescriptor,
    UnsupportedSetup,
    InputError,
    ComputeError,
    ResourceError,
    InvariantViolation,
    FatalWorkerFailure,
}

impl ErrorCode {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::UnsupportedCall => "UnsupportedCall",
            Self::InvalidDescriptor => "InvalidDescriptor",
            Self::UnsupportedSetup => "UnsupportedSetup",
            Self::InputError => "InputError",
            Self::ComputeError => "ComputeError",
            Self::ResourceError => "ResourceError",
            Self::InvariantViolation => "InvariantViolation",
            Self::FatalWorkerFailure => "FatalWorkerFailure",
        }
    }

    /// A failure of this kind leaves the rank unsafe to serve further work.
    pub(crate) const fn fatal(self) -> bool {
        matches!(self, Self::InvariantViolation | Self::FatalWorkerFailure)
    }

    /// Failures worth a full traceback in the rank's log.
    pub(crate) const fn traces(self) -> bool {
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

/// Failure texts that mark a context-corrupting CUDA failure, which requires
/// teardown even if allocation also failed.
///
/// A driver failure the worker observed itself carries its result code and
/// is classified by it. These texts classify failures that carry only a
/// message: an exception the numerical library raised, or a worker error
/// whose message embeds a lower-level failure.
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

/// Failure texts that mark an allocation failure, used like
/// [`FATAL_CUDA_MESSAGES`].
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
    pub(crate) fn new(code: ErrorCode, message: impl Into<String>) -> Self {
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
    /// A CUDA driver failure is classified by its result code. Any other
    /// failure is classified, in order, by a fatal CUDA text in its message,
    /// an out-of-memory text in its message, then its own kind.
    pub fn classify(error: Error, context: Option<&str>) -> Self {
        let mut message = error_message(&error);
        if let Some(context) = context.filter(|context| !context.is_empty()) {
            message = format!("{context}: {message}");
        }
        let code = match driver_failure(&error) {
            Some(failure) => driver_code(failure),
            None => {
                let lowered = message.to_lowercase();
                if FATAL_CUDA_MESSAGES
                    .iter()
                    .any(|text| lowered.contains(text))
                {
                    ErrorCode::FatalWorkerFailure
                } else if OUT_OF_MEMORY_MESSAGES
                    .iter()
                    .any(|text| lowered.contains(text))
                {
                    ErrorCode::ResourceError
                } else {
                    kind_code(&error)
                }
            }
        };
        let mut classified = Self::new(code, message);
        classified.detail.traceback = traceback(&error);
        classified
    }

    pub(crate) fn calls(&self) -> &[ErrorCallIdentity] {
        &self.detail.calls
    }

    pub(crate) fn traceback(&self) -> Option<&str> {
        self.detail.traceback.as_deref()
    }

    /// Name the phase that failed, unless one is already named.
    pub(crate) fn with_phase(mut self, phase: impl Into<String>) -> Self {
        self.detail.phase.get_or_insert_with(|| phase.into());
        self
    }

    /// Name the batch's call kind, unless one is already named.
    pub(crate) fn with_route(mut self, route: impl Into<String>) -> Self {
        self.detail.route.get_or_insert_with(|| route.into());
        self
    }

    /// Name the affected calls, unless some are already named.
    pub(crate) fn with_calls(mut self, calls: Vec<ErrorCallIdentity>) -> Self {
        if self.detail.calls.is_empty() {
            self.detail.calls = calls;
        }
        self
    }

    /// Record a failure observed while cleaning up after this one.
    pub(crate) fn note(&mut self, cleanup: &WorkerError) {
        self.detail
            .notes
            .push(format!("cleanup failed: {}", cleanup.detail.message));
    }

    /// The wire form sent to the engine.
    pub(crate) fn response(&self) -> WorkerResponseError {
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
        Error::Shared(shared) => error_message(shared),
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
        Error::Shared(shared) => traceback(shared),
        Error::Numerical { traceback, .. } if !traceback.is_empty() => Some(traceback.clone()),
        _ => None,
    }
}

/// The CUDA failure an error is, directly or as a shared failure.
fn driver_failure(error: &Error) -> Option<&uniserve_cuda::Error> {
    match error {
        Error::Cuda(failure) => Some(failure),
        Error::Shared(shared) => driver_failure(shared),
        _ => None,
    }
}

/// The code of a CUDA failure, by the result its call returned.
fn driver_code(failure: &uniserve_cuda::Error) -> ErrorCode {
    if failure.is_unrecoverable() {
        ErrorCode::FatalWorkerFailure
    } else if failure.is_out_of_memory() {
        ErrorCode::ResourceError
    } else {
        ErrorCode::ComputeError
    }
}

/// The code an error's own kind maps to.
fn kind_code(error: &Error) -> ErrorCode {
    match error {
        Error::Shared(shared) => kind_code(shared),
        Error::Invalid(_) => ErrorCode::InvalidDescriptor,
        Error::Resource(_) | Error::ReadBackpressure { .. } => ErrorCode::ResourceError,
        Error::Unsupported(_) | Error::Interface(_) | Error::TensorHandle | Error::HostName(_) => {
            ErrorCode::UnsupportedSetup
        }
        Error::Invariant(_) => ErrorCode::InvariantViolation,
        Error::Transport(_) | Error::Cuda(_) | Error::State(_) => ErrorCode::ComputeError,
        Error::Numerical { kind, .. } => numerical_code(kind),
    }
}

/// Map the kind of a failed library call to a code.
///
/// The numerical library translates every exception its functions raise to
/// one of its error kinds (`uniserve.errors`). The
/// remaining kinds are the ones TVM-FFI itself reports when a value that
/// crossed into Rust cannot be converted: a wrong type, a missing key,
/// an index out of range or a value out of range.
pub(crate) fn numerical_code(kind: &str) -> ErrorCode {
    match kind {
        "UnsupportedSetup" => ErrorCode::UnsupportedSetup,
        "OutOfMemory" => ErrorCode::ResourceError,
        "UnsupportedCall" => ErrorCode::UnsupportedCall,
        "InputError" | "KeyError" | "IndexError" | "TypeError" | "ValueError" => {
            ErrorCode::InputError
        }
        "InvariantViolation" => ErrorCode::InvariantViolation,
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
            error.response().message,
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
            ("UnsupportedCall", ErrorCode::UnsupportedCall),
            ("InputError", ErrorCode::InputError),
            ("InvariantViolation", ErrorCode::InvariantViolation),
            ("UnsupportedSetup", ErrorCode::UnsupportedSetup),
            ("ComputeError", ErrorCode::ComputeError),
            // A value TVM-FFI could not convert.
            ("TypeError", ErrorCode::InputError),
        ];
        for (kind, code) in cases {
            let error = WorkerError::classify(numerical(kind, "failed"), None);
            assert_eq!(error.code, code, "{kind}");
            assert_eq!(error.fatal, code == ErrorCode::InvariantViolation);
        }
    }

    #[test]
    fn driver_failures_are_classified_by_their_result_code() {
        let driver = |code, message| {
            Error::Cuda(uniserve_cuda::Error::Driver {
                call: "cuMemcpyBatchAsync_v2",
                code,
                name: "CUDA_ERROR",
                message,
            })
        };

        // The result code decides, whatever the driver's text says.
        let exhausted = WorkerError::classify(driver(2, "failed"), Some("batch export"));
        assert_eq!(exhausted.code, ErrorCode::ResourceError);
        assert!(!exhausted.fatal);

        let faulted = WorkerError::classify(driver(700, "failed"), None);
        assert_eq!(faulted.code, ErrorCode::FatalWorkerFailure);
        assert!(faulted.fatal);

        // CUDA_ERROR_INVALID_VALUE is a rejected call, even with this text.
        let rejected = WorkerError::classify(driver(1, "out of memory"), None);
        assert_eq!(rejected.code, ErrorCode::ComputeError);
    }

    #[test]
    fn a_shared_failure_keeps_the_class_of_the_failure_it_shares() {
        let cancelled = std::sync::Arc::new(Error::Resource("transfer read was cancelled"));
        let observed = WorkerError::classify(Error::Shared(cancelled), None);
        assert_eq!(observed.code, ErrorCode::ResourceError);
        assert_eq!(observed.response().message, "transfer read was cancelled");

        let failed = std::sync::Arc::new(numerical("OutOfMemory", "tried to allocate"));
        let observed = WorkerError::classify(Error::Shared(failed), Some("decode"));
        assert_eq!(observed.code, ErrorCode::ResourceError);
        assert_eq!(observed.traceback(), Some("Traceback"));
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
