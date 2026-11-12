//! The crate's error and its classification.
//!
//! Every module of this crate fails with [`Error`]. A failure that leaves a device context
//! unusable, such as an illegal address, is classified once into a [`Failure`] with a stable
//! [`ErrorCode`] and a fatality, which is what the engine needs to decide between a local retry,
//! stopping the device and the automatic rollback of an update on trial. The codes are the
//! protocol's vocabulary (`ipc::ErrorCode`), so the control thread reports them unchanged.

/// What an operation of the worker failed with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A socket, thread, shared-memory object or peer exchange failed, or a
    /// peer sent what its protocol does not allow. `source` is the error that
    /// caused it, when one did.
    #[error("{what}")]
    Transport {
        /// What failed.
        what: String,
        /// The failure that caused it.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    /// A CUDA driver or NCCL call failed.
    #[error(transparent)]
    Cuda(#[from] platform::Error),
    /// An operator's launch or a kernel's load failed.
    #[error(transparent)]
    Ops(#[from] ops::Error),
    /// A model's computation failed: a module, a tensor operation or a launch of the device
    /// backend, which the tensor library and the backend carry here as typed causes.
    #[error(transparent)]
    Nn(#[from] nn::Error),
    /// The media encoders or the muxer refused: a raster or a track the codecs cannot take, or an
    /// assembly that is not complete. The class is the media crate's own, kept whole so a caller
    /// can tell a refused request from a codec failure.
    #[error(transparent)]
    Media(#[from] media::Error),
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
    /// A library, module or value does not satisfy the interface it is used
    /// under: a missing descriptor, a kernel whose metadata contradicts its
    /// launch, a module path that names no file.
    #[error("{0}")]
    Interface(String),
    /// A failure as it happened, with the work in flight when it was taken.
    ///
    /// A device fault arrives at the control thread with a batch number and nothing else, and the
    /// device thread is the only owner of what it was doing: the bucket, its rows and the operator
    /// whose launch was issued. That work travels here, inside the error, so every report of the
    /// failure - the device's own line, the submission's outcome, the main process's reason - names
    /// it without any of them having to reach back for state that is gone by then.
    ///
    /// The wrapper is transparent: an illegal address is still a driver failure and its kind is
    /// still the inner error's, so classification does not change by being namable.
    #[error("{error} (in flight: {activity})")]
    InFlight {
        /// The work in flight, as the device thread states it.
        activity: String,
        /// The failure taken while it ran.
        error: Box<Error>,
    },
    /// A failure every party of one operation observes: the result of a read,
    /// the state of a pool or an export that several owners consult. It
    /// reports and classifies as the failure it shares.
    #[error(transparent)]
    Shared(std::sync::Arc<ipc::execution::Fault>),
}

impl Error {
    /// A transport failure without an underlying error, such as a malformed
    /// message from a peer.
    pub fn transport(what: impl Into<String>) -> Self {
        Self::Transport {
            what: what.into(),
            source: None,
        }
    }

    /// `error`, stated with the work that was in flight when it was taken.
    pub fn in_flight(activity: impl Into<String>, error: Error) -> Self {
        Self::InFlight {
            activity: activity.into(),
            error: Box::new(error),
        }
    }

    /// A transport failure `source` caused while the worker was doing
    /// `what`.
    pub fn transport_source(
        what: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Transport {
            what: what.into(),
            source: Some(Box::new(source)),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// A tensor operation's failure is a model computation's: the conversion through `nn::Error`
/// recovers the backend's typed cause the tensor library carried boxed.
impl From<uniserve_tensor::Error> for Error {
    fn from(error: uniserve_tensor::Error) -> Self {
        Self::Nn(nn::Error::from(error))
    }
}

/// A refusal of the device backend outside a tensor operation: a context, an allocation, a copy.
impl From<backend::Error> for Error {
    fn from(error: backend::Error) -> Self {
        Self::Nn(nn::Error::Backend(error))
    }
}

/// A second outcome of a completion that already has one is the worker's state error.
impl From<ipc::execution::Settled> for Error {
    fn from(_: ipc::execution::Settled) -> Self {
        Self::State("operation has already completed")
    }
}

/// A batch the crossing's own checks refuse (`ipc::execution`) is a batch this worker cannot serve.
impl From<ipc::ValidationError> for Error {
    fn from(error: ipc::ValidationError) -> Self {
        Self::Invalid(error.reason().to_owned())
    }
}

pub use ipc::ErrorCode;

/// A classified failure, as the main process reports it.
///
/// The code and fatality are inline; the message is boxed, so a `Result` that
/// carries a failure stays small on the success path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub code: ErrorCode,
    /// The device or process must stop serving after reporting it: the CUDA
    /// programming guide requires the process to be terminated and relaunched
    /// after an illegal address or a failed launch.
    pub fatal: bool,
    message: Box<str>,
    /// Secondary failures observed while cleaning up after this one.
    notes: Vec<String>,
}

impl Failure {
    /// A failure with an explicit code and its default fatality.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            fatal: code.fatal(),
            message: message.into().into_boxed_str(),
            notes: Vec::new(),
        }
    }

    /// Classify an internal error by its type. `context` prefixes the message, naming the
    /// operation that failed.
    ///
    /// A CUDA driver failure, the worker's own or one an operator launch or a model layer
    /// raised, is classified by its result code; any other failure by its own kind. The message
    /// never decides the class.
    pub fn classify(error: Error, context: Option<&str>) -> Self {
        let mut message = report(&error);
        if let Some(context) = context.filter(|context| !context.is_empty()) {
            message = format!("{context}: {message}");
        }
        let code = match driver_failure(&error) {
            Some(failure) => driver_code(failure),
            None => kind_code(&error),
        };
        Self::new(code, message)
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for Failure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)?;
        for note in &self.notes {
            write!(formatter, "; {note}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Failure {}

impl From<Error> for Failure {
    fn from(error: Error) -> Self {
        Self::classify(error, None)
    }
}

/// `error` and each of its sources, joined by `: `: the message of a failure
/// whose cause travels as its source.
pub fn report(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}

/// The CUDA failure an error is: the worker's own, an operator launch's, a model layer's, or one
/// a shared failure or the work in flight carries.
fn driver_failure(error: &Error) -> Option<&platform::Error> {
    match error {
        Error::Cuda(failure) => Some(failure),
        Error::Ops(failure) => ops_driver_failure(failure),
        Error::Nn(nn::Error::Backend(backend::Error::Platform(failure))) => Some(failure),
        Error::Nn(
            nn::Error::Ops(failure)
            | nn::Error::Backend(
                backend::Error::Ops(failure)
                | backend::Error::Launch {
                    source: failure, ..
                },
            ),
        ) => ops_driver_failure(failure),
        Error::Shared(shared) => shared.downcast_ref::<Error>().and_then(driver_failure),
        Error::InFlight { error, .. } => driver_failure(error),
        _ => None,
    }
}

/// The CUDA failure an operator's failure is, when it is one.
fn ops_driver_failure(error: &ops::Error) -> Option<&platform::Error> {
    match error {
        ops::Error::Platform(failure) => Some(failure),
        _ => None,
    }
}

/// The code of a CUDA failure, by the result its call returned.
fn driver_code(failure: &platform::Error) -> ErrorCode {
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
        // A shared failure is the worker's own whenever a worker raised it; one the crossing
        // raised by itself (`ipc::execution::Refusal`) is a computation that did not complete.
        Error::Shared(shared) => shared
            .downcast_ref::<Error>()
            .map_or(ErrorCode::ComputeError, kind_code),
        Error::InFlight { error, .. } => kind_code(error),
        Error::Invalid(_) => ErrorCode::InvalidDescriptor,
        Error::Resource(_) | Error::ReadBackpressure { .. } => ErrorCode::ResourceError,
        Error::Unsupported(_) | Error::Interface(_) => ErrorCode::UnsupportedSetup,
        Error::Invariant(_) => ErrorCode::InvariantViolation,
        Error::Transport { .. } | Error::Cuda(_) | Error::State(_) => ErrorCode::ComputeError,
        // An operator or model failure is a computation failure, unless its own
        // kind says otherwise; the message carries the library's text.
        Error::Ops(_) | Error::Nn(_) | Error::Media(_) => ErrorCode::ComputeError,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn driver(code: i32, message: &'static str) -> Error {
        Error::Cuda(platform::Error::Driver {
            call: "cuMemcpyBatchAsync_v2",
            code,
            name: "CUDA_ERROR",
            message,
        })
    }

    /// A driver failure an operator launch or a model layer raised is classified by its result
    /// code, as the worker's own is, and the message only reports it.
    #[test]
    fn a_driver_failure_inside_a_launch_is_classified_by_its_code() {
        let platform = || platform::Error::Driver {
            call: "cuLaunchKernel",
            code: 700,
            name: "CUDA_ERROR_ILLEGAL_ADDRESS",
            message: "an illegal memory access was encountered",
        };
        for error in [
            Error::Ops(ops::Error::Platform(platform())),
            Error::Nn(nn::Error::Backend(backend::Error::Platform(platform()))),
            Error::Nn(nn::Error::Ops(ops::Error::Platform(platform()))),
        ] {
            let failure = Failure::classify(error, Some("decode"));
            assert_eq!(failure.code, ErrorCode::FatalWorkerFailure);
            assert!(failure.fatal);
            assert!(failure.message().starts_with("decode: "));
        }
    }

    /// A message that only quotes a driver failure is classified by the error's own kind.
    #[test]
    fn a_message_does_not_decide_the_class() {
        let quoted = Failure::classify(
            Error::Interface("CUDA out of memory after an illegal memory access".into()),
            None,
        );
        assert_eq!(quoted.code, ErrorCode::UnsupportedSetup);
    }

    #[test]
    fn allocation_failures_are_resource_errors() {
        for error in [
            Error::Cuda(platform::Error::Driver {
                call: "cuMemCreate",
                code: 2,
                name: "CUDA_ERROR_OUT_OF_MEMORY",
                message: "out of memory",
            }),
            Error::Resource("host staging pool is full"),
        ] {
            let error = Failure::classify(error, None);
            assert_eq!(error.code, ErrorCode::ResourceError);
            assert!(!error.fatal);
        }
    }

    #[test]
    fn a_shared_failure_keeps_the_class_of_the_failure_it_shares() {
        let cancelled = std::sync::Arc::new(Error::Resource("transfer read was cancelled"));
        let observed = Failure::classify(Error::Shared(cancelled), None);
        assert_eq!(observed.code, ErrorCode::ResourceError);
        assert_eq!(observed.message(), "transfer read was cancelled");
    }

    #[test]
    fn a_driver_result_code_decides_whatever_the_driver_text_says() {
        // CUDA_ERROR_OUT_OF_MEMORY is the device budget, not a fault.
        assert!(!Failure::classify(driver(2, "failed"), None).fatal);
        // CUDA_ERROR_ILLEGAL_ADDRESS leaves the context unusable.
        let faulted = Failure::classify(driver(700, "failed"), None);
        assert_eq!(faulted.code, ErrorCode::FatalWorkerFailure);
        assert!(faulted.fatal);
        // CUDA_ERROR_INVALID_VALUE is a rejected call, even with this text.
        assert_eq!(
            Failure::classify(driver(1, "out of memory"), None).code,
            ErrorCode::ComputeError
        );
    }

    /// A failure states the work in flight when it was taken, and stating it changes nothing else:
    /// an illegal address is still a fatal driver failure, a refusal is still a refusal, and the
    /// message carries the bucket and the operator beside the driver's own text.
    #[test]
    fn a_failure_names_the_work_in_flight_without_losing_its_kind() {
        let faulted = Failure::classify(
            Error::in_flight(
                "warmup, capturing bucket decode_sampled/8 rows/512 tokens",
                driver(700, "an illegal memory access was encountered"),
            ),
            None,
        );
        assert_eq!(faulted.code, ErrorCode::FatalWorkerFailure);
        assert!(faulted.fatal);
        assert!(
            faulted
                .message()
                .contains("decode_sampled/8 rows/512 tokens"),
            "the fault does not name the work in flight: {}",
            faulted.message()
        );
        assert!(
            faulted.message().contains("illegal memory access"),
            "the fault lost the driver's own text: {}",
            faulted.message()
        );

        let refused = Failure::classify(
            Error::in_flight(
                "serving batch 18",
                Error::Invalid("a batch of 3 rows".into()),
            ),
            None,
        );
        assert_eq!(refused.code, ErrorCode::InvalidDescriptor);
        assert!(!refused.fatal);
    }

    #[test]
    fn a_broken_invariant_is_fatal() {
        let invariant = Failure::classify(Error::Invariant("a plane is mapped twice".into()), None);
        assert_eq!(invariant.code, ErrorCode::InvariantViolation);
        assert!(invariant.fatal);
    }
}
