//! The error of device work.
//!
//! Every module of the worker, its runners and the rank's memory fails with [`Error`]. A failed
//! job reaches its caller as this error in words, stated with the work in flight
//! ([`Error::InFlight`]); a failed call of a batch states its class in the batch's output
//! (`crate::batch::ErrorCode`), and a context that a failure left unusable is the worker's sticky
//! error (`Worker::sticky_error`).

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
    /// An operator's launch, a kernel's load, or a refusal of the device backend outside a tensor
    /// operation (a context, an allocation, a copy) failed.
    #[error(transparent)]
    Ops(#[from] ops::Error),
    /// A model's computation failed: a module, a tensor operation or a launch of the device
    /// backend, which the tensor library and `nn` carry here as typed causes.
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
    /// The failure of a batch that copies had started with, and what the worker returns for
    /// them: their result is unknown, so each is reported failed. It reports and classifies as
    /// the batch's failure.
    #[error("{error}")]
    Copies {
        /// The batch's failure.
        error: Box<Error>,
        /// A `Finish { ok: false }` per copy that started with the batch, stamped with the
        /// batch's group and number, returned once the device has passed what the batch issued.
        finished: Vec<storage::runtime::Request>,
    },
    /// A failure every party of one operation observes: the result of a read,
    /// the state of a pool or an export that several owners consult. It
    /// reports and classifies as the failure it shares.
    #[error(transparent)]
    Shared(std::sync::Arc<crate::worker::completion::Fault>),
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

    /// The `Finish { ok: false }` of each copy that started with a failed batch, which the
    /// batch's failure carries ([`Error::Copies`]), also inside the statement of the work in
    /// flight or a failure a submission shares: what the scheduler submits for those copies.
    /// Empty for any other failure.
    pub fn finished(&self) -> &[storage::runtime::Request] {
        match self {
            Self::Copies { finished, .. } => finished,
            Self::InFlight { error, .. } => error.finished(),
            Self::Shared(shared) => shared.downcast_ref::<Error>().map_or(&[], Error::finished),
            _ => &[],
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
impl From<tensor::Error> for Error {
    fn from(error: tensor::Error) -> Self {
        Self::Nn(nn::Error::from(error))
    }
}

/// A second outcome of a completion that already has one is the worker's state error.
impl From<crate::worker::completion::Settled> for Error {
    fn from(_: crate::worker::completion::Settled) -> Self {
        Self::State("operation has already completed")
    }
}

/// A batch or a frame the protocol's own checks refuse (`ipc`) is one this worker cannot serve.
impl From<crate::batch::ValidationError> for Error {
    fn from(error: crate::batch::ValidationError) -> Self {
        Self::Invalid(error.reason().to_owned())
    }
}

/// `error` and each of its sources, joined by `: `: the message of a failure
/// whose cause travels as its source.
pub(crate) fn report(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}
