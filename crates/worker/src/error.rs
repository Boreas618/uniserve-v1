//! Errors from request execution and resource ownership.

/// Invalid submissions and operations on unavailable worker state.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Transport(String),
    #[error("{0}")]
    Cuda(String),
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
    /// A Python numerical function raised. `kind` is the exception type and
    /// `traceback` the Python traceback carried by TVM-FFI.
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
