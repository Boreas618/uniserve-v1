//! Typed failures of the runtime-loaded CUDA, NVML and NCCL libraries.

use std::io;
use std::sync::Arc;

use crate::DeviceId;

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A failure reported by this crate.
///
/// The type is `Clone` because library loading is attempted once per process
/// and its outcome is returned to every later caller; foreign errors are kept
/// behind `Arc` for that reason.
#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    /// A shared library could not be opened (for example, no NVIDIA driver).
    #[error("cannot load {library}: {source}")]
    Load {
        library: &'static str,
        #[source]
        source: Arc<libloading::Error>,
    },

    /// A loaded library does not export a required entry point, typically
    /// because it predates the API.
    #[error("{library} does not export {symbol}: {source}")]
    Symbol {
        library: &'static str,
        symbol: &'static str,
        #[source]
        source: Arc<libloading::Error>,
    },

    /// A CUDA driver call returned a `CUresult` other than `CUDA_SUCCESS`.
    #[error("{call} failed with {name} ({code}): {message}")]
    Driver {
        call: &'static str,
        code: i32,
        name: &'static str,
        message: &'static str,
    },

    /// An NCCL call returned an `ncclResult_t` other than `ncclSuccess`.
    #[error("{call} failed with NCCL result {code}: {message}")]
    Nccl {
        call: &'static str,
        code: i32,
        message: &'static str,
    },

    /// An NVML call returned an `nvmlReturn_t` other than `NVML_SUCCESS`.
    #[error("{call} failed with NVML return {code}: {message}")]
    Nvml {
        call: &'static str,
        code: i32,
        message: &'static str,
    },

    /// The operation is not issued on a stream that is being captured into a
    /// CUDA graph.
    #[error("{call} is not permitted on a capturing stream")]
    Capturing { call: &'static str },

    /// The event has no recorded work to time or export.
    #[error("the CUDA event has not been recorded")]
    NotRecorded,

    /// A caller-supplied value violates the operation's contract.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// NVML attributes no device memory on the device to this process, for
    /// example inside a PID namespace NVML does not report.
    #[error("NVML reports no device memory for process {pid} on {device}")]
    ProcessNotFound { pid: u32, device: DeviceId },

    /// An operating-system call failed.
    #[error("{call} failed: {source}")]
    Io {
        call: &'static str,
        #[source]
        source: Arc<io::Error>,
    },

    /// Deferred releases failed since the previous [`crate::reclaim::flush`].
    #[error("{failed} deferred CUDA releases failed; first failure: {first}")]
    Reclaim { failed: usize, first: Box<Error> },
}

impl Error {
    pub(crate) fn io(call: &'static str, source: io::Error) -> Self {
        Self::Io {
            call,
            source: Arc::new(source),
        }
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidArgument(message.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_errors_name_the_call_and_status() {
        let error = Error::Driver {
            call: "cuMemAlloc_v2",
            code: 2,
            name: "CUDA_ERROR_OUT_OF_MEMORY",
            message: "out of memory",
        };

        assert_eq!(
            error.to_string(),
            "cuMemAlloc_v2 failed with CUDA_ERROR_OUT_OF_MEMORY (2): out of memory"
        );
    }

    #[test]
    fn reclaim_errors_report_the_first_failure() {
        let error = Error::Reclaim {
            failed: 3,
            first: Box::new(Error::Capturing {
                call: "cuMemcpyBatchAsync_v2",
            }),
        };

        assert_eq!(
            error.to_string(),
            "3 deferred CUDA releases failed; first failure: \
             cuMemcpyBatchAsync_v2 is not permitted on a capturing stream"
        );
    }

    #[test]
    fn process_errors_name_the_device() {
        let error = Error::ProcessNotFound {
            pid: 42,
            device: DeviceId(3),
        };

        assert_eq!(
            error.to_string(),
            "NVML reports no device memory for process 42 on cuda:3"
        );
    }
}
