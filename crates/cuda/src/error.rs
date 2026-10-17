//! Typed failures of the runtime-loaded CUDA, NVML and NCCL libraries.

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

    /// Deferred releases failed since the previous [`crate::reclaim::flush`].
    #[error("{failed} deferred CUDA releases failed; first failure: {first}")]
    Reclaim { failed: usize, first: Box<Error> },

    /// `atexit` refused the handler of
    /// [`crate::PrimaryContext::reset_all_at_exit`].
    #[error("cannot register the exit handler that destroys the device contexts")]
    ExitHandler,
}

/// `CUDA_ERROR_OUT_OF_MEMORY`: the call could not allocate enough memory or
/// other resources.
const CUDA_ERROR_OUT_OF_MEMORY: i32 = 2;

/// Driver results after which the process cannot continue using CUDA.
///
/// The driver documents each as leaving the process in an inconsistent
/// state in which further CUDA work returns the same error, or its context
/// as unusable (`CUDA_ERROR_ASSERT`). An uncorrectable ECC error is included
/// because the memory the process computes with can no longer be trusted.
const UNRECOVERABLE_DRIVER_RESULTS: &[i32] = &[
    214, // CUDA_ERROR_ECC_UNCORRECTABLE
    226, // CUDA_ERROR_CONTAINED
    700, // CUDA_ERROR_ILLEGAL_ADDRESS
    702, // CUDA_ERROR_LAUNCH_TIMEOUT
    710, // CUDA_ERROR_ASSERT
    714, // CUDA_ERROR_HARDWARE_STACK_ERROR
    715, // CUDA_ERROR_ILLEGAL_INSTRUCTION
    716, // CUDA_ERROR_MISALIGNED_ADDRESS
    717, // CUDA_ERROR_INVALID_ADDRESS_SPACE
    718, // CUDA_ERROR_INVALID_PC
    719, // CUDA_ERROR_LAUNCH_FAILED
    721, // CUDA_ERROR_TENSOR_MEMORY_LEAK
    810, // CUDA_ERROR_MPS_CLIENT_TERMINATED
    911, // CUDA_ERROR_EXTERNAL_DEVICE
];

impl Error {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidArgument(message.into())
    }

    /// The driver could not allocate memory or another resource for the
    /// call; the process and its contexts remain usable.
    pub fn is_out_of_memory(&self) -> bool {
        match self {
            Self::Driver { code, .. } => *code == CUDA_ERROR_OUT_OF_MEMORY,
            Self::Reclaim { first, .. } => first.is_out_of_memory(),
            _ => false,
        }
    }

    /// The driver reported a failure after which this process cannot
    /// continue using CUDA and must be restarted.
    pub fn is_unrecoverable(&self) -> bool {
        match self {
            Self::Driver { code, .. } => UNRECOVERABLE_DRIVER_RESULTS.contains(code),
            Self::Reclaim { first, .. } => first.is_unrecoverable(),
            _ => false,
        }
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
    fn driver_results_decide_the_failure_class() {
        let driver = |code| Error::Driver {
            call: "cuMemcpyBatchAsync_v2",
            code,
            name: "CUDA_ERROR",
            message: "failed",
        };

        assert!(driver(2).is_out_of_memory());
        assert!(!driver(2).is_unrecoverable());
        assert!(driver(700).is_unrecoverable());
        assert!(driver(719).is_unrecoverable());
        // CUDA_ERROR_INVALID_VALUE reports a rejected call.
        assert!(!driver(1).is_out_of_memory() && !driver(1).is_unrecoverable());

        let reclaim = Error::Reclaim {
            failed: 1,
            first: Box::new(driver(700)),
        };
        assert!(reclaim.is_unrecoverable());
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
