//! Device storage accounting for startup planning.
//!
//! A rank's device budget is a fraction of the device's total memory minus
//! what the process already holds, never more than the device has free. The
//! process figure is NVML's per-process usage, which counts every allocation
//! of the process's context: PyTorch's caching allocator (weights,
//! activations, graph pools), Rust driver allocations (cache planes, arenas,
//! workspaces), communicator buffers and loaded modules. The caching
//! allocator first returns unused cached blocks, so reusable cache does not
//! count against the grant.

use uniserve_cuda::{DeviceId, Nvml};

use crate::ffi::Function;
use crate::{Error, Result};

/// Per-process device storage measurements.
pub(crate) struct DeviceBudget {
    nvml: Nvml,
    /// `torch.cuda.synchronize(device)` and `torch.cuda.empty_cache()`,
    /// called before a measurement so blocks freed by completed work are
    /// returned to the device and process memory excludes reusable cache.
    synchronize: Function,
    empty_cache: Function,
}

/// A device's storage grant and its free bytes, at one measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Budget {
    /// Bytes the rank may still allocate on the device.
    pub available: u64,
    /// Bytes the device reports free.
    pub free: u64,
    /// Bytes the process holds.
    pub process: u64,
    /// Total device bytes.
    pub total: u64,
}

impl DeviceBudget {
    pub(crate) fn new(synchronize: Function, empty_cache: Function) -> Result<Self> {
        Ok(Self {
            nvml: Nvml::init()?,
            synchronize,
            empty_cache,
        })
    }

    /// Device memory this process holds on `device`, after PyTorch returns
    /// its unused cached blocks.
    ///
    /// Fails when NVML attributes no usage on the device to this process, as
    /// inside a PID namespace NVML does not report.
    pub(crate) fn process_bytes(&self, device: DeviceId) -> Result<u64> {
        self.synchronize
            .invoke(&[(&tvm_ffi::Any::from(i64::from(device.0))).into()])?;
        self.empty_cache.invoke(&[])?;
        self.nvml
            .process_memory(device)
            .map_err(|error| match error {
                uniserve_cuda::Error::ProcessNotFound { .. } => Error::Unsupported(format!(
                    "NVML reports no device storage for this process on {device}"
                )),
                error => Error::Cuda(error),
            })
    }

    /// The storage grant on `device` for a rank allowed `fraction` of it.
    pub(crate) fn budget(&self, device: DeviceId, fraction: f64) -> Result<Budget> {
        let process = self.process_bytes(device)?;
        let memory = device.memory_info()?;
        Ok(Budget {
            available: grant(memory.total, fraction, process, memory.free),
            free: memory.free,
            process,
            total: memory.total,
        })
    }
}

/// Total memory of a CUDA device, in bytes.
pub(crate) fn total_bytes(device: DeviceId) -> Result<u64> {
    Ok(device.memory_info()?.total)
}

/// `min(max(0, ⌊total × fraction⌋ − process), free)`.
pub(crate) fn grant(total: u64, fraction: f64, process: u64, free: u64) -> u64 {
    let limit = (total as f64 * fraction) as u64;
    limit.saturating_sub(process).min(free)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grant_is_the_fraction_left_after_process_storage() {
        let gib = 1 << 30;
        // 70% of 100 GiB minus 20 GiB held, within 60 GiB free.
        assert_eq!(grant(100 * gib, 0.7, 20 * gib, 60 * gib), 50 * gib);
        // Free memory bounds the grant when other processes hold storage.
        assert_eq!(grant(100 * gib, 0.7, 20 * gib, 30 * gib), 30 * gib);
        // A process beyond its fraction has no grant left.
        assert_eq!(grant(100 * gib, 0.5, 60 * gib, 40 * gib), 0);
    }
}
