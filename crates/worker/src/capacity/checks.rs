//! Startup storage checks against the rank's device grant. The warm rank's
//! whole runtime is checked by booking it in the GPU's ledger
//! (`DeviceBooking::submit`), which refuses it past the quota.

use std::collections::BTreeMap;

use super::device::{DeviceBudget, total_bytes};
use crate::config::{Device, WorkerConfig};
use crate::cuda_graph::graph_storage_budget_bytes;
use crate::{Error, Result};

/// Every CUDA device other than the primary one needs its own grant for its
/// fixed storage plus graph capture. Cache sizing already charged the primary
/// device's fixed storage.
pub(crate) fn auxiliary(
    config: &WorkerConfig,
    fixed: &BTreeMap<Device, u64>,
    budget: &DeviceBudget,
) -> Result<()> {
    for (&device, bytes) in fixed {
        let Device::Cuda(index) = device else {
            continue;
        };
        if device == config.device {
            continue;
        }
        let available = budget.budget(index, config.mem_fraction_static)?.available;
        let reserved = bytes + graph_storage_budget_bytes(total_bytes(index)? as i64);
        if reserved > available {
            return Err(Error::Unsupported(format!(
                "runtime storage on {device} requires {reserved} bytes, but its static storage grant is {available} bytes"
            )));
        }
    }
    Ok(())
}
