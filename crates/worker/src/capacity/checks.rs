//! Startup storage checks against the rank's device grant.

use std::collections::BTreeMap;

use super::device::{DeviceBudget, total_bytes};
use crate::config::{Device, WorkerConfig};
use crate::runner::graph_storage_budget_bytes;
use crate::{Error, Result};

/// Every CUDA device other than the primary one needs its own grant for its
/// fixed storage plus graph capture. KV sizing already charged the primary
/// device's fixed storage.
pub fn auxiliary(
    config: &WorkerConfig,
    fixed: &BTreeMap<String, u64>,
    budget: &DeviceBudget,
) -> Result<()> {
    for (name, bytes) in fixed {
        let Ok(Device::Cuda(index)) = Device::parse(name, "device") else {
            continue;
        };
        if *name == config.device {
            continue;
        }
        let available = budget.budget(index, config.kv_storage_fraction)?.available;
        let reserved = bytes + graph_storage_budget_bytes(total_bytes(index)? as i64);
        if reserved > available {
            return Err(Error::Unsupported(format!(
                "runtime storage on {name} requires {reserved} bytes, but its static storage grant is {available} bytes"
            )));
        }
    }
    Ok(())
}

/// After warmup and sealing, the rank's resident storage plus the product
/// storage not yet resident must fit its grant on every CUDA device.
/// `resident_products` reports the product bytes already allocated on a
/// device.
pub fn startup(
    config: &WorkerConfig,
    device_product_bytes: u64,
    resident_products: impl Fn(&str) -> u64,
    budget: &DeviceBudget,
) -> Result<()> {
    for name in config.devices() {
        let Ok(Device::Cuda(index)) = Device::parse(name, "device") else {
            continue;
        };
        // Product storage lives on the primary device only.
        let products = if name == config.device {
            device_product_bytes
        } else {
            0
        };
        let measured = budget.budget(index, config.kv_storage_fraction)?;
        let grant = (measured.total as f64 * config.kv_storage_fraction) as u64;
        let remaining = products.saturating_sub(resident_products(name));
        if remaining > measured.available || measured.process + remaining > grant {
            return Err(Error::Unsupported(format!(
                "initialized runtime on {name} exceeds its static storage grant: it requires {} bytes ({} process-resident bytes and {remaining} reserved product bytes) of a {grant}-byte grant, with {} device-free bytes",
                measured.process + remaining,
                measured.process,
                measured.free,
            )));
        }
    }
    Ok(())
}
