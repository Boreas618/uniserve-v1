//! Device storage capacity: the rank's grant on a GPU, its graph storage
//! budget, and the startup checks that refuse a runtime that does not fit.
//!
//! A deployment may use `⌊total × fraction⌋` of a GPU (its quota), and every
//! world of the deployment that uses the GPU, with the host's Transfer
//! Server, holds its storage within that one quota: the Storage Server keeps
//! the GPU's ledger. A rank's grant is the quota less everything the ledger
//! books beside the rank's own process, and less what the process holds now.
//!
//! The plan that sizes the weights, the cache, the products and the graph
//! pools from those figures is [`plan`]; it takes the rank's launch
//! parameters as [`CapacityRequest`], which the engine builds from the launch
//! description it already holds.

mod checks;
pub(crate) mod device;

pub(crate) use checks::auxiliary;
pub(crate) use device::{Budget, DeviceBooking, DeviceBudget, total_bytes};

/// The share of a device's bytes a rank may reserve for graph storage: the
/// capture pools of its bucket graphs and the scratch they allocate.
///
/// The figure is capacity planning, so it lives here and the arena that maps
/// the pools takes the bytes as an input. Ten percent of the device holds the
/// buckets of a causal language model and a media model of the slice with
/// room to spare; a rank that captures more books it as its own graph item,
/// so the ledger refuses it before the device runs out.
pub fn graph_storage_budget_bytes(total_device_bytes: i64) -> u64 {
    (total_device_bytes.max(0) as f64 * 0.10) as u64
}

/// A deployment's quota of a device of `total` bytes for `fraction` of it:
/// `⌊total × fraction⌋`.
pub fn quota(total: u64, fraction: f64) -> u64 {
    (total as f64 * fraction) as u64
}
