//! Device storage accounting against the GPU's storage ledger.
//!
//! A deployment may use `⌊total × fraction⌋` of a GPU (its quota), and every
//! world of the deployment that uses the GPU, with the host's Transfer
//! Server, holds its storage within that one quota: the host's Storage Server
//! keeps the GPU's ledger (`::storage::GpuLedger`). A rank's grant is the
//! quota less everything the ledger books beside the rank's own process
//! (another world's weights, KV cache segments, graph pool and process, the
//! Transfer Server's context, the storages this rank's scope retains) and
//! less what the process holds now, never more than the device has free.
//! In a deployment of one world this is the fraction of the device less
//! what the process and its retained storages hold; beside a predecessor
//! world it is also less everything the predecessor holds, so two worlds
//! never each size themselves to the whole quota.
//!
//! The process figure is NVML's per-process usage, which counts every
//! allocation of the process's context: PyTorch's caching allocator
//! (activations, graph pools), Rust driver allocations (arenas, workspaces),
//! communicator buffers and loaded modules. The caching allocator first
//! returns unused cached blocks, so reusable cache does not count against
//! the grant. NVML attributes storage held only through descriptors to no
//! process, so the storages the Storage Server retains count through the
//! ledger instead: the weights, the exclusive storages and the KV cache
//! segments. So does the part of the plugins' communication buffers NVML
//! attributes to no process, which counts as the process's own.
//!
//! Exclusive storages a predecessor of the rank's scope and world left that
//! the rank has not taken yet are about to be reused or released by its
//! owners, so they count as free.
//!
//! Once warm, the rank books what the server cannot see in the ledger
//! ([`DeviceBudget::book`]): its graph pools' storage and what its process
//! holds beside them, with the product storage it has yet to make resident.

use std::collections::BTreeMap;
use std::sync::Arc;

use ::storage::{GpuLedger, Item};
use cuda::{DeviceId, Nvml};

use crate::ffi::Function;
use crate::storage::retained::Retained;
use crate::{Error, Result};

/// Per-process device storage measurements against the GPU's ledger.
pub(crate) struct DeviceBudget {
    nvml: Nvml,
    retained: Arc<Retained>,
    /// The plugins' communication buffers NVML attributes to no process,
    /// by device.
    unattributed: BTreeMap<DeviceId, u64>,
    /// `torch.cuda.synchronize(device)` and `torch.cuda.empty_cache()`,
    /// called before a measurement so blocks freed by completed work are
    /// returned to the device and process storage excludes reusable cache.
    synchronize: Function,
    empty_cache: Function,
}

/// A device's storage grant and its free bytes, at one measurement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Budget {
    /// Bytes the rank may still allocate on the device.
    pub available: u64,
    /// Bytes the rank may hold beside its process once every other world
    /// of its deployment has left the device: the quota less the process
    /// and what the ledger books for the rank's own world and the Transfer
    /// Server.
    pub full: u64,
    /// Bytes the device reports free, with the scope's reclaimable
    /// retained bytes.
    pub free: u64,
    /// Bytes the process holds, with the storages its scope retains.
    pub process: u64,
    /// Total device bytes.
    pub total: u64,
    /// The deployment's ledger of the device's GPU at the measurement.
    pub ledger: GpuLedger,
}

impl DeviceBudget {
    pub(crate) fn new(
        synchronize: Function,
        empty_cache: Function,
        retained: Arc<Retained>,
        unattributed: BTreeMap<DeviceId, u64>,
    ) -> Result<Self> {
        Ok(Self {
            nvml: Nvml::init()?,
            retained,
            unattributed,
            synchronize,
            empty_cache,
        })
    }

    /// Device storage this process holds on `device` after PyTorch returns
    /// its unused cached blocks: NVML's figure with the plugins' buffers it
    /// attributes to no process, without the storages the server retains,
    /// among them the segments this process created and keeps the creating
    /// handles of, which NVML attributes to it ([`cuda::created_bytes`]).
    ///
    /// Fails when NVML attributes no usage on the device to this process, as
    /// inside a PID namespace NVML does not report.
    fn own_bytes(&self, device: DeviceId) -> Result<u64> {
        self.synchronize
            .invoke(&[(&tvm_ffi::Any::from(i64::from(device.0))).into()])?;
        self.empty_cache.invoke(&[])?;
        let process = self
            .nvml
            .process_storage(device)
            .map_err(|error| match error {
                cuda::Error::ProcessNotFound { .. } => Error::Unsupported(format!(
                    "NVML reports no device storage for this process on {device}"
                )),
                error => Error::Cuda(error),
            })?;
        let created = cuda::created_bytes(device);
        Ok(process.saturating_sub(created) + self.unattributed.get(&device).copied().unwrap_or(0))
    }

    /// Device storage this process holds on `device`, with the storages its
    /// scope retains there.
    pub(crate) fn process_bytes(&self, device: DeviceId) -> Result<u64> {
        Ok(self.own_bytes(device)? + self.retained.scope_bytes(device)?.held)
    }

    /// The storage grant on `device` for a rank allowed `fraction` of it,
    /// stating the rank's world's quota in the GPU's ledger.
    pub(crate) fn budget(&self, device: DeviceId, fraction: f64) -> Result<Budget> {
        let own = self.own_bytes(device)?;
        let scope = self.retained.scope_bytes(device)?;
        let storage = device.memory_info()?;
        // Stating the quota books nothing: the rank books its own items
        // once it is warm.
        let ledger = self.retained.book(
            device,
            storage.total,
            quota(storage.total, fraction),
            Vec::new(),
        )?;
        let quota = ledger.quota.unwrap_or(quota(storage.total, fraction));
        // The measurement of the process stands for what it, or another
        // process of its scope and world it replaces, booked; the
        // reclaimable storages count as free.
        let others = ledger
            .used()
            .saturating_sub(ledger.bytes(|entry| self.retained.books(entry)))
            .saturating_sub(scope.reclaimable);
        let world = self.retained.world();
        let stays = ledger
            .bytes(|entry| {
                !self.retained.books(entry)
                    && (entry.worlds.contains(&world) || entry.worlds.is_empty())
            })
            .saturating_sub(scope.reclaimable);
        let free = storage.free + scope.reclaimable;
        Ok(Budget {
            available: quota.saturating_sub(own + others).min(free),
            full: quota.saturating_sub(own + stays),
            free,
            process: own + scope.held,
            total: storage.total,
            ledger,
        })
    }

    /// What the warm rank books on `device`, for `fraction` of it: its graph
    /// pools' `graph` bytes, and what the process holds beside them and the
    /// retained storages, with the `pending` bytes of product storage it
    /// has yet to make resident.
    pub(crate) fn booking(
        &self,
        device: DeviceId,
        fraction: f64,
        graph: u64,
        pending: u64,
    ) -> Result<DeviceBooking> {
        let own = self.own_bytes(device)?;
        let total = device.memory_info()?.total;
        Ok(DeviceBooking {
            device,
            total,
            quota: quota(total, fraction),
            graph,
            process: own.saturating_sub(graph) + pending,
            profiling: 0,
        })
    }
}

/// What a warm rank books on one device: the items of its process the
/// Storage Server cannot see.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeviceBooking {
    pub device: DeviceId,
    /// The device's total storage.
    pub total: u64,
    /// The deployment's quota of the device.
    pub quota: u64,
    /// The storage the rank's graph pools map ([`Item::GraphPool`]).
    pub graph: u64,
    /// What the process holds beside its graph pools and the retained
    /// storages, with the product storage it has yet to make resident
    /// ([`Item::Process`]).
    pub process: u64,
    /// The device storage a profiling session's collectors hold while the
    /// session runs (the range profiler's pass), which the process's item
    /// books beside `process`; 0 without a session.
    pub profiling: u64,
}

impl DeviceBooking {
    /// Books the items in the GPU's ledger, the graph pool only while
    /// `graphs` are resident: a sleeping rank's graph storage is released.
    /// Returns the ledger after the booking.
    ///
    /// Fails when the booking takes the ledger past its quota: the rank's
    /// initialized runtime, with a profiling session's storage, exceeds what
    /// the deployment may hold.
    pub(crate) fn submit(&self, retained: &Retained, graphs: bool) -> Result<GpuLedger> {
        let graph = if graphs { self.graph } else { 0 };
        let process = self.process + self.profiling;
        let ledger = retained.book(
            self.device,
            self.total,
            self.quota,
            vec![(Item::GraphPool, graph), (Item::Process, process)],
        )?;
        tracing::info!(
            device = %self.device,
            graph_pool = graph,
            process,
            profiling = self.profiling,
            booked = ledger.used(),
            quota = ledger.quota,
            "booked the worker's device storage"
        );
        Ok(ledger)
    }
}

/// Total storage of a CUDA device, in bytes.
pub(crate) fn total_bytes(device: DeviceId) -> Result<u64> {
    Ok(device.memory_info()?.total)
}

/// A deployment's quota of a device of `total` bytes for `fraction` of it:
/// `⌊total × fraction⌋`.
pub(crate) fn quota(total: u64, fraction: f64) -> u64 {
    (total as f64 * fraction) as u64
}
