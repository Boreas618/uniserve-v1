//! Device storage accounting against the GPU's storage ledger.
//!
//! A deployment may use `⌊total × fraction⌋` of a GPU (its quota), and every
//! epoch of the deployment that uses the GPU, with the host's Transfer
//! Server, holds its storage within that one quota: the host's Storage Server
//! keeps the GPU's ledger (`::storage::GpuLedger`). A rank's grant is the
//! quota less everything the ledger books beside the rank's own process
//! (another epoch's weights, KV cache segments, graph pool and process, the
//! Transfer Server's context, the storages this rank's scope retains) and
//! less what the process holds now, never more than the device has free.
//! In a deployment of one epoch this is the fraction of the device less
//! what the process and its retained storages hold; beside a predecessor
//! epoch it is also less everything the predecessor holds, so two epochs
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
//! Exclusive storages a predecessor of the rank's scope and epoch left that
//! the rank has not taken yet are about to be reused or released by its
//! owners, so they count as free.
//!
//! Once warm, the rank books what the server cannot see in the ledger
//! ([`DeviceBudget::book`]): its graph pools' storage and what its process
//! holds beside them, with the product storage it has yet to make resident.
//!
//! An expert transport is such storage, and it comes late: the rank creates it
//! at its first capture, because only then does it hold the program that knows
//! its size. The plan therefore fits the cache around a bound of it
//! (`RankFacts::transport_bytes`), the rank compares that bound with what the
//! creation took, and the booking after warmup counts the real figure in the
//! process's item.

use std::collections::BTreeMap;
use std::sync::Arc;

use ::storage::{GpuLedger, Item};
use platform::{DeviceId, Nvml};

use crate::memory::retained::Retained;
use crate::{Error, Result};

/// Per-process device storage measurements against the GPU's ledger.
pub(crate) struct DeviceBudget {
    nvml: Nvml,
    retained: Arc<Retained>,
    /// The plugins' communication buffers NVML attributes to no process,
    /// by device.
    unattributed: BTreeMap<DeviceId, u64>,
    /// Synchronizes the rank's streams on a device before a measurement, so
    /// the figure counts the work the rank issued and nothing it already
    /// freed. The worker supplies it: it holds the streams.
    synchronize: Arc<dyn Fn(DeviceId) -> Result<()> + Send + Sync>,
}

impl DeviceBudget {
    pub(crate) fn new(
        synchronize: Arc<dyn Fn(DeviceId) -> Result<()> + Send + Sync>,
        retained: Arc<Retained>,
        unattributed: BTreeMap<DeviceId, u64>,
    ) -> Result<Self> {
        Ok(Self {
            nvml: Nvml::init()?,
            retained,
            unattributed,
            synchronize,
        })
    }

    /// Device storage this process holds on `device` after PyTorch returns
    /// its unused cached blocks: NVML's figure with the plugins' buffers it
    /// attributes to no process, without the storages the server retains,
    /// among them the segments this process created and keeps the creating
    /// handles of, which NVML attributes to it ([`platform::created_bytes`]).
    ///
    /// Fails when NVML attributes no usage on the device to this process, as
    /// inside a PID namespace NVML does not report.
    fn own_bytes(&self, device: DeviceId) -> Result<u64> {
        (self.synchronize)(device)?;
        let process = self
            .nvml
            .process_storage(device)
            .map_err(|error| match error {
                platform::Error::ProcessNotFound { .. } => Error::Unsupported(format!(
                    "NVML reports no device storage for this process on {device}"
                )),
                error => Error::Cuda(error),
            })?;
        let created = platform::created_bytes(device);
        Ok(process.saturating_sub(created) + self.unattributed.get(&device).copied().unwrap_or(0))
    }

    /// Device storage this process holds on `device`, with the storages its
    /// scope retains there.
    #[allow(
        dead_code,
        reason = "the warm booking reads `own_bytes`; this states the whole"
    )]
    pub(crate) fn process_bytes(&self, device: DeviceId) -> Result<u64> {
        Ok(self.own_bytes(device)? + self.retained.scope_bytes(device)?.held)
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
            quota: super::quota(total, fraction),
            graph,
            process: own.saturating_sub(graph) + pending,
            profiling: 0,
        })
    }
}

/// What a warm rank books on one device: the items of its process the
/// Storage Server cannot see.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceBooking {
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
    /// The device storage this rank's deployment may hold on `device`: the
    /// budget its metrics report states, and what every booking is checked
    /// against.
    pub fn budget(&self) -> u64 {
        self.quota
    }

    /// The bytes this rank books for its graph plan's pools, which the ledger
    /// releases while the rank sleeps and books again when it wakes.
    pub fn graph_bytes(&self) -> u64 {
        self.graph
    }

    /// The bytes the rank's process holds at the measurement, with a profiling
    /// session's storage included.
    pub fn process_bytes(&self) -> u64 {
        self.process + self.profiling
    }

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
