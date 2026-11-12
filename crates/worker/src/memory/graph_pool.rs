//! The graph pool of a device: one region per runner at a fixed address, and the scratch pool
//! beside it.
//!
//! A captured graph holds the addresses of the tensors its forward allocated, so the memory
//! behind a runner's graphs stays at its address for as long as the graphs exist. A
//! [`GraphPool`] maps one region per runner from the device's graph pool when it opens, at the
//! bytes the runner's profile run stated ([`Graphs::peak`](crate::Graphs::peak)), and lends
//! each to [`Graphs::new`](crate::Graphs::new) as a `platform::DeviceRange`. The runner's
//! `backend::Context` places every forward's tensors inside its region; the pool keeps no
//! account of them.
//!
//! The scratch pool holds what no serving forward reads: the regions the shadow outputs of
//! paired forwards are written in ([`GraphPool::shadow`]) and the state snapshots of a model
//! that keeps recurrent state ([`GraphPool::scratch`]).
//!
//! ## The quota
//!
//! Both pools map through one `platform::Allocator`, which states each pool's mapped bytes to
//! its `platform::Budget` before it maps. The budget here reserves the sum with the Storage
//! Server as the rank's [`Item::GraphPool`] in the GPU's quota, so a mapping the quota cannot
//! hold is refused before any storage is created, and the epochs that share the GPU see what
//! the rank maps.
//!
//! ## Sleep and wake
//!
//! [`GraphPool::suspend`] gives the pools' storage back to the device, keeping its contents on
//! the host and its addresses reserved, and the reservation in the quota drops with it.
//! [`GraphPool::restore`] maps storage at the same addresses again. The executor then renews
//! every runner's graphs ([`Graphs::renew`](crate::Graphs::renew)) before it replays one.

use std::sync::{Arc, Mutex};

use platform::{Allocator, Budget, Buffer, DeviceId, DeviceRange, PoolKind};
use proto::request::wire::{DeviceMemory, PoolMemory};
use storage::Item;

use crate::memory::client::{StorageClient, quota_limit};
use crate::runner::RunnerId;
use crate::{Error, Result, lock};

/// The pool kinds of a device's allocator with the names the usage report gives them.
const POOLS: [(PoolKind, &str); 3] = [
    (PoolKind::Graph, "graph"),
    (PoolKind::Scratch, "scratch"),
    (PoolKind::Resident, "resident"),
];

/// The reservation of a device's pools in its GPU's quota at the Storage Server.
///
/// The allocator states one pool's mapped bytes at a time; the quota holds one item for the
/// rank's pools. This keeps each pool's last stated bytes and reserves their sum, so the item
/// is what the allocator maps whenever a mapping succeeded.
struct Reservation {
    client: Arc<StorageClient>,
    /// The GPU's total bytes and the epoch's limit of them, which every reservation restates.
    total: u64,
    limit: u64,
    /// The bytes each pool last stated, in the order of [`POOLS`].
    mapped: Mutex<[u64; POOLS.len()]>,
}

impl Budget for Reservation {
    fn book(&self, device: DeviceId, pool: PoolKind, bytes: u64) -> platform::Result<()> {
        let mut mapped = lock(&self.mapped);

        let mut stated = *mapped;
        for ((kind, _), held) in POOLS.iter().zip(stated.iter_mut()) {
            if *kind == pool {
                *held = bytes;
            }
        }

        // A refused reservation leaves the quota and the kept bytes as they were: the
        // allocator then maps nothing.
        let sum = stated.iter().sum();
        self.client
            .reserve(device, self.total, self.limit, vec![(Item::GraphPool, sum)])
            .map_err(|error| platform::Error::InvalidArgument(error.to_string()))?;

        *mapped = stated;
        Ok(())
    }
}

/// The regions of a device's runners and its scratch pool; see the module documentation.
pub struct GraphPool {
    allocator: Allocator,
    /// The epoch's limit of the GPU, which the usage report states.
    limit: u64,
    /// Each runner's region, mapped for the life of the pool.
    regions: Vec<(RunnerId, Buffer)>,
}

impl GraphPool {
    /// Maps one region per runner on `device` and reserves the addresses of its scratch pool.
    ///
    /// `regions` are each runner's bytes, the peak of its profile run; `scratch` is the bytes
    /// the scratch pool is divided ([`pool_bytes`](super::division::pool_bytes)), and zero
    /// reserves none, for a rank that never shadows and keeps no state snapshot. `fraction` is the
    /// deployment's `mem_fraction_static`, whose limit the pools' reservation restates.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] when a runner is named twice; the driver's error when a region
    /// cannot be mapped, which carries the quota's refusal, naming every reservation, when the
    /// regions do not fit the GPU's quota. Nothing stays mapped after an error.
    pub fn open(
        client: Arc<StorageClient>,
        device: DeviceId,
        fraction: f64,
        regions: &[(RunnerId, u64)],
        scratch: u64,
    ) -> Result<Self> {
        let total = device.memory_info()?.total;
        let limit = quota_limit(total, fraction);

        let reservation = Reservation {
            client,
            total,
            limit,
            mapped: Mutex::new([0; POOLS.len()]),
        };
        let mut allocator = Allocator::new(&platform::Device::new(device)?, Arc::new(reservation))?;

        // Addresses cost nothing, so each pool reserves as many as the quota could ever let
        // it map: whatever granules the driver rounds a region up to, its address range holds
        // it, and the quota is what refuses a mapping.
        let addresses = usize::try_from(limit.max(1))
            .map_err(|_| Error::Resource("the GPU's quota is larger than the address space"))?;
        allocator.reserve(PoolKind::Graph, addresses)?;
        if scratch > 0 {
            allocator.reserve(PoolKind::Scratch, addresses)?;
        }

        let mut mapped: Vec<(RunnerId, Buffer)> = Vec::with_capacity(regions.len());
        for &(runner, bytes) in regions {
            if mapped.iter().any(|(held, _)| *held == runner) {
                return Err(Error::Invalid(format!(
                    "runner {} is given two regions of the graph pool",
                    runner.0
                )));
            }

            let bytes = usize::try_from(bytes).map_err(|_| {
                Error::Resource("a runner's region is larger than the address space")
            })?;
            mapped.push((runner, allocator.allocate(PoolKind::Graph, bytes)?));
        }

        Ok(Self {
            allocator,
            limit,
            regions: mapped,
        })
    }

    /// The region of `runner`: what [`Graphs::new`](crate::Graphs::new) takes. It is at least
    /// the bytes the runner was opened with, and stays at its address until the pool drops.
    ///
    /// The graphs made over the region are dropped before the pool is, and replay nothing
    /// between [`GraphPool::suspend`] and the renewal that follows [`GraphPool::restore`].
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a runner the pool was not opened with.
    pub fn region(&self, runner: RunnerId) -> Result<DeviceRange> {
        self.regions
            .iter()
            .find(|(held, _)| *held == runner)
            .map(|(_, buffer)| DeviceRange::from(buffer))
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "the graph pool holds no region for runner {}",
                    runner.0
                ))
            })
    }

    /// A region of `bytes` of the scratch pool for the shadow outputs of one runner's paired
    /// forwards: what [`Graphs::set_shadow`](crate::Graphs::set_shadow) takes.
    ///
    /// # Errors
    ///
    /// When the pool was opened without scratch, its addresses are exhausted, or the quota
    /// refuses the mapping.
    pub fn shadow(&mut self, bytes: u64) -> Result<ops::Twins> {
        Ok(ops::Twins::new(&mut self.allocator, bytes)?)
    }

    /// `bytes` of the scratch pool at an address the pool keeps: the state snapshots a model
    /// with recurrent state copies units through. The buffer is valid until the pool drops.
    ///
    /// # Errors
    ///
    /// As [`GraphPool::shadow`].
    pub fn scratch(&mut self, bytes: u64) -> Result<Buffer> {
        let bytes = usize::try_from(bytes)
            .map_err(|_| Error::Resource("a scratch buffer is larger than the address space"))?;

        Ok(self.allocator.allocate(PoolKind::Scratch, bytes)?)
    }

    /// The bytes the pools map now: what the rank's process holds through them, which a
    /// measurement of the process leaves out
    /// ([`process_bytes`](super::division::process_bytes)).
    pub fn mapped(&self) -> u64 {
        POOLS
            .iter()
            .map(|(pool, _)| self.allocator.mapped_bytes(*pool) as u64)
            .sum()
    }

    /// What the rank reports of the device's allocator: the limit its mappings are reserved
    /// against and the bytes each pool maps now.
    pub fn usage(&self) -> DeviceMemory {
        DeviceMemory {
            device: self.allocator.device().0 as u32,
            budget: self.limit,
            pools: POOLS
                .iter()
                .map(|(pool, name)| PoolMemory {
                    pool: (*name).to_owned(),
                    mapped: self.allocator.mapped_bytes(*pool) as u64,
                })
                .collect(),
        }
    }

    /// Gives the pools' storage back to the device for a sleeping rank: every region's bytes
    /// are kept in page-locked host memory, its addresses stay reserved, and the pools'
    /// reservation in the quota drops to nothing.
    ///
    /// The caller has waited for every stream that addresses the pools, and nothing reads or
    /// writes them until [`GraphPool::restore`] returned.
    ///
    /// # Errors
    ///
    /// When a copy or an unmapping fails; what was suspended before the failure stays
    /// suspended and [`GraphPool::restore`] brings it back.
    pub fn suspend(&mut self) -> Result<()> {
        Ok(self.allocator.suspend()?)
    }

    /// Maps storage at every region's address again, with the bytes it held, and reserves the
    /// pools in the quota again. Restoring pools that are not suspended does nothing.
    ///
    /// The executor renews every runner's graphs afterwards
    /// ([`Graphs::renew`](crate::Graphs::renew)): a replay needs its executables instantiated
    /// again once its region was mapped again.
    ///
    /// # Errors
    ///
    /// When the driver or the quota refuses a mapping: every pool then stays suspended, and
    /// the restore may be tried again.
    pub fn restore(&mut self) -> Result<()> {
        Ok(self.allocator.restore()?)
    }
}
