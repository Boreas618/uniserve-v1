//! Shared device budgets and allocation pools for captured numerical calls.

use cuda::DeviceId;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use indexmap::IndexMap;
use tvm_ffi::Any;

use super::storage::{PoolSlot, library_path};
use crate::ffi::{Function, Handle};
use crate::{Error, Result};

/// The addresses each pool reserves: more than any device's storage, so a
/// pool is bounded by its budget and the device, never by its reservation.
const POOL_ADDRESSES: usize = 1 << 38;

/// One allocator pool on a device: the device storage Rust owns
/// ([`PoolSlot`]), the numerical library's pool that allocates from it
/// (`uniserve.cuda_graph.pool`) and its allocator id.
pub(crate) struct GraphPool {
    pub(crate) device: DeviceId,
    pub(crate) id: (u64, u64),
    pub(crate) value: Handle,
    pub(crate) storage: PoolSlot,
}

impl GraphPool {
    /// A new pool on `device`: its device storage, and the library's pool
    /// allocating from it through `create` (`uniserve.cuda_graph.pool`).
    /// Returns the pool and the device's total storage.
    pub(crate) fn create(create: &Function, device: DeviceId) -> Result<(Self, i64)> {
        let storage = PoolSlot::reserve(device, POOL_ADDRESSES)?;
        let value: tvm_ffi::Array<Any> = create.call_as(&[
            (&Any::from(i64::from(device.0))).into(),
            (&Any::from(storage.slot() as i64)).into(),
            (&Any::from(tvm_ffi::String::from(library_path()?.as_str()))).into(),
        ])?;
        let item = |index: usize| {
            value
                .get(index)
                .map_err(|error| Error::numerical(create.name(), &error))
        };
        if value.len() != 4 {
            return Err(Error::Interface(format!(
                "{} returns [pool, id_a, id_b, total_bytes]",
                create.name()
            )));
        }
        let number = |index: usize| -> Result<i64> {
            i64::try_from(item(index)?).map_err(|error| Error::numerical(create.name(), &error))
        };
        let pool = Self {
            device,
            id: (number(1)? as u64, number(2)? as u64),
            value: Handle::new(item(0)?)?,
            storage,
        };
        Ok((pool, number(3)?))
    }

    /// A new pool on `device`, created by the numerical library's
    /// `uniserve.cuda_graph.pool`; see [`Self::create`].
    pub(crate) fn new(device: DeviceId) -> Result<Self> {
        let create = crate::ffi::Interpreter::import("uniserve.cuda_graph.pool")?;
        Ok(Self::create(&create, device)?.0)
    }

    /// Copies the pool's storage to the host and releases it on the device,
    /// keeping its addresses; see [`PoolSlot::suspend`].
    pub(crate) fn suspend(&self) -> Result<()> {
        self.storage.suspend()
    }

    /// Maps the pool's storage back with its contents.
    pub(crate) fn restore(&self) -> Result<()> {
        self.storage.restore()
    }
}

/// Pool sharing, startup residency and byte limits across numerical owners.
///
/// Capture catalogs remain with the numerical backend. It supplies allocator
/// snapshots and process usage at preparation boundaries; replay needs no
/// allocator inspection. Owners sharing pools must serialize graph replay.
pub(crate) struct GraphStorage {
    /// The pools retained for each owner's captures.
    owners: IndexMap<usize, Vec<Arc<GraphPool>>>,
    budgets: BTreeMap<DeviceId, u64>,
    outside: BTreeMap<DeviceId, i128>,
}

impl GraphStorage {
    pub(crate) fn new(budgets: BTreeMap<DeviceId, u64>) -> Self {
        Self {
            owners: IndexMap::new(),
            budgets,
            outside: BTreeMap::new(),
        }
    }

    /// Borrow shared pools before allocating the remaining devices.
    /// The caller commits only after all numerical allocations succeed.
    pub(crate) fn prepare(
        &self,
        owner: usize,
        share: Option<usize>,
    ) -> Result<Vec<Arc<GraphPool>>> {
        if self.owners.contains_key(&owner) {
            return Err(Error::State("graph storage owner is already registered"));
        }

        match share {
            Some(share) => self
                .owners
                .get(&share)
                .cloned()
                .ok_or(Error::State("shared graph storage owner is not registered")),
            None => Ok(Vec::new()),
        }
    }

    /// Install successfully allocated pools and defaults for unbudgeted devices.
    pub(crate) fn commit(
        &mut self,
        key: usize,
        pools: Vec<Arc<GraphPool>>,
        defaults: impl IntoIterator<Item = (DeviceId, u64)>,
    ) {
        for (device, amount) in defaults {
            self.budgets.entry(device).or_insert(amount);
        }
        self.owners.insert(key, pools);
    }

    /// Attribute a shared pool once, to its first remaining numerical owner.
    pub(crate) fn owner_bytes(
        &self,
        segments: &[(DeviceId, (u64, u64), u64)],
    ) -> BTreeMap<(usize, DeviceId), u64> {
        let mut owners = HashMap::new();
        for (&key, owner) in &self.owners {
            for pool in owner {
                owners.entry((pool.device, pool.id)).or_insert(key);
            }
        }

        let mut sizes = BTreeMap::new();
        for &(device, id, bytes) in segments {
            if let Some(&owner) = owners.get(&(device, id)) {
                *sizes.entry((owner, device)).or_default() += bytes;
            }
        }
        sizes
    }

    pub(crate) fn pool_bytes(
        &self,
        segments: &[(DeviceId, (u64, u64), u64)],
    ) -> BTreeMap<DeviceId, u64> {
        let mut sizes: BTreeMap<_, _> = self.budgets.keys().map(|&device| (device, 0)).collect();
        for ((_, device), used) in self.owner_bytes(segments) {
            *sizes.entry(device).or_default() += used;
        }
        sizes
    }

    /// Bind the process footprint outside graph pools at the start of capture.
    pub(crate) fn set_budget(&mut self, device: DeviceId, amount: u64, process: u64, pooled: u64) {
        self.outside
            .insert(device, i128::from(process) - i128::from(pooled));
        self.budgets.insert(device, amount);
    }

    /// Account for process growth while preparing, and only pools after seal.
    pub(crate) fn resident_bytes(
        &self,
        pools: &BTreeMap<DeviceId, u64>,
        mut process: impl FnMut(DeviceId) -> u64,
    ) -> BTreeMap<DeviceId, i128> {
        self.budgets
            .keys()
            .map(|&device| {
                let used = match self.outside.get(&device) {
                    Some(&outside) => i128::from(process(device)) - outside,
                    None => i128::from(pools.get(&device).copied().unwrap_or(0)),
                };
                (device, used)
            })
            .collect()
    }

    pub(crate) fn check(&self, residency: &BTreeMap<DeviceId, i128>) -> Result<()> {
        for (&device, &used) in residency {
            let budget = self.budgets[&device];
            if used > i128::from(budget) {
                return Err(Error::Invalid(format!(
                    "graph residency on {device} exceeds its byte budget ({used}>{budget})"
                )));
            }
        }
        Ok(())
    }

    /// Every pool of every owner, each once.
    pub(crate) fn pools(&self) -> Vec<Arc<GraphPool>> {
        let mut pools: Vec<Arc<GraphPool>> = Vec::new();
        for pool in self.owners.values().flatten() {
            if !pools.iter().any(|known| Arc::ptr_eq(known, pool)) {
                pools.push(Arc::clone(pool));
            }
        }
        pools
    }

    /// Serving growth belongs to the worker grant, not the sealed graph pools.
    pub(crate) fn seal(&mut self) {
        self.outside.clear();
    }

    pub(crate) fn close(&mut self) -> Vec<Vec<Arc<GraphPool>>> {
        self.outside.clear();
        self.owners.drain(..).map(|(_, pools)| pools).collect()
    }
}

/// The device share every rank reserves, whatever its graph policy, for
/// graph capture and the eager allocations warmup makes: the same when
/// planning and allocating. Warmup's residency is checked against it.
pub(crate) fn graph_storage_budget_bytes(total_device_bytes: i64) -> u64 {
    (total_device_bytes.max(0) as f64 * 0.10) as u64
}

/// Reserve the allocator pools of graph owner `key` on `devices`.
///
/// With `share`, the owner borrows that registered owner's pools (owners on
/// one stream run serially, so they can share intermediates); devices it
/// has no pool on get a new one. `create` is `uniserve.cuda_graph.pool`
/// ([`GraphPool::create`]). The owner is registered only
/// after every pool exists, so a failure leaves the storage unchanged.
pub(crate) fn reserve_pools(
    storage: &mut GraphStorage,
    create: &Function,
    key: usize,
    devices: &[DeviceId],
    share: Option<usize>,
) -> Result<Vec<Arc<GraphPool>>> {
    let shared = storage.prepare(key, share)?;
    let mut pools = Vec::new();
    let mut defaults = Vec::new();
    for &device in devices {
        if pools
            .iter()
            .any(|pool: &Arc<GraphPool>| pool.device == device)
        {
            continue;
        }
        if let Some(pool) = shared.iter().find(|pool| pool.device == device) {
            pools.push(Arc::clone(pool));
            continue;
        }
        let (pool, total) = GraphPool::create(create, device)?;
        defaults.push((device, graph_storage_budget_bytes(total)));
        pools.push(Arc::new(pool));
    }
    storage.commit(key, pools.clone(), defaults);
    Ok(pools)
}
