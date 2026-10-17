//! Shared device budgets and allocation pools for captured numerical calls.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use uniserve_cuda::DeviceId;

use indexmap::IndexMap;
use tvm_ffi::Any;

use crate::ffi::{Function, Handle};
use crate::{Error, Result};

/// One allocator pool of the numerical library (`uniserve.cuda_graph.pool`)
/// on a device: the library's pool handle and its allocator id.
pub struct GraphPool {
    pub device: DeviceId,
    pub id: (u64, u64),
    pub value: Handle,
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

    /// Serving growth belongs to the worker grant, not the sealed graph pools.
    pub(crate) fn seal(&mut self) {
        self.outside.clear();
    }

    pub(crate) fn close(&mut self) -> Vec<Vec<Arc<GraphPool>>> {
        self.outside.clear();
        self.owners.drain(..).map(|(_, pools)| pools).collect()
    }
}

/// The worker reserves the same graph share when planning and allocating.
pub(crate) fn graph_storage_budget_bytes(total_device_bytes: i64) -> u64 {
    (total_device_bytes.max(0) as f64 * 0.10) as u64
}

/// Reserve the allocator pools of graph owner `key` on `devices`.
///
/// With `share`, the owner borrows that registered owner's pools (owners on
/// one stream run serially, so they can share intermediates); devices it
/// has no pool on get a new one. `create` is `uniserve.cuda_graph.pool`,
/// returning `[pool, id_a, id_b, total_bytes]`. The owner is registered only
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
        let value: tvm_ffi::Array<Any> =
            create.call_as(&[(&Any::from(i64::from(device.0))).into()])?;
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
        let id = (number(1)? as u64, number(2)? as u64);
        defaults.push((device, graph_storage_budget_bytes(number(3)?)));
        pools.push(Arc::new(GraphPool {
            device,
            id,
            value: Handle::new(item(0)?)?,
        }));
    }
    storage.commit(key, pools.clone(), defaults);
    Ok(pools)
}
