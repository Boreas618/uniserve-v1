//! Same-epoch expert redistribution through retained weight copy manifests.

use std::collections::{BTreeMap, BTreeSet};

use proto::request::wire;
use serde::{Deserialize, Serialize};

/// The order of every named layer's experts, as the request interface states it.
fn wire_orders(
    orders: &BTreeMap<String, Vec<u32>>,
) -> std::collections::HashMap<String, wire::ExpertOrder> {
    orders
        .iter()
        .map(|(module, experts)| {
            (
                module.clone(),
                wire::ExpertOrder {
                    experts: experts.clone(),
                },
            )
        })
        .collect()
}

use crate::engine::client::EngineClient;
use crate::leader::{Leader, Replica, UpdateError};

/// Global physical-slot order of logical expert IDs, by model module path.
#[derive(Debug, Clone, Deserialize)]
pub struct Rebalance {
    pub order: BTreeMap<String, Vec<u32>>,
    /// Omission updates every independent replica, one at a time.
    pub replica: Option<usize>,
}

/// One rank's prepared allocation for a new expert order: the orders it took, the GPU it
/// allocated on and the bytes it reserves there (the `PrepareExperts` report).
#[derive(Serialize)]
struct Allocation {
    orders: BTreeMap<String, Vec<u32>>,
    gpu: storage::manifest::Gpu,
    bytes: u64,
}

impl TryFrom<wire::ExpertAllocation> for Allocation {
    type Error = String;

    fn try_from(allocation: wire::ExpertAllocation) -> Result<Self, Self::Error> {
        let gpu = allocation.gpu.as_slice().try_into().map_err(|_| {
            format!(
                "a rank allocated on a GPU named by {} bytes, not a UUID's 16",
                allocation.gpu.len()
            )
        })?;
        Ok(Self {
            orders: allocation
                .order
                .into_iter()
                .map(|(module, order)| (module, order.experts))
                .collect(),
            gpu,
            bytes: allocation.bytes,
        })
    }
}

impl Leader {
    /// Redistribute experts without changing numerical shapes or epoch identity.
    ///
    /// Copies run alongside inference. Only publication drains computations,
    /// remaps their fixed weight addresses and changes routing tensor contents.
    pub async fn rebalance_experts(
        &self,
        request: Rebalance,
    ) -> Result<serde_json::Value, UpdateError> {
        let _deployment = self.replicas.replacing.lock().await;
        let replicas = self.replicas();
        if let Some(index) = request.replica
            && !replicas.iter().any(|replica| replica.ordinal == index)
        {
            return Err(UpdateError::Refused(format!(
                "replica {index} does not exist"
            )));
        }
        let mut results = Vec::new();
        for replica in replicas {
            if request
                .replica
                .is_some_and(|index| index != replica.ordinal)
            {
                continue;
            }
            results.push(rebalance(&replica, &request.order).await?);
        }
        Ok(serde_json::json!({"replicas": results}))
    }
}

async fn rebalance(
    replica: &Replica,
    orders: &BTreeMap<String, Vec<u32>>,
) -> Result<serde_json::Value, UpdateError> {
    let _replacing = replica.replacing.lock().await;
    let client = replica.hold.client().ok_or(UpdateError::NoEngine)?;
    let prepared = prepare(&client, orders).await;
    let allocations = match prepared {
        Ok(allocations) => allocations,
        Err(error) => {
            // A rank may have prepared before another rejected its order.
            // Abort releases only incoming allocations; live routing is intact.
            if let Err(cleanup) = client.abort_experts().await {
                return Err(UpdateError::Failed(anyhow::anyhow!(
                    "{error}; releasing expert preparation failed: {cleanup}"
                )));
            }
            client
                .resize_cache(wire::CacheChange::Grow)
                .await
                .map_err(|cleanup| {
                    UpdateError::Failed(anyhow::anyhow!(
                        "{error}; restoring cache capacity failed: {cleanup}"
                    ))
                })?;
            return Err(UpdateError::Refused(error));
        }
    };

    let mut decided = service::lock(&replica.served)
        .clone()
        .ok_or(UpdateError::NoEngine)?;
    decided
        .expert_order
        .extend(orders.iter().map(|(name, ids)| {
            (
                name.clone(),
                wire::ExpertOrder {
                    experts: ids.clone(),
                },
            )
        }));
    // Once a rank starts remapping, recovery must load this distribution.
    // Daemon handoff and engine replacement share the locks held above.
    let committed = decided.expert_order.clone();
    *service::lock(&replica.served) = Some(decided);
    client
        .commit_experts(committed.into_iter().collect())
        .await
        .map_err(|error| {
            UpdateError::Failed(anyhow::anyhow!(
                "expert publication failed; engine recovery will load the committed order: {error}"
            ))
        })?;
    client
        .resize_cache(wire::CacheChange::Grow)
        .await
        .map_err(|error| {
            UpdateError::Failed(anyhow::anyhow!(
                "experts committed; restoring cache capacity failed: {error}"
            ))
        })?;
    Ok(serde_json::json!({
        "replica": replica.ordinal,
        "epoch": client.epoch(),
        "order": orders,
        "allocations": allocations,
    }))
}

async fn prepare(
    client: &EngineClient,
    orders: &BTreeMap<String, Vec<u32>>,
) -> Result<Vec<Allocation>, String> {
    let allocations = client
        .prepare_experts(wire_orders(orders))
        .await?
        .into_iter()
        .map(Allocation::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    let names: BTreeSet<_> = allocations
        .iter()
        .flat_map(|allocation| allocation.orders.keys())
        .collect();
    if let Some(unknown) = orders.keys().find(|name| !names.contains(name)) {
        return Err(format!("{unknown:?} is not a routed expert layer"));
    }
    let mut devices = BTreeMap::new();
    for allocation in &allocations {
        *devices.entry(allocation.gpu).or_default() += allocation.bytes;
    }
    client
        .reserve_storage(crate::engine::client::Room::Devices(devices))
        .await?;
    // The engine answers once every rank's copies completed.
    client.start_experts().await?;
    Ok(allocations)
}
