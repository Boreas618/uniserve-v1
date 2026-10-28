//! Same-world expert redistribution through retained weight copy manifests.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use proto::request::wire;
use serde::{Deserialize, Serialize};

use crate::engine::client::EngineClient;
use crate::leader::{Leader, Replica, UpdateError, locked};

/// Global physical-slot order of logical expert IDs, by model module path.
#[derive(Debug, Clone, Deserialize)]
pub struct Rebalance {
    pub order: BTreeMap<String, Vec<u32>>,
    /// Omission updates every independent replica, one at a time.
    pub replica: Option<usize>,
}

#[derive(Deserialize, Serialize)]
struct Planned {
    orders: BTreeMap<String, Vec<u32>>,
    gpu: storage::manifest::Gpu,
    bytes: u64,
}

impl Leader {
    /// Redistribute experts without changing numerical shapes or world identity.
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
    let plans = match prepared {
        Ok(plans) => plans,
        Err(error) => {
            // A rank may have prepared before another rejected its order.
            // Abort releases only incoming allocations; live routing is intact.
            if let Err(cleanup) = client.hook("uniserve.abort_experts", "{}").await {
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

    let mut decided = locked(&replica.served)
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
    *locked(&replica.served) = Some(decided);
    client.commit_experts(committed).await.map_err(|error| {
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
        "world": client.world(),
        "order": orders,
        "allocations": plans,
    }))
}

async fn prepare(
    client: &EngineClient,
    orders: &BTreeMap<String, Vec<u32>>,
) -> Result<Vec<Planned>, String> {
    let payload = serde_json::to_string(orders).map_err(|error| error.to_string())?;
    let plans = client
        .hook("uniserve.prepare_experts", &payload)
        .await?
        .iter()
        .map(|report| serde_json::from_str::<Planned>(report).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let names: BTreeSet<_> = plans.iter().flat_map(|plan| plan.orders.keys()).collect();
    if let Some(unknown) = orders.keys().find(|name| !names.contains(name)) {
        return Err(format!("{unknown:?} is not a routed expert layer"));
    }
    let mut devices = BTreeMap::new();
    for plan in &plans {
        *devices.entry(plan.gpu).or_default() += plan.bytes;
    }
    client.reserve_storage(0, &devices).await?;
    client.hook("uniserve.start_experts", "{}").await?;
    loop {
        let ready = client
            .hook("uniserve.experts_ready", "{}")
            .await?
            .iter()
            .map(|report| serde_json::from_str::<bool>(report).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        if ready.iter().all(|ready| *ready) {
            return Ok(plans);
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
