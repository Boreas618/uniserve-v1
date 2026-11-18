//! The experts' order (`moe_config.expert_order`) changed within the serving epoch, through
//! retained weight copy manifests.

use std::collections::{BTreeMap, BTreeSet};

use proto::request::wire;

use crate::engine::client::EngineClient;
use crate::leader::{Replica, UpdateError};

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

/// One rank's prepared allocation for a new expert order: the orders it took, the GPU it
/// allocated on and the bytes it reserves there (the `PrepareExperts` report).
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

/// Moves `replica`'s routed experts, on the engine `client` serves with, from the order
/// `current` states to the order `target` states (`moe_config.expert_order`) within the serving
/// epoch: every layer whose order changes is prepared on the copy stream beside the batches and
/// committed at one computation boundary, the numerical shapes and the epoch staying. A layer
/// `target` no longer names returns to the checkpoint's order. The caller holds the replica while
/// no engine replaces it.
///
/// Copies run alongside inference; only the commit drains computations, remaps their fixed weight
/// addresses and changes the routing tensors' contents.
///
/// # Errors
///
/// Refuses an order a rank cannot take (a layer that is not a routed expert layer, an order
/// that is not a permutation of the layer's experts), after which the routing is unchanged;
/// fails when the commit or the cache's growth back fails.
pub(crate) async fn reorder(
    replica: &Replica,
    client: &EngineClient,
    current: &BTreeMap<String, Vec<u32>>,
    target: &BTreeMap<String, Vec<u32>>,
) -> Result<(), UpdateError> {
    let mut orders: BTreeMap<String, Vec<u32>> = target
        .iter()
        .filter(|(module, order)| current.get(*module) != Some(*order))
        .map(|(module, order)| (module.clone(), order.clone()))
        .collect();
    // A layer the target no longer names takes the checkpoint's order back: its experts in
    // their own order, as many as the layer's current order holds.
    for (module, order) in current {
        if !target.contains_key(module) {
            let experts = u32::try_from(order.len()).unwrap_or(u32::MAX);
            orders.insert(module.clone(), (0..experts).collect());
        }
    }
    if orders.is_empty() {
        return Ok(());
    }
    if let Err(error) = prepare(client, &orders).await {
        // A rank may have prepared before another rejected its order. Abort releases only
        // incoming allocations; live routing is intact.
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

    // Once a rank starts remapping, a relaunch and a standby must load this order: the launch
    // they start from states it before the commit.
    if let Some(served) = service::lock(&replica.served).as_mut() {
        served.engine_config.moe.expert_order = target.clone();
    }
    // The commit states the whole order every rank routes by from then on: the target's
    // layers, and the checkpoint's order of the layers it no longer names.
    let mut committed = target.clone();
    committed.extend(
        orders
            .iter()
            .filter(|(module, _)| !target.contains_key(*module))
            .map(|(module, order)| (module.clone(), order.clone())),
    );
    client
        .commit_experts(wire_orders(&committed))
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
    tracing::info!(
        replica = replica.ordinal,
        layers = orders.len(),
        "the experts' new order is committed"
    );
    Ok(())
}

/// Prepares `orders` on every rank of `client`'s engine: each rank allocates the copies of its
/// new experts beside the serving ones, the KV cache makes room for them where a GPU's quota
/// lacks it, and the copies run; answers once every rank's copies completed.
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
