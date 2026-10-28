//! What a KV cache may hold of its GPU, from the GPU's memory ledger.
//!
//! The host's Storage Server keeps one ledger per GPU for the deployment
//! (`storage::GpuLedger`): the quota `⌊total × fraction⌋` of the whole GPU,
//! and every item the deployment's worlds and the Transfer Server hold on it
//! (weights, each KV cache segment, the other exclusive storages, graph
//! pools, process memory, staging, the Transfer Server's context), each once
//! however many worlds map it. Every worker reports its cache's GPU's ledger
//! with its segments (`uniserve.cache_memory`), and the limits of the two
//! periods in which two worlds hold memory on the GPU follow from it
//! (`specs/online-operations.md`, §6.4):
//!
//! - **Preparation** ([`preparing`]): the serving world's KV cache may keep
//!   the quota less everything else the ledger books (its own weights, graph
//!   pool and fixed part, the Transfer Server's context and staging; the
//!   context counted even while the Transfer Server holds none, since it
//!   books the context only from its first transfer on the GPU) and
//!   less what its successor adds: a graph pool of its own, a fixed part as
//!   large as the serving world's (process memory and exclusive storages,
//!   for the same configuration), its head segment, and no weights or KV
//!   cache of the keys it maps read-only. The serving world gives back the
//!   tail segments past that limit; a limit below the segments it cannot
//!   give back (its first, which holds the sentinel and the warmup scratch)
//!   makes the change infeasible.
//! - **Trial and after** ([`growable`]): a world's cache creates its
//!   deferred segments as the ledger's room holds them. The room is the
//!   quota less everything booked, so in the trial it is less what the
//!   previous world still holds by its retention mode (a parked world its
//!   graph pool, fixed part and the weights the successor does not map; a
//!   world kept as its weights those weights alone, once its processes
//!   ended), and after the commit the whole quota less the world's own
//!   items. What the previous world releases is thus transferred on the
//!   ledger to the successor, which creates segments into it at once.

use serde::{Deserialize, Serialize};
use storage::{GpuLedger, HostMemory, Item, LedgerEntry, Scope};

/// What `uniserve.cache_memory` reports of one worker.
#[derive(Debug, Clone, Deserialize)]
pub(super) struct CacheMemory {
    /// The segments of each cache plane, in unit order; empty when the
    /// planes are not split.
    #[serde(default)]
    pub(super) segments: Vec<ipc::CacheSegment>,
    /// Segments the worker holds: the first given back.
    #[serde(default)]
    pub(super) held: u32,
    /// Device bytes of one unit across the worker's planes.
    #[serde(default)]
    pub(super) unit_bytes: u64,
    /// Device bytes of the head segment a successor creates beside this
    /// cache's segments.
    #[serde(default)]
    pub(super) head_bytes: u64,
    /// The units a configured token capacity holds; a successor's head and
    /// the segments it inherits fit in them together. `None` for a cache
    /// sized by memory.
    #[serde(default)]
    pub(super) capacity_units: Option<u64>,
    /// Free bytes of the worker's device, which bound what the cache
    /// creates beside processes outside the deployment.
    #[serde(default)]
    pub(super) free_bytes: u64,
    /// The share of the device a worker's capacity plan keeps for graphs
    /// and activations: the graph pool a successor's warmup captures in.
    #[serde(default)]
    pub(super) graph_share_bytes: u64,
    /// The worker's host.
    #[serde(default)]
    pub(super) host: String,
    /// The worker's scope and world, which name its entries in the ledger.
    #[serde(default)]
    pub(super) scope: Option<Scope>,
    #[serde(default)]
    pub(super) world: Option<u64>,
    /// The deployment's ledger of the cache's GPU; `None` for a worker that
    /// keeps its memory itself, which no other world shares.
    #[serde(default)]
    pub(super) ledger: Option<GpuLedger>,
    /// What the worker's host admits of new host memory.
    #[serde(default)]
    pub(super) host_memory: Option<HostMemory>,
}

impl CacheMemory {
    /// The first unit of segment `index`.
    pub(super) fn first_unit(&self, index: u32) -> u32 {
        self.segments
            .iter()
            .take(index as usize)
            .map(|segment| segment.units)
            .sum()
    }

    /// Whether `entry` is the worker's own: of its scope and world alone.
    fn owns(&self, entry: &LedgerEntry) -> bool {
        entry.scope == self.scope && self.world.is_some_and(|world| entry.worlds == [world])
    }

    /// The device bytes of segment `index`: what the ledger books for it,
    /// or its units' bytes for a segment not created.
    pub(super) fn segment_bytes(&self, index: usize) -> u64 {
        let segment = &self.segments[index];
        let booked = self.ledger.as_ref().map_or(0, |ledger| {
            ledger.bytes(|entry| {
                entry.item == Item::CacheSegment
                    && entry.scope == self.scope
                    && entry.segment
                        == Some(storage::Segment {
                            world: segment.world,
                            ordinal: segment.ordinal,
                        })
            })
        });
        if booked > 0 {
            booked
        } else {
            u64::from(segment.units) * self.unit_bytes
        }
    }
}

/// One device's memory in the preparing phase: what each world and the
/// Transfer Server hold beside the serving world's KV cache, and the limit
/// that leaves it. Reported per GPU by a world change's plan and refusal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct Preparing {
    pub(super) host: String,
    /// The GPU, as `nvidia-smi` names it.
    pub(super) gpu: String,
    pub(super) scope: String,
    pub(super) world: u64,
    pub(super) total: u64,
    pub(super) quota: u64,
    /// What the serving world holds beside its KV cache, by item.
    pub(super) serving: Holdings,
    /// The serving world's KV cache segments of this worker.
    pub(super) serving_kv_cache: u64,
    /// What the successor adds beside what it maps read-only.
    pub(super) successor: Successor,
    /// The Transfer Server's context and staging.
    pub(super) transfer: u64,
    /// What else the ledger books on the GPU: other workers of the
    /// deployment on it.
    pub(super) others: u64,
    /// The serving world's KV cache limit: the quota less every other item,
    /// and no more than the device's free storage with that cache leaves the
    /// successor (negative when the other items overrun either).
    pub(super) kv_limit: i64,
    /// The device's free storage when planned: what processes outside the
    /// deployment, which no ledger books, leave the deployment's worlds.
    pub(super) free: u64,
    /// The bytes of the segments the serving world cannot give back (its
    /// first segment, and segments it inherited and does not own yet).
    pub(super) kv_floor: u64,
    /// The segments the serving world keeps of those it holds, and their
    /// bytes.
    pub(super) segments_kept: u32,
    pub(super) segments_held: u32,
    pub(super) kv_kept: u64,
    /// Whether the limit holds the segments the serving world cannot give
    /// back.
    pub(super) feasible: bool,
    /// Every entry of the GPU's ledger.
    pub(super) entries: Vec<String>,
    /// What the worker's host admits of new host memory.
    pub(super) host_memory: Option<HostMemory>,
}

/// A world's items beside its KV cache.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub(super) struct Holdings {
    pub(super) weights: u64,
    pub(super) graph_pool: u64,
    pub(super) process: u64,
    pub(super) exclusive: u64,
    pub(super) staging: u64,
}

/// What a successor world adds on the device in the preparing phase.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub(super) struct Successor {
    /// Weights of keys the serving world does not hold: none, since a world
    /// change maps every weight of the same key.
    pub(super) weights: u64,
    /// The graph pool its warmup captures in.
    pub(super) graph_pool: u64,
    /// Its fixed part: process memory and exclusive storages as large as
    /// the serving world's.
    pub(super) process: u64,
    pub(super) exclusive: u64,
    /// Its head segment.
    pub(super) head_segment: u64,
    /// What it prepares for the requests in flight: nothing, since it maps
    /// their KV cache read-only.
    pub(super) in_flight: u64,
}

impl Successor {
    fn total(&self) -> u64 {
        self.weights
            + self.graph_pool
            + self.process
            + self.exclusive
            + self.head_segment
            + self.in_flight
    }
}

/// The serving world's KV cache limit on the device of `report` in the
/// preparing phase, and the segments it keeps within it, of which `floor`
/// it cannot give back. Fails for a worker that reports no ledger or no
/// quota.
pub(super) fn preparing(report: &CacheMemory, floor: u32) -> Result<Preparing, String> {
    let (Some(ledger), Some(scope), Some(world)) = (&report.ledger, &report.scope, report.world)
    else {
        return Err("a worker that holds a split KV cache reports no memory ledger".to_owned());
    };
    let quota = ledger.quota.ok_or_else(|| {
        format!(
            "no world states a quota in the memory ledger of GPU {}",
            storage::protocol::gpu_name(&ledger.gpu)
        )
    })?;
    let own = |item: Item| ledger.bytes(|entry| entry.item == item && report.owns(entry));
    let serving = Holdings {
        weights: ledger.bytes(|entry| entry.item == Item::Weights && entry.worlds.contains(&world)),
        graph_pool: own(Item::GraphPool),
        process: own(Item::Process),
        exclusive: own(Item::Exclusive),
        staging: own(Item::Staging),
    };
    let serving_kv_cache = own(Item::CacheSegment);
    // The Transfer Server books its context only while it holds it, and the
    // change's transfers need it on the GPU: a context not booked yet counts
    // beside what the ledger books.
    let unbooked_context = ledger
        .transfer_context
        .saturating_sub(ledger.bytes(|entry| entry.item == Item::TransferContext));
    let transfer =
        ledger.bytes(|entry| entry.scope.is_none() && entry.worlds.is_empty()) + unbooked_context;
    let others = (ledger.used() + unbooked_context).saturating_sub(
        serving.weights
            + serving.graph_pool
            + serving.process
            + serving.exclusive
            + serving.staging
            + serving_kv_cache
            + transfer,
    );
    let successor = Successor {
        weights: 0,
        graph_pool: report.graph_share_bytes,
        process: serving.process,
        exclusive: serving.exclusive,
        head_segment: report.head_bytes,
        in_flight: 0,
    };
    let beside = ledger.used() + unbooked_context - serving_kv_cache + successor.total();
    // The successor's grant is never more than the device has free, so its
    // additions and the Transfer Server's context not created yet must fit
    // in the free storage and the tail the serving world gives back: a
    // process outside the deployment may hold more of the device than the
    // quota leaves it.
    let within_free = report.free_bytes as i64 + serving_kv_cache as i64
        - (successor.total() + unbooked_context) as i64;
    let kv_limit = (quota as i64 - beside as i64).min(within_free);

    // The largest prefix of the held segments within the limit, at least
    // the floor.
    let mut kept = 0_u64;
    let mut kv_floor = 0_u64;
    let mut segments_kept = 0;
    for index in 0..report.held {
        let bytes = report.segment_bytes(index as usize);
        if index < floor {
            kv_floor += bytes;
        } else if (kept + bytes) as i64 > kv_limit {
            break;
        }
        kept += bytes;
        segments_kept = index + 1;
    }
    Ok(Preparing {
        host: report.host.clone(),
        gpu: storage::protocol::gpu_name(&ledger.gpu),
        scope: scope.to_string(),
        world,
        total: ledger.total,
        quota,
        serving,
        serving_kv_cache,
        successor,
        transfer,
        others,
        kv_limit,
        free: report.free_bytes,
        kv_floor,
        segments_kept,
        segments_held: report.held,
        kv_kept: kept,
        feasible: kv_floor as i64 <= kv_limit,
        entries: ledger.entries.iter().map(LedgerEntry::describe).collect(),
        host_memory: report.host_memory,
    })
}

/// Why a world change does not fit beside the serving world: for every
/// device, the bytes of each item each world and the Transfer Server hold
/// and the limit they leave the serving world's KV cache, and for every
/// host what it admits of new host memory.
pub(super) fn refusal(plans: &[Preparing]) -> String {
    let mut lines = vec![
        "a successor world does not fit beside the serving world: on a GPU the serving \
         world's KV cache limit does not hold the segments it keeps in any case"
            .to_owned(),
    ];
    for plan in plans {
        let serving = &plan.serving;
        let successor = &plan.successor;
        lines.push(format!(
            "host {} GPU {} ({} of world {}): {} of a quota of {} bytes and {} bytes free; KV \
             cache limit {} bytes, {} of which it cannot give back{}",
            plan.host,
            plan.gpu,
            plan.scope,
            plan.world,
            if plan.feasible {
                "fits"
            } else {
                "does not fit"
            },
            plan.quota,
            plan.free,
            plan.kv_limit,
            plan.kv_floor,
            if plan.feasible { "" } else { " (infeasible)" },
        ));
        lines.push(format!(
            "  serving world: {} weights, {} graph pool, {} process, {} exclusive storages, \
             {} staging, {} KV cache",
            serving.weights,
            serving.graph_pool,
            serving.process,
            serving.exclusive,
            serving.staging,
            plan.serving_kv_cache,
        ));
        lines.push(format!(
            "  successor world: {} weights it does not map, {} graph pool, {} process, {} \
             exclusive storages, {} head segment, {} for the requests in flight",
            successor.weights,
            successor.graph_pool,
            successor.process,
            successor.exclusive,
            successor.head_segment,
            successor.in_flight,
        ));
        lines.push(format!(
            "  Transfer Server: {} bytes; other workers: {} bytes",
            plan.transfer, plan.others
        ));
        for entry in &plan.entries {
            lines.push(format!("  ledger: {entry}"));
        }
    }
    let mut hosts: Vec<(&str, &HostMemory)> = plans
        .iter()
        .filter_map(|plan| Some((plan.host.as_str(), plan.host_memory.as_ref()?)))
        .collect();
    hosts.sort_by_key(|(host, _)| *host);
    hosts.dedup_by_key(|(host, _)| *host);
    for (host, memory) in hosts {
        lines.push(format!(
            "host {host}: admits {} bytes of new host memory ({} available, less a {}-byte \
             reserve and {} bytes the page pools have yet to allocate); the successor needs \
             none, sharing the serving world's page pools",
            memory.room(),
            memory.available,
            memory.reserve,
            memory.pools_unallocated,
        ));
    }
    lines.join("\n")
}

/// How many of the segments of `report` past those it holds fit the
/// ledger's room now, and the device's free memory; with `released`, also
/// what the processes of the other world book, which end as they exit (a
/// previous world kept as its weights).
pub(super) fn growable(report: &CacheMemory, released: bool) -> u32 {
    let (Some(ledger), Some(world)) = (&report.ledger, report.world) else {
        return 0;
    };
    let Some(quota) = ledger.quota else {
        return 0;
    };
    let ending = if released {
        ledger.bytes(|entry| {
            entry.item.booked() && !entry.worlds.is_empty() && !entry.worlds.contains(&world)
        })
    } else {
        0
    };
    let mut room = (quota + ending)
        .saturating_sub(ledger.used())
        .min(report.free_bytes + ending);
    let mut fit = 0;
    for index in report.held as usize..report.segments.len() {
        let bytes = report.segment_bytes(index);
        if bytes > room {
            break;
        }
        room -= bytes;
        fit += 1;
    }
    fit
}
