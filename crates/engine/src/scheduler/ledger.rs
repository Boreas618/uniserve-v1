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
//!   for the same configuration), its head segment, the weight chunks it
//!   creates (those of other regions, for a world of another parallel
//!   configuration, as the daemon's copy manifests state per GPU), and no
//!   weights or KV cache of the keys it maps read-only. The serving world
//!   gives back the tail segments past that limit; a limit below the
//!   segments it cannot give back (its first, which holds the sentinel and
//!   the warmup scratch) makes the change infeasible.
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
    /// Whether a rank of the worker's model captures graphs under a policy
    /// that captures; a successor of a model that serves every call eagerly
    /// takes no graph share.
    #[serde(default)]
    pub(super) capturable: bool,
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
    pub(super) successor: Additions,
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
    /// The bytes of the units past the kept segments that hold idle cached
    /// pages: what evicting the prefix cache's idle pages from the tail
    /// frees before the requests holding the rest of it end (step 1 of
    /// `specs/online-operations.md` §6.5).
    pub(super) idle_cached_tail: u64,
    /// The KV cache the requests in flight hold on the device, which a
    /// promotion that moves it to the successor through host memory copies
    /// back over the host link (step 3).
    pub(super) in_flight_kv: u64,
    /// How long such a promotion waits for that copy at the host link's
    /// measured bandwidth, in milliseconds; `None` before the Transfer
    /// Server measured the link.
    pub(super) promotion_stall_ms: Option<u64>,
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
pub(super) struct Additions {
    /// Weights of keys the serving world does not hold: as many as it
    /// holds when the successor loads its weights from the model's files
    /// (their keys differ), the chunks it creates on the GPU when it holds
    /// other regions of the same weights (another parallel configuration),
    /// none when it maps them all.
    pub(super) weights: u64,
    /// The graph pool its warmup captures in.
    pub(super) graph_pool: u64,
    /// Its fixed part: process memory and exclusive storages as large as
    /// the serving world's.
    pub(super) process: u64,
    pub(super) exclusive: u64,
    /// Its head segment.
    pub(super) head_segment: u64,
    /// What it prepares for the requests in flight: nothing when it maps
    /// their KV cache read-only, the units they hold now when it computes
    /// their state again in units of its own.
    pub(super) in_flight: u64,
}

impl Additions {
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
/// preparing phase beside a successor that adds `adds`, and the segments it
/// keeps within it, of which `floor` it cannot give back. The requests in
/// flight hold `in_flight_units`, which a successor that computes their
/// state again takes of its own. Fails for a worker that reports no ledger
/// or no quota.
pub(super) fn preparing(
    report: &CacheMemory,
    floor: u32,
    adds: &crate::Successor,
    in_flight_units: u32,
    in_flight_tokens: &[u32],
) -> Result<Preparing, String> {
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
    let target = adds.rank_budgets.get(&ledger.gpu);
    let quota = target.map_or(quota, |target| quota.min(target.quota));
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
    let successor = Additions {
        weights: if adds.loads_weights {
            serving.weights
        } else {
            target.map_or(0, |target| target.weights)
        },
        // A successor captures graphs only when its policy captures and its
        // model can be captured.
        graph_pool: if let Some(target) = target {
            target.graphs
        } else if adds.captures_graphs && report.capturable {
            report.graph_share_bytes
        } else {
            0
        },
        process: target.map_or(serving.process, |target| {
            target.process.max(serving.process)
        }),
        exclusive: target.map_or(serving.exclusive, |target| target.fixed),
        head_segment: target.map_or(report.head_bytes, |target| target.cache),
        in_flight: {
            let requests = if adds.recomputes_cache {
                target.map_or(u64::from(in_flight_units) * report.unit_bytes, |target| {
                    target
                        .in_flight
                        .unwrap_or_else(|| target.request_bytes(in_flight_tokens))
                })
            } else {
                0
            };
            target.map_or(requests, |target| {
                requests.max(target.cache_capacity.saturating_sub(target.cache))
            })
        },
    };
    let in_flight_kv = u64::from(in_flight_units) * report.unit_bytes;
    let promotion_stall_ms = ledger
        .host_link_bytes_per_second
        .filter(|&bandwidth| bandwidth > 0)
        .map(|bandwidth| in_flight_kv.saturating_mul(1000).div_ceil(bandwidth));
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
        // The caller counts the idle cached pages past the kept segments.
        idle_cached_tail: 0,
        in_flight_kv,
        promotion_stall_ms,
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

/// One device's room for the device storage a profiling session holds:
/// what its ledger and its free storage leave, and the tail segments the KV
/// cache gives back to make the rest. Reported per GPU by the change's plan
/// and refusal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct Room {
    pub(super) host: String,
    /// The GPU, as `nvidia-smi` names it.
    pub(super) gpu: String,
    pub(super) quota: u64,
    /// What the ledger books.
    pub(super) booked: u64,
    /// The device's free storage when planned.
    pub(super) free: u64,
    /// The device storage the session holds.
    pub(super) bytes: u64,
    /// The bytes of the segments the cache cannot give back.
    pub(super) kv_floor: u64,
    /// The segments the cache keeps of those it holds.
    pub(super) segments_kept: u32,
    pub(super) segments_held: u32,
    /// Whether giving back every segment past the floor makes the room.
    pub(super) feasible: bool,
    /// Every entry of the GPU's ledger.
    pub(super) entries: Vec<String>,
}

/// The room on the device of `report` for `bytes` of device storage beside
/// everything its ledger books, of whose segments the cache cannot give
/// back the first `floor`: the cache keeps all it holds when the ledger's
/// room and the device's free storage hold `bytes` already, and otherwise
/// the largest prefix whose given-back tail makes the rest (each segment
/// given back frees its bytes of both). Fails for a worker that reports no
/// ledger or no quota.
pub(super) fn room(report: &CacheMemory, bytes: u64, floor: u32) -> Result<Room, String> {
    let Some(ledger) = &report.ledger else {
        return Err("a worker that holds a split KV cache reports no storage ledger".to_owned());
    };
    let gpu = storage::protocol::gpu_name(&ledger.gpu);
    let quota = ledger
        .quota
        .ok_or_else(|| format!("no world states a quota in the storage ledger of GPU {gpu}"))?;
    // A process outside the deployment may hold more of the device than the
    // quota leaves it.
    let available = quota.saturating_sub(ledger.used()).min(report.free_bytes);
    let mut freed = 0;
    let mut kept = report.held;
    while available + freed < bytes && kept > floor {
        kept -= 1;
        freed += report.segment_bytes(kept as usize);
    }
    Ok(Room {
        host: report.host.clone(),
        gpu,
        quota,
        booked: ledger.used(),
        free: report.free_bytes,
        bytes,
        kv_floor: (0..floor.min(report.held))
            .map(|index| report.segment_bytes(index as usize))
            .sum(),
        segments_kept: kept,
        segments_held: report.held,
        feasible: available + freed >= bytes,
        entries: ledger.entries.iter().map(LedgerEntry::describe).collect(),
    })
}

/// Why a profiling session's device storage does not fit: for every device,
/// what its ledger books of its quota, its free storage and the KV cache it
/// cannot give back.
pub(super) fn room_refusal(rooms: &[Room]) -> String {
    let bytes = rooms.first().map_or(0, |room| room.bytes);
    let mut lines = vec![format!(
        "{bytes} bytes of device storage for a profiling session do not fit: on a GPU the KV \
         cache's segments past those it cannot give back leave too little room"
    )];
    for room in rooms {
        lines.push(format!(
            "host {} GPU {}: {} of a quota of {} bytes booked and {} bytes free; {} bytes of \
             {} KV cache segments it cannot give back{}",
            room.host,
            room.gpu,
            room.booked,
            room.quota,
            room.free,
            room.kv_floor,
            room.segments_held,
            if room.feasible { "" } else { " (no room)" },
        ));
        for entry in &room.entries {
            lines.push(format!("  ledger: {entry}"));
        }
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
