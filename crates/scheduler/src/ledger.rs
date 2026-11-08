//! What a KV cache may hold of its GPU, from the GPU's storage ledger.
//!
//! The host's Storage Server keeps one ledger per GPU for the deployment
//! (`storage::GpuLedger`): the quota `⌊total × fraction⌋` of the whole GPU,
//! and every item the deployment's worlds and the Transfer Server hold on it
//! (weights, each KV cache segment, the other exclusive storages, graph
//! pools, process storage, staging, the Transfer Server's context), each once
//! however many worlds map it. Every worker reports its cache's GPU's ledger
//! with its segments (`uniserve.cache_storage`), and two limits follow from it:
//!
//! - **Room** ([`room`]): device storage beside the cache, a profiling
//!   session's, takes the room the ledger has, and the tail segments the cache
//!   gives back make the rest.
//! - **Growth** ([`growable`]): a cache creates its deferred segments as the
//!   ledger's room holds them, the room being the quota less everything booked.

use serde::{Deserialize, Serialize};
use storage::{GpuLedger, Item, LedgerEntry, Scope};

/// What `uniserve.cache_storage` reports of one worker.
#[derive(Debug, Clone, Deserialize)]
pub(super) struct CacheStorage {
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
    /// Free bytes of the worker's device, which bound what the cache
    /// creates beside processes outside the deployment.
    #[serde(default)]
    pub(super) free_bytes: u64,
    /// The worker's host.
    #[serde(default)]
    pub(super) host: String,
    /// The worker's scope and world, which name its entries in the ledger.
    #[serde(default)]
    pub(super) scope: Option<Scope>,
    #[serde(default)]
    pub(super) world: Option<u64>,
    /// The deployment's ledger of the cache's GPU; `None` for a worker that
    /// keeps its storage itself, which no other world shares.
    #[serde(default)]
    pub(super) ledger: Option<GpuLedger>,
}

impl CacheStorage {
    /// The first unit of segment `index`.
    pub(super) fn first_unit(&self, index: u32) -> u32 {
        self.segments
            .iter()
            .take(index as usize)
            .map(|segment| segment.units)
            .sum()
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
pub(super) fn room(report: &CacheStorage, bytes: u64, floor: u32) -> Result<Room, String> {
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

/// Why the requested device storage does not fit: for every device,
/// what its ledger books of its quota, its free storage and the KV cache it
/// cannot give back.
pub(super) fn room_refusal(rooms: &[Room]) -> String {
    let bytes = rooms.first().map_or(0, |room| room.bytes);
    let mut lines = vec![format!(
        "{bytes} bytes of additional device storage do not fit: on a GPU the KV \
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
/// ledger's room now, and the device's free storage; with `released`, also
/// what the processes of the other world book, which end as they exit (a
/// previous world kept as its weights).
pub(super) fn growable(report: &CacheStorage, released: bool) -> u32 {
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
