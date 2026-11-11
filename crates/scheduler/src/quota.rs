//! What a KV cache may hold of its GPU, from the GPU's storage usage.
//!
//! The host's Storage Server keeps one usage per GPU for the deployment
//! (`storage::GpuLedger`): the quota `⌊total × fraction⌋` of the whole GPU,
//! and every item the deployment's epochs and the Transfer Server hold on it
//! (weights, each KV cache segment, the other exclusive storages, graph
//! pools, process storage, staging, the Transfer Server's context), each once
//! however many epochs map it. Every worker reports its cache's GPU's usage
//! with its segments (`EpochOperation::CacheStorage`), and two limits follow from it:
//!
//! - **Room** ([`room`]): device storage beside the cache, a profiling
//!   session's, takes the room the usage has, and the tail segments the cache
//!   gives back make the rest.
//! - **Growth** ([`growable`]): a cache creates its deferred segments as the
//!   quota's room holds them, the room being the quota less everything reserved.

use ipc::memory::CacheStorageReport;
use serde::Serialize;
use storage::LedgerEntry;

/// One device's room for the device storage a profiling session holds:
/// what its usage and its free storage leave, and the tail segments the KV
/// cache gives back to make the rest. Reported per GPU by the change's output
/// and refusal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct Room {
    pub(super) host: String,
    /// The GPU, as `nvidia-smi` names it.
    pub(super) gpu: String,
    pub(super) quota: u64,
    /// What the usage counts.
    pub(super) reserved: u64,
    /// The device's free storage when prepared.
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
    /// Every entry of the GPU's usage.
    pub(super) entries: Vec<String>,
}

/// The room on the device of `report` for `bytes` of device storage beside
/// everything its usage counts, of whose segments the cache cannot give
/// back the first `floor`: the cache keeps all it holds when the usage's
/// room and the device's free storage hold `bytes` already, and otherwise
/// the largest prefix whose given-back tail makes the rest (each segment
/// given back frees its bytes of both). Fails for a worker that reports no
/// usage or no quota.
pub(super) fn room(report: &CacheStorageReport, bytes: u64, floor: u32) -> Result<Room, String> {
    let Some(usage) = &report.usage else {
        return Err("a worker that holds a split KV cache reports no storage usage".to_owned());
    };
    let gpu = storage::protocol::gpu_name(&usage.gpu);
    let quota = usage
        .quota
        .ok_or_else(|| format!("no epoch states a quota in the storage usage of GPU {gpu}"))?;
    // A process outside the deployment may hold more of the device than the
    // quota leaves it.
    let available = quota.saturating_sub(usage.used()).min(report.free_bytes);
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
        reserved: usage.used(),
        free: report.free_bytes,
        bytes,
        kv_floor: (0..floor.min(report.held))
            .map(|index| report.segment_bytes(index as usize))
            .sum(),
        segments_kept: kept,
        segments_held: report.held,
        feasible: available + freed >= bytes,
        entries: usage.entries.iter().map(LedgerEntry::describe).collect(),
    })
}

/// Why the requested device storage does not fit: for every device,
/// what its usage counts of its quota, its free storage and the KV cache it
/// cannot give back.
pub(super) fn room_refusal(rooms: &[Room]) -> String {
    let bytes = rooms.first().map_or(0, |room| room.bytes);
    let mut lines = vec![format!(
        "{bytes} bytes of additional device storage do not fit: on a GPU the KV \
         cache's segments past those it cannot give back leave too little room"
    )];
    for room in rooms {
        lines.push(format!(
            "host {} GPU {}: {} of a quota of {} bytes reserved and {} bytes free; {} bytes of \
             {} KV cache segments it cannot give back{}",
            room.host,
            room.gpu,
            room.reserved,
            room.quota,
            room.free,
            room.kv_floor,
            room.segments_held,
            if room.feasible { "" } else { " (no room)" },
        ));
        for entry in &room.entries {
            lines.push(format!("  usage: {entry}"));
        }
    }
    lines.join("\n")
}

/// How many of the segments of `report` past those it holds fit the
/// quota's room now, and the device's free storage; with `released`, also
/// what the processes of the other epoch reserve, which end as they exit (a
/// previous epoch kept as its weights).
pub(super) fn growable(report: &CacheStorageReport, released: bool) -> u32 {
    let (Some(usage), Some(epoch)) = (&report.usage, report.epoch) else {
        return 0;
    };
    let Some(quota) = usage.quota else {
        return 0;
    };
    let ending = if released {
        usage.bytes(|entry| {
            entry.item.booked() && !entry.epochs.is_empty() && !entry.epochs.contains(&epoch)
        })
    } else {
        0
    };
    let mut room = (quota + ending)
        .saturating_sub(usage.used())
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
