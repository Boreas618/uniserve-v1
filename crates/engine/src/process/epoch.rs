//! The epoch's jobs (`specs/updates.md` §3.7): the drain that ends an epoch's serving beside the
//! epoch that serves next, the release of the KV planes' segments a drain to another layout gives
//! back, the growth of the regions into memory the quota frees, and the promotion that opens
//! admission.
//!
//! A drain to an epoch of another layout gives the device memory back as its requests end: each
//! scheduler shrinks its device regions by `Resize` and reports each shrink (`Output::Shrunk`);
//! the engine then releases the segments of the KV planes the region no longer covers
//! ([`release`]) on every rank of the region, its peers' included, and reports the release to its
//! daemon once per host the ranks lie on ([`by_host`]); the daemon has the scheduler node of each
//! engine of the other epoch with ranks on that host grow into it on every rank it holds
//! ([`grow`]). Memory moves between the epochs' regions one segment at a time, and only through
//! the GPU's quota (`specs/buffer-pool.md` §14.1).
//!
//! The session's layout states which memory each region is: the head's view of it
//! ([`StorageClient::session`]) is where these jobs read a region's name, its reserved units,
//! its units, its segment table and the worker groups whose ranks map it.

use std::sync::Mutex;
use std::time::Instant;

use crate::memory::cache::info::CacheTable;
use anyhow::{Context as _, Result, bail};
use crossbeam_channel::{Receiver, Sender};
use proto::config::ParallelConfig;
use proto::request::MigrationCause;
use service::lock;
use storage::Segment;
use storage::runtime::{GroupId, Open, Opened, Tier};

use super::operations::{resume, stopped};
use crate::memory::client::StorageClient;
use crate::process::binding::{Released, WeightBinding};
use crate::scheduler::Message;

/// Starts the drain of every scheduler (`specs/scheduler-design.md` §3.8): each admits nothing
/// from here on, ends its queued requests as migratable at once and its running ones by
/// themselves or at `deadline`, and with `shrink` gives its device regions back as they end.
/// Returns each scheduler's reply, in the schedulers' order, which arrives once it holds no
/// request (and with `shrink`, once its regions answered their last `Resize`); a drained
/// scheduler then forms no batch until it is resumed ([`promote`]).
///
/// The replies are the caller's to wait for: a request ends only once the daemon acknowledged
/// its last tokens, which reach the daemon through the control thread.
///
/// # Errors
///
/// When a scheduler has stopped.
pub(crate) fn drain(
    schedulers: &[Sender<Message>],
    deadline: Instant,
    shrink: bool,
) -> Result<Vec<Receiver<()>>> {
    let mut replies = Vec::with_capacity(schedulers.len());
    for (index, inbox) in schedulers.iter().enumerate() {
        let (reply, replied) = crossbeam_channel::bounded(1);
        inbox
            .send(Message::Drain(
                MigrationCause::Drain,
                deadline,
                shrink,
                reply,
            ))
            .map_err(|_| stopped(index))?;
        replies.push(replied);
    }
    Ok(replies)
}

/// Releases what a shrink of device region `region` to `units` units gave back
/// (`Output::Shrunk`): the head withdraws the segments of the KV planes the region no longer
/// covers, the last first, and the ranks of the worker groups that map the region release them,
/// here and on the peers. Returns, for this process (`None`) and each peer that holds such
/// ranks (by its first rank), the CUDA ordinals of the devices whose ranks gave segments back,
/// and the bytes released on each, which the daemon is told per host (`Shrunk`, [`by_host`]);
/// no process and no byte when nothing was released.
///
/// The region holds `units` units from its reserved ones on, the lowest first, so a segment none
/// of whose units past the reserved ones lies below `reserved + units` holds none of the region's.
/// A shrink takes the free units from the top and never a held block, so the region may still
/// hold a unit higher up: the Storage Server then refuses that segment's withdrawal, and it and
/// the segments below it stay for a later shrink.
///
/// # Errors
///
/// A region the session does not state, or a rank's failure to release what the head withdrew.
pub(crate) fn release(
    head: &Mutex<StorageClient>,
    binding: &WeightBinding,
    region: u8,
    units: u32,
) -> Result<(Released, u64)> {
    let (from, released, part_bytes, groups) = {
        let mut head = lock(head);
        let (open, opened) = session(&head)?;
        let index = index_of(opened, region)?;
        let stated = open
            .regions
            .get(index)
            .context("the session states no such region")?;
        let table = opened.segments.get(index).cloned().unwrap_or_default();
        let (reserved, part_bytes) = (stated.reserved, stated.part_bytes);
        let groups = groups_of(open, index);

        let end = reserved.saturating_add(units);
        let first = table
            .iter()
            .position(|&(_, start, _)| start.max(reserved) >= end)
            .unwrap_or(table.len());

        let (from, withdrawn) = withdraw_from(&mut head, index, first);
        if let Err(refusal) = withdrawn {
            tracing::info!(
                region,
                units,
                %refusal,
                "a segment past the shrunk region still holds a unit of it, and stays"
            );
        }
        let released: u64 = table
            .iter()
            .skip(from)
            .map(|&(_, _, units)| u64::from(units))
            .sum();
        (from, released, part_bytes, groups)
    };
    if released == 0 {
        return Ok((Vec::new(), 0));
    }

    let devices = binding.release_segments(&groups, u32::try_from(from).unwrap_or(u32::MAX))?;
    Ok((devices, released.saturating_mul(part_bytes)))
}

/// The devices a release freed ([`release`]) by host, each host once, as the daemon is told them
/// (`Shrunk`, one per host): this process's on `here`, its host, and a peer's on the host
/// `parallel` places the peer's first rank on (the deployment numbers its ranks over its groups
/// in their order). A host where no released rank holds a device is left out: no quota there
/// changed.
pub(crate) fn by_host(
    parallel: &ParallelConfig,
    here: &str,
    released: Released,
) -> Vec<(String, Vec<u32>)> {
    let mut hosts: Vec<(String, Vec<u32>)> = Vec::new();
    for (first, devices) in released {
        if devices.is_empty() {
            continue;
        }
        let host = match first {
            None => here,
            Some(first) => {
                let placed = parallel
                    .worker_groups
                    .iter()
                    .flat_map(|group| &group.ranks)
                    .nth(usize::try_from(first).unwrap_or(usize::MAX));
                let Some(placed) = placed else {
                    tracing::warn!(first, "a peer's first rank is placed on no host");
                    continue;
                };
                placed.host.as_str()
            }
        };
        match hosts.iter_mut().find(|(known, _)| known == host) {
            Some((_, known)) => known.extend(devices),
            None => hosts.push((host.to_owned(), devices)),
        }
    }
    hosts
}

/// Grows the engine's regions into the memory its GPUs' quotas hold now (`specs/updates.md`
/// §3.7, `Promote { grow }`, `ResizeCache { Grow }`): the ranks of each region of the KV planes
/// create as many of the segments they gave back as their quotas and devices have room for, the
/// head grants them, and every scheduler resizes its device regions to every unit the session
/// numbers, which each reaches as far as its space's free ranges go. Returns each device
/// region's number with the units it reached.
///
/// # Errors
///
/// A rank's failure or refusal, the Storage Server's refusal of a grant, or a scheduler that
/// stopped.
pub(crate) fn grow(
    schedulers: &[Sender<Message>],
    head: &Mutex<StorageClient>,
    binding: &WeightBinding,
) -> Result<Vec<(u8, u32)>> {
    create_fitting(&mut lock(head), binding, None)?;
    grow_regions(schedulers, head)
}

/// The ranks' side of a grow ([`grow`]) and of a warm start: the ranks of each region of the KV
/// planes create as many of the segments they gave back as their quotas and devices have room
/// for, and the head grants them. A start creates its regions' segments before its warm-up, as
/// far as its quotas held them then, and gives back once warm what no longer fits beside the warm
/// processes ([`keep_warm_segments`]), so what it creates here is what a predecessor's drain
/// freed meanwhile (`specs/engine-worker-port.md` §1, "The process a division counts").
///
/// `capacity` is the tokens a deployment states its cache holds (`kv_cache_config.max_tokens`)
/// when a start creates its regions: such a region is created whole, and refused, naming the
/// capacity and the warm processes of this process's ranks, when a rank's quota cannot hold what
/// it gave back even as the epochs it starts beside drain (`Room::elsewhere`); what those epochs
/// hold now it grows into by `Grow` as they drain.
///
/// # Errors
///
/// A stated capacity the quota cannot hold, a rank's failure or refusal, or the Storage
/// Server's refusal of a grant.
pub(crate) fn create_fitting(
    head: &mut StorageClient,
    binding: &WeightBinding,
    capacity: Option<u64>,
) -> Result<()> {
    let (open, _) = session(head)?;
    let regions: Vec<(usize, String, Vec<GroupId>)> = kv_regions(open)
        .into_iter()
        .map(|index| {
            let name = open
                .regions
                .get(index)
                .map(|region| region.name.clone())
                .unwrap_or_default();
            (index, name, groups_of(open, index))
        })
        .collect();

    for (index, name, groups) in regions {
        if let Some(tokens) = capacity {
            whole(binding, &groups, (&name, tokens))?;
        }
        let fitting = binding.fitting_segments(&groups)?;
        if fitting == 0 {
            continue;
        }
        let tables = binding.create_segments(&groups, Some(fitting))?;
        let granted = grant_held(head, index, &tables)?;
        tracing::info!(
            region = %name,
            granted,
            "the ranks created the segments their quotas hold and the head granted them"
        );
    }
    Ok(())
}

/// Gives back, once a start's ranks are warm, the tail segments of each region of the KV planes
/// that their GPUs' quotas no longer hold beside their warm processes
/// (`specs/engine-worker-port.md` §1, "The process a division counts"). The start created, before
/// its warm-up, every whole segment its quotas held then beside the processes measured before it;
/// the warm-up's instantiated graphs and library state lay outside the quotas meanwhile. Each
/// region keeps the segments from the first that every rank of it, the peers' included, holds
/// beside its warm process (`WeightBinding::warm_segments`); the head withdraws the rest from the
/// region's space, the last first, and the ranks give them back, so each rank's warm process then
/// fits its quota and is reserved (`WeightBinding::reserve_serving`). A stated capacity that gave
/// a segment back here is refused by name when the ranks create what fits ([`create_fitting`]).
///
/// The session's regions hold no unit yet (the schedulers resize them after), so no withdrawal is
/// refused for a held block.
///
/// # Errors
///
/// A rank's failure to measure its process or to give segments back, or the Storage Server's
/// refusal of a withdrawal.
pub(crate) fn keep_warm_segments(head: &mut StorageClient, binding: &WeightBinding) -> Result<()> {
    let (open, _) = session(head)?;
    let regions: Vec<(usize, String, Vec<GroupId>)> = kv_regions(open)
        .into_iter()
        .map(|index| {
            let name = open
                .regions
                .get(index)
                .map(|region| region.name.clone())
                .unwrap_or_default();
            (index, name, groups_of(open, index))
        })
        .collect();

    for (index, name, groups) in regions {
        let Some(kept) = binding.warm_segments(&groups)? else {
            continue;
        };
        let granted = session(head)?.1.segments.get(index).map_or(0, Vec::len);
        let kept = usize::try_from(kept).unwrap_or(usize::MAX);
        if kept >= granted {
            continue;
        }

        let (from, withdrawn) = withdraw_from(head, index, kept);
        withdrawn.with_context(|| {
            format!(
                "withdrawing the segments of KV region {name} the quotas no longer hold beside \
                 the warm processes"
            )
        })?;
        binding.release_segments(&groups, u32::try_from(from).unwrap_or(u32::MAX))?;
        tracing::info!(
            region = %name,
            kept = from,
            given_back = granted - from,
            "the warm ranks gave back the segments their quotas no longer hold beside them"
        );
    }
    Ok(())
}

/// Refuses a KV region `name` of a stated capacity of `tokens` tokens whose ranks, those of the
/// worker groups `groups`, cannot hold every segment they gave back: what the deployment's quota
/// of a rank's GPU and its device have room for, with what other epochs of the deployment hold
/// there, which returns as they drain ([`create_fitting`]).
///
/// # Errors
///
/// The refusal, naming the capacity, the bytes the given-back segments take on the rank's GPU, its
/// room, and the warm processes this process's ranks reserved; a rank's failure to state its
/// segments or its room.
fn whole(binding: &WeightBinding, groups: &[GroupId], (name, tokens): (&str, u64)) -> Result<()> {
    // Every rank of a region holds the same segments; the units given back are the most any
    // rank's table leaves uncreated.
    let given_back: u64 = binding
        .segment_tables(groups)?
        .iter()
        .flatten()
        .map(|table| {
            table
                .segments
                .iter()
                .skip(table.held as usize)
                .map(|segment| u64::from(segment.units))
                .sum()
        })
        .max()
        .unwrap_or(0);
    if given_back == 0 {
        return Ok(());
    }

    for (gpu, room, unit_bytes) in binding.device_rooms(groups)? {
        let needed = given_back.saturating_mul(unit_bytes);
        if room.holds(needed, room.elsewhere) {
            continue;
        }
        let processes: Vec<String> = binding
            .warm_processes()
            .into_iter()
            .map(|(device, bytes)| format!("{bytes} bytes on device {}", device.0))
            .collect();
        let gpu: String = gpu.iter().map(|byte| format!("{byte:02x}")).collect();
        bail!(
            "kv_cache_config.max_tokens = {tokens} does not fit beside the warm process: KV \
             region {name} needs {needed} more bytes on GPU {gpu}, whose quota leaves {} and \
             whose device has {} free, with {} that other epochs hold; the warm processes of this \
             process's ranks hold {}",
            room.quota,
            room.free,
            room.elsewhere,
            if processes.is_empty() {
                "nothing reserved".to_owned()
            } else {
                processes.join(", ")
            }
        );
    }
    Ok(())
}

/// Opens admission on an engine a drain or its start parked (`docs/architecture.md`, `Promote`):
/// the graph pools a parked engine gave back are restored at the addresses the graphs hold, what
/// the ranks' processes hold beside their pools is reserved in their GPUs' quotas once, and
/// every scheduler admits again. The regions a drain shrank grow by [`grow`].
///
/// # Errors
///
/// A device or quota that refuses the graph pools or the reservation; nothing admits then.
pub(crate) fn promote(schedulers: &[Sender<Message>], binding: &WeightBinding) -> Result<()> {
    binding
        .restore_graphs()
        .context("restoring the promoted engine's graph pools")?;
    binding
        .reserve_serving()
        .context("reserving what the ranks hold in their devices' quotas")?;

    resume(schedulers);
    Ok(())
}

/// The session the head opened: its layout, and the buffer pool's answer with each region's
/// segment table as the Storage Server holds it.
///
/// # Errors
///
/// When the head opened no session: a peer's process, or one still starting.
pub(super) fn session(head: &StorageClient) -> Result<(&Open, &Opened)> {
    head.session()
        .context("this process holds no session with the buffer pool")
}

/// The index among the session's regions of the region the buffer pool numbers `number`.
fn index_of(opened: &Opened, number: u8) -> Result<usize> {
    opened
        .regions
        .iter()
        .position(|&(numbered, _)| numbered == number)
        .with_context(|| format!("the session states no region {number}"))
}

/// The regions of the session the ranks' KV planes back, by index: the device regions a class of
/// blocks of a fixed size takes its blocks from. A class sized by each request (a product, a
/// stored trajectory) takes its blocks from memory of the ranks' own processes.
pub(super) fn kv_regions(open: &Open) -> Vec<usize> {
    let mut regions: Vec<usize> = open
        .groups
        .iter()
        .flat_map(|group| group.regions.iter().zip(&open.classes))
        .filter(|(_, class)| class.units > 0)
        .flat_map(|(regions, _)| regions.iter().map(|&region| usize::from(region)))
        .filter(|&index| {
            open.regions
                .get(index)
                .is_some_and(|region| region.tier == Tier::Device)
        })
        .collect();

    regions.sort_unstable();
    regions.dedup();
    regions
}

/// The worker groups whose ranks map the session's region at `index`: those that take a class's
/// blocks from it or keep their slot rows in it.
pub(super) fn groups_of(open: &Open, index: usize) -> Vec<GroupId> {
    open.groups
        .iter()
        .filter(|group| {
            usize::from(group.slots) == index
                || group
                    .regions
                    .iter()
                    .flatten()
                    .any(|&region| usize::from(region) == index)
        })
        .map(|group| group.id)
        .collect()
}

/// The position, in the session's order of schedulers, of the scheduler whose queue pair uses the
/// region the buffer pool numbers `number`: the one whose `Resize` changes it.
///
/// # Errors
///
/// When no scheduler's queue pair uses the region.
pub(super) fn scheduler_of(opened: &Opened, number: u8) -> Result<usize> {
    opened
        .registers
        .iter()
        .position(|registers| {
            registers
                .capacity
                .iter()
                .any(|&(numbered, _)| numbered == number)
        })
        .with_context(|| format!("no scheduler's queue pair uses region {number}"))
}

/// Withdraws the segments of the session's region at `index` from position `first` of its
/// segment table on, the last first, so that what is withdrawn is always the table's tail.
/// Returns the position of the lowest segment withdrawn (the table's length when none was), from
/// which the ranks release them, and the first refusal, which leaves that segment and the ones
/// below it in the table: the Storage Server refuses a segment a region holds a unit of.
pub(super) fn withdraw_from(
    head: &mut StorageClient,
    index: usize,
    first: usize,
) -> (usize, Result<()>) {
    let (name, table) = match session(head) {
        Ok((open, opened)) => (
            open.regions.get(index).map(|region| region.name.clone()),
            opened.segments.get(index).cloned().unwrap_or_default(),
        ),
        Err(error) => return (0, Err(error)),
    };
    let Some(name) = name else {
        return (table.len(), Ok(()));
    };

    let mut from = table.len();
    for at in (first..table.len()).rev() {
        let Some(&(segment, _, _)) = table.get(at) else {
            break;
        };
        if let Err(refusal) = head.withdraw(&name, segment) {
            return (from, Err(refusal));
        }
        from = at;
    }
    (from, Ok(()))
}

/// Grants the segments every rank of the session's region at `index` holds now beyond those its
/// space's table holds, as `tables` (the region's ranks', none for a rank whose planes are not
/// split) state them: segment `i` holds the units from the sum of the units of the segments
/// before it. Returns how many were granted.
///
/// # Errors
///
/// The Storage Server's refusal.
pub(super) fn grant_held(
    head: &mut StorageClient,
    index: usize,
    tables: &[Option<CacheTable>],
) -> Result<usize> {
    // A segment joins the space only once every rank that maps the region holds it.
    let held = tables.iter().flatten().map(|table| table.held).min();
    let (Some(table), Some(held)) = (tables.iter().flatten().next(), held) else {
        return Ok(0);
    };

    let (name, granted) = {
        let (open, opened) = session(head)?;
        let name = open
            .regions
            .get(index)
            .map(|region| region.name.clone())
            .context("the session states no such region")?;
        (name, opened.segments.get(index).map_or(0, Vec::len))
    };

    let mut first: u32 = table
        .segments
        .iter()
        .take(granted)
        .map(|segment| segment.units)
        .sum();
    let mut count = 0;
    for segment in table.segments.iter().take(held as usize).skip(granted) {
        let stated = Segment {
            epoch: segment.epoch,
            ordinal: segment.ordinal,
        };
        head.grant(&name, stated, first, segment.units)?;
        first = first.saturating_add(segment.units);
        count += 1;
    }
    Ok(count)
}

/// Resizes the regions each scheduler's queue pair uses to what the session states of them, and
/// returns each region's number with the units it reached: every device region of the KV planes
/// to every unit the session numbers for it past its reserved ones, and every other region to
/// the units the session states it serves, within the numbering the buffer pool holds for it:
/// what the epoch's layout asked for, or the fewer a region of a warm media rank's slots kept
/// (`StorageClient::keep_units`). A grow takes what its space's free ranges hold and a shrink
/// frees what its blocks let it; either may stop short. Every region is asked before any reply
/// is waited for.
///
/// The session asks for no unit of a region whose memory the ranks create once it is open, so
/// this is how the start's regions, and a woken engine's, reach the segments the head granted.
/// A host region an earlier epoch opened keeps that epoch's numbering and units at `Open`, so
/// this is how the epoch's own statement of its host tier (`kv_cache_config.host_gib`) takes
/// effect on it (`specs/buffer-pool.md` §14.1).
///
/// # Errors
///
/// When the head opened no session, or a scheduler has stopped.
pub(super) fn grow_regions(
    schedulers: &[Sender<Message>],
    head: &Mutex<StorageClient>,
) -> Result<Vec<(u8, u32)>> {
    let targets: Vec<(usize, u8, u32)> = {
        let head = lock(head);
        let (open, opened) = session(&head)?;
        let planes = kv_regions(open);

        let mut targets = Vec::new();
        for (scheduler, registers) in opened.registers.iter().enumerate() {
            for &(number, _) in &registers.capacity {
                let Ok(index) = index_of(opened, number) else {
                    continue;
                };
                let Some(stated) = open.regions.get(index) else {
                    continue;
                };
                // A region of the KV planes asks for no unit at `Open`: it reaches every unit it
                // numbers as far as the segments granted to its space go. Every other region
                // serves the units the session states of it.
                let numbered = stated.limit.saturating_sub(stated.reserved);
                let units = match stated.tier {
                    Tier::Device if planes.contains(&index) => numbered,
                    Tier::Device | Tier::Host => stated.units.min(numbered),
                    Tier::Remote => continue,
                };
                targets.push((scheduler, number, units));
            }
        }
        targets
    };

    let mut replies = Vec::with_capacity(targets.len());
    for (scheduler, number, units) in targets {
        let inbox = schedulers
            .get(scheduler)
            .ok_or_else(|| stopped(scheduler))?;
        let (reply, replied) = crossbeam_channel::bounded(1);
        inbox
            .send(Message::Resize(number, units, reply))
            .map_err(|_| stopped(scheduler))?;
        replies.push((scheduler, number, replied));
    }

    replies
        .into_iter()
        .map(|(scheduler, number, replied)| {
            let reached = replied.recv().map_err(|_| stopped(scheduler))?;
            Ok((number, reached))
        })
        .collect()
}

#[cfg(test)]
mod tests;
