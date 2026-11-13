//! The epoch's jobs (`specs/updates.md` §3.7): the drain that ends an epoch's serving beside the
//! epoch that serves next, the release of the KV planes' segments a drain to another layout gives
//! back, the growth of the regions into memory the quota frees, and the promotion that opens
//! admission.
//!
//! A drain to an epoch of another layout gives the device memory back as its requests end: each
//! scheduler shrinks its device regions by `Resize` and reports each shrink (`Output::Shrunk`);
//! the engine then releases the segments of the KV planes the region no longer covers
//! ([`release`]), and reports the release to its daemon, which has the engines of the other
//! epoch on the host grow into it ([`grow`]). Memory moves between the epochs' regions one
//! segment at a time, and only through the GPU's quota (`specs/buffer-pool.md` §14.1).
//!
//! The session's layout states which memory each region is: the head's view of it
//! ([`StorageClient::session`]) is where these jobs read a region's name, its reserved units,
//! its units, its segment table and the worker groups whose ranks map it.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::memory::cache::info::CacheTable;
use anyhow::{Context as _, Result};
use crossbeam_channel::{Receiver, Sender};
use proto::request::MigrationCause;
use service::lock;
use storage::Segment;
use storage::runtime::{GroupId, Open, Opened, Tier};

use crate::worker::Worker;

use super::operations::{resume, stopped};
use crate::memory::client::StorageClient;
use crate::process::data_plane::DataPlane;
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
/// covers, the last first, and the ranks of the worker groups that map the region release them.
/// Returns the devices of those ranks and the bytes released on each, which the daemon is told
/// (`Shrunk`); no device and no byte when nothing was released.
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
    data: &DataPlane,
    workers: &[Arc<Worker>],
    region: u8,
    units: u32,
) -> Result<(Vec<u32>, u64)> {
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

    data.release_segments(&groups, u32::try_from(from).unwrap_or(u32::MAX))?;
    let devices = workers
        .iter()
        .filter(|worker| groups.contains(&worker.group()))
        .filter_map(|worker| worker.device())
        .filter_map(|device| u32::try_from(device.0).ok())
        .collect();
    Ok((devices, released.saturating_mul(part_bytes)))
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
    data: &DataPlane,
) -> Result<Vec<(u8, u32)>> {
    {
        let mut head = lock(head);
        let (open, _) = session(&head)?;
        let regions: Vec<(usize, Vec<GroupId>)> = kv_regions(open)
            .into_iter()
            .map(|index| (index, groups_of(open, index)))
            .collect();

        for (index, groups) in regions {
            let fitting = data.fitting_segments(&groups)?;
            if fitting == 0 {
                continue;
            }
            let tables = data.create_segments(&groups, Some(fitting))?;
            grant_held(&mut head, index, &tables)?;
        }
    }

    grow_regions(schedulers, head)
}

/// Opens admission on an engine a drain or its start parked (`docs/architecture.md`, `Promote`):
/// the graph pools a parked engine gave back are restored at the addresses the graphs hold, what
/// the ranks' processes hold beside their pools is reserved in their GPUs' quotas once, and
/// every scheduler admits again. The regions a drain shrank grow by [`grow`].
///
/// # Errors
///
/// A device or quota that refuses the graph pools or the reservation; nothing admits then.
pub(crate) fn promote(schedulers: &[Sender<Message>], data: &DataPlane) -> Result<()> {
    data.restore_graphs()
        .context("restoring the promoted engine's graph pools")?;
    data.reserve_serving()
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

/// Resizes every device region each scheduler's queue pair uses to every unit the session numbers
/// for it past its reserved ones, and returns each region's number with the units it reached: a
/// grow takes what its space's free ranges hold, and may stop short. Every region is asked before
/// any reply is waited for.
///
/// The session asks for no unit of a region whose memory the ranks create once it is open, so
/// this is how the start's regions, and a woken engine's, reach the segments the head granted.
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

        let mut targets = Vec::new();
        for (scheduler, registers) in opened.registers.iter().enumerate() {
            for &(number, _) in &registers.capacity {
                let stated = index_of(opened, number)
                    .ok()
                    .and_then(|index| open.regions.get(index));
                if let Some(stated) = stated.filter(|stated| stated.tier == Tier::Device) {
                    targets.push((
                        scheduler,
                        number,
                        stated.limit.saturating_sub(stated.reserved),
                    ));
                }
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
