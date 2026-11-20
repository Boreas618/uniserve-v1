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
//! the units it serves, its segment table and the worker groups whose ranks map it. A region of
//! the KV planes serves the epoch's capacity, within the numbering the buffer pool holds for it:
//! an epoch of the same layout shares the regions of the epoch it succeeds, which keep the
//! serving epoch's sizes through the target's start, and resizes them to its own capacity when
//! it is promoted ([`promote`]). A promotion of the epoch it succeeds, at a rollback, resizes them
//! to that epoch's sizes again, and an update abandoned before its switch resizes nothing. A
//! model that keeps a recurrent state has a state region beside each KV region, one checkpoint a
//! unit (`specs/engine-worker-port.md` §1.6): every job names the region whose memory the ranks
//! change, a growth creates a state region's segments before a KV region's, and a start's
//! give-back once warm takes a KV region's tail first.

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
use crate::memory::room::RoomAsk;
use crate::process::binding::{Released, WeightBinding};
use crate::scheduler::Message;

/// Starts the drain of every scheduler (`specs/scheduler-design.md` §3.8): each admits nothing
/// new from here on, ends its queued requests as migratable at once and its running ones by
/// themselves or at `deadline`; with no deadline (a complete drain, before an epoch that processes
/// requests otherwise) it runs every request it holds to its end, admitting the queued ones in
/// their turn. With `shrink` it gives its device regions back as they end.
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
    deadline: Option<Instant>,
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
/// A shrink to `units` gives back the region's units from `reserved + units` on, so a segment
/// none of whose units past the reserved ones lies below `reserved + units` holds none of the
/// region's once nothing holds a block there. A held block keeps its units in the region until
/// it is let go: the Storage Server then refuses that segment's withdrawal, and it and the
/// segments below it stay for a later shrink.
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

    let devices =
        binding.release_segments(&groups, region, u32::try_from(from).unwrap_or(u32::MAX))?;
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
/// head grants them, and every scheduler resizes its regions to the units the session states of
/// them ([`grow_regions`]), which each reaches as far as its space's free ranges go. Returns each
/// region's number with the units it reached.
///
/// The regions grow in rounds, each creating at most twice the segments per region of the round
/// before, from one, and resizing the schedulers' regions after it, until a round creates none.
/// The driver creates a segment's parts one at a time (one allocation per column of each plane,
/// some 6,000 for a 64-segment region of a data-parallel rank of a 48-layer model, which take
/// seconds), and the ranks of one process create no faster at once than one after another, so a
/// grow that created each region whole before resizing any left every region without units
/// until the last was created: a promotion of an engine whose drain gave all its memory back, a
/// rollback's, admitted nothing for as long. Rounds give every region units within the first,
/// and double what each holds every round after, in as many rounds as the bits of its count.
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
    let mut round = 1_u32;
    loop {
        let created = create_fitting(&mut lock(head), binding, None, Some(round))?;
        let reached = grow_regions(schedulers, head)?;
        if created == 0 {
            return Ok(reached);
        }
        round = round.saturating_mul(2);
    }
}

/// The ranks' side of a grow ([`grow`]) and of a warm start: the ranks of each device region of
/// the KV planes and of the state units create as many of the segments they gave back as their
/// quotas and devices have room for, and the head grants them; the state regions first, since a
/// request can run on few KV units but on no checkpoint. A start creates its regions' segments before its warm-up, as
/// far as its quotas held them then, and gives back once warm what no longer fits beside the warm
/// processes ([`keep_warm_segments`]), so what it creates here is what a predecessor's drain
/// freed meanwhile (`specs/engine-worker-port.md` §1, "The process a division counts"); a
/// relaunch, which created none past the memory its lost predecessor left, creates here what its
/// quota holds past it once its warm process has taken that predecessor's reservation over.
///
/// `capacity` is the tokens a deployment states its cache holds (`kv_cache_config.max_tokens`)
/// when a start creates its regions: such a region is created whole, and refused, naming the
/// capacity and the warm processes of this process's ranks, when a rank's quota cannot hold what
/// it gave back even as the epochs it starts beside drain (`Room::elsewhere`); what those epochs
/// hold now it grows into by `Grow` as they drain. `limit` caps the segments each region creates,
/// a round of a grow's ([`grow`]); `None` creates every one that fits. Returns the segments the
/// head granted, over every region.
///
/// # Errors
///
/// A stated capacity the quota cannot hold, a rank's failure or refusal, or the Storage
/// Server's refusal of a grant.
pub(crate) fn create_fitting(
    head: &mut StorageClient,
    binding: &WeightBinding,
    capacity: Option<u64>,
    limit: Option<u32>,
) -> Result<usize> {
    let mut regions = device_regions(head)?;
    regions.sort_by_key(|(.., state)| !*state);

    let mut created = 0;
    for (index, number, name, groups, _) in regions {
        if let Some(tokens) = capacity {
            whole(binding, &groups, number, (&name, tokens))?;
        }
        let fitting = binding.fitting_segments(&groups, number)?;
        let fitting = limit.map_or(fitting, |limit| fitting.min(limit));
        if fitting == 0 {
            continue;
        }

        let tables = binding.create_segments(&groups, number, Some(fitting))?;
        let granted = grant_held(head, index, &tables)?;
        created += granted;
        tracing::info!(
            region = %name,
            granted,
            "the ranks created the segments their quotas hold and the head granted them"
        );
    }
    Ok(created)
}

/// Asks `room` for room in the quota of every GPU whose warm process passes what the quota holds
/// by more than the start's own segments past each region's first give back
/// ([`WeightBinding::warm_rooms`]), on this process's ranks and every peer's, in one ask: a target
/// starting beside a serving epoch that holds the quota has the serving epoch give back from its
/// KV cache what keeping its warm segments ([`keep_warm_segments`]) could not. Nothing is asked
/// when every GPU's own segments cover its warm process.
///
/// # Errors
///
/// A rank's failure to measure its process or read its quota, or the room that was not made.
pub(crate) fn ask_warm_room(
    head: &StorageClient,
    binding: &WeightBinding,
    room: &dyn RoomAsk,
) -> Result<()> {
    let regions: Vec<(Vec<GroupId>, u8)> = device_regions(head)?
        .into_iter()
        .map(|(_, number, _, groups, _)| (groups, number))
        .collect();
    let rooms = binding
        .warm_rooms(&regions)
        .context("measuring the warm processes beside the KV planes")?;
    if rooms.is_empty() {
        return Ok(());
    }
    room.ask(&rooms).map_err(|reason| {
        anyhow::anyhow!(
            "the warm processes pass their GPUs' quotas by more than the start's own segments give \
             back, and no room was made: {reason}"
        )
    })
}

/// Gives back, once a start's ranks are warm, the tail segments of each device region that
/// their GPUs' quotas no longer hold beside their warm processes, the KV regions first: each
/// region's ranks measure their quotas after the region before gave back, so a state region's
/// segments go back only when the KV regions' could not make the room
/// (`specs/engine-worker-port.md` §1, "The process a division counts"). The start created, before
/// its warm-up, every whole segment its quotas held then beside the processes measured before it;
/// the warm-up's instantiated graphs and library state lay outside the quotas meanwhile. Each
/// region keeps the segments from the first that every rank of it, the peers' included, holds
/// beside its warm process (`WeightBinding::warm_segments`); the head withdraws the rest from the
/// region's space, the last first, and the ranks give them back, so each rank's warm process then
/// fits its quota and is reserved (`WeightBinding::reserve_serving`). A stated capacity that gave
/// a segment back here is refused by name when the ranks create what fits ([`create_fitting`]).
///
/// The session's regions hold no unit yet (the schedulers resize them once the engine is
/// promoted), so no withdrawal is refused for a held block.
///
/// # Errors
///
/// A rank's failure to measure its process or to give segments back, or the Storage Server's
/// refusal of a withdrawal.
pub(crate) fn keep_warm_segments(head: &mut StorageClient, binding: &WeightBinding) -> Result<()> {
    let mut regions = device_regions(head)?;
    regions.sort_by_key(|(.., state)| *state);

    for (index, number, name, groups, _) in regions {
        let Some(kept) = binding.warm_segments(&groups, number)? else {
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
                "withdrawing the segments of region {name} the quotas no longer hold beside the \
                 warm processes"
            )
        })?;
        binding.release_segments(&groups, number, u32::try_from(from).unwrap_or(u32::MAX))?;
        tracing::info!(
            region = %name,
            kept = from,
            given_back = granted - from,
            "the warm ranks gave back the segments their quotas no longer hold beside them"
        );
    }
    Ok(())
}

/// Refuses a device region `name` (numbered `number`) of a deployment that states a capacity of
/// `tokens` tokens whose ranks, those of the worker groups `groups`, cannot hold every segment
/// they gave back: what the deployment's quota
/// of a rank's GPU and its device have room for, with what other epochs of the deployment hold
/// there, which returns as they drain ([`create_fitting`]).
///
/// # Errors
///
/// The refusal, naming the capacity, the bytes the given-back segments take on the rank's GPU, its
/// room, and the warm processes this process's ranks reserved; a rank's failure to state its
/// segments or its room.
fn whole(
    binding: &WeightBinding,
    groups: &[GroupId],
    number: u8,
    (name, tokens): (&str, u64),
) -> Result<()> {
    // Every rank of a region holds the same segments; the units given back are the most any
    // rank's table leaves uncreated.
    let given_back: u64 = binding
        .segment_tables(groups, number)?
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

    for (gpu, room, unit_bytes) in binding.device_rooms(groups, number)? {
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
            "kv_cache_config.max_tokens = {tokens} does not fit beside the warm process: \
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
/// the ranks' processes hold beside their pools is reserved in their GPUs' quotas once, the
/// session's head (`head`, none on a peer, whose regions the scheduler node's head states) has
/// every scheduler resize its regions to the units the session states of them
/// ([`grow_regions`]), and every scheduler admits again. The regions a drain shrank grow by
/// [`grow`].
///
/// The resize is where the epoch's sizes take effect: the start resizes no region, and a parked
/// engine grows none ([`crate::process::lifecycle::grow`]), so the regions a target of the same
/// layout shares with the serving engine keep the serving epoch's sizes until the target's
/// promotion at the switch resizes them to its own device cache's capacity and host pools, and
/// an update abandoned before its switch leaves them as they were. A rollback's promotion of the
/// engine the target succeeded resizes them to that engine's sizes again. A shrink that stops
/// short of the blocks requests still hold goes on in the scheduler as they come free.
///
/// # Errors
///
/// A device or quota that refuses the graph pools or the reservation, or a scheduler that
/// stopped; nothing admits then.
pub(crate) fn promote(
    schedulers: &[Sender<Message>],
    head: Option<&Mutex<StorageClient>>,
    binding: &WeightBinding,
) -> Result<()> {
    binding
        .restore_graphs()
        .context("restoring the promoted engine's graph pools")?;
    binding
        .reserve_serving()
        .context("reserving what the ranks hold in their devices' quotas")?;

    // The regions are resized first, so the first request admitted finds what the epoch states.
    if let Some(head) = head {
        grow_regions(schedulers, head).context("resizing the promoted engine's regions")?;
    }
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

/// The regions of the session the ranks' KV planes and state units back, by index: the device
/// regions a class of blocks of a fixed size takes its blocks from. A class sized by each request
/// (a product, a stored trajectory) takes its blocks from memory of the ranks' own processes.
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

/// Whether the session's region at `index` is a state region: the device region of a class whose
/// live state is in the slot, the recurrent-state class (`specs/engine-worker-port.md` §1.6).
pub(super) fn is_state_region(open: &Open, index: usize) -> bool {
    open.groups.iter().any(|group| {
        group
            .regions
            .iter()
            .zip(&open.classes)
            .any(|(regions, class)| {
                class.state_in_slot && regions.iter().any(|&region| usize::from(region) == index)
            })
    })
}

/// The session's device regions of the ranks' KV planes and state units ([`kv_regions`]), each
/// with its index, its number, its name, the worker groups whose ranks map it and whether it is a
/// state region ([`is_state_region`]).
///
/// # Errors
///
/// When the head opened no session.
#[expect(
    clippy::type_complexity,
    reason = "what a region job reads of each device region, as the session states it"
)]
fn device_regions(head: &StorageClient) -> Result<Vec<(usize, u8, String, Vec<GroupId>, bool)>> {
    let (open, opened) = session(head)?;
    Ok(kv_regions(open)
        .into_iter()
        .filter_map(|index| {
            let name = open.regions.get(index)?.name.clone();
            let &(number, _) = opened.regions.get(index)?;
            let state = is_state_region(open, index);
            Some((index, number, name, groups_of(open, index), state))
        })
        .collect())
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

/// Resizes the regions each scheduler's queue pair uses to the units the session states each
/// serves, within the numbering the buffer pool holds for it, and returns each region's number
/// with the units it reached. A device region of the KV planes or of the state units serves the
/// epoch's capacity, from the numbering the epoch's own division stated (`StorageClient::open`);
/// every other region serves what the epoch's layout asked for, or the fewer a region of a warm
/// media rank's slots kept (`StorageClient::keep_units`). A grow takes what its space's free
/// ranges hold and a shrink frees what its blocks let it; either may stop short, and a scheduler
/// whose shrink stopped short shrinks the region again as the units held past it come free.
/// Every region is asked before any reply is waited for.
///
/// The session asks for no unit of a region whose memory the ranks create once it is open, so
/// this is how a promoted engine's regions ([`promote`]), and a woken engine's, reach the
/// segments the head granted. A region an earlier epoch opened keeps that epoch's numbering and
/// units at `Open`, so this is how the epoch's own statement of its device cache
/// (`kv_cache_config.max_tokens`) and of its host tier (`kv_cache_config.host_gib`) takes effect
/// on a region of the same layout (`specs/buffer-pool.md` §14.1), a shrink to fewer units and a
/// grow back alike, once the epoch is promoted.
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
                let Ok(index) = index_of(opened, number) else {
                    continue;
                };
                let Some(stated) = open.regions.get(index) else {
                    continue;
                };
                if stated.tier == Tier::Remote {
                    continue;
                }
                // The units the region serves, never past the numbering it shares with the
                // epochs that named it before: a region of the KV planes reaches them as far as
                // the segments granted to its space go.
                let numbered = stated.limit.saturating_sub(stated.reserved);
                targets.push((scheduler, number, stated.units.min(numbered)));
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
