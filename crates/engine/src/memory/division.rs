//! The division of a GPU's quota among a rank's storages, computed when the rank starts.
//!
//! A deployment may use `mem_fraction_static` of a GPU: its quota, which the host's Storage
//! Server keeps and which every epoch that maps the GPU's storage shares. A rank divides what
//! the quota leaves it before it runs anything on the device, from figures that are stated or
//! measured by then:
//!
//! | Region | Bytes |
//! | --- | --- |
//! | Weights | The checkpoint's, less what the server retains of them already |
//! | Graph pool: one region per runner | Each runner's profile-run peak (`Graphs::peak`) |
//! | Scratch pool | Each runner's shadow region, the scratch page set of a causal runner, and the state snapshots of a model that keeps recurrent state |
//! | Request slots | What one request holds on the rank (a causal runner's slot state with its page-table and recurrent-state rows, a denoising rank's trajectory row and the products a request takes in its buffer), times the request slots |
//! | Cache planes | What remains, in whole units |
//!
//! The functions are the steps of one computation, in this order:
//!
//! 1. [`process_bytes`] measures what the process holds on the device beside its pools and the
//!    server's storages: the CUDA context, the loaded kernels, the communicators.
//! 2. [`unreserved`] states the bytes of the quota the rank may divide: the limit less every
//!    reservation of the deployment's epochs and less the measured process.
//! 3. [`pool_bytes`] states what the graph pool and the scratch pool map for the runners.
//! 4. [`remainder`] takes every region but the cache out of the unreserved bytes and refuses,
//!    naming the region and the shortfall, when one does not fit.
//! 5. [`request_slots`] gives the request slots what the regions that do not grow with them and
//!    the cache's fewest units leave: the deployment's `max_running_requests`, or as many as
//!    fit when fewer do.
//! 6. [`cache_units`] gives the cache planes the remainder in whole units and agrees the count
//!    over the rank's group, so every rank of a group schedules against the same count.
//!
//! No step uses a fraction of the device or an allowance: a figure is a profile run's, an
//! owner's, the checkpoint's, the server's or a measurement. The input buffers of the runners
//! are page-locked host memory (`Graphs::input_peak` per slot) and take none of the quota.
//!
//! ## The shadow region's bound
//!
//! A paired forward launches every call of one operator whose kernel changed twice: the new
//! kernel into twins carved from the runner's shadow region (`ops::Op::shadow`), the serving
//! kernel into the real outputs. Each call is compared as soon as both kernels ran it: its pairs
//! are copied to page-locked host memory right behind the serving kernel, before any later
//! launch writes over the real outputs, and compared there (`Graphs::shadow`). The region
//! therefore holds one call's twins at a time: the call's outputs and the operands it writes in
//! place, each from an `ops::twins::ALIGNMENT` boundary.
//!
//! The figure is the profile run's: each launch of a recorded launch table states the bytes its
//! shadow carves (`ops::Call::shadow_bytes`, `ops::Context::record_launches`), the request
//! slots' state a commit writes and the outputs a speculating runner keeps included, and
//! `Graphs::shadow_peak` takes the largest over every profiled forward. A paired forward
//! computes over other pages than the profiled one: its cache writes write the scratch page
//! set's planes of one layer in place (`CausalLmRunner::cache_writes`), which the region holds
//! beside the largest call. A paired call that needs more than the region launches the serving
//! kernel alone and compares nothing: the update names the operator and the batch uncompared.
//!
//! The region is held from the rank's start, as the pools are, and not taken when an update is
//! prepared: a cache that fills the rank's share (no `kv_cache_config.max_tokens`) leaves no
//! memory to take then, and every update of the deployment compares its changed kernels on
//! served batches.
//!
//! ## The scratch page set
//!
//! A causal runner's paired forwards write their keys and values to a scratch page set in the
//! scratch pool, beside its shadow region (`KvCacheMemory::open_scratch`). The set is one set
//! the rows of a batch share, `ceil(T / page) + R` pages for the `T` tokens and `R` rows of the
//! runner's largest bucket (`CausalLmRunner::scratch_pages`), with a page table of a row per
//! page (`memory::cache::scratch::bytes`). The rows take its pages in order, each the pages of
//! the context it reads, so every batch whose rows' contexts sum to at most `T` tokens fits; a
//! batch whose rows need more is not paired, and the update names it uncompared.
//!
//! ## After the division
//!
//! The pools reserve what they map as [`storage::Item::GraphPool`] ([`super::graph_pool`]).
//! The division measures the process before the warm-up, which adds instantiated graphs and the
//! libraries' state to it. A text rank creates its cache's segments the quota holds now before
//! the warm-up, whose growth lies outside the quota meanwhile, and what fills a share last is
//! settled once the rank is warm, from [`process_bytes`] measured again: a media rank keeps the
//! request slots whose storages fit beside its warm process ([`excess`]) and gives the storages
//! of the rest back; a text rank gives back the tail segments of its cache whose bytes cover the
//! same excess; then the process's first rank on each device reserves [`storage::Item::Process`]
//! at that figure, the process's whole there, and its other ranks on the device reserve none, each
//! with whatever a profiling session's collectors hold, through
//! [`StorageClient::reserve`](super::client::StorageClient::reserve). The growth is measured,
//! never estimated.
//!
//! A region is sized by the profile run of the generation that serves. A generation whose
//! forwards allocate more than the region holds is refused at its capture by the context's
//! allocator; no room is kept for it.

use platform::{DeviceId, Nvml};
use storage::{Item, Quota, Scope};

use crate::parallel::host::HostGroup;
use crate::worker::{Error, Result};

/// `bytes` as a device pool maps them: whole allocation granules of `granularity` bytes, and
/// at least one, since a mapping of no bytes still takes a granule.
pub(crate) fn granules(bytes: u64, granularity: u64) -> u64 {
    bytes.max(1).next_multiple_of(granularity.max(1))
}

/// The device bytes this process holds on `device` beside its pools and the storages the
/// Storage Server retains: its CUDA context, the kernels it loaded and its communicators.
///
/// The figure is NVML's usage of the process, less the segments the process created for the
/// server and keeps the creating handles of (`platform::created_bytes`), which NVML counts
/// with the process, and less the `pools` bytes the graph and scratch pools of every rank of the
/// process on the device map ([`GraphPool::mapped`](super::graph_pool::GraphPool::mapped)), which
/// the quota holds as theirs. `unattributed` is the bytes of communication buffers NVML
/// attributes to no process, as their owners state them. The figure is the process's whole on
/// the device, however many ranks it holds there.
///
/// The caller measures after everything the process maps outside its pools exists, and waits
/// for its streams first, so the figure counts the work it issued.
///
/// # Errors
///
/// [`Error::Unsupported`] when NVML attributes no usage on the device to this process, as
/// inside a PID namespace NVML does not report; the driver's error otherwise.
pub(crate) fn process_bytes(
    nvml: &Nvml,
    device: DeviceId,
    pools: u64,
    unattributed: u64,
) -> Result<u64> {
    let process = nvml.process_storage(device).map_err(|error| match error {
        platform::Error::ProcessNotFound { .. } => Error::Unsupported(format!(
            "NVML reports no device storage for this process on {device}"
        )),
        error => Error::Cuda(error),
    })?;

    let retained = platform::created_bytes(device);
    Ok(process.saturating_sub(retained).saturating_sub(pools) + unattributed)
}

/// The bytes of a GPU's quota a starting rank of `scope` and `epoch` may divide.
///
/// `quota` is the deployment's quota of the GPU as the Storage Server states it
/// ([`StorageClient::usage`](super::client::StorageClient::usage)) after the epoch stated its
/// limit. Its usage counts every epoch that maps the GPU: weights and cache segments once
/// however many epochs map them, each epoch's pools and process, and the Transfer Server's
/// context, which is counted whether or not the server holds it now. Two kinds of reservation
/// are the rank's to take again and count as free:
///
/// - what a lost process of the same scope and epoch reserved for its pools and itself, which
///   the server keeps for the process that replaces it;
/// - `kept`, the bytes of the exclusive storages a predecessor left that the rank's owners
///   take again or release ([`StorageClient::scope_bytes`](super::client::StorageClient::scope_bytes)).
///
/// `process` is [`process_bytes`], and `free` the bytes the device has free now: the rank
/// divides no more than the device can map, whatever the quota allows.
///
/// # Errors
///
/// [`Error::State`] when no epoch stated the GPU's limit.
pub(crate) fn unreserved(
    quota: &Quota,
    scope: &Scope,
    epoch: u64,
    kept: u64,
    process: u64,
    free: u64,
) -> Result<u64> {
    let limit = quota.limit.ok_or(Error::State(
        "the GPU's quota states no limit: the epoch states its fraction first",
    ))?;

    let own = quota.bytes(|reservation| {
        reservation.item.of_process()
            && reservation.scope.as_ref() == Some(scope)
            && reservation.epochs == [epoch]
    });
    let transfers = unheld_context(quota);

    let reserved = quota
        .usage()
        .saturating_sub(own)
        .saturating_sub(kept)
        .saturating_add(transfers)
        .saturating_add(process);

    // The storages the rank takes again are mapped already, so the device's free bytes do not
    // count them.
    Ok(limit
        .saturating_sub(reserved)
        .min(free.saturating_add(kept)))
}

/// The bytes of the Transfer Server's CUDA context on the GPU of `quota` that the quota counts
/// but does not reserve now: the context's measured bytes while the server holds no context
/// there, which it reserves again before its next task on the GPU and which the quota refuses
/// once a storage took that room; zero while it holds the context, whose reservation is in the
/// quota's usage.
pub(crate) fn unheld_context(quota: &Quota) -> u64 {
    let held = quota
        .reservations
        .iter()
        .any(|reservation| reservation.item == Item::TransferContext);

    if held { 0 } else { quota.transfer_context }
}

/// The bytes by which the reservations of `quota` would pass its limit once the process of
/// `epoch` reserves its warm process of `process` bytes on the GPU ([`process_bytes`], measured
/// after the warm-up): the quota's usage, with the Transfer Server's context, which the quota
/// counts whether or not the server holds it now, and `process` in place of any process a rank
/// of the epoch reserved there before. Zero when they fit.
///
/// The ranks an epoch places on one GPU are threads of one main process, which NVML states
/// whole, so the figure is the process's on the GPU and its reservations there, whichever of
/// its ranks' scopes they are stated under, are the ones it replaces.
///
/// # Errors
///
/// [`Error::State`] when no epoch stated the GPU's limit.
pub(crate) fn excess(quota: &Quota, epoch: u64, process: u64) -> Result<u64> {
    let limit = quota.limit.ok_or(Error::State(
        "the GPU's quota states no limit: the epoch states its fraction first",
    ))?;

    let reserved = quota
        .bytes(|reservation| reservation.item == Item::Process && reservation.epochs == [epoch]);
    let transfers = unheld_context(quota);

    Ok(quota
        .usage()
        .saturating_sub(reserved)
        .saturating_add(transfers)
        .saturating_add(process)
        .saturating_sub(limit))
}

/// The bytes of `quota` that other epochs of the deployment hold and `epoch` does not: the
/// storages, pools and processes of an epoch that serves beside it, which return to the quota
/// and to the device once that epoch ends.
///
/// What `epoch` shares with them (weights both map) is not among them, nor are the Transfer
/// Server's items, which no epoch holds. Passed to [`unreserved`] as storage the rank counts as
/// free, it gives what the rank may divide as if its epoch served the GPU alone: the size a
/// device KV region of a layout new to the quota grows to as the serving epoch drains
/// (`specs/updates.md` §3.7).
pub(crate) fn held_elsewhere(quota: &Quota, epoch: u64) -> u64 {
    quota
        .bytes(|reservation| !reservation.epochs.is_empty() && !reservation.epochs.contains(&epoch))
}

/// The bytes the graph pool and the scratch pool of a device map for its runners, in that
/// order.
///
/// `regions` are the runners' profile-run peaks, each mapped as a region of its own;
/// `shadows` are the runners' shadow regions (the module documentation states their bound),
/// zero for a runner that is never shadowed; `snapshots` is the bytes of the state snapshots
/// a model with recurrent state copies units through, zero without one. Every mapping takes
/// whole granules of `granularity` bytes.
pub(crate) fn pool_bytes(
    granularity: u64,
    regions: &[u64],
    shadows: &[u64],
    snapshots: u64,
) -> (u64, u64) {
    let graph = regions
        .iter()
        .map(|bytes| granules(*bytes, granularity))
        .sum();

    let scratch = shadows
        .iter()
        .chain(std::iter::once(&snapshots))
        .filter(|bytes| **bytes > 0)
        .map(|bytes| granules(*bytes, granularity))
        .sum();

    (graph, scratch)
}

/// Takes the rank's regions out of the `unreserved` bytes of its GPU's quota and returns what
/// remains for the cache planes.
///
/// `regions` are `(name, bytes)` of the storages that do not grow with the request slots: the
/// weights the load retains, the graph pool, the scratch pool, and what any other owner
/// states. `slot_regions` are `(name, bytes per request slot)` of the storages that do, such
/// as the encoder cache and the latent buffers, each taken `slots` times.
///
/// # Errors
///
/// [`Error::Unsupported`] naming the first region the quota cannot hold, with the bytes it
/// needs, the bytes left for it and the shortfall.
pub(crate) fn remainder(
    unreserved: u64,
    regions: &[(&str, u64)],
    slots: u64,
    slot_regions: &[(&str, u64)],
) -> Result<u64> {
    let fixed = regions.iter().map(|(name, bytes)| (*name, Some(*bytes)));
    let by_slot = slot_regions
        .iter()
        .map(|(name, bytes)| (*name, bytes.checked_mul(slots)));

    let mut left = unreserved;
    for (name, bytes) in fixed.chain(by_slot) {
        left = bytes
            .and_then(|bytes| left.checked_sub(bytes))
            .ok_or_else(|| {
                // A slot region whose bytes over every slot overflow has no figure to name.
                let (needed, shortfall) = bytes.map_or_else(
                    || ("more than 2^64".to_owned(), "more than 2^64".to_owned()),
                    |bytes| (bytes.to_string(), (bytes - left).to_string()),
                );

                Error::Unsupported(format!(
                    "the GPU's quota cannot hold the rank's {name}: it needs {needed} bytes and \
                     {left} of the {unreserved} unreserved bytes are left for it, {shortfall} \
                     short"
                ))
            })?;
    }

    Ok(left)
}

/// The request slots a rank holds: the most, up to `stated` (the deployment's
/// `max_running_requests`), whose storages fit in `left`, the bytes the rank's share of its GPU
/// leaves once its weights, its graph and scratch pools and the fewest units of its cache are
/// taken out. `storage(slots)` is the bytes the storages the rank keeps per request slot take
/// for `slots` slots, which grows with them; a rank of no slot runs no request, so one slot is
/// the least it starts with.
///
/// # Errors
///
/// [`Error::Unsupported`] when `left` holds no slot, naming the bytes one slot needs; what
/// `storage` fails with.
pub(crate) fn request_slots<E: From<Error>>(
    stated: u32,
    left: u64,
    storage: impl Fn(u32) -> std::result::Result<u64, E>,
) -> std::result::Result<u32, E> {
    let one = storage(1)?;
    if one > left {
        return Err(E::from(Error::Unsupported(format!(
            "the rank's share of its GPU holds no request slot: one slot needs {one} bytes, and \
             {left} are left once its weights, its graph and scratch pools and the fewest units \
             of its cache are taken out, {} short",
            one - left
        ))));
    }

    // The storages grow with the slots, so the slots that fit are `1..=held`: a bisection over
    // `held` in `low..=high`, of which `low` fits.
    let (mut low, mut high) = (1, stated.max(1));
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if storage(middle)? <= left {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Ok(low)
}

/// The units of the rank's cache planes: what `remainder` holds in whole units of
/// `unit_bytes`, agreed over `group` as the smallest count any of its ranks holds.
///
/// `least` is the fewest units a cache serves with: the padding unit and one page of every
/// cache group. `capacity` is the units of a configured token capacity
/// (`kv_cache_config.max_tokens`), which sizes the planes instead of the remainder and is
/// refused, not trimmed, when the remainder cannot hold it.
///
/// A rank that maps the planes of a region another epoch created divides nothing for them: the
/// quota reserves them already, and its count is the region's segment table's.
///
/// # Errors
///
/// [`Error::Invalid`] for a unit of no bytes, a cache of no units, or a count past the unit
/// index range; [`Error::Unsupported`] with the shortfall when the remainder holds fewer than
/// the units required; what the group's reduction fails with.
pub(crate) fn cache_units(
    remainder: u64,
    unit_bytes: u64,
    least: u64,
    capacity: Option<u64>,
    group: &HostGroup,
) -> Result<u32> {
    if unit_bytes == 0 || least == 0 {
        return Err(Error::Invalid(
            "a cache's unit bytes and its fewest units must be positive".into(),
        ));
    }

    let held = remainder / unit_bytes;
    let required = capacity.map_or(least, |units| units.max(least));
    if held < required {
        let shortfall = required
            .saturating_mul(unit_bytes)
            .saturating_sub(remainder);

        return Err(Error::Unsupported(format!(
            "the GPU's quota cannot hold the rank's cache planes: {required} units of \
             {unit_bytes} bytes are required and {remainder} bytes remain, {shortfall} short"
        )));
    }

    let own = if capacity.is_some() { required } else { held };
    let own = i64::try_from(own)
        .map_err(|_| Error::Invalid("the cache exceeds the unit index range".into()))?;

    let agreed = group.all_reduce_min(&[own])?[0];
    u32::try_from(agreed)
        .map_err(|_| Error::Invalid("the cache exceeds the unit index range".into()))
}

/// The calls of one request a rank holds unresolved at once: `depth` submitted batches of up
/// to `calls` calls each, and never fewer than a producer and its consumer when one batch can
/// hold both. What a rank reports as `crate::batch::Capacity::call_window`.
///
/// # Errors
///
/// [`Error::Invalid`] when either bound is zero.
pub fn call_window(depth: usize, calls: usize) -> Result<usize> {
    if depth == 0 || calls == 0 {
        return Err(Error::Invalid(
            "the window of unresolved calls requires positive bounds".into(),
        ));
    }

    Ok((depth * calls).min(depth.max(2)))
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;
    use std::time::Duration;

    use storage::Reservation;

    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    /// The allocation granularity of the devices the tests divide.
    const GRANULE: u64 = 2 * MIB;

    fn scope(rank: u32) -> Scope {
        Scope {
            deployment: "model".into(),
            group: "text".into(),
            rank,
        }
    }

    fn reservation(item: Item, scope: Option<Scope>, epochs: &[u64], bytes: u64) -> Reservation {
        Reservation {
            item,
            scope,
            epochs: epochs.to_vec(),
            segment: None,
            bytes,
        }
    }

    /// The deployment's quota of a GPU with `limit` bytes and `reservations`.
    fn quota(limit: u64, reservations: Vec<Reservation>) -> Quota {
        Quota {
            gpu: [0; 16],
            deployment: "model".into(),
            total: 2 * limit,
            limit: Some(limit),
            reservations,
            host_link_bytes_per_second: None,
            transfer_context: 0,
        }
    }

    /// Runs `body` on every member of a loopback group of `size` ranks, one thread each.
    fn group<T: Send>(size: usize, body: impl Fn(HostGroup) -> T + Sync) -> Vec<T> {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let mut listener = Some(listener);

        std::thread::scope(|threads| {
            let handles: Vec<_> = (0..size)
                .map(|rank| {
                    let listener = listener.take().filter(|_| rank == 0);
                    let address = address.as_str();
                    let body = &body;

                    threads.spawn(move || {
                        let timeout = Duration::from_secs(30);
                        body(HostGroup::connect(address, listener, rank, size, timeout).unwrap())
                    })
                })
                .collect();

            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        })
    }

    /// Slots of 3 MiB each beside 1 MiB that does not grow with them, such as the padding row
    /// of a page table.
    fn storage(slots: u32) -> Result<u64> {
        Ok(MIB + u64::from(slots) * 3 * MIB)
    }

    #[test]
    fn a_rank_holds_the_stated_slots_or_as_many_as_its_share_leaves_room_for() {
        // Room for every stated slot: the deployment's bound.
        assert_eq!(request_slots(8, GIB, storage).unwrap(), 8);
        // Room for five slots and most of a sixth: five, whichever bound is stated above it.
        let five = storage(5).unwrap();
        assert_eq!(request_slots(8, five + 2 * MIB, storage).unwrap(), 5);
        assert_eq!(request_slots(128, five, storage).unwrap(), 5);
        // Exactly one slot's storages.
        assert_eq!(request_slots(8, 4 * MIB, storage).unwrap(), 1);
    }

    #[test]
    fn a_share_that_holds_no_slot_names_what_one_slot_needs() {
        let error: Error = request_slots(8, 4 * MIB - 1, storage).unwrap_err();
        let message = error.to_string();

        assert!(matches!(error, Error::Unsupported(_)), "{message}");
        assert!(
            message.contains(&format!("one slot needs {}", 4 * MIB)),
            "{message}"
        );
        assert!(message.contains("1 short"), "{message}");
    }

    #[test]
    fn a_pool_maps_every_region_in_whole_granules() {
        // Two runners' regions, of one byte past a granule and of three granules exactly; one
        // runner is shadowed, and the model keeps half a granule of state snapshots.
        let (graph, scratch) = pool_bytes(
            GRANULE,
            &[GRANULE + 1, 3 * GRANULE],
            &[GRANULE / 4, 0],
            GRANULE / 2,
        );

        assert_eq!(graph, 2 * GRANULE + 3 * GRANULE);
        // The shadow region and the snapshots are one mapping each; the runner that is never
        // shadowed maps nothing.
        assert_eq!(scratch, 2 * GRANULE);

        // A rank whose model keeps no state and whose runners are never shadowed maps no
        // scratch.
        assert_eq!(pool_bytes(GRANULE, &[GRANULE], &[0], 0), (GRANULE, 0));
    }

    /// Once warm, a process is reserved beside what its GPU's quota holds: the excess is what the
    /// reservations would pass the limit by, the process its epoch reserved there before being
    /// replaced and the Transfer Server's context counted whether or not the server holds it.
    #[test]
    fn a_warm_process_passes_its_quota_by_what_it_adds_past_the_limit() {
        let rank = scope(0);
        let mut quota = quota(
            70 * GIB,
            vec![
                reservation(Item::Weights, None, &[1], 40 * GIB),
                reservation(Item::GraphPool, Some(rank.clone()), &[1], 24 * GIB),
            ],
        );

        // A process of five GiB fits the six left; one of seven passes the limit by one.
        assert_eq!(excess(&quota, 1, 5 * GIB).unwrap(), 0);
        assert_eq!(excess(&quota, 1, 7 * GIB).unwrap(), GIB);

        // The process the epoch reserved before is replaced, under whichever of its ranks'
        // scopes it was stated: its ranks on the GPU share the one process.
        quota
            .reservations
            .push(reservation(Item::Process, Some(scope(1)), &[1], 3 * GIB));
        assert_eq!(excess(&quota, 1, 7 * GIB).unwrap(), GIB);

        // Another epoch's process is another process, which stays.
        assert_eq!(excess(&quota, 2, 7 * GIB).unwrap(), 4 * GIB);

        // The Transfer Server's context counts before the server holds it.
        quota.transfer_context = GIB;
        assert_eq!(excess(&quota, 1, 7 * GIB).unwrap(), 2 * GIB);
    }

    #[test]
    fn a_first_epoch_divides_its_limit_less_what_it_holds_already() {
        let weights = reservation(Item::Weights, None, &[1], 20 * GIB);
        let quota = quota(70 * GIB, vec![weights]);

        // The limit less the weights the server retains and the process the rank measured.
        let unreserved = unreserved(&quota, &scope(0), 1, 0, GIB, 100 * GIB).unwrap();
        assert_eq!(unreserved, 70 * GIB - 20 * GIB - GIB);
    }

    #[test]
    fn a_second_epoch_divides_what_the_first_left_of_the_shared_quota() {
        // The first epoch serves: its weights, which the second maps too and the quota
        // reserves once, its cache segments, its pools and its process.
        let first = scope(0);
        let quota = quota(
            70 * GIB,
            vec![
                reservation(Item::Weights, None, &[1, 2], 20 * GIB),
                reservation(Item::CacheSegment, Some(first.clone()), &[1], 30 * GIB),
                reservation(Item::GraphPool, Some(first.clone()), &[1], 4 * GIB),
                reservation(Item::Process, Some(first), &[1], GIB),
            ],
        );

        let unreserved = unreserved(&quota, &scope(0), 2, 0, GIB, 100 * GIB).unwrap();
        assert_eq!(unreserved, 70 * GIB - (20 + 30 + 4 + 1) * GIB - GIB);

        // The second epoch's regions come out of that, never out of the whole limit: its
        // weights are reserved already, so it takes its pools and gives the cache the rest.
        let (graph, scratch) = pool_bytes(GRANULE, &[4 * GIB], &[GIB], 0);
        let rest = remainder(
            unreserved,
            &[
                ("weights", 0),
                ("graph pool", graph),
                ("scratch pool", scratch),
            ],
            0,
            &[],
        )
        .unwrap();
        assert_eq!(rest, 14 * GIB - 4 * GIB - GIB);
    }

    #[test]
    fn an_epoch_of_another_layout_divides_as_if_it_served_alone_once_the_first_ends() {
        // The first epoch serves its cache, pools and process beside the second, which maps
        // the same weights; the Transfer Server holds its context against every epoch.
        let first = scope(0);
        let mut quota = quota(
            70 * GIB,
            vec![
                reservation(Item::Weights, None, &[1, 2], 20 * GIB),
                reservation(Item::CacheSegment, Some(first.clone()), &[1], 30 * GIB),
                reservation(Item::GraphPool, Some(first.clone()), &[1], 4 * GIB),
                reservation(Item::Process, Some(first), &[1], GIB),
                reservation(Item::TransferContext, None, &[], GIB),
            ],
        );
        quota.transfer_context = GIB;

        // What only the first epoch holds returns once it ends; the shared weights and the
        // Transfer Server's context stay.
        let elsewhere = held_elsewhere(&quota, 2);
        assert_eq!(elsewhere, (30 + 4 + 1) * GIB);

        // Now the second divides what the first left; alone, it divides the whole limit less
        // the weights, the context and its own process, and the device's free bytes grow by
        // what the first epoch gives back.
        let now = unreserved(&quota, &scope(0), 2, 0, GIB, 16 * GIB).unwrap();
        assert_eq!(now, 70 * GIB - (20 + 35 + 1) * GIB - GIB);
        let alone = unreserved(&quota, &scope(0), 2, elsewhere, GIB, 16 * GIB).unwrap();
        assert_eq!(alone, 70 * GIB - (20 + 1) * GIB - GIB);

        // The first epoch holds nothing elsewhere of the second.
        assert_eq!(held_elsewhere(&quota, 1), 0);
    }

    #[test]
    fn a_restarted_rank_takes_its_lost_process_s_reservations_and_storages_again() {
        let own = scope(0);
        let quota = quota(
            70 * GIB,
            vec![
                reservation(Item::Weights, None, &[1], 20 * GIB),
                reservation(Item::Exclusive, Some(own.clone()), &[1], 6 * GIB),
                reservation(Item::GraphPool, Some(own.clone()), &[1], 4 * GIB),
                reservation(Item::Process, Some(own), &[1], 2 * GIB),
                // Another rank's process is not this rank's to take.
                reservation(Item::Process, Some(scope(1)), &[1], 2 * GIB),
            ],
        );

        // The lost process's pools and process, and the 6 GiB of exclusive storages it left,
        // count as free; the weights, the other rank and the measured process do not.
        let unreserved = unreserved(&quota, &scope(0), 1, 6 * GIB, GIB, 100 * GIB).unwrap();
        assert_eq!(unreserved, 70 * GIB - 20 * GIB - 2 * GIB - GIB);
    }

    #[test]
    fn the_transfer_server_s_context_is_counted_once_whether_or_not_it_is_held() {
        let mut idle = quota(70 * GIB, Vec::new());
        idle.transfer_context = GIB;
        assert_eq!(
            unreserved(&idle, &scope(0), 1, 0, 0, 100 * GIB).unwrap(),
            69 * GIB
        );

        let mut held = quota(
            70 * GIB,
            vec![reservation(Item::TransferContext, None, &[], GIB)],
        );
        held.transfer_context = GIB;
        assert_eq!(
            unreserved(&held, &scope(0), 1, 0, 0, 100 * GIB).unwrap(),
            69 * GIB
        );
    }

    #[test]
    fn a_rank_divides_no_more_than_the_device_has_free() {
        let quota = quota(70 * GIB, Vec::new());

        // Another deployment holds the device down to 10 GiB free; the storages the rank
        // takes again are mapped already and add to what it may divide.
        assert_eq!(
            unreserved(&quota, &scope(0), 1, 0, 0, 10 * GIB).unwrap(),
            10 * GIB
        );
        assert_eq!(
            unreserved(&quota, &scope(0), 1, 2 * GIB, 0, 10 * GIB).unwrap(),
            12 * GIB
        );
    }

    #[test]
    fn a_quota_without_a_limit_divides_nothing() {
        let mut quota = quota(70 * GIB, Vec::new());
        quota.limit = None;

        assert!(matches!(
            unreserved(&quota, &scope(0), 1, 0, 0, GIB),
            Err(Error::State(_))
        ));
    }

    #[test]
    fn the_regions_are_taken_in_order_and_the_slot_regions_once_per_slot() {
        let rest = remainder(
            40 * GIB,
            &[
                ("weights", 20 * GIB),
                ("graph pool", 4 * GIB),
                ("scratch pool", GIB),
            ],
            8,
            &[("encoder cache", 64 * MIB), ("latent buffers", 192 * MIB)],
        )
        .unwrap();

        assert_eq!(rest, 40 * GIB - 25 * GIB - 8 * 256 * MIB);
    }

    #[test]
    fn a_quota_too_small_for_a_region_is_refused_by_name_with_the_shortfall() {
        // The weights fit and leave 2 GiB; the graph pool needs 3 GiB.
        let error = remainder(
            22 * GIB,
            &[("weights", 20 * GIB), ("graph pool", 3 * GIB)],
            0,
            &[],
        )
        .unwrap_err();

        let message = error.to_string();
        assert!(matches!(error, Error::Unsupported(_)), "{message}");
        assert!(message.contains("graph pool"), "{message}");
        assert!(message.contains(&format!("needs {}", 3 * GIB)), "{message}");
        assert!(message.contains(&format!("{GIB} short")), "{message}");

        // A slot region is refused the same way, at the bytes all its slots take.
        let error = remainder(GIB, &[], 8, &[("latent buffers", 256 * MIB)]).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("latent buffers"), "{message}");
        assert!(message.contains(&format!("{GIB} short")), "{message}");
    }

    #[test]
    fn the_cache_takes_the_remainder_in_whole_units() {
        let group = HostGroup::single();

        // Ten units and most of an eleventh fit: the cache holds ten.
        let units = cache_units(10 * MIB + MIB - 1, MIB, 3, None, &group).unwrap();
        assert_eq!(units, 10);

        // A remainder of exactly the fewest units serves with them.
        assert_eq!(cache_units(3 * MIB, MIB, 3, None, &group).unwrap(), 3);
    }

    #[test]
    fn a_remainder_below_the_fewest_units_is_refused_with_the_shortfall() {
        let error = cache_units(2 * MIB + 1, MIB, 3, None, &HostGroup::single()).unwrap_err();

        let message = error.to_string();
        assert!(matches!(error, Error::Unsupported(_)), "{message}");
        assert!(message.contains("cache planes"), "{message}");
        assert!(message.contains(&format!("{} short", MIB - 1)), "{message}");
    }

    #[test]
    fn a_configured_capacity_sizes_the_cache_and_is_refused_when_it_does_not_fit() {
        let group = HostGroup::single();

        // The remainder holds a hundred units; the configuration asks for forty.
        assert_eq!(
            cache_units(100 * MIB, MIB, 3, Some(40), &group).unwrap(),
            40
        );
        // A capacity below the fewest units a cache serves with is raised to them.
        assert_eq!(cache_units(100 * MIB, MIB, 3, Some(1), &group).unwrap(), 3);

        // A capacity the remainder cannot hold is refused, not trimmed.
        let error = cache_units(100 * MIB, MIB, 3, Some(120), &group).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("120 units"), "{message}");
        assert!(
            message.contains(&format!("{} short", 20 * MIB)),
            "{message}"
        );
    }

    #[test]
    fn a_cache_without_units_or_unit_bytes_is_refused() {
        let group = HostGroup::single();

        assert!(matches!(
            cache_units(GIB, 0, 3, None, &group),
            Err(Error::Invalid(_))
        ));
        assert!(matches!(
            cache_units(GIB, MIB, 0, None, &group),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn the_ranks_of_a_group_agree_the_smallest_count_any_of_them_holds() {
        // Rank 0 has 40 units' worth left, rank 1 has 25 and rank 2 has 32.
        let remainders = [40 * MIB, 25 * MIB + 7, 32 * MIB];

        let counts = group(3, |group| {
            cache_units(remainders[group.rank()], MIB, 3, None, &group).unwrap()
        });

        assert_eq!(counts, [25, 25, 25]);
    }

    #[test]
    fn the_window_of_unresolved_calls_holds_a_producer_and_its_consumer() {
        // One batch of sixteen calls can hold a producer and its consumer.
        assert_eq!(call_window(1, 16).unwrap(), 2);
        // One call per batch leaves one unresolved call per submitted batch.
        assert_eq!(call_window(4, 1).unwrap(), 4);
        assert_eq!(call_window(4, 16).unwrap(), 4);

        assert!(call_window(0, 16).is_err());
        assert!(call_window(1, 0).is_err());
    }
}
