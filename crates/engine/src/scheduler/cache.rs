//! What a scheduler knows of the cache (`specs/scheduler-design.md` §4.5), the decision half of
//! vLLM's block pool (`vllm/v1/core/block_pool.py`): the [`Cached`] row of every cached block it
//! knows of, in `Scheduler::cache` by [`Cached::id`] (vLLM's `BlockHashToBlockMap`), and per
//! region the eviction order of the blocks no request of the scheduler holds, least recently
//! used first, in `Scheduler::order` (the free block queue's order: `touch` takes a block out,
//! `free_blocks` puts it back last).
//!
//! Every completion read from the queue pair is applied here (`learn`), one dropped for a
//! forgotten owner too, since what it states of the buffer pool's cache holds all the same; a
//! request gives up the blocks it holds by `unpin`. Each change of a row is told to the policy
//! (`Cached`, `Hit`, `Freed`, `Encoded`, SD §5.2). At the end of a step `install_victims`
//! installs the head of each order by `Victims` and asks for a `List` when the buffer pool counts
//! evictable units the scheduler cannot account for. The table is exact for what this queue pair
//! commits and evicts; a wrong belief costs reuse, never a wrong block, since every request that
//! names a key is answered from the buffer pool's own cache.

use std::collections::HashSet;
use std::num::NonZeroU32;
use std::sync::OnceLock;
use std::time::Instant;

use scheduling_guest::{EncodingId, EncodingView, Event, Nanos, PageId, PageView};
use storage::Key;
use storage::runtime::{self, Cause, Completion, GroupId, Tier};

use super::Scheduler;

/// The tier bits of [`Cached::tiers`], as a `Listing` states them.
pub(super) const DEVICE: u8 = 1;
pub(super) const HOST: u8 = 2;
pub(super) const REMOTE: u8 = 4;

/// The low watermark: blocks of each class a running request holds beyond its next call (SD
/// §4.4).
pub(super) const LOW_WATERMARK: u32 = 1;

/// The tokens of a KV block when no group states the deployment's block size (BP §10).
const BLOCK_TOKENS: u32 = 64;

/// The bytes of a `Victims` record besides its keys (`storage::runtime`).
const VICTIMS_RECORD: usize = 14;

/// A cached block the scheduler knows of.
///
/// Invariant: it is in the order of each region that holds a copy of it on this scheduler's
/// side (a device region of its class in a group of `groups`, its class's host region) exactly
/// when `pins == 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cached {
    pub key: Key,
    pub class: u8,
    /// The tokens the block holds: the class's block size, or fewer for a request's tail; 0 when
    /// the scheduler learned only that the key is cached.
    pub tokens: u16,
    /// The tiers that hold a copy: [`DEVICE`], [`HOST`], [`REMOTE`].
    pub tiers: u8,
    /// The worker groups whose memory holds its device copy, one bit per group number.
    pub groups: u8,
    /// Requests of this scheduler holding it.
    pub pins: u32,
    /// Its last use on [`now_nanos`]'s clock: its rank in the eviction order.
    pub last_used: u64,
}

impl Cached {
    /// The number the scheduler knows a cached block by: its key's leading eight bytes,
    /// big-endian. It is the policy's `PageId` for a KV block and its `EncodingId` for an encoder
    /// output, so a policy's id finds its block in one lookup.
    pub fn id(key: &Key) -> u64 {
        let mut leading = [0; 8];
        leading.copy_from_slice(&key.0[..8]);
        u64::from_be_bytes(leading)
    }

    /// The block as the policy's page: one with a device copy is a device page, one on the host
    /// alone an offloaded page, as the system as built states them
    /// (`crates/scheduler/src/views.rs:793-803`).
    fn page(&self) -> PageView {
        PageView {
            id: PageId(Self::id(&self.key)),
            tokens: u32::from(self.tokens),
            last_used: Nanos(self.last_used),
            on_host: self.tiers & (DEVICE | HOST) == HOST,
            evictable: self.pins == 0,
        }
    }

    fn encoding(&self) -> EncodingView {
        EncodingView {
            id: EncodingId(Self::id(&self.key)),
            last_used: Nanos(self.last_used),
        }
    }
}

/// Nanoseconds on the scheduler's monotonic clock since its first reading in the process: the
/// clock of [`Cached::last_used`] and of every instant the policy is told (the guest's `Nanos`).
pub(super) fn now_nanos() -> u64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    let elapsed = ORIGIN.get_or_init(Instant::now).elapsed();
    u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the loop, admission, update and policy that call these are not in the crate yet"
    )
)]
impl Scheduler {
    /// Applies what a completion of `request` states of the cache (SD §4.5). A request answered
    /// `Copy` is narrowed to the keys the copy covers, for which the answer to its `Finish`
    /// speaks: the leading keys of a `Load`, the copied keys of a `Store`.
    pub(super) fn learn(&mut self, request: &mut runtime::Request, completion: &Completion) {
        use runtime::Request as R;
        let now = now_nanos();

        // The keys a completion lists as evicted left the regions its request took blocks from.
        let (taken, evicted) = match (&*request, completion) {
            (_, Completion::Allocated { runs, evicted })
            | (
                _,
                Completion::Copy {
                    to: runs, evicted, ..
                },
            ) => (runs.iter().map(|run| run.0).collect(), evicted.as_slice()),
            (R::Resize { region, .. }, Completion::Resized { evicted, .. }) => {
                (vec![*region], &evicted[..])
            }
            _ => (Vec::new(), &[][..]),
        };
        let mut taken: Vec<usize> = taken
            .into_iter()
            .filter_map(|number| self.region_index(number))
            .collect();
        taken.sort_unstable();
        taken.dedup();
        for (region, key) in taken
            .into_iter()
            .flat_map(|region| evicted.iter().map(move |key| (region, key)))
        {
            self.drop_copy(key, region);
        }

        if let (
            R::Load { keys, .. } | R::Store { keys, .. },
            Completion::Copy { keys: covered, .. },
        ) = (&mut *request, completion)
        {
            keys.retain(|key| covered.contains(key));
        }

        // The answer to a `Finish` is the result of its copy, with the copy's group as its stamp.
        let copied;
        let (request, copy) = match &*request {
            R::Finish { copy, after, .. } => match self.outstanding.get(&copy.0) {
                Some((_, request)) => {
                    copied = request.clone();
                    (&copied, Some(stamped_groups(after)))
                }
                None => return,
            },
            request => (request, None),
        };

        match (request, completion, copy) {
            (R::List, Completion::Listing { keys }, None) => self.listed(keys),
            (
                R::Commit {
                    owner,
                    class,
                    at,
                    key,
                    tokens,
                    after,
                },
                Completion::Committed,
                None,
            ) => {
                // Held by its request while the entry is at or after the table's `first`.
                let row = self.request_of(*owner);
                let held = row.and_then(|row| row.blocks.get(usize::from(*class)));
                let held = held.is_some_and(|blocks| blocks.first <= *at);
                self.hold(
                    key,
                    (*class, *tokens),
                    stamped_groups(after),
                    u32::from(held),
                    now,
                );
            }
            (
                R::Load {
                    owner, class, keys, ..
                },
                Completion::Loaded,
                Some(groups),
            ) => {
                let pins = u32::from(self.owners.contains_key(owner));
                for key in keys {
                    self.hold(key, (*class, 0), groups, pins, now);
                }
            }
            (
                R::Pin {
                    owner, class, keys, ..
                },
                Completion::Pinned { runs, host },
                None,
            ) => {
                // A returned admission's pins left with its owner's `Detach`.
                let Some((group, _)) = self
                    .request_of(*owner)
                    .and_then(|row| row.attachment(*owner))
                else {
                    return;
                };
                // A block has a run in each region of its class in the group.
                let per_block = self
                    .group(group)
                    .map_or(1, |found| found.regions(*class).len().max(1));
                let (held, rest) = keys.split_at((runs.len() / per_block).min(keys.len()));
                for key in held {
                    self.hold(key, (*class, 0), bit(group), 1, now);
                }
                // The keys after the pinned ones are not valid in the group; the first `host` of
                // them have a host copy, and the next one none.
                let host = usize::try_from(*host).unwrap_or(usize::MAX).min(rest.len());
                for (at, key) in rest.iter().enumerate() {
                    self.invalidate(key, bit(group));
                    let tiers = if at < host { HOST } else { 0 };
                    if at <= host {
                        self.update(key, Some((*class, 0)), |row| {
                            row.tiers = row.tiers & !HOST | tiers
                        });
                    }
                }
            }
            (
                R::Store {
                    keys,
                    tier: Tier::Host,
                    ..
                },
                Completion::Stored { .. },
                Some(_),
            ) => {
                for key in keys {
                    self.update(key, None, |row| row.tiers |= HOST);
                }
            }
            (
                R::Store {
                    tier: Tier::Remote, ..
                },
                Completion::Stored { keys },
                None,
            ) => {
                for key in keys {
                    self.update(key, None, |row| row.tiers |= REMOTE);
                }
            }
            (R::Prefetch { class, keys, .. }, Completion::Prefetched { found }, None) => {
                for key in keys
                    .iter()
                    .take(usize::try_from(*found).unwrap_or(usize::MAX))
                {
                    self.update(key, Some((*class, 0)), |row| row.tiers |= HOST);
                }
            }
            (R::Evict { .. }, Completion::Evicted { keys }, None) => {
                for key in keys {
                    self.invalidate(key, u8::MAX);
                }
            }
            // A key `cache` believed held that the buffer pool does not hold, or a host copy
            // another layout wrote that this one cannot read, is a miss from now on.
            (R::Clone { owner, key, .. }, Completion::Refused { cause: Cause::Key }, None) => {
                if let Some((group, _)) = self
                    .request_of(*owner)
                    .and_then(|row| row.attachment(*owner))
                {
                    self.invalidate(key, bit(group));
                }
            }
            (
                R::Restore { key, .. },
                Completion::Refused {
                    cause: Cause::Key | Cause::Class,
                },
                None,
            ) => {
                self.update(key, None, |row| row.tiers &= !HOST);
            }
            (
                R::Load { keys, .. },
                Completion::Refused {
                    cause: Cause::Class,
                },
                None,
            ) => {
                if let Some(key) = keys.first() {
                    self.update(key, None, |row| row.tiers &= !HOST);
                }
            }
            _ => {}
        }
    }

    /// `key`'s block is on the device in `groups`, held by `pins` more requests and used now.
    fn hold(&mut self, key: &Key, fresh: (u8, u16), groups: u8, pins: u32, now: u64) {
        self.update(key, Some(fresh), |row| {
            row.tiers |= DEVICE;
            row.groups |= groups;
            row.pins = row.pins.saturating_add(pins);
            row.last_used = now;
        });
    }

    /// A `Listing` states every cached key: a row it lists becomes what it states, keeping its
    /// holders and last use (a key new to `cache` was never used here and is first in its
    /// order); a row it does not list leaves `cache`.
    fn listed(&mut self, keys: &[(Key, u8, u16, u8, u8)]) {
        let listed: HashSet<u64> = keys.iter().map(|(key, ..)| Cached::id(key)).collect();
        let gone: Vec<Key> = self
            .cache
            .values()
            .map(|row| row.key)
            .filter(|key| !listed.contains(&Cached::id(key)))
            .collect();
        for key in &gone {
            self.update(key, None, |row| row.tiers = 0);
        }
        for &(key, class, tokens, tiers, groups) in keys {
            self.update(&key, Some((class, tokens)), |row| {
                if row.tiers == 0 {
                    row.last_used = 0;
                }
                *row = Cached {
                    class,
                    tokens,
                    tiers,
                    groups,
                    ..*row
                };
            });
        }
    }

    /// The copy of `key` in `region` (by index) left it: a host copy leaves the host tier, a
    /// device copy the groups whose blocks of the key's class lie there.
    fn drop_copy(&mut self, key: &Key, region: usize) {
        let Some(class) = self.cache.get(&Cached::id(key)).map(|row| row.class) else {
            return;
        };
        if self
            .regions
            .get(region)
            .is_some_and(|(_, stated)| stated.tier == Tier::Host)
        {
            self.update(key, None, |row| row.tiers &= !HOST);
            return;
        }
        let there = |group: &&super::WorkerGroup| {
            group
                .regions(class)
                .iter()
                .any(|&r| usize::from(r) == region)
        };
        let groups = self
            .groups
            .iter()
            .filter(there)
            .fold(0, |bits, group| bits | bit(group.id()));
        self.invalidate(key, groups);
    }

    /// `key`'s device copy is not valid in `groups`; it leaves the device once no group holds it.
    fn invalidate(&mut self, key: &Key, groups: u8) {
        self.update(key, None, |row| {
            row.groups &= !groups;
            if row.groups == 0 {
                row.tiers &= !DEVICE;
            }
        });
    }

    /// A request gave up the shared blocks of `keys`, in position order, by a `Release` or a
    /// `Detach` handed to the executors: those it holds by `Pinned`, `Loaded` or a held
    /// `Committed`, not those answered `Duplicate`. A block no request holds joins its orders as
    /// just used, the deepest key first in line, so that the prefix before it stays cached
    /// longest (vLLM's `free` frees a request's tail first).
    pub(super) fn unpin(&mut self, keys: &[Key]) {
        let now = now_nanos();
        for (offset, key) in (0u64..).zip(keys.iter().rev()) {
            self.update(key, None, |row| {
                row.pins = row.pins.saturating_sub(1);
                row.last_used = now.saturating_add(offset);
            });
        }
    }

    /// Applies `change` to the row of `key`, or to a new row of `fresh`'s class and tokens; a
    /// row left on no tier leaves `cache`. The orders follow the row, and the policy is told. A
    /// key unknown without `fresh`, or a second key under a known key's number (SD §4.5), is
    /// left alone.
    fn update(&mut self, key: &Key, fresh: Option<(u8, u16)>, change: impl FnOnce(&mut Cached)) {
        let id = Cached::id(key);
        let before = self.cache.get(&id).copied();
        let mut row = match (before, fresh) {
            (Some(row), _) if row.key == *key => row,
            (None, Some((class, tokens))) => {
                let last_used = now_nanos();
                Cached {
                    key: *key,
                    class,
                    tokens,
                    tiers: 0,
                    groups: 0,
                    pins: 0,
                    last_used,
                }
            }
            _ => return,
        };
        change(&mut row);
        let after = (row.tiers != 0).then_some(row);

        for (row, enter) in before
            .map(|row| (row, false))
            .into_iter()
            .chain(after.map(|row| (row, true)))
        {
            for (region, units) in self.placed(&row) {
                let (Some(order), Some(ordered)) =
                    (self.order.get_mut(region), self.ordered.get_mut(region))
                else {
                    continue;
                };
                if enter && order.insert((row.last_used, id)) {
                    *ordered = ordered.saturating_add(units);
                } else if !enter && order.remove(&(row.last_used, id)) {
                    *ordered = ordered.saturating_sub(units);
                }
            }
        }
        match after {
            Some(row) => self.cache.insert(id, row),
            None => self.cache.remove(&id),
        };
        self.tell(before, after);
    }

    /// The regions (by index) whose order holds `row`, each with the units the block takes
    /// there: none while a request holds it; else the device regions of its class in the groups
    /// that hold its device copy, and its class's host region.
    fn placed(&self, row: &Cached) -> Vec<(usize, u32)> {
        if row.pins > 0 {
            return Vec::new();
        }
        let devices = self
            .groups
            .iter()
            .filter(|group| row.tiers & DEVICE != 0 && row.groups & bit(group.id()) != 0);
        let class = self.classes.get(usize::from(row.class));
        let host = class
            .and_then(|class| class.host)
            .filter(|_| row.tiers & HOST != 0);
        let mut regions: Vec<u8> = devices
            .flat_map(|group| group.regions(row.class))
            .copied()
            .chain(host.map(|(host, _)| host))
            .collect();
        regions.sort_unstable();
        regions.dedup();
        regions
            .into_iter()
            .map(|region| {
                (
                    usize::from(region),
                    self.class_units(row.class, usize::from(region)),
                )
            })
            .collect()
    }

    /// The units a block of `class` takes in `region` (by index): a host copy's configured
    /// size, else the class's block size; 0 for a sized class, whose blocks state their own.
    fn class_units(&self, class: u8, region: usize) -> u32 {
        let Some(stated) = self.classes.get(usize::from(class)) else {
            return 0;
        };
        match stated.host {
            Some((host, units)) if usize::from(host) == region && units > 0 => units,
            _ => stated.units,
        }
    }

    /// Tells the policy how a row changed (SD §5.2). A KV block (of a class with a block size)
    /// that enters `cache`, or whose tier or evictability as the policy sees them changes, is
    /// `Cached`; one held by one more request and nothing else the policy sees is a `Hit`; one
    /// that leaves is `Freed`. An encoder output (of a sized class a group of the scheduler
    /// computes into device memory) that enters `cache` or is used again is `Encoded`. A stored
    /// trajectory, on the host alone, is no object of the policy's.
    fn tell(&mut self, before: Option<Cached>, after: Option<Cached>) {
        let Some(row) = after.or(before) else {
            return;
        };
        let id = Cached::id(&row.key);
        let seen = |row: Option<Cached>| row.map(|row| (row.page().on_host, row.pins == 0));
        let event = match (self.is_page(row.class), before, after) {
            (true, Some(old), Some(new)) if seen(before) == seen(after) => {
                if new.pins <= old.pins {
                    return;
                }
                Event::Hit { page: PageId(id) }
            }
            (true, _, Some(new)) => Event::Cached { page: new.page() },
            (true, _, None) => Event::Freed { page: PageId(id) },
            (false, _, Some(new))
                if self.is_encoding(new.class)
                    && before.is_none_or(|old| old.last_used != new.last_used) =>
            {
                Event::Encoded {
                    entry: new.encoding(),
                }
            }
            _ => return,
        };
        self.policy.events.push(event);
    }

    fn is_page(&self, class: u8) -> bool {
        self.classes
            .get(usize::from(class))
            .is_some_and(|stated| stated.units > 0)
    }

    fn is_encoding(&self, class: u8) -> bool {
        !self.is_page(class)
            && self
                .groups
                .iter()
                .any(|group| !group.regions(class).is_empty())
    }

    /// The tiers on which `cache` expects each of an admission's lookup keys of one class, read
    /// by an owner on `group`: [`DEVICE`] when its device copy is valid in `group`, [`HOST`] and
    /// [`REMOTE`] as `cache` holds them, 0 for a key it does not know. Memory's rule of the hit
    /// (SD §4.4) turns them into what the admission pins, loads, clones and prefetches.
    pub(super) fn expected_hit(&self, keys: &[Key], group: GroupId) -> Vec<u8> {
        let tiers = |key: &Key| {
            let row = self
                .cache
                .get(&Cached::id(key))
                .filter(|row| row.key == *key)?;
            let device = DEVICE * u8::from(row.groups & bit(group) != 0);
            Some(row.tiers & (device | HOST | REMOTE))
        };
        keys.iter().map(|key| tiers(key).unwrap_or(0)).collect()
    }

    /// Whether the block `id` names (a policy's `PageId` or `EncodingId`) is on the device with no
    /// request of this scheduler holding it: a victim the policy may name (SD §3.6).
    pub(super) fn cache_unpinned(&self, id: u64) -> bool {
        self.cache
            .get(&id)
            .is_some_and(|row| row.pins == 0 && row.tiers & DEVICE != 0)
    }

    /// Every block of `cache` as a policy's snapshot states it (SD §5.2): a `PageView` per KV
    /// block and an `EncodingView` per encoder output.
    pub(super) fn cache_views(&self) -> (Vec<PageView>, Vec<EncodingView>) {
        let (mut pages, mut encodings) = (Vec::new(), Vec::new());
        for row in self.cache.values() {
            if self.is_page(row.class) {
                pages.push(row.page());
            } else if self.is_encoding(row.class) {
                encodings.push(row.encoding());
            }
        }
        (pages, encodings)
    }

    /// At the end of a step (SD §3.1, §4.5), for each region of this queue pair whose order's
    /// head is not the one installed: `Victims { region, keys }` with the leading keys of the
    /// order whose units cover what one step's allocations can take there (the whole order, as
    /// far as a record holds it, while a drain shrinks the regions). Then `List`, when a region's
    /// `evictable` word counts more units than the scheduler can account for: blocks another
    /// queue pair or a closed session committed, which this queue pair's allocations evict only
    /// once a `Victims` names them.
    pub(super) fn install_victims(&mut self) {
        let shrinking = self.draining.as_ref().is_some_and(|(.., shrink)| *shrink);
        let fits = self.queue.max_record().saturating_sub(VICTIMS_RECORD) / size_of::<Key>();
        let regions: Vec<(u8, usize)> = self
            .registers
            .capacity
            .iter()
            .filter_map(|&(number, _)| Some((number, self.region_index(number)?)))
            .collect();

        for &(number, region) in &regions {
            let reserve = (!shrinking).then(|| self.reserve(region));
            let reserve = reserve.unwrap_or(u32::MAX);
            let (mut head, mut covered) = (Vec::new(), 0u32);
            for &(_, id) in self.order.get(region).into_iter().flatten() {
                if covered >= reserve || head.len() == fits {
                    break;
                }
                let units = self
                    .cache
                    .get(&id)
                    .map_or(0, |row| self.class_units(row.class, region));
                covered = covered.saturating_add(units.max(1));
                head.push(id);
            }
            if self.installed.get(region) != Some(&head) {
                let keys = head
                    .iter()
                    .filter_map(|id| Some(self.cache.get(id)?.key))
                    .collect();
                self.submit(runtime::Request::Victims {
                    region: number,
                    keys,
                });
                if let Some(installed) = self.installed.get_mut(region) {
                    *installed = head;
                }
            }
        }

        // The order cannot count the units of blocks that state their own size.
        let unknown = regions.iter().any(|&(_, region)| {
            let known = self
                .ordered
                .get(region)
                .copied()
                .unwrap_or(0)
                .saturating_add(self.pending_shared(region));
            self.classes_in(region)
                .all(|class| self.classes[usize::from(class)].units > 0)
                && self.evictable(region) > known
        });
        let listing = |request: &runtime::Request| *request == runtime::Request::List;
        let asked = self.held.iter().any(listing)
            || self
                .outstanding
                .values()
                .any(|(_, request)| listing(request));
        if unknown && !asked {
            self.submit(runtime::Request::List);
        }
    }

    /// The units of `order[region]` beyond the head installed there, which the buffer pool's
    /// `available` word does not count: it evicts them once a later `Victims` names them.
    pub(super) fn beyond_head(&self, region: usize) -> u32 {
        let (Some(order), Some(installed)) = (self.order.get(region), self.installed.get(region))
        else {
            return 0;
        };
        let kept = |id: &u64| {
            self.cache
                .get(id)
                .filter(|row| order.contains(&(row.last_used, *id)))
        };
        let head = installed
            .iter()
            .filter_map(kept)
            .map(|row| self.class_units(row.class, region));
        self.ordered.get(region).map_or(0, |units| {
            units.saturating_sub(head.fold(0, u32::saturating_add))
        })
    }

    /// The units one step's allocations can take from `region` (by index), SD §4.5: per class
    /// taken from it, the blocks of `max_num_batched_tokens` tokens of the deployment's block
    /// size and [`LOW_WATERMARK`] blocks per running request.
    fn reserve(&self, region: usize) -> u32 {
        let tokens = u32::try_from(self.config.max_num_batched_tokens).unwrap_or(u32::MAX);
        let stated = self
            .groups
            .iter()
            .find_map(|group| group.capacity.kv_cache.as_ref()?.block_size);
        let running = u32::try_from(self.running.len()).unwrap_or(u32::MAX);
        let blocks = tokens
            .div_ceil(stated.map_or(BLOCK_TOKENS, NonZeroU32::get))
            .saturating_add(LOW_WATERMARK.saturating_mul(running));
        let units = self
            .classes_in(region)
            .map(|class| self.class_units(class, region).max(1));
        units.fold(0, |reserve, units| {
            reserve.saturating_add(blocks.saturating_mul(units))
        })
    }

    /// The classes whose blocks `region` (by index) holds on this scheduler's side: as a device
    /// region of one of its groups, or as the class's host region.
    fn classes_in(&self, region: usize) -> impl Iterator<Item = u8> + '_ {
        let host = move |class: &runtime::Class| {
            class
                .host
                .is_some_and(|(host, _)| usize::from(host) == region)
        };
        let device = move |class: u8| {
            self.groups.iter().any(|group| {
                group
                    .regions(class)
                    .iter()
                    .any(|&r| usize::from(r) == region)
            })
        };
        (0..=u8::MAX)
            .zip(&self.classes)
            .filter(move |&(class, stated)| host(stated) || device(class))
            .map(|(class, _)| class)
    }

    /// An upper bound of the units of `region` (by index) that requests outstanding may have made
    /// shared and unpinned before their answers are read: a `Commit`'s block, a `Store`'s and a
    /// `Prefetch`'s keys, each counted at the region's largest block. The scheduler's own
    /// requests thus never make it `List`.
    fn pending_shared(&self, region: usize) -> u32 {
        let blocks = self.outstanding.values().map(|(_, request)| match request {
            runtime::Request::Commit { .. } => 1,
            runtime::Request::Store {
                keys,
                tier: Tier::Host,
                ..
            }
            | runtime::Request::Prefetch { keys, .. } => keys.len(),
            _ => 0,
        });
        let largest = self
            .classes_in(region)
            .map(|class| self.class_units(class, region))
            .max()
            .unwrap_or(0);
        u32::try_from(blocks.sum::<usize>())
            .unwrap_or(u32::MAX)
            .saturating_mul(largest)
    }
}

/// The bit of `group` in [`Cached::groups`].
fn bit(group: GroupId) -> u8 {
    1u8.checked_shl(u32::from(group.0)).unwrap_or(0)
}

/// The groups that stamped a request, as bits of [`Cached::groups`]: those whose device block
/// tables held its entry.
fn stamped_groups(after: &runtime::Stamps) -> u8 {
    after.iter().fold(0, |bits, &(group, _)| bits | bit(group))
}

#[cfg(test)]
mod tests {
    //! The cache as completions read from a real queue pair change it, over the session of
    //! `scheduler::tests::bench`: group A's blocks of class `KV` lie in the region numbered 4,
    //! whose capacity words are registers 2 (`available`) and 4 (`evictable`).

    use std::sync::atomic::Ordering;

    use scheduling_guest::{Event, PageId};
    use storage::Key;
    use storage::runtime::{self, Cause, Completion, OwnerId, Source, Tier};

    use super::{Cached, DEVICE, HOST, REMOTE};
    use crate::scheduler::tests::{A, B, Bench, KV, OWNER_A, bench, key};

    type Answer = (
        runtime::RequestId,
        Option<OwnerId>,
        runtime::Request,
        Completion,
    );

    /// Writes `request`, which takes no stamp, answers it with `completion`, and returns what the
    /// scheduler applies.
    fn exchange(
        bench: &mut Bench,
        request: runtime::Request,
        completion: &Completion,
    ) -> Vec<Answer> {
        bench.scheduler.submit(request);
        bench.scheduler.post().unwrap();
        let taken = bench.take();
        assert_eq!(taken.len(), 1, "one request was written");
        bench.complete(taken[0].0, completion);
        bench.answers()
    }

    /// Writes a request stamped by group A's executor (a `Commit`, or the `Finish` of a copy it
    /// ran), answers it with `completion`, and returns what the scheduler applies.
    fn stamped(
        bench: &mut Bench,
        mut request: runtime::Request,
        completion: &Completion,
    ) -> Vec<Answer> {
        if let runtime::Request::Commit { .. } = request {
            bench.scheduler.submit(request.clone());
            assert_eq!(bench.stamps(), [(A, request.clone())]);
        }
        if let runtime::Request::Commit { after, .. } | runtime::Request::Finish { after, .. } =
            &mut request
        {
            after.push((A, 1));
        }
        bench.scheduler.stamped(vec![request]);
        bench.scheduler.post().unwrap();
        let taken = bench.take();
        assert_eq!(taken.len(), 1, "the stamped request was written");
        bench.complete(taken[0].0, completion);
        bench.answers()
    }

    /// A `Listing` of 64-token blocks of class `KV` whose device copies lie in group A, each with
    /// its tiers.
    fn list(bench: &mut Bench, keys: &[(Key, u8)]) {
        let keys = keys
            .iter()
            .map(|&(key, tiers)| (key, KV, 64, tiers, 1))
            .collect();
        exchange(bench, runtime::Request::List, &Completion::Listing { keys });
    }

    /// What the scheduler writes into the submission queue now.
    fn written(bench: &mut Bench) -> Vec<runtime::Request> {
        bench.scheduler.post().unwrap();
        bench
            .take()
            .into_iter()
            .map(|(_, request)| request)
            .collect()
    }

    fn victims(keys: &[Key]) -> runtime::Request {
        runtime::Request::Victims {
            region: 4,
            keys: keys.to_vec(),
        }
    }

    fn commit(at: u32, key: Key) -> runtime::Request {
        runtime::Request::Commit {
            owner: OWNER_A,
            class: KV,
            at,
            key,
            tokens: 64,
            after: Vec::new(),
        }
    }

    fn page(byte: u8) -> PageId {
        PageId(Cached::id(&key(byte)))
    }

    fn lookup(count: u8) -> Vec<Key> {
        (1..=count).map(key).collect()
    }

    #[test]
    fn a_committed_block_joins_the_order_when_its_last_holder_gives_it_up() {
        let mut bench = bench(4096);
        bench.attach(&[(A, OWNER_A, 3)]);

        // The request still holds the block it committed: cached, held, in no order.
        stamped(&mut bench, commit(0, key(1)), &Completion::Committed);
        bench.scheduler.install_victims();
        assert!(written(&mut bench).is_empty());

        // Given up, it joins the order, whose head is installed once.
        bench.scheduler.unpin(&[key(1)]);
        bench.scheduler.install_victims();
        assert_eq!(written(&mut bench), [victims(&[key(1)])]);
        bench.scheduler.install_victims();
        assert!(written(&mut bench).is_empty());

        // A block committed for a request that left before the answer is in the order at once,
        // behind the block given up before it.
        bench.scheduler.submit(commit(1, key(2)));
        bench.stamps();
        bench.scheduler.submit(runtime::Request::Detach {
            owner: OWNER_A,
            after: Vec::new(),
        });
        let mut handed = commit(1, key(2));
        if let runtime::Request::Commit { after, .. } = &mut handed {
            after.push((A, 2));
        }
        bench.scheduler.stamped(vec![handed]);
        bench.scheduler.post().unwrap();
        let taken = bench.take();
        bench.complete(taken[0].0, &Completion::Committed);
        assert!(bench.answers().is_empty(), "its owner is forgotten");
        bench.scheduler.install_victims();
        assert_eq!(written(&mut bench), [victims(&[key(1), key(2)])]);

        // The policy saw the block enter held, become evictable, and the second enter.
        let events: Vec<(PageId, bool)> = bench
            .scheduler
            .policy
            .events
            .iter()
            .map(|event| match event {
                Event::Cached { page } => (page.id, page.evictable),
                other => panic!("unexpected event {other:?}"),
            })
            .collect();
        assert_eq!(events, [(page(1), false), (page(1), true), (page(2), true)]);
    }

    #[test]
    fn the_order_is_least_recently_used_first_and_available_counts_what_lies_beyond_its_head() {
        let mut bench = bench(4096);
        // One step of 128 tokens takes two blocks of 64 tokens: the head covers two units.
        bench.scheduler.config.max_num_batched_tokens = 128;
        list(
            &mut bench,
            &[(key(1), DEVICE), (key(2), DEVICE), (key(3), DEVICE)],
        );
        bench.scheduler.install_victims();
        assert_eq!(written(&mut bench), [victims(&[key(1), key(2)])]);

        // The buffer pool's word counts the head; the scheduler adds the block beyond it.
        bench.pool.register(2).store(5, Ordering::Release);
        bench.scheduler.read_capacity();
        assert_eq!(bench.scheduler.available[0], 6);

        // A block pinned and given up again is the most recently used.
        bench.attach(&[(A, OWNER_A, 3)]);
        let pin = runtime::Request::Pin {
            owner: OWNER_A,
            class: KV,
            at: 0,
            keys: vec![key(1)],
        };
        let pinned = Completion::Pinned {
            runs: vec![(4, 1, 1)],
            host: 0,
        };
        exchange(&mut bench, pin, &pinned);
        bench.scheduler.unpin(&[key(1)]);
        bench.scheduler.install_victims();
        assert_eq!(written(&mut bench), [victims(&[key(2), key(3)])]);
        bench.scheduler.read_capacity();
        assert_eq!(bench.scheduler.available[0], 6);
    }

    #[test]
    fn pins_evictions_and_refusals_correct_what_cache_expects() {
        let mut bench = bench(4096);
        let both = DEVICE | HOST;
        let held = [(key(1), DEVICE), (key(2), both), (key(3), DEVICE)];
        list(
            &mut bench,
            &[
                held[0],
                held[1],
                held[2],
                (key(4), DEVICE),
                (key(5), DEVICE),
            ],
        );
        let keys = lookup(6);
        assert_eq!(
            bench.scheduler.expected_hit(&keys, A),
            [DEVICE, both, DEVICE, DEVICE, DEVICE, 0]
        );
        assert_eq!(
            bench.scheduler.expected_hit(&keys, B),
            [0, HOST, 0, 0, 0, 0]
        );

        // One key of three pinned: the other two are not valid in the group, and the first of
        // them has a host copy, the second none.
        bench.attach(&[(A, OWNER_A, 3)]);
        let pin = runtime::Request::Pin {
            owner: OWNER_A,
            class: KV,
            at: 0,
            keys: keys[..3].to_vec(),
        };
        let pinned = Completion::Pinned {
            runs: vec![(4, 1, 1)],
            host: 1,
        };
        exchange(&mut bench, pin, &pinned);
        assert_eq!(
            bench.scheduler.expected_hit(&keys, A),
            [DEVICE, HOST, 0, DEVICE, DEVICE, 0]
        );

        // A key an allocation evicted, and a clone's source the buffer pool does not hold.
        let allocate = runtime::Request::Allocate {
            owner: OWNER_A,
            class: KV,
            at: 1,
            count: 1,
            units: 0,
        };
        let allocated = Completion::Allocated {
            runs: vec![(4, 2, 1)],
            evicted: vec![key(4)],
        };
        exchange(&mut bench, allocate, &allocated);
        let clone = runtime::Request::Clone {
            owner: OWNER_A,
            class: KV,
            at: 2,
            key: key(5),
        };
        exchange(
            &mut bench,
            clone,
            &Completion::Refused { cause: Cause::Key },
        );
        assert_eq!(
            bench.scheduler.expected_hit(&keys, A),
            [DEVICE, HOST, 0, 0, 0, 0]
        );
        assert!(
            !bench.scheduler.cache_unpinned(Cached::id(&key(1))),
            "it is held"
        );
        assert!(
            !bench.scheduler.cache_unpinned(Cached::id(&key(2))),
            "on the host alone"
        );

        // Keys left on no tier are gone, and the policy was told.
        let freed: Vec<PageId> = bench
            .scheduler
            .policy
            .events
            .iter()
            .filter_map(|event| match event {
                Event::Freed { page } => Some(*page),
                _ => None,
            })
            .collect();
        assert_eq!(freed, [page(3), page(4), page(5)]);
    }

    #[test]
    fn evictable_units_the_order_cannot_account_for_ask_for_one_list() {
        let mut bench = bench(4096);
        list(&mut bench, &[(key(1), DEVICE)]);

        // Three unpinned units in the region, of which the scheduler knows one. A `List` waits
        // for every outstanding answer, since its own may take the whole completion queue, and
        // a second step asks for none beside it.
        bench.pool.register(4).store(3, Ordering::Release);
        bench.scheduler.install_victims();
        bench.scheduler.install_victims();
        bench.scheduler.post().unwrap();
        let taken = bench.take();
        let requests: Vec<_> = taken.iter().map(|(_, request)| request.clone()).collect();
        assert_eq!(requests, [victims(&[key(1)])]);

        bench.complete(taken[0].0, &Completion::Done);
        bench.answers();
        assert_eq!(written(&mut bench), [runtime::Request::List]);
    }

    #[test]
    fn a_listing_states_the_whole_cache() {
        let mut bench = bench(4096);
        list(&mut bench, &[(key(1), DEVICE), (key(2), DEVICE | HOST)]);
        list(&mut bench, &[(key(2), DEVICE), (key(3), HOST)]);
        assert_eq!(
            bench.scheduler.expected_hit(&lookup(3), A),
            [0, DEVICE, HOST]
        );

        // A block on the host alone is an offloaded page to the policy.
        let (pages, encodings) = bench.scheduler.cache_views();
        let mut pages: Vec<(PageId, bool)> =
            pages.iter().map(|page| (page.id, page.on_host)).collect();
        pages.sort();
        assert_eq!(pages, [(page(2), false), (page(3), true)]);
        assert!(encodings.is_empty());
    }

    #[test]
    fn a_copy_names_the_keys_it_covers_and_the_answer_to_its_finish_places_them() {
        let mut bench = bench(4096);
        bench.attach(&[(A, OWNER_A, 3)]);

        // A load of three keys whose copy covers two: the request applied names those two.
        let load = |count| runtime::Request::Load {
            owner: OWNER_A,
            class: KV,
            at: 0,
            keys: lookup(count),
        };
        let copy = Completion::Copy {
            group: A,
            keys: lookup(2),
            from: vec![(9, 0, 1), (9, 1, 1)],
            to: vec![(4, 2, 1), (4, 3, 1)],
            evicted: Vec::new(),
        };
        let answers = exchange(&mut bench, load(3), &copy);
        assert_eq!(answers[0].2, load(2));

        let finish = runtime::Request::Finish {
            copy: answers[0].0,
            ok: true,
            after: Vec::new(),
        };
        let answers = stamped(&mut bench, finish, &Completion::Loaded);
        assert_eq!(answers[0].2, load(2));
        assert_eq!(
            bench.scheduler.expected_hit(&lookup(3), A),
            [DEVICE, DEVICE, 0]
        );
        assert!(
            !bench.scheduler.cache_unpinned(Cached::id(&key(1))),
            "loaded for the request"
        );
        bench.scheduler.unpin(&lookup(2));
        assert!(bench.scheduler.cache_unpinned(Cached::id(&key(1))));

        // The remote store's answer names the keys it stored; a prefetch brings the leading keys
        // the source found to the host.
        let remote = runtime::Request::Store {
            group: A,
            keys: vec![key(1)],
            tier: Tier::Remote,
        };
        exchange(
            &mut bench,
            remote,
            &Completion::Stored { keys: vec![key(1)] },
        );
        let prefetch = runtime::Request::Prefetch {
            class: KV,
            units: 0,
            keys: vec![key(4), key(5), key(6)],
            source: Source::Remote,
            deadline: 0,
        };
        exchange(&mut bench, prefetch, &Completion::Prefetched { found: 2 });
        assert_eq!(
            bench.scheduler.expected_hit(&lookup(6), A),
            [DEVICE | REMOTE, DEVICE, 0, HOST, HOST, 0]
        );
    }
}
