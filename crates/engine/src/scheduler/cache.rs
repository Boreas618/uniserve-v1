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
//! (`Cached`, `Hit`, `Released`, `Held`, `Freed`, `Encoded`, SD §5.2). Before a step schedules, `install_victims`
//! installs the head of each order by `Victims`, ahead of the step's allocations, and asks for a
//! `List` when the buffer pool counts evictable units the scheduler cannot account for. The table is exact for what this queue pair
//! commits and evicts; a wrong belief costs reuse, never a wrong block, since every request that
//! names a key is answered from the buffer pool's own cache.

use std::collections::HashSet;
use std::num::NonZeroU32;
use std::time::Instant;

use proto::request::RequestId;
use scheduling::{EncodingId, EncodingView, Event, Nanos, PageId, PageView};
use storage::Key;
use storage::runtime::{self, Cause, Completion, GroupId, Tier};

use super::{Scheduler, monotonic_nanos};

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
pub(crate) struct Cached {
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
    /// Its last use, in nanoseconds of the host's monotonic clock (`monotonic_nanos`): its rank
    /// in the eviction order.
    pub last_used: u64,
    /// Where its host region stood when a `Store` of it to the host last stored nothing for it:
    /// the region's count of `Store`s that stored a block (`Scheduler::stores`), and after a
    /// `Full` the region's `available` word. It is offered to the host again only once one of
    /// them has moved ([`Scheduler::store_refused`]); `None` when no `Store` of it failed.
    pub refused: Option<(u64, Option<u32>)>,
}

impl Cached {
    /// The number the scheduler knows a cached block by: its key's leading eight bytes,
    /// big-endian. It is the policy's `PageId` for a KV block and its `EncodingId` for an encoder
    /// output, so a policy's id finds its block in one lookup.
    pub(crate) fn id(key: &Key) -> u64 {
        let mut leading = [0; 8];
        leading.copy_from_slice(&key.0[..8]);
        u64::from_be_bytes(leading)
    }

    /// The block as the policy's page: one with a device copy is a device page, one on the host
    /// alone an offloaded page.
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

impl Scheduler {
    /// Applies what a completion of `request` states of the cache (SD §4.5). The answer to a
    /// `Finish` speaks for the copy's request as `outstanding` keeps it, narrowed to the keys the
    /// copy covers ([`covered`]).
    pub(super) fn learn(&mut self, request: &runtime::Request, completion: &Completion) {
        use runtime::Request as R;
        let now = monotonic_nanos(Instant::now());

        // A block given up before its commit was answered keeps the rank it was given up with.
        let given_up = match request {
            R::Commit { key, .. } => self.given_up.remove(&Cached::id(key)),
            _ => None,
        };

        // The keys a completion lists as evicted left the regions its request took blocks from.
        let (taken, evicted) = match (request, completion) {
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

        // The answer to a `Finish` is the result of its copy, with the copy's group as its stamp.
        let copied;
        let (request, copy) = match request {
            R::Finish { copy, after, .. } => match self.outstanding.get(&copy.0) {
                Some((_, request, _)) => {
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
                // Held by its request while the entry is at or after the table's `first`; one
                // its request gave up already enters the order as it was given up.
                let row = self.request_of(*owner);
                let held = row.and_then(|row| row.blocks.get(usize::from(*class)));
                let held = held.is_some_and(|blocks| blocks.first <= *at);
                let used = if held { now } else { given_up.unwrap_or(now) };
                self.hold(
                    key,
                    (*class, *tokens),
                    stamped_groups(after),
                    u32::from(held),
                    used,
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
                        self.change_row(key, Some((*class, 0)), |row| {
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
                // The keys of the copy are on the host, and the region takes copies again.
                for key in keys {
                    self.change_row(key, None, |row| row.tiers |= HOST);
                }
                let region = keys.first().and_then(|key| self.host_region_of(key));
                if let Some(stores) = region.and_then(|region| self.stores.get_mut(region)) {
                    *stores = stores.saturating_add(1);
                }
            }
            // A `Store` to the host that stored nothing for some of its keys: the host region had
            // no room for them, their copy failed or was given up, or the buffer pool found no
            // valid device copy of them to store, or found them stored already. Each waits for
            // what could change that answer before it is offered again (SD §4.4).
            (
                R::Store {
                    keys,
                    tier: Tier::Host,
                    ..
                },
                Completion::Full { .. },
                None,
            ) => self.refuse(keys, true),
            (
                R::Store {
                    keys,
                    tier: Tier::Host,
                    ..
                },
                Completion::Copy { keys: covered, .. },
                None,
            ) => {
                let left: Vec<Key> = keys
                    .iter()
                    .filter(|key| !covered.contains(key))
                    .copied()
                    .collect();
                self.refuse(&left, false);
            }
            (
                R::Store {
                    keys,
                    tier: Tier::Host,
                    ..
                },
                Completion::Stored { .. } | Completion::Done,
                _,
            ) => self.refuse(keys, false),
            (
                R::Store {
                    tier: Tier::Remote, ..
                },
                Completion::Stored { keys },
                None,
            ) => {
                for key in keys {
                    self.change_row(key, None, |row| row.tiers |= REMOTE);
                }
            }
            (R::Prefetch { class, keys, .. }, Completion::Prefetched { found }, None) => {
                for key in keys
                    .iter()
                    .take(usize::try_from(*found).unwrap_or(usize::MAX))
                {
                    self.change_row(key, Some((*class, 0)), |row| row.tiers |= HOST);
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
                self.change_row(key, None, |row| row.tiers &= !HOST);
            }
            (
                R::Load { keys, .. },
                Completion::Refused {
                    cause: Cause::Class,
                },
                None,
            ) => {
                if let Some(key) = keys.first() {
                    self.change_row(key, None, |row| row.tiers &= !HOST);
                }
            }
            _ => {}
        }
    }

    /// Records that a `Store` to the host stored nothing for `keys`: with `full` it was answered
    /// `Full`, and the block waits for its host region's `available` word to rise above what it
    /// shows now as well as for a `Store` into the region that stores a block, when `offload`
    /// offers it again.
    fn refuse(&mut self, keys: &[Key], full: bool) {
        for key in keys {
            let Some(region) = self.host_region_of(key) else {
                continue;
            };
            let stores = self.stores.get(region).copied().unwrap_or(0);
            let word = full.then(|| self.capacity_word(region, 0));
            self.change_row(key, None, |row| row.refused = Some((stores, word)));
            self.unstored.insert(Cached::id(key));
        }
    }

    /// Whether `row`'s last `Store` to the host stored nothing for it and nothing has happened
    /// since that could change the answer (SD §4.4 "The write policy"): no `Store` into its host
    /// region stored a block, and after a `Full` the region's `available` word has not risen
    /// above what it showed then. Such a block is not offered to the host.
    pub(super) fn store_refused(&self, row: &Cached) -> bool {
        let Some((stores, word)) = row.refused else {
            return false;
        };
        let Some(region) = self.host_region(row.class) else {
            return false;
        };
        let stored = self.stores.get(region).copied().unwrap_or(0);
        stored == stores && word.is_none_or(|word| self.capacity_word(region, 0) <= word)
    }

    /// The host region (by index) of `class`, which its `Store`s to the host fill.
    fn host_region(&self, class: u8) -> Option<usize> {
        let (region, _) = self.classes.get(usize::from(class))?.host?;
        Some(usize::from(region))
    }

    /// The host region of the class of the cached block of `key`.
    fn host_region_of(&self, key: &Key) -> Option<usize> {
        let row = self
            .cache
            .get(&Cached::id(key))
            .filter(|row| row.key == *key)?;
        self.host_region(row.class)
    }

    /// `key`'s block is on the device in `groups`, held by `pins` more requests and last used at
    /// `used` (nanoseconds, its rank in the eviction order).
    fn hold(&mut self, key: &Key, fresh: (u8, u16), groups: u8, pins: u32, used: u64) {
        self.change_row(key, Some(fresh), |row| {
            row.tiers |= DEVICE;
            row.groups |= groups;
            row.pins = row.pins.saturating_add(pins);
            row.last_used = used;
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
            self.change_row(key, None, |row| row.tiers = 0);
        }
        for &(key, class, tokens, tiers, groups) in keys {
            self.change_row(&key, Some((class, tokens)), |row| {
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
            self.change_row(key, None, |row| row.tiers &= !HOST);
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
        self.change_row(key, None, |row| {
            row.groups &= !groups;
            if row.groups == 0 {
                row.tiers &= !DEVICE;
            }
        });
    }

    /// Request `id` gave up the read-only blocks of `keys`, in position order, by a `Release` or
    /// a `Detach` handed to the executor: one fewer request holds each, but a key among its
    /// `Request::unpinned`, whose block `cache` never counted it a holder of (answered
    /// `Duplicate`, or an encoder output it computed and did not commit), which leaves that record
    /// instead. A block no request holds joins its orders as just used, the deepest key first in
    /// line, so that the prefix before it stays cached longest (vLLM's `free` frees a request's
    /// tail first). A block whose `Commit` the buffer pool has not answered is not in `cache`
    /// yet: its rank waits in `given_up` for its `Committed`, so that it joins the order where it
    /// would have had it been answered first.
    pub(super) fn unpin(&mut self, id: RequestId, keys: &[Key]) {
        let mut keys = keys.to_vec();
        if let Some(request) = self.requests.get_mut(&id) {
            keys.retain(
                |key| match request.unpinned.iter().position(|held| held == key) {
                    Some(at) => {
                        request.unpinned.swap_remove(at);
                        false
                    }
                    None => true,
                },
            );
        }

        let now = monotonic_nanos(Instant::now());
        let mut committing = None;
        for (offset, key) in (0u64..).zip(keys.iter().rev()) {
            let used = now.saturating_add(offset);
            let number = Cached::id(key);
            if !self.cache.get(&number).is_some_and(|row| row.key == *key) {
                let committing = committing.get_or_insert_with(|| self.committing());
                if committing.contains(&number) {
                    self.given_up.insert(number, used);
                }
                continue;
            }
            self.change_row(key, None, |row| {
                row.pins = row.pins.saturating_sub(1);
                row.last_used = used;
            });
        }
    }

    /// The keys, by [`Cached::id`], of this scheduler's `Commit`s the buffer pool has not
    /// answered: held, being stamped, waiting behind a request being stamped, or outstanding.
    fn committing(&self) -> HashSet<u64> {
        let stamping = self
            .stamping
            .values()
            .flat_map(|(request, _, following)| std::iter::once(request).chain(following));
        let outstanding = self.outstanding.values().map(|(_, request, _)| request);
        let commits = self.held.iter().chain(stamping).chain(outstanding);
        commits
            .filter_map(|request| match request {
                runtime::Request::Commit { key, .. } => Some(Cached::id(key)),
                _ => None,
            })
            .collect()
    }

    /// Applies `change` to the row of `key`, or to a new row of `fresh`'s class and tokens; a
    /// row left on no tier leaves `cache`. The orders follow the row, and the policy is told. A
    /// key unknown without `fresh`, or a second key under a known key's number (SD §4.5), is
    /// left alone.
    fn change_row(
        &mut self,
        key: &Key,
        fresh: Option<(u8, u16)>,
        change: impl FnOnce(&mut Cached),
    ) {
        let id = Cached::id(key);
        let before = self.cache.get(&id).copied();
        let mut row = match (before, fresh) {
            (Some(row), _) if row.key == *key => row,
            (None, Some((class, tokens))) => {
                let last_used = monotonic_nanos(Instant::now());
                Cached {
                    key: *key,
                    class,
                    tokens,
                    tiers: 0,
                    groups: 0,
                    pins: 0,
                    last_used,
                    refused: None,
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
    /// there: the device regions of its class in the groups that hold its device copy while no
    /// request holds it, and its class's host region while it has a host copy. A pin holds the
    /// device block a request reads (BP §4); the host copy is no request's, and the buffer pool's
    /// `evictable` word of the host region counts it whether or not its device block is pinned.
    fn placed(&self, row: &Cached) -> Vec<(usize, u32)> {
        let devices = self.groups.iter().filter(|group| {
            row.pins == 0 && row.tiers & DEVICE != 0 && row.groups & bit(group.id()) != 0
        });
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
    /// that the last request holding it gave up, and nothing else the policy sees changed, is one
    /// page of a `Released`, which lists the pages of consecutive releases in the order they
    /// joined the eviction order; one that no request held and a request holds now, nothing else
    /// the policy sees changed, is one page of a `Held`, which lists the pages of consecutive
    /// holds in the order they were taken; one that enters `cache`, or whose tier or evictability
    /// as the policy sees them changes otherwise, is `Cached`; one held by one more request and
    /// nothing else the policy sees is a `Hit`; one that leaves is `Freed`. An encoder output (of
    /// a sized class a group of the scheduler computes into device memory) that enters `cache` or
    /// is used again is `Encoded`. A stored trajectory, on the host alone, is no object of the
    /// policy's.
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
            // A release: only the holders and the last use moved, and the last use is the
            // release itself, which the event's place in the order states.
            (true, Some(old), Some(new))
                if old.pins > 0
                    && new.pins == 0
                    && old.tokens == new.tokens
                    && old.page().on_host == new.page().on_host =>
            {
                if let Some(Event::Released { pages }) = self.policy.events.last_mut() {
                    pages.push(PageId(id));
                    return;
                }
                Event::Released {
                    pages: vec![PageId(id)],
                }
            }
            // A hold: a request takes the block out of the eviction order, which the event
            // states, and uses it now, which its place in the order states.
            (true, Some(old), Some(new))
                if old.pins == 0
                    && new.pins > 0
                    && old.tokens == new.tokens
                    && old.page().on_host == new.page().on_host =>
            {
                if let Some(Event::Held { pages }) = self.policy.events.last_mut() {
                    pages.push(PageId(id));
                    return;
                }
                Event::Held {
                    pages: vec![PageId(id)],
                }
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

    /// A group whose memory holds `row`'s device copy, the first of the scheduler's groups that
    /// does: the group a `Store` of the block to the host names, whose workers copy it (SD
    /// §3.4, §4.4). `None` for a block with no device copy.
    pub(super) fn holder(&self, row: &Cached) -> Option<GroupId> {
        if row.tiers & DEVICE == 0 {
            return None;
        }
        self.groups
            .iter()
            .map(super::WorkerGroup::id)
            .find(|group| row.groups & bit(*group) != 0)
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

    /// Before a step schedules (SD §3.1, §4.5), for each region of this queue pair whose order's
    /// head is not the one installed: `Victims { region, keys }` with the leading keys of the
    /// order whose units cover what one step's allocations can take there (the whole order, as
    /// far as a record holds it, while a drain shrinks the regions). It is held ahead of the
    /// step's `Allocate`s, which `available` lets take those units. Then `List`, when a region's
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
                && self.capacity_word(region, 2) > known
        });
        let listing = |request: &runtime::Request| *request == runtime::Request::List;
        let asked = self.held.iter().any(listing)
            || self
                .outstanding
                .values()
                .any(|(_, request, _)| listing(request));
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
    /// taken from it, the blocks of `max_num_batched_tokens` tokens of the class's own block size
    /// and [`LOW_WATERMARK`] blocks per running request. A cache group's class (class `c` is
    /// cache group `c`) holds the block size its groups state for it; a sized class, whose blocks
    /// hold no tokens, is counted in blocks of the deployment's block size.
    fn reserve(&self, region: usize) -> u32 {
        let tokens = u32::try_from(self.config.max_num_batched_tokens).unwrap_or(u32::MAX);
        let stated = self
            .groups
            .iter()
            .find_map(|group| group.capacity.kv_cache.as_ref()?.block_size);
        let deployment = stated.map_or(BLOCK_TOKENS, NonZeroU32::get);
        let running = u32::try_from(self.running.len()).unwrap_or(u32::MAX);

        let reserve = self.classes_in(region).map(|class| {
            let block = self
                .groups
                .iter()
                .find_map(|group| group.capacity.cache_groups.get(usize::from(class)))
                .map_or(deployment, |&(_, block)| block);
            let blocks = tokens
                .div_ceil(block.max(1))
                .saturating_add(LOW_WATERMARK.saturating_mul(running));
            blocks.saturating_mul(self.class_units(class, region).max(1))
        });
        reserve.fold(0, u32::saturating_add)
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
        let blocks = self
            .outstanding
            .values()
            .map(|(_, request, _)| match request {
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

/// `request` as the scheduler applies its answer `completion`: a `Load` or a `Store` answered
/// `Copy` narrowed to the keys the copy covers, which are the leading keys of a `Load` that have
/// a host copy and the keys of a `Store` whose device copy is valid in its group (BP §11), and
/// for which the answer to its `Finish` speaks; any other request as it was written. The
/// executor is handed the request as it was written, which its completion answers (SD §2.1
/// `Executor::map`).
pub(super) fn covered(request: &runtime::Request, completion: &Completion) -> runtime::Request {
    let mut narrowed = request.clone();
    if let (
        runtime::Request::Load { keys, .. } | runtime::Request::Store { keys, .. },
        Completion::Copy { keys: copied, .. },
    ) = (&mut narrowed, completion)
    {
        keys.retain(|key| copied.contains(key));
    }
    narrowed
}

#[cfg(test)]
mod tests {
    //! The cache as completions read from a real queue pair change it, over the session of
    //! `scheduler::tests::bench`: group A's blocks of class `KV` lie in the region numbered 4,
    //! whose capacity words are registers 2 (`available`) and 4 (`evictable`).

    use std::num::NonZeroU32;
    use std::sync::atomic::Ordering;

    use crate::batch::cache::CacheGroupKind;
    use proto::config::KvCacheConfig;
    use scheduling::{Event, PageId};
    use storage::Key;
    use storage::runtime::{self, Cause, Completion, OwnerId, Source, Tier};

    use proto::request::RequestId;

    use super::{Cached, DEVICE, HOST, REMOTE};
    use crate::scheduler::tests::{A, B, Bench, KV, OWNER_A, PRODUCT, bench, key};

    /// The request `Bench::attach` adds.
    const ID: RequestId = RequestId(1);

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

    /// Writes a request stamped by group A's workers (a `Commit`, or the `Finish` of a copy it
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
        bench.scheduler.unpin(ID, &[key(1)]);
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
        // Its owner is forgotten, but a commit outlives its owner's `Detach` (BP §9, R6): the
        // answer is applied, for the write policy to store the block.
        let answers = bench.answers();
        assert!(matches!(answers[..], [(_, _, _, Completion::Committed)]));
        bench.scheduler.install_victims();
        assert_eq!(written(&mut bench), [victims(&[key(1), key(2)])]);

        // The policy saw the block enter held, its release, and the second enter given up.
        let events: Vec<(&str, PageId, bool)> = bench
            .scheduler
            .policy
            .events
            .iter()
            .flat_map(|event| match event {
                Event::Cached { page } => vec![("cached", page.id, page.evictable)],
                Event::Released { pages } => {
                    pages.iter().map(|page| ("released", *page, true)).collect()
                }
                other => panic!("unexpected event {other:?}"),
            })
            .collect();
        assert_eq!(
            events,
            [
                ("cached", page(1), false),
                ("released", page(1), true),
                ("cached", page(2), true)
            ]
        );
    }

    /// A request that gives up the blocks it holds tells the policy one `Released`, which lists
    /// them in the order they join the eviction order, the deepest first (SD §4.4): the order the
    /// policy keeps is the one the scheduler installs.
    #[test]
    fn a_release_of_many_blocks_is_one_event_in_the_eviction_order() {
        let mut bench = bench(4096);
        bench.attach(&[(A, OWNER_A, 3)]);
        for (at, byte) in [(0, 1), (1, 2), (2, 3)] {
            stamped(&mut bench, commit(at, key(byte)), &Completion::Committed);
        }
        bench.scheduler.install_victims();
        assert!(written(&mut bench).is_empty());
        bench.scheduler.policy.events.clear();

        bench.scheduler.unpin(ID, &[key(1), key(2), key(3)]);
        bench.scheduler.install_victims();
        assert_eq!(written(&mut bench), [victims(&[key(3), key(2), key(1)])]);

        let events = &bench.scheduler.policy.events;
        assert!(
            matches!(&events[..], [Event::Released { pages }] if *pages == [page(3), page(2), page(1)]),
            "{events:?}"
        );
    }

    /// An admission that pins blocks no request held tells the policy one `Held`, which lists
    /// them in the order they were pinned, and they leave the eviction order (SD §4.4, §4.5); a
    /// block pinned by one more request is a `Hit`.
    #[test]
    fn a_pin_of_many_evictable_blocks_is_one_event_in_the_order_they_were_taken() {
        let mut bench = bench(4096);
        list(
            &mut bench,
            &[(key(1), DEVICE), (key(2), DEVICE), (key(3), DEVICE)],
        );
        bench.scheduler.install_victims();
        assert_eq!(written(&mut bench), [victims(&[key(1), key(2), key(3)])]);
        bench.scheduler.policy.events.clear();

        bench.attach(&[(A, OWNER_A, 3)]);
        let pin = |keys: &[Key]| runtime::Request::Pin {
            owner: OWNER_A,
            class: KV,
            at: 0,
            keys: keys.to_vec(),
        };
        let pinned = |blocks: u32| Completion::Pinned {
            runs: (0..blocks).map(|block| (4, block + 1, 1)).collect(),
            host: 0,
        };
        exchange(&mut bench, pin(&[key(3), key(1)]), &pinned(2));
        exchange(&mut bench, pin(&[key(3)]), &pinned(1));
        bench.scheduler.install_victims();
        assert_eq!(written(&mut bench), [victims(&[key(2)])]);

        let events = &bench.scheduler.policy.events;
        assert!(
            matches!(
                &events[..],
                [Event::Held { pages }, Event::Hit { page: hit }]
                    if *pages == [page(3), page(1)] && *hit == page(3)
            ),
            "{events:?}"
        );
    }

    /// An admission that reuses a warm cache, pinning 1000 blocks of 64 tokens no request held,
    /// tells the policy one `Held` of them: the step's `observe` and `dispatch` together consume
    /// at most half of a call's fuel (SD §5.2 "Faults"), where restating each block as a page
    /// costs about as much as a call has. The policy saw the pins: a shortage of device units
    /// takes none of the blocks.
    #[test]
    fn an_admission_pinning_a_thousand_cached_blocks_stays_well_within_a_calls_fuel() {
        use scheduling::decision::{Room, Shortage, Storage, Victim};
        use scheduling::limits::Parameters;

        use crate::scheduler::packaged;
        use crate::scheduler::policy::{self, DEFAULT_CALL_FUEL, Policy};

        const BLOCKS: u64 = 1000;
        let numbered = |number: u64| {
            let mut bytes = [0; 32];
            bytes[..8].copy_from_slice(&(number + 1).to_be_bytes());
            Key(bytes)
        };
        let keys: Vec<Key> = (0..BLOCKS).map(numbered).collect();

        let mut bench = bench(1 << 20);
        let listed: Vec<(Key, u8)> = keys.iter().map(|&key| (key, DEVICE)).collect();
        list(&mut bench, &listed);
        bench.attach(&[(A, OWNER_A, 3)]);
        let snapshot = bench.scheduler.snapshot();
        let ceilings = policy::ceilings([]);
        bench.scheduler.policy =
            Policy::new(packaged("fcfs"), Parameters::default(), snapshot, &ceilings).unwrap();

        let pin = runtime::Request::Pin {
            owner: OWNER_A,
            class: KV,
            at: 0,
            keys: keys.clone(),
        };
        let pinned = Completion::Pinned {
            runs: vec![(4, 1, 1); keys.len()],
            host: 0,
        };
        exchange(&mut bench, pin, &pinned);

        let scheduler = &mut bench.scheduler;
        let room = Room {
            token_budget: 8192,
            call_budget: 128,
            page_tokens: 64,
            ..Room::default()
        };
        let fuel = scheduler.policy.fuel;
        let output = scheduler
            .policy
            .dispatch(&room, &mut scheduler.requests, &scheduler.groups)
            .unwrap();
        let spent = scheduler.policy.fuel - fuel;
        assert!(
            spent <= DEFAULT_CALL_FUEL / 2,
            "observe and dispatch took {spent} fuel"
        );
        assert!(output.offload.is_empty());

        let shortage = Shortage {
            kind: Storage::KvUnits,
            need: BLOCKS,
        };
        let victims = scheduler.policy.preempt(&shortage).unwrap();
        assert!(
            !victims
                .iter()
                .any(|victim| matches!(victim, Victim::Page(_))),
            "{victims:?}"
        );
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
        bench.scheduler.unpin(ID, &[key(1)]);
        bench.scheduler.install_victims();
        assert_eq!(written(&mut bench), [victims(&[key(2), key(3)])]);
        bench.scheduler.read_capacity();
        assert_eq!(bench.scheduler.available[0], 6);
    }

    /// The head of an order covers one step's tokens in blocks of the class's own size: a step
    /// of 128 tokens takes 8 blocks of a cache group of 16-token blocks, whatever block size the
    /// deployment states for its widest cache group (SD §4.5).
    #[test]
    fn the_head_covers_a_steps_tokens_in_blocks_of_the_classs_own_size() {
        let mut bench = bench(4096);
        bench.scheduler.config.max_num_batched_tokens = 128;
        let capacity = &mut bench.scheduler.groups[0].capacity;
        capacity.kv_cache = Some(KvCacheConfig {
            block_size: NonZeroU32::new(64),
            ..KvCacheConfig::default()
        });
        capacity.cache_groups = vec![(CacheGroupKind::Full, 16)];

        let listed: Vec<(Key, u8)> = lookup(10).into_iter().map(|key| (key, DEVICE)).collect();
        list(&mut bench, &listed);
        bench.scheduler.install_victims();
        assert_eq!(written(&mut bench), [victims(&lookup(8))]);
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

    /// A pin holds a key's device block; its host copy stays in the host region's order, which
    /// the buffer pool's `evictable` word of that region counts: no `List` is asked for it.
    #[test]
    fn a_pinned_blocks_host_copy_stays_in_the_host_regions_order() {
        let mut bench = bench(4096);
        // The region numbered 6 stands for the host region of class `KV`, its blocks one unit.
        bench.scheduler.classes[usize::from(KV)].host = Some((2, 1));
        bench.scheduler.classes[usize::from(PRODUCT)].units = 1;
        list(&mut bench, &[(key(1), DEVICE | HOST)]);
        bench.scheduler.install_victims();
        let host_victims = runtime::Request::Victims {
            region: 6,
            keys: vec![key(1)],
        };
        assert_eq!(written(&mut bench), [victims(&[key(1)]), host_victims]);

        // A request pins the device block; the host copy is still evictable there.
        bench.pool.register(10).store(1, Ordering::Release);
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

        // The device region's order loses the block, the host region's keeps it: the scheduler
        // accounts for every evictable unit and asks for no `List`.
        bench.scheduler.install_victims();
        bench.scheduler.post().unwrap();
        let taken = bench.take();
        let requests: Vec<_> = taken.iter().map(|(_, request)| request.clone()).collect();
        assert_eq!(requests, [victims(&[])]);
        bench.complete(taken[0].0, &Completion::Done);
        bench.answers();
        bench.scheduler.install_victims();
        assert!(written(&mut bench).is_empty());
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
            encodings: None,
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
        bench.scheduler.unpin(ID, &lookup(2));
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
