//! What a scheduler knows of the cache (`specs/scheduler-design.md` §4.5), the decision half of
//! vLLM's block pool (`vllm/v1/core/block_pool.py`): the [`Cached`] row of every cached block it
//! knows of, in `Kv::cache` by [`Cached::id`] (vLLM's `BlockHashToBlockMap`), and per region the
//! eviction order of the blocks no request of the scheduler holds, least recently used first, in
//! `Kv::order` (the free block queue's order: `touch` takes a block out, `free_blocks` puts it
//! back last).
//!
//! Every completion read from the queue pair is applied here ([`Kv::learn`]), one dropped for a
//! forgotten owner too, since what it states of the buffer pool's cache holds all the same; a
//! request gives up the blocks it holds by [`Kv::unpin`]. Each change of a row is told to the
//! policy (`Cached`, `Hit`, `Released`, `Held`, `Freed`, `Encoded`, SD §5.2), through the event
//! sink the caller hands in. Before a step's requests are written, [`Kv::victims`] states the
//! head of each order to install by `Victims`, ahead of the step's allocations, and asks for a
//! `List` when the buffer pool counts evictable units the scheduler cannot account for. The
//! table is exact for what this scheduler commits and evicts; a wrong belief costs reuse, never a
//! wrong block, since every request that names a key is answered from the buffer pool's own
//! cache (`specs/kv-cache.md` §3.4).

use std::collections::HashSet;
use std::time::Instant;

use proto::config::SchedulerConfig;
use scheduling::{EncodingId, EncodingView, Event, Nanos, PageId, PageView};
use storage::Key;
use storage::runtime::{self, Cause, Completion, GroupId, Tier};

use super::super::{Request, WorkerGroup, monotonic_nanos};
use super::Kv;

/// The tier bits of [`Cached::tiers`], as a `Listing` states them.
pub(crate) const DEVICE: u8 = 1;
pub(crate) const HOST: u8 = 2;
pub(crate) const REMOTE: u8 = 4;

/// The low watermark: blocks of each class a running request holds beyond its next call (SD
/// §4.4).
pub(crate) const LOW_WATERMARK: u32 = 1;

/// The bytes of a `Victims` record besides its keys (`storage::runtime`).
const VICTIMS_RECORD: usize = 14;

/// The bytes of an `Evict` record besides its keys (`storage::runtime`): the request's number,
/// its variant and the length of its key list.
const EVICT_RECORD: usize = 13;

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
    /// The worker groups that hold its device copy, one bit per group number: the copy lies in
    /// the group's region of its class, valid in the memory the group reads it in, so a `Pin` of
    /// the group finds it and the group's orders may evict it.
    pub groups: u8,
    /// Requests of this scheduler holding it.
    pub pins: u32,
    /// Its last use, in nanoseconds of the host's monotonic clock (`monotonic_nanos`): its rank
    /// in the eviction order.
    pub last_used: u64,
    /// Where its host region stood when a `Store` of it to the host last stored nothing for it:
    /// the region's count of `Store`s that stored a block (`Kv::stores`), and after a `Full` the
    /// region's `available` word. It is offered to the host again only once one of them has
    /// moved ([`Kv::store_refused`]); `None` when no `Store` of it failed.
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

    /// Whether its device copy is in the eviction order of `group`'s regions of its class: valid
    /// in the group's memory, and held by no request of this scheduler.
    fn ordered_in(&self, group: GroupId) -> bool {
        self.pins == 0 && self.tiers & DEVICE != 0 && self.groups & bit(group) != 0
    }
}

/// What each completion states of the cache (SD §4.5).
impl Kv {
    /// Applies what the completion `completion` of `asked` states of the cache, telling `told`
    /// what the policy sees change, and returns how many blocks it lists as evicted. `holder` is
    /// the request of the owner `asked` acts for, `None` once that owner is forgotten. The answer
    /// to a `Finish` speaks for the copy's request as the scheduler kept it, narrowed to the keys
    /// the copy covers ([`covered`]): the caller passes that request as `asked`, with `copy` the
    /// groups that stamped the `Finish` (the copy's group), as bits of [`Cached::groups`].
    pub(crate) fn learn(
        &mut self,
        (asked, copy): (&runtime::Request, Option<u8>),
        holder: Option<&Request>,
        completion: &Completion,
        groups: &[WorkerGroup],
        told: &mut Vec<Event>,
    ) -> usize {
        use runtime::Request as R;
        let now = monotonic_nanos(Instant::now());

        // A block given up before its commit was answered keeps the rank it was given up with.
        let given_up = match asked {
            R::Commit { key, .. } if copy.is_none() => {
                let id = Cached::id(key);
                if let Some(count) = self.committing.get_mut(&id) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        self.committing.remove(&id);
                    }
                }
                self.given_up.remove(&id)
            }
            _ => None,
        };

        // The keys a completion lists as evicted left the regions its request took blocks from,
        // a crossing's the regions of the runs it states.
        let (taken, evicted) = match (asked, completion, copy) {
            (_, Completion::Allocated { runs, evicted }, None)
            | (_, Completion::Replicated { runs, evicted }, None)
            | (
                _,
                Completion::Copy {
                    to: runs, evicted, ..
                },
                None,
            ) => (runs.iter().map(|run| run.0).collect(), evicted.as_slice()),
            (R::Resize { region, .. }, Completion::Resized { evicted, .. }, None) => {
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
        let mut count = taken.len() * evicted.len();
        for (region, key) in taken
            .into_iter()
            .flat_map(|region| evicted.iter().map(move |key| (region, key)))
        {
            self.drop_copy(key, region, groups, told);
        }
        // A prefetch took its blocks from its class's host region, by the session's index.
        if let (R::Prefetch { class, units, .. }, Completion::Prefetched { evicted, .. }, None) =
            (asked, completion, copy)
            && let Some((host, _)) = self.host_block(*class, *units)
        {
            count += evicted.len();
            for key in evicted {
                self.drop_copy(key, host, groups, told);
            }
        }

        if *asked == R::List && copy.is_none() {
            self.lists_answered += 1;
        }
        let attached = |owner| {
            holder
                .and_then(|row| row.attachment(owner))
                .map(|(group, _)| group)
        };
        match (asked, completion, copy) {
            (R::List, Completion::Listing { keys }, None) => self.listed(keys, groups, told),
            (
                R::Commit {
                    class,
                    at,
                    key,
                    tokens,
                    after,
                    ..
                },
                Completion::Committed,
                None,
            ) => {
                // Held by its request while the entry is at or after the table's `first`; one
                // its request gave up already enters the order as it was given up.
                let held = holder.and_then(|row| row.blocks.get(usize::from(*class)));
                let held = held.is_some_and(|blocks| blocks.first <= *at);
                let used = if held { now } else { given_up.unwrap_or(now) };
                let fresh = (*class, *tokens);
                let pins = u32::from(held);
                self.hold(
                    key,
                    fresh,
                    (stamped_groups(after), pins, used),
                    groups,
                    told,
                );
            }
            (R::Load { class, keys, .. }, Completion::Loaded, Some(stamped)) => {
                let pins = u32::from(holder.is_some());
                for key in keys {
                    self.hold(key, (*class, 0), (stamped, pins, now), groups, told);
                }
            }
            (
                R::Pin {
                    owner, class, keys, ..
                },
                Completion::Pinned { blocks, host },
                None,
            ) => {
                // A returned admission's pins left with its owner's `Detach`.
                let Some(group) = attached(*owner) else {
                    return count;
                };
                // The answer states one block per leading key it pinned.
                let (held, rest) = keys.split_at(blocks.len().min(keys.len()));
                for key in held {
                    self.hold(key, (*class, 0), (bit(group), 1, now), groups, told);
                }
                // The keys after the pinned ones are not valid in the group; the first `host` of
                // them have a host copy, and the next one none.
                let host = usize::try_from(*host).unwrap_or(usize::MAX).min(rest.len());
                for (at, key) in rest.iter().enumerate() {
                    self.invalidate(key, bit(group), groups, told);
                    let tiers = if at < host { HOST } else { 0 };
                    if at <= host {
                        self.change_row(key, Some((*class, 0)), groups, told, |row| {
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
                    self.change_row(key, None, groups, told, |row| row.tiers |= HOST);
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
            ) => self.refuse(keys, true, groups, told),
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
                self.refuse(&left, false, groups, told);
            }
            (
                R::Store {
                    keys,
                    tier: Tier::Host,
                    ..
                },
                Completion::Stored { .. } | Completion::Done,
                _,
            ) => self.refuse(keys, false, groups, told),
            (
                R::Store {
                    tier: Tier::Remote, ..
                },
                Completion::Stored { keys },
                None,
            ) => {
                for key in keys {
                    self.change_row(key, None, groups, told, |row| row.tiers |= REMOTE);
                }
            }
            (R::Prefetch { class, keys, .. }, Completion::Prefetched { found, .. }, None) => {
                for key in keys
                    .iter()
                    .take(usize::try_from(*found).unwrap_or(usize::MAX))
                {
                    let fresh = Some((*class, 0));
                    self.change_row(key, fresh, groups, told, |row| row.tiers |= HOST);
                }
            }
            (R::Evict { .. }, Completion::Evicted { keys }, None) => {
                count += keys.len();
                for key in keys {
                    self.invalidate(key, u8::MAX, groups, told);
                }
            }
            // A key `cache` believed held that the buffer pool does not hold, or a host copy
            // another layout wrote that this one cannot read, is a miss from now on.
            (R::Clone { owner, key, .. }, Completion::Refused { cause: Cause::Key }, None) => {
                if let Some(group) = attached(*owner) {
                    self.invalidate(key, bit(group), groups, told);
                }
            }
            (
                R::Restore { key, .. },
                Completion::Refused {
                    cause: Cause::Key | Cause::Class,
                },
                None,
            ) => {
                self.change_row(key, None, groups, told, |row| row.tiers &= !HOST);
            }
            (
                R::Load { keys, .. },
                Completion::Refused {
                    cause: Cause::Class,
                },
                None,
            ) => {
                if let Some(key) = keys.first() {
                    self.change_row(key, None, groups, told, |row| row.tiers &= !HOST);
                }
            }
            _ => {}
        }
        // A commit answered without a copy leaves no page to publish.
        if let (R::Commit { key, .. }, None) = (asked, copy) {
            self.events.settled(key);
        }
        count
    }

    /// Records that a `Store` to the host stored nothing for `keys`: with `full` it was answered
    /// `Full`, and the block waits for its host region's `available` word to rise above what it
    /// shows now as well as for a `Store` into the region that stores a block, when `offload`
    /// offers it again.
    fn refuse(&mut self, keys: &[Key], full: bool, groups: &[WorkerGroup], told: &mut Vec<Event>) {
        for key in keys {
            let Some(region) = self.host_region_of(key) else {
                continue;
            };
            let stores = self.stores.get(region).copied().unwrap_or(0);
            let word = full.then(|| self.word(region, 0));
            self.change_row(key, None, groups, told, |row| {
                row.refused = Some((stores, word))
            });
            self.unstored.insert(Cached::id(key));
        }
    }

    /// Whether `row`'s last `Store` to the host stored nothing for it and nothing has happened
    /// since that could change the answer (SD §4.4 "The write policy"): no `Store` into its host
    /// region stored a block, and after a `Full` the region's `available` word has not risen
    /// above what it showed then. Such a block is not offered to the host.
    fn store_refused(&self, row: &Cached) -> bool {
        let Some((stores, word)) = row.refused else {
            return false;
        };
        let Some(region) = self.host_region(row.class) else {
            return false;
        };
        let stored = self.stores.get(region).copied().unwrap_or(0);
        stored == stores && word.is_none_or(|word| self.word(region, 0) <= word)
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

    /// `key`'s block is on the device in the groups `held.0` (bits), held by `held.1` more
    /// requests and last used at `held.2` (nanoseconds, its rank in the eviction order); `fresh`
    /// is its class and tokens should it be new to `cache`.
    fn hold(
        &mut self,
        key: &Key,
        fresh: (u8, u16),
        (holders, pins, used): (u8, u32, u64),
        groups: &[WorkerGroup],
        told: &mut Vec<Event>,
    ) {
        self.change_row(key, Some(fresh), groups, told, |row| {
            row.tiers |= DEVICE;
            row.groups |= holders;
            row.pins = row.pins.saturating_add(pins);
            row.last_used = used;
        });
    }

    /// A `Listing` states every cached key, those the schedulers would evict first first: a row
    /// it lists becomes what it states, keeping its holders and last use; a key new to `cache`
    /// was never used here, and takes its place in the listing as its last use, which puts it
    /// before every block used here and keeps the listing's order among such keys, so that a
    /// scheduler that starts beside a warm cache (an epoch's target of the same layout) evicts
    /// first what the serving schedulers would evict first; a row it does not list leaves
    /// `cache`.
    fn listed(
        &mut self,
        keys: &[(Key, u8, u16, u8, u8, u8)],
        groups: &[WorkerGroup],
        told: &mut Vec<Event>,
    ) {
        let listed: HashSet<u64> = keys.iter().map(|(key, ..)| Cached::id(key)).collect();
        let gone: Vec<Key> = self
            .cache
            .values()
            .map(|row| row.key)
            .filter(|key| !listed.contains(&Cached::id(key)))
            .collect();
        for key in &gone {
            self.change_row(key, None, groups, told, |row| row.tiers = 0);
        }
        for (place, &(key, class, tokens, tiers, holders, _)) in (0u64..).zip(keys) {
            self.change_row(&key, Some((class, tokens)), groups, told, |row| {
                if row.tiers == 0 {
                    row.last_used = place;
                }
                *row = Cached {
                    class,
                    tokens,
                    tiers,
                    groups: holders,
                    ..*row
                };
            });
        }
    }

    /// The copy of `key` in `region` (by index) left it: a host copy leaves the host tier, a
    /// device copy the groups whose blocks of the key's class lie there.
    fn drop_copy(
        &mut self,
        key: &Key,
        region: usize,
        groups: &[WorkerGroup],
        told: &mut Vec<Event>,
    ) {
        let Some(class) = self.cache.get(&Cached::id(key)).map(|row| row.class) else {
            return;
        };
        if self.on_tier(region, Tier::Host) {
            self.change_row(key, None, groups, told, |row| row.tiers &= !HOST);
            return;
        }
        let there = |group: &&WorkerGroup| {
            group
                .regions(class)
                .iter()
                .any(|&r| usize::from(r) == region)
        };
        let holders = groups
            .iter()
            .filter(there)
            .fold(0, |bits, group| bits | bit(group.id()));
        self.invalidate(key, holders, groups, told);
    }

    /// `key`'s device copy is not valid in the groups `holders` (bits); it leaves the device once
    /// no group holds it.
    fn invalidate(
        &mut self,
        key: &Key,
        holders: u8,
        groups: &[WorkerGroup],
        told: &mut Vec<Event>,
    ) {
        self.change_row(key, None, groups, told, |row| {
            row.groups &= !holders;
            if row.groups == 0 {
                row.tiers &= !DEVICE;
            }
        });
    }

    /// `request` (`None` for a request that left) gave up the read-only blocks of `keys`, in
    /// position order, by a `Release` or a `Detach` the scheduler submits: one fewer request
    /// holds each, but a key among its `Request::unpinned`, whose block `cache` never counted it
    /// a holder of (answered `Duplicate`, or an encoder output it computed and did not commit),
    /// which leaves that record instead. A block no request holds joins its orders as just used,
    /// the deepest key first in line, so that the prefix before it stays cached longest (vLLM's
    /// `free` frees a request's tail first). A block whose `Commit` the buffer pool has not
    /// answered is not in `cache` yet: its rank waits in `given_up` for its `Committed`, so that
    /// it joins the order where it would have had it been answered first.
    pub(crate) fn unpin(
        &mut self,
        request: Option<&mut Request>,
        keys: &[Key],
        groups: &[WorkerGroup],
        told: &mut Vec<Event>,
    ) {
        let mut keys = keys.to_vec();
        if let Some(request) = request {
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
        for (offset, key) in (0u64..).zip(keys.iter().rev()) {
            let used = now.saturating_add(offset);
            let number = Cached::id(key);
            if !self.cache.get(&number).is_some_and(|row| row.key == *key) {
                if self.committing.contains_key(&number) {
                    self.given_up.insert(number, used);
                }
                continue;
            }
            self.change_row(key, None, groups, told, |row| {
                row.pins = row.pins.saturating_sub(1);
                row.last_used = used;
            });
        }
    }

    /// Applies `change` to the row of `key`, or to a new row of `fresh`'s class and tokens; a
    /// row left on no tier leaves `cache`. The orders follow the row, and the policy is told. A
    /// key unknown without `fresh`, or a second key under a known key's number (SD §4.5), is
    /// left alone.
    fn change_row(
        &mut self,
        key: &Key,
        fresh: Option<(u8, u16)>,
        groups: &[WorkerGroup],
        told: &mut Vec<Event>,
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
            for (region, units) in self.placed(&row, groups) {
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
        self.events.changed(key, after.map_or(0, |row| row.tiers));
        self.tell(before, after, groups, told);
    }

    /// The regions (by index) whose order holds `row`, each with the units the block takes
    /// there: the device regions of its class in the groups that hold its device copy while no
    /// request holds it, and its class's host region while it has a host copy. A pin holds the
    /// device block a request reads (BP §4); the host copy is no request's, and the buffer pool's
    /// `evictable` word of the host region counts it whether or not its device block is pinned.
    fn placed(&self, row: &Cached, groups: &[WorkerGroup]) -> Vec<(usize, u32)> {
        let devices = groups.iter().filter(|group| row.ordered_in(group.id()));
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

    /// Tells the policy, through `told`, how a row changed (SD §5.2). A KV block (of a class
    /// with a block size, in the KV region: a checkpoint is none, [`Self::is_page`]) that the
    /// last request holding it gave up, and nothing else the policy sees changed, is one page of
    /// a `Released`, which lists the pages of consecutive releases in the order they joined the
    /// eviction order; one that no request held and a request holds now, nothing else the policy
    /// sees changed, is one page of a `Held`, which lists the pages of consecutive holds in the
    /// order they were taken; one that enters `cache`, or whose tier or evictability as the
    /// policy sees them changes otherwise, is `Cached`; one held by one more request and nothing
    /// else the policy sees is a `Hit`; one that leaves is `Freed`. An encoder output (of a sized
    /// class a group of the scheduler computes into device memory) that enters `cache` or is used
    /// again is `Encoded`. A stored trajectory, on the host alone, is no object of the policy's.
    fn tell(
        &self,
        before: Option<Cached>,
        after: Option<Cached>,
        groups: &[WorkerGroup],
        told: &mut Vec<Event>,
    ) {
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
                if let Some(Event::Released { pages }) = told.last_mut() {
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
                if let Some(Event::Held { pages }) = told.last_mut() {
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
                if self.is_encoding(new.class, groups)
                    && before.is_none_or(|old| old.last_used != new.last_used) =>
            {
                Event::Encoded {
                    entry: new.encoding(),
                }
            }
            _ => return,
        };
        told.push(event);
    }

    /// Whether blocks of `class` are the policy's pages: KV blocks, of a class with a block size
    /// that lies in the KV region. A recurrent-state checkpoint lies in the state region, whose
    /// units no page of the policy's frees (`Storage::StateUnits`), and is no object of the
    /// policy's: its shortage takes requests, and the buffer pool evicts cached checkpoints in the
    /// order this scheduler installs.
    fn is_page(&self, class: u8) -> bool {
        self.classes
            .get(usize::from(class))
            .is_some_and(|stated| stated.units > 0 && !stated.state_in_slot)
    }

    /// Whether blocks of `class` are the policy's encoder outputs: of a class sized per block
    /// that one of `groups` computes into device memory.
    fn is_encoding(&self, class: u8, groups: &[WorkerGroup]) -> bool {
        self.classes
            .get(usize::from(class))
            .is_some_and(|stated| stated.units == 0)
            && groups.iter().any(|group| !group.regions(class).is_empty())
    }
}

/// What a decision reads of the cache (SD §4.4, §4.5).
impl Kv {
    /// The tiers on which `cache` expects each of an admission's lookup keys of one class, read
    /// by an owner on `group`: [`DEVICE`] when `group` holds its device copy, [`HOST`] and
    /// [`REMOTE`] as `cache` holds them, 0 for a key it does not know. The rule of the hit (SD
    /// §4.4) turns them into what the admission pins, loads, clones and prefetches.
    pub(crate) fn expected_hit(&self, keys: &[Key], group: GroupId) -> Vec<u8> {
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

    /// How many of `keys` name blocks in the eviction order of `group`'s regions: on the device
    /// in `group`, held by no request of this scheduler. `available` counts their units as
    /// evictable (the buffer pool's word those of the installed head, `beyond_head` the rest),
    /// and a `Pin` of them for an owner on `group` takes those units from every allocation, the
    /// pinning request's own included, from the moment the buffer pool takes it until its
    /// `Pinned` is read and the request holds them in `cache` too (SD §3.5, §4.5).
    pub(crate) fn unheld(&self, keys: &[Key], group: GroupId) -> u32 {
        let unheld = keys.iter().filter(|key| {
            self.cache
                .get(&Cached::id(key))
                .is_some_and(|row| row.key == **key && row.ordered_in(group))
        });
        u32::try_from(unheld.count()).unwrap_or(u32::MAX)
    }

    /// Per region of its class in `group` (by index), the units of the block of `key` that
    /// `available` counts as evictable, and that a `Pin` of it for an owner on `group` takes from
    /// every allocation (SD §3.5 `fits`, §4.5): none while a request of this scheduler holds it or
    /// its device copy is not valid in `group`. A block of a class with a block size counts its
    /// units wherever it lies in the order: the buffer pool's word counts the installed head,
    /// `beyond_head` the rest. A block of a sized class states its own size, which `order` does
    /// not carry, so it counts `size`, its size as the pinning request computes it, only where the
    /// head this queue pair installed names it, which the buffer pool's word counts, and nothing
    /// beyond the head.
    pub(crate) fn evictable(&self, key: &Key, group: &WorkerGroup, size: u32) -> Vec<(usize, u32)> {
        let id = Cached::id(key);
        let Some(row) = self
            .cache
            .get(&id)
            .filter(|row| row.key == *key && row.ordered_in(group.id()))
        else {
            return Vec::new();
        };
        let regions = group.regions(row.class).iter().map(|&r| usize::from(r));
        regions
            .filter_map(|region| match self.class_units(row.class, region) {
                0 => {
                    let installed = self.installed.get(region);
                    let head = installed.is_some_and(|head| head.contains(&id));
                    head.then_some((region, size))
                }
                units => Some((region, units)),
            })
            .collect()
    }

    /// The cached block of `id` (a policy's `PageId` or `EncodingId`) when it is on the device
    /// with no request of this scheduler holding it, a victim the policy may name (SD §3.6), and
    /// its blocks lie in `region` (an index) on the side of `groups`: its key and class.
    pub(crate) fn victim(&self, id: u64, region: usize, groups: &[WorkerGroup]) -> Option<Key> {
        let row = self
            .cache
            .get(&id)
            .filter(|row| row.pins == 0 && row.tiers & DEVICE != 0)?;
        let lies = self
            .classes_in(region, groups)
            .any(|held| held == row.class);
        lies.then_some(row.key)
    }

    /// The class of the cached block of `key`, when `cache` knows it.
    pub(crate) fn class_of(&self, key: &Key) -> Option<u8> {
        self.cache.get(&Cached::id(key)).map(|row| row.class)
    }

    /// A group that holds `row`'s device copy (`Cached::groups`), the first of `groups` that
    /// does: the group a `Store` of the block to the host names, whose workers copy it (SD §3.4,
    /// §4.4). `None` for a block with no device copy.
    fn holder(row: &Cached, groups: &[WorkerGroup]) -> Option<GroupId> {
        if row.tiers & DEVICE == 0 {
            return None;
        }
        groups
            .iter()
            .map(WorkerGroup::id)
            .find(|group| row.groups & bit(*group) != 0)
    }

    /// Every block of `cache` as a policy's snapshot states it (SD §5.2): a `PageView` per KV
    /// block and an `EncodingView` per encoder output.
    pub(crate) fn cache_views(&self, groups: &[WorkerGroup]) -> (Vec<PageView>, Vec<EncodingView>) {
        let (mut pages, mut encodings) = (Vec::new(), Vec::new());
        for row in self.cache.values() {
            if self.is_page(row.class) {
                pages.push(row.page());
            } else if self.is_encoding(row.class, groups) {
                encodings.push(row.encoding());
            }
        }
        (pages, encodings)
    }
}

/// What a step asks of the buffer pool for the cache (SD §3.1, §3.4, §4.4, §4.5).
impl Kv {
    /// The `Victims` each region of the queue pair needs before the step's requests are written,
    /// where the head of its order is not the one installed, and whether a `List` is due (SD
    /// §3.1, §4.5): `Victims { region, keys }` with the leading keys of the order whose units
    /// cover what one step's allocations can take there and `taking`, per region the units the
    /// requests not yet written take, a `Load` of an admission's host hit included (the whole
    /// order, as far as a record holds it, while `shrinking`, and while a region holds more units
    /// than its bound, `resize.rs`). The caller holds them ahead of those requests, which
    /// `available` lets take those units. `List` is due when a region's `evictable` word counts
    /// more units than the scheduler can account for, beyond `pending` blocks the answers not
    /// read yet may have made shared and unpinned (a `Commit`'s block, a `Store`'s and a
    /// `Prefetch`'s keys), and no `List` is unanswered: blocks another queue pair or a closed
    /// session committed, which this queue pair's allocations evict only once a `Victims` names
    /// them.
    ///
    /// One step's allocations take, per class taken from a region, the blocks of the
    /// configuration's `max_num_batched_tokens` tokens of the class's own block size and
    /// [`LOW_WATERMARK`] blocks for each of the `running` requests ([`Kv::reserve`]).
    pub(crate) fn victims(
        &mut self,
        (taking, pending): (&[u32], u32),
        (shrinking, running): (bool, usize),
        config: &SchedulerConfig,
        groups: &[WorkerGroup],
    ) -> (Vec<runtime::Request>, bool) {
        let fits = self.record.saturating_sub(VICTIMS_RECORD) / size_of::<Key>();
        let mut installing = Vec::new();
        for (number, region) in self.used.clone() {
            let taken = taking.get(region).copied().unwrap_or(0);
            // A region that shrinks may give up every block this scheduler knows to be unpinned.
            let whole = shrinking || self.excess(region) > 0;
            let reserve =
                (!whole).then(|| self.reserve(region, running, config, groups).max(taken));
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
                installing.push(runtime::Request::Victims {
                    region: number,
                    keys,
                });
                if let Some(installed) = self.installed.get_mut(region) {
                    *installed = head;
                }
            }
        }

        // The order cannot count the units of blocks that state their own size.
        let unknown = self.used.iter().any(|&(_, region)| {
            let largest = self
                .classes_in(region, groups)
                .map(|class| self.class_units(class, region))
                .max()
                .unwrap_or(0);
            let known = self
                .ordered
                .get(region)
                .copied()
                .unwrap_or(0)
                .saturating_add(pending.saturating_mul(largest));
            self.classes_in(region, groups)
                .all(|class| self.classes[usize::from(class)].units > 0)
                && self.word(region, 2) > known
        });
        let asked = self.lists_asked > self.lists_answered;
        (installing, unknown && !asked)
    }

    /// The units one step's allocations can take from `region` (by index), SD §4.5: per class
    /// taken from it, the blocks of `max_num_batched_tokens` tokens of the class's own block size
    /// and [`LOW_WATERMARK`] blocks per running request. A cache group's class (class `c` is
    /// cache group `c`) holds the block size its groups state for it; a sized class, whose blocks
    /// hold no tokens, is counted in blocks of the deployment's block size
    /// (`SchedulerConfig::block_tokens`).
    fn reserve(
        &self,
        region: usize,
        running: usize,
        config: &SchedulerConfig,
        groups: &[WorkerGroup],
    ) -> u32 {
        let tokens = u32::try_from(config.max_num_batched_tokens).unwrap_or(u32::MAX);
        let deployment = config.block_tokens;
        let running = u32::try_from(running).unwrap_or(u32::MAX);

        let reserve = self.classes_in(region, groups).map(|class| {
            let block = groups
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

    /// The policy's `offload` (SD §3.4): one `Store` to the host per group whose memory holds
    /// the device copies of the blocks it names, with their keys in its order. A `PageId` that is
    /// not in `cache`, not on the device, already on the host or in `storing` (keys a `Store` to
    /// the host the scheduler has not seen answered names: a block is offered to the host once,
    /// SD §4.4), whose last `Store` stored nothing while nothing that could change that has
    /// happened since ([`Kv::store_refused`]), or of a class without a host region is skipped.
    ///
    /// A policy names a block once. A block whose last `Store` to the host stored nothing is
    /// offered again, after the policy's blocks and least recently used first, once something
    /// that could change that answer has happened, while no request holds it and it has a device
    /// copy and no host copy (SD §4.4 "The write policy").
    pub(crate) fn offload(
        &mut self,
        pages: &[PageId],
        storing: &HashSet<Key>,
        groups: &[WorkerGroup],
    ) -> Vec<runtime::Request> {
        // The blocks offered again; one still held back stays recorded, and one with a host copy
        // or no device copy leaves the record.
        let mut again: Vec<(u64, u64)> = Vec::new();
        let mut held_back = HashSet::new();
        for id in std::mem::take(&mut self.unstored) {
            let Some(row) = self.cache.get(&id) else {
                continue;
            };
            if row.tiers & HOST != 0 || row.tiers & DEVICE == 0 {
                continue;
            }
            if row.pins > 0 || storing.contains(&row.key) || self.store_refused(row) {
                held_back.insert(id);
            } else {
                again.push((row.last_used, id));
            }
        }
        self.unstored = held_back;
        again.sort_unstable();

        let mut offered = HashSet::new();
        let mut stores: Vec<(GroupId, Vec<Key>)> = Vec::new();
        let named = pages.iter().map(|page| page.0);
        for id in named.chain(again.into_iter().map(|(_, id)| id)) {
            let Some(row) = self.cache.get(&id) else {
                continue;
            };
            let hosted = self
                .classes
                .get(usize::from(row.class))
                .is_some_and(|class| class.host.is_some());
            if row.tiers & HOST != 0
                || !hosted
                || storing.contains(&row.key)
                || self.store_refused(row)
                || !offered.insert(id)
            {
                continue;
            }
            let Some(group) = Self::holder(row, groups) else {
                continue;
            };

            match stores.iter_mut().find(|(store, _)| *store == group) {
                Some((_, keys)) => keys.push(row.key),
                None => stores.push((group, vec![row.key])),
            }
        }

        stores
            .into_iter()
            .map(|(group, keys)| runtime::Request::Store {
                group,
                keys,
                tier: Tier::Host,
            })
            .collect()
    }

    /// The cached blocks of region `region` (an index) a shrink to `units` evicts, in the
    /// eviction order the scheduler installs, beyond the units the region has available, each
    /// with its key: those with no host copy, which a demotion offers to the host tier first
    /// (`Message::Demote`).
    pub(crate) fn demotion(&self, region: usize, units: u32) -> Vec<(PageId, Key)> {
        let held = self.capacity(region);
        let free = self.available(region);
        let wanted = held.saturating_sub(units).saturating_sub(free);

        let mut freed = 0u32;
        let mut pages = Vec::new();
        for &(_, id) in self.order.get(region).into_iter().flatten() {
            if freed >= wanted {
                break;
            }
            let Some(row) = self.cache.get(&id) else {
                continue;
            };
            freed = freed.saturating_add(self.class_units(row.class, region));
            if row.tiers & HOST == 0 {
                pages.push((PageId(id), row.key));
            }
        }
        pages
    }

    /// `Flush` (SD §2.4): `Evict`s that name every key of `cache`, as many per record as a record
    /// holds; the buffer pool evicts those that no request pins, and `cache` keeps the keys its
    /// `Evicted` does not list.
    pub(crate) fn flush(&self) -> Vec<runtime::Request> {
        let fits = self.record.saturating_sub(EVICT_RECORD) / size_of::<Key>();
        let keys: Vec<Key> = self.cache.values().map(|row| row.key).collect();
        keys.chunks(fits.max(1))
            .map(|keys| runtime::Request::Evict {
                keys: keys.to_vec(),
            })
            .collect()
    }
}

/// The bit of `group` in [`Cached::groups`].
pub(crate) fn bit(group: GroupId) -> u8 {
    1u8.checked_shl(u32::from(group.0)).unwrap_or(0)
}

/// The groups that stamped a request, as bits of [`Cached::groups`]: those whose device block
/// tables held its entry.
pub(crate) fn stamped_groups(after: &runtime::Stamps) -> u8 {
    after.iter().fold(0, |bits, &(group, _)| bits | bit(group))
}

/// `request` as the scheduler applies its answer `completion`: a `Load` or a `Store` answered
/// `Copy` narrowed to the keys the copy covers, which are the leading keys of a `Load` that have
/// a host copy and the keys of a `Store` whose device copy is valid in its group (BP §11), and
/// for which the answer to its `Finish` speaks; a `Load` answered `Loaded` itself narrowed to no
/// key, since no leading key had a host copy the owner could not pin instead, so nothing was
/// copied and no entry was given (BP §11); any other request as it was written. The executor is
/// handed the request as it was written, which its completion answers (SD §2.1
/// `Executor::map`).
pub(crate) fn covered(request: &runtime::Request, completion: &Completion) -> runtime::Request {
    let mut narrowed = request.clone();
    match (&mut narrowed, completion) {
        (
            runtime::Request::Load { keys, .. } | runtime::Request::Store { keys, .. },
            Completion::Copy { keys: copied, .. },
        ) => keys.retain(|key| copied.contains(key)),
        (runtime::Request::Load { keys, .. }, Completion::Loaded) => keys.clear(),
        _ => {}
    }
    narrowed
}

#[cfg(test)]
mod tests {
    //! The cache as completions change it, over the session of `scheduler::tests::bench`:
    //! group A's blocks of class `KV` lie in the region numbered 4 (index 0), group B's in
    //! region 5 (index 1). Most cases drive a [`Kv`] built alone; the last three drive it
    //! through a scheduler on a real queue pair, where the scheduler's policy and its answers
    //! to copies take part.

    use std::num::NonZeroU32;
    use std::sync::atomic::Ordering;

    use crate::batch::cache::CacheGroupKind;
    use proto::config::{KvCacheConfig, SchedulerConfig};
    use scheduling::{Event, PageId};
    use storage::Key;
    use storage::runtime::{self, Cause, Completion, OwnerId, Tier};

    use super::super::super::tests::{A, B, Bench, KV, OWNER_A, PRODUCT, attached, bench, key};
    use super::super::super::{Request, WorkerGroup};
    use super::super::tests::kv;
    use super::{Cached, DEVICE, HOST, Kv, REMOTE};

    /// The KV decisions of the bench's session alone, with request 1 attached to group A under
    /// `OWNER_A` as the holder of what they are told, and the events the policy is told.
    struct Alone {
        kv: Kv,
        groups: Vec<WorkerGroup>,
        config: SchedulerConfig,
        holder: Request,
        told: Vec<Event>,
    }

    fn alone() -> Alone {
        let (kv, groups) = kv(4096);
        Alone {
            kv,
            groups,
            config: SchedulerConfig::default(),
            holder: attached(&[(A, OWNER_A, 3)]),
            told: Vec::new(),
        }
    }

    impl Alone {
        /// The completion `completion` of `asked`, which acts for the holder while `held`.
        fn learn(&mut self, asked: &runtime::Request, completion: &Completion, held: bool) {
            let holder = held.then_some(&self.holder);
            let groups = &self.groups;
            self.kv
                .learn((asked, None), holder, completion, groups, &mut self.told);
        }

        /// `commit`, submitted and stamped by group A's workers, answered `Committed`.
        fn commit(&mut self, commit: runtime::Request, held: bool) {
            self.kv.submitted(&commit);
            let mut stamped = commit;
            if let runtime::Request::Commit { after, .. } = &mut stamped {
                after.push((A, 1));
            }
            self.learn(&stamped, &Completion::Committed, held);
        }

        /// A `Listing` of 64-token blocks of class `KV` whose device copies lie in group A, each
        /// with its tiers.
        fn list(&mut self, keys: &[(Key, u8)]) {
            let keys = keys
                .iter()
                .map(|&(key, tiers)| (key, KV, 64, tiers, 1, 0))
                .collect();
            self.kv.submitted(&runtime::Request::List);
            self.learn(
                &runtime::Request::List,
                &Completion::Listing { keys },
                false,
            );
        }

        /// The holder gave up `keys`.
        fn unpin(&mut self, keys: &[Key]) {
            let groups = &self.groups;
            let holder = Some(&mut self.holder);
            self.kv.unpin(holder, keys, groups, &mut self.told);
        }

        /// The `Victims` a step installs with nothing else not yet written, and whether it asks
        /// for a `List`.
        fn victims(&mut self) -> (Vec<runtime::Request>, bool) {
            let taking = vec![0; self.kv.regions().len()];
            let (config, groups) = (&self.config, &self.groups);
            self.kv.victims((&taking, 0), (false, 0), config, groups)
        }
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

    fn pin(keys: &[Key]) -> runtime::Request {
        runtime::Request::Pin {
            owner: OWNER_A,
            class: KV,
            at: 0,
            keys: keys.to_vec(),
        }
    }

    fn pinned(blocks: u32, host: u32) -> Completion {
        Completion::Pinned {
            blocks: (0..blocks).map(|block| vec![(4, block + 1, 1)]).collect(),
            host,
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
        let mut alone = alone();

        // The request still holds the block it committed: cached, held, in no order.
        alone.commit(commit(0, key(1)), true);
        assert_eq!(alone.victims(), (Vec::new(), false));

        // Given up, it joins the order, whose head is installed once.
        alone.unpin(&[key(1)]);
        assert_eq!(alone.victims(), (vec![victims(&[key(1)])], false));
        assert_eq!(alone.victims(), (Vec::new(), false));

        // A block committed for a request that left before the answer is in the order at once,
        // behind the block given up before it: a commit outlives its owner's `Detach` (BP §9,
        // R6), and the write policy may store the block.
        alone.commit(commit(1, key(2)), false);
        assert_eq!(alone.victims(), (vec![victims(&[key(1), key(2)])], false));

        // The policy saw the block enter held, its release, and the second enter given up.
        let events: Vec<(&str, PageId, bool)> = alone
            .told
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

    /// A block its request gave up before its `Commit` was answered enters the eviction order
    /// as it was given up, ahead of a block given up after it but committed first.
    #[test]
    fn a_block_given_up_before_its_commit_is_answered_keeps_the_rank_it_was_given_up_with() {
        let mut alone = alone();
        alone.kv.submitted(&commit(0, key(1)));
        alone.unpin(&[key(1)]);
        alone.commit(commit(1, key(2)), true);
        alone.unpin(&[key(2)]);

        let mut answered = commit(0, key(1));
        if let runtime::Request::Commit { after, .. } = &mut answered {
            after.push((A, 1));
        }
        alone.learn(&answered, &Completion::Committed, false);
        assert_eq!(alone.victims(), (vec![victims(&[key(1), key(2)])], false));
    }

    /// A request that gives up the blocks it holds tells the policy one `Released`, which lists
    /// them in the order they join the eviction order, the deepest first (SD §4.4): the order the
    /// policy keeps is the one the scheduler installs.
    #[test]
    fn a_release_of_many_blocks_is_one_event_in_the_eviction_order() {
        let mut alone = alone();
        for (at, byte) in [(0, 1), (1, 2), (2, 3)] {
            alone.commit(commit(at, key(byte)), true);
        }
        assert_eq!(alone.victims(), (Vec::new(), false));
        alone.told.clear();

        alone.unpin(&[key(1), key(2), key(3)]);
        let head = victims(&[key(3), key(2), key(1)]);
        assert_eq!(alone.victims(), (vec![head], false));
        assert!(
            matches!(&alone.told[..], [Event::Released { pages }] if *pages == [page(3), page(2), page(1)]),
            "{:?}",
            alone.told
        );
    }

    /// An admission that pins blocks no request held tells the policy one `Held`, which lists
    /// them in the order they were pinned, and they leave the eviction order (SD §4.4, §4.5); a
    /// block pinned by one more request is a `Hit`.
    #[test]
    fn a_pin_of_many_evictable_blocks_is_one_event_in_the_order_they_were_taken() {
        let mut alone = alone();
        alone.list(&[(key(1), DEVICE), (key(2), DEVICE), (key(3), DEVICE)]);
        let head = victims(&[key(1), key(2), key(3)]);
        assert_eq!(alone.victims(), (vec![head], false));
        alone.told.clear();

        alone.learn(&pin(&[key(3), key(1)]), &pinned(2, 0), true);
        alone.learn(&pin(&[key(3)]), &pinned(1, 0), true);
        assert_eq!(alone.victims(), (vec![victims(&[key(2)])], false));
        assert!(
            matches!(
                &alone.told[..],
                [Event::Held { pages }, Event::Hit { page: hit }]
                    if *pages == [page(3), page(1)] && *hit == page(3)
            ),
            "{:?}",
            alone.told
        );
    }

    /// Blocks a scheduler never used enter its eviction order as its startup `Listing` orders
    /// them (the blocks the serving schedulers would evict first first), not by their keys: a
    /// scheduler that starts beside a warm cache installs a head that names them in the
    /// listing's order.
    #[test]
    fn listed_blocks_enter_the_eviction_order_as_the_listing_orders_them() {
        let mut alone = alone();
        // One step of 128 tokens takes two blocks of 64 tokens: the head covers two units.
        alone.config.max_num_batched_tokens = 128;
        alone.list(&[(key(3), DEVICE), (key(1), DEVICE), (key(2), DEVICE)]);
        assert_eq!(alone.victims(), (vec![victims(&[key(3), key(1)])], false));
    }

    #[test]
    fn the_order_is_least_recently_used_first_and_available_counts_what_lies_beyond_its_head() {
        let mut alone = alone();
        // One step of 128 tokens takes two blocks of 64 tokens: the head covers two units.
        alone.config.max_num_batched_tokens = 128;
        alone.list(&[(key(1), DEVICE), (key(2), DEVICE), (key(3), DEVICE)]);
        assert_eq!(alone.victims(), (vec![victims(&[key(1), key(2)])], false));

        // The buffer pool's word counts the head; the KV decisions add the block beyond it.
        alone.kv.set_words(0, [5, 0, 0]);
        alone.kv.read_capacity(&[]);
        assert_eq!(alone.kv.available(0), 6);

        // A block pinned and given up again is the most recently used.
        alone.learn(&pin(&[key(1)]), &pinned(1, 0), true);
        alone.unpin(&[key(1)]);
        assert_eq!(alone.victims(), (vec![victims(&[key(2), key(3)])], false));
        alone.kv.read_capacity(&[]);
        assert_eq!(alone.kv.available(0), 6);

        // What the requests not yet answered take is not available.
        alone.kv.read_capacity(&[(0, 4)]);
        assert_eq!(alone.kv.available(0), 2);
    }

    /// The head of an order covers one step's tokens in blocks of the class's own size: a step
    /// of 128 tokens takes 8 blocks of a cache group of 16-token blocks, whatever block size the
    /// deployment states for its widest cache group (SD §4.5).
    #[test]
    fn the_head_covers_a_steps_tokens_in_blocks_of_the_classs_own_size() {
        let mut alone = alone();
        alone.config.max_num_batched_tokens = 128;
        let capacity = &mut alone.groups[0].capacity;
        capacity.kv_cache = Some(KvCacheConfig {
            block_size: NonZeroU32::new(64),
            ..KvCacheConfig::default()
        });
        capacity.cache_groups = vec![(CacheGroupKind::Full, 16)];

        let listed: Vec<(Key, u8)> = lookup(10).into_iter().map(|key| (key, DEVICE)).collect();
        alone.list(&listed);
        assert_eq!(alone.victims(), (vec![victims(&lookup(8))], false));
    }

    #[test]
    fn pins_evictions_and_refusals_correct_what_cache_expects() {
        let mut alone = alone();
        let both = DEVICE | HOST;
        alone.list(&[
            (key(1), DEVICE),
            (key(2), both),
            (key(3), DEVICE),
            (key(4), DEVICE),
            (key(5), DEVICE),
        ]);
        let keys = lookup(6);
        assert_eq!(
            alone.kv.expected_hit(&keys, A),
            [DEVICE, both, DEVICE, DEVICE, DEVICE, 0]
        );
        assert_eq!(alone.kv.expected_hit(&keys, B), [0, HOST, 0, 0, 0, 0]);

        // One key of three pinned: the other two are not valid in the group, and the first of
        // them has a host copy, the second none.
        alone.learn(&pin(&keys[..3]), &pinned(1, 1), true);
        assert_eq!(
            alone.kv.expected_hit(&keys, A),
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
        alone.learn(&allocate, &allocated, true);
        let clone = runtime::Request::Clone {
            owner: OWNER_A,
            class: KV,
            at: 2,
            key: key(5),
        };
        alone.learn(&clone, &Completion::Refused { cause: Cause::Key }, true);
        assert_eq!(alone.kv.expected_hit(&keys, A), [DEVICE, HOST, 0, 0, 0, 0]);
        let groups = &alone.groups;
        assert!(
            alone.kv.victim(Cached::id(&key(1)), 0, groups).is_none(),
            "it is held"
        );
        assert!(
            alone.kv.victim(Cached::id(&key(2)), 0, groups).is_none(),
            "on the host alone"
        );

        // Keys left on no tier are gone, and the policy was told.
        let freed: Vec<PageId> = alone
            .told
            .iter()
            .filter_map(|event| match event {
                Event::Freed { page } => Some(*page),
                _ => None,
            })
            .collect();
        assert_eq!(freed, [page(3), page(4), page(5)]);
    }

    /// Evictable units the eviction order cannot account for, beyond what the answers not read
    /// yet may have made shared, ask for one `List`, and no more while it is unanswered.
    #[test]
    fn evictable_units_the_order_cannot_account_for_ask_for_one_list() {
        let mut alone = alone();
        alone.list(&[(key(1), DEVICE)]);

        // Three unpinned units in the region, of which the scheduler knows one.
        alone.kv.set_words(0, [0, 0, 3]);
        let taking = vec![0; alone.kv.regions().len()];
        let (config, groups) = (&alone.config, &alone.groups);
        let pending = alone.kv.victims((&taking, 2), (false, 0), config, groups);
        assert_eq!(
            pending,
            (vec![victims(&[key(1)])], false),
            "two may be pending"
        );

        assert_eq!(alone.victims(), (Vec::new(), true));
        alone.kv.submitted(&runtime::Request::List);
        assert_eq!(alone.victims(), (Vec::new(), false), "one is asked");
    }

    /// A pin holds a key's device block; its host copy stays in the host region's order, which
    /// the buffer pool's `evictable` word of that region counts: no `List` is asked for it.
    #[test]
    fn a_pinned_blocks_host_copy_stays_in_the_host_regions_order() {
        let mut alone = alone();
        // The region numbered 6 (index 2) stands for the host region of class `KV`, its blocks
        // one unit.
        alone.kv.classes_mut()[usize::from(KV)].host = Some((2, 1));
        alone.kv.classes_mut()[usize::from(PRODUCT)].units = 1;
        alone.list(&[(key(1), DEVICE | HOST)]);
        let host_victims = runtime::Request::Victims {
            region: 6,
            keys: vec![key(1)],
        };
        assert_eq!(
            alone.victims(),
            (vec![victims(&[key(1)]), host_victims], false)
        );

        // A request pins the device block; the host copy is still evictable there, and the
        // device region's order loses the block: every evictable unit is accounted for.
        alone.kv.set_words(2, [0, 0, 1]);
        alone.learn(&pin(&[key(1)]), &pinned(1, 0), true);
        assert_eq!(alone.victims(), (vec![victims(&[])], false));
    }

    #[test]
    fn a_listing_states_the_whole_cache() {
        let mut alone = alone();
        alone.list(&[(key(1), DEVICE), (key(2), DEVICE | HOST)]);
        alone.list(&[(key(2), DEVICE), (key(3), HOST)]);
        assert_eq!(alone.kv.expected_hit(&lookup(3), A), [0, DEVICE, HOST]);

        // A block on the host alone is an offloaded page to the policy.
        let (pages, encodings) = alone.kv.cache_views(&alone.groups);
        let mut pages: Vec<(PageId, bool)> =
            pages.iter().map(|page| (page.id, page.on_host)).collect();
        pages.sort();
        assert_eq!(pages, [(page(2), false), (page(3), true)]);
        assert!(encodings.is_empty());
    }

    /// The remote store's answer names the keys it stored, and a prefetch brings the leading
    /// keys the source found to the host.
    #[test]
    fn a_remote_store_and_a_prefetch_place_the_keys_they_name() {
        let mut alone = alone();
        alone.list(&[(key(1), DEVICE)]);
        let remote = runtime::Request::Store {
            group: A,
            keys: vec![key(1)],
            tier: Tier::Remote,
        };
        let stored = Completion::Stored { keys: vec![key(1)] };
        alone.learn(&remote, &stored, false);
        let prefetch = runtime::Request::Prefetch {
            group: A,
            class: KV,
            units: 0,
            keys: vec![key(4), key(5), key(6)],
            deadline: 0,
        };
        let prefetched = Completion::Prefetched {
            found: 2,
            evicted: Vec::new(),
        };
        alone.learn(&prefetch, &prefetched, false);
        assert_eq!(
            alone.kv.expected_hit(&lookup(6), A),
            [DEVICE | REMOTE, 0, 0, HOST, HOST, 0]
        );
    }

    /// The request `Bench::attach` adds.
    const ID: proto::request::RequestId = proto::request::RequestId(1);

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

    /// A `Listing` of 64-token blocks of class `KV` whose device copies lie in group A, through
    /// the bench's queue pair.
    fn listed(bench: &mut Bench, keys: &[(Key, u8)]) {
        let keys = keys
            .iter()
            .map(|&(key, tiers)| (key, KV, 64, tiers, 1, 0))
            .collect();
        exchange(bench, runtime::Request::List, &Completion::Listing { keys });
    }

    /// An admission that reuses a warm cache, pinning 1000 blocks of 64 tokens no request held,
    /// tells the policy one `Held` of them: the step's `observe` and `dispatch` together consume
    /// at most half of a call's fuel (SD §5.2 "Faults"), where restating each block as a page
    /// costs about as much as a call has. The policy saw the pins: a shortage of device units
    /// takes none of the blocks.
    #[test]
    fn an_admission_pinning_a_thousand_cached_blocks_stays_well_within_a_calls_fuel() {
        use scheduling::decision::{Room, Shortage, Storage, Victim};

        use crate::scheduler::policy::tests::{packaged, packaged_arguments};
        use proto::config::{DEFAULT_POLICY_CALL_FUEL, PolicyBudget};

        use crate::scheduler::policy::{self, Policy};

        const BLOCKS: u64 = 1000;
        let numbered = |number: u64| {
            let mut bytes = [0; 32];
            bytes[..8].copy_from_slice(&(number + 1).to_be_bytes());
            Key(bytes)
        };
        let keys: Vec<Key> = (0..BLOCKS).map(numbered).collect();

        let mut bench = bench(1 << 20);
        let listing: Vec<(Key, u8)> = keys.iter().map(|&key| (key, DEVICE)).collect();
        listed(&mut bench, &listing);
        bench.attach(&[(A, OWNER_A, 3)]);
        let snapshot = bench.scheduler.snapshot();
        let budget = PolicyBudget::default();
        let ceilings = policy::ceilings([], &budget);
        let arguments = packaged_arguments("");
        bench.scheduler.policy = Policy::new(
            packaged("arrival"),
            arguments,
            snapshot,
            &ceilings.limits,
            &budget,
        )
        .unwrap();

        let pinned = Completion::Pinned {
            blocks: vec![vec![(4, 1, 1)]; keys.len()],
            host: 0,
        };
        exchange(&mut bench, pin(&keys), &pinned);

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
            spent <= DEFAULT_POLICY_CALL_FUEL.get() / 2,
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

    /// A scheduler that starts beside a warm cache, as an epoch's target of the same layout does
    /// (its regions are the serving epoch's), learns every block from its startup `Listing` after
    /// its policy was instantiated on an empty cache, and tells the policy each of them as a
    /// `Cached` page in the next step's `observe`. With the cache of E1's deployment after a few
    /// minutes of load (2,048 blocks of 64 tokens with a device copy, 4,094 more on the host
    /// alone), the packaged policy takes that step without a fault, and has seen the pages: a
    /// shortage of device units names listed pages, none of them on the host alone.
    #[test]
    fn a_policy_observes_a_warm_cache_listed_at_the_start_without_a_fault() {
        use std::collections::HashSet;

        use scheduling::decision::{Room, Shortage, Storage};

        use crate::scheduler::policy::tests::{packaged, packaged_arguments};
        use proto::config::PolicyBudget;

        use crate::scheduler::policy::{self, Policy};

        const ON_DEVICE: u64 = 2048;
        const ON_HOST: u64 = 4094;
        let numbered = |number: u64| {
            let mut bytes = [0; 32];
            bytes[..8].copy_from_slice(&(number + 1).to_be_bytes());
            Key(bytes)
        };

        let mut bench = bench(1 << 22);
        let budget = PolicyBudget::default();
        let ceilings = policy::ceilings([], &budget);
        let empty = bench.scheduler.snapshot();
        let arguments = packaged_arguments("");
        bench.scheduler.policy = Policy::new(
            packaged("arrival"),
            arguments,
            empty,
            &ceilings.limits,
            &budget,
        )
        .unwrap();

        let listing: Vec<(Key, u8)> = (0..ON_DEVICE)
            .map(|number| (numbered(number), DEVICE | HOST))
            .chain((ON_DEVICE..ON_DEVICE + ON_HOST).map(|number| (numbered(number), HOST)))
            .collect();
        listed(&mut bench, &listing);

        let scheduler = &mut bench.scheduler;
        let room = Room {
            token_budget: 8192,
            call_budget: 128,
            page_tokens: 64,
            ..Room::default()
        };
        if let Err(fault) =
            scheduler
                .policy
                .dispatch(&room, &mut scheduler.requests, &scheduler.groups)
        {
            panic!("the step after the listing faults the policy: {fault}");
        }

        let shortage = Shortage {
            kind: Storage::KvUnits,
            need: 16,
        };
        let victims: Vec<u64> = scheduler
            .policy
            .preempt(&shortage)
            .unwrap()
            .into_iter()
            .filter_map(|victim| victim.page().map(|page| page.0))
            .collect();
        let on_device: HashSet<u64> = listing[..ON_DEVICE as usize]
            .iter()
            .map(|(key, _)| Cached::id(key))
            .collect();
        assert!(!victims.is_empty(), "the policy names no listed page");
        assert!(
            victims.iter().all(|page| on_device.contains(page)),
            "{victims:?}"
        );
    }

    /// A copy's request is applied as the keys its copy covers, and the answer to its `Finish`,
    /// stamped by the copy's group, places them for the scheduler's owner; the scheduler reads
    /// the capacity words the buffer pool writes for the KV decisions.
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
            parts: Vec::new(),
        };
        let answers = exchange(&mut bench, load(3), &copy);
        assert_eq!(answers[0].2, load(2));

        let mut finish = runtime::Request::Finish {
            copy: answers[0].0,
            ok: true,
            after: Vec::new(),
        };
        if let runtime::Request::Finish { after, .. } = &mut finish {
            after.push((A, 1));
        }
        bench.scheduler.stamped(vec![finish]);
        bench.scheduler.post().unwrap();
        let taken = bench.take();
        assert_eq!(taken.len(), 1, "the stamped request was written");
        bench.complete(taken[0].0, &Completion::Loaded);
        let answers = bench.answers();
        assert_eq!(answers[0].2, load(2));
        let kv = &bench.scheduler.kv;
        assert_eq!(kv.expected_hit(&lookup(3), A), [DEVICE, DEVICE, 0]);
        let groups = &bench.scheduler.groups;
        assert!(
            kv.victim(Cached::id(&key(1)), 0, groups).is_none(),
            "loaded for the request"
        );
        bench.scheduler.unpin(ID, &lookup(2));
        let kv = &bench.scheduler.kv;
        let groups = &bench.scheduler.groups;
        assert!(kv.victim(Cached::id(&key(1)), 0, groups).is_some());

        // The buffer pool's `available` word of region 4 is register 2; the two blocks given up
        // lie beyond the head the scheduler installed.
        bench.pool.register(2).store(7, Ordering::Release);
        bench.scheduler.read_capacity();
        assert_eq!(bench.scheduler.kv.available(0), 9);
    }
}
