//! What the buffer pool states about itself: the capacity words of a
//! region, what each block is observed as ([`BlockState`], the first row
//! that applies), what waits, and whether its tables are consistent.
//!
//! The capacity words are made of counts the regions keep: the free units,
//! the units of shared blocks nothing but their entry holds (evictable, and
//! available to the queue pairs whose order names their keys), and the
//! units of retired blocks. A block records what its units count as
//! ([`Counted`]), which follows from whether an entry holds it, how many
//! references make it private, and how many it has.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

use storage::protocol::{
    BlockCount, BlockState, BufferPoolListing, CopyListing, GroupListing, Key, Waiting,
};
use storage::runtime::Tier;

use super::{Block, BufferPool, ClientId, Copy, Counted, Deferred, Held, QueueId, SLOT, Target};

/// Per block, by its address: the key an entry caches it under.
type CachedBy = HashMap<*const Block, Key>;

/// Per block, by its address: the references that make it private.
type Owning = HashMap<*const Block, u32>;

/// What a block's units count as, with `references` references held: a
/// shared block only its entry holds is evictable, and a block no entry
/// holds and no reference makes private is retired.
fn counted(block: &Block, references: usize) -> Counted {
    match block.cached.get() {
        Some(key) if references == 1 => Counted::Evictable(key),
        Some(_) => Counted::Nothing,
        None if references > 0 && block.owned.get() == 0 => Counted::Retired,
        None => Counted::Nothing,
    }
}

impl BufferPool {
    /// Notes that the holders of a block changed: it is counted again once
    /// the handler running now is done ([`Self::settle_counts`]).
    pub(super) fn touch(&self, block: &Rc<Block>) {
        self.touched.borrow_mut().push(Rc::downgrade(block));
    }

    /// Counts again every block whose holders changed since the last time:
    /// the work of the requests that changed them, never a scan.
    pub(super) fn settle_counts(&mut self) {
        let touched = std::mem::take(self.touched.get_mut());
        for noted in touched {
            // A block gone meanwhile left its counts with its last reference.
            if let Some(block) = noted.upgrade() {
                // The reference upgraded here is no holder's.
                let now = counted(&block, Rc::strong_count(&block) - 1);
                self.count(&block, now);
            }
        }
    }

    /// The last reference to a block goes: its units count as nothing.
    pub(super) fn uncount(&mut self, block: &Block) {
        self.count(block, Counted::Nothing);
    }

    /// Moves a block's units from what they were counted as to `now`, in
    /// every region it has a run of.
    fn count(&mut self, block: &Block, now: Counted) {
        let was = block.counted.get();
        if was == now {
            return;
        }
        block.counted.set(now);
        for run in &block.runs {
            let region = &mut self.regions[usize::from(run.region())];
            for (state, adds) in [(was, false), (now, true)] {
                match state {
                    Counted::Nothing => {}
                    Counted::Evictable(key) => region.count_evictable(&key, run.count(), adds),
                    Counted::Retired => region.count_retiring(run.count(), adds),
                }
            }
        }
    }

    /// Whether a copy of the session `client` makes the block it writes
    /// private: it writes for a target that exists, in place of no content
    /// others read.
    pub(super) fn live(&self, client: ClientId, copy: &Copy) -> bool {
        self.target_exists(client, &copy.to) && !written_in_place(copy)
    }

    /// By a scan of every holder: the key each cached block is cached
    /// under, and the references that make each block private.
    fn holders(&self) -> (CachedBy, Owning) {
        let mut cached = HashMap::new();
        for (key, entry) in &self.cache {
            for (block, _) in entry.device.iter().chain(&entry.host) {
                cached.insert(Rc::as_ptr(block), *key);
            }
        }
        let mut owned: Owning = HashMap::new();
        let mut own = |block: &Rc<Block>| *owned.entry(Rc::as_ptr(block)).or_default() += 1;
        for owner in self.owners.values() {
            own(&owner.slot);
            for held in owner.tables.iter().flat_map(BTreeMap::values) {
                if let Held::Private(block) = held {
                    own(block);
                }
            }
        }
        for (_, work) in self.groups.values().flat_map(|group| &group.deferred) {
            if let Deferred::Commit { block, .. } = work {
                own(block);
            }
        }
        for ((queue, _), copies) in &self.copies {
            for copy in copies.iter().filter(|copy| self.live(queue.0, copy)) {
                own(&copy.write);
            }
        }
        (cached, owned)
    }

    /// Counts every block from its holders, and every region afresh: of a
    /// pool made from an image, which leaves the counts out.
    pub(super) fn recount(&mut self) {
        for region in &mut self.regions {
            region.recount();
        }
        let (cached, owned) = self.holders();
        let blocks: Vec<Rc<Block>> = self
            .blocks()
            .into_iter()
            .map(|(block, _, _)| Rc::clone(block))
            .collect();
        for block in blocks {
            let at = Rc::as_ptr(&block);
            block.cached.set(cached.get(&at).copied());
            block.owned.set(owned.get(&at).copied().unwrap_or(0));
            block.counted.set(Counted::Nothing);
            // The reference cloned here is no holder's.
            let now = counted(&block, Rc::strong_count(&block) - 1);
            self.count(&block, now);
        }
        self.touched.get_mut().clear();
    }
    /// Whether what a copy writes for still exists.
    fn target_exists(&self, client: ClientId, target: &Target) -> bool {
        match target {
            Target::Owner { owner, .. } | Target::Slot { owner } | Target::Group { owner, .. } => {
                self.owners.contains_key(&(client, *owner))
            }
            Target::Entry { .. } => true,
        }
    }

    /// Every block that something holds, with what it is observed as and
    /// the references its holders have: the first row that applies of
    /// shared (an entry of the cache), private (an owner's slot or private
    /// entry, a waiting commit, a copy writing it for a target that exists)
    /// and retired (only deferred releases, ended copies and reads).
    pub(super) fn blocks(&self) -> Vec<(&Rc<Block>, BlockState, usize)> {
        let mut found: HashMap<*const Block, (&Rc<Block>, BlockState, usize)> = HashMap::new();
        let mut hold = |block, observed: BlockState| {
            let entry = found
                .entry(Rc::as_ptr(block))
                .or_insert((block, observed, 0));
            entry.1 = entry.1.min(observed);
            entry.2 += 1;
        };

        for entry in self.cache.values() {
            for (block, _) in entry.device.iter().chain(&entry.host) {
                hold(block, BlockState::Shared);
            }
        }
        for owner in self.owners.values() {
            hold(&owner.slot, BlockState::Private);
            for held in owner.tables.iter().flat_map(BTreeMap::values) {
                match held {
                    Held::Private(block) => hold(block, BlockState::Private),
                    // A pin: the block is shared by its entry of the cache.
                    Held::Shared(block) => hold(block, BlockState::Retired),
                }
            }
        }
        for (_, work) in self.groups.values().flat_map(|group| &group.deferred) {
            match work {
                Deferred::Release { blocks } => blocks
                    .iter()
                    .for_each(|block| hold(block, BlockState::Retired)),
                Deferred::Commit { block, .. } => hold(block, BlockState::Private),
                Deferred::Transfer { .. } | Deferred::Finish { .. } => {}
            }
        }
        // A peer's hold reads a cached block, as a pin does.
        for block in self.peers.values().flatten() {
            hold(block, BlockState::Retired);
        }
        for ((queue, _), copies) in &self.copies {
            for copy in copies {
                copy.reads
                    .iter()
                    .for_each(|block| hold(block, BlockState::Retired));
                let written = self.target_exists(queue.0, &copy.to) && !written_in_place(copy);
                hold(
                    &copy.write,
                    if written {
                        BlockState::Private
                    } else {
                        BlockState::Retired
                    },
                );
            }
        }
        found.into_values().collect()
    }
}

/// Whether a copy writes a block that its holders keep reading: one more
/// memory of a block, or the remote store.
fn written_in_place(copy: &Copy) -> bool {
    copy.fill || matches!(copy.to, Target::Entry { remote: true, .. })
}

impl BufferPool {
    /// The number of blocks in each observed state, per region and class.
    pub fn counts(&self) -> BTreeMap<(u8, u8, BlockState), u32> {
        let mut counts = BTreeMap::new();
        for (block, observed, _) in self.blocks() {
            *counts
                .entry((block.region(), block.class, observed))
                .or_default() += 1;
        }
        counts
    }

    /// What the pool of `deployment` exports to an operator: per region and
    /// class the blocks in each observed state, every group of an open
    /// session with the stamps of what waits on it, and every request whose
    /// copies are in flight. A transfer that waits for a copy to end waits
    /// at stamp 0.
    pub fn listing(&self, deployment: &str) -> BufferPoolListing {
        let blocks = self
            .counts()
            .into_iter()
            .map(|((region, class, state), blocks)| {
                let region = self.region_name(region).unwrap_or_default().to_owned();
                let class = (class != SLOT).then_some(class);
                BlockCount {
                    region,
                    class,
                    state,
                    blocks,
                }
            });
        let waiting = |work: &Deferred| match work {
            Deferred::Release { .. } => Waiting::Release,
            Deferred::Commit { .. } => Waiting::Commit,
            Deferred::Transfer { .. } => Waiting::Transfer,
            Deferred::Finish { .. } => Waiting::Finish,
        };
        let mut groups: Vec<GroupListing> = self
            .groups
            .iter()
            .map(|((client, group), found)| GroupListing {
                epoch: self.epoch(*client).unwrap_or_default(),
                group: group.0,
                completed: found.completed,
                waiting: found
                    .deferred
                    .iter()
                    .map(|(stamp, work)| (*stamp, waiting(work)))
                    .collect(),
            })
            .collect();
        groups.sort_by_key(|group| (group.epoch, group.group));
        let mut copies: Vec<CopyListing> = self
            .copies
            .iter()
            .map(|((queue, request), copies)| CopyListing {
                epoch: self.epoch(queue.0),
                scheduler: queue.1,
                request: request.0,
                copies: copies.len() as u32,
                deadline: copies.first().and_then(|copy| copy.deadline),
            })
            .collect();
        copies.sort_by_key(|copy| (copy.epoch, copy.scheduler, copy.request));
        BufferPoolListing {
            deployment: deployment.to_owned(),
            blocks: blocks.collect(),
            groups,
            copies,
        }
    }

    /// What `Open` states of a region: its tier, the bytes of a unit, and
    /// its units.
    pub fn extent(&self, region: u8) -> Option<(Tier, u64, u32)> {
        let found = self.regions.get(usize::from(region))?;
        Some((found.tier, found.unit_bytes, found.units()))
    }

    /// The capacity words of a region for a queue pair: `available`, the
    /// units an allocation of the queue pair can take (free units and those
    /// of unpinned blocks in the order it installed); `retiring`, the units
    /// of retired blocks; `evictable`, the units of every shared block
    /// without a pin, in any order or none.
    pub fn capacity(&self, queue: QueueId, region: u8) -> (u32, u32, u32) {
        let found = &self.regions[usize::from(region)];
        (found.available(queue), found.retiring, found.evictable)
    }

    /// The capacity words of every region and queue pair, by a scan of
    /// every holder: what the counts must equal. Per region: its free
    /// units, evictable and retiring units, and per queue pair with an
    /// order there its installed units.
    fn scanned_words(&self) -> Vec<(u32, u32, u32, BTreeMap<QueueId, u32>)> {
        let blocks = self.blocks();
        let mut words: Vec<_> = self
            .regions
            .iter()
            .map(|region| (region.free_scanned(), 0, 0, BTreeMap::new()))
            .collect();
        let mut evictable: HashSet<*const Block> = HashSet::new();
        for (block, observed, _) in &blocks {
            let only_entry = *observed == BlockState::Shared && Rc::strong_count(block) == 1;
            if only_entry {
                evictable.insert(Rc::as_ptr(block));
            }
            for run in &block.runs {
                let counts = &mut words[usize::from(run.region())];
                if only_entry {
                    counts.1 += run.count();
                } else if *observed == BlockState::Retired {
                    counts.2 += run.count();
                }
            }
        }
        for (id, region) in self.regions.iter().enumerate() {
            for (queue, order) in &region.orders {
                let distinct: HashSet<&Key> = order.iter().collect();
                let installed = distinct
                    .into_iter()
                    .filter_map(|key| self.cache.get(key)?.copy_in(id as u8))
                    .filter(|block| evictable.contains(&Rc::as_ptr(block)))
                    .map(|block| block.units())
                    .sum();
                words[id].3.insert(*queue, installed);
            }
        }
        words
    }

    /// Checks the counts the capacity words are made of against a scan:
    /// what each block records against its holders, and each region's
    /// counts against its blocks and orders. Returns the first difference.
    fn check_counts(&self) -> Result<(), String> {
        if !self.touched.borrow().is_empty() {
            return Err("a handler left blocks it touched uncounted".into());
        }
        let (cached, owned) = self.holders();
        for (block, _, references) in self.blocks() {
            let at = Rc::as_ptr(block);
            let recorded = (block.cached.get(), block.owned.get());
            let held = (
                cached.get(&at).copied(),
                owned.get(&at).copied().unwrap_or(0),
            );
            let first = block.stated().next();
            if recorded != held {
                return Err(format!(
                    "block {first:?} records {recorded:?} of cache and owners, its holders {held:?}"
                ));
            }
            if block.counted.get() != counted(block, references) {
                return Err(format!(
                    "block {first:?} counts as {:?}, its holders make it {:?}",
                    block.counted.get(),
                    counted(block, references)
                ));
            }
        }
        for (id, (region, words)) in self.regions.iter().zip(self.scanned_words()).enumerate() {
            let (free, evictable, retiring, installed) = words;
            let counts = (region.free_units(), region.evictable, region.retiring);
            if counts != (free, evictable, retiring) {
                return Err(format!(
                    "region {id} counts {counts:?} free, evictable and retiring units, a scan \
                     {:?}",
                    (free, evictable, retiring)
                ));
            }
            let kept: BTreeMap<QueueId, u32> = region
                .installed
                .iter()
                .map(|(queue, (_, units))| (*queue, *units))
                .collect();
            let named = region.installed.iter().all(|(queue, (keys, _))| {
                let order = region.orders.get(queue).map_or(&[][..], Vec::as_slice);
                *keys == order.iter().copied().collect::<HashSet<Key>>()
            });
            if kept != installed || !named {
                return Err(format!(
                    "region {id} counts {kept:?} units in the queue pairs' orders, a scan \
                     {installed:?}"
                ));
            }
        }
        Ok(())
    }

    /// Checks that the tables are consistent: every unit of every region is
    /// in exactly one of its free list, the runs outside it, and one block;
    /// every block is held as often as its references say; a private block
    /// is in one place; a pin is of a cached block; an entry of an owner's
    /// table is valid in the owner's group; no two regions hold the same
    /// bytes; the counts the capacity words are made of are those a scan
    /// finds. Returns the first inconsistency.
    pub fn check(&self) -> Result<(), String> {
        let blocks = self.blocks();

        // P3 and P1: the units of each region, each in exactly one place.
        for (id, region) in self.regions.iter().enumerate() {
            let mut runs: Vec<(u32, u32)> =
                region.free_runs().chain(region.outside_runs()).collect();
            let held = blocks
                .iter()
                .flat_map(|(block, _, _)| &block.runs)
                .filter(|run| usize::from(run.region()) == id);
            runs.extend(held.map(|run| (run.first(), run.count())));
            runs.sort_unstable();

            let (mut next, limit) = region.span();
            for (first, count) in runs {
                if first != next {
                    return Err(format!(
                        "region {id}: unit {next} is in no run and unit {first} follows"
                    ));
                }
                next = first + count;
            }
            if next != limit {
                return Err(format!("region {id}: units end at {next}, not at {limit}"));
            }
            let free: u32 = region.free_runs().map(|(_, count)| count).sum();
            let outside: u32 = region.outside_runs().map(|(_, count)| count).sum();
            if region.units() + outside != limit - region.span().0 || free > region.units() {
                return Err(format!(
                    "region {id}: {} units, {free} free, {outside} outside",
                    region.units()
                ));
            }
        }

        // R2: a block has no holder the tables do not show.
        for (block, _, references) in &blocks {
            if Rc::strong_count(block) != *references {
                let (count, first) = (Rc::strong_count(block), block.stated().next());
                return Err(format!(
                    "block {first:?} has {count} references, its holders {references}"
                ));
            }
        }

        // P1: a private block is in one place.
        let mut private: HashMap<*const Block, usize> = HashMap::new();
        let owned = self.owners.values().flat_map(|owner| {
            let entries = owner.tables.iter().flat_map(BTreeMap::values);
            let private = entries
                .filter(|held| matches!(held, Held::Private(_)))
                .map(Held::block);
            private.chain(std::iter::once(&owner.slot))
        });
        let written = self
            .copies
            .values()
            .flatten()
            .filter(|copy| !written_in_place(copy) && !matches!(copy.to, Target::Slot { .. }));
        for block in owned.chain(written.map(|copy| &copy.write)) {
            *private.entry(Rc::as_ptr(block)).or_default() += 1;
        }
        let cached = |block: &Rc<Block>| {
            blocks.iter().any(|(other, observed, _)| {
                Rc::ptr_eq(other, block) && *observed == BlockState::Shared
            })
        };
        for (block, _, _) in &blocks {
            let places = private.get(&Rc::as_ptr(block)).copied().unwrap_or(0);
            if places > 1 || places == 1 && cached(block) {
                return Err(format!(
                    "block {:?} is private in {places} places, cached {}",
                    block.stated().next(),
                    cached(block)
                ));
            }
        }

        // A peer holds only host copies of the cache.
        for (peer, held) in &self.peers {
            if let Some(block) = held.iter().find(|block| !cached(block)) {
                return Err(format!(
                    "peer {peer} holds block {:?}, which is not cached",
                    block.stated().next()
                ));
            }
        }

        // P2: a pin is of a cached block; a table's entry is valid in its owner's group.
        for ((client, owner), found) in &self.owners {
            let memory = self.memory(*client, found.group);
            for held in found.tables.iter().flat_map(BTreeMap::values) {
                let block = held.block();
                if matches!(held, Held::Shared(_)) && !cached(block) {
                    return Err(format!(
                        "owner {} pins block {:?}, which is not cached",
                        owner.0,
                        block.stated().next()
                    ));
                }
                if block.valid.get() & memory == 0 {
                    return Err(format!(
                        "owner {} holds block {:?}, which is not valid in its group",
                        owner.0,
                        block.stated().next()
                    ));
                }
            }
        }

        // P1: a copy fills only memory in which its block is not valid.
        for copy in self.copies.values().flatten() {
            if copy.fill && copy.write.class != SLOT && copy.write.valid.get() & copy.memory != 0 {
                return Err(format!(
                    "a copy fills valid memory of block {:?}",
                    copy.write.stated().next()
                ));
            }
        }

        // R10: within a memory space, the bytes of the regions' units and of the quota are apart.
        for (space, quota) in self.quota.iter().enumerate() {
            let mut ranges: Vec<(u64, u64)> = quota
                .iter()
                .map(|(offset, bytes)| (*offset, *bytes))
                .collect();
            for region in self
                .regions
                .iter()
                .filter(|region| usize::from(region.space) == space)
            {
                let inside = region.free_runs().chain(
                    blocks
                        .iter()
                        .flat_map(|(block, _, _)| &block.runs)
                        .filter(|run| {
                            std::ptr::eq(&self.regions[usize::from(run.region())], region)
                        })
                        .map(|run| (run.first(), run.count())),
                );
                ranges.extend(inside.map(|(first, count)| {
                    (
                        u64::from(first) * region.unit_bytes,
                        u64::from(count) * region.unit_bytes,
                    )
                }));
            }
            ranges.sort_unstable();
            for pair in ranges.windows(2) {
                if pair[0].0 + pair[0].1 > pair[1].0 {
                    return Err(format!(
                        "memory space {space}: bytes at {} are in two places",
                        pair[1].0
                    ));
                }
            }
        }
        self.check_counts()
    }
}
