//! What the buffer pool states about itself: the capacity words of a
//! region, what each block is observed as ([`BlockState`], the first row
//! that applies), what waits, and whether its tables are consistent.

use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use storage::protocol::{
    BlockCount, BlockState, BufferPoolListing, CopyListing, GroupListing, Waiting,
};
use storage::runtime::Tier;

use super::{Block, BufferPool, ClientId, Copy, Deferred, Held, QueueId, SLOT, Target};

impl BufferPool {
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
        let unheld = |key| {
            let copy = self.cache.get(key).and_then(|entry| entry.copy_in(region));
            copy.filter(|block| Rc::strong_count(block) == 1)
                .map_or(0, |block| block.units())
        };
        let evictable = self.cache.keys().map(unheld).sum();
        let mut order: Vec<_> = found
            .orders
            .get(&queue)
            .map_or(&[][..], Vec::as_slice)
            .to_vec();
        order.sort_unstable();
        order.dedup();
        let installed: u32 = order.iter().map(unheld).sum();

        // A block is retired when every reference to it is a deferred
        // release's, a read's, or a write's for a target that is gone.
        let mut passing: HashMap<*const Block, (&Rc<Block>, usize)> = HashMap::new();
        let mut pass = |block| passing.entry(Rc::as_ptr(block)).or_insert((block, 0)).1 += 1;
        for (_, work) in self.groups.values().flat_map(|group| &group.deferred) {
            if let Deferred::Release { blocks } = work {
                blocks.iter().for_each(&mut pass);
            }
        }
        self.peers.values().flatten().for_each(&mut pass);
        for ((of, _), copies) in &self.copies {
            for copy in copies {
                copy.reads.iter().for_each(&mut pass);
                if !self.target_exists(of.0, &copy.to) {
                    pass(&copy.write);
                }
            }
        }
        let retired = passing
            .values()
            .filter(|(block, references)| Rc::strong_count(block) == *references);
        let retiring = retired
            .map(|(block, _)| {
                block
                    .runs
                    .iter()
                    .filter(|run| run.region() == region)
                    .map(super::Run::count)
                    .sum::<u32>()
            })
            .sum();

        (found.free_units() + installed, retiring, evictable)
    }

    /// Checks that the tables are consistent: every unit of every region is
    /// in exactly one of its free list, the runs outside it, and one block;
    /// every block is held as often as its references say; a private block
    /// is in one place; a pin is of a cached block; an entry of an owner's
    /// table is valid in the owner's group; no two regions hold the same
    /// bytes. Returns the first inconsistency.
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
        Ok(())
    }
}
