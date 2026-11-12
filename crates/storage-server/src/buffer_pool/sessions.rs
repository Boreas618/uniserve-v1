//! Sessions, regions and epochs: what `Open` and `Close` on the Storage
//! Server's socket, the loss of an engine and the end of an update's trial
//! do to the buffer pool.

use std::collections::{BTreeMap, VecDeque};

use storage::runtime::{GroupId, Tier};

use super::{
    BufferPool, Class, ClientId, Deferred, Group, Held, Region, insert_range, remove_range,
};

impl BufferPool {
    /// Adds a range of a memory space to its free ranges: its memory exists
    /// on every rank that maps the space and is in no region.
    pub fn grant(&mut self, space: u8, offset: u64, bytes: u64) {
        while self.quota.len() <= usize::from(space) {
            self.quota.push(BTreeMap::new());
        }
        insert_range(&mut self.quota[usize::from(space)], offset, bytes);
    }

    /// Takes a range out of a memory space's free ranges, before its
    /// memory is released. Returns false, and changes nothing, when a part
    /// of the range is in a region.
    pub fn withdraw(&mut self, space: u8, offset: u64, bytes: u64) -> bool {
        let ranges = self.quota.get_mut(usize::from(space));
        ranges.is_some_and(|ranges| remove_range(ranges, offset, bytes))
    }

    /// Opens a region of `limit` units over a memory space, of which the
    /// first `reserved` are never allocated. R10: its units enter it from
    /// the quota, up to `units` of them; the region states how many did.
    pub fn open_region(
        &mut self,
        tier: Tier,
        space: u8,
        unit_bytes: u64,
        reserved: u32,
        units: u32,
        limit: u32,
    ) -> u8 {
        let id = self.regions.len() as u8;
        let mut region = Region::new(id, tier, space, unit_bytes, reserved, 0, limit);
        if let Some(quota) = self.quota.get_mut(usize::from(space)) {
            region.grow(units, |offset, bytes| remove_range(quota, offset, bytes));
        }
        self.regions.push(region);
        id
    }

    /// Closes a region that no session names any more: its copies leave
    /// their entries and its memory returns to the quota.
    pub fn close_region(&mut self, region: u8) {
        let cached: Vec<_> = self
            .cache
            .iter()
            .filter(|(_, entry)| entry.copy_in(region).is_some())
            .map(|(key, _)| *key)
            .collect();
        for key in cached {
            self.evict_in(&key, region);
        }
        let found = &mut self.regions[usize::from(region)];
        let space = usize::from(found.space);
        for (offset, bytes) in found.shrink(u32::MAX) {
            insert_range(&mut self.quota[space], offset, bytes);
        }
    }

    /// Opens a session: an engine epoch, on trial or not, its classes with
    /// the one that holds stored trajectories, and its worker groups. Each
    /// group states the device memory its ranks address (a bit of a block's
    /// validity), the region of its slot rows, and per class the device
    /// regions its owners' blocks are taken from.
    pub fn open(
        &mut self,
        client: ClientId,
        epoch: u64,
        trial: bool,
        classes: (Vec<Class>, u8),
        groups: Vec<(GroupId, u8, u8, Vec<Vec<u8>>)>,
    ) {
        self.classes.insert(client, classes);
        self.sessions.insert(client, epoch);
        self.trial.entry(epoch).or_insert(trial);
        for (group, memory, slots, regions) in groups {
            let found = Group {
                memory,
                slots,
                regions,
                completed: 0,
                deferred: VecDeque::new(),
            };
            self.groups.insert((client, group), found);
        }
    }

    /// Closes a session, or takes note that its engine is gone (A7: its
    /// processes have exited and run no batch). Its copies end as failed;
    /// every stamp of its groups counts as reached; its commits and
    /// transfers that waited are dropped, since a block whose last batch
    /// may not have ended is never shared; its owners are detached. What it
    /// committed stays cached, and what its tasks hold stays held until
    /// they end (R9).
    pub fn close(&mut self, client: ClientId) {
        let mine: Vec<_> = self
            .copies
            .keys()
            .filter(|(queue, _)| queue.0 == client)
            .copied()
            .collect();
        for task in mine {
            let run_by_executor = self.copies[&task]
                .iter()
                .all(|copy| copy.deadline.is_none());
            if run_by_executor {
                self.finished(task.0, task.1, false);
            } else if self.answers.remove(&task).is_some() {
                self.stopped.push(task);
            }
        }

        let groups: Vec<_> = self
            .groups
            .keys()
            .filter(|(of, _)| *of == client)
            .copied()
            .collect();
        for group in groups {
            let Some(found) = self.groups.remove(&group) else {
                continue;
            };
            for (_, work) in found.deferred {
                match work {
                    Deferred::Release { blocks } => {
                        blocks.into_iter().for_each(|block| self.reclaim(block))
                    }
                    Deferred::Commit { block, .. } => self.reclaim(block),
                    Deferred::Transfer { .. } | Deferred::Finish { .. } => {}
                }
            }
        }

        let owners: Vec<_> = self
            .owners
            .keys()
            .filter(|(of, _)| *of == client)
            .copied()
            .collect();
        for owner in owners {
            let Some(found) = self.owners.remove(&owner) else {
                continue;
            };
            self.reclaim(found.slot);
            for table in found.tables {
                table
                    .into_values()
                    .map(Held::into_block)
                    .for_each(|block| self.reclaim(block));
            }
        }

        // The eviction orders of its queue pairs are dropped with it.
        for region in &mut self.regions {
            region.orders.retain(|queue, _| queue.0 != client);
        }
        self.numbers.retain(|queue, _| queue.0 != client);
        self.completions.retain(|(queue, _, _)| queue.0 != client);
        self.sessions.remove(&client);
        self.classes.remove(&client);
        self.sweep();
        self.settle();
    }

    /// The update of an epoch on trial is committed: what the epoch made is
    /// visible to every epoch.
    pub fn commit_update(&mut self, epoch: u64) {
        if let Some(trial) = self.trial.get_mut(&epoch) {
            *trial = false;
        }
        for entry in self.cache.values_mut() {
            let makers = entry
                .device
                .iter_mut()
                .map(|(_, maker)| maker)
                .chain(entry.host.iter_mut().map(|(_, maker)| maker));
            for maker in makers {
                if *maker == Some(epoch) {
                    *maker = None;
                }
            }
        }
    }

    /// The update of an epoch on trial is rolled back, once its session is
    /// closed: what the epoch made stays visible to it alone and is evicted.
    pub fn roll_back(&mut self, epoch: u64) {
        if self.trial.get(&epoch) == Some(&true) && !self.sessions.values().any(|of| *of == epoch) {
            self.trial.remove(&epoch);
            self.sweep();
        }
    }

    /// Evicts the copies an epoch made whose update was rolled back, as far
    /// as nothing else holds them: no epoch can use them.
    pub(super) fn sweep(&mut self) {
        let rolled =
            |maker: &Option<u64>| maker.is_some_and(|epoch| !self.trial.contains_key(&epoch));
        let mut copies = Vec::new();
        for (key, entry) in &self.cache {
            let blocks = entry.device.iter().chain(&entry.host);
            copies.extend(
                blocks
                    .filter(|(_, maker)| rolled(maker))
                    .map(|(block, _)| (*key, block.region())),
            );
        }
        for (key, region) in copies {
            self.evict_in(&key, region);
        }
    }
}
