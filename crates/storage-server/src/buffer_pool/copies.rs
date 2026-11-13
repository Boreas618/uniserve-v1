//! The copies an executor runs: `Clone`, `Load`, `Store` to the host, `Save`
//! and `Restore`, and the `Finish` that ends each.
//!
//! The handler allocates the destination and answers `Copy` with the runs to
//! read and to write. The copy holds what it reads and what it writes until
//! its `Finish` takes effect, which is when the `completed` of the copy's
//! group has reached the stamp of the `Finish`: the executor has then ended
//! the batch the copy ran with.

use std::rc::Rc;

use storage::protocol::Key;
use storage::runtime::{Cause, Completion, GroupId, OwnerId, RequestId, Stamps, Tier};

use super::{
    Block, BufferPool, ClientId, Copy, Deferred, EVERYWHERE, Entry, Held, QueueId, SLOT, Taken,
    Target,
};

impl BufferPool {
    /// Records the copies the request `task` started. Each holds what it
    /// reads and what it writes until it ends, and what it writes for a
    /// target that exists, in place of no content others read, is private
    /// by it.
    pub(super) fn hold_copies(&mut self, task: (QueueId, RequestId), copies: Vec<Copy>) {
        for copy in &copies {
            copy.reads.iter().for_each(|block| self.touch(block));
            if self.live(task.0.0, copy) {
                copy.write.own();
            }
            self.touch(&copy.write);
        }
        self.copies.insert(task, copies);
    }

    /// A copy of the session `client` ended: it lets go of what it read and
    /// what it wrote.
    pub(super) fn drop_copy(&mut self, client: ClientId, copy: Copy) {
        if self.live(client, &copy) {
            copy.write.disown();
        }
        for block in copy.reads {
            self.reclaim(block);
        }
        self.reclaim(copy.write);
    }

    /// Whether a copy reads or writes for the owner: one that writes for a
    /// position of its tables or for its slot, or reads its slot.
    pub(super) fn copied(&self, client: ClientId, owner: OwnerId) -> bool {
        let Some(found) = self.owners.get(&(client, owner)) else {
            return false;
        };
        self.copies_of(client).any(|copy| {
            let named = matches!(copy.to, Target::Owner { owner: target, .. } | Target::Slot { owner: target }
                if target == owner);
            named || copy.reads.iter().any(|block| Rc::ptr_eq(block, &found.slot))
        })
    }

    /// The class of a session that holds stored trajectories (R1).
    fn trajectory(&self, client: ClientId) -> Result<u8, Cause> {
        self.classes
            .get(&client)
            .map(|(_, trajectory)| *trajectory)
            .ok_or(Cause::Class)
    }

    /// The units of a host copy of a block of a class: the configured size,
    /// or the units that hold the block's bytes.
    fn host_units(&self, client: ClientId, class: u8, block: &Block) -> Result<(u8, u32), Cause> {
        let (region, units) = self.class(client, class)?.host.ok_or(Cause::Class)?;
        if units > 0 {
            return Ok((region, units));
        }
        let bytes = u64::from(block.units()) * self.regions[usize::from(block.region())].unit_bytes;
        let size = bytes.div_ceil(self.regions[usize::from(region)].unit_bytes);
        Ok((region, u32::try_from(size).map_err(|_| Cause::Size)?))
    }

    /// R5: a clone is a new private block that the executor fills from the
    /// cached one; the cached block is never handed out as private.
    pub(super) fn clone_block(
        &mut self,
        queue: QueueId,
        id: RequestId,
        owner: OwnerId,
        class: u8,
        at: u32,
        key: Key,
    ) -> Taken {
        let client = queue.0;
        let group = self.settled(client, owner)?.group;
        if self.class(client, class)?.state_in_slot {
            return Err(Cause::Class);
        }
        if !self.free_positions(client, owner, class, at, 1) {
            return Err(Cause::Position);
        }
        // The copy holds the block it reads, so no eviction for its
        // destination takes it.
        let source = Rc::clone(self.usable(client, group, class, &key).ok_or(Cause::Key)?);
        let region = source.region();
        let size = source.units();

        let Some(evicted) = self.room(queue, region, 1, size) else {
            return Ok(Some(Completion::Full { region }));
        };
        let mut blocks = self.obtain(region, class, 1, size, &evicted, 0);
        let write = blocks.pop().ok_or(Cause::Size)?;

        let (from, to) = (source.stated().collect(), write.stated().collect());
        let target = Target::Owner {
            owner,
            class,
            at,
            key: None,
        };
        let memory = self.memory(client, group);
        let copy = Copy {
            memory,
            reads: vec![source],
            write,
            to: target,
            fill: false,
            deadline: None,
        };
        self.hold_copies((queue, id), vec![copy]);
        Ok(Some(Completion::Copy {
            group,
            keys: vec![key],
            from,
            to,
            evicted,
        }))
    }

    /// Whether something fills `memory` of the block: a load, or a transfer.
    pub(super) fn filling(&self, block: &Rc<Block>, memory: u8) -> bool {
        let copies = self.copies.values().flatten();
        copies
            .into_iter()
            .any(|copy| copy.fill && copy.memory == memory && Rc::ptr_eq(&copy.write, block))
    }

    pub(super) fn load(
        &mut self,
        queue: QueueId,
        id: RequestId,
        owner: OwnerId,
        class: u8,
        at: u32,
        keys: &[Key],
    ) -> Taken {
        let client = queue.0;
        let group = self.settled(client, owner)?.group;
        self.class(client, class)?;
        let region = self.region(client, group, class)?;
        let memory = self.memory(client, group);

        // The leading keys that have a host copy the epoch may use and that
        // the owner cannot pin instead. For each, the host block, and the
        // device block to fill where the key has one that is valid in other
        // memories only and that a copy of this epoch may join (R8).
        let mut sources = Vec::new();
        for key in keys {
            let Some(entry) = self.cache.get(key).filter(|entry| entry.class == class) else {
                break;
            };
            let Some((host, _)) = entry
                .host
                .as_ref()
                .filter(|(_, maker)| self.visible(*maker, client))
            else {
                break;
            };
            let device = entry
                .device
                .iter()
                .find(|(block, _)| block.region() == region);
            let fills = match device {
                Some((block, maker))
                    if block.valid.get() & memory == 0 && *maker == self.made_by(client) =>
                {
                    // One memory of a block is filled by one copy at a time.
                    if self.filling(block, memory) {
                        break;
                    }
                    Some(Rc::clone(block))
                }
                Some((block, maker))
                    if block.valid.get() & memory != 0 && self.visible(*maker, client) =>
                {
                    break;
                }
                _ => None,
            };
            sources.push((*key, Rc::clone(host), fills));
        }
        if sources.is_empty() {
            return Ok(Some(Completion::Loaded));
        }
        if !self.free_positions(client, owner, class, at, sources.len() as u32) {
            return Err(Cause::Position);
        }

        // New blocks for the keys that fill none, all or none.
        let new = sources
            .iter()
            .filter(|(_, _, fills)| fills.is_none())
            .count() as u32;
        let size = self.class(client, class)?.units;
        let size = if size > 0 {
            size
        } else {
            // A sized class: the device block holds the host block's bytes.
            let host = &sources[0].1;
            let bytes =
                u64::from(host.units()) * self.regions[usize::from(host.region())].unit_bytes;
            u32::try_from(bytes.div_ceil(self.regions[usize::from(region)].unit_bytes))
                .map_err(|_| Cause::Size)?
        };
        let Some(evicted) = self.room(queue, region, new, size) else {
            return Ok(Some(Completion::Full { region }));
        };
        let mut blocks = self
            .obtain(region, class, new, size, &evicted, 0)
            .into_iter();

        let (mut from, mut to, mut copies) = (Vec::new(), Vec::new(), Vec::new());
        let stated: Vec<Key> = sources.iter().map(|(key, _, _)| *key).collect();
        for (position, (key, host, fills)) in (at..).zip(sources) {
            let fill = fills.is_some();
            let Some(write) = fills.or_else(|| blocks.next()) else {
                break;
            };
            from.extend(host.stated());
            to.extend(write.stated());
            let target = Target::Owner {
                owner,
                class,
                at: position,
                key: Some(key),
            };
            copies.push(Copy {
                memory,
                reads: vec![host],
                write,
                to: target,
                fill,
                deadline: None,
            });
        }
        self.hold_copies((queue, id), copies);
        Ok(Some(Completion::Copy {
            group,
            keys: stated,
            from,
            to,
            evicted,
        }))
    }

    pub(super) fn store(
        &mut self,
        queue: QueueId,
        id: RequestId,
        group: GroupId,
        keys: &[Key],
        tier: Tier,
    ) -> Taken {
        let client = queue.0;
        if !self.groups.contains_key(&(client, group)) {
            return Err(Cause::Group);
        }
        match tier {
            Tier::Host => {}
            Tier::Remote => return self.put(queue, id, keys),
            Tier::Device => return Err(Cause::Class),
        }
        let memory = self.memory(client, group);

        // Each named key that has a device copy valid in the group and
        // visible to its epoch, and no host copy.
        let mut sources = Vec::new();
        for key in keys {
            let Some(entry) = self.cache.get(key).filter(|entry| entry.host.is_none()) else {
                continue;
            };
            let device = entry.device.iter().find(|(block, maker)| {
                block.valid.get() & memory != 0 && self.visible(*maker, client)
            });
            let Some((block, _)) = device else { continue };
            let Ok((region, units)) = self.host_units(client, entry.class, block) else {
                continue;
            };
            sources.push((*key, Rc::clone(block), region, units));
        }
        let Some(&(_, _, region, size)) = sources.first() else {
            return Ok(Some(Completion::Stored { keys: Vec::new() }));
        };
        // One request stores blocks of one host region and size.
        sources.retain(|(_, _, other, units)| *other == region && *units == size);

        let Some(evicted) = self.room(queue, region, sources.len() as u32, size) else {
            return Ok(Some(Completion::Full { region }));
        };
        let class = sources[0].1.class;
        let blocks = self.obtain(
            region,
            class,
            sources.len() as u32,
            size,
            &evicted,
            EVERYWHERE,
        );

        let (mut from, mut to, mut copies) = (Vec::new(), Vec::new(), Vec::new());
        let stated: Vec<Key> = sources.iter().map(|(key, _, _, _)| *key).collect();
        for ((key, device, _, _), write) in sources.into_iter().zip(blocks) {
            from.extend(device.stated());
            to.extend(write.stated());
            let target = Target::Entry { key, remote: false };
            copies.push(Copy {
                memory: 0,
                reads: vec![device],
                write,
                to: target,
                fill: false,
                deadline: None,
            });
        }
        self.hold_copies((queue, id), copies);
        Ok(Some(Completion::Copy {
            group,
            keys: stated,
            from,
            to,
            evicted,
        }))
    }

    pub(super) fn save(
        &mut self,
        queue: QueueId,
        id: RequestId,
        owner: OwnerId,
        key: Key,
        units: u32,
    ) -> Taken {
        let client = queue.0;
        let found = self.settled(client, owner)?;
        let (group, slot) = (found.group, Rc::clone(&found.slot));
        // R1: one save or restore of an owner's slot at a time.
        if self.copied(client, owner) {
            return Err(Cause::Copy);
        }
        let trajectory = self.trajectory(client)?;
        if self
            .cache
            .get(&key)
            .is_some_and(|entry| entry.host.is_some() || entry.class != trajectory)
        {
            return Err(Cause::Key);
        }
        let (region, _) = self.class(client, trajectory)?.host.ok_or(Cause::Class)?;
        if units == 0 {
            return Err(Cause::Size);
        }

        let Some(evicted) = self.room(queue, region, 1, units) else {
            return Ok(Some(Completion::Full { region }));
        };
        let mut blocks = self.obtain(region, trajectory, 1, units, &evicted, EVERYWHERE);
        let write = blocks.pop().ok_or(Cause::Size)?;

        let (from, to) = (slot.stated().collect(), write.stated().collect());
        let target = Target::Entry { key, remote: false };
        let copy = Copy {
            memory: 0,
            reads: vec![slot],
            write,
            to: target,
            fill: false,
            deadline: None,
        };
        self.hold_copies((queue, id), vec![copy]);
        Ok(Some(Completion::Copy {
            group,
            keys: vec![key],
            from,
            to,
            evicted,
        }))
    }

    pub(super) fn restore(
        &mut self,
        queue: QueueId,
        id: RequestId,
        owner: OwnerId,
        key: Key,
    ) -> Taken {
        let client = queue.0;
        let found = self.settled(client, owner)?;
        let (group, slot) = (found.group, Rc::clone(&found.slot));
        if self.copied(client, owner) {
            return Err(Cause::Copy);
        }
        let entry = self
            .cache
            .get(&key)
            .filter(|entry| Ok(entry.class) == self.trajectory(client))
            .ok_or(Cause::Key)?;
        let (host, _) = entry
            .host
            .as_ref()
            .filter(|(_, maker)| self.visible(*maker, client))
            .ok_or(Cause::Key)?;
        let host = Rc::clone(host);

        let (from, to) = (host.stated().collect(), slot.stated().collect());
        let target = Target::Slot { owner };
        let copy = Copy {
            memory: 0,
            reads: vec![host],
            write: slot,
            to: target,
            fill: false,
            deadline: None,
        };
        self.hold_copies((queue, id), vec![copy]);
        Ok(Some(Completion::Copy {
            group,
            keys: vec![key],
            from,
            to,
            evicted: Vec::new(),
        }))
    }

    /// One `Finish` ends a copy: R1 refuses one that names no copy an
    /// executor of the session runs, or a copy whose `Finish` waits.
    pub(super) fn finish(
        &mut self,
        queue: QueueId,
        id: RequestId,
        copy: RequestId,
        ok: bool,
        after: Stamps,
    ) -> Taken {
        let run_by_executor = self
            .copies
            .get(&(queue, copy))
            .is_some_and(|copies| copies.iter().all(|copy| copy.deadline.is_none()));
        let waits = self.groups.values().flat_map(|group| &group.deferred).any(|(_, work)| {
            matches!(work, Deferred::Finish { queue: of, copy: other, .. } if *of == queue && *other == copy)
        });
        if !run_by_executor || waits {
            return Err(Cause::Copy);
        }
        self.stamped(queue.0, &after)?;

        self.defer(
            queue.0,
            after,
            Deferred::Finish {
                queue,
                request: id,
                copy,
                ok,
                rest: Vec::new(),
            },
        );
        Ok(None)
    }

    /// The end of a copy takes effect (R3: its executor has ended the batch
    /// it ran with). When it failed the destinations retire, nothing is
    /// cached and the positions are free.
    pub(super) fn finished(&mut self, queue: QueueId, copy: RequestId, ok: bool) -> Completion {
        let client = queue.0;
        let copies = self.copies.remove(&(queue, copy)).unwrap_or_default();
        let mut completion = Completion::Done;

        for copy in copies {
            let memory = copy.memory;
            if ok {
                completion = match copy.to {
                    Target::Owner {
                        owner,
                        class,
                        at,
                        key,
                    } => {
                        let (held, completion) = match key {
                            // A clone: a private block with the cached content.
                            None => {
                                copy.write.valid.set(memory);
                                (Held::Private(Rc::clone(&copy.write)), Completion::Cloned)
                            }
                            Some(key) => self.loaded(
                                client,
                                &copy,
                                key,
                                memory,
                                completion == Completion::Duplicate,
                            ),
                        };
                        if let Some(found) = self.owners.get_mut(&(client, owner)) {
                            if let Held::Private(block) = &held {
                                block.own();
                            }
                            found.tables[usize::from(class)].insert(at, held);
                        }
                        completion
                    }
                    Target::Entry { key, .. } => self.stored(client, &copy, key),
                    // The slot holds the stored trajectory.
                    Target::Slot { .. } => Completion::Restored,
                    Target::Group { .. } => Completion::Done,
                };
            }
            self.drop_copy(client, copy);
        }
        completion
    }

    /// A load's copy ended well: what the owner's table gets, and the answer
    /// of its `Finish`, which is `Duplicate` when a key of the request kept
    /// another block.
    fn loaded(
        &mut self,
        client: ClientId,
        copy: &Copy,
        key: Key,
        memory: u8,
        duplicate: bool,
    ) -> (Held, Completion) {
        let answer = if duplicate {
            Completion::Duplicate
        } else {
            Completion::Loaded
        };
        if copy.fill {
            // The group's memory of the shared block holds its content.
            copy.write.valid.set(copy.write.valid.get() | memory);
            return (Held::Shared(Rc::clone(&copy.write)), answer);
        }

        copy.write.valid.set(memory);
        let region = copy.write.region();
        let maker = self.made_by(client);
        let class = copy.write.class;
        let entry = self.cache.entry(key).or_insert(Entry {
            class,
            device: Vec::new(),
            host: None,
            tokens: 0,
            remote: false,
        });
        if entry.copy_in(region).is_some() {
            // R4: the key gained a device block meanwhile; this block stays
            // private to the owner, which only reads it.
            return (Held::Private(Rc::clone(&copy.write)), Completion::Duplicate);
        }
        // The block is shared on the device and pinned for the owner.
        entry.device.push((Rc::clone(&copy.write), maker));
        copy.write.cached.set(Some(key));
        (Held::Shared(Rc::clone(&copy.write)), answer)
    }

    /// A store's or a save's copy ended well: the host block is the key's
    /// host copy, unless the key gained one meanwhile (R4), in which case
    /// the block just written is released.
    fn stored(&mut self, client: ClientId, copy: &Copy, key: Key) -> Completion {
        let class = copy.write.class;
        // A save reads its owner's slot.
        let answer = if copy.reads.first().is_some_and(|block| block.class == SLOT) {
            Completion::Saved
        } else {
            Completion::Stored { keys: Vec::new() }
        };
        let maker = self.made_by(client);
        let entry = self.cache.entry(key).or_insert(Entry {
            class,
            device: Vec::new(),
            host: None,
            tokens: 0,
            remote: false,
        });
        if entry.host.is_none() {
            entry.host = Some((Rc::clone(&copy.write), maker));
            copy.write.cached.set(Some(key));
        }
        answer
    }
}
