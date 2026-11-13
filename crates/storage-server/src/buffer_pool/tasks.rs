//! Tasks: the copies the Transfer Server and the connector run. A transfer
//! moves an owner to another group, a crossing (`Replicate`) brings a block
//! into the memories another group reads it in, a prefetch brings keys into
//! the host tier, a put writes host copies to the remote store.
//!
//! A task is the copies of one request. They hold their blocks until the
//! task has ended (R9), whatever became of the request: answered at its
//! deadline, its owner detached, its session closed. The buffer pool lists
//! the tasks to start and to stop; whoever runs them reports each key done
//! and the task's end.

use std::rc::Rc;

use storage::protocol::Key;
use storage::runtime::{Cause, Completion, GroupId, OwnerId, RequestId, Stamps};

use super::region::Run;

use super::{
    Block, BufferPool, ClientId, Copy, Deferred, EVERYWHERE, Entry, Held, QueueId, SLOT, Taken,
    Target,
};

/// The deadline of a task that was told to stop: it has no further one.
const STOPPING: u64 = u64::MAX;

/// Runs of units as a completion states them.
pub(super) type Runs = Vec<storage::runtime::Run>;

impl BufferPool {
    /// The tasks to start since the last call.
    pub fn started(&mut self) -> Vec<(QueueId, RequestId)> {
        std::mem::take(&mut self.started)
    }

    /// The tasks to stop since the last call: past their deadline, or of a
    /// session that closed.
    pub fn stopped(&mut self) -> Vec<(QueueId, RequestId)> {
        std::mem::take(&mut self.stopped)
    }

    /// What a task copies: per copy, the key it is for if any, the runs it
    /// reads and writes, the memories that hold the content it reads and
    /// the memories it writes, as bits of a block's validity. A transfer's
    /// copies have no key and read and write the same runs, from a memory
    /// of the owner's group into the target group's, and its last copy
    /// takes the slot row in the target group. A crossing's copies have no
    /// key; each reads the block's run in the region the request named,
    /// from the memories of that run that hold the block, and writes the
    /// block's runs that lie in one memory of the reading group. A
    /// prefetch's copies read nothing and write the host parts of the group
    /// that asked; a put's read and write the host block it puts.
    pub fn work(&self, task: (QueueId, RequestId)) -> Vec<(Option<Key>, Runs, Runs, u64, u64)> {
        let copies = self.copies.get(&task).map_or(&[][..], Vec::as_slice);
        let stated = |copy: &Copy| {
            let key = match copy.to {
                Target::Entry { key, .. } => Some(key),
                _ => None,
            };
            // A transfer's last copy also holds the blocks it does not copy.
            let read = copy.reads.first();
            if let (Target::Memory { from, .. }, Some(block)) = (copy.to, read) {
                let runs = block.runs.borrow();
                let reads = runs.iter().filter(|run| run.region() == from);
                // A run of a region each memory holds lies in every memory
                // that holds the block; a run of a region whose units exist
                // once lies in that region's memory.
                let source = self.lies_in(from, u64::MAX) & block.valid.get();
                let written =
                    |run: &&Run| self.lies_in(run.region(), copy.memory) & copy.memory != 0;
                let writes = runs.iter().filter(written).map(Run::stated).collect();
                return (
                    key,
                    reads.map(Run::stated).collect(),
                    writes,
                    source,
                    copy.memory,
                );
            }
            let reads = read.map_or_else(Vec::new, |block| block.stated());
            let memories = read.map_or(0, |block| block.valid.get());
            (key, reads, copy.write.stated(), memories, copy.memory)
        };
        copies.iter().map(stated).collect()
    }

    /// Whether a task is outstanding: started and not ended, or not yet
    /// handed to whoever runs it. A restart in place waits until none is.
    pub fn tasks(&self) -> bool {
        let task = |copies: &Vec<Copy>| copies.iter().any(|copy| copy.deadline.is_some());
        !self.started.is_empty() || !self.stopped.is_empty() || self.copies.values().any(task)
    }

    /// A transfer waits for its stamps: until then the source group's
    /// batches may still write the owner's blocks.
    pub(super) fn transfer(
        &mut self,
        queue: QueueId,
        id: RequestId,
        owner: OwnerId,
        to: GroupId,
        after: Stamps,
    ) -> Taken {
        let client = queue.0;
        let from = self.settled(client, owner)?.group;
        if to == from || !self.groups.contains_key(&(client, to)) {
            return Err(Cause::Group);
        }
        // R1: an owner that a copy writes for, or whose crossing waits or
        // runs, is not transferred.
        if self.copied(client, owner) || self.replicating(client, owner, None) {
            return Err(Cause::Copy);
        }
        self.stamped(client, &after)?;

        if let Some(found) = self.owners.get_mut(&(client, owner)) {
            found.moving = true;
        }
        self.defer(
            client,
            after,
            Deferred::Transfer {
                queue,
                request: id,
                owner,
                to,
                rest: Vec::new(),
            },
        );
        Ok(None)
    }

    /// A transfer comes due: a task starts that copies into the target
    /// group's memory the owner's blocks that are not valid there. It holds
    /// every block of the owner and a slot row of the target group. Returns
    /// false when it must wait: a load or another owner's transfer fills the
    /// target group's memory of a block the owner holds.
    pub(super) fn start(
        &mut self,
        queue: QueueId,
        request: RequestId,
        owner: OwnerId,
        to: GroupId,
    ) -> bool {
        let client = queue.0;
        let target = self.memory(client, to);
        let (Some(found), Some(group)) = (
            self.owners.get(&(client, owner)),
            self.groups.get(&(client, to)),
        ) else {
            // The owner was detached meanwhile: nothing is held.
            self.complete(queue, request, Completion::Failed);
            return true;
        };
        let held = || {
            found
                .tables
                .iter()
                .flat_map(|table| table.values().map(Held::block))
        };
        if held().any(|block| block.valid.get() & target == 0 && self.filling(block, target)) {
            return false;
        }

        // Without a free row in the target group the transfer fails and nothing changes.
        let slots = group.slots;
        let (old, blocks): (Rc<Block>, Vec<Rc<Block>>) =
            (Rc::clone(&found.slot), held().cloned().collect());
        let Some(row) = self.regions[usize::from(slots)].take_row() else {
            if let Some(found) = self.owners.get_mut(&(client, owner)) {
                found.moving = false;
            }
            self.complete(queue, request, Completion::Failed);
            return true;
        };

        let deadline = Some(self.now.saturating_add(self.limits.0));
        let mut copies: Vec<Copy> = Vec::new();
        let mut kept = vec![old];
        for block in blocks {
            let copied = block.valid.get() & target == 0
                && !copies.iter().any(|copy| Rc::ptr_eq(&copy.write, &block));
            if copied {
                let reads = vec![Rc::clone(&block)];
                let to = Target::Group { owner, to };
                copies.push(Copy {
                    memory: target,
                    reads,
                    write: block,
                    to,
                    fill: true,
                    deadline,
                });
            } else {
                kept.push(block);
            }
        }
        // The slot's row: it reads the row of the source group, and holds
        // the blocks the task does not copy.
        let slot = Rc::new(Block::new(SLOT, vec![row], EVERYWHERE, 0));
        let to = Target::Group { owner, to };
        copies.push(Copy {
            memory: 0,
            reads: kept,
            write: slot,
            to,
            fill: false,
            deadline,
        });

        self.hold_copies((queue, request), copies);
        self.answers.insert((queue, request), Vec::new());
        self.started.push((queue, request));
        true
    }

    /// A transfer's task ended. When it copied everything and the owner is
    /// still there, the owner is attached to the target group with its new
    /// slot row; otherwise the owner stays where it was.
    fn transferred(
        &mut self,
        client: ClientId,
        owner: OwnerId,
        to: GroupId,
        copies: &[Copy],
        ok: bool,
    ) -> Completion {
        let target = self.memory(client, to);
        let committing: Vec<*const Block> = self.commits().map(Rc::as_ptr).collect();
        let Some(found) = self.owners.get_mut(&(client, owner)) else {
            return Completion::Failed;
        };
        found.moving = false;
        let Some(slot) = copies
            .iter()
            .find(|copy| !copy.fill)
            .filter(|_| ok)
            .map(|copy| Rc::clone(&copy.write))
        else {
            return Completion::Failed;
        };

        // The target group's memory holds what the task copied. A private
        // block that no commit froze is written there from now on, so the
        // source group's memory of it is stale.
        let mut runs = Vec::new();
        for (class, table) in found.tables.iter().enumerate() {
            for (position, held) in table {
                let block = held.block();
                let frozen =
                    matches!(held, Held::Shared(_)) || committing.contains(&Rc::as_ptr(block));
                if copies
                    .iter()
                    .any(|copy| copy.fill && Rc::ptr_eq(&copy.write, block))
                {
                    block.valid.set(if frozen {
                        block.valid.get() | target
                    } else {
                        target
                    });
                }
                runs.extend(
                    block
                        .stated()
                        .into_iter()
                        .map(|run| (class as u8, *position, run, !frozen)),
                );
            }
        }
        found.group = to;
        let row = slot.runs.borrow().first().map_or(0, Run::first);
        // The new row is the owner's slot, and the old one no longer.
        slot.own();
        let old = std::mem::replace(&mut found.slot, slot);
        old.disown();
        self.reclaim(old);
        Completion::Transferred { slot: row, runs }
    }

    /// Whether a crossing of the owner waits for its stamps or for a copy
    /// to end, or runs: any of its crossings, or with `entry` the one of its
    /// position of a class into a group.
    pub(super) fn replicating(
        &self,
        client: ClientId,
        owner: OwnerId,
        entry: Option<(u8, u32, GroupId)>,
    ) -> bool {
        let named = |class: u8, at: u32, group: GroupId| {
            entry.is_none_or(|entry| entry == (class, at, group))
        };
        let deferred = self
            .groups
            .iter()
            .filter(|((of, _), _)| *of == client)
            .flat_map(|(_, group)| &group.deferred);
        let waits = deferred.into_iter().any(|(_, work)| {
            matches!(work, Deferred::Replicate { owner: other, class, at, to, .. }
                if *other == owner && named(*class, *at, *to))
        });
        let runs = self.copies_of(client).any(|copy| {
            matches!(copy.to, Target::Memory { owner: other, class, at, group, .. }
                if other == owner && named(class, at, group))
        });
        waits || runs
    }

    /// The memories a group reads a class in: those of its regions and of
    /// its receiving regions of the class.
    fn reading(&self, client: ClientId, group: GroupId, class: u8) -> u64 {
        let Some(found) = self.groups.get(&(client, group)) else {
            return 0;
        };
        let own = 1 << found.memory;
        let lists = [&found.regions, &found.receives];
        let regions = lists
            .into_iter()
            .filter_map(|lists| lists.get(usize::from(class)))
            .flatten();
        regions.fold(0, |all, region| all | self.lies_in(*region, own))
    }

    /// A block's runs that a group reads the class in: those in its regions
    /// and receiving regions of the class, in the order the block holds them.
    fn read_runs(&self, client: ClientId, group: GroupId, class: u8, block: &Block) -> Runs {
        let Some(found) = self.groups.get(&(client, group)) else {
            return Vec::new();
        };
        let named = |lists: &Vec<Vec<u8>>, region: u8| {
            lists
                .get(usize::from(class))
                .is_some_and(|list| list.contains(&region))
        };
        let runs = block.runs.borrow();
        let read = runs.iter().filter(|run| {
            named(&found.regions, run.region()) || named(&found.receives, run.region())
        });
        read.map(Run::stated).collect()
    }

    /// `Replicate`: the block at an owner's position is to be valid in the
    /// memories `group` reads its class in. R1: the owner exists and does not
    /// move, the position holds an entry and no crossing of it into the group
    /// waits or runs, `from` is a region of one of its runs, and the group
    /// reads the class, with a run of the block or a receiving region in each
    /// memory it reads it in that does not hold the block. R11: the run in
    /// `from` lies in a memory that holds the block, which for a private
    /// block allocated over regions of several memories is the memory of
    /// `from` itself (A9). R8: an epoch on trial replicates no shared block
    /// another epoch may use. R5: a memory that holds the block is no target;
    /// when none is left the request is answered at once. Otherwise a run is taken in the
    /// group's receiving region of each target memory in which the block has
    /// none, all or none (`Full` changes nothing), and the crossing waits for
    /// its stamps holding the block (R3), as a `Release` of the entry would.
    pub(super) fn replicate(
        &mut self,
        queue: QueueId,
        id: RequestId,
        (owner, class, at): (OwnerId, u8, u32),
        from: u8,
        group: GroupId,
        after: Stamps,
    ) -> Taken {
        let client = queue.0;
        let found = self.settled(client, owner)?;
        let writer = found.group;
        self.class(client, class)?;
        self.stamped(client, &after)?;
        let held = found.tables[usize::from(class)]
            .get(&at)
            .ok_or(Cause::Position)?;
        let (block, private) = (Rc::clone(held.block()), matches!(held, Held::Private(_)));
        let reader = self.groups.get(&(client, group)).ok_or(Cause::Group)?;
        // A position whose crossing into the group waits or runs is not
        // replicated into it again, as a waiting commit's is not committed again.
        if self.replicating(client, owner, Some((class, at, group))) {
            return Err(Cause::Position);
        }
        let (regions, receives) = (
            reader
                .regions
                .get(usize::from(class))
                .cloned()
                .unwrap_or_default(),
            reader
                .receives
                .get(usize::from(class))
                .cloned()
                .unwrap_or_default(),
        );
        let reader_memory = 1 << reader.memory;
        if !block.regions().contains(&from) || regions.is_empty() && receives.is_empty() {
            return Err(Cause::Class);
        }

        // R8: a shared block is the entry's device copy, which an epoch on
        // trial fills in no memory unless that epoch made it.
        if !private {
            let entry = block.cached.get().and_then(|key| self.cache.get(&key));
            let maker = entry
                .and_then(|entry| {
                    entry
                        .device
                        .iter()
                        .find(|(copy, _)| Rc::ptr_eq(copy, &block))
                })
                .map(|(_, maker)| *maker);
            if maker.is_some_and(|maker| maker != self.made_by(client)) {
                return Err(Cause::Key);
            }
        }

        // A9: a private block allocated over regions of several memories
        // holds its content in the memory of the run its writer wrote, from
        // now on: a run in one of the writer's regions of the class, never
        // one a crossing took in a receiving region.
        let own = self.memory(client, writer);
        let written = self.groups.get(&(client, writer));
        let written = written.and_then(|found| found.regions.get(usize::from(class)));
        let written = written.cloned().unwrap_or_default();
        let mut valid = block.valid.get();
        if private && written.contains(&from) && self.spread(&written, own).count_ones() > 1 {
            valid |= self.lies_in(from, own);
        }
        // R11: the run read lies in a memory that holds the block.
        if self.lies_in(from, own) & valid == 0 {
            return Err(Cause::Class);
        }

        // R5: the memories the group reads the class in that do not hold the
        // block; in each, the block's run there or a run to take in the
        // receiving region there, of the bytes of the run it is read from.
        let targets = self.reading(client, group, class) & !valid;
        let bytes = {
            let runs = block.runs.borrow();
            let source = runs.iter().find(|run| run.region() == from);
            let units = source.map_or(0, Run::count);
            u64::from(units) * self.regions[usize::from(from)].unit_bytes
        };
        let mut taken: Vec<(u8, u32, Vec<Key>)> = Vec::new();
        for memory in (0..64)
            .map(|bit| 1u64 << bit)
            .filter(|bit| targets & bit != 0)
        {
            let has_run = block
                .regions()
                .iter()
                .any(|region| self.lies_in(*region, memory) == memory);
            if has_run {
                continue;
            }
            let Some(region) = receives
                .iter()
                .copied()
                .find(|region| self.lies_in(*region, reader_memory) == memory)
            else {
                return Err(Cause::Class);
            };
            let unit_bytes = self.regions[usize::from(region)].unit_bytes;
            let size = u32::try_from(bytes.div_ceil(unit_bytes)).map_err(|_| Cause::Size)?;
            // A run is taken as an allocation's is, evicting in the queue
            // pair's order: the units its `available` word counts.
            let Some(evicted) = self.room(queue, region, 1, size) else {
                return Ok(Some(Completion::Full { region }));
            };
            taken.push((region, size, evicted));
        }

        block.valid.set(valid);
        if !taken.is_empty() {
            // The block's units are counted again with its new runs.
            self.uncount(&block);
            for (region, size, evicted) in taken {
                for key in &evicted {
                    self.evict_in(key, region);
                }
                let run = self.regions[usize::from(region)].take(size);
                block.runs.borrow_mut().extend(run);
            }
            self.touch(&block);
        }
        if targets == 0 {
            let runs = self.read_runs(client, group, class, &block);
            return Ok(Some(Completion::Replicated { runs }));
        }

        // The crossing holds the block while it waits, as a release does.
        self.touch(&block);
        let crossing = Deferred::Replicate {
            queue,
            request: id,
            owner,
            class,
            at,
            block,
            from,
            to: group,
            rest: Vec::new(),
        };
        self.defer(client, after, crossing);
        Ok(None)
    }

    /// A crossing comes due: every stamp it waited for is reached, so no
    /// batch in flight writes its block. It waits while a load, a transfer
    /// or another crossing fills a memory the group reads the class in.
    /// Then, when the owner no longer holds the block at its position, it
    /// lets go of the block and is answered `Failed`; when every memory the
    /// group reads the class in holds the block, `Replicated` (R5);
    /// otherwise a task of the Transfer Server starts that copies the block
    /// into each memory that lacks it, holding the block until it ends (R9).
    /// Returns the crossing when it must wait.
    pub(super) fn cross(&mut self, work: Deferred) -> Option<Deferred> {
        let Deferred::Replicate {
            queue,
            request,
            owner,
            class,
            at,
            block,
            from,
            to,
            ..
        } = work
        else {
            return Some(work);
        };
        let client = queue.0;
        let reading = self.reading(client, to, class);
        let filled = (0..64)
            .map(|bit| 1u64 << bit)
            .any(|memory| reading & memory != 0 && self.filling(&block, memory));
        if filled {
            return Some(Deferred::Replicate {
                queue,
                request,
                owner,
                class,
                at,
                block,
                from,
                to,
                rest: Vec::new(),
            });
        }

        let held = self
            .owners
            .get(&(client, owner))
            .and_then(|found| found.tables[usize::from(class)].get(&at))
            .is_some_and(|held| Rc::ptr_eq(held.block(), &block));
        let targets = reading & !block.valid.get();
        if !held || targets == 0 {
            let completion = if held {
                Completion::Replicated {
                    runs: self.read_runs(client, to, class, &block),
                }
            } else {
                Completion::Failed
            };
            self.reclaim(block);
            self.complete(queue, request, completion);
            return None;
        }

        let deadline = Some(self.now.saturating_add(self.limits.0));
        let target = Target::Memory {
            owner,
            class,
            at,
            group: to,
            from,
        };
        let copies = (0..64)
            .map(|bit| 1u64 << bit)
            .filter(|memory| targets & memory != 0)
            .map(|memory| Copy {
                memory,
                reads: vec![Rc::clone(&block)],
                write: Rc::clone(&block),
                to: target,
                fill: true,
                deadline,
            })
            .collect();
        self.hold_copies((queue, request), copies);
        // The task's copies hold the block from now on.
        self.reclaim(block);
        self.answers.insert((queue, request), Vec::new());
        self.started.push((queue, request));
        None
    }

    /// A crossing's task ended (A5). When it copied everything, every memory
    /// it wrote holds the block, whatever became of its owner; the answer is
    /// the block's runs the group reads the class in while the owner still
    /// holds the block at its position, and `Failed` otherwise.
    fn replicated(
        &mut self,
        client: ClientId,
        copies: &[Copy],
        (owner, class, at): (OwnerId, u8, u32),
        group: GroupId,
        ok: bool,
    ) -> Completion {
        let Some(block) = copies.first().map(|copy| Rc::clone(&copy.write)) else {
            return Completion::Failed;
        };
        if ok {
            for copy in copies {
                copy.write.valid.set(copy.write.valid.get() | copy.memory);
            }
        }
        let held = self
            .owners
            .get(&(client, owner))
            .and_then(|found| found.tables[usize::from(class)].get(&at))
            .is_some_and(|held| Rc::ptr_eq(held.block(), &block));
        let completion = if ok && held {
            Completion::Replicated {
                runs: self.read_runs(client, group, class, &block),
            }
        } else {
            Completion::Failed
        };
        self.reclaim(block);
        completion
    }

    /// Whether a prefetch in flight brings the key.
    fn fetching(&self, key: &Key) -> bool {
        self.copies.values().flatten().any(|copy| {
            copy.deadline.is_some()
                && copy.reads.is_empty()
                && matches!(copy.to, Target::Entry { key: other, remote: false } if other == *key)
        })
    }

    /// `Prefetch`: a task brings into new host blocks, on the host parts
    /// the group copies the class with, the leading keys that are not on the
    /// host. A key whose host copy holds those parts counts as found; one
    /// whose host copy lies on other parts ends the leading keys, since a
    /// prefetch writes only a new host block.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefetch(
        &mut self,
        queue: QueueId,
        id: RequestId,
        group: GroupId,
        class: u8,
        units: u32,
        keys: &[Key],
        deadline: u64,
    ) -> Taken {
        let (region, fixed) = self.class(queue.0, class)?.host.ok_or(Cause::Class)?;
        if !self.groups.contains_key(&(queue.0, group)) {
            return Err(Cause::Group);
        }
        let size = match (fixed, units) {
            (0, stated) if stated > 0 => stated,
            (fixed, 0) if fixed > 0 => fixed,
            _ => return Err(Cause::Size),
        };
        let parts = self.parts(queue.0, group, class);

        let (mut found, mut copies) = (Vec::new(), Vec::new());
        for key in keys {
            match self.cache.get(key) {
                Some(entry) if entry.class != class => break,
                Some(Entry {
                    host: Some((host, maker, _)),
                    ..
                }) => {
                    let holds = self.loadable(queue.0, group, class, host);
                    if !copies.is_empty() || !self.visible(*maker, queue.0) || !holds {
                        break;
                    }
                    found.push(*key);
                    continue;
                }
                _ => {}
            }
            // One prefetch brings a key at a time, into a free host block:
            // a prefetch evicts nothing.
            if self.fetching(key) {
                break;
            }
            let Some(run) = self.regions[usize::from(region)].take(size) else {
                break;
            };
            // Valid on the group's parts once the connector wrote them.
            let write = Rc::new(Block::new(class, vec![run], 0, 0));
            let to = Target::Entry {
                key: *key,
                remote: false,
            };
            copies.push(Copy {
                memory: parts,
                reads: Vec::new(),
                write,
                to,
                fill: false,
                deadline: Some(deadline),
            });
        }
        if copies.is_empty() {
            return Ok(Some(Completion::Prefetched {
                found: found.len() as u32,
            }));
        }

        self.hold_copies((queue, id), copies);
        self.answers.insert((queue, id), found);
        self.started.push((queue, id));
        Ok(None)
    }

    /// `Store` to the remote store: the connector's task reads the host
    /// copies of the named keys that have one every epoch may use (R8) and
    /// that the remote store does not hold.
    pub(super) fn put(&mut self, queue: QueueId, id: RequestId, keys: &[Key]) -> Taken {
        let deadline = Some(self.now.saturating_add(self.limits.1));
        let mut copies: Vec<Copy> = Vec::new();
        for key in keys {
            let Some(Entry {
                host: Some((host, None, _)),
                remote: false,
                ..
            }) = self.cache.get(key)
            else {
                continue;
            };
            let putting = self.copies.values().flatten().chain(&copies).any(|copy| {
                matches!(copy.to, Target::Entry { key: other, remote: true } if other == *key)
            });
            if putting {
                continue;
            }
            let to = Target::Entry {
                key: *key,
                remote: true,
            };
            copies.push(Copy {
                memory: 0,
                reads: vec![Rc::clone(host)],
                write: Rc::clone(host),
                to,
                fill: false,
                deadline,
            });
        }
        if copies.is_empty() {
            return Ok(Some(Completion::Stored { keys: Vec::new() }));
        }

        self.hold_copies((queue, id), copies);
        self.answers.insert((queue, id), Vec::new());
        self.started.push((queue, id));
        Ok(None)
    }

    /// The next key of a prefetch or of a put is done: its host block holds
    /// `tokens` tokens read from a source that held the block throughout,
    /// or the remote store acknowledged its copy. The task's request is
    /// answered when its last key is done.
    pub fn done(&mut self, task: (QueueId, RequestId), tokens: u16) {
        let Some(copies) = self.copies.get_mut(&task) else {
            return;
        };
        if !matches!(
            copies.first(),
            Some(Copy {
                to: Target::Entry { .. },
                deadline: Some(_),
                ..
            })
        ) {
            return;
        }
        let copy = copies.remove(0);
        let last = copies.is_empty();
        let Target::Entry { key, remote } = copy.to else {
            return;
        };

        // A task answered at its deadline caches nothing more.
        if self.answers.contains_key(&task) {
            let class = copy.write.class;
            // A prefetched copy is taken as written in the encoding of the class the prefetch
            // names; a session that no longer states the class caches nothing of it.
            let written = self.class(task.0.0, class).ok();
            let written = written.map(|found| (found.tensors, found.encoding));
            let done = match self.cache.get_mut(&key) {
                // The remote store holds the content of the key's host copy.
                Some(entry) if remote => {
                    let same = entry
                        .host
                        .as_ref()
                        .is_some_and(|(host, ..)| Rc::ptr_eq(host, &copy.write));
                    entry.remote |= same;
                    same
                }
                None if remote => false,
                // What the remote store and the peers hold is what every epoch
                // may use, on the host parts the connector wrote.
                Some(entry) if entry.host.is_none() && entry.class == class => match written {
                    Some(encoding) => {
                        entry.host = Some((Rc::clone(&copy.write), None, encoding));
                        copy.write.cached.set(Some(key));
                        copy.write.valid.set(copy.memory);
                        true
                    }
                    None => false,
                },
                // R4: the key gained a host copy meanwhile and keeps it; it is
                // found where it holds the parts the prefetch was for.
                Some(entry) => {
                    let host = entry.host.as_ref();
                    let maker = host.and_then(|(_, maker, _)| *maker);
                    let holds = host
                        .is_some_and(|(block, ..)| block.valid.get() & copy.memory == copy.memory);
                    entry.class == class && self.visible(maker, task.0.0) && holds
                }
                None if written.is_none() => false,
                None => {
                    let host = written.map(|encoding| (Rc::clone(&copy.write), None, encoding));
                    copy.write.cached.set(Some(key));
                    copy.write.valid.set(copy.memory);
                    self.cache.insert(
                        key,
                        Entry {
                            class,
                            device: Vec::new(),
                            host,
                            tokens,
                            remote: false,
                        },
                    );
                    true
                }
            };
            if let Some(keys) = self.answers.get_mut(&task).filter(|_| done) {
                keys.push(key);
            }
        }
        self.drop_copy(task.0.0, copy);

        if last {
            self.copies.remove(&task);
            if let Some(keys) = self.answers.remove(&task) {
                let completion = if remote {
                    Completion::Stored { keys }
                } else {
                    Completion::Prefetched {
                        found: keys.len() as u32,
                    }
                };
                self.complete(task.0, task.1, completion);
            }
        }
        self.settle_counts();
    }

    /// A5: a task is over and touches its blocks no more. R9: what it held
    /// is released now. `ok` is whether a transfer copied everything; the
    /// keys of a prefetch or a put that were not reported done are not.
    pub fn ended(&mut self, task: (QueueId, RequestId), ok: bool) {
        let Some(copies) = self.copies.remove(&task) else {
            return;
        };
        let answer = self.answers.remove(&task);
        let completion = match copies.first().map(|copy| copy.to) {
            Some(Target::Group { owner, to }) => {
                let completion = self.transferred(task.0.0, owner, to, &copies, ok);
                answer.map(|_| completion)
            }
            Some(Target::Memory {
                owner,
                class,
                at,
                group,
                ..
            }) => {
                let completion = self.replicated(task.0.0, &copies, (owner, class, at), group, ok);
                answer.map(|_| completion)
            }
            Some(Target::Entry { remote: true, .. }) => {
                answer.map(|keys| Completion::Stored { keys })
            }
            Some(Target::Entry { remote: false, .. }) => {
                answer.map(|keys| Completion::Prefetched {
                    found: keys.len() as u32,
                })
            }
            _ => None,
        };

        for copy in copies {
            self.drop_copy(task.0.0, copy);
        }
        if let Some(completion) = completion {
            self.complete(task.0, task.1, completion);
        }
        self.sweep();
        self.settle();
        self.settle_counts();
    }

    /// Tells the buffer pool the time. A task past its deadline is told to
    /// stop; a prefetch and a put are answered at once with what they
    /// brought, a transfer when its task has ended. R9: the blocks a task
    /// holds stay held until its end. Returns whether a task passed its
    /// deadline.
    pub fn tick(&mut self, now: u64) -> bool {
        self.now = now;
        let late = |copies: &Vec<Copy>| {
            copies
                .first()
                .and_then(|copy| copy.deadline)
                .is_some_and(|deadline| deadline <= now)
        };
        let due: Vec<(QueueId, RequestId)> = self
            .copies
            .iter()
            .filter(|(_, copies)| late(copies))
            .map(|(task, _)| *task)
            .collect();
        let passed = !due.is_empty();

        for task in due {
            let Some(copies) = self.copies.get_mut(&task) else {
                continue;
            };
            for copy in copies.iter_mut() {
                copy.deadline = Some(STOPPING);
            }
            let completion = match copies.first().map(|copy| &copy.to) {
                Some(Target::Entry { remote: true, .. }) => self
                    .answers
                    .remove(&task)
                    .map(|keys| Completion::Stored { keys }),
                // A crossing past its deadline has failed; the end of its
                // task still makes the block valid where it copied it.
                Some(Target::Memory { .. }) => {
                    self.answers.remove(&task).map(|_| Completion::Failed)
                }
                Some(Target::Entry { remote: false, .. }) => {
                    self.answers
                        .remove(&task)
                        .map(|keys| Completion::Prefetched {
                            found: keys.len() as u32,
                        })
                }
                _ => None,
            };
            self.stopped.push(task);
            if let Some(completion) = completion {
                self.complete(task.0, task.1, completion);
            }
        }
        self.settle_counts();
        passed
    }
}
