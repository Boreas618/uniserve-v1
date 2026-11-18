//! Tasks: the copies the Transfer Server and the connector run. An import
//! fills an owner's blocks from a lend (`lends`), a crossing (`Replicate`) brings a block
//! into the memories another group reads it in, a prefetch brings keys into
//! the host tier, a put writes host copies to the remote store.
//!
//! A task is the copies of one request. They hold their blocks until the
//! task has ended (R9), whatever became of the request: answered at its
//! deadline, its owner detached, its session closed. The buffer pool lists
//! the tasks to start and to stop; whoever runs them reports each key done
//! and the task's end.

use std::rc::Rc;

use service::span::Stage;
use storage::protocol::Key;
use storage::runtime::{Cause, Completion, GroupId, OwnerId, RequestId, Stamps};

use super::region::Run;

use super::{
    Block, BufferPool, ClientId, Copy, Deadline, Deferred, Encoding, Entry, Held, Memories,
    QueueId, Taken, Target, Validity,
};

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
    /// reads and writes, where the content it reads is valid and the
    /// memories it writes. An import's copies are stated apart
    /// ([`BufferPool::import_work`]). A crossing's copies have no
    /// key; each reads the block's run in the region the request named,
    /// from the memories of that run that hold the block, and writes the
    /// block's runs that lie in one memory of the reading group. A
    /// prefetch's copies read nothing and write the host parts of the group
    /// that asked; a put's read and write the host block it puts.
    pub fn work(
        &self,
        task: (QueueId, RequestId),
    ) -> Vec<(Option<Key>, Runs, Runs, Validity, Memories)> {
        let copies = self.copies.get(&task).map_or(&[][..], Vec::as_slice);
        let stated = |copy: &Copy| {
            let key = match copy.to {
                Target::Entry { key, .. } => Some(key),
                _ => None,
            };
            let read = copy.reads.first();
            if let (Target::Memory { from, .. }, Some(block)) = (copy.to, read) {
                let runs = block.runs.borrow();
                let reads = runs.iter().filter(|run| run.region() == from);
                // A run of a region each memory holds lies in every memory
                // that holds the block; a run of a region whose units exist
                // once lies in that region's memory.
                let source = block.valid.get().within(self.lies_in(from, Memories::ALL));
                let written =
                    |run: &&Run| self.lies_in(run.region(), copy.memory).meets(copy.memory);
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
            let memories = read.map_or(Validity::NOWHERE, |block| block.valid.get());
            (key, reads, copy.write.stated(), memories, copy.memory)
        };
        copies.iter().map(stated).collect()
    }

    /// The earliest deadline of a task not yet told to stop, in nanoseconds
    /// of the host's monotonic clock: the time at which [`BufferPool::tick`]
    /// next has something to do. `None` when no task has one.
    pub fn next_deadline(&self) -> Option<u64> {
        let tasks = self
            .copies
            .values()
            .filter_map(|copies| copies.first()?.deadline?.at());
        tasks.chain(self.next_lease()).min()
    }

    /// Whether a task is outstanding: started and not ended, or not yet
    /// handed to whoever runs it. A restart in place waits until none is.
    pub fn tasks(&self) -> bool {
        let task = |copies: &Vec<Copy>| copies.iter().any(|copy| copy.deadline.is_some());
        !self.started.is_empty() || !self.stopped.is_empty() || self.copies.values().any(task)
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
    fn reading(&self, client: ClientId, group: GroupId, class: u8) -> Memories {
        let Some(found) = self.groups.get(&(client, group)) else {
            return Memories::NONE;
        };
        let own = Memories::one(found.memory);
        let lists = [&found.regions, &found.receives];
        let regions = lists
            .into_iter()
            .filter_map(|lists| lists.get(usize::from(class)))
            .flatten();
        regions.map(|region| self.lies_in(*region, own)).collect()
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
        let reader_memory = Memories::one(reader.memory);
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
        if private && written.contains(&from) && self.spread(&written, own).count() > 1 {
            valid = valid.and(self.lies_in(from, own));
        }
        // R11: the run read lies in a memory that holds the block.
        if !valid.meets(self.lies_in(from, own)) {
            return Err(Cause::Class);
        }

        // R5: the memories the group reads the class in that do not hold the
        // block; in each, the block's run there or a run to take in the
        // receiving region there, of the bytes of the run it is read from.
        let targets = valid.lacking(self.reading(client, group, class));
        let bytes = {
            let runs = block.runs.borrow();
            let source = runs.iter().find(|run| run.region() == from);
            let units = source.map_or(0, Run::count);
            u64::from(units) * self.regions[usize::from(from)].unit_bytes
        };
        let mut taken: Vec<(u8, u32, Vec<Key>)> = Vec::new();
        for memory in targets.each() {
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
        // The keys evicted for the runs taken, which the answer names (SD §4.5): the
        // crossing owes them until it is answered.
        let mut evicted = Vec::new();
        if !taken.is_empty() {
            // The block's units are counted again with its new runs.
            self.uncount(&block);
            for (region, size, keys) in taken {
                for key in &keys {
                    self.evict_in(key, region);
                }
                evicted.extend(keys);
                let run = self.regions[usize::from(region)].take(size);
                block.runs.borrow_mut().extend(run);
            }
            self.touch(&block);
        }
        if targets.is_empty() {
            let runs = self.read_runs(client, group, class, &block);
            return Ok(Some(Completion::Replicated { runs, evicted }));
        }
        self.answers.insert((queue, id), evicted);

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
    /// The keys its take evicted (`answers`) are named by its `Replicated`.
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
        let filled = reading.each().any(|memory| self.filling(&block, memory));
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
        let targets = block.valid.get().lacking(reading);
        if !held || targets.is_empty() {
            let evicted = self.answers.remove(&(queue, request)).unwrap_or_default();
            let completion = if held {
                Completion::Replicated {
                    runs: self.read_runs(client, to, class, &block),
                    evicted,
                }
            } else {
                Completion::Failed
            };
            self.reclaim(block);
            self.complete(queue, request, completion);
            return None;
        }

        let deadline = Some(Deadline::At(self.now.saturating_add(self.limits.0)));
        let target = Target::Memory {
            owner,
            class,
            at,
            group: to,
            from,
        };
        let copies = targets
            .each()
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
        // The task's copies hold the block from now on, and the task owes the keys its take
        // evicted.
        self.reclaim(block);
        self.answers.entry((queue, request)).or_default();
        self.started.push((queue, request));
        None
    }

    /// A crossing's task ended (A5). When it copied everything, every memory
    /// it wrote holds the block, whatever became of its owner; the answer is
    /// the block's runs the group reads the class in, with `evicted`, the
    /// keys its take evicted, while the owner still holds the block at its
    /// position, and `Failed` otherwise.
    fn replicated(
        &mut self,
        client: ClientId,
        copies: &[Copy],
        (owner, class, at): (OwnerId, u8, u32),
        (group, evicted): (GroupId, Vec<Key>),
        ok: bool,
    ) -> Completion {
        let Some(block) = copies.first().map(|copy| Rc::clone(&copy.write)) else {
            return Completion::Failed;
        };
        if ok {
            for copy in copies {
                copy.write
                    .valid
                    .set(copy.write.valid.get().and(copy.memory));
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
                evicted,
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
    /// prefetch writes only a new host block. The new blocks are taken as a
    /// `Store` to the host takes them, from the free units and then by
    /// evicting in the queue pair's order of the host region, for as many of
    /// the keys to bring as fit; the host copies of the keys found are not
    /// evicted for them. The answer lists the keys evicted.
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

        // The leading keys: those found on the host, then those to bring. The
        // found keys' host copies are held while room is made, so that no
        // eviction takes them (R2).
        let (mut found, mut held, mut wanted) = (Vec::new(), Vec::new(), Vec::new());
        for key in keys {
            match self.cache.get(key) {
                Some(entry) if entry.class != class => break,
                Some(Entry {
                    host: Some((host, maker, _)),
                    ..
                }) => {
                    let holds = self.loadable(queue.0, group, class, host);
                    if !wanted.is_empty() || !self.visible(*maker, queue.0) || !holds {
                        break;
                    }
                    found.push(*key);
                    held.push(Rc::clone(host));
                    continue;
                }
                _ => {}
            }
            // One prefetch at a time brings a key.
            if self.fetching(key) {
                break;
            }
            wanted.push(*key);
        }

        // As many of the keys to bring as the free units and the order make
        // room for.
        let mut evicted = Vec::new();
        while !wanted.is_empty() {
            let count = u32::try_from(wanted.len()).unwrap_or(u32::MAX);
            if let Some(keys) = self.room(queue, region, count, size) {
                evicted = keys;
                break;
            }
            wanted.pop();
        }
        drop(held);
        let count = u32::try_from(wanted.len()).unwrap_or(u32::MAX);
        let blocks = self.obtain(region, class, count, size, &evicted, Validity::NOWHERE);

        // Each block is valid on the group's parts once the connector wrote it.
        let copies: Vec<Copy> = wanted
            .iter()
            .zip(blocks)
            .map(|(key, write)| Copy {
                memory: parts,
                reads: Vec::new(),
                write,
                to: Target::Entry {
                    key: *key,
                    remote: false,
                },
                fill: false,
                deadline: Some(Deadline::At(deadline)),
            })
            .collect();
        if copies.is_empty() {
            return Ok(Some(Completion::Prefetched {
                found: found.len() as u32,
                evicted,
            }));
        }

        self.hold_copies((queue, id), copies);
        self.answers.insert((queue, id), found);
        self.evictions.insert((queue, id), evicted);
        self.started.push((queue, id));
        Ok(None)
    }

    /// A prefetch from the peers learned what the peers hold of its keys:
    /// per copy of the task, in order, the encoding of the peer's host copy
    /// it reads, `None` for a key no peer holds. A copy keeps its block of
    /// the class's host region where the peer's copy is in the class's own
    /// encoding; one in the encoding of the class's converted-load source
    /// (`Class::source`) moves to a new block of the source region, which a
    /// `Load` then answers with the two encodings to convert between.
    /// Returns the runs the leading copies write, up to the first that no
    /// peer copy fills: one without a peer copy, in another encoding, or
    /// without room in the source region. Nothing for a task answered
    /// already, which caches nothing more.
    pub fn peer_copies(&mut self, task: (QueueId, RequestId), held: &[Option<Encoding>]) -> Runs {
        if !self.answers.contains_key(&task) {
            return Vec::new();
        }
        let first = self.copies.get(&task).and_then(|copies| copies.first());
        let Some(class) = first.and_then(|copy| copy.write.class) else {
            return Vec::new();
        };
        let Ok(found) = self.class(task.0.0, class) else {
            return Vec::new();
        };
        let own = (found.tensors, found.encoding);
        let source = found
            .source
            .map(|(region, encoding)| (region, (found.tensors, encoding)));

        let mut runs = Vec::new();
        let mut evicted = Vec::new();
        for (index, encoding) in held.iter().enumerate() {
            let Some(copy) = self.copies.get(&task).and_then(|copies| copies.get(index)) else {
                break;
            };
            match (encoding, source) {
                (Some(encoding), _) if *encoding == own => {}
                (Some(encoding), Some((region, written))) if *encoding == written => {
                    let units = copy.write.units();
                    let Some(block) = self.staged(region, class, units, &mut evicted) else {
                        break;
                    };
                    self.restage(task, index, block);
                }
                _ => break,
            }
            match self.copies.get(&task).and_then(|copies| copies.get(index)) {
                Some(copy) => runs.extend(copy.write.stated()),
                None => break,
            }
        }

        if let Some(listed) = self.evictions.get_mut(&task) {
            listed.extend(evicted);
        }
        self.settle_counts();
        runs
    }

    /// A new block of `units` units of the converted-load source region
    /// `region` for `class`: from its free units, else in the place of its
    /// unheld host copies, lowest unit first. No queue pair orders the
    /// region, whose copies are the staged reads of prefetches, so the order
    /// is the units'; the keys whose copies leave are added to `evicted`.
    /// `None` when held blocks fill the region.
    fn staged(
        &mut self,
        region: u8,
        class: u8,
        units: u32,
        evicted: &mut Vec<Key>,
    ) -> Option<Rc<Block>> {
        loop {
            if let Some(run) = self.regions[usize::from(region)].take(units) {
                // The copy that writes it makes it private (`Self::restage`).
                return Some(Rc::new(Block::new(
                    Some(class),
                    vec![run],
                    Validity::NOWHERE,
                    0,
                )));
            }
            let lowest = self
                .cache
                .iter()
                .filter_map(|(key, entry)| {
                    let (block, ..) = entry.host.as_ref()?;
                    let unheld = block.region() == region && Rc::strong_count(block) == 1;
                    let first = block.runs.borrow().first().map(Run::first)?;
                    unheld.then_some((first, *key))
                })
                .min();
            let (_, key) = lowest?;
            if !self.evict_in(&key, region) {
                return None;
            }
            evicted.push(key);
        }
    }

    /// Copy `index` of `task` writes `block` instead of the block it was
    /// given, which it lets go.
    fn restage(&mut self, task: (QueueId, RequestId), index: usize, block: Rc<Block>) {
        let live = match self.copies.get(&task).and_then(|copies| copies.get(index)) {
            Some(copy) => self.live(task.0.0, copy),
            None => return,
        };
        let Some(copy) = self
            .copies
            .get_mut(&task)
            .and_then(|copies| copies.get_mut(index))
        else {
            return;
        };
        let given = std::mem::replace(&mut copy.write, block);
        if live {
            copy.write.own();
            given.disown();
        }
        let written = Rc::clone(&copy.write);
        self.touch(&written);
        self.reclaim(given);
    }

    /// The answer to the prefetch `task`, which found `found` leading keys:
    /// with the keys it evicted for its blocks.
    fn prefetched(&mut self, task: (QueueId, RequestId), found: u32) -> Completion {
        Completion::Prefetched {
            found,
            evicted: self.evictions.remove(&task).unwrap_or_default(),
        }
    }

    /// `Store` to the remote store: the connector's task reads the host
    /// copies of the named keys that have one every epoch may use (R8) and
    /// that the remote store does not hold.
    pub(super) fn put(&mut self, queue: QueueId, id: RequestId, keys: &[Key]) -> Taken {
        let deadline = Some(Deadline::At(self.now.saturating_add(self.limits.1)));
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
                memory: Memories::NONE,
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
            // names, or in its converted-load source's when it lies in the source region
            // ([`Self::peer_copies`]), with that class; a session that no longer states the
            // class caches nothing of it.
            let written = class.and_then(|class| {
                let found = self.class(task.0.0, class).ok()?;
                let region = copy.write.region();
                let staged = found.source.filter(|(source, _)| *source == region);
                let encoding = staged.map_or(found.encoding, |(_, encoding)| encoding);
                Some((class, (found.tensors, encoding)))
            });
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
                Some(entry) if entry.host.is_none() && Some(entry.class) == class => {
                    match written {
                        Some((_, encoding)) => {
                            entry.host = Some((Rc::clone(&copy.write), None, encoding));
                            copy.write.cached.set(Some(key));
                            copy.write.valid.set(Validity::In(copy.memory));
                            true
                        }
                        None => false,
                    }
                }
                // R4: the key gained a host copy meanwhile and keeps it; it is
                // found where it holds the parts the prefetch was for.
                Some(entry) => {
                    let host = entry.host.as_ref();
                    let maker = host.and_then(|(_, maker, _)| *maker);
                    let holds =
                        host.is_some_and(|(block, ..)| block.valid.get().holds(copy.memory));
                    Some(entry.class) == class && self.visible(maker, task.0.0) && holds
                }
                None => match written {
                    None => false,
                    Some((class, encoding)) => {
                        copy.write.cached.set(Some(key));
                        copy.write.valid.set(Validity::In(copy.memory));
                        self.cache.insert(
                            key,
                            Entry {
                                class,
                                device: Vec::new(),
                                host: Some((Rc::clone(&copy.write), None, encoding)),
                                tokens,
                                remote: false,
                            },
                        );
                        true
                    }
                },
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
                    self.prefetched(task, keys.len() as u32)
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
            // An import, of pages or of none.
            _ if self.imports.contains_key(&task) => {
                let completion = self.imported(task, &copies, ok);
                if let Some(trace) = self.traces.remove(&task) {
                    service::span::end(Stage::Handoff, trace);
                }
                answer.and(completion)
            }
            Some(Target::Memory {
                owner,
                class,
                at,
                group,
                ..
            }) => {
                // A crossing answered at its deadline still makes valid what it copied.
                let evicted = answer.clone().unwrap_or_default();
                let replicated = (owner, class, at);
                let completion =
                    self.replicated(task.0.0, &copies, replicated, (group, evicted), ok);
                answer.map(|_| completion)
            }
            Some(Target::Entry { remote: true, .. }) => {
                answer.map(|keys| Completion::Stored { keys })
            }
            Some(Target::Entry { remote: false, .. }) => {
                answer.map(|keys| self.prefetched(task, keys.len() as u32))
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
    ///
    /// The tasks that pass their deadlines at one tick are stopped and
    /// answered in the order of their deadlines, then of their requests: the
    /// pool is deterministic in its inputs (`specs/buffer-pool.md` §18), so
    /// nothing it produces may follow the order of a hash table.
    pub fn tick(&mut self, now: u64) -> bool {
        self.now = now;
        let expired = self.expire_lends(now);
        let late = |copies: &Vec<Copy>| {
            copies
                .first()
                .and_then(|copy| copy.deadline?.at())
                .filter(|deadline| *deadline <= now)
        };
        let mut due: Vec<(u64, (QueueId, RequestId))> = self
            .copies
            .iter()
            .filter_map(|(task, copies)| late(copies).map(|deadline| (deadline, *task)))
            .collect();
        due.sort_unstable();
        let passed = !due.is_empty();

        for (_, task) in due {
            let Some(copies) = self.copies.get_mut(&task) else {
                continue;
            };
            for copy in copies.iter_mut() {
                copy.deadline = Some(Deadline::Passed);
            }
            let target = copies.first().map(|copy| copy.to);
            let completion = match target {
                Some(Target::Entry { remote: true, .. }) => self
                    .answers
                    .remove(&task)
                    .map(|keys| Completion::Stored { keys }),
                // A crossing past its deadline has failed; the end of its
                // task still makes the block valid where it copied it.
                Some(Target::Memory { .. }) => {
                    self.answers.remove(&task).map(|_| Completion::Failed)
                }
                Some(Target::Entry { remote: false, .. }) => self
                    .answers
                    .remove(&task)
                    .map(|keys| self.prefetched(task, keys.len() as u32)),
                _ => None,
            };
            self.stopped.push(task);
            if let Some(completion) = completion {
                self.complete(task.0, task.1, completion);
            }
        }
        self.settle_counts();
        passed || expired
    }
}
