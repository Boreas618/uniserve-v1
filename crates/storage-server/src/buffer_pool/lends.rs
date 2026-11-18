//! Handoffs between replicas: a lend of a prefill replica's owner, and the import of its blocks
//! into a decode replica's owner (`specs/kv-cache.md` §4).
//!
//! - **`Lend`.** Once the stamps of the batches that may still write the owner's blocks are
//!   reached, the owner is lent: it takes no other request, its blocks stay as they are, and an
//!   import may read them. The lend ends when the importer returns it ([`BufferPool::give_back`])
//!   or its lease passes, and the owner is then released as a `Detach` releases it. A lend's
//!   end never frees a block something still reads: an import of this pool holds what its
//!   copies read, and a borrow of another host's server holds the lent blocks under its
//!   connection ([`BufferPool::borrow`]) until it returns them or its connection closes (R9).
//! - **`Import`.** The owner's private blocks at the positions named are written by a task of
//!   the Transfer Server, read from the lender's blocks at the same positions: a lend of this
//!   pool, whose blocks the task holds, or of another host's pool, which this host's server
//!   borrows. The owner takes no other request until the task ends. The task is answered
//!   `Done` when it copied every page, `Failed` otherwise; the main loop returns the lend once
//!   the task ended.

use std::collections::BTreeMap;
use std::rc::Rc;

use serde::{Deserialize, Serialize};
use service::span::Stage;

use super::region::Run;
use storage::runtime::{Cause, Completion, Lender, OwnerId, RequestId, Stamps};

use super::{
    Block, BufferPool, ClientId, Copy, Deadline, Deferred, Held, QueueId, Taken, Target, Validity,
};

/// A lent block by its class and position.
type LentBlock = (u8, u32, Rc<Block>);

/// What an import copies, as the main loop runs it ([`BufferPool::import_work`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportWork {
    /// The owner it imports from.
    pub lender: Lender,
    /// Whether the lender's host's Transfer Server writes the pages (a push), or this host's
    /// reads them.
    pub push: bool,
    /// The transport the pages cross between the two hosts by.
    pub transport: scheduling::kv::Transport,
    /// The memory the importing owner's group addresses.
    pub memory: String,
    /// Per page, its class and position, the runs of the importing owner's block, and for a
    /// lender of this pool the runs of its block, by the names of their regions.
    pub pages: Vec<(u8, u32, Vec<NamedRun>, Vec<NamedRun>)>,
    /// For a lender of this pool, the memory its group addresses.
    pub lender_memory: Option<String>,
}

/// What an import keeps beside its copies until its task starts.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Importing {
    /// The importing owner.
    pub(super) owner: OwnerId,
    pub(super) lender: Lender,
    pub(super) push: bool,
    pub(super) transport: scheduling::kv::Transport,
    /// The keys evicted for its blocks, which its answer lists.
    pub(super) evicted: Vec<storage::protocol::Key>,
}

/// The blocks a lent owner holds, by class and position, as a borrow states them
/// ([`BufferPool::borrow`]), with the memory its group addresses.
pub type LentBlocks = (String, Vec<(u8, u32, Vec<NamedRun>)>);

/// A run of units by the name of its region, its first unit and its count.
pub type NamedRun = (String, u32, u32);

impl BufferPool {
    /// A block's runs by the names of their regions.
    fn named_runs(&self, block: &Block) -> Vec<NamedRun> {
        block
            .stated()
            .into_iter()
            .map(|(region, first, count)| (self.region_name(region).to_owned(), first, count))
            .collect()
    }

    /// See [`storage::runtime::Request::Lend`]. R3: the lend takes effect once the stamps are
    /// reached; until then the owner takes no other request either.
    pub(super) fn lend(
        &mut self,
        queue: QueueId,
        request: RequestId,
        owner: OwnerId,
        lease: u64,
        after: Stamps,
    ) -> Taken {
        let client = queue.0;
        self.settled(client, owner)?;
        self.stamped(client, &after)?;
        // R1: an owner a copy writes for, or whose crossing waits or runs, is not lent.
        if self.copied(client, owner) || self.replicating(client, owner, None) {
            return Err(Cause::Copy);
        }
        if let Some(found) = self.owners.get_mut(&(client, owner)) {
            found.moving = true;
        }
        self.defer(
            client,
            after,
            Deferred::Lend {
                queue,
                request,
                owner,
                lease,
                rest: Vec::new(),
            },
        );
        Ok(None)
    }

    /// A lend takes effect: an import may read the owner's blocks until its lease passes.
    pub(super) fn lent(&mut self, queue: QueueId, request: RequestId, owner: OwnerId, lease: u64) {
        let client = queue.0;
        if self.owners.contains_key(&(client, owner)) {
            let deadline = self.now.saturating_add(lease);
            self.lends.insert((client, owner), deadline);
        }
        self.complete(queue, request, Completion::Done);
    }

    /// The lend of the owner `owner` of `client` ends: the owner is released as a `Detach`
    /// releases it, every stamp it waited for being reached. What still reads its blocks keeps
    /// them (R9). Returns whether such a lend stood.
    pub fn give_back(&mut self, client: ClientId, owner: OwnerId) -> bool {
        if self.lends.remove(&(client, owner)).is_none() {
            return false;
        }
        self.orphan(client, owner);
        if let Some(found) = self.owners.remove(&(client, owner)) {
            for block in super::disowned(found) {
                self.reclaim(block);
            }
        }
        self.sweep();
        self.settle();
        self.settle_counts();
        true
    }

    /// The lends whose lease passed by `now` end ([`Self::give_back`]), in the order of their
    /// owners. Returns whether one did.
    pub(super) fn expire_lends(&mut self, now: u64) -> bool {
        let due: Vec<(ClientId, OwnerId)> = self
            .lends
            .iter()
            .filter(|(_, deadline)| **deadline <= now)
            .map(|(lend, _)| *lend)
            .collect();
        for (client, owner) in &due {
            self.give_back(*client, *owner);
        }
        !due.is_empty()
    }

    /// The earliest lease of a lend: when [`Self::tick`] next ends one.
    pub(super) fn next_lease(&self) -> Option<u64> {
        self.lends.values().min().copied()
    }

    /// The blocks the lent owner `owner` of `client` holds, by class and position, with the
    /// memory its group addresses; `None` when no such lend stands.
    fn lent_blocks(&self, client: ClientId, owner: OwnerId) -> Option<(u8, Vec<LentBlock>)> {
        if !self.lends.contains_key(&(client, owner)) {
            return None;
        }
        let found = self.owners.get(&(client, owner))?;
        let memory = self.groups.get(&(client, found.group))?.memory;
        let blocks = found
            .tables
            .iter()
            .enumerate()
            .flat_map(|(class, table)| {
                table
                    .iter()
                    .map(move |(position, held)| (class as u8, *position, Rc::clone(held.block())))
            })
            .collect();
        Some((memory, blocks))
    }

    /// Holds the blocks the lent owner `owner` of `client` holds for `peer`, another host's
    /// Storage Server that imports them, until it returns the lend or its connection closes
    /// ([`Self::release_borrows`]); a held block is a holder like any other (R2). Returns the
    /// blocks and the memory their group addresses; `None` when no such lend stands.
    pub fn borrow(&mut self, peer: u64, client: ClientId, owner: OwnerId) -> Option<LentBlocks> {
        let (memory, blocks) = self.lent_blocks(client, owner)?;
        let held = blocks
            .iter()
            .map(|(_, _, block)| Rc::clone(block))
            .collect();
        for (_, _, block) in &blocks {
            self.touch(block);
        }
        self.borrows.insert((peer, client, owner), held);
        let stated = blocks
            .iter()
            .map(|(class, position, block)| (*class, *position, self.named_runs(block)))
            .collect();
        self.settle_counts();
        Some((self.memories[usize::from(memory)].0.clone(), stated))
    }

    /// The borrows of `peer` end: of the lend of `client`'s `owner`, or every one of the
    /// peer's. Returns whether one stood.
    pub fn release_borrows(&mut self, peer: u64, lend: Option<(ClientId, OwnerId)>) -> bool {
        let ended: Vec<(u64, ClientId, OwnerId)> = self
            .borrows
            .keys()
            .filter(|(of, client, owner)| {
                *of == peer && lend.is_none_or(|lend| lend == (*client, *owner))
            })
            .copied()
            .collect();
        for borrow in &ended {
            for block in self.borrows.remove(borrow).unwrap_or_default() {
                self.reclaim(block);
            }
        }
        self.sweep();
        self.settle_counts();
        !ended.is_empty()
    }

    /// See [`storage::runtime::Request::Import`]. R1: every position named is free, all or
    /// none take blocks, and the owner takes no other request until the task ends. A lender of
    /// this pool (`lender.endpoint` absent) must have its lend standing, or the import fails at
    /// once; its blocks at the same positions are what the task reads, and holds (R9).
    pub(super) fn import(
        &mut self,
        queue: QueueId,
        request: RequestId,
        owner: OwnerId,
        tables: &[(u8, u32, u32)],
        (lender, push, transport): (Lender, bool, scheduling::kv::Transport),
    ) -> Taken {
        let client = queue.0;
        let group = self.settled(client, owner)?.group;
        let memory = self.memory(client, group);

        // Each table's blocks: its class's size and its one region, with room for them all.
        let mut wanted: Vec<(u8, u32, u32, u8, u32)> = Vec::with_capacity(tables.len());
        for &(class, at, count) in tables {
            let size = self.class(client, class)?.units;
            if size == 0 {
                return Err(Cause::Size);
            }
            if count == 0 || !self.free_positions(client, owner, class, at, count) {
                return Err(Cause::Position);
            }
            let region = self.region(client, group, class)?;
            wanted.push((class, at, count, region, size));
        }

        // A lender of this pool lends the blocks the copies read.
        let mut reads: BTreeMap<(u8, u32), Rc<Block>> = BTreeMap::new();
        if lender.endpoint.is_none() {
            let Some((_, blocks)) = self.lent_blocks(lender.session, lender.owner) else {
                return Ok(Some(Completion::Failed));
            };
            reads = blocks
                .into_iter()
                .map(|(class, position, block)| ((class, position), block))
                .collect();
            let lacking = wanted.iter().any(|(class, at, count, ..)| {
                (*at..at + count).any(|position| !reads.contains_key(&(*class, position)))
            });
            if lacking {
                return Ok(Some(Completion::Failed));
            }
        }

        // All or none: every region has room for its blocks before any is taken.
        let mut needs: BTreeMap<u8, (u32, u32)> = BTreeMap::new();
        for (_, _, count, region, size) in &wanted {
            let need = needs.entry(*region).or_insert((0, *size));
            need.0 += count;
        }
        let mut evictions = Vec::new();
        for (region, (count, size)) in &needs {
            match self.room(queue, *region, *count, *size) {
                Some(keys) => evictions.push((*region, keys)),
                None => return Ok(Some(Completion::Full { region: *region })),
            }
        }
        for (region, keys) in &evictions {
            for key in keys {
                self.evict_in(key, *region);
            }
        }

        if let Some(found) = self.owners.get_mut(&(client, owner)) {
            found.moving = true;
        }
        let deadline = Some(Deadline::At(self.now.saturating_add(self.limits.0)));
        let mut copies = Vec::new();
        for (class, at, count, region, size) in wanted {
            for position in at..at + count {
                let runs = self.regions[usize::from(region)].take(size);
                // Valid in the owner's group's memory once the task wrote it; the copy that
                // writes it makes it private until it enters the owner's table.
                let block = Block::new(
                    Some(class),
                    runs.into_iter().collect(),
                    Validity::In(memory),
                    0,
                );
                copies.push(Copy {
                    memory,
                    reads: reads
                        .get(&(class, position))
                        .map(Rc::clone)
                        .into_iter()
                        .collect(),
                    write: Rc::new(block),
                    to: Target::Import {
                        owner,
                        class,
                        at: position,
                    },
                    fill: false,
                    deadline,
                });
            }
        }
        let task = (queue, request);
        // The handoff's span: from its task's start to its end, with the bytes it writes into
        // the importer's memory.
        if let Some(&trace) = self.traces.get(&task) {
            let run_bytes = |run: &Run| {
                let unit = self
                    .regions
                    .get(usize::from(run.region()))
                    .map_or(0, |region| region.unit_bytes);
                u64::from(run.count()) * unit
            };
            let bytes = copies
                .iter()
                .map(|copy| copy.write.runs.borrow().iter().map(run_bytes).sum::<u64>())
                .sum();
            service::span::bytes(Stage::Handoff, trace, bytes);
            service::span::begin(Stage::Handoff, trace);
        }
        // An import of no page still starts a task, of no copy: the main loop returns its lend.
        self.hold_copies(task, copies);
        self.imports.insert(
            task,
            Importing {
                owner,
                lender,
                push,
                transport,
                evicted: evictions.into_iter().flat_map(|(_, keys)| keys).collect(),
            },
        );
        self.answers.insert(task, Vec::new());
        self.started.push(task);
        Ok(None)
    }

    /// What the import `task` copies, for the main loop that runs it; `None` for a task that is
    /// no import.
    pub fn import_work(&self, task: (QueueId, RequestId)) -> Option<ImportWork> {
        let importing = self.imports.get(&task)?;
        let copies = self.copies.get(&task).map_or(&[][..], Vec::as_slice);
        let name = |memory: u8| self.memories[usize::from(memory)].0.clone();
        let client = task.0.0;
        let memory = self
            .owners
            .get(&(client, importing.owner))
            .and_then(|found| self.groups.get(&(client, found.group)))
            .map(|group| name(group.memory))
            .unwrap_or_default();
        let lender_memory = (importing.lender.endpoint.is_none())
            .then(|| {
                let lender = &importing.lender;
                let found = self.owners.get(&(lender.session, lender.owner))?;
                Some(name(
                    self.groups.get(&(lender.session, found.group))?.memory,
                ))
            })
            .flatten();
        let pages = copies
            .iter()
            .filter_map(|copy| match copy.to {
                Target::Import { class, at, .. } => Some((
                    class,
                    at,
                    self.named_runs(&copy.write),
                    copy.reads
                        .first()
                        .map(|block| self.named_runs(block))
                        .unwrap_or_default(),
                )),
                _ => None,
            })
            .collect();
        Some(ImportWork {
            lender: importing.lender.clone(),
            push: importing.push,
            transport: importing.transport,
            memory,
            pages,
            lender_memory,
        })
    }

    /// An import's task ended: the owner takes requests again, and with every page copied its
    /// blocks enter its table, private, answered `Imported` with their runs; otherwise they
    /// are released when the copies are dropped, answered `Failed`.
    pub(super) fn imported(
        &mut self,
        task: (QueueId, RequestId),
        copies: &[Copy],
        ok: bool,
    ) -> Option<Completion> {
        let importing = self.imports.remove(&task)?;
        let client = task.0.0;
        let found = self.owners.get_mut(&(client, importing.owner))?;
        found.moving = false;
        if !ok {
            return Some(Completion::Failed);
        }
        let mut runs = Vec::new();
        for copy in copies {
            if let Target::Import { class, at, .. } = copy.to {
                copy.write.own();
                runs.extend(copy.write.stated());
                found.tables[usize::from(class)].insert(at, Held::Private(Rc::clone(&copy.write)));
            }
        }
        for copy in copies {
            self.touch(&copy.write);
        }
        Some(Completion::Imported {
            runs,
            evicted: importing.evicted,
        })
    }
}
