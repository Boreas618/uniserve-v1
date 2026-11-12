//! Tasks: the copies the Transfer Server and the connector run. A transfer
//! moves an owner to another group, a prefetch brings keys into the host
//! tier, a put writes host copies to the remote store.
//!
//! A task is the copies of one request. They hold their blocks until the
//! task has ended (R9), whatever became of the request: answered at its
//! deadline, its owner detached, its session closed. The buffer pool lists
//! the tasks to start and to stop; whoever runs them reports each key done
//! and the task's end.

use std::cell::Cell;
use std::rc::Rc;

use storage::protocol::Key;
use storage::runtime::{Cause, Completion, GroupId, OwnerId, RequestId, Stamps};

use super::{
    Block, BufferPool, ClientId, Copy, Deferred, EVERYWHERE, Entry, Held, QueueId, SLOT, Taken,
    Target,
};

/// The deadline of a task that was told to stop: it has no further one.
const STOPPING: u64 = u64::MAX;

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

    /// What a task copies: per copy, the key it is for if any, and the runs
    /// it reads and writes. A transfer's copies read and write the same
    /// runs, in the memory of the owner's group and of the target group.
    pub fn work(
        &self,
        task: (QueueId, RequestId),
    ) -> Vec<(
        Option<Key>,
        Vec<storage::runtime::Run>,
        Vec<storage::runtime::Run>,
    )> {
        let copies = self.copies.get(&task).map_or(&[][..], Vec::as_slice);
        let stated = |copy: &Copy| {
            let key = match copy.to {
                Target::Entry { key, .. } => Some(key),
                _ => None,
            };
            // A transfer's last copy also holds the blocks it does not copy.
            let reads = copy
                .reads
                .first()
                .map_or_else(Vec::new, |block| block.stated().collect());
            (key, reads, copy.write.stated().collect())
        };
        copies.iter().map(stated).collect()
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
        // R1: an owner that a copy writes for is not transferred.
        if self.copied(client, owner) {
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
        let slot = Rc::new(Block {
            class: SLOT,
            runs: vec![row],
            valid: Cell::new(EVERYWHERE),
        });
        let to = Target::Group { owner, to };
        copies.push(Copy {
            memory: 0,
            reads: kept,
            write: slot,
            to,
            fill: false,
            deadline,
        });

        self.copies.insert((queue, request), copies);
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
                        .map(|run| (class as u8, *position, run, !frozen)),
                );
            }
        }
        found.group = to;
        let row = slot.runs.first().map_or(0, super::Run::first);
        let old = std::mem::replace(&mut found.slot, slot);
        self.reclaim(old);
        Completion::Transferred { slot: row, runs }
    }

    /// Whether a prefetch in flight brings the key.
    fn fetching(&self, key: &Key) -> bool {
        self.copies.values().flatten().any(|copy| {
            copy.deadline.is_some()
                && copy.reads.is_empty()
                && matches!(copy.to, Target::Entry { key: other, remote: false } if other == *key)
        })
    }

    /// `Prefetch`: a task brings into new host blocks the leading keys that
    /// are not on the host. A key that is there already counts as found.
    pub(super) fn prefetch(
        &mut self,
        queue: QueueId,
        id: RequestId,
        class: u8,
        units: u32,
        keys: &[Key],
        deadline: u64,
    ) -> Taken {
        let (region, fixed) = self.class(queue.0, class)?.host.ok_or(Cause::Class)?;
        let size = match (fixed, units) {
            (0, stated) if stated > 0 => stated,
            (fixed, 0) if fixed > 0 => fixed,
            _ => return Err(Cause::Size),
        };

        let (mut found, mut copies) = (Vec::new(), Vec::new());
        for key in keys {
            match self.cache.get(key) {
                Some(entry) if entry.class != class => break,
                Some(Entry {
                    host: Some((_, maker)),
                    ..
                }) => {
                    if !copies.is_empty() || !self.visible(*maker, queue.0) {
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
            let write = Rc::new(Block {
                class,
                runs: vec![run],
                valid: Cell::new(EVERYWHERE),
            });
            let to = Target::Entry {
                key: *key,
                remote: false,
            };
            copies.push(Copy {
                memory: 0,
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

        self.copies.insert((queue, id), copies);
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
                host: Some((host, None)),
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

        self.copies.insert((queue, id), copies);
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
            let done = match self.cache.get_mut(&key) {
                // The remote store holds the content of the key's host copy.
                Some(entry) if remote => {
                    let same = entry
                        .host
                        .as_ref()
                        .is_some_and(|(host, _)| Rc::ptr_eq(host, &copy.write));
                    entry.remote |= same;
                    same
                }
                None if remote => false,
                // What the remote store and the peers hold is what every epoch may use.
                Some(entry) if entry.host.is_none() && entry.class == class => {
                    entry.host = Some((Rc::clone(&copy.write), None));
                    true
                }
                // R4: the key gained a host copy meanwhile and keeps it.
                Some(entry) => {
                    let maker = entry.host.as_ref().and_then(|(_, maker)| *maker);
                    entry.class == class && self.visible(maker, task.0.0)
                }
                None => {
                    let host = Some((Rc::clone(&copy.write), None));
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
        for block in copy.reads {
            self.reclaim(block);
        }
        self.reclaim(copy.write);

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
    }

    /// A5: a task is over and touches its blocks no more. R9: what it held
    /// is released now. `ok` is whether a transfer copied everything; the
    /// keys of a prefetch or a put that were not reported done are not.
    pub fn ended(&mut self, task: (QueueId, RequestId), ok: bool) {
        let Some(copies) = self.copies.remove(&task) else {
            return;
        };
        let answer = self.answers.remove(&task);
        let completion = match copies.first().map(|copy| &copy.to) {
            Some(Target::Group { owner, to }) => {
                let completion = self.transferred(task.0.0, *owner, *to, &copies, ok);
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
            for block in copy.reads {
                self.reclaim(block);
            }
            self.reclaim(copy.write);
        }
        if let Some(completion) = completion {
            self.complete(task.0, task.1, completion);
        }
        self.sweep();
        self.settle();
    }

    /// Tells the buffer pool the time. A task past its deadline is told to
    /// stop; a prefetch and a put are answered at once with what they
    /// brought, a transfer when its task has ended. R9: the blocks a task
    /// holds stay held until its end.
    pub fn tick(&mut self, now: u64) {
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
    }
}
