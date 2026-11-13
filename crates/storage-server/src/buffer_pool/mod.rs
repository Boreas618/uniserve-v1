//! The buffer pool: the blocks of one deployment on one host, with their
//! holders.
//!
//! The buffer pool is the data plane for every memory a request holds. It
//! holds tables and performs what the schedulers ask on their queue pairs
//! ([`storage::runtime`]); it makes no choice a policy could make
//! differently. It is one value changed through `&mut self` on one thread,
//! so a handler is one atomic step, and what is not atomic is three steps:
//! taken, due, finished ([`Deferred`], [`Copy`]).
//!
//! A block has no state field. What a block is follows from who holds it
//! ([`BufferPool::observed`]): nobody (its runs are free), an [`Entry`] of
//! the cache with any number of pins and copies reading it (shared), an
//! owner's private entry, a waiting commit or a copy writing it (private),
//! or only deferred releases and ended copies (retired). The holders share
//! the block by `Rc`, and [`BufferPool::reclaim`] is the only code that
//! returns runs to a region: it needs the last reference.
//!
//! The rules R1 to R11 of the design are guards of this module; each is
//! marked where it stands. The design model is `verification/buffer-pool`.
//!
//! The types that hold blocks take the reference to a block as a parameter,
//! a shared `Rc<Block>` by default: the pool's whole state also travels as
//! an image ([`image`]), in which the same types name each block by its
//! index in one list of blocks, and from which a pool is made again.
//!
//! The capacity words cost no scan: each region keeps the counts they are
//! made of, and each block what it is counted as. A handler that changes a
//! block's holders notes the block ([`BufferPool::touch`]), and the block
//! is counted again from its holders once the handler is done
//! ([`BufferPool::settle_counts`]), so a turn costs the work of its own
//! requests. [`BufferPool::check`] compares the counts with a scan.

mod copies;
mod image;
mod peers;
mod region;
mod sessions;
mod state;
mod tasks;
pub mod trace;

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::rc::{Rc, Weak};

use serde::{Deserialize, Serialize};
use storage::protocol::Key;
use storage::runtime::{Cause, Completion, GroupId, OwnerId, Request, RequestId, Segments, Stamps};

use region::{Region, Run};
/// A kind of block, as a session states it in its layout; the pool keeps
/// its host region by the pool's number of that region.
pub use storage::runtime::Class;

/// A session: an engine epoch's.
pub type ClientId = u32;

/// A queue pair: its session and its scheduler's index in the session.
pub type QueueId = (ClientId, u16);

/// What a handler answers: a completion now, none until deferred work or a
/// task produces it, or the cause of a refusal (R1).
type Taken = Result<Option<Completion>, Cause>;

/// The class of a slot row's block.
const SLOT: u8 = u8::MAX;

/// The validity of a slot row: memory of the ranks' own processes, which
/// every memory that maps its region reads.
const EVERYWHERE: u64 = u64::MAX;

/// The most memories a deployment's sessions state, device memories, the
/// memories of the main processes that hold product buffers, and host parts
/// together: one bit each in a block's validity.
const MEMORIES: usize = 64;

/// What a block's units count as in the capacity words of their regions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Counted {
    /// Private, held as a shared block by more than its entry, or free.
    #[default]
    Nothing,
    /// Shared under the key and held by nothing but its entry:
    /// `evictable`, and `available` to the queue pairs whose order names
    /// the key.
    Evictable(Key),
    /// Retired: `retiring`.
    Retired,
}

/// A block's memory: one run for a KV block, an encoder output or a
/// trajectory; one run per backing rank of a product, and one more in each
/// memory a `Replicate` brought it to.
#[derive(Serialize, Deserialize)]
struct Block {
    class: u8,
    /// The runs, the first the one it was allocated with: a cached block is
    /// found in its entry by the region of its first run. A `Replicate`
    /// appends a run in a receiving region while others hold the block.
    runs: RefCell<Vec<Run>>,
    /// The memories that hold the block's content, a bit each: device
    /// memories and the memories of main processes for a device block, from
    /// its allocation, changed by a transfer, by a load that fills one more
    /// memory and by a `Replicate`; host parts for a host block, from the
    /// end of the copy that wrote it, grown by a store that fills the parts
    /// it lacks.
    valid: Cell<u64>,
    /// The key it is cached under, while an entry of the cache holds it.
    /// What follows is the counting's: an image leaves it out, and the pool
    /// made of the image finds it again from the holders.
    #[serde(skip)]
    cached: Cell<Option<Key>>,
    /// The references that make it private (§8): an owner's private entry
    /// or slot, a waiting commit, a copy writing it for a target that
    /// exists.
    #[serde(skip)]
    owned: Cell<u32>,
    /// What its units are counted as now.
    #[serde(skip)]
    counted: Cell<Counted>,
}

impl Block {
    /// A block of `runs` of a class, valid in the memories `valid`, with
    /// `owned` references that make it private.
    fn new(class: u8, runs: Vec<Run>, valid: u64, owned: u32) -> Self {
        Self {
            class,
            runs: RefCell::new(runs),
            valid: Cell::new(valid),
            cached: Cell::new(None),
            owned: Cell::new(owned),
            counted: Cell::new(Counted::Nothing),
        }
    }

    /// One more reference makes it private.
    fn own(&self) {
        self.owned.set(self.owned.get() + 1);
    }

    /// A reference that made it private is gone or no longer does.
    fn disown(&self) {
        self.owned.set(self.owned.get().saturating_sub(1));
    }

    /// The region of the block's first run: the one its entry finds it by
    /// when it is cached.
    fn region(&self) -> u8 {
        self.runs.borrow().first().map_or(0, Run::region)
    }

    fn units(&self) -> u32 {
        self.runs.borrow().iter().map(Run::count).sum()
    }

    /// The block's runs as a completion states them, in the order it holds them.
    fn stated(&self) -> Vec<storage::runtime::Run> {
        self.runs.borrow().iter().map(Run::stated).collect()
    }

    /// The regions of the block's runs, in the order it holds them.
    fn regions(&self) -> Vec<u8> {
        self.runs.borrow().iter().map(Run::region).collect()
    }
}

/// One entry of an owner's table.
#[derive(Serialize, Deserialize)]
enum Held<B = Rc<Block>> {
    Private(B),
    Shared(B),
}

impl Held {
    fn block(&self) -> &Rc<Block> {
        match self {
            Self::Private(block) | Self::Shared(block) => block,
        }
    }

    fn into_block(self) -> Rc<Block> {
        match self {
            Self::Private(block) | Self::Shared(block) => block,
        }
    }

    /// The block of an entry that leaves its table: a private entry no
    /// longer makes it private.
    fn released(self) -> Rc<Block> {
        if let Self::Private(block) = &self {
            block.disown();
        }
        self.into_block()
    }
}

/// The blocks of an owner that goes, its slot first: its slot and its
/// private entries no longer make them private.
fn disowned(owner: Owner) -> Vec<Rc<Block>> {
    owner.slot.disown();
    let mut blocks = vec![owner.slot];
    for table in owner.tables {
        blocks.extend(table.into_values().map(Held::released));
    }
    blocks
}

/// A request's attachment to one worker group.
#[derive(Serialize, Deserialize)]
struct Owner<B = Rc<Block>> {
    group: GroupId,
    slot: B,
    /// Per class, by position.
    tables: Vec<BTreeMap<u32, Held<B>>>,
    /// Whether its transfer waits or runs: it then takes only a detach.
    moving: bool,
}

/// The encoding a host copy is written in: the digests of the tensors and of
/// the layout of the class that wrote it (`Class::tensors`,
/// `Class::encoding`).
type Encoding = ([u8; 32], [u8; 32]);

/// A key's content, per tier.
#[derive(Serialize, Deserialize)]
struct Entry<B = Rc<Block>> {
    class: u8,
    /// One per region that holds a device copy: the block and the epoch on
    /// trial that made it, to which alone it is visible until its update
    /// is committed.
    device: Vec<(B, Option<u64>)>,
    /// The host copy, with the epoch on trial that made it and the encoding
    /// it is written in. A device copy needs none: a region is memory of
    /// one layout, which every session that names it shares.
    host: Option<(B, Option<u64>, Encoding)>,
    /// 64, or fewer for a request's last block.
    tokens: u16,
    /// The remote store acknowledged a copy.
    remote: bool,
}

impl<B> Entry<B> {
    /// Every copy of the key, on the device and then on the host, with the
    /// epoch on trial that made it.
    fn copies(&self) -> impl Iterator<Item = (&B, &Option<u64>)> {
        let device = self.device.iter().map(|(block, maker)| (block, maker));
        device.chain(self.host.iter().map(|(block, maker, _)| (block, maker)))
    }
}

impl Entry {
    /// The key's copy in a region, on the device or on the host.
    fn copy_in(&self, region: u8) -> Option<&Rc<Block>> {
        self.copies()
            .map(|(block, _)| block)
            .find(|block| block.region() == region)
    }
}

/// Work that waits until a worker group's `completed` reaches a stamp. The
/// stamps of the other groups it still waits for are `rest`.
#[derive(Serialize, Deserialize)]
enum Deferred<B = Rc<Block>> {
    Release {
        blocks: Vec<B>,
    },
    Commit {
        queue: QueueId,
        #[serde(with = "image::request_id")]
        request: RequestId,
        #[serde(with = "image::owner_id")]
        owner: OwnerId,
        class: u8,
        at: u32,
        block: B,
        key: Key,
        tokens: u16,
        rest: Stamps,
    },
    Transfer {
        queue: QueueId,
        #[serde(with = "image::request_id")]
        request: RequestId,
        #[serde(with = "image::owner_id")]
        owner: OwnerId,
        to: GroupId,
        rest: Stamps,
    },
    Finish {
        queue: QueueId,
        #[serde(with = "image::request_id")]
        request: RequestId,
        #[serde(with = "image::request_id")]
        copy: RequestId,
        ok: bool,
        rest: Stamps,
    },
    /// A crossing waits for the stamps of the batches that may still write
    /// its block, holding the block, and then while something else fills a
    /// memory it is to fill (at stamp 0 of the reading group).
    Replicate {
        queue: QueueId,
        #[serde(with = "image::request_id")]
        request: RequestId,
        #[serde(with = "image::owner_id")]
        owner: OwnerId,
        class: u8,
        at: u32,
        block: B,
        from: u8,
        to: GroupId,
        rest: Stamps,
    },
}

/// One worker group of one engine epoch.
#[derive(Serialize, Deserialize)]
struct Group<B = Rc<Block>> {
    /// The device memory its ranks address, as the number of its bit in a
    /// block's validity.
    memory: u8,
    /// The region of its slot rows.
    slots: u8,
    /// Per class, the device regions its owners' blocks are taken from: one,
    /// or one per backing rank of a product; none for a class it lacks.
    regions: Vec<Vec<u8>>,
    /// Per class, the regions a `Replicate` into its memories takes runs in:
    /// one in each memory its ranks read the class in and none of its
    /// `regions` lies in.
    receives: Vec<Vec<u8>>,
    /// Per class, the host parts its ranks copy with, as bits of a block's
    /// validity; 0 for a class without a host region.
    parts: Vec<u64>,
    /// The last batch whose end its workers have seen, as last read.
    completed: u64,
    /// What waits on it, in order of stamp.
    deferred: VecDeque<(u64, Deferred<B>)>,
}

impl Group {
    /// A group whose ranks address the memory of bit `memory`, with the
    /// region of its slot rows and per class its device regions, no
    /// receiving region, no host part, and nothing waiting on it.
    fn new(memory: u8, slots: u8, regions: Vec<Vec<u8>>) -> Self {
        Self {
            memory,
            slots,
            regions,
            receives: Vec::new(),
            parts: Vec::new(),
            completed: 0,
            deferred: VecDeque::new(),
        }
    }
}

/// Where the result of a copy goes.
#[derive(Clone, Copy, Serialize, Deserialize)]
enum Target {
    /// An entry of an owner's table: a private clone, or with a key a block
    /// that is cached and pinned for the owner.
    Owner {
        #[serde(with = "image::owner_id")]
        owner: OwnerId,
        class: u8,
        at: u32,
        key: Option<Key>,
    },
    /// A cache entry's copy on the host, or in the remote store.
    Entry { key: Key, remote: bool },
    /// The owner's slot, restored from a stored trajectory.
    Slot {
        #[serde(with = "image::owner_id")]
        owner: OwnerId,
    },
    /// The memory of the group an owner is transferred to.
    Group {
        #[serde(with = "image::owner_id")]
        owner: OwnerId,
        to: GroupId,
    },
    /// A crossing: the block at an owner's position, read from its run in
    /// region `from`, in one memory the group reads the class in.
    Memory {
        #[serde(with = "image::owner_id")]
        owner: OwnerId,
        class: u8,
        at: u32,
        group: GroupId,
        from: u8,
    },
}

/// A copy in flight, run by a worker, the Transfer Server or the
/// connector. It holds what it reads and what it writes until it has ended.
#[derive(Serialize, Deserialize)]
struct Copy<B = Rc<Block>> {
    /// The memories it writes, as bits of a block's validity: the device
    /// memory a clone, a load, a transfer or a crossing writes; the host
    /// parts a store, a save or a prefetch writes; 0 for a restore, which
    /// writes the slot row, and a put.
    memory: u64,
    reads: Vec<B>,
    /// The block it writes. A put to the remote store writes no block: this
    /// is the host block it reads.
    write: B,
    to: Target,
    /// Whether it fills one more memory of a block that others may read.
    fill: bool,
    /// A task's deadline, in nanoseconds of the host's monotonic clock; none
    /// for a copy a worker runs, which its scheduler gives up.
    deadline: Option<u64>,
}

/// The buffer pool of one deployment on one host.
pub struct BufferPool {
    /// Per session, its classes and which of them holds stored trajectories.
    classes: HashMap<ClientId, (Vec<Class>, u8)>,
    regions: Vec<Region>,
    /// The names of the memory spaces the sessions stated, by number.
    spaces: Vec<String>,
    /// The memories the sessions stated, by their bit in a block's
    /// validity: a device memory or a main process's memory by its name, a
    /// host part by its host and its part index; at most [`MEMORIES`].
    memories: Vec<(String, Option<u32>)>,
    /// Per memory space, the byte ranges in no region: the deployment's free quota there.
    quota: Vec<BTreeMap<u64, u64>>,
    /// Per memory space, its segment table: the segments the engines'
    /// ranks created and the head granted, which back its units.
    segments: Vec<Segments>,
    cache: HashMap<Key, Entry>,
    owners: HashMap<(ClientId, OwnerId), Owner>,
    groups: HashMap<(ClientId, GroupId), Group>,
    /// By the request that started each.
    copies: HashMap<(QueueId, RequestId), Vec<Copy>>,
    /// The tasks whose request is not answered yet, with the keys done so far.
    answers: HashMap<(QueueId, RequestId), Vec<Key>>,
    /// The epoch of each session.
    sessions: HashMap<ClientId, u64>,
    /// Per epoch, whether it is on trial; an epoch whose update was rolled back is absent.
    trial: HashMap<u64, bool>,
    /// Per queue pair, the highest owner number and the highest request number taken.
    numbers: HashMap<QueueId, (u64, u64)>,
    /// How long a transfer and a put to the remote store may take, in nanoseconds.
    limits: (u64, u64),
    /// The host's monotonic clock as last told, in nanoseconds.
    now: u64,
    /// The completions produced and not yet written to their queue pairs.
    completions: Vec<(QueueId, RequestId, Completion)>,
    /// The tasks to start and the tasks to stop, not yet handed to the
    /// Transfer Server or the connector.
    started: Vec<(QueueId, RequestId)>,
    stopped: Vec<(QueueId, RequestId)>,
    /// The host copies each peer, another host's Storage Server, holds for
    /// its reads, by the peer's connection: holders like any other.
    peers: HashMap<u64, Vec<Rc<Block>>>,
    /// The blocks whose holders the handler running now changed, counted
    /// again when it is done; empty between handlers.
    touched: RefCell<Vec<Weak<Block>>>,
}

impl BufferPool {
    /// A buffer pool without regions or sessions. `limits` is how long a
    /// transfer and a put to the remote store may take, in nanoseconds.
    pub fn new(limits: (u64, u64)) -> Self {
        Self {
            classes: HashMap::new(),
            regions: Vec::new(),
            spaces: Vec::new(),
            memories: Vec::new(),
            quota: Vec::new(),
            segments: Vec::new(),
            cache: HashMap::new(),
            owners: HashMap::new(),
            groups: HashMap::new(),
            copies: HashMap::new(),
            answers: HashMap::new(),
            sessions: HashMap::new(),
            trial: HashMap::new(),
            numbers: HashMap::new(),
            limits,
            now: 0,
            completions: Vec::new(),
            started: Vec::new(),
            stopped: Vec::new(),
            peers: HashMap::new(),
            touched: RefCell::new(Vec::new()),
        }
    }

    /// How long a transfer and a put to the remote store may take from now
    /// on, in nanoseconds; the tasks under way keep their deadlines.
    pub fn set_limits(&mut self, limits: (u64, u64)) {
        self.limits = limits;
    }

    /// Whether deferred work waits for a group's `completed` word: the
    /// buffer pool's thread then keeps reading the words.
    pub fn deferred(&self) -> bool {
        self.groups.values().any(|group| !group.deferred.is_empty())
    }

    /// The completions produced since the last call, for their queue pairs.
    pub fn completions(&mut self) -> Vec<(QueueId, RequestId, Completion)> {
        std::mem::take(&mut self.completions)
    }

    fn complete(&mut self, queue: QueueId, request: RequestId, completion: Completion) {
        self.completions.push((queue, request, completion));
    }

    /// R2: runs return to a region only here, and only with the last
    /// reference to the block; otherwise the block lives on with its other
    /// holders.
    fn reclaim(&mut self, block: Rc<Block>) {
        // The last reference leaves its counts with it; otherwise the block
        // is counted again from the holders it keeps.
        if Rc::strong_count(&block) == 1 {
            self.uncount(&block);
        } else {
            self.touch(&block);
        }
        if let Ok(block) = Rc::try_unwrap(block) {
            for run in block.runs.into_inner() {
                let region = &mut self.regions[usize::from(run.region())];
                if block.class == SLOT {
                    region.give_row(run);
                } else {
                    region.give(run);
                }
            }
        }
    }

    /// Takes one request of a queue pair and produces its completion: now,
    /// when its stamps are reached, or when its task ends.
    pub fn take(&mut self, queue: QueueId, id: RequestId, request: Request) {
        let result = self.numbered(queue, id).and_then(|()| match request {
            Request::Attach { owner, group } => self.attach(queue, owner, group),
            Request::Detach { owner, after } => self.detach(queue, owner, &after),
            Request::Allocate {
                owner,
                class,
                at,
                count,
                units,
            } => self.allocate(queue, owner, class, at, count, units),
            Request::Pin {
                owner,
                class,
                at,
                keys,
            } => self.pin(queue, owner, class, at, &keys),
            Request::Release {
                owner,
                class,
                at,
                count,
                after,
            } => self.release(queue, owner, class, at, count, &after),
            Request::Commit {
                owner,
                class,
                at,
                key,
                tokens,
                after,
            } => {
                let commit = Deferred::Commit {
                    queue,
                    request: id,
                    owner,
                    class,
                    at,
                    block: self.committed(queue, owner, class, at, &after)?,
                    key,
                    tokens,
                    rest: Vec::new(),
                };
                self.defer(queue.0, after, commit);
                Ok(None)
            }
            Request::Evict { keys } => Ok(Some(self.evict(&keys))),
            Request::Victims { region, keys } => self.victims(queue, region, keys),
            Request::List => Ok(Some(self.list(queue.0))),
            Request::Resize { region, units } => self.resize(queue, region, units),
            Request::Clone {
                owner,
                class,
                at,
                key,
            } => self.clone_block(queue, id, owner, class, at, key),
            Request::Load {
                owner,
                class,
                at,
                keys,
            } => self.load(queue, id, owner, class, at, &keys),
            Request::Store { group, keys, tier } => self.store(queue, id, group, &keys, tier),
            Request::Save { owner, key, units } => self.save(queue, id, owner, key, units),
            Request::Restore { owner, key } => self.restore(queue, id, owner, key),
            Request::Finish { copy, ok, after } => self.finish(queue, id, copy, ok, after),
            Request::Transfer {
                owner,
                group,
                after,
            } => self.transfer(queue, id, owner, group, after),
            Request::Prefetch {
                group,
                class,
                units,
                keys,
                source: _,
                deadline,
            } => self.prefetch(queue, id, group, class, units, &keys, deadline),
            Request::Replicate {
                owner,
                class,
                at,
                from,
                group,
                after,
            } => self.replicate(queue, id, (owner, class, at), from, group, after),
        });
        match result {
            Ok(Some(completion)) => self.complete(queue, id, completion),
            // The completion comes when the request's stamps are reached or its task ends.
            Ok(None) => {}
            // R1: a request that names what does not fit changes nothing.
            Err(cause) => self.complete(queue, id, Completion::Refused { cause }),
        }
        self.settle();
        self.settle_counts();
    }

    /// R1: a request's number is above every number its queue pair's
    /// requests had, and its queue pair is of an open session.
    fn numbered(&mut self, queue: QueueId, id: RequestId) -> Result<(), Cause> {
        if !self.sessions.contains_key(&queue.0) {
            return Err(Cause::Request);
        }
        let numbers = self.numbers.entry(queue).or_default();
        if id.0 <= numbers.1 {
            return Err(Cause::Request);
        }
        numbers.1 = id.0;
        Ok(())
    }

    /// The owner a request names, which is its session's (R1).
    fn owner(&self, client: ClientId, owner: OwnerId) -> Result<&Owner, Cause> {
        self.owners.get(&(client, owner)).ok_or(Cause::Owner)
    }

    /// As [`Self::owner`], for a request that an owner in transfer refuses.
    fn settled(&self, client: ClientId, owner: OwnerId) -> Result<&Owner, Cause> {
        let found = self.owner(client, owner)?;
        if found.moving {
            Err(Cause::Owner)
        } else {
            Ok(found)
        }
    }

    /// A class of a session (R1).
    fn class(&self, client: ClientId, class: u8) -> Result<&Class, Cause> {
        let (classes, _) = self.classes.get(&client).ok_or(Cause::Class)?;
        classes.get(usize::from(class)).ok_or(Cause::Class)
    }

    /// The copies of a session.
    fn copies_of(&self, client: ClientId) -> impl Iterator<Item = &Copy> {
        self.copies
            .iter()
            .filter(move |((queue, _), _)| queue.0 == client)
            .flat_map(|(_, copies)| copies)
    }

    /// Whether positions `at .. at + count` of the owner's table of a class
    /// are free: they hold no entry, and no copy writes for them.
    fn free_positions(
        &self,
        client: ClientId,
        owner: OwnerId,
        class: u8,
        at: u32,
        count: u32,
    ) -> bool {
        let Some(end) = at.checked_add(count) else {
            return false;
        };
        let Some(found) = self.owners.get(&(client, owner)) else {
            return false;
        };
        let copied = self.copies_of(client).any(|copy| {
            matches!(copy.to, Target::Owner { owner: target, class: of, at: position, .. }
                if target == owner && of == class && (at..end).contains(&position))
        });
        found.tables[usize::from(class)]
            .range(at..end)
            .next()
            .is_none()
            && !copied
    }

    /// Whether a copy made by `maker`, an epoch on trial or none, may be used
    /// by the session (R8).
    fn visible(&self, maker: Option<u64>, client: ClientId) -> bool {
        maker.is_none_or(|maker| self.sessions.get(&client) == Some(&maker))
    }

    /// The visibility of a copy that the session's epoch makes: its own while
    /// it is on trial.
    fn made_by(&self, client: ClientId) -> Option<u64> {
        let epoch = self.sessions.get(&client)?;
        (self.trial.get(epoch) != Some(&false)).then_some(*epoch)
    }

    /// A group's bit in a block's validity.
    fn memory(&self, client: ClientId, group: GroupId) -> u64 {
        self.groups
            .get(&(client, group))
            .map_or(0, |found| 1 << found.memory)
    }

    /// The memory in which a block's run in `region` lies, for a group whose
    /// memory is `own`: the region's memory when its units exist once, the
    /// group's when they exist in every memory that maps the region.
    fn lies_in(&self, region: u8, own: u64) -> u64 {
        let found = self.regions.get(usize::from(region));
        match found.and_then(|region| region.memory) {
            Some(memory) => 1 << memory,
            None => own,
        }
    }

    /// The memories in which runs in `regions` lie, for a group whose
    /// memory is `own`.
    fn spread(&self, regions: &[u8], own: u64) -> u64 {
        regions
            .iter()
            .fold(0, |all, region| all | self.lies_in(*region, own))
    }

    /// The host parts a group copies a class with.
    fn parts(&self, client: ClientId, group: GroupId, class: u8) -> u64 {
        let found = self.groups.get(&(client, group));
        let parts = found.and_then(|found| found.parts.get(usize::from(class)).copied());
        parts.unwrap_or(0)
    }

    /// R11: whether a group's owners may load or restore a host copy: it
    /// holds every host part the group copies the class with.
    fn loadable(&self, client: ClientId, group: GroupId, class: u8, host: &Block) -> bool {
        let parts = self.parts(client, group, class);
        host.valid.get() & parts == parts
    }

    /// The one device region of a group for a class whose blocks have one run.
    fn region(&self, client: ClientId, group: GroupId, class: u8) -> Result<u8, Cause> {
        let found = self.groups.get(&(client, group)).ok_or(Cause::Group)?;
        match found.regions.get(usize::from(class)).map(Vec::as_slice) {
            Some([region]) => Ok(*region),
            _ => Err(Cause::Class),
        }
    }

    fn attach(&mut self, queue: QueueId, owner: OwnerId, group: GroupId) -> Taken {
        let client = queue.0;
        // R1: an owner number is taken once on its queue pair, and once in its session.
        let taken = self.numbers.get(&queue).map_or(0, |numbers| numbers.0);
        if owner.0 <= taken || self.owners.contains_key(&(client, owner)) {
            return Err(Cause::Owner);
        }
        let slots = self.groups.get(&(client, group)).ok_or(Cause::Group)?.slots;
        let Some(run) = self.regions[usize::from(slots)].take_row() else {
            return Ok(Some(Completion::Full { region: slots }));
        };

        let slot = run.first();
        let classes = self
            .classes
            .get(&client)
            .map_or(0, |(classes, _)| classes.len());
        let tables = (0..classes).map(|_| BTreeMap::new()).collect();
        // The owner's slot makes the row private.
        let block = Rc::new(Block::new(SLOT, vec![run], EVERYWHERE, 1));
        self.owners.insert(
            (client, owner),
            Owner {
                group,
                slot: block,
                tables,
                moving: false,
            },
        );
        self.numbers.entry(queue).or_default().0 = owner.0;
        Ok(Some(Completion::Attached { slot }))
    }

    /// R1: every stamp names a group of the request's session.
    fn stamped(&self, client: ClientId, after: &Stamps) -> Result<(), Cause> {
        let known = after
            .iter()
            .all(|(group, _)| self.groups.contains_key(&(client, *group)));
        if known { Ok(()) } else { Err(Cause::Group) }
    }

    /// Holds `blocks` under every stamp of `after` that is not reached, then
    /// lets go of them: a block is reclaimed by the last reference (R3).
    fn retire(&mut self, client: ClientId, blocks: Vec<Rc<Block>>, after: &Stamps) {
        for (group, stamp) in after {
            if let Some(found) = self.groups.get_mut(&(client, *group))
                && found.completed < *stamp
            {
                let at = found.deferred.partition_point(|(other, _)| other <= stamp);
                found.deferred.insert(
                    at,
                    (
                        *stamp,
                        Deferred::Release {
                            blocks: blocks.clone(),
                        },
                    ),
                );
            }
        }
        for block in blocks {
            self.reclaim(block);
        }
    }

    fn detach(&mut self, queue: QueueId, owner: OwnerId, after: &Stamps) -> Taken {
        self.owner(queue.0, owner)?;
        self.stamped(queue.0, after)?;
        self.orphan(queue.0, owner);
        let found = self.owners.remove(&(queue.0, owner)).ok_or(Cause::Owner)?;

        // R6: a commit of this owner that waits holds its block and still
        // takes effect. R9: so does a transfer's task hold what it copies.
        let blocks = disowned(found);
        self.retire(queue.0, blocks, after);
        Ok(Some(Completion::Done))
    }

    /// The copies that write for an owner about to go write for a target
    /// that is gone: what they write is no longer private by them.
    fn orphan(&self, client: ClientId, owner: OwnerId) {
        for copy in self.copies_of(client) {
            let named = matches!(copy.to, Target::Owner { owner: target, .. }
                | Target::Slot { owner: target }
                | Target::Group { owner: target, .. } if target == owner);
            if named && self.live(client, copy) {
                copy.write.disown();
                self.touch(&copy.write);
            }
        }
    }

    /// The keys whose copies in `region` are to be evicted, in the order the
    /// queue pair installed there, so that `blocks` runs of `size` units are
    /// free; none when the whole order would not make them fit. Only a
    /// shared block without a pin (R7) and with no other holder (R2) is taken.
    fn room(&self, queue: QueueId, region: u8, blocks: u32, size: u32) -> Option<Vec<Key>> {
        let found = &self.regions[usize::from(region)];
        if size == 0 {
            return None;
        }
        // The free runs by first unit, neighbours joined, and the blocks of `size` they hold.
        let mut runs: BTreeMap<u32, u32> = found.free_runs().collect();
        let mut holds: u64 = runs.values().map(|count| u64::from(count / size)).sum();
        let mut keys = Vec::new();
        let mut seen = HashSet::new();

        for key in found.orders.get(&queue).into_iter().flatten() {
            if holds >= u64::from(blocks) {
                break;
            }
            let Some(block) = self.cache.get(key).and_then(|entry| entry.copy_in(region)) else {
                continue;
            };
            if Rc::strong_count(block) != 1 || !seen.insert(*key) {
                continue;
            }
            // A block holds runs in other regions too once a `Replicate`
            // brought it to another memory: only its runs here make room.
            let held = block.runs.borrow();
            for run in held.iter().filter(|run| run.region() == region) {
                // Join the run with the free runs that touch it.
                let (mut first, mut count) = (run.first(), run.count());
                if let Some(next) = runs.remove(&(first + count)) {
                    holds -= u64::from(next / size);
                    count += next;
                }
                let before = runs
                    .range(..first)
                    .next_back()
                    .map(|(start, length)| (*start, *length));
                if let Some((start, length)) = before
                    && start + length == first
                {
                    runs.remove(&start);
                    holds -= u64::from(length / size);
                    first = start;
                    count += length;
                }
                runs.insert(first, count);
                holds += u64::from(count / size);
            }
            keys.push(*key);
        }
        (holds >= u64::from(blocks)).then_some(keys)
    }

    /// Removes a key's copy in a region from the cache when nothing else
    /// holds it, and reclaims it. Returns whether it left.
    fn evict_in(&mut self, key: &Key, region: u8) -> bool {
        let Some(entry) = self.cache.get_mut(key) else {
            return false;
        };
        // R7: a pin is a holder; R2: so is deferred work and a copy.
        let unheld = |block: &Rc<Block>| block.region() == region && Rc::strong_count(block) == 1;
        let left = if let Some(at) = entry.device.iter().position(|(block, _)| unheld(block)) {
            entry.device.swap_remove(at).0
        } else if entry.host.as_ref().is_some_and(|(block, ..)| unheld(block)) {
            match entry.host.take() {
                Some((block, ..)) => block,
                None => return false,
            }
        } else {
            return false;
        };
        if entry.device.is_empty() && entry.host.is_none() && !entry.remote {
            self.cache.remove(key);
        }
        left.cached.set(None);
        self.reclaim(left);
        true
    }

    /// Takes `count` blocks of `size` units from a region, evicting `keys`
    /// first: what [`Self::room`] found to make them fit.
    fn obtain(
        &mut self,
        region: u8,
        class: u8,
        count: u32,
        size: u32,
        keys: &[Key],
        valid: u64,
    ) -> Vec<Rc<Block>> {
        for key in keys {
            self.evict_in(key, region);
        }
        let mut blocks = Vec::new();
        for _ in 0..count {
            let Some(run) = self.regions[usize::from(region)].take(size) else {
                break;
            };
            // The copy that writes it makes it private ([`Self::hold_copies`]).
            blocks.push(Rc::new(Block::new(class, vec![run], valid, 0)));
        }
        blocks
    }

    fn allocate(
        &mut self,
        queue: QueueId,
        owner: OwnerId,
        class: u8,
        at: u32,
        count: u32,
        units: u32,
    ) -> Taken {
        let client = queue.0;
        let group = self.settled(client, owner)?.group;
        // R1: a sized class states its size, and no other does.
        let size = match (self.class(client, class)?.units, units) {
            (0, stated) if stated > 0 => stated,
            (fixed, 0) if fixed > 0 => fixed,
            _ => return Err(Cause::Size),
        };
        if count == 0 || !self.free_positions(client, owner, class, at, count) {
            return Err(Cause::Position);
        }
        let regions = self.groups[&(client, group)]
            .regions
            .get(usize::from(class))
            .cloned()
            .unwrap_or_default();
        if regions.is_empty() {
            return Err(Cause::Class);
        }

        // All or none: every region must have room before any is touched.
        let mut evictions = Vec::new();
        for region in &regions {
            match self.room(queue, *region, count, size) {
                Some(keys) => evictions.push(keys),
                None => return Ok(Some(Completion::Full { region: *region })),
            }
        }

        // A block is valid where its owner's group writes it: in the one
        // memory its runs lie in, a group's for a region each memory holds.
        // A block whose runs lie in several memories, a product backed by
        // ranks of several processes, is valid in none until a `Replicate`
        // names the run its writer wrote: which process's runs a dealt call
        // wrote is the scheduler's knowledge (A9).
        let spread = self.spread(&regions, self.memory(client, group));
        let valid = if spread.count_ones() == 1 { spread } else { 0 };
        // Each is private by the owner's entry.
        let mut blocks: Vec<Block> = (0..count)
            .map(|_| Block::new(class, Vec::new(), valid, 1))
            .collect();
        for (region, keys) in regions.iter().zip(&evictions) {
            for key in keys {
                self.evict_in(key, *region);
            }
            for block in &mut blocks {
                block
                    .runs
                    .get_mut()
                    .extend(self.regions[usize::from(*region)].take(size));
            }
        }

        let runs = blocks.iter().flat_map(Block::stated).collect();
        if let Some(found) = self.owners.get_mut(&(client, owner)) {
            for (position, block) in (at..).zip(blocks) {
                found.tables[usize::from(class)].insert(position, Held::Private(Rc::new(block)));
            }
        }
        Ok(Some(Completion::Allocated {
            runs,
            evicted: evictions.concat(),
        }))
    }

    /// The device copy of a key that a group of a session may use: of the
    /// class, in the group's region, valid in the memory the group reads it
    /// in (its own for KV, its region's for a product) and visible to its
    /// epoch (R8).
    fn usable(&self, client: ClientId, group: GroupId, class: u8, key: &Key) -> Option<&Rc<Block>> {
        let region = self.region(client, group, class).ok()?;
        let memory = self.lies_in(region, self.memory(client, group));
        let entry = self.cache.get(key).filter(|entry| entry.class == class)?;
        let found = entry.device.iter().find(|(block, maker)| {
            block.region() == region
                && block.valid.get() & memory != 0
                && self.visible(*maker, client)
        });
        found.map(|(block, _)| block)
    }

    fn pin(&mut self, queue: QueueId, owner: OwnerId, class: u8, at: u32, keys: &[Key]) -> Taken {
        let client = queue.0;
        let group = self.settled(client, owner)?.group;
        self.class(queue.0, class)?;
        let blocks: Vec<Rc<Block>> = keys
            .iter()
            .map_while(|key| self.usable(client, group, class, key).cloned())
            .collect();
        if !self.free_positions(client, owner, class, at, blocks.len() as u32) {
            return Err(Cause::Position);
        }

        // The further leading keys that have a host copy the epoch may use
        // and the owner's group can load (R11).
        let on_host = |key: &&Key| {
            let host = self.cache.get(*key).and_then(|entry| entry.host.as_ref());
            host.is_some_and(|(block, maker, _)| {
                self.visible(*maker, client) && self.loadable(client, group, class, block)
            })
        };
        let host = keys[blocks.len()..].iter().take_while(on_host).count() as u32;

        let runs = blocks.iter().flat_map(|block| block.stated()).collect();
        // A pin holds the block: it is no longer evictable.
        blocks.iter().for_each(|block| self.touch(block));
        if let Some(found) = self.owners.get_mut(&(client, owner)) {
            for (position, block) in (at..).zip(blocks) {
                found.tables[usize::from(class)].insert(position, Held::Shared(block));
            }
        }
        Ok(Some(Completion::Pinned { runs, host }))
    }

    fn release(
        &mut self,
        queue: QueueId,
        owner: OwnerId,
        class: u8,
        at: u32,
        count: u32,
        after: &Stamps,
    ) -> Taken {
        let client = queue.0;
        self.settled(client, owner)?;
        self.class(queue.0, class)?;
        self.stamped(client, after)?;
        let end = at.checked_add(count).ok_or(Cause::Position)?;
        let found = self.owners.get_mut(&(client, owner)).ok_or(Cause::Owner)?;
        let table = &mut found.tables[usize::from(class)];
        // R1: every position named holds an entry.
        if count == 0 || table.range(at..end).count() as u32 != count {
            return Err(Cause::Position);
        }

        let blocks = (at..end)
            .filter_map(|position| table.remove(&position))
            .map(Held::released)
            .collect();
        self.retire(client, blocks, after);
        Ok(Some(Completion::Done))
    }

    /// The private block a `Commit` names (R1): its owner's, at a position
    /// no commit waits for, with one run.
    fn committed(
        &self,
        queue: QueueId,
        owner: OwnerId,
        class: u8,
        at: u32,
        after: &Stamps,
    ) -> Result<Rc<Block>, Cause> {
        let found = self.settled(queue.0, owner)?;
        self.class(queue.0, class)?;
        self.stamped(queue.0, after)?;
        let Some(Held::Private(block)) = found.tables[usize::from(class)].get(&at) else {
            return Err(Cause::Position);
        };
        // A position whose commit waits is not committed again, whatever
        // block it holds by now.
        let waits = self.groups.values().flat_map(|group| &group.deferred).any(|(_, work)| {
            matches!(work, Deferred::Commit { queue: of, owner: other, class: kind, at: position, .. }
                if of.0 == queue.0 && *other == owner && *kind == class && *position == at)
        });
        if waits {
            return Err(Cause::Position);
        }
        if block.runs.borrow().len() != 1 {
            return Err(Cause::Class);
        }
        // The waiting commit makes the block private too, whatever its
        // owner does meanwhile (R6).
        block.own();
        Ok(Rc::clone(block))
    }

    /// The blocks that waiting commits hold.
    fn commits(&self) -> impl Iterator<Item = &Rc<Block>> {
        let deferred = self.groups.values().flat_map(|group| &group.deferred);
        deferred.filter_map(|(_, work)| match work {
            Deferred::Commit { block, .. } => Some(block),
            _ => None,
        })
    }

    /// R3: queues `work` under the first stamp of `after` that is not
    /// reached, with the stamps after it as its `rest`; when every stamp is
    /// reached the work takes effect.
    fn defer(&mut self, client: ClientId, mut after: Stamps, mut work: Deferred) {
        while let Some((group, stamp)) = after.pop() {
            if let Some(found) = self.groups.get_mut(&(client, group))
                && found.completed < stamp
            {
                match &mut work {
                    Deferred::Commit { rest, .. }
                    | Deferred::Transfer { rest, .. }
                    | Deferred::Finish { rest, .. }
                    | Deferred::Replicate { rest, .. } => {
                        *rest = after;
                    }
                    Deferred::Release { .. } => {}
                }
                let at = found.deferred.partition_point(|(other, _)| *other <= stamp);
                found.deferred.insert(at, (stamp, work));
                return;
            }
        }
        if let Some(waits) = self.effect(client, work) {
            self.wait(client, waits);
        }
    }

    /// Queues a transfer or a crossing that is due and waits for a copy to
    /// end, under stamp 0 of the group it copies to: it is looked at again
    /// whenever deferred work is done.
    fn wait(&mut self, client: ClientId, work: Deferred) {
        let (Deferred::Transfer { to, .. } | Deferred::Replicate { to, .. }) = &work else {
            return;
        };
        if let Some(found) = self.groups.get_mut(&(client, *to)) {
            found.deferred.push_front((0, work));
        }
    }

    /// Records what a group's workers wrote into its `completed` word.
    pub fn advance(&mut self, client: ClientId, group: GroupId, completed: u64) {
        if let Some(found) = self.groups.get_mut(&(client, group))
            && completed > found.completed
        {
            found.completed = completed;
            self.settle();
            self.settle_counts();
        }
    }

    /// Does the deferred work whose stamps are reached, until none is left
    /// that can go on: work that ends a copy lets a waiting transfer start.
    fn settle(&mut self) {
        loop {
            let mut due = Vec::new();
            for ((client, _), group) in &mut self.groups {
                while group
                    .deferred
                    .front()
                    .is_some_and(|(stamp, _)| *stamp <= group.completed)
                {
                    due.extend(group.deferred.pop_front().map(|(_, work)| (*client, work)));
                }
            }

            let mut progress = false;
            for (client, work) in due {
                let rest = match &work {
                    Deferred::Commit { rest, .. }
                    | Deferred::Transfer { rest, .. }
                    | Deferred::Finish { rest, .. }
                    | Deferred::Replicate { rest, .. } => rest.clone(),
                    Deferred::Release { .. } => Vec::new(),
                };
                if !rest.is_empty() {
                    // The stamp of one group is reached: on to the next.
                    self.defer(client, rest, work);
                    progress = true;
                } else if let Some(waits) = self.effect(client, work) {
                    self.wait(client, waits);
                } else {
                    progress = true;
                }
            }
            if !progress {
                break;
            }
        }
    }

    /// Deferred work takes effect: every stamp it waited for is reached.
    /// Returns a transfer or a crossing that cannot start yet.
    fn effect(&mut self, client: ClientId, work: Deferred) -> Option<Deferred> {
        match work {
            Deferred::Release { blocks } => {
                for block in blocks {
                    self.reclaim(block);
                }
            }
            Deferred::Commit {
                queue,
                request,
                owner,
                class,
                at,
                block,
                key,
                tokens,
                ..
            } => {
                let completion = self.share(client, owner, class, at, block, key, tokens);
                self.complete(queue, request, completion);
            }
            Deferred::Finish {
                queue,
                request,
                copy,
                ok,
                ..
            } => {
                let completion = self.finished(queue, copy, ok);
                self.complete(queue, request, completion);
            }
            Deferred::Transfer {
                queue,
                request,
                owner,
                to,
                ..
            } => {
                if !self.start(queue, request, owner, to) {
                    return Some(Deferred::Transfer {
                        queue,
                        request,
                        owner,
                        to,
                        rest: Vec::new(),
                    });
                }
            }
            Deferred::Replicate { .. } => return self.cross(work),
        }
        None
    }

    /// A commit takes effect: the private block becomes shared under its key.
    #[allow(clippy::too_many_arguments)]
    fn share(
        &mut self,
        client: ClientId,
        owner: OwnerId,
        class: u8,
        at: u32,
        block: Rc<Block>,
        key: Key,
        tokens: u16,
    ) -> Completion {
        let region = block.region();
        let cached = self
            .cache
            .get(&key)
            .is_some_and(|entry| entry.class != class || entry.copy_in(region).is_some());
        // The commit's reference passes to the entry, or is let go.
        block.disown();
        // R4: a key that is cached keeps its block; this one stays private.
        if cached {
            self.reclaim(block);
            return Completion::Duplicate;
        }

        // R6: the block is the one the commit was taken for, whatever its
        // owner released since; the owner keeps a pin only if its table
        // still holds the block at the position.
        if let Some(found) = self.owners.get_mut(&(client, owner))
            && let Some(held) = found.tables[usize::from(class)].get_mut(&at)
            && matches!(held, Held::Private(other) if Rc::ptr_eq(other, &block))
        {
            block.disown();
            *held = Held::Shared(Rc::clone(&block));
        }
        block.cached.set(Some(key));
        self.touch(&block);
        let maker = self.made_by(client);
        let entry = self.cache.entry(key).or_insert(Entry {
            class,
            device: Vec::new(),
            host: None,
            tokens,
            remote: false,
        });
        entry.device.push((block, maker));
        Completion::Committed
    }

    fn evict(&mut self, keys: &[Key]) -> Completion {
        let mut evicted = Vec::new();
        for key in keys {
            // Every device copy of the key that nothing else holds.
            let regions: Vec<u8> = self.cache.get(key).map_or_else(Vec::new, |entry| {
                entry
                    .device
                    .iter()
                    .map(|(block, _)| block.region())
                    .collect()
            });
            let mut any = false;
            for region in regions {
                any |= self.evict_in(key, region);
            }
            if any {
                evicted.push(*key);
            }
        }
        Completion::Evicted { keys: evicted }
    }

    fn victims(&mut self, queue: QueueId, region: u8, keys: Vec<Key>) -> Taken {
        if usize::from(region) >= self.regions.len() {
            return Err(Cause::Class);
        }
        // The units of the order's blocks that may be evicted: a cost of
        // the order's length, once.
        let distinct: HashSet<&Key> = keys.iter().collect();
        let units = distinct
            .into_iter()
            .filter_map(|key| self.cache.get(key)?.copy_in(region))
            .filter(|block| matches!(block.counted.get(), Counted::Evictable(_)))
            .map(|block| block.units())
            .sum();
        self.regions[usize::from(region)].install(queue, keys, units);
        Ok(Some(Completion::Done))
    }

    fn list(&self, client: ClientId) -> Completion {
        // The session's groups: each one's number, the bit of its memory and its own bit.
        let groups: Vec<(GroupId, u64, u8)> = self
            .groups
            .iter()
            .filter(|((of, group), _)| *of == client && group.0 < 8)
            .map(|((_, group), found)| (*group, 1 << found.memory, 1 << group.0))
            .collect();

        let mut keys: Vec<(Key, u8, u16, u8, u8, u8)> = Vec::new();
        for (key, entry) in &self.cache {
            let device = entry
                .device
                .iter()
                .filter(|(_, maker)| self.visible(*maker, client));
            let valid = device.fold(0, |all, (block, _)| all | block.valid.get());
            let held = groups
                .iter()
                .filter(|(_, memory, _)| valid & memory != 0)
                .fold(0, |all, (_, _, bit)| all | bit);
            let host = entry
                .host
                .as_ref()
                .filter(|(_, maker, _)| self.visible(*maker, client));
            // The groups that can load the host copy: it holds their host parts (R11).
            let hosts = groups
                .iter()
                .filter(|(group, _, _)| {
                    host.is_some_and(|(block, ..)| {
                        self.loadable(client, *group, entry.class, block)
                    })
                })
                .fold(0, |all, (_, _, bit)| all | bit);
            let tiers =
                u8::from(held != 0) | u8::from(host.is_some()) << 1 | u8::from(entry.remote) << 2;
            if tiers != 0 {
                keys.push((*key, entry.class, entry.tokens, tiers, held, hosts));
            }
        }
        keys.sort_unstable_by_key(|listed| listed.0);
        Completion::Listing { keys }
    }

    fn resize(&mut self, queue: QueueId, region: u8, units: u32) -> Taken {
        let found = self.regions.get(usize::from(region)).ok_or(Cause::Class)?;
        let (now, space) = (found.units(), usize::from(found.space));
        let mut evicted = Vec::new();

        if units < now {
            // A shrink evicts in the queue pair's order and never takes a
            // private or pinned block; it may stop short.
            let wanted = now - units;
            for key in found.orders.get(&queue).cloned().unwrap_or_default() {
                if self.regions[usize::from(region)].free_units() >= wanted {
                    break;
                }
                if self.evict_in(&key, region) {
                    evicted.push(key);
                }
            }
            for (offset, bytes) in self.regions[usize::from(region)].shrink(wanted) {
                insert_range(&mut self.quota[space], offset, bytes);
            }
        } else if units > now {
            // R10: a unit joins the region only with bytes that are in no region.
            let quota = &mut self.quota[space];
            self.regions[usize::from(region)].grow(units - now, |offset, bytes| {
                remove_range(quota, offset, bytes)
            });
        }
        Ok(Some(Completion::Resized {
            units: self.regions[usize::from(region)].units(),
            evicted,
        }))
    }
}

/// Adds a byte range to a list of free ranges, joining neighbours.
fn insert_range(ranges: &mut BTreeMap<u64, u64>, mut offset: u64, mut bytes: u64) {
    if let Some(next) = ranges.remove(&(offset + bytes)) {
        bytes += next;
    }
    let before = ranges
        .range(..offset)
        .next_back()
        .map(|(first, length)| (*first, *length));
    if let Some((first, length)) = before
        && first + length == offset
    {
        ranges.remove(&first);
        offset = first;
        bytes += length;
    }
    ranges.insert(offset, bytes);
}

/// Takes a byte range out of a list of free ranges when one of them holds
/// all of it.
fn remove_range(ranges: &mut BTreeMap<u64, u64>, offset: u64, bytes: u64) -> bool {
    let holder = ranges
        .range(..=offset)
        .next_back()
        .map(|(first, length)| (*first, *length));
    let Some((first, length)) = holder.filter(|(first, length)| first + length >= offset + bytes)
    else {
        return false;
    };
    ranges.remove(&first);
    if first < offset {
        ranges.insert(first, offset - first);
    }
    if first + length > offset + bytes {
        ranges.insert(offset + bytes, first + length - offset - bytes);
    }
    true
}

#[cfg(test)]
mod enumeration;
#[cfg(test)]
mod tests;
