//! The device block table of one rank of a worker group, as its worker keeps it on the host.
//!
//! The buffer pool in the Storage Server is the authority for every block a request holds; the
//! device block table is a copy of what the buffer pool's owner tables say, and the kernels'
//! addresses come only from it. [`BlockTables`] is that copy on the host side: for every class,
//! by slot and position, the runs of the block in the memory the rank's process maps and whether
//! a batch may write it. The runner writes its changes into the pooled tables in device memory
//! and derives every address of a batch from it; a batch from the scheduler carries slots and
//! positions, never units.
//!
//! A value is the table of one rank and is owned by that rank's worker thread: every method
//! takes `&self` or `&mut self`, nothing is shared and nothing locks. The ranks of a group are
//! handed the same pairs and requests in the same order, so their tables differ only in which
//! run of a block each keeps.
//!
//! Every entry belongs to the owner whose request gave it. A slot holds the entries of the owner
//! attached to the group there and, in a group that reads a class another group writes, the
//! entries of that class's owner too; a request that takes entries away takes only its owner's.
//! A group whose ranks lie in several processes reads a product in each process's memory: where
//! its writer wrote it, from the writer's allocation, and elsewhere from the crossing that copied
//! it there (`Replicate`), so a rank holds an entry only once its process's memory holds the
//! block.
//!
//! ## What feeds the table
//!
//! Only two things change it, and both come from the scheduler through the worker's queue:
//!
//! - [`BlockTables::map`]: a request with its completion, and the slot the scheduler keeps the
//!   request's entries under in this group: the slot row of the request's owner on the group, or,
//!   for a group that reads a class it does not write (a decoder reading a media product), the
//!   slot of the request's own owner there. `(Attach, Attached)` and `(Transfer, Transferred)`
//!   attach an owner to the group in the slot they state. `(Allocate, Allocated)`, `(Pin,
//!   Pinned)` and `(Transfer, Transferred)` put runs at the positions the request named.
//!   `(Replicate, Replicated)` puts, read-only, the runs the crossing made valid at the position
//!   it named, those the process maps; a rank whose process held the block already keeps its
//!   entry. A `Copy` that answers a `Load`, `Clone`, `Store`, `Save` or `Restore` queues the
//!   copies the rank must run. The answer of a `Finish` to a `Load` or a `Clone` (`Loaded`,
//!   `Duplicate`, `Cloned`) puts the destination runs at the positions the `Load` or the `Clone`
//!   named. Any other pair gives the table nothing.
//! - [`BlockTables::apply`]: a `Release`, `Detach`, `Commit`, `Transfer`, `Replicate` or
//!   `Finish` on its way to the submission queue, before the worker stamps it. Every `Finish`
//!   passes through it, the worker's own that reports a copy run as well as the scheduler's
//!   give-up.
//!
//! ## The obligations of the control plane each method serves
//!
//! The numbers are those of the buffer pool design's queue pair section ("what the control
//! plane must do right").
//!
//! | Method | Obligation |
//! | --- | --- |
//! | [`BlockTables::map`] | 1: an entry exists only as a run a completion gave, at the position its request named. 4: an entry is writable only when it came from `Allocated`, from a `Clone` answered `Cloned` or as a private block of `Transferred`; the destination of a load or a clone is no entry until the `Finish` of its copy is answered, so only the copy writes it |
//! | [`BlockTables::apply`] | 2, the part before the stamp: the entries of a `Release`, a `Detach` or a `Transfer` leave, the entry of a `Commit` or a `Replicate` becomes read-only, a given-up copy that no batch has started is dropped and its destinations are no longer awaited. The stamp itself, the sequence of the last batch launched, is the worker's |
//! | [`BlockTables::run`], [`BlockTables::runs`], [`BlockTables::units`], [`BlockTables::entries`] | 1: the only source of a kernel's addresses and of a media copy's |
//! | [`BlockTables::writable`] | 4: whether a batch may write a position |
//! | [`BlockTables::take_changes`] | 1: what the runner writes into the device block table, each change once. A run is stated for reset only while its entry is in the table, so the units of a block that left are never touched. 4: and only while a batch may write it, so a block a `Commit` marked before the next batch is never reset |
//! | [`BlockTables::queued_copies`], [`BlockTables::start_copy`] | 2 and 4: a copy is started once, by the batch that takes it, and only that copy writes its destination; a copy given up before a batch takes it never runs |
//!
//! Obligation 3, advancing `completed`, concerns batches and is not the table's.
//!
//! ## Runs and regions
//!
//! A run is `(region, first unit, count)`, the region by the number the buffer pool's answer to
//! `Open` gave it. A block of a KV class, an encoder output or a trajectory has one run; a media
//! product has one run per backing rank, each in that rank's product buffer. The table is told
//! the regions its rank maps (its own memory: the KV units, the slot rows, its parts of the host
//! regions, its product buffer) and the product buffers of the other ranks of its process, which
//! its device reaches by peer access ([`BlockTables::with_peers`]). It keeps, of each block,
//! every run that lies in one of them: [`BlockTables::run`] is the one in the rank's own memory,
//! which the device block table and the kernels address, and [`BlockTables::runs`] every one, so
//! the rank that runs a call reaches the runs of the backing ranks of its process that do not,
//! and a decode reaches the run its successor reads. A block with no run there gives this rank no
//! entry. A copy is the rank's to run when its own memory holds both the run read and the run
//! written: an encoder output in another rank's product buffer is stored to the host by that rank
//! alone, onto that rank's part of the host block.
//!
//! What a copy moves is counted from every run its completion states, whichever rank maps them,
//! in the unit sizes the table is given for the session's regions: the same figure on every rank
//! of the group, so the group's workers agree on which queued copies start with a batch.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ops::{Range, RangeInclusive};

use storage::runtime::{Completion, GroupId, OwnerId, Request, RequestId, Run};

use crate::worker::{Error, Result};

/// One position whose entry changed since the changes were last taken: what the rank's device
/// block table must hold there now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Change {
    /// The slot whose table changed.
    pub slot: u32,
    /// The class of the table.
    pub class: u8,
    /// The position in the slot's table of the class.
    pub position: u32,
    /// The run the entry names in the rank's own memory; `None` when the position holds no
    /// entry, or one whose block lies in other ranks' memory alone, so the device block table
    /// addresses nothing there.
    pub run: Option<Run>,
    /// Whether the run's units were given by `Allocated` since the changes were last taken:
    /// the runner resets their per-unit state before a batch addresses them. Never set for a
    /// position without an entry, for a block that holds content, nor for an entry a batch may
    /// not write, since a reset writes the units.
    pub reset: bool,
}

/// The copies the rank runs for one request: a `Load`, `Clone`, `Store`, `Save` or `Restore`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockCopy {
    /// The request that started the copies, which their `Finish` names.
    pub request: RequestId,
    /// Each copy, from a run to a run, in the order the completion stated them. Empty when
    /// every copy of the request lies in other ranks' memory; the request still counts as run
    /// by this rank once it is started.
    pub runs: Vec<(Run, Run)>,
    /// The encoding the runs read were written in and the encoding of the class they write,
    /// when they differ (`storage::runtime::Completion::Copy`): the copies then lay the bytes
    /// out through the copy manifest of the two, which a copy of bytes as they are does not.
    pub encodings: Option<([u8; 32], [u8; 32])>,
}

/// The device block table of one rank of a worker group on the host: see the module
/// documentation.
pub struct BlockTables {
    group: GroupId,
    /// The regions whose memory the rank maps.
    regions: Vec<u8>,
    /// The product buffers of the process's other ranks, which the rank's device reaches by
    /// peer access.
    peers: Vec<u8>,
    /// By host region number, the bit of the rank's host part in a copy's statement of the parts
    /// it writes (`storage::runtime::Completion::Copy::parts`): its place among the host parts its
    /// group copies the region's class with, in the order of their hosts' names and part indices,
    /// `None` where they do not include it. A region not
    /// named is one whose group states no host part, whose every rank writes its own.
    parts: HashMap<u8, Option<u32>>,
    /// By region number, the bytes of a unit summed over the ranks that hold a part of it, for
    /// every region a copy of the group may name.
    unit_bytes: HashMap<u8, u64>,
    /// The slot under which each owner's entries are kept, and whether the owner is attached to
    /// the group there; an owner of another group, whose class this group reads, is not.
    slots: HashMap<OwnerId, (u32, bool)>,
    /// By slot, class and position: the runs of the block the process maps, in the order the
    /// completion stated them, at most one in the rank's own memory; whether a batch may write
    /// it; and the owner whose request gave it. Every owner named here is in `slots` under that
    /// slot.
    entries: BTreeMap<(u32, u8, u32), Kept>,
    /// The positions that changed since the changes were last taken, each with whether its
    /// entry's units were given by `Allocated` since then. The latest change of a position
    /// replaces the earlier ones.
    changed: BTreeMap<(u32, u8, u32), bool>,
    /// The copies no batch has started, oldest first, each with the bytes it moves summed over
    /// the group's ranks.
    queued: VecDeque<(BlockCopy, u64)>,
    /// By the request that started the copies of a `Load` or a `Clone`: its owner, whether the
    /// `Finish` that reports the copies run was applied, and the entries (class, position,
    /// run, writable) the owner gets when that `Finish` is answered, one per destination in
    /// the rank's memory. A loaded entry is read-only, also when its block stays private
    /// because the key kept another block; a cloned one is writable. One `Finish` ends a copy,
    /// so a give-up applied after the copy's own is discarded and the entries stay awaited.
    destinations: HashMap<RequestId, (OwnerId, bool, Vec<Entry>)>,
}

/// An entry of one slot's tables as a completion gives it: the class, the position, the runs the
/// process maps, and whether a batch may write it.
type Entry = (u8, u32, Vec<Run>, bool);

/// An entry as the table keeps it: the runs, whether a batch may write it, and its owner.
type Kept = (Vec<Run>, bool, OwnerId);

/// The encoding a copy reads and the encoding of the class it writes.
type Encodings = ([u8; 32], [u8; 32]);

/// The position `index` places after `at`.
fn position(at: u32, index: usize) -> Result<u32> {
    u32::try_from(index)
        .ok()
        .and_then(|index| at.checked_add(index))
        .ok_or_else(|| {
            Error::Invalid(format!(
                "a completion states {index} runs after position {at}, past the last position"
            ))
        })
}

/// Whether `run` is a run of units: it has some, and its last is on the unit axis.
fn check_run((region, first, count): Run) -> Result<()> {
    if count == 0 || first.checked_add(count).is_none() {
        return Err(Error::Invalid(format!(
            "a completion states the run of {count} units from unit {first} of region {region}, \
             which is no run of units"
        )));
    }

    Ok(())
}

/// Whether `slot`, under which a completion is mapped, is the slot the completion states.
fn check_stated(slot: u32, stated: u32) -> Result<()> {
    if slot != stated || slot == 0 {
        return Err(Error::Invalid(format!(
            "a completion that states slot {stated} is mapped under slot {slot}"
        )));
    }

    Ok(())
}

impl BlockTables {
    /// The empty table of a rank of `group` that maps the memory of `regions`, by the numbers
    /// the answer to `Open` gave them: the group's slot rows, the device and host regions of
    /// the group's classes that the rank addresses, and its own product buffer.
    ///
    /// `unit_bytes` states, by the same numbers, the bytes of a unit of every region a copy of
    /// the group may name, mapped by this rank or not, summed over the ranks that hold a part of
    /// it (`storage::runtime::Region::unit_bytes`). Every rank of the group is given the same,
    /// so each counts what a copy moves alike.
    pub fn new(group: GroupId, regions: &[u8], unit_bytes: &[(u8, u64)]) -> Self {
        Self {
            group,
            regions: regions.to_vec(),
            peers: Vec::new(),
            parts: HashMap::new(),
            unit_bytes: unit_bytes.iter().copied().collect(),
            slots: HashMap::new(),
            entries: BTreeMap::new(),
            changed: BTreeMap::new(),
            queued: VecDeque::new(),
            destinations: HashMap::new(),
        }
    }

    /// The table of a rank whose process holds other ranks too, `peers` their product buffers by
    /// the numbers the answer to `Open` gave them: the rank's device reaches them by peer access
    /// (`crate::memory::buffers::BufferMemory::connect`), so the table keeps the runs a block has
    /// there beside the one in the rank's own memory. Called before the table is fed.
    pub(crate) fn with_peers(mut self, peers: &[u8]) -> Self {
        self.peers = peers
            .iter()
            .copied()
            .filter(|region| !self.regions.contains(region))
            .collect();
        self
    }

    /// The table of a rank whose group states the host parts it copies with: `parts` states, per
    /// host region by the number the answer to `Open` gave it, the place of the rank's own host
    /// part among the group's host parts of the class the region holds copies of
    /// (`storage::runtime::Group::parts`), `None` where they do not include it. A copy to the
    /// host is the rank's to run only where its answer names the rank's part among those it
    /// writes. Called before the table is fed.
    pub(crate) fn with_parts(mut self, parts: &[(u8, Option<u32>)]) -> Self {
        self.parts = parts.iter().copied().collect();
        self
    }

    /// The worker group whose rank the table belongs to.
    pub(crate) fn group(&self) -> GroupId {
        self.group
    }

    /// The slot under which the entries of `owner` are kept: the one its `Attached` or its
    /// `Transferred` stated, or for an owner of another group the one its first completion here
    /// was mapped under. `None` for an owner the table does not hold.
    pub(crate) fn slot(&self, owner: OwnerId) -> Option<u32> {
        self.slots.get(&owner).map(|(slot, _)| *slot)
    }

    /// The run of the block that `slot` holds at `position` of `class` in the rank's own
    /// memory; `None` where the slot holds no entry, or one whose block lies in other ranks'
    /// memory alone.
    pub(crate) fn run(&self, slot: u32, class: u8, position: u32) -> Option<Run> {
        self.runs(slot, class, position)
            .iter()
            .copied()
            .find(|run| self.regions.contains(&run.0))
    }

    /// Every run of the block that `slot` holds at `position` of `class` in the memory the
    /// process maps, the rank's own and its other ranks' product buffers, in the order the
    /// completion stated them; empty where the slot holds no entry.
    pub(crate) fn runs(&self, slot: u32, class: u8, position: u32) -> &[Run] {
        self.entries
            .get(&(slot, class, position))
            .map_or(&[], |(runs, ..)| runs.as_slice())
    }

    /// The units of the block that `slot` holds at `position` of `class`: `first..first +
    /// count` of its run, one unit for a block of one unit and the block's whole size
    /// otherwise. `None` where the slot holds no entry.
    pub(crate) fn units(&self, slot: u32, class: u8, position: u32) -> Option<Range<u32>> {
        // A run enters the table only when `first + count` does not overflow (`check_run`).
        self.run(slot, class, position)
            .map(|(_, first, count)| first..first + count)
    }

    /// Whether a batch may write the block that `slot` holds at `position` of `class`: the
    /// entry came from `Allocated`, from `Cloned` or as a private block of `Transferred`, and
    /// no `Commit` or `Replicate` has named it. A position without an entry is not writable.
    pub fn writable(&self, slot: u32, class: u8, position: u32) -> bool {
        self.entries
            .get(&(slot, class, position))
            .is_some_and(|(_, writable, _)| *writable)
    }

    /// The entries of `slot` in `class` whose block has a run in the rank's own memory, in
    /// position order: each position with that run.
    pub fn entries(&self, slot: u32, class: u8) -> impl Iterator<Item = (u32, Run)> + '_ {
        self.entries
            .range((slot, class, 0)..=(slot, class, u32::MAX))
            .filter_map(|(&(_, _, position), (runs, ..))| {
                let own = runs.iter().find(|run| self.regions.contains(&run.0));
                own.map(|run| (position, *run))
            })
    }

    /// Takes the changes since the last call, in slot, class and position order, one per
    /// position that changed: the state the position has now, whatever it passed through.
    ///
    /// A change asks for a reset only of an entry that `Allocated` gave since the last call and
    /// that a batch may still write: a reset writes the units, and a `Commit` handed between the
    /// `Allocated` and the next batch made the block read-only (obligation 4).
    pub(crate) fn take_changes(&mut self) -> Vec<Change> {
        let changed = std::mem::take(&mut self.changed);

        changed
            .into_iter()
            .map(|((slot, class, position), allocated)| {
                let run = self.run(slot, class, position);
                let writable = self
                    .entries
                    .get(&(slot, class, position))
                    .is_some_and(|(_, writable, _)| *writable);

                Change {
                    slot,
                    class,
                    position,
                    run,
                    reset: allocated && run.is_some() && writable,
                }
            })
            .collect()
    }

    /// The copies no batch has started, oldest first, each with the bytes it moves summed over
    /// the group's ranks: every key at the larger of what it reads and what it writes, counted
    /// from the runs its completion states, also those in other ranks' memory. The figure is the
    /// same on every rank of the group.
    pub(crate) fn queued_copies(&self) -> impl Iterator<Item = (&BlockCopy, u64)> + '_ {
        self.queued.iter().map(|(copy, bytes)| (copy, *bytes))
    }

    /// Takes the oldest copy that no batch has started, for the batch about to launch: the
    /// copy is started and is not offered again. `None` when nothing is queued.
    pub(crate) fn start_copy(&mut self) -> Option<BlockCopy> {
        self.queued.pop_front().map(|(copy, _)| copy)
    }

    /// Takes the completion of `request`, whose number is `id`, and changes the table by what
    /// it gives: an owner's slot, runs at the positions the request named, or copies to run.
    /// `slot` is the slot under which the scheduler keeps the request's entries in this group
    /// (module documentation): the slot an `Attached` or a `Transferred` states, and for any
    /// other pair the slot of the owner the request names on this group, or of the request's own
    /// owner here when this group reads a class it does not write.
    ///
    /// A pair that gives the table nothing (`Done`, `Refused`, `Full`, `Committed`, `Evicted`
    /// and the like, a `Load` or a `Store` answered at once, the answer of a `Finish` whose
    /// copies fill no position here) changes nothing. The answer of a `Finish` for a copy that
    /// was given up or whose owner left since is such a pair. A `Copy` of a `Load` or a `Clone`
    /// whose owner left the group before it arrived queues the copies, which must run and end,
    /// and fills nothing: the owner is not held again. The caller hands a pair only
    /// before any `Release`, `Detach`, `Commit`, `Transfer` or `Replicate` that names its
    /// entries is applied, and hands a group that reads a product in a process its writer did
    /// not write it in the product's runs there only by the `Replicated` that made them valid.
    ///
    /// # Errors
    ///
    /// Nothing changes when the pair is refused: a completion that carries a slot, runs or
    /// copies and does not answer a request of its kind; a group that is not the table's; slot
    /// 0, which addresses nothing, or a slot other than the one the completion states or the
    /// owner's entries are kept under; an owner attached twice; a slot another owner attached
    /// to the group holds; a position that holds an entry already, other than the entry a
    /// `Replicated` names again; runs that do not divide into the request's blocks, a
    /// `Replicated` of no run, or a block with two runs in the rank's regions; a load or a clone
    /// whose destination the rank maps and whose source it does not; a copy that names a region
    /// whose unit size the table was not given.
    pub(crate) fn map(
        &mut self,
        slot: u32,
        id: RequestId,
        request: &Request,
        completion: &Completion,
    ) -> Result<()> {
        match (request, completion) {
            (Request::Attach { owner, group }, Completion::Attached { slot: stated }) => {
                self.check_group(*group)?;
                check_stated(slot, *stated)?;
                self.check_attach(*owner, slot)?;

                self.slots.insert(*owner, (slot, true));
                Ok(())
            }

            (
                Request::Allocate {
                    owner,
                    class,
                    at,
                    count,
                    ..
                },
                Completion::Allocated { runs, .. },
            ) => {
                // The runs are block-major: every block states one run per region of its
                // class, so the blocks divide the runs evenly.
                let blocks = *count as usize;
                if blocks == 0 || runs.is_empty() || !runs.len().is_multiple_of(blocks) {
                    return Err(Error::Invalid(format!(
                        "`Allocated` states {} runs for {count} blocks",
                        runs.len()
                    )));
                }

                let mut entries = Vec::new();
                for (index, block) in runs.chunks(runs.len() / blocks).enumerate() {
                    let reached = self.reached(block)?;
                    if !reached.is_empty() {
                        entries.push((*class, position(*at, index)?, reached, true));
                    }
                }

                self.check_held(*owner, slot)?;
                self.check_vacant(slot, &entries)?;
                self.put(*owner, slot, entries, true);
                Ok(())
            }

            (
                Request::Pin {
                    owner,
                    class,
                    at,
                    keys,
                },
                Completion::Pinned { runs, .. },
            ) => {
                // The leading keys that were pinned, one run each: a cached block has one run.
                if runs.len() > keys.len() {
                    return Err(Error::Invalid(format!(
                        "`Pinned` states {} runs for {} keys",
                        runs.len(),
                        keys.len()
                    )));
                }

                let mut entries = Vec::new();
                for (index, run) in runs.iter().enumerate() {
                    let reached = self.reached(std::slice::from_ref(run))?;
                    if !reached.is_empty() {
                        entries.push((*class, position(*at, index)?, reached, false));
                    }
                }

                self.check_held(*owner, slot)?;
                self.check_vacant(slot, &entries)?;
                self.put(*owner, slot, entries, false);
                Ok(())
            }

            (
                Request::Transfer { owner, group, .. },
                Completion::Transferred { slot: stated, runs },
            ) => {
                self.check_group(*group)?;
                check_stated(slot, *stated)?;

                // A block of several runs states one item per run under one class and
                // position; the rank keeps those the process maps, one at most in its own
                // memory.
                let mut kept: BTreeMap<(u8, u32), (Vec<Run>, bool)> = BTreeMap::new();
                for &(class, position, run, private) in runs {
                    check_run(run)?;
                    if !self.reaches(run.0) {
                        continue;
                    }

                    let (held, _) = kept
                        .entry((class, position))
                        .or_insert((Vec::new(), private));
                    held.push(run);
                    if held
                        .iter()
                        .filter(|run| self.regions.contains(&run.0))
                        .count()
                        > 1
                    {
                        return Err(Error::Invalid(format!(
                            "`Transferred` states two runs in the rank's regions at position \
                             {position} of class {class}"
                        )));
                    }
                }
                let entries: Vec<_> = kept
                    .into_iter()
                    .map(|((class, position), (runs, private))| (class, position, runs, private))
                    .collect();

                self.check_attach(*owner, slot)?;
                self.check_vacant(slot, &entries)?;

                // The task of the transfer copied the blocks' content into the group's
                // memory: nothing is reset.
                self.slots.insert(*owner, (slot, true));
                self.put(*owner, slot, entries, false);
                Ok(())
            }

            (
                Request::Load {
                    owner, class, at, ..
                },
                Completion::Copy {
                    group,
                    keys,
                    from,
                    to,
                    encodings,
                    parts,
                    ..
                },
            ) => self.queue(
                id,
                (*group, keys.len()),
                (from, to, *encodings, parts),
                Some((*owner, slot, *class, *at, false)),
            ),

            (
                Request::Clone {
                    owner, class, at, ..
                },
                Completion::Copy {
                    group,
                    keys,
                    from,
                    to,
                    encodings,
                    parts,
                    ..
                },
            ) => self.queue(
                id,
                (*group, keys.len()),
                (from, to, *encodings, parts),
                Some((*owner, slot, *class, *at, true)),
            ),

            (
                Request::Store { .. } | Request::Save { .. } | Request::Restore { .. },
                Completion::Copy {
                    group,
                    keys,
                    from,
                    to,
                    encodings,
                    parts,
                    ..
                },
            ) => self.queue(
                id,
                (*group, keys.len()),
                (from, to, *encodings, parts),
                None,
            ),

            (
                Request::Replicate {
                    owner,
                    class,
                    at,
                    group,
                    ..
                },
                Completion::Replicated { runs },
            ) => {
                self.check_group(*group)?;
                if runs.is_empty() {
                    return Err(Error::Invalid(format!(
                        "`Replicated` states no run for position {at} of class {class}"
                    )));
                }
                let reached = self.reached(runs)?;
                self.check_held(*owner, slot)?;

                // A block its allocation already gave this process stays the entry it is: the
                // crossing wrote other memory.
                if let Some((held, _, holder)) = self.entries.get(&(slot, *class, *at))
                    && holder == owner
                    && *held == reached
                {
                    return Ok(());
                }
                let entries = if reached.is_empty() {
                    Vec::new()
                } else {
                    vec![(*class, *at, reached, false)]
                };
                self.check_vacant(slot, &entries)?;

                // The crossing wrote the block's content into the run here: nothing is reset,
                // and no batch writes a copy another memory's readers hold too.
                self.put(*owner, slot, entries, false);
                Ok(())
            }

            (
                Request::Finish { copy, ok: true, .. },
                Completion::Loaded | Completion::Duplicate | Completion::Cloned,
            ) => self.fill(slot, *copy),

            (
                _,
                Completion::Attached { .. }
                | Completion::Allocated { .. }
                | Completion::Pinned { .. }
                | Completion::Transferred { .. }
                | Completion::Copy { .. }
                | Completion::Cloned
                | Completion::Replicated { .. },
            ) => Err(Error::Invalid(format!(
                "{completion:?} does not answer {request:?}"
            ))),

            _ => Ok(()),
        }
    }

    /// Applies a request that the worker stamps, before it is stamped and submitted:
    ///
    /// - `Release`: the owner's entries at the positions it names leave;
    /// - `Detach`, `Transfer`: the owner leaves with every entry of it, and the destinations of
    ///   a copy that was to fill a position of it are no longer awaited. The entries other
    ///   owners keep in the same slot stay. Its
    ///   copies that no batch has started stay queued until their `Finish`: the buffer pool
    ///   holds what a copy reads and writes until then. An owner whose transfer fails holds no
    ///   entry here and takes only its `Detach`;
    /// - `Commit`, `Replicate`: the entry it names becomes read-only, and stays so whatever the
    ///   answer: a crossing reads the block once no batch in flight writes it, and none launched
    ///   after the stamp does;
    /// - `Finish` that reports a copy run: the copy is offered to no further batch, which it
    ///   already is once started;
    /// - `Finish` that gives a copy up: a copy that no batch has started is dropped and never
    ///   runs, and the destinations of a load or a clone are no longer awaited, unless the
    ///   copy's own `Finish` was applied before, which ends it.
    ///
    /// A request that names an owner, a position or a copy the table does not hold changes
    /// nothing: every group that reads a class is handed the request, whether or not its
    /// table holds the entries.
    ///
    /// # Errors
    ///
    /// Nothing changes when the request is refused: a request a worker does not stamp, or
    /// a `Finish` that reports as run a copy no batch has started.
    pub(crate) fn apply(&mut self, request: &Request) -> Result<()> {
        match request {
            Request::Release {
                owner,
                class,
                at,
                count,
                ..
            } => {
                if let (Some(slot), Some(further)) = (self.slot(*owner), count.checked_sub(1)) {
                    let last = at.saturating_add(further);
                    self.remove(*owner, (slot, *class, *at)..=(slot, *class, last));
                }
                Ok(())
            }

            Request::Detach { owner, .. } | Request::Transfer { owner, .. } => {
                if let Some((slot, _)) = self.slots.remove(owner) {
                    self.remove(*owner, (slot, 0, 0)..=(slot, u8::MAX, u32::MAX));
                }
                self.destinations.retain(|_, (holder, ..)| holder != owner);
                Ok(())
            }

            // A crossing reads the block in the memory its writer wrote it in, so no later batch
            // writes it, as no later batch writes a block a commit shares.
            Request::Commit {
                owner, class, at, ..
            }
            | Request::Replicate {
                owner, class, at, ..
            } => {
                let entry = self
                    .slot(*owner)
                    .and_then(|slot| self.entries.get_mut(&(slot, *class, *at)));
                if let Some((_, writable, holder)) = entry
                    && holder == owner
                {
                    *writable = false;
                }
                Ok(())
            }

            Request::Finish { copy, ok: true, .. } => {
                if self
                    .queued
                    .iter()
                    .any(|(queued, _)| queued.request == *copy)
                {
                    return Err(Error::Invalid(format!(
                        "a `Finish` reports copy {} as run, and no batch has started it",
                        copy.0
                    )));
                }

                if let Some((_, ended, _)) = self.destinations.get_mut(copy) {
                    *ended = true;
                }
                Ok(())
            }

            Request::Finish {
                copy, ok: false, ..
            } => {
                self.queued.retain(|(queued, _)| queued.request != *copy);

                let open = self
                    .destinations
                    .get(copy)
                    .is_some_and(|(_, ended, _)| !ended);
                if open {
                    self.destinations.remove(copy);
                }
                Ok(())
            }

            other => Err(Error::Invalid(format!(
                "{other:?} is not a request a worker stamps"
            ))),
        }
    }

    /// Whether the rank's own memory holds `run`.
    ///
    /// # Errors
    ///
    /// As [`check_run`].
    fn own(&self, run: Run) -> Result<bool> {
        check_run(run)?;
        Ok(self.regions.contains(&run.0))
    }

    /// Whether a copy into host region `region` that writes the host parts `parts` (a bit per
    /// host part of the group, `storage::runtime::Completion::Copy::parts`) writes the rank's
    /// own: always where the group states no host part of the region's class.
    fn writes_part(&self, region: u8, parts: u64) -> bool {
        match self.parts.get(&region) {
            None => true,
            Some(None) => false,
            Some(Some(bit)) => parts & 1u64.checked_shl(*bit).unwrap_or(0) != 0,
        }
    }

    /// Whether the process maps `region`: the rank's own memory, or another rank's product
    /// buffer that the rank's device reaches.
    fn reaches(&self, region: u8) -> bool {
        self.regions.contains(&region) || self.peers.contains(&region)
    }

    /// The bytes of `run`, summed over the ranks that hold a part of its units.
    ///
    /// # Errors
    ///
    /// A region the table was given no unit size of.
    fn bytes(&self, run: Run) -> Result<u64> {
        let (region, _, count) = run;

        match self.unit_bytes.get(&region) {
            Some(unit) => Ok(unit.saturating_mul(u64::from(count))),
            None => Err(Error::Invalid(format!(
                "`Copy` names region {region}, whose unit size the block table was not given"
            ))),
        }
    }

    /// Of the runs of one block, those the process maps, in their order; empty for a block that
    /// lies in memory the process does not map.
    ///
    /// # Errors
    ///
    /// A run that is no run of units ([`check_run`]), or a block with two runs in the rank's own
    /// memory.
    fn reached(&self, block: &[Run]) -> Result<Vec<Run>> {
        let mut kept = Vec::new();
        let mut own = false;
        for &run in block {
            if self.own(run)? {
                if own {
                    return Err(Error::Invalid(format!(
                        "a block has two runs in the rank's regions, the second in region {}",
                        run.0
                    )));
                }
                own = true;
            }
            if self.reaches(run.0) {
                kept.push(run);
            }
        }

        Ok(kept)
    }

    fn check_group(&self, group: GroupId) -> Result<()> {
        if group != self.group {
            return Err(Error::Invalid(format!(
                "the block table of group {} is handed a completion for group {}",
                self.group.0, group.0
            )));
        }

        Ok(())
    }

    /// Whether the entries `owner` is given may be kept under `slot`: a slot that addresses
    /// something, and the one the owner's entries are kept under already, if any.
    fn check_held(&self, owner: OwnerId, slot: u32) -> Result<()> {
        if slot == 0 {
            return Err(Error::Invalid(format!(
                "the entries of owner {} are mapped under slot 0, which addresses nothing",
                owner.0
            )));
        }

        match self.slot(owner) {
            Some(held) if held != slot => Err(Error::Invalid(format!(
                "the entries of owner {} are mapped under slot {slot} and kept under slot {held}",
                owner.0
            ))),
            _ => Ok(()),
        }
    }

    /// Whether `owner` can attach to the group in `slot`: it is not attached already, its
    /// entries are not kept under another slot, and no other owner attached to the group holds
    /// the slot. Owners of other groups whose classes this group reads share the slot.
    fn check_attach(&self, owner: OwnerId, slot: u32) -> Result<()> {
        self.check_held(owner, slot)?;
        if self
            .slots
            .get(&owner)
            .is_some_and(|(_, attached)| *attached)
        {
            return Err(Error::Invalid(format!(
                "owner {} is attached to group {} twice",
                owner.0, self.group.0
            )));
        }

        let holder = self
            .slots
            .iter()
            .find(|(held, (at, attached))| *attached && *at == slot && **held != owner);
        if let Some((holder, _)) = holder {
            return Err(Error::Invalid(format!(
                "owner {} is given slot {slot}, which owner {} holds",
                owner.0, holder.0
            )));
        }

        Ok(())
    }

    /// Whether every position of `entries` is without an entry in `slot`.
    fn check_vacant(&self, slot: u32, entries: &[Entry]) -> Result<()> {
        let occupied = entries
            .iter()
            .find(|(class, position, ..)| self.entries.contains_key(&(slot, *class, *position)));

        match occupied {
            Some((class, position, ..)) => Err(Error::Invalid(format!(
                "a completion gives a run for position {position} of class {class} in slot \
                 {slot}, which holds an entry"
            ))),
            None => Ok(()),
        }
    }

    /// Puts `entries` of `owner` into `slot`'s tables and records the change of each, keeping
    /// the owner's entries under `slot` from here on; `allocated` states that their units were
    /// given by `Allocated`.
    fn put(&mut self, owner: OwnerId, slot: u32, entries: Vec<Entry>, allocated: bool) {
        self.slots.entry(owner).or_insert((slot, false));

        for (class, position, runs, writable) in entries {
            self.entries
                .insert((slot, class, position), (runs, writable, owner));
            self.changed.insert((slot, class, position), allocated);
        }
    }

    /// Removes the entries of `owner` in `range` and records the change of each.
    fn remove(&mut self, owner: OwnerId, range: RangeInclusive<(u32, u8, u32)>) {
        let left: Vec<_> = self
            .entries
            .range(range)
            .filter(|(_, (.., holder))| *holder == owner)
            .map(|(key, _)| *key)
            .collect();

        for key in left {
            self.entries.remove(&key);
            self.changed.insert(key, false);
        }
    }

    /// Queues the copies of request `id`, which its completion states for `keys` keys of
    /// `group`: for each key, from the run of `from` to the run of `to`, in `encodings` when the
    /// source's is not the destination's, writing the host parts `parts` names for a copy to the
    /// host (none named for any other), with the bytes they move over the group.
    /// `destination` is the owner, the slot its entries are kept under, the class, the first
    /// position and whether the request is a clone, for a request whose copies fill positions.
    ///
    /// The copy of an owner the table no longer holds, whose `Detach` or `Transfer` was applied
    /// before its `Copy` arrived, is queued all the same, since it must run and end with its
    /// `Finish`, and fills nothing: the owner gets no slot and no destination is awaited.
    fn queue(
        &mut self,
        id: RequestId,
        (group, keys): (GroupId, usize),
        (from, to, encodings, parts): (&[Run], &[Run], Option<Encodings>, &[u64]),
        destination: Option<(OwnerId, u32, u8, u32, bool)>,
    ) -> Result<()> {
        self.check_group(group)?;
        if from.len() != keys || to.len() != keys || !(parts.is_empty() || parts.len() == keys) {
            return Err(Error::Invalid(format!(
                "`Copy` states {} sources, {} destinations and the host parts of {} for {keys} \
                 keys",
                from.len(),
                to.len(),
                parts.len()
            )));
        }

        // A copy's group is its owner's, so an owner this table does not hold has left the
        // group: the slot the scheduler names for it is no longer the owner's to fill.
        let destination = destination.filter(|(owner, ..)| self.slots.contains_key(owner));
        if let Some((owner, slot, ..)) = destination {
            self.check_held(owner, slot)?;
        }

        let mut runs = Vec::new();
        let mut entries = Vec::new();
        // The bytes the copy moves over the group: each key at the larger of what it reads and
        // what it writes, whichever ranks hold the runs.
        let mut bytes: u64 = 0;
        for (index, (&source, &target)) in from.iter().zip(to).enumerate() {
            let (reads, writes) = (self.own(source)?, self.own(target)?);
            bytes = bytes.saturating_add(self.bytes(source)?.max(self.bytes(target)?));

            // A position whose block lies in the rank's memory gets an entry, so the rank must
            // be able to fill it.
            if writes && !reads && destination.is_some() {
                return Err(Error::Invalid(format!(
                    "`Copy` fills region {}, which the rank maps, from region {}, which it \
                     does not",
                    target.0, source.0
                )));
            }
            // A copy to the host writes only the host parts its answer names: a rank whose part
            // holds the key's host copy already leaves it as it is.
            let named = parts
                .get(index)
                .is_none_or(|parts| self.writes_part(target.0, *parts));
            if reads && writes && named {
                runs.push((source, target));
            }

            // A destination the process maps becomes an entry: in the rank's own memory the
            // rank fills it, in another rank's product buffer that rank does. Only a clone's
            // block is its owner's to write; a loaded one is read-only whatever answers the
            // `Finish`.
            if let Some((_, _, class, at, cloned)) = destination
                && self.reaches(target.0)
            {
                entries.push((class, position(at, index)?, vec![target], cloned));
            }
        }

        let copy = BlockCopy {
            request: id,
            runs,
            encodings,
        };
        self.queued.push_back((copy, bytes));
        if let Some((owner, ..)) = destination {
            self.destinations.insert(id, (owner, false, entries));
        }
        Ok(())
    }

    /// The answer of the `Finish` of `copy` reports the copies of a `Load` or of a `Clone`
    /// done: their destinations become entries at the positions the request named, under
    /// `slot`, which is where the request's owner keeps its entries.
    fn fill(&mut self, slot: u32, copy: RequestId) -> Result<()> {
        // A copy that fills no position here: a store, a save, a restore, or a copy that was
        // given up or whose owner left since.
        let Some((owner, _, entries)) = self.destinations.get(&copy) else {
            return Ok(());
        };
        let owner = *owner;
        self.check_held(owner, slot)?;
        self.check_vacant(slot, entries)?;

        // The copy wrote the blocks' content: nothing is reset.
        if let Some((_, _, entries)) = self.destinations.remove(&copy) {
            self.put(owner, slot, entries, false);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use storage::Key;
    use storage::runtime::{Cause, Tier};

    use super::*;

    const GROUP: GroupId = GroupId(0);

    /// The KV units on the device, the host pool of their copies, this rank's product buffer
    /// and another rank's.
    const DEVICE: u8 = 0;
    const HOST: u8 = 4;
    const PRODUCTS: u8 = 2;
    const OTHER_PRODUCTS: u8 = 3;

    /// The bytes of a unit of each region, summed over the ranks: a product unit is smaller than
    /// a host unit.
    const UNIT_BYTES: [(u8, u64); 4] = [
        (DEVICE, 1024),
        (HOST, 1024),
        (PRODUCTS, 256),
        (OTHER_PRODUCTS, 256),
    ];

    fn tables() -> BlockTables {
        BlockTables::new(GROUP, &[DEVICE, HOST, PRODUCTS], &UNIT_BYTES)
    }

    fn keys(count: u8) -> Vec<Key> {
        (0..count).map(|byte| Key([byte; 32])).collect()
    }

    /// The slot under which the entries of `owner` are kept, as the scheduler names it: 0, which
    /// addresses nothing, for an owner the table does not hold.
    fn slot_of(tables: &BlockTables, owner: u64) -> u32 {
        tables.slot(OwnerId(owner)).unwrap_or(0)
    }

    fn attach(tables: &mut BlockTables, owner: u64, slot: u32) {
        let request = Request::Attach {
            owner: OwnerId(owner),
            group: GROUP,
        };

        tables
            .map(slot, RequestId(0), &request, &Completion::Attached { slot })
            .expect("an owner is attached");
    }

    fn detach(owner: u64) -> Request {
        Request::Detach {
            owner: OwnerId(owner),
            after: vec![],
        }
    }

    /// Hands the table an `Allocate` of one block per run with its `Allocated`.
    fn allocate(
        tables: &mut BlockTables,
        owner: u64,
        class: u8,
        at: u32,
        runs: &[Run],
    ) -> Result<()> {
        let request = Request::Allocate {
            owner: OwnerId(owner),
            class,
            at,
            count: runs.len() as u32,
            units: 0,
        };
        let completion = Completion::Allocated {
            runs: runs.to_vec(),
            evicted: vec![],
        };

        let slot = slot_of(tables, owner);
        tables.map(slot, RequestId(0), &request, &completion)
    }

    /// Hands the table a `Load` of one key per pair of runs with the `Copy` that answers it.
    fn load(
        tables: &mut BlockTables,
        id: u64,
        owner: u64,
        at: u32,
        from: &[Run],
        to: &[Run],
    ) -> Result<()> {
        let request = Request::Load {
            owner: OwnerId(owner),
            class: 0,
            at,
            keys: keys(from.len() as u8),
        };
        let completion = Completion::Copy {
            group: GROUP,
            keys: keys(from.len() as u8),
            from: from.to_vec(),
            to: to.to_vec(),
            evicted: vec![],
            encodings: None,
            parts: Vec::new(),
        };

        let slot = slot_of(tables, owner);
        tables.map(slot, RequestId(id), &request, &completion)
    }

    fn finish(copy: u64, ok: bool) -> Request {
        Request::Finish {
            copy: RequestId(copy),
            ok,
            after: vec![],
        }
    }

    fn entries(tables: &BlockTables, slot: u32, class: u8) -> Vec<(u32, Run)> {
        tables.entries(slot, class).collect()
    }

    #[test]
    fn an_entry_appears_only_from_a_completion_at_the_position_its_request_named() {
        let mut tables = tables();
        attach(&mut tables, 1, 3);
        assert_eq!(tables.slot(OwnerId(1)), Some(3));

        // A request that got no blocks leaves the table as it was.
        let request = Request::Allocate {
            owner: OwnerId(1),
            class: 0,
            at: 5,
            count: 2,
            units: 0,
        };
        for completion in [
            Completion::Full { region: DEVICE },
            Completion::Refused {
                cause: Cause::Position,
            },
        ] {
            tables.map(3, RequestId(1), &request, &completion).unwrap();
        }
        assert_eq!(entries(&tables, 3, 0), []);

        allocate(&mut tables, 1, 0, 5, &[(DEVICE, 7, 1), (DEVICE, 9, 1)]).unwrap();
        assert_eq!(
            entries(&tables, 3, 0),
            [(5, (DEVICE, 7, 1)), (6, (DEVICE, 9, 1))]
        );
        assert_eq!(tables.run(3, 0, 6), Some((DEVICE, 9, 1)));
        assert_eq!(tables.units(3, 0, 6), Some(9..10));
        assert_eq!(tables.run(3, 0, 4), None);
        assert_eq!(tables.run(3, 1, 5), None);
        assert_eq!(tables.run(2, 0, 5), None);

        // Slot 0, a slot other than the one the owner's entries are kept under, a position
        // that holds an entry and a completion that does not answer its request give nothing.
        assert!(allocate(&mut tables, 2, 0, 0, &[(DEVICE, 11, 1)]).is_err());
        let elsewhere = Request::Allocate {
            owner: OwnerId(1),
            class: 0,
            at: 0,
            count: 1,
            units: 0,
        };
        let one = Completion::Allocated {
            runs: vec![(DEVICE, 11, 1)],
            evicted: vec![],
        };
        assert!(tables.map(4, RequestId(2), &elsewhere, &one).is_err());
        assert!(allocate(&mut tables, 1, 0, 4, &[(DEVICE, 11, 1), (DEVICE, 12, 1)]).is_err());
        let unanswered = Request::Attach {
            owner: OwnerId(4),
            group: GROUP,
        };
        let allocated = Completion::Allocated {
            runs: vec![(DEVICE, 11, 1)],
            evicted: vec![],
        };
        assert!(
            tables
                .map(3, RequestId(2), &unanswered, &allocated)
                .is_err()
        );

        assert_eq!(
            entries(&tables, 3, 0),
            [(5, (DEVICE, 7, 1)), (6, (DEVICE, 9, 1))]
        );
        assert_eq!(tables.slot(OwnerId(4)), None);
    }

    #[test]
    fn a_release_a_detach_and_a_transfer_remove_exactly_the_entries_they_name() {
        let mut tables = tables();
        for owner in 1..=3 {
            attach(&mut tables, owner, owner as u32);
            let first = owner as u32 * 10;
            let runs: Vec<Run> = (0..4).map(|unit| (DEVICE, first + unit, 1)).collect();
            allocate(&mut tables, owner, 0, 0, &runs).unwrap();
        }
        allocate(&mut tables, 1, 1, 0, &[(DEVICE, 50, 2)]).unwrap();

        // A release names positions of one class of one owner.
        let release = Request::Release {
            owner: OwnerId(1),
            class: 0,
            at: 1,
            count: 2,
            after: vec![],
        };
        tables.apply(&release).unwrap();
        assert_eq!(
            entries(&tables, 1, 0),
            [(0, (DEVICE, 10, 1)), (3, (DEVICE, 13, 1))]
        );
        assert_eq!(entries(&tables, 1, 1), [(0, (DEVICE, 50, 2))]);
        assert_eq!(entries(&tables, 2, 0).len(), 4);

        // A detach and a transfer name an owner: its slot and every entry of it.
        tables.apply(&detach(2)).unwrap();
        let transfer = Request::Transfer {
            owner: OwnerId(3),
            group: GroupId(1),
            after: vec![],
        };
        tables.apply(&transfer).unwrap();
        assert_eq!(entries(&tables, 2, 0), []);
        assert_eq!(entries(&tables, 3, 0), []);
        assert_eq!(tables.slot(OwnerId(2)), None);
        assert_eq!(tables.slot(OwnerId(3)), None);
        assert_eq!(entries(&tables, 1, 0).len(), 2);
        assert_eq!(entries(&tables, 1, 1).len(), 1);

        // A request for an owner the table does not hold changes nothing, and the slot that
        // was left is another owner's to take, without entries.
        tables.apply(&detach(2)).unwrap();
        tables.apply(&release).unwrap();
        attach(&mut tables, 4, 2);
        assert_eq!(entries(&tables, 2, 0), []);
        assert_eq!(entries(&tables, 1, 0).len(), 2);

        // Only a request a worker stamps is applied.
        let unstamped = Request::Attach {
            owner: OwnerId(5),
            group: GROUP,
        };
        assert!(tables.apply(&unstamped).is_err());
    }

    #[test]
    fn a_group_that_reads_a_class_keeps_its_entries_under_the_slot_of_its_own_owner() {
        // Owner 1 is the request's attachment to this group, in slot 2; owner 7 is its
        // attachment to the group that writes the product class 5, whose completions this
        // group is handed under slot 2 too.
        let mut tables = tables();
        attach(&mut tables, 1, 2);
        allocate(&mut tables, 1, 0, 0, &[(DEVICE, 7, 1)]).unwrap();

        let products = Request::Allocate {
            owner: OwnerId(7),
            class: 5,
            at: 0,
            count: 2,
            units: 3,
        };
        let allocated = Completion::Allocated {
            runs: vec![(PRODUCTS, 0, 3), (PRODUCTS, 3, 3)],
            evicted: vec![],
        };
        tables.map(2, RequestId(1), &products, &allocated).unwrap();
        assert_eq!(tables.slot(OwnerId(7)), Some(2));
        assert_eq!(
            entries(&tables, 2, 5),
            [(0, (PRODUCTS, 0, 3)), (1, (PRODUCTS, 3, 3))]
        );

        // The other group's owner does not hold the slot: a third owner cannot attach there,
        // and owner 7's entries are kept under no other slot.
        let attach_there = Request::Attach {
            owner: OwnerId(9),
            group: GROUP,
        };
        assert!(
            tables
                .map(
                    2,
                    RequestId(2),
                    &attach_there,
                    &Completion::Attached { slot: 2 }
                )
                .is_err()
        );
        let more = Request::Allocate {
            owner: OwnerId(7),
            class: 5,
            at: 2,
            count: 1,
            units: 3,
        };
        let one = Completion::Allocated {
            runs: vec![(PRODUCTS, 6, 3)],
            evicted: vec![],
        };
        assert!(tables.map(3, RequestId(3), &more, &one).is_err());

        // A request that takes entries away takes only its own owner's: the product owner's
        // detach leaves owner 1's entries, and owner 1's release leaves the products.
        let release = Request::Release {
            owner: OwnerId(1),
            class: 5,
            at: 0,
            count: 2,
            after: vec![],
        };
        tables.apply(&release).unwrap();
        assert_eq!(entries(&tables, 2, 5).len(), 2);

        tables.apply(&detach(7)).unwrap();
        assert_eq!(entries(&tables, 2, 5), []);
        assert_eq!(entries(&tables, 2, 0), [(0, (DEVICE, 7, 1))]);
        assert_eq!(tables.slot(OwnerId(7)), None);
        assert_eq!(tables.slot(OwnerId(1)), Some(2));
    }

    #[test]
    fn a_committed_entry_is_read_only() {
        let mut tables = tables();
        attach(&mut tables, 1, 1);
        allocate(&mut tables, 1, 0, 0, &[(DEVICE, 7, 1), (DEVICE, 8, 1)]).unwrap();
        assert!(tables.writable(1, 0, 0));
        assert!(tables.writable(1, 0, 1));
        assert!(!tables.writable(1, 0, 2));

        let commit = Request::Commit {
            owner: OwnerId(1),
            class: 0,
            at: 0,
            key: Key([1; 32]),
            tokens: 64,
            after: vec![],
        };
        tables.apply(&commit).unwrap();
        assert!(!tables.writable(1, 0, 0));
        assert!(tables.writable(1, 0, 1));
        assert_eq!(tables.run(1, 0, 0), Some((DEVICE, 7, 1)));

        // The block stays its owner's alone when the key is cached already, and is still
        // not written again.
        tables
            .map(1, RequestId(9), &commit, &Completion::Duplicate)
            .unwrap();
        assert!(!tables.writable(1, 0, 0));
    }

    /// A product its writer wrote in one process and its group reads in two, as the denoising
    /// group of a deployment over two hosts reads the conditioning its first text rank wrote: the
    /// crossing freezes the entry it reads, and its answer gives each rank the runs its own
    /// process maps, the reading process an entry it never had and the writing process none new.
    #[test]
    fn a_crossing_freezes_its_entry_and_maps_only_the_runs_of_each_process() {
        // The product buffer of the first rank of the group on the other host.
        const RECEIVING: u8 = 7;
        let mut writing = tables();
        let mut reading = BlockTables::new(GROUP, &[DEVICE, HOST, RECEIVING], &UNIT_BYTES);
        for tables in [&mut writing, &mut reading] {
            attach(tables, 1, 2);
            allocate(tables, 1, 5, 0, &[(PRODUCTS, 0, 3)]).unwrap();
        }
        assert!(writing.writable(2, 5, 0));
        assert_eq!(reading.runs(2, 5, 0), []);

        // Stamped by every rank of the group: the entry a batch wrote is frozen, and a rank
        // that holds none is left as it was.
        let replicate = Request::Replicate {
            owner: OwnerId(1),
            class: 5,
            at: 0,
            from: PRODUCTS,
            group: GROUP,
            after: vec![],
        };
        for tables in [&mut writing, &mut reading] {
            tables.apply(&replicate).unwrap();
            tables.take_changes();
        }
        assert!(!writing.writable(2, 5, 0));

        // The block's runs in the group's regions and its receiving region: each rank keeps
        // those of its own process, read-only, and resets nothing.
        let replicated = Completion::Replicated {
            runs: vec![(PRODUCTS, 0, 3), (RECEIVING, 9, 3)],
        };
        for tables in [&mut writing, &mut reading] {
            tables
                .map(2, RequestId(2), &replicate, &replicated)
                .unwrap();
        }
        assert_eq!(writing.runs(2, 5, 0), [(PRODUCTS, 0, 3)]);
        assert!(!writing.writable(2, 5, 0));
        assert_eq!(writing.take_changes(), []);
        assert_eq!(reading.runs(2, 5, 0), [(RECEIVING, 9, 3)]);
        assert!(!reading.writable(2, 5, 0));
        assert_eq!(
            reading.take_changes(),
            [Change {
                slot: 2,
                class: 5,
                position: 0,
                run: Some((RECEIVING, 9, 3)),
                reset: false
            }]
        );

        // An answer for another group, of no run, to another request, or giving a position
        // runs other than those its entry holds, changes nothing.
        let elsewhere = Request::Replicate {
            owner: OwnerId(1),
            class: 5,
            at: 0,
            from: PRODUCTS,
            group: GroupId(1),
            after: vec![],
        };
        let none = Completion::Replicated { runs: vec![] };
        let other = Completion::Replicated {
            runs: vec![(RECEIVING, 12, 3)],
        };
        let unanswered = Request::Release {
            owner: OwnerId(1),
            class: 5,
            at: 0,
            count: 1,
            after: vec![],
        };
        for (request, completion) in [
            (&elsewhere, &replicated),
            (&replicate, &none),
            (&unanswered, &replicated),
            (&replicate, &other),
        ] {
            assert!(reading.map(2, RequestId(3), request, completion).is_err());
        }
        assert_eq!(reading.runs(2, 5, 0), [(RECEIVING, 9, 3)]);

        // The owner's release takes the crossed entry away with the others.
        reading.apply(&unanswered).unwrap();
        assert_eq!(reading.runs(2, 5, 0), []);
    }

    /// A group whose ranks lie on two hosts copies a class with a host part on each, this rank's
    /// the second the group states: a store writes on the parts its answer names, so a rank whose
    /// part holds the host copy already leaves it as it is, and a group that states no host parts
    /// writes every rank's.
    #[test]
    fn a_copy_to_the_host_runs_only_on_the_host_parts_its_answer_names() {
        let mut stating = tables().with_parts(&[(HOST, Some(1))]);
        let store = Request::Store {
            group: GROUP,
            keys: keys(3),
            tier: Tier::Host,
        };
        // The second key's copy fills the other host's part alone.
        let stored = Completion::Copy {
            group: GROUP,
            keys: keys(3),
            from: vec![(DEVICE, 1, 1), (DEVICE, 2, 1), (DEVICE, 3, 1)],
            to: vec![(HOST, 4, 1), (HOST, 5, 1), (HOST, 6, 1)],
            evicted: vec![],
            encodings: None,
            parts: vec![0b11, 0b01, 0b10],
        };
        stating.map(0, RequestId(1), &store, &stored).unwrap();
        let copy = stating.start_copy().unwrap();
        assert_eq!(
            copy.runs,
            [
                ((DEVICE, 1, 1), (HOST, 4, 1)),
                ((DEVICE, 3, 1), (HOST, 6, 1))
            ]
        );

        // Parts stated for some keys and not others say nothing a rank can follow.
        let short = Completion::Copy {
            group: GROUP,
            keys: keys(3),
            from: vec![(DEVICE, 1, 1), (DEVICE, 2, 1), (DEVICE, 3, 1)],
            to: vec![(HOST, 4, 1), (HOST, 5, 1), (HOST, 6, 1)],
            evicted: vec![],
            encodings: None,
            parts: vec![0b11],
        };
        assert!(stating.map(0, RequestId(2), &store, &short).is_err());

        // A rank whose host part is none of the group's writes none; a group that states no host
        // part has every rank write its own.
        let mut outside = tables().with_parts(&[(HOST, None)]);
        outside.map(0, RequestId(3), &store, &stored).unwrap();
        assert_eq!(outside.start_copy().unwrap().runs, []);
        let mut unstated = tables();
        unstated.map(0, RequestId(4), &store, &stored).unwrap();
        assert_eq!(unstated.start_copy().unwrap().runs.len(), 3);
    }

    #[test]
    fn pinned_and_loaded_entries_are_never_writable() {
        let mut tables = tables();
        attach(&mut tables, 1, 1);

        // Two of three keys were pinned: the leading positions.
        let pin = Request::Pin {
            owner: OwnerId(1),
            class: 0,
            at: 0,
            keys: keys(3),
        };
        let pinned = Completion::Pinned {
            runs: vec![(DEVICE, 20, 1), (DEVICE, 21, 1)],
            host: 1,
        };
        tables.map(1, RequestId(1), &pin, &pinned).unwrap();
        assert_eq!(
            entries(&tables, 1, 0),
            [(0, (DEVICE, 20, 1)), (1, (DEVICE, 21, 1))]
        );

        // A load fills its positions when the `Finish` of its copies is answered, whether
        // the blocks became the keys' or stayed the owner's alone.
        load(&mut tables, 2, 1, 2, &[(HOST, 0, 1)], &[(DEVICE, 22, 1)]).unwrap();
        load(&mut tables, 3, 1, 3, &[(HOST, 1, 1)], &[(DEVICE, 23, 1)]).unwrap();
        assert_eq!(entries(&tables, 1, 0).len(), 2);

        for (copy, answer) in [(2, Completion::Loaded), (3, Completion::Duplicate)] {
            assert_eq!(tables.start_copy().map(|copy| copy.request.0), Some(copy));
            tables.apply(&finish(copy, true)).unwrap();
            tables
                .map(1, RequestId(10), &finish(copy, true), &answer)
                .unwrap();
        }
        assert_eq!(tables.run(1, 0, 2), Some((DEVICE, 22, 1)));
        assert_eq!(tables.run(1, 0, 3), Some((DEVICE, 23, 1)));

        for position in 0..4 {
            assert!(!tables.writable(1, 0, position));
        }
    }

    #[test]
    fn a_clone_is_an_entry_only_once_its_copy_is_answered() {
        let mut tables = tables();
        attach(&mut tables, 1, 1);

        let clone = Request::Clone {
            owner: OwnerId(1),
            class: 0,
            at: 6,
            key: Key([1; 32]),
        };
        let copy = Completion::Copy {
            group: GROUP,
            keys: keys(1),
            from: vec![(DEVICE, 30, 1)],
            to: vec![(DEVICE, 31, 1)],
            evicted: vec![],
            encodings: None,
            parts: Vec::new(),
        };
        tables.map(1, RequestId(5), &clone, &copy).unwrap();

        // Until then only the copy writes the destination.
        let started = tables.start_copy().unwrap();
        assert_eq!(started.runs, [((DEVICE, 30, 1), (DEVICE, 31, 1))]);
        tables.apply(&finish(5, true)).unwrap();
        assert_eq!(tables.run(1, 0, 6), None);
        assert!(!tables.writable(1, 0, 6));

        // One `Finish` ends a copy: a give-up that crosses the copy's own is discarded.
        tables.apply(&finish(5, false)).unwrap();

        tables
            .map(1, RequestId(6), &finish(5, true), &Completion::Cloned)
            .unwrap();
        assert_eq!(tables.run(1, 0, 6), Some((DEVICE, 31, 1)));
        assert!(tables.writable(1, 0, 6));
    }

    #[test]
    fn a_copy_is_queued_by_its_completion_and_dropped_by_a_give_up_before_it_starts() {
        let mut tables = tables();
        attach(&mut tables, 1, 1);

        let store = Request::Store {
            group: GROUP,
            keys: keys(2),
            tier: Tier::Host,
        };
        let copy = Completion::Copy {
            group: GROUP,
            keys: keys(2),
            from: vec![(DEVICE, 5, 1), (DEVICE, 6, 1)],
            to: vec![(HOST, 0, 1), (HOST, 1, 1)],
            evicted: vec![],
            encodings: None,
            parts: Vec::new(),
        };
        tables.map(0, RequestId(10), &store, &copy).unwrap();
        load(&mut tables, 11, 1, 0, &[(HOST, 2, 1)], &[(DEVICE, 7, 1)]).unwrap();

        // A copy that names a region of no stated unit size is refused, and queues nothing.
        let unstated = Completion::Copy {
            group: GROUP,
            keys: keys(1),
            from: vec![(9, 0, 1)],
            to: vec![(HOST, 5, 1)],
            evicted: vec![],
            encodings: None,
            parts: Vec::new(),
        };
        assert!(tables.map(0, RequestId(12), &store, &unstated).is_err());

        let queued: Vec<(u64, u64)> = tables
            .queued_copies()
            .map(|(copy, bytes)| (copy.request.0, bytes))
            .collect();
        assert_eq!(queued, [(10, 2048), (11, 1024)]);

        // A copy that no batch has started cannot have run, and stays queued.
        assert!(tables.apply(&finish(11, true)).is_err());
        assert_eq!(tables.queued_copies().count(), 2);

        // The load is given up before a batch takes it: it is never run and fills nothing.
        tables.apply(&finish(11, false)).unwrap();
        tables
            .map(1, RequestId(12), &finish(11, false), &Completion::Done)
            .unwrap();

        let started = tables.start_copy().unwrap();
        assert_eq!(started.request, RequestId(10));
        assert_eq!(
            started.runs,
            [
                ((DEVICE, 5, 1), (HOST, 0, 1)),
                ((DEVICE, 6, 1), (HOST, 1, 1))
            ]
        );
        assert_eq!(tables.start_copy(), None);
        assert_eq!(tables.queued_copies().count(), 0);
        assert_eq!(tables.run(1, 0, 0), None);

        // A started copy is offered to no further batch, also after its end is reported.
        tables.apply(&finish(10, true)).unwrap();
        let stored = Completion::Stored { keys: vec![] };
        tables
            .map(0, RequestId(14), &finish(10, true), &stored)
            .unwrap();
        assert_eq!(tables.start_copy(), None);

        // A load whose owner leaves while its copy runs fills nothing, and the slot's next
        // owner gets no entry from it.
        load(&mut tables, 15, 1, 0, &[(HOST, 3, 1)], &[(DEVICE, 8, 1)]).unwrap();
        assert_eq!(tables.start_copy().map(|copy| copy.request.0), Some(15));
        tables.apply(&detach(1)).unwrap();
        attach(&mut tables, 2, 1);

        tables.apply(&finish(15, true)).unwrap();
        tables
            .map(1, RequestId(16), &finish(15, true), &Completion::Loaded)
            .unwrap();
        assert_eq!(entries(&tables, 1, 0), []);
    }

    #[test]
    fn the_changes_are_taken_once_and_state_what_the_table_holds() {
        let mut tables = tables();
        attach(&mut tables, 1, 2);
        let change = |position, run, reset| Change {
            slot: 2,
            class: 0,
            position,
            run,
            reset,
        };

        allocate(&mut tables, 1, 0, 0, &[(DEVICE, 7, 1), (DEVICE, 8, 1)]).unwrap();
        assert_eq!(
            tables.take_changes(),
            [
                change(0, Some((DEVICE, 7, 1)), true),
                change(1, Some((DEVICE, 8, 1)), true)
            ]
        );
        assert_eq!(tables.take_changes(), []);

        // A commit changes no address; a release leaves the position without one.
        let commit = Request::Commit {
            owner: OwnerId(1),
            class: 0,
            at: 0,
            key: Key([1; 32]),
            tokens: 64,
            after: vec![],
        };
        tables.apply(&commit).unwrap();
        assert_eq!(tables.take_changes(), []);

        let release = |at| Request::Release {
            owner: OwnerId(1),
            class: 0,
            at,
            count: 1,
            after: vec![],
        };
        tables.apply(&release(1)).unwrap();
        assert_eq!(tables.take_changes(), [change(1, None, false)]);

        // A block that was allocated and released between two batches is not reset: its
        // units are no longer the owner's. A pinned block that took a released position is
        // stated once, with its content kept.
        allocate(&mut tables, 1, 0, 1, &[(DEVICE, 9, 1)]).unwrap();
        tables.apply(&release(1)).unwrap();
        allocate(&mut tables, 1, 0, 2, &[(DEVICE, 10, 1)]).unwrap();
        tables.apply(&release(2)).unwrap();
        let pin = Request::Pin {
            owner: OwnerId(1),
            class: 0,
            at: 2,
            keys: keys(1),
        };
        let pinned = Completion::Pinned {
            runs: vec![(DEVICE, 20, 1)],
            host: 0,
        };
        tables.map(2, RequestId(3), &pin, &pinned).unwrap();
        assert_eq!(
            tables.take_changes(),
            [
                change(1, None, false),
                change(2, Some((DEVICE, 20, 1)), false)
            ]
        );
        assert_eq!(tables.take_changes(), []);
    }

    #[test]
    fn an_allocated_entry_a_commit_marked_before_the_next_batch_is_not_reset() {
        let mut tables = tables();
        attach(&mut tables, 1, 2);
        allocate(&mut tables, 1, 0, 0, &[(DEVICE, 7, 1), (DEVICE, 8, 1)]).unwrap();

        // The first block is committed before a batch takes the changes: it is read-only, and a
        // reset would write its units. The second is still the owner's to write.
        let commit = Request::Commit {
            owner: OwnerId(1),
            class: 0,
            at: 0,
            key: Key([1; 32]),
            tokens: 64,
            after: vec![],
        };
        tables.apply(&commit).unwrap();

        let change = |position, unit, reset| Change {
            slot: 2,
            class: 0,
            position,
            run: Some((DEVICE, unit, 1)),
            reset,
        };
        assert_eq!(
            tables.take_changes(),
            [change(0, 7, false), change(1, 8, true)]
        );
    }

    #[test]
    fn a_copy_mapped_after_its_owner_left_runs_and_fills_nothing() {
        let mut tables = tables();
        attach(&mut tables, 1, 1);
        tables.apply(&detach(1)).unwrap();

        // The `Copy` of the owner's load arrives after its detach: the copy is queued, since it
        // must run and end with its `Finish`, and the owner is not held again.
        let request = Request::Load {
            owner: OwnerId(1),
            class: 0,
            at: 0,
            keys: keys(1),
        };
        let completion = Completion::Copy {
            group: GROUP,
            keys: keys(1),
            from: vec![(HOST, 0, 1)],
            to: vec![(DEVICE, 7, 1)],
            evicted: vec![],
            encodings: None,
            parts: Vec::new(),
        };
        tables.map(1, RequestId(5), &request, &completion).unwrap();
        assert_eq!(tables.slot(OwnerId(1)), None);

        // The slot's next owner gets nothing from the copy, which runs with the next batch,
        // whatever answers its `Finish`.
        attach(&mut tables, 2, 1);
        assert_eq!(
            tables.start_copy(),
            Some(BlockCopy {
                request: RequestId(5),
                runs: vec![((HOST, 0, 1), (DEVICE, 7, 1))],
                encodings: None,
            })
        );
        tables.apply(&finish(5, true)).unwrap();
        tables
            .map(1, RequestId(6), &finish(5, true), &Completion::Loaded)
            .unwrap();

        assert_eq!(entries(&tables, 1, 0), []);
        assert_eq!(tables.slot(OwnerId(1)), None);
        assert_eq!(tables.slot(OwnerId(2)), Some(1));
    }

    #[test]
    fn a_rank_keeps_the_run_of_a_block_in_its_own_region() {
        let mut tables = tables();
        attach(&mut tables, 1, 1);

        // Two products of three units, each backed by this rank and by another.
        let products = Request::Allocate {
            owner: OwnerId(1),
            class: 5,
            at: 0,
            count: 2,
            units: 3,
        };
        let allocated = Completion::Allocated {
            runs: vec![
                (PRODUCTS, 0, 3),
                (OTHER_PRODUCTS, 8, 3),
                (PRODUCTS, 3, 3),
                (OTHER_PRODUCTS, 11, 3),
            ],
            evicted: vec![],
        };
        tables.map(1, RequestId(1), &products, &allocated).unwrap();
        assert_eq!(
            entries(&tables, 1, 5),
            [(0, (PRODUCTS, 0, 3)), (1, (PRODUCTS, 3, 3))]
        );
        assert_eq!(tables.units(1, 5, 1), Some(3..6));

        // An owner that moves into the group brings every entry it holds, each writable
        // when its block is private; a completion for another group is not this table's.
        let transfer = |group| Request::Transfer {
            owner: OwnerId(2),
            group,
            after: vec![],
        };
        let transferred = Completion::Transferred {
            slot: 4,
            runs: vec![
                (0, 0, (DEVICE, 40, 1), false),
                (0, 1, (DEVICE, 41, 1), true),
                (5, 0, (OTHER_PRODUCTS, 14, 3), true),
                (5, 0, (PRODUCTS, 6, 3), true),
            ],
        };
        assert!(
            tables
                .map(4, RequestId(2), &transfer(GroupId(1)), &transferred)
                .is_err()
        );
        assert_eq!(tables.slot(OwnerId(2)), None);

        tables.take_changes();
        tables
            .map(4, RequestId(2), &transfer(GROUP), &transferred)
            .unwrap();
        assert_eq!(tables.slot(OwnerId(2)), Some(4));
        assert_eq!(
            entries(&tables, 4, 0),
            [(0, (DEVICE, 40, 1)), (1, (DEVICE, 41, 1))]
        );
        assert_eq!(entries(&tables, 4, 5), [(0, (PRODUCTS, 6, 3))]);
        assert!(!tables.writable(4, 0, 0));
        assert!(tables.writable(4, 0, 1));
        assert!(tables.take_changes().iter().all(|change| !change.reset));

        // A load into another rank's memory, and a store from it to a host pool this rank
        // maps, are not this rank's to run, and give it no entry; their requests are still
        // started and answered here.
        let encoded = Request::Load {
            owner: OwnerId(1),
            class: 6,
            at: 0,
            keys: keys(1),
        };
        let loaded = Completion::Copy {
            group: GROUP,
            keys: keys(1),
            from: vec![(HOST, 0, 3)],
            to: vec![(OTHER_PRODUCTS, 20, 3)],
            evicted: vec![],
            encodings: None,
            parts: Vec::new(),
        };
        tables.map(1, RequestId(3), &encoded, &loaded).unwrap();
        let store = Request::Store {
            group: GROUP,
            keys: keys(1),
            tier: Tier::Host,
        };
        let stored = Completion::Copy {
            group: GROUP,
            keys: keys(1),
            from: vec![(OTHER_PRODUCTS, 23, 3)],
            to: vec![(HOST, 3, 3)],
            evicted: vec![],
            encodings: None,
            parts: Vec::new(),
        };
        tables.map(0, RequestId(4), &store, &stored).unwrap();

        // Each still counts what it moves over the group, at its larger side: three host units.
        let bytes: Vec<u64> = tables.queued_copies().map(|(_, bytes)| bytes).collect();
        assert_eq!(bytes, [3 * 1024, 3 * 1024]);

        for request in [3, 4] {
            assert_eq!(
                tables.start_copy(),
                Some(BlockCopy {
                    request: RequestId(request),
                    runs: vec![],
                    encodings: None,
                })
            );
        }

        tables.apply(&finish(3, true)).unwrap();
        tables
            .map(1, RequestId(5), &finish(3, true), &Completion::Loaded)
            .unwrap();
        assert_eq!(entries(&tables, 1, 6), []);
    }

    #[test]
    fn a_rank_keeps_every_run_its_process_maps_and_runs_the_copies_of_its_own_memory() {
        // This rank's product buffer and the one of the other rank of its process.
        let mut tables = tables().with_peers(&[OTHER_PRODUCTS]);
        attach(&mut tables, 1, 1);

        // A product backed by both ranks: the entry keeps both runs, and the device table and
        // the kernels address the rank's own.
        let products = Request::Allocate {
            owner: OwnerId(1),
            class: 5,
            at: 0,
            count: 1,
            units: 3,
        };
        let allocated = Completion::Allocated {
            runs: vec![(PRODUCTS, 0, 3), (OTHER_PRODUCTS, 8, 3)],
            evicted: vec![],
        };
        tables.map(1, RequestId(1), &products, &allocated).unwrap();
        assert_eq!(
            tables.runs(1, 5, 0),
            [(PRODUCTS, 0, 3), (OTHER_PRODUCTS, 8, 3)]
        );
        assert_eq!(tables.run(1, 5, 0), Some((PRODUCTS, 0, 3)));
        assert!(tables.writable(1, 5, 0));

        // A block the other rank alone holds is an entry this rank reaches, with nothing for
        // its own device table to address.
        tables.take_changes();
        let pin = Request::Pin {
            owner: OwnerId(1),
            class: 6,
            at: 0,
            keys: keys(1),
        };
        let pinned = Completion::Pinned {
            runs: vec![(OTHER_PRODUCTS, 12, 2)],
            host: 0,
        };
        tables.map(1, RequestId(2), &pin, &pinned).unwrap();
        assert_eq!(tables.runs(1, 6, 0), [(OTHER_PRODUCTS, 12, 2)]);
        assert_eq!(tables.run(1, 6, 0), None);
        assert_eq!(entries(&tables, 1, 6), []);
        assert_eq!(
            tables.take_changes(),
            [Change {
                slot: 1,
                class: 6,
                position: 0,
                run: None,
                reset: false
            }]
        );

        // A load into the other rank's buffer is that rank's copy, which this rank starts with
        // its batch and does not run; its destination is an entry here once it is answered. The
        // copy carries the encodings its completion names.
        let load = Request::Load {
            owner: OwnerId(1),
            class: 6,
            at: 1,
            keys: keys(1),
        };
        let copy = Completion::Copy {
            group: GROUP,
            keys: keys(1),
            from: vec![(HOST, 4, 3)],
            to: vec![(OTHER_PRODUCTS, 0, 3)],
            evicted: vec![],
            encodings: Some(([1; 32], [2; 32])),
            parts: Vec::new(),
        };
        tables.map(1, RequestId(3), &load, &copy).unwrap();
        assert_eq!(
            tables.start_copy(),
            Some(BlockCopy {
                request: RequestId(3),
                runs: vec![],
                encodings: Some(([1; 32], [2; 32])),
            })
        );
        tables.apply(&finish(3, true)).unwrap();
        tables
            .map(1, RequestId(4), &finish(3, true), &Completion::Loaded)
            .unwrap();
        assert_eq!(tables.runs(1, 6, 1), [(OTHER_PRODUCTS, 0, 3)]);
        assert!(!tables.writable(1, 6, 1));

        // An owner that moves in keeps the runs its process maps of each block, one at most in
        // the rank's own memory.
        let transfer = Request::Transfer {
            owner: OwnerId(2),
            group: GROUP,
            after: vec![],
        };
        let transferred = Completion::Transferred {
            slot: 4,
            runs: vec![
                (5, 0, (OTHER_PRODUCTS, 14, 2), true),
                (5, 0, (PRODUCTS, 6, 2), true),
                (5, 0, (9, 0, 2), true),
            ],
        };
        tables
            .map(4, RequestId(5), &transfer, &transferred)
            .unwrap();
        assert_eq!(
            tables.runs(4, 5, 0),
            [(OTHER_PRODUCTS, 14, 2), (PRODUCTS, 6, 2)]
        );
        assert_eq!(tables.run(4, 5, 0), Some((PRODUCTS, 6, 2)));

        // A detach takes the owner's entries, every run of each with them.
        tables.apply(&detach(2)).unwrap();
        assert_eq!(tables.runs(4, 5, 0), []);
    }
}
