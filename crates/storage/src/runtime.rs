//! The runtime interface of the buffer pool: the requests a scheduler writes
//! into its submission queue and the completions the buffer pool writes into
//! the completion queue ([`crate::queue`]).
//!
//! The buffer pool is the data plane for every memory a request holds: its
//! slot, its KV blocks, its encoder outputs, its media products and its
//! stored trajectory. The scheduler decides; it asks for blocks by class,
//! position and count and names cached blocks by key, and the buffer pool
//! performs what it is asked and reports. Every request has exactly one
//! completion, which carries the request's number and its result and repeats
//! nothing the request said.
//!
//! A record is a little-endian byte string: the request's number, one byte
//! that names the variant, and the variant's fields in the order they are
//! declared; a list is its length as a `u32` followed by its elements. A
//! record that is not one of this version's is an error to its reader, which
//! the buffer pool answers with [`Completion::Refused`] where it can read the
//! number and drops otherwise.
//!
//! [`VERSION`] is stated when a session is opened; a session keeps its version
//! for its life, and a new request or completion is a new version.

use serde::{Deserialize, Serialize};

use crate::{Error, Key, Result};

/// The version of this interface.
pub const VERSION: u32 = 1;

/// A request's number on its queue pair, drawn by the scheduler from a
/// counter and never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(pub u64);

/// An owner's number within its session: a request's attachment to one worker
/// group. Drawn by the scheduler from a counter and never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OwnerId(pub u64);

/// A worker group of the session's engine epoch. The engine numbers the
/// groups of its session, so a group is one scheduler's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GroupId(pub u8);

/// Where a copy of a block lies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Device,
    Host,
    /// The remote store, behind the host tier.
    Remote,
}

/// Where a prefetch reads from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Remote,
    Peers,
}

/// What a refused request named that does not fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cause {
    Owner,
    Position,
    Key,
    Group,
    Class,
    Size,
    Copy,
    /// The request itself: its number was taken before, or its record is
    /// not one of this version's.
    Request,
}

/// What an engine epoch states to open its session with the buffer pool of
/// its deployment: its layout. It travels on the Storage Server's socket;
/// the answer is [`Opened`], with one queue pair's descriptor per scheduler.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Open {
    /// The version of this interface the engine speaks: [`VERSION`].
    pub version: u32,
    /// The engine epoch, and whether its update is on trial.
    pub epoch: u64,
    pub trial: bool,
    /// The regions of the layout. A region another session of the
    /// deployment stated under the same name is that region.
    pub regions: Vec<Region>,
    /// The classes; a request names a class by its index here.
    pub classes: Vec<Class>,
    /// The class of stored trajectories.
    pub trajectory: u8,
    pub groups: Vec<Group>,
    /// Per scheduler, its groups. Each scheduler gets a queue pair whose
    /// registers hold the `completed` word of each of its groups and the
    /// capacity words of each region its groups and their classes name.
    pub schedulers: Vec<Vec<GroupId>>,
}

/// The memory blocks are taken from, on one tier and in one layout.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Region {
    pub name: String,
    pub tier: Tier,
    /// The memory space: the unit axis of one physical layout. Regions that
    /// state one space are cut from the same allocations at the same
    /// positions.
    pub space: String,
    /// The bytes of a unit, summed over the ranks that hold a part of it.
    pub unit_bytes: u64,
    /// The units before this one are never allocated: unit 0 and the
    /// executor's scratch units of a cache, row 0 of the slot rows.
    pub reserved: u32,
    /// The units the engine asks to have in the region, and the limit of
    /// the region's numbering.
    pub units: u32,
    pub limit: u32,
}

/// A kind of block.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Class {
    /// The units of one block; 0 when each request states the size.
    pub units: u32,
    /// The host region of the class's host copies, as an index into
    /// [`Open::regions`], and the units of one host copy; 0 units when a
    /// host copy takes the units that hold the block's bytes.
    pub host: Option<(u8, u32)>,
    /// Whether the class's live state is in the slot, which a clone cannot copy.
    pub state_in_slot: bool,
}

/// One worker group of the epoch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Group {
    pub id: GroupId,
    /// The device memory the group's ranks address. Groups of two sessions
    /// that state one name read and write the same memory.
    pub memory: String,
    /// The region of its slot rows, and per class the device regions its
    /// owners' blocks are taken from: one, or one per backing rank of a
    /// product, none for a class the group lacks. Indices into
    /// [`Open::regions`].
    pub slots: u8,
    pub regions: Vec<Vec<u8>>,
}

/// The answer to [`Open`]. The descriptors of the queue pairs travel with
/// it, one per scheduler in the order of [`Open::schedulers`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Opened {
    /// Per region of [`Open::regions`]: the number completions name it by,
    /// and the units in it.
    pub regions: Vec<(u8, u32)>,
    /// Per scheduler, where its queue pair's registers lie.
    pub registers: Vec<Registers>,
}

/// The registers of one scheduler's queue pair.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Registers {
    /// Per group of the scheduler, in the order it listed them: the index
    /// of the group's `completed` word, which the group's executor writes.
    pub completed: Vec<u32>,
    /// Per region the scheduler uses, by its number: the index of the first
    /// of its three capacity words (`available`, `retiring`, `evictable`),
    /// which the buffer pool writes.
    pub capacity: Vec<(u8, u32)>,
}

/// The stamps of a request: per stamping group, the sequence of the last
/// batch its executor had launched when it applied the request.
pub type Stamps = Vec<(GroupId, u64)>;

/// A run of units as a completion states it: its region, its first unit and
/// its count.
pub type Run = (u8, u32, u32);

/// A request of the submission queue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Request {
    /// Create the owner with a slot row of the group.
    Attach { owner: OwnerId, group: GroupId },
    /// Release the owner's slot and every entry of its tables; the owner is gone.
    Detach { owner: OwnerId, after: Stamps },
    /// Private blocks at `at` and the positions after it, all or none;
    /// `units` is the size of each block of a sized class and 0 otherwise.
    Allocate {
        owner: OwnerId,
        class: u8,
        at: u32,
        count: u32,
        units: u32,
    },
    /// Pin the leading keys whose device copy the owner may use, at `at` and after.
    Pin {
        owner: OwnerId,
        class: u8,
        at: u32,
        keys: Vec<Key>,
    },
    /// Take entries out of the owner's table: a private block retires, a
    /// shared block loses a pin.
    Release {
        owner: OwnerId,
        class: u8,
        at: u32,
        count: u32,
        after: Stamps,
    },
    /// Make the private block at `at` shared under the key.
    Commit {
        owner: OwnerId,
        class: u8,
        at: u32,
        key: Key,
        tokens: u16,
        after: Stamps,
    },
    /// A private block at `at` that the executor fills from the cached one.
    Clone {
        owner: OwnerId,
        class: u8,
        at: u32,
        key: Key,
    },
    /// Move the owner to another group, copying its blocks to that group's memory.
    Transfer {
        owner: OwnerId,
        group: GroupId,
        after: Stamps,
    },
    /// Reclaim each named block that is shared and has no pin.
    Evict { keys: Vec<Key> },
    /// Replace the order in which this queue pair's allocations may evict in the region.
    Victims { region: u8, keys: Vec<Key> },
    /// Bring the leading keys that have a host copy to the device, pinned for the owner.
    Load {
        owner: OwnerId,
        class: u8,
        at: u32,
        keys: Vec<Key>,
    },
    /// Copy each named key one tier down: to the host, or from the host to
    /// the remote store.
    Store {
        group: GroupId,
        keys: Vec<Key>,
        tier: Tier,
    },
    /// Copy the owner's trajectory from its slot into a host block under the key.
    Save {
        owner: OwnerId,
        key: Key,
        units: u32,
    },
    /// Copy the trajectory cached under the key back into the owner's slot.
    Restore { owner: OwnerId, key: Key },
    /// The result of the copies the engine ran for the request `copy`.
    Finish {
        copy: RequestId,
        ok: bool,
        after: Stamps,
    },
    /// Bring the leading keys the source holds into the host tier by the
    /// deadline, in nanoseconds of the host's monotonic clock. The keys are
    /// of one class; `units` is the size of each block of a sized class and
    /// 0 otherwise.
    Prefetch {
        class: u8,
        units: u32,
        keys: Vec<Key>,
        source: Source,
        deadline: u64,
    },
    /// State every cached key.
    List,
    /// Take units out of the region's free list or put them back, up to `units` in all.
    Resize { region: u8, units: u32 },
}

/// The result a completion carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Completion {
    /// The request took effect and has nothing to state.
    Done,
    /// What the request names does not fit; nothing changed.
    Refused {
        cause: Cause,
    },
    /// The region cannot supply the blocks; nothing changed.
    Full {
        region: u8,
    },
    Attached {
        slot: u32,
    },
    /// The runs of the new blocks in position order, and the keys evicted for them.
    Allocated {
        runs: Vec<Run>,
        evicted: Vec<Key>,
    },
    /// The runs of the pinned blocks in position order; `host` counts the
    /// further leading keys that have a host copy.
    Pinned {
        runs: Vec<Run>,
        host: u32,
    },
    Committed,
    /// The key is cached already; the block stays private.
    Duplicate,
    /// Copies for the group's executor to run: for each of `keys`, from the
    /// run of `from` to the run of `to`; and the keys evicted for the destinations.
    Copy {
        group: GroupId,
        keys: Vec<Key>,
        from: Vec<Run>,
        to: Vec<Run>,
        evicted: Vec<Key>,
    },
    Cloned,
    Loaded,
    /// The keys now on the tier the store named; empty for a store to the host.
    Stored {
        keys: Vec<Key>,
    },
    Saved,
    Restored,
    /// The owner's slot in the group it moved to, and every entry it holds
    /// there: class, position, run, and whether the block is private.
    Transferred {
        slot: u32,
        runs: Vec<(u8, u32, Run, bool)>,
    },
    /// A transfer or a task failed; what it held was released.
    Failed,
    Evicted {
        keys: Vec<Key>,
    },
    /// How many of the leading keys the source held.
    Prefetched {
        found: u32,
    },
    /// Each cached key: its class, its tokens, the tiers that hold it (bit 0
    /// device, bit 1 host, bit 2 remote) and the groups whose memory holds
    /// its device copy (bit per group).
    Listing {
        keys: Vec<(Key, u8, u16, u8, u8)>,
    },
    /// The region's units after the resize, and the keys evicted for a shrink.
    Resized {
        units: u32,
        evicted: Vec<Key>,
    },
}

/// Writes fields into a record.
struct Writer<'a>(&'a mut Vec<u8>);

impl Writer<'_> {
    fn byte(&mut self, value: u8) {
        self.0.push(value);
    }

    fn short(&mut self, value: u16) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn word(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn long(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn key(&mut self, key: &Key) {
        self.0.extend_from_slice(&key.0);
    }

    fn keys(&mut self, keys: &[Key]) {
        self.word(keys.len() as u32);
        for key in keys {
            self.key(key);
        }
    }

    fn run(&mut self, run: &Run) {
        self.byte(run.0);
        self.word(run.1);
        self.word(run.2);
    }

    fn runs(&mut self, runs: &[Run]) {
        self.word(runs.len() as u32);
        for run in runs {
            self.run(run);
        }
    }

    fn stamps(&mut self, stamps: &Stamps) {
        self.word(stamps.len() as u32);
        for (group, sequence) in stamps {
            self.byte(group.0);
            self.long(*sequence);
        }
    }
}

/// Reads fields out of a record; every read fails past the record's end.
struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take(&mut self, count: usize) -> Result<&[u8]> {
        if self.0.len() < count {
            return Err(Error::Protocol("a record ends inside a field".into()));
        }
        let (head, rest) = self.0.split_at(count);
        self.0 = rest;
        Ok(head)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn short(&mut self) -> Result<u16> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn word(&mut self) -> Result<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn long(&mut self) -> Result<u64> {
        let mut value = [0; 8];
        value.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(value))
    }

    fn key(&mut self) -> Result<Key> {
        let mut key = [0; 32];
        key.copy_from_slice(self.take(32)?);
        Ok(Key(key))
    }

    /// A list's length, refused when the record cannot hold that many
    /// elements of at least `least` bytes each.
    fn length(&mut self, least: usize) -> Result<usize> {
        let length = self.word()? as usize;
        if length.saturating_mul(least) > self.0.len() {
            return Err(Error::Protocol("a list is longer than its record".into()));
        }
        Ok(length)
    }

    fn keys(&mut self) -> Result<Vec<Key>> {
        (0..self.length(32)?).map(|_| self.key()).collect()
    }

    fn run(&mut self) -> Result<Run> {
        Ok((self.byte()?, self.word()?, self.word()?))
    }

    fn runs(&mut self) -> Result<Vec<Run>> {
        (0..self.length(9)?).map(|_| self.run()).collect()
    }

    fn stamps(&mut self) -> Result<Stamps> {
        (0..self.length(9)?)
            .map(|_| Ok((GroupId(self.byte()?), self.long()?)))
            .collect()
    }

    fn flag(&mut self) -> Result<bool> {
        match self.byte()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(Error::Protocol(format!("{other} is not a flag"))),
        }
    }

    fn end(&self) -> Result<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(Error::Protocol(
                "a record has bytes after its last field".into(),
            ))
        }
    }
}

fn tier_byte(tier: Tier) -> u8 {
    match tier {
        Tier::Device => 0,
        Tier::Host => 1,
        Tier::Remote => 2,
    }
}

fn cause_byte(cause: Cause) -> u8 {
    match cause {
        Cause::Owner => 0,
        Cause::Position => 1,
        Cause::Key => 2,
        Cause::Group => 3,
        Cause::Class => 4,
        Cause::Size => 5,
        Cause::Copy => 6,
        Cause::Request => 7,
    }
}

/// The number a record begins with, readable even when the rest is not.
pub fn number(record: &[u8]) -> Option<RequestId> {
    let bytes = record.get(..8)?;
    let mut value = [0; 8];
    value.copy_from_slice(bytes);
    Some(RequestId(u64::from_le_bytes(value)))
}

impl Request {
    /// Appends the record of this request under `id` to `record`.
    pub fn encode(&self, id: RequestId, record: &mut Vec<u8>) {
        let mut writer = Writer(record);
        writer.long(id.0);
        match self {
            Self::Attach { owner, group } => {
                writer.byte(0);
                writer.long(owner.0);
                writer.byte(group.0);
            }
            Self::Detach { owner, after } => {
                writer.byte(1);
                writer.long(owner.0);
                writer.stamps(after);
            }
            Self::Allocate {
                owner,
                class,
                at,
                count,
                units,
            } => {
                writer.byte(2);
                writer.long(owner.0);
                writer.byte(*class);
                writer.word(*at);
                writer.word(*count);
                writer.word(*units);
            }
            Self::Pin {
                owner,
                class,
                at,
                keys,
            } => {
                writer.byte(3);
                writer.long(owner.0);
                writer.byte(*class);
                writer.word(*at);
                writer.keys(keys);
            }
            Self::Release {
                owner,
                class,
                at,
                count,
                after,
            } => {
                writer.byte(4);
                writer.long(owner.0);
                writer.byte(*class);
                writer.word(*at);
                writer.word(*count);
                writer.stamps(after);
            }
            Self::Commit {
                owner,
                class,
                at,
                key,
                tokens,
                after,
            } => {
                writer.byte(5);
                writer.long(owner.0);
                writer.byte(*class);
                writer.word(*at);
                writer.key(key);
                writer.short(*tokens);
                writer.stamps(after);
            }
            Self::Clone {
                owner,
                class,
                at,
                key,
            } => {
                writer.byte(6);
                writer.long(owner.0);
                writer.byte(*class);
                writer.word(*at);
                writer.key(key);
            }
            Self::Transfer {
                owner,
                group,
                after,
            } => {
                writer.byte(7);
                writer.long(owner.0);
                writer.byte(group.0);
                writer.stamps(after);
            }
            Self::Evict { keys } => {
                writer.byte(8);
                writer.keys(keys);
            }
            Self::Victims { region, keys } => {
                writer.byte(9);
                writer.byte(*region);
                writer.keys(keys);
            }
            Self::Load {
                owner,
                class,
                at,
                keys,
            } => {
                writer.byte(10);
                writer.long(owner.0);
                writer.byte(*class);
                writer.word(*at);
                writer.keys(keys);
            }
            Self::Store { group, keys, tier } => {
                writer.byte(11);
                writer.byte(group.0);
                writer.keys(keys);
                writer.byte(tier_byte(*tier));
            }
            Self::Save { owner, key, units } => {
                writer.byte(12);
                writer.long(owner.0);
                writer.key(key);
                writer.word(*units);
            }
            Self::Restore { owner, key } => {
                writer.byte(13);
                writer.long(owner.0);
                writer.key(key);
            }
            Self::Finish { copy, ok, after } => {
                writer.byte(14);
                writer.long(copy.0);
                writer.byte(u8::from(*ok));
                writer.stamps(after);
            }
            Self::Prefetch {
                class,
                units,
                keys,
                source,
                deadline,
            } => {
                writer.byte(15);
                writer.byte(*class);
                writer.word(*units);
                writer.keys(keys);
                writer.byte(match source {
                    Source::Remote => 0,
                    Source::Peers => 1,
                });
                writer.long(*deadline);
            }
            Self::List => writer.byte(16),
            Self::Resize { region, units } => {
                writer.byte(17);
                writer.byte(*region);
                writer.word(*units);
            }
        }
    }

    /// Reads a request and its number from a record.
    ///
    /// # Errors
    ///
    /// Fails when the record is not a request of this version.
    pub fn decode(record: &[u8]) -> Result<(RequestId, Self)> {
        let mut reader = Reader(record);
        let id = RequestId(reader.long()?);
        let request = match reader.byte()? {
            0 => Self::Attach {
                owner: OwnerId(reader.long()?),
                group: GroupId(reader.byte()?),
            },
            1 => Self::Detach {
                owner: OwnerId(reader.long()?),
                after: reader.stamps()?,
            },
            2 => Self::Allocate {
                owner: OwnerId(reader.long()?),
                class: reader.byte()?,
                at: reader.word()?,
                count: reader.word()?,
                units: reader.word()?,
            },
            3 => Self::Pin {
                owner: OwnerId(reader.long()?),
                class: reader.byte()?,
                at: reader.word()?,
                keys: reader.keys()?,
            },
            4 => Self::Release {
                owner: OwnerId(reader.long()?),
                class: reader.byte()?,
                at: reader.word()?,
                count: reader.word()?,
                after: reader.stamps()?,
            },
            5 => Self::Commit {
                owner: OwnerId(reader.long()?),
                class: reader.byte()?,
                at: reader.word()?,
                key: reader.key()?,
                tokens: reader.short()?,
                after: reader.stamps()?,
            },
            6 => Self::Clone {
                owner: OwnerId(reader.long()?),
                class: reader.byte()?,
                at: reader.word()?,
                key: reader.key()?,
            },
            7 => Self::Transfer {
                owner: OwnerId(reader.long()?),
                group: GroupId(reader.byte()?),
                after: reader.stamps()?,
            },
            8 => Self::Evict {
                keys: reader.keys()?,
            },
            9 => Self::Victims {
                region: reader.byte()?,
                keys: reader.keys()?,
            },
            10 => Self::Load {
                owner: OwnerId(reader.long()?),
                class: reader.byte()?,
                at: reader.word()?,
                keys: reader.keys()?,
            },
            11 => Self::Store {
                group: GroupId(reader.byte()?),
                keys: reader.keys()?,
                tier: match reader.byte()? {
                    1 => Tier::Host,
                    2 => Tier::Remote,
                    other => return Err(Error::Protocol(format!("a store names tier {other}"))),
                },
            },
            12 => Self::Save {
                owner: OwnerId(reader.long()?),
                key: reader.key()?,
                units: reader.word()?,
            },
            13 => Self::Restore {
                owner: OwnerId(reader.long()?),
                key: reader.key()?,
            },
            14 => Self::Finish {
                copy: RequestId(reader.long()?),
                ok: reader.flag()?,
                after: reader.stamps()?,
            },
            15 => Self::Prefetch {
                class: reader.byte()?,
                units: reader.word()?,
                keys: reader.keys()?,
                source: match reader.byte()? {
                    0 => Source::Remote,
                    1 => Source::Peers,
                    other => {
                        return Err(Error::Protocol(format!("a prefetch names source {other}")));
                    }
                },
                deadline: reader.long()?,
            },
            16 => Self::List,
            17 => Self::Resize {
                region: reader.byte()?,
                units: reader.word()?,
            },
            other => {
                return Err(Error::Protocol(format!(
                    "request variant {other} is not of version {VERSION}"
                )));
            }
        };
        reader.end()?;
        Ok((id, request))
    }
}

impl Completion {
    /// Appends the record of this completion of request `id` to `record`.
    pub fn encode(&self, id: RequestId, record: &mut Vec<u8>) {
        let mut writer = Writer(record);
        writer.long(id.0);
        match self {
            Self::Done => writer.byte(0),
            Self::Refused { cause } => {
                writer.byte(1);
                writer.byte(cause_byte(*cause));
            }
            Self::Full { region } => {
                writer.byte(2);
                writer.byte(*region);
            }
            Self::Attached { slot } => {
                writer.byte(3);
                writer.word(*slot);
            }
            Self::Allocated { runs, evicted } => {
                writer.byte(4);
                writer.runs(runs);
                writer.keys(evicted);
            }
            Self::Pinned { runs, host } => {
                writer.byte(5);
                writer.runs(runs);
                writer.word(*host);
            }
            Self::Committed => writer.byte(6),
            Self::Duplicate => writer.byte(7),
            Self::Copy {
                group,
                keys,
                from,
                to,
                evicted,
            } => {
                writer.byte(8);
                writer.byte(group.0);
                writer.keys(keys);
                writer.runs(from);
                writer.runs(to);
                writer.keys(evicted);
            }
            Self::Cloned => writer.byte(9),
            Self::Loaded => writer.byte(10),
            Self::Stored { keys } => {
                writer.byte(11);
                writer.keys(keys);
            }
            Self::Saved => writer.byte(12),
            Self::Restored => writer.byte(13),
            Self::Transferred { slot, runs } => {
                writer.byte(14);
                writer.word(*slot);
                writer.word(runs.len() as u32);
                for (class, at, run, private) in runs {
                    writer.byte(*class);
                    writer.word(*at);
                    writer.run(run);
                    writer.byte(u8::from(*private));
                }
            }
            Self::Failed => writer.byte(15),
            Self::Evicted { keys } => {
                writer.byte(16);
                writer.keys(keys);
            }
            Self::Prefetched { found } => {
                writer.byte(17);
                writer.word(*found);
            }
            Self::Listing { keys } => {
                writer.byte(18);
                writer.word(keys.len() as u32);
                for (key, class, tokens, tiers, groups) in keys {
                    writer.key(key);
                    writer.byte(*class);
                    writer.short(*tokens);
                    writer.byte(*tiers);
                    writer.byte(*groups);
                }
            }
            Self::Resized { units, evicted } => {
                writer.byte(19);
                writer.word(*units);
                writer.keys(evicted);
            }
        }
    }

    /// Reads a completion and the number of the request it answers from a record.
    ///
    /// # Errors
    ///
    /// Fails when the record is not a completion of this version.
    pub fn decode(record: &[u8]) -> Result<(RequestId, Self)> {
        let mut reader = Reader(record);
        let id = RequestId(reader.long()?);
        let completion = match reader.byte()? {
            0 => Self::Done,
            1 => Self::Refused {
                cause: match reader.byte()? {
                    0 => Cause::Owner,
                    1 => Cause::Position,
                    2 => Cause::Key,
                    3 => Cause::Group,
                    4 => Cause::Class,
                    5 => Cause::Size,
                    6 => Cause::Copy,
                    7 => Cause::Request,
                    other => {
                        return Err(Error::Protocol(format!("a refusal states cause {other}")));
                    }
                },
            },
            2 => Self::Full {
                region: reader.byte()?,
            },
            3 => Self::Attached {
                slot: reader.word()?,
            },
            4 => Self::Allocated {
                runs: reader.runs()?,
                evicted: reader.keys()?,
            },
            5 => Self::Pinned {
                runs: reader.runs()?,
                host: reader.word()?,
            },
            6 => Self::Committed,
            7 => Self::Duplicate,
            8 => Self::Copy {
                group: GroupId(reader.byte()?),
                keys: reader.keys()?,
                from: reader.runs()?,
                to: reader.runs()?,
                evicted: reader.keys()?,
            },
            9 => Self::Cloned,
            10 => Self::Loaded,
            11 => Self::Stored {
                keys: reader.keys()?,
            },
            12 => Self::Saved,
            13 => Self::Restored,
            14 => Self::Transferred {
                slot: reader.word()?,
                runs: (0..reader.length(15)?)
                    .map(|_| {
                        Ok((
                            reader.byte()?,
                            reader.word()?,
                            reader.run()?,
                            reader.flag()?,
                        ))
                    })
                    .collect::<Result<_>>()?,
            },
            15 => Self::Failed,
            16 => Self::Evicted {
                keys: reader.keys()?,
            },
            17 => Self::Prefetched {
                found: reader.word()?,
            },
            18 => Self::Listing {
                keys: (0..reader.length(37)?)
                    .map(|_| {
                        Ok((
                            reader.key()?,
                            reader.byte()?,
                            reader.short()?,
                            reader.byte()?,
                            reader.byte()?,
                        ))
                    })
                    .collect::<Result<_>>()?,
            },
            19 => Self::Resized {
                units: reader.word()?,
                evicted: reader.keys()?,
            },
            other => {
                return Err(Error::Protocol(format!(
                    "completion variant {other} is not of version {VERSION}"
                )));
            }
        };
        reader.end()?;
        Ok((id, completion))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(byte: u8) -> Key {
        Key([byte; 32])
    }

    #[test]
    fn every_request_reads_back_as_it_was_written() {
        let after = vec![(GroupId(0), 7), (GroupId(3), u64::MAX)];
        let requests = [
            Request::Attach {
                owner: OwnerId(1),
                group: GroupId(2),
            },
            Request::Detach {
                owner: OwnerId(u64::MAX),
                after: after.clone(),
            },
            Request::Allocate {
                owner: OwnerId(1),
                class: 4,
                at: 9,
                count: 3,
                units: 0,
            },
            Request::Pin {
                owner: OwnerId(1),
                class: 0,
                at: 0,
                keys: vec![key(1), key(2)],
            },
            Request::Release {
                owner: OwnerId(1),
                class: 1,
                at: 2,
                count: 5,
                after: vec![],
            },
            Request::Commit {
                owner: OwnerId(1),
                class: 0,
                at: 4,
                key: key(9),
                tokens: 64,
                after: after.clone(),
            },
            Request::Clone {
                owner: OwnerId(1),
                class: 0,
                at: 4,
                key: key(9),
            },
            Request::Transfer {
                owner: OwnerId(1),
                group: GroupId(1),
                after: after.clone(),
            },
            Request::Evict { keys: vec![] },
            Request::Victims {
                region: 2,
                keys: vec![key(3)],
            },
            Request::Load {
                owner: OwnerId(1),
                class: 0,
                at: 1,
                keys: vec![key(4)],
            },
            Request::Store {
                group: GroupId(0),
                keys: vec![key(5)],
                tier: Tier::Remote,
            },
            Request::Save {
                owner: OwnerId(1),
                key: key(6),
                units: 12,
            },
            Request::Restore {
                owner: OwnerId(1),
                key: key(6),
            },
            Request::Finish {
                copy: RequestId(77),
                ok: false,
                after,
            },
            Request::Prefetch {
                class: 0,
                units: 0,
                keys: vec![key(7), key(8)],
                source: Source::Peers,
                deadline: 123_456_789,
            },
            Request::List,
            Request::Resize {
                region: 1,
                units: 4096,
            },
        ];

        for (index, request) in requests.iter().enumerate() {
            let mut record = Vec::new();
            request.encode(RequestId(index as u64 + 10), &mut record);

            assert_eq!(number(&record), Some(RequestId(index as u64 + 10)));
            assert_eq!(
                Request::decode(&record).unwrap(),
                (RequestId(index as u64 + 10), request.clone())
            );
        }
    }

    #[test]
    fn every_completion_reads_back_as_it_was_written() {
        let completions = [
            Completion::Done,
            Completion::Refused { cause: Cause::Copy },
            Completion::Full { region: 3 },
            Completion::Attached { slot: 5 },
            Completion::Allocated {
                runs: vec![(0, 7, 1), (0, 9, 1)],
                evicted: vec![key(1)],
            },
            Completion::Pinned {
                runs: vec![(0, 2, 1)],
                host: 3,
            },
            Completion::Committed,
            Completion::Duplicate,
            Completion::Copy {
                group: GroupId(1),
                keys: vec![key(2)],
                from: vec![(4, 0, 1)],
                to: vec![(0, 3, 1)],
                evicted: vec![],
            },
            Completion::Cloned,
            Completion::Loaded,
            Completion::Stored {
                keys: vec![key(3), key(4)],
            },
            Completion::Saved,
            Completion::Restored,
            Completion::Transferred {
                slot: 2,
                runs: vec![(0, 0, (0, 5, 1), true), (0, 1, (0, 6, 1), false)],
            },
            Completion::Failed,
            Completion::Evicted { keys: vec![key(5)] },
            Completion::Prefetched { found: 2 },
            Completion::Listing {
                keys: vec![(key(6), 0, 64, 0b011, 0b01)],
            },
            Completion::Resized {
                units: 100,
                evicted: vec![key(7)],
            },
        ];

        for completion in &completions {
            let mut record = Vec::new();
            completion.encode(RequestId(42), &mut record);

            assert_eq!(
                Completion::decode(&record).unwrap(),
                (RequestId(42), completion.clone())
            );
        }
    }

    #[test]
    fn a_record_that_is_not_of_this_version_is_an_error() {
        let mut record = Vec::new();
        Request::Evict { keys: vec![key(1)] }.encode(RequestId(1), &mut record);

        // Cut inside a field, extended past the last field, an unknown variant,
        // and a list that states more elements than the record holds.
        assert!(Request::decode(&record[..record.len() - 1]).is_err());
        let mut longer = record.clone();
        longer.push(0);
        assert!(Request::decode(&longer).is_err());
        let mut unknown = record.clone();
        unknown[8] = 200;
        assert!(Request::decode(&unknown).is_err());
        let mut many = record.clone();
        many[9..13].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Request::decode(&many).is_err());

        // The number of a record is readable whatever follows it.
        assert_eq!(number(&unknown), Some(RequestId(1)));
        assert_eq!(number(&record[..4]), None);
    }
}
