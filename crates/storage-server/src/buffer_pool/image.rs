//! The pool's whole state as an image: what a restart in place hands to
//! its successor (`specs/buffer-pool.md` §14.1), and what the enumeration
//! of request sequences branches a state from.
//!
//! A block is shared by its holders, so an image lists every block once
//! and every holder names its block by the block's index in that list: the
//! types that hold blocks are the pool's own, with an index for the
//! reference. Serializing the pool writes its image; deserializing an image
//! makes the pool again, each block once, shared by every holder that names
//! it, so [`BufferPool::check`] holds of the pool made again when it held
//! of the pool the image was taken of. The pool an image was taken of is
//! not used afterwards, save that the enumeration goes on from both.
//!
//! An image is in the format of the build that wrote it ([`FORMAT`]), which
//! a restart in place states beside it. A successor reads its own build's
//! format and those before ([`format2`], [`format1`]): the server it
//! succeeds may be of an earlier build, and its sessions go on in the
//! versions of the interface they were opened with.

use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use serde::de::Error as _;
use serde::ser::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use storage::protocol::Key;
use storage::runtime::{Completion, GroupId, OwnerId, RequestId, Segments, VERSION};

use super::{
    Block, BufferPool, Class, ClientId, Copy, Deferred, Entry, Group, Held, Owner, QueueId, Region,
};

/// The format of the images this build writes: with what version 3 of the
/// runtime interface added and the version each session speaks, both of
/// which format 1 lacks, and with its completions not yet written as records
/// of this build's version ([`VERSION`]), which format 2 wrote in version 4.
pub(crate) const FORMAT: u32 = 3;

/// The image of a pool, over the regions `R` and the blocks `B`: borrowed
/// from a pool to write it, owned to make a pool of it. Maps keyed by more
/// than a number are lists of pairs.
#[derive(Serialize, Deserialize)]
struct Image<R, B> {
    classes: Vec<(ClientId, Vec<Class>, u8)>,
    regions: R,
    spaces: Vec<String>,
    memories: Vec<(String, Option<u32>)>,
    quota: Vec<BTreeMap<u64, u64>>,
    segments: Vec<Segments>,
    /// Every block a holder holds, once.
    blocks: B,
    cache: Vec<(Key, Entry<u32>)>,
    owners: Vec<(ClientId, u64, Owner<u32>)>,
    groups: Vec<(ClientId, GroupId, Group<u32>)>,
    copies: Vec<(QueueId, u64, Vec<Copy<u32>>)>,
    answers: Vec<(QueueId, u64, Vec<Key>)>,
    sessions: Vec<(ClientId, u64)>,
    trial: Vec<(u64, bool)>,
    numbers: Vec<(QueueId, (u64, u64))>,
    limits: (u64, u64),
    now: u64,
    /// The completions not yet written, each as its record
    /// ([`Completion::encode`]).
    completions: Vec<(QueueId, Vec<u8>)>,
    started: Vec<(QueueId, u64)>,
    stopped: Vec<(QueueId, u64)>,
    peers: Vec<(u64, Vec<u32>)>,
    /// The version of the runtime interface of each session admitted from
    /// its `Open`.
    versions: Vec<(ClientId, u32)>,
}

impl Serialize for BufferPool {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Every held block once, numbered in the order the holders are visited.
        let held = self.blocks();
        let index: HashMap<*const Block, u32> = held
            .iter()
            .enumerate()
            .map(|(number, (block, _, _))| (Rc::as_ptr(block), number as u32))
            .collect();
        if held.len() > u32::MAX as usize {
            return Err(S::Error::custom(
                "the pool holds too many blocks for an image",
            ));
        }
        // `blocks` visits every holder, so every block a holder holds is named.
        let mut name = |block: &Rc<Block>| {
            let number = index.get(&Rc::as_ptr(block)).copied();
            number.ok_or("a holder holds a block the pool does not list")
        };

        let mut cache = Vec::with_capacity(self.cache.len());
        for (key, entry) in &self.cache {
            cache.push((*key, named::<_, S::Error>(entry.map(&mut name))?));
        }
        let mut owners = Vec::with_capacity(self.owners.len());
        for ((client, owner), found) in &self.owners {
            owners.push((
                *client,
                owner.0,
                named::<_, S::Error>(found.map(&mut name))?,
            ));
        }
        let mut groups = Vec::with_capacity(self.groups.len());
        for ((client, group), found) in &self.groups {
            groups.push((*client, *group, named::<_, S::Error>(found.map(&mut name))?));
        }
        let mut copies = Vec::with_capacity(self.copies.len());
        for ((queue, request), found) in &self.copies {
            let mut mapped = Vec::with_capacity(found.len());
            for copy in found {
                mapped.push(named::<_, S::Error>(copy.map(&mut name))?);
            }
            copies.push((*queue, request.0, mapped));
        }
        let mut peers = Vec::with_capacity(self.peers.len());
        for (peer, blocks) in &self.peers {
            let mut mapped = Vec::with_capacity(blocks.len());
            for block in blocks {
                mapped.push(named::<_, S::Error>(name(block))?);
            }
            peers.push((*peer, mapped));
        }

        let completions = self.completions.iter().map(|(queue, request, completion)| {
            let mut record = Vec::new();
            completion.encode(*request, &mut record);
            (*queue, record)
        });
        let image = Image {
            classes: self
                .classes
                .iter()
                .map(|(client, (classes, trajectory))| (*client, classes.clone(), *trajectory))
                .collect(),
            regions: &self.regions,
            spaces: self.spaces.clone(),
            memories: self.memories.clone(),
            quota: self.quota.clone(),
            segments: self.segments.clone(),
            blocks: held
                .iter()
                .map(|(block, _, _)| Rc::as_ref(*block))
                .collect::<Vec<&Block>>(),
            cache,
            owners,
            groups,
            copies,
            answers: self
                .answers
                .iter()
                .map(|((queue, request), keys)| (*queue, request.0, keys.clone()))
                .collect(),
            sessions: self
                .sessions
                .iter()
                .map(|(client, epoch)| (*client, *epoch))
                .collect(),
            trial: self
                .trial
                .iter()
                .map(|(epoch, trial)| (*epoch, *trial))
                .collect(),
            numbers: self
                .numbers
                .iter()
                .map(|(queue, numbers)| (*queue, *numbers))
                .collect(),
            limits: self.limits,
            now: self.now,
            completions: completions.collect(),
            started: self
                .started
                .iter()
                .map(|(queue, id)| (*queue, id.0))
                .collect(),
            stopped: self
                .stopped
                .iter()
                .map(|(queue, id)| (*queue, id.0))
                .collect(),
            peers,
            versions: self
                .versions
                .iter()
                .map(|(client, version)| (*client, *version))
                .collect(),
        };
        image.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BufferPool {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let image = Image::<Vec<Region>, Vec<Block>>::deserialize(deserializer)?;
        Self::assemble(image, VERSION).map_err(D::Error::custom)
    }
}

impl BufferPool {
    /// The pool an owned image states, whose completions not yet written are
    /// records of the interface's version `records`.
    ///
    /// # Errors
    ///
    /// A holder that names a block the image does not list, or a record
    /// that is no completion.
    fn assemble(image: Image<Vec<Region>, Vec<Block>>, records: u32) -> Result<Self, String> {
        let blocks: Vec<Rc<Block>> = image.blocks.into_iter().map(Rc::new).collect();
        let mut block = |index: &u32| {
            blocks
                .get(*index as usize)
                .map(Rc::clone)
                .ok_or_else(|| format!("the image holds no block {index}"))
        };

        let mut pool = Self::new(image.limits);
        pool.classes = image
            .classes
            .into_iter()
            .map(|(client, classes, trajectory)| (client, (classes, trajectory)))
            .collect();
        pool.regions = image.regions;
        pool.spaces = image.spaces;
        pool.memories = image.memories;
        pool.quota = image.quota;
        pool.segments = image.segments;
        for (key, entry) in image.cache {
            pool.cache.insert(key, entry.map(&mut block)?);
        }
        for (client, owner, found) in image.owners {
            pool.owners
                .insert((client, OwnerId(owner)), found.map(&mut block)?);
        }
        for (client, group, found) in image.groups {
            pool.groups.insert((client, group), found.map(&mut block)?);
        }
        for (queue, request, found) in image.copies {
            let mut copies = Vec::with_capacity(found.len());
            for copy in found {
                copies.push(copy.map(&mut block)?);
            }
            pool.copies.insert((queue, RequestId(request)), copies);
        }
        for (peer, held) in image.peers {
            let mut holds = Vec::with_capacity(held.len());
            for index in &held {
                holds.push(block(index)?);
            }
            pool.peers.insert(peer, holds);
        }
        pool.answers = image
            .answers
            .into_iter()
            .map(|(queue, request, keys)| ((queue, RequestId(request)), keys))
            .collect();
        pool.sessions = image.sessions.into_iter().collect();
        pool.versions = image.versions.into_iter().collect();
        pool.trial = image.trial.into_iter().collect();
        pool.numbers = image.numbers.into_iter().collect();
        pool.now = image.now;
        for (queue, record) in image.completions {
            let (request, completion) =
                Completion::decode_as(&record, records).map_err(|error| error.to_string())?;
            pool.completions.push((queue, request, completion));
        }
        pool.started = image
            .started
            .into_iter()
            .map(|(queue, id)| (queue, RequestId(id)))
            .collect();
        pool.stopped = image
            .stopped
            .into_iter()
            .map(|(queue, id)| (queue, RequestId(id)))
            .collect();
        // The image leaves out the counts the capacity words are made of,
        // which follow from the holders' references alone.
        drop(blocks);
        pool.recount();
        Ok(pool)
    }
}

/// A block's number, or why a holder's block has none.
fn named<T, E: serde::ser::Error>(result: Result<T, &str>) -> Result<T, E> {
    result.map_err(E::custom)
}

impl<B> Held<B> {
    /// The same entry, its block named by `name`.
    fn map<C, E>(&self, name: &mut impl FnMut(&B) -> Result<C, E>) -> Result<Held<C>, E> {
        Ok(match self {
            Self::Private(block) => Held::Private(name(block)?),
            Self::Shared(block) => Held::Shared(name(block)?),
        })
    }
}

impl<B> Owner<B> {
    /// The same owner, its blocks named by `name`.
    fn map<C, E>(&self, name: &mut impl FnMut(&B) -> Result<C, E>) -> Result<Owner<C>, E> {
        let mut tables = Vec::with_capacity(self.tables.len());
        for table in &self.tables {
            let mut mapped = BTreeMap::new();
            for (position, held) in table {
                mapped.insert(*position, held.map(name)?);
            }
            tables.push(mapped);
        }
        Ok(Owner {
            group: self.group,
            slot: name(&self.slot)?,
            tables,
            moving: self.moving,
        })
    }
}

impl<B> Entry<B> {
    /// The same entry of the cache, its blocks named by `name`.
    fn map<C, E>(&self, name: &mut impl FnMut(&B) -> Result<C, E>) -> Result<Entry<C>, E> {
        let mut device = Vec::with_capacity(self.device.len());
        for (block, maker) in &self.device {
            device.push((name(block)?, *maker));
        }
        let host = match &self.host {
            Some((block, maker, encoding)) => Some((name(block)?, *maker, *encoding)),
            None => None,
        };
        Ok(Entry {
            class: self.class,
            device,
            host,
            tokens: self.tokens,
            remote: self.remote,
        })
    }
}

impl<B> Deferred<B> {
    /// The same deferred work, its blocks named by `name`.
    fn map<C, E>(&self, name: &mut impl FnMut(&B) -> Result<C, E>) -> Result<Deferred<C>, E> {
        Ok(match self {
            Self::Release { blocks } => {
                let mut mapped = Vec::with_capacity(blocks.len());
                for block in blocks {
                    mapped.push(name(block)?);
                }
                Deferred::Release { blocks: mapped }
            }
            Self::Commit {
                queue,
                request,
                owner,
                class,
                at,
                block,
                key,
                tokens,
                rest,
            } => Deferred::Commit {
                queue: *queue,
                request: *request,
                owner: *owner,
                class: *class,
                at: *at,
                block: name(block)?,
                key: *key,
                tokens: *tokens,
                rest: rest.clone(),
            },
            Self::Transfer {
                queue,
                request,
                owner,
                to,
                rest,
            } => Deferred::Transfer {
                queue: *queue,
                request: *request,
                owner: *owner,
                to: *to,
                rest: rest.clone(),
            },
            Self::Finish {
                queue,
                request,
                copy,
                ok,
                rest,
            } => Deferred::Finish {
                queue: *queue,
                request: *request,
                copy: *copy,
                ok: *ok,
                rest: rest.clone(),
            },
            Self::Replicate {
                queue,
                request,
                owner,
                class,
                at,
                block,
                from,
                to,
                rest,
            } => Deferred::Replicate {
                queue: *queue,
                request: *request,
                owner: *owner,
                class: *class,
                at: *at,
                block: name(block)?,
                from: *from,
                to: *to,
                rest: rest.clone(),
            },
        })
    }
}

impl<B> Group<B> {
    /// The same group, the blocks of its deferred work named by `name`.
    fn map<C, E>(&self, name: &mut impl FnMut(&B) -> Result<C, E>) -> Result<Group<C>, E> {
        let mut deferred = std::collections::VecDeque::with_capacity(self.deferred.len());
        for (stamp, work) in &self.deferred {
            deferred.push_back((*stamp, work.map(name)?));
        }
        Ok(Group {
            memory: self.memory,
            slots: self.slots,
            regions: self.regions.clone(),
            receives: self.receives.clone(),
            parts: self.parts.clone(),
            completed: self.completed,
            deferred,
        })
    }
}

impl<B> Copy<B> {
    /// The same copy, its blocks named by `name`.
    fn map<C, E>(&self, name: &mut impl FnMut(&B) -> Result<C, E>) -> Result<Copy<C>, E> {
        let mut reads = Vec::with_capacity(self.reads.len());
        for block in &self.reads {
            reads.push(name(block)?);
        }
        Ok(Copy {
            memory: self.memory,
            reads,
            write: name(&self.write)?,
            to: self.to,
            fill: self.fill,
            deadline: self.deadline,
        })
    }
}

/// A request's number in an image.
pub(super) mod request_id {
    use serde::{Deserialize, Deserializer, Serializer};
    use storage::runtime::RequestId;

    pub(in crate::buffer_pool) fn serialize<S: Serializer>(
        id: &RequestId,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(id.0)
    }

    pub(in crate::buffer_pool) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<RequestId, D::Error> {
        u64::deserialize(deserializer).map(RequestId)
    }
}

/// An owner's number in an image.
pub(super) mod owner_id {
    use serde::{Deserialize, Deserializer, Serializer};
    use storage::runtime::OwnerId;

    pub(in crate::buffer_pool) fn serialize<S: Serializer>(
        id: &OwnerId,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(id.0)
    }

    pub(in crate::buffer_pool) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<OwnerId, D::Error> {
        u64::deserialize(deserializer).map(OwnerId)
    }
}

/// Format 2: this build's image as the builds of version 4 of the runtime
/// interface wrote it, whose completions not yet written are records of
/// version 4: a crossing's answer names no evicted keys.
pub(crate) mod format2 {
    use serde::Deserialize;

    use super::super::{Block, BufferPool, Region};
    use super::Image as Current;

    /// This format's number.
    pub(crate) const FORMAT: u32 = 2;

    /// The version of the runtime interface the completions of format 2 are
    /// records of.
    const VERSION: u32 = 4;

    /// An image of format 2: this build's fields.
    #[derive(Deserialize)]
    #[serde(transparent)]
    pub(crate) struct Image(Current<Vec<Region>, Vec<Block>>);

    impl TryFrom<Image> for BufferPool {
        type Error = String;

        /// The pool an image of format 2 states, its completions read as
        /// records of version 4.
        fn try_from(image: Image) -> Result<Self, String> {
            Self::assemble(image.0, VERSION)
        }
    }
}

/// Format 1: the image of the builds before images stated their format,
/// which served version 2 of the runtime interface. It lacks what version 3
/// added: the memory a region lies in, a group's receiving regions and host
/// parts, the host parts among the memories, and a session's version, which
/// is 2. A memory is a device memory's name, a block's validity a byte in
/// which every bit set (255) was a block valid in every memory that maps it
/// (a slot row, a product of several regions), and the completions not yet
/// written are records of version 2.
pub(crate) mod format1 {
    use std::collections::{BTreeMap, VecDeque};

    use serde::Deserialize;
    use storage::protocol::Key;
    use storage::runtime::{GroupId, Segments};

    use super::super::region::format1::Region;
    use super::super::{Block, BufferPool, Class, ClientId, Copy, Deferred, Entry, Owner, QueueId};
    use super::Image as Current;

    /// This format's number.
    pub(crate) const FORMAT: u32 = 1;

    /// The validity format 1 gives a block valid in every memory that maps it.
    const EVERYWHERE: u64 = u8::MAX as u64;

    /// The version of the runtime interface every session of format 1 speaks.
    const VERSION: u32 = 2;

    /// An image of format 1, its fields in their order there.
    #[derive(Deserialize)]
    pub(crate) struct Image {
        classes: Vec<(ClientId, Vec<Class>, u8)>,
        regions: Vec<Region>,
        spaces: Vec<String>,
        memories: Vec<String>,
        quota: Vec<BTreeMap<u64, u64>>,
        segments: Vec<Segments>,
        blocks: Vec<Block>,
        cache: Vec<(Key, Entry<u32>)>,
        owners: Vec<(ClientId, u64, Owner<u32>)>,
        groups: Vec<(ClientId, GroupId, Group)>,
        copies: Vec<(QueueId, u64, Vec<Copy<u32>>)>,
        answers: Vec<(QueueId, u64, Vec<Key>)>,
        sessions: Vec<(ClientId, u64)>,
        trial: Vec<(u64, bool)>,
        numbers: Vec<(QueueId, (u64, u64))>,
        limits: (u64, u64),
        now: u64,
        completions: Vec<(QueueId, Vec<u8>)>,
        started: Vec<(QueueId, u64)>,
        stopped: Vec<(QueueId, u64)>,
        peers: Vec<(u64, Vec<u32>)>,
    }

    /// A worker group in format 1: no receiving region and no host part.
    #[derive(Deserialize)]
    struct Group {
        memory: u8,
        slots: u8,
        regions: Vec<Vec<u8>>,
        completed: u64,
        deferred: VecDeque<(u64, Deferred<u32>)>,
    }

    impl TryFrom<Image> for BufferPool {
        type Error = String;

        /// The pool an image of format 1 states, as this build holds it:
        /// every region's units exist in each memory that maps it, every
        /// memory is one of the device, a group copies with no host part, a
        /// block valid everywhere stays so, and every session speaks version 2.
        fn try_from(image: Image) -> Result<Self, String> {
            let blocks = image.blocks.into_iter().inspect(|block| {
                if block.valid.get() == EVERYWHERE {
                    block.valid.set(super::super::EVERYWHERE);
                }
            });
            let groups = image.groups.into_iter().map(|(client, id, group)| {
                let group = super::super::Group {
                    memory: group.memory,
                    slots: group.slots,
                    regions: group.regions,
                    receives: Vec::new(),
                    parts: Vec::new(),
                    completed: group.completed,
                    deferred: group.deferred,
                };
                (client, id, group)
            });
            let versions = image
                .sessions
                .iter()
                .map(|(client, _)| (*client, VERSION))
                .collect();
            let current = Current {
                classes: image.classes,
                regions: image.regions.into_iter().map(Into::into).collect(),
                spaces: image.spaces,
                memories: image
                    .memories
                    .into_iter()
                    .map(|name| (name, None))
                    .collect(),
                quota: image.quota,
                segments: image.segments,
                blocks: blocks.collect(),
                cache: image.cache,
                owners: image.owners,
                groups: groups.collect(),
                copies: image.copies,
                answers: image.answers,
                sessions: image.sessions,
                trial: image.trial,
                numbers: image.numbers,
                limits: image.limits,
                now: image.now,
                completions: image.completions,
                started: image.started,
                stopped: image.stopped,
                peers: image.peers,
                versions,
            };
            Self::assemble(current, VERSION)
        }
    }
}
