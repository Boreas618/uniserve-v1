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

use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use serde::de::Error as _;
use serde::ser::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use storage::protocol::Key;
use storage::runtime::{Completion, GroupId, OwnerId, RequestId, Segments};

use super::{
    Block, BufferPool, Class, ClientId, Copy, Deferred, Entry, Group, Held, Owner, QueueId, Region,
};

/// The image of a pool, over the regions `R` and the blocks `B`: borrowed
/// from a pool to write it, owned to make a pool of it. Maps keyed by more
/// than a number are lists of pairs.
#[derive(Serialize, Deserialize)]
struct Image<R, B> {
    classes: Vec<(ClientId, Vec<Class>, u8)>,
    regions: R,
    spaces: Vec<String>,
    memories: Vec<String>,
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
        };
        image.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BufferPool {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let image = Image::<Vec<Region>, Vec<Block>>::deserialize(deserializer)?;
        let blocks: Vec<Rc<Block>> = image.blocks.into_iter().map(Rc::new).collect();
        let mut block = |index: &u32| {
            blocks
                .get(*index as usize)
                .map(Rc::clone)
                .ok_or_else(|| D::Error::custom(format!("the image holds no block {index}")))
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
        pool.trial = image.trial.into_iter().collect();
        pool.numbers = image.numbers.into_iter().collect();
        pool.now = image.now;
        for (queue, record) in image.completions {
            let (request, completion) = Completion::decode(&record).map_err(D::Error::custom)?;
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
