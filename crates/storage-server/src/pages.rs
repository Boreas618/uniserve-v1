//! Host page pools (L2): equal slots of one memfd per NUMA node and page
//! shape, and the state of every slot.
//!
//! The server never touches the slots' bytes. Workers map a pool and copy
//! pages between their device cache and the slots on their own streams, at
//! the head's direction; the server keeps which page each slot holds and
//! who may write or read it:
//!
//! | State | Meaning |
//! | --- | --- |
//! | free | Unused |
//! | reserved | Being written by the worker of its writer scope |
//! | fetching | Being written by the connector, from the KV storage |
//! | ready | Holds a committed page and may be read |
//! | pinned | Ready and being read by workers; not evictable |
//! | uploading | Ready and being read by the connector; not evictable |
//!
//! A reservation takes a free slot, or evicts the least recently used ready
//! slot nobody reads. A reservation may name its pages as one sliding
//! window of attention, which is useless once it misses a page: when one of
//! its pages leaves the pool, the window ends, and its other pages that no
//! other window holds and nobody reads leave with it. A window may follow a
//! page of another pool, its anchor (the last full-attention page before
//! the window's boundary, without which the window cannot be reused): when
//! the anchor leaves its pool, the window ends too. A head's reservations and pins end with the head's
//! `commit`, `abort` or `unpin`. When the head's connection closes first,
//! they end once no worker of their writer or reader scope is connected,
//! because that worker's copies may still run until its process ends. The
//! connector's fetches and uploads end when it reports them, or when its
//! connection closes.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io;
use std::os::fd::OwnedFd;

use serde::{Deserialize, Serialize};
use storage::protocol::{Key, ListedPool, PageAnchor, PageShape, PageSlot, PageWindow, Scope};

type ClientId = u64;

/// The bytes of a page pool unless the server is started with another
/// size. The workers that copy through a pool register it for device
/// copies, which pins it: its memory is allocated whole when the first
/// worker maps it.
pub const DEFAULT_POOL_BYTES: u64 = 64 << 30;

/// The host memory a rank's page pools take at most when the host's memory
/// holds none of the pools a world asks for and a KV storage keeps every
/// page the pools commit: staging pools, through which pages pass into the
/// KV storage on their way out of the device (`specs/online-operations.md`
/// §6.5). The server admits them from its reserve, never past the host's
/// available memory.
pub const STAGING_POOL_BYTES: u64 = 256 << 20;

/// One slot's state.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum Slot {
    Free,
    Reserved {
        key: Key,
        writer: Scope,
        /// The head that reserved the slot, while it is connected.
        head: Option<ClientId>,
        /// The world whose head reserved the slot, which publishes its page.
        #[serde(default)]
        world: u64,
    },
    /// Reserved for the connector, which reads the page from the
    /// KV storage.
    Fetching {
        key: Key,
    },
    Ready {
        key: Key,
        /// When the page was last committed, looked up or read.
        touched: u64,
        /// The readers pinning the page, each with the head that pinned it
        /// while that head is connected.
        pins: Vec<(Scope, Option<ClientId>)>,
        /// The connector's uploads of the page in flight.
        uploads: u32,
        /// The world that published the page (`Visibility`); zero for a
        /// page every world shares, such as one read from the KV storage.
        #[serde(default)]
        world: u64,
    },
}

/// The world a discarded page is published by: no world reads it, and it
/// leaves the pool once nobody reads it.
pub(super) const DISCARDED: u64 = u64::MAX;

/// Which pages a head reads: those its own world published, and those every
/// world of its deployment shares. A world on trial beside the world it
/// succeeds (`private`) keeps what it publishes to itself until its commit
/// retires its predecessor; a rollback discards it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Visibility {
    /// The world of the head that reads.
    pub(super) world: u64,
    /// The deployment's worlds on trial: claimed and newer than its oldest
    /// claimed world.
    pub(super) private: BTreeSet<u64>,
}

impl Visibility {
    /// Whether a page `published` by that world is read.
    pub(super) fn sees(&self, published: u64) -> bool {
        published == self.world || self.shared(published)
    }

    /// Whether every world of the deployment reads a page `published` by
    /// that world.
    pub(super) fn shared(&self, published: u64) -> bool {
        published != DISCARDED && !self.private.contains(&published)
    }
}

impl Slot {
    /// Whether a ready slot may be evicted: nobody reads it.
    fn idle(pins: &[(Scope, Option<ClientId>)], uploads: u32) -> bool {
        pins.is_empty() && uploads == 0
    }
}

/// What a prefetch made of one page key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Fetch {
    /// The pool holds the page, or a writer has it reserved.
    Present,
    /// The slot reserved for the connector to read the page into.
    Reserved(u32),
    /// Prefetches hold their share of the slots, or every slot is held.
    Full,
}

/// The pages of one sliding window, and what it follows.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Window {
    slots: Vec<u32>,
    anchor: Option<PageAnchor>,
}

/// One page pool.
pub(super) struct Pool {
    /// The deployment whose workers share the pool.
    pub(super) deployment: String,
    pub(super) shape: PageShape,
    pub(super) numa_node: Option<u32>,
    /// The memfd of `slots × page_bytes` bytes.
    pub(super) descriptor: OwnedFd,
    slots: Vec<Slot>,
    /// The slot of every reserved or ready page.
    index: HashMap<Key, u32>,
    /// Ready, unpinned slots by the tick they were last touched: the first
    /// entry is the least recently used.
    evictable: BTreeMap<u64, u32>,
    free: Vec<u32>,
    /// Slots reserved for the connector's fetches.
    fetching: u64,
    tick: u64,
    windows: HashMap<u64, Window>,
    /// The windows each slot's page belongs to.
    member_of: HashMap<u32, Vec<u64>>,
    next_window: u64,
    /// The keys whose pages left the pool, and the windows that ended,
    /// since [`Pools::settle`] last took them.
    removed: Vec<Key>,
    ended: Vec<(u64, Option<PageAnchor>)>,
    /// The scopes, by world, whose workers mapped the pool: a pool no
    /// claimed world maps is the server's to release.
    users: BTreeSet<(Scope, u64)>,
}

/// Every page pool of the server.
pub(super) struct Pools {
    pools: HashMap<u64, Pool>,
    /// The live pool of each deployment, page shape and NUMA node.
    by_shape: HashMap<(String, PageShape, Option<u32>), u64>,
    next: u64,
    /// The bytes of a new pool whose worker states none, rounded down to
    /// whole pages.
    pool_bytes: u64,
    /// The windows that follow each anchor page, by the anchor's pool and
    /// key: each window's pool and id.
    anchors: HashMap<(u64, Key), Vec<(u64, u64)>>,
    /// Released pools the connector still reads or writes: they leave once
    /// it finishes ([`Self::reap`]).
    releasing: BTreeSet<u64>,
}

/// A pool's state as a hot restart hands it to the successor server; its
/// memfd travels apart.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct HandedPool {
    id: u64,
    deployment: String,
    shape: PageShape,
    numa_node: Option<u32>,
    slots: Vec<Slot>,
    index: Vec<(Key, u32)>,
    evictable: Vec<(u64, u32)>,
    free: Vec<u32>,
    fetching: u64,
    tick: u64,
    windows: Vec<(u64, Window)>,
    member_of: Vec<(u32, Vec<u64>)>,
    next_window: u64,
    removed: Vec<Key>,
    ended: Vec<(u64, Option<PageAnchor>)>,
    #[serde(default)]
    users: Vec<(Scope, u64)>,
}

/// A pool's identity: its deployment, page shape and NUMA node.
type PoolIdentity = (String, PageShape, Option<u32>);

/// The windows that follow an anchor page (`Pools::anchors`).
type AnchorWindows = ((u64, Key), Vec<(u64, u64)>);

/// Every pool's state as a hot restart hands it over; each pool's memfd
/// travels apart, in `pools` order.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct HandedPools {
    pools: Vec<HandedPool>,
    by_shape: Vec<(PoolIdentity, u64)>,
    next: u64,
    pool_bytes: u64,
    anchors: Vec<AnchorWindows>,
    releasing: Vec<u64>,
}

impl Pools {
    /// Every pool's state and memfd, for a successor server to go on with
    /// ([`Pools::resume`]); this server holds no pool afterwards.
    pub(super) fn hand_over(&mut self) -> (HandedPools, Vec<OwnedFd>) {
        let mut descriptors = Vec::with_capacity(self.pools.len());
        let pools = self
            .pools
            .drain()
            .map(|(id, pool)| {
                descriptors.push(pool.descriptor);
                HandedPool {
                    id,
                    deployment: pool.deployment,
                    shape: pool.shape,
                    numa_node: pool.numa_node,
                    slots: pool.slots,
                    index: pool.index.into_iter().collect(),
                    evictable: pool.evictable.into_iter().collect(),
                    free: pool.free,
                    fetching: pool.fetching,
                    tick: pool.tick,
                    windows: pool.windows.into_iter().collect(),
                    member_of: pool.member_of.into_iter().collect(),
                    next_window: pool.next_window,
                    removed: pool.removed,
                    ended: pool.ended,
                    users: pool.users.into_iter().collect(),
                }
            })
            .collect();
        (
            HandedPools {
                pools,
                by_shape: self.by_shape.drain().collect(),
                next: self.next,
                pool_bytes: self.pool_bytes,
                anchors: self.anchors.drain().collect(),
                releasing: std::mem::take(&mut self.releasing).into_iter().collect(),
            },
            descriptors,
        )
    }

    /// Goes on with the pools a predecessor handed over, with their memfds
    /// in its order.
    ///
    /// # Errors
    ///
    /// Fails when a memfd is missing.
    pub(super) fn resume(handed: HandedPools, descriptors: Vec<OwnedFd>) -> io::Result<Self> {
        if handed.pools.len() != descriptors.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "a handed-over page pool lacks its memory",
            ));
        }
        let pools = handed
            .pools
            .into_iter()
            .zip(descriptors)
            .map(|(pool, descriptor)| {
                (
                    pool.id,
                    Pool {
                        deployment: pool.deployment,
                        shape: pool.shape,
                        numa_node: pool.numa_node,
                        descriptor,
                        slots: pool.slots,
                        index: pool.index.into_iter().collect(),
                        evictable: pool.evictable.into_iter().collect(),
                        free: pool.free,
                        fetching: pool.fetching,
                        tick: pool.tick,
                        windows: pool.windows.into_iter().collect(),
                        member_of: pool.member_of.into_iter().collect(),
                        next_window: pool.next_window,
                        removed: pool.removed,
                        ended: pool.ended,
                        users: pool.users.into_iter().collect(),
                    },
                )
            })
            .collect();
        Ok(Self {
            pools,
            by_shape: handed.by_shape.into_iter().collect(),
            next: handed.next,
            pool_bytes: handed.pool_bytes,
            anchors: handed.anchors.into_iter().collect(),
            releasing: handed.releasing.into_iter().collect(),
        })
    }

    pub(super) fn new(pool_bytes: u64) -> Self {
        Self {
            pools: HashMap::new(),
            by_shape: HashMap::new(),
            next: 1,
            pool_bytes,
            anchors: HashMap::new(),
            releasing: BTreeSet::new(),
        }
    }

    /// The live pool of `deployment` for pages of `shape` on `numa_node`.
    pub(super) fn find(
        &self,
        deployment: &str,
        shape: PageShape,
        numa_node: Option<u32>,
    ) -> Option<u64> {
        self.by_shape
            .get(&(deployment.to_owned(), shape, numa_node))
            .copied()
    }

    /// The slots of a new pool of `bytes` (the server's default when
    /// `None`) for pages of `page_bytes`: its whole pages.
    pub(super) fn slots_of(&self, page_bytes: u64, bytes: Option<u64>) -> io::Result<u64> {
        let bytes = bytes.unwrap_or(self.pool_bytes);
        if page_bytes == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a page pool needs pages of at least one byte",
            ));
        }
        let slots = bytes / page_bytes;
        if slots == 0 || slots > u64::from(u32::MAX) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("a page pool of {bytes} bytes cannot hold pages of {page_bytes} bytes"),
            ));
        }
        Ok(slots)
    }

    /// Create the pool of `deployment` for pages of `shape` on `numa_node`,
    /// of `slots` slots ([`Self::slots_of`]).
    pub(super) fn create(
        &mut self,
        deployment: &str,
        shape: PageShape,
        numa_node: Option<u32>,
        slots: u64,
    ) -> io::Result<u64> {
        let descriptor = shared_memory("uniserve-page-pool", slots * shape.page_bytes, numa_node)?;
        let pool = self.next;
        self.next += 1;
        self.pools.insert(
            pool,
            Pool {
                deployment: deployment.to_owned(),
                shape,
                numa_node,
                descriptor,
                slots: vec![Slot::Free; slots as usize],
                index: HashMap::new(),
                evictable: BTreeMap::new(),
                free: (0..slots as u32).rev().collect(),
                fetching: 0,
                tick: 0,
                windows: HashMap::new(),
                member_of: HashMap::new(),
                next_window: 1,
                removed: Vec::new(),
                ended: Vec::new(),
                users: BTreeSet::new(),
            },
        );
        self.by_shape
            .insert((deployment.to_owned(), shape, numa_node), pool);
        tracing::info!(
            pool,
            deployment,
            slots,
            page_bytes = shape.page_bytes,
            numa_node,
            "created a page pool"
        );
        Ok(pool)
    }

    /// The bytes the pools have yet to allocate: their sizes less what
    /// their memfds hold. A pool's pages are allocated when a worker first
    /// registers it, so a new pool's whole size is yet to come.
    pub(super) fn unallocated(&self) -> u64 {
        self.pools
            .values()
            .map(|pool| {
                let size = pool.slots() * pool.shape.page_bytes;
                let held = rustix::fs::fstat(&pool.descriptor)
                    .map_or(0, |stat| (stat.st_blocks as u64).saturating_mul(512));
                size.saturating_sub(held)
            })
            .sum()
    }

    /// `world`'s worker of `scope` maps `pool`.
    pub(super) fn add_user(&mut self, pool: u64, scope: &Scope, world: u64) {
        if let Some(entry) = self.pools.get_mut(&pool) {
            entry.users.insert((scope.clone(), world));
        }
    }

    /// `retired` worlds of `deployment` map no pool any more.
    pub(super) fn retire_users(&mut self, deployment: &str, retired: &BTreeSet<u64>) {
        for pool in self.pools.values_mut() {
            if pool.deployment == deployment {
                pool.users.retain(|(_, world)| !retired.contains(world));
            }
        }
    }

    /// The live pools of `deployment` no world maps, with their bytes.
    pub(super) fn unreferenced(&self, deployment: &str) -> Vec<(u64, u64)> {
        self.pools
            .iter()
            .filter(|(id, pool)| {
                pool.deployment == deployment
                    && pool.users.is_empty()
                    && !self.releasing.contains(id)
            })
            .map(|(&id, pool)| (id, pool.slots() * pool.shape.page_bytes))
            .collect()
    }

    /// Release `pools`, as [`Self::release`] releases a deployment's.
    /// Returns the pools that left at once.
    pub(super) fn release_pools(&mut self, pools: &[u64]) -> Vec<u64> {
        self.by_shape.retain(|_, id| !pools.contains(id));
        let live: Vec<u64> = pools
            .iter()
            .copied()
            .filter(|id| self.pools.contains_key(id))
            .collect();
        self.releasing.extend(live);
        self.reap()
    }

    /// Whether `pool` is released and waits for the connector to finish.
    pub(super) fn released(&self, pool: u64) -> bool {
        self.releasing.contains(&pool)
    }

    /// The deployments that own pools.
    pub(super) fn deployments(&self) -> BTreeSet<String> {
        self.pools
            .values()
            .map(|pool| pool.deployment.clone())
            .collect()
    }

    /// Release the pools of `deployment`: no worker maps them again. Those
    /// the connector has no page of in flight leave at once and are
    /// returned; the others leave when it finishes ([`Self::reap`]).
    pub(super) fn release(&mut self, deployment: &str) -> Vec<u64> {
        self.by_shape.retain(|(owner, _, _), _| owner != deployment);
        let mine: Vec<u64> = self
            .pools
            .iter()
            .filter(|(_, pool)| pool.deployment == deployment)
            .map(|(&id, _)| id)
            .collect();
        self.releasing.extend(mine);
        self.reap()
    }

    /// Remove the released pools the connector no longer reads or writes.
    /// Returns the pools removed.
    pub(super) fn reap(&mut self) -> Vec<u64> {
        let idle: Vec<u64> = self
            .releasing
            .iter()
            .copied()
            .filter(|id| self.pools.get(id).is_none_or(|pool| !pool.connector_busy()))
            .collect();
        for id in &idle {
            self.releasing.remove(id);
            if let Some(pool) = self.pools.remove(id) {
                tracing::info!(
                    pool = id,
                    deployment = pool.deployment,
                    bytes = pool.slots() * pool.shape.page_bytes,
                    "released a page pool"
                );
            }
        }
        if !idle.is_empty() {
            self.anchors.retain(|(pool, _), followers| {
                followers.retain(|(follower, _)| !idle.contains(follower));
                !idle.contains(pool) && !followers.is_empty()
            });
        }
        idle
    }

    pub(super) fn get(&self, pool: u64) -> Option<&Pool> {
        self.pools.get(&pool)
    }

    pub(super) fn get_mut(&mut self, pool: u64) -> Option<&mut Pool> {
        self.pools.get_mut(&pool)
    }

    /// Reserve a slot in `pool` for each of `keys` the pool holds no page
    /// of ([`Pool::reserve`]); with `window`, the keys are one sliding
    /// window. `None` for an unknown pool.
    pub(super) fn reserve(
        &mut self,
        pool: u64,
        writer: &Scope,
        head: ClientId,
        keys: &[Key],
        window: Option<PageWindow>,
        visibility: &Visibility,
    ) -> Option<Vec<PageSlot>> {
        let entry = self.pools.get_mut(&pool)?;
        let (slots, id) = entry.reserve(writer, head, keys, window, visibility);
        if let (Some(id), Some(anchor)) = (id, window.and_then(|window| window.anchor)) {
            self.anchors
                .entry((anchor.pool, anchor.key))
                .or_default()
                .push((pool, id));
        }
        self.settle();
        Some(slots)
    }

    /// Discard the pages `world` of `deployment` published while on trial
    /// ([`Pool::discard_where`]).
    pub(super) fn discard_world(&mut self, deployment: &str, world: u64) {
        self.discard_where(deployment, |published| published == world);
    }

    /// Discard every page of `deployment` that world `except` did not
    /// publish ([`Pool::discard_where`]); returns how many.
    pub(super) fn discard_except(&mut self, deployment: &str, except: u64) -> u64 {
        self.discard_where(deployment, |published| published != except)
    }

    fn discard_where(&mut self, deployment: &str, discarded: impl Fn(u64) -> bool) -> u64 {
        let mut pages = 0;
        for pool in self.pools.values_mut() {
            if pool.deployment == deployment {
                pages += pool.discard_where(&discarded);
            }
        }
        self.settle();
        pages
    }

    /// The pages `world` of `deployment` published, by pool.
    pub(super) fn published_by(&self, deployment: &str, world: u64) -> Vec<(u64, Vec<(Key, u32)>)> {
        self.pools
            .iter()
            .filter(|(id, pool)| pool.deployment == deployment && !self.releasing.contains(id))
            .map(|(&id, pool)| (id, pool.published_by(world)))
            .filter(|(_, pages)| !pages.is_empty())
            .collect()
    }

    /// End the windows whose anchors left their pools, and forget the
    /// windows that ended, until no more pages leave. Every operation that
    /// can take a page out of a pool runs this after it.
    pub(super) fn settle(&mut self) {
        loop {
            let mut removed = Vec::new();
            for (&id, pool) in &mut self.pools {
                removed.extend(pool.removed.drain(..).map(|key| (id, key)));
                for (window, anchor) in pool.ended.drain(..) {
                    let Some(anchor) = anchor else {
                        continue;
                    };
                    if let Some(followers) = self.anchors.get_mut(&(anchor.pool, anchor.key)) {
                        followers.retain(|follower| *follower != (id, window));
                        if followers.is_empty() {
                            self.anchors.remove(&(anchor.pool, anchor.key));
                        }
                    }
                }
            }
            if removed.is_empty() {
                return;
            }
            for anchor in removed {
                for (pool, window) in self.anchors.remove(&anchor).unwrap_or_default() {
                    if let Some(pool) = self.pools.get_mut(&pool) {
                        pool.end_window(window);
                    }
                }
            }
        }
    }

    /// End the reservations and pins of a head whose connection closed: at
    /// once for a scope no connected worker has, and otherwise when that
    /// worker's connection closes (`worker_gone`).
    pub(super) fn head_gone(&mut self, head: ClientId, connected: &dyn Fn(&Scope) -> bool) {
        for pool in self.pools.values_mut() {
            pool.orphan(head, connected);
        }
        self.settle();
    }

    /// End the orphaned reservations and pins of `scope`, whose last
    /// connected worker left.
    pub(super) fn worker_gone(&mut self, scope: &Scope) {
        for pool in self.pools.values_mut() {
            pool.release_orphans(scope);
        }
        self.settle();
    }

    /// The connector is gone: its fetches and uploads end.
    pub(super) fn connector_gone(&mut self) {
        for pool in self.pools.values_mut() {
            pool.connector_gone();
        }
        self.settle();
    }

    pub(super) fn listing(&self) -> Vec<ListedPool> {
        let mut pools: Vec<ListedPool> = self
            .pools
            .iter()
            .map(|(&pool, entry)| entry.listed(pool))
            .collect();
        pools.sort_by_key(|pool| pool.pool);
        pools
    }
}

impl Pool {
    pub(super) fn slots(&self) -> u64 {
        self.slots.len() as u64
    }

    fn touch(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Whether the pool holds each of `keys` committed, in a page `visibility`
    /// reads; the pages it holds become the most recently used.
    pub(super) fn lookup(&mut self, keys: &[Key], visibility: &Visibility) -> Vec<bool> {
        keys.iter()
            .map(|key| {
                let Some(&slot) = self.index.get(key) else {
                    return false;
                };
                if !matches!(self.slots[slot as usize], Slot::Ready { world, .. } if visibility.sees(world))
                {
                    return false;
                }
                self.refresh(slot);
                true
            })
            .collect()
    }

    /// Make a ready slot the most recently used.
    fn refresh(&mut self, slot: u32) {
        let tick = self.touch();
        if let Slot::Ready {
            touched,
            pins,
            uploads,
            ..
        } = &mut self.slots[slot as usize]
        {
            if Slot::idle(pins, *uploads) {
                self.evictable.remove(touched);
                self.evictable.insert(tick, slot);
            }
            *touched = tick;
        }
    }

    /// A slot for each key the pool holds no page of that `visibility`
    /// reads, written by the worker of `writer` for `head`. A page of the
    /// key that another world keeps to itself, and nobody reads, gives way.
    /// With `window`, the pages of `keys` the pool reserves or holds become
    /// one window, whose id is returned.
    fn reserve(
        &mut self,
        writer: &Scope,
        head: ClientId,
        keys: &[Key],
        window: Option<PageWindow>,
        visibility: &Visibility,
    ) -> (Vec<PageSlot>, Option<u64>) {
        let slots: Vec<PageSlot> = keys
            .iter()
            .map(|key| {
                if !self.yield_unseen(key, visibility) {
                    return PageSlot::Present;
                }
                let Some(slot) = self.take_slot() else {
                    return PageSlot::Full;
                };
                self.slots[slot as usize] = Slot::Reserved {
                    key: *key,
                    writer: writer.clone(),
                    head: Some(head),
                    world: visibility.world,
                };
                self.index.insert(*key, slot);
                PageSlot::Reserved {
                    slot: u64::from(slot),
                }
            })
            .collect();
        let Some(window) = window else {
            return (slots, None);
        };
        // A page reserved or held already belongs to the window as well; a
        // page left out for want of a slot leaves the window short, and its
        // writer aborts the rest.
        let members: Vec<u32> = keys
            .iter()
            .filter_map(|key| self.index.get(key).copied())
            .collect();
        let id = self.next_window;
        self.next_window += 1;
        for slot in &members {
            self.member_of.entry(*slot).or_default().push(id);
        }
        self.windows.insert(
            id,
            Window {
                slots: members,
                anchor: window.anchor,
            },
        );
        (slots, Some(id))
    }

    /// A free slot, or the least recently used evictable one, emptied.
    fn take_slot(&mut self) -> Option<u32> {
        if let Some(slot) = self.free.pop() {
            return Some(slot);
        }
        let (_, slot) = self.evictable.pop_first()?;
        if let Slot::Ready { key, .. } = &self.slots[slot as usize] {
            let key = *key;
            self.index.remove(&key);
            self.slots[slot as usize] = Slot::Free;
            self.leave(slot, key);
        }
        Some(slot)
    }

    /// `slot`'s page of `key` left the pool: the windows it belonged to end.
    fn leave(&mut self, slot: u32, key: Key) {
        self.removed.push(key);
        for window in self.member_of.remove(&slot).unwrap_or_default() {
            self.end_window(window);
        }
    }

    /// End window `id`: its pages that no other window holds and nobody
    /// reads leave the pool. Pinned, uploading and unwritten pages stay as
    /// pages of no window.
    pub(super) fn end_window(&mut self, id: u64) {
        let Some(window) = self.windows.remove(&id) else {
            return;
        };
        self.ended.push((id, window.anchor));
        for slot in window.slots {
            let Some(windows) = self.member_of.get_mut(&slot) else {
                continue;
            };
            windows.retain(|member| *member != id);
            if !windows.is_empty() {
                continue;
            }
            self.member_of.remove(&slot);
            let idle = matches!(
                &self.slots[slot as usize],
                Slot::Ready { pins, uploads, .. } if Slot::idle(pins, *uploads)
            );
            if idle {
                self.free_slot(slot);
            }
        }
    }

    /// Reserved pages become ready. Keys not reserved are skipped: a
    /// reservation the server ended already, or a page another writer
    /// holds. Returns the pages committed, with their slots.
    pub(super) fn commit(&mut self, keys: &[Key]) -> Vec<(Key, u32)> {
        let mut committed = Vec::new();
        for key in keys {
            let Some(&slot) = self.index.get(key) else {
                continue;
            };
            let Slot::Reserved { world, .. } = self.slots[slot as usize] else {
                continue;
            };
            self.ready(*key, slot, world);
            committed.push((*key, slot));
        }
        committed
    }

    /// `slot` holds `key`'s page, which `world` published, readable and the
    /// most recently used.
    fn ready(&mut self, key: Key, slot: u32, world: u64) {
        let tick = self.touch();
        self.slots[slot as usize] = Slot::Ready {
            key,
            touched: tick,
            pins: Vec::new(),
            uploads: 0,
            world,
        };
        self.evictable.insert(tick, slot);
    }

    /// Whether the pool holds `key`'s committed page in a page `visibility`
    /// reads.
    pub(super) fn holds(&self, key: &Key, visibility: &Visibility) -> bool {
        self.index.get(key).is_some_and(|&slot| {
            matches!(self.slots[slot as usize], Slot::Ready { world, .. } if visibility.sees(world))
        })
    }

    /// The world that published the committed page in `slot`.
    pub(super) fn publisher(&self, slot: u32) -> Option<u64> {
        match self.slots.get(slot as usize)? {
            Slot::Ready { world, .. } => Some(*world),
            _ => None,
        }
    }

    /// Free `key`'s page when `visibility` does not read it and nobody else
    /// does: the slot it held is free for a page `visibility` reads. Whether
    /// the pool now holds no page of `key`.
    fn yield_unseen(&mut self, key: &Key, visibility: &Visibility) -> bool {
        let Some(&held) = self.index.get(key) else {
            return true;
        };
        let unseen_idle = matches!(
            &self.slots[held as usize],
            Slot::Ready { world, pins, uploads, .. }
                if !visibility.sees(*world) && Slot::idle(pins, *uploads)
        );
        if unseen_idle {
            self.free_slot(held);
        }
        unseen_idle
    }

    /// A slot for the connector to read each key's page into, for the keys
    /// the pool holds no page of that `visibility` reads. Fetches hold at
    /// most half of the slots, so that prefetching never takes the slots
    /// that workers write.
    pub(super) fn prefetch(&mut self, keys: &[Key], visibility: &Visibility) -> Vec<Fetch> {
        let limit = self.slots() / 2;
        keys.iter()
            .map(|key| {
                if !self.yield_unseen(key, visibility) {
                    return Fetch::Present;
                }
                if self.fetching >= limit {
                    return Fetch::Full;
                }
                let Some(slot) = self.take_slot() else {
                    return Fetch::Full;
                };
                self.slots[slot as usize] = Slot::Fetching { key: *key };
                self.index.insert(*key, slot);
                self.fetching += 1;
                Fetch::Reserved(slot)
            })
            .collect()
    }

    /// The connector read `pages` into their slots: `results[i]` is whether
    /// page `i` arrived whole. Arrived pages become ready; the others'
    /// slots become free.
    pub(super) fn fetched(&mut self, pages: &[(Key, u32)], results: &[bool]) {
        for (index, (key, slot)) in pages.iter().enumerate() {
            let fetching =
                matches!(&self.slots[*slot as usize], Slot::Fetching { key: held } if held == key);
            if !fetching {
                continue;
            }
            if results.get(index).copied().unwrap_or(false) {
                self.fetching -= 1;
                // The KV storage holds only pages every world shares.
                self.ready(*key, *slot, 0);
            } else {
                self.free_slot(*slot);
            }
        }
    }

    /// The connector reads `pages` from their slots: they are not evicted
    /// until [`Pool::uploaded`].
    pub(super) fn upload(&mut self, pages: &[(Key, u32)]) {
        for (key, slot) in pages {
            if let Slot::Ready {
                key: held,
                touched,
                pins,
                uploads,
                ..
            } = &mut self.slots[*slot as usize]
                && held == key
            {
                if Slot::idle(pins, *uploads) {
                    self.evictable.remove(touched);
                }
                *uploads += 1;
            }
        }
    }

    /// The connector's reads of `pages`, which [`Pool::upload`] held, ended.
    pub(super) fn uploaded(&mut self, pages: &[(Key, u32)]) {
        for (key, slot) in pages {
            if let Slot::Ready {
                key: held,
                touched,
                pins,
                uploads,
                ..
            } = &mut self.slots[*slot as usize]
                && held == key
                && *uploads > 0
            {
                *uploads -= 1;
                if Slot::idle(pins, *uploads) {
                    let (touched, slot) = (*touched, *slot);
                    self.idle_again(touched, slot);
                }
            }
        }
    }

    /// The connector is gone: its fetches end unwritten and its uploads
    /// end.
    pub(super) fn connector_gone(&mut self) {
        for slot in 0..self.slots.len() as u32 {
            match &mut self.slots[slot as usize] {
                Slot::Fetching { .. } => self.free_slot(slot),
                Slot::Ready {
                    touched,
                    pins,
                    uploads,
                    ..
                } if *uploads > 0 => {
                    *uploads = 0;
                    if pins.is_empty() {
                        self.evictable.insert(*touched, slot);
                    }
                }
                _ => {}
            }
        }
    }

    /// Whether the connector reads pages into the pool or writes pages of
    /// it to the KV storage.
    fn connector_busy(&self) -> bool {
        self.fetching > 0
            || self
                .slots
                .iter()
                .any(|slot| matches!(slot, Slot::Ready { uploads, .. } if *uploads > 0))
    }

    /// Reserved pages were not written: their slots become free.
    pub(super) fn abort(&mut self, keys: &[Key]) {
        for key in keys {
            let Some(&slot) = self.index.get(key) else {
                continue;
            };
            if matches!(self.slots[slot as usize], Slot::Reserved { .. }) {
                self.free_slot(slot);
            }
        }
    }

    fn free_slot(&mut self, slot: u32) {
        let key = match &self.slots[slot as usize] {
            Slot::Reserved { key, .. } => *key,
            Slot::Fetching { key } => {
                self.fetching -= 1;
                *key
            }
            Slot::Ready {
                key,
                touched,
                pins,
                uploads,
                ..
            } => {
                if Slot::idle(pins, *uploads) {
                    self.evictable.remove(touched);
                }
                *key
            }
            Slot::Free => return,
        };
        self.index.remove(&key);
        self.slots[slot as usize] = Slot::Free;
        self.free.push(slot);
        self.leave(slot, key);
    }

    /// Discard the pages published by the worlds `discarded` picks (a world
    /// on trial that a rollback retires, or every world but a committed one
    /// of other numerics): those nobody reads leave the pool, and no world
    /// reads the others, which leave it once their readers unpin them, nor
    /// the pages they reserved and commit later. Returns how many.
    pub(super) fn discard_where(&mut self, discarded: impl Fn(u64) -> bool) -> u64 {
        let mut pages = 0;
        for slot in 0..self.slots.len() as u32 {
            let idle = match &mut self.slots[slot as usize] {
                Slot::Ready {
                    world: published,
                    pins,
                    uploads,
                    ..
                } if *published != DISCARDED && discarded(*published) => {
                    *published = DISCARDED;
                    pages += 1;
                    Slot::idle(pins, *uploads)
                }
                Slot::Reserved {
                    world: published, ..
                } if *published != DISCARDED && discarded(*published) => {
                    *published = DISCARDED;
                    pages += 1;
                    false
                }
                _ => false,
            };
            if idle {
                self.free_slot(slot);
            }
        }
        pages
    }

    /// The pages `world` published, with their slots: those every world
    /// shares once the commit of `world` retires its predecessor.
    pub(super) fn published_by(&self, world: u64) -> Vec<(Key, u32)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, state)| match state {
                Slot::Ready {
                    key,
                    world: published,
                    ..
                } if *published == world => Some((*key, slot as u32)),
                _ => None,
            })
            .collect()
    }

    /// Pin the committed pages of `keys` that `visibility` reads for the
    /// worker of `reader`.
    pub(super) fn pin(
        &mut self,
        reader: &Scope,
        head: ClientId,
        keys: &[Key],
        visibility: &Visibility,
    ) -> Vec<Option<u64>> {
        keys.iter()
            .map(|key| {
                let slot = *self.index.get(key)?;
                let tick = self.touch();
                let Slot::Ready {
                    touched,
                    pins,
                    uploads,
                    world,
                    ..
                } = &mut self.slots[slot as usize]
                else {
                    return None;
                };
                if !visibility.sees(*world) {
                    return None;
                }
                if Slot::idle(pins, *uploads) {
                    self.evictable.remove(touched);
                }
                *touched = tick;
                pins.push((reader.clone(), Some(head)));
                Some(u64::from(slot))
            })
            .collect()
    }

    /// The worker of `reader` read the pages of `keys`: one pin each ends.
    pub(super) fn unpin(&mut self, reader: &Scope, keys: &[Key]) {
        for key in keys {
            let Some(&slot) = self.index.get(key) else {
                continue;
            };
            self.unpin_slot(slot, |(scope, _)| scope == reader);
        }
    }

    /// End the first pin of `slot` that `matches` names.
    fn unpin_slot(&mut self, slot: u32, matches: impl Fn(&(Scope, Option<ClientId>)) -> bool) {
        let Slot::Ready {
            touched,
            pins,
            uploads,
            ..
        } = &mut self.slots[slot as usize]
        else {
            return;
        };
        if let Some(position) = pins.iter().position(matches) {
            pins.remove(position);
            if Slot::idle(pins, *uploads) {
                let touched = *touched;
                self.idle_again(touched, slot);
            }
        }
    }

    /// A ready `slot`, last touched at `touched`, that nobody reads any more:
    /// a discarded page, which no world reads, leaves the pool; another
    /// waits for eviction, least recently used first.
    fn idle_again(&mut self, touched: u64, slot: u32) {
        if matches!(
            self.slots[slot as usize],
            Slot::Ready {
                world: DISCARDED,
                ..
            }
        ) {
            self.free_slot(slot);
        } else {
            self.evictable.insert(touched, slot);
        }
    }

    /// End `head`'s reservations and pins of scopes without a connected
    /// worker, and orphan the others.
    fn orphan(&mut self, head: ClientId, connected: &dyn Fn(&Scope) -> bool) {
        for slot in 0..self.slots.len() as u32 {
            match &mut self.slots[slot as usize] {
                Slot::Reserved {
                    writer,
                    head: owner,
                    ..
                } if *owner == Some(head) => {
                    if connected(writer) {
                        *owner = None;
                    } else {
                        self.free_slot(slot);
                    }
                }
                Slot::Ready { pins, .. } => {
                    let mut gone = 0;
                    for (reader, owner) in pins.iter_mut() {
                        if *owner == Some(head) {
                            if connected(reader) {
                                *owner = None;
                            } else {
                                gone += 1;
                                *owner = Some(ClientId::MAX);
                            }
                        }
                    }
                    for _ in 0..gone {
                        self.unpin_slot(slot, |(_, owner)| *owner == Some(ClientId::MAX));
                    }
                }
                _ => {}
            }
        }
    }

    /// End the orphaned reservations and pins of `scope`.
    fn release_orphans(&mut self, scope: &Scope) {
        for slot in 0..self.slots.len() as u32 {
            match &self.slots[slot as usize] {
                Slot::Reserved { writer, head, .. } if head.is_none() && writer == scope => {
                    self.free_slot(slot);
                }
                Slot::Ready { pins, .. } => {
                    let orphans = pins
                        .iter()
                        .filter(|(reader, owner)| owner.is_none() && reader == scope)
                        .count();
                    for _ in 0..orphans {
                        self.unpin_slot(slot, |(reader, owner)| owner.is_none() && reader == scope);
                    }
                }
                _ => {}
            }
        }
    }

    fn listed(&self, pool: u64) -> ListedPool {
        let (mut ready, mut reserved, mut pinned, mut uploading) = (0, 0, 0, 0);
        for slot in &self.slots {
            match slot {
                Slot::Free | Slot::Fetching { .. } => {}
                Slot::Reserved { .. } => reserved += 1,
                Slot::Ready { pins, uploads, .. } => {
                    ready += 1;
                    pinned += u64::from(!pins.is_empty());
                    uploading += u64::from(*uploads > 0);
                }
            }
        }
        ListedPool {
            pool,
            deployment: self.deployment.clone(),
            shape: self.shape,
            numa_node: self.numa_node,
            slots: self.slots(),
            ready,
            reserved,
            pinned,
            fetching: self.fetching,
            uploading,
        }
    }
}

/// A memfd named `name` of `bytes`, backed by huge pages when the host has
/// them to give, whose pages are allocated on `numa_node` when one is
/// given: a page pool, or the host copy of a world's weights.
pub(crate) fn shared_memory(name: &str, bytes: u64, numa_node: Option<u32>) -> io::Result<OwnedFd> {
    use rustix::fs::{MemfdFlags, ftruncate, memfd_create};

    // Huge pages hold a pool in fewer TLB entries for the copies that read
    // and write it. A hugetlb memfd takes any size, but a shared mapping
    // reserves its pages: a mapping the host cannot reserve fails, so one
    // is made and dropped here before the pool is handed out.
    const HUGE_PAGE: u64 = 2 << 20;
    let huge = bytes.next_multiple_of(HUGE_PAGE);
    let (descriptor, size) = match memfd_create(name, MemfdFlags::CLOEXEC | MemfdFlags::HUGETLB) {
        Ok(descriptor) if ftruncate(&descriptor, huge).is_ok() && reserves(&descriptor, huge) => {
            (descriptor, huge)
        }
        _ => {
            let descriptor = memfd_create(name, MemfdFlags::CLOEXEC)?;
            ftruncate(&descriptor, bytes)?;
            (descriptor, bytes)
        }
    };
    if let Some(node) = numa_node {
        prefer_node(&descriptor, size, node)?;
    }
    Ok(descriptor)
}

/// Make `node` the preferred NUMA node of the pages of the shared memory
/// object `descriptor` holds, `bytes` long: a shared object keeps the
/// policy of a mapping of it for every process that maps it later, so the
/// pages land on `node` whoever first touches them. Pages fall back to
/// other nodes when `node` has none free.
pub(crate) fn prefer_node(descriptor: &OwnedFd, bytes: u64, node: u32) -> io::Result<()> {
    use rustix::mm::{MapFlags, ProtFlags, mmap, munmap};

    // The node mask holds bits for nodes 0..MASK_BITS.
    const MASK_BITS: u32 = 1024;
    if node >= MASK_BITS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("NUMA node {node} is beyond the nodes a page pool binds to"),
        ));
    }
    let length = usize::try_from(bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "a page pool too large"))?;
    let mut mask = [0_u64; (MASK_BITS / 64) as usize];
    mask[(node / 64) as usize] |= 1 << (node % 64);
    // SAFETY: a new shared mapping of the pool's memfd, which this function
    // unmaps; nothing reads or writes it.
    let address = unsafe {
        mmap(
            std::ptr::null_mut(),
            length,
            ProtFlags::READ,
            MapFlags::SHARED,
            descriptor,
            0,
        )
    }?;
    // SAFETY: mbind of the mapping made above, with a node mask of
    // `MASK_BITS` bits.
    let bound = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            address,
            length,
            libc::MPOL_PREFERRED,
            mask.as_ptr(),
            u64::from(MASK_BITS) + 1,
            0_u32,
        )
    };
    let error = (bound != 0).then(io::Error::last_os_error);
    // SAFETY: the mapping made above.
    unsafe { munmap(address, length) }?;
    match error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Whether a shared mapping of `bytes` of `descriptor` can be made.
fn reserves(descriptor: &OwnedFd, bytes: u64) -> bool {
    use rustix::mm::{MapFlags, ProtFlags, mmap, munmap};

    let Ok(length) = usize::try_from(bytes) else {
        return false;
    };
    // SAFETY: a new mapping this function makes and unmaps; nothing reads
    // or writes it.
    match unsafe {
        mmap(
            std::ptr::null_mut(),
            length,
            ProtFlags::READ,
            MapFlags::SHARED,
            descriptor,
            0,
        )
    } {
        Ok(address) => {
            // SAFETY: the mapping made above.
            let _ = unsafe { munmap(address, length) };
            true
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(rank: u32) -> Scope {
        Scope {
            deployment: "model".into(),
            group: "text".into(),
            rank,
        }
    }

    fn key(byte: u8) -> Key {
        Key([byte; 32])
    }

    /// What a head of a deployment with no world on trial reads.
    fn all() -> Visibility {
        Visibility {
            world: 0,
            private: BTreeSet::new(),
        }
    }

    /// A pool of `slots` pages of 16 bytes and layout `layout`, in `pools`.
    fn add_pool(pools: &mut Pools, layout: u8, slots: u64) -> u64 {
        let shape = PageShape {
            layout: key(layout),
            kv_head_offset: 0,
            kv_head_count: 1,
            page_bytes: 16,
        };
        let slots = pools.slots_of(16, Some(16 * slots)).unwrap();
        pools.create("deployment", shape, None, slots).unwrap()
    }

    /// The servers' pools with one pool of `slots` pages, and its id.
    fn pools(slots: u64) -> (Pools, u64) {
        let mut pools = Pools::new(0);
        let id = add_pool(&mut pools, 0, slots);
        (pools, id)
    }

    /// Reserve and commit `keys` in `pool`, as one window with `window`.
    fn store(pools: &mut Pools, pool: u64, keys: &[Key], window: Option<PageWindow>) {
        let slots = pools
            .reserve(pool, &scope(0), 1, keys, window, &all())
            .unwrap();
        assert!(
            slots
                .iter()
                .all(|slot| matches!(slot, PageSlot::Reserved { .. } | PageSlot::Present)),
            "{slots:?}"
        );
        pools.get_mut(pool).unwrap().commit(keys);
    }

    fn held(pools: &mut Pools, pool: u64, keys: &[Key]) -> Vec<bool> {
        pools.get_mut(pool).unwrap().lookup(keys, &all())
    }

    #[test]
    fn committed_pages_are_found_and_evicted_least_recently_used_first() {
        let (mut pools, id) = pools(2);
        assert!(matches!(
            pools
                .reserve(id, &scope(0), 1, &[key(1), key(2)], None, &all())
                .unwrap()[..],
            [PageSlot::Reserved { .. }, PageSlot::Reserved { .. }]
        ));
        // Reserved pages are not readable yet.
        assert_eq!(held(&mut pools, id, &[key(1)]), [false]);
        pools.get_mut(id).unwrap().commit(&[key(1), key(2)]);
        assert_eq!(
            held(&mut pools, id, &[key(1), key(3), key(2)]),
            [true, false, true]
        );

        // Touching the first page leaves the second least recently used.
        assert_eq!(held(&mut pools, id, &[key(1)]), [true]);
        assert_eq!(
            pools
                .reserve(id, &scope(0), 1, &[key(1)], None, &all())
                .unwrap(),
            vec![PageSlot::Present]
        );
        assert!(matches!(
            pools
                .reserve(id, &scope(0), 1, &[key(3)], None, &all())
                .unwrap()[..],
            [PageSlot::Reserved { .. }]
        ));
        assert_eq!(held(&mut pools, id, &[key(2), key(1)]), [false, true]);
    }

    #[test]
    fn pinned_and_reserved_slots_are_not_evicted() {
        let (mut pools, id) = pools(2);
        pools.reserve(id, &scope(0), 1, &[key(1), key(2)], None, &all());
        let pool = pools.get_mut(id).unwrap();
        pool.commit(&[key(1)]);
        let slot = pool_slot_of(pool, key(1));
        assert_eq!(
            pool.pin(&scope(1), 1, &[key(1), key(9)], &all()),
            vec![Some(slot), None]
        );
        assert_eq!(
            pools
                .reserve(id, &scope(0), 1, &[key(3)], None, &all())
                .unwrap(),
            vec![PageSlot::Full]
        );

        pools.get_mut(id).unwrap().unpin(&scope(1), &[key(1)]);
        assert!(matches!(
            pools
                .reserve(id, &scope(0), 1, &[key(3)], None, &all())
                .unwrap()[..],
            [PageSlot::Reserved { .. }]
        ));
        // Aborting frees the reserved slot of the unwritten page.
        pools.get_mut(id).unwrap().abort(&[key(2)]);
        assert!(matches!(
            pools
                .reserve(id, &scope(0), 1, &[key(4)], None, &all())
                .unwrap()[..],
            [PageSlot::Reserved { .. }]
        ));
    }

    /// A window whose least recently used page is evicted leaves the pool
    /// whole, while a page of no window stays.
    #[test]
    fn a_window_is_evicted_whole() {
        let (mut pools, id) = pools(4);
        store(
            &mut pools,
            id,
            &[key(1), key(2)],
            Some(PageWindow { anchor: None }),
        );
        store(&mut pools, id, &[key(3)], None);
        // The window's second page is touched; its first is the least
        // recently used page of the pool.
        held(&mut pools, id, &[key(2)]);
        held(&mut pools, id, &[key(3)]);

        store(&mut pools, id, &[key(4), key(5)], None);
        assert_eq!(
            held(&mut pools, id, &[key(1), key(2), key(3), key(4), key(5)]),
            [false, false, true, true, true]
        );
    }

    /// A window ends when its anchor leaves another pool; a pinned page of
    /// the window stays readable, as a page of no window.
    #[test]
    fn a_window_ends_with_its_anchor() {
        let (mut pools, full) = pools(2);
        let window = add_pool(&mut pools, 1, 4);
        store(&mut pools, full, &[key(1)], None);
        let anchor = Some(PageWindow {
            anchor: Some(PageAnchor {
                pool: full,
                key: key(1),
            }),
        });
        store(&mut pools, window, &[key(11), key(12)], anchor);
        pools
            .get_mut(window)
            .unwrap()
            .pin(&scope(1), 1, &[key(12)], &all());

        // Two more full-attention pages evict the anchor.
        store(&mut pools, full, &[key(2), key(3)], None);
        assert_eq!(held(&mut pools, full, &[key(1)]), [false]);
        assert_eq!(held(&mut pools, window, &[key(11), key(12)]), [false, true]);
    }

    /// A page two windows share stays while one of them does.
    #[test]
    fn a_page_of_two_windows_stays_with_the_window_that_remains() {
        let (mut pools, id) = pools(4);
        let window = Some(PageWindow { anchor: None });
        store(&mut pools, id, &[key(1), key(2)], window);
        store(&mut pools, id, &[key(2), key(3)], window);
        held(&mut pools, id, &[key(2), key(3)]);

        // The first window's own page is the least recently used.
        store(&mut pools, id, &[key(4), key(5)], None);
        assert_eq!(
            held(&mut pools, id, &[key(1), key(2), key(3)]),
            [false, true, true]
        );
    }

    #[test]
    fn a_pool_for_a_numa_node_allocates_its_pages_there() {
        use std::os::fd::AsFd;

        use rustix::mm::{MapFlags, ProtFlags, mmap, munmap};

        // The last node with CPUs, as a host's sockets are: on a host of one
        // node, node 0.
        let online = std::fs::read_to_string("/sys/devices/system/node/has_cpu")
            .unwrap_or_else(|_| "0".into());
        let node: u32 = online
            .trim()
            .rsplit([',', '-'])
            .next()
            .and_then(|node| node.parse().ok())
            .unwrap_or(0);
        let bytes = 4 << 20;
        let descriptor = shared_memory("uniserve-page-pool", bytes, Some(node)).unwrap();

        // A worker's first touch allocates the pages, wherever it runs.
        // SAFETY: a new shared mapping of the memfd, unmapped below.
        let address = unsafe {
            mmap(
                std::ptr::null_mut(),
                bytes as usize,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                descriptor.as_fd(),
                0,
            )
        }
        .unwrap();
        // SAFETY: writes inside the mapping made above.
        unsafe { std::ptr::write_bytes(address.cast::<u8>(), 1, bytes as usize) };
        let maps = std::fs::read_to_string("/proc/self/numa_maps").unwrap();
        let line = maps
            .lines()
            .find(|line| line.starts_with(&format!("{:x} ", address as usize)))
            .unwrap()
            .to_owned();
        // SAFETY: the mapping made above.
        unsafe { munmap(address, bytes as usize) }.unwrap();
        assert!(line.contains(&format!("prefer:{node}")), "{line}");
        assert!(line.contains(&format!("N{node}=")), "{line}");
    }

    fn pool_slot_of(pool: &Pool, key: Key) -> u64 {
        u64::from(pool.index[&key])
    }

    #[test]
    fn fetches_hold_half_the_slots_and_uploads_keep_their_pages() {
        let (mut pools, id) = pools(4);
        let pool = pools.get_mut(id).unwrap();
        let fetches = pool.prefetch(&[key(1), key(2), key(3)], &all());
        let [Fetch::Reserved(first), Fetch::Reserved(second), Fetch::Full] = fetches[..] else {
            panic!("fetches took {fetches:?}");
        };
        // A page being fetched is neither readable nor reserved again.
        assert_eq!(pool.lookup(&[key(1)], &all()), [false]);
        assert_eq!(
            pools
                .reserve(id, &scope(0), 1, &[key(1)], None, &all())
                .unwrap(),
            vec![PageSlot::Present]
        );

        // The first page arrived whole, the second did not.
        let pool = pools.get_mut(id).unwrap();
        let pages = [(key(1), first), (key(2), second)];
        pool.fetched(&pages, &[true, false]);
        assert_eq!(pool.lookup(&[key(1), key(2)], &all()), [true, false]);
        assert_eq!(pool.listed(1).fetching, 0);

        // A page the connector uploads is not evicted until it is done.
        pool.upload(&pages[..1]);
        let reserved = pools
            .reserve(
                id,
                &scope(0),
                1,
                &[key(5), key(6), key(7), key(8)],
                None,
                &all(),
            )
            .unwrap();
        assert_eq!(reserved[3], PageSlot::Full);
        assert!(pools.get(id).unwrap().holds(&key(1), &all()));
        pools.get_mut(id).unwrap().uploaded(&pages[..1]);
        assert!(matches!(
            pools
                .reserve(id, &scope(0), 1, &[key(9)], None, &all())
                .unwrap()[..],
            [PageSlot::Reserved { .. }]
        ));
        assert!(!pools.get(id).unwrap().holds(&key(1), &all()));
    }

    #[test]
    fn a_closed_head_leaves_its_slots_to_the_workers_still_connected() {
        let (mut pools, id) = pools(3);
        pools.reserve(id, &scope(0), 7, &[key(1)], None, &all());
        pools.reserve(id, &scope(1), 7, &[key(2)], None, &all());
        pools.reserve(id, &scope(0), 7, &[key(3)], None, &all());
        let pool = pools.get_mut(id).unwrap();
        pool.commit(&[key(3)]);
        pool.pin(&scope(1), 7, &[key(3)], &all());

        // The worker of rank 0 is gone; rank 1 may still copy.
        pool.orphan(7, &|scope| scope.rank == 1);
        assert_eq!(pool.listed(1).reserved, 1);
        assert_eq!(pool.listed(1).pinned, 1);

        pool.release_orphans(&scope(1));
        let listed = pool.listed(1);
        assert_eq!((listed.reserved, listed.pinned, listed.ready), (0, 0, 1));
    }
}
