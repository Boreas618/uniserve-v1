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
//! | fetching | Being written by the connector, from the distributed tier |
//! | ready | Holds a committed page and may be read |
//! | pinned | Ready and being read by workers; not evictable |
//! | uploading | Ready and being read by the connector; not evictable |
//!
//! A reservation takes a free slot, or evicts the least recently used ready
//! slot nobody reads. A head's reservations and pins end with the head's
//! `commit`, `abort` or `unpin`. When the head's connection closes first,
//! they end once no worker of their writer or reader scope is connected,
//! because that worker's copies may still run until its process ends. The
//! connector's fetches and uploads end when it reports them, or when its
//! connection closes.

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::os::fd::OwnedFd;

use crate::protocol::{Key, ListedPool, PageShape, PageSlot, Scope};

type ClientId = u64;

/// The bytes of a page pool unless the server is started with another
/// size. The workers that copy through a pool register it for device
/// copies, which pins it: its memory is allocated whole when the first
/// worker maps it.
pub const DEFAULT_POOL_BYTES: u64 = 64 << 30;

/// One slot's state.
#[derive(Debug, Clone)]
enum Slot {
    Free,
    Reserved {
        key: Key,
        writer: Scope,
        /// The head that reserved the slot, while it is connected.
        head: Option<ClientId>,
    },
    /// Reserved for the connector, which reads the page from the
    /// distributed tier.
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
    },
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

/// One page pool.
pub(super) struct Pool {
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
}

/// Every page pool of the server.
pub(super) struct Pools {
    pools: HashMap<u64, Pool>,
    by_shape: HashMap<(PageShape, Option<u32>), u64>,
    next: u64,
    /// The bytes of each new pool, rounded down to whole pages.
    pool_bytes: u64,
}

impl Pools {
    pub(super) fn new(pool_bytes: u64) -> Self {
        Self {
            pools: HashMap::new(),
            by_shape: HashMap::new(),
            next: 1,
            pool_bytes,
        }
    }

    /// The pool of `shape` on `numa_node`, created when absent.
    pub(super) fn pool(&mut self, shape: PageShape, numa_node: Option<u32>) -> io::Result<u64> {
        if let Some(&pool) = self.by_shape.get(&(shape, numa_node)) {
            return Ok(pool);
        }
        if shape.page_bytes == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a page pool needs pages of at least one byte",
            ));
        }
        let slots = self.pool_bytes / shape.page_bytes;
        if slots == 0 || slots > u64::from(u32::MAX) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "a page pool of {} bytes cannot hold pages of {} bytes",
                    self.pool_bytes, shape.page_bytes
                ),
            ));
        }
        let descriptor = pool_memory(slots * shape.page_bytes, numa_node)?;
        let pool = self.next;
        self.next += 1;
        self.pools.insert(
            pool,
            Pool {
                shape,
                numa_node,
                descriptor,
                slots: vec![Slot::Free; slots as usize],
                index: HashMap::new(),
                evictable: BTreeMap::new(),
                free: (0..slots as u32).rev().collect(),
                fetching: 0,
                tick: 0,
            },
        );
        self.by_shape.insert((shape, numa_node), pool);
        tracing::info!(
            pool,
            slots,
            page_bytes = shape.page_bytes,
            numa_node,
            "created a page pool"
        );
        Ok(pool)
    }

    pub(super) fn get(&self, pool: u64) -> Option<&Pool> {
        self.pools.get(&pool)
    }

    pub(super) fn get_mut(&mut self, pool: u64) -> Option<&mut Pool> {
        self.pools.get_mut(&pool)
    }

    /// End the reservations and pins of a head whose connection closed: at
    /// once for a scope no connected worker has, and otherwise when that
    /// worker's connection closes (`worker_gone`).
    pub(super) fn head_gone(&mut self, head: ClientId, connected: &dyn Fn(&Scope) -> bool) {
        for pool in self.pools.values_mut() {
            pool.orphan(head, connected);
        }
    }

    /// End the orphaned reservations and pins of `scope`, whose last
    /// connected worker left.
    pub(super) fn worker_gone(&mut self, scope: &Scope) {
        for pool in self.pools.values_mut() {
            pool.release_orphans(scope);
        }
    }

    /// The connector is gone: its fetches and uploads end.
    pub(super) fn connector_gone(&mut self) {
        for pool in self.pools.values_mut() {
            pool.connector_gone();
        }
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

    /// The length of the leading run of `keys` committed in the pool; the
    /// pages of the run become the most recently used.
    pub(super) fn lookup(&mut self, keys: &[Key]) -> u32 {
        let mut run = 0;
        for key in keys {
            let Some(&slot) = self.index.get(key) else {
                break;
            };
            if !matches!(self.slots[slot as usize], Slot::Ready { .. }) {
                break;
            }
            self.refresh(slot);
            run += 1;
        }
        run
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

    /// A slot for each key the pool holds no page of, written by the worker
    /// of `writer` for `head`.
    pub(super) fn reserve(
        &mut self,
        writer: &Scope,
        head: ClientId,
        keys: &[Key],
    ) -> Vec<PageSlot> {
        keys.iter()
            .map(|key| {
                if self.index.contains_key(key) {
                    return PageSlot::Present;
                }
                let Some(slot) = self.take_slot() else {
                    return PageSlot::Full;
                };
                self.slots[slot as usize] = Slot::Reserved {
                    key: *key,
                    writer: writer.clone(),
                    head: Some(head),
                };
                self.index.insert(*key, slot);
                PageSlot::Reserved {
                    slot: u64::from(slot),
                }
            })
            .collect()
    }

    /// A free slot, or the least recently used evictable one, emptied.
    fn take_slot(&mut self) -> Option<u32> {
        if let Some(slot) = self.free.pop() {
            return Some(slot);
        }
        let (_, slot) = self.evictable.pop_first()?;
        if let Slot::Ready { key, .. } = &self.slots[slot as usize] {
            self.index.remove(key);
        }
        self.slots[slot as usize] = Slot::Free;
        Some(slot)
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
            if !matches!(self.slots[slot as usize], Slot::Reserved { .. }) {
                continue;
            }
            self.ready(*key, slot);
            committed.push((*key, slot));
        }
        committed
    }

    /// `slot` holds `key`'s page, readable and the most recently used.
    fn ready(&mut self, key: Key, slot: u32) {
        let tick = self.touch();
        self.slots[slot as usize] = Slot::Ready {
            key,
            touched: tick,
            pins: Vec::new(),
            uploads: 0,
        };
        self.evictable.insert(tick, slot);
    }

    /// Whether the pool holds `key`'s committed page.
    pub(super) fn holds(&self, key: &Key) -> bool {
        self.index
            .get(key)
            .is_some_and(|&slot| matches!(self.slots[slot as usize], Slot::Ready { .. }))
    }

    /// A slot for the connector to read each key's page into, for the keys
    /// the pool holds no page of. Fetches hold at most half of the slots, so
    /// that prefetching never takes the slots that workers write.
    pub(super) fn prefetch(&mut self, keys: &[Key]) -> Vec<Fetch> {
        let limit = self.slots() / 2;
        keys.iter()
            .map(|key| {
                if self.index.contains_key(key) {
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
                self.ready(*key, *slot);
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
            } = &mut self.slots[*slot as usize]
                && held == key
                && *uploads > 0
            {
                *uploads -= 1;
                if Slot::idle(pins, *uploads) {
                    self.evictable.insert(*touched, *slot);
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
        match &self.slots[slot as usize] {
            Slot::Reserved { key, .. } => {
                self.index.remove(key);
            }
            Slot::Fetching { key } => {
                self.index.remove(key);
                self.fetching -= 1;
            }
            Slot::Ready {
                key,
                touched,
                pins,
                uploads,
            } => {
                self.index.remove(key);
                if Slot::idle(pins, *uploads) {
                    self.evictable.remove(touched);
                }
            }
            Slot::Free => return,
        }
        self.slots[slot as usize] = Slot::Free;
        self.free.push(slot);
    }

    /// Pin the committed pages of `keys` for the worker of `reader`.
    pub(super) fn pin(&mut self, reader: &Scope, head: ClientId, keys: &[Key]) -> Vec<Option<u64>> {
        keys.iter()
            .map(|key| {
                let slot = *self.index.get(key)?;
                let tick = self.touch();
                let Slot::Ready {
                    touched,
                    pins,
                    uploads,
                    ..
                } = &mut self.slots[slot as usize]
                else {
                    return None;
                };
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
                self.evictable.insert(*touched, slot);
            }
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

/// A memfd of `bytes`, backed by huge pages when the host has them to give,
/// whose pages are allocated on `numa_node` when one is given.
fn pool_memory(bytes: u64, numa_node: Option<u32>) -> io::Result<OwnedFd> {
    use rustix::fs::{MemfdFlags, ftruncate, memfd_create};

    // Huge pages hold a pool in fewer TLB entries for the copies that read
    // and write it. A hugetlb memfd takes any size, but a shared mapping
    // reserves its pages: a mapping the host cannot reserve fails, so one
    // is made and dropped here before the pool is handed out.
    const HUGE_PAGE: u64 = 2 << 20;
    let huge = bytes.next_multiple_of(HUGE_PAGE);
    let (descriptor, size) = match memfd_create(
        "uniserve-page-pool",
        MemfdFlags::CLOEXEC | MemfdFlags::HUGETLB,
    ) {
        Ok(descriptor) if ftruncate(&descriptor, huge).is_ok() && reserves(&descriptor, huge) => {
            (descriptor, huge)
        }
        _ => {
            let descriptor = memfd_create("uniserve-page-pool", MemfdFlags::CLOEXEC)?;
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
fn prefer_node(descriptor: &OwnedFd, bytes: u64, node: u32) -> io::Result<()> {
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

    fn pool(slots: u64) -> Pool {
        let shape = PageShape {
            content: key(0),
            kv_head_offset: 0,
            kv_head_count: 1,
            page_bytes: 16,
        };
        let mut pools = Pools::new(16 * slots);
        let id = pools.pool(shape, None).unwrap();
        pools.pools.remove(&id).unwrap()
    }

    #[test]
    fn committed_pages_are_found_and_evicted_least_recently_used_first() {
        let mut pool = pool(2);
        assert!(matches!(
            pool.reserve(&scope(0), 1, &[key(1), key(2)])[..],
            [PageSlot::Reserved { .. }, PageSlot::Reserved { .. }]
        ));
        // Reserved pages are not readable yet.
        assert_eq!(pool.lookup(&[key(1)]), 0);
        pool.commit(&[key(1), key(2)]);
        assert_eq!(pool.lookup(&[key(1), key(2), key(3)]), 2);

        // Touching the first page leaves the second least recently used.
        assert_eq!(pool.lookup(&[key(1)]), 1);
        assert_eq!(
            pool.reserve(&scope(0), 1, &[key(1)]),
            vec![PageSlot::Present]
        );
        assert!(matches!(
            pool.reserve(&scope(0), 1, &[key(3)])[..],
            [PageSlot::Reserved { .. }]
        ));
        assert_eq!(pool.lookup(&[key(2)]), 0);
        assert_eq!(pool.lookup(&[key(1)]), 1);
    }

    #[test]
    fn pinned_and_reserved_slots_are_not_evicted() {
        let mut pool = pool(2);
        pool.reserve(&scope(0), 1, &[key(1), key(2)]);
        pool.commit(&[key(1)]);
        let slot = pool_slot_of(&pool, key(1));
        assert_eq!(
            pool.pin(&scope(1), 1, &[key(1), key(9)]),
            vec![Some(slot), None]
        );
        assert_eq!(pool.reserve(&scope(0), 1, &[key(3)]), vec![PageSlot::Full]);

        pool.unpin(&scope(1), &[key(1)]);
        assert!(matches!(
            pool.reserve(&scope(0), 1, &[key(3)])[..],
            [PageSlot::Reserved { .. }]
        ));
        // Aborting frees the reserved slot of the unwritten page.
        pool.abort(&[key(2)]);
        assert!(matches!(
            pool.reserve(&scope(0), 1, &[key(4)])[..],
            [PageSlot::Reserved { .. }]
        ));
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
        let descriptor = pool_memory(bytes, Some(node)).unwrap();

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
        let mut pool = pool(4);
        let fetches = pool.prefetch(&[key(1), key(2), key(3)]);
        let [Fetch::Reserved(first), Fetch::Reserved(second), Fetch::Full] = fetches[..] else {
            panic!("fetches took {fetches:?}");
        };
        // A page being fetched is neither readable nor reserved again.
        assert_eq!(pool.lookup(&[key(1)]), 0);
        assert_eq!(
            pool.reserve(&scope(0), 1, &[key(1)]),
            vec![PageSlot::Present]
        );

        // The first page arrived whole, the second did not.
        let pages = [(key(1), first), (key(2), second)];
        pool.fetched(&pages, &[true, false]);
        assert_eq!(pool.lookup(&[key(1), key(2)]), 1);
        assert_eq!(pool.listed(1).fetching, 0);

        // A page the connector uploads is not evicted until it is done.
        pool.upload(&pages[..1]);
        let reserved = pool.reserve(&scope(0), 1, &[key(5), key(6), key(7), key(8)]);
        assert_eq!(reserved[3], PageSlot::Full);
        assert!(pool.holds(&key(1)));
        pool.uploaded(&pages[..1]);
        assert!(matches!(
            pool.reserve(&scope(0), 1, &[key(9)])[..],
            [PageSlot::Reserved { .. }]
        ));
        assert!(!pool.holds(&key(1)));
    }

    #[test]
    fn a_closed_head_leaves_its_slots_to_the_workers_still_connected() {
        let mut pool = pool(3);
        pool.reserve(&scope(0), 7, &[key(1)]);
        pool.reserve(&scope(1), 7, &[key(2)]);
        pool.reserve(&scope(0), 7, &[key(3)]);
        pool.commit(&[key(3)]);
        pool.pin(&scope(1), 7, &[key(3)]);

        // The worker of rank 0 is gone; rank 1 may still copy.
        pool.orphan(7, &|scope| scope.rank == 1);
        assert_eq!(pool.listed(1).reserved, 1);
        assert_eq!(pool.listed(1).pinned, 1);

        pool.release_orphans(&scope(1));
        let listed = pool.listed(1);
        assert_eq!((listed.reserved, listed.pinned, listed.ready), (0, 0, 1));
    }
}
