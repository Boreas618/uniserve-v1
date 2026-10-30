//! Scheduler-owned paged key/value unit allocation, window release, and prefix
//! reuse.
//!
//! The scheduler chooses every physical cache unit before a call is dispatched;
//! workers read and write only the units named in a call's block tables.
//! [`BlockPool`] owns physical unit availability and prefix-cache metadata.
//! [`BlockTable`] maps one sequence's logical pages of one cache group to
//! reference-counted pages of units. Dropping the final reference returns the
//! units to the pool. [`PrefixCacheCoordinator`] applies allocation, window
//! release and prefix reuse across every cache group of a request at once.
//!
//! Key concepts:
//!
//! - Unit pool: the worker stores every cache group in one pool of equally
//!   sized units. A group's logical page holds `block_size` tokens of every
//!   layer in the group and occupies `units_per_page` units, so groups with
//!   different page shapes share one free queue and one capacity, and
//!   admission and growth compare the sum of every group's unit demand with
//!   the free units.
//! - Unit zero: the worker treats unit zero as its padding sentinel, valid
//!   storage in every plane, so the pool never allocates it and excludes it
//!   from capacity.
//! - Free queue: an intrusive doubly linked queue of unreferenced units.
//!   Allocation takes units from the head and released units join the tail,
//!   one page's units consecutively. An unreferenced unit of a published
//!   prefix page ([`UnitState::Cached`]) therefore stays reusable until
//!   allocation reaches it; allocation then evicts the whole page. A prefix
//!   hit on an unreferenced page unlinks its units from wherever they sit.
//! - Sliding windows: a table of a sliding-window group keeps pages from its
//!   `start_page` on. Pages before it are retired, and a worker never reads
//!   them. After a call completes, [`PrefixCacheCoordinator::release_window`]
//!   retires the pages that lie entirely before the oldest history position
//!   any later reader needs.
//! - Prefix identity: a published page is keyed by a chained 64-bit hash per
//!   group, but a match also requires equal tokens, the same group and the
//!   same loaded `WorkerEndpoint`. The token comparison guards against hash
//!   collisions, and the endpoint comparison confines reuse to the loaded
//!   worker incarnation that computed and retains the page.
//! - Reuse: each group's attention type has a manager ([`manager`]) that
//!   says which pages a reuse at a boundary needs: every full-attention
//!   page before it, and each sliding-window group's window of pages just
//!   before it. The same managers decide reuse in the device cache and,
//!   through [`KvOffloadManager`], in the host KV cache and the KV storage.
//! - State checkpoints: a state group (`CacheGroupKind::State`) holds the
//!   checkpoints of the layers a worker keeps per request slot, such as a
//!   recurrent state: page `k` holds the state after token `(k + 1) *
//!   block_size - 1`, so a reuse at a boundary needs the page that ends
//!   there, as a one-token window does. A table holds units only for the
//!   pages that hold a checkpoint, and the padding sentinel for the others
//!   ([`TablePage`]): a prompt call checkpoints the last boundary it reaches
//!   (the chunk boundaries of chunked prefill), a decode call only a
//!   multiple of the state interval ([`DEFAULT_STATE_INTERVAL`]), as SGLang
//!   tracks a recurrent state at the end of each prefill chunk and every
//!   `mamba_track_interval` decoded tokens
//!   (`refs/sglang/python/sglang/srt/managers/schedule_batch.py`,
//!   `_mamba_radix_cache_v2_req_prepare_for_extend` and the decode track
//!   mask) and vLLM's `mamba_cache_mode = "align"` caches the state at the
//!   end of each scheduler step on a block boundary
//!   (`refs/vllm/vllm/config/cache.py`). Checkpoint pages publish, reuse
//!   and copy through the host KV cache as K/V pages do.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, Weak};

use ipc::WorkerEndpoint;
use ipc::{CacheGroup, UnitId};

use crate::lock;

mod encoder_cache;
mod freeq;
pub(crate) mod manager;
mod offload;

pub(crate) use encoder_cache::{EncoderCacheManager, EncoderKey};
use freeq::UnitMeta;
pub use offload::HostCacheSize;
pub(crate) use offload::{
    DEFAULT_BOUNDARY_INTERVAL, HostCacheDemand, KvOffloadManager, OffloadKind, RankCache,
};

/// The decoded tokens between two checkpoints of a state group's request,
/// unless the deployment sets another (`kv_cache_config.state_interval`):
/// SGLang's default `mamba_track_interval`
/// (`refs/sglang/python/sglang/srt/arg_groups/fields/exec_.py`). A request
/// resumed by its tokens recomputes at most this many decoded tokens.
pub(crate) const DEFAULT_STATE_INTERVAL: usize = 256;

/// Computes an incremental per-page prefix hash with 64-bit FNV-1a.
///
/// `parent` is the hash of the preceding page (or the chain seed for the
/// first page), so equal hashes identify equal complete prefixes up to hash
/// collisions. `group_id` and `modality_tag` separate chains that hash the
/// same tokens for different cache groups or modalities. FNV-1a is not
/// collision resistant, so callers must still compare exact tokens before
/// reusing a page.
pub(crate) fn block_hash(parent: u64, group_id: u32, modality_tag: u8, tokens: &[u32]) -> u64 {
    use std::hash::Hasher as _;

    let mut hasher = fnv::FnvHasher::default();
    hasher.write(&parent.to_le_bytes());
    hasher.write(&group_id.to_le_bytes());
    hasher.write(&[modality_tag]);
    hasher.write(&(tokens.len() as u32).to_le_bytes());
    for token in tokens {
        hasher.write(&token.to_le_bytes());
    }
    hasher.finish()
}

/// The modality byte of a page hash: every published page holds text
/// (understanding-branch) tokens. The byte is part of the page keys the host
/// KV cache and KV storage find pages by, so its value is fixed.
pub(crate) const TEXT_MODALITY_TAG: u8 = 0;

/// Ownership state of one physical cache unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnitState {
    /// Unreferenced and unpublished; unit zero stays in this state.
    Free,
    /// Referenced by a table whose pages have been allocated but not yet
    /// activated by the scheduler.
    Reserved,
    /// Referenced by a table after activation, or acquired from the prefix
    /// cache.
    Active,
    /// Unreferenced but still part of a published prefix page. The unit sits
    /// in the free queue, and allocation evicts its page when it reaches it.
    Cached,
}

/// One published page as a world hands its prefix cache index to its
/// successor: its group, chained hash, tokens and units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedPage {
    pub group: u32,
    pub hash: u64,
    pub tokens: Vec<u32>,
    pub units: Vec<u32>,
    pub resume: Option<proto::request::CacheResume>,
}

/// Aggregate cache counters retained by the block pool.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockPoolStats {
    /// Allocatable units, excluding the unit-zero sentinel.
    pub(crate) total: usize,
    /// Units in the free queue, including [`UnitState::Cached`] ones.
    pub(crate) free: usize,
    /// Successful `BlockPool::allocate` calls, not units.
    pub(crate) allocations: u64,
    /// Pages removed from the prefix index, whether by allocation reuse,
    /// republication, or worker invalidation.
    pub(crate) evictions: u64,
    /// Pages published to the prefix index for reuse across requests; a
    /// request's private continuation alias does not count.
    pub(crate) pages_stored: u64,
}

/// Identity of one published prefix page.
struct PublishedPage {
    hash: u64,
    group: u32,
    /// Exact token content used to disambiguate equal prefix hashes.
    tokens: Vec<u32>,
    /// Loaded Worker whose rank group retains the page.
    source: Arc<WorkerEndpoint>,
    /// The same immutable bytes may serve public reuse and several private
    /// continuations. `None` names the public alias; no alias copies units.
    aliases: Vec<Option<proto::request::CacheResume>>,
    /// The page's units in page order; the first one keys the page.
    units: Box<[UnitId]>,
    /// The page's key in the host KV cache, once the page was offloaded
    /// there: a withdrawal that evicts the page has its copy verified or
    /// made again under it first.
    host_key: Option<storage::Key>,
}

/// An idle cached page held while a withdrawal evicts it, so that its units
/// keep their bytes until its host KV cache copy is verified or made: its
/// reference, cache group, the worker that retains it and its key in the
/// host KV cache, when it was offloaded there.
pub(crate) struct IdlePage {
    pub(crate) page: PageRef,
    pub(crate) group: u32,
    pub(crate) source: Arc<WorkerEndpoint>,
    pub(crate) host_key: Option<storage::Key>,
}

/// Pool state guarded by one mutex.
///
/// `PageRef::drop` locks this state, and `std::sync::Mutex` is not reentrant,
/// so no page reference may be dropped while it is locked.
struct PoolInner {
    /// Per-unit metadata indexed by unit id.
    meta: Vec<UnitMeta>,
    fq_head: Option<u32>,
    fq_tail: Option<u32>,
    /// Queued units, including [`UnitState::Cached`] ones.
    free_count: usize,
    /// Published pages keyed by their first unit.
    pages: HashMap<u32, PublishedPage>,
    /// First units of every published page copy of a hash. Copies differ by
    /// group, source worker or, after a hash collision, by tokens.
    hash_to_pages: HashMap<u64, Vec<u32>>,
    stats: BlockPoolStats,
    /// Units `1..=scratch`, which the pool never allocates.
    scratch: u32,
    /// The first unit withdrawn from allocation, as the storage holding the
    /// units from it on is given back (`BlockPool::withdraw`); the unit
    /// count while none is.
    withdrawn: u32,
    /// The units the workers map read-only from the predecessor world's
    /// planes (`BlockPool::inherit`): never queued or allocated, their
    /// imported pages acquired and released, until the world takes them
    /// (`BlockPool::take_inherited`).
    inherited: std::ops::Range<u32>,
    /// Of the inherited units, those the predecessor lent this world for
    /// its trial (`BlockPool::borrow`): the workers map them writable and
    /// the pool allocates them as its own.
    borrowed: Vec<std::ops::Range<u32>>,
    /// The units this world lent its successor at its park
    /// (`BlockPool::lend`), which the successor may write: never queued or
    /// allocated until the world takes them back (`BlockPool::reclaim`).
    lent: Vec<std::ops::Range<u32>>,
}

/// The units of `range` below `end`.
fn units_below(range: &std::ops::Range<u32>, end: u32) -> u32 {
    range.end.min(end).saturating_sub(range.start.min(end))
}

impl PoolInner {
    /// Whether the world may not write `unit`: it maps it read-only from
    /// the predecessor's planes, or lent it to its successor.
    fn foreign(&self, unit: u32) -> bool {
        let borrowed = || self.borrowed.iter().any(|range| range.contains(&unit));
        let lent = self.lent.iter().any(|range| range.contains(&unit));
        lent || (self.inherited.contains(&unit) && !borrowed())
    }

    /// Appends a unit to the free queue.
    fn fq_push_back(&mut self, index: u32) {
        let tail = self.fq_tail;
        let meta = &mut self.meta[index as usize];
        meta.fq_prev = tail;
        meta.fq_next = None;
        meta.in_fq = true;
        if let Some(tail) = tail {
            self.meta[tail as usize].fq_next = Some(index);
        } else {
            self.fq_head = Some(index);
        }
        self.fq_tail = Some(index);
        self.free_count += 1;
    }

    /// Unlinks a queued unit from the free queue; does nothing otherwise.
    fn fq_unlink(&mut self, index: u32) {
        if !self.meta[index as usize].in_fq {
            return;
        }
        let previous = self.meta[index as usize].fq_prev;
        let next = self.meta[index as usize].fq_next;
        if let Some(previous) = previous {
            self.meta[previous as usize].fq_next = next;
        } else {
            self.fq_head = next;
        }
        if let Some(next) = next {
            self.meta[next as usize].fq_prev = previous;
        } else {
            self.fq_tail = previous;
        }
        let meta = &mut self.meta[index as usize];
        meta.fq_prev = None;
        meta.fq_next = None;
        meta.in_fq = false;
        self.free_count = self.free_count.saturating_sub(1);
    }

    /// Removes one published page copy from the prefix index, preserving
    /// references.
    ///
    /// Does nothing when `head` keys no published page. Otherwise every unit
    /// of the page loses its identity, unreferenced units revert from
    /// `Cached` to `Free` and keep their free-queue position, other copies of
    /// the same hash stay published, and the call counts one eviction.
    fn remove_page(&mut self, head: u32) {
        let Some(page) = self.pages.remove(&head) else {
            return;
        };
        for unit in page.units.iter() {
            let meta = &mut self.meta[unit.0 as usize];
            meta.published = None;
            if meta.state == UnitState::Cached {
                meta.state = UnitState::Free;
            }
        }
        if let Some(copies) = self.hash_to_pages.get_mut(&page.hash) {
            copies.retain(|candidate| *candidate != head);
            if copies.is_empty() {
                self.hash_to_pages.remove(&page.hash);
            }
        }
        self.stats.evictions += 1;
    }

    /// Releases one reference to every unit of a page.
    ///
    /// The final release queues the units at the tail in page order. Units of
    /// a published page stay in the prefix index as `Cached`, so a later
    /// request can reacquire the page until allocation reaches it.
    fn release_page(&mut self, units: &[UnitId]) {
        for unit in units {
            let foreign = self.foreign(unit.0);
            let meta = &mut self.meta[unit.0 as usize];
            debug_assert!(meta.ref_cnt > 0, "cache unit reference count underflow");
            meta.ref_cnt = meta.ref_cnt.saturating_sub(1);

            // The `in_fq` check keeps a unit from being queued twice.
            if meta.ref_cnt != 0 || meta.in_fq {
                continue;
            }
            // A withdrawn unit leaves the pool once its last holder lets it
            // go, its page with it.
            if unit.0 >= self.withdrawn {
                if let Some(head) = meta.published {
                    self.remove_page(head);
                }
                self.meta[unit.0 as usize].state = UnitState::Free;
                continue;
            }
            // An inherited unit keeps its page for later readers and is
            // never allocated: the world does not write it, nor a unit it
            // lent until it takes it back.
            if foreign {
                meta.state = if meta.published.is_some() {
                    UnitState::Cached
                } else {
                    UnitState::Free
                };
                continue;
            }
            let meta = &mut self.meta[unit.0 as usize];
            meta.state = if meta.published.is_some() {
                UnitState::Cached
            } else {
                UnitState::Free
            };
            self.fq_push_back(unit.0);
        }
        self.stats.free = self.free_count;
    }

    /// The withdrawn units requests still hold.
    fn withdrawn_held(&self) -> usize {
        self.meta[self.withdrawn as usize..]
            .iter()
            .filter(|meta| meta.ref_cnt > 0)
            .count()
    }

    /// Recounts the allocatable units: those below the first withdrawn,
    /// beside the sentinel, the scratch, the inherited units not borrowed
    /// and the lent ones.
    fn recount(&mut self) {
        let below = |ranges: &[std::ops::Range<u32>]| -> u32 {
            ranges
                .iter()
                .map(|range| units_below(range, self.withdrawn))
                .sum()
        };
        let foreign = units_below(&self.inherited, self.withdrawn)
            .saturating_sub(below(&self.borrowed))
            + below(&self.lent);
        self.stats.total = self
            .withdrawn
            .saturating_sub(1 + self.scratch)
            .saturating_sub(foreign) as usize;
        self.stats.free = self.free_count;
    }

    /// Returns the first unit of a published copy with this exact identity.
    fn find_page(
        &self,
        hash: u64,
        group: u32,
        tokens: &[u32],
        source: &WorkerEndpoint,
        resume: Option<proto::request::CacheResume>,
    ) -> Option<u32> {
        self.hash_to_pages.get(&hash)?.iter().copied().find(|head| {
            self.pages.get(head).is_some_and(|page| {
                page.group == group
                    && page.source.as_ref() == source
                    && page.tokens == tokens
                    && page.aliases.contains(&resume)
            })
        })
    }
}

/// A counted reference to one logical page: `units_per_page` physical units
/// of one cache group, in page order.
///
/// Each handle accounts for one reference to every unit of the page, and
/// dropping it releases them. The handle holds the pool weakly, so a handle
/// that outlives its pool drops without effect. Dropping a handle locks the
/// pool: never drop one while holding the pool's lock.
pub(crate) struct PageRef {
    units: Box<[UnitId]>,
    pool: Weak<Mutex<PoolInner>>,
}

impl PageRef {
    /// Returns the page's units in page order.
    pub(crate) fn units(&self) -> &[UnitId] {
        &self.units
    }
}

impl Clone for PageRef {
    fn clone(&self) -> Self {
        if let Some(pool) = self.pool.upgrade() {
            let mut inner = lock(&pool);
            for unit in &self.units {
                inner.meta[unit.0 as usize].ref_cnt += 1;
            }
        }
        Self {
            units: self.units.clone(),
            pool: Weak::clone(&self.pool),
        }
    }
}

impl Drop for PageRef {
    /// Returns this reference to the pool, if the pool still exists.
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            lock(&pool).release_page(&self.units);
        }
    }
}

impl fmt::Debug for PageRef {
    /// Formats the value for diagnostic output.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("PageRef").field(&self.units).finish()
    }
}

/// Physical cache units, free capacity, unit references, and prefix-cache
/// metadata.
///
/// Clones share one pool. Each method locks the pool independently, so a
/// sequence of calls is not atomic as a whole.
#[derive(Clone)]
pub(crate) struct BlockPool {
    inner: Arc<Mutex<PoolInner>>,
}

impl BlockPool {
    /// Creates a pool of `num_units` units, including the unit-zero sentinel.
    /// Units `1..=scratch` are never allocated: they are the workers' warmup
    /// scratch, which a standby worker warms up on while the pool's units
    /// serve (`PrefixCacheInfo::scratch_units`).
    ///
    /// # Panics
    ///
    /// Panics when the pool holds no allocatable unit.
    pub(crate) fn new(num_units: u32, scratch: u32) -> Self {
        assert!(
            num_units > scratch + 1,
            "a cache unit pool requires an allocatable unit"
        );
        let mut inner = PoolInner {
            meta: (0..num_units).map(|_| UnitMeta::new()).collect(),
            fq_head: None,
            fq_tail: None,
            free_count: 0,
            pages: HashMap::new(),
            hash_to_pages: HashMap::new(),
            stats: BlockPoolStats {
                total: (num_units - 1 - scratch) as usize,
                ..BlockPoolStats::default()
            },
            scratch,
            withdrawn: num_units,
            inherited: 0..0,
            borrowed: Vec::new(),
            lent: Vec::new(),
        };

        // Unit zero is the worker's padding sentinel and is never queued,
        // nor are the scratch units after it.
        for unit in scratch + 1..num_units {
            inner.fq_push_back(unit);
        }
        inner.stats.free = inner.free_count;
        Self {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    /// Withdraws the units from `first` on from allocation, as the storage
    /// holding them is to be given back: the queued ones leave the queue,
    /// their published pages the prefix index, and the ones requests hold
    /// leave as their holders let them go. Returns how many are held.
    pub(crate) fn withdraw(&self, first: u32) -> usize {
        let mut inner = lock(&self.inner);
        let end = inner.meta.len() as u32;
        let first = first.max(inner.scratch + 1).min(end);
        if first >= inner.withdrawn {
            return inner.withdrawn_held();
        }
        let from = inner.withdrawn;
        inner.withdrawn = first;
        for unit in first..from {
            if let Some(head) = inner.meta[unit as usize].published {
                inner.remove_page(head);
            }
            inner.fq_unlink(unit);
        }
        inner.recount();
        inner.withdrawn_held()
    }

    /// The units from `first` on, short of the withdrawn ones, that hold
    /// published pages no request holds: the idle cached pages a withdrawal
    /// from `first` evicts.
    pub(crate) fn cached_from(&self, first: u32) -> u32 {
        let inner = lock(&self.inner);
        let end = inner.withdrawn.min(inner.meta.len() as u32);
        (first.min(end)..end)
            .filter(|&unit| {
                let meta = &inner.meta[unit as usize];
                meta.state == UnitState::Cached && meta.published.is_some()
            })
            .count() as u32
    }

    /// The withdrawn units requests still hold.
    pub(crate) fn withdrawn_held(&self) -> usize {
        lock(&self.inner).withdrawn_held()
    }

    /// The first unit withdrawn (`withdraw`); the unit count while none is.
    pub(crate) fn withdrawn(&self) -> u32 {
        lock(&self.inner).withdrawn
    }

    /// Whether some units are withdrawn.
    pub(crate) fn has_withdrawn(&self) -> bool {
        let inner = lock(&self.inner);
        (inner.withdrawn as usize) < inner.meta.len()
    }

    /// Returns every withdrawn unit to allocation: their storage was created
    /// again and reset. A unit a request still holds, as when a restore
    /// cancels a shrink, is queued once its holder lets it go.
    pub(crate) fn restore(&self) {
        let end = lock(&self.inner).meta.len() as u32;
        self.restore_to(end);
    }

    /// Returns the withdrawn units below `end` to allocation, as
    /// [`Self::restore`] returns them all: the storage of the segments
    /// holding them was created.
    pub(crate) fn restore_to(&self, end: u32) {
        let mut inner = lock(&self.inner);
        let end = end.min(inner.meta.len() as u32);
        let from = inner.withdrawn;
        if end <= from {
            return;
        }
        inner.withdrawn = end;
        for unit in from..end {
            // A lent unit waits for its reclaim, wherever it lies.
            if inner.meta[unit as usize].ref_cnt > 0
                || inner.lent.iter().any(|lent| lent.contains(&unit))
            {
                continue;
            }
            let meta = &mut inner.meta[unit as usize];
            meta.state = UnitState::Free;
            meta.published = None;
            inner.fq_push_back(unit);
        }
        inner.recount();
    }

    /// Takes units `range` out of allocation: the workers map them
    /// read-only from the predecessor world's planes, whose pages the pool
    /// imports ([`Self::import`]) and which the world never writes until it
    /// takes them ([`Self::take_inherited`]).
    pub(crate) fn inherit(&self, range: std::ops::Range<u32>) {
        let mut inner = lock(&self.inner);
        for unit in range.clone() {
            inner.fq_unlink(unit);
        }
        inner.inherited = range;
        inner.recount();
    }

    /// The units mapped read-only from the predecessor world.
    pub(crate) fn inherited(&self) -> std::ops::Range<u32> {
        lock(&self.inner).inherited.clone()
    }

    /// Whether readers other than its holder may read `page`: the prefix
    /// cache publishes one of its units, which any request can acquire, or
    /// the predecessor world writes them, mapped read-only here. No call may
    /// write such a page.
    pub(crate) fn shared(&self, page: &PageRef) -> bool {
        let inner = lock(&self.inner);
        page.units
            .iter()
            .any(|unit| inner.foreign(unit.0) || inner.meta[unit.0 as usize].published.is_some())
    }

    /// Returns the inherited units to allocation: the predecessor world is
    /// gone and the workers took its storage. Their pages stay published.
    pub(crate) fn take_inherited(&self) {
        let mut inner = lock(&self.inner);
        let range = std::mem::replace(&mut inner.inherited, 0..0);
        inner.borrowed.clear();
        for unit in range {
            let meta = &inner.meta[unit as usize];
            if meta.ref_cnt > 0 || meta.in_fq || unit >= inner.withdrawn {
                continue;
            }
            inner.fq_push_back(unit);
        }
        inner.recount();
    }

    /// Borrows the inherited units `ranges` for the world's trial: its
    /// predecessor lent them at its park ([`Self::lend`]) and the workers
    /// map them writable. The pool queues them behind its own free units and
    /// allocates them as its own; the predecessor's pages they hold stay
    /// published until allocation reaches them. Units outside the inherited
    /// ones are left out. Returns how many units it borrowed.
    pub(crate) fn borrow(&self, ranges: &[std::ops::Range<u32>]) -> u32 {
        let mut inner = lock(&self.inner);
        let inherited = inner.inherited.clone();
        let mut borrowed = 0;
        for range in ranges {
            let range = range.start.max(inherited.start)..range.end.min(inherited.end);
            if range.is_empty() {
                continue;
            }
            for unit in range.clone() {
                let meta = &inner.meta[unit as usize];
                if meta.ref_cnt > 0 || meta.in_fq || unit >= inner.withdrawn {
                    continue;
                }
                inner.fq_push_back(unit);
            }
            borrowed += range.end - range.start;
            inner.borrowed.push(range);
        }
        inner.recount();
        borrowed
    }

    /// Lends the units `ranges` to the world that succeeds this one, at the
    /// park: the successor may write them, so the pool forgets the pages
    /// they hold and allocates none of them until it takes them back
    /// ([`Self::reclaim`]). Returns how many pages it forgot.
    pub(crate) fn lend(&self, ranges: &[std::ops::Range<u32>]) -> usize {
        let mut inner = lock(&self.inner);
        let mut heads = Vec::new();
        for unit in ranges.iter().cloned().flatten() {
            if let Some(head) = inner.meta[unit as usize].published {
                heads.push(head);
            }
            inner.fq_unlink(unit);
        }
        heads.sort_unstable();
        heads.dedup();
        for &head in &heads {
            inner.remove_page(head);
        }
        inner.lent.extend(ranges.iter().cloned());
        inner.recount();
        heads.len()
    }

    /// The units the world lent its successor and has not taken back.
    pub(crate) fn lent(&self) -> Vec<std::ops::Range<u32>> {
        lock(&self.inner).lent.clone()
    }

    /// Takes back the units lent to the successor world ([`Self::lend`]),
    /// whose processes are gone: they return to allocation holding no page.
    /// Returns how many units it took back.
    pub(crate) fn reclaim(&self) -> u32 {
        let mut inner = lock(&self.inner);
        let lent = std::mem::take(&mut inner.lent);
        let mut reclaimed = 0;
        for unit in lent.into_iter().flatten() {
            let meta = &inner.meta[unit as usize];
            if meta.ref_cnt > 0 || meta.in_fq || unit >= inner.withdrawn {
                continue;
            }
            inner.fq_push_back(unit);
            reclaimed += 1;
        }
        inner.recount();
        reclaimed
    }

    /// The published pages whose units the pool holds (none withdrawn), as
    /// a successor world imports them.
    pub(crate) fn export(&self) -> Vec<IndexedPage> {
        let inner = lock(&self.inner);
        let mut pages: Vec<IndexedPage> = inner
            .pages
            .values()
            .filter(|page| page.units.iter().all(|unit| unit.0 < inner.withdrawn))
            .flat_map(|page| {
                page.aliases.iter().map(|resume| IndexedPage {
                    group: page.group,
                    hash: page.hash,
                    tokens: page.tokens.clone(),
                    units: page.units.iter().map(|unit| unit.0).collect(),
                    resume: *resume,
                })
            })
            .collect();
        pages.sort_by_key(|page| page.units.first().copied());
        pages
    }

    /// Imports the predecessor world's published pages, each unit `offset`
    /// further into this pool, as unreferenced pages of `source`: requests
    /// read them from the inherited units. A page whose units fall outside
    /// the inherited units, or that holds a unit published already, is
    /// left out. Returns how many pages it imported.
    pub(crate) fn import(
        &self,
        pages: &[IndexedPage],
        offset: u32,
        source: &Arc<WorkerEndpoint>,
    ) -> usize {
        let mut inner = lock(&self.inner);
        let mut imported = 0;
        for page in pages {
            let Some(units) = page
                .units
                .iter()
                .map(|unit| unit.checked_add(offset))
                .collect::<Option<Vec<u32>>>()
            else {
                continue;
            };
            let Some(&head) = units.first() else {
                continue;
            };
            if let Some(existing) = inner.pages.get_mut(&head)
                && existing.hash == page.hash
                && existing.group == page.group
                && existing.tokens == page.tokens
                && existing
                    .units
                    .iter()
                    .map(|unit| unit.0)
                    .eq(units.iter().copied())
                && existing.source == *source
            {
                if !existing.aliases.contains(&page.resume) {
                    existing.aliases.push(page.resume);
                    if page.resume.is_none() {
                        inner.stats.pages_stored += 1;
                    }
                }
                continue;
            }
            if units.iter().any(|unit| {
                !inner.inherited.contains(unit)
                    || inner.meta[*unit as usize].published.is_some()
                    || inner.meta[*unit as usize].ref_cnt > 0
            }) {
                continue;
            }
            for unit in &units {
                let meta = &mut inner.meta[*unit as usize];
                meta.published = Some(head);
                meta.state = UnitState::Cached;
            }
            inner.pages.insert(
                head,
                PublishedPage {
                    aliases: vec![page.resume],
                    hash: page.hash,
                    group: page.group,
                    tokens: page.tokens.clone(),
                    source: Arc::clone(source),
                    units: units.into_iter().map(UnitId).collect(),
                    // The predecessor's offloads keyed the pages it published.
                    host_key: None,
                },
            );
            inner.hash_to_pages.entry(page.hash).or_default().push(head);
            if page.resume.is_none() {
                inner.stats.pages_stored += 1;
            }
            imported += 1;
        }
        imported
    }

    /// Retires a completed request's private aliases without invalidating
    /// public reuse or another continuation of the same physical pages.
    pub(crate) fn forget_resume(&self, resume: proto::request::CacheResume) {
        let mut inner = lock(&self.inner);
        let mut retired = Vec::new();
        for (&head, page) in &mut inner.pages {
            page.aliases.retain(|alias| *alias != Some(resume));
            if page.aliases.is_empty() {
                retired.push(head);
            }
        }
        for head in retired {
            inner.remove_page(head);
        }
    }

    /// Returns the allocatable units, excluding the unit-zero sentinel.
    pub(crate) fn usable_units(&self) -> usize {
        lock(&self.inner).stats.total
    }

    /// Returns the units allocatable once no unit is withdrawn: the most a
    /// request can ever take.
    pub(crate) fn capacity_units(&self) -> usize {
        let inner = lock(&self.inner);
        inner.meta.len() - 1 - inner.scratch as usize
    }

    /// Returns the queued units, including `Cached` ones, which allocation
    /// evicts when it reaches them.
    pub(crate) fn free_units(&self) -> usize {
        lock(&self.inner).free_count
    }

    /// Returns a snapshot of the current statistics.
    pub(crate) fn stats(&self) -> BlockPoolStats {
        lock(&self.inner).stats
    }

    /// Reserves `count` pages of `units_per_page` units each, evicting cached
    /// contents as necessary.
    ///
    /// Takes the units from the head of the free queue in order. A unit that
    /// still belongs to a published page evicts that page. Returns the pages
    /// `Reserved` with one reference each, or `None`, changing nothing, when
    /// fewer units are free than the pages need.
    pub(crate) fn allocate(&self, count: usize, units_per_page: usize) -> Option<Vec<PageRef>> {
        let mut inner = lock(&self.inner);
        let needed = count.checked_mul(units_per_page)?;
        if needed > inner.free_count || units_per_page == 0 {
            return None;
        }

        // Page references are built only after selection succeeds: dropping
        // one here would re-enter this pool's lock.
        let mut pages = Vec::with_capacity(count);
        for _ in 0..count {
            let mut units = Vec::with_capacity(units_per_page);
            for _ in 0..units_per_page {
                // The free count covers every unit taken here.
                let Some(index) = inner.fq_head else {
                    break;
                };
                inner.fq_unlink(index);
                if let Some(head) = inner.meta[index as usize].published {
                    inner.remove_page(head);
                }
                let meta = &mut inner.meta[index as usize];
                meta.state = UnitState::Reserved;
                meta.ref_cnt = 1;
                units.push(UnitId(index));
            }
            pages.push(PageRef {
                units: units.into_boxed_slice(),
                pool: Arc::downgrade(&self.inner),
            });
        }
        inner.stats.allocations += 1;
        inner.stats.free = inner.free_count;
        Some(pages)
    }

    /// Reserve logical units in the unmapped tail for host-staged pages.
    /// Only a standby cache uses these references. The caller maps their
    /// segments and copies the bytes before admitting any request to them.
    pub(crate) fn reserve_future(&self, units_per_page: usize) -> Option<PageRef> {
        let mut inner = lock(&self.inner);
        let units: Vec<_> = (inner.withdrawn.max(inner.scratch + 1)..inner.meta.len() as u32)
            .filter(|unit| {
                !inner.inherited.contains(unit) && inner.meta[*unit as usize].ref_cnt == 0
            })
            .take(units_per_page)
            .map(UnitId)
            .collect();
        if units.is_empty() || units.len() != units_per_page {
            return None;
        }
        for unit in &units {
            let meta = &mut inner.meta[unit.0 as usize];
            meta.state = UnitState::Reserved;
            meta.ref_cnt = 1;
        }
        Some(PageRef {
            units: units.into_boxed_slice(),
            pool: Arc::downgrade(&self.inner),
        })
    }

    /// Marks this pool's `Reserved` units of `pages` as `Active`.
    ///
    /// Units in any other state, or pages from another pool, are unchanged.
    pub(crate) fn activate<'a>(&self, pages: impl IntoIterator<Item = &'a PageRef>) {
        let mut inner = lock(&self.inner);
        for page in pages {
            // Only this pool's units are activated; the page's pool is
            // compared by address, which needs no upgrade.
            if !std::ptr::eq(page.pool.as_ptr(), Arc::as_ptr(&self.inner)) {
                continue;
            }
            for unit in page.units.iter() {
                let meta = &mut inner.meta[unit.0 as usize];
                if meta.state == UnitState::Reserved {
                    meta.state = UnitState::Active;
                }
            }
        }
    }

    /// Returns a unit's lifecycle state.
    #[cfg(test)]
    fn unit_state(&self, unit: UnitId) -> UnitState {
        lock(&self.inner).meta[unit.0 as usize].state
    }

    /// Publishes one physical prefix page of `group` under its loaded Worker
    /// identity.
    ///
    /// Callers publish only pages whose contents the worker has already
    /// computed. The call does nothing when `source` already publishes these
    /// exact tokens under `hash` in `group` on some page. Otherwise any
    /// earlier numerical identity of the page is removed first. Equal bytes
    /// keep their public and private aliases on the same physical units.
    pub(crate) fn publish(
        &self,
        page: &PageRef,
        group: u32,
        hash: u64,
        tokens: &[u32],
        source: &Arc<WorkerEndpoint>,
        resume: Option<proto::request::CacheResume>,
    ) {
        let mut inner = lock(&self.inner);
        if inner
            .find_page(hash, group, tokens, source, resume)
            .is_some()
        {
            return;
        }

        let head = page.units[0].0;
        if let Some(published) = inner.pages.get_mut(&head)
            && published.units == page.units
            && published.hash == hash
            && published.group == group
            && published.tokens == tokens
            && published.source == *source
        {
            published.aliases.push(resume);
            if resume.is_none() {
                inner.stats.pages_stored += 1;
            }
            return;
        }

        // A unit belongs to at most one page, so the page's first unit keys
        // its identity; any earlier identity of these units is replaced.
        for unit in page.units.iter() {
            if let Some(head) = inner.meta[unit.0 as usize].published {
                inner.remove_page(head);
            }
        }
        for unit in page.units.iter() {
            inner.meta[unit.0 as usize].published = Some(head);
        }
        inner.pages.insert(
            head,
            PublishedPage {
                aliases: vec![resume],
                hash,
                group,
                tokens: tokens.to_vec(),
                source: Arc::clone(source),
                units: page.units.clone(),
                host_key: None,
            },
        );
        inner.hash_to_pages.entry(hash).or_default().push(head);
        if resume.is_none() {
            inner.stats.pages_stored += 1;
        }
    }

    /// Finds a prefix page of `group` retained by this exact loaded Worker.
    ///
    /// Returns the page's first unit and whether the free queue holds it:
    /// the page is unreferenced and allocatable, and its units count as free
    /// capacity until acquired. An inherited page's units are never queued,
    /// since the world does not write them. The lookup acquires nothing: the page can be evicted before
    /// the caller acts on it, so reuse must go through [`BlockPool::acquire`],
    /// which checks the identity again.
    pub(crate) fn lookup(
        &self,
        hash: u64,
        group: u32,
        tokens: &[u32],
        source: &WorkerEndpoint,
        resume: Option<proto::request::CacheResume>,
    ) -> Option<(UnitId, bool)> {
        let inner = lock(&self.inner);
        let head = inner.find_page(hash, group, tokens, source, resume)?;
        Some((UnitId(head), inner.meta[head as usize].in_fq))
    }

    /// Acquires an active reference to a published page that still matches
    /// its identity.
    ///
    /// Unreferenced units of the page leave the free queue and stop counting
    /// as free capacity. Returns `None` when `head` keys no page with this
    /// identity or a reference count would overflow.
    pub(crate) fn acquire(
        &self,
        head: UnitId,
        hash: u64,
        group: u32,
        tokens: &[u32],
        source: &WorkerEndpoint,
        resume: Option<proto::request::CacheResume>,
    ) -> Option<PageRef> {
        let mut inner = lock(&self.inner);
        let page = inner.pages.get(&head.0)?;
        if page.hash != hash
            || page.group != group
            || page.tokens != tokens
            || page.source.as_ref() != source
            || !page.aliases.contains(&resume)
        {
            return None;
        }
        let units = page.units.clone();
        if units
            .iter()
            .any(|unit| inner.meta[unit.0 as usize].ref_cnt == u32::MAX)
        {
            return None;
        }
        for unit in units.iter() {
            inner.fq_unlink(unit.0);
            let meta = &mut inner.meta[unit.0 as usize];
            meta.ref_cnt += 1;
            meta.state = UnitState::Active;
        }
        inner.stats.free = inner.free_count;
        Some(PageRef {
            units,
            pool: Arc::downgrade(&self.inner),
        })
    }

    /// Notes `key` as the host KV cache key of the published page `page`,
    /// which is being offloaded there under it.
    pub(crate) fn set_host_key(&self, page: &PageRef, key: storage::Key) {
        let mut inner = lock(&self.inner);
        if let Some(published) = inner.pages.get_mut(&page.units[0].0)
            && published.units == page.units
        {
            published.host_key = Some(key);
        }
    }

    /// Acquires every idle cached page with a unit from `first` on, short of
    /// the withdrawn units: the pages a withdrawal from `first` evicts, held
    /// so that their bytes stay until the caller lets them go.
    pub(crate) fn acquire_idle_from(&self, first: u32) -> Vec<IdlePage> {
        let mut inner = lock(&self.inner);
        let end = inner.withdrawn.min(inner.meta.len() as u32);
        let heads: Vec<u32> = inner
            .pages
            .iter()
            .filter(|(_, page)| {
                page.aliases.contains(&None)
                    && page.units.iter().any(|unit| (first..end).contains(&unit.0))
                    && page.units.iter().all(|unit| {
                        let meta = &inner.meta[unit.0 as usize];
                        meta.state == UnitState::Cached && meta.ref_cnt == 0
                    })
            })
            .map(|(&head, _)| head)
            .collect();
        let mut idle = Vec::with_capacity(heads.len());
        for head in heads {
            let Some(page) = inner.pages.get(&head) else {
                continue;
            };
            let (units, group, source, host_key) = (
                page.units.clone(),
                page.group,
                Arc::clone(&page.source),
                page.host_key,
            );
            for unit in units.iter() {
                inner.fq_unlink(unit.0);
                let meta = &mut inner.meta[unit.0 as usize];
                meta.ref_cnt += 1;
                meta.state = UnitState::Active;
            }
            idle.push(IdlePage {
                page: PageRef {
                    units,
                    pool: Arc::downgrade(&self.inner),
                },
                group,
                source,
                host_key,
            });
        }
        inner.stats.free = inner.free_count;
        idle
    }

    /// Returns the number of published prefix pages, referenced or not.
    pub(crate) fn cached_pages(&self) -> usize {
        lock(&self.inner).pages.len()
    }

    /// Revokes every published page's prefix identity, as
    /// [`Self::invalidate_source`] does for one incarnation: later prompts
    /// compute their prefixes anew. Returns how many pages it revoked.
    pub(crate) fn flush(&self) -> usize {
        let mut inner = lock(&self.inner);
        let heads = inner.pages.keys().copied().collect::<Vec<_>>();
        for &head in &heads {
            inner.remove_page(head);
        }
        heads.len()
    }

    /// Revokes cache lookup for one lost incarnation without freeing live
    /// request units.
    ///
    /// Referenced units keep their owners and reference counts; only their
    /// prefix identity is removed. Unreferenced units turn from `Cached` into
    /// `Free` without changing the free capacity.
    pub(crate) fn invalidate_source(&self, source: &WorkerEndpoint) {
        let mut inner = lock(&self.inner);
        let heads = inner
            .pages
            .iter()
            .filter(|(_, page)| page.source.as_ref() == source)
            .map(|(head, _)| *head)
            .collect::<Vec<_>>();
        for head in heads {
            inner.remove_page(head);
        }
    }
}

impl fmt::Debug for BlockPool {
    /// Formats the value for diagnostic output.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.stats();
        formatter
            .debug_struct("BlockPool")
            .field("usable_units", &stats.total)
            .field("free_units", &stats.free)
            .finish()
    }
}

/// The page shape and history policy of one cache group, in scheduler terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GroupShape {
    /// Tokens per logical page.
    pub(crate) block_size: usize,
    /// Units one logical page occupies.
    pub(crate) units_per_page: usize,
    /// History tokens any reader needs, or `None` for full attention. A
    /// state group's reader needs one: the checkpoint ending at its first
    /// token.
    pub(crate) window: Option<usize>,
    /// Whether the group holds state checkpoints, whose tables hold units
    /// only for the pages that hold a checkpoint
    /// ([`GroupShape::checkpoint_page`]).
    pub(crate) state: bool,
    /// Whether the group holds the draft cache, optional state a request
    /// may lack: its tables take units as the pool allows
    /// ([`PrefixCacheCoordinator::ensure_capacity`]), its pages never bound
    /// a reuse of the other groups, and a page's key covers the token after
    /// the page, which its last entry reads.
    pub(crate) draft: bool,
}

/// Where a call growing a request's tables checkpoints its state groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Checkpoint {
    /// A prompt call (or a load of a reused prefix) keeps the state at the
    /// last page boundary it reaches.
    Boundary,
    /// A decode call keeps the state at a boundary only when it is a
    /// multiple of the state interval.
    Interval(usize),
    /// Positions a call's drafter writes past its tokens keep no state.
    Never,
}

impl GroupShape {
    /// A K/V group's shape: pages of `block_size` tokens of `units_per_page`
    /// units, reading the `window` tokens before each reader, or every
    /// earlier token without one.
    #[cfg(test)]
    pub(crate) fn attention(
        block_size: usize,
        units_per_page: usize,
        window: Option<usize>,
    ) -> Self {
        Self {
            block_size,
            units_per_page,
            window,
            state: false,
            draft: false,
        }
    }

    /// The logical page a call reading from token `read_start` and holding
    /// `total_tokens` tokens checkpoints into: the page ending at the last
    /// boundary it reaches, when that boundary follows `read_start` and the
    /// call checkpoints it. `None` for a K/V group.
    fn checkpoint_page(
        &self,
        read_start: usize,
        total_tokens: usize,
        checkpoint: Checkpoint,
    ) -> Option<usize> {
        if !self.state {
            return None;
        }
        let boundary = total_tokens / self.block_size * self.block_size;
        let kept = match checkpoint {
            Checkpoint::Boundary => true,
            Checkpoint::Interval(interval) => boundary.is_multiple_of(interval.max(1)),
            Checkpoint::Never => false,
        };
        (boundary > read_start && kept).then(|| boundary / self.block_size - 1)
    }

    /// The reuse rules of the group's attention type; a state group's are
    /// those of a one-token window.
    pub(crate) fn manager(&self) -> manager::AttentionManager {
        match self.window {
            None => manager::AttentionManager::Full(manager::FullAttentionManager {
                block_size: self.block_size,
            }),
            Some(window) => {
                manager::AttentionManager::SlidingWindow(manager::SlidingWindowManager {
                    block_size: self.block_size,
                    window,
                })
            }
        }
    }

    /// Derives the scheduler shape of a worker-reported group.
    pub(crate) fn from_group(group: &CacheGroup) -> Self {
        let state = group.kind.is_state();
        Self {
            block_size: group.block_size as usize,
            units_per_page: group.units_per_page as usize,
            window: if state {
                Some(1)
            } else {
                group.kind.window().map(|window| window as usize)
            },
            state,
            draft: group.kind.is_draft(),
        }
    }

    /// Tokens a page's key covers: its own, and for the draft cache the
    /// token after it.
    fn key_tokens(&self) -> usize {
        self.block_size + usize::from(self.draft)
    }

    /// Returns the first logical page a read starting at token `read_start`
    /// needs: page zero for full attention, otherwise the page holding the
    /// oldest history token of the window.
    fn first_needed_page(&self, read_start: usize) -> usize {
        self.window.map_or(0, |window| {
            read_start.saturating_sub(window) / self.block_size
        })
    }

    /// Returns the logical pages that cover `tokens` tokens.
    fn pages_covering(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.block_size)
    }
}

/// One logical page held by a table, with the allocation serial of its first
/// unit when the table allocated it fresh.
#[derive(Debug)]
struct TablePage {
    /// The page's units; `None` for a state group's page that holds no
    /// checkpoint, which workers see as the padding sentinel and never read.
    page: Option<PageRef>,
    /// `None` for a page acquired from the prefix cache, which already holds
    /// computed keys and values; otherwise the owning allocation's unit serial at the
    /// page's allocation.
    serial: Option<u64>,
}

/// The pages a call needs a table to gain: the table's start, how many
/// pages it appends and which of them take units, and which page it already
/// holds without units that now takes them.
struct Growth {
    start: usize,
    /// One entry per appended page: whether it takes units.
    appended: Vec<bool>,
    /// A page the table holds without units that takes them now.
    filled: Option<usize>,
}

impl Growth {
    /// The pages the growth allocates units for.
    fn pages(&self) -> usize {
        self.appended.iter().filter(|held| **held).count() + usize::from(self.filled.is_some())
    }
}

/// One sequence's logical-page to physical-unit mapping in one cache group.
///
/// Logical pages `start_page..end_page` are held; earlier pages are retired
/// tombstones that no worker reads, which only a sliding-window group and a
/// state group produce. A state group's table holds units only for its
/// pages that hold a checkpoint.
#[derive(Debug)]
pub(crate) struct BlockTable {
    group_id: usize,
    shape: GroupShape,
    start_page: usize,
    pages: VecDeque<TablePage>,
    /// `(start_page, end_page)` most recently declared to workers, or `None`
    /// before the first declaration, after a rollback, or once a page the
    /// table held without units took them.
    sent: Option<(usize, usize)>,
}

impl BlockTable {
    /// Creates an empty table for one cache group.
    pub(crate) fn new(group_id: usize, shape: GroupShape) -> Self {
        assert!(
            shape.block_size > 0 && shape.units_per_page > 0,
            "cache page shape must be positive"
        );
        Self {
            group_id,
            shape,
            start_page: 0,
            pages: VecDeque::new(),
            sent: None,
        }
    }

    /// Returns the cache group identifier.
    pub(crate) fn group_id(&self) -> usize {
        self.group_id
    }

    /// Returns the first logical page still held.
    pub(crate) fn start_page(&self) -> usize {
        self.start_page
    }

    /// Returns the logical page after the last held one.
    pub(crate) fn end_page(&self) -> usize {
        self.start_page + self.pages.len()
    }

    /// Returns the absolute token extent the table covers.
    pub(crate) fn capacity_tokens(&self) -> usize {
        self.end_page() * self.shape.block_size
    }

    /// Physical units held by this table, excluding sparse checkpoint holes.
    pub(crate) fn held_units(&self) -> usize {
        self.pages
            .iter()
            .filter_map(|entry| entry.page.as_ref())
            .map(|page| page.units().len())
            .sum()
    }

    /// Returns the held pages' units, page-major from `start_page`; a page
    /// without units names the padding sentinel, unit 0, in each position.
    pub(crate) fn unit_ids(&self) -> Vec<UnitId> {
        let sentinel = vec![UnitId(0); self.shape.units_per_page];
        self.pages
            .iter()
            .flat_map(|entry| {
                entry
                    .page
                    .as_ref()
                    .map_or(sentinel.as_slice(), |page| page.units())
                    .iter()
                    .copied()
            })
            .collect()
    }

    /// Tokens a page's key covers: its own, and for the draft cache the
    /// token after it.
    pub(crate) fn key_tokens(&self) -> usize {
        self.shape.key_tokens()
    }

    /// Returns the held page at absolute logical `index`, if held with
    /// units.
    pub(crate) fn page(&self, index: usize) -> Option<&PageRef> {
        index
            .checked_sub(self.start_page)
            .and_then(|offset| self.pages.get(offset))
            .and_then(|entry| entry.page.as_ref())
    }

    /// Returns the units of fresh pages allocated at or after `serial`, which
    /// a worker must reset before any call uses them.
    pub(crate) fn fresh_units(&self, serial: u64) -> Vec<UnitId> {
        self.pages
            .iter()
            .filter(|entry| entry.serial.is_some_and(|value| value >= serial))
            .filter_map(|entry| entry.page.as_ref())
            .flat_map(|page| page.units().iter().copied())
            .collect()
    }

    /// Whether the held page interval differs from the one last declared.
    pub(crate) fn changed_since_sent(&self) -> bool {
        self.sent != Some((self.start_page, self.end_page()))
    }

    /// Marks the table's `Reserved` units as `Active`.
    pub(crate) fn activate(&self, pool: &BlockPool) {
        pool.activate(self.pages.iter().filter_map(|entry| entry.page.as_ref()));
    }

    /// Records the held page interval as declared to workers.
    pub(crate) fn mark_sent(&mut self) {
        self.sent = Some((self.start_page, self.end_page()));
    }

    /// Forgets the last declaration, so the next dispatch resends the table.
    pub(crate) fn clear_sent(&mut self) {
        self.sent = None;
    }

    /// Returns the pages a call reading from token `read_start` and holding
    /// `total_tokens` tokens still needs.
    ///
    /// Pages are only appended, so a non-empty table keeps its start. An
    /// empty table that never held a page starts at the first page the read
    /// needs; a full-attention table always starts at page zero. A K/V
    /// table's pages all take units; a state table's take units only for
    /// the page the call checkpoints into ([`GroupShape::checkpoint_page`]),
    /// which may be a page the table already holds without them, or holds
    /// in units `shared` says other readers read. The forward writes its
    /// checkpoint into whatever units the table names for that page, so a
    /// call that evaluates tokens again up to a checkpoint the table holds
    /// from the prefix cache (a draft cache refill) writes it into fresh
    /// units of its own, and the shared page stays as its readers found it.
    fn demand(
        &self,
        read_start: usize,
        total_tokens: usize,
        checkpoint: Checkpoint,
        shared: impl Fn(&PageRef) -> bool,
    ) -> Growth {
        let fresh = self.pages.is_empty() && self.start_page == 0;
        let start = if fresh {
            self.shape.first_needed_page(read_start)
        } else {
            self.start_page
        };
        let end = self.shape.pages_covering(total_tokens);
        let current_end = if fresh { start } else { self.end_page() };
        let target = self
            .shape
            .checkpoint_page(read_start, total_tokens, checkpoint);
        let appended = (current_end..end.max(current_end))
            .map(|page| !self.shape.state || target == Some(page))
            .collect();
        let filled = target.filter(|&page| {
            page >= start
                && page < current_end
                && page
                    .checked_sub(self.start_page)
                    .and_then(|offset| self.pages.get(offset))
                    .is_some_and(|entry| entry.page.as_ref().is_none_or(&shared))
        });
        Growth {
            start,
            appended,
            filled,
        }
    }

    /// Releases the units of the held pages before absolute logical page
    /// `bound` except `keep`, then retires the leading pages without units.
    ///
    /// A state table keeps the checkpoint a refill of the request's draft
    /// cache continues from while its decoding moves on.
    fn retire_except(&mut self, keep: &[Option<usize>], bound: usize) {
        let mut released = false;
        for page in self.start_page..bound.min(self.end_page()) {
            if keep.contains(&Some(page)) {
                continue;
            }
            if let Some(entry) = self.pages.get_mut(page - self.start_page)
                && entry.page.take().is_some()
            {
                entry.serial = None;
                released = true;
            }
        }
        if released {
            self.clear_sent();
        }
        self.retire_before(
            keep.iter()
                .flatten()
                .copied()
                .min()
                .unwrap_or(bound)
                .min(bound),
        );
    }

    /// Whether every page covering tokens `[start, end)` holds units.
    pub(crate) fn holds(&self, start: usize, end: usize) -> bool {
        if start >= end {
            return true;
        }
        let first = start / self.shape.block_size;
        let last = end.div_ceil(self.shape.block_size);
        first >= self.start_page
            && last <= self.end_page()
            && (first..last).all(|page| self.page(page).is_some())
    }

    /// Releases every page's units, keeping the pages without them: the
    /// table holds none of its cache. Returns the pages released.
    fn release_units(&mut self) -> usize {
        let mut released = 0;
        for entry in &mut self.pages {
            if entry.page.take().is_some() {
                entry.serial = None;
                released += 1;
            }
        }
        if released > 0 {
            self.clear_sent();
        }
        released
    }

    /// Retires held pages before absolute logical page `page`.
    ///
    /// Dropping a page releases its units: a published page's units stay
    /// matchable as `Cached`, others become `Free`.
    fn retire_before(&mut self, page: usize) {
        while self.start_page < page {
            if self.pages.pop_front().is_none() {
                // An empty table moves its origin without holding pages.
                self.start_page = page;
                return;
            }
            self.start_page += 1;
        }
    }
}

/// One sequence's page tables, one per cache group in group order.
///
/// Every page is a counted reference, so dropping the allocation releases the
/// sequence's references, and a page shared through the prefix cache stays
/// resident while another table still holds it.
#[derive(Debug)]
pub(crate) struct CacheAllocation {
    pub(crate) tables: Vec<BlockTable>,
    /// Units allocated fresh to the tables so far. Each fresh page records
    /// the serial of its first unit, so a dispatch can name the units
    /// allocated after a watermark.
    allocated_units: u64,
    /// Immutable source pages retained until the first forward over their
    /// private copies completes. The destination belongs to `tables`.
    copies: Vec<(usize, usize, PageRef)>,
}

impl CacheAllocation {
    /// Returns the serial of the next freshly allocated unit.
    pub(crate) fn allocated_units(&self) -> u64 {
        self.allocated_units
    }

    /// Release immutable sources after the first forward's device work,
    /// including its initial unit copies, has completed.
    pub(crate) fn complete_copies(&mut self) {
        self.copies.clear();
    }

    /// Copies accompanying the fresh units declared since `serial`.
    pub(crate) fn copies(&self, group: usize, serial: u64) -> Vec<(UnitId, UnitId)> {
        self.copies
            .iter()
            .filter(|(owner, _, _)| *owner == group)
            .filter_map(|(_, page, source)| {
                let table = &self.tables[group];
                // A deeper host hit can move a sliding window past the
                // local tail before its first forward is submitted.
                let offset = page.checked_sub(table.start_page)?;
                let entry = table.pages.get(offset)?;
                let target = entry
                    .page
                    .as_ref()
                    .filter(|_| entry.serial.is_some_and(|value| value >= serial))?;
                Some((source, target))
            })
            .flat_map(|(source, target)| {
                source
                    .units()
                    .iter()
                    .copied()
                    .zip(target.units().iter().copied())
            })
            .collect()
    }

    /// Forces every table to be declared again at the next dispatch.
    pub(crate) fn clear_sent(&mut self) {
        for table in &mut self.tables {
            table.clear_sent();
        }
    }
}

/// Prefix lookup result for one scheduler-owned sequence.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct PrefixLookup {
    /// Chained hash of every complete prompt page, indexed by group and then
    /// by logical page. The scheduler keeps these to publish the prompt
    /// through [`PrefixCacheCoordinator::publish_computed`] as prefill
    /// completes, and extends them over the tokens the request generates.
    /// Empty when prefix caching does not apply to the request.
    pub(crate) page_hashes: Vec<Vec<u64>>,
    /// Leading prompt tokens acquired from the prefix cache. A final
    /// partial page is copied before the resumed computation appends to it.
    pub(crate) cached_tokens: usize,
    /// Leading prompt tokens whose draft cache was acquired from the prefix
    /// cache with them; at most `cached_tokens`, zero without a draft
    /// cache.
    pub(crate) draft_tokens: usize,
}

/// Read-only prefix-cache probe result across every cache group.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct PrefixHit {
    /// Leading prompt tokens every required group can reuse, including a
    /// stored partial tail when the group's computation permits it.
    pub(crate) cached_tokens: usize,
    /// Units of the pages a hit would acquire that the free queue holds:
    /// unreferenced and allocatable, which an inherited unit never is.
    ///
    /// Those units count as free capacity until acquired, so admission
    /// subtracts them from the free units.
    pub(crate) cached_free_units: usize,
}

/// The pages a validated prefix of `tokens` tokens needs from each group.
#[derive(Debug, Default)]
struct PrefixSelection {
    tokens: usize,
    resume: Option<proto::request::CacheResume>,
    /// Per group, the group's first needed page and, for it and each later
    /// needed page in logical order, the page's first unit and whether the
    /// page is currently unreferenced. The draft cache's are its leading
    /// pages the prefix holds, which may be fewer than the prefix's.
    pages: Vec<(usize, Vec<(UnitId, bool)>)>,
    /// A final immutable partial page per group. Admission copies it into
    /// fresh writable units; it never appends into the cached source.
    tails: Vec<Option<TailPage>>,
}

#[derive(Debug, Clone, Copy)]
struct TailPage {
    head: UnitId,
    cached: bool,
    hash: u64,
    tokens: usize,
}

/// The namespace a request's prefix pages are hashed in: its isolation key
/// (`CachePolicy::isolation_key`), and the seeds of the world whose state it
/// continues with (`CachePolicy::state_seeds`) in place of the pool's own.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PrefixScope<'a> {
    pub(crate) isolation_key: Option<u64>,
    pub(crate) state_seeds: &'a [u64],
    pub(crate) resume: Option<proto::request::CacheResume>,
}

impl<'a> PrefixScope<'a> {
    /// The scope `cache` states.
    pub(crate) fn of(cache: &'a proto::request::CachePolicy) -> Self {
        Self {
            isolation_key: cache.isolation_key,
            state_seeds: &cache.state_seeds,
            resume: cache.resume,
        }
    }
}

/// The seed of the prefix hash chains of the pages a worker computes, from
/// the identity it reports of its model and of what its forwards compute
/// (`PrefixCacheInfo::identity`): the first eight bytes of its SHA-256, so
/// that every build derives the same seed from the same identity. Zero for
/// a worker that reports none.
pub(crate) fn prefix_seed(identity: &str) -> u64 {
    use sha2::Digest as _;
    if identity.is_empty() {
        return 0;
    }
    let digest = sha2::Sha256::digest(identity.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().unwrap_or_default())
}

/// Coordinates capacity, window release and prefix state across cache
/// groups.
///
/// Callers pass one [`CacheAllocation`] whose tables match `groups` in order;
/// the methods return `None` or `false` for any other table set.
#[derive(Debug)]
pub(crate) struct PrefixCacheCoordinator {
    groups: Vec<GroupShape>,
    prefix_enabled: bool,
    /// Root of each group's hash chain; an isolation key is mixed into it.
    hash_seeds: Vec<u64>,
}

impl PrefixCacheCoordinator {
    /// Creates a coordinator for the given cache groups with prefix caching
    /// enabled and a zero chain seed.
    pub(crate) fn new(groups: Vec<GroupShape>) -> Self {
        Self {
            hash_seeds: vec![0; groups.len()],
            groups,
            prefix_enabled: true,
        }
    }

    /// Root subsequent page hashes at each group's upstream computation.
    pub(crate) fn set_seeds(&mut self, seeds: Vec<u64>) {
        assert_eq!(seeds.len(), self.groups.len());
        self.hash_seeds = seeds;
    }

    pub(crate) fn seeds(&self) -> &[u64] {
        &self.hash_seeds
    }

    fn group_seed(&self, group: usize, scope: PrefixScope<'_>) -> u64 {
        // A predecessor without speculation has no draft group. Its
        // required state retains its roots; new optional draft state starts
        // under the serving computation's own key when refilled.
        let root = scope
            .state_seeds
            .get(group)
            .copied()
            .unwrap_or(self.hash_seeds[group]);
        scope
            .isolation_key
            .map_or(root, |key| root ^ key.rotate_left(17))
    }

    /// Returns the cache groups in table order.
    pub(crate) fn groups(&self) -> &[GroupShape] {
        &self.groups
    }

    /// Enables or disables prefix-cache lookup and insertion.
    pub(crate) fn set_prefix_enabled(&mut self, enabled: bool) {
        self.prefix_enabled = enabled;
    }

    /// Returns the largest block size of any group a request needs, in
    /// tokens: every reuse boundary is a multiple of it. The draft cache's
    /// pages, which a request may lack, bound no reuse.
    pub(crate) fn max_block_size(&self) -> usize {
        self.groups
            .iter()
            .filter(|group| !group.draft)
            .map(|group| group.block_size)
            .max()
            .unwrap_or(1)
    }

    /// The draft cache's group, when the pool keeps one.
    pub(crate) fn draft_group(&self) -> Option<usize> {
        self.groups.iter().position(|group| group.draft)
    }

    /// Creates empty tables for every group.
    pub(crate) fn empty(&self) -> CacheAllocation {
        CacheAllocation {
            tables: self
                .groups
                .iter()
                .enumerate()
                .map(|(group, shape)| BlockTable::new(group, *shape))
                .collect(),
            allocated_units: 0,
            copies: Vec::new(),
        }
    }

    /// Whether `allocation` holds one table per group in group order.
    fn matches(&self, allocation: &CacheAllocation) -> bool {
        allocation.tables.len() == self.groups.len()
            && allocation
                .tables
                .iter()
                .enumerate()
                .all(|(group, table)| table.group_id == group && table.shape == self.groups[group])
    }

    /// Returns the units that growing `allocation` for a call reading from
    /// token `read_start` and holding `total_tokens` tokens would allocate.
    ///
    /// A full-attention group needs every page before `total_tokens`; a
    /// sliding-window group needs only the pages intersecting
    /// `[read_start - window, total_tokens)`; a state group only the page
    /// the call checkpoints into, unless the table holds it in units of its
    /// own; the draft cache, which takes units only as the pool allows,
    /// none. Returns `None` when the tables do not match the groups.
    pub(crate) fn units_needed(
        &self,
        pool: &BlockPool,
        allocation: &CacheAllocation,
        read_start: usize,
        total_tokens: usize,
        checkpoint: Checkpoint,
    ) -> Option<usize> {
        if !self.matches(allocation) {
            return None;
        }
        Some(
            allocation
                .tables
                .iter()
                .filter(|table| !table.shape.draft)
                .map(|table| {
                    table
                        .demand(read_start, total_tokens, checkpoint, |page| {
                            pool.shared(page)
                        })
                        .pages()
                        * table.shape.units_per_page
                })
                .sum(),
        )
    }

    /// Returns the units a request that will reuse a `prefix_tokens`-long
    /// cached prefix allocates to hold `total_tokens` tokens.
    ///
    /// Complete prefix pages are shared. A partial tail needs a fresh copy,
    /// so the first allocated page starts at the floor of the prefix's page
    /// index. A state group allocates the checkpoint at the last boundary
    /// reached by the first call.
    pub(crate) fn units_after_prefix(&self, prefix_tokens: usize, total_tokens: usize) -> usize {
        self.groups
            .iter()
            .filter(|group| !group.draft)
            .map(|group| {
                let pages = if group.state {
                    usize::from(
                        group
                            .checkpoint_page(prefix_tokens, total_tokens, Checkpoint::Boundary)
                            .is_some(),
                    )
                } else {
                    group
                        .pages_covering(total_tokens)
                        .saturating_sub(prefix_tokens / group.block_size)
                };
                pages * group.units_per_page
            })
            .sum()
    }

    /// Returns the units that tables holding every page of `tokens` tokens
    /// in every group occupy, as a worst-case reservation made up front. A
    /// state group's table holds one checkpoint, at the last boundary.
    pub(crate) fn units_for_tokens(&self, tokens: usize) -> usize {
        self.groups
            .iter()
            .filter(|group| !group.draft)
            .map(|group| {
                let pages = if group.state {
                    usize::from(tokens >= group.block_size)
                } else {
                    group.pages_covering(tokens)
                };
                pages * group.units_per_page
            })
            .sum()
    }

    /// Returns the fewest units that must coexist to compute a `tokens`-long
    /// prompt in chunks of at most `chunk` tokens.
    ///
    /// Full-attention groups hold every page; a sliding-window group holds
    /// only the pages one chunk's window and query can intersect; a state
    /// group the checkpoint a chunk starts from and the one it writes.
    pub(crate) fn min_units_for_prompt(&self, tokens: usize, chunk: usize) -> usize {
        self.groups
            .iter()
            .filter(|group| !group.draft)
            .map(|group| {
                let pages = group.pages_covering(tokens);
                let pages = if group.state {
                    pages.min(2)
                } else {
                    group.window.map_or(pages, |window| {
                        pages.min((window + chunk.max(1)).div_ceil(group.block_size) + 1)
                    })
                };
                pages * group.units_per_page
            })
            .sum()
    }

    /// Grows every group table atomically for a call that reads from token
    /// `read_start` and holds `total_tokens` tokens, checkpointing its
    /// state groups as `checkpoint` says.
    ///
    /// The sum of every group's unit demand must fit the free units; the call
    /// then allocates them all or nothing. Returns `None`, with every table
    /// unchanged, when the tables do not match the groups or the pool lacks
    /// free units. The draft cache's table then grows as the units left
    /// allow ([`Self::grow_draft`]).
    pub(crate) fn ensure_capacity(
        &self,
        pool: &BlockPool,
        allocation: &mut CacheAllocation,
        read_start: usize,
        total_tokens: usize,
        checkpoint: Checkpoint,
    ) -> Option<()> {
        let demand = self.units_needed(pool, allocation, read_start, total_tokens, checkpoint)?;
        let growths: Vec<Growth> = allocation
            .tables
            .iter()
            .map(|table| {
                if table.shape.draft {
                    // The draft table grows on its own below.
                    Growth {
                        start: table.start_page,
                        appended: Vec::new(),
                        filled: None,
                    }
                } else {
                    table.demand(read_start, total_tokens, checkpoint, |page| {
                        pool.shared(page)
                    })
                }
            })
            .collect();
        if demand > pool.free_units() {
            return None;
        }

        // Allocate every group's pages before appending any, so a shortage
        // (which the free-unit check above excludes) cannot leave a partial
        // growth. Dropping unappended pages releases them without holding
        // the pool's lock.
        let mut grown = Vec::with_capacity(allocation.tables.len());
        for (table, growth) in allocation.tables.iter().zip(&growths) {
            let count = growth.pages();
            let allocated = if count == 0 {
                Vec::new()
            } else {
                pool.allocate(count, table.shape.units_per_page)?
            };
            grown.push(allocated);
        }
        for ((table, growth), pages) in allocation.tables.iter_mut().zip(growths).zip(grown) {
            if table.pages.is_empty() && table.start_page == 0 {
                table.start_page = growth.start;
            }
            let mut pages = pages.into_iter();
            let mut take = |allocation_units: &mut u64| {
                let page = pages.next()?;
                let serial = *allocation_units;
                *allocation_units += page.units().len() as u64;
                Some(TablePage {
                    page: Some(page),
                    serial: Some(serial),
                })
            };
            if let Some(index) = growth.filled
                && let Some(entry) = take(&mut allocation.allocated_units)
                && let Some(slot) = table.pages.get_mut(index - table.start_page)
            {
                *slot = entry;
                // The page's units changed within the declared interval.
                table.clear_sent();
            }
            for held in growth.appended {
                let entry = if held {
                    take(&mut allocation.allocated_units)
                } else {
                    None
                };
                table.pages.push_back(entry.unwrap_or(TablePage {
                    page: None,
                    serial: None,
                }));
            }
        }
        if let Some(group) = self.draft_group() {
            self.grow_draft(pool, allocation, group, read_start, total_tokens);
        }
        Some(())
    }

    /// Grows the draft cache's table for a call reading from token
    /// `read_start` and holding `total_tokens` tokens: it appends the pages
    /// up to `total_tokens`, and the pages from `read_start`'s on that hold
    /// no units take them, all of them when the pool has the free units,
    /// none otherwise. A page left without units holds no draft cache: the
    /// call writes no entry there, and its request's draft cache stops
    /// before it until a refill fills it.
    fn grow_draft(
        &self,
        pool: &BlockPool,
        allocation: &mut CacheAllocation,
        group: usize,
        read_start: usize,
        total_tokens: usize,
    ) {
        let table = &mut allocation.tables[group];
        let shape = table.shape;
        let end = shape.pages_covering(total_tokens);
        while table.end_page() < end {
            table.pages.push_back(TablePage {
                page: None,
                serial: None,
            });
        }
        let first = (read_start / shape.block_size).max(table.start_page);
        let missing: Vec<usize> = (first..end)
            .filter(|&page| table.page(page).is_none())
            .collect();
        if missing.is_empty() || missing.len() * shape.units_per_page > pool.free_units() {
            return;
        }
        let Some(pages) = pool.allocate(missing.len(), shape.units_per_page) else {
            return;
        };
        for (index, page) in missing.into_iter().zip(pages) {
            let serial = allocation.allocated_units;
            allocation.allocated_units += page.units().len() as u64;
            if let Some(entry) = table.pages.get_mut(index - table.start_page) {
                *entry = TablePage {
                    page: Some(page),
                    serial: Some(serial),
                };
            }
        }
        // Pages within the declared interval took units.
        table.clear_sent();
    }

    /// Releases the units of `allocation`'s draft cache, keeping its pages
    /// without units: the request holds no draft cache from then on, until
    /// a refill gives its pages units again. Returns the pages released.
    pub(crate) fn release_draft(&self, allocation: &mut CacheAllocation) -> usize {
        if !self.matches(allocation) {
            return 0;
        }
        self.draft_group()
            .map_or(0, |group| allocation.tables[group].release_units())
    }

    /// Whether a call can resume the state of `allocation`'s request at
    /// token `tokens`: from nothing at token zero, and otherwise from the
    /// checkpoint its state table holds there. A pool without layers kept
    /// per request slot resumes anywhere.
    pub(crate) fn resumes_at(&self, allocation: &CacheAllocation, tokens: usize) -> bool {
        if !self.matches(allocation) {
            return false;
        }
        allocation
            .tables
            .iter()
            .filter(|table| table.shape.state)
            .all(|table| tokens == 0 || table.page(tokens / table.shape.block_size - 1).is_some())
    }

    /// The tokens between checkpoints of the pool's layers kept per request
    /// slot, `None` without such layers.
    pub(crate) fn state_block_size(&self) -> Option<usize> {
        self.groups
            .iter()
            .find(|group| group.state)
            .map(|group| group.block_size)
    }

    /// Makes `allocation`'s state tables hold their pages from the one
    /// ending at token `tokens` on, prepending pages without units before a
    /// table's start: a refill that evaluates the request's tokens from
    /// there checkpoints into them.
    pub(crate) fn hold_states_from(&self, allocation: &mut CacheAllocation, tokens: usize) {
        if !self.matches(allocation) {
            return;
        }
        for table in &mut allocation.tables {
            if !table.shape.state {
                continue;
            }
            let first = tokens / table.shape.block_size;
            let first = first.saturating_sub(usize::from(first > 0));
            if first < table.start_page {
                while table.start_page > first {
                    table.pages.push_front(TablePage {
                        page: None,
                        serial: None,
                    });
                    table.start_page -= 1;
                }
                table.clear_sent();
            }
        }
    }

    /// Whether `allocation`'s draft cache pages covering tokens `[start,
    /// end)` hold units; `false` without a draft cache.
    pub(crate) fn holds_draft(
        &self,
        allocation: &CacheAllocation,
        start: usize,
        end: usize,
    ) -> bool {
        self.matches(allocation)
            && self
                .draft_group()
                .is_some_and(|group| allocation.tables[group].holds(start, end))
    }

    /// Retires sliding-window and state pages no later reader needs.
    ///
    /// `visible` is the request's accepted cache extent after a completed call.
    /// Calls apply in submission order, so every earlier reader has
    /// completed, and every later call reads from at least `visible -
    /// window`. Pages that lie entirely before that bound are retired; a
    /// published page stays matchable as `Cached`. A state group keeps the
    /// latest complete checkpoint, which a migrating request continues from,
    /// and the one ending at `refill`, from which a refill of the request's
    /// draft cache continues. Full-attention tables are unchanged. Returns
    /// `false` when the tables do not match the groups.
    pub(crate) fn release_window(
        &self,
        allocation: &mut CacheAllocation,
        visible: usize,
        refill: Option<usize>,
    ) -> bool {
        if !self.matches(allocation) {
            return false;
        }
        for table in &mut allocation.tables {
            if let Some(window) = table.shape.window {
                let bound = visible.saturating_sub(window) / table.shape.block_size;
                if table.shape.state {
                    // Decode can cross many page boundaries between state
                    // checkpoints. Keep the latest page actually written,
                    // excluding partial pages and future reservations.
                    let complete = (visible / table.shape.block_size).min(table.end_page());
                    let latest = (table.start_page..complete)
                        .rev()
                        .find(|&page| table.page(page).is_some());
                    let refill = refill
                        .and_then(|tokens| (tokens / table.shape.block_size).checked_sub(1))
                        .filter(|&page| table.page(page).is_some());
                    table.retire_except(&[latest, refill], bound);
                } else {
                    table.retire_before(bound);
                }
            }
        }
        true
    }

    /// Computes group-specific chained hashes for each complete prompt page.
    ///
    /// A trailing partial page is not hashed, so only complete pages are ever
    /// published. The chains are rooted at the pool's seed, or at the seed
    /// of the world whose state the request continues with. An isolation key
    /// is rotated and XORed into the root, so requests with different keys
    /// compute different hash chains and, barring a hash collision, reuse
    /// none of each other's pages. A key of zero leaves the root unchanged
    /// and therefore shares pages with requests that carry no key.
    fn prefix_hashes(&self, prompt: &[u32], scope: PrefixScope) -> Vec<Vec<u64>> {
        let mut hashes = vec![Vec::new(); self.groups.len()];
        self.extend_hashes(&mut hashes, CachedTokens::prompt(prompt), scope);
        hashes
    }

    /// Extends each group's hash chain in `hashes` over the complete pages of
    /// `tokens` it does not cover yet, as [`Self::prefix_hashes`] chains a
    /// prompt's pages: a chain over a request's prompt extended over the
    /// tokens it generated equals the chain over the prompt of its
    /// migration, whose prompt continues with those tokens. `hashes` holds
    /// one chain per group, each a prefix of the chain over `tokens`.
    pub(crate) fn extend_hashes(
        &self,
        hashes: &mut [Vec<u64>],
        tokens: CachedTokens<'_>,
        scope: PrefixScope,
    ) {
        for (group, (shape, chain)) in self.groups.iter().zip(hashes.iter_mut()).enumerate() {
            let seed = self.group_seed(group, scope);
            // A draft cache page's key covers the token after it.
            let key = shape.key_tokens();
            let complete = (tokens.len() + shape.block_size).saturating_sub(key) / shape.block_size;
            let mut parent = chain.last().copied().unwrap_or(seed);
            for page in chain.len()..complete {
                let start = page * shape.block_size;
                let span = tokens.span(start..start + key);
                parent = block_hash(parent, group as u32, TEXT_MODALITY_TAG, &span);
                chain.push(parent);
            }
        }
    }

    /// Finds the longest prompt prefix every group can reuse from the
    /// device cache.
    ///
    /// The prefix leaves at least one prompt token to compute, so prefill
    /// still produces the first sampled token's logits. Complete pages first
    /// select the deepest boundary every group's manager accepts with the pages the pool holds
    /// ([`manager::deepest_boundary`]): a full-attention group needs every
    /// page before it, a sliding-window group only the pages intersecting
    /// its window before it.
    fn select_prefix(
        &self,
        pool: &BlockPool,
        prompt: &[u32],
        hashes: &[Vec<u64>],
        source: &WorkerEndpoint,
        scope: PrefixScope,
        public: bool,
    ) -> PrefixSelection {
        let resume = scope.resume;
        let step = self.max_block_size();
        let limit = prompt.len().saturating_sub(1) / step * step;

        // One lookup per complete prompt page and group; each hit records
        // the page's first unit and whether it is unreferenced.
        let hits = self
            .groups
            .iter()
            .enumerate()
            .map(|(group, shape)| {
                let pages =
                    (prompt.len().saturating_sub(1) / shape.block_size).min(hashes[group].len());
                (0..pages)
                    .map(|page| {
                        let start = page * shape.block_size;
                        let span = &prompt[start..start + shape.key_tokens()];
                        pool.lookup(hashes[group][page], group as u32, span, source, resume)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // Misses before each page, so a window check is one subtraction. The
        // draft cache bounds no reuse: the boundary is the other groups'.
        let required: Vec<usize> = (0..self.groups.len())
            .filter(|&group| !self.groups[group].draft)
            .collect();
        let misses = required
            .iter()
            .map(|&group| manager::misses(hits[group].iter().map(Option::is_some)))
            .collect::<Vec<_>>();
        let managers: Vec<_> = required
            .iter()
            .map(|&group| self.groups[group].manager())
            .collect();
        let tokens = manager::deepest_boundary(&managers, step, limit, &misses);

        // The accepted boundary leaves no miss in any needed range; a range
        // that still holds one selects no prefix rather than a table with a
        // hole. The draft cache takes its leading pages the pool holds
        // within the prefix.
        let pages = self
            .groups
            .iter()
            .enumerate()
            .map(|(group, shape)| {
                if shape.draft {
                    let held = hits[group]
                        .iter()
                        .take(tokens / shape.block_size)
                        .map_while(|hit| *hit)
                        .collect::<Vec<_>>();
                    return Some((0, held));
                }
                let needed = shape.manager().needed_pages(tokens);
                let (start, end) = (needed.start, needed.end);
                let selected = hits[group][start..end]
                    .iter()
                    .copied()
                    .collect::<Option<Vec<_>>>()?;
                Some((start, selected))
            })
            .collect::<Option<Vec<_>>>();
        let mut private = pages.map_or_else(PrefixSelection::default, |pages| PrefixSelection {
            tokens,
            resume,
            pages,
            tails: vec![None; self.groups.len()],
        });
        // Complete pages remain shared. Beyond their common boundary, all
        // required groups must supply the same token extent; recurrent state
        // still ends only at an actual checkpoint boundary.
        let upper = prompt.len().saturating_sub(1).min(tokens + step - 1);
        for end in (tokens + 1..=upper).rev() {
            let mut tails = vec![None; self.groups.len()];
            let valid = required.iter().enumerate().all(|(index, &group)| {
                let shape = self.groups[group];
                let full = end / shape.block_size;
                let rest = end % shape.block_size;
                if shape.state && rest != 0 {
                    return false;
                }
                let start = shape.first_needed_page(end).min(full);
                let misses = &misses[index];
                if full >= misses.len() || misses[full] != misses[start] {
                    return false;
                }
                if rest == 0 {
                    return true;
                }
                let parent = full
                    .checked_sub(1)
                    .map_or_else(|| self.group_seed(group, scope), |page| hashes[group][page]);
                let span = &prompt[full * shape.block_size..end];
                let hash = block_hash(parent, group as u32, TEXT_MODALITY_TAG, span);
                let Some((head, cached)) = pool.lookup(hash, group as u32, span, source, resume)
                else {
                    return false;
                };
                tails[group] = Some(TailPage {
                    head,
                    cached,
                    hash,
                    tokens: rest,
                });
                true
            });
            if !valid {
                continue;
            }
            private.pages = self
                .groups
                .iter()
                .enumerate()
                .map(|(group, shape)| {
                    let full = end / shape.block_size;
                    if shape.draft {
                        return (
                            0,
                            hits[group]
                                .iter()
                                .take(full)
                                .map_while(|hit| *hit)
                                .collect(),
                        );
                    }
                    let start = shape.first_needed_page(end).min(full);
                    (
                        start,
                        hits[group][start..full].iter().flatten().copied().collect(),
                    )
                })
                .collect();
            private.tokens = end;
            private.tails = tails;
            break;
        }
        // A continuation keeps its origin through mapped worlds as well.
        // Public pages remain eligible under the ordinary read policy;
        // acquisition uses whichever complete prefix is longer.
        if resume.is_some() && public {
            let shared = self.select_prefix(
                pool,
                prompt,
                hashes,
                source,
                PrefixScope {
                    resume: None,
                    ..scope
                },
                false,
            );
            if shared.tokens > private.tokens {
                return shared;
            }
        }
        private
    }

    /// Measures the longest reusable prompt prefix without acquiring page
    /// references.
    ///
    /// Admission uses the result to estimate the units a request needs
    /// before committing to it. A page found here can be evicted before
    /// [`PrefixCacheCoordinator::acquire_prefix`] runs, so the hit is an
    /// estimate. Returns an empty hit when prefix caching is disabled,
    /// `cache_read` is false, or the request carries images.
    pub(crate) fn probe_prefix(
        &self,
        pool: &BlockPool,
        prompt: &[u32],
        cache_read: bool,
        has_images: bool,
        scope: PrefixScope,
        source: &WorkerEndpoint,
    ) -> PrefixHit {
        // Public keys cover tokens only. A private continuation reuses its
        // own held state even when shared caching is disabled or the prompt
        // contains images; its source request handle scopes every page.
        if scope.resume.is_none() && (!self.prefix_enabled || !cache_read || has_images) {
            return PrefixHit::default();
        }
        let hashes = self.prefix_hashes(prompt, scope);
        let selection = self.select_prefix(
            pool,
            prompt,
            &hashes,
            source,
            scope,
            self.prefix_enabled && cache_read && !has_images,
        );
        let cached_free_units = selection
            .pages
            .iter()
            .zip(&self.groups)
            .map(|((_, pages), shape)| {
                pages.iter().filter(|(_, cached)| *cached).count() * shape.units_per_page
            })
            .sum::<usize>()
            + selection
                .tails
                .iter()
                .zip(&self.groups)
                .filter(|(tail, _)| tail.is_some_and(|tail| tail.cached))
                .map(|(_, shape)| shape.units_per_page)
                .sum::<usize>();
        PrefixHit {
            cached_tokens: selection.tokens,
            cached_free_units,
        }
    }

    /// Acquires the longest reusable prompt prefix into empty tables.
    ///
    /// Full-attention tables take every page before the prefix length;
    /// sliding-window tables take only their window's pages and start at the
    /// first of them. Acquisition is all or nothing: when a page no longer
    /// matches its identity, every reference taken is dropped and no prefix
    /// is reused. The returned hashes are computed even when `cache_read` is
    /// false, so the prompt can still be published as prefill completes.
    /// Returns `None` when the tables do not match the groups or are not
    /// empty.
    #[expect(
        clippy::too_many_arguments,
        reason = "one lookup's prompt, cache policy and source worker"
    )]
    pub(crate) fn acquire_prefix(
        &self,
        pool: &BlockPool,
        allocation: &mut CacheAllocation,
        prompt: &[u32],
        cache_read: bool,
        has_images: bool,
        scope: PrefixScope,
        source: &WorkerEndpoint,
    ) -> Option<PrefixLookup> {
        if !self.matches(allocation)
            || allocation
                .tables
                .iter()
                .any(|table| !table.pages.is_empty() || table.start_page != 0)
        {
            return None;
        }

        // Empty hashes also prevent prefix-cache insertion for image requests; see
        // `probe_prefix` for why images are excluded.
        if scope.resume.is_none() && (!self.prefix_enabled || has_images) {
            return Some(PrefixLookup::default());
        }
        let hashes = self.prefix_hashes(prompt, scope);
        let mut result = PrefixLookup {
            page_hashes: hashes,
            cached_tokens: 0,
            draft_tokens: 0,
        };
        if !cache_read && scope.resume.is_none() {
            return Some(result);
        }

        let selection = self.select_prefix(
            pool,
            prompt,
            &result.page_hashes,
            source,
            scope,
            self.prefix_enabled && cache_read && !has_images,
        );
        let acquire = |group: usize, start: usize, pages: &[(UnitId, bool)]| {
            let shape = self.groups[group];
            pages
                .iter()
                .enumerate()
                .map(|(offset, (head, _))| {
                    let page = start + offset;
                    let first = page * shape.block_size;
                    let span = &prompt[first..first + shape.key_tokens()];
                    // `acquire` rechecks each page's identity.
                    pool.acquire(
                        *head,
                        result.page_hashes[group][page],
                        group as u32,
                        span,
                        source,
                        selection.resume,
                    )
                })
                .collect::<Option<Vec<_>>>()
        };
        let acquired = selection
            .pages
            .iter()
            .enumerate()
            .map(|(group, (start, pages))| {
                if self.groups[group].draft {
                    // A draft page that changed identity leaves the request
                    // without its draft cache alone.
                    return Some((*start, acquire(group, *start, pages).unwrap_or_default()));
                }
                Some((*start, acquire(group, *start, pages)?))
            })
            .collect::<Option<Vec<_>>>();
        // A page that changed identity since selection leaves the request
        // without a reused prefix; the references taken so far drop here,
        // without the pool's lock held.
        let Some(acquired) = acquired else {
            return Some(result);
        };

        let tails = selection
            .tails
            .iter()
            .enumerate()
            .map(|(group, tail)| {
                let Some(tail) = tail else {
                    return Some(None);
                };
                let shape = self.groups[group];
                let first = selection.tokens / shape.block_size * shape.block_size;
                let source_page = pool.acquire(
                    tail.head,
                    tail.hash,
                    group as u32,
                    &prompt[first..first + tail.tokens],
                    source,
                    selection.resume,
                )?;
                let target = pool.allocate(1, shape.units_per_page)?.pop()?;
                Some(Some((source_page, target)))
            })
            .collect::<Option<Vec<_>>>();
        let Some(tails) = tails else {
            return Some(result);
        };

        let source = Arc::new(source.clone());
        for (group, ((table, (start, references)), tail)) in allocation
            .tables
            .iter_mut()
            .zip(acquired)
            .zip(tails)
            .enumerate()
        {
            if let Some(resume) = scope.resume {
                for (offset, reference) in references.iter().enumerate() {
                    let page = start + offset;
                    let first = page * table.shape.block_size;
                    pool.publish(
                        reference,
                        group as u32,
                        result.page_hashes[group][page],
                        &prompt[first..first + table.shape.key_tokens()],
                        &source,
                        Some(resume),
                    );
                }
            }
            if table.shape.draft {
                result.draft_tokens = references.len() * table.shape.block_size;
            }
            table.start_page = start;
            table
                .pages
                .extend(references.into_iter().map(|page| TablePage {
                    page: Some(page),
                    serial: None,
                }));
            if let Some((source, target)) = tail {
                let page = table.end_page();
                let serial = allocation.allocated_units;
                allocation.allocated_units += target.units().len() as u64;
                table.pages.push_back(TablePage {
                    page: Some(target),
                    serial: Some(serial),
                });
                allocation.copies.push((group, page, source));
            }
        }
        result.cached_tokens = selection.tokens;
        Some(result)
    }

    /// Publish immutable partial pages after the request's writes have
    /// drained. Ordinary preemption and world promotion use the same
    /// continuation lookup; consumers copy these pages before appending.
    #[expect(
        clippy::too_many_arguments,
        reason = "one drained request's cache publication"
    )]
    pub(crate) fn publish_tails(
        &self,
        pool: &BlockPool,
        allocation: &CacheAllocation,
        tokens: CachedTokens<'_>,
        computed: usize,
        scope: PrefixScope,
        public: bool,
        source: &Arc<WorkerEndpoint>,
    ) {
        if (!public && scope.resume.is_none()) || !self.matches(allocation) {
            return;
        }
        let computed = computed.min(tokens.len());
        let mut hashes = vec![Vec::new(); self.groups.len()];
        self.extend_hashes(&mut hashes, tokens, scope);
        for (group, table) in allocation.tables.iter().enumerate() {
            let seed = self.group_seed(group, scope);
            let shape = table.shape;
            // Request-slot checkpoints and speculative draft state have
            // their own completed-boundary rules, unchanged here.
            if shape.state || shape.draft || computed.is_multiple_of(shape.block_size) {
                continue;
            }
            let page = computed / shape.block_size;
            let Some(reference) = table.page(page) else {
                continue;
            };
            let parent = page
                .checked_sub(1)
                .map_or(seed, |index| hashes[group][index]);
            let span = tokens.span(page * shape.block_size..computed);
            let hash = block_hash(parent, group as u32, TEXT_MODALITY_TAG, &span);
            if public && self.prefix_enabled {
                pool.publish(reference, group as u32, hash, &span, source, None);
            }
            if let Some(resume) = scope.resume {
                pool.publish(reference, group as u32, hash, &span, source, Some(resume));
            }
        }
    }

    /// Publishes the complete computed pages of every group into the prefix
    /// cache: the prompt's, and those of the tokens the request generated
    /// after it, whose pages a migration of the request finds as its prompt's.
    ///
    /// `hashes` chains every complete page of `tokens` (the prompt's from
    /// [`PrefixCacheCoordinator::acquire_prefix`], the generated tokens'
    /// through [`PrefixCacheCoordinator::extend_hashes`]), and `published`
    /// records per group how many leading pages were already published; both
    /// advance together. A page is published once the worker has computed
    /// all of its tokens (`computed_tokens`) and while the table still holds
    /// it, so a caller that publishes after every forward, before releasing
    /// any window page, publishes every page of a sliding-window group before
    /// it leaves the window. The draft cache's pages publish once its
    /// entries are valid through their last token (`draft_tokens`): each
    /// entry reads the token after its own. Returns `true` without
    /// publishing when neither public reuse nor a private continuation is
    /// eligible, or `hashes` is empty, and
    /// `false` when the tables or hashes do not match the groups.
    #[expect(
        clippy::too_many_arguments,
        reason = "one request's computed blocks, publication policy and source worker"
    )]
    pub(crate) fn publish_computed(
        &self,
        pool: &BlockPool,
        allocation: &CacheAllocation,
        tokens: CachedTokens<'_>,
        hashes: &[Vec<u64>],
        computed_tokens: usize,
        draft_tokens: usize,
        published: &mut Vec<usize>,
        cache_write: bool,
        source: &Arc<WorkerEndpoint>,
        resume: Option<proto::request::CacheResume>,
    ) -> bool {
        let public = self.prefix_enabled && cache_write;
        if (!public && resume.is_none()) || hashes.is_empty() {
            return true;
        }
        if !self.matches(allocation) || hashes.len() != self.groups.len() {
            return false;
        }
        published.resize(self.groups.len(), 0);

        let computed = computed_tokens.min(tokens.len());
        for (group, (table, group_hashes)) in allocation.tables.iter().zip(hashes).enumerate() {
            let block_size = table.shape.block_size;
            let valid = if table.shape.draft {
                draft_tokens.min(computed)
            } else {
                computed
            };
            let complete = (valid / block_size).min(group_hashes.len());
            for (page, hash) in group_hashes
                .iter()
                .enumerate()
                .take(complete)
                .skip(published[group])
            {
                // Retired pages cannot be inserted into the prefix cache, nor
                // can a draft cache page without units.
                let Some(reference) = table.page(page) else {
                    continue;
                };
                let first = page * block_size;
                let span = tokens.span(first..first + table.shape.key_tokens());
                if public {
                    pool.publish(reference, group as u32, *hash, &span, source, None);
                }
                if let Some(resume) = resume {
                    pool.publish(reference, group as u32, *hash, &span, source, Some(resume));
                }
            }
            published[group] = published[group].max(complete);
        }
        true
    }
}

/// The tokens whose cache a request's tables hold, in order: its prompt,
/// then the tokens it generated since it was admitted, whose cache each
/// decode writes as it reads them. A migrated request's prompt already ends
/// with the tokens it generated before.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CachedTokens<'a> {
    prompt: &'a [u32],
    generated: &'a [u32],
}

impl<'a> CachedTokens<'a> {
    /// A prompt alone.
    pub(crate) fn prompt(prompt: &'a [u32]) -> Self {
        Self {
            prompt,
            generated: &[],
        }
    }

    /// A prompt followed by the tokens generated after it.
    pub(crate) fn new(prompt: &'a [u32], generated: &'a [u32]) -> Self {
        Self { prompt, generated }
    }

    /// The tokens held.
    pub(crate) fn len(&self) -> usize {
        self.prompt.len() + self.generated.len()
    }

    /// The prompt's tokens, which lead.
    pub(crate) fn prompt_len(&self) -> usize {
        self.prompt.len()
    }

    /// The tokens of `range`, which must lie within [`Self::len`]; borrowed
    /// unless the range spans the prompt's end.
    pub(crate) fn span(&self, range: std::ops::Range<usize>) -> std::borrow::Cow<'a, [u32]> {
        let split = self.prompt.len();
        if range.end <= split {
            std::borrow::Cow::Borrowed(&self.prompt[range])
        } else if range.start >= split {
            std::borrow::Cow::Borrowed(&self.generated[range.start - split..range.end - split])
        } else {
            let mut tokens = self.prompt[range.start..].to_vec();
            tokens.extend_from_slice(&self.generated[..range.end - split]);
            std::borrow::Cow::Owned(tokens)
        }
    }

    /// The tokens of `range`, or `None` when it reaches past [`Self::len`].
    pub(crate) fn get(&self, range: std::ops::Range<usize>) -> Option<std::borrow::Cow<'a, [u32]>> {
        (range.start <= range.end && range.end <= self.len()).then(|| self.span(range))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint() -> Arc<WorkerEndpoint> {
        Arc::new(ipc::WorkerInfo::default().endpoint)
    }

    #[test]
    fn private_recovery_pages_are_held_and_reused_only_by_their_continuation() {
        let pool = BlockPool::new(8, 0);
        let coordinator = PrefixCacheCoordinator::new(vec![full(4, 1)]);
        let source = endpoint();
        let tokens = vec![1, 2, 3, 4, 5];
        let resume = proto::request::CacheResume {
            world: 3,
            request_id: 7,
        };
        let scope = PrefixScope {
            resume: Some(resume),
            ..Default::default()
        };
        let hashes = coordinator.prefix_hashes(&tokens, scope);
        let page = pool.allocate(1, 1).unwrap().remove(0);
        pool.activate([&page]);
        let lease = page.clone();
        drop(page);
        assert_eq!(pool.free_units(), 6);
        pool.publish(&lease, 0, hashes[0][0], &tokens[..4], &source, Some(resume));
        assert_eq!(
            coordinator
                .probe_prefix(&pool, &tokens, true, false, PrefixScope::default(), &source)
                .cached_tokens,
            0
        );
        let other = PrefixScope {
            resume: Some(proto::request::CacheResume {
                world: 3,
                request_id: 8,
            }),
            ..scope
        };
        assert_eq!(
            coordinator
                .probe_prefix(&pool, &tokens, true, false, other, &source)
                .cached_tokens,
            0
        );
        let mut allocation = coordinator.empty();
        assert_eq!(
            coordinator
                .probe_prefix(&pool, &tokens, false, true, scope, &source)
                .cached_tokens,
            4
        );
        let hit = coordinator
            .acquire_prefix(&pool, &mut allocation, &tokens, false, true, scope, &source)
            .unwrap();
        assert_eq!(hit.cached_tokens, 4);
        drop(lease);
        assert_eq!(pool.free_units(), 6);
        drop(allocation);
        assert_eq!(pool.free_units(), 7);
        let successor = BlockPool::new(16, 0);
        successor.inherit(8..16);
        successor.import(&pool.export(), 8, &source);
        assert_eq!(
            coordinator
                .probe_prefix(&successor, &tokens, false, true, scope, &source)
                .cached_tokens,
            4,
        );
        assert!(pool.acquire_idle_from(1).is_empty());
    }

    #[test]
    fn a_continuation_reuses_public_pages_only_under_its_read_policy() {
        let pool = BlockPool::new(8, 0);
        let coordinator = PrefixCacheCoordinator::new(vec![full(4, 1)]);
        let source = endpoint();
        let tokens = vec![1, 2, 3, 4, 5];
        let _published = publish(&coordinator, &pool, &tokens, &source);
        let scope = PrefixScope {
            resume: Some(proto::request::CacheResume {
                world: 3,
                request_id: 7,
            }),
            ..Default::default()
        };
        for (request, (read, images, expected)) in
            [(true, false, 4), (false, false, 0), (true, true, 0)]
                .into_iter()
                .enumerate()
        {
            // Each policy is a fresh request, before it acquires a private
            // alias to any public prefix it is allowed to read.
            let scope = PrefixScope {
                resume: scope.resume.map(|resume| proto::request::CacheResume {
                    request_id: resume.request_id + request as u64,
                    ..resume
                }),
                ..scope
            };
            assert_eq!(
                coordinator
                    .probe_prefix(&pool, &tokens, read, images, scope, &source)
                    .cached_tokens,
                expected,
            );
            let mut allocation = coordinator.empty();
            assert_eq!(
                coordinator
                    .acquire_prefix(
                        &pool,
                        &mut allocation,
                        &tokens,
                        read,
                        images,
                        scope,
                        &source
                    )
                    .unwrap()
                    .cached_tokens,
                expected,
            );
        }
    }

    #[test]
    fn mapped_pages_preserve_public_and_private_reuse_on_the_same_units() {
        let pool = BlockPool::new(16, 0);
        let coordinator = PrefixCacheCoordinator::new(vec![full(4, 1)]);
        let source = endpoint();
        let tokens = vec![1, 2, 3, 4, 5];
        drop(publish(&coordinator, &pool, &tokens, &source));
        let resume = proto::request::CacheResume {
            world: 3,
            request_id: 7,
        };
        let scope = PrefixScope {
            resume: Some(resume),
            ..Default::default()
        };
        let mut allocation = coordinator.empty();
        assert_eq!(
            coordinator
                .acquire_prefix(&pool, &mut allocation, &tokens, true, false, scope, &source,)
                .unwrap()
                .cached_tokens,
            4
        );
        drop(allocation);

        let successor = BlockPool::new(32, 0);
        successor.inherit(16..32);
        successor.import(&pool.export(), 16, &source);
        successor.take_inherited();
        let free = successor.free_units();
        let mut public = coordinator.empty();
        let mut private = coordinator.empty();
        for (allocation, scope, read) in [
            (&mut public, PrefixScope::default(), true),
            (&mut private, scope, false),
        ] {
            assert_eq!(
                coordinator
                    .acquire_prefix(&successor, allocation, &tokens, read, false, scope, &source,)
                    .unwrap()
                    .cached_tokens,
                4
            );
        }
        assert_eq!(successor.free_units(), free - 1);
        drop((public, private));
        successor.forget_resume(resume);
        assert_eq!(
            coordinator
                .probe_prefix(&successor, &tokens, false, false, scope, &source,)
                .cached_tokens,
            0
        );
        assert_eq!(
            coordinator
                .probe_prefix(
                    &successor,
                    &tokens,
                    true,
                    false,
                    PrefixScope::default(),
                    &source,
                )
                .cached_tokens,
            4
        );
    }

    #[test]
    fn private_publication_keeps_the_latest_completed_state_checkpoint() {
        let pool = BlockPool::new(64, 0);
        let coordinator = PrefixCacheCoordinator::new(vec![full(4, 1), state(4, 2)]);
        let source = endpoint();
        let tokens: Vec<_> = (1..=25).collect();
        let resume = proto::request::CacheResume {
            world: 3,
            request_id: 7,
        };
        let scope = PrefixScope {
            resume: Some(resume),
            ..Default::default()
        };
        let mut allocation = coordinator.empty();
        let lookup = coordinator
            .acquire_prefix(
                &pool,
                &mut allocation,
                &tokens,
                false,
                false,
                scope,
                &source,
            )
            .unwrap();
        let mut published = Vec::new();
        coordinator
            .ensure_capacity(&pool, &mut allocation, 0, 10, Checkpoint::Boundary)
            .unwrap();
        for (start, end) in [(0, 10), (10, 14), (14, 18)] {
            if start > 0 {
                coordinator
                    .ensure_capacity(&pool, &mut allocation, start, end, Checkpoint::Interval(8))
                    .unwrap();
            }
            assert!(coordinator.publish_computed(
                &pool,
                &allocation,
                CachedTokens::prompt(&tokens),
                &lookup.page_hashes,
                end,
                0,
                &mut published,
                false,
                &source,
                Some(resume),
            ));
            assert!(coordinator.release_window(&mut allocation, end, None));
            let checkpoint = if end < 16 { 8 } else { 16 };
            assert!(allocation.tables[1].holds(checkpoint - 4, checkpoint));
            assert_eq!(
                coordinator
                    .probe_prefix(&pool, &tokens, false, false, scope, &source,)
                    .cached_tokens,
                checkpoint
            );
            assert_eq!(
                coordinator
                    .probe_prefix(&pool, &tokens, true, false, PrefixScope::default(), &source,)
                    .cached_tokens,
                0
            );
        }
        // A future checkpoint's reserved units are not yet a completed state.
        coordinator
            .ensure_capacity(&pool, &mut allocation, 18, 24, Checkpoint::Interval(8))
            .unwrap();
        assert!(coordinator.release_window(&mut allocation, 22, None));
        assert!(allocation.tables[1].holds(12, 16));
        assert_eq!(
            coordinator
                .probe_prefix(&pool, &tokens, false, false, scope, &source,)
                .cached_tokens,
            16
        );
    }

    fn full(block_size: usize, units_per_page: usize) -> GroupShape {
        GroupShape::attention(block_size, units_per_page, None)
    }

    fn sliding(block_size: usize, units_per_page: usize, window: usize) -> GroupShape {
        GroupShape::attention(block_size, units_per_page, Some(window))
    }

    fn state(block_size: usize, units_per_page: usize) -> GroupShape {
        GroupShape::from_group(&ipc::CacheGroup {
            kind: ipc::CacheGroupKind::State,
            block_size: block_size as u32,
            units_per_page: units_per_page as u32,
            layer_ids: vec![0],
            num_kv_heads: 1,
            total_kv_heads: 1,
            kv_head_offset: 0,
            head_dim: 1,
        })
    }

    #[test]
    fn a_preempted_partial_page_resumes_in_writable_units() {
        for (shape, inherited) in [
            (full(64, 1), false),
            (full(64, 1), true),
            (sliding(64, 1, 128), false),
            (sliding(64, 1, 128), true),
        ] {
            let pool = BlockPool::new(128, 0);
            let coordinator = PrefixCacheCoordinator::new(vec![shape]);
            let source = endpoint();
            let tokens: Vec<u32> = (0..145).collect();
            let scope = PrefixScope {
                resume: Some(proto::request::CacheResume {
                    world: 1,
                    request_id: 9,
                }),
                ..Default::default()
            };
            let allocation = publish(&coordinator, &pool, &tokens, &source);
            let hashes = coordinator.prefix_hashes(&tokens, scope);
            assert!(coordinator.publish_computed(
                &pool,
                &allocation,
                CachedTokens::prompt(&tokens),
                &hashes,
                tokens.len(),
                0,
                &mut Vec::new(),
                false,
                &source,
                scope.resume,
            ));
            coordinator.publish_tails(
                &pool,
                &allocation,
                CachedTokens::prompt(&tokens),
                tokens.len(),
                scope,
                false,
                &source,
            );
            let original = allocation.tables[0].unit_ids();
            drop(allocation);
            let (pool, offset) = if inherited {
                let successor = BlockPool::new(256, 0);
                successor.inherit(128..256);
                successor.import(&pool.export(), 128, &source);
                (successor, 128)
            } else {
                (pool, 0)
            };
            let mut prompt = tokens;
            prompt.push(145);
            let free = pool.free_units();
            let mut resumed = coordinator.empty();
            let hit = coordinator
                .acquire_prefix(&pool, &mut resumed, &prompt, false, false, scope, &source)
                .unwrap();
            assert_eq!(hit.cached_tokens, 145);
            let units = resumed.tables[0].unit_ids();
            assert_eq!(
                units[..2],
                original[..2]
                    .iter()
                    .map(|unit| UnitId(unit.0 + offset))
                    .collect::<Vec<_>>()
            );
            assert_ne!(units[2], UnitId(original[2].0 + offset));
            if inherited {
                assert!(units[2].0 < 128);
                assert_eq!(pool.free_units(), free - 1);
            }
            // The numerical consumer receives the immutable source and its
            // private destination together with the newly assigned units.
            assert_eq!(
                resumed.copies(0, 0),
                vec![(UnitId(original[2].0 + offset), units[2])]
            );
            let foreign = PrefixScope {
                resume: Some(proto::request::CacheResume {
                    world: 1,
                    request_id: 10,
                }),
                ..scope
            };
            assert_eq!(
                coordinator
                    .probe_prefix(&pool, &prompt, false, false, foreign, &source)
                    .cached_tokens,
                0
            );
            if shape.window.is_some() {
                // Host admission can advance beyond the local partial
                // page. No copy may name a destination it released.
                assert!(coordinator.release_window(&mut resumed, 512, None));
                assert!(resumed.copies(0, 0).is_empty());
            }
            resumed.complete_copies();
            assert!(resumed.copies(0, 0).is_empty());
        }
    }

    /// Prefills `prompt` into fresh tables and publishes every complete page.
    fn publish(
        coordinator: &PrefixCacheCoordinator,
        pool: &BlockPool,
        prompt: &[u32],
        source: &Arc<WorkerEndpoint>,
    ) -> CacheAllocation {
        let mut allocation = coordinator.empty();
        let lookup = coordinator
            .acquire_prefix(
                pool,
                &mut allocation,
                prompt,
                false,
                false,
                PrefixScope::default(),
                source,
            )
            .unwrap();
        coordinator
            .ensure_capacity(pool, &mut allocation, 0, prompt.len(), Checkpoint::Boundary)
            .unwrap();
        let mut published = Vec::new();
        assert!(coordinator.publish_computed(
            pool,
            &allocation,
            CachedTokens::prompt(prompt),
            &lookup.page_hashes,
            prompt.len(),
            0,
            &mut published,
            true,
            source,
            None,
        ));
        allocation
    }

    /// Publishes only the chosen logical pages of one group of `prompt`, each
    /// on its own freshly allocated units, and returns their references.
    fn publish_pages(
        coordinator: &PrefixCacheCoordinator,
        pool: &BlockPool,
        prompt: &[u32],
        group: usize,
        pages: &[usize],
        source: &Arc<WorkerEndpoint>,
    ) -> Vec<PageRef> {
        let shape = coordinator.groups()[group];
        let hashes = coordinator.prefix_hashes(prompt, PrefixScope::default());
        pages
            .iter()
            .map(|page| {
                let reference = pool.allocate(1, shape.units_per_page).unwrap().remove(0);
                let tokens = &prompt[page * shape.block_size..(page + 1) * shape.block_size];
                pool.publish(
                    &reference,
                    group as u32,
                    hashes[group][*page],
                    tokens,
                    source,
                    None,
                );
                reference
            })
            .collect()
    }

    /// A chain extended over the tokens a request generated equals the
    /// chain over its migration's prompt, which continues with them, so the
    /// migration finds every page the request generated.
    #[test]
    fn a_chain_extended_over_generated_tokens_is_its_migrations_chain() {
        let coordinator = PrefixCacheCoordinator::new(vec![full(4, 1), sliding(2, 1, 8)]);
        let prompt: Vec<u32> = (10..19).collect();
        let generated: Vec<u32> = (100..107).collect();
        let migrated: Vec<u32> = prompt.iter().chain(&generated).copied().collect();

        let mut hashes = coordinator.prefix_hashes(
            &prompt,
            PrefixScope {
                isolation_key: Some(7),
                state_seeds: &[],
                resume: None,
            },
        );
        coordinator.extend_hashes(
            &mut hashes,
            CachedTokens::new(&prompt, &generated),
            PrefixScope {
                isolation_key: Some(7),
                state_seeds: &[],
                resume: None,
            },
        );
        assert_eq!(
            hashes,
            coordinator.prefix_hashes(
                &migrated,
                PrefixScope {
                    isolation_key: Some(7),
                    state_seeds: &[],
                    resume: None,
                }
            )
        );
        assert_eq!(hashes[0].len(), 4);
        assert_eq!(hashes[1].len(), 8);

        // Another isolation key chains other pages.
        assert_ne!(
            hashes,
            coordinator.prefix_hashes(
                &migrated,
                PrefixScope {
                    isolation_key: Some(8),
                    state_seeds: &[],
                    resume: None,
                }
            )
        );
    }

    /// A page that spans the prompt's end reads both halves, in order.
    #[test]
    fn cached_tokens_read_pages_across_the_prompts_end() {
        let prompt = [1, 2, 3];
        let generated = [4, 5, 6, 7];
        let tokens = CachedTokens::new(&prompt, &generated);
        assert_eq!(tokens.len(), 7);
        assert_eq!(tokens.prompt_len(), 3);
        assert_eq!(&*tokens.span(0..2), &[1, 2]);
        assert_eq!(&*tokens.span(2..4), &[3, 4]);
        assert_eq!(&*tokens.span(4..6), &[5, 6]);
        assert!(tokens.get(6..8).is_none());
        assert_eq!(tokens.get(6..7).as_deref(), Some(&[7][..]));
    }

    /// Pages of generated tokens publish once computed, and a lookup of the
    /// migration's prompt reuses them.
    #[test]
    fn published_generated_pages_serve_the_migrations_prompt() {
        let pool = BlockPool::new(16, 0);
        let coordinator = PrefixCacheCoordinator::new(vec![full(4, 1)]);
        let source = endpoint();
        let prompt: Vec<u32> = (10..18).collect();
        let generated: Vec<u32> = (100..110).collect();
        let tokens = CachedTokens::new(&prompt, &generated);

        let mut allocation = coordinator.empty();
        let mut lookup = coordinator
            .acquire_prefix(
                &pool,
                &mut allocation,
                &prompt,
                false,
                false,
                PrefixScope::default(),
                &source,
            )
            .unwrap();
        coordinator
            .ensure_capacity(
                &pool,
                &mut allocation,
                0,
                tokens.len(),
                Checkpoint::Boundary,
            )
            .unwrap();
        coordinator.extend_hashes(&mut lookup.page_hashes, tokens, PrefixScope::default());
        let mut published = Vec::new();
        assert!(coordinator.publish_computed(
            &pool,
            &allocation,
            tokens,
            &lookup.page_hashes,
            tokens.len(),
            0,
            &mut published,
            true,
            &source,
            None,
        ));
        assert_eq!(published, vec![4]);

        let migrated: Vec<u32> = prompt.iter().chain(&generated).copied().collect();
        let hit = coordinator.probe_prefix(
            &pool,
            &migrated,
            true,
            false,
            PrefixScope::default(),
            &source,
        );
        assert_eq!(hit.cached_tokens, 16);
    }

    #[test]
    fn pages_of_several_units_draw_from_one_pool_and_return_together() {
        // Nine allocatable units: a sliding page takes three, a full page one.
        let pool = BlockPool::new(10, 0);
        let coordinator = PrefixCacheCoordinator::new(vec![full(8, 1), sliding(4, 3, 100)]);
        let mut allocation = coordinator.empty();

        // Eight tokens need one full page and two sliding pages: 1 + 6 units.
        assert_eq!(
            coordinator.units_needed(&pool, &allocation, 0, 8, Checkpoint::Boundary),
            Some(7)
        );
        coordinator
            .ensure_capacity(&pool, &mut allocation, 0, 8, Checkpoint::Boundary)
            .unwrap();
        assert_eq!(pool.free_units(), 2);
        assert_eq!(allocation.tables[1].unit_ids().len(), 6);
        assert!(
            allocation
                .tables
                .iter()
                .flat_map(BlockTable::unit_ids)
                .all(|unit| unit.0 != 0)
        );

        // Growing to twelve tokens needs a full page and a sliding page,
        // four units, but only two are free: nothing is allocated.
        assert!(
            coordinator
                .ensure_capacity(&pool, &mut allocation, 8, 12, Checkpoint::Boundary)
                .is_none()
        );
        assert_eq!(pool.free_units(), 2);
        assert_eq!(allocation.tables[0].end_page(), 1);
        assert_eq!(allocation.tables[1].end_page(), 2);

        drop(allocation);
        assert_eq!(pool.free_units(), 9);
    }

    #[test]
    fn the_warmup_scratch_units_are_never_allocated() {
        // Units 1..=3 are scratch: six of ten units are allocatable.
        let pool = BlockPool::new(10, 3);
        assert_eq!((pool.usable_units(), pool.free_units()), (6, 6));
        let coordinator = PrefixCacheCoordinator::new(vec![full(1, 1)]);
        let mut allocation = coordinator.empty();
        coordinator
            .ensure_capacity(&pool, &mut allocation, 0, 6, Checkpoint::Boundary)
            .unwrap();
        let units: Vec<u32> = allocation.tables[0]
            .unit_ids()
            .iter()
            .map(|unit| unit.0)
            .collect();
        assert_eq!(units.len(), 6);
        assert!(units.iter().all(|unit| *unit > 3));
        assert!(
            coordinator
                .ensure_capacity(&pool, &mut allocation, 6, 7, Checkpoint::Boundary)
                .is_none()
        );
    }

    #[test]
    fn withdrawn_units_leave_allocation_until_restored() {
        // Fifteen allocatable units; a request holds units 1..=12.
        let pool = BlockPool::new(16, 0);
        let coordinator = PrefixCacheCoordinator::new(vec![full(1, 1)]);
        let mut allocation = coordinator.empty();
        coordinator
            .ensure_capacity(&pool, &mut allocation, 0, 12, Checkpoint::Boundary)
            .unwrap();
        let held: Vec<u32> = allocation.tables[0]
            .unit_ids()
            .iter()
            .map(|unit| unit.0)
            .collect();
        assert_eq!(held, (1..=12).collect::<Vec<_>>());

        // Withdrawing from unit 8 takes the free units 13..=15 out of the
        // queue; the request still holds 8..=12.
        assert_eq!(pool.withdraw(8), 5);
        assert_eq!(pool.withdrawn_held(), 5);
        assert_eq!((pool.usable_units(), pool.capacity_units()), (7, 15));
        assert_eq!(pool.free_units(), 0);

        // Released, the withdrawn units stay out of allocation and the
        // others return to it.
        drop(allocation);
        assert_eq!(pool.withdrawn_held(), 0);
        assert_eq!(pool.free_units(), 7);
        let mut allocation = coordinator.empty();
        coordinator
            .ensure_capacity(&pool, &mut allocation, 0, 7, Checkpoint::Boundary)
            .unwrap();
        assert!(
            allocation.tables[0]
                .unit_ids()
                .iter()
                .all(|unit| unit.0 < 8)
        );
        assert!(
            coordinator
                .ensure_capacity(&pool, &mut allocation, 7, 8, Checkpoint::Boundary)
                .is_none()
        );

        // A restore returns every withdrawn unit to allocation.
        pool.restore();
        assert!(!pool.has_withdrawn());
        assert_eq!((pool.usable_units(), pool.free_units()), (15, 8));
        drop(allocation);
        assert_eq!(pool.free_units(), 15);
    }

    /// The idle cached pages from a unit on are those published pages no
    /// request holds: a page still held, an unpublished free unit and a
    /// withdrawn unit do not count.
    #[test]
    fn cached_pages_from_a_unit_count_the_idle_published_ones() {
        let pool = BlockPool::new(16, 0);
        let coordinator = PrefixCacheCoordinator::new(vec![full(1, 1)]);
        let source = endpoint();
        let prompt: Vec<u32> = (100..112).collect();
        // Twelve pages on units 1..=12, all held by the request.
        let held = publish(&coordinator, &pool, &prompt, &source);
        assert_eq!(pool.cached_from(1), 0);

        // Released, they are cached and idle.
        drop(held);
        assert_eq!(pool.cached_from(1), 12);
        assert_eq!(pool.cached_from(9), 4);
        // A withdrawal from unit 11 evicts the two pages there.
        pool.withdraw(11);
        assert_eq!(pool.cached_from(9), 2);
    }

    #[test]
    fn a_restore_queues_the_withdrawn_units_requests_hold_once_released() {
        let pool = BlockPool::new(8, 0);
        let coordinator = PrefixCacheCoordinator::new(vec![full(1, 1)]);
        let mut allocation = coordinator.empty();
        coordinator
            .ensure_capacity(&pool, &mut allocation, 0, 7, Checkpoint::Boundary)
            .unwrap();
        assert_eq!(pool.withdraw(4), 4);
        pool.restore();
        assert_eq!((pool.usable_units(), pool.free_units()), (7, 0));
        drop(allocation);
        assert_eq!(pool.free_units(), 7);
    }

    #[test]
    fn admission_compares_the_sum_of_group_demand_with_free_units() {
        // Eight tokens need two units in one group and three in the other.
        // With one of five units taken, the four free units cover either
        // group alone, which a per-group minimum would accept, but not both.
        let pool = BlockPool::new(6, 0);
        let coordinator = PrefixCacheCoordinator::new(vec![full(4, 1), full(8, 3)]);
        let mut allocation = coordinator.empty();
        assert_eq!(
            coordinator.units_needed(&pool, &allocation, 0, 8, Checkpoint::Boundary),
            Some(5)
        );
        let blocker = pool.allocate(1, 1).unwrap();
        assert!(
            coordinator
                .ensure_capacity(&pool, &mut allocation, 0, 8, Checkpoint::Boundary)
                .is_none()
        );
        drop(blocker);
        coordinator
            .ensure_capacity(&pool, &mut allocation, 0, 8, Checkpoint::Boundary)
            .unwrap();
        assert_eq!(pool.free_units(), 0);
    }

    #[test]
    fn sliding_pages_retire_only_behind_the_window_and_keep_published_copies() {
        let pool = BlockPool::new(64, 0);
        let source = endpoint();
        let coordinator = PrefixCacheCoordinator::new(vec![full(4, 1), sliding(4, 2, 6)]);
        let prompt = (1..=16).collect::<Vec<u32>>();
        let mut allocation = publish(&coordinator, &pool, &prompt, &source);
        let free = pool.free_units();
        let retired = (0..2)
            .flat_map(|page| allocation.tables[1].page(page).unwrap().units().to_vec())
            .collect::<Vec<_>>();

        // Visible 16 with window 6 keeps tokens [10, 16): pages 2 and 3.
        assert!(coordinator.release_window(&mut allocation, 16, None));
        let table = &allocation.tables[1];
        assert_eq!((table.start_page(), table.end_page()), (2, 4));
        assert_eq!(table.unit_ids().len(), 4);
        assert!(table.page(1).is_none());
        assert_eq!(allocation.tables[0].start_page(), 0);

        // The two retired pages (four units) return as published, still
        // matchable copies.
        assert_eq!(pool.free_units(), free + 4);
        assert!(
            retired
                .iter()
                .all(|unit| pool.unit_state(*unit) == UnitState::Cached)
        );
        let hit = coordinator.probe_prefix(
            &pool,
            &prompt[..13],
            true,
            false,
            PrefixScope::default(),
            &source,
        );
        assert_eq!(hit.cached_tokens, 12);

        // Growth appends after the window and never moves the start back.
        coordinator
            .ensure_capacity(&pool, &mut allocation, 16, 20, Checkpoint::Boundary)
            .unwrap();
        assert_eq!(allocation.tables[1].start_page(), 2);
        assert_eq!(allocation.tables[1].end_page(), 5);
    }

    /// A state table holds units only for the checkpoint each call keeps:
    /// a prompt call's last boundary, and a decode call's boundary at a
    /// multiple of the state interval. Published checkpoints serve a later
    /// prompt's reuse, which ends where one is held.
    #[test]
    fn state_tables_hold_only_checkpoints_and_reuse_ends_at_one() {
        let pool = BlockPool::new(128, 0);
        let source = endpoint();
        let coordinator = PrefixCacheCoordinator::new(vec![full(4, 1), state(4, 3)]);
        let tokens = (1..=24).collect::<Vec<u32>>();
        let mut allocation = coordinator.empty();
        let lookup = coordinator
            .acquire_prefix(
                &pool,
                &mut allocation,
                &tokens,
                false,
                false,
                PrefixScope::default(),
                &source,
            )
            .unwrap();

        // A first chunk of ten tokens covers three pages and checkpoints
        // boundary 8 alone: one state page of three units.
        let free = pool.free_units();
        coordinator
            .ensure_capacity(&pool, &mut allocation, 0, 10, Checkpoint::Boundary)
            .unwrap();
        assert_eq!(free - pool.free_units(), 3 + 3);
        assert_eq!(
            allocation
                .tables
                .iter()
                .map(BlockTable::held_units)
                .sum::<usize>(),
            free - pool.free_units(),
        );
        let units = allocation.tables[1].unit_ids();
        assert_eq!(units.len(), 9);
        assert!(
            units[..3]
                .iter()
                .chain(&units[6..])
                .all(|unit| *unit == UnitId(0))
        );
        assert!(units[3..6].iter().all(|unit| *unit != UnitId(0)));
        assert!(allocation.tables[1].page(0).is_none());
        assert_eq!(allocation.tables[1].fresh_units(0), units[3..6].to_vec());

        // Publication skips the pages without a checkpoint; release keeps
        // the latest completed checkpoint across non-checkpoint boundaries.
        let mut published = Vec::new();
        assert!(coordinator.publish_computed(
            &pool,
            &allocation,
            CachedTokens::prompt(&tokens),
            &lookup.page_hashes,
            10,
            0,
            &mut published,
            true,
            &source,
            None,
        ));
        assert!(coordinator.release_window(&mut allocation, 8, None));
        assert_eq!(allocation.tables[1].start_page(), 1);
        assert!(coordinator.release_window(&mut allocation, 10, None));
        assert!(allocation.tables[1].holds(4, 8));

        // The second chunk checkpoints boundary 16; decoding checkpoints
        // boundary 24, a multiple of the interval, but not boundary 20, in
        // the page the table held without units since token 20.
        coordinator
            .ensure_capacity(&pool, &mut allocation, 10, 18, Checkpoint::Boundary)
            .unwrap();
        assert!(allocation.tables[1].page(3).is_some());
        assert!(allocation.tables[1].page(4).is_none());
        for position in 18..23 {
            coordinator
                .ensure_capacity(
                    &pool,
                    &mut allocation,
                    position,
                    position + 1,
                    Checkpoint::Interval(8),
                )
                .unwrap();
        }
        assert!(allocation.tables[1].page(4).is_none());
        assert!(allocation.tables[1].page(5).is_none());
        allocation.tables[1].mark_sent();
        coordinator
            .ensure_capacity(&pool, &mut allocation, 23, 24, Checkpoint::Interval(8))
            .unwrap();
        assert!(allocation.tables[1].page(5).is_some());
        assert!(allocation.tables[1].changed_since_sent());

        // A later prompt of the first 13 tokens finds no checkpoint at 12,
        // the deepest boundary it may reuse, and reuses the one at 8: its
        // state table starts at the checkpoint's page.
        let hit = coordinator.probe_prefix(
            &pool,
            &tokens[..13],
            true,
            false,
            PrefixScope::default(),
            &source,
        );
        assert_eq!(hit.cached_tokens, 8);
        let mut target = coordinator.empty();
        let reused = coordinator
            .acquire_prefix(
                &pool,
                &mut target,
                &tokens[..13],
                true,
                false,
                PrefixScope::default(),
                &source,
            )
            .unwrap();
        assert_eq!(reused.cached_tokens, 8);
        assert_eq!(
            (target.tables[1].start_page(), target.tables[1].end_page()),
            (1, 2)
        );
        assert_eq!(target.tables[1].unit_ids(), units[3..6].to_vec());
    }

    /// A request resumed from a checkpoint the prefix cache shares (in this
    /// world, or mapped read-only from the predecessor's) and admitted
    /// short of the next boundary still holds that page as its latest
    /// checkpoint. A refill of its draft cache evaluates its tokens again
    /// up to that boundary: its tables name fresh units of the request's
    /// own for the checkpoint it writes, and the shared page keeps its
    /// units for its other readers.
    #[test]
    fn a_refill_checkpoints_into_units_of_its_own() {
        let coordinator = PrefixCacheCoordinator::new(vec![full(4, 1), state(4, 3)]);
        let source = endpoint();
        let prompt = (1..=10).collect::<Vec<u32>>();
        for inherited in [false, true] {
            let publisher = BlockPool::new(64, 0);
            drop(publish(&coordinator, &publisher, &prompt, &source));
            let pool = if inherited {
                let successor = BlockPool::new(128, 0);
                successor.inherit(64..128);
                successor.import(&publisher.export(), 64, &source);
                successor
            } else {
                publisher
            };

            // The hit ends at the checkpoint of boundary 8, and the
            // admission's forward reaches no later boundary.
            let mut resumed = coordinator.empty();
            let hit = coordinator
                .acquire_prefix(
                    &pool,
                    &mut resumed,
                    &prompt,
                    true,
                    false,
                    PrefixScope::default(),
                    &source,
                )
                .unwrap();
            assert_eq!(hit.cached_tokens, 8);
            coordinator
                .ensure_capacity(&pool, &mut resumed, 8, 10, Checkpoint::Boundary)
                .unwrap();
            let shared = resumed.tables[1].page(1).unwrap().units().to_vec();
            assert_eq!(shared.iter().all(|unit| unit.0 >= 64), inherited);

            // The refill's last chunk ends at boundary 8.
            coordinator.hold_states_from(&mut resumed, 0);
            assert_eq!(
                coordinator.units_needed(&pool, &resumed, 0, 8, Checkpoint::Boundary),
                Some(3)
            );
            coordinator
                .ensure_capacity(&pool, &mut resumed, 0, 8, Checkpoint::Boundary)
                .unwrap();
            let written = resumed.tables[1].page(1).unwrap().units().to_vec();
            assert!(
                written
                    .iter()
                    .all(|unit| !shared.contains(unit) && unit.0 < 64)
            );
            let serial = resumed.allocated_units - 3;
            assert_eq!(resumed.tables[1].fresh_units(serial), written);

            let mut other = coordinator.empty();
            let hit = coordinator
                .acquire_prefix(
                    &pool,
                    &mut other,
                    &prompt,
                    true,
                    false,
                    PrefixScope::default(),
                    &source,
                )
                .unwrap();
            assert_eq!(hit.cached_tokens, 8);
            assert_eq!(other.tables[1].page(1).unwrap().units(), shared.as_slice());
        }
    }

    #[test]
    fn unpublished_sliding_pages_return_free() {
        let pool = BlockPool::new(16, 0);
        let coordinator = PrefixCacheCoordinator::new(vec![sliding(4, 1, 4)]);
        let mut allocation = coordinator.empty();
        coordinator
            .ensure_capacity(&pool, &mut allocation, 0, 12, Checkpoint::Boundary)
            .unwrap();
        let first = allocation.tables[0].page(0).unwrap().units()[0];
        assert!(coordinator.release_window(&mut allocation, 12, None));
        assert_eq!(allocation.tables[0].start_page(), 2);
        assert_eq!(pool.unit_state(first), UnitState::Free);
        assert_eq!(pool.free_units(), 14);
    }

    #[test]
    fn a_fresh_sliding_table_starts_at_the_first_page_its_read_needs() {
        let pool = BlockPool::new(32, 0);
        let coordinator = PrefixCacheCoordinator::new(vec![full(4, 1), sliding(4, 1, 5)]);
        let mut allocation = coordinator.empty();

        // A read from token 16 needs history [11, 20): sliding pages 2..5,
        // while the full group needs pages 0..5.
        assert_eq!(
            coordinator.units_needed(&pool, &allocation, 16, 20, Checkpoint::Boundary),
            Some(5 + 3)
        );
        coordinator
            .ensure_capacity(&pool, &mut allocation, 16, 20, Checkpoint::Boundary)
            .unwrap();
        assert_eq!(allocation.tables[1].start_page(), 2);
        assert_eq!(allocation.tables[1].end_page(), 5);
        assert_eq!(allocation.tables[0].start_page(), 0);
    }

    #[test]
    fn prefix_validation_iterates_until_every_sliding_window_holds() {
        let pool = BlockPool::new(64, 0);
        let source = endpoint();
        // A prefix of M tokens needs group 0 pages intersecting [M - 4, M)
        // (4-token pages) and group 1 pages intersecting [M - 8, M) (8-token
        // pages). Candidates are multiples of 8.
        let coordinator = PrefixCacheCoordinator::new(vec![sliding(4, 1, 4), sliding(8, 1, 8)]);
        let prompt = (1..=40).collect::<Vec<u32>>();
        // Group 0 misses pages 3 and 7; group 1 misses page 2. Group 0 lowers
        // 32 to 24, group 1 lowers 24 to 16, group 0 then lowers 16 to 8,
        // where both windows hold: group 0 page 1 and group 1 page 0.
        let _first = publish_pages(&coordinator, &pool, &prompt, 0, &[1, 5], &source);
        let _second = publish_pages(&coordinator, &pool, &prompt, 1, &[0, 1, 3], &source);

        let hit =
            coordinator.probe_prefix(&pool, &prompt, true, false, PrefixScope::default(), &source);
        assert_eq!(hit.cached_tokens, 8);

        let mut target = coordinator.empty();
        let lookup = coordinator
            .acquire_prefix(
                &pool,
                &mut target,
                &prompt,
                true,
                false,
                PrefixScope::default(),
                &source,
            )
            .unwrap();
        assert_eq!(lookup.cached_tokens, 8);
        assert_eq!(
            (target.tables[0].start_page(), target.tables[0].end_page()),
            (1, 2)
        );
        assert_eq!(
            (target.tables[1].start_page(), target.tables[1].end_page()),
            (0, 1)
        );
    }

    #[test]
    fn full_groups_lock_the_whole_prefix_and_sliding_groups_only_their_window() {
        let pool = BlockPool::new(128, 0);
        let source = endpoint();
        // Full pages of 8 tokens and sliding pages of 4 with a 6-token window.
        let coordinator = PrefixCacheCoordinator::new(vec![full(8, 1), sliding(4, 2, 6)]);
        let prompt = (1..=40).collect::<Vec<u32>>();
        // Every full page is cached, but the sliding group holds only pages
        // 2 and 3: the window of M = 16. M = 32 and M = 24 need sliding pages
        // 6 and 7, and 4 and 5.
        let full_pages = publish_pages(&coordinator, &pool, &prompt, 0, &[0, 1, 2, 3, 4], &source);
        let sliding_pages = publish_pages(&coordinator, &pool, &prompt, 1, &[2, 3], &source);
        drop(full_pages);
        drop(sliding_pages);

        // Every hit page is unreferenced: two full units and two sliding
        // pages of two units each count as free until acquired.
        let hit =
            coordinator.probe_prefix(&pool, &prompt, true, false, PrefixScope::default(), &source);
        assert_eq!(hit.cached_tokens, 16);
        assert_eq!(hit.cached_free_units, 2 + 4);

        let mut target = coordinator.empty();
        let lookup = coordinator
            .acquire_prefix(
                &pool,
                &mut target,
                &prompt,
                true,
                false,
                PrefixScope::default(),
                &source,
            )
            .unwrap();
        assert_eq!(lookup.cached_tokens, 16);
        assert_eq!(
            (target.tables[0].start_page(), target.tables[0].end_page()),
            (0, 2)
        );
        assert_eq!(
            (target.tables[1].start_page(), target.tables[1].end_page()),
            (2, 4)
        );
        assert!(
            target
                .tables
                .iter()
                .flat_map(BlockTable::unit_ids)
                .all(|unit| pool.unit_state(unit) == UnitState::Active)
        );
        // Acquired pages hold computed keys and values and are never declared fresh.
        assert!(
            target
                .tables
                .iter()
                .all(|table| table.fresh_units(0).is_empty())
        );

        // The first prefill chunk appends after the prefix in every group.
        coordinator
            .ensure_capacity(&pool, &mut target, 16, 24, Checkpoint::Boundary)
            .unwrap();
        assert_eq!(target.tables[0].end_page(), 3);
        assert_eq!(target.tables[1].end_page(), 6);
    }

    #[test]
    fn a_full_group_miss_limits_the_sliding_prefix() {
        let pool = BlockPool::new(64, 0);
        let source = endpoint();
        let coordinator = PrefixCacheCoordinator::new(vec![sliding(4, 1, 4), full(8, 1)]);
        let prompt = (1..=33).collect::<Vec<u32>>();
        let owner = publish(&coordinator, &pool, &prompt[..16], &source);

        // Only 16 prompt tokens were published: the full group limits every
        // group to 16 even though a longer prompt is probed.
        let hit =
            coordinator.probe_prefix(&pool, &prompt, true, false, PrefixScope::default(), &source);
        assert_eq!(hit.cached_tokens, 16);
        // Both groups' hit pages are referenced by the owner.
        assert_eq!(hit.cached_free_units, 0);
        drop(owner);
        let hit =
            coordinator.probe_prefix(&pool, &prompt, true, false, PrefixScope::default(), &source);
        // Unreferenced hits count as free: sliding page 3 and full pages 0
        // and 1.
        assert_eq!(hit.cached_free_units, 3);
    }

    /// Pages are rooted at the seed of what the workers that computed them
    /// compute: a pool whose workers compute otherwise hits none of them,
    /// but a request continuing the state those workers reached finds them
    /// under their seed.
    #[test]
    fn a_request_continuing_another_worlds_state_finds_its_pages() {
        let pool = BlockPool::new(64, 0);
        let source = endpoint();
        let prompt = (1..=9).collect::<Vec<u32>>();
        let groups = vec![
            full(4, 1),
            GroupShape {
                draft: true,
                ..full(4, 1)
            },
        ];
        let mut previous = PrefixCacheCoordinator::new(groups.clone());
        previous.set_seeds(vec![prefix_seed("previous"), prefix_seed("draft")]);
        drop(publish(&previous, &pool, &prompt, &source));

        let mut serving = PrefixCacheCoordinator::new(groups);
        serving.set_seeds(vec![prefix_seed("previous"), prefix_seed("revised-draft")]);
        assert_eq!(
            serving
                .probe_prefix(&pool, &prompt, true, false, PrefixScope::default(), &source)
                .cached_tokens,
            8
        );
        serving.set_seeds(vec![prefix_seed("serving"), prefix_seed("revised-draft")]);
        let own =
            serving.probe_prefix(&pool, &prompt, true, false, PrefixScope::default(), &source);
        assert_eq!(own.cached_tokens, 0);
        let kept = PrefixScope {
            isolation_key: None,
            state_seeds: &[prefix_seed("previous")],
            resume: None,
        };
        let continued = serving.probe_prefix(&pool, &prompt, true, false, kept, &source);
        assert_eq!(continued.cached_tokens, 8);
    }

    /// A successor's pages imported into the units it inherits are hit like
    /// its own, but are no free capacity until the commit returns those
    /// units to allocation.
    #[test]
    fn imported_pages_count_as_free_only_once_their_units_are_taken() {
        let coordinator = PrefixCacheCoordinator::new(vec![full(4, 1)]);
        let source = endpoint();
        let prompt = (1..=9).collect::<Vec<u32>>();
        let predecessor = BlockPool::new(64, 0);
        drop(publish(&coordinator, &predecessor, &prompt, &source));
        let index = predecessor.export();
        assert_eq!(index.len(), 2);

        let successor = BlockPool::new(64, 0);
        successor.inherit(16..32);
        assert_eq!(successor.import(&index, 16, &source), 2);
        let free = successor.free_units();
        let hit = coordinator.probe_prefix(
            &successor,
            &prompt,
            true,
            false,
            PrefixScope::default(),
            &source,
        );
        assert_eq!(hit.cached_tokens, 8);
        assert_eq!(hit.cached_free_units, 0);
        assert_eq!(successor.free_units(), free);

        successor.take_inherited();
        let hit = coordinator.probe_prefix(
            &successor,
            &prompt,
            true,
            false,
            PrefixScope::default(),
            &source,
        );
        assert_eq!(hit.cached_tokens, 8);
        assert_eq!(hit.cached_free_units, 2);
    }

    /// A predecessor's pool of 16 units whose units 8, 9 and 10 hold
    /// `prompt`'s two complete pages and its partial last page, then released.
    fn pages_past_a_filler(
        coordinator: &PrefixCacheCoordinator,
        prompt: &[u32],
        source: &Arc<WorkerEndpoint>,
    ) -> BlockPool {
        let pool = BlockPool::new(16, 0);
        let filler = pool.allocate(7, 1).unwrap();
        drop(publish(coordinator, &pool, prompt, source));
        drop(filler);
        pool
    }

    /// The inherited units a successor borrows are allocated as its own on
    /// trial, after its own free units, and the predecessor's pages on them
    /// are hit until allocation reaches them; the inherited units it does
    /// not borrow stay out of allocation until the commit.
    #[test]
    fn a_successor_allocates_the_inherited_units_it_borrows() {
        let coordinator = PrefixCacheCoordinator::new(vec![full(4, 1)]);
        let source = endpoint();
        let prompt = (1..=9).collect::<Vec<u32>>();
        let predecessor = pages_past_a_filler(&coordinator, &prompt, &source);

        // The successor maps the predecessor's 16 units after its own 16.
        let successor = BlockPool::new(32, 0);
        successor.inherit(16..32);
        assert_eq!(successor.import(&predecessor.export(), 16, &source), 2);
        assert_eq!(successor.borrow(std::slice::from_ref(&(24..32))), 8);
        assert_eq!(successor.free_units(), 15 + 8);
        assert_eq!(successor.usable_units(), 15 + 8);
        let hit = coordinator.probe_prefix(
            &successor,
            &prompt,
            true,
            false,
            PrefixScope::default(),
            &source,
        );
        assert_eq!(hit.cached_tokens, 8);
        assert_eq!(hit.cached_free_units, 2);

        let pages = successor.allocate(15 + 8, 1).unwrap();
        let units: Vec<u32> = pages.iter().map(|page| page.units()[0].0).collect();
        assert!(
            units[..15].iter().all(|unit| (1..16).contains(unit)),
            "{units:?}"
        );
        assert!(
            units[15..].iter().all(|unit| (24..32).contains(unit)),
            "{units:?}"
        );
        assert!(pages.iter().all(|page| !successor.shared(page)));
        assert!(successor.allocate(1, 1).is_none());
        let hit = coordinator.probe_prefix(
            &successor,
            &prompt,
            true,
            false,
            PrefixScope::default(),
            &source,
        );
        assert_eq!(hit.cached_tokens, 0);

        successor.take_inherited();
        assert_eq!(successor.free_units(), 8);
        drop(pages);
        assert_eq!(successor.free_units(), 31);
    }

    /// A world allocates none of the units it lent its successor, whose
    /// pages it forgets, until it takes them back.
    #[test]
    fn a_world_allocates_no_unit_it_lent_until_it_takes_them_back() {
        let coordinator = PrefixCacheCoordinator::new(vec![full(4, 1)]);
        let source = endpoint();
        let prompt = (1..=9).collect::<Vec<u32>>();
        let pool = pages_past_a_filler(&coordinator, &prompt, &source);
        assert_eq!(pool.free_units(), 15);

        assert_eq!(pool.lend(std::slice::from_ref(&(8..16))), 2);
        assert_eq!(pool.lent(), vec![8..16]);
        assert_eq!(pool.free_units(), 7);
        assert_eq!(pool.usable_units(), 7);
        let hit =
            coordinator.probe_prefix(&pool, &prompt, true, false, PrefixScope::default(), &source);
        assert_eq!(hit.cached_tokens, 0);
        let pages = pool.allocate(7, 1).unwrap();
        assert!(pages.iter().all(|page| page.units()[0].0 < 8));
        assert!(pool.allocate(1, 1).is_none());

        assert_eq!(pool.reclaim(), 8);
        assert!(pool.lent().is_empty());
        assert_eq!(pool.free_units(), 8);
        assert_eq!(pool.usable_units(), 15);
        drop(pages);
        assert_eq!(pool.free_units(), 15);
    }

    #[test]
    fn prefix_copies_follow_worker_incarnations_without_releasing_active_pages() {
        let pool = BlockPool::new(5, 0);
        let coordinator = PrefixCacheCoordinator::new(vec![full(4, 1)]);
        let prompt = [1, 2, 3, 4, 5];
        let first = endpoint();
        let second = Arc::new(WorkerEndpoint {
            incarnation: "replica-loaded".into(),
            ..first.as_ref().clone()
        });
        let owners = [&first, &second].map(|source| {
            let allocation = publish(&coordinator, &pool, &prompt, source);
            assert_eq!(
                coordinator
                    .probe_prefix(&pool, &prompt, true, false, PrefixScope::default(), source)
                    .cached_tokens,
                4
            );
            allocation
        });
        let replacement = WorkerEndpoint {
            incarnation: "reloaded".into(),
            ..first.as_ref().clone()
        };
        assert_eq!(
            coordinator
                .probe_prefix(
                    &pool,
                    &prompt,
                    true,
                    false,
                    PrefixScope::default(),
                    &replacement
                )
                .cached_tokens,
            0
        );

        let free_before = pool.free_units();
        pool.invalidate_source(&first);
        assert_eq!(pool.free_units(), free_before);
        assert_eq!(
            coordinator
                .probe_prefix(&pool, &prompt, true, false, PrefixScope::default(), &first)
                .cached_tokens,
            0
        );
        let mut consumer = coordinator.empty();
        let reused = coordinator
            .acquire_prefix(
                &pool,
                &mut consumer,
                &prompt,
                true,
                false,
                PrefixScope::default(),
                &second,
            )
            .unwrap();
        assert_eq!(reused.cached_tokens, 4);
        assert_eq!(
            consumer.tables[0].unit_ids(),
            owners[1].tables[0].unit_ids()[..1]
        );
        let [first_owner, _] = owners;
        drop(first_owner);
        assert_eq!(pool.free_units(), free_before + 2);
    }

    #[test]
    fn fresh_units_follow_the_allocation_serial_in_every_group() {
        let pool = BlockPool::new(32, 0);
        let coordinator = PrefixCacheCoordinator::new(vec![full(4, 1), sliding(2, 2, 4)]);
        let mut allocation = coordinator.empty();
        coordinator
            .ensure_capacity(&pool, &mut allocation, 0, 4, Checkpoint::Boundary)
            .unwrap();
        let watermark = allocation.allocated_units();
        assert_eq!(watermark, 1 + 4);
        allocation.tables.iter_mut().for_each(BlockTable::mark_sent);
        assert!(!allocation.tables.iter().any(BlockTable::changed_since_sent));

        // Growth to six tokens adds one full page and one sliding page; only
        // those units are fresh past the watermark, and both tables changed.
        coordinator
            .ensure_capacity(&pool, &mut allocation, 4, 6, Checkpoint::Boundary)
            .unwrap();
        assert_eq!(allocation.tables[0].fresh_units(watermark).len(), 1);
        assert_eq!(allocation.tables[1].fresh_units(watermark).len(), 2);
        assert!(allocation.tables.iter().all(BlockTable::changed_since_sent));
    }
}
