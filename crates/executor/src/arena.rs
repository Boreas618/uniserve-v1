//! The graph pools: one reserved range per stream, and the static activation plan of each bucket.
//!
//! A captured graph addresses fixed buffers, so a bucket's activations must live at addresses that
//! do not move while the graph exists, and a serving step must not map or allocate anything
//! (specs/worker.md 4.5). This module owns the executor side of that:
//!
//! - [`Pools`] reserves one virtual range per stream through `platform`'s allocator, which books
//!   every mapping against the device's budget before it maps (4.1);
//! - the first capture of a bucket plans it: the model's allocations come from `nn::Arena` in a
//!   fixed order, and the pool records where each landed (the plan);
//! - every later use of the bucket reconstructs `nn::Arena` at the plan's base, so the same
//!   allocation sequence returns the same addresses and nothing is mapped.
//!
//! `nn::Arena` itself is the bump allocator the model reads through `Forward` (the model swarm's
//! ruling: it knows a base pointer, a length and a plan, and makes no driver call). Its plan is
//! built here, because only the executor knows when a capture is planning and when it is replaying.
//!
//! Captures run largest first (see [`crate::plan::Buckets::capture_order`]), so a smaller bucket's
//! plan reuses the mapped blocks of a larger one instead of growing the pool.

use std::collections::{BTreeMap, HashMap};

use platform::{Allocator, Buffer, DeviceId, DevicePtr, PoolKind};

use crate::plan::Bucket;
use crate::{Error, Result};

/// What the pool arithmetic answers when a reservation does not fit in `usize`.
const POOL_TOO_LARGE: &str = "a device pool is larger than memory";

/// The executor's name for one of a runner's streams.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct StreamId(pub usize);

/// Where one bucket's activations live in a pool.
///
/// The plan is the extent of the bucket's static allocations, not their list: `nn::Arena` hands
/// out addresses in the order the forward reserves them, and every capture and every replay of a
/// bucket walks the same sequence, so the addresses are the same each time. What the pool has to
/// remember is how far a bucket reaches, to map that much of its range and to check it fits.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// The bytes the bucket's plan uses, from the pool's base.
    bytes: u64,
}

impl Plan {
    /// The bytes the bucket's plan uses.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Whether the plan holds no allocation.
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
}

/// What one stream's pool holds: its reserved range, and the plans of its buckets.
struct Pool {
    buffer: Buffer,
    plans: HashMap<Bucket, Plan>,
}

/// The device's graph pools, by stream.
///
/// One `Pools` per device, shared by every runner that issues on it: two runners each reserving
/// their own range would double-book the device's graph storage and break 4.1's budget by
/// construction. Each stream gets one buffer of its share; the ledger books each bucket's extent
/// inside it.
pub struct Pools {
    /// The device's graph allocator, the one this pool reserved.
    allocator: Allocator,
    /// The pool's reservation per stream and every plan's booking.
    ledger: Ledger,
    pools: HashMap<StreamId, Pool>,
    /// How many streams share the reservation.
    streams: usize,
    /// Whether the scratch pool the shadow pair's region comes from is reserved.
    scratch_reserved: bool,
}

/// What a runner's graph pools hold, for the metrics report and item 5's invariant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolMetrics {
    /// Pools with a reserved range.
    pub streams: usize,
    /// Bytes of virtual address space reserved over every pool.
    pub reserved: u64,
    /// Bytes mapped over every pool.
    pub mapped: u64,
    /// The device budget the mappings are booked against.
    pub budget: u64,
}

/// What the device's one allocator has mapped, per pool kind (4.1, item 5's invariant).
///
/// One allocator per device owns all three pools, so these figures are one booking's totals and
/// not the sum of two owners' books: the graph range the captured graphs address, the scratch the
/// shadow pair's region and the paired forward's pages come from, and the resident pools another
/// owner carves. `budget` is the allowance they are all booked against.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceMemory {
    /// Bytes mapped for captured graphs and their arena plans.
    pub graph: u64,
    /// Bytes mapped for scratch: the shadow pair's region and the paired forward's pages.
    pub scratch: u64,
    /// Bytes mapped for resident storage by its owner, when the executor shares the allocator.
    pub resident: u64,
    /// The device budget every mapping is booked against.
    pub budget: u64,
}

/// The books of a runner's graph pools: the reservation per stream, each bucket's planned extent,
/// and how much of each pool is mapped.
///
/// No driver call happens here, so the two facts item 5 asserts - `mapped <= budget` and a purged
/// generation's bytes coming back - are decided in one place and can be checked without a device.
/// [`Pools`] maps and unmaps exactly what these books say.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Ledger {
    /// The bytes each stream's range reserves.
    reservation: u64,
    /// The device bytes every mapping is booked against (4.1).
    budget: u64,
    streams: BTreeMap<StreamId, StreamBooks>,
}

/// One stream's books: what its plans need and what is mapped of them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct StreamBooks {
    mapped: u64,
    /// The extent of each planned bucket.
    extents: BTreeMap<Bucket, u64>,
}

impl Ledger {
    fn new(reservation: u64, budget: u64) -> Self {
        Self {
            reservation,
            budget,
            streams: BTreeMap::new(),
        }
    }

    /// Books a new stream's reserved range, refusing a reservation the device budget cannot hold
    /// beside the streams already reserved (4.1: a booking past the budget fails, it does not wait).
    fn open(&mut self, stream: StreamId) -> Result<()> {
        if self.streams.contains_key(&stream) {
            return Ok(());
        }
        let reserved = self
            .reservation
            .saturating_mul(self.streams.len() as u64 + 1);
        if reserved > self.budget {
            return Err(Error::Invalid(format!(
                "a graph pool of {} bytes on stream {stream:?} would reserve {reserved} bytes of a {}-byte budget",
                self.reservation, self.budget
            )));
        }
        self.streams.insert(stream, StreamBooks::default());
        Ok(())
    }

    /// Books `bucket`'s extent on `stream` and answers the bytes that must be mapped for it.
    fn note(&mut self, stream: StreamId, bucket: Bucket, bytes: u64) -> Result<u64> {
        if bytes > self.reservation {
            return Err(Error::Invalid(format!(
                "bucket {bucket} needs {bytes} bytes, the pool reserves {}",
                self.reservation
            )));
        }
        let books = self
            .streams
            .get_mut(&stream)
            .ok_or_else(|| Error::State("the graph pool was not opened".into()))?;
        books.extents.insert(bucket, bytes);
        let needed = books.extents.values().copied().max().unwrap_or(0);
        let map = needed.saturating_sub(books.mapped);
        books.mapped = books.mapped.max(needed);
        Ok(map)
    }

    /// Books `bucket`'s release on `stream` and answers the bytes that must be unmapped: the pool
    /// shrinks to the largest extent it still plans.
    fn release(&mut self, stream: StreamId, bucket: Bucket) -> u64 {
        let Some(books) = self.streams.get_mut(&stream) else {
            return 0;
        };
        if books.extents.remove(&bucket).is_none() {
            return 0;
        }
        let kept = books.extents.values().copied().max().unwrap_or(0);
        let unmap = books.mapped.saturating_sub(kept);
        books.mapped = kept;
        unmap
    }

    /// Books a stream's whole release.
    fn close(&mut self, stream: StreamId) -> u64 {
        self.streams.remove(&stream).map_or(0, |books| books.mapped)
    }

    fn mapped(&self) -> u64 {
        self.streams.values().map(|books| books.mapped).sum()
    }

    fn metrics(&self) -> PoolMetrics {
        PoolMetrics {
            streams: self.streams.len(),
            reserved: self.reservation * self.streams.len() as u64,
            mapped: self.mapped(),
            budget: self.budget,
        }
    }
}

/// The device's allowance, as the allocator books against it (4.1).
///
/// `platform::Allocator` calls `book` with the NEW CUMULATIVE TOTAL of one
/// `(device, pool)` before it maps anything, so a total past the allowance is
/// refused and nothing is mapped; the books are the allocator's, and this
/// answers from the budget the graph pools were sized against. A refusal books
/// nothing because there is nothing here to change: the total is the allocator's
/// own figure, not a running count of ours that a failed call could leave
/// inflated.
impl platform::Budget for Ledger {
    fn book(&self, device: DeviceId, pool: PoolKind, bytes: u64) -> platform::Result<()> {
        if bytes > self.budget {
            return Err(platform::Error::InvalidArgument(format!(
                "{pool:?} on device {device:?} would hold {bytes} bytes of a {}-byte allowance",
                self.budget
            )));
        }
        Ok(())
    }
}

impl Pools {
    /// Reserves the device's graph pool and divides it among `streams` ranges.
    ///
    /// `bytes` is the device's whole graph storage (`capacity::graph_storage_budget_bytes`) and
    /// each stream gets `bytes / streams` of it: a captured graph addresses fixed buffers, so a
    /// stream's range must not move while its graphs live, and two streams' work runs at once.
    /// `budget` is the device budget the pool is booked against, kept apart from the reservation so
    /// a report of one cannot pass the other's invariant by construction.
    pub fn new(allocator: Allocator, bytes: u64, budget: u64, streams: usize) -> Result<Self> {
        let streams = streams.max(1);
        let mut allocator = allocator;
        let size = usize::try_from(bytes)
            .map_err(|_| Error::Resource("the graph storage is larger than memory".into()))?;
        allocator.reserve(PoolKind::Graph, size)?;
        Ok(Self {
            allocator,
            ledger: Ledger::new(bytes / streams as u64, budget),
            pools: HashMap::new(),
            streams,
            scratch_reserved: false,
        })
    }

    /// The same, reserving every pool the device maps in one place (4.1): the graph range the
    /// captured graphs address, the scratch the shadow pair's region and the paired forward's
    /// pages come from, and the resident range the storage owners carve.
    ///
    /// One `Pools` per device is the only allocator on it, so the three pool kinds are one booking
    /// and item 5's `graph + scratch + resident <= budget` is decided by one book.
    pub fn with_pools(allocator: Allocator, graph: u64, scratch: u64, resident: u64, budget: u64, streams: usize) -> Result<Self> {
        let mut pools = Self::new(allocator, graph, budget, streams)?;
        for (pool, bytes) in [(PoolKind::Scratch, scratch), (PoolKind::Resident, resident)] {
            if bytes == 0 {
                continue;
            }
            let size = usize::try_from(bytes)
                .map_err(|_| Error::Resource(POOL_TOO_LARGE))?;
            pools.allocator.reserve(pool, size)?;
        }
        pools.scratch_reserved = scratch != 0;
        Ok(pools)
    }

    /// Maps `bytes` of `pool` through the device's one allocator, for an owner that carves its own
    /// storage (the resident planes, a scratch page set).
    pub fn allocate(&mut self, pool: PoolKind, bytes: u64) -> Result<Buffer> {
        let size = usize::try_from(bytes)
            .map_err(|_| Error::Resource("an allocation is larger than memory".into()))?;
        Ok(self.allocator.allocate(pool, size)?)
    }

    /// Gives a `pool` block back to the device's allocator; the block stays mapped, as 4.1 has it.
    pub fn free(&mut self, buffer: Buffer) {
        self.allocator.free(buffer);
    }

    /// The bytes `pool` has mapped, as the device's one allocator books them (4.1).
    pub fn mapped_bytes(&self, pool: PoolKind) -> u64 {
        self.allocator.mapped_bytes(pool) as u64
    }

    /// The device's allowance.
    pub fn budget(&self) -> u64 {
        self.ledger.budget
    }

    /// The bytes each stream's range reserves.
    pub fn reserved(&self) -> u64 {
        self.ledger.reservation
    }

    /// The device's allocator, for the one other pool the executor carves from it.
    ///
    /// `ops::Twins` is allocator-backed (the shadow pair's region), so it is taken from the same
    /// allocator the graph pool reserved: one allocator per device, so the device's budget books
    /// the twins beside the graphs rather than behind a second ledger.
    pub fn allocator(&mut self) -> &mut Allocator {
        &mut self.allocator
    }

    /// Reserves the scratch pool the shadow pair's region comes from, once, on first use.
    pub fn reserve_scratch(&mut self, bytes: u64) -> Result<()> {
        if self.scratch_reserved {
            return Ok(());
        }
        let size = usize::try_from(bytes)
            .map_err(|_| Error::Resource("the shadow region is larger than memory".into()))?;
        self.allocator.reserve(PoolKind::Scratch, size)?;
        self.scratch_reserved = true;
        Ok(())
    }

    /// The bytes the device's graph pool has mapped, as the allocator books them (4.1).
    pub fn mapped(&self) -> u64 {
        self.allocator.mapped_bytes(PoolKind::Graph) as u64
    }

    /// How many streams share the range.
    pub fn streams(&self) -> usize {
        self.streams
    }

    /// The device's mapped bytes per pool kind, from the one allocator that books them all.
    pub fn memory(&self) -> DeviceMemory {
        DeviceMemory {
            graph: self.allocator.mapped_bytes(PoolKind::Graph) as u64,
            scratch: self.allocator.mapped_bytes(PoolKind::Scratch) as u64,
            resident: self.allocator.mapped_bytes(PoolKind::Resident) as u64,
            budget: self.ledger.budget,
        }
    }

    /// What the device's graph pools hold and what they may hold (4.1, item 5's invariant).
    pub fn metrics(&self) -> PoolMetrics {
        let booked = self.ledger.metrics();
        PoolMetrics {
            streams: self.pools.len(),
            reserved: booked.reserved,
            mapped: self.mapped(),
            budget: booked.budget,
        }
    }

    /// The plan of `bucket` on `stream`, if it was captured.
    pub fn plan(&self, stream: StreamId, bucket: Bucket) -> Option<&Plan> {
        self.pools.get(&stream)?.plans.get(&bucket)
    }

    /// The base address of `stream`'s range, allocating the range on first use.
    pub fn base(&mut self, stream: StreamId) -> Result<DevicePtr> {
        let pool = self.pool(stream)?;
        Ok(DevicePtr::from_address(pool.buffer.address()))
    }

    /// Begins planning (first capture) or replaying `bucket` on `stream`, and answers whether the
    /// capture is planning (true) or replaying a recorded plan (false).
    pub fn begin(&mut self, stream: StreamId, bucket: Bucket) -> Result<bool> {
        let pool = self.pool(stream)?;
        Ok(!pool.plans.contains_key(&bucket))
    }

    /// Records the plan of a bucket's first capture: how far its arena reached in the range.
    ///
    /// `nn::Arena` hands out addresses in the order the forward reserves them, and every capture
    /// and replay of a bucket walks the same sequence, so the extent is all the pool has to
    /// remember: the addresses repeat. A zero extent is a bucket without activations.
    pub fn record(&mut self, stream: StreamId, bucket: Bucket, bytes: u64) -> Result<()> {
        self.ledger.note(stream, bucket, bytes)?;
        let pool = self.pool(stream)?;
        pool.plans.insert(bucket, Plan { bytes });
        Ok(())
    }

    /// Stops planning `bucket` on `stream` once its graph is gone (7.3's purge).
    ///
    /// The booking goes; the pool's mapping stays, because 4.1 returns physical storage to the
    /// device only when the pool is destroyed or shrunk. Answers the bytes the bucket no longer
    /// books.
    pub fn release_bucket(&mut self, stream: StreamId, bucket: Bucket) -> Result<u64> {
        let bytes = self
            .pools
            .get(&stream)
            .and_then(|pool| pool.plans.get(&bucket).map(Plan::bytes))
            .unwrap_or(0);
        self.ledger.release(stream, bucket);
        if let Some(pool) = self.pools.get_mut(&stream) {
            pool.plans.remove(&bucket);
        }
        Ok(bytes)
    }

    /// Gives `stream`'s range back: its plans go and the pool takes the buffer's block back.
    pub fn release(&mut self, stream: StreamId) -> Result<()> {
        self.ledger.close(stream);
        if let Some(pool) = self.pools.remove(&stream) {
            self.allocator.free(pool.buffer);
        }
        Ok(())
    }

    /// Releases every stream's pool.
    pub fn release_all(&mut self) -> Result<()> {
        let streams: Vec<StreamId> = self.pools.keys().copied().collect();
        for stream in streams {
            self.release(stream)?;
        }
        Ok(())
    }

    fn pool(&mut self, stream: StreamId) -> Result<&mut Pool> {
        if !self.pools.contains_key(&stream) {
            self.ledger.open(stream)?;
            let bytes = usize::try_from(self.ledger.reservation).map_err(|_| {
                Error::Resource("a stream's graph range is larger than memory".into())
            })?;
            let buffer = self.allocator.allocate(PoolKind::Graph, bytes)?;
            self.pools.insert(
                stream,
                Pool {
                    buffer,
                    plans: HashMap::new(),
                },
            );
        }
        self.pools
            .get_mut(&stream)
            .ok_or_else(|| Error::State("the graph pool was not created".into()))
    }
}

impl std::fmt::Debug for Pools {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Pools")
            .field("bytes", &self.ledger.reservation)
            .field("mapped", &self.mapped())
            .field("streams", &self.pools.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Variant;

    /// A capture shape of `rows` decode rows.
    fn bucket(rows: u32) -> Bucket {
        Bucket {
            variant: Variant::DecodeGreedy,
            rows,
            tokens: 1,
        }
    }

    #[test]
    fn a_reservation_past_the_device_budget_is_refused_before_it_maps() {
        let mut ledger = Ledger::new(1000, 1500);
        ledger.open(StreamId(0)).expect("the first stream fits");
        let refused = ledger.open(StreamId(1));
        assert!(
            refused.is_err(),
            "1000 + 1000 bytes past a 1500-byte budget"
        );
        // The refusal leaves the books as they were, so nothing is reserved twice.
        assert_eq!(ledger.metrics().streams, 1);
    }

    #[test]
    fn a_plan_maps_only_the_bytes_it_adds_and_never_past_the_reservation() {
        let mut ledger = Ledger::new(1000, 4000);
        ledger.open(StreamId(0)).expect("a stream fits");
        assert_eq!(ledger.note(StreamId(0), bucket(1), 800).expect("fits"), 800);
        // A smaller bucket lives inside the mapped range: nothing new is mapped.
        assert_eq!(ledger.note(StreamId(0), bucket(2), 300).expect("fits"), 0);
        assert_eq!(ledger.mapped(), 800);
        assert!(ledger.note(StreamId(0), bucket(3), 2000).is_err());
    }

    #[test]
    fn a_purged_bucket_returns_its_bytes_and_the_pool_shrinks_to_what_it_still_plans() {
        let mut ledger = Ledger::new(1000, 4000);
        ledger.open(StreamId(0)).expect("a stream fits");
        ledger.note(StreamId(0), bucket(1), 800).expect("fits");
        ledger.note(StreamId(0), bucket(2), 600).expect("fits");
        assert_eq!(ledger.mapped(), 800);
        // Purging the largest plan shrinks the mapping to the next one.
        assert_eq!(ledger.release(StreamId(0), bucket(1)), 200);
        assert_eq!(ledger.mapped(), 600);
        // Purging a bucket that was never planned gives nothing back.
        assert_eq!(ledger.release(StreamId(0), bucket(9)), 0);
        assert_eq!(ledger.release(StreamId(0), bucket(2)), 600);
        assert_eq!(ledger.mapped(), 0);
    }

    #[test]
    fn two_generations_capture_inside_one_mapped_range() {
        let mut ledger = Ledger::new(1000, 4000);
        ledger.open(StreamId(0)).expect("a stream fits");
        // The old generation's plan, then the new generation's plan of the same bucket: the
        // second capture reuses the mapped range instead of mapping beside it.
        ledger.note(StreamId(0), bucket(1), 900).expect("fits");
        ledger.note(StreamId(0), bucket(1), 900).expect("fits");
        assert_eq!(ledger.mapped(), 900);
        assert!(ledger.metrics().mapped <= ledger.metrics().budget);
    }

    #[test]
    fn a_pool_that_closes_gives_all_of_its_bytes_back() {
        let mut ledger = Ledger::new(1000, 4000);
        ledger.open(StreamId(0)).expect("a stream fits");
        ledger.open(StreamId(1)).expect("a second stream fits");
        ledger.note(StreamId(0), bucket(1), 700).expect("fits");
        ledger.note(StreamId(1), bucket(1), 400).expect("fits");
        assert_eq!(ledger.close(StreamId(0)), 700);
        assert_eq!(
            ledger.metrics(),
            PoolMetrics {
                streams: 1,
                reserved: 1000,
                mapped: 400,
                budget: 4000,
            }
        );
    }
}
