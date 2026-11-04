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

use std::collections::HashMap;
use std::sync::Arc;

use platform::{Allocator, Buffer, DevicePtr};

use crate::plan::Bucket;
use crate::{Error, Result};

/// The executor's name for one of a runner's streams.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct StreamId(pub usize);

/// Where one bucket's activations live in a pool.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// The byte offset of each allocation from the pool's base, in allocation order.
    offsets: Vec<u64>,
    /// The bytes the bucket's plan uses.
    bytes: u64,
}

impl Plan {
    /// The bytes the bucket's plan uses.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The pool offset of the allocation with this ordinal.
    pub fn offset(&self, ordinal: usize) -> Option<u64> {
        self.offsets.get(ordinal).copied()
    }

    /// How many allocations the plan holds.
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Whether the plan holds no allocation.
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

/// The pool of one stream: its reserved range, how much of it is mapped, and each bucket's plan.
struct Pool {
    buffer: Buffer,
    mapped: u64,
    plans: HashMap<Bucket, Plan>,
}

/// Every graph pool of one runner, by stream.
pub struct Pools {
    allocator: Arc<Allocator>,
    bytes: u64,
    pools: HashMap<StreamId, Pool>,
}

impl Pools {
    /// Reserves `bytes` of virtual address space per stream, on demand.
    ///
    /// The range is reserved, not mapped: physical blocks are mapped as a bucket's plan needs them
    /// and booked against the device budget then. A reservation past the budget fails here.
    pub fn new(allocator: Arc<Allocator>, bytes: u64) -> Self {
        Self {
            allocator,
            bytes,
            pools: HashMap::new(),
        }
    }

    /// The bytes each stream's range reserves.
    pub fn reserved(&self) -> u64 {
        self.bytes
    }

    /// The bytes mapped over every pool.
    pub fn mapped(&self) -> u64 {
        self.pools.values().map(|pool| pool.mapped).sum()
    }

    /// The plan of `bucket` on `stream`, if it was captured.
    pub fn plan(&self, stream: StreamId, bucket: Bucket) -> Option<&Plan> {
        self.pools.get(&stream)?.plans.get(&bucket)
    }

    /// The pool's base address on `stream`, reserving the range on first use.
    pub fn base(&mut self, stream: StreamId) -> Result<DevicePtr> {
        let pool = self.pool(stream)?;
        Ok(pool.buffer.ptr())
    }

    /// Begins planning (first capture) or replaying `bucket` on `stream`, and answers whether the
    /// capture is planning (true) or replaying a recorded plan (false).
    pub fn begin(&mut self, stream: StreamId, bucket: Bucket) -> Result<bool> {
        let pool = self.pool(stream)?;
        Ok(!pool.plans.contains_key(&bucket))
    }

    /// Records the plan of a bucket's first capture, from the allocations the model made.
    ///
    /// `offsets` are the pool offsets `nn::Arena` handed out in allocation order, relative to the
    /// pool's base, and `bytes` the extent they reached. An empty plan is a bucket without
    /// activations, which is legal.
    pub fn record(&mut self, stream: StreamId, bucket: Bucket, offsets: Vec<u64>, bytes: u64) -> Result<()> {
        if bytes > self.bytes {
            return Err(Error::Invalid(format!(
                "bucket {bucket} needs {bytes} bytes, the pool reserves {}",
                self.bytes
            )));
        }
        let pool = self.pool(stream)?;
        // The plan's extent is what must be mapped, so map up to it and book it.
        if bytes > pool.mapped {
            pool.buffer.map(bytes)?;
            pool.mapped = bytes;
        }
        pool.plans.insert(bucket, Plan { offsets, bytes });
        Ok(())
    }

    /// Releases every plan and the pool's mappings of one stream (an executor close).
    pub fn release(&mut self, stream: StreamId) -> Result<()> {
        if let Some(mut pool) = self.pools.remove(&stream) {
            if pool.mapped > 0 {
                pool.buffer.unmap(pool.mapped)?;
            }
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
            let buffer = self.allocator.reserve(self.bytes)?;
            self.pools.insert(
                stream,
                Pool {
                    buffer,
                    mapped: 0,
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
            .field("bytes", &self.bytes)
            .field("mapped", &self.mapped())
            .field("streams", &self.pools.len())
            .finish()
    }
}
