//! Bounded export ranges and asynchronous observation of remote readers.

use crate::transfer::{ACK_SLOTS, CLAIMED};
use std::collections::BTreeMap;
use std::sync::Arc;

use cuda::{
    CopyBatch, DeviceGuard, DeviceId, Event, EventFlags, PinnedBuffer, SourceOrder, Stream,
};

use crate::memory::allocation::Storage;
use crate::stream::{Fill, Issue};
use crate::{DescriptorGrants, Error, Result};

pub(crate) const HEADER_BYTES: usize = ACK_SLOTS * size_of::<u32>();
pub(crate) const CHUNK_ALIGNMENT: usize = 512;

/// Bytes of the chunk holding a payload of `bytes`: the reader-word header
/// and the payload, rounded up to [`CHUNK_ALIGNMENT`].
fn chunk_bytes(bytes: usize) -> Option<usize> {
    bytes
        .checked_add(HEADER_BYTES)
        .and_then(|value| value.checked_add(CHUNK_ALIGNMENT - 1))
        .map(|value| value / CHUNK_ALIGNMENT * CHUNK_ALIGNMENT)
}

/// Bytes of a pool that holds `count` payloads of at most `bytes` each at
/// once, every payload in its own chunk.
pub(crate) fn pool_bytes(count: u64, bytes: usize) -> u64 {
    let chunk = chunk_bytes(bytes).unwrap_or(usize::MAX) as u64;
    count.saturating_mul(chunk)
}

/// An immutable byte range. The payload follows the remote-reader header.
pub(crate) struct PoolChunk {
    pub(crate) offset: usize,
    pub(crate) nbytes: usize,
}

impl PoolChunk {
    pub(crate) fn payload_offset(&self) -> usize {
        self.offset + HEADER_BYTES
    }
}

struct Allocation {
    chunk: Arc<PoolChunk>,
    // None denotes a live export; Some selects readers after revocation.
    readers: Option<u64>,
    grant: Option<(Arc<DescriptorGrants>, String)>,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if let Some((grants, export)) = &self.grant {
            grants.release(export);
        }
    }
}

/// Own chunk placement and retirement within one device's exported mapping.
///
/// `backing` retains the exportable allocation the pool places chunks in.
/// Reader words are copied on a private stream. Polling observes a completed
/// snapshot and never waits for device work on the host.
pub(crate) struct VmmPool {
    backing: Option<Arc<Storage>>,
    device: DeviceId,
    address: usize,
    capacity: usize,
    watermark: usize,
    allocations: BTreeMap<usize, Allocation>,
    /// Producer streams with header initialization this pool has not
    /// drained, one handle per stream.
    initializers: Vec<Stream>,
    stream: Stream,
    completion: Event,
    observed: Vec<usize>,
    readback: Vec<PinnedBuffer>,
    readback_slot: usize,
    failed: bool,
}

impl VmmPool {
    /// The mapping must cover `capacity` writable bytes on `device` and remain
    /// valid as long as `backing` is held. Construction performs no device copy.
    ///
    /// # Safety
    /// `address` must identify that mapping.
    pub(crate) unsafe fn new(
        backing: Arc<Storage>,
        device: DeviceId,
        address: usize,
        capacity: usize,
    ) -> Result<Self> {
        Ok(Self {
            backing: Some(backing),
            device,
            address,
            capacity,
            watermark: 0,
            allocations: BTreeMap::new(),
            initializers: Vec::new(),
            stream: Stream::new(device)?,
            completion: Event::new(EventFlags::default()),
            observed: Vec::new(),
            readback: Vec::new(),
            readback_slot: 0,
            failed: false,
        })
    }

    pub(crate) fn backing(&self) -> Result<&Arc<Storage>> {
        self.backing
            .as_ref()
            .ok_or(Error::State("VMM pool is closed"))
    }

    /// Reserve a payload and clear its reader words in the order of the
    /// producer `issue`'s stream. None reports capacity exhaustion without
    /// reserving any bytes. The pool shares that stream until close, which
    /// drains the header initialization issued here.
    pub(crate) fn reserve(
        &mut self,
        bytes: usize,
        issue: impl Issue,
    ) -> Result<Option<Arc<PoolChunk>>> {
        let stream = issue
            .stream()
            .ok_or_else(|| Error::Invalid("a VMM pool chunk is cleared on a stream".into()))?;
        self.backing()?;
        if self.failed {
            return Err(Error::Resource("VMM pool device completion is unknown"));
        }
        if bytes == 0 {
            return Err(Error::Invalid(
                "a pool chunk needs a positive length".into(),
            ));
        }
        let span =
            chunk_bytes(bytes).ok_or_else(|| Error::Invalid("VMM payload is too large".into()))?;
        let Some(offset) = self.free_offset(span) else {
            return Ok(None);
        };
        let chunk = Arc::new(PoolChunk {
            offset,
            nbytes: span,
        });
        self.allocations.insert(
            offset,
            Allocation {
                chunk: Arc::clone(&chunk),
                readers: None,
                grant: None,
            },
        );

        // A default-stream handle resolves against the current context.
        let _device = DeviceGuard::new(self.device)?;
        if !self
            .initializers
            .iter()
            .any(|known| known.raw() == stream.raw())
        {
            self.initializers.push(stream.clone());
        }
        let header = (self.address + offset) as u64;
        // SAFETY: this reservation owns all 64 aligned header words. Later
        // payload copies and the export's readiness event use this same
        // stream.
        let filled = unsafe {
            issue.fill(Fill {
                address: header,
                value: 0,
                width: 4,
                count: ACK_SLOTS,
            })
        };
        if let Err(error) = filled {
            self.failed = true;
            return Err(error);
        }
        Ok(Some(chunk))
    }

    fn free_offset(&mut self, span: usize) -> Option<usize> {
        if span <= self.capacity - self.watermark {
            let offset = self.watermark;
            self.watermark += span;
            return Some(offset);
        }

        let mut cursor = 0;
        for (&offset, allocation) in &self.allocations {
            if span <= offset - cursor {
                return Some(cursor);
            }
            cursor = offset + allocation.chunk.nbytes;
        }
        if span <= self.capacity - cursor {
            self.watermark = self.watermark.max(cursor + span);
            Some(cursor)
        } else {
            None
        }
    }

    /// Return a range whose accesses the caller has already drained. Releasing
    /// a retired or replaced handle cannot release a different live export.
    pub(crate) fn release(&mut self, chunk: &Arc<PoolChunk>) -> Result<()> {
        let Some(allocation) = self.allocations.get(&chunk.offset) else {
            return Ok(());
        };
        if !Arc::ptr_eq(&allocation.chunk, chunk) {
            return Ok(());
        }
        if allocation.readers.is_some() {
            return Err(Error::State("VMM chunk still awaits reader retirement"));
        }
        self.remove(chunk.offset);
        Ok(())
    }

    /// Revoke a chunk after its consuming calls have resolved. Existing remote
    /// readers may still access it; no new reader may claim it after this call.
    pub(crate) fn retire(
        &mut self,
        chunk: &Arc<PoolChunk>,
        consumers: &[usize],
        producer: &Event,
        grant: Option<(Arc<DescriptorGrants>, String)>,
    ) -> Result<()> {
        let mut readers = 0;
        for &slot in consumers {
            if slot >= ACK_SLOTS {
                return Err(Error::Invalid(
                    "VMM acknowledgment slot is out of range".into(),
                ));
            }
            readers |= 1u64 << slot;
        }
        let allocation = self
            .allocations
            .get_mut(&chunk.offset)
            .filter(|allocation| Arc::ptr_eq(&allocation.chunk, chunk))
            .ok_or(Error::State("VMM chunk is no longer allocated"))?;
        if allocation.readers.is_some() {
            return Ok(());
        }

        if readers == 0 && producer.query()? {
            if let Some((grants, export)) = grant {
                grants.release(&export);
            }
            self.remove(chunk.offset);
            return Ok(());
        }

        // Enqueue this dependency now, before the event pool can reuse the
        // producer event. A later snapshot must not observe the header before
        // its initialization or outlive the producer's final payload write.
        self.stream.wait_event(producer)?;
        allocation.readers = Some(readers);
        allocation.grant = grant;
        Ok(())
    }

    pub(crate) fn awaiting_acknowledgment(&self) -> bool {
        self.allocations
            .values()
            .any(|allocation| allocation.readers.is_some())
    }

    /// Consume only completed readback, then submit the next bounded snapshot.
    pub(crate) fn reap(&mut self) -> Result<()> {
        self.backing()?;
        if self.failed {
            return Err(Error::Resource("VMM pool device completion is unknown"));
        }
        let result = self.poll();
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    #[expect(
        clippy::expect_used,
        reason = "each snapshot retains its retired ranges and pinned destination"
    )]
    fn poll(&mut self) -> Result<()> {
        if !self.observed.is_empty() {
            if !self.completion.query()? {
                return Ok(());
            }
            let mut index = 0;
            for offset in std::mem::take(&mut self.observed) {
                let allocation = &self.allocations[&offset];
                let readers = allocation.readers.expect("snapshot owns retired chunk");
                if readers == 0 {
                    self.remove(offset);
                    continue;
                }
                let address = self.readback[self.readback_slot].address() as usize;
                // SAFETY: the completed snapshot contains this chunk's full
                // header, and no DMA writes this pinned allocation until the
                // next snapshot is submitted below.
                let words = unsafe {
                    std::slice::from_raw_parts(
                        (address + index * HEADER_BYTES) as *const u32,
                        ACK_SLOTS,
                    )
                };
                if words
                    .iter()
                    .enumerate()
                    .all(|(slot, word)| readers & (1 << slot) == 0 || *word != CLAIMED)
                {
                    self.remove(offset);
                }
                index += 1;
            }
        }

        let observed: Vec<_> = self
            .allocations
            .iter()
            .filter_map(|(&offset, allocation)| allocation.readers.map(|_| offset))
            .collect();
        if observed.is_empty() {
            return Ok(());
        }
        let watched: Vec<_> = observed
            .iter()
            .copied()
            .filter(|offset| self.allocations[offset].readers != Some(0))
            .collect();
        let bytes = watched.len() * HEADER_BYTES;
        let destination = if bytes == 0 {
            0
        } else {
            // Freeing pinned memory can wait for unrelated device work.
            // Reuse buffers by capacity and release the cache only at close;
            // geometric growth bounds its total size by twice its largest buffer.
            self.readback_slot = match self
                .readback
                .iter()
                .position(|buffer| buffer.len() >= bytes)
            {
                Some(slot) => slot,
                None => {
                    self.readback
                        .push(PinnedBuffer::new(self.device, bytes.next_power_of_two())?);
                    self.readback.len() - 1
                }
            };
            self.readback[self.readback_slot].address()
        };
        let mut copies = CopyBatch::new(SourceOrder::Stream);
        for (index, offset) in watched.iter().enumerate() {
            copies.push(
                destination + (index * HEADER_BYTES) as u64,
                (self.address + offset) as u64,
                HEADER_BYTES,
            );
        }
        // Retain the observed ranges even if a copy or fence submission fails.
        // Shutdown drains this stream before either mapping can be destroyed.
        self.observed = observed;
        // SAFETY: the observed headers stay mapped and the pinned destination
        // stays allocated until the completion event below has been observed.
        unsafe { copies.submit(&self.stream) }?;
        self.completion.record(&self.stream)?;
        Ok(())
    }

    fn remove(&mut self, offset: usize) {
        self.allocations.remove(&offset);
        if self.allocations.is_empty() {
            self.watermark = 0;
        }
    }

    /// The caller ends all local and remote uses before closing the pool.
    /// Drain owned readback and return numerical backing outside owner locks.
    pub(crate) fn close(&mut self) -> Result<Option<Arc<Storage>>> {
        if self.backing.is_none() {
            return Ok(None);
        }
        // A default-stream initializer resolves against the current context.
        let _device = DeviceGuard::new(self.device)?;
        for stream in &self.initializers {
            stream.synchronize()?;
        }
        self.initializers.clear();
        self.stream.synchronize()?;
        self.observed.clear();
        self.readback.clear();
        self.allocations.clear();
        Ok(self.backing.take())
    }
}

impl Drop for VmmPool {
    fn drop(&mut self) {
        // After a failed close the device completion is unknown: both DMA
        // mappings stay until the process exits.
        if self.close().is_err() {
            crate::leak(self.backing.take());
            crate::leak(std::mem::take(&mut self.readback));
        }
    }
}
