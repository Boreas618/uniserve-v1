//! Host staging rings retained until their asynchronous copies complete.

use std::sync::Arc;

use uniserve_cuda::{Event, EventFlags, Stream};

use super::memory::Memory;
use crate::{Error, Result};

/// A fixed ring of host copy sources: page-locked for a CUDA device,
/// pageable for a CPU lane.
///
/// The owner fills a slot and enqueues its copy, then records the slot on
/// the copy stream before acquiring it again. Acquiring a slot waits only
/// for that slot's previous copy; unrelated device work does not prevent
/// reuse. Slots are round-robin, not individually leased.
pub struct HostBuffers {
    buffers: Vec<Arc<Memory>>,
    events: Vec<Event>,
    device: Option<i32>,
    cursor: usize,
}

impl HostBuffers {
    /// `depth` slots of `bytes` each, pinned for `device`'s copies.
    pub fn allocate(device: Option<i32>, bytes: usize, depth: usize) -> Result<Self> {
        if depth == 0 {
            return Err(Error::Invalid("host buffer depth must be positive".into()));
        }
        let buffers = (0..depth)
            .map(|_| match device {
                Some(device) => Memory::pinned(device, bytes.max(1)),
                None => Ok(Memory::host(bytes.max(1))),
            })
            .collect::<Result<Vec<_>>>()?;
        let events = match device {
            Some(_) => (0..depth)
                .map(|_| Event::new(EventFlags::default()))
                .collect(),
            None => Vec::new(),
        };
        Ok(Self {
            buffers,
            events,
            device,
            cursor: 0,
        })
    }

    /// Host bytes of a ring.
    pub fn bytes(bytes: usize, depth: usize) -> u64 {
        (bytes.max(1) * depth) as u64
    }

    pub fn device(&self) -> Option<i32> {
        self.device
    }

    /// Wait for the next slot's last copy, then lend its buffer.
    pub fn acquire(&mut self) -> Result<(usize, &Arc<Memory>)> {
        if self.buffers.is_empty() {
            return Err(Error::State("host buffers are closed"));
        }

        let slot = self.cursor;
        if let Some(event) = self.events.get(slot)
            && !event.query()?
        {
            event.synchronize()?;
        }

        self.cursor = (slot + 1) % self.buffers.len();
        Ok((slot, &self.buffers[slot]))
    }

    /// Fence the copy just submitted from `slot` on `stream`. Host rings
    /// need no fence: their copies complete before the call returns.
    #[expect(
        clippy::mem_forget,
        reason = "failed copy fencing cannot release its source allocations"
    )]
    pub fn record_copy(&mut self, slot: usize, stream: &Stream) -> Result<()> {
        if self.device.is_none() {
            return Ok(());
        }
        let event = self
            .events
            .get(slot)
            .ok_or_else(|| Error::Invalid("host buffer slot is outside its ring".into()))?;
        let result = event.record(stream);
        if result.is_err() {
            // The copy was submitted before this fence. If recording fails,
            // its sources must neither be reused nor released as completed.
            std::mem::forget(std::mem::take(&mut self.buffers));
            self.events.clear();
        }
        result.map_err(Error::Cuda)
    }

    /// Drain copies and release the ring.
    pub fn close(&mut self) -> Result<()> {
        for event in &self.events {
            if !event.query()? {
                event.synchronize()?;
            }
        }
        self.events.clear();
        self.buffers.clear();
        Ok(())
    }
}

impl Drop for HostBuffers {
    #[expect(
        clippy::mem_forget,
        reason = "unknown device completion must retain its source allocations"
    )]
    fn drop(&mut self) {
        if self.close().is_err() {
            // A failed CUDA wait cannot establish that the source is reusable.
            std::mem::forget(std::mem::take(&mut self.buffers));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::Backing;

    #[test]
    fn host_rings_cycle_their_slots() -> Result<()> {
        let mut ring = HostBuffers::allocate(None, 16, 2)?;
        let (first, memory) = ring.acquire()?;
        let first_address = memory.address();
        let (second, _) = ring.acquire()?;
        let (third, memory) = ring.acquire()?;
        assert_eq!((first, second, third), (0, 1, 0));
        assert_eq!(memory.address(), first_address);
        assert!(HostBuffers::allocate(None, 16, 0).is_err());
        Ok(())
    }
}
