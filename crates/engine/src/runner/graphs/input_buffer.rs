//! The input buffer of one runner: where a replay's host bytes wait for their copy to the device.
//!
//! A replay reads two kinds of host bytes: the batch's columns, which fill the runner's input
//! tensors, and the host values its forward computes, which the graph's input copies read. Both
//! are copied to the device asynchronously, so their source must stay unchanged until the stream
//! has passed the copies. The buffer is therefore a ring of slots of page-locked memory: a
//! replay takes the next slot, writes it, issues its copies and records the slot's event behind
//! them; the slot is taken again only once that event has completed. With more slots than
//! replays in flight the worker thread never waits here.
//!
//! One slot is `[columns | values]`: the bytes of the runner's input tensors in the layout of
//! their device block, then the area the forward's host-to-device copies write at the offsets the
//! bucket's capture recorded.

use platform::{CopyBatch, DeviceId, Event, EventFlags, PinnedBuffer, SourceOrder, Stream};

use crate::worker::{Error, Result};
use ops::allocator::ALIGNMENT;

/// The memory behind the ring.
enum Memory {
    /// Page-locked memory of a device's context, which copies on a stream read.
    Pinned(PinnedBuffer),
    /// Ordinary host memory, for graphs that hold no device or capture beside a runner's:
    /// nothing is ever copied from it.
    Host(Vec<u8>),
}

/// A ring of slots of host memory, each holding one replay's columns and host values.
pub(crate) struct InputBuffer {
    memory: Memory,
    /// The bytes of the columns area of a slot, a multiple of [`ALIGNMENT`].
    columns: u64,
    /// The bytes of the values area of a slot, a multiple of [`ALIGNMENT`].
    values: u64,
    /// One event per slot, recorded on the stream behind the slot's copies. A slot whose event
    /// was never recorded, or has completed, is free.
    events: Vec<Event>,
    /// The slot the next replay takes.
    next: usize,
}

impl InputBuffer {
    /// A ring of `slots` slots of page-locked memory on `device`, each of `columns` bytes of
    /// input tensors and `values` bytes of host values.
    ///
    /// # Errors
    ///
    /// When `slots` is zero or the page-locked allocation fails.
    pub(crate) fn pinned(
        device: DeviceId,
        columns: u64,
        values: u64,
        slots: usize,
    ) -> Result<Self> {
        let (columns, values) = (aligned(columns), aligned(values));
        let bytes = ring_bytes(columns, values, slots)?;
        let memory = Memory::Pinned(PinnedBuffer::new(device, bytes)?);

        Ok(Self::over(memory, columns, values, slots))
    }

    /// A ring of the same layout over ordinary host memory, for graphs that hold no device.
    ///
    /// # Errors
    ///
    /// When `slots` is zero.
    pub(crate) fn host(columns: u64, values: u64, slots: usize) -> Result<Self> {
        let (columns, values) = (aligned(columns), aligned(values));
        let bytes = ring_bytes(columns, values, slots)?;

        Ok(Self::over(
            Memory::Host(vec![0; bytes]),
            columns,
            values,
            slots,
        ))
    }

    fn over(memory: Memory, columns: u64, values: u64, slots: usize) -> Self {
        Self {
            memory,
            columns,
            values,
            events: (0..slots)
                .map(|_| Event::new(EventFlags::default()))
                .collect(),
            next: 0,
        }
    }

    /// The bytes of the values area of a slot.
    pub(crate) fn values_bytes(&self) -> u64 {
        self.values
    }

    /// Whether the ring is over ordinary host memory, from which nothing is copied.
    pub(crate) fn is_host(&self) -> bool {
        matches!(self.memory, Memory::Host(_))
    }

    /// Takes the next slot of the ring for one replay and returns it.
    ///
    /// Blocks on the slot's event when the stream has not yet passed the copies of the replay
    /// that used the slot last.
    ///
    /// # Errors
    ///
    /// When the event cannot be waited for.
    pub(crate) fn acquire(&mut self) -> Result<usize> {
        let slot = self.next;
        self.next = (slot + 1) % self.events.len();

        // An event that was never recorded completes at once.
        self.events[slot].synchronize()?;
        Ok(slot)
    }

    /// The columns area of `slot`, which the runner fills with the bytes of its input tensors.
    pub(crate) fn columns_mut(&mut self, slot: usize) -> &mut [u8] {
        let start = self.slot_offset(slot);
        let end = start + self.columns as usize;

        &mut self.bytes_mut()[start..end]
    }

    /// The values area of `slot`, which a replay of fixed host values fills with the bytes its
    /// graph's capture computed.
    pub(crate) fn values_mut(&mut self, slot: usize) -> &mut [u8] {
        let start = self.slot_offset(slot) + self.columns as usize;
        let end = start + self.values as usize;

        &mut self.bytes_mut()[start..end]
    }

    /// The values area of `slot`: its first byte and its length, as a context's input buffer.
    pub(crate) fn values_ptr(&mut self, slot: usize) -> (*mut u8, u64) {
        let start = self.slot_offset(slot) + self.columns as usize;
        let values = self.values;

        (self.bytes_mut()[start..].as_mut_ptr(), values)
    }

    /// Issues the copies of the replay that filled `slot` on `stream` as one batched copy. The
    /// caller releases the slot behind everything that reads it ([`InputBuffer::release`]).
    ///
    /// `inputs` is the device address of the runner's input tensors, which receive the first
    /// `column_bytes` bytes of the slot's columns area; `copies` are the graph's input copies as
    /// `(offset in the values area, device address, bytes)`.
    ///
    /// # Errors
    ///
    /// When the ring holds no page-locked memory, a copy lies outside its area, or the driver
    /// refuses the copy or the event.
    pub(crate) fn issue(
        &mut self,
        slot: usize,
        stream: &Stream,
        inputs: u64,
        column_bytes: u64,
        copies: &[(u64, u64, u64)],
    ) -> Result<()> {
        let Memory::Pinned(memory) = &self.memory else {
            return Err(Error::State(
                "an input buffer over ordinary host memory issues no copy",
            ));
        };
        if column_bytes > self.columns {
            return Err(Error::Invariant(format!(
                "{column_bytes} bytes of input tensors exceed the {} bytes of a slot's columns",
                self.columns
            )));
        }

        let base = memory.address() + self.slot_offset(slot) as u64;
        let values = base + self.columns;

        // The destinations are disjoint: the input tensors' block, and one block per input copy
        // that the capture's allocator gave to no other tensor of the forward.
        let mut batch = CopyBatch::new(SourceOrder::Stream);
        batch.push(inputs, base, column_bytes as usize);
        for &(offset, address, bytes) in copies {
            if offset + bytes > self.values {
                return Err(Error::Invariant(format!(
                    "an input copy of {bytes} bytes at offset {offset} lies outside the {} bytes \
                     of a slot's values",
                    self.values
                )));
            }
            batch.push(address, values + offset, bytes as usize);
        }

        // SAFETY: the sources are this slot's page-locked bytes, which nothing writes until the
        // event the caller records behind them has completed (`acquire`); the destinations are the runner's
        // input tensors and blocks of its region, which its `Graphs` keeps mapped.
        unsafe { batch.submit(stream) }?;
        Ok(())
    }

    /// Records the event of `slot` on `stream`, behind every copy issued so far that reads the
    /// slot: the slot is taken again once the stream has passed it.
    ///
    /// # Errors
    ///
    /// When the driver refuses the event.
    pub(crate) fn release(&mut self, slot: usize, stream: &Stream) -> Result<()> {
        Ok(self.events[slot].record(stream)?)
    }

    /// The first byte of `slot` in the ring's memory.
    fn slot_offset(&self, slot: usize) -> usize {
        slot * (self.columns + self.values) as usize
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        match &mut self.memory {
            // SAFETY: the buffer is this ring's own, borrowed mutably with it; a slot is written
            // only after `acquire` waited for the copies that read it.
            Memory::Pinned(memory) => unsafe { memory.as_mut_slice() },
            Memory::Host(memory) => memory,
        }
    }
}

/// `bytes` rounded up to the alignment of a slot's areas.
fn aligned(bytes: u64) -> u64 {
    bytes.next_multiple_of(ALIGNMENT)
}

/// The bytes of a ring of `slots` slots; at least one alignment unit, as an allocation of no
/// byte is refused by the driver.
fn ring_bytes(columns: u64, values: u64, slots: usize) -> Result<usize> {
    if slots == 0 {
        return Err(Error::Invalid(
            "an input buffer holds at least one slot".to_owned(),
        ));
    }

    Ok(((columns + values) as usize * slots).max(ALIGNMENT as usize))
}
