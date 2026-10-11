//! `Issue`: where a function issues the work of a stream.
//!
//! A lane's execution thread is the only issuer of work on its stream. The
//! owners of request storage decide that work: which copies, fills and
//! fences a batch needs, in which order. They take what they issue to as an
//! [`Issue`], so the same function serves a caller that holds the stream
//! and one that does not. A [`Stream`] (`&Stream`, or `Option<&Stream>`
//! where host storage has no stream) issues at once, on the calling thread.
//!
//! Host bookkeeping happens when the function is called, whatever it issues
//! to. When the device sees the work is the issuer's matter, so a function
//! generic over [`Issue`] never reads back what it issued.

use std::sync::Arc;

use uniserve_cuda::{CopyBatch, Event, Stream, StridedCopy};

use super::EventPool;
use crate::ffi::DeferredCall;
use crate::transfer::copy_on_host;
use crate::{Error, Result};

/// A fill of `count` elements of `width` bytes (1, 2 or 4) with the low
/// bytes of `value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fill {
    pub address: u64,
    pub value: u32,
    pub width: usize,
    pub count: usize,
}

/// What a function issues stream work to. See the module documentation.
pub trait Issue: Copy {
    /// What issues writes of host memory in this issuer's order.
    type Host: Issue;

    /// The stream the work is issued on; `None` for host storage, whose
    /// work is plain memory writes.
    fn stream(&self) -> Option<&Stream>;

    /// The issuer of writes between host ranges, which no stream carries:
    /// they are made where this issuer's work is issued, in its order.
    fn host(&self) -> Self::Host;

    /// Copies unordered among themselves, after the work issued before.
    ///
    /// # Safety
    /// Every destination stays valid until the copies ran. A source of a
    /// [`uniserve_cuda::SourceOrder::DuringCall`] batch is host memory
    /// readable during this call, and need not outlive it; a source of a
    /// [`uniserve_cuda::SourceOrder::Stream`] batch stays valid until the
    /// copies ran.
    unsafe fn copy(&self, copies: &CopyBatch) -> Result<()>;

    /// The copies of a strided plan, in stream order of their sources.
    ///
    /// # Safety
    /// Both operands stay valid until the copies ran.
    unsafe fn copy_strided(&self, copies: &StridedCopy) -> Result<()>;

    /// # Safety
    /// The destination holds `fill.count` aligned elements and stays valid
    /// until the fill ran.
    unsafe fn fill(&self, fill: Fill) -> Result<()>;

    /// Record `event` after the work issued before it.
    fn record(&self, event: &Arc<Event>) -> Result<()>;

    /// Order the work issued after this behind `event`'s latest record.
    fn wait(&self, event: &Arc<Event>) -> Result<()>;

    /// Fire the completion wake of `events` once `event` completes.
    /// `event` is recorded by this issuer or was recorded before, so the
    /// wake is ordered behind that record wherever it is made.
    fn wake(&self, events: &EventPool, event: &Arc<Event>) -> Result<()>;

    /// A Python call whose device work belongs to the stream, in order.
    fn call(&self, call: DeferredCall) -> Result<()>;
}

/// Copy the entries of `copies` between host ranges.
///
/// # Safety
/// Every range is valid host memory and no destination overlaps a source.
unsafe fn copy_host(copies: &CopyBatch) {
    for (destination, source, bytes) in copies.iter() {
        // SAFETY: forwarded from the caller.
        unsafe {
            std::ptr::copy_nonoverlapping(source as *const u8, destination as *mut u8, bytes);
        }
    }
}

/// Write `fill` into host memory.
///
/// # Safety
/// The destination is valid host memory of `fill.count` elements.
unsafe fn fill_host(fill: Fill) {
    let bytes = fill.value.to_le_bytes();
    for index in 0..fill.count {
        // SAFETY: forwarded from the caller.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (fill.address + (index * fill.width) as u64) as *mut u8,
                fill.width,
            );
        }
    }
}

/// Issue `fill` on `stream`.
///
/// # Safety
/// As for [`Issue::fill`].
unsafe fn fill_device(fill: Fill, stream: &Stream) -> Result<()> {
    let Fill {
        address,
        value,
        width,
        count,
    } = fill;
    // SAFETY: forwarded from the caller.
    unsafe {
        match width {
            1 => uniserve_cuda::memset_d8(address, value as u8, count, stream)?,
            2 => uniserve_cuda::memset_d16(address, value as u16, count, stream)?,
            4 => uniserve_cuda::memset_d32(address, value, count, stream)?,
            _ => {
                return Err(Error::Invariant(format!(
                    "a fill writes elements of 1, 2 or 4 bytes, not {width}"
                )));
            }
        }
    }
    Ok(())
}

/// Issue at once: on the stream, or as host memory writes without one.
impl Issue for Option<&Stream> {
    type Host = Self;

    fn stream(&self) -> Option<&Stream> {
        *self
    }

    fn host(&self) -> Self {
        None
    }

    unsafe fn copy(&self, copies: &CopyBatch) -> Result<()> {
        match self {
            // SAFETY: forwarded from the caller.
            Some(stream) => unsafe { copies.submit(stream) }?,
            // SAFETY: forwarded from the caller; without a stream every
            // range is host memory.
            None => unsafe { copy_host(copies) },
        }
        Ok(())
    }

    unsafe fn copy_strided(&self, copies: &StridedCopy) -> Result<()> {
        match self {
            // SAFETY: forwarded from the caller.
            Some(stream) => unsafe { copies.submit(stream) }?,
            // SAFETY: forwarded from the caller.
            None => unsafe { copy_on_host(copies) },
        }
        Ok(())
    }

    unsafe fn fill(&self, fill: Fill) -> Result<()> {
        match self {
            // SAFETY: forwarded from the caller.
            Some(stream) => unsafe { fill_device(fill, stream) },
            None => {
                // SAFETY: forwarded from the caller.
                unsafe { fill_host(fill) };
                Ok(())
            }
        }
    }

    fn record(&self, event: &Arc<Event>) -> Result<()> {
        let stream = self.ok_or(Error::State("an event is recorded on a stream"))?;
        Ok(event.record(stream)?)
    }

    fn wait(&self, event: &Arc<Event>) -> Result<()> {
        let stream = self.ok_or(Error::State("an event is waited on by a stream"))?;
        Ok(stream.wait_event(event)?)
    }

    fn wake(&self, events: &EventPool, event: &Arc<Event>) -> Result<()> {
        let Some((notification, wake)) = events.notification(event)? else {
            return Ok(());
        };
        notification.wait_event(event)?;
        // The host function only signals the service's wake: it never calls
        // CUDA or blocks, as host functions must not.
        notification.launch_host_fn(move || wake.wake())?;
        Ok(())
    }

    fn call(&self, call: DeferredCall) -> Result<()> {
        call.invoke()
    }
}

/// Issue at once on the stream.
impl<'a> Issue for &'a Stream {
    type Host = Option<&'a Stream>;

    fn stream(&self) -> Option<&Stream> {
        Some(self)
    }

    fn host(&self) -> Self::Host {
        None
    }

    unsafe fn copy(&self, copies: &CopyBatch) -> Result<()> {
        // SAFETY: forwarded from the caller.
        unsafe { Some(*self).copy(copies) }
    }

    unsafe fn copy_strided(&self, copies: &StridedCopy) -> Result<()> {
        // SAFETY: forwarded from the caller.
        unsafe { Some(*self).copy_strided(copies) }
    }

    unsafe fn fill(&self, fill: Fill) -> Result<()> {
        // SAFETY: forwarded from the caller.
        unsafe { Some(*self).fill(fill) }
    }

    fn record(&self, event: &Arc<Event>) -> Result<()> {
        Some(*self).record(event)
    }

    fn wait(&self, event: &Arc<Event>) -> Result<()> {
        Some(*self).wait(event)
    }

    fn wake(&self, events: &EventPool, event: &Arc<Event>) -> Result<()> {
        Some(*self).wake(events, event)
    }

    fn call(&self, call: DeferredCall) -> Result<()> {
        Some(*self).call(call)
    }
}
