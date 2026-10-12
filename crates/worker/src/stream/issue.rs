//! `Issue`: where a function issues the work of a stream.
//!
//! A lane's execution thread is the only issuer of work on its stream. The
//! owners of request storage decide that work on the control thread: which
//! copies, fills and fences a batch needs, in which order. They take what
//! they issue to as an [`Issue`]:
//!
//! - a [`Stream`] (`&Stream`, or `Option<&Stream>` where host storage has no
//!   stream) issues at once, on the calling thread;
//! - a job's recorder ([`crate::lane::Job::on`]) appends the same work to
//!   the job as operations, which the lane's execution thread performs when
//!   the job reaches it.
//!
//! Host bookkeeping is the same either way and happens when the function is
//! called. What differs is when the device sees the work, so a function
//! generic over [`Issue`] never reads back what it issued. An event recorded
//! into a job reports itself incomplete until the record is made
//! ([`uniserve_cuda::Event::queue_record`]), so storage fenced by it is not
//! returned early.

use std::sync::Arc;

use uniserve_cuda::{CopyBatch, Event, Stream, StridedCopy};

use super::{Deferred, EventPool};
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

/// A fill of `height` rows, `pitch` bytes apart, of `count` elements of
/// `width` bytes (1 or 4) each, with the low bytes of `value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowFill {
    pub address: u64,
    pub pitch: usize,
    pub value: u32,
    pub width: usize,
    pub count: usize,
    pub height: usize,
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

    /// A fill of equally spaced rows (`cuMemsetD2D8Async`,
    /// `cuMemsetD2D32Async`).
    ///
    /// # Safety
    /// Every row holds `fill.count` aligned elements in the memory of the
    /// stream's device (host memory without a stream) and stays valid until
    /// the fill ran.
    unsafe fn memset_2d(&self, fill: RowFill) -> Result<()>;

    /// Record `event` after the work issued before it.
    fn record(&self, event: &Arc<Event>) -> Result<()>;

    /// Order the work issued after this behind `event`'s latest record.
    fn wait(&self, event: &Arc<Event>) -> Result<()>;

    /// Fire the completion wake of `events` once `event` completes.
    /// `event` is recorded by this issuer or was recorded before, so the
    /// wake is ordered behind that record wherever it is made.
    fn wake(&self, events: &EventPool, event: &Arc<Event>) -> Result<()>;

    /// Run `owner` once every one of `fences` completed: the release of
    /// storage that the work they fence addressed ([`EventPool::defer`]).
    /// Each fence is recorded by this issuer or was recorded before, and
    /// one retained reference of each goes with the deferral. A stream,
    /// which made its records already, hands the deferral to `events` at
    /// once; a job keeps it until the job is submitted, so the pool never
    /// follows an event whose record is still to be made.
    fn defer(&self, events: &EventPool, fences: Vec<Arc<Event>>, owner: Deferred) -> Result<()>;

    /// A Python call whose device work belongs to the stream, in order.
    fn call(&self, call: DeferredCall) -> Result<()>;

    /// Run `function` once the work issued before it completed. It touches
    /// host memory only: it neither calls CUDA nor blocks. Without a stream
    /// it runs in the order of this issuer's host writes.
    fn launch_host_fn(&self, function: HostFunction) -> Result<()>;

    /// Wait for the work issued before this to complete. A stream waits
    /// during the call; a job waits where its execution thread performs
    /// this, so what the wait makes readable is read from the job's
    /// completion on.
    fn synchronize(&self) -> Result<()>;

    /// Issue what `record` issues to this issuer as one attempt that gives
    /// its storage back when it fails (an export). A job keeps of a failed
    /// attempt only the event operations: the events it queued are still
    /// recorded, and the work that addressed the returned storage is not
    /// performed. A stream issued the attempt's work at once and drains it
    /// where the attempt releases its storage.
    fn attempt<T>(&self, record: impl FnOnce() -> Result<T>) -> Result<T> {
        record()
    }
}

/// A function a stream runs on the host between its device work.
pub(crate) type HostFunction = Box<dyn FnOnce() + Send>;

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
    if fill.width == 1 {
        // SAFETY: forwarded from the caller.
        unsafe { std::ptr::write_bytes(fill.address as *mut u8, fill.value as u8, fill.count) };
        return;
    }

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

/// Write the rows of `fill` into host memory.
///
/// # Safety
/// Every row is valid host memory of `fill.count` elements.
unsafe fn fill_host_rows(fill: RowFill) {
    for row in 0..fill.height {
        // SAFETY: forwarded from the caller.
        unsafe {
            fill_host(Fill {
                address: fill.address + (row * fill.pitch) as u64,
                value: fill.value,
                width: fill.width,
                count: fill.count,
            });
        }
    }
}

/// Issue the rows of `fill` on `stream`.
///
/// # Safety
/// As for [`Issue::memset_2d`].
unsafe fn fill_device_rows(fill: RowFill, stream: &Stream) -> Result<()> {
    let RowFill {
        address,
        pitch,
        value,
        width,
        count,
        height,
    } = fill;
    // SAFETY: forwarded from the caller.
    unsafe {
        match width {
            1 => uniserve_cuda::memset_2d_d8(address, pitch, value as u8, count, height, stream)?,
            4 => uniserve_cuda::memset_2d_d32(address, pitch, value, count, height, stream)?,
            _ => {
                return Err(Error::Invariant(format!(
                    "a row fill writes elements of 1 or 4 bytes, not {width}"
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

    unsafe fn memset_2d(&self, fill: RowFill) -> Result<()> {
        match self {
            // SAFETY: forwarded from the caller.
            Some(stream) => unsafe { fill_device_rows(fill, stream) },
            None => {
                // SAFETY: forwarded from the caller.
                unsafe { fill_host_rows(fill) };
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

    fn defer(&self, events: &EventPool, fences: Vec<Arc<Event>>, owner: Deferred) -> Result<()> {
        events.defer(fences, owner)
    }

    fn call(&self, call: DeferredCall) -> Result<()> {
        call.invoke()
    }

    fn launch_host_fn(&self, function: HostFunction) -> Result<()> {
        match self {
            // The stream's own method, by name: `stream` is a `&&Stream`
            // here, on which a method call resolves to this trait's method
            // of `&Stream`, which issues through this implementation again.
            Some(stream) => Stream::launch_host_fn(stream, function)?,
            None => function(),
        }
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        match self {
            // The stream's own method, by name, as above.
            Some(stream) => Ok(Stream::synchronize(stream)?),
            None => Ok(()),
        }
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

    unsafe fn memset_2d(&self, fill: RowFill) -> Result<()> {
        // SAFETY: forwarded from the caller.
        unsafe { Some(*self).memset_2d(fill) }
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

    fn defer(&self, events: &EventPool, fences: Vec<Arc<Event>>, owner: Deferred) -> Result<()> {
        Some(*self).defer(events, fences, owner)
    }

    fn call(&self, call: DeferredCall) -> Result<()> {
        Some(*self).call(call)
    }

    fn launch_host_fn(&self, function: HostFunction) -> Result<()> {
        Some(*self).launch_host_fn(function)
    }

    fn synchronize(&self) -> Result<()> {
        Issue::synchronize(&Some(*self))
    }
}
