//! CUDA events: completion fences, timing and interprocess handles.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use crate::driver::{CUDA_ERROR_NOT_READY, Handle, RawIpcEventHandle, driver, in_context};
use crate::stream::{CaptureStatus, Retention};
use crate::{DeviceId, Error, PrimaryContext, Result, Stream};

const CU_EVENT_DISABLE_TIMING: u32 = 2;
const CU_EVENT_INTERPROCESS: u32 = 4;
const CU_EVENT_RECORD_EXTERNAL: u32 = 1;

/// Creation flags of an [`Event`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EventFlags {
    /// Records timestamps for [`Event::elapsed_ms`]. Timing events cost more
    /// to record and cannot be interprocess.
    pub timing: bool,
    /// The event can be exported with [`Event::ipc_handle`].
    pub interprocess: bool,
    /// During stream capture, records and waits become explicit graph nodes
    /// (`CU_EVENT_RECORD_EXTERNAL` / `CU_EVENT_WAIT_EXTERNAL`), so the
    /// dependency also holds across separately captured graph segments.
    /// Outside capture the same handle uses ordinary flags, as CUDA rejects the
    /// external flags there.
    pub external: bool,
}

/// The 64-byte interprocess handle of an event (`CUipcEventHandle`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IpcEventHandle(pub [u8; 64]);

/// One driver event and the context it was created in.
pub(crate) struct EventHandle {
    raw: Handle,
    context: Handle,
    // Keeps `context` alive through destruction of `raw`.
    _retention: Retention,
}

// SAFETY: CUDA event handles may be used from any host thread. The handle is
// shared through Arc so every driver call holds it alive; only Drop destroys it.
unsafe impl Send for EventHandle {}
unsafe impl Sync for EventHandle {}

impl EventHandle {
    pub(crate) fn raw(&self) -> Handle {
        self.raw
    }
}

impl Drop for EventHandle {
    fn drop(&mut self) {
        if let Ok(driver) = driver() {
            // SAFETY: owned and destroyed once. The driver defers physical
            // destruction while recorded work is pending.
            let _ = unsafe { (driver.event_destroy)(self.raw) };
        }
    }
}

/// A CUDA event that is created lazily, in the context of the stream it is
/// first recorded on.
///
/// An event belongs to one CUDA context, and the worker records the same
/// logical fence on streams of different contexts (primary or SM partition).
/// Recording on a stream of another context replaces the driver event with a
/// new one in that context; waits and queries always observe the most recent
/// record. The event retains the context it was created in, so a fence may
/// outlive the stream that recorded it.
///
/// Queries, waits and drop do not change the calling thread's context.
///
/// A thread that hands a record to another thread announces it first
/// ([`Event::queue_record`]): the work the record will capture is not issued
/// yet, so the event is not complete until that record is made.
pub struct Event {
    flags: EventFlags,
    current: Mutex<Option<Arc<EventHandle>>>,
    /// Records announced and not yet made.
    queued: AtomicUsize,
}

impl Event {
    pub fn new(flags: EventFlags) -> Self {
        Self {
            flags,
            current: Mutex::new(None),
            queued: AtomicUsize::new(0),
        }
    }

    /// Announce one record of this event that another thread will make.
    /// Until [`Self::record`] makes it, [`Self::query`] reports the event
    /// incomplete, whatever its earlier records captured.
    pub fn queue_record(&self) {
        self.queued.fetch_add(1, Ordering::AcqRel);
    }

    pub fn flags(&self) -> EventFlags {
        self.flags
    }

    pub(crate) fn current(&self) -> Option<Arc<EventHandle>> {
        self.current
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn driver_flags(&self) -> u32 {
        let mut flags = 0;
        if !self.flags.timing {
            flags |= CU_EVENT_DISABLE_TIMING;
        }
        if self.flags.interprocess {
            flags |= CU_EVENT_INTERPROCESS;
        }
        flags
    }

    /// Captures the work queued on `stream` so far. This is the oldest
    /// announced record, if one is.
    pub fn record(&self, stream: &Stream) -> Result<()> {
        // An unannounced record leaves the count at zero.
        let _ = self
            .queued
            .try_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                queued.checked_sub(1)
            });

        let driver = driver()?;
        let record_flags = if self.flags.external && stream.capture_status()? != CaptureStatus::None
        {
            CU_EVENT_RECORD_EXTERNAL
        } else {
            0
        };

        let mut current = self.current.lock().unwrap_or_else(PoisonError::into_inner);
        let handle = match &mut *current {
            Some(handle) if handle.context == stream.context() => handle,
            slot => {
                let mut raw = std::ptr::null_mut();
                let flags = self.driver_flags();
                in_context(stream.context(), |driver| {
                    // SAFETY: the driver writes an event in the current context.
                    driver.check(
                        unsafe { (driver.event_create)(&mut raw, flags) },
                        "cuEventCreate",
                    )
                })?;
                slot.insert(Arc::new(EventHandle {
                    raw,
                    context: stream.context(),
                    _retention: stream.retention(),
                }))
            }
        };

        // SAFETY: both the event and the stream are live and share a context.
        driver.check(
            unsafe { (driver.event_record_with_flags)(handle.raw, stream.handle(), record_flags) },
            "cuEventRecordWithFlags",
        )
    }

    /// Whether all work captured by the most recent record has completed.
    /// An event that was never recorded is complete, and an event with an
    /// announced record still to make is not. Device faults are errors.
    pub fn query(&self) -> Result<bool> {
        if self.queued.load(Ordering::Acquire) > 0 {
            return Ok(false);
        }
        let Some(handle) = self.current() else {
            return Ok(true);
        };
        let driver = driver()?;
        // SAFETY: the handle is live while the Arc is held.
        let status = unsafe { (driver.event_query)(handle.raw) };
        if status == CUDA_ERROR_NOT_READY {
            return Ok(false);
        }
        driver.check(status, "cuEventQuery")?;
        Ok(true)
    }

    /// Blocks the calling thread until the most recent record completes.
    pub fn synchronize(&self) -> Result<()> {
        let Some(handle) = self.current() else {
            return Ok(());
        };
        let driver = driver()?;
        // SAFETY: the handle is live while the Arc is held.
        driver.check(
            unsafe { (driver.event_synchronize)(handle.raw) },
            "cuEventSynchronize",
        )
    }

    /// Milliseconds between this event's and `end`'s most recent records.
    /// Both must be timing events recorded in the same context; this does not
    /// wait, so it fails with `CUDA_ERROR_NOT_READY` until both completed.
    pub fn elapsed_ms(&self, end: &Event) -> Result<f32> {
        let start = self.current().ok_or(Error::NotRecorded)?;
        let end = end.current().ok_or(Error::NotRecorded)?;
        let driver = driver()?;
        let mut milliseconds = 0.0;
        // SAFETY: both handles are live while the Arcs are held.
        driver.check(
            unsafe { (driver.event_elapsed_time)(&mut milliseconds, start.raw, end.raw) },
            "cuEventElapsedTime_v2",
        )?;
        Ok(milliseconds)
    }

    /// Exports the current driver event of an interprocess event. A later
    /// record on a stream of another context creates a different driver event,
    /// so exporters record on one context only.
    pub fn ipc_handle(&self) -> Result<IpcEventHandle> {
        let handle = self.current().ok_or(Error::NotRecorded)?;
        let mut exported = RawIpcEventHandle { reserved: [0; 64] };
        in_context(handle.context, |driver| {
            // SAFETY: the handle is live; the driver writes 64 bytes.
            driver.check(
                unsafe { (driver.ipc_get_event_handle)(&mut exported, handle.raw) },
                "cuIpcGetEventHandle",
            )
        })?;
        Ok(IpcEventHandle(exported.reserved))
    }

    /// Opens another process's interprocess event in `device`'s primary
    /// context, which the imported event retains.
    ///
    /// The imported event can be queried, synchronized and waited on; recording
    /// it on a stream of a different context replaces it with a local event.
    pub fn open_ipc(device: DeviceId, handle: &IpcEventHandle) -> Result<Self> {
        let context = PrimaryContext::retain(device)?;
        let mut raw = std::ptr::null_mut();
        in_context(context.raw(), |driver| {
            // SAFETY: the driver writes an event handle in the current context.
            let status = unsafe {
                (driver.ipc_open_event_handle)(&mut raw, RawIpcEventHandle { reserved: handle.0 })
            };
            driver.check(status, "cuIpcOpenEventHandle")
        })?;

        Ok(Self {
            flags: EventFlags {
                timing: false,
                interprocess: true,
                external: false,
            },
            current: Mutex::new(Some(Arc::new(EventHandle {
                raw,
                context: context.raw(),
                _retention: Retention::Primary { _context: context },
            }))),
            queued: AtomicUsize::new(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_event_with_an_announced_record_is_not_complete() -> Result<()> {
        let event = Event::new(EventFlags::default());
        // Never recorded: nothing to wait for.
        assert!(event.query()?);

        // A record handed to another thread captures work not issued yet.
        event.queue_record();
        assert!(!event.query()?);
        Ok(())
    }
}
