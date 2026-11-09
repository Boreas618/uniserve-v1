//! Streams, capture status and host functions.

use std::sync::Arc;

use crate::allocator::DevicePtr;
use crate::driver::{ContextScope, Handle, driver, in_context};
use crate::event::Event;
use crate::{DeviceId, Error, PrimaryContext, Result};

pub(crate) const CU_STREAM_NON_BLOCKING: u32 = 1;
const CU_EVENT_WAIT_EXTERNAL: u32 = 1;

/// What keeps a stream's (and its events') CUDA context alive.
#[derive(Clone)]
pub(crate) enum Retention {
    Primary { _context: PrimaryContext },
}

/// The capture state of a stream (`CUstreamCaptureStatus`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureStatus {
    None,
    Active,
    /// Capture began but was invalidated; `cuStreamEndCapture` must still run.
    Invalidated,
}

/// The condition that releases a stream's wait on a 32-bit flag
/// (`CUstreamWaitValue_flags`).
///
/// The remaining operations the driver offers are left out on purpose:
/// `CU_STREAM_WAIT_VALUE_NOR` needs a device capability query and is
/// deprecated in favour of the memory-operation API, and
/// `CU_STREAM_WAIT_VALUE_FLUSH` is for remote writes over a fabric, which this
/// crate does not carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitValue {
    /// Releases when `(int32_t)(flag - value) >= 0`
    /// (`CU_STREAM_WAIT_VALUE_GEQ`): a cyclic comparison that ignores
    /// wraparound, so a flag that passes the value and wraps still releases the
    /// wait rather than blocking it forever.
    AtLeast,
    /// Releases when the flag equals `value` (`CU_STREAM_WAIT_VALUE_EQ`).
    Equal,
    /// Releases when any bit of `value` is set in the flag
    /// (`CU_STREAM_WAIT_VALUE_AND`).
    AnyBitSet,
}

/// `CU_STREAM_WAIT_VALUE_GEQ`.
pub(crate) const CU_STREAM_WAIT_VALUE_GEQ: u32 = 0x0;
/// `CU_STREAM_WAIT_VALUE_EQ`.
pub(crate) const CU_STREAM_WAIT_VALUE_EQ: u32 = 0x1;
/// `CU_STREAM_WAIT_VALUE_AND`.
pub(crate) const CU_STREAM_WAIT_VALUE_AND: u32 = 0x2;

/// A handle to a CUDA stream this process created, or to one borrowed from
/// another owner.
///
/// Cloning a handle shares the stream: every clone submits to the same
/// driver stream, and a stream this process created is destroyed when its
/// last handle drops. A component that submits to a stream another
/// component created therefore holds a clone, which keeps the stream and its
/// context alive for as long as it is used.
///
/// Created streams are non-blocking with respect to the legacy NULL stream
/// and retain the primary context they belong to. Dropping the last handle does not wait: `cuStreamDestroy`
/// returns immediately and the driver releases the stream once its queued
/// work completes.
#[derive(Clone)]
pub struct Stream {
    raw: Handle,
    device: DeviceId,
    context: Handle,
    retention: Retention,
    /// Destroys a created stream with its last handle; `None` for a stream
    /// borrowed from another owner.
    _created: Option<Arc<Created>>,
}

// SAFETY: the driver accepts stream handles from any host thread. Shared
// references only submit work or query state; destruction happens once, when
// the last handle drops.
unsafe impl Send for Stream {}
// SAFETY: as for Send above; methods of a shared reference pass the immutable
// handles only to thread-safe driver calls.
unsafe impl Sync for Stream {}

/// A driver stream this process created, destroyed on drop.
struct Created {
    raw: Handle,
    /// Keeps the stream's context alive until the stream is destroyed.
    _retention: Retention,
}

// SAFETY: as for `Stream`; the driver accepts `cuStreamDestroy` from any host
// thread.
unsafe impl Send for Created {}
// SAFETY: as for Send above; nothing reads the handle through a shared
// reference, and only Drop passes it to the driver.
unsafe impl Sync for Created {}

impl Created {
    fn new(raw: Handle, retention: &Retention) -> Option<Arc<Self>> {
        Some(Arc::new(Self {
            raw,
            _retention: retention.clone(),
        }))
    }
}

impl Drop for Created {
    fn drop(&mut self) {
        if let Ok(driver) = driver() {
            // SAFETY: the stream was created by this process and is destroyed
            // once. The driver defers physical release until its queued work
            // completes.
            let _ = unsafe { (driver.stream_destroy)(self.raw) };
        }
    }
}

impl Stream {
    /// Creates a non-blocking stream of default priority in `device`'s
    /// primary context.
    pub fn new(device: DeviceId) -> Result<Self> {
        Self::with_priority(device, 0)
    }

    /// Creates a non-blocking stream with a CUDA priority (lower is higher
    /// priority; the driver clamps values outside the device's range).
    pub fn with_priority(device: DeviceId, priority: i32) -> Result<Self> {
        let context = PrimaryContext::retain(device)?;
        let mut raw = std::ptr::null_mut();
        in_context(context.raw(), |driver| {
            // SAFETY: the driver writes a stream handle in the current context.
            driver.check(
                unsafe {
                    (driver.stream_create_with_priority)(&mut raw, CU_STREAM_NON_BLOCKING, priority)
                },
                "cuStreamCreateWithPriority",
            )
        })?;

        let context_handle = context.raw();
        let retention = Retention::Primary { _context: context };
        Ok(Self {
            raw,
            device,
            context: context_handle,
            _created: Created::new(raw, &retention),
            retention,
        })
    }

    /// Creates an independent owned stream in this stream's context with the
    /// same priority. It retains whatever this stream retains; forking a
    /// borrowed stream inherits the borrow's contract.
    pub fn fork(&self) -> Result<Self> {
        let priority = self.priority()?;
        let mut raw = std::ptr::null_mut();
        in_context(self.context, |driver| {
            // SAFETY: the driver writes a stream handle in self's context.
            driver.check(
                unsafe {
                    (driver.stream_create_with_priority)(&mut raw, CU_STREAM_NON_BLOCKING, priority)
                },
                "cuStreamCreateWithPriority",
            )
        })?;

        Ok(Self {
            raw,
            device: self.device,
            context: self.context,
            _created: Created::new(raw, &self.retention),
            retention: self.retention.clone(),
        })
    }

    /// The `CUstream` handle, as passed to `torch.cuda.ExternalStream`.
    pub fn raw(&self) -> usize {
        self.raw as usize
    }

    pub(crate) fn handle(&self) -> Handle {
        self.raw
    }

    pub(crate) fn context(&self) -> Handle {
        self.context
    }

    /// Makes the stream's context current on the calling thread. Copies and
    /// fills need it to resolve unified addresses (VMM mappings in
    /// particular), and batch copies fail without one.
    pub(crate) fn enter(&self) -> Result<ContextScope> {
        ContextScope::enter(self.context)
    }

    pub(crate) fn retention(&self) -> Retention {
        self.retention.clone()
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn priority(&self) -> Result<i32> {
        let driver = driver()?;
        let mut priority = 0;
        driver.check(
            // SAFETY: self is a live stream; the driver writes one int to the local
            // `priority`.
            unsafe { (driver.stream_get_priority)(self.raw, &mut priority) },
            "cuStreamGetPriority",
        )?;
        Ok(priority)
    }

    pub fn capture_status(&self) -> Result<CaptureStatus> {
        let driver = driver()?;
        let mut status = 0;
        driver.check(
            // SAFETY: self is a live stream; the driver writes one enum value to
            // the local `status`.
            unsafe { (driver.stream_is_capturing)(self.raw, &mut status) },
            "cuStreamIsCapturing",
        )?;

        Ok(match status {
            0 => CaptureStatus::None,
            1 => CaptureStatus::Active,
            _ => CaptureStatus::Invalidated,
        })
    }

    /// Fails with [`Error::Capturing`] while the stream is being captured.
    pub(crate) fn require_eager(&self, call: &'static str) -> Result<()> {
        match self.capture_status()? {
            CaptureStatus::None => Ok(()),
            _ => Err(Error::Capturing { call }),
        }
    }

    /// Blocks the calling thread until all work queued on the stream completes.
    pub fn synchronize(&self) -> Result<()> {
        let driver = driver()?;
        // SAFETY: self is a live stream.
        driver.check(
            unsafe { (driver.stream_synchronize)(self.raw) },
            "cuStreamSynchronize",
        )
    }

    /// Orders later work on this stream after the event's most recent record.
    /// An event that was never recorded imposes no dependency.
    ///
    /// For an event created with [`crate::EventFlags::external`], a wait issued
    /// during capture becomes an explicit graph node.
    pub fn wait_event(&self, event: &Event) -> Result<()> {
        let Some(handle) = event.current() else {
            return Ok(());
        };
        let flags = if event.flags().external && self.capture_status()? != CaptureStatus::None {
            CU_EVENT_WAIT_EXTERNAL
        } else {
            0
        };

        let driver = driver()?;
        // SAFETY: both handles are live; the event Arc is held for the call.
        driver.check(
            unsafe { (driver.stream_wait_event)(self.raw, handle.raw(), flags) },
            "cuStreamWaitEvent",
        )
    }

    /// Writes `value` into the 32-bit flag at `address` in stream order
    /// (`cuStreamWriteValue32`).
    ///
    /// This is how one stream tells another that the work before this point is
    /// done, without a host round trip. The write fences before it — the
    /// driver's `__threadfence_system` equivalent, scoped to the stream — so a
    /// stream that waits on the flag observes every write this stream issued
    /// before it. There is no weaker mode to ask for: the driver's
    /// `CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER` is not supported by the
    /// entry point this crate resolves (`cuda.h`, `CUstreamWriteValue_flags`).
    ///
    /// `address` is device memory of this stream's device or host memory mapped
    /// into it, and it must hold a 32-bit aligned value that stays valid until
    /// the write runs — for a captured graph, until every replay does.
    pub fn write_value32(&self, address: DevicePtr, value: u32) -> Result<()> {
        let driver = driver()?;
        // SAFETY: the caller keeps the flag valid until the write runs, and the
        // stream owns the ordering of its own work.
        driver.check(
            unsafe { (driver.stream_write_value32)(self.raw, address.address(), value, 0) },
            "cuStreamWriteValue32",
        )
    }

    /// Holds later work on this stream until the 32-bit flag at `address`
    /// satisfies `kind` against `value` (`cuStreamWaitValue32`).
    ///
    /// Work ordered after this call does not run, and its reads do not observe
    /// the flag's writer's writes, until the condition holds. A waiting stream
    /// blocks on the device: nothing on the host polls, and the flag may be
    /// written by a kernel or by another stream's [`Stream::write_value32`].
    ///
    /// `address` must hold a 32-bit aligned value that stays valid until the
    /// wait is satisfied; a wait whose flag is never written blocks the stream
    /// for as long as the stream lives.
    ///
    /// A wait captured into a graph must have its flag written by something
    /// outside that graph's own replay: a graph whose only writer is a node of
    /// the same graph blocks on its first replay, and every replay after it.
    pub fn wait_value32(&self, address: DevicePtr, value: u32, kind: WaitValue) -> Result<()> {
        let flags = match kind {
            WaitValue::AtLeast => CU_STREAM_WAIT_VALUE_GEQ,
            WaitValue::Equal => CU_STREAM_WAIT_VALUE_EQ,
            WaitValue::AnyBitSet => CU_STREAM_WAIT_VALUE_AND,
        };

        let driver = driver()?;
        // SAFETY: the caller keeps the flag valid until the wait is satisfied.
        driver.check(
            unsafe { (driver.stream_wait_value32)(self.raw, address.address(), value, flags) },
            "cuStreamWaitValue32",
        )
    }

    /// Runs `action` on a driver thread after all preceding work on the stream
    /// completes, and before any later work starts.
    ///
    /// The action must not call CUDA and must not block on anything that waits
    /// for this stream: the driver serializes host
    /// functions, and either can deadlock outstanding device work. A host
    /// function captured into a graph would run once per replay, so this
    /// rejects capturing streams. If submission fails, the action is dropped
    /// without running.
    pub fn launch_host_fn<F>(&self, action: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        unsafe extern "C" fn invoke<F: FnOnce()>(data: Handle) {
            // SAFETY: launch_host_fn transferred exactly one boxed action to
            // this single invocation.
            let action = unsafe { Box::from_raw(data.cast::<F>()) };
            action();
        }

        self.require_eager("cuLaunchHostFunc")?;
        let driver = driver()?;
        let data = Box::into_raw(Box::new(action));

        // SAFETY: the stream is live. On success the driver owns `data` and
        // `invoke` consumes it exactly once after preceding work completes.
        let status = unsafe { (driver.launch_host_func)(self.raw, invoke::<F>, data.cast()) };
        if status != crate::driver::CUDA_SUCCESS {
            // SAFETY: a rejected launch did not transfer ownership.
            drop(unsafe { Box::from_raw(data) });
        }
        driver.check(status, "cuLaunchHostFunc")
    }
}
