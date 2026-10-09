//! Streams, capture status and host functions.

use std::sync::Arc;

use crate::driver::{ContextScope, Handle, driver, in_context};
use crate::event::Event;
use crate::green::{self, GreenContext};
use crate::{DeviceId, Error, PrimaryContext, Result};

const CU_STREAM_NON_BLOCKING: u32 = 1;
const CU_EVENT_WAIT_EXTERNAL: u32 = 1;

/// What keeps a stream's (and its events') CUDA context alive.
#[derive(Clone)]
pub(crate) enum Retention {
    /// The creator of a borrowed stream keeps its context alive.
    Borrowed,
    Primary {
        _context: PrimaryContext,
    },
    Green(Arc<GreenContext>),
}

/// The capture state of a stream (`CUstreamCaptureStatus`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureStatus {
    None,
    Active,
    /// Capture began but was invalidated; `cuStreamEndCapture` must still run.
    Invalidated,
}

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
/// and retain the context they belong to (a primary context or an SM
/// partition). Dropping the last handle does not wait: `cuStreamDestroy`
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

    /// Borrows a stream created outside this crate (for example by PyTorch
    /// or NCCL) from its integer handle. A stream this crate created is
    /// shared by cloning its handle instead.
    ///
    /// # Safety
    /// `raw` must be a live stream on `device`, and its creator must keep the
    /// stream and its context alive for as long as the returned value, its
    /// clones, any stream forked from it, and any event recorded on it are
    /// used.
    pub unsafe fn borrowed(device: DeviceId, raw: usize) -> Result<Self> {
        let raw = raw as Handle;
        let driver = driver()?;
        let mut context = std::ptr::null_mut();
        // SAFETY: the caller guarantees the stream is live.
        driver.check(
            unsafe { (driver.stream_get_ctx)(raw, &mut context) },
            "cuStreamGetCtx",
        )?;

        Ok(Self {
            raw,
            device,
            context,
            retention: Retention::Borrowed,
            _created: None,
        })
    }

    pub(crate) fn from_green(
        device: DeviceId,
        raw: Handle,
        green: Arc<GreenContext>,
    ) -> Result<Self> {
        let driver = driver()?;
        let mut context = std::ptr::null_mut();
        // SAFETY: raw was just created in the retained green context.
        let status = unsafe { (driver.stream_get_ctx)(raw, &mut context) };
        if let Err(error) = driver.check(status, "cuStreamGetCtx") {
            // SAFETY: the new stream is owned here and has no work.
            let _ = unsafe { (driver.stream_destroy)(raw) };
            return Err(error);
        }

        let retention = Retention::Green(green);
        Ok(Self {
            raw,
            device,
            context,
            _created: Created::new(raw, &retention),
            retention,
        })
    }

    /// Divides `device` into disjoint SM partitions (green contexts) of exactly
    /// `counts[i]` SMs and returns one owned stream per partition.
    pub fn partition(device: DeviceId, counts: &[u32]) -> Result<Vec<Self>> {
        green::partition(device, counts)
    }

    /// Creates an independent owned stream in this stream's context (or SM
    /// partition) with the same priority. It retains whatever this stream
    /// retains; forking a borrowed stream inherits the borrow's contract.
    pub fn fork(&self) -> Result<Self> {
        let driver = driver()?;
        let priority = self.priority()?;
        let mut green = std::ptr::null_mut();
        // SAFETY: self is a live stream.
        driver.check(
            unsafe { (driver.stream_get_green_ctx)(self.raw, &mut green) },
            "cuStreamGetGreenCtx",
        )?;

        let mut raw = std::ptr::null_mut();
        if green.is_null() {
            in_context(self.context, |driver| {
                // SAFETY: the driver writes a stream handle in self's context.
                driver.check(
                    unsafe {
                        (driver.stream_create_with_priority)(
                            &mut raw,
                            CU_STREAM_NON_BLOCKING,
                            priority,
                        )
                    },
                    "cuStreamCreateWithPriority",
                )
            })?;
        } else {
            // SAFETY: the green context is live while self's retention is.
            driver.check(
                unsafe {
                    (driver.green_ctx_stream_create)(
                        &mut raw,
                        green,
                        CU_STREAM_NON_BLOCKING,
                        priority,
                    )
                },
                "cuGreenCtxStreamCreate",
            )?;
        }

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
        // SAFETY: self is a live stream; the driver writes one int.
        driver.check(
            unsafe { (driver.stream_get_priority)(self.raw, &mut priority) },
            "cuStreamGetPriority",
        )?;
        Ok(priority)
    }

    /// Whether the stream runs in an SM partition (a green context).
    pub fn is_partitioned(&self) -> Result<bool> {
        let driver = driver()?;
        let mut green = std::ptr::null_mut();
        // SAFETY: self is a live stream; the driver writes one handle.
        driver.check(
            unsafe { (driver.stream_get_green_ctx)(self.raw, &mut green) },
            "cuStreamGetGreenCtx",
        )?;
        Ok(!green.is_null())
    }

    /// The SMs available to kernels on this stream: the partition's count for
    /// a green-context stream, otherwise the context's (whole device's).
    pub fn sm_count(&self) -> Result<u32> {
        if let Retention::Green(green) = &self.retention {
            return Ok(green.sm_count());
        }
        green::context_sm_count(self.context)
    }

    pub fn capture_status(&self) -> Result<CaptureStatus> {
        let driver = driver()?;
        let mut status = 0;
        // SAFETY: self is a live stream; the driver writes one enum value.
        driver.check(
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

    /// Runs `action` on a driver thread after all preceding work on the stream
    /// completes, and before any later work starts.
    ///
    /// The action must not call CUDA and must not block on anything that waits
    /// for this stream (including the Python GIL): the driver serializes host
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
