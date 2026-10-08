//! Primary context retention and per-thread selection.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::driver::{Handle, driver};
use crate::{DeviceId, Result};

/// A retained reference to a device's primary context.
///
/// Cloning shares one driver reference; the last clone releases it with
/// `cuDevicePrimaryCtxRelease`. The driver destroys the primary context when
/// its final reference (including PyTorch's) is released, so the worker keeps
/// one `PrimaryContext` per placed device for its whole lifetime.
#[derive(Clone)]
pub struct PrimaryContext {
    retained: Arc<Retained>,
}

struct Retained {
    device: DeviceId,
    handle: i32,
    raw: Handle,
}

// SAFETY: retaining, pushing and releasing a primary context are thread-safe
// driver operations. The raw handle is only passed to the driver.
unsafe impl Send for Retained {}
unsafe impl Sync for Retained {}

impl PrimaryContext {
    /// Retains the device's primary context, creating it if needed.
    pub fn retain(device: DeviceId) -> Result<Self> {
        let driver = driver()?;
        let handle = device.handle()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: the driver writes the context handle for a valid device.
        driver.check(
            unsafe { (driver.primary_ctx_retain)(&mut raw, handle) },
            "cuDevicePrimaryCtxRetain",
        )?;

        Ok(Self {
            retained: Arc::new(Retained {
                device,
                handle,
                raw,
            }),
        })
    }

    pub fn device(&self) -> DeviceId {
        self.retained.device
    }

    pub(crate) fn raw(&self) -> Handle {
        self.retained.raw
    }

    /// Makes this context current on the calling thread until the guard drops.
    pub fn enter(&self) -> Result<DeviceGuard> {
        let driver = driver()?;
        // SAFETY: self retains the context for the guard's lifetime.
        driver.check(
            unsafe { (driver.ctx_push)(self.raw()) },
            "cuCtxPushCurrent_v2",
        )?;

        Ok(DeviceGuard {
            _context: self.clone(),
            _thread_bound: PhantomData,
        })
    }
}

impl Drop for Retained {
    fn drop(&mut self) {
        if let Ok(driver) = driver() {
            // SAFETY: releases exactly the reference retained at construction.
            // A failure cannot be reported from drop and leaves one reference.
            let _ = unsafe { (driver.primary_ctx_release)(self.handle) };
        }
    }
}

/// Selects a device's primary context on the calling thread.
///
/// A fresh host thread (for example one PyTorch never touched) has no current
/// driver context, and driver calls that allocate or create objects need one.
/// The guard pushes the primary context and pops it on drop, restoring the
/// thread's previous context. It does not change any PyTorch stream selection.
///
/// The guard is neither `Send` nor `Sync`: the context stack is per thread
/// and must be popped by the thread that pushed it.
pub struct DeviceGuard {
    _context: PrimaryContext,
    _thread_bound: PhantomData<*const ()>,
}

impl DeviceGuard {
    /// Retains and enters `device`'s primary context.
    pub fn new(device: DeviceId) -> Result<Self> {
        PrimaryContext::retain(device)?.enter()
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        if let Ok(driver) = driver() {
            let mut popped = std::ptr::null_mut();
            // SAFETY: pops the context this guard pushed on this thread.
            let _ = unsafe { (driver.ctx_pop)(&mut popped) };
        }
    }
}
