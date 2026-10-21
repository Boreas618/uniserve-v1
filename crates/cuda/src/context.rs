//! Primary context retention and per-thread selection.

use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

use crate::driver::{Handle, driver};
use crate::{DeviceId, Result};

/// A retained reference to a device's primary context.
///
/// Cloning shares one driver reference; the last clone releases it with
/// `cuDevicePrimaryCtxRelease`. The driver destroys the primary context when
/// its final reference (including PyTorch's) is released, so the worker keeps
/// one `PrimaryContext` per placed device for its whole lifetime. PyTorch's
/// reference outlives every owner here, so a process destroys its contexts
/// explicitly at its end ([`PrimaryContext::reset_all`]).
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
// SAFETY: as for Send above; shared references only copy the immutable fields
// out for those driver calls.
unsafe impl Sync for Retained {}

impl PrimaryContext {
    /// Retains the device's primary context, creating it if needed.
    pub fn retain(device: DeviceId) -> Result<Self> {
        let driver = driver()?;
        let handle = device.handle()?;
        let mut raw = std::ptr::null_mut();
        driver.check(
            // SAFETY: the driver writes the context handle to the local `raw` for
            // `handle`, a valid device from `DeviceId::handle`.
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
        driver.check(
            // SAFETY: `self` retains the context, and the guard keeps a clone of it
            // while the context is current.
            unsafe { (driver.ctx_push)(self.raw()) },
            "cuCtxPushCurrent_v2",
        )?;

        Ok(DeviceGuard {
            _context: self.clone(),
            _thread_bound: PhantomData,
        })
    }

    /// Destroys every active primary context of this process with
    /// `cuDevicePrimaryCtxReset`, whatever references to it remain, and
    /// returns the devices whose context was destroyed.
    ///
    /// Some driver state outlives the process unless its context is
    /// destroyed: the driver backs the interprocess events of a context with
    /// a POSIX shared-memory object named for the process
    /// (`/dev/shm/cuda.shm.<uid>.<pid>.<n>`, ids in hexadecimal; see
    /// `ipc::launch::remove_event_memory`), created by the first
    /// `cuEventCreate` with `CU_EVENT_INTERPROCESS`, and unlinks it only when
    /// the context is destroyed. Destroying the events does not unlink it, and neither does
    /// the process's exit, by `exit` or `_exit`. A context the CUDA runtime
    /// retains (PyTorch's) is never released before a rank process ends, so
    /// the rank destroys its contexts as the last step of an orderly end.
    ///
    /// Releases queued on the reclaimer run first, since each needs its
    /// context. Destroying a context waits for the work queued in it, so the
    /// call does not return while some of that work cannot complete. A
    /// process that never used the driver through this crate has no context
    /// to destroy here, and the call loads nothing. Every active context is
    /// reset even when a reset or a queued release fails; the first failure
    /// is returned.
    ///
    /// # Safety
    ///
    /// Nothing in the process uses any device afterwards: every owner of
    /// device state, in Rust or in a library such as PyTorch, has been
    /// closed or is never used again, and no other thread makes driver
    /// calls meanwhile. The handles of every context's objects, including
    /// this crate's owners still alive, dangle once it is destroyed.
    pub unsafe fn reset_all() -> Result<Vec<DeviceId>> {
        let Some(driver) = crate::driver::initialized() else {
            return Ok(Vec::new());
        };
        let mut outcome = crate::reclaim::flush();

        let mut count = 0;
        // SAFETY: the driver writes the number of visible devices.
        let counted = driver.check(
            unsafe { (driver.device_get_count)(&mut count) },
            "cuDeviceGetCount",
        );
        outcome = outcome.and(counted);

        let mut reset = Vec::new();
        for ordinal in 0..count {
            let device = DeviceId(ordinal);
            let destroyed = device.handle().and_then(|handle| {
                let (mut flags, mut active) = (0u32, 0i32);
                driver.check(
                    // SAFETY: the driver writes the state of a valid device's
                    // primary context to the locals `flags` and `active`.
                    unsafe { (driver.primary_ctx_get_state)(handle, &mut flags, &mut active) },
                    "cuDevicePrimaryCtxGetState",
                )?;
                if active == 0 {
                    return Ok(false);
                }
                driver.check(
                    // SAFETY: the caller guarantees that nothing uses the context
                    // again; the next retain would create a new one.
                    unsafe { (driver.primary_ctx_reset)(handle) },
                    "cuDevicePrimaryCtxReset_v2",
                )?;
                Ok(true)
            });
            match destroyed {
                Ok(true) => reset.push(device),
                Ok(false) => {}
                Err(error) => outcome = outcome.and(Err(error)),
            }
        }
        outcome.map(|()| reset)
    }

    /// Arranges for [`Self::reset_all`] to run when this process exits by
    /// `exit`: returning from `main` or `std::process::exit`, as a test
    /// harness does once every test has completed and a Python interpreter
    /// does after its finalization. The first interprocess event a process
    /// creates calls this ([`crate::EventFlags::interprocess`]): such a
    /// process would otherwise leave the events' shared memory behind, as
    /// the CUDA runtime's convention of calling `cudaDeviceReset` before
    /// exiting avoids. A rank never exits by `exit`: it destroys its contexts
    /// itself when nothing it issued can still run, and ends by `_exit`.
    ///
    /// The driver is loaded first, so the exit handlers it registers as it
    /// loads run after this one, while it still serves the reset. Later
    /// calls register nothing more. A failure of the reset at exit is
    /// reported on standard error.
    ///
    /// # Safety
    ///
    /// When the process exits, no thread uses a device any more, and the work
    /// queued on every device can complete: the reset waits for it.
    pub(crate) unsafe fn reset_all_at_exit() -> Result<()> {
        static REGISTERED: OnceLock<bool> = OnceLock::new();

        driver()?;
        // SAFETY: registers a handler that takes no arguments and does not
        // unwind; the caller's guarantee covers the reset it makes.
        let registered = *REGISTERED.get_or_init(|| unsafe { libc::atexit(reset_at_exit) } == 0);
        if registered {
            Ok(())
        } else {
            Err(crate::Error::ExitHandler)
        }
    }
}

/// The exit handler [`PrimaryContext::reset_all_at_exit`] registers.
extern "C" fn reset_at_exit() {
    // SAFETY: the registrant guaranteed that nothing uses a device once the
    // process exits.
    if let Err(error) = unsafe { PrimaryContext::reset_all() } {
        use std::io::Write;
        // A closed standard error must not panic inside an exit handler.
        let _ = writeln!(
            std::io::stderr(),
            "cuda: destroying the device contexts at exit failed: {error}"
        );
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
