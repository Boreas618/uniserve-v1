//! Device code this process loads from PTX text, and launches of its
//! kernels.
//!
//! A [`Module`] is JIT-compiled by the driver for its device when loaded and
//! stays loaded for the process's life: a kernel captured into a CUDA graph
//! keeps calling it, and nothing tells when the last such graph is gone.

use std::ffi::{CString, c_void};

use crate::driver::{Handle, in_context};
use crate::{DeviceId, Error, PrimaryContext, Result, Stream};

/// A loaded module of one device's primary context.
pub struct Module {
    context: PrimaryContext,
    raw: Handle,
}

// SAFETY: a module handle is an immutable driver object, usable from any
// thread with its context current.
unsafe impl Send for Module {}
// SAFETY: as above.
unsafe impl Sync for Module {}

/// A kernel of a [`Module`].
#[derive(Clone, Copy)]
pub struct Function {
    raw: Handle,
}

// SAFETY: a function handle is an immutable driver object.
unsafe impl Send for Function {}
// SAFETY: as above.
unsafe impl Sync for Function {}

impl Module {
    /// Load the PTX text `ptx` on `device`, compiled by the driver for it.
    ///
    /// # Errors
    ///
    /// Fails when the driver cannot compile or load it.
    pub fn load_ptx(device: DeviceId, ptx: &str) -> Result<Self> {
        let context = PrimaryContext::retain(device)?;
        let text = CString::new(ptx).map_err(|error| Error::InvalidArgument(error.to_string()))?;
        let mut raw = std::ptr::null_mut();
        in_context(context.raw(), |driver| {
            driver.check(
                // SAFETY: the text is NUL-terminated PTX; the driver writes
                // the module handle.
                unsafe { (driver.module_load_data)(&mut raw, text.as_ptr().cast()) },
                "cuModuleLoadData",
            )
        })?;
        Ok(Self { context, raw })
    }

    /// The kernel named `name`.
    pub fn function(&self, name: &str) -> Result<Function> {
        let name = CString::new(name).map_err(|error| Error::InvalidArgument(error.to_string()))?;
        let mut raw = std::ptr::null_mut();
        in_context(self.context.raw(), |driver| {
            driver.check(
                // SAFETY: the module is loaded; the driver writes the handle.
                unsafe { (driver.module_get_function)(&mut raw, self.raw, name.as_ptr()) },
                "cuModuleGetFunction",
            )
        })?;
        Ok(Function { raw })
    }
}

impl Function {
    /// Launch the kernel on `stream` with `grid` blocks of `block` threads
    /// and no dynamic shared memory; `parameters` points at each kernel
    /// parameter's value, in order. A launch on a capturing stream is
    /// captured.
    ///
    /// # Safety
    /// `parameters` matches the kernel's parameter list, and the memory the
    /// kernel addresses outlives every execution of the launch (including
    /// the replays of a graph that captured it).
    pub unsafe fn launch(
        &self,
        stream: &Stream,
        grid: u32,
        block: u32,
        parameters: &mut [*mut c_void],
    ) -> Result<()> {
        let driver = crate::driver::driver()?;
        let _context = stream.enter()?;
        driver.check(
            // SAFETY: forwarded from the caller; the stream is live.
            unsafe {
                (driver.launch_kernel)(
                    self.raw,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    stream.handle(),
                    parameters.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}
