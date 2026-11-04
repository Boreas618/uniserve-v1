//! Device code loaded as a CUDA library with `cuLibraryLoadData`: cubin,
//! fatbin or PTX images, and the kernels resolved from them.
//!
//! A [`Library`] is what the design calls a module (4.2): a compiled kernel
//! program for one descriptor, loaded on its device. Its kernels are
//! resolved with `cuLibraryGetKernel` and `cuKernelGetFunction`. The library
//! unloads with `cuLibraryUnload` when its last owner drops it; an owner keeps
//! it beside every [`Function`] it launches, and a captured graph that calls
//! its kernels is rebound or destroyed first (design 7.3, purge), because CUDA
//! does not define a captured kernel's behaviour once its library is unloaded.

use std::ffi::CString;

use crate::driver::{self, Handle};
use crate::launch::Function;
use crate::{Device, DeviceId, Error, PrimaryContext, Result, reclaim};

/// A module of one device loaded as a library (`CUlibrary`).
///
/// The image may be a cubin, a fatbin, or PTX text: the driver tells the forms
/// apart, reading an ELF image by its own headers and compiling PTX for the
/// device. Loading retains the device's primary context, which every kernel
/// of the library is resolved in.
pub struct Library {
    raw: Handle,
    context: PrimaryContext,
    device: DeviceId,
}

// SAFETY: a library handle is an immutable driver object usable from any
// thread with its context; the retained context is thread-safe.
unsafe impl Send for Library {}
// SAFETY: as for Send above; shared references only pass the immutable handle
// to driver calls.
unsafe impl Sync for Library {}

impl Library {
    /// Loads `image` on `device`.
    ///
    /// # Errors
    ///
    /// Fails with the driver's status when the image is not a valid library
    /// for the device — an empty image, another architecture's cubin, or PTX
    /// the driver cannot compile.
    pub fn load(device: &Device, image: &[u8]) -> Result<Self> {
        if image.is_empty() {
            return Err(Error::invalid("a library image is empty"));
        }

        let context = PrimaryContext::retain(device.id())?;
        // PTX text reaches the driver as a C string; a binary image's size
        // comes from its ELF headers, and the appended NUL lies outside them.
        let mut code = Vec::with_capacity(image.len() + 1);
        code.extend_from_slice(image);
        code.push(0);

        let mut raw = std::ptr::null_mut();
        driver::in_context(context.raw(), |driver| {
            driver.check(
                // SAFETY: the image is live for the call and NUL-terminated;
                // no JIT or library options are given, and the driver writes
                // the library handle to the local `raw`.
                unsafe {
                    (driver.library_load_data)(
                        &mut raw,
                        code.as_ptr().cast(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        0,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        0,
                    )
                },
                "cuLibraryLoadData",
            )
        })?;

        Ok(Self {
            raw,
            context,
            device: device.id(),
        })
    }

    /// The kernel named `name`, resolved in this device's primary context.
    ///
    /// The function is valid while this library is loaded.
    pub fn kernel(&self, name: &str) -> Result<Function> {
        let name =
            CString::new(name).map_err(|error| Error::invalid(format!("kernel name: {error}")))?;
        let mut kernel: Handle = std::ptr::null_mut();
        let mut function: Handle = std::ptr::null_mut();
        driver::in_context(self.context.raw(), |driver| {
            driver.check(
                // SAFETY: the library is loaded; the driver writes the kernel
                // handle to the local `kernel`.
                unsafe { (driver.library_get_kernel)(&mut kernel, self.raw, name.as_ptr()) },
                "cuLibraryGetKernel",
            )?;
            driver.check(
                // SAFETY: the kernel belongs to the library; the current
                // context is this library's, and the driver writes the
                // function handle to the local `function`.
                unsafe { (driver.kernel_get_function)(&mut function, kernel) },
                "cuKernelGetFunction",
            )
        })?;
        Ok(Function::new(function, self.context.clone()))
    }

    /// The device the library is loaded on.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// The `CUlibrary` handle.
    pub fn raw(&self) -> usize {
        self.raw as usize
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        let (context, raw) = (self.context.clone(), self.raw as usize);
        // The release never blocks the dropping thread (see `reclaim`), and
        // the owner guarantees no executable graph still calls these kernels.
        reclaim::defer(move || {
            driver::in_context(context.raw(), |driver| {
                driver.check(
                    // SAFETY: the library is owned and unloaded once.
                    unsafe { (driver.library_unload)(raw as Handle) },
                    "cuLibraryUnload",
                )
            })
        });
    }
}
