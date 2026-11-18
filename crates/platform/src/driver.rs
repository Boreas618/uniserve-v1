//! The `libcuda.so.1` symbol table, resolved once per process.
//!
//! The NVIDIA driver package installs the driver under its stable system soname, which is opened
//! by that name; no toolkit or user path is involved. Every entry point is resolved when the table
//! is first needed and `cuInit` runs once. A driver whose CUDA version (`cuDriverGetVersion`) is
//! older than the `cuda.h` the bindings are generated from ([`sys::CUDA_VERSION`]) is refused with
//! [`Error::Version`]: every entry point of the table is one that release provides, so none is
//! optional.
//!
//! The structures and constants handed to the driver are the generated ones of
//! [`crate::sys::cuda`]; this module only adds their constructors.

use std::ffi::{CStr, c_char, c_void};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use libloading::Library;

use crate::sys::cuda as sys;
use crate::{Error, Result};

pub(crate) type Handle = *mut c_void;
pub(crate) type CuResult = sys::CUresult;
pub(crate) type HostFn = unsafe extern "C" fn(Handle);

pub(crate) const LIBCUDA: &str = "libcuda.so.1";

/// `CUgraphExecUpdateResultInfo`: what `cuGraphExecUpdate` reports beside its status, the update's
/// result and the nodes it failed at.
#[repr(C)]
pub(crate) struct ExecUpdateInfo {
    /// The `CUgraphExecUpdateResult`; zero (`CU_GRAPH_EXEC_UPDATE_SUCCESS`) on success.
    pub(crate) result: u32,
    pub(crate) error_node: Handle,
    pub(crate) error_from_node: Handle,
}

/// What a refused driver is told to install: the NVIDIA driver of the CUDA release the bindings
/// are generated from, or a later one.
const DRIVER_PACKAGE: &str = "the NVIDIA driver of CUDA 13.0 or later";

impl sys::CUmemLocation {
    /// The storage of the device `ordinal`.
    pub(crate) fn device(ordinal: i32) -> Self {
        Self {
            type_: sys::CU_MEM_LOCATION_TYPE_DEVICE,
            id: ordinal,
        }
    }
}

impl sys::CUmemAllocationProp {
    /// Device-resident physical storage of the device `ordinal`, exportable as `handle_types`.
    /// The Windows metadata and the allocation flags stay zero.
    pub(crate) fn device(ordinal: i32, handle_types: sys::CUmemAllocationHandleType) -> Self {
        Self {
            type_: sys::CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes: handle_types,
            location: sys::CUmemLocation::device(ordinal),
            ..Self::default()
        }
    }
}

impl sys::CUmemAccessDesc {
    /// The device `ordinal`'s access to a mapping, with `CU_MEM_ACCESS_FLAGS_PROT_*` `flags`.
    pub(crate) fn device(ordinal: i32, flags: sys::CUmemAccess_flags) -> Self {
        Self {
            location: sys::CUmemLocation::device(ordinal),
            flags,
        }
    }
}

/// Declares the driver table: one typed field per entry point.
macro_rules! driver_table {
    ($($field:ident = $symbol:literal: fn($($arg:ty),* $(,)?);)*) => {
        pub(crate) struct Driver {
            $(pub(crate) $field: unsafe extern "C" fn($($arg),*) -> CuResult,)*
            // Declared last: the library outlives every copied function pointer.
            _library: Library,
        }

        impl Driver {
            fn resolve(library: Library) -> Result<Self> {
                // SAFETY: every signature matches the CUDA 13 `cuda.h`
                // prototype of the named, versioned entry point.
                unsafe {
                    Ok(Self {
                        $($field: symbol(&library, $symbol)?,)*
                        _library: library,
                    })
                }
            }
        }
    };
}

driver_table! {
    init = "cuInit": fn(u32);
    driver_get_version = "cuDriverGetVersion": fn(*mut i32);
    device_get = "cuDeviceGet": fn(*mut i32, i32);
    device_get_count = "cuDeviceGetCount": fn(*mut i32);
    device_get_attribute = "cuDeviceGetAttribute": fn(*mut i32, sys::CUdevice_attribute, i32);
    device_get_uuid = "cuDeviceGetUuid_v2": fn(*mut sys::CUuuid, i32);
    device_can_access_peer = "cuDeviceCanAccessPeer": fn(*mut i32, i32, i32);
    get_error_name = "cuGetErrorName": fn(CuResult, *mut *const c_char);
    get_error_string = "cuGetErrorString": fn(CuResult, *mut *const c_char);
    primary_ctx_retain = "cuDevicePrimaryCtxRetain": fn(*mut Handle, i32);
    primary_ctx_release = "cuDevicePrimaryCtxRelease_v2": fn(i32);
    primary_ctx_get_state = "cuDevicePrimaryCtxGetState": fn(i32, *mut u32, *mut i32);
    primary_ctx_reset = "cuDevicePrimaryCtxReset_v2": fn(i32);
    ctx_push = "cuCtxPushCurrent_v2": fn(Handle);
    ctx_pop = "cuCtxPopCurrent_v2": fn(*mut Handle);
    ctx_get_stream_priority_range = "cuCtxGetStreamPriorityRange": fn(*mut i32, *mut i32);
    thread_exchange_capture_mode = "cuThreadExchangeStreamCaptureMode": fn(
        *mut sys::CUstreamCaptureMode,
    );
    profiler_start = "cuProfilerStart": fn();
    profiler_stop = "cuProfilerStop": fn();

    stream_create_with_priority = "cuStreamCreateWithPriority": fn(*mut Handle, u32, i32);
    stream_get_priority = "cuStreamGetPriority": fn(Handle, *mut i32);
    stream_is_capturing = "cuStreamIsCapturing": fn(Handle, *mut sys::CUstreamCaptureStatus);
    stream_query = "cuStreamQuery": fn(Handle);
    stream_synchronize = "cuStreamSynchronize": fn(Handle);
    stream_destroy = "cuStreamDestroy_v2": fn(Handle);
    stream_wait_event = "cuStreamWaitEvent": fn(Handle, Handle, u32);
    stream_write_value32 = "cuStreamWriteValue32_v2": fn(Handle, u64, u32, u32);
    stream_wait_value32 = "cuStreamWaitValue32_v2": fn(Handle, u64, u32, u32);
    launch_host_func = "cuLaunchHostFunc": fn(Handle, HostFn, Handle);
    graph_launch = "cuGraphLaunch": fn(Handle, Handle);
    graph_instantiate = "cuGraphInstantiateWithFlags": fn(*mut Handle, Handle, u64);
    graph_exec_destroy = "cuGraphExecDestroy": fn(Handle);
    graph_exec_update = "cuGraphExecUpdate_v2": fn(Handle, Handle, *mut ExecUpdateInfo);

    event_create = "cuEventCreate": fn(*mut Handle, u32);
    event_record_with_flags = "cuEventRecordWithFlags": fn(Handle, Handle, u32);
    event_query = "cuEventQuery": fn(Handle);
    event_synchronize = "cuEventSynchronize": fn(Handle);
    event_elapsed_time = "cuEventElapsedTime_v2": fn(*mut f32, Handle, Handle);
    event_destroy = "cuEventDestroy_v2": fn(Handle);
    ipc_get_event_handle = "cuIpcGetEventHandle": fn(*mut sys::CUipcEventHandle, Handle);
    ipc_open_event_handle = "cuIpcOpenEventHandle": fn(*mut Handle, sys::CUipcEventHandle);

    mem_alloc = "cuMemAlloc_v2": fn(*mut u64, usize);
    mem_free = "cuMemFree_v2": fn(u64);
    mem_get_info = "cuMemGetInfo_v2": fn(*mut usize, *mut usize);
    mem_host_alloc = "cuMemHostAlloc": fn(*mut Handle, usize, u32);
    mem_free_host = "cuMemFreeHost": fn(Handle);
    mem_host_register = "cuMemHostRegister_v2": fn(Handle, usize, u32);
    mem_host_unregister = "cuMemHostUnregister": fn(Handle);

    mem_get_allocation_granularity = "cuMemGetAllocationGranularity": fn(
        *mut usize, *const sys::CUmemAllocationProp, sys::CUmemAllocationGranularity_flags,
    );
    mem_create = "cuMemCreate": fn(*mut u64, usize, *const sys::CUmemAllocationProp, u64);
    mem_release = "cuMemRelease": fn(u64);
    mem_address_reserve = "cuMemAddressReserve": fn(*mut u64, usize, usize, u64, u64);
    mem_address_free = "cuMemAddressFree": fn(u64, usize);
    mem_map = "cuMemMap": fn(u64, usize, usize, u64, u64);
    mem_unmap = "cuMemUnmap": fn(u64, usize);
    mem_set_access = "cuMemSetAccess": fn(u64, usize, *const sys::CUmemAccessDesc, usize);
    mem_export_to_shareable_handle = "cuMemExportToShareableHandle": fn(
        Handle, u64, sys::CUmemAllocationHandleType, u64,
    );
    mem_import_from_shareable_handle = "cuMemImportFromShareableHandle": fn(
        *mut u64, Handle, sys::CUmemAllocationHandleType,
    );
    mem_retain_allocation_handle = "cuMemRetainAllocationHandle": fn(*mut u64, Handle);

    memcpy_async = "cuMemcpyAsync": fn(u64, u64, usize, Handle);
    memset_d8_async = "cuMemsetD8Async": fn(u64, u8, usize, Handle);
    memset_d16_async = "cuMemsetD16Async": fn(u64, u16, usize, Handle);
    memset_d32_async = "cuMemsetD32Async": fn(u64, u32, usize, Handle);
    memset_d2d8_async = "cuMemsetD2D8Async": fn(u64, usize, u8, usize, usize, Handle);
    memset_d2d32_async = "cuMemsetD2D32Async": fn(u64, usize, u32, usize, usize, Handle);
    // The batch copies, in their CUDA 13 `_v2` form (no failure index).
    memcpy_batch_async = "cuMemcpyBatchAsync_v2": fn(
        *mut u64, *mut u64, *mut usize, usize, *mut sys::CUmemcpyAttributes, *mut usize, usize,
        Handle,
    );
    memcpy_3d_batch_async = "cuMemcpy3DBatchAsync_v2": fn(
        usize, *mut sys::CUDA_MEMCPY3D_BATCH_OP, u64, Handle,
    );

    library_load_data = "cuLibraryLoadData": fn(
        *mut Handle, *const c_void, *mut i32, *mut *mut c_void, u32, *mut i32,
        *mut *mut c_void, u32,
    );
    library_get_kernel = "cuLibraryGetKernel": fn(*mut Handle, Handle, *const c_char);
    kernel_get_function = "cuKernelGetFunction": fn(*mut Handle, Handle);
    library_unload = "cuLibraryUnload": fn(Handle);

    launch_kernel_ex = "cuLaunchKernelEx": fn(
        *const sys::CUlaunchConfig, Handle, *mut *mut c_void, *mut *mut c_void,
    );
    func_set_attribute = "cuFuncSetAttribute": fn(Handle, sys::CUfunction_attribute, i32);

    stream_begin_capture = "cuStreamBeginCapture_v2": fn(Handle, sys::CUstreamCaptureMode);
    stream_end_capture = "cuStreamEndCapture": fn(Handle, *mut Handle);
    stream_get_capture_info = "cuStreamGetCaptureInfo_v3": fn(
        Handle, *mut sys::CUstreamCaptureStatus, *mut u64, *mut Handle, *mut *const Handle,
        *mut *const sys::CUgraphEdgeData, *mut usize,
    );
    graph_destroy = "cuGraphDestroy": fn(Handle);
    graph_get_nodes = "cuGraphGetNodes": fn(Handle, *mut Handle, *mut usize);
    graph_node_get_type = "cuGraphNodeGetType": fn(Handle, *mut sys::CUgraphNodeType);
    func_get_param_info = "cuFuncGetParamInfo": fn(
        Handle, usize, *mut usize, *mut usize,
    );

    tensor_map_encode_tiled = "cuTensorMapEncodeTiled": fn(
        *mut sys::CUtensorMap, i32, u32, *mut c_void, *const u64, *const u64, *const u32,
        *const u32, sys::CUtensorMapInterleave, sys::CUtensorMapSwizzle,
        sys::CUtensorMapL2promotion, sys::CUtensorMapFloatOOBfill,
    );
}

/// Resolves one entry point of `libcuda.so.1`.
///
/// # Safety
/// `T` must be the function-pointer type of the named symbol's C prototype.
unsafe fn symbol<T: Copy>(library: &Library, name: &'static str) -> Result<T> {
    // SAFETY: forwarded from the caller; the pointer is copied out while the
    // library is alive and the table keeps the library alive with it.
    unsafe { library.get::<T>(name.as_bytes()) }
        .map(|symbol| *symbol)
        .map_err(|source| Error::Symbol {
            library: LIBCUDA,
            symbol: name,
            source: Arc::new(source),
        })
}

/// The process driver table, set by the first [`driver`] call.
static DRIVER: OnceLock<Result<Driver>> = OnceLock::new();

/// Returns the process driver table, loading it, running `cuInit` and checking the driver's CUDA
/// version on first use. A failed load is cached and returned to every caller.
pub(crate) fn driver() -> Result<&'static Driver> {
    DRIVER
        .get_or_init(|| {
            // SAFETY: loading libcuda runs only the driver's own initializers.
            let library = unsafe { Library::new(LIBCUDA) }.map_err(|source| Error::Load {
                library: LIBCUDA,
                source: Arc::new(source),
            })?;
            let driver = Driver::resolve(library)?;

            // SAFETY: cuInit takes flags by value and must be passed zero.
            let status = unsafe { (driver.init)(0) };
            driver.check(status, "cuInit")?;

            let mut version = 0;
            driver.check(
                // SAFETY: the driver writes one int to the local `version`.
                unsafe { (driver.driver_get_version)(&mut version) },
                "cuDriverGetVersion",
            )?;
            require_version(version)?;
            Ok(driver)
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Refuses a driver whose CUDA version is older than the `cuda.h` the bindings are generated
/// from. Both are CUDA version codes, `major * 1000 + minor * 10`, as `cuDriverGetVersion` and
/// `CUDA_VERSION` state them.
fn require_version(version: i32) -> Result<()> {
    let required = sys::CUDA_VERSION as i32;
    if version >= required {
        return Ok(());
    }
    Err(Error::Version {
        library: LIBCUDA,
        found: cuda_release(version),
        required: cuda_release(required),
        package: DRIVER_PACKAGE,
    })
}

/// A CUDA version code as its release, such as `CUDA 13.0` for 13000.
pub(crate) fn cuda_release(code: i32) -> String {
    format!("CUDA {}.{}", code / 1000, code % 1000 / 10)
}

/// Returns the process driver table if an earlier [`driver`] call loaded and
/// initialized it, without loading it otherwise.
pub(crate) fn initialized() -> Option<&'static Driver> {
    DRIVER.get().and_then(|driver| driver.as_ref().ok())
}

/// Where the crate's address reservations start: 4 TiB.
///
/// A reservation without an address hint takes the lowest free device
/// addresses, from 13 GiB up, and CUPTI's range profiler needs free
/// addresses among those for the device storage of its pass: on GB200 with
/// CUDA 13 and `nvidia-cuda-cupti` 13.0, reservations without a hint that
/// take the addresses up to 1 TiB leave a pass of 2048 ranges (2.9 GB)
/// unable to start (`CUPTI_ERROR_UNKNOWN`), whatever the device has free,
/// while 35 TiB of reservations hinted from 4 TiB on leave it starting. The
/// crate's reservations, each up to a device's whole storage (a graph pool,
/// a mapping of weights or KV cache planes), are placed from here on,
/// leaving the lower addresses to the driver and CUPTI.
const RESERVATIONS_BASE: u64 = 1 << 42;

/// Where hinted reservations wrap back to [`RESERVATIONS_BASE`]: 128 TiB,
/// the top of a 47-bit user address space.
const RESERVATIONS_END: u64 = 1 << 47;

/// The hint of the next reservation: past the end of the last one.
static NEXT_RESERVATION: AtomicU64 = AtomicU64::new(RESERVATIONS_BASE);

/// Reserves `bytes` of device addresses aligned to `alignment` (both
/// multiples of the allocation granularity), hinted past the last such
/// reservation from [`RESERVATIONS_BASE`] on; returns the range's base. The
/// hint is a preference: the driver places a range whose hinted addresses
/// are taken (by a reservation of another library) wherever it has room.
///
/// # Errors
///
/// Fails when the driver cannot reserve the range anywhere.
pub(crate) fn reserve_addresses(driver: &Driver, bytes: usize, alignment: usize) -> Result<u64> {
    let alignment = alignment.max(1) as u64;
    let span = bytes as u64;
    let mut hint = 0;
    let _ = NEXT_RESERVATION.try_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
        let start = next.next_multiple_of(alignment);
        hint = if start + span <= RESERVATIONS_END {
            start
        } else {
            RESERVATIONS_BASE.next_multiple_of(alignment)
        };
        Some(hint + span)
    });
    let mut address = 0;
    driver.check(
        // SAFETY: the driver writes the base of a new range to the local
        // `address`.
        unsafe { (driver.mem_address_reserve)(&mut address, bytes, alignment as usize, hint, 0) },
        "cuMemAddressReserve",
    )?;
    Ok(address)
}

impl Driver {
    /// Converts a `CUresult` into a typed error naming the failed call.
    pub(crate) fn check(&self, status: CuResult, call: &'static str) -> Result<()> {
        if status == sys::CUDA_SUCCESS {
            return Ok(());
        }

        Err(Error::Driver {
            call,
            // `CUresult` is a C enum; the error carries the value the driver documents.
            code: status as i32,
            name: self.describe(self.get_error_name, status),
            message: self.describe(self.get_error_string, status),
        })
    }

    fn describe(
        &self,
        lookup: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
        status: CuResult,
    ) -> &'static str {
        let mut text = std::ptr::null();
        // SAFETY: the driver writes a pointer to a static, NUL-terminated
        // string owned by libcuda, which is never unloaded (the table lives in
        // a process-wide static).
        unsafe {
            if lookup(status, &mut text) != sys::CUDA_SUCCESS || text.is_null() {
                return "unrecognized CUDA status";
            }
            CStr::from_ptr(text)
                .to_str()
                .unwrap_or("non-UTF-8 CUDA status")
        }
    }
}

/// Makes a context current on the calling thread until dropped, restoring
/// the thread's previous context. Not `Send`: the context stack is per thread.
pub(crate) struct ContextScope {
    driver: &'static Driver,
    _thread_bound: PhantomData<*const ()>,
}

impl ContextScope {
    /// The caller keeps `context` alive for the scope's lifetime.
    pub(crate) fn enter(context: Handle) -> Result<Self> {
        let driver = driver()?;
        // SAFETY: the caller guarantees the context is live.
        driver.check(unsafe { (driver.ctx_push)(context) }, "cuCtxPushCurrent_v2")?;
        Ok(Self {
            driver,
            _thread_bound: PhantomData,
        })
    }
}

impl Drop for ContextScope {
    fn drop(&mut self) {
        let mut popped = std::ptr::null_mut();
        // SAFETY: pops exactly the context pushed by `enter`, on its thread.
        let _ = unsafe { (self.driver.ctx_pop)(&mut popped) };
    }
}

/// Runs `operation` with `context` current on the calling thread.
///
/// The caller keeps `context` alive for the duration of the call.
pub(crate) fn in_context<T>(
    context: Handle,
    operation: impl FnOnce(&'static Driver) -> Result<T>,
) -> Result<T> {
    let scope = ContextScope::enter(context)?;
    operation(scope.driver)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A driver older than the `cuda.h` the bindings are generated from is refused by name, with
    /// both releases and the package to install; that release and later ones are accepted.
    #[test]
    fn a_driver_older_than_the_bindings_is_refused() {
        assert!(require_version(sys::CUDA_VERSION as i32).is_ok());
        assert!(require_version(sys::CUDA_VERSION as i32 + 10).is_ok());

        let error = require_version(12080).expect_err("CUDA 12.8 predates the bindings");
        let text = error.to_string();
        assert!(
            matches!(
                error,
                Error::Version {
                    library: LIBCUDA,
                    ..
                }
            ),
            "{text}"
        );
        assert!(
            text.contains("CUDA 12.8") && text.contains("CUDA 13.0"),
            "{text}"
        );
        assert!(text.contains(DRIVER_PACKAGE), "{text}");
    }
}
