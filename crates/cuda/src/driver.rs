//! The `libcuda.so.1` symbol table, resolved once per process.
//!
//! Every entry point is resolved when the table is first needed, and `cuInit`
//! runs once. Entry points marked lazy are stored as `Result`s: a driver that
//! predates them still serves every other call and reports
//! [`Error::Symbol`] only when the lazy API is used.

use std::ffi::{CStr, c_char, c_void};
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

use libloading::Library;

use crate::{Error, Result};

pub(crate) type Handle = *mut c_void;
pub(crate) type CuResult = i32;
pub(crate) type HostFn = unsafe extern "C" fn(Handle);

pub(crate) const LIBCUDA: &str = "libcuda.so.1";

pub(crate) const CUDA_SUCCESS: CuResult = 0;
pub(crate) const CUDA_ERROR_INVALID_VALUE: CuResult = 1;
pub(crate) const CUDA_ERROR_NOT_READY: CuResult = 600;

/// `CUipcEventHandle`: an opaque 64-byte token passed by value.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RawIpcEventHandle {
    pub(crate) reserved: [u8; 64],
}

/// `CUmemLocation`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MemLocation {
    pub(crate) kind: i32,
    pub(crate) id: i32,
}

pub(crate) const CU_MEM_LOCATION_TYPE_DEVICE: i32 = 1;

/// `CUmemAllocationProp` (v1). `win32_metadata` and `flags` stay zero on Linux.
#[repr(C)]
pub(crate) struct MemAllocationProp {
    pub(crate) kind: i32,
    pub(crate) requested_handle_types: i32,
    pub(crate) location: MemLocation,
    pub(crate) win32_metadata: Handle,
    pub(crate) flags: [u8; 8],
}

pub(crate) const CU_MEM_ALLOCATION_TYPE_PINNED: i32 = 1;

impl MemAllocationProp {
    /// Device-resident physical memory exportable as `handle_types`.
    pub(crate) fn device(ordinal: i32, handle_types: i32) -> Self {
        Self {
            kind: CU_MEM_ALLOCATION_TYPE_PINNED,
            requested_handle_types: handle_types,
            location: MemLocation {
                kind: CU_MEM_LOCATION_TYPE_DEVICE,
                id: ordinal,
            },
            win32_metadata: std::ptr::null_mut(),
            flags: [0; 8],
        }
    }
}

/// `CUmemAccessDesc`.
#[repr(C)]
pub(crate) struct MemAccessDesc {
    pub(crate) location: MemLocation,
    pub(crate) flags: i32,
}

pub(crate) const CU_MEM_ACCESS_FLAGS_PROT_READ: i32 = 1;
pub(crate) const CU_MEM_ACCESS_FLAGS_PROT_READWRITE: i32 = 3;

/// `CUmemcpyAttributes` (v1).
#[repr(C)]
pub(crate) struct MemcpyAttributes {
    pub(crate) source_order: i32,
    pub(crate) source_hint: MemLocation,
    pub(crate) destination_hint: MemLocation,
    pub(crate) flags: u32,
}

/// The pointer arm of `CUmemcpy3DOperand`. The union's array arm
/// (`CUarray` + `CUoffset3D`) has the same 32-byte size and 8-byte alignment.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Memcpy3DOperand {
    pub(crate) kind: i32,
    pub(crate) pointer: u64,
    pub(crate) row_length: usize,
    pub(crate) layer_height: usize,
    pub(crate) hint: MemLocation,
}

pub(crate) const CU_MEMCPY_OPERAND_TYPE_POINTER: i32 = 1;

/// `CUDA_MEMCPY3D_BATCH_OP` (v1).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Memcpy3DBatchOp {
    pub(crate) source: Memcpy3DOperand,
    pub(crate) destination: Memcpy3DOperand,
    pub(crate) extent: [usize; 3],
    pub(crate) source_order: i32,
    pub(crate) flags: u32,
}

/// `CUDA_MEMCPY3D` (v2). The array handles and reserved pointers stay null:
/// operands are host or device pointers.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Memcpy3D {
    pub(crate) source_x: usize,
    pub(crate) source_y: usize,
    pub(crate) source_z: usize,
    pub(crate) source_lod: usize,
    pub(crate) source_kind: u32,
    pub(crate) source_host: usize,
    pub(crate) source_device: u64,
    pub(crate) source_array: usize,
    pub(crate) reserved0: usize,
    pub(crate) source_pitch: usize,
    pub(crate) source_height: usize,

    pub(crate) destination_x: usize,
    pub(crate) destination_y: usize,
    pub(crate) destination_z: usize,
    pub(crate) destination_lod: usize,
    pub(crate) destination_kind: u32,
    pub(crate) destination_host: usize,
    pub(crate) destination_device: u64,
    pub(crate) destination_array: usize,
    pub(crate) reserved1: usize,
    pub(crate) destination_pitch: usize,
    pub(crate) destination_height: usize,

    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) depth: usize,
}

/// `CUmemorytype` values of host and device pointers.
pub(crate) const CU_MEMORYTYPE_HOST: u32 = 1;
pub(crate) const CU_MEMORYTYPE_DEVICE: u32 = 2;

/// `CU_POINTER_ATTRIBUTE_MEMORY_TYPE`: the `CUmemorytype` of an address.
pub(crate) const CU_POINTER_ATTRIBUTE_MEMORY_TYPE: i32 = 2;

/// `CUdevResource` (v1): a type tag, 92 opaque bytes and a 48-byte union.
/// Only `smCount` of the SM arm is read; the driver owns the rest.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DevResource {
    pub(crate) kind: i32,
    pub(crate) internal: [u8; 92],
    pub(crate) sm_count: u32,
    pub(crate) reserved: [u8; 44],
}

impl DevResource {
    pub(crate) fn empty() -> Self {
        Self {
            kind: 0,
            internal: [0; 92],
            sm_count: 0,
            reserved: [0; 44],
        }
    }
}

pub(crate) const CU_DEV_RESOURCE_TYPE_SM: i32 = 1;

/// Declares the driver table: one typed field per entry point.
macro_rules! driver_table {
    (
        eager { $($field:ident = $symbol:literal: fn($($arg:ty),* $(,)?);)* }
        lazy { $($lazy:ident = $lazy_symbol:literal: fn($($lazy_arg:ty),* $(,)?);)* }
    ) => {
        pub(crate) struct Driver {
            $(pub(crate) $field: unsafe extern "C" fn($($arg),*) -> CuResult,)*
            $(pub(crate) $lazy: Result<unsafe extern "C" fn($($lazy_arg),*) -> CuResult>,)*
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
                        $($lazy: symbol(&library, $lazy_symbol),)*
                        _library: library,
                    })
                }
            }
        }
    };
}

driver_table! {
    eager {
        init = "cuInit": fn(u32);
        device_get = "cuDeviceGet": fn(*mut i32, i32);
        device_get_count = "cuDeviceGetCount": fn(*mut i32);
        device_get_attribute = "cuDeviceGetAttribute": fn(*mut i32, i32, i32);
        device_get_uuid = "cuDeviceGetUuid_v2": fn(*mut [u8; 16], i32);
        device_can_access_peer = "cuDeviceCanAccessPeer": fn(*mut i32, i32, i32);
        get_error_name = "cuGetErrorName": fn(CuResult, *mut *const c_char);
        get_error_string = "cuGetErrorString": fn(CuResult, *mut *const c_char);
        primary_ctx_retain = "cuDevicePrimaryCtxRetain": fn(*mut Handle, i32);
        primary_ctx_release = "cuDevicePrimaryCtxRelease_v2": fn(i32);
        primary_ctx_get_state = "cuDevicePrimaryCtxGetState": fn(i32, *mut u32, *mut i32);
        primary_ctx_reset = "cuDevicePrimaryCtxReset_v2": fn(i32);
        ctx_push = "cuCtxPushCurrent_v2": fn(Handle);
        ctx_pop = "cuCtxPopCurrent_v2": fn(*mut Handle);
        thread_exchange_capture_mode = "cuThreadExchangeStreamCaptureMode": fn(*mut i32);
        profiler_start = "cuProfilerStart": fn();
        profiler_stop = "cuProfilerStop": fn();

        stream_create_with_priority = "cuStreamCreateWithPriority": fn(*mut Handle, u32, i32);
        stream_get_priority = "cuStreamGetPriority": fn(Handle, *mut i32);
        stream_get_ctx = "cuStreamGetCtx": fn(Handle, *mut Handle);
        stream_is_capturing = "cuStreamIsCapturing": fn(Handle, *mut i32);
        stream_synchronize = "cuStreamSynchronize": fn(Handle);
        stream_destroy = "cuStreamDestroy_v2": fn(Handle);
        stream_wait_event = "cuStreamWaitEvent": fn(Handle, Handle, u32);
        launch_host_func = "cuLaunchHostFunc": fn(Handle, HostFn, Handle);
        graph_launch = "cuGraphLaunch": fn(Handle, Handle);

        device_get_dev_resource = "cuDeviceGetDevResource": fn(i32, *mut DevResource, i32);
        dev_sm_resource_split_by_count = "cuDevSmResourceSplitByCount": fn(
            *mut DevResource, *mut u32, *const DevResource, *mut DevResource, u32, u32,
        );
        dev_resource_generate_desc = "cuDevResourceGenerateDesc": fn(*mut Handle, *mut DevResource, u32);
        green_ctx_create = "cuGreenCtxCreate": fn(*mut Handle, Handle, i32, u32);
        green_ctx_destroy = "cuGreenCtxDestroy": fn(Handle);
        green_ctx_stream_create = "cuGreenCtxStreamCreate": fn(*mut Handle, Handle, u32, i32);
        green_ctx_get_dev_resource = "cuGreenCtxGetDevResource": fn(Handle, *mut DevResource, i32);
        stream_get_green_ctx = "cuStreamGetGreenCtx": fn(Handle, *mut Handle);
        ctx_get_dev_resource = "cuCtxGetDevResource": fn(Handle, *mut DevResource, i32);

        event_create = "cuEventCreate": fn(*mut Handle, u32);
        event_record_with_flags = "cuEventRecordWithFlags": fn(Handle, Handle, u32);
        event_query = "cuEventQuery": fn(Handle);
        event_synchronize = "cuEventSynchronize": fn(Handle);
        event_elapsed_time = "cuEventElapsedTime_v2": fn(*mut f32, Handle, Handle);
        event_destroy = "cuEventDestroy_v2": fn(Handle);
        ipc_get_event_handle = "cuIpcGetEventHandle": fn(*mut RawIpcEventHandle, Handle);
        ipc_open_event_handle = "cuIpcOpenEventHandle": fn(*mut Handle, RawIpcEventHandle);

        mem_alloc = "cuMemAlloc_v2": fn(*mut u64, usize);
        mem_free = "cuMemFree_v2": fn(u64);
        mem_get_info = "cuMemGetInfo_v2": fn(*mut usize, *mut usize);
        mem_host_alloc = "cuMemHostAlloc": fn(*mut Handle, usize, u32);
        mem_free_host = "cuMemFreeHost": fn(Handle);
        mem_host_register = "cuMemHostRegister_v2": fn(Handle, usize, u32);
        mem_host_unregister = "cuMemHostUnregister": fn(Handle);

        mem_get_allocation_granularity = "cuMemGetAllocationGranularity": fn(
            *mut usize, *const MemAllocationProp, i32,
        );
        mem_create = "cuMemCreate": fn(*mut u64, usize, *const MemAllocationProp, u64);
        mem_release = "cuMemRelease": fn(u64);
        mem_address_reserve = "cuMemAddressReserve": fn(*mut u64, usize, usize, u64, u64);
        mem_address_free = "cuMemAddressFree": fn(u64, usize);
        mem_map = "cuMemMap": fn(u64, usize, usize, u64, u64);
        mem_unmap = "cuMemUnmap": fn(u64, usize);
        mem_set_access = "cuMemSetAccess": fn(u64, usize, *const MemAccessDesc, usize);
        mem_export_to_shareable_handle = "cuMemExportToShareableHandle": fn(Handle, u64, i32, u64);
        mem_import_from_shareable_handle = "cuMemImportFromShareableHandle": fn(*mut u64, Handle, i32);
        mem_retain_allocation_handle = "cuMemRetainAllocationHandle": fn(*mut u64, Handle);

        memcpy_async = "cuMemcpyAsync": fn(u64, u64, usize, Handle);
        memcpy_3d_async = "cuMemcpy3DAsync_v2": fn(*const Memcpy3D, Handle);
        pointer_get_attribute = "cuPointerGetAttribute": fn(*mut c_void, i32, u64);
        memset_d8_async = "cuMemsetD8Async": fn(u64, u8, usize, Handle);
        memset_d16_async = "cuMemsetD16Async": fn(u64, u16, usize, Handle);
        memset_d32_async = "cuMemsetD32Async": fn(u64, u32, usize, Handle);
        memset_d2d8_async = "cuMemsetD2D8Async": fn(u64, usize, u8, usize, usize, Handle);
        memset_d2d32_async = "cuMemsetD2D32Async": fn(u64, usize, u32, usize, usize, Handle);
    }
    lazy {
        // CUDA 12.8 batch APIs; the CUDA 13 `_v2` signatures have no failIdx.
        memcpy_batch_async = "cuMemcpyBatchAsync_v2": fn(
            *mut u64, *mut u64, *mut usize, usize, *mut MemcpyAttributes, *mut usize, usize, Handle,
        );
        memcpy_3d_batch_async = "cuMemcpy3DBatchAsync_v2": fn(usize, *mut Memcpy3DBatchOp, u64, Handle);
    }
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

/// Returns the process driver table, loading it and running `cuInit` on first
/// use. A failed load is cached and returned to every caller.
pub(crate) fn driver() -> Result<&'static Driver> {
    DRIVER
        .get_or_init(|| {
            // The driver is installed under its stable system soname; no
            // toolkit or user path is involved.
            // SAFETY: loading libcuda runs only the driver's own initializers.
            let library = unsafe { Library::new(LIBCUDA) }.map_err(|source| Error::Load {
                library: LIBCUDA,
                source: Arc::new(source),
            })?;
            let driver = Driver::resolve(library)?;

            // SAFETY: cuInit takes flags by value and must be passed zero.
            let status = unsafe { (driver.init)(0) };
            driver.check(status, "cuInit")?;
            Ok(driver)
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Returns the process driver table if an earlier [`driver`] call loaded and
/// initialized it, without loading it otherwise.
pub(crate) fn initialized() -> Option<&'static Driver> {
    DRIVER.get().and_then(|driver| driver.as_ref().ok())
}

impl Driver {
    /// Converts a `CUresult` into a typed error naming the failed call.
    pub(crate) fn check(&self, status: CuResult, call: &'static str) -> Result<()> {
        if status == CUDA_SUCCESS {
            return Ok(());
        }

        Err(Error::Driver {
            call,
            code: status,
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
            if lookup(status, &mut text) != CUDA_SUCCESS || text.is_null() {
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
