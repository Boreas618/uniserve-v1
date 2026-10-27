//! NVIDIA's CUPTI, loaded at runtime: one subscriber per process, shared by
//! every collector that observes the process's CUDA work.
//!
//! CUPTI admits one subscriber of its callback API per process, and one set
//! of activity buffer callbacks. A process therefore subscribes once
//! ([`subscriber::subscribe`]) and every collector reaches CUPTI through that
//! subscription:
//!
//! - [`blocking`]: the driver calls that make a marked thread wait for the
//!   device, counted by thread;
//! - [`activity`]: the device time of registered streams, from activity
//!   records;
//! - [`timeline`]: every activity record and every runtime and driver API
//!   call, delivered as decoded [`records`] to a recording's sink;
//! - [`pm_sampling`], [`range_profiler`] and [`pc_sampling`]: hardware
//!   counters sampled over time, counters per kernel or per user range in
//!   one pass, and warp program counters with their stall reasons; the
//!   counter configurations and their evaluation are [`metrics`].
//!
//! A profiler that injected CUPTI (Nsight Systems) holds the subscription:
//! [`subscriber::subscribe`] then fails with [`Error::Cupti`] (code 39) and
//! none of these collectors is available.
//!
//! Nothing here calls `cuptiFinalize`: after CUPTI recorded a serving
//! worker's activity, it left the device work submitted afterwards waiting
//! forever (CUPTI 2025.3.0 of the CUDA 13.0 toolkit, replaying CUDA graphs).
//! A collector that stops disables what it enabled; the library stays
//! loaded.
//!
//! The process loads the CUPTI beside the CUDA runtime it already loaded
//! ([`library`]): CUPTI belongs to a CUDA release, and both the toolkit and
//! the Python environment's CUDA packages keep `libcupti.so.N` beside
//! `libcudart.so.N`. A worker thereby loads the CUPTI its environment pins
//! with the runtime PyTorch runs on. The toolkit's CUPTI 2025.3.0 stalled a
//! serving worker that replayed CUDA graphs while kernels were recorded,
//! within a minute of load; the environment's `nvidia-cuda-cupti` 13.0.85
//! served the same load.

pub mod activity;
pub mod blocking;
pub mod metrics;
pub mod pc_sampling;
pub mod pm_sampling;
pub mod range_profiler;
pub mod records;
pub mod subscriber;
pub mod timeline;

use std::ffi::{CStr, c_char, c_void};
use std::sync::{Arc, OnceLock};

use libloading::Library;

use crate::{Error, Result};

/// The CUPTI library of the loader's search path, for a process that
/// loaded no CUDA runtime.
const LIBCUPTI: &str = "libcupti.so";

/// The CUPTI library this process loads: the CUPTI the process named as its
/// NVTX injection library (`NVTX_INJECTION64_PATH`), which NVTX already
/// loaded; else `libcupti.so.N` in the directory of the `libcudart.so.N` the
/// process loaded, when that file exists; else [`LIBCUPTI`]. Resolved once,
/// at the first load.
pub fn library() -> &'static str {
    static LIBRARY: OnceLock<String> = OnceLock::new();
    LIBRARY.get_or_init(|| {
        injected()
            .or_else(beside_runtime)
            .unwrap_or_else(|| LIBCUPTI.to_owned())
    })
}

/// The CUPTI `NVTX_INJECTION64_PATH` names, if it names an existing
/// `libcupti.so*` (a profiler's own injection library is not CUPTI).
fn injected() -> Option<String> {
    let path = std::env::var("NVTX_INJECTION64_PATH").ok()?;
    let named = std::path::Path::new(&path);
    let file = named.file_name()?.to_string_lossy();
    (file.starts_with("libcupti.so") && named.is_file()).then_some(path)
}

/// `libcupti.so.N` beside the first `libcudart.so.N` among the objects the
/// process loaded, if it exists.
fn beside_runtime() -> Option<String> {
    unsafe extern "C" fn visit(
        info: *mut libc::dl_phdr_info,
        _size: usize,
        found: *mut c_void,
    ) -> libc::c_int {
        // SAFETY: the loader passes a valid entry, whose name is a C string
        // or null, and `found` is the caller's `Option<String>`.
        let (name, found) = unsafe { ((*info).dlpi_name, &mut *found.cast::<Option<String>>()) };
        if name.is_null() {
            return 0;
        }
        // SAFETY: as above.
        let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
        let Some((directory, file)) = name.rsplit_once('/') else {
            return 0;
        };
        let Some(version) = file.strip_prefix("libcudart.so.") else {
            return 0;
        };
        let major = version.split('.').next().unwrap_or(version);
        *found = Some(format!("{directory}/libcupti.so.{major}"));
        1
    }

    let mut found: Option<String> = None;
    // SAFETY: the callback only reads the loader's entries and writes the
    // local, which outlives the call.
    unsafe { libc::dl_iterate_phdr(Some(visit), (&raw mut found).cast()) };
    found.filter(|path| std::path::Path::new(path).is_file())
}

pub(crate) type CuptiResult = i32;
pub(crate) type Callback = unsafe extern "C" fn(*mut c_void, i32, u32, *const c_void);
pub(crate) type BufferRequested = unsafe extern "C" fn(*mut *mut u8, *mut usize, *mut usize);
pub(crate) type BufferCompleted = unsafe extern "C" fn(*mut c_void, u32, *mut u8, usize, usize);

/// `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`: another CUPTI client,
/// such as a profiler, holds the process's subscription.
pub const MULTIPLE_SUBSCRIBERS: i32 = 39;

/// Declares the CUPTI table: one typed field per entry point. The profiling
/// entry points are lazy (`Result`s), so that a CUPTI without them still
/// serves the callback and activity APIs and fails only the collector that
/// needs them.
macro_rules! cupti_table {
    (
        eager { $($field:ident = $symbol:literal: fn($($arg:ty),* $(,)?);)* }
        lazy { $($lazy:ident = $lazy_symbol:literal: fn($($lazy_arg:ty),* $(,)?);)* }
    ) => {
        /// The `libcupti.so` entry points this crate calls.
        pub(crate) struct Cupti {
            $(pub(crate) $field: unsafe extern "C" fn($($arg),*) -> CuptiResult,)*
            $(pub(crate) $lazy: Result<unsafe extern "C" fn($($lazy_arg),*) -> CuptiResult>,)*
            // Declared last: the library outlives every copied function pointer.
            _library: Library,
        }

        impl Cupti {
            fn resolve(library: Library) -> Result<Self> {
                // SAFETY: every signature is the C prototype of the named
                // entry point in the CUPTI 13 headers.
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

cupti_table! {
    eager {
        subscribe = "cuptiSubscribe": fn(*mut *mut c_void, Callback, *mut c_void);
        enable_callback = "cuptiEnableCallback": fn(u32, *mut c_void, i32, u32);
        enable_domain = "cuptiEnableDomain": fn(u32, *mut c_void, i32);
        register_buffers = "cuptiActivityRegisterCallbacks": fn(BufferRequested, BufferCompleted);
        activity_enable = "cuptiActivityEnable": fn(u32);
        activity_disable = "cuptiActivityDisable": fn(u32);
        flush_all = "cuptiActivityFlushAll": fn(u32);
        next_record = "cuptiActivityGetNextRecord": fn(*mut u8, usize, *mut *mut u8);
        stream_id = "cuptiGetStreamIdEx": fn(*mut c_void, *mut c_void, u8, *mut u32);
        context_id = "cuptiGetContextId": fn(*mut c_void, *mut u32);
        dropped_records = "cuptiActivityGetNumDroppedRecords": fn(*mut c_void, u32, *mut usize);
        timestamp = "cuptiGetTimestamp": fn(*mut u64);
        version = "cuptiGetVersion": fn(*mut u32);
    }
    lazy {
        profiler_initialize = "cuptiProfilerInitialize": fn(*mut c_void);
        chip_name = "cuptiDeviceGetChipName": fn(*mut c_void);
        counter_availability = "cuptiProfilerGetCounterAvailability": fn(*mut c_void);
        host_initialize = "cuptiProfilerHostInitialize": fn(*mut c_void);
        host_deinitialize = "cuptiProfilerHostDeinitialize": fn(*mut c_void);
        host_add_metrics = "cuptiProfilerHostConfigAddMetrics": fn(*mut c_void);
        host_config_size = "cuptiProfilerHostGetConfigImageSize": fn(*mut c_void);
        host_config = "cuptiProfilerHostGetConfigImage": fn(*mut c_void);
        host_passes = "cuptiProfilerHostGetNumOfPasses": fn(*mut c_void);
        host_evaluate = "cuptiProfilerHostEvaluateToGpuValues": fn(*mut c_void);
        range_enable = "cuptiRangeProfilerEnable": fn(*mut c_void);
        range_disable = "cuptiRangeProfilerDisable": fn(*mut c_void);
        range_set_config = "cuptiRangeProfilerSetConfig": fn(*mut c_void);
        range_start = "cuptiRangeProfilerStart": fn(*mut c_void);
        range_stop = "cuptiRangeProfilerStop": fn(*mut c_void);
        range_push = "cuptiRangeProfilerPushRange": fn(*mut c_void);
        range_pop = "cuptiRangeProfilerPopRange": fn(*mut c_void);
        range_decode = "cuptiRangeProfilerDecodeData": fn(*mut c_void);
        range_data_size = "cuptiRangeProfilerGetCounterDataSize": fn(*mut c_void);
        range_data_initialize = "cuptiRangeProfilerCounterDataImageInitialize": fn(*mut c_void);
        range_data_info = "cuptiRangeProfilerGetCounterDataInfo": fn(*mut c_void);
        range_info = "cuptiRangeProfilerCounterDataGetRangeInfo": fn(*mut c_void);
        pm_enable = "cuptiPmSamplingEnable": fn(*mut c_void);
        pm_disable = "cuptiPmSamplingDisable": fn(*mut c_void);
        pm_set_config = "cuptiPmSamplingSetConfig": fn(*mut c_void);
        pm_start = "cuptiPmSamplingStart": fn(*mut c_void);
        pm_stop = "cuptiPmSamplingStop": fn(*mut c_void);
        pm_decode = "cuptiPmSamplingDecodeData": fn(*mut c_void);
        pm_availability = "cuptiPmSamplingGetCounterAvailability": fn(*mut c_void);
        pm_data_size = "cuptiPmSamplingGetCounterDataSize": fn(*mut c_void);
        pm_data_initialize = "cuptiPmSamplingCounterDataImageInitialize": fn(*mut c_void);
        pm_data_info = "cuptiPmSamplingGetCounterDataInfo": fn(*mut c_void);
        pm_sample_info = "cuptiPmSamplingCounterDataGetSampleInfo": fn(*mut c_void);
        pc_enable = "cuptiPCSamplingEnable": fn(*mut c_void);
        pc_disable = "cuptiPCSamplingDisable": fn(*mut c_void);
        pc_configure = "cuptiPCSamplingSetConfigurationAttribute": fn(*mut c_void);
        pc_start = "cuptiPCSamplingStart": fn(*mut c_void);
        pc_stop = "cuptiPCSamplingStop": fn(*mut c_void);
        pc_get_data = "cuptiPCSamplingGetData": fn(*mut c_void);
        pc_stall_count = "cuptiPCSamplingGetNumStallReasons": fn(*mut c_void);
        pc_stall_reasons = "cuptiPCSamplingGetStallReasons": fn(*mut c_void);
    }
}

// SAFETY: the table holds function pointers into a library that is never
// unloaded while the table lives; CUPTI's entry points are thread-safe.
unsafe impl Send for Cupti {}
// SAFETY: as above.
unsafe impl Sync for Cupti {}

static CUPTI: OnceLock<Result<Cupti>> = OnceLock::new();

/// The process's CUPTI table, loading the library on first use.
pub(crate) fn cupti() -> Result<&'static Cupti> {
    CUPTI
        .get_or_init(|| {
            // SAFETY: loading libcupti runs only CUPTI's own initializers.
            let library = unsafe { Library::new(library()) }.map_err(|source| Error::Load {
                library: library(),
                source: Arc::new(source),
            })?;
            Cupti::resolve(library)
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// # Safety
/// `T` must be the function-pointer type of the named symbol's C prototype.
unsafe fn symbol<T: Copy>(loaded: &Library, name: &'static str) -> Result<T> {
    // SAFETY: forwarded from the caller; the table keeps the library alive.
    unsafe { loaded.get::<T>(name.as_bytes()) }
        .map(|symbol| *symbol)
        .map_err(|source| Error::Symbol {
            library: library(),
            symbol: name,
            source: Arc::new(source),
        })
}

pub(crate) fn check(code: CuptiResult, call: &'static str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Cupti { call, code })
    }
}

/// Calls a lazy entry point with its parameter struct: fails when CUPTI
/// lacks the entry point or the call fails.
///
/// # Safety
/// `params` is the entry point's parameter struct, with every field the
/// call reads set.
pub(crate) unsafe fn call<P>(
    entry: &Result<unsafe extern "C" fn(*mut c_void) -> CuptiResult>,
    name: &'static str,
    params: &mut P,
) -> Result<()> {
    let entry = entry.as_ref().map_err(Clone::clone)?;
    // SAFETY: forwarded from the caller.
    check(unsafe { entry((params as *mut P).cast()) }, name)
}

/// CUPTI's current timestamp, in nanoseconds: the clock of every activity
/// record, PM sample and API record CUPTI and this crate produce.
pub fn timestamp() -> Result<u64> {
    let cupti = cupti()?;
    let mut now = 0;
    // SAFETY: CUPTI writes one value.
    check(unsafe { (cupti.timestamp)(&mut now) }, "cuptiGetTimestamp")?;
    Ok(now)
}

/// The CUPTI API version (`CUPTI_API_VERSION` of the loaded library).
pub fn version() -> Result<u32> {
    let cupti = cupti()?;
    let mut version = 0;
    // SAFETY: CUPTI writes one value.
    check(unsafe { (cupti.version)(&mut version) }, "cuptiGetVersion")?;
    Ok(version)
}

/// A C string CUPTI owns, copied; `None` for a null pointer.
///
/// # Safety
/// `text` is null or a NUL-terminated string valid during the call.
pub(crate) unsafe fn text(text: *const c_char) -> Option<String> {
    if text.is_null() {
        return None;
    }
    // SAFETY: forwarded from the caller.
    Some(
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// The size CUPTI's profiler parameter structs state in `structSize`: the
/// offset of their last field plus its size (`CUPTI_PROFILER_STRUCT_SIZE`).
macro_rules! struct_size {
    ($type:ty, $last:ident) => {{
        let probe = std::mem::MaybeUninit::<$type>::uninit();
        let base = probe.as_ptr();
        // SAFETY: only the field's address is taken, never its value.
        let field = unsafe { &raw const (*base).$last };
        const fn size_of_pointee<T>(_: *const T) -> usize {
            std::mem::size_of::<T>()
        }
        (field as usize - base as usize) + size_of_pointee(field)
    }};
}
pub(crate) use struct_size;

#[cfg(test)]
pub(crate) mod tests {
    use std::ffi::c_void;
    use std::sync::Mutex;

    use crate::{DeviceAllocation, Function, Module, Result, Stream};

    /// CUPTI is process-wide: its device tests run one at a time.
    pub(crate) static SERIAL: Mutex<()> = Mutex::new(());

    /// A kernel whose every thread counts to its second parameter and stores
    /// the count through its first.
    const SPIN: &str = ".version 8.0\n.target sm_90\n.address_size 64\n\
        .visible .entry spin(.param .u64 out, .param .u32 n) {\n\
        .reg .u64 %rd<2>; .reg .u32 %r<3>; .reg .pred %p;\n\
        ld.param.u64 %rd1, [out]; ld.param.u32 %r1, [n]; mov.u32 %r2, 0;\n\
        LOOP: add.u32 %r2, %r2, 1; setp.lt.u32 %p, %r2, %r1; @%p bra LOOP;\n\
        st.global.u32 [%rd1], %r2; ret; }\n";

    /// The spin kernel of `stream`'s device, and the word it stores into.
    pub(crate) struct Spin {
        _module: Module,
        function: Function,
        out: DeviceAllocation,
    }

    impl Spin {
        pub(crate) fn load(stream: &Stream) -> Result<Self> {
            let module = Module::load_ptx(stream.device(), SPIN)?;
            let function = module.function("spin")?;
            Ok(Self {
                _module: module,
                function,
                out: DeviceAllocation::linear(stream.device(), 4)?,
            })
        }

        /// Launch it on `stream` with `blocks` blocks of 128 threads that
        /// each count to `count`.
        pub(crate) fn launch(&self, stream: &Stream, blocks: u32, mut count: u32) -> Result<()> {
            let mut out = self.out.address();
            let mut parameters: [*mut c_void; 2] = [(&raw mut out).cast(), (&raw mut count).cast()];
            // SAFETY: the parameters are the kernel's; the word outlives the
            // launch, which the tests synchronize.
            unsafe { self.function.launch(stream, blocks, 128, &mut parameters) }
        }
    }

    /// A graph of what `issue` issues on `stream`, captured and instantiated.
    pub(crate) fn captured(
        stream: &Stream,
        issue: impl FnOnce(&Stream) -> Result<()>,
    ) -> Result<crate::GraphExec> {
        type Capture = unsafe extern "C" fn(*mut c_void, i32) -> i32;
        type Captured = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32;
        type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;
        // SAFETY: loading the driver runs only its initializers; each type is
        // its symbol's C prototype.
        let (begin, end, destroy) = unsafe {
            let driver = Box::leak(Box::new(libloading::Library::new("libcuda.so.1").unwrap()));
            (
                *driver.get::<Capture>(b"cuStreamBeginCapture_v2").unwrap(),
                *driver.get::<Captured>(b"cuStreamEndCapture").unwrap(),
                *driver.get::<Destroy>(b"cuGraphDestroy").unwrap(),
            )
        };
        let _context = stream.enter()?;
        let mut graph = std::ptr::null_mut();
        // SAFETY: the stream is live; `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`
        // confines the capture to this thread, and the driver writes the
        // captured graph, which is destroyed once instantiated.
        unsafe {
            assert_eq!(begin(stream.handle(), 1), 0);
            issue(stream)?;
            assert_eq!(end(stream.handle(), &mut graph), 0);
            let graph_exec = crate::GraphExec::instantiate(stream.device(), graph as usize);
            assert_eq!(destroy(graph), 0);
            graph_exec
        }
    }
}
