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
//! The subscription lasts as long as the process: the collectors share it,
//! so none ends it, and a collector that stops disables what it enabled.
//!
//! The process loads the CUPTI of CUDA 13 by the soname its package installs
//! on the loader's default search path ([`library`]), and refuses one older
//! than [`MINIMUM_VERSION`] with [`Error::Version`]. The same library is
//! NVTX's injection: [`initialize_nvtx_injection`] is the entry an executable
//! that emits NVTX ranges registers with NVTX, so the ranges reach this CUPTI
//! without the process naming a library in its environment.

pub mod activity;
pub mod blocking;
pub mod metrics;
pub mod pc_sampling;
pub mod pm_sampling;
pub mod range_profiler;
pub mod records;
pub mod subscriber;
pub mod timeline;

use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::{Arc, OnceLock};

use libloading::Library;

use crate::sys::cupti as sys;
use crate::{Error, Result};

/// CUPTI of CUDA 13, by its soname.
const LIBCUPTI: &str = "libcupti.so.13";

/// The oldest CUPTI the process accepts, by the `CUPTI_API_VERSION` the
/// loaded library reports: release 13.4.92 (130401).
///
/// The bindings are generated from the CUPTI headers of CUDA 13.0
/// ([`sys::CUPTI_API_VERSION`]), whose entry points and records later
/// releases keep, but the earlier releases of CUDA 13 fail the work CUPTI
/// records. CUDA 13.0's own CUPTI (2025.3.0, 130000) faults inside
/// `cuGraphLaunch` once graph launches traced by node fill the device buffers
/// it pre-allocates, and stalls a serving worker that replays CUDA graphs
/// while kernels are recorded. Release 13.0.85 (130001) has the device write
/// software kernel timestamps into host records its worker thread may already
/// have freed, which on GB200 faults the device front end (Xid 31). Release
/// 13.4.92 records every kernel, copy and fill of 1,024 launches of a
/// 1,000-kernel graph with its timestamps.
pub const MINIMUM_VERSION: u32 = 130_401;

/// What a refused CUPTI is told to install.
const CUPTI_PACKAGE: &str = "CUPTI 13.4.92 or later (NVIDIA's cuda-cupti package of CUDA 13)";

/// The CUPTI library this process loads, by soname.
pub fn library() -> &'static str {
    LIBCUPTI
}

pub(crate) type CuptiResult = sys::CUptiResult;
pub(crate) type Callback = unsafe extern "C" fn(*mut c_void, u32, u32, *const c_void);
pub(crate) type BufferRequested = unsafe extern "C" fn(*mut *mut u8, *mut usize, *mut usize);
pub(crate) type BufferCompleted = unsafe extern "C" fn(*mut c_void, u32, *mut u8, usize, usize);
/// A profiling entry point, which takes its parameter struct.
pub(crate) type Entry = unsafe extern "C" fn(*mut c_void) -> CuptiResult;

/// Declares the CUPTI table: one typed field per entry point, every one
/// resolved at load (each is an entry point of every release from
/// [`MINIMUM_VERSION`] on).
macro_rules! cupti_table {
    (
        api { $($field:ident = $symbol:literal: fn($($arg:ty),* $(,)?);)* }
        profiling { $($entry:ident = $entry_symbol:literal;)* }
    ) => {
        /// The `libcupti.so.13` entry points this crate calls.
        pub(crate) struct Cupti {
            $(pub(crate) $field: unsafe extern "C" fn($($arg),*) -> CuptiResult,)*
            $(pub(crate) $entry: Entry,)*
            /// `InitializeInjectionNvtx2`, NVTX's injection entry.
            nvtx_injection: unsafe extern "C" fn(*const c_void) -> c_int,
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
                        $($entry: symbol(&library, $entry_symbol)?,)*
                        nvtx_injection: symbol(&library, "InitializeInjectionNvtx2")?,
                        _library: library,
                    })
                }
            }
        }
    };
}

cupti_table! {
    api {
        subscribe = "cuptiSubscribe": fn(*mut *mut c_void, Callback, *mut c_void);
        enable_callback = "cuptiEnableCallback": fn(u32, *mut c_void, u32, u32);
        enable_domain = "cuptiEnableDomain": fn(u32, *mut c_void, u32);
        register_buffers = "cuptiActivityRegisterCallbacks": fn(BufferRequested, BufferCompleted);
        activity_enable = "cuptiActivityEnable": fn(u32);
        activity_disable = "cuptiActivityDisable": fn(u32);
        flush_all = "cuptiActivityFlushAll": fn(u32);
        flush_period = "cuptiActivityFlushPeriod": fn(u32);
        next_record = "cuptiActivityGetNextRecord": fn(*mut u8, usize, *mut *mut u8);
        stream_id = "cuptiGetStreamIdEx": fn(*mut c_void, *mut c_void, u8, *mut u32);
        context_id = "cuptiGetContextId": fn(*mut c_void, *mut u32);
        dropped_records = "cuptiActivityGetNumDroppedRecords": fn(*mut c_void, u32, *mut usize);
        timestamp = "cuptiGetTimestamp": fn(*mut u64);
        version = "cuptiGetVersion": fn(*mut u32);
    }
    profiling {
        profiler_initialize = "cuptiProfilerInitialize";
        chip_name = "cuptiDeviceGetChipName";
        counter_availability = "cuptiProfilerGetCounterAvailability";
        host_initialize = "cuptiProfilerHostInitialize";
        host_deinitialize = "cuptiProfilerHostDeinitialize";
        host_add_metrics = "cuptiProfilerHostConfigAddMetrics";
        host_config_size = "cuptiProfilerHostGetConfigImageSize";
        host_config = "cuptiProfilerHostGetConfigImage";
        host_passes = "cuptiProfilerHostGetNumOfPasses";
        host_evaluate = "cuptiProfilerHostEvaluateToGpuValues";
        range_enable = "cuptiRangeProfilerEnable";
        range_disable = "cuptiRangeProfilerDisable";
        range_set_config = "cuptiRangeProfilerSetConfig";
        range_start = "cuptiRangeProfilerStart";
        range_stop = "cuptiRangeProfilerStop";
        range_push = "cuptiRangeProfilerPushRange";
        range_pop = "cuptiRangeProfilerPopRange";
        range_decode = "cuptiRangeProfilerDecodeData";
        range_data_size = "cuptiRangeProfilerGetCounterDataSize";
        range_data_initialize = "cuptiRangeProfilerCounterDataImageInitialize";
        range_data_info = "cuptiRangeProfilerGetCounterDataInfo";
        range_info = "cuptiRangeProfilerCounterDataGetRangeInfo";
        pm_enable = "cuptiPmSamplingEnable";
        pm_disable = "cuptiPmSamplingDisable";
        pm_set_config = "cuptiPmSamplingSetConfig";
        pm_start = "cuptiPmSamplingStart";
        pm_stop = "cuptiPmSamplingStop";
        pm_decode = "cuptiPmSamplingDecodeData";
        pm_availability = "cuptiPmSamplingGetCounterAvailability";
        pm_data_size = "cuptiPmSamplingGetCounterDataSize";
        pm_data_initialize = "cuptiPmSamplingCounterDataImageInitialize";
        pm_data_info = "cuptiPmSamplingGetCounterDataInfo";
        pm_sample_info = "cuptiPmSamplingCounterDataGetSampleInfo";
        pc_enable = "cuptiPCSamplingEnable";
        pc_disable = "cuptiPCSamplingDisable";
        pc_configure = "cuptiPCSamplingSetConfigurationAttribute";
        pc_start = "cuptiPCSamplingStart";
        pc_stop = "cuptiPCSamplingStop";
        pc_get_data = "cuptiPCSamplingGetData";
        pc_stall_count = "cuptiPCSamplingGetNumStallReasons";
        pc_stall_reasons = "cuptiPCSamplingGetStallReasons";
    }
}

// SAFETY: the table holds function pointers into a library that is never
// unloaded while the table lives; CUPTI's entry points are thread-safe.
unsafe impl Send for Cupti {}
// SAFETY: as above.
unsafe impl Sync for Cupti {}

static CUPTI: OnceLock<Result<Cupti>> = OnceLock::new();

/// The process's CUPTI table, loading the library and checking its version on
/// first use. A failed load is cached and returned to every caller.
pub(crate) fn cupti() -> Result<&'static Cupti> {
    CUPTI
        .get_or_init(|| {
            // SAFETY: loading libcupti runs only CUPTI's own initializers.
            let library = unsafe { Library::new(LIBCUPTI) }.map_err(|source| Error::Load {
                library: LIBCUPTI,
                source: Arc::new(source),
            })?;
            let cupti = Cupti::resolve(library)?;

            let mut version = 0;
            // SAFETY: CUPTI writes one value.
            check(unsafe { (cupti.version)(&mut version) }, "cuptiGetVersion")?;
            require_version(version)?;
            Ok(cupti)
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Refuses a CUPTI whose `CUPTI_API_VERSION` is older than
/// [`MINIMUM_VERSION`].
fn require_version(version: u32) -> Result<()> {
    if version >= MINIMUM_VERSION {
        return Ok(());
    }
    Err(Error::Version {
        library: LIBCUPTI,
        found: format!("CUPTI API version {version}"),
        required: format!("CUPTI API version {MINIMUM_VERSION}"),
        package: CUPTI_PACKAGE,
    })
}

/// NVTX's injection entry (`InitializeInjectionNvtx2`) for the CUPTI this
/// process loads: NVTX calls it at the process's first NVTX call with its
/// export table, and from then on hands its marks and ranges to CUPTI, which
/// records them in the timeline's activity records.
///
/// An executable that emits NVTX ranges registers this entry with NVTX's
/// static injection (the `InitializeInjectionNvtx2_fnptr` symbol of NVTX's
/// load sequence), which NVTX consults after a library named in its
/// environment, so a profiler that injects its own library keeps the marks.
/// Answers 0, NVTX's failure, when CUPTI is missing or refused; NVTX then
/// emits nothing.
///
/// # Safety
///
/// Called only by NVTX's loader, with the export table function it passes.
pub unsafe extern "C" fn initialize_nvtx_injection(get_export_table: *const c_void) -> c_int {
    match cupti() {
        // SAFETY: forwarded from NVTX's loader, which passes its export table.
        Ok(cupti) => unsafe { (cupti.nvtx_injection)(get_export_table) },
        Err(_) => 0,
    }
}

/// # Safety
/// `T` must be the function-pointer type of the named symbol's C prototype.
unsafe fn symbol<T: Copy>(loaded: &Library, name: &'static str) -> Result<T> {
    // SAFETY: forwarded from the caller; the table keeps the library alive.
    unsafe { loaded.get::<T>(name.as_bytes()) }
        .map(|symbol| *symbol)
        .map_err(|source| Error::Symbol {
            library: LIBCUPTI,
            symbol: name,
            source: Arc::new(source),
        })
}

pub(crate) fn check(code: CuptiResult, call: &'static str) -> Result<()> {
    if code == sys::CUPTI_SUCCESS {
        Ok(())
    } else {
        // `CUptiResult` is a C enum; the error carries the value CUPTI documents.
        Err(Error::Cupti {
            call,
            code: code as i32,
        })
    }
}

/// Driver calls as `(callback id, name)` pairs: each id is the generated
/// `CUPTI_DRIVER_TRACE_CBID_<name>` of the call's name, so a call the headers
/// do not declare fails to compile and the name a collector reports is the
/// call's own.
macro_rules! driver_calls {
    ($($call:ident),* $(,)?) => {
        &[$((
            paste::paste!(crate::sys::cupti::[<CUPTI_DRIVER_TRACE_CBID_ $call>]),
            stringify!($call),
        )),*]
    };
}
pub(crate) use driver_calls;

/// Calls a profiling entry point with its parameter struct.
///
/// # Safety
/// `params` is the entry point's parameter struct, with every field the
/// call reads set.
pub(crate) unsafe fn call<P>(entry: &Entry, name: &'static str, params: &mut P) -> Result<()> {
    // SAFETY: forwarded from the caller.
    check(unsafe { entry((params as *mut P).cast()) }, name)
}

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

    use crate::{
        Capture, CaptureMode, Device, DeviceAllocation, Function, Graph, LaunchConfig, Library,
        Result, Stream,
    };

    /// CUPTI is process-wide: its device tests run one at a time.
    pub(crate) static SERIAL: Mutex<()> = Mutex::new(());

    /// The spin kernel's PTX (`spin.ptx`): every thread counts to its second
    /// parameter and stores the count through its first.
    const SPIN: &[u8] = include_bytes!("spin.ptx");

    /// The spin kernel of `stream`'s device, and the word it stores into.
    pub(crate) struct Spin {
        _library: Library,
        function: Function,
        out: DeviceAllocation,
        /// The device's multiprocessors: one block on each keeps the whole
        /// device busy.
        multiprocessors: u32,
    }

    impl Spin {
        pub(crate) fn load(stream: &Stream) -> Result<Self> {
            let library = Library::load(&Device::new(stream.device())?, SPIN)?;
            let function = library.kernel("spin")?;
            Ok(Self {
                _library: library,
                function,
                out: DeviceAllocation::linear(stream.device(), 4)?,
                multiprocessors: stream.device().multiprocessors()?,
            })
        }

        /// Launch it on `stream` with one block on each multiprocessor of the
        /// device, every thread counting to `count`.
        pub(crate) fn occupy(&self, stream: &Stream, count: u32) -> Result<()> {
            self.launch(stream, self.multiprocessors, count)
        }

        /// Launch it on `stream` with `blocks` blocks of 128 threads that
        /// each count to `count`.
        pub(crate) fn launch(&self, stream: &Stream, blocks: u32, mut count: u32) -> Result<()> {
            let mut out = self.out.address();
            let mut parameters: [*mut c_void; 2] = [(&raw mut out).cast(), (&raw mut count).cast()];
            let config = LaunchConfig::new([blocks, 1, 1], [128, 1, 1]);
            // The parameters are the kernel's, and the word outlives the
            // launch, which the tests synchronize.
            self.function.launch_ex(stream, &config, &mut parameters)
        }
    }

    /// Keep the captured source graph alive while `record` instantiates and
    /// exercises it. The caller synchronizes its launches before returning.
    pub(crate) fn with_captured_graph<T>(
        stream: &Stream,
        issue: impl FnOnce(&Stream) -> Result<()>,
        record: impl FnOnce(Graph) -> Result<T>,
    ) -> Result<T> {
        let capture = Capture::begin(stream, CaptureMode::Relaxed)?;
        issue(stream)?;
        record(capture.end()?)
    }

    /// A CUPTI older than the minimum — CUDA 13.0's own, which faults the
    /// graph replays it records — is refused by name with the package to
    /// install; the minimum and later releases are accepted.
    #[test]
    fn a_cupti_older_than_the_minimum_is_refused() {
        assert!(super::require_version(super::MINIMUM_VERSION).is_ok());

        let error = super::require_version(130_000).expect_err("CUDA 13.0's CUPTI is refused");
        let text = error.to_string();
        assert!(
            matches!(
                error,
                crate::Error::Version {
                    library: super::LIBCUPTI,
                    ..
                }
            ),
            "{text}"
        );
        assert!(
            text.contains("130000") && text.contains(super::CUPTI_PACKAGE),
            "{text}"
        );
    }
}
