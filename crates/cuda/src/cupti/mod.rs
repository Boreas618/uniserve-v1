//! Driver calls that make the host wait for the device, observed on marked
//! threads through CUPTI's callback API.
//!
//! [`BlockingWatch::start`] loads CUPTI and subscribes the process
//! to every driver call that blocks its calling thread until device work
//! completes: stream, event and context synchronization, synchronous
//! copies, `cuMemFree` and `cuMemFreeHost` (which the driver documents as
//! possibly synchronizing implicitly) and `cuMemUnmap`. Runtime API calls,
//! PyTorch's included, reach these driver calls too, so the driver domain
//! covers both.
//!
//! CUPTI invokes the callback on the calling thread as the call is entered.
//! A thread inside a [`Watched`] scope counts the call in its own counters
//! by call, which the scope reads ([`Watched::blocking_calls`],
//! [`Watched::calls`]) and a thread can snapshot at any time
//! ([`Calls::current`]); a thread outside any scope is not counted. The
//! callback only increments thread-local counters.
//!
//! CUPTI admits one subscriber per process: a profiler that injected CUPTI
//! (Nsight Systems) holds it, and [`BlockingWatch::start`] then fails with
//! [`Error::Cupti`] (code 39); the caller goes on without the watch.
//!
//! Dropping the watch ends the subscription; a later
//! [`BlockingWatch::start`] subscribes again. The library stays loaded and
//! initialized: `cuptiFinalize`, which would remove it, left the device work
//! a serving worker submitted afterwards waiting forever once CUPTI had
//! recorded its activity (CUPTI 2025.3.0 of the CUDA 13.0 toolkit, replaying
//! CUDA graphs), so nothing here calls it.
//!
//! The process loads the CUPTI beside the CUDA runtime it already loaded
//! ([`library`]): CUPTI belongs to a CUDA release, and both the toolkit and
//! the Python environment's CUDA packages keep `libcupti.so.N` beside
//! `libcudart.so.N`. A worker thereby loads the CUPTI its environment pins
//! with the runtime PyTorch runs on. The toolkit's CUPTI 2025.3.0 stalled a
//! serving worker that replayed CUDA graphs while kernels were recorded,
//! within a minute of load; the environment's `nvidia-cuda-cupti` 13.0.85
//! served the same load.
//!
//! [`activity`] accounts the device time of registered streams from CUPTI's
//! activity records.

pub mod activity;

use std::cell::Cell;
use std::ffi::{CStr, c_void};
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

use libloading::Library;

use crate::{Error, Result};

/// The CUPTI library of the loader's search path, for a process that
/// loaded no CUDA runtime.
const LIBCUPTI: &str = "libcupti.so";

/// The CUPTI library this process loads: `libcupti.so.N` in the directory of
/// the `libcudart.so.N` the process loaded, when that file exists, and
/// [`LIBCUPTI`] otherwise. Resolved once, at the first load.
pub fn library() -> &'static str {
    static LIBRARY: OnceLock<String> = OnceLock::new();
    LIBRARY.get_or_init(|| beside_runtime().unwrap_or_else(|| LIBCUPTI.to_owned()))
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

/// `CUPTI_CB_DOMAIN_DRIVER_API`.
const DRIVER_DOMAIN: i32 = 1;

/// `CUPTI_API_ENTER`.
const API_ENTER: i32 = 0;

/// The watched driver calls: `CUpti_driver_api_trace_cbid` and name. Every
/// call here returns only once the device work it waits for completed.
const WATCHED: &[(u32, &str)] = &[
    (17, "cuCtxSynchronize"),
    (800, "cuCtxSynchronize_v2"),
    (121, "cuEventSynchronize"),
    (126, "cuStreamSynchronize"),
    (440, "cuStreamSynchronize_ptsz"),
    (38, "cuMemFreeHost"),
    (245, "cuMemFree_v2"),
    (552, "cuMemUnmap"),
    (43, "cuMemcpyHtoD"),
    (45, "cuMemcpyDtoH"),
    (47, "cuMemcpyDtoD"),
    (49, "cuMemcpyDtoA"),
    (51, "cuMemcpyAtoD"),
    (53, "cuMemcpyHtoA"),
    (54, "cuMemcpyAtoH"),
    (55, "cuMemcpyAtoA"),
    (56, "cuMemcpy2D"),
    (57, "cuMemcpy2DUnaligned"),
    (58, "cuMemcpy3D"),
    (248, "cuMemcpy_v2"),
    (276, "cuMemcpyHtoD_v2"),
    (278, "cuMemcpyDtoH_v2"),
    (280, "cuMemcpyDtoD_v2"),
    (282, "cuMemcpyAtoH_v2"),
    (284, "cuMemcpyAtoD_v2"),
    (285, "cuMemcpyDtoA_v2"),
    (286, "cuMemcpyAtoA_v2"),
    (287, "cuMemcpy2D_v2"),
    (288, "cuMemcpy2DUnaligned_v2"),
    (290, "cuMemcpy3D_v2"),
    (292, "cuMemcpyHtoA_v2"),
    (305, "cuMemcpy"),
    (318, "cuMemcpyPeer"),
    (320, "cuMemcpy3DPeer"),
    (397, "cuMemcpyHtoD_v2_ptds"),
    (398, "cuMemcpyDtoH_v2_ptds"),
    (399, "cuMemcpyDtoD_v2_ptds"),
    (408, "cuMemcpy_ptds"),
];

thread_local! {
    /// How many [`Watched`] scopes the thread is inside.
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    /// The watched calls the thread made inside a scope, by entry of
    /// [`WATCHED`].
    static CALLS: [Cell<u64>; WATCHED.len()] = const { [const { Cell::new(0) }; WATCHED.len()] };
}

type CuptiResult = i32;
type Callback = unsafe extern "C" fn(*mut c_void, i32, u32, *const c_void);

/// The `libcupti.so` entry points the watch and the activity records use.
struct Cupti {
    _library: Library,
    subscribe: unsafe extern "C" fn(*mut *mut c_void, Callback, *mut c_void) -> CuptiResult,
    enable_callback: unsafe extern "C" fn(u32, *mut c_void, i32, u32) -> CuptiResult,
    unsubscribe: unsafe extern "C" fn(*mut c_void) -> CuptiResult,
    register_buffers:
        unsafe extern "C" fn(activity::BufferRequested, activity::BufferCompleted) -> CuptiResult,
    activity_enable: unsafe extern "C" fn(u32) -> CuptiResult,
    activity_disable: unsafe extern "C" fn(u32) -> CuptiResult,
    flush_all: unsafe extern "C" fn(u32) -> CuptiResult,
    next_record: unsafe extern "C" fn(*mut u8, usize, *mut *mut u8) -> CuptiResult,
    stream_id: unsafe extern "C" fn(*mut c_void, *mut c_void, u8, *mut u32) -> CuptiResult,
    context_id: unsafe extern "C" fn(*mut c_void, *mut u32) -> CuptiResult,
    dropped_records: unsafe extern "C" fn(*mut c_void, u32, *mut usize) -> CuptiResult,
}

// SAFETY: the table holds function pointers into a library that is never
// unloaded while the table lives; CUPTI's entry points are thread-safe.
unsafe impl Send for Cupti {}
// SAFETY: as above.
unsafe impl Sync for Cupti {}

static CUPTI: OnceLock<Result<Cupti>> = OnceLock::new();

fn cupti() -> Result<&'static Cupti> {
    CUPTI
        .get_or_init(|| {
            // SAFETY: loading libcupti runs only CUPTI's own initializers.
            let library = unsafe { Library::new(library()) }.map_err(|source| Error::Load {
                library: library(),
                source: Arc::new(source),
            })?;
            // SAFETY: each type is the C prototype of the named entry point.
            unsafe {
                Ok(Cupti {
                    subscribe: symbol(&library, "cuptiSubscribe")?,
                    enable_callback: symbol(&library, "cuptiEnableCallback")?,
                    unsubscribe: symbol(&library, "cuptiUnsubscribe")?,
                    register_buffers: symbol(&library, "cuptiActivityRegisterCallbacks")?,
                    activity_enable: symbol(&library, "cuptiActivityEnable")?,
                    activity_disable: symbol(&library, "cuptiActivityDisable")?,
                    flush_all: symbol(&library, "cuptiActivityFlushAll")?,
                    next_record: symbol(&library, "cuptiActivityGetNextRecord")?,
                    stream_id: symbol(&library, "cuptiGetStreamIdEx")?,
                    context_id: symbol(&library, "cuptiGetContextId")?,
                    dropped_records: symbol(&library, "cuptiActivityGetNumDroppedRecords")?,
                    _library: library,
                })
            }
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

fn check(code: CuptiResult, call: &'static str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Cupti { call, code })
    }
}

/// The process's subscription to the watched driver calls; unsubscribes on
/// drop.
pub struct BlockingWatch {
    subscriber: *mut c_void,
}

// SAFETY: the subscriber handle is an opaque CUPTI token valid on any thread.
unsafe impl Send for BlockingWatch {}

impl BlockingWatch {
    /// Subscribe to the watched driver calls. Fails when `libcupti.so` is
    /// missing or another CUPTI client holds the process's subscription.
    pub fn start() -> Result<Self> {
        let cupti = cupti()?;
        let mut subscriber = std::ptr::null_mut();
        // SAFETY: CUPTI writes the subscriber handle; the callback is a
        // plain function that only touches atomics and thread locals.
        let status =
            unsafe { (cupti.subscribe)(&mut subscriber, on_driver_call, std::ptr::null_mut()) };
        check(status, "cuptiSubscribe")?;
        let watch = Self { subscriber };
        for &(cbid, _) in WATCHED {
            // SAFETY: the subscriber is live; the domain and ids are CUPTI's.
            let status = unsafe { (cupti.enable_callback)(1, subscriber, DRIVER_DOMAIN, cbid) };
            check(status, "cuptiEnableCallback")?;
        }
        Ok(watch)
    }
}

impl Drop for BlockingWatch {
    fn drop(&mut self) {
        if let Ok(cupti) = cupti() {
            // SAFETY: the subscriber is live until this call.
            let _ = unsafe { (cupti.unsubscribe)(self.subscriber) };
        }
    }
}

/// The CUPTI callback: counts a watched call entered on a watched thread.
unsafe extern "C" fn on_driver_call(
    _userdata: *mut c_void,
    domain: i32,
    cbid: u32,
    data: *const c_void,
) {
    if domain != DRIVER_DOMAIN || data.is_null() || DEPTH.get() == 0 {
        return;
    }
    // SAFETY: for the API domains `data` is a `CUpti_CallbackData`, whose
    // first field is the `CUpti_ApiCallbackSite`.
    if unsafe { *data.cast::<i32>() } != API_ENTER {
        return;
    }
    if let Some(index) = WATCHED.iter().position(|&(watched, _)| watched == cbid) {
        CALLS.with(|calls| calls[index].set(calls[index].get() + 1));
    }
}

/// A snapshot of the calling thread's counters of watched calls.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Calls([u64; WATCHED.len()]);

impl Default for Calls {
    fn default() -> Self {
        Self([0; WATCHED.len()])
    }
}

impl Calls {
    /// The calling thread's counters now.
    pub fn current() -> Self {
        Self(CALLS.with(|calls| std::array::from_fn(|index| calls[index].get())))
    }

    /// The calls counted between `earlier` and this snapshot of the same
    /// thread, by call name; calls without any are left out.
    pub fn since(&self, earlier: &Self) -> Vec<(&'static str, u64)> {
        WATCHED
            .iter()
            .zip(self.0.iter().zip(&earlier.0))
            .filter_map(|(&(_, name), (&now, &then))| {
                let count = now.saturating_sub(then);
                (count > 0).then_some((name, count))
            })
            .collect()
    }
}

/// A scope in which the calling thread's watched driver calls are counted.
/// It stays on the thread that entered it.
pub struct Watched {
    /// The thread's counters when the scope was entered.
    entered: Calls,
    _thread: PhantomData<*const ()>,
}

impl Watched {
    /// Count the calling thread's watched calls until the scope ends.
    pub fn enter() -> Self {
        DEPTH.set(DEPTH.get() + 1);
        Self {
            entered: Calls::current(),
            _thread: PhantomData,
        }
    }

    /// Watched calls this thread made since the scope was entered.
    pub fn blocking_calls(&self) -> u64 {
        self.calls().iter().map(|&(_, count)| count).sum()
    }

    /// The watched calls this thread made since the scope was entered, by
    /// call name; calls it did not make are left out.
    pub fn calls(&self) -> Vec<(&'static str, u64)> {
        Calls::current().since(&self.entered)
    }
}

impl Drop for Watched {
    fn drop(&mut self) {
        DEPTH.set(DEPTH.get().saturating_sub(1));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::{DeviceAllocation, DeviceId, Stream, lock, memset_d8};

    /// CUPTI is process-wide: its device tests run one at a time.
    pub(crate) static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    #[ignore = "requires a CUDA device"]
    fn only_a_watched_thread_counts_its_waits_on_the_device() -> Result<()> {
        let _serial = lock(&SERIAL);
        let watch = BlockingWatch::start()?;
        let stream = Stream::new(DeviceId(0))?;

        // Outside a scope a wait is not counted.
        let before = Calls::current();
        stream.synchronize()?;
        assert_eq!(Calls::current(), before);

        let scope = Watched::enter();
        stream.synchronize()?;
        assert_eq!(scope.blocking_calls(), 1);
        assert_eq!(scope.calls(), vec![("cuStreamSynchronize", 1)]);
        drop(scope);
        drop(watch);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_stopped_watch_counts_nothing_until_a_new_one_starts() -> Result<()> {
        let _serial = lock(&SERIAL);
        let stream = Stream::new(DeviceId(0))?;
        let scope = Watched::enter();
        let watch = BlockingWatch::start()?;
        drop(activity::StreamActivity::start([("lane", &stream)])?);
        drop(watch);
        stream.synchronize()?;
        assert_eq!(scope.blocking_calls(), 0);

        let watch = BlockingWatch::start()?;
        let recording = activity::StreamActivity::start([("lane", &stream)])?;
        let bytes = 256 << 20;
        let memory = DeviceAllocation::linear(DeviceId(0), bytes)?;
        // SAFETY: the allocation holds `bytes` and outlives the fill, which
        // the synchronization waits for.
        unsafe { memset_d8(memory.address(), 0, bytes, &stream)? };
        stream.synchronize()?;
        assert_eq!(scope.calls(), vec![("cuStreamSynchronize", 1)]);
        let taken = activity::tests::taken_until("lane", |taken| taken.busy_us > 0);
        assert!(taken.busy_us > 0, "{taken:?}");
        drop(recording);
        drop(watch);
        Ok(())
    }
}
