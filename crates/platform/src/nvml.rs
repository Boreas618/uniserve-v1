//! NVML bindings (`libnvidia-ml.so.1`): per-process and device storage usage,
//! and which processes a device lists.
//!
//! NVML is a management library and creates no CUDA context, so a process
//! that holds no context, such as the Storage Server, queries it too. It ships
//! with the driver and is opened by its soname; a session refuses an NVML whose
//! CUDA driver version is older than the `cuda.h` the crate's bindings are
//! generated from.
//!
//! A session's opening and closing are ordered against CUPTI's attachment to
//! the CUDA driver ([`between_sessions`]).

use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use libloading::Library;

use crate::sys::nvml as sys;
use crate::{DeviceId, DeviceUuid, Error, Result};

const LIBNVML: &str = "libnvidia-ml.so.1";

/// Held while an NVML session opens or closes, and while a CUPTI call that may
/// attach CUPTI to the CUDA driver runs ([`between_sessions`]).
static SESSIONS: Mutex<()> = Mutex::new(());

/// Runs `attach`, a CUPTI call that may be the process's first attachment of
/// CUPTI to the CUDA driver, while no NVML session opens or closes in the
/// process, and returns its result.
///
/// `nvmlInit` and `nvmlShutdown` take a recursive lock of the CUDA driver's,
/// through lock and unlock functions the driver lends NVML; they are the only
/// NVML entry points that take it. CUPTI's first attachment to the driver in a
/// process (its first `cuptiSubscribe`, `cuptiActivityRegisterCallbacks`,
/// `cuptiActivityEnable` or `cuptiProfilerInitialize`) makes both functions
/// return without touching the lock, for the rest of the process. A thread
/// inside `nvmlInit` or `nvmlShutdown` at that moment therefore never releases
/// the lock, and every thread already waiting for it in another session's
/// opening or closing waits forever, in `nvmlInitWithFlags` or `nvmlShutdown`
/// (driver 580.82.07). Every opening, closing and attachment takes this one
/// lock, so an attachment runs only between them. Each holds it for one driver
/// call; once CUPTI is attached, the openings and closings no longer take the
/// driver's lock.
pub(crate) fn between_sessions<T>(attach: impl FnOnce() -> T) -> T {
    let _sessions = SESSIONS.lock().unwrap_or_else(PoisonError::into_inner);
    attach()
}

/// What a refused NVML is told to install: NVML ships with the driver.
const NVML_PACKAGE: &str = "the NVIDIA driver of CUDA 13.0 or later";

type NvmlReturn = sys::nvmlReturn_t;
type NvmlDevice = *mut c_void;
type ProcessInfo = sys::nvmlProcessInfo_t;

/// `NVML_STRUCT_VERSION(Memory, 2)`: the struct size with the version in the
/// top byte, as `nvml.h` defines the macro over `sizeof(nvmlMemory_v2_t)` (a
/// macro over `sizeof` has no value bindgen can generate).
const NVML_MEMORY_V2: u32 = size_of::<sys::nvmlMemory_v2_t>() as u32 | (2 << 24);

/// `usedGpuMemory` when the driver cannot attribute storage to the process:
/// the header's `NVML_VALUE_NOT_AVAILABLE` (`-1`) as the unsigned field holds it.
const NVML_VALUE_NOT_AVAILABLE: u64 = sys::NVML_VALUE_NOT_AVAILABLE as u64;

/// The signature of NVML's `_v3` process queries.
type ProcessQuery = unsafe extern "C" fn(NvmlDevice, *mut u32, *mut ProcessInfo) -> NvmlReturn;

struct Api {
    init: unsafe extern "C" fn() -> NvmlReturn,
    shutdown: unsafe extern "C" fn() -> NvmlReturn,
    error_string: unsafe extern "C" fn(NvmlReturn) -> *const c_char,
    cuda_driver_version: unsafe extern "C" fn(*mut i32) -> NvmlReturn,
    device_by_uuid: unsafe extern "C" fn(*const c_char, *mut NvmlDevice) -> NvmlReturn,
    compute_processes: ProcessQuery,
    graphics_processes: ProcessQuery,
    memory_info: unsafe extern "C" fn(NvmlDevice, *mut sys::nvmlMemory_v2_t) -> NvmlReturn,
    // Declared last: the library outlives every copied function pointer.
    _library: Library,
}

fn api() -> Result<&'static Api> {
    static API: OnceLock<Result<Api>> = OnceLock::new();

    API.get_or_init(|| {
        // SAFETY: loading NVML runs only its own initializers.
        let library = unsafe { Library::new(LIBNVML) }.map_err(|source| Error::Load {
            library: LIBNVML,
            source: Arc::new(source),
        })?;
        // SAFETY: each signature matches the `nvml.h` prototype of the symbol.
        unsafe {
            Ok(Api {
                init: symbol(&library, "nvmlInit_v2")?,
                shutdown: symbol(&library, "nvmlShutdown")?,
                error_string: symbol(&library, "nvmlErrorString")?,
                cuda_driver_version: symbol(&library, "nvmlSystemGetCudaDriverVersion_v2")?,
                device_by_uuid: symbol(&library, "nvmlDeviceGetHandleByUUID")?,
                compute_processes: symbol(&library, "nvmlDeviceGetComputeRunningProcesses_v3")?,
                graphics_processes: symbol(&library, "nvmlDeviceGetGraphicsRunningProcesses_v3")?,
                memory_info: symbol(&library, "nvmlDeviceGetMemoryInfo_v2")?,
                _library: library,
            })
        }
    })
    .as_ref()
    .map_err(Clone::clone)
}

/// # Safety
/// `T` must be the function-pointer type of the symbol's C prototype.
unsafe fn symbol<T: Copy>(library: &Library, name: &'static str) -> Result<T> {
    // SAFETY: forwarded from the caller.
    unsafe { library.get::<T>(name.as_bytes()) }
        .map(|symbol| *symbol)
        .map_err(|source| Error::Symbol {
            library: LIBNVML,
            symbol: name,
            source: Arc::new(source),
        })
}

impl Api {
    fn check(&self, status: NvmlReturn, call: &'static str) -> Result<()> {
        if status == sys::NVML_SUCCESS {
            return Ok(());
        }
        // SAFETY: NVML returns a static, NUL-terminated description for every
        // code, owned by the never-unloaded library.
        let message = unsafe {
            let text = (self.error_string)(status);
            if text.is_null() {
                "unrecognized NVML return"
            } else {
                CStr::from_ptr(text)
                    .to_str()
                    .unwrap_or("non-UTF-8 NVML return")
            }
        };
        Err(Error::Nvml {
            call,
            // `nvmlReturn_t` is a C enum; the error carries the value NVML documents.
            code: status as i32,
            message,
        })
    }
}

/// Refuses an NVML whose CUDA driver version is older than the `cuda.h` the
/// crate's bindings are generated from. NVML ships with the driver, so the
/// version it reports is the driver's (a CUDA version code, as
/// `cuDriverGetVersion` states it).
fn require_version(version: i32) -> Result<()> {
    let required = crate::sys::cuda::CUDA_VERSION as i32;
    if version >= required {
        return Ok(());
    }
    Err(Error::Version {
        library: LIBNVML,
        found: crate::driver::cuda_release(version),
        required: crate::driver::cuda_release(required),
        package: NVML_PACKAGE,
    })
}

/// Device storage reported by NVML, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NvmlMemory {
    pub total: u64,
    /// Memory reserved by the driver and firmware.
    pub reserved: u64,
    pub free: u64,
    pub used: u64,
}

/// An initialized NVML session (`nvmlInit` / `nvmlShutdown` on drop).
///
/// NVML is thread-safe, so the session is `Send + Sync`.
///
/// Devices are addressed by CUDA ordinal and matched to NVML through their
/// UUID, because NVML enumerates physical devices regardless of
/// `CUDA_VISIBLE_DEVICES`.
pub struct Nvml {
    api: &'static Api,
}

impl Nvml {
    /// Starts a session.
    ///
    /// # Errors
    ///
    /// [`Error::Load`] without the driver's NVML, and [`Error::Version`] when
    /// the CUDA driver version NVML reports is older than the `cuda.h` the
    /// crate's bindings are generated from.
    pub fn init() -> Result<Self> {
        let api = api()?;
        // SAFETY: nvmlInit takes no arguments and is reference counted. It takes the CUDA
        // driver's lock, which a CUPTI attachment must not find held ([`between_sessions`]).
        api.check(between_sessions(|| unsafe { (api.init)() }), "nvmlInit_v2")?;
        // The session exists from here on, so a refusal below shuts it down on the way out.
        let session = Self { api };

        let mut version = 0;
        api.check(
            // SAFETY: NVML writes one int to the local `version`.
            unsafe { (api.cuda_driver_version)(&mut version) },
            "nvmlSystemGetCudaDriverVersion_v2",
        )?;
        require_version(version)?;
        Ok(session)
    }

    fn device(&self, device: DeviceId) -> Result<NvmlDevice> {
        self.device_by_uuid(device.uuid()?)
    }

    /// The NVML handle of the physical device `uuid` names.
    fn device_by_uuid(&self, uuid: DeviceUuid) -> Result<NvmlDevice> {
        let uuid = CString::new(uuid.to_string())
            .map_err(|_| Error::invalid("device UUIDs contain no NUL byte"))?;
        let mut handle = std::ptr::null_mut();
        self.api.check(
            // SAFETY: the UUID is a NUL-terminated string that outlives the call;
            // NVML writes one handle to the local `handle`.
            unsafe { (self.api.device_by_uuid)(uuid.as_ptr(), &mut handle) },
            "nvmlDeviceGetHandleByUUID",
        )?;
        Ok(handle)
    }

    /// Device storage that NVML attributes to this process on `device`: every
    /// allocation of its context, including the allocator's graph and scratch
    /// pools, communicator buffers and loaded modules.
    ///
    /// Design 8's NVML rule keeps storage that this process creates and then
    /// re-imports from its exported descriptor off that account, because the
    /// creating handle is dropped; [`crate::Buffer::export`] is the half of it
    /// this crate provides.
    ///
    /// Fails with [`Error::ProcessNotFound`] when NVML lists no usage for this
    /// PID, as happens inside a PID namespace NVML does not report.
    pub fn process_storage(&self, device: DeviceId) -> Result<u64> {
        let handle = self.device(device)?;
        let processes = self.processes(self.api.compute_processes, handle)?;
        let pid = std::process::id();

        let usage: Vec<u64> = processes
            .iter()
            .filter(|process| process.pid == pid)
            .map(|process| process.usedGpuMemory)
            .collect();
        if usage.is_empty() || usage.contains(&NVML_VALUE_NOT_AVAILABLE) {
            return Err(Error::ProcessNotFound { pid, device });
        }
        Ok(usage.iter().sum())
    }

    /// Whether the device `uuid` lists process `pid` as a compute or graphics
    /// process: whether the process still holds a context there.
    pub fn lists_process(&self, uuid: DeviceUuid, pid: u32) -> Result<bool> {
        let handle = self.device_by_uuid(uuid)?;
        for query in [self.api.compute_processes, self.api.graphics_processes] {
            if self
                .processes(query, handle)?
                .iter()
                .any(|process| process.pid == pid)
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether no compute or graphics process holds a context on `device`.
    /// A replica update uses idle devices without interrupting another
    /// serving process or taking its device allocations.
    pub fn is_idle(&self, device: DeviceId) -> Result<bool> {
        let handle = self.device(device)?;
        for query in [self.api.compute_processes, self.api.graphics_processes] {
            if !self.processes(query, handle)?.is_empty() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The processes one of NVML's `_v3` queries lists for `handle`.
    fn processes(&self, query: ProcessQuery, handle: NvmlDevice) -> Result<Vec<ProcessInfo>> {
        let call = "nvmlDevice*RunningProcesses_v3";
        let mut processes = Vec::new();
        loop {
            // The process list can grow between the size query and the read;
            // NVML then reports INSUFFICIENT_SIZE with the new count.
            let mut count = processes.len() as u32;
            // SAFETY: the buffer holds `count` entries; NVML writes at most
            // that many and stores the number written (or needed) in count.
            let status = unsafe { query(handle, &mut count, processes.as_mut_ptr()) };
            if status == sys::NVML_ERROR_INSUFFICIENT_SIZE {
                processes.resize(count as usize + 4, ProcessInfo::default());
                continue;
            }
            self.api.check(status, call)?;
            processes.truncate(count as usize);
            return Ok(processes);
        }
    }

    /// Total, reserved, free and used storage of `device`.
    pub fn memory_info(&self, device: DeviceId) -> Result<NvmlMemory> {
        let handle = self.device(device)?;
        let mut storage = sys::nvmlMemory_v2_t {
            version: NVML_MEMORY_V2,
            ..Default::default()
        };
        self.api.check(
            // SAFETY: NVML fills the local struct, whose version field names its
            // `nvmlMemory_v2_t` layout; `handle` came from NVML for this device.
            unsafe { (self.api.memory_info)(handle, &mut storage) },
            "nvmlDeviceGetMemoryInfo_v2",
        )?;
        Ok(NvmlMemory {
            total: storage.total,
            reserved: storage.reserved,
            free: storage.free,
            used: storage.used,
        })
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        // SAFETY: balances the nvmlInit of this session. It takes the CUDA driver's lock, as
        // nvmlInit does ([`between_sessions`]).
        let _ = between_sessions(|| unsafe { (self.api.shutdown)() });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An NVML of a driver older than the bindings is refused by name with the
    /// package to install; the bindings' release and later ones are accepted.
    #[test]
    fn an_nvml_older_than_the_bindings_is_refused() {
        assert!(require_version(crate::sys::cuda::CUDA_VERSION as i32).is_ok());

        let error = require_version(12090).expect_err("CUDA 12.9 predates the bindings");
        let text = error.to_string();
        assert!(
            matches!(
                error,
                Error::Version {
                    library: LIBNVML,
                    ..
                }
            ),
            "{text}"
        );
        assert!(
            text.contains("CUDA 12.9") && text.contains(NVML_PACKAGE),
            "{text}"
        );
    }
}
