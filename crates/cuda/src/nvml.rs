//! NVML bindings (`libnvidia-ml.so.1`): per-process and device memory usage,
//! and which processes a device lists.
//!
//! NVML is a management library and creates no CUDA context, so a process
//! that holds no context, such as the Storage Server, queries it too.

use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::{Arc, OnceLock};

use libloading::Library;

use crate::{DeviceId, DeviceUuid, Error, Result};

const LIBNVML: &str = "libnvidia-ml.so.1";

type NvmlReturn = i32;
type NvmlDevice = *mut c_void;

const NVML_SUCCESS: NvmlReturn = 0;
const NVML_ERROR_INSUFFICIENT_SIZE: NvmlReturn = 7;
/// `usedGpuMemory` when the driver cannot attribute memory to the process.
const NVML_VALUE_NOT_AVAILABLE: u64 = u64::MAX;

/// `nvmlProcessInfo_t` (v2), returned by the `_v3` process queries.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ProcessInfo {
    pid: u32,
    used_gpu_memory: u64,
    gpu_instance_id: u32,
    compute_instance_id: u32,
}

/// `nvmlMemory_v2_t`.
#[repr(C)]
#[derive(Default)]
struct RawMemory {
    version: u32,
    total: u64,
    reserved: u64,
    free: u64,
    used: u64,
}

/// `NVML_STRUCT_VERSION(Memory, 2)`: the struct size with the version in the
/// top byte.
const NVML_MEMORY_V2: u32 = size_of::<RawMemory>() as u32 | (2 << 24);

/// The signature of NVML's `_v3` process queries.
type ProcessQuery = unsafe extern "C" fn(NvmlDevice, *mut u32, *mut ProcessInfo) -> NvmlReturn;

struct Api {
    init: unsafe extern "C" fn() -> NvmlReturn,
    shutdown: unsafe extern "C" fn() -> NvmlReturn,
    error_string: unsafe extern "C" fn(NvmlReturn) -> *const c_char,
    device_by_uuid: unsafe extern "C" fn(*const c_char, *mut NvmlDevice) -> NvmlReturn,
    compute_processes: ProcessQuery,
    graphics_processes: ProcessQuery,
    memory_info: unsafe extern "C" fn(NvmlDevice, *mut RawMemory) -> NvmlReturn,
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
        if status == NVML_SUCCESS {
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
            code: status,
            message,
        })
    }
}

/// Device memory reported by NVML, in bytes.
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
    pub fn init() -> Result<Self> {
        let api = api()?;
        // SAFETY: nvmlInit takes no arguments and is reference counted.
        api.check(unsafe { (api.init)() }, "nvmlInit_v2")?;
        Ok(Self { api })
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

    /// Device memory that NVML attributes to this process on `device`: every
    /// allocation of its context, including PyTorch's cached segments, graph
    /// pools, VMM arenas, communicator buffers and loaded modules.
    ///
    /// Fails with [`Error::ProcessNotFound`] when NVML lists no usage for this
    /// PID, as happens inside a PID namespace NVML does not report.
    pub fn process_memory(&self, device: DeviceId) -> Result<u64> {
        let handle = self.device(device)?;
        let processes = self.processes(self.api.compute_processes, handle)?;
        let pid = std::process::id();

        let usage: Vec<u64> = processes
            .iter()
            .filter(|process| process.pid == pid)
            .map(|process| process.used_gpu_memory)
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
            if status == NVML_ERROR_INSUFFICIENT_SIZE {
                processes.resize(count as usize + 4, ProcessInfo::default());
                continue;
            }
            self.api.check(status, call)?;
            processes.truncate(count as usize);
            return Ok(processes);
        }
    }

    /// Total, reserved, free and used memory of `device`.
    pub fn memory_info(&self, device: DeviceId) -> Result<NvmlMemory> {
        let handle = self.device(device)?;
        let mut memory = RawMemory {
            version: NVML_MEMORY_V2,
            ..RawMemory::default()
        };
        self.api.check(
            // SAFETY: NVML fills the local struct, whose version field names its
            // `nvmlMemory_v2_t` layout; `handle` came from NVML for this device.
            unsafe { (self.api.memory_info)(handle, &mut memory) },
            "nvmlDeviceGetMemoryInfo_v2",
        )?;
        Ok(NvmlMemory {
            total: memory.total,
            reserved: memory.reserved,
            free: memory.free,
            used: memory.used,
        })
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        // SAFETY: balances the nvmlInit of this session.
        let _ = unsafe { (self.api.shutdown)() };
    }
}
