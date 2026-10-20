//! The NVML query a shoot-down waits on: which processes a GPU still lists.
//!
//! NVML is a management library and creates no CUDA context. It is loaded
//! at run time (`libnvidia-ml.so.1`), so the Storage Server links nothing
//! of CUDA and runs on a host without a driver until a shoot-down needs it.

use std::ffi::{CString, c_char, c_void};
use std::io;
use std::sync::OnceLock;

use libloading::Library;

const LIBNVML: &str = "libnvidia-ml.so.1";

type NvmlReturn = i32;
type NvmlDevice = *mut c_void;

const NVML_SUCCESS: NvmlReturn = 0;
const NVML_ERROR_INSUFFICIENT_SIZE: NvmlReturn = 7;

/// `nvmlProcessInfo_t` (v2), returned by the `_v3` process queries.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ProcessInfo {
    pid: u32,
    used_gpu_memory: u64,
    gpu_instance_id: u32,
    compute_instance_id: u32,
}

type ProcessQuery = unsafe extern "C" fn(NvmlDevice, *mut u32, *mut ProcessInfo) -> NvmlReturn;

struct Api {
    device_by_uuid: unsafe extern "C" fn(*const c_char, *mut NvmlDevice) -> NvmlReturn,
    compute_processes: ProcessQuery,
    graphics_processes: ProcessQuery,
    // Declared last: the library outlives every copied function pointer.
    _library: Library,
}

/// The initialized NVML session, loaded once per process.
fn api() -> io::Result<&'static Api> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        // SAFETY: loading NVML runs only its own initializers.
        let library = unsafe { Library::new(LIBNVML) }
            .map_err(|error| format!("loading {LIBNVML}: {error}"))?;
        // SAFETY: each type matches the `nvml.h` prototype of its symbol.
        let api = unsafe {
            let init = *library
                .get::<unsafe extern "C" fn() -> NvmlReturn>(b"nvmlInit_v2\0")
                .map_err(|error| error.to_string())?;
            let status = init();
            if status != NVML_SUCCESS {
                return Err(format!("nvmlInit_v2 returned {status}"));
            }
            Api {
                device_by_uuid: *library
                    .get(b"nvmlDeviceGetHandleByUUID\0")
                    .map_err(|error| error.to_string())?,
                compute_processes: *library
                    .get(b"nvmlDeviceGetComputeRunningProcesses_v3\0")
                    .map_err(|error| error.to_string())?,
                graphics_processes: *library
                    .get(b"nvmlDeviceGetGraphicsRunningProcesses_v3\0")
                    .map_err(|error| error.to_string())?,
                _library: library,
            }
        };
        Ok(api)
    })
    .as_ref()
    .map_err(|message| io::Error::other(message.clone()))
}

/// NVML's identifier of a GPU UUID: `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.
pub(crate) fn uuid_name(uuid: &[u8; 16]) -> String {
    let hex: String = uuid.iter().map(|byte| format!("{byte:02x}")).collect();
    format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Whether the GPU with `uuid` lists process `pid` as a compute or graphics
/// process: whether the process still holds a context there.
pub(crate) fn lists(uuid: &[u8; 16], pid: i32) -> io::Result<bool> {
    let api = api()?;
    let name = CString::new(uuid_name(uuid)).map_err(io::Error::other)?;
    let mut device: NvmlDevice = std::ptr::null_mut();
    // SAFETY: `name` is NUL-terminated and `device` receives one handle.
    let status = unsafe { (api.device_by_uuid)(name.as_ptr(), &mut device) };
    if status != NVML_SUCCESS {
        return Err(io::Error::other(format!(
            "nvmlDeviceGetHandleByUUID({}) returned {status}",
            uuid_name(uuid)
        )));
    }
    for query in [api.compute_processes, api.graphics_processes] {
        if processes(query, device)?.contains(&(pid as u32)) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The process ids one of NVML's process queries lists for `device`.
fn processes(query: ProcessQuery, device: NvmlDevice) -> io::Result<Vec<u32>> {
    let mut capacity = 64_u32;
    loop {
        let mut infos = vec![ProcessInfo::default(); capacity as usize];
        let mut count = capacity;
        // SAFETY: `infos` holds `count` entries for NVML to fill.
        let status = unsafe { query(device, &mut count, infos.as_mut_ptr()) };
        match status {
            NVML_SUCCESS => {
                return Ok(infos[..count as usize]
                    .iter()
                    .map(|info| info.pid)
                    .collect());
            }
            // Processes appeared since the count was taken: ask again with
            // room for them.
            NVML_ERROR_INSUFFICIENT_SIZE => capacity = count.max(capacity) * 2,
            _ => {
                return Err(io::Error::other(format!(
                    "NVML's process query returned {status}"
                )));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuids_name_gpus_as_nvml_does() {
        let uuid = [
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
            0xcd, 0xef,
        ];
        assert_eq!(uuid_name(&uuid), "GPU-12345678-9abc-def0-0123-456789abcdef");
    }
}
