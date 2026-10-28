//! The entry points of NIXL's C API this crate calls, resolved from
//! `libnixl_capi.so` when the library loads ([`crate::library`]).
//!
//! The library is NIXL's own C binding (`src/bindings/rust/wrapper.cpp`,
//! built with NIXL's `-Drust=true`), and every declaration here follows the
//! release this crate binds, NIXL 1.5.1, as its `src/bindings/rust/wrapper.h`
//! declares it: each C enum is an `int`, each handle an opaque pointer the
//! library allocates and frees, and every function returns a status. A
//! library of another release whose entry points differ in arguments would
//! be misread, so the build that provides the library is pinned with this
//! crate (see `docs/nixl.md` for sources, patches and build commands).

use std::ffi::{c_char, c_int, c_uint, c_void};

use libloading::os::unix::Library;

/// `nixl_capi_status_t`.
pub(crate) type Status = c_int;
pub(crate) const SUCCESS: Status = 0;
pub(crate) const IN_PROG: Status = 1;

/// The name of a status, as `nixl_capi_status_t` names it.
pub(crate) fn status_name(status: Status) -> &'static str {
    match status {
        SUCCESS => "SUCCESS",
        IN_PROG => "IN_PROG",
        -1 => "ERROR_INVALID_PARAM",
        -2 => "ERROR_BACKEND",
        -3 => "ERROR_INVALID_STATE",
        -4 => "ERROR_EXCEPTION",
        -5 => "ERROR_NO_TELEMETRY",
        -6 => "ERROR_NOT_FOUND",
        _ => "unknown status",
    }
}

/// `nixl_capi_mem_type_t`.
pub(crate) type MemType = c_int;
pub(crate) const MEM_DRAM: MemType = 0;
pub(crate) const MEM_VRAM: MemType = 1;

/// `nixl_capi_xfer_op_t`.
pub(crate) type XferOp = c_int;
pub(crate) const XFER_OP_READ: XferOp = 0;

/// `nixl_capi_thread_sync_t`: the agent leaves synchronization to its
/// caller, which serializes every call on the agent ([`crate::Agent`]).
pub(crate) type ThreadSync = c_int;
pub(crate) const THREAD_SYNC_NONE: ThreadSync = 0;

/// `nixl_capi_agent_config_t`.
#[repr(C)]
pub(crate) struct AgentConfig {
    pub(crate) enable_prog_thread: bool,
    pub(crate) enable_listen_thread: bool,
    pub(crate) listen_port: c_int,
    pub(crate) thread_sync: ThreadSync,
    pub(crate) num_workers: c_uint,
    pub(crate) pthr_delay_us: u64,
    pub(crate) lthr_delay_us: u64,
    pub(crate) capture_telemetry: bool,
}

/// Every handle type of the C API (`nixl_capi_agent_t`,
/// `nixl_capi_reg_dlist_t`, ...): a pointer to a structure the library
/// owns.
pub(crate) type Handle = *mut c_void;

/// The entry points, resolved once from the loaded library, which stays
/// loaded for the process's life.
pub(crate) struct Api {
    pub(crate) create_configured_agent:
        unsafe extern "C" fn(*const c_char, *const AgentConfig, *mut Handle) -> Status,
    pub(crate) destroy_agent: unsafe extern "C" fn(Handle) -> Status,
    pub(crate) get_local_md: unsafe extern "C" fn(Handle, *mut *mut c_void, *mut usize) -> Status,
    pub(crate) load_remote_md:
        unsafe extern "C" fn(Handle, *const c_void, usize, *mut *mut c_char) -> Status,
    pub(crate) invalidate_remote_md: unsafe extern "C" fn(Handle, *const c_char) -> Status,
    pub(crate) get_plugin_params:
        unsafe extern "C" fn(Handle, *const c_char, *mut Handle, *mut Handle) -> Status,
    pub(crate) destroy_mem_list: unsafe extern "C" fn(Handle) -> Status,
    pub(crate) destroy_params: unsafe extern "C" fn(Handle) -> Status,
    pub(crate) params_add: unsafe extern "C" fn(Handle, *const c_char, *const c_char) -> Status,
    pub(crate) create_backend:
        unsafe extern "C" fn(Handle, *const c_char, Handle, *mut Handle) -> Status,
    pub(crate) destroy_backend: unsafe extern "C" fn(Handle) -> Status,
    pub(crate) create_reg_dlist: unsafe extern "C" fn(MemType, *mut Handle) -> Status,
    pub(crate) destroy_reg_dlist: unsafe extern "C" fn(Handle) -> Status,
    pub(crate) reg_dlist_add_desc:
        unsafe extern "C" fn(Handle, usize, usize, u64, *const c_void, usize) -> Status,
    pub(crate) register_mem: unsafe extern "C" fn(Handle, Handle, Handle) -> Status,
    pub(crate) deregister_mem: unsafe extern "C" fn(Handle, Handle, Handle) -> Status,
    pub(crate) create_xfer_dlist: unsafe extern "C" fn(MemType, *mut Handle) -> Status,
    pub(crate) destroy_xfer_dlist: unsafe extern "C" fn(Handle) -> Status,
    pub(crate) xfer_dlist_add_desc: unsafe extern "C" fn(Handle, usize, usize, u64) -> Status,
    pub(crate) create_xfer_req: unsafe extern "C" fn(
        Handle,
        XferOp,
        Handle,
        Handle,
        *const c_char,
        *mut Handle,
        Handle,
    ) -> Status,
    pub(crate) post_xfer_req: unsafe extern "C" fn(Handle, Handle, Handle) -> Status,
    pub(crate) get_xfer_status: unsafe extern "C" fn(Handle, Handle) -> Status,
    pub(crate) release_xfer_req: unsafe extern "C" fn(Handle, Handle) -> Status,
    pub(crate) destroy_xfer_req: unsafe extern "C" fn(Handle) -> Status,
}

/// Resolves `$name` from `$library` as the type of the field it fills.
macro_rules! entry {
    ($library:expr, $name:literal) => {{
        // SAFETY: the field's type is the function's signature in the
        // release's `wrapper.h`; the library stays loaded for the process's
        // life, so the pointer never dangles.
        let symbol = unsafe { $library.get(concat!($name, "\0").as_bytes()) }
            .map_err(|error| format!("{}: {error}", $name))?;
        *symbol
    }};
}

impl Api {
    /// Resolves every entry point from `library`, naming the first that is
    /// missing.
    pub(crate) fn resolve(library: &Library) -> Result<Self, String> {
        Ok(Self {
            create_configured_agent: entry!(library, "nixl_capi_create_configured_agent"),
            destroy_agent: entry!(library, "nixl_capi_destroy_agent"),
            get_local_md: entry!(library, "nixl_capi_get_local_md"),
            load_remote_md: entry!(library, "nixl_capi_load_remote_md"),
            invalidate_remote_md: entry!(library, "nixl_capi_invalidate_remote_md"),
            get_plugin_params: entry!(library, "nixl_capi_get_plugin_params"),
            destroy_mem_list: entry!(library, "nixl_capi_destroy_mem_list"),
            destroy_params: entry!(library, "nixl_capi_destroy_params"),
            params_add: entry!(library, "nixl_capi_params_add"),
            create_backend: entry!(library, "nixl_capi_create_backend"),
            destroy_backend: entry!(library, "nixl_capi_destroy_backend"),
            create_reg_dlist: entry!(library, "nixl_capi_create_reg_dlist"),
            destroy_reg_dlist: entry!(library, "nixl_capi_destroy_reg_dlist"),
            reg_dlist_add_desc: entry!(library, "nixl_capi_reg_dlist_add_desc"),
            register_mem: entry!(library, "nixl_capi_register_mem"),
            deregister_mem: entry!(library, "nixl_capi_deregister_mem"),
            create_xfer_dlist: entry!(library, "nixl_capi_create_xfer_dlist"),
            destroy_xfer_dlist: entry!(library, "nixl_capi_destroy_xfer_dlist"),
            xfer_dlist_add_desc: entry!(library, "nixl_capi_xfer_dlist_add_desc"),
            create_xfer_req: entry!(library, "nixl_capi_create_xfer_req"),
            post_xfer_req: entry!(library, "nixl_capi_post_xfer_req"),
            get_xfer_status: entry!(library, "nixl_capi_get_xfer_status"),
            release_xfer_req: entry!(library, "nixl_capi_release_xfer_req"),
            destroy_xfer_req: entry!(library, "nixl_capi_destroy_xfer_req"),
        })
    }
}

unsafe extern "C" {
    /// The C library's `free`, which releases the buffers the C API
    /// allocates for its caller (metadata, agent names).
    pub(crate) fn free(pointer: *mut c_void);
}
