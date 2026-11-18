//! The entry points of Mooncake Transfer Engine's C API these bindings call, resolved from
//! `libtransfer_engine.so` when the library loads ([`super::library`]).
//!
//! The library is the Transfer Engine's shared library of the Mooncake release these bindings
//! bind ([`super::RELEASE`]), and every declaration here follows its
//! `mooncake-transfer-engine/include/transfer_engine_c.h`: the engine is an opaque pointer the
//! library allocates and frees, a segment is an `int32_t` the engine numbers, a batch a
//! `uint64_t`, and every other call returns an `int`, 0 on success and a negative code of
//! `include/error.h` otherwise. A library of another release whose entry points differ in
//! arguments would be misread, so the Mooncake package that provides the library is the release
//! these bindings state (`docs/mooncake.md`).

use std::ffi::{c_char, c_int, c_void};

use libloading::os::unix::Library;

/// `transfer_engine_t`: the engine the library allocates.
pub(crate) type Handle = *mut c_void;

/// `segment_id_t`: another engine's storage, as this engine numbers it once opened.
pub(crate) type SegmentId = i32;

/// `batch_id_t`: a batch of transfers the engine allocated.
pub(crate) type BatchId = u64;

/// `INVALID_BATCH`: what `allocateBatchID` returns when it allocates none.
pub(crate) const INVALID_BATCH: BatchId = u64::MAX;

/// `OPCODE_READ`, `OPCODE_WRITE`.
pub(crate) const OPCODE_READ: c_int = 0;
pub(crate) const OPCODE_WRITE: c_int = 1;

/// `transfer_status::status`: a transfer still under way, and the states it ends in.
pub(crate) const STATUS_WAITING: c_int = 0;
pub(crate) const STATUS_PENDING: c_int = 1;
pub(crate) const STATUS_INVALID: c_int = 2;
pub(crate) const STATUS_CANCELED: c_int = 3;
pub(crate) const STATUS_COMPLETED: c_int = 4;
pub(crate) const STATUS_TIMEOUT: c_int = 5;
pub(crate) const STATUS_FAILED: c_int = 6;

/// `ERR_BATCH_BUSY`: a batch whose transfers have not all ended cannot be freed.
pub(crate) const ERR_BATCH_BUSY: c_int = -4;

/// The name of a transfer status, as `transfer_engine_c.h` names it.
pub(crate) fn status_name(status: c_int) -> &'static str {
    match status {
        STATUS_WAITING => "WAITING",
        STATUS_PENDING => "PENDING",
        STATUS_INVALID => "INVALID",
        STATUS_CANCELED => "CANCELED",
        STATUS_COMPLETED => "COMPLETED",
        STATUS_TIMEOUT => "TIMEOUT",
        STATUS_FAILED => "FAILED",
        _ => "unknown status",
    }
}

/// The name of a return code, as `include/error.h` names it.
pub(crate) fn code_name(code: c_int) -> &'static str {
    match code {
        0 => "OK",
        -1 => "ERR_INVALID_ARGUMENT",
        -2 => "ERR_TOO_MANY_REQUESTS",
        -3 => "ERR_ADDRESS_NOT_REGISTERED",
        ERR_BATCH_BUSY => "ERR_BATCH_BUSY",
        -6 => "ERR_DEVICE_NOT_FOUND",
        -7 => "ERR_ADDRESS_OVERLAPPED",
        -101 => "ERR_DNS",
        -102 => "ERR_SOCKET",
        -103 => "ERR_MALFORMED_JSON",
        -104 => "ERR_REJECT_HANDSHAKE",
        -200 => "ERR_METADATA",
        -201 => "ERR_ENDPOINT",
        -202 => "ERR_CONTEXT",
        -300 => "ERR_NUMA",
        -301 => "ERR_CLOCK",
        -302 => "ERR_MEMORY",
        -303 => "ERR_NOT_IMPLEMENTED",
        _ => "unknown code",
    }
}

/// `struct transfer_request`: one contiguous transfer between `source`, an address of this
/// engine's registered storage, and `target_offset`, an address of the segment `target_id`'s
/// registered storage (an address in the process that registered it, not an offset).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Request {
    pub(crate) opcode: c_int,
    pub(crate) source: *mut c_void,
    pub(crate) target_id: SegmentId,
    pub(crate) target_offset: u64,
    pub(crate) length: u64,
}

/// `struct transfer_status`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Status {
    pub(crate) status: c_int,
    pub(crate) transferred_bytes: u64,
}

/// The entry points, resolved once from the loaded library, which stays loaded for the
/// process's life.
pub(crate) struct Api {
    pub(crate) create:
        unsafe extern "C" fn(*const c_char, *const c_char, *const c_char, u64, c_int) -> Handle,
    pub(crate) destroy: unsafe extern "C" fn(Handle),
    pub(crate) local_name: unsafe extern "C" fn(Handle, *mut c_char, usize) -> c_int,
    pub(crate) register:
        unsafe extern "C" fn(Handle, *mut c_void, usize, *const c_char, c_int) -> c_int,
    pub(crate) unregister: unsafe extern "C" fn(Handle, *mut c_void) -> c_int,
    pub(crate) open_uncached: unsafe extern "C" fn(Handle, *const c_char) -> SegmentId,
    pub(crate) allocate_batch: unsafe extern "C" fn(Handle, usize) -> BatchId,
    pub(crate) submit: unsafe extern "C" fn(Handle, BatchId, *mut Request, usize) -> c_int,
    pub(crate) status: unsafe extern "C" fn(Handle, BatchId, usize, *mut Status) -> c_int,
    pub(crate) free_batch: unsafe extern "C" fn(Handle, BatchId) -> c_int,
}

/// Resolves `$name` from `$library` as the type of the field it fills.
macro_rules! entry {
    ($library:expr, $name:literal) => {{
        // SAFETY: the field's type is the function's signature in the release's
        // `transfer_engine_c.h`; the library stays loaded for the process's life, so the pointer
        // never dangles.
        let symbol = unsafe { $library.get(concat!($name, "\0").as_bytes()) }
            .map_err(|error| format!("{}: {error}", $name))?;
        *symbol
    }};
}

impl Api {
    /// Resolves every entry point from `library`, naming the first that is missing.
    pub(crate) fn resolve(library: &Library) -> Result<Self, String> {
        Ok(Self {
            create: entry!(library, "createTransferEngine"),
            destroy: entry!(library, "destroyTransferEngine"),
            local_name: entry!(library, "getLocalIpAndPort"),
            register: entry!(library, "registerLocalMemory"),
            unregister: entry!(library, "unregisterLocalMemory"),
            open_uncached: entry!(library, "openSegmentNoCache"),
            allocate_batch: entry!(library, "allocateBatchID"),
            submit: entry!(library, "submitTransfer"),
            status: entry!(library, "getTransferStatus"),
            free_batch: entry!(library, "freeBatchID"),
        })
    }
}
