//! NCCL's C ABI, resolved from the process's `libnccl.so.2`.
//!
//! PyTorch loads its NCCL before the worker creates communicators, so opening
//! the soname returns that same library; no package or user path is guessed.

use std::ffi::{CStr, c_char, c_void};
use std::sync::{Arc, OnceLock};

use libloading::Library;

use crate::{Error, Result};

pub(super) const LIBNCCL: &str = "libnccl.so.2";

pub(super) type Handle = *mut c_void;
pub(super) type NcclResult = i32;

const NCCL_SUCCESS: NcclResult = 0;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct RawUniqueId {
    pub(super) bytes: [u8; 128],
}

type Collective = unsafe extern "C" fn(Handle, Handle, usize, i32, Handle, Handle) -> NcclResult;
type Reduction =
    unsafe extern "C" fn(Handle, Handle, usize, i32, i32, Handle, Handle) -> NcclResult;
type PointToPoint = unsafe extern "C" fn(Handle, usize, i32, i32, Handle, Handle) -> NcclResult;

pub(super) struct Api {
    error_string: unsafe extern "C" fn(NcclResult) -> *const c_char,
    pub(super) get_unique_id: unsafe extern "C" fn(*mut RawUniqueId) -> NcclResult,
    pub(super) comm_init_rank:
        unsafe extern "C" fn(*mut Handle, i32, RawUniqueId, i32) -> NcclResult,
    pub(super) comm_destroy: unsafe extern "C" fn(Handle) -> NcclResult,
    pub(super) comm_abort: unsafe extern "C" fn(Handle) -> NcclResult,
    pub(super) all_reduce: Reduction,
    pub(super) reduce_scatter: Reduction,
    pub(super) all_gather: Collective,
    /// `ncclBroadcast(send, recv, count, type, root, comm, stream)`.
    pub(super) broadcast: Reduction,
    pub(super) send: PointToPoint,
    pub(super) recv: PointToPoint,
    pub(super) group_start: unsafe extern "C" fn() -> NcclResult,
    pub(super) group_end: unsafe extern "C" fn() -> NcclResult,
    // An entry point of newer releases (2.28).
    pub(super) all_to_all: Result<Collective>,
    // Declared last: the library outlives every copied function pointer.
    _library: Library,
}

/// # Safety
/// `T` must be the function-pointer type of the symbol's C prototype.
unsafe fn symbol<T: Copy>(library: &Library, name: &'static str) -> Result<T> {
    // SAFETY: forwarded from the caller.
    unsafe { library.get::<T>(name.as_bytes()) }
        .map(|symbol| *symbol)
        .map_err(|source| Error::Symbol {
            library: LIBNCCL,
            symbol: name,
            source: Arc::new(source),
        })
}

pub(super) fn api() -> Result<&'static Api> {
    static API: OnceLock<Result<Api>> = OnceLock::new();

    API.get_or_init(|| {
        // SAFETY: loading NCCL runs only its own initializers.
        let library = unsafe { Library::new(LIBNCCL) }.map_err(|source| Error::Load {
            library: LIBNCCL,
            source: Arc::new(source),
        })?;
        // SAFETY: each signature matches the `nccl.h` prototype of the symbol.
        unsafe {
            Ok(Api {
                error_string: symbol(&library, "ncclGetErrorString")?,
                get_unique_id: symbol(&library, "ncclGetUniqueId")?,
                comm_init_rank: symbol(&library, "ncclCommInitRank")?,
                comm_destroy: symbol(&library, "ncclCommDestroy")?,
                comm_abort: symbol(&library, "ncclCommAbort")?,
                all_reduce: symbol(&library, "ncclAllReduce")?,
                reduce_scatter: symbol(&library, "ncclReduceScatter")?,
                all_gather: symbol(&library, "ncclAllGather")?,
                broadcast: symbol(&library, "ncclBroadcast")?,
                send: symbol(&library, "ncclSend")?,
                recv: symbol(&library, "ncclRecv")?,
                group_start: symbol(&library, "ncclGroupStart")?,
                group_end: symbol(&library, "ncclGroupEnd")?,
                all_to_all: symbol(&library, "ncclAlltoAll"),
                _library: library,
            })
        }
    })
    .as_ref()
    .map_err(Clone::clone)
}

impl Api {
    pub(super) fn check(&self, status: NcclResult, call: &'static str) -> Result<()> {
        if status == NCCL_SUCCESS {
            return Ok(());
        }
        // SAFETY: NCCL returns a static, NUL-terminated description for every
        // result code, owned by the never-unloaded library.
        let message = unsafe {
            let text = (self.error_string)(status);
            if text.is_null() {
                "unrecognized NCCL result"
            } else {
                CStr::from_ptr(text)
                    .to_str()
                    .unwrap_or("non-UTF-8 NCCL result")
            }
        };
        Err(Error::Nccl {
            call,
            code: status,
            message,
        })
    }
}
