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

/// `ncclConfig_t` from the NCCL 2.30 headers. Fields left at
/// `NCCL_CONFIG_UNDEF_INT` keep NCCL's environment and topology tuning; the
/// `size` and `version` fields let older libraries read the prefix they know.
#[repr(C)]
pub(super) struct Config {
    size: usize,
    magic: u32,
    version: u32,
    blocking: i32,
    cga_cluster_size: i32,
    min_ctas: i32,
    max_ctas: i32,
    net_name: *const c_char,
    split_share: i32,
    traffic_class: i32,
    comm_name: *const c_char,
    collnet_enable: i32,
    cta_policy: i32,
    shrink_share: i32,
    nvls_ctas: i32,
    channels_per_net_peer: i32,
    nvlink_centric_sched: i32,
    graph_usage_mode: i32,
    rma_contexts: i32,
    max_p2p_peers: i32,
}

const NCCL_CONFIG_UNDEF_INT: i32 = i32::MIN;

impl Config {
    pub(super) fn new(cta_policy: i32) -> Self {
        Self {
            size: size_of::<Self>(),
            magic: 0xcafe_beef,
            version: 23004,
            blocking: NCCL_CONFIG_UNDEF_INT,
            cga_cluster_size: NCCL_CONFIG_UNDEF_INT,
            min_ctas: NCCL_CONFIG_UNDEF_INT,
            max_ctas: NCCL_CONFIG_UNDEF_INT,
            net_name: std::ptr::null(),
            split_share: NCCL_CONFIG_UNDEF_INT,
            traffic_class: NCCL_CONFIG_UNDEF_INT,
            comm_name: std::ptr::null(),
            collnet_enable: NCCL_CONFIG_UNDEF_INT,
            cta_policy,
            shrink_share: NCCL_CONFIG_UNDEF_INT,
            nvls_ctas: NCCL_CONFIG_UNDEF_INT,
            channels_per_net_peer: NCCL_CONFIG_UNDEF_INT,
            nvlink_centric_sched: NCCL_CONFIG_UNDEF_INT,
            graph_usage_mode: NCCL_CONFIG_UNDEF_INT,
            rma_contexts: NCCL_CONFIG_UNDEF_INT,
            max_p2p_peers: NCCL_CONFIG_UNDEF_INT,
        }
    }
}

type Collective = unsafe extern "C" fn(Handle, Handle, usize, i32, Handle, Handle) -> NcclResult;
type Reduction =
    unsafe extern "C" fn(Handle, Handle, usize, i32, i32, Handle, Handle) -> NcclResult;
type PointToPoint = unsafe extern "C" fn(Handle, usize, i32, i32, Handle, Handle) -> NcclResult;

pub(super) struct Api {
    error_string: unsafe extern "C" fn(NcclResult) -> *const c_char,
    pub(super) get_unique_id: unsafe extern "C" fn(*mut RawUniqueId) -> NcclResult,
    pub(super) comm_init_rank_config:
        unsafe extern "C" fn(*mut Handle, i32, RawUniqueId, i32, *mut Config) -> NcclResult,
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
    // Entry points of newer releases (AlltoAll: 2.28; windows: 2.27).
    pub(super) all_to_all: Result<Collective>,
    pub(super) window_register:
        Result<unsafe extern "C" fn(Handle, Handle, usize, *mut Handle, i32) -> NcclResult>,
    pub(super) window_deregister: Result<unsafe extern "C" fn(Handle, Handle) -> NcclResult>,
    pub(super) mem_alloc: Result<unsafe extern "C" fn(*mut Handle, usize) -> NcclResult>,
    pub(super) mem_free: Result<unsafe extern "C" fn(Handle) -> NcclResult>,
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
                comm_init_rank_config: symbol(&library, "ncclCommInitRankConfig")?,
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
                window_register: symbol(&library, "ncclCommWindowRegister"),
                window_deregister: symbol(&library, "ncclCommWindowDeregister"),
                mem_alloc: symbol(&library, "ncclMemAlloc"),
                mem_free: symbol(&library, "ncclMemFree"),
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
