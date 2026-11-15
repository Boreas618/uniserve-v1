//! NCCL's C ABI, resolved from `libnccl.so.2`.
//!
//! NVIDIA's NCCL package installs the library under its soname on the loader's default search
//! path, and it is opened by that name. The bindings are generated from the `nccl.h` of the
//! release [`sys::NCCL_VERSION_CODE`] names, which is also the oldest library accepted: an older
//! one is refused with [`Error::Version`] at load. That release is the project's pin, and older
//! ones are not merely missing attributes: NCCL 2.27.7 crashes on the first replay of a CUDA graph
//! that holds a collective. Every entry point the crate calls exists in that release, so all are
//! resolved at load.

use std::ffi::{CStr, c_char, c_void};
use std::sync::{Arc, OnceLock};

use libloading::Library;

use crate::sys::nccl as sys;
use crate::{Error, Result};

pub(super) const LIBNCCL: &str = "libnccl.so.2";

/// What a refused library is told to install.
const NCCL_PACKAGE: &str = "NCCL 2.30.4 or later (NVIDIA's libnccl2 package)";

pub(super) type Handle = *mut c_void;
pub(super) type NcclResult = sys::ncclResult_t;

/// `NCCL_CONFIG_UNDEF_INT`: an attribute left to NCCL's default. The header defines it as
/// `INT_MIN` of `<limits.h>`, which is `i32::MIN` for the 32-bit `int` of every target the
/// crate builds for.
const NCCL_CONFIG_UNDEF_INT: i32 = i32::MIN;

impl sys::ncclConfig_t {
    /// `NCCL_CONFIG_INITIALIZER`: every attribute left to NCCL's default, stating the size, magic
    /// and version of the header the bindings are generated from. NCCL reads a config's attributes
    /// as of the version it states.
    pub(super) fn initializer() -> Self {
        Self {
            size: std::mem::size_of::<Self>(),
            magic: sys::NCCL_API_MAGIC,
            version: sys::NCCL_VERSION_CODE,
            blocking: NCCL_CONFIG_UNDEF_INT,
            cgaClusterSize: NCCL_CONFIG_UNDEF_INT,
            minCTAs: NCCL_CONFIG_UNDEF_INT,
            maxCTAs: NCCL_CONFIG_UNDEF_INT,
            netName: std::ptr::null(),
            splitShare: NCCL_CONFIG_UNDEF_INT,
            trafficClass: NCCL_CONFIG_UNDEF_INT,
            commName: std::ptr::null(),
            collnetEnable: NCCL_CONFIG_UNDEF_INT,
            CTAPolicy: NCCL_CONFIG_UNDEF_INT,
            shrinkShare: NCCL_CONFIG_UNDEF_INT,
            nvlsCTAs: NCCL_CONFIG_UNDEF_INT,
            nChannelsPerNetPeer: NCCL_CONFIG_UNDEF_INT,
            nvlinkCentricSched: NCCL_CONFIG_UNDEF_INT,
            graphUsageMode: NCCL_CONFIG_UNDEF_INT,
            numRmaCtx: NCCL_CONFIG_UNDEF_INT,
            maxP2pPeers: NCCL_CONFIG_UNDEF_INT,
        }
    }
}

type Collective = unsafe extern "C" fn(Handle, Handle, usize, i32, Handle, Handle) -> NcclResult;
type Reduction =
    unsafe extern "C" fn(Handle, Handle, usize, i32, i32, Handle, Handle) -> NcclResult;
type PointToPoint = unsafe extern "C" fn(Handle, usize, i32, i32, Handle, Handle) -> NcclResult;

pub(super) struct Api {
    error_string: unsafe extern "C" fn(NcclResult) -> *const c_char,
    get_version: unsafe extern "C" fn(*mut i32) -> NcclResult,
    pub(super) get_unique_id: unsafe extern "C" fn(*mut sys::ncclUniqueId) -> NcclResult,
    /// `ncclCommInitRankConfig(comm, nranks, id, rank, config)`.
    pub(super) comm_init_rank_config: unsafe extern "C" fn(
        *mut Handle,
        i32,
        sys::ncclUniqueId,
        i32,
        *mut sys::ncclConfig_t,
    ) -> NcclResult,
    /// `ncclCommSplit(comm, color, key, newcomm, config)`; a NULL config takes the defaults.
    pub(super) comm_split:
        unsafe extern "C" fn(Handle, i32, i32, *mut Handle, *mut sys::ncclConfig_t) -> NcclResult,
    pub(super) comm_destroy: unsafe extern "C" fn(Handle) -> NcclResult,
    pub(super) comm_abort: unsafe extern "C" fn(Handle) -> NcclResult,
    pub(super) all_reduce: Reduction,
    pub(super) all_gather: Collective,
    /// `ncclBroadcast(send, recv, count, type, root, comm, stream)`.
    pub(super) broadcast: Reduction,
    pub(super) send: PointToPoint,
    pub(super) recv: PointToPoint,
    pub(super) group_start: unsafe extern "C" fn() -> NcclResult,
    pub(super) group_end: unsafe extern "C" fn() -> NcclResult,
    pub(super) all_to_all: Collective,
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
        let api = unsafe {
            Api {
                error_string: symbol(&library, "ncclGetErrorString")?,
                get_version: symbol(&library, "ncclGetVersion")?,
                get_unique_id: symbol(&library, "ncclGetUniqueId")?,
                comm_init_rank_config: symbol(&library, "ncclCommInitRankConfig")?,
                comm_split: symbol(&library, "ncclCommSplit")?,
                comm_destroy: symbol(&library, "ncclCommDestroy")?,
                comm_abort: symbol(&library, "ncclCommAbort")?,
                all_reduce: symbol(&library, "ncclAllReduce")?,
                all_gather: symbol(&library, "ncclAllGather")?,
                broadcast: symbol(&library, "ncclBroadcast")?,
                send: symbol(&library, "ncclSend")?,
                recv: symbol(&library, "ncclRecv")?,
                group_start: symbol(&library, "ncclGroupStart")?,
                group_end: symbol(&library, "ncclGroupEnd")?,
                all_to_all: symbol(&library, "ncclAlltoAll")?,
                _library: library,
            }
        };

        let mut version = 0;
        // SAFETY: NCCL writes one int to the local `version`.
        api.check(unsafe { (api.get_version)(&mut version) }, "ncclGetVersion")?;
        require_version(version)?;
        Ok(api)
    })
    .as_ref()
    .map_err(Clone::clone)
}

/// Refuses an NCCL older than the `nccl.h` the bindings are generated from. Both are NCCL version
/// codes, `major * 10000 + minor * 100 + patch` from release 2.9 on.
fn require_version(version: i32) -> Result<()> {
    let required = sys::NCCL_VERSION_CODE as i32;
    if version >= required {
        return Ok(());
    }
    Err(Error::Version {
        library: LIBNCCL,
        found: release(version),
        required: release(required),
        package: NCCL_PACKAGE,
    })
}

/// An NCCL version code as its release, such as `NCCL 2.30.4` for 23004.
fn release(code: i32) -> String {
    format!("NCCL {}.{}.{}", code / 10000, code / 100 % 100, code % 100)
}

impl Api {
    pub(super) fn check(&self, status: NcclResult, call: &'static str) -> Result<()> {
        if status == sys::ncclSuccess {
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
            // `ncclResult_t` is a C enum; the error carries the value NCCL documents.
            code: status as i32,
            message,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An NCCL older than the generated header — the host's 2.27.7, which crashes on the first
    /// replay of a graph that holds a collective — is refused by name with the package to
    /// install; the header's release and later ones are accepted.
    #[test]
    fn an_nccl_older_than_the_bindings_is_refused() {
        assert!(require_version(sys::NCCL_VERSION_CODE as i32).is_ok());
        assert!(require_version(23100).is_ok());

        let error = require_version(22707).expect_err("2.27.7 predates the bindings");
        let text = error.to_string();
        assert!(
            matches!(
                error,
                Error::Version {
                    library: LIBNCCL,
                    ..
                }
            ),
            "{text}"
        );
        assert!(
            text.contains("NCCL 2.27.7") && text.contains("NCCL 2.30.4"),
            "{text}"
        );
        assert!(text.contains(NCCL_PACKAGE), "{text}");
    }

    /// The initializer states the generated header's size, magic and version, which NCCL checks
    /// before it reads any attribute.
    #[test]
    fn the_config_initializer_states_the_headers_version() {
        let config = sys::ncclConfig_t::initializer();
        assert_eq!(config.size, std::mem::size_of::<sys::ncclConfig_t>());
        assert_eq!(config.magic, sys::NCCL_API_MAGIC);
        assert_eq!(config.version, sys::NCCL_VERSION_CODE);
        assert_eq!(config.graphUsageMode, NCCL_CONFIG_UNDEF_INT);
    }
}
