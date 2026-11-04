//! The one module of `ops` that names `platform`'s types.
//!
//! `ops` depends on `platform` and on nothing else: it takes the CUDA boundary's devices, streams,
//! loaded modules, buffers and communicators, and hands back numbers and views. Every call into
//! that crate is made here, so a name that moves or changes shape in `platform` is a change in this
//! file and nowhere else.
//!
//! What `ops` needs from `platform`, in one list:
//!
//! - `DevicePtr`: a device address it can read as a `u64` (kernel parameters carry addresses) and
//!   move by a byte offset (a twin buffer hands out regions of one allocation).
//! - `Stream`: a `&Stream` to launch on and the raw handle a native entry receives.
//! - `Library` and `Function`: a cubin library loaded on a device, the kernel inside it, and a
//!   launch that takes everything a kernel launch needs (see [`Launch`]).
//! - `NativeLibrary` and `Symbol`: a `dlopen`ed library and a symbol in it.
//! - `Allocator` and `Buffer`: the scratch region the shadow buffers live in.
//! - `Communicator`: the collectives a layer runs through an operator.
//! - a device-to-device copy on a stream, for the shadow buffers of in-place operands.
//! - `platform::Error`, folded into [`crate::Error::Platform`].

use std::ffi::c_void;

pub use platform::Error as PlatformError;
pub use platform::{
    Allocator, Buffer, Communicator, DevicePtr, Function, Library, NativeLibrary, Stream, Symbol,
};

/// The device address a launch parameter carries.
pub(crate) fn address(pointer: DevicePtr) -> u64 {
    pointer.address()
}

/// A device address moved `bytes` further.
pub(crate) fn offset(pointer: DevicePtr, bytes: u64) -> DevicePtr {
    pointer.offset(bytes)
}

/// The raw stream handle a native entry receives.
pub(crate) fn stream_handle(stream: &Stream) -> *mut std::ffi::c_void {
    stream.handle()
}

/// Everything one cubin launch needs besides the argument array.
///
/// The thread-block shape, dynamic shared memory, cluster size, programmatic dependent launch,
/// cooperative launch and global scratch all come from the module's Triton metadata; the grid comes
/// from the operator (specs/worker.md 4.2, 5.3). Only what has no other home travels here.
pub(crate) struct Launch<'a> {
    pub blocks: [u32; 3],
    pub threads: u32,
    pub shared: u32,
    pub cluster: u32,
    pub pdl: bool,
    pub cooperative: bool,
    pub scratch: u64,
    pub parameters: &'a mut [*mut c_void],
}

/// Launch `function` on `stream`.
pub(crate) fn launch(
    function: &Function,
    stream: &Stream,
    launch: Launch<'_>,
) -> platform::Result<()> {
    let Launch {
        blocks,
        threads,
        shared,
        cluster,
        pdl,
        cooperative,
        scratch,
        parameters,
    } = launch;
    function.launch(
        stream,
        blocks,
        threads,
        shared,
        cluster,
        pdl,
        cooperative,
        scratch,
        parameters,
    )
}

/// Copy `bytes` device bytes from `source` to `destination` on `stream`.
pub(crate) fn copy_bytes(
    stream: &Stream,
    destination: DevicePtr,
    source: DevicePtr,
    bytes: u64,
) -> platform::Result<()> {
    platform::copy_bytes(stream, destination, source, bytes)
}

/// A device-to-device all-to-all of `counts` per rank through `communicator`.
///
/// # Safety
///
/// `send` and `receive` address `counts[i]` elements of `dtype` for each rank `i`, bytes that stay
/// valid until the collective completes.
pub(crate) unsafe fn all_to_all(
    communicator: &Communicator,
    send: DevicePtr,
    receive: DevicePtr,
    counts: &[u64],
    dtype: u32,
    stream: &Stream,
) -> platform::Result<()> {
    // SAFETY: forwarded from the caller.
    unsafe { communicator.all_to_all(send, receive, counts, dtype, stream) }
}
