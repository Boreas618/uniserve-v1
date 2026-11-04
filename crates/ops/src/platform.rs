//! The one module of `ops` that names `platform`'s types.
//!
//! `ops` depends on `platform` and on nothing else: it takes the CUDA boundary's devices, streams,
//! loaded libraries, buffers and communicators, and hands back numbers and views. Every call into
//! that crate is made here, so a name that moves or changes shape in `platform` is a change in this
//! file and nowhere else.
//!
//! What `ops` needs from `platform`, in one list:
//!
//! - [`Device`]: where a cubin library is loaded.
//! - [`Stream`] and its raw handle: where a launch is issued, and what a native entry receives.
//! - [`Library`] and [`Function`]: a cubin library loaded on a device, the kernel inside it, a
//!   [`LaunchConfig`] that carries a three-dimensional grid, the block, dynamic shared memory, a
//!   cluster, programmatic dependent launch and cooperative launch, and a function attribute for a
//!   dynamic shared-memory size beyond the driver's default.
//! - [`NativeLibrary`] and its typed entry lookup: a `dlopen`ed library and the symbol in it.
//! - [`Allocator`], [`PoolKind`] and [`Buffer`]: the scratch region the shadow buffers live in.
//! - a device-to-device copy on a stream, for the shadow buffers of in-place operands.
//! - [`Communicator`]: the collectives a layer runs through an operator.
//! - a device-to-device copy on a stream, for the shadow buffers of in-place operands.
//! - `platform::Error`, folded into [`crate::Error::Platform`].
//!
//! Device addresses are `platform`'s `u64` values; `ops` carries them in its own [`DevicePtr`] so a
//! view's address cannot be confused with a scalar.

use std::ffi::c_void;

pub use platform::DataType as CollectiveType;
pub use platform::DevicePtr as PlatformDevicePtr;
pub use platform::Error as PlatformError;
pub use platform::nccl::{Communicator, ReduceOp};
pub use platform::{
    Allocator, Buffer, Cluster, Device, DeviceRange, Function, FunctionAttribute, LaunchConfig,
    Library, NativeLibrary, PoolKind, Stream,
};

pub use crate::tensor::DevicePtr;

use crate::tensor::DType;

/// The device address a launch parameter carries.
pub(crate) fn address(pointer: DevicePtr) -> u64 {
    pointer.address()
}

/// A device address moved `bytes` further.
pub(crate) fn offset(pointer: DevicePtr, bytes: u64) -> DevicePtr {
    DevicePtr::new(pointer.address() + bytes)
}

/// The raw stream handle a native entry receives (`void* stream`).
pub(crate) fn stream_handle(stream: &Stream) -> *mut c_void {
    stream.raw() as *mut c_void
}

/// Everything one cubin launch needs besides the argument array.
///
/// The thread-block shape, dynamic shared memory, cluster size, programmatic dependent launch,
/// cooperative launch and global scratch all come from the module's Triton metadata; the grid comes
/// from the operator (specs/worker.md 4.2, 5.3). Only what has no other home travels here.
pub(crate) struct Launch {
    pub blocks: [u32; 3],
    pub threads: u32,
    pub shared: u32,
    pub cluster: u32,
    pub pdl: bool,
    pub cooperative: bool,
}

impl Launch {
    /// The platform configuration of this launch.
    fn config(&self) -> LaunchConfig {
        LaunchConfig {
            grid: self.blocks,
            block: [self.threads, 1, 1],
            shared_memory: self.shared,
            // A cluster of one block is no cluster: the attribute is only set when the module asks
            // for a real one.
            cluster: (self.cluster > 1).then(|| Cluster::new([self.cluster, 1, 1])),
            pdl: self.pdl,
            cooperative: self.cooperative,
        }
    }
}

/// Launch `function` on `stream`.
pub(crate) fn launch(
    function: &Function,
    stream: &Stream,
    launch: Launch,
    parameters: &mut [*mut c_void],
) -> platform::Result<()> {
    function.launch_ex(stream, &launch.config(), parameters)
}

/// The driver's default limit on a kernel's dynamic shared memory, in bytes.
///
/// A kernel that declares more needs [`FunctionAttribute::MaxDynamicSharedMemory`] set before its
/// first launch.
pub(crate) const DEFAULT_SHARED_LIMIT: u32 = 48 * 1024;

/// Lift the function's dynamic shared-memory limit to `bytes`.
pub(crate) fn raise_shared_limit(function: &Function, bytes: u32) -> platform::Result<()> {
    function.set_attribute(FunctionAttribute::MaxDynamicSharedMemory, bytes as i32)
}

/// Load a cubin, fatbin or PTX image as a library on `device`.
pub(crate) fn load_library(device: &Device, image: &[u8]) -> platform::Result<Library> {
    Library::load(device, image)
}

/// The kernel named `name` in `library`.
pub(crate) fn library_kernel(library: &Library, name: &str) -> platform::Result<Function> {
    library.kernel(name)
}

/// Resolve the C entry `name` in a native library with the signature `F`.
///
/// # Safety
///
/// `F` is the signature the package declares for `name`; a wrong one is undefined behaviour when
/// the entry is called.
pub(crate) unsafe fn native_entry<F: Copy>(
    library: &NativeLibrary,
    name: &str,
) -> platform::Result<F> {
    // SAFETY: forwarded from the caller.
    unsafe { library.entry::<F>(name) }
}

/// Copy `bytes` device bytes from `source` to `destination` on `stream`.
pub(crate) fn copy_bytes(
    stream: &Stream,
    destination: DevicePtr,
    source: DevicePtr,
    bytes: u64,
) -> crate::Result<()> {
    let bytes = usize::try_from(bytes).map_err(|_| {
        crate::Error::Shape(format!("a {bytes}-byte copy is larger than usize::MAX"))
    })?;
    // SAFETY: both ranges address device memory the caller keeps alive until the copy completes
    // and until the stream work after it finishes.
    unsafe { platform::memcpy_async(destination.address(), source.address(), bytes, stream) }?;
    Ok(())
}

/// A device-to-device all-to-all of `count` elements per rank through `communicator`.
///
/// # Safety
///
/// `send` and `receive` address `count` elements of `dtype` for each rank, bytes that stay valid
/// until the collective completes.
pub(crate) fn all_to_all(
    communicator: &mut Communicator,
    send: DevicePtr,
    receive: DevicePtr,
    count: usize,
    dtype: CollectiveType,
    stream: &Stream,
) -> crate::Result<()> {
    let bytes = dtype.bytes() * count;
    let send = DeviceRange::new(PlatformDevicePtr::from_address(send.address()), bytes);
    let receive = DeviceRange::new(PlatformDevicePtr::from_address(receive.address()), bytes);
    communicator
        .all_to_all(stream, send, receive, count, dtype)
        .map_err(crate::Error::Platform)
}

/// The collective element type for one of `ops`' element types, when NCCL speaks it.
///
/// A sub-byte type has no collective counterpart: it travels inside a program's own encoded layout,
/// never as a buffer a collective reads.
pub fn collective_type(dtype: DType) -> Option<CollectiveType> {
    Some(match dtype {
        DType::U8 => CollectiveType::U8,
        DType::I32 => CollectiveType::I32,
        DType::I64 => CollectiveType::I64,
        DType::F16 => CollectiveType::F16,
        DType::BF16 => CollectiveType::Bf16,
        DType::F32 => CollectiveType::F32,
        DType::F8E4M3 => CollectiveType::F8E4M3,
        DType::F4E2M1 => return None,
    })
}
