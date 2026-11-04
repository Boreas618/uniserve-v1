//! The CUDA boundary of UniServe: driver, virtual memory, device memory and
//! allocator, streams, events, graphs, kernel launch, NVML, NCCL, cuBLASLt and
//! cuDNN bindings. No other crate calls CUDA.
//!
//! Every native library is opened at runtime with `libloading`: building this
//! crate needs no CUDA toolkit, and a machine without a driver fails only when
//! a binding is first used, with [`Error::Load`].
//!
//! The crate exposes safe owners for each driver resource:
//!
//! | Resource | Owner | Release |
//! | --- | --- | --- |
//! | Primary context | [`PrimaryContext`], [`DeviceGuard`] | release / pop on drop; destroyed at the process's end by [`PrimaryContext::reset_all`] |
//! | Stream | [`Stream`] | `cuStreamDestroy` on drop |
//! | Event | [`Event`] | `cuEventDestroy` on drop |
//! | Graph executable | [`GraphExec`] (borrowed) | by its owner |
//! | Library loaded from a cubin or PTX image | [`Library`], [`Function`] | `cuLibraryUnload` on the last owner's drop |
//! | Native kernel library | [`NativeLibrary`] | `dlclose` on the last owner's drop |
//! | Device storage, mapped from a pool | [`Allocator`], [`Buffer`] | `cuMemUnmap` and `cuMemRelease` only when the pool is destroyed or shrunk; a freed block stays mapped for reuse |
//! | Device storage, standalone | [`DeviceAllocation`], [`PeerMapping`] | deferred to the reclaimer |
//! | Host storage | [`PinnedBuffer`] (deferred), [`HostRegistration`] (synchronous) | on drop |
//! | NCCL | [`Communicator`] | explicit `destroy`, and only after every graph that captured one of its collectives is gone |
//! | cuDNN | [`Cudnn`] | `cudnnDestroy` on drop |
//! | NVML | [`Nvml`] | `nvmlShutdown` on drop |
//!
//! Device addresses are plain `u64` values (`CUdeviceptr`, unified with host
//! pointers). Operations that read or write through an address are `unsafe`:
//! the caller keeps the storage alive until the stream work completes.
//! Releasing storage never synchronizes on the dropping thread; see
//! [`reclaim`].

mod allocator;
mod context;
mod copy;
mod cublas;
mod cudnn;
pub mod cupti;
mod device;
mod driver;
mod dtype;
mod error;
mod event;
mod graph;
mod launch;
mod library;
mod memory;
mod native;
pub mod nccl;
mod nvml;
mod pool;
pub mod reclaim;
pub mod shm;
mod stream;
mod tensormap;
mod vmm;

pub use allocator::{Allocator, Budget, Buffer, PoolKind};
pub use context::{DeviceGuard, PrimaryContext};
pub use copy::{
    CopyBatch, Pitched, PitchedBatch, SourceOrder, Strided, StridedCopy, memcpy_async,
    memset_2d_d8, memset_2d_d32, memset_d8, memset_d16, memset_d32,
};
pub use cublas::{
    Algorithm, ComputeType, Epilogue, Matmul, MatmulArgs, MatmulBytes, MatmulDescriptor,
    MatrixLayout,
};
pub use cudnn::{
    Convolution, ConvolutionArgs, ConvolutionBytes, ConvolutionPlan, Cudnn, Engine, Tensor,
};
pub use device::{
    Device, DeviceAttribute, DeviceId, DeviceUuid, MemoryInfo, profiler_start, profiler_stop,
};
pub use dtype::DataType;
pub use error::{Error, Result};
pub use event::{Event, EventFlags, IpcEventHandle};
pub use graph::{
    Capture, CaptureMode, Graph, GraphExec, GraphNode, GraphNodeKind, KernelNodeParams,
    OwnedKernelNodeParams,
};
pub use launch::{Cluster, ClusterScheduling, Function, FunctionAttribute, LaunchConfig};
pub use library::Library;
pub use memory::{DeviceAllocation, HostRegistration, PinnedBuffer};
pub use native::NativeLibrary;
pub use nccl::{Communicator, UniqueId};
pub use nvml::{Nvml, NvmlMemory};
pub use pool::VirtualPool;
pub use stream::{CaptureStatus, Stream};
pub use tensormap::{
    Interleave, L2Promotion, OobFill, Swizzle, TensorDataType, TensorMap, TensorMapSpec,
};
pub use vmm::{
    HandleType, PeerMapping, PhysicalAllocation, Protection, SegmentedMapping, ShareableHandle,
    created_bytes,
};

/// Lock `mutex`, keeping the state a panicking holder left.
///
/// Every build profile sets `panic = "abort"`, so a holder's panic ends the
/// process before another thread observes the poisoned lock. Only test
/// builds, which ignore that setting and unwind, reach the recovery; it
/// returns the guard rather than raising a second panic in the thread that
/// locks next.
pub(crate) fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Owners shared between worker threads (the builder thread, transfer
    /// pools, the reclaimer) must be `Send + Sync`; communicators move
    /// between threads.
    #[test]
    fn owners_cross_threads() {
        fn shared<T: Send + Sync>() {}
        fn movable<T: Send>() {}

        shared::<Stream>();
        shared::<Event>();
        shared::<GraphExec>();
        shared::<PrimaryContext>();
        shared::<DeviceAllocation>();
        shared::<PeerMapping>();
        shared::<PinnedBuffer>();
        shared::<HostRegistration>();
        shared::<Nvml>();
        movable::<nccl::Communicator>();
    }
}
