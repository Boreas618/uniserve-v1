//! CUDA driver, virtual memory, NVML and NCCL bindings for the UniServe worker.
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
//! | Stream | [`Stream`], [`GreenContext`] (SM partitions) | `cuStreamDestroy` on drop |
//! | Event | [`Event`] | `cuEventDestroy` on drop |
//! | Graph executable | [`GraphExec`] (borrowed) | by its owner |
//! | Device memory | [`DeviceAllocation`], [`PeerMapping`] | deferred to the reclaimer |
//! | Host memory | [`PinnedBuffer`] (deferred), [`HostRegistration`] (synchronous) | on drop |
//! | NCCL | [`nccl::Communicator`] | explicit `destroy` |
//! | NVML | [`Nvml`] | `nvmlShutdown` on drop |
//!
//! Device addresses are plain `u64` values (`CUdeviceptr`, unified with host
//! pointers). Operations that read or write through an address are `unsafe`:
//! the caller keeps the memory alive until the stream work completes.
//! Releasing memory never synchronizes on the dropping thread; see
//! [`reclaim`].

mod context;
mod copy;
mod device;
mod driver;
mod error;
mod event;
mod graph;
mod green;
mod memory;
pub mod nccl;
mod nvml;
mod pool;
pub mod reclaim;
pub mod shm;
mod stream;
mod vmm;

pub use context::{DeviceGuard, PrimaryContext};
pub use copy::{
    CopyBatch, Pitched, PitchedBatch, SourceOrder, Strided, StridedCopy, memcpy_async,
    memset_2d_d8, memset_2d_d32, memset_d8, memset_d16, memset_d32,
};
pub use device::{
    DeviceAttribute, DeviceId, DeviceUuid, MemoryInfo, profiler_start, profiler_stop,
};
pub use error::{Error, Result};
pub use event::{Event, EventFlags, IpcEventHandle};
pub use graph::GraphExec;
pub use green::GreenContext;
pub use memory::{DeviceAllocation, HostRegistration, PinnedBuffer};
pub use nvml::{Nvml, NvmlMemory};
pub use pool::VirtualPool;
pub use stream::{CaptureStatus, Stream};
pub use vmm::{HandleType, PeerMapping, Protection, SegmentedMapping, ShareableHandle};

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

    /// Owners shared between worker threads (lanes, transfer pools, the
    /// reclaimer) must be `Send + Sync`; communicators move between threads.
    #[test]
    fn owners_cross_threads() {
        fn shared<T: Send + Sync>() {}
        fn movable<T: Send>() {}

        shared::<Stream>();
        shared::<Event>();
        shared::<GraphExec>();
        shared::<PrimaryContext>();
        shared::<GreenContext>();
        shared::<DeviceAllocation>();
        shared::<PeerMapping>();
        shared::<PinnedBuffer>();
        shared::<HostRegistration>();
        shared::<Nvml>();
        movable::<nccl::Communicator>();
    }
}
