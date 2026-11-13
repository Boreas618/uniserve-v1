//! A node's main process for one epoch.
//!
//! The daemon starts one main process per node and epoch and hands it one
//! launch file ([`proto::config::EngineArgs`]): the epoch's engine
//! configuration, what the process takes over from the one it succeeds, the
//! request address, the Storage Server namespacing, the Build Server's
//! socket, the devices, the rank and the durations. [`run`] reads it, starts
//! the control thread, one scheduler per DP rank, one executor per device and
//! the blocking pool, and serves until its daemon commands it down or the
//! epoch fails. [`record`] runs the same start to the point where every
//! device has recorded the packages its runners launch, and serves nothing.

mod catalogue;
pub mod data_plane;
pub mod executor;
#[expect(
    dead_code,
    reason = "the process, the one caller of the module installation, is not declared in the crate"
)]
mod modules;
/// The parts of the main process that start its ranks and record their kernels; the process's
/// control thread and its start are not declared in the crate yet.
mod process {
    #[expect(
        dead_code,
        reason = "the main process's start, the caller of the served rank start, is not declared                   in the crate"
    )]
    pub(crate) mod ranks;
    pub(crate) mod record;
}
#[expect(
    dead_code,
    reason = "the main process's start and the media rank, the callers of the rank's accessors               and of its parallel shape, are not declared in the crate"
)]
mod rank;
pub mod scheduler;
pub mod storage_client;

pub use catalogue::resolve;
pub use process::record::{Recorded, record};
pub use scheduler::Error;
