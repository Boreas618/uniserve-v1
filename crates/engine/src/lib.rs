//! A node's main process for one epoch.
//!
//! The daemon starts one main process per node and epoch and hands it one
//! launch file ([`proto::config::EngineArgs`]): the epoch's engine
//! configuration, what the process takes over from the one it succeeds, the
//! request address, the Storage Server namespacing, the Build Server's
//! socket, the devices, the rank and the durations. [`run`] reads it, starts
//! the control thread, one scheduler per DP rank and one executor per device,
//! and serves until its daemon commands it down or the epoch fails. [`serve`]
//! is the same control thread over two given connections and a [`Start`], the
//! function that builds the started epoch ([`Engine`]): the binary's loads
//! ranks, a test's builds fakes. [`record`] runs the start to the point where
//! every device has recorded the packages its runners launch, and serves
//! nothing.

mod catalogue;
pub mod data_plane;
pub mod executor;
mod modules;
mod process;
mod rank;
pub mod scheduler;
pub mod storage_client;
/// The Storage Server and launch the unit tests start from, which the integration tests share.
#[cfg(test)]
#[path = "../tests/support/mod.rs"]
mod support;

pub use catalogue::resolve;
pub use process::ranks::text_layout;
pub use process::record::{Recorded, record};
pub use process::{Engine, Start, connect, run, serve};
pub use scheduler::Error;
