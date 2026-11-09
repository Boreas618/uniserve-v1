//! A node's main process for one epoch.
//!
//! The daemon starts one main process per node and epoch and hands it one
//! launch file ([`proto::config::EngineArgs`]): the epoch's engine
//! configuration, what the process takes over from the one it succeeds, the
//! request address, the Storage Server namespacing, the Program Server's
//! socket, the devices, the rank and the durations. [`run`] reads it, starts
//! the control thread, one scheduler per DP rank, one executor per device and
//! the blocking pool, and serves until its daemon commands it down or the
//! epoch fails.

mod catalogue;
mod process;
mod programs;
mod rank;

pub use catalogue::Catalogue;
pub use process::run;
