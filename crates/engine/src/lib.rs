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
mod modules;
mod process;
mod rank;

pub use catalogue::resolve;
pub use process::{Recorded, record, run};
