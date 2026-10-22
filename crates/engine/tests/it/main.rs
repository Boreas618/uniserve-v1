//! The engine's integration tests: its scheduler over the simulator, and
//! real CPU worker processes over the channel.

#![cfg(test)]

mod cache_memory;
mod canvas;
mod control_plane;
mod image_ipc;
mod media_lanes;
mod migration;
mod multiproc_ipc;
mod readout;
mod support;
