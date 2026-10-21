//! The engine's integration tests: its scheduler over the simulator, and
//! real CPU worker processes over the channel.

#![cfg(test)]

mod canvas;
mod control_plane;
mod image_ipc;
mod media_lanes;
mod multiproc_ipc;
mod readout;
mod support;
