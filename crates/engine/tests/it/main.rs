//! The engine's integration tests: its scheduler over the simulator, and
//! real CPU worker processes over the channel.

#![cfg(test)]

mod cache_storage;
mod canvas;
mod control_plane;
mod encoder_outputs;
mod expert_rebalance;
mod image_ipc;
mod media_occupancy;
mod media_preemption;
mod migration;
mod multiproc_ipc;
mod policy_switch;
mod readout;
mod speculation;
mod support;
