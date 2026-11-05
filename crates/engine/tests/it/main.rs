//! The engine's integration tests: its scheduler over the simulator, and
//! real CPU executor processes over the channel.

#![cfg(test)]
// A case that loses its `#[test]` compiles as a private function and runs nothing;
// the only signal is a dead-code warning, and a warning nobody reads during a
// green run is how one stayed silent. Denied here so it is a build failure.
#![deny(dead_code)]

mod boot;
mod cache_storage;
mod call_counts;
mod canvas;
mod control_plane;
mod encoder_outputs;
mod error_taxonomy;
mod expert_rebalance;
mod in_process;
mod in_process_completion;
mod in_process_images;
mod in_process_groups;
mod in_process_media;
mod in_process_stalls;
mod in_process_worlds;
mod media_occupancy;
mod media_preemption;
mod migration;
mod multiproc_ipc;
mod policy_switch;
mod process_link;
mod readout;
mod speculation;
mod support;
