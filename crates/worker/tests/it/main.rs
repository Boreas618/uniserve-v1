//! The worker runtime's integration tests.

#![cfg(test)]
#![recursion_limit = "256"]

mod buffer;
mod config;
mod gpu_storage;
mod interpreter;
mod microbatch_lanes;
mod output;
mod products;
mod registration;
mod stream_collectives;
mod transports;
