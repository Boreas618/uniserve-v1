//! The worker runtime's integration tests.

#![cfg(test)]
#![recursion_limit = "256"]

mod buffer;
mod config;
mod device;
mod gpu_storage;
mod interpreter;
mod output;
mod products;
mod registration;
mod transports;
