//! The Storage Server's integration tests, through its client.

#![cfg(test)]

mod claim;
mod hosts;
mod programs;
mod remote;
mod restart;
mod server;
mod sessions;
mod trace;
mod transfer;

/// How long a test waits for the server's answer, completion or event before it fails: a bound
/// on a test that would otherwise hang, far above what any of them takes.
const PATIENCE: std::time::Duration = std::time::Duration::from_secs(30);
