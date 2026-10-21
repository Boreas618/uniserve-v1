//! The daemon's integration tests: daemons in this process, and in one
//! process of their own, driven over the process interface as engines drive
//! them, with real child processes.

#![cfg(test)]

mod nodes;
mod supervision;
mod support;
