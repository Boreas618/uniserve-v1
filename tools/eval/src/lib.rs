//! Measures a serving system through its public HTTP interface: one measurement
//! point per run, written to a bundle, and the checks of bundles against their
//! controls (`specs/eval-harness.md`). The `uniserve-eval` executable is its
//! command line.

#![recursion_limit = "256"]

pub mod admin;
pub mod bundle;
pub mod check;
pub mod compare;
pub mod dataset;
pub mod load;
pub mod media;
pub mod metrics;
pub mod pipeline;
pub mod pyrandom;
pub mod record;
pub mod score;
pub mod server;
pub mod spec;
pub mod stops;
pub mod task;
pub mod telemetry;
pub mod transport;
