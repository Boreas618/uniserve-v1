//! The interfaces' integration tests: requests and events through their
//! messages, frames that carry descriptors between two ends of a connection, and the
//! configuration sections as records of another build.

#![cfg(test)]

mod config;
mod descriptors;
mod request;
mod versions;
