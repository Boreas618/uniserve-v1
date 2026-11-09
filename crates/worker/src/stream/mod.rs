//! Execution streams, the rank's set of them, and the events that order work
//! across them.

pub mod events;
pub mod issue;
#[cfg(test)]
pub mod recorded;

pub use events::{DRAIN_LIMIT, Deferral, Deferred, EventPool, JobId, Submitter, drain_event};
pub use issue::{Fill, HostFunction, Issue, RowFill};
