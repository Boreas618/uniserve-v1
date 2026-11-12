//! What orders host work behind a device's streams: the events storage owners wait on before
//! they release storage that device work addressed.

pub mod events;

pub use events::{DRAIN_LIMIT, EventPool, JobId, Release, Submitter, drain_event};
