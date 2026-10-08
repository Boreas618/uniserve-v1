//! Bounded host work: input preparation, copies and media codecs.

mod lane;
pub(crate) mod mux;
pub(crate) mod reader;
pub(crate) mod task;

pub use lane::{HostAction, HostLane, HostTask};
