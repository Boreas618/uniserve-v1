//! Bounded host work: input preparation, copies and media codecs.

pub(crate) mod image;
mod lane;

pub use lane::{HostAction, HostLane, HostTask};
