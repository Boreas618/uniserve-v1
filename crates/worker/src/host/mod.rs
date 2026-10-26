//! Bounded host work: input preparation, copies and media codecs.

pub(crate) mod image;
mod pool;

pub(crate) use pool::{HostAction, HostPool, HostTask};
