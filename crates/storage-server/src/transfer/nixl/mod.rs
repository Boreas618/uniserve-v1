//! NIXL agents with the UCX backend: registration of host and device
//! storage, exchange of agent metadata, and READs the destination posts and
//! waits for on the host.
//!
//! A transfer between two agents takes three steps:
//!
//! 1. each process creates an [`Agent`] and registers the storage a transfer
//!    reads or writes ([`Agent::register`]), device storage (including
//!    storage `platform::vmm` maps) or host storage, each region under a key of
//!    its own ([`Region::key`]) and for the access it needs ([`Access`]):
//!    a READ's destination for reads and writes, a source mapped read-only
//!    for reads alone;
//! 2. the source's metadata ([`Agent::metadata`]) and the source regions
//!    with their keys travel to the destination over the caller's channel,
//!    and the destination loads the metadata ([`Agent::load_peer`]);
//! 3. the destination posts a READ of spans of the source's regions into its
//!    own ([`Agent::read`]) and polls it until it completes ([`Transfer::poll`]).
//!
//! The module binds NIXL's C API itself, resolving the entry points it calls when the library
//! loads at run time from the loader's default search path ([`library`]): building the Storage
//! Server needs no NIXL installation, no binary links NIXL, and a process that creates no agent
//! (every process but a Transfer Server that READs from or is read by another host) never loads
//! it. A registration for reads alone ([`Access::Read`]) needs the read-only registration of the
//! NIXL and UCX packages `docs/nixl.md` names.

mod agent;
mod capi;
mod library;
mod ports;
mod regions;

pub(crate) use agent::{Agent, Backend, Transfer};
pub(crate) use regions::{Access, Region, Registration, Segment, Span};

pub(crate) use crate::Error;

/// The NIXL release whose C API these bindings bind.
pub(crate) const RELEASE: &str = "1.5.1";

pub(crate) type Result<T, E = Error> = std::result::Result<T, E>;

/// `Ok` for a call that returned `SUCCESS`, the failure of `call` otherwise.
pub(crate) fn check(call: &'static str, status: capi::Status) -> Result<()> {
    if status == capi::SUCCESS {
        Ok(())
    } else {
        Err(Error::Nixl { call, status })
    }
}

/// The name of a NIXL status, as its C API declares it.
pub(crate) fn status_name(status: i32) -> &'static str {
    capi::status_name(status)
}

#[cfg(test)]
mod tests;
