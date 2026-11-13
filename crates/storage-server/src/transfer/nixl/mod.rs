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
//! The module binds NIXL's C API (`libnixl_capi.so` of NIXL 1.5.1) itself,
//! resolving the entry points it calls when the library loads at run time
//! ([`load_library`]): building the Storage Server needs no NIXL
//! installation, no binary links NIXL, and a process that creates no agent
//! (every process but a Transfer Server that READs from or is read by
//! another host) never loads it. Read-only host storage needs NIXL and UCX
//! built with the patches in `crates/storage-server/patches` (`docs/nixl.md`).

mod agent;
mod capi;
mod error;
mod library;
mod regions;

pub(crate) use agent::{Agent, Transfer};
pub(crate) use error::{Error, Result};
pub(crate) use library::load_library;
pub(crate) use regions::{Access, Region, Registration, Segment, Span};

#[cfg(test)]
mod tests;
