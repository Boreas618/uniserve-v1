//! NIXL agents with the UCX backend: registration of host and device
//! memory, exchange of agent metadata, and READs the destination posts and
//! waits for on the host.
//!
//! A transfer between two agents takes three steps:
//!
//! 1. each process creates an [`Agent`] and registers the memory a transfer
//!    reads or writes ([`Agent::register`]), device memory (including
//!    memory `cuda::vmm` maps) or host memory, each region under a key of
//!    its own ([`Region::key`]) and for the access it needs ([`Access`]):
//!    a READ's destination for reads and writes, a source mapped read-only
//!    for reads alone;
//! 2. the source's metadata ([`Agent::metadata`]) and the source regions
//!    with their keys travel to the destination over the caller's channel,
//!    and the destination loads the metadata ([`Agent::load_peer`]);
//! 3. the destination posts a READ of spans of the source's regions into its
//!    own ([`Agent::read`]) and waits for it ([`Transfer::wait`]).
//!
//! The crate binds NIXL's C API (`libnixl_capi.so` of NIXL 1.5.1) itself,
//! resolving the entry points it calls when the library loads at run time
//! ([`load_library`]): building it needs no NIXL installation, and a
//! process that creates no agent never loads NIXL.

mod agent;
mod capi;
mod error;
mod library;
mod memory;

pub use agent::{Agent, Transfer};
pub use error::{Error, Result};
pub use library::load_library;
pub use memory::{Access, Memory, Region, Registration, Span};
