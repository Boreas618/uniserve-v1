//! NIXL agents with the UCX backend: registration of host and device
//! memory, exchange of agent metadata, and READs the destination posts and
//! waits for on the host.
//!
//! A transfer between two agents takes three steps:
//!
//! 1. each process creates an [`Agent`] and registers the memory a transfer
//!    reads or writes ([`Agent::register`]), device memory (including
//!    memory `cuda::vmm` maps) or host memory, each region under a key of
//!    its own ([`Region::key`]);
//! 2. the source's metadata ([`Agent::metadata`]) and the source regions
//!    with their keys travel to the destination over the caller's channel,
//!    and the destination loads the metadata ([`Agent::load_peer`]);
//! 3. the destination posts a READ of spans of the source's regions into its
//!    own ([`Agent::read`]) and waits for it ([`Transfer::wait`]).
//!
//! NIXL's C API library, `libnixl_capi.so`, is loaded at run time
//! ([`load_library`]): building this crate needs no NIXL installation, and
//! a process that creates no agent never loads NIXL.

mod agent;
mod error;
mod library;
mod memory;

pub use agent::{Agent, Transfer};
pub use error::{Error, Result};
pub use library::load_library;
pub use memory::{Memory, Region, Registration, Span};
