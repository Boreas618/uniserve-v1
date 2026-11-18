//! Mooncake Transfer Engine: registration of host and device storage, the segment another host's
//! engine opens it by, and READs and WRITEs the engine posts and the host polls.
//!
//! A transfer between two engines takes three steps:
//!
//! 1. each process creates an [`Engine`] and registers the storage a transfer reads or writes
//!    ([`Engine::register`]), device storage (including storage `platform::vmm` maps) or host
//!    storage, remotely accessible;
//! 2. the name of the source's engine ([`Engine::name`], the segment name) and the addresses of
//!    its registered ranges travel to the other side over the caller's channel, and that side
//!    opens the segment ([`Engine::open`]), which fetches its description from the engine by
//!    the peer-to-peer handshake;
//! 3. the reading side posts a READ of spans of the segment into its own registered storage
//!    ([`Engine::read`]), or the writing side a WRITE of its own into the segment
//!    ([`Engine::write`]), and polls the batch until it completes ([`Batch::poll`]).
//!
//! The module binds the Transfer Engine's C API (`transfer_engine_c.h`) itself, resolving the
//! entry points it calls when the library loads at run time from the loader's default search
//! path ([`library`]): building the Storage Server needs no Mooncake installation, no binary
//! links the Transfer Engine, and a process that starts no engine (every process but a Transfer
//! Server whose handoffs take the Mooncake transport) never loads it.

mod capi;
mod engine;
mod library;
mod topology;

pub(crate) use capi::SegmentId;
pub(crate) use engine::{Batch, Engine, Registration, Span};
pub(crate) use topology::Topology;

pub(crate) use crate::Error;

/// The Mooncake release whose Transfer Engine C API these bindings bind: the release the
/// workspace's `mooncake_store` binding pins.
pub(crate) const RELEASE: &str = "0.3.13.post1";

pub(crate) type Result<T, E = Error> = std::result::Result<T, E>;

/// The settings the Transfer Engine reads from the environment when an engine starts, which the
/// `storage-server` command states in its process before any thread starts
/// ([`crate::command::run`]), so its Transfer Server and its calibration step, and every child
/// it starts, hold them: device storage registers through its dma-buf (`WITH_NVIDIA_PEERMEM=0`),
/// as UCX registers it for NIXL, since the server's device storage is CUDA VMM allocations,
/// which the engine's default, a registration of the address through nvidia-peermem, refuses.
pub(crate) const ENVIRONMENT: [(&str, &str); 1] = [("WITH_NVIDIA_PEERMEM", "0")];

/// The name of a return code of the C API, as `include/error.h` names it.
pub(crate) fn code_name(code: i32) -> &'static str {
    capi::code_name(code)
}

/// The name of a transfer's status, as `transfer_engine_c.h` names it.
pub(crate) fn status_name(status: i32) -> &'static str {
    capi::status_name(status)
}
