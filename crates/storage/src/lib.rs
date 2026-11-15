//! The Storage Server's protocol and its client: how every process reaches
//! the storage whose contents outlive it.
//!
//! Each host runs one Storage Server per serving user
//! (`uniserve storage-server`, the `storage-server` crate), a host service
//! (the `service` crate) listening on [`default_socket`], started with the
//! [`Settings`] a deployment states. It keeps model weights, the device prefix
//! cache, the buffer pool and the latent buffers of every worker on the host,
//! so that a restarted worker maps the retained storage instead of loading it
//! again.
//!
//! - [`protocol`]: the messages, of one version per release checked at the
//!   connection's hello, each a frame with its file descriptors on the workspace's one
//!   connection (`service::Connection`).
//! - [`queue`]: the queue pair in shared memory between a scheduler and the
//!   buffer pool.
//! - [`runtime`]: the requests and completions that travel on a queue pair.
//! - [`Client`]: one connection, used synchronously by workers and
//!   asynchronously by the head.
//! - [`settings`]: the settings a server serves with, which a deployment
//!   states and a server reports in its readiness record.
//! - [`connector`]: the settings of the Mooncake connector a server runs,
//!   which keeps committed pages in the KV storage (L3).
//! - [`manifest`]: what retained state's bytes are (the parts of logical
//!   tensors they hold, in which encodings), the chunks its storage is
//!   allocated and keyed in, and the copy manifests that turn one parallel
//!   configuration's state into another's.
//!
//! The server holds storage and never interprets it: clients compute every
//! key and define what the bytes mean. Bytes move on the clients' own
//! streams, or through the server's Transfer Server, which copies between
//! the storage the server holds at a client's request.

pub mod client;
pub mod connector;
mod error;
pub mod key;
pub mod manifest;
pub mod protocol;
pub mod queue;
pub mod runtime;
pub mod settings;

pub use client::{ClaimClient, ClaimHandoff, Client};
pub use error::{Error, Result};
pub use queue::QueuePair;
pub use settings::{Readiness, ServerSettings, Settings};

/// The host service name of the Storage Server: its command and the name of its socket and the
/// files beside it in the user's runtime directory.
pub const SERVICE: &str = "storage-server";

/// The socket of this host's Storage Server for the calling user: `storage-server.sock` in the
/// user's runtime directory ([`service::RuntimeDirectory`]), where the running server also
/// publishes its readiness record (`service::host::running`).
///
/// # Errors
///
/// Fails when the user has no runtime directory.
pub fn default_socket() -> std::io::Result<std::path::PathBuf> {
    Ok(service::RuntimeDirectory::of_user()?.socket(SERVICE))
}
pub use protocol::{
    Access, Claim, Claimant, HostStorage, Item, Key, Listing, Location, Mode, Quota, Reservation,
    Role, Scope, Segment, StorageRecord, TensorRecord,
};
