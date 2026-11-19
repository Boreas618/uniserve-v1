//! The Storage Server's protocol and its client: how every process reaches
//! the storage whose contents outlive it.
//!
//! Each deployment runs one Storage Server on each of its nodes
//! (`uniserve storage-server`, the `storage-server` crate), a server of the
//! deployment (`service::host`) listening on its [`socket`], started with the
//! [`Settings`] the deployment states. It keeps the model weights, the device
//! prefix cache, the buffer pool and the latent buffers of every worker of the
//! deployment on the node, so that a restarted worker maps the retained storage
//! instead of loading it again.
//!
//! - [`protocol`]: the messages, in a window of versions agreed at the
//!   connection's hello, each a frame with its file descriptors on the workspace's one
//!   connection (`service::Connection`).
//! - [`queue`]: the queue pair in shared memory between a scheduler and the
//!   buffer pool, with the doorbell each end sleeps on.
//! - [`runtime`]: the requests and completions that travel on a queue pair.
//! - [`Client`]: one connection, used synchronously by workers and
//!   asynchronously by the head.
//! - [`settings`]: the settings a server serves with, which a deployment
//!   states and a server reports in its readiness record, and which of them
//!   a running server takes and which need a new server process.
//! - [`events`]: the KV events of the pages a cache holds, in SGLang's ZMQ
//!   format, which KV-aware routers outside the deployment read.
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
pub mod events;
pub mod key;
pub mod manifest;
pub mod protocol;
pub mod queue;
pub mod runtime;
pub mod settings;

pub use client::{ClaimClient, ClaimHandoff, Client};
pub use error::{Error, Result};
pub use queue::{Interest, QueueDescriptors, QueuePair, Signals};
pub use settings::{Readiness, ServerSettings, Settings};

/// The server name of the Storage Server: its command and the name of its socket and the files
/// beside it in its deployment's runtime directory.
pub const SERVICE: &str = "storage-server";

/// The socket of the Storage Server of `deployment` in the runtime directory `runtime` (the
/// user's, [`service::RuntimeDirectory::of_user`], on every node): `storage-server.sock` in the
/// deployment's runtime directory ([`service::RuntimeDirectory::deployment`]), where the running
/// server also publishes its readiness record (`service::host::running`).
///
/// # Errors
///
/// Fails when `deployment` names no deployment's directory, or it cannot be created.
pub fn socket(
    runtime: &service::RuntimeDirectory,
    deployment: &str,
) -> std::io::Result<std::path::PathBuf> {
    Ok(runtime.deployment(deployment)?.socket(SERVICE))
}
pub use protocol::{
    Access, Claim, Claimant, HostStorage, Item, Key, Listing, Location, Mode, Quota, Reservation,
    Role, Scope, Segment, StorageRecord, TensorRecord,
};
