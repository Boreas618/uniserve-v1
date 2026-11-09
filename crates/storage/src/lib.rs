//! The Storage Server's protocol and its client: how every process reaches
//! the storage whose contents outlive it.
//!
//! Each host runs one Storage Server per serving user
//! (`uniserve storage-server`, the `storage-server` crate), a host service
//! (the `service` crate) listening on [`default_socket`]. It keeps model weights, the device prefix
//! cache, the buffer pool and the latent banks of every worker on the host,
//! so that a restarted worker maps the retained storage instead of loading it
//! again.
//!
//! - [`protocol`]: the messages, versioned independently of the build.
//! - [`transport`]: framed messages with file descriptors over a Unix
//!   socket.
//! - [`Client`]: one connection, used synchronously by workers and
//!   asynchronously by the head.
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
pub mod key;
pub mod manifest;
pub mod protocol;
pub mod transport;

pub use client::{Answer, Client, Error, Got, Result};

/// The socket of this host's Storage Server for the calling user:
/// `storage-server.sock` in the user's service directory
/// ([`service::runtime_directory`]).
pub fn default_socket() -> std::path::PathBuf {
    service::runtime_directory().join("storage-server.sock")
}
pub use protocol::{
    Access, Claim, GpuLedger, HostStorage, Item, Key, LedgerEntry, Listing, Location, Mode, Role,
    Scope, Segment, StorageRecord, TensorRecord,
};
