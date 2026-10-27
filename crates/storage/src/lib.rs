//! The Storage Server's protocol and its client: how every process reaches
//! the memory whose contents outlive it.
//!
//! Each host runs one Storage Server per serving user (`uniserve storage`,
//! the `storage-server` crate). It keeps model weights, the device prefix
//! cache, the buffer pool and the latent banks of every worker on the host,
//! so that a restarted worker maps the retained memory instead of loading it
//! again.
//!
//! - [`protocol`]: the messages, versioned independently of the build.
//! - [`transport`]: framed messages with file descriptors over a Unix
//!   socket.
//! - [`Client`]: one connection, used synchronously by workers and
//!   asynchronously by the head.
//! - [`socket`]: where the server listens, and starting it when absent.
//! - [`connector`]: the settings of the Mooncake connector a server runs,
//!   which keeps committed pages in the KV storage (L3).
//!
//! The server holds memory and never interprets it: clients compute every
//! key and define what the bytes mean. Bytes move on the clients' own
//! streams, or through the server's Transfer Server, which copies between
//! the memory the server holds at a client's request.

pub mod client;
pub mod connector;
pub mod protocol;
pub mod socket;
pub mod transport;

pub use client::{Answer, Client, Error, Got, Result};
pub use protocol::{
    Access, Claim, Key, Listing, Mode, Placement, Role, Scope, StorageRecord, TensorRecord,
};
