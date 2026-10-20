//! The Storage Server: memory whose contents outlive the head and the
//! workers.
//!
//! Each host runs one `uniserve-storage-server` per serving user. It keeps
//! model weights, the device prefix cache, the buffer pool and the latent
//! banks of every worker on the host, so that a restarted worker maps the
//! retained memory instead of loading it again.
//!
//! - [`protocol`]: the messages, versioned independently of the build.
//! - [`transport`]: framed messages with file descriptors over a Unix
//!   socket.
//! - [`Client`]: one connection, used synchronously by workers and
//!   asynchronously by the head.
//! - [`Server`]: the server's state and its loop.
//! - [`socket`]: where the server listens, and starting it when absent.
//!
//! The server holds memory and never interprets it: clients compute every
//! key, define what the bytes mean, and run every device copy.

pub mod client;
pub mod protocol;
pub mod server;
pub mod socket;
pub mod transport;

pub use client::{Answer, Client, Error, Got, Result};
pub use protocol::{
    Access, Claim, Key, Listing, Mode, Placement, Role, Scope, StorageRecord, TensorRecord,
};
pub use server::Server;
