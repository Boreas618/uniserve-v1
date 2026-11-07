//! What a main process states to its host's Storage Server.

use serde::{Deserialize, Serialize};

/// The Storage Server a worker retains its storage at, and its deployment's
/// claim there.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageLaunch {
    /// The Unix socket of the worker's host's Storage Server.
    pub socket: String,
    /// The deployment the worker's scope belongs to: the served model name.
    pub deployment: String,
    /// The token of the claim the head holds at that server, which the
    /// worker states so that the server knows it belongs to the current
    /// head.
    pub claim: [u8; 16],
    /// Whether the engine copies the worker's cache pages through the host
    /// KV cache: the worker then maps a page pool per cache group, holding
    /// the tokens the engine sizes the world's host KV cache at
    /// (`uniserve.host_cache`).
    pub host_kv_cache: bool,
    /// Where an engine on another host reaches the server (`host:port`), for
    /// page requests; `None` on the engine's host, or when the server accepts
    /// no other host.
    pub tcp: Option<String>,
    /// Where processes of this host reach the deployment's Storage Servers
    /// on its other nodes (`host:port`), each of which may hold the storage
    /// of the deployment's other worlds: the weights a rank receives from
    /// another host. Empty for a deployment of one node.
    pub peers: Vec<String>,
    /// The world the worker belongs to: the exclusive storages it creates
    /// are its world's, and it retains what it uses in its world.
    pub world: u64,
    /// Whether the worker stands by for the worker of its scope: it maps
    /// the exclusive storages that worker holds without taking write
    /// access, warms up on scratch units, parks with the storages read-only
    /// and serves once promoted.
    pub standby: bool,
    /// How the worker, parked as a standby, keeps the device storage of its
    /// graph pools and scratch (`worker_config.standby_storage`).
    pub standby_storage: crate::StandbyStorage,
}
