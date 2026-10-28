//! The settings of a host's Mooncake connector, which keeps committed pages
//! in the distributed host KV cache (L3) behind the host's page pools.
//!
//! The Storage Server runs the connector (the `storage-server` crate). A
//! process that starts a Storage Server states the connector's settings on
//! the server's command line ([`Settings::arguments`]).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// How the connector reaches Mooncake: the arguments of the client's
/// `setup`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settings {
    /// The master's `host:port`.
    pub master: String,
    /// The transfer engine's metadata server: `P2PHANDSHAKE`, or an
    /// `http://`, `etcd://` or `redis://` URL.
    pub metadata: String,
    /// The transfer protocol: `tcp` or `rdma`.
    pub protocol: String,
    /// The network devices for `rdma`; empty to select automatically.
    pub device: String,
    /// The name other Mooncake clients reach this host's transfer engine
    /// by; the host's name when unset.
    pub hostname: Option<String>,
    /// Host storage this connector contributes to Mooncake's pool, in bytes;
    /// zero makes it a client only.
    pub segment_bytes: u64,
    /// `libmooncake_store.so`, built with `WITH_STORE_C_SHARED`; the
    /// loader's search path finds it when unset.
    pub library: Option<PathBuf>,
}

impl Settings {
    /// The command-line arguments of `uniserve storage serve` that state
    /// these settings.
    pub fn arguments(&self) -> Vec<String> {
        let mut arguments = vec![
            "--mooncake-master".to_owned(),
            self.master.clone(),
            "--mooncake-metadata".to_owned(),
            self.metadata.clone(),
            "--mooncake-protocol".to_owned(),
            self.protocol.clone(),
            "--mooncake-device".to_owned(),
            self.device.clone(),
            "--mooncake-segment-bytes".to_owned(),
            self.segment_bytes.to_string(),
        ];
        if let Some(hostname) = &self.hostname {
            arguments.push("--mooncake-hostname".to_owned());
            arguments.push(hostname.clone());
        }
        if let Some(library) = &self.library {
            arguments.push("--mooncake-library".to_owned());
            arguments.push(library.display().to_string());
        }
        arguments
    }
}
