//! The settings of a host's Mooncake connector, which keeps the pages of the
//! buffer pools' host regions in the distributed host KV cache (L3).
//!
//! A deployment states these settings (`kv_cache_config.mooncake`), and the
//! Storage Server runs the connector (the `storage-server` crate) as its
//! child, with them as part of the server's own
//! ([`crate::settings::Settings`]). The connector opens Mooncake Store's C
//! API by its soname ([`LIBRARY`]) from the loader's default search path,
//! where the Mooncake package installs it, and names its host to other
//! Mooncake clients by the host's own name.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// The soname of Mooncake Store's C API (built with `WITH_STORE_C_SHARED`),
/// of the Mooncake release the workspace's `mooncake_store` binding pins.
pub const LIBRARY: &str = "libmooncake_store.so";

/// How the connector reaches Mooncake: the arguments of the client's
/// `setup`, each but the master with its documented default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// Mooncake's master.
    pub master: service::Endpoint,
    /// The transfer engine's metadata service; peer-to-peer handshakes by
    /// default.
    #[serde(default = "Metadata::default")]
    pub metadata: Metadata,
    /// The transfer protocol; TCP by default.
    #[serde(default = "Transport::default")]
    pub transport: Transport,
    /// The network devices for RDMA, comma-separated as Mooncake's
    /// `setup` takes them; empty, the default, selects them automatically.
    #[serde(default)]
    pub device: String,
    /// Host storage each host contributes to Mooncake's pool, in GiB; zero,
    /// the default, makes the hosts clients of segments others provide.
    #[serde(default)]
    pub segment_gib: u64,
    /// The master's KV event publisher (`host:port`, its `--kv_events_bind_endpoint`), whose
    /// stored and removed object events the cluster directory subscribes to, so that routing
    /// sees which pages Mooncake holds; none, the default, leaves the directory without
    /// Mooncake's pages. The server that holds the directory (the leader node's) subscribes
    /// when it starts.
    #[serde(default)]
    pub events: Option<service::Endpoint>,
}

impl Settings {
    /// The host storage each host contributes to Mooncake's pool, in bytes.
    pub fn segment_bytes(&self) -> u64 {
        self.segment_gib << 30
    }
}

/// Where Mooncake's transfer engines find each other's metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Metadata {
    /// Peer-to-peer handshakes between the transfer engines, with no
    /// metadata server (`P2PHANDSHAKE`, Mooncake's own default).
    #[default]
    P2pHandshake,
    /// A metadata server at an `http://`, `etcd://` or `redis://` URL.
    Server(String),
}

/// Mooncake's own name for peer-to-peer handshakes.
const P2P_HANDSHAKE: &str = "P2PHANDSHAKE";

impl fmt::Display for Metadata {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::P2pHandshake => formatter.write_str(P2P_HANDSHAKE),
            Self::Server(url) => formatter.write_str(url),
        }
    }
}

impl FromStr for Metadata {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text == P2P_HANDSHAKE {
            return Ok(Self::P2pHandshake);
        }
        match text.split_once("://") {
            Some(("http" | "etcd" | "redis", rest)) if !rest.is_empty() => {
                Ok(Self::Server(text.to_owned()))
            }
            _ => Err(format!(
                "{text:?} is no metadata service: write {P2P_HANDSHAKE} or an http://, etcd:// \
                 or redis:// URL"
            )),
        }
    }
}

impl Serialize for Metadata {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Metadata {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// The protocol Mooncake's transfer engine moves pages with.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transport {
    #[default]
    Tcp,
    Rdma,
}

impl Transport {
    /// The protocol's name in Mooncake's `setup`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Tcp => "tcp",
            Self::Rdma => "rdma",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A metadata service is peer-to-peer handshakes or a server URL of a
    /// scheme Mooncake reads; anything else is refused.
    #[test]
    fn metadata_is_handshakes_or_a_server_url() {
        assert_eq!(
            "P2PHANDSHAKE".parse::<Metadata>().unwrap(),
            Metadata::P2pHandshake
        );
        assert_eq!(
            "etcd://10.0.0.1:2379".parse::<Metadata>().unwrap(),
            Metadata::Server("etcd://10.0.0.1:2379".into())
        );
        for refused in ["p2p", "ftp://host", "http://", ""] {
            assert!(refused.parse::<Metadata>().is_err(), "{refused}");
        }
    }
}
