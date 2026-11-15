//! How a host's Storage Server serves: the one settings document a server reads when it starts
//! and reports in its readiness record.
//!
//! A deployment states the settings of the Storage Server it uses ([`ServerSettings`], the
//! deployment's `storage_config`, with the connector its `kv_cache_config.mooncake` states), and
//! the daemon that starts the host's server passes them as one document on the server's standard
//! input (`storage-server serve --settings -`); an operator writes the same document to a file
//! (`--settings PATH`). A running server reports the settings it serves with in its readiness
//! record ([`Readiness`]), so a daemon whose deployment states other settings refuses to share
//! it instead of serving with whichever settings the first starter gave.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::connector;

/// A Storage Server's readiness record: its endpoints (`storage`, its socket; `peer`, its TCP
/// endpoint for other hosts) and its settings.
pub type Readiness = service::Readiness<Settings>;

/// Everything a Storage Server serves with.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// The server's own settings.
    #[serde(default)]
    pub server: ServerSettings,
    /// The Mooncake connector the server runs as its child, which keeps the pages of the buffer
    /// pools' host regions in Mooncake (L3); none when unset.
    #[serde(default)]
    pub connector: Option<connector::Settings>,
}

/// A Storage Server's own settings (a deployment's `storage_config`), each with its documented
/// default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerSettings {
    /// Also accept other hosts' heads and peers on this TCP endpoint (`host:port`; port 0 picks
    /// one, which the server's readiness record states as `peer`); local connections only when
    /// unset.
    pub tcp: Option<service::Endpoint>,
    /// Host storage, in GiB, that new host storage (a host region, the host copy of an epoch's
    /// weights) leaves available to the host's other processes: storage the host's available
    /// storage less this reserve cannot hold is refused (SGLang's host KV cache keeps 10 GiB the
    /// same way). Default 10.
    pub host_reserve_gib: u64,
    /// How long one chunk of the Transfer Server's copies lasts at its link's bandwidth, in
    /// microseconds: a copy engine turns to the executing epoch's own copies, and to transfers of
    /// a higher priority, between chunks, so a chunk is kept far below one scheduler step.
    /// Default 1000.
    pub transfer_chunk_us: u64,
    /// Whether the Transfer Server copies on this host's GPUs: it then requires the CUDA driver
    /// and copies on every GPU the driver shows; otherwise it copies host storage alone and loads
    /// no driver. Default true.
    pub transfer_gpus: bool,
    /// How long a buffer pool's transfer of a request between worker groups may take, in
    /// seconds, before it is stopped and answered as failed. Default 60.
    pub transfer_deadline_s: u64,
    /// How long a buffer pool's put of host copies to the remote store may take, in seconds,
    /// before it is stopped and answered with the keys put so far. Default 60.
    pub put_deadline_s: u64,
    /// Record the trace of every buffer pool's inputs in this directory: per deployment, every
    /// input its pool acts on and every completion it produces (`specs/buffer-pool.md` §18). A
    /// failure to write stops the trace, not the pool. No trace when unset.
    pub buffer_pool_trace: Option<PathBuf>,
    /// Check every buffer pool's tables after an input it acts on, whenever its thread has run at
    /// least nine times the previous check's duration since that check ended, so that checking
    /// takes at most a tenth of the thread that answers the steps: an inconsistency is logged when
    /// the pool turns inconsistent and counted in its listing (`check_failures`). Default off.
    pub buffer_pool_check: bool,
    /// How long the server waits before it starts a child process again (its Mooncake connector
    /// or its Transfer Server) after one ended, in seconds; the wait doubles with each process in
    /// a row that ended before it connected, up to `child_restart_max_s`, so a child that cannot
    /// start (its Mooncake master unreachable, a GPU without room for its context) is not started
    /// in a loop. Default 5.
    pub child_restart_s: u64,
    /// The longest wait before a child process starts again, in seconds. Default 300.
    pub child_restart_max_s: u64,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            tcp: None,
            host_reserve_gib: 10,
            transfer_chunk_us: 1000,
            transfer_gpus: true,
            transfer_deadline_s: 60,
            put_deadline_s: 60,
            buffer_pool_trace: None,
            buffer_pool_check: false,
            child_restart_s: 5,
            child_restart_max_s: 300,
        }
    }
}

impl ServerSettings {
    /// Host storage new host storage leaves available to the host's other processes, in bytes.
    pub fn host_reserve_bytes(&self) -> u64 {
        self.host_reserve_gib << 30
    }

    /// How long one chunk of the Transfer Server's copies lasts.
    pub fn transfer_chunk(&self) -> Duration {
        Duration::from_micros(self.transfer_chunk_us)
    }

    /// How long a buffer pool's transfer between worker groups may take.
    pub fn transfer_deadline(&self) -> Duration {
        Duration::from_secs(self.transfer_deadline_s)
    }

    /// How long a buffer pool's put to the remote store may take.
    pub fn put_deadline(&self) -> Duration {
        Duration::from_secs(self.put_deadline_s)
    }
}

impl Settings {
    /// The settings a document states: TOML, as an operator writes it and a daemon passes it.
    ///
    /// # Errors
    ///
    /// Fails with the parser's message when the document is no settings document: a key no
    /// section knows is refused rather than ignored.
    pub fn parse(document: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(document)
    }

    /// The settings as the TOML document [`Settings::parse`] reads.
    ///
    /// # Errors
    ///
    /// Fails when a value has no TOML form.
    pub fn document(&self) -> Result<String, toml::ser::Error> {
        toml::to_string(self)
    }
}

/// A Storage Server serves with other settings than its starter states: the server keeps its
/// settings, and the starter is refused rather than served with settings it did not state.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "the Storage Server {pid} serves with other settings ({running:?}) than stated ({stated:?}); \
     stop it (`uniserve storage-server stop`) or state its settings"
)]
pub struct Mismatch {
    /// The serving server's process.
    pub pid: u32,
    /// The settings it serves with.
    pub running: Box<Settings>,
    /// The settings its starter stated.
    pub stated: Box<Settings>,
}

impl Mismatch {
    /// `Err` naming both when `record` states other settings than `stated`.
    ///
    /// # Errors
    ///
    /// The mismatch, when the settings differ.
    pub fn check(record: &Readiness, stated: &Settings) -> Result<(), Self> {
        if record.settings == *stated {
            return Ok(());
        }
        Err(Self {
            pid: record.pid,
            running: Box::new(record.settings.clone()),
            stated: Box::new(stated.clone()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Settings read back from the document they write, and a document that states nothing is
    /// the defaults.
    #[test]
    fn settings_read_back_from_their_document() {
        let settings = Settings {
            server: ServerSettings {
                tcp: Some("10.0.0.2:0".parse().unwrap()),
                host_reserve_gib: 4,
                buffer_pool_check: true,
                ..ServerSettings::default()
            },
            connector: Some(connector::Settings {
                master: "10.0.0.1:50051".parse().unwrap(),
                metadata: connector::Metadata::Server("http://10.0.0.1:8080/metadata".into()),
                transport: connector::Transport::Rdma,
                device: "mlx5_0,mlx5_1".into(),
                segment_gib: 4,
            }),
        };
        let document = settings.document().unwrap();
        assert_eq!(Settings::parse(&document).unwrap(), settings);
        assert_eq!(Settings::parse("").unwrap(), Settings::default());
    }

    /// A key no section knows is refused.
    #[test]
    fn an_unknown_setting_is_refused() {
        assert!(Settings::parse("[server]\nhost_reserve = 4\n").is_err());
    }
}
