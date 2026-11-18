//! How a deployment's Storage Server serves: the one settings document a server reads when it
//! starts, takes while it runs, and reports in its readiness record.
//!
//! A deployment states the settings of its Storage Servers ([`ServerSettings`], the deployment's
//! `storage_config`, with the connector its `kv_cache_config.mooncake` states), and the daemon
//! that starts the deployment's server on a node passes them as one document on the server's
//! standard input (`storage-server serve --settings -`); an operator writes the same document to
//! a file (`--settings PATH`). A running server reports the settings it serves with in its
//! readiness record ([`Readiness`]).
//!
//! A change of the settings reaches the running server one of two ways, as
//! [`Settings::restarts`] tells them apart. A setting the server reads each time it acts on it
//! (a limit, a deadline, a pace, a wait) is taken by the running server at once
//! (`Request::Configure`); a setting fixed when the server's processes start (its listener for
//! other hosts, its Transfer Server's calibration and transports, its connector) or recorded in
//! the buffer pools' trace from its image on needs a new server process, which a restart in place
//! starts with the new settings (`Request::Restart`).

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
    /// The host's transfer calibration (JSON), which `storage-server calibrate` writes: the
    /// GPUs the Transfer Server copies on, each link's bandwidth and per-copy cost, the device
    /// storage its CUDA context takes on each GPU, and the rate of copies from other hosts. The
    /// server reads it when it starts and refuses to start on a calibration it cannot read. No
    /// Transfer Server runs when unset, and transfers are refused as unavailable.
    pub transfer_calibration: Option<PathBuf>,
    /// How long one chunk of the Transfer Server's copies lasts at its link's bandwidth, in
    /// microseconds: a copy engine turns to the executing epoch's own copies, and to transfers of
    /// a higher priority, between chunks, so a chunk is kept far below one scheduler step.
    /// Default 1000.
    pub transfer_chunk_us: u64,
    /// How often a running transfer task reports its progress at most, in milliseconds: the
    /// bytes it copied, which its submitter's events and the server's listing state. Default 100.
    pub transfer_progress_ms: u64,
    /// The transports the Transfer Server's NIXL agent takes for READs between hosts, over the
    /// host's usable RDMA ports. Default `selected`.
    pub nixl_transports: Transports,
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
    /// How long the server waits before it starts a child process again (its Mooncake connector
    /// or its Transfer Server) after one ended, in seconds; the wait doubles with each process in
    /// a row that ended before it connected, up to `child_restart_max_s`, so a child that cannot
    /// start (its Mooncake master unreachable, a GPU without room for its context) is not started
    /// in a loop. Default 5.
    pub child_restart_s: u64,
    /// The longest wait before a child process starts again, in seconds. Default 300.
    pub child_restart_max_s: u64,
    /// The TCP endpoint (`host:port`) of the Storage Server of the deployment's leader node,
    /// which holds the cluster directory: this server reports its node's pages there every
    /// `directory_period_ms`. Unset on the leader's node, whose server holds the directory. The
    /// leader's daemon states it for every node from its own server's endpoint.
    pub directory: Option<service::Endpoint>,
    /// How often the server reports its node's pages to the cluster directory, in milliseconds:
    /// what the directory states of a node lags its pages by up to this period. Default 1000.
    pub directory_period_ms: u64,
    /// How long a process the server shot down, or an engine's process it follows, may take
    /// after its exit to leave every GPU whose storage it held, in seconds: past it the
    /// shoot-down fails (the GPU needs a reset) and the requests that waited for it fail.
    /// Default 60.
    pub shoot_down_s: u64,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            tcp: None,
            host_reserve_gib: 10,
            transfer_calibration: None,
            transfer_chunk_us: 1000,
            transfer_progress_ms: 100,
            nixl_transports: Transports::Selected,
            transfer_deadline_s: 60,
            put_deadline_s: 60,
            buffer_pool_trace: None,
            child_restart_s: 5,
            child_restart_max_s: 300,
            shoot_down_s: 60,
            directory: None,
            directory_period_ms: 1000,
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

    /// How often a running transfer task reports its progress at most.
    pub fn transfer_progress(&self) -> Duration {
        Duration::from_millis(self.transfer_progress_ms)
    }

    /// How long a buffer pool's transfer between worker groups may take.
    pub fn transfer_deadline(&self) -> Duration {
        Duration::from_secs(self.transfer_deadline_s)
    }

    /// How long a buffer pool's put to the remote store may take.
    pub fn put_deadline(&self) -> Duration {
        Duration::from_secs(self.put_deadline_s)
    }

    /// How long a process may take to leave every GPU it held once it exited.
    pub fn shoot_down(&self) -> Duration {
        Duration::from_secs(self.shoot_down_s)
    }

    /// How often the server reports its node's pages to the cluster directory.
    pub fn directory_period(&self) -> Duration {
        Duration::from_millis(self.directory_period_ms.max(1))
    }

    /// Whether a server serving with these settings needs a new process to serve with `next`;
    /// otherwise the running server takes `next` at once (`Request::Configure`).
    ///
    /// A new process is needed for what is fixed when the server's processes start: the
    /// listener other hosts reach it at (`tcp`), and its Transfer Server's calibration and NIXL
    /// transports, which the Transfer Server reads once when it starts. The buffer pools' trace
    /// (`buffer_pool_trace`) starts from each pool's image, which a pool states faithfully only
    /// while no task of it is in flight, as at a restart in place; and while a trace records, the
    /// deadlines of the pools' tasks (`transfer_deadline_s`, `put_deadline_s`) are part of each
    /// pool's state, which the trace holds only from an image, so they change with a new process
    /// too. Every other setting the server reads each time it acts on it: the host reserve at an
    /// admission, the deadlines at a task's start, the chunk and progress period at the Transfer
    /// Server's next chunk, the waits before a child starts again and the shoot-down's deadline
    /// when they begin; the cluster directory's leader and period at the next report.
    pub fn restarts(&self, next: &Self) -> bool {
        // Every setting is named, so a new one does not compile until it is classified here.
        let Self {
            tcp,
            host_reserve_gib: _,
            transfer_calibration,
            transfer_chunk_us: _,
            transfer_progress_ms: _,
            nixl_transports,
            transfer_deadline_s,
            put_deadline_s,
            buffer_pool_trace,
            child_restart_s: _,
            child_restart_max_s: _,
            shoot_down_s: _,
            directory: _,
            directory_period_ms: _,
        } = self;
        let traced = buffer_pool_trace.is_some();
        let deadlines = *transfer_deadline_s != next.transfer_deadline_s
            || *put_deadline_s != next.put_deadline_s;
        *tcp != next.tcp
            || *transfer_calibration != next.transfer_calibration
            || *nixl_transports != next.nixl_transports
            || *buffer_pool_trace != next.buffer_pool_trace
            || (traced && deadlines)
    }
}

/// The transports the Transfer Server's NIXL agent (UCX backend) takes between hosts. Either
/// way UCX reaches other hosts through the host's usable RDMA ports alone: those whose link is
/// active and which hold an address (a GID).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transports {
    /// The transports UCX selects by itself.
    Selected,
    /// RDMA between processes (UCX's `rc`) and CUDA copies within one (`cuda_copy`): the bytes of
    /// a READ cross the network adapter even between two processes of one host, as between two
    /// hosts.
    Rdma,
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

    /// Whether a server serving with these settings needs a new process to serve with `next`
    /// ([`ServerSettings::restarts`]): the Mooncake connector, a child process the server starts
    /// with its settings, is replaced with the server's process too.
    pub fn restarts(&self, next: &Self) -> bool {
        self.connector != next.connector || self.server.restarts(&next.server)
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
                ..ServerSettings::default()
            },
            connector: Some(connector::Settings {
                master: "10.0.0.1:50051".parse().unwrap(),
                metadata: connector::Metadata::Server("http://10.0.0.1:8080/metadata".into()),
                transport: connector::Transport::Rdma,
                device: "mlx5_0,mlx5_1".into(),
                segment_gib: 4,
                events: Some("10.0.0.1:50053".parse().unwrap()),
            }),
        };
        let document = settings.document().unwrap();
        assert_eq!(Settings::parse(&document).unwrap(), settings);
        assert_eq!(Settings::parse("").unwrap(), Settings::default());
    }

    /// The settings a server reads each time it acts on them change in the running server; the
    /// settings fixed when its processes start, and the trace with what it records from a pool's
    /// image on, need a new server process.
    #[test]
    fn a_running_server_takes_what_it_reads_when_it_acts() {
        let serving = ServerSettings::default();
        let taken = ServerSettings {
            host_reserve_gib: 4,
            transfer_chunk_us: 500,
            transfer_progress_ms: 50,
            transfer_deadline_s: 30,
            put_deadline_s: 30,
            child_restart_s: 1,
            child_restart_max_s: 10,
            shoot_down_s: 20,
            directory: Some("10.0.0.1:7100".parse().unwrap()),
            directory_period_ms: 250,
            ..serving.clone()
        };
        assert!(!serving.restarts(&taken));

        let restarted = [
            ServerSettings {
                tcp: Some("0.0.0.0:0".parse().unwrap()),
                ..serving.clone()
            },
            ServerSettings {
                transfer_calibration: Some("/etc/uniserve/calibration.json".into()),
                ..serving.clone()
            },
            ServerSettings {
                nixl_transports: Transports::Rdma,
                ..serving.clone()
            },
            ServerSettings {
                buffer_pool_trace: Some("/var/trace".into()),
                ..serving.clone()
            },
        ];
        for next in &restarted {
            assert!(serving.restarts(next), "{next:?}");
            assert!(next.restarts(&serving), "{next:?}");
        }

        // While a trace records, the pools' deadlines are part of what it records.
        let traced = ServerSettings {
            buffer_pool_trace: Some("/var/trace".into()),
            ..serving.clone()
        };
        let shorter = ServerSettings {
            transfer_deadline_s: 5,
            ..traced.clone()
        };
        assert!(traced.restarts(&shorter));
        let paced = ServerSettings {
            transfer_chunk_us: 250,
            ..traced.clone()
        };
        assert!(!traced.restarts(&paced));

        // The connector is a child the server starts with its settings.
        let connected = Settings {
            connector: Some(connector::Settings {
                master: "10.0.0.1:50051".parse().unwrap(),
                metadata: connector::Metadata::Server("http://10.0.0.1:8080/metadata".into()),
                transport: connector::Transport::Tcp,
                device: String::new(),
                segment_gib: 4,
                events: None,
            }),
            ..Settings::default()
        };
        assert!(Settings::default().restarts(&connected));
    }

    /// A key no section knows is refused.
    #[test]
    fn an_unknown_setting_is_refused() {
        assert!(Settings::parse("[server]\nhost_reserve = 4\n").is_err());
    }
}
