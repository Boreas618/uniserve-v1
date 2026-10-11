//! The launch descriptor a rank process is started with.
//!
//! A rank is started with the head's address and its own identity, and the
//! head answers the rank's registration with one [`LaunchDescriptor`], which
//! carries every other launch value. The rank's runtime validates and
//! resolves the raw values once, before any model work, and reports a
//! violation as a usage error.
//!
//! Every tuning value is stated explicitly: the launching side is the single
//! source of defaults and the worker never re-derives one.
//!
//! Both launching sides, the head and a host's `uniserve-host` launcher, start
//! a rank with [`rank_command`] and hand it the sockets it serves with
//! [`inherit_listener`].

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::net::TcpListener;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::process::CommandExt;
use std::process::Command;

use serde::{Deserialize, Serialize};
use uniserve_core::{Canvas, CanvasSampling, ComponentConfig};

use crate::frame::Message;
use crate::{LaneConfig, ValidationResult};

/// The command that starts one rank with the interpreter `python`.
///
/// argv carries the address of the head the rank connects to and the rank's
/// identity: its worker group, its rank, and the generation of the launch
/// that starts it, which its registration states. The caller adds the
/// environment and any inherited sockets.
pub fn rank_command(
    python: impl AsRef<OsStr>,
    head: &str,
    worker_id: &str,
    rank: u32,
    generation: u64,
) -> Command {
    let mut command = Command::new(python);
    command
        .arg("-m")
        .arg("uniserve_models.worker")
        .arg("--head")
        .arg(head)
        .arg("--worker-id")
        .arg(worker_id)
        .arg("--rank")
        .arg(rank.to_string())
        .arg("--generation")
        .arg(generation.to_string());
    command
}

/// Makes the process `command` starts inherit `listener` at the descriptor
/// number returned, which the caller names in that process's launch
/// descriptor.
///
/// A worker group's first rank serves the group's collective store. The
/// process that spawns that rank binds the store's listening socket before
/// the rank exists and passes it across the rank's exec, so the port is never
/// released between the reservation and the store serving on it.
///
/// The command owns `listener` from here on and closes it when the command is
/// dropped, so the socket stays open until the spawn has duplicated it and the
/// spawning process holds no listening copy afterwards. Close-on-exec is
/// cleared only in the forked child: the descriptor keeps the flag in the
/// spawning process, so no other process it starts meanwhile inherits the
/// socket. A failure to clear the flag fails the spawn.
pub fn inherit_listener(command: &mut Command, listener: TcpListener) -> RawFd {
    let fd = listener.as_raw_fd();

    // SAFETY: the closure runs in the forked child between fork and exec, and
    // only calls `fcntl`, which is async-signal-safe. `listener` is moved into
    // the closure, so the descriptor is open in the parent at fork time.
    unsafe {
        command.pre_exec(move || {
            // FD_CLOEXEC is the only descriptor flag, so clearing every flag
            // leaves the socket open across exec at the same number.
            if libc::fcntl(listener.as_raw_fd(), libc::F_SETFD, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    fd
}

/// Every value one rank process is launched with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchDescriptor {
    /// Logical worker this rank belongs to.
    pub worker_id: String,
    /// This rank's word in every published product's acknowledgment array.
    pub acknowledgment_slot: u32,
    /// Acknowledgment slots of the ranks on this rank's host.
    pub host_slots: Vec<u32>,
    /// Whether some consumer of this rank's products runs on another host.
    pub products_cross_hosts: bool,
    /// Most requests outstanding on the rank's channel at once.
    pub queue_depth: usize,
    /// Largest frame payload in bytes, in either direction.
    pub ipc_payload_cap: usize,

    /// Checkpoint path or repository; empty for the stub model.
    pub model: String,
    /// Base checkpoint an adapter checkpoint extends.
    pub base_model: Option<String>,
    /// Torch device name of the rank's primary device (`cpu`, `cuda`, `cuda:N`).
    pub device: String,
    /// Global rank within the worker.
    pub rank: u32,
    /// Rank index among the worker's ranks on the same host.
    pub local_rank: u32,
    /// Number of ranks in the worker.
    pub world_size: u32,
    /// Components this worker places, by name.
    pub components: BTreeMap<String, ComponentConfig>,
    /// Component names every group of the deployment places.
    pub deployment_components: Vec<String>,
    /// Comma-separated capability groups to serve; `None` serves every
    /// capability the model implements.
    pub supported_calls: Option<String>,
    /// Comma-separated transfer mechanisms this rank binds.
    pub transfer_backends: String,
    /// Comma-separated transfer mechanisms this rank exports through.
    pub export_backends: String,

    /// `host:port` of the worker group's host-collective rendezvous.
    pub rendezvous_address: Option<String>,
    /// Inherited listening socket the group's first rank accepts its
    /// rendezvous peers on.
    pub rendezvous_listen_fd: Option<i32>,
    /// Membership in the deployment's expert-parallel union.
    pub expert_parallel: Option<ExpertParallelDescriptor>,
    /// `model` or `experts`.
    pub role: String,
    /// Concurrent microbatches of a disaggregated attention rank.
    pub expert_microbatches: u32,
    /// Torch process-group backend override.
    pub distributed_backend: Option<String>,
    /// Tower placement, `tower=text:<device>;gen:<device>`.
    pub mesh: Option<String>,
    /// Execution lanes and their SM partitions.
    pub lane: Vec<LaneConfig>,
    /// Load the deterministic stub model instead of a checkpoint.
    pub no_model: bool,
    /// Acknowledges that the stub model produces synthetic outputs.
    pub allow_stub: bool,

    /// Checkpoint format or loading mode.
    pub load_format: String,
    pub download_dir: Option<String>,
    pub load_threads: Option<u32>,
    pub checksum_manifest: Option<String>,
    /// Compute dtype name.
    pub model_dtype: String,
    /// Precision selectors, a JSON object.
    pub quantization_config: serde_json::Value,
    pub kv_cache_dtype: Option<String>,
    /// Fraction of device memory this rank may hold, in (0, 1].
    pub kv_memory_fraction: f64,
    pub kv_token_capacity: Option<u64>,
    /// Attention backend name.
    pub attention_backend: String,
    pub block_size: Option<u32>,
    pub max_batch_calls: u32,
    pub max_batch_tokens: u32,
    pub max_request_pool_size: u32,
    pub max_model_len: u32,
    pub max_video_seconds: f64,
    pub max_condition_rows: u32,
    /// Image features the rank's encoder cache keeps resident.
    pub encoder_cache_entries: u32,
    pub ffmpeg: String,
    pub min_video_seconds: Option<f64>,

    /// `off`, `auto` or `full`.
    pub graph_policy: String,
    /// Comma-separated strictly increasing bucket lists; `None` keeps the
    /// worker's defaults.
    pub decode_graph_batch_sizes: Option<String>,
    pub prefill_cuda_graph: bool,
    pub prefill_outputs: bool,
    pub prefill_graph_token_sizes: Option<String>,
    pub flow_cuda_graph: bool,
    pub flow_graph_batch_sizes: Option<String>,
    /// Comma-separated `HEIGHTxWIDTH` image capture sizes.
    pub flow_graph_shapes: Option<String>,
    pub video_text_capacities: Option<String>,
    /// Canvases a video deployment prepares and admits, in the order the
    /// deployment names them; empty prepares every canvas the denoiser
    /// offers.
    pub video_canvases: Vec<Canvas>,
    pub canvas_sampling: Option<CanvasSampling>,

    /// FlashInfer attention options, in bytes and backend names.
    pub flashinfer_workspace_size: u64,
    /// `true` or `false`; `None` lets the backend decide.
    pub flashinfer_use_tensor_core: Option<String>,
    pub flashinfer_decode_backend: String,
    pub flashinfer_prefill_backend: String,
    pub flashinfer_decode_split_tile_size: Option<u32>,
    pub flashinfer_prefill_split_tile_size: Option<u32>,
    pub flashinfer_disable_split_kv: bool,
}

/// One rank's place in the deployment's expert-parallel union.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertParallelDescriptor {
    /// Rank within the union.
    pub rank: u32,
    /// Ranks in the union.
    pub size: u32,
    /// Leading union ranks that run attention (disaggregated experts); zero
    /// for colocated expert parallelism.
    pub attention_ranks: u32,
    /// `host:port` of the union's rendezvous.
    pub address: Option<String>,
    /// Inherited listening socket, held only by union rank 0.
    pub listen_fd: Option<i32>,
    /// `alltoall`, `megamoe`, `dwdp` or `deepep`.
    pub exchange: String,
}

impl Message for LaunchDescriptor {
    /// The descriptor carries raw launch values; the rank that receives it
    /// resolves and checks them, so nothing is refused here.
    fn validate(&self) -> ValidationResult<()> {
        Ok(())
    }
}
