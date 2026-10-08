//! The launch descriptor a rank process is started with.
//!
//! The engine (or a host's `uniserve-host` launcher for a rank placed on
//! another host) writes one [`LaunchDescriptor`] as JSON next to the rank and
//! passes its path on argv. The rank's runtime deserializes it with the same
//! type, so the engine, the launcher and the worker always agree on the field
//! set: an unknown key is refused rather than ignored. The descriptor carries
//! raw launch values; the worker validates and resolves them once, before any
//! model work, and reports a violation as a usage error.
//!
//! Every tuning value is stated explicitly: the launching side is the single
//! source of defaults and the worker never re-derives one.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use uniserve_core::{CanvasSampling, ComponentConfig};

use crate::LaneConfig;

/// Descriptor key naming the inherited descriptor number of the listening
/// socket the group's first rank accepts its host-collective peers on.
pub use uniserve_core::launch::RENDEZVOUS_LISTEN_FD;

/// Every value one rank process is launched with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchDescriptor {
    /// Logical worker this rank belongs to.
    pub worker_id: String,
    /// `host:port` the rank reports its channel endpoint to.
    pub registration_address: String,
    /// Channel mechanism the rank serves its endpoint over
    /// (`SHARED_STORAGE_CHANNEL` or `SOCKET_CHANNEL`).
    pub channel_transport: String,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
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

impl LaunchDescriptor {
    /// Parses a descriptor, refusing unknown and missing keys.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Serializes the descriptor as pretty-printed JSON.
    pub fn to_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec_pretty(self)
    }
}
