//! The launch descriptor a rank process is started with.
//!
//! A rank is started with the engine's address, its own identity and the launch
//! it belongs to, and the engine answers the rank's registration with one
//! [`LaunchDescriptor`], which
//! carries every other launch value. The rank's runtime resolves the raw
//! values once, before any model work, and reports a value it cannot run
//! with as a usage error.
//!
//! Every launch value is stated explicitly: the launching side is the single
//! source of defaults and validates the values it owns, and the rank never
//! re-derives or re-checks one.

use std::collections::BTreeMap;

use crate::{CanvasSampling, ComponentConfig};
use serde::{Deserialize, Serialize};

/// The Storage Server a worker retains its memory at, and its deployment's
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
    /// on its other nodes (`host:port`), each of which may hold the memory
    /// of the deployment's other worlds: the weights a rank receives from
    /// another host. Empty for a deployment of one node.
    pub peers: Vec<String>,
    /// The world the worker belongs to: the exclusive storages it creates
    /// are its world's, and it retains what it uses in its world.
    pub world: u64,
    /// Whether the worker stands by for the worker of its scope: it maps
    /// the exclusive storages that worker holds without taking write
    /// access, warms up on scratch units, parks with the storages read-only
    /// and serves once promoted (`WorkerRequest::Promote`).
    pub standby: bool,
    /// How the worker, parked as a standby, keeps the device memory of its
    /// graph pools and scratch (`worker_config.standby_memory`).
    pub standby_memory: crate::StandbyMemory,
}

/// Every value one rank process is launched with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchDescriptor {
    /// This rank's word in every published product's acknowledgment array.
    pub acknowledgment_slot: u32,
    /// Acknowledgment slots of the ranks on this rank's host.
    pub host_slots: Vec<u32>,
    /// Whether some consumer of this rank's products runs on another host.
    pub products_cross_hosts: bool,
    /// Most requests outstanding on the rank's channel at once.
    pub queue_depth: usize,
    /// The engine's watchdog timeout in milliseconds (`worker_config.watchdog_timeout`):
    /// how long a batch may go without progress of the group's ranks before
    /// the engine replaces them. The rank reports a stalled job
    /// ([`WorkerResponse::Stalled`](crate::WorkerResponse)) once one job has
    /// run for half of it, so the report precedes the engine's expiry; at
    /// least two.
    pub watchdog_timeout_ms: u64,

    /// Checkpoint path or repository; empty for the stub model.
    pub model: String,
    /// A plugin's weight source (`scheme://...`) that streams the
    /// checkpoint's tensors in place of its files; `None` reads the files.
    #[serde(default)]
    pub weights: Option<String>,
    /// Where the worker retains its weights and exclusive storages.
    pub storage: StorageLaunch,
    /// The deployment's name of the node the worker runs on: the node's
    /// `host_identity`. Peers compare it to tell whether a product or a peer
    /// allocation is on their own node.
    pub host: String,
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
    /// Whether the worker's tensor-parallel ranks divide the routed experts
    /// among themselves, each keeping whole experts, instead of dividing
    /// every expert's channels.
    pub tensor_experts: bool,
    /// `model` or `experts`.
    pub role: String,
    /// Concurrent microbatches of a disaggregated attention rank.
    pub expert_microbatches: u32,
    /// Torch process-group backend override.
    pub distributed_backend: Option<String>,
    /// Generation-tower placement, `tower=gen:<device>`: the tower's modules
    /// on a device other than the rank's.
    pub mesh: Option<String>,
    /// Load the deterministic stub model instead of a checkpoint.
    pub no_model: bool,

    /// Checkpoint format or loading mode.
    pub load_format: String,
    pub download_dir: Option<String>,
    pub load_threads: Option<u32>,
    /// The tensor- and expert-parallel degrees retained weight memory is
    /// chunked for; empty for every degree up to the GPUs the worker sees.
    pub parallel_degrees: Vec<u32>,
    /// Compute dtype name.
    pub model_dtype: String,
    /// Precision selectors, a JSON object.
    pub quantization_config: serde_json::Value,
    pub prefix_cache_dtype: Option<String>,
    /// Fraction of device memory this rank may hold for static allocations,
    /// weights and prefix cache, in (0, 1].
    pub mem_fraction_static: f64,
    /// Prefix cache capacity in tokens; `None` sizes it from memory.
    pub max_total_tokens: Option<u64>,
    /// The KV cache of the rank of the same scope in the world this one
    /// succeeds, whose segments the rank maps read-only behind a head
    /// segment of its own; `None` for a world that succeeds none.
    pub predecessor: Option<crate::PredecessorCache>,
    /// The exact KV cache segments the rank's planes are made of: those of
    /// the worker a standby rank group's worker stands by for, or those a
    /// world's rank held before its relaunch; `None` lays the planes out
    /// afresh.
    pub segments: Option<crate::CacheTable>,
    /// The weights the rank receives from another world of its host, as
    /// the daemon planned a change of the parallel configuration; `None`
    /// for a rank that binds the weights it retained or loads them.
    pub received_weights: Option<ReceivedWeights>,
    /// Attention provider name of the rank's model, as `attention_backend`
    /// gave it. The rank's attention registry (`uniserve.backends.attention`)
    /// owns the names and refuses one it does not provide when the model
    /// loads; `auto` selects a native kernel per call.
    pub attention_backend: String,
    pub block_size: Option<u32>,
    pub max_batch_calls: u32,
    pub max_batch_tokens: u32,
    pub max_request_pool_size: u32,
    pub max_model_len: u32,
    pub max_video_seconds: f64,
    /// Condition rows one video request may occupy; `None` provisions the
    /// largest condition set the rank's denoiser admits.
    pub max_condition_rows: Option<u32>,
    /// Image features the rank's encoder cache keeps resident.
    pub encoder_cache_entries: u32,
    pub ffmpeg: String,
    pub min_video_seconds: Option<f64>,

    /// `off`, `auto` or `full`.
    pub graph_policy: String,
    /// Strictly increasing bucket lists; `None` keeps the rank's catalog.
    pub decode_graph_batch_sizes: Option<Vec<u32>>,
    pub prefill_outputs: bool,
    /// The most draft tokens a call proposes with the model's drafter; zero
    /// for a deployment that does not speculate, whose cache keeps no draft
    /// group.
    pub draft_tokens: u32,
    pub prefill_graph_token_sizes: Option<Vec<u32>>,
    /// Distinct `[height, width]` image capture sizes in pixels.
    pub flow_graph_shapes: Option<Vec<[u32; 2]>>,
    /// Strictly increasing prompt-token capacities of a video denoiser's
    /// layouts; `None` keeps the rank's default steps.
    pub video_text_capacities: Option<Vec<u32>>,
    /// Resolution classes (`768p`) a video deployment prepares, crossed
    /// with `video_aspect_ratios` resolution-major, in the order the
    /// deployment names them. The video denoiser names the canvas of each
    /// pair; empty takes every class it offers.
    pub video_resolutions: Vec<String>,
    /// Display aspect ratios (`16:9`) a video deployment prepares; empty
    /// takes every ratio the video denoiser offers.
    pub video_aspect_ratios: Vec<String>,
    pub canvas_sampling: Option<CanvasSampling>,

    /// FlashInfer attention options, in bytes and backend names.
    pub flashinfer_workspace_size: u64,
    /// `None` lets the backend decide.
    pub flashinfer_use_tensor_core: Option<bool>,
    pub flashinfer_decode_backend: String,
    pub flashinfer_prefill_backend: String,
    pub flashinfer_decode_split_tile_size: Option<u32>,
    pub flashinfer_prefill_split_tile_size: Option<u32>,
    pub flashinfer_disable_split_kv: bool,

    /// Whether the fence probes record: the device switch's initial state.
    #[serde(default)]
    pub fence_probes: bool,
    /// The components whose module instances carry fence probes, as name
    /// patterns; none when empty.
    #[serde(default)]
    pub fence_components: Vec<String>,
    /// Whether the worker enables PC sampling before it captures its CUDA
    /// graphs and keeps it enabled for its life.
    #[serde(default)]
    pub graph_pc_sampling: bool,
}

/// The weights a rank receives from the world its own succeeds, as the
/// daemon planned the change (`specs/online-operations.md`, §4.7).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceivedWeights {
    /// The world whose retained weights are the copy manifest's sources.
    pub world: u64,
    /// Serialized storage::manifest::weights::Transfer selected by the daemon.
    pub manifest: Vec<u8>,
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
