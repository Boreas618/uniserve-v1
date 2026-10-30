//! WorkerGroup process configuration, IPC execution, rank aggregation, and recovery.
//!
//! The engine drives GPU and host work through worker processes. A
//! `WorkerGroup` (in `group`) is one worker group's cooperating workers: it projects
//! each submitted `Batch` onto the ranks whose components own its calls, joins
//! their agreeing results into `WorkerResult`s, and replaces the whole group
//! when a rank is lost. `WorkerExecutor` (in `executor`) submits explicitly
//! targeted work to the groups of a deployment. `process` owns one rank's
//! channel, `registry` the listener ranks connect to and register at, and
//! `processes` the engine's connection to the deployment's daemons, which
//! start the ranks on every node, hold the sockets their stores serve on and
//! the deployment's Storage Server claims (`claims`).

pub(crate) mod claims;
mod executor;
mod group;
mod incarnations;
mod process;
pub(crate) mod processes;
pub(crate) mod registry;

pub use executor::WorkerExecutor;
pub use group::{CallReaders, WorkerGroup};
pub(crate) use incarnations::IncarnationRegistry;
use process::{PendingWorker, WorkerProcess};
use proto::request::{ExpertExchange, FlashInferBackend, GraphPolicy, LoadFormat};

/// Backpressure and terminal failures returned by a WorkerGroup batch submission.
#[derive(Debug, thiserror::Error)]
pub enum BatchSubmitError {
    /// Returns the unaccepted physical submission when the group has no capacity.
    ///
    /// The batch is handed back unchanged so the caller can resubmit it
    /// later; nothing was recorded or sent for it.
    #[error("worker batch queue is full")]
    WouldBlock(Box<ipc::Batch>),
    /// Reports a terminal transport or execution failure.
    #[error(transparent)]
    Failed(anyhow::Error),
}

/// Everything required to create or recreate one worker process.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct WorkerProcessArgs {
    /// Worker group identity assigned by the engine.
    pub worker_id: String,
    /// The `uniserve` executable each worker of the group runs as (its
    /// `worker` command), and that starts a host's Storage Server when the
    /// host has none (its `storage serve` command).
    pub executable: std::path::PathBuf,
    /// The deployment the group belongs to, which its workers' storage
    /// scopes name: the served model name.
    pub deployment: String,
    /// Whether the engine copies the group's cache pages through the
    /// host KV cache (`kv_cache_config.host_tier`).
    pub host_kv_cache: bool,
    /// The Mooncake connector a host's Storage Server the deployment starts
    /// runs (`kv_cache_config.mooncake`), keeping committed pages in the
    /// KV storage.
    pub kv_storage: Option<storage::connector::Settings>,
    /// Whether the group keeps a standby rank group beside its serving one
    /// (`worker_config.standby`), promoted when the serving group fails.
    pub keep_standby: bool,
    /// Whether this launch's workers stand by for the group's serving
    /// workers: they map the exclusive storages those hold without taking
    /// write access, warm up on scratch units, and park until promoted.
    pub standby: bool,
    /// How a standby launch keeps its graph storage while it parks
    /// (`worker_config.standby_storage`).
    pub standby_storage: proto::request::StandbyStorage,
    /// The exact KV cache segments the group's planes are made of, and the
    /// leading ones its ranks hold: the serving workers' for a standby
    /// launch, the lost ranks' for a relaunch, the committed world's for an
    /// engine that replaces a lost one (`EngineConfig::segments`); `None`
    /// lays them out afresh, as a world's first launch does.
    pub segments: Option<ipc::CacheTable>,
    /// The world the group's workers belong to, which keys the storage they
    /// create and retains what they use.
    pub world: u64,
    /// Worlds of the deployment whose storage ends on every node before the
    /// group's workers start: a world on trial a rollback discards, whose
    /// storage the previous world, started again, needs.
    pub retire: Vec<u64>,
    /// The KV cache of the predecessor world's group of the same identity,
    /// whose segments the group's workers map read-only; `None` for a world
    /// that succeeds none.
    pub predecessor: Option<ipc::PredecessorCache>,
    /// Map writable warmup storage and reserve the remaining virtual cache
    /// pool for a staged transfer target.
    pub stage_cache: bool,
    /// Existing Host cache keys this world's restart keeps for rollback.
    pub cache_keys: Vec<[u8; 32]>,
    /// The weights the group's workers receive from the world the engine
    /// succeeds, as the daemon planned a change of the parallel
    /// configuration; `None` binds or loads them.
    pub received_weights: Option<ipc::ReceivedWeights>,
    /// Model identifier or local model path.
    pub model: String,
    /// A plugin's weight source (`scheme://...`) that streams the
    /// checkpoint's tensors in place of its files; `None` reads the files.
    pub weights: Option<String>,
    /// Ordered physical members of this worker group.
    pub ranks: Vec<crate::WorkerRank>,
    /// This process's own host identity. The engine owns exactly the ranks
    /// whose placement host matches it, so a placement may name its host
    /// explicitly instead of relying on a reserved local name.
    pub host: String,
    /// Whether this group owns model capabilities or shared routed experts.
    pub role: crate::WorkerGroupRole,
    /// Component membership and parallel geometry.
    pub components: std::collections::BTreeMap<String, crate::ComponentConfig>,
    /// Every worker's component membership, keyed by worker identity, this
    /// group included. A rank is told which ranks read its products, and the
    /// consuming component may belong to another worker.
    pub peers: std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<String, crate::ComponentConfig>,
    >,
    /// The leader daemon's process socket, through which the engine starts
    /// the group's workers on their nodes (`unix:<path>`).
    pub daemon: String,
    /// How long the group waits for progress of its ranks (a joined result)
    /// while a batch is pending before it treats them as hung and replaces
    /// them, failing the requests they hold (`worker_config.watchdog_timeout`). A rank
    /// that dies is detected at once; this bounds a rank that is alive but
    /// stuck, for example in a collective. The launch descriptor carries it
    /// to every rank, which reports a job that has run for half of it
    /// (`WorkerResponse::Stalled`), so a rank that stays inside a call is
    /// reported, and named in the failure of its requests, before this
    /// expires, even for a forward that started up to half of it after the
    /// group's last progress.
    pub watchdog_timeout: std::time::Duration,
    /// Launch the worker without model weights, using its deterministic test
    /// model. Only the engine's own tests and examples set this; the serving
    /// command line cannot request it.
    pub stub: bool,
    /// Maximum number of physical batches concurrently in flight per rank.
    pub queue_depth: usize,
    /// Optional prefix cache capacity in tokens (`kv_cache_config.max_tokens`).
    pub max_total_tokens: Option<u64>,
    /// Tokens per cache page of the cache group with the widest token rows;
    /// the worker derives every other group's block size from it. `None`
    /// lets the worker choose it from its cache layers and attention
    /// kernels.
    pub block_size: Option<u32>,
    /// Maximum calls accepted in one worker batch.
    pub max_batch_calls: u32,
    /// Maximum tokens accepted in one worker batch.
    pub max_batch_tokens: u32,
    /// Request rows each rank holds: the size of the worker's request pool,
    /// whose rows carry per-request state such as canvases, sampling state
    /// and block tables. A worker that fits its request tensors to device
    /// storage takes this as the largest pool it may choose.
    pub max_request_pool_size: u32,
    /// Attention provider name, forwarded to the ranks verbatim. The ranks'
    /// attention registry owns the names and refuses one it does not
    /// provide when the model loads; `auto` lets it select per call.
    pub attention_backend: String,
    /// Product transport exposed by the worker pool.
    pub transfer: crate::executor::TransferConfig,
    /// Checkpoint loader format.
    pub load_format: LoadFormat,
    /// Optional cache directory for downloaded model artifacts.
    pub download_dir: Option<std::path::PathBuf>,
    /// Optional number of concurrent checkpoint reader threads.
    pub load_threads: Option<u32>,
    /// The tensor- and expert-parallel degrees retained weight storage is
    /// chunked for; empty for every degree up to the GPUs the worker sees.
    pub parallel_degrees: Vec<u32>,
    /// Numeric data type used by model parameters and activations.
    pub dtype: crate::ModelDtype,
    /// WorkerGroup-specific quantization policy.
    pub quantization_config: serde_json::Value,
    /// Numeric data type of the KV (prefix) cache, when explicitly selected.
    pub kv_cache_dtype: Option<ipc::PrefixCacheDtype>,
    /// Each rank's share of its device's total storage, which bounds the
    /// process's static allocations, weights and KV cache;
    /// `WorkerGroupConfig::validate_mem_fraction_static` states its range.
    pub mem_fraction_static: f64,
    /// Generation-tower placement, `tower=gen:<device>`: the tower's modules
    /// on a device other than the rank's; the rank parses it.
    pub worker_mesh: Option<String>,
    /// Optional process-world communication backend.
    pub distributed_backend: Option<String>,
    /// Module graph coverage policy.
    pub graph_policy: GraphPolicy,
    /// Strictly increasing decode batch sizes captured as CUDA graphs;
    /// `None` keeps the rank's catalog.
    pub decode_graph_batch_sizes: Option<Vec<u32>>,
    /// Whether any prefill call of the deployment samples a token or scores
    /// its prompt. Without it every prefill only writes the prefix cache, and
    /// the worker's prefill graphs stop at the final layer's cache write.
    pub prefill_outputs: bool,
    /// The most draft tokens a call proposes with the model's drafter; zero
    /// when the deployment does not speculate.
    pub draft_tokens: u32,
    /// Strictly increasing prefill token counts captured as CUDA graphs;
    /// `None` keeps the rank's catalog.
    pub prefill_graph_token_sizes: Option<Vec<u32>>,
    /// Distinct image capture sizes, `[height, width]` in pixels; `None`
    /// keeps the rank's catalog.
    pub flow_graph_shapes: Option<Vec<[u32; 2]>>,
    /// Strictly increasing text capacities, in prompt tokens, of a video
    /// denoiser's layouts below the context length; `None` keeps the rank's
    /// default steps.
    pub video_text_capacities: Option<Vec<u32>>,
    /// Resolution classes (`768p`) a video deployment prepares, crossed with
    /// `video_aspect_ratios` resolution-major in the order the deployment
    /// names them. Its workers' video denoiser names the canvas of each pair;
    /// the workers prepare exactly those and report them back, with their
    /// names, as the only canvases requests may target. Empty takes every
    /// class the denoiser offers.
    pub video_resolutions: Vec<String>,
    /// Display aspect ratios (`16:9`) a video deployment prepares; empty
    /// takes every ratio the denoiser offers.
    pub video_aspect_ratios: Vec<String>,
    /// Block-diffusion sampling of every canvas the deployment generates, as
    /// the server resolves it; `None` when the served model generates no
    /// canvases. The worker keeps the argmax history its stability
    /// threshold needs, prepares its canvas steps for exactly this sampling,
    /// and refuses a request that carries another.
    pub canvas_sampling: Option<ipc::CanvasSampling>,
    /// FlashInfer workspace capacity in bytes.
    pub flashinfer_workspace_size: u64,
    /// FlashInfer tensor-core decode selection; `None` lets the backend
    /// decide.
    pub flashinfer_use_tensor_core: Option<bool>,
    /// FlashInfer backend used for decode attention.
    pub flashinfer_decode_backend: FlashInferBackend,
    /// FlashInfer backend used for prefill attention.
    pub flashinfer_prefill_backend: FlashInferBackend,
    /// Optional split-KV tile size used for FlashInfer decode attention.
    pub flashinfer_decode_split_tile_size: Option<u32>,
    /// Optional split-KV tile size used for FlashInfer prefill attention.
    pub flashinfer_prefill_split_tile_size: Option<u32>,
    /// Whether FlashInfer may split the KV sequence across thread blocks.
    pub flashinfer_split_kv: bool,
    /// Whether the fence probes record (`profiling_config.fence_probes`).
    pub fence_probes: bool,
    /// The components whose module instances carry fence probes, as name
    /// patterns (`profiling_config.fence_components`).
    pub fence_components: Vec<String>,
    /// The probed components whose probes are off, as name patterns
    /// (`profiling_config.fence_probes_off`).
    pub fence_probes_off: Vec<String>,
    /// Whether the workers keep PC sampling enabled for their life
    /// (`profiling_config.graph_pc_sampling`).
    pub graph_pc_sampling: bool,
    /// Maximum model context length in tokens.
    pub max_model_len: u32,
    /// Maximum accepted video duration in seconds. The frontend states it
    /// from the serving options; the default is the longest the video API
    /// admits.
    pub max_video_seconds: f64,
    /// Most denoiser rows a video request's conditions may take, which bounds
    /// the condition products a video worker provisions; `None` provisions
    /// the largest condition set the worker's denoiser admits.
    pub max_condition_rows: Option<u32>,
    /// Image features each rank's encoder cache keeps resident, which bounds
    /// the context images one request may pin.
    pub encoder_cache_entries: u32,
    /// The `ffmpeg` executable a media reader decodes reference videos with.
    pub ffmpeg: std::path::PathBuf,
    /// Shortest video duration in seconds the serving API admits, which
    /// bounds the frame counts a video worker provisions from below; `None`
    /// provisions every frame count the model generates.
    pub min_video_seconds: Option<f64>,
    /// This group's place in the expert-parallel world its data-parallel
    /// replica joins, when the deployment shards routed experts across the
    /// replicas; `None` keeps every expert on the group's own ranks.
    pub expert_parallel: Option<ExpertParallelPlacement>,
    /// Whether the group's tensor-parallel ranks divide the routed experts
    /// among themselves, each keeping whole experts and the group summing
    /// their outputs, instead of dividing every expert's channels.
    pub tensor_experts: bool,
    /// Logical expert IDs in global physical-slot order, retained on recovery.
    pub expert_order: std::collections::BTreeMap<String, Vec<u32>>,
    /// Independent numerical microbatches overlapping attention and remote experts.
    pub expert_microbatches: u32,
}

/// A worker group's consecutive ranks in the deployment's expert union.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ExpertParallelPlacement {
    /// The group's first physical rank in the expert union.
    pub rank: u32,
    /// Number of physical ranks sharing the experts.
    pub size: u32,
    /// Leading source-only ranks; zero for colocated expert parallelism.
    pub attention_ranks: u32,
    /// TCP address of the world's rendezvous store. `WorkerGroup::spawn_all`
    /// reserves it once the deployment's hosts are known; world rank 0 serves
    /// it on a socket the engine binds.
    pub address: Option<String>,
    /// How the world exchanges tokens at every expert layer.
    pub exchange: ExpertExchange,
}

/// Parks until one descriptor becomes readable or `timeout` expires.
///
/// Returning `Ok` does not say which happened or which descriptor fired;
/// callers re-poll their channels afterwards. An empty `fds` is a bounded
/// sleep: nothing in the engine unparks the waiting thread. Wake events are
/// the ones each `pollfd` requests, plus the error and hangup conditions poll
/// always reports. Fails on a poll error other than an interrupt, and on
/// every call with descriptors on targets other than Linux.
pub(crate) fn park_descriptors(
    fds: &[libc::pollfd],
    timeout: std::time::Duration,
) -> anyhow::Result<()> {
    if fds.is_empty() {
        std::thread::park_timeout(timeout);
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    {
        let mut pollfds = fds.to_vec();
        // The millisecond timeout rounds a sub-millisecond remainder up, so a
        // nonzero wait never becomes a zero-timeout poll that spins, and is
        // capped at the largest value poll accepts.
        let timeout_ms = timeout
            .as_millis()
            .saturating_add(u128::from(
                !timeout.subsec_nanos().is_multiple_of(1_000_000),
            ))
            .min(i32::MAX as u128) as i32;
        loop {
            // SAFETY: `pollfds` is a live contiguous allocation for the stated
            // element count and poll does not retain the pointer.
            let result = unsafe {
                libc::poll(
                    pollfds.as_mut_ptr(),
                    pollfds.len() as libc::nfds_t,
                    timeout_ms,
                )
            };
            if result >= 0 {
                return Ok(());
            }
            // An interrupted poll retries with the full timeout rather than
            // the remainder, so a signal can extend the wait.
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                return Err(error.into());
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = timeout;
        anyhow::bail!("worker progress descriptors require Linux poll support")
    }
}
