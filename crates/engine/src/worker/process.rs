//! One rank's process and channel, from spawn to shutdown.
//!
//! `PendingRank::spawn_rank` derives the rank's launch descriptor and starts
//! the rank process here, or hands the launch to the host's launcher
//! (`launcher::RemoteHost`) when the placement puts the rank on another host.
//! The rank connects to the head and registers (`registry::RankRegistry`);
//! `PendingRank::adopt` answers on that connection with the descriptor and
//! yields a `RankProcess`, which `WorkerGroup` (in `instance`) drives:
//! `finish_startup` validates the ranks' startup reports, `submit_batch` and
//! `poll_batch` exchange batches and results, and `close_ranks` or
//! `terminate` retires the rank.
//!
//! A rank's connection is its liveness: the head treats a rank whose
//! connection has closed as dead. A rank whose lane stays inside one job
//! reports it unasked (`WorkerResponse::Stalled`), and the head fails that
//! rank as it fails a dead one. A rank this head started is its child, a
//! `RankChild`, so whichever path reaps it (an observed exit, a kill at a
//! deadline, a termination) first removes the CUDA driver memory a rank that
//! did not end in order left behind.
//!
//! The host exchanges batch descriptors and bounded result values while
//! tensors, cache pages, and latent storage remain worker-resident.
//!
//! The module also defines the launch settings `WorkerProcessArgs` carries
//! (`LaneConfig`, `FlashInferBackend`), that type's defaults, and the launch
//! descriptor built from it.

use crate::executor::WorkerResult;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::BatchSubmitError;
use crate::executor::WorkerExecError;
use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uniserve_ipc::Channel;
use uniserve_ipc::launch::RankChild;
use uniserve_ipc::{
    Batch, DEFAULT_COMPONENT, ExpertParallelDescriptor, LaunchDescriptor, WorkerInfo,
    WorkerRequest, WorkerResponse,
};

use crate::worker::WorkerProcessArgs;

/// Maximum time `request_close` spends draining in-flight responses from a
/// still-alive worker before it sends the shutdown request. A hung (alive but
/// unresponsive) worker must not be able to block shutdown forever.
const WORKER_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);
/// How long the ranks closed together by `close_ranks` have, from the last
/// shutdown request, to acknowledge it, finish their teardown and exit. A
/// local process still running afterwards is killed, which abandons whatever
/// its teardown had not yet released.
const WORKER_EXIT_GRACE: Duration = Duration::from_secs(30);
/// Longest wait between checks for the exit of a rank `close_ranks` is
/// closing. A local process exit raises no wake, and a launcher reports a
/// remote exit within its own 100-millisecond read timeout.
const EXIT_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// Interval between progress logs while a rank prepares during startup.
const STARTUP_LOG_INTERVAL: Duration = Duration::from_secs(30);
/// Longest wait between liveness checks (`RankProcess::check_worker`) while
/// waiting on a rank, which bounds how late a local rank's exit status is
/// read once its connection has closed.
const WORKER_CHECK_INTERVAL: Duration = Duration::from_millis(500);

pub use uniserve_ipc::LaneConfig;

/// FlashInfer implementation selected in [`WorkerProcessArgs`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlashInferBackend {
    /// Lets the worker select a compatible backend.
    #[default]
    Auto,
    /// Uses FlashAttention 2 kernels.
    Fa2,
    /// Uses FlashAttention 3 kernels.
    Fa3,
}

impl FlashInferBackend {
    /// Returns the stable command-line spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Fa2 => "fa2",
            Self::Fa3 => "fa3",
        }
    }
}

impl std::str::FromStr for FlashInferBackend {
    type Err = FlashInferBackendParseError;

    /// Parses the spelling [`FlashInferBackend::as_str`] produces.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "fa2" => Ok(Self::Fa2),
            "fa3" => Ok(Self::Fa3),
            _ => Err(FlashInferBackendParseError(value.to_owned())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported FlashInfer backend {0:?}")]
/// Error returned for an unsupported FlashInfer backend name.
pub struct FlashInferBackendParseError(String);

impl WorkerProcessArgs {
    /// The interpreter that runs rank processes when none is configured: the
    /// one installed next to the running binary (an environment's `bin/`),
    /// then the active virtual environment's, then `python3` on `PATH`.
    pub fn default_python() -> std::path::PathBuf {
        if let Ok(executable) = std::env::current_exe()
            && let Some(directory) = executable.parent()
        {
            for name in ["python3", "python"] {
                let candidate = directory.join(name);
                if candidate.is_file() {
                    return candidate;
                }
            }
        }

        if let Some(environment) = std::env::var_os("VIRTUAL_ENV") {
            let candidate = std::path::Path::new(&environment)
                .join("bin")
                .join("python");
            if candidate.is_file() {
                return candidate;
            }
        }

        "python3".into()
    }

    /// Parses a quantization policy as a command line states it: a JSON
    /// object such as `{"mode":"balanced"}`, where an empty object selects
    /// the model's default policy.
    ///
    /// The signature suits a command-line value parser. Fails on text that is
    /// not JSON, and on JSON that is not an object, such as an array or a
    /// scalar.
    pub fn parse_quantization_config(text: &str) -> anyhow::Result<Value> {
        let value: Value = serde_json::from_str(text)
            .map_err(|error| anyhow::anyhow!("invalid JSON object: {error}"))?;
        anyhow::ensure!(value.is_object(), "expected a JSON object");
        Ok(value)
    }
}

impl Default for WorkerProcessArgs {
    /// Returns worker launch settings suitable for a single local rank.
    fn default() -> Self {
        Self {
            worker_id: "worker".into(),
            python: Self::default_python(),
            launcher_timeout: std::time::Duration::from_secs(120),
            result_deadline: std::time::Duration::from_secs(300),
            model: String::new(),
            base_model: None,
            ranks: vec![crate::WorkerRank {
                node: "localhost".into(),
                device: "cuda:0".into(),
            }],
            host: "localhost".into(),
            // One local rank running one component. A launch replaces this
            // with the components the model being served declared.
            components: crate::WorkerConfig::single_component(DEFAULT_COMPONENT, 1),
            role: crate::WorkerRole::Model,
            peers: Default::default(),
            stub: false,
            queue_depth: 2,
            max_message_bytes: 8 << 20,
            cache_token_capacity: None,
            block_size: None,
            max_batch_calls: 128,
            max_batch_tokens: 16_384,
            // The engine's default running-request limit plus the largest
            // row reserve a runtime keeps outside it.
            max_request_pool_size: (crate::scheduler::DEFAULT_MAX_NUM_SEQS
                + crate::scheduler::MAX_FLOW_PREFIX_ROWS) as u32,
            attention_backend: uniserve_ipc::AttentionBackend::Auto,
            capability_groups: Vec::new(),
            transfer: Default::default(),
            load_format: "auto".to_string(),
            download_dir: None,
            load_threads: None,
            checksum_manifest: None,
            model_dtype: uniserve_core::ModelDtype::BFloat16,
            quantization_config: serde_json::json!({}),
            prefix_cache_dtype: None,
            cache_storage_fraction: 0.70,
            mesh: None,
            distributed_backend: None,
            lanes: Vec::new(),
            graph_policy: "auto".into(),
            decode_graph_batch_sizes: None,
            prefill_cuda_graph: true,
            prefill_outputs: true,
            prefill_graph_token_sizes: None,
            flow_cuda_graph: true,
            flow_graph_batch_sizes: None,
            flow_graph_shapes: None,
            video_text_capacities: None,
            video_canvases: Vec::new(),
            canvas_sampling: None,
            flashinfer_workspace_size: 512 * 1024 * 1024,
            flashinfer_use_tensor_core: None,
            flashinfer_decode_backend: FlashInferBackend::Fa2,
            flashinfer_prefill_backend: FlashInferBackend::Auto,
            flashinfer_decode_split_tile_size: None,
            flashinfer_prefill_split_tile_size: None,
            flashinfer_disable_split_kv: false,
            max_model_len: 8192,
            max_video_seconds: 15.0,
            max_condition_rows: 0,
            encoder_cache_entries: 256,
            ffmpeg: "ffmpeg".into(),
            min_video_seconds: None,
            expert_parallel: None,
            expert_microbatches: 1,
        }
    }
}

impl WorkerProcessArgs {
    /// Returns one rank's index among the ranks on its own host and that
    /// host's rank count.
    ///
    /// These are the host-relative `LOCAL_RANK` and `LOCAL_WORLD_SIZE`
    /// values: a rank's position among the ranks the placement puts on the
    /// same node, and how many ranks that node holds. They coincide with the
    /// global values while an instance occupies one host. `rank` must index
    /// `self.ranks`.
    fn host_slot(&self, rank: u32) -> anyhow::Result<(u32, u32)> {
        let rank = rank as usize;
        let node = &self.ranks[rank].node;
        let resident = |placed: &&crate::WorkerRank| &placed.node == node;

        // The rank's host index counts the ranks placed before it on its node.
        let index = self.ranks[..rank].iter().filter(resident).count();
        let count = self.ranks.iter().filter(resident).count();
        Ok((
            u32::try_from(index).context("a host's rank index exceeds u32")?,
            u32::try_from(count).context("a host's rank count exceeds u32")?,
        ))
    }

    /// Returns the number of ranks in this group.
    fn world_size(&self) -> u32 {
        self.ranks.len() as u32
    }

    /// Builds the typed launch descriptor the worker process consumes.
    ///
    /// Every tuning value is stated explicitly, so the launching side is the
    /// single source of defaults and the worker never re-derives one. The
    /// head answers the rank's registration with it; only the head's address
    /// and the rank's identity travel on argv.
    ///
    /// `rank` must index `self.ranks`. Fails when a host-relative rank value
    /// does not fit `u32` or the role does not serialize.
    pub(crate) fn launch_descriptor(
        &self,
        rank: u32,
        rendezvous: Option<&str>,
        rendezvous_listen_fd: Option<std::os::fd::RawFd>,
        expert_listen_fd: Option<std::os::fd::RawFd>,
    ) -> anyhow::Result<LaunchDescriptor> {
        let depth = self.queue_depth.max(1);
        let max_payload = self.max_message_bytes.max(1);
        // Resolve mechanism ownership from the physical rank's incident edges.
        let (transfer_backends, export_backends) =
            self.transfer.rank_backends(&self.worker_id, rank);
        let names = |backends: &std::collections::BTreeSet<crate::executor::TransferBackend>| {
            backends
                .iter()
                .map(|backend| backend.as_str())
                .collect::<Vec<_>>()
                .join(",")
        };
        let path = |path: &std::path::Path| path.to_string_lossy().into_owned();

        // Every component any group of the deployment places, so each rank
        // resolves the capabilities the deployment as a whole selects, such
        // as which of a checkpoint's denoisers it serves.
        let deployment = self
            .peers
            .values()
            .flat_map(|components| components.keys())
            .chain(self.components.keys())
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();

        Ok(LaunchDescriptor {
            worker_id: self.worker_id.clone(),
            // A rank cannot name the ranks that read what it publishes: it
            // knows its own component, not which component consumes its
            // products. The head states a product's readers on the call that
            // produces it and gives the rank its own acknowledgment slot here.
            acknowledgment_slot: self.transfer.acknowledgment_slot(&self.worker_id, rank),
            // Which of those slots are on this rank's host decides the
            // mechanism a host product is published over.
            host_slots: self.transfer.host_slots(&self.worker_id, rank),
            // Readiness is a producer synchronize only where an interprocess
            // event cannot carry it, which is where a consumer is on another
            // host. Only the placement knows that.
            products_cross_hosts: self.transfer.products_cross_hosts(&self.worker_id, rank),
            queue_depth: depth,
            ipc_payload_cap: max_payload,
            // Half the group's deadline: see `WorkerProcessArgs::result_deadline`.
            stall_bound_ms: u64::try_from((self.result_deadline / 2).as_millis())
                .unwrap_or(u64::MAX),
            model: self.model.clone(),
            base_model: self.base_model.as_deref().map(path),
            device: self.ranks[rank as usize].device.clone(),
            rank,
            local_rank: self.host_slot(rank)?.0,
            world_size: self.world_size(),
            components: self.components.clone(),
            deployment_components: deployment,
            // `None` lets the worker serve every capability group it
            // implements.
            supported_calls: (!self.capability_groups.is_empty())
                .then(|| self.capability_groups.join(",")),
            transfer_backends: names(&transfer_backends),
            export_backends: names(&export_backends),
            // Every rank of a group meets its peers at this address. The
            // group's first rank accepts them on the socket it inherits at
            // the named descriptor rather than binding the port itself; a
            // first rank placed elsewhere has the number filled in by the
            // launcher that holds its host's reservation.
            rendezvous_address: rendezvous.map(str::to_owned),
            rendezvous_listen_fd,
            // A replica sharing its experts joins the expert-parallel union
            // at the address the union's rank 0 accepts on, through the
            // socket it inherits; the other ranks receive only the address.
            expert_parallel: self
                .expert_parallel
                .as_ref()
                .map(|placement| -> anyhow::Result<_> {
                    Ok(ExpertParallelDescriptor {
                        rank: placement.rank + rank,
                        size: placement.size,
                        attention_ranks: placement.attention_ranks,
                        address: placement.address.clone(),
                        listen_fd: expert_listen_fd,
                        exchange: serde_name(&placement.exchange)?,
                    })
                })
                .transpose()?,
            role: serde_name(&self.role)?,
            expert_microbatches: self.expert_microbatches,
            distributed_backend: self.distributed_backend.clone(),
            mesh: self.mesh.clone(),
            lane: self.lanes.clone(),
            // The worker refuses `no_model` without `allow_stub`, so a stub
            // launch states both.
            no_model: self.stub,
            allow_stub: self.stub,
            load_format: self.load_format.clone(),
            download_dir: self.download_dir.as_deref().map(path),
            load_threads: self.load_threads,
            checksum_manifest: self.checksum_manifest.as_deref().map(path),
            model_dtype: self.model_dtype.as_str().to_owned(),
            quantization_config: self.quantization_config.clone(),
            prefix_cache_dtype: self
                .prefix_cache_dtype
                .as_ref()
                .map(|value| value.as_str().to_owned()),
            cache_memory_fraction: self.cache_storage_fraction,
            cache_token_capacity: self.cache_token_capacity,
            attention_backend: self.attention_backend,
            block_size: self.block_size,
            max_batch_calls: self.max_batch_calls,
            max_batch_tokens: self.max_batch_tokens,
            max_request_pool_size: self.max_request_pool_size,
            max_model_len: self.max_model_len,
            max_video_seconds: self.max_video_seconds,
            max_condition_rows: self.max_condition_rows,
            encoder_cache_entries: self.encoder_cache_entries,
            ffmpeg: path(&self.ffmpeg),
            min_video_seconds: self.min_video_seconds,
            graph_policy: self.graph_policy.clone(),
            decode_graph_batch_sizes: self.decode_graph_batch_sizes.clone(),
            prefill_cuda_graph: self.prefill_cuda_graph,
            prefill_outputs: self.prefill_outputs,
            prefill_graph_token_sizes: self.prefill_graph_token_sizes.clone(),
            flow_cuda_graph: self.flow_cuda_graph,
            flow_graph_batch_sizes: self.flow_graph_batch_sizes.clone(),
            flow_graph_shapes: self.flow_graph_shapes.clone(),
            video_text_capacities: self.video_text_capacities.clone(),
            video_canvases: self.video_canvases.clone(),
            canvas_sampling: self.canvas_sampling,
            flashinfer_workspace_size: self.flashinfer_workspace_size,
            flashinfer_use_tensor_core: self.flashinfer_use_tensor_core.clone(),
            flashinfer_decode_backend: self.flashinfer_decode_backend.as_str().to_owned(),
            flashinfer_prefill_backend: self.flashinfer_prefill_backend.as_str().to_owned(),
            flashinfer_decode_split_tile_size: self.flashinfer_decode_split_tile_size,
            flashinfer_prefill_split_tile_size: self.flashinfer_prefill_split_tile_size,
            flashinfer_disable_split_kv: self.flashinfer_disable_split_kv,
        })
    }
}

/// The serialized name of a unit enum variant, as the descriptor carries it.
fn serde_name(value: &impl Serialize) -> anyhow::Result<String> {
    match serde_json::to_value(value)? {
        Value::String(name) => Ok(name),
        other => bail!("expected a named variant, found {other}"),
    }
}

/// One adopted rank: its channel, its process when this engine started it,
/// and the requests outstanding on it.
///
/// Dropping it closes the rank alone through `close_ranks`.
pub(super) struct RankProcess {
    client: Channel,
    /// The rank's validated startup report; the default value until
    /// `finish_startup` stores it.
    info: WorkerInfo,
    /// The startup cancellation flag shared by the ranks launched together,
    /// which the teardown of any of them sets. `WorkerGroup` detaches it
    /// with `set_startup_cancel(None)` once the group is ready.
    startup_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// The rank's process, when this engine started it. A rank placed on
    /// another host is started by that host's launcher and its process is
    /// never visible here; its exit surfaces through the channel once the
    /// connection closes, as a transport error from a receive.
    child: Option<RankChild>,
    /// Most requests outstanding at once on the rank.
    depth: usize,
    rank: u32,
    /// Submitted batches awaiting a response, by message identity.
    pending: HashMap<u64, OutstandingKind>,
    /// Routed results and errors, in arrival order, that `poll_batch` has
    /// not returned yet.
    ready: VecDeque<anyhow::Result<WorkerResult>>,
    /// Next message identity to assign. Identities start at 1 and are never
    /// zero, which is the identity a channel gives a response that states
    /// none.
    next_message_id: u64,
    /// How far the rank's retirement has progressed.
    retirement: Retirement,
    /// Whether the startup report has been asked for and has not arrived.
    starting: bool,
}

/// How far a rank's retirement has progressed.
enum Retirement {
    /// Neither `request_close` nor `terminate` has run.
    Serving,
    /// `request_close` has asked the rank to shut down, and `await_close`
    /// has not yet waited for it. `acknowledgment` is the shutdown request's
    /// message identity while its answer is outstanding; it is `None` when
    /// the rank had already exited or the request could not be sent.
    Closing { acknowledgment: Option<u64> },
    /// The rank was closed or terminated; nothing remains to do.
    Retired,
}

/// One launched rank between its start and its registration.
///
/// The rank's channel exists once the rank has connected and registered, so
/// everything the rank is answered with is retained here meanwhile.
pub(super) struct PendingRank {
    /// Released by adoption; a local rank still held here is killed when
    /// this value drops, as it does when the launch fails.
    child: PendingChild,
    rank: u32,
    /// Most requests outstanding at once on the rank.
    depth: usize,
    /// Largest encoded message in bytes, which `adopt` sets as the bound of
    /// what the head sends the rank.
    max_payload: usize,
    /// Moves by adoption alongside the process it cancels.
    startup_abort: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The launch descriptor `adopt` answers the rank's registration with.
    descriptor: LaunchDescriptor,
}

/// The process of a launched rank that has not been adopted yet, when this
/// engine started it.
struct PendingChild(Option<RankChild>);

impl PendingChild {
    /// Hands the process to the adopting rank, which owns its termination.
    fn release(mut self) -> Option<RankChild> {
        self.0.take()
    }
}

/// What an outstanding request was sent for; only batch submissions are
/// tracked in `RankProcess::pending`.
enum OutstandingKind {
    Batch { batch_id: u64 },
}

/// Caching-allocator variables PyTorch reads, the first taking precedence.
const ALLOCATOR_VARIABLES: [&str; 2] = ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"];

/// The transparent-huge-page option of mimalloc, which PyTorch builds in as
/// its CPU allocator on aarch64 Linux.
const HOST_ALLOCATOR_THP_VARIABLE: &str = "MIMALLOC_ALLOW_THP";

/// Variables a rank's CUDA and NCCL libraries read before their first use,
/// with the value a rank runs with unless the head sets the variable itself.
///
/// - `CUDA_MODULE_LOADING=EAGER`: lazy module loading synchronizes the
///   context when a kernel is first used, which can deadlock against a
///   collective another stream is waiting in for a peer.
/// - `NCCL_RUNTIME_CONNECT=0`: deferred peer handshakes would allocate
///   connection buffers after the rank sealed its device storage.
const RUNTIME_DEFAULTS: [(&str, &str); 2] = [
    ("CUDA_MODULE_LOADING", "EAGER"),
    ("NCCL_RUNTIME_CONNECT", "0"),
];

/// Returns the variables every rank is launched with, given a lookup into
/// the head's environment.
///
/// Every rank serves varying shapes from expandable caching-allocator
/// segments, unless the head's environment already configures the caching
/// allocator under either name, in which case each rank receives exactly
/// the head's variables. Tensor export never depends on the caching
/// allocator: a device product is exported from the rank's own VMM arena or
/// copied into its bounded VMM pool, both reserved outside the allocator.
///
/// Every rank also keeps its host heap out of transparent huge pages unless
/// the head sets `MIMALLOC_ALLOW_THP` itself. mimalloc otherwise advises its
/// 1 GiB arenas for huge pages. On a kernel with 64 KiB base pages a huge
/// page is 512 MiB, and each time khugepaged collapses one it holds the
/// rank's memory map for about 90 ms, which stalls the rank's service
/// thread. With the option off, mimalloc disables transparent huge pages
/// for the rank process.
///
/// The [`RUNTIME_DEFAULTS`] are stated in the environment rather than set by
/// the rank, so they precede any CUDA or NCCL initialization whatever the
/// rank imports first; the head's own values win.
///
/// The variables are set explicitly on each rank's command, even where a
/// local child would inherit them, because a launcher on another host
/// receives only the variables the command sets and applies them over its
/// own environment.
fn worker_environment(
    head: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Vec<(&'static str, std::ffi::OsString)> {
    let mut configured: Vec<_> = ALLOCATOR_VARIABLES
        .into_iter()
        .filter_map(|name| head(name).map(|value| (name, value)))
        .collect();
    if configured.is_empty() {
        configured.push(("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True".into()));
    }
    configured.push((
        HOST_ALLOCATOR_THP_VARIABLE,
        head(HOST_ALLOCATOR_THP_VARIABLE).unwrap_or_else(|| "0".into()),
    ));
    for (name, value) in RUNTIME_DEFAULTS {
        configured.push((name, head(name).unwrap_or_else(|| value.into())));
    }
    configured
}

/// Bound listening sockets a rank inherits and serves collective stores on.
#[derive(Default)]
pub(crate) struct RankSockets {
    /// The group's rendezvous store, served by the group's first rank.
    pub(crate) group: Option<std::net::TcpListener>,
    /// The expert-parallel world's store, served by that world's rank 0.
    pub(crate) experts: Option<std::net::TcpListener>,
}

impl PendingRank {
    /// Spawns one rank, which connects to the head at `head` and registers as
    /// a member of its group's launch `generation`.
    ///
    /// Nothing here waits for the rank. `rendezvous` is the group's
    /// collective store address, and `sockets.group` the socket this rank
    /// serves that store on when it is the group's first rank and this
    /// process spawns it; the rank inherits the socket and this process keeps
    /// no copy. `store_fd` is the descriptor number a first rank on another
    /// host inherits that socket at, as its launcher reported it. With
    /// `remote`, the launch goes to that host's launcher and no process is
    /// started here. `startup_abort` is the startup cancellation flag shared
    /// by the ranks launched together. `sockets.experts` is the socket this
    /// rank serves its expert-parallel world's store on, when it is that
    /// world's rank 0.
    ///
    /// Fails when the rank's host-relative values or its descriptor cannot
    /// be built, the remote delivery fails, or the process cannot be spawned.
    #[expect(
        clippy::too_many_arguments,
        reason = "one rank's launch: its arguments, sockets, host and generation"
    )]
    pub(crate) fn spawn_rank(
        args: &WorkerProcessArgs,
        rank: u32,
        rendezvous: Option<&str>,
        sockets: RankSockets,
        store_fd: Option<std::os::fd::RawFd>,
        remote: Option<&mut super::launcher::RemoteHost<'_>>,
        startup_abort: std::sync::Arc<std::sync::atomic::AtomicBool>,
        head: &str,
        generation: u64,
    ) -> anyhow::Result<Self> {
        let depth = args.queue_depth.max(1);
        let max_payload = args.max_message_bytes.max(1);
        let world_size = args.world_size();

        let mut cmd = uniserve_ipc::launch::rank_command(
            &args.python,
            head,
            &args.worker_id,
            rank,
            generation,
        );

        // Torch and the numerical libraries read the host-relative pair; the
        // global pair names the rank's place in the whole process world.
        let (local_rank, local_world_size) = args.host_slot(rank)?;
        cmd.env("RANK", rank.to_string())
            .env("WORLD_SIZE", world_size.to_string())
            .env("LOCAL_RANK", local_rank.to_string())
            .env("LOCAL_WORLD_SIZE", local_world_size.to_string());

        // The environment is stated even for a local child, which
        // would inherit them, so that a remote launch carries them as well.
        cmd.envs(worker_environment(|name| std::env::var_os(name)));

        // The engine's working directory leads the worker's import path.
        if let Ok(cwd) = std::env::current_dir() {
            let pp = std::env::var("PYTHONPATH").unwrap_or_default();
            cmd.env("PYTHONPATH", format!("{}:{}", cwd.display(), pp));
        }

        // The command owns the store socket from here and closes this
        // process's copy when it is dropped, after the spawn below.
        let rendezvous_listen_fd = sockets
            .group
            .map(|listener| uniserve_ipc::launch::inherit_listener(&mut cmd, listener))
            .or(store_fd);
        let expert_listen_fd = sockets
            .experts
            .map(|listener| uniserve_ipc::launch::inherit_listener(&mut cmd, listener));

        // One typed descriptor carries every launch value. The head keeps it
        // and answers the rank's registration with it.
        let descriptor =
            args.launch_descriptor(rank, rendezvous, rendezvous_listen_fd, expert_listen_fd)?;

        // A rank placed on another host is started by that host's launcher,
        // which needs the same command and environment this process would
        // have used. The head derives both once, here, for either path.
        let child = match remote {
            Some(remote) => {
                remote.deliver(rank, &args.python, &args.worker_id, generation, head, &cmd)?;
                None
            }
            None => Some(RankChild::spawn(&mut cmd).context("spawning the rank process")?),
        };

        Ok(Self {
            child: PendingChild(child),
            rank,
            depth,
            max_payload,
            startup_abort,
            descriptor,
        })
    }

    /// Fails by name when the rank exited before registering.
    ///
    /// `RankRegistry::collect` calls this, through the `alive` callback `adopt_ranks`
    /// passes it, while the launch is incomplete. Reaps the process when it
    /// has exited.
    pub(crate) fn check_alive(&mut self) -> anyhow::Result<()> {
        // A rank started by another host's launcher has no process here. Its
        // exit before registration reaches the head as a report from the
        // launcher that owns it, which `adopt_ranks` checks separately.
        let Some(child) = self.child.0.as_mut() else {
            return Ok(());
        };
        match child.try_wait() {
            Ok(None) => Ok(()),
            Ok(Some(status)) => Err(anyhow::anyhow!(
                "rank {} exited with {status} before registering",
                self.rank
            )),
            Err(error) => Err(error).context("checking a launched rank"),
        }
    }

    /// Answers the rank's registration on `channel` with its launch
    /// descriptor and takes the connection as the rank's channel.
    ///
    /// The returned `RankProcess` owns the rank's process, when this engine
    /// spawned it. Its `info` stays the default until `finish_startup` runs.
    pub(crate) fn adopt(self, mut channel: Channel) -> anyhow::Result<RankProcess> {
        let PendingRank {
            child,
            rank,
            depth,
            max_payload,
            startup_abort,
            descriptor,
        } = self;

        // The process stays owned by `child` until the answer is sent, so a
        // failed send terminates the rank instead of orphaning it.
        channel
            .send(&descriptor)
            .with_context(|| format!("sending rank {rank} its launch descriptor"))?;
        channel.set_bound(max_payload);
        tracing::info!(rank, "a rank registered and received its launch descriptor");

        Ok(RankProcess {
            client: channel,
            info: WorkerInfo::default(),
            startup_cancel: Some(startup_abort),
            child: child.release(),
            depth,
            rank,
            pending: HashMap::new(),
            ready: VecDeque::new(),
            next_message_id: 1,
            retirement: Retirement::Serving,
            starting: false,
        })
    }
}

impl Drop for PendingChild {
    /// A rank that never registered has no channel to close through, so a
    /// failed launch terminates its process here rather than leaking it.
    fn drop(&mut self) {
        if let Some(child) = self.0.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl RankProcess {
    /// Asks the rank for its startup report.
    ///
    /// The rank answers only once its model is built and warmed up;
    /// `poll_startup` takes the answer.
    fn request_startup(&mut self) -> anyhow::Result<()> {
        let message_id = self.alloc_call_id();
        let mut request = WorkerRequest::info();
        request.set_call_id(Some(message_id));
        self.client.send(&request)?;
        self.starting = true;
        Ok(())
    }

    /// Takes the rank's startup report if it has arrived, as `info`. Returns
    /// whether the rank's startup is complete.
    ///
    /// Fails when the rank reports that its startup failed, and when its
    /// connection has closed, which is how a rank that died while starting
    /// is noticed.
    fn poll_startup(&mut self) -> anyhow::Result<bool> {
        let response = self
            .client
            .try_recv::<WorkerResponse>()
            .with_context(|| format!("rank {} was lost during startup", self.rank))?;
        // A rank that is ready stays under watch until its peers are: the
        // receive above is what notices its connection closing.
        if !self.starting {
            return Ok(true);
        }
        let Some(response) = response else {
            return Ok(false);
        };
        let info = match response {
            WorkerResponse::Info { info, .. } => info,
            WorkerResponse::Error { error, .. } => {
                tracing::error!(
                    rank = self.rank,
                    "rank failed during startup: {}",
                    error.message
                );
                bail!(
                    "rank {} failed during startup: {}",
                    self.rank,
                    error.message
                )
            }
            WorkerResponse::Result { result, .. } => {
                // Receiving claims any media the result published, so the
                // shared-memory objects are released rather than left named.
                drop(WorkerResult::receive(result));
                bail!("unexpected startup result response");
            }
            other => bail!("unexpected startup response: {:?}", other.kind()),
        };

        self.check_worker("Worker startup")?;
        self.info = info;
        self.starting = false;
        tracing::info!(?self.info, "worker ready");
        // Warmup and capture have connected every communicator of a ready
        // rank, so the NCCL names it maps now are those its reaping removes
        // should it die without destroying them.
        if let Some(child) = self.child.as_mut()
            && let Err(error) = child.note_nccl_segments()
        {
            tracing::debug!(rank = self.rank, %error, "the rank's NCCL names could not be noted");
        }
        Ok(true)
    }

    /// Allocates the next worker message identifier.
    fn alloc_call_id(&mut self) -> u64 {
        let id = self.next_message_id;
        self.next_message_id += 1;
        id
    }

    /// Replaces the startup cancellation flag `check_worker` consults.
    ///
    /// `WorkerGroup` passes `None` once the group is ready, so a later exit
    /// no longer reads as a cancelled startup on the surviving ranks.
    pub(crate) fn set_startup_cancel(
        &mut self,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) {
        self.startup_cancel = cancel;
    }

    /// Aborts peers when an unfinished deployment is being dismantled.
    /// Ready ranks have detached the flag, so ordinary shutdown is unaffected.
    pub(crate) fn cancel_startup(&self) {
        if let Some(cancel) = &self.startup_cancel {
            cancel.store(true, std::sync::atomic::Ordering::Release);
        }
    }

    /// Terminates a failed or cancelled rank and waits for process-owned resources to retire.
    ///
    /// Sends no shutdown request and discards outstanding and routed results.
    pub(crate) fn terminate(&mut self) {
        self.retirement = Retirement::Retired;
        // A rank this engine started is killed and reaped here. A rank
        // elsewhere has no process here: `WorkerGroup` recovery stops it
        // through `LauncherRegistry::stop_worker`, and a launcher terminates
        // every rank it started when its connection to the head closes.
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.pending.clear();
        self.ready.clear();
    }

    /// Checks child liveness and cancellation of an unfinished startup.
    ///
    /// Fails when the startup cancellation flag is set, when the local
    /// process has exited (reaping it), or when querying the process fails. A
    /// rank on another host is checked for cancellation only.
    pub(super) fn check_worker(&mut self, context: &str) -> anyhow::Result<()> {
        if self
            .startup_cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(std::sync::atomic::Ordering::Acquire))
        {
            bail!("worker startup cancelled during {context}");
        }
        if let Some(child) = self.child.as_mut()
            && let Some(status) = child.try_wait()?
        {
            bail!("worker process exited during {context}: {status}");
        }
        Ok(())
    }

    /// Drains ready responses and routes them into `ready`, and returns how
    /// many were routed.
    ///
    /// A transport, execution, or routing error, or a stall report, stops
    /// the scan and is returned; responses routed before it stay in
    /// `ready`. The channel reports its close, which is the rank's death,
    /// as an error only after every response that arrived whole has been
    /// routed.
    fn drain_ready(&mut self) -> anyhow::Result<usize> {
        let mut drained = 0usize;
        while let Some(response) = self.client.try_recv::<WorkerResponse>()? {
            // A stall report answers no request: it fails the rank.
            if let WorkerResponse::Stalled { stall } = response {
                return Err(self.stalled(&stall));
            }
            // A response answers the request whose identity it states.
            let message_id = response
                .message_id()
                .context("worker response states no message identity")?;
            let kind = self.pending.remove(&message_id).with_context(|| {
                format!("worker response {message_id} answers no outstanding request")
            })?;
            self.route(kind, response)?;
            drained += 1;
        }
        Ok(drained)
    }

    /// Routes one batch response and appends its result to `ready`.
    ///
    /// The result is received, which claims any media it published
    /// (`WorkerResult::receive`). Returns an error and appends nothing when
    /// the worker answered with an
    /// execution error (as a `WorkerExecError`) or an unexpected response
    /// kind, a product locator names an endpoint other than this rank's
    /// reported one, or the result's batch identity differs from the
    /// submitted batch.
    fn route(&mut self, kind: OutstandingKind, response: WorkerResponse) -> anyhow::Result<()> {
        let OutstandingKind::Batch { batch_id } = kind;
        let report = match response {
            WorkerResponse::Result { result, .. } => Ok(WorkerResult::receive(result)),
            WorkerResponse::Error { error, .. } => Err(WorkerExecError {
                batch_id: Some(batch_id),
                fatal: error.fatal,
                code: error.code,
                message: error.message,
                phase: error.phase,
                route: error.route,
                calls: error.calls,
            }
            .into()),
            other => Err(anyhow::anyhow!(
                "unexpected execute response kind: {:?}",
                other.kind()
            )),
        };
        let report = report?;

        // A product must name this rank's reported endpoint, which identifies
        // the incarnation whose startup report `finish_startup` accepted.
        for product in &report.products {
            anyhow::ensure!(
                product
                    .value
                    .locators()
                    .all(|locator| locator.source == self.info.endpoint),
                "worker published a product from an unbound rank incarnation"
            );
        }
        anyhow::ensure!(
            report.batch_id == batch_id,
            "worker result batch id mismatch: expected {batch_id}, got {}",
            report.batch_id
        );
        self.ready.push_back(Ok(report));
        Ok(())
    }

    /// Logs a lane the rank reports stalled and returns the failure of the
    /// rank, which names the lane and the forward its job belongs to.
    ///
    /// The rank's lane stays inside one job, so the rank cannot complete
    /// the batches behind it: the group fails it as it fails a lost rank.
    fn stalled(&self, stall: &uniserve_ipc::Stall) -> anyhow::Error {
        let lane = stall.lane.as_ref().map_or_else(
            || "the Python thread".to_owned(),
            |lane| format!("lane {lane}"),
        );
        let calls = stall
            .calls
            .iter()
            .map(|call| format!("{}/{:?}", call.request_key.request_id.0, call.call_id))
            .collect::<Vec<_>>();
        tracing::error!(
            rank = self.rank,
            lane = %lane,
            batch_id = ?stall.batch_id,
            route = ?stall.route,
            method = ?stall.method,
            calls = ?calls,
            running_ms = stall.running_ms,
            "a rank reports a lane stalled inside one job"
        );

        let forward = match stall.batch_id {
            Some(batch_id) => format!(
                "a job of batch {batch_id} ({}, method {}, calls [{}])",
                stall.route.as_deref().unwrap_or("no calls"),
                stall.method.as_deref().unwrap_or("none"),
                calls.join(", "),
            ),
            None => format!(
                "a job of no batch (method {})",
                stall.method.as_deref().unwrap_or("none")
            ),
        };
        anyhow::anyhow!(
            "{lane} stalled: it has performed {forward} for {:.1} s without completing it",
            Duration::from_millis(stall.running_ms).as_secs_f64()
        )
    }

    /// Returns the descriptors that signal this rank's progress: its
    /// connection, whose close is also the rank's death.
    pub(crate) fn progress_fds(&self) -> Vec<libc::pollfd> {
        self.client.poll_fds().to_vec()
    }
}

impl RankProcess {
    /// Returns metadata for the physical worker.
    pub(super) fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Submits one physical run and records its outstanding message identity.
    ///
    /// Hands the batch back as `WouldBlock` when `depth` requests are already
    /// outstanding; the batch is not recorded as pending then.
    /// Fails when draining earlier responses or the send fails.
    pub(super) fn submit_batch(&mut self, batch: Batch) -> Result<(), BatchSubmitError> {
        // Answered requests free their slots before the depth check.
        self.drain_ready().map_err(BatchSubmitError::Failed)?;
        if self.pending.len() >= self.depth {
            return Err(BatchSubmitError::WouldBlock(Box::new(batch)));
        }
        let batch_id = batch.batch_id;
        let message_id = self.alloc_call_id();
        let mut req = WorkerRequest::submit(batch);
        req.set_call_id(Some(message_id));
        self.client
            .send(&req)
            .map_err(|error| BatchSubmitError::Failed(error.into()))?;
        self.pending
            .insert(message_id, OutstandingKind::Batch { batch_id });
        Ok(())
    }

    /// Returns completed responses before failing outstanding work on disconnect.
    ///
    /// Returns the next entry of `ready`, waiting up to `timeout`, and
    /// `Ok(None)` when the timeout passes with nothing ready. An entry is a
    /// result or the error that draining or routing a response produced.
    /// Results already received are returned before a rank exit, startup
    /// cancellation, the rank's connection closing, or a failed wait fails
    /// the call.
    pub(super) fn poll_batch(&mut self, timeout: Duration) -> anyhow::Result<Option<WorkerResult>> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(result) = self.ready.pop_front() {
                return result.map(Some);
            }
            if let Err(error) = self.drain_ready() {
                // Complete replies remain ahead of a trailing transport or
                // decoding error in the same drain.
                self.ready.push_back(Err(error));
                continue;
            }
            if !self.ready.is_empty() {
                continue;
            }
            self.check_worker("executor poll")?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            self.client.wait(remaining.min(WORKER_CHECK_INTERVAL))?;
        }
    }

    /// Drains outstanding calls and asks the rank to shut down.
    ///
    /// A startup that was cancelled goes straight to `terminate`. Otherwise
    /// only a first call acts, and none after `terminate`. A live rank gets up
    /// to `WORKER_DRAIN_TIMEOUT` to answer outstanding batches and is then
    /// sent a shutdown request, without waiting for its answer; `await_close`
    /// completes the retirement. A failure to reach the rank is absorbed, and
    /// `await_close` then bounds the rank by its deadline alone.
    fn request_close(&mut self) {
        // A rank of this launch exited before the group became ready, so this
        // rank is terminated without draining or a shutdown request.
        if self
            .startup_cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(std::sync::atomic::Ordering::Acquire))
        {
            self.terminate();
            return;
        }
        if !matches!(self.retirement, Retirement::Serving) {
            return;
        }

        let mut acknowledgment = None;
        // A rank started elsewhere is never observed to have exited here, so
        // shutdown drains its channel as it would a live local rank.
        let exited = matches!(
            self.child.as_mut().map(RankChild::try_wait),
            Some(Ok(Some(_)))
        );
        if !exited {
            // Bound response draining so shutdown can advance to graceful termination
            // and, if necessary, forced process cleanup.
            let drain_deadline = Instant::now() + WORKER_DRAIN_TIMEOUT;
            while !self.pending.is_empty() {
                if Instant::now() >= drain_deadline {
                    tracing::warn!(
                        pending = self.pending.len(),
                        "worker did not drain in-flight responses before shutdown deadline"
                    );
                    break;
                }
                match self.drain_ready() {
                    Ok(0) => {
                        if self.check_worker("shutdown drain").is_err() {
                            break;
                        }
                        let remaining = drain_deadline.saturating_duration_since(Instant::now());
                        let _ = self.client.wait(remaining.min(WORKER_CHECK_INTERVAL));
                    }
                    Ok(_) => {}
                    Err(_) => break,
                }
            }

            // A rank still running after the drain is asked to shut down.
            if !matches!(
                self.child.as_mut().map(RankChild::try_wait),
                Some(Ok(Some(_)))
            ) {
                let message_id = self.alloc_call_id();
                let mut req = WorkerRequest::close();
                req.set_call_id(Some(message_id));
                acknowledgment = self.client.send(&req).ok().map(|()| message_id);
            }
        }
        self.retirement = Retirement::Closing { acknowledgment };
    }

    /// Waits until `deadline` for a rank `request_close` asked to shut down,
    /// then kills its local process if it is still running.
    ///
    /// The shutdown request stays outstanding until the rank answers it, so
    /// the answer, which the rank sends before its teardown begins, always
    /// has a receiver. A local rank is retired once its process exits,
    /// answered or not. The process of a rank elsewhere belongs to its
    /// launcher, so that rank is retired once it answers or its channel
    /// fails; `close_ranks` then follows it to its exit. Acts only on a rank
    /// that is closing.
    fn await_close(&mut self, deadline: Instant) {
        let Retirement::Closing { mut acknowledgment } =
            std::mem::replace(&mut self.retirement, Retirement::Retired)
        else {
            return;
        };

        loop {
            // A failed receive means the rank's end of the channel is gone,
            // which answers the request as finally as a response does. A
            // response to anything else is a result nobody waits for any more.
            while let Some(message_id) = acknowledgment {
                match self.client.try_recv::<WorkerResponse>() {
                    Ok(Some(response)) if response.message_id() != Some(message_id) => {}
                    Ok(None) => break,
                    Ok(Some(_)) | Err(_) => acknowledgment = None,
                }
            }
            let retired = match self.child.as_mut() {
                Some(child) => !matches!(child.try_wait(), Ok(None)),
                None => acknowledgment.is_none(),
            };
            if retired {
                return;
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            // A response ends the wait; a process exit does not, so the wait
            // is also bounded by the exit poll interval. A closed channel
            // fails the wait at once and still gets the interval, so the
            // loop never spins.
            let interval = remaining.min(EXIT_POLL_INTERVAL);
            if self.client.wait(interval).is_err() {
                std::thread::sleep(interval);
            }
        }

        if let Some(child) = self.child.as_mut() {
            tracing::warn!(
                rank = self.rank,
                "worker did not exit before the shutdown deadline; killing it"
            );
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Waits until every rank in `ranks` has reported that it is ready, and
/// checks each report against its launch.
///
/// A rank answers only once its model is built and warmed up, and ranks that
/// initialize together wait for one another, so the ranks are asked first
/// and then awaited together: the wait has no deadline, and ends by any
/// rank's failure. A rank that dies while starting closes its connection,
/// which fails the startup of all of them by that rank's name instead of
/// leaving its peers waiting for it.
pub(super) fn finish_startup(ranks: &mut [&mut RankProcess]) -> anyhow::Result<()> {
    for rank in ranks.iter_mut() {
        rank.request_startup()?;
    }

    let started = Instant::now();
    let mut last_log = started;
    let mut last_worker_check = started;
    loop {
        // Every rank is polled and parked on, including those already
        // ready, so that the close of any connection ends the wait.
        let mut ready = true;
        let mut descriptors = Vec::new();
        for rank in ranks.iter_mut() {
            ready &= rank.poll_startup()?;
            descriptors.extend(rank.progress_fds());
        }
        if ready {
            return Ok(());
        }

        if last_worker_check.elapsed() >= WORKER_CHECK_INTERVAL {
            for rank in ranks.iter_mut() {
                if let Err(lost) = rank.check_worker("Worker startup") {
                    // A rank that reported its failure before it exited is
                    // named by its own error, which its connection may still
                    // hold.
                    rank.poll_startup()?;
                    return Err(lost);
                }
            }
            last_worker_check = Instant::now();
        }
        if last_log.elapsed() >= STARTUP_LOG_INTERVAL {
            tracing::info!(
                elapsed_secs = started.elapsed().as_secs(),
                "workers still busy during startup"
            );
            last_log = Instant::now();
        }
        let until_check = WORKER_CHECK_INTERVAL.saturating_sub(last_worker_check.elapsed());
        let until_log = STARTUP_LOG_INTERVAL.saturating_sub(last_log.elapsed());
        crate::worker::park_descriptors(&descriptors, until_check.min(until_log))?;
    }
}

/// Retires the ranks of one worker group together, absorbing every failure.
///
/// A rank's orderly teardown retires the communicators and process groups it
/// shares with the group's other ranks (`Worker.close` on the Python side),
/// and that retirement completes on any rank only once every rank has
/// reached it. Every rank is therefore asked to shut down, after draining its
/// outstanding calls, before any is waited for, and all of them then share
/// one `WORKER_EXIT_GRACE` deadline, after which a local process still
/// running is killed. Asking one rank at a time would leave the first one
/// waiting on peers that have not been asked, until it is killed with its
/// communicators unreleased.
///
/// `launchers` names the registry that started the group's ranks on other
/// hosts, and the group's worker id. A rank answers its shutdown request
/// before its teardown begins, and a launcher kills the ranks it still runs
/// once the head lets go of the registry, so the ranks elsewhere are held
/// until their launchers report their exits, against the same deadline.
pub(super) fn close_ranks(
    ranks: &mut [RankProcess],
    launchers: Option<(&super::launcher::Launchers, &str)>,
) {
    for rank in ranks.iter_mut() {
        rank.request_close();
    }

    // The ranks elsewhere that were asked to shut down. A rank whose request
    // could not be sent is unreachable, and nothing it could still release
    // waits on this process.
    let remote: BTreeSet<u32> = ranks
        .iter()
        .filter(|rank| {
            rank.child.is_none()
                && matches!(
                    rank.retirement,
                    Retirement::Closing {
                        acknowledgment: Some(_)
                    }
                )
        })
        .map(|rank| rank.rank)
        .collect();

    let deadline = Instant::now() + WORKER_EXIT_GRACE;
    for rank in ranks.iter_mut() {
        rank.await_close(deadline);
    }
    if let Some((registry, worker_id)) = launchers {
        await_remote_exits(registry, worker_id, remote, deadline);
    }
}

/// Waits until `deadline` for the launchers to report the exit of each rank
/// in `remote`, all ranks of `worker_id` on other hosts.
///
/// Gives up when the registry cannot be locked. A rank still running at the
/// deadline is left to its launcher, which kills it once the head releases
/// the registry.
fn await_remote_exits(
    registry: &super::launcher::Launchers,
    worker_id: &str,
    mut remote: BTreeSet<u32>,
    deadline: Instant,
) {
    while !remote.is_empty() {
        let Ok(mut hosts) = super::launcher::lock(registry) else {
            return;
        };
        for (_, exit) in hosts.drain_exits(worker_id) {
            remote.remove(&exit.rank);
        }
        drop(hosts);
        if remote.is_empty() {
            return;
        }

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            tracing::warn!(
                worker = worker_id,
                ranks = ?remote,
                "ranks on other hosts did not exit before the shutdown deadline"
            );
            return;
        }
        std::thread::sleep(remaining.min(EXIT_POLL_INTERVAL));
    }
}

impl Drop for RankProcess {
    /// Closes the rank as `close_ranks` would close it alone. A rank on
    /// another host is not followed to its exit, since no launcher registry
    /// is at hand.
    fn drop(&mut self) {
        self.cancel_startup();
        close_ranks(std::slice::from_mut(self), None);
    }
}

#[cfg(test)]
mod tests {
    use super::{LaneConfig, WorkerProcessArgs, worker_environment};
    use std::collections::BTreeMap;
    use std::ffi::OsString;

    /// A rank receives the operator's local base checkpoint, and none when
    /// the operator named none, so the worker then reads the base from the
    /// Hugging Face cache.
    #[test]
    fn the_launch_descriptor_carries_a_named_base_checkpoint() {
        let mut args = WorkerProcessArgs {
            model: "/models/FastH3-OmniRef".into(),
            ..WorkerProcessArgs::default()
        };
        let descriptor = args
            .launch_descriptor(0, None, None, None)
            .expect("descriptor");
        assert_eq!(descriptor.base_model, None);

        args.base_model = Some("/models/MiniMax-H3".into());
        let descriptor = args
            .launch_descriptor(0, None, None, None)
            .expect("descriptor");
        assert_eq!(descriptor.base_model.as_deref(), Some("/models/MiniMax-H3"));
    }

    /// The head answers a rank's registration with its launch descriptor as
    /// one frame, so every field must survive the frame's encoding, with and
    /// without the values that are optional.
    #[test]
    fn the_launch_descriptor_round_trips_through_a_frame() {
        let mut args = WorkerProcessArgs::default();
        for base_model in [None, Some("/models/MiniMax-H3".into())] {
            args.base_model = base_model;
            let descriptor = args
                .launch_descriptor(0, Some("127.0.0.1:29500"), Some(7), None)
                .expect("descriptor");
            let frame = uniserve_ipc::Frame::encode(&descriptor).expect("a frame");
            let received: uniserve_ipc::LaunchDescriptor = frame.decode().expect("a descriptor");
            assert_eq!(received, descriptor);
        }
    }

    /// `--lane` accepts only the fields the worker applies, so a misspelled
    /// optional limit fails at argument parsing instead of leaving the lane
    /// without that limit.
    #[test]
    fn a_lane_naming_an_unknown_field_is_refused() {
        let lane = r#"{"lane_id":"decode","sm_budget":64,"domains":["decode"]"#;
        let parsed = format!("{lane}}}")
            .parse::<LaneConfig>()
            .expect("decode lane");
        let args = WorkerProcessArgs {
            lanes: vec![parsed],
            ..WorkerProcessArgs::default()
        };
        let descriptor = args
            .launch_descriptor(0, None, None, None)
            .expect("descriptor");
        assert_eq!(
            serde_json::to_value(&descriptor.lane[0].call_kinds).expect("kinds"),
            serde_json::json!(["decode", "verify", "token_denoising"])
        );

        for misspelled in [r#""max_batch_token":4096"#, r#""max_inflght":2"#] {
            let parsed = format!("{lane},{misspelled}}}").parse::<LaneConfig>();
            assert!(parsed.is_err(), "{misspelled} must be refused");
        }
    }

    /// A lane partitions compute only: its calls share the worker-wide prefix cache
    /// and latent pools, whose placement the scheduler owns, so a lane
    /// naming a capacity of its own is refused rather than accepted without
    /// effect.
    #[test]
    fn a_lane_naming_a_pool_capacity_is_refused() {
        let lane = r#"{"lane_id":"decode","sm_budget":64,"domains":["decode"]"#;
        for capacity in [
            r#""kv_capacity_tokens":4096"#,
            r#""latent_capacity_units":8"#,
        ] {
            let parsed = format!("{lane},{capacity}}}").parse::<LaneConfig>();
            assert!(parsed.is_err(), "{capacity} must be refused");
        }
    }

    /// Every rank runs with the head's caching-allocator configuration,
    /// under whichever names the head sets, or with expandable segments when
    /// the head configures none, and with its host heap out of transparent
    /// huge pages unless the head decides otherwise. The result names each
    /// variable, since a remote launch carries only the variables its
    /// command sets.
    #[test]
    fn ranks_run_with_the_heads_allocator_configuration() {
        let resolve = |head: &[(&'static str, &str)]| {
            let head: BTreeMap<_, _> = head.iter().copied().collect();
            worker_environment(|name| head.get(name).map(OsString::from))
                .into_iter()
                .filter(|(name, _)| !name.starts_with("CUDA_") && !name.starts_with("NCCL_"))
                .collect::<BTreeMap<_, _>>()
        };
        let expected = |variables: &[(&'static str, &str)]| {
            variables
                .iter()
                .map(|&(name, value)| (name, OsString::from(value)))
                .collect::<BTreeMap<_, _>>()
        };

        assert_eq!(
            resolve(&[]),
            expected(&[
                ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
                ("MIMALLOC_ALLOW_THP", "0"),
            ])
        );
        for head in [
            vec![(
                "PYTORCH_CUDA_ALLOC_CONF",
                "garbage_collection_threshold:0.6",
            )],
            vec![("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")],
            vec![
                ("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync"),
                ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False"),
            ],
        ] {
            let mut ranks = head.clone();
            ranks.push(("MIMALLOC_ALLOW_THP", "0"));
            assert_eq!(resolve(&head), expected(&ranks));
        }
        assert_eq!(
            resolve(&[("MIMALLOC_ALLOW_THP", "1")]),
            expected(&[
                ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
                ("MIMALLOC_ALLOW_THP", "1"),
            ])
        );
    }

    /// Eager CUDA module loading and connect-at-startup NCCL precede any
    /// CUDA or NCCL use in every rank, unless the head sets either itself.
    #[test]
    fn runtime_defaults_yield_to_the_heads_values() {
        let resolve = |head: &[(&'static str, &str)]| {
            let head: BTreeMap<_, _> = head.iter().copied().collect();
            worker_environment(|name| head.get(name).map(OsString::from))
                .into_iter()
                .filter(|(name, _)| name.starts_with("CUDA_") || name.starts_with("NCCL_"))
                .collect::<BTreeMap<_, _>>()
        };
        assert_eq!(
            resolve(&[]),
            BTreeMap::from([
                ("CUDA_MODULE_LOADING", OsString::from("EAGER")),
                ("NCCL_RUNTIME_CONNECT", OsString::from("0")),
            ])
        );
        assert_eq!(
            resolve(&[
                ("CUDA_MODULE_LOADING", "LAZY"),
                ("NCCL_RUNTIME_CONNECT", "1")
            ]),
            BTreeMap::from([
                ("CUDA_MODULE_LOADING", OsString::from("LAZY")),
                ("NCCL_RUNTIME_CONNECT", OsString::from("1")),
            ])
        );
    }
}
