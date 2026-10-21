//! The launch settings `WorkerProcessArgs` carries (`LaneConfig`,
//! [`FlashInferBackend`], [`GraphPolicy`], [`LoadFormat`]), that type's
//! defaults, the launch descriptor a rank
//! is started with, and the environment every rank is launched with, given the
//! head's own.

use anyhow::{Context, bail};
use ipc::{DEFAULT_COMPONENT, ExpertParallelDescriptor, LaunchDescriptor};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::worker::WorkerProcessArgs;

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

/// Which GPU computations a rank captures as CUDA graphs at startup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphPolicy {
    /// Every call serves eagerly.
    Off,
    /// Captures graphs unless serving does not capture the attention
    /// provider's calls, and then serves eagerly.
    #[default]
    Auto,
    /// Requires graphs and refuses an attention provider serving does not
    /// capture.
    Full,
}

/// How a rank reads its checkpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadFormat {
    /// Detects the checkpoint files' own format.
    #[default]
    Auto,
    /// Reads safetensors files.
    Safetensors,
    /// Reads PyTorch pickle files.
    Pt,
    /// Initializes weights without reading the checkpoint.
    Dummy,
    /// Reads the checkpoint one layer at a time.
    Layered,
}

impl WorkerProcessArgs {
    /// The `uniserve` executable that runs workers when none is configured:
    /// the one installed next to the running binary, when that directory is a
    /// Python environment's `bin/`; then the active virtual environment's;
    /// then `uniserve` on `PATH`.
    ///
    /// The executable runs workers in the environment it is installed in,
    /// so a head built outside one, such as a development build, takes the
    /// active environment's.
    pub fn default_executable() -> std::path::PathBuf {
        const EXECUTABLE: &str = "uniserve";
        let environment_bin = |directory: &std::path::Path| {
            ["python", "python3"]
                .iter()
                .any(|name| directory.join(name).is_file())
                .then(|| directory.join(EXECUTABLE))
                .filter(|candidate| candidate.is_file())
        };

        if let Ok(executable) = std::env::current_exe()
            && let Some(candidate) = executable.parent().and_then(environment_bin)
        {
            return candidate;
        }
        if let Some(environment) = std::env::var_os("VIRTUAL_ENV")
            && let Some(candidate) =
                environment_bin(&std::path::Path::new(&environment).join("bin"))
        {
            return candidate;
        }
        EXECUTABLE.into()
    }
}

impl WorkerProcessArgs {
    /// Call batches kept in flight against a rank unless a launch sets its
    /// queue depth.
    pub const DEFAULT_QUEUE_DEPTH: usize = 2;
    /// How long a submitted batch may go without rank progress before the
    /// group is replaced (`worker_config.watchdog_timeout`), as SGLang's default.
    pub const DEFAULT_WATCHDOG_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);
    /// Share of device memory a rank may hold for weights and KV cache.
    pub const DEFAULT_MEM_FRACTION_STATIC: f64 = 0.70;
    /// Attention provider name: a native kernel chosen per call.
    pub const DEFAULT_ATTENTION_BACKEND: &'static str = "auto";
    /// FlashInfer workspace capacity in bytes.
    pub const DEFAULT_FLASHINFER_WORKSPACE_SIZE: u64 = 512 * 1024 * 1024;
    /// FlashInfer backend of decode attention.
    pub const DEFAULT_FLASHINFER_DECODE_BACKEND: FlashInferBackend = FlashInferBackend::Fa2;
    /// FlashInfer backend of prefill attention.
    pub const DEFAULT_FLASHINFER_PREFILL_BACKEND: FlashInferBackend = FlashInferBackend::Auto;
    /// The `ffmpeg` executable a media reader decodes reference videos with.
    pub const DEFAULT_FFMPEG: &'static str = "ffmpeg";
    /// The longest video duration, in seconds, the video API admits; the
    /// frontend states a deployment's own bound.
    pub const DEFAULT_MAX_VIDEO_SECONDS: f64 = 15.0;
    /// Independent microbatches of attention and remote experts.
    pub const DEFAULT_EXPERT_MICROBATCHES: u32 = 1;
    /// Numeric data type of model parameters and activations.
    pub const DEFAULT_DTYPE: crate::ModelDtype = crate::ModelDtype::BFloat16;
    /// Whether every prefill call replays a CUDA graph.
    pub const DEFAULT_PREFILL_CUDA_GRAPH: bool = true;
    /// Whether image denoising calls replay CUDA graphs.
    pub const DEFAULT_FLOW_CUDA_GRAPH: bool = true;
}

impl Default for WorkerProcessArgs {
    /// Returns worker launch settings suitable for a single local rank.
    ///
    /// The tuning values are the `DEFAULT_*` constants of this type, which
    /// the `uniserve` command line also takes as its option defaults.
    fn default() -> Self {
        Self {
            worker_id: "worker".into(),
            executable: Self::default_executable(),
            // A launch the server does not name serves a deployment of its
            // own, so independent launches never contest one claim.
            deployment: format!("deployment-{}", uuid::Uuid::new_v4().simple()),
            page_tier: false,
            storage_connector: None,
            keep_standby: false,
            standby: false,
            cache_units: None,
            daemon: String::new(),
            watchdog_timeout: Self::DEFAULT_WATCHDOG_TIMEOUT,
            model: String::new(),
            ranks: vec![crate::WorkerRank {
                host: "localhost".into(),
                device: "cuda:0".into(),
            }],
            host: "localhost".into(),
            // One local rank running one component. A launch replaces this
            // with the components the model being served declared.
            components: crate::WorkerGroupConfig::single_component(DEFAULT_COMPONENT, 1),
            role: crate::WorkerGroupRole::Model,
            peers: Default::default(),
            stub: false,
            queue_depth: Self::DEFAULT_QUEUE_DEPTH,
            max_total_tokens: None,
            block_size: None,
            max_batch_calls: 128,
            max_batch_tokens: 16_384,
            // The engine's default running-request limit plus the largest
            // row reserve a runtime keeps outside it.
            max_request_pool_size: (crate::scheduler::DEFAULT_MAX_RUNNING_REQUESTS
                + crate::scheduler::MAX_GUIDANCE_PREFIX_ROWS)
                as u32,
            attention_backend: Self::DEFAULT_ATTENTION_BACKEND.into(),
            transfer: Default::default(),
            load_format: LoadFormat::default(),
            download_dir: None,
            load_threads: None,
            dtype: Self::DEFAULT_DTYPE,
            quantization_config: serde_json::json!({}),
            kv_cache_dtype: None,
            mem_fraction_static: Self::DEFAULT_MEM_FRACTION_STATIC,
            worker_mesh: None,
            distributed_backend: None,
            lanes: Vec::new(),
            graph_policy: GraphPolicy::default(),
            decode_graph_batch_sizes: None,
            prefill_cuda_graph: Self::DEFAULT_PREFILL_CUDA_GRAPH,
            prefill_outputs: true,
            prefill_graph_token_sizes: None,
            flow_cuda_graph: Self::DEFAULT_FLOW_CUDA_GRAPH,
            flow_graph_batch_sizes: None,
            flow_graph_shapes: None,
            video_text_capacities: None,
            video_resolutions: Vec::new(),
            video_aspect_ratios: Vec::new(),
            canvas_sampling: None,
            flashinfer_workspace_size: Self::DEFAULT_FLASHINFER_WORKSPACE_SIZE,
            flashinfer_use_tensor_core: None,
            flashinfer_decode_backend: Self::DEFAULT_FLASHINFER_DECODE_BACKEND,
            flashinfer_prefill_backend: Self::DEFAULT_FLASHINFER_PREFILL_BACKEND,
            flashinfer_decode_split_tile_size: None,
            flashinfer_prefill_split_tile_size: None,
            flashinfer_split_kv: true,
            max_model_len: 8192,
            max_video_seconds: Self::DEFAULT_MAX_VIDEO_SECONDS,
            max_condition_rows: None,
            encoder_cache_entries: 256,
            ffmpeg: Self::DEFAULT_FFMPEG.into(),
            min_video_seconds: None,
            expert_parallel: None,
            expert_microbatches: Self::DEFAULT_EXPERT_MICROBATCHES,
        }
    }
}

impl WorkerProcessArgs {
    /// Returns one rank's index among the ranks on its own host and that
    /// host's rank count.
    ///
    /// These are the host-relative `LOCAL_RANK` and `LOCAL_WORLD_SIZE`
    /// values: a rank's position among the ranks the placement puts on the
    /// same host, and how many ranks that host holds. They coincide with the
    /// global values while a group occupies one host. `rank` must index
    /// `self.ranks`.
    pub(super) fn host_slot(&self, rank: u32) -> anyhow::Result<(u32, u32)> {
        let rank = rank as usize;
        let host = &self.ranks[rank].host;
        let resident = |placed: &&crate::WorkerRank| &placed.host == host;

        // The rank's host index counts the ranks placed before it on its host.
        let index = self.ranks[..rank].iter().filter(resident).count();
        let count = self.ranks.iter().filter(resident).count();
        Ok((
            u32::try_from(index).context("a host's rank index exceeds u32")?,
            u32::try_from(count).context("a host's rank count exceeds u32")?,
        ))
    }

    /// Returns the number of ranks in this group.
    pub(super) fn world_size(&self) -> u32 {
        self.ranks.len() as u32
    }

    /// Builds the typed launch descriptor the worker process consumes.
    ///
    /// Every launch value is stated explicitly, so the launching side is the
    /// single source of defaults and the rank never re-derives one. The head
    /// answers the rank's registration with it; only the head's address and
    /// the rank's identity travel on argv.
    ///
    /// `rank` must index `self.ranks`. Fails when a host-relative rank value
    /// does not fit `u32` or the role does not serialize.
    pub(crate) fn launch_descriptor(
        &self,
        rank: u32,
        storage: ipc::StorageLaunch,
        rendezvous: Option<&str>,
        rendezvous_listen_fd: Option<std::os::fd::RawFd>,
        expert_listen_fd: Option<std::os::fd::RawFd>,
    ) -> anyhow::Result<LaunchDescriptor> {
        let depth = self.queue_depth.max(1);
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
            // The rank reports a stalled lane after half of it: see
            // `WorkerProcessArgs::watchdog_timeout`.
            watchdog_timeout_ms: u64::try_from(self.watchdog_timeout.as_millis())
                .unwrap_or(u64::MAX),
            model: self.model.clone(),
            storage,
            host: self.ranks[rank as usize].host.clone(),
            device: self.ranks[rank as usize].device.clone(),
            rank,
            local_rank: self.host_slot(rank)?.0,
            world_size: self.world_size(),
            components: self.components.clone(),
            deployment_components: deployment,
            transfer_backends: names(&transfer_backends),
            export_backends: names(&export_backends),
            // Every rank of a group meets its peers at this address. The
            // group's first rank accepts them on the socket it inherits at
            // the named descriptor rather than binding the port itself; the
            // daemon of its node holds the reservation until it spawns it.
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
            mesh: self.worker_mesh.clone(),
            lane: self.lanes.clone(),
            no_model: self.stub,
            load_format: serde_name(&self.load_format)?,
            download_dir: self.download_dir.as_deref().map(path),
            load_threads: self.load_threads,
            model_dtype: self.dtype.as_str().to_owned(),
            quantization_config: self.quantization_config.clone(),
            prefix_cache_dtype: self
                .kv_cache_dtype
                .as_ref()
                .map(|value| value.as_str().to_owned()),
            mem_fraction_static: self.mem_fraction_static,
            max_total_tokens: self.max_total_tokens,
            cache_units: self.cache_units,
            attention_backend: self.attention_backend.clone(),
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
            graph_policy: serde_name(&self.graph_policy)?,
            decode_graph_batch_sizes: self.decode_graph_batch_sizes.clone(),
            prefill_cuda_graph: self.prefill_cuda_graph,
            prefill_outputs: self.prefill_outputs,
            prefill_graph_token_sizes: self.prefill_graph_token_sizes.clone(),
            flow_cuda_graph: self.flow_cuda_graph,
            flow_graph_batch_sizes: self.flow_graph_batch_sizes.clone(),
            flow_graph_shapes: self.flow_graph_shapes.clone(),
            video_text_capacities: self.video_text_capacities.clone(),
            video_resolutions: self.video_resolutions.clone(),
            video_aspect_ratios: self.video_aspect_ratios.clone(),
            canvas_sampling: self.canvas_sampling,
            flashinfer_workspace_size: self.flashinfer_workspace_size,
            flashinfer_use_tensor_core: self.flashinfer_use_tensor_core,
            flashinfer_decode_backend: self.flashinfer_decode_backend.as_str().to_owned(),
            flashinfer_prefill_backend: self.flashinfer_prefill_backend.as_str().to_owned(),
            flashinfer_decode_split_tile_size: self.flashinfer_decode_split_tile_size,
            flashinfer_prefill_split_tile_size: self.flashinfer_prefill_split_tile_size,
            flashinfer_disable_split_kv: !self.flashinfer_split_kv,
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
/// The variables are set explicitly on each rank's command, because the
/// daemon that starts the rank applies only the variables the command sets
/// over its own environment.
pub(super) fn worker_environment(
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

#[cfg(test)]
mod tests {
    use super::{WorkerProcessArgs, worker_environment};
    use ipc::{LaneConfig, LaneSpec, StorageLaunch};
    use std::collections::BTreeMap;
    use std::ffi::OsString;

    /// A storage launch as a claim at a host's server states it.
    fn storage() -> StorageLaunch {
        StorageLaunch {
            socket: "/run/uniserve/storage.sock".into(),
            deployment: "model".into(),
            claim: [7; 16],
            page_tier: true,
            tcp: Some("10.0.0.2:4100".into()),
            standby: false,
        }
    }

    /// The head answers a rank's registration with its launch descriptor as
    /// one frame, so every field must survive the frame's encoding, with and
    /// without the values that are optional.
    #[test]
    fn the_launch_descriptor_round_trips_through_a_frame() {
        let args = WorkerProcessArgs::default();
        for (rendezvous, generation) in [(None, None), (Some("127.0.0.1:29500"), Some(7))] {
            let descriptor = args
                .launch_descriptor(0, storage(), rendezvous, generation, None)
                .expect("descriptor");
            let frame = ipc::Frame::encode(&descriptor).expect("a frame");
            let received: ipc::LaunchDescriptor = frame.decode().expect("a descriptor");
            assert_eq!(received, descriptor);
        }
    }

    /// A lane binds the call kinds of the domains it names, and a field it
    /// does not take is refused when it is read, instead of leaving the lane
    /// without that limit.
    #[test]
    fn a_lane_naming_an_unknown_field_is_refused() {
        let lane = r#"{"lane_id":"decode","sm_budget":64,"domains":["decode"]"#;
        let spec = serde_json::from_str::<LaneSpec>(&format!("{lane}}}")).expect("decode lane");
        let args = WorkerProcessArgs {
            lanes: vec![LaneConfig::try_from(spec).expect("a valid lane")],
            ..WorkerProcessArgs::default()
        };
        let descriptor = args
            .launch_descriptor(0, storage(), None, None, None)
            .expect("descriptor");
        assert_eq!(
            serde_json::to_value(&descriptor.lane[0].call_kinds).expect("kinds"),
            serde_json::json!(["decode", "verify", "token_denoising"])
        );

        for misspelled in [r#""max_batch_token":4096"#, r#""max_inflght":2"#] {
            let parsed = serde_json::from_str::<LaneSpec>(&format!("{lane},{misspelled}}}"));
            assert!(parsed.is_err(), "{misspelled} must be refused");
        }
    }

    /// A lane without SM partition or with a zero batch or in-flight limit
    /// is refused when it is resolved, before any rank starts, since a rank
    /// checks only what spans its lanes.
    #[test]
    fn a_lane_with_a_zero_budget_or_limit_is_refused() {
        let lane = r#"{"lane_id":"decode","domains":["decode"]"#;
        for field in [
            r#""sm_budget":0"#,
            r#""sm_budget":64,"max_batch_calls":0"#,
            r#""sm_budget":64,"max_batch_tokens":0"#,
            r#""sm_budget":64,"max_inflight":0"#,
        ] {
            let spec = serde_json::from_str::<LaneSpec>(&format!("{lane},{field}}}"))
                .expect("a lane spec");
            assert!(
                LaneConfig::try_from(spec).is_err(),
                "{field} must be refused"
            );
        }
    }

    /// Every rank runs with the head's caching-allocator configuration,
    /// under whichever names the head sets, or with expandable segments when
    /// the head configures none, and with its host heap out of transparent
    /// huge pages unless the head decides otherwise. The result names each
    /// variable, since a remote launch carries only the variables its
    /// command sets.
    #[test]
    fn workers_run_with_the_heads_allocator_configuration() {
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
