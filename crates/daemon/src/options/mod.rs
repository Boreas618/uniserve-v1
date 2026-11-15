//! Serving options: one typed configuration tree, read from a TOML file,
//! a few command-line flags and `--set PATH=VALUE` assignments.
//!
//! [`ServerArgs`] is the single declaration of every option, its type and
//! its default. Its top level holds the options most deployments state; the
//! rest sit in `*_config` tables, one per owner of the behavior they
//! configure (`cuda_graph_config` for graph capture, `moe_config` for expert
//! parallelism, `video_config` for video generation, and so on).
//! [`ServerArgs::resolve`] reads the tree: a `--config` file first, then the
//! flags that name a path of it, then `--set` assignments, each overriding the
//! last. The tree is the server's configuration: each part of the server reads
//! its own table, and [`ServerArgs::engine_config`] resolves what of it builds
//! the engine processes.
//!
//! Some options configure what one kind of model serves: the token batch
//! budgets, `chat_config` and `image_fetch_config` chat models, and
//! `video_config`, `media_config` and `video_job_config` video models. Startup
//! refuses such an option stated for a checkpoint of another kind rather than
//! ignoring it ([`Stated::refuse_foreign_options`]).

mod args;

use std::collections::BTreeSet;
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::supervision::DaemonOptions;
use anyhow::Context as _;
use proto::request::ModelDtype;
pub use proto::request::StandbyStorage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use processing::Family;
use processing::Serving;

pub use args::{Stated, overridden, printed, resolve_text};
pub use proto::config::{
    ChatConfig, ChatTemplateContentFormat, CudaGraphConfig, DEFAULT_FENCE_MODULES, DeviceConfig,
    FlashInferConfig, ImageFetchConfig, ImageSize, KvCacheConfig, LoadConfig, LoggingConfig,
    MediaConfig, MigrationConfig, MoeConfig, ProfilingConfig, ReasoningParser, SchedulerConfig,
    ServerConfig, SpeculativeConfig, VideoConfig, VideoJobConfig, WorkerConfig,
};
use proto::config::{positive, values};

/// Environment variable read for the API key when `server_config.api_key`
/// is unset.
const API_KEY_ENV: &str = "UNISERVE_API_KEY";

/// Options that place ranks without a deployment file, refused beside one.
const RANK_LAYOUT_SHORTHAND: [&str; 5] = [
    "tensor_parallel_size",
    "device_config.device",
    "device_config.devices",
    "device_config.worker_hosts",
    "device_config.queue_depth",
];

/// The kinds of model whose behavior some options configure, with those
/// options by configuration path.
///
/// Chat models schedule token batches and render chat prompts, which a video
/// pipeline does neither of; the video tables configure behavior only video
/// models have.
const KIND_OPTIONS: [(Kind, &[&str]); 2] = [
    (
        Kind::Chat,
        &[
            "max_num_batched_tokens",
            "scheduler_config.chunked_prefill_size",
            "speculative_config",
            "chat_config",
            "image_fetch_config",
        ],
    ),
    (
        Kind::Video,
        &[
            "video_config",
            "media_config",
            "video_job_config",
            "lora_path",
        ],
    ),
];

/// A kind of model some options belong to.
#[derive(Debug, Clone, Copy)]
enum Kind {
    Chat,
    Video,
}

impl Kind {
    /// Whether `family` is of this kind.
    fn serves(self, family: &Family) -> bool {
        match (self, family.serving) {
            (Self::Chat, Serving::Chat(_)) | (Self::Video, Serving::Video(_)) => true,
            (Self::Chat, Serving::Video(_)) | (Self::Video, Serving::Chat(_)) => false,
        }
    }

    /// The kind as the refusal names it.
    fn name(self) -> &'static str {
        match self {
            Self::Chat => "chat models",
            Self::Video => "video generation",
        }
    }
}

impl Stated {
    /// Refuses a stated option that configures a kind of model other than
    /// the served model's, rather than ignoring it.
    ///
    /// # Errors
    ///
    /// Fails, naming the first such option and the served family, when a
    /// video pipeline states a chat option, or a chat model states a video
    /// table.
    pub fn refuse_foreign_options(&self, family: &Family) -> anyhow::Result<()> {
        for (kind, paths) in KIND_OPTIONS {
            if kind.serves(family) {
                continue;
            }
            if let Some(path) = paths.iter().find(|path| self.contains(path)) {
                anyhow::bail!(
                    "`{path}` configures {}, but the served checkpoint is a {} model; remove it",
                    kind.name(),
                    family.id
                );
            }
        }
        Ok(())
    }
}

/// Every serving option.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, clap::Args)]
#[serde(default, deny_unknown_fields)]
pub struct ServerArgs {
    /// Local model directory or Hugging Face repository; required.
    #[arg(value_name = "MODEL")]
    pub model: Option<String>,
    /// The single model name the API serves; `model` when unset.
    #[arg(long)]
    pub served_model_name: Option<String>,
    /// Longest sequence a request may reach, in tokens. Unset, a checkpoint
    /// with a root `config.json` uses its `max_position_embeddings` (8192
    /// when it declares none) and a video pipeline its family's default
    /// prompt bound.
    #[arg(long)]
    pub max_model_len: Option<NonZeroU32>,
    /// A weight adapter fused into the checkpoint's weights at load, for
    /// every request of the deployment: an adapter file, or a local directory
    /// holding one. The model family must recognize it; video families only.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long, value_name = "PATH")]
    pub lora_path: Option<PathBuf>,
    /// Element type of parameters and activations.
    #[arg(skip = ServerArgs::default().dtype)]
    pub dtype: ModelDtype,
    /// Precision policy: a `mode` and representations selected by module
    /// under `modules`, which the model family interprets and refuses when
    /// it does not serve them; empty selects the checkpoint's own
    /// quantization.
    #[arg(skip)]
    pub quantization: proto::config::Quantization,
    /// Options of the model's multimodal processor; the processor refuses
    /// one it does not take.
    #[arg(skip)]
    pub mm_processor_kwargs: serde_json::Map<String, Value>,
    /// Attention package (the paged-attention operator's backend) for every
    /// attention call, by the name the ranks' attention registry gives it;
    /// `auto` selects a native kernel per call.
    #[arg(skip)]
    pub attention_backend: String,
    /// The attention package that fills prompt processing; unset takes
    /// `attention_backend` (`prefill_attention_backend`).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(skip)]
    pub prefill_attention_backend: Option<String>,
    /// The attention package that fills one decoding token; unset takes
    /// `attention_backend` (`decode_attention_backend`, §5.2).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(skip)]
    pub decode_attention_backend: Option<String>,
    /// The grouped expert GEMM package (`moe_runner_backend`, §5.1, §9);
    /// unset lets the daemon resolve it when it prepares the epoch.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(skip)]
    pub moe_runner_backend: Option<String>,
    /// This host's Build Server (`build_config`): the socket the daemon
    /// prepares an epoch's modules at (§5.1, §6.3), the Build Server's
    /// default socket when unset, and the settings it starts the host's
    /// Build Server with when none serves there. A host-layer value: the
    /// daemon reads it alone, and a change of it is a hot restart.
    #[arg(skip)]
    pub build_config: build_server::command::Settings,
    /// Concurrently resident requests, clamped to what the workers hold.
    #[arg(long, default_value_t = ServerArgs::default().max_running_requests)]
    pub max_running_requests: NonZeroUsize,
    /// Token work admitted in one scheduling step, 8192 when unset; token
    /// models only.
    #[arg(skip)]
    pub max_num_batched_tokens: Option<NonZeroUsize>,
    /// The most likely tokens a request may ask for at a generated or a
    /// prompt position (`top_logprobs`, `prompt_logprobs`), which every
    /// sampling graph computes; a request asking for more is refused. Token
    /// models only.
    #[arg(long, default_value_t = ServerArgs::default().max_logprobs)]
    pub max_logprobs: u32,
    /// Each rank's share of its device's total storage for weights, static
    /// allocations and the KV cache, in (0, 1]. A deployment group's
    /// `mem_fraction_static` overrides it.
    #[arg(long, default_value_t = ServerArgs::default().mem_fraction_static)]
    pub mem_fraction_static: f64,
    /// Tensor-parallel workers of each scheduler's group when no
    /// `worker_groups` file is given.
    #[arg(long, default_value_t = ServerArgs::default().tensor_parallel_size)]
    pub tensor_parallel_size: NonZeroUsize,
    /// Schedulers, each with its own scheduler, KV cache and worker
    /// groups. With a `worker_groups` file, its model groups form the engine
    /// cores as equal consecutive blocks.
    #[arg(long, default_value_t = ServerArgs::default().data_parallel_size)]
    pub data_parallel_size: NonZeroUsize,
    /// Deployment file placing worker groups and their blocks.
    #[arg(long, value_name = "FILE")]
    pub worker_groups: Option<PathBuf>,
    /// This node's host name; its daemon runs the ranks placed on this
    /// host, and a deployment rank entry without a host names it.
    #[arg(long, default_value_t = ServerArgs::default().host_identity)]
    pub host_identity: String,
    /// Host running the engine and ranks whose configuration omits a host.
    /// Unset uses this node's identity. The engine can move to another
    /// joined node through an epoch change; the daemon's identity stays local.
    #[arg(skip)]
    pub engine_host: Option<String>,
    /// Nodes of the deployment, each running one `uniserve serve`.
    #[arg(long, default_value_t = ServerArgs::default().nnodes)]
    pub nnodes: NonZeroU32,
    /// This node's rank among `nnodes`. The leader, rank 0, serves the API
    /// and runs the engine; every other node runs only its daemon, which
    /// joins the leader's.
    #[arg(long, default_value_t = ServerArgs::default().node_rank)]
    pub node_rank: u32,
    /// The leader's join endpoint (`host:port`): on the leader, the host its
    /// members reach it at and the port it binds on every interface (port 0
    /// picks one, which its readiness record states as `join`); on a member,
    /// the endpoint the leader's readiness record states. Required with
    /// several nodes.
    #[arg(long, value_name = "HOST:PORT")]
    pub dist_init_addr: Option<service::Endpoint>,
    #[arg(skip)]
    pub server_config: ServerConfig,
    /// The daemon's listeners (`admin_config`): the client API's address and
    /// the admin API's Unix sockets. How the HTTP server answers is
    /// `server_config`'s.
    #[arg(skip)]
    pub admin_config: AdminConfig,
    /// The settings of the Storage Server of every host the deployment
    /// serves on (`storage_config`): a daemon that starts a host's server
    /// starts it with them, and one that finds a server serving with others
    /// refuses to share it.
    #[arg(skip)]
    pub storage_config: storage::ServerSettings,
    #[arg(skip)]
    pub load_config: LoadConfig,
    #[arg(skip)]
    pub kv_cache_config: KvCacheConfig,
    #[arg(skip)]
    pub scheduler_config: SchedulerConfig,
    #[arg(skip)]
    pub speculative_config: SpeculativeConfig,
    #[arg(skip)]
    pub migration_config: MigrationConfig,
    #[arg(skip)]
    pub update_config: UpdateConfig,
    #[arg(skip)]
    pub device_config: DeviceConfig,
    #[arg(skip)]
    pub moe_config: MoeConfig,
    #[arg(skip)]
    pub cuda_graph_config: CudaGraphConfig,
    #[arg(skip)]
    pub flashinfer_config: FlashInferConfig,
    #[arg(skip)]
    pub worker_config: WorkerConfig,
    #[arg(skip)]
    pub profiling_config: ProfilingConfig,
    #[arg(skip)]
    pub logging_config: LoggingConfig,
    #[arg(skip)]
    pub chat_config: ChatConfig,
    #[arg(skip)]
    pub image_fetch_config: ImageFetchConfig,
    #[arg(skip)]
    pub video_config: VideoConfig,
    #[arg(skip)]
    pub media_config: MediaConfig,
    #[arg(skip)]
    pub video_job_config: VideoJobConfig,

    // The command line's inputs that are not options: where the options come from, and the
    // shortcuts for the three nested options the flags name. None is part of the configuration's
    // tree, so a printed configuration never carries one.
    /// TOML file of serving options; the flags and `--set` override it.
    #[serde(skip)]
    #[arg(long = "config", value_name = "FILE")]
    pub config_file: Option<PathBuf>,
    /// Set the option at PATH, such as `video_config.max_seconds=10` or
    /// `moe_config.exchange=deepep`; repeatable, and it overrides the flags
    /// and the file. VALUE is a TOML value (`2`, `true`, `[1024, 4096]`,
    /// `{ mode = "balanced" }`) or JSON, and otherwise text; a list is an
    /// array. One invocation may not set both a table and an option inside it.
    #[serde(skip)]
    #[arg(long = "set", value_name = "PATH=VALUE")]
    pub set: Vec<String>,
    /// TCP address the daemon's own server listens on (`admin_config.host`).
    #[serde(skip)]
    #[arg(long)]
    pub host: Option<String>,
    /// TCP port the daemon's own server listens on (`admin_config.port`).
    #[serde(skip)]
    #[arg(long)]
    pub port: Option<u16>,
    /// The inherited descriptor the daemon writes its readiness record to
    /// (`service::Readiness`, one JSON line per state): once its listeners
    /// are bound (`listening`, with their endpoints) and once it serves
    /// (`serving`).
    #[serde(skip)]
    #[arg(long, value_name = "FD")]
    pub ready_fd: Option<i32>,
    /// The descriptor of the handoff socket to the daemon this one succeeds,
    /// whose deployment it takes over, as `POST /admin/restart` starts its
    /// successor (`daemon::restart`); every other argument is ignored.
    #[serde(skip)]
    #[arg(long, hide = true, value_name = "FD")]
    pub predecessor: Option<i32>,
}

impl Default for ServerArgs {
    fn default() -> Self {
        Self {
            model: None,
            served_model_name: None,
            max_model_len: None,
            lora_path: None,
            dtype: ModelDtype::BFloat16,
            quantization: proto::config::Quantization::default(),
            mm_processor_kwargs: serde_json::Map::new(),
            attention_backend: "auto".to_owned(),
            prefill_attention_backend: None,
            decode_attention_backend: None,
            moe_runner_backend: None,
            build_config: build_server::command::Settings::default(),
            max_running_requests: positive(128),
            max_num_batched_tokens: None,
            // vLLM's default, which clients asking for `top_logprobs` expect to be served.
            max_logprobs: 20,
            mem_fraction_static: 0.7,
            tensor_parallel_size: NonZeroUsize::MIN,
            data_parallel_size: NonZeroUsize::MIN,
            worker_groups: None,
            host_identity: "localhost".to_owned(),
            engine_host: None,
            nnodes: NonZeroU32::MIN,
            node_rank: 0,
            dist_init_addr: None,
            server_config: ServerConfig::default(),
            admin_config: AdminConfig::default(),
            storage_config: storage::ServerSettings::default(),
            load_config: LoadConfig::default(),
            kv_cache_config: KvCacheConfig::default(),
            scheduler_config: SchedulerConfig::default(),
            speculative_config: SpeculativeConfig::default(),
            migration_config: MigrationConfig::default(),
            update_config: UpdateConfig::default(),
            device_config: DeviceConfig::default(),
            moe_config: MoeConfig::default(),
            cuda_graph_config: CudaGraphConfig::default(),
            flashinfer_config: FlashInferConfig::default(),
            worker_config: WorkerConfig::default(),
            profiling_config: ProfilingConfig::default(),
            logging_config: LoggingConfig::default(),
            chat_config: ChatConfig::default(),
            image_fetch_config: ImageFetchConfig::default(),
            video_config: VideoConfig::default(),
            media_config: MediaConfig::default(),
            video_job_config: VideoJobConfig::default(),
            config_file: None,
            set: Vec::new(),
            host: None,
            port: None,
            ready_fd: None,
            predecessor: None,
        }
    }
}

/// The daemon's listeners (`admin_config`): the address its HTTP server
/// serves the client API on, and the two Unix sockets that serve the admin
/// API apart from it, one for the operator and one for a harness. Access to
/// the admin API is those sockets' file permissions. The daemon binds every
/// listener when it starts and states their endpoints in its readiness
/// record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AdminConfig {
    /// TCP address the server listens on.
    pub host: String,
    /// TCP port the server listens on; 0, the default, picks a free one.
    pub port: u16,
    /// Unix domain socket to listen on instead of TCP; a stale socket file
    /// is replaced and the socket file is removed at shutdown.
    pub uds: Option<String>,
    /// The Unix socket the operator's admin API listens on, created with
    /// mode `0600`; unset places it in the daemon's instance directory
    /// (`admin.sock`).
    pub admin_uds: Option<String>,
    /// The Unix socket the harness's subset of the admin API listens on,
    /// created with mode `0660`, so that the operator grants a harness
    /// access through the socket's group; unset places it in the daemon's
    /// instance directory (`harness.sock`).
    pub harness_uds: Option<String>,
    /// The option paths a harness may set through `PUT /admin/config` on the
    /// harness listener, beside the package selections it may always set.
    pub harness_options: Vec<String>,
}

impl AdminConfig {
    /// The Unix socket the operator's admin API listens on, in `instance` unless stated.
    pub fn admin_socket(&self, instance: &Path) -> PathBuf {
        self.admin_uds
            .as_ref()
            .map_or_else(|| instance.join("admin.sock"), PathBuf::from)
    }

    /// The Unix socket the harness's subset of the admin API listens on, in `instance` unless
    /// stated.
    pub fn harness_socket(&self, instance: &Path) -> PathBuf {
        self.harness_uds
            .as_ref()
            .map_or_else(|| instance.join("harness.sock"), PathBuf::from)
    }
}

impl Default for AdminConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_owned(),
            port: 0,
            uds: None,
            admin_uds: None,
            harness_uds: None,
            harness_options: Vec::new(),
        }
    }
}

/// `current` with `target`'s daemon options (those `daemon_options_differ`
/// compares): the configuration a hot restart of the daemon serves when an
/// update changes both the daemon and what an epoch executes, before the
/// daemon's successor changes the epoch.
pub fn with_daemon_options(current: &ServerArgs, target: &ServerArgs) -> ServerArgs {
    ServerArgs {
        server_config: target.server_config.clone(),
        admin_config: target.admin_config.clone(),
        logging_config: target.logging_config.clone(),
        chat_config: target.chat_config.clone(),
        image_fetch_config: target.image_fetch_config.clone(),
        media_config: target.media_config.clone(),
        video_job_config: target.video_job_config.clone(),
        migration_config: target.migration_config.clone(),
        mm_processor_kwargs: target.mm_processor_kwargs.clone(),
        ..current.clone()
    }
}

/// Whether `current` and `target` differ in the options the daemon reads
/// alone (its HTTP server, the request and output processing, the media
/// and video jobs, logging and migration), which a hot restart of the
/// daemon applies; the engine configuration covers every other option.
pub fn daemon_options_differ(current: &ServerArgs, target: &ServerArgs) -> bool {
    current.server_config != target.server_config
        || current.admin_config != target.admin_config
        || current.logging_config != target.logging_config
        || current.chat_config != target.chat_config
        || current.image_fetch_config != target.image_fetch_config
        || current.media_config != target.media_config
        || current.video_job_config != target.video_job_config
        || current.migration_config != target.migration_config
        || current.mm_processor_kwargs != target.mm_processor_kwargs
        // This host's Build Server: the daemon reads it alone, and an
        // epoch's modules are prepared against it (§5.1).
        || current.build_config != target.build_config
}

/// How the deployment moves to a new configuration in another epoch
/// (`update_config`, `PUT /admin/config`): what of the previous epoch its
/// trial keeps for a rollback, when the trial rolls back by itself, how the
/// transfers yield to the serving epoch, and how long the switch lets the
/// requests in flight run on. A value an update states in its own assignments
/// governs that update alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UpdateConfig {
    /// What the previous epoch keeps while the new one is on trial.
    pub previous_epoch: PreviousEpoch,
    /// When a trial rolls back without the operator: a fatal worker error,
    /// the engine process's exit, a stalled job and non-finite logits always
    /// roll back; these thresholds, which have no default, add the rest.
    pub rollback: RollbackConfig,
    /// How much longer than before the change a serving scheduler's step
    /// may take while the successor prepares, as a fraction (`0.1` for
    /// 10%), past which the leader halves the byte budgets of the links the
    /// change's transfers use, and restores them once the steps are back
    /// within it. Steps are compared within their batch-size range. The
    /// default is the value measured on GB200 (`docs/configuration.md`).
    pub transfer_slowdown: f64,
    /// Seconds a draining engine steps its running requests beside the
    /// engine promoted before it, after which it ends the ones still running
    /// as migratable: it bounds the time two epochs share each GPU.
    pub drain_s: f64,
}

/// The default `update_config.drain_s`: the seconds a switch lets the
/// requests in flight end by themselves.
pub const DRAIN_S: f64 = 5.0;

/// The default `update_config.transfer_slowdown`: on GB200, above the
/// serving steps' second-to-second variation without transfers and below
/// the slowdowns copies at a link's whole budget cause
/// (`docs/configuration.md`).
pub const TRANSFER_SLOWDOWN: f64 = 0.1;

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            previous_epoch: PreviousEpoch::Parked,
            rollback: RollbackConfig::default(),
            transfer_slowdown: TRANSFER_SLOWDOWN,
            drain_s: DRAIN_S,
        }
    }
}

/// What the previous epoch keeps during its successor's trial
/// (`update_config.previous_epoch`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreviousEpoch {
    /// Its processes, communicators, graph storage, weights and the KV cache
    /// its successor maps: a rollback resumes it at once.
    Parked,
    /// Its weights (and the KV cache its successor maps): a rollback starts
    /// its processes again, which bind them.
    Weights,
    /// The KV cache its successor maps, and in host storage the weights its
    /// successor does not map: a rollback copies them back and starts its
    /// processes again.
    Host,
}

/// The thresholds past which a trial rolls back by itself
/// (`update_config.rollback`). None has a default: each is measured on the
/// target workload before it is set.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RollbackConfig {
    /// The share of the requests that end during the trial, in `(0, 1]`,
    /// past which it rolls back when they failed.
    pub failure_rate: Option<f64>,
    /// How much longer than the previous epoch's before its promotion the
    /// new epoch's average step may take, as a fraction (`0.2` for 20%).
    pub step_time_regression: Option<f64>,
    /// How much lower than the previous epoch's before its promotion the
    /// new epoch's token throughput may fall, as a fraction.
    pub throughput_regression: Option<f64>,
    /// The span the step time and throughput are averaged over, before the
    /// promotion and during the trial; required with either regression.
    #[serde(with = "values::optional_duration")]
    pub window: Option<Duration>,
}

impl ServerArgs {
    /// The configuration this deployment's epoch engines are built from.
    ///
    /// The executable's identity is this crate's [`proto::BUILD`]: today the
    /// daemon and the engine process are one binary, so a deployment cannot
    /// run them at different builds. A daemon that starts an engine of
    /// another build compares that executable's identity instead (§7.4).
    pub fn engine_config(&self) -> anyhow::Result<proto::config::EngineConfig> {
        let engine_host = self
            .engine_host
            .clone()
            .unwrap_or_else(|| self.host_identity.clone());
        let worker_groups = match &self.worker_groups {
            Some(path) => {
                let text = std::fs::read_to_string(path)
                    .with_context(|| format!("reading the deployment file {}", path.display()))?;
                proto::config::WorkerGroupConfig::parse_file(&text, &engine_host)
                    .with_context(|| format!("placing the deployment file {}", path.display()))?
            }
            None => proto::config::WorkerGroupConfig::shorthand(
                &engine_host,
                &self.device_config,
                self.tensor_parallel_size.get(),
                self.data_parallel_size.get(),
            )?,
        };
        Ok(proto::config::EngineConfig {
            executable: proto::BUILD.to_owned(),
            model: proto::config::ModelConfig {
                model: self.model().to_owned(),
                base: None,
                lora_path: self
                    .lora_path
                    .as_ref()
                    .map(|path| path.to_string_lossy().into_owned()),
                max_model_len: self.max_model_len.map_or(0, NonZeroU32::get),
                dtype: self.dtype,
                quantization: self.quantization.clone(),
                generation_limits: Default::default(),
                bos: 0,
                eos: Vec::new(),
            },
            parallel: proto::config::ParallelConfig {
                tensor_parallel_size: self.tensor_parallel_size,
                data_parallel_size: self.data_parallel_size,
                engine_host,
                worker_groups,
            },
            deployment: self.served_model_name().to_owned(),
            max_running_requests: self.max_running_requests,
            max_num_batched_tokens: self.max_num_batched_tokens,
            max_logprobs: self.max_logprobs,
            attention_backend: self.attention_backend.clone(),
            prefill_attention_backend: self.prefill_attention_backend.clone(),
            decode_attention_backend: self.decode_attention_backend.clone(),
            moe_runner_backend: self.moe_runner_backend.clone(),
            device: self.device_config.clone(),
            scheduler: self.scheduler_config.clone(),
            media: self.media_config.clone(),
            mem_fraction_static: self.mem_fraction_static,
            load: self.load_config.clone(),
            kv_cache: self.kv_cache_config.clone(),
            moe: self.moe_config.clone(),
            cuda_graph: self.cuda_graph_config.clone(),
            worker: self.worker_config.clone(),
            speculative: self.speculative_config.clone(),
            video: self.video_config.clone(),
            profiling: self.profiling_config.clone(),
        })
    }

    /// The configured model: a local directory or a Hugging Face repository.
    ///
    /// Empty while unstated; [`ServerArgs::validate`] refuses that.
    pub fn model(&self) -> &str {
        self.model.as_deref().unwrap_or_default()
    }

    /// The single model name the API serves: `served_model_name`, else the
    /// model.
    pub fn served_model_name(&self) -> &str {
        self.served_model_name
            .as_deref()
            .unwrap_or_else(|| self.model())
    }

    /// Checks the rules the option types cannot express on their own.
    ///
    /// `stated` records the paths the operator set, so the rank layout
    /// shorthand is refused beside a deployment file. Rules that need the
    /// served model's family run at startup instead
    /// ([`Stated::refuse_foreign_options`]).
    ///
    /// # Errors
    ///
    /// Fails when `model` is unset, `mem_fraction_static` lies outside
    /// (0, 1], a rank layout-shorthand option is stated with `worker_groups`, the
    /// server host or Unix socket path is blank, or
    /// `video_config.max_seconds` lies outside the durations the video API
    /// admits.
    pub fn validate(&self, stated: &Stated) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.node_rank < self.nnodes.get(),
            "`node_rank` {} lies outside the {} nodes `nnodes` states",
            self.node_rank,
            self.nnodes
        );
        anyhow::ensure!(
            self.nnodes.get() == 1 || self.dist_init_addr.is_some(),
            "a deployment of several nodes needs the leader's address: set `dist_init_addr`"
        );
        anyhow::ensure!(
            !self.leads() || self.model.is_some() || self.nnodes.get() == 1,
            "`model` is required on the leader of several nodes: name MODEL on the command line \
             or set `model` in --config"
        );
        // A token bound the deployment states above its own graph catalog is
        // refused here, by the option, before any weight loads: the engine
        // admits no batch larger than its largest captured bucket, so the
        // policy's declared limits would be refused at the first prepare
        // instead (`crates/scheduling/src/limits.rs`). A stated value is never
        // clamped — the deployment either means it or is wrong.
        let graphs = &self.cuda_graph_config;
        increasing(
            "cuda_graph_config.decode_batch_sizes",
            &graphs.decode_batch_sizes,
        )?;
        increasing(
            "cuda_graph_config.prefill_token_sizes",
            &graphs.prefill_token_sizes,
        )?;
        distinct("cuda_graph_config.flow_shapes", &graphs.flow_shapes)?;
        increasing(
            "video_config.text_capacities",
            &self.video_config.text_capacities,
        )?;
        let tiers = self.cuda_graph_config.tiers();
        let mut token_bounds = Vec::new();
        if let Some(budget) = self.max_num_batched_tokens {
            token_bounds.push((
                "max_num_batched_tokens",
                u32::try_from(budget.get()).unwrap_or(u32::MAX),
            ));
        }
        if let Some(chunk) = self.scheduler_config.chunked_prefill_size {
            token_bounds.push((
                "scheduler_config.chunked_prefill_size",
                u32::try_from(chunk.get()).unwrap_or(u32::MAX),
            ));
        }
        if let Some(ceiling) = tiers.tokens {
            for (option, value) in token_bounds {
                anyhow::ensure!(
                    value <= ceiling,
                    "`{option}` is {value}, above the {ceiling} tokens the deployment's captured \
                     graphs bound a batch to: the engine admits no batch larger than its largest \
                     bucket, so the scheduling policy's limits would be refused at the first \
                     prepare. State a bound at or below {ceiling}, or capture a larger bucket in \
                     `cuda_graph_config`"
                );
            }
        }
        // A batch of requests is bounded by the largest *decode* graph, which is
        // a row count: a prefill graph's token count says nothing about how many
        // requests a decode step may hold.
        if let Some(ceiling) = tiers.rows {
            let batch = u32::try_from(self.scheduler_config.max_batch.get()).unwrap_or(u32::MAX);
            // A batch the deployment leaves unstated follows the catalog
            // (`policy_arguments` takes the smaller of the two), so only a
            // stated batch can be outside it.
            anyhow::ensure!(
                batch <= ceiling || !stated.contains("scheduler_config.max_batch"),
                "`scheduler_config.max_batch` is {batch}, above the {ceiling} requests the \
                 deployment's captured decode graphs bound a batch to: the engine admits no batch \
                 wider than its largest decode bucket, so the scheduling policy's limits would be \
                 refused at the first prepare. State a batch at or below {ceiling}, or capture a \
                 larger decode bucket in `cuda_graph_config`"
            );
        }
        let speculation = &self.speculative_config;
        if let Some(speedup) = speculation.expected_speedup {
            anyhow::ensure!(
                speedup.is_finite() && speedup >= 1.0,
                "invalid `speculative_config.expected_speedup`: {speedup} must be at least 1"
            );
        }
        if let Some(cost) = speculation.refill_token_cost {
            anyhow::ensure!(
                cost.is_finite() && cost >= 0.0,
                "invalid `speculative_config.refill_token_cost`: {cost} must not be negative"
            );
        }
        let slowdown = self.update_config.transfer_slowdown;
        anyhow::ensure!(
            slowdown.is_finite() && slowdown > 0.0,
            "invalid `update_config.transfer_slowdown`: {slowdown} must be positive"
        );
        let drain = self.update_config.drain_s;
        anyhow::ensure!(
            drain.is_finite() && drain >= 0.0,
            "invalid `update_config.drain_s`: {drain} must be a number of seconds, zero or more"
        );
        let rollback = &self.update_config.rollback;
        if let Some(rate) = rollback.failure_rate {
            anyhow::ensure!(
                rate.is_finite() && rate > 0.0 && rate <= 1.0,
                "invalid `update_config.rollback.failure_rate`: {rate} must be in (0, 1]"
            );
        }
        for (name, ratio) in [
            ("step_time_regression", rollback.step_time_regression),
            ("throughput_regression", rollback.throughput_regression),
        ] {
            if let Some(ratio) = ratio {
                anyhow::ensure!(
                    ratio.is_finite() && ratio > 0.0,
                    "invalid `update_config.rollback.{name}`: {ratio} must be positive"
                );
                anyhow::ensure!(
                    rollback.window.is_some_and(|window| !window.is_zero()),
                    "`update_config.rollback.{name}` is measured over \
                     `update_config.rollback.window`: set it"
                );
            }
        }
        // The shorthand puts its ranks, in rank order, on the CUDA devices
        // the order names (the engine refuses two ranks of a host on one).
        let devices = &self.device_config.devices;
        if !devices.is_empty() {
            let ranks = self.tensor_parallel_size.get() * self.data_parallel_size.get();
            anyhow::ensure!(
                self.device_config.device == proto::config::DeviceKind::Cuda,
                "`device_config.devices` orders CUDA devices: `device_config.device` is `{}`",
                self.device_config.device
            );
            anyhow::ensure!(
                devices.len() == ranks,
                "`device_config.devices` names {} devices for the {ranks} ranks of \
                 `tensor_parallel_size` × `data_parallel_size`",
                devices.len()
            );
        }
        // Each rank grants itself this share of every device's total
        // storage, so any finite value in (0, 1] is meaningful.
        anyhow::ensure!(
            self.mem_fraction_static.is_finite()
                && self.mem_fraction_static > 0.0
                && self.mem_fraction_static <= 1.0,
            "invalid `mem_fraction_static`: {} must be finite and in (0, 1]",
            self.mem_fraction_static
        );
        if self.worker_groups.is_some()
            && let Some(option) = RANK_LAYOUT_SHORTHAND
                .iter()
                .find(|option| stated.contains(option))
        {
            anyhow::bail!(
                "`{option}` places ranks without a deployment file; the `worker_groups` file \
                 states its own rank layout"
            );
        }
        // The daemon binds `admin_config`'s address, so an empty one fails
        // here rather than at the bind.
        let admin = &self.admin_config;
        match &admin.uds {
            Some(path) => anyhow::ensure!(
                !path.trim().is_empty(),
                "`admin_config.uds` must not be empty"
            ),
            None => anyhow::ensure!(
                !admin.host.trim().is_empty(),
                "`admin_config.host` must not be empty"
            ),
        }
        for (name, path) in [
            ("admin_uds", &admin.admin_uds),
            ("harness_uds", &admin.harness_uds),
        ] {
            anyhow::ensure!(
                path.as_ref().is_none_or(|path| !path.trim().is_empty()),
                "`admin_config.{name}` must not be empty"
            );
        }
        anyhow::ensure!(
            admin.admin_uds.is_none() || admin.admin_uds != admin.harness_uds,
            "`admin_config.admin_uds` and `admin_config.harness_uds` name one socket"
        );
        anyhow::ensure!(
            self.dist_init_addr
                .as_ref()
                .is_none_or(|endpoint| endpoint.port().is_some()),
            "`dist_init_addr` is the leader's TCP endpoint (`host:port`)"
        );
        // The model holds the capacity to its own bounds when its processor
        // is bound; a duration that is no positive number admits nothing.
        anyhow::ensure!(
            self.video_config.max_seconds.is_finite() && self.video_config.max_seconds > 0.0,
            "`video_config.max_seconds` must be a positive number of seconds, got {}",
            self.video_config.max_seconds
        );
        Ok(())
    }

    /// Whether this node is the leader, which serves the API and runs the
    /// engine.
    pub fn leads(&self) -> bool {
        self.node_rank == 0
    }

    /// The leader's request socket, which its engine processes connect back
    /// to: this process's own, in the daemon's instance directory `instance`. A hot restart's
    /// successor serves in its predecessor's instance beside it, so each daemon process names
    /// its own.
    pub fn request_socket(&self, instance: &Path) -> PathBuf {
        instance.join(format!("requests.{}.sock", std::process::id()))
    }

    /// The options of this node's daemon. The leader's process socket lies in the daemon's
    /// instance directory `instance`; `executable` starts the node's Storage Server when it has
    /// none, with the deployment's [`ServerArgs::storage_settings`].
    pub fn daemon_options(
        &self,
        devices: crate::supervision::DeviceProbe,
        instance: &Path,
    ) -> DaemonOptions {
        DaemonOptions {
            nnodes: self.nnodes.get(),
            node_rank: self.node_rank,
            host: self.host_identity.clone(),
            dist_init_addr: self.dist_init_addr.clone(),
            socket: instance.join("process.sock"),
            // The host services are commands of this executable.
            executable: std::env::current_exe().unwrap_or_else(|_| "uniserve".into()),
            join_timeout: crate::supervision::JOIN_TIMEOUT,
            devices,
            build_server: self.build_config.clone(),
            storage: self.storage_settings(),
        }
    }

    /// The settings of the Storage Server of every host the deployment serves on: its
    /// `storage_config`, with the Mooncake connector `kv_cache_config.mooncake` states. A
    /// deployment that other nodes may join (`dist_init_addr`) has its servers accept other
    /// hosts on every interface, on a port each picks, unless `storage_config.tcp` states an
    /// endpoint: the other nodes' Transfer Servers read their storage.
    pub fn storage_settings(&self) -> storage::Settings {
        let mut server = self.storage_config.clone();
        if server.tcp.is_none() && self.dist_init_addr.is_some() {
            server.tcp = Some(service::Endpoint::Tcp {
                host: std::net::Ipv4Addr::UNSPECIFIED.to_string(),
                port: 0,
            });
        }
        storage::Settings {
            server,
            connector: self.kv_cache_config.mooncake.clone(),
        }
    }
}

/// Returns the bearer token the server requires, if any.
///
/// [`ServeArgs::resolve`] stores it in `server_config.api_key`.
///
/// The configured key takes precedence over the `UNISERVE_API_KEY`
/// environment variable. Each source is trimmed, and a blank value counts as
/// unset, so a blank configured key falls through to the environment
/// variable.
pub fn configured_api_key(configured: Option<&str>) -> Option<String> {
    non_empty_secret(configured).or_else(|| {
        std::env::var(API_KEY_ENV)
            .ok()
            .and_then(|value| non_empty_secret(Some(&value)))
    })
}

/// Returns the secret with surrounding whitespace trimmed, or `None` when it
/// is absent or blank.
fn non_empty_secret(value: Option<&str>) -> Option<String> {
    let trimmed = value?.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Checks that a stated capture or capacity list is strictly increasing: the workers capture or
/// size each entry once, in order.
fn increasing(path: &str, values: &Option<Vec<NonZeroU32>>) -> anyhow::Result<()> {
    if let Some(values) = values {
        anyhow::ensure!(
            values.windows(2).all(|pair| pair[0] < pair[1]),
            "`{path}` must be strictly increasing, got {values:?}"
        );
    }
    Ok(())
}

/// Checks that stated image capture sizes are distinct: the workers capture each size once.
fn distinct(path: &str, shapes: &Option<Vec<ImageSize>>) -> anyhow::Result<()> {
    if let Some(shapes) = shapes {
        anyhow::ensure!(
            shapes.iter().collect::<BTreeSet<_>>().len() == shapes.len(),
            "`{path}` repeats an image size"
        );
    }
    Ok(())
}
