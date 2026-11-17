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
use proto::admin::{PreviousEpoch, RollbackConfig};
pub use proto::config::{
    ChatConfig, ChatTemplateContentFormat, CudaGraphConfig, DeviceConfig, ImageFetchConfig,
    KvCacheConfig, LoadConfig, LoggingConfig, MediaConfig, MigrationConfig, MoeConfig,
    ProfilingConfig, ReasoningParser, SchedulerConfig, ServerConfig, SpeculativeConfig,
    VideoConfig, VideoJobConfig, WorkerConfig,
};
use proto::config::{positive, values};

use crate::api::model::assets::{HubAsset, Source};

/// The source text of a checkpoint stated as a repository or a directory, whichever is stated; a
/// directory that is not UTF-8 is refused by [`source`].
fn stated<'a>(repository: Option<&'a str>, directory: Option<&'a Path>) -> Option<&'a str> {
    repository.or_else(|| directory.and_then(Path::to_str))
}

/// The checkpoint the option `name` (a repository) or `<name>_dir` (a directory) states, of
/// which at most one is stated.
fn source(
    name: &str,
    repository: Option<&str>,
    directory: Option<&Path>,
) -> anyhow::Result<Option<Source>> {
    match (repository, directory) {
        (None, None) => Ok(None),
        (Some(repository), None) => Ok(Some(Source::Hub(HubAsset::parse(repository)?))),
        (None, Some(directory)) => {
            anyhow::ensure!(
                directory.to_str().is_some(),
                "`{name}_dir` {} is not UTF-8, which the engines' configuration states paths in",
                directory.display()
            );
            Ok(Some(Source::Directory(directory.to_path_buf())))
        }
        (Some(_), Some(_)) => {
            anyhow::bail!("`{name}` and `{name}_dir` both state one checkpoint: state one of them")
        }
    }
}

/// What an option is bound to beside every deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scope {
    /// Every deployment.
    Every,
    /// The deployments of one kind of model, by what the served family's `Serving` declares; it
    /// is refused for a model of another kind rather than ignored
    /// ([`Stated::refuse_foreign_options`]).
    ///
    /// Chat models schedule token batches, render chat prompts and call a causal language
    /// model's operators (attention, the grouped expert GEMM and the expert exchange, whose
    /// kernel backends select modules of the epoch a video pipeline would fix and never run);
    /// the video tables configure behavior only video models have.
    Kind(Kind),
    /// The rank layout shorthand, which places ranks without `worker_groups` and is refused
    /// beside them.
    Shorthand,
}

/// Classifies every field of a struct of the configuration, by its path: it expands to an
/// exhaustive pattern of the struct, so a field added to it does not compile until it is
/// classified here, and to each field's path (`prefix` and its name) with its scope.
macro_rules! classified {
    ($type:path, $prefix:literal, { $($field:ident: $scope:expr),* $(,)? }) => {{
        #[allow(dead_code, reason = "the pattern is the completeness check, never run")]
        fn exhaustive(value: &$type) {
            let $type { $($field: _),* } = value;
        }
        [$((concat!($prefix, stringify!($field)), $scope)),*]
    }};
}

impl ServerArgs {
    /// The scope of every option that is not every deployment's, by configuration path: the
    /// classification of every field of the configuration, and of the tables classified field
    /// by field.
    fn scoped_options() -> impl Iterator<Item = (&'static str, Scope)> {
        use Scope::{Every, Kind as Of, Shorthand};
        let top = classified!(ServerArgs, "", {
            model: Every,
            model_dir: Every,
            served_model_name: Every,
            max_model_len: Every,
            lora: Of(Kind::Video),
            lora_dir: Of(Kind::Video),
            dtype: Every,
            quantization: Every,
            mm_processor_kwargs: Every,
            attention_backend: Of(Kind::Chat),
            prefill_attention_backend: Of(Kind::Chat),
            decode_attention_backend: Of(Kind::Chat),
            moe_runner_backend: Of(Kind::Chat),
            build_config: Every,
            max_running_requests: Every,
            max_num_batched_tokens: Of(Kind::Chat),
            max_logprobs: Every,
            mem_fraction_static: Every,
            tensor_parallel_size: Shorthand,
            data_parallel_size: Every,
            worker_groups: Every,
            host_identity: Every,
            engine_host: Every,
            nnodes: Every,
            node_rank: Every,
            dist_init_addr: Every,
            server_config: Every,
            admin_config: Every,
            storage_config: Every,
            load_config: Every,
            kv_cache_config: Every,
            // Classified field by field below.
            scheduler_config: Every,
            speculative_config: Of(Kind::Chat),
            migration_config: Every,
            update_config: Every,
            // Classified field by field below.
            device_config: Every,
            moe_config: Of(Kind::Chat),
            cuda_graph_config: Every,
            worker_config: Every,
            profiling_config: Every,
            logging_config: Every,
            chat_config: Of(Kind::Chat),
            image_fetch_config: Of(Kind::Chat),
            video_config: Of(Kind::Video),
            media_config: Of(Kind::Video),
            video_job_config: Of(Kind::Video),
            // The command line's inputs, which are no option and are never stated.
            config_files: Every,
            set: Every,
            host: Every,
            port: Every,
            source: Every,
            ready_fd: Every,
            predecessor: Every,
        });
        let scheduler = classified!(SchedulerConfig, "scheduler_config.", {
            policy: Every,
            order: Every,
            max_queued_requests: Every,
            chunked_prefill_size: Of(Kind::Chat),
            max_batch: Every,
            policy_budget: Every,
            // Values the configuration resolves, which no statement names.
            max_num_batched_tokens: Every,
            max_running_requests: Every,
            draft_tokens: Every,
            graph_tiers: Every,
            block_tokens: Every,
            prefetch_timeout_base: Every,
            prefetch_timeout_per_ki_token: Every,
            report_period: Every,
        });
        let devices = classified!(DeviceConfig, "device_config.", {
            device: Shorthand,
            devices: Shorthand,
            worker_hosts: Shorthand,
            queue_depth: Shorthand,
        });
        top.into_iter()
            .chain(scheduler)
            .chain(devices)
            .filter(|(_, scope)| *scope != Every)
    }
}

/// A kind of model some options belong to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        for (path, scope) in ServerArgs::scoped_options() {
            if let Scope::Kind(kind) = scope
                && !kind.serves(family)
                && self.contains(path)
            {
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
    /// The Hugging Face checkpoint the deployment serves, `<owner>/<name>[@<revision>][/<folder>]`:
    /// a repository at its default branch or at a branch, tag or commit, and the folder of it that
    /// holds the checkpoint when it is not the root, read from the standard cache's snapshot of
    /// the revision; this or `model_dir` is required.
    #[arg(value_name = "MODEL")]
    pub model: Option<String>,
    /// A local checkpoint directory the deployment serves, read in place, instead of a
    /// repository (`model`).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long, value_name = "DIR")]
    pub model_dir: Option<PathBuf>,
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
    /// every request of the deployment: the Hugging Face repository, or the
    /// folder of one (`<owner>/<name>[@<revision>][/<folder>]`, as `model`),
    /// that holds the adapter in its family's declared file beside the
    /// contract of the student it is. Video families only.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long, value_name = "REPOSITORY")]
    pub lora: Option<String>,
    /// The adapter (`lora`) as a local directory holding it, read in place.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long, value_name = "DIR")]
    pub lora_dir: Option<PathBuf>,
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
    /// attention call, by its backend name; unset, the Build Server selects
    /// each attention descriptor key's package among the installed ones that
    /// cover it, by manifest priority.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(skip)]
    pub attention_backend: Option<String>,
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
    /// Tensor-parallel workers of each scheduler's group when the
    /// configuration states no `worker_groups`.
    #[arg(long, default_value_t = ServerArgs::default().tensor_parallel_size)]
    pub tensor_parallel_size: NonZeroUsize,
    /// Schedulers, each with its own scheduler, KV cache and worker
    /// groups. With `worker_groups`, its model groups form the engine
    /// cores as equal consecutive blocks.
    #[arg(long, default_value_t = ServerArgs::default().data_parallel_size)]
    pub data_parallel_size: NonZeroUsize,
    /// The deployment layout: one `[[worker_groups]]` table per group of
    /// workers, placing its ranks on devices and hosts and naming the blocks
    /// it runs. Unset, the rank-layout shorthand (`tensor_parallel_size`,
    /// `data_parallel_size`, `device_config`) places the ranks.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(skip)]
    pub worker_groups: Option<Vec<proto::config::WorkerGroupConfig>>,
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
    // shortcuts for the four nested options the flags name. None is part of the configuration's
    // tree, so a printed configuration never carries one.
    /// TOML files of serving options, each overriding the ones before it;
    /// the flags and `--set` override them. A deployment's layout file
    /// (`[[worker_groups]]`) is one such file.
    #[serde(skip)]
    #[arg(long = "config", value_name = "FILE")]
    pub config_files: Vec<PathBuf>,
    /// Set the option at PATH, such as `video_config.max_seconds=10` or
    /// `moe_config.exchange="deepep"`; repeatable, and it overrides the flags
    /// and the file. VALUE is one TOML value, as the file writes it (`2`,
    /// `true`, `"deepep"`, `[1024, 4096]`, `{ mode = "balanced" }`): text is
    /// quoted and a list is an array. One invocation may not set both a table
    /// and an option inside it.
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
    /// The UniServe source tree this host's Build Server builds from
    /// (`build_config.source`): its compiler environment and the serving
    /// packages it installs.
    #[serde(skip)]
    #[arg(long, value_name = "DIR")]
    pub source: Option<PathBuf>,
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
            model_dir: None,
            served_model_name: None,
            max_model_len: None,
            lora: None,
            lora_dir: None,
            dtype: ModelDtype::BFloat16,
            quantization: proto::config::Quantization::default(),
            mm_processor_kwargs: serde_json::Map::new(),
            attention_backend: None,
            prefill_attention_backend: None,
            decode_attention_backend: None,
            moe_runner_backend: None,
            build_config: build_server::command::Settings::default(),
            max_running_requests: positive(128),
            max_num_batched_tokens: None,
            // The most the OpenAI API admits per position (`top_logprobs` lies in 0..=20), so
            // every request of that API is served.
            max_logprobs: 20,
            mem_fraction_static: 0.7,
            tensor_parallel_size: NonZeroUsize::MIN,
            data_parallel_size: NonZeroUsize::MIN,
            worker_groups: None,
            host_identity: this_host(),
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
            worker_config: WorkerConfig::default(),
            profiling_config: ProfilingConfig::default(),
            logging_config: LoggingConfig::default(),
            chat_config: ChatConfig::default(),
            image_fetch_config: ImageFetchConfig::default(),
            video_config: VideoConfig::default(),
            media_config: MediaConfig::default(),
            video_job_config: VideoJobConfig::default(),
            config_files: Vec::new(),
            set: Vec::new(),
            host: None,
            port: None,
            source: None,
            ready_fd: None,
            predecessor: None,
        }
    }
}

/// The daemon's listeners (`admin_config`): the address its HTTP server
/// serves the client API on, and how the admin API, which two Unix sockets of
/// the daemon's instance directory serve apart from it (`admin.sock` for the
/// operator, `harness.sock` for a harness), answers. Access to the admin API
/// is those sockets' file permissions. The daemon binds every listener when it
/// starts and states their endpoints in its readiness record, which is where a
/// caller finds them.
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
    /// The option paths a harness may set through `PUT /admin/config` on the
    /// harness listener, beside the package selections it may always set.
    pub harness_options: Vec<String>,
    /// The admin API's event log (`GET /admin/events`).
    pub event_log: EventLog,
}

impl Default for AdminConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_owned(),
            port: 0,
            uds: None,
            harness_options: Vec::new(),
            event_log: EventLog::default(),
        }
    }
}

/// The admin API's event log (`admin_config.event_log`): what it keeps and how long a reader
/// waits for the next event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EventLog {
    /// The events the log keeps, the newest; an older one leaves it, and a reader that asks for
    /// it learns that events were lost (`gap`).
    pub capacity: NonZeroUsize,
    /// The longest a `GET /admin/events?wait=S` request waits for a new event; a longer `wait`
    /// waits this long. It bounds how long a long-polling reader holds a connection.
    #[serde(with = "values::duration")]
    pub longest_wait: Duration,
}

/// The default `admin_config.event_log.capacity`: the events of many updates, each of which
/// records a few phases and operations, so that a manager that reads after a pause still finds
/// the events of what it waited on.
pub const EVENT_LOG_CAPACITY: NonZeroUsize = NonZeroUsize::new(1024).unwrap();

/// The default `admin_config.event_log.longest_wait`: a minute, under the idle timeouts of
/// common HTTP proxies a reader may reach the socket through.
pub const EVENT_LOG_LONGEST_WAIT: Duration = Duration::from_secs(60);

impl Default for EventLog {
    fn default() -> Self {
        Self {
            capacity: EVENT_LOG_CAPACITY,
            longest_wait: EVENT_LOG_LONGEST_WAIT,
        }
    }
}

/// The options the daemon reads alone: its HTTP server and listeners, the request and output
/// processing (the chat options, the model's processor options, the image and condition media
/// sources), the video jobs, migration, logging, this host's Build Server, against which an
/// epoch's modules are prepared, and the updates' options, whose statistics window each
/// replica's observer is sized to when it starts. A hot restart of the daemon applies a change
/// of them; every other option is the engine configuration's or the deployment's layout and
/// hosts'. An update's own assignments to `update_config` govern that update alone and change
/// no option of the daemon's (`updates::change::resolve`).
#[derive(Debug, Clone, PartialEq)]
pub struct DaemonPart {
    pub server_config: ServerConfig,
    pub admin_config: AdminConfig,
    pub logging_config: LoggingConfig,
    pub chat_config: ChatConfig,
    pub image_fetch_config: ImageFetchConfig,
    pub media_config: MediaConfig,
    pub video_job_config: VideoJobConfig,
    pub migration_config: MigrationConfig,
    pub mm_processor_kwargs: serde_json::Map<String, Value>,
    pub build_config: build_server::command::Settings,
    pub update_config: UpdateConfig,
}

impl ServerArgs {
    /// The daemon's part of this configuration.
    ///
    /// Every option is named here, so an option added to [`ServerArgs`] does not compile until
    /// it is placed in the daemon's part or left to the rest of the configuration.
    pub fn daemon_part(&self) -> DaemonPart {
        let Self {
            // The engine configuration and the deployment's layout and hosts.
            model: _,
            model_dir: _,
            served_model_name: _,
            max_model_len: _,
            lora: _,
            lora_dir: _,
            dtype: _,
            quantization: _,
            attention_backend: _,
            prefill_attention_backend: _,
            decode_attention_backend: _,
            moe_runner_backend: _,
            max_running_requests: _,
            max_num_batched_tokens: _,
            max_logprobs: _,
            mem_fraction_static: _,
            tensor_parallel_size: _,
            data_parallel_size: _,
            worker_groups: _,
            host_identity: _,
            engine_host: _,
            nnodes: _,
            node_rank: _,
            dist_init_addr: _,
            storage_config: _,
            load_config: _,
            kv_cache_config: _,
            scheduler_config: _,
            speculative_config: _,
            device_config: _,
            moe_config: _,
            cuda_graph_config: _,
            worker_config: _,
            profiling_config: _,
            video_config: _,
            // The command line's inputs, which are no option.
            config_files: _,
            set: _,
            host: _,
            port: _,
            source: _,
            ready_fd: _,
            predecessor: _,
            // The daemon's part.
            server_config,
            admin_config,
            logging_config,
            chat_config,
            image_fetch_config,
            media_config,
            video_job_config,
            migration_config,
            mm_processor_kwargs,
            build_config,
            update_config,
        } = self;
        DaemonPart {
            server_config: server_config.clone(),
            admin_config: admin_config.clone(),
            logging_config: logging_config.clone(),
            chat_config: chat_config.clone(),
            image_fetch_config: image_fetch_config.clone(),
            media_config: media_config.clone(),
            video_job_config: video_job_config.clone(),
            migration_config: migration_config.clone(),
            mm_processor_kwargs: mm_processor_kwargs.clone(),
            build_config: build_config.clone(),
            update_config: update_config.clone(),
        }
    }

    /// This configuration with `part` as its daemon's part: what a hot restart of the daemon
    /// serves when an update changes both the daemon and what an epoch executes, before the
    /// daemon's successor changes the epoch.
    pub fn with_daemon_part(&self, part: DaemonPart) -> Self {
        let DaemonPart {
            server_config,
            admin_config,
            logging_config,
            chat_config,
            image_fetch_config,
            media_config,
            video_job_config,
            migration_config,
            mm_processor_kwargs,
            build_config,
            update_config,
        } = part;
        Self {
            server_config,
            admin_config,
            logging_config,
            chat_config,
            image_fetch_config,
            media_config,
            video_job_config,
            migration_config,
            mm_processor_kwargs,
            build_config,
            update_config,
            ..self.clone()
        }
    }
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
    /// within it. Steps are compared within their batch-size range. Like the
    /// rollback thresholds it has no default: it is measured on the target
    /// hardware and workload (`docs/configuration.md`); unset, the transfers
    /// take their links' whole budgets.
    pub transfer_slowdown: Option<f64>,
    /// The span of serving steps before a preparation that each scheduler's
    /// baseline step times are read from (`transfer_slowdown`).
    #[serde(with = "values::duration")]
    pub baseline: Duration,
    /// The fewest batches of one batch-size range that make a scheduler's
    /// baseline step time in that range, so that a range the load visited
    /// for a few steps sets none.
    pub baseline_batches: NonZeroU32,
    /// Seconds a draining engine steps its running requests beside the
    /// engine promoted before it, after which it ends the ones still running
    /// as migratable: it bounds the time two epochs share each GPU.
    pub drain_s: f64,
}

/// The default `update_config.drain_s`: the seconds a switch lets the
/// requests in flight end by themselves.
pub const DRAIN_S: f64 = 5.0;

/// The default `update_config.baseline`: a minute of a serving epoch's steps,
/// which holds every scheduler's steps of the batch sizes its load visits.
pub const BASELINE: Duration = Duration::from_secs(60);

/// The default `update_config.baseline_batches`: enough batches of one size
/// that a few steps the load passed through set no baseline.
pub const BASELINE_BATCHES: NonZeroU32 = NonZeroU32::new(20).unwrap();

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            previous_epoch: PreviousEpoch::Parked,
            rollback: RollbackConfig::default(),
            transfer_slowdown: None,
            baseline: BASELINE,
            baseline_batches: BASELINE_BATCHES,
            drain_s: DRAIN_S,
        }
    }
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
            Some(stated) => proto::config::WorkerGroupConfig::place(stated.clone(), &engine_host)
                .context("placing `worker_groups`")?,
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
                // The stated sources; the leader resolves each to the directory its engines
                // open when it reads the checkpoint (`Leader::prepare`).
                model: self.model().to_owned(),
                base: None,
                lora_path: stated(self.lora.as_deref(), self.lora_dir.as_deref())
                    .map(str::to_owned),
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

    /// The configured model as it is stated: the repository (`model`) or the
    /// directory (`model_dir`).
    ///
    /// Empty while unstated; [`ServerArgs::validate`] refuses that.
    pub fn model(&self) -> &str {
        stated(self.model.as_deref(), self.model_dir.as_deref()).unwrap_or_default()
    }

    /// Whether the configuration states a model, as a repository or a directory.
    pub fn states_model(&self) -> bool {
        self.model.is_some() || self.model_dir.is_some()
    }

    /// Where the configured model lies ([`Source`]).
    ///
    /// # Errors
    ///
    /// Fails when neither or both of `model` and `model_dir` are stated, or
    /// `model` names no repository.
    pub fn model_source(&self) -> anyhow::Result<Source> {
        source("model", self.model.as_deref(), self.model_dir.as_deref())?
            .context("the configuration states no model: name MODEL or set `model_dir`")
    }

    /// Where the configured adapter lies ([`Source`]); `None` when the
    /// configuration states none.
    ///
    /// # Errors
    ///
    /// Fails when both `lora` and `lora_dir` are stated, or `lora` names no
    /// repository.
    pub fn adapter_source(&self) -> anyhow::Result<Option<Source>> {
        source("lora", self.lora.as_deref(), self.lora_dir.as_deref())
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
    /// shorthand is refused beside `worker_groups`. Rules that need the
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
            !self.leads() || self.states_model() || self.nnodes.get() == 1,
            "`model` is required on the leader of several nodes: name MODEL on the command line \
             or set `model` in --config"
        );
        // A token bound the deployment states above its own graph catalog is
        // refused here, by the option, before any weight loads: the engine
        // admits no batch larger than its largest captured bucket, so the
        // policy's declared limits would be refused at the first prepare
        // instead (`crates/scheduling/src/limits.rs`). A stated value is never
        // clamped — the deployment either means it or is wrong.
        // A checkpoint is stated once; the resolver checks a repository's name when it resolves
        // the checkpoint (`ServerArgs::model_source`).
        for (name, repository, directory) in [
            ("model", self.model.is_some(), self.model_dir.is_some()),
            ("lora", self.lora.is_some(), self.lora_dir.is_some()),
        ] {
            anyhow::ensure!(
                !(repository && directory),
                "`{name}` and `{name}_dir` both state one checkpoint: state one of them"
            );
        }
        let graphs = &self.cuda_graph_config;
        increasing(
            "cuda_graph_config.decode_batch_sizes",
            &graphs.decode_batch_sizes,
        )?;
        increasing(
            "cuda_graph_config.prefill_token_sizes",
            &graphs.prefill_token_sizes,
        )?;
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
        if let Some(slowdown) = self.update_config.transfer_slowdown {
            anyhow::ensure!(
                slowdown.is_finite() && slowdown > 0.0,
                "invalid `update_config.transfer_slowdown`: {slowdown} must be positive"
            );
        }
        anyhow::ensure!(
            !self.update_config.baseline.is_zero(),
            "invalid `update_config.baseline`: a baseline spans some time"
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
            && let Some((option, _)) = Self::scoped_options()
                .find(|(path, scope)| *scope == Scope::Shorthand && stated.contains(path))
        {
            anyhow::bail!(
                "`{option}` places ranks without a deployment layout; the stated `worker_groups` \
                 place their own ranks"
            );
        }
        // A stated key is the token itself: an empty one would admit an empty bearer token.
        anyhow::ensure!(
            self.server_config
                .api_key
                .as_ref()
                .is_none_or(|key| !key.is_empty()),
            "`server_config.api_key` must not be empty"
        );
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
        anyhow::ensure!(
            self.dist_init_addr
                .as_ref()
                .is_none_or(|endpoint| endpoint.port().is_some()),
            "`dist_init_addr` is the leader's TCP endpoint (`host:port`)"
        );
        // The model holds the capacity to its own bounds when its processor
        // is bound; a duration that is no positive number admits nothing.
        if let Some(seconds) = self.video_config.max_seconds {
            anyhow::ensure!(
                seconds.is_finite() && seconds > 0.0,
                "`video_config.max_seconds` must be a positive number of seconds, got {seconds}"
            );
        }
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
    /// instance directory `instance`; the node's Storage Server, when it has none, starts with the
    /// deployment's [`ServerArgs::storage_settings`].
    pub fn daemon_options(&self, instance: &Path) -> DaemonOptions {
        DaemonOptions {
            nnodes: self.nnodes.get(),
            node_rank: self.node_rank,
            host: self.host_identity.clone(),
            dist_init_addr: self.dist_init_addr.clone(),
            socket: instance.join("process.sock"),
            join_timeout: crate::supervision::JOIN_TIMEOUT,
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

/// This host's name, `uname(2)`'s node name: the host identity a node states unless its
/// configuration names another, and the host its unplaced ranks run on.
fn this_host() -> String {
    rustix::system::uname()
        .nodename()
        .to_string_lossy()
        .into_owned()
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
