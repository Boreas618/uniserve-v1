//! The deployment's configuration: one typed tree, the `config.toml` of a policy package.
//!
//! [`Config`] is the single declaration of every option, its type, its default and how a change
//! of it takes effect (`#[reload(...)]`, [`proto::reload`]). Its top level holds the options
//! most deployments state; the rest sit in `*_config` tables, one per owner of the behavior they
//! configure (`cuda_graph_config` for graph capture, `moe_config` for expert parallelism,
//! `video_config` for video generation, and so on). [`parse`] reads a package's `config.toml`;
//! each part of the server reads its own table, and [`Config::engine_config`] resolves what of
//! it builds the engine processes. No option comes from the command line, and none states which
//! node serves it: a node finds its place in `hosts` by its own host name ([`Identity`]).
//!
//! Some options configure what one kind of model serves: the token batch budgets,
//! `chat_config` and `image_fetch_config` chat models, and `video_config`, `media_config` and
//! `video_job_config` video models. A start refuses such an option stated for a checkpoint of
//! another kind rather than ignoring it ([`Stated::refuse_foreign_options`]).

mod args;
pub mod program;

use std::num::{NonZeroU32, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::supervision::DaemonOptions;
use anyhow::Context as _;
use proto::reload::Reload;
use proto::request::ModelDtype;
pub use proto::request::StandbyStorage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use processing::Family;
use processing::Serving;

pub use args::{Stated, parse, printed};
use proto::admin::{InFlight, PreviousEpoch, RollbackConfig};
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

impl Config {
    /// The scope of every option that is not every deployment's, by configuration path: the
    /// classification of every field of the configuration, and of the tables classified field
    /// by field.
    fn scoped_options() -> impl Iterator<Item = (&'static str, Scope)> {
        use Scope::{Every, Kind as Of, Shorthand};
        let top = classified!(Config, "", {
            model: Every,
            model_dir: Every,
            served_model_name: Every,
            max_model_len: Every,
            lora: Of(Kind::Video),
            lora_dir: Of(Kind::Video),
            dtype: Every,
            quantization: Every,
            mm_processor_kwargs: Every,
            kernels: Every,
            build_config: Every,
            max_running_requests: Every,
            max_num_batched_tokens: Of(Kind::Chat),
            max_logprobs: Every,
            mem_fraction_static: Every,
            tensor_parallel_size: Shorthand,
            data_parallel_size: Every,
            worker_groups: Every,
            hosts: Every,
            engine_host: Every,
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
            kv_events_config: Of(Kind::Chat),
            logging_config: Every,
            export_config: Every,
            dynamo_config: Of(Kind::Chat),
            replicas: Of(Kind::Chat),
            chat_config: Of(Kind::Chat),
            image_fetch_config: Of(Kind::Chat),
            video_config: Of(Kind::Video),
            media_config: Of(Kind::Video),
            video_job_config: Of(Kind::Video),
            programs: Every,
            // A video request is computed where it arrives: its replicas are mixed.
            paused: Every,
            asleep: Every,
            flush: Every,
            retire: Every,
            adapters: Of(Kind::Chat),
        });
        let scheduler = classified!(SchedulerConfig, "scheduler_config.", {
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
            roles: Every,
            lease: Every,
            storage: Every,
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
        for (path, scope) in Config::scoped_options() {
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Reload)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// The Hugging Face checkpoint the deployment serves, `<owner>/<name>[@<revision>][/<folder>]`:
    /// a repository at its default branch or at a branch, tag or commit, and the folder of it that
    /// holds the checkpoint when it is not the root, read from the standard cache's snapshot of
    /// the revision; this or `model_dir` is required.
    #[reload(epoch)]
    pub model: Option<String>,
    /// A local checkpoint directory the deployment serves, read in place, instead of a
    /// repository (`model`).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[reload(epoch)]
    pub model_dir: Option<PathBuf>,
    /// The single model name the API serves; `model` when unset. A change renames the
    /// deployment's API from the next request on: a request naming the former name is refused,
    /// and the requests in flight under it end as they would have.
    #[reload(step)]
    pub served_model_name: Option<String>,
    /// Longest sequence a request may reach, in tokens. Unset, a checkpoint
    /// with a root `config.json` uses its `max_position_embeddings` (8192
    /// when it declares none) and a video pipeline its family's default
    /// prompt bound.
    #[reload(epoch)]
    pub max_model_len: Option<NonZeroU32>,
    /// A weight adapter fused into the checkpoint's weights at load, for
    /// every request of the deployment: the Hugging Face repository, or the
    /// folder of one (`<owner>/<name>[@<revision>][/<folder>]`, as `model`),
    /// that holds the adapter in its family's declared file beside the
    /// contract of the student it is. Video families only.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[reload(epoch)]
    pub lora: Option<String>,
    /// The adapter (`lora`) as a local directory holding it, read in place.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[reload(epoch)]
    pub lora_dir: Option<PathBuf>,
    /// Element type of parameters and activations.
    #[reload(epoch)]
    pub dtype: ModelDtype,
    /// Precision policy: a `mode` and representations selected by module
    /// under `modules`, which the model family interprets and refuses when
    /// it does not serve them; empty selects the checkpoint's own
    /// quantization.
    #[reload(epoch)]
    pub quantization: proto::config::Quantization,
    /// Options of the model's multimodal processor; the processor refuses
    /// one it does not take.
    #[reload(daemon)]
    pub mm_processor_kwargs: serde_json::Map<String, Value>,
    /// The kernel each descriptor key is pinned to (`[kernels]`): for each operator, by its name
    /// as a table, its descriptor keys in the canonical text the kernel observations label them
    /// by (`op`, `descriptor`), each pinned to `backend@revision` or `backend@latest`, which the
    /// apply fixes to the latest installed revision. A key the table does not list is selected
    /// by the Build Server among the installed packages that cover it, by manifest priority. A
    /// change is a kernel generation: the serving engines capture the graphs of the buckets
    /// whose kernels change in the background and switch bucket by bucket.
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    #[reload(generation)]
    pub kernels: proto::module::Kernels,
    /// The deployment's Build Server on each node (`build_config`): the settings the node's
    /// daemon starts it with, which prepares the epochs' modules (§5.1, §6.3) in the
    /// deployment's own directories. The daemon reads it alone: a change of it is a hot restart,
    /// whose successor replaces a Build Server that serves with other settings.
    #[reload(daemon)]
    pub build_config: build_server::command::Settings,
    /// Concurrently resident requests, clamped to what the workers hold.
    #[reload(epoch)]
    pub max_running_requests: NonZeroUsize,
    /// Token work admitted in one scheduling step, 8192 when unset; token
    /// models only.
    #[reload(step)]
    pub max_num_batched_tokens: Option<NonZeroUsize>,
    /// The most likely tokens a request may ask for at a generated or a
    /// prompt position (`top_logprobs`, `prompt_logprobs`), which every
    /// sampling graph computes; a request asking for more is refused. Token
    /// models only. A change is a kernel generation: the serving engines
    /// capture every graph again for the new candidates in the background and
    /// switch to them whole.
    #[reload(generation)]
    pub max_logprobs: u32,
    /// Each rank's share of its device's total storage for weights, static
    /// allocations and the KV cache, in (0, 1]. A deployment group's
    /// `mem_fraction_static` overrides it. Lowered, it takes effect in the serving engines: the
    /// KV cache gives back its tail, the blocks it evicts demoted to the host tier first, and
    /// every rank reserves under the lower share; raised, it takes a new epoch, whose ranks
    /// divide the larger share ([`fraction_effect`]).
    #[reload(by = fraction_effect)]
    pub mem_fraction_static: f64,
    /// Tensor-parallel workers of each scheduler's group when the
    /// configuration states no `worker_groups`.
    #[reload(epoch)]
    pub tensor_parallel_size: NonZeroUsize,
    /// Schedulers, each with its own scheduler, KV cache and worker
    /// groups. With `worker_groups`, its model groups form the engine
    /// cores as equal consecutive blocks.
    #[reload(epoch)]
    pub data_parallel_size: NonZeroUsize,
    /// The deployment layout: one `[[worker_groups]]` table per group of
    /// workers, placing its ranks on devices and hosts and naming the blocks
    /// it runs. Unset, the rank-layout shorthand (`tensor_parallel_size`,
    /// `data_parallel_size`, `device_config`) places the ranks. A change of the groups' queue
    /// depths alone takes effect at the next step; any other is a new epoch ([`groups_effect`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[reload(by = groups_effect)]
    pub worker_groups: Option<Vec<proto::config::WorkerGroupConfig>>,
    /// The deployment's nodes, by host name, the leader first: each node runs `uniserve serve`
    /// with the same package, finds itself in the list by its own host name (`uname -n`), and
    /// joins the leader at `admin_config.join` of the first host. Empty, the deployment is one
    /// node on whichever host serves it.
    #[reload(epoch)]
    pub hosts: Vec<String>,
    /// Host running the engine and ranks whose configuration omits a host.
    /// Unset uses the leader's host. The engine can move to another
    /// node of `hosts` through an epoch change; the daemon stays on its node.
    #[reload(epoch)]
    pub engine_host: Option<String>,
    #[reload(daemon)]
    pub server_config: ServerConfig,
    /// The daemon's listeners (`admin_config`): the client API's address and
    /// the admin API's Unix sockets. How the HTTP server answers is
    /// `server_config`'s.
    #[reload(daemon)]
    pub admin_config: AdminConfig,
    /// The settings of the deployment's Storage Server on every node it serves on
    /// (`storage_config`). A change of the settings a running server reads when it acts takes
    /// effect at the next step, in the running servers; one of the settings fixed when a server's
    /// processes start replaces every node's server, and the engines follow as an epoch
    /// ([`storage_effect`]).
    #[reload(by = storage_effect)]
    pub storage_config: storage::ServerSettings,
    #[reload(epoch)]
    pub load_config: LoadConfig,
    #[reload(fields)]
    pub kv_cache_config: KvCacheConfig,
    #[reload(fields)]
    pub scheduler_config: SchedulerConfig,
    #[reload(fields)]
    pub speculative_config: SpeculativeConfig,
    #[reload(daemon)]
    pub migration_config: MigrationConfig,
    #[reload(daemon)]
    pub update_config: UpdateConfig,
    #[reload(fields)]
    pub device_config: DeviceConfig,
    #[reload(fields)]
    pub moe_config: MoeConfig,
    #[reload(generation)]
    pub cuda_graph_config: CudaGraphConfig,
    #[reload(fields)]
    pub worker_config: WorkerConfig,
    #[reload(fields)]
    pub profiling_config: ProfilingConfig,
    /// The KV events the schedulers publish in SGLang's ZMQ format (`kv_events_config`).
    #[reload(fields)]
    pub kv_events_config: proto::config::KvEventsConfig,
    #[reload(daemon)]
    pub logging_config: LoggingConfig,
    #[reload(daemon)]
    pub export_config: ExportConfig,
    /// The Dynamo sidecar of the leader daemon (`dynamo_config`).
    #[reload(daemon)]
    pub dynamo_config: DynamoConfig,
    #[reload(daemon)]
    pub chat_config: ChatConfig,
    #[reload(daemon)]
    pub image_fetch_config: ImageFetchConfig,
    #[reload(epoch)]
    pub video_config: VideoConfig,
    #[reload(daemon)]
    pub media_config: MediaConfig,
    #[reload(daemon)]
    pub video_job_config: VideoJobConfig,
    /// The package's programs, by name (`[programs.<name>]`): each one's source, period and
    /// arguments. A change replaces the instance of each program whose entry or module changed
    /// between two of its calls; the others keep theirs. A package that lists no scheduling
    /// program runs `uniserve:arrival`.
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    #[reload(program)]
    pub programs: std::collections::BTreeMap<String, program::Program>,
    /// The deployment's replicas by id (`[replicas.<id>]`), each one scheduler with its own KV
    /// cache, numbered `0` up to `data_parallel_size` in the order of the worker groups. Every
    /// replica is built to serve both phases, and its `role` decides only which requests it
    /// takes from the next step on: a change starts no process and moves no data, and a
    /// request a replica took serves there to its end. A replica the table does not list is
    /// mixed. Token models only.
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    #[reload(step)]
    pub replicas: std::collections::BTreeMap<String, ReplicaConfig>,
    /// Whether new requests wait in the daemon instead of reaching an engine; the requests in
    /// flight go on. A change takes effect from the next request on.
    #[reload(step)]
    pub paused: bool,
    /// Whether the engines give back their KV cache and CUDA graph storage and keep their
    /// weights. While asleep new requests wait; once awake again, the token requests that were
    /// in flight are recomputed and the video requests continue from their stored step. A
    /// change takes effect at the engines' next step boundary.
    #[reload(step)]
    pub asleep: bool,
    /// How many times the prefix cache was flushed: each change of the number revokes every
    /// cached prefix page at the engines' next step boundary, the requests in flight keeping
    /// the pages they hold.
    #[reload(step)]
    pub flush: u64,
    /// Whether the deployment's storage (its weights and caches) is released at its Storage
    /// Servers once it comes down at its stop signal; otherwise it stays retained for a start
    /// of the same deployment. Read when the deployment comes down.
    #[reload(step)]
    pub retire: bool,
    /// The LoRA adapters the workers hold (`[adapters]`), each the directory of an adapter
    /// under the name a request states as its model to compute with it. A change loads the
    /// adapters it adds into free slots of the workers' adapter pool and unloads the ones it
    /// removes, from the next request on; an engine that starts later loads every one of them.
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    #[reload(step)]
    pub adapters: std::collections::BTreeMap<String, String>,
}

impl Default for Config {
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
            kernels: std::collections::BTreeMap::new(),
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
            hosts: Vec::new(),
            engine_host: None,
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
            kv_events_config: proto::config::KvEventsConfig::default(),
            logging_config: LoggingConfig::default(),
            export_config: ExportConfig::default(),
            dynamo_config: DynamoConfig::default(),
            chat_config: ChatConfig::default(),
            image_fetch_config: ImageFetchConfig::default(),
            video_config: VideoConfig::default(),
            media_config: MediaConfig::default(),
            video_job_config: VideoJobConfig::default(),
            programs: std::collections::BTreeMap::new(),
            replicas: std::collections::BTreeMap::new(),
            paused: false,
            asleep: false,
            flush: 0,
            retire: false,
            adapters: std::collections::BTreeMap::new(),
        }
    }
}

/// One replica of the deployment (`[replicas.<id>]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReplicaConfig {
    /// Which requests the replica takes: `prefill`, the prompts of requests whose tokens
    /// another replica generates, which hand their KV off once the first token is sampled;
    /// `decode`, the continuations of those requests, which import the KV handed off; or
    /// `mixed`, every request, computed and generated in place.
    pub role: proto::config::Role,
}

impl Default for ReplicaConfig {
    fn default() -> Self {
        Self {
            role: proto::config::Role::Mixed,
        }
    }
}

/// The daemon's listeners (`admin_config`): the address its HTTP server serves the client API
/// on and the port the leader joins the deployment's other nodes at. The admin API is served
/// apart from them, on the Unix socket `admin.sock` of the daemon's instance directory, whose
/// file permissions (its owner alone) are its access. The daemon binds every listener when it
/// starts and states their endpoints in its readiness record, which is where a caller finds
/// them.
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
    /// The TCP port the leader accepts the deployment's other nodes on, on every interface,
    /// and the port they join it at on the first host of `hosts`; bound only by a deployment
    /// of several nodes.
    pub join: u16,
}

impl Default for AdminConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_owned(),
            port: 0,
            uds: None,
            join: JOIN_PORT,
        }
    }
}

/// Where the leader pushes the deployment's observations (`export_config`): an OpenTelemetry
/// Collector's OTLP/HTTP endpoint, which forwards them to Prometheus or another backend. The
/// admin API stays the one reading entry; the push is what a store outside the deployment
/// receives. With no endpoint nothing is pushed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExportConfig {
    /// The collector's OTLP/HTTP base URL, such as `http://collector:4318`; the metrics go to
    /// its `/v1/metrics` and the events to its `/v1/logs`, encoded as Protobuf.
    pub endpoint: Option<String>,
    /// How often the leader pushes: every completed fine bucket of `stats` and every event
    /// appended since the last push that the collector accepted.
    #[serde(with = "values::duration")]
    pub period: std::time::Duration,
}

/// The default `export_config.period`: a scrape interval's order, so a collector's series move as
/// often as a pull would have moved them.
pub const EXPORT_PERIOD: std::time::Duration = std::time::Duration::from_secs(10);

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            endpoint: None,
            period: EXPORT_PERIOD,
        }
    }
}

/// The Dynamo sidecar (`dynamo_config`): the leader daemon registers the deployment in Dynamo,
/// as Dynamo's sidecars register an engine that serves on its own, over Dynamo's default
/// planes: etcd discovery, the TCP request plane and the ZMQ event plane
/// (`crate::sidecar`). With no `etcd`, the deployment is no Dynamo worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DynamoConfig {
    /// The etcd the workers register in and the planner's decisions are read from, as the base
    /// URL of its JSON gateway (`http://etcd:2379`, Dynamo's `ETCD_ENDPOINTS`).
    pub etcd: Option<String>,
    /// The Dynamo namespace the workers register in (`DYN_NAMESPACE`).
    pub namespace: String,
    /// What a worker is: each replica, or the whole deployment.
    pub workers: Workers,
    /// The host the frontends reach the sidecar's request plane and its event publishers at;
    /// the leader's host name when unset.
    pub host: Option<String>,
    /// The TCP port of the request plane, on every interface; 0, the default, picks a free one.
    pub port: u16,
    /// The time to live of each worker's etcd lease, which the sidecar renews at a third of it
    /// (Dynamo's `ETCD_LEASE_TTL`).
    #[serde(with = "values::duration")]
    pub lease_ttl: Duration,
    /// How often the sidecar reads the planner's decision and publishes the workers' loads.
    #[serde(with = "values::duration")]
    pub period: Duration,
}

impl Default for DynamoConfig {
    fn default() -> Self {
        Self {
            etcd: None,
            namespace: "dynamo".to_owned(),
            workers: Workers::Replicas,
            host: None,
            port: 0,
            lease_ttl: Duration::from_secs(30),
            period: Duration::from_secs(1),
        }
    }
}

/// What a Dynamo worker of the deployment is (`dynamo_config.workers`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Workers {
    /// Each replica is a worker of its role (`replicas.<id>.role`), which a frontend pairs and
    /// routes to itself, and which the planner's decisions set.
    Replicas,
    /// The deployment is one aggregated worker, whose data-parallel ranks are its replicas; its
    /// routing program chooses within it.
    Deployment,
}

/// The default `admin_config.join`: a fixed port, since the other nodes of a deployment reach the
/// leader at it before they can read anything the leader reports.
pub const JOIN_PORT: u16 = 29700;

/// How the deployment moves to a new configuration in a generation or another epoch
/// (`update_config`): how long the trial lasts before it commits, what of the previous epoch it
/// keeps for a rollback, when it rolls back by itself, how the transfers yield to the serving
/// epoch, how long the switch lets the requests in flight run on, and how they continue.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UpdateConfig {
    /// How long a trial lasts: an update that no fault and no threshold rolled back by then
    /// commits, and the kept side is released.
    #[serde(with = "values::duration")]
    pub trial: Duration,
    /// How the requests an epoch's switch migrates continue on the target: `keep`, finding
    /// their committed blocks by their keys, or `recompute`, computing their state again.
    pub in_flight: InFlight,
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

/// The default `update_config.trial`: half a minute of serving on the new, within which a fault
/// or a threshold the operator set rolls it back.
pub const TRIAL: Duration = Duration::from_secs(30);

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
            trial: TRIAL,
            in_flight: InFlight::Keep,
            previous_epoch: PreviousEpoch::Parked,
            rollback: RollbackConfig::default(),
            transfer_slowdown: None,
            baseline: BASELINE,
            baseline_batches: BASELINE_BATCHES,
            drain_s: DRAIN_S,
        }
    }
}

impl Config {
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
            .unwrap_or_else(|| self.leader_host());
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
            max_running_requests: self.max_running_requests,
            max_num_batched_tokens: self.max_num_batched_tokens,
            max_logprobs: self.max_logprobs,
            device: self.device_config.clone(),
            scheduler: self.scheduler_config.clone(),
            roles: self.roles(),
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
            kv_events: self.kv_events_config.clone(),
        })
    }

    /// The role of each of the deployment's replicas, by its index: what `replicas` states, and
    /// mixed for a replica it does not list.
    pub fn roles(&self) -> Vec<proto::config::Role> {
        (0..self.data_parallel_size.get())
            .map(|index| {
                self.replicas
                    .get(&index.to_string())
                    .map_or(proto::config::Role::Mixed, |replica| replica.role)
            })
            .collect()
    }

    /// The configured model as it is stated: the repository (`model`) or the
    /// directory (`model_dir`).
    ///
    /// Empty while unstated; [`Config::validate`] refuses that.
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
    /// admits, a program's name is not one plain name or its period is zero, or the `[kernels]`
    /// table pins an empty key or one backend of an operator at two revisions. Whether the
    /// table's operators are ones the deployment's engine launches is the engine revision's to
    /// state, which the apply fixes (`engine::control::with_store_modules`), not this build's.
    pub fn validate(&self, stated: &Stated) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.states_model(),
            "the package names no model: set `model` or `model_dir` in its `config.toml`"
        );
        // Every scheduler binds the KV events' endpoints offset by its number from the first's.
        let events = &self.kv_events_config;
        for (name, endpoint) in [
            ("endpoint", &events.endpoint),
            ("replay_endpoint", &events.replay_endpoint),
        ] {
            if let Some(endpoint) = endpoint {
                anyhow::ensure!(
                    proto::config::KvEventsConfig::offset(endpoint, 0).is_some(),
                    "`kv_events_config.{name}` {endpoint} is no `tcp://<host>:<port>` endpoint"
                );
            }
        }
        let mut hosts = std::collections::BTreeSet::new();
        for host in &self.hosts {
            anyhow::ensure!(!host.trim().is_empty(), "`hosts` names an empty host");
            anyhow::ensure!(hosts.insert(host), "`hosts` names {host} twice");
        }
        anyhow::ensure!(
            self.hosts.len() < 2 || self.admin_config.join != 0,
            "`admin_config.join` is the port the other nodes reach the leader at: a deployment of \
             several nodes states a fixed one"
        );
        // One backend serves an operator at one revision (`BuildServer::cover`), so the keys of
        // an operator pin each backend at one revision.
        for (op, pins) in &self.kernels {
            let mut revisions: std::collections::BTreeMap<&str, (&str, Option<u32>)> =
                std::collections::BTreeMap::new();
            for (key, pin) in pins {
                anyhow::ensure!(
                    !key.is_empty(),
                    "`kernels.{op}` pins an empty descriptor key"
                );
                let (first, revision) = *revisions
                    .entry(pin.backend.as_str())
                    .or_insert((key.as_str(), pin.revision));
                anyhow::ensure!(
                    revision == pin.revision,
                    "`kernels.{op}` pins {pin} for {key:?} and another revision of {} for \
                     {first:?}: one backend serves an operator at one revision",
                    pin.backend
                );
            }
        }
        // An adapter is named as a request's model, which the served model's name already is.
        for (name, path) in &self.adapters {
            anyhow::ensure!(
                !name.is_empty(),
                "`adapters` names an adapter by an empty name"
            );
            anyhow::ensure!(
                name != self.served_model_name(),
                "`adapters.{name}` is the served model's name: name the adapter otherwise"
            );
            anyhow::ensure!(
                !path.is_empty(),
                "`adapters.{name}` states no directory of the adapter"
            );
        }
        if let Some(session) = &self.profiling_config.session {
            session
                .validate()
                .map_err(|error| anyhow::anyhow!("`profiling_config.session`: {error}"))?;
        }
        // A zero watchdog bound would report every job as it starts, and a zero report period
        // would report at every turn of a control loop, whether a start or a change states it.
        for (name, value) in [
            ("watchdog_timeout", self.worker_config.watchdog_timeout),
            ("report_period", self.worker_config.report_period),
        ] {
            anyhow::ensure!(
                !value.is_zero(),
                "invalid `worker_config.{name}`: it must be positive"
            );
        }
        // A replica is one of the deployment's schedulers, by its index; a request that one
        // prefills is generated on another that decodes.
        for id in self.replicas.keys() {
            let index = id
                .parse::<usize>()
                .ok()
                .filter(|index| id == &index.to_string());
            anyhow::ensure!(
                index.is_some_and(|index| index < self.data_parallel_size.get()),
                "`replicas.{id}` names no replica: the replicas are numbered 0 to {} \
                 (`data_parallel_size`)",
                self.data_parallel_size.get() - 1
            );
        }
        let roles = self.roles();
        anyhow::ensure!(
            roles.iter().any(|role| role.prefills()) && roles.iter().any(|role| role.decodes()),
            "`replicas` leaves no replica to compute prompts or none to generate tokens: a \
             deployment keeps a replica of each, or a mixed one"
        );
        // A program's name names its saved state's file and the revisions it writes.
        for (name, program) in &self.programs {
            anyhow::ensure!(
                program::plain(name),
                "`programs.{name}` is not one plain name (letters, digits, `-`, `_`, `.`)"
            );
            anyhow::ensure!(
                program.period.is_none_or(|period| !period.is_zero()),
                "`programs.{name}.period` is zero: state how often the program steps"
            );
        }
        // A token bound the deployment states above its own graph catalog is
        // refused here, by the option, before any weight loads: the engine
        // admits no batch larger than its largest captured bucket, so the
        // policy's declared limits would be refused at the first prepare
        // instead (`crates/scheduling/src/limits.rs`). A stated value is never
        // clamped — the deployment either means it or is wrong.
        // A checkpoint is stated once; the resolver checks a repository's name when it resolves
        // the checkpoint (`Config::model_source`).
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

    /// This node's place in the deployment: its own host name found in `hosts`, or, for a
    /// deployment that lists no hosts, the one node on this host.
    ///
    /// # Errors
    ///
    /// Fails when `hosts` lists hosts and this host's name is not one of them.
    pub fn identity(&self) -> anyhow::Result<Identity> {
        Identity::of(&self.hosts, &this_host(), self.admin_config.join)
    }

    /// The leader's host: the first of `hosts`, or this host for a deployment that lists none.
    pub fn leader_host(&self) -> String {
        self.hosts.first().cloned().unwrap_or_else(this_host)
    }

    /// The leader's request socket, which its engine processes connect back
    /// to: this process's own, in the daemon's instance directory `instance`. A hot restart's
    /// successor serves in its predecessor's instance beside it, so each daemon process names
    /// its own.
    pub fn request_socket(&self, instance: &Path) -> PathBuf {
        instance.join(format!("requests.{}.sock", std::process::id()))
    }

    /// The options of this node's daemon of the deployment `deployment`, whose runtime directory
    /// is `runtime`. The leader's process socket lies in the daemon's instance directory
    /// `instance`; the deployment's Storage Server
    /// on the node serves with the deployment's [`Config::storage_settings`], and its Build Server
    /// with `build_config`.
    pub fn daemon_options(
        &self,
        identity: &Identity,
        instance: &Path,
        runtime: service::RuntimeDirectory,
        deployment: &str,
    ) -> DaemonOptions {
        DaemonOptions {
            deployment: deployment.to_owned(),
            runtime,
            gpus: crate::supervision::gpus::node(),
            nnodes: identity.nodes,
            node_rank: identity.rank,
            host: identity.host.clone(),
            dist_init_addr: identity.join.clone(),
            socket: instance.join("process.sock"),
            join_timeout: crate::supervision::JOIN_TIMEOUT,
            build_server: self.build_config.clone(),
            storage: self.storage_settings(),
        }
    }

    /// The settings of the Storage Server of every host the deployment serves on: its
    /// `storage_config`, with the Mooncake connector `kv_cache_config.mooncake` states. A
    /// deployment of several nodes has its servers accept other hosts on every interface, on a
    /// port each picks, unless `storage_config.tcp` states an endpoint: the other nodes'
    /// Transfer Servers read their storage.
    pub fn storage_settings(&self) -> storage::Settings {
        let mut server = self.storage_config.clone();
        if server.tcp.is_none() && self.hosts.len() > 1 {
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

/// How a change of `storage_config` from `previous` to `next` takes effect: at the next step when
/// the running Storage Servers take every changed setting (`storage::ServerSettings::restarts`),
/// otherwise by replacing them, which a start always does.
fn storage_effect(
    previous: Option<&storage::ServerSettings>,
    next: &storage::ServerSettings,
) -> proto::reload::Effect {
    match previous {
        Some(previous) if !previous.restarts(next) => proto::reload::Effect::Step,
        _ => proto::reload::Effect::Storage,
    }
}

/// How a change of `mem_fraction_static` from `previous` to `next` takes effect: a lower share
/// in the serving engines, which give back what it leaves no room for (`step`); a higher one, and
/// a start, as a new epoch, whose ranks divide the larger share.
fn fraction_effect(previous: Option<&f64>, next: &f64) -> proto::reload::Effect {
    match previous {
        Some(previous) if next <= previous => proto::reload::Effect::Step,
        _ => proto::reload::Effect::Epoch,
    }
}

/// How a change of `worker_groups` from `previous` to `next` takes effect: at the next step when
/// the groups differ in their queue depths, which no storage is sized by (the schedulers keep
/// each group's depth of batches in flight from their next step's snapshot), and in the shares
/// of their GPUs a group states, each no higher than before (as [`fraction_effect`]); otherwise,
/// and at a start, as a new epoch, since the groups place the ranks.
fn groups_effect(
    previous: Option<&Option<Vec<proto::config::WorkerGroupConfig>>>,
    next: &Option<Vec<proto::config::WorkerGroupConfig>>,
) -> proto::reload::Effect {
    let placement = |groups: &Option<Vec<proto::config::WorkerGroupConfig>>| {
        groups.clone().map(|mut groups| {
            for group in &mut groups {
                group.queue_depth = 0;
                group.mem_fraction_static = None;
            }
            groups
        })
    };
    let Some(previous) = previous else {
        return proto::reload::Effect::Epoch;
    };
    let lowered =
        previous
            .iter()
            .flatten()
            .zip(next.iter().flatten())
            .all(
                |(before, after)| match (before.mem_fraction_static, after.mem_fraction_static) {
                    (Some(before), Some(after)) => after <= before,
                    (None, None) => true,
                    _ => false,
                },
            );
    if lowered && placement(previous) == placement(next) {
        proto::reload::Effect::Step
    } else {
        proto::reload::Effect::Epoch
    }
}

/// A node's place in its deployment (`specs/configuration.md` §9, item 2): no option states it;
/// it follows from the package's `hosts` and the host name of the node that serves the package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    /// This node's host name, which the ranks a deployment layout places on it name.
    pub host: String,
    /// This node's position in `hosts`; the leader is 0.
    pub rank: u32,
    /// The deployment's nodes.
    pub nodes: u32,
    /// The endpoint the leader accepts the other nodes on (`admin_config.join` on the first
    /// host), for a deployment of several nodes.
    pub join: Option<service::Endpoint>,
}

impl Identity {
    /// The place of the node on host `this` in a deployment of `hosts` whose leader accepts
    /// the others at port `join`.
    ///
    /// # Errors
    ///
    /// Fails when `hosts` lists hosts and `this` is not one of them.
    pub fn of(hosts: &[String], this: &str, join: u16) -> anyhow::Result<Self> {
        let Some(leader) = hosts.first() else {
            return Ok(Self {
                host: this.to_owned(),
                rank: 0,
                nodes: 1,
                join: None,
            });
        };
        let rank = hosts.iter().position(|host| host == this).with_context(|| {
            format!(
                "this host, {this}, is not one of the deployment's hosts ({}): serve the package \
                 on one of them",
                hosts.join(", ")
            )
        })?;
        let nodes = u32::try_from(hosts.len()).context("the deployment lists too many hosts")?;
        Ok(Self {
            host: this.to_owned(),
            rank: u32::try_from(rank).context("the deployment lists too many hosts")?,
            nodes,
            join: (nodes > 1).then(|| service::Endpoint::Tcp {
                host: leader.clone(),
                port: join,
            }),
        })
    }

    /// Whether this node is the leader, which serves the API, applies every package and runs
    /// the engines' control.
    pub fn leads(&self) -> bool {
        self.rank == 0
    }
}

/// This host's name, `uname(2)`'s node name: how a node finds itself in `hosts`, and the host a
/// deployment that lists none runs on.
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
