//! Serving options: one typed configuration tree, read from a TOML file,
//! a few command-line flags and `--set PATH=VALUE` assignments.
//!
//! [`ServeConfig`] is the single declaration of every option, its type and
//! its default. Its top level holds the options most deployments state; the
//! rest sit in `*_config` tables, one per owner of the behavior they
//! configure (`cuda_graph_config` for graph capture, `moe_config` for expert
//! parallelism, `video_config` for MiniMax-H3 video generation, and so on).
//! [`ServeArgs`] reads the tree: a `--config` file first, then the flags that
//! name a path of it, then `--set` assignments, each overriding the last.
//! The tree is the server's configuration: each part of the server reads its
//! own table, and [`ServeConfig::engine_spec`] states what of it concerns
//! the engine process.
//!
//! Some options configure one model family: the token batch budgets,
//! `chat_config` and `image_fetch_config` token models, `video_config`,
//! `media_config` and `video_job_config` MiniMax-H3. Startup refuses such an
//! option stated for a checkpoint of another family rather than ignoring it
//! ([`Stated::refuse_foreign_options`]).

mod args;

use std::collections::BTreeSet;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::supervision::DaemonOptions;
use anyhow::Context as _;
use proto::config::{MAX_VIDEO_SECONDS, MIN_VIDEO_SECONDS};
pub use proto::request::StandbyStorage;
use proto::request::wire;
use proto::request::{ModelDtype, PrefixCacheDtype};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use models::ModelDescription;

pub use args::{FLAG_PATHS, Listen, ServeArgs, Stated, overridden, printed, resolve_text};
pub use proto::config::{
    ChatConfig, ChatTemplateContentFormatOption, CudaGraphConfig, DEFAULT_FENCE_COMPONENTS,
    DeviceConfig, FlashInferConfig, ImageFetchConfig, ImageSize, KvCacheConfig, LoadConfig,
    LoggingConfig, MediaConfig, MigrationConfig, MoeConfig, MooncakeConfig, ProfilingConfig,
    ReasoningParser, SchedulerConfig, ServerConfig, SpeculativeConfig, VideoConfig, VideoJobConfig,
    WorkerConfig,
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

/// The model families whose behavior some options configure, with those
/// options by configuration path.
///
/// Token models schedule token batches and render chat prompts, which a
/// video pipeline does neither of; the video tables configure behavior only
/// their family has.
const FAMILY_OPTIONS: [(Family, &[&str]); 2] = [
    (
        Family::TokenModels,
        &[
            "max_num_batched_tokens",
            "scheduler_config.chunked_prefill_size",
            "speculative_config",
            "chat_config",
            "image_fetch_config",
        ],
    ),
    (
        Family::MiniMaxH3,
        &["video_config", "media_config", "video_job_config"],
    ),
];

/// A group of model families some options belong to.
#[derive(Debug, Clone, Copy)]
enum Family {
    TokenModels,
    MiniMaxH3,
}

impl Family {
    /// Whether `description` belongs to this family.
    fn serves(self, description: ModelDescription) -> bool {
        match self {
            Self::TokenModels => !description.is_pipeline(),
            Self::MiniMaxH3 => description == ModelDescription::MiniMaxH3,
        }
    }

    /// The family as the refusal names it.
    fn name(self) -> &'static str {
        match self {
            Self::TokenModels => "token models",
            Self::MiniMaxH3 => "MiniMax-H3 video generation",
        }
    }
}

impl Stated {
    /// Refuses a stated option that configures a family other than the
    /// served model's, rather than ignoring it.
    ///
    /// # Errors
    ///
    /// Fails, naming the first such option and the served model, when a
    /// video pipeline states a token-model option, or a token model states a
    /// MiniMax-H3 table.
    pub fn refuse_foreign_options(&self, description: ModelDescription) -> anyhow::Result<()> {
        for (family, paths) in FAMILY_OPTIONS {
            if family.serves(description) {
                continue;
            }
            if let Some(path) = paths.iter().find(|path| self.contains(path)) {
                anyhow::bail!(
                    "`{path}` configures {}, but the served checkpoint is a {} model; remove it",
                    family.name(),
                    description.id()
                );
            }
        }
        Ok(())
    }
}

/// Every serving option.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServeConfig {
    /// Local model directory or Hugging Face repository; required.
    pub model: Option<String>,
    /// The single model name the API serves; `model` when unset.
    pub served_model_name: Option<String>,
    /// Longest sequence a request may reach, in tokens. Unset, a checkpoint
    /// with a root `config.json` uses its `max_position_embeddings` (8192
    /// when it declares none) and a MiniMax-H3 pipeline 16384 prompt tokens.
    pub max_model_len: Option<NonZeroU32>,
    /// Element type of parameters and activations.
    pub dtype: ModelDtype,
    /// Precision policy, such as `mode = "balanced"` or per-component
    /// selections under `components`; empty selects the model's default.
    pub quantization: serde_json::Map<String, Value>,
    /// The retired name of `quantization`. A configuration that states it is
    /// refused with the name to state instead, and it is never written back.
    #[serde(rename = "quantization_config", skip_serializing)]
    pub retired_quantization_config: Option<Value>,
    /// Options of the model's multimodal processor; the processor refuses
    /// one it does not take.
    pub mm_processor_kwargs: serde_json::Map<String, Value>,
    /// Attention program (the paged-attention operator's backend) for every
    /// attention call, by the name the ranks' attention registry gives it;
    /// `auto` selects a native kernel per call.
    pub attention_backend: String,
    /// The attention program that fills prompt processing; unset takes
    /// `attention_backend` (`prefill_attention_backend`, `specs/worker.md`
    /// §5.2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_attention_backend: Option<String>,
    /// The attention program that fills one decoding token; unset takes
    /// `attention_backend` (`decode_attention_backend`, §5.2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_attention_backend: Option<String>,
    /// The grouped expert GEMM program (`moe_runner_backend`, §5.1, §9);
    /// unset lets the daemon resolve it when it prepares the epoch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub moe_runner_backend: Option<String>,
    /// This host's Program Server socket (`program_server`), which the
    /// daemon prepares an epoch's programs against (§5.1, §6.3); unset takes
    /// the Program Server's default socket. A host-layer value: the daemon
    /// reads it alone, and a change of it is a hot restart.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub program_server: Option<PathBuf>,
    /// Concurrently resident requests, clamped to what the workers hold.
    pub max_running_requests: NonZeroUsize,
    /// Token work admitted in one scheduling step, 8192 when unset; token
    /// models only.
    pub max_num_batched_tokens: Option<NonZeroUsize>,
    /// Each rank's share of its device's total storage for weights, static
    /// allocations and the KV cache, in (0, 1]. A deployment group's
    /// `mem_fraction_static` overrides it.
    pub mem_fraction_static: f64,
    /// Tensor-parallel workers of each scheduler's group when no
    /// `worker_groups` file is given.
    pub tensor_parallel_size: NonZeroUsize,
    /// Schedulers, each with its own scheduler, KV cache and worker
    /// groups. With a `worker_groups` file, its model groups form the engine
    /// cores as equal consecutive blocks.
    pub data_parallel_size: NonZeroUsize,
    /// Deployment file placing worker groups and their components.
    pub worker_groups: Option<PathBuf>,
    /// This node's host name; its daemon runs the ranks placed on this
    /// host, and a deployment rank entry without a host names it.
    pub host_identity: String,
    /// Host running the engine and ranks whose configuration omits a host.
    /// Unset uses this node's identity. The engine can move to another
    /// joined node through a world change; the daemon's identity stays local.
    pub engine_host: Option<String>,
    /// Nodes of the deployment, each running one `uniserve serve`.
    pub nnodes: NonZeroU32,
    /// This node's rank among `nnodes`. The leader, rank 0, serves the API
    /// and runs the engine; every other node runs only its daemon, which
    /// joins the leader's.
    pub node_rank: u32,
    /// The leader's `host:port`, which the other nodes' daemons join at and
    /// the leader binds; required with several nodes.
    pub dist_init_addr: Option<String>,
    pub server_config: ServerConfig,
    /// The daemon's own HTTP server (`admin_config`): the address this node's
    /// admin API listens on (design §7.7). The application's server is
    /// `server_config`'s, which the frontend reads.
    pub admin_config: AdminConfig,
    pub load_config: LoadConfig,
    pub kv_cache_config: KvCacheConfig,
    pub scheduler_config: SchedulerConfig,
    pub speculative_config: SpeculativeConfig,
    pub migration_config: MigrationConfig,
    pub update_config: UpdateConfig,
    pub device_config: DeviceConfig,
    pub moe_config: MoeConfig,
    pub cuda_graph_config: CudaGraphConfig,
    pub flashinfer_config: FlashInferConfig,
    pub worker_config: WorkerConfig,
    pub profiling_config: ProfilingConfig,
    pub logging_config: LoggingConfig,
    pub chat_config: ChatConfig,
    pub image_fetch_config: ImageFetchConfig,
    pub video_config: VideoConfig,
    pub media_config: MediaConfig,
    pub video_job_config: VideoJobConfig,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            model: None,
            served_model_name: None,
            max_model_len: None,
            dtype: ModelDtype::BFloat16,
            quantization: serde_json::Map::new(),
            retired_quantization_config: None,
            mm_processor_kwargs: serde_json::Map::new(),
            attention_backend: "auto".to_owned(),
            prefill_attention_backend: None,
            decode_attention_backend: None,
            moe_runner_backend: None,
            program_server: None,
            max_running_requests: positive(128),
            max_num_batched_tokens: None,
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
        }
    }
}

/// The daemon's HTTP server (`admin_config`): the address this node's admin
/// API listens on (design §7.7). The application's own server is
/// `server_config`'s; the two are separate options so the management API and
/// the application API can listen at different addresses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AdminConfig {
    /// TCP address the server listens on.
    pub host: String,
    /// TCP port the server listens on.
    pub port: u16,
    /// Unix domain socket to listen on instead of TCP; a stale socket file
    /// is replaced and the socket file is removed at shutdown.
    pub uds: Option<String>,
}

impl Default for AdminConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_owned(),
            port: 8000,
            uds: None,
        }
    }
}

/// `current` with `target`'s daemon options (those `daemon_options_differ`
/// compares): the configuration a hot restart of the daemon serves when an
/// update changes both the daemon and what a world executes, before the
/// daemon's successor changes the world.
pub fn with_daemon_options(current: &ServeConfig, target: &ServeConfig) -> ServeConfig {
    ServeConfig {
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
/// daemon applies; the engine spec covers every other option.
pub fn daemon_options_differ(current: &ServeConfig, target: &ServeConfig) -> bool {
    current.server_config != target.server_config
        || current.admin_config != target.admin_config
        || current.logging_config != target.logging_config
        || current.chat_config != target.chat_config
        || current.image_fetch_config != target.image_fetch_config
        || current.media_config != target.media_config
        || current.video_job_config != target.video_job_config
        || current.migration_config != target.migration_config
        || current.mm_processor_kwargs != target.mm_processor_kwargs
        // Where this host's Program Server listens: the daemon reads it
        // alone, and an epoch's programs are prepared against it (§5.1).
        || current.program_server != target.program_server
}

/// How the deployment moves to a new configuration in another world
/// (`update_config`, `PUT /admin/config`): what of the previous world its
/// trial keeps for a rollback, when the trial rolls back by itself, and how
/// the transfers and the warming yield to the serving world. A value an
/// update states in its own assignments governs that update alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UpdateConfig {
    /// What the previous world keeps while the new one is on trial.
    pub previous_world: PreviousWorld,
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
    /// The most tokens a request in flight may have computed past the pages
    /// the host KV cache holds of it at the switchover; unset uses the largest
    /// source or target cache page.
    pub cache_tail_tokens: Option<std::num::NonZeroU32>,
    /// Source prefill tokens per scheduling step after the first warming
    /// round; unset uses the largest source or target cache page.
    pub cache_prefill_tokens: Option<std::num::NonZeroU32>,
}

/// The default `update_config.transfer_slowdown`: on GB200, above the
/// serving steps' second-to-second variation without transfers and below
/// the slowdowns copies at a link's whole budget cause
/// (`docs/configuration.md`).
pub const TRANSFER_SLOWDOWN: f64 = 0.1;

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            previous_world: PreviousWorld::Parked,
            rollback: RollbackConfig::default(),
            transfer_slowdown: TRANSFER_SLOWDOWN,
            cache_tail_tokens: None,
            cache_prefill_tokens: None,
        }
    }
}

/// What the previous world keeps during its successor's trial
/// (`update_config.previous_world`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreviousWorld {
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
    /// How much longer than the previous world's before its promotion the
    /// new world's average step may take, as a fraction (`0.2` for 20%).
    pub step_time_regression: Option<f64>,
    /// How much lower than the previous world's before its promotion the
    /// new world's token throughput may fall, as a fraction.
    pub throughput_regression: Option<f64>,
    /// The span the step time and throughput are averaged over, before the
    /// promotion and during the trial; required with either regression.
    #[serde(with = "values::optional_duration")]
    pub window: Option<Duration>,
}

impl ServeConfig {
    /// The configured model: a local directory or a Hugging Face repository.
    ///
    /// Empty while unstated; [`ServeConfig::validate`] refuses that.
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
                matches!(self.device_config.device.as_str(), "cuda" | "gpu"),
                "`device_config.devices` orders CUDA devices: `device_config.device` is {:?}",
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
        let server = &self.server_config;
        match &server.uds {
            Some(path) => anyhow::ensure!(
                !path.trim().is_empty(),
                "`server_config.uds` must not be empty"
            ),
            None => anyhow::ensure!(
                !server.host.trim().is_empty(),
                "`server_config.host` must not be empty"
            ),
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
        // A smaller capacity admits no request and a larger one exceeds what
        // the video API accepts.
        let seconds = MIN_VIDEO_SECONDS..=MAX_VIDEO_SECONDS;
        anyhow::ensure!(
            seconds.contains(&self.video_config.max_seconds),
            "`video_config.max_seconds` must lie in [{}, {}], got {}",
            seconds.start(),
            seconds.end(),
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
    /// to: this process's own, in the user's runtime directory.
    pub fn request_socket(&self) -> PathBuf {
        runtime_directory().join(format!("requests-{}.sock", std::process::id()))
    }

    /// The options of this node's daemon. The leader's process socket is
    /// this process's own, in the user's runtime directory; `executable`
    /// starts the node's Storage Server when it has none.
    pub fn daemon_options(&self) -> DaemonOptions {
        let directory = runtime_directory();
        DaemonOptions {
            nnodes: self.nnodes.get(),
            node_rank: self.node_rank,
            host: self.host_identity.clone(),
            dist_init_addr: self.dist_init_addr.clone(),
            socket: directory.join(format!("process-{}.sock", std::process::id())),
            executable: self.worker_config.executable.clone(),
            join_timeout: crate::supervision::JOIN_TIMEOUT,
        }
    }

    /// The engine spec these options state, without the model's facts,
    /// which the leader resolves from the checkpoint.
    ///
    /// Values the options name by a word (a policy, a backend, an element
    /// type) travel as that word, which the engine reads; the rank layout
    /// travels as stated, which the engine places.
    ///
    /// # Errors
    ///
    /// Fails, naming the option, when a capture list is not strictly
    /// increasing or repeats a shape, or a batch bound exceeds the worker's
    /// `u32` fields; and when the deployment file cannot be read.
    pub fn engine_spec(&self) -> anyhow::Result<wire::EngineSpec> {
        let graphs = &self.cuda_graph_config;
        let kv_cache = &self.kv_cache_config;
        let worker = &self.worker_config;
        let text = |path: &Option<PathBuf>| {
            path.as_ref()
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        let bounded = |value: usize, option: &str| {
            u32::try_from(value)
                .with_context(|| format!("`{option}` exceeds the worker field width"))
        };
        let scheduling = wire::Scheduling {
            max_batch: bounded(
                self.scheduler_config.max_batch.get(),
                "scheduler_config.max_batch",
            )?,
            max_num_batched_tokens: self
                .max_num_batched_tokens
                .map(|tokens| bounded(tokens.get(), "max_num_batched_tokens"))
                .transpose()?,
            max_running_requests: bounded(self.max_running_requests.get(), "max_running_requests")?,
            max_queued_requests: self
                .scheduler_config
                .max_queued_requests
                .map(|requests| requests.get() as u64),
            chunked_prefill_size: self
                .scheduler_config
                .chunked_prefill_size
                .map(|size| bounded(size.get(), "scheduler_config.chunked_prefill_size"))
                .transpose()?,
        };
        Ok(wire::EngineSpec {
            // A deployment is what one leader serves under its served name.
            deployment: self.served_model_name().to_owned(),
            model: self.model().to_owned(),
            host: self
                .engine_host
                .as_ref()
                .unwrap_or(&self.host_identity)
                .clone(),
            rank_layout: Some(wire::RankLayout {
                worker_groups: text(&self.worker_groups),
                worker_groups_content: self
                    .worker_groups
                    .as_deref()
                    .map(|path| {
                        std::fs::read_to_string(path).with_context(|| {
                            format!("reading the deployment file {}", path.display())
                        })
                    })
                    .transpose()?
                    .unwrap_or_default(),
                tensor_parallel_size: bounded(
                    self.tensor_parallel_size.get(),
                    "tensor_parallel_size",
                )?,
                data_parallel_size: bounded(self.data_parallel_size.get(), "data_parallel_size")?,
                device: self.device_config.device.clone(),
                devices: self.device_config.devices.clone(),
                queue_depth: bounded(
                    self.device_config.queue_depth.get(),
                    "device_config.queue_depth",
                )?,
                worker_hosts: self.device_config.worker_hosts.clone(),
                mem_fraction_static: self.mem_fraction_static,
            }),
            transfer: self.device_config.transfer.clone().unwrap_or_default(),
            scheduling: Some(scheduling),
            kv_cache: Some(wire::KvCache {
                max_tokens: kv_cache.max_tokens.map(NonZeroU64::get),
                block_size: kv_cache.block_size.map(NonZeroU32::get),
                dtype: kv_cache
                    .dtype
                    .map(PrefixCacheDtype::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                prefix_caching: kv_cache.prefix_caching,
                host_tier: kv_cache.host_tier,
                mooncake: kv_cache.mooncake.as_ref().map(|mooncake| wire::Mooncake {
                    master: mooncake.master.clone(),
                    metadata_server: mooncake.metadata_server.clone(),
                    protocol: mooncake.protocol.clone(),
                    device: mooncake.device.clone(),
                    segment_bytes: mooncake.global_segment_gib << 30,
                    library: text(&mooncake.library),
                }),
                boundary_interval: kv_cache.boundary_interval.map(NonZeroU32::get),
                state_interval: kv_cache.state_interval.map(NonZeroU32::get),
                host_ratio: kv_cache.host_ratio,
                host_bytes: kv_cache.host_gib.map(|gib| gib.get() << 30),
            }),
            workers: Some(wire::Workers {
                executable: worker.executable.to_string_lossy().into_owned(),
                watchdog_timeout_ms: u64::try_from(worker.watchdog_timeout.as_millis())
                    .unwrap_or(u64::MAX),
                mesh: worker.mesh.clone().unwrap_or_default(),
                standby: worker.standby,
                attention_backend: self.attention_backend.clone(),
                dtype: word(&self.dtype),
                quantization: Value::Object(self.quantization.clone()).to_string(),
                load_format: word(&self.load_config.format),
                download_dir: text(&self.load_config.download_dir),
                load_threads: self.load_config.threads.map(NonZeroU32::get),
                standby_storage: word(&worker.standby_storage),
                weights: self.load_config.weights.clone().unwrap_or_default(),
                parallel_degrees: self
                    .load_config
                    .parallel_degrees
                    .iter()
                    .map(|degree| degree.get())
                    .collect(),
            }),
            cuda_graphs: Some(wire::CudaGraphs {
                policy: word(&graphs.policy),
                decode_batch_sizes: increasing(
                    "cuda_graph_config.decode_batch_sizes",
                    &graphs.decode_batch_sizes,
                )?,
                prefill_token_sizes: increasing(
                    "cuda_graph_config.prefill_token_sizes",
                    &graphs.prefill_token_sizes,
                )?,
                flow_shapes: distinct("cuda_graph_config.flow_shapes", &graphs.flow_shapes)?,
            }),
            // Video workers size resident media state for the longest
            // duration served and provision only the frame counts the API
            // admits, and prepare exactly the canvases these resolutions and
            // aspect ratios name.
            video: Some(wire::Video {
                max_seconds: self.video_config.max_seconds,
                min_seconds: MIN_VIDEO_SECONDS,
                resolutions: self.video_config.resolutions.clone(),
                aspect_ratios: self.video_config.aspect_ratios.clone(),
                text_capacities: increasing(
                    "video_config.text_capacities",
                    &self.video_config.text_capacities,
                )?,
                max_condition_rows: self.video_config.max_condition_rows,
                ffmpeg: self.media_config.ffmpeg.to_string_lossy().into_owned(),
            }),
            experts: Some(wire::Experts {
                expert_parallel: self.moe_config.expert_parallel,
                exchange: word(&self.moe_config.exchange),
                microbatches: self.moe_config.microbatches.get(),
            }),
            model_facts: None,
            world: 0,
            predecessor: None,
            stage_cache: false,
            cache_keys: Vec::new(),
            segments: Vec::new(),
            retire: Vec::new(),
            speculation: (self.speculative_config.num_draft_tokens > 0).then_some(
                wire::Speculation {
                    num_draft_tokens: self.speculative_config.num_draft_tokens,
                },
            ),
            profiling: Some(wire::Profiling {
                fence_probes: self.profiling_config.fence_probes,
                fence_components: self.profiling_config.fence_components.clone(),
                fence_probes_off: self.profiling_config.fence_probes_off.clone(),
                graph_pc_sampling: self.profiling_config.graph_pc_sampling,
            }),
            received_weights: Vec::new(),
            expert_order: Default::default(),
        })
    }
}

/// The user's runtime directory, where this node's sockets lie beside its
/// Storage Server's.
/// The directory this daemon keeps its runtime files in: its sockets, and the
/// launch and build-config files its main processes read.
pub(crate) fn runtime_directory() -> PathBuf {
    storage::socket::default_socket()
        .parent()
        .map_or_else(std::env::temp_dir, Path::to_path_buf)
}

/// The word a serving option names `value` by, as the option tree prints
/// it.
fn word<T: Serialize>(value: &T) -> String {
    match serde_json::to_value(value) {
        Ok(Value::String(word)) => word,
        _ => String::new(),
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

/// Checks that a stated capture or capacity list is strictly increasing.
fn increasing(path: &str, values: &Option<Vec<NonZeroU32>>) -> anyhow::Result<Option<wire::Sizes>> {
    let Some(values) = values else {
        return Ok(None);
    };
    anyhow::ensure!(
        values.windows(2).all(|pair| pair[0] < pair[1]),
        "`{path}` must be strictly increasing, got {values:?}"
    );
    Ok(Some(wire::Sizes {
        values: values.iter().map(|value| value.get()).collect(),
    }))
}

/// Checks that stated image capture sizes are distinct.
fn distinct(path: &str, shapes: &Option<Vec<ImageSize>>) -> anyhow::Result<Option<wire::Shapes>> {
    let Some(shapes) = shapes else {
        return Ok(None);
    };
    anyhow::ensure!(
        shapes.iter().collect::<BTreeSet<_>>().len() == shapes.len(),
        "`{path}` repeats an image size"
    );
    Ok(Some(wire::Shapes {
        shapes: shapes
            .iter()
            .map(|shape| wire::Shape {
                height: shape.0[0],
                width: shape.0[1],
            })
            .collect(),
    }))
}
