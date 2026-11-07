//! The deployment's composition: its executor groups, its schedulers and their
//! handles.
//!
//! A deployment runs one scheduler per DP rank. [`SchedulerHandle`] owns a
//! scheduler's thread and is the front door request producers submit and
//! control requests through.
//!
//! [`ExecutorGroupConfig`] is the static deployment description: each group has
//! an ordered list of physical [`ExecutorRank`]s and named components whose
//! `ranks` index into that list. [`EngineConfig`] adds the scheduler's budgets
//! and the facts of what it serves. A main process builds one scheduler per
//! rank it holds with `SchedulerHandle::over_batches_telling`, over the
//! devices of its own process; `SchedulerHandle::with_executor` takes the CPU
//! simulation backend instead.

use ipc::DEFAULT_COMPONENT;
use ipc::MediaCall;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

#[cfg(feature = "testing")]
use crate::executor::Executor;
use crate::executor::ExecutorGroupId;
use crate::handle::{CommandSender, EventRx, SubmitError};
use crate::scheduler::guidance_prefix_rows;
#[cfg(feature = "testing")]
use crate::scheduler::{
    DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_RUNNING_REQUESTS,
};
use crate::scheduler::{Scheduler, SpecialTokenIds};
use crate::scheduler::{SchedulerConfig, SchedulerStats};
use crate::{CommandWaker, GenerationLimits, Request};
use anyhow::Context as _;
use executor::batch::Batches;
use ipc::MainProcessInfo;
use ipc::RequestId;
use proto::config::BlockConfig;
#[cfg(test)]
use proto::config::BlockDistribution;
use proto::request::ExpertExchange;

/// One cooperative member's physical execution endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutorRank {
    /// Host identity: the daemon of the node with this host identity starts
    /// the rank.
    pub host: String,
    /// Device string, such as `cuda:0` for a GPU rank or `cpu` for a CPU executor.
    pub device: String,
}

/// The numerical work an executor group owns in a deployment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutorGroupRole {
    /// Model capabilities, request state, and (when present) attention/KV.
    #[default]
    Model,
    /// A model group that prefills prompts only: a request continues on a
    /// `Decode` group of its scheduler, which installs the KV cache the
    /// prefill exported (prefill/decode disaggregation).
    Prefill,
    /// A model group that decodes only, continuing requests a `Prefill`
    /// group prefilled.
    Decode,
    /// Routed experts shared by the deployment's schedulers.
    Experts,
}

impl ExecutorGroupRole {
    /// Whether the group serves requests: every role but `Experts`.
    pub fn serves_requests(self) -> bool {
        self != Self::Experts
    }
}

/// Static computation components and their ordered physical rank membership.
///
/// A deployment file ([`crate::Deployment`]) or the command-line shorthand
/// ([`ExecutorGroupConfig::data_parallel`]) builds them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExecutorGroupConfig {
    pub id: ExecutorGroupId,
    pub role: ExecutorGroupRole,
    /// Ordered physical members; a rank's position is its rank number within
    /// the group.
    pub ranks: Vec<ExecutorRank>,
    /// Components keyed by name, each listing indices into `ranks`.
    pub components: BTreeMap<String, BlockConfig>,
    /// Maximum number of physical runs in flight per rank.
    pub queue_depth: usize,
    /// Per-rank share of device storage available to this group for static
    /// allocations (weights and KV cache), instead of the deployment-wide
    /// `--mem-fraction-static`.
    ///
    /// Distinct groups may share one physical GPU. Their configured
    /// shares must leave enough aggregate headroom for the device runtime;
    /// validation checks each group alone, never the sum across groups, and
    /// accepts the range [`ExecutorGroupConfig::validate_mem_fraction_static`] states.
    pub mem_fraction_static: Option<f64>,
}

impl ExecutorGroupConfig {
    /// Validates one group in isolation.
    ///
    /// Fails when the identity is not a valid `ExecutorGroupId`, `queue_depth` is
    /// zero, `validate_mem_fraction_static` refuses a set `mem_fraction_static`, or
    /// `validate_members` rejects the ranks and components.
    pub fn validate(&self) -> anyhow::Result<()> {
        ExecutorGroupId::new(self.id.0.clone())?;
        anyhow::ensure!(
            self.queue_depth > 0,
            "executor {} queue depth must be positive",
            self.id
        );
        if let Some(fraction) = self.mem_fraction_static {
            Self::validate_mem_fraction_static(fraction)
                .map_err(|error| anyhow::anyhow!("executor {}: {error}", self.id))?;
        }
        Self::validate_members(&self.ranks, &self.components)?;
        anyhow::ensure!(
            self.components.is_empty() != self.role.serves_requests(),
            "model executors require components; expert executors have no request components"
        );
        Ok(())
    }

    /// Checks a group's static storage fraction, `mem_fraction_static`.
    ///
    /// Each rank grants itself this share of every device's total storage,
    /// less what it already holds and never more than the device has free, so
    /// any finite value in (0, 1] is meaningful, 1 granting the whole device.
    pub fn validate_mem_fraction_static(fraction: f64) -> anyhow::Result<()> {
        anyhow::ensure!(
            fraction.is_finite() && fraction > 0.0 && fraction <= 1.0,
            "mem_fraction_static {fraction} must be finite and in (0, 1]"
        );
        Ok(())
    }

    /// Validates a group's rank list and its component membership.
    ///
    /// Ranks must be nonempty, name a host and a device, and not repeat a
    /// `(host, device)` pair unless the device is `cpu`. Components must be
    /// valid for their executor role; each needs a nonempty name, a nonempty list of distinct
    /// in-range rank indices, a `parallel_config` whose
    /// `BlockConfig::world_size` succeeds, and a positive `units_per_rank`.
    /// A component without a distribution must have a parallel world size
    /// equal to its rank count; one with a distribution must have world size
    /// one.
    pub fn validate_members(
        ranks: &[ExecutorRank],
        components: &BTreeMap<String, BlockConfig>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(!ranks.is_empty(), "executor requires rank members");
        let mut devices = BTreeSet::new();
        for rank in ranks {
            anyhow::ensure!(
                !rank.host.is_empty() && !rank.device.is_empty(),
                "rank requires host and device"
            );
            // Within a group a `(host, device)` pair names one rank, except
            // `cpu`, which several ranks on one host may share.
            anyhow::ensure!(
                devices.insert((&rank.host, &rank.device)) || rank.device == "cpu",
                "executor repeats a physical device"
            );
        }
        for (name, entry) in components {
            anyhow::ensure!(!name.is_empty(), "component name must not be empty");
            anyhow::ensure!(
                !entry.ranks.is_empty() && entry.ranks.iter().all(|&rank| rank < ranks.len()),
                "component {name} contains invalid members"
            );
            entry.validate()?;
        }
        Ok(())
    }

    /// Validates unique group identities and every local rank layout.
    ///
    /// Component names may repeat across groups. Repeated names are
    /// replicas of the same numerical component; the scheduler binds each
    /// request to one of them and keeps that affinity for the request lifetime.
    ///
    /// Physical devices may repeat across groups; only `validate_members`
    /// checks device reuse, and it does so within one group.
    pub fn validate_all(workers: &[Self]) -> anyhow::Result<()> {
        anyhow::ensure!(!workers.is_empty(), "engine requires executors");
        let mut ids = BTreeSet::new();
        for worker in workers {
            worker.validate()?;
            anyhow::ensure!(
                ids.insert(&worker.id),
                "executor group identity is repeated"
            );
        }
        // A request prefilled on a prefill group decodes on a decode group,
        // which holds the same text model and nothing else.
        let has = |role| workers.iter().any(|worker| worker.role == role);
        anyhow::ensure!(
            has(ExecutorGroupRole::Prefill) == has(ExecutorGroupRole::Decode),
            "prefill groups and decode groups are configured together"
        );
        for worker in workers {
            if matches!(
                worker.role,
                ExecutorGroupRole::Prefill | ExecutorGroupRole::Decode
            ) {
                anyhow::ensure!(
                    worker.components.keys().eq([DEFAULT_COMPONENT]),
                    "executor {}: a prefill or decode group places the `{DEFAULT_COMPONENT}` component alone",
                    worker.id
                );
            }
        }
        Ok(())
    }

    /// One component tensor-parallel over every rank.
    ///
    /// The `uniserve` CLI uses this rank layout when no `--executor-groups` configuration
    /// is given, and the default engine settings and the simulation use it. A
    /// deployment with several components, or with other partitions, writes
    /// its own configuration.
    pub fn single_component(name: &str, rank_count: usize) -> BTreeMap<String, BlockConfig> {
        BTreeMap::from([(
            name.into(),
            BlockConfig {
                tensor_parallel_size: rank_count,
                ..BlockConfig::parallel((0..rank_count).collect())
            },
        )])
    }

    /// Expands the single-group CLI shorthand into explicit device membership
    /// across the named hosts, carrying the given components.
    ///
    /// Ranks are assigned in blocks, so the lowest ranks stay on the first
    /// host, and each host numbers its devices from zero. A rank layout on one
    /// host is the same arithmetic with one block. A `device` of `cuda` or
    /// `gpu` becomes `cuda:<local index>`; any other string is used verbatim
    /// for every rank.
    ///
    /// `components` must already be resolved for `rank_count`; this assigns
    /// rank identity and nothing else. The group is named `model` and has no
    /// `mem_fraction_static` of its own. Nothing is validated here, and an empty
    /// `hosts` yields no ranks.
    pub fn placed(
        hosts: &[String],
        device: &str,
        rank_count: usize,
        queue_depth: usize,
        components: BTreeMap<String, BlockConfig>,
    ) -> Self {
        // A host takes its share of the ranks, and the first hosts take the
        // remainder, so a count that does not divide evenly still places every
        // rank and keeps each host's block contiguous.
        let host_count = hosts.len().max(1);
        let share = rank_count / host_count;
        let remainder = rank_count % host_count;
        let mut rank_layout = Vec::with_capacity(rank_count);
        for (index, host) in hosts.iter().enumerate() {
            let count = share + usize::from(index < remainder);
            for local in 0..count {
                rank_layout.push(ExecutorRank {
                    host: host.clone(),
                    device: if matches!(device, "cuda" | "gpu") {
                        format!("cuda:{local}")
                    } else {
                        device.into()
                    },
                });
            }
        }
        Self {
            id: ExecutorGroupId("model".into()),
            role: ExecutorGroupRole::Model,
            ranks: rank_layout,
            components,
            queue_depth,
            mem_fraction_static: None,
        }
    }

    /// Expands the data-parallel CLI shorthand into one group of
    /// `ranks_per_core` ranks for each of `cores` schedulers, each carrying
    /// `components`.
    ///
    /// All `cores * ranks_per_core` ranks are placed at once as
    /// [`ExecutorGroupConfig::placed`] places them, in blocks over `hosts`, and
    /// consecutive runs of `ranks_per_core` form the groups, so the first
    /// schedulers fill the engine's host first. The group of one scheduler
    /// is `model`; the group of core `i` of several is `model-<i>`.
    pub fn data_parallel(
        hosts: &[String],
        device: &str,
        cores: usize,
        ranks_per_core: usize,
        queue_depth: usize,
        components: BTreeMap<String, BlockConfig>,
    ) -> Vec<Self> {
        let placed = Self::placed(
            hosts,
            device,
            cores * ranks_per_core,
            queue_depth,
            components,
        );
        if cores <= 1 {
            return vec![placed];
        }
        placed
            .ranks
            .chunks(ranks_per_core.max(1))
            .enumerate()
            .map(|(index, ranks)| Self {
                id: ExecutorGroupId(format!("model-{index}")),
                role: ExecutorGroupRole::Model,
                ranks: ranks.to_vec(),
                components: placed.components.clone(),
                queue_depth,
                mem_fraction_static: None,
            })
            .collect()
    }
}

/// Configuration of one deployment's schedulers: one per DP rank,
/// `data_parallel_size` of them.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Model generation requirements, intersected with loaded executor capacities.
    pub generation_limits: GenerationLimits,
    /// Maximum number of calls assembled into a single forward batch.
    pub max_batch: usize,
    /// Maximum number of tokens scheduled in one engine step.
    pub max_num_batched_tokens: usize,
    /// Maximum number of concurrently running requests.
    pub max_running_requests: usize,
    /// Most requests, waiting or finished with undelivered events, the
    /// scheduler holds before it rejects a submission as overloaded; `None`
    /// queues without bound.
    pub max_queued_requests: Option<usize>,
    /// When a request whose draft cache is missing has it refilled.
    /// The captured graph sizes the deployment states, in tokens, ascending
    /// and without duplicates (`Room.graph_tiers`). Empty when the spec states
    /// none; see `SchedulerConfig::graph_tiers`.
    pub graph_tiers: Vec<u32>,
    /// Per-request token ceiling for one prefill chunk.
    pub chunked_prefill_size: usize,
    /// Reuse and retain prompt cache prefixes across requests. Disabling this
    /// does not remove the cache storage needed within an active request.
    pub prefix_cache: bool,
    /// The pages of the largest block size between the aligned boundaries
    /// a long prompt offloads its sliding windows to the host KV cache
    /// before.
    pub boundary_interval: usize,
    /// The decoded tokens between two checkpoints of a request's state
    /// groups (`Scheduler::set_state_interval`).
    pub state_interval: usize,
    /// How large the world's host KV cache is (`kv_cache_config.host_ratio`
    /// and `host_gib`).
    pub host_cache: crate::kv_cache::HostCacheSize,
    /// Static computation components and physical rank membership.
    pub worker_groups: Vec<ExecutorGroupConfig>,
    /// Number of schedulers `worker_groups` forms, as equal consecutive
    /// blocks of model groups. One means the groups form one scheduler.
    pub data_parallel_size: usize,
    /// Whether the schedulers shard the model's routed experts, and how
    /// they exchange tokens: each scheduler's group is then one executor,
    /// keeps its share of every expert layer and exchanges tokens with the
    /// others at those layers, while attention and every other layer stay
    /// data-parallel. `None` keeps every expert on every scheduler.
    pub expert_parallel: Option<ExpertExchange>,
    /// Model identifier or local model path.
    pub model: String,
    /// Numeric data type of the model's parameters and activations.
    pub dtype: crate::ModelDtype,
    /// The deployment the schedulers serve, which their storage scopes name:
    /// the served model name.
    pub deployment: String,
    /// This process's own host identity.
    pub host: String,
    /// Whether the schedulers copy cache pages through the host KV cache
    /// (`kv_cache_config.host_tier`).
    pub host_kv_cache: bool,
    /// Whether the host KV cache keeps committed pages in the KV storage
    /// (`kv_cache_config.mooncake`).
    pub kv_storage: bool,
    /// Block-diffusion sampling of every canvas the deployment generates;
    /// `None` when the served model generates no canvases.
    pub canvas_sampling: Option<ipc::CanvasSampling>,
    /// The most draft tokens a call proposes with the model's drafter; zero
    /// when the deployment does not speculate.
    pub draft_tokens: u32,
    /// Beginning-of-sequence token identifier.
    pub bos: u32,
    /// Token identifiers that terminate generation.
    pub eos: Vec<u32>,
    /// Token identifier that terminates encoded image content.
    pub end_of_image: u32,
    /// The world the engine executes (`EngineSpec::world`): its storage is
    /// keyed and retained under it.
    pub world: u64,
    /// Whether the schedulers start on trial, as a successor world's do: they
    /// serve once promoted ([`SchedulerHandle::promote`]), and until the commit
    /// ([`SchedulerHandle::commit`]) a fatal failure ends the world.
    pub trial: bool,
}

/// The launch-file and build-config facts a main process holds about the ranks
/// its schedulers run, and nothing else.
///
/// [`EngineConfig::of`] derives every other field of [`EngineConfig`] from
/// these: from the model's own configuration where the launch file states it,
/// from the build configuration's constants, or from the platform's defaults. A
/// fact the derivation needs and these facts do not state is refused by name,
/// never defaulted: a scheduler built on a guessed shape would decide on work
/// it does not run.
///
/// Each field names its source: *launch* is the launch file the daemon writes
/// for this world (`EngineSpec`, and the deployment it places), *model* is the
/// model's own configuration the launcher reads at the model path and states in
/// the launch file, and *build config* is what this build was compiled for.
/// Nothing here is a scheduling decision: those belong to the policy program
/// (§4.4).
#[derive(Clone, Debug, Default)]
pub struct RankFacts {
    /// The model's path or served name (*launch*: `model`).
    pub model: Option<String>,
    /// The deployment the ranks belong to (*launch*: `deployment`), which keys its host KV cache.
    pub deployment: Option<String>,
    /// Whether the deployment keeps a host KV cache (*build config*: `kv_cache.host_tier`).
    pub host_kv_cache: bool,
    /// The model's numerical dtype (*launch*: `worker_config.dtype`).
    pub dtype: Option<crate::ModelDtype>,
    /// The world the ranks execute (*launch*: `world`).
    pub world: Option<u64>,
    /// The executor group's identity (*launch*: `device_config`).
    pub group: Option<String>,
    /// This process's rank within the group (*launch*: `device_config`).
    pub rank: Option<usize>,
    /// The group's rank count (*launch*: `device_config`).
    pub world_size: Option<usize>,
    /// The group's devices, in rank order (*launch*: `device_config`).
    pub devices: Option<Vec<String>>,
    /// The group's hosts, in rank order (*launch*: `device_config`).
    pub hosts: Option<Vec<String>>,
    /// Resident request slots per scheduler (*launch*:
    /// `scheduling.max_running_requests`).
    pub max_running_requests: Option<usize>,
    /// Calls one step may assemble (*launch*: `scheduling.max_batch`).
    pub max_batch: Option<usize>,
    /// Tokens one step may carry (*launch*: `scheduling.max_num_batched_tokens`).
    pub max_num_batched_tokens: Option<usize>,
    /// The host KV cache's size (*launch*: `kv_cache_config.host_ratio` and
    /// `host_bytes`); `None` for a deployment that states none.
    pub host_cache: Option<crate::kv_cache::HostCacheSize>,
    /// The captured graph sizes in tokens, ascending (*launch*:
    /// `cuda_graphs.decode_batch_sizes` and `prefill_token_sizes`).
    pub graph_tiers: Option<Vec<u32>>,
    /// Beginning-of-sequence token (*model*: the tokenizer's).
    pub bos: Option<u32>,
    /// The tokens that terminate generation (*model*: the tokenizer's).
    pub eos: Option<Vec<u32>>,
    /// The token that ends encoded image content (*model*: the model's
    /// configuration).
    pub end_of_image: Option<u32>,
    /// What one request may be given to generate (*model*: the model's own
    /// generation limits).
    pub generation_limits: Option<GenerationLimits>,
}

/// A fact the derivation needs, named with where it comes from.
fn fact<'a, T>(value: &'a Option<T>, name: &str, source: &str) -> anyhow::Result<&'a T> {
    value.as_ref().with_context(|| {
        format!("the engine's facts state no {name}, which is the {source}'s to state")
    })
}

impl EngineConfig {
    /// Builds the configuration of the schedulers a main process runs over
    /// its ranks, from the facts it holds ([`RankFacts`]).
    ///
    /// Everything not stated there is derived here: the KV cache's host size,
    /// the page boundaries and the capture catalog's bounds. A fact the
    /// derivation needs and the facts do not state fails with the fact's name
    /// and its source.
    pub fn of(facts: &RankFacts) -> anyhow::Result<Self> {
        let model = fact(&facts.model, "model", "launch file")?.clone();
        let dtype = *fact(&facts.dtype, "dtype", "launch file")?;
        let world = *fact(&facts.world, "world", "launch file")?;
        let group = fact(&facts.group, "executor group identity", "launch file")?.clone();
        let rank = *fact(&facts.rank, "rank", "launch file")?;
        let world_size = *fact(&facts.world_size, "world size", "launch file")?;
        let devices = fact(&facts.devices, "devices", "launch file")?.clone();
        let hosts = fact(&facts.hosts, "hosts", "launch file")?.clone();
        anyhow::ensure!(
            devices.len() == world_size && hosts.len() == world_size,
            "the engine's facts state {world_size} ranks but {} devices and {} hosts: \
             the launch file places every rank on one device of one host",
            devices.len(),
            hosts.len()
        );
        anyhow::ensure!(
            rank < world_size,
            "the engine's facts place rank {rank} of a group of {world_size}"
        );
        let max_running_requests = *fact(
            &facts.max_running_requests,
            "max_running_requests",
            "launch file",
        )?;
        let max_batch = *fact(&facts.max_batch, "max_batch", "launch file")?;
        let max_num_batched_tokens = *fact(
            &facts.max_num_batched_tokens,
            "max_num_batched_tokens",
            "launch file",
        )?;
        anyhow::ensure!(
            max_batch > 0 && max_num_batched_tokens > 0 && max_running_requests > 0,
            "the engine's batch, token, and request budgets must be positive"
        );

        let ranks: Vec<ExecutorRank> = hosts
            .into_iter()
            .zip(devices)
            .map(|(host, device)| ExecutorRank { host, device })
            .collect();
        let host = ranks
            .get(rank)
            .map(|rank| rank.host.clone())
            .unwrap_or_default();
        let group_config = ExecutorGroupConfig {
            id: ExecutorGroupId(group),
            role: crate::ExecutorGroupRole::Model,
            ranks,
            components: ExecutorGroupConfig::single_component(DEFAULT_COMPONENT, world_size),
            // The ranks are the executor's own jobs: no run is queued.
            queue_depth: 1,
            mem_fraction_static: None,
        };
        group_config.validate()?;

        Ok(Self {
            generation_limits: fact(
                &facts.generation_limits,
                "generation limits",
                "model's own configuration",
            )?
            .clone(),
            max_batch,
            max_num_batched_tokens,
            max_running_requests,
            max_queued_requests: None,
            graph_tiers: fact(&facts.graph_tiers, "graph tiers", "launch file")?.clone(),
            chunked_prefill_size: crate::scheduler::DEFAULT_CHUNKED_PREFILL_SIZE,
            prefix_cache: true,
            boundary_interval: crate::kv_cache::DEFAULT_BOUNDARY_INTERVAL,
            state_interval: crate::kv_cache::DEFAULT_STATE_INTERVAL,
            host_cache: facts.host_cache.clone().unwrap_or_default(),
            worker_groups: vec![group_config],
            data_parallel_size: 1,
            expert_parallel: None,
            model,
            dtype,
            // A main process that names no deployment serves one of its own,
            // so two of them never contest one storage scope.
            deployment: facts.deployment.clone().unwrap_or_else(unnamed_deployment),
            host,
            host_kv_cache: facts.host_kv_cache,
            kv_storage: false,
            canvas_sampling: None,
            draft_tokens: 0,
            bos: *fact(&facts.bos, "bos token", "model's tokenizer")?,
            eos: fact(&facts.eos, "eos tokens", "model's tokenizer")?.clone(),
            end_of_image: *fact(
                &facts.end_of_image,
                "image end token",
                "model's own configuration",
            )?,
            world,
            trial: false,
        })
    }
}

/// A storage scope no other deployment names.
pub(crate) fn unnamed_deployment() -> String {
    format!("deployment-{}", uuid::Uuid::new_v4().simple())
}

impl EngineConfig {
    /// Builds a minimal configuration for the CPU simulation backend.
    ///
    /// Pair this with [`SchedulerHandle::with_executor`].
    #[cfg(feature = "testing")]
    pub fn sim(model: impl Into<String>) -> Self {
        Self {
            generation_limits: crate::sim::SimEngine::generation_limits(),
            max_batch: DEFAULT_MAX_BATCH,
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_running_requests: DEFAULT_MAX_RUNNING_REQUESTS,
            max_queued_requests: None,
            chunked_prefill_size: crate::scheduler::DEFAULT_CHUNKED_PREFILL_SIZE,
            graph_tiers: Vec::new(),
            prefix_cache: true,
            boundary_interval: crate::kv_cache::DEFAULT_BOUNDARY_INTERVAL,
            state_interval: crate::kv_cache::DEFAULT_STATE_INTERVAL,
            host_cache: crate::kv_cache::HostCacheSize::default(),
            worker_groups: vec![ExecutorGroupConfig::placed(
                &["localhost".to_owned()],
                "cpu",
                1,
                2,
                ExecutorGroupConfig::single_component(DEFAULT_COMPONENT, 1),
            )],
            data_parallel_size: 1,
            expert_parallel: None,
            model: model.into(),
            dtype: crate::ModelDtype::BFloat16,
            deployment: unnamed_deployment(),
            host: "localhost".to_owned(),
            host_kv_cache: false,
            kv_storage: false,
            canvas_sampling: None,
            draft_tokens: 0,
            bos: 0,
            // `SimEngine` fabricates this fake EOS id after `text_len` tokens; the
            // scheduler must recognize it to finish a sim request.
            eos: vec![151645],
            end_of_image: 0,
            world: 0,
            trial: false,
        }
    }

    /// Returns the control tokens for a scheduler request.
    fn control_tokens(&self) -> SpecialTokenIds {
        SpecialTokenIds {
            bos: self.bos,
            eos: self.eos.clone(),
            end_of_image: self.end_of_image,
        }
    }
}

/// One DP rank's scheduler, running on a thread of its own, and the front
/// door through which its requests are submitted and controlled.
///
/// The thread owns the [`Scheduler`] and its executor; the handle reaches
/// it only through the scheduler's command channel, and keeps what the
/// scheduler reported at start: the executor capabilities, the media units of
/// each routed media call, the generation limits and the live statistics.
/// The dead latch records a fatal executor or executor failure, after which
/// the handle refuses submissions. Dropping the handle shuts the scheduler
/// down and joins its thread.
pub struct SchedulerHandle {
    commands: CommandSender,
    info: MainProcessInfo,
    /// Media units one call of each routed media call covers at once
    /// (`ExecutorInfo::media_units`).
    media_units: BTreeMap<MediaCall, u32>,
    stats: Arc<SchedulerStats>,
    generation_limits: GenerationLimits,
    /// The executor group whose KV cache the scheduler's pool allocates, by
    /// which a successor world matches the scheduler's prefix cache index;
    /// `None` without a prefix cache.
    cache_group: Option<String>,
    /// The latent seed of what the scheduler's denoising computes, which
    /// keys the trajectories it stores (`Scheduler::latent_seed`).
    latent_seed: u64,
    /// The served model identifier.
    model: String,
    /// Set after `Scheduler::run` returns a fatal exit (executor or executor
    /// failure). A scheduler thread panic does not set it.
    dead: Arc<AtomicBool>,
    /// The first error of that fatal exit (`Scheduler::failure_handle`).
    failure: Arc<std::sync::OnceLock<String>>,
    /// Whether the scheduler started on trial, as a successor world's do;
    /// its commit ends the trial.
    trial: bool,
    /// The device-side submission path of the executor's own crate. A handle
    /// built over a simulated executor has none; the shadow pass and the
    /// in-process submission both need it.
    batches: Option<Arc<dyn Batches>>,
    /// Request ids, shared with the warm-up's view of this scheduler.
    next_id: Arc<AtomicU64>,
    /// Set once a policy instance actually serves (§7.3); see `wait_serving`.
    serving: tokio::sync::watch::Receiver<bool>,
    /// The generation each of this scheduler's batches is stamped with: the
    /// number the control thread's switchover arm last wrote (§7.3). One cell
    /// per rank, shared with the scheduler thread and the in-process submission
    /// path, which reads it once per step.
    generation: Arc<std::sync::atomic::AtomicU64>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl SchedulerHandle {
    /// [`Self::over_batches_telling`] with nothing told of a step's progress.
    ///
    /// # Errors
    ///
    /// Fails as [`Self::over_batches_telling`] does.
    #[cfg(feature = "testing")]
    pub fn over_batches(config: EngineConfig, batches: Arc<dyn Batches>) -> anyhow::Result<Self> {
        Self::over_batches_telling(config, batches, CommandWaker::noop())
    }

    /// Starts one scheduler thread over the executor crate's batch path — the
    /// in-process submission path of §2.4 — and returns its handle, with
    /// `progress` told whenever a step produced events or an answer: what the
    /// thread reading them waits on.
    ///
    /// The pool's capabilities come from what the executor reports
    /// (`Batches::inventory`), and the batches go to `batches` itself. The
    /// scheduler decides nothing until a policy serves it: the daemon's first
    /// `Precopy` and `Switchover` install one (§7.3), and until then a step
    /// forms no batch.
    ///
    /// Fails when the config names more than one rank, when the executor reports
    /// no pool or pools that disagree about a component, when a pool serves fewer
    /// request rows than `max_running_requests` running requests need, or when
    /// the scheduler's thread cannot be started.
    pub fn over_batches_telling(
        config: EngineConfig,
        batches: Arc<dyn Batches>,
        progress: CommandWaker,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            config.data_parallel_size == 1,
            "one handle is one scheduler thread, but the config states {} ranks: start one \
             handle per rank",
            config.data_parallel_size
        );
        let info = crate::executor::ExecutorInfo::from_inventory(batches.inventory())?;
        Self::check_pools(&config, &info)?;
        let runtime = info.runtime_info()?;
        Self::check_rows(&runtime, config.max_running_requests)?;
        let tokens = config.control_tokens();
        let submissions = crate::scheduler::submissions_over(Arc::clone(&batches))?;
        let waker = submissions.waker();
        let scheduler = Scheduler::new(
            submissions,
            tokens,
            Self::scheduler_config(&config),
            config.dtype,
            config.generation_limits.clone(),
        )?;
        let mut scheduler = Self::configured(scheduler, &config)?;
        scheduler.set_progress(progress);
        let mut handle = Self::spawn(scheduler, config.model.clone(), waker)?;
        handle.batches = Some(batches);
        // The serving path's warm-up is the serving path's: it runs once the
        // epoch's first policy serves, on a thread of its own, and startup does
        // not wait for it.
        crate::warmup::after_serving(&handle, &config);
        Ok(handle)
    }

    /// Runs a scheduler over a simulated executor.
    ///
    /// The command waker is a no-op, so a command does not interrupt the
    /// scheduler's idle park. `config.worker_groups` is not read.
    #[cfg(feature = "testing")]
    pub fn with_executor(
        config: EngineConfig,
        executor: Box<dyn Executor>,
    ) -> anyhow::Result<Self> {
        Self::assemble(config, executor, CommandWaker::noop())
    }

    /// Refuses a token runtime whose executors hold fewer request rows than
    /// `max_running_requests` running requests need.
    ///
    /// A token runtime serves those requests only while its executors hold that
    /// many rows beyond the guidance prefix's; the scheduler would otherwise
    /// lower the limit. Executors without a prefix cache fit their request pools
    /// to device storage, and the scheduler bounds their requests by the rows
    /// they report.
    fn check_rows(info: &MainProcessInfo, max_running_requests: usize) -> anyhow::Result<()> {
        let request_rows = info.request_slots as usize;
        let needed_rows = max_running_requests + guidance_prefix_rows(info);
        anyhow::ensure!(
            !info.uses_cache() || request_rows >= needed_rows,
            "the workers hold {request_rows} request rows, but {max_running_requests} running \
             requests need {needed_rows}"
        );
        Ok(())
    }

    /// Refuses a config whose pools are not the ones the executor reports.
    ///
    /// A scheduler configured for a pool the executor does not serve, or for a
    /// rank count it does not serve, would place calls on ranks it does not
    /// have; the config and the executor are two statements of one deployment,
    /// so they are compared before anything runs.
    fn check_pools(
        config: &EngineConfig,
        info: &crate::executor::ExecutorInfo,
    ) -> anyhow::Result<()> {
        for group in &config.worker_groups {
            anyhow::ensure!(
                info.workers.iter().any(|(served, _)| *served == group.id),
                "the config names the pool {}, which the executor does not serve",
                group.id.0
            );
            // `from_inventory` states every pool's rank count; an executor
            // that reports none places the pool on one rank.
            let served = info.layouts.get(&group.id).map_or(1, |layout| layout.ranks);
            anyhow::ensure!(
                served as usize == group.ranks.len(),
                "the config places {} ranks of pool {}, but the executor serves {served}",
                group.ranks.len(),
                group.id.0
            );
        }
        Ok(())
    }

    /// The scheduler configuration `config` states: its budgets, its capture
    /// catalog and its generation sizes.
    fn scheduler_config(config: &EngineConfig) -> SchedulerConfig {
        SchedulerConfig {
            max_batch: config.max_batch,
            max_num_batched_tokens: config.max_num_batched_tokens.max(1),
            max_running_requests: config.max_running_requests.max(1),
            max_queued_requests: config.max_queued_requests,
            chunked_prefill_size: config.chunked_prefill_size.max(1),
            canvas: config.canvas_sampling,
            draft_tokens: config.draft_tokens as usize,
            graph_tiers: config.graph_tiers.clone(),
        }
    }

    /// Builds the scheduler `config` states over `executor` and runs it.
    #[cfg(feature = "testing")]
    fn assemble(
        config: EngineConfig,
        executor: Box<dyn Executor>,
        waker: CommandWaker,
    ) -> anyhow::Result<Self> {
        let tokens = config.control_tokens();
        let scheduler = Scheduler::new(
            crate::scheduler::submissions(executor),
            tokens,
            Self::scheduler_config(&config),
            config.dtype,
            config.generation_limits.clone(),
        )?;
        let scheduler = Self::configured(scheduler, &config)?;
        Self::spawn(scheduler, config.model.clone(), waker)
    }

    /// Applies `config`'s per-world settings to a freshly built scheduler: its
    /// world identity, its prefix reuse and state interval, its prefill groups,
    /// and the host storages its ranks map.
    fn configured(mut scheduler: Scheduler, config: &EngineConfig) -> anyhow::Result<Scheduler> {
        scheduler.set_world(config.world);
        scheduler.set_prefix_reuse(config.prefix_cache);
        scheduler.set_state_interval(config.state_interval);
        scheduler.set_trial(config.trial);
        scheduler.set_prefill_groups(
            config
                .worker_groups
                .iter()
                .filter(|group| group.role == ExecutorGroupRole::Prefill)
                .map(|group| group.id.clone()),
        );
        if config.prefix_cache && config.host_kv_cache {
            scheduler
                .enable_kv_offload(
                    &config.deployment,
                    config.world,
                    &config.host,
                    config.boundary_interval,
                    config.host_cache,
                    config.kv_storage,
                )
                .context("sizing the world's host KV cache")?;
        }
        // Denoising ranks that map host latent pools report them: a
        // preempted request's trajectory (a video's, or an image's) is
        // stored there.
        scheduler.enable_trajectory_store(
            &storage::socket::default_socket(),
            &config.deployment,
            config.world,
            &config.host,
        );
        // Ranks executing video encoders that map host encoder pools report
        // them: a request's encoder outputs are stored and found there.
        scheduler.enable_encoder_store(
            &storage::socket::default_socket(),
            &config.deployment,
            config.world,
            &config.host,
        );
        Ok(scheduler)
    }

    /// Runs `scheduler`, which serves `model`, on a thread of its own.
    /// `waker` is fired after every command is enqueued; it is the
    /// executor's command waker for an executor that parks between steps, or
    /// the no-op waker for one the scheduler polls.
    ///
    /// # Errors
    ///
    /// Fails when the executor's media routing is inconsistent or the thread
    /// cannot be started.
    pub fn spawn(
        scheduler: Scheduler,
        model: impl Into<String>,
        waker: CommandWaker,
    ) -> anyhow::Result<Self> {
        // Requests size their media work by these widths before they are
        // submitted.
        let media_units = scheduler.media_units()?;
        let info = scheduler.info().clone();
        let generation_limits = scheduler.generation_limits().clone();
        let cache_group = scheduler.cache_group();
        let latent_seed = scheduler.latent_seed();
        let stats = scheduler.stats_handle();
        let failure = scheduler.failure_handle();
        // Taken before the scheduler moves onto its thread.
        let serving = scheduler.serving_receiver();
        let generation = scheduler.generation_handle();
        let trial = scheduler.on_trial();

        let (sender, receiver) = crossbeam_channel::unbounded();
        let dead = Arc::new(AtomicBool::new(false));
        let thread = {
            let dead = Arc::clone(&dead);
            std::thread::Builder::new()
                .name("scheduler".into())
                .spawn(move || {
                    if scheduler.run(receiver) {
                        dead.store(true, Ordering::SeqCst);
                    }
                })
                .context("failed to spawn the scheduler thread")?
        };
        Ok(Self {
            commands: CommandSender::with_waker(sender, waker),
            info,
            media_units,
            stats,
            generation_limits,
            cache_group,
            latent_seed,
            model: model.into(),
            dead,
            failure,
            trial,
            batches: None,
            next_id: Arc::new(AtomicU64::new(1)),
            serving,
            generation,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Everything the serving path's warm-up submits through, so a thread of
    /// the engine's can warm this rank without holding the handle it owns.
    pub(crate) fn warmup(&self) -> crate::warmup::Warm {
        crate::warmup::Warm {
            info: self.info.clone(),
            generation_limits: self.generation_limits.clone(),
            next_id: Arc::clone(&self.next_id),
            dead: Arc::clone(&self.dead),
            failure: Arc::clone(&self.failure),
            commands: self.commands.clone(),
        }
    }

    /// The generation this scheduler's batches carry: the one the epoch last
    /// switched to and has not purged (§7.3).
    ///
    /// The cell is shared with the scheduler thread, which reads it once per
    /// step; the control thread's switchover arm writes it while it holds the
    /// generations' lock, so a step never mixes two generations.
    pub fn generation(&self) -> Arc<std::sync::atomic::AtomicU64> {
        Arc::clone(&self.generation)
    }

    /// Records the serving generation, from the control thread's switchover arm.
    pub fn set_generation(&self, generation: u64) {
        self.generation
            .store(generation, std::sync::atomic::Ordering::Relaxed);
    }

    /// The scheduler's serving signal, for a thread that waits on it without
    /// holding the handle ([`crate::warmup::after_serving`]).
    pub fn serving(&self) -> tokio::sync::watch::Receiver<bool> {
        self.serving.clone()
    }

    /// The policy slot of this scheduler, as the control thread holds it (§7.3):
    /// the program serving and the one a precopy prepared.
    ///
    /// The slot lives on the scheduler thread, which alone instantiates, adopts
    /// and reverts a policy; the handle asks that thread.
    pub fn policy_mut(&self) -> crate::handle::PolicySlotHandle {
        crate::handle::PolicySlotHandle::new(self.commands.clone())
    }

    /// One of §7.3's steps 3 to 6, applied at this scheduler's next boundary
    /// where no batch and no submission is in flight, and answered with the step
    /// it applied at.
    pub fn stage(
        &self,
        stage: crate::scheduler::Stage,
    ) -> crossbeam_channel::Receiver<Result<crate::handle::StageApplied, String>> {
        self.commands.stage(stage)
    }

    /// §7.3 step 2's shadow pass for the model generation `generation` (front's
    /// two axes: the model's generation, not a launch's): the following step's
    /// batches also run in shadow, against the resident data and without
    /// writing it.
    ///
    /// Answered once one step's batches have been shadowed, or with the reason
    /// the pass could not run.
    pub fn shadow(&self, generation: u64) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.shadow(generation)
    }

    /// One of the daemon's request commands, applied to the request it names,
    /// answered when it was enqueued.
    ///
    /// A command whose bookkeeping is the connection's — a consumer's position,
    /// a takeover — is answered with that reason: the engine's routing holds the
    /// request's stream, not a scheduler.
    pub fn request_command(
        &self,
        command: proto::request::wire::Command,
    ) -> crossbeam_channel::Receiver<Result<(), String>> {
        let (answer, done) = crossbeam_channel::bounded(1);
        let _ = answer.send(self.commands.request_command(command));
        done
    }

    /// One command of the program-layer update (§2.3, §7.3), as this scheduler
    /// applies it: the stages, at the boundary they name.
    ///
    /// A `Switchover` that names the step its expert group agreed is applied at
    /// that step alone; one that arrives after it is refused rather than applied
    /// at a boundary the other ranks did not take.
    pub fn command(
        &self,
        command: ipc::ProgramCommand,
    ) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.program_command(command)
    }

    /// One of the epoch's engine commands, as this scheduler applies it (§7.3):
    /// a stage is answered when it applies; the commands that belong to the
    /// main process or to the process holding the Program Server are answered
    /// with that reason.
    pub fn engine_command(
        &self,
        command: proto::process::Command,
    ) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.engine_command(command)
    }

    /// What this scheduler has done and what its policy has cost (§12), answered
    /// by the scheduler thread.
    pub fn counters(&self) -> crossbeam_channel::Receiver<crate::handle::Counters> {
        self.commands.counters()
    }

    /// Returns the executor-reported runtime capabilities.
    ///
    /// With several ExecutorGroups this is the executor's merged view
    /// (`ExecutorInfo::runtime_info`), not any single group's report.
    pub fn info(&self) -> &MainProcessInfo {
        &self.info
    }

    /// Returns the media units one call of `call` covers at once: the ranks
    /// of a distributed serving component times its units per rank, or one;
    /// `None` when no executor serves `call`.
    pub fn media_units(&self, call: MediaCall) -> Option<u32> {
        self.media_units.get(&call).copied()
    }

    /// Returns the generation limits resolved against executor capabilities.
    pub fn generation_limits(&self) -> GenerationLimits {
        self.generation_limits.clone()
    }

    /// Returns the executor group whose KV cache the scheduler's pool
    /// allocates, which names its prefix cache index for a successor world;
    /// `None` without a prefix cache.
    pub fn cache_group(&self) -> Option<String> {
        self.cache_group.clone()
    }

    /// Returns the latent seed of what the scheduler's denoising computes,
    /// which keys the trajectories it stores at preemption; zero when it
    /// stores none.
    pub fn latent_seed(&self) -> u64 {
        self.latent_seed
    }

    /// Promotes the scheduler, whose world succeeds another: its executors
    /// adopt the predecessor's cache flags of the units they inherit, and
    /// its pool imports the predecessor's published pages, from which
    /// requests reuse their prefixes. Blocks until it is done.
    ///
    /// # Errors
    ///
    /// Fails when the world's own cache, grown into the predecessor's storage
    /// freed at once when it `grows`, cannot hold the requests the
    /// predecessor hands over (which `recompute` from their tokens or not),
    /// an executor refuses or dies, or the scheduler has stopped.
    pub fn promote(
        &self,
        parked: crate::scheduler::ParkedCache,
        recompute: bool,
        grows: bool,
    ) -> anyhow::Result<()> {
        self.commands
            .promote(parked, recompute, grows)
            .recv()
            .map_err(|_| anyhow::anyhow!("the scheduler stopped while it promoted"))?
            .map_err(|error| anyhow::anyhow!(error))
    }

    /// Returns live scheduler statistics shared with the scheduler thread.
    pub fn stats(&self) -> &Arc<SchedulerStats> {
        &self.stats
    }

    /// Returns the served model identifier.
    pub fn model_name(&self) -> &str {
        &self.model
    }

    /// Allocates the next request id of this scheduler's own requests.
    ///
    /// Ids start at 1 and are unique for the lifetime of this scheduler;
    /// `Relaxed` suffices because only uniqueness is required.
    pub fn next_request_id(&self) -> RequestId {
        RequestId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Returns whether the scheduler stopped on a terminal executor or
    /// executor failure: from its first fatal error on, before its thread
    /// ends the requests it holds.
    pub fn is_dead(&self) -> bool {
        self.dead.load(Ordering::SeqCst) || self.failure.get().is_some()
    }

    /// The first error of the scheduler's fatal stop, once it stops.
    pub fn failure(&self) -> Option<String> {
        self.failure.get().cloned()
    }

    /// Whether the scheduler started on trial, as a successor world's do.
    pub fn starts_on_trial(&self) -> bool {
        self.trial
    }

    /// Submits a request and returns its bounded event stream.
    ///
    /// The returned receiver cancels the request when dropped before its
    /// terminal event.
    ///
    /// # Errors
    ///
    /// Fails with `SubmitError::Dead` once the dead latch is set and with
    /// `SubmitError::Closed` when the command channel no longer accepts
    /// commands.
    pub fn submit(&self, request: impl Into<Request>) -> Result<EventRx, SubmitError> {
        if self.is_dead() {
            return Err(SubmitError::Dead);
        }
        self.commands.submit(request.into())
    }

    /// Submits requests in one command and returns their event streams in
    /// submission order: the scheduler queues every one of them before it
    /// forms its next batch, as it would requests from concurrent clients
    /// that all arrived before that batch. Separate `submit` calls give no
    /// such guarantee.
    ///
    /// # Errors
    ///
    /// As [`Self::submit`].
    pub fn submit_all(&self, requests: Vec<Request>) -> Result<Vec<EventRx>, SubmitError> {
        if self.is_dead() {
            return Err(SubmitError::Dead);
        }
        self.commands.submit_all(requests)
    }

    /// Cancels a request.
    pub fn cancel(&self, id: RequestId) {
        self.commands.cancel(id);
    }

    /// Finishes a request successfully, as when its decoder matched a stop
    /// string.
    pub fn stop(&self, id: RequestId) {
        self.commands.stop(id);
    }

    /// Acknowledges that the first `output_token_count` public tokens of a
    /// request with stop strings complete none of them.
    pub fn acknowledge_at(&self, id: RequestId, output_token_count: usize) {
        self.commands.acknowledge_at(id, output_token_count);
    }

    /// Aborts a request immediately without applying client cancellation
    /// semantics.
    pub fn abort(&self, id: RequestId) {
        self.commands.abort(id);
    }

    /// Parks the scheduler: every token request it holds that resumes from
    /// its generated tokens (`GenerationRequest::is_migratable`) ends as
    /// migratable, a queued one at once, a running one once its calls in
    /// flight drain, after every token it generated; the other requests
    /// finish; then it executes nothing more until resumed. The receiver
    /// gets its prefix cache index and what its migrating requests held
    /// once it parked. When it `lend`s, the world that succeeds it maps its
    /// cache: it lends that world the segments no request in flight
    /// continues from (`Scheduler::park`).
    pub fn park(
        &self,
        lend: bool,
    ) -> crossbeam_channel::Receiver<Result<crate::scheduler::ParkedCache, String>> {
        self.commands.park(lend)
    }

    /// Reports the KV cache each of the scheduler's executor groups holds now,
    /// which a world succeeding this one maps; the receiver gets it.
    pub fn describe_cache(
        &self,
    ) -> crossbeam_channel::Receiver<Result<Vec<crate::scheduler::GroupCache>, String>> {
        self.commands.describe_cache()
    }

    /// Accepted cache lengths of requests that can move to a successor.
    pub fn describe_requests(
        &self,
    ) -> crossbeam_channel::Receiver<Vec<proto::request::wire::RequestPrefix>> {
        self.commands.describe_requests()
    }

    /// §7.3 step 2: prepares `program`'s policy on this scheduler's state,
    /// without putting it in; the answer arrives when the blocking thread that
    /// loaded it is done.
    pub fn precopy_policy(
        &self,
        program: crate::scheduler::PolicyProgram,
        source: std::sync::Arc<dyn crate::scheduler::PolicySource + Send + Sync>,
    ) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.precopy_policy(program, source)
    }

    /// §7.3 step 3: the prepared policy answers from this scheduler's next
    /// batch boundary.
    pub fn switchover_policy(&self) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.switchover_policy()
    }

    /// §7.3 step 5: probation ended; the replaced policy stays until the next
    /// precopy.
    pub fn commit_policy(&self) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.commit_policy()
    }

    /// §7.3 step 4: the previous policy answers again, on the state this
    /// scheduler holds when the stage applies.
    pub fn revert_policy(&self) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.revert_policy()
    }

    /// §7.3 step 6: the module a revert could return to is dropped.
    pub fn purge_policy(&self) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.purge_policy()
    }

    /// §7.3 step 2's failure: a precopied generation is dropped.
    pub fn discard_policy(&self) {
        self.commands.discard_policy();
    }

    /// The step this scheduler reached and §12's cost counters for its policy,
    /// or a timeout when the scheduler does not answer.
    pub fn progress(&self) -> crossbeam_channel::Receiver<crate::scheduler::Progress> {
        self.commands.progress()
    }

    /// Asks this scheduler for the policy failure it recorded since the last
    /// call, if it recorded one (§7.3).
    ///
    /// The answer is polled, not waited for: the caller of the stats tick is
    /// the engine's control thread, and a scheduler that does not answer must
    /// not hold it.
    pub fn take_policy_report(
        &self,
    ) -> crossbeam_channel::Receiver<Option<crate::scheduler::PolicyReverted>> {
        self.commands.take_policy_report()
    }

    /// The state the epoch's policy is instantiated with, as this scheduler
    /// holds it and with `ranks` as the control thread knows them (§7.3);
    /// `None` when the scheduler does not answer in time.
    ///
    /// The state is the mechanism's, not a serving policy's, so an epoch's
    /// first precopy instantiates its policy on it before any policy serves.
    ///
    /// The control thread instantiates its route instance on it and reverts
    /// there after a `route` failure, so a failure costs one boundary, not the
    /// epoch.
    pub fn policy_state(
        &self,
        ranks: Vec<scheduling::guest::view::RankView>,
    ) -> Option<crate::scheduler::TrialState> {
        // A scheduler answers at its next command boundary, and one whose executor has no wake
        // reaches it when its park ends, which is `IDLE_LIVENESS_POLL` away at most.
        const POLICY_STATE_WAIT: std::time::Duration = std::time::Duration::from_secs(5);
        self.commands
            .policy_state(ranks)
            .recv_timeout(POLICY_STATE_WAIT)
            .ok()
            .flatten()
    }

    /// Publishes the committed world's immutable pages on every claimed node.
    pub fn publish_world(&self) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.publish_world()
    }

    /// Pin or publish a scheduler's pages for a shared transfer round.
    pub fn cache_warm(
        &self,
        operation: storage::manifest::cache::Operation,
        together: Arc<crate::HookBarrier>,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.cache_warm(operation, together)
    }

    /// Ends the storage of `worlds` on the nodes of the scheduler's executors,
    /// which the deployment's claims cover; the receiver gets whether it did.
    pub fn retire(&self, worlds: Vec<u64>) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.retire(worlds)
    }

    /// Keeps the weights alone of `worlds` on the nodes of the scheduler's
    /// executors, which the deployment's claims cover, in host storage when
    /// `to_host`; the receiver gets each node's report.
    pub fn keep_weights(
        &self,
        worlds: Vec<u64>,
        to_host: bool,
        cache_keys: Vec<Vec<u8>>,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.keep_weights(worlds, to_host, cache_keys)
    }

    /// Reads the storage ledgers of the nodes of the scheduler's executors,
    /// which the deployment's claims cover; the receiver gets each node's
    /// as JSON.
    pub fn ledgers(&self) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.ledgers()
    }

    /// Halves, or restores when not `halve`, the byte budgets of the links
    /// the deployment's transfers use on the nodes of the scheduler's
    /// executors, which the deployment's claims cover; the receiver gets the
    /// hosts it throttled.
    pub fn throttle(
        &self,
        halve: bool,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.throttle(halve)
    }

    /// Resumes a parked scheduler; the receiver gets whether it did. When
    /// it `reclaim`s, the world it lent cache units to writes none of them
    /// any more, and it takes them back.
    pub fn resume(&self, reclaim: bool) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.resume(reclaim)
    }

    /// Commits the scheduler's world, which succeeded another: its executors
    /// take the predecessor's storage they mapped read-only. The receiver gets
    /// what each executor reported.
    pub fn commit(&self) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.commit()
    }

    /// Revokes every cached prefix page; requests in flight keep theirs.
    pub fn flush_cache(&self) {
        self.commands.flush_cache();
    }

    /// Runs the executor hook `name` with the JSON `payload` on every executor
    /// once the batches in flight complete, together with the other
    /// schedulers of `together` (`HookBarrier`); the receiver gets each
    /// executor's result, or why the hook failed.
    pub fn hook(
        &self,
        name: String,
        payload: String,
        together: Arc<crate::HookBarrier>,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.hook(name, payload, together)
    }

    /// Takes other scheduling limits and policy from the next step on; the
    /// receiver gets whether the scheduler did.
    pub fn configure(
        &self,
        limits: SchedulerConfig,
    ) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.configure(limits)
    }

    /// Rewrites the weights in place from the checkpoint at `model_path`,
    /// the requests in flight continuing as `in_flight` says; the receiver
    /// gets what each executor reported, or why the update failed.
    pub fn update_weights(
        &self,
        model_path: String,
        in_flight: crate::WeightsInFlight,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.update_weights(model_path, in_flight)
    }

    /// Commit background expert copies and preserve the order on rank recovery.
    pub fn commit_experts(
        &self,
        order: BTreeMap<String, Vec<u32>>,
        together: Arc<crate::HookBarrier>,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.commit_experts(order, together)
    }

    /// Gives back or restores KV cache storage; the receiver gets what each
    /// executor reported, or why the change failed.
    pub fn resize_cache(
        &self,
        change: crate::handle::CacheChange,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.resize_cache(change)
    }

    /// Shuts down the scheduler, tears down its executor, and joins its
    /// thread. Idempotent.
    ///
    /// Blocks until the scheduler thread exits; on shutdown `Scheduler::run`
    /// first finishes queued and running requests with `Aborted` and closes
    /// the executor. A poisoned thread lock or a panicked scheduler thread is
    /// logged, not propagated.
    pub fn shutdown(&self) {
        self.commands.shutdown();
        let thread = match self.thread.lock() {
            Ok(mut guard) => guard.take(),
            Err(error) => {
                tracing::warn!("scheduler thread lock poisoned during shutdown");
                error.into_inner().take()
            }
        };
        if let Some(thread) = thread
            && thread.join().is_err()
        {
            tracing::warn!("scheduler thread panicked during shutdown");
        }
    }

    /// Shuts every scheduler of `schedulers` down together: each is asked
    /// before any is joined. Schedulers whose executors share an expert world
    /// must close those executors at once, since a closing expert rank waits
    /// for every peer to leave the union.
    pub fn shutdown_all<'a, I>(schedulers: I)
    where
        I: IntoIterator<Item = &'a SchedulerHandle>,
        I::IntoIter: Clone,
    {
        let schedulers = schedulers.into_iter();
        for scheduler in schedulers.clone() {
            scheduler.commands.shutdown();
        }
        for scheduler in schedulers {
            scheduler.shutdown();
        }
    }
}

impl Drop for SchedulerHandle {
    /// Runs `SchedulerHandle::shutdown`, blocking until the scheduler thread
    /// exits.
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The facts a launch file and the model's configuration state for one
    /// rank of a one-rank group.
    fn rank_facts() -> RankFacts {
        RankFacts {
            model: Some("Qwen/Qwen3-30B-A3B".to_owned()),
            dtype: Some(crate::ModelDtype::BFloat16),
            world: Some(7),
            group: Some("model".to_owned()),
            rank: Some(0),
            world_size: Some(1),
            devices: Some(vec!["cuda:0".to_owned()]),
            hosts: Some(vec!["node-a".to_owned()]),
            max_running_requests: Some(24),
            max_batch: Some(32),
            max_num_batched_tokens: Some(4096),
            graph_tiers: Some(vec![16, 32]),
            bos: Some(1),
            eos: Some(vec![2, 3]),
            end_of_image: Some(4),
            generation_limits: Some(GenerationLimits::default()),
            ..RankFacts::default()
        }
    }

    /// What the facts state is the configuration, and what they do not state
    /// comes from the build configuration and the platform's defaults.
    #[test]
    fn a_rank_of_facts_builds_its_schedulers_configuration() -> anyhow::Result<()> {
        let config = EngineConfig::of(&rank_facts())?;
        assert_eq!(config.max_batch, 32);
        assert_eq!(config.max_num_batched_tokens, 4096);
        assert_eq!(config.max_running_requests, 24);
        assert_eq!(config.world, 7);
        assert_eq!(config.bos, 1);
        assert_eq!(config.eos, [2, 3]);
        assert_eq!(config.graph_tiers, [16, 32]);
        // One group, placed as the launch file places it, with the component
        // the group's ranks divide.
        let [group] = &config.worker_groups[..] else {
            panic!("the facts name one group");
        };
        assert_eq!(group.id.0, "model");
        assert_eq!(group.ranks.len(), 1);
        assert_eq!(group.ranks[0].host, "node-a");
        assert_eq!(group.ranks[0].device, "cuda:0");
        assert_eq!(config.data_parallel_size, 1);
        // Derived: the build configuration's prefill chunk and the platform's
        // prefix cache.
        assert_eq!(
            config.chunked_prefill_size,
            crate::scheduler::DEFAULT_CHUNKED_PREFILL_SIZE
        );
        assert!(config.prefix_cache);
        assert_eq!(config.model, "Qwen/Qwen3-30B-A3B");
        Ok(())
    }

    /// A fact the configuration needs and the facts do not state is refused by
    /// name, with where it comes from: never defaulted to a guess.
    #[test]
    fn a_missing_fact_is_refused_by_name_and_source() {
        let mut facts = rank_facts();
        facts.generation_limits = None;
        let refused = EngineConfig::of(&facts).expect_err("the limits are needed");
        let refused = format!("{refused:#}");
        assert!(refused.contains("generation limits"), "{refused}");
        assert!(refused.contains("model's own configuration"), "{refused}");

        let mut facts = rank_facts();
        facts.world = None;
        let refused = EngineConfig::of(&facts).expect_err("the world is needed");
        let refused = format!("{refused:#}");
        assert!(refused.contains("world"), "{refused}");
        assert!(refused.contains("launch file"), "{refused}");
    }

    /// The rank layout must name one device and one host per rank.
    #[test]
    fn a_rank_layout_that_does_not_cover_every_rank_is_refused() {
        let mut facts = rank_facts();
        facts.world_size = Some(2);
        let refused = EngineConfig::of(&facts).expect_err("two ranks need two devices");
        let refused = format!("{refused:#}");
        assert!(refused.contains("2 ranks"), "{refused}");
    }

    /// A host takes a contiguous block of the ranks, the first hosts take the
    /// remainder, and a rank's device is its local index on its own host.
    #[test]
    fn a_rank_layout_shares_ranks_across_the_named_hosts() {
        let components = ExecutorGroupConfig::single_component("model", 1);

        let one =
            ExecutorGroupConfig::placed(&["compute-0".to_owned()], "cpu", 2, 1, components.clone());
        assert!(
            one.ranks.iter().all(|rank| rank.host == "compute-0"),
            "a rank was placed on a host the rank layout does not name: {:?}",
            one.ranks
        );

        // Three ranks over two hosts: the first host takes the remainder, so
        // every rank is placed and each host's block stays contiguous.
        let two = ExecutorGroupConfig::placed(
            &["compute-0".to_owned(), "compute-1".to_owned()],
            "cuda",
            3,
            1,
            components,
        );
        let hosts: Vec<_> = two.ranks.iter().map(|rank| rank.host.as_str()).collect();
        assert_eq!(hosts, ["compute-0", "compute-0", "compute-1"]);
        // A device name is the rank's place on its own host, so each host
        // starts at its first device.
        let devices: Vec<_> = two.ranks.iter().map(|rank| rank.device.as_str()).collect();
        assert_eq!(devices, ["cuda:0", "cuda:1", "cuda:0"]);
    }

    /// Entries covering the shapes a rank layout must validate: one component
    /// parallel over every rank, one dividing its output into units, and one
    /// placed alone.
    fn mixed_components(rank_count: usize) -> BTreeMap<String, BlockConfig> {
        let members: Vec<_> = (0..rank_count).collect();
        let mut components = ExecutorGroupConfig::single_component("parallel", rank_count);
        components.insert(
            "divided".into(),
            BlockConfig {
                ranks: members,
                tensor_parallel_size: 1,
                pipeline_parallel_size: 1,
                sequence_parallel: Default::default(),
                distribution: Some(BlockDistribution::TemporalUnits),
                units_per_rank: 1,
                weights: Default::default(),
            },
        );
        components.insert("alone".into(), BlockConfig::parallel(vec![0]));
        components
    }

    /// Groups of different sizes validate independently, and a group whose
    /// component spans more ranks than the group holds is refused.
    #[test]
    fn static_components_validate_their_own_parallel_members() -> anyhow::Result<()> {
        let mut workers = Vec::new();
        for (name, degree) in [("first", 1), ("second", 4), ("third", 2)] {
            let mut worker = ExecutorGroupConfig::placed(
                &["localhost".to_owned()],
                "cuda",
                degree,
                2,
                ExecutorGroupConfig::single_component(name, degree),
            );
            worker.id = ExecutorGroupId(name.into());
            workers.push(worker);
        }
        ExecutorGroupConfig::validate_all(&workers)?;

        workers[1].ranks.pop();
        assert!(ExecutorGroupConfig::validate_all(&workers).is_err());
        Ok(())
    }

    #[test]
    fn the_shorthand_blocks_ranks_across_the_hosts_it_is_given() {
        // Eight devices over two four-device hosts, the engine's host first as
        // the CLI lists them. A deployment that places its muxer on rank zero
        // needs the lowest ranks on the first host, and each host must number
        // its own devices from zero.
        let hosts = ["rank-0".to_owned(), "rank-1".to_owned()];
        let worker = ExecutorGroupConfig::placed(&hosts, "cuda", 8, 2, mixed_components(8));

        let rank_layout: Vec<_> = worker
            .ranks
            .iter()
            .map(|rank| (rank.host.as_str(), rank.device.as_str()))
            .collect();
        assert_eq!(
            rank_layout,
            vec![
                ("rank-0", "cuda:0"),
                ("rank-0", "cuda:1"),
                ("rank-0", "cuda:2"),
                ("rank-0", "cuda:3"),
                ("rank-1", "cuda:0"),
                ("rank-1", "cuda:1"),
                ("rank-1", "cuda:2"),
                ("rank-1", "cuda:3"),
            ]
        );
        assert!(worker.validate().is_ok());

        // A component the model spans over the group keeps all eight ranks,
        // and one it places alone stays on rank zero, which is the engine's
        // host.
        for entry in ["parallel", "divided"] {
            assert_eq!(
                worker.components[entry].ranks.len(),
                8,
                "{entry} spans the executor group"
            );
        }
        assert_eq!(worker.components["alone"].ranks, vec![0]);
    }

    /// The two-rank groups of four schedulers over two four-device hosts
    /// fill the engine's host first, and each group keeps its ranks on one
    /// host.
    #[test]
    fn the_data_parallel_shorthand_places_engine_cores_in_host_blocks() {
        let hosts = ["rank-0".to_owned(), "rank-1".to_owned()];
        let groups = ExecutorGroupConfig::data_parallel(
            &hosts,
            "cuda",
            4,
            2,
            2,
            ExecutorGroupConfig::single_component("model", 2),
        );

        let rank_layout: Vec<_> = groups
            .iter()
            .map(|worker| {
                (
                    worker.id.to_string(),
                    worker
                        .ranks
                        .iter()
                        .map(|rank| format!("{}/{}", rank.host, rank.device))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        assert_eq!(
            rank_layout,
            vec![
                (
                    "model-0".into(),
                    vec!["rank-0/cuda:0".into(), "rank-0/cuda:1".into()]
                ),
                (
                    "model-1".into(),
                    vec!["rank-0/cuda:2".into(), "rank-0/cuda:3".into()]
                ),
                (
                    "model-2".into(),
                    vec!["rank-1/cuda:0".into(), "rank-1/cuda:1".into()]
                ),
                (
                    "model-3".into(),
                    vec!["rank-1/cuda:2".into(), "rank-1/cuda:3".into()]
                ),
            ]
        );
        assert!(ExecutorGroupConfig::validate_all(&groups).is_ok());

        // One scheduler keeps the single-group identity.
        let single = ExecutorGroupConfig::data_parallel(
            &hosts,
            "cuda",
            1,
            2,
            2,
            ExecutorGroupConfig::single_component("model", 2),
        );
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].id.to_string(), "model");
    }

    #[test]
    fn a_rank_count_that_does_not_divide_its_hosts_still_places_every_rank() {
        // The first hosts take the remainder, which keeps each host's block
        // contiguous and rank zero on the engine's host.
        let hosts = ["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let worker = ExecutorGroupConfig::placed(
            &hosts,
            "cuda",
            8,
            2,
            ExecutorGroupConfig::single_component(DEFAULT_COMPONENT, 8),
        );

        let hosts: Vec<_> = worker.ranks.iter().map(|rank| rank.host.as_str()).collect();
        assert_eq!(hosts, vec!["a", "a", "a", "b", "b", "b", "c", "c"]);
        assert_eq!(worker.ranks.len(), 8);
    }

    /// Reordering a component's ranks is a valid rank layout; repeating a rank
    /// is not.
    #[test]
    fn entry_geometry_uses_unique_ordered_rank_members() {
        let mut worker = ExecutorGroupConfig::placed(
            &["localhost".to_owned()],
            "cuda",
            4,
            2,
            mixed_components(4),
        );
        assert!(worker.validate().is_ok());

        worker
            .components
            .get_mut("parallel")
            .unwrap()
            .ranks
            .swap(0, 3);
        assert!(worker.validate().is_ok());

        worker.components.get_mut("parallel").unwrap().ranks[1] = 3;
        assert!(worker.validate().is_err());
    }

    #[test]
    fn static_bindings_reject_duplicate_identities_and_allow_component_replicas() {
        let worker = ExecutorGroupConfig::placed(
            &["localhost".to_owned()],
            "cuda",
            1,
            2,
            ExecutorGroupConfig::single_component(DEFAULT_COMPONENT, 1),
        );
        let mut other = worker.clone();
        assert!(ExecutorGroupConfig::validate_all(&[worker.clone(), other.clone()]).is_err());

        // Distinct identities on the same device validate, whether the groups
        // replicate one component name or hold different ones.
        other.id = ExecutorGroupId("other".into());
        assert!(ExecutorGroupConfig::validate_all(&[worker.clone(), other.clone()]).is_ok());

        let entry = other.components.remove(DEFAULT_COMPONENT).unwrap();
        other.components.insert("divided".into(), entry);
        assert!(ExecutorGroupConfig::validate_all(&[worker, other]).is_ok());
    }
}
