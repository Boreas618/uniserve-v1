//! The deployment's composition: its worker groups, its schedulers and their
//! handles.
//!
//! A deployment runs one scheduler per DP rank. [`SchedulerHandle`] owns a
//! scheduler's thread and is the front door request producers submit and
//! control requests through.
//!
//! [`WorkerGroupConfig`] is the static deployment description: each WorkerGroup has
//! an ordered list of physical [`WorkerRank`]s and named components whose
//! `ranks` index into that list. [`EngineConfig`] adds scheduler budgets,
//! transfer bindings, and rank launch defaults. `SchedulerHandle::start_all`
//! turns that configuration into launched worker processes and one scheduler
//! per DP rank; `SchedulerHandle::with_executor` accepts an executor built
//! elsewhere, such as the CPU simulation backend.

#[cfg(any(feature = "testing", test))]
use ipc::DEFAULT_COMPONENT;
use ipc::MediaCall;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::executor::{Executor, TransferConfig, WorkerGroupId};
use crate::handle::{CommandSender, EventRx, SubmitError};
#[cfg(feature = "testing")]
use crate::scheduler::{
    DEFAULT_CHUNKED_PREFILL_SIZE, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_RUNNING_REQUESTS,
};
use crate::scheduler::{MAX_GUIDANCE_PREFIX_ROWS, guidance_prefix_rows};
use crate::scheduler::{Scheduler, SpecialTokenIds};
use crate::scheduler::{SchedulerConfig, SchedulerStats};
use crate::worker::{WorkerExecutor, WorkerGroup, WorkerProcessArgs};
use crate::{CommandWaker, GenerationLimits, Request};
use anyhow::Context as _;
#[cfg(test)]
use ipc::ComponentDistribution;
use ipc::WorkerInfo;
use ipc::{ComponentConfig, ParallelConfig, RequestId};
use proto::request::{ExpertExchange, SchedulingPolicy};

/// One cooperative member's physical execution endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkerRank {
    /// Host identity: the daemon of the node with this host identity starts
    /// the rank.
    pub host: String,
    /// Device string, such as `cuda:0` for a GPU rank or `cpu` for a CPU worker.
    /// On the edges `TransferConfig::with_worker_defaults` derives, device
    /// products use CUDA VMM only between distinct ranks whose devices both
    /// start with `cuda:`.
    pub device: String,
}

/// The numerical work a worker group owns in a deployment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerGroupRole {
    /// Model capabilities, request state, and (when present) attention/KV.
    #[default]
    Model,
    /// Routed experts shared by the deployment's schedulers.
    Experts,
}

/// Static computation components and their ordered physical rank membership.
///
/// A deployment file ([`crate::Deployment`]) or the command-line shorthand
/// ([`WorkerGroupConfig::data_parallel`]) builds them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WorkerGroupConfig {
    pub id: WorkerGroupId,
    pub role: WorkerGroupRole,
    /// Ordered physical members; a rank's position is its rank number within
    /// the group.
    pub ranks: Vec<WorkerRank>,
    /// Components keyed by name, each listing indices into `ranks`.
    pub components: BTreeMap<String, ComponentConfig>,
    /// Maximum number of physical runs in flight per rank, launched as
    /// `WorkerProcessArgs::queue_depth`.
    pub queue_depth: usize,
    /// Per-rank share of device memory available to this worker process for
    /// static allocations (weights and KV cache).
    ///
    /// When set, `SchedulerHandle::start_all` launches the group's ranks with it as
    /// `WorkerProcessArgs::mem_fraction_static` instead of the engine-wide
    /// default, which `--mem-fraction-static` sets.
    ///
    /// Distinct WorkerGroups may share one physical GPU. Their configured
    /// shares must leave enough aggregate headroom for the device runtime;
    /// validation checks each group alone, never the sum across groups, and
    /// accepts the range [`WorkerGroupConfig::validate_mem_fraction_static`] states.
    pub mem_fraction_static: Option<f64>,
}

impl WorkerGroupConfig {
    /// Validates one WorkerGroup in isolation.
    ///
    /// Fails when the identity is not a valid `WorkerGroupId`, `queue_depth` is
    /// zero, `validate_mem_fraction_static` refuses a set `mem_fraction_static`, or
    /// `validate_members` rejects the ranks and components.
    pub fn validate(&self) -> anyhow::Result<()> {
        WorkerGroupId::new(self.id.0.clone())?;
        anyhow::ensure!(
            self.queue_depth > 0,
            "worker {} queue depth must be positive",
            self.id
        );
        if let Some(fraction) = self.mem_fraction_static {
            Self::validate_mem_fraction_static(fraction)
                .map_err(|error| anyhow::anyhow!("worker {}: {error}", self.id))?;
        }
        Self::validate_members(&self.ranks, &self.components)?;
        anyhow::ensure!(
            self.components.is_empty() == (self.role == WorkerGroupRole::Experts),
            "model workers require components; expert workers have no request components"
        );
        Ok(())
    }

    /// Checks a per-process static memory fraction: a group's
    /// `mem_fraction_static` or the engine-wide
    /// `WorkerProcessArgs::mem_fraction_static` default.
    ///
    /// Each rank grants itself this share of every device's total storage,
    /// less what its process already holds and never more than the device
    /// has free, so any finite value in (0, 1] is meaningful, 1 granting the
    /// whole device. The worker refuses any other value at startup; checking
    /// here refuses it before a rank is launched.
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
    /// valid for their worker role; each needs a nonempty name, a nonempty list of distinct
    /// in-range rank indices, a `parallel_config` whose
    /// `ParallelConfig::world_size` succeeds, and a positive `units_per_rank`.
    /// A component without a distribution must have a parallel world size
    /// equal to its rank count; one with a distribution must have world size
    /// one.
    ///
    /// `WorkerGroup::spawn_all` calls this again for each group before
    /// starting its ranks.
    pub fn validate_members(
        ranks: &[WorkerRank],
        components: &BTreeMap<String, ComponentConfig>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(!ranks.is_empty(), "worker requires rank members");
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
                "worker repeats a physical device"
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

    /// Validates unique WorkerGroup identities and every local placement.
    ///
    /// Component names may repeat across WorkerGroups. Repeated names are
    /// replicas of the same numerical component; the scheduler binds each
    /// request to one of them and keeps that affinity for the request lifetime.
    ///
    /// Physical devices may repeat across groups; only `validate_members`
    /// checks device reuse, and it does so within one group.
    pub fn validate_all(workers: &[Self]) -> anyhow::Result<()> {
        anyhow::ensure!(!workers.is_empty(), "engine requires workers");
        let mut ids = BTreeSet::new();
        for worker in workers {
            worker.validate()?;
            anyhow::ensure!(ids.insert(&worker.id), "WorkerGroup identity is repeated");
        }
        Ok(())
    }

    /// One component tensor-parallel over every rank.
    ///
    /// The `uniserve` CLI uses this placement when no `--worker-groups` configuration
    /// is given, and the default engine settings and the simulation use it. A
    /// deployment with several components, or with other partitions, writes
    /// its own configuration.
    pub fn single_component(name: &str, rank_count: usize) -> BTreeMap<String, ComponentConfig> {
        BTreeMap::from([(
            name.into(),
            ComponentConfig::parallel(
                (0..rank_count).collect(),
                ParallelConfig {
                    tensor_parallel_size: rank_count,
                    ..Default::default()
                },
            ),
        )])
    }

    /// Expands the single-group CLI shorthand into explicit device membership
    /// across the named hosts, carrying the given components.
    ///
    /// Ranks are assigned in blocks, so the lowest ranks stay on the first
    /// host, and each host numbers its devices from zero. A placement on one
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
        components: BTreeMap<String, ComponentConfig>,
    ) -> Self {
        // A host takes its share of the ranks, and the first hosts take the
        // remainder, so a count that does not divide evenly still places every
        // rank and keeps each host's block contiguous.
        let host_count = hosts.len().max(1);
        let share = rank_count / host_count;
        let remainder = rank_count % host_count;
        let mut placement = Vec::with_capacity(rank_count);
        for (index, host) in hosts.iter().enumerate() {
            let count = share + usize::from(index < remainder);
            for local in 0..count {
                placement.push(WorkerRank {
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
            id: WorkerGroupId("model".into()),
            role: WorkerGroupRole::Model,
            ranks: placement,
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
    /// [`WorkerGroupConfig::placed`] places them, in blocks over `hosts`, and
    /// consecutive runs of `ranks_per_core` form the groups, so the first
    /// schedulers fill the head's host first. The group of one scheduler
    /// is `model`; the group of core `i` of several is `model-<i>`.
    pub fn data_parallel(
        hosts: &[String],
        device: &str,
        cores: usize,
        ranks_per_core: usize,
        queue_depth: usize,
        components: BTreeMap<String, ComponentConfig>,
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
                id: WorkerGroupId(format!("model-{index}")),
                role: WorkerGroupRole::Model,
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
    /// Model generation requirements, intersected with loaded worker capacities.
    pub generation_limits: GenerationLimits,
    /// Maximum number of calls assembled into a single forward batch.
    pub max_batch: usize,
    /// Maximum number of tokens scheduled in one engine step.
    pub max_num_batched_tokens: usize,
    /// Maximum number of concurrently running requests. `SchedulerHandle::start_all`
    /// sizes every worker's request pool from it.
    pub max_running_requests: usize,
    /// Most requests, waiting or finished with undelivered events, the
    /// scheduler holds before it rejects a submission as overloaded; `None`
    /// queues without bound.
    pub max_queued_requests: Option<usize>,
    /// Per-request token ceiling for one prefill chunk.
    pub chunked_prefill_size: usize,
    /// Reuse and retain prompt cache prefixes across requests. Disabling this
    /// does not remove the cache storage needed within an active request.
    pub prefix_cache: bool,
    /// Waiting queue policy for admitting/scheduling requests.
    pub schedule_policy: SchedulingPolicy,
    /// Static computation components and physical rank membership.
    pub worker_groups: Vec<WorkerGroupConfig>,
    /// Per-edge data-plane transfer backend selection (`placement_config.transfer`), e.g.
    /// `encoder->prefill=posix_shm,prefill->decode=cuda_vmm`. Participating worker
    /// ranks receive the selected transport. Before process launch,
    /// `SchedulerHandle::start_all` binds a default for every ordered rank pair, within
    /// and across WorkerGroups, that no configured edge covers, derived from
    /// rank host/device coordinates.
    pub transfer: TransferConfig,
    /// Number of schedulers `worker_groups` forms, as equal consecutive
    /// blocks of model groups ([`SchedulerHandle::start_all`]). One means the
    /// groups form one scheduler.
    pub data_parallel_size: usize,
    /// Whether the schedulers shard the model's routed experts, and how
    /// they exchange tokens: each scheduler's group is then one worker,
    /// keeps its share of every expert layer and exchanges tokens with the
    /// others at those layers, while attention and every other layer stay
    /// data-parallel. `None` keeps every expert on every scheduler.
    pub expert_parallel: Option<ExpertExchange>,
    /// Rank launch defaults refined by each WorkerGroup configuration.
    ///
    /// Every constructor also reads the model identifier (`model`) and
    /// `model_dtype` from here.
    pub worker_process: WorkerProcessArgs,
    /// Beginning-of-sequence token identifier.
    pub bos: u32,
    /// Token identifiers that terminate generation.
    pub eos: Vec<u32>,
    /// Token identifier that terminates encoded image content.
    pub end_of_image: u32,
}

impl EngineConfig {
    /// Builds a minimal configuration for the CPU simulation backend.
    ///
    /// Pair this with [`SchedulerHandle::with_executor`]:
    /// [`SchedulerHandle::start_all`] cannot build a `Sim` backend because it
    /// has no spawnable worker process.
    #[cfg(feature = "testing")]
    pub fn sim(model: impl Into<String>) -> Self {
        let worker_process = WorkerProcessArgs {
            model: model.into(),
            ranks: WorkerGroupConfig::placed(
                &["localhost".to_owned()],
                "cpu",
                1,
                2,
                WorkerGroupConfig::single_component(DEFAULT_COMPONENT, 1),
            )
            .ranks,
            block_size: Some(64),
            queue_depth: 2,
            max_batch_calls: DEFAULT_MAX_BATCH as u32,
            max_batch_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS as u32,
            ..WorkerProcessArgs::default()
        };
        Self {
            generation_limits: crate::sim::SimEngine::generation_limits(),
            max_batch: DEFAULT_MAX_BATCH,
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_running_requests: DEFAULT_MAX_RUNNING_REQUESTS,
            max_queued_requests: None,
            chunked_prefill_size: DEFAULT_CHUNKED_PREFILL_SIZE,
            prefix_cache: true,
            schedule_policy: SchedulingPolicy::Fcfs,
            worker_groups: vec![WorkerGroupConfig::placed(
                &["localhost".to_owned()],
                "cpu",
                1,
                2,
                WorkerGroupConfig::single_component(DEFAULT_COMPONENT, 1),
            )],
            transfer: TransferConfig::default(),
            data_parallel_size: 1,
            expert_parallel: None,
            worker_process,
            bos: 0,
            // `SimEngine` fabricates this fake EOS id after `text_len` tokens; the
            // scheduler must recognize it to finish a sim request.
            eos: vec![151645],
            end_of_image: 0,
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
/// scheduler reported at start: the worker capabilities, the media units of
/// each routed media call, the generation limits and the live statistics.
/// The dead latch records a fatal executor or worker failure, after which
/// the handle refuses submissions. Dropping the handle shuts the scheduler
/// down and joins its thread.
pub struct SchedulerHandle {
    commands: CommandSender,
    info: WorkerInfo,
    /// Media units one call of each routed media call covers at once
    /// (`ExecutorInfo::media_units`).
    media_units: BTreeMap<MediaCall, u32>,
    stats: Arc<SchedulerStats>,
    generation_limits: GenerationLimits,
    /// The served model identifier.
    model: String,
    /// Set after `Scheduler::run` returns a fatal exit (executor or worker
    /// failure). A scheduler thread panic does not set it.
    dead: Arc<AtomicBool>,
    next_id: AtomicU64,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl SchedulerHandle {
    /// Launches the deployment's workers and runs one scheduler per DP rank,
    /// `config.data_parallel_size` of them, each over its own copy of the
    /// deployment's model worker groups.
    ///
    /// `config.worker_groups` lists the schedulers' model groups as equal
    /// consecutive blocks. Each scheduler gets its own thread, executor,
    /// cache pool, prefix cache and request rows over its own groups, the
    /// way SGLang runs one scheduler per data-parallel rank behind its
    /// `DataParallelController` and vLLM one engine core per data-parallel
    /// rank. Every group of every scheduler launches through one
    /// `WorkerGroup::spawn_all`, so the schedulers load their models at the
    /// same time and one daemon per node serves all of them.
    ///
    /// Schedulers exchange no products: every configured transfer edge must
    /// stay inside one scheduler, and each binds its default edges over its
    /// own groups only.
    ///
    /// Fails when the size is zero or does not divide the groups, when the
    /// schedulers' groups are not the same deployment on different ranks
    /// (group count, rank counts, components, queue depth and
    /// mem_fraction_static must match position by position), when a
    /// configured edge crosses schedulers or names an unconfigured worker
    /// group, when the workers of a token runtime hold too few request rows
    /// for `max_running_requests` running requests, or when a group cannot
    /// be launched, a scheduler cannot be built or its warmup fails.
    pub fn start_all(config: EngineConfig) -> anyhow::Result<Vec<Self>> {
        let size = config.data_parallel_size;
        WorkerGroupConfig::validate_all(&config.worker_groups)?;
        let (model_workers, expert_workers): (Vec<_>, Vec<_>) = config
            .worker_groups
            .iter()
            .cloned()
            .partition(|worker| worker.role == WorkerGroupRole::Model);
        anyhow::ensure!(
            size > 0 && !model_workers.is_empty() && model_workers.len().is_multiple_of(size),
            "{} worker groups do not form {size} equal schedulers",
            model_workers.len()
        );
        let core_groups: Vec<&[WorkerGroupConfig]> =
            model_workers.chunks(model_workers.len() / size).collect();
        for core_group in &core_groups[1..] {
            for (group, first) in core_group.iter().zip(core_groups[0]) {
                anyhow::ensure!(
                    group.ranks.len() == first.ranks.len()
                        && group.components == first.components
                        && group.queue_depth == first.queue_depth
                        && group.mem_fraction_static == first.mem_fraction_static,
                    "schedulers must place the same groups: worker group {} \
                     differs from worker group {}",
                    group.id,
                    first.id
                );
            }
        }

        let disaggregated = !expert_workers.is_empty();
        anyhow::ensure!(
            config.worker_process.expert_microbatches >= 1
                && (config.worker_process.expert_microbatches == 1 || disaggregated),
            "expert microbatches must number at least one and require disaggregated placement \
             when greater than one"
        );
        if disaggregated {
            anyhow::ensure!(
                matches!(
                    config.expert_parallel,
                    Some(ExpertExchange::DeepEp | ExpertExchange::MegaMoe)
                ) && core_groups.iter().all(|core_group| core_group.len() == 1),
                "disaggregated experts require deepep or megamoe and one model group per scheduler"
            );
        } else {
            anyhow::ensure!(
                config.expert_parallel.is_none()
                    || (config.expert_parallel != Some(ExpertExchange::DeepEp)
                        && size > 1
                        && core_groups.iter().all(|core_group| {
                            core_group.len() == 1 && core_group[0].ranks.len() == 1
                        })),
                "colocated expert parallelism requires two or more schedulers of one \
                 one-worker group each; deepep requires dedicated expert workers"
            );
        }
        let attention_ranks = if disaggregated {
            model_workers
                .iter()
                .map(|worker| worker.ranks.len())
                .sum::<usize>()
        } else {
            0
        };
        let expert_world_size = if disaggregated {
            attention_ranks
                + expert_workers
                    .iter()
                    .map(|worker| worker.ranks.len())
                    .sum::<usize>()
        } else {
            size
        };

        // Each scheduler resolves its transfer edges over its own groups.
        // The edges `with_worker_defaults` adds only name the core's groups,
        // so checking the configured ones covers every edge.
        let mut transfers = Vec::with_capacity(size);
        let mut assigned = 0;
        for core_group in &core_groups {
            let members = core_group
                .iter()
                .map(|worker| worker.id.clone())
                .collect::<BTreeSet<_>>();
            let edges = config
                .transfer
                .edges
                .iter()
                .filter(|edge| members.contains(&edge.source_worker))
                .cloned()
                .collect::<Vec<_>>();
            for edge in &edges {
                anyhow::ensure!(
                    members.contains(&edge.destination_worker),
                    "transfer edge {} -> {} leaves its source's scheduler",
                    edge.source_worker,
                    edge.destination_worker
                );
            }
            assigned += edges.len();
            transfers.push(
                TransferConfig {
                    edges,
                    ..config.transfer.clone()
                }
                .with_worker_defaults(core_group)?,
            );
        }
        anyhow::ensure!(
            assigned == config.transfer.edges.len(),
            "transfer edge names an unconfigured worker"
        );

        // Every group holds a request row for each running request the
        // scheduler may keep resident, plus the rows a runtime keeps outside
        // that limit. Only the loaded capabilities decide that reserve
        // (`guidance_prefix_rows`), so the launch sizes for the largest one.
        let max_running_requests = config.max_running_requests.max(1);
        let max_request_pool_size = u32::try_from(max_running_requests + MAX_GUIDANCE_PREFIX_ROWS)
            .context("max_running_requests exceeds the worker request-pool field")?;

        // Each group launches from the shared defaults with its own identity,
        // placement, components, queue depth, and mem_fraction_static, plus its
        // scheduler's resolved transfer edges and the components of every
        // group of its scheduler (`peers`), from which the group resolves
        // the ranks, possibly in other groups, that read its products.
        let mut arguments = Vec::with_capacity(config.worker_groups.len());
        let mut union_rank = 0;
        for (core_group, transfer) in core_groups.iter().zip(&transfers) {
            let peers = core_group
                .iter()
                .map(|worker| (worker.id.to_string(), worker.components.clone()))
                .collect::<BTreeMap<_, _>>();
            for worker in *core_group {
                arguments.push(WorkerProcessArgs {
                    worker_id: worker.id.to_string(),
                    role: worker.role,
                    ranks: worker.ranks.clone(),
                    components: worker.components.clone(),
                    peers: peers.clone(),
                    queue_depth: worker.queue_depth,
                    max_request_pool_size,
                    mem_fraction_static: worker
                        .mem_fraction_static
                        .unwrap_or(config.worker_process.mem_fraction_static),
                    transfer: transfer.clone(),
                    expert_parallel: config.expert_parallel.map(|exchange| {
                        crate::worker::ExpertParallelPlacement {
                            rank: union_rank as u32,
                            size: expert_world_size as u32,
                            attention_ranks: attention_ranks as u32,
                            address: None,
                            exchange,
                        }
                    }),
                    ..config.worker_process.clone()
                });
                union_rank += worker.ranks.len();
            }
        }
        for worker in &expert_workers {
            arguments.push(WorkerProcessArgs {
                worker_id: worker.id.to_string(),
                role: worker.role,
                ranks: worker.ranks.clone(),
                components: worker.components.clone(),
                peers: BTreeMap::new(),
                queue_depth: worker.queue_depth,
                max_request_pool_size,
                mem_fraction_static: worker
                    .mem_fraction_static
                    .unwrap_or(config.worker_process.mem_fraction_static),
                transfer: transfers[0].clone(),
                expert_parallel: config.expert_parallel.map(|exchange| {
                    crate::worker::ExpertParallelPlacement {
                        rank: union_rank as u32,
                        size: expert_world_size as u32,
                        attention_ranks: attention_ranks as u32,
                        address: None,
                        exchange,
                    }
                }),
                ..config.worker_process.clone()
            });
            union_rank += worker.ranks.len();
        }

        // `spawn_all` returns groups in argument order, which pairs each group
        // with its identity and, block by block, with its scheduler.
        let launched = WorkerGroup::spawn_all(arguments)?;
        let mut groups = model_workers
            .iter()
            .chain(&expert_workers)
            .map(|worker| worker.id.clone())
            .zip(launched)
            .collect::<Vec<_>>();
        // Shared expert groups have one lifecycle owner. They advertise no
        // request capabilities, so they never become a scheduler's own.
        let mut shared = groups.split_off(model_workers.len());
        let mut groups = groups.into_iter();
        let mut cores = Vec::with_capacity(size);
        for (index, (core_group, transfer)) in core_groups.iter().zip(transfers).enumerate() {
            let mut workers = groups.by_ref().take(core_group.len()).collect::<Vec<_>>();
            if index == 0 {
                workers.append(&mut shared);
            }
            let executor = WorkerExecutor::try_new(workers, transfer.clone())?;

            // A token runtime serves `max_running_requests` running requests only while
            // its workers hold that many request rows beyond the reserve; the
            // scheduler would otherwise lower the limit. Workers without a prefix
            // cache fit their request pools to device storage, and the
            // scheduler bounds their requests by the rows they report.
            let info = executor.info().runtime_info()?;
            let request_rows = info.request_slots as usize;
            let needed_rows = max_running_requests + guidance_prefix_rows(&info);
            anyhow::ensure!(
                !info.uses_cache() || request_rows >= needed_rows,
                "the workers hold {request_rows} request rows, but {max_running_requests} running \
                 requests need {needed_rows}"
            );

            let waker = executor.command_waker();
            let mut placement = core_group.to_vec();
            if index == 0 {
                placement.extend(expert_workers.iter().cloned());
            }
            let core_config = EngineConfig {
                worker_groups: placement,
                transfer,
                data_parallel_size: 1,
                expert_parallel: None,
                ..config.clone()
            };
            let scheduler = Self::assemble(core_config.clone(), Box::new(executor), waker)?;
            // Each scheduler's serving path runs the requests its workers'
            // reports call for before the scheduler serves.
            crate::warmup::warm(&scheduler, &core_config)?;
            cores.push(scheduler);
        }
        Ok(cores)
    }

    /// Runs a scheduler over an executor supplied by a higher composition
    /// layer, such as the CPU simulation backend.
    ///
    /// The command waker is a no-op, so a command does not interrupt the
    /// scheduler's idle park. Neither `config.worker_groups` nor
    /// `config.transfer` is read.
    pub fn with_executor(
        config: EngineConfig,
        executor: Box<dyn Executor>,
    ) -> anyhow::Result<Self> {
        Self::assemble(config, executor, CommandWaker::noop())
    }

    /// Builds the scheduler `config` states over `executor` and runs it.
    fn assemble(
        config: EngineConfig,
        executor: Box<dyn Executor>,
        waker: CommandWaker,
    ) -> anyhow::Result<Self> {
        let mut scheduler = Scheduler::new(
            executor,
            config.control_tokens(),
            SchedulerConfig {
                max_batch: config.max_batch,
                max_num_batched_tokens: config.max_num_batched_tokens.max(1),
                max_running_requests: config.max_running_requests.max(1),
                max_queued_requests: config.max_queued_requests,
                chunked_prefill_size: config.chunked_prefill_size.max(1),
                policy: config.schedule_policy,
                canvas: config.worker_process.canvas_sampling,
            },
            config.worker_process.dtype,
            config.generation_limits,
        )?;
        scheduler.set_prefix_reuse(config.prefix_cache);
        if config.prefix_cache && config.worker_process.page_tier {
            scheduler.enable_page_tier(
                &config.worker_process.deployment,
                &config.worker_process.host,
            );
        }
        Self::spawn(scheduler, config.worker_process.model, waker)
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
        let stats = scheduler.stats_handle();

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
            model: model.into(),
            dead,
            next_id: AtomicU64::new(1),
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Returns the worker-reported runtime capabilities.
    ///
    /// With several WorkerGroups this is the executor's merged view
    /// (`ExecutorInfo::runtime_info`), not any single group's report.
    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Returns the media units one call of `call` covers at once: the ranks
    /// of a distributed serving component times its units per rank, or one;
    /// `None` when no worker serves `call`.
    pub fn media_units(&self, call: MediaCall) -> Option<u32> {
        self.media_units.get(&call).copied()
    }

    /// Returns the generation limits resolved against worker capabilities.
    pub fn generation_limits(&self) -> GenerationLimits {
        self.generation_limits.clone()
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

    /// Returns whether the scheduler stopped on a terminal worker or
    /// executor failure.
    pub fn is_dead(&self) -> bool {
        self.dead.load(Ordering::SeqCst)
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

    /// Entries covering the shapes a placement must validate: one component
    /// parallel over every rank, one dividing its output into units, and one
    /// placed alone.
    fn mixed_components(rank_count: usize) -> BTreeMap<String, ComponentConfig> {
        let members: Vec<_> = (0..rank_count).collect();
        let mut components = WorkerGroupConfig::single_component("parallel", rank_count);
        components.insert(
            "divided".into(),
            ComponentConfig {
                ranks: members,
                parallel_config: ParallelConfig::default(),
                distribution: Some(ComponentDistribution::TemporalUnits),
                units_per_rank: 1,
                weights: Default::default(),
            },
        );
        components.insert(
            "alone".into(),
            ComponentConfig::parallel(vec![0], ParallelConfig::default()),
        );
        components
    }

    /// Groups of different sizes validate independently, and a group whose
    /// component spans more ranks than the group holds is refused.
    #[test]
    fn static_components_validate_their_own_parallel_members() -> anyhow::Result<()> {
        let mut workers = Vec::new();
        for (name, degree) in [("first", 1), ("second", 4), ("third", 2)] {
            let mut worker = WorkerGroupConfig::placed(
                &["localhost".to_owned()],
                "cuda",
                degree,
                2,
                WorkerGroupConfig::single_component(name, degree),
            );
            worker.id = WorkerGroupId(name.into());
            workers.push(worker);
        }
        WorkerGroupConfig::validate_all(&workers)?;

        workers[1].ranks.pop();
        assert!(WorkerGroupConfig::validate_all(&workers).is_err());
        Ok(())
    }

    #[test]
    fn the_shorthand_blocks_ranks_across_the_hosts_it_is_given() {
        // Eight devices over two four-device hosts, the head's host first as
        // the CLI lists them. `WorkerGroup` refuses a muxer off the head's
        // host, so a deployment that places its muxer on rank zero needs the
        // lowest ranks on the first host, and each host must number its own
        // devices from zero.
        let hosts = ["rank-0".to_owned(), "rank-1".to_owned()];
        let worker = WorkerGroupConfig::placed(&hosts, "cuda", 8, 2, mixed_components(8));

        let placement: Vec<_> = worker
            .ranks
            .iter()
            .map(|rank| (rank.host.as_str(), rank.device.as_str()))
            .collect();
        assert_eq!(
            placement,
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
        // and one it places alone stays on rank zero, which is the head's host.
        for entry in ["parallel", "divided"] {
            assert_eq!(
                worker.components[entry].ranks.len(),
                8,
                "{entry} spans the worker group"
            );
        }
        assert_eq!(worker.components["alone"].ranks, vec![0]);
    }

    /// The two-rank groups of four schedulers over two four-device hosts
    /// fill the head's host first, and each group keeps its ranks on one host.
    #[test]
    fn the_data_parallel_shorthand_places_engine_cores_in_host_blocks() {
        let hosts = ["rank-0".to_owned(), "rank-1".to_owned()];
        let groups = WorkerGroupConfig::data_parallel(
            &hosts,
            "cuda",
            4,
            2,
            2,
            WorkerGroupConfig::single_component("model", 2),
        );

        let placement: Vec<_> = groups
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
            placement,
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
        assert!(WorkerGroupConfig::validate_all(&groups).is_ok());

        // One scheduler keeps the single-group identity.
        let single = WorkerGroupConfig::data_parallel(
            &hosts,
            "cuda",
            1,
            2,
            2,
            WorkerGroupConfig::single_component("model", 2),
        );
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].id.to_string(), "model");
    }

    /// Schedulers must place the same deployment on different ranks, and
    /// the size must divide the groups; both are refused before any worker
    /// launches.
    #[test]
    fn engine_cores_refuse_unequal_deployments() {
        let hosts = ["localhost".to_owned()];
        let mut config = EngineConfig {
            worker_groups: WorkerGroupConfig::data_parallel(
                &hosts,
                "cpu",
                2,
                1,
                2,
                WorkerGroupConfig::single_component("model", 1),
            ),
            data_parallel_size: 3,
            ..EngineConfig::sim("sim-model")
        };
        assert!(SchedulerHandle::start_all(config.clone()).is_err());

        config.data_parallel_size = 2;
        config.worker_groups[1].queue_depth = 3;
        let error = SchedulerHandle::start_all(config)
            .err()
            .expect("unequal schedulers are refused");
        assert!(error.to_string().contains("schedulers must place"));
    }

    /// Experts shard across two or more schedulers of one worker each; any
    /// other placement is refused before a worker launches.
    #[test]
    fn expert_parallelism_needs_several_one_worker_engine_cores() {
        let hosts = ["localhost".to_owned()];
        for (cores, ranks) in [(1, 1), (2, 2)] {
            let config = EngineConfig {
                worker_groups: WorkerGroupConfig::data_parallel(
                    &hosts,
                    "cpu",
                    cores,
                    ranks,
                    2,
                    WorkerGroupConfig::single_component("model", ranks),
                ),
                data_parallel_size: cores,
                expert_parallel: Some(ExpertExchange::AllToAll),
                ..EngineConfig::sim("sim-model")
            };
            let error = SchedulerHandle::start_all(config)
                .err()
                .expect("the placement is refused");
            assert!(error.to_string().contains("expert parallelism"), "{error}");
        }
    }

    #[test]
    fn a_rank_count_that_does_not_divide_its_hosts_still_places_every_rank() {
        // The first hosts take the remainder, which keeps each host's block
        // contiguous and rank zero on the head's host.
        let hosts = ["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let worker = WorkerGroupConfig::placed(
            &hosts,
            "cuda",
            8,
            2,
            WorkerGroupConfig::single_component(DEFAULT_COMPONENT, 8),
        );

        let hosts: Vec<_> = worker.ranks.iter().map(|rank| rank.host.as_str()).collect();
        assert_eq!(hosts, vec!["a", "a", "a", "b", "b", "b", "c", "c"]);
        assert_eq!(worker.ranks.len(), 8);
    }

    /// Reordering a component's ranks is a valid placement; repeating a rank
    /// is not.
    #[test]
    fn entry_geometry_uses_unique_ordered_rank_members() {
        let mut worker =
            WorkerGroupConfig::placed(&["localhost".to_owned()], "cuda", 4, 2, mixed_components(4));
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
        let worker = WorkerGroupConfig::placed(
            &["localhost".to_owned()],
            "cuda",
            1,
            2,
            WorkerGroupConfig::single_component(DEFAULT_COMPONENT, 1),
        );
        let mut other = worker.clone();
        assert!(WorkerGroupConfig::validate_all(&[worker.clone(), other.clone()]).is_err());

        // Distinct identities on the same device validate, whether the groups
        // replicate one component name or hold different ones.
        other.id = WorkerGroupId("other".into());
        assert!(WorkerGroupConfig::validate_all(&[worker.clone(), other.clone()]).is_ok());

        let entry = other.components.remove(DEFAULT_COMPONENT).unwrap();
        other.components.insert("divided".into(), entry);
        assert!(WorkerGroupConfig::validate_all(&[worker, other]).is_ok());
    }
}
