//! The deployment's worlds and how it moves between them
//! (`PUT /admin/config`, `specs/online-operations.md`).
//!
//! A world is the processes that execute one configuration of the
//! deployment: an engine and its workers, their communicators, graphs and
//! the storage they created. A deployment has two slots for worlds, and at
//! most one world executes. An update to a configuration that changes what
//! a world executes takes four steps:
//!
//! 1. **Preparation.** The serving world gives back the tail of its KV
//!    cache the successor needs (its requests run on; its capacity drops),
//!    and the successor starts in the other slot: its workers map what they
//!    share with the serving world read-only (the weights of the same keys,
//!    and the KV cache segments of the serving world's workers of the same
//!    scope), create the rest of their own, warm up and capture their graphs.
//!    The serving world executes throughout. A successor of another
//!    parallel configuration holds other regions of the logical tensors on
//!    each GPU: before anything starts the daemon plans its ranks' weights,
//!    their GPUs and the copy manifests that bring them over
//!    ([`redistribution`]), and the successor maps the weight chunks whose
//!    keys (the regions they hold) lie on its GPUs, receives the rest from
//!    the serving world, and maps none of its KV cache, which it lays out
//!    for its own heads.
//! 2. **Promotion.** At a step boundary new requests wait, the serving world
//!    parks (it ends every token request that resumes from its generated
//!    tokens as migratable and executes nothing more, keeping its storage and
//!    its prefix cache index), and the successor imports that index into
//!    the units it maps and serves: every migrated request is submitted
//!    again with the tokens it generated and reuses the pages of every
//!    complete page of them, computing at most its last page. On a
//!    successor of another parallel configuration each computes its tokens
//!    again.
//! 3. **Trial.** The successor executes; the previous world is kept as
//!    `update_config.previous_world` states: parked, its weights alone, or
//!    its weights in host storage. The trial ends only with the operator's
//!    commit, and rolls back by itself on a fault of the successor (a fatal
//!    worker error, its engine process's exit, a stalled job, non-finite
//!    logits) or past a threshold of `update_config.rollback`.
//! 4. **Commit** or **rollback.** A commit releases the previous world: its
//!    processes exit, the successor takes the segments it mapped read-only
//!    and retires the previous world's storage, and grows its cache into the
//!    storage freed. A rollback parks and closes the successor, resumes the
//!    previous world (or starts it again on its kept weights), retires the
//!    successor's storage, and marks the configuration unavailable: applying
//!    it again needs `force`.

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use proto::report::SchedulerStats;
use proto::request::wire;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::engine::client::{EngineClient, EngineState, Successor};
use crate::engine::hold::Kept;
use crate::leader::{Leader, Replica, UpdateError, locked, stats_sink};
use crate::options::{PreviousWorld, RollbackOptions, ServeConfig};

mod change;
pub(crate) mod group;
mod redistribution;
mod warming;
use change::{Numerics, Observed, Step, Treatment, WorldChange};
use redistribution::Redistribution;

/// An irrevocable operator decision, retained through partial publication,
/// worker recovery and daemon handoff until every successor commits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Commit {
    worlds: Vec<u64>,
    retire: Vec<u64>,
    nodes: Option<Vec<String>>,
    pub(crate) pending: Vec<u64>,
}

/// How the requests in flight at a promotion continue when the new world
/// computes other numerics than the one they ran on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InFlight {
    /// Each continues with the state it reached, which the new world reads
    /// as its own; the pages it writes afterwards reach no shared tier.
    #[default]
    Keep,
    /// Each computes its state again from its tokens, exactly.
    Recompute,
}

/// A change of the deployment's configuration (`PUT /admin/config`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ApplyRequest {
    /// Update this independent replica. Without an index, a deployment
    /// update advances through replicas as each trial is committed.
    pub replica: Option<usize>,
    /// The whole target configuration as TOML, as `--config` reads it;
    /// `None` takes the current one.
    pub config: Option<String>,
    /// `PATH=VALUE` assignments over that configuration, as `--set` states
    /// them.
    pub set: Vec<String>,
    pub in_flight: InFlight,
    /// Report the plan and change nothing.
    pub dry_run: bool,
    /// Apply a configuration a rollback marked unavailable.
    pub force: bool,
}

/// What an apply changed.
#[derive(Debug, Clone, PartialEq)]
pub enum Applied {
    /// The target is the current configuration.
    Unchanged,
    /// The schedulers took the new limits at their next step, or the daemon
    /// reads the options that changed from its next request on.
    InPlace(serde_json::Value),
    /// The plan of a dry run, which changed nothing.
    Planned(serde_json::Value),
    /// The daemon's own options or executable change: a hot restart of the
    /// daemon with the target configuration applies them. When the change
    /// also changes what a world executes, the daemon's successor applies
    /// `then`, the same change, which then changes the world alone: the
    /// orchestrator changes before what it orchestrates.
    Daemon {
        config: Box<ServeConfig>,
        executable: Option<String>,
        then: Option<Box<ApplyRequest>>,
    },
    /// A new world executes on trial.
    World(serde_json::Value),
}

/// The previous world a trial keeps.
pub(crate) enum Previous {
    /// Its engine, parked: a rollback resumes it.
    Parked(Arc<EngineClient>),
    /// Its weights alone: a rollback starts its engine again, which binds
    /// them.
    Weights,
}

/// A world on trial and what it rolls back to.
pub(crate) struct Trial {
    /// The world on trial.
    pub(crate) world: u64,
    /// Its configuration's digest, which a rollback marks unavailable.
    pub(crate) digest: String,
    /// The previous world's identity, spec and configuration.
    pub(crate) previous_world: u64,
    pub(crate) previous_spec: wire::EngineSpec,
    pub(crate) previous_config: ServeConfig,
    pub(crate) previous: Previous,
    /// Completed Host copies of the predecessor's in-flight cache pages.
    pub(crate) cache_backup: Vec<storage::manifest::cache::Snapshot>,
    /// The previous world's retention mode, as the status reports it: the
    /// mode the update asked for, or `weights` when its weights could not
    /// move to host storage.
    pub(crate) mode: PreviousWorld,
    /// Why the previous world's weights stayed on the GPUs when the update
    /// asked to keep them in host storage: each node's refusal.
    pub(crate) host_refused: Option<String>,
    pub(crate) promoted: Instant,
    pub(crate) promoted_at: f64,
    /// The previous world's step time and throughput before the promotion.
    pub(crate) baseline: Option<Rates>,
    pub(crate) thresholds: RollbackOptions,
    /// The request outcomes counted when the trial began.
    pub(crate) outcomes: (u64, u64),
    /// How long requests waited at the promotion.
    pub(crate) pause: Duration,
    /// Whether the previous world serves eagerly since the successor
    /// prepared (`update_config.eager_prepare`): its graphs replay again if
    /// it resumes.
    pub(crate) eager: bool,
}

/// A configuration a rollback marked unavailable, and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Unavailable {
    pub digest: String,
    /// The world that executed it and was rolled back.
    pub world: u64,
    pub reason: String,
    /// Seconds since the Unix epoch.
    pub at: f64,
}

/// A world change in progress, as the status reports it.
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    /// The step the change executes ([`Step::name`]).
    pub phase: String,
    /// When the change began, in seconds since the Unix epoch.
    pub started: f64,
    /// The world being prepared, once it starts.
    pub world: Option<u64>,
    /// What the change does otherwise than its options ask: a preparation
    /// whose transfers start without a baseline of some scheduler's steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// What every node's Storage Server had counted of the deployment's
    /// transfers when the change began, by host (`None` when they could
    /// not be read): the bytes the change moved are what they counted
    /// since.
    #[serde(skip)]
    moved_from: Option<BTreeMap<String, Vec<storage::protocol::TransferTotal>>>,
    /// When new requests began to wait for the promotion.
    #[serde(skip)]
    paused: Option<Instant>,
    /// The serving world's step time against its baseline, as the transfer
    /// throttling read it, while the successor prepares.
    #[serde(skip)]
    steps: Option<Arc<std::sync::Mutex<crate::interference::StepTime>>>,
}

/// The deployment's worlds as the leader tracks them.
#[derive(Default)]
pub(crate) struct WorldState {
    /// The world the executing engine runs.
    pub(crate) executing: u64,
    pub(crate) trial: Option<Trial>,
    pub(crate) progress: Option<Progress>,
    pub(crate) unavailable: BTreeMap<String, Unavailable>,
    /// What the last world change, commit or rollback reported.
    pub(crate) last: Option<serde_json::Value>,
}

/// Average step time and throughput of one world over a span.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rates {
    /// Microseconds per batch.
    pub step_us: f64,
    /// Tokens per second the forwards computed.
    pub tokens_per_second: f64,
}

/// One statistics report of one scheduler of one world: the interval's
/// host-observed batch time, batches, query tokens and numerical faults.
pub(crate) struct Sample {
    pub(crate) at: Instant,
    pub(crate) world: u64,
    pub(crate) scheduler: u32,
    pub(crate) batch_us: u64,
    pub(crate) batches: u64,
    pub(crate) tokens: u64,
    numerical_faults: u64,
}

/// The schedulers' statistics of every world, kept over the longest span a
/// trial measures, from which a trial reads the previous world's baseline
/// and its own rates and faults.
#[derive(Default)]
pub(crate) struct Observer {
    samples: std::sync::Mutex<VecDeque<Sample>>,
    /// The profiling sessions' progress the workers reported, keyed
    /// `{session}/{counter}` (`ForwardStats::profile`), summed.
    profile: std::sync::Mutex<BTreeMap<String, u64>>,
}

/// The longest span statistics are kept for.
const OBSERVED: Duration = Duration::from_secs(600);

impl Observer {
    /// Records the report of scheduler `scheduler` of `world`.
    pub(crate) fn record(&self, world: u64, scheduler: u32, stats: &SchedulerStats) {
        let tokens = stats
            .worker_forward_stats
            .as_ref()
            .map_or(0, |forward| forward.mode_tokens.values().sum());
        if let Some(forward) = &stats.worker_forward_stats {
            let mut profile = locked(&self.profile);
            for (key, value) in &forward.profile {
                *profile.entry(key.clone()).or_default() += value;
            }
        }
        let now = Instant::now();
        let mut samples = locked(&self.samples);
        samples.push_back(Sample {
            at: now,
            world,
            scheduler,
            batch_us: stats.batch_roundtrip_us,
            batches: stats.batch_count,
            tokens,
            numerical_faults: stats.numerical_faults,
        });
        while samples
            .front()
            .is_some_and(|sample| now.duration_since(sample.at) > OBSERVED)
        {
            samples.pop_front();
        }
    }

    /// `world`'s rates over the `window` before now, from `since` on; `None`
    /// without a batch in it.
    pub(crate) fn rates(&self, world: u64, since: Instant, window: Duration) -> Option<Rates> {
        let now = Instant::now();
        let from = since.max(now.checked_sub(window).unwrap_or(since));
        let samples = locked(&self.samples);
        let (mut batch_us, mut batches, mut tokens) = (0_u64, 0_u64, 0_u64);
        let mut first: Option<Instant> = None;
        for sample in samples
            .iter()
            .filter(|sample| sample.world == world && sample.at >= from)
        {
            batch_us += sample.batch_us;
            batches += sample.batches;
            tokens += sample.tokens;
            first.get_or_insert(sample.at);
        }
        if batches == 0 {
            return None;
        }
        let span = now
            .duration_since(first.unwrap_or(from))
            .max(Duration::from_secs(1));
        Some(Rates {
            step_us: batch_us as f64 / batches as f64,
            tokens_per_second: tokens as f64 / span.as_secs_f64(),
        })
    }

    /// Calls `visit` with every report of `world` made in `[from, until)`,
    /// in the order they arrived.
    pub(crate) fn samples(
        &self,
        world: u64,
        from: Instant,
        until: Instant,
        mut visit: impl FnMut(&Sample),
    ) {
        for sample in locked(&self.samples)
            .iter()
            .filter(|sample| sample.world == world && sample.at >= from && sample.at < until)
        {
            visit(sample);
        }
    }

    /// Profiling session `session`'s progress the workers reported: the
    /// steps their windows took, the bytes of their files, and the workers
    /// whose window ended.
    pub(crate) fn profile(&self, session: u64) -> (u64, u64, u64) {
        let profile = locked(&self.profile);
        let counter = |name: &str| {
            profile
                .get(&format!("{session}/{name}"))
                .copied()
                .unwrap_or_default()
        };
        (counter("steps"), counter("bytes"), counter("ended"))
    }

    /// The calls of `world` that produced non-finite logits since `since`.
    pub(crate) fn numerical_faults(&self, world: u64, since: Instant) -> u64 {
        locked(&self.samples)
            .iter()
            .filter(|sample| sample.world == world && sample.at >= since)
            .map(|sample| sample.numerical_faults)
            .sum()
    }
}

/// The digest of a configuration, which names it in the status and the
/// unavailable marks: of its printed TOML.
pub(crate) fn digest(config: &ServeConfig) -> anyhow::Result<String> {
    use sha2::Digest as _;
    let printed = crate::options::printed(config)?;
    Ok(sha2::Sha256::digest(printed.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

/// The phase of a rollback in progress, which the status reports as it
/// reports a change's steps.
const ROLLBACK: &str = "rollback";

/// What the deployment's transfer tasks copied so far on each node of
/// `nodes` (each node's storage report, `EngineClient::describe_storage`),
/// by host.
fn node_transfers<'a>(
    nodes: impl Iterator<Item = &'a serde_json::Value>,
) -> BTreeMap<String, Vec<storage::protocol::TransferTotal>> {
    nodes
        .filter_map(|node| {
            let host = node["host"].as_str()?.to_owned();
            let totals = serde_json::from_value(node["transfers"].clone()).unwrap_or_default();
            Some((host, totals))
        })
        .collect()
}

/// The GPUs the deployment's worlds hold storage on, from every node's
/// storage report (`nodes`): each one's host and name (`GPU-…`, as
/// `nvidia-smi` prints its UUID), its total and quota and the bytes the
/// deployment's ledger books there. A manager of several deployments reads
/// which GPUs they share from it.
fn deployment_gpus(nodes: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut gpus = Vec::new();
    for node in nodes {
        let host = node["host"].as_str().unwrap_or_default();
        let ledgers: Vec<storage::protocol::GpuLedger> =
            serde_json::from_value(node["ledgers"].clone()).unwrap_or_default();
        for ledger in ledgers
            .iter()
            .filter(|ledger| !ledger.deployment.is_empty())
        {
            gpus.push(serde_json::json!({
                "host": host,
                "gpu": storage::protocol::gpu_name(&ledger.gpu),
                "total": ledger.total,
                "quota": ledger.quota,
                "used": ledger.used(),
            }));
        }
    }
    gpus
}

/// What every node's Storage Server has counted of the deployment's
/// transfers, by host; `None` when the engine cannot read them.
async fn transfer_totals(
    running: &EngineClient,
) -> Option<BTreeMap<String, Vec<storage::protocol::TransferTotal>>> {
    let nodes = tokio::time::timeout(LEDGER_READ, running.describe_storage())
        .await
        .ok()?
        .ok()?;
    let nodes: Vec<serde_json::Value> = nodes
        .iter()
        .filter_map(|node| serde_json::from_str(node).ok())
        .collect();
    Some(node_transfers(nodes.iter()))
}

/// What the deployment's transfers moved between two readings of every
/// node's totals, `from` and `to`: the bytes copied, those submitted, and
/// both by priority. A node absent from `from` (one that joined since)
/// moved all it counted.
fn moved(
    from: &BTreeMap<String, Vec<storage::protocol::TransferTotal>>,
    to: &BTreeMap<String, Vec<storage::protocol::TransferTotal>>,
) -> serde_json::Value {
    let mut priorities = BTreeMap::<String, (u64, u64)>::new();
    for (host, totals) in to {
        for total in totals {
            let earlier = from.get(host).and_then(|totals| {
                totals
                    .iter()
                    .find(|earlier| earlier.priority == total.priority)
            });
            let name = serde_json::to_value(total.priority)
                .ok()
                .and_then(|name| name.as_str().map(str::to_owned))
                .unwrap_or_default();
            let entry = priorities.entry(name).or_default();
            entry.0 += total
                .copied
                .saturating_sub(earlier.map_or(0, |earlier| earlier.copied));
            entry.1 += total
                .bytes
                .saturating_sub(earlier.map_or(0, |earlier| earlier.bytes));
        }
    }
    serde_json::json!({
        "bytes": priorities.values().map(|(copied, _)| copied).sum::<u64>(),
        "submitted_bytes": priorities.values().map(|(_, submitted)| submitted).sum::<u64>(),
        "by_priority": priorities
            .iter()
            .map(|(priority, (copied, submitted))| {
                (priority.clone(), serde_json::json!({ "bytes": copied, "submitted_bytes": submitted }))
            })
            .collect::<serde_json::Map<_, _>>(),
    })
}

/// How often a trial checks its thresholds.
const TRIAL_POLL: Duration = Duration::from_millis(500);

/// How long the closed engine of a world takes to exit.
const CLOSED_EXIT: Duration = Duration::from_secs(60);

/// How long the status waits for the storage ledgers, which each node's
/// daemon reads from its Storage Server.
const LEDGER_READ: Duration = Duration::from_secs(10);

/// Whether `target` differs from `served` in the fence probes' switch
/// alone, which the workers change in place. Every other part of the
/// profiling configuration shapes the CUDA graphs a world captures at its
/// start, the probed components and whether PC sampling was enabled before
/// the capture (`profiling_config.graph_pc_sampling`), and takes a new
/// world.
fn switches_probes_only(served: &wire::EngineSpec, target: &wire::EngineSpec) -> bool {
    let mut switched = comparable(served);
    if let (Some(served), Some(target)) = (switched.profiling.as_mut(), target.profiling.as_ref()) {
        served.fence_probes = target.fence_probes;
        served.fence_probes_off.clone_from(&target.fence_probes_off);
    }
    switched == comparable(target)
}

/// The parts of `spec` a configuration states: without the world's identity,
/// the layout of its cache, the worlds it retires as it starts and the
/// weights it receives, which the world's history sets.
fn comparable(spec: &wire::EngineSpec) -> wire::EngineSpec {
    let mut comparable = wire::EngineSpec {
        world: 0,
        predecessor: None,
        segments: Vec::new(),
        retire: Vec::new(),
        received_weights: Vec::new(),
        expert_order: Default::default(),
        stage_cache: false,
        cache_keys: Vec::new(),
        ..spec.clone()
    };
    // The file path is diagnostic; the resolved contents define the ranks.
    if let Some(ranks) = comparable.placement.as_mut() {
        ranks.worker_groups.clear();
    }
    comparable
}

fn devices(
    spec: &wire::EngineSpec,
) -> anyhow::Result<std::collections::BTreeSet<(String, String)>> {
    Ok(engine::EngineConfig::from_spec(spec)?
        .worker_groups
        .into_iter()
        .flat_map(|group| group.ranks.into_iter().map(|rank| (rank.host, rank.device)))
        .collect())
}

/// Keeps an existing replica's assigned devices when an update changes its
/// execution options without requesting a different set of ranks.
#[expect(
    clippy::unwrap_used,
    reason = "EngineConfig::from_spec requires the ranks before this mutation"
)]
fn retain_devices(served: &wire::EngineSpec, target: &mut wire::EngineSpec) -> anyhow::Result<()> {
    if served
        .placement
        .as_ref()
        .is_some_and(|ranks| ranks.worker_groups_content.is_empty())
    {
        return Ok(());
    }
    let source = engine::EngineConfig::from_spec(served)?;
    let mut destination = engine::EngineConfig::from_spec(target)?;
    for group in &mut destination.worker_groups {
        if let Some(before) = source
            .worker_groups
            .iter()
            .find(|before| before.id == group.id)
            && before.ranks.len() == group.ranks.len()
        {
            group.ranks.clone_from(&before.ranks);
        }
    }
    target.placement.as_mut().unwrap().worker_groups_content =
        engine::Deployment::snapshot(&destination.worker_groups)?;
    Ok(())
}

/// Assigns each distinct requested GPU once. Several numerical worker
/// groups sharing a GPU keep that relationship in the successor.
#[expect(
    clippy::unwrap_used,
    reason = "EngineConfig::from_spec requires the ranks before this mutation"
)]
fn assign_devices(
    target: &mut wire::EngineSpec,
    mut available: BTreeMap<String, Vec<u32>>,
    explicit: bool,
) -> anyhow::Result<()> {
    let mut config = engine::EngineConfig::from_spec(target)?;
    let mut assigned = BTreeMap::<(String, String), String>::new();
    for group in &mut config.worker_groups {
        for rank in &mut group.ranks {
            let key = (rank.host.clone(), rank.device.clone());
            if let Some(device) = assigned.get(&key) {
                rank.device.clone_from(device);
                continue;
            }
            let free = available
                .get_mut(&rank.host)
                .with_context(|| format!("no GPU inventory for host {}", rank.host))?;
            let index = if explicit {
                let device = rank
                    .device
                    .strip_prefix("cuda:")
                    .context("a replica rank must name a CUDA device")?
                    .parse::<u32>()?;
                free.iter().position(|candidate| *candidate == device)
                    .with_context(|| format!("host {} device {} is occupied; the replica successor needs disjoint GPUs", rank.host, rank.device))?
            } else {
                anyhow::ensure!(
                    !free.is_empty(),
                    "host {} has insufficient free GPUs for the replica successor",
                    rank.host
                );
                0
            };
            let device = format!("cuda:{}", free.remove(index));
            assigned.insert(key, device.clone());
            rank.device = device;
        }
    }
    target.placement.as_mut().unwrap().worker_groups_content =
        engine::Deployment::snapshot(&config.worker_groups)?;
    Ok(())
}

/// Whether `target` states another parallel configuration than `served`:
/// other ranks, devices or hosts (a deployment file by its content), another
/// device of a generation tower
/// (the worker mesh), another division of the routed experts, or another
/// page size of the KV cache. Each GPU of the successor then holds other
/// regions of the logical tensors: its weights are keyed by those regions,
/// so a chunk of the same key on its GPU is mapped and the rest
/// redistributed, and its KV cache is laid out for its own heads and
/// pages, so it maps none of the serving world's segments and the requests
/// in flight compute their state again.
fn parallel_change(served: &wire::EngineSpec, target: &wire::EngineSpec) -> bool {
    let ranks = |spec: &wire::EngineSpec| {
        spec.placement.as_ref().map(|placement| {
            (
                placement.worker_groups_content.clone(),
                placement.tensor_parallel_size,
                placement.data_parallel_size,
                placement.device.clone(),
                placement.devices.clone(),
                placement.worker_hosts.clone(),
            )
        })
    };
    let mesh = |spec: &wire::EngineSpec| spec.workers.as_ref().map(|workers| workers.mesh.clone());
    let pages = |spec: &wire::EngineSpec| spec.kv_cache.as_ref().map(|cache| cache.block_size);
    served.host != target.host
        || ranks(served) != ranks(target)
        || mesh(served) != mesh(target)
        || pages(served) != pages(target)
        || served.experts != target.experts
}

/// Changes outside a successor's current process-control contract. Rank
/// resources and numerical layouts are planned by the target world.
fn differing(served: &wire::EngineSpec, target: &wire::EngineSpec) -> Vec<&'static str> {
    let mut parts = Vec::new();
    let mut differs = |name: &'static str, same: bool| {
        if !same {
            parts.push(name);
        }
    };
    differs("the deployment", served.deployment == target.deployment);
    parts
}

/// The parts of the engine spec a world change changes that key the
/// weights by what they are loaded from (`reloads`): the model's files, the
/// dtype, the quantization, the load format and the weight source. The
/// successor loads the weights of other keys from the model's files and
/// maps those whose keys stay.
fn reloads(served: &wire::EngineSpec, target: &wire::EngineSpec) -> Vec<&'static str> {
    let mut parts = Vec::new();
    if served.model != target.model {
        parts.push("the model");
    }
    if let (Some(old), Some(new)) = (&served.workers, &target.workers) {
        for (name, same) in [
            ("the dtype", old.dtype == new.dtype),
            (
                "the quantization",
                old.quantization_config == new.quantization_config,
            ),
            ("the load format", old.load_format == new.load_format),
            ("the weight source", old.weights == new.weights),
        ] {
            if !same {
                parts.push(name);
            }
        }
    }
    parts
}

/// The encoding of a world's KV cache: its configured dtype, else the
/// compute dtype each layer's cache follows, with the quantization that may
/// scale it.
fn kv_encoding(spec: &wire::EngineSpec) -> (String, String, String) {
    let configured = spec
        .kv_cache
        .as_ref()
        .map(|kv| kv.dtype.clone())
        .unwrap_or_default();
    let (compute, quantization) = spec
        .workers
        .as_ref()
        .map_or_else(Default::default, |workers| {
            (workers.dtype.clone(), workers.quantization_config.clone())
        });
    (configured, compute, quantization)
}

/// Whether the successor's KV cache is encoded otherwise than the serving
/// world's, which it then cannot read: a class-丙 change of
/// `specs/online-operations.md` §4.6 without a conversion, so the requests
/// in flight compute their state again and the successor maps none of the
/// serving world's segments.
fn kv_encoding_changes(served: &wire::EngineSpec, target: &wire::EngineSpec) -> bool {
    kv_encoding(served) != kv_encoding(target)
}

/// Changes that need the target ranks' capacity plan even when their
/// numerical tensor regions stay on the same devices. A different quota,
/// queue, cache or media configuration must not inherit a budget measured
/// for the serving world's resources.
fn resource_change(served: &wire::EngineSpec, target: &wire::EngineSpec) -> bool {
    served.placement != target.placement
        || served.transfer != target.transfer
        || served.kv_cache != target.kv_cache
        || served.video != target.video
}

/// A successor with another capacity lays out its own segments. Importing
/// the source's full segments could exceed a smaller target capacity;
/// warming copies only the accepted prefixes into the target's own pool.
fn cache_capacity_change(served: &wire::EngineSpec, target: &wire::EngineSpec) -> bool {
    let tokens = |spec: &wire::EngineSpec| spec.kv_cache.as_ref().and_then(|kv| kv.max_tokens);
    tokens(served) != tokens(target)
}

/// Whether a world succeeding the one `running` executes, which adds
/// `successor` beside what it shares with it, fits beside it on every
/// device, from each GPU's storage ledger: the report `PUT
/// /admin/config` plans with, and the refusal when it does not fit. The
/// serving engine computes, for each worker that holds the KV cache, what
/// the serving world, the successor and the Transfer Server hold on its GPU
/// and the KV cache limit that leaves the serving world, and fails, with
/// every item of every GPU and host, when a limit cannot hold the segments
/// the serving world keeps in any case.
async fn storage_plan(
    running: &[Arc<EngineClient>],
    successor: &mut Successor,
    budgets: &BTreeMap<storage::manifest::Gpu, storage::manifest::weights::RankBudget>,
    target_hosts: &BTreeMap<storage::manifest::Gpu, String>,
    control: &Arc<crate::engine::control::EngineControl>,
    previous_world: PreviousWorld,
) -> (serde_json::Value, Option<String>) {
    // Each policy spends only resources whose owner can release them while
    // serving continues: tail eviction, host staging, then graph retirement.
    let ordinary = *successor;
    let staged = Successor {
        host_cache: ordinary.unmapped_cache,
        ..ordinary
    };
    let mut policies = vec![ordinary];
    if staged != ordinary {
        policies.push(staged);
    }
    if !ordinary.eager_source {
        policies.push(Successor {
            eager_source: true,
            ..staged
        });
        if staged != ordinary {
            // A host refusal can still admit a wholly device-resident
            // successor after its predecessor releases the graph pool.
            policies.push(Successor {
                eager_source: true,
                ..ordinary
            });
        }
    }
    let mut refused = Vec::new();
    for policy in policies {
        let plans = futures::future::join_all(
            running
                .iter()
                .map(|client| client.make_room(wire::CacheChange::Plan, policy, None, budgets)),
        )
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map(|plans| plans.into_iter().flatten().collect::<Vec<_>>());
        match plans {
            Ok(results) => {
                let report = match storage_report(
                    &results,
                    previous_world,
                    policy,
                    budgets,
                    target_hosts,
                    control,
                )
                .await
                {
                    Ok(report) => report,
                    Err(error) => {
                        refused.push(error);
                        continue;
                    }
                };
                if report["feasible"] == true {
                    *successor = policy;
                    return (report, None);
                }
                refused.push(report.to_string());
            }
            Err(error) => refused.push(error),
        }
    }
    let refusal = refused.join("; ");
    (
        serde_json::json!({ "feasible": false, "refusal": refusal }),
        Some(refusal),
    )
}

/// The storage report of the engine's per-device plans (`ledger::Preparing`
/// of the engine, as JSON) for a successor that adds `successor` and a
/// previous world kept as `previous_world`: each GPU's, with what each
/// step of `specs/online-operations.md` §6.5 frees there ([`relief`]), and
/// each host's admission of new host storage. A successor sharing the
/// serving world's page pools needs none; a previous world kept in host
/// storage moves there the weights a successor that loads its own does not
/// map (all of the serving world's on each GPU, at most), which each host's
/// admission must hold, or they stay on the GPUs. Host staging accounts for
/// source rollback copies and assigned target rank reservations separately;
/// target-only hosts are queried through their existing process ledger.
async fn storage_report(
    plans: &[String],
    previous_world: PreviousWorld,
    successor: Successor,
    budgets: &BTreeMap<storage::manifest::Gpu, storage::manifest::weights::RankBudget>,
    target_hosts: &BTreeMap<storage::manifest::Gpu, String>,
    control: &Arc<crate::engine::control::EngineControl>,
) -> Result<serde_json::Value, String> {
    let weights_to_host = previous_world == PreviousWorld::Host && successor.loads_weights;
    let mut gpus: Vec<serde_json::Value> = plans
        .iter()
        .filter_map(|plan| serde_json::from_str(plan).ok())
        .collect();
    for gpu in &mut gpus {
        gpu["steps"] = relief(gpu, previous_world, successor);
    }
    // The serving world's weights on each GPU of each host, once per GPU
    // however many of its workers report it.
    let mut weights: BTreeMap<(String, String), u64> = BTreeMap::new();
    for gpu in &gpus {
        if let (Some(host), Some(device)) = (gpu["host"].as_str(), gpu["gpu"].as_str()) {
            let bytes = gpu["serving"]["weights"].as_u64().unwrap_or(0);
            weights.insert((host.to_owned(), device.to_owned()), bytes);
        }
    }
    let mut storage = BTreeMap::<String, storage::protocol::HostStorage>::new();
    let mut staging = BTreeMap::<String, u64>::new();
    for gpu in &gpus {
        let (Some(host), Some(available)) = (gpu["host"].as_str(), gpu.get("host_storage")) else {
            continue;
        };
        storage.entry(host.to_owned()).or_insert(
            serde_json::from_value(available.clone()).map_err(|error| error.to_string())?,
        );
        if successor.host_cache {
            // With a rank plan, the source contributes only its rollback
            // copy. Target layouts are charged at their own hosts below.
            let key = if target_hosts.is_empty() {
                "host_staging"
            } else {
                "in_flight_kv"
            };
            *staging.entry(host.to_owned()).or_default() += gpu[key].as_u64().unwrap_or(0);
        }
    }
    if successor.host_cache {
        for (gpu, host) in target_hosts {
            let reservation = budgets
                .get(gpu)
                .and_then(|budget| budget.in_flight)
                .ok_or_else(|| {
                    format!(
                        "target GPU {} has no continuation reservation",
                        storage::protocol::gpu_name(gpu)
                    )
                })?;
            *staging.entry(host.clone()).or_default() += reservation.bytes;
        }
        let missing: Vec<_> = staging
            .keys()
            .filter(|host| !storage.contains_key(*host))
            .cloned()
            .collect();
        if !missing.is_empty() {
            let control = Arc::clone(control);
            storage.extend(
                tokio::task::spawn_blocking(move || control.host_storage(&missing))
                    .await
                    .map_err(|error| error.to_string())?
                    .map_err(|error| error.to_string())?,
            );
        }
    }
    let mut hosts = Vec::new();
    for (host, storage) in storage {
        let room = storage.room();
        let staged = staging.get(&host).copied().unwrap_or(0);
        let required: u64 = staged
            + if weights_to_host {
                weights
                    .iter()
                    .filter(|((on, _), _)| on == &host)
                    .map(|(_, bytes)| bytes)
                    .sum()
            } else {
                0
            };
        hosts.push(serde_json::json!({
            "host": host,
            "available": storage.available,
            "reserve": storage.reserve,
            "unallocated": storage.unallocated,
            "admits": room,
            "required": required,
            "cache_staging": staged,
            "previous_world_weights": weights_to_host.then_some(required - staged),
            "feasible": required <= room,
        }));
    }
    Ok(serde_json::json!({
        "feasible": gpus.iter().all(|gpu| gpu["feasible"] == true)
            && hosts.iter().all(|host| host["feasible"] == true),
        "host_staging": successor.host_cache,
        "eager": successor.eager_source,
        "gpus": gpus,
        "hosts": hosts,
    }))
}

/// What each step of `specs/online-operations.md` §6.5 frees on the GPU of
/// the plan `gpu` (`ledger::Preparing` of the engine), in bytes, and the
/// promotion stall of moving the requests' KV cache in flight through host
/// storage (step 3), whether or not the plan takes each step, for a
/// successor that adds `successor` and a previous
/// world kept as `previous_world`. In the preparing phase: the idle cached
/// pages of the tail the serving world gives back (step 1), the rest of
/// that tail (step 2), the warm copies a successor that recomputes the
/// cache would otherwise prepare on the device (step 3) and the serving
/// world's graph pool (step 4). In the trial: the previous world's KV cache
/// segments the successor does not map, past its first (step 5, a parked
/// previous world), its weights the successor does not map (step 6, kept in
/// host storage), and its processes' graph pool, process storage and
/// exclusive storages with those segments (step 7, kept as its weights or
/// in host storage).
fn relief(
    gpu: &serde_json::Value,
    previous_world: PreviousWorld,
    successor: Successor,
) -> serde_json::Value {
    let bytes = |value: &serde_json::Value| value.as_u64().unwrap_or(0);
    let serving = &gpu["serving"];
    let kv_kept = bytes(&gpu["kv_kept"]);
    let idle = bytes(&gpu["idle_cached_tail"]);
    let tail = bytes(&gpu["serving_kv_cache"]).saturating_sub(kv_kept);
    // The previous world's segments the successor does not map: all it
    // keeps when the successor maps none of its cache.
    let unmapped = if successor.unmapped_cache { kv_kept } else { 0 };
    let parked = previous_world == PreviousWorld::Parked;
    serde_json::json!({
        "preparation": {
            "idle_prefix_pages": idle,
            "kv_tail": tail.saturating_sub(idle),
            "warm_copies": bytes(&gpu["staging_frees"]),
            "graph_pool": bytes(&serving["graph_pool"]),
        },
        "trial": {
            "unmapped_kv_segments": if parked {
                unmapped.saturating_sub(bytes(&gpu["kv_floor"]))
            } else {
                0
            },
            "weights_to_host": if previous_world == PreviousWorld::Host && successor.loads_weights {
                bytes(&serving["weights"])
            } else {
                0
            },
            "processes": if parked {
                0
            } else {
                bytes(&serving["graph_pool"])
                    + bytes(&serving["process"])
                    + bytes(&serving["exclusive"])
                    + unmapped
            },
        },
        "promotion_stall_ms": gpu["staging_stall_ms"],
    })
}

/// One serving GPU's figures of a world change's storage plan that its
/// copies move, summed over the workers that report the GPU.
#[derive(Debug, Default)]
struct PlannedGpu {
    host: String,
    /// The KV cache the requests in flight hold on the GPU.
    in_flight_kv: u64,
    /// What they take in the successor's layout there, when it maps none of
    /// the serving world's cache.
    in_flight_target: u64,
    /// Their last pages holding less than a page, which a successor that
    /// maps the cache copies at its promotion.
    partial_tail_kv: u64,
    /// The promotion's wait for copies staged in host storage, at the host
    /// link's measured bandwidth.
    staging_stall_ms: Option<u64>,
}

/// The serving GPUs of the storage plan `storage` ([`storage_report`]),
/// by name.
fn planned_gpus(storage: Option<&serde_json::Value>) -> BTreeMap<String, PlannedGpu> {
    let mut gpus = BTreeMap::<String, PlannedGpu>::new();
    for plan in storage
        .and_then(|storage| storage["gpus"].as_array())
        .into_iter()
        .flatten()
    {
        let Some(name) = plan["gpu"].as_str() else {
            continue;
        };
        let bytes = |key: &str| plan[key].as_u64().unwrap_or(0);
        let gpu = gpus.entry(name.to_owned()).or_default();
        plan["host"]
            .as_str()
            .unwrap_or_default()
            .clone_into(&mut gpu.host);
        gpu.in_flight_kv += bytes("in_flight_kv");
        gpu.in_flight_target += bytes("in_flight_target");
        gpu.partial_tail_kv += bytes("partial_tail_kv");
        gpu.staging_stall_ms = gpu.staging_stall_ms.max(plan["staging_stall_ms"].as_u64());
    }
    gpus
}

/// What a world change moves over each link, by the step that moves it,
/// from its plan: the weight manifests of a successor of another parallel
/// configuration, the storage plan's per-GPU figures of the KV cache of the
/// requests in flight, and the target ranks' reservations for them.
struct PlannedCopies {
    gpus: BTreeMap<String, PlannedGpu>,
    /// Each target GPU's reservation for the requests in flight, in bytes of
    /// its layout, and its host: empty for a successor laid out as the
    /// serving world on its GPUs.
    targets: BTreeMap<String, (u64, String)>,
    /// The weight manifests' link loads, as the redistribution reports
    /// them.
    weights: Vec<serde_json::Value>,
}

impl PlannedCopies {
    fn new(
        storage: Option<&serde_json::Value>,
        budgets: &BTreeMap<storage::manifest::Gpu, storage::manifest::weights::RankBudget>,
        target_hosts: &BTreeMap<storage::manifest::Gpu, String>,
        redistributions: &[&serde_json::Value],
    ) -> Self {
        let targets = budgets
            .iter()
            .filter_map(|(gpu, budget)| {
                let bytes = budget.in_flight?.bytes;
                Some((
                    storage::protocol::gpu_name(gpu),
                    (bytes, target_hosts.get(gpu).cloned().unwrap_or_default()),
                ))
            })
            .collect();
        // The manifests report their links as the planner states them.
        let loads: Vec<storage::manifest::LinkLoad> = redistributions
            .iter()
            .filter_map(|report| report["manifests"].as_array())
            .flatten()
            .filter_map(|manifest| manifest["links"].as_array())
            .flatten()
            .filter_map(|load| {
                Some(storage::manifest::LinkLoad {
                    source: serde_json::from_value(load["source"].clone()).ok()?,
                    target: serde_json::from_value(load["target"].clone()).ok()?,
                    bytes: load["bytes"].as_u64()?,
                    seconds: load["seconds"].as_f64().unwrap_or(0.0),
                })
            })
            .collect();
        Self {
            gpus: planned_gpus(storage),
            targets,
            weights: change::link_loads(&loads, target_hosts),
        }
    }

    /// The serving GPUs that hold the requests' KV cache.
    fn sources(&self) -> Vec<serde_json::Value> {
        self.gpus
            .iter()
            .filter(|(_, gpu)| gpu.in_flight_kv > 0)
            .map(|(name, gpu)| change::gpu_end(name, Some(&gpu.host)))
            .collect()
    }

    /// The warming's copies: into each GPU's successor, or, when staged,
    /// into host storage, beside the rollback copies of the serving
    /// world's pages there. A successor laid out as the serving world
    /// copies each GPU's requests on the GPU; one of other target ranks
    /// receives each target GPU's reservation from the serving GPUs, whose
    /// division among them the warming's cache manifest decides from the
    /// layouts the successor's workers report: such a copy names its
    /// `sources` rather than one.
    fn warming(&self, staged: bool) -> Vec<serde_json::Value> {
        let mut links = Vec::new();
        if staged {
            for (name, gpu) in &self.gpus {
                let mut backup = change::link(
                    change::gpu_end(name, Some(&gpu.host)),
                    change::host_end(&gpu.host),
                    gpu.in_flight_kv,
                );
                backup["copy"] = "rollback".into();
                links.push(backup);
            }
        }
        if self.targets.is_empty() {
            for (name, gpu) in &self.gpus {
                let target = if staged {
                    change::host_end(&gpu.host)
                } else {
                    change::gpu_end(name, Some(&gpu.host))
                };
                links.push(change::link(
                    change::gpu_end(name, Some(&gpu.host)),
                    target,
                    gpu.in_flight_target,
                ));
            }
        } else {
            for (name, (bytes, host)) in &self.targets {
                let target = if staged {
                    change::host_end(host)
                } else {
                    change::gpu_end(name, Some(host))
                };
                links.push(serde_json::json!({
                    "sources": self.sources(),
                    "target": target,
                    "bytes": bytes,
                }));
            }
        }
        links
    }

    /// The staged copies a promotion loads from host storage into each
    /// GPU of the successor.
    fn materializing(&self) -> Vec<serde_json::Value> {
        if self.targets.is_empty() {
            self.gpus
                .iter()
                .map(|(name, gpu)| {
                    change::link(
                        change::host_end(&gpu.host),
                        change::gpu_end(name, Some(&gpu.host)),
                        gpu.in_flight_target,
                    )
                })
                .collect()
        } else {
            self.targets
                .iter()
                .map(|(name, (bytes, host))| {
                    change::link(
                        change::host_end(host),
                        change::gpu_end(name, Some(host)),
                        *bytes,
                    )
                })
                .collect()
        }
    }

    /// The requests' partial last pages a successor that maps the serving
    /// world's cache copies into units of its own on each GPU.
    fn partial_tails(&self) -> Vec<serde_json::Value> {
        self.gpus
            .iter()
            .map(|(name, gpu)| {
                let end = change::gpu_end(name, Some(&gpu.host));
                change::link(end.clone(), end, gpu.partial_tail_kv)
            })
            .collect()
    }

    /// The plan's steps with what each moves over each link, and the
    /// details the operator decides by (`WorldChange::planned_steps`).
    fn steps(&self, change: &WorldChange) -> Vec<serde_json::Value> {
        change.planned_steps(|step| match step {
            Step::Prepare if change.redistributes && !change.successor.loads_weights => {
                serde_json::json!({ "links": self.weights })
            }
            Step::WarmCache => serde_json::json!({
                "staged": change.successor.host_cache,
                "links": self.warming(change.successor.host_cache),
            }),
            Step::Materialize => serde_json::json!({
                "links": self.materializing(),
                "stall_ms": self.gpus.values().filter_map(|gpu| gpu.staging_stall_ms).max(),
            }),
            Step::Promote if !change.successor.unmapped_cache => {
                serde_json::json!({ "links": self.partial_tails() })
            }
            Step::KeepPrevious => serde_json::json!({ "previous_world": change.previous_world }),
            _ => serde_json::Value::Null,
        })
    }

    /// What the promotion fills in for the requests in flight, beyond what
    /// they continue from, while new requests wait (§4.5.2): the bytes it
    /// copies (the staged copies over the host links, or each request's
    /// partial last page on its GPU), how long the staged copies hold it at
    /// the measured bandwidth, and the tokens each request computes at most
    /// (`null` for all of its context, when it computes its state again).
    fn backfill(
        &self,
        change: &WorldChange,
        requests: usize,
        tail_tokens: Option<u32>,
    ) -> serde_json::Value {
        let staged = change.successor.host_cache;
        let bytes: u64 = if staged {
            self.materializing()
                .iter()
                .filter_map(|link| link["bytes"].as_u64())
                .sum()
        } else if change.successor.unmapped_cache {
            0
        } else {
            self.gpus.values().map(|gpu| gpu.partial_tail_kv).sum()
        };
        // Warmed requests compute the tokens past the copies of the last
        // warming round, at most its limit; those that map their pages
        // compute none.
        let treatment = change.derived_state(Observed::default()).kv_pages.treatment;
        let tokens = match treatment {
            Treatment::Recomputed => None,
            _ if change.successor.unmapped_cache => tail_tokens,
            _ => Some(0),
        };
        serde_json::json!({
            "bytes": bytes,
            "stall_ms": staged
                .then(|| self.gpus.values().filter_map(|gpu| gpu.staging_stall_ms).max())
                .flatten(),
            "requests": requests,
            "tail_tokens_per_request": tokens,
        })
    }
}

impl Leader {
    /// Applies a configuration through the replica that owns its worlds.
    pub async fn apply(&self, request: ApplyRequest) -> Result<Applied, UpdateError> {
        let _replacing = self
            .replicas
            .replacing
            .try_lock()
            .map_err(|_| UpdateError::InProgress)?;
        let replicas = self.replicas();
        if locked(&self.replicas.commit).is_some() {
            return Err(UpdateError::InProgress);
        }
        if let Some(trial) = locked(&self.replicas.group).as_ref() {
            return Err(UpdateError::OnTrial(trial.world()));
        }
        if let Some(world) = replicas.iter().find_map(|replica| {
            locked(&replica.worlds)
                .trial
                .as_ref()
                .map(|trial| trial.world)
        }) {
            return Err(UpdateError::OnTrial(world));
        }
        let current = locked(&replicas[0].config).clone();
        let (target, stated, update) = resolve(&current, &request)?;
        let count = if target.moe_config.expert_parallel {
            1
        } else {
            target.data_parallel_size.get()
        };
        if count != replicas.len() {
            if request.replica.is_some() {
                return Err(UpdateError::Refused(
                    "a change across replica boundaries applies to the whole deployment".into(),
                ));
            }
            return group::apply(self, &request, target, stated, update).await;
        }
        let indices: Vec<_> = match request.replica {
            Some(index) => vec![index],
            None => (0..replicas.len()).collect(),
        };
        if !request.dry_run {
            locked(&self.replicas.pending).clear();
        }
        let mut reports = Vec::new();
        for (position, index) in indices.iter().copied().enumerate() {
            let replica = replicas
                .get(index)
                .ok_or_else(|| UpdateError::Refused(format!("no replica {index}")))?;
            match Box::pin(replica.apply(request.clone())).await? {
                Applied::World(report) => {
                    *locked(&self.replicas.pending) = indices[position + 1..]
                        .iter()
                        .map(|index| (*index, request.clone()))
                        .collect();
                    return Ok(Applied::World(report));
                }
                applied @ Applied::Daemon { .. } => {
                    if request.replica.is_some() {
                        return Err(UpdateError::Refused(
                            "daemon options apply to the deployment, not one replica".to_owned(),
                        ));
                    }
                    return Ok(applied);
                }
                Applied::Planned(report) | Applied::InPlace(report) => reports.push(report),
                Applied::Unchanged => {}
            }
        }
        if reports.is_empty() {
            return Ok(Applied::Unchanged);
        }
        let report = if reports.len() == 1 {
            reports.remove(0)
        } else {
            serde_json::json!({ "replicas": reports })
        };
        Ok(if request.dry_run {
            Applied::Planned(report)
        } else {
            Applied::InPlace(report)
        })
    }

    /// Commits the executing trial and releases its previous world.
    pub async fn commit(&self) -> Result<serde_json::Value, UpdateError> {
        let _replacing = self.replicas.replacing.lock().await;
        if locked(&self.replicas.commit).is_none() {
            if locked(&self.replicas.group).is_some() {
                group::decide_commit(&self.replicas).await?;
            } else {
                self.trial()?.decide_commit(&self.replicas).await?;
            }
        }
        complete_commit(&self.replicas).await
    }

    /// Restores the previous world of the executing trial.
    pub async fn rollback(&self, reason: String) -> Result<serde_json::Value, UpdateError> {
        let _replacing = self.replicas.replacing.lock().await;
        if locked(&self.replicas.commit).is_some() {
            return Err(UpdateError::Refused(
                "commit is decided; finish the pending commit".into(),
            ));
        }
        if locked(&self.replicas.group).is_some() {
            return group::rollback(&self.replicas, reason).await;
        }
        let replica = self.trial()?;
        locked(&self.replicas.pending).clear();
        replica.rollback(reason).await
    }

    /// Reports the configuration and resource ownership of the replicas.
    pub async fn config_status(&self) -> serde_json::Value {
        let replicas = self.replicas();
        let client = replicas
            .iter()
            .filter_map(|replica| replica.hold().client())
            .find(|client| matches!(client.state(), EngineState::Ready | EngineState::Parked));
        let ledgers = match client {
            Some(client) => {
                match tokio::time::timeout(LEDGER_READ, client.describe_storage()).await {
                    Ok(Ok(nodes)) => serde_json::Value::Array(
                        nodes
                            .iter()
                            .filter_map(|node| serde_json::from_str(node).ok())
                            .collect(),
                    ),
                    Ok(Err(error)) => serde_json::json!({ "error": error }),
                    Err(_) => serde_json::json!({
                        "error": format!("the ledgers were not read within {LEDGER_READ:?}")
                    }),
                }
            }
            None => serde_json::Value::Null,
        };
        // The bytes a change in progress moved: what every node's Transfer
        // Server copied for the deployment since it began, read with the
        // ledgers. One change runs at a time, so every transfer since is
        // the change's.
        let totals = ledgers.as_array().map(|nodes| node_transfers(nodes.iter()));
        let mut statuses = Vec::new();
        for replica in replicas {
            let mut status = replica.config_status();
            if !status["update"].is_null() {
                status["update"]["moved"] = match (replica.moved_from(), &totals) {
                    (Some(from), Some(totals)) => moved(&from, totals),
                    _ => serde_json::Value::Null,
                };
            }
            statuses.push(status);
        }
        let mut status = statuses
            .iter()
            .find(|status| !status["trial"].is_null() || !status["update"].is_null())
            .unwrap_or(&statuses[0])
            .clone();
        status["replicas"] = statuses.into();
        status["gpus"] = ledgers.as_array().map_or(serde_json::Value::Null, |nodes| {
            deployment_gpus(nodes).into()
        });
        status["ledgers"] = ledgers;
        if let Some(trial) = locked(&self.replicas.group).as_ref() {
            status["trial"] = trial.status();
        }
        status["commit"] = serde_json::json!(&*locked(&self.replicas.commit));
        status["pending_replicas"] = locked(&self.replicas.pending)
            .iter()
            .map(|(index, _)| *index)
            .collect::<Vec<_>>()
            .into();
        status
    }

    fn trial(&self) -> Result<Arc<Replica>, UpdateError> {
        self.replicas()
            .into_iter()
            .find(|replica| locked(&replica.worlds).trial.is_some())
            .ok_or_else(|| UpdateError::Refused("no world is on trial".to_owned()))
    }
}

fn resolve(
    current: &ServeConfig,
    request: &ApplyRequest,
) -> Result<
    (
        ServeConfig,
        crate::options::Stated,
        crate::options::UpdateOptions,
    ),
    UpdateError,
> {
    let (mut target, stated) = match &request.config {
        Some(text) => crate::options::resolve_text(text, &request.set),
        None => crate::options::overridden(current, &request.set)
            .map(|(config, paths)| (config, crate::options::Stated::from_paths(paths))),
    }
    .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
    target
        .validate(&stated)
        .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
    let update = target.update_config.clone();
    if request.config.is_none() {
        target.update_config = current.update_config.clone();
    }
    if target.served_model_name() != current.served_model_name() {
        return Err(UpdateError::Refused(
            "the served model name names the deployment: deploy another one instead".into(),
        ));
    }
    Ok((target, stated, update))
}

/// Records the operator's decision before publication or retirement. A
/// subsequent error can leave work pending, but cannot make rollback legal.
fn begin_commit(registry: &crate::leader::Replicas, current: &[Arc<Replica>], retire: Vec<u64>) {
    let worlds: Vec<_> = current
        .iter()
        .map(|replica| locked(&replica.worlds).executing)
        .collect();
    for replica in current {
        if let Some(spec) = locked(&replica.served).as_mut() {
            spec.predecessor = None;
            spec.stage_cache = false;
            spec.cache_keys.clear();
            spec.retire = retire.clone();
        }
        replica.hold.keep(None);
    }
    *locked(&registry.commit) = Some(Commit {
        pending: worlds.clone(),
        worlds,
        retire,
        nodes: None,
    });
}

/// Completes a recorded decision. Publication progress survives both an
/// error response and daemon handoff; node and engine operations are
/// idempotent when a response was lost after the operation took effect.
#[expect(
    clippy::unwrap_used,
    reason = "the deployment replacement lock retains the decision until its final take"
)]
pub(crate) async fn complete_commit(
    registry: &Arc<crate::leader::Replicas>,
) -> Result<serde_json::Value, UpdateError> {
    let decision = locked(&registry.commit)
        .clone()
        .context("no commit is pending")?;
    let replicas = registry.serving.borrow().clone();
    if !decision.nodes.as_ref().is_some_and(Vec::is_empty) {
        let control = Arc::clone(replicas[0].control.get().ok_or(UpdateError::NoEngine)?);
        let worlds = decision.worlds.clone();
        let mut nodes = decision.nodes;
        let (result, nodes) = tokio::task::spawn_blocking(move || {
            let result = control.publish_worlds(&worlds, &mut nodes);
            (result, nodes)
        })
        .await
        .context("the publication task did not complete")?;
        locked(&registry.commit).as_mut().unwrap().nodes = nodes;
        result.context("commit is decided; publication remains pending")?;
    }
    for world in decision.pending {
        let replica = replicas
            .iter()
            .find(|replica| locked(&replica.worlds).executing == world)
            .context("a committed world has no replica owner")?;
        let _owner = replica.replacing.lock().await;
        let client = match replica.hold.client() {
            Some(client)
                if !matches!(client.state(), EngineState::Failed(_) | EngineState::Closed) =>
            {
                client
            }
            _ => {
                // Publication of the entire decision completed above. Only
                // now may recovery start a spec that retires predecessors.
                let client = replica.recover().await?;
                replica.monitor();
                client
            }
        };
        client
            .commit(decision.retire.clone())
            .await
            .map_err(|error| {
                anyhow::anyhow!("commit is decided; world {world} remains pending: {error}")
            })?;
        let segments = client.describe_cache().await.map_err(|error| {
            anyhow::anyhow!(
                "commit is decided; reading world {world}'s cache remains pending: {error}"
            )
        })?;
        if let Some(spec) = locked(&replica.served).as_mut() {
            spec.segments = segments;
            spec.retire.clear();
        }
        if let Err(error) = client.resize_cache(wire::CacheChange::Grow).await {
            warn!(%error, world, "the committed world's cache did not grow");
        }
        locked(&registry.commit)
            .as_mut()
            .unwrap()
            .pending
            .retain(|pending| *pending != world);
    }
    let completed = locked(&registry.commit).take().unwrap();
    let mut committed = serde_json::json!({
        "world": (completed.worlds.len() == 1).then(|| completed.worlds[0]),
        "previous_world": (completed.retire.len() == 1).then(|| completed.retire[0]),
        "worlds": completed.worlds,
        "previous_worlds": completed.retire,
    });
    for replica in &replicas {
        let mut state = locked(&replica.worlds);
        if completed.worlds.contains(&state.executing) {
            state.last = Some(committed.clone());
        }
    }
    loop {
        let Some((index, request)) = locked(&registry.pending).pop_front() else {
            break;
        };
        let replica = Arc::clone(&registry.serving.borrow()[index]);
        match Box::pin(replica.apply(request)).await {
            Ok(Applied::World(report)) => {
                committed["next_trial"] = report;
                break;
            }
            Ok(Applied::InPlace(_) | Applied::Unchanged) => {}
            Ok(Applied::Daemon { .. } | Applied::Planned(_)) => {
                locked(&registry.pending).clear();
                committed["rollout_error"] = "the next replica requires a deployment update".into();
                break;
            }
            Err(error) => {
                locked(&registry.pending).clear();
                committed["rollout_error"] = error.to_string().into();
                break;
            }
        }
    }
    Ok(committed)
}

impl Replica {
    /// The digest of `target`, which names it, unless a rollback marked it
    /// unavailable and the update does not `force` it.
    fn available(&self, target: &ServeConfig, force: bool) -> Result<String, UpdateError> {
        let target_digest =
            digest(target).map_err(|error| UpdateError::Failed(error.context("digest")))?;
        if !force && let Some(unavailable) = locked(&self.worlds).unavailable.get(&target_digest) {
            return Err(UpdateError::Refused(format!(
                "the configuration {target_digest} was rolled back from world {} ({}); \
                 apply it again with force",
                unavailable.world, unavailable.reason
            )));
        }
        Ok(target_digest)
    }

    /// Changes the deployment to the configuration `request` states, as
    /// the module describes, choosing the least the change needs: nothing
    /// for the current configuration; the schedulers' next step for their
    /// limits alone; a hot restart of the daemon for its own options; and a
    /// new world for anything a world executes.
    ///
    /// # Errors
    ///
    /// Refuses another change in progress or a world on trial (`409`), a
    /// configuration that does not parse or validate, one a rollback marked
    /// unavailable unless `force`, and a change the successor cannot map;
    /// fails when the successor does not start (the serving world serves
    /// on) or its promotion fails (the serving world resumes).
    pub(crate) async fn apply(
        self: &Arc<Self>,
        request: ApplyRequest,
    ) -> Result<Applied, UpdateError> {
        let _replacing = self
            .replacing
            .try_lock()
            .map_err(|_| UpdateError::InProgress)?;
        if let Some(trial) = &locked(&self.worlds).trial {
            return Err(UpdateError::OnTrial(trial.world));
        }
        if self.asleep.load(Ordering::Acquire) {
            return Err(UpdateError::Refused(
                "the engine sleeps: wake it (/admin/wake_up) before updating it".to_owned(),
            ));
        }
        let current = locked(&self.config).clone();
        let (mut target, stated, mut update) = resolve(&current, &request)?;
        let mut target_digest = self.available(&target, request.force)?;
        if target == current && target.worker_groups.is_none() {
            return Ok(Applied::Unchanged);
        }
        let served = locked(&self.served).clone().ok_or(UpdateError::NoEngine)?;
        // The target's engine spec is resolved as a deployment's start
        // resolves it: from the options, completed by what the model's files
        // determine.
        let prepared = Leader::prepare(&target, &stated)
            .await
            .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
        let prepared_specs = engine::EngineConfig::replica_specs(&prepared.spec)
            .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
        let prepared_spec = prepared_specs.get(self.ordinal).cloned().ok_or_else(|| {
            UpdateError::Refused("the target changes the replica group".to_owned())
        })?;
        let mut target_spec = wire::EngineSpec {
            world: 0,
            predecessor: None,
            segments: Vec::new(),
            ..prepared_spec
        };
        let explicit_devices = target.worker_groups.is_some()
            || (stated.contains("placement_config.device")
                && target.placement_config.device != "cuda")
            || !target.placement_config.devices.is_empty();
        if !explicit_devices
            && target.tensor_parallel_size == current.tensor_parallel_size
            && target.data_parallel_size == current.data_parallel_size
            && target.placement_config.worker_hosts == current.placement_config.worker_hosts
            && target_spec.host == served.host
        {
            retain_devices(&served, &mut target_spec)
                .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
        }
        let running = self.hold().client().ok_or(UpdateError::NoEngine)?;
        let needs_world = comparable(&served) != comparable(&target_spec)
            && !crate::leader::configurable(
                &comparable(&served),
                &comparable(&target_spec),
                running.info().capacity.as_ref(),
            )
            && !switches_probes_only(&served, &target_spec);
        if prepared_specs.len() > 1 && needs_world {
            let control = Arc::clone(self.control.get().ok_or(UpdateError::NoEngine)?);
            let available = tokio::task::spawn_blocking(move || control.available_devices())
                .await
                .context("the device inventory task did not complete")??;
            assign_devices(&mut target_spec, available, explicit_devices)
                .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
        }
        let shared_devices = !devices(&served)?.is_disjoint(&devices(&target_spec)?);
        let mut world_changes = comparable(&served) != comparable(&target_spec);
        let daemon_changes = crate::options::daemon_options_differ(&current, &target);
        if target == current && !world_changes {
            return Ok(Applied::Unchanged);
        }

        // A world change: what each derived state of the serving world
        // becomes in its successor. Weights of other keys are loaded from the
        // model's files and those whose keys stay are mapped; the KV cache
        // is mapped unless the successor cannot read it (another encoding)
        // or the requests in flight compute their state again under other
        // weights.
        let unmapped = differing(&served, &target_spec);
        let reloaded = reloads(&served, &target_spec);
        let mut redistributes = parallel_change(&served, &target_spec);
        // Plan the target's own resources whenever ranks, quotas or pools
        // change. Only a change of parallel ranks may choose a device order;
        // a quota or queue adjustment keeps the declared assignment.
        let redistribution = if world_changes
            && (redistributes || resource_change(&served, &target_spec))
            && unmapped.is_empty()
        {
            let requests = running
                .describe_requests()
                .await
                .map_err(UpdateError::Refused)?;
            let planned = redistribution::plan(
                &self.manifests,
                self.control
                    .get()
                    .ok_or(UpdateError::NoEngine)?
                    .process_address(),
                self.control
                    .get()
                    .ok_or(UpdateError::NoEngine)?
                    .next_engine(),
                &[running.world()],
                &target_spec,
                redistributes && !stated.contains("placement_config.devices"),
                &requests,
            )
            .await
            .map_err(|error| {
                UpdateError::Refused(format!(
                    "the target world's resources cannot be planned: {error:#}"
                ))
            })?;
            if let Some(planned) = planned.as_ref().filter(|planned| planned.chosen) {
                target.placement_config.devices = planned.devices.clone();
                if let Some(placement) = target_spec.placement.as_mut() {
                    placement.devices = planned.devices.clone();
                }
                target_digest = self.available(&target, request.force)?;
                if target == current && target.worker_groups.is_none() {
                    return Ok(Applied::Unchanged);
                }
                world_changes = comparable(&served) != comparable(&target_spec);
                redistributes = parallel_change(&served, &target_spec);
            }
            planned.filter(|_| world_changes)
        } else {
            None
        };
        if let Some(planned) = &redistribution {
            target_spec.received_weights = planned.received.clone();
        }
        // Target rank resources include the concrete weight copies. A
        // successor with no reusable source weights loads its checkpoint.
        let budgets = redistribution
            .as_ref()
            .map(|planned| planned.budgets.clone())
            .unwrap_or_default();
        let target_hosts = redistribution
            .as_ref()
            .map(|planned| planned.hosts.clone())
            .unwrap_or_default();
        // A successor of another parallel configuration or another KV cache
        // encoding maps none of the serving world's cache, and one that
        // loads other weights does so when the requests in flight compute
        // their state again.
        let mut successor = Successor {
            loads_weights: !reloaded.is_empty() || (redistributes && redistribution.is_none()),
            unmapped_cache: redistributes
                || kv_encoding_changes(&served, &target_spec)
                || cache_capacity_change(&served, &target_spec)
                || (!reloaded.is_empty() && request.in_flight == InFlight::Recompute),
            host_cache: false,
            eager_source: update.eager_prepare,
            captures_graphs: target_spec
                .cuda_graphs
                .as_ref()
                .is_none_or(|graphs| graphs.policy != "off"),
        };
        let in_place = crate::leader::configurable(
            &comparable(&served),
            &comparable(&target_spec),
            running.info().capacity.as_ref(),
        );
        // A successor serving other model files processes requests as the
        // daemon does only with the same tokenizer, templates, generation
        // defaults and model facts.
        if served.model != target_spec.model {
            crate::leader::same_processing(&served.model, &target_spec.model)
                .await
                .map_err(UpdateError::Refused)?;
            if served.model_facts != target_spec.model_facts {
                return Err(UpdateError::Refused(format!(
                    "the model at {} has other generation limits, control tokens or context \
                     length than the served model, which change how requests are processed: \
                     deploy it instead",
                    target_spec.model
                )));
            }
        }
        // A change a new world executes is checked against every device's
        // storage ledger first: the serving world's KV cache limit beside a
        // successor, with the weights it creates, per GPU and host
        // (`specs/online-operations.md` §6.6).
        let storage = if world_changes && !in_place && unmapped.is_empty() && shared_devices {
            Some(
                storage_plan(
                    std::slice::from_ref(&running),
                    &mut successor,
                    &budgets,
                    &target_hosts,
                    self.control.get().ok_or(UpdateError::NoEngine)?,
                    update.previous_world,
                )
                .await,
            )
        } else {
            None
        };
        update.eager_prepare = successor.eager_source;
        // Every decision the world change executes, made once: the plan
        // reports them and `change_world` follows them.
        let change = WorldChange {
            successor,
            redistributes,
            shrinks: shared_devices,
            reencodes_cache: kv_encoding_changes(&served, &target_spec),
            in_flight: request.in_flight,
            previous_world: update.previous_world,
            serving_drafts: served.speculation.is_some(),
            successor_drafts: target_spec.speculation.is_some(),
        };
        // The steps and treatments of a change a new world executes; a
        // change in place or of the daemon alone has none, and one the
        // successor cannot map is refused below.
        let new_world = world_changes
            && !in_place
            && !switches_probes_only(&served, &target_spec)
            && unmapped.is_empty();
        let copies = PlannedCopies::new(
            storage.as_ref().map(|(plan, _)| plan),
            &budgets,
            &target_hosts,
            &redistribution
                .as_ref()
                .map(|planned| &planned.report)
                .into_iter()
                .collect::<Vec<_>>(),
        );
        // A warmed request computes at most the tokens past the last
        // warming round's copies: the configured bound, else a page, of at
        // most 64 tokens when the page size is the kernels' choice
        // (`kv_cache_config.block_size`).
        let tail_tokens = update
            .cache_tail_tokens
            .map(std::num::NonZeroU32::get)
            .or_else(|| {
                Some(
                    target_spec
                        .kv_cache
                        .as_ref()
                        .and_then(|cache| cache.block_size)
                        .unwrap_or(64),
                )
            });
        let plan = serde_json::json!({
            "from": locked(&self.worlds).executing,
            "replica": self.ordinal,
            "shared_devices": shared_devices,
            "devices": devices(&target_spec)?,
            "digest": target_digest,
            "world": world_changes,
            "daemon": daemon_changes,
            "previous_world": update.previous_world,
            "in_flight": request.in_flight,
            "steps": new_world.then(|| copies.steps(&change)),
            "derived_state": new_world.then(|| change.derived_state(Observed::default())),
            "promotion_backfill": new_world
                .then(|| copies.backfill(&change, running.in_flight(), tail_tokens)),
            "reloads": reloaded,
            "differs": unmapped,
            "in_flight_requests": running.in_flight(),
            "storage": storage.as_ref().map(|(plan, _)| plan),
            "redistribution": redistribution.as_ref().map(|planned| &planned.report),
        });
        if request.dry_run {
            return Ok(Applied::Planned(plan));
        }
        if !world_changes {
            if daemon_changes {
                return Ok(Applied::Daemon {
                    config: Box::new(target),
                    executable: None,
                    then: None,
                });
            }
            *locked(&self.config) = target;
            return Ok(Applied::InPlace(plan));
        }
        // The daemon orchestrates what changes in its world, its scheduling
        // limits included: a hot restart applies its own options first, and
        // its successor applies the change again, which then leaves the
        // daemon's options as they are.
        if daemon_changes {
            return Ok(Applied::Daemon {
                config: Box::new(crate::options::with_daemon_options(&current, &target)),
                executable: None,
                then: Some(Box::new(request)),
            });
        }
        // The fence probes' switch alone changes in place: every worker
        // writes its device switch at the engine's next step boundary, and
        // the graphs that captured the probes stay.
        if switches_probes_only(&served, &target_spec) {
            let enabled = target_spec
                .profiling
                .as_ref()
                .is_some_and(|profiling| profiling.fence_probes);
            let off = target_spec
                .profiling
                .as_ref()
                .map(|profiling| profiling.fence_probes_off.clone())
                .unwrap_or_default();
            running
                .hook(
                    "uniserve.fence_switch",
                    &serde_json::json!({ "enabled": enabled, "off": off }).to_string(),
                )
                .await
                .map_err(|error| {
                    anyhow::anyhow!("the workers did not switch the probes: {error}")
                })?;
            info!(
                enabled,
                ?off,
                "the workers switched the fence probes in place"
            );
            let mut spec = served;
            spec.profiling = target_spec.profiling;
            *locked(&self.served) = Some(spec);
            *locked(&self.config) = target;
            return Ok(Applied::InPlace(plan));
        }
        // Scheduling alone, within the batch the serving world's workers
        // were started for, changes in place at the schedulers' next step.
        if in_place {
            let scheduling = target_spec.scheduling.clone().unwrap_or_default();
            running
                .configure(scheduling)
                .await
                .map_err(|error| anyhow::anyhow!("the engine did not take the limits: {error}"))?;
            info!("the engine took the new scheduling limits in place");
            let mut spec = served;
            spec.scheduling = target_spec.scheduling;
            *locked(&self.served) = Some(spec);
            *locked(&self.config) = target;
            return Ok(Applied::InPlace(plan));
        }
        if !unmapped.is_empty() {
            return Err(UpdateError::Refused(format!(
                "the update changes {}, which decides how the successor world divides the \
                 weights and the KV cache: a world change carries over only what keeps them \
                 divided alike",
                unmapped.join(", ")
            )));
        }
        // A successor that does not fit beside the serving world on some
        // GPU is refused before anything starts, naming each item there.
        if let Some((_, Some(refusal))) = storage {
            return Err(UpdateError::Refused(refusal));
        }
        let redistribution = match redistribution {
            Some(Redistribution {
                refusal: Some(refusal),
                ..
            }) => return Err(UpdateError::Refused(refusal)),
            Some(planned) => Some(planned.report),
            None => None,
        };
        // The world change holds every stage's state across its awaits; boxed,
        // it does not grow the caller's future by that much.
        let report = Box::pin(self.change_world(
            &running,
            served,
            current,
            target,
            target_spec,
            target_digest,
            &update,
            change,
            &budgets,
            &target_hosts,
            redistribution,
        ))
        .await?;
        Ok(Applied::World(report))
    }

    /// Moves the deployment from the world `running` executes to a
    /// successor executing `target`, up to its trial (steps 1 to 3 of the
    /// module), taking the steps `change` decided in their order and
    /// reporting what became of each derived state. A successor of another
    /// parallel configuration maps none of the serving world's KV cache, and
    /// the requests in flight continue from the pages warmed into it; it
    /// creates the resources `budgets` states on each GPU, which the
    /// serving world's cache makes room for.
    #[expect(
        clippy::too_many_arguments,
        reason = "one world change: both worlds' specs and configurations and its options"
    )]
    async fn change_world(
        self: &Arc<Self>,
        running: &Arc<EngineClient>,
        mut served: wire::EngineSpec,
        current: ServeConfig,
        target: ServeConfig,
        mut spec: wire::EngineSpec,
        target_digest: String,
        update: &crate::options::UpdateOptions,
        change: WorldChange,
        budgets: &BTreeMap<storage::manifest::Gpu, storage::manifest::weights::RankBudget>,
        target_hosts: &BTreeMap<storage::manifest::Gpu, String>,
        redistribution: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, UpdateError> {
        let started = Instant::now();
        let previous_world = running.world();
        let (adds, in_flight) = (change.successor, change.in_flight);
        // What the deployment's transfers had copied on every node as the
        // change begins: the status reports what it moved since.
        let moved_from = transfer_totals(running).await;
        self.progress(change.steps(Numerics::SAME)[0], None);
        self.progress_edit(|progress| progress.moved_from = moved_from);
        // The serving world gives its graph storage to the successor's
        // captures first, so the shrink that follows leaves it to the
        // successor.
        let eager_since = if adds.eager_source {
            self.progress(Step::Eager, None);
            if let Err(error) = running.hook("uniserve.eager", r#"{"eager":true}"#).await {
                // Ranks that switched replay their graphs again; the others
                // switch nothing.
                replay_graphs(running, true).await;
                self.progress_done();
                return Err(UpdateError::Refused(format!(
                    "the serving world cannot run eagerly while its successor prepares \
                     (update_config.eager_prepare): {error}"
                )));
            }
            Some(Instant::now())
        } else {
            None
        };
        // The serving world's cache gives back the tail past the limit each
        // GPU's ledger leaves it beside the successor; its requests run on.
        let shrunk =
            if change.shrinks {
                self.progress(Step::Shrink, None);
                running
                .make_room(wire::CacheChange::Shrink, adds, update.preempt_after, budgets)
                .await
                .map_err(|error| {
                    self.progress_done();
                    anyhow::anyhow!(
                        "the KV cache did not make room for a successor world; the serving world \
                     serves on: {error}"
                    )
                })
            } else {
                Ok(Vec::new())
            };
        // A shrink refused or failed leaves the serving world serving: it
        // replays its graphs again when it ran eagerly for the successor.
        let shrunk = match shrunk {
            Ok(shrunk) => shrunk,
            Err(error) => {
                replay_graphs(running, update.eager_prepare).await;
                return Err(error.into());
            }
        };
        let control = self.control.get().ok_or(UpdateError::NoEngine)?;
        let storage = storage_report(
            &shrunk,
            update.previous_world,
            adds,
            budgets,
            target_hosts,
            control,
        )
        .await
        .and_then(|report| {
            if report["feasible"] == true {
                Ok(report)
            } else {
                Err(report.to_string())
            }
        });
        let storage = match storage {
            Ok(report) => report,
            Err(error) => {
                self.restore_tail(running).await;
                self.progress_done();
                replay_graphs(running, update.eager_prepare).await;
                return Err(UpdateError::Refused(error));
            }
        };
        // A successor of another parallel configuration lays its cache out
        // for the heads it holds, and one of another encoding cannot read the
        // serving world's: either maps none of its segments.
        let segments = if adds.unmapped_cache {
            Vec::new()
        } else {
            running.describe_cache().await.map_err(|error| {
                self.progress_done();
                anyhow::anyhow!("the serving world did not describe its KV cache: {error}")
            })?
        };
        spec.world = 0;
        spec.stage_cache = adds.host_cache;
        spec.predecessor = Some(wire::Predecessor {
            world: previous_world,
            segments,
        });

        // A serving world that switched to eager forwards steps slower than
        // its graphs did, so the successor starts once its eager steps make
        // the transfers' baseline, or its schedulers without one prove idle
        // (`crate::interference`).
        let baseline_wait = match eager_since {
            Some(since) => {
                self.progress(Step::Baseline, None);
                let waiting = Instant::now();
                let idle = crate::interference::eager_baseline(
                    &self.observer,
                    previous_world,
                    running.info().schedulers,
                    since,
                )
                .await;
                if !idle.is_empty() {
                    self.progress_note(format!(
                        "the serving world's schedulers {idle:?} ran no step for {:?} after \
                         their switch to eager forwards: the successor's transfers start \
                         without a baseline of their steps",
                        crate::interference::IDLE
                    ));
                }
                Some(waiting.elapsed())
            }
            None => None,
        };

        let control = Arc::clone(self.control.get().ok_or(UpdateError::NoEngine)?);
        // The successor takes the control's next world identity as it
        // starts: one change runs at a time.
        self.progress(Step::Prepare, Some(control.next_engine()));
        let sink = stats_sink(
            self.model_name.clone(),
            Arc::clone(&self.observer),
            self.ordinal,
        );
        info!(previous_world, "preparing a successor world");
        // While the successor prepares, its transfers yield to the serving
        // world's steps (`update_config.transfer_slowdown`).
        let interference = {
            let done = tokio_util::sync::CancellationToken::new();
            let throttling = crate::interference::Throttling::new(
                &self.observer,
                previous_world,
                update.transfer_slowdown,
                Instant::now(),
                eager_since,
            );
            let steps = Arc::new(std::sync::Mutex::new(
                crate::interference::StepTime::default(),
            ));
            self.progress_edit(|progress| progress.steps = Some(Arc::clone(&steps)));
            let task = tokio::spawn(crate::interference::control(
                Arc::clone(running),
                vec![(Arc::clone(&self.observer), throttling)],
                crate::interference::INTERVAL,
                done.clone(),
                steps,
            ));
            (done, task)
        };
        let start_spec = spec.clone();
        let prepared = async {
            let successor = Arc::new(
                tokio::task::spawn_blocking(move || control.start(start_spec, sink))
                    .await
                    .context("the successor's start did not complete")?
                    .context("the successor world did not start")?,
            );
            // The successor's workers derived their numerical identity:
            // whether they compute other numerics decides what the warming
            // and the promotion do with the requests' state.
            let (before, after) = (running.info(), successor.info());
            let numerics = Numerics {
                pages: before.prefix_seeds != after.prefix_seeds,
                trajectories: before.latent_seed != after.latent_seed,
            };
            let warmed = if change.warms(numerics) {
                self.progress(Step::WarmCache, Some(successor.world()));
                match warming::prepare(
                    std::slice::from_ref(running),
                    &successor,
                    change.reuse(numerics),
                    update.cache_tail_tokens.map(std::num::NonZeroU32::get),
                    update.cache_prefill_tokens.map(std::num::NonZeroU32::get),
                    adds.host_cache,
                )
                .await
                {
                    Ok(report) => Some(report),
                    Err(error) => {
                        self.close(&successor).await;
                        return Err(anyhow::anyhow!("cache warming failed: {error}"));
                    }
                }
            } else {
                None
            };
            Ok((successor, numerics, warmed))
        }
        .await;
        let (done, task) = interference;
        done.cancel();
        let throttled = task.await.unwrap_or_default();
        let (successor, numerics, warmed) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                self.progress_done();
                let restored = running.resize_cache(wire::CacheChange::Restore).await;
                replay_graphs(running, update.eager_prepare).await;
                return Err(match restored {
                    Ok(_) => error.context("the serving world serves on"),
                    Err(restoring) => error.context(format!(
                        "the serving world serves on without the tail of its KV cache, which it \
                         did not create again: {restoring}"
                    )),
                }
                .into());
            }
        };
        let world = successor.world();
        let prepare = started.elapsed();
        self.progress(Step::Park, Some(world));

        // The promotion: the serving world's step boundary, its index, the
        // successor's import.
        let baseline = update.rollback.window.and_then(|window| {
            self.observer.rates(
                previous_world,
                started.checked_sub(window).unwrap_or(started),
                window,
            )
        });
        // A successor computing other numerics than the serving world (its
        // pages' prefix seed, or its stored trajectories' latent seed,
        // differs) reads none of its pages or trajectories, but the requests
        // in flight continue with the state they reached unless the change
        // recomputes them.
        let before = running.info();
        let numerics_change = numerics.pages || numerics.trajectories;
        let hold = Arc::clone(self.hold());
        // Only these continuations may reuse state computed with the prior
        // numerics, whether its backing is mapped or copied.
        if numerics_change && in_flight == InFlight::Keep {
            hold.keep(Some(Kept {
                world: previous_world,
                prefix_seeds: numerics.pages.then(|| before.prefix_seeds.clone()),
                latent_seed: numerics.trajectories.then_some(before.latent_seed),
            }));
        }
        // A profiling session belongs to the serving world: it ends, its
        // files written, before the world parks.
        crate::profiling::end_for_world(self, Some(running), "world change").await;
        let paused = Instant::now();
        hold.hold();
        self.progress_edit(|progress| progress.paused = Some(paused));
        let index = match running.park().await {
            Ok(index) => index,
            Err(error) => {
                // The serving world serves on: it resumes, and the successor
                // closes with its storage.
                hold.keep(None);
                let _ = running.resume(vec![world]).await;
                hold.serve(Arc::clone(running));
                self.close(&successor).await;
                self.restore_tail(running).await;
                replay_graphs(running, update.eager_prepare).await;
                let _ = warming::resume_prefill(std::slice::from_ref(running)).await;
                self.progress_done();
                return Err(anyhow::anyhow!("the serving world did not park: {error}").into());
            }
        };
        let parked = paused.elapsed();
        // Recompute only when requested for changed numerics or when the
        // encoding has no registered exact conversion.
        let recompute = (numerics_change && in_flight == InFlight::Recompute)
            || warmed
                .as_ref()
                .is_some_and(|report| report.recompute.is_some());
        // A previous world kept parked keeps its storage until the commit; one
        // kept as its weights frees the rest as the successor serves.
        let grows = update.previous_world != PreviousWorld::Parked;
        let index = if adds.unmapped_cache {
            Vec::new()
        } else {
            index
        };
        let mut evicted: Option<serde_json::Value> = None;
        // The pause's parts after the park, each measured where it ends:
        // the parked world's pending prefills and the eviction of its cache
        // segments, the successor's loading of the copies staged in host
        // storage (both only when the warm copies were staged there), and
        // the successor's promotion.
        let mut evicting = Duration::ZERO;
        let mut materializing = Duration::ZERO;
        let mut promoting = Duration::ZERO;
        let mut materialized = Vec::new();
        let promotion: anyhow::Result<()> = async {
            let step = Instant::now();
            warming::resume_prefill(std::slice::from_ref(running))
                .await
                .map_err(anyhow::Error::msg)?;
            if adds.host_cache {
                self.progress(Step::Evict, None);
                let results = running
                    .resize_cache(wire::CacheChange::Evict)
                    .await
                    .map_err(anyhow::Error::msg)?;
                evicted = results
                    .iter()
                    .find_map(|result| serde_json::from_str(result).ok());
                evicting = step.elapsed();
                self.progress(Step::Materialize, None);
                let step = Instant::now();
                materialized = warming::materialize(&successor)
                    .await
                    .map_err(anyhow::Error::msg)?;
                materializing = step.elapsed();
            } else {
                evicting = step.elapsed();
            }
            self.progress(Step::Promote, None);
            let step = Instant::now();
            successor.promote(index, recompute, grows).await?;
            promoting = step.elapsed();
            Ok(())
        }
        .await;
        if let Err(error) = promotion {
            hold.keep(None);
            let _ = successor.hook("uniserve.cache_cancel", "null").await;
            self.close(&successor).await;
            let _ = running.resume(vec![world]).await;
            if let Some(warmed) = &warmed {
                warming::restore(running, &warmed.backup)
                    .await
                    .map_err(anyhow::Error::msg)?;
            }
            hold.serve(Arc::clone(running));
            self.restore_tail(running).await;
            replay_graphs(running, update.eager_prepare).await;
            self.progress_done();
            return Err(anyhow::anyhow!(
                "the successor world was not promoted; the previous world resumed: {error}"
            )
            .into());
        }
        hold.serve(Arc::clone(&successor));
        let pause = paused.elapsed();
        info!(
            world,
            previous_world,
            pause_ms = pause.as_millis() as u64,
            park_ms = parked.as_millis() as u64,
            evict_ms = evicting.as_millis() as u64,
            materialize_ms = materializing.as_millis() as u64,
            promote_ms = promoting.as_millis() as u64,
            "the successor world serves on trial"
        );
        crate::leader::reload_adapters(self, &successor).await;

        // The previous world as the trial keeps it.
        self.progress(Step::KeepPrevious, None);
        let previous = match update.previous_world {
            PreviousWorld::Parked => Previous::Parked(Arc::clone(running)),
            PreviousWorld::Weights | PreviousWorld::Host => {
                let closed = Arc::clone(running);
                let exit = tokio::task::spawn_blocking(move || {
                    closed.shutdown();
                    closed.await_exit(CLOSED_EXIT)
                })
                .await
                .context("the previous world's shutdown did not complete")?;
                if let Err(error) = exit {
                    warn!("the previous world's engine process: {error:#}");
                }
                Previous::Weights
            }
        };
        // The successor's cache grows into the storage the previous world
        // freed: a previous world kept as its weights released, with its
        // processes, their graph pools and process storage, and keeps its
        // weights alone, so every other storage of it the successor does
        // not map is released first; kept in host storage, its weights that
        // the successor does not map move there too. Each GPU's ledger
        // leaves the successor all of it at once. A node that cannot move
        // the weights (no Transfer Server, or more than the host admits)
        // keeps them on its GPUs, and the previous world is kept as its
        // weights.
        let to_host = update.previous_world == PreviousWorld::Host;
        let mut mode = update.previous_world;
        let mut host_refused = None;
        let mut moved_bytes = 0;
        match &previous {
            Previous::Parked(parked) => {
                // A parked previous world whose cache the successor maps none
                // of keeps it for a rollback alone: it evicts every segment
                // past its first once their idle pages reached the host KV
                // cache, and the successor grows into them (step 5 of
                // `specs/online-operations.md` §6.5). A rollback reads them
                // back from there and computes the rest again. Host staging
                // evicted them at the promotion, where the successor loaded
                // its staged cache.
                if !(adds.unmapped_cache && adds.host_cache) {
                    if adds.unmapped_cache {
                        match parked.resize_cache(wire::CacheChange::Evict).await {
                            Ok(results) => {
                                evicted = results
                                    .iter()
                                    .find_map(|result| serde_json::from_str(result).ok());
                            }
                            Err(error) => {
                                warn!("the parked world did not evict its KV cache: {error}");
                            }
                        }
                    }
                    if let Err(error) = successor.resize_cache(wire::CacheChange::Grow).await {
                        warn!("the successor's writable cache did not grow: {error}");
                    }
                }
            }
            Previous::Weights => {
                match successor
                    .grow_keeping_weights(
                        vec![previous_world],
                        to_host,
                        warmed
                            .as_ref()
                            .map_or_else(Vec::new, |report| warming::keys(&report.backup)),
                    )
                    .await
                {
                    Ok(nodes) => {
                        let refusals: Vec<String> = nodes
                            .iter()
                            .filter_map(|node| {
                                let refused = node["refused"].as_str()?;
                                Some(format!("{}: {refused}", node["host"].as_str()?))
                            })
                            .collect();
                        moved_bytes = nodes
                            .iter()
                            .filter_map(|node| node["moved_bytes"].as_u64())
                            .sum::<u64>();
                        if !refusals.is_empty() {
                            host_refused = Some(refusals.join("; "));
                        }
                    }
                    Err(error) => {
                        warn!("the successor's KV cache did not grow: {error}");
                        if to_host {
                            host_refused = Some(error);
                        }
                    }
                }
            }
        }
        if let Some(refused) = &host_refused {
            warn!(
                refused,
                "the previous world's weights stay on the GPUs: it is kept as its weights"
            );
            mode = PreviousWorld::Weights;
        }

        // A relaunch of the world binds the weights it retains: it receives
        // none again.
        let mut served_spec = spec;
        served_spec.world = world;
        served_spec.received_weights.clear();
        served_spec.stage_cache = false;
        served_spec.cache_keys.clear();
        *locked(&self.served) = Some(served_spec);
        *locked(&self.config) = target;
        // The steps the change took, with what each moved over each link,
        // and what became of each derived state under the numerics the
        // successor computes and the conversion the warming found.
        let weight_links = PlannedCopies::new(
            None,
            budgets,
            target_hosts,
            &redistribution.iter().collect::<Vec<_>>(),
        )
        .weights;
        let steps: Vec<serde_json::Value> = change
            .steps(numerics)
            .into_iter()
            .map(|step| {
                let mut entry = serde_json::json!({ "step": step.name() });
                match step {
                    Step::Prepare if change.redistributes && !adds.loads_weights => {
                        entry["links"] = weight_links.clone().into();
                    }
                    Step::WarmCache => {
                        entry["links"] = warmed
                            .as_ref()
                            .map(|report| change::link_loads(&report.links, target_hosts))
                            .into();
                    }
                    Step::Materialize => {
                        entry["links"] = change::link_loads(&materialized, target_hosts).into();
                    }
                    Step::KeepPrevious => entry["previous_world"] = serde_json::json!(mode),
                    _ => {}
                }
                entry
            })
            .collect();
        let derived_state = change.derived_state(Observed {
            numerics: Some(numerics),
            converted: warmed
                .as_ref()
                .map(|report| report.recompute.is_none() || change.recomputes(numerics)),
        });
        let report = serde_json::json!({
            "world": world,
            "previous_world": previous_world,
            "prepare_seconds": prepare.as_secs_f64(),
            "eager_baseline_seconds": baseline_wait.map(|wait| wait.as_secs_f64()),
            "transfer_halvings": throttled.halvings,
            "transfer_restorations": throttled.restorations,
            "park_seconds": parked.as_secs_f64(),
            "evict_seconds": evicting.as_secs_f64(),
            "materialize_seconds": materializing.as_secs_f64(),
            "promote_seconds": promoting.as_secs_f64(),
            "pause_seconds": pause.as_secs_f64(),
            "previous_world_kept": mode,
            "host_refused": host_refused,
            "moved_to_host_bytes": to_host.then_some(moved_bytes),
            "evicted": evicted,
            "numerics": if numerics_change { "changed" } else { "unchanged" },
            "parallel": if change.redistributes { "changed" } else { "unchanged" },
            "redistribution": redistribution,
            "in_flight": if numerics_change { Some(in_flight) } else { None },
            "steps": steps,
            "derived_state": derived_state,
            "cache_warming": warmed,
            "storage": storage,
        });
        {
            served.cache_keys = warmed
                .as_ref()
                .map_or_else(Vec::new, |report| warming::keys(&report.backup));
            let mut worlds = locked(&self.worlds);
            worlds.executing = world;
            worlds.progress = None;
            worlds.trial = Some(Trial {
                cache_backup: warmed
                    .as_ref()
                    .map_or_else(Vec::new, |report| report.backup.clone()),
                world,
                digest: target_digest,
                previous_world,
                previous_spec: served,
                previous_config: current,
                previous,
                mode,
                host_refused,
                promoted: paused,
                promoted_at: now(),
                baseline,
                thresholds: update.rollback.clone(),
                outcomes: crate::leader::outcomes(),
                pause,
                eager: update.eager_prepare,
            });
            worlds.last = Some(report.clone());
        }
        watch(Arc::clone(self), Arc::downgrade(&successor));
        tokio::spawn(watch_trial(Arc::clone(self), world));
        Ok(report)
    }

    /// Decides the current trial and closes its predecessor. Publication
    /// and retirement complete through the deployment's durable decision.
    async fn decide_commit(
        self: &Arc<Self>,
        registry: &Arc<crate::leader::Replicas>,
    ) -> Result<(), UpdateError> {
        let _replacing = self.replacing.lock().await;
        let trial = locked(&self.worlds)
            .trial
            .take()
            .ok_or_else(|| UpdateError::Refused("no world is on trial".into()))?;
        begin_commit(registry, &[Arc::clone(self)], vec![trial.previous_world]);
        if let Previous::Parked(previous) = trial.previous {
            close(&previous).await;
        }
        Ok(())
    }

    /// Rolls the world on trial back to the previous world, for `reason`.
    ///
    /// # Errors
    ///
    /// Refuses when no world is on trial; fails when the previous world
    /// does not serve again, which leaves no world serving.
    pub(crate) async fn rollback(
        self: &Arc<Self>,
        reason: String,
    ) -> Result<serde_json::Value, UpdateError> {
        let _replacing = self.replacing.lock().await;
        rollback(self, reason).await
    }

    /// This replica's configuration, its two slots, the change in progress,
    /// the trial and its update history.
    ///
    /// The first slot holds the world that serves (or, while requests wait
    /// for a promotion or a rollback, the one that served last); the second
    /// the world beside it. Each slot's `state` is one of `empty`,
    /// `preparing`, `executing`, `parked`, `weights` (its weights alone),
    /// `host` (its weights in host storage) or `unavailable` (its engine
    /// failed or ended, or a rollback stops it), with the `reason` of an
    /// unavailable one.
    pub(crate) fn config_status(&self) -> serde_json::Value {
        let config = locked(&self.config).clone();
        let printed = crate::options::printed(&config).unwrap_or_default();
        let digest = digest(&config).unwrap_or_default();
        let worlds = locked(&self.worlds);
        let hold = self.hold();
        let progress = worlds.progress.as_ref();
        let rolling_back = progress.is_some_and(|progress| progress.phase == ROLLBACK);
        let slot = |world: u64, state: &str| serde_json::json!({ "world": world, "state": state });
        let unavailable = |world: u64, reason: &str| serde_json::json!({ "world": world, "state": "unavailable", "reason": reason });
        let (serving, executing) = match hold.client() {
            Some(client) => {
                let world = client.world();
                let state = match client.state() {
                    EngineState::Standby => slot(world, "preparing"),
                    EngineState::Ready => slot(world, "executing"),
                    EngineState::Parked => slot(world, "parked"),
                    EngineState::Failed(reason) => unavailable(world, &reason),
                    EngineState::Closed => unavailable(world, "its engine process ended"),
                };
                (world, state)
            }
            // No engine takes requests: the serving world parked for a
            // promotion, a rollback stops the world on trial, or the
            // deployment recovers a lost engine or closed.
            None => {
                let world = worlds.executing;
                let state = match hold.state() {
                    crate::engine::hold::HoldState::Closed(reason) => unavailable(world, &reason),
                    _ if rolling_back => unavailable(world, "a rollback stops it"),
                    _ if progress.is_some_and(|progress| progress.paused.is_some()) => {
                        slot(world, "parked")
                    }
                    _ => unavailable(world, "requests wait for its engine"),
                };
                (world, state)
            }
        };
        let other = match (&worlds.trial, progress) {
            (Some(trial), _) => slot(
                trial.previous_world,
                match trial.mode {
                    PreviousWorld::Parked => "parked",
                    PreviousWorld::Weights => "weights",
                    PreviousWorld::Host => "host",
                },
            ),
            // Once its successor serves, the world that served is parked
            // until the change keeps it as it states, or stops while a
            // rollback resumes the previous world.
            (
                None,
                Some(Progress {
                    world: Some(world), ..
                }),
            ) if *world == serving => {
                if rolling_back {
                    unavailable(worlds.executing, "a rollback stops it")
                } else {
                    slot(worlds.executing, "parked")
                }
            }
            (
                None,
                Some(Progress {
                    world: Some(world), ..
                }),
            ) => slot(*world, "preparing"),
            _ => serde_json::json!({ "world": null, "state": "empty" }),
        };
        let update = progress.map(|progress| {
            let mut update = serde_json::json!(progress);
            update["seconds"] = (now() - progress.started).max(0.0).into();
            update["pause_seconds"] = progress
                .paused
                .map(|paused| paused.elapsed().as_secs_f64())
                .into();
            update["step_time"] = progress
                .steps
                .as_ref()
                .map(|steps| *locked(steps))
                .map_or(serde_json::Value::Null, |steps| serde_json::json!(steps));
            update
        });
        let trial = worlds.trial.as_ref().map(|trial| {
            let rates = trial
                .thresholds
                .window
                .and_then(|window| self.observer.rates(trial.world, trial.promoted, window));
            let (ended, failed) = crate::leader::outcomes();
            serde_json::json!({
                "world": trial.world,
                "previous_world": trial.previous_world,
                "previous_world_kept": trial.mode,
                "host_refused": trial.host_refused,
                "seconds": trial.promoted.elapsed().as_secs_f64(),
                "promoted_at": trial.promoted_at,
                "pause_seconds": trial.pause.as_secs_f64(),
                "thresholds": trial.thresholds,
                "requests_ended": ended.saturating_sub(trial.outcomes.0),
                "requests_failed": failed.saturating_sub(trial.outcomes.1),
                "baseline": trial.baseline,
                "rates": rates,
                "step_time_change": match (trial.baseline, rates) {
                    (Some(baseline), Some(rates)) if baseline.step_us > 0.0 =>
                        Some(rates.step_us / baseline.step_us - 1.0),
                    _ => None,
                },
            })
        });
        serde_json::json!({
            "config": printed,
            "digest": digest,
            "slots": [executing, other],
            "update": update,
            "trial": trial,
            "unavailable": worlds.unavailable.values().collect::<Vec<_>>(),
            "last": worlds.last,
        })
    }

    /// What every node had counted of the deployment's transfers when the
    /// change in progress began, by host.
    fn moved_from(&self) -> Option<BTreeMap<String, Vec<storage::protocol::TransferTotal>>> {
        locked(&self.worlds)
            .progress
            .as_ref()
            .and_then(|progress| progress.moved_from.clone())
    }

    /// Records that the change in progress (a new one when none is) takes
    /// `step`, preparing `world` when given; what it recorded before stays.
    fn progress(&self, step: Step, world: Option<u64>) {
        let mut worlds = locked(&self.worlds);
        let progress = worlds.progress.get_or_insert_with(|| Progress {
            phase: String::new(),
            started: now(),
            world: None,
            note: None,
            moved_from: None,
            paused: None,
            steps: None,
        });
        step.name().clone_into(&mut progress.phase);
        if world.is_some() {
            progress.world = world;
        }
    }

    /// Changes what the change in progress recorded, if one is.
    fn progress_edit(&self, edit: impl FnOnce(&mut Progress)) {
        if let Some(progress) = locked(&self.worlds).progress.as_mut() {
            edit(progress);
        }
    }

    /// Notes `note` on the change in progress, for its later phases too.
    fn progress_note(&self, note: String) {
        self.progress_edit(|progress| progress.note = Some(note));
    }

    fn progress_done(&self) {
        locked(&self.worlds).progress = None;
    }

    /// Shuts the engine `client` serves down and waits for its process.
    async fn close(&self, client: &Arc<EngineClient>) {
        close(client).await;
    }

    /// Creates again the tail of the KV cache the serving world gave back for
    /// a successor that did not take over, once the successor is closed.
    async fn restore_tail(&self, running: &Arc<EngineClient>) {
        if let Err(error) = running.resize_cache(wire::CacheChange::Restore).await {
            warn!("the serving world did not create the tail of its KV cache again: {error}");
        }
    }
}

/// Has `running`, which served eagerly while a successor prepared (when
/// `eager`), replay its graphs again now that it serves on.
async fn replay_graphs(running: &Arc<EngineClient>, eager: bool) {
    if !eager {
        return;
    }
    if let Err(error) = running.hook("uniserve.eager", r#"{"eager":false}"#).await {
        warn!("the serving world did not replay its graphs again: {error}");
    }
}

/// Shuts the engine `client` serves down and waits for its process.
async fn close(client: &Arc<EngineClient>) {
    let closed = Arc::clone(client);
    let exit = tokio::task::spawn_blocking(move || {
        closed.shutdown();
        closed.await_exit(CLOSED_EXIT)
    })
    .await;
    match exit {
        Ok(Ok(_)) => {}
        Ok(Err(error)) => warn!("an engine process: {error:#}"),
        Err(error) => warn!("an engine's shutdown: {error}"),
    }
}

/// Ends the report of the change in progress of a replica when dropped.
struct Reported<'a>(&'a crate::leader::Replica);

impl Drop for Reported<'_> {
    fn drop(&mut self) {
        self.0.progress_done();
    }
}

/// Rolls the world on trial back to the previous world (see the module);
/// the caller holds the replacement lock.
pub(crate) async fn rollback(
    replica: &Arc<crate::leader::Replica>,
    reason: String,
) -> Result<serde_json::Value, UpdateError> {
    if let Some(registry) = replica.registry.upgrade() {
        locked(&registry.pending).clear();
    }
    let trial = locked(&replica.worlds)
        .trial
        .take()
        .ok_or_else(|| UpdateError::Refused("no world is on trial".to_owned()))?;
    warn!(world = trial.world, previous_world = trial.previous_world, %reason, "rolling back");
    let hold = Arc::clone(&replica.hold);
    // The requests returning to the previous world recompute what the world
    // on trial computed for them.
    hold.keep(None);
    let paused = Instant::now();
    // The status reports the rollback, however it ends, until it ends.
    locked(&replica.worlds).progress = Some(Progress {
        phase: ROLLBACK.to_owned(),
        started: now(),
        world: Some(trial.previous_world),
        note: None,
        moved_from: None,
        paused: Some(paused),
        steps: None,
    });
    let _reported = Reported(replica);
    // The hold names no engine while it holds: the world on trial is taken
    // first.
    let failing = hold.client();
    // A profiling session of the world on trial ends with it.
    crate::profiling::end_for_world(replica, failing.as_deref(), "rollback").await;
    hold.hold();
    // The world on trial preempts its requests, which wait for the previous
    // world: its live schedulers park, and a scheduler that failed handed
    // its requests back as it stopped. An engine whose process ended handed
    // them back as it ended (`EngineClient`).
    if let Some(failing) = &failing
        && failing.state() != EngineState::Closed
    {
        match tokio::time::timeout(CLOSED_EXIT, failing.park()).await {
            Ok(Ok(_)) => {}
            Ok(Err(error)) => info!(%error, "the world on trial parked in part"),
            Err(_) => warn!(
                "the world on trial did not park within {CLOSED_EXIT:?}; closing it with the \
                 requests it holds"
            ),
        }
    }
    // The pause's parts, as a promotion reports its park: stopping the world
    // on trial, resuming (or starting again) the previous world, and
    // restoring what the trial changed of it.
    let stopped = paused.elapsed();
    let parked = matches!(trial.previous, Previous::Parked(_));
    let previous = match trial.previous {
        Previous::Parked(previous) => {
            // It resumes at once: it needs nothing the world on trial holds,
            // whose storage it retires below, once that world's processes
            // are gone and no claim waits for them.
            previous.resume(Vec::new()).await.map_err(|error| {
                hold.close(format!("the previous world did not resume: {error}"));
                anyhow::anyhow!("the previous world did not resume: {error}")
            })?;
            replay_graphs(&previous, trial.eager).await;
            previous
        }
        Previous::Weights => {
            // The previous world starts again in the storage the world on
            // trial holds, which its close and retirement free.
            if let Some(failing) = &failing {
                close(failing).await;
            }
            let control = Arc::clone(replica.control.get().ok_or(UpdateError::NoEngine)?);
            let sink = stats_sink(
                replica.model_name.clone(),
                Arc::clone(&replica.observer),
                replica.ordinal,
            );
            // The world on trial, closed, is retired before the previous
            // world's workers start: they need its storage (weights of its
            // own included) within the deployment's quota.
            let spec = wire::EngineSpec {
                retire: vec![trial.world],
                ..trial.previous_spec.clone()
            };
            let started = tokio::task::spawn_blocking(move || control.start(spec, sink))
                .await
                .context("the previous world's start did not complete")
                .and_then(|started| started.context("the previous world did not start"));
            let started = match started {
                Ok(started) => Arc::new(started),
                Err(error) => {
                    hold.close(format!("the previous world did not start again: {error:#}"));
                    return Err(error.into());
                }
            };
            started
                .resume(vec![trial.world])
                .await
                .map_err(|error| anyhow::anyhow!("retiring world {}: {error}", trial.world))?;
            started
        }
    };
    let resumed = paused.elapsed();
    warming::restore(&previous, &trial.cache_backup)
        .await
        .map_err(anyhow::Error::msg)?;
    hold.serve(Arc::clone(&previous));
    // The pause ends as the previous world admits requests again, as a
    // promotion's ends as its successor does; the segments the previous
    // world gave its successor are created again while it serves.
    let pause = paused.elapsed();
    // A parked previous world serves while the world on trial closes: it
    // uses no storage the world on trial could write. Once that world's
    // processes are gone, it retires their storage and creates again the
    // cache segments it gave them (`Restore`).
    if parked {
        if let Some(failing) = &failing {
            close(failing).await;
        }
        if let Err(error) = previous.resume(vec![trial.world]).await {
            warn!(
                world = trial.world,
                "the world on trial was not retired: {error}"
            );
        }
    }
    crate::leader::reload_adapters(replica, &previous).await;
    if let Err(error) = previous.resize_cache(wire::CacheChange::Restore).await {
        warn!("the previous world's KV cache was not restored: {error}");
    }
    let mut previous_spec = trial.previous_spec;
    previous_spec.stage_cache = false;
    previous_spec.cache_keys.clear();
    *locked(&replica.served) = Some(previous_spec);
    *locked(&replica.config) = trial.previous_config;
    let report = serde_json::json!({
        "rolled_back": trial.world,
        "world": trial.previous_world,
        "reason": reason,
        "pause_seconds": pause.as_secs_f64(),
        "stop_seconds": stopped.as_secs_f64(),
        "resume_seconds": (resumed - stopped).as_secs_f64(),
        "restore_seconds": (pause - resumed).as_secs_f64(),
        "trial_seconds": trial.promoted.elapsed().as_secs_f64(),
    });
    {
        let mut worlds = locked(&replica.worlds);
        worlds.executing = trial.previous_world;
        worlds.unavailable.insert(
            trial.digest.clone(),
            Unavailable {
                digest: trial.digest,
                world: trial.world,
                reason,
                at: now(),
            },
        );
        worlds.last = Some(report.clone());
    }
    watch(Arc::clone(replica), Arc::downgrade(&previous));
    info!(
        world = trial.previous_world,
        "the previous world serves again"
    );
    Ok(report)
}

/// Watches the engine `client` serves for its loss (`watch_engine`), on a
/// task of its own whose future the rollback it may run does not contain.
fn watch(replica: Arc<crate::leader::Replica>, client: std::sync::Weak<EngineClient>) {
    let watching: std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> =
        Box::pin(crate::leader::watch_engine(replica, client));
    tokio::spawn(watching);
}

/// Watches world `world`'s trial and rolls it back on a fault or past a
/// threshold; ends with the trial.
async fn watch_trial(replica: Arc<crate::leader::Replica>, world: u64) {
    loop {
        tokio::time::sleep(TRIAL_POLL).await;
        let reason = {
            let worlds = locked(&replica.worlds);
            let Some(trial) = worlds.trial.as_ref().filter(|trial| trial.world == world) else {
                return;
            };
            trial_fault(&replica, trial)
        };
        let Some(reason) = reason else {
            continue;
        };
        let _replacing = replica.replacing.lock().await;
        if !locked(&replica.worlds)
            .trial
            .as_ref()
            .is_some_and(|trial| trial.world == world)
        {
            return;
        }
        if let Err(error) = rollback(&replica, reason).await {
            tracing::error!("the rollback failed: {error:#}");
        }
        return;
    }
}

/// Why `trial` rolls back now, if it does.
fn trial_fault(replica: &crate::leader::Replica, trial: &Trial) -> Option<String> {
    let client = replica.hold.client();
    match client.as_ref().map(|client| client.state()) {
        Some(EngineState::Failed(reason)) => {
            return Some(format!("world {} failed: {reason}", trial.world));
        }
        Some(EngineState::Closed) | None => {
            return Some(format!("world {}'s engine process ended", trial.world));
        }
        _ => {}
    }
    let faults = replica
        .observer
        .numerical_faults(trial.world, trial.promoted);
    if faults > 0 {
        return Some(format!(
            "world {} produced non-finite logits in {faults} calls",
            trial.world
        ));
    }
    let thresholds = &trial.thresholds;
    if let Some(rate) = thresholds.failure_rate {
        let (ended, failed) = crate::leader::outcomes();
        let ended = ended.saturating_sub(trial.outcomes.0);
        let failed = failed.saturating_sub(trial.outcomes.1);
        if ended > 0 && failed as f64 / ended as f64 > rate {
            return Some(format!(
                "{failed} of the {ended} requests that ended on world {} failed, past the \
                 failure rate {rate}",
                trial.world
            ));
        }
    }
    if let (Some(window), Some(baseline)) = (thresholds.window, trial.baseline)
        && trial.promoted.elapsed() >= window
        && let Some(rates) = replica.observer.rates(trial.world, trial.promoted, window)
    {
        if let Some(regression) = thresholds.step_time_regression
            && baseline.step_us > 0.0
            && rates.step_us > baseline.step_us * (1.0 + regression)
        {
            return Some(format!(
                "world {}'s steps take {:.0} us against {:.0} us before, past the regression \
                 {regression}",
                trial.world, rates.step_us, baseline.step_us
            ));
        }
        if let Some(regression) = thresholds.throughput_regression
            && baseline.tokens_per_second > 0.0
            && rates.tokens_per_second < baseline.tokens_per_second * (1.0 - regression)
        {
            return Some(format!(
                "world {} computes {:.0} tokens/s against {:.0} before, past the regression \
                 {regression}",
                trial.world, rates.tokens_per_second, baseline.tokens_per_second
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn host_admission_includes_new_target_nodes_once_per_rank() {
        use proto::process::{self, downstream, reply, upstream};
        use storage::manifest::weights::{CacheReservation, RankBudget};

        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("process.sock");
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        // This protocol peer is the target node's existing storage ledger.
        let node = std::thread::spawn(move || {
            let mut connection =
                proto::Connection::from_unix(listener.accept().unwrap().0).unwrap();
            let _: process::Hello = connection.recv(Duration::from_secs(5)).unwrap().unwrap();
            connection.send(&process::Welcome::default()).unwrap();
            let message: process::Upstream =
                connection.recv(Duration::from_secs(5)).unwrap().unwrap();
            let Some(upstream::Kind::Request(request)) = message.kind else {
                panic!("expected a process request");
            };
            connection.send(&process::Downstream {
                kind: Some(downstream::Kind::Reply(process::Reply {
                    id: request.id,
                    error: String::new(),
                    kind: Some(reply::Kind::Ledger(process::LedgerReport {
                        ledgers: serde_json::json!({
                            "ledgers": [],
                            "host_storage": {"available": 800, "reserve": 100, "unallocated": 100},
                        }).to_string(),
                    })),
                })),
            }).unwrap();
        });
        let control = Arc::new(
            crate::engine::control::EngineControl::new(
                &directory.path().join("requests.sock"),
                format!("unix:{}", socket.display()),
                "source".into(),
                "model".into(),
            )
            .unwrap(),
        );
        let budgets = [(1, 100), (2, 300), (3, 400)]
            .into_iter()
            .map(|(gpu, bytes)| {
                (
                    [gpu; 16],
                    RankBudget {
                        in_flight: Some(CacheReservation {
                            bytes,
                            mapped: bytes * 2,
                        }),
                        ..Default::default()
                    },
                )
            })
            .collect();
        let hosts = BTreeMap::from([
            ([1; 16], "source".into()),
            ([2; 16], "target".into()),
            ([3; 16], "target".into()),
        ]);
        let plans = [serde_json::json!({
            "host": "source", "gpu": "GPU-source", "feasible": true,
            "in_flight_kv": 200,
            "host_staging": 900,
            "host_storage": {"available": 1000, "reserve": 100, "unallocated": 0},
        })
        .to_string()];
        let report = storage_report(
            &plans,
            PreviousWorld::Parked,
            Successor {
                host_cache: true,
                unmapped_cache: true,
                ..Default::default()
            },
            &budgets,
            &hosts,
            &control,
        )
        .await
        .unwrap();
        node.join().unwrap();
        let hosts = report["hosts"].as_array().unwrap();
        let source = hosts.iter().find(|host| host["host"] == "source").unwrap();
        let target = hosts.iter().find(|host| host["host"] == "target").unwrap();
        assert_eq!(source["required"], 300);
        assert_eq!(source["feasible"], true);
        assert_eq!(target["required"], 700);
        assert_eq!(target["admits"], 600);
        assert_eq!(target["feasible"], false);
        assert_eq!(report["feasible"], false);
    }

    fn stats(batch_us: u64, batches: u64, tokens: u64, faults: u64) -> SchedulerStats {
        SchedulerStats {
            batch_roundtrip_us: batch_us,
            batch_count: batches,
            numerical_faults: faults,
            worker_forward_stats: Some(proto::report::ForwardStats {
                mode_tokens: BTreeMap::from([("decode".to_owned(), tokens)]),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// A world's rates average its own batches over the window, apart from
    /// another world's.
    #[test]
    fn rates_average_one_worlds_batches() {
        let observer = Observer::default();
        let since = Instant::now();
        observer.record(1, 0, &stats(1_000, 10, 500, 0));
        observer.record(1, 0, &stats(3_000, 10, 500, 0));
        observer.record(2, 0, &stats(90_000, 1, 1, 2));
        let rates = observer
            .rates(1, since, Duration::from_secs(60))
            .expect("world 1 ran batches");
        assert!((rates.step_us - 200.0).abs() < 1e-9, "{rates:?}");
        assert!(rates.tokens_per_second > 0.0);
        assert_eq!(observer.numerical_faults(1, since), 0);
        assert_eq!(observer.numerical_faults(2, since), 2);
        assert!(observer.rates(3, since, Duration::from_secs(60)).is_none());
    }

    /// A change of the scheduling alone, within the batch the workers were
    /// started for, is in place for a world that a change started, whose
    /// spec carries its identity and its cache's layout.
    #[test]
    fn only_the_fence_probes_switch_changes_profiling_in_place() {
        let spec = |profiling: wire::Profiling| wire::EngineSpec {
            profiling: Some(profiling),
            ..wire::EngineSpec::default()
        };
        let served = spec(wire::Profiling {
            fence_probes: true,
            fence_components: vec!["*.Transformer".to_owned()],
            fence_probes_off: Vec::new(),
            graph_pc_sampling: false,
        });
        let switched = spec(wire::Profiling {
            fence_probes: false,
            ..served.profiling.clone().unwrap()
        });
        assert!(switches_probes_only(&served, &switched));
        // A component's probes switch off in place too.
        let thinned = spec(wire::Profiling {
            fence_probes_off: vec!["*.Attention".to_owned()],
            ..served.profiling.clone().unwrap()
        });
        assert!(switches_probes_only(&served, &thinned));
        for world in [
            wire::Profiling {
                graph_pc_sampling: true,
                ..served.profiling.clone().unwrap()
            },
            wire::Profiling {
                fence_components: vec!["*.TransformerLayer".to_owned()],
                ..served.profiling.clone().unwrap()
            },
        ] {
            assert!(!switches_probes_only(&served, &spec(world)));
        }
    }

    #[test]
    fn a_scheduling_change_of_a_successor_world_is_in_place() {
        let scheduling = |running: u32| wire::Scheduling {
            max_running_requests: running,
            max_batch: 8,
            ..Default::default()
        };
        let served = wire::EngineSpec {
            model: "a".into(),
            world: 3,
            segments: vec![wire::CacheSegments {
                worker_group: "model".into(),
                ..Default::default()
            }],
            scheduling: Some(scheduling(8)),
            ..Default::default()
        };
        let target = wire::EngineSpec {
            world: 0,
            segments: Vec::new(),
            scheduling: Some(scheduling(4)),
            ..served.clone()
        };
        let capacity = wire::BatchCapacity {
            max_batch: 8,
            max_num_batched_tokens: 8192,
            max_running_requests: 8,
        };
        assert!(crate::leader::configurable(
            &comparable(&served),
            &comparable(&target),
            Some(&capacity)
        ));
    }

    /// The resolved deployment carries the ranks it read even when the
    /// file is changed in place before the engine starts.
    #[test]
    fn a_resolved_deployment_keeps_its_ranks_when_the_file_changes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("deployment.toml");
        let spec = |ranks: &str| {
            std::fs::write(
                &path,
                format!(
                    "[[worker_groups]]\nid = \"model\"\nranks = {ranks}\nqueue_depth = 2\n\n\
                     [worker_groups.components.model]\ntensor_parallel_size = 2\n"
                ),
            )
            .unwrap();
            let config = crate::options::ServeConfig {
                worker_groups: Some(path.clone()),
                ..Default::default()
            };
            let mut resolved = config.engine_spec().unwrap();
            resolved.model_facts = Some(wire::ModelFacts::default());
            resolved
        };
        let served = spec(r#"["rank-0/cuda:0", "rank-0/cuda:1"]"#);
        let moved = spec(r#"["rank-1/cuda:0", "rank-1/cuda:1"]"#);
        std::fs::remove_file(path).unwrap();
        for (resolved, host) in [(served, "rank-0"), (moved, "rank-1")] {
            let engine = engine::EngineConfig::from_spec(&resolved).unwrap();
            let ranks = &engine.worker_groups[0].ranks;
            assert_eq!(ranks.len(), 2);
            assert!(ranks.iter().all(|rank| rank.host == host));
            assert_eq!(ranks[0].device, "cuda:0");
            assert_eq!(ranks[1].device, "cuda:1");
        }
    }

    /// A node's storage report as its daemon answers it, with what the
    /// deployment's transfers copied there by priority.
    fn node(host: &str, totals: &[(storage::protocol::Priority, u64, u64)]) -> String {
        let transfers: Vec<_> = totals
            .iter()
            .map(
                |&(priority, bytes, copied)| storage::protocol::TransferTotal {
                    priority,
                    bytes,
                    copied,
                },
            )
            .collect();
        serde_json::json!({ "host": host, "ledgers": [], "transfers": transfers }).to_string()
    }

    /// The status names each slot's state as a change goes: one world
    /// executes beside an empty slot; a successor prepares beside it while
    /// the change reports its step, what the deployment's transfers moved
    /// since it began and the serving world's steps against their
    /// baseline; requests wait at the promotion; the previous world is
    /// parked once its successor serves and kept in host storage during the
    /// trial; a rollback stops the world on trial; a world whose engine is
    /// gone is unavailable.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_status_reports_each_slots_state_and_the_change_in_progress() {
        use storage::protocol::Priority::{Background, Warming};

        let mut executor = engine::SimExecutor::new(engine::SimEngine::new());
        let nodes = executor.report_ledgers();
        let core = engine::SchedulerHandle::with_executor(
            engine::EngineConfig::sim(crate::test_support::SERVED_MODEL),
            Box::new(executor),
        )
        .expect("start a sim scheduler");
        let leader =
            crate::test_support::sim_engine_over(crate::model::ModelParameters::Qwen3, vec![core]);
        let replica = Arc::clone(&leader.replicas()[0]);
        let client = replica.hold().client().expect("the replica serves");
        let world = client.world();
        *locked(&nodes) = vec![node("rank-0", &[(Background, 100, 40)])];

        let status = leader.config_status().await;
        assert_eq!(
            status["slots"],
            serde_json::json!([
                { "world": world, "state": "executing" },
                { "world": null, "state": "empty" },
            ])
        );
        assert!(status["update"].is_null(), "{status}");
        assert_eq!(status["gpus"], serde_json::json!([]));

        // The GPUs the deployment books storage on, by name.
        let ledger = storage::protocol::GpuLedger {
            gpu: [7; 16],
            deployment: crate::test_support::SERVED_MODEL.to_owned(),
            total: 1000,
            quota: Some(900),
            entries: vec![storage::protocol::LedgerEntry {
                item: storage::protocol::Item::Weights,
                scope: None,
                worlds: vec![world],
                segment: None,
                bytes: 300,
            }],
            host_link_bytes_per_second: None,
            transfer_context: 0,
        };
        *locked(&nodes) = vec![
            serde_json::json!({
                "host": "rank-0",
                "ledgers": [ledger],
                "transfers": [{ "priority": "background", "bytes": 100, "copied": 40 }],
            })
            .to_string(),
        ];
        let status = leader.config_status().await;
        assert_eq!(
            status["gpus"],
            serde_json::json!([{
                "host": "rank-0",
                "gpu": storage::protocol::gpu_name(&[7; 16]),
                "total": 1000,
                "quota": 900,
                "used": 300,
            }])
        );

        // A successor prepares: the change began with the nodes' totals,
        // and a node joined since.
        let from = transfer_totals(&client).await.expect("the totals are read");
        replica.progress(Step::Prepare, Some(world + 1));
        replica.progress_edit(|progress| {
            progress.moved_from = Some(from);
            progress.steps = Some(Arc::new(std::sync::Mutex::new(
                crate::interference::StepTime {
                    ratio: Some(1.25),
                    slowdown: 0.1,
                    halved: true,
                    throttled: crate::interference::Throttled {
                        halvings: 2,
                        restorations: 1,
                    },
                },
            )));
        });
        *locked(&nodes) = vec![
            node("rank-0", &[(Warming, 64, 64), (Background, 1100, 640)]),
            node("rank-1", &[(Background, 10, 10)]),
        ];
        let status = leader.config_status().await;
        assert_eq!(
            status["slots"][1],
            serde_json::json!({ "world": world + 1, "state": "preparing" })
        );
        let update = &status["update"];
        assert_eq!(update["phase"], "prepare", "{update}");
        assert_eq!(update["world"], world + 1);
        assert_eq!(update["moved"]["bytes"], 600 + 64 + 10, "{update}");
        assert_eq!(update["moved"]["submitted_bytes"], 1000 + 64 + 10);
        assert_eq!(update["moved"]["by_priority"]["warming"]["bytes"], 64);
        assert_eq!(update["moved"]["by_priority"]["background"]["bytes"], 610);
        assert_eq!(update["step_time"]["ratio"], 1.25);
        assert_eq!(update["step_time"]["halved"], true);
        assert_eq!(update["step_time"]["halvings"], 2);
        assert!(
            update["seconds"]
                .as_f64()
                .is_some_and(|seconds| seconds >= 0.0)
        );
        assert!(update["pause_seconds"].is_null());

        // At the promotion requests wait, the serving world parks.
        replica.hold().hold();
        replica.progress(Step::Park, None);
        replica.progress_edit(|progress| progress.paused = Some(Instant::now()));
        let status = leader.config_status().await;
        assert_eq!(
            status["slots"],
            serde_json::json!([
                { "world": world, "state": "parked" },
                { "world": world + 1, "state": "preparing" },
            ])
        );
        assert!(status["update"]["pause_seconds"].as_f64().is_some());

        // Once the successor serves, the world that served stays parked
        // until the change keeps it.
        let previous = world + 100;
        replica.hold().serve(Arc::clone(&client));
        locked(&replica.worlds).executing = previous;
        replica.progress(Step::KeepPrevious, Some(world));
        let status = leader.config_status().await;
        assert_eq!(
            status["slots"],
            serde_json::json!([
                { "world": world, "state": "executing" },
                { "world": previous, "state": "parked" },
            ])
        );

        // The trial keeps the previous world's weights in host storage.
        replica.progress_done();
        {
            let mut worlds = locked(&replica.worlds);
            worlds.executing = world;
            worlds.trial = Some(Trial {
                world,
                digest: String::new(),
                previous_world: previous,
                previous_spec: wire::EngineSpec::default(),
                previous_config: ServeConfig::default(),
                previous: Previous::Weights,
                cache_backup: Vec::new(),
                mode: PreviousWorld::Host,
                host_refused: None,
                promoted: Instant::now(),
                promoted_at: now(),
                baseline: None,
                thresholds: RollbackOptions::default(),
                outcomes: crate::leader::outcomes(),
                pause: Duration::from_millis(15),
                eager: false,
            });
        }
        let status = leader.config_status().await;
        assert_eq!(
            status["slots"],
            serde_json::json!([
                { "world": world, "state": "executing" },
                { "world": previous, "state": "host" },
            ])
        );
        assert!(status["update"].is_null());
        let trial = &status["trial"];
        assert_eq!(trial["previous_world_kept"], "host");
        assert_eq!(trial["pause_seconds"], 0.015);
        assert!(trial["requests_ended"].is_u64() && trial["requests_failed"].is_u64());

        // A rollback stops the world on trial while the previous world
        // starts again.
        {
            let mut worlds = locked(&replica.worlds);
            worlds.trial = None;
            worlds.progress = Some(Progress {
                phase: ROLLBACK.to_owned(),
                started: now(),
                world: Some(previous),
                note: None,
                moved_from: None,
                paused: Some(Instant::now()),
                steps: None,
            });
        }
        replica.hold().hold();
        let status = leader.config_status().await;
        assert_eq!(
            status["slots"],
            serde_json::json!([
                { "world": world, "state": "unavailable", "reason": "a rollback stops it" },
                { "world": previous, "state": "preparing" },
            ])
        );

        // A world whose engine no longer serves is unavailable.
        replica.progress_done();
        replica.hold().close("the engine process ended");
        let status = leader.config_status().await;
        assert_eq!(
            status["slots"],
            serde_json::json!([
                { "world": world, "state": "unavailable", "reason": "the engine process ended" },
                { "world": null, "state": "empty" },
            ])
        );
        client.shutdown();
    }

    /// Two serving GPUs of one host, as a world change's storage plan
    /// reports them.
    fn two_gpus() -> serde_json::Value {
        serde_json::json!({ "gpus": [
            {
                "host": "rank-0", "gpu": "GPU-a", "in_flight_kv": 1000,
                "in_flight_target": 800, "partial_tail_kv": 32, "staging_stall_ms": 5,
            },
            {
                "host": "rank-0", "gpu": "GPU-b", "in_flight_kv": 500,
                "in_flight_target": 400, "partial_tail_kv": 16, "staging_stall_ms": 9,
            },
        ] })
    }

    fn step<'a>(steps: &'a [serde_json::Value], name: &str) -> &'a serde_json::Value {
        steps
            .iter()
            .find(|step| step["step"] == name)
            .unwrap_or_else(|| panic!("no step {name} in {steps:?}"))
    }

    /// A plan names what each step moves over each link: a change that
    /// maps the cache copies each request's partial last page on its GPU at
    /// the promotion, all it fills in then.
    #[test]
    fn a_mapped_change_fills_in_partial_pages_at_its_promotion() {
        let change = WorldChange {
            successor: Successor::default(),
            redistributes: false,
            shrinks: true,
            reencodes_cache: false,
            in_flight: InFlight::Keep,
            previous_world: PreviousWorld::Parked,
            serving_drafts: false,
            successor_drafts: false,
        };
        let storage = two_gpus();
        let copies = PlannedCopies::new(Some(&storage), &BTreeMap::new(), &BTreeMap::new(), &[]);
        let steps = copies.steps(&change);
        assert!(steps.iter().all(|step| step["step"] != "warm_cache"));
        let gpu = |name: &str| serde_json::json!({ "gpu": name, "host": "rank-0" });
        assert_eq!(
            step(&steps, "promote")["links"],
            serde_json::json!([
                { "source": gpu("GPU-a"), "target": gpu("GPU-a"), "bytes": 32 },
                { "source": gpu("GPU-b"), "target": gpu("GPU-b"), "bytes": 16 },
            ])
        );
        assert_eq!(step(&steps, "keep_previous")["previous_world"], "parked");
        assert_eq!(
            copies.backfill(&change, 3, Some(64)),
            serde_json::json!({
                "bytes": 48, "stall_ms": null, "requests": 3, "tail_tokens_per_request": 0,
            })
        );
    }

    /// A successor of another cache capacity that stages its warm copies in
    /// host storage copies the requests' pages there twice (for the
    /// rollback and in its own layout) and loads its layout over the host
    /// links at the promotion, whose wait the measured bandwidth bounds.
    #[test]
    fn a_staged_change_loads_its_copies_over_the_host_links() {
        let change = WorldChange {
            successor: Successor {
                unmapped_cache: true,
                host_cache: true,
                ..Successor::default()
            },
            redistributes: false,
            shrinks: true,
            reencodes_cache: false,
            in_flight: InFlight::Keep,
            previous_world: PreviousWorld::Weights,
            serving_drafts: false,
            successor_drafts: false,
        };
        let storage = two_gpus();
        let copies = PlannedCopies::new(Some(&storage), &BTreeMap::new(), &BTreeMap::new(), &[]);
        let steps = copies.steps(&change);
        let names: Vec<_> = steps.iter().map(|step| step["step"].clone()).collect();
        assert_eq!(
            names,
            [
                "shrink",
                "prepare",
                "warm_cache",
                "park",
                "evict",
                "materialize",
                "promote",
                "keep_previous",
                "trial"
            ]
        );
        let gpu = |name: &str| serde_json::json!({ "gpu": name, "host": "rank-0" });
        let host = serde_json::json!({ "host_storage": "rank-0" });
        let warm = step(&steps, "warm_cache");
        assert_eq!(warm["staged"], true);
        assert_eq!(
            warm["links"],
            serde_json::json!([
                { "source": gpu("GPU-a"), "target": host, "bytes": 1000, "copy": "rollback" },
                { "source": gpu("GPU-b"), "target": host, "bytes": 500, "copy": "rollback" },
                { "source": gpu("GPU-a"), "target": host, "bytes": 800 },
                { "source": gpu("GPU-b"), "target": host, "bytes": 400 },
            ])
        );
        let materialize = step(&steps, "materialize");
        assert_eq!(
            materialize["links"],
            serde_json::json!([
                { "source": host, "target": gpu("GPU-a"), "bytes": 800 },
                { "source": host, "target": gpu("GPU-b"), "bytes": 400 },
            ])
        );
        assert_eq!(materialize["stall_ms"], 9);
        assert_eq!(
            copies.backfill(&change, 2, Some(64)),
            serde_json::json!({
                "bytes": 1200, "stall_ms": 9, "requests": 2, "tail_tokens_per_request": 64,
            })
        );
    }

    /// A successor of another parallel configuration receives its weights
    /// over the manifests' links and each target GPU's reservation for the
    /// requests from the serving GPUs, whose division the warming's cache
    /// manifest decides.
    #[test]
    fn a_redistribution_names_its_weight_links_and_its_warm_copies_sources() {
        use storage::manifest::weights::{CacheReservation, RankBudget};
        use storage::protocol::LinkEnd;

        let change = WorldChange {
            successor: Successor {
                unmapped_cache: true,
                ..Successor::default()
            },
            redistributes: true,
            shrinks: true,
            reencodes_cache: false,
            in_flight: InFlight::Keep,
            previous_world: PreviousWorld::Parked,
            serving_drafts: false,
            successor_drafts: false,
        };
        let target = [3; 16];
        let budgets = BTreeMap::from([(
            target,
            RankBudget {
                in_flight: Some(CacheReservation {
                    bytes: 700,
                    mapped: 900,
                }),
                ..Default::default()
            },
        )]);
        let hosts = BTreeMap::from([(target, "rank-1".to_owned())]);
        let load = |bytes: u64| {
            serde_json::json!({
                "source": LinkEnd::Device { gpu: [1; 16] },
                "target": LinkEnd::Device { gpu: target },
                "bytes": bytes,
                "seconds": 0.25,
            })
        };
        let redistribution = serde_json::json!({ "manifests": [
            { "links": [load(4096)] },
            { "links": [load(1024)] },
        ] });
        let storage = two_gpus();
        let copies = PlannedCopies::new(Some(&storage), &budgets, &hosts, &[&redistribution]);
        let steps = copies.steps(&change);
        let target_name = storage::protocol::gpu_name(&target);
        assert_eq!(
            step(&steps, "prepare")["links"],
            serde_json::json!([{
                "source": { "gpu": storage::protocol::gpu_name(&[1; 16]), "host": null },
                "target": { "gpu": target_name, "host": "rank-1" },
                "bytes": 5120,
                "seconds": 0.5,
            }])
        );
        assert_eq!(
            step(&steps, "warm_cache")["links"],
            serde_json::json!([{
                "sources": [
                    { "gpu": "GPU-a", "host": "rank-0" },
                    { "gpu": "GPU-b", "host": "rank-0" },
                ],
                "target": { "gpu": target_name, "host": "rank-1" },
                "bytes": 700,
            }])
        );
        // The warmed requests compute at most a page at the promotion.
        assert_eq!(
            copies.backfill(&change, 2, Some(64))["tail_tokens_per_request"],
            64
        );
    }
}
