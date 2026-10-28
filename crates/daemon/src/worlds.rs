//! The deployment's worlds and how it moves between them
//! (`PUT /admin/config`, `specs/online-operations.md`).
//!
//! A world is the processes that execute one configuration of the
//! deployment: an engine and its workers, their communicators, graphs and
//! the memory they created. A deployment has two slots for worlds, and at
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
//!    tokens as migratable and executes nothing more, keeping its memory and
//!    its prefix cache index), and the successor imports that index into
//!    the units it maps and serves: every migrated request is submitted
//!    again with the tokens it generated and reuses the pages of every
//!    complete page of them, computing at most its last page. On a
//!    successor of another parallel configuration each computes its tokens
//!    again.
//! 3. **Trial.** The successor executes; the previous world is kept as
//!    `update_config.previous_world` states: parked, its weights alone, or
//!    its weights in host memory. The trial ends only with the operator's
//!    commit, and rolls back by itself on a fault of the successor (a fatal
//!    worker error, its engine process's exit, a stalled job, non-finite
//!    logits) or past a threshold of `update_config.rollback`.
//! 4. **Commit** or **rollback.** A commit releases the previous world: its
//!    processes exit, the successor takes the segments it mapped read-only
//!    and retires the previous world's memory, and grows its cache into the
//!    memory freed. A rollback parks and closes the successor, resumes the
//!    previous world (or starts it again on its kept weights), retires the
//!    successor's memory, and marks the configuration unavailable: applying
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

pub(crate) mod group;
mod redistribution;
mod warming;
use redistribution::Redistribution;

/// An irrevocable operator decision, retained through partial publication,
/// worker recovery and daemon handoff until every successor commits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Commit {
    worlds: Vec<u64>,
    retire: Vec<u64>,
    discard_pages: bool,
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
    /// The previous world's retention mode, as the status reports it: the
    /// mode the update asked for, or `weights` when its weights could not
    /// move to host memory.
    pub(crate) mode: PreviousWorld,
    /// Why the previous world's weights stayed on the GPUs when the update
    /// asked to keep them in host memory: each node's refusal.
    pub(crate) host_refused: Option<String>,
    /// Whether the world on trial keys its pages otherwise than the previous
    /// world (another prefix seed): committed, it hits none of the pages the
    /// previous world published, which the host KV caches then discard.
    pub(crate) pages_change: bool,
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Progress {
    pub phase: String,
    pub started: f64,
    /// The world being prepared, once it starts.
    pub world: Option<u64>,
    /// What the change does otherwise than its options ask: a preparation
    /// whose transfers start without a baseline of some scheduler's steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
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

/// How often a trial checks its thresholds.
const TRIAL_POLL: Duration = Duration::from_millis(500);

/// How long the closed engine of a world takes to exit.
const CLOSED_EXIT: Duration = Duration::from_secs(60);

/// How long the status waits for the memory ledgers, which each node's
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
    ranks(served) != ranks(target)
        || mesh(served) != mesh(target)
        || pages(served) != pages(target)
        || served.experts != target.experts
}

/// What a world change keeps of the serving world's derived state, by the
/// parts of the engine spec that decide each one's keys: the weights and
/// the KV cache the successor maps. Names the parts that differ. A change
/// of the parallel configuration ([`parallel_change`]) is not among them:
/// it redistributes both.
fn differing(served: &wire::EngineSpec, target: &wire::EngineSpec) -> Vec<&'static str> {
    let parallel = parallel_change(served, target);
    let mut parts = Vec::new();
    let mut differs = |name: &'static str, same: bool| {
        if !same {
            parts.push(name);
        }
    };
    differs("the deployment", served.deployment == target.deployment);
    differs("the host", served.host == target.host);
    // The queue depth and memory fraction of the same ranks.
    let sizing = |spec: &wire::EngineSpec| {
        spec.placement.as_ref().map(|placement| {
            (
                placement.queue_depth,
                placement.mem_fraction_static.to_bits(),
            )
        })
    };
    differs(
        "the placement",
        parallel || sizing(served) == sizing(target),
    );
    differs("the product transport", served.transfer == target.transfer);
    // The KV cache but its page size, which a parallel change covers, and
    // its encoding, which the successor computes again in a cache of its own
    // (`kv_encoding_changes`).
    differs("the KV cache", same_kv_cache_layout(served, target));
    differs("the video options", served.video == target.video);
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

/// Whether two specs' KV caches are the same but for their dtype and their
/// page size, which a successor that maps none of the cache lays out anew.
fn same_kv_cache_layout(served: &wire::EngineSpec, target: &wire::EngineSpec) -> bool {
    let layout = |spec: &wire::EngineSpec| {
        spec.kv_cache.clone().map(|kv| wire::KvCache {
            dtype: String::new(),
            block_size: None,
            ..kv
        })
    };
    layout(served) == layout(target)
}

/// Whether a world succeeding the one `running` executes, which adds
/// `successor` beside what it shares with it, fits beside it on every
/// device, from each GPU's memory ledger: the report `PUT
/// /admin/config` plans with, and the refusal when it does not fit. The
/// serving engine computes, for each worker that holds the KV cache, what
/// the serving world, the successor and the Transfer Server hold on its GPU
/// and the KV cache limit that leaves the serving world, and fails, with
/// every item of every GPU and host, when a limit cannot hold the segments
/// the serving world keeps in any case.
async fn memory_plan(
    running: &EngineClient,
    successor: Successor,
    budgets: &BTreeMap<storage::manifest::Gpu, storage::manifest::weights::RankBudget>,
    previous_world: PreviousWorld,
) -> (serde_json::Value, Option<String>) {
    match running
        .make_room(wire::CacheChange::Plan, successor, None, budgets)
        .await
    {
        Ok(results) => (memory_report(&results, previous_world, successor), None),
        Err(refusal) => (
            serde_json::json!({ "feasible": false, "refusal": refusal }),
            Some(refusal),
        ),
    }
}

/// The memory report of the engine's per-device plans (`ledger::Preparing`
/// of the engine, as JSON) for a successor that adds `successor` and a
/// previous world kept as `previous_world`: each GPU's, with what each
/// step of `specs/online-operations.md` §6.5 frees there ([`steps`]), and
/// each host's admission of new host memory. A successor sharing the
/// serving world's page pools needs none; a previous world kept in host
/// memory moves there the weights a successor that loads its own does not
/// map (all of the serving world's on each GPU, at most), which each host's
/// admission must hold, or they stay on the GPUs.
fn memory_report(
    plans: &[String],
    previous_world: PreviousWorld,
    successor: Successor,
) -> serde_json::Value {
    let weights_to_host = previous_world == PreviousWorld::Host && successor.loads_weights;
    let mut gpus: Vec<serde_json::Value> = plans
        .iter()
        .filter_map(|plan| serde_json::from_str(plan).ok())
        .collect();
    for gpu in &mut gpus {
        gpu["steps"] = steps(gpu, previous_world, successor);
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
    let mut hosts: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for gpu in &gpus {
        let (Some(host), Some(memory)) = (gpu["host"].as_str(), gpu.get("host_memory")) else {
            continue;
        };
        let room = memory["available"]
            .as_u64()
            .unwrap_or(0)
            .saturating_sub(memory["reserve"].as_u64().unwrap_or(0))
            .saturating_sub(memory["pools_unallocated"].as_u64().unwrap_or(0));
        let required: u64 = if weights_to_host {
            weights
                .iter()
                .filter(|((on, _), _)| on == host)
                .map(|(_, bytes)| bytes)
                .sum()
        } else {
            0
        };
        hosts.entry(host.to_owned()).or_insert_with(|| {
            serde_json::json!({
                "host": host,
                "available": memory["available"],
                "reserve": memory["reserve"],
                "pools_unallocated": memory["pools_unallocated"],
                "admits": room,
                "required": required,
                "previous_world_weights": weights_to_host.then_some(required),
                "feasible": required <= room,
            })
        });
    }
    serde_json::json!({
        "feasible": gpus.iter().all(|gpu| gpu["feasible"] == true),
        "gpus": gpus,
        "hosts": hosts.into_values().collect::<Vec<_>>(),
    })
}

/// What each step of `specs/online-operations.md` §6.5 frees on the GPU of
/// the plan `gpu` (`ledger::Preparing` of the engine), in bytes, and the
/// promotion stall of moving the requests' KV cache in flight through host
/// memory (step 3), for a successor that adds `successor` and a previous
/// world kept as `previous_world`. In the preparing phase: the idle cached
/// pages of the tail the serving world gives back (step 1), the rest of
/// that tail (step 2), the warm copies a successor that recomputes the
/// cache would otherwise prepare on the device (step 3) and the serving
/// world's graph pool (step 4). In the trial: the previous world's KV cache
/// segments the successor does not map, past its first (step 5, a parked
/// previous world), its weights the successor does not map (step 6, kept in
/// host memory), and its processes' graph pool, process memory and
/// exclusive storages with those segments (step 7, kept as its weights or
/// in host memory).
fn steps(
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
            "warm_copies": bytes(&gpu["successor"]["in_flight"]),
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
        "promotion_stall_ms": gpu["promotion_stall_ms"],
    })
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
                match tokio::time::timeout(LEDGER_READ, client.describe_memory()).await {
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
        let mut statuses = Vec::new();
        for replica in replicas {
            statuses.push(replica.config_status());
        }
        let mut status = statuses
            .iter()
            .find(|status| !status["trial"].is_null() || !status["update"].is_null())
            .unwrap_or(&statuses[0])
            .clone();
        status["replicas"] = statuses.into();
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
fn begin_commit(
    registry: &crate::leader::Replicas,
    current: &[Arc<Replica>],
    retire: Vec<u64>,
    discard_pages: bool,
) {
    let worlds: Vec<_> = current
        .iter()
        .map(|replica| locked(&replica.worlds).executing)
        .collect();
    for replica in current {
        if let Some(spec) = locked(&replica.served).as_mut() {
            spec.predecessor = None;
            spec.retire = retire.clone();
        }
        replica.hold.keep(None);
    }
    *locked(&registry.commit) = Some(Commit {
        pending: worlds.clone(),
        worlds,
        retire,
        discard_pages,
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
            .commit(decision.retire.clone(), decision.discard_pages)
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
        let (mut target, stated, update) = resolve(&current, &request)?;
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
        // A change of the parallel configuration plans what each GPU of the
        // successor maps, receives and creates of the weights, and, unless
        // the update states it, the ranks' device order (§4.7). The order
        // the daemon chose is the target configuration's.
        let redistribution = if world_changes && redistributes && unmapped.is_empty() {
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
                !stated.contains("placement_config.devices"),
            )
            .await
            .map_err(|error| {
                UpdateError::Refused(format!(
                    "the change of the parallel configuration cannot be planned: {error:#}"
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
            planned.filter(|_| world_changes && redistributes)
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
        // A successor of another parallel configuration or another KV cache
        // encoding maps none of the serving world's cache, and one that
        // loads other weights does so when the requests in flight compute
        // their state again.
        let successor = Successor {
            loads_weights: !reloaded.is_empty() || (redistributes && redistribution.is_none()),
            unmapped_cache: redistributes
                || kv_encoding_changes(&served, &target_spec)
                || (!reloaded.is_empty() && request.in_flight == InFlight::Recompute),
            captures_graphs: target_spec
                .cuda_graphs
                .as_ref()
                .is_none_or(|graphs| graphs.policy != "off"),
        };
        let weights = if !unmapped.is_empty() {
            "unmapped"
        } else if successor.loads_weights {
            "loaded"
        } else if redistributes {
            "redistributed"
        } else {
            "mapped"
        };
        let kv_cache = if !unmapped.is_empty() {
            "unmapped"
        } else if successor.unmapped_cache {
            if !reloaded.is_empty() && request.in_flight == InFlight::Recompute {
                "recomputed"
            } else {
                "warming"
            }
        } else {
            "mapped"
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
        // memory ledger first: the serving world's KV cache limit beside a
        // successor, with the weights it creates, per GPU and host
        // (`specs/online-operations.md` §6.6).
        let memory = if world_changes && !in_place && unmapped.is_empty() && shared_devices {
            Some(memory_plan(&running, successor, &budgets, update.previous_world).await)
        } else {
            None
        };
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
            "weights": weights,
            "kv_cache": kv_cache,
            "reloads": reloaded,
            "differs": unmapped,
            "in_flight_requests": running.in_flight(),
            "memory": memory.as_ref().map(|(plan, _)| plan),
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
            running
                .hook(
                    "uniserve.fence_switch",
                    &serde_json::json!({ "enabled": enabled }).to_string(),
                )
                .await
                .map_err(|error| {
                    anyhow::anyhow!("the workers did not switch the probes: {error}")
                })?;
            info!(enabled, "the workers switched the fence probes in place");
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
        if let Some((_, Some(refusal))) = memory {
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
        let report = self
            .change_world(
                &running,
                served,
                current,
                target,
                target_spec,
                target_digest,
                &update,
                request.in_flight,
                successor,
                redistributes,
                &budgets,
                redistribution,
            )
            .await?;
        Ok(Applied::World(report))
    }

    /// Moves the deployment from the world `running` executes to a
    /// successor executing `target`, up to its trial (steps 1 to 3 of the
    /// module). A successor of another parallel configuration
    /// (`redistributes`) maps none of the serving world's KV cache, and the
    /// requests in flight compute their state again on it; it creates the
    /// resources `budgets` states on each GPU, which the serving world's
    /// cache makes room for.
    #[expect(
        clippy::too_many_arguments,
        reason = "one world change: both worlds' specs and configurations and its options"
    )]
    async fn change_world(
        self: &Arc<Self>,
        running: &Arc<EngineClient>,
        served: wire::EngineSpec,
        current: ServeConfig,
        target: ServeConfig,
        mut spec: wire::EngineSpec,
        target_digest: String,
        update: &crate::options::UpdateOptions,
        in_flight: InFlight,
        adds: Successor,
        redistributes: bool,
        budgets: &BTreeMap<storage::manifest::Gpu, storage::manifest::weights::RankBudget>,
        redistribution: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, UpdateError> {
        let started = Instant::now();
        let previous_world = running.world();
        let shared_devices = !devices(&served)?.is_disjoint(&devices(&spec)?);
        // The serving world gives its graph memory to the successor's
        // captures first, so the shrink that follows leaves it to the
        // successor.
        let eager_since = if update.eager_prepare {
            self.progress("eager", None);
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
        self.progress("shrink", None);
        // The serving world's cache gives back the tail past the limit each
        // GPU's ledger leaves it beside the successor; its requests run on.
        let shrunk =
            if shared_devices {
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
        let memory = memory_report(&shrunk, update.previous_world, adds);
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
                self.progress("baseline", None);
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

        self.progress("prepare", None);
        let control = Arc::clone(self.control.get().ok_or(UpdateError::NoEngine)?);
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
            let task = tokio::spawn(crate::interference::control(
                Arc::clone(running),
                vec![(Arc::clone(&self.observer), throttling)],
                crate::interference::INTERVAL,
                done.clone(),
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
            let changed = running.info().prefix_seed != successor.info().prefix_seed;
            let warmed = if adds.unmapped_cache && !(changed && in_flight == InFlight::Recompute) {
                self.progress("warm_cache", Some(successor.world()));
                match warming::prepare(
                    std::slice::from_ref(running),
                    &successor,
                    changed,
                    update.cache_tail_tokens.map(std::num::NonZeroU32::get),
                    update.cache_prefill_tokens.map(std::num::NonZeroU32::get),
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
            Ok((successor, warmed))
        }
        .await;
        let (done, task) = interference;
        done.cancel();
        let throttled = task.await.unwrap_or_default();
        let (successor, warmed) = match prepared {
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
        self.progress("promote", Some(world));

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
        let (before, after) = (running.info(), successor.info());
        let pages_change = before.prefix_seed != after.prefix_seed;
        let trajectories_change = before.latent_seed != after.latent_seed;
        let numerics_change = pages_change || trajectories_change;
        let hold = Arc::clone(self.hold());
        // Only these continuations may reuse state computed with the prior
        // numerics, whether its backing is mapped or copied.
        if numerics_change && in_flight == InFlight::Keep {
            hold.keep(Some(Kept {
                world: previous_world,
                prefix_seed: pages_change.then_some(before.prefix_seed),
                latent_seed: trajectories_change.then_some(before.latent_seed),
            }));
        }
        // A profiling session belongs to the serving world: it ends, its
        // files written, before the world parks.
        crate::profiling::end_for_world(self, Some(running), "world change").await;
        let paused = Instant::now();
        hold.hold();
        let index = match running.park().await {
            Ok(index) => index,
            Err(error) => {
                // The serving world serves on: it resumes, and the successor
                // closes with its memory.
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
        // A previous world kept parked keeps its memory until the commit; one
        // kept as its weights frees the rest as the successor serves.
        let grows = update.previous_world != PreviousWorld::Parked;
        let index = if adds.unmapped_cache {
            Vec::new()
        } else {
            index
        };
        let promotion: anyhow::Result<()> = async {
            warming::resume_prefill(std::slice::from_ref(running))
                .await
                .map_err(anyhow::Error::msg)?;
            successor.promote(index, recompute, grows).await?;
            Ok(())
        }
        .await;
        if let Err(error) = promotion {
            hold.keep(None);
            let _ = running.resume(vec![world]).await;
            hold.serve(Arc::clone(running));
            self.close(&successor).await;
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
            "the successor world serves on trial"
        );
        crate::leader::reload_adapters(self, &successor).await;

        // The previous world as the trial keeps it.
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
        // The successor's cache grows into the memory the previous world
        // freed: a previous world kept as its weights released, with its
        // processes, their graph pools and process memory, and keeps its
        // weights alone, so every other storage of it the successor does
        // not map is released first; kept in host memory, its weights that
        // the successor does not map move there too. Each GPU's ledger
        // leaves the successor all of it at once. A node that cannot move
        // the weights (no Transfer Server, or more than the host admits)
        // keeps them on its GPUs, and the previous world is kept as its
        // weights.
        let to_host = update.previous_world == PreviousWorld::Host;
        let mut mode = update.previous_world;
        let mut host_refused = None;
        let mut moved_bytes = 0;
        let mut evicted: Option<serde_json::Value> = None;
        match &previous {
            Previous::Parked(parked) => {
                // A parked previous world whose cache the successor maps none
                // of keeps it for a rollback alone: it evicts every segment
                // past its first once their idle pages reached the host KV
                // cache, and the successor grows into them (step 5 of
                // `specs/online-operations.md` §6.5). A rollback reads them
                // back from there and computes the rest again.
                if adds.unmapped_cache {
                    match parked.resize_cache(wire::CacheChange::Evict).await {
                        Ok(results) => {
                            evicted = results
                                .iter()
                                .find_map(|result| serde_json::from_str(result).ok());
                        }
                        Err(error) => warn!("the parked world did not evict its KV cache: {error}"),
                    }
                }
                if let Err(error) = successor.resize_cache(wire::CacheChange::Grow).await {
                    warn!("the successor's KV cache did not grow: {error}");
                }
            }
            Previous::Weights => {
                match successor
                    .grow_keeping_weights(vec![previous_world], to_host)
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
        *locked(&self.served) = Some(served_spec);
        *locked(&self.config) = target;
        let report = serde_json::json!({
            "world": world,
            "previous_world": previous_world,
            "prepare_seconds": prepare.as_secs_f64(),
            "eager_baseline_seconds": baseline_wait.map(|wait| wait.as_secs_f64()),
            "transfer_halvings": throttled.halvings,
            "transfer_restorations": throttled.restorations,
            "park_seconds": parked.as_secs_f64(),
            "pause_seconds": pause.as_secs_f64(),
            "previous_world_kept": mode,
            "host_refused": host_refused,
            "moved_to_host_bytes": to_host.then_some(moved_bytes),
            "evicted": evicted,
            "numerics": if numerics_change { "changed" } else { "unchanged" },
            "parallel": if redistributes { "changed" } else { "unchanged" },
            "redistribution": redistribution,
            "in_flight": if numerics_change { Some(in_flight) } else { None },
            "weights": if adds.loads_weights {
                "loaded"
            } else if redistributes {
                "redistributed"
            } else {
                "mapped"
            },
            "kv_cache": if recompute { "recomputed" } else if warmed.is_some() { "warmed" } else { "mapped" },
            "cache_warming": warmed,
            "memory": memory,
        });
        {
            let mut worlds = locked(&self.worlds);
            worlds.executing = world;
            worlds.progress = None;
            worlds.trial = Some(Trial {
                world,
                digest: target_digest,
                previous_world,
                previous_spec: served,
                previous_config: current,
                previous,
                mode,
                host_refused,
                pages_change,
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
        begin_commit(
            registry,
            &[Arc::clone(self)],
            vec![trial.previous_world],
            trial.pages_change,
        );
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

    /// This replica's configuration, executing world, trial and update history.
    pub(crate) fn config_status(&self) -> serde_json::Value {
        let config = locked(&self.config).clone();
        let printed = crate::options::printed(&config).unwrap_or_default();
        let digest = digest(&config).unwrap_or_default();
        let worlds = locked(&self.worlds);
        let hold = self.hold();
        let executing = hold.client().map(|client| {
            serde_json::json!({
                "world": client.world(),
                "state": match client.state() {
                    EngineState::Standby => "preparing",
                    EngineState::Ready => "executing",
                    EngineState::Parked => "parked",
                    EngineState::Failed(_) => "failed",
                    EngineState::Closed => "closed",
                },
            })
        });
        let other = worlds.trial.as_ref().map(|trial| {
            serde_json::json!({
                "world": trial.previous_world,
                "state": match trial.mode {
                    PreviousWorld::Parked => "parked",
                    PreviousWorld::Weights => "weights",
                    PreviousWorld::Host => "host",
                },
            })
        });
        let other = other.or_else(|| {
            worlds.progress.as_ref().and_then(|progress| {
                progress
                    .world
                    .map(|world| serde_json::json!({ "world": world, "state": "preparing" }))
            })
        });
        let trial = worlds.trial.as_ref().map(|trial| {
            let rates = trial
                .thresholds
                .window
                .and_then(|window| self.observer.rates(trial.world, trial.promoted, window));
            serde_json::json!({
                "world": trial.world,
                "previous_world": trial.previous_world,
                "previous_world_kept": trial.mode,
                "host_refused": trial.host_refused,
                "seconds": trial.promoted.elapsed().as_secs_f64(),
                "promoted_at": trial.promoted_at,
                "pause_seconds": trial.pause.as_secs_f64(),
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
            "update": worlds.progress,
            "trial": trial,
            "unavailable": worlds.unavailable.values().collect::<Vec<_>>(),
            "last": worlds.last,
        })
    }

    fn progress(&self, phase: &str, world: Option<u64>) {
        let mut worlds = locked(&self.worlds);
        let started = worlds
            .progress
            .as_ref()
            .map_or_else(now, |progress| progress.started);
        let previous = worlds.progress.as_ref();
        worlds.progress = Some(Progress {
            phase: phase.to_owned(),
            started,
            world: world.or(previous.and_then(|progress| progress.world)),
            note: previous.and_then(|progress| progress.note.clone()),
        });
    }

    /// Notes `note` on the change in progress, for its later phases too.
    fn progress_note(&self, note: String) {
        if let Some(progress) = locked(&self.worlds).progress.as_mut() {
            progress.note = Some(note);
        }
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
    if let Some(failing) = failing {
        if failing.state() != EngineState::Closed {
            match tokio::time::timeout(CLOSED_EXIT, failing.park()).await {
                Ok(Ok(_)) => {}
                Ok(Err(error)) => info!(%error, "the world on trial parked in part"),
                Err(_) => warn!(
                    "the world on trial did not park within {CLOSED_EXIT:?}; closing it with \
                     the requests it holds"
                ),
            }
        }
        close(&failing).await;
    }
    let previous = match trial.previous {
        Previous::Parked(previous) => {
            previous.resume(vec![trial.world]).await.map_err(|error| {
                hold.close(format!("the previous world did not resume: {error}"));
                anyhow::anyhow!("the previous world did not resume: {error}")
            })?;
            replay_graphs(&previous, trial.eager).await;
            previous
        }
        Previous::Weights => {
            let control = Arc::clone(replica.control.get().ok_or(UpdateError::NoEngine)?);
            let sink = stats_sink(
                replica.model_name.clone(),
                Arc::clone(&replica.observer),
                replica.ordinal,
            );
            // The world on trial, closed, is retired before the previous
            // world's workers start: they need its memory (weights of its
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
    hold.serve(Arc::clone(&previous));
    crate::leader::reload_adapters(replica, &previous).await;
    if let Err(error) = previous.resize_cache(wire::CacheChange::Restore).await {
        warn!("the previous world's KV cache was not restored: {error}");
    }
    let pause = paused.elapsed();
    *locked(&replica.served) = Some(trial.previous_spec);
    *locked(&replica.config) = trial.previous_config;
    let report = serde_json::json!({
        "rolled_back": trial.world,
        "world": trial.previous_world,
        "reason": reason,
        "pause_seconds": pause.as_secs_f64(),
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
            graph_pc_sampling: false,
        });
        let switched = spec(wire::Profiling {
            fence_probes: false,
            ..served.profiling.clone().unwrap()
        });
        assert!(switches_probes_only(&served, &switched));
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

    /// Each step of a world change frees what its retention mode and the
    /// successor leave to it: the tail's idle pages and the rest of the tail
    /// in the preparation; in the trial, a parked previous world's segments
    /// the successor does not map, past its first, or, kept as its weights
    /// or in host memory, its processes with those segments, and in host
    /// memory its weights the successor does not map.
    #[test]
    fn each_step_frees_what_the_mode_and_the_successor_leave_it() {
        let gpu = serde_json::json!({
            "serving": {"weights": 600, "graph_pool": 50, "process": 30, "exclusive": 5},
            "serving_kv_cache": 800,
            "kv_kept": 500,
            "kv_floor": 100,
            "idle_cached_tail": 120,
            "successor": {"in_flight": 70},
            "promotion_stall_ms": 9,
        });
        let reloading = Successor {
            loads_weights: true,
            unmapped_cache: true,
            captures_graphs: true,
        };
        let parked = steps(&gpu, PreviousWorld::Parked, reloading);
        assert_eq!(
            parked["preparation"],
            serde_json::json!({
                "idle_prefix_pages": 120,
                "kv_tail": 180,
                "warm_copies": 70,
                "graph_pool": 50,
            })
        );
        assert_eq!(
            parked["trial"],
            serde_json::json!({
                "unmapped_kv_segments": 400,
                "weights_to_host": 0,
                "processes": 0,
            })
        );
        assert_eq!(parked["promotion_stall_ms"], 9);

        let host = steps(&gpu, PreviousWorld::Host, reloading);
        assert_eq!(
            host["trial"],
            serde_json::json!({
                "unmapped_kv_segments": 0,
                "weights_to_host": 600,
                "processes": 585,
            })
        );
        // A successor that maps the weights and the cache leaves the trial
        // the previous world's processes alone.
        let weights = steps(&gpu, PreviousWorld::Weights, Successor::default());
        assert_eq!(
            weights["trial"],
            serde_json::json!({
                "unmapped_kv_segments": 0,
                "weights_to_host": 0,
                "processes": 85,
            })
        );
    }

    /// A world change carries over a change of the build, and one of the
    /// weights' sources (the successor loads them) or of the KV cache's
    /// encoding (the successor maps none of it), apart from a parallel
    /// change; a change of the KV cache's size is not carried over.
    #[test]
    fn a_change_reloads_weights_and_recomputes_the_cache_it_cannot_read() {
        let served = wire::EngineSpec {
            model: "a".into(),
            kv_cache: Some(wire::KvCache::default()),
            workers: Some(wire::Workers {
                executable: "old".into(),
                dtype: "bfloat16".into(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut target = served.clone();
        if let Some(workers) = target.workers.as_mut() {
            workers.executable = "new".into();
        }
        assert!(differing(&served, &target).is_empty());
        assert!(reloads(&served, &target).is_empty());
        assert!(!kv_encoding_changes(&served, &target));

        target.model = "b".into();
        assert!(differing(&served, &target).is_empty());
        assert_eq!(reloads(&served, &target), vec!["the model"]);
        assert!(!kv_encoding_changes(&served, &target));

        // The cache follows the compute dtype unless it states its own.
        if let Some(workers) = target.workers.as_mut() {
            workers.dtype = "float16".into();
        }
        assert!(kv_encoding_changes(&served, &target));
        if let Some(kv) = target.kv_cache.as_mut() {
            kv.dtype = "fp8_e4m3".into();
        }
        assert!(differing(&served, &target).is_empty());
        assert!(kv_encoding_changes(&served, &target));

        // The page size is a parallel change; the cache's size is not
        // carried over.
        if let Some(kv) = target.kv_cache.as_mut() {
            kv.block_size = Some(32);
        }
        assert!(differing(&served, &target).is_empty());
        assert!(parallel_change(&served, &target));
        if let Some(kv) = target.kv_cache.as_mut() {
            kv.max_tokens = Some(1024);
        }
        assert_eq!(differing(&served, &target), vec!["the KV cache"]);
    }

    /// Another parallel configuration redistributes the weights and the KV
    /// cache rather than being refused; the sizing of the same ranks is
    /// still refused.
    #[test]
    fn a_parallel_configuration_change_redistributes() {
        let served = wire::EngineSpec {
            placement: Some(wire::Placement {
                tensor_parallel_size: 2,
                data_parallel_size: 2,
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut target = served.clone();
        if let Some(placement) = target.placement.as_mut() {
            placement.tensor_parallel_size = 1;
            placement.data_parallel_size = 4;
        }
        target.experts = Some(wire::Experts {
            expert_parallel: true,
            ..Default::default()
        });
        assert!(parallel_change(&served, &target));
        assert!(differing(&served, &target).is_empty());

        let mut sized = served.clone();
        if let Some(placement) = sized.placement.as_mut() {
            placement.mem_fraction_static = 0.5;
        }
        assert!(!parallel_change(&served, &sized));
        assert_eq!(differing(&served, &sized), vec!["the placement"]);
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
}
