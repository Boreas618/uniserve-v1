//! The update entry: one change of the deployment's configuration, decided
//! by the layer that owns it (`PUT /admin/config`, `specs/worker.md` §7).
//!
//! A change is one of three kinds. What the main processes are built from —
//! their executable, the parallel ranks, the deployment file, the model's
//! files, the placement, the KV cache's options — is `BuildConfig`'s, and
//! the epoch layer switches to a new epoch (§7.4, [`crate::updates::switch`]).
//! What they execute — the programs — is the program layer's, prepared
//! against every host's Program Server before the processes switch to them
//! (§7.2). Everything else is this daemon's own: its options change in a hot
//! restart (§7.5), and the scheduling limits change in place at the engine's
//! next step boundary, within the batch its workers were started for.
//!
//! `Leader::apply` decides in that order and answers what it applied;
//! `Leader::commit` and `Leader::rollback` are the switch's. A change that
//! fits no layer is refused by name: the mechanism that moved a deployment
//! to a successor world, with its trial and its rollback, is gone, and
//! nothing here starts a second engine beside the one that serves.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Context as _;
use proto::request::wire;
use tracing::info;

use crate::engine::client::EngineState;
use crate::leader::{Leader, Replica, UpdateError, locked};
use crate::options::ServeConfig;
use crate::updates::change::{Applied, ApplyRequest, resolve};
use crate::updates::epoch::Unavailable;

/// The deployment's worlds as the leader tracks them.
#[derive(Default)]
pub(crate) struct WorldState {
    /// The world the executing engine runs.
    pub(crate) executing: u64,
    pub(crate) unavailable: BTreeMap<String, Unavailable>,
    /// What the last world change, commit or rollback reported.
    pub(crate) last: Option<serde_json::Value>,
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

impl Leader {
    /// Applies a configuration through the replica that owns its worlds.
    pub async fn apply(&self, request: ApplyRequest) -> Result<Applied, UpdateError> {
        let _replacing = self
            .replicas
            .replacing
            .try_lock()
            .map_err(|_| UpdateError::InProgress)?;
        let replicas = self.replicas();
        let current = locked(&replicas[0].config).clone();
        let (target, stated, _update) = resolve(&current, &request)?;
        // A change of what the main processes build is the epoch layer's
        // (§7.4): one switch for the deployment, before any worlds work. The
        // program layer's and the host layer's paths stay below until their
        // own switches replace them.
        if let Some(applied) = self.apply_epoch(&current, &target, request.dry_run)? {
            return Ok(applied);
        }
        // The program layer (§7.2): only what the serving epoch's processes
        // execute differs, and the programs are prepared against every host's
        // Program Server before the processes switch to them. A program-layer
        // update is named by the epoch it executes in; `route` answers `None`
        // when the programs a configuration selects do not differ, and the
        // world path below decides the rest.
        let epoch = self
            .updates
            .epochs()
            .serving()
            .map_or(0, |(epoch, _)| epoch);
        if let Some(update) = self.apply_programs(epoch, &target, request.dry_run).await? {
            return Ok(Applied::Program(serde_json::json!({
                "change": "program",
                "epoch": epoch,
                "update": update.number,
            })));
        }
        let indices: Vec<_> = match request.replica {
            Some(index) => vec![index],
            None => (0..replicas.len()).collect(),
        };
        let mut reports = Vec::new();
        for index in indices.iter().copied() {
            let replica = replicas
                .get(index)
                .ok_or_else(|| UpdateError::Refused(format!("no replica {index}")))?;
            match Box::pin(replica.apply(request.clone())).await? {
                applied @ Applied::Daemon { .. } => {
                    if request.replica.is_some() {
                        return Err(UpdateError::Refused(
                            "daemon options apply to the deployment, not one replica".to_owned(),
                        ));
                    }
                    return Ok(applied);
                }
                Applied::Planned(report) | Applied::InPlace(report) => reports.push(report),
                applied @ (Applied::Epoch(_) | Applied::Program(_)) => return Ok(applied),
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

    /// The epoch layer's arm of [`Leader::apply`]: a change that starts main
    /// processes which differ in what they build is a switch between epochs
    /// (`specs/worker.md` §7.4), and the two deployments alone decide it. It
    /// answers `Ok(None)` when the change is another layer's — the host path,
    /// the programs, or nothing.
    ///
    /// The change this builds is the switch's whole input: the serving epoch
    /// and its launch generation, the epoch the target becomes, one spawn
    /// generation for the deployment (above every rank's standby, §2.5), the
    /// target's executable and the arguments every engine process of this
    /// daemon starts with.
    fn apply_epoch(
        &self,
        current: &ServeConfig,
        target: &ServeConfig,
        dry_run: bool,
    ) -> Result<Option<Applied>, UpdateError> {
        let previous = crate::epoch::Deployment::of(current.clone());
        let next = crate::epoch::Deployment::of(target.clone());
        if crate::epoch::layer(&previous, &next) != crate::epoch::Layer::Epoch {
            return Ok(None);
        }
        // One update at a time (§7.2): a program update on probation is the
        // record of what serves, so an epoch switch over it is refused by
        // name, before the build config is written or a child is started.
        if let Some(in_flight) = crate::updates::in_flight(
            self.updates.running(),
            self.epochs().program_update().is_some(),
        ) {
            return Err(UpdateError::Refused(format!(
                "another update is in flight: {in_flight}"
            )));
        }
        // One main process per node and one launch file per process: the file
        // names its peers' request addresses, which this daemon does not know
        // for a node it does not run on, so it can state a world of one only.
        // A world of more is refused by name instead of silently writing
        // `world_size = 1` for two processes (exec's rule: the world is its
        // peers and itself).
        // One main process, not only one node: a data-parallel deployment runs
        // one replica per main process, and this daemon writes one launch file
        // and starts one child per node. The count is the target's, decided
        // from the same expression the replica rollout uses.
        let processes = if target.moe_config.expert_parallel {
            1
        } else {
            target.data_parallel_size.get()
        };
        if processes != 1 {
            return Err(UpdateError::Refused(format!(
                "the target epoch runs {processes} data-parallel replicas, and the epoch layer \
                 starts one main process per node: a replica's launch needs the contents carried \
                 in `process::Spawn` or a world that states every replica's addresses"
            )));
        }
        if target.nnodes.get() != 1 {
            return Err(UpdateError::Refused(format!(
                "the target epoch runs {} nodes, and a launch file written here names no peer: a \
                 multi-node epoch needs the launch contents carried in `process::Spawn` or a \
                 world that states its peers' request addresses",
                target.nnodes.get()
            )));
        }
        let epochs = self.updates.epochs();
        let (previous_epoch, serving) = epochs.serving().ok_or_else(|| {
            UpdateError::Refused("no epoch serves: there is nothing to switch from".to_owned())
        })?;
        let previous_generation = serving.generation.ok_or_else(|| {
            UpdateError::Refused("the serving epoch names no launch generation".to_owned())
        })?;
        // The deployment's argv comes from its leader node's control: the
        // arguments an engine process of this daemon starts with are the
        // same for every host (the daemon's sockets travel with them).
        let replicas = self.replicas();
        let control = replicas[0].control.get().ok_or(UpdateError::NoEngine)?;
        // One number for the whole deployment: every rank's counter is
        // advanced, and the highest is the generation each child of the epoch
        // is spawned at, so none collides with a rank's standby (§2.5). A dry
        // run advances the counters too and reports the number the switch
        // would use; the counters are monotone, not a promise.
        let group = crate::engine::control::ENGINE_GROUP;
        let spawn_generation = (0..u64::from(target.nnodes.get()))
            .map(|rank| epochs.next_generation(group, rank))
            .max()
            .unwrap_or(0);
        let target_epoch = previous_epoch + 1;
        // The epoch's own build config is written once, and the launch file
        // every process of it reads names that path (§2.3). The world is the
        // epoch's identity, which the switch claims.
        control
            .write_build_config(target_epoch, &next.build_config)
            .map_err(UpdateError::Failed)?;
        let mut spec = target.engine_spec().map_err(UpdateError::Failed)?;
        spec.world = target_epoch;
        let change = crate::updates::epoch::EpochChange {
            previous: previous_epoch,
            target: target_epoch,
            // The update's number is the epoch it creates: epochs only grow.
            update: crate::epoch::Update::new(target_epoch, next.clone()),
            group: group.to_owned(),
            start: control
                .epoch_start(&spec, spawn_generation)
                .map_err(UpdateError::Failed)?,
            previous_generation,
            spawn_generation,
        };
        self.updates.apply(&previous, &next, change, dry_run)
    }

    /// Commits the update in flight: the switch's, whose commit is a command
    /// to the target's processes, recorded when every one of them
    /// acknowledges (§7.4).
    pub async fn commit(&self) -> Result<serde_json::Value, UpdateError> {
        let _replacing = self.replicas.replacing.lock().await;
        if self.updates.running() {
            if self.updates.commit() {
                return Ok(self
                    .updates
                    .report()
                    .unwrap_or_else(|| serde_json::json!({})));
            }
        }
        Ok(serde_json::json!({ "commit": "no update in flight" }))
    }

    /// Restores the previous epoch of the update in flight.
    pub async fn rollback(&self, reason: String) -> Result<serde_json::Value, UpdateError> {
        let _replacing = self.replicas.replacing.lock().await;
        if self.updates.running() {
            if self.updates.rollback(&reason) {
                return Ok(self
                    .updates
                    .report()
                    .unwrap_or_else(|| serde_json::json!({})));
            }
        }
        Err(UpdateError::Refused(
            "no update is in flight: the deployment already serves the configuration the file              states"
                .into(),
        ))
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
        let statuses: Vec<_> = replicas
            .into_iter()
            .map(|replica| replica.config_status())
            .collect();
        let mut status = statuses.first().cloned().unwrap_or_default();
        status["replicas"] = statuses.into();
        status["gpus"] = ledgers.as_array().map_or(serde_json::Value::Null, |nodes| {
            deployment_gpus(nodes).into()
        });
        status["ledgers"] = ledgers;
        status
    }
}

/// Completes a recorded decision. Publication progress survives both an
/// error response and daemon handoff; node and engine operations are
/// idempotent when a response was lost after the operation took effect.
#[expect(
    clippy::unwrap_used,
    reason = "the deployment replacement lock retains the decision until its final take"
)]

impl Replica {
    /// Reports what the replica executes and what its configuration is: the
    /// engine that takes requests with its state, the digest of the
    /// configuration, and the configurations a rollback marked unavailable.
    pub(crate) fn config_status(&self) -> serde_json::Value {
        let config = locked(&self.config).clone();
        let printed = crate::options::printed(&config).unwrap_or_default();
        let digest = digest(&config).unwrap_or_default();
        let worlds = locked(&self.worlds);
        let hold = self.hold();
        let slot = |world: u64, state: &str| serde_json::json!({ "world": world, "state": state });
        let unavailable = |world: u64, reason: &str| serde_json::json!({ "world": world, "state": "unavailable", "reason": reason });
        let executing = match hold.client() {
            Some(client) => {
                let world = client.world();
                match client.state() {
                    EngineState::Standby => slot(world, "preparing"),
                    EngineState::Ready => slot(world, "executing"),
                    EngineState::Parked => slot(world, "parked"),
                    EngineState::Failed(reason) => unavailable(world, &reason),
                    EngineState::Closed => unavailable(world, "its engine process ended"),
                }
            }
            None => unavailable(worlds.executing, "requests wait for its engine"),
        };
        serde_json::json!({
            "config": printed,
            "digest": digest,
            "executing": executing,
            "unavailable": worlds.unavailable.values().collect::<Vec<_>>(),
            "last": worlds.last,
        })
    }

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
        if self.asleep.load(Ordering::Acquire) {
            return Err(UpdateError::Refused(
                "the engine sleeps: wake it (/admin/wake_up) before updating it".to_owned(),
            ));
        }
        let current = locked(&self.config).clone();
        let (target, stated, _update) = resolve(&current, &request)?;
        // A configuration a rollback marked unavailable stays refused unless
        // the change is forced.
        self.available(&target, request.force)?;
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
        let daemon_changes = crate::options::daemon_options_differ(&current, &target);
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
        // The daemon orchestrates what changes in its own options: a hot
        // restart applies them first, and its successor applies the change
        // again, which then leaves the daemon's options as they are.
        if daemon_changes && !in_place {
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
            return Ok(Applied::InPlace(
                serde_json::json!({ "in_place": ["the fence probes"] }),
            ));
        }
        // Scheduling alone, within the batch the serving engine's workers
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
            return Ok(Applied::InPlace(
                serde_json::json!({ "in_place": ["the scheduling limits"] }),
            ));
        }
        // The layers above decide every change that reaches here: the epoch
        // layer owns what the main processes are built from (the executable,
        // the parallel ranks, the deployment file, the model's files, the
        // placement and the KV cache's options, all of them `BuildConfig`),
        // the program layer what they execute, and this one the daemon's own
        // options and the scheduling limits, in place. A change that fits
        // none of them would have needed a successor world, which this
        // daemon no longer runs: it is refused by name here rather than
        // half served.
        Err(UpdateError::Refused(format!(
            "the change fits no layer: it would have needed a successor engine ({}) and the \
             epoch layer starts one main process from the build config",
            differing(&served, &target_spec).join(", ")
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    async fn the_status_reports_the_executing_engine_and_the_gpus_the_deployment_books() {
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
            status["executing"],
            serde_json::json!({ "world": world, "state": "executing" })
        );
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

        // A world whose engine no longer serves is unavailable.
        replica.hold().close("the engine process ended");
        let status = leader.config_status().await;
        assert_eq!(
            status["executing"],
            serde_json::json!({
                "world": world,
                "state": "unavailable",
                "reason": "the engine process ended",
            })
        );
        client.shutdown();
    }
}
