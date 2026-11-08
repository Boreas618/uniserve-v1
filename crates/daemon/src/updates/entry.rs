//! The update entry: one change of the deployment's configuration, decided
//! by the layer that owns it (`PUT /admin/config`, `specs/worker.md` §7).
//!
//! A change is one of three kinds. What the main processes are built from —
//! their executable, the parallel ranks, the deployment file, the model's
//! files, the rank layout, the KV cache's options — is `proto::config::EngineConfig`'s, and
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

use proto::config::EngineConfig;
use tracing::info;

use crate::engine::client::EngineState;
use crate::engine::hold::HoldState;
use crate::leader::{Leader, Replica, UpdateError, locked};
use crate::options::ServerArgs;
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
pub(crate) fn digest(config: &ServerArgs) -> anyhow::Result<String> {
    use sha2::Digest as _;
    let printed = crate::options::printed(config)?;
    Ok(sha2::Sha256::digest(printed.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

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
fn switches_probes_only(served: &EngineConfig, target: &EngineConfig) -> bool {
    let mut switched = comparable(served);
    switched.profiling.fence_probes = target.profiling.fence_probes;
    switched
        .profiling
        .fence_probes_off
        .clone_from(&target.profiling.fence_probes_off);
    switched == comparable(target)
}

/// The parts of `config` a replica compares: all of it but what the program
/// layer owns (§7.7), the program backends, the scheduling policy, the expert
/// exchange and the speculative cost figures. A change reaches a replica only
/// once the program layer has answered for those.
fn comparable(config: &EngineConfig) -> EngineConfig {
    let mut comparable = config.clone();
    comparable.attention_backend = String::new();
    comparable.prefill_attention_backend = None;
    comparable.decode_attention_backend = None;
    comparable.moe_runner_backend = None;
    comparable.scheduler.policy = proto::config::SchedulerConfig::default().policy;
    comparable.moe.exchange = proto::config::MoeConfig::default().exchange;
    comparable.speculative.expected_speedup = None;
    comparable.speculative.refill_token_cost = None;
    comparable
}

/// Keeps an existing replica's assigned devices when an update changes its
/// execution options without requesting a different set of ranks: each
/// group of `target` that `served` places with as many ranks keeps the
/// served group's ranks.
fn retain_devices(served: &EngineConfig, target: &mut EngineConfig) {
    for group in &mut target.parallel.worker_groups {
        if let Some(before) = served
            .parallel
            .worker_groups
            .iter()
            .find(|before| before.id == group.id)
            && before.ranks.len() == group.ranks.len()
        {
            group.ranks.clone_from(&before.ranks);
        }
    }
}

/// Changes outside a successor's current process-control contract. Rank
/// resources and numerical layouts are planned by the target world.
fn differing(served: &EngineConfig, target: &EngineConfig) -> Vec<&'static str> {
    let mut parts = Vec::new();
    if served.deployment != target.deployment {
        parts.push("the deployment");
    }
    parts
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
        // A switch that ended still hands the replica to the epoch that serves, and the serving
        // client is the one this change would be built from.
        if !request.dry_run && replicas.iter().any(|replica| replica.follows_a_switch()) {
            return Err(UpdateError::InProgress);
        }
        let current = locked(&replicas[0].config).clone();
        let (target, _stated, update) = resolve(&current, &request)?;
        // A target that reverted with a failure is unavailable until it is
        // applied with `force` (§7.2, §7.3 step 4), whichever layer would
        // apply it: the epoch layer's switch and the program layer both end
        // by recording the update here.
        let resolved_target = crate::leader::deployment(&target)?;
        if !request.force
            && let Some(failure) = self.updates.epochs().unavailable(&resolved_target)
        {
            return Err(UpdateError::Refused(format!(
                "the configuration reverted with a failure ({}); apply it again with force",
                failure.reason
            )));
        }
        // The same mark carried across a handover: a predecessor daemon that
        // adopted it kept it in the replica's state, not in an update record.
        if !request.force {
            replicas[0].available(&target, request.force)?;
        }
        // One update at a time (§7.2, §7.4), whichever layer the change is
        // the other's: while a switch runs or a program update is on
        // probation, the `served` spec and the records this entry would read
        // belong to that update. A dry run answers without touching them.
        if !request.dry_run
            && let Some(update) = crate::updates::in_flight(
                self.updates.running(),
                self.epochs().program_update().is_some(),
            )
        {
            return Err(UpdateError::Refused(format!(
                "another update is in flight: {update}"
            )));
        }
        // A change of what the main processes build is the epoch layer's
        // (§7.4): one switch for the deployment, decided before either of the
        // other two layers looks at the change.
        if let Some(applied) = self
            .apply_epoch(&current, &target, &update, request.dry_run)
            .await?
        {
            return Ok(applied);
        }
        // The program layer (§7.2): only what the serving epoch's processes
        // execute differs, and the programs are prepared against every host's
        // Program Server before the processes switch to them. A program-layer
        // update is named by the epoch it executes in; the program layer
        // answers `None` when the programs a configuration selects do not
        // differ, and what remains is the daemon's own options and the
        // scheduling limits.
        let epoch = self
            .updates
            .epochs()
            .serving()
            .map_or(0, |(epoch, _)| epoch);
        if let Some(update) = self
            .apply_programs(epoch, &target, &update, request.force, request.dry_run)
            .await?
        {
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
    async fn apply_epoch(
        &self,
        current: &ServerArgs,
        target: &ServerArgs,
        update: &crate::options::UpdateConfig,
        dry_run: bool,
    ) -> Result<Option<Applied>, UpdateError> {
        let previous = crate::leader::deployment(current)?;
        let next = crate::leader::deployment(target)?;
        if crate::epoch::layer(&previous, &next) != crate::epoch::Layer::Epoch {
            return Ok(None);
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
        // Whether this daemon has an engine process of its own is asked first:
        // a daemon that has none runs no epoch change at all, and the missing
        // means — not the state of its epochs — is what its caller has to fix,
        // so the answer is "no engine process here" (§7.2: the service cannot
        // apply the change) rather than a refusal of a valid one.
        //
        // The deployment's argv comes from its leader node's control: the
        // arguments an engine process of this daemon starts with are the
        // same for every host (the daemon's sockets travel with them).
        let replicas = self.replicas();
        let control = replicas[0].control.get().ok_or(UpdateError::NoEngine)?;
        let serving_client = replicas[0].hold().client().ok_or(UpdateError::NoEngine)?;
        let epochs = self.updates.epochs();
        let (previous_epoch, serving) = epochs.serving().ok_or_else(|| {
            UpdateError::Refused("no epoch serves: there is nothing to switch from".to_owned())
        })?;
        // The previous epoch's processes are addressed by their LAUNCH
        // generation — `Drain`, `Stop` and `Purge` take it — never by the
        // record's `generation`, which is the model generation the main process
        // numbers for its programs (`EpochRecord`, `ManagedProcess`). The
        // authority for that number is the client itself: it is the engine this
        // daemon started or adopted, so it is the process those commands must
        // reach. The process table is no substitute — it holds no row for the
        // epoch the daemon started with (that child is `EngineControl`'s, not
        // the roster's) and can hold a lingering replacement above the one that
        // serves.
        let previous_generation = serving_client.engine().ok_or_else(|| {
            UpdateError::Refused(format!(
                "the main process serving epoch {} has no launch generation: this daemon neither \
                 started nor adopted it",
                serving.world
            ))
        })?;
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
        // The epoch's number is the world it executes, so it is above every epoch and world the
        // daemon knows: an epoch a revert retired keeps its number, a world's storage ends with the
        // world it was claimed for, and the first epoch's world is its first engine's, not 0.
        let target_epoch = epochs
            .records()
            .iter()
            .map(|(epoch, record)| (*epoch).max(record.world))
            .max()
            .unwrap_or(previous_epoch)
            .max(previous_epoch)
            + 1;
        // The target's engine configuration is resolved as a deployment's start resolves it: the
        // options state the scheduler and the rank layout, and the model's files complete what they
        // determine. Its launch file carries it whole (§2.3).
        let engine = Leader::prepare(&target, &crate::options::Stated::from_paths(Vec::new()))
            .await
            .map_err(|error| UpdateError::Refused(format!("{error:#}")))?
            .engine;
        // Other model files serve the deployment's requests only when they are processed as the
        // serving ones are: the same tokenizer, templates, generation defaults and model facts.
        if current.model() != target.model() {
            crate::leader::same_processing(current.model(), target.model())
                .await
                .map_err(UpdateError::Refused)?;
            let served = locked(&replicas[0].served).clone();
            if served.is_some_and(|served| served.engine_config.model.facts != engine.model.facts) {
                return Err(UpdateError::Refused(format!(
                    "the model at {} has other generation limits, control tokens or context \
                     length than the served model, which change how requests are processed: \
                     deploy it instead",
                    target.model()
                )));
            }
        }
        let mut args = proto::config::EngineArgs::new(engine);
        args.world = target_epoch;
        // §7.4 step 1: the target's main processes start as the serving
        // process's standby — they map the predecessor's cache read-only and
        // park — so the storage of an epoch has one writer while both run.
        // The predecessor's segments are the ones it holds now, which only its
        // engine can state; without them the child would map nothing and the
        // launch would be a serving process's, claiming the same scopes
        // writably. The engine states as well the world that created them, which
        // is what they are keyed by, and not the target's own `args.world`: the
        // successor reads one world while it serves the next, an epoch that took
        // storage over keys it as its creator did, and one relaunched created
        // its own.
        args.predecessor = Some(serving_client.describe_cache().await.map_err(|reason| {
            UpdateError::Refused(format!(
                "the serving epoch named no cache segments for its standby: {reason}"
            ))
        })?);
        let joining = (!dry_run).then(|| args.clone());
        let change = crate::updates::epoch::EpochChange {
            previous: previous_epoch,
            target: target_epoch,
            // The update's number is the epoch it creates: epochs only grow.
            update: crate::epoch::Update::new(target_epoch, next.clone()),
            group: group.to_owned(),
            start: control
                .epoch_start(&args, spawn_generation)
                .map_err(UpdateError::Failed)?,
            previous_world: serving.world,
            previous_generation,
            spawn_generation,
            keeps: update.previous_world,
        };
        let programs = next.programs.clone();
        let servers = self.program_servers(&target);
        let leader = self.program_leader_host(&target, &servers);
        // A session does not cross epochs (`docs/profiling.md`): its ranks write their files while
        // the epoch still executes, before the switch parks it.
        if !dry_run {
            self.end_profiles("world change").await;
        }
        let applied = self.updates.apply(&previous, &next, change, dry_run);
        if let (Ok(_), Some(args)) = (&applied, joining) {
            self.updates.await_programs(&target.host_identity);
            replicas[0].follow_switch(
                Arc::clone(control),
                Arc::clone(&self.updates),
                crate::leader::SwitchTarget {
                    args,
                    config: target.clone(),
                    programs,
                    servers,
                    leader,
                    engine: spawn_generation,
                    epoch: target_epoch,
                    update: update.clone(),
                },
            );
        }
        applied
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

    /// Ends every replica's running profiling session, because its epoch stops executing.
    async fn end_profiles(&self, reason: &str) {
        for replica in self.replicas() {
            let client = replica.hold().client();
            crate::profiling::end_for_world(&replica, client.as_deref(), reason).await;
        }
    }

    /// Restores the previous epoch of the update in flight.
    pub async fn rollback(&self, reason: String) -> Result<serde_json::Value, UpdateError> {
        let _replacing = self.replicas.replacing.lock().await;
        if self.updates.running() {
            self.end_profiles("rollback").await;
            if self.updates.rollback(&reason) {
                return Ok(self
                    .updates
                    .report()
                    .unwrap_or_else(|| serde_json::json!({})));
            }
        }
        Err(UpdateError::Refused(
            "no update is in flight: the deployment already serves the configuration \
             the file states"
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
        // Each node's daemon reads its ledgers over its claim; an engine this daemon did not launch
        // (a simulated one) answers for itself.
        let control = replicas
            .iter()
            .find_map(|replica| replica.control.get().cloned());
        let ledgers = match client {
            Some(client) => {
                let read = async {
                    match control {
                        Some(control) => {
                            tokio::task::spawn_blocking(move || control.storage_reports())
                                .await
                                .map_err(|error| error.to_string())?
                                .map_err(|error| format!("{error:#}"))
                        }
                        None => client.describe_storage().await,
                    }
                };
                match tokio::time::timeout(LEDGER_READ, read).await {
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
        // An object even when no replica answers, so the fields below are
        // written into it rather than into a null.
        let mut status = statuses
            .first()
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        status["replicas"] = statuses.into();
        status["gpus"] = ledgers.as_array().map_or(serde_json::Value::Null, |nodes| {
            deployment_gpus(nodes).into()
        });
        status["ledgers"] = ledgers;
        // The update in flight, or the one that ended last, with its trial.
        status["update"] = self.updates.status().unwrap_or_default();
        // A revert's mark lives with the epoch layer's last update; the
        // replicas' own marks are the ones a handover carried in.
        if let Some(update) = self.updates.epochs().update()
            && let Some(failure) = update.unavailable()
        {
            let mut marks = status["unavailable"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            if !marks.iter().any(|mark| mark["reason"] == failure.reason) {
                marks.push(serde_json::json!({
                    "digest": digest(&update.target.config).unwrap_or_default(),
                    "update": update.number,
                    "reason": failure.reason,
                }));
            }
            status["unavailable"] = marks.into();
        }
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
            // A hold that closed names why it closed, and it names it to the
            // requests it fails: the status states the same reason rather
            // than that requests merely wait for an engine that will not come.
            None => match hold.state() {
                HoldState::Closed(reason) => unavailable(worlds.executing, &reason),
                _ => unavailable(worlds.executing, "requests wait for its engine"),
            },
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
    fn available(&self, target: &ServerArgs, force: bool) -> Result<String, UpdateError> {
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
        let (target, stated, _) = resolve(&current, &request)?;
        // A configuration a rollback marked unavailable stays refused unless
        // the change is forced.
        self.available(&target, request.force)?;
        if target == current && target.worker_groups.is_none() {
            return Ok(Applied::Unchanged);
        }
        let served = locked(&self.served).clone().ok_or(UpdateError::NoEngine)?;
        // The target's engine configuration is resolved as a deployment's
        // start resolves it: from the options, completed by what the model's
        // files determine, and divided into the deployment's replicas.
        let prepared = Leader::prepare(&target, &stated)
            .await
            .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
        let replicas = crate::leader::replica_configs(&prepared.engine)
            .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
        let mut target_engine = replicas.get(self.ordinal).cloned().ok_or_else(|| {
            UpdateError::Refused("the target changes the replica group".to_owned())
        })?;
        let explicit_devices = target.worker_groups.is_some()
            || (stated.contains("device_config.device") && target.device_config.device != "cuda")
            || !target.device_config.devices.is_empty();
        if !explicit_devices
            && target.tensor_parallel_size == current.tensor_parallel_size
            && target.data_parallel_size == current.data_parallel_size
            && target.device_config.worker_hosts == current.device_config.worker_hosts
            && target_engine.parallel.engine_host == served.engine_config.parallel.engine_host
        {
            retain_devices(&served.engine_config, &mut target_engine);
        }
        let running = self.hold().client().ok_or(UpdateError::NoEngine)?;
        let daemon_changes = crate::options::daemon_options_differ(&current, &target);
        let in_place = crate::leader::configurable(
            &comparable(&served.engine_config),
            &comparable(&target_engine),
            running.info().capacity.as_ref(),
        );
        // A successor serving other model files processes requests as the
        // daemon does only with the same tokenizer, templates, generation
        // defaults and model facts.
        let served_model = &served.engine_config.model;
        if served_model.model != target_engine.model.model {
            crate::leader::same_processing(&served_model.model, &target_engine.model.model)
                .await
                .map_err(UpdateError::Refused)?;
            if served_model.facts != target_engine.model.facts {
                return Err(UpdateError::Refused(format!(
                    "the model at {} has other generation limits, control tokens or context \
                     length than the served model, which change how requests are processed: \
                     deploy it instead",
                    target_engine.model.model
                )));
            }
        }
        // A dry run answers what the change would do and changes nothing: no
        // handover, no engine command, no configuration written back.
        if request.dry_run {
            return Ok(Applied::Planned(serde_json::json!({
                "daemon": daemon_changes,
                "scheduling": in_place,
                "fence_probes": switches_probes_only(&served.engine_config, &target_engine),
            })));
        }
        // The daemon orchestrates what changes in its own options: a hot
        // restart applies them first, and its successor applies the change
        // again, which then leaves the daemon's options as they are.
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
        if switches_probes_only(&served.engine_config, &target_engine) {
            let enabled = target_engine.profiling.fence_probes;
            let off = target_engine.profiling.fence_probes_off.clone();
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
            let mut args = served;
            args.engine_config = target_engine;
            *locked(&self.served) = Some(args);
            *locked(&self.config) = target;
            return Ok(Applied::InPlace(
                serde_json::json!({ "in_place": ["the fence probes"] }),
            ));
        }
        // A scheduling limit is a parameter of the policy program, so the program layer above has
        // moved the schedulers to it (§7.7) or found they run it already. What is left is the
        // record of it here: the arguments the next apply compares with.
        if in_place {
            let mut args = served;
            args.engine_config = target_engine;
            *locked(&self.served) = Some(args);
            *locked(&self.config) = target;
            return Ok(Applied::InPlace(
                serde_json::json!({ "in_place": ["the scheduling limits"] }),
            ));
        }
        // The layers above decide every change that reaches here: the epoch
        // layer owns what the main processes are built from (the executable,
        // the parallel ranks, the deployment file, the model's files, the
        // rank layout and the KV cache's options, all of them `proto::config::EngineConfig`),
        // the program layer what they execute, and this one the daemon's own
        // options and the scheduling limits, in place. A change that fits
        // none of them would have needed a successor world, which this
        // daemon no longer runs: it is refused by name here rather than
        // half served.
        Err(UpdateError::Refused(format!(
            "the change fits no layer: it would have needed a successor engine ({}) and the \
             epoch layer starts one main process from the build config",
            differing(&served.engine_config, &target_engine).join(", ")
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default deployment's engine configuration.
    fn config() -> EngineConfig {
        crate::options::ServerArgs::default()
            .engine_config()
            .expect("the configuration resolves")
    }

    /// The fence probes' switch, and a component's probes switched off, change in place; the
    /// probed components and PC sampling shape the captured graphs and take a new world.
    #[test]
    fn only_the_fence_probes_switch_changes_profiling_in_place() {
        let profiled = |profiling: proto::config::ProfilingConfig| EngineConfig {
            profiling,
            ..config()
        };
        let served = profiled(proto::config::ProfilingConfig {
            fence_probes: true,
            fence_components: vec!["*.Transformer".to_owned()],
            fence_probes_off: Vec::new(),
            graph_pc_sampling: false,
        });
        let switched = profiled(proto::config::ProfilingConfig {
            fence_probes: false,
            ..served.profiling.clone()
        });
        assert!(switches_probes_only(&served, &switched));
        // A component's probes switch off in place too.
        let thinned = profiled(proto::config::ProfilingConfig {
            fence_probes_off: vec!["*.Attention".to_owned()],
            ..served.profiling.clone()
        });
        assert!(switches_probes_only(&served, &thinned));
        for world in [
            proto::config::ProfilingConfig {
                graph_pc_sampling: true,
                ..served.profiling.clone()
            },
            proto::config::ProfilingConfig {
                fence_components: vec!["*.TransformerLayer".to_owned()],
                ..served.profiling.clone()
            },
        ] {
            assert!(!switches_probes_only(&served, &profiled(world)));
        }
    }

    /// A change of the scheduling limits alone, within the batch the workers were started for, is
    /// in place, and a value the program layer owns does not stand in its way: the program layer
    /// answered for it before the change reached the replica.
    #[test]
    fn a_scheduling_change_is_in_place_beside_what_the_program_layer_owns() {
        let mut served = config();
        served.scheduler.max_batch = std::num::NonZeroUsize::new(8).unwrap();
        served.max_running_requests = std::num::NonZeroUsize::new(8).unwrap();
        let mut target = served.clone();
        target.max_running_requests = std::num::NonZeroUsize::new(4).unwrap();
        target.attention_backend = "trtllm".to_owned();
        let capacity = proto::request::wire::BatchCapacity {
            max_batch: 8,
            max_num_batched_tokens: 8192,
            max_running_requests: 8,
        };
        assert!(crate::leader::configurable(
            &comparable(&served),
            &comparable(&target),
            Some(&capacity)
        ));
        assert!(!crate::leader::configurable(
            &served,
            &target,
            Some(&capacity)
        ));
    }

    /// The resolved deployment carries the ranks it read even when the
    /// file is changed in place before the engine starts.
    #[test]
    fn a_resolved_deployment_keeps_its_ranks_when_the_file_changes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("deployment.toml");
        let resolve = |ranks: &str| {
            std::fs::write(
                &path,
                format!(
                    "[[worker_groups]]\nid = \"model\"\nranks = {ranks}\nqueue_depth = 2\n\n\
                     [worker_groups.blocks.model]\ntensor_parallel_size = 2\n"
                ),
            )
            .unwrap();
            crate::options::ServerArgs {
                worker_groups: Some(path.clone()),
                ..Default::default()
            }
            .engine_config()
            .unwrap()
        };
        let served = resolve(r#"["rank-0/cuda:0", "rank-0/cuda:1"]"#);
        let moved = resolve(r#"["rank-1/cuda:0", "rank-1/cuda:1"]"#);
        std::fs::remove_file(path).unwrap();
        for (resolved, host) in [(served, "rank-0"), (moved, "rank-1")] {
            let ranks = &resolved.parallel.worker_groups[0].ranks;
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
        use storage::protocol::Priority::Background;

        let mut worker = engine::SimWorker::new(engine::SimEngine::new());
        let nodes = worker.report_ledgers();
        let core = engine::SchedulerHandle::with_worker(
            &engine::sim_args(crate::test_support::SERVED_MODEL),
            engine::SchedulerConfig::default(),
            Box::new(worker),
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
