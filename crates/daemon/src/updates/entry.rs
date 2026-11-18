//! The update entry: one change of the deployment's configuration, executed as its declared
//! effects (`specs/configuration.md` §7).
//!
//! The target is a whole configuration, compared with the current one option by option
//! ([`proto::reload::changes`]) and merged into the effects each component executes
//! ([`proto::reload::Effects`]):
//!
//! - the daemon's own options: a hot restart of the daemon, whose successor applies what else
//!   the change states;
//! - an epoch (the engine package, the build configuration, the checkpoint) or a replaced
//!   Storage Server: an update in its epoch form, target engines started beside the serving
//!   ones;
//! - kernel or program modules: an update in its generation form, inside the serving engines,
//!   once the target's module set is fixed at the Build Servers, so a revision installed since
//!   the serving epoch's apply is a change too;
//! - step options: in place, at the engines' next step.
//!
//! `Leader::apply` decides in that order and starts the update, which commits by itself once
//! its trial lasted `update_config.trial`. The update itself is
//! `crate::updates::update::Update`, advanced by the engines' acknowledgements; what the daemon
//! does beside it for a replica (moving requests to the engine that serves, reading the trial's
//! thresholds, committing at the trial's end) is [`Replica::follow`].

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use proto::admin::{
    ConfigStatus, Executing, ExecutingState, Gpu, GpuShare, GpuUuid, InFlight, InPlaceChange,
    NodeUsage, Quotas, ReplicaStatus, Switched,
};
use proto::config::EngineConfig;
use proto::reload::{Effect, Effects, Reload as _};
use tracing::info;

use crate::config::{Config, Stated};
use crate::engine::client::EngineState;
use crate::engine::hold::HoldState;
use crate::epoch::{Change, Epoch};
use crate::leader::{Leader, Replica, UpdateError};
use crate::updates::change::Applied;
use crate::updates::update::{EpochSwitch, Node, Update};

/// The GPUs the deployment's epochs hold storage on, from every node's storage report.
fn deployment_gpus(nodes: &[NodeUsage]) -> Vec<GpuShare> {
    nodes
        .iter()
        .flat_map(|node| {
            node.quotas
                .iter()
                .filter(|quota| quota.deployment.is_some())
                .map(|quota| GpuShare {
                    gpu: Gpu {
                        host: node.host.clone(),
                        gpu: GpuUuid(quota.gpu),
                    },
                    total: quota.total,
                    quota: quota.limit,
                    used: quota.usage(),
                })
        })
        .collect()
}

/// The options the workers switch in place, at the engine's next step boundary: the fence
/// probes' switch and the classes whose probes stay off. Every other profiling option shapes the
/// CUDA graphs an epoch captures.
const FENCE_SWITCHES: [&str; 2] = [
    "profiling_config.fence_probes",
    "profiling_config.fence_probes_off",
];

/// Whether `changes` are the fence probes' switches alone.
fn switches_probes_only(changes: &[proto::reload::Change]) -> bool {
    !changes.is_empty()
        && changes
            .iter()
            .all(|change| FENCE_SWITCHES.contains(&change.path.as_str()))
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

impl Leader {
    /// Changes the deployment to `target`, whose `config.toml` states `stated`, as operation
    /// `number`: the least its declared effects need (module documentation).
    ///
    /// An update it starts runs on after this returns: its phases end on the engines'
    /// acknowledgements, it commits by itself at its trial's end, and `GET /admin/status` and
    /// the events report it.
    ///
    /// # Errors
    ///
    /// `UpdateError::InProgress` while another change or update runs;
    /// `UpdateError::Refused` for a target a rollback marked unavailable unless `force`, a module
    /// set the Build Servers cannot select, or a change no form moves.
    pub(crate) async fn apply(
        &self,
        target: Config,
        stated: Stated,
        force: bool,
        number: u64,
    ) -> Result<Applied, UpdateError> {
        let _replacing = self
            .replicas
            .replacing
            .try_lock()
            .map_err(|_| UpdateError::InProgress)?;
        let replicas = self.replicas();
        if self.updates.in_flight().is_some() {
            return Err(UpdateError::InProgress);
        }
        let current = service::lock(&replicas[0].config).clone();
        if target.served_model_name() != current.served_model_name() {
            return Err(UpdateError::Refused(
                "the served model name names the deployment's storage: serve another package \
                 instead"
                    .into(),
            ));
        }
        // The one selection of an apply: every interface's revision is fixed at the Build
        // Servers before anything is compared, so a revision installed since the serving epoch
        // was applied is a change. The daemon's own options change first, by a hot restart whose
        // successor applies the change again, so nothing is selected for them.
        let named = crate::epoch::change(
            &crate::leader::deployment(&current)?,
            &crate::leader::deployment(&target)?,
        );
        if named == Change::Host {
            return self.apply_in_place(target, stated).await;
        }
        let mut next = crate::leader::deployment(&target)?;
        next.modules = self.fix_modules(&target).await?;
        let Some((serving, record)) = self.updates.epochs().serving() else {
            return self.apply_in_place(target, stated).await;
        };
        let form = if named == Change::Epoch || record.engine != next.modules.engine {
            Effect::Epoch
        } else if record.kernels != next.modules.kernels || record.policy != next.modules.policy {
            Effect::Generation
        } else {
            return self.apply_in_place(target, stated).await;
        };
        if !force && let Some(mark) = self.updates.unavailable(&next) {
            return Err(UpdateError::Refused(format!(
                "the configuration was rolled back from epoch {} ({}); apply it again with force",
                mark.epoch, mark.reason
            )));
        }
        let options = target.update_config.clone();
        if form == Effect::Epoch {
            Box::pin(self.epoch_update(number, &current, &target, next, serving, &options)).await
        } else {
            self.generation_update(number, next, serving, &options)
                .await
        }
    }

    /// The per-replica path of a change that starts no update: the daemon's own options (a
    /// hot restart), the options the engines take at their next step, or nothing.
    async fn apply_in_place(&self, target: Config, stated: Stated) -> Result<Applied, UpdateError> {
        let mut changed = Vec::new();
        for replica in self.replicas() {
            match Box::pin(replica.apply(target.clone(), stated.clone())).await? {
                applied @ (Applied::Daemon { .. } | Applied::Update(_)) => return Ok(applied),
                Applied::InPlace(changes) => changed.extend(changes),
                Applied::Unchanged => {}
            }
        }
        Ok(if changed.is_empty() {
            Applied::Unchanged
        } else {
            Applied::InPlace(changed)
        })
    }

    /// Fixes the module set `target` runs: every interface's revision, as
    /// the deployment's Build Server on this node selects it with `cover`, with every module built
    /// on every host `target`'s processes run on before anything starts from
    /// it.
    async fn fix_modules(&self, target: &Config) -> Result<crate::epoch::ModuleSet, UpdateError> {
        let named = crate::epoch::modules_of(target);
        // A daemon that starts no engine of its own has no Build Server to
        // select at, no daemons of other hosts to reach, and no engine to move.
        let control = Arc::clone(
            self.replicas()[0]
                .control
                .get()
                .ok_or(UpdateError::NoEngine)?,
        );
        let hosts = target
            .engine_config()
            .and_then(|engine| crate::leader::epoch_hosts(&engine))
            .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
        let local = crate::engine::control::BuildServer {
            host: control.host().to_owned(),
            socket: control.build_server().to_owned(),
            targets: Vec::new(),
        };
        crate::leader::fix_modules(local, Some(control), hosts, named).await
    }

    /// Starts an update in its generation form: the serving engines prepare
    /// `next`'s kernel and policy modules and switch to them.
    async fn generation_update(
        &self,
        number: u64,
        next: Epoch,
        serving: crate::epoch::Recorded,
        options: &crate::config::UpdateConfig,
    ) -> Result<Applied, UpdateError> {
        let nodes = self.serving_nodes()?;
        if nodes.is_empty() {
            return Err(UpdateError::NoEngine);
        }
        let update = Update::generation(
            number,
            next.clone(),
            Arc::clone(self.updates.epochs()),
            serving,
            crate::engine::control::ENGINE_GROUP.to_owned(),
            nodes,
            options.trial,
        );
        let report = self.updates.apply(update)?;
        for replica in self.replicas() {
            replica.follow(
                Arc::clone(&self.updates),
                number,
                options.clone(),
                crate::leader::Following::Generation {
                    config: Box::new(next.config.clone()),
                    modules: next.modules.clone(),
                },
            );
        }
        Ok(Applied::Update(Box::new(report)))
    }

    /// The serving engines, one node each: the processes a generation update
    /// commands, by the launch generation their reports state, each with the world's first rank
    /// on its host and the launch of the world's scheduler node, which holds the world's first
    /// rank.
    ///
    /// # Errors
    ///
    /// Refuses a replica whose configuration does not place its world, an engine on a host its
    /// world places no rank on, or a world whose first rank no engine holds.
    fn serving_nodes(&self) -> Result<Vec<Node>, UpdateError> {
        let mut nodes: Vec<Node> = Vec::new();
        for replica in self.replicas() {
            let Some(client) = replica.hold().client() else {
                continue;
            };
            let config = service::lock(&replica.config)
                .engine_config()
                .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
            // The layout the replica serves in, from what its engines were launched with.
            let layout = service::lock(&replica.served)
                .as_ref()
                .map(|served| served.engine_config.layout())
                .ok_or_else(|| {
                    UpdateError::Refused("a replica with an engine serves no launch".to_owned())
                })?;
            let ranks = crate::engine::control::world_hosts(&config)
                .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
            let first_of = |host: &str| {
                ranks
                    .iter()
                    .find(|ranks| ranks.host == host)
                    .map(|ranks| ranks.first)
                    .ok_or_else(|| {
                        UpdateError::Refused(format!(
                            "an engine serves on {host}, where its world places no rank"
                        ))
                    })
            };
            let processes = client.processes();
            let mut placed = Vec::with_capacity(processes.len());
            for process in processes {
                placed.push((first_of(&process.host)?, process));
            }
            let head = placed
                .iter()
                .find(|(first, _)| *first == 0)
                .map(|(_, process)| process.engine)
                .ok_or_else(|| {
                    UpdateError::Refused("no engine holds its world's first rank".to_owned())
                })?;
            for (rank, process) in placed {
                let scopes = crate::engine::control::host_scopes(&config, &process.host);
                nodes.push(Node::new(
                    process.host,
                    rank,
                    process.engine,
                    head,
                    scopes,
                    layout.clone(),
                ));
            }
        }
        Ok(nodes)
    }

    /// Starts an update in its epoch form: `next`'s engines start beside the
    /// serving ones as parked engines of the target epoch, on every node and
    /// for every replica the target's configuration places, and service moves
    /// to them.
    ///
    /// Every target engine starts from a launch file of its own, which states
    /// the epoch's whole module set, the engine it succeeds, its own launch
    /// generation (above every rank's standby) and, for a world of several
    /// hosts, the sockets its scheduler node reserved.
    #[allow(clippy::too_many_arguments)]
    async fn epoch_update(
        &self,
        number: u64,
        current: &Config,
        target: &Config,
        next: Epoch,
        serving: crate::epoch::Recorded,
        options: &crate::config::UpdateConfig,
    ) -> Result<Applied, UpdateError> {
        let refused = |error: anyhow::Error| UpdateError::Refused(format!("{error:#}"));
        let replicas = self.replicas();
        let control = replicas[0].control.get().ok_or(UpdateError::NoEngine)?;
        // Every engine process that serves, of every replica and on every
        // node: each is drained by its launch generation.
        let nodes = self.serving_nodes()?;
        if nodes.is_empty() {
            return Err(UpdateError::Refused(format!(
                "no engine serving epoch {} has a launch generation: this daemon neither \
                 started nor adopted one",
                serving.number()
            )));
        }
        let epochs = self.updates.epochs();
        // The target's number is above every epoch the daemon knows: an epoch
        // a rollback retired keeps its number.
        let target_epoch = epochs
            .records()
            .keys()
            .copied()
            .max()
            // The serving epoch is recorded, so the records hold a number.
            .unwrap_or(serving.number())
            + 1;

        // The target's engine configuration is resolved as a deployment's
        // start resolves it: the options state the scheduler and the rank
        // layout, and the model's files complete what they determine.
        let engine = Leader::prepare(
            &self.families,
            target,
            &crate::config::Stated::from_paths(Vec::new()),
        )
        .await
        .map_err(refused)?
        .engine;
        // Other model files serve the deployment's requests only when they
        // are processed as the serving ones are.
        if current.model() != target.model() {
            let served = service::lock(&replicas[0].served).clone();
            if let Some(served) = &served {
                crate::leader::same_processing(
                    &self.families,
                    &served.engine_config.model.model,
                    &engine.model.model,
                )
                .await
                .map_err(UpdateError::Refused)?;
            }
            if served.is_some_and(|served| !served.engine_config.model.processes_as(&engine.model))
            {
                return Err(UpdateError::Refused(format!(
                    "the model at {} has other generation limits, control tokens or context \
                     length than the served model, which change how requests are processed: \
                     deploy it instead",
                    target.model()
                )));
            }
        }

        // One launch number per engine of every target replica's world, from the deployment's
        // one allocator, its engine control: none collides with a standby's or with any other
        // engine of the deployment.
        let launches =
            crate::leader::epoch_launches(&engine, &next.modules, target_epoch).map_err(refused)?;
        let configs: Vec<_> = launches
            .iter()
            .map(|args| args.engine_config.clone())
            .collect();
        let mut worlds = Vec::with_capacity(launches.len());
        for args in launches {
            let hosts =
                crate::engine::control::world_hosts(&args.engine_config).map_err(refused)?;
            let numbers: Vec<u64> = hosts.iter().map(|_| control.launch_number()).collect();
            worlds.push((args, hosts, numbers));
        }
        // A replica's layout (`EngineConfig::layout`: its parallel configuration with its devices
        // and hosts, slot count, KV layout and graph buckets, not the cache's size or tiers) names
        // its device regions. A serving replica whose layout a target replica shares keeps its
        // regions and their segments, which that target maps, through its drain; one whose layout
        // no target shares gives its memory back (`shrink`), and the targets of layouts of their
        // own grow into it (`Node::moves_across`). A change of the replica count alone keeps the
        // layout of every replica it leaves in place.
        let target_layouts: Vec<_> = configs.iter().map(EngineConfig::layout).collect();
        // Each target world's launch files. A target starts parked: its launch
        // names the epoch it succeeds.
        let mut launches = Vec::new();
        let mut targets = Vec::with_capacity(worlds.len());
        for (args, _, numbers) in worlds {
            let mut parked = args.clone();
            parked.predecessor = Some(proto::request::wire::Predecessor {
                epoch: serving.number(),
            });
            let (placing, numbering) = (Arc::clone(control), numbers.clone());
            let world =
                tokio::task::spawn_blocking(move || placing.target_world(&parked, &numbering))
                    .await
                    .map_err(|error| UpdateError::Failed(anyhow::anyhow!("{error}")))?
                    .map_err(UpdateError::Failed)?;
            let processes = world
                .iter()
                .map(|(launch, _)| (launch.host.clone(), launch.engine))
                .collect();
            launches.extend(world);
            targets.push((args, processes));
        }

        #[expect(
            clippy::expect_used,
            reason = "a `Recorded` epoch's record exists: no record is ever removed"
        )]
        let serving_record = epochs
            .get(serving.number())
            .expect("a recorded epoch's record is never removed");
        let mut record = serving_record;
        record.config = next.engine.clone();
        record.engine = next.modules.engine.clone();
        record.kernels = next.modules.kernels.clone();
        record.policy = next.modules.policy.clone();
        record.previous = None;
        // The target's generation is set only by its own engines' reports.
        record.generation = None;

        let update = Update::epoch(
            number,
            next.clone(),
            Arc::clone(epochs),
            serving,
            crate::engine::control::ENGINE_GROUP.to_owned(),
            nodes,
            EpochSwitch {
                epoch: target_epoch,
                record,
                launches,
                keeps: options.previous_epoch,
                recompute: options.in_flight == InFlight::Recompute,
                drain: Duration::from_secs_f64(options.drain_s),
                target_layouts,
                retire: self.updates.take_retire(),
            },
            options.trial,
        );
        // A profiling session does not cross epochs: its ranks write their
        // files while the epoch still executes, before the switch drains it.
        self.end_profiles("epoch change").await;
        let report = self.updates.apply(update)?;
        replicas[0].follow(
            Arc::clone(&self.updates),
            number,
            options.clone(),
            crate::leader::Following::Epoch(crate::leader::EpochTarget {
                control: Arc::clone(control),
                targets,
                config: Box::new(target.clone()),
                epoch: target_epoch,
                keep: options.in_flight == InFlight::Keep,
            }),
        );
        Ok(Applied::Update(Box::new(report)))
    }

    /// Ends every replica's running profiling session, because its epoch stops executing.
    async fn end_profiles(&self, reason: &str) {
        for replica in self.replicas() {
            let client = replica.hold().client();
            crate::profiling::end_for_epoch(&replica, client.as_deref(), reason).await;
        }
    }

    /// Reports the configuration and resource ownership of the replicas.
    ///
    /// # Errors
    ///
    /// Fails when a replica's configuration cannot be printed.
    pub(crate) async fn config_status(&self) -> anyhow::Result<ConfigStatus> {
        let replicas = self.replicas();
        let client = replicas
            .iter()
            .filter_map(|replica| replica.hold().client())
            .find(|client| matches!(client.state(), EngineState::Serving | EngineState::Parked));
        // Each node's daemon reads its quotas over its claim.
        let control = replicas
            .iter()
            .find_map(|replica| replica.control.get().cloned());
        let quotas = match (client, control) {
            (Some(_), Some(control)) => {
                let read = async {
                    tokio::task::spawn_blocking(move || control.storage_reports())
                        .await
                        .map_err(|error| error.to_string())?
                        .map_err(|error| format!("{error:#}"))
                };
                // Each node's daemon answers within the process interface's own bound.
                Some(match read.await {
                    Ok(nodes) => Quotas::Read(nodes),
                    Err(error) => Quotas::Unread { error },
                })
            }
            _ => None,
        };
        // The bytes a change in progress moved: what every node's Transfer
        // Server copied for the deployment since it began, read with the
        // quotas. One change runs at a time, so every transfer since is
        // the change's.
        let replicas = replicas
            .into_iter()
            .map(|replica| replica.config_status())
            .collect::<anyhow::Result<Vec<_>>>()?;
        let gpus = match &quotas {
            Some(Quotas::Read(nodes)) => Some(deployment_gpus(nodes)),
            Some(Quotas::Unread { .. }) | None => None,
        };
        Ok(ConfigStatus {
            first: replicas.first().cloned(),
            replicas,
            gpus,
            quotas,
            // The update in flight, or the one that ended last, with its trial.
            update: self.updates.status(),
            unavailable: self.updates.unavailable_marks(),
            // The epochs and the processes this daemon manages sit beside the configuration:
            // `EpochRecord`s are the daemon's record of what it executes, not the engine's
            // (`crate::epoch`).
            epochs: self.updates.epochs().statuses(),
            processes: self.updates.epochs().processes(),
        })
    }
}

impl Replica {
    /// Reports what the replica executes and what its configuration is: the
    /// engine that takes requests with its state, and the digest of the
    /// configuration.
    ///
    /// # Errors
    ///
    /// Fails when the configuration cannot be printed.
    pub(crate) fn config_status(&self) -> anyhow::Result<ReplicaStatus> {
        let config = service::lock(&self.config).clone();
        let printed = crate::config::printed(&config)?;
        let digest = crate::updates::digest_of(&printed);
        let hold = self.hold();
        let slot = |epoch: u64, state: ExecutingState| Executing {
            epoch,
            state,
            reason: None,
        };
        let unavailable = |epoch: u64, reason: &str| Executing {
            epoch,
            state: ExecutingState::Unavailable,
            reason: Some(reason.to_owned()),
        };
        // The epoch a replica without an engine last executed is the one its
        // served arguments name.
        let served_epoch = service::lock(&self.served)
            .as_ref()
            .map_or(0, |args| args.epoch);
        let executing = match hold.client() {
            Some(client) => {
                let epoch = client.epoch();
                match client.state() {
                    EngineState::Starting => slot(epoch, ExecutingState::Starting),
                    EngineState::Parked => slot(epoch, ExecutingState::Parked),
                    EngineState::Serving => slot(epoch, ExecutingState::Serving),
                    EngineState::Draining => slot(epoch, ExecutingState::Draining),
                    EngineState::Failed(reason) => unavailable(epoch, &reason),
                    EngineState::Stopped => unavailable(epoch, "its engine process ended"),
                }
            }
            // A hold that closed names why it closed, and it names it to the
            // requests it fails: the status states the same reason rather
            // than that requests merely wait for an engine that will not come.
            None => match hold.state() {
                HoldState::Closed(reason) => unavailable(served_epoch, &reason),
                _ => unavailable(served_epoch, "requests wait for its engine"),
            },
        };
        Ok(ReplicaStatus {
            config: printed,
            digest,
            executing,
        })
    }

    /// Applies what of `target` no update moves: nothing for the current configuration; the
    /// daemon's own options, by a hot restart; the fence probes' switch, which the workers make
    /// at their next step; and the record of the other options that take effect at a step, which
    /// the engines' scheduling module received with its generation.
    ///
    /// # Errors
    ///
    /// Refuses another change in progress (`409`), and a target whose model files process
    /// requests otherwise than the served ones.
    pub(crate) async fn apply(
        self: &Arc<Self>,
        target: Config,
        stated: Stated,
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
        let current = service::lock(&self.config).clone();
        let changes = proto::reload::changes(Some(&current), &target);
        if changes.is_empty() {
            return Ok(Applied::Unchanged);
        }
        // The daemon orchestrates what changes in its own options: a hot restart applies them
        // first, and its successor applies the rest of the change.
        if Effects::of(&changes).daemon {
            let mut successor = current;
            successor.take(&target, Effect::Daemon);
            return Ok(Applied::Daemon {
                config: Box::new(successor),
            });
        }
        let served = service::lock(&self.served)
            .clone()
            .ok_or(UpdateError::NoEngine)?;
        // The target's engine configuration is resolved as a deployment's start resolves it:
        // from the options, completed by what the model's files determine, and divided into the
        // deployment's replicas.
        let prepared = Leader::prepare(&self.families, &target, &stated)
            .await
            .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
        let replicas = crate::leader::replica_configs(&prepared.engine)
            .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
        let mut target_engine = replicas.get(self.ordinal).cloned().ok_or_else(|| {
            UpdateError::Refused("the target changes the replica group".to_owned())
        })?;
        let explicit_devices = target.worker_groups.is_some()
            || (stated.contains("device_config.device")
                && target.device_config.device != proto::config::DeviceKind::Cuda)
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
        // The fence probes' switch changes in place: every worker writes its device switch at
        // the engine's next step boundary, and the graphs that captured the probes stay.
        let switched = if switches_probes_only(&changes) {
            let enabled = target_engine.profiling.fence_probes;
            let off = target_engine.profiling.fence_probes_off.clone();
            running
                .switch_fences(enabled, off.clone())
                .await
                .map_err(|error| {
                    anyhow::anyhow!("the workers did not switch the probes: {error}")
                })?;
            info!(
                enabled,
                ?off,
                "the workers switched the fence probes in place"
            );
            Switched::FenceProbes
        } else if changes.iter().all(|change| change.effect == Effect::Step) {
            // The scheduling limits travel with the scheduling module's generation, which moved
            // the schedulers before this path: what is left is the record of them.
            Switched::SchedulingLimits
        } else {
            // Every change an update moves was decided before this path.
            let paths: Vec<_> = changes.iter().map(|change| change.path.as_str()).collect();
            return Err(UpdateError::Refused(format!(
                "no form of update moves the change of {}",
                paths.join(", ")
            )));
        };
        let mut args = served;
        args.engine_config = target_engine;
        *service::lock(&self.served) = Some(args);
        *service::lock(&self.config) = target;
        Ok(Applied::InPlace(vec![InPlaceChange {
            replica: self.ordinal,
            switched,
        }]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fence probes' switch, and a module's probes switched off, change in place; the
    /// probed modules and PC sampling shape the captured graphs and take a new epoch.
    #[test]
    fn only_the_fence_probes_switch_changes_profiling_in_place() {
        let profiled = |profiling: proto::config::ProfilingConfig| Config {
            profiling_config: profiling,
            ..Config::default()
        };
        let served = profiled(proto::config::ProfilingConfig {
            fence_probes: true,
            fence_modules: Some(vec!["*.Transformer".to_owned()]),
            fence_probes_off: Vec::new(),
            graph_pc_sampling: false,
            nvtx_ranges: false,
        });
        let changed = |target: &Config| proto::reload::changes(Some(&served), target);
        let switched = profiled(proto::config::ProfilingConfig {
            fence_probes: false,
            ..served.profiling_config.clone()
        });
        assert!(switches_probes_only(&changed(&switched)));
        // A module's probes switch off in place too.
        let thinned = profiled(proto::config::ProfilingConfig {
            fence_probes_off: vec!["*.Attention".to_owned()],
            ..served.profiling_config.clone()
        });
        assert!(switches_probes_only(&changed(&thinned)));
        for epoch in [
            proto::config::ProfilingConfig {
                graph_pc_sampling: true,
                ..served.profiling_config.clone()
            },
            proto::config::ProfilingConfig {
                fence_modules: Some(vec!["*.TransformerLayer".to_owned()]),
                ..served.profiling_config.clone()
            },
        ] {
            let changes = changed(&profiled(epoch));
            assert!(!switches_probes_only(&changes));
            assert_eq!(Effects::of(&changes).heaviest(), Some(Effect::Epoch));
        }
        assert!(!switches_probes_only(&[]));
    }

    /// The target's layout differs from the serving one's in its parallel
    /// configuration, its devices, its slot count, its KV layout and its graph
    /// buckets, and not in the cache's size or its tiers.
    #[test]
    fn a_layout_is_the_parallel_configuration_slots_kv_layout_and_graph_buckets() {
        let served = Config::default()
            .engine_config()
            .expect("the configuration resolves");
        let same = |change: &dyn Fn(&mut EngineConfig)| {
            let mut target = served.clone();
            change(&mut target);
            served.layout() == target.layout()
        };
        assert!(same(&|_| {}));
        assert!(same(
            &|target| target.kv_cache.max_tokens = std::num::NonZeroU64::new(4096)
        ));
        assert!(same(
            &|target| target.kv_cache.host_tier = !target.kv_cache.host_tier
        ));
        assert!(!same(&|target| target.parallel.worker_groups[0].ranks[0]
            .device =
            proto::config::RankDevice::cuda(3)));
        assert!(!same(
            &|target| target.max_running_requests = std::num::NonZeroUsize::new(3).unwrap()
        ));
        assert!(!same(
            &|target| target.kv_cache.block_size = std::num::NonZeroU32::new(32)
        ));
        assert!(!same(&|target| {
            target.cuda_graph.decode_batch_sizes =
                Some(vec![std::num::NonZeroU32::new(2).unwrap()]);
        }));
    }

    /// The status names the epoch that executes; the engine reads no storage
    /// quotas (each node's daemon reads them over its claim), so it reserves no
    /// GPUs; and an epoch whose engine is gone is unavailable.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_status_reports_the_executing_engine_and_one_that_is_gone() {
        let leader = crate::test_support::served_engine(crate::test_support::CHAT);
        let replica = Arc::clone(&leader.replicas()[0]);
        let client = replica.hold().client().expect("the replica serves");
        let epoch = client.epoch();
        // The replica serves the arguments its engine was launched with, as a deployment's
        // start records them.
        *service::lock(&replica.served) = Some(testkit::args(crate::test_support::SERVED_MODEL));

        let status = leader.config_status().await.unwrap();
        let executing = |status: &ConfigStatus| status.first.as_ref().unwrap().executing.clone();
        assert_eq!(
            executing(&status),
            Executing {
                epoch,
                state: ExecutingState::Serving,
                reason: None,
            }
        );
        // The engine refuses the quota read, so the status states no GPUs.
        assert_eq!(status.gpus, None);

        // An epoch whose engine no longer serves is unavailable.
        replica.hold().close("the engine process ended");
        let status = leader.config_status().await.unwrap();
        assert_eq!(
            executing(&status),
            Executing {
                epoch,
                state: ExecutingState::Unavailable,
                reason: Some("the engine process ended".to_owned()),
            }
        );
        client.shutdown();
    }
}
