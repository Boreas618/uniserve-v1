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

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use proto::admin::InFlight;
use proto::config::EngineConfig;
use proto::reload::{Effect, Effects, Reload as _};
use tracing::info;

use crate::config::{Config, Stated};
use crate::epoch::{Change, Epoch};
use crate::leader::{Leader, Replica, UpdateError};
use crate::updates::change::Applied;
use crate::updates::update::{EpochSwitch, Node, Update};

/// The options the workers switch in place, at the engine's next step boundary: the fence
/// probes' switch, the classes whose probes stay off and the kernel probes' switch. The other
/// profiling options of the engines shape the CUDA graphs they capture.
const FENCE_SWITCHES: [&str; 3] = [
    "profiling_config.fence_probes",
    "profiling_config.fence_probes_off",
    "profiling_config.kernel_probes",
];

/// The options that take effect at a step which the daemon applies itself, rather than the
/// schedulers reading them from their next step's snapshot: the deployment's name, admission and
/// sleep, its prefix cache's flush, its adapters, its profiling session, how far back a read of
/// its spans reaches, and what it does to its storage when it comes down.
const DAEMON_STEPS: [&str; 8] = [
    "served_model_name",
    "paused",
    "asleep",
    "flush",
    "retire",
    "adapters",
    "profiling_config.session",
    "profiling_config.span_window",
];

/// Whether `target` places the kernel probes elsewhere than `serving`: a kernel generation,
/// whose captures move them.
fn probes_move(serving: &EngineConfig, target: &EngineConfig) -> bool {
    serving.profiling.kernel_instance != target.profiling.kernel_instance
}

/// Whether `changes` switch the fence probes.
fn switches_probes(changes: &[proto::reload::Change]) -> bool {
    changes
        .iter()
        .any(|change| FENCE_SWITCHES.contains(&change.path.as_str()))
}

/// Whether `changes` move a step value the schedulers read from their next step's snapshot.
fn tunes(changes: &[proto::reload::Change]) -> bool {
    changes.iter().any(|change| {
        change.effect == Effect::Step
            && !FENCE_SWITCHES.contains(&change.path.as_str())
            && !DAEMON_STEPS.contains(&change.path.as_str())
    })
}

/// The serving engines of `replicas`, one node each ([`Leader::serving_nodes`]), each executing
/// the epoch its replica serves.
///
/// # Errors
///
/// As [`Leader::serving_nodes`].
fn serving_nodes_of(replicas: &[Arc<Replica>]) -> Result<Vec<Node>, UpdateError> {
    let mut nodes: Vec<Node> = Vec::new();
    for replica in replicas {
        let Some(client) = replica.hold().client() else {
            continue;
        };
        let config = service::lock(&replica.config)
            .engine_config()
            .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
        // The layout the replica serves in and its epoch, from what its engines were launched
        // with.
        let (layout, epoch) = service::lock(&replica.served)
            .as_ref()
            .map(|served| (served.engine_config.layout(), served.epoch))
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
                epoch,
            ));
        }
    }
    Ok(nodes)
}

/// The indices of the replicas an epoch change to `targets`, the launch of each target replica,
/// running `modules`, moves to a new epoch: every replica when the change gives the deployment
/// another number of replicas, and otherwise each replica whose launch the change touches: one
/// that serves no launch, runs another engine package, or whose build part
/// ([`EngineConfig::build_part`]) the target's differs from. A change that touches none, a
/// change of the module set's build alone, moves them all.
fn moved_replicas(
    replicas: &[Arc<Replica>],
    targets: &[proto::config::EngineArgs],
    modules: &crate::epoch::ModuleSet,
) -> Vec<usize> {
    let every = (0..targets.len()).collect();
    if targets.len() != replicas.len() {
        return every;
    }
    let moved: Vec<usize> = replicas
        .iter()
        .zip(targets)
        .enumerate()
        .filter(|(_, (replica, target))| {
            let served = service::lock(&replica.served);
            served.as_ref().is_none_or(|served| {
                served.modules.engine != modules.engine
                    || served.engine_config.build_part() != target.engine_config.build_part()
            })
        })
        .map(|(index, _)| index)
        .collect();
    if moved.is_empty() { every } else { moved }
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
    /// `number`, with the package's `programs` as its apply fixed them: the least its declared
    /// effects need (module documentation). The scheduling program among the programs is the
    /// engines' policy; a change of the others' alone moves no engine.
    ///
    /// An update it starts runs on after this returns: its phases end on the engines'
    /// acknowledgements, it commits by itself at its trial's end, and its `log` events of
    /// source `update` report each phase it enters.
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
        revision: u64,
        programs: BTreeMap<String, proto::module::Program>,
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
        // The one selection of an apply: every interface's revision is fixed at the Build
        // Servers before anything is compared, so a revision installed since the serving epoch
        // was applied is a change. The daemon's own options change first, by a hot restart whose
        // successor applies the change again, so nothing is selected for them.
        let named = crate::epoch::change(
            &crate::leader::deployment(&current)?,
            &crate::leader::deployment(&target)?,
        );
        if named == Change::Host {
            return self.apply_in_place(target, stated, revision).await;
        }
        let mut next = crate::leader::deployment(&target)?;
        next.modules = self.fix_modules(&target, programs).await?;
        let Some((serving, record)) = self.updates.epochs().serving() else {
            return self.apply_in_place(target, stated, revision).await;
        };
        let form = if named == Change::Epoch || record.engine != next.modules.engine {
            Effect::Epoch
        } else if named == Change::Generation
            || record.kernels != next.modules.kernels
            || record.policy != next.modules.policy
            || probes_move(&record.config, &next.engine)
        {
            Effect::Generation
        } else {
            return self.apply_in_place(target, stated, revision).await;
        };
        if !force && let Some(mark) = self.updates.unavailable(&next) {
            return Err(UpdateError::Refused(format!(
                "the configuration was rolled back from epoch {} ({}); apply it again with force",
                mark.epoch, mark.reason
            )));
        }
        let options = target.update_config.clone();
        if form == Effect::Epoch {
            Box::pin(self.epoch_update(number, &target, next, serving, &options, revision)).await
        } else {
            // The generation switches bucket by bucket and includes no step: the change's step
            // and order options take effect in place first, with the options the graphs are
            // captured with kept until the generation commits, which takes the whole target.
            let mut lighter = target.clone();
            lighter.take(&current, Effect::Generation);
            Box::pin(self.apply_in_place(lighter, stated, revision)).await?;
            self.generation_update(number, next, serving, &options)
                .await
        }
    }

    /// What a generation update to `target`'s kernels would take on every serving engine's
    /// devices beside the graphs they serve: the kernel packages `target` names fixed at this
    /// node's Build Server as an apply fixes them (`with_store_modules`, nothing built), and each
    /// replica's engine measuring the graphs the packages that differ from the serving epoch's
    /// would capture again (`MeasureRoom`). Empty when the kernels are the serving epoch's, or
    /// no engine serves.
    ///
    /// # Errors
    ///
    /// The Build Server's refusal of `target`'s pins, or an engine's failure to measure.
    pub(crate) async fn rooms(
        &self,
        target: &Config,
    ) -> Result<Vec<proto::request::wire::DeviceRoom>, UpdateError> {
        let Some((_, record)) = self.updates.epochs().serving() else {
            return Ok(Vec::new());
        };
        let Some(control) = self.replicas()[0].control.get().cloned() else {
            return Ok(Vec::new());
        };
        let local = crate::engine::control::BuildServer {
            host: control.host().to_owned(),
            socket: control.build_server().to_owned(),
            targets: Vec::new(),
        };
        let named = crate::epoch::modules_of(target);
        let fixed = tokio::task::spawn_blocking(move || {
            crate::engine::control::with_store_modules(&local, &named)
        })
        .await
        .map_err(|error| UpdateError::Failed(anyhow::anyhow!("fixing the kernels: {error}")))?
        .map_err(|error| UpdateError::Refused(format!("fixing the kernels: {error:#}")))?;
        // A generation that places the probes elsewhere captures every bucket again.
        let probes = |engine: &EngineConfig| {
            (
                engine.profiling.kernel_instance,
                engine.profiling.fence_modules.clone(),
            )
        };
        let current = service::lock(&self.replicas()[0].config).engine_config();
        let recapture = match (target.engine_config(), current) {
            (Ok(target), Ok(current)) => probes(&target) != probes(&current),
            _ => false,
        };
        if fixed.kernels == record.kernels && !recapture {
            return Ok(Vec::new());
        }
        let mut rooms = Vec::new();
        for replica in self.replicas() {
            let Some(client) = replica.hold().client() else {
                continue;
            };
            let measured = client
                .measure_room(&fixed.kernels, recapture)
                .await
                .map_err(|error| {
                    UpdateError::Failed(anyhow::anyhow!("measuring the generation's room: {error}"))
                })?;
            rooms.extend(measured);
        }
        Ok(rooms)
    }

    /// Whether `target`, the current configuration again, selects other module revisions than
    /// the serving epoch runs: a `latest` the Build Servers resolve to a revision installed since
    /// the serving epoch's apply. Such a target is a change though no option changed
    /// (`specs/configuration.md` §8).
    ///
    /// # Errors
    ///
    /// As [`Leader::apply`] when the module set cannot be fixed.
    pub(crate) async fn reselects(
        &self,
        target: &Config,
        programs: BTreeMap<String, proto::module::Program>,
    ) -> Result<bool, UpdateError> {
        let modules = self.fix_modules(target, programs).await?;
        let Some((_, record)) = self.updates.epochs().serving() else {
            return Ok(false);
        };
        Ok(record.engine != modules.engine
            || record.kernels != modules.kernels
            || record.policy != modules.policy)
    }

    /// The per-replica path of a change that starts no update: the daemon's own options (a
    /// hot restart), the options the replicas take in place, or nothing. The deployment's own
    /// in-place options follow once every replica took its own: its served name, which every
    /// replica's requests are named by, and its adapters, which every replica's workers hold in
    /// one slot each.
    async fn apply_in_place(
        &self,
        target: Config,
        stated: Stated,
        revision: u64,
    ) -> Result<Applied, UpdateError> {
        let current = service::lock(&self.replicas()[0].config).clone();
        let mut changed = false;
        for replica in self.replicas() {
            match Box::pin(replica.apply(target.clone(), stated.clone(), revision)).await? {
                applied @ (Applied::Daemon { .. } | Applied::Update(_)) => return Ok(applied),
                Applied::InPlace => changed = true,
                Applied::Unchanged => {}
            }
        }
        if target.served_model_name() != current.served_model_name() {
            self.rename(target.served_model_name());
        }
        if target.adapters != current.adapters {
            self.hold_adapters(&target.adapters).await?;
        }
        Ok(if changed {
            Applied::InPlace
        } else {
            Applied::Unchanged
        })
    }

    /// Fixes the module set `target` runs: every interface's revision, as
    /// the deployment's Build Server on this node selects it with `cover`, with every module built
    /// on every host `target`'s processes run on before anything starts from
    /// it.
    async fn fix_modules(
        &self,
        target: &Config,
        programs: BTreeMap<String, proto::module::Program>,
    ) -> Result<crate::epoch::ModuleSet, UpdateError> {
        let named = crate::epoch::with_programs(crate::epoch::modules_of(target), target, programs);
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
        serving_nodes_of(&self.replicas())
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
        target: &Config,
        next: Epoch,
        serving: crate::epoch::Recorded,
        options: &crate::config::UpdateConfig,
        revision: u64,
    ) -> Result<Applied, UpdateError> {
        let refused = |error: anyhow::Error| UpdateError::Refused(format!("{error:#}"));
        let replicas = self.replicas();
        let control = replicas[0].control.get().ok_or(UpdateError::NoEngine)?;
        // Every engine process that serves the deployment, on every node: those of the replicas
        // the update moves are each drained by its launch generation, and the others serve on.
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
        let prepared = Leader::prepare(
            &self.families,
            target,
            &crate::config::Stated::from_paths(Vec::new()),
        )
        .await
        .map_err(refused)?;
        let engine = prepared.engine.clone();
        // Other model files may process requests otherwise than the served ones: another
        // tokenizer, chat template, generation defaults or model facts. The target then serves
        // with a request processor of its own from its switch on, and the requests the serving
        // epoch holds continue nowhere else: it runs every one to its end (a complete drain),
        // none migrating, and a rollback does the same the other way.
        let served = service::lock(&replicas[0].served).clone();
        let processes_otherwise = match served {
            Some(served) if served.engine_config.model.model != engine.model.model => {
                let files = crate::leader::same_processing(
                    &self.families,
                    &served.engine_config.model.model,
                    &engine.model.model,
                )
                .await;
                files.is_err() || !served.engine_config.model.processes_as(&engine.model)
            }
            _ => false,
        };
        if processes_otherwise {
            info!(
                model = target.model(),
                "the target's model files process requests otherwise: it serves with a processor \
                 of its own, and the serving epoch runs its requests to their ends"
            );
        }

        // One launch number per engine of every target replica's world, from the deployment's
        // one allocator, its engine control: none collides with a standby's or with any other
        // engine of the deployment.
        let launches =
            crate::leader::epoch_launches(&engine, &next.modules, target_epoch, revision)
                .map_err(refused)?;
        // The update moves the replicas the change touches, and the others serve on in the
        // epochs they serve.
        let moved = moved_replicas(&replicas, &launches, &next.modules);
        let (moving, staying): (Vec<_>, Vec<_>) = replicas
            .iter()
            .enumerate()
            .partition(|(index, _)| moved.contains(index));
        let moving: Vec<Arc<Replica>> = moving.into_iter().map(|(_, r)| Arc::clone(r)).collect();
        let staying: Vec<Arc<Replica>> = staying.into_iter().map(|(_, r)| Arc::clone(r)).collect();
        let nodes = serving_nodes_of(&moving)?;
        let bystanders = serving_nodes_of(&staying)?;
        let predecessors: Vec<u64> = moved
            .iter()
            .map(|index| {
                replicas
                    .get(*index)
                    .and_then(|replica| service::lock(&replica.served).as_ref().map(|s| s.epoch))
                    .unwrap_or(serving.number())
            })
            .collect();
        let launches: Vec<_> = launches
            .into_iter()
            .enumerate()
            .filter(|(index, _)| moved.contains(index))
            .map(|(_, args)| args)
            .collect();
        if moved.len() < replicas.len() {
            info!(
                moved = ?moved,
                replicas = replicas.len(),
                "the epoch change moves the replicas it touches, and the others serve on"
            );
        }
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
        for ((args, _, numbers), predecessor) in worlds.into_iter().zip(&predecessors) {
            let mut parked = args.clone();
            parked.predecessor = Some(proto::request::wire::Predecessor {
                epoch: *predecessor,
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
                complete: processes_otherwise,
                target_layouts,
                retire: self.updates.take_retire(),
                bystanders,
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
                moved,
                config: Box::new(target.clone()),
                epoch: target_epoch,
                keep: options.in_flight == InFlight::Keep,
                processing: processes_otherwise.then(|| Box::new(prepared)),
            }),
        );
        Ok(Applied::Update(Box::new(report)))
    }

    /// Takes at a start what of `config` the daemon executes itself rather than the engines'
    /// launch: a start is a change from nothing, so the deployment comes up holding new requests
    /// (`paused`), with its adapters (`adapters`) and profiling session
    /// (`profiling_config.session`), and asleep (`asleep`) last, as a change to `config` would
    /// leave it.
    ///
    /// # Errors
    ///
    /// As [`Replica::take`] and [`Leader::hold_adapters`].
    pub(crate) async fn take_start(&self, config: &Config) -> Result<(), UpdateError> {
        for replica in self.replicas() {
            if config.paused {
                replica.pause();
            }
            if let Some(session) = &config.profiling_config.session {
                let client = replica.hold().client().ok_or(UpdateError::NoEngine)?;
                replica.profile(&client, Some(session.clone())).await?;
            }
        }
        if !config.adapters.is_empty() {
            self.hold_adapters(&config.adapters).await?;
        }
        if config.asleep {
            for replica in self.replicas() {
                let client = replica.hold().client().ok_or(UpdateError::NoEngine)?;
                replica.sleep(&client).await?;
            }
        }
        Ok(())
    }

    /// Ends every replica's running profiling session, because its epoch stops executing.
    async fn end_profiles(&self, reason: &str) {
        for replica in self.replicas() {
            let client = replica.hold().client();
            crate::profiling::end_for_epoch(&replica, client.as_deref(), reason).await;
        }
    }
}

impl Replica {
    /// Applies what of `target` no update moves: nothing for the current configuration; the
    /// daemon's own options, by a hot restart; and, in place, the replica's step and order
    /// options: its admission (`paused`) and sleep (`asleep`), the fence probes' switch, the
    /// experts' order, the prefix cache's flush, the profiling session, and the step values the
    /// schedulers read from their next step's snapshot. An engine that wakes does so before
    /// anything else of the change reaches it, and one that falls asleep does so last.
    ///
    /// # Errors
    ///
    /// Refuses another change in progress (`409`) and a change no in-place effect moves; fails
    /// with the reason the engine did not take a change.
    pub(crate) async fn apply(
        self: &Arc<Self>,
        target: Config,
        stated: Stated,
        revision: u64,
    ) -> Result<Applied, UpdateError> {
        let _replacing = self
            .replacing
            .try_lock()
            .map_err(|_| UpdateError::InProgress)?;
        let current = service::lock(&self.config).clone();
        // The programs are replaced by their hosts between two calls: the leader's by the node,
        // the scheduling program by a generation of its module, the KV program by the Storage
        // Servers. No engine moves for them here.
        let changes: Vec<_> = proto::reload::changes(Some(&current), &target)
            .into_iter()
            .filter(|change| change.effect != Effect::Program)
            .collect();
        if changes.is_empty() {
            *service::lock(&self.config) = target;
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
        self.take(
            &current,
            target,
            &changes,
            target_engine,
            Some(served),
            revision,
        )
        .await
    }

    /// Takes in place the changes `changes` of `current` that `target` makes, `target_engine`
    /// being the engine configuration it resolves to and `served` the launch the serving engine
    /// serves with: the replica's admission (`paused`) and sleep (`asleep`), the fence probes'
    /// switch, the experts' order, the prefix cache's flush, the profiling session, and the step
    /// values the schedulers read from their next step's snapshot. An engine that wakes does so
    /// before anything else of the change reaches it, and one that falls asleep does so last.
    /// The replica then serves `target`, and a relaunch or a standby starts from the launch with
    /// `target_engine` of revision `revision`. The caller holds the replica while no engine
    /// replaces it.
    ///
    /// # Errors
    ///
    /// Refuses a change no in-place effect moves; fails when no engine serves, or with the
    /// reason the engine did not take a change.
    pub(crate) async fn take(
        self: &Arc<Self>,
        current: &Config,
        target: Config,
        changes: &[proto::reload::Change],
        target_engine: EngineConfig,
        served: Option<proto::config::EngineArgs>,
        revision: u64,
    ) -> Result<Applied, UpdateError> {
        // Every change an update moves was decided before this path.
        let unmoved: Vec<_> = changes
            .iter()
            .filter(|change| !matches!(change.effect, Effect::Step | Effect::Order))
            .map(|change| change.path.as_str())
            .collect();
        if !unmoved.is_empty() {
            return Err(UpdateError::Refused(format!(
                "no form of update moves the change of {}",
                unmoved.join(", ")
            )));
        }
        let running = self.hold().client().ok_or(UpdateError::NoEngine)?;
        if current.asleep && !target.asleep {
            self.wake(&running).await?;
        }
        if current.paused != target.paused {
            if target.paused {
                self.pause();
            } else {
                self.resume();
            }
        }
        // The fence probes' switch: every worker writes its device switch at the engine's next
        // step boundary, and the graphs that captured the probes stay.
        if switches_probes(changes) {
            let enabled = target_engine.profiling.fence_probes;
            let off = target_engine.profiling.fence_probes_off.clone();
            let kernels = target_engine.profiling.kernel_probes;
            running
                .switch_fences(enabled, off.clone(), kernels)
                .await
                .map_err(|error| {
                    anyhow::anyhow!("the workers did not switch the probes: {error}")
                })?;
            info!(
                enabled,
                ?off,
                kernels,
                "the workers switched the fence probes in place"
            );
        }
        if current.moe_config.expert_order != target.moe_config.expert_order {
            crate::experts::reorder(
                self,
                &running,
                &current.moe_config.expert_order,
                &target.moe_config.expert_order,
            )
            .await?;
        }
        if current.flush != target.flush {
            self.flush_cache(&running);
        }
        // A lower share of the GPUs: the cache gives back what it leaves no room for, demoting
        // the blocks it evicts to the host tier first, and every rank reserves under it.
        let shares = |config: &EngineConfig| {
            let groups: BTreeMap<String, f64> = config
                .parallel
                .worker_groups
                .iter()
                .filter_map(|group| Some((group.id.0.clone(), group.mem_fraction_static?)))
                .collect();
            (config.mem_fraction_static, groups)
        };
        let lowered = shares(&target_engine);
        if served
            .as_ref()
            .is_some_and(|served| shares(&served.engine_config) != lowered)
        {
            let (deployment, groups) = lowered;
            running
                .lower_fractions(deployment, groups.clone())
                .await
                .map_err(|error| {
                    anyhow::anyhow!("the engine did not take the lower shares of the GPUs: {error}")
                })?;
            info!(
                deployment,
                ?groups,
                "the engine took the lower shares of the GPUs"
            );
        }
        if current.profiling_config.session != target.profiling_config.session {
            self.profile(&running, target.profiling_config.session.clone())
                .await?;
        }
        // The step values: every scheduler reads them from its next step on, and its batches
        // name the revision they are of.
        if tunes(changes) {
            let tuning = proto::config::Tuning::of(&target_engine, revision);
            running.tune(&tuning).await.map_err(|error| {
                anyhow::anyhow!("the engine did not take the step values: {error}")
            })?;
            info!(
                revision,
                "the schedulers take the step values at their next step"
            );
        }
        if !current.asleep && target.asleep {
            self.sleep(&running).await?;
        }
        // A relaunch and a standby start from what serves now, of this revision.
        if let Some(mut args) = served {
            args.engine_config = target_engine;
            args.revision = revision;
            *service::lock(&self.served) = Some(args);
        }
        *service::lock(&self.config) = target;
        Ok(Applied::InPlace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fence probes' switch, and a module's probes switched off, change in place; the
    /// probed modules shape the captured graphs and take a kernel generation, and PC sampling,
    /// the process's one CUPTI mode every graph is instantiated under, a new epoch.
    #[test]
    fn only_the_fence_probes_switch_changes_profiling_in_place() {
        let profiled = |profiling: proto::config::ProfilingConfig| Config {
            profiling_config: profiling,
            ..Config::default()
        };
        let served = profiled(proto::config::ProfilingConfig {
            fence_probes: true,
            fence_modules: Some(vec!["*.Transformer".to_owned()]),
            ..proto::config::ProfilingConfig::default()
        });
        let changed = |target: &Config| proto::reload::changes(Some(&served), target);
        let switched = profiled(proto::config::ProfilingConfig {
            fence_probes: false,
            ..served.profiling_config.clone()
        });
        assert!(switches_probes(&changed(&switched)));
        assert!(!tunes(&changed(&switched)));
        // A module's probes switch off in place too.
        let thinned = profiled(proto::config::ProfilingConfig {
            fence_probes_off: vec!["*.Attention".to_owned()],
            ..served.profiling_config.clone()
        });
        assert!(switches_probes(&changed(&thinned)));
        let sampled = changed(&profiled(proto::config::ProfilingConfig {
            graph_pc_sampling: true,
            ..served.profiling_config.clone()
        }));
        assert!(!switches_probes(&sampled));
        assert_eq!(Effects::of(&sampled).heaviest(), Some(Effect::Epoch));
        // The probed modules are captured into the graphs, again by a kernel generation.
        let moved = changed(&profiled(proto::config::ProfilingConfig {
            fence_modules: Some(vec!["*.TransformerLayer".to_owned()]),
            ..served.profiling_config.clone()
        }));
        assert!(!switches_probes(&moved));
        assert_eq!(Effects::of(&moved).heaviest(), Some(Effect::Generation));
        // From no probe, a new epoch installs the range markers.
        let unprobed = profiled(proto::config::ProfilingConfig {
            fence_modules: Some(Vec::new()),
            ..served.profiling_config.clone()
        });
        let probed = proto::reload::changes(Some(&unprobed), &served);
        assert_eq!(Effects::of(&probed).heaviest(), Some(Effect::Epoch));
        assert!(!switches_probes(&[]));
    }

    /// The options the daemon applies itself take effect at a step and send the schedulers no
    /// step values; a scheduling limit does, and the experts' order is its own effect.
    #[test]
    fn the_daemons_own_step_options_send_no_step_values() {
        let served = Config::default();
        let changed = |change: &dyn Fn(&mut Config)| {
            let mut target = served.clone();
            change(&mut target);
            proto::reload::changes(Some(&served), &target)
        };
        for daemon in [
            changed(&|target| target.paused = true),
            changed(&|target| target.asleep = true),
            changed(&|target| target.flush = 1),
            changed(&|target| target.retire = true),
            changed(&|target| target.served_model_name = Some("renamed".to_owned())),
            changed(&|target| {
                target
                    .adapters
                    .insert("sql".to_owned(), "/adapters/sql".to_owned());
            }),
            changed(&|target| {
                target.profiling_config.session = Some(proto::profile::SessionRequest::default());
            }),
        ] {
            assert_eq!(
                Effects::of(&daemon).heaviest(),
                Some(Effect::Step),
                "{daemon:?}"
            );
            assert!(!tunes(&daemon), "{daemon:?}");
        }
        let limits =
            changed(&|target| target.max_num_batched_tokens = std::num::NonZeroUsize::new(4096));
        assert!(tunes(&limits));
        let order = changed(&|target| {
            target
                .moe_config
                .expert_order
                .insert("model.layers.0.mlp".to_owned(), vec![1, 0]);
        });
        assert_eq!(Effects::of(&order).heaviest(), Some(Effect::Order));
        assert!(!tunes(&order));
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
}
