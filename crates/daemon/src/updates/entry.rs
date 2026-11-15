//! The update entry: one change of the deployment's configuration, applied
//! as the least it needs (`PUT /admin/config`).
//!
//! A change moves one of four things, decided by comparing what it moves
//! ([`crate::epoch::change`]) once the target's module set is fixed at the
//! Build Servers:
//!
//! - the daemon's own options: a hot restart of the daemon, whose successor
//!   applies what else the change states;
//! - the engine package, the build configuration or the checkpoint: an update
//!   in its epoch form, target engines started beside the serving ones;
//! - kernel or policy packages, the policy's parameters included: an update
//!   in its generation form, inside the serving engines;
//! - nothing an engine executes: in place, or nothing at all.
//!
//! `Leader::apply` decides in that order and starts the update;
//! `Leader::commit` and `Leader::rollback` decide it. The update itself is
//! `crate::updates::update::Update`, advanced by the engines'
//! acknowledgements; what the daemon does beside it for a replica (moving
//! requests to the engine that serves, reading the trial's thresholds,
//! enforcing the deadline) is [`Replica::follow`].

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use proto::admin::{
    ApplyRequest, ConfigStatus, DryRun, EngineLaunch, Executing, ExecutingState, Failure,
    FailureKind, Gpu, GpuShare, GpuUuid, InFlight, InPlaceChange, NodeUsage, Quotas, ReplicaDryRun,
    ReplicaStatus, Switched, UpdateReport,
};
use proto::config::EngineConfig;
use tracing::info;

use crate::engine::client::EngineState;
use crate::engine::hold::HoldState;
use crate::epoch::{Change, Epoch};
use crate::leader::{Leader, Replica, UpdateError};
use crate::options::ServerArgs;
use crate::updates::change::{Applied, resolve};
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

/// Whether `target` differs from `served` in the profiler ranges' switch
/// alone, which the workers change in place. Every other part of the
/// profiling configuration shapes the CUDA graphs an epoch captures at its
/// start, the modules it times and whether PC sampling was enabled before
/// the capture (`profiling_config.graph_pc_sampling`), and takes a new
/// epoch.
fn switches_probes_only(served: &EngineConfig, target: &EngineConfig) -> bool {
    let mut switched = comparable(served);
    switched.profiling.fence_probes = target.profiling.fence_probes;
    switched
        .profiling
        .fence_probes_off
        .clone_from(&target.profiling.fence_probes_off);
    switched == comparable(target)
}

/// The parts of `config` a replica compares in place: all of it but what a
/// module selection owns (the package backends, the scheduling policy, the
/// expert exchange and the speculative cost figures). A change reaches a
/// replica only once the module set has accounted for those.
///
/// Every field of the configuration is named here, so a field added to it does not compile
/// until it is classified as compared or as a module selection's.
fn comparable(config: &EngineConfig) -> EngineConfig {
    let EngineConfig {
        executable,
        model,
        parallel,
        deployment,
        max_running_requests,
        max_num_batched_tokens,
        max_logprobs,
        attention_backend: _,
        prefill_attention_backend: _,
        decode_attention_backend: _,
        moe_runner_backend: _,
        device,
        scheduler,
        media,
        mem_fraction_static,
        load,
        kv_cache,
        moe,
        cuda_graph,
        worker,
        speculative,
        video,
        profiling,
    } = config.clone();
    EngineConfig {
        executable,
        model,
        parallel,
        deployment,
        max_running_requests,
        max_num_batched_tokens,
        max_logprobs,
        // The package backends are the module set's.
        attention_backend: None,
        prefill_attention_backend: None,
        decode_attention_backend: None,
        moe_runner_backend: None,
        device,
        // The scheduling policy's package is the module set's; its parameters are compared.
        scheduler: proto::config::SchedulerConfig {
            policy: proto::config::SchedulerConfig::default().policy,
            ..scheduler
        },
        media,
        mem_fraction_static,
        load,
        kv_cache,
        // The expert exchange's package is the module set's.
        moe: proto::config::MoeConfig {
            exchange: None,
            ..moe
        },
        cuda_graph,
        worker,
        // The speculative cost figures are the scheduling policy package's parameters.
        speculative: proto::config::SpeculativeConfig {
            expected_speedup: None,
            refill_token_cost: None,
            ..speculative
        },
        video,
        profiling,
    }
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

/// The options that select a deployment's packages: the engine package, the
/// scheduling policy and each operator's implementation. A harness may set
/// them without the operator listing them.
pub(crate) const MODULE_OPTIONS: [&str; 8] = [
    "worker_config.engine",
    "worker_config.engine_revision",
    "scheduler_config.policy",
    "attention_backend",
    "prefill_attention_backend",
    "decode_attention_backend",
    "moe_runner_backend",
    "moe_config.exchange",
];

/// The parts of a change no form of update moves.
fn differing(served: &EngineConfig, target: &EngineConfig) -> Vec<&'static str> {
    let mut parts = Vec::new();
    if served.deployment != target.deployment {
        parts.push("the deployment");
    }
    parts
}

/// Whether the replicas `target` places lay their device regions out
/// otherwise than the replicas `serving` runs: another parallel configuration
/// (degrees, forms, worker groups, devices, hosts) or another number of
/// replicas, another slot count, KV layout (block size, cache precision) or
/// set of graph buckets. Two epochs of one layout name the same regions; two
/// of different layouts have regions of their own over the same device
/// memory, so the drain gives the serving epoch's back (`shrink`) and the
/// target grows into it. The cache's size and its tiers are no layout. The
/// engine names its device KV regions by the same layout
/// (`EngineConfig::layout`).
fn layouts_differ(serving: &[EngineConfig], target: &[EngineConfig]) -> bool {
    serving.len() != target.len()
        || serving
            .iter()
            .zip(target)
            .any(|(serving, target)| serving.layout() != target.layout())
}

impl Leader {
    /// Applies a configuration change as operation `number`, as the module
    /// describes: the least the change needs.
    ///
    /// An update it starts runs on after this returns: its phases end on the
    /// engines' acknowledgements, and `GET /admin/status`, the events and
    /// `GET /admin/config` report it.
    ///
    /// # Errors
    ///
    /// `UpdateError::InProgress` while another change or update runs;
    /// `UpdateError::Refused` for a configuration that does not parse or
    /// validate, one a rollback marked unavailable unless `force`, a module
    /// set the Build Servers cannot select, or a change no form moves.
    pub(crate) async fn apply(
        &self,
        request: ApplyRequest,
        number: u64,
    ) -> Result<Applied, UpdateError> {
        let _replacing = self
            .replicas
            .replacing
            .try_lock()
            .map_err(|_| UpdateError::InProgress)?;
        let replicas = self.replicas();
        if !request.dry_run && self.updates.in_flight().is_some() {
            return Err(UpdateError::InProgress);
        }
        let current = service::lock(&replicas[0].config).clone();
        let (target, _stated, options) = resolve(&current, &request)?;

        let named_previous = crate::leader::deployment(&current)?;
        let named_next = crate::leader::deployment(&target)?;
        let named = crate::epoch::change(&named_previous, &named_next);
        if named == Change::Host {
            return self.apply_in_place(request).await;
        }

        // The one selection of an apply: every interface's revision is fixed
        // at the Build Servers before anything is compared, so a revision
        // installed since the serving epoch was applied is a change. A dry
        // run of an epoch change starts nothing and selects nothing.
        let mut next = named_next;
        if !(request.dry_run && named == Change::Epoch) {
            next.modules = self.fix_modules(&target).await?;
        }
        let Some((serving, record)) = self.updates.epochs().serving() else {
            return self.apply_in_place(request).await;
        };
        let change = if named == Change::Epoch || record.engine != next.modules.engine {
            Change::Epoch
        } else if record.kernels != next.modules.kernels || record.policy != next.modules.policy {
            Change::Generation
        } else {
            Change::None
        };
        if change == Change::None {
            return self.apply_in_place(request).await;
        }
        if !request.force
            && let Some(mark) = self.updates.unavailable(&next)
        {
            return Err(UpdateError::Refused(format!(
                "the configuration was rolled back from epoch {} ({}); apply it again with force",
                mark.epoch, mark.reason
            )));
        }
        let deadline = request
            .deadline_s
            .map(|seconds| Instant::now() + Duration::from_secs_f64(seconds));

        match change {
            Change::Epoch => {
                Box::pin(self.epoch_update(
                    number, &current, &target, next, serving, &options, &request, deadline,
                ))
                .await
            }
            _ => {
                self.generation_update(number, next, serving, &options, &request, deadline)
                    .await
            }
        }
    }

    /// Whether a harness may apply `request`: it selects packages
    /// ([`MODULE_OPTIONS`]) and sets only the options the operator lists for
    /// it (`admin_config.harness_options`), by assignment. A whole
    /// configuration, one replica of the deployment and `force` are the
    /// operator's.
    ///
    /// # Errors
    ///
    /// The refusal, naming what the harness may not state.
    pub(crate) fn harness_may_apply(&self, request: &ApplyRequest) -> Result<(), String> {
        if request.config.is_some() {
            return Err(
                "a harness states assignments (`set`), not a whole configuration".to_owned(),
            );
        }
        if request.force {
            return Err("applying a configuration a rollback marked unavailable (`force`) is the operator's".to_owned());
        }
        if request.replica.is_some() {
            return Err("a harness applies to the deployment, not to one replica".to_owned());
        }
        let allowed = service::lock(&self.replicas()[0].config)
            .admin_config
            .harness_options
            .clone();
        for assignment in &request.set {
            let path = assignment
                .split_once('=')
                .map_or(assignment.as_str(), |(path, _)| path)
                .trim();
            let permitted =
                MODULE_OPTIONS.contains(&path) || allowed.iter().any(|option| option == path);
            if !permitted {
                return Err(format!(
                    "a harness may not set `{path}`: it is neither a package selection nor an \
                     option the operator lists in `admin_config.harness_options`"
                ));
            }
        }
        Ok(())
    }

    /// The per-replica path of a change that starts no update: the daemon's
    /// own options (a hot restart), the options the workers switch in place,
    /// or nothing.
    async fn apply_in_place(&self, request: ApplyRequest) -> Result<Applied, UpdateError> {
        let replicas = self.replicas();
        let indices: Vec<_> = match request.replica {
            Some(index) => vec![index],
            None => (0..replicas.len()).collect(),
        };
        // What each replica the change reaches would change, for a dry run, or changed.
        let mut dry_runs = Vec::new();
        let mut changed = Vec::new();
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
                Applied::DryRun(DryRun::InPlace { replicas }) => dry_runs.extend(replicas),
                Applied::InPlace(changes) => changed.extend(changes),
                applied @ (Applied::DryRun(_) | Applied::Update(_)) => return Ok(applied),
                Applied::Unchanged => {}
            }
        }
        Ok(if !dry_runs.is_empty() {
            Applied::DryRun(DryRun::InPlace { replicas: dry_runs })
        } else if !changed.is_empty() {
            Applied::InPlace(changed)
        } else {
            Applied::Unchanged
        })
    }

    /// Fixes the module set `target` runs: every interface's revision, as
    /// this host's Build Server's `cover` selects it, with every module built
    /// on every host `target`'s processes run on before anything starts from
    /// it.
    async fn fix_modules(
        &self,
        target: &ServerArgs,
    ) -> Result<crate::epoch::ModuleSet, UpdateError> {
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
            socket: target
                .build_config
                .socket()
                .map_err(|error| UpdateError::Refused(format!("{error}")))?,
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
        options: &crate::options::UpdateConfig,
        request: &ApplyRequest,
        deadline: Option<Instant>,
    ) -> Result<Applied, UpdateError> {
        if request.dry_run {
            return Ok(Applied::DryRun(DryRun::Generation {
                epoch: serving.number(),
                modules: crate::updates::change::modules_report(&next.modules),
            }));
        }
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
            deadline,
        );
        let report = self.updates.apply(update)?;
        for replica in self.replicas() {
            replica.follow(
                Arc::clone(&self.updates),
                number,
                options.clone(),
                deadline,
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
                nodes.push(Node::new(process.host, rank, process.engine, head, scopes));
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
        current: &ServerArgs,
        target: &ServerArgs,
        next: Epoch,
        serving: crate::epoch::Recorded,
        options: &crate::options::UpdateConfig,
        request: &ApplyRequest,
        deadline: Option<Instant>,
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
            &crate::options::Stated::from_paths(Vec::new()),
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
        let serving_configs: Vec<_> = replicas
            .iter()
            .filter_map(|replica| {
                service::lock(&replica.served)
                    .as_ref()
                    .map(|args| args.engine_config.clone())
            })
            .collect();
        let shrink = layouts_differ(&serving_configs, &configs);
        if request.dry_run {
            return Ok(Applied::DryRun(DryRun::Epoch {
                epoch: target_epoch,
                engine: next.modules.engine.clone(),
                replicas: worlds.len(),
                engines: worlds
                    .iter()
                    .map(|(_, hosts, numbers)| {
                        hosts
                            .iter()
                            .zip(numbers)
                            .map(|(host, number)| EngineLaunch {
                                host: host.host.clone(),
                                launch_generation: *number,
                            })
                            .collect()
                    })
                    .collect(),
                shrink,
            }));
        }

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
                recompute: request.in_flight == InFlight::Recompute,
                drain: Duration::from_secs_f64(options.drain_s),
                shrink,
                retire: self.updates.take_retire(),
            },
            deadline,
        );
        // A profiling session does not cross epochs: its ranks write their
        // files while the epoch still executes, before the switch drains it.
        self.end_profiles("epoch change").await;
        let report = self.updates.apply(update)?;
        replicas[0].follow(
            Arc::clone(&self.updates),
            number,
            options.clone(),
            deadline,
            crate::leader::Following::Epoch(crate::leader::EpochTarget {
                control: Arc::clone(control),
                targets,
                config: Box::new(target.clone()),
                epoch: target_epoch,
                keep: request.in_flight == InFlight::Keep,
            }),
        );
        Ok(Applied::Update(Box::new(report)))
    }

    /// Commits the update in flight: the decision is recorded before the kept
    /// side is released.
    pub(crate) async fn commit(&self) -> Result<UpdateReport, UpdateError> {
        let _replacing = self.replicas.replacing.lock().await;
        self.updates.commit().ok_or_else(|| {
            UpdateError::Refused("no update is in flight: there is nothing to commit".to_owned())
        })
    }

    /// Ends every replica's running profiling session, because its epoch stops executing.
    async fn end_profiles(&self, reason: &str) {
        for replica in self.replicas() {
            let client = replica.hold().client();
            crate::profiling::end_for_epoch(&replica, client.as_deref(), reason).await;
        }
    }

    /// Rolls the update in flight back, with `reason` as the operator's or
    /// the submitter's own account.
    pub(crate) async fn rollback(&self, reason: String) -> Result<UpdateReport, UpdateError> {
        let _replacing = self.replicas.replacing.lock().await;
        if self.updates.in_flight().is_some() {
            self.end_profiles("rollback").await;
        }
        self.updates
            .rollback(Failure::new(FailureKind::Decision, reason))
            .ok_or_else(|| {
                UpdateError::Refused(
                    "no update is in flight: the deployment already serves the configuration \
                     the file states"
                        .into(),
                )
            })
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
        let printed = crate::options::printed(&config)?;
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

    /// Applies what of `request` no update moves: nothing for the current
    /// configuration; the workers' own switch for the profiler ranges; the
    /// record of scheduling limits the policy package already took; and a
    /// hot restart of the daemon for its own options.
    ///
    /// # Errors
    ///
    /// Refuses another change in progress (`409`), a configuration that does
    /// not parse or validate, and a change no form of update moves.
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
        let current = service::lock(&self.config).clone();
        let (target, stated, _) = resolve(&current, &request)?;
        if target == current && target.worker_groups.is_none() {
            return Ok(Applied::Unchanged);
        }
        let served = service::lock(&self.served)
            .clone()
            .ok_or(UpdateError::NoEngine)?;
        // The target's engine configuration is resolved as a deployment's
        // start resolves it: from the options, completed by what the model's
        // files determine, and divided into the deployment's replicas.
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
        let daemon_changes = current.daemon_part() != target.daemon_part();
        let in_place = crate::leader::configurable(
            &comparable(&served.engine_config),
            &comparable(&target_engine),
            &running.info().capacity,
        );
        // A successor serving other model files processes requests as the
        // daemon does only with the same tokenizer, templates, generation
        // defaults and model facts.
        let served_model = &served.engine_config.model;
        if served_model.model != target_engine.model.model {
            crate::leader::same_processing(
                &self.families,
                &served_model.model,
                &target_engine.model.model,
            )
            .await
            .map_err(UpdateError::Refused)?;
            if !served_model.processes_as(&target_engine.model) {
                return Err(UpdateError::Refused(format!(
                    "the model at {} has other generation limits, control tokens or context \
                     length than the served model, which change how requests are processed: \
                     deploy it instead",
                    target_engine.model.model
                )));
            }
        }
        // A dry run states what the change would do and changes nothing: no
        // handover, no engine command, no configuration written back.
        if request.dry_run {
            return Ok(Applied::DryRun(DryRun::InPlace {
                replicas: vec![ReplicaDryRun {
                    replica: self.ordinal,
                    daemon: daemon_changes,
                    scheduling: in_place,
                    fence_probes: switches_probes_only(&served.engine_config, &target_engine),
                }],
            }));
        }
        // The daemon orchestrates what changes in its own options: a hot
        // restart applies them first, and its successor applies the change
        // again, which then leaves the daemon's options as they are.
        if daemon_changes {
            return Ok(Applied::Daemon {
                config: Box::new(current.with_daemon_part(target.daemon_part())),
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
            let mut args = served;
            args.engine_config = target_engine;
            *service::lock(&self.served) = Some(args);
            *service::lock(&self.config) = target;
            return Ok(Applied::InPlace(vec![InPlaceChange {
                replica: self.ordinal,
                switched: Switched::FenceProbes,
            }]));
        }
        // A scheduling limit is a parameter of the policy package, so a generation update has
        // moved the schedulers to it or they run it already. What is left is the record of it
        // here: the arguments the next apply compares with.
        if in_place {
            let mut args = served;
            args.engine_config = target_engine;
            *service::lock(&self.served) = Some(args);
            *service::lock(&self.config) = target;
            return Ok(Applied::InPlace(vec![InPlaceChange {
                replica: self.ordinal,
                switched: Switched::SchedulingLimits,
            }]));
        }
        // Every change an update moves was decided before this path: what
        // reaches here changed nothing an epoch is built from and no module,
        // and is none of the options changed in place either.
        Err(UpdateError::Refused(format!(
            "no form of update moves the change ({}): deploy the configuration instead",
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

    /// The fence probes' switch, and a module's probes switched off, change in place; the
    /// probed modules and PC sampling shape the captured graphs and take a new epoch.
    #[test]
    fn only_the_fence_probes_switch_changes_profiling_in_place() {
        let profiled = |profiling: proto::config::ProfilingConfig| EngineConfig {
            profiling,
            ..config()
        };
        let served = profiled(proto::config::ProfilingConfig {
            fence_probes: true,
            fence_modules: Some(vec!["*.Transformer".to_owned()]),
            fence_probes_off: Vec::new(),
            graph_pc_sampling: false,
            nvtx_ranges: false,
        });
        let switched = profiled(proto::config::ProfilingConfig {
            fence_probes: false,
            ..served.profiling.clone()
        });
        assert!(switches_probes_only(&served, &switched));
        // A module's probes switch off in place too.
        let thinned = profiled(proto::config::ProfilingConfig {
            fence_probes_off: vec!["*.Attention".to_owned()],
            ..served.profiling.clone()
        });
        assert!(switches_probes_only(&served, &thinned));
        for epoch in [
            proto::config::ProfilingConfig {
                graph_pc_sampling: true,
                ..served.profiling.clone()
            },
            proto::config::ProfilingConfig {
                fence_modules: Some(vec!["*.TransformerLayer".to_owned()]),
                ..served.profiling.clone()
            },
        ] {
            assert!(!switches_probes_only(&served, &profiled(epoch)));
        }
    }

    /// A change of the scheduling limits alone, within the batch the workers were started for, is
    /// in place, and a value a module selection owns does not stand in its way: the module set was
    /// compared before the change reached the replica.
    #[test]
    fn a_scheduling_change_is_in_place_beside_what_a_module_selection_owns() {
        let mut served = config();
        served.scheduler.max_batch = std::num::NonZeroUsize::new(8).unwrap();
        served.max_running_requests = std::num::NonZeroUsize::new(8).unwrap();
        let mut target = served.clone();
        target.max_running_requests = std::num::NonZeroUsize::new(4).unwrap();
        target.attention_backend = Some("attention-c".to_owned());
        let capacity = proto::request::wire::BatchCapacity {
            max_batch: 8,
            max_num_batched_tokens: 8192,
            max_running_requests: 8,
        };
        assert!(crate::leader::configurable(
            &comparable(&served),
            &comparable(&target),
            &capacity
        ));
        assert!(!crate::leader::configurable(&served, &target, &capacity));
    }

    /// The target's layout differs from the serving one's in its parallel
    /// configuration, its devices, its slot count, its KV layout, its graph
    /// buckets and its number of replicas, and not in the cache's size or its
    /// tiers.
    #[test]
    fn a_layout_is_the_parallel_configuration_slots_kv_layout_and_graph_buckets() {
        let served = config();
        let same = |change: &dyn Fn(&mut EngineConfig)| {
            let mut target = served.clone();
            change(&mut target);
            !layouts_differ(std::slice::from_ref(&served), &[target])
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
        assert!(layouts_differ(&[served.clone(), served.clone()], &[served]));
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
