//! The leader's served model: one model's input processing, its engine
//! process, and the output processing of every request.
//!
//! A [`Leader`] owns the model's [`InputProcessor`] and the client of the
//! engine process that schedules its requests ([`EngineClient`]), as vLLM's
//! `AsyncLLM` owns its input and output processors over its engine core
//! client and SGLang's tokenizer manager drives its schedulers. Every
//! request takes one path:
//!
//! 1. its images are fetched on the async runtime and its prompt rendered and
//!    tokenized on the blocking pool ([`InputProcessor`]);
//! 2. [`Leader::submit`] hands the tokenized request through the migrator
//!    to the engine process, which routes it to one of its schedulers and
//!    returns its events ([`RequestEvents`]), across the engines that serve
//!    it when it migrates;
//! 3. the events are assembled into public [`RequestOutput`] events
//!    ([`output::generation_output`], [`output::media_output`]) and wrapped
//!    in [`RequestStats`], which records the request's metrics once.
//!
//! Dropping the returned stream drops its `RequestEvents`, which cancels the
//! request in the engine that serves it; that is the only cancellation path.
//!
//! [`RequestOutput`]: crate::output::RequestOutput
//! [`output::generation_output`]: crate::output::generation_output
//! [`output::media_output`]: crate::output::media_output

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::metrics::{EngineLabels, METRICS};
use anyhow::Context as _;
use futures::StreamExt as _;
use proto::request::wire;
use proto::request::{GenerationLimits, Request};
use tokio::sync::watch;
use tracing::{info, warn};

use crate::engine::client::{EngineClient, EngineInfo, EngineState, StatsSink, SubmitError};
use crate::engine::control::EngineControl;
use crate::engine::hold::{HoldState, RequestHold};
use crate::input::media::MediaConnector;
use crate::input::processor::LoadModel as _;
use crate::input::video::{ProcessedVideo, VideoGenerationRequest, VideoService};
use crate::input::{
    GenerateRequest, InputProcessor, LoadedModel, LoadedRenderer, Prompt, RequestRenderer,
    ServeError, TokenizeError,
};
use crate::metrics::log_stats::StatsLogger;
use crate::metrics::stats::{self as metrics, RequestStats};
use crate::migration::{Migrator, RequestEvents};
use crate::model::{ModelConfig, ModelParameters};
use crate::options::{ReasoningParser, ServerArgs, Stated};
use crate::output::RequestOutputStream;

/// How long the epoch's Program Servers may take to select its programs.
///
/// A Program Server that selects a program compiles it when it has no module:
/// a cold store builds every kernel of the epoch, which minutes of compiler
/// work per kernel makes long. The budget is the engine's own for the precopy
/// (§7.3), and it is a backstop rather than the signal a reader follows: the
/// line the install prints before it waits, and the Program Server's own
/// module progress, say whether the selection is working or stopped.
const PREPARE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20 * 60);

/// How often a wait on the epoch's programs says it is still waiting.
///
/// The budgets below are longer than a run, on purpose: a cold store's modules
/// are minutes of compiler work, and an experiment that lasts fifteen minutes
/// would otherwise end without a single line saying where the daemon was. The
/// heartbeat turns a wait that outlives the run into a trail of lines that name
/// the step the daemon is in and how long it has been in it.
const HEARTBEAT: std::time::Duration = std::time::Duration::from_secs(60);

/// How long the main processes may take to take the epoch's programs.
///
/// A backstop beyond the engine's own budget for the steps (§7.3: the precopy
/// waits up to 20 minutes, a switch, commit, revert or purge 2), which fire
/// first and name the process that did not answer.
const SWITCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(25 * 60);

/// One model served over its engine process.
///
/// Construction spawns Tokio tasks, so every constructor must be called
/// within a Tokio runtime.
pub struct Leader {
    /// The single model name the API serves.
    served_model_name: String,
    /// The model the request and scheduler metrics are labelled with.
    model_name: String,
    processor: Arc<InputProcessor>,
    /// What the engine the model was bound to reported it serves.
    info: EngineInfo,
    /// Independent replica worlds behind the public request entry point.
    pub(crate) replicas: Arc<Replicas>,
    images: MediaConnector,
    /// The epoch layer: the epoch records, the switch in flight and the
    /// adapter that commands the deployment's main processes (§7.4). It
    /// shares the records with the admin's `GET /admin/config`.
    pub(crate) updates: Arc<crate::updates::driver::Driver>,
    /// Whether every request logs a summary line when it ends.
    log_requests: bool,
    /// Periodic throughput logging, stopped with the engine.
    _stats_logger: Option<StatsLogger>,
}

/// The serving replica set and its sequential rollout. Replicas keep a
/// weak reference so an automatic rollback can end the rollout and restore
/// the deployment's request routing.
pub(crate) struct Replicas {
    pub(crate) serving: watch::Sender<Vec<Arc<Replica>>>,
    pub(crate) replacing: tokio::sync::Mutex<()>,
    pub(crate) adapters: Arc<std::sync::Mutex<BTreeMap<String, Adapter>>>,
    pub(crate) schedulers: Arc<std::sync::atomic::AtomicUsize>,
    /// The update layer, so a replica that loses its engine asks whether an
    /// update in flight owns that decision (§7.2, §7.4).
    pub(crate) updates: std::sync::Weak<crate::updates::driver::Driver>,
}

/// Keeps deployment updates and replica recovery out of a daemon handoff.
pub(crate) struct Quiescence<'a> {
    _deployment: tokio::sync::MutexGuard<'a, ()>,
    _replicas: Vec<tokio::sync::OwnedMutexGuard<()>>,
}

/// A task an engine serves, which selects the API routes the server
/// registers, as vLLM's `supported_tasks` selects its routers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Chat completions (`/v1/chat/completions`): every model that renders
    /// a chat prompt.
    Generate,
    /// Video generation (`/v1/videos`, `/v1/capabilities`).
    VideoGeneration,
}

/// What a replica needs to follow an epoch switch onto its target (§7.4).
pub(crate) struct SwitchTarget {
    /// The arguments the target's process started with.
    pub(crate) args: proto::config::EngineArgs,
    /// The configuration the target serves.
    pub(crate) config: ServerArgs,
    /// The programs the target takes before it is promoted.
    pub(crate) programs: crate::epoch::FilledPrograms,
    /// The Program Servers that select those programs' revisions, and the host that leads them.
    pub(crate) servers: Vec<crate::engine::control::ProgramServer>,
    pub(crate) leader: String,
    /// The target's launch generation, the number its links state.
    pub(crate) engine: u64,
    /// The epoch the target serves.
    pub(crate) epoch: u64,
    /// The options that govern this update (`update_config`, with what its own `set` stated).
    pub(crate) update: crate::options::UpdateConfig,
}

/// Records each scheduler's statistics under `model_name`, and with
/// `observer`, which reads each world's rates.
pub(crate) fn stats_sink(
    model_name: String,
    observer: Arc<crate::observe::Observer>,
    replica: usize,
) -> StatsSink {
    Box::new(move |world, scheduler, snapshot| {
        observer.record(world, scheduler, &snapshot);
        metrics::record_scheduler_stats(
            &METRICS.scheduler,
            &model_name,
            scheduler + replica as u32,
            &snapshot,
        );
    })
}

/// How many requests ended since the daemon started, and how many failed.
pub(crate) fn outcomes() -> (u64, u64) {
    let outcomes = &metrics::OUTCOMES;
    (
        outcomes.ended.load(Ordering::Relaxed),
        outcomes.failed.load(Ordering::Relaxed),
    )
}

/// The engine configuration of each data-parallel replica of `config`: independent replicas each
/// run their own engine process on their own worker groups, while schedulers whose expert-parallel
/// groups exchange experts stay in one.
///
/// # Errors
///
/// Fails when the worker groups do not form equal replicas, when the transfer edges do not parse,
/// or when a transfer edge connects two replicas.
pub(crate) fn replica_configs(
    config: &proto::config::EngineConfig,
) -> anyhow::Result<Vec<proto::config::EngineConfig>> {
    let count = config.parallel.data_parallel_size.get();
    if count == 1 || config.moe.expert_parallel {
        return Ok(vec![config.clone()]);
    }
    let groups = &config.parallel.worker_groups;
    anyhow::ensure!(
        !groups.is_empty() && groups.len().is_multiple_of(count),
        "the worker groups must form {count} equal independent replicas"
    );
    let transfers = config
        .device
        .transfer
        .as_deref()
        .unwrap_or_default()
        .parse::<engine::TransferConfig>()
        .context("invalid `device_config.transfer`")?;
    groups
        .chunks(groups.len() / count)
        .map(|groups| {
            let contains =
                |id: &proto::config::WorkerGroupId| groups.iter().any(|group| &group.id == id);
            let mut local = engine::TransferConfig::default();
            for edge in &transfers.edges {
                let source = contains(&edge.source_worker);
                let destination = contains(&edge.destination_worker);
                anyhow::ensure!(
                    source == destination,
                    "a transfer edge connects independent replicas: {} -> {}",
                    edge.source_worker,
                    edge.destination_worker
                );
                if source {
                    local.edges.push(edge.clone());
                }
            }
            let mut replica = config.clone();
            replica.parallel.data_parallel_size = std::num::NonZeroUsize::MIN;
            replica.parallel.worker_groups = groups.to_vec();
            replica.device.transfer = (!local.edges.is_empty()).then(|| local.to_string());
            Ok(replica)
        })
        .collect()
}

/// What a leader prepares before it has an engine (`Leader::prepare`).
pub(crate) struct Prepared {
    model_config: ModelConfig,
    tokenizer: crate::tokenizer::DynTokenizer,
    renderer: LoadedRenderer,
    /// The deployment's engine configuration, with the facts the model's files determine.
    pub(crate) engine: proto::config::EngineConfig,
    max_model_len: u32,
}

impl Leader {
    pub(crate) fn replicas(&self) -> Vec<Arc<Replica>> {
        self.replicas.serving.borrow().clone()
    }

    /// Assigns a new request to the least loaded ready replica. A request
    /// stays with that replica's migrator through its world transitions.
    async fn replica_for_request(&self) -> Result<Arc<Replica>, SubmitError> {
        let mut observed = self.replicas.serving.subscribe();
        'changed: loop {
            let replicas = observed.borrow_and_update().clone();
            if let Some(replica) = replicas
                .iter()
                .filter(|replica| replica.hold.is_serving())
                .min_by_key(|replica| replica.hold.client().map_or(0, |client| client.in_flight()))
            {
                return Ok(Arc::clone(replica));
            }
            let mut ready: futures::stream::FuturesUnordered<_> = replicas
                .into_iter()
                .map(|replica| async move { replica.hold.ready(None).await.map(|_| replica) })
                .collect();
            let mut failure = None;
            loop {
                tokio::select! {
                    changed = observed.changed() => {
                        if changed.is_err() {
                            return Err(SubmitError::Unavailable("the daemon stopped serving requests".into()));
                        }
                        continue 'changed;
                    }
                    result = ready.next() => match result {
                        Some(Ok(replica)) => return Ok(replica),
                        Some(Err(error)) => failure = Some(error),
                        None => return Err(failure.unwrap_or_else(|| SubmitError::Unavailable("no replica serves requests".into()))),
                    }
                }
            }
        }
    }

    /// Ends every replica's parked standby (§2.5).
    ///
    /// A daemon that hands the deployment to a successor calls this before the
    /// handoff: the successor numbers and starts its own standbys, and a parked
    /// process of the old daemon would otherwise keep a second engine's memory
    /// on the device for a deployment that never knew it was there.
    pub(crate) async fn end_standbys(&self) {
        for replica in self.replicas() {
            replica.end_standby().await;
        }
    }

    /// Stops every replica and its workers.
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        for replica in self.replicas() {
            replica.shutdown().await?;
        }
        Ok(())
    }

    /// Releases every adopted engine connection after a daemon handoff.
    pub async fn detach(&self) -> anyhow::Result<()> {
        for replica in self.replicas() {
            replica.detach().await?;
        }
        Ok(())
    }

    pub(crate) async fn quiesce(&self) -> Quiescence<'_> {
        let deployment = self.replicas.replacing.lock().await;
        let mut replicas = Vec::new();
        for replica in self.replicas() {
            replicas.push(Arc::clone(&replica.replacing).lock_owned().await);
        }
        Quiescence {
            _deployment: deployment,
            _replicas: replicas,
        }
    }

    pub(crate) fn engine_handoff(&self) -> anyhow::Result<Vec<EngineHandoff>> {
        // A handover passes the serving engine, its world and the
        // configuration to the successor, not the update in progress: an
        // epoch switch's children and phases or a program trial's record do
        // not cross it, so the handover waits for the update to end
        // (`specs/worker.md` §7.5, one update at a time).
        if let Some(update) = crate::updates::in_flight(
            self.updates.running(),
            self.epochs().program_update().is_some(),
        ) {
            anyhow::bail!("{update}: a daemon handover waits until the update ends");
        }
        self.replicas()
            .iter()
            .map(|replica| replica.engine_handoff())
            .collect()
    }

    /// The deployment's aggregate serving state.
    pub fn status(&self) -> LeaderStatus {
        let replicas = self.replicas();
        let states: Vec<_> = replicas.iter().map(|replica| replica.status()).collect();
        let mut state = states[0].clone();
        if states.iter().any(|state| state.engine == "ready") {
            state.engine = "ready".to_owned();
            state.reason = None;
        }
        state.paused = states.iter().all(|state| state.paused);
        state.asleep = states.iter().all(|state| state.asleep);
        state.replacing = self.replicas.replacing.try_lock().is_err()
            || states.iter().any(|state| state.replacing);
        state.in_flight = states.iter().map(|state| state.in_flight).sum();
        state
    }

    /// Holds new requests across all replicas.
    pub fn pause(&self) -> bool {
        self.replicas()
            .iter()
            .map(|replica| replica.pause())
            .fold(true, |all, paused| all & paused)
    }

    /// Lets new requests reach every ready replica.
    pub fn resume(&self) {
        for replica in self.replicas() {
            replica.resume();
        }
    }

    pub async fn sleep(&self) -> Result<Vec<String>, UpdateError> {
        let _replacing = self.replicas.replacing.lock().await;
        let mut results = Vec::new();
        for replica in self.replicas() {
            results.extend(replica.sleep().await?);
        }
        Ok(results)
    }

    pub async fn wake(&self) -> Result<Vec<String>, UpdateError> {
        let _replacing = self.replicas.replacing.lock().await;
        let mut results = Vec::new();
        for replica in self.replicas() {
            results.extend(replica.wake().await?);
        }
        Ok(results)
    }

    pub async fn hook(&self, name: &str, payload: &str) -> Result<Vec<String>, UpdateError> {
        let mut results = Vec::new();
        for replica in self.replicas() {
            results.extend(replica.hook(name, payload).await?);
        }
        Ok(results)
    }

    /// The LoRA adapter named `name` the workers hold, which a request
    /// naming it as its model computes with.
    pub fn adapter(&self, name: &str) -> Option<crate::input::AdapterSelection> {
        locked(&self.replicas.adapters)
            .get(name)
            .map(|adapter| crate::input::AdapterSelection {
                slot: adapter.slot,
                identity: adapter.identity.clone(),
            })
    }

    /// The names of the LoRA adapters the workers hold.
    pub fn adapter_names(&self) -> Vec<String> {
        locked(&self.replicas.adapters).keys().cloned().collect()
    }

    /// Loads the LoRA adapter `body` names (`lora_name`, and its `lora_path`)
    /// into a free slot, as vLLM's `load_lora_adapter` does: every rank
    /// publishes its tensors at the Storage Server under the slot (`adapter`).
    /// A new engine, after a hot update or a relaunch, loads it again before
    /// it serves. What computes with an adapter is a kernel program, so a
    /// request naming it is refused until one serves (`docs/admin.md`).
    ///
    /// # Errors
    ///
    /// Refuses a body without a name, the served model's name and an
    /// adapter loaded already; fails when no engine serves, or with the
    /// reason a worker gave.
    pub async fn load_adapter(
        &self,
        mut body: serde_json::Value,
    ) -> Result<Vec<String>, UpdateError> {
        let name = body["lora_name"]
            .as_str()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| UpdateError::Refused("load_lora_adapter names no lora_name".to_owned()))?
            .to_owned();
        if name == self.served_model_name {
            return Err(UpdateError::Refused(format!(
                "{name} is the served model's name"
            )));
        }
        let _deployment = self.replicas.replacing.lock().await;
        let replicas = self.replicas();
        let mut owners = Vec::with_capacity(replicas.len());
        let mut clients = Vec::with_capacity(replicas.len());
        for replica in &replicas {
            owners.push(Arc::clone(&replica.replacing).lock_owned().await);
            clients.push(replica.hold.client().ok_or(UpdateError::NoEngine)?);
        }
        let slot = {
            let adapters = locked(&self.replicas.adapters);
            if adapters.contains_key(&name) {
                return Err(UpdateError::Refused(format!(
                    "the adapter {name} is loaded already"
                )));
            }
            (1..)
                .find(|slot| adapters.values().all(|adapter| adapter.slot != *slot))
                .unwrap_or(u32::MAX)
        };
        body["adapter"] = slot.into();
        let payload = body.to_string();
        let mut results = Vec::new();
        for client in &clients {
            match client.hook("load_lora_adapter", &payload).await {
                Ok(loaded) => results.extend(loaded),
                Err(error) => {
                    // The name becomes public only after every replica loads.
                    // Release any partial loads before returning the failure.
                    for client in &clients {
                        if let Err(cleanup) = client.hook("unload_lora_adapter", &payload).await {
                            warn!(%cleanup, adapter = name, "could not release a partial adapter load");
                        }
                    }
                    return Err(UpdateError::Failed(anyhow::anyhow!(
                        "load_lora_adapter: {error}"
                    )));
                }
            }
        }
        let identity = format!("{name}#{}", uuid::Uuid::new_v4().simple());
        locked(&self.replicas.adapters).insert(
            name.clone(),
            Adapter {
                slot,
                identity,
                payload,
            },
        );
        info!(adapter = name, slot, "loaded the adapter");
        Ok(results)
    }

    /// Unloads the LoRA adapter `body` names (`lora_name`): requests stop
    /// naming it at once, and every rank releases the adapter's slot.
    ///
    /// # Errors
    ///
    /// Refuses a name no loaded adapter has; fails when no engine serves, or
    /// with the reason a worker gave, after which the adapter remains unavailable to requests.
    pub async fn unload_adapter(
        &self,
        mut body: serde_json::Value,
    ) -> Result<Vec<String>, UpdateError> {
        let name = body["lora_name"].as_str().unwrap_or_default().to_owned();
        let _deployment = self.replicas.replacing.lock().await;
        let replicas = self.replicas();
        let mut owners = Vec::with_capacity(replicas.len());
        let mut clients = Vec::with_capacity(replicas.len());
        for replica in &replicas {
            owners.push(Arc::clone(&replica.replacing).lock_owned().await);
            clients.push(replica.hold.client().ok_or(UpdateError::NoEngine)?);
        }
        let adapter = locked(&self.replicas.adapters)
            .remove(&name)
            .ok_or_else(|| UpdateError::Refused(format!("no adapter {name:?} is loaded")))?;
        body["adapter"] = adapter.slot.into();
        let mut results = Vec::new();
        let mut failures = Vec::new();
        for client in clients {
            match client.hook("unload_lora_adapter", &body.to_string()).await {
                Ok(unloaded) => results.extend(unloaded),
                Err(error) => failures.push(error),
            }
        }
        // A partially unloaded adapter must remain unavailable to requests.
        if !failures.is_empty() {
            return Err(UpdateError::Failed(anyhow::anyhow!(
                "unload_lora_adapter: {}",
                failures.join("; ")
            )));
        }
        info!(adapter = name, slot = adapter.slot, "unloaded the adapter");
        Ok(results)
    }

    pub async fn update_weights(
        &self,
        model_path: &str,
        in_flight: WeightsInFlight,
    ) -> Result<Vec<String>, UpdateError> {
        let _replacing = self.replicas.replacing.lock().await;
        let mut results = Vec::new();
        for replica in self.replicas() {
            results.extend(replica.update_weights(model_path, in_flight).await?);
        }
        Ok(results)
    }

    pub fn flush_cache(&self) -> Result<(), UpdateError> {
        for replica in self.replicas() {
            replica.flush_cache()?;
        }
        Ok(())
    }

    /// Loads the configured model and starts the engine process that serves
    /// it, with its schedulers and worker processes.
    ///
    /// `config` must have passed [`ServerArgs::validate`]; `stated` holds the
    /// option paths the operator set; `daemon` is this node's daemon's
    /// process socket, through which the engine process and every worker
    /// start. The engine's start blocks until every rank has loaded the
    /// model, so it runs on the blocking pool.
    ///
    /// # Errors
    ///
    /// Fails when the model assets cannot be resolved, when an option of
    /// another model family was stated ([`Stated::refuse_foreign_options`]),
    /// when the options do not resolve into an engine configuration
    /// ([`ServerArgs::engine_config`]), when the engine process fails to
    /// start, or when the model cannot be bound to the capabilities its
    /// workers report.
    pub async fn start(
        config: &ServerArgs,
        stated: &Stated,
        daemon: String,
        commands: Option<&crate::supervision::ProcessCommands>,
        updates: Arc<crate::updates::driver::Driver>,
    ) -> anyhow::Result<Self> {
        let prepared = Self::prepare(config, stated).await?;
        info!(parallel = ?prepared.engine.parallel, "starting the engine process");
        let control = Arc::new(EngineControl::new(
            &config.request_socket(),
            daemon,
            config.host_identity.clone(),
            config.served_model_name().to_owned(),
            config.program_server.clone(),
        )?);
        let replicas = replica_configs(&prepared.engine)?;
        let clients =
            futures::future::join_all(replicas.into_iter().enumerate().map(|(index, replica)| {
                let model = config.model().to_owned();
                let starting = Arc::clone(&control);
                tokio::task::spawn_blocking(move || {
                    let observer = Arc::new(crate::observe::Observer::default());
                    let sink = stats_sink(model, Arc::clone(&observer), index);
                    let mut args = proto::config::EngineArgs::new(replica);
                    let client = starting.start(args.clone(), sink)?;
                    args.world = client.world();
                    Ok::<_, anyhow::Error>((client, args, observer))
                })
            }))
            .await
            .into_iter()
            .map(|started| started.context("the engine startup task did not complete")?)
            .collect::<anyhow::Result<Vec<_>>>()
            .context("failed to start the UniServe engine")?;
        // The main processes this node started, by the engine number their
        // reports carry, for the epoch's first program switch.
        let engines: Vec<u64> = clients
            .iter()
            .filter_map(|(client, _, _)| client.engine())
            .collect();
        let world = clients.first().map_or(0, |(client, _, _)| client.world());
        let mut clients = clients.into_iter();
        let (client, served, observer) = clients
            .next()
            .context("the engine configuration starts no replicas")?;
        let leader = Self::finish(
            config,
            prepared,
            Arc::clone(&control),
            client,
            served,
            observer,
            updates,
        )?;
        for (index, (client, served, observer)) in clients.enumerate() {
            let replica = Replica::new(
                config,
                Arc::new(client),
                observer,
                index + 1,
                Arc::downgrade(&leader.replicas),
                Arc::clone(&leader.replicas.adapters),
            );
            let _ = replica.control.set(Arc::clone(&control));
            *locked(&replica.served) = Some(served);
            replica.monitor();
            leader
                .replicas
                .serving
                .send_modify(|replicas| replicas.push(replica));
        }
        if let Some(commands) = commands {
            // The epoch's record commands the same main processes the
            // daemon's channel reaches, for every later program-layer
            // update (§2.3, §7.3).
            leader.epochs().supervise(commands.clone());
            leader
                .install_programs(commands, config, engines, world)
                .await
                .context("installing the epoch's programs in its main processes")?;
            // A committed world's tensors are what the next world starts from: until it is
            // published its storage is private to it, and a target of another world finds nothing.
            let publishing = Arc::clone(&control);
            tokio::task::spawn_blocking(move || publishing.publish_worlds(&[world], &mut None))
                .await
                .context("publishing the first world")?
                .context("publishing the first world")?;
            // The epoch's programs serve, which is what the standbys were waiting to prepare from.
            for replica in leader.replicas() {
                replica.prepare_standby();
            }
        }
        Ok(leader)
    }

    /// The programs `config` names for what this deployment launches.
    ///
    /// The configuration selects a backend for attention and for the grouped experts, which are a
    /// causal language model's operators. A video deployment launches neither, so a slot filled for
    /// them would pin a program no call reaches: every operator it runs is filled from the store.
    fn named_programs(&self, config: &ServerArgs) -> crate::epoch::FilledPrograms {
        let mut filled = crate::epoch::FilledPrograms::of(config);
        if self.video_denoiser().is_some() {
            filled.kernels.clear();
        }
        filled
    }

    /// Prepares the epoch's programs and switches this node's main processes
    /// to them (§7.3 steps 2-3, §3): a main process reports ready without
    /// waiting for warmup and serves only once the switchover has installed
    /// the epoch's programs, the scheduling policy among them, so the daemon
    /// switches every main process once before it accepts a request.
    ///
    /// The generation is each main process's own, as its `Precopied` names
    /// it, and the first epoch is committed at once: it has no previous
    /// generation to keep as a last resort, so there is nothing to purge.
    ///
    /// # Errors
    ///
    /// Refuses when a Program Server cannot select the epoch's programs, and
    /// fails with the first main process that refuses or does not answer.
    async fn install_programs(
        &self,
        commands: &crate::supervision::ProcessCommands,
        config: &ServerArgs,
        engines: Vec<u64>,
        world: u64,
    ) -> anyhow::Result<()> {
        let filled = self.named_programs(config);
        // The startup switch installs the epoch's programs as its record names
        // them: the scheduling policy *and* the kernel of every operator the
        // deployment selects. A main process resolves no kernel itself — its
        // backends are the installed programs, and a kernel it was never given
        // fails the first request that needs it — so an epoch that prepared the
        // policy alone serves its first token with nothing to run.
        //
        // A Program Server without a program for one of those operators refuses
        // the epoch by name here, before it serves, which is the correct
        // refusal: the deployment selects a backend whose package is not
        // installed.
        let servers = self.program_servers(config);
        let leader_host = self.program_leader_host(config, &servers);
        // Each step of the install says so before it waits: a deployment that
        // hangs here is otherwise silent for as long as the run lasts, and the
        // last line printed is the only thing that names where it waited.
        tracing::info!(
            world,
            engines = ?engines,
            kernels = filled.kernels.len(),
            policy = %filled.policy.backend,
            leader = %leader_host,
            servers = ?servers
                .iter()
                .map(|server| format!("{} at {}", server.host, server.socket.display()))
                .collect::<Vec<_>>(),
            "preparing the epoch's programs at its Program Servers"
        );
        let prepared = {
            let servers = servers.clone();
            let leader_host = leader_host.clone();
            let filled = filled.clone();
            // The deadline below names the wait, so the blocking task records
            // which of the client's two unbounded waits it is in.
            let steps = crate::engine::control::PrepareSteps::new();
            let recorded = steps.clone();
            let preparing = tokio::task::spawn_blocking(move || {
                crate::engine::control::prepare_epoch(
                    &servers,
                    &leader_host,
                    &filled,
                    &[],
                    &recorded,
                )
            });
            let started = tokio::time::Instant::now();
            let deadline = started + PREPARE_TIMEOUT;
            let mut preparing = preparing;
            loop {
                let left = deadline.saturating_duration_since(tokio::time::Instant::now());
                match tokio::time::timeout(left.min(HEARTBEAT), &mut preparing).await {
                    Ok(joined) => break joined.context("the epoch's programs")?,
                    Err(_) if left <= HEARTBEAT => anyhow::bail!(
                        "the epoch's Program Servers selected nothing within {} s: \
                         the daemon was {}",
                        PREPARE_TIMEOUT.as_secs(),
                        steps.describe()
                    ),
                    Err(_) => tracing::info!(
                        waited_s = started.elapsed().as_secs(),
                        step = %steps.describe(),
                        "still preparing the epoch's programs"
                    ),
                }
            }
            .context("preparing the epoch's programs")?
        };
        tracing::info!(
            world,
            kernels = prepared.kernels.len(),
            policy = %prepared.policy.backend,
            "the epoch's programs are resolved; sending them to the main processes"
        );
        // The record states the programs the main processes took, revisions
        // included — the ones `prepare_epoch` selected — and not the ones the
        // configuration asked for: a filled program's revision is zero until a
        // Program Server selects one, so recording the request would leave
        // `GET /admin/config` naming numbers no package has and a later
        // handover to a joining process (`Epochs::hand_to`) sending a revision
        // the store never built.
        let deployment = crate::epoch::Deployment::with_programs(config.clone(), prepared.clone())?;
        let processes: Vec<crate::engine::control::EpochProcess> = engines
            .into_iter()
            .map(|engine| crate::engine::control::EpochProcess {
                host: config.host_identity.clone(),
                engine,
            })
            .collect();
        let commands = commands.clone();
        let installed = deployment.clone();
        let switching = tokio::task::spawn_blocking(move || {
            let mut switch = crate::engine::control::ProgramSwitch::new(
                &commands,
                processes,
                crate::epoch::Update::new(0, installed),
            );
            // The epoch's programs, kernels and policy alike: the main
            // processes serve their first request with them.
            switch.precopy(&prepared.kernels, &prepared.policy)?;
            switch.switchover()?;
            switch.commit()?;
            Ok::<_, crate::epoch::Failure>(switch.generations().to_vec())
        });
        let generations = match tokio::time::timeout(SWITCH_TIMEOUT, switching).await {
            Ok(joined) => joined.context("switching the main processes to the epoch's programs")?,
            Err(_) => anyhow::bail!(
                "the main processes took the epoch's programs in neither Precopy, \
                 Switchover nor Commit within {} s (the last step logged above is \
                 where the daemon waited)",
                SWITCH_TIMEOUT.as_secs()
            ),
        }
        .map_err(|failure| {
            anyhow::anyhow!(
                "the main processes did not take the epoch's programs ({}): {}",
                failure.kind,
                failure.reason
            )
        })?;
        // The daemon's record of what executes: the epoch's build config and
        // the programs its main processes took, with the number they gave
        // them (`EpochRecord::generation`), which `GET /admin/config`
        // reports and a later update is routed against (§7.2, §7.4).
        // The first epoch of a daemon is epoch 0 (§7.4).
        let epoch = self.epochs().install(0, &deployment, world, &generations);
        // The process table names every main process this daemon manages, and
        // the serving one is the first (§7.4): a switch addresses the process it
        // replaces by this generation, `GET /admin/config` lists it, and without
        // the row a deployment started here looks like one with no main process
        // at all — which is what an epoch switch refuses for.
        if let Some(engine) = self
            .replicas
            .serving
            .borrow()
            .first()
            .and_then(|replica| replica.hold().client())
            .and_then(|client| client.engine())
        {
            let mut processes = self.epochs().processes();
            processes.retain(|process| process.generation != engine);
            processes.push(crate::epoch::ManagedProcess {
                role: crate::epoch::Role::MainProcess,
                group: crate::engine::control::ENGINE_GROUP.to_owned(),
                rank: 0,
                pid: None,
                epoch: Some(world),
                build: deployment.engine.executable.clone(),
                generation: engine,
                state: crate::epoch::ProcessState::Serving,
            });
            self.epochs().set_processes(processes);
        }
        tracing::info!(
            epoch,
            world,
            generations = ?generations,
            "the main processes serve the epoch's programs"
        );
        Ok(())
    }

    /// What a leader prepares before it has an engine: the model's files
    /// and the engine configuration of the engine that serves it, as
    /// [`Leader::start`] describes.
    ///
    /// # Errors
    ///
    /// As [`Leader::start`], before the engine starts.
    pub(crate) async fn prepare(config: &ServerArgs, stated: &Stated) -> anyhow::Result<Prepared> {
        let LoadedModel {
            config: model_config,
            tokenizer,
            renderer,
        } = ModelConfig::load(config)
            .await
            .with_context(|| format!("failed to resolve model assets for `{}`", config.model()))?;
        // Options of another model family than the served one are refused
        // rather than ignored.
        stated.refuse_foreign_options(model_config.description())?;
        let max_model_len = model_config.max_model_len;

        // The options state the scheduler and the rank layout; the loaded model
        // completes what it determines.
        let mut engine = config.engine_config()?;
        let control_tokens = special_token_ids(&model_config);
        engine.model.facts = Some(proto::config::ModelFacts {
            limits: model_config.generation_limits(config.dtype),
            bos: control_tokens.bos,
            eos: control_tokens.eos,
            end_of_image: control_tokens.end_of_image,
            max_model_len,
            // No model this daemon serves generates canvases: canvas sampling
            // reaches a worker with the request that needs it, and every
            // prefill returns the outputs its caller asked for.
            canvas_sampling: None,
            prefill_outputs: true,
        });
        // Video workers prepare exactly the canvases their denoiser names for
        // the configured resolutions and aspect ratios, refusing a pair it
        // does not generate, and report them back as the only canvases
        // requests may target. Other models prepare none.
        if !matches!(model_config.parameters, ModelParameters::MiniMaxH3) {
            engine.video.resolutions.clear();
            engine.video.aspect_ratios.clear();
        }

        Ok(Prepared {
            model_config,
            tokenizer,
            renderer,
            engine,
            max_model_len,
        })
    }

    /// The leader of `prepared`'s model over the engine `client` serves,
    /// which `control` started (or adopted) with the arguments `served`.
    ///
    /// # Errors
    ///
    /// Fails when the model cannot be bound to the capabilities the
    /// engine's workers report.
    pub(crate) fn finish(
        config: &ServerArgs,
        prepared: Prepared,
        control: Arc<EngineControl>,
        client: EngineClient,
        served: proto::config::EngineArgs,
        observer: Arc<crate::observe::Observer>,
        updates: Arc<crate::updates::driver::Driver>,
    ) -> anyhow::Result<Self> {
        let Prepared {
            model_config,
            tokenizer,
            renderer,
            engine: _,
            max_model_len: _,
        } = prepared;
        let model_name = config.model().to_owned();
        let info = client.info().clone();

        // A video checkpoint serves what its placed denoiser declares in the
        // worker handshake, planned against the conditioner's vision
        // processor.
        let renderer = match renderer {
            LoadedRenderer::Chat { renderer, .. } => {
                // No other served model's processor takes options.
                anyhow::ensure!(
                    config.mm_processor_kwargs.is_empty(),
                    "invalid mm_processor_kwargs: the processor of `{}` takes no options",
                    config.model()
                );
                RequestRenderer::Chat(*renderer)
            }
            LoadedRenderer::Video(vision) => {
                anyhow::ensure!(
                    !info.video_denoiser.is_empty(),
                    "the video worker reported no video denoiser"
                );
                let denoiser = models::video::DenoiserInfo::decode(&info.video_denoiser)
                    .context("the video worker described its denoiser unreadably")?;
                denoiser
                    .validate()
                    .map_err(|message| anyhow::anyhow!("the video worker's denoiser: {message}"))?;
                // Reference images are banded to the units one latent
                // encoding round covers on the deployment's latent encoder
                // while a request's prompt is encoded.
                let latent_encoding_units = info.latent_encoding_units.max(1);
                // The video processor's options are its reference sizing,
                // within what the denoiser declares.
                let references = models::video::plan::ReferenceConfig::from_kwargs(
                    &config.mm_processor_kwargs,
                    &denoiser.references,
                )
                .map_err(|message| anyhow::anyhow!("invalid mm_processor_kwargs: {message}"))?;
                RequestRenderer::Video(
                    VideoService::new(
                        denoiser,
                        vision,
                        config.video_config.max_seconds,
                        latent_encoding_units,
                        &config.media_config,
                        references,
                        Arc::clone(&tokenizer),
                    )
                    .context("failed to bind the video denoiser")?,
                )
            }
        };

        let processor = InputProcessor::new(
            model_config,
            tokenizer,
            renderer,
            info.limits.clone(),
            config.chat_config.reasoning_parser == ReasoningParser::Auto,
        )
        .context("failed to bind the configured model description")?;
        let images = MediaConnector::from_image_options(&config.image_fetch_config)
            .context("failed to initialize the image URL fetcher")?;
        let engine = Self::observed(
            config.served_model_name().to_owned(),
            model_name,
            vec![Arc::new(client)],
            processor,
            images,
            config,
            observer,
            updates,
        );
        let replica = Arc::clone(&engine.replicas.serving.borrow()[0]);
        let _ = replica.control.set(control);
        *locked(&replica.served) = Some(served);
        Ok(engine)
    }

    /// Serves `processor`'s model under `served_model_name` over the
    /// independent replicas in `clients`, labelling metrics with `model_name`.
    /// `images` resolves the image references of every request, and `options`
    /// states the request and statistics logging (`logging_config`) and
    /// request migration (`migration_config`).
    pub fn new(
        served_model_name: String,
        model_name: String,
        clients: Vec<Arc<EngineClient>>,
        processor: InputProcessor,
        images: MediaConnector,
        options: &ServerArgs,
        updates: Arc<crate::updates::driver::Driver>,
    ) -> Self {
        Self::observed(
            served_model_name,
            model_name,
            clients,
            processor,
            images,
            options,
            Arc::default(),
            updates,
        )
    }

    /// As [`Self::new`], with `observer` reading the engine's statistics
    /// for its world's trials.
    #[expect(
        clippy::too_many_arguments,
        reason = "a leader's model, engine and options, and its statistics' observer"
    )]
    fn observed(
        served_model_name: String,
        model_name: String,
        clients: Vec<Arc<EngineClient>>,
        processor: InputProcessor,
        images: MediaConnector,
        options: &ServerArgs,
        observer: Arc<crate::observe::Observer>,
        updates: Arc<crate::updates::driver::Driver>,
    ) -> Self {
        let logging = &options.logging_config;
        let mut info = clients[0].info().clone();
        info.schedulers = clients.iter().map(|client| client.info().schedulers).sum();
        let schedulers = Arc::new(std::sync::atomic::AtomicUsize::new(
            (info.schedulers as usize).max(options.data_parallel_size.get()),
        ));
        let stats_logger = logging
            .log_stats
            .then(|| StatsLogger::start(model_name.clone(), Arc::clone(&schedulers)));
        let replica_options = ServerArgs {
            model: Some(model_name.clone()),
            served_model_name: Some(served_model_name.clone()),
            ..options.clone()
        };
        let adapters = Arc::new(std::sync::Mutex::new(BTreeMap::new()));
        let replicas = Arc::new_cyclic(|registry| Replicas {
            serving: watch::Sender::new(
                clients
                    .into_iter()
                    .enumerate()
                    .map(|(index, client)| {
                        Replica::new(
                            &replica_options,
                            client,
                            Arc::clone(&observer),
                            index,
                            std::sync::Weak::clone(registry),
                            Arc::clone(&adapters),
                        )
                    })
                    .collect(),
            ),
            replacing: tokio::sync::Mutex::new(()),
            adapters: Arc::clone(&adapters),
            schedulers,
            updates: Arc::downgrade(&updates),
        });
        for replica in replicas.serving.borrow().iter() {
            replica.monitor();
        }
        Self {
            served_model_name,
            model_name,
            processor: Arc::new(processor),
            info,
            replicas,
            images,
            updates,
            log_requests: logging.log_requests,
            _stats_logger: stats_logger,
        }
    }

    /// The loaded model's input processor.
    pub fn processor(&self) -> &InputProcessor {
        &self.processor
    }

    /// The single model name the API serves.
    pub fn served_model_name(&self) -> &str {
        &self.served_model_name
    }

    /// The generation limits the engine serves.
    pub fn generation_limits(&self) -> GenerationLimits {
        self.info.limits.clone()
    }

    /// The video denoiser this deployment serves; `None` for a model that
    /// generates no video.
    pub fn video_denoiser(&self) -> Option<&models::video::DenoiserInfo> {
        self.processor.video_service().map(VideoService::denoiser)
    }

    /// The tasks this engine serves.
    ///
    /// A video model serves video generation alone; every other model serves
    /// chat completions, and a model that generates images serves image
    /// generation as well.
    pub fn supported_tasks(&self) -> Vec<Task> {
        let mut tasks = Vec::new();
        if self.processor.video_service().is_some() {
            tasks.push(Task::VideoGeneration);
        } else {
            tasks.push(Task::Generate);
        }
        tasks
    }

    /// Whether the engine can accept requests: whether a ready engine
    /// serves them.
    pub fn is_healthy(&self) -> bool {
        self.replicas
            .serving
            .borrow()
            .iter()
            .any(|replica| replica.hold.is_serving())
    }

    /// The connector every request's image references go through.
    pub(crate) fn images(&self) -> &MediaConnector {
        &self.images
    }

    /// Fetches the input images of a generation request, renders and
    /// tokenizes it, and streams its output.
    ///
    /// A chat prompt's `image_url` parts are fetched only for a model that
    /// accepts image input; for any other model, preprocessing rejects them
    /// without network access.
    ///
    /// # Errors
    ///
    /// Fails when an image cannot be fetched, preprocessing refuses the
    /// request, or no scheduler accepts it.
    pub async fn generate(
        &self,
        request: GenerateRequest,
    ) -> Result<RequestOutputStream, ServeError> {
        // Compilation, and with it the request's latency, is measured from
        // here, so image fetching and time queued for the blocking pool
        // count toward it.
        let started = Instant::now();
        let request_id = request.request_id.clone();
        let image_urls = match &request.prompt {
            Prompt::Chat(chat) if self.processor.supports_image_input() => chat.image_urls(),
            _ => Vec::new(),
        };
        let images = self
            .images
            .fetch_images(image_urls)
            .await
            .map_err(|error| ServeError::ImageInput {
                request_id: request_id.clone(),
                index: error.index,
                source: error.source,
            })?;
        let processor = Arc::clone(&self.processor);
        let (request, output) =
            preprocess(&request_id, move || processor.preprocess(request, images)).await?;

        let compile_us = elapsed_us(started);
        let (events, core) = self.submit(request.into()).await?;
        let output = crate::output::generation_output(
            request_id.clone(),
            started,
            compile_us,
            output,
            events,
        );
        Ok(self.stats(request_id, started, core, output))
    }

    /// Submits one generation request and returns what its handoffable body
    /// is built from: the request's engine events, the pipeline that produces
    /// its output, how long it took to reach the engine, and the recorder of
    /// its statistics.
    ///
    /// The caller drives the body one engine event at a time
    /// (`output::run::GenerationRun`), so the response can stop at an engine
    /// event boundary and go on in a successor (design `specs/worker.md`
    /// §7.5.1).
    ///
    /// # Errors
    ///
    /// Fails as [`Self::generate`] does.
    pub(crate) async fn generate_handed(
        &self,
        request: GenerateRequest,
    ) -> Result<
        (
            RequestEvents,
            crate::output::OutputParams,
            u64,
            crate::metrics::stats::RequestOutcome,
        ),
        ServeError,
    > {
        let started = Instant::now();
        let request_id = request.request_id.clone();
        let image_urls = match &request.prompt {
            Prompt::Chat(chat) if self.processor.supports_image_input() => chat.image_urls(),
            _ => Vec::new(),
        };
        let images = self
            .images
            .fetch_images(image_urls)
            .await
            .map_err(|error| ServeError::ImageInput {
                request_id: request_id.clone(),
                index: error.index,
                source: error.source,
            })?;
        let processor = Arc::clone(&self.processor);
        let (request, output) =
            preprocess(&request_id, move || processor.preprocess(request, images)).await?;

        let compile_us = elapsed_us(started);
        let (events, core) = self.submit(request.into()).await?;
        let labels = EngineLabels {
            model_name: self.model_name.clone(),
            engine: core,
        };
        let log = self.log_requests.then_some(request_id);
        let outcome = crate::metrics::stats::RequestOutcome::new(labels, started, log);
        Ok((events, output, compile_us, outcome))
    }

    /// The model's tokenizer, which a successor builds a handed-over body's
    /// decoder from.
    pub(crate) fn tokenizer(&self) -> crate::tokenizer::DynTokenizer {
        self.processor.tokenizer()
    }

    /// The statistics recorder of a request this process takes over, under
    /// the replica that serves it and the instant it entered the deployment.
    pub(crate) fn request_outcome(
        &self,
        request_id: &str,
        replica: u32,
        started: std::time::Instant,
    ) -> crate::metrics::stats::RequestOutcome {
        let labels = EngineLabels {
            model_name: self.model_name.clone(),
            engine: replica,
        };
        let log = self.log_requests.then(|| request_id.to_owned());
        crate::metrics::stats::RequestOutcome::new(labels, started, log)
    }

    /// Takes one request's engine event stream over from the process that
    /// consumed it, going on after the position it reached (design §7.5.1
    /// step 4): the request keeps generating in the engine that serves it,
    /// and is neither cancelled nor submitted again. Both handed kinds of
    /// response call it, with the replica and identity their state carries.
    ///
    /// # Errors
    ///
    /// Fails when the state names no replica or no engine identity, that
    /// replica serves no requests, or the engine refuses the takeover.
    pub(crate) fn take_over_request(
        &self,
        engine_replica: u32,
        engine_request_id: &str,
        consumed: u64,
    ) -> anyhow::Result<RequestEvents> {
        let replica = self
            .replicas()
            .get(engine_replica as usize)
            .cloned()
            .with_context(|| format!("the handed response names replica {engine_replica}"))?;
        let client = replica
            .hold
            .client()
            .context("the handed response's replica has no engine connection")?;
        let id = proto::request::RequestId(
            engine_request_id
                .parse()
                .context("the handed response's engine identity")?,
        );
        let stream = client.take_over(id, consumed)?;
        Ok(RequestEvents::taken_over(stream, consumed))
    }

    /// Fetches, probes and plans a video request's media, presents its
    /// prompt, and streams its output.
    ///
    /// Also returns what the request generates: its duration, frames and
    /// canvas.
    ///
    /// # Errors
    ///
    /// Fails when the model serves no video, the video service refuses the
    /// request, or no scheduler accepts it.
    pub async fn generate_video(
        &self,
        request_id: String,
        request: VideoGenerationRequest,
    ) -> Result<(ProcessedVideo, RequestOutputStream), ServeError> {
        let started = Instant::now();
        let video =
            self.processor
                .video_service()
                .ok_or_else(|| ServeError::UnsupportedFeature {
                    request_id: request_id.clone(),
                    feature: "video_generation",
                })?;
        let (request, processed, media) = video
            .process(&request_id, &request, self.processor.config().max_model_len)
            .await?;
        let prompt_tokens = request.prompt_token_ids.len();
        let (events, core) = self.submit(Request::Diffusion(request)).await?;
        let output = crate::output::media_output(
            request_id.clone(),
            started,
            elapsed_us(started),
            prompt_tokens,
            events,
        );
        // The workers read the condition media while the request runs, which
        // ends with its event stream.
        let output: RequestOutputStream = Box::pin(output.inspect(move |_| {
            let _ = &media;
        }));
        Ok((processed, self.stats(request_id, started, core, output)))
    }

    /// Records `output`'s request metrics under the scheduler that serves
    /// it.
    fn stats(
        &self,
        request_id: String,
        started: Instant,
        core: u32,
        output: RequestOutputStream,
    ) -> RequestOutputStream {
        let labels = EngineLabels {
            model_name: self.model_name.clone(),
            engine: core,
        };
        let log = self.log_requests.then_some(request_id);
        Box::pin(RequestStats::new(output, labels, started, log))
    }

    /// Submits one tokenized request to the engine, which routes it to its
    /// least loaded live scheduler; while no engine is ready, the request
    /// waits in the hold until one is.
    ///
    /// Returns the request's event stream and the index of the scheduler
    /// that serves it. Dropping the stream before its terminal event cancels
    /// the request.
    ///
    /// # Errors
    ///
    /// Fails when the engine serves no requests.
    pub(crate) async fn submit(
        &self,
        request: Request,
    ) -> Result<(RequestEvents, u32), SubmitError> {
        let replica = self.replica_for_request().await?;
        let (events, scheduler) = replica.migrator.submit(request).await?;
        // A request the engine ended at submission names no scheduler; its
        // stream holds its terminal event.
        Ok((events, scheduler.unwrap_or(0) + replica.ordinal as u32))
    }
}

impl Leader {
    /// The epochs this node keeps: the record the admin API reports and a
    /// program-layer update lives in (§7.2, §7.3).
    pub(crate) fn epochs(&self) -> Arc<crate::epoch::Epochs> {
        // One record for the daemon: the epoch layer's driver owns it, the
        // program layer and the admin API read it (§7.2, §7.3, §7.4).
        Arc::clone(self.updates.epochs())
    }

    /// The configuration the deployment serves now: what a change is
    /// resolved against before it is routed (§7.2).
    ///
    /// # Errors
    ///
    /// `UpdateError::Failed` when no replica serves — the same answer
    /// [`Self::apply_programs`] gives, so a caller may ask either first.
    pub(crate) fn serving_config(&self) -> Result<ServerArgs, UpdateError> {
        self.replicas()
            .first()
            .map(|replica| locked(&replica.config).clone())
            .ok_or_else(|| UpdateError::Failed(anyhow::anyhow!("no replica serves")))
    }

    /// This node's main processes, by the engine number their reports carry
    /// and the node they run on: the processes a program-layer update
    /// commands (§7.3).
    pub(crate) fn main_processes(&self) -> Vec<crate::engine::control::EpochProcess> {
        self.replicas()
            .iter()
            .filter_map(|replica| {
                let engine = replica.hold.client()?.engine()?;
                let host = locked(&replica.config).host_identity.clone();
                Some(crate::engine::control::EpochProcess { host, engine })
            })
            .collect()
    }

    /// Routes `target` and applies its program-layer change to this node's
    /// main processes: the target's programs are prepared at every host's
    /// Program Server, precopied into each main process, switched to, and
    /// recorded as the epoch's probation (§7.3 steps 2-4).
    ///
    /// Answers the update that is on probation. `Ok(None)` when the change
    /// moves nothing at the program layer — it belongs to the host layer or
    /// the epoch layer, or fills the programs the epoch already executes —
    /// which is where a caller falls through to the world path.
    ///
    /// With `dry_run`, the routing decision alone is made and nothing is
    /// applied, so a caller can answer what a change would move. `force`
    /// applies an update a revert marked unavailable (§7.2).
    ///
    /// # Errors
    ///
    /// `UpdateError::Refused` when a Program Server cannot select the
    /// target's programs; `UpdateError::Failed` when a main process refuses
    /// the precopy or the switchover, or does not answer. What serves is
    /// untouched either way.
    pub(crate) async fn apply_programs(
        &self,
        number: u64,
        target: &ServerArgs,
        options: &crate::options::UpdateConfig,
        force: bool,
        dry_run: bool,
    ) -> Result<Option<crate::epoch::Update>, UpdateError> {
        let current = self.serving_config()?;
        // One update at a time (§7.2, §7.4): an epoch switch and a program
        // update are each the deployment's one update, and the program
        // layer's callers reach this method directly as well as through the
        // entry, so the check lives here. A dry run answers without touching
        // the processes or the records the update in flight owns, which is
        // what the entry does for the layers above (§7.2).
        if !dry_run
            && let Some(update) = crate::updates::in_flight(
                self.updates.running(),
                self.epochs().program_update().is_some(),
            )
        {
            return Err(UpdateError::Refused(format!(
                "another update is in flight: {update}"
            )));
        }
        // A target that reverted with a failure is unavailable until it is
        // applied with `force` (§7.2, §7.3 step 4), the program layer's as
        // much as the epoch layer's.
        let resolved_target = deployment(target)?;
        if !force && let Some(failure) = self.epochs().unavailable(&resolved_target) {
            return Err(UpdateError::Refused(format!(
                "the configuration reverted with a failure ({}); apply it again with force",
                failure.reason
            )));
        }
        let Some(update) = self.route(&current, target, false).await? else {
            return Ok(None);
        };
        if dry_run {
            return Ok(Some(update));
        }
        let mut update = update;
        update.number = number;
        let epochs = Arc::clone(&self.epochs());
        let processes = self.main_processes();
        let recorded = update.clone();
        let promotion = std::time::Instant::now();
        tokio::task::spawn_blocking(move || epochs.apply_programs(update, processes, force))
            .await
            .map_err(|error| UpdateError::Failed(anyhow::anyhow!("applying the update: {error}")))?
            .map_err(|failure| {
                UpdateError::Failed(anyhow::anyhow!(
                    "the main processes did not take the programs ({}): {}",
                    failure.kind,
                    failure.reason
                ))
            })?;
        // The programs moved in place; a parked standby prepared from the old
        // ones is replaced (§2.5).
        for replica in self.replicas.serving.borrow().iter() {
            replica.restart_standby();
        }
        self.watch_program_trial(number, options, promotion);
        Ok(Some(recorded))
    }

    /// Reads update `number`'s thresholds for as long as it stands on probation in the serving
    /// epoch (§7.3 step 4), and reverts it through the path a main process's failure takes once
    /// one is passed. Serving and the update share a world, so the baseline ends at `promotion`,
    /// when the update began to switch, and the trial's reports begin after the switch.
    fn watch_program_trial(
        &self,
        number: u64,
        options: &crate::options::UpdateConfig,
        promotion: std::time::Instant,
    ) {
        let window = options.rollback.window;
        let observed = self
            .replicas()
            .iter()
            .filter_map(|replica| {
                let world = replica.hold.client()?.world();
                Some(crate::updates::trial::Observed::new(
                    Arc::clone(&replica.observer),
                    world,
                    world,
                    window,
                    promotion,
                ))
            })
            .collect();
        let trial = Arc::new(crate::updates::trial::Trial::begin(
            options.rollback.clone(),
            observed,
            outcomes(),
        ));
        self.updates.begin_trial(Arc::clone(&trial));
        let updates = Arc::clone(&self.updates);
        let epochs = Arc::clone(&self.epochs());
        tokio::spawn(async move {
            let standing = Arc::clone(&epochs);
            let reverting = Arc::clone(&epochs);
            crate::updates::trial::watch(
                &trial,
                outcomes,
                move || {
                    standing
                        .program_update()
                        .is_some_and(|program| program.update.number == number)
                },
                move |failure| async move {
                    warn!(
                        number,
                        kind = %failure.kind,
                        reason = %failure.reason,
                        "the update on trial passed a rollback threshold; it reverts"
                    );
                    let _ = tokio::task::spawn_blocking(move || reverting.program_failure(failure))
                        .await;
                },
            )
            .await;
            updates.end_trial(serde_json::json!({
                "change": "program",
                "update": number,
                "failure": epochs.update().and_then(|update| update.unavailable().map(|failure| {
                    serde_json::json!({ "kind": failure.kind, "reason": failure.reason })
                })),
            }));
        });
    }

    /// Ends update `number`'s program-layer probation: every main process
    /// commits the generation it executes and purges the one it no longer
    /// runs, and the epoch's record keeps no previous generation (§7.3 steps
    /// 5-6).
    ///
    /// Answers `Ok(false)` when update `number` is not a program-layer one,
    /// which is where a caller falls through to the world path.
    ///
    /// # Errors
    ///
    /// Fails when `number` names no update, when it is not the update on
    /// probation, or when a main process does not answer the commit.
    pub(crate) async fn commit_programs(&self, number: u64) -> Result<bool, UpdateError> {
        let program = self
            .epochs()
            .program_update()
            .filter(|program| program.update.number == number);
        let Some(program) = program else {
            return Ok(false);
        };
        let epochs = Arc::clone(&self.epochs());
        let committed = tokio::task::spawn_blocking(move || epochs.commit_programs())
            .await
            .map_err(|error| {
                UpdateError::Failed(anyhow::anyhow!("committing the update: {error}"))
            })?
            .map_err(UpdateError::Failed)?;
        // The configuration is what the next apply starts from and what a successor takes over:
        // one that kept the values before this update would undo it at the next change.
        if committed {
            for replica in self.replicas() {
                *locked(&replica.config) = program.update.target.config.clone();
            }
        }
        Ok(committed)
    }

    /// Returns every main process to the generation it served before update
    /// `number`, with `reason`, and records the update as reverted: it is
    /// unavailable until it is applied with `force` (§7.2, §7.3 step 5).
    ///
    /// Answers `Ok(false)` when update `number` is not a program-layer one.
    ///
    /// # Errors
    ///
    /// Fails when a main process does not answer the revert. What serves is
    /// the previous generation either way.
    pub(crate) async fn rollback_programs(
        &self,
        number: u64,
        reason: &str,
    ) -> Result<bool, UpdateError> {
        let program = self
            .epochs()
            .program_update()
            .filter(|program| program.update.number == number);
        if program.is_none() {
            return Ok(false);
        }
        let epochs = Arc::clone(&self.epochs());
        let reason = reason.to_owned();
        tokio::task::spawn_blocking(move || epochs.revert_programs(&reason))
            .await
            .map_err(|error| UpdateError::Failed(anyhow::anyhow!("reverting the update: {error}")))?
            .map_err(UpdateError::Failed)
    }
}

/// The layer a configuration change moves on, and the update a program-layer
/// change starts.
///
/// A change is routed by what it moves and never by what it says (§7.2): the
/// daemon's and the frontend's own options are the host layer, which a hot
/// restart applies (§7.5); a different build config is the epoch layer,
/// whose programs are prepared when the epoch is built (§7.4); and a change
/// whose build config is the same moves only the programs, which the serving
/// main processes take in place (§7.3).
///
/// The target's programs are prepared at every host's Program Server before
/// anything is compared, so a newly installed revision at an unchanged
/// configuration is a program-layer change: the comparison sees the
/// revisions the Program Servers select, not the ones the serving epoch
/// pinned. `prepare` is blocking I/O, so it runs off the async runtime.
///
/// # Errors
///
/// Refuses a target whose programs a Program Server cannot select or build.
impl Leader {
    pub(crate) async fn route(
        &self,
        current: &ServerArgs,
        target: &ServerArgs,
        force: bool,
    ) -> Result<Option<crate::epoch::Update>, UpdateError> {
        if crate::epoch::host_layer_differ(current, target) {
            return Ok(None);
        }
        let engine = resolved(target)?;
        if resolved(current)?.build_hash() != engine.build_hash() {
            return Ok(None);
        }
        // From here only the programs can differ. The descriptors a
        // `prepare` builds modules for come from the epoch's engine configuration,
        // which this decision does not resolve: the main process resolves
        // the modules it needs when it loads them (`resolve`), so the
        // selection and the revisions are what this call is for.
        let filled = self.named_programs(target);
        let servers = self.program_servers(target);
        let leader_host = self.program_leader_host(target, &servers);
        // Bounded like the epoch install's: the client's two waits have no
        // deadline, so an unanswering Program Server would otherwise hold the
        // update without saying anything at all.
        let steps = crate::engine::control::PrepareSteps::new();
        let recorded = steps.clone();
        let prepared = {
            let servers = servers.clone();
            let leader_host = leader_host.clone();
            let filled = filled.clone();
            let preparing = tokio::task::spawn_blocking(move || {
                let complete =
                    crate::engine::control::with_store_programs(&servers, &leader_host, &filled)?;
                crate::engine::control::prepare_epoch(
                    &servers,
                    &leader_host,
                    &complete,
                    &[],
                    &recorded,
                )
            });
            let started = tokio::time::Instant::now();
            let deadline = started + PREPARE_TIMEOUT;
            let mut preparing = preparing;
            let prepared = loop {
                let left = deadline.saturating_duration_since(tokio::time::Instant::now());
                match tokio::time::timeout(left.min(HEARTBEAT), &mut preparing).await {
                    Ok(joined) => {
                        break joined.map_err(|error| {
                            UpdateError::Failed(anyhow::anyhow!(
                                "preparing the target's programs: {error}"
                            ))
                        })?;
                    }
                    // Not a refusal: the deployment asked for nothing the
                    // Program Server turned down — the server did not answer
                    // at all. The change may be legitimate, so this is the
                    // deployment's failure (500), not the request's (400).
                    Err(_) if left <= HEARTBEAT => {
                        return Err(UpdateError::Failed(anyhow::anyhow!(
                            "preparing the target's programs: no Program Server answered \
                             within {} s: the daemon was {}",
                            PREPARE_TIMEOUT.as_secs(),
                            steps.describe()
                        )));
                    }
                    Err(_) => tracing::info!(
                        waited_s = started.elapsed().as_secs(),
                        step = %steps.describe(),
                        "still preparing the target's programs"
                    ),
                }
            };
            prepared.map_err(|error| {
                UpdateError::Refused(format!("preparing the target's programs: {error:#}"))
            })?
        };
        let next = crate::epoch::Deployment {
            config: target.clone(),
            engine,
            programs: prepared,
        };
        // Nothing the configuration states and no revision the Program
        // Servers select differs: the change applies to nothing, and the
        // world-level path answers it as unchanged.
        if next.programs == filled {
            return Ok(None);
        }
        // What the configuration names is a part of what serves: the rest of an epoch's programs
        // come from the store. A target that resolves to what the epoch runs changes nothing,
        // whatever else the same apply changed.
        //
        // The epoch's record holds what its configuration named, and the target's resolution holds
        // the store's programs beside that, so the two are compared over what the epoch named: a
        // comparison of the whole sets differs for every target, and every change of a limit or a
        // switch would then replace all of an epoch's programs with themselves.
        if self.epochs().serving().is_some_and(|(_, record)| {
            record.policy == next.programs.policy
                && filled
                    .kernels
                    .keys()
                    .all(|key| record.kernels.contains_key(key))
                && record
                    .kernels
                    .iter()
                    .all(|(key, program)| next.programs.kernels.get(key) == Some(program))
        }) {
            return Ok(None);
        }
        // The reverted-update refusal is the daemon's record's, checked when
        // the update executes (`Epochs::unavailable`): this decision holds no
        // record of its own.
        let _ = force;
        Ok(Some(crate::epoch::Update::new(0, next)))
    }

    /// Where each host of the deployment prepares an epoch's programs: the
    /// Program Server socket the configuration names, or the Program
    /// Server's default socket on that host (§5.1, §6.3). The targets are the
    /// Program Server's own: the daemon does not query the devices of another
    /// host to name them.
    pub(crate) fn program_servers(
        &self,
        config: &ServerArgs,
    ) -> Vec<crate::engine::control::ProgramServer> {
        let socket = config
            .program_server
            .clone()
            .unwrap_or_else(program_server::command::default_socket);
        let mut servers = Vec::new();
        for replica in self.replicas() {
            let Some(control) = replica.control.get() else {
                continue;
            };
            let host = control.host().to_owned();
            if servers
                .iter()
                .any(|server: &crate::engine::control::ProgramServer| server.host == host)
            {
                continue;
            }
            servers.push(crate::engine::control::ProgramServer {
                host,
                socket: socket.clone(),
                targets: Vec::new(),
            });
        }
        servers
    }

    /// The host whose Program Server selects the epoch's revisions first,
    /// which every other host is then asked for exactly: the epoch's engine
    /// host when the configuration names one, and otherwise the first host
    /// of the deployment.
    pub(crate) fn program_leader_host(
        &self,
        config: &ServerArgs,
        servers: &[crate::engine::control::ProgramServer],
    ) -> String {
        config
            .engine_host
            .clone()
            .or_else(|| servers.first().map(|server| server.host.clone()))
            .or_else(|| {
                self.replicas().first().and_then(|replica| {
                    replica
                        .control
                        .get()
                        .map(|control| control.host().to_owned())
                })
            })
            .unwrap_or_default()
    }
}

/// A leader's engine as a hot restart hands it over (`Leader::engine_handoff`).
pub(crate) struct EngineHandoff {
    pub(crate) client: Arc<EngineClient>,
    pub(crate) engine: u64,
    pub(crate) next_engine: u64,
    pub(crate) served: proto::config::EngineArgs,
    pub(crate) config: ServerArgs,
    /// New requests wait (`Leader::pause`).
    pub(crate) paused: bool,
    /// The engine sleeps, and whether its sleep paused new requests.
    pub(crate) asleep: bool,
    pub(crate) sleep_paused: bool,
    /// The configurations a rollback marked unavailable.
    pub(crate) unavailable: Vec<crate::updates::epoch::Unavailable>,
}

/// What the leader serves, for `GET /admin/status`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct LeaderStatus {
    /// The served model name, the deployment's name.
    pub deployment: String,
    /// The model the deployment serves.
    pub model: String,
    /// `ready`, `standby`, `draining`, `failed` or `closed` for the engine
    /// that serves; `holding` while an engine is being replaced; `stopped`
    /// once none will serve.
    pub engine: String,
    /// Why the engine failed or no engine will serve.
    pub reason: Option<String>,
    /// Whether new requests wait (`/admin/pause_generation`).
    pub paused: bool,
    /// Whether the engine gave back its KV cache and graph storage
    /// (`/admin/sleep`).
    pub asleep: bool,
    /// Whether an engine is being replaced: by an update or a relaunch.
    pub replacing: bool,
    /// The requests the serving engine has in flight.
    pub in_flight: usize,
    /// The executable the serving engine and its workers run.
    pub executable: Option<String>,
}

/// How the requests in flight continue through an in-place weight update
/// (`POST /admin/update_weights`; vLLM's `keep` and `abort`, SGLang's
/// `in_place`, `retract` and `abort`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightsInFlight {
    /// Each continues on the new weights with the KV cache it reached; the
    /// pages it writes afterwards reach no shared tier.
    #[default]
    Keep,
    /// Each computes its state again with the new weights.
    Recompute,
    /// Each ends, aborted.
    Abort,
}

/// The engine configuration `config` resolves into, refused as a request when it does not:
/// its deployment file does not read or does not place.
pub(crate) fn resolved(config: &ServerArgs) -> Result<proto::config::EngineConfig, UpdateError> {
    config.engine_config().map_err(|error| {
        UpdateError::Refused(format!("the configuration does not resolve: {error:#}"))
    })
}

/// The deployment `config` describes, refused as [`resolved`] refuses.
pub(crate) fn deployment(config: &ServerArgs) -> Result<crate::epoch::Deployment, UpdateError> {
    crate::epoch::Deployment::of(config.clone()).map_err(|error| {
        UpdateError::Refused(format!("the configuration does not resolve: {error:#}"))
    })
}

/// Why a change of the deployment did not happen or did not complete.
#[derive(Debug, thiserror::Error)]
pub enum UpdateError {
    /// Another change is in progress.
    #[error("a change of the deployment is in progress")]
    InProgress,
    /// A world is on trial: it is committed or rolled back first.
    #[error("world {0} is on trial: commit it or roll it back first")]
    OnTrial(u64),
    /// The configuration is invalid, or asks for a change the deployment
    /// does not make.
    #[error("{0}")]
    Refused(String),
    /// The leader runs no engine process of its own to update.
    #[error("this leader runs no engine process of its own")]
    NoEngine,
    /// The update failed; the message says whether the running engine
    /// still serves.
    #[error(transparent)]
    Failed(#[from] anyhow::Error),
}

/// Whether `updated` differs from `served` in its scheduling limits alone, within
/// `capacity`, the batch the running engine's workers were started for: at
/// most as many calls and tokens a step and running requests. The
/// schedulers take such limits in place; any other change starts a standby
/// engine.
pub(crate) fn configurable(
    served: &proto::config::EngineConfig,
    updated: &proto::config::EngineConfig,
    capacity: Option<&wire::BatchCapacity>,
) -> bool {
    let Some(capacity) = capacity else {
        return false;
    };
    let limits = |config: &proto::config::EngineConfig| {
        (
            config.scheduler.max_batch,
            config.scheduler.max_queued_requests,
            config.scheduler.chunked_prefill_size,
            config.max_num_batched_tokens,
            config.max_running_requests,
        )
    };
    let mut same_otherwise = updated.clone();
    same_otherwise.scheduler = served.scheduler.clone();
    same_otherwise.max_num_batched_tokens = served.max_num_batched_tokens;
    same_otherwise.max_running_requests = served.max_running_requests;
    // A step's tokens left to the engine's default stay as they were.
    let tokens = match updated.max_num_batched_tokens {
        Some(tokens) => u64::try_from(tokens.get())
            .is_ok_and(|tokens| tokens <= capacity.max_num_batched_tokens),
        None => served.max_num_batched_tokens.is_none(),
    };
    let within = |value: std::num::NonZeroUsize, bound: u32| {
        u32::try_from(value.get()).is_ok_and(|value| value <= bound)
    };
    same_otherwise == *served
        && limits(served) != limits(updated)
        && tokens
        && (capacity.max_batch == 0 || within(updated.scheduler.max_batch, capacity.max_batch))
        && within(updated.max_running_requests, capacity.max_running_requests)
}

/// Whether the daemon processes requests for the checkpoint at `path` as
/// it does for the served model at `served`: the same tokenizer, tokenizer
/// settings, generation defaults and chat template, byte for byte.
pub(crate) async fn same_processing(served: &str, path: &str) -> Result<(), String> {
    use crate::model::assets::model_files::ResolvedModelFiles;
    // A diffusers checkpoint keeps what processes a request in directories of its own, and states
    // no tokenizer at its root for the model's files to resolve.
    if std::path::Path::new(served).join(MODULAR_INDEX).is_file() {
        return same_pipeline(std::path::Path::new(served), std::path::Path::new(path));
    }
    let resolve = |model: &str| {
        let model = model.to_owned();
        async move {
            ResolvedModelFiles::new(&model)
                .await
                .map_err(|error| format!("resolving the files of {model}: {error}"))
        }
    };
    let (old, new) = (resolve(served).await?, resolve(path).await?);
    let read = |file: Option<&std::path::Path>| -> Result<Option<Vec<u8>>, String> {
        file.map(|file| std::fs::read(file).map_err(|error| format!("{}: {error}", file.display())))
            .transpose()
    };
    for (name, old, new) in [
        (
            "tokenizer.json",
            Some(old.tokenizer_path.as_path()),
            Some(new.tokenizer_path.as_path()),
        ),
        (
            "tokenizer_config.json",
            old.tokenizer_config_path.as_deref(),
            new.tokenizer_config_path.as_deref(),
        ),
        (
            "generation_config.json",
            old.generation_config_path.as_deref(),
            new.generation_config_path.as_deref(),
        ),
        (
            "the chat template",
            old.chat_template_path.as_deref(),
            new.chat_template_path.as_deref(),
        ),
    ] {
        if read(old)? != read(new)? {
            return Err(format!(
                "the checkpoint at {path} has another {name} than the served model, which \
                 changes how requests are processed: deploy it instead"
            ));
        }
    }
    Ok(())
}

/// The index a diffusers checkpoint states its components in.
const MODULAR_INDEX: &str = "modular_model_index.json";

/// The components of a diffusers checkpoint that decide how a request is processed.
const PROCESSING_COMPONENTS: [&str; 4] = ["tokenizer", "processor", "scheduler", "audio_scheduler"];

/// [`same_processing`] for a diffusers checkpoint: the same index, and every file of the components
/// that process a request the same.
fn same_pipeline(served: &std::path::Path, path: &std::path::Path) -> Result<(), String> {
    let read = |file: &std::path::Path| {
        std::fs::read(file).map_err(|error| format!("{}: {error}", file.display()))
    };
    if read(&served.join(MODULAR_INDEX))? != read(&path.join(MODULAR_INDEX))? {
        return Err(format!(
            "the checkpoint at {} states other components than the served model: deploy it instead",
            path.display()
        ));
    }
    for component in PROCESSING_COMPONENTS {
        let directory = served.join(component);
        let entries = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(format!("{}: {error}", directory.display())),
        };
        for entry in entries {
            let entry = entry.map_err(|error| format!("{}: {error}", directory.display()))?;
            let name = entry.file_name();
            if read(&entry.path())? != read(&path.join(component).join(&name))? {
                return Err(format!(
                    "the checkpoint at {} has another {component}/{} than the served model, \
                     which changes how requests are processed: deploy it instead",
                    path.display(),
                    name.to_string_lossy()
                ));
            }
        }
    }
    Ok(())
}

/// The value behind `mutex`; every holder replaces whole values.
pub(crate) fn locked<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// One independent replica's engine lifecycle and its
/// worlds (`crate::updates::entry`).
pub(crate) struct Replica {
    pub(crate) registry: std::sync::Weak<Replicas>,
    pub(crate) ordinal: usize,
    served_model_name: String,
    pub(crate) migrator: Migrator,
    /// The hold every request passes.
    pub(crate) hold: Arc<RequestHold>,
    /// What starts the leader's engine processes; unset for an engine run
    /// elsewhere.
    pub(crate) control: std::sync::OnceLock<Arc<EngineControl>>,
    /// The arguments the serving engine serves with: the ones it was started
    /// with and the scheduling it took in place since, which a relaunch and the
    /// next update start from; unset with `control`.
    pub(crate) served: std::sync::Mutex<Option<proto::config::EngineArgs>>,
    /// The options the serving engine's configuration was resolved from, which
    /// an update overrides.
    pub(crate) config: std::sync::Mutex<ServerArgs>,
    /// The model their statistics are labelled with.
    pub(crate) model_name: String,
    /// Held while an engine is replaced, by a world change, a commit, a
    /// rollback or a relaunch, so that replacements do not overlap.
    pub(crate) replacing: Arc<tokio::sync::Mutex<()>>,
    /// The engine gave back its KV cache and graph storage (`Leader::sleep`).
    pub(crate) asleep: AtomicBool,
    /// The sleep paused new requests, which the wake lets through.
    sleep_paused: AtomicBool,
    /// The LoRA adapters the workers hold, by name (`Leader::load_adapter`).
    pub(crate) adapters: Arc<std::sync::Mutex<BTreeMap<String, Adapter>>>,
    /// The engine that executes, and the configurations a rollback marked
    /// unavailable.
    pub(crate) worlds: std::sync::Mutex<crate::updates::entry::WorldState>,
    /// Every engine's scheduler statistics, which the status reads.
    pub(crate) observer: Arc<crate::observe::Observer>,
    /// The profiling session that runs, or the last that ended
    /// (`crate::profiling`).
    pub(crate) profiles: crate::profiling::Profiles,
    /// Numerical manifests reused when planning successor rank budgets.
    pub(crate) manifests: std::sync::Mutex<storage::manifest::Planner>,
    /// The parked standby standing beside the serving engine (§2.5), if one is
    /// live. Started by [`Replica::keep_standby`], taken by
    /// [`Replica::promote_standby`].
    pub(crate) standby: std::sync::Mutex<Option<Standby>>,
    /// Serializes starting a standby: two callers that both saw none must not
    /// both start a process, since a second engine of the same scope is a
    /// whole copy of its device storage.
    standby_starting: tokio::sync::Mutex<()>,
    /// Whether the epoch's programs serve here, which is what a standby is
    /// prepared from.
    programs_serve: std::sync::atomic::AtomicBool,
    /// Whether this daemon handed the engine to a successor (§7.5.1): the
    /// successor owns its life then, and a watch that sees its client go away
    /// must not relaunch it.
    pub(crate) handed_over: AtomicBool,
    /// Whether an epoch switch owns this replica's engines: from the moment its target serves
    /// until the switch settled and the hold names the epoch that serves on. An engine lost
    /// meanwhile is the switch's to judge, not the watcher's to relaunch.
    following: AtomicBool,
}

impl Replica {
    pub(crate) fn new(
        options: &ServerArgs,
        client: Arc<EngineClient>,
        observer: Arc<crate::observe::Observer>,
        ordinal: usize,
        registry: std::sync::Weak<Replicas>,
        adapters: Arc<std::sync::Mutex<BTreeMap<String, Adapter>>>,
    ) -> Arc<Self> {
        let hold = Arc::new(RequestHold::new());
        let replica = Arc::new(Replica {
            registry,
            ordinal,
            served_model_name: options.served_model_name().to_owned(),
            migrator: Migrator::new(Arc::clone(&hold), options.migration_config.max_migrations),
            hold: Arc::clone(&hold),
            control: std::sync::OnceLock::new(),
            served: std::sync::Mutex::new(None),
            config: std::sync::Mutex::new(options.clone()),
            model_name: options.model().to_owned(),
            replacing: Arc::new(tokio::sync::Mutex::new(())),
            asleep: AtomicBool::new(false),
            sleep_paused: AtomicBool::new(false),
            adapters,
            worlds: std::sync::Mutex::new(crate::updates::entry::WorldState {
                executing: client.world(),
                ..Default::default()
            }),
            observer,
            profiles: crate::profiling::Profiles::default(),
            manifests: std::sync::Mutex::new(storage::manifest::Planner::default()),
            standby: std::sync::Mutex::new(None),
            standby_starting: tokio::sync::Mutex::new(()),
            programs_serve: std::sync::atomic::AtomicBool::new(false),
            handed_over: AtomicBool::new(false),
            following: AtomicBool::new(false),
        });
        hold.serve(client);
        replica
    }

    /// Begins failure handling after the owner has its serving or trial
    /// state. Standby owners are not recovered as committed replicas.
    #[expect(
        clippy::expect_used,
        reason = "the caller has constructed and assigned this replica before monitoring it"
    )]
    pub(crate) fn monitor(self: &Arc<Self>) {
        let client = self
            .hold
            .client()
            .expect("a constructed replica has its engine");
        tokio::spawn(watch_engine(Arc::clone(self), Arc::downgrade(&client)));
        self.start_standby();
    }

    pub(crate) fn hold(&self) -> &Arc<RequestHold> {
        &self.hold
    }

    /// Moves this replica onto an epoch switch's target when the switch serves it (§7.4).
    ///
    /// The target's child dials this daemon's request socket as it starts, and the switch commands
    /// it on its process link from then on. The frontend of this process moves when the record says
    /// the target serves: the responses the drained epoch ends then migrate to the client the hold
    /// serves, and a switch that fails never moves it.
    pub(crate) fn follow_switch(
        self: &Arc<Self>,
        control: Arc<EngineControl>,
        updates: Arc<crate::updates::driver::Driver>,
        target: SwitchTarget,
    ) {
        let replica = Arc::clone(self);
        let SwitchTarget {
            mut args,
            config,
            programs,
            servers,
            leader,
            engine,
            epoch,
            update,
        } = target;
        let mut served = updates.epochs().served();
        let serving_config = config.clone();
        let publishing = Arc::clone(&control);
        tokio::spawn(async move {
            // While the target prepares, its transfers yield to the serving epoch's steps
            // (`update_config.transfer_slowdown`); the guard ends that however the preparation ends.
            let preparing = tokio_util::sync::CancellationToken::new();
            let preparation = preparing.clone().drop_guard();
            if let Some(serving) = replica.hold.client() {
                let steps = Arc::new(std::sync::Mutex::new(
                    crate::interference::StepTime::default(),
                ));
                updates.watch_transfers(Arc::clone(&steps));
                let throttling = crate::interference::Throttling::new(
                    &replica.observer,
                    serving.world(),
                    update.transfer_slowdown,
                    std::time::Instant::now(),
                );
                tokio::spawn(crate::interference::control(
                    Arc::clone(&publishing) as Arc<dyn crate::interference::Throttle>,
                    vec![(Arc::clone(&replica.observer), throttling)],
                    crate::interference::INTERVAL,
                    preparing,
                    steps,
                ));
            }
            let sink = stats_sink(
                replica.model_name.clone(),
                Arc::clone(&replica.observer),
                replica.ordinal,
            );
            let world = args.world;
            let joined =
                tokio::task::spawn_blocking(move || control.join_target(engine, world, sink)).await;
            let client = match joined {
                Ok(Ok(client)) => Arc::new(client),
                Ok(Err(error)) => {
                    warn!(engine, "the switch's target did not join: {error:#}");
                    return;
                }
                Err(error) => {
                    warn!(engine, "joining the switch's target ended: {error}");
                    return;
                }
            };
            // The target parks with no programs: it takes them here, and the switch promotes it
            // once it holds them.
            let host = config.host_identity.clone();
            let handing = Arc::clone(updates.epochs());
            let pinning = Arc::clone(&updates);
            let target_page = args
                .engine_config
                .kv_cache
                .block_size
                .map_or(0, std::num::NonZeroU32::get);
            let mut source = None;
            let handed = tokio::task::spawn_blocking(move || {
                // The update names backends, and a revision is the Program Server's to select.
                let steps = crate::engine::control::PrepareSteps::new();
                let pinned = crate::engine::control::prepare_epoch(
                    &servers,
                    &leader,
                    &programs,
                    &[],
                    &steps,
                )
                .map_err(|error| crate::epoch::Failure {
                    kind: "prepare_failed".to_owned(),
                    reason: format!("{error:#}"),
                    device: None,
                })?;
                pinning.pin(&pinned);
                handing.hand_target(&config.host_identity, engine, &config, &pinned)
            })
            .await;
            match handed {
                Ok(Ok(())) => {
                    source = replica.warm_target(&updates, &update, target_page).await;
                    updates.handed(&host);
                }
                Ok(Err(failure)) => {
                    warn!(
                        engine,
                        "the switch's target took no programs: {}", failure.reason
                    );
                    updates.rollback(&failure.reason);
                    return;
                }
                Err(error) => {
                    warn!(
                        engine,
                        "handing the switch's target its programs ended: {error}"
                    );
                    updates.rollback(&error.to_string());
                    return;
                }
            }
            let settled = async {
                while !updates.settled() {
                    tokio::time::sleep(crate::updates::trial::POLL).await;
                }
            };
            let promoted = async {
                while *served.borrow_and_update() < epoch {
                    if served.changed().await.is_err() {
                        return false;
                    }
                }
                true
            };
            let served_target = tokio::select! {
                promoted = promoted => promoted,
                () = settled => false,
            };
            if !served_target {
                // The target fell before the frontend moved: the serving epoch serves on, at the
                // prefill budget it was configured with.
                if let Some(source) = source.take() {
                    restore_prefill(source).await;
                }
                return;
            }
            drop(preparation);
            // From here the switch owns both epochs' engines: the previous one is stopped when
            // the trial begins, which may come before the hold names the target.
            replica.following.store(true, Ordering::SeqCst);
            // The promotion restores the prefill budget the warming limited.
            if let Some(source) = source.take() {
                restore_prefill(source).await;
            }
            // The target maps the serving epoch's cache storage, and two epochs must not write it
            // at once: a request reaches the target only once the epoch that serves stopped
            // stepping. The drained epoch's state says the requests that move wait meanwhile.
            while updates.phase() == Some(crate::updates::switch::Phase::Serving) {
                tokio::time::sleep(DRAIN_POLL).await;
            }
            args.predecessor = None;
            let world = args.world;
            let before = locked(&replica.served).replace(args);
            // The configuration is the one the target serves from here, as its arguments are, and a
            // revert puts both back: the next apply starts from it, and the operator reads it.
            let config_before = std::mem::replace(&mut *locked(&replica.config), serving_config);
            // The previous client stays until the switch settles: it is what keeps the previous
            // epoch's process, which the switch drains next and serves again on a revert.
            let previous = replica.hold.client();
            replica.hold.serve(Arc::clone(&client));
            tokio::spawn(watch_engine(Arc::clone(&replica), Arc::downgrade(&client)));
            info!(engine, epoch, "the switch's target serves this replica");
            // The trial (§7.3 step 4, §7.4 step 3): the steps the previous epoch served before the
            // promotion are what the target's are read against, until the switch settles.
            let promoted = updates
                .promotion()
                .map_or_else(std::time::Instant::now, |(promoted, _)| promoted);
            let previous_world = previous.as_ref().map_or(0, |client| client.world());
            let trial = Arc::new(crate::updates::trial::Trial::begin(
                update.rollback.clone(),
                vec![crate::updates::trial::Observed::new(
                    Arc::clone(&replica.observer),
                    world,
                    previous_world,
                    update.rollback.window,
                    promoted,
                )],
                outcomes(),
            ));
            updates.begin_trial(Arc::clone(&trial));
            let mut relaunched = None;
            let host = locked(&replica.config).host_identity.clone();
            let mut keeps = update.previous_world != crate::options::PreviousWorld::Parked;
            while !updates.settled() {
                // The trial stopped the previous epoch's process: once it ended its node keeps
                // the weights alone.
                if keeps
                    && previous
                        .as_ref()
                        .is_some_and(|client| client.state() == EngineState::Closed)
                {
                    keeps = false;
                    updates.previous_ended(&host);
                }
                if let Some(failure) = trial.breach(outcomes())
                    && updates.fail_trial(failure.clone())
                {
                    warn!(
                        epoch,
                        kind = %failure.kind,
                        reason = %failure.reason,
                        "the epoch on trial passed a rollback threshold; it reverts"
                    );
                }
                // A revert of an epoch whose processes the trial stopped starts them again on
                // the weights they kept. The switch asks once the target stopped.
                if relaunched.is_none() && !updates.relaunch_due().is_empty() {
                    relaunched = Some(
                        replica
                            .relaunch_previous(
                                &publishing,
                                &updates,
                                before.clone(),
                                &config_before,
                            )
                            .await,
                    );
                }
                tokio::time::sleep(crate::updates::trial::POLL).await;
            }
            let committed = updates
                .epochs()
                .serving()
                .is_some_and(|(serving, _)| serving == epoch);
            if !committed {
                *locked(&replica.served) = before;
                *locked(&replica.config) = config_before;
                match relaunched {
                    Some(Ok(started)) => {
                        replica.hold.serve(Arc::clone(&started));
                        tokio::spawn(watch_engine(Arc::clone(&replica), Arc::downgrade(&started)));
                    }
                    Some(Err(error)) => {
                        let reason = format!("the previous epoch did not start again: {error:#}");
                        tracing::error!(%reason, "no engine serves requests");
                        replica.hold.close(reason);
                    }
                    None => {
                        if let Some(previous) = previous {
                            replica.hold.serve(previous);
                        }
                    }
                }
                warn!(
                    engine,
                    epoch, "the switch returned this replica to the previous epoch"
                );
                replica.start_standby();
            }
            replica.following.store(false, Ordering::SeqCst);
            if committed {
                // The committed target is what the next switch starts from.
                let published = tokio::task::spawn_blocking(move || {
                    publishing.publish_worlds(&[epoch], &mut None)
                })
                .await;
                if !matches!(published, Ok(Ok(()))) {
                    warn!(
                        epoch,
                        "publishing the committed epoch's world failed: {published:?}"
                    );
                }
                // The standby of the epoch before ended with it (`worker_config.standby`).
                replica.start_standby();
            }
        });
    }

    /// Runs the switch's warming once its target parked (§7.4 step 1), and answers the serving
    /// client whose prefill budget it limited, for the caller to restore. A warming that fails
    /// costs the requests in flight the tail they compute on the target and nothing else, so it
    /// neither fails the switch nor delays it further.
    async fn warm_target(
        self: &Arc<Self>,
        updates: &Arc<crate::updates::driver::Driver>,
        update: &crate::options::UpdateConfig,
        target_page: u32,
    ) -> Option<Arc<EngineClient>> {
        let source = self.hold.client()?;
        loop {
            match updates.parked() {
                Some(true) => break,
                Some(false) => tokio::time::sleep(crate::updates::trial::POLL).await,
                None => return None,
            }
        }
        let asking = Arc::clone(&source);
        let warmed = crate::updates::warming::warm(
            |operation| {
                let asking = Arc::clone(&asking);
                async move { asking.cache_warm(operation).await }
            },
            update.cache_tail_tokens.map(std::num::NonZeroU32::get),
            update.cache_prefill_tokens.map(std::num::NonZeroU32::get),
            target_page,
            crate::updates::warming::LIMIT,
        )
        .await;
        match warmed {
            Ok(warming) => {
                info!(?warming, "the requests in flight warmed the target's cache");
                updates.warmed(serde_json::to_value(&warming).unwrap_or_default());
            }
            Err(error) => {
                warn!(%error, "the warming ended: the requests compute their tails on the target");
                updates.warmed(serde_json::json!({ "error": error }));
            }
        }
        Some(source)
    }

    /// Starts the previous epoch again for a revert (§7.4): its processes exited for the trial
    /// (`update_config.previous_world`), the target stopped, and the epoch binds the weights it
    /// kept. The switch waits for this process exactly as it waits for one that drained.
    ///
    /// A start that fails leaves no engine: the switch ends all the same, since nothing it commands
    /// can serve the epoch, and the caller closes the hold with the reason.
    async fn relaunch_previous(
        self: &Arc<Self>,
        control: &Arc<EngineControl>,
        updates: &Arc<crate::updates::driver::Driver>,
        args: Option<proto::config::EngineArgs>,
        config: &ServerArgs,
    ) -> anyhow::Result<Arc<EngineClient>> {
        let host = config.host_identity.clone();
        let generation = updates
            .epochs()
            .next_generation(crate::engine::control::ENGINE_GROUP, 0);
        let started = self
            .start_previous(control, updates, args, config, generation)
            .await;
        // Whether or not it serves, the switch has nothing left to wait for.
        updates.relaunched(&host, generation);
        started
    }

    /// The process of [`Self::relaunch_previous`], holding the epoch's programs.
    async fn start_previous(
        self: &Arc<Self>,
        control: &Arc<EngineControl>,
        updates: &Arc<crate::updates::driver::Driver>,
        args: Option<proto::config::EngineArgs>,
        config: &ServerArgs,
        generation: u64,
    ) -> anyhow::Result<Arc<EngineClient>> {
        let args = args.context("the replica has no served arguments")?;
        let sink = stats_sink(
            self.model_name.clone(),
            Arc::clone(&self.observer),
            self.ordinal,
        );
        let starting = Arc::clone(control);
        let client = Arc::new(
            tokio::task::spawn_blocking(move || starting.start_at(args, generation, sink))
                .await??,
        );
        reload_adapters(self, &client).await;
        let handing = Arc::clone(updates.epochs());
        let host = config.host_identity.clone();
        let config = config.clone();
        let handed =
            tokio::task::spawn_blocking(move || handing.hand_to(&host, generation, &config));
        match tokio::time::timeout(SWITCH_TIMEOUT, handed).await {
            Ok(joined) => joined
                .context("handing the epoch's programs to the started engine")?
                .map_err(|failure| {
                    anyhow::anyhow!(
                        "the started engine did not take the epoch's programs ({}): {}",
                        failure.kind,
                        failure.reason
                    )
                })?,
            Err(_) => anyhow::bail!(
                "the started engine took the epoch's programs in neither Precopy, Switchover nor \
                 Commit within {} s",
                SWITCH_TIMEOUT.as_secs()
            ),
        }
        info!(
            generation,
            "the previous epoch started again on the weights it kept"
        );
        Ok(client)
    }

    /// Restarts the committed world while the caller owns its replacement
    /// lock. A pending commit must finish publication before calling this.
    pub(crate) async fn recover(self: &Arc<Self>) -> anyhow::Result<Arc<EngineClient>> {
        let control = Arc::clone(
            self.control
                .get()
                .context("the replica has no engine control")?,
        );
        let args = locked(&self.served)
            .clone()
            .context("the replica has no served arguments")?;
        self.hold.hold();
        let sink = stats_sink(
            self.model_name.clone(),
            Arc::clone(&self.observer),
            self.ordinal,
        );
        let client =
            Arc::new(tokio::task::spawn_blocking(move || control.start(args, sink)).await??);
        reload_adapters(self, &client).await;
        // The epoch's programs, before this process serves: a relaunched main
        // process starts with none, and the epoch they belong to was installed
        // for its predecessor, so nothing else sends them. Without this the
        // process waits on a program-layer command that never comes and
        // answers 503 to every request (§2.5, §7.5).
        if let Some(epochs) = self
            .registry
            .upgrade()
            .and_then(|registry| registry.updates.upgrade())
            .map(|driver| Arc::clone(driver.epochs()))
        {
            let config = locked(&self.config).clone();
            let host = config.host_identity.clone();
            let engine = client
                .engine()
                .context("the relaunched engine states no number")?;
            let handing = Arc::clone(&epochs);
            let handed =
                tokio::task::spawn_blocking(move || handing.hand_to(&host, engine, &config));
            match tokio::time::timeout(SWITCH_TIMEOUT, handed).await {
                Ok(joined) => joined
                    .context("handing the epoch's programs to the relaunched engine")?
                    .map_err(|failure| {
                        anyhow::anyhow!(
                            "the relaunched engine did not take the epoch's programs ({}): {}",
                            failure.kind,
                            failure.reason
                        )
                    })?,
                Err(_) => anyhow::bail!(
                    "the relaunched engine took the epoch's programs in neither Precopy, \
                     Switchover nor Commit within {} s",
                    SWITCH_TIMEOUT.as_secs()
                ),
            }
        }
        self.hold.serve(Arc::clone(&client));
        info!("the relaunched engine serves");
        self.start_standby();
        Ok(client)
    }
}

/// How often a switch reads whether the serving epoch drained: part of its pause.
const DRAIN_POLL: std::time::Duration = std::time::Duration::from_millis(5);

/// Gives `source` back the prefill budget a warming limited.
async fn restore_prefill(source: Arc<EngineClient>) {
    let restored = source
        .cache_warm(storage::manifest::cache::Operation::Prefill { tokens: None })
        .await;
    if let Err(error) = restored {
        warn!(%error, "the warming's prefill limit stays until this engine ends");
    }
}

/// Whether a replica keeps a standby (§2.5): the deployment reserved room for
/// one (`worker_config.standby`), the replica's engine serves rather than being
/// parked, draining or failing, and the world it executes is the world the
/// deployment serves. A standby anywhere else is a second engine of a scope
/// nothing routes to, or a copy of the device storage the serving engine needs.
fn keeps_standby(wanted: bool, serving: EngineState, world_serves: bool) -> bool {
    wanted && world_serves && serving == EngineState::Ready
}

/// A standby main process started beside this replica's serving engine (§2.5):
/// a second process of the serving world, parked with its scope's storage
/// claimed read-only, promoted in place when the serving engine is lost.
pub(crate) struct Standby {
    pub(crate) client: Arc<EngineClient>,
    /// The launch generation the standby was started under, from the epoch
    /// layer's one allocator.
    pub(crate) generation: u64,
    /// Whether it holds the epoch's programs and has captured its graphs, so a
    /// promotion is the promotion alone (§2.5).
    prepared: bool,
}

impl Replica {
    /// Prepares the parked standby in the background, once the epoch's programs
    /// serve: the caller is what makes them serve.
    pub(crate) fn prepare_standby(self: &Arc<Self>) {
        self.programs_serve.store(true, Ordering::SeqCst);
        let replica = Arc::clone(self);
        tokio::spawn(async move {
            let _starting = replica.standby_starting.lock().await;
            replica.prepare_parked_standby().await;
        });
    }

    /// Hands the parked standby the epoch's programs (§2.5): it captures its
    /// graphs on its own warmup scratch, and the graphs wait at their addresses
    /// for a promotion that then has nothing left to build.
    ///
    /// A standby the epoch's programs have not reached is kept: the promotion
    /// hands them over itself, later than it should, and still serves. The
    /// caller holds `standby_starting`, so a promotion never meets a handover
    /// half done.
    async fn prepare_parked_standby(self: &Arc<Self>) {
        if !self.programs_serve.load(Ordering::SeqCst) {
            return;
        }
        let parked = locked(&self.standby)
            .as_ref()
            .filter(|standby| !standby.prepared)
            .map(|standby| (Arc::clone(&standby.client), standby.generation));
        let Some((client, generation)) = parked else {
            return;
        };
        match self.hand_programs(&client).await {
            Ok(()) => {
                if let Some(standby) = locked(&self.standby)
                    .as_mut()
                    .filter(|standby| standby.generation == generation)
                {
                    standby.prepared = true;
                }
                info!(generation, "the standby captured its graphs and stands by");
            }
            Err(reason) => warn!(
                generation,
                "the standby did not take the epoch's programs while parked: {reason}; a \
                 promotion hands them over itself"
            ),
        }
    }

    /// Gives `client`'s process the programs the epoch serves.
    async fn hand_programs(self: &Arc<Self>, client: &Arc<EngineClient>) -> Result<(), String> {
        let Some(epochs) = self
            .registry
            .upgrade()
            .and_then(|registry| registry.updates.upgrade())
            .map(|driver| Arc::clone(driver.epochs()))
        else {
            return Ok(());
        };
        let config = locked(&self.config).clone();
        let host = config.host_identity.clone();
        let engine = client
            .engine()
            .ok_or_else(|| "the standby states no engine number".to_owned())?;
        let handed = tokio::task::spawn_blocking(move || epochs.hand_to(&host, engine, &config));
        match tokio::time::timeout(SWITCH_TIMEOUT, handed).await {
            Ok(Ok(Ok(()))) => Ok(()),
            Ok(Ok(Err(failure))) => Err(format!("{}: {}", failure.kind, failure.reason)),
            Ok(Err(error)) => Err(format!("the handover did not complete: {error}")),
            Err(_) => Err(format!(
                "it answered nothing within {} s",
                SWITCH_TIMEOUT.as_secs()
            )),
        }
    }

    /// Whether a switch still decides which engine serves this replica.
    pub(crate) fn follows_a_switch(&self) -> bool {
        self.following.load(Ordering::SeqCst)
    }

    /// Starts the standby in the background, if this replica serves and has
    /// none. Errors are logged: a replica without a standby still serves, and a
    /// loss then relaunches ([`Replica::recover`]) instead of promoting.
    pub(crate) fn start_standby(self: &Arc<Self>) {
        let replica = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(error) = replica.keep_standby().await {
                warn!(
                    model = %replica.model_name,
                    "the standby could not be started: {error:#}"
                );
            }
        });
    }

    /// Whether the world this replica's engine executes is still the world the
    /// deployment serves. A switch moves service to its target through the
    /// frontend, and the replica's own engine is then the drained previous
    /// epoch's: it keeps no standby and is not promoted, because nothing routes
    /// to it and a promoted process would be a second engine of a purged world.
    fn serves_serving_epoch(&self) -> bool {
        let Some(driver) = self
            .registry
            .upgrade()
            .and_then(|registry| registry.updates.upgrade())
        else {
            return true;
        };
        let Some((_, record)) = driver.epochs().serving() else {
            return true;
        };
        let Some(served) = locked(&self.served).as_ref().map(|args| args.world) else {
            return true;
        };
        record.world == served
    }

    /// Ends the parked standby, if one is live: dropping its client closes the
    /// control connection, which ends its process and releases the read-only
    /// claim it holds on its world's storage.
    pub(crate) fn drop_standby(&self) {
        let dropped = { locked(&self.standby).take() };
        if let Some(standby) = dropped {
            info!(
                generation = standby.generation,
                "the standby is dropped: its world no longer serves"
            );
        }
    }

    /// Replaces the parked standby with one prepared from what the deployment
    /// runs now: a program update moves the serving engine in place, and a
    /// standby started before it holds the programs the update replaced. A
    /// promoted standby would then serve kernels the epoch's record no longer
    /// names, so the parked process is dropped and another started.
    pub(crate) fn restart_standby(self: &Arc<Self>) {
        // Dropping the parked client closes its control connection, which ends
        // its process.
        let replaced = { locked(&self.standby).take() };
        drop(replaced);
        self.start_standby();
    }

    /// Starts the standby that stands beside this replica's serving engine, if
    /// one is wanted and none is live (§2.5).
    ///
    /// The standby stands by for the world the serving engine executes: it is
    /// started under a number from the epoch layer's one allocator, claims the
    /// scope's storage read-only (the launch file states the role), and parks
    /// once it is prepared. A replica whose engine is itself parked, draining or
    /// failing has no standby: there is nothing to stand beside.
    ///
    /// # Errors
    ///
    /// Fails when the process cannot be started or does not report that it
    /// stands by. The caller decides whether to retry.
    pub(crate) async fn keep_standby(self: &Arc<Self>) -> anyhow::Result<()> {
        // `worker_config.standby` decides whether a standby exists at all
        // (§2.5): each standby rank holds a CUDA context, graph pools and
        // workspaces of its own, and a deployment that has not reserved them
        // cannot hold a second engine of the same scope.
        let wanted = locked(&self.config).worker_config.standby;
        if !wanted {
            return Ok(());
        }
        let Some(control) = self.control.get().cloned() else {
            return Ok(());
        };
        let _starting = self.standby_starting.lock().await;
        let Some(serving) = self.hold.client() else {
            return Ok(());
        };
        if !keeps_standby(wanted, serving.state(), self.serves_serving_epoch()) {
            return Ok(());
        }
        {
            let mut slot = locked(&self.standby);
            match slot.as_ref().map(|standby| standby.client.state()) {
                // A parked standby is the one this replica wants.
                Some(EngineState::Standby) => return Ok(()),
                // A standby that failed or ended is replaced: dropping the
                // client closes its control connection, and its process ends.
                Some(_) => *slot = None,
                None => {}
            }
        }
        let Some(mut args) = locked(&self.served).clone() else {
            return Ok(());
        };
        let Some(driver) = self
            .registry
            .upgrade()
            .and_then(|registry| registry.updates.upgrade())
        else {
            return Ok(());
        };
        let epochs = Arc::clone(driver.epochs());
        let generation = epochs.next_generation(crate::engine::control::ENGINE_GROUP, 0);
        // The standby executes the serving engine's world: it maps that world's
        // weights and its KV cache read-only, ready to be promoted into the same
        // scope (§2.5). What a successor maps is the cache the serving engine
        // holds **now**, not the segments it started with: a cache that grew or
        // was resized since the launch would hand the standby units that are
        // shorter than the ones it will be promoted into, and the failure would
        // surface at promotion rather than here. `DescribeCache` is the same
        // account the epoch switch's target is built from.
        let predecessor = match serving.describe_cache().await {
            Ok(predecessor) => predecessor,
            Err(reason) => {
                // No standby is better than one that maps the wrong units: the
                // loss path relaunches the engine instead (`Replica::recover`).
                warn!(
                    model = %self.model_name,
                    "the standby was not started: the serving engine did not describe its cache: {reason}"
                );
                return Ok(());
            }
        };
        args.predecessor = Some(predecessor);
        let sink = stats_sink(
            self.model_name.clone(),
            Arc::clone(&self.observer),
            self.ordinal,
        );
        let client = Arc::new(
            tokio::task::spawn_blocking(move || control.start_at(args, generation, sink)).await??,
        );
        info!(
            generation,
            "the standby is prepared beside the serving engine"
        );
        *locked(&self.standby) = Some(Standby {
            client,
            generation,
            prepared: false,
        });
        self.prepare_parked_standby().await;
        Ok(())
    }

    /// Promotes the parked standby in place if this replica has one (§2.5), and
    /// answers the client that now serves.
    ///
    /// The promoted process is the same process, with its graphs at their
    /// addresses: it takes the storage its scope claimed read-only and serves
    /// under the serving engine's world. A standby that cannot be promoted is
    /// dropped, and the caller's loss path relaunches the engine instead.
    pub(crate) async fn promote_standby(self: &Arc<Self>) -> Option<Arc<EngineClient>> {
        // A standby being started or prepared is waited for: it is closer to serving than a
        // process this replica has yet to launch.
        let starting = self.standby_starting.lock().await;
        let standby = locked(&self.standby).take()?;
        self.hold.hold();
        // A standby prepares while it parks (`prepare_parked_standby`), and one that is lost to
        // before its programs reached it takes them here. A promotion that named it serving with
        // no programs would answer 503 to every request, so the daemon hands them over before it
        // moves the frontend, and the engine's own refusal of a programless promotion is a guard
        // rather than the only defence.
        if !standby.prepared
            && let Err(reason) = self.hand_programs(&standby.client).await
        {
            warn!(
                generation = standby.generation,
                "the standby did not take the epoch's programs: {reason}; the engine is relaunched"
            );
            return None;
        }
        drop(starting);
        match standby.client.promote(Vec::new(), false, true).await {
            Ok(()) => {
                self.hold.serve(Arc::clone(&standby.client));
                info!(
                    generation = standby.generation,
                    "the standby was promoted and serves"
                );
                self.start_standby();
                Some(standby.client)
            }
            Err(error) => {
                warn!(
                    generation = standby.generation,
                    "the standby could not be promoted: {error}; the engine is relaunched"
                );
                None
            }
        }
    }
}

impl Replica {
    /// Shuts the engine down: every scheduler finishes its outstanding
    /// requests as aborted and closes its workers, and the engine process
    /// exits. Runs on the blocking pool. Repeated calls are harmless.
    ///
    /// # Errors
    ///
    /// Ends the parked standby this replica keeps, if it has one (§2.5).
    ///
    /// A standby is a process of its own: ending it here means nothing of this
    /// replica's stays on the device after the daemon is gone, and a successor
    /// daemon starts its own instead of adopting a process it never numbered.
    /// Dropping the client closes the control connection, which the standby's
    /// engine treats as the end of the process.
    pub(crate) async fn end_standby(&self) {
        let standby = { locked(&self.standby).take() };
        if let Some(standby) = standby {
            if let Err(error) = tokio::task::spawn_blocking(move || standby.client.shutdown()).await
            {
                warn!("the standby's shutdown did not complete: {error}");
            }
        }
    }

    /// Fails when the blocking shutdown task does not complete, which happens
    /// only if it panics or its runtime is shutting down.
    pub(crate) async fn shutdown(&self) -> anyhow::Result<()> {
        self.end_standby().await;
        let hold = self.hold();
        let client = hold.client();
        hold.close("the daemon is shutting down");
        let Some(client) = client else {
            return Ok(());
        };
        tokio::task::spawn_blocking(move || client.shutdown())
            .await
            .context("engine shutdown did not complete")
    }

    /// Lets go of the engine without shutting it down, as the daemon a hot
    /// restart replaced does once its requests ended: the successor serves
    /// the engine from then on.
    ///
    /// # Errors
    ///
    /// Fails when the client's close does not complete.
    pub(crate) async fn detach(&self) -> anyhow::Result<()> {
        let hold = self.hold();
        let client = hold.client();
        hold.close("the daemon handed the deployment to its successor");
        let Some(client) = client else {
            return Ok(());
        };
        tokio::task::spawn_blocking(move || client.detach())
            .await
            .context("the engine connection's close did not complete")
    }

    /// What a hot restart hands the successor of this leader's engine: its
    /// client, its identity and the next engine identity, the arguments it
    /// serves with and the options they were resolved from.
    ///
    /// # Errors
    ///
    /// Fails when no engine this leader started serves.
    pub(crate) fn engine_handoff(&self) -> anyhow::Result<EngineHandoff> {
        let client = self
            .hold()
            .client()
            .context("no engine serves the deployment")?;
        let engine = client
            .engine()
            .context("the serving engine is not one this daemon started")?;
        let control = self
            .control
            .get()
            .context("this leader starts no engine of its own")?;
        let served = locked(&self.served)
            .clone()
            .context("this leader starts no engine of its own")?;
        Ok(EngineHandoff {
            next_engine: control.next_engine(),
            client,
            engine,
            served,
            config: locked(&self.config).clone(),
            paused: self.hold().is_paused(),
            asleep: self.asleep.load(Ordering::Acquire),
            sleep_paused: self.sleep_paused.load(Ordering::Acquire),
            unavailable: locked(&self.worlds).unavailable.values().cloned().collect(),
        })
    }

    /// Takes over the configurations a rollback marked unavailable on the
    /// leader a hot restart replaced.
    pub(crate) fn adopt_unavailable(&self, unavailable: Vec<crate::updates::epoch::Unavailable>) {
        let mut worlds = locked(&self.worlds);
        for mark in unavailable {
            worlds.unavailable.insert(mark.digest.clone(), mark);
        }
    }

    /// Takes over the pause and sleep of the leader a hot
    /// restart replaced (`EngineHandoff`).
    pub(crate) fn adopt_pause(&self, paused: bool, asleep: bool, sleep_paused: bool) {
        if paused {
            self.hold().pause();
        }
        self.asleep.store(asleep, Ordering::Release);
        self.sleep_paused.store(sleep_paused, Ordering::Release);
    }
}
impl Replica {
    /// What the leader serves now.
    pub(crate) fn status(&self) -> LeaderStatus {
        let hold = self.hold();
        let (engine, reason) = match hold.state() {
            HoldState::Engine(state) => match state {
                EngineState::Ready => ("ready", None),
                EngineState::Standby => ("standby", None),
                EngineState::Parked => ("parked", None),
                EngineState::Failed(reason) => ("failed", Some(reason)),
                EngineState::Closed => ("closed", None),
            },
            HoldState::Holding => ("holding", None),
            HoldState::Closed(reason) => ("stopped", Some(reason)),
        };
        LeaderStatus {
            deployment: self.served_model_name.clone(),
            model: self.model_name.clone(),
            engine: engine.to_owned(),
            reason,
            paused: hold.is_paused(),
            asleep: self.asleep.load(Ordering::Acquire),
            replacing: self.replacing.try_lock().is_err(),
            in_flight: hold.client().map_or(0, |client| client.in_flight()),
            executable: locked(&self.served).as_ref().map(|args| {
                args.engine_config
                    .worker
                    .executable
                    .to_string_lossy()
                    .into_owned()
            }),
        }
    }

    /// Holds new requests until [`Self::resume`]; the requests in flight go
    /// on. Returns whether requests were held already.
    pub(crate) fn pause(&self) -> bool {
        let paused = self.hold().pause();
        if !paused {
            info!("new requests wait: generation paused");
        }
        paused
    }

    /// Lets new requests reach the engine again.
    pub(crate) fn resume(&self) {
        self.hold().resume();
        info!("generation continues");
    }

    /// Has the serving engine give back its KV cache and CUDA graph storage
    /// and keep its weights, as vLLM's `sleep` does: new requests wait, the
    /// token requests in flight, image-generating ones included, are
    /// recomputed once it wakes, and the video requests in flight continue
    /// from their stored step. Returns what each worker reported.
    ///
    /// # Errors
    ///
    /// Fails when no engine serves, or with the reason the engine did not
    /// sleep; requests wait until a wake all the same.
    pub(crate) async fn sleep(&self) -> Result<Vec<String>, UpdateError> {
        let _replacing = self.replacing.lock().await;
        let hold = self.hold();
        let client = hold.client().ok_or(UpdateError::NoEngine)?;
        if self.asleep.swap(true, Ordering::AcqRel) {
            return Ok(Vec::new());
        }
        if !hold.pause() {
            self.sleep_paused.store(true, Ordering::Release);
        }
        let results = client
            .resize_cache(wire::CacheChange::Sleep)
            .await
            .map_err(|error| {
                UpdateError::Failed(anyhow::anyhow!(
                    "the engine did not sleep, and requests wait until /admin/wake_up: {error}"
                ))
            })?;
        info!("the engine sleeps: its KV cache and graph storage are given back");
        Ok(results)
    }

    /// Restores what [`Self::sleep`] gave back, and lets the requests it
    /// held through. Returns what each worker reported.
    ///
    /// # Errors
    ///
    /// Fails when no engine serves, or with the reason the engine did not
    /// wake; requests keep waiting then.
    pub(crate) async fn wake(&self) -> Result<Vec<String>, UpdateError> {
        let _replacing = self.replacing.lock().await;
        let hold = self.hold();
        let client = hold.client().ok_or(UpdateError::NoEngine)?;
        let results = client
            .resize_cache(wire::CacheChange::Wake)
            .await
            .map_err(|error| {
                UpdateError::Failed(anyhow::anyhow!("the engine did not wake: {error}"))
            })?;
        self.asleep.store(false, Ordering::Release);
        if self.sleep_paused.swap(false, Ordering::AcqRel) {
            hold.resume();
        }
        info!("the engine woke");
        Ok(results)
    }

    /// Runs the data plane's operation `name` with the JSON `payload`, and
    /// returns each rank's JSON result.
    ///
    /// # Errors
    ///
    /// Fails when no engine serves, or with the reason a worker gave.
    pub(crate) async fn hook(&self, name: &str, payload: &str) -> Result<Vec<String>, UpdateError> {
        let _replacing = self.replacing.lock().await;
        let client = self.hold().client().ok_or(UpdateError::NoEngine)?;
        let results = client
            .hook(name, payload)
            .await
            .map_err(|error| UpdateError::Failed(anyhow::anyhow!("worker hook {name}: {error}")))?;
        info!(hook = name, workers = results.len(), "ran the worker hook");
        Ok(results)
    }

    /// Rewrites the engine's weights in place from the checkpoint at
    /// `model_path`, another checkpoint of the served model, as SGLang's
    /// `update_weights_from_disk` does, or from a plugin's weight source
    /// (`scheme://...`) that streams the served checkpoint's tensors, and
    /// returns what each worker reported. New requests wait; the requests
    /// in flight continue as `in_flight` says (`engine::WeightsInFlight`):
    /// on the new weights with the KV cache they reached, recomputed with
    /// the new weights once the others finish, or aborted. A relaunch or an
    /// update after loads the new checkpoint.
    ///
    /// # Errors
    ///
    /// Refuses a checkpoint the daemon would process requests of
    /// differently (another tokenizer, chat template, generation defaults
    /// or model facts); fails when no engine serves, or with the reason a
    /// worker did not update, after which the weights are those of before.
    pub(crate) async fn update_weights(
        self: &Arc<Self>,
        model_path: &str,
        in_flight: WeightsInFlight,
    ) -> Result<Vec<String>, UpdateError> {
        let _replacing = self.replacing.lock().await;
        let hold = self.hold();
        let client = hold.client().ok_or(UpdateError::NoEngine)?;
        let mut config = locked(&self.config).clone();
        // A plugin's weight source streams the tensors of the served
        // checkpoint, whose configuration and tokenizer stay; a directory
        // is another checkpoint of the model, checked to process requests
        // alike.
        let streamed = model_path.contains("://");
        if streamed {
            config.load_config.weights = Some(model_path.to_owned());
        } else {
            same_processing(config.model(), model_path)
                .await
                .map_err(UpdateError::Refused)?;
            config.model = Some(model_path.to_owned());
            config.load_config.weights = None;
            let prepared = Leader::prepare(&config, &Stated::from_paths(Vec::new()))
                .await
                .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
            let facts = locked(&self.served)
                .as_ref()
                .map(|args| args.engine_config.model.facts.clone());
            if facts.is_some_and(|facts| facts != prepared.engine.model.facts) {
                return Err(UpdateError::Refused(format!(
                    "the checkpoint at {model_path} has other model facts (generation limits, \
                     special tokens or context length) than the served model: deploy it instead"
                )));
            }
        }

        let paused = !hold.pause();
        let started = Instant::now();
        let results = client.update_weights(model_path, in_flight).await;
        if paused {
            hold.resume();
        }
        let results = results.map_err(|error| {
            UpdateError::Failed(anyhow::anyhow!("the weights were not updated: {error}"))
        })?;
        if let Some(args) = locked(&self.served).as_mut() {
            if !streamed {
                args.engine_config.model.model = model_path.to_owned();
            }
            args.engine_config
                .load
                .weights
                .clone_from(&config.load_config.weights);
        }
        *locked(&self.config) = config;
        // A parked standby mapped the weights this update replaced, and a promotion would serve
        // them: it starts again on the new ones (§2.5).
        self.restart_standby();
        info!(
            model = model_path,
            elapsed_ms = started.elapsed().as_millis() as u64,
            ?in_flight,
            "the engine updated its weights in place"
        );
        Ok(results)
    }

    /// Has the serving engine revoke every cached prefix page; the
    /// requests in flight keep theirs.
    ///
    /// # Errors
    ///
    /// Fails when no engine serves.
    pub(crate) fn flush_cache(&self) -> Result<(), UpdateError> {
        let client = self.hold().client().ok_or(UpdateError::NoEngine)?;
        client.flush_cache();
        info!("flushed the prefix cache");
        Ok(())
    }
}

/// A LoRA adapter the workers hold.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Adapter {
    /// The slot of the workers' adapter pool it lies in.
    pub(crate) slot: u32,
    /// What keeps its cache entries apart from every other adapter's, one
    /// loaded into the same slot before included: its name and a load's
    /// random identity.
    pub(crate) identity: String,
    /// The load's JSON payload, with which a new engine's workers load it
    /// again.
    pub(crate) payload: String,
}

/// Loads every adapter of `replica` into `client`'s workers, as an engine
/// that replaces another starts without them. An adapter that does not load
/// leaves the registry, so that requests naming it are refused instead of
/// computing without it.
pub(crate) async fn reload_adapters(replica: &Replica, client: &EngineClient) {
    let adapters: Vec<(String, Adapter)> = locked(&replica.adapters)
        .iter()
        .map(|(name, adapter)| (name.clone(), adapter.clone()))
        .collect();
    for (name, adapter) in adapters {
        match client.hook("load_lora_adapter", &adapter.payload).await {
            Ok(_) => info!(
                adapter = name,
                slot = adapter.slot,
                "loaded the adapter again"
            ),
            Err(error) => {
                locked(&replica.adapters).remove(&name);
                warn!(
                    adapter = name,
                    "the adapter did not load into the new engine, and requests naming it are \
                     refused: {error}"
                );
            }
        }
    }
}

/// Watches the engine `client` names while the hold serves it, and starts
/// another in its place when it is lost: its workers failed for good or its
/// process ended.
///
/// The hold holds requests while the new engine starts; the lost engine's
/// token requests end as migratable and wait there, and the new engine
/// serves them and every request that arrives meanwhile. The engine's
/// workers find their weights retained at the Storage Servers. An engine the
/// hold no longer serves, as after a world change or a shutdown, is left as
/// it is. Without an engine process of the leader's own, or when the new
/// engine does not start, the hold closes and held and later requests fail.
pub(crate) async fn watch_engine(replica: Arc<Replica>, mut client: std::sync::Weak<EngineClient>) {
    loop {
        let Some(states) = client.upgrade().map(|client| client.states()) else {
            return;
        };
        let reason = lost(states, &replica).await;
        // A handover moves the engine, its client and the responses in flight to the
        // successor: this daemon's watch sees that client go away by design, and a
        // relaunch here would start a second engine for a deployment another daemon
        // serves.
        if replica
            .handed_over
            .load(std::sync::atomic::Ordering::Acquire)
        {
            tracing::info!(
                %reason,
                "this daemon handed its engine over; the successor owns it, so it is not relaunched here"
            );
            return;
        }
        let _replacing = replica.replacing.lock().await;
        let Some(lost_client) = client.upgrade() else {
            return;
        };
        if !replica.hold.serves(&lost_client) {
            return;
        }
        drop(lost_client);
        // Once the switch's target serves, the engine this replica lost is
        // the target's: the switch judges its death itself and reverts to the
        // previous epoch, and a relaunch from here would start the target
        // again from a `served` spec the switch is in the middle of reverting.
        // Before the switchover the lost engine is the previous epoch's, which
        // a relaunch restores as any loss does.
        if replica.following.load(Ordering::SeqCst) {
            tracing::info!(%reason, "the engine ended under an epoch switch, which owns it");
            return;
        }
        if let Some(registry) = replica.registry.upgrade()
            && let Some(updates) = registry.updates.upgrade()
            && updates.phase().is_some_and(|phase| {
                matches!(
                    phase,
                    crate::updates::switch::Phase::Serving
                        | crate::updates::switch::Phase::Probation
                        | crate::updates::switch::Phase::Committing
                        | crate::updates::switch::Phase::Reverting
                )
            })
        {
            tracing::error!(
                %reason,
                "the engine was lost while the switch's target served; the switch reverts"
            );
            return;
        }
        // A world the deployment no longer serves is relaunched, not promoted:
        // the frontend serves another epoch, and a revert to this one has to
        // find its engine alive.
        if replica.serves_serving_epoch()
            && let Some(promoted) = replica.promote_standby().await
        {
            tracing::info!(%reason, "the standby promoted in place; it serves");
            client = Arc::downgrade(&promoted);
            continue;
        }
        tracing::error!(%reason, "the engine was lost; starting another");
        match replica.recover().await {
            Ok(relaunched) => {
                client = Arc::downgrade(&relaunched);
            }
            Err(error) => {
                let reason =
                    format!("the engine was lost ({reason}) and did not start again: {error:#}");
                tracing::error!(%reason, "no engine serves requests");
                replica.hold.close(reason);
                return;
            }
        }
    }
}

/// Waits until the engine whose states `states` observes stops serving,
/// and returns why. A draining or standing-by engine is no loss — but a world
/// the deployment stops serving keeps no standby, and a parked engine is the
/// first moment the watcher learns the switch drained it: the standby's
/// read-only claim on that world would outlive the purge that frees it.
async fn lost(mut states: watch::Receiver<EngineState>, replica: &Replica) -> String {
    loop {
        match states.borrow_and_update().clone() {
            EngineState::Parked | EngineState::Standby => {
                if !replica.serves_serving_epoch() {
                    replica.drop_standby();
                }
            }
            EngineState::Ready => {}
            EngineState::Failed(error) => return format!("the engine failed: {error}"),
            EngineState::Closed => return "the engine process ended".to_owned(),
        }
        if states.changed().await.is_err() {
            return "the engine process ended".to_owned();
        }
    }
}

/// Runs synchronous preprocessing (chat rendering, tokenization) on the
/// blocking pool.
///
/// A panicked or cancelled task becomes a tokenization error of
/// `request_id`. Dropping the returned future does not stop the work; its
/// result is discarded.
async fn preprocess<T: Send + 'static>(
    request_id: &str,
    work: impl FnOnce() -> Result<T, ServeError> + Send + 'static,
) -> Result<T, ServeError> {
    tokio::task::spawn_blocking(work)
        .await
        .unwrap_or_else(|error| {
            Err(ServeError::Tokenize {
                request_id: request_id.to_owned(),
                source: TokenizeError::Task(error),
            })
        })
}

/// Microseconds elapsed since `started`.
fn elapsed_us(started: Instant) -> u64 {
    started.elapsed().as_micros() as u64
}

/// The control tokens a model's engine schedules with.
struct SpecialTokenIds {
    bos: u32,
    eos: Vec<u32>,
    end_of_image: u32,
}

/// Resolves the control tokens the engine uses for the loaded model.
///
/// `bos` and `end_of_image` are each `0`: every profile the daemon serves
/// states neither, and the engine reads `0` as "no such token". The EOS list
/// is the model's complete EOS set with the primary EOS placed first (and
/// added when the set lacks it). The order matters to the engine scheduler,
/// which substitutes `eos[0]` for the token of a completion that commits
/// none. The primary EOS is `ModelConfig::primary_eos_token_id`; without it
/// the list keeps the set's ascending order.
fn special_token_ids(model: &ModelConfig) -> SpecialTokenIds {
    let primary_eos = model.primary_eos_token_id;
    let mut eos = model.eos_token_ids.iter().copied().collect::<Vec<_>>();
    if let Some(primary_eos) = primary_eos {
        eos.retain(|value| *value != primary_eos);
        eos.insert(0, primary_eos);
    }
    SpecialTokenIds {
        bos: 0,
        eos,
        end_of_image: 0,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use crate::metrics::METRICS;
    use engine::{SchedulerHandle, SimEngine, SimWorker, Worker};
    use futures::{StreamExt as _, poll};
    use proto::request::{
        Event, GenerationConstraint, GenerationRequest, ImageGenerationConfig, Request,
    };
    use proto::request::{ImageParams, SamplingParams};

    use super::{EngineState, Leader, keeps_standby};
    use crate::input::GenerateRequest;
    use crate::model::ModelParameters;
    use crate::openai::ChatCompletionRequest;
    use crate::openai::chat_completions::{
        ChatResponseContext, collect_chat_completion, lower_chat_request,
    };
    use crate::output::{FinishReason, RequestOutput};
    use crate::test_support::{
        SERVED_MODEL, configured_tokenizer, leader_over, sim_client, sim_core, sim_engine_over,
        sim_model,
    };

    /// A Qwen3 engine over one scheduler, named `engine_model`, that drives
    /// `worker`.
    ///
    /// The scheduler's model name labels the process-wide request metrics,
    /// so a test that reads them names its own engine.
    fn qwen3_engine(worker: SimEngine, engine_model: &str) -> Leader {
        let core = sim_core(engine::sim_args(engine_model), worker);
        sim_engine_over(ModelParameters::Qwen3, vec![core])
    }

    /// `request` as the chat route lowers it for the served model.
    fn lowered(id: &str, request: ChatCompletionRequest) -> GenerateRequest {
        lower_chat_request(id.to_owned(), SERVED_MODEL, request)
            .unwrap()
            .0
    }

    /// Greedy chat request of at most `max_completion_tokens` tokens that may
    /// sample only `a`, stopping on each of `stop`.
    fn chat_request(max_completion_tokens: u32, stop: &[&str]) -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": "prompt"}],
            "max_completion_tokens": max_completion_tokens,
            "temperature": 0.0,
            "allowed_token_ids": [u32::from(b'a')],
            "stop": stop,
        }))
        .unwrap()
    }

    /// Reads `stream` to its end and returns its terminal reason and usage.
    async fn finish(
        mut stream: crate::output::RequestOutputStream,
    ) -> (FinishReason, crate::output::RequestUsage) {
        let mut finished = None;
        while let Some(event) = stream.next().await {
            if let RequestOutput::Finished { reason, usage } = event.unwrap() {
                finished = Some((reason, usage));
            }
        }
        finished.expect("the request finishes")
    }

    /// Returns whether the rendered process-wide metrics hold `line`.
    fn rendered_metrics_hold(line: &str) -> bool {
        let rendered = METRICS.render().unwrap();
        rendered.lines().any(|rendered| rendered == line)
    }

    /// A text request that generates until the simulator's EOS.
    fn text_request() -> Request {
        GenerationRequest {
            request_id: Default::default(),
            prompt_token_ids: vec![1, 2, 3, 4],
            multimodal_inputs: Default::default(),
            negative_prompt_token_ids: Vec::new(),
            constraint: GenerationConstraint::UndOnly,
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_images: 1,
            max_und_tokens: 64,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            image_generation: ImageGenerationConfig::default(),
            readout: Vec::new(),
            generated_token_ids: Vec::new(),
            adapter: 0,
            generated_images: Vec::new(),
            trajectory: Default::default(),
        }
        .into()
    }

    /// An engine that is lost while it serves ends the leader's serving: it
    /// reports itself unhealthy and refuses every later request, naming the
    /// loss.
    #[tokio::test]
    async fn a_lost_engine_fails_later_requests() {
        let core = sim_core(engine::sim_args("lost-engine-model"), SimEngine::new());
        let client = Arc::new(sim_client(vec![core]));
        let engine = leader_over(
            sim_model(ModelParameters::Qwen3),
            configured_tokenizer(),
            "lost-engine-model".into(),
            vec![Arc::clone(&client)],
        );
        assert!(engine.is_healthy());

        // The engine process ends without the leader asking it to.
        tokio::task::spawn_blocking(move || client.shutdown())
            .await
            .unwrap();
        for _ in 0..100 {
            if !engine.is_healthy() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(!engine.is_healthy());

        let error = engine.submit(text_request()).await.err();
        assert!(matches!(
            error,
            Some(crate::engine::client::SubmitError::Unavailable(_))
        ));
    }

    /// Concurrent requests use independent replicas, and losing one leaves
    /// the other available through the same public service.
    #[tokio::test]
    async fn concurrent_requests_spread_over_engine_cores() {
        // Every wait in this test is bounded and named: an unbounded wait here
        // reads as a suite that hung, with no line saying which of the seven
        // answers the leader never gave.
        let mut clients = Vec::new();
        for _ in 0..2 {
            let core = sim_core(engine::sim_args("sim-model"), SimEngine::new());
            clients.push(Arc::new(sim_client(vec![core])));
        }
        let engine = leader_over(
            sim_model(ModelParameters::Qwen3),
            configured_tokenizer(),
            "sim-model".into(),
            clients.clone(),
        );

        // Both requests are submitted before either stream is read, so the
        // first is still in flight when the second is routed.
        let mut streams = Vec::new();
        let mut serving = Vec::new();
        for index in 0..2 {
            let (stream, core) = within(
                10,
                &format!("the {} request's admission", index + 1),
                engine.submit(text_request()),
            )
            .await
            .expect("submit request");
            streams.push(stream);
            serving.push(core);
        }
        assert_ne!(serving[0], serving[1]);
        for (index, stream) in streams.iter_mut().enumerate() {
            within(10, &format!("the {} request's batch", index + 1), async {
                while let Some(event) = stream.next().await {
                    if let Event::Finished { reason, .. } = event {
                        assert_eq!(reason, proto::request::FinishReason::Eos);
                        break;
                    }
                }
            })
            .await;
        }
        let lost = Arc::clone(&clients[0]);
        within(
            15,
            "the lost replica's teardown",
            tokio::task::spawn_blocking(move || {
                lost.shutdown();
                lost.await_exit(Duration::from_secs(5)).unwrap();
            }),
        )
        .await
        .unwrap();
        assert!(engine.is_healthy());
        let (mut stream, replica) = within(
            10,
            "the request after the lost replica",
            engine.submit(text_request()),
        )
        .await
        .unwrap();
        assert_eq!(replica, 1);
        let mut finished = false;
        within(10, "the surviving replica's batch", async {
            while let Some(event) = stream.next().await {
                if let Event::Finished { reason, .. } = event {
                    assert_eq!(reason, proto::request::FinishReason::Eos);
                    finished = true;
                }
            }
        })
        .await;
        assert!(finished);
        within(10, "the replica's shutdown", engine.shutdown())
            .await
            .expect("shutdown replica");
    }

    /// Awaits `future`, failing by name when it does not answer within
    /// `seconds`: a test whose unbounded wait hangs says which answer it never
    /// got instead of outliving the suite's bound in silence.
    async fn within<T>(
        seconds: u64,
        what: &str,
        future: impl std::future::Future<Output = T>,
    ) -> T {
        match tokio::time::timeout(Duration::from_secs(seconds), future).await {
            Ok(value) => value,
            Err(_) => panic!("{what} did not answer within {seconds} s"),
        }
    }

    /// Simulated worker whose teardown lasts until another task releases
    /// it, as closing worker processes that drain and exit takes time.
    struct GatedCloseWorker {
        inner: SimWorker,
        release: std::sync::mpsc::Receiver<()>,
        released: Arc<AtomicBool>,
    }

    impl Worker for GatedCloseWorker {
        fn info(&self) -> &engine::WorkerInfo {
            self.inner.info()
        }

        fn is_ready(&self, worker: &engine::WorkerGroupId) -> bool {
            self.inner.is_ready(worker)
        }

        fn has_capacity(&self, worker: &engine::WorkerGroupId) -> bool {
            self.inner.has_capacity(worker)
        }

        fn command_has_capacity(&self, command: &ipc::BatchCommand) -> bool {
            self.inner.command_has_capacity(command)
        }

        fn submit(
            &mut self,
            batch: engine::ExecutionBatch,
        ) -> Result<(), engine::WorkerSubmitError> {
            self.inner.submit(batch)
        }

        fn poll(
            &mut self,
            timeout: Duration,
        ) -> Result<Option<engine::BatchResult>, engine::WorkerError> {
            self.inner.poll(timeout)
        }

        fn close(&mut self) -> Result<(), engine::WorkerError> {
            // The wait is bounded so that a blocked runtime fails the test
            // instead of hanging it.
            let released = self.release.recv_timeout(Duration::from_secs(5)).is_ok();
            self.released.store(released, Ordering::SeqCst);
            self.inner.close()
        }
    }

    /// Shutdown waits for worker teardown without holding the runtime
    /// thread: on a single-threaded runtime, the task that ends the teardown
    /// runs while `shutdown` is pending.
    #[tokio::test(flavor = "current_thread")]
    async fn shutdown_leaves_the_runtime_free_during_worker_teardown() {
        let (release_tx, release) = std::sync::mpsc::channel();
        let released = Arc::new(AtomicBool::new(false));
        let core = SchedulerHandle::with_worker(
            &engine::sim_args("sim-model"),
            engine::SchedulerConfig::default(),
            Box::new(GatedCloseWorker {
                inner: SimWorker::new(SimEngine::new()),
                release,
                released: Arc::clone(&released),
            }),
        )
        .expect("start a sim scheduler");
        let engine = sim_engine_over(ModelParameters::Qwen3, vec![core]);

        let releaser = tokio::spawn(async move {
            let _ = release_tx.send(());
        });
        engine.shutdown().await.expect("shutdown engine");
        releaser.await.expect("release task");

        assert!(
            released.load(Ordering::SeqCst),
            "worker teardown held the runtime thread"
        );
    }

    /// A chat prompt that exceeds the total capacity of a two-unit prefix
    /// cache is returned as a stream and then rejected by engine admission.
    /// Collecting the completion must report that rejection as HTTP 400.
    #[tokio::test]
    async fn chat_admission_rejection_remains_a_bad_request() {
        let mut worker = SimEngine::new();
        // Unit zero is the sentinel: two allocatable 64-token pages remain.
        worker.set_num_units(3);
        let engine = qwen3_engine(worker, SERVED_MODEL);
        let request = serde_json::from_value(serde_json::json!({
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": "a".repeat(1024)}],
            "max_completion_tokens": 8
        }))
        .unwrap();

        let stream = engine
            .generate(lowered("rejected-chat", request))
            .await
            .unwrap();
        let context = ChatResponseContext {
            request_id: "rejected-chat".to_string(),
            response_model: SERVED_MODEL.to_string(),
            include_usage: false,
            requested_logprobs: false,
            include_prompt_logprobs: false,
            include_reasoning: false,
            return_token_ids: false,
            return_tokens_as_token_ids: false,
        };
        let error = collect_chat_completion(stream, context, 0)
            .await
            .unwrap_err();

        assert_eq!(error.status_code(), axum::http::StatusCode::BAD_REQUEST);
        engine.shutdown().await.unwrap();
    }

    /// A request's reported latency covers the time it waited for the
    /// blocking pool before preprocessing ran.
    #[test]
    fn reported_latency_includes_time_waiting_for_preprocessing() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let engine = qwen3_engine(SimEngine::new(), SERVED_MODEL);

            // Hold the only blocking thread so preprocessing waits in the
            // queue for a measurable interval before it runs.
            let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let occupied = tokio::task::spawn_blocking(move || {
                ready_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            });
            ready_rx.await.unwrap();
            let wait = Duration::from_millis(25);
            let mut pending = Box::pin(engine.generate(lowered("timed", chat_request(8, &["aa"]))));
            assert!(poll!(pending.as_mut()).is_pending());
            tokio::time::sleep(wait).await;
            release_tx.send(()).unwrap();
            occupied.await.unwrap();

            let (_, usage) = finish(pending.await.unwrap()).await;
            assert!(
                usage.timings.total_us >= wait.as_micros() as u64,
                "{:?}",
                usage.timings
            );
            engine.shutdown().await.unwrap();
        });
    }

    /// A completed request is recorded once in the request metrics under its
    /// engine's model name: the token counters and token histograms carry the
    /// counts of its usage report, its finish reason is counted, and each
    /// latency histogram whose input it carries holds one observation.
    #[tokio::test]
    async fn completed_request_is_recorded_in_request_metrics() {
        let engine_model = "request-metrics-model";
        let engine = qwen3_engine(SimEngine::new(), engine_model);

        let stream = engine
            .generate(lowered("metrics", chat_request(3, &[])))
            .await
            .unwrap();
        let (reason, usage) = finish(stream).await;
        assert_eq!(reason, FinishReason::Length);
        assert_eq!(usage.output_tokens, 3);

        let labels = format!("model_name=\"{engine_model}\",engine=\"0\"");
        for expected in [
            format!("uniserve:request_success_total{{{labels},finished_reason=\"length\"}} 1"),
            format!(
                "uniserve:prompt_tokens_total{{{labels}}} {}",
                usage.prompt_tokens
            ),
            format!("uniserve:generation_tokens_total{{{labels}}} 3"),
            format!("uniserve:request_prompt_tokens_count{{{labels}}} 1"),
            format!("uniserve:request_generation_tokens_count{{{labels}}} 1"),
            format!("uniserve:request_max_num_generation_tokens_count{{{labels}}} 1"),
            format!("uniserve:e2e_request_latency_seconds_count{{{labels}}} 1"),
            format!("uniserve:request_queue_time_seconds_count{{{labels}}} 1"),
            format!("uniserve:time_to_first_token_seconds_count{{{labels}}} 1"),
        ] {
            assert!(rendered_metrics_hold(&expected), "missing `{expected}`");
        }
        engine.shutdown().await.unwrap();
    }

    /// Each request is recorded under the scheduler that served it.
    #[tokio::test]
    async fn request_metrics_name_the_engine_core_that_served_each_request() {
        let engine_model = "core-metrics-model";
        let cores = (0..2)
            .map(|_| sim_core(engine::sim_args(engine_model), SimEngine::new()))
            .collect();
        let engine = sim_engine_over(ModelParameters::Qwen3, cores);

        // Both requests are submitted before either stream is read, so the
        // first is still in flight when the second is routed.
        let mut streams = Vec::new();
        for id in ["core-a", "core-b"] {
            let request = lowered(id, chat_request(3, &[]));
            streams.push(engine.generate(request).await.unwrap());
        }
        for stream in streams {
            assert_eq!(finish(stream).await.0, FinishReason::Length);
        }

        for core in 0..2 {
            let expected = format!(
                "uniserve:request_success_total{{model_name=\"{engine_model}\",\
                 engine=\"{core}\",finished_reason=\"length\"}} 1"
            );
            assert!(rendered_metrics_hold(&expected), "missing `{expected}`");
        }
        engine.shutdown().await.unwrap();
    }

    /// A client that disconnects mid-generation is counted as an aborted
    /// request with its latency, while its partial output, which never
    /// reached a usage report, adds nothing to the token metrics.
    #[tokio::test]
    async fn disconnected_request_is_recorded_as_aborted_without_token_usage() {
        let engine_model = "disconnect-metrics-model";
        let mut worker = SimEngine::new();
        worker.set_text_len(1_000_000);
        let engine = qwen3_engine(worker, engine_model);

        let mut stream = engine
            .generate(lowered("disconnect", chat_request(1_000, &[])))
            .await
            .unwrap();
        loop {
            match stream.next().await.unwrap().unwrap() {
                RequestOutput::TextDelta { text } if !text.is_empty() => break,
                _ => {}
            }
        }
        drop(stream);

        let labels = format!("model_name=\"{engine_model}\",engine=\"0\"");
        for expected in [
            format!("uniserve:request_success_total{{{labels},finished_reason=\"abort\"}} 1"),
            format!("uniserve:e2e_request_latency_seconds_count{{{labels}}} 1"),
        ] {
            assert!(rendered_metrics_hold(&expected), "missing `{expected}`");
        }
        let rendered = METRICS.render().unwrap();
        for family in [
            "uniserve:generation_tokens_total",
            "uniserve:request_generation_tokens_count",
        ] {
            let series = format!("{family}{{{labels}}}");
            assert!(
                !rendered.lines().any(|line| line.starts_with(&series)),
                "unexpected `{series}` in:\n{rendered}"
            );
        }
        engine.shutdown().await.unwrap();
    }

    /// Scheduling alone, within the batch the workers were started for, is
    /// taken in place; a limit beyond it, any other change, or no change
    /// starts a standby engine.
    #[test]
    fn scheduling_within_the_workers_batch_changes_in_place() {
        use super::configurable;
        use proto::request::wire;

        let base = crate::options::ServerArgs::default()
            .engine_config()
            .expect("the configuration resolves");
        let config = |batch: usize, tokens: Option<usize>, running: usize| {
            let mut config = base.clone();
            config.scheduler.max_batch = std::num::NonZeroUsize::new(batch).unwrap();
            config.max_num_batched_tokens = tokens.and_then(std::num::NonZeroUsize::new);
            config.max_running_requests = std::num::NonZeroUsize::new(running).unwrap();
            config
        };
        let served = config(128, None, 128);
        let capacity = wire::BatchCapacity {
            max_batch: 128,
            max_num_batched_tokens: 8192,
            max_running_requests: 128,
        };
        let within = [config(64, Some(4096), 64), config(128, Some(8192), 128)];
        for updated in &within {
            assert!(
                configurable(&served, updated, Some(&capacity)),
                "{updated:?}"
            );
            assert!(!configurable(&served, updated, None), "{updated:?}");
        }
        let mut captured_otherwise = within[0].clone();
        captured_otherwise.cuda_graph.decode_batch_sizes =
            Some(vec![std::num::NonZeroU32::new(8).unwrap()]);
        let beyond = [
            config(256, None, 128),
            config(128, Some(16384), 128),
            config(128, None, 256),
            served.clone(),
            captured_otherwise,
        ];
        for updated in &beyond {
            assert!(
                !configurable(&served, updated, Some(&capacity)),
                "{updated:?}"
            );
        }
        // A step's tokens the served configuration states cannot return to the
        // engine's default in place.
        let stated = config(128, Some(4096), 128);
        assert!(!configurable(
            &stated,
            &config(128, None, 128),
            Some(&capacity)
        ));
    }

    /// The engine configuration of a deployment of `data` schedulers of `tensor` ranks each,
    /// whose experts the schedulers exchange when `experts` holds.
    fn deployment(tensor: usize, data: usize, experts: bool) -> proto::config::EngineConfig {
        let mut options = crate::options::ServerArgs::default();
        options.tensor_parallel_size = std::num::NonZeroUsize::new(tensor).unwrap();
        options.data_parallel_size = std::num::NonZeroUsize::new(data).unwrap();
        options.moe_config.expert_parallel = experts;
        options.engine_config().expect("the configuration resolves")
    }

    /// Independent data-parallel replicas each take their own scheduler's ranks, on the devices
    /// the deployment assigned them, and every other part of the configuration as it is.
    #[test]
    fn independent_replicas_keep_their_assigned_tensor_parallel_ranks() {
        let config = deployment(2, 2, false);
        let replicas = super::replica_configs(&config).unwrap();
        let devices: Vec<Vec<String>> = replicas
            .iter()
            .map(|replica| {
                replica
                    .parallel
                    .worker_groups
                    .iter()
                    .flat_map(|group| group.ranks.iter().map(|rank| rank.device.clone()))
                    .collect()
            })
            .collect();
        assert_eq!(
            devices,
            vec![vec!["cuda:0", "cuda:1"], vec!["cuda:2", "cuda:3"]]
        );
        assert!(replicas.iter().all(|replica| {
            replica.parallel.data_parallel_size.get() == 1 && replica.scheduler == config.scheduler
        }));
    }

    /// Schedulers that exchange experts communicate, so they stay in one replica.
    #[test]
    fn communicating_expert_ranks_stay_in_one_replica() {
        let replicas = super::replica_configs(&deployment(1, 4, true)).unwrap();
        assert_eq!(replicas.len(), 1);
        assert_eq!(
            replicas[0]
                .parallel
                .worker_groups
                .iter()
                .map(|group| group.ranks.len())
                .sum::<usize>(),
            4
        );
    }

    /// A standby exists only where the deployment reserved room for one and the
    /// replica's world is the world the deployment serves (§2.5). Each of the
    /// three conditions is a whole engine's device storage when it is wrong.
    #[test]
    fn a_standby_exists_only_where_the_deployment_reserved_one_and_the_world_serves() {
        assert!(keeps_standby(true, EngineState::Ready, true));
        // Nothing reserved: a second engine would take the serving engine's
        // memory, so the deployment asks for none.
        assert!(!keeps_standby(false, EngineState::Ready, true));
        // A parked, standing-by, failed or closed engine has nothing to stand
        // beside, and a standby of a standby is a third copy.
        assert!(!keeps_standby(true, EngineState::Parked, true));
        assert!(!keeps_standby(true, EngineState::Standby, true));
        assert!(!keeps_standby(
            true,
            EngineState::Failed("a fatal fault".into()),
            true
        ));
        assert!(!keeps_standby(true, EngineState::Closed, true));
        // A world a switch replaced keeps no standby: nothing routes to it, and
        // its read-only claim would outlive the purge.
        assert!(!keeps_standby(true, EngineState::Ready, false));
    }
}
