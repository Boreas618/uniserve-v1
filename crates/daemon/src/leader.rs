//! The leader's served model: one model's input processing, its engine
//! process, and the output processing of every request.
//!
//! A [`Leader`] owns the model's [`Processor`] and the client of the
//! engine process that schedules its requests ([`EngineClient`]), as vLLM's
//! `AsyncLLM` owns its input and output processors over its engine core
//! client and SGLang's tokenizer manager drives its schedulers. Every
//! request takes one path:
//!
//! 1. its images are fetched on the async runtime and its prompt rendered and
//!    tokenized on the blocking pool ([`Processor`]);
//! 2. [`Leader::submit`] hands the tokenized request through the migrator
//!    to the engine process, which routes it to one of its schedulers and
//!    returns its events ([`RequestEvents`]), across the engines that serve
//!    it when it migrates;
//! 3. the events are assembled into public [`RequestOutput`] events: a
//!    token request's by the body that writes its response
//!    (`openai::run::GenerationRun`, through the `OutputProcessor`
//!    [`Leader::generate`] built), a media request's by
//!    [`output::media_output`] wrapped in `RequestStats`; either records the
//!    request's metrics once.
//!
//! Dropping a request's `RequestEvents` cancels the request in the engine
//! that serves it; that is the only cancellation path.
//!
//! [`RequestOutput`]: crate::api::output::RequestOutput
//! [`output::media_output`]: crate::api::output::media_output

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use crate::updates::update::Phase;

use crate::metrics::{EngineLabels, METRICS};
use anyhow::Context as _;
use futures::StreamExt as _;
use proto::request::Request;
use proto::request::wire;
use tokio::sync::watch;
use tracing::{info, warn};

use crate::api::input::media::MediaConnector;
use crate::api::input::video::{ProcessedVideo, VideoService};
use crate::api::input::{
    AdapterSelection, LoadedModel, LoadedRenderer, Processor, RequestRenderer, ServeError,
    TokenizeError,
};
use crate::api::model::ModelConfig;
use crate::api::output::RequestOutputStream;
use crate::api::protocol::VideoGenerationRequest;
use crate::engine::client::{EngineClient, EngineState, StatsSink, SubmitError};
use crate::engine::control::EngineControl;
use crate::engine::hold::{HoldState, RequestHold};
use crate::metrics::log_stats::StatsLogger;
use crate::metrics::stats::{self as metrics, RequestStats};
use crate::migration::{Migrator, RequestEvents};
use crate::options::{ReasoningParser, ServerArgs, Stated};

/// One model served over its engine process.
///
/// Construction spawns Tokio tasks, so every constructor must be called
/// within a Tokio runtime.
pub struct Leader {
    /// The model families the executable serves, which every preparation of an epoch resolves
    /// its checkpoint with.
    pub(crate) families: processing::Registry,
    /// The single model name the API serves.
    served_model_name: String,
    /// The model the request and scheduler metrics are labelled with.
    model_name: String,
    processor: Arc<Processor>,
    /// Independent replica epochs behind the public request entry point.
    pub(crate) replicas: Arc<Replicas>,
    images: MediaConnector,
    /// The updater: the epoch records, the update in flight and the engine
    /// processes it commands. It shares the records with the admin's
    /// `GET /admin/config`.
    pub(crate) updates: Arc<crate::updates::Updater>,
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
    /// The updater, so a replica that loses its engine asks whether an
    /// update in flight owns that decision.
    pub(crate) updates: std::sync::Weak<crate::updates::Updater>,
    /// The new requests that wait for a replica to accept them: while
    /// generation is paused, the engine sleeps, or no engine is ready
    /// (`LeaderStatus::held`).
    pub(crate) held: AtomicUsize,
}

/// Counts one new request in [`Replicas::held`] while it waits for a
/// replica, whichever way the wait ends: served, failed, or dropped with its
/// client.
struct Waiting<'a>(&'a AtomicUsize);

impl<'a> Waiting<'a> {
    fn new(held: &'a AtomicUsize) -> Self {
        held.fetch_add(1, Ordering::Relaxed);
        Self(held)
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
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

/// What a replica does beside an update in flight, by the update's form.
pub(crate) enum Following {
    /// The serving engine moves to other kernel or policy modules in place:
    /// a commit makes `config` and `modules` what a relaunch and a standby
    /// start from.
    Generation {
        config: Box<ServerArgs>,
        modules: crate::epoch::ModuleSet,
    },
    /// The target engines of another epoch serve the deployment from the
    /// first one's `Serving`.
    Epoch {
        /// What starts and joins the replicas' engines.
        control: Arc<EngineControl>,
        /// Each target replica: the arguments it serves with, once it is no
        /// longer an engine that succeeds another, and its world's engines
        /// by host and launch generation, the scheduler node's first.
        targets: Vec<(proto::config::EngineArgs, Vec<(String, u64)>)>,
        /// The configuration the target serves.
        config: Box<ServerArgs>,
        /// The epoch the target serves.
        epoch: u64,
    },
}

/// Records each scheduler's statistics under `model_name`, and with
/// `observer`, which reads each epoch's rates.
pub(crate) fn stats_sink(
    model_name: String,
    observer: Arc<crate::observe::Observer>,
    replica: usize,
) -> StatsSink {
    Box::new(move |epoch, scheduler, snapshot| {
        observer.record(epoch, scheduler, &snapshot);
        metrics::record_scheduler_stats(
            &METRICS.scheduler,
            &model_name,
            scheduler + replica as u32,
            &snapshot,
        );
    })
}

/// How many requests ended since the daemon started, and how many failed.
pub(crate) fn request_counts() -> (u64, u64) {
    let counts = &metrics::OUTCOMES;
    (
        counts.ended.load(Ordering::Relaxed),
        counts.failed.load(Ordering::Relaxed),
    )
}

/// Fixes the module set `named` states: every interface's revision as
/// `local`, this daemon's own host's Build Server, selects it through `cover`
/// ([`crate::engine::control::with_store_modules`]), then every module built
/// at the Build Server of every host in `hosts`, the hosts the epoch's
/// processes run on, before any of them is spawned
/// ([`crate::engine::control::prepare_epoch`]): `local` at its socket, and
/// every other host through its daemon, which `nodes` reaches. A main process
/// a node starts from the set then reads its engine from that node's store,
/// and a spawn builds nothing.
///
/// This is the one place a deployment's packages are selected: when a
/// configuration is applied, at startup and at every later apply. The set it
/// returns is part of the epoch's record and of every launch file made from
/// it.
///
/// The wait ends with the Build Servers' answers: each reports every module's build as it goes
/// (`engine::control::prepare_epoch`), and a host whose daemon is lost fails the preparation.
///
/// # Errors
///
/// `UpdateError::Refused` when a Build Server holds no package, or not the
/// revision, the deployment names, or cannot build a module, and when a host
/// of `hosts` cannot prepare: no daemon runs there, it is lost, or its Build
/// Server prepares other modules than `local`; `UpdateError::Failed` when the
/// preparation's thread ended without an answer.
pub(crate) async fn fix_modules(
    local: crate::engine::control::BuildServer,
    nodes: Option<Arc<EngineControl>>,
    hosts: Vec<String>,
    named: crate::epoch::ModuleSet,
) -> Result<crate::epoch::ModuleSet, UpdateError> {
    let fixed = tokio::task::spawn_blocking(move || {
        let complete = crate::engine::control::with_store_modules(&local, &named)?;
        crate::engine::control::prepare_epoch(&local, nodes.as_deref(), &hosts, &complete)
    })
    .await
    .map_err(|error| UpdateError::Failed(anyhow::anyhow!("fixing the module set: {error}")))?;
    fixed.map_err(|error| UpdateError::Refused(format!("fixing the module set: {error:#}")))
}

/// The modules `config` names for a deployment that serves video or not.
///
/// The configuration selects a backend for attention and for the grouped
/// experts, which are a causal language model's operators. A video deployment
/// launches neither, so a slot filled for them would fix a package no call
/// reaches: every operator it runs is selected from the store.
pub(crate) fn named_modules(config: &ServerArgs, video: bool) -> crate::epoch::ModuleSet {
    let mut named = crate::epoch::modules_of(config);
    if video {
        named.kernels.clear();
    }
    named
}

/// The hosts the processes of an epoch of `config` run on, each once, in the order its replicas
/// and their worlds name them: every replica's scheduler node and its peers'
/// ([`crate::engine::control::world_hosts`]). Each prepares the epoch's module set before any of
/// them is spawned ([`fix_modules`]).
///
/// # Errors
///
/// Fails when the worker groups do not form equal replicas or a replica's ranks cannot be placed.
pub(crate) fn epoch_hosts(config: &proto::config::EngineConfig) -> anyhow::Result<Vec<String>> {
    let mut hosts: Vec<String> = Vec::new();
    for replica in replica_configs(config)? {
        for placed in crate::engine::control::world_hosts(&replica)? {
            if !hosts.contains(&placed.host) {
                hosts.push(placed.host);
            }
        }
    }
    Ok(hosts)
}

/// The engine configuration of each data-parallel replica of `config`: independent replicas each
/// run their own engine process on their own worker groups, while schedulers that share their
/// experts in an expert union stay in one ([`proto::config::EngineConfig::expert_union`]).
/// Replicas whose experts are sharded over each one's own tensor-parallel ranks are independent
/// and keep the flag.
///
/// # Errors
///
/// Fails when the worker groups do not form equal replicas, or when a deployment that shards its
/// groups' experts over their tensor-parallel ranks selects an exchange: no token moves between
/// those ranks, and the exchanges serve expert unions.
pub(crate) fn replica_configs(
    config: &proto::config::EngineConfig,
) -> anyhow::Result<Vec<proto::config::EngineConfig>> {
    let union = config.expert_union();
    let tensor_groups = config.parallel.worker_groups.iter().any(|group| {
        group
            .blocks
            .get(proto::config::DEFAULT_BLOCK)
            .is_some_and(|block| block.tensor_parallel_size > 1)
    });
    anyhow::ensure!(
        !config.moe.expert_parallel
            || union
            || !tensor_groups
            || config.moe.exchange == proto::config::DEFAULT_EXPERT_EXCHANGE,
        "`moe_config.exchange` is {:?} and `moe_config.expert_parallel` shards each worker \
         group's experts over its tensor-parallel ranks, which exchange no token: the exchanges \
         serve schedulers of one rank each and dedicated expert workers \
         (docs/expert_parallelism.md)",
        config.moe.exchange
    );

    let count = config.parallel.data_parallel_size.get();
    if count == 1 || union {
        return Ok(vec![config.clone()]);
    }
    let groups = &config.parallel.worker_groups;
    anyhow::ensure!(
        !groups.is_empty() && groups.len().is_multiple_of(count),
        "the worker groups must form {count} equal independent replicas"
    );
    Ok(groups
        .chunks(groups.len() / count)
        .map(|groups| {
            let mut replica = config.clone();
            replica.parallel.data_parallel_size = std::num::NonZeroUsize::MIN;
            replica.parallel.worker_groups = groups.to_vec();
            replica
        })
        .collect())
}

/// The launch arguments of every replica of an epoch numbered `epoch`: each replica's engine
/// configuration ([`replica_configs`]) running `modules`, every one stating the epoch's one number.
///
/// An epoch numbers one engine build with its build configuration (`docs/admin.md`), however many
/// independent replicas that configuration divides into: their storages, their claims and their
/// buffer pool sessions at a host's Storage Server are the epoch's, a commit retires them together
/// as the previous epoch, and the next epoch's number lies above every one of them. A process
/// that replaces a replica's lost engine, and its standby, go on with the number.
///
/// # Errors
///
/// Fails as [`replica_configs`] fails.
pub(crate) fn epoch_launches(
    config: &proto::config::EngineConfig,
    modules: &crate::epoch::ModuleSet,
    epoch: u64,
) -> anyhow::Result<Vec<proto::config::EngineArgs>> {
    Ok(replica_configs(config)?
        .into_iter()
        .map(|replica| {
            let mut args = proto::config::EngineArgs::new(replica, modules.clone());
            args.epoch = epoch;
            args
        })
        .collect())
}

/// What a leader prepares before it has an engine (`Leader::prepare`).
pub(crate) struct Prepared {
    families: processing::Registry,
    pub(crate) model_config: ModelConfig,
    tokenizer: crate::api::tokenizer::DynTokenizer,
    renderer: LoadedRenderer,
    /// The deployment's engine configuration, with the facts the model's files determine.
    pub(crate) engine: proto::config::EngineConfig,
}

impl Leader {
    pub(crate) fn replicas(&self) -> Vec<Arc<Replica>> {
        self.replicas.serving.borrow().clone()
    }

    /// Assigns a new request to the least loaded ready replica. A request
    /// stays with that replica's migrator through its epoch transitions.
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
            // The request waits on every replica at once, and counts once.
            let _waiting = Waiting::new(&self.replicas.held);
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

    /// Ends every replica's parked standby.
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
        // A handover passes the serving engine, its epoch and the
        // configuration to the successor, not the update in flight: an
        // update does not cross a daemon handover, so the handover waits for
        // it to end.
        if let Some((number, phase)) = self.updates.in_flight() {
            anyhow::bail!(
                "update {number} is {}: a daemon handover waits until the update ends",
                phase.name()
            );
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
        if states.iter().any(|state| state.engine == "serving") {
            state.engine = "serving".to_owned();
            state.reason = None;
        }
        state.update = self
            .updates
            .in_flight()
            .map(|(number, phase)| UpdateStatus {
                number,
                phase: phase.name().to_owned(),
            });
        state.paused = states.iter().all(|state| state.paused);
        state.asleep = states.iter().all(|state| state.asleep);
        state.replacing = self.replicas.replacing.try_lock().is_err()
            || states.iter().any(|state| state.replacing);
        state.in_flight = states.iter().map(|state| state.in_flight).sum();
        state.held = self.replicas.held.load(Ordering::Relaxed);
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

    /// Has every replica's engine sleep ([`Replica::sleep`]).
    ///
    /// # Errors
    ///
    /// The first replica's failure.
    pub async fn sleep(&self) -> Result<(), UpdateError> {
        let _replacing = self.replicas.replacing.lock().await;
        for replica in self.replicas() {
            replica.sleep().await?;
        }
        Ok(())
    }

    /// Has every replica's engine wake ([`Replica::wake`]).
    ///
    /// # Errors
    ///
    /// The first replica's failure.
    pub async fn wake(&self) -> Result<(), UpdateError> {
        let _replacing = self.replicas.replacing.lock().await;
        for replica in self.replicas() {
            replica.wake().await?;
        }
        Ok(())
    }

    /// The LoRA adapter named `name` the workers hold, which a request
    /// naming it as its model computes with.
    pub fn adapter(&self, name: &str) -> Option<crate::api::input::AdapterSelection> {
        service::lock(&self.replicas.adapters)
            .get(name)
            .map(|adapter| crate::api::input::AdapterSelection {
                slot: adapter.slot,
                identity: adapter.identity.clone(),
            })
    }

    /// The names of the LoRA adapters the workers hold.
    pub fn adapter_names(&self) -> Vec<String> {
        service::lock(&self.replicas.adapters)
            .keys()
            .cloned()
            .collect()
    }

    /// Loads the LoRA adapter `body` names (`lora_name`, and its `lora_path`)
    /// into a free slot, as vLLM's `load_lora_adapter` does: every rank
    /// publishes its tensors at the Storage Server under the slot (`adapter`).
    /// A new engine, after a hot update or a relaunch, loads it again before
    /// it serves. What computes with an adapter is a kernel package, so a
    /// request naming it is refused until one serves (`docs/admin.md`).
    ///
    /// # Errors
    ///
    /// Refuses a body without a name, the served model's name and an
    /// adapter loaded already; fails when no engine serves, or with the
    /// reason a worker gave.
    pub async fn load_adapter(
        &self,
        body: serde_json::Value,
    ) -> Result<Vec<serde_json::Value>, UpdateError> {
        let name = body["lora_name"]
            .as_str()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| UpdateError::Refused("load_lora_adapter names no lora_name".to_owned()))?
            .to_owned();
        let path = body["lora_path"]
            .as_str()
            .filter(|path| !path.is_empty())
            .ok_or_else(|| UpdateError::Refused("load_lora_adapter names no lora_path".to_owned()))?
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
            let adapters = service::lock(&self.replicas.adapters);
            if adapters.contains_key(&name) {
                return Err(UpdateError::Refused(format!(
                    "the adapter {name} is loaded already"
                )));
            }
            (1..)
                .find(|slot| adapters.values().all(|adapter| adapter.slot != *slot))
                .unwrap_or(u32::MAX)
        };
        let mut results = Vec::new();
        for client in &clients {
            match client.load_adapter(slot, &path).await {
                Ok(loaded) => results.extend(loaded.iter().map(adapter_report)),
                Err(error) => {
                    // The name becomes public only after every replica loads.
                    // Release any partial loads before returning the failure.
                    for client in &clients {
                        if let Err(cleanup) = client.unload_adapter(slot).await {
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
        service::lock(&self.replicas.adapters).insert(
            name.clone(),
            Adapter {
                slot,
                identity,
                path,
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
        body: serde_json::Value,
    ) -> Result<Vec<serde_json::Value>, UpdateError> {
        let name = body["lora_name"].as_str().unwrap_or_default().to_owned();
        let _deployment = self.replicas.replacing.lock().await;
        let replicas = self.replicas();
        let mut owners = Vec::with_capacity(replicas.len());
        let mut clients = Vec::with_capacity(replicas.len());
        for replica in &replicas {
            owners.push(Arc::clone(&replica.replacing).lock_owned().await);
            clients.push(replica.hold.client().ok_or(UpdateError::NoEngine)?);
        }
        let adapter = service::lock(&self.replicas.adapters)
            .remove(&name)
            .ok_or_else(|| UpdateError::Refused(format!("no adapter {name:?} is loaded")))?;
        let mut results = Vec::new();
        let mut failures = Vec::new();
        for client in clients {
            match client.unload_adapter(adapter.slot).await {
                Ok(unloaded) => results.extend(unloaded.iter().map(adapter_report)),
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
    ) -> Result<Vec<wire::WeightReport>, UpdateError> {
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
    /// start, and `instance` the daemon's instance directory, where the engines' request socket
    /// lies. The engine's start blocks until every rank has loaded the
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
    pub(crate) async fn start(
        families: &processing::Registry,
        config: &ServerArgs,
        stated: &Stated,
        daemon: String,
        instance: &std::path::Path,
        updates: Arc<crate::updates::Updater>,
    ) -> anyhow::Result<Self> {
        let prepared = Self::prepare(families, config, stated).await?;
        info!(parallel = ?prepared.engine.parallel, "starting the engine process");
        let control = Arc::new(EngineControl::new(
            &config.request_socket(instance),
            daemon,
            config.host_identity.clone(),
            config.served_model_name().to_owned(),
            crate::engine::control::ensure_build_server(&config.build_config)?,
        )?);

        // The epoch's whole module set is fixed before any engine starts: the
        // engine, policy and kernel revisions the Build Server's `cover`
        // selects, each module built at every host the epoch runs on. Every
        // launch file states it, so an engine resolves what its profile runs
        // name and serves, and a relaunch and a standby run exactly what their
        // epoch runs.
        let local = crate::engine::control::BuildServer {
            host: control.host().to_owned(),
            socket: config.build_config.socket()?,
            targets: Vec::new(),
        };
        let hosts = epoch_hosts(&prepared.engine)?;
        let named = named_modules(config, prepared.model_config.video().is_some());
        let modules = fix_modules(local, Some(Arc::clone(&control)), hosts, named)
            .await
            .map_err(anyhow::Error::from)
            .context("fixing the deployment's module set")?;
        info!(
            engine = %modules.engine.backend,
            revision = modules.engine.revision,
            kernels = modules.kernels.len(),
            policy = %modules.policy.backend,
            "the epoch's module set is fixed"
        );

        // The deployment's first epoch takes the number of its first engine, and every replica
        // starts in it.
        let launches = epoch_launches(&prepared.engine, &modules, control.next_engine())?;
        let clients =
            futures::future::join_all(launches.into_iter().enumerate().map(|(index, mut args)| {
                let model = config.model().to_owned();
                let starting = Arc::clone(&control);
                tokio::task::spawn_blocking(move || {
                    let observer = Arc::new(crate::observe::Observer::default());
                    let sink = stats_sink(model, Arc::clone(&observer), index);
                    let client = starting.start(args.clone(), sink)?;
                    args.epoch = client.epoch();
                    Ok::<_, anyhow::Error>((client, args, observer))
                })
            }))
            .await
            .into_iter()
            .map(|started| started.context("the engine startup task did not complete")?)
            .collect::<anyhow::Result<Vec<_>>>()
            .context("failed to start the UniServe engine")?;
        let epoch = clients.first().map_or(0, |(client, _, _)| client.epoch());
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
                leader.families.clone(),
                config,
                Arc::new(client),
                observer,
                index + 1,
                Arc::downgrade(&leader.replicas),
                Arc::clone(&leader.replicas.adapters),
            );
            let _ = replica.control.set(Arc::clone(&control));
            *service::lock(&replica.served) = Some(served);
            replica.monitor();
            leader
                .replicas
                .serving
                .send_modify(|replicas| replicas.push(replica));
        }

        // The daemon's record of what executes: the epoch's build
        // configuration and the module set its launch files stated, which
        // `GET /admin/config` reports and a later change is compared with.
        let deployment = crate::epoch::Epoch::with_modules(config.clone(), modules)?;
        leader.epochs().install(epoch, &deployment, &[]);
        // The process table names every engine this daemon manages, and the
        // serving one is the first: an update addresses the engine it drains
        // by this launch generation, and `GET /admin/config` lists it.
        if let Some(engine) = leader
            .replicas
            .serving
            .borrow()
            .first()
            .and_then(|replica| replica.hold().client())
            .and_then(|client| client.engine())
        {
            let mut processes = leader.epochs().processes();
            processes.retain(|process| process.generation != engine);
            processes.push(crate::epoch::ManagedProcess {
                group: crate::engine::control::ENGINE_GROUP.to_owned(),
                rank: 0,
                pid: None,
                epoch: Some(epoch),
                build: deployment.engine.executable.clone(),
                generation: engine,
                state: crate::epoch::ProcessState::Serving,
            });
            leader.epochs().set_processes(processes);
        }
        // A serving epoch's tensors are what the next epoch starts from:
        // until it is published its storage is private to it, and a target
        // of another epoch finds nothing.
        let publishing = Arc::clone(&control);
        tokio::task::spawn_blocking(move || publishing.publish_epochs(&[epoch], &mut None))
            .await
            .context("publishing the first epoch")?
            .context("publishing the first epoch")?;
        info!(epoch, "the engines serve the epoch's module set");
        Ok(leader)
    }

    /// The modules `config` names for what this deployment launches.
    pub(crate) fn named_modules(&self, config: &ServerArgs) -> crate::epoch::ModuleSet {
        named_modules(config, self.video_service().is_some())
    }

    /// What a leader prepares before it has an engine: the model's files
    /// and the engine configuration of the engine that serves it, as
    /// [`Leader::start`] describes.
    ///
    /// # Errors
    ///
    /// As [`Leader::start`], before the engine starts.
    pub(crate) async fn prepare(
        families: &processing::Registry,
        config: &ServerArgs,
        stated: &Stated,
    ) -> anyhow::Result<Prepared> {
        let LoadedModel {
            config: model_config,
            tokenizer,
            renderer,
        } = ModelConfig::load(
            config.model(),
            config.max_model_len,
            &config.chat_config,
            families,
        )
        .await
        .with_context(|| format!("failed to resolve model assets for `{}`", config.model()))?;
        // Options of another model family than the served one are refused
        // rather than ignored.
        stated.refuse_foreign_options(&model_config.family)?;
        let max_model_len = model_config.max_model_len;

        // The options state the scheduler and the rank layout; the loaded model
        // completes what it determines, a media pipeline's prompt length
        // included, which its deployment must state (`ModelConfig::from_pipeline`).
        let mut engine = config.engine_config()?;
        let control_tokens = special_token_ids(&model_config);
        let model = &mut engine.model;
        model.base = base_release(&model.model)?;
        model.max_model_len = max_model_len;
        model.generation_limits = model_config.generation_limits(config.dtype);
        model.bos = control_tokens.bos;
        model.eos = control_tokens.eos;
        // Video workers prepare exactly the configured sizes, refusing one
        // their model does not generate, and report them back as the only
        // sizes requests may target; the family states the sizes and the
        // longest output a configuration leaves unset. Other models prepare
        // none.
        match model_config.video() {
            Some(family) => {
                let video = &mut engine.video;
                video.sizes.get_or_insert_with(|| {
                    family
                        .default_sizes()
                        .iter()
                        .map(|&size| size.to_owned())
                        .collect()
                });
                video
                    .max_seconds
                    .get_or_insert_with(|| family.default_max_seconds());
            }
            None => engine.video.sizes = None,
        }

        Ok(Prepared {
            families: families.clone(),
            model_config,
            tokenizer,
            renderer,
            engine,
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
        updates: Arc<crate::updates::Updater>,
    ) -> anyhow::Result<Self> {
        let Prepared {
            families,
            model_config,
            tokenizer,
            renderer,
            engine: _,
        } = prepared;
        let model_name = config.model().to_owned();
        let info = client.info().clone();

        // A video checkpoint serves what its placed denoiser declares in the
        // worker handshake, prepared against the conditioner's vision
        // processor.
        // A chat family that takes images lowers them through its bound processor.
        let mut images = None;
        let renderer = match renderer {
            LoadedRenderer::Chat {
                renderer,
                images: processor,
            } => {
                images = processor;
                // No other served model's processor takes options.
                anyhow::ensure!(
                    config.mm_processor_kwargs.is_empty(),
                    "invalid mm_processor_kwargs: the processor of `{}` takes no options",
                    config.model()
                );
                RequestRenderer::Chat(*renderer)
            }
            LoadedRenderer::Video { family, files } => {
                anyhow::ensure!(
                    !info.video_denoiser.is_empty(),
                    "the video worker reported no video denoiser"
                );
                let processor = family
                    .bind(processing::VideoBinding {
                        description: &info.video_denoiser,
                        max_seconds: served
                            .engine_config
                            .video
                            .max_seconds
                            .context("the leader states a video deployment's longest output")?,
                        processor_options: &config.mm_processor_kwargs,
                        files: &files,
                    })
                    .map_err(|message| anyhow::anyhow!("binding the video processor: {message}"))?;
                RequestRenderer::Video(
                    VideoService::new(processor, &config.media_config, Arc::clone(&tokenizer))
                        .context("failed to bind the video service")?,
                )
            }
        };

        let processor = Processor::new(
            model_config,
            tokenizer,
            renderer,
            info.limits.clone(),
            config.chat_config.reasoning_parser == ReasoningParser::Auto,
        )
        .context("failed to bind the configured model description")?
        .with_images(images);
        let images = MediaConnector::from_image_options(&config.image_fetch_config)
            .context("failed to initialize the image URL fetcher")?;
        let engine = Self::observed(
            families,
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
        *service::lock(&replica.served) = Some(served);
        Ok(engine)
    }

    /// Serves `processor`'s model under `served_model_name` over the
    /// independent replicas in `clients`, labelling metrics with `model_name`.
    /// `images` resolves the image references of every request, and `options`
    /// states the request and statistics logging (`logging_config`) and
    /// request migration (`migration_config`).
    #[cfg(test)]
    pub(crate) fn new(
        served_model_name: String,
        model_name: String,
        clients: Vec<Arc<EngineClient>>,
        processor: Processor,
        images: MediaConnector,
        options: &ServerArgs,
        updates: Arc<crate::updates::Updater>,
    ) -> Self {
        Self::observed(
            processing::Registry::default(),
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
    /// for its epoch's trials.
    #[expect(
        clippy::too_many_arguments,
        reason = "a leader's model, engine and options, and its statistics' observer"
    )]
    fn observed(
        families: processing::Registry,
        served_model_name: String,
        model_name: String,
        clients: Vec<Arc<EngineClient>>,
        processor: Processor,
        images: MediaConnector,
        options: &ServerArgs,
        observer: Arc<crate::observe::Observer>,
        updates: Arc<crate::updates::Updater>,
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
                            families.clone(),
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
            updates: Arc::downgrade(&updates),
            held: AtomicUsize::new(0),
        });
        for replica in replicas.serving.borrow().iter() {
            replica.monitor();
        }
        Self {
            families,
            served_model_name,
            model_name,
            processor: Arc::new(processor),
            replicas,
            images,
            updates,
            log_requests: logging.log_requests,
            _stats_logger: stats_logger,
        }
    }

    /// The loaded model's input processor.
    pub fn processor(&self) -> &Processor {
        &self.processor
    }

    /// The single model name the API serves.
    pub fn served_model_name(&self) -> &str {
        &self.served_model_name
    }

    /// The video service of this deployment; `None` for a model that
    /// generates no video.
    pub fn video_service(&self) -> Option<&VideoService> {
        self.processor.video_service()
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

    /// Lowers one chat request through the deployment's `Processor`, submits it, and
    /// returns what its response body is built from: the request's engine events and the
    /// output processor that turns them into the response named `request_id`.
    ///
    /// The caller drives the body one engine event at a time
    /// (`openai::run::GenerationRun`), so the response can stop at an engine
    /// event boundary and go on in a successor.
    ///
    /// The request's `image_url` parts are fetched only for a model that
    /// accepts image input; for any other model, input processing refuses them
    /// without network access. `adapter` is the LoRA adapter the request's
    /// model names, when it names one.
    ///
    /// # Errors
    ///
    /// Fails when an image cannot be fetched, input processing refuses the
    /// request, or no scheduler accepts it.
    pub(crate) async fn generate(
        &self,
        request_id: String,
        request: crate::api::protocol::ChatCompletionRequest,
        adapter: Option<AdapterSelection>,
    ) -> Result<
        (
            Box<dyn crate::api::events::Events>,
            crate::api::output::OutputProcessor,
        ),
        crate::api::openai::ApiError,
    > {
        let started = Instant::now();
        let image_urls = if self.processor.supports_image_input() {
            request.image_urls()
        } else {
            Vec::new()
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
        let (request, mut lowering) = preprocess(&request_id, {
            let request_id = request_id.clone();
            move || processor.process(&request_id, &request, adapter.as_ref(), images)
        })
        .await?;

        lowering.compile_us = elapsed_us(started);
        let (events, core) = self
            .submit(request.into())
            .await
            .map_err(ServeError::from)?;
        let labels = EngineLabels {
            model_name: self.model_name.clone(),
            engine: core,
        };
        let log = self.log_requests.then_some(request_id);
        let stats = crate::metrics::stats::StatLogger::new(labels, started, log);
        // Dropping the events on a failure cancels the request in its engine.
        let output = crate::api::output::OutputProcessor::new(
            self.processor.tokenizer(),
            lowering,
            Box::new(stats),
        )?;

        Ok((Box::new(events), output))
    }

    /// The model's tokenizer, which a successor builds the decoder of a
    /// handoff's body from.
    pub(crate) fn tokenizer(&self) -> crate::api::tokenizer::DynTokenizer {
        self.processor.tokenizer()
    }

    /// The statistics recorder of a request this process takes over, under
    /// the replica that serves it and the instant it entered the deployment.
    pub(crate) fn stat_logger(
        &self,
        request_id: &str,
        replica: u32,
        started: std::time::Instant,
    ) -> crate::metrics::stats::StatLogger {
        let labels = EngineLabels {
            model_name: self.model_name.clone(),
            engine: replica,
        };
        let log = self.log_requests.then(|| request_id.to_owned());
        crate::metrics::stats::StatLogger::new(labels, started, log)
    }

    /// Takes one request's engine event stream over from the process that
    /// consumed it, going on after the position it reached:
    /// the request keeps generating in the engine that serves it,
    /// and is neither cancelled nor submitted again. Both kinds of response
    /// in a handoff call it, with the replica and identity their state carries.
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
            .with_context(|| format!("the response's handoff names replica {engine_replica}"))?;
        let client = replica
            .hold
            .client()
            .context("the replica of the response's handoff has no engine connection")?;
        let id = proto::request::RequestId(
            engine_request_id
                .parse()
                .context("the engine identity of the response's handoff")?,
        );
        let stream = client.take_over(id, consumed)?;
        Ok(RequestEvents::taken_over(stream, consumed))
    }

    /// Fetches, probes and prepares a video request's media, presents its
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
        let (request, processed) = video
            .process(&request_id, &request, self.processor.config().max_model_len)
            .await?;
        let prompt_tokens = request.prompt_token_ids.len();
        let (events, core) = self.submit(Request::Diffusion(request)).await?;
        let output = crate::api::output::media_output(
            request_id.clone(),
            started,
            elapsed_us(started),
            prompt_tokens,
            Box::new(events),
        );
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
    /// The epochs this node keeps: the records the admin API reports and an
    /// update writes.
    pub(crate) fn epochs(&self) -> Arc<crate::epoch::Epochs> {
        Arc::clone(self.updates.epochs())
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
    pub(crate) unavailable: Vec<crate::updates::Unavailable>,
}

/// What the leader serves, for `GET /admin/status`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct LeaderStatus {
    /// The served model name, the deployment's name.
    pub deployment: String,
    /// The model the deployment serves.
    pub model: String,
    /// The lifecycle state of the engine requests go to: `starting`,
    /// `parked`, `serving`, `draining`, `stopped` or `failed`; `holding`
    /// while an engine is being replaced; `closed` once none will serve.
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
    /// The new requests the daemon holds before any engine takes them: while
    /// generation is paused, the engine sleeps, or no engine is ready.
    pub held: usize,
    /// The launch generation of the engine that serves (the scheduler
    /// node's, for a worker group over several hosts); `None` while none
    /// serves, or for an engine this daemon did not start. It names that
    /// process's row in `GET /admin/config`'s `processes`.
    pub generation: Option<u64>,
    /// The launch generation of each engine of the parked standby beside
    /// the serving engine (`worker_config.standby`), one per host in the
    /// world's order: the serving `generation` a promotion leaves is its
    /// first. Empty while there is none.
    pub standby: Vec<u64>,
    /// The update in flight; `None` when none is.
    pub update: Option<UpdateStatus>,
}

/// The update in flight, for `GET /admin/status`: `GET /admin/config` holds
/// its whole report.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct UpdateStatus {
    /// The admin operation that applied it.
    pub number: u64,
    /// `preparing`, `switching` or `trial`.
    pub phase: String,
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

/// The deployment `config` describes, refused as a request when it does not resolve: its
/// deployment file does not read or does not place.
pub(crate) fn deployment(config: &ServerArgs) -> Result<crate::epoch::Epoch, UpdateError> {
    crate::epoch::Epoch::of(config.clone()).map_err(|error| {
        UpdateError::Refused(format!("the configuration does not resolve: {error:#}"))
    })
}

/// Why a change of the deployment did not happen or did not complete.
#[derive(Debug, thiserror::Error)]
pub enum UpdateError {
    /// Another change is in progress.
    #[error("a change of the deployment is in progress")]
    InProgress,
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
    use crate::api::model::assets::model_files::ResolvedModelFiles;
    // A diffusers checkpoint keeps what processes a request in directories of its own, and states
    // no tokenizer at its root for the model's files to resolve.
    let pipeline = std::path::Path::new(served)
        .join(processing::RootFile::ModularIndex.name())
        .is_file();
    if pipeline {
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

/// The parts of a diffusers checkpoint that decide how a request is processed.
const PROCESSING_PARTS: [&str; 4] = ["tokenizer", "processor", "scheduler", "audio_scheduler"];

/// [`same_processing`] for a diffusers checkpoint: the same index, and every file of the parts
/// that process a request the same.
fn same_pipeline(served: &std::path::Path, path: &std::path::Path) -> Result<(), String> {
    let read = |file: &std::path::Path| {
        std::fs::read(file).map_err(|error| format!("{}: {error}", file.display()))
    };
    // The index either checkpoint publishes, the same in both.
    let published = |root: &std::path::Path, index: &str| root.join(index).is_file();
    for index in [processing::RootFile::ModularIndex.name()] {
        let same = match (published(served, index), published(path, index)) {
            (false, false) => true,
            (true, true) => read(&served.join(index))? == read(&path.join(index))?,
            _ => false,
        };
        if !same {
            return Err(format!(
                "the checkpoint at {} states other parts than the served model: deploy it instead",
                path.display()
            ));
        }
    }
    for part in PROCESSING_PARTS {
        let directory = served.join(part);
        let entries = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(format!("{}: {error}", directory.display())),
        };
        for entry in entries {
            let entry = entry.map_err(|error| format!("{}: {error}", directory.display()))?;
            let name = entry.file_name();
            if read(&entry.path())? != read(&path.join(part).join(&name))? {
                return Err(format!(
                    "the checkpoint at {} has another {part}/{} than the served model, \
                     which changes how requests are processed: deploy it instead",
                    path.display(),
                    name.to_string_lossy()
                ));
            }
        }
    }
    Ok(())
}

/// One independent replica: the engine its requests go to, the arguments
/// that engine serves with, and its standby.
pub(crate) struct Replica {
    /// The model families the executable serves, which an update of this replica resolves its
    /// target checkpoint with.
    pub(crate) families: processing::Registry,
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
    /// Held while an engine is replaced, by an epoch change, a commit, a
    /// rollback or a relaunch, so that replacements do not overlap.
    pub(crate) replacing: Arc<tokio::sync::Mutex<()>>,
    /// The engine gave back its KV cache and graph storage (`Leader::sleep`).
    pub(crate) asleep: AtomicBool,
    /// The sleep paused new requests, which the wake lets through.
    sleep_paused: AtomicBool,
    /// The LoRA adapters the workers hold, by name (`Leader::load_adapter`).
    pub(crate) adapters: Arc<std::sync::Mutex<BTreeMap<String, Adapter>>>,
    /// Every engine's scheduler statistics, which the status reads.
    pub(crate) observer: Arc<crate::observe::Observer>,
    /// The profiling session that runs, or the last that ended
    /// (`crate::profiling`).
    pub(crate) profiles: crate::profiling::Profiles,
    /// The parked standby standing beside the serving engine, if one is
    /// live. Started by [`Replica::keep_standby`], taken by
    /// [`Replica::promote_standby`].
    pub(crate) standby: std::sync::Mutex<Option<Standby>>,
    /// The failure-domain rules over the serving engine's processes
    /// (`crate::standby::Standbys`), which decide whether its standby starts,
    /// is promoted or is ended; built for the engine that serves, and anew
    /// when another serves.
    domain: std::sync::Mutex<Option<crate::standby::Standbys>>,
    /// Serializes starting a standby: two callers that both saw none must not
    /// both start a process, since a second engine of the same scope is a
    /// whole copy of its device storage.
    standby_starting: tokio::sync::Mutex<()>,
    /// Whether this daemon handed the engine to a successor: the
    /// successor owns its life then, and a watch that sees its client go away
    /// must not relaunch it.
    pub(crate) handoff_sent: AtomicBool,
    /// Whether an epoch update owns this replica's engines: from the moment its target serves
    /// until the update ended and the hold names the epoch that serves on. An engine lost
    /// meanwhile is the update's to judge, not the watcher's to relaunch.
    following: AtomicBool,
}

impl Replica {
    pub(crate) fn new(
        families: processing::Registry,
        options: &ServerArgs,
        client: Arc<EngineClient>,
        observer: Arc<crate::observe::Observer>,
        ordinal: usize,
        registry: std::sync::Weak<Replicas>,
        adapters: Arc<std::sync::Mutex<BTreeMap<String, Adapter>>>,
    ) -> Arc<Self> {
        let hold = Arc::new(RequestHold::new());
        let replica = Arc::new(Replica {
            families,
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
            observer,
            profiles: crate::profiling::Profiles::default(),
            standby: std::sync::Mutex::new(None),
            domain: std::sync::Mutex::new(None),
            standby_starting: tokio::sync::Mutex::new(()),
            handoff_sent: AtomicBool::new(false),
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

    /// Follows update `number` for this replica until it ends: what the
    /// daemon does beside the update's own phase machine.
    ///
    /// For both forms it reads the trial's thresholds (`options.rollback`)
    /// and rolls the update back at its `deadline`. For the generation form
    /// it keeps the committed configuration and module set, and replaces the
    /// standby with one of them. For the epoch form, which the first replica
    /// follows for the whole deployment, it joins every target replica's
    /// engines, routes new requests to the target from its first `Serving`,
    /// and back to the previous epoch from its first `Serving` again when the
    /// update rolls back.
    ///
    /// The task wakes on the update's own progress
    /// ([`crate::updates::Updater::watch`]), on the deadline and, during the
    /// trial, on the interval the thresholds are read at.
    pub(crate) fn follow(
        self: &Arc<Self>,
        updater: Arc<crate::updates::Updater>,
        number: u64,
        options: crate::options::UpdateConfig,
        deadline: Option<Instant>,
        following: Following,
    ) {
        let replica = Arc::clone(self);
        tokio::spawn(async move {
            match following {
                Following::Generation { config, modules } => {
                    replica
                        .follow_generation(&updater, number, &options, deadline, *config, modules)
                        .await;
                }
                Following::Epoch {
                    control,
                    targets,
                    config,
                    epoch,
                } => {
                    // The epoch form's follower holds both epochs' launch
                    // arguments and configurations across its whole wait.
                    Box::pin(replica.follow_epoch(
                        &updater, number, &options, deadline, control, targets, *config, epoch,
                    ))
                    .await;
                }
            }
        });
    }

    /// Waits for the next thing an update's follower acts on: the update's
    /// progress, its `deadline` (which rolls it back, once), or, once its
    /// trial began (`on_trial`), the next reading of the trial's thresholds
    /// (a threshold passed rolls it back). Returns update `number`'s phase
    /// afterwards; `None` once another update, or none, is what the updater
    /// states.
    async fn next_phase(
        updater: &crate::updates::Updater,
        number: u64,
        phases: &mut watch::Receiver<Option<(u64, Phase)>>,
        deadline: &mut Option<Instant>,
        on_trial: bool,
    ) -> Option<Phase> {
        let expiry = async {
            match *deadline {
                Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
                None => std::future::pending().await,
            }
        };
        let reading = async {
            if on_trial {
                tokio::time::sleep(crate::updates::trial::INTERVAL).await;
            } else {
                std::future::pending::<()>().await;
            }
        };
        tokio::select! {
            changed = phases.changed() => {
                if changed.is_err() {
                    return None;
                }
            }
            () = expiry => {
                *deadline = None;
                updater.expire(number);
            }
            () = reading => {
                if let Some(failure) = updater.read_trial(number, request_counts()) {
                    warn!(
                        number,
                        kind = %failure.kind,
                        reason = %failure.reason,
                        "the update on trial passed a rollback threshold; it rolls back"
                    );
                }
            }
        }
        // The watch only wakes: the phase is the update's own, which the
        // updater keeps after the update ended and another began.
        phases.borrow_and_update();
        updater.phase_of(number)
    }

    /// [`Self::follow`] for the generation form.
    async fn follow_generation(
        self: &Arc<Self>,
        updater: &Arc<crate::updates::Updater>,
        number: u64,
        options: &crate::options::UpdateConfig,
        mut deadline: Option<Instant>,
        config: ServerArgs,
        modules: crate::epoch::ModuleSet,
    ) {
        let mut phases = updater.watch();
        let mut on_trial = false;
        let mut phase = updater.phase_of(number);
        loop {
            match phase {
                Some(Phase::Trial) if !on_trial => {
                    // Serving and the update share an epoch, so the baseline
                    // ends where the switch began.
                    let switch = updater.switch_began().unwrap_or_else(Instant::now);
                    let observed = self.hold.client().map(|client| {
                        crate::updates::trial::Baseline::new(
                            Arc::clone(&self.observer),
                            client.epoch(),
                            client.epoch(),
                            options.rollback.window,
                            switch,
                        )
                    });
                    on_trial = updater.begin_trial(crate::updates::trial::Trial::begin(
                        options.rollback.clone(),
                        observed.into_iter().collect(),
                        request_counts(),
                    ));
                }
                Some(Phase::Committed) => {
                    // The configuration and the module set are what the next
                    // apply compares with, what a relaunch and a standby
                    // start from and what a successor daemon takes over.
                    *service::lock(&self.config) = config;
                    if let Some(args) = service::lock(&self.served).as_mut() {
                        args.modules = modules;
                    }
                    // The parked standby holds the generation the commit
                    // released: it is replaced by one of the committed set.
                    self.restart_standby();
                    return;
                }
                // Through the trial the launch arguments and the standby keep
                // the previous module set, which is what serves again.
                Some(Phase::RolledBack | Phase::Abandoned) | None => return,
                Some(Phase::Preparing | Phase::Switching | Phase::Trial) => {}
            }
            phase = Self::next_phase(updater, number, &mut phases, &mut deadline, on_trial).await;
        }
    }

    /// [`Self::follow`] for the epoch form, for the deployment's every
    /// replica.
    ///
    /// The two epochs need not have as many replicas: target replica `j` is
    /// served under the deployment's replica `j`, made when the serving epoch
    /// has fewer, and a serving replica past the target's last hands the
    /// requests it migrates to a target replica as its own route; a rollback
    /// does the same the other way. Each replica's hold serves its new engine
    /// at once, and requests go to the replicas whose engine serves, so none
    /// waits for a target that is not yet ready.
    #[allow(clippy::too_many_arguments)]
    async fn follow_epoch(
        self: &Arc<Self>,
        updater: &Arc<crate::updates::Updater>,
        number: u64,
        options: &crate::options::UpdateConfig,
        mut deadline: Option<Instant>,
        control: Arc<EngineControl>,
        targets: Vec<(proto::config::EngineArgs, Vec<(String, u64)>)>,
        config: ServerArgs,
        epoch: u64,
    ) {
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        let previous: Vec<Arc<Replica>> = registry.serving.borrow().clone();
        let mut phases = updater.watch();

        // While the targets prepare, their transfers yield to the serving
        // epoch's steps (`update_config.transfer_slowdown`); the guard ends
        // that however the preparation ends.
        let preparing = tokio_util::sync::CancellationToken::new();
        let preparation = preparing.clone().drop_guard();
        let throttled: Vec<_> = previous
            .iter()
            .filter_map(|replica| {
                // Without a stated slowdown the transfers take their links' whole budgets.
                let slowdown = options.transfer_slowdown?;
                let serving = replica.hold.client()?;
                let throttling = crate::interference::Throttling::new(
                    &replica.observer,
                    serving.epoch(),
                    slowdown,
                    Instant::now(),
                );
                Some((Arc::clone(&replica.observer), throttling))
            })
            .collect();
        let steps = Arc::new(std::sync::Mutex::new(
            crate::interference::StepTime::default(),
        ));
        updater.watch_transfers(Arc::clone(&steps));
        tokio::spawn(crate::interference::control(
            Arc::clone(&control) as Arc<dyn crate::interference::Throttle>,
            throttled,
            crate::interference::INTERVAL,
            preparing,
            steps,
        ));

        // Every target replica's engines dial this daemon's request socket as
        // they start, and state that they parked once they resolved their
        // modules and captured their graphs.
        let observers: Vec<Arc<crate::observe::Observer>> = (0..targets.len())
            .map(|index| {
                previous
                    .get(index)
                    .map_or_else(Default::default, |replica| Arc::clone(&replica.observer))
            })
            .collect();
        let joining =
            targets
                .iter()
                .zip(&observers)
                .enumerate()
                .map(|(index, (target, observer))| {
                    let sink = stats_sink(self.model_name.clone(), Arc::clone(observer), index);
                    let (control, processes) = (Arc::clone(&control), target.1.clone());
                    tokio::task::spawn_blocking(move || {
                        control.join_target(&processes, epoch, sink)
                    })
                });
        let mut clients = Vec::with_capacity(targets.len());
        for joined in futures::future::join_all(joining).await {
            let reason = match joined {
                Ok(Ok(client)) => {
                    clients.push(Arc::new(client));
                    continue;
                }
                Ok(Err(error)) => format!("the update's target did not join: {error:#}"),
                Err(error) => format!("joining the update's target ended: {error}"),
            };
            warn!(epoch, "{reason}");
            // Before the switch a rollback abandons the update; nothing that
            // serves changed.
            updater.rollback(crate::epoch::Failure {
                kind: "prepare_failed".to_owned(),
                reason,
                device: None,
            });
            return;
        }

        // Until a target serves, the deployment's requests are the previous
        // epoch's.
        let mut served = updater.epochs().served();
        let mut phase = updater.phase_of(number);
        loop {
            if *served.borrow_and_update() >= epoch {
                break;
            }
            if phase.is_none_or(Phase::ended) {
                return;
            }
            tokio::select! {
                changed = served.changed() => {
                    if changed.is_err() {
                        return;
                    }
                }
                next = Self::next_phase(updater, number, &mut phases, &mut deadline, false) => {
                    phase = next;
                }
            }
        }
        drop(preparation);

        // From here the update owns both epochs' engines: an engine lost is
        // the update's to judge, not the watcher's to relaunch. What the
        // previous epoch served with stays until the update ends: it serves
        // again on a rollback.
        let before: Vec<_> = previous
            .iter()
            .map(|replica| {
                (
                    replica.hold.client(),
                    service::lock(&replica.served).clone(),
                )
            })
            .collect();
        let config_before = service::lock(&self.config).clone();
        let mut routed = Vec::with_capacity(clients.len());
        for (index, (client, (args, _))) in clients.iter().zip(&targets).enumerate() {
            let replica = match previous.get(index) {
                Some(replica) => Arc::clone(replica),
                None => {
                    let replica = Replica::new(
                        self.families.clone(),
                        &config,
                        Arc::clone(client),
                        Arc::clone(&observers[index]),
                        index,
                        Arc::downgrade(&registry),
                        Arc::clone(&self.adapters),
                    );
                    let _ = replica.control.set(Arc::clone(&control));
                    replica
                }
            };
            *service::lock(&replica.served) = Some(args.clone());
            *service::lock(&replica.config) = config.clone();
            routed.push(replica);
        }
        // Requests go to `serving`; each of `replicas` (`serving` first) has its
        // hold serve its own engine of `clients`, and one past the last a
        // client of another, to which it migrates what it holds.
        let route = |replicas: &[Arc<Replica>],
                     serving: &[Arc<Replica>],
                     clients: &[Option<Arc<EngineClient>>]| {
            for (index, replica) in replicas.iter().enumerate() {
                replica.following.store(true, Ordering::SeqCst);
                let client = clients
                    .get(index)
                    .or_else(|| clients.get(index % clients.len().max(1)));
                if let Some(client) = client.cloned().flatten() {
                    replica.hold.serve(client);
                }
            }
            registry.serving.send_replace(serving.to_vec());
        };
        let target_clients: Vec<_> = clients.iter().cloned().map(Some).collect();
        let mut everyone = routed.clone();
        everyone.extend(previous.iter().skip(routed.len()).cloned());
        route(&everyone, &routed, &target_clients);
        for (replica, client) in routed.iter().zip(&clients) {
            tokio::spawn(watch_engine(Arc::clone(replica), Arc::downgrade(client)));
        }
        info!(
            epoch,
            replicas = routed.len(),
            "the update's target serves the deployment"
        );

        // The trial: the steps the previous epoch served before the switch
        // are what the target's are read against, replica by replica, from
        // when the switch ended.
        let previous_epoch = before
            .iter()
            .find_map(|(client, _)| client.as_ref())
            .map_or(0, |client| client.epoch());
        let mut on_trial = false;

        // The previous engines' ends, when the trial stops them: their nodes
        // then keep the weights alone.
        let mut ending: Vec<(Vec<u64>, watch::Receiver<EngineState>)> =
            if options.previous_epoch == crate::options::PreviousEpoch::Parked {
                Vec::new()
            } else {
                before
                    .iter()
                    .filter_map(|(client, _)| client.as_ref())
                    .map(|client| (launches(client), client.states()))
                    .collect()
            };
        let mut serving_again: Vec<Option<Arc<EngineClient>>> =
            before.iter().map(|(client, _)| client.clone()).collect();
        let mut relaunched = vec![false; previous.len()];
        let mut returned = false;
        let mut phase = updater.phase_of(number);
        while phase.is_some_and(|phase| !phase.ended()) {
            // The trial begins once every serving engine drained: the target's
            // steps while both epochs stepped, its warm-up and the first steps
            // of the requests migrated to it are not its trial's.
            if phase == Some(Phase::Trial) && !on_trial {
                let switch = updater.switch_began().unwrap_or_else(Instant::now);
                let baselines = routed
                    .iter()
                    .take(previous.len())
                    .map(|replica| {
                        crate::updates::trial::Baseline::new(
                            Arc::clone(&replica.observer),
                            epoch,
                            previous_epoch,
                            options.rollback.window,
                            switch,
                        )
                    })
                    .collect();
                on_trial = updater.begin_trial(crate::updates::trial::Trial::begin(
                    options.rollback.clone(),
                    baselines,
                    request_counts(),
                ));
            }
            // A rollback of an epoch whose engines the trial stopped, or that
            // exited, starts them again on the weights they kept, a replica's
            // world at a time.
            let due = updater.relaunch_due();
            let starting: Vec<usize> = (0..previous.len())
                .filter(|index| {
                    !relaunched[*index]
                        && before[*index].0.as_ref().is_some_and(|client| {
                            launches(client).iter().any(|launch| due.contains(launch))
                        })
                })
                .collect();
            if !starting.is_empty() {
                for index in starting {
                    relaunched[index] = true;
                    let replica = &previous[index];
                    let stopped = before[index].0.as_deref().map(launches).unwrap_or_default();
                    let started = replica
                        .relaunch_previous(&control, updater, before[index].1.clone(), &stopped)
                        .await;
                    match started {
                        Ok(started) => {
                            tokio::spawn(watch_engine(
                                Arc::clone(replica),
                                Arc::downgrade(&started),
                            ));
                            serving_again[index] = Some(started);
                        }
                        Err(error) => {
                            let reason =
                                format!("the previous epoch did not start again: {error:#}");
                            tracing::error!(%reason, "no engine serves the replica's requests");
                            serving_again[index] = None;
                            replica.hold.close(reason);
                        }
                    }
                }
                returned = false;
                phase = updater.phase_of(number);
                continue;
            }
            // From the previous epoch's first `Serving` of a rollback, new
            // requests return to it.
            if !returned && *served.borrow_and_update() < epoch {
                returned = true;
                for (replica, (_, args)) in previous.iter().zip(&before) {
                    *service::lock(&replica.served) = args.clone();
                    *service::lock(&replica.config) = config_before.clone();
                }
                let mut everyone = previous.clone();
                everyone.extend(routed.iter().skip(previous.len()).cloned());
                route(&everyone, &previous, &serving_again);
                warn!(
                    epoch,
                    "the update returns the deployment to the previous epoch"
                );
            }
            let ended = async {
                if ending.is_empty() {
                    return std::future::pending().await;
                }
                let waits = ending.iter_mut().map(|(_, states)| {
                    Box::pin(async move {
                        while *states.borrow_and_update() != EngineState::Stopped {
                            if states.changed().await.is_err() {
                                break;
                            }
                        }
                    })
                });
                futures::future::select_all(waits).await.1
            };
            tokio::select! {
                index = ended => {
                    let (stopped, _) = ending.swap_remove(index);
                    for launch in stopped {
                        updater.previous_ended(launch);
                    }
                }
                changed = served.changed() => {
                    if changed.is_err() {
                        break;
                    }
                }
                next = Self::next_phase(
                    updater,
                    number,
                    &mut phases,
                    &mut deadline,
                    on_trial,
                ) => phase = next,
            }
        }

        if phase == Some(Phase::Committed) {
            // The committed target is what the next update starts from.
            let publishing = Arc::clone(&control);
            let published =
                tokio::task::spawn_blocking(move || publishing.publish_epochs(&[epoch], &mut None))
                    .await;
            if !matches!(published, Ok(Ok(()))) {
                warn!(
                    epoch,
                    "publishing the committed epoch failed: {published:?}"
                );
            }
        } else if !returned {
            for (replica, (_, args)) in previous.iter().zip(&before) {
                *service::lock(&replica.served) = args.clone();
                *service::lock(&replica.config) = config_before.clone();
            }
            let mut everyone = previous.clone();
            everyone.extend(routed.iter().skip(previous.len()).cloned());
            route(&everyone, &previous, &serving_again);
        }
        for replica in previous.iter().chain(&routed) {
            replica.following.store(false, Ordering::SeqCst);
        }
        // The standby of the epoch that no longer serves ended with it
        // (`worker_config.standby`).
        for replica in registry.serving.borrow().iter() {
            replica.start_standby();
        }
    }

    /// Starts the previous epoch's world of this replica again for a
    /// rollback: its engines `stopped` (their launch generations, in world
    /// order) exited for the trial (`update_config.previous_epoch`), and the
    /// engines bind the weights their nodes kept. The update waits for these
    /// engines exactly as it waits for ones that were promoted again.
    ///
    /// A start that fails leaves no engine: the update ends all the same,
    /// since nothing it commands can serve the epoch, and the caller closes
    /// the hold with the reason.
    async fn relaunch_previous(
        self: &Arc<Self>,
        control: &Arc<EngineControl>,
        updater: &Arc<crate::updates::Updater>,
        args: Option<proto::config::EngineArgs>,
        stopped: &[u64],
    ) -> anyhow::Result<Arc<EngineClient>> {
        let generation = updater
            .epochs()
            .next_generation(crate::engine::control::ENGINE_GROUP, 0);
        let started = async {
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
            info!(
                generation,
                "the previous epoch started again on the weights it kept"
            );
            anyhow::Ok(client)
        }
        .await;
        // Whether or not it serves, the update has nothing left to wait for
        // of these engines.
        let launched = started
            .as_ref()
            .map(|client| launches(client))
            .unwrap_or_default();
        for (index, previous) in stopped.iter().enumerate() {
            let launch = launched.get(index).copied().unwrap_or(*previous);
            updater.relaunched(*previous, launch);
        }
        started
    }

    /// Restarts the committed epoch while the caller owns its replacement
    /// lock: the engine starts from the arguments this replica keeps, whose
    /// launch file states the epoch's module set.
    pub(crate) async fn recover(self: &Arc<Self>) -> anyhow::Result<Arc<EngineClient>> {
        let control = Arc::clone(
            self.control
                .get()
                .context("the replica has no engine control")?,
        );
        let args = service::lock(&self.served)
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
        self.hold.serve(Arc::clone(&client));
        info!("the relaunched engine serves");
        self.start_standby();
        Ok(client)
    }
}

/// The launch generations of `client`'s world, the scheduler node's first.
fn launches(client: &EngineClient) -> Vec<u64> {
    client
        .processes()
        .into_iter()
        .map(|process| process.engine)
        .collect()
}

/// Whether a replica keeps a standby: the deployment reserved room for one
/// (`worker_config.standby`), the replica's engine serves rather than being
/// parked, draining or failing, and its epoch is the one the deployment
/// serves. A standby anywhere else is a second engine of a scope nothing
/// routes to, or a copy of the device storage the serving engine needs.
fn keeps_standby(wanted: bool, serving: EngineState, epoch_serves: bool) -> bool {
    wanted && epoch_serves && serving == EngineState::Serving
}

/// The standby started beside this replica's serving engine: a parked engine of the serving
/// epoch on every host its worker group spans, started as an epoch update's target is (its launch
/// file states the epoch's module set and the engine it succeeds, so it resolves its modules,
/// captures its graphs and parks by itself), promoted in place together when the serving engine
/// is lost.
pub(crate) struct Standby {
    /// The standby world's scheduler node client, which holds its peers'.
    pub(crate) client: Arc<EngineClient>,
    /// Each host's standby engine and the launch generation it was started under, from the
    /// deployment's one allocator, in the world's order.
    pub(crate) generations: Vec<(String, u64)>,
}

impl Standby {
    /// The scheduler node's standby generation, which names the standby in the log.
    fn generation(&self) -> u64 {
        self.generations
            .first()
            .map_or(0, |(_, generation)| *generation)
    }
}

/// The failure-domain rules' view of the serving engine: one engine process per host of its worker
/// group, which form one domain when there are several (`crate::standby::Standbys`).
fn domain_processes(
    serving: &EngineClient,
    config: &proto::config::EngineConfig,
) -> Vec<crate::standby::MainProcess> {
    let processes = serving.processes();
    let ranks = crate::engine::control::world_hosts(config).unwrap_or_default();
    let spans = processes.len() > 1;
    processes
        .into_iter()
        .map(|process| crate::standby::MainProcess {
            rank: ranks
                .iter()
                .find(|host| host.host == process.host)
                .map_or(0, |host| host.first),
            host: process.host,
            group: crate::engine::control::ENGINE_GROUP.to_owned(),
            generation: process.engine,
            standby_generation: 0,
            epoch: spans.then(|| serving.epoch()),
        })
        .collect()
}

impl Replica {
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

    /// Whether the epoch this replica's engine executes is still the epoch the
    /// deployment serves. A switch moves service to its target through the
    /// frontend, and the replica's own engine is then the drained previous
    /// epoch's: it keeps no standby and is not promoted, because nothing routes
    /// to it and a promoted process would be a second engine of a released epoch.
    fn serves_serving_epoch(&self) -> bool {
        let Some(updater) = self
            .registry
            .upgrade()
            .and_then(|registry| registry.updates.upgrade())
        else {
            return true;
        };
        let Some((epoch, _)) = updater.epochs().serving() else {
            return true;
        };
        let Some(served) = service::lock(&self.served).as_ref().map(|args| args.epoch) else {
            return true;
        };
        epoch == served
    }

    /// The failure-domain rules over `serving`'s engines: the rules this replica keeps, or
    /// new ones when `serving` is another engine than theirs (a promotion or a relaunch replaced
    /// it).
    fn domain_of(
        &self,
        serving: &EngineClient,
    ) -> std::sync::MutexGuard<'_, Option<crate::standby::Standbys>> {
        let config = service::lock(&self.config).engine_config().ok();
        let mut domain = service::lock(&self.domain);
        let current = domain.as_ref().is_some_and(|standbys| {
            let members: Vec<u64> = standbys
                .domains()
                .iter()
                .flatten()
                .map(|process| process.generation)
                .collect();
            let serving: Vec<u64> = serving
                .processes()
                .iter()
                .map(|process| process.engine)
                .collect();
            members == serving
        });
        if !current && let Some(config) = config {
            *domain = Some(crate::standby::Standbys::new(domain_processes(
                serving, &config,
            )));
        }
        domain
    }

    /// Applies the rules' `actions` that end standby processes: the standby world they name is
    /// dropped, and with it every one of its processes.
    fn end_standby_world(&self, actions: &[crate::standby::Action]) {
        let ends = actions.iter().any(|action| {
            matches!(
                action,
                crate::standby::Action::Terminate { .. } | crate::standby::Action::Stop { .. }
            )
        });
        if ends && let Some(standby) = service::lock(&self.standby).take() {
            info!(
                generation = standby.generation(),
                "the failure domain's standby epoch is ended"
            );
        }
    }

    /// Ends the parked standby, if one is live: dropping its client closes the
    /// control connections, which ends its processes and releases the read-only
    /// claims they hold on their epoch's storage.
    pub(crate) fn drop_standby(&self) {
        let retired: Vec<crate::standby::Action> = {
            let mut domain = service::lock(&self.domain);
            let hosts: Vec<String> = domain
                .as_ref()
                .map(|standbys| {
                    standbys
                        .domains()
                        .iter()
                        .flatten()
                        .map(|process| process.host.clone())
                        .collect()
                })
                .unwrap_or_default();
            domain
                .as_mut()
                .map(|standbys| {
                    hosts
                        .iter()
                        .flat_map(|host| standbys.retired(host))
                        .collect()
                })
                .unwrap_or_default()
        };
        self.end_standby_world(&retired);
        let dropped = { service::lock(&self.standby).take() };
        if let Some(standby) = dropped {
            info!(
                generation = standby.generation(),
                "the standby is dropped: its epoch no longer serves"
            );
        }
    }

    /// Replaces the parked standby with one prepared from what the deployment
    /// runs now: a generation update moves the serving engine in place, and a
    /// standby started before it holds the modules the update replaced. A
    /// promoted standby would then serve kernels the epoch's record no longer
    /// names, so the parked processes are dropped and others started.
    pub(crate) fn restart_standby(self: &Arc<Self>) {
        let lost: Vec<crate::standby::Action> = {
            let mut domain = service::lock(&self.domain);
            let host = domain.as_ref().and_then(|standbys| {
                standbys
                    .domains()
                    .first()
                    .and_then(|members| members.first().map(|process| process.host.clone()))
            });
            match (domain.as_mut(), host) {
                (Some(standbys), Some(host)) => standbys.standby_lost(&host),
                _ => Vec::new(),
            }
        };
        self.end_standby_world(&lost);
        // Dropping the parked client closes its control connections, which ends
        // its processes.
        let replaced = { service::lock(&self.standby).take() };
        drop(replaced);
        self.start_standby();
    }

    /// What each host of the serving engine can keep of a parked standby (`StandbyRoom`), as the
    /// Storage Servers' quotas state it now.
    async fn standby_rooms(
        &self,
        control: &Arc<EngineControl>,
    ) -> BTreeMap<String, crate::standby::StandbyRoom> {
        let reading = Arc::clone(control);
        let reports = tokio::task::spawn_blocking(move || reading.storage_reports()).await;
        let deployment = self.served_model_name.clone();
        match reports {
            Ok(Ok(reports)) => crate::standby::rooms(&reports, &deployment),
            Ok(Err(error)) => {
                warn!("the nodes' quotas could not be read: {error:#}");
                BTreeMap::new()
            }
            Err(error) => {
                warn!("reading the nodes' quotas did not complete: {error}");
                BTreeMap::new()
            }
        }
    }

    /// Starts the standby that stands beside this replica's serving engine, if
    /// one is wanted and none is live.
    ///
    /// The failure-domain rules decide (`crate::standby::Standbys::serve`): the serving epoch's
    /// engines are one domain, one per host its worker group spans, and the domain keeps a
    /// standby of every member or of none, all resident or all suspended (the deployment's
    /// `worker_config.standby_storage`, its `auto` decided by what each host's devices can keep).
    /// The standby world is started under numbers from the epoch records' one allocator, claims
    /// each scope's storage read-only (the launch file states the role and the storage mode), and
    /// parks once it is prepared. A replica whose engine is itself parked, draining or failing has
    /// no standby: there is nothing to stand beside.
    ///
    /// # Errors
    ///
    /// Fails when the processes cannot be started or do not report that they
    /// stand by. The caller decides whether to retry.
    pub(crate) async fn keep_standby(self: &Arc<Self>) -> anyhow::Result<()> {
        // `worker_config.standby` decides whether a standby exists at all:
        // each standby rank holds a CUDA context, graph pools and
        // workspaces of its own, and a deployment that has not reserved them
        // cannot hold a second engine of the same scope.
        let (wanted, storage) = {
            let config = service::lock(&self.config);
            (
                config.worker_config.standby,
                config.worker_config.standby_storage,
            )
        };
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
            let mut slot = service::lock(&self.standby);
            match slot.as_ref().map(|standby| standby.client.state()) {
                // A parked standby is the one this replica wants.
                Some(EngineState::Parked) => return Ok(()),
                // A standby that failed or ended is replaced: dropping the
                // client closes its control connections, and its processes end.
                Some(_) => *slot = None,
                None => {}
            }
        }
        let Some(mut args) = service::lock(&self.served).clone() else {
            return Ok(());
        };
        let Some(updater) = self
            .registry
            .upgrade()
            .and_then(|registry| registry.updates.upgrade())
        else {
            return Ok(());
        };
        let epochs = Arc::clone(updater.epochs());
        // The engine control numbered every engine it started, those of other hosts
        // included: the standby's numbers come after them.
        epochs.launched_up_to(
            crate::engine::control::ENGINE_GROUP,
            0,
            control.next_engine().saturating_sub(1),
        );
        let rooms = self.standby_rooms(&control).await;
        let options = crate::standby::StandbyOptions {
            enabled: wanted,
            storage,
        };

        // The rules' round: every member's next number from the one counter, then the spawns. One
        // counter numbers every engine of the deployment, so the members' numbers differ.
        let (spawns, members) = {
            let mut domain = self.domain_of(&serving);
            let Some(standbys) = domain.as_mut() else {
                return Ok(());
            };
            let members = standbys
                .domains()
                .into_iter()
                .flatten()
                .cloned()
                .collect::<Vec<_>>();
            let mut actions = standbys.serve(crate::epoch::Status::Serving, options, &rooms);
            if actions
                .iter()
                .any(|action| matches!(action, crate::standby::Action::Allocate { .. }))
            {
                for action in &actions {
                    if let crate::standby::Action::Allocate { host, group, .. } = action {
                        let generation = epochs.next_generation(group, 0);
                        standbys.allocated(host, generation);
                    }
                }
                actions = standbys.serve(crate::epoch::Status::Serving, options, &rooms);
            }
            let spawns: Vec<(String, u64, crate::standby::Storage)> = actions
                .into_iter()
                .filter_map(|action| match action {
                    crate::standby::Action::Spawn {
                        host,
                        generation,
                        storage,
                        ..
                    } => Some((host, generation, storage)),
                    _ => None,
                })
                .collect();
            (spawns, members)
        };
        if spawns.is_empty() {
            info!(
                model = %self.model_name,
                hosts = ?rooms.keys().collect::<Vec<_>>(),
                "the failure domain keeps no standby: a member's devices cannot keep one"
            );
            return Ok(());
        }
        // The world's processes in its own order, each at the number the rules spawned it at.
        let generations: Vec<(String, u64)> = members
            .iter()
            .filter_map(|member| {
                spawns
                    .iter()
                    .find(|(host, _, _)| *host == member.host)
                    .map(|(host, generation, _)| (host.clone(), *generation))
            })
            .collect();
        let storage = spawns[0].2;

        // The standby executes the serving engine's epoch and parks: its
        // launch names the engine it succeeds, and as an epoch switch's target
        // it opens a session of its own with the buffer pool while it prepares.
        args.predecessor = Some(wire::Predecessor {
            epoch: serving.epoch(),
        });
        args.standby_storage = match storage {
            crate::standby::Storage::Resident => crate::options::StandbyStorage::Resident,
            crate::standby::Storage::Suspended => crate::options::StandbyStorage::Suspended,
        };
        let sink = stats_sink(
            self.model_name.clone(),
            Arc::clone(&self.observer),
            self.ordinal,
        );
        let numbers: Vec<u64> = generations
            .iter()
            .map(|(_, generation)| *generation)
            .collect();
        let started =
            tokio::task::spawn_blocking(move || control.start_world(args, &numbers, sink)).await?;
        let client = match started {
            Ok(client) => Arc::new(client),
            Err(error) => {
                self.lose_standby(&generations);
                return Err(error);
            }
        };
        if let Some(standbys) = service::lock(&self.domain).as_mut() {
            for (host, generation) in &generations {
                standbys.parked(host, *generation);
            }
        }
        info!(
            generations = ?generations,
            storage = ?storage,
            "the standby epoch of the failure domain is parked"
        );
        *service::lock(&self.standby) = Some(Standby {
            client,
            generations,
        });
        Ok(())
    }

    /// The standby of the hosts `generations` names did not start: the domain keeps none, and a
    /// loss relaunches it.
    fn lose_standby(&self, generations: &[(String, u64)]) {
        let actions: Vec<crate::standby::Action> = match service::lock(&self.domain).as_mut() {
            Some(standbys) => generations
                .iter()
                .flat_map(|(host, _)| standbys.standby_lost(host))
                .collect(),
            None => Vec::new(),
        };
        self.end_standby_world(&actions);
    }

    /// Promotes the parked standby in place if this replica has one, and
    /// returns the client that now serves.
    ///
    /// The failure-domain rules decide (`crate::standby::Standbys::failed`): the domain's parked
    /// standbys are promoted together, every promotion issued before any is awaited, or, when a
    /// member has none, the domain is relaunched instead and this returns `None`. The promoted
    /// processes are the same processes, with their graphs at their addresses: each takes the
    /// storage its scope claimed read-only and serves under the serving engine's epoch.
    pub(crate) async fn promote_standby(self: &Arc<Self>) -> Option<Arc<EngineClient>> {
        // A standby being started or prepared is waited for: it is closer to serving than a
        // process this replica has yet to launch.
        let starting = self.standby_starting.lock().await;
        let failure = crate::epoch::Failure {
            kind: "process_exit".to_owned(),
            reason: "the serving engine ended".to_owned(),
            device: None,
        };
        let promotions: Vec<(String, u64)> = {
            let mut domain = service::lock(&self.domain);
            let standbys = domain.as_mut()?;
            let host = standbys
                .domains()
                .first()
                .and_then(|members| members.first().map(|process| process.host.clone()))?;
            standbys
                .failed(&host, &failure)
                .into_iter()
                .filter_map(|action| match action {
                    crate::standby::Action::Promote { host, generation } => {
                        Some((host, generation))
                    }
                    _ => None,
                })
                .collect()
        };
        if promotions.is_empty() {
            // A member has no parked standby: the domain is relaunched.
            let dropped = service::lock(&self.standby).take();
            drop(dropped);
            *service::lock(&self.domain) = None;
            return None;
        }
        let standby = service::lock(&self.standby).take()?;
        self.hold.hold();
        drop(starting);
        // The promotion travels on each engine's process link, as an epoch
        // update's does; no state is handed over, because the engine that
        // held it is gone. Every promotion is issued before any is awaited:
        // the ranks' collectives resume together.
        let updater = self
            .registry
            .upgrade()
            .and_then(|registry| registry.updates.upgrade())?;
        let ranks = service::lock(&self.config)
            .engine_config()
            .ok()
            .and_then(|config| crate::engine::control::world_hosts(&config).ok())
            .unwrap_or_default();
        for (host, generation) in &promotions {
            let rank = ranks
                .iter()
                .find(|ranks| ranks.host == *host)
                .map_or(0, |ranks| ranks.first);
            updater.promote(
                host,
                crate::engine::control::ENGINE_GROUP,
                rank,
                *generation,
            );
        }
        match standby.client.await_serving().await {
            Ok(()) => {
                if let Some(standbys) = service::lock(&self.domain).as_mut() {
                    for (host, generation) in &promotions {
                        standbys.promoted(host, *generation);
                    }
                }
                // The promoted world is the engine that serves; the rules over it start anew.
                *service::lock(&self.domain) = None;
                self.hold.serve(Arc::clone(&standby.client));
                info!(
                    generations = ?standby.generations,
                    "the failure domain's standby epoch was promoted"
                );
                self.start_standby();
                Some(standby.client)
            }
            Err(error) => {
                if let Some(standbys) = service::lock(&self.domain).as_mut() {
                    for (host, _) in &promotions {
                        standbys.promote_failed(host);
                    }
                }
                *service::lock(&self.domain) = None;
                warn!(
                    generation = standby.generation(),
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
    /// Ends the parked standby this replica keeps, if it has one.
    ///
    /// A standby is a process of its own: ending it here means nothing of this
    /// replica's stays on the device after the daemon is gone, and a successor
    /// daemon starts its own instead of adopting a process it never numbered.
    /// Dropping the client closes the control connection, which the standby's
    /// engine treats as the end of the process.
    pub(crate) async fn end_standby(&self) {
        *service::lock(&self.domain) = None;
        let standby = { service::lock(&self.standby).take() };
        if let Some(standby) = standby
            && let Err(error) = tokio::task::spawn_blocking(move || standby.client.shutdown()).await
        {
            warn!("the standby's shutdown did not complete: {error}");
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
        let served = service::lock(&self.served)
            .clone()
            .context("this leader starts no engine of its own")?;
        Ok(EngineHandoff {
            next_engine: control.next_engine(),
            client,
            engine,
            served,
            config: service::lock(&self.config).clone(),
            paused: self.hold().is_paused(),
            asleep: self.asleep.load(Ordering::Acquire),
            sleep_paused: self.sleep_paused.load(Ordering::Acquire),
            unavailable: self
                .registry
                .upgrade()
                .and_then(|registry| registry.updates.upgrade())
                .map(|updater| updater.unavailable_marks())
                .unwrap_or_default(),
        })
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
                EngineState::Starting => ("starting", None),
                EngineState::Parked => ("parked", None),
                EngineState::Serving => ("serving", None),
                EngineState::Draining => ("draining", None),
                EngineState::Stopped => ("stopped", None),
                EngineState::Failed(reason) => ("failed", Some(reason)),
            },
            HoldState::Holding => ("holding", None),
            HoldState::Closed(reason) => ("closed", Some(reason)),
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
            // The leader's status counts the deployment's held requests.
            held: 0,
            generation: hold.client().and_then(|client| client.engine()),
            standby: service::lock(&self.standby)
                .as_ref()
                .map(|standby| {
                    standby
                        .generations
                        .iter()
                        .map(|(_, generation)| *generation)
                        .collect()
                })
                .unwrap_or_default(),
            // The leader's status states the deployment's update.
            update: None,
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
    pub(crate) async fn sleep(&self) -> Result<(), UpdateError> {
        let _replacing = self.replacing.lock().await;
        let hold = self.hold();
        let client = hold.client().ok_or(UpdateError::NoEngine)?;
        if self.asleep.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        if !hold.pause() {
            self.sleep_paused.store(true, Ordering::Release);
        }
        client
            .resize_cache(wire::CacheChange::Sleep)
            .await
            .map_err(|error| {
                UpdateError::Failed(anyhow::anyhow!(
                    "the engine did not sleep, and requests wait until /admin/wake_up: {error}"
                ))
            })?;
        info!("the engine sleeps: its KV cache and graph storage are given back");
        Ok(())
    }

    /// Restores what [`Self::sleep`] gave back, and lets the requests it
    /// held through. Returns what each worker reported.
    ///
    /// # Errors
    ///
    /// Fails when no engine serves, or with the reason the engine did not
    /// wake; requests keep waiting then.
    pub(crate) async fn wake(&self) -> Result<(), UpdateError> {
        let _replacing = self.replacing.lock().await;
        let hold = self.hold();
        let client = hold.client().ok_or(UpdateError::NoEngine)?;
        client
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
        Ok(())
    }

    /// Runs `operation`, `what` for its failure, on the serving engine's client while no engine
    /// replaces it.
    ///
    /// # Errors
    ///
    /// Fails when no engine serves, or with the reason the engine gave.
    pub(crate) async fn operate<T, F, Done>(
        &self,
        what: &str,
        operation: F,
    ) -> Result<T, UpdateError>
    where
        F: FnOnce(Arc<EngineClient>) -> Done,
        Done: std::future::Future<Output = Result<T, String>>,
    {
        let _replacing = self.replacing.lock().await;
        let client = self.hold().client().ok_or(UpdateError::NoEngine)?;
        operation(client)
            .await
            .map_err(|error| UpdateError::Failed(anyhow::anyhow!("{what}: {error}")))
    }

    /// Rewrites the engine's weights in place from the checkpoint at
    /// `model_path`, another checkpoint of the served model, as SGLang's
    /// `update_weights_from_disk` does, or from a plugin's weight source
    /// (`scheme://...`) that streams the served checkpoint's tensors, and
    /// returns what each worker reported. New requests wait; the requests
    /// in flight continue as `in_flight` says (`scheduler::WeightsInFlight`):
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
    ) -> Result<Vec<wire::WeightReport>, UpdateError> {
        let _replacing = self.replacing.lock().await;
        let hold = self.hold();
        let client = hold.client().ok_or(UpdateError::NoEngine)?;
        let mut config = service::lock(&self.config).clone();
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
            let prepared =
                Leader::prepare(&self.families, &config, &Stated::from_paths(Vec::new()))
                    .await
                    .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
            let served = service::lock(&self.served)
                .as_ref()
                .map(|args| args.engine_config.model.clone());
            if served.is_some_and(|served| !served.processes_as(&prepared.engine.model)) {
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
        if let Some(args) = service::lock(&self.served).as_mut() {
            if !streamed {
                args.engine_config.model.model = model_path.to_owned();
                args.engine_config.model.base =
                    base_release(model_path).map_err(UpdateError::Failed)?;
            }
            args.engine_config
                .load
                .weights
                .clone_from(&config.load_config.weights);
        }
        *service::lock(&self.config) = config;
        // A parked standby mapped the weights this update replaced, and a promotion would serve
        // them: it starts again on the new ones.
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
    /// The adapter's directory (`lora_path`), from which a new engine's workers load it again.
    pub(crate) path: String,
}

/// One rank's report of an adapter's slot, as the admin API answers it: the slot, and the tensors
/// a load mapped there.
fn adapter_report(report: &wire::AdapterReport) -> serde_json::Value {
    match report.tensors {
        Some(tensors) => serde_json::json!({ "adapter": report.slot, "tensors": tensors }),
        None => serde_json::json!({ "adapter": report.slot }),
    }
}

/// Loads every adapter of `replica` into `client`'s workers, as an engine
/// that replaces another starts without them. An adapter that does not load
/// leaves the registry, so that requests naming it are refused instead of
/// computing without it.
pub(crate) async fn reload_adapters(replica: &Replica, client: &EngineClient) {
    let adapters: Vec<(String, Adapter)> = service::lock(&replica.adapters)
        .iter()
        .map(|(name, adapter)| (name.clone(), adapter.clone()))
        .collect();
    for (name, adapter) in adapters {
        match client.load_adapter(adapter.slot, &adapter.path).await {
            Ok(_) => info!(
                adapter = name,
                slot = adapter.slot,
                "loaded the adapter again"
            ),
            Err(error) => {
                service::lock(&replica.adapters).remove(&name);
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
/// hold no longer serves, as after an epoch change or a shutdown, is left as
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
            .handoff_sent
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
        // Once an epoch update's target serves, the engine this replica lost
        // is the target's: the update judges its death itself and rolls back
        // to the previous epoch, and a relaunch from here would start the
        // target again from arguments the update is putting back. Before the
        // switch the lost engine is the previous epoch's, which a relaunch
        // restores as any loss does.
        if replica.following.load(Ordering::SeqCst) {
            tracing::info!(%reason, "the engine ended under an epoch update, which owns it");
            return;
        }
        // An epoch the deployment no longer serves is relaunched, not
        // promoted: requests go to another epoch, and a rollback to this one
        // has to find its engine alive.
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

/// Waits until the engine whose states `states` observes is lost, and
/// returns why. A draining or parked engine is no loss, but an epoch the
/// deployment stops serving keeps no standby, and a parked engine is the
/// first moment the watcher learns an update drained it: the standby's
/// read-only claim on that epoch would outlive the release that frees it.
async fn lost(mut states: watch::Receiver<EngineState>, replica: &Replica) -> String {
    loop {
        match states.borrow_and_update().clone() {
            EngineState::Parked | EngineState::Draining => {
                if !replica.serves_serving_epoch() {
                    replica.drop_standby();
                }
            }
            EngineState::Starting | EngineState::Serving => {}
            EngineState::Failed(error) => return format!("the engine failed: {error}"),
            EngineState::Stopped => return "the engine process ended".to_owned(),
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
}

/// The local snapshot of the base release the checkpoint at `model` continues, when it pins one
/// (`crate::api::model::assets::pipeline::pinned_snapshot`): an engine opens the checkpoint and its
/// base as one checkpoint, and reads where the weights already are.
fn base_release(model: &str) -> anyhow::Result<Option<String>> {
    let snapshot =
        crate::api::model::assets::pipeline::pinned_snapshot(std::path::Path::new(model))
            .with_context(|| format!("reading the base release {model} pins"))?;
    Ok(snapshot.map(|path| path.to_string_lossy().into_owned()))
}

/// Resolves the control tokens the engine uses for the loaded model.
///
/// `bos` is `0`: no profile the daemon serves states one, and the engine
/// reads `0` as "no such token". The EOS list
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
    SpecialTokenIds { bos: 0, eos }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use crate::metrics::METRICS;
    use proto::request::SamplingParams;
    use proto::request::{Event, GenerationRequest, Request};

    use super::{EngineState, Leader, keeps_standby};
    use crate::test_support::{
        SERVED_MODEL, STALL, configured_tokenizer, engine_of, fake_client, leader_over,
        letter_fake, post_json, router, send, send_streamed, test_model, text_fake,
    };
    use testkit::{FakeExecutor, Fixture};

    /// A Qwen3 engine over one scheduler, named `engine_model`, over the fake
    /// `shape` makes of the one that samples `a` ([`letter_fake`]).
    ///
    /// The engine's model name labels the process-wide request metrics,
    /// so a test that reads them names its own engine.
    fn qwen3_engine(
        engine_model: &str,
        shape: impl Fn(FakeExecutor) -> FakeExecutor + Send + 'static,
    ) -> Leader {
        engine_of(
            crate::test_support::CHAT,
            testkit::args(engine_model),
            Fixture::new(move |fake| shape(letter_fake(fake))),
        )
    }

    /// The default fake's shape: each request's text ends after
    /// `crate::test_support::TEXT_LEN` tokens.
    fn ending(fake: FakeExecutor) -> FakeExecutor {
        fake.with_text_len(crate::test_support::TEXT_LEN)
    }

    /// A client of an engine of one scheduler, named `engine_model`, over the
    /// default fake.
    fn client_of(engine_model: &str) -> Arc<crate::engine::client::EngineClient> {
        Arc::new(fake_client(
            testkit::args(engine_model),
            Fixture::new(text_fake),
        ))
    }

    /// Greedy chat request of at most `max_completion_tokens` tokens that may
    /// sample only `a`, stopping on each of `stop`, streamed when `stream`.
    fn chat_request(max_completion_tokens: u32, stop: &[&str], stream: bool) -> serde_json::Value {
        serde_json::json!({
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": "prompt"}],
            "max_completion_tokens": max_completion_tokens,
            "temperature": 0.0,
            "allowed_token_ids": [u32::from(b'a')],
            "stop": stop,
            "stream": stream,
        })
    }

    /// Sends `body` through the chat route of `engine`'s router.
    async fn chat(
        router: &axum::Router,
        body: &serde_json::Value,
    ) -> (axum::http::StatusCode, serde_json::Value) {
        let (status, _, document) = send(router, post_json("/v1/chat/completions", body)).await;
        (status, document)
    }

    /// Waits until the rendered process-wide metrics hold `line`.
    async fn metrics_reach(line: &str) {
        let deadline = tokio::time::Instant::now() + STALL;
        while !rendered_metrics_hold(line) {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the metrics never held `{line}`"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
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
            sampling: SamplingParams::default(),
            max_und_tokens: 64,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            generated_token_ids: Vec::new(),
            adapter: 0,
        }
        .into()
    }

    /// The limits an engine reports, which the leader checks requests against,
    /// are the lower of its launch's and what its groups were sized for: a
    /// family that takes images bounds none of their encodings itself, and an
    /// engine whose groups encode no image reports no capacity for one.
    #[tokio::test]
    async fn an_engine_reports_no_image_capacity_its_groups_were_not_sized_for() {
        let mut launch = testkit::args("image-model");
        launch.engine_config.model.generation_limits = proto::request::GenerationLimits {
            max_vit_grid_tokens: u32::MAX,
            max_vision_feature_bytes: u64::MAX,
            encoder_cache_entries: u32::MAX,
            ..Default::default()
        };
        let client =
            tokio::task::spawn_blocking(move || fake_client(launch, Fixture::new(text_fake)))
                .await
                .unwrap();

        let limits = &client.info().limits;
        assert_eq!(limits.max_vit_grid_tokens, 0);
        assert_eq!(limits.max_vision_feature_bytes, 0);
        assert_eq!(limits.encoder_cache_entries, 0);
        tokio::task::spawn_blocking(move || client.shutdown())
            .await
            .unwrap();
    }

    /// An engine that is lost while it serves ends the leader's serving: it
    /// reports itself unhealthy and refuses every later request, naming the
    /// loss.
    #[tokio::test]
    async fn a_lost_engine_fails_later_requests() {
        let client = client_of("lost-engine-model");
        let engine = leader_over(
            test_model(crate::test_support::CHAT),
            configured_tokenizer(),
            "lost-engine-model".into(),
            vec![Arc::clone(&client)],
        );
        assert!(engine.is_healthy());

        // The engine process ends without the leader asking it to.
        tokio::task::spawn_blocking(move || client.shutdown())
            .await
            .unwrap();
        // The leader learns of the loss from the engine's connection, a
        // moment after the process ended.
        let deadline = tokio::time::Instant::now() + STALL;
        while engine.is_healthy() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the leader still reports the lost engine healthy"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

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
        // responses the leader never gave.
        let mut clients = Vec::new();
        for _ in 0..2 {
            clients.push(client_of("sim-model"));
        }
        let engine = leader_over(
            test_model(crate::test_support::CHAT),
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

    /// Awaits `future`, failing by name when it does not respond within
    /// `seconds`: a test whose unbounded wait hangs says which response it never
    /// got instead of outliving the suite's bound in silence.
    async fn within<T>(
        seconds: u64,
        what: &str,
        future: impl std::future::Future<Output = T>,
    ) -> T {
        match tokio::time::timeout(Duration::from_secs(seconds), future).await {
            Ok(value) => value,
            Err(_) => panic!("{what} did not respond within {seconds} s"),
        }
    }

    /// A chat prompt that exceeds the total capacity of a two-unit prefix
    /// cache is submitted and then rejected by engine admission. The
    /// non-streamed response holds its head for the result, so it reports
    /// that rejection as HTTP 400.
    #[tokio::test]
    async fn chat_admission_rejection_remains_a_bad_request() {
        // Unit zero is the padding unit: two allocatable 64-token blocks remain.
        let fixture = Fixture {
            units: 3,
            ..Fixture::new(text_fake)
        };
        let engine = Arc::new(engine_of(
            crate::test_support::CHAT,
            testkit::args(SERVED_MODEL),
            fixture,
        ));
        let router = router(Arc::clone(&engine));
        let request = serde_json::json!({
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": "a".repeat(1024)}],
            "max_completion_tokens": 8
        });

        let (status, document) = chat(&router, &request).await;

        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST, "{document}");
        assert_eq!(document["error"]["type"], "invalid_request_error");
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
            let engine_model = "preprocessing-latency-model";
            let engine = Arc::new(qwen3_engine(engine_model, ending));
            let router = router(Arc::clone(&engine));

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
            let pending = tokio::spawn({
                let router = router.clone();
                async move { chat(&router, &chat_request(8, &["aa"], false)).await }
            });
            tokio::time::sleep(wait).await;
            release_tx.send(()).unwrap();
            occupied.await.unwrap();

            let (status, document) = pending.await.unwrap();
            assert_eq!(status, axum::http::StatusCode::OK, "{document}");
            let labels = format!("model_name=\"{engine_model}\",engine=\"0\"");
            let series = format!("uniserve:e2e_request_latency_seconds_sum{{{labels}}} ");
            let rendered = METRICS.render().unwrap();
            let seconds: f64 = rendered
                .lines()
                .find_map(|line| line.strip_prefix(&series))
                .unwrap_or_else(|| panic!("no `{series}` in:\n{rendered}"))
                .parse()
                .unwrap();
            assert!(seconds >= wait.as_secs_f64(), "{seconds} s");
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
        let engine = Arc::new(qwen3_engine(engine_model, ending));
        let router = router(Arc::clone(&engine));

        let (status, document) = chat(&router, &chat_request(3, &[], false)).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{document}");
        assert_eq!(document["choices"][0]["finish_reason"], "length");
        assert_eq!(document["usage"]["completion_tokens"], 3);
        let prompt_tokens = &document["usage"]["prompt_tokens"];

        let labels = format!("model_name=\"{engine_model}\",engine=\"0\"");
        for expected in [
            format!("uniserve:request_success_total{{{labels},finished_reason=\"length\"}} 1"),
            format!("uniserve:prompt_tokens_total{{{labels}}} {prompt_tokens}"),
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

    /// A request reports, as its usage's `prompt_tokens_details.cached_tokens`, the prompt
    /// tokens its admission found cached: none for the first request of a prompt, and for the
    /// same prompt again the reuse limit, its whole blocks short of its last token, which is
    /// computed (`(prompt_tokens - 1)` rounded down to the block). A stream that asks for usage
    /// reports the same in its usage chunk.
    #[tokio::test]
    async fn a_repeated_prompt_reports_its_cached_tokens() {
        let engine = Arc::new(qwen3_engine("cached-tokens-model", ending));
        let router = router(Arc::clone(&engine));
        let body = |stream: bool| {
            let mut body = serde_json::json!({
                "model": SERVED_MODEL,
                "messages": [{"role": "user", "content": "b".repeat(200)}],
                "max_completion_tokens": 4,
                "temperature": 0.0,
                "allowed_token_ids": [u32::from(b'a')],
                "stream": stream,
            });
            if stream {
                body["stream_options"] = serde_json::json!({"include_usage": true});
            }
            body
        };
        let cached =
            |usage: &serde_json::Value| usage["prompt_tokens_details"]["cached_tokens"].as_u64();

        let (status, first) = chat(&router, &body(false)).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{first}");
        assert_eq!(cached(&first["usage"]), Some(0), "{first}");

        let (status, second) = chat(&router, &body(false)).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{second}");
        let prompt = second["usage"]["prompt_tokens"].as_u64().unwrap();
        let block = u64::from(testkit::BLOCK_TOKENS);
        let reused = (prompt - 1) / block * block;
        assert!(
            reused > 0,
            "a prompt of {prompt} tokens holds a whole block"
        );
        assert_eq!(cached(&second["usage"]), Some(reused), "{second}");

        let mut stream =
            send_streamed(&router, post_json("/v1/chat/completions", &body(true))).await;
        let mut text = String::new();
        while let Some(frame) = tokio::time::timeout(STALL, stream.next_frame())
            .await
            .expect("the stream writes")
        {
            text.push_str(std::str::from_utf8(&frame).unwrap());
        }
        let usage: Vec<serde_json::Value> = text
            .split("\n\n")
            .filter_map(|event| event.strip_prefix("data: "))
            .filter(|payload| *payload != "[DONE]")
            .filter_map(|payload| {
                serde_json::from_str::<serde_json::Value>(payload)
                    .unwrap()
                    .get("usage")
                    .cloned()
            })
            .collect();
        let [usage] = &usage[..] else {
            panic!("one usage chunk: {text}");
        };
        assert_eq!(cached(usage), Some(reused), "{text}");
        engine.shutdown().await.unwrap();
    }

    /// Each request is recorded under the scheduler that served it.
    #[tokio::test]
    async fn request_metrics_name_the_engine_core_that_served_each_request() {
        let engine_model = "core-metrics-model";
        let engine = Arc::new(engine_of(
            crate::test_support::CHAT,
            testkit::launch(engine_model, 2),
            Fixture::new(text_fake),
        ));
        let router = router(Arc::clone(&engine));

        // Both requests are in flight together, so the second is routed
        // while the first is still served.
        let request = chat_request(3, &[], false);
        let ((first, _), (second, _)) =
            futures::join!(chat(&router, &request), chat(&router, &request));
        assert_eq!(first, axum::http::StatusCode::OK);
        assert_eq!(second, axum::http::StatusCode::OK);

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
        let engine = Arc::new(qwen3_engine(engine_model, |fake| fake));
        let router = router(Arc::clone(&engine));

        let mut stream = send_streamed(
            &router,
            post_json("/v1/chat/completions", &chat_request(1_000, &[], true)),
        )
        .await;
        loop {
            let frame = tokio::time::timeout(STALL, stream.next_frame())
                .await
                .expect("the stream writes")
                .expect("the stream goes on");
            if std::str::from_utf8(&frame).is_ok_and(|frame| frame.contains(r#""content":"a"#)) {
                break;
            }
        }
        // The connection is gone: the body's frames have no reader.
        drop(stream);

        let labels = format!("model_name=\"{engine_model}\",engine=\"0\"");
        for expected in [
            format!("uniserve:request_success_total{{{labels},finished_reason=\"abort\"}} 1"),
            format!("uniserve:e2e_request_latency_seconds_count{{{labels}}} 1"),
        ] {
            metrics_reach(&expected).await;
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
    /// with expert parallelism (`moe_config.expert_parallel`) when `experts` holds.
    fn deployment(tensor: usize, data: usize, experts: bool) -> proto::config::EngineConfig {
        let mut options = crate::options::ServerArgs {
            tensor_parallel_size: std::num::NonZeroUsize::new(tensor).unwrap(),
            data_parallel_size: std::num::NonZeroUsize::new(data).unwrap(),
            ..Default::default()
        };
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
                    .flat_map(|group| group.ranks.iter().map(|rank| rank.device.to_string()))
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

    /// Schedulers of several tensor-parallel ranks shard their experts over their own ranks and
    /// exchange no token: with expert parallelism they are independent replicas, each its own
    /// engine on its own ranks launched in the epoch's one number, and each keeps the flag.
    #[test]
    fn tensor_parallel_replicas_sharding_their_experts_are_independent() {
        let config = deployment(2, 2, true);
        let modules = crate::epoch::modules_of(&crate::options::ServerArgs::default());
        let launches = super::epoch_launches(&config, &modules, 7).unwrap();
        let devices: Vec<Vec<String>> = launches
            .iter()
            .map(|launch| {
                launch
                    .engine_config
                    .parallel
                    .worker_groups
                    .iter()
                    .flat_map(|group| group.ranks.iter().map(|rank| rank.device.to_string()))
                    .collect()
            })
            .collect();
        assert_eq!(
            devices,
            vec![vec!["cuda:0", "cuda:1"], vec!["cuda:2", "cuda:3"]]
        );
        assert!(launches.iter().all(|launch| {
            let replica = &launch.engine_config;
            launch.epoch == 7
                && replica.parallel.data_parallel_size.get() == 1
                && replica.moe.expert_parallel
        }));
    }

    /// The exchanges serve expert unions: a deployment that shards its groups' experts over
    /// their tensor-parallel ranks and names another exchange than the default is refused,
    /// while schedulers of one rank each take any exchange.
    #[test]
    fn tensor_parallel_groups_sharding_their_experts_take_no_exchange() {
        for (tensor, data) in [(4, 1), (2, 2)] {
            let mut config = deployment(tensor, data, true);
            config.moe.exchange = "deepep".to_owned();
            let error = super::replica_configs(&config)
                .expect_err("an exchange beside tensor-parallel experts")
                .to_string();
            assert!(error.contains("deepep"), "{error}");
        }

        let mut union = deployment(1, 4, true);
        union.moe.exchange = "deepep".to_owned();
        assert_eq!(super::replica_configs(&union).unwrap().len(), 1);
    }

    /// A standby exists only where the deployment reserved room for one and the
    /// replica's epoch is the epoch the deployment serves. Each of the
    /// three conditions is a whole engine's device storage when it is wrong.
    #[test]
    fn a_standby_exists_only_where_the_deployment_reserved_one_and_the_epoch_serves() {
        assert!(keeps_standby(true, EngineState::Serving, true));
        // Nothing reserved: a second engine would take the serving engine's
        // memory, so the deployment asks for none.
        assert!(!keeps_standby(false, EngineState::Serving, true));
        // A parked, standing-by, failed or closed engine has nothing to stand
        // beside, and a standby of a standby is a third copy.
        assert!(!keeps_standby(true, EngineState::Parked, true));
        assert!(!keeps_standby(true, EngineState::Parked, true));
        assert!(!keeps_standby(
            true,
            EngineState::Failed("a fatal fault".into()),
            true
        ));
        assert!(!keeps_standby(true, EngineState::Stopped, true));
        // An epoch a switch replaced keeps no standby: nothing routes to it, and
        // its read-only claim would outlive the release.
        assert!(!keeps_standby(true, EngineState::Serving, false));
    }
}
