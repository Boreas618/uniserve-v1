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

use anyhow::Context as _;
use futures::StreamExt as _;
use proto::admin::{Phase, ServingState};
use proto::request::Request;
use proto::request::wire;
use tokio::sync::watch;
use tracing::{debug, info, warn};

use crate::api::input::media::MediaConnector;
use crate::api::input::video::{ProcessedVideo, VideoService};
use crate::api::input::{
    AdapterSelection, LoadedBinding, LoadedModel, Processor, ServeError, TokenizeError,
};
use crate::api::model::ModelConfig;
use crate::api::output::RequestOutputStream;
use crate::api::program::Program;
use crate::api::protocol::VideoGenerationRequest;
use crate::config::{Config, Stated};
use crate::engine::client::{EngineClient, EngineReport, EngineState, StatsSink, SubmitError};
use crate::engine::control::EngineControl;
use crate::engine::hold::{HoldState, RequestHold};
use crate::migration::{Migrator, Next, RequestEvents, Target};
use crate::observation::log::StatsLogger;
use crate::observation::record::{self, Recorded, Recorder};
use scheduling::routing::Role;

/// One model served over its engine process.
///
/// Construction spawns Tokio tasks, so every constructor must be called
/// within a Tokio runtime.
pub struct Leader {
    /// The single model name the API serves, which a change of `served_model_name` renames from
    /// the next request on.
    served_model_name: std::sync::Mutex<Arc<str>>,
    /// Independent replica epochs behind the public request entry point.
    pub(crate) replicas: Arc<Replicas>,
    images: MediaConnector,
    /// The updater: the epoch records, the update in flight and the engine
    /// processes it commands, whose states the deployment's status rows sample
    /// (`crate::status`).
    pub(crate) updates: Arc<crate::updates::Updater>,
    /// Whether every request logs a summary line when it ends.
    log_requests: bool,
    /// The deployment's writer of the observation store.
    recorder: Recorder,
    /// The routing program every new request is routed through, while one runs
    /// (`crate::programs`).
    router: Arc<crate::programs::Router>,
    /// The cluster directory the routing program's prefix hits are read from.
    directory: Arc<crate::directory::Directory>,
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
    /// (`DeploymentStatus::held`).
    pub(crate) held: AtomicUsize,
    /// The last profiling session's identity, which every replica's session of it states
    /// (`crate::profiling`).
    profile_sessions: std::sync::atomic::AtomicU64,
    /// The request processor of the model files the serving epoch runs (its tokenizer, chat
    /// template and model facts): a request takes the one that serves when it arrives and keeps
    /// it to its end, and an epoch switch to model files that process requests otherwise
    /// replaces it ([`Replicas::process_with`]).
    processor: std::sync::Mutex<Arc<Processor>>,
}

impl Replicas {
    /// The request processor new requests are processed with.
    pub(crate) fn processor(&self) -> Arc<Processor> {
        Arc::clone(&service::lock(&self.processor))
    }

    /// Processes new requests with `processor` from now on, and returns the one it replaces:
    /// the requests processed with that one keep it to their ends.
    pub(crate) fn process_with(&self, processor: Arc<Processor>) -> Arc<Processor> {
        std::mem::replace(&mut *service::lock(&self.processor), processor)
    }

    /// The identity of the deployment's next profiling session.
    pub(crate) fn next_profile(&self) -> u64 {
        self.profile_sessions.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// The replicas that serve requests now.
    pub(crate) fn serving_now(&self) -> Vec<Arc<Replica>> {
        let replicas = self.serving.borrow().clone();
        replicas
            .into_iter()
            .filter(|replica| replica.hold.is_serving())
            .collect()
    }
}

/// A refused request goes where the deployment's own rule routes it now ([`pair`]): a new
/// request as any new one, a continuation to the least loaded scheduler whose role generates
/// tokens.
impl crate::migration::Reroute for Replicas {
    fn reroute(&self, request: &Request) -> Option<Target> {
        let serving = self.serving_now();
        let candidates = candidates(&serving);
        let route = pair(&loads(&candidates), request)?;
        Some(target(&candidates, route).1)
    }
}

/// One scheduler of a serving replica, as a route chooses among them: the replica, the
/// scheduler's index in its engine and in the deployment, its role, its requests in flight and
/// its latest report's load.
struct Candidate {
    replica: Arc<Replica>,
    scheduler: u32,
    /// The deployment's index of the scheduler, which `replicas.<id>` names it by.
    index: usize,
    role: Role,
    running: u32,
    load: Option<crate::observe::Load>,
    /// How its prefix is looked up in the cluster directory; `None` for an engine run elsewhere
    /// or one that keys no pages.
    placed: Option<crate::directory::Placed>,
}

/// Every scheduler of `serving`'s engines, in the deployment's order of replicas and schedulers.
fn candidates(serving: &[Arc<Replica>]) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    for replica in serving {
        let Some(client) = replica.hold.client() else {
            continue;
        };
        let epoch = client.epoch();
        let running = client.in_flight_by_scheduler();
        // Every scheduler of an engine keys its pages from the engine's seeds, on the engine's
        // node: its scheduler node.
        let placed = {
            let served = service::lock(&replica.served);
            let seed = client.info().prefix_seeds.first().copied();
            served
                .as_ref()
                .zip(seed)
                .map(|(args, seed)| crate::directory::Placed {
                    seed,
                    block: args.engine_config.kv_cache.block_tokens(),
                    host: args.engine_config.parallel.engine_host.clone(),
                })
        };
        for (scheduler, running) in running.into_iter().enumerate() {
            let scheduler = u32::try_from(scheduler).unwrap_or(u32::MAX);
            candidates.push(Candidate {
                index: replica_id(replica.ordinal, scheduler) as usize,
                role: replica.role(scheduler),
                load: replica.observer.load(epoch, scheduler),
                replica: Arc::clone(replica),
                scheduler,
                running,
                placed: placed.clone(),
            });
        }
    }
    candidates
}

/// Each of `candidates`' role and requests in flight, which the deployment's own route weighs
/// ([`pair`]).
fn loads(candidates: &[Candidate]) -> Vec<(Role, u32)> {
    let load = |candidate: &Candidate| (candidate.role, candidate.running);
    candidates.iter().map(load).collect()
}

/// The prefix of the prompt `tokens` of isolation key `isolation` the cluster directory knows
/// each candidate holds, by tier, with the prompt's page keys
/// (`crate::directory::Directory::hits`); a candidate whose `placed` entry is `None` holds none.
/// Blocks on the directory's answer.
fn prefix(
    directory: &crate::directory::Directory,
    (tokens, isolation): (&[u32], Option<u64>),
    placed: &[Option<crate::directory::Placed>],
) -> (Vec<storage::Key>, Vec<Vec<scheduling::routing::Hit>>) {
    let known: Vec<(usize, crate::directory::Placed)> = placed
        .iter()
        .enumerate()
        .filter_map(|(index, placed)| Some((index, placed.clone()?)))
        .collect();
    let replicas: Vec<crate::directory::Placed> =
        known.iter().map(|(_, placed)| placed.clone()).collect();
    let (keys, found) = directory.hits(tokens, isolation, &replicas);
    let mut hits = vec![Vec::new(); placed.len()];
    for ((index, _), found) in known.iter().zip(found) {
        hits[*index] = found;
    }
    (keys, hits)
}

/// The routing program's input for `request` of session `session` over `candidates`: each
/// scheduler's role, requests in flight and latest report's load, and the prefix the cluster
/// directory knows it holds (`prefix`: the prompt's page keys and each candidate's hits).
fn routing(
    request: &Request,
    session: Option<&str>,
    candidates: &[Candidate],
    (keys, hits): (Vec<storage::Key>, Vec<Vec<scheduling::routing::Hit>>),
) -> scheduling::routing::Routing {
    let prompt = match request {
        Request::Generation(generation) => generation.prompt_token_ids.len(),
        Request::Diffusion(_) => 0,
    };
    let replicas = candidates
        .iter()
        .zip(hits)
        .map(|(candidate, hits)| {
            let load = candidate.load.unwrap_or_default();
            let usage = |tier, pages: proto::report::Pages| scheduling::routing::Usage {
                tier,
                used: pages.used,
                capacity: pages.capacity,
            };
            scheduling::routing::Replica {
                id: candidate.index.to_string(),
                role: candidate.role,
                queued: load.queued_tokens,
                remaining: scheduling::time::Duration(load.step_us.saturating_mul(1_000)),
                running: candidate.running,
                usage: vec![
                    usage(scheduling::kv::Tier::Device, load.device),
                    usage(scheduling::kv::Tier::Host, load.host),
                ],
                reservable: load.device.capacity.saturating_sub(load.device.used),
                hits,
            }
        })
        .collect();
    scheduling::routing::Routing {
        request: scheduling::routing::Request {
            prompt: u32::try_from(prompt).unwrap_or(u32::MAX),
            prefix: keys
                .into_iter()
                .map(|key| scheduling::kv::Key(key.0))
                .collect(),
            session: session.map(str::to_owned),
        },
        replicas,
    }
}

/// The deployment's own route of `request` among `candidates`, each scheduler's role and requests
/// in flight, as indices into them: the least loaded scheduler whose role takes the request
/// first, by requests in flight, the first of them on a tie. A continuation
/// (it names a lend or tokens it generated) goes to one that generates tokens; any other token
/// request to one that computes prompts, generated there too when that one is mixed and handed
/// off to the least loaded that generates tokens otherwise; a video request to a mixed one.
/// `None` when no candidate's role takes it.
fn pair(candidates: &[(Role, u32)], request: &Request) -> Option<(usize, usize)> {
    let least = |takes: &dyn Fn(Role) -> bool| {
        (0..candidates.len())
            .filter(|&index| takes(candidates[index].0))
            .min_by_key(|&index| candidates[index].1)
    };
    let generation = match request {
        Request::Generation(generation) => generation,
        Request::Diffusion(_) => {
            let mixed = least(&|role| role == Role::Mixed)?;
            return Some((mixed, mixed));
        }
    };
    if generation.lend.is_some() || !generation.generated_token_ids.is_empty() {
        let decode = least(&Role::decodes)?;
        return Some((decode, decode));
    }
    let prefill = least(&Role::prefills)?;
    if candidates[prefill].0 == Role::Mixed {
        return Some((prefill, prefill));
    }
    let decode = least(&Role::decodes)?;
    Some((prefill, decode))
}

/// The target `route`, indices into `candidates`, names, with the replica that takes the request
/// first.
fn target(candidates: &[Candidate], (prefill, decode): (usize, usize)) -> (Arc<Replica>, Target) {
    let next = |candidate: &Candidate| Next {
        hold: Arc::clone(&candidate.replica.hold),
        scheduler: Some(candidate.scheduler),
    };
    let first = &candidates[prefill];
    let target = Target {
        first: next(first),
        decode: (decode != prefill).then(|| next(&candidates[decode])),
    };
    (Arc::clone(&first.replica), target)
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
        config: Box<Config>,
        modules: crate::epoch::ModuleSet,
    },
    /// The target engines of another epoch serve the deployment from the
    /// first one's `Serving`.
    Epoch(EpochTarget),
}

/// The target of an epoch update, which its follower joins and routes the deployment to.
pub(crate) struct EpochTarget {
    /// What starts and joins the replicas' engines.
    pub(crate) control: Arc<EngineControl>,
    /// Each target replica: the arguments it serves with, once it is no
    /// longer an engine that succeeds another, and its world's engines
    /// by host and launch generation, the scheduler node's first.
    pub(crate) targets: Vec<(proto::config::EngineArgs, Vec<(String, u64)>)>,
    /// The deployment's replicas the update moves, by index ([`crate::updates::entry::Sides`]):
    /// every replica when the target has another number of replicas, which `targets` then all
    /// replace; and otherwise the replicas the change touches, in the order of `targets`, each
    /// replaced by the target of its index, the others serving on in their epochs.
    pub(crate) moved: Vec<usize>,
    /// The configuration the target serves.
    pub(crate) config: Box<Config>,
    /// The epoch the target serves.
    pub(crate) epoch: u64,
    /// Whether the requests in flight at the switch continue with the state they reached
    /// where the target computes otherwise (`in_flight = keep`).
    pub(crate) keep: bool,
    /// The target's model files as the leader processes requests with them, when they process
    /// requests otherwise than the served ones: the deployment's request processor is the
    /// target's from its switch on, and the previous one's again on a rollback.
    pub(crate) processing: Option<Box<Prepared>>,
}

/// The id of scheduler `scheduler` of engine `ordinal` among the deployment's replicas, which
/// `replicas.<id>` names it by: the deployment's index of the scheduler. An engine of several
/// schedulers is the deployment's one engine (an expert union), whose schedulers are numbered as
/// the deployment's; every other engine runs one scheduler, numbered as the engine.
pub(crate) fn replica_id(ordinal: usize, scheduler: u32) -> u32 {
    u32::try_from(ordinal)
        .unwrap_or(u32::MAX)
        .saturating_add(scheduler)
}

/// Records what replica `replica`'s engine reports into the deployment's observation store
/// through `recorder`: each scheduler's statistics, also read by `observer` for each epoch's
/// rates, and its main process's allocators. A scheduler's rows are labelled with its replica's
/// id ([`replica_id`]), which `replicas.<id>` names it by.
pub(crate) fn stats_sink(
    observer: Arc<crate::observe::Observer>,
    recorder: Recorder,
    replica: usize,
) -> StatsSink {
    let replica = replica as u32;
    Box::new(move |report| match report {
        EngineReport::Scheduler {
            epoch,
            scheduler,
            stats,
        } => {
            observer.record(epoch, scheduler, &stats);
            let at = record::Scheduler {
                replica: replica_id(replica as usize, scheduler),
                scheduler,
            };
            recorder.scheduler(at, &stats);
        }
        EngineReport::Memory(devices) => recorder.memory(replica, &devices),
    })
}

/// How many requests ended since the daemon started, and how many failed.
pub(crate) fn request_counts() -> (u64, u64) {
    let counts = &record::OUTCOMES;
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

/// The processing program `modules` fixes (`ModuleSet::processing`), resolved at the Build
/// Server at `socket`, which builds its module first when its store does not hold it, and
/// loaded: the request side of the epoch the set is of.
///
/// # Errors
///
/// Fails when the set fixes no processing program, or the program cannot be resolved or loaded.
pub(crate) async fn load_program(
    socket: &std::path::Path,
    modules: &crate::epoch::ModuleSet,
) -> anyhow::Result<Arc<Program>> {
    let revision = modules
        .processing
        .clone()
        .context("the epoch's module set fixes no processing program")?;
    let socket = socket.to_owned();
    let program =
        tokio::task::spawn_blocking(move || crate::api::program::fetch(&socket, &revision))
            .await
            .context("loading the processing program did not complete")??;
    info!(
        program = %format!("{}@{}", program.revision().backend, program.revision().revision),
        families = program.families().len(),
        "the epoch's processing program is loaded"
    );
    Ok(program)
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
/// Each engine's schedulers serve the roles the deployment's schedulers of their indices serve
/// (`replicas.<id>.role`): an engine of the expert union numbers them as the deployment does,
/// and each independent replica's one scheduler takes the role of its own index.
///
/// Each independent replica's engine host is the host of its first rank: its scheduler node
/// holds its rank 0 and its schedulers ([`crate::engine::control::world_hosts`]), so a replica
/// the deployment places on another host (`device_config.worker_hosts`) runs its engine there.
///
/// # Errors
///
/// Fails when the worker groups do not form equal replicas.
pub(crate) fn replica_configs(
    config: &proto::config::EngineConfig,
) -> anyhow::Result<Vec<proto::config::EngineConfig>> {
    let union = config.expert_union();
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
        .enumerate()
        .map(|(index, groups)| {
            let mut replica = config.clone();
            replica.parallel.data_parallel_size = std::num::NonZeroUsize::MIN;
            replica.parallel.worker_groups = groups.to_vec();
            // The engine's one scheduler is the deployment's scheduler `index`, and serves its
            // role.
            replica.roles = config.roles.get(index).copied().into_iter().collect();
            if let Some(first) = groups.iter().flat_map(|group| group.ranks.iter()).next() {
                replica.parallel.engine_host = first.host.clone();
            }
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
    revision: u64,
) -> anyhow::Result<Vec<proto::config::EngineArgs>> {
    Ok(replica_configs(config)?
        .into_iter()
        .zip(0u32..)
        .map(|(replica, ordinal)| {
            let mut args = proto::config::EngineArgs::new(replica, modules.clone());
            args.epoch = epoch;
            args.revision = revision;
            args.replica = ordinal;
            args
        })
        .collect())
}

/// What a leader prepares before it has an engine (`Leader::prepare`).
pub(crate) struct Prepared {
    pub(crate) model_config: ModelConfig,
    tokenizer: crate::api::tokenizer::DynTokenizer,
    /// The checkpoint's family in the epoch's processing program.
    binding: LoadedBinding,
    /// The deployment's engine configuration, with the facts the model's files determine.
    pub(crate) engine: proto::config::EngineConfig,
}

/// Where a submitted request is served: its replica, and the scheduler of that
/// replica's engine, which a request the engine ended at submission names
/// none of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Placement {
    replica: u32,
    scheduler: Option<u32>,
}

impl Placement {
    /// Where a request served here was measured: its replica and its scheduler.
    fn at(self) -> (u32, Option<u32>) {
        (self.replica, self.scheduler)
    }
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

    /// Where a new request goes ([`Target`]): the routing program's choice among every serving
    /// replica's schedulers while one runs (`crate::programs::Router`), and otherwise, or when
    /// its route is refused, the deployment's own rule ([`pair`]). A deployment whose schedulers
    /// are all mixed and runs no program gives the request to the least loaded replica, whose
    /// engine gives it to its least loaded scheduler.
    ///
    /// A route that names two replicas computes the prompt on the first, which hands the KV off
    /// once the first token is sampled, and generates the tokens on the second
    /// ([`crate::migration::Next`]).
    async fn route(
        &self,
        request: &Request,
        session: Option<&str>,
    ) -> Result<(Arc<Replica>, Target), SubmitError> {
        let least = self.replica_for_request().await?;
        let serving = self.replicas.serving_now();
        let candidates = candidates(&serving);
        let mixed = candidates
            .iter()
            .all(|candidate| candidate.role == Role::Mixed);
        if mixed && !self.router.routes() {
            let target = Target {
                first: Next {
                    hold: Arc::clone(&least.hold),
                    scheduler: None,
                },
                decode: None,
            };
            return Ok((least, target));
        }

        let route = if self.router.routes() {
            // The directory answers over a socket: the question waits off the runtime's threads.
            let directory = Arc::clone(&self.directory);
            let prompt = match request {
                Request::Generation(generation) => (
                    generation.prompt_token_ids.clone(),
                    generation.cache.isolation_key,
                ),
                Request::Diffusion(_) => (Vec::new(), None),
            };
            let placed: Vec<_> = candidates
                .iter()
                .map(|candidate| candidate.placed.clone())
                .collect();
            let hits = tokio::task::spawn_blocking(move || {
                prefix(&directory, (&prompt.0, prompt.1), &placed)
            })
            .await
            .map_err(|error| SubmitError::Unavailable(format!("routing failed: {error}")))?;
            self.router
                .route(&routing(request, session, &candidates, hits))
        } else {
            None
        };
        let route = route
            .map(|route| (route.prefill, route.decode))
            .or_else(|| pair(&loads(&candidates), request));
        Ok(match route {
            Some(route) => target(&candidates, route),
            None => {
                let target = Target {
                    first: Next {
                        hold: Arc::clone(&least.hold),
                        scheduler: None,
                    },
                    decode: None,
                };
                (least, target)
            }
        })
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

    /// The changes of the deployment's serving state from now on ([`StatusChanges`]): a caller
    /// takes them before it reads [`Leader::status`], so that a change after the read is one
    /// they report.
    pub(crate) fn status_changes(&self) -> StatusChanges {
        StatusChanges::of(&self.replicas)
    }

    /// Resolves once no replica's engine has a request in flight: what a deployment's end waits
    /// for before it shuts its engines down.
    pub(crate) async fn drained(&self) {
        for replica in self.replicas() {
            if let Some(client) = replica.hold().client() {
                client.drained().await;
            }
        }
    }

    /// The new requests the daemon holds before a replica takes them: while generation is
    /// paused, the engines sleep, or no engine is ready.
    pub(crate) fn held(&self) -> usize {
        self.replicas.held.load(Ordering::Relaxed)
    }

    /// Whether every replica holds new requests (`paused`, or an engine that sleeps).
    pub(crate) fn paused(&self) -> bool {
        self.replicas()
            .iter()
            .all(|replica| replica.hold().is_paused())
    }

    /// Whether the deployment's storage is released once it comes down (`retire`).
    pub(crate) fn retires(&self) -> bool {
        service::lock(&self.replicas()[0].config).retire
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

    /// Makes the LoRA adapters every replica's workers hold `target`'s (`adapters`, name to
    /// directory): an adapter `target` no longer states, or states with another directory, is
    /// unloaded, requests naming it refused at once; one it states that is not held is loaded
    /// into a free slot of the workers' adapter pool, every rank publishing its tensors at the
    /// Storage Server under the slot, and requests naming it compute with it once every replica
    /// loaded it. A new engine, after an update or a relaunch, loads them again before it serves.
    /// What computes with an adapter is a kernel package, so a request naming one is refused
    /// until one serves (`docs/admin.md`). The caller holds the deployment while no change
    /// replaces its engines.
    ///
    /// # Errors
    ///
    /// Fails when no engine serves, when every adapter slot is taken, or with the reason a
    /// worker gave; an adapter that did not load is not held, and one that did not unload stays
    /// unavailable to requests.
    pub(crate) async fn hold_adapters(
        &self,
        target: &BTreeMap<String, String>,
    ) -> Result<(), UpdateError> {
        let replicas = self.replicas();
        let mut owners = Vec::with_capacity(replicas.len());
        let mut clients = Vec::with_capacity(replicas.len());
        for replica in &replicas {
            owners.push(Arc::clone(&replica.replacing).lock_owned().await);
            clients.push(replica.hold.client().ok_or(UpdateError::NoEngine)?);
        }
        let held: Vec<(String, Adapter)> = service::lock(&self.replicas.adapters)
            .iter()
            .map(|(name, adapter)| (name.clone(), adapter.clone()))
            .collect();
        for (name, adapter) in held {
            if target.get(&name) == Some(&adapter.path) {
                continue;
            }
            // Requests stop naming it before its slot is released.
            service::lock(&self.replicas.adapters).remove(&name);
            let mut failures = Vec::new();
            for client in &clients {
                if let Err(error) = client.unload_adapter(adapter.slot).await {
                    failures.push(error);
                }
            }
            if !failures.is_empty() {
                return Err(UpdateError::Failed(anyhow::anyhow!(
                    "unloading the adapter {name}: {}",
                    failures.join("; ")
                )));
            }
            info!(adapter = name, slot = adapter.slot, "unloaded the adapter");
        }
        for (name, path) in target {
            if service::lock(&self.replicas.adapters).contains_key(name) {
                continue;
            }
            let slot = {
                let adapters = service::lock(&self.replicas.adapters);
                // Slot 0 is the base model's; a request names its adapter's slot.
                (1..=u32::MAX)
                    .find(|slot| adapters.values().all(|adapter| adapter.slot != *slot))
                    .ok_or_else(|| {
                        UpdateError::Refused(
                            "every adapter slot of the workers is taken".to_owned(),
                        )
                    })?
            };
            for client in &clients {
                if let Err(error) = client.load_adapter(slot, path).await {
                    // The name becomes public only after every replica loads: the partial loads
                    // are released before the failure returns.
                    for client in &clients {
                        if let Err(cleanup) = client.unload_adapter(slot).await {
                            warn!(%cleanup, adapter = name, "could not release a partial adapter load");
                        }
                    }
                    return Err(UpdateError::Failed(anyhow::anyhow!(
                        "loading the adapter {name}: {error}"
                    )));
                }
            }
            let identity = format!("{name}#{}", uuid::Uuid::new_v4().simple());
            service::lock(&self.replicas.adapters).insert(
                name.clone(),
                Adapter {
                    slot,
                    identity,
                    path: path.clone(),
                },
            );
            info!(adapter = name, slot, "loaded the adapter");
        }
        drop(owners);
        Ok(())
    }

    /// The settings of the deployment's Storage Servers as the serving configuration states
    /// them (`Config::storage_settings`).
    pub(crate) fn storage_settings(&self) -> storage::Settings {
        service::lock(&self.replicas()[0].config).storage_settings()
    }

    /// The settings of the deployment's Build Server on the leader's node: the `build_config`
    /// of the daemon options the deployment serves with, which every replica's configuration
    /// carries. A package's other settings take effect only through the hot restart its change
    /// is, whose successor serves with them.
    pub(crate) fn build_settings(&self) -> build_server::command::Settings {
        service::lock(&self.replicas()[0].config)
            .build_config
            .clone()
    }

    /// Has the deployment's Storage Server on every node serve with `settings`
    /// (`EngineControl::configure_storage`), and returns once each does: what a change of
    /// `storage_config` or of the Mooncake connector does to the servers, before the engines
    /// follow. A deployment whose engine has not started has no server to change yet: each node
    /// starts its server with the settings the claims state.
    ///
    /// # Errors
    ///
    /// Fails when a node's server cannot take the settings.
    pub(crate) async fn configure_storage(
        &self,
        settings: storage::Settings,
    ) -> anyhow::Result<()> {
        let Some(control) = self.replicas()[0].control.get().cloned() else {
            return Ok(());
        };
        tokio::task::spawn_blocking(move || control.configure_storage(&settings))
            .await
            .context("the change of the Storage Servers' settings did not complete")?
    }

    /// Has the deployment's Storage Server on every node run `program` as the deployment's KV
    /// program from now on, or none (`EngineControl::program`): the leader's node steps its
    /// tiering hook, and every node saves its state in its own `program-state/`.
    ///
    /// # Errors
    ///
    /// Fails when the deployment's engines have not started, so no node's daemon is known yet,
    /// or a node's server refuses the program.
    pub(crate) async fn set_kv_program(
        &self,
        program: Option<crate::programs::KvSetup>,
    ) -> anyhow::Result<()> {
        let Some(control) = self.replicas()[0].control.get().cloned() else {
            anyhow::ensure!(
                program.is_none(),
                "the deployment's engines have not started: no node's Storage Server runs yet"
            );
            return Ok(());
        };
        let started = tokio::task::spawn_blocking(move || control.program(program.as_ref()))
            .await
            .context("setting the KV program did not complete")??;
        for (host, start) in started {
            info!(
                host,
                ?start,
                "the node's Storage Server runs the KV program"
            );
        }
        Ok(())
    }

    /// What the deployment's KV program logged and how its outputs fared on every node since the
    /// last call, each record with its node's host (`EngineControl::program_records`); none
    /// before the deployment's engines started.
    ///
    /// # Errors
    ///
    /// Fails when a node's records cannot be read.
    pub(crate) async fn kv_records(
        &self,
    ) -> anyhow::Result<Vec<(String, storage::protocol::ProgramRecord)>> {
        let Some(control) = self.replicas()[0].control.get().cloned() else {
            return Ok(Vec::new());
        };
        tokio::task::spawn_blocking(move || control.program_records())
            .await
            .context("reading the KV program's records did not complete")?
    }

    /// Loads the configured model and starts the engine process that serves
    /// it, with its schedulers and worker processes.
    ///
    /// `config` must have passed [`Config::validate`]; `stated` holds the
    /// option paths the operator set; `daemon` is this node's daemon's
    /// process socket, through which the engine process and every worker
    /// start, and `instance` the daemon's instance directory, where the engines' request socket
    /// lies. `recorder` writes the deployment's observation store, which the engines' reports,
    /// its requests and its HTTP layer record into. `revision` is the configuration revision the
    /// deployment starts at, whose step values the engines' schedulers begin with, and `lock` the
    /// module revisions its apply fixed when it starts again, `None` to fix them now. The
    /// engine's start blocks until every rank has loaded the model, so it runs on the blocking
    /// pool.
    ///
    /// # Errors
    ///
    /// Fails when the model assets cannot be resolved, when an option of
    /// another model family was stated ([`Stated::refuse_foreign_options`]),
    /// when the options do not resolve into an engine configuration
    /// ([`Config::engine_config`]), when the engine process fails to
    /// start, or when the model cannot be bound to the capabilities its
    /// workers report.
    #[expect(
        clippy::too_many_arguments,
        reason = "the options as validated and as stated, the node's daemon and instance, the \
                  deployment's updater and observation writer, and the revision started with \
                  its module lock"
    )]
    pub(crate) async fn start(
        config: &Config,
        stated: &Stated,
        daemon: String,
        instance: &std::path::Path,
        updates: Arc<crate::updates::Updater>,
        recorder: Recorder,
        revision: u64,
        lock: Option<crate::epoch::ModuleSet>,
    ) -> anyhow::Result<Self> {
        // The deployment's name is the updater's: every node's daemon and servers know it by it.
        let deployment = updates.deployment().to_owned();
        let build_server = crate::engine::control::ensure_build_server(
            &service::RuntimeDirectory::of_user()?,
            &deployment,
            &config.build_config,
        )?;
        let control = Arc::new(EngineControl::new(
            &config.request_socket(instance),
            daemon,
            config.leader_host(),
            deployment,
            build_server.clone(),
        )?);

        // The epoch's whole module set is fixed before any engine starts: the
        // engine, policy and kernel revisions the Build Server's `cover`
        // selects, each module built at every host the epoch runs on. Every
        // launch file states it, so an engine resolves what its profile runs
        // name and serves, and a relaunch and a standby run exactly what their
        // epoch runs. A revision that starts again runs the revisions its
        // apply fixed (`lock`), never what `latest` names by now.
        let local = crate::engine::control::BuildServer {
            host: control.host().to_owned(),
            socket: build_server,
            targets: Vec::new(),
        };
        let hosts = epoch_hosts(&config.engine_config()?)?;
        let named = lock.unwrap_or_else(|| crate::epoch::modules_of(config));
        let modules = fix_modules(local, Some(Arc::clone(&control)), hosts.clone(), named)
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
        // The epoch's request side is the processing program its module set fixed, built from
        // the same tree as its engine; the checkpoint resolves among its families.
        let program = load_program(control.build_server(), &modules).await?;
        let prepared = Self::prepare(&program, config, stated).await?;
        info!(parallel = ?prepared.engine.parallel, "starting the engine process");

        // The deployment's first epoch takes the number of its first engine, and every replica
        // starts in it.
        let epoch = control.next_engine();
        let launches = epoch_launches(&prepared.engine, &modules, epoch, revision)?;
        // Each replica's statistics are kept for the longest span its updates read back.
        let span = crate::observe::span(&config.update_config);
        let clients =
            futures::future::join_all(launches.into_iter().enumerate().map(|(index, args)| {
                let starting = Arc::clone(&control);
                let recorder = recorder.clone();
                tokio::task::spawn_blocking(move || {
                    let observer = Arc::new(crate::observe::Observer::new(span));
                    let sink = stats_sink(Arc::clone(&observer), recorder, index);
                    let client = starting.start(args.clone(), sink)?;
                    Ok::<_, anyhow::Error>((client, args, observer))
                })
            }))
            .await
            .into_iter()
            .map(|started| started.context("the engine startup task did not complete")?)
            .collect::<anyhow::Result<Vec<_>>>()
            .context("failed to start the UniServe engine")?;
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
            recorder,
        )?;
        for (index, (client, served, observer)) in clients.enumerate() {
            let replica = Replica::new(
                config,
                Arc::new(client),
                observer,
                leader.recorder.clone(),
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
        // the status rows report and a later change is compared with.
        let deployment = crate::epoch::Epoch::with_modules(config.clone(), modules)?;
        leader.epochs().install(epoch, &deployment, &[]);
        // The process table names every engine this daemon manages, and the
        // serving one is the first: an update addresses the engine it drains
        // by this launch generation, and its `process` events name it.
        if let Some((engine, interface)) = leader
            .replicas
            .serving
            .borrow()
            .first()
            .and_then(|replica| replica.hold().client())
            .and_then(|client| Some((client.engine()?, client.interface())))
        {
            let mut processes = leader.epochs().processes();
            processes.retain(|process| process.generation != engine);
            processes.push(proto::admin::ManagedProcess {
                group: crate::engine::control::ENGINE_GROUP.to_owned(),
                rank: 0,
                pid: None,
                epoch: Some(epoch),
                build: deployment.engine.executable.clone(),
                generation: engine,
                state: proto::admin::ProcessState::Serving,
                interface,
            });
            leader.epochs().set_processes(processes);
        }
        // A serving epoch's tensors are what the next epoch starts from:
        // until it is published its storage is private to it, and a target
        // of another epoch finds nothing. Each of the epoch's engines claimed
        // its own node's storage before it mapped any, so the nodes that
        // claimed the epoch are the hosts it runs on: a node that holds none
        // of its ranks claimed nothing and is not asked.
        let publishing = Arc::clone(&control);
        tokio::task::spawn_blocking(move || publishing.publish_epochs(&[epoch], &hosts))
            .await
            .context("publishing the first epoch")?
            .context("publishing the first epoch")?;
        info!(epoch, "the engines serve the epoch's module set");
        Ok(leader)
    }

    /// The module set the deployment's first replica serves: what its epoch's apply fixed, and a
    /// committed generation since. `None` before an engine serves.
    pub(crate) fn modules(&self) -> Option<crate::epoch::ModuleSet> {
        let replicas = self.replicas();
        let served = service::lock(&replicas.first()?.served);
        served.as_ref().map(|args| args.modules.clone())
    }

    /// What a leader prepares before it has an engine: the model's files
    /// and the engine configuration of the engine that serves it, as
    /// [`Leader::start`] describes.
    ///
    /// # Errors
    ///
    /// As [`Leader::start`], before the engine starts.
    pub(crate) async fn prepare(
        program: &Arc<Program>,
        config: &Config,
        stated: &Stated,
    ) -> anyhow::Result<Prepared> {
        // The process's one view of the Hugging Face Hub, from the standard environment, through
        // which every repository the deployment states is resolved.
        let hub = crate::api::model::assets::Hub::from_env()?;
        let LoadedModel {
            config: model_config,
            tokenizer,
            binding,
            checkpoint,
        } = ModelConfig::load(
            &hub,
            &config.model_source()?,
            config.adapter_source()?.as_ref(),
            config.max_model_len,
            &config.chat_config,
            &config.mm_processor_kwargs,
            program,
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
        // The engines open the checkpoint where it was resolved: its directory, the release a
        // layered checkpoint continues, and the adapter's directory.
        model.model = utf8(checkpoint.directory)?;
        model.base = checkpoint.base.map(utf8).transpose()?;
        model.lora_path = checkpoint.adapter.map(utf8).transpose()?;
        model.max_model_len = max_model_len;
        model.generation_limits = model_config.generation_limits();
        model.bos = control_tokens.bos;
        model.eos = control_tokens.eos;
        // Video workers prepare exactly the configured sizes, refusing one
        // their model does not generate, and report them back as the only
        // sizes requests may target; the family states the sizes and the
        // longest output a configuration leaves unset. Another family's
        // configuration states no `video_config` (`refuse_foreign_options`).
        if let Some(pipeline) = model_config.video() {
            let video = &mut engine.video;
            video
                .sizes
                .get_or_insert_with(|| pipeline.default_sizes.clone());
            video
                .max_seconds
                .get_or_insert(pipeline.default_max_seconds);
        }

        Ok(Prepared {
            model_config,
            tokenizer,
            binding,
            engine,
        })
    }

    /// The leader of `prepared`'s model over the engine `client` serves,
    /// which `control` started (or adopted) with the arguments `served`, its
    /// statistics read through `observer` and recorded through `recorder`.
    ///
    /// # Errors
    ///
    /// Fails when the model cannot be bound to the capabilities the
    /// engine's workers report.
    #[expect(
        clippy::too_many_arguments,
        reason = "the options, the prepared model, the engine's control, client and served \
                  arguments, and the deployment's observer, updater and recorder"
    )]
    pub(crate) fn finish(
        config: &Config,
        prepared: Prepared,
        control: Arc<EngineControl>,
        client: EngineClient,
        served: proto::config::EngineArgs,
        observer: Arc<crate::observe::Observer>,
        updates: Arc<crate::updates::Updater>,
        recorder: Recorder,
    ) -> anyhow::Result<Self> {
        let processor = Self::processor_of(config, prepared, client.info(), &served)?;
        let engine = Self::new(
            vec![Arc::new(client)],
            processor,
            config,
            observer,
            recorder,
            updates,
        )?;
        let replica = Arc::clone(&engine.replicas.serving.borrow()[0]);
        let _ = replica.control.set(control);
        *service::lock(&replica.served) = Some(served);
        Ok(engine)
    }

    /// The request processor of `prepared`'s model files under `config`, bound to what the
    /// engine `info` describes started with the arguments `served`: the tokenizer, the chat
    /// renderer and the images it lowers, or the video service its denoiser declares, and the
    /// limits its workers report.
    ///
    /// # Errors
    ///
    /// Fails when the model cannot be bound to the capabilities the engine's workers report.
    pub(crate) fn processor_of(
        config: &Config,
        prepared: Prepared,
        info: &crate::engine::client::EngineInfo,
        served: &proto::config::EngineArgs,
    ) -> anyhow::Result<Processor> {
        let Prepared {
            model_config,
            tokenizer,
            binding,
            engine: _,
        } = prepared;

        // A video checkpoint serves what its placed denoiser declares in the worker handshake:
        // its family binds to the description, and its requests are processed through the
        // binding's video service.
        let (binding, video) = match binding {
            LoadedBinding::Chat(binding) => (binding, None),
            LoadedBinding::Video { program, bind } => {
                anyhow::ensure!(
                    !info.video_denoiser.is_empty(),
                    "the video worker reported no video denoiser"
                );
                let video = scheduling::processing::video::Binding {
                    description: info.video_denoiser.clone(),
                    max_seconds: served
                        .engine_config
                        .video
                        .max_seconds
                        .context("the leader states a video deployment's longest output")?,
                };
                let bind = scheduling::processing::Bind {
                    video: Some(video),
                    ..*bind
                };
                let binding = program
                    .bind(bind, Arc::clone(&tokenizer))
                    .context("binding the video processor")?;
                let service = VideoService::new(Arc::clone(&binding), &config.media_config)
                    .context("failed to bind the video service")?;
                (binding, Some(service))
            }
        };

        let processor =
            Processor::new(model_config, tokenizer, binding, video, info.limits.clone())
                .context("failed to bind the configured model description")?;
        Ok(processor)
    }

    /// Serves `processor`'s model over the independent replicas in `clients`, each replica's
    /// statistics read through `observer`, under the names `options` states (its
    /// `served_model_name` for the API, its model for the metrics' labels), resolving every
    /// request's image references as its `image_fetch_config` states, and logging and migrating
    /// requests as its `logging_config` and `migration_config` state.
    ///
    /// # Errors
    ///
    /// Fails when the image fetcher cannot be built.
    pub(crate) fn new(
        clients: Vec<Arc<EngineClient>>,
        processor: Processor,
        options: &Config,
        observer: Arc<crate::observe::Observer>,
        recorder: Recorder,
        updates: Arc<crate::updates::Updater>,
    ) -> anyhow::Result<Self> {
        let served_model_name = options.served_model_name().to_owned();
        let images = MediaConnector::new(&options.image_fetch_config)
            .context("failed to initialize the image URL fetcher")?;
        let logging = &options.logging_config;
        let stats_logger = logging
            .log_stats
            .then(|| StatsLogger::start(recorder.clone(), logging.stats_interval));
        // Each replica's options name the checkpoint as the package states it, `model` or
        // `model_dir` alone: a hot restart hands them to a successor, which validates them as a
        // start validates a package.
        let replica_options = Config {
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
                            recorder.clone(),
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
            profile_sessions: std::sync::atomic::AtomicU64::new(0),
            processor: std::sync::Mutex::new(Arc::new(processor)),
        });
        for replica in replicas.serving.borrow().iter() {
            replica.monitor();
        }
        let directory = Arc::new(crate::directory::Directory::new(updates.deployment()));
        Ok(Self {
            served_model_name: std::sync::Mutex::new(Arc::from(served_model_name)),
            replicas,
            images,
            updates,
            log_requests: logging.log_requests,
            recorder,
            router: Arc::default(),
            directory,
            _stats_logger: stats_logger,
        })
    }

    /// The slot of the routing program every new request is routed through.
    pub(crate) fn router(&self) -> &Arc<crate::programs::Router> {
        &self.router
    }

    /// The request processor of the model files the serving epoch runs: new requests are
    /// processed with it, and a request keeps the one it arrived with to its end.
    pub fn processor(&self) -> Arc<Processor> {
        self.replicas.processor()
    }

    /// The deployment's writer of the observation store.
    pub(crate) fn recorder(&self) -> &Recorder {
        &self.recorder
    }

    /// The single model name the API serves.
    pub fn served_model_name(&self) -> Arc<str> {
        Arc::clone(&service::lock(&self.served_model_name))
    }

    /// Serves the API under `name` from the next request on: a request that names the former
    /// name is refused, and the requests in flight under it end as they would have.
    pub(crate) fn rename(&self, name: &str) {
        let former = std::mem::replace(&mut *service::lock(&self.served_model_name), name.into());
        info!(%former, name, "the deployment serves under its new name");
    }

    /// The tasks this engine serves: what the served family declares it serves.
    pub fn supported_tasks(&self) -> Vec<Task> {
        match self.processor().config().family.serving {
            scheduling::processing::Serving::Chat { .. } => vec![Task::Generate],
            scheduling::processing::Serving::Video(_) => vec![Task::VideoGeneration],
        }
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
        trace: u64,
    ) -> Result<
        (
            Box<dyn crate::api::events::Events>,
            crate::api::output::OutputProcessor,
        ),
        crate::api::openai::ApiError,
    > {
        let started = Instant::now();
        let session = request.prompt_cache_key.clone();
        // The request is processed, and its output decoded, by the processor that serves when it
        // arrives, whatever replaces it meanwhile.
        let processor = self.processor();
        let image_urls = if processor.supports_image_input() {
            request.image_urls()
        } else {
            Vec::new()
        };
        let fetching = !image_urls.is_empty();
        let fetch = fetching.then(|| service::span::enter(service::span::Stage::Fetch, trace));
        let images = self
            .images
            .fetch_images(image_urls)
            .await
            .map_err(|error| ServeError::ImageInput {
                request_id: request_id.clone(),
                index: error.index,
                source: error.source,
            })?;
        drop(fetch);
        let (request, lowering, parser) = preprocess(&request_id, {
            let request_id = request_id.clone();
            let processor = Arc::clone(&processor);
            move || processor.process(&request_id, &request, adapter.as_ref(), images, trace)
        })
        .await?;

        let compile_us = elapsed_us(started);
        let (events, placement) = self
            .submit(request.into(), session.as_deref())
            .await
            .map_err(ServeError::from)?;
        let log = self.log_requests.then_some(request_id);
        let stats = record::Request::new(self.recorder.clone(), placement.at(), started, log)
            .session(session)
            .traced(trace);
        // Dropping the events on a failure cancels the request in its engine.
        let output = crate::api::output::OutputProcessor::new(
            processor.tokenizer(),
            lowering,
            parser,
            compile_us,
            Box::new(stats),
        )?
        .traced(trace);

        Ok((Box::new(events), output))
    }

    /// The statistics recorder of a request this process takes over, under
    /// the replica that serves it and the instant it entered the deployment.
    pub(crate) fn stat_logger(
        &self,
        request_id: &str,
        replica: u32,
        started: std::time::Instant,
    ) -> record::Request {
        // The process that handed the request over named its replica alone.
        let log = self.log_requests.then(|| request_id.to_owned());
        record::Request::new(self.recorder.clone(), (replica, None), started, log)
    }

    /// Takes one request's engine event stream over from the process that
    /// consumed it, going on after the position it reached:
    /// the request keeps generating in the engine that serves it,
    /// and is neither cancelled nor submitted again. Both kinds of response
    /// in a handoff call it, with the replica and identity their state carries
    /// and where its stop-string acknowledgements stand
    /// (`EngineClient::take_over`).
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
        acknowledgement: Option<&proto::handoff::Acknowledgement>,
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
        let stream = client.take_over(id, consumed, acknowledgement)?;
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
        trace: u64,
    ) -> Result<(ProcessedVideo, RequestOutputStream), ServeError> {
        let started = Instant::now();
        let processor = self.processor();
        let video = processor
            .video_service()
            .ok_or_else(|| ServeError::UnsupportedFeature {
                request_id: request_id.clone(),
                feature: "video_generation",
            })?;
        let (request, processed) = video
            .process(
                &request_id,
                &request,
                processor.config().max_model_len,
                trace,
            )
            .await?;
        let prompt_tokens = request.prompt_token_ids.len();
        let (events, placement) = self
            .submit(Request::Diffusion(Box::new(request)), None)
            .await?;
        let output = crate::api::output::media_output(
            request_id.clone(),
            started,
            elapsed_us(started),
            prompt_tokens,
            Box::new(events),
        );
        Ok((
            processed,
            self.stats(request_id, started, placement, output, trace),
        ))
    }

    /// Records `output`'s request metrics under where it is served, as the request of trace id
    /// `trace`.
    fn stats(
        &self,
        request_id: String,
        started: Instant,
        placement: Placement,
        output: RequestOutputStream,
        trace: u64,
    ) -> RequestOutputStream {
        let log = self.log_requests.then_some(request_id);
        let request =
            record::Request::new(self.recorder.clone(), placement.at(), started, log).traced(trace);
        Box::pin(Recorded::new(output, request))
    }

    /// Submits one tokenized request of session `session` (the caller's `prompt_cache_key`, the
    /// routing program's session hint) to the engine, which routes it to its
    /// least loaded live scheduler; while no engine is ready, the request
    /// waits in the hold until one is.
    ///
    /// Returns the request's event stream and where it is served. Dropping
    /// the stream before its terminal event cancels the request.
    ///
    /// # Errors
    ///
    /// Fails when the engine serves no requests.
    pub(crate) async fn submit(
        &self,
        request: Request,
        session: Option<&str>,
    ) -> Result<(RequestEvents, Placement), SubmitError> {
        use service::span::{Stage, enter};

        // The route the routing program chose, then the hand-over until the engine accepted.
        let trace = request.trace();
        let route = (trace != 0).then(|| enter(Stage::Route, trace));
        let (replica, target) = self.route(&request, session).await?;
        drop(route);
        let submit = (trace != 0).then(|| enter(Stage::Submit, trace));
        let (events, scheduler) = replica
            .migrator
            .submit(request, target.first.scheduler, target.decode)
            .await?;
        drop(submit);
        // Its rows are the scheduler's replica's, by the id `replicas.<id>` names it by.
        let placement = Placement {
            replica: scheduler.map_or(replica.ordinal as u32, |scheduler| {
                replica_id(replica.ordinal, scheduler)
            }),
            scheduler,
        };
        Ok((events, placement))
    }
}

/// A scheduler of a serving replica, numbered across the deployment: scheduler `s` of replica
/// `r` is the deployment's scheduler `r + s` (an engine of several schedulers is the
/// deployment's one replica), the number `replicas.<id>` and `kv_events_config` name it by.
pub(crate) struct Numbered {
    pub(crate) number: u32,
    pub(crate) replica: Arc<Replica>,
    pub(crate) scheduler: u32,
    /// The configuration its engine serves with: its arguments', or the one its options
    /// resolve to for an engine the daemon started with none.
    pub(crate) engine: Option<proto::config::EngineConfig>,
    /// What its engine reported it serves.
    pub(crate) info: Option<crate::engine::client::EngineInfo>,
}

impl Leader {
    /// Every scheduler of the replicas that serve, in the order of their numbers.
    pub(crate) fn numbered(&self) -> Vec<Numbered> {
        let mut numbered = Vec::new();
        for replica in self.replicas() {
            let Some(client) = replica.hold.client().filter(|_| replica.hold.is_serving()) else {
                continue;
            };
            let served = service::lock(&replica.served)
                .as_ref()
                .map(|args| args.engine_config.clone());
            let engine = served.or_else(|| service::lock(&replica.config).engine_config().ok());
            let info = client.info().clone();
            for scheduler in 0..info.schedulers.max(1) {
                numbered.push(Numbered {
                    number: u32::try_from(replica.ordinal).unwrap_or(u32::MAX) + scheduler,
                    replica: Arc::clone(&replica),
                    scheduler,
                    engine: engine.clone(),
                    info: Some(info.clone()),
                });
            }
        }
        numbered.sort_by_key(|numbered| numbered.number);
        numbered
    }

    /// Submits `request` routed as its submitter chose, without the routing program: its prompt
    /// to the deployment's scheduler `prefill` and its tokens to scheduler `decode`, as a
    /// routing program's route names them. The submitter checked the replicas' roles. Two
    /// schedulers hand the request off as a route of the routing program that names two
    /// replicas does: the prefill's computes the prompt and lends its KV, and the decode's
    /// continues the request from it.
    ///
    /// # Errors
    ///
    /// Fails when no serving replica is scheduler `prefill`, or as [`Leader::submit`] fails.
    pub(crate) async fn submit_routed(
        &self,
        request: Request,
        prefill: u32,
        decode: u32,
    ) -> Result<RequestEvents, SubmitError> {
        let serving = self.numbered();
        let found = |number: u32| serving.iter().find(|numbered| numbered.number == number);
        let (Some(at), Some(to)) = (found(prefill), found(decode)) else {
            return Err(SubmitError::Unavailable(format!(
                "no serving replica is scheduler {prefill} or {decode}"
            )));
        };
        // Two replicas hand the request off once its prompt is computed: the decode replica
        // continues it from the KV the prefill replica lends.
        let handoff = (prefill != decode).then(|| Next {
            hold: Arc::clone(&to.replica.hold),
            scheduler: Some(to.scheduler),
        });
        let (events, _) = at
            .replica
            .migrator
            .submit(request, Some(at.scheduler), handoff)
            .await?;
        Ok(events)
    }
}

impl Leader {
    /// The spans every process of the deployment keeps in its rings that ended within
    /// `profiling_config.span_window`, as `span` events (`crate::observation::spans`): each
    /// replica's engine's, read through its request interface, and every node's daemon's and
    /// Storage Server's, read through the nodes' daemons. A process that cannot be read is
    /// logged, and the others' spans are read.
    pub(crate) async fn spans(&self) -> Vec<crate::observation::Event> {
        use crate::observation::spans::{Place, event, since};

        let replicas = self.replicas();
        let Some(first) = replicas.first() else {
            return Vec::new();
        };
        let window = service::lock(&first.config).profiling_config.span_window;
        let revision = self.recorder.revision();
        // Every process leaves out what ended before the window, so no read carries it.
        let since = since(std::time::SystemTime::now(), window);

        let mut events = Vec::new();
        for replica in &replicas {
            let Some(client) = replica.hold().client() else {
                continue;
            };
            match client.read_spans(since).await {
                Ok((epoch, spans)) => {
                    let place = Place {
                        replica: Some(replica.ordinal),
                        epoch: Some(epoch),
                        host: None,
                    };
                    events.extend(spans.iter().map(|span| event(span, place, revision)));
                }
                Err(error) => warn!(
                    replica = replica.ordinal,
                    "the spans of the replica's engine were not read: {error}"
                ),
            }
        }

        let Some(control) = first.control.get().cloned() else {
            return events;
        };
        match tokio::task::spawn_blocking(move || control.node_spans(since)).await {
            Ok(Ok(nodes)) => {
                for (host, spans) in &nodes {
                    let place = Place {
                        host: Some(host),
                        ..Place::default()
                    };
                    events.extend(spans.iter().map(|span| event(span, place, revision)));
                }
            }
            Ok(Err(error)) => warn!("the nodes' spans were not read: {error:#}"),
            Err(error) => warn!("reading the nodes' spans did not complete: {error}"),
        }
        events
    }
}

/// The spans of a serving deployment, which its leader reads for the observation store's
/// queries (`crate::observation::spans::Source`); none once the leader is gone.
pub(crate) struct DeploymentSpans(pub(crate) std::sync::Weak<Leader>);

impl crate::observation::spans::Source for DeploymentSpans {
    fn read(&self) -> futures::future::BoxFuture<'static, Vec<crate::observation::Event>> {
        let leader = self.0.upgrade();
        Box::pin(async move {
            match leader {
                Some(leader) => leader.spans().await,
                None => Vec::new(),
            }
        })
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
    pub(crate) config: Config,
    /// New requests wait (`Leader::pause`).
    pub(crate) paused: bool,
    /// The engine sleeps, and whether its sleep paused new requests.
    pub(crate) asleep: bool,
    pub(crate) sleep_paused: bool,
    /// The configurations a rollback marked unavailable.
    pub(crate) unavailable: Vec<proto::admin::Unavailable>,
}

/// What one replica serves, of which the deployment's status ([`DeploymentStatus`]) is made.
pub(crate) struct ReplicaStatus {
    /// The lifecycle state of the engine the replica's requests go to.
    pub(crate) engine: ServingState,
    /// Why the engine failed or no engine will serve.
    pub(crate) reason: Option<String>,
    /// The epoch the replica executes, or last executed while no engine serves it.
    pub(crate) epoch: u64,
    /// The requests the serving engine has in flight.
    pub(crate) in_flight: usize,
    /// The launch generation of the engine that serves (the scheduler node's, for a world
    /// over several hosts); `None` while none serves, or for an engine this daemon did not
    /// start.
    pub(crate) generation: Option<u64>,
    /// The launch generation of each member of the parked standby beside the serving engine, in
    /// the world's order: the serving generation a promotion leaves is its first.
    pub(crate) standby: Vec<u64>,
}

/// The changes of a deployment's serving state ([`Leader::status_changes`]): its set of
/// replicas, and each replica's hold (the engine that serves it, its holding, closing or pause)
/// and that engine's lifecycle state. What [`Leader::status`] reads changes only through them.
pub(crate) struct StatusChanges {
    replicas: watch::Receiver<Vec<Arc<Replica>>>,
    holds: Vec<crate::engine::hold::HoldChanges>,
    engines: Vec<watch::Receiver<EngineState>>,
}

impl StatusChanges {
    /// The changes of `replicas` from now on.
    fn of(replicas: &Replicas) -> Self {
        let mut changes = Self {
            replicas: replicas.serving.subscribe(),
            holds: Vec::new(),
            engines: Vec::new(),
        };
        changes.follow();
        changes
    }

    /// Follows the replicas that serve now: their holds and the engines those serve with.
    fn follow(&mut self) {
        let current = self.replicas.borrow_and_update().clone();
        self.holds = current
            .iter()
            .map(|replica| replica.hold().changes())
            .collect();
        self.engines = current
            .iter()
            .filter_map(|replica| replica.hold().client())
            .map(|client| client.states())
            .collect();
    }

    /// Resolves at the deployment's next change of serving state, and follows the replicas and
    /// engines that serve after it.
    pub(crate) async fn changed(&mut self) {
        {
            let Self {
                replicas,
                holds,
                engines,
            } = self;
            let mut waits: Vec<
                std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>>,
            > = vec![Box::pin(async move {
                // A deployment whose replica set is gone changes no more.
                if replicas.changed().await.is_err() {
                    std::future::pending::<()>().await;
                }
            })];
            for hold in holds.iter_mut() {
                waits.push(Box::pin(hold.changed()));
            }
            for engine in engines.iter_mut() {
                waits.push(Box::pin(async move {
                    if engine.changed().await.is_err() {
                        std::future::pending::<()>().await;
                    }
                }));
            }
            futures::future::select_all(waits).await;
        }
        self.follow();
    }
}

/// The deployment `config` describes, refused as a request when it does not resolve: its
/// deployment file does not read or does not place.
pub(crate) fn deployment(config: &Config) -> Result<crate::epoch::Epoch, UpdateError> {
    crate::epoch::Epoch::of(config.clone()).map_err(|error| {
        UpdateError::Refused(format!("the configuration does not resolve: {error:#}"))
    })
}

/// Why a change of the deployment did not happen or did not complete.
#[derive(Debug, thiserror::Error)]
pub(crate) enum UpdateError {
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

/// Whether the daemon processes requests for the checkpoint in the directory `path` as it does
/// for the served checkpoint in the directory `served`, whose family of the epoch's processing
/// `program` both must declare: the files its family reads to process a request, each the same
/// in both, byte for byte, and published by both or by neither.
///
/// What a family reads is declared by the program, not probed: a chat family's request
/// processing reads the checkpoint's tokenizer and the files its binding reads (its tokenizer
/// settings, generation defaults and chat template where the family keeps them,
/// [`crate::api::model::assets::ChatFiles`]); a video family's reads its pipeline index and
/// every file of the pipeline parts it declares (`Pipeline::processing_parts`). The model
/// facts (`config.json`) are also compared by the engine configuration's own comparison
/// (`proto::config::ModelConfig::processes_as`).
///
/// # Errors
///
/// The reason the checkpoints differ, or the reason a checkpoint's files cannot be resolved or
/// read.
pub(crate) async fn same_processing(
    program: &Program,
    served: &str,
    path: &str,
) -> Result<(), String> {
    let declared = |directory: &str| {
        crate::api::model::assets::declared_family(std::path::Path::new(directory), program)
            .map_err(|error| format!("resolving the family of {directory}: {error}"))
    };
    let family = declared(served)?;
    let other = declared(path)?;
    if other.id != family.id {
        return Err(format!(
            "the checkpoint at {path} is a {} model, not a {} model as the served one: deploy it \
             instead",
            other.id, family.id
        ));
    }
    match &family.serving {
        scheduling::processing::Serving::Video(video) => same_pipeline(
            family.root,
            &video.processing_parts,
            std::path::Path::new(served),
            std::path::Path::new(path),
        ),
        scheduling::processing::Serving::Chat { files } => same_model_files(files, served, path),
    }
}

/// [`same_processing`] for a chat checkpoint: the files of it request processing reads, as its
/// family declares them (`declared`), and its tokenizer.
fn same_model_files(
    declared: &[scheduling::processing::CheckpointFile],
    served: &str,
    path: &str,
) -> Result<(), String> {
    use crate::api::model::assets::ChatFiles;

    let locate = |directory: &str| {
        ChatFiles::locate(std::path::Path::new(directory), declared)
            .map_err(|error| format!("resolving the files of {directory}: {error}"))
    };
    let (old, new) = (locate(served)?, locate(path)?);
    let listed = |files: &ChatFiles| -> BTreeMap<String, std::path::PathBuf> {
        files.processing_files().into_iter().collect()
    };
    let (old, new) = (listed(&old), listed(&new));
    let names: std::collections::BTreeSet<&String> = old.keys().chain(new.keys()).collect();
    for name in names {
        if read_published(old.get(name).map(std::path::PathBuf::as_path))?
            != read_published(new.get(name).map(std::path::PathBuf::as_path))?
        {
            return Err(format!(
                "the checkpoint at {path} has another {name} than the served model, which \
                 changes how requests are processed: deploy it instead"
            ));
        }
    }
    Ok(())
}

/// The contents of `file`, `None` when the checkpoint publishes none.
fn read_published(file: Option<&std::path::Path>) -> Result<Option<Vec<u8>>, String> {
    file.map(|file| std::fs::read(file).map_err(|error| format!("{}: {error}", file.display())))
        .transpose()
}

/// The files of the directory `directory` by name, `None` when the checkpoint has no such
/// directory.
fn part_files(
    directory: &std::path::Path,
) -> Result<Option<BTreeMap<std::ffi::OsString, std::path::PathBuf>>, String> {
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("{}: {error}", directory.display())),
    };
    let mut files = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("{}: {error}", directory.display()))?;
        files.insert(entry.file_name(), entry.path());
    }
    Ok(Some(files))
}

/// [`same_processing`] for a diffusers checkpoint: the same pipeline index, and in every part of
/// `parts` the same files, compared both ways.
fn same_pipeline(
    index: scheduling::processing::RootFile,
    parts: &[String],
    served: &std::path::Path,
    path: &std::path::Path,
) -> Result<(), String> {
    // A part is compared by the files its directory holds, which a checkpoint states only where
    // it lies: one named by its repository is resolved file by file, never listed.
    for root in [served, path] {
        if !root.is_dir() {
            return Err(format!(
                "{} is no checkpoint directory: a pipeline's parts are compared where they lie",
                root.display()
            ));
        }
    }
    let published = |root: &std::path::Path| {
        let file = root.join(index.name());
        read_published(file.is_file().then_some(file.as_path()))
    };
    if published(served)? != published(path)? {
        return Err(format!(
            "the checkpoint at {} states other parts than the served model: deploy it instead",
            path.display()
        ));
    }
    for part in parts {
        let (old, new) = (
            part_files(&served.join(part))?,
            part_files(&path.join(part))?,
        );
        let differs = match (old, new) {
            (None, None) => None,
            (Some(old), Some(new)) => {
                let names: std::collections::BTreeSet<_> = old.keys().chain(new.keys()).collect();
                let mut differs = None;
                for name in names {
                    let old = read_published(old.get(name).map(std::path::PathBuf::as_path))?;
                    let new = read_published(new.get(name).map(std::path::PathBuf::as_path))?;
                    if old != new {
                        differs = Some(format!("{part}/{}", name.to_string_lossy()));
                        break;
                    }
                }
                differs
            }
            _ => Some((*part).to_owned()),
        };
        if let Some(file) = differs {
            return Err(format!(
                "the checkpoint at {} has another {file} than the served model, which changes \
                 how requests are processed: deploy it instead",
                path.display()
            ));
        }
    }
    Ok(())
}

/// One independent replica: the engine its requests go to, the arguments
/// that engine serves with, and its standby.
pub(crate) struct Replica {
    pub(crate) registry: std::sync::Weak<Replicas>,
    pub(crate) ordinal: usize,
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
    pub(crate) config: std::sync::Mutex<Config>,
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
    /// The deployment's writer of the observation store, which the statistics sinks of the
    /// engines this replica starts record through.
    pub(crate) recorder: Recorder,
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
    /// The role of the engine's scheduler `scheduler`, as the arguments the engine serves with
    /// state it (`replicas.<id>.role`); mixed for an engine run elsewhere.
    pub(crate) fn role(&self, scheduler: u32) -> Role {
        let served = service::lock(&self.served);
        let roles = served.as_ref().map(|args| &args.engine_config.roles);
        roles
            .and_then(|roles| roles.get(scheduler as usize).copied())
            .unwrap_or(Role::Mixed)
    }

    pub(crate) fn new(
        options: &Config,
        client: Arc<EngineClient>,
        observer: Arc<crate::observe::Observer>,
        recorder: Recorder,
        ordinal: usize,
        registry: std::sync::Weak<Replicas>,
        adapters: Arc<std::sync::Mutex<BTreeMap<String, Adapter>>>,
    ) -> Arc<Self> {
        let hold = Arc::new(RequestHold::new());
        let reroute: std::sync::Weak<Replicas> = std::sync::Weak::clone(&registry);
        let replica = Arc::new(Replica {
            registry,
            ordinal,
            migrator: Migrator::new(
                Arc::clone(&hold),
                options.migration_config.max_migrations,
                Some(reroute as std::sync::Weak<dyn crate::migration::Reroute>),
            ),
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
            recorder,
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
    /// and commits the update once its trial lasted `options.trial`. For the generation form
    /// it keeps the committed configuration and module set, and replaces the
    /// standby with one of them. For the epoch form, which the first replica
    /// follows for the whole deployment, it joins every target replica's
    /// engines, routes new requests to the target from its first `Serving`,
    /// and back to the previous epoch from its first `Serving` again when the
    /// update rolls back.
    ///
    /// The task wakes on the update's own progress
    /// ([`crate::updates::Updater::watch`]), at the trial's end and, during the
    /// trial, on the interval the thresholds are read at.
    pub(crate) fn follow(
        self: &Arc<Self>,
        updater: Arc<crate::updates::Updater>,
        number: u64,
        options: crate::config::UpdateConfig,
        following: Following,
    ) {
        // The update's options may read back further than the deployment's own: every replica
        // keeps its reports for the update's span from now on.
        let span = crate::observe::span(&options);
        let replicas = self
            .registry
            .upgrade()
            .map(|registry| registry.serving.borrow().clone())
            .unwrap_or_default();
        for replica in replicas.iter().chain(std::iter::once(self)) {
            replica.observer.retain(span);
        }
        let replica = Arc::clone(self);
        tokio::spawn(async move {
            match following {
                Following::Generation { config, modules } => {
                    replica
                        .follow_generation(&updater, number, &options, *config, modules)
                        .await;
                }
                Following::Epoch(target) => {
                    // The epoch form's follower holds both epochs' launch
                    // arguments and configurations across its whole wait.
                    Box::pin(replica.follow_epoch(&updater, number, &options, target)).await;
                }
            }
        });
    }

    /// Waits for the next thing an update's follower acts on: the update's
    /// progress, its trial's end (which commits it), or, once its trial
    /// began, the next report of the replicas `trial` observes, on which the
    /// trial's thresholds are read (a threshold passed rolls it back).
    /// Returns update `number`'s phase afterwards; `None` once another
    /// update, or none, is what the updater states.
    async fn next_phase(
        updater: &crate::updates::Updater,
        number: u64,
        phases: &mut watch::Receiver<Option<(u64, Phase)>>,
        trial: &mut [watch::Receiver<u64>],
    ) -> Option<Phase> {
        let ends = updater.ends(number);
        let expiry = async {
            match ends {
                Some(ends) => tokio::time::sleep_until(ends.into()).await,
                None => std::future::pending().await,
            }
        };
        let reading = crate::observe::next_record(trial);
        tokio::select! {
            changed = phases.changed() => {
                if changed.is_err() {
                    return None;
                }
            }
            () = expiry => updater.conclude(number),
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
        options: &crate::config::UpdateConfig,
        config: Config,
        modules: crate::epoch::ModuleSet,
    ) {
        let mut phases = updater.watch();
        // The reports the trial reads, once it began.
        let mut trial = Vec::new();
        let mut phase = updater.phase_of(number);
        loop {
            match phase {
                Some(Phase::Trial) if trial.is_empty() => {
                    // Serving and the update share an epoch, so the baseline
                    // ends where the switch began.
                    let epoch = self.hold.client().map(|client| client.epoch());
                    let began = updater.begin_trial(|switch| {
                        crate::updates::trial::Trial::begin(
                            options.rollback.clone(),
                            epoch
                                .map(|epoch| {
                                    crate::updates::trial::Baseline::new(
                                        Arc::clone(&self.observer),
                                        epoch,
                                        Some(epoch),
                                        options.rollback.window,
                                        switch,
                                    )
                                })
                                .into_iter()
                                .collect(),
                            request_counts(),
                        )
                    });
                    if began {
                        trial.push(self.observer.records());
                    }
                }
                Some(Phase::Committed) => {
                    // The configuration and the module set are what the next
                    // apply compares with, what a relaunch and a standby
                    // start from and what a successor daemon takes over.
                    let target = config.engine_config();
                    *service::lock(&self.config) = config;
                    if let Some(args) = service::lock(&self.served).as_mut() {
                        args.modules = modules;
                        // The options the graphs were captured with, which a relaunch starts
                        // with.
                        if let Ok(target) = &target {
                            args.engine_config.take_generation(target);
                        }
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
            phase = Self::next_phase(updater, number, &mut phases, &mut trial).await;
        }
    }

    /// [`Self::follow`] for the epoch form, for the deployment's replicas the update moves
    /// (`EpochTarget::moved`): every replica, or the ones the change touches, the others serving
    /// on in their epochs untouched.
    ///
    /// The two epochs need not have as many replicas: target replica `j` is
    /// served under the deployment's replica `j`, made when the serving epoch
    /// has fewer, and a serving replica past the target's last hands the
    /// requests it migrates to a target replica as its own route; a rollback
    /// does the same the other way. Each replica's hold serves its new engine
    /// at once, and requests go to the replicas whose engine serves, so none
    /// waits for a target that is not yet ready.
    async fn follow_epoch(
        self: &Arc<Self>,
        updater: &Arc<crate::updates::Updater>,
        number: u64,
        options: &crate::config::UpdateConfig,
        target: EpochTarget,
    ) {
        let EpochTarget {
            control,
            targets,
            moved,
            config,
            epoch,
            keep,
            processing,
        } = target;
        let config = *config;
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        // The replicas the update moves: every replica, or those the change touches, which their
        // targets replace one for one while the others serve on as they are (a partial update,
        // which only a target of as many replicas as the deployment's makes).
        let deployment: Vec<Arc<Replica>> = registry.serving.borrow().clone();
        let previous: Vec<Arc<Replica>> = moved
            .iter()
            .filter_map(|index| deployment.get(*index).cloned())
            .collect();
        let partial = previous.len() < deployment.len();
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
                    options,
                    Instant::now(),
                );
                Some((Arc::clone(&replica.observer), throttling))
            })
            .collect();
        let steps = Arc::new(std::sync::Mutex::new(proto::admin::StepTime::default()));
        updater.watch_transfers(Arc::clone(&steps));
        tokio::spawn(crate::interference::control(
            Arc::clone(&control) as Arc<dyn crate::interference::Throttle>,
            throttled,
            preparing,
            steps,
        ));

        // Every target replica's engines dial this daemon's request socket as
        // they start, and state that they parked once they resolved their
        // modules and captured their graphs.
        let observers: Vec<Arc<crate::observe::Observer>> = (0..targets.len())
            .map(|index| match previous.get(index) {
                Some(replica) => Arc::clone(&replica.observer),
                None => Arc::new(crate::observe::Observer::new(crate::observe::span(options))),
            })
            .collect();
        let joining =
            targets
                .iter()
                .zip(&observers)
                .enumerate()
                .map(|(index, (target, observer))| {
                    let ordinal = previous.get(index).map_or(index, |replica| replica.ordinal);
                    let sink = stats_sink(Arc::clone(observer), self.recorder.clone(), ordinal);
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
            updater.rollback(proto::admin::Failure::new(
                proto::admin::FailureKind::PrepareFailed,
                reason,
            ));
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
                next = Self::next_phase(updater, number, &mut phases, &mut []) => {
                    phase = next;
                }
            }
        }
        drop(preparation);

        // Model files that process requests otherwise serve with a request processor of their
        // own, bound to what the target's engines describe; one that does not bind ends the
        // update before its switch.
        let target_processor = match processing.map(|prepared| {
            Leader::processor_of(&config, *prepared, clients[0].info(), &targets[0].0)
        }) {
            None => None,
            Some(Ok(processor)) => Some(Arc::new(processor)),
            Some(Err(error)) => {
                let reason = format!("the target's request processor did not bind: {error:#}");
                warn!(epoch, "{reason}");
                updater.rollback(proto::admin::Failure::new(
                    proto::admin::FailureKind::PrepareFailed,
                    reason,
                ));
                return;
            }
        };

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
                        &config,
                        Arc::clone(client),
                        Arc::clone(&observers[index]),
                        self.recorder.clone(),
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
        // client of another, to which it migrates what it holds. With `keeps`,
        // the requests that leave each hold's engine continue with the state
        // they reached where the next engine computes otherwise.
        let route = |replicas: &[Arc<Replica>],
                     serving: &[Arc<Replica>],
                     clients: &[Option<Arc<EngineClient>>],
                     keeps: bool| {
            for (index, replica) in replicas.iter().enumerate() {
                replica.following.store(true, Ordering::SeqCst);
                if let Some(client) = routed_client(clients, index) {
                    if keeps
                        && let Some(previous) = replica.hold.client()
                        && let Some(kept) = crate::engine::hold::Kept::between(&previous, &client)
                    {
                        replica.hold.keep(kept);
                    }
                    replica.hold.serve(client);
                }
            }
            // A partial update replaces the moved replicas' engines in place: the deployment's
            // replicas stay the same.
            if partial {
                registry.serving.send_replace(deployment.clone());
            } else {
                registry.serving.send_replace(serving.to_vec());
            }
        };
        let target_clients: Vec<_> = clients.iter().cloned().map(Some).collect();
        let mut everyone = routed.clone();
        everyone.extend(previous.iter().skip(routed.len()).cloned());
        route(&everyone, &routed, &target_clients, keep);
        // New requests are processed by the target's processor from the switch on; those the
        // previous processor processed run to their ends on the previous epoch, which drains
        // completely (`EpochSwitch::complete`).
        let previous_processor = target_processor.map(|processor| registry.process_with(processor));
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
            .map(|client| client.epoch());
        // The reports the trial reads, once it began.
        let mut trial = Vec::new();

        let mut serving_again: Vec<Option<Arc<EngineClient>>> =
            before.iter().map(|(client, _)| client.clone()).collect();
        let mut relaunched = vec![false; previous.len()];
        let mut returned = false;
        let mut phase = updater.phase_of(number);
        while phase.is_some_and(|phase| !phase.ended()) {
            // The trial begins once every serving engine drained: the target's
            // steps while both epochs stepped, its warm-up and the first steps
            // of the requests migrated to it are not its trial's.
            if phase == Some(Phase::Trial) && trial.is_empty() {
                let judged = || routed.iter().take(previous.len());
                let began = updater.begin_trial(|switch| {
                    let baselines = judged()
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
                    crate::updates::trial::Trial::begin(
                        options.rollback.clone(),
                        baselines,
                        request_counts(),
                    )
                });
                if began {
                    trial = judged().map(|replica| replica.observer.records()).collect();
                }
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
                route(&everyone, &previous, &serving_again, false);
                if let Some(processor) = &previous_processor {
                    registry.process_with(Arc::clone(processor));
                }
                warn!(
                    epoch,
                    "the update returns the deployment to the previous epoch"
                );
            }
            tokio::select! {
                changed = served.changed() => {
                    if changed.is_err() {
                        break;
                    }
                }
                next = Self::next_phase(
                    updater,
                    number,
                    &mut phases,
                    &mut trial,
                ) => phase = next,
            }
        }

        // The committed target stays routed, and its update published it
        // (`Update::committed`). An update that ended otherwise routes the
        // requests back to the previous epoch, unless its rollback did.
        if phase != Some(Phase::Committed) && !returned {
            for (replica, (_, args)) in previous.iter().zip(&before) {
                *service::lock(&replica.served) = args.clone();
                *service::lock(&replica.config) = config_before.clone();
            }
            let mut everyone = previous.clone();
            everyone.extend(routed.iter().skip(previous.len()).cloned());
            route(&everyone, &previous, &serving_again, false);
            if let Some(processor) = &previous_processor {
                registry.process_with(Arc::clone(processor));
            }
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
        let generation = control.launch_number();
        let started = async {
            let args = args.context("the replica has no served arguments")?;
            let sink = stats_sink(
                Arc::clone(&self.observer),
                self.recorder.clone(),
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
            Arc::clone(&self.observer),
            self.recorder.clone(),
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

/// The engine the hold of an epoch's replica `index` serves while `clients` serve the
/// deployment: its own engine, or, for a replica past their last, the engine it hands the
/// requests it holds to, the replicas past the last spread over `clients` in index order;
/// none when no engine of `clients` serves.
fn routed_client(clients: &[Option<Arc<EngineClient>>], index: usize) -> Option<Arc<EngineClient>> {
    let spread = index.checked_rem(clients.len())?;
    clients
        .get(index)
        .or(clients.get(spread))
        .cloned()
        .flatten()
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

/// The first rank each host of `config`'s world holds, by host.
///
/// # Errors
///
/// Fails when the world's ranks cannot be placed.
fn first_ranks(config: &proto::config::EngineConfig) -> anyhow::Result<BTreeMap<String, u32>> {
    Ok(crate::engine::control::world_hosts(config)?
        .into_iter()
        .map(|placed| (placed.host, placed.first))
        .collect())
}

/// The first rank `host` holds of the world `ranks` places ([`first_ranks`]).
///
/// # Errors
///
/// Fails when the world places no rank on `host`.
fn first_rank(ranks: &BTreeMap<String, u32>, host: &str) -> anyhow::Result<u32> {
    ranks
        .get(host)
        .copied()
        .with_context(|| format!("the world places no rank on host {host}"))
}

/// The failure-domain rules' view of the serving engine: one engine process per host of its worker
/// group, which form one domain when there are several (`crate::standby::Standbys`).
///
/// # Errors
///
/// Fails when the world's ranks cannot be placed, or place none on a host the engine runs on.
fn domain_processes(
    serving: &EngineClient,
    config: &proto::config::EngineConfig,
) -> anyhow::Result<Vec<crate::standby::MainProcess>> {
    let processes = serving.processes();
    let ranks = first_ranks(config)?;
    let spans = processes.len() > 1;
    processes
        .into_iter()
        .map(|process| {
            Ok(crate::standby::MainProcess {
                rank: first_rank(&ranks, &process.host)?,
                host: process.host,
                group: crate::engine::control::ENGINE_GROUP.to_owned(),
                generation: process.engine,
                standby_generation: None,
                epoch: spans.then(|| serving.epoch()),
            })
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

    /// Whether the epoch this replica's launch states is an epoch the deployment serves: each
    /// replica serves the epoch that last moved it, which is serving or on trial. A switch moves
    /// the replica's launch to its target, and its own engine is then the drained previous
    /// epoch's: it keeps no standby and is not promoted, because nothing routes to it and a
    /// promoted process would be a second engine of a released epoch.
    ///
    /// A replica whose deployment is ending, or that runs no engine of its own, serves no epoch;
    /// before the deployment recorded its first epoch, the epoch the replica starts is the one it
    /// serves.
    fn serves_serving_epoch(&self) -> bool {
        let Some(updater) = self
            .registry
            .upgrade()
            .and_then(|registry| registry.updates.upgrade())
        else {
            return false;
        };
        let Some(served) = service::lock(&self.served).as_ref().map(|args| args.epoch) else {
            return false;
        };
        let epochs = updater.epochs();
        if epochs.serving().is_none() {
            return true;
        }
        epochs.get(served).is_some_and(|record| {
            matches!(
                record.status,
                proto::admin::EpochState::Serving | proto::admin::EpochState::Trial
            )
        })
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
            match domain_processes(serving, &config) {
                Ok(processes) => *domain = Some(crate::standby::Standbys::new(processes)),
                Err(error) => {
                    warn!("the serving engine forms no failure domain: {error:#}");
                    *domain = None;
                }
            }
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
                generations = ?standby.generations,
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
                generations = ?standby.generations,
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
        let deployment = control.deployment().to_owned();
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
        debug!(model = %self.model_name, "keeping the replica's standby");
        let Some(control) = self.control.get().cloned() else {
            debug!("no standby: the replica has no engine control");
            return Ok(());
        };
        let _starting = self.standby_starting.lock().await;
        let Some(serving) = self.hold.client() else {
            debug!("no standby: no engine serves the replica");
            return Ok(());
        };
        let serves_epoch = self.serves_serving_epoch();
        if !keeps_standby(wanted, serving.state(), serves_epoch) {
            debug!(
                state = ?serving.state(),
                serves_epoch,
                "no standby: the serving engine keeps none"
            );
            return Ok(());
        }
        // A standby of another epoch than the one the replica serves stood beside the engine a
        // switch replaced: it is ended with its read-only claim on that epoch, whatever states
        // the replaced engine passed through on its way out.
        let served_epoch = service::lock(&self.served).as_ref().map(|args| args.epoch);
        let stale = service::lock(&self.standby)
            .as_ref()
            .is_some_and(|standby| Some(standby.client.epoch()) != served_epoch);
        if stale {
            self.drop_standby();
        }
        {
            let mut slot = service::lock(&self.standby);
            match slot.as_ref().map(|standby| standby.client.state()) {
                // A parked standby of the served epoch is the one this replica wants.
                Some(EngineState::Parked) => {
                    debug!("no new standby: a parked one stands beside the replica");
                    return Ok(());
                }
                // A standby that failed or ended is replaced: dropping the
                // client closes its control connections, and its processes end.
                Some(_) => *slot = None,
                None => {}
            }
        }
        let Some(mut args) = service::lock(&self.served).clone() else {
            debug!("no standby: the replica has no served arguments");
            return Ok(());
        };
        // A replica whose deployment's updater is gone keeps no standby.
        if self
            .registry
            .upgrade()
            .and_then(|registry| registry.updates.upgrade())
            .is_none()
        {
            debug!("no standby: the deployment's updater is gone");
            return Ok(());
        }
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
            let mut actions = standbys.serve(proto::admin::EpochState::Serving, options, &rooms);
            if actions
                .iter()
                .any(|action| matches!(action, crate::standby::Action::Allocate { .. }))
            {
                for action in &actions {
                    if let crate::standby::Action::Allocate { host, .. } = action {
                        standbys.allocated(host, control.launch_number());
                    }
                }
                actions = standbys.serve(proto::admin::EpochState::Serving, options, &rooms);
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
            crate::standby::Storage::Resident => crate::config::StandbyStorage::Resident,
            crate::standby::Storage::Suspended => crate::config::StandbyStorage::Suspended,
        };
        let sink = stats_sink(
            Arc::clone(&self.observer),
            self.recorder.clone(),
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
        let failure = proto::admin::Failure::new(
            proto::admin::FailureKind::ProcessExit,
            "the serving engine ended".to_owned(),
        );
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
        let placed = service::lock(&self.config)
            .engine_config()
            .and_then(|config| first_ranks(&config))
            .and_then(|ranks| {
                promotions
                    .iter()
                    .map(|(host, generation)| Ok((host, *generation, first_rank(&ranks, host)?)))
                    .collect::<anyhow::Result<Vec<_>>>()
            });
        let placed = match placed {
            Ok(placed) => placed,
            Err(error) => {
                warn!(
                    generations = ?standby.generations,
                    "the standby cannot be promoted, its world placing no rank: {error:#}; the \
                     engine is relaunched"
                );
                *service::lock(&self.domain) = None;
                return None;
            }
        };
        for (host, generation, rank) in placed {
            updater.promote(host, crate::engine::control::ENGINE_GROUP, rank, generation);
        }
        match standby.client.await_serving().await {
            Ok(()) => {
                if let Some(standbys) = service::lock(&self.domain).as_mut() {
                    for (host, generation) in &promotions {
                        standbys.promoted(host, *generation);
                    }
                }
                // A standby parked with the step values of its launch: it takes the replica's
                // current ones before it serves a request.
                let served = service::lock(&self.served).clone();
                if let Some(served) = served {
                    let tuning = proto::config::Tuning::of(&served.engine_config, served.revision);
                    if let Err(error) = standby.client.tune(&tuning).await {
                        warn!(%error, "the promoted standby keeps its launch's step values");
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
                    generations = ?standby.generations,
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
    pub(crate) fn status(&self) -> ReplicaStatus {
        let hold = self.hold();
        let (engine, reason) = match hold.state() {
            HoldState::Engine(state) => match state {
                EngineState::Starting => (ServingState::Starting, None),
                EngineState::Parked => (ServingState::Parked, None),
                EngineState::Serving => (ServingState::Serving, None),
                EngineState::Draining => (ServingState::Draining, None),
                EngineState::Stopped => (ServingState::Stopped, None),
                EngineState::Failed(reason) => (ServingState::Failed, Some(reason)),
            },
            HoldState::Holding => (ServingState::Holding, None),
            HoldState::HandedOver => (ServingState::HandedOver, None),
            HoldState::Closed(reason) => (ServingState::Closed, Some(reason)),
        };
        // The epoch a replica without an engine last executed is the one its served arguments
        // name.
        let epoch = hold.client().map_or_else(
            || {
                service::lock(&self.served)
                    .as_ref()
                    .map_or(0, |args| args.epoch)
            },
            |client| client.epoch(),
        );
        ReplicaStatus {
            engine,
            reason,
            epoch,
            in_flight: hold.client().map_or(0, |client| client.in_flight()),
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

    /// Has the serving engine `client` give back its KV cache and CUDA graph storage and keep
    /// its weights, as vLLM's `sleep` does: new requests wait, the token requests in flight,
    /// image-generating ones included, are recomputed once it wakes, and the video requests in
    /// flight continue from their stored step (`asleep`). The caller holds the replica while no
    /// engine replaces it.
    ///
    /// # Errors
    ///
    /// Fails with the reason the engine did not sleep; requests wait until it wakes all the
    /// same.
    pub(crate) async fn sleep(&self, client: &EngineClient) -> Result<(), UpdateError> {
        if self.asleep.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        if !self.hold().pause() {
            self.sleep_paused.store(true, Ordering::Release);
        }
        client
            .resize_cache(wire::CacheChange::Sleep)
            .await
            .map_err(|error| {
                UpdateError::Failed(anyhow::anyhow!(
                    "the engine did not sleep, and requests wait until it wakes: {error}"
                ))
            })?;
        info!("the engine sleeps: its KV cache and graph storage are given back");
        Ok(())
    }

    /// Has the serving engine `client` restore what [`Self::sleep`] gave back, and lets the
    /// requests it held through. The caller holds the replica while no engine replaces it.
    ///
    /// # Errors
    ///
    /// Fails with the reason the engine did not wake; requests keep waiting then.
    pub(crate) async fn wake(&self, client: &EngineClient) -> Result<(), UpdateError> {
        client
            .resize_cache(wire::CacheChange::Wake)
            .await
            .map_err(|error| {
                UpdateError::Failed(anyhow::anyhow!("the engine did not wake: {error}"))
            })?;
        self.asleep.store(false, Ordering::Release);
        if self.sleep_paused.swap(false, Ordering::AcqRel) {
            self.hold().resume();
        }
        info!("the engine woke");
        Ok(())
    }

    /// Has the serving engine `client` revoke every cached prefix page; the requests in flight
    /// keep theirs (`flush`).
    pub(crate) fn flush_cache(&self, client: &EngineClient) {
        client.flush_cache();
        info!("flushed the prefix cache");
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

/// A resolved checkpoint directory as the engine configuration states it.
///
/// # Errors
///
/// Fails for a directory whose path is not UTF-8.
fn utf8(directory: std::path::PathBuf) -> anyhow::Result<String> {
    directory.into_os_string().into_string().map_err(|path| {
        anyhow::anyhow!(
            "the checkpoint directory {} is not UTF-8, which the engines' configuration states              paths in",
            std::path::Path::new(&path).display()
        )
    })
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

    use crate::api::events::Events as _;
    use proto::request::SamplingParams;
    use proto::request::{Event, GenerationRequest, Request};

    use super::{EngineState, Leader, keeps_standby};
    use crate::test_support::{
        SERVED_MODEL, configured_tokenizer, engine_of, fake_client, leader_over, letter_fake,
        post_json, router, send, send_streamed, sse_documents, test_model, text_fake,
    };
    use testkit::{FakeExecutor, Fixture};

    /// A chat engine over one scheduler, named `engine_model`, over the fake
    /// `shape` makes of the one that samples `a` ([`letter_fake`]).
    ///
    /// The engine's model name labels the process-wide request metrics,
    /// so a test that reads them names its own engine.
    fn letter_engine(
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

    /// The rows `sql` selects from `engine`'s observation store.
    async fn rows(engine: &Leader, sql: &str) -> Vec<serde_json::Value> {
        let rows = engine.recorder().store().query(sql).await.unwrap();
        serde_json::from_slice(&rows.json().unwrap()).unwrap()
    }

    /// The sum of the values of metric `name` measured at scheduler `scheduler` of the test's one
    /// engine, the replica of the same id, with `and` a further condition of the rows.
    async fn total(engine: &Leader, name: &str, scheduler: u32, and: &str) -> f64 {
        let sql = format!(
            "SELECT coalesce(sum(value), 0) AS total FROM stats WHERE name = '{name}' \
             AND labels['replica'] = '{scheduler}' AND labels['scheduler'] = '{scheduler}' {and}"
        );
        rows(engine, &sql).await[0]["total"].as_f64().unwrap()
    }

    /// The observations histogram `name` holds at scheduler `scheduler` of the test's one engine,
    /// the replica of the same id: their count and sum.
    async fn observed(engine: &Leader, name: &str, scheduler: u32) -> (u64, f64) {
        let sql = format!(
            "SELECT histogram FROM stats WHERE name = '{name}' \
             AND labels['replica'] = '{scheduler}' AND labels['scheduler'] = '{scheduler}'"
        );
        rows(engine, &sql)
            .await
            .iter()
            .fold((0, 0.0), |(count, sum), row| {
                let histogram = &row["histogram"];
                (
                    count + histogram["count"].as_u64().unwrap(),
                    sum + histogram["sum"].as_f64().unwrap(),
                )
            })
    }

    /// How many requests of scheduler `scheduler` of replica 0 ended as `outcome`.
    async fn outcomes(engine: &Leader, scheduler: u32, outcome: &str) -> f64 {
        let and = format!("AND labels['outcome'] = '{outcome}'");
        total(engine, "uniserve_requests_total", scheduler, &and).await
    }

    /// A text request that generates until the simulator's EOS.
    fn text_request() -> Request {
        GenerationRequest {
            trace: 0,
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
            hand_off: false,
            lend: None,
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

        // The engine process ends without the leader asking it to; its
        // connection's end is the state the leader reads.
        let mut states = client.states();
        let stopping = Arc::clone(&client);
        tokio::task::spawn_blocking(move || stopping.shutdown())
            .await
            .unwrap();
        while *states.borrow_and_update() != EngineState::Stopped {
            states.changed().await.expect("the client states its end");
        }
        assert!(!engine.is_healthy());

        let error = engine.submit(text_request(), None).await.err();
        assert!(matches!(
            error,
            Some(crate::engine::client::SubmitError::Unavailable(_))
        ));
    }

    /// Concurrent requests use independent replicas, and losing one leaves
    /// the other available through the same public service.
    #[tokio::test]
    async fn concurrent_requests_spread_over_engine_cores() {
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
        for _ in 0..2 {
            let (stream, core) = engine
                .submit(text_request(), None)
                .await
                .expect("submit request");
            streams.push(stream);
            serving.push(core);
        }
        assert_ne!(serving[0], serving[1]);
        for stream in &mut streams {
            while let Some(event) = stream.recv().await {
                if let Event::Finished { reason, .. } = event {
                    assert_eq!(reason, proto::request::FinishReason::Eos);
                    break;
                }
            }
        }
        let lost = Arc::clone(&clients[0]);
        let mut states = lost.states();
        tokio::task::spawn_blocking(move || lost.shutdown())
            .await
            .unwrap();
        while *states.borrow_and_update() != EngineState::Stopped {
            states.changed().await.expect("the client states its end");
        }
        assert!(engine.is_healthy());
        let (mut stream, replica) = engine.submit(text_request(), None).await.unwrap();
        assert_eq!(replica.replica, 1);
        let mut finished = false;
        while let Some(event) = stream.recv().await {
            if let Event::Finished { reason, .. } = event {
                assert_eq!(reason, proto::request::FinishReason::Eos);
                finished = true;
            }
        }
        assert!(finished);
        engine.shutdown().await.expect("shutdown replica");
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
            let engine = Arc::new(letter_engine(engine_model, ending));
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
            let (_, seconds) = observed(&engine, "uniserve_e2e_request_latency_seconds", 0).await;
            assert!(seconds >= wait.as_secs_f64(), "{seconds} s");
            engine.shutdown().await.unwrap();
        });
    }

    /// A completed request is recorded once in the observation store under the scheduler that
    /// served it: the token counters and token histograms carry the counts of its usage report,
    /// its finish reason is counted, each latency histogram whose input it carries holds one
    /// observation, and one `request` event states its lengths.
    #[tokio::test]
    async fn completed_request_is_recorded_in_the_store() {
        let engine = Arc::new(letter_engine("request-metrics-model", ending));
        let router = router(Arc::clone(&engine));

        let (status, document) = chat(&router, &chat_request(3, &[], false)).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{document}");
        assert_eq!(document["choices"][0]["finish_reason"], "length");
        assert_eq!(document["usage"]["completion_tokens"], 3);
        let prompt_tokens = &document["usage"]["prompt_tokens"];

        assert_eq!(outcomes(&engine, 0, "length").await, 1.0);
        assert_eq!(
            total(&engine, "uniserve_prompt_tokens_total", 0, "").await,
            prompt_tokens.as_f64().unwrap()
        );
        assert_eq!(
            total(&engine, "uniserve_generation_tokens_total", 0, "").await,
            3.0
        );
        for histogram in [
            "uniserve_request_prompt_tokens",
            "uniserve_request_generation_tokens",
            "uniserve_e2e_request_latency_seconds",
            "uniserve_request_queue_time_seconds",
            "uniserve_ttft_seconds",
            "uniserve_tpot_seconds",
        ] {
            assert_eq!(observed(&engine, histogram, 0).await.0, 1, "{histogram}");
        }
        let events = rows(
            &engine,
            "SELECT fields FROM events WHERE kind = 'request' AND labels['scheduler'] = '0'",
        )
        .await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["fields"]["output_tokens"], "3");
        assert_eq!(events[0]["fields"]["outcome"], "length");
        engine.shutdown().await.unwrap();
    }

    /// A request reports, as its usage's `prompt_tokens_details.cached_tokens`, the prompt
    /// tokens its admission found cached: none for the first request of a prompt, and for the
    /// same prompt again the reuse limit, its whole blocks short of its last token, which is
    /// computed (`(prompt_tokens - 1)` rounded down to the block). A stream that asks for usage
    /// reports the same in its usage chunk.
    #[tokio::test]
    async fn a_repeated_prompt_reports_its_cached_tokens() {
        let engine = Arc::new(letter_engine("cached-tokens-model", ending));
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
        while let Some(frame) = stream.next_frame().await {
            text.push_str(std::str::from_utf8(&frame).unwrap());
        }
        let usage: Vec<serde_json::Value> = sse_documents(&text)
            .into_iter()
            .filter_map(|document| document.get("usage").cloned())
            .collect();
        let [usage] = &usage[..] else {
            panic!("one usage chunk: {text}");
        };
        assert_eq!(cached(usage), Some(reused), "{text}");
        engine.shutdown().await.unwrap();
    }

    /// Each request is recorded under the scheduler that served it.
    #[tokio::test]
    async fn requests_are_recorded_under_the_scheduler_that_served_each() {
        let engine = Arc::new(engine_of(
            crate::test_support::CHAT,
            testkit::launch("core-metrics-model", 2),
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

        for scheduler in 0..2 {
            assert_eq!(
                outcomes(&engine, scheduler, "length").await,
                1.0,
                "scheduler {scheduler}"
            );
        }
        engine.shutdown().await.unwrap();
    }

    /// A client that disconnects mid-generation is counted as an aborted
    /// request with its latency, while its partial output, which never
    /// reached a usage report, adds nothing to the token metrics.
    #[tokio::test]
    async fn disconnected_request_is_recorded_as_aborted_without_token_usage() {
        let engine = Arc::new(letter_engine("disconnect-metrics-model", |fake| fake));
        let router = router(Arc::clone(&engine));

        let mut stream = send_streamed(
            &router,
            post_json("/v1/chat/completions", &chat_request(1_000, &[], true)),
        )
        .await;
        loop {
            let frame = stream.next_frame().await.expect("the stream goes on");
            if std::str::from_utf8(&frame).is_ok_and(|frame| frame.contains(r#""content":"a"#)) {
                break;
            }
        }
        // The connection is gone: the body's frames have no reader.
        drop(stream);

        while outcomes(&engine, 0, "abort").await < 1.0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            observed(&engine, "uniserve_e2e_request_latency_seconds", 0)
                .await
                .0,
            1
        );
        assert_eq!(
            total(&engine, "uniserve_generation_tokens_total", 0, "").await,
            0.0
        );
        assert_eq!(
            observed(&engine, "uniserve_request_generation_tokens", 0)
                .await
                .0,
            0
        );
        engine.shutdown().await.unwrap();
    }

    /// The device strings of `ranks` local ranks of a default deployment's device kind.
    fn rank_devices(ranks: std::ops::Range<u32>) -> Vec<String> {
        let kind = crate::config::Config::default().device_config.device;
        ranks
            .map(|rank| kind.rank_device(rank).to_string())
            .collect()
    }

    /// The engine configuration of a deployment of `data` schedulers of `tensor` ranks each,
    /// with expert parallelism (`moe_config.expert_parallel`) when `experts` holds.
    fn deployment(tensor: usize, data: usize, experts: bool) -> proto::config::EngineConfig {
        let mut options = crate::config::Config {
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
        assert_eq!(devices, vec![rank_devices(0..2), rank_devices(2..4)]);
        assert!(replicas.iter().all(|replica| {
            replica.parallel.data_parallel_size.get() == 1 && replica.scheduler == config.scheduler
        }));
    }

    /// Each replica's role reaches the engine that runs its scheduler: an independent replica's
    /// engine states the role of its own id, and the expert union's one engine every
    /// scheduler's, by the deployment's ids.
    #[test]
    fn every_engine_states_the_roles_of_its_replicas() {
        use scheduling::routing::Role;

        let configured = |experts: bool| {
            let mut options = crate::config::Config {
                data_parallel_size: std::num::NonZeroUsize::new(3).unwrap(),
                ..Default::default()
            };
            options.moe_config.expert_parallel = experts;
            for (id, role) in [("0", Role::Prefill), ("2", Role::Decode)] {
                options
                    .replicas
                    .insert(id.to_owned(), crate::config::ReplicaConfig { role });
            }
            options.engine_config().expect("the configuration resolves")
        };

        let independent = super::replica_configs(&configured(false)).unwrap();
        let roles: Vec<&[Role]> = independent.iter().map(|engine| &engine.roles[..]).collect();
        assert_eq!(
            roles,
            [&[Role::Prefill][..], &[Role::Mixed], &[Role::Decode]]
        );
        let union = super::replica_configs(&configured(true)).unwrap();
        let [engine] = &union[..] else {
            panic!("an expert union runs one engine");
        };
        assert_eq!(engine.roles, [Role::Prefill, Role::Mixed, Role::Decode]);
        assert_eq!(
            [(0, 0), (1, 0), (2, 0), (0, 2)]
                .map(|(ordinal, scheduler)| { super::replica_id(ordinal, scheduler) }),
            [0, 1, 2, 2],
            "a replica's id is its scheduler's index in the deployment"
        );
    }

    /// The deployment's own route: a new token request to the least loaded scheduler that
    /// computes prompts, generated there when it is mixed and handed off to the least loaded
    /// that generates tokens otherwise; a continuation to one that generates tokens; a video
    /// request to a mixed one; nothing when no role takes the request.
    #[test]
    fn a_route_pairs_the_roles_that_take_the_request() {
        use proto::request::{DiffusionRequest, DiffusionSamplingParams, Lend, Trajectory};
        use scheduling::routing::Role;

        let fresh = text_request();
        let Request::Generation(generation) = &fresh else {
            unreachable!("a text request is a token request");
        };
        let continued = Request::Generation(Box::new(GenerationRequest {
            lend: Some(Lend {
                endpoint: None,
                session: 1,
                owner: 2,
                tokens: 16,
                stored: 0,
            }),
            ..(**generation).clone()
        }));
        let video = Request::Diffusion(Box::new(DiffusionRequest {
            request_id: Default::default(),
            prompt_token_ids: vec![1, 2],
            conditioning: Vec::new(),
            conditions: Vec::new(),
            media: Vec::new(),
            priority: 0,
            sampling: DiffusionSamplingParams {
                num_frames: 25,
                video_units: 1,
                seed: 3,
                width: 64,
                height: 32,
                condition_rows: 0,
            },
            trajectory: Trajectory::default(),
            encoder_inputs: None,
            trace: 0,
        }));

        let split = [
            (Role::Prefill, 3),
            (Role::Decode, 5),
            (Role::Prefill, 1),
            (Role::Decode, 2),
        ];
        assert_eq!(super::pair(&split, &fresh), Some((2, 3)));
        assert_eq!(super::pair(&split, &continued), Some((3, 3)));
        assert_eq!(super::pair(&split, &video), None);

        let mixed = [(Role::Prefill, 4), (Role::Mixed, 1), (Role::Decode, 0)];
        assert_eq!(super::pair(&mixed, &fresh), Some((1, 1)));
        assert_eq!(super::pair(&mixed, &continued), Some((2, 2)));
        assert_eq!(super::pair(&mixed, &video), Some((1, 1)));

        assert_eq!(super::pair(&[(Role::Decode, 0)], &fresh), None);
    }

    /// Replicas spread over a second host run their engines where their ranks are: each one's
    /// scheduler node is the host of its first rank, and its world places.
    #[test]
    fn replicas_on_another_host_run_their_engine_there() {
        let mut options = crate::config::Config {
            data_parallel_size: std::num::NonZeroUsize::new(4).unwrap(),
            hosts: vec!["rank-1".to_owned()],
            ..Default::default()
        };
        options.device_config.worker_hosts = vec!["rank-2".to_owned()];
        let config = options.engine_config().expect("the configuration resolves");

        let replicas = super::replica_configs(&config).unwrap();
        let hosts: Vec<&str> = replicas
            .iter()
            .map(|replica| replica.parallel.engine_host.as_str())
            .collect();
        assert_eq!(hosts, ["rank-1", "rank-1", "rank-2", "rank-2"]);
        for replica in &replicas {
            let world = crate::engine::control::world_hosts(replica).expect("the replica places");
            assert_eq!(world.len(), 1);
            assert_eq!(world[0].host, replica.parallel.engine_host);
        }
        assert_eq!(super::epoch_hosts(&config).unwrap(), ["rank-1", "rank-2"]);
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
        let modules = crate::epoch::modules_of(&crate::config::Config::default());
        let launches = super::epoch_launches(&config, &modules, 7, 0).unwrap();
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
        assert_eq!(devices, vec![rank_devices(0..2), rank_devices(2..4)]);
        assert!(launches.iter().all(|launch| {
            let replica = &launch.engine_config;
            launch.epoch == 7
                && replica.parallel.data_parallel_size.get() == 1
                && replica.moe.expert_parallel
        }));
    }

    /// A pipeline checkpoint processes requests as the served one only with the same index and,
    /// in every part its family declares, the same files, whichever side holds a file alone; a
    /// part the family does not declare decides nothing.
    #[tokio::test]
    async fn a_pipeline_differs_by_a_part_file_either_side_holds_alone() {
        // The tests' program's video family declares its `processor` part.
        let program = crate::api::test_support::program();
        let checkpoint = |files: &[(&str, &str)]| {
            let directory = tempfile::tempdir().unwrap();
            std::fs::write(
                directory
                    .path()
                    .join(scheduling::processing::RootFile::ModularIndex.name()),
                r#"{"_class_name": "TestVideoPipeline"}"#,
            )
            .unwrap();
            for (name, text) in files {
                let path = directory.path().join(name);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, text).unwrap();
            }
            directory
        };
        let compare = |served: &tempfile::TempDir, other: &tempfile::TempDir| {
            let program = Arc::clone(&program);
            let served = served.path().to_str().unwrap().to_owned();
            let other = other.path().to_str().unwrap().to_owned();
            async move { super::same_processing(&program, &served, &other).await }
        };

        let processor = ("processor/preprocessor_config.json", "{}");
        let served = checkpoint(&[processor, ("scheduler/config.json", "{}")]);
        let same = checkpoint(&[processor, ("scheduler/config.json", "{\"steps\": 8}")]);
        compare(&served, &same)
            .await
            .expect("an undeclared part decides nothing");

        let more = checkpoint(&[processor, ("processor/extra.json", "{}")]);
        for (from, to) in [(&served, &more), (&more, &served)] {
            let error = compare(from, to)
                .await
                .expect_err("a file one side holds alone");
            assert!(error.contains("processor/extra.json"), "{error}");
        }
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
