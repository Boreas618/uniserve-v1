//! The started epoch of a main process, [`Engine`], and how it is built and ended.
//!
//! An engine has two sides. The schedulers' side is one [`Scheduler`] per data-parallel rank on
//! a thread of its own, each over its end of a queue pair of the epoch's session with the buffer
//! pool and an executor of its worker groups, and the routing policy of a node with several.
//! [`Engine::new`] builds it over a session and one executor per scheduler, which is where the
//! binary and a test of the control thread part: the binary's executors are the ranks', a test's
//! are fakes, and everything above them is the same. The ranks' side is what the rank start
//! adds: the node's workers, the weight binding over the ranks' storages, the head of the session,
//! the Build Server and the links to the epoch's other main processes.
//!
//! [`start`] is the binary's construction (`specs/engine-worker-port.md` §1.1,
//! `specs/buffer-pool.md` §14.1): the ranks load to the division of their GPUs' quotas, the head
//! opens the session stating each region's units with no segment, every rank maps the segments
//! the answer states and creates the rest, the head grants what was created, the workers are
//! spawned and warm, the schedulers are built over them, and each grows its KV regions into the
//! granted segments. A start is not an update (`specs/updates.md` §2): the generation it serves
//! is [`FIRST_GENERATION`], with nothing kept beside it.
//!
//! [`Engine::close`] ends the epoch in the order its parts depend on one another: the schedulers,
//! then the workers, the peer links, the session, and the devices' contexts.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::batch::Capacity;
use crate::batch::Phase;
use crate::executor::Route;
use crate::executor::peers::{Liveness, Remote, Upstream};
use crate::profiling::counters::Counters;
use crate::worker::Worker;
use crate::worker::completion::Outcome;
use anyhow::{Context as _, Result, bail};
use build_server::Target;
use crossbeam_channel::{Receiver, Sender};
use ops::Generation;
use platform::DeviceId;
use proto::config::{EngineArgs, SchedulerConfig, WorkerGroupConfig};
use proto::request::{GenerationFeatures, GenerationLimits};
use scheduling::limits::{Limits, Parameters};
use scheduling::view::{RankView, Snapshot};
use storage::QueuePair;
use storage::runtime::{GroupId, Open, Opened};

use super::epoch;
use super::ranks::{
    Backing, FIRST_GENERATION, LoadedRank, Meeting, bind, capacities, group_numbers, layout,
    load_ranks, open_ranks, spawn,
};
use crate::executor::{Executor, Shadow, WorkerExecutor};
use crate::memory::client::StorageClient;
use crate::modules::load;
use crate::process::binding::{Target as LoadTarget, WeightBinding};
use crate::process::ranks::rank::ModelRunner;
use crate::process::ranks::rank::text::Keys;
use crate::scheduler::policy::{self, Policy};
use crate::scheduler::{Message, Output, Scheduler};

/// The device a launch names when its ranks are backed by nothing outside the process
/// (`engine_config.device.device`): [`Backing::Meta`].
const META_DEVICE: &str = "meta";

/// How long the scheduler node waits for every peer main process to register, and a peer for the
/// scheduler node to accept it: each starts at once and registers before it loads anything.
const PEER_START: Duration = Duration::from_secs(300);

/// The started epoch: its schedulers and the ranks they drive (module documentation).
pub struct Engine {
    /// One per data-parallel rank, in the order of the session's schedulers.
    pub(crate) schedulers: Vec<SchedulerThread>,
    /// The routing policy's instance, which chooses a new request's scheduler; `None` with one
    /// scheduler, which every request goes to.
    pub(crate) route: Option<Policy>,
    /// The generation limits requests are checked against, as the schedulers took them.
    pub(crate) limits: GenerationLimits,
    /// What each worker group of the session serves, by its number.
    pub(crate) capacities: Vec<(GroupId, Capacity)>,
    /// The ceilings a policy of this engine declares its limits within: those of every group's
    /// captured buckets.
    pub(crate) ceilings: Limits,
    /// The generation batches are assembled for, which the schedulers read and a switch moves.
    pub(crate) served: Arc<AtomicU64>,
    /// What ends the session with the buffer pool once the workers have stopped: the head's
    /// `Close`, or whatever stands for the buffer pool in its place.
    pub(crate) session: Box<dyn FnOnce() -> Result<()> + Send>,
    /// The node's workers, one per rank, in rank order; empty for an engine of no rank.
    pub(crate) workers: Vec<Arc<Worker>>,
    /// The node's workers and peer links by worker group, which the schedulers' executors
    /// hand their jobs to.
    pub(crate) groups: Option<Arc<Route>>,
    /// The changes the engine makes to what its ranks hold; of no rank by default, so every
    /// operation acts on none.
    pub(crate) binding: Arc<WeightBinding>,
    /// The head of the session with the buffer pool, which grants and withdraws its segments.
    pub(crate) head: Option<Arc<Mutex<StorageClient>>>,
    /// The Build Server's client and the targets of the node's devices, which a generation's
    /// kernels are selected and installed through; a preparation fails without it.
    pub(crate) build: Option<(Arc<Mutex<build_server::Client>>, Vec<Target>)>,
    /// On the scheduler node, the links to the epoch's other main processes.
    pub(crate) remotes: Vec<Arc<Remote>>,
    /// On a peer, its link to the scheduler node.
    pub(crate) upstream: Option<Upstream>,
    /// Whether every peer link still serves, which ends the process when one is lost.
    pub(crate) liveness: Option<Arc<Liveness>>,
    /// CUPTI's always-on counters of the ranks' streams and of the threads that wait for the
    /// devices, for as long as the engine serves; `None` without devices, or when the process is
    /// not subscribed to CUPTI.
    pub(crate) counters: Option<Counters>,
}

/// One scheduler of an [`Engine`] and how the control thread reaches it.
pub(crate) struct SchedulerThread {
    /// The scheduler's one channel in.
    pub(crate) inbox: Sender<Message>,
    /// The scheduler's one channel out.
    pub(crate) outputs: Receiver<Output>,
    /// The shadow pass a generation update opens over the scheduler's batches.
    pub(crate) shadow: Arc<Mutex<Shadow>>,
    /// The thread the scheduler runs on until it stops.
    pub(crate) thread: JoinHandle<()>,
}

impl Engine {
    /// Each scheduler's inbox, in the order of the session's schedulers: what a job sends its
    /// messages to.
    pub(crate) fn inboxes(&self) -> Vec<Sender<Message>> {
        self.schedulers
            .iter()
            .map(|scheduler| scheduler.inbox.clone())
            .collect()
    }

    /// Each scheduler's shadow pass, in the order of the session's schedulers.
    pub(crate) fn shadows(&self) -> Vec<Arc<Mutex<Shadow>>> {
        self.schedulers
            .iter()
            .map(|scheduler| Arc::clone(&scheduler.shadow))
            .collect()
    }

    /// The schedulers of the session `open` stated and `opened` answered, one per entry of
    /// `open.schedulers` in that order, each on its thread over its end of its queue pair in
    /// `pairs` and the executor `executor_of(index, groups, inbox)` gives it: the executor of
    /// its worker groups `groups`, which sends to its channel `inbox`, and the shadow pass over
    /// its batches. `capacities` is what each worker group of the session serves, by its number;
    /// `policy` the module set's policy module and its parameters, which every scheduler and, with
    /// several, the routing policy instantiate; `session` what ends the session.
    ///
    /// Each scheduler's policy is instantiated on the scheduler's own state, so the replicas and
    /// the phases its groups serve are what it starts from. The generation batches are assembled
    /// for is [`FIRST_GENERATION`]. The ranks' side is empty: no worker, a weight binding of no
    /// rank, no head, Build Server or peer link.
    ///
    /// # Errors
    ///
    /// A layout whose schedulers, groups or queue pairs do not match, an executor that cannot be
    /// built, a policy that does not instantiate or declares limits outside a scheduler's
    /// ceilings, a scheduler the startup checks refuse, or a thread that cannot start.
    #[expect(
        clippy::too_many_arguments,
        reason = "the schedulers' side comes from five owners: the launch, the session with its \
                  queue pairs, the groups' capacities, the policy and the executors"
    )]
    pub fn new(
        launch: &EngineArgs,
        open: &Open,
        opened: &Opened,
        pairs: Vec<QueuePair>,
        capacities: Vec<(GroupId, Capacity)>,
        (module, parameters): (Arc<wasmtime::Module>, Parameters),
        mut executor_of: impl FnMut(
            usize,
            &[GroupId],
            Sender<Message>,
        ) -> Result<(Box<dyn Executor>, Arc<Mutex<Shadow>>)>,
        session: Box<dyn FnOnce() -> Result<()> + Send>,
    ) -> Result<Self> {
        let config = &launch.engine_config;
        if pairs.len() != open.schedulers.len() {
            bail!(
                "a session of {} schedulers was answered with {} queue pairs",
                open.schedulers.len(),
                pairs.len()
            );
        }
        let (numbers, _) = group_numbers(
            &config.parallel.worker_groups,
            config.parallel.data_parallel_size.get(),
        )?;

        // The workers compute `max_logprobs` candidates at every position: what a request may ask
        // for. The model's facts state no feature before its workers do: the engine serves the
        // features its schedulers' groups serve, which each scheduler resolves for its own. Every
        // numeric limit is the lower of the launch's and what the groups were sized for.
        let ceilings = policy::ceilings(capacities.iter().map(|(_, capacity)| capacity));
        let mut limits = lowered(
            GenerationLimits {
                max_logprobs: config.max_logprobs,
                ..config.model.generation_limits.clone()
            },
            &capacities,
            &ceilings,
        );
        let served = Arc::new(AtomicU64::new(FIRST_GENERATION));

        // The schedulers' limits as the deployment states them, which each scheduler lowers to
        // what its groups' workers captured; the captured sizes in tokens are the graph tiers.
        let mut tiers: Vec<u32> = capacities
            .iter()
            .flat_map(|(_, capacity)| capacity.buckets.iter().map(|shape| shape.tokens))
            .collect();
        tiers.sort_unstable();
        tiers.dedup();
        let scheduling = SchedulerConfig::of(
            config,
            config.scheduler.max_batch.get(),
            config
                .max_num_batched_tokens
                .map_or(config.scheduler.max_num_batched_tokens, NonZeroUsize::get),
            tiers,
        );

        let mut schedulers = Vec::with_capacity(open.schedulers.len());
        let mut features = GenerationFeatures::empty();
        for (index, (listed, pair)) in open.schedulers.iter().zip(pairs).enumerate() {
            // The scheduler's groups, each with its configuration and what its ranks serve.
            let mut groups: Vec<(WorkerGroupConfig, Capacity)> = Vec::with_capacity(listed.len());
            for id in listed {
                let group = numbers
                    .iter()
                    .position(|number| number == id)
                    .map(|position| config.parallel.worker_groups[position].clone())
                    .with_context(|| format!("the deployment numbers no group {}", id.0))?;
                let capacity = capacities
                    .iter()
                    .find(|(group, _)| group == id)
                    .map(|(_, capacity)| capacity.clone())
                    .with_context(|| format!("worker group {} states no capacity", id.0))?;
                groups.push((group, capacity));
            }

            let (inbox, mailbox) = crossbeam_channel::unbounded();
            let (output, outputs) = crossbeam_channel::unbounded();
            let (executor, shadow) = executor_of(index, listed, inbox.clone())?;

            // The scheduler instantiates its policy on its own state.
            let scheduler = Scheduler::new(
                scheduling.clone(),
                limits.clone(),
                open,
                opened,
                index,
                pair,
                groups,
                Arc::clone(&module),
                parameters.clone(),
                executor,
                output,
                Arc::clone(&served),
            )
            .with_context(|| format!("starting scheduler {index}"))?;
            features |= scheduler.limits().features;

            let thread = std::thread::Builder::new()
                .name(format!("scheduler-{index}"))
                .spawn(move || scheduler.run(mailbox))
                .with_context(|| format!("starting scheduler {index}'s thread"))?;
            schedulers.push(SchedulerThread {
                inbox,
                outputs,
                shadow,
                thread,
            });
        }

        limits.features = features;

        // With several schedulers the routing policy chooses each new request's, from every
        // scheduler as a rank of its own.
        let route = if schedulers.len() > 1 {
            let ranks = (0..schedulers.len())
                .map(|index| RankView {
                    index,
                    in_flight: 0,
                    alive: true,
                })
                .collect();
            let snapshot = Snapshot {
                ranks,
                ..Snapshot::default()
            };
            Some(Policy::new(module, parameters, snapshot, &ceilings)?)
        } else {
            None
        };

        Ok(Self {
            schedulers,
            route,
            limits,
            capacities,
            ceilings,
            served,
            session,
            workers: Vec::new(),
            groups: None,
            binding: Arc::new(WeightBinding::new(Vec::new(), LoadTarget::default())),
            head: None,
            build: None,
            remotes: Vec::new(),
            upstream: None,
            liveness: None,
            counters: None,
        })
    }

    /// Ends the epoch in the order its parts depend on one another:
    ///
    /// 1. every scheduler is told to shut down, which ends every request it holds as aborted;
    ///    each output it sends until its channel closes goes to `forward` with the scheduler's
    ///    index, the events of those requests among them, and its thread is joined;
    /// 2. the workers' queues close, ending the job in flight and cancelling the rest, and
    ///    their threads are joined;
    /// 3. the peer links close, which the epoch's other main processes read as this one's end;
    /// 4. the session ends ([`Engine::new`]'s `session`), the workers having stopped;
    /// 5. the devices' primary contexts are reset, which releases the driver state the process
    ///    created, when the engine's workers served on devices and nothing else holds the
    ///    owners of their storage (`docs/architecture.md`, the orderly close).
    ///
    /// The caller makes no use of a device afterwards: this is the main process's last.
    ///
    /// # Errors
    ///
    /// The first failure: a scheduler or a worker thread that panicked, the session's end, or
    /// a context's reset. Every step runs whatever an earlier one returned.
    pub fn close(self, mut forward: impl FnMut(usize, Output)) -> Result<()> {
        let Self {
            schedulers,
            route,
            workers,
            groups,
            binding,
            head,
            build,
            remotes,
            upstream,
            liveness,
            session,
            counters,
            ..
        } = self;
        let mut outcome: Result<()> = Ok(());

        // A scheduler holds a sender to its own channel through its executor's waiters, so its
        // loop ends at `Shutdown`, not when this side's sender goes.
        for scheduler in &schedulers {
            let _ = scheduler.inbox.send(Message::Shutdown);
        }
        for (index, scheduler) in schedulers.into_iter().enumerate() {
            for output in &scheduler.outputs {
                forward(index, output);
            }
            if scheduler.thread.join().is_err() {
                outcome = outcome.and(Err(anyhow::anyhow!("scheduler {index} panicked")));
            }
        }
        drop(route);
        // The counters record the streams the workers issue on, and stop before they close.
        drop(counters);

        // A worker's thread holds its runners, and an expert union's runners reach the other
        // members' workers: every queue closes before any worker is joined.
        let devices = workers.iter().any(|worker| worker.device().is_some());
        for worker in &workers {
            worker.queue().close();
        }
        drop(groups);
        for worker in workers {
            match Arc::into_inner(worker) {
                Some(worker) => {
                    if let Err(error) = worker.close() {
                        outcome = outcome.and(Err(error.into()));
                    }
                }
                None => tracing::warn!(
                    "a worker is still held after the engine closed; its thread ends when the \
                     last holder drops it"
                ),
            }
        }
        // The weight binding holds the owners of every rank's device storage.
        let released = Arc::into_inner(binding).is_some();

        for remote in &remotes {
            remote.close();
        }
        if let Some(upstream) = &upstream {
            upstream.close();
        }
        drop((remotes, upstream, liveness, build));

        outcome = outcome.and(session().context("ending the session with the buffer pool"));
        drop(head);

        if devices && released {
            // SAFETY: the engine's workers are joined and the owners of their devices' storage
            // released above; the main process makes no use of a device after its engine's close,
            // which is its last step, and no other thread of it issues on one.
            let reset = unsafe { platform::PrimaryContext::reset_all() };
            outcome = outcome.and(reset.map(drop).context("resetting the devices' contexts"));
        } else if devices {
            tracing::warn!(
                "the weight binding is still held after the engine closed: the devices' contexts \
                 are left to the process's exit"
            );
        }
        outcome
    }
}

/// Starts the epoch `launch` describes on this node: its ranks, the session with the buffer
/// pool, the workers and the schedulers (module documentation). Its checkpoint is served by
/// the family of `registry` that claims the class it declares, its ranks' storage is at the
/// Storage Server the daemon's claim answered with (`storage`), their kernels are resolved
/// through `build`, the Build Server's client, and `policy` is the module set's scheduling
/// policy, resolved through the same client (`modules::load::policy`). `liveness` is what a lost
/// peer link marks; an engine of one main process holds it nowhere.
///
/// The ranks load onto the meta device when the launch's `engine_config.device.device` is
/// `meta`: nothing is allocated, no kernel is resolved, and the generation served holds none.
///
/// # Errors
///
/// What the family, a rank's load or opening, the session, a warm-up, a scheduler or the
/// policy refuses; a rank on a device with no Build Server to resolve its kernels; and a peer
/// main process, whose ranks map their memory from the session the scheduler node opens, which
/// no peer frame carries to it.
pub(crate) fn start(
    launch: &EngineArgs,
    registry: &nn::family::Registry,
    storage: &crate::memory::client::StorageLaunch,
    build: Option<Arc<Mutex<build_server::Client>>>,
    policy: (Arc<wasmtime::Module>, Parameters),
    liveness: Arc<Liveness>,
) -> Result<Engine> {
    let family =
        crate::process::catalogue::resolve(registry, Path::new(&launch.engine_config.model.model))?;
    let model_runner = ModelRunner::of(launch, family)?;
    let meta = launch.engine_config.device.device == META_DEVICE;

    // CUPTI accepts one subscriber per process, and what it observes begins at the subscription,
    // so a process that serves on devices subscribes before it loads a module
    // (`docs/profiling.md`).
    if !meta {
        crate::profiling::counters::subscribe();
    }

    // The epoch's other main processes, before any rank loads: the ranks' host group and their
    // communicators span every host, so each process must know the others are there first.
    let config = &launch.engine_config.parallel;
    let (numbers, _) = group_numbers(&config.worker_groups, config.data_parallel_size.get())?;
    let (meeting, links) = meet(launch)?;
    let remotes = match links {
        Links::Alone => Vec::new(),
        Links::Accepted(channels) => {
            let mut remotes = Vec::with_capacity(channels.len());
            for (rank, channel) in channels {
                let held = held_groups(&config.worker_groups, &numbers, rank)?;
                remotes.push((held, Remote::spawn(rank, channel, Arc::clone(&liveness))?));
            }
            remotes
        }
        Links::Registered(channel) => {
            drop(channel);
            bail!(
                "this main process is a peer of its epoch's scheduler node, and its ranks map \
                 their memory from the session the scheduler node opens, which no peer frame \
                 carries"
            )
        }
    };

    // Step 9 of each rank: the kernels its profile run names, selected for the devices'
    // architectures and installed on its own device as the start's generation.
    let targets = if meta {
        Vec::new()
    } else {
        model_runner.targets()?
    };
    let resolve = |device: DeviceId, keys: &Keys| -> Result<Arc<Generation>> {
        let build = build
            .as_ref()
            .context("the start holds no Build Server client to resolve a device's kernels")?;
        let mut by_operator: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (operator, key) in keys {
            by_operator
                .entry((*operator).to_owned())
                .or_default()
                .insert(key.clone());
        }

        let mut client = service::lock(build);
        let (packages, routes) =
            load::select(&mut client, &targets, &launch.modules.kernels, &by_operator)?;
        let entries = load::install(&mut client, device, &packages, &routes)?;
        Ok(Arc::new(Generation::new(FIRST_GENERATION, entries)))
    };
    let backing = if meta {
        Backing::Meta
    } else {
        Backing::Serving {
            storage,
            kernels: &resolve,
        }
    };
    let mut ranks = load_ranks(model_runner.ranks()?, backing, launch, meeting)?;

    // The session, opened before the ranks map memory: its device KV regions state no segment.
    // From the answer on, the layout states each region numbered as the buffer pool holds it: a
    // region an earlier epoch named keeps that epoch's numbering.
    let mut head = StorageClient::connect_head(
        Path::new(&storage.socket),
        &launch.engine_config.deployment,
        launch.epoch,
    )?;
    let pairs = head.open(layout(launch, &ranks)?)?;
    let (open, opened) = head
        .session()
        .map(|(open, opened)| (open.clone(), opened.clone()))
        .context("the session with the buffer pool is not open")?;

    // Every rank maps the segments the answer states and creates the rest, which the head
    // grants once every rank holds them.
    let created = open_ranks(&mut ranks, &open, &opened)?;
    for (region, segments) in &created {
        for &(segment, first, units) in segments {
            head.grant(region, segment, first, units)?;
        }
    }
    bind(&mut ranks, &open, &opened)?;
    let (route, workers) = spawn(&mut ranks, &head, &pairs, remotes.clone())?;

    // Range markers and PC sampling attach to what a graph captures, so they precede the
    // captures of the warm-up.
    if !meta {
        instrument(launch, model_runner.devices())?;
    }
    warm_up(&workers)?;

    // The always-on counters, once the ranks are warm: every stream a runner computes on is
    // `device`, and every one the copies of a rank's blocks move on is `copy`. A graph's launch
    // is one record unless the deployment keeps PC sampling on, which refuses that record.
    let streams: Vec<(&'static str, platform::Stream)> = ranks
        .iter()
        .flat_map(|rank| rank.streams.iter().cloned())
        .collect();
    let counters = if streams.is_empty() {
        None
    } else {
        Counters::start(&streams, !launch.engine_config.profiling.graph_pc_sampling)
    };

    let capacities = capacities(&ranks, &open)?;
    let binding = weight_binding(launch, &model_runner, &ranks, &workers)?;
    drop(ranks);

    let head = Arc::new(Mutex::new(head));
    let session: Box<dyn FnOnce() -> Result<()> + Send> = {
        let head = Arc::clone(&head);
        Box::new(move || service::lock(&head).close())
    };
    let mut engine = Engine::new(
        launch,
        &open,
        &opened,
        pairs,
        capacities,
        policy,
        |_, groups, inbox| {
            let shadow = Arc::new(Mutex::new(Shadow::default()));
            let groups =
                WorkerExecutor::new(Arc::clone(&route), groups, inbox, Arc::clone(&shadow))?;
            Ok((Box::new(groups) as Box<dyn Executor>, shadow))
        },
        session,
    )?;

    // The schedulers grow their device regions, the KV regions into the granted segments: no unit
    // entered a region before its memory existed. A start that fails ends its process, whose
    // session the Storage Server closes once the process has exited (`specs/buffer-pool.md`
    // §14.1).
    epoch::grow_regions(&engine.inboxes(), &head)?;

    engine.workers = workers;
    engine.groups = Some(route);
    engine.binding = binding;
    engine.counters = counters;
    engine.head = Some(head);
    engine.build = build.map(|client| (client, targets));
    if !remotes.is_empty() {
        engine.liveness = Some(liveness);
    }
    engine.remotes = remotes.into_iter().map(|(_, remote)| remote).collect();
    Ok(engine)
}

/// `configured`, the generation limits a launch states, lowered to what the worker groups of
/// `capacities`, whose captured buckets bound `ceilings`, were sized for: the limits the engine
/// checks requests against and reports to its daemon.
///
/// An image encoding of a kind no group computes (`VisionEncode`, `LatentEncode`) has no
/// capacity: its grid tokens and feature bytes are zero, and so are the encoder cache's entries
/// when no group encodes at all. An encoding a group computes is written whole by one prefill
/// into the cache, so its grid takes at most the tokens a step's budget holds, and its features
/// the bytes of that many tokens of the widest row a group of that kind writes
/// (`Capacity::feature_bytes`).
fn lowered(
    configured: GenerationLimits,
    capacities: &[(GroupId, Capacity)],
    ceilings: &Limits,
) -> GenerationLimits {
    let mut limits = configured;
    let tokens = ceilings.max_num_batched_tokens;

    // The grid tokens and the feature bytes a group of the kind that computes `phase` holds;
    // nothing when no group does.
    let held = |phase: Phase, grid: u32, bytes: u64| -> (u32, u64) {
        let rows = capacities
            .iter()
            .map(|(_, capacity)| capacity)
            .filter(|capacity| capacity.serves(phase))
            .map(|capacity| u64::from(capacity.feature_bytes))
            .max();
        match rows {
            Some(row) => {
                let grid = grid.min(tokens);
                (grid, bytes.min(u64::from(grid).saturating_mul(row)))
            }
            None => (0, 0),
        }
    };
    (limits.max_vit_grid_tokens, limits.max_vision_feature_bytes) = held(
        Phase::VisionEncode,
        limits.max_vit_grid_tokens,
        limits.max_vision_feature_bytes,
    );
    (limits.max_vae_grid_tokens, limits.max_latent_feature_bytes) = held(
        Phase::LatentEncode,
        limits.max_vae_grid_tokens,
        limits.max_latent_feature_bytes,
    );

    let encodes = capacities.iter().any(|(_, capacity)| {
        capacity.serves(Phase::VisionEncode) || capacity.serves(Phase::LatentEncode)
    });
    if !encodes {
        limits.encoder_cache_entries = 0;
    }
    limits
}

/// Captures every bucket of every worker's runners, all at once, and waits for each.
///
/// # Errors
///
/// The first worker whose warm-up failed or was cancelled.
fn warm_up(workers: &[Arc<Worker>]) -> Result<()> {
    let submissions = workers
        .iter()
        .map(|worker| worker.warm_up())
        .collect::<crate::worker::Result<Vec<_>>>()?;

    for (index, submission) in submissions.into_iter().enumerate() {
        match submission.wait() {
            Outcome::Success(_) => {}
            Outcome::Failed(fault) => bail!("rank {index}'s warm-up failed: {fault}"),
            Outcome::Cancelled => bail!("rank {index}'s warm-up was cancelled"),
        }
    }
    Ok(())
}

/// The weight binding over `ranks`, each served by its worker of `workers`: what the engine
/// changes of the ranks' storages, and how it loads the model the launch serves again.
fn weight_binding(
    launch: &EngineArgs,
    model_runner: &ModelRunner,
    ranks: &[LoadedRank],
    workers: &[Arc<Worker>],
) -> Result<Arc<WeightBinding>> {
    let mut served = Vec::with_capacity(ranks.len());
    for (rank, worker) in ranks.iter().zip(workers) {
        if let Some(rank) = rank.data_rank(worker)? {
            served.push(rank);
        }
    }

    let target = LoadTarget {
        checkpoint: PathBuf::from(&launch.engine_config.model.model),
        experts: model_runner.experts(),
    };
    Ok(Arc::new(WeightBinding::new(served, target)))
}

/// Installs what the deployment's always-on profiling captures into every graph of `devices`:
/// the range markers of the classes `profiling_config.fence_components` names, and PC sampling
/// when `graph_pc_sampling` keeps it on, which CUPTI attributes to a graph only when enabled
/// before its capture.
///
/// # Errors
///
/// When a device cannot load the markers' kernels or enable sampling.
fn instrument(launch: &EngineArgs, devices: &[DeviceId]) -> Result<()> {
    let profiling = &launch.engine_config.profiling;
    if !profiling.fence_components.is_empty() {
        crate::profiling::markers::Markers::install(
            devices,
            profiling.fence_components.clone(),
            profiling.fence_probes,
            &profiling.fence_probes_off,
        )
        .context("installing the range markers")?;
    }

    if profiling.graph_pc_sampling {
        for device in devices {
            let context = platform::PrimaryContext::retain(*device)?;
            crate::profiling::pc_sampling::enable(&context)
                .context("enabling PC sampling for this rank's graphs")?;
        }
    }
    Ok(())
}

/// The session's numbers of the worker groups `groups` (numbered as `numbers`) that hold a rank on
/// the host of the deployment's rank `rank`: the groups a peer link whose ranks start at `rank`
/// serves ranks of.
///
/// # Errors
///
/// A rank the deployment's groups do not place.
fn held_groups(
    groups: &[WorkerGroupConfig],
    numbers: &[GroupId],
    rank: u32,
) -> Result<Vec<GroupId>> {
    let (group, position) = WorkerGroupConfig::place_rank(groups, rank as usize)
        .with_context(|| format!("the deployment's worker groups place no rank {rank}"))?;
    let host = &groups[group].ranks[position].host;

    Ok(groups
        .iter()
        .zip(numbers)
        .filter(|(group, _)| group.ranks.iter().any(|placed| placed.host == *host))
        .map(|(_, number)| *number)
        .collect())
}

/// The channels to the epoch's other main processes, as [`meet`] opened them.
enum Links {
    /// This process holds every rank of its group.
    Alone,
    /// The scheduler node's: every peer's registered channel, by the first rank it serves.
    Accepted(Vec<(u32, service::Connection)>),
    /// A peer's: its channel to the scheduler node.
    Registered(service::Connection),
}

/// Meets the epoch's other main processes the launch names: the scheduler node accepts every
/// peer's registration on the listener it inherited, and a peer dials it and registers. Returns
/// where the group's ranks form their host group, and the channels.
///
/// # Errors
///
/// Fails when a peer does not register in time, or registers for another epoch or build.
fn meet(launch: &EngineArgs) -> Result<(Meeting, Links)> {
    let Some(world) = &launch.world else {
        return Ok((Meeting::Local, Links::Alone));
    };
    match world.inherited {
        Some(inherited) => {
            // SAFETY: the daemon reserved both sockets for this process and its supervisor
            // placed them at these descriptors (`Spawn::inherit`); nothing else in the process
            // owns them.
            let (channel, store) = unsafe {
                (
                    <std::net::TcpListener as std::os::fd::FromRawFd>::from_raw_fd(
                        inherited.channel,
                    ),
                    <std::net::TcpListener as std::os::fd::FromRawFd>::from_raw_fd(inherited.store),
                )
            };
            let listener = crate::executor::peers::link::Listener::from_tcp(channel)
                .context("taking the peer listener this process inherited")?;
            tracing::info!(
                peers = world.peers,
                channel = %world.channel,
                "awaiting the epoch's peer main processes"
            );
            let accepted = crate::executor::peers::accept(
                &listener,
                world.peers as usize,
                launch.engine,
                PEER_START,
            )?;
            Ok((
                Meeting::World {
                    address: world.store.clone(),
                    listener: Some(store),
                },
                Links::Accepted(accepted),
            ))
        }
        None => {
            let channel = crate::executor::peers::register(
                &world.channel,
                launch.rank,
                world.scheduler,
                PEER_START,
            )?;
            tracing::info!(
                rank = launch.rank,
                scheduler = %world.channel,
                "registered with the epoch's scheduler node"
            );
            Ok((
                Meeting::World {
                    address: world.store.clone(),
                    listener: None,
                },
                Links::Registered(channel),
            ))
        }
    }
}

#[cfg(test)]
mod tests;
