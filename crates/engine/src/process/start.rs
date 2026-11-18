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
//! `specs/buffer-pool.md` §14.1), in this order:
//!
//! 1. the ranks load to the division of their GPUs' quotas, measuring their processes before any
//!    warm-up, and resolve the kernels their profile runs name into the generation they serve;
//! 2. the head opens the session stating each region's units with no segment;
//! 3. every rank maps the segments the answer states and creates the segments of its KV region
//!    its GPU's quota holds free now, beside its process measured before the warm-up, and its
//!    generation takes the kernels its cache and recurrent state launch over the region's
//!    numbering;
//! 4. the workers are spawned and warm, their growth lying outside the quotas meanwhile, and the
//!    head grants the created segments;
//! 5. what fills each share last is settled from the warm processes, measured, never estimated
//!    (`specs/engine-worker-port.md` §1, "The process a division counts"): each media group keeps
//!    the request slots whose storages fit beside its ranks' warm processes and gives the rest
//!    back, the ranks of each KV region give back the tail segments their quotas no longer hold
//!    beside their warm processes, every rank reserves its warm process in its GPU's quota, and a
//!    region beside a draining predecessor creates what its quota then holds, which the head
//!    grants; a stated capacity (`kv_cache_config.max_tokens`) is created whole before the
//!    warm-up and refused by name when it does not fit beside the warm processes;
//! 6. the schedulers are built over the workers, sizing no region: each region keeps the units
//!    the buffer pool holds for it until the engine is promoted
//!    ([`crate::process::epoch::promote`]), when its scheduler grows the KV regions into the
//!    granted segments, resizes the regions of a media group to the slots it kept, and resizes the
//!    regions a target of the same layout shares with the serving epoch to the target's sizes.
//!
//! A standby of the serving engine maps its epoch's regions and reserves its process when it is
//! promoted, as the process it replaces held them until then: it commits nothing at its start. A
//! start is not an update (`specs/updates.md` §2): the generation it serves is
//! [`FIRST_GENERATION`], with nothing kept beside it.
//!
//! An epoch whose worker groups have ranks on several hosts runs one main process per host. The
//! scheduler node's opens the session at its host's Storage Server and relays it to every peer,
//! whose ranks open over it while its own do; once every rank is warm, it grants the segments
//! every rank of a region created, places each peer rank's units at its buffer pool
//! (`PlaceRemote`), and schedules every group within what all of its ranks state. A peer opens the
//! session's host regions at its own host's Storage Server, where its ranks take their parts, and
//! serves the scheduler node's jobs and operations from then on (`specs/buffer-pool.md` §15).
//!
//! [`Engine::close`] ends the epoch in the order its parts depend on one another: the schedulers,
//! then the workers, the peer links, the session, and the devices' contexts.

use std::collections::{BTreeMap, BTreeSet};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::batch::Phase;
use crate::batch::{Capacity, Generations};
use crate::executor::Route;
use crate::executor::peers::messages::{FromPeer, MainProcessResponse};
use crate::executor::peers::{Answer, Liveness, Remote, Upstream};
use crate::profiling::counters::Counters;
use crate::worker::Worker;
use crate::worker::completion::Outcome;
use anyhow::{Context as _, Result, bail};
use arc_swap::ArcSwap;
use build_server::Target;
use crossbeam_channel::{Receiver, Sender};
use ops::KernelEntry;
use platform::DeviceId;
use proto::config::{
    DeviceKind, EngineArgs, EngineConfig, SchedulerConfig, Tuning, WorkerGroupConfig,
};
use proto::request::{GenerationFeatures, GenerationLimits};
use scheduling::limits::Parameters;
use storage::QueuePair;
use storage::runtime::{GroupId, Open, Opened, Segments};

use super::ranks::{
    Backing, FIRST_GENERATION, LoadedRank, Meeting, Place, Resolve, bind, capacities,
    group_numbers, keep_warm_slots, kept_regions, layout, load_ranks, open_ranks, spawn,
};
use super::{epoch, operations};
use crate::executor::{Executor, Shadow, WorkerExecutor};
use crate::memory::client::{StorageClient, StorageLaunch};
use crate::modules::load;
use crate::parallel::Rendezvous;
use crate::process::binding::{Target as LoadTarget, WeightBinding};
use crate::process::ranks::rank::ModelRunner;
use crate::process::ranks::rank::text::Keys;
use crate::scheduler::policy::{self, Ceilings};
use crate::scheduler::{Message, Output, Scheduler};

/// The started epoch: its schedulers and the ranks they drive (module documentation).
pub struct Engine {
    /// One per data-parallel rank, in the order of the session's schedulers.
    pub(crate) schedulers: Vec<SchedulerThread>,
    /// The generation limits requests are checked against, as the schedulers took them.
    pub(crate) limits: GenerationLimits,
    /// What each worker group of the session serves, by its number.
    pub(crate) capacities: Vec<(GroupId, Capacity)>,
    /// What every group's captured buckets bound a step to, and the bounds a policy of this
    /// engine declares its limits within.
    pub(crate) ceilings: Ceilings,
    /// The step values every scheduler reads from its next step on, which a `Tune` replaces.
    pub(crate) tuning: Arc<ArcSwap<Tuning>>,
    /// The requests every scheduler runs at once as it started: the deployment's
    /// `max_running_requests`, within the slots the groups' ranks hold.
    pub(crate) max_running_requests: usize,
    /// The generation batches are assembled for, which the schedulers read and a switch moves.
    pub(crate) served: Arc<Generations>,
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
    /// The kernel packages the first generation serves, by slot: what the ranks' selections of
    /// their profiled keys name together (`crate::modules::select::served`); none for ranks that
    /// resolve no kernel, as on the meta device.
    pub(crate) kernels: proto::module::Packages,
    /// On the scheduler node, the links to the epoch's other main processes.
    pub(crate) remotes: Vec<Arc<Remote>>,
    /// On a peer, its link to the scheduler node.
    pub(crate) upstream: Option<Upstream>,
    /// On a peer, where the end of its own preparation of each generation is told, with the
    /// operators it has no comparison for or its failure: the scheduler node's preparation waits
    /// for it before its shadow pass reaches this process's ranks (`DataOperation::Prepared`).
    pub(crate) prepared: Option<Sender<operations::Preparation>>,
    /// Whether every peer link still serves, which ends the process when one is lost.
    pub(crate) liveness: Option<Arc<Liveness>>,
    /// CUPTI's always-on counters of the ranks' streams and of the threads that wait for the
    /// devices, for as long as the engine serves; `None` without devices, or when the process is
    /// not subscribed to CUPTI.
    pub(crate) counters: Option<Counters>,
    /// The video denoiser of a denoising rank of this process as its family describes it to
    /// request processing (`nn::family::Description::details`), which the daemon binds the video
    /// processor with; empty for an engine that generates no video.
    pub(crate) video: Vec<u8>,
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
    /// `policy` the module set's policy module and its parameters, which every scheduler
    /// instantiates; `session` what ends the session.
    ///
    /// Each scheduler's policy is instantiated on the scheduler's own state, so the replicas and
    /// the phases its groups serve are what it starts from. The generation batches are assembled
    /// for is [`FIRST_GENERATION`]. The ranks' side is empty: no worker, a weight binding of no
    /// rank, no head, Build Server, kernel package or peer link.
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
        let budget = &config.scheduler.policy_budget;
        let ceilings = policy::ceilings(capacities.iter().map(|(_, capacity)| capacity), budget);
        let mut limits = lowered(
            GenerationLimits {
                max_logprobs: config.max_logprobs,
                ..config.model.generation_limits.clone()
            },
            &capacities,
            &ceilings,
        );
        let served = Arc::new(Generations::new(FIRST_GENERATION, FIRST_GENERATION));
        // The schedulers' first step values are the launch's, of the revision it states.
        let tuning = Arc::new(ArcSwap::from_pointee(Tuning::of(config, launch.revision)));

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
            step_budget(config, &ceilings),
            tiers,
        );

        let mut schedulers = Vec::with_capacity(open.schedulers.len());
        let mut features = GenerationFeatures::empty();
        let mut max_running_requests = scheduling.max_running_requests;
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
                Arc::clone(&tuning),
            )
            .with_context(|| format!("starting scheduler {index}"))?;
            features |= scheduler.limits().features;
            max_running_requests = max_running_requests.min(scheduler.max_running_requests());

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

        Ok(Self {
            schedulers,
            limits,
            capacities,
            ceilings,
            tuning,
            max_running_requests,
            served,
            session,
            workers: Vec::new(),
            groups: None,
            binding: Arc::new(WeightBinding::new(
                Vec::new(),
                LoadTarget::default(),
                Vec::new(),
            )),
            head: None,
            build: None,
            kernels: proto::module::Packages::new(),
            remotes: Vec::new(),
            upstream: None,
            prepared: None,
            liveness: None,
            counters: None,
            video: Vec::new(),
        })
    }

    /// Ends the epoch in the order its parts depend on one another:
    ///
    /// 1. every scheduler is told to shut down, which ends every request it holds as aborted
    ///    and stops it once the copies its workers hold and its stores to the host have ended,
    ///    the workers running them meanwhile; each output it sends until its channel closes goes
    ///    to `forward` with the scheduler's index, the events of those requests among them, and
    ///    its thread is joined;
    /// 2. the workers' queues close, ending the job in flight and cancelling the rest, and
    ///    their threads are joined;
    /// 3. the peer links close, which the epoch's other main processes read as this one's end,
    ///    and a peer's thread that runs the scheduler node's operations ends with its link;
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
            prepared,
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
        for remote in &remotes {
            remote.close();
        }
        if let Some(upstream) = &upstream {
            upstream.close();
        }
        drop((remotes, upstream, liveness, build, prepared));

        // The weight binding holds the owners of every rank's device storage; a peer's
        // operations thread held it until its link closed.
        let released = Arc::into_inner(binding).is_some();

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
/// policy refuses; a rank on a device with no Build Server to resolve its kernels; a peer that
/// failed its start or whose link was lost; and segments the ranks of one region created
/// differently, here and on peers.
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
    let meta = launch.engine_config.device.device == DeviceKind::Meta;

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
    let (remotes, upstream) = match links {
        Links::Alone => (Vec::new(), None),
        Links::Accepted(channels) => {
            let mut remotes = Vec::with_capacity(channels.len());
            for (rank, channel) in channels {
                let held = held_groups(&config.worker_groups, &numbers, rank)?;
                remotes.push((held, Remote::spawn(rank, channel, Arc::clone(&liveness))?));
            }
            (remotes, None)
        }
        Links::Registered(channel) => (Vec::new(), Some(channel)),
    };

    // Step 9 of each rank: the kernels its profile run names, selected for the devices'
    // architectures and installed on its own device as the start's generation, and once the
    // session numbered the rank's cache the kernels its cache launches over that numbering. Each
    // selection is kept: together they are what the first generation serves, which an update is
    // compared with, and not the module set's candidates, which name every package installed.
    let selections: Mutex<Vec<proto::module::Packages>> = Mutex::new(Vec::new());
    let targets = if meta {
        Vec::new()
    } else {
        model_runner.targets()?
    };
    let build = build.map(|client| (client, targets));
    let resolve = |device: DeviceId, keys: &Keys| -> Result<Vec<KernelEntry>> {
        let (build, targets) = build
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
            load::select(&mut client, targets, &launch.modules.kernels, &by_operator)?;
        let entries = load::install(&mut client, device, &packages, &routes)?;
        service::lock(&selections).push(packages);
        Ok(entries)
    };
    let backing = if meta {
        // The ranks are placed on this host's CUDA devices whatever they load onto: a member of
        // an expert union shapes its exchange for the multiprocessors the driver counts on its
        // device, as a serving start of the launch does.
        Backing::Meta {
            multiprocessors: DeviceId::multiprocessors,
        }
    } else {
        Backing::Serving {
            storage,
            kernels: &resolve,
        }
    };
    let mut ranks = load_ranks(model_runner.ranks()?, backing, launch, meeting)?;
    let resolved = (!meta).then_some(&resolve as &Resolve<'_>);
    let served = || crate::modules::select::served(service::lock(&selections).iter());
    if let Some(channel) = upstream {
        let mut engine = start_peer(
            launch,
            storage,
            (&model_runner, ranks),
            (build.clone(), resolved),
            channel,
            liveness,
        )?;
        engine.kernels = served();
        return Ok(engine);
    }

    // The session, opened before the ranks map memory: its device KV regions state no segment.
    // From the answer on, the layout states each region numbered as the buffer pool holds it: a
    // region an earlier epoch named keeps that epoch's numbering. The session is this engine's,
    // named by its launch generation: a standby opens its own beside the serving engine's, of the
    // same epoch, and maps the regions both name.
    let mut head = StorageClient::connect_head(
        Path::new(&storage.socket),
        &launch.deployment,
        launch.epoch,
        launch.engine,
    )?;
    let pairs = head.open(layout(launch, &ranks)?)?;
    let (open, opened) = head
        .session()
        .map(|(open, opened)| (open.clone(), opened.clone()))
        .context("the session with the buffer pool is not open")?;

    // Every rank, here and on every peer, maps the segments the answer states and creates the
    // first ones the warm-up reads, which the head grants once every rank holds them.
    let answers = remotes
        .iter()
        .map(|(_, remote)| Ok((remote.rank(), remote.session(&open, &opened)?)))
        .collect::<Result<Vec<(u32, Answer)>>>()?;
    let mut created = open_ranks(&mut ranks, &open, &opened, resolved)?;
    let kernels = served();
    bind(&mut ranks, &open, &opened, true)?;

    // Range markers, PC sampling and the tracing mode of a graph's launch attach to what a graph
    // captures and instantiates, so they precede the warm-up's captures, whose executables are
    // current at the tracing mode's renewal count of their instantiation.
    if !meta {
        instrument(launch, &model_runner, build.as_ref())?;
    }
    let registers = |group: GroupId| head.completed_registers(group, &pairs);
    let (route, workers) = spawn(&mut ranks, (&open, &opened), &registers, remotes.clone())?;

    // A peer's ranks warm up at once with these: the captures of a group's ranks reach its
    // collectives together.
    warm_up(&workers)?;

    // What every peer's ranks hold, once they are warm: their segments are granted with these
    // ranks', and where their units lie is stated at this host's buffer pool for them.
    let mut stated_by_peers = Vec::new();
    for (rank, answer) in answers {
        let (endpoint, theirs, places, capacities) = placed(rank, &answer)?;
        for (region, segments) in theirs {
            agree(&mut created, region, segments)?;
        }
        for (region, memory, part, runs) in places {
            head.place_remote(&region, &memory, part, &endpoint, &runs)?;
        }
        stated_by_peers.extend(capacities);
    }
    for (region, segments) in &created {
        for &(segment, first, units) in segments {
            head.grant(region, segment, first, units)?;
        }
    }

    // What fills each share last, settled from the warm processes: the slots a media group
    // keeps, the segments of each KV region the quotas hold beside the warm processes, each
    // rank's warm process in its GPU's quota, and what a predecessor's drain freed meanwhile,
    // here and on every peer. A standby commits nothing until it is promoted. A peer kept its
    // media groups' slots before it answered.
    let binding = weight_binding(launch, &model_runner, &ranks, &workers, remotes.clone())?;
    if !standby(launch) {
        keep_warm_slots(&mut ranks)?;
        epoch::keep_warm_segments(&mut head, &binding)?;
        binding
            .reserve_serving()
            .context("reserving the warm processes in their GPUs' quotas")?;
        let capacity = launch
            .engine_config
            .kv_cache
            .max_tokens
            .map(NonZeroU64::get);
        epoch::create_fitting(&mut head, &binding, capacity, None)?;
    }

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

    let peers = binding.served_by_peers()?;
    let capacities = capacities(&ranks, &open, (&peers, stated_by_peers))?;

    // The session's regions of each media group serve the slots it kept, as every rank of it
    // states them; the schedulers resize them so once the engine is promoted.
    for (region, units) in kept_regions(launch, &ranks, &open, &capacities)? {
        head.keep_units(region, units)?;
    }
    let (open, opened) = head
        .session()
        .map(|(open, opened)| (open.clone(), opened.clone()))
        .context("the session with the buffer pool is not open")?;
    let video = ranks
        .iter()
        .filter_map(|rank| rank.video().ok())
        .map(|video| &video.description)
        .find(|description| !description.is_empty())
        .cloned()
        .unwrap_or_default();
    drop(ranks);

    let head = Arc::new(Mutex::new(head));
    let session: Box<dyn FnOnce() -> Result<()> + Send> = {
        let head = Arc::clone(&head);
        Box::new(move || service::lock(&head).close())
    };

    // The schedulers start over the regions as the buffer pool holds them, and the start resizes
    // none: each takes the units the session states of it once the engine is promoted
    // (`epoch::promote`), the KV regions growing into the granted segments and a media group's
    // regions shrinking to the slots it kept. A target of the serving epoch's layout names the
    // serving engine's regions, which so keep the serving epoch's sizes while the target prepares
    // and after an update abandoned before its switch: the target's process then ends unpromoted,
    // and the Storage Server closes its session once the process has exited
    // (`specs/buffer-pool.md` §14.1).
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

    engine.workers = workers;
    engine.groups = Some(route);
    engine.binding = binding;
    engine.counters = counters;
    engine.head = Some(head);
    engine.kernels = kernels;
    engine.video = video;
    engine.build = build;
    if !remotes.is_empty() {
        engine.liveness = Some(liveness);
    }
    engine.remotes = remotes.into_iter().map(|(_, remote)| remote).collect();
    Ok(engine)
}

/// Where a peer's ranks placed their units, as its answer to the relayed session states it
/// ([`MainProcessResponse::Placed`]).
type Placed = (
    String,
    Vec<(String, Segments)>,
    Vec<Place>,
    Vec<(GroupId, Capacity)>,
);

/// The answer of the peer whose ranks start at `rank` to the relayed session: its Storage
/// Server's endpoint, the segments its ranks created, where their units lie, and what they serve.
/// The peer answers once its ranks are open, bound, spawned and warm; its link fails the wait if
/// the peer's process exits or its host stops answering meanwhile.
///
/// # Errors
///
/// When the peer fails its start, or its link is lost.
fn placed(rank: u32, answer: &Answer) -> Result<Placed> {
    let answered = answer
        .recv()
        .map_err(|_| anyhow::anyhow!("rank {rank}'s main process link ended before it answered"))?
        .map_err(|reason| {
            anyhow::anyhow!("rank {rank}'s main process failed to start: {reason}")
        })?;
    match answered {
        MainProcessResponse::Placed {
            endpoint,
            created,
            places,
            capacities,
        } => Ok((endpoint, created, places, capacities)),
        other => bail!(
            "rank {rank}'s main process answered the session with {:?}",
            other.kind()
        ),
    }
}

/// Adds `segments`, those the ranks of another process created of `region`, to `created`, those
/// of every region by name: every rank that maps a region creates the same segments of it.
///
/// # Errors
///
/// Ranks of the region that created different segments.
fn agree(created: &mut Vec<(String, Segments)>, region: String, segments: Segments) -> Result<()> {
    match created.iter().find(|(name, _)| *name == region) {
        Some((_, ours)) if *ours != segments => bail!(
            "the ranks of region {region} created different segments here and on a peer: \
             {ours:?} and {segments:?}"
        ),
        Some(_) => Ok(()),
        None => {
            created.push((region, segments));
            Ok(())
        }
    }
}

/// Starts this process as a peer of its epoch's scheduler node, over `channel`, its registered
/// connection, once its ranks (`ranks`, of `model_runner`) loaded (module documentation): it
/// reads the session the scheduler node relays, opens the session's host regions at its own
/// host's Storage Server (`storage`), opens, binds, spawns and warms its ranks over the relayed
/// session, resolving through `resolve` the kernels of their caches over the session's
/// numbering, answers with what they hold, and serves the scheduler node's jobs and operations,
/// among them the reservation of its ranks' warm processes and the segments their quotas then
/// hold, which the scheduler node's start asks for once every rank is warm.
///
/// The engine of a peer holds no scheduler and no session head: the scheduler node's schedulers
/// drive its ranks, and its `session` closes the host regions' session at its host.
///
/// # Errors
///
/// As [`start`]'s steps, or a session that does not arrive.
#[expect(
    clippy::type_complexity,
    reason = "the Build Server's client with the devices' targets, and the resolution of the \
              ranks' kernels through it"
)]
fn start_peer(
    launch: &EngineArgs,
    storage: &StorageLaunch,
    (model_runner, mut ranks): (&ModelRunner, Vec<LoadedRank>),
    (build, resolve): (
        Option<(Arc<Mutex<build_server::Client>>, Vec<Target>)>,
        Option<&Resolve<'_>>,
    ),
    mut channel: service::Connection,
    liveness: Arc<Liveness>,
) -> Result<Engine> {
    let (id, open, opened) = crate::executor::peers::session(&mut channel)?;

    // The parts of the session's host regions lie on this host, created by its Storage Server,
    // which follows the epoch's processes for the session's close as for any session.
    let mut hosts = StorageClient::connect_head(
        Path::new(&storage.socket),
        &launch.deployment,
        launch.epoch,
        launch.engine,
    )?;
    hosts.open_hosts(&open)?;

    let created = open_ranks(&mut ranks, &open, &opened, resolve)?;
    let places = bind(&mut ranks, &open, &opened, false)?;

    // What attaches to the graphs precedes the workers and the warm-up, as on the scheduler node.
    if launch.engine_config.device.device != DeviceKind::Meta {
        instrument(launch, model_runner, build.as_ref())?;
    }
    // A peer holds no queue pair: its ended sequences reach the scheduler node over its link.
    let (route, workers) = spawn(
        &mut ranks,
        (&open, &opened),
        &|_| Ok(Vec::new()),
        Vec::new(),
    )?;
    warm_up(&workers)?;

    // The slots its media groups keep beside their warm processes, which its answer states; the
    // scheduler node has its ranks reserve their warm processes and create the segments their
    // quotas then hold once every rank answered.
    if !standby(launch) {
        keep_warm_slots(&mut ranks)?;
    }

    let streams: Vec<(&'static str, platform::Stream)> = ranks
        .iter()
        .flat_map(|rank| rank.streams.iter().cloned())
        .collect();
    let counters = if streams.is_empty() {
        None
    } else {
        Counters::start(&streams, !launch.engine_config.profiling.graph_pc_sampling)
    };

    let capacities = capacities(&ranks, &open, (&[], Vec::new()))?;
    let binding = weight_binding(launch, model_runner, &ranks, &workers, Vec::new())?;
    drop(ranks);

    let placed = MainProcessResponse::Placed {
        endpoint: storage.tcp.clone().unwrap_or_default(),
        created,
        places,
        capacities: capacities.clone(),
    };
    channel
        .deliver(&FromPeer {
            id,
            response: placed,
        })
        .context("answering the scheduler node's session")?;

    let (prepared, preparations) = crossbeam_channel::unbounded();
    // A peer waits for its own preparation of a generation within the epoch's comparison bound,
    // as the scheduler node's shadow pass does.
    let bound = launch.engine_config.worker.compare_timeout;
    let operate = operations::operator(Arc::clone(&binding), workers.clone(), preparations, bound);
    let upstream = Upstream::serve(channel, Arc::clone(&route), Arc::clone(&liveness), operate)?;

    let hosts = Mutex::new(hosts);
    let session: Box<dyn FnOnce() -> Result<()> + Send> =
        Box::new(move || service::lock(&hosts).close());
    let ceilings = policy::ceilings(
        capacities.iter().map(|(_, capacity)| capacity),
        &launch.engine_config.scheduler.policy_budget,
    );
    Ok(Engine {
        schedulers: Vec::new(),
        limits: launch.engine_config.model.generation_limits.clone(),
        capacities,
        ceilings,
        // A peer runs no scheduler: its step values are the launch's and go nowhere.
        tuning: Arc::new(ArcSwap::from_pointee(Tuning::of(
            &launch.engine_config,
            launch.revision,
        ))),
        // A peer runs no scheduler: the scheduler node serves the deployment's requests within
        // its ranks' slots, and the peer states the launch's bound.
        max_running_requests: launch.engine_config.max_running_requests.get(),
        served: Arc::new(Generations::new(FIRST_GENERATION, FIRST_GENERATION)),
        session,
        workers,
        groups: Some(route),
        binding,
        head: None,
        build,
        // The caller states what the peer's ranks selected.
        kernels: proto::module::Packages::new(),
        video: Vec::new(),
        remotes: Vec::new(),
        upstream: Some(upstream),
        prepared: Some(prepared),
        liveness: Some(liveness),
        counters,
    })
}

/// The tokens one scheduling step holds (`SchedulerConfig::max_num_batched_tokens`, SD §7.1): the
/// deployment's `max_num_batched_tokens`; when it states none, what the workers' captured catalog
/// bounds a step to, the widest prefill bucket of `ceilings`; and the longest sequence only when
/// no group captured a prefill bucket. A scheduler budgets each step by the step values it
/// reads, and by this figure where they state no token budget (`scheduler::Budget`).
pub(crate) fn step_budget(config: &EngineConfig, ceilings: &Ceilings) -> usize {
    // A deployment that captured no prefill bucket (a media pipeline) admits its longest
    // sequence in one step.
    let captured = match ceilings.tokens {
        u32::MAX => usize::try_from(config.model.max_model_len).unwrap_or(usize::MAX),
        tokens => usize::try_from(tokens).unwrap_or(usize::MAX),
    };
    config
        .max_num_batched_tokens
        .map_or(captured, NonZeroUsize::get)
        .max(1)
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
    ceilings: &Ceilings,
) -> GenerationLimits {
    let mut limits = configured;
    let tokens = ceilings.tokens;

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

/// Whether `launch` starts a standby of its epoch's serving engine: it succeeds an engine of its
/// own epoch (a switch's target succeeds another epoch's), whose regions it maps and whose
/// reservations in the GPUs' quotas the Storage Server keeps for the process that replaces it.
/// It reserves its warm process and grows its regions once it is promoted.
pub(super) fn standby(launch: &EngineArgs) -> bool {
    launch
        .predecessor
        .as_ref()
        .is_some_and(|predecessor| predecessor.epoch == launch.epoch)
}

/// Captures every bucket of every worker's runners, each worker's on its capture thread, all
/// workers at once, and waits for each: a capture runs nothing, and a rehearsal meets no other
/// rank, so no worker's warm-up waits for another's.
///
/// The start creates its storage before the warm-up, as much of each GPU's quota as holds it, so
/// what the warm-up adds to a process (instantiated graphs, the libraries' state) lies in the
/// device's memory outside the quota until the start gives back what no longer fits beside the
/// warm process (`epoch::keep_warm_segments`). Where other deployments' storage or other
/// processes hold that memory, a warm-up allocation fails, and so does the start: the failure
/// names the bytes the device had free as the warm-up began.
///
/// # Errors
///
/// The first worker whose warm-up failed or was cancelled.
fn warm_up(workers: &[Arc<Worker>]) -> Result<()> {
    // What each worker's device has free as the warm-up begins: the room its growth has.
    let free: Vec<Option<(i32, u64)>> = workers
        .iter()
        .map(|worker| {
            let device = worker.device()?;
            let free = device.memory_info().ok()?.free;
            Some((device.0, free))
        })
        .collect();
    let submissions: Vec<_> = workers.iter().map(|worker| worker.warm_up()).collect();

    for (index, submission) in submissions.into_iter().enumerate() {
        match submission?.wait() {
            Outcome::Success(_) => {}
            Outcome::Failed(fault) => match free.get(index).copied().flatten() {
                Some((device, free)) => bail!(
                    "rank {index}'s warm-up failed: {fault}; the warm-up grows the process beside \
                     the storage the deployment's quota holds, into device {device}'s memory \
                     outside the quota, of which {free} bytes were free as it began"
                ),
                None => bail!("rank {index}'s warm-up failed: {fault}"),
            },
            Outcome::Cancelled => bail!("rank {index}'s warm-up was cancelled"),
        }
    }
    Ok(())
}

/// The weight binding over `ranks`, each served by its worker of `workers`, and the ranks of the
/// peers behind `remotes`: what the engine changes of the ranks' storages, and how it loads the
/// model the launch serves again.
fn weight_binding(
    launch: &EngineArgs,
    model_runner: &ModelRunner,
    ranks: &[LoadedRank],
    workers: &[Arc<Worker>],
    remotes: Vec<(Vec<GroupId>, Arc<Remote>)>,
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
    Ok(Arc::new(WeightBinding::new(served, target, remotes)))
}

/// Installs what the deployment's always-on profiling captures into every graph of the devices
/// `model_runner` serves, after setting the NVTX ranges' switch as `profiling_config.nvtx_ranges`
/// states: the range markers of the classes `profiling_config.fence_modules` names,
/// or of the family's step stack when it names none ([`ModelRunner::step_stack`]), with their
/// kernels resolved through `build`, the Build Server's client and the devices' targets, as every
/// operator's are; and PC sampling when `graph_pc_sampling` keeps it on, which CUPTI attributes
/// to a graph only when enabled before its capture; otherwise the counters' record of each graph
/// launch as a whole, in which the executables are then instantiated
/// ([`crate::profiling::counters::record_whole_graphs`]).
///
/// Called before the ranks' workers start and capture the warm-up's graphs.
///
/// # Errors
///
/// A deployment that places markers with no Build Server client, the Build Server's refusal to
/// select or build the markers' package, or a device that cannot load its kernels or enable
/// sampling.
fn instrument(
    launch: &EngineArgs,
    model_runner: &ModelRunner,
    build: Option<&(Arc<Mutex<build_server::Client>>, Vec<Target>)>,
) -> Result<()> {
    let profiling = &launch.engine_config.profiling;
    crate::profiling::ranges::set_enabled(profiling.nvtx_ranges);
    let devices = model_runner.devices();
    let classes = crate::profiling::markers::classes(profiling, model_runner.step_stack());
    if !classes.is_empty() {
        let (client, targets) =
            build.context("the start holds no Build Server client to resolve the range markers")?;
        let mut client = service::lock(client);
        let (packages, routes) = load::select(
            &mut client,
            targets,
            &launch.modules.kernels,
            &crate::profiling::markers::keys(),
        )?;
        let kernels = devices
            .iter()
            .map(|&device| {
                Ok((
                    device,
                    load::install(&mut client, device, &packages, &routes)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        crate::profiling::markers::Markers::install(
            kernels,
            classes,
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
    } else {
        crate::profiling::counters::record_whole_graphs();
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
/// peer's registration on the listener it inherited, and a peer connects to it and registers,
/// each within the epoch's join bound (`worker_config.join_timeout`). Returns where the group's
/// ranks form their host group, and the channels.
///
/// # Errors
///
/// Fails when a peer does not register in time, or registers for another epoch or build.
fn meet(launch: &EngineArgs) -> Result<(Meeting, Links)> {
    let Some(world) = &launch.world else {
        return Ok((Meeting::Local, Links::Alone));
    };
    let join = launch.engine_config.worker.join_timeout;
    let rendezvous = |listener| {
        Meeting::World(Rendezvous {
            listener,
            endpoint: world.store.clone(),
            epoch: world.scheduler,
            timeout: join,
        })
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
            let listener = service::Listener::from_tcp(channel)
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
                join,
            )?;
            Ok((rendezvous(Some(store)), Links::Accepted(accepted)))
        }
        None => {
            let channel = crate::executor::peers::register(
                &world.channel,
                launch.rank,
                world.scheduler,
                join,
            )?;
            tracing::info!(
                rank = launch.rank,
                scheduler = %world.channel,
                "registered with the epoch's scheduler node"
            );
            Ok((rendezvous(None), Links::Registered(channel)))
        }
    }
}

#[cfg(test)]
mod tests;
