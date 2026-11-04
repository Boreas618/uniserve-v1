//! `uniserve engine`: a node's main process for one epoch (`specs/worker.md`
//! §2.1, §2.3, §2.4, §4).
//!
//! The daemon starts one main process per node and epoch, from the epoch's build
//! config. This module is the command and the object model of §4: it computes
//! nothing itself; it assembles
//!
//! - the [`Control`] thread, which serves the daemon's request interface and the
//!   epoch's peer main processes and never waits for a job (§2.3, §2.4);
//! - one `engine::Scheduler` per DP rank, on the node that holds them;
//! - one `executor::device::Executor` per device, the only thread that issues to
//!   that device's streams;
//! - the node's resident memory ([`Memory`]);
//! - the blocking pool host work runs on ([`ThreadPool`]),
//!
//! and drives the program-layer update of §7.3 between them: precopy on the
//! blocking pool and the executors, shadow execution against the scheduler's
//! resident requests, switchover at a batch boundary, probation, commit or
//! revert, and purge after the old generation's last batch.
//!
//! The one `match` on a model class (§4.3) is here too: this executable links
//! the model crates, and nothing else in the tree names a concrete model.
//!
//! ## What this module still waits for
//!
//! Each of these is one named publisher, and the code is written against the
//! name so the merge is a compile rather than a rewrite:
//!
//! - **sched** (`crates/engine`): `SchedulerHandle::{start(config,
//!   Arc<dyn Batches>), submit, shadow, stage, command, engine_command,
//!   counters, stop}` (M505, M714). `stage(Stage, generation)` applies §7.3
//!   steps 3-6 at a batch boundary and answers the step it applied at, because
//!   an expert group's executors must switch at the same step on every rank.
//!   `schedulers` stays empty until it exists, and a node that holds no
//!   scheduler serves only what its peers ask of it.
//! - **front** (`crates/proto`): the epoch's programs and the serving revision
//!   on the welcome (§5.1, §7.3), the program commands and statuses in
//!   `process.proto` mirroring [`ipc::ProgramCommand`]/[`ipc::ProgramStatus`],
//!   and the per-device counters `FromEngine::Device` that [`Self::send_metrics`]
//!   fills (§2.3).
//! - **builder03** (`crates/executor`): `capacity::plan(&CapacityRequest) ->
//!   Result<Plan>`, whose producer is still `crates/worker/src/worker/planning.rs`
//!   and still entangled with the launch machinery §4 deletes, and the Storage
//!   Server client's publish/map surface that a weight store stands on.
//! - **nobody yet**: the `nn::RetainedWeights` production implementation. Its two
//!   methods (`retained`, `publish`) are all a checkpoint load needs, and the
//!   only implementor in the tree is a test double, so no main process can load
//!   a model until it exists.
//!
//! The driver of §7.3 below does not wait on any of them: it is one function over
//! a command enum (`Self::program`), so a new wire arm is one `match` line.

//! A serving batch's `BatchOutput` is emitted from here too: the daemon's
//! request interface carries `Emitted` events, and the tokens come back with the
//! scheduler's completed batch in logits order.
//!
//! ## The daemon's program-layer sequence
//!
//! A standby main process starts, and the daemon then drives one update per
//! generation through the request interface, at no other entry point:
//!
//! 1. `ToEngine::Precopy { kernels, policy }`: the programs of the new
//!    generation, as revisions. Each node runs [`MainProcess::precopy`] and
//!    answers `FromEngine::Report::Precopied { generation }`, or
//!    `PrecopyFailed { reason }`, which discards the generation and leaves
//!    serving on the old one.
//! 2. `ToEngine::Switchover { generation }`: the node swaps at the next batch
//!    boundary and answers `Switched { generation }`. A main process serves
//!    nothing before its first switchover; the daemon only lets the frontend
//!    submit after it.
//! 3. `Committed { generation }` ends probation; `Reverted { reason }` puts the
//!    previous generation back and keeps the update unavailable until forced;
//!    `Purged { generation }` releases a generation after its last batch.
//!    `PolicyReverted { scheduler, backend, revision, reason, fatal }` is the
//!    policy's own revert, and `Stalled`, `Fault` and `NonFiniteLogits` are
//!    §7.3 step 4's automatic-revert failures.
//!
//! The two `ToEngine` variants and the report variants are front's additions to
//! `crates/proto/src/request.proto`; this module reads and writes them under
//! exactly those names. Nothing program-content crosses that protocol: a node
//! resolves its modules from the Program Server itself.
//!
//! The executor's own calls (`Executor::{spawn, submit, precopy, switchover,
//! purge, watchdog, probe, close}`, `Submission::wait`) are published in
//! `crates/executor` and are used as they are written there.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context as _, Result};
use clap::Args;

use executor::batch::Submission;
use executor::blocking::{BlockingKind, ThreadPool};
use executor::completion::Outcome;
use executor::control::{Control, Endpoints, Ready};
use executor::device::{Executor, Probe};
use executor::host::{HostAction, HostTask, Lane};
use executor::hot_update::{Action, HotUpdate};
use executor::memory::retained::Retained;
use executor::memory::weights::WeightStore;
use executor::memory::Memory;
use executor::watchdog::{Running, Watchdog};
use crate::engine_plan::EpochPlan;
use daemon::epoch::BuildConfig;
use engine::{PolicyProgram, PolicySource, ProgramSource, SchedulerHandle};
use ipc::{ProgramCommand, ProgramStatus, UpdateFailure};
use ops::{Generation, KernelEntry};
use program_server::program::Programs;
use program_server::{Client, Descriptors, Module, PrepareRequest, Program, ProgramRequest, Selector,
    Target};

/// The epoch as this node takes it at start.
///
/// The daemon fixes the epoch with a [`BuildConfig`] (design §7.2 and §7.7) and
/// fills its operators and its scheduling interface with programs (§5.1); the
/// main process never receives program content, only the programs' names, and it
/// resolves their modules from the Program Server itself.
pub(crate) struct Epoch {
    /// The configuration that fixed the epoch.
    pub build: BuildConfig,
    /// The programs that fill the epoch's operators and its scheduling
    /// interface, by slot.
    pub programs: Programs,
    /// The program revision the daemon numbered for the generation that serves
    /// once this node switches over (§7.3).
    pub generation: u64,
    /// The host identity of the node this process runs on, which the daemon knows and which the
    /// Storage Server's claim states.
    pub host: String,
}

/// The options of `uniserve engine`: the launch file the daemon writes for this process (§2.3).
///
/// Everything else the process needs is that file's; the daemon's own account of who this process
/// is, which build config it serves and where it reaches the epoch's interfaces. A flag beside it
/// would be a second account of the same thing, and the two would drift.
#[derive(Debug, Clone, Args)]
pub(crate) struct EngineArgs {
    /// The launch file the daemon wrote for this main process.
    #[arg(long)]
    pub launch: PathBuf,
}

/// One engine process's launch, as the daemon writes it (`notes` of the launch contract).
///
/// The claim token and the Storage Server's socket are not here: they are runtime facts of the node
/// this process runs on, which the daemon answers when the process asks for them, so a launch file
/// cannot carry a claim that has not been minted yet.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Launch {
    /// The epoch's build config, as the daemon wrote it: a path, since the config is the epoch's and
    /// not the process's.
    pub build_config: PathBuf,
    /// The checkpoint the epoch serves.
    pub model: PathBuf,
    /// The leader daemon's request interface, `unix:<path>` or `<host>:<port>`.
    pub requests: String,
    /// The engine number the daemon started this process under.
    pub engine: u64,
    /// The Program Server's socket.
    pub program_server: String,
    /// The devices this rank serves, as CUDA ordinals.
    pub devices: Vec<u32>,
    /// This rank's worker group, which keys its storage claims.
    pub group: String,
    /// This rank's ordinal inside its group.
    pub rank: u32,
    /// The deployment's name, under which the Storage Servers key its claims.
    pub deployment: String,
    /// The world the epoch executes, which keys the storage it creates.
    pub world: u64,
    /// Whether this node holds the epoch's schedulers.
    #[serde(default)]
    pub scheduler_node: bool,
    /// The addresses on which this node accepts its peer main processes' connections.
    #[serde(default)]
    pub peers: Vec<String>,
    /// How long a job may run without progress before it is reported `Stalled`, when the daemon
    /// states it apart from the build config (§2.4).
    #[serde(default)]
    pub watchdog_timeout_s: Option<u64>,
    /// The probation an update must survive before the daemon commits it (§7.3 step 4).
    #[serde(default)]
    pub rollback_s: Option<u64>,
    /// The placement's queue depth for this rank's worker groups: how many batches the rank's
    /// schedulers keep in flight, which the capacity plan relays through (§4.1).
    ///
    /// Required, not defaulted: a zero depth plans a rank with no relay and silently halves the
    /// pipeline's slack.
    pub queue_depth: u32,
    /// The number of ranks in this process's world, of which this group's rank is one.
    ///
    /// Required, not defaulted: a world of one on a multi-node deployment sizes this rank's storage
    /// claim for one rank, and the second node's claim would collide with it.
    pub world_size: u32,
}

impl Launch {
    /// The launch file's contents.
    fn read(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("opening the launch file {}", path.display()))?;
        serde_json::from_reader(file)
            .with_context(|| format!("reading the launch file {}", path.display()))
    }

    /// §2.4's watchdog bound: the launch's when the daemon states one apart from the build config.
    fn watchdog_timeout(&self) -> Option<Duration> {
        self.watchdog_timeout_s.map(Duration::from_secs)
    }
}

/// Asks this node's daemon for the node's Storage Server (§8).
///
/// The socket and the claim are runtime facts of the node: the daemon mints the claim when it starts
/// the server, so a launch file cannot carry either. The answer states the socket the server listens
/// on, the token this rank presents, and where another host reaches its pages.
fn claim(control: &mut Control, launch: &Launch, host: &str) -> Result<ipc::StorageLaunch> {
    let request = proto::process::Request {
        id: 1,
        kind: Some(proto::process::request::Kind::Claim(proto::process::Claim {
            host: host.to_owned(),
            deployment: launch.deployment.clone(),
            scopes: vec![proto::process::Scope {
                group: launch.group.clone(),
                rank: launch.rank,
            }],
            // A server the daemon starts for this claim takes the deployment's connector settings;
            // the main process has none of its own to state.
            server_arguments: Vec::new(),
            world: launch.world,
            retire: Vec::new(),
        })),
    };
    control.daemon_mut().send(&request)?;
    let reply: proto::process::Reply = control
        .daemon_mut()
        .recv(Duration::from_secs(60))?
        .context("the daemon closed the request interface while this process claimed its storage")?;
    if !reply.error.is_empty() {
        bail!("the daemon refused this rank's claim: {}", reply.error);
    }
    let Some(proto::process::reply::Kind::Claimed(claimed)) = reply.kind else {
        bail!("the daemon answered a claim with another reply");
    };
    Ok(ipc::StorageLaunch {
        socket: claimed.socket,
        deployment: launch.deployment.clone(),
        claim: claim_token(&claimed.claim)?,
        // The host KV cache is the deployment's, and a rank whose storage another host reaches is
        // told so by the answer's `tcp`; the rest of the launch is the epoch's.
        host_kv_cache: false,
        tcp: (!claimed.tcp.is_empty()).then_some(claimed.tcp),
        peers: Vec::new(),
        world: launch.world,
        standby: false,
        standby_storage: ipc::StandbyStorage::Auto,
    })
}

/// The 32 hex digits of a claim token, as the Storage Server takes it.
fn claim_token(bytes: &[u8]) -> Result<[u8; 16]> {
    bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("a claim token is 16 bytes, not {}", bytes.len()))
}

/// Runs the `uniserve engine` command.
pub(crate) fn run(
    args: &EngineArgs,
    lifeline: Option<i32>,
    log: &crate::cli::LogLevels,
) -> Result<ExitCode> {
    crate::follow_lifeline(lifeline);
    crate::logging::init_tracing(
        "Engine",
        log.log_level.as_deref(),
        log.log_level_http.as_deref(),
    );

    let launch = Launch::read(&args.launch)?;
    let build: BuildConfig = serde_json::from_reader(
        std::fs::File::open(&launch.build_config)
            .with_context(|| format!("opening {}", launch.build_config.display()))?,
    )
    .with_context(|| {
        format!(
            "reading the build config {}",
            launch.build_config.display()
        )
    })?;
    let mut main = MainProcess::start(&launch, &build)?;
    let ending = main.serve();
    main.close();
    if let Some(failure) = ending? {
        bail!("the main process ended: {}", failure.reason());
    }
    Ok(ExitCode::SUCCESS)
}

/// One node's main process for one epoch: the object model of §4.
///
/// Every field is owned by exactly one thread: the control thread owns itself,
/// each executor thread owns its runners and streams, each scheduler thread its
/// mechanism. The fields below are the handles those threads were handed and the
/// owners that must outlive them.
pub(crate) struct MainProcess {
    control: Control,
    /// This node's schedulers, one per DP rank. The handle is the interface: each scheduler's thread
    /// owns the instance, so every call the control thread makes goes through the handle and the
    /// command channel carries the mutation (§2.4).
    schedulers: Vec<engine::SchedulerHandle>,
    executors: Vec<Executor>,
    memory: Option<Memory>,
    blocking: ThreadPool,
    /// The blocking lane module loads run on (§2.4, §7.3 step 2).
    modules: Lane<Install>,
    /// The program-layer update's phases and bookkeeping (§7.3).
    update: HotUpdate,
    /// The stall and sticky-error rules of §2.4.
    watchdog: Watchdog,
    /// The Program Server client, shared with the module-load lane.
    programs: Arc<Mutex<Client>>,
    /// The precopy between serve passes, if one is in flight (§7.3 step 2).
    precopy: Option<Precopy>,
    /// The generations the last precopy built and the switchover has not taken:
    /// one per device, in `executors` order. A revert drops them, because the
    /// generation they hold never serves.
    prepared: Vec<Arc<Generation>>,
    /// The generations serving before the current ones, kept through probation so
    /// a revert can serve them again (§7.3 step 4). Their graphs stay in the
    /// executors either way; this is the process's own record of what to serve.
    retired: Vec<Arc<Generation>>,
    /// The generation each device serves now, in `executors` order.
    generations: Vec<Arc<Generation>>,
    /// What this node resolves a module for (§5.3).
    targets: Vec<Target>,
    /// The Program Server's socket, which a policy's source resolves its module through.
    program_server: PathBuf,
    /// Set by the daemon's `Shutdown`: the epoch ends at the end of this pass (§2.3).
    stopping: bool,
}

impl MainProcess {
    /// Connects to the daemon, loads the epoch's model, and starts the epoch's
    /// threads in the order §2.4 lists.
    ///
    /// The epoch's configuration is resolved against this process's launch first
    /// ([`EpochPlan`]), because three of the things the main process needs are the
    /// node's rather than the epoch's: which devices it serves, where its Program
    /// Server is, and what those devices measure.
    fn start(launch: &Launch, build: &BuildConfig) -> Result<Self> {
        let plan = EpochPlan::of(launch, build)?;
        let mut control = Control::connect(
            &Endpoints {
                requests: launch.requests.clone(),
                peers: launch.peers.clone(),
            },
            launch.engine,
        )?;
        // The request interface's handshake: this process says who it is, and the
        // daemon answers with the epoch this node serves (§2.3).
        let epoch = handshake(&mut control, launch, build)?;

        // §8: the rank's Storage Server and its claim are the node's, which the daemon answers for;
        // the retained weights outlive every world of the epoch, so the store is built once, before
        // the model loads, and keeps what it maps for the epoch's life.
        let storage = claim(&mut control, launch, &epoch.host)?;
        let client = Retained::connect(&storage, &launch.group, launch.rank, epoch.generation)
            .context("connecting to the Storage Server")?;
        // A weight's region is one device's address, so the store maps on the device this rank's
        // model is loaded for; the client is shared, the mapping is not.
        let weight_device = *plan
            .devices()
            .first()
            .context("this rank serves no device to map its weights on")?;
        let mut weights = WeightStore::new(Arc::clone(&client), weight_device);
        let placement = Placement::of(launch, build)?;
        // One model per rank: every device's runner reads the same weights, so a second device costs
        // one `Arc` and not a second load.
        let model = Arc::new(
            load_model(&launch.model, &placement, &mut weights)
                .context("loading the epoch's model")?,
        );

        // The owners of §4 are opened before any thread that borrows them, and
        // released in reverse order by `close`. The weights are mapped before the capacity plan
        // measures, so the plan sees a device with the model already in it - the figure it computes
        // is what is left, not what was there before the load. Pools sized first and weights after
        // would over-commit the device by exactly the model's size.
        let mut memory = plan.open_memory(&model, &mut &*client)?;
        let blocking = ThreadPool::new(plan.blocking_limits());
        let modules = blocking.lane::<Install>(BlockingKind::ModuleLoad)?;
        let programs = Arc::new(Mutex::new(Client::connect(plan.program_server())?));

        // §2.4's watchdog exists before any device command: `start`'s own commands
        // are bounded by the same stall bound as every command the control thread
        // issues afterwards.
        let watchdog = Watchdog::new(
            launch
                .watchdog_timeout()
                .unwrap_or_else(|| plan.watchdog_timeout()),
        )?;
        let update = HotUpdate::new(epoch.generation);

        let mut executors = Vec::with_capacity(plan.devices().len());
        let mut generations = Vec::with_capacity(plan.devices().len());
        // The device's one pool set is the resident owners' allocator (§4.1): it is opened with the
        // memory and handed to the executor thread, so every owner and every runner books against
        // the same allowance.
        let pools = memory.pools();
        let device = *plan
            .devices()
            .first()
            .context("this rank serves no device")?;
        let scratch = plan.graph(device)?.scratch;
        for device in plan.devices() {
            let generation = Arc::new(Generation::new(epoch.generation, std::iter::empty()));
            let runners = plan.runners(*device, Arc::clone(&model), &mut memory, &generation)?;
            let executor = Executor::spawn(
                *device,
                runners,
                plan.events(),
                Arc::clone(&pools),
                scratch,
            )?;
            // §7.3 step 2 shadows a replaced operator beside the serving one, so the region it runs
            // in is mapped before any batch: the executor books it from the scratch pool the graph
            // storage sized. Then §4.5's warmup runs every captured bucket once, so the first
            // request does not pay for a capture. Nothing is serving yet, so the control thread
            // waits here; `serve` never does (§2.4).
            let shadow = executor.set_shadow()?;
            device_command(&watchdog, *device, "shadow region", shadow)
                .with_context(|| format!("mapping the shadow region of device {}", device.0))?;
            let warmup = executor.warmup()?;
            device_command(&watchdog, *device, "warmup", warmup)
                .with_context(|| format!("warming up device {}", device.0))?;
            executors.push(executor);
            generations.push(generation);
        }

        // This node's schedulers, one thread per DP rank, each handing its batches to the executors
        // above through `executor::Batches` (§2.4, §4.4). A node that holds none keeps an empty list
        // and serves only what its peers' schedulers ask of it.
        let schedulers: Vec<engine::SchedulerHandle> = Vec::new();

        Ok(Self {
            control,
            schedulers,
            executors,
            memory: Some(memory),
            blocking,
            modules,
            update,
            watchdog,
            programs,
            precopy: None,
            prepared: Vec::new(),
            retired: Vec::new(),
            generations,
            targets: plan.targets().to_vec(),
            program_server: plan.program_server().to_path_buf(),
            stopping: false,
        })
    }

    /// Serves until the daemon closes the process or a failure ends it.
    ///
    /// The control thread waits on its channels, never later than the §2.4
    /// deadline; each pass runs the watchdog over the executors and handles what
    /// arrived.
    fn serve(&mut self) -> Result<Option<UpdateFailure>> {
        let mut running = Vec::new();
        let mut reported = Instant::now();
        loop {
            if self.stopping {
                return Ok(None);
            }
            let deadline = self.watchdog.deadline(oldest(&running));
            // One metrics report a second, and a final one on the way out
            // (§7.3: the daemon exports the engine's counters as metrics).
            if reported.elapsed() >= METRICS_EVERY {
                self.send_metrics();
                reported = Instant::now();
            }
            for event in self.control.wait(deadline)? {
                match event {
                    Ready::Daemon => {
                        if let Some(failure) = self.daemon_messages()? {
                            return Ok(Some(failure));
                        }
                    }
                    Ready::PeerConnecting => {
                        self.control.accept()?;
                    }
                    Ready::Peer(rank) => {
                        if let Some(failure) = self.peer_messages(rank)? {
                            return Ok(Some(failure));
                        }
                    }
                    Ready::Woken | Ready::Idle => {}
                }
            }

            // A precopy in flight advances one stage per pass, never waiting on
            // its builds or its devices (§7.3 step 2).
            self.advance_precopy()?;

            running = self.read_executors();
            if let Some(failure) = self.watch(&running)? {
                return Ok(Some(failure));
            }
        }
    }

    /// The oldest unfinished command of each executor, in rank order, as §2.4
    /// reads them.
    fn read_executors(&self) -> Vec<Running> {
        self.executors
            .iter()
            .enumerate()
            .filter_map(|(rank, executor)| {
                let reading = executor.watchdog();
                let watch = reading.longest()?;
                Some(Running {
                    rank,
                    batch: watch.batch.map(|batch| batch.0),
                    since: watch.started,
                })
            })
            .collect()
    }

    /// The §2.4 rules: a job past half the watchdog timeout is reported with its
    /// batch, and while a job has run for a second the devices' contexts are
    /// probed once a second for a sticky error. Both are failures of the epoch
    /// (§2.5), and of an update on probation (§7.3 step 4).
    fn watch(&mut self, running: &[Running]) -> Result<Option<UpdateFailure>> {
        let now = Instant::now();
        for (_, stall) in self.watchdog.stalls(now, running) {
            if let Some(ending) = self.failure(UpdateFailure::Stalled { stall })? {
                return Ok(Some(ending));
            }
        }

        if self.watchdog.probe_due(now, oldest(running)) {
            self.watchdog.probed(now);
            // The probe leaves an answered error latched in the device rather than failing, so a
            // device that errored and one that is healthy are read in one pass, and only then is
            // the update's decision table applied to what was found.
            let mut sticky = Vec::new();
            for (rank, executor) in self.executors.iter().enumerate() {
                let Probe::Sticky(error) = executor.probe()? else {
                    continue;
                };
                let batch = running
                    .iter()
                    .find(|job| job.rank == rank)
                    .and_then(|job| job.batch);
                sticky.push(UpdateFailure::Device {
                    device: u32::try_from(executor.device().0).unwrap_or_default(),
                    error: error.to_string(),
                    batch,
                });
            }
            for failure in sticky {
                if let Some(ending) = self.failure(failure)? {
                    return Ok(Some(ending));
                }
            }
        }
        Ok(None)
    }

    /// Applies the §7.3 decision table to `failure`: a precopy failure discards
    /// the new generation and serving goes on; a failure on probation reverts
    /// the update; anything else ends the epoch's service. Returns the failure
    /// that ends service, if any.
    fn failure(&mut self, failure: UpdateFailure) -> Result<Option<UpdateFailure>> {
        match self.update.action(&failure) {
            Action::Discard => {
                let generation = self.update.phase().generation().unwrap_or_default();
                self.update.precopy_failed(generation, failure.clone())?;
                self.report(ProgramStatus::PrecopyFailed {
                    generation,
                    failure,
                });
                Ok(None)
            }
            Action::Revert => {
                let generation = self.update.phase().generation().unwrap_or_default();
                self.revert(generation, failure);
                Ok(None)
            }
            Action::Epoch => Ok(Some(failure)),
        }
    }

    /// Serves the daemon's buffered messages: request submissions and commands
    /// go to the schedulers, program-layer commands to [`Self::program`].
    fn daemon_messages(&mut self) -> Result<Option<UpdateFailure>> {
        loop {
            let Some(message) = self.control.daemon_mut().try_recv()? else {
                return Ok(None);
            };
            if let Some(failure) = self.daemon_message(message)? {
                return Ok(Some(failure));
            }
        }
    }

    /// Serves the daemon's request interface: submissions and per-request
    /// commands are routed to the scheduler that holds the request (§2.3), and
    /// program-layer commands run the driver of §7.3.
    fn daemon_message(
        &mut self,
        message: proto::request::wire::ToEngine,
    ) -> Result<Option<UpdateFailure>> {
        let Some(kind) = message.kind else {
            return Ok(None);
        };
        match kind {
            proto::request::wire::to_engine::Kind::Submit(submit) => {
                // Every request of one submission is queued before the next
                // batch is formed (§2.3).
                for request in submit.requests {
                    let request = proto::request::Request::from_wire(request).map_err(|error| {
                        anyhow::anyhow!("a submission did not convert: {error}")
                    })?;
                    let rank = self.route(&request);
                    let Some(scheduler) = self.schedulers.get(rank) else {
                        bail!("a request was routed to rank {rank}, which holds no scheduler");
                    };
                    scheduler.submit(request)?;
                }
                Ok(None)
            }
            proto::request::wire::to_engine::Kind::Command(command) => {
                self.request_command(command)?;
                Ok(None)
            }
            proto::request::wire::to_engine::Kind::Engine(command) => self.engine_command(command),
        }
    }

    /// Serves one command of the daemon's process interface (§2.3).
    ///
    /// `ipc::ProgramCommand` is the one form the driver reads, whether the command came from the
    /// daemon's Protobuf or from a peer of this epoch, so the conversion happens here and nowhere
    /// else. The daemon's `Commit` and `Revert` carry no generation of their own; the one they
    /// name is the generation this process is serving.
    fn process_command(
        &mut self,
        command: proto::process::Command,
    ) -> Result<Option<UpdateFailure>> {
        let current = self.update.current();
        let Some(program) = ProgramCommand::from_command(command.clone(), current)? else {
            // Park, Promote, Drain, Pause, Serve and Shutdown are the epoch layer's: every scheduler
            // applies them at its own next boundary, and answers when it did (§2.5, §7.3 step 6).
            for scheduler in &self.schedulers {
                answer(
                    scheduler.engine_command(command.clone()),
                    self.watchdog.stall_bound(),
                    "an engine command",
                )?;
            }
            return Ok(None);
        };
        match self.program(program, None)? {
            // A precopy answers when its installs finish, from `advance_precopy`.
            None => Ok(None),
            Some(status) => {
                self.report(status);
                Ok(None)
            }
        }
    }

    /// Serves one peer's program-layer messages: the scheduler node's fan-out
    /// of §7.3 step 2 and the commands of steps 3-6.
    fn peer_messages(&mut self, rank: u32) -> Result<Option<UpdateFailure>> {
        loop {
            let Some(command) = ({
                let Some(channel) = self.control.peer_mut(rank) else {
                    return Ok(None);
                };
                channel.try_recv::<ProgramCommand>()?
            }) else {
                return Ok(None);
            };
            if let Some(status) = self.program(command, Some(rank))? {
                self.answer(Some(rank), status);
            }
        }
    }

    /// §7.3: one program-layer command.
    ///
    /// A precopy answers later, from [`Self::advance_precopy`], because its
    /// module builds run between serve passes; every other stage answers at
    /// once.
    fn program(
        &mut self,
        command: ProgramCommand,
        reply: Option<u32>,
    ) -> Result<Option<ProgramStatus>> {
        let status = match command {
            ProgramCommand::Precopy {
                generation,
                programs,
                policy,
            } => {
                self.start_precopy(generation, &programs, &policy, reply)?;
                return Ok(None);
            }
            ProgramCommand::Switchover { generation, step } => self.switchover(generation, step)?,
            ProgramCommand::Commit { generation } => self.commit(generation)?,
            ProgramCommand::Revert { generation } => self.revert(
                generation,
                UpdateFailure::Policy {
                    error: "the daemon reverted the update".into(),
                },
            ),
            ProgramCommand::Purge { generation } => self.purge(generation)?,
        };
        Ok(Some(status))
    }

    /// Delivers a program-layer status: to the peer that asked, and to the
    /// daemon.
    fn answer(&mut self, reply: Option<u32>, status: ProgramStatus) {
        if let Some(rank) = reply {
            if let Some(channel) = self.control.peer_mut(rank) {
                if let Err(error) = channel.send(&status) {
                    tracing::warn!(%error, rank, "answering a peer's program command failed");
                }
            }
        }
        self.report(status);
    }

    /// §7.3 step 2, first half: states the new generation to the Program Server
    /// and admits one install per device on the module-load lane, whose
    /// builds run while serving goes on.
    fn start_precopy(
        &mut self,
        generation: u64,
        programs: &Programs,
        policy: &Program,
        reply: Option<u32>,
    ) -> Result<()> {
        if let Err(error) = self.update.begin_precopy(generation) {
            let failure = UpdateFailure::Precopy {
                reason: error.to_string(),
            };
            self.answer(
                reply,
                ProgramStatus::PrecopyFailed {
                    generation,
                    failure,
                },
            );
            return Ok(());
        }

        // The descriptors each operator must cover are exactly the keys the
        // serving generation holds: an update changes implementations and never
        // the set of descriptors (§7.3 step 2, §5.3).
        let mut descriptors = Vec::with_capacity(programs.len());
        let mut requests = Vec::with_capacity(programs.len());
        for (slot, program) in programs {
            let Some(operator) = ops::operator_name(&slot.op) else {
                return self.abandon_precopy(
                    generation,
                    UpdateFailure::Precopy {
                        reason: format!("operator {} has no implementation in this build", slot.op),
                    },
                    reply,
                );
            };
            let keys = descriptor_keys(&self.generations[0], operator);
            if keys.is_empty() {
                return self.abandon_precopy(
                    generation,
                    UpdateFailure::Precopy {
                        reason: format!(
                            "operator {} holds no descriptor in the serving generation",
                            slot.op
                        ),
                    },
                    reply,
                );
            }
            descriptors.push(Descriptors {
                op: slot.op.clone(),
                keys,
            });
            requests.push(ProgramRequest {
                slot: slot.clone(),
                selector: Selector::Fixed(program.clone()),
            });
        }

        let mut tasks = Vec::with_capacity(self.executors.len());
        for (index, executor) in self.executors.iter().enumerate() {
            let ordinal = executor.device();
            let device = match platform::Device::new(ordinal) {
                Ok(device) => device,
                Err(error) => {
                    return self.abandon_precopy(
                        generation,
                        UpdateFailure::Precopy {
                            reason: format!("opening device {}: {error}", ordinal.0),
                        },
                        reply,
                    );
                }
            };
            let Some(target) = self.targets.get(index).cloned() else {
                return self.abandon_precopy(
                    generation,
                    UpdateFailure::Precopy {
                        reason: format!("device {} has no build target", ordinal.0),
                    },
                    reply,
                );
            };
            let install = Install {
                programs: Arc::clone(&self.programs),
                device,
                target: target.clone(),
                request: PrepareRequest {
                    targets: vec![target],
                    programs: requests.clone(),
                    descriptors: descriptors.clone(),
                },
            };
            match self.modules.admit(install) {
                Ok(task) => tasks.push(task),
                Err(error) => {
                    return self.abandon_precopy(
                        generation,
                        UpdateFailure::Precopy {
                            reason: format!("admitting a module load: {error}"),
                        },
                        reply,
                    );
                }
            }
        }

        self.precopy = Some(Precopy {
            generation,
            reply,
            policy: policy.clone(),
            next: Vec::new(),
            policies: Vec::new(),
            stage: PrecopyStage::Modules(tasks),
        });
        Ok(())
    }

    /// §7.3 step 2, second half: advances the precopy at most one stage. It
    /// never waits on a job: a stage that has not finished leaves the precopy
    /// as it is, and the next serve pass looks again.
    fn advance_precopy(&mut self) -> Result<()> {
        let Some(mut precopy) = self.precopy.take() else {
            return Ok(());
        };
        let generation = precopy.generation;
        let stage = std::mem::replace(&mut precopy.stage, PrecopyStage::Done);

        match stage {
            PrecopyStage::Modules(tasks) => {
                let mut loaded = Vec::with_capacity(tasks.len());
                for task in &tasks {
                    match task.outcome() {
                        None => {
                            // A build is still running: the tasks keep their
                            // results and the lane keeps its capacity.
                            precopy.stage = PrecopyStage::Modules(tasks);
                            self.precopy = Some(precopy);
                            return Ok(());
                        }
                        Some(Outcome::Success(entries)) => loaded.push((*entries).clone()),
                        Some(Outcome::Failed(error)) => {
                            return self.abandon_precopy(
                                generation,
                                UpdateFailure::Precopy {
                                    reason: format!("loading this node's kernels: {error}"),
                                },
                                precopy.reply,
                            );
                        }
                        Some(Outcome::Cancelled) => {
                            return self.abandon_precopy(
                                generation,
                                UpdateFailure::Precopy {
                                    reason: "the module load was cancelled".to_owned(),
                                },
                                precopy.reply,
                            );
                        }
                    }
                }
                self.after_modules(precopy, loaded)
            }
            PrecopyStage::Executors(submissions) => {
                // The policy's module is built on each scheduler's blocking thread, so its answer
                // arrives here as well: a policy that did not prepare fails the update before any
                // device cuts over, which is what §4.4's dry run is for.
                match poll_answers(&precopy.policies) {
                    Some(StageFailure::Pending) => {
                        precopy.stage = PrecopyStage::Executors(submissions);
                        self.precopy = Some(precopy);
                        return Ok(());
                    }
                    Some(StageFailure::Failed(reason)) => {
                        return self.abandon_precopy(
                            generation,
                            UpdateFailure::Policy { error: reason },
                            precopy.reply,
                        );
                    }
                    None => {}
                }
                if let Some(failure) = unfinished(&submissions, "preparing this node's devices") {
                    match failure {
                        StageFailure::Pending => {
                            precopy.stage = PrecopyStage::Executors(submissions);
                            self.precopy = Some(precopy);
                            return Ok(());
                        }
                        StageFailure::Failed(reason) => {
                            return self.abandon_precopy(
                                generation,
                                UpdateFailure::Precopy { reason },
                                precopy.reply,
                            );
                        }
                    }
                }

                // Shadow execution: every scheduler shadows one step of its own resident requests
                // against the prepared generation, in the buckets those requests occupy, and
                // answers once that step ran (§7.3 step 2). The paired kernels compare themselves
                // by `Op::tolerance` and never write resident memory.
                let mut shadow = Vec::with_capacity(self.schedulers.len());
                for scheduler in &self.schedulers {
                    shadow.push(scheduler.shadow(generation));
                }
                precopy.stage = PrecopyStage::Shadow(shadow);
                self.precopy = Some(precopy);
                Ok(())
            }
            PrecopyStage::Shadow(answers) => {
                // The shadow pass runs on the scheduler's own thread, one step of the batches it
                // formed, so the serve pass polls it rather than waiting (§2.4).
                match poll_answers(&answers) {
                    Some(StageFailure::Pending) => {
                        precopy.stage = PrecopyStage::Shadow(answers);
                        self.precopy = Some(precopy);
                        return Ok(());
                    }
                    Some(StageFailure::Failed(reason)) => {
                        return self.abandon_precopy(
                            generation,
                            UpdateFailure::Precopy { reason },
                            precopy.reply,
                        );
                    }
                    None => {}
                }

                // The generations built here are *prepared*, not serving: `generations` is what each
                // device serves now, and a precopy that finished is not serving until the switchover
                // (§7.3 steps 2 and 3). They wait here, so a descriptor lookup still reads the
                // serving generation and a revert has nothing to undo.
                self.prepared = std::mem::take(&mut precopy.next);
                self.update.precopied(generation)?;
                self.answer(precopy.reply, ProgramStatus::Precopied { generation });
                Ok(())
            }
            PrecopyStage::Done => Ok(()),
        }
    }

    /// The generations the loaded kernels build, the candidate policy
    /// instantiated, and the executors' rebind or recapture admitted.
    fn after_modules(&mut self, precopy: Precopy, loaded: Vec<Vec<KernelEntry>>) -> Result<()> {
        let generation = precopy.generation;
        // One generation per device: a kernel belongs to the device context
        // that loaded it.
        let mut next = Vec::with_capacity(loaded.len());
        for (index, entries) in loaded.into_iter().enumerate() {
            let current = Arc::clone(&self.generations[index]);
            next.push(Arc::new(current.replace(generation, entries)));
        }

        // The candidate policy is instantiated on the current snapshot, and
        // every decision it would make is dry-run against the hard constraints
        // before anything switches (§4.4, §7.3 step 2).
        let policies = match self.request_policy(&precopy.policy) {
            Ok(policies) => policies,
            Err(error) => {
                return self.abandon_precopy(
                    generation,
                    UpdateFailure::Policy {
                        error: format!("{error:#}"),
                    },
                    precopy.reply,
                );
            }
        };

        // Rebind or recapture, as jobs between serving batches (§4.5).
        let mut submissions = Vec::with_capacity(next.len());
        for (index, built) in next.iter().enumerate() {
            submissions.push(self.executors[index].precopy(Arc::clone(built))?);
        }
        let mut precopy = precopy;
        precopy.next = next;
        precopy.policies = policies;
        precopy.stage = PrecopyStage::Executors(submissions);
        self.precopy = Some(precopy);
        Ok(())
    }

    /// A failed precopy: the new generation is discarded and serving goes on
    /// (§7.3 step 2).
    fn abandon_precopy(
        &mut self,
        generation: u64,
        failure: UpdateFailure,
        reply: Option<u32>,
    ) -> Result<()> {
        self.discard_policy();
        // The executors already rebound or recaptured this generation's graphs, and a generation
        // that never serves must give its captures back (§7.3 step 2): `purge` releases what is
        // older than the serving generation, which a discarded one is not, so the executor drops it
        // by name. A device still busy with the capture is answered by the queue's own order.
        for executor in &self.executors {
            match executor.discard(generation) {
                Ok(submission) => {
                    if let Err(error) =
                        device_command(&self.watchdog, executor.device(), "discard", submission)
                    {
                        tracing::warn!(%error, "discarding a failed generation on a device failed");
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, "queueing a failed generation's discard failed");
                }
            }
        }
        self.precopy = None;
        if let Err(error) = self.update.precopy_failed(generation, failure.clone()) {
            tracing::error!(%error, "discarding a failed precopy failed");
        }
        self.answer(
            reply,
            ProgramStatus::PrecopyFailed {
                generation,
                failure,
            },
        );
        Ok(())
    }

    /// §7.3 step 3: cut over at the batch boundary whose step the scheduler put
    /// the command at; an expert group's ranks switch at the same step.
    fn switchover(&mut self, generation: u64, step: Option<u64>) -> Result<ProgramStatus> {
        self.switchover_policy(generation, step);
        for executor in &self.executors {
            let device = executor.device();
            let submission = executor.switchover(generation)?;
            if let Err(error) = device_command(&self.watchdog, device, "switchover", submission) {
                // A device that did not cut over would leave this node serving two generations at
                // once, so §7.3 step 4's revert runs on every device: the generation that served
                // before the update is the one serving after it.
                tracing::error!(%error, "cutting over failed; reverting the update");
                let failure = UpdateFailure::Device {
                    device: u32::try_from(device.0).unwrap_or_default(),
                    error: error.to_string(),
                    batch: None,
                };
                return Ok(self.revert(generation, failure));
            }
        }
        // The state machine records the cut-over only when every device serves the new generation,
        // and the process's record of what serves follows it.
        self.update.switchover(generation, Instant::now())?;
        if self.prepared.is_empty() {
            tracing::warn!(generation, "a switchover found no prepared generation");
        } else {
            self.retired = std::mem::replace(&mut self.generations, std::mem::take(&mut self.prepared));
        }
        Ok(ProgramStatus::Switched { generation })
    }

    /// §7.3 step 5: probation ended in a commit; the previous generation is kept
    /// until its purge.
    fn commit(&mut self, generation: u64) -> Result<ProgramStatus> {
        self.update.commit(generation)?;
        // Probation ended in a commit, so the generation before the update can never serve again.
        // Its graphs and kernels live until the purge, which is the executor's book to keep.
        self.retired.clear();
        self.commit_policy(generation);
        Ok(ProgramStatus::Committed { generation })
    }

    /// §7.3 step 4: revert to the generation serving before the update. The old
    /// generation and its nodes stayed through probation, so the revert is the
    /// reverse switchover and nothing has to be prepared again.
    fn revert(&mut self, generation: u64, failure: UpdateFailure) -> ProgramStatus {
        if let Err(error) = self.update.revert(generation, failure.clone()) {
            tracing::error!(%error, "reverting a program-layer update failed");
            return ProgramStatus::Failed {
                generation,
                failure,
            };
        }
        self.revert_policy(generation);
        // The generation that served before the update serves again, and the one built for the
        // update keeps nothing: its kernels are released and its captures are never taken over.
        if !self.retired.is_empty() {
            self.generations = std::mem::take(&mut self.retired);
        }
        self.prepared.clear();
        let current = self.update.current();
        for executor in &self.executors {
            let device = executor.device();
            match executor.switchover(current) {
                Ok(submission) => {
                    if let Err(error) = device_command(&self.watchdog, device, "revert", submission) {
                        tracing::error!(%error, "reverting a device's generation failed");
                    }
                }
                Err(error) => {
                    tracing::error!(%error, "queueing a device's revert failed");
                }
            }
        }
        ProgramStatus::Reverted {
            generation,
            reason: failure.reason(),
        }
    }

    /// §7.3 step 6: release the generation before the serving one, once its last
    /// batch completed. The executor's queue orders the purge behind every batch
    /// it already holds, and `Captured` keeps each kernel alive until the graph
    /// that uses it is rebound or destroyed.
    fn purge(&mut self, generation: u64) -> Result<ProgramStatus> {
        if let Err(error) = self.update.request_purge(generation) {
            bail!("purging generation {generation}: {error}");
        }
        // A purge has no policy half: the scheduling policy is one program per epoch and its
        // modules are released with the scheduler, not with a generation's graphs.
        for executor in &self.executors {
            let device = executor.device();
            let submission = executor.purge(generation)?;
            if let Err(error) = device_command(&self.watchdog, device, "purge", submission) {
                // The generation still serves nothing, so a purge that did not finish leaks its
                // storage rather than breaking correctness: it is reported and the epoch serves on.
                let failure = UpdateFailure::Device {
                    device: u32::try_from(device.0).unwrap_or_default(),
                    error: error.to_string(),
                    batch: None,
                };
                return Ok(ProgramStatus::Failed {
                    generation,
                    failure,
                });
            }
        }
        self.update.purged(generation)?;
        Ok(ProgramStatus::Purged { generation })
    }

    /// Routes a request to the scheduler that serves it (§4.4 `route`).
    fn route(&self, _request: &proto::request::Request) -> usize {
        // The routing policy runs on the scheduler node; a request submitted to
        // a node that holds no scheduler belongs to this node's single rank.
        if self.schedulers.len() <= 1 {
            0
        } else {
            // Several DP ranks on one node: the mechanism's policy instance
            // picks the rank (§4.4). Until that call is published, the first
            // scheduler takes it, which is the single-rank case and the
            // identity of an unloaded deployment.
            0
        }
    }

    /// Routes a per-request command to the scheduler that holds the request.
    ///
    /// The request's scheduler is the rank the submission was routed to; the
    /// daemon addresses commands to the main process, and the mechanism keeps
    /// the rank of every request it accepted.
    fn request_command(&mut self, command: proto::request::wire::Command) -> Result<()> {
        match self.schedulers.as_slice() {
            [scheduler] => answer(
                scheduler.request_command(command),
                self.watchdog.stall_bound(),
                "a request command",
            ),
            [] => bail!("a request command arrived at a node that holds no scheduler"),
            schedulers => bail!(
                "a request command needs the rank the request was routed to, among {} schedulers",
                schedulers.len()
            ),
        }
    }

    /// A command to the whole main process: shutdown, drain, park or promote,
    /// and the program-layer commands a single-rank node receives directly.
    fn engine_command(
        &mut self,
        command: proto::request::wire::EngineCommand,
    ) -> Result<Option<UpdateFailure>> {
        // The request connection carries the epoch's own lifecycle commands. `Shutdown` ends the
        // epoch: every request finishes as aborted, the workers close and the process exits, which
        // is what `close` does in the order the owners need (§2.3). The rest are the routing's and
        // the daemon's successor machinery's - they apply to connections this node does not hold -
        // while the epoch layer's Park, Promote, Serve, Drain and Pause arrive on the process
        // connection, where the schedulers take them.
        if let Some(proto::request::wire::engine_command::Kind::Shutdown(_)) = command.kind {
            self.stopping = true;
        }
        Ok(None)
    }

    /// Sends the engine's counters to the daemon, one report a second and a
    /// final flush when the process ends (`specs/worker.md` §2.3).
    ///
    /// The executor states its own figures: the launches and device nanoseconds
    /// its runners counted, the jobs in flight the control thread's watchdog
    /// reads, and the pools the arena books against the rank's budget - which is
    /// the invariant of §4.1 and §8, a device never mapping more than its budget
    /// allows. The scheduler-owned counters (`steps`, `batches`, `tokens`, the
    /// policy's fuel and calls) come from the schedulers' handle and stay zero
    /// until it exists.
    fn send_metrics(&mut self) {
        let mut kernel_launches = 0_u64;
        let mut device_ns = 0_u64;
        let mut queue = Vec::with_capacity(self.executors.len());
        let mut memory = Vec::with_capacity(self.executors.len());
        for (rank, executor) in self.executors.iter().enumerate() {
            let counters = executor.counters();
            kernel_launches += counters.kernel_launches;
            device_ns += counters.device_ns;
            queue.push(proto::request::wire::QueueDepth {
                rank: rank as u32,
                depth: executor.watchdog().jobs.len() as u32,
            });
            // One allocator books the device's three pools (`arena::DeviceMemory`), so the three
            // figures sum against one budget and item 5's invariant is reported as it is decided.
            let pools = executor.memory();
            memory.push(proto::request::wire::DeviceMemory {
                device: executor.device().0 as u32,
                budget: pools.budget,
                pools: ["graph", "scratch", "resident"]
                    .into_iter()
                    .zip([pools.graph, pools.scratch, pools.resident])
                    .map(|(pool, mapped)| proto::request::wire::PoolMemory {
                        pool: pool.to_owned(),
                        mapped,
                    })
                    .collect(),
            });
        }
        let metrics = proto::request::wire::Metrics {
            steps: 0,
            batches: 0,
            tokens: 0,
            kernel_launches,
            device_ns,
            queue,
            policy_fuel: 0,
            policy_calls: 0,
            memory,
        };
        if let Err(error) = self
            .control
            .daemon_mut()
            .send(&proto::request::wire::FromEngine {
                kind: Some(proto::request::wire::from_engine::Kind::Metrics(metrics)),
            })
        {
            tracing::warn!(%error, "reporting the engine's metrics failed");
        }
    }

    /// Reports a program-layer status to the daemon.
    ///
    /// The request interface gains the program-layer statuses as append-only
    /// fields (`notes/front/executor-root-package.md`); until that field is in
    /// the schema, the status reaches the daemon through the epoch record the
    /// daemon itself reads back at `GET /admin/config` and through the log.
    fn report(&mut self, status: ProgramStatus) {
        // The daemon reads an epoch's status back at `GET /admin/config`; the
        // main process states it on the request interface (front's `Report`
        // vocabulary, `crates/proto/src/process.proto`).
        tracing::info!(?status, "a program-layer status is reported");
    }

    /// Stops every thread in dependency order and releases the owners (§2.4).
    fn close(&mut self) {
        for executor in std::mem::take(&mut self.executors) {
            if let Err(error) = executor.close() {
                tracing::warn!(%error, "closing a device executor failed");
            }
        }
        self.schedulers.clear();
        for error in self.blocking.close() {
            tracing::warn!(%error, "closing a blocking lane failed");
        }
        if let Some(memory) = self.memory.take() {
            if let Err(error) = memory.close() {
                tracing::warn!(%error, "closing the resident owners failed");
            }
        }
        self.control.close();
    }
}

impl MainProcess {
    /// §7.3 step 2's policy half: every scheduler prepares the update's policy program on its own
    /// state, from the module the Program Server resolves for this host's baseline target, and
    /// answers on a channel the serve pass polls.
    ///
    /// The build runs on the scheduler's own blocking thread, never here: a policy's module is
    /// compiled the first time it is asked for, and §2.4 forbids the control thread to wait on it.
    fn request_policy(
        &self,
        policy: &Program,
    ) -> Result<Vec<crossbeam_channel::Receiver<std::result::Result<(), String>>>> {
        let program = PolicyProgram::from_arguments(
            policy.backend.clone(),
            policy.revision,
            &policy.arguments.iter().map(|(name, value)| (name.clone(), value.to_string())).collect(),
        )
        .map_err(|error| anyhow::anyhow!("the update's policy program: {error}"))?;
        let source = Arc::new(
            ProgramSource::connect(&self.program_server)
                .map_err(|error| anyhow::anyhow!("reaching the Program Server: {error}"))?,
        );
        let source: Arc<dyn PolicySource + Send + Sync> = source;
        Ok(self
            .schedulers
            .iter()
            .map(|scheduler| scheduler.precopy_policy(program.clone(), Arc::clone(&source)))
            .collect())
    }

    /// §7.3 step 3's policy half: every scheduler's next batch boundary answers from the prepared
    /// policy. `step` is the boundary the epoch's schedulers agreed on, which only the scheduler
    /// node can name; the control thread does not choose one.
    fn switchover_policy(&self, _generation: u64, _step: Option<u64>) {
        for scheduler in &self.schedulers {
            if let Err(error) = answer(scheduler.switchover_policy(), self.watchdog.stall_bound(), "the policy switchover") {
                tracing::error!(%error, "switching the scheduling policy failed");
            }
        }
    }

    /// §7.3 step 5's policy half: probation ended in a commit, so the prepared policy is the serving
    /// one and the previous module is gone.
    fn commit_policy(&self, _generation: u64) {
        for scheduler in &self.schedulers {
            scheduler.commit_policy();
        }
    }

    /// §7.3 step 4's policy half: the previous policy module answers again from the next boundary,
    /// and the candidate is dropped at it.
    fn revert_policy(&self, _generation: u64) {
        for scheduler in &self.schedulers {
            if let Err(error) = answer(scheduler.revert_policy(), self.watchdog.stall_bound(), "the policy revert") {
                tracing::error!(%error, "reverting the scheduling policy failed");
            }
        }
    }

    /// A precopy that failed discards its policy candidate, so the next update's program is the only
    /// one that can serve.
    fn discard_policy(&self) {
        // The candidate is dropped by the revert every scheduler applies; a scheduler that never
        // switched has nothing to drop, and the stage it is asked for is the one that says so.
        for scheduler in &self.schedulers {
            let _ = scheduler.revert_policy();
        }
    }
}

/// Polls one set of scheduler answers (§7.3 steps 2 and 3).
///
/// The schedulers answer at their own boundary, so the serve pass polls them: the control thread
/// may not wait on a stage whose length the daemon's load decides (§2.4).
fn poll_answers(
    answers: &[crossbeam_channel::Receiver<std::result::Result<(), String>>],
) -> Option<StageFailure> {
    for receiver in answers {
        match receiver.try_recv() {
            Ok(Ok(())) => {}
            Ok(Err(reason)) => return Some(StageFailure::Failed(reason)),
            Err(crossbeam_channel::TryRecvError::Empty) => return Some(StageFailure::Pending),
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                return Some(StageFailure::Failed(
                    "a scheduler of the group is gone".to_owned(),
                ));
            }
        }
    }
    None
}

/// Waits on one scheduler's answer, no longer than the stall bound (§2.4).
fn answer(
    receiver: crossbeam_channel::Receiver<std::result::Result<(), String>>,
    bound: Duration,
    what: &str,
) -> Result<()> {
    match receiver.recv_timeout(bound) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => bail!("{what} failed: {error}"),
        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
            bail!("{what} did not answer within the stall bound")
        }
        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
            bail!("{what} lost its scheduler")
        }
    }
}

/// How often the engine reports its counters.
const METRICS_EVERY: Duration = Duration::from_secs(1);

/// The program slot that holds the epoch's scheduling policy (§4.4, §5.1).
const SCHEDULING: &str = "scheduling";

/// One device's half of §7.3 step 2, on the module-load lane: the Program
/// Server compiles and caches what the generation's descriptors need, each
/// module is resolved for this node's target, and each is loaded into this
/// device's context as a kernel.
///
/// It runs on the blocking pool because compiling can take minutes and no
/// device stream and no scheduler thread may wait on it (§2.4).
struct Install {
    programs: Arc<Mutex<Client>>,
    device: platform::Device,
    target: Target,
    request: PrepareRequest,
}

/// One module the Program Server resolved, held open while its kernel loads (`specs/worker.md`
/// §4.2, §6.3).
///
/// The server sends the module's files as open handles rather than as paths, so the image is read
/// through the handle: a cubin's bytes are small and read once, and a native library is loaded
/// through the handle's `/proc/self/fd` path, which names the same inode the server sent. Every
/// handle stays open until the kernel has loaded, so neither read reaches a file that has gone.
struct Resolved {
    image: Option<Vec<u8>>,
    metadata: Option<Vec<u8>>,
    native: Option<(std::fs::File, String)>,
    entry: String,
}

impl Resolved {
    /// The module as `ops::install` takes it.
    fn as_ops(&self) -> executor::Result<ops::Module<'_>> {
        match (&self.image, &self.native) {
            (Some(image), _) => Ok(ops::Module::Cubin {
                image,
                metadata: self.metadata.as_deref(),
                entry: &self.entry,
            }),
            (None, Some((_, path))) => Ok(ops::Module::Native {
                path: std::path::Path::new(path),
                entry: &self.entry,
            }),
            (None, None) => Err(executor::Error::Invalid(
                "a resolved module carries neither a cubin image nor a native library".to_owned(),
            )),
        }
    }
}

/// Reads a resolved module into the form `ops::install` takes, or refuses a form this build cannot
/// install (§4.2): a scheduling policy's wasm program and a program the main process runs as its
/// own process are not kernels.
fn resolved(module: program_server::Module) -> executor::Result<Resolved> {
    use std::io::Read as _;
    use std::os::fd::AsRawFd as _;

    let invalid = |reason: String| executor::Error::Invalid(reason);
    let mut files = module.files.into_iter();
    match module.form {
        program_server::manifest::ModuleForm::Cubin => {
            let mut image = None;
            let mut metadata = None;
            for file in &mut files {
                let mut bytes = Vec::with_capacity(file.size as usize);
                let mut handle = &file.file;
                handle
                    .read_to_end(&mut bytes)
                    .map_err(|error| invalid(format!("reading {}: {error}", file.name)))?;
                if module.metadata.as_deref() == Some(file.name.as_str()) {
                    metadata = Some(bytes);
                } else {
                    image = Some(bytes);
                }
            }
            let Some(image) = image else {
                return Err(invalid("a cubin module carries no image".to_owned()));
            };
            // The image's kernel symbol is its metadata's name; without metadata the package names
            // the entry itself, and `ops` falls back to the module's single entry.
            let entry = match &metadata {
                Some(metadata) => ops::CubinMetadata::parse(metadata)
                    .map_err(|error| invalid(format!("the cubin's metadata: {error}")))?
                    .name,
                None => String::new(),
            };
            Ok(Resolved {
                image: Some(image),
                metadata,
                native: None,
                entry,
            })
        }
        program_server::manifest::ModuleForm::Native => {
            let Some(entry) = module.entries.get(program_server::manifest::LAUNCH_ENTRY) else {
                return Err(invalid(
                    "a native module states no launch entry".to_owned(),
                ));
            };
            let Some(file) = files.next() else {
                return Err(invalid("a native module carries no library".to_owned()));
            };
            // `dlopen` opens a path, and the handle the server sent has none of its own: the fd is
            // the same inode, and the handle stays ours for as long as the library is loaded.
            let path = format!("/proc/self/fd/{}", file.file.as_raw_fd());
            Ok(Resolved {
                image: None,
                metadata: None,
                native: Some((file.file, path)),
                entry: entry.clone(),
            })
        }
        other => Err(invalid(format!(
            "a {other:?} module is not a kernel this process installs"
        ))),
    }
}

impl HostAction for Install {
    type Output = Vec<KernelEntry>;

    fn run(&self) -> executor::Result<Self::Output> {
        let mut client = self
            .programs
            .lock()
            .map_err(|_| executor::Error::State("the Program Server client was poisoned".into()))?;
        let prepared = client
            .prepare(&self.request, &mut |progress| {
                tracing::debug!(
                    path = %progress.path,
                    state = ?progress.state,
                    "building a module for the new generation"
                );
            })
            .map_err(|error| {
                executor::Error::Invalid(format!("building the new generation: {error}"))
            })?;

        let mut entries = Vec::new();
        for chosen in &prepared.programs {
            let operator = ops::operator_name(&chosen.slot.op).ok_or_else(|| {
                executor::Error::Invalid(format!(
                    "operator {} has no implementation in this build",
                    chosen.slot.op
                ))
            })?;
            let keys: Vec<String> = self
                .request
                .descriptors
                .iter()
                .filter(|descriptor| descriptor.op == chosen.slot.op)
                .flat_map(|descriptor| descriptor.keys.iter().cloned())
                .collect();
            for key in keys {
                let module = client
                    .resolve(
                        &chosen.slot.op,
                        &chosen.program,
                        &self.target,
                        &key,
                        &mut |progress| {
                            tracing::debug!(
                                path = %progress.path,
                                state = ?progress.state,
                                "resolving a module of the new generation"
                            );
                        },
                    )
                    .map_err(|error| {
                        executor::Error::Invalid(format!(
                            "resolving {} for {key}: {error}",
                            chosen.slot.op
                        ))
                    })?;
                let module = resolved(module)?;
                let entry = ops::install(&self.device, operator, &key, module.as_ops()?)
                    .map_err(|error| {
                        executor::Error::Invalid(format!("installing {operator} for {key}: {error}"))
                    })?;
                if let Some(entry) = entry {
                    entries.push(entry);
                }
            }
        }
        Ok(entries)
    }

    fn release(&self) -> executor::Result<()> {
        Ok(())
    }

    fn input_outcome(&self) -> Outcome {
        Outcome::Success(())
    }
}

/// The precopy of one generation, advanced between serve passes (§7.3 step 2).
struct Precopy {
    generation: u64,
    /// The peer whose command this precopy answers, when a peer asked.
    reply: Option<u32>,
    /// The policy program the update installs.
    policy: Program,
    /// The generation built for each device, in `executors` order.
    next: Vec<Arc<Generation>>,
    /// The answers of every scheduler that is preparing the update's policy program. The build runs
    /// on the schedulers' own blocking threads, so the serve pass polls them instead of waiting.
    policies: Vec<crossbeam_channel::Receiver<std::result::Result<(), String>>>,
    stage: PrecopyStage,
}

/// The stage a precopy is in.
enum PrecopyStage {
    /// The module-load lane builds, resolves and installs each device's
    /// kernels.
    Modules(Vec<Arc<HostTask<Install>>>),
    /// The executors rebind or recapture their buckets.
    Executors(Vec<Submission>),
    /// The shadow pass compares the new implementations on the resident requests, one step per
    /// scheduler; each answers when its step ran.
    Shadow(Vec<crossbeam_channel::Receiver<std::result::Result<(), String>>>),
    /// A stage being replaced; the driver never observes it.
    Done,
}

/// What polling a stage's submissions found.
enum StageFailure {
    /// At least one submission has not finished.
    Pending,
    /// A submission failed, or was cancelled.
    Failed(String),
}

/// Whether every submission finished, and if not, why the first unfinished or
/// failed one did not.
fn unfinished(submissions: &[Submission], what: &str) -> Option<StageFailure> {
    for submission in submissions {
        match submission.outcome() {
            None => return Some(StageFailure::Pending),
            Some(Outcome::Success(_)) => {}
            Some(Outcome::Failed(error)) => {
                return Some(StageFailure::Failed(format!("{what}: {error}")));
            }
            Some(Outcome::Cancelled) => {
                return Some(StageFailure::Failed(format!("{what}: cancelled")));
            }
        }
    }
    None
}

/// The descriptor keys `generation` holds for `operator`: what a precopy must
/// cover, and exactly what it never adds to (§7.3 step 2, §5.3).
fn descriptor_keys(generation: &Generation, operator: &'static str) -> Vec<String> {
    generation
        .iter()
        .filter(|(name, _, _)| *name == operator)
        .map(|(_, key, _)| key.to_owned())
        .collect()
}

/// The scheduling policy's source: sched's `PolicySource`, filled from this
/// node's Program Server client (§7.3 step 2, §4.4).
///
/// A policy is a program like any other, and its module reaches the main process
/// only from the Program Server: the daemon names the program's revision, never
/// its bytes. A scheduling policy has no per-deployment descriptor, so the
/// descriptor key is `"default"`.
pub(crate) struct ProgramPolicy {
    programs: Client,
    targets: Vec<Target>,
}

impl ProgramPolicy {
    /// One module per target, in `targets` order: the policy a node's schedulers
    /// instantiate as their candidate.
    pub(crate) fn load(&mut self, program: &Program) -> Result<Vec<Module>> {
        let mut modules = Vec::with_capacity(self.targets.len());
        for target in &self.targets {
            modules.push(self.programs.resolve(
                "scheduling",
                program,
                target,
                "default",
                &mut |progress| {
                    tracing::debug!(
                        state = ?progress.state,
                        "resolving the scheduling policy"
                    );
                },
            )?);
        }
        Ok(modules)
    }
}

/// The oldest of the executors' jobs.
fn oldest(running: &[Running]) -> Option<Running> {
    running.iter().min_by_key(|job| job.since).copied()
}

/// Whether a submission completed successfully.
fn completed<T>(outcome: Outcome<T>) -> Result<()> {
    match outcome {
        Outcome::Success(_) => Ok(()),
        Outcome::Failed(error) => Err(anyhow::anyhow!("{error}")),
        Outcome::Cancelled => bail!("the command was cancelled"),
    }
}

/// Waits on one device command, and no longer than §2.4's stall bound.
///
/// The control thread issues a handful of device commands of its own - the switchover, a revert and
/// a purge - and none of them may hold it indefinitely: a device that has not answered by half the
/// watchdog timeout is stalled, and a stall is a failure the epoch reports rather than something the
/// control thread waits out.
fn device_command(
    watchdog: &Watchdog,
    device: platform::DeviceId,
    what: &str,
    submission: Submission,
) -> Result<()> {
    match submission.wait_timeout(watchdog.stall_bound()) {
        Some(Outcome::Success(_)) => Ok(()),
        Some(Outcome::Failed(error)) => bail!("the {what} of device {} failed: {error}", device.0),
        Some(Outcome::Cancelled) => bail!("the {what} of device {} was cancelled", device.0),
        None => bail!(
            "the {what} of device {} did not finish within the stall bound",
            device.0
        ),
    }
}

/// The handshake with the daemon, which answers with the epoch this node serves
/// (`specs/worker.md` §2.3).
///
/// The daemon states three things: that this build may serve the epoch, the
/// programs that fill the epoch's operators and its scheduling interface (their
/// names and arguments, never their content, §5.1), and the program revision the
/// daemon numbered for the generation that serves once this node switches over
/// (§7.3). The build config is the one the daemon wrote for this node, read
/// before the handshake; the welcome confirms it is the same epoch.
fn handshake(control: &mut Control, launch: &Launch, build: &BuildConfig) -> Result<Epoch> {
    control.daemon_mut().send(&proto::request::wire::Hello {
        engine: launch.engine,
        build: proto::BUILD.to_owned(),
        major: proto::PROTOCOL_MAJOR,
        minor: proto::PROTOCOL_MINOR,
    })?;
    let welcome: proto::request::wire::Welcome = control
        .daemon_mut()
        .recv(Duration::from_secs(60))?
        .context("the daemon closed the request interface during the handshake")?;
    if !welcome.refusal.is_empty() {
        bail!("the daemon refused this main process: {}", welcome.refusal);
    }
    let spec = welcome.spec.unwrap_or_default();
    // The epoch's programs and the generation that serves at start are not in the spec yet: the
    // programs arrive as the daemon's first `Precopy` (§7.3 step 2) and the generation number is the
    // daemon's own. Until it states one, this process serves generation zero, whose `Precopy` names
    // generation one - the daemon's next number (M951).
    Ok(Epoch {
        build: build.clone(),
        programs: Programs::new(),
        generation: 0,
        host: spec.host,
    })
}

/// The model class a checkpoint declares, as its `config.json` states it.
fn architectural_class(checkpoint: &Path) -> Result<String> {
    let path = checkpoint.join("config.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    let config: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", path.display()))?;
    let class = config
        .get("architectures")
        .and_then(|architectures| architectures.as_array())
        .and_then(|architectures| architectures.first())
        .and_then(|class| class.as_str())
        .ok_or_else(|| anyhow::anyhow!("{} declares no model class", path.display()))?;
    Ok(class.to_owned())
}

/// How this rank takes part in the deployment's parallel shape (§4.3, §5.1), as
/// the build config fixes it and the daemon's launch places it.
struct Placement {
    tensor_parallel: u32,
    tensor_rank: u32,
    expert_parallel: u32,
    expert_rank: u32,
    load_options: qwen3_moe::LoadOptions,
}

impl Placement {
    /// This rank's part of the epoch's parallel shape.
    ///
    /// Tensor parallelism splits every layer of the group's ranks; expert parallelism splits the
    /// routed experts of the deployment's schedulers. A rank's ordinal places it in both: its
    /// tensor rank is its position inside its scheduler's group, and its expert rank is which
    /// scheduler's share of the experts it keeps.
    fn of(launch: &Launch, build: &BuildConfig) -> Result<Self> {
        let tensor_parallel = u32::try_from(build.tensor_parallel_size.get())
            .context("a tensor-parallel degree larger than u32::MAX")?;
        if tensor_parallel == 0 {
            bail!("a tensor-parallel degree of zero splits nothing");
        }
        let data_parallel = u32::try_from(build.data_parallel_size.get())
            .context("a data-parallel degree larger than u32::MAX")?;
        let expert_parallel = if build.expert_parallel { data_parallel } else { 1 };
        Ok(Self {
            tensor_parallel,
            tensor_rank: launch.rank % tensor_parallel,
            expert_parallel,
            expert_rank: (launch.rank / tensor_parallel) % expert_parallel,
            // The deployment's page size is the cache's (`kv_cache_config.block_size`), and the
            // attention program's workspace is the load config's; neither is ours to guess, so the
            // model's own defaults stand until the loader takes them from the build config.
            load_options: qwen3_moe::LoadOptions::default(),
        })
    }
}

/// The captured bucket set the epoch's runners are bound to (§4.5), and the same sizes as the
/// capacity plan needs them.
///
/// A bucket set is fixed when the executor is bound, so a batch whose shape no bucket holds is a
/// configuration error found at startup and never a scheduling surprise at the first odd-sized
/// batch. The set comes from the build config's graph sizes, or from the deployment's catalog when
/// they are unset, and every size is clamped to what the batch plan actually admits: a bucket larger
/// than the batch bound would be captured and never used.
struct GraphPlan {
    /// What each runner captures, ascending.
    buckets: executor::plan::Buckets,
    /// Decode row counts, as the capacity plan sizes the decode graph pools.
    decode: Vec<usize>,
    /// Prefill token counts, as the capacity plan sizes the prefill graph pools.
    prefill: Vec<usize>,
    /// Media flow shapes `(tokens, latent)`.
    flow: Vec<(usize, usize)>,
}

impl GraphPlan {
    /// The plan for a deployment whose batch plan admits `rows` rows and `tokens` tokens.
    ///
    /// The sizes are the deployment's, or the executor's catalog where it names none, and every one
    /// above the bound is dropped: a bucket is captured for each size, so a size no batch can fill
    /// would cost a graph and its pool forever. Nothing fits is refused here, at startup. `sampled`
    /// also captures the sampling variants, which a deployment that samples needs and one that
    /// serves logits only does not pay for.
    fn of(
        graphs: &daemon::options::CudaGraphOptions,
        rows: usize,
        tokens: usize,
        sampled: bool,
    ) -> Result<Self> {
        if graphs.policy == proto::request::GraphPolicy::Off {
            return Ok(Self {
                buckets: executor::plan::Buckets::default(),
                decode: Vec::new(),
                prefill: Vec::new(),
                flow: Vec::new(),
            });
        }
        let (rows, tokens) = (
            u32::try_from(rows).context("a batch wider than u32::MAX rows")?,
            u32::try_from(tokens).context("a batch longer than u32::MAX tokens")?,
        );
        let decode = graphs
            .decode_batch_sizes
            .as_ref()
            .map(|sizes| sizes.iter().map(|size| size.get()).collect::<Vec<u32>>());
        let prefill = graphs
            .prefill_token_sizes
            .as_ref()
            .map(|sizes| sizes.iter().map(|size| size.get()).collect::<Vec<u32>>());
        let buckets = executor::plan::Buckets::for_batch(
            decode.as_deref(),
            prefill.as_deref(),
            rows,
            tokens,
            sampled,
        )?;
        // The capacity plan sizes the pools from the sizes the buckets were captured at, so the two
        // cannot disagree about what the epoch captures.
        let decode: Vec<usize> = decode
            .unwrap_or_else(executor::plan::decode_catalog)
            .into_iter()
            .filter(|size| *size != 0 && *size <= rows)
            .map(|size| size as usize)
            .collect();
        let prefill: Vec<usize> = prefill
            .unwrap_or_else(executor::plan::prefill_catalog)
            .into_iter()
            .filter(|size| *size != 0 && *size <= tokens)
            .map(|size| size as usize)
            .collect();
        let flow = graphs
            .flow_shapes
            .as_ref()
            .map(|shapes| {
                shapes
                    .iter()
                    .map(|shape| (shape.0[0] as usize, shape.0[1] as usize))
                    .collect()
            })
            .unwrap_or_else(|| {
                executor::plan::FLOW_CATALOG
                    .into_iter()
                    .map(|(height, width)| (height as usize, width as usize))
                    .collect()
            });
        Ok(Self {
            buckets,
            decode,
            prefill,
            flow,
        })
    }
}

/// The one `match` on model class (§4.3): the executable links the model crates,
/// and nothing else in the tree names a concrete model.
///
/// The class is the checkpoint's own declaration (`config.json`'s single
/// `architectures` entry), so a checkpoint that names a model this build does not
/// link is refused here rather than loaded as something else.
fn load_model(
    checkpoint: &Path,
    placement: &Placement,
    weights: &mut dyn nn::RetainedWeights,
) -> Result<qwen3_moe::Qwen3Moe> {
    match architectural_class(checkpoint)?.as_str() {
        "Qwen3MoeForCausalLM" => {
            // §8: a weight converted once is published to retained storage and mapped read-only
            // afterwards; the checkpoint's own representation stands unless the deployment fixed
            // another, which is the build config's `dtype` and `quantization`.
            let opened = nn::Checkpoint::open(checkpoint)?;
            let mut loader = nn::WeightLoader::new(&opened, weights, None, None);
            let config = qwen3_moe::Config::read(checkpoint)?;
            let partition = qwen3_moe::Partition::new(
                &config,
                placement.tensor_parallel as usize,
                placement.tensor_rank as usize,
                placement.expert_parallel as usize,
                placement.expert_rank as usize,
            )?;
            Ok(qwen3_moe::load(
                checkpoint,
                partition,
                &mut loader,
                placement.load_options,
            )?)
        }
        other => bail!("unsupported model class {other}"),
    }
}
