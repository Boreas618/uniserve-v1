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

use std::collections::BTreeMap;
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
use engine::{PolicyProgram, PolicySlot, PolicySource, Scheduler, Stage};
use ipc::{ProgramCommand, ProgramStatus, UpdateFailure};
use nn::WeightLoader;
use ops::{Generation, KernelEntry};
use program_server::program::Programs;
use program_server::{
    Client, Descriptors, Module, PrepareRequest, Program, ProgramRequest, ProgramSlot, Selector,
    Target,
};
use daemon::epoch::BuildConfig;

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
}

/// The options of `uniserve engine`.
///
/// The daemon launches one main process per node and epoch, from the epoch's
/// build config: the options name where that config is and how this node takes
/// part in the epoch, and nothing that the build config states.
#[derive(Debug, Clone, Args)]
pub(crate) struct EngineArgs {
    /// The epoch's build config, which the daemon wrote: design 7.2's values
    /// that fix an epoch (`daemon::epoch::BuildConfig`).
    #[arg(long)]
    pub build_config: PathBuf,
    /// The checkpoint the epoch serves: a local directory.
    #[arg(long)]
    pub model: PathBuf,
    /// The leader daemon's request interface, `unix:<path>` or `<host>:<port>`.
    #[arg(long)]
    pub requests: String,
    /// The engine number the daemon started this main process under.
    #[arg(long)]
    pub engine: u64,
    /// This node's rank in the deployment.
    #[arg(long, default_value_t = 0)]
    pub node_rank: u32,
    /// This rank's worker group, which keys its storage claims.
    #[arg(long, default_value = "model-0")]
    pub group: String,
    /// This rank's ordinal inside its group.
    #[arg(long, default_value_t = 0)]
    pub rank: u32,
    /// The devices this node's rank serves, as CUDA ordinals.
    #[arg(long = "device")]
    pub devices: Vec<u32>,
    /// The Storage Server's socket for this deployment.
    #[arg(long)]
    pub storage_socket: PathBuf,
    /// The deployment's name, under which the Storage Servers key its claims.
    #[arg(long)]
    pub deployment: String,
    /// The world the epoch executes, which keys the storage it creates.
    #[arg(long)]
    pub world: u64,
    /// The write claim this rank presents to the Storage Server, 32 hex digits.
    #[arg(long, default_value = "00000000000000000000000000000000")]
    pub claim: String,
    /// The Program Server's socket.
    #[arg(long)]
    pub program_server: PathBuf,
    /// A job that runs longer than half of this is reported as `Stalled` (§2.4).
    #[arg(long, default_value = "30")]
    pub watchdog_timeout: f64,
    /// The probation an update must survive before the daemon commits it (§7.3).
    #[arg(long, default_value = "10")]
    pub rollback: f64,
    /// Whether this node holds the epoch's schedulers: a DP rank's scheduler
    /// runs here and drives the other nodes' executors (§2.1).
    #[arg(long)]
    pub scheduler_node: bool,
    /// The addresses on which this node accepts its peer main processes'
    /// connections, when it accepts any.
    #[arg(long = "peer")]
    pub peers: Vec<String>,
}

impl EngineArgs {
    /// This rank's Storage Server launch (`specs/worker.md` §8): the server's socket, the
    /// deployment its claim belongs to, the claim itself and the world exclusive storages are
    /// created in.
    fn storage_launch(&self) -> Result<ipc::StorageLaunch> {
        Ok(ipc::StorageLaunch {
            socket: self.storage_socket.display().to_string(),
            deployment: self.deployment.clone(),
            claim: claim_token(&self.claim)?,
            // The host KV cache and another host's reach are the deployment's; a rank that needs
            // either is told so by the daemon that starts it.
            host_kv_cache: false,
            tcp: None,
            peers: Vec::new(),
            world: self.world,
            standby: false,
            standby_storage: ipc::StandbyStorage::Auto,
        })
    }
}

/// The 32 hex digits of a claim token, as the Storage Server takes it.
fn claim_token(text: &str) -> Result<[u8; 16]> {
    let text = text.trim();
    if text.len() != 32 {
        bail!("a claim is 32 hex digits, not {} characters", text.len());
    }
    let mut claim = [0_u8; 16];
    for (index, byte) in claim.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16)
            .with_context(|| format!("claim {text} is not hexadecimal"))?;
    }
    Ok(claim)
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

    let build: BuildConfig = serde_json::from_reader(
        std::fs::File::open(&args.build_config)
            .with_context(|| format!("opening {}", args.build_config.display()))?,
    )
    .with_context(|| format!("reading the build config {}", args.build_config.display()))?;
    let mut main = MainProcess::start(args, &build)?;
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
    schedulers: Vec<engine::Scheduler>,
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
    /// The generation each device serves now, in `executors` order.
    generations: Vec<Arc<Generation>>,
    /// What this node resolves a module for (§5.3).
    targets: Vec<Target>,
}

impl MainProcess {
    /// Connects to the daemon, loads the epoch's model, and starts the epoch's
    /// threads in the order §2.4 lists.
    fn start(args: &EngineArgs, build: &BuildConfig) -> Result<Self> {
        let mut control = Control::connect(
            &Endpoints {
                requests: args.requests.clone(),
                peers: args.peers.clone(),
            },
            args.engine,
        )?;
        // The request interface's handshake: this process says who it is, and
        // the daemon answers with the engine spec and the epoch record (§2.3).
        let epoch = handshake(&mut control, args, &build)?;

        // §8: the rank's retained weights outlive every world of the epoch, so the store is built
        // once, before the model loads, and keeps what it maps for the epoch's life.
        let storage = Retained::connect(
            &args.storage_launch()?,
            &args.group,
            args.rank,
            epoch.generation,
        )
        .context("connecting to the Storage Server")?;
        let mut weights = WeightStore::new(Arc::clone(&storage));
        let placement = Placement::of(&args, build)?;
        let model = load_model(&args.model, &placement, &mut weights)
            .context("loading the epoch's model")?;

        // The owners of §4 are opened before any thread that borrows them, and
        // released in reverse order by `close`.
        let mut memory = build.open_memory()?;
        let blocking = ThreadPool::new(build.blocking_limits());
        let modules = blocking.lane::<Install>(BlockingKind::ModuleLoad)?;
        let programs = Arc::new(Mutex::new(Client::connect(build.program_server())?));

        let mut executors = Vec::with_capacity(build.devices().len());
        let mut generations = Vec::with_capacity(build.devices().len());
        for device in build.devices() {
            let generation = Arc::new(Generation::new(epoch.generation, std::iter::empty()));
            let runners = build.runners(device, &model, &mut memory, &generation)?;
            let executor = Executor::spawn(device, runners, build.events(), build.graph(device)?)?;
            // §7.3 step 2 shadows a replaced operator beside the serving one, so the region it runs
            // in is mapped before any batch: the executor books it from the scratch pool the graph
            // storage sized. Then §4.5's warmup runs every captured bucket once, so the first
            // request does not pay for a capture. Nothing is serving yet, so the control thread
            // waits here; `serve` never does (§2.4).
            completed(executor.set_shadow()?.wait())
                .with_context(|| format!("mapping the shadow region of device {}", device.0))?;
            completed(executor.warmup()?.wait())
                .with_context(|| format!("warming up device {}", device.0))?;
            executors.push(executor);
            generations.push(generation);
        }
        // The executors own the device streams from here on; the blocks and
        // their memory stay with the process, which outlives every executor.
        // This node's schedulers, one thread per DP rank, each handing its
        // batches to the executors above through `executor::Batches` (§2.4,
        // §4.4). The handle is sched's (M561); a node that holds none keeps an
        // empty list and serves only what its peers' schedulers ask of it.
        let schedulers: Vec<engine::SchedulerHandle> = if args.scheduler_node {
            Vec::new()
        } else {
            Vec::new()
        };

        let update = HotUpdate::new(epoch.generation);
        let watchdog = Watchdog::new(build.watchdog_timeout())?;

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
            generations,
            targets: build.targets(),
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
                    let Some(scheduler) = self.schedulers.get_mut(rank) else {
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
        let Some(command) = ProgramCommand::from_command(command, current)? else {
            // Park, Promote, Drain, Pause and Serve are the epoch layer's, not the program
            // layer's; the daemon's own path runs them.
            return Ok(None);
        };
        match self.program(command, None)? {
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

                // Shadow execution: one batch per occupied bucket on the
                // resident requests, sent to every rank of the group that holds
                // them (§7.3 step 2).
                let mut batches = Vec::new();
                for scheduler in &mut self.schedulers {
                    batches.extend(scheduler.shadow(generation)?);
                }
                let mut shadow = Vec::with_capacity(batches.len());
                for batch in batches {
                    let Some(executor) = self.executors.get(batch.rank) else {
                        return self.abandon_precopy(
                            generation,
                            UpdateFailure::Precopy {
                                reason: format!("a shadow batch named rank {}", batch.rank),
                            },
                            precopy.reply,
                        );
                    };
                    shadow.push(executor.submit(batch.batch)?);
                }
                precopy.stage = PrecopyStage::Shadow(shadow);
                self.precopy = Some(precopy);
                Ok(())
            }
            PrecopyStage::Shadow(submissions) => {
                if let Some(failure) = unfinished(&submissions, "a shadow batch") {
                    match failure {
                        StageFailure::Pending => {
                            precopy.stage = PrecopyStage::Shadow(submissions);
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

                self.generations = std::mem::take(&mut precopy.next);
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
        if let Err(error) = precopy_policy(&mut self.schedulers, &precopy.policy) {
            return self.abandon_precopy(
                generation,
                UpdateFailure::Policy {
                    error: format!("{error:#}"),
                },
                precopy.reply,
            );
        }

        // Rebind or recapture, as jobs between serving batches (§4.5).
        let mut submissions = Vec::with_capacity(next.len());
        for (index, built) in next.iter().enumerate() {
            submissions.push(self.executors[index].precopy(Arc::clone(built))?);
        }
        let mut precopy = precopy;
        precopy.next = next;
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
        discard_policy(&mut self.schedulers);
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
        switchover_policy(&mut self.schedulers, generation, step)?;
        for executor in &self.executors {
            if let Err(error) = completed(executor.switchover(generation)?.wait()) {
                bail!("switching device {}: {error}", executor.device().0);
            }
        }
        self.update.switchover(generation, Instant::now())?;
        Ok(ProgramStatus::Switched { generation })
    }

    /// §7.3 step 5: probation ended in a commit; the previous generation is kept
    /// until its purge.
    fn commit(&mut self, generation: u64) -> Result<ProgramStatus> {
        self.update.commit(generation)?;
        commit_policy(&mut self.schedulers, generation)?;
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
        if let Err(error) = revert_policy(&mut self.schedulers, generation) {
            tracing::error!(%error, "reverting the scheduling policy failed");
        }
        let current = self.update.current();
        for executor in &self.executors {
            match executor.switchover(current) {
                Ok(submission) => {
                    if let Err(error) = completed(submission.wait()) {
                        tracing::error!(
                            %error,
                            device = executor.device().0,
                            "reverting a device's generation failed"
                        );
                    }
                }
                Err(error) => tracing::error!(
                    %error,
                    device = executor.device().0,
                    "queueing a device's revert failed"
                ),
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
        for scheduler in &mut self.schedulers {
            scheduler.policy_mut().request(Stage::Purge, generation)?;
        }
        for executor in &self.executors {
            if let Err(error) = completed(executor.purge(generation)?.wait()) {
                bail!("purging device {}: {error}", executor.device().0);
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
        match self.schedulers.as_mut_slice() {
            [scheduler] => scheduler.command(command),
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
        for scheduler in &mut self.schedulers {
            scheduler.engine_command(&command)?;
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

/// §7.3 step 2's policy half: the update's policy program is prepared on the
/// scheduler's own snapshot and adopted as a candidate, which the switchover puts
/// in the serving instance's place.
///
/// The state a candidate is admitted on is the scheduler's - the live objects, the
/// step's room and the shortage the mechanism faces - so the scheduler takes the
/// program and prepares it, and this side never holds a slot.
fn precopy_policy(schedulers: &mut [Scheduler], policy: &Program) -> Result<()> {
    for scheduler in schedulers {
        scheduler.precopy_policy(policy)?;
    }
    Ok(())
}

/// §7.3 step 3's policy half: the switchover is asked for on the slot, and the
/// mechanism applies it at this rank's next batch boundary.
///
/// `step` is the boundary the epoch's schedulers agreed on, which only the
/// scheduler node can name; `None` means the command came from the daemon, which
/// does not choose a boundary, so each scheduler switches at its own next one.
fn switchover_policy(schedulers: &mut [Scheduler], _generation: u64, _step: Option<u64>) -> Result<()> {
    for scheduler in schedulers {
        scheduler.policy_mut().request(Stage::Switchover);
    }
    Ok(())
}

/// §7.3 step 5's policy half: probation ended in a commit, so the candidate is
/// the serving instance and the previous one is gone.
fn commit_policy(schedulers: &mut [Scheduler], _generation: u64) -> Result<()> {
    for scheduler in schedulers {
        scheduler.policy_mut().request(Stage::Commit);
    }
    Ok(())
}

/// §7.3 step 4's policy half: the previous policy module serves again from the
/// next batch boundary, and the candidate is dropped at it.
fn revert_policy(schedulers: &mut [Scheduler], _generation: u64) -> Result<()> {
    for scheduler in schedulers {
        scheduler.policy_mut().request(Stage::Revert);
    }
    Ok(())
}

/// A discarded generation's policy slot: the candidate never serves (§7.3 step 2,
/// a precopy that failed).
fn discard_policy(schedulers: &mut [Scheduler]) {
    for scheduler in schedulers {
        scheduler.policy_mut().discard();
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
    stage: PrecopyStage,
}

/// The stage a precopy is in.
enum PrecopyStage {
    /// The module-load lane builds, resolves and installs each device's
    /// kernels.
    Modules(Vec<Arc<HostTask<Install>>>),
    /// The executors rebind or recapture their buckets.
    Executors(Vec<Submission>),
    /// The shadow batches compare the new implementations on the resident
    /// requests.
    Shadow(Vec<Submission>),
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

/// The handshake with the daemon, which answers with the epoch this node serves
/// (`specs/worker.md` §2.3).
///
/// The daemon states three things: that this build may serve the epoch, the
/// programs that fill the epoch's operators and its scheduling interface (their
/// names and arguments, never their content, §5.1), and the program revision the
/// daemon numbered for the generation that serves once this node switches over
/// (§7.3). The build config is the one the daemon wrote for this node, read
/// before the handshake; the welcome confirms it is the same epoch.
fn handshake(control: &mut Control, args: &EngineArgs, build: &BuildConfig) -> Result<Epoch> {
    control.daemon_mut().send(&proto::request::wire::Hello {
        engine: args.engine,
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
    // The epoch's programs and the revision that serves from the first
    // switchover are front's fields on the welcome (`crates/proto`'s
    // `Welcome`/`process.proto`); until they are in the schema this process
    // serves nothing, which is the state a standby main process is in anyway.
    Ok(Epoch {
        build: build.clone(),
        programs: Programs::new(),
        generation: 0,
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
    fn of(args: &EngineArgs, build: &BuildConfig) -> Result<Self> {
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
            tensor_rank: args.rank % tensor_parallel,
            expert_parallel,
            expert_rank: (args.rank / tensor_parallel) % expert_parallel,
            // The deployment's page size is the cache's (`kv_cache_config.block_size`), and the
            // attention program's workspace is the load config's; neither is ours to guess, so the
            // model's own defaults stand until the loader takes them from the build config.
            load_options: qwen3_moe::LoadOptions::default(),
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
