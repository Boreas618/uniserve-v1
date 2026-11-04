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
//! ## Interfaces this file is written against
//!
//! Three parties publish the calls the assembly and the driver make, and none of
//! them is in this branch's tree yet. Their names here are the ones exec asked
//! for, so that the merge is a compile, not a rewrite:
//!
//! - program-server: `BuildConfig` (design §5.1's epoch record) and the
//!   `Client`/`Program`/`Programs` precopy resolves modules with.
//! - sched (`crates/engine`): `Scheduler` with its `PolicySlot` for §7.3 steps
//!   2-5 (`precopy`, `discard`, `request(Stage::{Switchover, Revert, Commit,
//!   Purge})`, `boundary`) and the shadow batches of the currently resident
//!   requests, `Scheduler::shadow`.
//! - model (`crates/ops`): `install(device, operator, descriptor, module)` and
//!   `operator_name(name)`, the only place the driver crosses a program's name
//!   with the operator that fills it.
//! `Client::resolve`, the `Program`/`Programs` an epoch record names, and the
//! build config all come from program-server, so the three bullets above name
//! that one crate twice by design of the record, not by accident.
//!
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

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail};
use clap::Args;

use executor::blocking::ThreadPool;
use executor::completion::Outcome;
use executor::control::{Control, Endpoints, Ready};
use executor::device::{Executor, Probe};
use executor::hot_update::{Action, HotUpdate};
use executor::memory::Memory;
use executor::watchdog::{Running, Watchdog};
use ipc::{ProgramCommand, ProgramStatus, UpdateFailure};
use nn::{Checkpoint, Model, Partition, WeightLoader};
use ops::Generation;
use program_server::{BuildConfig, Client, EpochRecord, Module, Program, Programs, Target};

use crate::cli::LogLevels;
use crate::logging;

/// The options of `uniserve engine`.
///
/// The daemon launches one main process per node and epoch. Everything the
/// process is built from comes from its build config (§7.1); the options name
/// only where that config is and how this node takes part in the epoch.
#[derive(Debug, Clone, Args)]
pub(crate) struct EngineArgs {
    /// The epoch's build config for this node, which the daemon wrote.
    #[arg(long)]
    pub build_config: PathBuf,
    /// The leader daemon's request interface, `unix:<path>` or `<host>:<port>`.
    #[arg(long)]
    pub requests: String,
    /// The engine number the daemon started this main process under.
    #[arg(long)]
    pub engine: u64,
    /// This node's rank in the deployment.
    #[arg(long, default_value_t = 0)]
    pub node_rank: u32,
    /// Whether this node holds the epoch's schedulers: a DP rank's scheduler
    /// runs here and drives the other nodes' executors (§2.1).
    #[arg(long)]
    pub scheduler_node: bool,
    /// The addresses on which this node accepts its peer main processes'
    /// connections, when it accepts any.
    #[arg(long = "peer")]
    pub peers: Vec<String>,
}

/// Runs the `engine` command.
pub(crate) fn run(args: &EngineArgs, lifeline: Option<i32>, log: &LogLevels) -> Result<ExitCode> {
    crate::follow_lifeline(lifeline);
    logging::init_tracing(
        "Engine",
        log.log_level.as_deref(),
        log.log_level_http.as_deref(),
    );

    let build = BuildConfig::read(&args.build_config)
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
    memory: Memory,
    blocking: ThreadPool,
    /// The program-layer update's phases and bookkeeping (§7.3).
    update: HotUpdate,
    /// The stall and sticky-error rules of §2.4.
    watchdog: Watchdog,
    /// The Program Server client precopy resolves modules with (§6.3).
    programs: Client,
    /// The generation each device serves now, in `executors` order.
    generations: Vec<Arc<Generation>>,
    /// What this node resolves a module for (§5.3).
    targets: Vec<program_server::Target>,
    /// The epoch record the daemon pinned (§5.1).
    record: EpochRecord,
}

impl MainProcess {
    /// Builds the main process for `build`: connects to the daemon, loads the
    /// epoch's model, and starts the epoch's threads in the order §2.4 lists.
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
        let record = handshake(&mut control, args, build)?;

        let model = load_model(
            record.model_class(),
            build.checkpoint(),
            build.partition(),
            build.loader(),
        )
        .context("loading the epoch's model")?;

        // The owners of §4 are opened before any thread that borrows them, and
        // released in reverse order by `close`.
        let mut memory = build.open_memory()?;
        let blocking = ThreadPool::new(build.blocking_limits());
        let mut executors = Vec::with_capacity(build.devices().len());
        let mut generations = Vec::with_capacity(build.devices().len());
        for device in build.devices() {
            let generation = Arc::new(record.generation_kernels(build, device)?);
            let runners = build.runners(device, &model, &mut memory, &generation)?;
            executors.push(Executor::spawn(
                device,
                runners,
                build.streams(device)?,
                build.events(),
            )?);
            generations.push(generation);
        }
        // The executors own the device streams from here on; the blocks and
        // their memory stay with the process, which outlives every executor.
        let schedulers = match args.scheduler_node {
            true => build.schedulers(&record, &mut memory)?,
            false => Vec::new(),
        };

        let update = HotUpdate::new(record.generation());
        let watchdog = Watchdog::new(build.watchdog_timeout())?;

        Ok(Self {
            control,
            schedulers,
            executors,
            memory,
            blocking,
            update,
            watchdog,
            programs: Client::connect(build.program_server())?,
            generations,
            targets: build.targets(),
            record,
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
            for (rank, executor) in self.executors.iter().enumerate() {
                let Probe::Sticky(error) = executor.probe()? else {
                    continue;
                };
                let batch = running
                    .iter()
                    .find(|job| job.rank == rank)
                    .and_then(|job| job.batch);
                let failure = UpdateFailure::Device {
                    device: u32::try_from(executor.device().0).unwrap_or_default(),
                    error: error.to_string(),
                    batch,
                };
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
                    let request = proto::request::Request::from_wire(request)
                        .map_err(|error| anyhow::anyhow!("a submission did not convert: {error}"))?;
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
            proto::request::wire::to_engine::Kind::Engine(command) => {
                self.engine_command(command)
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
            if let Some(status) = self.program(command)? {
                if let Some(channel) = self.control.peer_mut(rank) {
                    channel.send(&status)?;
                }
            }
        }
    }

    /// §7.3: one program-layer command.
    fn program(&mut self, command: ProgramCommand) -> Result<Option<ProgramStatus>> {
        let status = match command {
            ProgramCommand::Precopy {
                generation,
                programs,
            } => self.precopy(generation, &programs)?,
            ProgramCommand::Switchover { generation, step } => {
                self.switchover(generation, step)?
            }
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

    /// §7.3 step 2: resolve and load the new generation on the blocking pool,
    /// build it per device, let the scheduler instantiate the candidate policy
    /// and dry-run its decisions, ask each executor for its rebinds or
    /// recaptures, and run one shadow batch on the resident requests.
    fn precopy(&mut self, generation: u64, programs: &Programs) -> Result<ProgramStatus> {
        if let Err(error) = self.update.begin_precopy(generation) {
            return Ok(ProgramStatus::PrecopyFailed {
                generation,
                failure: UpdateFailure::Precopy {
                    reason: error.to_string(),
                },
            });
        }

        // One generation per device: a Kernel belongs to the device context that
        // loaded it, so the same modules are loaded once per device.
        let mut next = Vec::with_capacity(self.executors.len());
        for index in 0..self.executors.len() {
            let current = Arc::clone(&self.generations[index]);
            match self.build_generation(index, generation, programs, &current) {
                Ok(built) => next.push(built),
                Err(error) => {
                    let failure = UpdateFailure::Precopy {
                        reason: format!(
                            "resolving device {}: {error:#}",
                            self.executors[index].device().0
                        ),
                    };
                    return self.precopy_failed(generation, failure);
                }
            }
        }

        // The candidate scheduling policy is instantiated on the current
        // snapshot, and every decision is dry-run against the hard constraints
        // before anything switches (§4.4, §7.3 step 2).
        if let Err(error) = precopy_policy(&mut self.schedulers, &programs.policy) {
            let failure = UpdateFailure::Policy {
                error: format!("{error:#}"),
            };
            return self.precopy_failed(generation, failure);
        }

        // Rebind or recapture, as jobs between serving batches (§4.5).
        for (index, built) in next.iter().enumerate() {
            let prepared = self.executors[index].precopy(Arc::clone(built))?;
            if let Err(error) = completed(prepared.wait()) {
                let failure = UpdateFailure::Precopy {
                    reason: format!(
                        "preparing device {}: {error}",
                        self.executors[index].device().0
                    ),
                };
                return self.precopy_failed(generation, failure);
            }
        }

        // Shadow execution: one batch per occupied bucket on the resident
        // requests; the new implementation writes twins and is compared within
        // the operator's tolerance, never writing resident memory (§7.3).
        for scheduler in &mut self.schedulers {
            for shadow in scheduler.shadow(generation)? {
                let submission = self
                    .executors
                    .get(shadow.rank)
                    .ok_or_else(|| anyhow::anyhow!("a shadow batch named rank {}", shadow.rank))?
                    .submit(shadow.batch)?;
                if let Err(error) = completed(submission.wait()) {
                    let failure = UpdateFailure::Precopy {
                        reason: format!("shadow batch on rank {}: {error}", shadow.rank),
                    };
                    return self.precopy_failed(generation, failure);
                }
            }
        }

        self.generations = next;
        self.update.precopied(generation)?;
        Ok(ProgramStatus::Precopied { generation })
    }

    /// Builds one device's generation of the new programs: an operator whose
    /// program did not change keeps its kernels, and a changed one resolves its
    /// modules and loads them into this device's context.
    fn build_generation(
        &mut self,
        index: usize,
        generation: u64,
        programs: &Programs,
        current: &Arc<Generation>,
    ) -> Result<Arc<Generation>> {
        let ordinal = self.executors[index].device();
        let device = platform::Device::new(ordinal)
            .with_context(|| format!("opening device {}", ordinal.0))?;
        let mut entries = Vec::new();
        for (slot, program) in &programs.kernels {
            let operator = ops::operator_name(&slot.op).ok_or_else(|| {
                anyhow::anyhow!(
                    "operator {} of program {} has no implementation in this build",
                    slot.op,
                    program.backend
                )
            })?;
            // A program-layer switch changes implementations, never the set of
            // descriptors: it resolves exactly the keys the current generation
            // holds for this operator (§7.3).
            let keys: Vec<String> = current
                .iter()
                .filter(|(name, _, _)| *name == operator)
                .map(|(_, key, _)| key.to_owned())
                .collect();
            if keys.is_empty() {
                bail!(
                    "operator {} holds no descriptor in the serving generation",
                    slot.op
                );
            }
            for key in keys {
                for target in &self.targets {
                    let module = self.programs.resolve(
                        &slot.op,
                        program,
                        target,
                        &key,
                        &mut |progress| {
                            tracing::debug!(
                                op = %slot.op,
                                descriptor = %key,
                                state = ?progress.state,
                                "resolving a module of the new generation"
                            );
                        },
                    )?;
                    if let Some(entry) = ops::install(&device, operator, &key, module)? {
                        entries.push(entry);
                    }
                }
            }
        }
        Ok(Arc::new(current.replace(generation, entries)))
    }

    /// A precopy failure: the new generation is discarded and serving goes on
    /// (§7.3 step 2).
    fn precopy_failed(
        &mut self,
        generation: u64,
        failure: UpdateFailure,
    ) -> Result<ProgramStatus> {
        discard_policy(&mut self.schedulers);
        self.update
            .precopy_failed(generation, failure.clone())?;
        Ok(ProgramStatus::PrecopyFailed {
            generation,
            failure,
        })
    }

    /// §7.3 step 3: cut over at the batch boundary whose step the scheduler put
    /// the command at; an expert group's ranks switch at the same step.
    fn switchover(&mut self, generation: u64, step: u64) -> Result<ProgramStatus> {
        switchover_policy(&mut self.schedulers, generation, step)?;
        for executor in &self.executors {
            if let Err(error) = completed(executor.switchover(generation)?.wait()) {
                bail!("switching device {}: {error}", executor.device().0);
            }
        }
        self.update.switchover(generation)?;
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
        ProgramStatus::Reverted { generation }
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
    /// final flush when the process ends.
    ///
    /// Sources: sched owns `steps`, `batches`, `tokens`, `policy_fuel` and
    /// `policy_calls`; the executor owns `kernel_launches`, `device_ns` and
    /// each rank's queue depth. Neither publishes counters yet (the executor
    /// offers `watchdog` and `probe` only), so every field is reported as zero
    /// rather than estimated: an honest zero the daemon can graph, not a guess.
    fn send_metrics(&mut self) {
        let metrics = proto::request::wire::Metrics {
            steps: 0,
            batches: 0,
            tokens: 0,
            kernel_launches: 0,
            device_ns: 0,
            queue: self
                .executors
                .iter()
                .enumerate()
                .map(|(rank, _)| proto::request::wire::QueueDepth {
                    rank: rank as u32,
                    depth: 0,
                })
                .collect(),
            policy_fuel: 0,
            policy_calls: 0,
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
        self.record.record_status(&status);
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
        self.memory.close();
        self.control.close();
    }
}

/// §7.3 step 2's policy half: sched's `PolicySlot` instantiates the candidate
/// scheduling policy on the current snapshot and dry-runs every decision it
/// would make against the mechanism's hard constraints (§4.4).
fn precopy_policy(schedulers: &mut [Scheduler], policy: &Program) -> Result<()> {
    for scheduler in schedulers {
        scheduler.policy_mut().precopy(policy)?;
    }
    Ok(())
}

/// §7.3 step 3's policy half: the switchover is requested on the slot, and the
/// mechanism executes it at the rank's batch boundary whose step is `step`.
fn switchover_policy(schedulers: &mut [Scheduler], generation: u64, step: u64) -> Result<()> {
    for scheduler in schedulers {
        let slot: &mut PolicySlot = scheduler.policy_mut();
        slot.request(Stage::Switchover, generation)?;
        slot.boundary(step)?;
    }
    Ok(())
}

/// §7.3 step 5's policy half: probation ended in a commit.
fn commit_policy(schedulers: &mut [Scheduler], generation: u64) -> Result<()> {
    for scheduler in schedulers {
        scheduler.policy_mut().request(Stage::Commit, generation)?;
    }
    Ok(())
}

/// §7.3 step 4's policy half: the update is reverted and the previous policy
/// module serves again from the next batch boundary.
fn revert_policy(schedulers: &mut [Scheduler], generation: u64) -> Result<()> {
    for scheduler in schedulers {
        scheduler.policy_mut().request(Stage::Revert, generation)?;
    }
    Ok(())
}

/// A discarded generation's policy slot: the candidate never serves.
fn discard_policy(schedulers: &mut [Scheduler]) {
    for scheduler in schedulers {
        if let Err(error) = scheduler.policy_mut().discard() {
            tracing::warn!(%error, "discarding a precopied scheduling policy failed");
        }
    }
}

/// How often the engine reports its counters.
const METRICS_EVERY: Duration = Duration::from_secs(1);

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

/// The handshake with the daemon, which answers with the engine spec and the
/// epoch record (§2.3).
fn handshake(
    control: &mut Control,
    args: &EngineArgs,
    build: &BuildConfig,
) -> Result<EpochRecord> {
    control.daemon_mut().send(&proto::request::wire::Hello {
        engine: args.engine,
        node_rank: args.node_rank,
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
    EpochRecord::from_welcome(&welcome, build).context("the daemon's engine spec")
}

/// The one `match` on model class (§4.3): the executable links the model crates,
/// and nothing else in the tree names a concrete model.
fn load_model(
    model_class: &str,
    checkpoint: &Checkpoint,
    partition: &Partition,
    loader: &mut WeightLoader,
) -> Result<Model> {
    match model_class {
        "Qwen3MoeForCausalLM" => qwen3_moe::load(checkpoint, partition, loader),
        "MiniMaxH3ModularPipeline" => minimax_h3::load(checkpoint, partition, loader),
        other => bail!("unsupported model class {other}"),
    }
}
