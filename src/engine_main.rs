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

use crate::engine_plan::EpochPlan;
use daemon::epoch::BuildConfig;
use engine::{PolicyProgram, PolicySource, ProgramSource};
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
use ipc::{ProgramCommand, ProgramStatus, UpdateFailure};
use ops::{Generation, KernelEntry};
use program_server::program::Programs;
use program_server::{
    Client, Descriptors, PrepareRequest, Program, ProgramRequest, Selector, Target,
};

/// The epoch as this node takes it at start.
///
/// The daemon fixes the epoch with a [`BuildConfig`] (design §7.2 and §7.7) and
/// fills its operators and its scheduling interface with programs (§5.1); the
/// main process never receives program content, only the programs' names, and it
/// resolves their modules from the Program Server itself.
pub(crate) struct Epoch {
    /// The program revision the daemon numbered for the generation that serves
    /// once this node switches over (§7.3). Zero until the daemon's first `Precopy` names the
    /// number it prepares: the number is the daemon's and this process never invents one.
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
    /// The leader daemon's request interface, `unix:<path>` or `<host>:<port>`. It speaks the
    /// request vocabulary: `wire::{Hello, Welcome, ToEngine, FromEngine}` - the epoch's
    /// submissions, its per-request commands, and this process's accepted requests, events and
    /// metrics.
    pub requests: String,
    /// The daemon's process interface, `unix:<path>` or `<host>:<port>`: the supervision frames
    /// this process claims its storage with and reports to, and the one place a program-layer
    /// command can reach it. It speaks `process::{Hello, Welcome, Upstream, Downstream}`, which is
    /// a different vocabulary from `requests`, not a second address of the same one.
    pub processes: String,
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
    /// The probation an update must survive before the daemon commits it (§7.3 step 4). The daemon
    /// measures the probation and decides the commit; this process reports what happened and does
    /// not judge it, so the field is read where the epoch is decided and not here.
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
    /// Whether this process was started as the standby of the epoch that serves: it maps its
    /// scope's storage, loads the epoch's weights, opens its pools and captures its graphs, and
    /// then parks - executing nothing and admitting nothing - and reports `Parked`. The daemon
    /// promotes it in place (`Promote`, answered `Promoted`) and only then does the frontend
    /// submit to it (§2.5, §7.4).
    ///
    /// Defaulted rather than required, so that a launch file written before this field existed
    /// still reads: an initial process is not a standby, and `false` says exactly that. A standby
    /// that claimed its storage writable would give its scope two writers, which is why the role
    /// is the launcher's statement rather than the process's guess.
    #[serde(default)]
    pub standby: bool,
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

/// This process's connection to its daemon's process interface (§2.3).
///
/// The daemon has two interfaces and they are two sockets with two vocabularies: the request one
/// carries the epoch's requests, their events and the metrics, and this one carries the supervision
/// frames - the storage claim, the §7.3 reports, and every command the daemon sends an epoch. A
/// frame of one sent on the other is unreadable at the far end, which is why the claim and the
/// reports live here and not on the request connection.
struct ProcessInterface {
    connection: proto::Connection,
    /// The commands read while waiting for the answer to something else. A daemon may send a
    /// `Precopy` while a claim is outstanding, and dropping it would lose the epoch's first
    /// generation.
    pending: std::collections::VecDeque<proto::process::Command>,
    /// The daemon closed the interface: the epoch is over, and the process says so and goes.
    closed: bool,
}

impl ProcessInterface {
    /// Connects to `address` and states this process's identity, which the daemon answers with the
    /// deployment's nodes or a refusal (§2.3).
    fn connect(address: &str, deployment: &str) -> Result<Self> {
        let mut connection = proto::Connection::connect(address).with_context(|| {
            format!("connecting to the daemon's process interface at {address}")
        })?;
        connection.send(&proto::process::Hello {
            major: proto::PROTOCOL_MAJOR,
            minor: proto::PROTOCOL_MINOR,
            build: proto::BUILD.to_owned(),
            peer: Some(proto::process::hello::Peer::Engine(
                proto::process::Engine {
                    deployment: deployment.to_owned(),
                },
            )),
        })?;
        // The welcome is the daemon's acceptance of this process: a node that has not admitted it
        // would mint a claim for a process it does not supervise, so nothing is asked before it
        // arrives.
        let welcome: proto::process::Welcome = connection
            .recv(HANDSHAKE_TIMEOUT)?
            .context("the daemon did not answer this main process's process handshake")?;
        if !welcome.refusal.is_empty() {
            bail!(
                "the daemon refused this main process's process connection: {}",
                welcome.refusal
            );
        }
        // A process the daemon admits and lists no node for is a refusal too: the nodes are what the
        // claim and every later request are answered by, so a welcome with none would let this
        // process run on a deployment the daemon does not know it on.
        if welcome.nodes.is_empty() {
            bail!(
                "the daemon admitted this main process but listed no node of the deployment, so \
                 nothing here can be claimed or supervised"
            );
        }
        tracing::info!(
            nodes = welcome.nodes.len(),
            build = %welcome.build,
            "connected to the deployment's process interface"
        );
        Ok(Self {
            connection,
            pending: std::collections::VecDeque::new(),
            closed: false,
        })
    }

    /// Receives one `Downstream` message, waiting no longer than `timeout`.
    fn receive(
        connection: &mut proto::Connection,
        timeout: Duration,
        what: &str,
    ) -> Result<proto::process::Downstream> {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                bail!("the daemon did not answer {what} within {timeout:?}");
            }
            connection.wait(remaining)?;
            if let Some(message) = connection.try_recv::<proto::process::Downstream>()? {
                return Ok(message);
            }
        }
    }

    /// Sends one request and answers its reply, filing every command that arrives meanwhile.
    fn request(
        &mut self,
        request: proto::process::Request,
        timeout: Duration,
    ) -> Result<proto::process::Reply> {
        let id = request.id;
        self.connection.send(&proto::process::Upstream {
            kind: Some(proto::process::upstream::Kind::Request(request)),
        })?;
        loop {
            let message = Self::receive(&mut self.connection, timeout, "a claim")?;
            match message.kind {
                Some(proto::process::downstream::Kind::Reply(reply)) if reply.id == id => {
                    return Ok(reply);
                }
                Some(proto::process::downstream::Kind::Reply(reply)) => {
                    tracing::warn!(
                        id = reply.id,
                        "an answer to a request this process did not send"
                    );
                }
                Some(proto::process::downstream::Kind::Event(event)) => self.file(event),
                None => {}
            }
        }
    }

    /// Sends one report of this process's state, which the daemon needs no answer to (§7.3).
    fn report(&mut self, report: proto::process::Report) -> Result<()> {
        self.connection.send(&proto::process::Upstream {
            kind: Some(proto::process::upstream::Kind::Report(report)),
        })?;
        Ok(())
    }

    /// The daemon's next command for this epoch, if one has arrived (§7.3).
    ///
    /// Nothing waits here: a command that has not arrived is one the next serve pass looks for.
    fn take_command(&mut self) -> Result<Option<proto::process::Command>> {
        // The daemon owns this process's life: the interface closing is the end of the epoch and not
        // an error, so it stops the loop by name instead of surfacing as a failure the daemon never
        // sees. A process that ends silently on a closed interface is how a run reads as a lost
        // engine with no side of its own in the log.
        loop {
            let message = match self.connection.try_recv::<proto::process::Downstream>() {
                Ok(Some(message)) => message,
                Ok(None) => break,
                Err(proto::Error::Closed) => {
                    tracing::info!("the daemon closed the process interface, so this process ends");
                    self.closed = true;
                    return Ok(None);
                }
                Err(error) => return Err(error.into()),
            };
            match message.kind {
                Some(proto::process::downstream::Kind::Event(event)) => self.file(event),
                Some(proto::process::downstream::Kind::Reply(reply)) => {
                    tracing::warn!(
                        id = reply.id,
                        "an answer to a request this process did not send"
                    );
                }
                None => {}
            }
        }
        Ok(self.pending.pop_front())
    }

    /// Whether the daemon closed the interface, which ends this process.
    const fn closed(&self) -> bool {
        self.closed
    }

    /// Files what the daemon sent unasked.
    fn file(&mut self, event: proto::process::Event) {
        match event.kind {
            Some(proto::process::event::Kind::Command(command)) => self.pending.push_back(command),
            // A group's exit is a worker's report; this process supervises no group.
            Some(proto::process::event::Kind::Exit(exit)) => {
                tracing::warn!(
                    group = %exit.group,
                    rank = exit.rank,
                    status = %exit.status,
                    "an exit report names a group this process does not supervise"
                );
            }
            None => {}
        }
    }
}

/// Asks this node's daemon for the node's Storage Server (§8).
///
/// The socket and the claim are runtime facts of the node: the daemon mints the claim when it starts
/// the server, so a launch file cannot carry either. The answer states the socket the server listens
/// on, the token this rank presents, and where another host reaches its pages.
fn claim(
    process: &mut ProcessInterface,
    launch: &Launch,
    host: &str,
    host_kv_cache: bool,
) -> Result<ipc::StorageLaunch> {
    let request = proto::process::Request {
        id: 1,
        kind: Some(proto::process::request::Kind::Claim(
            proto::process::Claim {
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
            },
        )),
    };
    let reply = process.request(request, CLAIM_TIMEOUT)?;
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
        // The host KV cache is the deployment's decision, from the build config: a rank whose
        // deployment keeps a host tier maps a page pool per cache group, and one that does not
        // maps nothing. Stating `false` for a deployment that asked for it would size the world's
        // host KV cache with no rank holding a page of it.
        host_kv_cache,
        tcp: (!claimed.tcp.is_empty()).then_some(claimed.tcp),
        // The deployment's *storage* peers - where another node's Storage Server is reached for the
        // weights a rank receives from it - are empty because nothing this process is told states
        // them: the claim's answer carries this node's own server only. A deployment of several
        // hosts needs them before a cross-host weight transfer can be planned, and until then this
        // rank maps what its own node holds (a named gap for 7.4, not a default).
        peers: Vec::new(),
        world: launch.world,
        // The standby role is the epoch layer's (§2.5, §7.4): a standby maps its scope's storage
        // read-only until it is promoted. Nothing the launch file states tells this process it is
        // one, and claiming writable storage as a standby would give the scope two writers - so the
        // role has to be stated before this process ever starts as a standby.
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
    .with_context(|| format!("reading the build config {}", launch.build_config.display()))?;
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
    /// This node's device executors, shared with the schedulers that submit through them: one
    /// executor thread per device, reached both by this process's own verbs (`precopy`,
    /// `switchover`, `warmup`, `purge`) and by the scheduler's `Batches`.
    executors: Vec<Arc<Executor>>,
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
    /// The warmups a switchover queued, one per device, with the generation each was queued for
    /// (§4.5). They are read on later passes rather than waited for: a warmup is a device job of its
    /// own, one bucket per queue turn behind serving, and a model with many buckets warms for longer
    /// than the daemon waits for `Switched`.
    warming: Vec<(u32, Submission, u64)>,
    /// How many switchovers this epoch has served (§7.3 step 3). The epoch's first one is what
    /// installs a policy, so only it is checked for a policy that never arrived.
    switchovers: u32,
    /// When the next check for a serving policy is due, and since when none serves (§7.3 step 3).
    ///
    /// `crates/engine`'s first switchover with no prepared candidate adopts no policy, calls
    /// `note_policy_serves` never, raises nothing and answers `Ok(None)`; a scheduler with no policy
    /// forms no batch, so the epoch would sit with nothing running while the daemon believes the
    /// cut-over succeeded. The check asks each scheduler's policy slot, which answers within its own
    /// command timeout, and reads the answer in the watchdog's stall bound rather than waiting.
    policy_probe_at: Option<Instant>,
    policy_missing_since: Option<Instant>,
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
    /// The daemon's process interface: the claim, the §7.3 reports and every command the daemon
    /// sends this epoch (§2.3).
    process: ProcessInterface,
    /// Set by the daemon's `Shutdown`: the epoch ends at the end of this pass (§2.3).
    stopping: bool,
    /// The admitted requests' event streams, by the daemon's request identity: the scheduler
    /// answers a submission with the stream of what the request produced, and this process is what
    /// carries those events back to the daemon (§2.3). A stream leaves when its request ends.
    /// Each admitted request's scheduler rank beside its stream: a command for the request has to
    /// reach the scheduler that serves it, and the command names the request, not the rank.
    streams: Vec<(u64, usize, engine::EventRx)>,
    /// Whether this process admits and executes: a drain parks every scheduler, and only the
    /// `Serve` that names the epoch again resumes it in place (§2.3, §7.4 step 2). A process that
    /// has not reached its first switchover admits nothing whatever the daemon has said: the epoch's
    /// first generation holds no kernel and no capture until the daemon's precopy fills it and the
    /// switchover warms it, and a batch admitted before then would reach a graph that does not
    /// exist. The boundary is necessary and not sufficient: the warmup queues one capture per
    /// device and runs them behind serving batches, so admission is owed to it as well
    /// ([`Self::open_admission`]).
    serving: bool,
    /// Whether the epoch's first switchover has happened: the boundary the first batch is served
    /// at (§4.5, §7.3 step 3).
    boundary: bool,
    /// A `Serve` that arrived before the boundary: the epoch may admit, but not yet.
    pending_serve: bool,
    /// Whether admission is owed to the warmup: the switchover that opens the epoch queues one
    /// capture per device and does not wait for them, so a batch admitted the moment the boundary
    /// opened could be planned into a bucket whose capture has not run - a replay finds no graph of
    /// it and answers an invariant error that names a capture bug. The epoch opens serving when the
    /// last submission finishes (`advance_warmup`), and a `Drain` drops what was owed (§4.5).
    admission_owed: bool,
    /// The launch's own request-slot count, which the daemon records as the epoch's capacity: the
    /// worker's row count includes the guidance prefix, which is not a request.
    max_running_requests: u32,
    /// Whether this process was launched as the epoch's standby (§2.5): it admits nothing until
    /// the daemon promotes it *and* the epoch reaches its boundary.
    standby: bool,
    /// Whether a promoted standby has been promoted: a standby that has not been stays silent
    /// through the boundary, and one that has follows the rule of any other process.
    promoted: bool,
    /// What each scheduler parked with, in `schedulers` order: kept from the drain to the resume,
    /// which adopts it, so a resumed epoch reuses the pages it still holds instead of reading them
    /// again.
    parked: Option<Vec<engine::ParkedCache>>,
    /// The frontend's admission gate, the `/admin/pause_generation` of §2.3: submissions are
    /// refused while it is held, and no request in flight is ended by it.
    paused: bool,
    /// The generations this process has released: a purge is a release of what it holds, so a
    /// second request for one of them is answered rather than run again.
    released: std::collections::BTreeSet<u64>,
    /// The probation an update must survive before it is committed, when the launch file states
    /// one (§7.3 step 4). The daemon decides the commit; this is the process's own guard against
    /// an update that is left on probation with nobody watching: it reverts at the threshold
    /// rather than serving a generation that was never accepted.
    rollback: Option<Duration>,
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
        // The world this rank belongs to is the daemon's, and both of its facts decide the storage
        // this process claims and the peers it exchanges products with. A zero world names no
        // world, and a peer list on a world of one is a second node this epoch does not have: each
        // is refused here, by name, rather than at the claim or the first transfer.
        if launch.world == 0 {
            bail!("the launch file states world 0, which names no world for this rank to serve");
        }
        if launch.world_size == 1 && !launch.peers.is_empty() {
            bail!(
                "the launch file states a world of one rank and {} peer address(es): a peer is \
                 another main process of the same world, and this epoch has no other",
                launch.peers.len()
            );
        }
        let epoch = handshake(&mut control, launch)?;
        // §7.3 step 1: the daemon's launch path blocks until this process states what it is, and it
        // sends the epoch's first program-layer command only once it has. `Starting` goes out
        // before the model load, which is minutes long, so the daemon's log and the caller's show a
        // process that is alive rather than one that vanished after the storage claim; the
        // `Ready`/`Standby` at the end of `start` is what lets the epoch begin.
        announce(
            &mut control,
            proto::request::wire::EngineState::Starting,
            None,
        )?;
        // The daemon's two interfaces are two sockets: the epoch's requests and metrics travel on
        // the request connection above, and everything the daemon asks of this process - the claim,
        // the reports of §7.3, and every command of the program layer - travels on this one.
        let mut process = ProcessInterface::connect(&launch.processes, &launch.deployment)?;

        // §8: the rank's Storage Server and its claim are the node's, which the daemon answers for;
        // the retained weights outlive every world of the epoch, so the store is built once, before
        // the model loads, and keeps what it maps for the epoch's life.
        // A host tier is the build config's, and it needs prefix caching to have pages to keep.
        let host_kv_cache = build.kv_cache.host_tier && build.kv_cache.prefix_caching;
        let storage = claim(&mut process, launch, &epoch.host, host_kv_cache)?;
        let client = Retained::connect(&storage, &launch.group, launch.rank, epoch.generation)
            .context("connecting to the Storage Server")?;
        // A weight's region is one device's address, so the store maps on the device this rank's
        // model is loaded for; the client is shared, the mapping is not.
        let weight_device = *plan
            .devices()
            .first()
            .context("this rank serves no device to map its weights on")?;
        let mut weights = WeightStore::new(Arc::clone(&client), weight_device);
        // One model per rank: every device's runner reads the same weights, so a second device costs
        // one `Arc` and not a second load. The load and the capacity plan are one call because the
        // order between them is the plan's: the weights are mapped first, so what the plan measures
        // is what is left of the device rather than what was there before the model.
        let (model, mut memory, inventory) = plan
            .load_and_open(&mut weights, &mut &*client)
            .context("loading the epoch's model")?;

        // The owners of §4 are opened before any thread that borrows them, and
        // released in reverse order by `close`. The weights are mapped before the capacity plan
        // measures, so the plan sees a device with the model already in it - the figure it computes
        // is what is left, not what was there before the load. Pools sized first and weights after
        // would over-commit the device by exactly the model's size.
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
        // Only the scratch figure is read here: the device thread starts before any batch and the
        // shadow region is the one pool it is sized with. The resident pool is the capacity plan's
        // and `open_memory` opens it, so this call states none.
        let scratch = plan.graph(device, 0)?.scratch;
        for device in plan.devices() {
            let generation = Arc::new(Generation::new(epoch.generation, std::iter::empty()));
            let runners = plan.runners(*device, Arc::clone(&model), &mut memory, &generation)?;
            let executor =
                Executor::spawn(*device, runners, plan.events(), Arc::clone(&pools), scratch)?;
            // §7.3 step 2 shadows a replaced operator beside the serving one, so the region it runs
            // in is mapped before any batch: the executor books it from the scratch pool the graph
            // storage sized. Nothing is serving yet, so the control thread waits here; `serve`
            // never does (§2.4).
            //
            // §4.5's warmup is not here: the epoch's first generation holds no kernel until the
            // daemon's precopy fills it - and a precopy that names no kernel (the startup switch)
            // fills it from the store's built programs - so there is no bucket to warm yet. It runs
            // at the first switchover, which is the boundary the first request is served at.
            let shadow = executor.set_shadow()?;
            device_command(&watchdog, *device, "shadow region", shadow)
                .with_context(|| format!("mapping the shadow region of device {}", device.0))?;
            executors.push(Arc::new(executor));
            generations.push(generation);
        }

        // This node's schedulers, one thread per DP rank, each handing its batches to the executors
        // above through `executor::Batches` (§2.4, §4.4). A node that holds none keeps an empty list
        // and serves only what its peers' schedulers ask of it; a node that is the scheduler node
        // and holds none would accept requests and dispatch nothing, which is refused by name.
        let schedulers = schedulers(&plan, launch, &memory, &model, &executors, &inventory)?;

        let mut process = Self {
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
            warming: Vec::new(),
            switchovers: 0,
            policy_probe_at: None,
            policy_missing_since: None,
            prepared: Vec::new(),
            retired: Vec::new(),
            generations,
            targets: plan.targets().to_vec(),
            program_server: plan.program_server().to_path_buf(),
            process,
            stopping: false,
            streams: Vec::new(),
            // Nothing is admitted until the epoch's first switchover, which is the boundary the
            // first batch is served at: before it the generation holds no kernel and no capture, so
            // a batch would find no graph. A standby parks itself here as well, and the `Promote`
            // that follows does not lift the boundary either (2.5, 7.4).
            serving: false,
            boundary: false,
            pending_serve: false,
            admission_owed: false,
            max_running_requests: u32::try_from(build.max_running_requests.get())
                .unwrap_or(u32::MAX),
            standby: launch.standby,
            promoted: false,
            parked: None,
            paused: false,
            rollback: launch.rollback_s.map(Duration::from_secs),
            released: std::collections::BTreeSet::new(),
        };
        if launch.standby {
            process.park_standby()?;
        }
        // What the daemon waits for before the epoch's first `Precopy`: a process started beside a
        // serving one is `Standby`, and one launched to serve is `Ready`. Waiting to be ready until
        // this process had a policy would be waiting for the command that waits for it.
        let state = if launch.standby {
            proto::request::wire::EngineState::Standby
        } else {
            proto::request::wire::EngineState::Ready
        };
        process.announce(state)?;
        Ok(process)
    }

    /// §2.5: a standby that has prepared announces itself and executes nothing until promoted.
    ///
    /// Its schedulers hold no request and have served no world, so parking them would ask for a
    /// cache handover that belongs to a drained epoch; what makes this a park is that nothing is
    /// admitted. `Promote` is the command that ends it.
    ///
    /// The report means exactly "prepared and admitted nothing", which is what the daemon gates
    /// its promotion on: it is sent after `start` has mapped the weights, opened the memory and
    /// started the runners, and *before* the epoch's first generation has been warmed - a warmup
    /// belongs to the first switchover, and a standby's own generation reaches it when the daemon
    /// sends the epoch's programs.
    fn park_standby(&mut self) -> Result<()> {
        self.serving = false;
        self.send_report(proto::process::Report {
            kind: Some(proto::process::report::Kind::Parked(
                proto::process::Parked {
                    generation: self.update.current(),
                },
            )),
        });
        Ok(())
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
                // The report is stated once more on the way out: an epoch's counters are additive
                // and a frontend that only saw the passes would lose the last second of them.
                self.send_metrics();
                return Ok(None);
            }
            // Two obligations the control thread has that no event announces: the counters' report,
            // and a probation running out. An idle epoch has no events at all, so a deadline naming
            // only the stall rule would leave both to whatever traffic arrived - the dashboard would
            // see nothing of a server nobody is asking, and a generation nobody accepted would sit
            // on probation past the window that reverts it (§7.3 step 4). Only a pending probation
            // adds its poll, so an idle epoch without one keeps waking for its counters alone.
            let stall = self.watchdog.deadline(oldest(&running));
            let mut due = reported + METRICS_EVERY;
            if self.rollback.is_some()
                && matches!(
                    self.update.phase(),
                    executor::hot_update::Phase::Probation { .. }
                )
            {
                due = due.min(Instant::now() + PROBATION_POLL);
            }
            let deadline = Some(stall.map_or(due, |stall| stall.min(due)));
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

            // What the admitted requests produced since the last pass.
            self.forward_events();

            // The daemon's commands for this epoch: the program layer's stages, the epoch layer's
            // park and serve, and the shutdown. They arrive on the process interface, so a pass
            // that did not look for them would never see an update (§7.3).
            while let Some(command) = self.process.take_command()? {
                if let Some(failure) = self.process_command(command)? {
                    return Ok(Some(failure));
                }
            }
            // The daemon owns this process's life: an interface it closed is the end of the epoch,
            // not a failure to report to it, so the pass stops by name and the process exits cleanly.
            if self.process.closed() {
                self.stopping = true;
            }

            // A precopy in flight advances one stage per pass, never waiting on
            // its builds or its devices (§7.3 step 2).
            self.advance_precopy()?;
            if let Some(failure) = self.advance_warmup()? {
                return Ok(Some(failure));
            }

            running = self.read_executors();
            if let Some(failure) = self.watch(&running)? {
                return Ok(Some(failure));
            }
            if let Some(failure) = self.watch_schedulers()? {
                return Ok(Some(failure));
            }
            if let Some(failure) = self.advance_policy_check(&running)? {
                return Ok(Some(failure));
            }
            if let Some(failure) = self.watch_probation()? {
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
                let reason = failure.reason();
                self.revert(generation, failure)?;
                // The revert answers the daemon itself; the failure that caused it is what the
                // revert's own report carries.
                tracing::info!(generation, %reason, "an update on probation was reverted");
                Ok(None)
            }
            Action::Epoch => {
                // The three failures that end the epoch's service are said before it ends: the
                // daemon replaces a stalled rank on the report, and a report that never arrives
                // leaves it reading a process that exited for no stated reason (§2.4, §2.5).
                self.report_ending(&failure);
                Ok(Some(failure))
            }
        }
    }

    /// States the failure that ends this epoch's service to the daemon (§2.4, §2.5, §7.3 step 4).
    ///
    /// A failure with no report of its own is left unsaid rather than given a shape it does not
    /// have; the process's exit still reaches the daemon.
    fn report_ending(&mut self, failure: &UpdateFailure) {
        use proto::process::report::Kind;

        let kind = match failure {
            UpdateFailure::Stalled { stall } => Kind::Stalled(proto::process::Stalled {
                generation: self.update.current(),
                batch: stall
                    .batch_id
                    .map_or_else(String::new, |batch| batch.to_string()),
                // The job's own name, as the executor classifies it: what stalled, not which
                // operator it was in.
                job: stall
                    .route
                    .clone()
                    .or_else(|| stall.method.clone())
                    .unwrap_or_default(),
                elapsed_ms: stall.running_ms,
            }),
            UpdateFailure::Device {
                device,
                error,
                batch: _,
            } => Kind::Fault(proto::process::Fault {
                kind: "sticky".to_owned(),
                reason: error.clone(),
                device: *device,
            }),
            UpdateFailure::NonFiniteLogits { request } => {
                Kind::NonFiniteLogits(proto::process::NonFiniteLogits {
                    request: format!("{request:?}"),
                })
            }
            _ => return,
        };
        self.send_report(proto::process::Report { kind: Some(kind) });
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
                for wire in submit.requests {
                    let id = wire.id;
                    // A drained epoch and a held gate admit nothing (§2.3): the refusal is retryable
                    // and the request is the frontend's to resubmit when the epoch serves again.
                    if !self.serving || self.paused {
                        self.reject(
                            id,
                            proto::request::RejectionKind::Overloaded,
                            if self.paused {
                                "the frontend's admissions are held".to_owned()
                            } else if !self.boundary {
                                "the epoch is starting and has reached no batch boundary yet"
                                    .to_owned()
                            } else {
                                "the epoch is drained and serves nothing".to_owned()
                            },
                        );
                        continue;
                    }
                    let request = match proto::request::Request::from_wire(wire) {
                        Ok(request) => request,
                        Err(error) => {
                            // A request that does not convert is the daemon's, not this process's:
                            // it ends with the refusal's event and every other request of the
                            // submission is still admitted.
                            self.reject(
                                id,
                                proto::request::RejectionKind::Invalid,
                                error.to_string(),
                            );
                            continue;
                        }
                    };
                    let rank = self.route(&request);
                    let Some(scheduler) = self.schedulers.get(rank) else {
                        bail!("a request was routed to rank {rank}, which holds no scheduler");
                    };
                    match scheduler.submit(request) {
                        Ok(stream) => {
                            self.streams.push((id, rank, stream));
                            self.send_event(proto::request::wire::from_engine::Kind::Accepted(
                                proto::request::wire::Accepted {
                                    request: id,
                                    scheduler: rank as u32,
                                },
                            ));
                        }
                        Err(error) => {
                            let kind = match error {
                                engine::SubmitError::Closed | engine::SubmitError::Dead => {
                                    proto::request::RejectionKind::Overloaded
                                }
                            };
                            self.reject(id, kind, error.to_string());
                        }
                    }
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
        // A revert carries the daemon's own account of why, which `ipc::ProgramCommand` does not
        // keep: read it here so the report repeats the operator's reason rather than inventing one.
        let reverted_by = match &command.kind {
            Some(proto::process::command::Kind::Revert(revert)) => {
                Some(revert.reason.clone()).filter(|reason| !reason.is_empty())
            }
            _ => None,
        };
        let Some(program) = ProgramCommand::from_command(command.clone(), current)? else {
            // The rest are this process's own lifecycle, and no scheduler's: sched's handle
            // answers every one of them with exactly that (a park, promote, drain, serve or pause
            // is the engine process's own command). Forwarding them there and ending the epoch on
            // the refusal is how a drain used to end the epoch it was meant to keep alive.
            match command.kind {
                Some(proto::process::command::Kind::Drain(drain)) => self.drain(drain.epoch)?,
                Some(proto::process::command::Kind::Serve(serve)) => self.resume(serve)?,
                Some(proto::process::command::Kind::Pause(pause)) => {
                    self.paused = pause.paused;
                }
                // A standby parks and is then promoted in place (§2.5). Both commands park and
                // resume the schedulers, and each has an answer the daemon's epoch layer waits
                // for: it promotes a child only after `Parked`, and switches the frontend only
                // after `Promoted`.
                Some(proto::process::command::Kind::Park(_)) => {
                    self.park_schedulers("parking for a promotion")?;
                    self.send_report(proto::process::Report {
                        kind: Some(proto::process::report::Kind::Parked(
                            proto::process::Parked {
                                generation: current,
                            },
                        )),
                    });
                }
                Some(proto::process::command::Kind::Promote(_)) => {
                    self.promoted = true;
                    // A promoted standby admits on the same rule as any other process: once the
                    // generation it will serve has been warmed (§4.5).
                    self.open_admission()
                        .context("promoting this process in place")?;
                    if !self.boundary {
                        // A promoted standby serves only once the switchover that fills its
                        // generation has happened, the same rule a fresh process follows.
                        self.serving = false;
                        self.pending_serve = true;
                    }
                    self.send_report(proto::process::Report {
                        kind: Some(proto::process::report::Kind::Promoted(
                            proto::process::Promoted {
                                generation: current,
                            },
                        )),
                    });
                }
                // The five program commands were converted above; a command that states no kind
                // states nothing to apply.
                _ => {}
            }
            return Ok(None);
        };
        match self.program_with(program, None, reverted_by)? {
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
        self.program_with(command, reply, None)
    }

    /// §7.3: one program-layer command, with the daemon's own reason when it stated one.
    ///
    /// A revert's reason is the daemon's account of why an operator ended the probation; it is
    /// carried into the epoch's record instead of a reason this process makes up. A peer's command
    /// states none, and the epoch's own failures state their own.
    fn program_with(
        &mut self,
        command: ProgramCommand,
        reply: Option<u32>,
        reverted_by: Option<String>,
    ) -> Result<Option<ProgramStatus>> {
        // §7.3's stages are the epoch's steps, and a step that never arrives is the failure this
        // line is for: it names the command and the generation, so a log that stops after "claimed
        // the deployment's storage" says which step the process is waiting for.
        tracing::info!(
            stage = program_stage(&command),
            boundary = self.boundary,
            serving = self.serving,
            warming = self.warming.len(),
            "a program-layer command arrived"
        );
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
            // The operator's rollback is the daemon's decision, and its reason - when the
            // daemon states one - is what the record keeps. It is not a policy failure: nothing
            // about the scheduling policy failed, and saying it did would put a policy error in
            // the epoch's record and in the daemon's account of why the update ended.
            ProgramCommand::Revert { generation } => self.revert(
                generation,
                UpdateFailure::Threshold {
                    reason: reverted_by.unwrap_or_else(|| "the daemon reverted the update".into()),
                },
            )?,
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

        // A `Precopy` that names no kernel program is the startup switch: the daemon states the
        // scheduling policy alone, because on a host that runs the worker path there is no kernel
        // package to prepare (`crates/daemon/src/leader.rs`, `install_programs`). This process
        // installs what it can actually launch, so an unnamed set is resolved from the store's
        // listing rather than captured into an empty generation.
        //
        // A policy-only *update* also names no kernel, and resolving there would be wrong twice
        // over: every operator would be re-installed as a new generation, so every kernel's `Arc`
        // would differ from the serving one and every captured bucket would be rebound or
        // recaptured for a change no program made. The two states are told apart by what this
        // process already serves - a startup has no kernel at all, an update has all of them.
        let never_installed = self
            .generations
            .iter()
            .all(|generation| generation.is_empty());
        let resolved;
        let mut candidates = Candidates::new();
        let programs = if programs.is_empty() && never_installed {
            let (programs, listed) = match self.listed_programs() {
                Ok(resolved) => resolved,
                Err(failure) => return self.abandon_precopy(generation, failure, reply),
            };
            tracing::info!(
                operators = programs.len(),
                alternative = listed.values().filter(|ordered| ordered.len() > 1).count(),
                "the precopy named no kernel program: installing this host's built programs"
            );
            candidates = listed;
            resolved = programs;
            &resolved
        } else {
            programs
        };

        let (tasks, unavailable) = match self.admit_installs(programs, &BTreeMap::new()) {
            Ok(admitted) => admitted,
            Err(failure) => return self.abandon_precopy(generation, failure, reply),
        };

        self.precopy = Some(Precopy {
            generation,
            reply,
            policy: policy.clone(),
            programs: programs.clone(),
            candidates,
            tried: std::collections::BTreeSet::new(),
            backends: BTreeMap::new(),
            unavailable,
            learned: BTreeMap::new(),
            policy_requested: false,
            next: Vec::new(),
            policies: Vec::new(),
            stage: PrecopyStage::Modules(tasks),
        });
        Ok(())
    }
    /// The Program Server's store as it stands: every installed package, its revision and the
    /// descriptors its modules cover for each target.
    fn listing(&self) -> std::result::Result<program_server::Listing, String> {
        let mut client = self
            .programs
            .lock()
            .map_err(|_| "the Program Server client was poisoned".to_owned())?;
        client
            .list()
            .map_err(|error| format!("reading the Program Server's store: {error}"))
    }

    /// The programs this host has built, one per operator, for a precopy that names none.
    ///
    /// The epoch's first generation has no serving generation to take keys from and a startup
    /// switch states the scheduling policy alone, so the engine resolves what it can launch: the
    /// store's listing is the authority, because its packages are the modules actually built for
    /// this host's targets.
    ///
    /// A program is chosen by **identity**, not by a specification: the operator's highest installed
    /// revision, ties broken by the backend's name so the choice is the same on every run. The
    /// listing carries no build arguments and none are needed - the store matches a program on
    /// `(operator, backend)` and fills the package's own parameter defaults, so `arguments` only ever
    /// overrides a default that the package already states.
    ///
    /// An operator built under more than one backend is one slot with several candidates, which is
    /// why the answer carries the whole ordered list: the first is installed, and the walk moves to
    /// the next one if the first cannot serve a descriptor it meets (`advance_candidate`). A
    /// deployment that wants one particular backend says so through the daemon, which names it in
    /// the `Precopy`, and then the walk does not choose at all.
    fn listed_programs(&self) -> std::result::Result<(Programs, Candidates), UpdateFailure> {
        let invalid = |reason: String| UpdateFailure::Precopy { reason };
        let listing = self.listing().map_err(invalid)?;
        let chosen = select_programs(&listing.packages, &self.targets);
        if chosen.is_empty() {
            return Err(invalid(format!(
                "the precopy named no kernel program and this host has built none: the store's \
                 listing holds {} package(s) and none has a module for this host's targets, so \
                 there is nothing to install",
                listing.packages.len()
            )));
        }
        tracing::info!(
            packages = listing.packages.len(),
            operators = chosen.len(),
            "resolved this host's built programs for a precopy that named none"
        );
        let mut programs = Programs::new();
        let mut candidates = Candidates::new();
        for (op, ordered) in chosen {
            let Some((backend, revision)) = ordered.first() else {
                continue;
            };
            programs.insert(
                program_server::program::ProgramSlot {
                    op: op.clone(),
                    phase: None,
                },
                Program {
                    backend: backend.clone(),
                    revision: *revision,
                    // The package's own defaults are the arguments the operator's modules were
                    // built with; the deployment overrides them by naming a program.
                    arguments: Default::default(),
                },
            );
            candidates.insert(op, ordered);
        }
        Ok((programs, candidates))
    }

    /// The module-load tasks one generation needs, one per device (§7.3 step 2).
    ///
    /// An update changes implementations and never the set of descriptors, so the keys each
    /// operator must cover are the ones the serving generation holds. The epoch's *first*
    /// generation has no serving generation to take them from, and nothing else can state them: a
    /// descriptor is what the model's forward computes for a bucket, and only a walk of that
    /// forward says which ones it launches. So the first generation starts with the keys already
    /// learned and finds the rest by walking - each attempt that finds a kernel missing names the
    /// operator and key it needed, and the install is admitted again with that key added.
    fn admit_installs(
        &mut self,
        programs: &Programs,
        learned: &BTreeMap<String, Vec<String>>,
    ) -> std::result::Result<(Vec<Arc<HostTask<Install>>>, UnavailableCell), UpdateFailure> {
        let invalid = |reason: String| UpdateFailure::Precopy { reason };
        // The Program Server's listing, read at most once and only when a program's keys cannot be
        // taken from a generation that already launched it: the epoch's first generation has
        // nothing to copy from, and the walk would otherwise discover one descriptor per capture.
        let mut listing: Option<program_server::Listing> = None;
        let mut descriptors = Vec::with_capacity(programs.len());
        let mut requests = Vec::with_capacity(programs.len());
        for (slot, program) in programs {
            let Some(operator) = ops::operator_name(&slot.op) else {
                return Err(invalid(format!(
                    "operator {} has no implementation in this build",
                    slot.op
                )));
            };
            let serving = descriptor_keys(&self.generations[0], operator);
            // A generation that launched this operator states the keys it runs; a key the walk has
            // taught since is added to them, never substituted for them. The first generation has
            // nothing serving, so its keys come from the package - once, not one capture at a time -
            // and a taught key joins them for the packages the store does not describe.
            let mut keys = serving;
            let mut known: Vec<String> = learned.get(operator).cloned().unwrap_or_default();
            if keys.is_empty() {
                if listing.is_none() {
                    let read = self.listing().map_err(invalid)?;
                    listing = Some(read);
                }
                // The slot was filled immediately above. Reading it through a match rather than a
                // panic keeps a fault in this function a named refusal, not a dead process: the
                // workspace denies `expect` for exactly that reason.
                let declared = match listing.as_ref() {
                    Some(store) => built_keys(store, &slot.op, program, &self.targets),
                    None => {
                        return Err(invalid(
                            "the Program Server's listing was not read".to_owned(),
                        ));
                    }
                };
                if declared.is_empty() {
                    // The listing names the modules this host has *built*, not what the package
                    // declares, so an empty answer is a host that built none yet - not a program
                    // that covers nothing. The walk is the authority on which descriptors the
                    // model needs, and it teaches them one capture at a time; the listing only
                    // makes the first capture enough when the store already holds them.
                    tracing::debug!(
                        op = %slot.op,
                        backend = %program.backend,
                        revision = program.revision,
                        "no module of this package is built yet: the walk will name its descriptors"
                    );
                }
                keys = declared;
            }
            keys.append(&mut known);
            keys.sort_unstable();
            keys.dedup();
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
        // One cell shared by every device's load of this generation: the first structured refusal
        // any of them meets is the one the walk answers with another candidate program.
        let unavailable = UnavailableCell::default();
        for (index, executor) in self.executors.iter().enumerate() {
            let ordinal = executor.device();
            let device = platform::Device::new(ordinal)
                .map_err(|error| invalid(format!("opening device {}: {error}", ordinal.0)))?;
            let Some(target) = self.targets.get(index).cloned() else {
                return Err(invalid(format!("device {} has no build target", ordinal.0)));
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
                unavailable: Arc::clone(&unavailable),
            };
            match self.modules.admit(install) {
                Ok(task) => tasks.push(task),
                Err(error) => {
                    return Err(invalid(format!("admitting a module load: {error}")));
                }
            }
        }
        Ok((tasks, unavailable))
    }

    /// Moves the generation on to the next program its operator may be filled by (§7.3 step 2).
    ///
    /// Each candidate this host built for the operator is tried once, in a stated order: the
    /// candidate whose built modules cover the most of the keys learned so far first, then the
    /// highest revision, then the backend's name - so two runs choose alike and a store with two
    /// backends costs one extra install rather than a failed generation. When every candidate has
    /// been tried the refusal names the operator, the descriptor it could not serve, and every
    /// backend tried; it never falls back silently and never speaks generically.
    ///
    /// A program the daemon named is the deployment's choice and is not second-guessed here: with no
    /// candidate list the walk answers [`Advance::Named`] and the refusal stands as it is.
    fn advance_candidate(&mut self, precopy: &mut Precopy, refused: &Unavailable) -> Advance {
        let op = refused.op().to_owned();
        let Some(ordered) = precopy.candidates.get(&op).cloned() else {
            return Advance::Named;
        };
        let Some((slot, current)) = precopy
            .programs
            .iter()
            .find(|(slot, _)| slot.op == op)
            .map(|(slot, program)| (slot.clone(), program.clone()))
        else {
            return Advance::Named;
        };
        precopy
            .tried
            .insert((current.backend.clone(), current.revision));
        let tried = precopy.backends.entry(op.clone()).or_default();
        if !tried.contains(&current.backend) {
            tried.push(current.backend.clone());
        }

        let listing = match self.listing() {
            Ok(listing) => listing,
            Err(reason) => return Advance::Failed(reason),
        };
        let learned = precopy.learned.get(&op).cloned().unwrap_or_default();
        let mut remaining: Vec<(usize, (String, u32))> = ordered
            .into_iter()
            .filter(|candidate| !precopy.tried.contains(candidate))
            .map(|candidate| {
                let program = Program {
                    backend: candidate.0.clone(),
                    revision: candidate.1,
                    arguments: Default::default(),
                };
                let covers = built_keys(&listing, &op, &program, &self.targets)
                    .iter()
                    .filter(|key| learned.contains(key))
                    .count();
                (covers, candidate)
            })
            .collect();
        // The stated order: coverage of the keys learned so far, then the highest revision, then the
        // backend name. Coverage only orders; it never excludes a candidate.
        remaining.sort_by(|left, right| {
            right
                .0
                .cmp(&left.0)
                .then_with(|| right.1 .1.cmp(&left.1 .1))
                .then_with(|| left.1 .0.as_str().cmp(right.1 .0.as_str()))
        });
        let Some((covers, next)) = remaining.into_iter().next() else {
            // Every candidate is spent: the refusal names the operator, what it could not serve and
            // every backend that was tried, so the next reader knows what to install.
            let backends = precopy.backends.get(&op).cloned().unwrap_or_default();
            let descriptor = refused.descriptor().unwrap_or("the key it was asked for");
            return Advance::Exhausted(format!(
                "{op} has no program this host built that serves {descriptor}: every candidate was \
                 tried (backends: {}), so this generation cannot be installed from this host's \
                 store",
                backends.join(", ")
            ));
        };
        tracing::info!(
            op = %op,
            from = %current.backend,
            to = %next.0,
            revision = next.1,
            covers,
            learned = learned.len(),
            "a program cannot serve this generation's shape: trying the operator's next candidate"
        );
        precopy.programs.insert(
            slot,
            Program {
                backend: next.0.clone(),
                revision: next.1,
                arguments: Default::default(),
            },
        );
        match self.admit_installs(&precopy.programs, &precopy.learned) {
            Ok((tasks, unavailable)) => {
                precopy.unavailable = unavailable;
                Advance::Retry(tasks)
            }
            Err(failure) => Advance::Failed(failure.reason()),
        }
    }

    /// Reads the warmups the switchovers queued (§4.5): a finished one leaves the list, and one
    /// that failed reverts the generation it was warming.
    ///
    /// A warmup is the first run of every captured bucket of the generation that just took over, so
    /// it is where a capture that cannot be made on this device surfaces - after the daemon was told
    /// the cut-over happened, which is why the failure goes through the update's decision table
    /// rather than through the switchover's answer.
    fn advance_warmup(&mut self) -> Result<Option<UpdateFailure>> {
        let mut pending = Vec::with_capacity(self.warming.len());
        for (device, submission, generation) in std::mem::take(&mut self.warming) {
            match submission.outcome() {
                None => pending.push((device, submission, generation)),
                Some(Outcome::Success(_)) => {
                    tracing::info!(device, generation, "a generation's warmup finished");
                }
                Some(outcome) => {
                    self.warming = pending;
                    let reason = match outcome {
                        Outcome::Failed(error) => error.to_string(),
                        _ => "the warmup was cancelled".to_owned(),
                    };
                    let failure = UpdateFailure::Device {
                        device,
                        error: format!(
                            "warming generation {generation} on device {device}: {reason}"
                        ),
                        batch: None,
                    };
                    // The table decides, as it does for every other failure: on probation this
                    // reverts the generation, and an update the daemon already committed leaves a
                    // device that cannot run it, which ends the epoch rather than serving on.
                    return self.failure(failure);
                }
            }
        }
        self.warming = pending;
        if self.warming.is_empty() && self.admission_owed {
            // The last capture of the epoch's warmup has run, so every bucket a batch can be planned
            // into exists. A warmup that failed returns above and its revert clears what was owed.
            self.open_admission()
                .context("opening admission after the warmup")?;
        }
        Ok(None)
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
                            // A program the store refused for a shape of this generation is not
                            // yet a failed generation: the operator may have another program this
                            // host built, and the walk tries each once before it gives up.
                            let refused = precopy
                                .unavailable
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner())
                                .clone();
                            if let Some(refused) = refused {
                                match self.advance_candidate(&mut precopy, &refused) {
                                    Advance::Retry(fresh) => {
                                        precopy.stage = PrecopyStage::Modules(fresh);
                                        self.precopy = Some(precopy);
                                        return Ok(());
                                    }
                                    Advance::Exhausted(reason) | Advance::Failed(reason) => {
                                        return self.abandon_precopy(
                                            generation,
                                            UpdateFailure::Precopy { reason },
                                            precopy.reply,
                                        );
                                    }
                                    Advance::Named => {}
                                }
                            }
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
                        // A walk that found a kernel missing names the operator and key it needed:
                        // that is how the epoch's first generation learns what it must install.
                        StageFailure::Failed(reason) => {
                            let Some((operator, key)) = missing_kernel(&submissions) else {
                                return self.abandon_precopy(
                                    generation,
                                    UpdateFailure::Precopy { reason },
                                    precopy.reply,
                                );
                            };
                            if precopy
                                .learned
                                .get(&operator)
                                .is_some_and(|keys| keys.contains(&key))
                            {
                                // The walk named a key this generation already installed, so no
                                // further attempt can succeed: the loop stops with both names
                                // rather than retrying a key it holds.
                                let cause = if precopy.programs.is_empty() {
                                    " - and the daemon named no kernel program for this \
                                     generation, so there is nothing to install it from (the \
                                     startup switch states the scheduling policy alone)"
                                } else {
                                    ""
                                };
                                return self.abandon_precopy(
                                    generation,
                                    UpdateFailure::Precopy {
                                        reason: format!(
                                            "{operator} still has no kernel under {key} after it \
                                             was installed: the walk and the install disagree{cause}"
                                        ),
                                    },
                                    precopy.reply,
                                );
                            }
                            tracing::info!(
                                %operator,
                                %key,
                                generation,
                                "the walk names a descriptor key this generation learns"
                            );
                            precopy.learned.entry(operator).or_default().push(key);
                            let tasks =
                                match self.admit_installs(&precopy.programs, &precopy.learned) {
                                    Ok((tasks, unavailable)) => {
                                        precopy.unavailable = unavailable;
                                        tasks
                                    }
                                    Err(failure) => {
                                        return self.abandon_precopy(
                                            generation,
                                            failure,
                                            precopy.reply,
                                        );
                                    }
                                };
                            precopy.stage = PrecopyStage::Modules(tasks);
                            self.precopy = Some(precopy);
                            return Ok(());
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
        //
        // Once per precopy, not once per install: the descriptor discovery loop comes back through
        // here with a larger key set, and a policy build on each pass would certify the same
        // program again on the scheduler's blocking thread, minutes at a time, while the update
        // waits for a generation that only needs another kernel.
        let policies = if precopy.policy_requested {
            Vec::new()
        } else {
            match self.request_policy(&precopy.policy) {
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
            }
        };

        // Rebind or recapture, as jobs between serving batches (§4.5).
        let mut submissions = Vec::with_capacity(next.len());
        for (index, built) in next.iter().enumerate() {
            submissions.push(self.executors[index].precopy(Arc::clone(built))?);
        }
        let mut precopy = precopy;
        precopy.next = next;
        if !policies.is_empty() {
            precopy.policies = policies;
            precopy.policy_requested = true;
        }
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
        // that never serves must give its captures back (§7.3 step 2). A purge names a generation a
        // switchover or a revert released, and this one was never cut over, so it is discarded: the
        // executor drops exactly this generation's graphs and kernels. A device still busy with the
        // capture is answered by the queue's own order.
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
        // A switchover to a generation this process never prepared cannot succeed, and cutting the
        // devices over to try it costs a live graph rebuild to end where it started: it is refused
        // before any device is touched, and the generation that serves keeps serving.
        //
        // "Prepared nothing" and "prepared another" are the same refusal to the daemon and two
        // different bugs in the log, so the reason names which one. The second is the state a lost
        // `Precopied` reply leaves: the daemon gave up on a precopy that then completed, retried
        // `/admin/config`, and now names a generation this process never built. Letting it through
        // would move the scheduling policy to `generation` first, then fail on the devices, and
        // report a device fault for a stale number.
        let stale = self
            .prepared
            .iter()
            .map(|built| built.number())
            .find(|number| *number != generation);
        let refused = if self.prepared.is_empty() {
            Some(format!(
                "a switchover to generation {generation}, which this process prepared nothing \
                 for: the precopy that builds it either failed or never arrived"
            ))
        } else {
            stale.map(|other| {
                format!(
                    "a switchover to generation {generation}, while this process holds generation \
                     {other} prepared and never cut over: the precopy that builds {generation} \
                     either failed or never arrived"
                )
            })
        };
        if let Some(reason) = refused {
            return Ok(ProgramStatus::PrecopyFailed {
                generation,
                failure: UpdateFailure::Precopy { reason },
            });
        }
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
                return self.revert(generation, failure);
            }
        }
        // The record moves before the warmup: probation begins the moment every device serves the
        // new generation, and the warmup is the last part of the cut-over rather than of the
        // generation that served before it. So a warmup that fails is a revert of a generation that
        // *served* - a reverse switchover on every device - and not a discard of one that never
        // cut over, which is what the phase decides, and it cannot be asked before this line.
        self.update.switchover(generation, Instant::now())?;
        self.retired = std::mem::replace(&mut self.generations, std::mem::take(&mut self.prepared));

        // §4.5: the first request served on a generation must not pay for a capture, so every
        // captured bucket of the generation that just took over runs once, here, between batches.
        // A switchover later in the epoch warms the buckets the new kernel programs changed; the
        // first one also warms the buckets no update touched, because until now nothing served
        // them.
        // The warmups are queued, not waited for: each is a device job that runs one bucket per
        // queue turn behind serving batches, so a model with many buckets warms for minutes while
        // the daemon waits two minutes for `Switched` - and a control thread that waited here would
        // answer no command, no request event and no metrics for the same minutes. `advance_warmup`
        // reads them on the passes that follow, and a device that fails reverts the update.
        for executor in &self.executors {
            let device = executor.device();
            let ordinal = u32::try_from(device.0).unwrap_or_default();
            match executor.warmup() {
                Ok(submission) => self.warming.push((ordinal, submission, generation)),
                Err(error) => {
                    let failure = UpdateFailure::Device {
                        device: ordinal,
                        error: format!("queueing the warmup of generation {generation}: {error}"),
                        batch: None,
                    };
                    if let Some(ending) = self.failure(failure)? {
                        return Ok(ProgramStatus::Failed {
                            generation,
                            failure: ending,
                        });
                    }
                    return Ok(ProgramStatus::Switched { generation });
                }
            }
        }

        tracing::info!(
            generation,
            devices = self.executors.len(),
            queued = self.warming.len(),
            "the switchover queued the epoch's warmup and set the boundary"
        );
        // The boundary the first batch is served at (§4.5): every device serves this generation
        // and its buckets are warming behind the batches that follow. A `Serve` that arrived while
        // the generation was still being filled is honoured now.
        self.boundary = true;
        // An epoch admits once this boundary is behind it and nothing has parked it since: a
        // non-standby process was launched to serve, a standby only after the daemon promoted it,
        // and a drained one waits for the `Serve` that names the epoch again. The start of an epoch
        // carries no `Serve` - the daemon tells the frontend where requests go and the process was
        // launched serving - so the boundary itself is what opens admission.
        let admitted = self.pending_serve || self.parked.is_none();
        if admitted && (self.promoted || !self.standby) {
            // Admission is the end of the warmup, not the boundary: the buckets the epoch serves
            // are the ones the warmup captures, and a batch admitted before the last of them has
            // run and been captured would be planned into a bucket with no graph. So the answer to
            // the daemon leaves `Switched` inside its window and the opening happens in
            // `advance_warmup`, which is what honours an early `Serve` when a drain parked the
            // schedulers first - a process that never parked resumes nothing and simply admits.
            self.open_admission()
                .context("resuming for the epoch's first switchover")?;
        }
        // The other silence of §7.3 step 3: a first switchover that installed no policy adopts
        // nothing and says nothing, so this process asks the policy slots once and, if none serves,
        // asks again one stall bound later before it names the failure. The check is queued here
        // rather than run here, so the daemon's `Switched` is never delayed by it.
        self.switchovers = self.switchovers.saturating_add(1);
        if self.switchovers == 1 {
            self.policy_probe_at = Some(Instant::now());
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

    /// §2.3, §7.4 step 2: the epoch drains at the boundary its schedulers reach, and the process
    /// goes on living.
    ///
    /// Every scheduler parks: its migratable requests end with their facts kept, the others finish,
    /// and it executes nothing more. What it parks with is kept, so the `Serve` that names the epoch
    /// again resumes it in place instead of reading its pages back. Weights and storage stay mapped
    /// and the threads stay up - a drain is not an exit, and the daemon's `Serve` is what ends it.
    fn drain(&mut self, epoch: u64) -> Result<()> {
        // An update cannot cross the drain: its switchover takes effect at a batch boundary and a
        // drained epoch reaches none, so what the precopy built is given back now rather than left
        // prepared for a boundary that cannot come.
        self.drop_prepared("the epoch drained before its switchover")?;
        self.park_schedulers("draining the epoch")?;
        self.send_report(proto::process::Report {
            kind: Some(proto::process::report::Kind::Drained(
                proto::process::Drained { epoch },
            )),
        });
        Ok(())
    }

    /// Gives back what an in-flight update prepared, by name.
    ///
    /// A precopy still running is abandoned through its own path, which discards the captures on
    /// every device, reinstates the serving policy and answers the peer or the daemon that asked.
    /// A generation already prepared is discarded the same way: the device queue orders the discard
    /// after the walk that captured it, and the generation that serves keeps its plans.
    fn drop_prepared(&mut self, reason: &str) -> Result<()> {
        if let Some(precopy) = self.precopy.take() {
            let generation = precopy.generation;
            let reply = precopy.reply;
            self.precopy = Some(precopy);
            return self.abandon_precopy(
                generation,
                UpdateFailure::Precopy {
                    reason: reason.to_owned(),
                },
                reply,
            );
        }
        let Some(generation) = self.prepared.first().map(|built| built.number()) else {
            return Ok(());
        };
        self.discard_policy();
        for executor in &self.executors {
            match executor.discard(generation) {
                Ok(submission) => {
                    if let Err(error) =
                        device_command(&self.watchdog, executor.device(), "discard", submission)
                    {
                        // The generation never serves: a discard that did not finish holds storage
                        // until the purge, and the epoch that drained is not made to fail for it.
                        tracing::warn!(%error, generation, "discarding a prepared generation failed");
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, generation, "queueing a prepared generation's discard failed");
                }
            }
        }
        self.prepared.clear();
        Ok(())
    }

    /// §2.3: epoch `serve.epoch` serves from now on, at this process.
    ///
    /// A process that drained resumes the schedulers it parked, with the cache it parked with; one
    /// that never drained already serves and has nothing to resume. The request address in the
    /// command is the one this process bound at launch: the daemon names it so that the frontend
    /// submits to the right place, not so that a process moves its socket.
    fn resume(&mut self, serve: proto::process::Serve) -> Result<()> {
        if !self.boundary {
            // The daemon has said the epoch serves; the process cannot admit until its first
            // generation is filled and warmed. The request waits in the frontend, which retries the
            // submission it was refused.
            self.pending_serve = true;
            return Ok(());
        }
        // A `Serve` that arrives while the warmup still has to capture this generation's buckets is
        // owed rather than refused: admission opens when the last submission finishes, the same way
        // it does at a switchover (§4.5).
        self.open_admission()
            .with_context(|| format!("resuming epoch {}", serve.epoch))
    }

    /// Opens admission, or owes it to the warmup (§4.5).
    ///
    /// A bucket is served only after its capture has run: a replay reads the captures of the
    /// generation serving, and a bucket captured by no one is an invariant error at the first batch
    /// rather than a capture that is slow. `warmup` queues one capture of every bucket per device
    /// and returns at once - it runs one bucket per queue turn behind serving, which for a model
    /// with many buckets is minutes - so the switchover answers the daemon and leaves the opening
    /// here. A `Drain` clears what was owed, because the epoch it was owed to is no longer serving.
    fn open_admission(&mut self) -> Result<()> {
        if !self.warming.is_empty() {
            tracing::info!(
                warming = self.warming.len(),
                "admission is owed to the warmup, which has not finished"
            );
            self.admission_owed = true;
            return Ok(());
        }
        tracing::info!(
            boundary = self.boundary,
            pending_serve = self.pending_serve,
            "opening admission"
        );
        self.admission_owed = false;
        // An early `Serve` is honoured here rather than at its arrival: this is the pass that
        // admits, so the request waiting in the frontend is exactly what it was waiting for.
        self.pending_serve = false;
        self.resume_schedulers()
    }

    /// §7.3 step 4: an update left on probation past the window the launch file states is
    /// reverted, so a deployment never serves a generation nobody accepted.
    ///
    /// The daemon commits; this is the guard for when it does not - an operator who closed the
    /// dashboard, a commit that never arrived. Without a stated window there is no guard, which is
    /// the daemon's decision and not this process's.
    fn watch_probation(&mut self) -> Result<Option<UpdateFailure>> {
        let Some(window) = self.rollback else {
            return Ok(None);
        };
        if !self.update.probation_expired(Instant::now(), window) {
            return Ok(None);
        }
        let failure = UpdateFailure::Threshold {
            reason: format!(
                "the update was on probation for longer than the {} seconds the launch file allows",
                window.as_secs()
            ),
        };
        tracing::info!(
            seconds = window.as_secs(),
            "an update outlived its probation and is reverted"
        );
        // A device that cannot put the generation that served back is a device with no generation at
        // all: the revert's refusal is the epoch's, and it ends by that name rather than continuing to
        // serve a generation the process has just decided it cannot stand behind.
        let generation = self.update.current();
        self.revert(generation, failure)?;
        Ok(None)
    }

    /// §7.3 step 3: a first switchover that installed no policy, said by name instead of waited for.
    ///
    /// A first switchover with no prepared candidate adopts no policy and answers `Ok(None)`; a
    /// scheduler with no policy forms no batch, so the epoch sits with nothing running while the
    /// daemon believes the cut-over succeeded - the same silence the missing startup report had.
    /// The policy slot is asked whether one serves, and the answer is read rather than waited for:
    /// a probe is due at once after the switchover and again one stall bound later, and the failure
    /// is only spoken when no device has a job in flight, so a busy scheduler cannot be mistaken
    /// for one that never adopted a policy.
    fn advance_policy_check(&mut self, running: &[Running]) -> Result<Option<UpdateFailure>> {
        let Some(due) = self.policy_probe_at else {
            return Ok(None);
        };
        let now = Instant::now();
        if now < due {
            return Ok(None);
        }
        self.policy_probe_at = None;

        let serving = self
            .schedulers
            .iter()
            .filter(|scheduler| scheduler.policy_mut().running().is_some())
            .count();
        if serving == self.schedulers.len() {
            if self.policy_missing_since.take().is_some() {
                tracing::info!(serving, "the switchover's policy serves every scheduler");
            }
            return Ok(None);
        }
        if !running.is_empty() {
            // A device is busy, so the slots' silence is not evidence: look again next pass.
            self.policy_probe_at = Some(now + POLICY_PROBE_INTERVAL);
            return Ok(None);
        }
        match self.policy_missing_since {
            None => {
                self.policy_missing_since = Some(now);
                self.policy_probe_at = Some(now + self.watchdog.stall_bound());
                tracing::warn!(
                    serving,
                    schedulers = self.schedulers.len(),
                    "a switchover installed no policy on every scheduler; looking again in the watchdog's stall bound"
                );
                Ok(None)
            }
            Some(since) => {
                self.policy_missing_since = None;
                let failure = UpdateFailure::Policy {
                    error: format!(
                        "the epoch's first switchover installed no policy: {} of this node's {} \
                         schedulers serve one after {}s, and a scheduler with no policy forms no \
                         batch, so an admitted request cannot be planned",
                        serving,
                        self.schedulers.len(),
                        since.elapsed().as_secs()
                    ),
                };
                self.failure(failure)
            }
        }
    }

    /// A scheduler thread that stopped ends the epoch's service, and the epoch says so.
    ///
    /// With no scheduler no batch is formed, so no job runs and no job can stall: the stall rule
    /// reports nothing and every request waits for a batch that will never be planned. The reason
    /// the thread stopped is its own - a policy that trapped or ran out of fuel is the usual one
    /// (§4.4) - and the update's decision table decides whether that ends the epoch or reverts the
    /// update on probation (§7.3 step 4).
    fn watch_schedulers(&mut self) -> Result<Option<UpdateFailure>> {
        for (rank, scheduler) in self.schedulers.iter().enumerate() {
            if !scheduler.is_dead() {
                continue;
            }
            let failure = UpdateFailure::Policy {
                error: scheduler
                    .failure()
                    .unwrap_or_else(|| format!("the scheduler of rank {rank} stopped")),
            };
            return self.failure(failure);
        }
        Ok(None)
    }

    /// Parks every scheduler, keeping what each parks with for the resume.
    ///
    /// Each park is bounded by the watchdog's stall bound, the same rule the scheduler stage calls
    /// follow (§2.4): a scheduler that cannot park is a stall the epoch has to name rather than a
    /// control thread that waits forever.
    fn park_schedulers(&mut self, what: &str) -> Result<()> {
        let bound = self.watchdog.stall_bound();
        let mut parked = Vec::with_capacity(self.schedulers.len());
        for scheduler in &self.schedulers {
            match scheduler.park(false).recv_timeout(bound) {
                Ok(Ok(cache)) => parked.push(cache),
                Ok(Err(reason)) => bail!("{what}: a scheduler did not park: {reason}"),
                Err(error) => bail!("{what}: a scheduler did not park within {bound:?}: {error}"),
            }
        }
        self.parked = Some(parked);
        self.serving = false;
        // A drain ends the epoch this process was serving, so admission owed to a warmup of it is
        // owed to nothing: the `Serve` that names the epoch again opens the next one.
        self.admission_owed = false;
        Ok(())
    }

    /// Resumes every scheduler with the cache it parked with: the pages it still holds are reused
    /// and the facts the drain kept are resubmitted by the frontend (§2.3).
    fn resume_schedulers(&mut self) -> Result<()> {
        let Some(parked) = self.parked.take() else {
            self.serving = true;
            return Ok(());
        };
        for (scheduler, cache) in self.schedulers.iter().zip(parked) {
            // A resumed epoch recomputes from the tokens the frontend resubmits rather than lending
            // from a predecessor: it is the same world, and the requests it ended at the drain are
            // new requests to it.
            scheduler
                .promote(cache, true, false)
                .context("resuming a scheduler")?;
        }
        self.serving = true;
        Ok(())
    }

    /// §7.3 step 4: revert to the generation serving before the update. The old
    /// generation and its nodes stayed through probation, so the revert is the
    /// reverse switchover and nothing has to be prepared again.
    fn revert(&mut self, generation: u64, failure: UpdateFailure) -> Result<ProgramStatus> {
        self.policy_probe_at = None;
        self.policy_missing_since = None;
        // The phase before the call decides what a device does with the update's generation: a
        // generation that was prepared but never cut over never served, so nothing goes back and it
        // is discarded; a generation on probation served, and its reverse switchover runs. The
        // machine answers `Serving` either way once the call returns, so the phase is read first.
        let phase = self.update.phase();
        if let Err(error) = self.update.revert(generation, failure.clone()) {
            tracing::error!(%error, "reverting a program-layer update failed");
            return Ok(ProgramStatus::Failed {
                generation,
                failure,
            });
        }
        self.revert_policy(generation);
        // The generation that served before the update serves again, and the one built for the
        // update keeps nothing: its kernels are released and its captures are never taken over.
        let kept = !self.retired.is_empty();
        if kept {
            self.generations = std::mem::take(&mut self.retired);
        }
        self.prepared.clear();
        // The warmups still queued warm the generation this revert went back from: what runs when
        // they reach the device is the generation that serves, so nothing hangs - the list is what
        // stopped describing serving, and a failure read from a stale entry would revert an update
        // that is over.
        self.warming.clear();
        // What the update's warmup still owed is owed no longer. An epoch that kept the generation
        // which served before it goes on serving, and that generation's buckets are captured - a
        // cancelled warmup of the update's is nothing to wait for. A revert of the epoch's first
        // switchover keeps nothing, and its end is the report rather than an admission.
        let owed = std::mem::take(&mut self.admission_owed);
        if owed && kept && self.boundary && (self.promoted || !self.standby) {
            self.resume_schedulers()
                .context("resuming the generation a revert went back to")?;
        }
        let current = self.update.current();
        for executor in &self.executors {
            let device = executor.device();
            // §7.3 step 4's revert is the reverse switchover, and `Executor::revert` is the verb
            // that runs it: a `switchover` back to the kept generation would find nothing prepared
            // (the cut-over took `next`) and refuse by name - the current generation would be left
            // serving with no kernels and no graphs.
            // A device that cannot go back to the generation that served is a device with no
            // generation at all: the update's kernels are the ones its graphs were rebound to, and a
            // revert that half happened leaves the epoch serving from neither. So a failed revert
            // ends the epoch by name instead of leaving it to fail on a batch that finds no graph.
            let queued = match phase {
                // §7.3 step 2's discard: the generation's graphs and kernels go, and the generation
                // that serves is untouched. A reverse switchover here would be refused by name -
                // nothing was switched over - and would end the epoch over an update that never cut.
                executor::hot_update::Phase::Precopied { .. } => executor.discard(generation),
                _ => executor.revert(current),
            };
            let submission = queued.with_context(|| {
                format!(
                    "queueing device {}'s revert to generation {current}",
                    device.0
                )
            })?;
            device_command(&self.watchdog, device, "revert", submission)?;
        }
        // No purge follows a revert: the daemon purges the generation a *commit* released, and its
        // reverted path sends nothing (§7.3 step 5 and 6). The generation that served and went back
        // would otherwise keep its graphs, its kernels and its arena until the process exits - a
        // whole generation of device memory - so it is released here, now that every device has
        // finished going back and no batch of it can be in flight.
        if matches!(phase, executor::hot_update::Phase::Probation { .. }) {
            match self.purge(generation) {
                Ok(ProgramStatus::Purged { .. }) => {}
                Ok(other) => tracing::warn!(
                    ?other,
                    "the generation the revert went back from was not released"
                ),
                Err(error) => {
                    // The revert stands: serving is correct and what a failed release costs is the
                    // reverted generation's storage, which the process keeps until it exits.
                    tracing::error!(%error, "releasing the generation a revert went back from failed");
                }
            }
        }
        Ok(ProgramStatus::Reverted {
            generation,
            reason: failure.reason(),
        })
    }

    /// §7.3 step 6: release the generation before the serving one, once its last
    /// batch completed. The executor's queue orders the purge behind every batch
    /// it already holds, and `Captured` keeps each kernel alive until the graph
    /// that uses it is rebound or destroyed.
    fn purge(&mut self, generation: u64) -> Result<ProgramStatus> {
        // A purge releases what this process holds of a generation, so a number it has already
        // released is answered rather than refused: the revert below releases a reverted generation
        // itself, and a daemon that later purges the same number asks for a release that happened.
        if self.released.contains(&generation) {
            return Ok(ProgramStatus::Purged { generation });
        }
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
        self.released.insert(generation);
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
        // A command names its request, not the scheduler serving it, so the rank is read from the
        // stream the submission answered with. A request that is not in flight here has already
        // ended: a cancel or a stop for it is a no-op, not a fault, and ending an epoch because a
        // client disconnected a moment too late would be the daemon's most common message.
        let Some((_, rank, _)) = self
            .streams
            .iter()
            .find(|(id, _, _)| *id == command.request)
        else {
            tracing::debug!(
                request = command.request,
                "a request command names a request this epoch no longer serves"
            );
            return Ok(());
        };
        let Some(scheduler) = self.schedulers.get(*rank) else {
            tracing::debug!(
                request = command.request,
                rank,
                "a request command names a rank this node does not hold"
            );
            return Ok(());
        };
        answer(
            scheduler.request_command(command),
            self.watchdog.stall_bound(),
            "a request command",
        )
    }

    /// A command to the whole main process: shutdown, drain, park or promote,
    /// and the program-layer commands a single-rank node receives directly.
    fn engine_command(
        &mut self,
        command: proto::request::wire::EngineCommand,
    ) -> Result<Option<UpdateFailure>> {
        // §2.3: the epoch's lifecycle - Park, Promote, Serve, Drain, Pause, the program layer's
        // stages - arrives on the process connection and is applied by the driver. On this
        // connection the epoch owns exactly one command, `Shutdown`: every request finishes as
        // aborted and the owners close in the order they need (`close`).
        //
        // The rest are the engine surface of the worker era, and the daemon's migration, expert
        // and admin paths still send them here. Each is answered with what it needs instead of
        // being dropped: a command a process does not serve is otherwise a caller waiting for a
        // reply that never comes, which is a hang with no name in it.
        let refused = match command.kind {
            Some(proto::request::wire::engine_command::Kind::Shutdown(_)) => {
                self.stopping = true;
                return Ok(None);
            }
            Some(proto::request::wire::engine_command::Kind::Hook(hook)) => Some(
                proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                    id: hook.id,
                    results: Vec::new(),
                    error: WORKER_ERA.to_owned(),
                }),
            ),
            Some(proto::request::wire::engine_command::Kind::ResizeCache(resize)) => Some(
                proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                    id: resize.id,
                    results: Vec::new(),
                    error: WORKER_ERA.to_owned(),
                }),
            ),
            Some(proto::request::wire::engine_command::Kind::UpdateWeights(update)) => Some(
                proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                    id: update.id,
                    results: Vec::new(),
                    error: WORKER_ERA.to_owned(),
                }),
            ),
            Some(proto::request::wire::engine_command::Kind::CommitExperts(commit)) => Some(
                proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                    id: commit.id,
                    results: Vec::new(),
                    error: WORKER_ERA.to_owned(),
                }),
            ),
            Some(proto::request::wire::engine_command::Kind::CacheWarm(warm)) => Some(
                proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                    id: warm.id,
                    results: Vec::new(),
                    error: WORKER_ERA.to_owned(),
                }),
            ),
            // The worker era's park, resume and trial commit move a worker world between
            // processes. The epoch parks, resumes and commits through the process connection, in
            // front of the frontend rather than behind it (§2.5, §7.4).
            Some(proto::request::wire::engine_command::Kind::Park(park)) => Some(
                proto::request::wire::from_engine::Kind::Parked(proto::request::wire::Parked {
                    id: park.id,
                    error: EPOCH_LIFECYCLE.to_owned(),
                    index: Vec::new(),
                }),
            ),
            Some(proto::request::wire::engine_command::Kind::Resume(resume)) => Some(
                proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                    id: resume.id,
                    results: Vec::new(),
                    error: EPOCH_LIFECYCLE.to_owned(),
                }),
            ),
            Some(proto::request::wire::engine_command::Kind::Commit(commit)) => Some(
                proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                    id: commit.id,
                    results: Vec::new(),
                    error: EPOCH_LIFECYCLE.to_owned(),
                }),
            ),
            // The scheduling limits are the schedulers', applied at their next step through the
            // policy slot the process connection names (§4.4).
            Some(proto::request::wire::engine_command::Kind::Configure(configure)) => {
                Some(proto::request::wire::from_engine::Kind::Configured(
                    proto::request::wire::Configured {
                        id: configure.id,
                        error: EPOCH_LIFECYCLE.to_owned(),
                    },
                ))
            }
            // The storage ledgers and the transfer budgets are each node's daemon's, read over
            // its claim (§8); an epoch's main process holds no claim of the deployment's.
            Some(proto::request::wire::engine_command::Kind::DescribeStorage(describe)) => Some(
                proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                    id: describe.id,
                    results: Vec::new(),
                    error: LEDGERS.to_owned(),
                }),
            ),
            Some(proto::request::wire::engine_command::Kind::ThrottleTransfers(throttle)) => Some(
                proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                    id: throttle.id,
                    results: Vec::new(),
                    error: LEDGERS.to_owned(),
                }),
            ),
            // What the prefix cache holds is the schedulers' answer, and they are asked through
            // the process connection.
            Some(proto::request::wire::engine_command::Kind::DescribeCache(describe)) => {
                Some(proto::request::wire::from_engine::Kind::CacheDescribed(
                    proto::request::wire::CacheDescribed {
                        id: describe.id,
                        error: EPOCH_LIFECYCLE.to_owned(),
                        segments: Vec::new(),
                    },
                ))
            }
            Some(proto::request::wire::engine_command::Kind::DescribeRequests(describe)) => {
                Some(proto::request::wire::from_engine::Kind::RequestsDescribed(
                    proto::request::wire::RequestsDescribed {
                        id: describe.id,
                        error: EPOCH_LIFECYCLE.to_owned(),
                        requests: Vec::new(),
                    },
                ))
            }
            // Promote, FlushCache and Attach state no id and are answered with a state this
            // process does not own: there is nothing to answer on, and a log line is what is
            // left.
            Some(other) => {
                tracing::warn!(
                    "the daemon sent an engine command this epoch does not serve: {}",
                    engine_command_name(&other)
                );
                None
            }
            None => None,
        };
        if let Some(kind) = refused {
            self.send_event(kind);
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
        // The schedulers state their own figures; a node that holds none reports the executors'
        // and a step count of zero, because no step happened here.
        let (mut steps, mut batches, mut tokens, mut policy_fuel, mut policy_calls) =
            (0, 0, 0, 0, 0);
        for scheduler in &self.schedulers {
            // A scheduler answers its counters on its own thread: a metrics report is not worth
            // stalling the control thread for, so the pass takes what is there.
            let Ok(counters) = scheduler.counters().recv_timeout(METRICS_WAIT) else {
                continue;
            };
            steps += counters.steps;
            batches += counters.batches;
            tokens += counters.tokens;
            policy_fuel += counters.policy_fuel;
            policy_calls += counters.policy_calls;
            // The launches and the device time are the executors', counted where they are issued:
            // a scheduler's copy of them would be a second source for one figure.
        }
        let metrics = proto::request::wire::Metrics {
            steps,
            batches,
            tokens,
            kernel_launches,
            device_ns,
            queue,
            policy_fuel,
            policy_calls,
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
    fn report(&mut self, status: ProgramStatus) {
        // §7.3's answers, in the vocabulary the daemon's epoch record reads: a precopy that
        // finished, the switchover, the commit, the revert and the purge, each naming its
        // generation, and a failure carrying the first error. The daemon needs no answer to any of
        // them; it marks the epoch's state with what it is told, and the probation's rules are
        // applied on these reports.
        let Some(report) = report_of(&status) else {
            tracing::debug!(?status, "a program-layer status needs no report");
            return;
        };
        self.send_report(report);
    }

    /// Sends one message of the request interface to the daemon (§2.3).
    fn send_event(&mut self, kind: proto::request::wire::from_engine::Kind) {
        if let Err(error) = self
            .control
            .daemon_mut()
            .send(&proto::request::wire::FromEngine { kind: Some(kind) })
        {
            tracing::warn!(%error, "sending an event of an admitted request failed");
        }
    }

    /// Ends one request with its refusal: a submission this process will not admit is answered with
    /// the event that says so, because the daemon has no other way to learn it (§2.3).
    fn reject(&mut self, request: u64, kind: proto::request::RejectionKind, message: String) {
        let event = proto::request::wire::Event {
            request,
            kind: Some(proto::request::wire::event::Kind::Rejected(
                proto::request::wire::Rejected {
                    kind: kind as i32,
                    message,
                },
            )),
        };
        self.send_event(proto::request::wire::from_engine::Kind::Event(event));
    }

    /// Carries every admitted request's events to the daemon, and forgets a request whose stream
    /// ended (§2.3).
    ///
    /// Nothing here waits: a request that produced nothing this pass produces nothing, and the
    /// serve loop comes back to it.
    fn forward_events(&mut self) {
        // The streams are taken out so that sending an event can borrow the process: the map is
        // small, and the survivors go back before the pass returns.
        let streams = std::mem::take(&mut self.streams);
        let mut running = Vec::with_capacity(streams.len());
        for (id, rank, mut stream) in streams {
            loop {
                match stream.try_recv() {
                    Ok(event) => {
                        let terminal = matches!(
                            event,
                            engine::Event::Finished { .. }
                                | engine::Event::Rejected { .. }
                                | engine::Event::Error { .. }
                                | engine::Event::Migratable { .. }
                        );
                        let (wire, media) = event.to_wire(proto::request::RequestId(id));
                        if media.is_some() {
                            // An artifact's bytes travel beside its event (`ArtifactPart`), and this
                            // process does not carry them: a daemon that reads this event waits for
                            // bytes that never come. It is reachable only if a media request was
                            // admitted, and the guard against that is the scheduler's - a device
                            // with no media runner refuses the phase at admission. So this is the
                            // alarm, not the guard: it fires when the admission rule has been
                            // bypassed, and the event goes on because the outcome it states is the
                            // request's own.
                            tracing::error!(
                                request = id,
                                "a request produced media artifacts, which this process does not \
                                 carry: the daemon will wait for bytes that never come"
                            );
                        }
                        self.send_event(proto::request::wire::from_engine::Kind::Event(wire));
                        if terminal {
                            break;
                        }
                    }
                    // A request that produced nothing yet keeps its stream.
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                        running.push((id, rank, stream));
                        break;
                    }
                    // The scheduler that served the request is gone, which ends it.
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
                }
            }
        }
        self.streams = running;
    }

    /// States one report to the daemon (§7.3).
    ///
    /// A report is unasked, so a failure to state it costs the daemon's record and not this
    /// process's service: it is logged, and the epoch's own transition has already happened.
    /// States what this process is on the request interface (§7.3 step 1).
    ///
    /// The daemon's launch path waits for `Ready` (or `Standby`, for a process started beside a
    /// serving one) and records the info it carries for the epoch. Nothing else on either side moves
    /// before it: the first `Precopy` carries the scheduling policy and the first `Switchover` fills
    /// the generation, so a process that waited to be ready until it had a policy would wait for the
    /// command that waits for it - the deadlock the caller's stuck run shows.
    fn announce(&mut self, state: proto::request::wire::EngineState) -> Result<()> {
        let info = match state {
            proto::request::wire::EngineState::Ready
            | proto::request::wire::EngineState::Standby => Some(self.engine_info()?),
            _ => None,
        };
        tracing::info!(
            state = ?state,
            schedulers = self.schedulers.len(),
            devices = self.executors.len(),
            "stating what this process is to the daemon"
        );
        announce(&mut self.control, state, info)
    }

    /// What the daemon records of this process: the limits every request is checked against, the
    /// batch the workers were started for, and the KV cache segments a successor on the same
    /// devices maps (§7.3 step 1).
    ///
    /// The figures are the workers' own (`SchedulerHandle::info`), except the cache segments, which
    /// each scheduler describes for itself: a successor needs what the workers hold now, not what
    /// this process planned.
    fn engine_info(&self) -> Result<proto::request::wire::EngineInfo> {
        let first = self
            .schedulers
            .first()
            .context("this process runs no scheduler, so it states no engine info")?;
        let info = first.info();
        let mut segments = Vec::new();
        for scheduler in &self.schedulers {
            // A group that cannot describe its cache is reported with what the others stated rather
            // than failing the start: the segments are what a *successor* maps, and a successor that
            // finds one missing refuses by name, while a process that refused to state what it is
            // over this would hang the epoch before its first policy - the deadlock this report
            // exists to end.
            let groups = match scheduler
                .describe_cache()
                .recv_timeout(self.watchdog.stall_bound())
            {
                Ok(Ok(groups)) => groups,
                Ok(Err(error)) => {
                    tracing::warn!(%error, "a worker group did not describe its cache");
                    continue;
                }
                Err(error) => {
                    tracing::warn!(%error, "a scheduler did not describe its cache in time");
                    continue;
                }
            };
            segments.extend(groups.into_iter().map(|group| {
                proto::request::wire::CacheSegments {
                    worker_group: group.worker_group,
                    segments: group
                        .segments
                        .into_iter()
                        .map(|segment| proto::request::wire::CacheSegment {
                            world: segment.world,
                            ordinal: segment.ordinal,
                            units: segment.units,
                        })
                        .collect(),
                    held: group.held,
                    units: group.units,
                    scratch: group.scratch,
                }
            }));
        }
        Ok(proto::request::wire::EngineInfo {
            limits: Some((&first.generation_limits()).into()),
            video_denoiser: info
                .video_denoiser
                .as_ref()
                .map(|denoiser| denoiser.description.clone())
                .unwrap_or_default(),
            latent_encoding_units: first.media_units(ipc::MediaCall::LatentEncoding).unwrap_or(0),
            schedulers: u32::try_from(self.schedulers.len()).unwrap_or(u32::MAX),
            capacity: Some(proto::request::wire::BatchCapacity {
                max_batch: info.max_batch_calls,
                max_num_batched_tokens: u64::from(info.max_batch_tokens),
                // The launch's own figure, not the worker's row count: the rows include the
                // guidance prefix, which is not a request.
                max_running_requests: self.max_running_requests,
            }),
            segments,
            prefix_seeds: info.prefix_cache.as_ref().map_or_else(Vec::new, |cache| {
                cache
                    .seeds()
                    .into_iter()
                    .zip(&cache.groups)
                    .filter_map(|(seed, group)| {
                        (!matches!(group.kind, ipc::CacheGroupKind::Draft)).then_some(seed)
                    })
                    .collect()
            }),
            latent_seed: first.latent_seed(),
        })
    }

    fn send_report(&mut self, report: proto::process::Report) {
        if let Err(error) = self.process.report(report) {
            tracing::warn!(%error, "reporting this epoch's state to the daemon failed");
        }
    }

    /// Stops every thread in dependency order and releases the owners (§2.4).
    fn close(&mut self) {
        // §2.3: a shutdown finishes every request in flight as aborted, and the daemon learns that
        // from the request's own event - a connection that closes under it says only that the
        // process ended, not which request was lost with it.
        for (request, _, _) in std::mem::take(&mut self.streams) {
            let event = proto::request::wire::Event {
                request,
                kind: Some(proto::request::wire::event::Kind::Error(
                    proto::request::wire::Failed {
                        message: "the epoch was shut down with this request in flight".to_owned(),
                    },
                )),
            };
            self.send_event(proto::request::wire::from_engine::Kind::Event(event));
        }
        // The schedulers go first: their `Batches` holds a clone of every executor, and a device
        // thread cannot be joined while a scheduler could still submit to it.
        self.schedulers.clear();
        for executor in std::mem::take(&mut self.executors) {
            let Some(executor) = Arc::into_inner(executor) else {
                tracing::warn!(
                    "a device executor is still held after its schedulers ended; its thread ends \
                     when the last holder drops it"
                );
                continue;
            };
            if let Err(error) = executor.close() {
                tracing::warn!(%error, "closing a device executor failed");
            }
        }
        for error in self.blocking.close() {
            tracing::warn!(%error, "closing a blocking lane failed");
        }
        // The caches a park handed back name segments of the pools below: they go before the
        // resident owners close, so nothing the memory releases is still described as parked.
        self.parked = None;
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
            &policy
                .arguments
                .iter()
                .map(|(name, value)| (name.clone(), value.to_string()))
                .collect(),
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
            if let Err(error) = answer(
                scheduler.switchover_policy(),
                self.watchdog.stall_bound(),
                "the policy switchover",
            ) {
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
            if let Err(error) = answer(
                scheduler.revert_policy(),
                self.watchdog.stall_bound(),
                "the policy revert",
            ) {
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

/// The report one program-layer status states (§7.3).
///
/// A status the daemon learns another way - none today - answers `None`, and the caller logs it
/// instead of inventing a report.
fn report_of(status: &ProgramStatus) -> Option<proto::process::Report> {
    use proto::process::report::Kind;

    let kind = match status {
        ProgramStatus::Precopied { generation } => Kind::Precopied(proto::process::Precopied {
            generation: *generation,
        }),
        ProgramStatus::PrecopyFailed { failure, .. } => {
            Kind::PrecopyFailed(proto::process::PrecopyFailed {
                reason: failure.reason(),
            })
        }
        ProgramStatus::Switched { generation } => Kind::Switched(proto::process::Switched {
            generation: *generation,
        }),
        ProgramStatus::Committed { .. } => Kind::Committed(proto::process::Committed {}),
        ProgramStatus::Reverted { reason, .. } => Kind::Reverted(proto::process::Reverted {
            reason: reason.clone(),
        }),
        ProgramStatus::Purged { generation } => Kind::Purged(proto::process::Purged {
            generation: *generation,
        }),
        // A policy that failed is reported with the backend and revision it came from, so the
        // daemon's record takes the previous policy back (§4.4, §7.3).
        ProgramStatus::PolicyReverted {
            scheduler,
            backend,
            revision,
            reason,
            fatal,
        } => Kind::PolicyReverted(proto::process::PolicyReverted {
            scheduler: Some(*scheduler),
            backend: backend.clone(),
            revision: *revision,
            reason: reason.clone(),
            fatal: *fatal,
        }),
        // A probation failure is not a fault of the device: the revert that follows it states the
        // generation it left and the first error, so reporting it twice would name a device the
        // failure did not happen on (§7.3 step 4).
        ProgramStatus::Failed { .. } => return None,
    };
    Some(proto::process::Report { kind: Some(kind) })
}

/// The operator and descriptor key a failed walk named, when that is what failed (§7.3 step 2).
///
/// A generation holds every kernel its buckets launch, and a walk that finds one missing says
/// which: this is the only producer of an epoch's first generation's descriptor keys, because a
/// descriptor is what the model's forward computes for a bucket and nothing outside the forward can
/// list them.
fn missing_kernel(submissions: &[Submission]) -> Option<(String, String)> {
    /// The kernel an error names as missing, whether it reached the executor as an operator's
    /// error or wrapped in the model layer's: a capture reports it through
    /// `nn::Forward::operate`, which resolves the kernel and converts the operator's error into the
    /// layer's own. Matching only the operator's arm would leave every capture's miss unrecognised,
    /// and the discovery loop would abandon a generation it could have completed.
    fn named(error: &executor::Error) -> Option<(&'static str, &str)> {
        let ops = match error {
            executor::Error::Ops(ops) => ops,
            executor::Error::Nn(nn::Error::Ops(ops)) => ops,
            _ => return None,
        };
        match ops {
            ops::Error::MissingKernel {
                operator,
                descriptor,
                ..
            } => Some((operator, descriptor.as_str())),
            _ => None,
        }
    }
    for submission in submissions {
        let Some(Outcome::Failed(error)) = submission.outcome() else {
            continue;
        };
        if let Some((operator, descriptor)) = named(&error) {
            return Some((operator.to_owned(), descriptor.to_owned()));
        }
    }
    None
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

/// What the engine answers a command of the worker era's surface: the operation named a worker
/// world of its own, and this epoch executes batches itself (§2.4).
const WORKER_ERA: &str =
    "this epoch has no worker of that kind: the command names a worker-era engine, and the epoch's \
     executors are the process's own";

/// What the engine answers a command whose owner is the epoch layer: the same operation exists,
/// and it is asked for through the process connection, where the driver applies it (§2.5, §7.4).
const EPOCH_LIFECYCLE: &str =
    "this command is the epoch layer's, applied through the process connection rather than the \
     request connection";

/// States a main process's state on the request interface (§7.3 step 1).
fn announce(
    control: &mut Control,
    state: proto::request::wire::EngineState,
    info: Option<proto::request::wire::EngineInfo>,
) -> Result<()> {
    control
        .daemon_mut()
        .send(&proto::request::wire::FromEngine {
            kind: Some(proto::request::wire::from_engine::Kind::State(
                proto::request::wire::State {
                    state: state as i32,
                    info,
                    error: String::new(),
                },
            )),
        })
        .context("stating what this process is to the daemon")
}

/// What the engine answers a command over the deployment's storage ledgers or transfer budgets:
/// each node's daemon reads them over the claim it holds, not through an epoch's main process (§8).
const LEDGERS: &str =
    "the storage ledgers and transfer budgets are read by each node's daemon over its claim, not \
     by an epoch's main process";

/// The §7.3 stage a program-layer command belongs to, for the log line each one leaves.
fn program_stage(command: &ProgramCommand) -> &'static str {
    match command {
        ProgramCommand::Precopy { .. } => "precopy",
        ProgramCommand::Switchover { .. } => "switchover",
        ProgramCommand::Commit { .. } => "commit",
        ProgramCommand::Revert { .. } => "revert",
        ProgramCommand::Purge { .. } => "purge",
    }
}

/// The name of an engine command, for a log line about one this epoch does not serve.
fn engine_command_name(kind: &proto::request::wire::engine_command::Kind) -> &'static str {
    use proto::request::wire::engine_command::Kind;
    match kind {
        Kind::Shutdown(_) => "shutdown",
        Kind::Promote(_) => "promote",
        Kind::FlushCache(_) => "flush_cache",
        Kind::Hook(_) => "hook",
        Kind::Configure(_) => "configure",
        Kind::Attach(_) => "attach",
        Kind::ResizeCache(_) => "resize_cache",
        Kind::UpdateWeights(_) => "update_weights",
        Kind::Park(_) => "park",
        Kind::Resume(_) => "resume",
        Kind::Commit(_) => "commit",
        Kind::DescribeCache(_) => "describe_cache",
        Kind::DescribeStorage(_) => "describe_storage",
        Kind::ThrottleTransfers(_) => "throttle_transfers",
        Kind::DescribeRequests(_) => "describe_requests",
        Kind::CacheWarm(_) => "cache_warm",
        Kind::CommitExperts(_) => "commit_experts",
    }
}

/// Every program this host may fill an operator with, by operator, in the order the walk tries them
/// (§7.3 step 2): the highest revision first, ties by backend name.
type Candidates = BTreeMap<String, Vec<(String, u32)>>;

/// The first refusal the installs of one generation met, shared by the precopy and every device's
/// load so the walk can answer it with the operator's next candidate program.
type UnavailableCell = Arc<Mutex<Option<Unavailable>>>;

/// What the walk does when a program cannot serve a shape of this generation (§7.3 step 2).
enum Advance {
    /// The operator's next candidate was chosen and its install admitted.
    Retry(Vec<Arc<HostTask<Install>>>),
    /// Every program this host built for the operator has been tried.
    Exhausted(String),
    /// The programs are the daemon's, so there is no candidate to move to.
    Named,
    /// The store could not be read, so no candidate can be chosen.
    Failed(String),
}

/// How often the engine reports its counters.
const METRICS_EVERY: Duration = Duration::from_secs(1);

/// How closely an update's probation is watched while one is pending: the phase machine states an
/// expiry, and the control thread asks it at least this often. A rollback window is stated in whole
/// seconds, so the poll costs a wake per tenth of a second for the length of the window.
const PROBATION_POLL: Duration = Duration::from_millis(100);

/// How often a device that is busy delays the question asked of the policy slots: a probe that finds
/// a device running proves nothing about whether a policy serves, so it is asked again after this.
const POLICY_PROBE_INTERVAL: Duration = Duration::from_millis(250);

/// How long a metrics pass waits on one scheduler's counters: the report is sent once a second and a
/// pass that waited longer than this would be reporting what the next pass reports anyway.
const METRICS_WAIT: Duration = Duration::from_millis(50);

/// How long the daemon's process interface may take to answer the handshake, which it answers at
/// once: the wait exists to fail a wrong address rather than hang the process.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

/// How long the daemon may take to answer this rank's storage claim (§8): starting a Storage
/// Server for the node is the slow part, and it is the daemon's to do.
const CLAIM_TIMEOUT: Duration = Duration::from_secs(60);

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
    /// The Program Server's own refusal, when it named a program that cannot serve a shape of this
    /// generation. The task fails with a message either way; this keeps the structured cause so the
    /// walk can ask the operator's next candidate program instead of giving up on the generation.
    unavailable: UnavailableCell,
}

/// A program that cannot serve a shape of this generation: the two refusals that mean "the walk may
/// try the operator's next candidate program" rather than "this generation cannot be built".
#[derive(Clone, Debug)]
enum Unavailable {
    /// The program is installed, and no module of it covers this descriptor key.
    Descriptor {
        op: String,
        backend: String,
        revision: u32,
        descriptor: String,
    },
    /// No revision of the backend is installed for the operator.
    Installed { op: String, backend: String },
}

impl Unavailable {
    /// Reads a Program Server refusal, when it is one of the two the walk can answer.
    fn of(error: &program_server::Error) -> Option<Self> {
        match error {
            program_server::Error::UnsupportedDescriptor {
                op,
                backend,
                revision,
                descriptor,
            } => Some(Self::Descriptor {
                op: op.clone(),
                backend: backend.clone(),
                revision: *revision,
                descriptor: descriptor.clone(),
            }),
            program_server::Error::NotInstalled { op, backend } => Some(Self::Installed {
                op: op.clone(),
                backend: backend.clone(),
            }),
            _ => None,
        }
    }

    /// The operator whose program cannot serve the shape.
    fn op(&self) -> &str {
        match self {
            Self::Descriptor { op, .. } | Self::Installed { op, .. } => op,
        }
    }

    /// The backend the refusal names.
    fn backend(&self) -> &str {
        match self {
            Self::Descriptor { backend, .. } | Self::Installed { backend, .. } => backend,
        }
    }

    /// The descriptor the program could not serve, when the refusal named one.
    fn descriptor(&self) -> Option<&str> {
        match self {
            Self::Descriptor { descriptor, .. } => Some(descriptor),
            Self::Installed { .. } => None,
        }
    }
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
                Some(metadata) => {
                    ops::CubinMetadata::parse(metadata)
                        .map_err(|error| invalid(format!("the cubin's metadata: {error}")))?
                        .name
                }
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
                return Err(invalid("a native module states no launch entry".to_owned()));
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

impl Install {
    /// Keeps the first structured refusal this device's load met.
    fn note(&self, error: &program_server::Error) {
        if let Some(unavailable) = Unavailable::of(error) {
            let mut held = self
                .unavailable
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if held.is_none() {
                *held = Some(unavailable);
            }
        }
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
                self.note(&error);
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
                        self.note(&error);
                        executor::Error::Invalid(format!(
                            "resolving {} for {key}: {error}",
                            chosen.slot.op
                        ))
                    })?;
                let module = resolved(module)?;
                let entry = ops::install(&self.device, operator, &key, module.as_ops()?).map_err(
                    |error| {
                        executor::Error::Invalid(format!(
                            "installing {operator} for {key}: {error}"
                        ))
                    },
                )?;
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
    /// The kernel programs the update installs, by slot: kept so the discovery loop can admit the
    /// installs again with a key it just learned, or with the operator's next candidate program.
    programs: Programs,
    /// Every program this host may fill an operator with, when this generation's programs came from
    /// the store's listing rather than from the daemon. Empty when the daemon named them: a named
    /// program is the deployment's choice and the walk does not second-guess it.
    candidates: Candidates,
    /// The candidates this generation has already asked for. Each is tried once, and none again.
    tried: std::collections::BTreeSet<(String, u32)>,
    /// The backends tried for each operator, so a refusal names every one of them.
    backends: BTreeMap<String, Vec<String>>,
    /// The refusal the generation's installs met, shared with the device loads that found it.
    unavailable: UnavailableCell,
    /// The descriptor keys the walk has taught this generation, by operator. The epoch's first
    /// generation has no serving generation to take them from (§7.3 step 2).
    learned: BTreeMap<String, Vec<String>>,
    /// Whether every scheduler has already been asked for the candidate policy: the discovery loop
    /// re-enters the stages, and the policy is prepared once.
    policy_requested: bool,
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

/// The programs each operator of `packages` may be installed from, in the order §7.3 step 2's walk
/// tries them: the highest installed revision first, and among equal revisions the backend whose
/// name sorts first, so the order is the same on every run.
///
/// Nothing here excludes a candidate. An operator built under two backends is one slot with two
/// programs rather than one choice made before the walk, because whether a program can serve a
/// descriptor is only known when the walk names the descriptor (`advance_candidate`).
///
/// Two kinds of package are left out, and both would cost more than they buy:
///
/// - one with no module built for one of `targets`. Installing it asks `prepare` to *build* it for
///   this host, so a store built for other targets would spend a startup's whole precopy clock
///   compiling modules the first token does not need. The operator then stays missing, and the walk
///   names it - the honest answer for a host the store was not built for.
/// - one whose operator this binary does not implement: `admit_installs` refuses a precopy for such
///   an operator, and a program this process could never launch is not worth failing a switchover
///   over.
fn select_programs(
    packages: &[program_server::protocol::ListedPackage],
    targets: &[Target],
) -> BTreeMap<String, Vec<(String, u32)>> {
    let mut chosen: BTreeMap<String, Vec<(String, u32)>> = BTreeMap::new();
    for package in packages {
        if package.modules.is_empty() {
            tracing::debug!(
                op = %package.op,
                backend = %package.backend,
                "a listed package has no built module"
            );
            continue;
        }
        if !package
            .modules
            .iter()
            .any(|module| targets.contains(&module.target))
        {
            tracing::debug!(
                op = %package.op,
                backend = %package.backend,
                "the store holds this operator for another target: not installed"
            );
            continue;
        }
        if ops::operator_name(&package.op).is_none() {
            tracing::debug!(
                op = %package.op,
                backend = %package.backend,
                "the store holds an operator this build does not implement"
            );
            continue;
        }
        let candidate = (package.backend.clone(), package.revision);
        let held = chosen.entry(package.op.clone()).or_default();
        if let Some((backend, revision)) = held.last() {
            if *backend == candidate.0 && *revision == candidate.1 {
                continue;
            }
            if *revision != candidate.1 || backend.as_str() <= candidate.0.as_str() {
                tracing::info!(
                    op = %package.op,
                    taken = %candidate.0,
                    revision = candidate.1,
                    held = %backend,
                    held_revision = revision,
                    "an operator is built under more than one backend: both are candidates, the \
                     walk tries the one that can serve its descriptor"
                );
            }
        }
        held.push(candidate);
    }
    for candidates in chosen.values_mut() {
        candidates.sort_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| left.0.as_str().cmp(right.0.as_str()))
        });
    }
    chosen
}

/// The descriptor keys this host has already built for a named package, for this node's targets.
///
/// The epoch's first generation has no serving generation to copy keys from, and without this the
/// walk learns one descriptor per capture. What the Program Server lists is what it holds *built* -
/// not what the package declares - so this accelerates a store that was built ahead of the epoch
/// (every deployment whose programs were built offline) and answers nothing on a host that has built
/// none yet, where the walk remains the source of truth. A module built for another target does not
/// cover this node and is not claimed.
fn built_keys(
    listing: &program_server::Listing,
    op: &str,
    program: &Program,
    targets: &[program_server::Target],
) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    for package in &listing.packages {
        if package.op != op
            || package.backend != program.backend
            || package.revision != program.revision
        {
            continue;
        }
        for module in &package.modules {
            if !targets.contains(&module.target) || keys.contains(&module.descriptor) {
                continue;
            }
            keys.push(module.descriptor.clone());
        }
    }
    keys
}

fn descriptor_keys(generation: &Generation, operator: &'static str) -> Vec<String> {
    generation
        .iter()
        .filter(|(name, _, _)| *name == operator)
        .map(|(_, key, _)| key.to_owned())
        .collect()
}

/// The oldest of the executors' jobs.
fn oldest(running: &[Running]) -> Option<Running> {
    running.iter().min_by_key(|job| job.since).copied()
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
/// This node's scheduler threads, one per DP rank (§2.4, §4.4).
///
/// A node that is not the scheduler node serves only what its peers' schedulers ask of it, so it
/// builds none. The scheduler node builds one thread per data-parallel rank, each handed this
/// node's executors through `executor::Batches`: the scheduler forms a batch, the executor issues
/// it, and the scheduler reads the result - nothing crosses threads except the batch and its
/// result.
fn schedulers(
    plan: &EpochPlan,
    launch: &Launch,
    memory: &Memory,
    model: &qwen3_moe::Qwen3Moe,
    executors: &[Arc<Executor>],
    rank: &ipc::WorkerInfo,
) -> Result<Vec<engine::SchedulerHandle>> {
    if !launch.scheduler_node {
        return Ok(Vec::new());
    }
    if executors.is_empty() {
        bail!("the scheduler node holds no device to serve a batch on");
    }
    // One scheduler per scheduler thread: the configuration is this rank's, derived from what the
    // launch states, the build config fixes and the model's own geometry, and it refuses by name
    // when a fact it needs is one the launch file does not carry.
    let facts = plan.rank_facts(memory, model)?;
    let config = engine::EngineConfig::of(&facts)?;
    // The scheduler submits through the same devices this process drives, so a batch and a
    // switchover reach one executor thread per device and never two.
    // The scheduler plans against a description of the pool, and this process is the one pool it
    // holds: its single rank is this one, its capacities are the launch's (the rank reports what it
    // holds, not what it was launched for), and its longest sequence is the model's bound. The
    // inventory is the executor's to hold and the boot's to state, because neither number can be
    // derived from a batch (4.4).
    let inventory = executor::batch::Inventory {
        group: launch.group.clone(),
        ranks: vec![rank.clone()],
        placement: executor::batch::Placement {
            queue_depth: plan.queue_depth(),
            ranks: plan.world_size(),
            components: engine::WorkerGroupConfig::single_component(
                ipc::DEFAULT_COMPONENT,
                plan.world_size(),
            ),
        },
        max_model_len: plan.max_kv_len(model)?,
    };
    let batches: Arc<dyn executor::batch::Batches> = Arc::new(
        executor::batch::Executors::shared(executors.iter().map(Arc::clone).collect())?
            .stating(vec![inventory]),
    );
    let handle = engine::SchedulerHandle::over_batches(config, batches)?;
    Ok(vec![handle])
}

/// The daemon states that this build may serve the epoch, this node's host identity, and, later,
/// the program revision of the generation it prepares (§7.3). The programs themselves never come
/// through this interface: the daemon names them and this process resolves their modules from the
/// Program Server (§5.1). The build config is the one the daemon wrote for this node, which `run`
/// reads before the handshake.
fn handshake(control: &mut Control, launch: &Launch) -> Result<Epoch> {
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
    // The epoch's programs and the generation that serves at start are not in the spec: the
    // programs arrive as the daemon's first `Precopy` (§7.3 step 2) and the generation number is the
    // daemon's own. Until it states one, this process serves generation zero, whose `Precopy` names
    // the daemon's next number.
    //
    // The host identity is the daemon's fact and nothing else can supply it: the Storage Server
    // keys this rank's claim by it, so an empty one would mint a claim on a host that does not
    // exist and every peer would compare its products against the wrong node.
    if spec.host.is_empty() {
        bail!(
            "the daemon's welcome states no host identity, which this rank's storage claim is keyed by"
        );
    }
    Ok(Epoch {
        generation: 0,
        host: spec.host,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use program_server::protocol::{ListedModule, ListedPackage};
    use program_server::ModuleForm;

    /// A listed package with one built module for `target`.
    fn package(op: &str, backend: &str, revision: u32, target: &str) -> ListedPackage {
        ListedPackage {
            op: op.to_owned(),
            backend: backend.to_owned(),
            revision,
            modules: vec![ListedModule {
                target: Target::new(target),
                descriptor: "a-descriptor".to_owned(),
                form: ModuleForm::Cubin,
            }],
        }
    }

    fn targets() -> Vec<Target> {
        vec![Target::new("sm_90a")]
    }

    /// An operator name this build implements, taken from the build rather than written down, so a
    /// renamed operator does not fail a test about selection.
    fn implemented() -> &'static str {
        ops::operators()
            .first()
            .copied()
            .expect("the build implements operators")
    }

    /// The later installation of an operator is the one the walk tries first, whatever order the
    /// store lists them in.
    #[test]
    fn the_highest_revision_of_an_operator_is_chosen() {
        let op = implemented();
        for packages in [
            vec![
                package(op, "triton", 1, "sm_90a"),
                package(op, "triton", 3, "sm_90a"),
            ],
            vec![
                package(op, "triton", 3, "sm_90a"),
                package(op, "triton", 1, "sm_90a"),
            ],
        ] {
            let chosen = select_programs(&packages, &targets());
            assert_eq!(
                chosen.get(op).and_then(|ordered| ordered.first()),
                Some(&("triton".to_owned(), 3)),
                "the highest revision is tried first"
            );
        }
    }

    /// When two backends hold the same revision, the one that sorts first is tried first - and the
    /// other stays a candidate, so a program that cannot serve the walk's descriptor is answered
    /// rather than fatal.
    #[test]
    fn a_tie_on_revision_is_broken_by_the_backend_name() {
        let op = implemented();
        for order in [["triton", "flashinfer"], ["flashinfer", "triton"]] {
            let packages = order
                .into_iter()
                .map(|backend| package(op, backend, 2, "sm_90a"))
                .collect::<Vec<_>>();
            let chosen = select_programs(&packages, &targets());
            assert_eq!(
                chosen.get(op).and_then(|ordered| ordered.first()),
                Some(&("flashinfer".to_owned(), 2)),
                "the same backend is tried first whatever order the store lists them in"
            );
            assert_eq!(
                chosen[op].len(),
                2,
                "and the other backend is still a candidate"
            );
        }
    }

    /// Three packages that cannot launch here, for three different reasons, and one that can: only
    /// the last is installed.
    #[test]
    fn only_packages_this_host_could_launch_are_installed() {
        let op = implemented();
        let mut unbuilt = package(op, "triton", 9, "sm_90a");
        unbuilt.modules.clear();
        let packages = vec![
            unbuilt,
            package(op, "triton", 8, "sm_70"),
            package("an-operator-this-build-lacks", "triton", 7, "sm_90a"),
            package(op, "triton", 1, "sm_90a"),
        ];

        let chosen = select_programs(&packages, &targets());
        assert_eq!(chosen.len(), 1, "one operator can launch here: {chosen:?}");
        assert_eq!(
            chosen[op],
            vec![("triton".to_owned(), 1)],
            "the package with a module for this target, whatever the others claim"
        );
    }
}
