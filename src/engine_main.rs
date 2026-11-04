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
use crate::engine_plan::EpochPlan;
use daemon::epoch::BuildConfig;
use engine::{PolicyProgram, PolicySource, ProgramSource};
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
            peer: Some(proto::process::hello::Peer::Engine(proto::process::Engine {
                deployment: deployment.to_owned(),
            })),
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
                    tracing::warn!(id = reply.id, "an answer to a request this process did not send");
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
        while let Some(message) = self.connection.try_recv::<proto::process::Downstream>()? {
            match message.kind {
                Some(proto::process::downstream::Kind::Event(event)) => self.file(event),
                Some(proto::process::downstream::Kind::Reply(reply)) => {
                    tracing::warn!(id = reply.id, "an answer to a request this process did not send");
                }
                None => {}
            }
        }
        Ok(self.pending.pop_front())
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
) -> Result<ipc::StorageLaunch> {
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
    streams: Vec<(u64, engine::EventRx)>,
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
        // The daemon's two interfaces are two sockets: the epoch's requests and metrics travel on
        // the request connection above, and everything the daemon asks of this process - the claim,
        // the reports of §7.3, and every command of the program layer - travels on this one.
        let mut process = ProcessInterface::connect(&launch.processes, &launch.deployment)?;

        // §8: the rank's Storage Server and its claim are the node's, which the daemon answers for;
        // the retained weights outlive every world of the epoch, so the store is built once, before
        // the model loads, and keeps what it maps for the epoch's life.
        let storage = claim(&mut process, launch, &epoch.host)?;
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
        let (model, mut memory) = plan
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
            // storage sized. Nothing is serving yet, so the control thread waits here; `serve`
            // never does (§2.4).
            //
            // §4.5's warmup is not here: the epoch's first generation holds no kernel until the
            // daemon's precopy fills it, so there is no bucket to warm. It runs at the first
            // switchover, which is the boundary the first request is served at.
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
        let schedulers = schedulers(&plan, launch, &memory, &model, &executors)?;

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
            process,
            stopping: false,
            streams: Vec::new(),
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
                    let request = match proto::request::Request::from_wire(wire) {
                        Ok(request) => request,
                        Err(error) => {
                            // A request that does not convert is the daemon's, not this process's:
                            // it ends with the refusal's event and every other request of the
                            // submission is still admitted.
                            self.reject(id, proto::request::RejectionKind::Invalid, error.to_string());
                            continue;
                        }
                    };
                    let rank = self.route(&request);
                    let Some(scheduler) = self.schedulers.get(rank) else {
                        bail!("a request was routed to rank {rank}, which holds no scheduler");
                    };
                    match scheduler.submit(request) {
                        Ok(stream) => {
                            self.streams.push((id, stream));
                            self.send_event(
                                proto::request::wire::from_engine::Kind::Accepted(
                                    proto::request::wire::Accepted {
                                        request: id,
                                        scheduler: rank as u32,
                                    },
                                ),
                            );
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
        // A drain is the epoch layer's, and it answers the daemon: the frontend admits nothing
        // from the boundary the schedulers reached, and the facts of the requests in flight are
        // kept for the `Serve` that resumes this process on another request address (§2.3).
        let draining = match &command.kind {
            Some(proto::process::command::Kind::Drain(drain)) => Some(drain.epoch),
            _ => None,
        };
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
            if let Some(epoch) = draining {
                self.send_report(proto::process::Report {
                    kind: Some(proto::process::report::Kind::Drained(
                        proto::process::Drained { epoch },
                    )),
                });
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

        let tasks = match self.admit_installs(programs, &BTreeMap::new()) {
            Ok(tasks) => tasks,
            Err(failure) => return self.abandon_precopy(generation, failure, reply),
        };

        self.precopy = Some(Precopy {
            generation,
            reply,
            policy: policy.clone(),
            programs: programs.clone(),
            learned: BTreeMap::new(),
            policy_requested: false,
            next: Vec::new(),
            policies: Vec::new(),
            stage: PrecopyStage::Modules(tasks),
        });
        Ok(())
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
    ) -> std::result::Result<Vec<Arc<HostTask<Install>>>, UpdateFailure> {
        let invalid = |reason: String| UpdateFailure::Precopy { reason };
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
            let keys = if serving.is_empty() {
                learned.get(operator).cloned().unwrap_or_default()
            } else {
                serving
            };
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
            let device = platform::Device::new(ordinal)
                .map_err(|error| invalid(format!("opening device {}: {error}", ordinal.0)))?;
            let Some(target) = self.targets.get(index).cloned() else {
                return Err(invalid(format!(
                    "device {} has no build target",
                    ordinal.0
                )));
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
                    return Err(invalid(format!("admitting a module load: {error}")));
                }
            }
        }
        Ok(tasks)
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
                                return self.abandon_precopy(
                                    generation,
                                    UpdateFailure::Precopy {
                                        reason: format!(
                                            "{operator} still has no kernel under {key} after it \
                                             was installed: the walk and the install disagree"
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
                            let tasks = match self.admit_installs(
                                &precopy.programs,
                                &precopy.learned,
                            ) {
                                Ok(tasks) => tasks,
                                Err(failure) => {
                                    return self.abandon_precopy(generation, failure, precopy.reply);
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
        // A switchover to a generation this process never prepared cannot succeed, and cutting the
        // devices over to try it costs a live graph rebuild to end where it started: it is refused
        // before any device is touched, and the generation that serves keeps serving.
        if self.prepared.is_empty() {
            return Ok(ProgramStatus::PrecopyFailed {
                generation,
                failure: UpdateFailure::Precopy {
                    reason: format!(
                        "a switchover to generation {generation}, which this process prepared \
                         nothing for: the precopy that builds it either failed or never arrived"
                    ),
                },
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
        // §4.5: the first request served on a generation must not pay for a capture, so every
        // captured bucket of the generation that just took over runs once, here, between batches.
        // A switchover later in the epoch warms the buckets the new kernel programs changed; the
        // first one also warms the buckets no update touched, because until now nothing served
        // them.
        for executor in &self.executors {
            let device = executor.device();
            let submission = executor
                .warmup()
                .with_context(|| format!("warming up device {}", device.0))?;
            if let Err(error) = device_command(&self.watchdog, device, "warmup", submission) {
                tracing::error!(%error, "warming the new generation failed; reverting the update");
                let failure = UpdateFailure::Device {
                    device: u32::try_from(device.0).unwrap_or_default(),
                    error: error.to_string(),
                    batch: None,
                };
                return self.revert(generation, failure);
            }
        }

        // The state machine records the cut-over only when every device serves the new generation,
        // and the process's record of what serves follows it.
        self.update.switchover(generation, Instant::now())?;
        self.retired =
            std::mem::replace(&mut self.generations, std::mem::take(&mut self.prepared));
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
    fn revert(&mut self, generation: u64, failure: UpdateFailure) -> Result<ProgramStatus> {
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
        if !self.retired.is_empty() {
            self.generations = std::mem::take(&mut self.retired);
        }
        self.prepared.clear();
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
            let submission = executor.revert(current).with_context(|| {
                format!("queueing device {}'s revert to generation {current}", device.0)
            })?;
            device_command(&self.watchdog, device, "revert", submission)?;
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
        // The schedulers state their own figures; a node that holds none reports the executors'
        // and a step count of zero, because no step happened here.
        let (mut steps, mut batches, mut tokens, mut policy_fuel, mut policy_calls) = (0, 0, 0, 0, 0);
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
        for (id, mut stream) in streams {
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
                            // process does not carry them yet: the event alone would make a daemon
                            // wait for bytes that never come, so the request is refused by name.
                            tracing::warn!(
                                request = id,
                                "a request produced media artifacts, which this process does not carry"
                            );
                        }
                        self.send_event(proto::request::wire::from_engine::Kind::Event(wire));
                        if terminal {
                            break;
                        }
                    }
                    // A request that produced nothing yet keeps its stream.
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                        running.push((id, stream));
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
        for (request, _) in std::mem::take(&mut self.streams) {
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

/// The report one program-layer status states (§7.3).
///
/// A status the daemon learns another way - none today - answers `None`, and the caller logs it
/// instead of inventing a report.
fn report_of(status: &ProgramStatus) -> Option<proto::process::Report> {
    use proto::process::report::Kind;

    let kind = match status {
        ProgramStatus::Precopied { generation } => {
            Kind::Precopied(proto::process::Precopied {
                generation: *generation,
            })
        }
        ProgramStatus::PrecopyFailed { failure, .. } => {
            Kind::PrecopyFailed(proto::process::PrecopyFailed {
                reason: failure.reason(),
            })
        }
        ProgramStatus::Switched { generation } => Kind::Switched(proto::process::Switched {
            generation: *generation,
        }),
        ProgramStatus::Committed { .. } => Kind::Committed(proto::process::Committed {}),
        ProgramStatus::Reverted { reason, .. } => {
            Kind::Reverted(proto::process::Reverted {
                reason: reason.clone(),
            })
        }
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
    for submission in submissions {
        let Some(Outcome::Failed(error)) = submission.outcome() else {
            continue;
        };
        if let executor::Error::Ops(ops::Error::MissingKernel {
            operator,
            descriptor,
            ..
        }) = &*error
        {
            return Some(((*operator).to_owned(), descriptor.clone()));
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

/// How often the engine reports its counters.
const METRICS_EVERY: Duration = Duration::from_secs(1);

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
    /// The kernel programs the update installs, by slot: kept so the discovery loop can admit the
    /// installs again with a key it just learned.
    programs: Programs,
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

/// The descriptor keys `generation` holds for `operator`: what a precopy must
/// cover, and exactly what it never adds to (§7.3 step 2, §5.3).
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
    let batches: Arc<dyn executor::batch::Batches> = Arc::new(
        executor::batch::Executors::shared(executors.iter().map(Arc::clone).collect())?,
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



