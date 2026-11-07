//! `uniserve engine`: a node's main process for one epoch (`specs/executor.md`
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
//!   Result<Plan>`, whose producer the retired executor held and which is still
//!   entangled with the launch machinery §4 deletes, and the Storage
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

use anyhow::{Context as _, Result, bail};
use clap::Args;

use crate::engine_plan::{EpochPlan, PrefillCatalogue};
use daemon::epoch::BuildConfig;
use engine::{PolicyProgram, PolicySource, ProgramSource};
use executor::batch::Submission;
use executor::blocking::{BlockingKind, ThreadPool};
use executor::completion::Outcome;
use executor::control::{Control, Endpoints, Ready};
use executor::device::{Executor, Probe};
use executor::host::{HostAction, HostTask, Lane};
use executor::hot_update::{Action, HotUpdate};
use executor::memory::Memory;
use executor::memory::retained::Retained;
use executor::memory::weights::WeightStore;
use executor::parallel::host::{HostGroup, Rendezvous, STARTUP_TIMEOUT};
use executor::runners::products::{GroupProducts, RankProducts};
use executor::watchdog::{Running, Watchdog};
use ipc::{ProgramCommand, ProgramStatus, UpdateFailure};
use ops::{Generation, KernelEntry};
use program_server::program::Programs;
use program_server::{
    Client, Descriptors, PhaseKind, PrepareRequest, Program, ProgramRequest, ProgramSlot, Selector,
    Target,
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
    /// The world of the epoch this process succeeds, whose storages it maps until it is promoted.
    pub predecessor_world: Option<u64>,
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
    /// This rank's executor group, which keys its storage claims.
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
    /// The placement's queue depth for this rank's executor groups: how many batches the rank's
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
    fn connect(address: &str, deployment: &str, engine: u64) -> Result<Self> {
        // A process link that cannot name its engine is refused here rather than sent: zero states
        // "no main process", and a daemon that accepted it would file this connection in the place
        // of a process it can never command. The launch file states the number, so a launch without
        // one is the deployment's fault and this is where it is named.
        if engine == 0 {
            bail!(
                "the launch file states engine 0, which names no main process: the daemon addresses \
                 the epoch's program-layer commands by this number, so a process it started under \
                 none cannot be commanded"
            );
        }
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
                    // The daemon keys this connection by the number the epoch layer started this
                    // process under, so a command it sends for engine N reaches this process and
                    // not whichever connection the daemon happened to accept Nth.
                    engine,
                    // This is the process link: the one the daemon's program-layer commands
                    // travel down. The request link, on its own socket, states `REQUESTS`, so the
                    // two links of one engine are never confused for one another.
                    interface: proto::process::Interface::Process as i32,
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
            // The engine number this link claims, so this process's line and the daemon's
            // "a process connected engine=N" are matched by a number rather than by their order in
            // the log - which is the confusion that cost a run.
            engine,
            interface = "process",
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
                    tracing::warn!(
                        "the daemon closed this process's interface; whether this epoch ends \
                         depends on whether another daemon holds it"
                    );
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
            // A group's exit is an executor's report; this process supervises no group.
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

/// One rank of this process, loaded: what its executor is built from.
struct LoadedRank {
    /// The rank's own plan, whose one device is the rank's.
    plan: EpochPlan,
    /// The rank's Storage Server client, whose scope is the rank's.
    client: Arc<Retained>,
    /// The rank's partition of the model, mapped on its device.
    model: crate::engine_plan::Served,
    /// The rank's resident owners.
    memory: Memory,
    /// What the rank reports of itself.
    inventory: ipc::MainProcessInfo,
    /// The rank's graph storage figures.
    graph: executor::device::GraphStorage,
    /// The collective groups its runners issue on.
    groups: crate::engine_plan::RankGroups,
}

// SAFETY: a rank is built on its load thread and handed once to the thread that spawns its executor,
// and nothing in it belongs to the thread that built it. The field that needs this is the model: its
// `nn::Block` holds `Arc<dyn Encoder>` and its siblings, whose trait objects state no `Send`, while the
// concrete blocks behind them are the ones a runner already moves to its device thread
// (`M: Encoder + Send + Sync`) and hold only device addresses and immutable host tables.
unsafe impl Send for LoadedRank {}

/// Loads every rank of this process, each on a thread of its own.
///
/// The ranks agree their capacity and exchange their communicators' ids over one host group, whose
/// collectives every member must reach, so they run side by side rather than one after another; the
/// weights of four ranks then also load at once. The group is the one the design's capacity plan
/// agrees over, formed here over the loopback because this process holds all its members.
fn load_ranks(
    plans: Vec<EpochPlan>,
    storage: &ipc::StorageLaunch,
    launch: &Launch,
    epoch: &Epoch,
) -> Result<Vec<LoadedRank>> {
    let count = plans.len();
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))
        .context("binding the rendezvous of this process's ranks")?;
    let address = listener
        .local_addr()
        .context("reading the rendezvous of this process's ranks")?
        .to_string();
    let mut listener = Some(listener);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(count);
        for (index, plan) in plans.into_iter().enumerate() {
            let rendezvous = match listener.take() {
                Some(listener) => Rendezvous::serve(address.clone(), listener),
                None => Rendezvous::connect(address.clone()),
            };
            let handle = std::thread::Builder::new()
                .name(format!("rank-load-{index}"))
                .spawn_scoped(scope, move || {
                    load_rank(plan, rendezvous, index, count, storage, launch, epoch)
                })
                .context("starting a rank's load")?;
            handles.push(handle);
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("a rank's load panicked"))?
            })
            .collect()
    })
}

/// Joins the group, opens the rank's collectives, loads its partition and opens its memory.
fn load_rank(
    plan: EpochPlan,
    rendezvous: Rendezvous,
    index: usize,
    count: usize,
    storage: &ipc::StorageLaunch,
    launch: &Launch,
    epoch: &Epoch,
) -> Result<LoadedRank> {
    let host = HostGroup::connect(rendezvous, index, count, STARTUP_TIMEOUT)
        .context("joining this process's ranks")?;
    // §8: the rank's Storage Server and its claim are the node's, which the daemon answered for; the
    // retained weights outlive every world of the epoch, so the store is built once, before the
    // model loads, and keeps what it maps for the epoch's life.
    let client = Retained::connect(
        storage,
        &launch.group,
        plan.rank(),
        epoch.generation,
        epoch.predecessor_world,
    )
    .context("connecting to the Storage Server")?;
    // A weight's region is one device's address, so the store maps on the device this rank's model
    // is loaded for; the client is shared, the mapping is not.
    let device = *plan
        .devices()
        .first()
        .context("this rank serves no device to map its weights on")?;
    let shape = plan.parallel_shape()?;
    // NCCL keeps its own device buffers, so a communicator opens before the capacity plan measures
    // what is left of the device.
    let groups = crate::engine_plan::RankGroups {
        sequence: group_over(&host, device, shape.sequence, shape.sequence_rank, count)
            .context("forming the sequence group")?,
        tensor: group_over(&host, device, shape.tensor, shape.tensor_rank, count)
            .context("forming the tensor group")?,
    };
    let mut weights = WeightStore::new(Arc::clone(&client), device);
    // The load and the capacity plan are one call because the order between them is the plan's: the
    // weights are mapped first, so what the plan measures is what is left of the device rather than
    // what was there before the model.
    let (model, memory, inventory, graph) = plan
        .load_and_open(&mut weights, &mut &*client, &host)
        .context("loading the epoch's model")?;
    Ok(LoadedRank {
        plan,
        client,
        model,
        memory,
        inventory,
        graph,
        groups,
    })
}

/// The collective group of `size` ranks this rank is member `rank` of, absent for a group of one.
///
/// The first member draws the group's id and the host group carries it (`parallel`'s statement of
/// what that channel is for); `Communicator::init` then blocks until every member joined.
fn group_over(
    host: &HostGroup,
    device: platform::DeviceId,
    size: usize,
    rank: usize,
    ranks: usize,
) -> Result<Option<Arc<nn::Group>>> {
    if size == 1 {
        return Ok(None);
    }
    if size != ranks {
        bail!(
            "a parallel group of {size} ranks in a process of {ranks}: this process forms one \
             group of all its ranks, and a deployment of several groups per host is not served"
        );
    }
    let id = match rank {
        0 => Some(platform::UniqueId::new()?),
        _ => None,
    };
    let bytes = host.broadcast_bytes(id.as_ref().map(|id| &id.0[..]), 0)?;
    let id = platform::UniqueId(bytes.try_into().map_err(|bytes: Vec<u8>| {
        anyhow::anyhow!("a communicator's id is 128 bytes, not {}", bytes.len())
    })?);
    let size = i32::try_from(size).context("a group wider than NCCL addresses")?;
    let member = i32::try_from(rank).context("a rank past what NCCL addresses")?;
    let communicator = platform::Communicator::init(device, size, member, &id)
        .context("joining a communicator")?;
    Ok(Some(Arc::new(nn::Group::new(communicator, size as u64)?)))
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
    ranks: &[u32],
) -> Result<ipc::StorageLaunch> {
    let request = proto::process::Request {
        id: 1,
        kind: Some(proto::process::request::Kind::Claim(
            proto::process::Claim {
                host: host.to_owned(),
                deployment: launch.deployment.clone(),
                scopes: ranks
                    .iter()
                    .map(|rank| proto::process::Scope {
                        group: launch.group.clone(),
                        rank: *rank,
                    })
                    .collect(),
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
        // read-only until it is promoted, and claiming it writable would give the scope two
        // writers. The launch file states the role, so the process never has to infer it.
        standby: launch.standby,
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
    // Why this process ends is the daemon's own diagnostic: the supervisor reports
    // "the engine process ended" and nothing more, so a handover that loses its engine
    // has no line to read unless this one says which way it went. A commanded end and a
    // failed epoch both stop the process, and only the second is a fault.
    match &ending {
        Ok(None) => tracing::warn!("the main process ends: a shutdown was commanded"),
        Ok(Some(failure)) => {
            tracing::warn!(
                reason = failure.reason(),
                "the main process ends: the epoch failed"
            )
        }
        Err(error) => tracing::warn!(%error, "the main process ends: the control thread failed"),
    }
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
/// How many of a request's events this process keeps for a takeover to replay.
///
/// The event window is the credit a consumer reports in, not the distance it can be behind by:
/// this process writes every event as it is produced, the daemon files them without bound, and the
/// body writes them to a socket the client drains at its own speed, so at a fast decode a response
/// that stops reading is tens of events behind. The pause is that case at its sharpest - a
/// predecessor's consumer stops at the pause while this process goes on producing - and it begins
/// before this process has an attached front to tell it so, so the wide bound is the ordinary one
/// and pruning on a consumer's report is what keeps the journal small in service.
///
/// **It is a memory budget, not a protocol bound.** An event is a token or a scheduling fact,
/// about 100 bytes with its wire form: 4096 of them is roughly 400 KB for one request, so a
/// deployment whose every stream stalls at the bound spends about 50 MB at 128 streams. Raising it
/// buys a longer stall before a takeover is refused; it does not change what the refusal means. A
/// takeover whose position is older than what is kept is refused by name rather than served as a
/// stream with a hole in it.
pub(crate) const HANDOVER_JOURNAL: usize = 4096;

/// One admitted request and the connection its events belong to.
///
/// A request belongs to the daemon front that submitted it until a `Takeover`
/// moves it to another (§7.5.1 step 4). The events the front's consumer has
/// not reported as consumed are kept, because a takeover from a position names
/// what has to be replayed: the predecessor freezes its response at the pause,
/// so its consumer stops advancing while this process goes on producing, and
/// the successor's body resumes from the position it last rendered.
struct Served {
    /// The daemon's identity for the request.
    request: u64,
    /// The scheduler rank a command for the request is routed to.
    rank: usize,
    /// What the request produced.
    events: engine::EventRx,
    /// The daemon front it belongs to: 0 is the daemon that launched this
    /// process, and each attached daemon adds one.
    front: usize,
    /// Events written to a front so far, which is the position a takeover
    /// names and the consumer reports.
    sent: u64,
    /// The events written to a front that a takeover from an earlier position
    /// has to replay, by their position.
    ///
    /// Bounded by [`HANDOVER_JOURNAL`], not by the event window the consumer's
    /// credit names: this process hands every event to the daemon as it is
    /// produced and the daemon files them into a channel it drains at the
    /// client's pace, so the consumer can be further behind the producer than any
    /// window. The events the consumer reported consumed are dropped as they are
    /// reported.
    kept: std::collections::VecDeque<KeptEvent>,
}

/// One journaled event: its position in the request's stream, the event, and the artifact bytes it
/// travels with. A replay without the bytes would be an artifact event the daemon cannot read.
type KeptEvent = (
    u64,
    proto::request::wire::Event,
    Option<std::sync::Arc<dyn proto::request::ArtifactMedia>>,
);

/// The most artifact bytes one frame carries, so a frame's size does not grow with the clip.
const ARTIFACT_PART: usize = 1 << 20;

pub(crate) struct MainProcess {
    control: Control,
    /// This node's schedulers, one per DP rank. The handle is the interface: each scheduler's thread
    /// owns the instance, so every call the control thread makes goes through the handle and the
    /// command channel carries the mutation (§2.4).
    schedulers: Vec<engine::SchedulerHandle>,
    /// One statistics reporter per scheduler, in the schedulers' order.
    ///
    /// A reporter keeps the cumulative baselines of the counters it has already
    /// reported, so each increment appears in exactly one snapshot: the same
    /// reporter must see every pass (§2.3's `/metrics`, `uniserve:num_requests_running`
    /// and its kind).
    stats_reporters: Vec<engine::SchedulerStatsReporter>,
    /// This node's device executors, shared with the schedulers that submit through them: one
    /// executor thread per device, reached both by this process's own verbs (`precopy`,
    /// `switchover`, `warmup`, `purge`) and by the scheduler's `Batches`.
    executors: Vec<Arc<Executor>>,
    memory: Vec<Memory>,
    /// The rank's Storage Server client, kept for the two calls of §2.5 that reach its ledger and no
    /// other: a park gives the cache's segments back, and a wake maps them again. Every caller takes
    /// `&mut` of its own `&Retained` (`impl ExclusiveStorage for &Retained`), so the device thread can
    /// hold one beside the control thread without a lock.
    storage: Vec<Arc<Retained>>,
    /// Whether this process's segments are given back to the Storage Server: a park sets it, a wake
    /// clears it. It exists so a process that never parked sends no wake at all - a wake for a pool
    /// that never slept is a no-op, but its round trip would still wait behind a queued capture on
    /// the first `Serve`, and the first token must not pay for a mechanism it did not use.
    slept: bool,
    blocking: ThreadPool,
    /// The blocking lane the store's listing and the module loads run on (§2.4, §7.3 step 2).
    modules: Lane<ModuleWork>,
    /// The store's listing read for the precopy in flight (§7.3 step 2). The lane reads it once and
    /// every later reader of that precopy - the keys a package declares, the candidates an operator
    /// has - answers from here, so the unbounded call is made where the lane is and never twice.
    store: Option<program_server::Listing>,
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
    /// When this process stated itself as `Ready` or `Standby`, and how many program-layer
    /// commands it has been given since the epoch began: a process that was read as ready and is
    /// given no command looks exactly like one that is working, from the daemon's side and from
    /// its own.
    announced_at: Option<Instant>,
    /// The state this process last stated on the request interface (§7.3 step 1). An attaching
    /// daemon is told the same state on the connection it gives (§7.5.1 step 2): a successor that
    /// is not told what this engine is never counts it as serving.
    reported_state: i32,
    /// Whether the daemon that launched this process closed its front: at a handover the
    /// predecessor leaves, and the attached daemons serve on. Stated once, and not polled
    /// again, because a closed connection answers every pass with the same error.
    daemon_closed: bool,
    /// What this process last announced it is, kept so that a daemon attaching mid-epoch is
    /// told it without asking the schedulers anything: the attach is a handover's step and a
    /// control thread blocked in a cache description cannot serve one.
    announced: Option<proto::request::wire::EngineInfo>,
    /// Whether the closing of the daemon's process interface has been reported: the interface
    /// stays closed, and the line is worth stating once.
    noted_process_closed: bool,
    commands_seen: u64,
    /// The two waits of the epoch's start that no event ends: the daemon's first command, and this
    /// node's warmup.
    command_wait: Wait,
    warmup_wait: Wait,
    /// A switchover whose answer is owed to the warmup it queued: the generation, and the peer to
    /// answer when admission opens. `Switched` means this process serves, so it waits for the
    /// buckets a batch would be planned into; a warmup that fails answers the switchover instead.
    pending_switch: Option<(u64, Option<u32>)>,
    /// When the warmup now running was queued: a warmup that outlives twice the stall bound is
    /// failed by name rather than left to answer nothing.
    warming_since: Option<Instant>,
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
    /// The kernel programs the serving generation was built from, by slot (§7.3 step 2). The shadow
    /// pass pairs a *changed* operator's kernel with the one serving, so "changed" is a fact about
    /// programs and never about generation numbers.
    serving_programs: Programs,
    /// The programs of the generation `retired` holds: a revert serves that generation again, and
    /// the changed set of the update after it is measured against the programs it serves under.
    retired_programs: Programs,
    /// The programs of the generation `prepared` holds, waiting for the switchover to make them
    /// serving.
    prepared_programs: Programs,
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
    /// Each admitted request with the stream it produces, the scheduler rank a command for it
    /// reaches, and the daemon front it belongs to (§7.5.1 step 4).
    streams: Vec<Served>,
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
    /// executor's row count includes the guidance prefix, which is not a request.
    max_running_requests: u32,
    /// The catalogue of prefill buckets this rank captures and whether the deployment stated it
    /// (§4.5): the ceiling a policy's `max_num_batched_tokens` is refused against (§4.4), named
    /// with the policy's own figure when that refusal is rendered.
    catalogue: PrefillCatalogue,
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
    /// A drain in progress: its schedulers park on the passes that follow (`advance_drain`).
    draining: Option<Draining>,
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
        let mut process =
            ProcessInterface::connect(&launch.processes, &launch.deployment, launch.engine)?;
        // Every command of the program layer arrives on this socket, and a stage waits for the
        // command before it: the control thread sleeps on it as it does on the request socket.
        control.watch_process(Some(process.connection.descriptor()));

        // §8: the rank's Storage Server and its claim are the node's, which the daemon answers for;
        // the retained weights outlive every world of the epoch, so the store is built once, before
        // the model loads, and keeps what it maps for the epoch's life.
        // A host tier is the build config's, and it needs prefix caching to have pages to keep.
        let host_kv_cache = build.kv_cache.host_tier && build.kv_cache.prefix_caching;
        // §2.1: this process serves every rank of its node, each on its own device.
        let rank_plans = plan.ranks()?;
        let rank_numbers: Vec<u32> = rank_plans.iter().map(EpochPlan::rank).collect();
        let storage = claim(&mut process, launch, &epoch.host, host_kv_cache, &rank_numbers)?;
        let ranks = load_ranks(rank_plans, &storage, launch, &epoch)?;

        // The owners of §4 are opened before any thread that borrows them, and
        // released in reverse order by `close`. The weights are mapped before the capacity plan
        // measures, so the plan sees a device with the model already in it - the figure it computes
        // is what is left, not what was there before the load. Pools sized first and weights after
        // would over-commit the device by exactly the model's size.
        let blocking = ThreadPool::new(plan.blocking_limits());
        let modules = blocking.lane::<ModuleWork>(BlockingKind::ModuleLoad)?;
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
        // A rank with no device serves nothing, and saying it here names the deployment's fault
        // rather than leaving the pool's figure to a plan that never sized one.
        if plan.devices().is_empty() {
            bail!("this rank serves no device");
        }
        let mut memories = Vec::with_capacity(ranks.len());
        let mut clients = Vec::with_capacity(ranks.len());
        let mut models = Vec::with_capacity(ranks.len());
        let mut inventories = Vec::with_capacity(ranks.len());
        let first_plan = ranks
            .first()
            .map(|rank| rank.plan.scheduled())
            .context("this process serves no rank")?;
        // The group's products, which every rank's runners reach through their own view of them.
        let group_products = GroupProducts::new(ranks.len());
        let rank_count = ranks.len() as u32;
        for (rank_index, rank) in ranks.into_iter().enumerate() {
            let LoadedRank {
                plan: rank_plan,
                client,
                model,
                mut memory,
                mut inventory,
                graph,
                groups,
            } = rank;
            // The scheduler sees the group as one pool, and each rank encodes its own units, so the
            // pool holds every rank's host tasks.
            inventory.host_task_capacity *= rank_count;
            // The device's one pool set is the resident owners' allocator (§4.1): it is opened with
            // the memory and handed to the executor thread, so every owner and every runner books
            // against the same allowance.
            let pools = memory.pools();
            // The scratch pool's figure comes from the one call that sized it, with the memory: the
            // shadow region and the state group's snapshot are one reservation (4.5), and a second
            // computation here could disagree with the pool the allocator reserved.
            let scratch = graph.scratch;
            // The shadow region alone: the twins allocate it, and the reservation beside it holds
            // the state group's snapshot too (7.3, 4.5).
            let shadow = graph.shadow;
            let products: Arc<dyn executor::runners::products::Products> = Arc::new(
                RankProducts::new(Arc::clone(&group_products), rank_index),
            );
            for device in rank_plan.devices() {
                let generation = Arc::new(Generation::new(epoch.generation, std::iter::empty()));
                let runners = rank_plan.runners(
                    *device,
                    &model,
                    &mut memory,
                    &generation,
                    &blocking,
                    &groups,
                )?;
                let executor = Executor::spawn(
                    *device,
                    runners,
                    rank_plan.events(),
                    Arc::clone(&pools),
                    Arc::clone(&products),
                    scratch,
                    shadow,
                )?;
                // §7.3 step 2 shadows a replaced operator beside the serving one, so the region it
                // runs in is mapped before any batch: the executor books it from the scratch pool
                // the graph storage sized. Nothing is serving yet, so the control thread waits here;
                // `serve` never does (§2.4).
                //
                // §4.5's warmup is not here: the epoch's first generation holds no kernel until the
                // daemon's precopy fills it - and a precopy that names no kernel (the startup
                // switch) fills it from the store's built programs - so there is no bucket to warm
                // yet. It runs at the first switchover, which is the boundary the first request is
                // served at.
                let shadow = executor.set_shadow()?;
                device_command(&watchdog, *device, "shadow region", shadow)
                    .with_context(|| format!("mapping the shadow region of device {}", device.0))?;
                // §2.5's two commands reach the Storage Server's ledger from the device thread,
                // which owns the stream a wake resets its units on, so the client is bound beside
                // the cache handle rather than kept only here.
                executor.bind_storage(Arc::clone(&client));
                executors.push(Arc::new(executor));
                generations.push(generation);
            }
            memories.push(memory);
            clients.push(client);
            models.push(model);
            inventories.push(inventory);
        }

        // This node's schedulers, one thread per DP rank, each handing its batches to the executors
        // above through `executor::Batches` (§2.4, §4.4). A node that holds none keeps an empty list
        // and serves only what its peers' schedulers ask of it; a node that is the scheduler node
        // and holds none would accept requests and dispatch nothing, which is refused by name.
        let schedulers = schedulers(
            &first_plan,
            launch,
            &models[0],
            &executors,
            &inventories[0],
            control.wake(),
        )?;

        let stats_reporters = (0..schedulers.len())
            .map(|_| engine::SchedulerStatsReporter::default())
            .collect();
        let mut process = Self {
            control,
            schedulers,
            stats_reporters,
            executors,
            memory: memories,
            storage: clients,
            slept: false,
            blocking,
            modules,
            store: None,
            update,
            watchdog,
            programs,
            precopy: None,
            warming: Vec::new(),
            switchovers: 0,
            announced_at: None,
            reported_state: proto::request::wire::EngineState::Starting as i32,
            daemon_closed: false,
            announced: None,
            noted_process_closed: false,
            commands_seen: 0,
            command_wait: Wait::default(),
            warmup_wait: Wait::default(),
            pending_switch: None,
            warming_since: None,
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
            catalogue: plan.prefill_catalogue(),
            serving_programs: Programs::new(),
            retired_programs: Programs::new(),
            prepared_programs: Programs::new(),
            standby: launch.standby,
            promoted: false,
            parked: None,
            draining: None,
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
        // The same release the `Park` command makes: this process serves nothing from here until a
        // promotion, and a standby holding the cache's segments while a successor loads is the
        // memory the handover needs. A pool that is not segmented has nothing to give back, and the
        // line says so.
        if let Err(error) = self.give_back_segments() {
            tracing::warn!(
                %error,
                "this standby could not give the cache's segments back; it parks with them still \
                 mapped"
            );
        }
        // The launch generation, which is what `Parked` states: the daemon files a report by it
        // (`supervision/roster.rs:236-244` reads `launch_of(engine, parked.generation)`), and the
        // program generation is zero here - a standby parks before any `Precopy` names one - so a
        // report stating that would name a launch that does not exist and be dropped. The `Park`
        // command states the same number for the same reason.
        self.send_report(proto::process::Report {
            kind: Some(proto::process::report::Kind::Parked(
                proto::process::Parked {
                    generation: self.control.engine(),
                },
            )),
        });
        Ok(())
    }

    /// The program generation this process holds and would serve from now on (§7.3).
    ///
    /// A prepared generation is the one a switchover takes, so it is what a promotion is about. With
    /// none prepared this is `HotUpdate`'s current generation, which is the daemon's own number from
    /// the last `Precopy` or switchover and `0` before any - never the launch number, which is what
    /// `Promoted.generation` states and a different question entirely.
    fn served_generation(&self) -> u64 {
        self.prepared
            .first()
            .map(|generation| generation.number())
            .unwrap_or_else(|| self.update.current())
    }

    /// §2.5: gives the parked process's cache segments back to the Storage Server, and states what
    /// went back and what stayed.
    ///
    /// The segments are the only pool a sleep releases: the graph pools and the scratch region are
    /// reservations whose captures keep their addresses (the segments' offsets are fixed inside one
    /// reserved range, so a wake is a remap and never a recapture), and the resident owners map
    /// through the same ledger for the process's life. A pool that is not segmented has nothing to
    /// give back, and the line says that rather than reading as a release that happened.
    fn give_back_segments(&mut self) -> Result<()> {
        let kept_before = self.memory.first().map(Memory::mapped_resident);
        // One command per device, and the control thread waits for each: the release is what the
        // successor's memory depends on, so a park that has not given its segments back is not
        // reported `Parked` yet. The device queue serialises commands, so this runs between
        // batches.
        for executor in &self.executors {
            let device = executor.device();
            let submission = executor
                .sleep()
                .with_context(|| format!("queueing device {device}'s sleep"))?;
            device_command(&self.watchdog, device, "the park's sleep", submission)?;
        }
        self.slept = true;
        let kept = self.memory.first().map(Memory::mapped_resident);
        if let (Some(before), Some(after)) = (kept_before, kept) {
            tracing::info!(
                given_back = before.saturating_sub(after),
                kept = after,
                pool = "the cache's segments",
                devices = self.executors.len(),
                "the park gave back the segments it no longer serves, and kept the pools it serves \
                 with"
            );
        }
        Ok(())
    }

    /// §2.5: takes the cache's segments back on every device, `limit` at a time, before admission
    /// opens. A wake is a remap, not a repair: the generation's captures keep their addresses, so no
    /// warmup follows it, and a wake that fails leaves admission closed and fails its caller by name.
    fn take_back_segments(&mut self) -> Result<()> {
        if !self.slept {
            return Ok(());
        }
        let kept_before = self.memory.first().map(Memory::mapped_resident);
        for executor in &self.executors {
            let device = executor.device();
            // No limit: this process takes back everything the Storage Server still holds for its
            // rank. A limit is a bound on a smaller wake, not a demand that everything return.
            let submission = executor
                .wake(None)
                .with_context(|| format!("queueing device {device}'s wake"))?;
            device_command(&self.watchdog, device, "the wake", submission)?;
        }
        self.slept = false;
        // The description this process answers a cache query from is refreshed here, and this is the
        // one place it can go stale: a wake maps the segments again, and a segment that came back at
        // a new offset is a different answer for a successor. Every scheduler answers it here - the
        // streams are not paused - where asking from the handover's own control pass would wait.
        if let Ok(info) = self.engine_info() {
            self.announced = Some(info);
        }
        let kept = self.memory.first().map(Memory::mapped_resident);
        if let (Some(before), Some(after)) = (kept_before, kept) {
            tracing::info!(
                taken_back = after.saturating_sub(before),
                kept_after = after,
                limit = "every segment the storage holds",
                devices = self.executors.len(),
                "the wake took its segments back before admission opened"
            );
        }
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
            let pass = Instant::now();
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
            // A transition moves one stage a pass, and what ends a stage is a device, a Program
            // Server or a scheduler rather than a socket. Passes run back to back while one is in
            // flight, so each stage costs what its work costs: a recovery is made of these stages,
            // and a client's stream stands still for all of them.
            if self.precopy.is_some() || !self.warming.is_empty() || self.draining.is_some() {
                due = due.min(Instant::now() + TRANSITION_POLL);
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
                    // Every front reads the same way: a frame from an attached daemon is served
                    // exactly as one from the daemon that launched this process, and the front it
                    // came from travels with it.
                    Ready::Daemon | Ready::Attached(_) => {
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
                    // What these announce is read by the pass below, whatever woke it.
                    Ready::Process | Ready::Woken | Ready::Idle => {}
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
            // A daemon owns this process's life: an interface it closed is the end of the epoch,
            // not a failure to report to it, so the pass stops by name and the process exits
            // cleanly. At a hot restart the interface that closes is the predecessor's, and the
            // successor holds the deployment: this process is still serving it, and ending here
            // would leave the successor with a front into nothing (§7.5.1).
            if self.process.closed() {
                self.control.watch_process(None);
                let serving = self.control.live_attached();
                if serving == 0 {
                    self.stopping = true;
                } else if !self.noted_process_closed {
                    self.noted_process_closed = true;
                    tracing::warn!(
                        serving,
                        "a daemon closed this process's interface; the attached daemons still \
                         hold this engine, so it serves on"
                    );
                }
            }

            // A precopy in flight advances one stage per pass, never waiting on
            // its builds or its devices (§7.3 step 2).
            self.note_waits();
            self.advance_precopy()?;
            self.advance_drain()?;
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
            // A pass that took seconds was blocked somewhere inside it, and a process that is
            // blocked reads as one that never answered: a handover's attach waits on exactly
            // that difference.
            if pass.elapsed() > Duration::from_secs(5) {
                tracing::warn!(
                    millis = pass.elapsed().as_millis(),
                    "a serve pass took longer than five seconds"
                );
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
                if let Err(error) = self.update.precopy_failed(generation, failure.clone()) {
                    // The precopy this failure belongs to is no longer the one in flight - a revert
                    // or a later precopy took its place. The failure is still reported, and a
                    // serving process is not ended over bookkeeping that moved on.
                    tracing::warn!(generation, %error, "a precopy failure arrived for no precopy");
                }
                self.report(ProgramStatus::PrecopyFailed {
                    generation,
                    failure,
                });
                Ok(None)
            }
            Action::Revert => {
                let generation = self.update.phase().generation().unwrap_or_default();
                let reason = failure.reason();
                let status = self.revert(generation, failure)?;
                // The answer goes out from here: a revert this process started - a warmup that
                // failed, a device that could not continue - has no command to answer, and the
                // daemon waiting on the switchover learns it from this report or not at all. Two
                // minutes of silence and then a timeout is what its absence looks like.
                self.report(status);
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
        // Every front is read in turn, the launching daemon's first: a submission names the front
        // it came from, and so does every command for a request already admitted.
        let fronts = 1 + self.control.attached_fronts();
        for front in 0..fronts {
            if front == 0 && self.daemon_closed {
                continue;
            }
            loop {
                // A frame from a closed front ends that front and nothing else: its requests stay
                // parked for the takeover that claims them (§7.5.1 step 4).
                let Some(connection) = self.control.front_mut(front) else {
                    break;
                };
                let message = match connection.try_recv() {
                    Ok(Some(message)) => message,
                    Ok(None) => break,
                    Err(error) => {
                        if front == 0 {
                            // The launching daemon leaving is the handover's own step: the
                            // attached daemons hold this process, and a front polled after it
                            // closed answers every pass with the same error - which is a log
                            // that hides the handover's own lines behind it.
                            self.daemon_closed = true;
                            tracing::warn!(
                                %error,
                                "the daemon that launched this process closed its front; the \
                                 attached daemons serve on, so this front is not read again"
                            );
                        } else {
                            tracing::warn!(%error, front, "reading a daemon front failed");
                            self.control.detach(front);
                        }
                        break;
                    }
                };
                if let Some(failure) = self.daemon_message(front, message)? {
                    return Ok(Some(failure));
                }
            }
        }
        Ok(None)
    }

    /// Serves the daemon's request interface: submissions and per-request
    /// commands are routed to the scheduler that holds the request (§2.3), and
    /// program-layer commands run the driver of §7.3.
    fn daemon_message(
        &mut self,
        front: usize,
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
                            front,
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
                                front,
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
                            self.streams.push(Served {
                                request: id,
                                rank,
                                events: stream,
                                front,
                                sent: 0,
                                kept: std::collections::VecDeque::new(),
                            });
                            self.send_event(
                                front,
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
                            self.reject(front, id, kind, error.to_string());
                        }
                    }
                }
                Ok(None)
            }
            proto::request::wire::to_engine::Kind::Command(command) => {
                self.request_command(front, command)?;
                Ok(None)
            }
            proto::request::wire::to_engine::Kind::Engine(command) => {
                // What a daemon commands of the engine arrives on the connection it made, and
                // a handover's attach is the one such command whose silence reads as a hang.
                // Stated where it is read, so "the engine never saw it" and "the engine
                // refused it" are different lines in a run.
                tracing::info!(
                    front,
                    command = command.kind.as_ref().map_or("none", engine_command_name),
                    "the engine read an engine command"
                );
                self.engine_command(front, command)
            }
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
                    // §2.5: the schedulers are parked and nothing is in flight, so the cache's
                    // segments can go back to the Storage Server - a parked process that keeps them
                    // holds a successor's memory. A park that cannot give them back is still a park:
                    // the process serves nothing, and failing the epoch over a release would end a
                    // process that a promotion can still use. It says so instead.
                    if let Err(error) = self.give_back_segments() {
                        tracing::warn!(
                            %error,
                            "the park could not give the cache's segments back; it is parked with \
                             them still mapped"
                        );
                    }
                    self.send_report(proto::process::Report {
                        kind: Some(proto::process::report::Kind::Parked(
                            proto::process::Parked {
                                // The daemon's epoch layer promotes a child by its
                                // launch generation: this process's own number, not
                                // the program generation (which is zero until a
                                // `Precopy` names one).
                                generation: self.control.engine(),
                            },
                        )),
                    });
                }
                Some(proto::process::command::Kind::Promote(_)) => {
                    self.promote_in_place()?;
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
        self.commands_seen += 1;
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
                // The receipt: this line says the daemon's step reached this process and what it
                // carried, before anything is resolved, built or installed. A run that stops with
                // no such line stopped before the daemon sent it, and one that stops after it
                // stopped here.
                tracing::info!(
                    generation,
                    operators = programs.len(),
                    policy = %policy.backend,
                    policy_revision = policy.revision,
                    "the daemon's Precopy arrived"
                );
                self.start_precopy(generation, &programs, &policy, reply)?;
                return Ok(None);
            }
            ProgramCommand::Switchover { generation, step } => {
                tracing::info!(
                    generation,
                    step = step.unwrap_or_default(),
                    "received the daemon's switchover"
                );
                match self.switchover(generation, step, reply)? {
                    Some(status) => status,
                    // The answer is owed to the warmup: it is sent from `advance_warmup` when
                    // admission opens, or replaced by the failure that keeps admission shut.
                    None => return Ok(None),
                }
            }
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
        // §7.3's steps are answered here. A run that stops between two of them needs to know which
        // one was answered last, and the daemon's own log of its commands is not this side's.
        // The reason travels with the status, and a run that reads only this log needs it too.
        let failure = match &status {
            ProgramStatus::PrecopyFailed { failure, .. } => format!("{failure:?}"),
            _ => String::new(),
        };
        tracing::info!(
            status = status_name(&status),
            to_peer = reply.is_some(),
            failure,
            "answering the epoch's program layer"
        );
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
    /// and admits the lane's work - the store's listing, then one install per
    /// device - whose builds run while serving goes on.
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
        // scheduling policy alone, because on a host that runs the executor path there is no kernel
        // package to prepare (`crates/daemon/src/leader.rs`, `install_programs`). This process
        // installs what it can actually launch, so the store's listing is read first, on the module
        // lane, rather than capturing an empty generation.
        //
        // A policy-only *update* also names no kernel, and resolving there would be wrong twice
        // over: every operator would be re-installed as a new generation, so every kernel's `Arc`
        // would differ from the serving one and every captured bucket would be rebound or
        // recaptured for a change no program made. The two states are told apart by what this
        // process already serves - a startup has no kernel at all, an update has all of them.
        //
        // Each precopy reads the store at most once, and never one precopy's answer in another's
        // place: a package built between two updates is a program the second one may install.
        //
        // The first generation's read is the lane's. An update's - the keys a package declares, the
        // candidates an operator has - is still made here, on the control thread, by whichever step
        // asks for it first; it is the same call the startup switch no longer makes, and the step
        // that would carry it to the lane is the candidate walk, not this one.
        self.store = None;
        let never_installed = self
            .generations
            .iter()
            .all(|generation| generation.is_empty());
        // An update names the epoch's whole set, and only what differs from what serves is
        // installed: a kernel that stays keeps its `Arc`, so its buckets and its graphs stay too.
        let updated;
        let programs = if never_installed {
            programs
        } else {
            updated = updated_programs(programs, &self.serving_programs);
            &updated
        };
        let mut precopy = Precopy {
            generation,
            reply,
            changed: Vec::new(),
            policy: policy.clone(),
            programs: programs.clone(),
            candidates: Candidates::new(),
            tried: std::collections::BTreeSet::new(),
            backends: BTreeMap::new(),
            unavailable: UnavailableCell::default(),
            stage_name: "modules",
            since: Instant::now(),
            warned: false,
            learned: BTreeMap::new(),
            policy_requested: false,
            next: Vec::new(),
            policies: Vec::new(),
            stage: PrecopyStage::Done,
        };

        if never_installed {
            // The store's listing is the lane's first work and the one call that leaves this
            // process before a module does: it is admitted where a build is admitted, so a store
            // that never answers costs the lane's capacity and not the control thread.
            let list = ListStore {
                programs: Arc::clone(&self.programs),
            };
            match self.modules.admit(ModuleWork::List(list)) {
                Ok(task) => {
                    precopy.stage = PrecopyStage::Modules(ModuleLane::Listing(task));
                    self.precopy = Some(precopy);
                    return Ok(());
                }
                Err(error) => {
                    return self.abandon_precopy(
                        generation,
                        UpdateFailure::Precopy {
                            reason: format!("admitting the store's listing: {error}"),
                        },
                        reply,
                    );
                }
            }
        }

        tracing::info!(
            generation,
            operators = programs.len(),
            candidates = precopy
                .candidates
                .values()
                .filter(|ordered| ordered.len() > 1)
                .count(),
            devices = self.executors.len(),
            "this process is installing the generation the daemon named"
        );
        let (tasks, unavailable) = match self.admit_installs(programs, &BTreeMap::new()) {
            Ok(admitted) => admitted,
            Err(failure) => return self.abandon_precopy(generation, failure, reply),
        };
        precopy.unavailable = unavailable;
        precopy.stage = PrecopyStage::Modules(ModuleLane::Installs(tasks));
        self.precopy = Some(precopy);
        Ok(())
    }

    /// The epoch's first generation: the store's set, with the deployment's named programs deciding
    /// the phases their own packages cover (§7.3 step 2).
    ///
    /// A named program chooses its operator's implementation - its backend and revision - and never
    /// states the operator set: a deployment that pins one operator's kernel still runs every other
    /// operator its buckets launch. So the first generation takes this host's store as its set and
    /// the named programs resolved against it, which is the only reading under which `kernels=1` on
    /// a host that built twenty-seven launches a model rather than a fragment.
    ///
    /// A named program also states nothing about the phases its operator's key spans, and an
    /// operator split across two packages - a prefill module and a decode one - needs both, so the
    /// name chooses the implementation *for its own keys* and the store keeps the other phases
    /// (`operator_slots`).
    fn first_generation(
        &self,
        named: &Programs,
    ) -> std::result::Result<(Programs, Candidates), UpdateFailure> {
        let (listed, ordered) = self.listed_programs(named)?;
        tracing::info!(
            slots = listed.len(),
            named = named.len(),
            operators = ordered.len(),
            alternative = ordered.values().filter(|held| held.len() > 1).count(),
            "this host's built programs, with the deployment's named ones resolved first"
        );
        Ok((listed, ordered))
    }

    /// The Program Server's store as it stands: every installed package, its revision and the
    /// descriptors its modules cover for each target.
    fn listing(&self) -> std::result::Result<program_server::Listing, String> {
        // One read per precopy, on the module-load lane; every later reader of that precopy answers
        // from it. The update asks twice - once for the set, once for the keys a package declares -
        // and a second read here would put the unbounded call back on the control thread, which is
        // where the lane exists to keep it from. A reader with no listing in hand is one outside a
        // precopy, and it reads the store with the lines below.
        if let Some(store) = &self.store {
            return Ok(store.clone());
        }
        tracing::info!("asking the Program Server for its installed programs");
        let mut client = self
            .programs
            .lock()
            .map_err(|_| "the Program Server client was poisoned".to_owned())?;
        let listing = client
            .list()
            .map_err(|error| format!("reading the Program Server's store: {error}"))?;
        tracing::info!(
            packages = listing.packages.len(),
            "the Program Server's listing answered"
        );
        Ok(listing)
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
    fn listed_programs(
        &self,
        named: &Programs,
    ) -> std::result::Result<(Programs, Candidates), UpdateFailure> {
        let invalid = |reason: String| UpdateFailure::Precopy { reason };
        let listing = self.listing().map_err(invalid)?;
        let chosen = select_programs(&listing.packages, &self.targets);
        if chosen.is_empty() && named.is_empty() {
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
            named = named.len(),
            "resolved this host's built programs, the deployment's named ones first"
        );
        // What this host's store supplies, against what this build implements: the difference is
        // the fact an operator needs, because it names the package nobody built (an operator whose
        // module has no package cannot be supplied by the deployment either), and it is bounded -
        // an operator the deployment never launches may be in it, which is why the line states the
        // count rather than claiming these are needed.
        let supplied: std::collections::BTreeSet<&str> = chosen
            .keys()
            .filter_map(|op| ops::operator_name(op))
            .collect();
        let missing: Vec<&str> = ops::operators()
            .iter()
            .copied()
            .filter(|operator| !supplied.contains(operator))
            .collect();
        if !missing.is_empty() {
            tracing::warn!(
                supplied = supplied.len(),
                implemented = ops::operators().len(),
                missing = ?missing,
                "this host's store supplies fewer operators than the build implements"
            );
        }
        // What this host has built for every package the store holds, keyed by the package -
        // (operator, backend, revision) - because that is what a phase split is read from: the keys
        // a package built are the phases it can serve.
        let mut built: BTreeMap<(String, String, u32), Vec<String>> = BTreeMap::new();
        for package in &listing.packages {
            let mut keys: Vec<String> = package
                .modules
                .iter()
                .filter(|module| self.targets.contains(&module.target))
                .map(|module| module.descriptor.clone())
                .collect();
            keys.sort_unstable();
            keys.dedup();
            if !keys.is_empty() {
                built.insert(
                    (
                        package.op.clone(),
                        package.backend.clone(),
                        package.revision,
                    ),
                    keys,
                );
            }
        }
        resolve_slots(named, &chosen, &built, &listing.packages).map_err(invalid)
    }

    /// One operator's program from this host's store, for a walk that named an operator the
    /// deployment left unnamed.
    ///
    /// The store's listing is the authority for what this host can launch: its packages are the
    /// modules built for this host's targets. The first candidate of the operator is taken - the
    /// deployment expressed no preference, so the store's order decides - and `None` says the store
    /// describes nothing for the operator, which the caller states as a missing package rather than
    /// a missing key.
    fn borrow_program(
        &self,
        operator: &str,
    ) -> std::result::Result<Option<(program_server::program::ProgramSlot, Program)>, String> {
        let listing = self.listing()?;
        for (op, ordered) in select_programs(&listing.packages, &self.targets) {
            if ops::operator_name(&op) != Some(operator) {
                continue;
            }
            let Some((backend, revision)) = ordered.first() else {
                continue;
            };
            return Ok(Some((
                program_server::program::ProgramSlot { op, phase: None, part: None },
                Program {
                    backend: backend.clone(),
                    revision: *revision,
                    arguments: Default::default(),
                },
            )));
        }
        // A package the store holds and has never built is invisible to `select_programs`, and an
        // operator no model forward reaches is exactly that: `slot_commit` is launched by the
        // runner's own step, so no capture it is missing from comes before the walk names it, and
        // nothing in the deployment can name its package either. The store holds the package and
        // declares the keys it can build (`[kernel.descriptors]`), so installing it and asking for
        // the walked key is what builds it - otherwise a generation refuses an operator the store
        // holds, which reads as a store defect and is not one.
        match declared_program(&self.listing()?.packages, operator) {
            Some((op, backend, revision)) => Ok(Some((
                program_server::program::ProgramSlot { op, phase: None, part: None },
                Program {
                    backend,
                    revision,
                    arguments: Default::default(),
                },
            ))),
            None => Ok(None),
        }
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
    ) -> std::result::Result<(Vec<Arc<HostTask<ModuleWork>>>, UnavailableCell), UpdateFailure> {
        let invalid = |reason: String| UpdateFailure::Precopy { reason };
        // The Program Server's listing, read at most once and only when a program's keys cannot be
        // taken from a generation that already launched it: the epoch's first generation has
        // nothing to copy from, and the walk would otherwise discover one descriptor per capture.
        let mut listing: Option<program_server::Listing> = None;
        // One request per package, not one per generation: the Program Server hands every chosen
        // slot of an operator all the keys a request names for that operator, so a generation whose
        // operator the store split across two backends - fa4's prefill module and trtllm's decode
        // one - would have each of them asked for the other's keys, and the package that cannot
        // build a phase it does not declare would refuse the generation. The keys of a slot are
        // its own package's, which is what makes the package the request's unit.
        let mut groups: BTreeMap<(String, u32), (Vec<ProgramRequest>, Vec<Descriptors>)> =
            BTreeMap::new();
        for (slot, program) in programs {
            let Some(operator) = ops::operator_name(&slot.op) else {
                return Err(invalid(format!(
                    "operator {} has no implementation in this build",
                    slot.op
                )));
            };
            // A generation may hold two packages of one operator, one per phase: a key belongs to
            // the phase its own text states, so a phase's slot takes only the keys of its phase and
            // no package is ever asked for a key it cannot build. A key that states no phase fits
            // every slot - it is the only key a phase-less operator has.
            let mut keys: Vec<String> = descriptor_keys(&self.generations[0], operator)
                .into_iter()
                .filter(|key| key_fits_phase(key, slot.phase))
                .collect();
            // A generation that launched this operator states the keys it runs; a key the walk has
            // taught since is added to them, never substituted for them. The first generation has
            // nothing serving, so its keys come from the package - once, not one capture at a time -
            // and a taught key joins them for the packages the store does not describe.
            let mut known: Vec<String> = learned
                .get(operator)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|key| key_fits_phase(key, slot.phase))
                .collect();
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
            // A slot of a divided operator holds its own package's descriptors alone: a generation
            // that launched the operator states all of them.
            if slot.part.is_some() {
                if listing.is_none() {
                    listing = Some(self.listing().map_err(invalid)?);
                }
                if let Some(store) = listing.as_ref() {
                    let built = built_keys(store, &slot.op, program, &self.targets);
                    keys.retain(|key| built.contains(key));
                }
            }
            keys.sort_unstable();
            keys.dedup();
            let group = groups
                .entry((program.backend.clone(), program.revision))
                .or_default();
            merge_request(group, slot, program, keys);
        }

        let mut tasks = Vec::with_capacity(self.executors.len());
        // One cell shared by every device's load of this generation: the first structured refusal
        // any of them meets is the one the walk answers with another candidate program.
        let unavailable = UnavailableCell::default();
        for (index, executor) in self.executors.iter().enumerate() {
            let ordinal = executor.device();
            let device = platform::Device::new(ordinal)
                .map_err(|error| invalid(format!("opening device {}: {error}", ordinal.0)))?;
            // `targets` holds each distinct architecture once, so a device's own is read from it.
            let target = crate::engine_plan::target_of(&device);
            if !self.targets.contains(&target) {
                return Err(invalid(format!("device {} has no build target", ordinal.0)));
            }
            // One request per package, in the store's own order so two runs ask alike.
            let requests: Vec<PrepareRequest> = groups
                .values()
                .map(|(programs, descriptors)| PrepareRequest {
                    targets: vec![target.clone()],
                    programs: programs.clone(),
                    descriptors: descriptors.clone(),
                })
                .collect();
            let install = Install {
                programs: Arc::clone(&self.programs),
                device,
                target,
                requests,
                unavailable: Arc::clone(&unavailable),
            };
            match self.modules.admit(ModuleWork::Install(install)) {
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
        // The slot the walk moves on is the one whose *program* the store refused: an operator the
        // store splits across two packages has a slot per phase, and advancing the other one would
        // replace a program that served its own keys perfectly.
        let refused_backend = refused.backend().to_owned();
        let Some((slot, current)) = precopy
            .programs
            .iter()
            .find(|(slot, program)| slot.op == op && program.backend == refused_backend)
            .or_else(|| precopy.programs.iter().find(|(slot, _)| slot.op == op))
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
                .then_with(|| right.1.1.cmp(&left.1.1))
                .then_with(|| left.1.0.as_str().cmp(right.1.0.as_str()))
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
        // A warmup still running past twice the stall bound is a device that will not finish it: its
        // answer is owed to the daemon's switchover wait, so silence here is a timeout with no cause
        // in the log. Failing it by name reverts the update and leaves the process serving what it
        // held.
        if let (Some(since), true) = (self.warming_since, !self.warming.is_empty()) {
            if since.elapsed() > self.watchdog.stall_bound() * 2 {
                let (device, generation) = self
                    .warming
                    .first()
                    .map_or((0, 0), |(device, _, generation)| (*device, *generation));
                let failure = UpdateFailure::Device {
                    device,
                    error: format!(
                        "warming generation {generation} on device {device} did not finish in {}s",
                        since.elapsed().as_secs()
                    ),
                    batch: None,
                };
                self.warming.clear();
                self.warming_since = None;
                self.pending_switch = None;
                return self.failure(failure);
            }
        }
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
                    // The answer owed to the switchover goes out from the revert below, as
                    // `Reverted` - the status the daemon's wait accepts for a generation that took
                    // over and went back. Reading it as served is what this ordering prevents.
                    self.pending_switch = None;
                    self.warming_since = None;
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
        // The switchover's answer is owed to the warmup, and it is sent here - after admission
        // opened - so that `Switched` states the one thing it means: this process can serve.
        if self.warming.is_empty() {
            if let Some((generation, reply)) = self.pending_switch.take() {
                tracing::info!(
                    generation,
                    "answering the switchover now that admission is open"
                );
                self.answer(reply, ProgramStatus::Switched { generation });
            }
        }
        Ok(None)
    }

    /// Reports the two waits of the epoch's start when they outlast a stall bound: the daemon's
    /// first program-layer command, and this node's warmup. Neither fails anything - the failures
    /// they precede state themselves - but a run that hangs here now says what it was waiting for.
    fn note_waits(&mut self) {
        let now = Instant::now();
        let bound = self.watchdog.stall_bound();
        self.command_wait
            .note(self.announced_at.is_some() && self.commands_seen == 0, now);
        self.command_wait
            .report("the daemon's first program-layer command", 0, bound, now);
        // The report is due inside the daemon's own window for the switchover this warmup answers: a
        // wait reported later is a wait nobody reads, because the daemon gives up first and the log
        // says only that something timed out.
        let warmup_bound = bound.min(DAEMON_COMMAND_WINDOW / 2);
        self.warmup_wait.note(!self.warming.is_empty(), now);
        let generation = self
            .warming
            .first()
            .map_or(0, |(_, _, generation)| *generation);
        if self
            .warmup_wait
            .report("this node's warmup", generation, warmup_bound, now)
        {
            // The thread that is stuck cannot say what it is stuck in, so each device states the job
            // it performs: a bucket's capture and a launch look identical from outside and are
            // different defects.
            for executor in &self.executors {
                tracing::warn!(
                    device = executor.device().0,
                    performing = executor
                        .performing()
                        .unwrap_or_else(|| "between commands".to_owned()),
                    captures = executor.capturing(),
                    "a device that has not finished its warmup"
                );
            }
        }
    }

    /// The operators this node's devices could not compare, each with its reason, in device order
    /// and without repeats (7.3 step 2).
    ///
    /// It is carried by the precopy's answer so an update that has no evidence for an operator says
    /// so in the record the daemon reads, not only in this process's log: empty is the statement
    /// that every operator the update replaced was compared.
    fn uncompared(&self) -> Vec<String> {
        let mut entries: Vec<String> = Vec::new();
        for executor in &self.executors {
            for entry in executor.uncompared() {
                if !entries.contains(&entry) {
                    entries.push(entry);
                }
            }
        }
        entries
    }

    /// §7.3 step 2, second half: advances the precopy at most one stage. It
    /// never waits on a job: a stage that has not finished leaves the precopy
    /// as it is, and the next serve pass looks again.
    fn advance_precopy(&mut self) -> Result<()> {
        let Some(mut precopy) = self.precopy.take() else {
            return Ok(());
        };
        let generation = precopy.generation;
        // Every stage of §7.3 step 2 is polled rather than waited for, so a stage that never
        // advances is the one shape of hang this process can have. It says so twice: when it enters
        // a stage, and once, by name, when that stage has stood longer than the watchdog's stall
        // bound - the same bound the §2.4 rule uses for a job, applied to the step itself.
        let entered = stage_name(&precopy.stage);
        if entered != precopy.stage_name {
            precopy.stage_name = entered;
            precopy.since = Instant::now();
            precopy.warned = false;
            tracing::info!(
                generation,
                stage = entered,
                "the precopy entered a stage of the update"
            );
        } else if !precopy.warned && precopy.since.elapsed() > self.watchdog.stall_bound() {
            precopy.warned = true;
            tracing::warn!(
                generation,
                stage = entered,
                waiting_on = %pending_work(&precopy.stage, &self.executors, &precopy.policies),
                waited_s = precopy.since.elapsed().as_secs(),
                "the precopy has stood in this stage past the watchdog's stall bound and nothing \
                 has answered: the Program Server, a device or a scheduler is the one to look at"
            );
        }
        // What the stage was waiting on, read before the stage is taken out: a bound that named
        // `no stage` because the driver had already replaced it would say nothing at all, which is
        // the opposite of what the line is for.
        let pending = pending_work(&precopy.stage, &self.executors, &precopy.policies);
        let stage = std::mem::replace(&mut precopy.stage, PrecopyStage::Done);

        match stage {
            PrecopyStage::Modules(ModuleLane::Listing(task)) => {
                let answer = match task.outcome() {
                    // The store has not answered yet: the lane keeps the task and this pass ends
                    // with the precopy where it was, as every stage of §7.3 step 2 does.
                    None => {
                        precopy.stage = PrecopyStage::Modules(ModuleLane::Listing(task));
                        self.precopy = Some(precopy);
                        return Ok(());
                    }
                    Some(Outcome::Success(answer)) => match &*answer {
                        ModuleAnswer::Listed(listing) => listing.clone(),
                        // This stage holds the listing alone; an install here is a fault in this
                        // process, and it is refused by name rather than read as a listing.
                        ModuleAnswer::Installed(_) => {
                            return self.abandon_precopy(
                                generation,
                                UpdateFailure::Precopy {
                                    reason: "the module lane answered the store's listing with a \
                                             kernel load"
                                        .to_owned(),
                                },
                                precopy.reply,
                            );
                        }
                    },
                    // The listing's own refusal already names the read: it is the store's answer,
                    // not the store, that failed.
                    Some(Outcome::Failed(error)) => {
                        return self.abandon_precopy(
                            generation,
                            UpdateFailure::Precopy {
                                reason: format!("reading the Program Server's store: {error}"),
                            },
                            precopy.reply,
                        );
                    }
                    Some(Outcome::Cancelled) => {
                        return self.abandon_precopy(
                            generation,
                            UpdateFailure::Precopy {
                                reason: "the store's listing was cancelled".to_owned(),
                            },
                            precopy.reply,
                        );
                    }
                };

                // The store has said what it holds: the first generation's set is resolved from it
                // and the installs are admitted on this same lane, behind the listing.
                self.store = Some(answer);
                let named = precopy.programs.clone();
                let (resolved, candidates) = match self.first_generation(&named) {
                    Ok(resolved) => resolved,
                    Err(failure) => {
                        return self.abandon_precopy(generation, failure, precopy.reply);
                    }
                };
                tracing::info!(
                    generation,
                    operators = resolved.len(),
                    candidates = candidates
                        .values()
                        .filter(|ordered| ordered.len() > 1)
                        .count(),
                    devices = self.executors.len(),
                    "this process is installing the generation the daemon named"
                );
                let (tasks, unavailable) = match self.admit_installs(&resolved, &BTreeMap::new()) {
                    Ok(admitted) => admitted,
                    Err(failure) => {
                        return self.abandon_precopy(generation, failure, precopy.reply);
                    }
                };
                precopy.programs = resolved;
                precopy.candidates = candidates;
                precopy.unavailable = unavailable;
                precopy.stage = PrecopyStage::Modules(ModuleLane::Installs(tasks));
                self.precopy = Some(precopy);
                Ok(())
            }
            PrecopyStage::Modules(ModuleLane::Installs(tasks)) => {
                let mut loaded = Vec::with_capacity(tasks.len());
                for task in &tasks {
                    match task.outcome() {
                        None => {
                            // A build is still running: the tasks keep their
                            // results and the lane keeps its capacity.
                            precopy.stage = PrecopyStage::Modules(ModuleLane::Installs(tasks));
                            self.precopy = Some(precopy);
                            return Ok(());
                        }
                        Some(Outcome::Success(answer)) => match &*answer {
                            ModuleAnswer::Installed(entries) => loaded.push(entries.clone()),
                            // This stage holds installs; the listing has its own arm above.
                            ModuleAnswer::Listed(_) => {
                                return self.abandon_precopy(
                                    generation,
                                    UpdateFailure::Precopy {
                                        reason: "the module lane answered a kernel load with the \
                                                 store's listing"
                                            .to_owned(),
                                    },
                                    precopy.reply,
                                );
                            }
                        },
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
                                        precopy.stage =
                                            PrecopyStage::Modules(ModuleLane::Installs(fresh));
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
                // A device that never answers is a process that serves nothing until the epoch
                // ends: the stage is polled, so without this bound the only line it prints is the
                // stall warning, and the run waits out the daemon's whole prepare bound to learn
                // what the log already showed. Preparing an installed generation is a rebind or a
                // recapture - seconds of capture and a pinned upload, never a compile - so a stage
                // that has stood past twice the watchdog's stall bound is failed here by name,
                // saying what never answered. Serving keeps what it had; the update does not.
                if precopy.since.elapsed() > self.watchdog.stall_bound() * 2 {
                    // The arm holds two waits, and naming the wrong one costs a run: a policy build
                    // that never answers is not a device that never answered, and the previous
                    // message said the device either way. The policy is the narrower of the two, so
                    // it is asked first.
                    let policy = matches!(
                        poll_answers(&mut precopy.policies),
                        Some(StageFailure::Pending)
                    );
                    let cause = if policy {
                        "a scheduler's policy build never answered"
                    } else {
                        "this node's devices never answered their preparation"
                    };
                    return self.abandon_precopy(
                        generation,
                        UpdateFailure::Precopy {
                            reason: format!(
                                "{cause} after {}s ({pending}): the wait named there is the one \
                                 to look at",
                                precopy.since.elapsed().as_secs()
                            ),
                        },
                        precopy.reply,
                    );
                }
                // The policy's module is built on each scheduler's blocking thread, so its answer
                // arrives here as well: a policy that did not prepare fails the update before any
                // device cuts over, which is what §4.4's dry run is for.
                match poll_answers(&mut precopy.policies) {
                    Some(StageFailure::Pending) => {
                        precopy.stage = PrecopyStage::Executors(submissions);
                        self.precopy = Some(precopy);
                        return Ok(());
                    }
                    Some(StageFailure::Failed(reason)) => {
                        let stated = precopy
                            .policy
                            .arguments
                            .get(TOKEN_LIMIT)
                            .map(|value| value.to_string());
                        let error = limit_refusal(&reason, stated.as_deref(), self.catalogue);
                        return self.abandon_precopy(
                            generation,
                            UpdateFailure::Policy { error },
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
                            // The walk can name an operator this generation holds no program for:
                            // a deployment names the implementations it pins and says nothing about
                            // the rest, so the rest come from this host's store - one borrow per
                            // operator, and a store that describes none stops the update with a
                            // sentence naming the package rather than a key.
                            if !precopy.names(&operator) {
                                match self.borrow_program(&operator) {
                                    Ok(Some((slot, program))) => {
                                        tracing::info!(
                                            %operator,
                                            backend = %program.backend,
                                            revision = program.revision,
                                            generation,
                                            "the walk named an operator the deployment left \
                                             unnamed: this host's store supplies it"
                                        );
                                        precopy.programs.insert(slot, program);
                                    }
                                    Ok(None) => {
                                        return self.abandon_precopy(
                                            generation,
                                            UpdateFailure::Precopy {
                                                reason: format!(
                                                    "{operator} has no kernel in this host's \
                                                     store and the deployment does not name one"
                                                ),
                                            },
                                            precopy.reply,
                                        );
                                    }
                                    Err(error) => {
                                        return self.abandon_precopy(
                                            generation,
                                            UpdateFailure::Precopy { reason: error },
                                            precopy.reply,
                                        );
                                    }
                                }
                            }
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
                            precopy.stage = PrecopyStage::Modules(ModuleLane::Installs(tasks));
                            self.precopy = Some(precopy);
                            return Ok(());
                        }
                    }
                }

                // §7.3 step 2's shadow pass pairs a changed operator's kernel with the one serving,
                // in the buckets a resident request occupies, so it is owed only when an operator's
                // program changed (§4.5). The set is empty on an epoch's first precopy - nothing
                // serves, so no kernel has an old one to be paired against - and asking every
                // scheduler there waits for a step an idle epoch never runs.
                let changed = changed_operators(&precopy.programs, &self.serving_programs);
                if changed.is_empty() {
                    tracing::info!(
                        generation,
                        "no operator's program changed, so this precopy asks for no shadow step"
                    );
                    return self.finish_precopy(precopy);
                }
                tracing::info!(
                    generation,
                    changed = changed.len(),
                    operators = %changed.join(", "),
                    "the operators this update changed, whose kernels the shadow step pairs"
                );

                // Shadow execution: every scheduler shadows one step of its own resident requests
                // against the prepared generation, in the buckets those requests occupy, and
                // answers once that step ran (§7.3 step 2). The paired kernels compare themselves
                // by `Op::tolerance` and never write resident memory.
                let mut shadow = Vec::with_capacity(self.schedulers.len());
                for scheduler in &self.schedulers {
                    shadow.push(scheduler.shadow(generation));
                }
                precopy.changed = changed;
                precopy.stage = PrecopyStage::Shadow(shadow);
                self.precopy = Some(precopy);
                Ok(())
            }
            PrecopyStage::Shadow(mut answers) => {
                // A shadow step runs inside a batch the scheduler forms from its resident requests.
                // An engine with none forms no batch, so the step never runs and nothing is
                // compared - and a comparison nobody made is how a wrong kernel reaches a
                // deployment. The update is abandoned with the operators named, rather than
                // waiting out the daemon's whole bound and then passing without the comparison.
                if precopy.since.elapsed() > self.watchdog.stall_bound() * 2 {
                    let reason =
                        shadow_stall_message(&precopy.changed, precopy.since.elapsed().as_secs());
                    return self.abandon_precopy(
                        generation,
                        UpdateFailure::Precopy { reason },
                        precopy.reply,
                    );
                }
                // The shadow pass runs on the scheduler's own thread, one step of the batches it
                // formed, so the serve pass polls it rather than waiting (§2.4).
                match poll_answers(&mut answers) {
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

                self.finish_precopy(precopy)
            }
            PrecopyStage::Done => Ok(()),
        }
    }

    /// A precopy that needs no further stage: its generation is built, its policy instantiated and
    /// its executors rebound or recaptured, so it waits as *prepared* for the switchover.
    ///
    /// Reached from the shadow stage and, directly, from a precopy whose changed set is empty - the
    /// two are the same finished state, and a first generation takes the second because there is no
    /// old kernel to shadow.
    fn finish_precopy(&mut self, mut precopy: Precopy) -> Result<()> {
        let generation = precopy.generation;
        // The generations built here are *prepared*, not serving: `generations` is what each device
        // serves now, and a precopy that finished is not serving until the switchover (§7.3 steps 2
        // and 3). They wait here, so a descriptor lookup still reads the serving generation and a
        // revert has nothing to undo.
        self.prepared = std::mem::take(&mut precopy.next);
        // What serves after the switchover: what serves now, with what this precopy installed.
        self.prepared_programs = self.serving_programs.clone();
        self.prepared_programs.extend(precopy.programs.clone());
        if let Err(error) = self.update.precopied(generation) {
            // The generation this precopy built is not the one the machine waits for - a revert or a
            // later precopy took its place while the builds ran. The revert path discards what was
            // built and answers by name; the process goes on serving what it served.
            tracing::warn!(generation, %error, "a precopy finished for an update that moved on");
            return self
                .revert(
                    generation,
                    UpdateFailure::Precopy {
                        reason: format!(
                            "the precopy finished for a generation that is not in flight: {error}"
                        ),
                    },
                )
                .map(|_| ());
        }
        let mut uncompared = self.uncompared();
        if precopy.changed.is_empty() {
            // "Nothing to compare" and "nobody compared it" are different facts, and only one of
            // them is safe: a precopy that shadowed nothing says which it was.
            uncompared.insert(
                0,
                "no operator's program changed, so no kernel had an older one to compare with"
                    .to_owned(),
            );
        }
        self.answer(
            precopy.reply,
            ProgramStatus::Precopied {
                generation,
                uncompared,
            },
        );
        Ok(())
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
    fn switchover(
        &mut self,
        generation: u64,
        step: Option<u64>,
        reply: Option<u32>,
    ) -> Result<Option<ProgramStatus>> {
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
            return Ok(Some(ProgramStatus::PrecopyFailed {
                generation,
                failure: UpdateFailure::Precopy { reason },
            }));
        }
        // Before the devices cut over and before admission can open on a batch: the generation a
        // step stamps its batches with is the one whose graphs the runner has to find.
        self.record_generation(generation);
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
                return self.revert(generation, failure).map(Some);
            }
        }
        // The record moves before the warmup: probation begins the moment every device serves the
        // new generation, and the warmup is the last part of the cut-over rather than of the
        // generation that served before it. So a warmup that fails is a revert of a generation that
        // *served* - a reverse switchover on every device - and not a discard of one that never
        // cut over, which is what the phase decides, and it cannot be asked before this line.
        if let Err(error) = self.update.switchover(generation, Instant::now()) {
            // The devices cut over and the machine did not: a switchover for a generation the
            // update is not on probation for is a stale message, and the state that holds is the
            // generation that served. The reverse switchover runs, and its answer is what the
            // daemon reads - a protocol message does not end a process that can still serve.
            tracing::warn!(generation, %error, "a switchover names no generation on probation");
            let failure = UpdateFailure::Threshold {
                reason: format!(
                    "the switchover names a generation that is not on probation: {error}"
                ),
            };
            return self.revert(generation, failure).map(Some);
        }
        self.retired = std::mem::replace(&mut self.generations, std::mem::take(&mut self.prepared));
        // The programs move with the generations, as one record: the next precopy's changed set is
        // measured against what serves, and a revert serves the programs the retired pair holds.
        self.retired_programs = std::mem::replace(
            &mut self.serving_programs,
            std::mem::take(&mut self.prepared_programs),
        );

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
                        return Ok(Some(ProgramStatus::Failed {
                            generation,
                            failure: ending,
                        }));
                    }
                    return Ok(Some(ProgramStatus::Switched { generation }));
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
        match switchover_answer(self.warming.len(), generation) {
            SwitchoverAnswer::Now(status) => Ok(Some(status)),
            // `Switched` states that this process serves, and it cannot serve until the buckets a
            // batch may be planned into exist - admission opens in `advance_warmup`. So the answer
            // waits for it: a warmup that fails then answers with the failure, and the daemon never
            // opens a frontend onto a process that cannot take a request.
            SwitchoverAnswer::AfterTheWarmup => {
                self.pending_switch = Some((generation, reply));
                self.warming_since = Some(Instant::now());
                Ok(None)
            }
        }
    }

    /// §7.3 step 5: probation ended in a commit; the previous generation is kept
    /// until its purge.
    fn commit(&mut self, generation: u64) -> Result<ProgramStatus> {
        let status = answer_commit(&mut self.update, generation);
        if matches!(status, ProgramStatus::Committed { .. }) {
            // Probation ended in a commit, so the generation before the update can never serve
            // again. Its graphs and kernels live until the purge, which is the executor's book to
            // keep - and the programs it was built from go with the record of it, since no revert
            // can serve it.
            self.retired.clear();
            self.retired_programs.clear();
            self.commit_policy(generation);
        }
        Ok(status)
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
        // A scheduler parks once its requests have ended, and a request ends only when the daemon
        // acknowledges it through this thread, so the answers are polled from the serve pass:
        // waiting here would hold the acknowledgement the park is waiting for.
        self.draining = Some(Draining {
            epoch,
            answers: self.schedulers.iter().map(|scheduler| scheduler.park(false)).collect(),
            parked: Vec::with_capacity(self.schedulers.len()),
            since: Instant::now(),
        });
        self.advance_drain()
    }

    /// Takes the answers of a drain in progress, and states `Drained` when every scheduler parked.
    fn advance_drain(&mut self) -> Result<()> {
        let bound = self.watchdog.stall_bound();
        let Some(mut draining) = self.draining.take() else {
            return Ok(());
        };
        while let Some(answer) = draining.answers.first() {
            match answer.try_recv() {
                Ok(Ok(cache)) => {
                    draining.parked.push(cache);
                    draining.answers.remove(0);
                }
                Ok(Err(reason)) => bail!("draining the epoch: a scheduler did not park: {reason}"),
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    if draining.since.elapsed() > bound {
                        bail!("draining the epoch: a scheduler did not park within {bound:?}");
                    }
                    self.draining = Some(draining);
                    return Ok(());
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    bail!("draining the epoch: a scheduler is gone");
                }
            }
        }
        self.parked = Some(draining.parked);
        self.serving = false;
        // A drain ends the epoch this process was serving, so admission owed to a warmup of it is
        // owed to nothing: the `Serve` that names the epoch again opens the next one.
        self.admission_owed = false;
        self.send_report(proto::process::Report {
            kind: Some(proto::process::report::Kind::Drained(
                proto::process::Drained {
                    epoch: draining.epoch,
                },
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
        self.prepared_programs.clear();
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
        // §2.5: a process that parked gave its segments back, so nothing may be admitted until they
        // are mapped again - and a wake that fails fails the resume by name rather than serving
        // unmapped pages.
        self.take_back_segments()
            .with_context(|| format!("waking epoch {} before it admits", serve.epoch))?;
        // A `Serve` that arrives while the warmup still has to capture this generation's buckets is
        // owed rather than refused: admission opens when the last submission finishes, the same way
        // it does at a switchover (§4.5).
        self.open_admission()
            .with_context(|| format!("resuming epoch {}", serve.epoch))
    }

    /// Promotes this standby in place (§2.5).
    fn promote_in_place(&mut self) -> Result<()> {
        // §2.5: a standby is promoted to serve the generation the epoch prepared in it.
        // A promotion that finds nothing prepared - the precopy was discarded, or the
        // epoch moved on before it ran - is a promotion of a process that cannot serve,
        // and it is refused here, by name, before admission opens: `Promoted` followed
        // by a first request that finds no graph would tell the daemon the promotion
        // succeeded and move the frontend onto a process with no kernels.
        // A standby that took the epoch's programs serves them already: its precopy, switchover
        // and commit ran when the daemon handed them over, so its generation is the serving one.
        let held = self.prepared.len()
            + self
                .generations
                .iter()
                .filter(|generation| !generation.is_empty())
                .count();
        if let Some(reason) = promote_refusal(self.standby, held) {
            tracing::warn!(%reason, "a promotion this process cannot serve is refused");
            self.send_report(proto::process::Report {
                kind: Some(proto::process::report::Kind::Fault(proto::process::Fault {
                    kind: "promote".to_owned(),
                    reason: reason.clone(),
                    device: 0,
                })),
            });
            bail!("{reason}");
        }
        // §2.5: the promotion runs on the addresses the predecessor's segments held, so
        // they are mapped again before anything is admitted, and a wake that fails fails
        // the promotion by name.
        self.take_back_segments()?;
        self.promoted = true;
        // The generation the schedulers must stamp this process's batches with: the one it
        // holds. A promotion is the one cut-over that reaches no `switchover` of this
        // process's own, so nothing else records it, and a promoted standby whose batches
        // were stamped 0 would find no graph - the same defect the switchover had.
        self.record_generation(self.served_generation());
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
        // The two numbers answer different questions and both are needed: the launch
        // generation is what the epoch layer promotes a child by, and the program
        // generation is the one this process holds - the fact the daemon compares with
        // its own before it moves the frontend, so a promotion of a generation the
        // epoch has reverted is refused there instead of served here (§7.3, §2.5).
        self.send_report(proto::process::Report {
            kind: Some(proto::process::report::Kind::Promoted(
                proto::process::Promoted {
                    generation: self.control.engine(),
                    program_generation: self.served_generation(),
                },
            )),
        });
        // The request link says so too: the daemon's client waits on it to move its frontend.
        self.announce(proto::request::wire::EngineState::Ready)?;
        Ok(())
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

        // The scheduler's own signal, not a reading of its policy slot: `note_policy_serves` fires
        // where a policy actually adopts, so a probe cannot disagree with the fact it reports. A
        // probe that finds every scheduler serving clears the missing state and says nothing - a log
        // that states both that a policy serves and that none was installed is worse than either.
        let serving = self
            .schedulers
            .iter()
            .filter(|scheduler| *scheduler.serving().borrow())
            .count();
        if serving == self.schedulers.len() {
            self.policy_missing_since = None;
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
        let bound = self.watchdog.stall_bound();
        for (scheduler, cache) in self.schedulers.iter().zip(parked) {
            // A resumed epoch recomputes from the tokens the frontend resubmits rather than lending
            // from a predecessor: it is the same world, and the requests it ended at the drain are
            // new requests to it.
            scheduler
                .promote(cache, true, false)
                .context("resuming a scheduler")?;
            // The promotion states the cache; it is the resume that ends the park. A scheduler left
            // parked takes requests into its queue and admits none of them.
            match scheduler.resume(true).recv_timeout(bound) {
                Ok(Ok(())) => {}
                Ok(Err(reason)) => bail!("resuming a scheduler: {reason}"),
                Err(error) => bail!("resuming a scheduler: no answer within {bound:?}: {error}"),
            }
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
        // The batches that form from here serve the generation that served before the update, so the
        // number they are stamped with is that one: a revert that left the update's number in place
        // would stamp every batch with a generation whose graphs are about to be released.
        self.record_generation(self.update.current());
        self.revert_policy(generation);
        // The generation that served before the update serves again, and the one built for the
        // update keeps nothing: its kernels are released and its captures are never taken over.
        let kept = !self.retired.is_empty();
        if kept {
            self.generations = std::mem::take(&mut self.retired);
            self.serving_programs = std::mem::take(&mut self.retired_programs);
        }
        self.prepared.clear();
        self.prepared_programs.clear();
        // The warmups still queued warm the generation this revert went back from: what runs when
        // they reach the device is the generation that serves, so nothing hangs - the list is what
        // stopped describing serving, and a failure read from a stale entry would revert an update
        // that is over.
        self.warming.clear();
        self.warming_since = None;
        // The answer owed to that warmup goes with it: the daemon's switchover wait is answered by
        // the revert's own report, and a later `Switched` would state a generation that does not
        // serve.
        self.pending_switch = None;
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
            // A purge for a generation this process holds nothing of - one it never built, or one a
            // revert already released - asks for a release that has happened, and the honest answer
            // is that this process holds nothing of it. The warning is what tells a race from a
            // store-side mistake; neither is a reason to end a serving process.
            tracing::warn!(generation, %error, "a purge names a generation no update holds");
            self.released.insert(generation);
            return Ok(ProgramStatus::Purged { generation });
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
        if let Err(error) = self.update.purged(generation) {
            // The release happened on every device; only the bookkeeping did not recognise the
            // generation. Recording it is the whole of the answer, and the warning keeps the race
            // in the log rather than in the exit code.
            tracing::warn!(generation, %error, "a purged generation no update held");
        }
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
    fn request_command(
        &mut self,
        front: usize,
        command: proto::request::wire::Command,
    ) -> Result<()> {
        // Two of a request's commands are the connection's bookkeeping rather than a scheduler's,
        // and this process is the connection: it writes every event to the front the request
        // belongs to and keeps what a takeover has to replay. Handling them here keeps the
        // scheduler's refusal, which is right for a scheduler, from ending the epoch on the first
        // long request: the daemon reports a consumer's position every `EVENT_WINDOW / 2` events.
        match command.kind.as_ref() {
            Some(proto::request::wire::command::Kind::Consumed(consumed)) => {
                let Some(served) = self
                    .streams
                    .iter_mut()
                    .find(|served| served.request == command.request)
                else {
                    tracing::debug!(
                        request = command.request,
                        "a consumer's position names a request this epoch no longer serves"
                    );
                    return Ok(());
                };
                // Everything the consumer has read can no longer be replayed, which is what bounds
                // the journal.
                served
                    .kept
                    .retain(|(position, _, _)| *position > consumed.events);
                return Ok(());
            }
            Some(proto::request::wire::command::Kind::Takeover(takeover)) => {
                let Some(index) = self
                    .streams
                    .iter()
                    .position(|served| served.request == command.request)
                else {
                    // A handover's takeover names a request this process kept running for it:
                    // one it does not hold is either a request that ended first or a front that
                    // named the wrong identity, and only the log can say which.
                    tracing::warn!(
                        request = command.request,
                        front,
                        from = takeover.events,
                        "a takeover names a request this epoch no longer serves"
                    );
                    return Ok(());
                };
                // The request is this daemon's from the position its consumer states: what the
                // predecessor produced past it is replayed, because its own consumer stopped
                // reading at the pause.
                //
                // The journal and the position are read out before the process is borrowed again:
                // a refusal writes to a front, and the stream is a field of the same process.
                let kept = self.streams[index].kept.clone();
                let sent = self.streams[index].sent;
                // A position older than the journal cannot be replayed: the events between it and
                // the oldest one kept were dropped, and handing the successor what is left is a
                // stream that skips tokens with nothing saying so. It is refused, naming the two
                // positions and how far behind the consumer was, so the daemon ends the response
                // instead of serving a body with a hole in it.
                if let (Some((oldest, _, _)), true) = (kept.front(), sent > takeover.events) {
                    if takeover.events + 1 < *oldest {
                        let oldest = *oldest;
                        let behind = sent - takeover.events;
                        tracing::error!(
                            request = command.request,
                            front,
                            from = takeover.events,
                            oldest,
                            sent,
                            behind,
                            "a takeover names a position this process no longer holds: the \
                             consumer was more events behind than the journal kept"
                        );
                        self.reject(
                            front,
                            command.request,
                            proto::request::RejectionKind::Overloaded,
                            format!(
                                "the request could not be taken over: its consumer's position \
                                 {} is {} event(s) behind, and this process kept the last {} of \
                                 them from position {} - the handover produced more events of \
                                 this request than a pause may",
                                takeover.events,
                                behind,
                                kept.len(),
                                oldest,
                            ),
                        );
                        return Ok(());
                    }
                }
                let replay = replay_after(&kept, takeover.events);
                self.streams[index].front = front;
                for (event, media) in replay {
                    tracing::debug!(
                        request = command.request,
                        front,
                        "replaying an event a taken-over request produced during the pause"
                    );
                    self.send_stream_event(front, event, media);
                }
                tracing::info!(
                    request = command.request,
                    front,
                    from = takeover.events,
                    "a request was taken over by another daemon front"
                );
                return Ok(());
            }
            _ => {}
        }
        // A command names its request, not the scheduler serving it, so the rank is read from the
        // stream the submission answered with. A request that is not in flight here has already
        // ended: a cancel or a stop for it is a no-op, not a fault, and ending an epoch because a
        // client disconnected a moment too late would be the daemon's most common message.
        let Some(rank) = self
            .streams
            .iter()
            .find(|served| served.request == command.request)
            .map(|served| served.rank)
        else {
            tracing::debug!(
                request = command.request,
                "a request command names a request this epoch no longer serves"
            );
            return Ok(());
        };
        let Some(scheduler) = self.schedulers.get(rank) else {
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
        front: usize,
        command: proto::request::wire::EngineCommand,
    ) -> Result<Option<UpdateFailure>> {
        // §2.3: the epoch's lifecycle - Park, Promote, Serve, Drain, Pause, the program layer's
        // stages - arrives on the process connection and is applied by the driver. On this
        // connection the epoch owns exactly one command, `Shutdown`: every request finishes as
        // aborted and the owners close in the order they need (`close`).
        //
        // The rest are the engine surface of the executor era, and the daemon's migration, expert
        // and admin paths still send them here. Each is answered with what it needs instead of
        // being dropped: a command a process does not serve is otherwise a caller waiting for a
        // reply that never comes, which is a hang with no name in it.
        let refused = match command.kind {
            Some(proto::request::wire::engine_command::Kind::Shutdown(_)) => {
                self.stopping = true;
                return Ok(None);
            }
            // §2.5: a lost process's standby is promoted by the daemon that held the two, on this
            // connection, because no epoch switch is running to command it on the process one. The
            // daemon moves the frontend when this process states that it serves.
            Some(proto::request::wire::engine_command::Kind::Promote(_)) => {
                self.promote_in_place()?;
                return Ok(None);
            }
            // §7.5.1 step 2: a hot restart's successor gives this engine a second daemon front, so
            // the responses in flight can move to it while the process goes on serving. A refusal
            // is a failed handover and not this process's end: the daemon that asked is still
            // serving the requests it took.
            Some(proto::request::wire::engine_command::Kind::Attach(attach)) => {
                let state = self.reported_state;
                let info = self.announced_info();
                match self.control.attach(&attach.requests, state, info) {
                    Ok(front) => tracing::info!(
                        daemon = %attach.requests,
                        front,
                        "a daemon attached to this engine"
                    ),
                    Err(error) => tracing::warn!(
                        daemon = %attach.requests,
                        %error,
                        "attaching a daemon failed; this engine keeps the fronts it has"
                    ),
                }
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
            // The executor era's park, resume and trial commit move an executor world between
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
            // What the prefix cache holds is answered here, from what this process announced.
            //
            // A standby that starts on these devices maps the serving epoch's segments read-only
            // (`specs/executor.md` 7.4 step 1), so reading them moves nothing and is not a lifecycle
            // command. The answer is the description this process gave the daemon when it became
            // ready: the segments are the pool's, and they are the same for as long as the epoch
            // serves. Asking every scheduler afresh is what cannot be done from here - a scheduler
            // holding paused streams does not answer until its stall bound, and a control thread
            // waiting on one cannot serve the handover's own commands (`501fa5e6f`).
            Some(proto::request::wire::engine_command::Kind::DescribeCache(describe)) => {
                let (error, segments) = match &self.announced {
                    Some(info) => (String::new(), info.segments.clone()),
                    None => (
                        "this process has announced no cache segments: it has not become ready"
                            .to_owned(),
                        Vec::new(),
                    ),
                };
                Some(proto::request::wire::from_engine::Kind::CacheDescribed(
                    proto::request::wire::CacheDescribed {
                        id: describe.id,
                        error,
                        segments,
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
            // FlushCache and Attach state no id and are answered with a state this process does
            // not own: there is nothing to answer on, and a log line is what is left.
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
            self.send_event(front, kind);
        }
        Ok(None)
    }

    /// Sends the engine's counters to the daemon, one report a second and a
    /// final flush when the process ends (`specs/executor.md` §2.3).
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
        // The scheduler statistics travel beside the counters: a daemon records
        // `uniserve:num_requests_running` and its kind from this snapshot, so a process that
        // sent only the allocator's figures would report an idle deployment forever.
        let mut snapshots = Vec::with_capacity(self.schedulers.len());
        for (index, (scheduler, reporter)) in self
            .schedulers
            .iter()
            .zip(&mut self.stats_reporters)
            .enumerate()
        {
            let snapshot = reporter.snapshot(scheduler.stats());
            if let Ok(snapshot) = snapshot.encode() {
                snapshots.push(proto::request::wire::Stats {
                    scheduler: index as u32,
                    snapshot,
                });
            }
        }
        // Every live front is told: the daemon that launched this process, and each daemon
        // that attached to it (§7.5.1 step 2). A handover closes the launching daemon's front
        // and leaves the successor's serving, so a report aimed at index 0 alone would stop
        // exactly when the successor owns the engine - and the successor's `/metrics` would
        // hold no figure of what it inherited. Each daemon keeps its own registry, so a report
        // on each front is one figure per registry.
        let fronts = self.control.report_fronts();
        let mut delivered = 0;
        let mut failure = None;
        for front in fronts {
            let Some(connection) = self.control.front_mut(front) else {
                continue;
            };
            let sent = connection
                .send(&proto::request::wire::FromEngine {
                    kind: Some(proto::request::wire::from_engine::Kind::Metrics(
                        metrics.clone(),
                    )),
                })
                .and_then(|()| {
                    for stats in &snapshots {
                        connection.send(&proto::request::wire::FromEngine {
                            kind: Some(proto::request::wire::from_engine::Kind::Stats(
                                stats.clone(),
                            )),
                        })?;
                    }
                    Ok(())
                });
            match sent {
                Ok(()) => delivered += 1,
                Err(error) => failure = Some(error),
            }
        }
        if delivered == 0 {
            if let Some(error) = failure {
                tracing::warn!(
                    %error,
                    "reporting the engine's metrics failed: no front of this process is live"
                );
            }
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
    fn send_event(&mut self, front: usize, kind: proto::request::wire::from_engine::Kind) {
        let Some(connection) = self.control.front_mut(front) else {
            // The front this request belongs to detached. Its new owner names itself with a
            // `Takeover`, and until then nobody is reading: the request's events are kept for the
            // takeover to replay.
            tracing::debug!(
                front,
                "a request's front is gone; its events are kept for a takeover"
            );
            return;
        };
        if let Err(error) = connection.send(&proto::request::wire::FromEngine { kind: Some(kind) })
        {
            tracing::warn!(%error, front, "sending an event of an admitted request failed");
        }
    }

    /// Hands one event of a request's stream to its front, with the artifact bytes it travels with.
    ///
    /// A front on this host maps the shared-memory object, whose descriptor rides the event's own
    /// frame. A front on another host, or bytes that no descriptor names, takes bounded parts ahead
    /// of the event, and the daemon writes each part into the artifact's final host storage.
    fn send_stream_event(
        &mut self,
        front: usize,
        event: proto::request::wire::Event,
        media: Option<std::sync::Arc<dyn proto::request::ArtifactMedia>>,
    ) {
        use proto::request::wire::{ArtifactPart, FromEngine, from_engine::Kind};

        let Some(media) = media else {
            self.send_event(front, Kind::Event(event));
            return;
        };
        let Some(connection) = self.control.front_mut(front) else {
            tracing::debug!(
                front,
                "a request's front is gone; its artifact is kept for a takeover"
            );
            return;
        };
        let frame = FromEngine {
            kind: Some(Kind::Event(event)),
        };
        let descriptor = media
            .descriptor()
            .filter(|_| connection.peer_ip().is_none());
        let sent = match descriptor {
            Some(descriptor) => connection.send_with_descriptor(&frame, descriptor),
            None => media
                .bytes()
                .chunks(ARTIFACT_PART)
                .try_for_each(|data| {
                    connection.send(&FromEngine {
                        kind: Some(Kind::ArtifactPart(ArtifactPart {
                            size: media.bytes().len() as u64,
                            data: data.to_vec(),
                        })),
                    })
                })
                .and_then(|()| connection.send(&frame)),
        };
        if let Err(error) = sent {
            tracing::warn!(%error, front, "sending a request's artifact failed");
        }
    }

    /// Ends one request with its refusal: a submission this process will not admit is answered with
    /// the event that says so, because the daemon has no other way to learn it (§2.3).
    fn reject(
        &mut self,
        front: usize,
        request: u64,
        kind: proto::request::RejectionKind,
        message: String,
    ) {
        let event = proto::request::wire::Event {
            request,
            kind: Some(proto::request::wire::event::Kind::Rejected(
                proto::request::wire::Rejected {
                    kind: kind as i32,
                    message,
                },
            )),
        };
        self.send_event(front, proto::request::wire::from_engine::Kind::Event(event));
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
        // Whether a handover is open, read once for the pass: a second front that has attached and
        // not yet taken its requests over. The journal's bound is wider while it is, because a
        // predecessor's consumer stops reading at the pause while this process keeps producing.
        let mut running = Vec::with_capacity(streams.len());
        for mut served in streams {
            let id = served.request;
            loop {
                match served.events.try_recv() {
                    Ok(event) => {
                        let terminal = matches!(
                            event,
                            engine::Event::Finished { .. }
                                | engine::Event::Rejected { .. }
                                | engine::Event::Error { .. }
                                | engine::Event::Migratable { .. }
                        );
                        let (wire, media) = event.to_wire(proto::request::RequestId(id));
                        // The position this event is written at, and the copy a takeover from an
                        // earlier position replays.
                        served.sent += 1;
                        keep_event(
                            &mut served.kept,
                            served.sent,
                            wire.clone(),
                            media.clone(),
                            HANDOVER_JOURNAL,
                        );
                        self.send_stream_event(served.front, wire, media);
                        if terminal {
                            break;
                        }
                    }
                    // A request that produced nothing yet keeps its stream.
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                        running.push(served);
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
        self.reported_state = state as i32;
        let info = match state {
            proto::request::wire::EngineState::Ready
            | proto::request::wire::EngineState::Standby => Some(self.engine_info()?),
            _ => None,
        };
        self.announced = info.clone();
        tracing::info!(
            state = ?state,
            schedulers = self.schedulers.len(),
            devices = self.executors.len(),
            "stating what this process is to the daemon"
        );
        if matches!(
            state,
            proto::request::wire::EngineState::Ready | proto::request::wire::EngineState::Standby
        ) {
            self.announced_at.get_or_insert_with(Instant::now);
        }
        announce(&mut self.control, state, info)
    }

    /// What an attaching daemon is told this process is: the info it announced, for a state that
    /// carries one, and nothing otherwise.
    fn announced_info(&mut self) -> Option<proto::request::wire::EngineInfo> {
        // The value announced, not a fresh one: describing the cache asks every scheduler for
        // what it holds, and a scheduler that is busy does not answer for as long as the stall
        // bound - which is a control thread that cannot serve the handover's own attach
        // (`501fa5e6f`: the engine read the attach and then said nothing for sixty seconds).
        self.announced.clone()
    }

    /// What the daemon records of this process: the limits every request is checked against, the
    /// batch the executors were started for, and the KV cache segments a successor on the same
    /// devices maps (§7.3 step 1).
    ///
    /// The figures are the executors' own (`SchedulerHandle::info`), except the cache segments, which
    /// each scheduler describes for itself: a successor needs what the executors hold now, not what
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
                    tracing::warn!(%error, "an executor group did not describe its cache");
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
            latent_encoding_units: first
                .media_units(ipc::MediaCall::LatentEncoding)
                .unwrap_or(0),
            schedulers: u32::try_from(self.schedulers.len()).unwrap_or(u32::MAX),
            capacity: Some(proto::request::wire::BatchCapacity {
                max_batch: info.max_batch_calls,
                max_num_batched_tokens: u64::from(info.max_batch_tokens),
                // The launch's own figure, not the executor's row count: the rows include the
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
        for served in std::mem::take(&mut self.streams) {
            let event = proto::request::wire::Event {
                request: served.request,
                kind: Some(proto::request::wire::event::Kind::Error(
                    proto::request::wire::Failed {
                        message: "the epoch was shut down with this request in flight".to_owned(),
                    },
                )),
            };
            self.send_event(
                served.front,
                proto::request::wire::from_engine::Kind::Event(event),
            );
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
        for memory in std::mem::take(&mut self.memory) {
            if let Err(error) = memory.close() {
                tracing::warn!(%error, "closing the resident owners failed");
            }
        }
        self.control.close();
    }
}

impl MainProcess {
    /// §7.3 step 2's policy half: every scheduler prepares the update's policy program on its own
    /// state, from the module the Program Server resolves under the store's wasm key, and answers on
    /// a channel the serve pass polls.
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
        let source = Arc::new(ProgramSource::connect(&self.program_server));
        let source: Arc<dyn PolicySource + Send + Sync> = source;
        // The start marker of §7.3 step 2's policy half: the resolve that follows runs on a
        // scheduler's blocking thread and prints its own target and spans, and without this line a
        // log cannot tell a precopy that began from one whose command was never sent.
        tracing::info!(
            backend = %program.backend,
            revision = program.revision,
            schedulers = self.schedulers.len(),
            socket = %self.program_server.display(),
            "this precopy's policy preparation has begun"
        );
        Ok(self
            .schedulers
            .iter()
            .map(|scheduler| scheduler.precopy_policy(program.clone(), Arc::clone(&source)))
            .collect())
    }

    /// §7.3 step 3's policy half: every scheduler's next batch boundary answers from the prepared
    /// policy. `step` is the boundary the epoch's schedulers agreed on, which only the scheduler
    /// node can name; the control thread does not choose one.
    /// Records on every scheduler the generation whose kernels the next step's batches serve
    /// (§4.2, §7.3).
    ///
    /// A step reads this cell once and stamps every batch it forms with it, and a runner serves a
    /// batch from the graphs captured for that number. A control thread that cut the devices over
    /// without recording it left the first batch of the deployment stamped `0`, and the runner
    /// answered `no graph of bucket ... in generation 0 or in a generation it replaces`: the graphs
    /// were captured under the generation the daemon named, and no batch ever asked for it.
    fn record_generation(&self, generation: u64) {
        for scheduler in &self.schedulers {
            scheduler.set_generation(generation);
        }
        // One line, because the figure a batch carries is invisible until a runner refuses it: a run
        // whose first batch names a generation this line never stated is a run where something cut a
        // generation over without recording it.
        tracing::info!(
            generation,
            schedulers = self.schedulers.len(),
            "the generation this process's batches carry is recorded"
        );
    }

    fn switchover_policy(&self, generation: u64, _step: Option<u64>) {
        // This states the candidate, not the serving policy: a scheduler adopts at its next batch
        // boundary, which is after this returns, so a "serving" figure here would be a second
        // statement of a fact `serving()` reports - and the two disagreed in a run. What this can
        // say is whether anything was prepared to adopt.
        let prepared = self
            .schedulers
            .iter()
            .filter(|scheduler| scheduler.policy_mut().prepared().is_some())
            .count();
        if prepared == self.schedulers.len() {
            tracing::info!(
                generation,
                prepared,
                schedulers = self.schedulers.len(),
                "every scheduler holds a prepared policy candidate"
            );
        } else {
            tracing::warn!(
                generation,
                prepared,
                schedulers = self.schedulers.len(),
                "a scheduler holds no prepared policy candidate: a switchover adopts nothing there"
            );
        }
        for scheduler in &self.schedulers {
            report_policy_answer(
                scheduler.switchover_policy(),
                self.watchdog.stall_bound(),
                "the policy switchover",
            );
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
            report_policy_answer(
                scheduler.revert_policy(),
                self.watchdog.stall_bound(),
                "the policy revert",
            );
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
/// The name of a program-layer status, for the line that says which step was answered.
fn status_name(status: &ProgramStatus) -> &'static str {
    match status {
        ProgramStatus::Precopied { .. } => "precopied",
        ProgramStatus::PrecopyFailed { .. } => "precopy_failed",
        ProgramStatus::Switched { .. } => "switched",
        ProgramStatus::Committed { .. } => "committed",
        ProgramStatus::Reverted { .. } => "reverted",
        ProgramStatus::Purged { .. } => "purged",
        ProgramStatus::PolicyReverted { .. } => "policy_reverted",
        ProgramStatus::Failed { .. } => "failed",
    }
}

fn report_of(status: &ProgramStatus) -> Option<proto::process::Report> {
    use proto::process::report::Kind;

    let kind = match status {
        ProgramStatus::Precopied {
            generation,
            uncompared,
        } => Kind::Precopied(proto::process::Precopied {
            generation: *generation,
            uncompared: uncompared.clone(),
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
    answers: &mut Vec<crossbeam_channel::Receiver<std::result::Result<(), String>>>,
) -> Option<StageFailure> {
    // An answer is taken once and its sender is gone after it, so a receiver that answered leaves
    // the list: polling it again would read a scheduler that finished as one that was lost.
    while let Some(receiver) = answers.first() {
        match receiver.try_recv() {
            Ok(Ok(())) => {
                answers.remove(0);
            }
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

/// A drain whose schedulers have not all parked yet.
struct Draining {
    /// The epoch `Drained` names.
    epoch: u64,
    /// The schedulers still to answer, in order.
    answers: Vec<crossbeam_channel::Receiver<std::result::Result<engine::ParkedCache, String>>>,
    /// What the ones that answered parked with, in the schedulers' order.
    parked: Vec<engine::ParkedCache>,
    /// When the drain began, which the stall bound is measured from.
    since: Instant,
}

/// Logs one scheduler's answer to a policy step when it comes, off the control thread.
///
/// A scheduler answers at its next batch boundary, and it reaches that boundary only while the
/// control thread drains what it sends: a control thread that waited here held the scheduler it
/// waited for, and a switchover under traffic stood still for the whole stall bound (§2.4).
fn report_policy_answer(
    receiver: crossbeam_channel::Receiver<std::result::Result<(), String>>,
    bound: Duration,
    what: &'static str,
) {
    std::thread::spawn(move || {
        if let Err(error) = answer(receiver, bound, what) {
            tracing::error!(%error, "a scheduling policy step failed");
        }
    });
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

/// What the engine answers a command of the executor era's surface: the operation named an executor
/// world of its own, and this epoch executes batches itself (§2.4).
const WORKER_ERA: &str = "this epoch has no worker of that kind: the command names a worker-era engine, and the epoch's \
     executors are the process's own";

/// The events a takeover from `from` replays: everything this process produced past the position
/// the successor's consumer states.
///
/// The predecessor freezes its response at the pause, so its consumer stops advancing while this
/// process goes on producing, and the successor resumes from what it last rendered. Without the
/// replay those events are rendered nowhere and the response loses the tokens the pause covered.
fn replay_after(
    kept: &std::collections::VecDeque<KeptEvent>,
    from: u64,
) -> Vec<(
    proto::request::wire::Event,
    Option<std::sync::Arc<dyn proto::request::ArtifactMedia>>,
)> {
    kept.iter()
        .filter(|(position, _, _)| *position > from)
        .map(|(_, event, media)| (event.clone(), media.clone()))
        .collect()
}

/// Journals one event at `position`, dropping the oldest past `bound`.
fn keep_event(
    kept: &mut std::collections::VecDeque<KeptEvent>,
    position: u64,
    event: proto::request::wire::Event,
    media: Option<std::sync::Arc<dyn proto::request::ArtifactMedia>>,
    bound: usize,
) {
    kept.push_back((position, event, media));
    while kept.len() > bound {
        kept.pop_front();
    }
}

/// What the engine answers a command whose owner is the epoch layer: the same operation exists,
/// and it is asked for through the process connection, where the driver applies it (§2.5, §7.4).
const EPOCH_LIFECYCLE: &str = "this command is the epoch layer's, applied through the process connection rather than the \
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
const LEDGERS: &str = "the storage ledgers and transfer budgets are read by each node's daemon over its claim, not \
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
    Retry(Vec<Arc<HostTask<ModuleWork>>>),
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

/// How long a serve pass sleeps while a precopy, a warmup or a drain is in flight.
const TRANSITION_POLL: Duration = Duration::from_millis(1);

/// How often a device that is busy delays the question asked of the policy slots: a probe that finds
/// a device running proves nothing about whether a policy serves, so it is asked again after this.
const POLICY_PROBE_INTERVAL: Duration = Duration::from_millis(250);

/// The window a daemon gives one program-layer command before it gives up on it, as this process
/// must assume it (`crates/daemon/src/engine/control.rs`'s `COMMAND_TIMEOUT`): the engine does not
/// know that constant, it knows that its own reports must arrive inside the window or a run says
/// nothing but a timeout.
const DAEMON_COMMAND_WINDOW: Duration = Duration::from_secs(120);

/// How long a metrics pass waits on one scheduler's counters: the report is sent once a second and a
/// pass that waited longer than this would be reporting what the next pass reports anyway.
const METRICS_WAIT: Duration = Duration::from_millis(50);

/// How long the daemon's process interface may take to answer the handshake, which it answers at
/// once: the wait exists to fail a wrong address rather than hang the process.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

/// How long the daemon may take to answer this rank's storage claim (§8): starting a Storage
/// Server for the node is the slow part, and it is the daemon's to do.
const CLAIM_TIMEOUT: Duration = Duration::from_secs(60);

/// The store's listing, on the module-load lane (§2.4, §7.3 step 2).
///
/// This is the one call that leaves this process before a generation is prepared, and the one
/// with no deadline of its own: the client's connect and its reply read are unbounded, so a store
/// that never answers holds whoever asked. On the lane that cost is a build slot. On the control
/// thread it would be the thread itself - the one that answers the daemon, the requests, the
/// metrics and the watchdog, whose silence reads exactly like a stalled device.
struct ListStore {
    programs: Arc<Mutex<Client>>,
}

impl HostAction for ListStore {
    type Output = program_server::Listing;

    fn run(&self) -> executor::Result<Self::Output> {
        tracing::info!("asking the Program Server for its installed programs");
        let mut client = self
            .programs
            .lock()
            .map_err(|_| executor::Error::State("the Program Server client was poisoned".into()))?;
        let listing = client.list().map_err(|error| {
            executor::Error::Invalid(format!("reading the Program Server's store: {error}"))
        })?;
        tracing::info!(
            packages = listing.packages.len(),
            "the Program Server's listing answered"
        );
        Ok(listing)
    }

    /// Nothing to give back: the client this action borrows belongs to the process, and its reply
    /// holds no device or pinned memory.
    fn release(&self) -> executor::Result<()> {
        Ok(())
    }

    fn input_outcome(&self) -> Outcome {
        Outcome::Success(())
    }
}

/// One piece of the module-load lane's work (§2.4, §7.3 step 2).
///
/// The lane has two tenants and one order: the store's listing, which says what this host has
/// built, and the installs that follow from it. Keeping them in one lane keeps that order where
/// it belongs - nothing is installed before the store has said what it holds - and keeps both off
/// the control thread.
enum ModuleWork {
    List(ListStore),
    Install(Install),
}

impl HostAction for ModuleWork {
    type Output = ModuleAnswer;

    fn run(&self) -> executor::Result<Self::Output> {
        match self {
            Self::List(list) => list.run().map(ModuleAnswer::Listed),
            Self::Install(install) => install.run().map(ModuleAnswer::Installed),
        }
    }

    fn release(&self) -> executor::Result<()> {
        match self {
            Self::List(list) => list.release(),
            Self::Install(install) => install.release(),
        }
    }

    fn input_outcome(&self) -> Outcome {
        match self {
            Self::List(list) => list.input_outcome(),
            Self::Install(install) => install.input_outcome(),
        }
    }
}

/// What one piece of the module lane's work answered.
enum ModuleAnswer {
    /// The store's catalog: every package this host has built, with its modules.
    Listed(program_server::Listing),
    /// One device's kernels for the generation being prepared.
    Installed(Vec<KernelEntry>),
}

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
    /// One request per package the generation's programs come from: the Program Server names one
    /// program per operator in a request, and an operator two packages split is asked of each in
    /// its own request, with only the keys that package built.
    requests: Vec<PrepareRequest>,
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

/// One module the Program Server resolved, held open while its kernel loads (`specs/executor.md`
/// §4.2, §6.3).
///
/// The server sends the module's files as open handles rather than as paths, so the image is read
/// through the handle: a cubin's bytes are small and read once, and a native library is loaded
/// through the handle's `/proc/self/fd` path, which names the same inode the server sent. Every
/// handle stays open until the kernel has loaded, so neither read reaches a file that has gone.
struct Resolved {
    image: Option<Vec<u8>>,
    metadata: Option<Vec<u8>>,
    native: Option<NativeModule>,
    entry: String,
}

/// The path `dlopen` loads a native module from (§4.2).
///
/// `dlopen` takes a path, and the handle the Program Server sends has none of its own. The file is
/// used where it still has a name - glibc keys a loaded object by the name it was opened with, so a
/// name this process derives from a descriptor would answer for the next module to land on that
/// number, and one library's handle would serve another. When the store's file has no name left, the
/// bytes are copied to one this process owns and that copy goes when the module does: `dlopen` maps
/// the file, so the mapping outlives the name.
struct NativeModule {
    path: std::path::PathBuf,
    /// The copy, when the handle the server sent named no file of its own.
    copy: Option<std::path::PathBuf>,
}

impl NativeModule {
    /// Whether this module was loaded from a copy rather than from the store's own file.
    fn copied(&self) -> bool {
        self.copy.is_some()
    }
}

impl Drop for NativeModule {
    fn drop(&mut self) {
        let Some(copy) = &self.copy else {
            return;
        };
        if let Err(error) = std::fs::remove_file(copy) {
            tracing::warn!(
                path = %copy.display(),
                %error,
                "removing this process's copy of a native module failed"
            );
        }
    }
}

/// The file a native module's library is, and the path it is loaded from.
///
/// **Never by position.** A `paged_attention` module holds `module.so` beside `cubins/`, and `cubins`
/// sorts before `module.so`, so taking the first file handed `dlopen` a cubin and the answer was
/// `cannot open shared object file`. The library is the file that defines the entry the package
/// declared, which is the property `dlopen` is for; a module whose files define it in more than one
/// place, or in none, is refused by name with its files listed.
///
/// A module with siblings is never loaded from a copy, and never from a descriptor number. Its entry
/// receives no path - `ops::Entry` takes the call structure and the stream - so the object finds its
/// own `cubins/` from the file it was loaded from: `paged_attention`'s module does it with `dladdr`
/// of its own entry and takes the directory of `dli_fname`
/// (`crates/ops/kernels/paged_attention/trtllm/src/entry.cpp`, `module_directory`). `dli_fname` is the
/// path handed to `dlopen`, so that path must be the module's own file: `/proc/self/fd/<n>` gives the
/// directory `/proc/self/fd`, and a copy gives the copy's directory. Both are refusals here instead
/// of a load that dies at the first cubin.
fn native_library(
    files: Vec<program_server::ModuleFile>,
    entry: &str,
) -> executor::Result<NativeModule> {
    let invalid = |reason: String| executor::Error::Invalid(reason);
    let siblings = files.len() > 1;
    let chosen = match files.len() {
        0 => return Err(invalid("a native module carries no library".to_owned())),
        // One file: the module's library is that file, and there is no order to get wrong.
        1 => 0,
        _ => {
            let defining: Vec<usize> = files
                .iter()
                .enumerate()
                .filter(|(_, file)| defines_entry(file, entry))
                .map(|(index, _)| index)
                .collect();
            let names = |chosen: &[usize]| {
                chosen
                    .iter()
                    .filter_map(|index| files.get(*index))
                    .map(|file| file.name.clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            match defining.as_slice() {
                [only] => *only,
                [] => {
                    return Err(invalid(format!(
                        "no file of this native module defines the entry the package declared \
                         ({entry}), so nothing in it is that module's library: {}",
                        names(&(0..files.len()).collect::<Vec<_>>())
                    )));
                }
                several => {
                    return Err(invalid(format!(
                        "several files of this native module define the entry the package declared \
                         ({entry}), so its library is ambiguous: {}",
                        names(several)
                    )));
                }
            }
        }
    };
    let file = files
        .into_iter()
        .nth(chosen)
        .ok_or_else(|| invalid("the chosen library file is missing".to_owned()))?;
    if siblings {
        // The question a reader of a failed load has: where does this object look for the files
        // beside it? A module that resolves them itself uses the path it was loaded from - `dladdr`
        // of its own entry - so that directory is the answer, and `origin` says whether it also
        // states `$ORIGIN` in a run path, which is the other way a sibling can be found.
        let module = match std::fs::read_link(format!("/proc/self/fd/{}", as_raw_fd(&file))) {
            Ok(target) => target,
            Err(_) => std::path::PathBuf::new(),
        };
        tracing::info!(
            library = %file.name,
            loaded_from = %module.display(),
            directory = %module.parent().map(|path| path.display().to_string()).unwrap_or_default(),
            origin = ?origin_in_runpath(&file),
            "this native module's library sits beside other files of the module"
        );
    }
    native_module(&file, !siblings)
}

/// Whether a file defines `entry` among the functions it exports.
fn defines_entry(file: &program_server::ModuleFile, entry: &str) -> bool {
    bytes_of(file)
        .and_then(|bytes| defined_names(&bytes))
        .is_some_and(|names| names.iter().any(|name| name == entry))
}

/// The file's bytes, read from the start: the handle the server sent may have been read already.
fn bytes_of(file: &program_server::ModuleFile) -> Option<Vec<u8>> {
    use std::io::{Read as _, Seek as _};
    let mut handle = &file.file;
    handle.seek(std::io::SeekFrom::Start(0)).ok()?;
    let mut bytes = Vec::with_capacity(file.size as usize);
    handle.read_to_end(&mut bytes).ok()?;
    Some(bytes)
}

/// Whether the object states `$ORIGIN` in its run path: the one way it can find a sibling file by
/// its own location. `None` when the object states no run path, or cannot be read.
fn origin_in_runpath(file: &program_server::ModuleFile) -> Option<bool> {
    let bytes = bytes_of(file)?;
    let path = dynamic_string(&bytes, &[29, 15])?;
    Some(path.contains("$ORIGIN"))
}

/// The descriptor behind a module's file, for the line that says where the object will look for its
/// siblings.
fn as_raw_fd(file: &program_server::ModuleFile) -> std::os::fd::RawFd {
    use std::os::fd::AsRawFd as _;
    file.file.as_raw_fd()
}

/// The path a native module is loaded from: the store's own file when it still has one, and a copy
/// this process owns when the handle the server sent has outlived its name.
fn native_module(
    file: &program_server::ModuleFile,
    copy_allowed: bool,
) -> executor::Result<NativeModule> {
    use std::io::Read as _;
    use std::os::fd::AsRawFd as _;

    let invalid = |reason: String| executor::Error::Invalid(reason);
    let link = format!("/proc/self/fd/{}", file.file.as_raw_fd());
    if let Ok(target) = std::fs::read_link(&link) {
        let named = !target.to_string_lossy().ends_with(" (deleted)");
        if named && target.is_file() {
            return Ok(NativeModule {
                path: target,
                copy: None,
            });
        }
    }

    // The handle names nothing this process can open: the server unlinked the file after sending
    // it, or the descriptor addresses something that is not a file. One copy, under a name no
    // other module shares - and only for a module that stands alone, because a copy is a different
    // location and every path the module resolves from its own location would break.
    if !copy_allowed {
        return Err(invalid(format!(
            "the library of this native module ({}) has no name this process can open, and the \
             module holds files beside it: loading it from a copy would lose them, because the \
             entry receives no path and the object finds them through its own location",
            file.name
        )));
    }
    static COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let ordinal = COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!("uniserve-native-{}", std::process::id()));
    std::fs::create_dir_all(&directory)
        .map_err(|error| invalid(format!("preparing {}: {error}", directory.display())))?;
    let name: String = file
        .name
        .chars()
        .map(|held| {
            if held.is_ascii_alphanumeric() || held == '.' || held == '_' || held == '-' {
                held
            } else {
                '_'
            }
        })
        .collect();
    let path = directory.join(format!("{ordinal}-{name}"));
    let mut bytes = Vec::with_capacity(file.size as usize);
    (&file.file)
        .read_to_end(&mut bytes)
        .map_err(|error| invalid(format!("reading {}: {error}", file.name)))?;
    std::fs::write(&path, &bytes)
        .map_err(|error| invalid(format!("writing {}: {error}", path.display())))?;
    tracing::warn!(
        module = %file.name,
        path = %path.display(),
        "the store's file has no name this process can open, so the module is loaded from a copy \
         it owns"
    );
    Ok(NativeModule {
        copy: Some(path.clone()),
        path,
    })
}

impl Resolved {
    /// Whether this module was loaded from a copy rather than from the store's own file: the one
    /// fact that tells a store whose handles name their files from a store whose do not.
    fn copied(&self) -> bool {
        self.native.as_ref().is_some_and(NativeModule::copied)
    }

    /// The module as `ops::install` takes it.
    fn as_ops(&self) -> executor::Result<ops::Module<'_>> {
        match (&self.image, &self.native) {
            (Some(image), _) => Ok(ops::Module::Cubin {
                image,
                metadata: self.metadata.as_deref(),
                entry: &self.entry,
            }),
            (None, Some(native)) => Ok(ops::Module::Native {
                path: native.path.as_path(),
                entry: &self.entry,
            }),
            (None, None) => Err(executor::Error::Invalid(
                "a resolved module carries neither a cubin image nor a native library".to_owned(),
            )),
        }
    }
}

/// An error with every cause behind it, in one line.
///
/// A refusal that names only its own variant is the one thing a log cannot act on: `platform: ...`
/// with the driver's own message behind it says what failed, and the chain is where that message
/// lives. Each cause is taken from `source`, so nothing here has to know which errors there are.
fn causes<E: std::error::Error + std::fmt::Debug>(error: &E) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    // A failure whose own message is empty is the one a reader cannot act on, and an error type is
    // free to have one: the whole value says what it was, so it is what gets said.
    if text.trim().is_empty() {
        return format!("{error:?}");
    }
    text
}

/// States what an object defines when the entry a package declared is not among its symbols.
///
/// `dlsym` answers yes or no and nothing else, so a `NativeSymbol` failure names the entry that was
/// looked for and leaves "what does this object have" unanswered - the fact that turns the failure
/// into a fix. The line never changes the failure: the install still returns exactly what it did.
fn report_missing_entry(error: &ops::Error) {
    let ops::Error::Platform(platform::Error::NativeSymbol { path, symbol, .. }) = error else {
        return;
    };
    match defined_entries(path) {
        Some((total, names)) => tracing::warn!(
            symbol = %symbol,
            path = %path.display(),
            defined = total,
            entries = %names.join(", "),
            "the object does not define the entry the package declared: these are the entries it \
             does define"
        ),
        None => tracing::warn!(
            symbol = %symbol,
            path = %path.display(),
            "the object does not define the entry the package declared, and this process could \
             not read the object's own entry table"
        ),
    }
}

/// The functions an ELF object defines for the dynamic linker, by name: how many, and the first few.
///
/// Only the dynamic symbol table and its string table are read, and every read is bounds-checked
/// against the file. Anything this reader does not understand - another class, another byte order,
/// a table that runs off the end - is `None`, so nothing here can become the fault it reports.
fn defined_entries(path: &std::path::Path) -> Option<(usize, Vec<String>)> {
    let bytes = std::fs::read(path).ok()?;
    let names = defined_names(&bytes)?;
    let first = names.iter().take(8).cloned().collect();
    Some((names.len(), first))
}

/// Every function name an ELF object defines for the dynamic linker.
///
/// The same table `defined_entries` reads, whole: this is what answers whether a particular file is
/// the library of a module - the file that defines the entry the package declared.
fn defined_names(bytes: &[u8]) -> Option<Vec<String>> {
    const SHT_DYNSYM: u32 = 11;
    const STT_FUNC: u8 = 2;

    let sections = sections(bytes)?;
    let symbols = sections
        .iter()
        .position(|section| section.kind == SHT_DYNSYM)?;
    let (symbols_at, symbols_size) = (sections[symbols].at, sections[symbols].size);
    let strings = sections.get(sections[symbols].link)?;
    let table = bytes.get(strings.at..strings.at.checked_add(strings.size)?)?;

    let mut names = Vec::new();
    let limit = symbols_at.checked_add(symbols_size)?;
    let mut at = symbols_at;
    while at.checked_add(24)? <= limit {
        let name = quarter(bytes, at)? as usize;
        let info = *bytes.get(at.checked_add(4)?)?;
        let section = half(bytes, at.checked_add(6)?)?;
        at = at.checked_add(24)?;
        if name == 0 || section == 0 || (info & 0xf) != STT_FUNC {
            continue;
        }
        let tail = table.get(name..)?;
        let end = tail.iter().position(|held| *held == 0)?;
        names.push(std::str::from_utf8(&tail[..end]).ok()?.to_owned());
    }
    Some(names)
}

/// One section header, as much of it as this file reads.
struct Section {
    kind: u32,
    at: usize,
    size: usize,
    link: usize,
}

/// The section headers of a 64-bit little-endian ELF object.
fn sections(bytes: &[u8]) -> Option<Vec<Section>> {
    const ELFCLASS64: u8 = 2;
    const ELFDATA2LSB: u8 = 1;

    if bytes.get(0..4)? != b"\x7fELF"
        || *bytes.get(4)? != ELFCLASS64
        || *bytes.get(5)? != ELFDATA2LSB
    {
        return None;
    }
    let at = word(bytes, 0x28)? as usize;
    let entry = half(bytes, 0x3a)? as usize;
    let count = half(bytes, 0x3c)? as usize;
    let mut sections = Vec::with_capacity(count.min(256));
    for index in 0..count {
        let header = at.checked_add(index.checked_mul(entry)?)?;
        sections.push(Section {
            kind: quarter(bytes, header.checked_add(4)?)?,
            at: word(bytes, header.checked_add(0x18)?)? as usize,
            size: word(bytes, header.checked_add(0x20)?)? as usize,
            link: quarter(bytes, header.checked_add(0x28)?)? as usize,
        });
    }
    Some(sections)
}

/// The run path an ELF object states, from the first dynamic entry of the given tags.
fn dynamic_string(bytes: &[u8], tags: &[u64]) -> Option<String> {
    const SHT_DYNAMIC: u32 = 6;

    let sections = sections(bytes)?;
    let dynamic = sections
        .iter()
        .find(|section| section.kind == SHT_DYNAMIC)?;
    let strings = sections.get(dynamic.link)?;
    let table = bytes.get(strings.at..strings.at.checked_add(strings.size)?)?;
    let limit = dynamic.at.checked_add(dynamic.size)?;
    let mut at = dynamic.at;
    while at.checked_add(16)? <= limit {
        let tag = word(bytes, at)?;
        let value = word(bytes, at.checked_add(8)?)? as usize;
        at = at.checked_add(16)?;
        if !tags.contains(&tag) {
            continue;
        }
        let tail = table.get(value..)?;
        let end = tail.iter().position(|held| *held == 0)?;
        return std::str::from_utf8(&tail[..end]).ok().map(str::to_owned);
    }
    None
}

fn word(bytes: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(at..at.checked_add(8)?)?.try_into().ok()?,
    ))
}

fn half(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        bytes.get(at..at.checked_add(2)?)?.try_into().ok()?,
    ))
}

fn quarter(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(at..at.checked_add(4)?)?.try_into().ok()?,
    ))
}

/// Reads a resolved module into the form `ops::install` takes, or refuses a form this build cannot
/// install (§4.2): a scheduling policy's wasm program and a program the main process runs as its
/// own process are not kernels.
fn resolved(module: program_server::Module) -> executor::Result<Resolved> {
    use std::io::Read as _;

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
            Ok(Resolved {
                image: None,
                metadata: None,
                native: Some(native_library(files.collect(), entry)?),
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

    /// One package's worth of the generation: its programs prepared, each key resolved for this
    /// node's target, and each module loaded into this device's context as a kernel.
    ///
    /// Answers the entries this device installed for that package, and how many of them were
    /// copied out of the store rather than loaded where they lie.
    fn install_group(
        &self,
        client: &mut Client,
        request: &PrepareRequest,
    ) -> executor::Result<PackageInstall> {
        // The one call that leaves this process while a generation is prepared: a run that hangs
        // before it reaches the Program Server and one that hangs inside it look the same from the
        // daemon, so both sides state it.
        // The packages are named, not counted: an operator the store splits across two backends is
        // asked in two requests, and a log that says only `programs=1` twice leaves the reader to
        // guess which package carries which phase's keys.
        let asking: Vec<String> = request
            .descriptors
            .iter()
            .map(|descriptors| {
                // One request may carry several of one package's operators, and a split operator's
                // slot states the phase its keys belong to - so both come from the entry whose
                // operator these descriptors are for.
                let entry = request
                    .programs
                    .iter()
                    .find(|entry| entry.slot.op == descriptors.op);
                let phase = entry
                    .and_then(|entry| entry.slot.phase)
                    .map(|phase| format!(" phase={}", phase.as_str()))
                    .unwrap_or_default();
                match entry.map(|entry| &entry.selector) {
                    Some(program_server::Selector::Fixed(program)) => format!(
                        "{}/{}@{}{} keys={}",
                        descriptors.op,
                        program.backend,
                        program.revision,
                        phase,
                        descriptors.keys.len()
                    ),
                    _ => format!(
                        "{}{} keys={}",
                        descriptors.op,
                        phase,
                        descriptors.keys.len()
                    ),
                }
            })
            .collect();
        tracing::info!(
            programs = request.programs.len(),
            descriptors = request.descriptors.len(),
            packages = ?asking,
            target = %self.target,
            "asking the Program Server to prepare this generation's modules"
        );
        let prepared = client
            .prepare(request, &mut |progress| {
                tracing::debug!(
                    path = %progress.path,
                    state = ?progress.state,
                    "building a module for the new generation"
                );
            })
            .map_err(|error| {
                self.note(&error);
                executor::Error::Invalid(format!("building the new generation: {}", causes(&error)))
            })?;

        let mut entries = Vec::new();
        let mut copies = 0_usize;
        let mut asked: Vec<(String, usize)> = Vec::with_capacity(prepared.programs.len());
        for chosen in &prepared.programs {
            let operator = ops::operator_name(&chosen.slot.op).ok_or_else(|| {
                executor::Error::Invalid(format!(
                    "operator {} has no implementation in this build",
                    chosen.slot.op
                ))
            })?;
            let keys: Vec<String> = request
                .descriptors
                .iter()
                .filter(|descriptor| descriptor.op == chosen.slot.op)
                .flat_map(|descriptor| descriptor.keys.iter().cloned())
                .collect();
            asked.push((
                format!("{}/{}", chosen.slot.op, chosen.program.backend),
                keys.len(),
            ));
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
                        tracing::warn!(
                            op = %chosen.slot.op,
                            key = %key,
                            error = ?error,
                            "resolving this generation's module failed"
                        );
                        executor::Error::Invalid(format!(
                            "resolving {} for {key}: {}",
                            chosen.slot.op,
                            causes(&error)
                        ))
                    })?;
                let module = resolved(module)?;
                if module.copied() {
                    copies += 1;
                }
                let entry = match ops::install(&self.device, operator, &key, module.as_ops()?) {
                    Ok(entry) => entry,
                    Err(error) => {
                        // The refusal that travels is one line; this is the whole value, so a
                        // failure whose own message is empty still shows which call, status and
                        // code it carried.
                        tracing::warn!(
                            operator,
                            key = %key,
                            error = ?error,
                            "installing this generation's kernel failed"
                        );
                        report_missing_entry(&error);
                        return Err(executor::Error::Invalid(format!(
                            "installing {operator} for {key}: {}",
                            causes(&error)
                        )));
                    }
                };
                if let Some(entry) = entry {
                    entries.push(entry);
                }
            }
        }
        // A package that installed nothing is not yet a failed generation: another package of the
        // same generation may have installed every kernel this device needs, and a program this
        // store has built no module of is the walk's business - it names the key it was expected to
        // serve, one capture at a time. The generation-wide refusal is `run`'s, where the answer of
        // every package is known.
        if !prepared.programs.is_empty() {
            if entries.is_empty() {
                let asked_here = asked
                    .iter()
                    .map(|(program, keys)| format!("{program} with {keys} key(s)"))
                    .collect::<Vec<_>>()
                    .join(", ");
                tracing::warn!(
                    target = %self.target,
                    asked = %asked_here,
                    "the Program Server prepared this package's programs and this device installed \
                     no module of any of them"
                );
            }
            tracing::info!(
                modules = entries.len(),
                copies,
                target = %self.target,
                "this package's modules are installed for the new generation"
            );
        }
        Ok(PackageInstall {
            entries,
            copies,
            prepared: asked,
        })
    }
}

/// What one package's request answered for this device, before the generation's kernels are gathered.
struct PackageInstall {
    /// The kernels this package installed into this device's context.
    entries: Vec<KernelEntry>,
    /// How many of them were copied out of the store rather than loaded where they lie.
    copies: usize,
    /// Each program the store selected for this package's request, and how many keys it was given.
    /// Read by the generation-wide refusal, which needs the key counts to say which of the two
    /// causes it met: a program given no key was never asked for a module, and one given keys whose
    /// build answered nothing is the store's own refusal.
    prepared: Vec<(String, usize)>,
}

impl HostAction for Install {
    type Output = Vec<KernelEntry>;

    /// Every package's half of §7.3 step 2 for this device, as one answer.
    ///
    /// The lane is admitted one task per device, so the packages are asked one after another under
    /// one lock of the client: each is its own request, and the entries of all of them are this
    /// device's kernels for the generation.
    fn run(&self) -> executor::Result<Self::Output> {
        let mut client = self
            .programs
            .lock()
            .map_err(|_| executor::Error::State("the Program Server client was poisoned".into()))?;
        let mut entries = Vec::new();
        let mut copies = 0_usize;
        let mut prepared: Vec<(String, usize)> = Vec::new();
        for request in &self.requests {
            let mut package = self.install_group(&mut client, request)?;
            entries.append(&mut package.entries);
            copies += package.copies;
            prepared.append(&mut package.prepared);
        }
        // A generation whose every package prepared its programs and installed no module of any of
        // them is a failure of the store's build, not a generation with fewer kernels: the walk
        // would find every operator missing and the update would stand in its device stage until a
        // bound ended it. One package with nothing is not this: the generation's other packages are
        // asked in their own requests, and a package this store has built nothing for leaves its
        // key to the walk, which names it.
        if entries.is_empty() && !prepared.is_empty() {
            let asked = prepared
                .iter()
                .map(|(program, keys)| format!("{program} with {keys} key(s)"))
                .collect::<Vec<_>>()
                .join(", ");
            tracing::warn!(
                target = %self.target,
                packages = self.requests.len(),
                "the Program Server prepared this generation's programs and this device installed \
                 no module of any of them"
            );
            return Err(executor::Error::Invalid(format!(
                "the Program Server prepared {} program(s) for target {} and this device installed \
                 no module of any of them ({asked}): a program given no key was never asked for a \
                 module, and one given keys built nothing",
                prepared.len(),
                self.target
            )));
        }
        // A generation's kernels are named once each: two packages of one operator cover disjoint
        // phases, and the same key twice would be the same kernel installed twice.
        entries.sort_by(|left, right| {
            (left.operator, &left.descriptor).cmp(&(right.operator, &right.descriptor))
        });
        entries.dedup_by(|left, right| {
            left.operator == right.operator && left.descriptor == right.descriptor
        });
        tracing::info!(
            modules = entries.len(),
            copies,
            packages = self.requests.len(),
            target = %self.target,
            "this device's modules are installed for the new generation"
        );
        Ok(entries)
    }

    fn release(&self) -> executor::Result<()> {
        Ok(())
    }

    fn input_outcome(&self) -> Outcome {
        Outcome::Success(())
    }
}

/// Why a promotion this process cannot serve is refused, or `None` when it can serve.
///
/// A standby is promoted to serve the generation the epoch prepared in it (§2.5), and nothing in
/// `Promote` states that generation - the message is empty and `Promoted` answers with the launch
/// number the epoch layer promotes by - so this process cannot tell a generation the epoch still
/// holds from one it has reverted. What it *can* tell is whether it holds a generation at all: a
/// promotion with nothing prepared is a process that would open admission with no kernel and no
/// graph, and answer `Promoted` to a daemon that then moves the frontend onto it.
fn promote_refusal(standby: bool, prepared: usize) -> Option<String> {
    (standby && prepared == 0).then(|| {
        "this standby holds no prepared generation: its precopy was discarded or never ran, so          promoting it would serve a generation this process does not hold"
            .to_owned()
    })
}

/// The programs of an update that differ from what serves.
///
/// The update names every operator the store holds, and an operator this process does not serve is
/// no part of it. A program is compared by its backend and revision: its arguments are the package's
/// defaults, which the two sides state in different words.
fn updated_programs(named: &Programs, serving: &Programs) -> Programs {
    named
        .iter()
        .filter(|(slot, program)| {
            serving
                .iter()
                .filter(|(held, _)| {
                    held.op == slot.op
                        && (held.phase == slot.phase
                            || held.phase.is_none()
                            || slot.phase.is_none())
                })
                .any(|(_, held)| {
                    held.backend != program.backend || held.revision != program.revision
                })
        })
        .map(|(slot, program)| (slot.clone(), program.clone()))
        .collect()
}

/// The operators whose programs differ between what serves and what a precopy prepared (§7.3).
///
/// `changed` is not `new`. An operator the prepared generation holds and the serving one does not
/// has no old kernel to pair a shadowed step against, so it is not changed; the set is exactly the
/// intersection where the programs differ. It is empty when nothing serves yet - there is no old
/// kernel then either - which is what makes an epoch's first precopy pass without a shadow stage,
/// and why no rule here needs a generation number.
///
/// An operator is one operator however many packages fill it: a store that splits attention into a
/// prefill module and a decode one holds two slots of it, and changing either of them changes what
/// the operator launches. So the comparison is between the two sides' *sets* of programs per
/// operator, not between slots that a split may have renamed.
fn changed_operators(prepared: &Programs, serving: &Programs) -> Vec<String> {
    fn of<'a>(programs: &'a Programs, op: &str) -> Vec<(Option<PhaseKind>, &'a Program)> {
        let mut held: Vec<(Option<PhaseKind>, &Program)> = programs
            .iter()
            .filter(|(slot, _)| slot.op == op)
            .map(|(slot, program)| (slot.phase, program))
            .collect();
        held.sort_by(|left, right| (left.0, &left.1.backend).cmp(&(right.0, &right.1.backend)));
        held
    }
    let mut operators: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for slot in prepared.keys().chain(serving.keys()) {
        operators.insert(slot.op.as_str());
    }
    let mut changed: Vec<String> = Vec::new();
    for op in operators {
        let next = of(prepared, op);
        let old = of(serving, op);
        // An operator either side does not hold is new or retired, not changed: neither has a pair
        // of programs to compare.
        if next.is_empty() || old.is_empty() {
            continue;
        }
        if next != old {
            changed.push(op.to_owned());
        }
    }
    changed
}

/// The failure a shadow step that never ran is (§7.3 step 2).
///
/// The step runs inside a batch, and an engine with no resident request forms none: the stage waits
/// for something that cannot happen, so the update is abandoned with the operators named rather than
/// passing without the comparison it was there to make.
fn shadow_stall_message(changed: &[String], waited: u64) -> String {
    let count = changed.len();
    let operators = changed.join(", ");
    let noun = if count == 1 {
        "changed operator"
    } else {
        "changed operators"
    };
    format!(
        "the shadow step of {count} {noun} ({operators}) waited {waited}s: no resident request ran \
         a step, so nothing was compared - the update is abandoned rather than passed without the \
         comparison it was prepared for"
    )
}

/// The argument a policy's step is bounded by: the figure §4.4 checks against this rank's
/// catalogue of prefill buckets.
const TOKEN_LIMIT: &str = "max_num_batched_tokens";

/// The sentence a policy's token-limit refusal needs beside it (§4.4).
///
/// The mechanism's own words are `max_num_batched_tokens is 8192, above the build config's 512`,
/// and every figure in them is right and none of them is actionable: `512` is this rank's
/// catalogue of captured prefill buckets rather than a name, and `8192` is a figure no one in the
/// deployment chose when the policy's program names none. The refusal itself is the mechanism's
/// and is not changed here - what is added is where each figure came from, and the two changes
/// that would satisfy it.
///
/// A message that is not this refusal is answered unchanged, so every other policy failure reads
/// exactly as the mechanism wrote it.
fn limit_refusal(error: &str, stated: Option<&str>, catalogue: PrefillCatalogue) -> String {
    if !error.contains(TOKEN_LIMIT) || !error.contains("above the build config's") {
        return error.to_owned();
    }
    // The policy's own figure: the mechanism renders `{name} is {limit}, above …`, and that figure
    // is what a deployment would have to raise the catalogue to for the policy to fit.
    let declared = error
        .split(&format!("{TOKEN_LIMIT} is "))
        .nth(1)
        .map(|rest| {
            rest.chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
        })
        .filter(|digits| !digits.is_empty());
    let origin = if catalogue.stated {
        "the deployment's cuda_graph.prefill_token_sizes"
    } else {
        "the build's default catalogue, which this deployment's launch file does not state"
    };
    // Whose figure the policy's is: a program that names no such argument states none, and the
    // policy then applies its own default - the number nobody in this deployment chose.
    let source_of_policy = match stated {
        Some(value) => format!("the deployment states {TOKEN_LIMIT} = {value}"),
        None => {
            format!("the deployment states no {TOKEN_LIMIT}, so this is the policy's own default")
        }
    };
    let bound = catalogue.bound;
    let ways = match &declared {
        Some(figure) => format!(
            "state {TOKEN_LIMIT} at or below {bound}, or capture prefill buckets up to {figure}"
        ),
        None => {
            format!("state {TOKEN_LIMIT} at or below {bound}, or capture wider prefill buckets")
        }
    };
    format!(
        "{error}. This rank captures prefill buckets up to {bound} token(s), from {origin}; \
         {source_of_policy}. Either {ways}."
    )
}

/// The precopy of one generation, advanced between serve passes (§7.3 step 2).
struct Precopy {
    generation: u64,
    /// The peer whose command this precopy answers, when a peer asked.
    reply: Option<u32>,
    /// The operators whose program this precopy changed, in sorted order: the set the shadow pass
    /// pairs, and the set a stall names. Empty when nothing changed - a first generation, or an
    /// update that touched no kernel.
    changed: Vec<String>,
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
    /// The stage the precopy is in, since when, and whether the wait has already been reported:
    /// every stage of §7.3 step 2 is a poll, so a stage that never advances is a run that hangs
    /// without a line unless the process says which step it is standing on.
    stage_name: &'static str,
    since: Instant,
    warned: bool,
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

impl Precopy {
    /// Whether this generation holds a program for `operator`.
    ///
    /// A program is named by a slot, and a slot names an operator and an optional phase, so the
    /// question a walk asks - "does this generation have an implementation of this operator?" - is
    /// answered by any slot of that operator, whatever phase it states.
    fn names(&self, operator: &str) -> bool {
        self.programs
            .keys()
            .any(|slot| ops::operator_name(&slot.op) == Some(operator))
    }
}

/// The stage a precopy is in.
enum PrecopyStage {
    /// The module-load lane, in the two steps the update runs it in (§7.3 step 2).
    Modules(ModuleLane),
    /// The executors rebind or recapture their buckets.
    Executors(Vec<Submission>),
    /// The shadow pass compares the new implementations on the resident requests, one step per
    /// scheduler; each answers when its step ran.
    Shadow(Vec<crossbeam_channel::Receiver<std::result::Result<(), String>>>),
    /// A stage being replaced; the driver never observes it.
    Done,
}

/// The module-load lane's two steps, in the order the update needs them.
enum ModuleLane {
    /// The store's listing, alone on the lane: the installs are derived from it.
    Listing(Arc<HostTask<ModuleWork>>),
    /// One install per device for the generation the listing resolved.
    Installs(Vec<Arc<HostTask<ModuleWork>>>),
}

/// A wait of the control path: when it began and when it was last reported.
///
/// Every step of §7.3 is polled rather than blocked on, so a step that never advances is a run
/// that hangs without a line. Each such wait reports itself by name, generation and elapsed time
/// once per stall bound, and never ends the epoch: the command's own failure path decides, and a
/// deadline that killed would be a second failure beside the one being chased.
#[derive(Clone, Copy, Debug, Default)]
struct Wait {
    since: Option<Instant>,
    reported_at: Option<Instant>,
}

impl Wait {
    /// Notes whether the wait is on: it begins on the first pass that finds it waiting, and the
    /// next pass that finds it over clears it.
    fn note(&mut self, waiting: bool, now: Instant) {
        if waiting {
            self.since.get_or_insert(now);
        } else {
            self.since = None;
            self.reported_at = None;
        }
    }

    /// Reports the wait when it has stood a stall bound since it began, and again once per bound
    /// after each report. Returns whether this pass reported it.
    fn report(
        &mut self,
        name: &'static str,
        generation: u64,
        bound: Duration,
        now: Instant,
    ) -> bool {
        let Some(since) = self.since else {
            return false;
        };
        if now.duration_since(since) < bound {
            return false;
        }
        if self
            .reported_at
            .is_some_and(|at| now.duration_since(at) < bound)
        {
            return false;
        }
        self.reported_at = Some(now);
        tracing::warn!(
            generation,
            wait = name,
            waited_s = now.duration_since(since).as_secs(),
            "the control path is still waiting on this step and nothing has answered it yet"
        );
        true
    }
}

/// The name of a precopy stage, for the line that says which step the process is standing on.
fn stage_name(stage: &PrecopyStage) -> &'static str {
    match stage {
        PrecopyStage::Modules(_) => "the module load lane",
        PrecopyStage::Executors(_) => "this node's devices",
        PrecopyStage::Shadow(_) => "the schedulers' shadow step",
        PrecopyStage::Done => "done",
    }
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
/// What one stage is still waiting for, in one phrase.
///
/// A stage's name says where a hang is, not which of the stage's several waits it is: the module
/// lane holds one task per device, the executor stage one submission per device, and the shadow
/// stage one step per scheduler. The stall line prints the counts so a run names the wait that
/// never answered without a second run built for the purpose.
/// What a device stage is waiting on: each unanswered device's own account, and each scheduler's
/// answer to the policy it was asked to build.
///
/// The two waits live in one stage, and a report that names only the devices says "the device thread
/// is the one to look at" while the device has answered and a policy build is what never came back -
/// which is a wrong name, worse than none. The policy figure is stated first for that reason: it is
/// the wait the devices' own account cannot show.
fn device_work(
    submissions: &[Submission],
    executors: &[Arc<Executor>],
    policies: &[crossbeam_channel::Receiver<std::result::Result<(), String>>],
) -> String {
    // The label alone says which stage stood; the device thread's own account says whether it is
    // working (a capture in progress, a bucket being recaptured) or waiting on something outside it.
    // A capture is open or it is not, and the two look identical from the control thread without it.
    let mut pending = Vec::new();
    for (rank, submission) in submissions.iter().enumerate() {
        if submission.outcome().is_some() {
            continue;
        }
        let Some(executor) = executors.get(rank) else {
            pending.push(format!("rank {rank}: no executor"));
            continue;
        };
        let performing = executor
            .performing()
            .unwrap_or_else(|| "nothing".to_owned());
        pending.push(format!(
            "rank {rank}: {performing}, {} open capture(s)",
            executor.capturing()
        ));
    }
    let answered = submissions.len() - pending.len();
    let devices = format!(
        "{answered} of {} device preparation(s) answered ({})",
        submissions.len(),
        pending.join("; ")
    );
    let policies: String = if policies.is_empty() {
        "no policy build was asked for".to_owned()
    } else {
        let answered = policies.iter().filter(|answer| !answer.is_empty()).count();
        format!("{answered} of {} policy build(s) answered", policies.len())
    };
    format!("{devices}; {policies}")
}

fn pending_work(
    stage: &PrecopyStage,
    executors: &[Arc<Executor>],
    policies: &[crossbeam_channel::Receiver<std::result::Result<(), String>>],
) -> String {
    match stage {
        PrecopyStage::Modules(ModuleLane::Listing(task)) => match task.outcome() {
            None => "the Program Server's store listing".to_owned(),
            Some(_) => "the store's listing (answered)".to_owned(),
        },
        PrecopyStage::Modules(ModuleLane::Installs(tasks)) => {
            let answered = tasks.iter().filter(|task| task.outcome().is_some()).count();
            format!("{answered} of {} kernel load(s) answered", tasks.len())
        }
        PrecopyStage::Executors(submissions) => device_work(submissions, executors, policies),
        PrecopyStage::Shadow(answers) => {
            let answered = answers.iter().filter(|answer| !answer.is_empty()).count();
            format!("{answered} of {} shadow step(s) answered", answers.len())
        }
        // The driver replaces the stage with this before it looks at it, so it is never the wait.
        PrecopyStage::Done => "no stage".to_owned(),
    }
}

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
/// The package the store holds for `operator` when none of them is built: the highest revision,
/// ties broken by the backend's name so the choice is the same on every run.
///
/// This is the answer for an operator nothing has captured yet - the package that states its keys
/// in its manifest but has no module for this host - which is what `select_programs` cannot see,
/// because its own question is what this host can launch *now*.
/// The answer to a `Commit`.
///
/// A commit whose generation is not on probation names nothing the machine holds: the daemon may be
/// one message behind the revert that its own rollback threshold caused, so its commit arrives after
/// the update ended. The state that holds is the one before the update - it serves, nothing was
/// committed - and that is the answer. A protocol message never ends a process that can still serve.
/// What a switchover answers: the status now, or nothing until its warmup has run.
///
/// `Switched` states that this process serves, and it cannot serve a batch before the buckets that
/// batch may be planned into exist - so a switchover that queued a warmup answers after admission
/// opens, and a warmup that fails answers the switchover with the failure instead. A switchover with
/// nothing to warm has nothing to wait for.
#[derive(Debug)]
enum SwitchoverAnswer {
    Now(ProgramStatus),
    AfterTheWarmup,
}

fn switchover_answer(warming: usize, generation: u64) -> SwitchoverAnswer {
    if warming == 0 {
        SwitchoverAnswer::Now(ProgramStatus::Switched { generation })
    } else {
        SwitchoverAnswer::AfterTheWarmup
    }
}

fn answer_commit(update: &mut executor::hot_update::HotUpdate, generation: u64) -> ProgramStatus {
    // An epoch switch commits its target, whose programs were committed when the daemon handed them
    // over: the generation it names already serves with nothing on trial, so the answer is that.
    if update.current() == generation
        && matches!(update.phase(), executor::hot_update::Phase::Serving)
    {
        return ProgramStatus::Committed { generation };
    }
    match update.commit(generation) {
        Ok(()) => ProgramStatus::Committed { generation },
        Err(error) => {
            tracing::warn!(
                generation,
                serving = update.current(),
                %error,
                "a commit names a generation that is not on probation"
            );
            ProgramStatus::Reverted {
                generation,
                reason: format!(
                    "the commit names a generation that is not in flight ({error}): generation {} \
                     serves",
                    update.current()
                ),
            }
        }
    }
}

fn declared_program(
    packages: &[program_server::protocol::ListedPackage],
    operator: &str,
) -> Option<(String, String, u32)> {
    let mut declared: Vec<&program_server::protocol::ListedPackage> = packages
        .iter()
        .filter(|package| ops::operator_name(&package.op) == Some(operator))
        .collect();
    declared.sort_by(|left, right| {
        right
            .revision
            .cmp(&left.revision)
            .then_with(|| left.backend.cmp(&right.backend))
    });
    let package = declared.first()?;
    Some((
        package.op.clone(),
        package.backend.clone(),
        package.revision,
    ))
}

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
            // A store built for another device is the one store-side defect a run cannot name from
            // the outside: the package is there, the operator exists, and nothing installs. Both
            // sets are printed, because the difference between them is the fix.
            tracing::info!(
                op = %package.op,
                backend = %package.backend,
                held = ?package
                    .modules
                    .iter()
                    .map(|module| module.target.to_string())
                    .collect::<Vec<_>>(),
                wanted = ?targets.iter().map(ToString::to_string).collect::<Vec<_>>(),
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

/// The phase a descriptor key states, when its operator's key names one.
///
/// A kernel program may be split by phase, and the phase is the first field of the key's own
/// `Display` form (`phase_decode_dtype_bf16_kv_...`). An operator whose key states no phase - a
/// `gemm`, an `expert_exchange` - answers `None`, and that `None` is what keeps such an operator
/// from ever being split across packages: nothing in its keys says which of them a package covers.
fn key_phase(key: &str) -> Option<PhaseKind> {
    let rest = key.strip_prefix("phase_")?;
    let (name, _) = rest.split_once('_')?;
    PhaseKind::ALL
        .into_iter()
        .find(|phase| phase.as_str() == name)
}

/// Whether a descriptor key belongs to a slot's phase.
///
/// A generation may hold two packages of one operator, one per phase, and the Program Server hands
/// a request's program every key the request names for its operator: a slot that took the other
/// phase's key would have its package asked for a key it cannot build. A key whose text states no
/// phase fits every slot - it is the only kind a phase-less operator has.
fn key_fits_phase(key: &str, phase: Option<PhaseKind>) -> bool {
    match phase {
        None => true,
        Some(phase) => key_phase(key).map_or(true, |held| held == phase),
    }
}

/// The slots one operator's generation holds, from what each candidate package built for this host.
///
/// An operator may be built by more than one package - `paged_attention` is fa4's prefill module and
/// trtllm's decode one - and then its generation must hold *both*: the forward launches a prefill
/// key and a decode key, and a generation that holds one package serves one of them and refuses the
/// other ("no kernel for operator paged_attention descriptor phase_prefill_..."). So the packages
/// are split by the phase their own built keys state, one slot per phase, `holders` being in the
/// deployment's preference order.
///
/// The split is only stated when it is exact: every key of a holder names a phase, and no phase is
/// claimed twice. A package covering everything, an operator whose keys state no phase, or packages
/// overlapping in one phase fall back to the single slot this has always been - the deployment's
/// preference, or the store's order, decides - because a wrong split is worse than no split: two
/// slots of one operator would have each package asked for a key it cannot serve.
fn operator_slots(
    op: &str,
    holders: &[(Program, Vec<String>)],
    fallback: Option<&Program>,
) -> Vec<(ProgramSlot, Program)> {
    let slot = |phase: Option<PhaseKind>| ProgramSlot {
        op: op.to_owned(),
        phase,
        part: None,
    };
    let (claims, refusal) = split_claims(holders);
    if refusal.is_none() && claims.len() > 1 {
        return claims
            .into_iter()
            .map(|(phase, index)| (slot(Some(phase)), holders[index].0.clone()))
            .collect();
    }
    // Packages whose built keys share none divide the operator by descriptor - a bf16 module and an
    // f32 one, whose keys state no phase - so each holds a slot of its own: one program would serve
    // its own keys and refuse the other's at the first capture that launches them.
    if divides_descriptors(holders) {
        return holders
            .iter()
            .map(|(program, _)| {
                (
                    ProgramSlot {
                        part: Some(program.backend.clone()),
                        ..slot(None)
                    },
                    program.clone(),
                )
            })
            .collect();
    }
    // The fallback is right - a wrong split would have a package asked for a key it cannot build -
    // and it used to be silent, which is what made a run's `no kernel for operator X descriptor
    // phase_...` read as a fresh mystery instead of the consequence of a decision. So the decision
    // says itself here, in the words `split_claims` gives it.
    if let Some(reason) = refusal {
        let packages: Vec<&str> = holders
            .iter()
            .map(|(program, _)| program.backend.as_str())
            .collect();
        tracing::warn!(
            op = %op,
            %reason,
            packages = ?packages,
            "this operator's phases are not split across packages, so one program fills it: a key \
             of another phase will fail at its capture unless every package covers it"
        );
    }
    match holders.first().map(|(program, _)| program).or(fallback) {
        Some(program) => vec![(slot(None), program.clone())],
        None => Vec::new(),
    }
}

/// Whether every package built keys of its own, none of them shared: then each one's descriptors are
/// its alone, and the generation needs them all.
fn divides_descriptors(holders: &[(Program, Vec<String>)]) -> bool {
    if holders.len() < 2 || holders.iter().any(|(_, keys)| keys.is_empty()) {
        return false;
    }
    let mut seen = std::collections::BTreeSet::new();
    holders
        .iter()
        .flat_map(|(_, keys)| keys)
        .all(|key| seen.insert(key))
}

/// The phases an operator's packages claim, and - when they cannot be split across them - why.
///
/// `claims` is one entry per (phase, package) a package's built keys name; an operator whose claims
/// are one per phase is split across them, and anything else is filled by one program. `Some(reason)`
/// is the two cases that refuse a split, stated with the packages that brought them about: two
/// packages building one phase, or a package whose key states no phase at all. `None` with fewer than
/// two packages is not a refusal - there is nothing to split.
fn split_claims(holders: &[(Program, Vec<String>)]) -> (Vec<(PhaseKind, usize)>, Option<String>) {
    let phaseful = holders
        .iter()
        .all(|(_, keys)| keys.iter().all(|key| key_phase(key).is_some()));
    let mut claims: Vec<(PhaseKind, usize)> = Vec::new();
    let mut clash: Option<(PhaseKind, usize, usize)> = None;
    if phaseful && holders.len() > 1 {
        for (index, (_, keys)) in holders.iter().enumerate() {
            let mut phases: Vec<PhaseKind> = keys.iter().filter_map(|key| key_phase(key)).collect();
            phases.sort_unstable();
            phases.dedup();
            for phase in phases {
                if let Some((_, held)) = claims.iter().find(|(claimed, _)| *claimed == phase) {
                    clash = clash.or(Some((phase, *held, index)));
                }
                claims.push((phase, index));
            }
        }
    }
    let refusal = match (holders.len(), clash) {
        (0 | 1, _) => None,
        (_, Some((phase, first, second))) => Some(format!(
            "two packages build the {} phase: {} and {}",
            phase, holders[first].0.backend, holders[second].0.backend
        )),
        (_, None) if !phaseful => {
            let unphased = holders
                .iter()
                .find(|(_, keys)| keys.iter().any(|key| key_phase(key).is_none()))
                .map(|(program, _)| program.backend.as_str())
                .unwrap_or("one of them");
            Some(format!("a key {unphased} built states no phase"))
        }
        _ => None,
    };
    (claims, refusal)
}

/// The generation's programs, from the deployment's named programs and this host's store.
///
/// `chosen` is the store's candidates per operator, in its own preference order; `built` is what
/// each package built for this host's targets, keyed by the package. Pure, so the resolution a run
/// depends on can be tested with the shape a `Precopy` actually hands it.
fn resolve_slots(
    named: &Programs,
    chosen: &BTreeMap<String, Vec<(String, u32)>>,
    built: &BTreeMap<(String, String, u32), Vec<String>>,
    packages: &[program_server::protocol::ListedPackage],
) -> std::result::Result<(Programs, Candidates), String> {
    let keys_of = |op: &str, program: &Program| -> Vec<String> {
        built
            .get(&(op.to_owned(), program.backend.clone(), program.revision))
            .cloned()
            .unwrap_or_default()
    };
    let mut programs = Programs::new();
    let mut candidates = Candidates::new();
    // Every operator the store supplies, and every one the deployment names - a named program whose
    // operator the store does not supply is still installed, and the walk reports what it cannot
    // bind.
    let mut operators: Vec<String> = chosen.keys().cloned().collect();
    for slot in named.keys() {
        if !operators.contains(&slot.op) {
            operators.push(slot.op.clone());
        }
    }
    let named_here = |op: &str| -> Vec<(Option<PhaseKind>, Program)> {
        named
            .iter()
            .filter(|(slot, _)| slot.op == op)
            .map(|(slot, program)| (slot.phase, program.clone()))
            .collect()
    };
    for op in operators {
        let stated = named_here(&op);
        // A deployment that names a program has chosen an operator's implementation, and the name
        // is a fact rather than a preference: this host either built that program or it did not.
        // When it did not, substituting another program is not the smaller failure - the named
        // program's keys are unknown here, so a substitute may cover no more than the phase it was
        // not named for, and the walk finds the rest one capture at a time (a generation that holds
        // one program of a two-phase operator stands at its first warmup). The refusal names what
        // the store does hold, which is the fact the operator needs to build or to name.
        if let Some((_, program)) = stated
            .iter()
            .find(|(_, program)| keys_of(&op, program).is_empty())
        {
            // What the store has built for the operator, and what it holds without a module: the
            // two are different facts and the message states both, because a package that is
            // declared and unbuilt is a store build away from serving, while a package that is not
            // there at all is a deployment question.
            let held: Vec<String> = chosen
                .get(&op)
                .into_iter()
                .flatten()
                .map(|(backend, revision)| format!("{backend} revision {revision}"))
                .collect();
            let built_here = |backend: &str, revision: u32| {
                chosen
                    .get(&op)
                    .is_some_and(|held| held.iter().any(|(b, r)| b == backend && *r == revision))
            };
            let declared: Vec<String> = packages
                .iter()
                .filter(|package| {
                    package.op == op && !built_here(&package.backend, package.revision)
                })
                .map(|package| format!("{} revision {}", package.backend, package.revision))
                .collect();
            let mut answer = format!(
                "the deployment names {op}/{} revision {} and this host has built no module of it \
                 for its targets: the store holds {}",
                program.backend,
                program.revision,
                if held.is_empty() {
                    format!("no built package of {op}")
                } else {
                    held.join(", ")
                }
            );
            if !declared.is_empty() {
                answer.push_str(&format!(
                    " and declares {} without a module for this host's targets",
                    declared.join(", ")
                ));
            }
            answer.push_str(
                " - build the named package for this host's targets, or name a backend the store \
                 holds",
            );
            return Err(answer);
        }
        // A deployment that states a backend per phase - a prefill attention backend and a decode
        // one - has already said which keys each of them covers, so nothing here is decided from
        // the store: both programs are the operator's, each in its own phase.
        if stated.iter().any(|(phase, _)| phase.is_some()) {
            for (phase, program) in &stated {
                programs.insert(
                    ProgramSlot {
                        op: op.clone(),
                        phase: *phase,
                        part: None,
                    },
                    program.clone(),
                );
            }
            if let Some(ordered) = chosen.get(&op) {
                candidates.insert(op, ordered.clone());
            }
            continue;
        }
        // The deployment's program comes first: it decides the phases its own package covers, and
        // the store's order decides the rest. A program is one package however it was written down,
        // so the named form and the store's form of the same package are one entry - with the
        // deployment's arguments, which are the ones the epoch pinned.
        let mut ordered: Vec<Program> = stated.into_iter().map(|(_, program)| program).collect();
        for (backend, revision) in chosen.get(&op).cloned().unwrap_or_default() {
            ordered.push(Program {
                backend,
                revision,
                arguments: Default::default(),
            });
        }
        let mut seen: Vec<(String, u32)> = Vec::with_capacity(ordered.len());
        ordered.retain(|program| {
            let package = (program.backend.clone(), program.revision);
            if seen.contains(&package) {
                return false;
            }
            seen.push(package);
            true
        });
        // What each program built for this host's targets: the keys a phase split is read from.
        let holders: Vec<(Program, Vec<String>)> = ordered
            .iter()
            .map(|program| (program.clone(), keys_of(&op, program)))
            .filter(|(_, keys)| !keys.is_empty())
            .collect();
        let was_named = named.keys().any(|slot| slot.op == op);
        if was_named && holders.first().map(|(program, _)| program) != ordered.first() {
            // The deployment named a program this host has built nothing of, and another package
            // has: the store decides rather than the name, so say so - silently serving another
            // backend than the one the deployment named would be the harder fact to find.
            tracing::warn!(
                op = %op,
                named = %ordered.first().map(|program| program.backend.as_str()).unwrap_or(""),
                built = %holders.first().map(|(program, _)| program.backend.as_str()).unwrap_or(""),
                "the named program is not built for this host's targets: the store's own package \
                 fills its operator"
            );
        }
        for (slot, program) in operator_slots(&op, &holders, ordered.first()) {
            programs.insert(slot, program);
        }
        if let Some(ordered) = chosen.get(&op) {
            candidates.insert(op, ordered.clone());
        }
    }
    Ok((programs, candidates))
}

/// Adds one slot's keys to a package's request, merging with a slot of an operator the package
/// already fills.
///
/// A request names one program per operator, and the Program Server hands that program every key
/// the request names for the operator, so two slots of one operator filled by the same package
/// belong in one entry rather than two that would each be asked for both key sets.
fn merge_request(
    group: &mut (Vec<ProgramRequest>, Vec<Descriptors>),
    slot: &ProgramSlot,
    program: &Program,
    keys: Vec<String>,
) {
    if let Some(index) = group
        .1
        .iter()
        .position(|descriptor| descriptor.op == slot.op)
    {
        let descriptor = &mut group.1[index];
        descriptor.keys.extend(keys);
        descriptor.keys.sort_unstable();
        descriptor.keys.dedup();
        return;
    }
    group.0.push(ProgramRequest {
        slot: slot.clone(),
        selector: Selector::Fixed(program.clone()),
    });
    group.1.push(Descriptors {
        op: slot.op.clone(),
        keys,
    });
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
/// (`specs/executor.md` §2.3).
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
    model: &crate::engine_plan::Served,
    executors: &[Arc<Executor>],
    rank: &ipc::MainProcessInfo,
    control: ipc::Wake,
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
    let facts = plan.rank_facts(model)?;
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
            // Every component the rank reports is placed on the group's ranks. A text rank reports
            // the one default component, and a media rank one per block of its pipeline, which the
            // scheduler reads a call's width and owner from.
            components: rank
                .components
                .iter()
                .flat_map(|component| {
                    engine::ExecutorGroupConfig::single_component(&component.name, plan.world_size())
                })
                .collect(),
        },
        max_model_len: plan.max_kv_len(model)?,
    };
    let batches: Arc<dyn executor::batch::Batches> = Arc::new(
        executor::batch::Executors::shared(executors.iter().map(Arc::clone).collect())?
            .stating(vec![inventory]),
    );
    // The control thread forwards what a scheduler produced, so it sleeps until a step says there
    // is something: a request's tokens reach the daemon when they exist, not at the next pass.
    let progress = engine::CommandWaker::new(move || control.wake());
    let handle = engine::SchedulerHandle::over_batches_telling(config, batches, progress)?;
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
        // The request link: requests, their events and the epoch's metrics. The process link
        // states `PROCESS` on its own socket, and both name the same engine number.
        interface: proto::process::Interface::Requests as i32,
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
        predecessor_world: spec.predecessor.as_ref().map(|predecessor| predecessor.world),
        host: spec.host,
    })
}

#[cfg(test)]
mod tests {

    /// An update names the whole set, and what it installs is what differs from what serves.
    ///
    /// The operator that stayed keeps its kernel, and with it every bucket it launches in: loading it
    /// again read as a changed program, so an update of `rms_norm` compared `paged_attention` instead.
    #[test]
    fn an_update_installs_only_the_programs_that_changed() {
        let program = |backend: &str, revision| Program {
            backend: backend.to_owned(),
            revision,
            arguments: Default::default(),
        };
        let slot = |op: &str| program_server::program::ProgramSlot::all(op);
        let serving: Programs = [
            (slot("rms_norm"), program("triton", 1)),
            (slot("paged_attention"), program("trtllm", 1)),
        ]
        .into_iter()
        .collect();
        let named: Programs = [
            (slot("rms_norm"), program("triton", 2)),
            (slot("paged_attention"), program("trtllm", 1)),
            (slot("gelu"), program("triton", 1)),
        ]
        .into_iter()
        .collect();
        let updated = updated_programs(&named, &serving);
        assert_eq!(updated.len(), 1, "{updated:?}");
        assert_eq!(updated[&slot("rms_norm")].revision, 2);
    }

    use super::*;
    /// A takeover replays what the predecessor produced past the position the successor's consumer
    /// states, and nothing else (§7.5.1 step 4).
    ///
    /// The pause freezes the predecessor's response, so its consumer stops advancing while this
    /// process goes on producing; without the replay those events are rendered nowhere and the
    /// response loses the tokens the pause covered.
    #[test]
    fn a_takeover_replays_only_the_events_past_the_position_it_names() {
        let event = |request: u64, message: &str| proto::request::wire::Event {
            request,
            kind: Some(proto::request::wire::event::Kind::Error(
                proto::request::wire::Failed {
                    message: message.to_owned(),
                },
            )),
        };
        let kept: std::collections::VecDeque<_> = (1..=4).map(|at| (at, event(9, "e"), None)).collect();

        assert_eq!(replay_after(&kept, 0).len(), 4, "a takeover from the start");
        assert_eq!(
            replay_after(&kept, 2).len(),
            2,
            "only the events past the position"
        );
        assert!(
            replay_after(&kept, 4).is_empty(),
            "a consumer at the last event has nothing left to replay"
        );
    }

    /// A takeover replays a backlog wider than the consumer's event window.
    ///
    /// The window is the credit a consumer reports in, not the distance it can
    /// be behind by: the daemon files this process's events without bound and
    /// writes them at the client's pace, so at a fast decode the body of a
    /// response that pauses is tens of events behind what was produced. A
    /// journal capped at the window drops the oldest of those and the handover
    /// loses exactly them.
    #[test]
    fn a_takeover_replays_a_backlog_wider_than_the_event_window() {
        let event = |request: u64, message: &str| proto::request::wire::Event {
            request,
            kind: Some(proto::request::wire::event::Kind::Error(
                proto::request::wire::Failed {
                    message: message.to_owned(),
                },
            )),
        };
        let backlog = proto::request::EVENT_WINDOW as usize * 3;
        let mut kept = std::collections::VecDeque::new();
        for at in 1..=backlog as u64 {
            keep_event(&mut kept, at, event(9, "e"), None, HANDOVER_JOURNAL);
        }

        assert!(
            kept.len() > proto::request::EVENT_WINDOW as usize,
            "the journal holds the whole backlog: {} of {backlog}",
            kept.len()
        );
        assert_eq!(
            replay_after(&kept, 1).len(),
            backlog - 1,
            "a takeover from the first event replays the backlog whole"
        );
    }

    /// The journal drops its oldest only past its bound, and what it holds
    /// is replayable from any position inside it.
    #[test]
    fn the_journal_is_bounded_by_its_own_bound_and_not_by_the_window() {
        let event = |request: u64, message: &str| proto::request::wire::Event {
            request,
            kind: Some(proto::request::wire::event::Kind::Error(
                proto::request::wire::Failed {
                    message: message.to_owned(),
                },
            )),
        };
        let mut kept = std::collections::VecDeque::new();
        let positions = HANDOVER_JOURNAL as u64 + 10;
        for at in 1..=positions {
            keep_event(&mut kept, at, event(9, "e"), None, HANDOVER_JOURNAL);
        }

        assert_eq!(
            kept.len(),
            HANDOVER_JOURNAL,
            "the journal keeps its own bound"
        );
        assert_eq!(
            kept.front().map(|(position, _, _)| *position),
            Some(positions - HANDOVER_JOURNAL as u64 + 1),
            "the oldest event kept is the one at the bound"
        );
        assert_eq!(
            replay_after(&kept, 0).len(),
            HANDOVER_JOURNAL,
            "a takeover from before the journal replays all of it"
        );
    }

    /// An epoch's first precopy has no serving generation, so no operator has an old kernel to pair
    /// a shadowed step against: the changed set is empty and the precopy finishes without asking
    /// for a step an idle epoch never runs.
    #[test]
    fn a_first_generation_changes_no_operator_and_asks_for_no_shadow() {
        let mut prepared = Programs::new();
        prepared.insert(
            program_server::program::ProgramSlot {
                op: "paged_attention".to_owned(),
                phase: None,
                part: None,
            },
            Program {
                backend: "fa4".to_owned(),
                revision: 3,
                arguments: Default::default(),
            },
        );

        assert!(
            changed_operators(&prepared, &Programs::new()).is_empty(),
            "a first generation reported a changed operator, which would wait for a shadow step \
             that cannot run"
        );
    }

    /// `changed` is the intersection where the two programs differ, and not `new`: an operator the
    /// prepared generation adds has no old kernel either, so it is not changed.
    #[test]
    fn only_an_operator_whose_program_differs_is_changed() {
        let slot = |op: &str| program_server::program::ProgramSlot {
            op: op.to_owned(),
            phase: None,
            part: None,
        };
        let program = |backend: &str, revision: u32| Program {
            backend: backend.to_owned(),
            revision,
            arguments: Default::default(),
        };
        let mut serving = Programs::new();
        serving.insert(slot("gemm"), program("cublas", 1));
        serving.insert(slot("softmax"), program("reference", 2));

        let mut prepared = serving.clone();
        prepared.insert(slot("gemm"), program("cublas", 2));
        prepared.insert(slot("rms_norm"), program("reference", 1));

        assert_eq!(changed_operators(&prepared, &serving), vec!["gemm"]);
    }

    /// A shadow step runs inside a batch a resident request formed; an engine with none forms no
    /// batch, so the stage fails by name rather than waiting out the daemon's bound and passing
    /// without the comparison.
    #[test]
    fn a_shadow_step_no_request_can_run_fails_naming_the_operators() {
        let changed = vec!["gemm".to_owned(), "paged_attention".to_owned()];

        let message = shadow_stall_message(&changed, 300);

        assert!(
            message.contains("2 changed operators (gemm, paged_attention)"),
            "the operators are not named: {message}"
        );
        assert!(
            message.contains("no resident request ran a step, so nothing was compared"),
            "the cause is not named: {message}"
        );
        assert!(
            message.contains("abandoned rather than passed"),
            "the decision is not named: {message}"
        );
        assert!(
            shadow_stall_message(&["gemm".to_owned()], 5).contains("1 changed operator (gemm)"),
            "one operator does not read as one"
        );
    }

    /// The refusal a deployment reader acts on names both figures, where each came from, and the
    /// two changes that would satisfy it - and keeps the mechanism's own words in front of them.
    #[test]
    fn a_token_limit_refusal_names_the_catalogue_and_where_the_policys_figure_came_from() {
        let catalogue = PrefillCatalogue {
            bound: 512,
            stated: true,
        };
        let mechanism = "max_num_batched_tokens is 8192, above the build config's 512";

        let refusal = limit_refusal(mechanism, None, catalogue);

        assert!(
            refusal.starts_with(mechanism),
            "the mechanism's own refusal was not kept: {refusal}"
        );
        assert!(refusal.contains("up to 512 token(s)"), "{refusal}");
        assert!(
            refusal.contains("the deployment's cuda_graph.prefill_token_sizes"),
            "the catalogue's origin is not named: {refusal}"
        );
        assert!(
            refusal.contains("the deployment states no max_num_batched_tokens"),
            "the policy's figure is not attributed to the policy: {refusal}"
        );
        assert!(
            refusal.contains("state max_num_batched_tokens at or below 512"),
            "the first way out is missing: {refusal}"
        );
        assert!(
            refusal.contains("capture prefill buckets up to 8192"),
            "the second way out is missing: {refusal}"
        );
    }

    /// A deployment that states the token bound is named as the one that chose it, and a catalogue
    /// the launch file does not state says so rather than reading as the deployment's own figure.
    #[test]
    fn a_token_limit_the_deployment_stated_is_attributed_to_the_deployment() {
        let catalogue = PrefillCatalogue {
            bound: 512,
            stated: false,
        };

        let refusal = limit_refusal(
            "max_num_batched_tokens is 4096, above the build config's 512",
            Some("4096"),
            catalogue,
        );

        assert!(
            refusal.contains("the deployment states max_num_batched_tokens = 4096"),
            "{refusal}"
        );
        assert!(
            refusal.contains("the build's default catalogue"),
            "an unstated catalogue reads as stated: {refusal}"
        );
    }

    /// Every other policy failure is the mechanism's sentence and nothing else: a rendering that
    /// guessed at a message it does not know would put a wrong cause in a log.
    #[test]
    fn a_policy_failure_that_is_not_a_token_limit_is_answered_unchanged() {
        let catalogue = PrefillCatalogue {
            bound: 512,
            stated: true,
        };
        for mechanism in [
            "max_batch is 512, above the build config's 16",
            "the policy's trial refused every decision",
            "timeout is zero; a resident request would be starved from the first step",
        ] {
            assert_eq!(
                limit_refusal(mechanism, None, catalogue),
                mechanism,
                "a message that is not the token refusal was rewritten"
            );
        }
    }

    /// A named program chooses its operator's implementation and never states the operator set:
    /// the generation's programs are the store's packages, split by the phases their own built keys
    /// state, with the deployment's preference deciding where two of them could serve.
    ///
    /// This is the shape `attention_backend` needs: `paged_attention` is fa4's prefill module and
    /// trtllm's decode one, and a generation holding only one of them serves one phase and refuses
    /// the other ("no kernel for operator paged_attention descriptor phase_prefill_...").
    #[test]
    fn an_operator_two_packages_build_is_bound_once_per_phase() {
        let program = |backend: &str, revision: u32| Program {
            backend: backend.to_owned(),
            revision,
            arguments: Default::default(),
        };
        let prefill = vec!["phase_prefill_dtype_bf16_head_dim_128".to_owned()];
        let decode = vec!["phase_decode_dtype_bf16_kv_bf16_head_dim_128".to_owned()];

        // fa4 built the prefill cell and trtllm the decode one: both are the operator's, and each
        // keeps only the keys its own package built.
        let holders = vec![
            (program("fa4", 3), prefill.clone()),
            (program("trtllm", 7), decode.clone()),
        ];
        let slots = operator_slots(
            "paged_attention",
            &holders,
            holders.first().map(|(held, _)| held),
        );

        assert_eq!(slots.len(), 2, "one operator's two packages were collapsed");
        let phases: Vec<Option<PhaseKind>> = slots.iter().map(|(slot, _)| slot.phase).collect();
        assert_eq!(
            phases,
            vec![Some(PhaseKind::Prefill), Some(PhaseKind::Decode)]
        );
        assert_eq!(
            slots
                .iter()
                .map(|(_, held)| held.backend.as_str())
                .collect::<Vec<_>>(),
            vec!["fa4", "trtllm"],
            "the phase split bound a package to the other phase's cell"
        );

        // One package covering both phases is one slot, as it always was.
        let both = vec![(
            program("fa4", 3),
            prefill.iter().cloned().chain(decode.clone()).collect(),
        )];
        let slots = operator_slots("paged_attention", &both, both.first().map(|(held, _)| held));
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].0.phase, None);

        // An operator whose keys state no phase - a gemm - is never split on a guess.
        let holders = vec![
            (
                program("cublas", 1),
                vec!["dtype_f16_m_128_n_128_k_7168".to_owned()],
            ),
            (
                program("cutlass", 2),
                vec!["dtype_f16_m_128_n_128_k_7168".to_owned()],
            ),
        ];
        let slots = operator_slots("gemm", &holders, holders.first().map(|(held, _)| held));
        assert_eq!(slots.len(), 1, "a phase-less operator was split");
        assert_eq!(slots[0].0.phase, None);

        // Two packages claiming one phase is a states-nothing case, not a split: the deployment's
        // preference decides, and neither package is asked for the other's key.
        let holders = vec![
            (program("fa4", 3), prefill.clone()),
            (program("trtllm", 7), prefill),
        ];
        let slots = operator_slots(
            "paged_attention",
            &holders,
            holders.first().map(|(held, _)| held),
        );
        assert_eq!(slots.len(), 1, "an overlapping split was stated");
        assert_eq!(slots[0].1.backend, "fa4");
    }

    /// A promotion with nothing prepared is refused, and one with a generation is not: the engine
    /// cannot see the epoch's current generation (`Promote` states none), so what it can refuse is
    /// a process that holds none at all.
    #[test]
    fn a_promotion_with_nothing_prepared_is_refused_and_one_with_a_generation_is_not() {
        assert!(
            promote_refusal(true, 0).is_some(),
            "a standby with nothing prepared would have been promoted into an empty generation"
        );
        assert!(
            promote_refusal(true, 1).is_none(),
            "a standby holding its prepared generation was refused"
        );
        assert!(
            promote_refusal(false, 0).is_none(),
            "an initial process, which is never promoted, was refused"
        );
        let reason = promote_refusal(true, 0).unwrap_or_default();
        assert!(reason.contains("no prepared generation"), "{reason}");
        assert!(reason.contains("discarded"), "{reason}");
    }

    /// The shape the run at `b43aaac5e` met, and the one that stopped it: a deployment names **one**
    /// backend for attention (`attention_backend=trtllm`, a phase-less slot), this host's store
    /// holds that package's decode module *and* fa4's prefill one. The generation must hold both -
    /// the forward launches a prefill key and a decode key - and the deployment's arguments stay on
    /// the package it named.
    ///
    /// The run's log named the failure exactly: `slots=25 named=1 operators=25 alternative=1`, one
    /// program per operator, and then `generation 1 has no kernel for operator paged_attention
    /// descriptor phase_prefill_...`. The package was written down twice in one list - once as the
    /// deployment named it and once as the store lists it, differing only in the arguments - so the
    /// second copy's decode key collided with the first's and no split was stated.
    #[test]
    fn a_deployment_that_names_one_backend_still_holds_the_phases_the_store_builds() {
        let prefill = "phase_prefill_dtype_bf16_kv_bf16_out_bf16_head_dim_128_page_64_q_heads_32";
        let decode = "phase_decode_dtype_bf16_kv_bf16_out_bf16_head_dim_128_page_64_q_heads_32";
        let mut arguments = toml::Table::new();
        arguments.insert("window".to_owned(), toml::Value::Boolean(false));

        // What the `Precopy` hands the install: one phase-less slot for attention, naming trtllm.
        let mut named = Programs::new();
        named.insert(
            ProgramSlot {
                op: "paged_attention".to_owned(),
                phase: None,
                part: None,
            },
            Program {
                backend: "trtllm".to_owned(),
                revision: 1,
                arguments: arguments.clone(),
            },
        );

        // What this host's store holds for the operator: both packages, in its own order.
        let mut chosen: BTreeMap<String, Vec<(String, u32)>> = BTreeMap::new();
        chosen.insert(
            "paged_attention".to_owned(),
            vec![("fa4".to_owned(), 1), ("trtllm".to_owned(), 1)],
        );
        let mut built: BTreeMap<(String, String, u32), Vec<String>> = BTreeMap::new();
        built.insert(
            ("paged_attention".to_owned(), "fa4".to_owned(), 1),
            vec![prefill.to_owned()],
        );
        built.insert(
            ("paged_attention".to_owned(), "trtllm".to_owned(), 1),
            vec![decode.to_owned()],
        );

        let (programs, candidates) =
            resolve_slots(&named, &chosen, &built, &[]).expect("the store built what was named");

        assert_eq!(
            programs.len(),
            2,
            "the generation held one program of an operator whose keys span two phases, which is \
             the stop this test is written from"
        );
        assert_eq!(
            programs
                .get(&ProgramSlot {
                    op: "paged_attention".to_owned(),
                    phase: Some(PhaseKind::Prefill),
                    part: None,
                })
                .map(|program| program.backend.as_str()),
            Some("fa4"),
            "the prefill key's package was not bound for prefill"
        );
        let decode_slot = ProgramSlot {
            op: "paged_attention".to_owned(),
            phase: Some(PhaseKind::Decode),
            part: None,
        };
        let Some(decode_program) = programs.get(&decode_slot).cloned() else {
            panic!("the decode phase bound no program");
        };
        assert_eq!(decode_program.backend, "trtllm");
        assert_eq!(
            decode_program.arguments, arguments,
            "the deployment's arguments were replaced by the store's defaults"
        );
        assert_eq!(
            candidates.get("paged_attention").map(Vec::len),
            Some(2),
            "the walk lost the operator's alternatives"
        );
    }

    /// The other half of the same rule: when the named package covers every phase its operator's
    /// keys span, one slot is right and a split would be two packages asked for the same keys.
    #[test]
    fn a_named_package_that_covers_every_phase_stays_one_slot() {
        let prefill = "phase_prefill_dtype_bf16";
        let decode = "phase_decode_dtype_bf16";
        let mut named = Programs::new();
        named.insert(
            ProgramSlot {
                op: "paged_attention".to_owned(),
                phase: None,
                part: None,
            },
            Program {
                backend: "fa4".to_owned(),
                revision: 1,
                arguments: Default::default(),
            },
        );
        let mut chosen: BTreeMap<String, Vec<(String, u32)>> = BTreeMap::new();
        chosen.insert(
            "paged_attention".to_owned(),
            vec![("fa4".to_owned(), 1), ("trtllm".to_owned(), 1)],
        );
        let mut built: BTreeMap<(String, String, u32), Vec<String>> = BTreeMap::new();
        built.insert(
            ("paged_attention".to_owned(), "fa4".to_owned(), 1),
            vec![prefill.to_owned(), decode.to_owned()],
        );
        built.insert(
            ("paged_attention".to_owned(), "trtllm".to_owned(), 1),
            vec![decode.to_owned()],
        );

        let (programs, _) =
            resolve_slots(&named, &chosen, &built, &[]).expect("the store built what was named");

        assert_eq!(
            programs.len(),
            1,
            "one package covering both phases was split"
        );
        let slot = ProgramSlot {
            op: "paged_attention".to_owned(),
            phase: None,
            part: None,
        };
        assert_eq!(
            programs.get(&slot).map(|program| program.backend.as_str()),
            Some("fa4")
        );
    }

    /// The three shapes `split_claims` decides, and the reason each refusal gives: the run at
    /// `b43aaac5e` was the second one, and it failed a page later with `no kernel for operator
    /// paged_attention descriptor phase_prefill_...` because nothing said the split had been
    /// declined.
    #[test]
    fn a_split_is_stated_exactly_or_its_refusal_names_what_stopped_it() {
        let program = |backend: &str| Program {
            backend: backend.to_owned(),
            revision: 1,
            arguments: Default::default(),
        };
        let prefill = "phase_prefill_dtype_bf16".to_owned();
        let decode = "phase_decode_dtype_bf16".to_owned();

        // Two packages, one phase each: split, and nothing to state.
        let split = vec![
            (program("trtllm"), vec![decode.clone()]),
            (program("fa4"), vec![prefill.clone()]),
        ];
        let (claims, refusal) = split_claims(&split);
        assert_eq!(claims.len(), 2, "the phases the packages claim");
        assert_eq!(refusal, None, "a clean split must not refuse itself");

        // Two packages building one phase: the refusal names both.
        let clash = vec![
            (program("trtllm"), vec![decode.clone()]),
            (program("fa4"), vec![decode.clone()]),
        ];
        let (_, refusal) = split_claims(&clash);
        let reason = refusal.expect("two packages building one phase is a refusal");
        assert!(
            reason.contains("trtllm") && reason.contains("fa4"),
            "the refusal named neither package: {reason}"
        );
        assert!(
            reason.contains("decode"),
            "the refusal did not name the phase both build: {reason}"
        );

        // A key that states no phase fits every slot, so the operator is filled by one program.
        let unphased = vec![
            (program("trtllm"), vec![decode.clone()]),
            (program("fa4"), vec!["dtype_bf16_kv_bf16".to_owned()]),
        ];
        let (_, refusal) = split_claims(&unphased);
        let reason = refusal.expect("a key with no phase is a refusal");
        assert!(
            reason.contains("fa4") && reason.contains("no phase"),
            "the refusal did not name the package whose key states no phase: {reason}"
        );

        // One package is not a refusal: there is nothing to split.
        let single = vec![(program("trtllm"), vec![decode, prefill])];
        let (_, refusal) = split_claims(&single);
        assert_eq!(
            refusal, None,
            "one package covering every phase is the answer, not a refusal"
        );
    }

    /// A slot's keys are the ones its phase owns: a generation that holds two packages of one
    /// operator must never name the other phase's key to either of them, because the Program Server
    /// hands a request's program every key the request names for its operator and a package cannot
    /// build a phase it does not declare.
    #[test]
    fn a_slot_takes_the_keys_of_its_own_phase() {
        let prefill = "phase_prefill_dtype_bf16_head_dim_128";
        let decode = "phase_decode_dtype_bf16_kv_bf16_head_dim_128";
        let gemm = "dtype_f16_m_128_n_128_k_7168";

        assert!(key_fits_phase(prefill, Some(PhaseKind::Prefill)));
        assert!(!key_fits_phase(decode, Some(PhaseKind::Prefill)));
        assert!(key_fits_phase(decode, Some(PhaseKind::Decode)));
        assert!(
            !key_fits_phase(prefill, Some(PhaseKind::Decode)),
            "a decode slot would have been asked for the prefill key"
        );
        // A phase-less slot is the whole operator, and a phase-less key fits any slot.
        assert!(key_fits_phase(prefill, None));
        assert!(key_fits_phase(gemm, None));
        assert!(key_fits_phase(gemm, Some(PhaseKind::Decode)));
    }

    /// An operator is one operator however many packages fill it: a store that splits attention
    /// into a prefill module and a decode one holds two slots of it, and changing either changes
    /// what the operator launches, so the shadow stage is owed for it.
    #[test]
    fn an_operator_filled_by_two_packages_is_one_changed_operator() {
        let program = |backend: &str, revision: u32| Program {
            backend: backend.to_owned(),
            revision,
            arguments: Default::default(),
        };
        let slot = |phase: Option<PhaseKind>| ProgramSlot {
            op: "paged_attention".to_owned(),
            phase,
            part: None,
        };
        // What serves: one package filling both phases.
        let mut serving = Programs::new();
        serving.insert(slot(None), program("fa4", 3));
        // What was prepared: the store's split - the prefill cell unchanged, the decode cell taken
        // over by another package.
        let mut prepared = Programs::new();
        prepared.insert(slot(Some(PhaseKind::Prefill)), program("fa4", 3));
        prepared.insert(slot(Some(PhaseKind::Decode)), program("trtllm", 7));

        assert_eq!(
            changed_operators(&prepared, &serving),
            vec!["paged_attention"],
            "a split of one operator was not seen as a change of that operator"
        );
        // The same two slots on both sides is no change: a split that serves on both sides is the
        // same operator running the same programs.
        assert!(changed_operators(&prepared, &prepared).is_empty());
    }

    /// The Program Server hands a request's program every key the request names for its operator,
    /// so a generation's slots are grouped by the package that fills them: two slots of one
    /// operator from one package are one request entry with both key sets, and two packages are
    /// two requests that never name each other's keys.
    #[test]
    fn a_package_asks_once_per_operator_for_the_keys_it_built() {
        let program = |backend: &str, revision: u32| Program {
            backend: backend.to_owned(),
            revision,
            arguments: Default::default(),
        };
        let slot = |op: &str, phase: Option<PhaseKind>| ProgramSlot {
            op: op.to_owned(),
            phase,
            part: None,
        };

        let mut group: (Vec<ProgramRequest>, Vec<Descriptors>) = (Vec::new(), Vec::new());
        let prefill = program("fa4", 3);
        merge_request(
            &mut group,
            &slot("paged_attention", Some(PhaseKind::Prefill)),
            &prefill,
            vec!["phase_prefill_dtype_bf16".to_owned()],
        );
        merge_request(
            &mut group,
            &slot("paged_attention", Some(PhaseKind::Decode)),
            &prefill,
            vec!["phase_decode_dtype_bf16".to_owned()],
        );
        merge_request(
            &mut group,
            &slot("gemm", None),
            &prefill,
            vec!["dtype_f16_m_128_n_128_k_7168".to_owned()],
        );

        assert_eq!(group.0.len(), 2, "one package asked twice for one operator");
        assert!(
            group
                .0
                .iter()
                .all(|request| request.selector == Selector::Fixed(prefill.clone()))
        );
        let attention = group
            .1
            .iter()
            .find(|descriptor| descriptor.op == "paged_attention")
            .map(|descriptor| descriptor.keys.clone())
            .unwrap_or_default();
        assert_eq!(
            attention,
            vec![
                "phase_decode_dtype_bf16".to_owned(),
                "phase_prefill_dtype_bf16".to_owned()
            ],
            "a phase's key was dropped from the package's one request"
        );
    }

    use program_server::ModuleForm;
    use program_server::protocol::{ListedModule, ListedPackage};

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

    /// The library of a multi-file native module is chosen by what it defines, never by position.
    ///
    /// The shape that failed a run: a `paged_attention` module holds `module.so` beside `cubins/`, and
    /// `cubins/` sorts first, so the first file handed to `dlopen` was a cubin. Here two real objects
    /// stand in for those files with the library named so it sorts **second**, and the entry declared
    /// is one only it defines.
    ///
    /// The directory is the module, and the mechanism is the package's own: `paged_attention`'s entry
    /// takes no path and finds `cubins/` from `dladdr` of its own symbol, whose `dli_fname` is the
    /// path this process handed `dlopen`. So the path must be the module's own file - which is what
    /// the assertions below hold: the library is that file, it is not a copy, and the sibling sits in
    /// the directory the object will look in. A `/proc/self/fd/<n>` path would give it
    /// `/proc/self/fd`, and a copy would give it the copy's directory.
    #[test]
    fn a_native_library_is_the_file_that_defines_the_entry_not_the_first_one() {
        let Some(libc) = loaded_library("libc.so.6") else {
            return;
        };
        let Some(libm) = loaded_library("libm.so.6") else {
            return;
        };
        let directory =
            std::env::temp_dir().join(format!("uniserve-native-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).expect("a temporary directory for this test");
        let sibling = directory.join("aa-sibling.so");
        let library = directory.join("zz-module.so");
        std::fs::copy(&libm, &sibling).expect("the sibling object is copied");
        std::fs::copy(&libc, &library).expect("the library object is copied");

        let module = native_library(
            vec![
                opened_file("aa-sibling.so", &sibling),
                opened_file("zz-module.so", &library),
            ],
            "malloc",
        )
        .expect("the file that defines the entry is the module's library");
        assert_eq!(
            module.path, library,
            "the library was resolved by position in the module, not by the entry it defines"
        );
        assert!(
            !module.copied(),
            "the library has a name of its own, so no copy is made"
        );
        assert!(
            module.path.with_file_name("aa-sibling.so").is_file(),
            "the module's sibling is not reachable from the location the library is loaded from"
        );

        // The same module with its library unlinked after the handle was taken: a copy would carry
        // no sibling, so the module is refused by name rather than loaded from one.
        let nameless = directory.join("nameless.so");
        std::fs::copy(&libc, &nameless)
            .expect("the library object is copied under a temporary name");
        let handle = std::fs::File::open(&nameless).expect("the copy opens");
        std::fs::remove_file(&nameless).expect("the copy is unlinked, as a store's handle may be");

        let refused = match native_library(
            vec![
                opened_file("aa-sibling.so", &sibling),
                program_server::ModuleFile {
                    name: "nameless.so".to_owned(),
                    size: std::fs::metadata(&sibling)
                        .map(|meta| meta.len())
                        .unwrap_or_default(),
                    file: handle,
                },
            ],
            "malloc",
        ) {
            Ok(_) => panic!("a module with siblings was loaded from a copy"),
            Err(error) => format!("{error}"),
        };
        assert!(
            refused.contains("nameless.so") && refused.contains("copy"),
            "the refusal names neither the file nor the copy: {refused}"
        );

        let _ = std::fs::remove_dir_all(&directory);
    }

    /// The object a pattern names, as the process's own maps record it: a real ELF on this host.
    fn loaded_library(pattern: &str) -> Option<std::path::PathBuf> {
        let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
        maps.lines()
            .filter_map(|line| line.split_whitespace().last())
            .find(|path| path.contains(pattern) && path.starts_with('/'))
            .map(std::path::PathBuf::from)
    }

    /// One file of a module under test, opened the way the store's handles are.
    fn opened_file(name: &str, path: &std::path::Path) -> program_server::ModuleFile {
        program_server::ModuleFile {
            name: name.to_owned(),
            size: std::fs::metadata(path)
                .map(|meta| meta.len())
                .unwrap_or_default(),
            file: std::fs::File::open(path).expect("the test's object opens"),
        }
    }

    /// The entry reader answers about a real object, and says nothing it cannot check.
    ///
    /// Every Linux process has its C library loaded, and this one is asked about the object the maps
    /// name: the count and the first names are the reader's own answer, not a restatement of it. A
    /// host whose maps name no such path is the one case where the test has nothing to ask.
    #[test]
    fn the_entry_reader_lists_the_functions_an_object_defines() {
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
        let libc = maps
            .lines()
            .filter_map(|line| line.split_whitespace().last())
            .find(|path| path.contains("libc.so.6") && path.starts_with('/'))
            .map(str::to_owned);
        let Some(libc) = libc else {
            return;
        };
        let (total, names) = defined_entries(std::path::Path::new(&libc))
            .unwrap_or_else(|| panic!("{libc} is a loaded object and its entries were not read"));
        assert!(
            total > 100,
            "{libc} defines {total} functions, too few for a C library"
        );
        assert_eq!(names.len(), 8, "the first names of {libc}: {names:?}");
        assert!(
            names.iter().all(
                |name| name.starts_with(|held: char| held.is_ascii_alphabetic() || held == '_')
            ),
            "a name read out of an object is not a name: {names:?}"
        );
    }

    /// One store, read twice: with the deployment naming one backend and with it naming nothing.
    /// The two must produce the same slot set, because the name chooses the implementation of the
    /// keys its own package covers and the store keeps the other phases. A relaunch hands the engine
    /// the epoch's kernels - the named form of a start that named nothing - so the two forms meeting
    /// one store is the equality a relaunch depends on.
    #[test]
    fn a_named_start_and_an_unnamed_one_agree_on_the_slots_of_one_store() {
        let prefill = "phase_prefill_dtype_bf16_kv_bf16_out_bf16_head_dim_128_page_64_q_heads_32";
        let decode = "phase_decode_dtype_bf16_kv_bf16_out_bf16_head_dim_128_page_64_q_heads_32";
        let mut chosen: BTreeMap<String, Vec<(String, u32)>> = BTreeMap::new();
        chosen.insert(
            "paged_attention".to_owned(),
            vec![("fa4".to_owned(), 1), ("trtllm".to_owned(), 1)],
        );
        let mut built: BTreeMap<(String, String, u32), Vec<String>> = BTreeMap::new();
        built.insert(
            ("paged_attention".to_owned(), "fa4".to_owned(), 1),
            vec![prefill.to_owned()],
        );
        built.insert(
            ("paged_attention".to_owned(), "trtllm".to_owned(), 1),
            vec![decode.to_owned()],
        );
        let mut named = Programs::new();
        named.insert(
            ProgramSlot {
                op: "paged_attention".to_owned(),
                phase: None,
                part: None,
            },
            Program {
                backend: "trtllm".to_owned(),
                revision: 1,
                arguments: Default::default(),
            },
        );

        let (unnamed, _) = resolve_slots(&Programs::new(), &chosen, &built, &[])
            .expect("the store decides when nothing is named");
        let (with_name, _) =
            resolve_slots(&named, &chosen, &built, &[]).expect("the named package is built here");

        let slots = |programs: &Programs| {
            let mut held: Vec<(String, Option<PhaseKind>, String)> = programs
                .iter()
                .map(|(slot, program)| (slot.op.clone(), slot.phase, program.backend.clone()))
                .collect();
            held.sort();
            held
        };
        assert_eq!(
            slots(&with_name),
            slots(&unnamed),
            "naming one backend changed the slot set of one store"
        );
        assert_eq!(
            slots(&with_name),
            vec![
                (
                    "paged_attention".to_owned(),
                    Some(PhaseKind::Decode),
                    "trtllm".to_owned()
                ),
                (
                    "paged_attention".to_owned(),
                    Some(PhaseKind::Prefill),
                    "fa4".to_owned()
                ),
            ],
            "the deployment's own backend did not take the phase its keys cover"
        );
    }

    /// A program this host has built nothing of is refused by name rather than substituted. The
    /// keys the name owned are not known here, so a substitute may cover a different phase and the
    /// walk would find out one capture at a time - which is how a relaunch stood past the watchdog
    /// at `this node's devices` while the log said the store had filled the operator.
    #[test]
    fn a_named_program_this_host_did_not_build_is_refused_by_name() {
        let prefill = "phase_prefill_dtype_bf16_kv_bf16_out_bf16_head_dim_128_page_64_q_heads_32";
        let mut chosen: BTreeMap<String, Vec<(String, u32)>> = BTreeMap::new();
        chosen.insert("paged_attention".to_owned(), vec![("fa4".to_owned(), 1)]);
        let mut built: BTreeMap<(String, String, u32), Vec<String>> = BTreeMap::new();
        built.insert(
            ("paged_attention".to_owned(), "fa4".to_owned(), 1),
            vec![prefill.to_owned()],
        );
        let mut named = Programs::new();
        named.insert(
            ProgramSlot {
                op: "paged_attention".to_owned(),
                phase: None,
                part: None,
            },
            Program {
                backend: "trtllm".to_owned(),
                revision: 1,
                arguments: Default::default(),
            },
        );

        // The store holds the named package and has never built it: that is a store build away
        // from serving, and the refusal says so rather than naming a substitution.
        let packages = vec![program_server::protocol::ListedPackage {
            op: "paged_attention".to_owned(),
            backend: "trtllm".to_owned(),
            revision: 1,
            modules: Vec::new(),
        }];
        let error = resolve_slots(&named, &chosen, &built, &packages)
            .expect_err("nothing of the named package was built here");
        assert!(
            error.contains("paged_attention/trtllm revision 1"),
            "the refusal does not name the program the deployment named: {error}"
        );
        assert!(
            error.contains("fa4 revision 1"),
            "the refusal does not name what the store holds instead: {error}"
        );
        assert!(
            error.contains("declares trtllm revision 1 without a module"),
            "the refusal does not tell a declared-but-unbuilt package from a missing one: {error}"
        );
    }

    /// An operator no model forward reaches - the runner launches `slot_commit` itself - is a
    /// package the store holds and has never built, so `select_programs` cannot see it. The walk
    /// names the operator, and the declaration is what lets this host build the key it needs: the
    /// highest revision of the operator's packages, ties broken by the backend so two runs choose
    /// alike.
    #[test]
    fn a_declared_package_is_chosen_for_an_operator_nothing_built() {
        let package = |backend: &str, revision: u32| program_server::protocol::ListedPackage {
            op: "slot_commit".to_owned(),
            backend: backend.to_owned(),
            revision,
            modules: Vec::new(),
        };
        let packages = vec![
            package("triton", 1),
            package("triton", 3),
            package("cuda", 3),
        ];
        assert_eq!(
            declared_program(&packages, "slot_commit"),
            Some(("slot_commit".to_owned(), "cuda".to_owned(), 3)),
            "the highest revision did not win, or the tie was not broken by the backend's name"
        );
        assert_eq!(
            declared_program(&packages, "paged_attention"),
            None,
            "an operator the store holds no package of was answered"
        );
    }

    /// The state a daemon's late messages meet: an update that went to probation and was reverted by
    /// the rollback threshold, so every command still in flight names a generation no longer there.
    fn update_on_probation_then_reverted() -> executor::hot_update::HotUpdate {
        let mut update = executor::hot_update::HotUpdate::new(1);
        update.begin_precopy(2).expect("a precopy may begin");
        update.precopied(2).expect("the precopy lands");
        update
            .switchover(2, Instant::now())
            .expect("the switchover takes effect");
        update
            .revert(
                2,
                UpdateFailure::Threshold {
                    reason: "the rollback threshold".into(),
                },
            )
            .expect("the revert lands");
        update
    }

    /// An epoch switch's commit reaches a target whose programs were committed at their handover:
    /// the generation it names serves with nothing on trial, so it is answered as committed. A
    /// refusal here left the switch waiting for an acknowledgement, and the previous epoch's process
    /// running, for good.
    #[test]
    fn a_commit_of_the_generation_that_serves_is_committed() {
        let mut update = update_on_probation_then_reverted();
        match answer_commit(&mut update, 1) {
            ProgramStatus::Committed { generation } => assert_eq!(generation, 1),
            other => panic!("a commit of what serves answered {other:?}"),
        }
    }

    /// A commit that races the revert is answered with the state that holds, and the process keeps
    /// serving the generation before the update. Ending the main process over it is the failure this
    /// test exists for: a message whose whole job is to be answered must not kill its answerer.
    #[test]
    fn a_commit_after_a_revert_answers_with_the_generation_that_serves() {
        let mut update = update_on_probation_then_reverted();
        match answer_commit(&mut update, 2) {
            ProgramStatus::Reverted { generation, reason } => {
                assert_eq!(generation, 2);
                assert!(
                    reason.contains("not in flight"),
                    "the answer does not say why the commit did not apply: {reason}"
                );
                assert!(
                    reason.contains("generation 1 serves"),
                    "the answer does not name what serves instead: {reason}"
                );
            }
            other => panic!("a raced commit answered {other:?} rather than the state that holds"),
        }
        assert_eq!(
            update.current(),
            1,
            "the generation before the update does not serve after a raced commit"
        );
    }

    /// The rest of the family after the same revert: a purge, a second revert and a second precopy
    /// each name a generation no longer in flight, and the machine answers each without leaving the
    /// generation that serves.
    #[test]
    fn the_program_commands_after_a_revert_all_name_what_they_cannot_apply() {
        let mut update = update_on_probation_then_reverted();
        assert!(
            update.request_purge(2).is_err(),
            "the reverted generation is still awaiting a purge, so this test proves nothing"
        );
        assert!(update.commit(2).is_err());
        assert!(
            update
                .revert(
                    2,
                    UpdateFailure::Threshold {
                        reason: "again".into()
                    }
                )
                .is_err(),
            "a revert of a generation that is not on probation applied"
        );
        assert!(
            update.precopied(2).is_err(),
            "a precopy with none in flight applied"
        );
        assert_eq!(
            update.current(),
            1,
            "the serving generation moved under commands it could not apply"
        );
    }

    /// A switchover answers `Switched` only when it can serve: a warmup still running means the
    /// buckets a batch would be planned into do not exist yet, so the answer waits for admission.
    /// The run this is written from answered first and the daemon opened a frontend onto a process
    /// whose warmup then failed.
    #[test]
    fn a_switchover_answers_after_its_warmup_not_before() {
        match switchover_answer(0, 3) {
            SwitchoverAnswer::Now(ProgramStatus::Switched { generation }) => {
                assert_eq!(generation, 3)
            }
            SwitchoverAnswer::Now(other) => {
                panic!("a switchover with nothing to warm answered {other:?}")
            }
            SwitchoverAnswer::AfterTheWarmup => {
                panic!("a switchover with nothing to warm deferred its answer forever")
            }
        }
        assert!(
            matches!(switchover_answer(2, 3), SwitchoverAnswer::AfterTheWarmup),
            "a switchover answered before the warmup it queued had run"
        );
    }
}
